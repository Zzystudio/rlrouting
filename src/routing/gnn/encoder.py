from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class GATEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        out_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        self.num_layers = num_layers
        self.dropout = dropout
        self.out_dim = out_dim

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_dim = node_dim
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(in_dim, hidden_dim // heads, heads=heads,
                        edge_dim=edge_dim, concat=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim
        self.convs.append(
            GATConv(in_dim, out_dim, heads=1, edge_dim=edge_dim, concat=False)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr)
        return x

    def graph_embedding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = self.forward(x, edge_index, edge_attr)
        mean = global_mean_pool(h, batch)
        maxp = global_max_pool(h, batch)
        return torch.cat([mean, maxp], dim=-1)


class SubGNN(nn.Module):
    def __init__(
        self,
        node_dim: int = 32,
        edge_dim: int = 16,
        subgraph: str = "full",
        hidden_dim: int = 48,
        num_layers: int = 2,
        heads: int = 3,
        out_dim: int = 48,
    ):
        super().__init__()
        self.subgraph = subgraph
        self.encoder = GATEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            out_dim=out_dim,
        )
        self.out_dim = out_dim * 2  # mean + max pool

    def forward(
        self,
        data,  # RoutingGraphData or separate tensors
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from ..graph.circuit_dag import RoutingGraphData as RGD
        device = next(self.encoder.parameters()).device

        if isinstance(data, RGD):
            d = data
            sub = self.subgraph
            if sub == "logic":
                x = _to_tensor(d.gate_feat)
                ei = _to_long(d.dep_edge_index)
                ea = _to_tensor(d.dep_edge_attr)
            elif sub == "physics":
                x = _to_tensor(d.qubit_feat)
                ei = _to_long(d.coupling_edge_index)
                ea = _to_tensor(d.coupling_edge_attr)
            elif sub == "mapping":
                x = _to_tensor(np.concatenate([d.gate_feat, d.qubit_feat], axis=0))
                ei = _to_long(d.map_edge_index)
                ea = _to_tensor(d.map_edge_attr)
            else:  # "full"
                pyg = d.to_pyg("full")
                x = pyg.x
                ei = pyg.edge_index
                ea = pyg.edge_attr
                if batch is not None:
                    return self.encoder.graph_embedding(
                        x.to(device), ei.to(device), ea.to(device),
                        batch.to(device),
                    )
                B = torch.zeros(x.size(0), dtype=torch.long, device=device)
                return self.encoder.graph_embedding(
                    x.to(device), ei.to(device), ea.to(device), B,
                )
        elif hasattr(data, "x"):
            x = data.x
            ei = data.edge_index
            ea = data.edge_attr
            if batch is None and hasattr(data, "batch"):
                batch = data.batch
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            return self.encoder.graph_embedding(
                x.to(device), ei.to(device), ea.to(device),
                batch.to(device),
            )
        else:
            x = data
            ei = edge_index
            ea = edge_attr

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.encoder.graph_embedding(
            x.to(device), ei.to(device), ea.to(device), batch.to(device),
        )

    def node_embeddings(self, data) -> torch.Tensor:
        from ..graph.circuit_dag import RoutingGraphData as RGD
        assert isinstance(data, RGD), "node_embeddings expects RoutingGraphData"
        pyg = data.to_pyg("full")
        device = next(self.encoder.parameters()).device
        h = self.encoder.forward(
            pyg.x.to(device), pyg.edge_index.to(device), pyg.edge_attr.to(device)
        )
        qubit_h = h[-data.num_physical:]  # last P nodes = qubit nodes
        return qubit_h

    def node_embeddings_batch(self, data_list) -> list:
        """批量 GNN 前向：将所有图拼接为一张大图，一次 forward。

        Returns: list of (P_i, out_dim) qubit embeddings，顺序与 data_list 一致。
        """
        from ..graph.circuit_dag import RoutingGraphData as RGD
        if not data_list:
            return []
        device = next(self.encoder.parameters()).device

        xs, batch, slices = [], [], []
        ei_parts, ea_parts = [], []
        node_offset = 0
        for i, d in enumerate(data_list):
            assert isinstance(d, RGD), "node_embeddings_batch expects RoutingGraphData list"
            G, P = d.num_gates, d.num_physical
            xs.append(d.gate_feat)
            xs.append(d.qubit_feat)
            batch.extend([i] * (G + P))
            slices.append((node_offset + G, node_offset + G + P))

            if d.dep_edge_index.shape[1] > 0:
                ei = d.dep_edge_index.copy()
                ei += node_offset
                ei_parts.append(ei)
                ea_parts.append(d.dep_edge_attr)
            if d.coupling_edge_index.shape[1] > 0:
                ei = d.coupling_edge_index.copy()
                ei += node_offset + G
                ei_parts.append(ei)
                ea_parts.append(d.coupling_edge_attr)
            if d.map_edge_index.shape[1] > 0:
                ei = d.map_edge_index.copy()
                ei[0] += node_offset
                ei[1] += node_offset + G
                ei_parts.append(ei)
                ea_parts.append(d.map_edge_attr)
            node_offset += G + P

        x = torch.tensor(np.concatenate(xs, axis=0), dtype=torch.float, device=device)
        if ei_parts:
            edge_index = torch.tensor(np.concatenate(ei_parts, axis=1), dtype=torch.long, device=device)
            edge_attr = torch.tensor(np.concatenate(ea_parts, axis=0), dtype=torch.float, device=device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.empty((0, d.dep_edge_attr.shape[1]), dtype=torch.float, device=device)

        h = self.encoder.forward(x, edge_index, edge_attr)
        return [h[s:e] for s, e in slices]


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float)


def _to_long(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.long)
