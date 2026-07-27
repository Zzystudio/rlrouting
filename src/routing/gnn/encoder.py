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
                    return self.encoder.graph_embedding(x, ei, ea, batch)
                B = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                return self.encoder.graph_embedding(x, ei, ea, B)
        elif hasattr(data, "x"):
            x = data.x
            ei = data.edge_index
            ea = data.edge_attr
            if batch is None and hasattr(data, "batch"):
                batch = data.batch
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            return self.encoder.graph_embedding(x, ei, ea, batch)
        else:
            x = data
            ei = edge_index
            ea = edge_attr

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.encoder.graph_embedding(x, ei, ea, batch)


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float)


def _to_long(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.long)
