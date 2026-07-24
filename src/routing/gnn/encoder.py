# ============================================================================
# encoder.py
# 图神经网络编码器：使用带边特征的图注意力网络 (GAT) 对路由图做消息传递，
# 输出节点嵌入与图级嵌入。
# ============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class GATEncoder(nn.Module):
    """多层 GAT 编码器（支持边特征参与注意力计算）。"""

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
        assert hidden_dim % heads == 0, "hidden_dim 必须能被 heads 整除"
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
        # 最后一层：单头，输出 out_dim
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
        return torch.cat([mean, maxp], dim=-1)  # (B, 2*out_dim)


class SubGNN(nn.Module):
    """Multi-GNN 的子网络：仅关注某一类边（热弛豫 / 门错误 / 串扰 / 读出）。"""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        edge_type: str,
        hidden_dim: int = 48,
        num_layers: int = 2,
        heads: int = 3,
        out_dim: int = 48,
    ):
        super().__init__()
        self.edge_type = edge_type
        self.encoder = GATEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            out_dim=out_dim,
        )
        self.out_dim = out_dim * 2  # mean + max

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        from ..graph.features import mask_edge_by_type
        import numpy as np
        # 仅保留本子网络关心的边类型
        ea_np = edge_attr.detach().cpu().numpy()
        masked_np = mask_edge_by_type(ea_np, self.edge_type)
        masked = torch.tensor(masked_np, dtype=edge_attr.dtype, device=edge_attr.device)
        return self.encoder.graph_embedding(x, edge_index, masked, batch)
