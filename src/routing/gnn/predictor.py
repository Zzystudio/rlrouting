# ============================================================================
# predictor.py
# Multi-GNN 保真度（可靠性）预测器。
#
# 参考论文思路：用多个 GNN 分别对不同的噪声来源（热弛豫、门错误、串扰、
# 读出错误）进行建模，最后融合为电路整体输出保真度的预测。
#
# 相比单 GNN，多 GNN 结构： (1) 子网络可并行训练、推理更快；
# (2) 各噪声来源被解耦，便于分析与课程式训练。
# ============================================================================

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import SubGNN
from ..graph.features import NODE_FEATURE_DIM, EDGE_FEATURE_DIM


class MultiGNNTidelityPredictor(nn.Module):
    """由 4 个子 GNN 组成的保真度预测器。"""

    SUB_NETS = {
        "thermal": "acts_on",    # 热弛豫：经 acts_on 边吸收单比特 T1/T2
        "gate": "acts_on",       # 门错误：经 acts_on 边吸收门错误率
        "crosstalk": "couples",  # 串扰：经 couples 边吸收 ZZ 串扰
        "readout": "acts_on",    # 读出：经 acts_on 边吸收测量错误
    }

    def __init__(
        self,
        node_dim: int = NODE_FEATURE_DIM,
        edge_dim: int = EDGE_FEATURE_DIM,
        hidden_dim: int = 48,
        num_layers: int = 2,
        heads: int = 3,
        sub_out_dim: int = 48,
        fusion_hidden: int = 128,
    ):
        super().__init__()
        self.subs = nn.ModuleDict({
            name: SubGNN(
                node_dim=node_dim,
                edge_dim=edge_dim,
                edge_type=etype,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                heads=heads,
                out_dim=sub_out_dim,
            )
            for name, etype in self.SUB_NETS.items()
        })
        fused_dim = sum(s.out_dim for s in self.subs.values())
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Linear(fusion_hidden // 2, 1 + len(self.SUB_NETS)),  # 总保真度 + 4 个分项
        )

    def forward(self, data) -> torch.Tensor:
        """输入 PyG Batch/Data，输出预测张量。

        返回的最后一维： [0] 总保真度(logit, 经 sigmoid 得到 [0,1])，
        [1:] 各子网络分项保真度(logit)。
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        embeds = []
        for net in self.subs.values():
            embeds.append(net(x, edge_index, edge_attr, batch))
        fused = torch.cat(embeds, dim=-1)
        return self.fusion(fused)

    def graph_embedding(self, data) -> torch.Tensor:
        """返回融合前的图级嵌入（供 RL 作为状态特征）。"""
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        embeds = []
        for net in self.subs.values():
            embeds.append(net(x, edge_index, edge_attr, batch))
        return torch.cat(embeds, dim=-1)

    @torch.no_grad()
    def predict_fidelity(self, data) -> torch.Tensor:
        """仅返回 [0,1] 的总保真度预测。"""
        out = self.forward(data)
        return torch.sigmoid(out[:, 0])

    def predict_components(self, data):
        """返回总保真度与各分项保真度（均已 sigmoid 到 [0,1]）。"""
        out = self.forward(data)
        return torch.sigmoid(out)


def fidelity_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """保真度回归损失：MSE（作用于总保真度 logit 对应 sigmoid 前的数值）。

    这里对 (sigmoid(pred)-target)^2 做 MSE，并对分项保真度施加一致性约束。
    """
    total = torch.sigmoid(pred_logits[:, 0])
    loss = nn.functional.mse_loss(total, target)
    # 分项保真度的乘积应接近总保真度（软约束）
    comps = torch.sigmoid(pred_logits[:, 1:])
    comp_prod = torch.clamp(comps.prod(dim=1), 1e-6, 1.0)
    loss += 0.1 * nn.functional.mse_loss(comp_prod, target)
    return loss
