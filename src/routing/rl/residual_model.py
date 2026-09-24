# ============================================================================
# residual_model.py — 残差回归模型 V1（flat obs → MLP → 逐动作 Δ̂/sign 头）
#
# R0 仲裁蒸馏标签：Δ(s,a) = Q(a) − Q(ASAP)（1-ply 前瞻残差，CRN 配对）。
# V1 只验信号（G1：排序能力），V2(SubGNN) 仅当 V1 通过且余量不足时启动。
# 输入为 ClockedRoutingEnv(scheduling_only) 的 flat obs（use_gnn=False），
# 输出 n_actions 维 Δ̂ 与 sign logits；训练/评估只监督合法候选条目。
# ============================================================================
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualRegressor(nn.Module):
    """flat obs → MLP → (delta[n_actions], sign_logits[n_actions])。"""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 1024):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(hidden // 2, n_actions)
        self.sign_head = nn.Linear(hidden // 2, n_actions)

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        return self.delta_head(h), self.sign_head(h)

    @torch.no_grad()
    def score_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """返回指定 (obs, action) 对的 Δ̂（部署评分用）。"""
        d, _ = self.forward(obs)
        return d.gather(1, actions.unsqueeze(1)).squeeze(1)
