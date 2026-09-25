"""v0 GNN 价值网络（Sprint 2）—— 状态图 → 图级嵌入 → 标量 V_θ(s) ≈ M1-episode 值。

设计要点（对齐 20260922 方案 Phase 3 + 距离注入修正）：
  - 节点：n_phys 个 qubit 节点 + 剩余 2Q 门节点（数量可变）
  - **距离显式注入**：门节点特征含 d(q0,q1)/diam —— 3 层 GNN 感受野 <
    line_16q 直径 15，局部消息传递物理上看不见远距逻辑对，必须显式给距离
  - 边：couples（硬件耦合）/ maps_to（门→两端点物理比特）
  - 全局：进度 / 剩余门数占比 / 平均距离（池化后拼接）
  - 编码器：2 层 GINConv（hidden 64）→ mean+max 池化 → MLP → 标量
  - 训练目标：Huber(V_θ(s), label(s))，label = M1-episode 值（8-10q 实证 = V*）

推理成本：单图前向 ~1ms CPU（小图 GPU launch 开销反而亏）；状态图按 state_key
缓存（MCTS 叶子评估用）。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool

from .pure_env import PureRoutingEnv

N_QUBIT_FEAT = 5
N_GATE_FEAT = 5
N_GLOBAL_FEAT = 3


def build_state_graph(env: PureRoutingEnv) -> Data:
    """状态 → PyG Data。节点 = n_phys qubit + n_rem 门；特征见类 docstring。"""
    n_phys = env.n_phys
    diam = max(1, int(env.dist.max()))
    G = max(1, len(env._twoq))
    E = max(1, env.num_edges)
    nq = env.num_logical
    mapping = env.mapping

    rem = env.remaining_2q()
    ready_set = set(env.ready_2q())

    # qubit 特征
    pending_count = np.zeros(n_phys, dtype=np.float32)
    pending_dist_sum = np.zeros(n_phys, dtype=np.float32)
    pending_n = np.zeros(n_phys, dtype=np.float32)
    for idx in rem:
        q0, q1 = env.dag.gates[idx].qubits
        p0, p1 = mapping[q0], mapping[q1]
        d = float(env.dist[p0, p1])
        pending_count[p0] += 1.0
        pending_count[p1] += 1.0
        pending_dist_sum[p0] += d
        pending_dist_sum[p1] += d
        pending_n[p0] += 1.0
        pending_n[p1] += 1.0
    degree = np.zeros(n_phys, dtype=np.float32)
    for (p, q) in env.coupling_map:
        degree[p] += 1.0
        degree[q] += 1.0

    qx = np.zeros((n_phys, N_QUBIT_FEAT), dtype=np.float32)
    for p in range(n_phys):
        qx[p, 0] = (mapping[p] if p < len(mapping) else -1) / max(1, nq)
        qx[p, 1] = degree[p] / max(1, E)
        qx[p, 2] = min(1.0, env.swaps_since_exec[p] / 5.0)
        qx[p, 3] = pending_count[p] / max(1, G)
        qx[p, 4] = pending_dist_sum[p] / max(1, pending_n[p]) / max(1, diam) \
            if pending_n[p] > 0 else 0.0

    # 门特征
    gx = np.zeros((len(rem), N_GATE_FEAT), dtype=np.float32)
    for j, idx in enumerate(rem):
        q0, q1 = env.dag.gates[idx].qubits
        p0, p1 = mapping[q0], mapping[q1]
        gx[j, 0] = float(env.dist[p0, p1]) / max(1, diam)   # 距离注入（关键）
        gx[j, 1] = q0 / max(1, nq)
        gx[j, 2] = q1 / max(1, nq)
        gx[j, 3] = 1.0 if idx in ready_set else 0.0
        gx[j, 4] = min(1.0, len(env.dag.gates[idx].predecessors) / 4.0)

    x = torch.as_tensor(np.concatenate([qx, gx], axis=0), dtype=torch.float32)

    # 边：couples（qubit-qubit）+ maps_to（门→两 qubit）
    src, tgt, ea = [], [], []
    for (p, q) in env.coupling_map:
        src += [p, q]
        tgt += [q, p]
        ea += [1.0, 1.0]
    for j, idx in enumerate(rem):
        q0, q1 = env.dag.gates[idx].qubits
        p0, p1 = mapping[q0], mapping[q1]
        gn = n_phys + j
        src += [gn, gn]
        tgt += [p0, p1]
        ea += [2.0, 2.0]
    edge_index = torch.as_tensor([src, tgt], dtype=torch.long)
    edge_attr = torch.as_tensor(ea, dtype=torch.float32).unsqueeze(-1)

    # 全局
    rem_dists = [float(env.dist[mapping[env.dag.gates[i].qubits[0]],
                                mapping[env.dag.gates[i].qubits[1]]])
                 for i in rem]
    glob = np.array([
        float(env.executed_mask.bit_count() / max(1, env.dag.num_gates)),
        len(rem) / max(1, G),
        float(np.mean(rem_dists) / max(1, diam)) if rem_dists else 0.0,
    ], dtype=np.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                glob=torch.as_tensor(glob, dtype=torch.float32),
                num_nodes=x.shape[0])


class GNNValueNet(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.conv1 = GINEConv(nn.Sequential(
            nn.Linear(N_QUBIT_FEAT, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden)), edge_dim=1)
        self.conv2 = GINEConv(nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden)), edge_dim=1)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + N_GLOBAL_FEAT, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, data: Data) -> torch.Tensor:
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        g = data.glob.reshape(-1, N_GLOBAL_FEAT)
        x1 = self.conv1(x, ei, ea)
        x2 = self.conv2(x1, ei, ea)
        h = torch.cat([global_mean_pool(x2, data.batch),
                       global_max_pool(x2, data.batch),
                       g], dim=-1)
        return self.head(h).squeeze(-1)

    @torch.no_grad()
    def predict(self, env: PureRoutingEnv) -> float:
        data = build_state_graph(env)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        return float(self(data).item())


# 图缓存（按 state_key）——MCTS 叶子重复评估时省构建成本
_graph_cache: Dict[Tuple[int, Tuple[int, ...]], Data] = {}


def cached_graph(env: PureRoutingEnv) -> Data:
    key = env.state_key()
    d = _graph_cache.get(key)
    if d is None:
        d = build_state_graph(env)
        d.batch = torch.zeros(d.num_nodes, dtype=torch.long)
        if len(_graph_cache) > 200_000:
            _graph_cache.clear()
        _graph_cache[key] = d
    return d


class CachedGNNValueNet(nn.Module):
    """带状态缓存 + 批量推理的 GNN 价值网络（MCTS 用）。"""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = GNNValueNet(hidden)

    def load_state_dict(self, state_dict, strict=True):
        return self.net.load_state_dict(state_dict, strict=strict)

    def state_dict(self, *args, **kwargs):
        return self.net.state_dict(*args, **kwargs)

    def predict(self, env: PureRoutingEnv) -> float:
        return float(self.net(cached_graph(env)).item())

    def predict_batch(self, envs) -> np.ndarray:
        datas = [cached_graph(e) for e in envs]
        batch = torch.cat([torch.full((d.num_nodes,), i, dtype=torch.long)
                           for i, d in enumerate(datas)])
        merged = Data(
            x=torch.cat([d.x for d in datas]),
            edge_index=torch.cat([d.edge_index + off for off, d in
                                  zip(np.cumsum([0] + [d.num_nodes for d in datas[:-1]]),
                                      datas)], dim=1),
            edge_attr=torch.cat([d.edge_attr for d in datas]),
            glob=torch.stack([d.glob for d in datas]),
        )
        merged.batch = batch
        with torch.no_grad():
            return self.net(merged).cpu().numpy()
