"""v0 价值网络 —— 极简 MLP on 手工特征（P2a 控制组，先于 GNN）。

核心科研问题（20260922 方案十五节）：GNN/MLP 能否学到 quantum routing 的
long-horizon cost-to-go？第一步用 12 维手工特征做控制组，把"表征难"与
"价值学不出"解耦；若 MLP 已可拟合 V* 且 OOD 泛化可接受，再考虑 GNN。

特征设计（state_features, 12 维，全部归一化到 ~[0,1]）：
  f0  n_rem_2q / G                    剩余 2Q 门占比
  f1  n_ready / n_rem                 前沿占比（依赖就绪程度）
  f2  (avg_dist - 1) / (diam-1)       剩余门平均距离（超出 1 的部分）
  f3  (max_dist - 1) / (diam-1)       剩余门最大距离
  f4  sum_dist / (E · n_rem)          归一化总距离需求
  f5  n_adjacent / n_rem              已邻接剩余门占比（免费执行潜力）
  f6  n_legal / E                     可用动作占比
  f7  n_long(≥3) / n_rem              远距门占比（需多次 SWAP）
  f8  std_dist / diam                 距离分布离散度
  f9  ready_avg_dist                  仅前沿门的平均距离
  f10 n_qubits_with_pending / nq      有剩余门负载的比特占比
  f11 h(s) / diam                     A* 可采纳启发式 max(d-1)（强特征）
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .pure_env import PureRoutingEnv

FEAT_DIM = 12


def _dist_matrix(env: PureRoutingEnv) -> np.ndarray:
    return env.dist


def state_features(env: PureRoutingEnv) -> np.ndarray:
    nq = env.num_logical
    n_phys = env.n_phys
    diam = max(1, int(env.dist.max()))
    G = max(1, len(env._twoq))
    E = max(1, env.num_edges)

    rem = env.remaining_2q()
    n_rem = len(rem)
    if n_rem == 0:
        return np.zeros(FEAT_DIM, dtype=np.float32)

    pairs = [(env.dag.gates[i].qubits[0], env.dag.gates[i].qubits[1]) for i in rem]
    dists = np.array([env.dist[env.mapping[a], env.mapping[b]] for a, b in pairs],
                     dtype=np.float32)
    ready_set = set(env.ready_2q())
    n_ready = sum(1 for i in rem if i in ready_set)
    n_adj = int((dists <= 1.0).sum())
    n_long = int((dists >= 3.0).sum())
    ready_dists = [env.dist[env.mapping[a], env.mapping[b]] for i, (a, b) in
                   zip(rem, pairs) if i in ready_set]
    h = max(0.0, float((dists - 1.0).max()))

    pending_q = set()
    for a, b in pairs:
        pending_q.add(a)
        pending_q.add(b)

    f = np.zeros(FEAT_DIM, dtype=np.float32)
    f[0] = n_rem / G
    f[1] = n_ready / max(1, n_rem)
    f[2] = (float(dists.mean()) - 1.0) / max(1, diam - 1)
    f[3] = (float(dists.max()) - 1.0) / max(1, diam - 1)
    f[4] = float(dists.sum()) / (E * max(1, n_rem))
    f[5] = n_adj / max(1, n_rem)
    f[6] = len(env.legal_actions()) / E
    f[7] = n_long / max(1, n_rem)
    f[8] = float(dists.std()) / max(1, diam)
    f[9] = (float(np.mean(ready_dists)) - 1.0) / max(1, diam - 1) if ready_dists else 0.0
    f[10] = len(pending_q) / max(1, nq)
    f[11] = h / max(1, diam)
    return f


class ValueNet(nn.Module):
    def __init__(self, in_dim: int = FEAT_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict(self, env: PureRoutingEnv) -> float:
        x = torch.as_tensor(state_features(env), dtype=torch.float32).unsqueeze(0)
        return float(self(x).item())


def train_value_net(net: ValueNet, X: np.ndarray, y: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    epochs: int = 400, lr: float = 1e-3,
                    batch: int = 128, patience: int = 80,
                    device: str = "cpu") -> Tuple[ValueNet, dict]:
    """训练 V_θ(s) ≈ V*(s)（MSE），early stop on val。返回 (net, history)。"""
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    Xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    n = len(Xt)
    best_mae, best_state, bad = float("inf"), None, 0
    history = []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            pred = net(Xt[idx])
            loss = loss_fn(pred, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        net.eval()
        with torch.no_grad():
            pred_v = net(Xv)
            mae = float((pred_v - yv).abs().mean().item())
        history.append({"epoch": ep, "train_mse": total / n, "val_mae": mae})
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net, {"best_val_mae": best_mae, "epochs_done": len(history),
                 "history": history}


def value_greedy_policy(net: ValueNet):
    """value-greedy 诊断：π(s) = argmin_a V_θ(successor(s))——不带搜索测 V_θ 质量。"""
    def _p(env: PureRoutingEnv) -> int:
        best_a, best_v = None, float("inf")
        with torch.no_grad():
            x0 = torch.as_tensor(state_features(env), dtype=torch.float32)
            for a, (mask2, mapping2) in env.all_successors():
                env2 = env.clone()
                env2.set_state(mask2, mapping2)
                x = torch.as_tensor(state_features(env2), dtype=torch.float32).unsqueeze(0)
                v = float(net(x).item())
                if v < best_v:
                    best_v, best_a = v, a
        return best_a
    return _p
