# ============================================================================
# value_net.py — Phase B: 学习价值函数（AlphaZero-lite，doc/20260922训练方案.md）
#
# 最小可行版：只学 V_φ（预测"从状态 s 到完成还需多少 swap"），π 保持
# sabre-score 先验。V_φ 在 MCTS self-play 自举数据上训练（精确 swap 目标，
# 无保真度模拟噪声），叶子评估用网络（~0.1ms）替代 rollout（~15ms）→
# MCTS sims 可提高 5-10 倍，突破 Phase A 的高分支/算力瓶颈。
# ============================================================================
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .mcts import MCTSConfig, mcts_search_p, joint_prior, cheap_rollout_value

FEAT_DIM = 12


def state_features(env) -> np.ndarray:
    """轻量状态特征（无 GNN），供 V_φ 预测剩余 swap 数。

    全部来自 env 现有方法与 dag/hw（O(G) 级，快）。12 维。
    """
    G = max(1, env.dag.num_gates)
    nq = max(1, env.num_qubits)
    n2q_total = max(1, len(env.dag.two_qubit_gates()))
    executed = env.executed
    rem_2q = sum(1 for g in env.dag.gates
                 if g.is_two_qubit and g.index not in executed)
    rem_all = G - len(executed)
    ready = env._ready_2q_gates()
    n_ready = len(ready)
    dist = env._dist()
    dmax = max(1.0, float(dist.max()))
    front_avg = (env._front_layer_dist() / max(1, n_ready)) if n_ready else 0.0
    ext_avg, n_ext = env._extended_set_dist()
    ext_avg = (ext_avg / max(1, n_ext)) if n_ext else 0.0
    bu = env._busy_until()
    busy_frac = float(np.mean(bu > env.clock + 1e-9))
    decay = env._qubit_swap_since_exec
    mapping = env.mapping
    dsum = 0.0
    dcnt = 0
    adj_cnt = 0
    for g in env.dag.gates:
        if g.is_two_qubit and g.index not in executed:
            pa, pb = mapping[g.qubits[0]], mapping[g.qubits[1]]
            dsum += dist[pa, pb]
            dcnt += 1
            if env.hw.adj[pa, pb] > 0:
                adj_cnt += 1
    mean_dist = dsum / max(1, dcnt) if dcnt else 0.0
    return np.array([
        len(executed) / G,                 # 0 progress
        rem_2q / n2q_total,                # 1 剩余 2q 比例
        rem_all / G,                       # 2 剩余全门比例
        min(front_avg / dmax, 2.0),        # 3 front 平均距离
        min(ext_avg / dmax, 2.0),          # 4 ext 平均距离
        n_ready / max(1, nq // 2),         # 5 ready 2q 数
        busy_frac,                         # 6 锁占用比例
        min(float(decay.mean()) / 5.0, 1.0),   # 7 平均 decay
        min(float(decay.max()) / 5.0, 1.0),    # 8 最大 decay
        min(mean_dist / dmax, 2.0),        # 9 未执行 2q 平均距离
        adj_cnt / max(1, rem_2q),          # 10 相邻比例
        env._swap_counter / max(1, G),     # 11 已用 swap 密度
    ], dtype=np.float32)


class ValueNet(nn.Module):
    """MLP 价值网络：state_features → 剩余 swap 数（回归）。"""

    def __init__(self, in_dim: int = FEAT_DIM, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.mlp(x).squeeze(-1)

    @torch.no_grad()
    def predict(self, feat: np.ndarray) -> float:
        x = torch.tensor(np.asarray(feat, dtype=np.float32), device=next(
            self.parameters()).device)
        return float(self(x).item())

    def make_value_fn(self):
        """返回 MCTS 叶子评估可调用：state -> 剩余 swap（正，越小越好）。
        MCTS 内部 outcome = -value（最大化负 swap）。"""
        self.eval()

        def vfn(s):
            f = torch.tensor(state_features(s), dtype=torch.float32,
                             device=next(self.parameters()).device)
            return float(self(f).item())
        return vfn


def _run_traj(env, policy, cap=4000):
    """用给定策略跑完一条轨迹，返回 (feats, sw_at, total_swaps)。"""
    env.reset()
    feats = []
    sw_at = []
    steps = 0
    while len(env.executed) < env.dag.num_gates and steps < cap:
        feats.append(state_features(env))
        sw_at.append(env._swap_counter)
        a = policy(env)
        try:
            env.step(a)
        except RuntimeError:
            break
        steps += 1
        if len(env.executed) >= env.dag.num_gates:
            break
    return feats, sw_at, env._swap_counter


def make_sabre_expert_policy(swap_seq, coupling_map):
    """SABRE 脚本路由专家策略：EXEC/SKIP 走守卫，换手取 SABRE 序列。

    swap_seq: run_sabre 物理线路中按序提取的 [(p,q),...]（物理索引）。
    返回策略 callable(env) -> action。
    """
    edge_lookup = {}
    for i, (p, q) in enumerate(coupling_map):
        edge_lookup[(p, q)] = i
        edge_lookup[(q, p)] = i
    seq = list(swap_seq)

    def policy(e):
        e._update_candidates()
        mask = e.get_action_mask()
        E = e.num_edges
        K = e.max_ready
        legal_exec = [i for i in range(E, E + K) if mask[i]]
        if legal_exec:
            return legal_exec[0]
        ready_adj = e._ready_2q_adjacent()
        if ready_adj and mask[e.skip_action]:
            return e.skip_action
        while seq:
            p, q = seq[0]
            ei = edge_lookup.get((p, q), edge_lookup.get((q, p), -1))
            if ei >= 0 and mask[ei]:
                seq.pop(0)
                return ei
            if mask[e.skip_action]:
                return e.skip_action      # 锁窗内 → 等锁释放
            seq.pop(0)                    # 死锁/未占用等 → 跳过该 SABRE swap
        return e.mimic_swap_index()
    return policy


def collect_selfplay(env_builder: Callable, circuits: List[Tuple],
                     cfg: MCTSConfig, net: Optional[ValueNet] = None,
                     tau: float = 1.0, expert_frac: float = 0.5,
                     expert_policy: Optional[Callable] = None,
                     seed: int = 0) -> List[Tuple[np.ndarray, float]]:
    """MCTS self-play + 专家轨迹混合生成数据。

    - expert_frac 比例的电路用专家策略（expert_policy 提供则用之，否则
      deterministic mimic）跑 → 低 swap 高质量轨迹 → V 学到"好策略下价值"
    - 其余用 MCTS 采样（探索、多样性）
    所有目标 = 实际剩余 swap（精确）。
    """
    from .mcts import deterministic_policy_step
    expert = expert_policy if expert_policy is not None else deterministic_policy_step
    rows: List[Tuple[np.ndarray, float]] = []
    rng = np.random.default_rng(seed)
    if net is not None:
        cfg.value_fn = net.make_value_fn()
    for dag, layout in circuits:
        if rng.random() < expert_frac:
            env = env_builder(dag, layout)
            feats, sw_at, total = _run_traj(env, expert)
            for f, sw in zip(feats, sw_at):
                rows.append((f, float(total - sw)))
            continue
        env = env_builder(dag, layout)
        env.reset()
        feats, sw_at = [], []
        steps = 0
        while len(env.executed) < env.dag.num_gates and steps < 4000:
            feats.append(state_features(env))
            sw_at.append(env._swap_counter)
            a, probs = mcts_search_p(env, cfg)
            if probs is None or float(probs.sum()) <= 0:
                priors, legal = joint_prior(env, cfg)
                probs = np.zeros(env.action_space.n)
                probs[legal] = priors[legal]
            if tau <= 0.0:
                # τ=0：argmax（与评估一致 → 价值自洽，消除乐观偏差）
                a = int(np.argmax(probs))
            else:
                p = probs ** (1.0 / tau)
                p = p / p.sum()
                a = int(rng.choice(len(p), p=p))
            try:
                env.step(a)
            except RuntimeError:
                break
            steps += 1
            if len(env.executed) >= env.dag.num_gates:
                break
        total = env._swap_counter
        for f, sw in zip(feats, sw_at):
            rows.append((f, float(total - sw)))
    return rows


def train_value(net: ValueNet, data: List[Tuple[np.ndarray, float]],
                epochs: int = 40, lr: float = 1e-3, batch: int = 256,
                device: str = "cpu") -> float:
    """训练 V_φ：MSE 回归剩余 swap 数。返回最终 loss。"""
    net.to(device)
    X = np.array([d[0] for d in data], dtype=np.float32)
    y = np.array([d[1] for d in data], dtype=np.float32)
    # 归一化目标（剩余 swap 量级差异大）
    y_mean = float(y.mean())
    y_std = float(y.std()) + 1e-8
    yn = (y - y_mean) / y_std
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()
    final = 0.0
    for ep in range(epochs):
        idx = np.random.permutation(len(X))
        ep_loss = 0.0
        nb = 0
        for i in range(0, len(X), batch):
            b = idx[i:i + batch]
            xb = torch.tensor(X[b], device=device)
            yb = torch.tensor(yn[b], device=device)
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        final = ep_loss / max(1, nb)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  [train_value] epoch {ep}: loss={final:.4f}")
    return final


# ============================================================================
# Stage 1: 相对价值（advantage）+ 不确定性 ensemble
# Q(s,a) = rollout锚(s_a) + λ(σ)·(μ_A(s,a) − β'·σ_A(s,a))
# 学习 A(s,a) = J_guard − J_a（"此动作比守卫动作好多少"），而非绝对剩余 swap。
# ensemble (μ,σ)：OOD 态 σ 大 → λ→0 + 悲观 → Q 退回 rollout 锚。
# ============================================================================
from .mcts import deterministic_policy_step as _guard_step

ADV_FEAT_DIM = 4   # 动作特征：type / prior_prob / extra / is_guard


def action_features(env, action: int, cfg) -> np.ndarray:
    """动作特征（4 维），与 12 维状态特征拼接喂 advantage 模型。"""
    priors, _ = joint_prior(env, cfg)
    E = env.num_edges
    K = env.max_ready
    guard = _guard_step(env)
    if action < E:
        typ = 0.0
        sc = env._edge_sabre_core_features()[action, 0]
        extra = sc
    elif action < E + K:
        typ = 1.0
        gi = env._candidate_slots[action - E]
        rem = env.dag.remaining_depths()
        md = max(1, env.dag.max_depth())
        extra = rem.get(gi, 0) / md if gi is not None else 0.0
    else:
        typ = 2.0
        extra = 0.0
    return np.array([typ, float(priors[action]), float(extra),
                     float(action == guard)], dtype=np.float32)


def rollout_cost(env, cfg, cap_steps: int = 30) -> float:
    """从 env 当前状态按守卫策略跑到完成（capped），返回剩余 swap 数（正）。"""
    return -cheap_rollout_value(env, cfg)


class AdvantageEnsemble:
    """K 个 MLP 的 ensemble，预测 A(s,a) = J_guard − J_a。μ=均值 σ=成员标准差。"""

    def __init__(self, in_dim: int = FEAT_DIM + ADV_FEAT_DIM, hidden: int = 64,
                 K: int = 5, seed: int = 0):
        self.K = K
        self.members = [ValueNet(in_dim=in_dim, hidden=hidden) for _ in range(K)]
        self.rng = np.random.default_rng(seed)

    def predict(self, feat: np.ndarray, a_feat: np.ndarray) -> Tuple[float, float]:
        x = np.concatenate([feat, a_feat]).astype(np.float32)
        xt = torch.tensor(x, dtype=torch.float32, device=next(
            self.members[0].parameters()).device)
        outs = [float(m(xt).item()) for m in self.members]
        mu = float(np.mean(outs))
        sig = float(np.std(outs))
        return mu, sig

    def make_adv_fn(self, env_builder=None):
        """返回 (s, a) -> (μ_A, σ_A)。s 为决策前状态。"""
        for m in self.members:
            m.eval()
        cfg = MCTSConfig()   # 仅 action_features 用先验

        def adv_fn(s, a):
            f = state_features(s)
            af = action_features(s, a, cfg)
            x = torch.tensor(np.concatenate([f, af]).astype(np.float32),
                             device=next(self.members[0].parameters()).device)
            outs = [float(m(x).item()) for m in self.members]
            return float(np.mean(outs)), float(np.std(outs))
        return adv_fn


def collect_advantage_data(env_builder, circuits, cfg, top_k: int = 5,
                           policy: Optional[Callable] = None,
                           stride: int = 2, max_states: int = 200,
                           max_traj_steps: int = 400,
                           seed: int = 0) -> List[Tuple[np.ndarray, float]]:
    """生成 (state+action 特征, advantage A=J_guard−J_a) 数据。

    沿轨迹（policy 提供，默认守卫策略；DAgger 时传 MCTS 策略）的每个状态
    （每隔 stride 个取一个，每电路上限 max_states 控成本），对守卫动作 +
    先验 top-K 动作各跑一次 capped rollout 得 J，A = J_guard − J_a。
    """
    traj_policy = policy if policy is not None else _guard_step
    rows: List[Tuple[np.ndarray, float]] = []
    for dag, layout in circuits:
        env = env_builder(dag, layout)
        env.reset()
        steps = 0
        n_labelled = 0
        while len(env.executed) < env.dag.num_gates and steps < max_traj_steps \
                and n_labelled < max_states:
            if steps % stride == 0:
                priors, legal = joint_prior(env, cfg)
                guard = _guard_step(env)
                cand = sorted(set([guard] + list(
                    np.flatnonzero(legal)[np.argsort(priors[legal])[::-1][:top_k]])))
                cg = env.clone(); cg.step(guard, compute_obs=False)
                J_guard = rollout_cost(cg, cfg)
                for a in cand:
                    ca = env.clone()
                    try:
                        ca.step(int(a), compute_obs=False)
                    except RuntimeError:
                        continue
                    J_a = rollout_cost(ca, cfg)
                    A = J_guard - J_a
                    af = action_features(env, int(a), cfg)
                    rows.append((np.concatenate(
                        [state_features(env), af]).astype(np.float32),
                        float(A)))
                n_labelled += 1
            a = traj_policy(env, cfg) if policy is not None else traj_policy(env)
            try:
                env.step(a)
            except RuntimeError:
                break
            steps += 1
            if len(env.executed) >= env.dag.num_gates:
                break
    return rows


def train_ensemble(ens: AdvantageEnsemble, data: List[Tuple[np.ndarray, float]],
                   epochs: int = 30, lr: float = 1e-3, batch: int = 256,
                   device: str = "cpu") -> float:
    """训练 ensemble：每成员用 bootstrap 子样本 MSE 回归 advantage。"""
    X = np.array([d[0] for d in data], dtype=np.float32)
    y = np.array([d[1] for d in data], dtype=np.float32)
    y_mean = float(y.mean()); y_std = float(y.std()) + 1e-8
    yn = (y - y_mean) / y_std
    final = 0.0
    for m in ens.members:
        m.to(device)
        idx = ens.rng.integers(0, len(X), len(X))   # bootstrap 重采样
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        lossf = nn.MSELoss()
        for ep in range(epochs):
            perm = ens.rng.permutation(len(idx))
            for i in range(0, len(idx), batch):
                b = idx[perm[i:i + batch]]
                xb = torch.tensor(X[b], device=device)
                yb = torch.tensor(yn[b], device=device)
                opt.zero_grad()
                loss = lossf(m(xb), yb)
                loss.backward()
                opt.step()
            final = float(loss.item())
    return final
