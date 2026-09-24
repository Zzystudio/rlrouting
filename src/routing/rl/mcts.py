# ============================================================================
# mcts.py — 联合换手+调度 MCTS 搜索式路由（doc/20260922训练方案.md 设计）
#
# Phase A：推理算子。MCTS 在时钟化 MDP 上联合搜索 {SWAP 边 | EXEC 门 | SKIP}：
# - 先验 π(a|s)：三组状态相关联合先验（swap←sabre_score / exec←criticality
#   / skip←锁状态调制），把算力压到歧义分支
# - 叶子评估：确定性 rollout（mimic+ASAP-EXEC+锁等待守卫，无 NN）+ 解析余项
# - 确定性环境 → max backup（参数可切 avg）
# - 自适应触发：top-2 先验差距 < δ 或锁窗歧义才搜，否则直接 argmax
# 纯搜索 + 启发式，零 GNN/零 NN critic 依赖（避开 NN critic 失准的坑）。
# ============================================================================
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class MCTSConfig:
    def __init__(self, sims: int = 48, c_puct: float = 2.0,
                 max_depth: int = 15, rollout_k: int = 30,
                 backup: str = "max", swap_alpha: float = 2.0,
                 w_exec: float = 2.0, w_skip_base: float = 0.4,
                 skip_lock_boost: float = 1.5, trigger_delta: float = 0.08,
                 rem_2q_coef: float = 0.5, rem_dist_coef: float = 0.5,
                 expand_k: int = 8, rollout_mode: str = "cap",
                 seed: int = 0):
        self.sims = sims
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.rollout_k = rollout_k
        self.backup = backup
        self.swap_alpha = swap_alpha
        self.w_exec = w_exec
        self.w_skip_base = w_skip_base
        self.skip_lock_boost = skip_lock_boost
        self.trigger_delta = trigger_delta
        self.rem_2q_coef = rem_2q_coef
        self.rem_dist_coef = rem_dist_coef
        self.expand_k = expand_k      # 只展开先验 top-K 动作（分支 ~50 太大）
        self.rollout_mode = rollout_mode  # 'cap'=短 rollout+余项（快）| 'full'=完整贪心（准）
        self.value_fn = None          # 学习价值函数（Phase B）：state -> 价值（正=好）。
                                      # 非 None 时叶子用它替代 rollout。
        # ---- Stage 1：相对价值 + 不确定性 ----
        self.adv_fn = None            # (parent_state, action) -> (μ_A, σ_A)
        self.lambda0 = 1.0            # Q = rollout锚 + λ(σ)·(μ_A − β'·σ_A)
        self.beta = 1.0               # λ = λ0/(1+β·σ)
        self.beta_pess = 0.5          # 悲观系数
        self.sigma_trigger = None     # σ 阈值：σ>δ₂ → 强制搜索（None=关）
        self.rng = np.random.default_rng(seed)
        # 统计
        self.n_searched = 0
        self.n_total_sims = 0


class PUCTNode:
    __slots__ = ("visits", "q", "children", "parent", "action")

    def __init__(self, parent=None, action=None):
        self.visits = 0
        self.q = 0.0          # backup='max' 时=最优子结果（首次访问赋值）；
                              # 'avg' 时=累计和。注意：max 下初始 0 会让负
                              # outcome 卡 0（max(0,-50)=0），故首次访问直接赋
        self.children = {}    # action -> PUCTNode
        self.parent = parent
        self.action = action


def joint_prior(env, cfg: MCTSConfig) -> Tuple[np.ndarray, np.ndarray]:
    """三组联合先验 → (probs[full action space], legal mask)。

    显式刷新候选集（MCTS 内部 step 用 compute_obs=False 加速，候选集由
    这里按需重建，而非依赖 _obs）。
    """
    env._update_candidates()
    mask = env.get_action_mask()
    E = env.num_edges
    K = env.max_ready
    total = E + K + 2
    logits = np.full(total, -1e9)
    # ---- SWAP 组：sabre_score（低=好）----
    sc = env._edge_sabre_core_features()
    unmapped = env.get_unmapped_mask()
    deadlock = env.get_deadlock_mask()
    sc_legal = ~unmapped & ~deadlock
    if not sc_legal.any():
        sc_legal = ~unmapped
    logits[:E] = np.where(sc_legal, -cfg.swap_alpha * sc[:, 0], -1e9)
    # ---- EXEC 组：criticality + ready_age ----
    rem = env.dag.remaining_depths()
    md = max(1, env.dag.max_depth())
    for slot in range(K):
        gi = env._candidate_slots[slot]
        if gi is None or not mask[E + slot]:
            continue
        crit = rem.get(gi, 0) / md
        age = env._ready_since.get(gi, 0) / 16.0
        logits[E + slot] = cfg.w_exec * (crit + 0.3 * age)
    # ---- SKIP 组：锁状态调制 ----
    skip_logit = cfg.w_skip_base
    if mask[env.skip_action]:
        ready_adj = env._ready_2q_adjacent()
        legal_exec = [i for i in range(E, E + K) if mask[i]]
        n_locked = max(0, len(ready_adj) - len(legal_exec)) if ready_adj else 0
        skip_logit += cfg.skip_lock_boost * n_locked
        if legal_exec:
            skip_logit -= 1.0          # 有可执行门时压低 SKIP（ASAP 语义）
    logits[env.skip_action] = skip_logit
    # ---- softmax over legal ----
    legal = mask.astype(bool)
    m = logits[legal]
    m = m - m.max()
    p = np.exp(m)
    p = p / p.sum()
    probs = np.zeros(total)
    probs[legal] = p
    return probs, legal


def deterministic_policy_step(env) -> int:
    """叶子 rollout 策略（守卫版确定性推理策略）：
    EXEC 优先 → 就绪但被锁 → SKIP 等锁 → mimic swap。"""
    env._update_candidates()
    mask = env.get_action_mask()
    E = env.num_edges
    K = env.max_ready
    legal_exec = [i for i in range(E, E + K) if mask[i]]
    if legal_exec:
        return legal_exec[0]
    ready_adj = env._ready_2q_adjacent()
    if ready_adj and mask[env.skip_action]:
        return env.skip_action
    return env.mimic_swap_index()


def stochastic_policy_step(env, rng, skip_prob=0.7, tie_eps=0.05) -> int:
    """随机 rollout 策略（近贪心 + 随机性打破不动点）。

    MCTS 叶子价值若用确定性贪心 rollout，价值函数与贪心自洽 → 搜索永远
    确认贪心（不动点）。随机 rollout 的价值 = 随机近贪心玩法的期望结果，
    使 MCTS 能发现"贪心第一步较差但后续更好"的非短视分支。
    """
    env._update_candidates()
    mask = env.get_action_mask()
    E = env.num_edges
    K = env.max_ready
    legal_exec = [i for i in range(E, E + K) if mask[i]]
    if legal_exec:
        return legal_exec[rng.integers(len(legal_exec))]
    ready_adj = env._ready_2q_adjacent()
    if ready_adj and mask[env.skip_action]:
        if rng.random() < skip_prob:
            return env.skip_action
    return env.mimic_swap_index(tie_eps=tie_eps, rng=rng)


def full_rollout_value(env, cap: int = 600) -> float:
    """完整贪心 rollout 到完成（确定性策略），返回 -(新增 swap 数)。

    能区分"灾难性贪心续"（mimic 在 wadd 类电路 flail 到上千 swap）与正常续
    ——MCTS 借此逃出贪心的局部陷阱。贵（~1s/sim），配低 sims 用。
    """
    sw0 = env._swap_counter
    steps = 0
    n_gates = env.dag.num_gates
    while len(env.executed) < n_gates and steps < cap:
        try:
            a = deterministic_policy_step(env)
            _, _, done, trunc, _ = env.step(a, compute_obs=False)
        except RuntimeError:
            break
        steps += 1
        if done or trunc:
            break
    sw = env._swap_counter - sw0
    if len(env.executed) < n_gates:
        rem_2q = sum(1 for g in env.dag.gates
                     if g.is_two_qubit and g.index not in env.executed)
        sw += 0.5 * rem_2q
    return -sw


def cheap_rollout_value(env, cfg: MCTSConfig) -> float:
    """随机短 rollout（≤rollout_k 步）+ 解析余项 → 叶子价值。

    余项 = rem_2q_coef·剩余2Q门数 + rem_dist_coef·平均 front 距离，
    保证对未完成状态的价值估计单调（更少剩余 = 更好）。
    """
    sw0 = env._swap_counter
    steps = 0
    n_gates = env.dag.num_gates
    while len(env.executed) < n_gates and steps < cfg.rollout_k:
        try:
            a = stochastic_policy_step(env, cfg.rng)
            _, _, done, trunc, _ = env.step(a, compute_obs=False)
        except RuntimeError:
            break
        steps += 1
        if done or trunc:
            break
    sw = env._swap_counter - sw0
    if len(env.executed) < n_gates:
        rem_2q = sum(1 for g in env.dag.gates
                     if g.is_two_qubit and g.index not in env.executed)
        front_avg = env._front_layer_dist() / max(1, len(env._ready_2q_gates()))
        sw += cfg.rem_2q_coef * rem_2q + cfg.rem_dist_coef * front_avg
    return -sw


def _select_child(node: PUCTNode, priors: np.ndarray, cfg: MCTSConfig) -> int:
    best_a, best_v = -1, -np.inf
    for a, child in node.children.items():
        q = child.q if cfg.backup == "max" else child.q / max(1, child.visits)
        u = cfg.c_puct * priors[a] * np.sqrt(node.visits) / (1 + child.visits)
        v = q + u
        if v > best_v:
            best_v, best_a = v, a
    return best_a


def is_done(env) -> bool:
    return len(env.executed) == env.dag.num_gates


def _leaf_outcome(s, cfg: MCTSConfig) -> float:
    """叶子评估：学习价值函数（Phase B）或 rollout。"""
    if cfg.value_fn is not None:
        return float(cfg.value_fn(s))
    if cfg.rollout_mode == "full":
        return full_rollout_value(s)
    return cheap_rollout_value(s, cfg)


def mcts_search(env, cfg: MCTSConfig) -> int:
    """对当前 env 状态做一次 MCTS，返回最优动作（不修改原 env）。"""
    a, _ = mcts_search_p(env, cfg)
    return a


def _rollout_anchor(s, cfg: MCTSConfig) -> float:
    """Q 的 rollout 锚（无学习修正时的叶子评估）。"""
    if cfg.rollout_mode == "full":
        return full_rollout_value(s)
    return cheap_rollout_value(s, cfg)


def mcts_search_p(env, cfg: MCTSConfig) -> Tuple[int, Optional[np.ndarray]]:
    """MCTS 搜索，返回 (最优动作, 根访问分布 probs[full action space])。

    probs 用于 self-play 动作采样（π_MCTS ∝ N(root,a)^(1/τ)，τ 在调用方）。
    未展开的动作 probs=0。
    Stage 1：每条边的价值 = rollout锚(子状态) + λ(σ)·(μ_A − β'·σ_A)，
    其中 (μ_A, σ_A) = adv_fn(父状态, 动作)——相对守卫的改进 + 不确定性
    悲观修正（OOD 态 σ 大 → λ→0 + 悲观 → Q 退回 rollout 锚）。
    """
    root_env = env.clone()
    root = PUCTNode()
    for _ in range(cfg.sims):
        node = root
        s = root_env.clone()
        depth = 0
        edge_advs: list = []          # [(node, (μ_A, σ_A) or None)]
        # selection
        while node.children and depth < cfg.max_depth and not is_done(s):
            priors, _ = joint_prior(s, cfg)
            a = _select_child(node, priors, cfg)
            adv = cfg.adv_fn(s, a) if cfg.adv_fn is not None else None
            edge_advs.append((node, adv))
            node = node.children[a]
            try:
                s.step(a, compute_obs=False)
            except RuntimeError:
                edge_advs.pop()
                break
            depth += 1
        # expansion + simulation
        if not is_done(s) and depth < cfg.max_depth:
            priors, legal = joint_prior(s, cfg)
            legal_actions = np.flatnonzero(legal)
            topk = list(legal_actions[np.argsort(
                priors[legal_actions])[::-1][:cfg.expand_k]])
            guard = deterministic_policy_step(s)
            if guard not in topk and legal[guard]:
                topk.append(guard)   # 守卫动作恒在候选（保证最差 = 1 步前瞻≥守卫）
            topk = sorted(topk)[:cfg.expand_k]
            for a in topk:
                node.children[a] = PUCTNode(parent=node, action=int(a))
            a = int(topk[np.argmax(priors[topk])])
            adv = cfg.adv_fn(s, a) if cfg.adv_fn is not None else None
            edge_advs.append((node, adv))
            node = node.children[a]
            try:
                s.step(a, compute_obs=False)
            except RuntimeError:
                edge_advs = []
                outcome = -1e6
                node.visits += 1
                node.q = outcome if node.visits == 1 else (
                    max(node.q, outcome) if cfg.backup == "max"
                    else node.q + outcome)
                node = node.parent
                continue
            base = 0.0 if is_done(s) else _rollout_anchor(s, cfg)
        else:
            base = 0.0 if is_done(s) else _rollout_anchor(s, cfg)
        # backup：每条边价值 = base + λ·(μ_A − β'·σ_A)，逐边回传
        for (nd, adv) in reversed(edge_advs):
            if adv is not None:
                mu_a, sig_a = adv
                lam = cfg.lambda0 / (1.0 + cfg.beta * sig_a)
                outcome = base + lam * (mu_a - cfg.beta_pess * sig_a)
            else:
                outcome = base
            nd.visits += 1
            if nd.visits == 1:
                nd.q = outcome
            elif cfg.backup == "max":
                nd.q = max(nd.q, outcome)
            else:
                nd.q += outcome
    cfg.n_total_sims += cfg.sims
    # 根节点访问分布（未展开=0）
    probs = np.zeros(env.action_space.n)
    if root.children:
        visits = np.array([c.visits for c in root.children.values()])
        if visits.sum() > 0:
            p = visits / visits.sum()
            for (a, _c), pi in zip(root.children.items(), p):
                probs[int(a)] = float(pi)
        else:
            for a in root.children:
                probs[int(a)] = 1.0 / len(root.children)
    # 根节点选 Q 最优（tie-break 用先验）
    priors, _ = joint_prior(env, cfg)
    best_a, best_q = -1, -np.inf
    for a, child in root.children.items():
        if child.visits == 0:
            continue  # 未访问子节点不参与 argmax（Q=0 会 beats 所有负值动作）
        q = child.q if cfg.backup == "max" else child.q / child.visits
        if q > best_q + 1e-9 or (abs(q - best_q) <= 1e-9 and best_a != -1
                                 and priors[a] > priors[best_a]):
            best_q, best_a = q, a
    if best_a == -1:
        # 全部未访问（sims < 展开数）→ 回退先验 top-1（≈守卫选择）
        best_a = int(np.argmax(priors))
    return best_a, probs


def mcts_decide(env, cfg: MCTSConfig) -> Tuple[int, bool]:
    """自适应触发 + MCTS。返回 (action, searched)。

    触发条件：先验歧义（top-2 差 < δ）∨ 锁窗歧义 ∨ 价值不确定性（σ>δ₂）。
    """
    priors, legal = joint_prior(env, cfg)
    order = np.argsort(priors)[::-1]
    top2_gap = (priors[order[0]] - priors[order[1]]
                if len(order) > 1 else 1.0)
    E = env.num_edges
    K = env.max_ready
    legal_exec = [i for i in range(E, E + K) if legal[i]]
    ready_adj = env._ready_2q_adjacent()
    lock_ambiguous = (len(ready_adj) > 0) and (not legal_exec)
    sigma_high = False
    if cfg.adv_fn is not None and cfg.sigma_trigger is not None:
        guard = deterministic_policy_step(env)
        for a in [guard] + list(order[:cfg.expand_k]):
            _mu, sig = cfg.adv_fn(env, int(a))
            if sig > cfg.sigma_trigger:
                sigma_high = True
                break
    if (top2_gap >= cfg.trigger_delta and not lock_ambiguous
            and not sigma_high):
        return int(order[0]), False
    cfg.n_searched += 1
    return mcts_search(env, cfg), True
