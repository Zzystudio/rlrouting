"""sabre_heuristic — v0 独立 SABRE decay 打分（移植自 env.py:_edge_sabre_core_features 的
sabre_score 核心，去掉时钟化框架依赖），用于三处：
  1. 先验   P(a|s) ∝ exp(-β·score(s,a))
  2. greedy rollout 策略（叶子评估与 greedy 基线）
  3. 诊断：SABRE 打分与最优动作的相关性

打分语义（与 qiskit with_decay(0.001,5) 对齐）:
    score(e) = max(decay_p, decay_q) × (d_front + 0.5·d_ext)
    decay_p   = 1 + 0.001 × (自该物理比特上次 2Q 执行以来的换位数)，cap 5
    d_front   = 虚拟换位后 ready 2Q 门平均跳数距离
    d_ext     = 虚拟换位后扩展集（frontier 后继 BFS，cap=20）2Q 门平均跳数距离
低分 = 好。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .pure_env import PureRoutingEnv

DECAY_RATE = 0.001
DECAY_CAP = 5.0
EXT_SET_SIZE = 20


class SabreScorer:
    def __init__(self, ext_set_size: int = EXT_SET_SIZE):
        self.ext_set_size = ext_set_size

    # -- ext 集（移植 env.py:815-836 的 SABRE 式后继 BFS）----------------------
    def _ext_set(self, env: PureRoutingEnv) -> List[int]:
        ready = env.ready_2q()
        ready_idx = set(ready)
        succ = env.dag.successors()
        executed = env.executed_mask
        bit = env._bit
        ext_gates: List[int] = []
        seen = set(ready_idx)
        frontier = list(ready_idx)
        while frontier and len(ext_gates) < self.ext_set_size:
            nxt = []
            for gi in frontier:
                for sj in succ.get(gi, []):
                    if sj in seen or not env.dag.gates[sj].is_two_qubit:
                        continue
                    seen.add(sj)
                    nxt.append(sj)
                    if all((executed & bit[p]) != 0 or p in ready_idx or p in seen
                           for p in env.dag.gates[sj].predecessors):
                        ext_gates.append(sj)
                        if len(ext_gates) >= self.ext_set_size:
                            break
                if len(ext_gates) >= self.ext_set_size:
                    break
            frontier = nxt
        return ext_gates

    def scores(self, env: PureRoutingEnv,
               dist_override: Optional[np.ndarray] = None) -> np.ndarray:
        """per-edge 原始打分（低=好）。两端均空的边（no-op）置 +inf。
        dist_override: 噪声加权距离矩阵（N1 噪声感知 rollout 用）。"""
        E = env.num_edges
        out = np.full(E, np.inf, dtype=np.float64)
        ready = env.ready_2q()
        pairs = [(env.mapping[env.dag.gates[g].qubits[0]],
                  env.mapping[env.dag.gates[g].qubits[1]]) for g in ready]
        n_ready = max(1, len(ready))
        dist = env.dist if dist_override is None else dist_override
        inv = env._inv
        decay = np.minimum(1.0 + DECAY_RATE * env.swaps_since_exec, DECAY_CAP)

        ext_gates = self._ext_set(env) if ready else []
        n_ext = max(1, len(ext_gates))
        ext_qubits = [(env.dag.gates[g].qubits[0], env.dag.gates[g].qubits[1])
                      for g in ext_gates]

        for i, (p, q) in enumerate(env.coupling_map):
            lp, lq = inv[p], inv[q]
            if lp == -1 and lq == -1:
                continue
            m = list(env.mapping)
            if lp != -1 and lq != -1:
                m[lp], m[lq] = m[lq], m[lp]
            elif lp != -1:
                m[lp] = q
            else:
                m[lq] = p
            d_front = 0.0
            for (a, b) in pairs:
                na = q if a == p else (p if a == q else a)
                nb = q if b == p else (p if b == q else b)
                d_front += float(dist[na, nb])
            d_front /= n_ready
            d_ext = 0.0
            for (qa, qb) in ext_qubits:
                d_ext += float(dist[m[qa], m[qb]])
            d_ext /= n_ext
            md = max(decay[p], decay[q])
            out[i] = md * (d_front + 0.5 * d_ext)
        return out

    def best_action(self, env: PureRoutingEnv,
                    dist_override: Optional[np.ndarray] = None) -> int:
        sc = self.scores(env, dist_override)
        legal = env.legal_actions()
        sc = sc[legal]
        return legal[int(np.argmin(sc))]

    def prior(self, env: PureRoutingEnv, beta: float) -> np.ndarray:
        """P(a|s) ∝ exp(-β·score)，仅在合法动作上归一化。"""
        sc = self.scores(env)
        legal = env.legal_actions()
        vals = np.exp(-beta * sc[legal])
        p = np.zeros(env.num_edges, dtype=np.float64)
        p[legal] = vals / max(vals.sum(), 1e-12)
        return p


def greedy_rollout(env: PureRoutingEnv, scorer: SabreScorer,
                   max_steps: int = 5000,
                   cycle_detect: bool = True) -> Tuple[int, bool]:
    """greedy SABRE 策略：每步 argmin score 直到 terminal。

    16q 上发现 SABRE 贪心会在部分实例上振荡循环（decay 0.001 太弱，8q 不出现）：
    cycle_detect=True 时跟踪状态访问，检测到重复状态即提前终止返回
    (steps, False)——避免 5000 步空转，且让调用方知道这是循环失败而非成功。
    """
    steps = 0
    seen = set() if cycle_detect else None
    while not env.is_terminal():
        if steps >= max_steps:
            return steps, False
        if seen is not None:
            key = env.state_key()
            if key in seen:
                return steps, False  # 循环
            seen.add(key)
        e = scorer.best_action(env)
        env.step(e)
        steps += 1
    return steps, True


def random_rollout(env: PureRoutingEnv, rng: np.random.Generator,
                   max_steps: int = 5000) -> Tuple[int, bool]:
    steps = 0
    while not env.is_terminal():
        if steps >= max_steps:
            return steps, False
        legal = env.legal_actions()
        e = int(legal[rng.integers(0, len(legal))])
        env.step(e)
        steps += 1
    return steps, True


# ---------------------------------------------------------------------------
# 随机化 SABRE rollout（标签集成用）—— 确定性 argmin 会让 ensemble std≡0，
# 必须注入随机性：softmax(-β·score) 采样 + ε-greedy 两档。
# ---------------------------------------------------------------------------
def robust_rollout(env: PureRoutingEnv, scorer: SabreScorer,
                   retries: int = 3, beta: float = 5.0, epsilon: float = 0.1,
                   seed: int = 0, max_steps: int = 2000) -> Tuple[int, bool]:
    """循环鲁棒 rollout：确定性 SABRE 贪心；若循环，用随机 tie-break 重试
    （softmax β=5 采样，模拟 qiskit trials 的扰动），取最优结果。

    动机：-5000 惩罚会误杀"greedy 会循环但搜索能找到出路"的好状态；
    重试让循环状态有机会逃逸，rollout 仍是 SABRE 风格但不再否决。
    """
    n0, ok0 = greedy_rollout(env.clone(), scorer, max_steps=max_steps)
    if ok0:
        return n0, True
    best = n0
    rng = np.random.default_rng(seed)
    for _ in range(retries):
        e = env.clone()
        n, ok = stochastic_rollout(e, scorer, beta=beta, epsilon=epsilon,
                                   rng=rng, max_steps=max_steps)
        if ok:
            return n, True
        best = min(best, n)
    return best, False


def sample_action(env: PureRoutingEnv, scorer: SabreScorer,
                  beta: float = 1.0, epsilon: float = 0.0,
                  rng: Optional[np.random.Generator] = None) -> int:
    """采样动作：ε 概率随机，否则 softmax(-β·score) 加权采样。"""
    rng = rng if rng is not None else np.random.default_rng()
    sc = scorer.scores(env)
    legal = env.legal_actions()
    if epsilon > 0.0 and rng.random() < epsilon:
        return int(legal[rng.integers(0, len(legal))])
    vals = sc[legal]
    vals = np.nan_to_num(vals, nan=1e9, posinf=1e9)
    w = np.exp(-beta * (vals - vals.min()))
    w /= w.sum()
    return int(legal[rng.choice(len(legal), p=w)])


def stochastic_rollout(env: PureRoutingEnv, scorer: SabreScorer,
                       beta: float = 1.0, epsilon: float = 0.1,
                       rng: Optional[np.random.Generator] = None,
                       max_steps: int = 5000,
                       cycle_detect: bool = True) -> Tuple[int, bool]:
    rng = rng if rng is not None else np.random.default_rng()
    steps = 0
    seen = set() if cycle_detect else None
    while not env.is_terminal():
        if steps >= max_steps:
            return steps, False
        if seen is not None:
            key = env.state_key()
            if key in seen:
                return steps, False  # 循环（16q 上 SABRE 贪心会振荡，需检测）
            seen.add(key)
        env.step(sample_action(env, scorer, beta, epsilon, rng))
        steps += 1
    return steps, True


def ensemble_rollout_value(env: PureRoutingEnv, scorer: SabreScorer,
                           n: int = 8, beta: float = 1.0,
                           epsilon: float = 0.1, seed: int = 0,
                           max_steps: int = 2000) -> Tuple[float, float, int]:
    """N 次随机化 SABRE rollout 集成。返回 (mean_swaps, std_swaps, n_ok)。

    失败的 rollout（超 max_steps）丢弃并计数；全部失败返回 (inf, 0, 0)。
    """
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        e = env.clone()
        n_s, ok = stochastic_rollout(e, scorer, beta, epsilon, rng,
                                     max_steps=max_steps)
        if ok:
            vals.append(float(n_s))
    if not vals:
        return float("inf"), 0.0, 0
    a = np.asarray(vals, dtype=np.float64)
    return float(a.mean()), float(a.std()), int(len(a))
