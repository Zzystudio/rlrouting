"""v0 MCTS 内核 —— 经典 PUCT，prior 与 value 正交可插拔。

U(s,a) = Q(s,a) + c_puct · P(s,a) · sqrt(N(s)) / (1 + N(s,a))

设计（对齐 20260922 方案第八节）：
  - prior ∈ {uniform, sabre}（learned 留 P2/P4）
  - value ∈ {oracle(-V*), rollout_random, rollout_sabre}（learned V_θ 留 P2/P3）
  - backup: avg（默认）或 max
  - 根选择: argmax N（访问数）——规避 Stage-0 教训：未访问子节点 Q=0 恒
    beats 负值，若按 Q 选会挑到从未评估的动作
  - 每次决策重建树（经典做法）；搜索中收集访问状态供 P2 搜索分布数据集

Leaf 评估（backup 到路径上所有祖先）：
  - oracle: ExactSolver.solve(child) → -V*（= 剩余最优 SWAP 数取负）
  - rollout_*: 固定 rollout 策略跑到底 → -swaps
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .exact_solver import ExactSolver
from .pure_env import PureRoutingEnv
from .sabre_heuristic import SabreScorer, greedy_rollout, random_rollout, robust_rollout


@dataclass
class MCTSConfig:
    sims: int = 128
    c_puct: float = 2.0
    top_k: int = 0                 # 0 = 展开全部合法动作
    prior: str = "uniform"         # uniform | sabre
    value: str = "oracle"          # oracle | rollout_random | rollout_sabre
    backup: str = "avg"            # avg | max
    beta: float = 2.0              # sabre prior 温度
    max_depth: int = 200
    rollout_cap: int = 0           # >0: rollout 步数上限（M0 诊断用，超限返回 -cap）
    seed: int = 0


class PUCTNode:
    __slots__ = ("state", "parent", "action", "priors", "prior", "children",
                 "N", "Q", "expanded", "n_legal")

    def __init__(self, state, parent=None, action=-1, priors=None, prior=0.0,
                 n_legal=0):
        self.state = state          # (executed_mask, mapping)
        self.parent = parent
        self.action = action
        self.priors = priors        # np.ndarray (E,)（仅根/展开节点，供 top-K 排序）
        self.prior = prior          # 本边先验 P(parent, action)（标量）
        self.children: Dict[int, PUCTNode] = {}
        self.N = 0.0
        self.Q = 0.0
        self.expanded = False
        self.n_legal = n_legal


def _prior(env: PureRoutingEnv, cfg: MCTSConfig, scorer: Optional[SabreScorer]) -> np.ndarray:
    if cfg.prior == "sabre" and scorer is not None:
        return scorer.prior(env, cfg.beta)
    legal = env.legal_actions()
    p = np.zeros(env.num_edges, dtype=np.float64)
    p[legal] = 1.0 / len(legal)
    return p


def _eval_state(env: PureRoutingEnv, cfg: MCTSConfig,
                solver: Optional[ExactSolver],
                value_net=None, noise_ctx=None) -> float:
    """叶子价值（越大越好）——对 v0 即 -剩余SWAP数 的估计。"""
    if cfg.value == "oracle":
        assert solver is not None
        v = solver.solve(env.executed_mask, env.mapping)
        return float(-v) if v is not None else float("-inf")
    if cfg.value == "learned":
        assert value_net is not None
        return -float(value_net.predict(env))
    if cfg.value == "noise":
        assert noise_ctx is not None
        from .noise_rollout import noise_aware_rollout
        e2 = env.clone()
        cost, ok, _tl = noise_aware_rollout(
            e2, noise_ctx["config"], dist_noise=noise_ctx.get("dist_noise"))
        return -cost if ok else -1e6
    if cfg.value == "rollout_random":
        rng = np.random.default_rng(cfg.seed)
        n, ok = random_rollout(env.clone(), rng,
                               max_steps=cfg.rollout_cap or 5000)
        if not ok:
            # 有限惩罚而非 -inf：循环/超限分支代价 = cap，树能据此避开而非整棵作废
            return float(-(cfg.rollout_cap or 5000))
        return float(-n)
    if cfg.value == "rollout_sabre":
        n, ok = robust_rollout(env.clone(), SabreScorer(),
                               max_steps=cfg.rollout_cap or 2000)
        if not ok:
            return float(-(cfg.rollout_cap or 2000))
        return float(-n)
    raise ValueError(f"unknown value type: {cfg.value}")


def _backup(node: PUCTNode, v: float, cfg: MCTSConfig, leaf_depth: int) -> None:
    """回传叶子价值 v（= -剩余SWAP），沿路径计入已走 SWAP 数：
    深度 d 节点的 playout 价值 = v - (leaf_depth - d)（= -从该节点到终点的总成本）。
    否则根 Q 只含剩余成本、系统性高估——价值标签不可用（2026-09-24 修复）。
    """
    d = leaf_depth
    while node is not None:
        node.N += 1.0
        v_node = v - (leaf_depth - d)
        if cfg.backup == "avg":
            node.Q += (v_node - node.Q) / node.N
        elif cfg.backup == "max":
            node.Q = max(node.Q, v_node)
        d -= 1
        node = node.parent


def _select_child(node: PUCTNode, cfg: MCTSConfig) -> PUCTNode:
    best, best_u = None, float("-inf")
    sqrt_n = np.sqrt(node.N)
    for a, c in node.children.items():
        u = c.Q + cfg.c_puct * c.prior * sqrt_n / (1.0 + c.N)
        if u > best_u:
            best_u, best = u, c
    return best


def mcts_search(env: PureRoutingEnv, cfg: MCTSConfig,
                solver: Optional[ExactSolver] = None,
                scorer: Optional[SabreScorer] = None,
                value_net=None, noise_ctx=None,
                visited_log: Optional[List[Tuple[Tuple[int, Tuple[int, ...]], float]]] = None,
                avoid_states: Optional[set] = None
                ) -> Tuple[int, Dict]:
    """单次决策：在 env 当前状态上建树搜索，返回 (动作, 统计)。"""
    if env.is_terminal():
        return -1, {"terminal": True}
    root = PUCTNode(env.state_key(), n_legal=len(env.legal_actions()))
    root.priors = _prior(env, cfg, scorer)
    root.expanded = True

    work_env = env.clone()
    expansions = 0
    for _ in range(cfg.sims):
        # -- 选择：沿 UCB 下潜到未展开叶子（用 work_env 同步状态） --
        node = root
        work_env.set_state(*node.state)
        depth = 0
        while node.expanded and node.children and not work_env.is_terminal():
            child = _select_child(node, cfg)
            work_env.step(child.action)
            node = child
            depth += 1
            if depth > cfg.max_depth:
                break

        # -- 评估叶子 --
        if work_env.is_terminal():
            v = 0.0
        else:
            v = _eval_state(work_env, cfg, solver, value_net, noise_ctx)
            if v == float("-inf"):
                continue  # oracle 超预算：跳过该 playout
            # -- 展开叶子：建子节点 --
            if visited_log is not None:
                visited_log.append((work_env.state_key(), v))
            node.expanded = True
            node.n_legal = len(work_env.legal_actions())
            node.priors = _prior(work_env, cfg, scorer)
            succ = work_env.all_successors()
            if cfg.top_k > 0 and len(succ) > cfg.top_k:
                order = np.argsort(-node.priors[work_env.legal_actions()])
                legal = work_env.legal_actions()
                keep = set(legal[order[:cfg.top_k]].tolist())
                succ = [(a, s) for (a, s) in succ if a in keep]
            for a, s in succ:
                node.children[a] = PUCTNode(s, parent=node, action=a,
                                            prior=float(node.priors[a]),
                                            n_legal=0)
        expansions += 1
        _backup(node, v, cfg, depth)

    # -- 根选择：argmax N（未访问子节点不参与）；avoid_states 非空时
    #    优先选择不回到已访问状态的动作（episode 级防循环，superko 类比）--
    legal = env.legal_actions()
    scored = []
    for a in legal:
        c = root.children.get(a)
        if c is None or c.N <= 0:
            continue
        loops = (avoid_states is not None and env.successor_state(a) in avoid_states)
        scored.append((loops, -c.N, a, c))
    scored.sort(key=lambda t: (t[0], t[1]))
    if scored and not scored[0][0]:
        best_a = scored[0][2]
        best_n = -scored[0][1]
    elif scored:
        best_a = scored[0][2]  # 全部循环：取访问最多者
        best_n = -scored[0][1]
    else:
        best_a = int(legal[np.argmax(root.priors[legal])])
        best_n = 0.0
    # best_child_Q：最佳动作子节点的 Q（1 步前瞻最优值，比 root 均值更接近 V*）。
    # Q=-cost → 最佳 = max(Q)（最低成本）；min(Q) 会选到最差分支（2026-09-24 修复）。
    bq = None
    for a in legal:
        c = root.children.get(a)
        if c is not None and c.N > 0:
            bq = c.Q if bq is None else max(bq, c.Q)
    stats = {"sims": cfg.sims, "expansions": expansions, "root_N": root.N,
             "best_a": best_a, "best_N": best_n, "root_Q": float(root.Q),
             "best_child_Q": float(bq) if bq is not None else float(root.Q)}
    return best_a, stats


def mcts_episode(env: PureRoutingEnv, cfg: MCTSConfig,
                 solver: Optional[ExactSolver] = None,
                 scorer: Optional[SabreScorer] = None,
                 value_net=None, noise_ctx=None,
                 max_steps: int = 5000,
                 collect_log: bool = False,
                 avoid_cycles: bool = True) -> Tuple[int, bool, Dict, Optional[List]]:
    """整局 MCTS 路由。返回 (swaps, ok, stats, visited_log)。

    avoid_cycles=True：episode 级防循环（根选择避开已访问状态）。对 rollout
    价值无害；对 learned V **有害**——V_θ 偏好的动作被强行改道 → 越改越差
    （2026-09-25：rank 模型无 avoidance 24 步收敛，带 avoidance 游走 600 步）。
    """
    visited_log: List[Tuple[Tuple[int, Tuple[int, ...]], float]] = [] if collect_log else None
    seen_states = {env.state_key()} if avoid_cycles else None
    steps = 0
    while not env.is_terminal():
        if steps >= max_steps:
            return steps, False, {"aborted": True}, visited_log
        a, st = mcts_search(env, cfg, solver=solver, scorer=scorer,
                            value_net=value_net, noise_ctx=noise_ctx,
                            visited_log=visited_log,
                            avoid_states=seen_states)
        env.step(a)
        if seen_states is not None:
            seen_states.add(env.state_key())
        steps += 1
    stats = {"steps": steps, "swaps": steps}
    return steps, True, stats, visited_log
