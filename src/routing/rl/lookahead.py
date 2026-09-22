# ============================================================================
# lookahead.py
# 训练期 beam expectimax 展开（doc/20260916训练方案.md §4）：
# 从当前状态按 Actor top-B 候选动作逐层展开 K 层，叶节点用 V_LA 头批量评估，
# expectimax 回溯得到 (a*, v*, q_roots)。
#
#   V_LA(s_t) ← max_{a∈topB} [ r(s_t,a) + γ·V_child(s_{t+1}(a)) ]
#   V_child(s) = 0            （done/truncated）
#              = V_LA(s)      （depth K 叶节点，自举）
#              = max_{a'} Q   （中间节点）
#
# 复用现有机制：env.clone() / step(compute_obs=False) / build_graph_data() /
# gnn.node_embeddings_batched / agent._forward_obs_batch_vla。
# 掩码逻辑与 eval_policy.evaluate_circuit_beam 逐行对齐。
# ============================================================================

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch


def _build_mask(env, agent) -> torch.Tensor:
    """与 evaluate_circuit_beam 对齐的动作掩码（valid edges + deadlock +
    unmapped + commit-by-phase），返回 (1, n_a) bool。"""
    n_a = agent.num_edges + 1 if agent.with_commit else agent.num_edges
    mask = torch.zeros(n_a, dtype=torch.bool, device=agent.device)
    mask[:len(env.coupling_map)] = True
    if hasattr(env, 'get_deadlock_mask'):
        dm = env.get_deadlock_mask()
        um = env.get_unmapped_mask()
        for i in range(min(len(dm), len(mask))):
            if dm[i] or um[i]:
                mask[i] = False
    if agent.with_commit:
        mask[agent.num_edges] = env.mapping_phase
    return mask.unsqueeze(0)


def _top_actions(agent, obs, mask: torch.Tensor, beam_width: int) -> List[int]:
    """masked logits 的 top-B 候选动作。"""
    logits, _ = agent._forward_obs(obs, action_mask=mask)
    scores = logits[0].clone()
    scores[~mask[0]] = -1e9
    k = min(beam_width, int(mask[0].sum().item()))
    return scores.topk(k).indices.tolist()


def _batched_clone_obs(agent, clones) -> List[np.ndarray]:
    """批量计算 clone 状态的 obs（单次 GNN 批量前向替代逐 clone 计算）。"""
    if not clones:
        return []
    if agent.gnn is not None:
        gds = [c.build_graph_data() for c in clones]
        qhs = agent.gnn.node_embeddings_batched(gds)
        return [c._obs(qubit_h=qh.cpu().numpy()) for c, qh in zip(clones, qhs)]
    return [c._obs() for c in clones]


def _batched_vla(agent, clones) -> List[float]:
    """批量 V_LA 叶评估（一次 GNN 批量 + 一次 EdgeActorCritic 批量）。"""
    obs_list = _batched_clone_obs(agent, clones)
    if not obs_list:
        return []
    _, vla = agent._forward_obs_batch_vla(np.stack(obs_list))
    return [float(v) for v in vla]


@torch.no_grad()
def beam_expectimax(env, agent, obs: np.ndarray,
                    beam_width: int = 3, depth: int = 2,
                    gamma: float = 0.99) -> Tuple[int, float, List[Tuple[int, float]]]:
    """从 (env, obs) 做受限 expectimax 展开（全程 no_grad：诊断/目标计算，
    不参与反向图——训练 rollout 期间 GNN 参数带梯度，必须隔离）。

    返回 (a_star, v_star, q_roots)：
      a_star  : beam 最优首动作（蒸馏目标用）
      v_star  : 根节点 expectimax 值（V_LA 回归目标）
      q_roots : 根候选 (action, Q) 列表（actor-beam agreement 诊断用）
    """
    mask = _build_mask(env, agent)
    cand = _top_actions(agent, obs, mask, beam_width)
    if not cand:
        # 全掩码状态（所有边被死锁/未映射掩掉）：无候选可展开，
        # 返回空 q_roots，调用方应退化为非 anchor
        return 0, 0.0, []
    # A1'：root 层同样施用超预算×无进展惩罚（与部署 beam 每步一致）
    root_budget = getattr(env, 'sabre_swap_budget', None)
    root_lam = getattr(env, 'lambda_budget', 0.0)
    root_phi = env._phi() if (root_budget is not None and root_lam > 0) else None

    # levels[d-1] = 深度 d 的节点列表；(clone, r_from_parent, done, parent_idx)
    lvl = []
    for a in cand:
        c = env.clone()
        _, r_c, done_c, trunc_c, _ = c.step(a, compute_obs=False)
        if (root_budget is not None and not done_c and not trunc_c):
            over = c._swap_counter - root_budget
            if over > 0:
                prog = ((len(c.executed) - len(env.executed)) > 0
                        or (c._phi() > root_phi + 1e-9))
                if not prog:
                    r_c -= root_lam * over
        lvl.append((c, float(r_c), bool(done_c or trunc_c), -1))
    levels = [lvl]

    # 逐层展开（每层非终节点批量 obs，一次 GNN 批量）
    for _d in range(2, depth + 1):
        prev = levels[-1]
        open_nodes = [n for n in prev if not n[2]]
        open_obs = iter(_batched_clone_obs(agent, [n[0] for n in open_nodes]))
        # A1'：与部署 beam 同口径的超预算×无进展惩罚（0917 §2.6.2）
        budget = getattr(env, 'sabre_swap_budget', None)
        lam = getattr(env, 'lambda_budget', 0.0)
        cur = []
        for idx, (clone, _r, done, _p) in enumerate(prev):
            if done:
                continue
            obs_c = next(open_obs)
            m_c = _build_mask(clone, agent)
            parent_phi = clone._phi() if (budget is not None and lam > 0) else None
            for a2 in _top_actions(agent, obs_c, m_c, beam_width):
                c2 = clone.clone()
                _, r2, d2, t2, _ = c2.step(a2, compute_obs=False)
                # 超预算且既未解锁门、也未改善 Φ 的候选：按超支深度罚
                if (budget is not None and not d2 and not t2):
                    over = c2._swap_counter - budget
                    if over > 0:
                        prog = ((len(c2.executed) - len(clone.executed)) > 0
                                or (c2._phi() > parent_phi + 1e-9))
                        if not prog:
                            r2 -= lam * over
                cur.append((c2, float(r2), bool(d2 or t2), idx))
        levels.append(cur)

    # 叶评估（最后一层非终节点批量 V_LA）
    leaf = levels[-1]
    vla_vals = iter(_batched_vla(agent, [n[0] for n in leaf if not n[2]]))
    vals = [0.0 if done else next(vla_vals) for (_, _, done, _) in leaf]

    # expectimax 回溯：V(node) = 0(done) | max_{children}(r_child + γ·V_child)
    for d in range(len(levels) - 2, -1, -1):
        parents, children = levels[d], levels[d + 1]
        new_vals = []
        for i, (_c, _r, done, _p) in enumerate(parents):
            if done:
                new_vals.append(0.0)
                continue
            best = None
            for j, (_cc, r_c, _dc, pj) in enumerate(children):
                if pj != i:
                    continue
                q = r_c + gamma * vals[j]
                best = q if best is None or q > best else best
            new_vals.append(best if best is not None else 0.0)
        vals = new_vals

    # 根候选 Q：done → r（无自举）；否则 r + γ·V
    q_roots: List[Tuple[int, float]] = []
    for i, (_c, r1, done, _p) in enumerate(levels[0]):
        q = r1 if done else r1 + gamma * vals[i]
        q_roots.append((cand[i], q))

    a_star, v_star = max(q_roots, key=lambda x: x[1])
    return int(a_star), float(v_star), q_roots
