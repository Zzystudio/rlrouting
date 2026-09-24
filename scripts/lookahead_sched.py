#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Layer 1 调度候选：margin-gated 精确前瞻算子（MCTS Stage 1 的调度形态）。

机制（MCTS Phase A 诊断的直接推论）：
- 候选 = asap 动作 + 至多 k 个备选 EXEC 槽位（+SKIP 备选），启发式剪枝；
- 每个候选：clone → step → **确定性 asap rollout 到终点** → 调度感知解析
  fid 作叶子价值（环境确定性 → 无采样噪声）；
- 仅当最优备选超过 asap 自身价值 + margin 才偏离——margin 使轨迹贴近贪心，
  rollout 的"此后 asap"假设与现实一致，消除复合失效；
- 只在多合法槽位状态搜索（单候选状态无决策自由度）。

与 MCTS Phase A 的差异：动作空间 K+1（无 SWAP，平价由结构保证）、价值=
精确确定性 rollout+解析 fid（SNR≈∞）、偏离加 margin 门。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from scripts.sched_common import run_sched_episode


def make_lookahead_policy(analytic_fn: Callable,
                          k_alts: int = 2,
                          margin: float = 0.01,
                          verbose: bool = False):
    """返回 decide_fn(env) -> action，供贪心式 episode 驱动器使用。

    analytic_fn: _make_sched_analytic_fn(config) 的 fn(env)->float。
    margin: 绝对 fid 安全边际（备选必须严格优于此量才偏离 asap）。
    """

    def rollout_value(env0, a) -> Optional[float]:
        s = env0.clone()
        try:
            _, _, done, trunc, _ = s.step(a, compute_obs=False)
        except RuntimeError:
            return None
        if trunc:
            return None
        if not done:
            ok, _ = run_sched_episode(s, "asap")
            if not ok:
                return None
        return float(analytic_fn(s))

    def decide(env):
        env._update_candidates()
        mask = env.get_action_mask()
        E, K = env.num_edges, env.max_ready
        legal = [i for i in range(E, E + K) if mask[i]]
        det = legal[0] if legal else env.skip_action
        if len(legal) <= 1:
            return det
        v_det = rollout_value(env, det)
        if v_det is None:
            return det
        best_a, best_v = det, v_det
        for a in legal[1:1 + k_alts]:
            v = rollout_value(env, a)
            if v is not None and v > best_v:
                best_a, best_v = a, v
        if best_a != det and best_v > v_det + margin:
            if verbose:
                print(f"    [lookahead] deviate: {det} -> {best_a} "
                      f"(+{best_v - v_det:.4f})")
            return best_a
        return det

    return decide


def run_lookahead_episode(env, decide_fn, max_steps=4000):
    """与 run_sched_episode 同构，但动作由 decide_fn(env) 给出。"""
    env.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        try:
            a = decide_fn(env)
        except RuntimeError:
            return False, steps
        try:
            _, _, done, trunc, _ = env.step(a, compute_obs=False)
        except RuntimeError:
            return False, steps
        done = done or trunc
        steps += 1
    return done, steps
