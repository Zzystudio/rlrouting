#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""方向2共享工具：SABRE 脚本路由（Layer 0）+ 调度-only 环境/策略。

被 export_sabre_routed.py（Layer 0 缓存 + sref）与
probe_sched_fid_spread.py（探针 0a）共用。

约定：电路经完整 SABRE（SabreLayout+SabreSwap，不分解门）路由为物理电路，
SWAP 已物化为 2Q 门（时长 0.9us，与 GATE_DURATION_TABLE / v3 一致），
ClockedRoutingEnv(scheduling_only=True) 恒等映射 + 屏蔽 SWAP 动作，
RL/启发式只做 EXEC/SKIP 时序调度。
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.transpiler.passes import SabreLayout, SabreSwap

from routing.graph.circuit_dag import CircuitDAG
from routing.rl.env_clocked import ClockedRoutingEnv


# ---------------------------------------------------------------------------
#  Layer 0：完整 SABRE 路由
# ---------------------------------------------------------------------------
def sabre_route_full(qc, config, swap_trials: int = 5, seed: int = 0):
    """SabreLayout + SabreSwap（不 transpile，保门型与 v3 时长表兼容）。

    返回 (phys, n_swaps, init_layout)。
    """
    cmap = CouplingMap(list(config.coupling_map))
    c = qc
    if qc.num_qubits < cmap.size():
        c = QuantumCircuit(cmap.size(), qc.num_clbits)
        c.compose(qc, inplace=True)
    pm = PassManager([
        SabreLayout(coupling_map=cmap, seed=seed),
        SabreSwap(coupling_map=cmap, heuristic="decay", trials=swap_trials,
                  seed=seed),
    ])
    phys = pm.run(c)
    n_swaps = sum(1 for inst in phys.data if inst.operation.name == "swap")
    layout = None
    lay = pm.property_set.get("layout")
    if lay is not None:
        try:
            vb = lay.get_virtual_bits()
            n = qc.num_qubits
            layout = [int(vb[qc.qubits[i]].index) for i in range(n)]
        except Exception:
            layout = None
    return phys, n_swaps, layout


def sabre_route_best(qc, config, outer_seeds: int = 3, swap_trials: int = 5,
                     seed0: int = 0):
    """best-of-N 外层重启：多 seed × trials 取 swap 最少（平局取先）。

    返回 (phys, n_swaps, layout, best_seed)。
    """
    best = None
    for k in range(outer_seeds):
        seed = seed0 + k * 1013
        phys, nsw, layout = sabre_route_full(qc, config,
                                             swap_trials=swap_trials,
                                             seed=seed)
        if best is None or nsw < best[1]:
            best = (phys, nsw, layout, seed)
    return best


def routed_cache_path(cache_dir: str, topo_name: str, rel_path: str) -> str:
    base = os.path.splitext(rel_path)[0].replace(os.sep, "_")
    return os.path.join(cache_dir, topo_name, base + ".routed.pkl")


def save_routed_cache(path: str, phys, layout, n_swaps: int, trials: int,
                      seed: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"phys": phys, "layout": layout, "swaps": int(n_swaps),
                     "trials": int(trials), "seed": int(seed)}, f)


def load_routed_cache(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def parity_check(phys, n_swaps: int) -> bool:
    """平价回归校验：DAG 重建后 swap 门数一致。"""
    dag = CircuitDAG.from_circuit(phys)
    n = sum(1 for g in dag.gates if g.is_two_qubit and g.name == "swap")
    return n == n_swaps


# ---------------------------------------------------------------------------
#  调度-only 环境
# ---------------------------------------------------------------------------
def make_sched_env(phys, hw, cm, max_edges, max_num_qubits=20, max_ready=24,
                   max_episode_steps=3000, step_cap_factor=2.0):
    """routed 物理电路 → scheduling_only 时钟化环境（恒等映射、SWAP 屏蔽）。"""
    dag = CircuitDAG.from_circuit(phys)
    return ClockedRoutingEnv(
        dag, hw, cm, reward_mode="routing", reward_potential=True,
        pot_progress_b=0.2, swap_price_scale=4.6,
        mapping_phase=False, init_mapping=None,
        max_ready=max_ready, max_num_edges=max_edges,
        max_num_qubits=max_num_qubits,
        max_episode_steps=max_episode_steps, step_cap_factor=step_cap_factor,
        use_gnn=False, lookahead_features=True, edge_noise_features=True,
        beta_noise=0.5, shaping_gamma=0.99, eta_shape=0.3, alpha_ext=0.5,
        scheduling_only=True)


# ---------------------------------------------------------------------------
#  调度策略（EXEC 槽位选择 + 锁等待）
# ---------------------------------------------------------------------------
def pick_asap(legal, rng=None):
    """ASAP：priority-K 槽序（slot 0 = 最高优先级）。"""
    return legal[0]


def pick_anti(legal, rng=None):
    """反优先级：最后一个合法槽位。"""
    return legal[-1]


def pick_random(legal, rng):
    """均匀随机合法 EXEC 槽位。"""
    return int(legal[rng.integers(len(legal))])


POLICIES = {"asap": pick_asap, "anti": pick_anti, "random": pick_random}


def run_sched_episode(env, policy: str = "asap", rng=None, max_steps=4000):
    """调度-only episode：EXEC 优先（按 policy 选槽），无可 EXEC 则 SKIP 等锁。

    compute_obs=False 全程（无 GNN），返回 (done, steps)。
    """
    env.reset()
    done = False
    steps = 0
    E = env.num_edges
    K = env.max_ready
    while not done and steps < max_steps:
        env._update_candidates()
        mask = env.get_action_mask()
        legal = [i for i in range(E, E + K) if mask[i]]
        if legal:
            a = POLICIES[policy](legal, rng)
        else:
            a = env.skip_action
        try:
            _, _, done, trunc, _ = env.step(a, compute_obs=False)
        except RuntimeError:
            return False, steps
        done = done or trunc
        steps += 1
    return done, steps


def sched_episode_stats(env):
    """episode 后的 (swaps, makespan, par_density)。"""
    makespan = env.clock
    serial = float(getattr(env.timing, "serial_dur", 0.0))
    par = serial / makespan if makespan > 1e-9 else 0.0
    return env._swap_counter, makespan, par
