"""baselines — v0 基线：env 工厂 / 拓扑加载 / Random / Greedy / SABRE(外部)。

初始映射协议（主）：identity —— 所有方法在同一初始映射下比较纯路由。
SABRE 基线：Qiskit SabreSwap（routing.sabre_route），trials/seed 协议可控；
            mean-over-seeds 与 best-of-trials 两条参照线。
"""

from __future__ import annotations

import json
import time
from typing import List, Optional, Tuple

import numpy as np

from ..graph.circuit_dag import CircuitDAG
from ..graph.features import HardwareFeatures
from .pure_env import PureRoutingEnv, raw_hop_distance


def load_topo(path: str) -> Tuple[dict, List[Tuple[int, int]], HardwareFeatures]:
    """加载拓扑 JSON，返回 (topo_dict, coupling_map, hw)。hw 仅用于 dist/adj。"""
    topo, coupling_map, hw, _cfg = load_topo_full(path)
    return topo, coupling_map, hw


def load_topo_full(path: str) -> Tuple[dict, List[Tuple[int, int]], HardwareFeatures, object]:
    """加载拓扑 JSON，返回 (topo_dict, coupling_map, hw, NoiseConfig)。"""
    with open(path) as f:
        topo = json.load(f)
    coupling_map = [tuple(e) for e in topo["coupling_map"]]
    cfg = _topo_to_config(topo)
    hw = HardwareFeatures.from_noise_config(cfg)
    return topo, coupling_map, hw, cfg


def _topo_to_config(topo: dict):
    from sim.sim import NoiseConfig  # 仅复用数据类，避免引入 RL 模块（PYTHONPATH=src 约定）

    dp = topo["device_params"]
    coupling_map = [tuple(e) for e in topo["coupling_map"]]

    def _norm_pair_dict(obj):
        """dict 键归一化为 int 元组（t287 系列 JSON 用字符串键 '(1, 0)'）。"""
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if isinstance(k, str):
                    a, b = k.strip("()").split(",")
                    out[(int(a), int(b))] = float(v)
                else:
                    out[(int(k[0]), int(k[1]))] = float(v)
            return out
        if isinstance(obj, list):
            return {(int(r[0]), int(r[1])): float(r[2]) for r in obj}
        return obj

    tqe = _norm_pair_dict(dp.get("two_q_gate_error", 0.01))
    if isinstance(tqe, dict):
        tqe.update({(int(k[1]), int(k[0])): v for (k, v) in list(tqe.items())})
    cs = _norm_pair_dict(topo.get("crosstalk_strength"))
    if isinstance(cs, dict):
        cs.update({(int(k[1]), int(k[0])): v for (k, v) in list(cs.items())})
    return NoiseConfig(
        t1_times=dp["t1_times"],
        t2_times=dp["t2_times"],
        freq_ghz=dp["freq_ghz"],
        single_q_gate_error=dp.get("single_q_gate_error", 0.001),
        two_q_gate_error=tqe,
        coupling_map=coupling_map,
        readout_error=dp["readout_error"],
        shots=dp.get("shots", 1024),
        crosstalk_strength=cs,
    )


def make_env(dag: CircuitDAG, coupling_map: List[Tuple[int, int]],
             initial_mapping: Optional[Tuple[int, ...]] = None) -> PureRoutingEnv:
    n_phys = max(max(e) for e in coupling_map) + 1
    dist = raw_hop_distance(coupling_map, n_phys)
    return PureRoutingEnv(dag=dag, coupling_map=coupling_map, dist=dist,
                          mapping=initial_mapping)


def run_episode(env: PureRoutingEnv, policy_fn,
                max_steps: int = 5000) -> Tuple[int, bool, float]:
    """跑一个 episode。policy_fn(env) -> 动作索引。返回 (swaps, 成功, 墙钟秒)。"""
    t0 = time.perf_counter()
    steps = 0
    while not env.is_terminal():
        if steps >= max_steps:
            return steps, False, time.perf_counter() - t0
        e = policy_fn(env)
        env.step(e)
        steps += 1
    return steps, True, time.perf_counter() - t0


def random_policy(rng: np.random.Generator):
    def _p(env: PureRoutingEnv) -> int:
        legal = env.legal_actions()
        return int(legal[rng.integers(0, len(legal))])
    return _p


def greedy_dist_policy(env: PureRoutingEnv) -> int:
    """纯距离贪心：argmin Σ_{ready} dist(虚拟换位后)。无 decay / ext。"""
    ready = env.ready_2q()
    pairs = [(env.mapping[env.dag.gates[g].qubits[0]],
              env.mapping[env.dag.gates[g].qubits[1]]) for g in ready]
    dist = env.dist
    inv = env._inv
    best_e, best_d = None, float("inf")
    for i, (p, q) in enumerate(env.coupling_map):
        lp, lq = inv[p], inv[q]
        if lp == -1 and lq == -1:
            continue
        d = 0.0
        for (a, b) in pairs:
            na = q if a == p else (p if a == q else a)
            nb = q if b == p else (p if b == q else b)
            d += float(dist[na, nb])
        if d < best_d:
            best_d, best_e = d, i
    return best_e


def sabre_num_swaps(circuit, coupling_map: List[Tuple[int, int]],
                    trials: int = 20, seed: int = 0,
                    initial_layout: Optional[list] = None) -> Tuple[int, float, Optional[list]]:
    """Qiskit SabreSwap 外部基线。返回 (num_swaps, wall_ms, initial_layout)。"""
    from qiskit import QuantumCircuit
    from qiskit.transpiler import CouplingMap, Layout, PassManager
    from qiskit.transpiler.passes import SabreSwap as QiskitSabreSwap, SetLayout

    cm = CouplingMap(list(coupling_map))
    passes = []
    if initial_layout is not None:
        qc0 = QuantumCircuit(cm.size())
        passes.append(SetLayout(Layout.from_intlist(list(initial_layout), qc0.qregs[0])))
    passes.append(QiskitSabreSwap(coupling_map=cm, heuristic="decay", trials=trials,
                                  seed=seed))
    pm = PassManager(passes)
    t0 = time.perf_counter()
    phys = pm.run(circuit)
    wall = (time.perf_counter() - t0) * 1000.0
    n_swaps = sum(1 for inst, qargs, cargs in phys.data if inst.name == "swap")
    return n_swaps, wall, None
