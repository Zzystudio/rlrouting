# ============================================================================
# routing.py
# 顶层路由接口：给定逻辑电路与噪声配置，输出映射到物理拓扑后的电路。
#
# 工作流程
# --------
# 1. 解析电路为 DAG；
# 2. 贪心 SWAP 路由：对不相邻的双比特门，沿耦合图插入 SWAP 直至相邻；
# 3. 返回物理电路与统计信息（SWAP 数、初始映射）。
# ============================================================================

from __future__ import annotations

from typing import List, Optional, Tuple

from .graph.circuit_dag import CircuitDAG
from .graph.features import HardwareFeatures


def _append_gate(phys, g, phys_qubits):
    """将逻辑门按当前物理映射（物理比特列表）追加到物理电路。"""
    if g.is_measure:
        phys.measure(phys_qubits[0], g.clbits[0])
    else:
        phys.append(g.operation, phys_qubits)


def greedy_route(
    circuit,
    config,
    seed: int = 0,
):
    """贪心 SWAP 路由，返回 (物理电路, 信息字典)。"""
    from qiskit import QuantumCircuit

    dag = CircuitDAG.from_circuit(circuit)
    hw = HardwareFeatures.from_noise_config(config)
    coupling_map = list(config.coupling_map)
    n = circuit.num_qubits

    mapping = list(range(n))

    phys = QuantumCircuit(n, circuit.num_clbits)
    inv = {p: l for l, p in enumerate(mapping)}

    def do_swap(p: int, q: int):
        phys.swap(p, q)
        l1, l2 = inv[p], inv[q]
        mapping[l1], mapping[l2] = mapping[l2], mapping[l1]
        inv[p], inv[q] = inv[q], inv[p]

    num_swaps = 0
    for g in dag.gates:
        k = len(g.qubits)
        if k == 2 and g.is_two_qubit:
            qa, qb = g.qubits
            pa, pb = mapping[qa], mapping[qb]
            while hw.adj[pa, pb] == 0:
                # 选择能使两比特距离减小最多的耦合边做 SWAP
                best_edge, best_dist = None, float("inf")
                for (u, v) in coupling_map:
                    for (su, sv) in ((u, v), (v, u)):
                        tmp_l1, tmp_l2 = inv[su], inv[sv]
                        tmp_map = mapping.copy()
                        tmp_map[tmp_l1], tmp_map[tmp_l2] = tmp_map[tmp_l2], tmp_map[tmp_l1]
                        npa, npb = tmp_map[qa], tmp_map[qb]
                        d = hw.dist[npa, npb]
                        if d < best_dist:
                            best_dist = d
                            best_edge = (su, sv)
                if best_edge is None:
                    raise RuntimeError("耦合图不连通，无法路由")
                do_swap(*best_edge)
                num_swaps += 1
                pa, pb = mapping[qa], mapping[qb]
            _append_gate(phys, g, [pa, pb])
        elif k == 1:
            _append_gate(phys, g, [mapping[g.qubits[0]]])
        else:
            # 多于 2 比特的门（如 ccx）：初版按当前映射直接放置（简化）
            _append_gate(phys, g, [mapping[q] for q in g.qubits])

    info = {
        "initial_layout": list(mapping),
        "num_swaps": num_swaps,
        "num_physical_gates": phys.num_nonlocal_gates
        if hasattr(phys, "num_nonlocal_gates") else None,
    }
    return phys, info


def route_circuit(circuit, config, **kwargs):
    """路由入口。"""
    return greedy_route(circuit, config, **kwargs)
