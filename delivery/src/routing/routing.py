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

import time
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
    n_phys = max(max(e) for e in coupling_map) + 1

    mapping = list(range(n_phys))

    phys = QuantumCircuit(n_phys, circuit.num_clbits)
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


def sabre_route(
    circuit,
    config,
    heuristic: str = "decay",
    swap_trials: int = 20,
    seed: int = 0,
):
    """使用 Qiskit SabreSwap 做路由，返回 (物理电路, 信息字典)。"""
    from qiskit import QuantumCircuit
    from qiskit.transpiler import PassManager, CouplingMap
    from qiskit.transpiler.passes import SabreSwap as QiskitSabreSwap

    coupling_list = list(config.coupling_map)
    cm = CouplingMap(coupling_list)

    qc = circuit
    if circuit.num_qubits < cm.size():
        qc = QuantumCircuit(cm.size(), circuit.num_clbits)
        qc.compose(circuit, inplace=True)

    sabre = QiskitSabreSwap(coupling_map=cm, heuristic=heuristic, trials=swap_trials, seed=seed)
    pm = PassManager(sabre)

    t0 = time.perf_counter()
    phys = pm.run(qc)
    wall_time_ms = (time.perf_counter() - t0) * 1000

    num_swaps = sum(1 for inst, qargs, cargs in phys.data if inst.name == "swap")

    # 由最终布局 + 逆推路由 SWAP 还原「初始布局」(logical->physical 列表)。
    # SabreSwap 只在 property_set 中保留 final_layout，无 initial 键，故需重建。
    final_layout = pm.property_set.get("final_layout")
    initial_layout = None
    if final_layout is not None:
        vb = final_layout.get_virtual_bits()
        layout = {q._index if hasattr(q, "_index") else q: p for q, p in vb.items()}
        for inst, qargs, cargs in reversed(list(phys.data)):
            if inst.name == "swap":
                i, j = qargs[0]._index, qargs[1]._index
                layout[i], layout[j] = layout[j], layout[i]
        initial_layout = [layout[i] for i in range(phys.num_qubits)]

    info = {
        "num_swaps": num_swaps,
        "num_physical_gates": phys.num_nonlocal_gates
        if hasattr(phys, "num_nonlocal_gates") else None,
        "initial_layout": initial_layout,
        "wall_time_ms": wall_time_ms,
    }
    return phys, info


def route_circuit(circuit, config, **kwargs):
    """路由入口。"""
    return greedy_route(circuit, config, **kwargs)
