# ============================================================================
# data_gen.py
# 训练数据生成：随机电路 + 随机初始映射 -> 在噪声模拟器上求真实保真度，
# 作为 Multi-GNN 保真度预测器的监督标签。
# ============================================================================

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np

from routing.graph.circuit_dag import CircuitDAG, build_routing_graph
from routing.graph.features import HardwareFeatures


@dataclass
class Sample:
    """单个训练样本：路由图 + 目标保真度。"""
    data: object          # RoutingGraphData
    fidelity: float


def random_circuit(num_qubits: int, depth: int, seed: int):
    """生成不含测量的随机电路（便于态矢量保真度计算）。"""
    from qiskit.circuit.random import random_circuit as _rc
    qc = _rc(num_qubits, depth, measure=False, seed=seed)
    return qc


def _ideal_statevector(circuit, skip_transpile: bool = False):
    """无噪声态矢量。"""
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    sim = AerSimulator(method="statevector")
    tc = circuit.copy() if skip_transpile else transpile(
        circuit, basis_gates=["rz", "sx", "x", "cx"])
    tc.save_statevector()
    result = sim.run(tc, shots=1).result()
    return np.asarray(result.data()["statevector"])


def physical_circuit_fidelity(
    logical_circuit, layout: List[int], config,
    simulator: Optional[object] = None,
) -> float:
    """给定逻辑电路与初始映射，转译后在噪声模拟器上求真实保真度。

    可复用外部 NoiseSimulator 实例避免重复构建噪声模型。
    """
    from qiskit import transpile
    from sim.sim import NoiseSimulator
    from utils.metrics import state_fidelity, counts_fidelity

    coupling_map = [list(e) for e in config.coupling_map]
    transpiled = transpile(
        logical_circuit,
        coupling_map=coupling_map,
        initial_layout=list(layout),
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=1,
    )
    ideal_sv = _ideal_statevector(transpiled, skip_transpile=True)
    if simulator is None:
        simulator = NoiseSimulator(config)
    try:
        noisy_dm = simulator.run_and_get_statevector(
            transpiled, skip_transpile=True)
    except Exception:
        counts = simulator.run(transpiled, shots=2048, skip_transpile=True)
        ideal_counts = _ideal_counts(transpiled)
        return counts_fidelity(ideal_counts, counts)
    return state_fidelity(ideal_sv, np.asarray(noisy_dm.data))


def _ideal_counts(circuit, skip_transpile: bool = False) -> dict:
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    sim = AerSimulator(method="statevector")
    tc = circuit if skip_transpile else transpile(
        circuit, basis_gates=["rz", "sx", "x", "cx"])
    tc.measure_all()
    result = sim.run(tc, shots=2048).result()
    return result.get_counts()


def random_layout(num_qubits: int, rng: random.Random) -> List[int]:
    """随机初始映射（排列）。"""
    perm = list(range(num_qubits))
    rng.shuffle(perm)
    return perm


def generate_dataset(
    config,
    num_circuits: int = 50,
    layouts_per_circuit: int = 4,
    depth: int = 6,
    seed: int = 0,
) -> List[Sample]:
    """生成 (路由图, 保真度) 训练样本列表。

    使用共享 NoiseSimulator 实例避免重复构建噪声模型。
    """
    from sim.sim import NoiseSimulator
    simulator = NoiseSimulator(config)
    return list(sample_stream(
        config, num_circuits, layouts_per_circuit, depth, seed,
        simulator=simulator,
    ))


def sample_stream(
    config,
    num_circuits: int = 50,
    layouts_per_circuit: int = 4,
    depth: int = 6,
    seed: int = 0,
    simulator: Optional[object] = None,
) -> Iterator[Sample]:
    """按需生成样本的迭代器（节省内存）。

    可传入共享的 NoiseSimulator 避免重复构建噪声模型。
    """
    from sim.sim import NoiseSimulator
    if simulator is None:
        simulator = NoiseSimulator(config)
    rng = random.Random(seed)
    hw = HardwareFeatures.from_noise_config(config)
    coupling_map = list(config.coupling_map)
    n = len(config.t1_times)
    for c in range(num_circuits):
        qc = random_circuit(n, depth, seed=seed * 1000 + c)
        dag = CircuitDAG.from_circuit(qc)
        for _ in range(layouts_per_circuit):
            layout = random_layout(n, rng)
            data = build_routing_graph(dag, layout, hw, coupling_map)
            fid = physical_circuit_fidelity(
                qc, layout, config, simulator=simulator)
            yield Sample(data=data, fidelity=float(fid))