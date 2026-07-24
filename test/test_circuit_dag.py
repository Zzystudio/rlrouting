# ============================================================================
# test_circuit_dag.py
# 验证电路 DAG 构建与噪声特征编码的基本正确性。
# ============================================================================

import numpy as np
import pytest

from routing.graph.circuit_dag import CircuitDAG, build_routing_graph
from routing.graph.features import HardwareFeatures, NODE_FEATURE_DIM, EDGE_FEATURE_DIM


def _make_config(n=4):
    from sim.sim import NoiseConfig
    coupling = [(i, i + 1) for i in range(n - 1)]
    return NoiseConfig(
        t1_times=[50.0] * n,
        t2_times=[70.0] * n,
        freq_ghz=[5.0] * n,
        single_q_gate_error=0.001,
        two_q_gate_error=0.01,
        coupling_map=coupling,
        readout_error=[0.02] * n,
    )


def _make_circuit():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return qc


def test_dag_parse():
    qc = _make_circuit()
    dag = CircuitDAG.from_circuit(qc)
    assert dag.num_gates == 3
    assert dag.num_logical_qubits == 3
    # cx(0,1) 依赖 h(0)
    cx0 = dag.gates[1]
    assert 0 in cx0.predecessors


def test_build_graph_shapes():
    qc = _make_circuit()
    config = _make_config(3)
    hw = HardwareFeatures.from_noise_config(config)
    dag = CircuitDAG.from_circuit(qc)
    mapping = [0, 1, 2]
    data = build_routing_graph(dag, mapping, hw, list(config.coupling_map))
    N = dag.num_gates + dag.num_logical_qubits
    assert data.node_features.shape == (N, NODE_FEATURE_DIM)
    assert data.edge_index.shape[0] == 2
    if data.edge_attr.size > 0:
        assert data.edge_attr.shape[1] == EDGE_FEATURE_DIM
    assert len(data.coupling_edges) == len(config.coupling_map)


def test_crosstalk_in_couples_edges():
    qc = _make_circuit()
    config = _make_config(3)
    hw = HardwareFeatures.from_noise_config(config)
    dag = CircuitDAG.from_circuit(qc)
    data = build_routing_graph(dag, [0, 1, 2], hw, list(config.coupling_map))
    # 存在 couples 边（type 索引 1 列 > 0）
    couples_col = data.edge_attr[:, 1]
    assert np.any(couples_col > 0)
