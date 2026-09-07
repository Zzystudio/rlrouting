import numpy as np
import pytest

from routing.timing import (
    CircuitTiming, GreedyScheduler, schedule_events, schedule_routed_circuit,
    GATE_DURATION_TABLE, SWAP_DURATION_US,
)
from routing.graph.circuit_dag import CircuitDAG
from routing.graph.features import HardwareFeatures
from sim.sim import NoiseConfig
from qiskit import QuantumCircuit


def _simple_dag():
    """构造一个含 4 个 2Q 门、依赖链 a->b->c->d 的 DAG（强制串行），并行度应低。"""
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(1, 2)
    qc.cx(0, 1)
    return CircuitDAG.from_circuit(qc)


def _hw(n):
    coupling = [(i, i + 1) for i in range(n - 1)]
    config = NoiseConfig(
        t1_times=[50.0] * n, t2_times=[70.0] * n, freq_ghz=[5.0] * n,
        single_q_gate_error=0.001, two_q_gate_error=0.01,
        coupling_map=coupling, readout_error=[0.02] * n,
    )
    return HardwareFeatures.from_noise_config(config)


def test_event_scheduler_serial_chain_makespan():
    """依赖链 a->b->c->d：事件级 makespan ≈ Σ(dur)，并行度≈1。"""
    dag = _simple_dag()
    hw = _hw(3)
    mapping = list(range(3))
    timing = CircuitTiming.create(3)
    sched = GreedyScheduler()
    total_serial = 0.0
    while len(timing.gate_end) < dag.num_gates:
        ready_1q, ready_2q = [], []
        for g in dag.gates:
            if g.index in timing.gate_end:
                continue
            if all(p in timing.gate_end for p in g.predecessors):
                (ready_2q if g.is_two_qubit else ready_1q).append(g.index)
        placed, adv, xt = schedule_events(ready_1q, ready_2q, dag, mapping,
                                          timing, hw, sched)
        if not placed:
            break
        for (gidx, s, e, _) in placed:
            timing.gate_end[gidx] = e
            total_serial += (e - s)
    # 4 个 cx 串行（依赖链）=> makespan ≈ 4*0.3
    assert timing.total_time == pytest.approx(4 * 0.30, abs=1e-6)
    assert total_serial == pytest.approx(4 * 0.30, abs=1e-6)


def test_event_scheduler_parallel_lower_than_lockstep():
    """独立并行 2Q 门：事件级 makespan < 锁步（每轮均按最大门计）。"""
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cx(2, 3)
    dag = CircuitDAG.from_circuit(qc)
    hw = _hw(4)
    mapping = list(range(4))
    timing = CircuitTiming.create(4)
    sched = GreedyScheduler()
    ready_1q, ready_2q = [], []
    for g in dag.gates:
        if all(p in timing.gate_end for p in g.predecessors):
            (ready_2q if g.is_two_qubit else ready_1q).append(g.index)
    placed, adv, xt = schedule_events(ready_1q, ready_2q, dag, mapping,
                                      timing, hw, sched)
    # 两个独立 cx 并行 => makespan = 0.3（而非锁步的 0.3*轮数）
    assert timing.total_time == pytest.approx(0.30, abs=1e-6)
    assert len(placed) == 2


def test_swap_duration_decomposed():
    """SWAP 时长 = 0.9µs（3×CX）。"""
    assert SWAP_DURATION_US == pytest.approx(0.9)
    assert GATE_DURATION_TABLE["swap"] == pytest.approx(0.9)


def test_xtalk_soft_constraint_runs():
    """A3：zz>alpha 的并行门应被延后（验证不崩溃且返回有限值）。"""
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(1, 2)
    dag = CircuitDAG.from_circuit(qc)
    hw = _hw(3)
    hw.zz[0, 1] = 0.05
    hw.zz[1, 2] = 0.05
    hw.zz[1, 0] = 0.05
    hw.zz[2, 1] = 0.05
    mapping = list(range(3))
    timing = CircuitTiming.create(3)
    sched = GreedyScheduler()
    ready_1q, ready_2q = [], []
    for g in dag.gates:
        if all(p in timing.gate_end for p in g.predecessors):
            (ready_2q if g.is_two_qubit else ready_1q).append(g.index)
    _, _, _ = schedule_events(ready_1q, ready_2q, dag, mapping, timing, hw, sched, xtalk_alpha=0.0)
    t_no = timing.total_time
    timing2 = CircuitTiming.create(3)
    _, _, _ = schedule_events(ready_1q, ready_2q, dag, mapping, timing2, hw, sched, xtalk_alpha=0.01)
    t_yes = timing2.total_time
    assert np.isfinite(t_no) and np.isfinite(t_yes)


def test_schedule_routed_circuit_runs():
    qc = QuantumCircuit(5)
    for i in range(4):
        qc.cx(i, i + 1)
    qc.measure_all()
    dag = CircuitDAG.from_circuit(qc)
    hw = _hw(5)
    mk, xt, stats = schedule_routed_circuit(dag, hw)
    assert mk > 0
    assert "density" in stats
    assert "peak_parallel" in stats
    assert stats["makespan_us"] == pytest.approx(mk)


def test_crosstalk_1hop_detected():
    """1-hop（交叉比特对在某耦合边相邻）并行门应被记入串扰，crosstalk_events 正确累加。"""
    qc = QuantumCircuit(5)
    qc.cx(0, 1)
    qc.cx(2, 3)   # 与 (0,1) 的交叉对 (1,2) 为耦合边 → 1-hop 相邻
    dag = CircuitDAG.from_circuit(qc)
    hw = _hw(5)   # 线形拓扑 (i,i+1)，zz 默认 ~0.02
    mapping = list(range(5))
    timing = CircuitTiming.create(5)
    sched = GreedyScheduler()
    ready_1q, ready_2q = [], []
    for g in dag.gates:
        if all(p in timing.gate_end for p in g.predecessors):
            (ready_2q if g.is_two_qubit else ready_1q).append(g.index)
    placed, adv, xt = schedule_events(ready_1q, ready_2q, dag, mapping,
                                      timing, hw, sched)
    # 1-hop 串扰（交叉对 (1,2) 相邻）应被记入
    assert timing.crosstalk_events > 0
    assert xt > 0
    # crosstalk_events 与返回值 xt 一致
    assert timing.crosstalk_events == pytest.approx(xt)
