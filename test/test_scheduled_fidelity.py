import numpy as np
import pytest

from qiskit import QuantumCircuit

from sim.sim import NoiseConfig
from sim.trajectory_sim import TrajectorySimulator, schedule_phys_circuit


def _config(n=4, sqe=0.001, tqe=0.01, ro=None, t1=50.0, t2=70.0,
            coupling=None, crosstalk_strength=None, shots=1024, idle_time=0.1,
            single_time=0.1):
    if coupling is None:
        coupling = [(i, i + 1) for i in range(n - 1)]
    readout = [ro] * n if ro is not None else None
    return NoiseConfig(
        t1_times=[t1] * n, t2_times=[t2] * n, freq_ghz=[5.0] * n,
        single_q_gate_error=sqe, two_q_gate_error=tqe,
        coupling_map=coupling, crosstalk_strength=crosstalk_strength,
        readout_error=readout,
        single_gate_time=single_time, two_gate_time=0.3, idle_time=idle_time,
        shots=shots,
    )


def _build_waves(config, spec):
    """spec: list of (dw, [(phys_idx, qubits, is_2q), ...])"""
    return [(dw, gates) for dw, gates in spec]


def test_scheduled_parallel_reduces_idle_decoherence():
    """并行（同波不冲突的双比特门）空闲退相干更少 → 保真度高于串行。"""
    cfg = _config(n=4, sqe=0.001, tqe=0.01, ro=None,
                  t1=50.0, t2=70.0, crosstalk_strength={})
    ts = TrajectorySimulator(cfg, num_trajectories=192, seed=3)
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(2, 3)
    meas = qc.copy()
    meas.measure_all()
    serial = _build_waves(cfg, [
        (cfg.single_gate_time, [(0, (0,), False)]),
        (cfg.two_gate_time, [(1, (0, 1), True)]),
        (cfg.two_gate_time, [(2, (2, 3), True)]),
    ])
    parallel = _build_waves(cfg, [
        (cfg.single_gate_time, [(0, (0,), False)]),
        (cfg.two_gate_time, [(1, (0, 1), True), (2, (2, 3), True)]),
    ])
    f_serial = ts.fidelity_scheduled(meas, serial)
    f_par = ts.fidelity_scheduled(meas, parallel)
    assert f_par > f_serial


def test_dynamic_crosstalk_lowers_parallel_1hop():
    """同波内两对不相交双比特门（交叉对在耦合图相邻，1-hop）触发动态串扰
    → 并行保真度低于串行（串行无同时执行的双门，无动态串扰）。

    关闭退相干与门错误，使串扰成为唯一噪声差。前置 H/CX 使相关比特处于
    叠加态（ZZ 串扰非本征态），否则串扰不改变保真度。线形拓扑 (0,1)(1,2)(2,3)
    上 cx(0,1) 与 cx(2,3) 同波，交叉对 (1,2) 为耦合边 → 动态串扰。
    """
    cfg = _config(n=4, sqe=0.0, tqe=0.0, ro=None,
                  t1=1e6, t2=2e6,
                  crosstalk_strength={(0, 1): 0.1, (1, 2): 0.1, (2, 3): 0.1})
    ts = TrajectorySimulator(cfg, num_trajectories=256, seed=7)
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.h(3)
    qc.cx(0, 1)
    qc.cx(2, 3)
    meas = qc.copy()
    meas.measure_all()
    serial = _build_waves(cfg, [
        (cfg.single_gate_time, [(0, (0,), False)]),
        (cfg.single_gate_time, [(1, (1,), False)]),
        (cfg.single_gate_time, [(2, (2,), False)]),
        (cfg.single_gate_time, [(3, (3,), False)]),
        (cfg.two_gate_time, [(4, (0, 1), True)]),
        (cfg.two_gate_time, [(5, (2, 3), True)]),
    ])
    parallel = _build_waves(cfg, [
        (cfg.single_gate_time, [(0, (0,), False)]),
        (cfg.single_gate_time, [(1, (1,), False)]),
        (cfg.single_gate_time, [(2, (2,), False)]),
        (cfg.single_gate_time, [(3, (3,), False)]),
        (cfg.two_gate_time, [(4, (0, 1), True), (5, (2, 3), True)]),
    ])
    f_serial = ts.fidelity_scheduled(meas, serial)
    f_par = ts.fidelity_scheduled(meas, parallel)
    assert f_par < f_serial


def test_dynamic_crosstalk_absent_when_not_1hop():
    """两对不相交双比特门若交叉对不在耦合图相邻（非 1-hop），不触发动态串扰，
    并行保真度应高于串行（空闲退相干更少）。用更短 T2/更多轨迹放大空闲差、降噪。
    """
    n = 5
    coupling = [(i, i + 1) for i in range(n - 1)]
    cfg = _config(n=n, sqe=0.001, tqe=0.01, ro=None,
                  t1=20.0, t2=15.0, coupling=coupling,
                  crosstalk_strength={(i, i + 1): 0.05 for i in range(n - 1)})
    ts = TrajectorySimulator(cfg, num_trajectories=2000, seed=13)
    qc = QuantumCircuit(n)
    qc.h(0); qc.h(1); qc.h(3); qc.h(4)
    qc.cx(0, 1)
    qc.cx(3, 4)
    meas = qc.copy()
    meas.measure_all()
    serial = _build_waves(cfg, [
        (cfg.single_gate_time, [(0, (0,), False)]),
        (cfg.single_gate_time, [(1, (1,), False)]),
        (cfg.single_gate_time, [(2, (3,), False)]),
        (cfg.single_gate_time, [(3, (4,), False)]),
        (cfg.two_gate_time, [(4, (0, 1), True)]),
        (cfg.two_gate_time, [(5, (3, 4), True)]),
    ])
    parallel = _build_waves(cfg, [
        (cfg.single_gate_time, [(0, (0,), False)]),
        (cfg.single_gate_time, [(1, (1,), False)]),
        (cfg.single_gate_time, [(2, (3,), False)]),
        (cfg.single_gate_time, [(3, (4,), False)]),
        (cfg.two_gate_time, [(4, (0, 1), True), (5, (3, 4), True)]),
    ])
    f_serial = ts.fidelity_scheduled(meas, serial)
    f_par = ts.fidelity_scheduled(meas, parallel)
    assert f_par > f_serial


def test_idle_decoherence_lowers_fidelity():
    """门执行前插入一段空闲（其他比特退相干）→ 保真度下降，验证空闲退相干建模。

    注意：须先制造叠加态（H+CX → Bell），否则 |00> 等计算基态对退相干免疫，
    空闲噪声测不出差异。用 5.0us 空闲窗口使效应明显大于轨迹采样方差。
    """
    cfg = _config(n=4, sqe=0.001, tqe=0.01, ro=None,
                  t1=50.0, t2=70.0, crosstalk_strength={(0, 1): 0.05, (1, 2): 0.05})
    ts = TrajectorySimulator(cfg, num_trajectories=400, seed=11)
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    meas = qc.copy()
    meas.measure_all()
    no_idle = _build_waves(cfg, [(cfg.two_gate_time, [(0, (0, 1), True)])])
    with_idle = _build_waves(cfg, [
        (5.0, []),  # 空波：所有比特空闲 5.0us
        (cfg.two_gate_time, [(0, (0, 1), True)]),
    ])
    f_no = ts.fidelity_scheduled(meas, no_idle)
    f_idle = ts.fidelity_scheduled(meas, with_idle)
    assert f_no > f_idle


def test_scheduled_batch_matches_scalar():
    """向量化 evolve_scheduled_batch 与标量版本在采样噪声内一致。"""
    cfg = _config(n=4, sqe=0.002, tqe=0.02,
                  crosstalk_strength={(0, 1): 0.05, (1, 2): 0.05})
    ts = TrajectorySimulator(cfg, num_trajectories=400, seed=5)
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    meas = qc.copy()
    meas.measure_all()
    waves = schedule_phys_circuit(meas, cfg.single_gate_time, cfg.two_gate_time)
    svs_batch = ts.run_trajectories_scheduled(
        meas, waves, num_trajectories=400, skip_transpile=True).statevectors
    svs_scalar = np.array(
        [ts.evolve_scheduled(meas, waves, apply_noise=True) for _ in range(400)])
    ideal = ts._evolve(meas, apply_noise=False)
    fb = float(np.mean([abs(np.vdot(ideal, s)) ** 2 for s in svs_batch]))
    fs = float(np.mean([abs(np.vdot(ideal, s)) ** 2 for s in svs_scalar]))
    assert abs(fb - fs) < 0.05


def test_coherent_crosstalk_smooth_and_monotonic():
    """相干 ZZ 串扰（exp(-iθ Z⊗Z)）的性质：

    1. 确定性：同一 seed 两次计算完全一致（不再像旧硬 Z 翻转那样逐轨迹抛骰子，
       命中叠加态即正交归零）；
    2. 无正交归零：对叠加态施加单个 ZZ 旋转，保真度严格落在 (0,1)（不会因
       命中而整条轨迹归零）；
    3. 单调：θ 越大，平行调度的动态串扰保真度越低。
    """
    cfg = _config(n=4, sqe=0.0, tqe=0.0, ro=None,
                  t1=1e6, t2=2e6,
                  crosstalk_strength={(0, 1): 0.05, (1, 2): 0.05, (2, 3): 0.05})
    qc = QuantumCircuit(4)
    qc.h(0); qc.h(1); qc.h(2); qc.h(3)
    qc.cx(0, 1)
    qc.cx(2, 3)
    meas = qc.copy()
    meas.measure_all()
    parallel = [(cfg.single_gate_time, [(0, (0,), False)]),
                (cfg.single_gate_time, [(1, (1,), False)]),
                (cfg.single_gate_time, [(2, (2,), False)]),
                (cfg.single_gate_time, [(3, (3,), False)]),
                (cfg.two_gate_time, [(4, (0, 1), True), (5, (2, 3), True)])]

    ts_a = TrajectorySimulator(cfg, num_trajectories=64, seed=7)
    ts_b = TrajectorySimulator(cfg, num_trajectories=64, seed=7)
    fa = ts_a.fidelity_scheduled(meas, parallel)
    fb = ts_b.fidelity_scheduled(meas, parallel)
    assert fa == fb  # 确定性
    assert 0.0 < fa < 1.0  # 无正交归零

    # θ 越大越低：把强度放大 4 倍
    cfg_hi = _config(n=4, sqe=0.0, tqe=0.0, ro=None,
                     t1=1e6, t2=2e6,
                     crosstalk_strength={(0, 1): 0.20, (1, 2): 0.20, (2, 3): 0.20})
    ts_hi = TrajectorySimulator(cfg_hi, num_trajectories=64, seed=7)
    f_hi = ts_hi.fidelity_scheduled(meas, parallel)
    assert f_hi < fa


def test_schedule_phys_circuit_waves():
    """schedule_phys_circuit 对已有电路做贪心波次编排，输出与 evolve_scheduled 兼容。"""
    cfg = _config(n=4, crosstalk_strength={})
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)
    meas = qc.copy()
    meas.measure_all()
    waves = schedule_phys_circuit(meas, cfg.single_gate_time, cfg.two_gate_time)
    # 前两门不冲突 → 同一波；第三门与前两波共享比特 → 新波
    assert len(waves) == 2
    # 校验 phys_idx 落在有效区间
    n_instr = len(meas.data)
    for _, gates in waves:
        for (idx, _qs, _is2) in gates:
            assert 0 <= idx < n_instr


def test_env_get_schedule_waves():
    """env 在调度模式下应产出与 _phys_circuit 对齐的调度波形。"""
    from routing.graph.circuit_dag import CircuitDAG
    from routing.graph.features import HardwareFeatures
    from routing.rl.env import RoutingEnv
    from utils.data_gen import random_circuit

    n = 5
    coupling = [(i, i + 1) for i in range(n - 1)]
    cfg = _config(n=n, crosstalk_strength={})
    qc = random_circuit(n, 5, seed=4)
    dag = CircuitDAG.from_circuit(qc)
    hw = HardwareFeatures.from_noise_config(cfg)
    env = RoutingEnv(dag, hw, coupling, reward_mode="routing",
                     seed=9, use_scheduler=True)
    env.reset()
    done = False
    steps = 0
    while not done and steps < 500:
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
        steps += 1
    waves = env.get_schedule_waves()
    assert waves is not None
    n_instr = len(env._phys_circuit.data)
    for _, gates in waves:
        for (idx, _qs, _is2) in gates:
            assert 0 <= idx < n_instr
