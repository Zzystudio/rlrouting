import numpy as np
import pytest
from itertools import combinations

from qiskit import QuantumCircuit, transpile

from sim.sim import NoiseConfig
from sim.trajectory_sim import (
    TrajectorySimulator,
    TrajectoryResult,
    trajectory_circuit_fidelity,
    _reduce_phys_circuit_for_fidelity,
)
from utils.metrics import state_fidelity


def _full_config(n, cm):
    t1 = [30.0 + i for i in range(n)]
    t2 = [60.0 + i for i in range(n)]  # t2 <= 2*t1
    freq = [5.0] * n
    sqe = [0.001 + 0.0001 * i for i in range(n)]
    ro = [0.01 + 0.001 * i for i in range(n)]
    tqe = {(a, b): 0.01 + 0.001 * (a + b) for a, b in cm}
    tqe.update({(b, a): v for (a, b), v in list(tqe.items())})
    cts = {(a, b): 0.001 + 0.0001 * (a + b) for a, b in cm}
    cts.update({(b, a): v for (a, b), v in list(cts.items())})
    return NoiseConfig(
        t1_times=t1, t2_times=t2, freq_ghz=freq,
        single_q_gate_error=sqe, two_q_gate_error=tqe, coupling_map=cm,
        readout_error=ro, crosstalk_strength=cts,
        single_gate_time=0.1, two_gate_time=0.3, idle_time=0.1, shots=1024,
    )


def _subset_circuit(n, used, gate_seq):
    """在 n 比特电路上、只用到 used 中的比特，按 gate_seq 施加门。

    gate_seq 元素 = (name, (qubit_idx_in_used,...), param=None)
    """
    qc = QuantumCircuit(n)
    for name, qs, *rest in gate_seq:
        param = rest[0] if rest else None
        if param is None:
            getattr(qc, name)(*[used[i] for i in qs])
        else:
            getattr(qc, name)(param, *[used[i] for i in qs])
    return qc


def _reduced_circuit(k, gate_seq):
    qc = QuantumCircuit(k)
    for name, qs, *rest in gate_seq:
        param = rest[0] if rest else None
        if param is None:
            getattr(qc, name)(*qs)
        else:
            getattr(qc, name)(param, *qs)
    return qc


# ----------------------------------------------------------------------- #
# 物理比特子集截断：对保真度精确等价（同种子）
# ----------------------------------------------------------------------- #
def test_truncation_equals_full_simulation():
    """被用到的物理比特子集截断后，保真度必须与「等价约化问题」完全一致。

    数学依据：|0> 是热噪声（振幅/相位阻尼）的不动点，未用比特恒为 |0>，
    保真度贡献因子恒为 1；退极化与相干 ZZ 串扰只触及被作用比特。故只对子集
    模拟并索引重标号，结果与全比特模拟逐位相等。
    """
    n = 8
    full_cm = [(a, b) for a, b in combinations(range(n), 2)]  # 全连接避免转译插 SWAP
    used = [0, 2, 5]
    k = len(used)
    red_cm = [(a, b) for a, b in combinations(range(k), 2)]

    full_cfg = _full_config(n, full_cm)
    # 手搓「仅含 used 比特」的约化配置（参数取自 used 对应索引）
    red_cfg = NoiseConfig(
        t1_times=[full_cfg.t1_times[i] for i in used],
        t2_times=[full_cfg.t2_times[i] for i in used],
        freq_ghz=[full_cfg.freq_ghz[i] for i in used],
        single_q_gate_error=[full_cfg.single_q_gate_error[i] for i in used],
        two_q_gate_error={(a, b): full_cfg.two_q_gate_error[(used[a], used[b])]
                          for a, b in red_cm},
        coupling_map=red_cm,
        readout_error=[full_cfg.readout_error[i] for i in used],
        crosstalk_strength={(a, b): full_cfg.crosstalk_strength[(used[a], used[b])]
                             for a, b in red_cm},
        single_gate_time=0.1, two_gate_time=0.3, idle_time=0.1, shots=1024,
    )

    gate_seq = [("h", (0,)), ("cx", (0, 1)), ("cx", (1, 2)),
                ("rz", (2,), 0.5), ("cx", (0, 2))]
    phys8 = _subset_circuit(n, used, gate_seq)
    rc3 = _reduced_circuit(k, gate_seq)

    for scheduled in (False, True):
        f_red = trajectory_circuit_fidelity(rc3, red_cfg, num_trajectories=64,
                                            seed=123, scheduled=scheduled)
        f_trunc = trajectory_circuit_fidelity(phys8, full_cfg, num_trajectories=64,
                                              seed=123, scheduled=scheduled)
        assert f_trunc == pytest.approx(f_red, abs=1e-12), \
            f"scheduled={scheduled}: 截断 {f_trunc} != 约化 {f_red}"


def test_truncation_noop_when_all_used():
    """所有比特均被使用时，截断应为无操作（结果不变）。"""
    n = 5
    cm = [(a, b) for a, b in combinations(range(n), 2)]
    cfg = _full_config(n, cm)
    qc = QuantumCircuit(n)
    qc.h(0); qc.cx(0, 1); qc.cx(2, 3); qc.cx(3, 4); qc.rz(0.3, 4)
    rc, rcfg = _reduce_phys_circuit_for_fidelity(qc, cfg)
    assert rc.num_qubits == n
    assert rcfg is cfg



def _config(n=3, sqe=0.001, tqe=0.01, ro=0.02, t1=50.0, t2=70.0,
            coupling=None, crosstalk=None, shots=1024, idle_time=0.1,
            single_time=0.1):
    if coupling is None:
        coupling = [(i, i + 1) for i in range(n - 1)]
    readout = [ro] * n if ro is not None else None
    return NoiseConfig(
        t1_times=[t1] * n, t2_times=[t2] * n, freq_ghz=[5.0] * n,
        single_q_gate_error=sqe, two_q_gate_error=tqe,
        coupling_map=coupling, crosstalk_strength=crosstalk,
        readout_error=readout,
        single_gate_time=single_time, two_gate_time=0.3, idle_time=idle_time,
        shots=shots,
    )


def _transpiled(qc, n=None):
    if n is None:
        n = qc.num_qubits
    coupling = [[i, i + 1] for i in range(n - 1)]
    return transpile(qc, coupling_map=coupling,
                     basis_gates=["rz", "sx", "x", "cx", "id"],
                     optimization_level=1)


def _line_circuit(n, depth=1):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


# ----------------------------------------------------------------------- #
# 基础：无噪声时应与精确状态向量一致
# ----------------------------------------------------------------------- #
def test_noiseless_matches_statevector():
    cfg = _config(sqe=0.0, tqe=0.0, ro=None, t1=1e6, t2=2e6)
    ts = TrajectorySimulator(cfg, num_trajectories=16, seed=1)
    qc = _transpiled(_line_circuit(3))
    result = ts.run_trajectories(qc, skip_transpile=True)
    assert result.statevectors.shape[0] == 16
    for t in range(16):
        assert np.allclose(result.statevectors[t], result.statevectors[0])
    # L2 范数应为 1（GHZ 振幅为 1/sqrt(2)，L1 为 sqrt(2)）
    assert np.isclose(np.linalg.norm(result.statevectors[0]), 1.0, atol=1e-9)
    amp = np.abs(result.statevectors[0]) ** 2
    assert np.isclose(amp[0] + amp[7], 1.0, atol=1e-9)  # GHZ: 000 + 111


def test_noiseless_fidelity_one():
    cfg = _config(sqe=0.0, tqe=0.0, ro=None, t1=1e6, t2=1e6)
    sim = TrajectorySimulator(cfg, num_trajectories=8, seed=1)
    qc = _transpiled(_line_circuit(3))
    res = sim.run_trajectories(qc, skip_transpile=True)
    assert res.fidelity(sim.ideal_statevector(qc)) == pytest.approx(1.0, abs=1e-9)


def test_counts_noiseless_ghz():
    """无噪声 GHZ 测量只应集中在 |000> 与 |111>。"""
    cfg = _config(n=3, sqe=0.0, tqe=0.0, ro=None, t1=1e6, t2=1e6, shots=512)
    sim = TrajectorySimulator(cfg, seed=1)
    qc = _transpiled(_line_circuit(3))
    qc.measure_all()
    counts = sim.run(qc, shots=512, skip_transpile=True)
    assert sum(counts.values()) == 512
    ideal_frac = sum(counts.get(k, 0) for k in ("000", "111")) / 512
    assert ideal_frac > 0.98


# ----------------------------------------------------------------------- #
# 密度矩阵接口（.data）与统计口径一致性
# ----------------------------------------------------------------------- #
def test_data_consistency():
    """rho trace == 1，且 fidelity(ideal) == state_fidelity(ideal, rho)。"""
    cfg = _config()
    sim = TrajectorySimulator(cfg, num_trajectories=512, seed=1)
    qc = _transpiled(_line_circuit(3))
    res = sim.run_trajectories(qc, skip_transpile=True)
    rho = res.data
    assert rho.shape == (8, 8)
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-9)
    ideal = sim.ideal_statevector(qc)
    assert res.fidelity(ideal) == pytest.approx(
        state_fidelity(ideal, rho), abs=1e-9)


def test_large_n_data_raises_but_fidelity_ok():
    """大 n 时 .data 抛错提示改 fidelity，fidelity 本身不受限。"""
    n = 16
    cfg = _config(n=n, sqe=0.0, tqe=0.0, ro=None)
    sim = TrajectorySimulator(cfg, num_trajectories=8, seed=1)
    qc = _transpiled(_line_circuit(n))
    res = sim.run_trajectories(qc, skip_transpile=True)
    with pytest.raises(RuntimeError):
        _ = res.data
    ideal = sim.ideal_statevector(qc)
    assert 0.0 <= res.fidelity(ideal) <= 1.0


# ----------------------------------------------------------------------- #
# 与 Aer density_matrix 对拍
# ----------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [3, 4])
def test_trajectory_matches_density_matrix(n):
    """同一噪声语义下，轨迹保真度应与 Aer density_matrix 求解一致。

    reference 噪声模型按与轨迹模拟器相同的语义构造：
    - 单比特门：thermal + depolarizing 组合
    - CX：per-edge 退极化 + 默认串扰 ZZ（记作 depol ∘ ZZ 合成）
    两者是同一通道的两种求解器，统计误差应 < 0.02。
    """
    from qiskit_aer.noise import (
        NoiseModel, thermal_relaxation_error,
        depolarizing_error, pauli_error,
    )
    from qiskit_aer import AerSimulator

    coupling = [(i, i + 1) for i in range(n - 1)]
    cfg = _config(n=n, sqe=0.001, tqe=0.01, ro=0.02, t1=50.0, t2=70.0)
    sim = TrajectorySimulator(cfg, num_trajectories=8192, seed=1)

    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    tr = _transpiled(qc, n)
    ideal = sim.ideal_statevector(tr)

    # ---- 构造语义一致的 reference NoiseModel ----
    nm = NoiseModel()
    single_err = depolarizing_error(0.001, 1).compose(
        thermal_relaxation_error(50.0, 70.0, 0.1))
    nm.add_all_qubit_quantum_error(single_err, ["rz", "sx", "x", "h"])
    nm.add_all_qubit_quantum_error(
        thermal_relaxation_error(50.0, 70.0, 0.1), ["id"])
    for (a, b) in coupling + [(b, a) for (a, b) in coupling]:
        # depol(0.01) ∘ ZZ(0.1*0.01)。按 sim.py default 串扰强度 0.001
        edge_err = depolarizing_error(0.01, 2).compose(
            pauli_error([("ZZ", 0.001), ("II", 0.999)]))
        nm.add_quantum_error(edge_err, ["cx"], [a, b])
    cs = AerSimulator(noise_model=nm, method="density_matrix",
                      basis_gates=nm.basis_gates,
                      coupling_map=[list(e) for e in coupling])
    circ = tr.copy()
    circ.save_density_matrix()
    rho = np.asarray(cs.run(circ, shots=1).result().data()["density_matrix"])
    fid_dm = state_fidelity(ideal, rho)

    fid_traj = sim.run_trajectories(tr, skip_transpile=True).fidelity(ideal)
    assert abs(fid_traj - fid_dm) < 0.02, \
        f"轨迹 {fid_traj:.4f} vs 参考密度矩阵 {fid_dm:.4f}"


def test_t1_decay():
    """赋予 |1>，久置应按 1-exp(-t/T1) 衰到 |0>。"""
    n = 1
    # idle_time=10, T1=10 -> p_reset = 1 - exp(-1) ≈ 0.632
    cfg = _config(n=n, sqe=0.0, tqe=0.0, ro=None, t1=10.0, t2=20.0,
                  idle_time=10.0, shots=4096)
    sim = TrajectorySimulator(cfg, num_trajectories=4096, seed=1)
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.id(0)
    counts = sim.run(qc, shots=4096, skip_transpile=True)
    p0 = counts.get("0", 0) / 4096
    assert 0.55 < p0 < 0.72, f"P(|0>)={p0:.3f}"


def test_readout_error():
    """读出错误应翻转部分结果。"""
    cfg = _config(n=1, sqe=0.0, tqe=0.0, ro=0.0, t1=1e6, t2=1e6, shots=1024)
    sim_ok = TrajectorySimulator(cfg, num_trajectories=8, seed=1)
    qc = QuantumCircuit(1)
    qc.id(0)
    counts_ok = sim_ok.run(qc, shots=1024, skip_transpile=True)
    assert counts_ok.get("0", 0) == 1024

    cfgro = _config(n=1, sqe=0.0, tqe=0.0, ro=0.5, t1=1e6, t2=1e6, shots=1024)
    sim_ro = TrajectorySimulator(cfgro, num_trajectories=8, seed=1)
    counts_ro = sim_ro.run(qc, shots=1024, skip_transpile=True)
    frac1 = counts_ro.get("1", 0) / 1024
    assert 0.35 < frac1 < 0.65, f"翻转率 {frac1:.3f}"


def test_seed_reproducible():
    cfg = _config(n=3)
    a = TrajectorySimulator(cfg, num_trajectories=32, seed=7) \
        .run_trajectories(_transpiled(_line_circuit(3)), skip_transpile=True)
    b = TrajectorySimulator(cfg, num_trajectories=32, seed=7) \
        .run_trajectories(_transpiled(_line_circuit(3)), skip_transpile=True)
    assert np.allclose(a.statevectors, b.statevectors)


def test_batch_api():
    cfg = _config(n=3, shots=64)
    sim = TrajectorySimulator(cfg, num_trajectories=16, seed=1)
    qclist = [_transpiled(_line_circuit(3)), _transpiled(_line_circuit(3))]
    counts = sim.run_batch([c for c in qclist], shots=64, skip_transpile=True)
    assert all(sum(c.values()) == 64 for c in counts)
    res_list = sim.run_batch_statevector(qclist, skip_transpile=True)
    assert len(res_list) == 2 and all(isinstance(r, TrajectoryResult)
                                      for r in res_list)


def test_validate_t2_twice_t1():
    cfg = _config(t1=10.0, t2=30.0)  # 30 > 2*10
    with pytest.raises(ValueError):
        TrajectorySimulator(cfg)


# ----------------------------------------------------------------------- #
# 位序（bitorder）回归：非 GHZ 电路须与 qiskit/Aer 完全一致
# ----------------------------------------------------------------------- #
def test_bitorder_matches_qiskit_statevector():
    """x(q0) 应落在 index 1（little-endian），而非 8（big-endian）。"""
    from qiskit_aer import AerSimulator

    cfg = _config(n=4, sqe=0.0, tqe=0.0, ro=None, t1=1e6, t2=2e6)
    ts = TrajectorySimulator(cfg, num_trajectories=8, seed=1)

    def aer(qc):
        cc = qc.copy()
        cc.save_statevector()
        return np.asarray(AerSimulator(method="statevector").run(cc)
                          .result().data()["statevector"])

    cases = []
    qc = QuantumCircuit(4); qc.x(0); cases.append(("x(q0)", qc))
    qc = QuantumCircuit(4); qc.x(3); cases.append(("x(q3)", qc))
    qc = QuantumCircuit(4); qc.h(0); qc.cx(0, 1); cases.append(("bell(0,1)", qc))
    QC = QuantumCircuit(4); QC.h(2); QC.cx(2, 3); cases.append(("bell(2,3)", QC))
    rng = np.random.default_rng(7)
    qc = QuantumCircuit(4)
    for _ in range(5):
        qs = rng.choice(4, 2, replace=False)
        if rng.random() < 0.5:
            qc.h(int(qs[0]))
        else:
            qc.rx(rng.uniform(0, np.pi), int(qs[0]))
        qc.cx(int(qs[0]), int(qs[1]))
    cases.append(("random", qc))

    for label, qc in cases:
        qc_t = _transpiled(qc, n=4)
        tr_sv = ts.ideal_statevector(qc_t)
        aer_sv = aer(qc_t)
        assert np.max(np.abs(tr_sv - aer_sv)) < 1e-12, (
            f"{label}: 状态向量与 qiskit/Aer 不一致")


def test_bitorder_counts_xq2():
    """counts 位序：x(q2) 无噪声应只在 '0100' 聚集（qiskit little-endian）。"""
    cfg = _config(n=4, sqe=0.0, tqe=0.0, ro=None, t1=1e6, t2=2e6, shots=20000)
    ts = TrajectorySimulator(cfg, num_trajectories=8, seed=1)
    qc = QuantumCircuit(4, 4)
    qc.x(2)
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    cnt = ts.run(qc, shots=20000)
    assert cnt.get("0100", 0) > 0.98 * 20000
    assert sum(cnt.values()) == 20000