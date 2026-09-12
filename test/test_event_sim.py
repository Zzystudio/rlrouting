# ============================================================================
# test_event_sim.py
# 事件级模拟器（trajectory_sim_v2）验证 harness
#
# 精确参考：用 qiskit DensityMatrix 按「同一编译动作流」逐动作施加**精确噪声
# 通道**（qiskit Kraus：depolarizing_error / amplitude_damping_error ∘
# phase_damping_error，与 v2 MC 轨迹的通道定义逐项对应），从而把 v2 的蒙特
# 卡洛实现与解析通道对拍；时间线编译复用 v2 的 _prepare_events（编译器本身
# 由独立的单元测试覆盖）。
#
# 覆盖用例：
#   a) 空闲热弛豫时长精确性（解析 exp(-t/T1) 对拍）
#   b) 相邻边 CX 并发：动态 ZZ 随重叠时长缩放（cos²(θ·ov/T) 解析对拍）
#   c) spectator 1Q 门与 CX 并发（解析对拍）
#   d) rz 零时长：施加酉矩阵但不施加噪声
#   e) swap 事件噪声 ≈ 3×CX 串行；swap_xtalk 开关行为
#   f) always-on 空闲 ZZ：开关行为与解析对拍
#   g) 端到端迷你电路 v2(MC) vs 精确参考 |ΔF| < 0.02
#   回归：与 v1 legacy 同步波退化对齐、互斥/一致性校验、日志转换、ASAP 调度
# ============================================================================
import math

import numpy as np
import pytest

from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, Operator, Kraus
from qiskit_aer.noise import (
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
)

from sim.sim import NoiseConfig
from sim.trajectory_sim import TrajectorySimulator
from sim.trajectory_sim_v2 import (
    EventNoiseConfig,
    EventTrajectorySimulator,
    timing_log_to_events,
    schedule_phys_circuit_events,
    validate_events,
    trajectory_circuit_fidelity_events,
    make_event_fidelity_fn,
    _reduce_phys_circuit_for_fidelity_v2,
    _free_intervals,
    _intersect_free,
)
from utils.metrics import state_fidelity

LINE4 = [(0, 1), (1, 2), (2, 3)]
LINE3 = [(0, 1), (1, 2)]

# 4x4 酉矩阵（qargs=[q0, q1]，qiskit 小端约定：qargs[0] 为最低位）
# CX：q0=control, q1=target → |q0 q1⟩ → |q0, q1⊕q0⟩
CX_MAT = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
                  dtype=complex)
SWAP_MAT = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                    dtype=complex)
CZ_MAT = np.diag([1, 1, 1, -1]).astype(complex)


def _zz_mat(theta: float) -> np.ndarray:
    """exp(-i·θ·Z⊗Z)，qargs=[q1,q2] 下 |b1=b2| 分量 e^{-iθ}、其余 e^{+iθ}。"""
    return np.diag([np.exp(-1j * theta), np.exp(1j * theta),
                    np.exp(1j * theta), np.exp(-1j * theta)]).astype(complex)


def _thermal_kraus(cfg, q: int, t_us: float) -> Kraus:
    """与 v2 MC 轨迹完全一致的热弛豫通道（解析 Kraus）。"""
    t1 = cfg.t1_times[q]
    t2 = cfg.t2_times[q]
    p_reset = 1.0 - np.exp(-t_us / t1)
    p_z = 1.0 - np.exp(t_us / t1 - 2.0 * t_us / t2)
    err = amplitude_damping_error(p_reset).compose(phase_damping_error(p_z))
    return Kraus(err.to_quantumchannel())


def _depol_kraus(p: float, n_q: int) -> Kraus:
    return Kraus(depolarizing_error(p, n_q).to_quantumchannel())


def _unitary_cfg(n=4, cm=LINE4, **kw):
    """无随机噪声的干净配置（退相干/退极化关闭），用于解析对拍。"""
    base = dict(
        t1_times=[1e9] * n, t2_times=[2e9] * n, freq_ghz=[5.0] * n,
        single_q_gate_error=0.0, two_q_gate_error=0.0, coupling_map=list(cm),
    )
    base.update(kw)
    if "crosstalk_strength" not in base:
        base["crosstalk_strength"] = {}
    return EventNoiseConfig(**base)


def _noisy_cfg(n=4, cm=LINE4, **kw):
    base = dict(
        t1_times=[50.0] * n, t2_times=[70.0] * n, freq_ghz=[5.0] * n,
        single_q_gate_error=0.001, two_q_gate_error=0.01,
        coupling_map=list(cm),
    )
    base.update(kw)
    return EventNoiseConfig(**base)


def _exact_reference_fidelity(circuit: QuantumCircuit, events, cfg) -> float:
    """DensityMatrix 精确参考：同一动作流 + 精确噪声通道。"""
    sim = EventTrajectorySimulator(cfg, num_trajectories=1, seed=0)
    actions, total = sim._prepare_events(circuit, events)
    n = sim.n_qubits
    two_time = max(float(cfg.two_gate_time), 1e-9)
    eps = 1e-12
    rho = DensityMatrix.from_int(0, 2 ** n)
    t_last = [0.0] * n
    for act in actions:
        if act[0] == "zz":
            _, _, q1, q2, ang = act
            if ang != 0.0:
                rho = rho.evolve(Operator(_zz_mat(ang)), qargs=[q1, q2])
            continue
        _, s, e, op, qs, mat = act
        dur = e - s
        if dur > eps:
            for q in qs:
                idle = s - t_last[q]
                if idle > eps:
                    rho = rho.evolve(_thermal_kraus(cfg, q, idle), qargs=[q])
        if op == "swap":
            rho = rho.evolve(Operator(SWAP_MAT), qargs=[qs[0], qs[1]])
            if dur > eps:
                p = sim._two_error(qs[0], qs[1])
                th = sim._crosstalk_theta(qs[0], qs[1])
                for _ in range(3):
                    if p > 0:
                        rho = rho.evolve(_depol_kraus(p, 2),
                                         qargs=[qs[0], qs[1]])
                    if th != 0.0:
                        rho = rho.evolve(Operator(_zz_mat(th)),
                                         qargs=[qs[0], qs[1]])
                for q in qs:
                    rho = rho.evolve(_thermal_kraus(cfg, q, dur), qargs=[q])
        elif op in ("cx", "cz"):
            m = CX_MAT if op == "cx" else CZ_MAT
            rho = rho.evolve(Operator(m), qargs=[qs[0], qs[1]])
            if dur > eps:
                p = sim._two_error(qs[0], qs[1])
                if p > 0:
                    rho = rho.evolve(_depol_kraus(p, 2),
                                     qargs=[qs[0], qs[1]])
                th = sim._crosstalk_theta(qs[0], qs[1]) * dur / two_time
                if th != 0.0:
                    rho = rho.evolve(Operator(_zz_mat(th)),
                                     qargs=[qs[0], qs[1]])
                for q in qs:
                    rho = rho.evolve(_thermal_kraus(cfg, q, dur), qargs=[q])
        else:
            rho = rho.evolve(Operator(mat), qargs=[qs[0]])
            if dur > eps:
                if op != "id":
                    p = sim._one_error(qs[0])
                    if p > 0:
                        rho = rho.evolve(_depol_kraus(p, 1), qargs=[qs[0]])
                rho = rho.evolve(_thermal_kraus(cfg, qs[0], dur),
                                 qargs=[qs[0]])
        for q in qs:
            t_last[q] = e
    for q in range(n):
        idle = total - t_last[q]
        if idle > eps:
            rho = rho.evolve(_thermal_kraus(cfg, q, idle), qargs=[q])
    ideal = Statevector(sim._evolve(circuit, apply_noise=False))
    return state_fidelity(np.asarray(ideal.data), rho.data)


# ---------------------------------------------------------------------------
# a) 空闲热弛豫
# ---------------------------------------------------------------------------
def test_a_idle_thermal_analytic():
    cfg = _noisy_cfg(n=2, cm=[(0, 1)], single_q_gate_error=0.0)
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.id(0)
    events = [(0.0, 0.035, "x", (0,), 0), (0.035, 1.035, "id", (0,), 1)]
    sim = EventTrajectorySimulator(cfg, num_trajectories=1024, seed=0)
    fid = sim.fidelity_events(qc, events)
    # 轨迹保真度 ∈ {0,1}：均值 = P(无振幅跳变) = exp(-t_total/T1)
    expected = math.exp(-1.035 / 50.0)
    assert abs(fid - expected) < 0.02
    ref = _exact_reference_fidelity(qc, events, cfg)
    assert abs(fid - ref) < 0.01


def test_a_idle_tail_window():
    # 收尾空闲：事件结束后到 total 的尾段也要退相干
    cfg = _noisy_cfg(n=2, cm=[(0, 1)], single_q_gate_error=0.0)
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.id(1)
    # x 门 0.035µs，总时长 2.0µs → 尾段空闲 1.965µs
    events = [(0.0, 0.035, "x", (0,), 0), (1.0, 2.0, "id", (1,), 1)]
    sim = EventTrajectorySimulator(cfg, num_trajectories=1024, seed=0)
    fid = sim.fidelity_events(qc, events)
    expected = math.exp(-2.0 / 50.0)
    assert abs(fid - expected) < 0.02


# ---------------------------------------------------------------------------
# b) 动态串扰：重叠时长缩放
# ---------------------------------------------------------------------------
def test_b_dynamic_zz_overlap_scaling():
    # crosstalk_strength 直接作为 θ(rad)，χ_rate = θ/two_gate_time = 1 rad/µs
    cfg = _unitary_cfg(4, LINE4,
                       crosstalk_strength={(1, 2): 0.3, (2, 1): 0.3})
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(2, 3)
    sim = EventTrajectorySimulator(cfg, num_trajectories=256, seed=0)
    for start2, ov in [(0.0, 0.3), (0.15, 0.15), (0.3, 0.0)]:
        events = [(0.0, 0.035, "h", (0,), 0),
                  (0.035, 0.335, "cx", (0, 1), 1),
                  (start2 + 0.035, start2 + 0.335, "cx", (2, 3), 2)]
        fid = sim.fidelity_events(qc, events)
        # 全部通道为酉：每条轨迹保真度确定 = cos²(θ_dyn), θ_dyn = 1·ov
        expected = math.cos(ov) ** 2
        assert abs(fid - expected) < 1e-6, f"ov={ov}: {fid} vs {expected}"
        ref = _exact_reference_fidelity(qc, events, cfg)
        assert abs(fid - ref) < 1e-8


# ---------------------------------------------------------------------------
# c) spectator 1Q 门并发
# ---------------------------------------------------------------------------
def test_c_spectator_1q_concurrent():
    cfg = _unitary_cfg(4, LINE4,
                       crosstalk_strength={(1, 2): 0.3, (2, 1): 0.3})
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(2)
    events = [(0.0, 0.035, "h", (0,), 0),
              (0.035, 0.335, "cx", (0, 1), 1),
              (0.035, 0.335, "h", (2,), 2)]
    sim = EventTrajectorySimulator(cfg, num_trajectories=256, seed=0)
    fid = sim.fidelity_events(qc, events)
    # ZZ(1,2)：Bell(0,1) 两分量 |00⟩/|11⟩ 相位 e^{∓iθ} → 相对相位 2θ
    # ψ_id = ½(|000⟩+|001⟩+|110⟩+|111⟩)，ZZ 相位 {e^{-iθ}, e^{+iθ},
    # e^{+iθ}, e^{-iθ}} → ⟨ψ|ψ_t⟩ = cosθ → F = cos²θ
    expected = math.cos(0.3) ** 2
    assert abs(fid - expected) < 1e-6
    ref = _exact_reference_fidelity(qc, events, cfg)
    assert abs(fid - ref) < 1e-8


# ---------------------------------------------------------------------------
# d) rz 零时长
# ---------------------------------------------------------------------------
def test_d_rz_zero_duration_unitary_only():
    cfg = _unitary_cfg(2, [(0, 1)])
    qc = QuantumCircuit(2)
    qc.rz(1.234, 0)
    events = [(0.0, 0.0, "rz", (0,), 0)]
    sim = EventTrajectorySimulator(cfg, num_trajectories=64, seed=0)
    fid = sim.fidelity_events(qc, events)
    assert abs(fid - 1.0) < 1e-9
    # 酉矩阵必须真正被施加（若跳过 rz，与 ideal 的保真度 < 1）
    sv = sim.evolve_events(qc, events)
    ideal = sim._evolve(qc, apply_noise=False)
    overlap = abs(np.vdot(ideal, sv)) ** 2
    assert abs(overlap - 1.0) < 1e-9


def test_d_rz_no_noise_realistic():
    # 真实噪声配置下 rz 零时长不应引入任何退相干
    cfg = _noisy_cfg(n=1, cm=[], single_q_gate_error=0.01)
    qc = QuantumCircuit(1)
    qc.rz(0.7, 0)
    events = [(0.0, 0.0, "rz", (0,), 0)]
    sim = EventTrajectorySimulator(cfg, num_trajectories=256, seed=0)
    fid = sim.fidelity_events(qc, events)
    assert abs(fid - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# e) swap 事件 ≈ 3×CX；swap_xtalk 开关
# ---------------------------------------------------------------------------
def test_e_swap_equivalent_3cx():
    cfg = _noisy_cfg(n=2, cm=[(0, 1)])
    qc_swap = QuantumCircuit(2)
    qc_swap.swap(0, 1)
    events_swap = [(0.0, 0.9, "swap", (0, 1), 0)]
    qc_3cx = QuantumCircuit(2)
    qc_3cx.cx(0, 1)
    qc_3cx.cx(0, 1)
    qc_3cx.cx(0, 1)
    events_3cx = [(0.0, 0.3, "cx", (0, 1), 0),
                  (0.3, 0.6, "cx", (0, 1), 1),
                  (0.6, 0.9, "cx", (0, 1), 2)]
    sim = EventTrajectorySimulator(cfg, num_trajectories=512, seed=3)
    f_swap = sim.fidelity_events(qc_swap, events_swap)
    f_3cx = sim.fidelity_events(qc_3cx, events_3cx)
    assert f_swap < 0.999 and f_3cx < 0.999  # 噪声确实存在
    assert abs(f_swap - f_3cx) < 0.02
    ref = _exact_reference_fidelity(qc_swap, events_swap, cfg)
    assert abs(f_swap - ref) < 0.02


def test_e_swap_noise_off():
    # swap_noise=False（v1 语义）：swap 仅推进时钟，无任何门噪声
    cfg = _noisy_cfg(n=2, cm=[(0, 1)])
    cfg.swap_noise = False
    qc = QuantumCircuit(2)
    qc.swap(0, 1)
    events = [(0.0, 0.9, "swap", (0, 1), 0)]
    sim = EventTrajectorySimulator(cfg, num_trajectories=64, seed=0)
    # 含噪声配置下关掉 swap 噪声 → 纯酉演化 → F = 1
    assert abs(sim.fidelity_events(qc, events) - 1.0) < 1e-9
    # 对照：默认 swap_noise=True 时有噪声（复用同缓存配置则需新实例）
    cfg_on = _noisy_cfg(n=2, cm=[(0, 1)])
    sim_on = EventTrajectorySimulator(cfg_on, num_trajectories=64, seed=0)
    assert sim_on.fidelity_events(qc, events) < 0.999


def test_e_swap_xtalk_flag():
    base = dict(crosstalk_strength={(1, 2): 0.3, (2, 1): 0.3})
    qc = QuantumCircuit(4)
    qc.swap(0, 1)
    qc.h(2)
    qc.cx(2, 3)
    # 时间序：swap[0,0.9] 与 h(2)[0.035,0.07]、cx(2,3)[0.07,0.37] 重叠
    # （h 先于 cx 完成，满足共享比特的数据依赖；事件顺序与电路指令序一致）
    events = [(0.0, 0.9, "swap", (0, 1), 0),
              (0.035, 0.07, "h", (2,), 1),
              (0.07, 0.37, "cx", (2, 3), 2)]
    sim_off = EventTrajectorySimulator(
        _unitary_cfg(4, LINE4, **base), num_trajectories=64, seed=0)
    assert abs(sim_off.fidelity_events(qc, events) - 1.0) < 1e-9
    cfg_on = _unitary_cfg(4, LINE4, **base)
    cfg_on.swap_xtalk = True
    sim_on = EventTrajectorySimulator(cfg_on, num_trajectories=64, seed=0)
    fid = sim_on.fidelity_events(qc, events)
    # 两个动态 ZZ(1,2)：swap&h(2) ov=0.035 → θ₁=0.035（q2 叠加态上相对相位
    # 2θ₁）；swap&cx(2,3) ov=0.3 → θ₂=0.3（作用在 Bell(2,3) 上）。
    # ⟨ψ_id|ψ_t⟩ = ½(e^{-iθ₂} + e^{i(θ₂+2θ₁)})
    th1, th2 = 0.035, 0.3
    inner = 0.5 * (complex(math.cos(th2), -math.sin(th2))
                   + complex(math.cos(th2 + 2 * th1), math.sin(th2 + 2 * th1)))
    assert abs(fid - abs(inner) ** 2) < 1e-6


# ---------------------------------------------------------------------------
# f) always-on 空闲 ZZ
# ---------------------------------------------------------------------------
def test_f_always_on_zz_toggle():
    qc = QuantumCircuit(3)
    qc.h(1)
    qc.x(0)
    qc.id(1)
    events = [(0.0, 0.035, "h", (1,), 0),
              (0.0, 0.035, "x", (0,), 1),
              (0.035, 1.035, "id", (1,), 2)]
    # OFF（默认）：F = 1
    cfg_off = _unitary_cfg(3, LINE3)
    sim_off = EventTrajectorySimulator(cfg_off, num_trajectories=64, seed=0)
    assert abs(sim_off.fidelity_events(qc, events) - 1.0) < 1e-9
    # ON：q1/q2 双空闲 [0.035, 1.035]，θ = 0.5·1.0 = 0.5 → F = cos²(0.5)
    cfg_on = _unitary_cfg(3, LINE3, always_on_zz={(1, 2): 0.5})
    sim_on = EventTrajectorySimulator(cfg_on, num_trajectories=64, seed=0)
    fid = sim_on.fidelity_events(qc, events)
    assert abs(fid - math.cos(0.5) ** 2) < 1e-6
    ref = _exact_reference_fidelity(qc, events, cfg_on)
    assert abs(fid - ref) < 1e-8


def test_f_always_on_segments():
    # 单元测试：忙碌区间求空闲 + 区间求交
    assert _free_intervals([], 4.0) == [(0.0, 4.0)]
    assert _free_intervals([(0.0, 1.0), (2.0, 3.0)], 4.0) == \
        [(1.0, 2.0), (3.0, 4.0)]
    assert _intersect_free([(1.0, 2.0), (3.0, 4.0)], [(0.5, 3.5)]) == \
        [(1.0, 2.0), (3.0, 3.5)]
    assert _intersect_free([(0.0, 1.0)], [(2.0, 3.0)]) == []


def test_f_always_on_reduce_remap():
    # 缩减路径保留并重映射 always_on_zz
    cfg = _unitary_cfg(4, LINE4, always_on_zz={(0, 1): 0.5, (2, 3): 0.2})
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    rc, rcfg, remap = _reduce_phys_circuit_for_fidelity_v2(qc, cfg)
    assert remap == {0: 0, 1: 1}
    assert rcfg.always_on_zz == {(0, 1): 0.5}
    assert rcfg.swap_xtalk is False


# ---------------------------------------------------------------------------
# g) 端到端 vs 精确参考
# ---------------------------------------------------------------------------
def _end_to_end_case():
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.h(2)
    circuit.rz(0.5, 2)
    circuit.cx(1, 2)
    circuit.x(3)
    circuit.cx(2, 3)
    circuit.h(0)
    events = [
        (0.0, 0.035, "h", (0,), 0),
        (0.035, 0.335, "cx", (0, 1), 1),
        (0.035, 0.07, "h", (2,), 2),
        (0.07, 0.07, "rz", (2,), 3),
        (0.335, 0.635, "cx", (1, 2), 4),
        (0.335, 0.37, "x", (3,), 5),
        (0.635, 0.935, "cx", (2, 3), 6),
        (0.635, 0.67, "h", (0,), 7),
    ]
    return circuit, events


def test_g_end_to_end_vs_reference():
    circuit, events = _end_to_end_case()
    cfg = _noisy_cfg(4, LINE4)
    sim = EventTrajectorySimulator(cfg, num_trajectories=256, seed=11)
    fid = sim.fidelity_events(circuit, events)
    ref = _exact_reference_fidelity(circuit, events, cfg)
    assert abs(fid - ref) < 0.02
    assert 0.0 < fid < 1.0


def test_g_end_to_end_vs_reference_with_always_on():
    circuit, events = _end_to_end_case()
    cfg = _noisy_cfg(4, LINE4, always_on_zz={(0, 1): 0.2, (2, 3): 0.1})
    sim = EventTrajectorySimulator(cfg, num_trajectories=256, seed=11)
    fid = sim.fidelity_events(circuit, events)
    ref = _exact_reference_fidelity(circuit, events, cfg)
    assert abs(fid - ref) < 0.02


def test_g_trajectory_circuit_fidelity_events():
    # 独立电路入口（基线评估路径）：不传事件，内部平铺 ASAP
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    cfg = _noisy_cfg(4, LINE4)
    fid = trajectory_circuit_fidelity_events(circuit, cfg,
                                             num_trajectories=64, seed=0)
    assert 0.0 < fid < 1.0


# ---------------------------------------------------------------------------
# 回归：与 v1 legacy 同步波退化对齐
# ---------------------------------------------------------------------------
def test_regression_alignment_legacy_serial():
    cfg = _noisy_cfg(4, LINE4)
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.h(3)
    waves = [
        (0.1, [(0, (0,), False)]),
        (0.3, [(1, (0, 1), True)]),
        (0.3, [(2, (2, 3), True)]),
        (0.1, [(3, (3,), False)]),
    ]
    events = [
        (0.0, 0.1, "h", (0,), 0),
        (0.1, 0.4, "cx", (0, 1), 1),
        (0.4, 0.7, "cx", (2, 3), 2),
        (0.7, 0.8, "h", (3,), 3),
    ]
    legacy = TrajectorySimulator(cfg, num_trajectories=64, seed=7)
    v2 = EventTrajectorySimulator(cfg, num_trajectories=64, seed=7)
    f_legacy = legacy.fidelity_scheduled(qc, waves)
    f_v2 = v2.fidelity_events(qc, events)
    assert abs(f_legacy - f_v2) < 1e-6


def test_regression_alignment_legacy_concurrent():
    # 同步纯 cx 波：v1 每波一次固定 θ；v2 χ_rate·ov = θ/0.3·0.3 = θ，应一致
    cfg = _noisy_cfg(4, LINE4)
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)
    waves = [
        (0.3, [(0, (0, 1), True), (1, (2, 3), True)]),
        (0.3, [(2, (1, 2), True)]),
    ]
    events = [
        (0.0, 0.3, "cx", (0, 1), 0),
        (0.0, 0.3, "cx", (2, 3), 1),
        (0.3, 0.6, "cx", (1, 2), 2),
    ]
    legacy = TrajectorySimulator(cfg, num_trajectories=64, seed=7)
    v2 = EventTrajectorySimulator(cfg, num_trajectories=64, seed=7)
    f_legacy = legacy.fidelity_scheduled(qc, waves)
    f_v2 = v2.fidelity_events(qc, events)
    assert abs(f_legacy - f_v2) < 1e-6


# ---------------------------------------------------------------------------
# 回归：校验 / 转换 / 调度
# ---------------------------------------------------------------------------
def test_validation_errors():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    ok = [(0.0, 0.035, "h", (0,), 0), (0.035, 0.335, "cx", (0, 1), 1)]
    validate_events(ok, qc)  # 不抛错
    with pytest.raises(ValueError):
        validate_events(ok[:1], qc)  # 数量不一致
    bad_op = [(0.0, 0.035, "x", (0,), 0), (0.035, 0.335, "cx", (0, 1), 1)]
    with pytest.raises(ValueError):
        validate_events(bad_op, qc)  # op 不匹配
    bad_qs = [(0.0, 0.035, "h", (0,), 0), (0.035, 0.335, "cx", (1, 0), 1)]
    with pytest.raises(ValueError):
        validate_events(bad_qs, qc)  # 比特序不匹配
    overlap = [(0.0, 0.2, "h", (0,), 0), (0.035, 0.335, "cx", (0, 1), 1)]
    with pytest.raises(ValueError):
        validate_events(overlap, qc)  # 比特互斥被违反
    ecr_qc = QuantumCircuit(2)
    ecr_qc.ecr(0, 1)
    ecr = [(0.0, 0.4, "ecr", (0, 1), 0)]
    with pytest.raises(NotImplementedError):
        validate_events(ecr, ecr_qc)  # 不支持的双比特门


def test_timing_log_to_events():
    qc = QuantumCircuit(2)
    qc.h(1)
    qc.swap(0, 1)
    qc.measure_all()
    log = [
        {"kind": "1q", "gate_idx": 0, "op": "h", "qubits": [1],
         "start": 0.0, "end": 0.035, "wave": 0},
        {"kind": "swap", "gate_idx": -1, "op": "swap", "qubits": [0, 1],
         "start": 0.035, "end": 0.935, "wave": 1},
        {"kind": "measure", "gate_idx": -2, "op": "measure", "qubits": [0],
         "start": 1.0, "end": 3.0, "wave": 2},
    ]
    events = timing_log_to_events(log, circuit=qc)
    assert events == [
        (0.0, 0.035, "h", (1,), 0),
        (0.035, 0.935, "swap", (0, 1), 1),
    ]
    events_r = timing_log_to_events(log, circuit=qc, remap={0: 1, 1: 0})
    assert events_r[0][3] == (0,)
    assert events_r[1][3] == (1, 0)
    # 无电路时 phys_idx 退化为日志序号
    events_nc = timing_log_to_events(log)
    assert events_nc[1][4] == 1


def test_schedule_phys_circuit_events():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(2)
    events = schedule_phys_circuit_events(qc)
    assert len(events) == 3
    assert events[0] == (0.0, 0.035, "h", (0,), 0)
    assert events[1][2:] == ("cx", (0, 1), 1)
    assert events[1][0] == pytest.approx(0.035)
    assert events[1][1] == pytest.approx(0.335)
    assert events[2] == (0.0, 0.035, "h", (2,), 2)
    events2 = schedule_phys_circuit_events(qc, durations={"h": 0.1})
    assert events2[0] == (0.0, 0.1, "h", (0,), 0)
    assert events2[1][0] == pytest.approx(0.1)
    assert events2[1][1] == pytest.approx(0.4)


def test_make_event_fidelity_fn():
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    log = [
        {"kind": "1q", "gate_idx": 0, "op": "h", "qubits": [0],
         "start": 0.0, "end": 0.035, "wave": 0},
        {"kind": "2q", "gate_idx": 1, "op": "cx", "qubits": [0, 1],
         "start": 0.035, "end": 0.335, "wave": 1},
    ]
    cfg = _noisy_cfg(3, LINE3)

    class _FakeTiming:
        def __init__(self, lg):
            self.schedule_log = lg

    class _FakeEnv:
        def __init__(self, phys, lg):
            self._phys_circuit = phys
            self.timing = _FakeTiming(lg) if lg is not None else None

    fn = make_event_fidelity_fn(cfg, num_trajectories=64, seed=0)
    fid = fn(_FakeEnv(circuit, log))
    assert 0.0 < fid < 1.0
    # 无调度日志 → 回退平铺 ASAP，不抛错
    fid2 = fn(_FakeEnv(circuit, None))
    assert 0.0 < fid2 < 1.0
    # 日志与电路不一致 → 回退，不抛错
    bad_log = [{"kind": "1q", "gate_idx": 0, "op": "x", "qubits": [0],
                "start": 0.0, "end": 0.035, "wave": 0}]
    fid3 = fn(_FakeEnv(circuit, bad_log))
    assert 0.0 < fid3 < 1.0


def test_empty_events():
    cfg = _noisy_cfg(2, [(0, 1)])
    qc = QuantumCircuit(2)
    sim = EventTrajectorySimulator(cfg, num_trajectories=16, seed=0)
    assert abs(sim.fidelity_events(qc, []) - 1.0) < 1e-9


def test_run_events_counts():
    cfg = _noisy_cfg(2, [(0, 1)])
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.measure_all()
    events = [(0.0, 0.035, "x", (0,), 0)]
    sim = EventTrajectorySimulator(cfg, num_trajectories=8, seed=0)
    counts = sim.run_events(qc, events, shots=64)
    assert sum(counts.values()) == 64
    assert all(set(k) <= {"0", "1"} for k in counts)
    # x(0) + 低噪声 → '10'（q0=1）应占绝大多数
    dominant = max(counts, key=counts.get)
    assert counts[dominant] / 64 > 0.9
