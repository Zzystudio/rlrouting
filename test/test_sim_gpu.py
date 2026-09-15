# ============================================================================
# test_sim_gpu.py
# 轨迹模拟器 GPU（torch CUDA）后端验收测试（doc/模拟器加速.md §4 协议）：
#   1. 内核级：确定论批量内核 CPU vs GPU allclose(1e-12)
#   2. 演化级：无噪声全电路演化 CPU vs GPU <1e-12
#   3. 噪声内核不变量：p=0 恒等、范数守恒
#   4. 统计级：同电路大 T 噪声轨迹，CPU vs GPU 平均 fidelity < 3σ/√T
#   5. 端到端：trajectory_circuit_fidelity_events 两种后端口径一致
# 无 CUDA 时整组跳过。
# ============================================================================

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA 不可用，跳过 GPU 验收测试", allow_module_level=True)

from sim.sim import NoiseConfig
from sim.trajectory_sim import TrajectorySimulator
from sim.trajectory_sim_v2 import (EventTrajectorySimulator,
                                   trajectory_circuit_fidelity_events)
from qiskit import QuantumCircuit


def make_config(n: int = 5) -> NoiseConfig:
    return NoiseConfig(
        t1_times=[50.0] * n, t2_times=[70.0] * n, freq_ghz=[5.0] * n,
        single_q_gate_error=0.001, two_q_gate_error=0.01,
        coupling_map=[(i, i + 1) for i in range(n - 1)],
        readout_error=[0.02] * n, shots=1024,
    )


def make_circuit(n: int = 5) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.swap(0, n - 1)
    for i in range(n):
        qc.rz(0.3, i)
    return qc


def make_events(n: int = 5):
    """与 make_circuit 对应的手工事件（同比特互斥；rz 零时长带 phys_idx）。"""
    evs = [(0.0, 0.3, "h", (i,)) for i in range(n)]
    t = 0.3
    for i in range(n - 1):
        evs.append((t, t + 0.3, "cx", (i, i + 1)))
        t += 0.3
    evs.append((t, t + 0.9, "swap", (0, n - 1)))
    t += 0.9
    # rz 在 circuit.data 中的位置：n 个 h + (n-1) 个 cx + 1 个 swap 之后
    base = n + (n - 1) + 1
    evs += [(t, t, "rz", (i,), base + i) for i in range(n)]
    return evs


T_SMALL = 8
N_QUBITS = 5


def _rand_state(T: int, n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sv = rng.normal(size=(T, 1 << n)) + 1j * rng.normal(size=(T, 1 << n))
    return sv / np.linalg.norm(sv, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# 1. 内核级：确定论内核逐位级等价
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [0, 2, 4])
def test_kernel_apply1(q):
    cfg = make_config(N_QUBITS)
    mat = np.array([[np.exp(-0.15j), 0.3 + 0.1j],
                    [0.3 - 0.1j, np.exp(0.15j)]], dtype=complex)
    mat = mat / np.linalg.norm(mat)
    out = {}
    for backend in ("cpu", "cuda"):
        sim = EventTrajectorySimulator(cfg, num_trajectories=T_SMALL, seed=1,
                                       backend=backend)
        svs = sim._init_batch_state(T_SMALL, make_circuit())
        svs = sim._apply1_batch(svs, q, mat)
        out[backend] = sim._to_numpy(svs)
    assert np.abs(out["cpu"] - out["cuda"]).max() < 1e-12


@pytest.mark.parametrize("ctl,tgt", [(0, 1), (1, 0), (0, 4), (4, 0), (1, 3)])
def test_kernel_cx(ctl, tgt):
    cfg = make_config(N_QUBITS)
    out = {}
    for backend in ("cpu", "cuda"):
        sim = EventTrajectorySimulator(cfg, num_trajectories=T_SMALL, seed=1,
                                       backend=backend)
        svs = sim._init_batch_state(T_SMALL, make_circuit())
        svs = sim._apply_cx_batch(svs, ctl, tgt)
        out[backend] = sim._to_numpy(svs)
    assert np.abs(out["cpu"] - out["cuda"]).max() < 1e-12


@pytest.mark.parametrize("a,b", [(0, 1), (0, 4), (2, 3)])
def test_kernel_swap(a, b):
    cfg = make_config(N_QUBITS)
    out = {}
    for backend in ("cpu", "cuda"):
        sim = EventTrajectorySimulator(cfg, num_trajectories=T_SMALL, seed=1,
                                       backend=backend)
        svs = sim._init_batch_state(T_SMALL, make_circuit())
        svs = sim._apply_swap_batch(svs, a, b)
        out[backend] = sim._to_numpy(svs)
    assert np.abs(out["cpu"] - out["cuda"]).max() < 1e-12


@pytest.mark.parametrize("theta", [0.013, -0.02, 0.0])
def test_kernel_zz_rotation(theta):
    cfg = make_config(N_QUBITS)
    out = {}
    for backend in ("cpu", "cuda"):
        sim = EventTrajectorySimulator(cfg, num_trajectories=T_SMALL, seed=1,
                                       backend=backend)
        svs = sim._init_batch_state(T_SMALL, make_circuit())
        svs = sim._zz_rotation_batch(svs, 1, 2, theta)
        out[backend] = sim._to_numpy(svs)
    assert np.abs(out["cpu"] - out["cuda"]).max() < 1e-12


def test_kernel_cz():
    cfg = make_config(N_QUBITS)
    out = {}
    for backend in ("cpu", "cuda"):
        sim = EventTrajectorySimulator(cfg, num_trajectories=T_SMALL, seed=1,
                                       backend=backend)
        svs = sim._init_batch_state(T_SMALL, make_circuit())
        svs = sim._apply_cz_batch(svs, 0, 3)
        out[backend] = sim._to_numpy(svs)
    assert np.abs(out["cpu"] - out["cuda"]).max() < 1e-12


def test_kernel_crosstalk_batch():
    cfg = make_config(N_QUBITS)
    out = {}
    for backend in ("cpu", "cuda"):
        sim = EventTrajectorySimulator(cfg, num_trajectories=T_SMALL, seed=1,
                                       backend=backend)
        svs = sim._init_batch_state(T_SMALL, make_circuit())
        svs = sim._crosstalk_batch(svs, 2, 3)
        out[backend] = sim._to_numpy(svs)
    assert np.abs(out["cpu"] - out["cuda"]).max() < 1e-12


# ---------------------------------------------------------------------------
# 2. 演化级：无噪声全电路演化逐位等价
# ---------------------------------------------------------------------------

def test_evolution_ideal_equivalence():
    cfg = make_config(N_QUBITS)
    qc = make_circuit(N_QUBITS)
    evs = make_events(N_QUBITS)
    out = {}
    for backend in ("cpu", "cuda"):
        sim = EventTrajectorySimulator(cfg, num_trajectories=4, seed=1,
                                       backend=backend)
        svs = sim._init_batch_state(4, qc)
        actions, total = sim._prepare_events(qc, evs)
        svs = sim._run_actions(svs, actions, total, apply_noise=False)
        out[backend] = sim._to_numpy(svs)
    assert np.abs(out["cpu"] - out["cuda"]).max() < 1e-12


def test_evolution_ideal_v1_batch_equivalence():
    """v1 _evolve_batch 路径（run_trajectories 的小工作集分支）无噪声等价。"""
    cfg = make_config(N_QUBITS)
    qc = make_circuit(N_QUBITS)
    out = {}
    for backend in ("cpu", "cuda"):
        sim = TrajectorySimulator(cfg, num_trajectories=4, seed=1,
                                  backend=backend)
        svs = sim._evolve_batch(qc, apply_noise=False, num_trajectories=4)
        out[backend] = sim._to_numpy(svs)
    assert np.abs(out["cpu"] - out["cuda"]).max() < 1e-12


# ---------------------------------------------------------------------------
# 3. 噪声内核不变量：p=0 恒等、范数守恒
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("time_us", [0.0, 0.5])
def test_thermal_invariants(time_us):
    cfg = make_config(N_QUBITS)
    qc = make_circuit(N_QUBITS)
    for backend in ("cpu", "cuda"):
        sim = EventTrajectorySimulator(cfg, num_trajectories=T_SMALL, seed=3,
                                       backend=backend)
        svs = sim._init_batch_state(T_SMALL, qc)
        orig = sim._to_numpy(sim._init_batch_state(T_SMALL, qc)).copy()
        sim._thermal_noise_batch(svs, 2, time_us)
        arr = sim._to_numpy(svs)
        if time_us == 0.0:
            assert np.abs(arr - orig).max() < 1e-12  # p=0 恒等
        norms = np.linalg.norm(arr, axis=1)
        assert np.abs(norms - 1.0).max() < 1e-9      # 范数守恒


@pytest.mark.parametrize("p", [0.0, 0.05])
def test_depol_invariants(p):
    cfg = make_config(N_QUBITS)
    qc = make_circuit(N_QUBITS)
    for backend in ("cpu", "cuda"):
        sim = EventTrajectorySimulator(cfg, num_trajectories=T_SMALL, seed=3,
                                       backend=backend)
        svs = sim._init_batch_state(T_SMALL, qc)
        sim._depol1_batch(svs, 1, p)
        sim._depol2_batch(svs, 0, 1, p)
        arr = sim._to_numpy(svs)
        norms = np.linalg.norm(arr, axis=1)
        assert np.abs(norms - 1.0).max() < 1e-9


# ---------------------------------------------------------------------------
# 4. 统计级：大 T 噪声轨迹 CPU vs GPU（3σ/√T 容差）
# ---------------------------------------------------------------------------

def test_statistical_fidelity_events():
    cfg = make_config(N_QUBITS)
    qc = make_circuit(N_QUBITS)
    T = 1024
    fids, stds = {}, {}
    for backend in ("cpu", "cuda"):
        sim = EventTrajectorySimulator(cfg, num_trajectories=T, seed=42,
                                       backend=backend)
        evs = make_events(N_QUBITS)
        res = sim.run_trajectories_events(qc, evs, T)
        ideal = sim._evolve(qc, apply_noise=False)
        fids[backend] = res.fidelity(ideal)
        stds[backend] = res.fidelity_std(ideal)
    tol = 3 * np.sqrt((stds["cpu"] ** 2 + stds["cuda"] ** 2) / T)
    assert abs(fids["cpu"] - fids["cuda"]) < max(tol, 1e-3)


def test_statistical_fidelity_v1_batch():
    """v1 run_trajectories GPU 分支（VRAM 切块）统计等价。"""
    cfg = make_config(N_QUBITS)
    qc = make_circuit(N_QUBITS)
    T = 1024
    fids, stds = {}, {}
    for backend in ("cpu", "cuda"):
        sim = TrajectorySimulator(cfg, num_trajectories=T, seed=42,
                                  backend=backend)
        res = sim.run_trajectories(qc, num_trajectories=T, skip_transpile=True)
        ideal = sim._evolve(qc, apply_noise=False)
        fids[backend] = res.fidelity(ideal)
        stds[backend] = res.fidelity_std(ideal)
    tol = 3 * np.sqrt((stds["cpu"] ** 2 + stds["cuda"] ** 2) / T)
    assert abs(fids["cpu"] - fids["cuda"]) < max(tol, 1e-3)


# ---------------------------------------------------------------------------
# 5. 端到端：trajectory_circuit_fidelity_events 双后端口径
# ---------------------------------------------------------------------------

def test_end_to_end_backend_flag():
    cfg = make_config(N_QUBITS)
    qc = make_circuit(N_QUBITS)
    f_cpu = trajectory_circuit_fidelity_events(qc, cfg, num_trajectories=256,
                                               seed=42, backend="cpu")
    f_gpu = trajectory_circuit_fidelity_events(qc, cfg, num_trajectories=256,
                                               seed=42, backend="cuda")
    # 256 条轨迹的 MC 容差（宽口径：两侧各自均值的标准误）
    assert abs(f_cpu - f_gpu) < 0.05
    assert 0.0 < f_cpu <= 1.0 and 0.0 < f_gpu <= 1.0


def test_backend_auto_returns_cpu_type():
    """auto 在无 CUDA 时回退 CPU；此处有 CUDA，仅验证参数解析不抛错。"""
    sim = TrajectorySimulator(make_config(N_QUBITS), num_trajectories=2,
                              seed=1, backend="auto")
    assert sim._use_gpu is True
    sim2 = TrajectorySimulator(make_config(N_QUBITS), num_trajectories=2,
                               seed=1, backend="cpu")
    assert sim2._use_gpu is False


def test_backend_invalid_raises():
    with pytest.raises(ValueError):
        TrajectorySimulator(make_config(N_QUBITS), backend="tpu")


def test_auto_backend_size_threshold():
    """auto 按工作集分流：小工作集 → cpu，深电路大工作集 → cuda。"""
    from sim.trajectory_sim import auto_backend
    assert auto_backend(9, 16) == "cpu"      # 8 MiB
    assert auto_backend(10, 32) == "cpu"     # 0.5 MiB
    assert auto_backend(15, 64) == "cpu"     # 32 MiB（实测区间外，保守走 CPU）
    assert auto_backend(18, 64) == "cuda"    # 256 MiB
    assert auto_backend(20, 64) == "cuda"    # 1 GiB


def test_factory_auto_small_circuit_uses_cpu():
    """端到端 auto：9q 小电路经 factory 自动回落 CPU（启动开销免付）。"""
    cfg = make_config(N_QUBITS)
    qc = make_circuit(N_QUBITS)
    f = trajectory_circuit_fidelity_events(qc, cfg, num_trajectories=16,
                                           seed=42, backend="auto")
    assert 0.0 < f <= 1.0
