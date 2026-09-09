# ============================================================================
# trajectory_sim.py
# 基于轨迹采样（蒙特卡洛波函数）+ 状态向量的噪声模拟器
#
# 相比 sim.py（Aer density_matrix，内存 4^n），本实现直接维护单个纯态状态
# 向量（内存 2^n，复数 128 位），对每条噪声通道（热弛豫/退极化/ZZ 串扰/读出）
# 做 Kraus 的 Monte-Carlo 采样，最后对多条轨迹平均：
#
#   F = (1/T) * sum_t |<psi_ideal | psi_t>|^2        （态保真度）
#
# 这样既保留密度矩阵能捕获的噪声统计，又将内存需求从 4^n 降到 2^n
# （n=20 -> 16 TB -> ~1 KB），可以支撑更大规模电路的噪声感知训练。
#
# 接口与 sim.sim.NoiseSimulator 兼容：
#   - run(circuit, shots)                 -> counts dict
#   - run_and_get_counts()                -> counts dict
#   - run_and_get_statevector()           -> TrajectoryResult（含 .data）
#   - run_batch() / run_batch_statevector()
#   - _transpile()
# ============================================================================

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from qiskit import QuantumCircuit, transpile

from sim.sim import NoiseConfig

logger = logging.getLogger(__name__)

# 平均密度矩阵内存上限：超过则禁止 .data（建议改用 TrajectoryResult.fidelity）
_DENSITY_MEM_LIMIT = 1 << 31  # 2 GiB

# 单比特相对旋转（rz）出参用；与 qiskit 一致含全局相位 e^{-iθ/2}
def _rz_matrix(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=complex,
    )


# 基础单/双比特酉矩阵
_U1 = {  # 1-qubit gates
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
    "h": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "t": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
    "tdg": np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex),
    "sx": np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex) / 2,
}

_SQ_MATS = dict(_U1)

# 双比特退极化用 15 个非单位 Pauli（(pauli_a, pauli_b)）
_TWOQ_PAULIS: List[Tuple[str, str]] = [
    (a, b) for a in ("I", "X", "Y", "Z") for b in ("I", "X", "Y", "Z")
    if (a, b) != ("I", "I")
]


class TrajectoryResult:
    """一次轨迹采样结果：纯态状态向量集合（兼容旧 DensityMatrix 用法）。

    Attributes
    ----------
    statevectors : np.ndarray (T, 2^n) complex128
        每条轨迹的末态。
    data : np.ndarray (2^n, 2^n) complex128
        平均密度矩阵 rho = (1/T) sum_t |psi_t><psi_t|。
        大 n（超内存阈值）访问时抛 RuntimeError，提示改用 TrajectoryResult.fidelity。
    """

    def __init__(self, statevectors: np.ndarray):
        self.statevectors = np.asarray(statevectors, dtype=complex)
        self.num_trajectories = int(self.statevectors.shape[0])
        self._data: Optional[np.ndarray] = None

    def per_trajectory_fidelity(self, ideal_sv: np.ndarray) -> np.ndarray:
        """每个轨迹计算 F_t = |<psi_ideal|psi_t>|^2。"""
        psi = np.asarray(ideal_sv, dtype=complex).reshape(-1)
        if psi.shape != (self.statevectors.shape[1],):
            raise ValueError(f"ideal_sv 长度 {psi.shape[0]} 与状态向量 {self.statevectors.shape[1]} 不一致")
        # <psi_ideal|psi_t> = sum conj(psi_ideal) * psi_t
        inner = self.statevectors @ psi.conj()
        return np.abs(inner) ** 2

    def fidelity(self, ideal_sv: np.ndarray) -> float:
        """轨迹平均态保真度 F = (1/T) sum |<psi_ideal|psi_t>|^2 ∈ [0,1]。"""
        return float(np.clip(np.mean(self.per_trajectory_fidelity(ideal_sv)), 0, 1))

    def fidelity_std(self, ideal_sv: np.ndarray) -> float:
        return float(np.std(self.per_trajectory_fidelity(ideal_sv)))

    @property
    def data(self):
        """兼容旧接口的平均密度矩阵（内存受限保护）。"""
        if self._data is None:
            n_entries = self.statevectors.shape[1]
            bytes_needed = n_entries * n_entries * 16
            if bytes_needed > _DENSITY_MEM_LIMIT:
                raise RuntimeError(
                    f"平均密度矩阵需要 {bytes_needed / 2 ** 30:.2f} GiB 内存（阈值 "
                    f"{_DENSITY_MEM_LIMIT / 2 ** 30:.0f} GiB），n 过大，请改用 "
                    f"TrajectoryResult.fidelity(ideal_sv)。"
                )
            self._data = (self.statevectors.conj().T @ self.statevectors) / self.num_trajectories
        return self._data

    def _init_data(self) -> None:
        self._data = None


class TrajectorySimulator:
    """
    基于轨迹采样 + 状态向量的噪声模拟器。

    Parameters
    ----------
    config : NoiseConfig
        复用 sim.sim.NoiseConfig（T1/T2、门错误率、拓扑、串扰、读出）。
    num_trajectories : int
        每条电路的采样轨迹条数（越大统计误差越小）。默认 64。
    seed : int
        随机种子（便于复现）。
    """

    def __init__(self, config: NoiseConfig, num_trajectories: int = 16,
                 seed: Optional[int] = None):
        self.config = config
        self.num_trajectories = num_trajectories
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.n_qubits = len(config.t1_times)
        self._all = None  # 惰性分配：crosstalk 相位索引用的全比特基态数组
        self._validate_config()

    # ------------------------------------------------------------------ #
    # 配置校验
    # ------------------------------------------------------------------ #
    def _validate_config(self) -> None:
        n = self.n_qubits
        if len(self.config.t2_times) != n:
            raise ValueError("t1_times 与 t2_times 长度不一致")
        if self.config.readout_error is not None and len(self.config.readout_error) != n:
            raise ValueError("readout_error 长度必须等于比特数")
        for i in range(n):
            t1 = self.config.t1_times[i]
            t2 = self.config.t2_times[i]
            if t1 <= 0 or t2 <= 0:
                raise ValueError(f"T1/T2[{i}] 必须为正")
            # 相位阻尼概率须非负：pz = 1 - exp(-2t/T2)/exp(-t/T1) >= 0  <=> t2 <= 2*t1
            if t2 > 2 * t1:
                raise ValueError(
                    f"T2[{i}]={t2} > 2*T1[{i}]={2*t1}，轨迹模拟要求 T2 <= 2*T1。"
                    f"这是热弛豫通道相位阻尼概率非负的必要条件。"
                )

    # ------------------------------------------------------------------ #
    # 顶层操作：酉演化 + 串扰/退极化 + 热弛豫
    # ------------------------------------------------------------------ #
    def _initial_state(self) -> np.ndarray:
        sv = np.zeros(2 ** self.n_qubits, dtype=complex)
        sv[0] = 1.0
        return sv

    @staticmethod
    def _axis(q: int, n: int) -> int:
        """qiskit 用 little-endian（qubit 0 为最低位）；
        numpy reshape 后 axis 0 是最高位，故 qubit q -> axis n-1-q。"""
        return n - 1 - q

    def _transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """转译到基础门（包含 'id'，以便插入空闲时间）。"""
        return transpile(
            circuit,
            basis_gates=["rz", "sx", "x", "cx", "id"],
            coupling_map=[list(e) for e in self.config.coupling_map],
            optimization_level=1,
        )

    def _gate_matrix(self, name: str, params) -> np.ndarray:
        if name == "rz":
            return _rz_matrix(float(params[0]))
        if name in _U1:
            return _U1[name]
        raise ValueError(f"不受支持的单比特门: {name}")

    def _apply1(self, sv: np.ndarray, q: int, mat: np.ndarray) -> np.ndarray:
        """作用任意 2x2 门到比特 q（qiskit little-endian 顺序）。

        原地实现：reshape(2^(n-1-q), 2, 2^q) 使中轴 stride=2^q 对应比特 q，
        |0>/|1> 切片为 strided view（零拷贝），无 moveaxis/ascontiguousarray。
        """
        nq = self.n_qubits
        s = sv.reshape(1 << (nq - 1 - q), 2, 1 << q)
        a0 = s[:, 0, :]  # |0> 分量
        a1 = s[:, 1, :]  # |1> 分量
        tmp = mat[0, 0] * a0 + mat[0, 1] * a1       # 新 |0>
        s[:, 1, :] = mat[1, 0] * a0 + mat[1, 1] * a1  # 新 |1>（旧 a0/a1 未动）
        s[:, 0, :] = tmp
        return sv

    def _apply1_batch(self, svs: np.ndarray, q: int, mat: np.ndarray) -> np.ndarray:
        """批量作用 2x2 门：svs (T, 2^n) -> (T, 2^n)，原地、零拷贝 strided 视图。

        把比特 q 置于中间轴的 strided 形状 (T, A, 2, C)，|0>/|1> 分量直接
        做 2x2 线性组合；避免 moveaxis/ascontiguousarray 的整块 gather
        （2^n 数组上的 gather 延迟受限，是旧版单轨迹 31s 的次因）。
        """
        T = svs.shape[0]
        nq = self.n_qubits
        s = svs.reshape(T, 1 << (nq - 1 - q), 2, 1 << q)  # 零拷贝视图
        a0 = s[:, :, 0, :]
        a1 = s[:, :, 1, :]
        tmp = mat[0, 0] * a0 + mat[0, 1] * a1
        a1_new = mat[1, 0] * a0 + mat[1, 1] * a1
        a1[...] = a1_new
        a0[...] = tmp
        return svs

    def _apply_pauli1_batch(self, svs: np.ndarray, q: int, kind: str) -> np.ndarray:
        mat = {"X": _U1["x"], "Y": _U1["y"], "Z": _U1["z"]}[kind]
        return self._apply1_batch(svs, q, mat)

    def _apply_cx(self, sv: np.ndarray, ctl: int, tgt: int) -> np.ndarray:
        """CNOT: 控制 ctl，目 tgt（原地，零拷贝 strided view 实现）。"""
        nq = self.n_qubits
        lo = min(ctl, tgt)
        hi = max(ctl, tgt)
        ctl_high = ctl > tgt
        A = 1 << (nq - 1 - hi)
        B = 1 << (hi - lo - 1)
        C = 1 << lo
        s = sv.reshape(A, 2, B, 2, C)   # 轴 1 = 位 hi，轴 3 = 位 lo
        if ctl_high:
            # ctl 在轴 1：ctl=1 时翻转 tgt（轴 3）
            a = s[:, 1, :, 0, :].copy()  # 保持 (A,B,C) 布局
            s[:, 1, :, 0, :] = s[:, 1, :, 1, :]
            s[:, 1, :, 1, :] = a
        else:
            # tgt 在轴 1，ctl 在轴 3：ctl(轴3)=1 时翻转 tgt(轴1)
            a = s[:, 0, :, 1, :].copy()
            s[:, 0, :, 1, :] = s[:, 1, :, 1, :]
            s[:, 1, :, 1, :] = a
        return sv

    def _apply_cx_batch(self, svs: np.ndarray, ctl: int, tgt: int) -> np.ndarray:
        """批量 CNOT：svs (T, 2^n) -> (T, 2^n)。

        切片拷贝实现：reshape 视图 (T, A, 2, B, 2, C)（ctl 轴在 tgt 轴前），
        仅 1 次 copy + 2 次半数组切片赋值，全部连续访问。
        """
        T = svs.shape[0]
        nq = self.n_qubits
        pc = max(ctl, tgt)  # 高位比特位置
        pt = min(ctl, tgt)  # 低位比特位置
        ctl_high = ctl > tgt  # 控制位是否在高位（轴 2）
        A = 1 << (nq - 1 - pc)
        B = 1 << (pc - pt - 1)
        C = 1 << pt
        m = svs.reshape(T, A, 2, B, 2, C)  # 轴 2 = 位 pc，轴 4 = 位 pt
        out = m.copy()
        if ctl_high:
            # ctl 在轴 2，tgt 在轴 4：ctl=1 时翻转 tgt
            out[..., 1, :, 0, :] = m[..., 1, :, 1, :]
            out[..., 1, :, 1, :] = m[..., 1, :, 0, :]
        else:
            # tgt 在轴 2，ctl 在轴 4：ctl(轴4)=1 时翻转 tgt(轴2)
            out[..., 0, :, 1, :] = m[..., 1, :, 1, :]
            out[..., 1, :, 1, :] = m[..., 0, :, 1, :]
        return out.reshape(T, -1)

    def _apply_swap(self, sv: np.ndarray, a: int, b: int) -> np.ndarray:
        s = sv.reshape((2,) * self.n_qubits)
        return np.swapaxes(s, self._axis(a, self.n_qubits),
                           self._axis(b, self.n_qubits)).reshape(-1)

    def _apply_swap_batch(self, svs: np.ndarray, a: int, b: int) -> np.ndarray:
        T = svs.shape[0]
        s = svs.reshape(T, *((2,) * self.n_qubits))
        s = np.swapaxes(s, self._axis(a, self.n_qubits) + 1,
                        self._axis(b, self.n_qubits) + 1)
        return s.reshape(T, -1)

    # ------------------------------------------------------------------ #
    # 噪声通道（Monte Carlo Kraus 采样）
    # ------------------------------------------------------------------ #
    def _thermal_noise(self, sv: np.ndarray, q: int,
                       time_us: float) -> np.ndarray:
        """
        单比特 T1/T2 热弛豫 = 振幅阻尼 + 相位阻尼 两段 MC 采样（Kraus）。

        振幅阻尼（K1: |1>->sqrt(p)|0> 的跳变概率 = p_reset * P1）：
          - 跳变：|1> 分量搬到 |0> 分量（进入 |0> 态）
          - 无跳变：|1> 分量乘 sqrt(1-p_reset)
        相位阻尼（K1=|1><1| 上跳变概率 = p_z * P1）：
          - 跳变：只保留 |1> 分量
          - 无跳变：|1> 分量乘 sqrt(1-p_z)

        t2 <= 2*t1 保证 p_z >= 0。
        """
        t1 = self.config.t1_times[q]
        t2 = self.config.t2_times[q]

        p_reset = 1.0 - np.exp(-time_us / t1)
        # 相位阻尼概率：需要 t2 <= 2*t1 才非负
        p_z = 1.0 - np.exp(time_us / t1 - 2.0 * time_us / t2)

        nq = self.n_qubits
        s = sv.reshape(1 << (nq - 1 - q), 2, 1 << q)
        a0 = s[:, 0, :]  # |0> 分量（strided view，零拷贝）
        a1 = s[:, 1, :]  # |1> 分量

        # --- 振幅阻尼 ---
        p1 = float(np.sum(np.abs(a1) ** 2))
        if p1 > 0 and self.rng.random() < p_reset * p1:
            a0[:] = a1                              # K1：跳变到 |0>
            a1[:] = 0.0
            n2 = p1                                  # 跳变后 norm²=p1
        else:
            a1 *= np.sqrt(max(1.0 - p_reset, 0.0))  # K0：无跳变
            n2 = 1.0 - p1 * p_reset                  # 解析归一化（省全数组归约）
        sv *= np.sqrt(1.0 / max(n2, 1e-30))

        # ---- 相位阻尼 ---
        p1 = float(np.sum(np.abs(a1) ** 2))
        if p1 > 0 and self.rng.random() < p_z * p1:
            a0[:] = 0.0                                # 只保留 |1> 分量
            n2 = p1
        else:
            a1 *= np.sqrt(max(1.0 - p_z, 0.0))
            n2 = 1.0 - p1 * p_z
        sv *= np.sqrt(1.0 / max(n2, 1e-30))

        return sv

    def _thermal_noise_batch(self, svs: np.ndarray, q: int,
                              time_us: float) -> np.ndarray:
        """
        批量单比特 T1/T2 热弛豫（MC 采样逐轨迹独立，向量化实现）。
        语义与 _thermal_noise 完全一致：每条轨迹独立抛骰子做
        振幅阻尼 + 相位阻尼两段 Kraus 采样。

        性能关键：用零拷贝 strided 视图把比特 q 放到轴 2（形状
        (T, A, 2, C)），直接在原状态向量上做 |0>/|1> 分量运算，
        避免 moveaxis/ascontiguousarray 导致的整块 gather（2^n 数组上的
        gather 是延迟受限、极慢，是旧版单轨迹 31s 的主因）。归一化
        用解析式 norm² = 1 - p1*p（无跳变）或 p1（跳变）。
        """
        T = svs.shape[0]
        t1 = self.config.t1_times[q]
        t2 = self.config.t2_times[q]
        p_reset = 1.0 - np.exp(-time_us / t1)
        p_z = 1.0 - np.exp(time_us / t1 - 2.0 * time_us / t2)  # t2<=2t1 非负
        nq = self.n_qubits
        # 比特 q 置于中间轴的 strided 视图（无拷贝、无 gather）
        s = svs.reshape(T, 1 << (nq - 1 - q), 2, 1 << q)
        a0 = s[:, :, 0, :]
        a1 = s[:, :, 1, :]

        # --- 振幅阻尼 ---
        p1 = np.sum(np.abs(a1.reshape(T, -1)) ** 2, axis=1)  # 连续视图上求和，避免 strided gather
        jump = self.rng.random(T) < p_reset * p1
        if jump.any():
            a0[jump] = a1[jump]
            a1[jump] = 0.0
        nojump = ~jump
        if nojump.any():
            a1[nojump] *= np.sqrt(max(1.0 - p_reset, 0.0))
        n2 = np.where(jump, p1, 1.0 - p1 * p_reset)
        s *= np.sqrt(1.0 / np.maximum(n2, 1e-30)).reshape(T, 1, 1, 1)

        # --- 相位阻尼 ---
        p1 = np.sum(np.abs(a1.reshape(T, -1)) ** 2, axis=1)
        jump = self.rng.random(T) < p_z * p1
        if jump.any():
            a0[jump] = 0.0
        nojump = ~jump
        if nojump.any():
            a1[nojump] *= np.sqrt(max(1.0 - p_z, 0.0))
        n2 = np.where(jump, p1, 1.0 - p1 * p_z)
        s *= np.sqrt(1.0 / np.maximum(n2, 1e-30)).reshape(T, 1, 1, 1)
        return svs

    @staticmethod
    def _renormalize(arr: np.ndarray) -> None:
        """原地归一化任意形状数组（保 a[0] 与 a[1] 子空间的整体范数）。"""
        n = np.sqrt(np.sum(np.abs(arr) ** 2))
        if n > 0:
            arr[...] /= n

    def _depol1(self, sv: np.ndarray, q: int, p: float) -> np.ndarray:
        """单比特退极化（与 qiskit depolarizing_error 一致）：

        恒等概率 = 1 - 3p/4，X/Y/Z 各 p/4。
        """
        if p > 0 and self.rng.random() < 0.75 * p:
            kind = ("X", "Y", "Z")[self.rng.integers(3)]
            return self._apply_pauli1(sv, q, kind)
        return sv

    def _depol1_batch(self, svs: np.ndarray, q: int, p: float) -> np.ndarray:
        """批量单比特退极化：逐轨迹独立采样 X/Y/Z。"""
        T = svs.shape[0]
        if p > 0:
            jump = self.rng.random(T) < 0.75 * p
            idx = np.where(jump)[0]
            if idx.size:
                kinds = self.rng.integers(3, size=idx.size)
                for kind, name in enumerate(("X", "Y", "Z")):
                    sel = idx[kinds == kind]
                    if sel.size:
                        svs[sel] = self._apply_pauli1_batch(svs[sel], q, name)
        return svs

    def _apply_pauli1(self, sv: np.ndarray, q: int, kind: str) -> np.ndarray:
        mat = {"X": _U1["x"], "Y": _U1["y"], "Z": _U1["z"]}[kind]
        return self._apply1(sv, q, mat)

    def _depol2(self, sv: np.ndarray, q1: int, q2: int, p: float) -> np.ndarray:
        """双比特退极化（与 qiskit depolarizing_error 一致）：

        恒等概率 = 1 - 15p/16，15 个非单位 Pauli 各 p/16。
        """
        if p > 0 and self.rng.random() < 15 / 16.0 * p:
            pa, pb = _TWOQ_PAULIS[self.rng.integers(15)]
            if pa != "I":
                sv = self._apply_pauli1(sv, q1, pa)
            if pb != "I":
                sv = self._apply_pauli1(sv, q2, pb)
        return sv

    def _depol2_batch(self, svs: np.ndarray, q1: int, q2: int,
                      p: float) -> np.ndarray:
        """批量双比特退极化：逐轨迹独立采样 15 个 Pauli 组合。"""
        T = svs.shape[0]
        if p > 0:
            jump = self.rng.random(T) < 15 / 16.0 * p
            idx = np.where(jump)[0]
            if idx.size:
                kinds = self.rng.integers(15, size=idx.size)
                for k, (pa, pb) in enumerate(_TWOQ_PAULIS):
                    sel = idx[kinds == k]
                    if sel.size:
                        sub = svs[sel]
                        if pa != "I":
                            sub = self._apply_pauli1_batch(sub, q1, pa)
                        if pb != "I":
                            sub = self._apply_pauli1_batch(sub, q2, pb)
                        svs[sel] = sub
        return svs

    def _zz_xor_indices(self, q1: int, q2: int) -> np.ndarray:
        """返回比特 q1、q2 取值不同的基态索引（exp(-iθ Z⊗Z) 中相位为 e^{+iθ} 的子集）。

        惰性分配 self._all（2^n 整数数组）并按需复用；每对不缓存，避免大 n 下
        字典内存爆炸（调用成本仅一次 O(n) 位运算，相对 70 波演化可忽略）。
        """
        if self._all is None:
            self._all = np.arange(2 ** self.n_qubits)
        b1 = (self._all >> q1) & 1
        b2 = (self._all >> q2) & 1
        return np.nonzero(b1 != b2)[0]

    def _crosstalk(self, sv: np.ndarray, q1: int, q2: int) -> np.ndarray:
        """相干 ZZ 串扰：在交叉对 (q1,q2) 施加 U = exp(-iθ·Z⊗Z)。

        diag(e^{-iθ}, e^{+iθ}, e^{+iθ}, e^{-iθ}) for (00,01,10,11)：先整体乘
        e^{-iθ}，再对 q1⊕q2=1 的基态乘 e^{+2iθ}。

        相比旧的硬 Z 翻转（概率性 Pauli-Z，命中叠加态即正交归零 → 保真度 0/1
        二值），相干旋转是平滑、确定性的小幅损伤（每轨迹保真度 ≈ cos²θ），
        既符合真实 ZZ 串扰物理（小角度相干旋转），也消除 0/1 采样病态、显著降低
        评估所需轨迹数。θ（弧度）由边强度决定，与奖励侧 xtalk 代理同源。
        """
        theta = self._crosstalk_theta(q1, q2)
        if theta != 0.0:
            sv *= np.exp(-1j * theta)
            sv[self._zz_xor_indices(q1, q2)] *= np.exp(2j * theta)
        return sv

    def _crosstalk_batch(self, svs: np.ndarray, q1: int, q2: int) -> np.ndarray:
        """批量相干 ZZ 串扰（svs 形状 (T, 2^n)，按比特轴共享的 xor 索引一次性作用）。"""
        theta = self._crosstalk_theta(q1, q2)
        if theta != 0.0:
            svs *= np.exp(-1j * theta)
            svs[:, self._zz_xor_indices(q1, q2)] *= np.exp(2j * theta)
        return svs

    def _crosstalk_theta(self, q1: int, q2: int) -> float:
        """相干 ZZ 旋转角 θ（弧度）。

        若显式给定 crosstalk_strength，则直接作为 θ（合成拓扑的标称值，量级
        0.005–0.025 rad）；否则回落到 0.1 × 该耦合边的双比特门错误率（与旧
        硬 Z 模型同量级标定，但此处解释为相干旋转角而非破坏概率）。
        """
        if self.config.crosstalk_strength is not None:
            s = self.config.crosstalk_strength.get(
                (q1, q2), self.config.crosstalk_strength.get((q2, q1), 0.0)
            )
        else:
            tqe = self.config.two_q_gate_error
            if isinstance(tqe, dict):
                base = tqe.get((q1, q2), tqe.get((q2, q1), 0.001))
            else:
                base = float(tqe)
            s = 0.1 * base
        return float(s)

    # ------------------------------------------------------------------ #
    # 电路执行
    # ------------------------------------------------------------------ #
    def _evolve(self, circuit: QuantumCircuit, apply_noise: bool) -> np.ndarray:
        """遍历电路指令，返回末态状态向量（无测量）。"""
        sv = self._initial_state()
        if circuit.global_phase:
            sv *= np.exp(1j * float(circuit.global_phase))
        single_time = self.config.single_gate_time
        idle_time = self.config.idle_time

        for inst in circuit.data:
            op = inst.operation
            name = op.name.lower()
            if name == "barrier" or name == "measure":
                continue
            qubits = [q._index for q in inst.qubits]
            if name == "cx":
                ctl, tgt = qubits[0], qubits[1]
                sv = self._apply_cx(sv, ctl, tgt)
                if apply_noise:
                    edge_err = self._two_error(ctl, tgt)
                    sv = self._depol2(sv, ctl, tgt, edge_err)
                    sv = self._crosstalk(sv, ctl, tgt)
            elif name == "swap":
                sv = self._apply_swap(sv, qubits[0], qubits[1])
            elif name == "id":
                if apply_noise:
                    sv = self._thermal_noise(sv, qubits[0], idle_time)
            else:
                # 单比特门
                mat = self._single_qubit_matrix(op, name)
                sv = self._apply1(sv, qubits[0], mat)
                if apply_noise:
                    sv = self._thermal_noise(sv, qubits[0], single_time)
                    sv = self._depol1(sv, qubits[0], self._one_error(qubits[0]))
        return sv

    def _evolve_batch(self, circuit: QuantumCircuit, apply_noise: bool,
                      num_trajectories: int) -> np.ndarray:
        """
        批量演化：一次遍历电路，同时演化 num_trajectories 条轨迹。

        状态形状 (T, 2^n) 单数组，各门/噪声通道对 batch 轴做 numpy 批量操作，
        消除了逐轨迹 Python 循环（P0 向量化优化，20q 下 ~8x 提速）。
        语义与 _evolve 完全一致（噪声 MC 采样逐轨迹独立）。
        """
        svs = np.tile(self._initial_state(), (num_trajectories, 1))
        if circuit.global_phase:
            svs *= np.exp(1j * float(circuit.global_phase))
        single_time = self.config.single_gate_time
        idle_time = self.config.idle_time

        for inst in circuit.data:
            op = inst.operation
            name = op.name.lower()
            if name == "barrier" or name == "measure":
                continue
            qubits = [q._index for q in inst.qubits]
            if name == "cx":
                ctl, tgt = qubits[0], qubits[1]
                svs = self._apply_cx_batch(svs, ctl, tgt)
                if apply_noise:
                    edge_err = self._two_error(ctl, tgt)
                    svs = self._depol2_batch(svs, ctl, tgt, edge_err)
                    svs = self._crosstalk_batch(svs, ctl, tgt)
            elif name == "swap":
                svs = self._apply_swap_batch(svs, qubits[0], qubits[1])
            elif name == "id":
                if apply_noise:
                    svs = self._thermal_noise_batch(svs, qubits[0], idle_time)
            else:
                # 单比特门
                mat = self._gate_matrix(name, op.params)
                svs = self._apply1_batch(svs, qubits[0], mat)
                if apply_noise:
                    svs = self._thermal_noise_batch(svs, qubits[0], single_time)
                    svs = self._depol1_batch(svs, qubits[0],
                                             self._one_error(qubits[0]))
        return svs

    def _one_error(self, q: int) -> float:
        e = self.config.single_q_gate_error
        return float(e[q]) if isinstance(e, (list, tuple)) else float(e)

    # ------------------------------------------------------------------ #
    # 调度感知演化（scheduling-aware）：按波（wave）推进，每比特独立时钟
    # 建模空闲退相干 + 同波不相交双比特门间隔一个耦合边的 1-hop 动态串扰。
    # ------------------------------------------------------------------ #
    def _wave_dynamic_crosstalk_pairs(self, gates) -> List[Tuple[int, int]]:
        """同波内两对不相交双比特门，其交叉比特对若在某耦合边相邻（1-hop），
        则该交叉对承受额外 ZZ 串扰。返回本波所有此类相邻交叉对（去重）。

        仅统计真正不相交的双比特门对：调度器强制同波比特互不相交，故「相邻边
        共享比特」（如 (0,1) 与 (1,2) 同波）在真实调度下不可能出现，无需建模。
        """
        two_q = [(qs[0], qs[1]) for (_, qs, is_2q) in gates if is_2q]
        if len(two_q) < 2:
            return []
        adj = set()
        for (p1, p2) in self.config.coupling_map:
            adj.add((p1, p2))
            adj.add((p2, p1))
        seen = set()
        pairs = []
        for i in range(len(two_q)):
            for j in range(i + 1, len(two_q)):
                a, b = two_q[i]
                c, d = two_q[j]
                for x, y in ((a, c), (a, d), (b, c), (b, d)):
                    if (x, y) in adj:
                        key = (min(x, y), max(x, y))
                        if key not in seen:
                            seen.add(key)
                            pairs.append((x, y))
        return pairs

    @staticmethod
    def _single_qubit_matrix(op, name: str) -> np.ndarray:
        """支持的单比特门用查表/解析式，其余回退到 qiskit Operator 通用展开。"""
        if name in _U1 or name == "rz":
            return _U1[name] if name in _U1 else _rz_matrix(float(op.params[0]))
        from qiskit.quantum_info import Operator
        return np.asarray(Operator(op).data, dtype=complex)

    def evolve_scheduled(self, circuit: QuantumCircuit, waves, apply_noise: bool) -> np.ndarray:
        """按调度波形演化，返回末态状态向量（无测量）。

         与 _evolve（串行、忽略编排）不同，这里：
          - 维护每比特「上次门结束时刻」t_last_end[q]，空闲区间施加热弛豫；
          - 同波内两对不相交双比特门，其交叉比特对若在某耦合边相邻（1-hop），
            在该交叉对上额外施加一次 ZZ 串扰。
         串行调度（每波一门）退化为与 _evolve 一致的噪声（仅门时长退相干），
         并行调度因空闲比特退相干更少 → 保真度更高。
        """
        sv = self._initial_state()
        if circuit.global_phase:
            sv *= np.exp(1j * float(circuit.global_phase))
        single_time = self.config.single_gate_time
        two_time = self.config.two_gate_time
        t_last_end = [0.0] * self.n_qubits
        t = 0.0
        total_time = float(sum(dw for (dw, _g) in waves))

        for (dw, gates) in waves:
            active_qs = set()
            for (_, qs, _is2) in gates:
                for q in qs:
                    active_qs.add(q)
            # 空闲退相干（马尔可夫：exp(-γT) 可复合）统一在「下一次该比特被使用
            # 前」由门前空闲热弛豫施加（见下），无需逐波对每个空闲比特整波施一次；
            # 这样把每波 ~O(n_idle) 次热弛豫降到每门 ~O(1) 次，结果等价且更快。
            # 处理本波各门
            for (phys_idx, qs, is_2q) in gates:
                op = circuit.data[phys_idx].operation
                name = op.name.lower()
                if name == "swap":
                    sv = self._apply_swap(sv, qs[0], qs[1])
                    t_last_end[qs[0]] = t + two_time
                    t_last_end[qs[1]] = t + two_time
                    continue
                if is_2q:
                    ctl, tgt = qs[0], qs[1]
                    gtime = two_time
                else:
                    ctl = tgt = qs[0]
                    gtime = single_time
                # 门前的空闲退相干
                if apply_noise:
                    if t - t_last_end[ctl] > 0:
                        sv = self._thermal_noise(sv, ctl, t - t_last_end[ctl])
                    if is_2q and t - t_last_end[tgt] > 0:
                        sv = self._thermal_noise(sv, tgt, t - t_last_end[tgt])
                if is_2q:
                    sv = self._apply_cx(sv, ctl, tgt)
                    if apply_noise:
                        edge_err = self._two_error(ctl, tgt)
                        sv = self._depol2(sv, ctl, tgt, edge_err)
                        sv = self._crosstalk(sv, ctl, tgt)
                else:
                    mat = self._single_qubit_matrix(op, name)
                    sv = self._apply1(sv, ctl, mat)
                    if apply_noise:
                        sv = self._depol1(sv, ctl, self._one_error(ctl))
                if apply_noise:
                    sv = self._thermal_noise(sv, ctl, gtime)
                    if is_2q:
                        sv = self._thermal_noise(sv, tgt, gtime)
                t_last_end[ctl] = t + gtime
                if is_2q:
                    t_last_end[tgt] = t + gtime
            # 同波内多对不相交双比特门（间隔一个耦合边）的 1-hop 动态 ZZ 串扰
            if apply_noise:
                for (x, y) in self._wave_dynamic_crosstalk_pairs(gates):
                    sv = self._crosstalk(sv, x, y)
            t += dw
        # 收尾：对整段电路末尾（最后一次门之后）仍空闲的比特施加剩余退相干
        if apply_noise:
            for q in range(self.n_qubits):
                idle = total_time - t_last_end[q]
                if idle > 1e-12:
                    sv = self._thermal_noise(sv, q, idle)
        return sv

    def evolve_scheduled_batch(self, svs: np.ndarray, circuit: QuantumCircuit,
                                waves, apply_noise: bool) -> np.ndarray:
        """向量化版 evolve_scheduled：svs 形状 (T, 2^n)，逐波对所有轨迹同时演化。

        每比特时钟（t_last_end）是确定性的（与 MC 采样无关），故可跨轨迹共享；
        门噪声（退极化/串扰/热弛豫）各自逐轨迹独立采样（用 *_batch kernel）。
        """
        T = svs.shape[0]
        single_time = self.config.single_gate_time
        two_time = self.config.two_gate_time
        t_last_end = [0.0] * self.n_qubits
        t = 0.0
        total_time = float(sum(dw for (dw, _g) in waves))
        for (dw, gates) in waves:
            active_qs = set()
            for (_, qs, _is2) in gates:
                for q in qs:
                    active_qs.add(q)
            # 空闲退相干统一在「下一次该比特被使用前」由门前空闲热弛豫施加（见下），
            # 无需逐波对每个空闲比特整波施一次（结果等价，且省去 O(n_idle) 次全数组遍历）。
            for (phys_idx, qs, is_2q) in gates:
                op = circuit.data[phys_idx].operation
                name = op.name.lower()
                if name == "swap":
                    svs = self._apply_swap_batch(svs, qs[0], qs[1])
                    t_last_end[qs[0]] = t + two_time
                    t_last_end[qs[1]] = t + two_time
                    continue
                if is_2q:
                    ctl, tgt = qs[0], qs[1]
                    gtime = two_time
                else:
                    ctl = tgt = qs[0]
                    gtime = single_time
                if apply_noise:
                    if t - t_last_end[ctl] > 0:
                        svs = self._thermal_noise_batch(svs, ctl, t - t_last_end[ctl])
                    if is_2q and t - t_last_end[tgt] > 0:
                        svs = self._thermal_noise_batch(svs, tgt, t - t_last_end[tgt])
                if is_2q:
                    svs = self._apply_cx_batch(svs, ctl, tgt)
                    if apply_noise:
                        edge_err = self._two_error(ctl, tgt)
                        svs = self._depol2_batch(svs, ctl, tgt, edge_err)
                        svs = self._crosstalk_batch(svs, ctl, tgt)
                else:
                    mat = self._single_qubit_matrix(op, name)
                    svs = self._apply1_batch(svs, ctl, mat)
                    if apply_noise:
                        svs = self._depol1_batch(svs, ctl, self._one_error(ctl))
                if apply_noise:
                    svs = self._thermal_noise_batch(svs, ctl, gtime)
                    if is_2q:
                        svs = self._thermal_noise_batch(svs, tgt, gtime)
                t_last_end[ctl] = t + gtime
                if is_2q:
                    t_last_end[tgt] = t + gtime
            # 同波内多对不相交双比特门（间隔一个耦合边）的 1-hop 动态 ZZ 串扰
            if apply_noise:
                for (x, y) in self._wave_dynamic_crosstalk_pairs(gates):
                    svs = self._crosstalk_batch(svs, x, y)
            t += dw
        # 收尾：对整段电路末尾（最后一次门之后）仍空闲的比特施加剩余退相干
        if apply_noise:
            for q in range(self.n_qubits):
                idle = total_time - t_last_end[q]
                if idle > 1e-12:
                    svs = self._thermal_noise_batch(svs, q, idle)
        return svs

    def run_trajectories_scheduled(self, circuit: QuantumCircuit, waves,
                                   num_trajectories: Optional[int] = None,
                                   skip_transpile: bool = False) -> "TrajectoryResult":
        """采样 num_trajectories 条调度感知噪声轨迹。"""
        if num_trajectories is None:
            num_trajectories = self.num_trajectories
        svs = np.array([self._initial_state() for _ in range(num_trajectories)])
        svs = self.evolve_scheduled_batch(svs, circuit, waves, apply_noise=True)
        return TrajectoryResult(svs)

    def fidelity_scheduled(self, circuit: QuantumCircuit, waves,
                           ideal_sv: Optional[np.ndarray] = None,
                           num_trajectories: Optional[int] = None,
                           skip_transpile: bool = False) -> float:
        """调度感知平均态保真度 F = mean_t |<psi_ideal|psi_t>|^2。

        ideal 与调度无关（酉演化与编排无关），故用无噪声串行演化作参考。
        """
        if ideal_sv is None:
            ideal_sv = self._evolve(circuit, apply_noise=False)
        res = self.run_trajectories_scheduled(circuit, waves, num_trajectories,
                                              skip_transpile=True)
        return res.fidelity(ideal_sv)

    def _two_error(self, q1: int, q2: int) -> float:
        e = self.config.two_q_gate_error
        if isinstance(e, dict):
            return float(e.get((q1, q2), e.get((q2, q1), 0.001)))
        return float(e)

    def _ideal_statevector(self, circuit: QuantumCircuit) -> np.ndarray:
        if circuit.num_clbits > 0:
            raise ValueError("_ideal_statevector 要求电路不含测量")
        return self._evolve(circuit, apply_noise=False)

    def ideal_statevector(self, circuit: QuantumCircuit) -> np.ndarray:
        """公开接口：返回电路（ideal, 无噪声）状态向量。"""
        return self._ideal_statevector(circuit)

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    # 批量演化的缓存友好工作集上限（字节）。超过则退回逐轨迹循环：
    # 批量 (T, 2^n) 数组超出 L3 时失去缓存局部性，反而比串行慢。
    _BATCH_WS_LIMIT = 8 << 20  # 8 MiB

    def run_trajectories(self, circuit: QuantumCircuit,
                         num_trajectories: Optional[int] = None,
                         skip_transpile: bool = False) -> TrajectoryResult:
        """采样 num_trajectories 条噪声轨迹，返回末态集合。

        小工作集（n 或 T 较小）用向量化批量演化（(T, 2^n) 单数组，
        消除 Python 循环开销）；大工作集（如 20q × T=16）退回逐轨迹
        循环以保持 L3 缓存局部性（实测批量反而慢 ~2x）。
        """
        if num_trajectories is None:
            num_trajectories = self.num_trajectories
        if not skip_transpile:
            circuit = self._transpile(circuit)
        ws = num_trajectories * (1 << self.n_qubits) * 16  # complex128 字节
        if ws <= self._BATCH_WS_LIMIT:
            svs = self._evolve_batch(circuit, apply_noise=True,
                                     num_trajectories=num_trajectories)
        else:
            svs = np.array(
                [self._evolve(circuit, apply_noise=True)
                 for _ in range(num_trajectories)]
            )
        return TrajectoryResult(svs)

    def run_statevector(self, circuit: QuantumCircuit,
                        num_trajectories: Optional[int] = None,
                        skip_transpile: bool = False):
        """历史兼容名：同 run_trajectories。"""
        return self.run_trajectories(circuit, num_trajectories, skip_transpile)

    def run_and_get_statevector(self, circuit: QuantumCircuit,
                                num_trajectories: Optional[int] = None,
                                skip_transpile: bool = False) -> TrajectoryResult:
        """返回 TrajectoryResult（.data 为平均密度矩阵，.fidelity 也支持）。"""
        return self.run_trajectories(circuit, num_trajectories, skip_transpile)

    def run(self, circuit: QuantumCircuit, shots: Optional[int] = None,
            skip_transpile: bool = False) -> dict:
        """执行电路返回 counts。

        轨迹采样：每个 shot 独立执行一条噪声轨迹后再采样测量结果，
        因此总轨迹数 = shots（代价与 shots 成正比）。
        """
        if shots is None:
            shots = self.config.shots
        if not skip_transpile:
            circuit = self._transpile(circuit)
        measured = sorted({
            q._index
            for inst in circuit.data
            if inst.operation.name.lower() == "measure"
            for q in inst.qubits
        })
        if not measured:
            measured = list(range(self.n_qubits))
        counts: Dict[str, int] = {}
        for _ in range(shots):
            sv = self._evolve(circuit, apply_noise=True)   # 独立轨迹
            outcome = self._sample(sv, measured)
            counts[outcome] = counts.get(outcome, 0) + 1
        return counts

    def _sample(self, sv: np.ndarray, measured: List[int]) -> str:
        """从状态向量按概率采样一个比特串（带读出错误）。"""
        prob = np.abs(sv) ** 2
        idx = self.rng.choice(2 ** self.n_qubits, p=prob)
        bits = [(idx >> q) & 1 for q in measured]
        # 读出错误
        if self.config.readout_error is not None:
            for i, q in enumerate(measured):
                err = self.config.readout_error[q]
                if err > 0 and self.rng.random() < err:
                    bits[i] ^= 1
        return "".join(str(b) for b in reversed(bits))

    def run_batch(self, circuits: List[QuantumCircuit],
                  shots: Optional[int] = None,
                  skip_transpile: bool = False) -> List[dict]:
        return [self.run(c, shots=shots, skip_transpile=skip_transpile) for c in circuits]

    def run_batch_statevector(self, circuits: List[QuantumCircuit],
                              skip_transpile: bool = False) -> List[TrajectoryResult]:
        return [self.run_trajectories(c, skip_transpile=skip_transpile) for c in circuits]

    def fidelity(self, circuit: QuantumCircuit,
                 ideal_sv: Optional[np.ndarray] = None,
                 num_trajectories: Optional[int] = None,
                 skip_transpile: bool = False) -> float:
        """返回电路的平均保真度 F = mean_t |<psi_ideal|psi_t>|^2。"""
        if not skip_transpile:
            circuit = self._transpile(circuit)
        if ideal_sv is None:
            ideal_sv = self._ideal_statevector(circuit)
        res = self.run_trajectories(circuit, num_trajectories, skip_transpile=True)
        return res.fidelity(ideal_sv)


def _reduce_phys_circuit_for_fidelity(phys_circuit: QuantumCircuit,
                                      config: NoiseConfig):
    """将路由后的物理电路 + 噪声配置裁剪到「实际被作用」的量子比特子集。

    返回 (reduced_circuit, reduced_config)，使状态向量模拟仅在 2^{k}（k<=n）
    上进行而非 2^{n}。

    精确性依据：在振幅/相位阻尼 + 退极化 + 相干 ZZ 串扰模型下，从未被作用的比
    特恒为 |0>（|0> 是振幅与相位阻尼的不动点；退极化与串扰只触及被作用比特）。
    因此只模拟被用到的比特并对索引重标号，得到状态相对理想约化态的保真度，等于
    全比特保真度（未用 |0> 比特贡献因子恒为 1）。即本裁剪对保真度**精确等价**，
    不引入近似。
    """
    cfg = config
    n = len(cfg.t1_times)
    used = set()
    for inst in phys_circuit.data:
        oname = inst.operation.name.lower()
        if oname in ("measure", "barrier"):
            continue
        for q in inst.qubits:
            used.add(q._index)
    if len(used) == n:
        return phys_circuit, cfg

    used_list = sorted(used)
    remap = {old: new for new, old in enumerate(used_list)}
    k = len(used_list)
    rc = QuantumCircuit(k)
    rc.global_phase = phys_circuit.global_phase
    for inst in phys_circuit.data:
        oname = inst.operation.name.lower()
        if oname in ("measure", "barrier"):
            continue
        new_qs = [rc.qubits[remap[q._index]] for q in inst.qubits]
        rc.append(inst.operation, new_qs)

    # 子集化逐比特 / 逐边噪声参数（按物理比特索引继承）
    t1 = [cfg.t1_times[i] for i in used_list]
    t2 = [cfg.t2_times[i] for i in used_list]
    freq = [cfg.freq_ghz[i] for i in used_list] if cfg.freq_ghz else None
    sqe = ([cfg.single_q_gate_error[i] for i in used_list]
           if isinstance(cfg.single_q_gate_error, (list, tuple))
           else cfg.single_q_gate_error)
    ro = ([cfg.readout_error[i] for i in used_list]
          if cfg.readout_error is not None else None)

    used_set = set(used_list)
    cm = [(remap[a], remap[b]) for (a, b) in cfg.coupling_map
          if a in used_set and b in used_set]
    # 安全检查：确保 reduced coupling map 覆盖所有 used qubits
    cm_qubits = set()
    for a, b in cm:
        cm_qubits.add(a)
        cm_qubits.add(b)
    if not cm_qubits.issuperset(set(range(k))):
        return phys_circuit, cfg

    tqe = cfg.two_q_gate_error
    if isinstance(tqe, dict):
        new_tqe: dict = {}
        for (a, b), e in tqe.items():
            if a in used_set and b in used_set:
                new_tqe[(remap[a], remap[b])] = e
                new_tqe[(remap[b], remap[a])] = e
    else:
        new_tqe = tqe

    cts = cfg.crosstalk_strength
    if cts is not None:
        new_cts: dict = {}
        for (a, b), e in cts.items():
            if a in used_set and b in used_set:
                new_cts[(remap[a], remap[b])] = e
                new_cts[(remap[b], remap[a])] = e
    else:
        new_cts = None

    reduced = NoiseConfig(
        t1_times=t1,
        t2_times=t2,
        freq_ghz=freq,
        single_q_gate_error=sqe,
        two_q_gate_error=new_tqe,
        coupling_map=cm,
        readout_error=ro,
        crosstalk_strength=new_cts,
        single_gate_time=cfg.single_gate_time,
        two_gate_time=cfg.two_gate_time,
        idle_time=cfg.idle_time,
        shots=cfg.shots,
    )
    return rc, reduced


def trajectory_circuit_fidelity(phys_circuit: QuantumCircuit,
                                config: NoiseConfig,
                                num_trajectories: int = 16,
                                seed: Optional[int] = None,
                                scheduled: bool = False) -> float:
    """对一条物理电路计算轨迹平均态保真度 F = mean_t |<psi_ideal|psi_t>|^2。

    Parameters
    ----------
    phys_circuit : QuantumCircuit
        路由结果（可含 SWAP），内部会 measure_all 后转译。
    config : NoiseConfig
        噪声配置（与 sim.sim.NoiseSimulator 兼容）。
    num_trajectories : int
        采样轨迹条数。
    seed : Optional[int]
        随机种子。
    scheduled : bool
        若为 True，对电路做贪心波次编排（同波内比特不冲突），按调度感知
        演化估算保真度（空闲退相干更少 + 同波相邻双比特门动态串扰）。
    """
    rc, rconfig = _reduce_phys_circuit_for_fidelity(phys_circuit, config)
    sim = TrajectorySimulator(rconfig, num_trajectories=num_trajectories, seed=seed)
    meas = rc.copy()
    meas.measure_all()
    ideal_sv = sim._evolve(meas, apply_noise=False)
    if scheduled:
        waves = schedule_phys_circuit(meas, rconfig.single_gate_time,
                                      rconfig.two_gate_time)
        return sim.fidelity_scheduled(meas, waves, ideal_sv=ideal_sv,
                                      skip_transpile=True)
    meas_no_meas = meas.copy()
    meas_no_meas.remove_final_measurements()
    ideal_sv = sim.ideal_statevector(meas_no_meas)
    return sim.fidelity(meas, ideal_sv=ideal_sv, skip_transpile=True)


def schedule_phys_circuit(phys_circuit: QuantumCircuit, single_time: float, two_time: float):
    """对一条已映射的物理电路做贪心波次编排（同波内比特不冲突）。

    返回 waves 列表，元素 = (dw, [(phys_idx, qubits, is_2q), ...])，与
    evolve_scheduled 的输入兼容（phys_idx 为 phys_circuit.data 的索引，跳过
    measure/barrier）。用于基线（SABRE 等）在公平条件下的调度感知保真度评估。
    """
    waves = []
    used_qubits = set()
    cur = []
    dw = 0.0
    for idx, instr in enumerate(phys_circuit.data):
        op = instr.operation
        name = op.name.lower()
        if name in ("measure", "barrier"):
            continue
        qs = [phys_circuit.find_bit(q).index for q in instr.qubits]
        dur = two_time if len(qs) == 2 else single_time
        if qs[0] in used_qubits or (len(qs) == 2 and qs[1] in used_qubits):
            waves.append((dw, cur))
            used_qubits = set()
            cur = []
            dw = 0.0
        cur.append((idx, tuple(qs), len(qs) == 2))
        used_qubits.update(qs)
        dw = max(dw, dur)
    if cur:
        waves.append((dw, cur))
    return waves


def make_trajectory_fidelity_fn(config: NoiseConfig,
                                num_trajectories: int = 16,
                                seed: Optional[int] = None,
                                scheduled: bool = False):
    """构造 RoutingEnv 的 fidelity_fn hook（闭包捕获当前噪声配置与模拟器）。

    返回 fn(env) -> float：取 env._phys_circuit 计算轨迹平均态保真度。
    训练侧每个 episode 的噪声配置会被扰动，需按 episode 重新构造此函数。

    scheduled=True 时使用调度感知演化（env 须开启 use_scheduler）；若 env
    未提供调度波形（非调度模式），自动回退到串行演化。
    """
    def trajectory_fidelity(env) -> float:
        rc, rconfig = _reduce_phys_circuit_for_fidelity(env._phys_circuit, config)
        sim = TrajectorySimulator(rconfig, num_trajectories=num_trajectories,
                                  seed=seed)
        meas = rc.copy()
        meas.measure_all()
        if scheduled:
            # 转译到基础门后贪心波次编排：与基线 schedule_phys_circuit 一致，
            # 且能正确处理 cz/ecr 等非 cx 双比特门（evolve_scheduled 仅支持 cx/swap）。
            meas_t = sim._transpile(meas)
            meas_t_no_meas = meas_t.copy()
            meas_t_no_meas.remove_final_measurements()
            ideal_sv = sim.ideal_statevector(meas_t_no_meas)
            waves = schedule_phys_circuit(meas_t, rconfig.single_gate_time,
                                         rconfig.two_gate_time)
            return sim.fidelity_scheduled(meas_t, waves, ideal_sv=ideal_sv,
                                          skip_transpile=True)
        meas_no_meas = meas.copy()
        meas_no_meas.remove_final_measurements()
        ideal_sv = sim.ideal_statevector(meas_no_meas)
        return sim.fidelity(meas, ideal_sv=ideal_sv, skip_transpile=True)

    return trajectory_fidelity


def make_analytic_fidelity_fn(config: NoiseConfig,
                               include_thermal: bool = True,
                               include_crosstalk: bool = False):
    """构造解析保真度代理的 fidelity_fn hook（O(门数)，无 2^k 指数）。

    对每条门/噪声信道做一阶错误累积：log F ≈ Σ log(1 - ε_i)，
    其中 ε_i 为各门的退极化错误率 +（可选）热弛豫 +（可选）串扰。
    适用于 16q+ 大电路训练（trajectory_sim 在 20q 上 2^k×门数 成本过高）。

    返回 fn(env) -> float：取 env._phys_circuit 计算解析保真度。
    """
    sqe = config.single_q_gate_error
    tqe = config.two_q_gate_error
    cts = config.crosstalk_strength

    def analytic_fidelity(env) -> float:
        rc, rconfig = _reduce_phys_circuit_for_fidelity(env._phys_circuit, config)
        # 使用裁剪后的噪声参数
        sqe_r = rconfig.single_q_gate_error
        tqe_r = rconfig.two_q_gate_error
        cts_r = rconfig.crosstalk_strength
        logF = 0.0
        for inst in rc.data:
            op = inst.operation
            name = op.name.lower()
            if name in ("measure", "barrier", "id"):
                if name == "id" and include_thermal:
                    # idle 退相干：ε_idle ≈ t/T1·½ + t/T2·½（一阶近似）
                    qs = [q._index for q in inst.qubits]
                    q = qs[0]
                    t1 = rconfig.t1_times[q] if rconfig.t1_times else 50.0
                    t2 = rconfig.t2_times[q] if rconfig.t2_times else 70.0
                    t = rconfig.single_gate_time
                    eps_idle = t / t1 * 0.5 + t / t2 * 0.5
                    logF += math.log(max(1.0 - eps_idle, 1e-10))
                continue
            qs = [q._index for q in inst.qubits]
            if name == "cx":
                a, b = qs
                if isinstance(tqe_r, dict):
                    eps = tqe_r.get((a, b), tqe_r.get((b, a), 0.01))
                else:
                    eps = float(tqe_r)
                logF += math.log(max(1.0 - eps, 1e-10))
            elif name == "swap":
                a, b = qs
                if isinstance(tqe_r, dict):
                    eps = tqe_r.get((a, b), tqe_r.get((b, a), 0.01))
                else:
                    eps = float(tqe_r)
                logF += 3.0 * math.log(max(1.0 - eps, 1e-10))
            else:
                # 单比特门
                q = qs[0]
                if isinstance(sqe_r, (list, tuple)):
                    eps = sqe_r[q] if q < len(sqe_r) else 0.01
                else:
                    eps = float(sqe_r)
                logF += math.log(max(1.0 - eps, 1e-10))
            # 可选：相干 ZZ 串扰（二阶 θ² 惩罚）
            if include_crosstalk and name in ("cx", "swap") and cts_r is not None:
                a, b = qs[:2]
                theta = cts_r.get((a, b), cts_r.get((b, a), 0.0))
                if theta > 0:
                    logF -= theta * theta  # 二阶相干错误近似
            # 可选：门后热弛豫（idle 退相干）
            if include_thermal and name != "id":
                for q in qs:
                    t1 = rconfig.t1_times[q] if rconfig.t1_times else 50.0
                    t2 = rconfig.t2_times[q] if rconfig.t2_times else 70.0
                    t = rconfig.two_gate_time if len(qs) == 2 else rconfig.single_gate_time
                    eps_th = t / t1 * 0.5 + t / t2 * 0.5
                    if eps_th > 1e-10:
                        logF += math.log(max(1.0 - eps_th, 1e-10))
        return max(math.exp(logF), 1e-10)

    return analytic_fidelity


class LocalFidTracker:
    """增量前缀保真度追踪器（local fidelity shaping 用）。

    对 env 已执行的物理电路（env._phys_circuit）做**串行口径**的前缀演化：
    维护 1 条理想态 + K 条噪声轨迹，只对新增指令做增量演化，返回当前
    前缀保真度的 log：logF = log(mean_k |<psi_ideal|psi_k>|^2)。

    - swap 按 eval 端口径展开为 3×CX（(a,b),(b,a),(a,b)），每条 CX 带 depol2+串扰；
    - 2q 门（cx/cz/ecr/swap 外）回退 qiskit Operator 4x4 矩阵，噪声按该边错误率；
    - 1q 门作用 + thermal(depol 单比特错误)（thermal 每门时长一次，与 _evolve 一致）；
    - 惰性维度：首次 update 收集全部已用物理比特并 reduce 重映射；
      后续若出现新物理比特则全量重放（重置），否则增量演化。

    Attributes
    ----------
    n_phys_used : int         当前维度 k（状态向量 2^k）
    logF : float              最近一次 update 后的 log 前缀保真度
    """

    def __init__(self, config: NoiseConfig, num_trajectories: int = 4,
                 seed: Optional[int] = None, swap_as_3cx: bool = True,
                 phys_qubits: Optional[List[int]] = None):
        self.base_config = config
        self.num_trajectories = num_trajectories
        self.seed = seed
        self.swap_as_3cx = swap_as_3cx
        self._phys_qubits = phys_qubits   # 固定物理比特集（None=惰性收集）
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self):
        self._sim = None            # TrajectorySimulator（reduce config，k 维）
        self._used_phys: List[int] = []
        self._remap: dict = {}
        self._n_evolved = 0         # 已演化 phys_circuit.data 指令数
        self.sv_ideal = None
        self.svs_noisy = None
        self.logF = 0.0
        self._rng = np.random.default_rng(self.seed)

    def _collect_used(self, phys_circuit) -> List[int]:
        used = []
        for inst in phys_circuit.data:
            name = inst.operation.name.lower()
            if name in ("measure", "barrier"):
                continue
            for q in inst.qubits:
                qi = q._index
                if qi not in used:
                    used.append(qi)
        return sorted(used)

    def _build_sim(self, used_phys: List[int]):
        """按已用物理比特子集构建 reduced TrajectorySimulator。"""
        cfg = self.base_config
        n = len(cfg.t1_times)
        used = used_phys if used_phys else list(range(n))
        self._used_phys = list(used)
        self._remap = {old: new for new, old in enumerate(used)}
        k = len(used)
        t1 = [cfg.t1_times[i] for i in used]
        t2 = [cfg.t2_times[i] for i in used]
        freq = [cfg.freq_ghz[i] for i in used] if cfg.freq_ghz else None
        sqe = ([cfg.single_q_gate_error[i] for i in used]
               if isinstance(cfg.single_q_gate_error, (list, tuple))
               else cfg.single_q_gate_error)
        ro = ([cfg.readout_error[i] for i in used]
              if cfg.readout_error is not None else None)
        used_set = set(used)
        cm = [(self._remap[a], self._remap[b]) for (a, b) in cfg.coupling_map
              if a in used_set and b in used_set]
        tqe = cfg.two_q_gate_error
        if isinstance(tqe, dict):
            new_tqe = {}
            for (a, b), e in tqe.items():
                if a in used_set and b in used_set:
                    new_tqe[(self._remap[a], self._remap[b])] = e
                    new_tqe[(self._remap[b], self._remap[a])] = e
        else:
            new_tqe = tqe
        cts = cfg.crosstalk_strength
        if cts is not None:
            new_cts = {}
            for (a, b), e in cts.items():
                if a in used_set and b in used_set:
                    new_cts[(self._remap[a], self._remap[b])] = e
                    new_cts[(self._remap[b], self._remap[a])] = e
        else:
            new_cts = None
        rcfg = NoiseConfig(
            t1_times=t1, t2_times=t2, freq_ghz=freq,
            single_q_gate_error=sqe, two_q_gate_error=new_tqe,
            coupling_map=cm, readout_error=ro, crosstalk_strength=new_cts,
            single_gate_time=cfg.single_gate_time,
            two_gate_time=cfg.two_gate_time,
            idle_time=cfg.idle_time, shots=cfg.shots,
        )
        self._sim = TrajectorySimulator(rcfg, num_trajectories=self.num_trajectories,
                                        seed=None)
        self._sim.rng = self._rng          # 与 tracker 共享 rng
        k = len(used)
        self.sv_ideal = self._sim._initial_state()
        self.svs_noisy = np.array(
            [self._sim._initial_state() for _ in range(self.num_trajectories)])

    def _apply_gate(self, inst) -> None:
        """对理想态 + 噪声轨迹应用一条指令（含噪声），并更新 t_last。"""
        op = inst.operation
        name = op.name.lower()
        if name in ("measure", "barrier"):
            return
        qs = [self._remap[q._index] for q in inst.qubits]
        sim = self._sim
        # 理想态：纯酉
        self._apply_unitary(self.sv_ideal, name, op, qs, noise=False)
        # 噪声轨迹
        for i in range(self.num_trajectories):
            self._apply_unitary(self.svs_noisy[i], name, op, qs, noise=True)

    def _apply_unitary(self, sv, name, op, qs, noise: bool) -> None:
        sim = self._sim
        if name == "swap" and self.swap_as_3cx:
            # swap = cx(a,b); cx(b,a); cx(a,b)
            a, b = qs[0], qs[1]
            self._cx_evolve(sv, a, b, noise)
            self._cx_evolve(sv, b, a, noise)
            self._cx_evolve(sv, a, b, noise)
            return
        if len(qs) == 2:
            if name == "cx":
                self._cx_evolve(sv, qs[0], qs[1], noise)
            else:
                # cz/ecr 等：qiskit Operator 4x4
                from qiskit.quantum_info import Operator
                mat = np.asarray(Operator(op).data, dtype=complex)
                sim._apply1q_matrix_2q(sv, qs[0], qs[1], mat) if hasattr(
                    sim, "_apply1q_matrix_2q") else self._apply2_general(
                    sv, qs[0], qs[1], mat)
                if noise:
                    err = sim._two_error(qs[0], qs[1])
                    sv = sim._depol2(sv, qs[0], qs[1], err)
                    sv = sim._crosstalk(sv, qs[0], qs[1])
        else:
            q = qs[0]
            mat = sim._single_qubit_matrix(op, name)
            sv = sim._apply1(sv, q, mat)
            if noise:
                sv = sim._thermal_noise(sv, q, sim.config.single_gate_time)
                sv = sim._depol1(sv, q, sim._one_error(q))

    def _cx_evolve(self, sv, ctl, tgt, noise: bool) -> None:
        sim = self._sim
        sim._apply_cx(sv, ctl, tgt)
        if noise:
            err = sim._two_error(ctl, tgt)
            sv = sim._depol2(sv, ctl, tgt, err)
            sv = sim._crosstalk(sv, ctl, tgt)

    @staticmethod
    def _apply2_general(sv, a, b, mat):
        """作用 4x4 双比特酉到 (a,b)，little-endian 索引。"""
        n = int(round(np.log2(sv.size)))
        lo, hi = sorted((a, b))
        # 构造 4 个 2-bit 逻辑索引段：高位 hi 在 axis hi，低位 lo 在 axis lo
        A = 1 << (n - 1 - hi)
        Bm = 1 << (hi - lo - 1)
        C = 1 << lo
        s = sv.reshape(A, 2, Bm, 2, C)   # axis1=hi, axis3=lo
        # 4 组合 (hi,lo) → 目标段
        comb = np.empty((A, 2, Bm, 2, C), dtype=complex)
        comb[:, 0, :, 0, :] = mat[0, 0] * s[:, 0, :, 0, :] + mat[0, 1] * s[:, 0, :, 1, :] \
                              + mat[0, 2] * s[:, 1, :, 0, :] + mat[0, 3] * s[:, 1, :, 1, :]
        comb[:, 0, :, 1, :] = mat[1, 0] * s[:, 0, :, 0, :] + mat[1, 1] * s[:, 0, :, 1, :] \
                              + mat[1, 2] * s[:, 1, :, 0, :] + mat[1, 3] * s[:, 1, :, 1, :]
        comb[:, 1, :, 0, :] = mat[2, 0] * s[:, 0, :, 0, :] + mat[2, 1] * s[:, 0, :, 1, :] \
                              + mat[2, 2] * s[:, 1, :, 0, :] + mat[2, 3] * s[:, 1, :, 1, :]
        comb[:, 1, :, 1, :] = mat[3, 0] * s[:, 0, :, 0, :] + mat[3, 1] * s[:, 0, :, 1, :] \
                              + mat[3, 2] * s[:, 1, :, 0, :] + mat[3, 3] * s[:, 1, :, 1, :]
        s[...] = comb

    # ------------------------------------------------------------------ #
    def update(self, env) -> float:
        """增量消化 env._phys_circuit 中新增指令，返回 log 前缀保真度。

        维度固定为 phys_qubits（构造时给定）或首次 update 时惰性收集的全部
        已用物理比特。此后不扩维（避免 rng 序列错位导致增量≠全量）。
        """
        pc = env._phys_circuit
        if self._sim is None:
            if self._phys_qubits is not None:
                used = sorted(self._phys_qubits)
            else:
                used = self._collect_used(pc) or list(range(len(self.base_config.t1_times)))
            self._build_sim(used)
            self._n_evolved = 0
        data = pc.data
        while self._n_evolved < len(data):
            inst = data[self._n_evolved]
            if inst.operation.name.lower() not in ("measure", "barrier"):
                # 防御：遇到未映射物理比特（不应发生），跳过并复位
                qi = [q._index for q in inst.qubits]
                if any(q not in self._remap for q in qi):
                    self._n_evolved += 1
                    continue
            self._apply_gate(inst)
            self._n_evolved += 1
        if self.sv_ideal is None:
            self.logF = 0.0
            return 0.0
        # F = mean_k |<ideal|noisy_k>|^2
        inner = self.svs_noisy @ self.sv_ideal.conj()
        F = float(np.clip(np.mean(np.abs(inner) ** 2), 1e-12, 1.0))
        self.logF = float(np.log(F))
        return self.logF