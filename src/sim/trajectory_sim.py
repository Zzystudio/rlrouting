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
from typing import Dict, List, Optional, Tuple

import numpy as np

from qiskit import QuantumCircuit, transpile

from sim.sim import NoiseConfig

logger = logging.getLogger(__name__)

# 平均密度矩阵内存上限：超过则禁止 .data（建议改用 TrajectoryResult.fidelity）
_DENSITY_MEM_LIMIT = 1 << 31  # 2 GiB

# 单比特相对旋转（rz）出参用
def _rz_matrix(theta: float) -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, np.exp(1j * theta)]], dtype=complex)


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

    def __init__(self, config: NoiseConfig, num_trajectories: int = 64,
                 seed: Optional[int] = None):
        self.config = config
        self.num_trajectories = num_trajectories
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.n_qubits = len(config.t1_times)
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
        """作用任意 2x2 门到比特 q（qiskit little-endian 顺序）。"""
        sv = sv.reshape((2,) * self.n_qubits)
        moved = np.ascontiguousarray(np.moveaxis(sv, q, 0))  # (2, rest...)
        out = mat @ moved.reshape(2, -1)
        out = np.moveaxis(out.reshape((2,) * self.n_qubits), 0, q)
        return out.reshape(-1)

    def _apply_cx(self, sv: np.ndarray, ctl: int, tgt: int) -> np.ndarray:
        """CNOT: 控制 ctl，目 tgt。"""
        s = sv.reshape((2,) * self.n_qubits)
        s = np.moveaxis(s, (ctl, tgt), (0, 1))
        s = np.ascontiguousarray(s)
        m = s.reshape(2, 2, -1)
        # 当 ctl=1 时翻转 tgt
        tmp = m[1].copy()
        m[1, 0] = tmp[1]
        m[1, 1] = tmp[0]
        out = np.moveaxis(s, (0, 1), (ctl, tgt))
        return out.reshape(-1)

    def _apply_swap(self, sv: np.ndarray, a: int, b: int) -> np.ndarray:
        s = sv.reshape((2,) * self.n_qubits)
        return np.swapaxes(s, a, b).reshape(-1)

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

        s = sv.reshape((2,) * self.n_qubits)
        moved = np.ascontiguousarray(np.moveaxis(s, q, 0))
        a = moved.reshape(2, -1)     # a[0]=|0> 分量，a[1]=|1> 分量（视图）

        # --- 振幅阻尼 ---
        p1 = float(np.sum(np.abs(a[1]) ** 2))
        if p1 > 0 and self.rng.random() < p_reset * p1:
            a[0] = a[1]                              # K1：跳变到 |0>
            a[1] = 0.0
        else:
            a[1] *= np.sqrt(max(1.0 - p_reset, 0.0))  # K0：无跳变
        self._renormalize(moved)

        # ---- 相位阻尼 ---
        p1 = float(np.sum(np.abs(a[1]) ** 2))
        if p1 > 0 and self.rng.random() < p_z * p1:
            a[0] = 0.0                                # 只保留 |1> 分量
        else:
            a[1] *= np.sqrt(max(1.0 - p_z, 0.0))
        self._renormalize(moved)

        return np.moveaxis(moved, 0, q).reshape(-1)

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

    def _crosstalk(self, sv: np.ndarray, q1: int, q2: int) -> np.ndarray:
        """串扰：在两比特上施加 ZZ 错误（概率由 crosstalk_strength 决定）。"""
        if self.config.crosstalk_strength is not None:
            p = self.config.crosstalk_strength.get(
                (q1, q2), self.config.crosstalk_strength.get((q2, q1), 0.0)
            )
        else:
            # 默认强度 = 0.1 * 双比特门错误率（与 sim.py 一致）
            tqe = self.config.two_q_gate_error
            if isinstance(tqe, dict):
                vals = [v for k, v in tqe.items() if k[0] < k[1]]
                base = float(np.mean(vals)) if vals else 0.001
            else:
                base = float(tqe)
            p = 0.1 * base
        if p > 0 and self.rng.random() < p:
            # ZZ = Z1 * Z2
            sv = self._apply_pauli1(sv, q1, "Z")
            sv = self._apply_pauli1(sv, q2, "Z")
        return sv

    # ------------------------------------------------------------------ #
    # 电路执行
    # ------------------------------------------------------------------ #
    def _evolve(self, circuit: QuantumCircuit, apply_noise: bool) -> np.ndarray:
        """遍历电路指令，返回末态状态向量（无测量）。"""
        sv = self._initial_state()
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
                mat = self._gate_matrix(name, op.params)
                sv = self._apply1(sv, qubits[0], mat)
                if apply_noise:
                    sv = self._thermal_noise(sv, qubits[0], single_time)
                    sv = self._depol1(sv, qubits[0], self._one_error(qubits[0]))
        return sv

    def _one_error(self, q: int) -> float:
        e = self.config.single_q_gate_error
        return float(e[q]) if isinstance(e, (list, tuple)) else float(e)

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
    def run_trajectories(self, circuit: QuantumCircuit,
                         num_trajectories: Optional[int] = None,
                         skip_transpile: bool = False) -> TrajectoryResult:
        """采样 num_trajectories 条噪声轨迹，返回末态集合。"""
        if num_trajectories is None:
            num_trajectories = self.num_trajectories
        if not skip_transpile:
            circuit = self._transpile(circuit)
        svs = np.empty((num_trajectories, 2 ** self.n_qubits), dtype=complex)
        for t in range(num_trajectories):
            svs[t] = self._evolve(circuit, apply_noise=True)
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