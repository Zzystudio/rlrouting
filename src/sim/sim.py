# ============================================================================
# noise_simulator.py
# 基于 Qiskit Aer 的增强噪声模拟器框架
# 支持：单比特相干时间(T1, T2)、单/双比特门错误率、拓扑耦合图、读出错误、
#       串扰噪声（基于拓扑的ZZ串扰）
# 错误传播通过密度矩阵模拟器自然实现
# ============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    thermal_relaxation_error,
    depolarizing_error,
    pauli_error,
    ReadoutError,
)


# ----------------------------- 配置类 ---------------------------------------
@dataclass
class NoiseConfig:
    """
    噪声模拟器的配置参数。
    所有时间单位均为微秒 (µs)。
    """
    # 单比特参数（按量子比特索引顺序）
    t1_times: List[float]               # 每个量子比特的T1弛豫时间
    t2_times: List[float]               # 每个量子比特的T2退相干时间
    freq_ghz: List[float]               # 每个量子比特的工作频率 (GHz) —— 目前仅作预留

    # 门错误率
    # single_q_gate_error:  float 表示所有比特相同的错误率,
    #                       List[float] 表示每个比特独立错误率.
    single_q_gate_error: Union[float, List[float]]
    # two_q_gate_error:  float 表示所有耦合边相同的错误率,
    #                    Dict[(int,int),float] 表示每条边独立错误率,
    #                    dict 中应包含 coupling_map 中每条边的正反两个方向.
    two_q_gate_error: Union[float, Dict[Tuple[int, int], float]]

    # 拓扑结构（耦合图）
    coupling_map: List[Tuple[int, int]] # 例如 [(0,1), (1,2), (2,3)] 表示线性相邻

    # 读出错误 (可选)
    readout_error: Optional[List[float]] = None  # 每个比特的读出错误概率，若None则使用默认值

    # 串扰相关参数 (可选)
    # 如果未提供，将基于 two_q_gate_error 和 coupling_map 自动生成一个默认串扰强度
    crosstalk_strength: Optional[Dict[Tuple[int, int], float]] = None
    # 键为 (q1, q2)，值为该比特对之间的串扰错误概率 (例如ZZ错误概率)

    # 门执行时间 (用于热弛豫计算)
    single_gate_time: float = 0.1       # 单比特门持续时间 (µs)
    two_gate_time: float = 0.3          # 双比特门持续时间 (µs)
    idle_time: float = 0.1              # 空闲等待时间 (µs) —— 用于模拟等待时的退相干

    # 模拟器附加选项
    shots: int = 1024                   # 默认采样次数
    parallel: bool = True               # 是否启用多线程并行
    max_parallel_threads: int = 0       # 0 = 使用 CPU 核数
    device: str = 'CPU'                 # 模拟设备: 'CPU' 或 'GPU' (需要 GPU 版 qiskit-aer)


# ----------------------------- 模拟器类 ---------------------------------------
class NoiseSimulator:
    """
    增强型噪声模拟器。根据 NoiseConfig 构建一个含串扰、热弛豫、门错误和读出错误的噪声模型，
    并提供执行量子电路的方法。
    """

    def __init__(self, config: NoiseConfig):
        self.config = config
        self._validate_config()
        self.noise_model = self._build_noise_model()
        self._simulator = None  # 懒加载

    def _validate_config(self):
        """简单验证配置的合理性"""
        n_qubits = len(self.config.t1_times)
        if len(self.config.t2_times) != n_qubits:
            raise ValueError("t1_times 和 t2_times 长度必须一致")
        if len(self.config.freq_ghz) != n_qubits:
            raise ValueError("freq_ghz 长度必须与量子比特数一致")
        if self.config.readout_error is not None:
            if len(self.config.readout_error) != n_qubits:
                raise ValueError("readout_error 长度必须与量子比特数一致")

    # ========== 构建噪声模型 =================================================
    def _build_noise_model(self) -> NoiseModel:
        logger = logging.getLogger('qiskit_aer.noise.noise_model')
        prev_level = logger.level
        logger.setLevel(logging.ERROR)
        try:
            noise_model = NoiseModel()
            self._add_combined_gate_errors(noise_model)
            self._add_crosstalk(noise_model)
            self._add_readout_error(noise_model)
            return noise_model
        finally:
            logger.setLevel(prev_level)

    def _add_combined_gate_errors(self, noise_model: NoiseModel):
        """为每个比特组合退极化 + 热弛豫，作为一个量子错误添加到该比特。"""
        n_qubits = len(self.config.t1_times)
        single_gates = ['rz', 'sx', 'x', 'y', 'z', 'h']

        sqe = self.config.single_q_gate_error
        if isinstance(sqe, (list, tuple)):
            single_depols = [depolarizing_error(float(sqe[i]), 1) for i in range(n_qubits)]
        else:
            single_depols = [depolarizing_error(float(sqe), 1)] * n_qubits

        for i in range(n_qubits):
            thermal = thermal_relaxation_error(
                t1=self.config.t1_times[i],
                t2=self.config.t2_times[i],
                time=self.config.single_gate_time,
            )
            combined = single_depols[i].compose(thermal)
            noise_model.add_quantum_error(combined, single_gates + ['s', 't'], [i])

            idle_thermal = thermal_relaxation_error(
                t1=self.config.t1_times[i],
                t2=self.config.t2_times[i],
                time=self.config.idle_time,
            )
            noise_model.add_quantum_error(idle_thermal, ['id'], [i])

        tqe = self.config.two_q_gate_error
        if isinstance(tqe, dict):
            for (q1, q2), err in tqe.items():
                if q1 >= n_qubits or q2 >= n_qubits:
                    continue
                if err > 0:
                    cx_depol = depolarizing_error(err, 2)
                    noise_model.add_quantum_error(cx_depol, ['cx'], [q1, q2])
        elif tqe > 0:
            cx_depol = depolarizing_error(tqe, 2)
            noise_model.add_all_qubit_quantum_error(cx_depol, ['cx'])

    def _add_crosstalk(self, noise_model: NoiseModel):
        n_qubits = len(self.config.t1_times)
        if self.config.crosstalk_strength is None:
            tqe = self.config.two_q_gate_error
            if isinstance(tqe, dict):
                vals = [v for k, v in tqe.items() if k[0] < k[1]]
                default_strength = 0.1 * float(np.mean(vals)) if vals else 0.001
            else:
                default_strength = 0.1 * tqe
            crosstalk_map = {}
            for q1, q2 in self.config.coupling_map:
                if (q2, q1) not in crosstalk_map:
                    crosstalk_map[(q1, q2)] = default_strength
        else:
            crosstalk_map = self.config.crosstalk_strength

        for (q1, q2), strength in crosstalk_map.items():
            if q1 >= n_qubits or q2 >= n_qubits:
                raise ValueError(f"串扰涉及非法量子比特索引: ({q1}, {q2})")
            if strength <= 0:
                continue
            zz_error = pauli_error([('ZZ', strength), ('II', 1 - strength)])
            if isinstance(self.config.two_q_gate_error, dict):
                cx_err = self.config.two_q_gate_error.get((q1, q2), 0.01)
                cx_depol = depolarizing_error(cx_err, 2)
            elif hasattr(self, '_cx_error_base'):
                cx_depol = self._cx_error_base
            else:
                cx_depol = None
            combined = cx_depol.compose(zz_error) if cx_depol is not None else zz_error
            noise_model.add_quantum_error(combined, ['cx'], [q1, q2])
            noise_model.add_quantum_error(combined, ['cx'], [q2, q1])

    def _add_readout_error(self, noise_model: NoiseModel):
        """添加读出错误"""
        n_qubits = len(self.config.t1_times)
        if self.config.readout_error is None:
            # 默认 2% 的读出错误
            readout_errors = [0.02] * n_qubits
        else:
            readout_errors = self.config.readout_error

        for i, err in enumerate(readout_errors):
            # 读出错误矩阵: [[P(0|0), P(1|0)], [P(0|1), P(1|1)]]
            # 假设对称错误: P(0|1) = P(1|0) = err
            error = ReadoutError([[1 - err, err], [err, 1 - err]])
            noise_model.add_readout_error(error, [i])

    # ========== 模拟器执行 =================================================
    def get_simulator(self) -> AerSimulator:
        """获取或创建 AerSimulator 实例（懒加载）"""
        if self._simulator is None:
            kwargs = dict(
                noise_model=self.noise_model,
                basis_gates=self.noise_model.basis_gates,
                coupling_map=self.config.coupling_map,
                method='density_matrix',
                device=self.config.device,
            )
            if self.config.parallel:
                kwargs['max_parallel_threads'] = self.config.max_parallel_threads or 0
            self._simulator = AerSimulator(**kwargs)
        return self._simulator

    def _transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """将电路转译到基础门和拓扑"""
        return transpile(
            circuit,
            basis_gates=self.noise_model.basis_gates,
            coupling_map=[list(e) for e in self.config.coupling_map],
            optimization_level=1,
        )

    def run(self, circuit: QuantumCircuit, shots: Optional[int] = None,
            skip_transpile: bool = False) -> dict:
        """
        执行电路并返回计数结果 (counts)
        自动根据耦合图进行转译（transpile）以确保电路符合拓扑。
        若 circuit 已转译过，设置 skip_transpile=True 跳过重复转译。
        """
        if shots is None:
            shots = self.config.shots
        if not skip_transpile:
            circuit = self._transpile(circuit)
        simulator = self.get_simulator()
        job = simulator.run(circuit, shots=shots)
        result = job.result()
        return result.get_counts()

    def run_and_get_counts(self, circuit: QuantumCircuit,
                           shots: Optional[int] = None) -> dict:
        """同 run，返回 counts"""
        return self.run(circuit, shots)

    def run_and_get_statevector(self, circuit: QuantumCircuit,
                                skip_transpile: bool = False):
        """
        返回电路的密度矩阵（电路不应包含测量操作）。
        若 circuit 已转译过，设置 skip_transpile=True 跳过重复转译。
        """
        if circuit.num_clbits > 0:
            raise ValueError("run_and_get_statevector 要求电路不含测量操作")
        if not skip_transpile:
            circuit = self._transpile(circuit)
        simulator = self.get_simulator()
        circ = circuit.copy()
        circ.save_density_matrix()
        job = simulator.run(circ, shots=1)
        result = job.result()
        return result.data()['density_matrix']

    def run_batch(self, circuits: List[QuantumCircuit],
                  shots: Optional[int] = None,
                  skip_transpile: bool = False) -> List[dict]:
        """
        批量执行多个电路并返回计数结果。
        利用 AerSimulator 内部的并行能力，比逐个调用 run() 快得多。
        """
        if shots is None:
            shots = self.config.shots
        if not skip_transpile:
            circuits = [self._transpile(c) for c in circuits]
        simulator = self.get_simulator()
        job = simulator.run(circuits, shots=shots)
        result = job.result()
        return [result.get_counts(i) for i in range(len(circuits))]

    def run_batch_statevector(self, circuits: List[QuantumCircuit],
                              skip_transpile: bool = False):
        """
        批量计算多个无测量电路的密度矩阵。
        """
        if not skip_transpile:
            circuits = [self._transpile(c) for c in circuits]
        simulator = self.get_simulator()
        circs = []
        for c in circuits:
            if c.num_clbits > 0:
                raise ValueError("run_batch_statevector 要求电路不含测量操作")
            cc = c.copy()
            cc.save_density_matrix()
            circs.append(cc)
        job = simulator.run(circs, shots=1)
        result = job.result()
        return [result.data(i)['density_matrix'] for i in range(len(circuits))]


# ============================ 使用示例 ========================================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    # 1. 配置参数（以 3 比特线性拓扑为例）
    config = NoiseConfig(
        t1_times=[50.0, 50.0, 50.0],          # µs
        t2_times=[70.0, 70.0, 70.0],
        freq_ghz=[5.0, 5.0, 5.0],
        single_q_gate_error=0.001,            # 0.1%
        two_q_gate_error=0.01,                # 1%
        coupling_map=[(0, 1), (1, 2)],
        readout_error=[0.02, 0.02, 0.02],     # 2%
        single_gate_time=0.1,
        two_gate_time=0.3,
        idle_time=0.1,
        shots=2048,
    )

    # 2. 创建模拟器
    sim = NoiseSimulator(config)

    # 3. 构造一个示例电路 (GHZ 态)
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()

    # 4. 运行并打印结果
    counts = sim.run(qc)
    print("模拟结果:", counts)

    # 5. 查看噪声模型的基本信息（可选）
    print("噪声模型包含以下噪声指令:")
    for gate in sim.noise_model.noise_instructions:
        print(f"  - {gate}")
    print("噪声量子比特:")
    for qubit in sim.noise_model.noise_qubits:
        print(f"  - {qubit}")