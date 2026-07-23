# ============================================================================
# noise_simulator.py
# 基于 Qiskit Aer 的增强噪声模拟器框架
# 支持：单比特相干时间(T1, T2)、单/双比特门错误率、拓扑耦合图、读出错误、
#       串扰噪声（基于拓扑的ZZ串扰）
# 错误传播通过密度矩阵模拟器自然实现
# ============================================================================

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
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
from qiskit_aer import Aer  # 用于获取 backends


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

    # 门错误率 (平均错误概率)
    single_q_gate_error: float          # 单比特门（如 U1,U2,U3,SX,RZ 等）的错误率
    two_q_gate_error: float             # 双比特门（通常是 CNOT）的错误率

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
        """根据配置构建完整的 NoiseModel"""
        noise_model = NoiseModel()

        # 1. 热弛豫噪声 (T1, T2)
        self._add_thermal_relaxation(noise_model)

        # 2. 门错误 (退极化)
        self._add_gate_errors(noise_model)

        # 3. 串扰噪声 (基于拓扑的ZZ串扰)
        self._add_crosstalk(noise_model)

        # 4. 读出错误
        self._add_readout_error(noise_model)

        return noise_model

    def _add_thermal_relaxation(self, noise_model: NoiseModel):
        """为所有量子比特添加热弛豫错误"""
        n_qubits = len(self.config.t1_times)
        # 单比特门的热弛豫
        for i in range(n_qubits):
            error = thermal_relaxation_error(
                t1=self.config.t1_times[i],
                t2=self.config.t2_times[i],
                time=self.config.single_gate_time,
            )
            # 这些是 Qiskit 中常用的单比特门
            noise_model.add_quantum_error(error, ['rz', 'sx', 'x', 'y', 'z', 'h'], [i])
            # 空闲时间等待也会引入退相干
            idle_error = thermal_relaxation_error(
                t1=self.config.t1_times[i],
                t2=self.config.t2_times[i],
                time=self.config.idle_time,
            )
            noise_model.add_quantum_error(idle_error, ['id'], [i])

        # 双比特门的热弛豫通过 depolarizing_error 在 _add_gate_errors 中处理
        # 这里不重复添加，避免双重计数

    def _add_gate_errors(self, noise_model: NoiseModel):
        """添加退极化错误表示门操作错误"""
        # 单比特门错误
        single_error = depolarizing_error(self.config.single_q_gate_error, 1)
        # 应用到常用单比特门
        noise_model.add_all_qubit_quantum_error(single_error, ['rz', 'sx', 'x', 'y', 'z', 'h', 's', 't'])

        # 双比特门错误（通常只针对 CNOT）
        if self.config.two_q_gate_error > 0:
            two_error = depolarizing_error(self.config.two_q_gate_error, 2)
            noise_model.add_all_qubit_quantum_error(two_error, ['cx'])

    def _add_crosstalk(self, noise_model: NoiseModel):
        """
        添加基于拓扑的串扰噪声。采用 ZZ 耦合作为典型的串扰模型。
        如果配置中未提供串扰强度字典，则根据双比特门错误率和耦合图自动生成默认值。
        """
        n_qubits = len(self.config.t1_times)
        # 确定串扰强度
        if self.config.crosstalk_strength is None:
            # 默认：每条边上的串扰概率 = 双比特门错误率的 10%
            default_strength = 0.1 * self.config.two_q_gate_error
            crosstalk_map = {}
            for q1, q2 in self.config.coupling_map:
                # 去重：如果同时包含 (q1,q2) 和 (q2,q1)，只保留一个
                if (q2, q1) not in crosstalk_map:
                    crosstalk_map[(q1, q2)] = default_strength
                # 如果已存在反向，则忽略（因为无向）
        else:
            crosstalk_map = self.config.crosstalk_strength

        # 为每对相邻比特添加 ZZ 串扰错误
        for (q1, q2), strength in crosstalk_map.items():
            if q1 >= n_qubits or q2 >= n_qubits:
                raise ValueError(f"串扰涉及非法量子比特索引: ({q1}, {q2})")
            if strength <= 0:
                continue
            # 构造 Pauli 错误: 以概率 strength 施加 ZZ 门
            # 注意：pauli_error 需要概率和对应的 Pauli 字符串
            # 这里使用 'II' 表示无错误，'ZZ' 表示 ZZ 错误
            zz_error = pauli_error([('ZZ', strength), ('II', 1 - strength)])
            # 将该错误添加到 CNOT 门上（表示在执行 CNOT 时受到串扰）
            noise_model.add_quantum_error(zz_error, ['cx'], [q1, q2])
            # 也可以添加到单比特门，表示相邻比特在空闲时产生的串扰（可选）
            # 但那样更复杂，先忽略。

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
            self._simulator = AerSimulator(
                noise_model=self.noise_model,
                basis_gates=self.noise_model.basis_gates,
                coupling_map=self.config.coupling_map,
                # 使用密度矩阵方法来模拟混合态和错误传播
                method='density_matrix',
                # 可选: 提高模拟速度
                # device='GPU'  # 如果有 GPU 支持
            )
        return self._simulator

    def run(self, circuit: QuantumCircuit, shots: Optional[int] = None) -> dict:
        """
        执行电路并返回计数结果 (counts)
        自动根据耦合图进行转译（transpile）以确保电路符合拓扑。
        """
        if shots is None:
            shots = self.config.shots

        # 将电路转译到基础门和拓扑
        transpiled = transpile(
            circuit,
            basis_gates=self.noise_model.basis_gates,
            coupling_map=[list(edge) for edge in self.config.coupling_map],
            optimization_level=1,  # 适度的优化
        )

        simulator = self.get_simulator()
        job = simulator.run(transpiled, shots=shots)
        result = job.result()
        return result.get_counts()

    def run_and_get_counts(self, circuit: QuantumCircuit, shots: Optional[int] = None) -> dict:
        """同 run，返回 counts"""
        return self.run(circuit, shots)

    def run_and_get_statevector(self, circuit: QuantumCircuit):
        """
        返回电路的密度矩阵（电路不应包含测量操作）。
        如果电路包含测量，将抛出异常。
        """
        if circuit.num_clbits > 0:
            raise ValueError("run_and_get_statevector 要求电路不含测量操作")

        simulator = self.get_simulator()
        # 使用 save_density_matrix 保存密度矩阵结果
        circuit_with_save = circuit.copy()
        circuit_with_save.save_density_matrix()

        job = simulator.run(circuit_with_save, shots=1)
        result = job.result()
        return result.data()['density_matrix']


# ============================ 使用示例 ========================================
if __name__ == "__main__":
    # 1. 配置参数（以 3 比特线性拓扑为例）
    config = NoiseConfig(
        t1_times=[50.0, 50.0, 50.0],          # µs
        t2_times=[70.0, 70.0, 70.0],
        freq_ghz=[5.0, 5.0, 5.0],
        single_q_gate_error=0.001,            # 0.1%
        two_q_gate_error=0.01,                # 1%
        coupling_map=[(0, 1), (1, 2)],
        readout_error=[0.02, 0.02, 0.02],     # 2%
        # 可选：自定义串扰强度
        # crosstalk_strength={(0,1): 0.005, (1,2): 0.005},
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