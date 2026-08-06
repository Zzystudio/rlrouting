# 噪声模拟器 (Noise Simulator)

基于 Qiskit Aer 的增强噪声模拟器框架，用于模拟真实量子硬件中的各类噪声效应。

## 功能特性

- **热弛豫噪声**：基于 T1 弛豫时间和 T2 退相干时间
- **门错误**：单/双比特门退极化错误
- **拓扑耦合图**：支持任意量子比特连接拓扑
- **读出错误**：可配置的测量错误率
- **串扰噪声**：基于拓扑的 ZZ 耦合串扰
- **密度矩阵模拟**：自然支持混合态和错误传播

## 依赖

```
numpy
qiskit >= 2.5
qiskit-aer >= 0.17
```

## 快速开始

```python
from sim.sim import NoiseConfig, NoiseSimulator
from qiskit import QuantumCircuit

# 1. 配置噪声参数
config = NoiseConfig(
    t1_times=[50.0, 50.0, 50.0],        # 每个比特的 T1 时间 (µs)
    t2_times=[70.0, 70.0, 70.0],        # 每个比特的 T2 时间 (µs)
    freq_ghz=[5.0, 5.0, 5.0],          # 每个比特的工作频率 (GHz)
    single_q_gate_error=0.001,          # 单比特门错误率 (0.1%)
    two_q_gate_error=0.01,              # 双比特门错误率 (1%)
    coupling_map=[(0, 1), (1, 2)],      # 线性拓扑: 0-1-2
)

# 2. 创建模拟器
sim = NoiseSimulator(config)

# 3. 构造量子电路
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.measure_all()

# 4. 运行模拟
counts = sim.run(qc, shots=2048)
print(counts)
```

## API 参考

### NoiseConfig

噪声模拟器的配置数据类。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `t1_times` | `List[float]` | 是 | 每个量子比特的 T1 弛豫时间 (µs) |
| `t2_times` | `List[float]` | 是 | 每个量子比特的 T2 退相干时间 (µs) |
| `freq_ghz` | `List[float]` | 是 | 每个量子比特的工作频率 (GHz) |
| `single_q_gate_error` | `float` | 是 | 单比特门平均错误概率 |
| `two_q_gate_error` | `float` | 是 | 双比特门平均错误概率 |
| `coupling_map` | `List[Tuple[int,int]]` | 是 | 量子比特耦合图 |
| `readout_error` | `List[float]` | 否 | 每个比特的读出错误概率，默认 0.02 |
| `crosstalk_strength` | `Dict[Tuple,Tuple,float]` | 否 | 自定义串扰强度，默认为 `0.1 * two_q_gate_error` |
| `single_gate_time` | `float` | 否 | 单比特门持续时间，默认 0.1 µs |
| `two_gate_time` | `float` | 否 | 双比特门持续时间，默认 0.3 µs |
| `idle_time` | `float` | 否 | 空闲等待时间，默认 0.1 µs |
| `shots` | `int` | 否 | 默认采样次数，默认 1024 |
| `device` | `str` | 否 | 模拟设备，`'CPU'` 或 `'GPU'`，默认 `'CPU'`（需 `qiskit-aer-gpu`） |

### NoiseSimulator

噪声模拟器主类。

#### 构造函数

```python
NoiseSimulator(config: NoiseConfig)
```

根据配置构建完整的噪声模型。

#### 方法

| 方法 | 说明 |
|------|------|
| `run(circuit, shots=None)` | 执行电路，返回测量计数结果 |
| `run_and_get_counts(circuit, shots=None)` | 同 `run()` |
| `run_and_get_statevector(circuit)` | 返回密度矩阵（电路不能含测量） |
| `get_simulator()` | 获取底层 `AerSimulator` 实例 |

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `config` | `NoiseConfig` | 配置参数 |
| `noise_model` | `NoiseModel` | 构建好的 Qiskit 噪声模型 |

## 噪声模型详解

### 1. 热弛豫 (Thermal Relaxation)

基于 T1/T2 参数模拟量子比特的能量弛豫和相位退相干：

- **T1**：激发态衰减时间常数
- **T2**：相位退相干时间常数（T2 ≤ 2×T1）
- 应用于所有单比特门（`rz`, `sx`, `x`, `y`, `z`, `h`）和空闲状态

### 2. 门错误 (Gate Errors)

使用退极化通道（depolarizing channel）建模门操作错误：

- 单比特门错误率 `p`：以概率 `p` 将量子态映射到完全混合态
- 双比特门错误率 `p`：对两比特系统施加退极化

### 3. 串扰噪声 (Crosstalk)

基于拓扑结构的 ZZ 耦合串扰：

- 对每对相邻比特施加 `Pauli-ZZ` 错误
- 默认强度为 `0.1 × two_q_gate_error`
- 仅在执行 CNOT 门时触发

### 4. 读出错误 (Readout Errors)

对称测量错误模型：

```
P(0|0) = P(1|1) = 1 - err
P(0|1) = P(1|0) = err
```

## 注意事项

1. **参数约束**：`T2 ≤ 2 × T1`，否则会抛出 `NoiseError`
2. **比特数一致**：`t1_times`, `t2_times`, `freq_ghz` 长度必须相同
3. **run_and_get_statevector**：电路不能包含测量操作
4. **转译**：`run()` 方法会自动调用 `transpile()` 适配拓扑结构

---

## 轨迹采样状态向量模拟器（TrajectorySimulator）

`sim/trajectory_sim.py` 提供基于 **轨迹采样（蒙特卡洛波函数）+ 状态向量** 的噪声
模拟器，用于解决密度矩阵 4^n 内存爆炸问题（n=20 需要 ~16 TB）。

### 核心思路

维护单个纯态状态向量（内存 2^n，n=20 时单条轨迹仅 ~16 KB * 8），对每条噪声通道
逐门做 Kraus 的 Monte-Carlo 采样（量子轨线）：

- **热弛豫**（T1/T2）：三段采样——以概率 `p_reset * P(|1>)` 跳变到 `|0>`；
  否则以概率 `p_z * P(|1>)` 相位翻转；都无跳变则 `|1>` 分量乘 `exp(-t/T2)`。
- **单比特门退极化**：等价于 qiskit `depolarizing_error(p, 1)`——恒等概率
  `1 - 3p/4`，X/Y/Z 各 `p/4`。
- **双比特 CNOT 退极化**：15 个非单位 Pauli 各 `p/16`（恒等 `1 - 15p/16`）。
- **串扰 ZZ**：在两比特上以 `crosstalk_strength`（默认 `0.1 * two_q_gate_error`）
  施加 `Z⊗Z`，与 CNOT 退极化叠加合成（`depol ∘ ZZ`）。
- **读出错误**：按 `readout_error` 对称翻转测量结果。

保真度用轨迹平均：`F = (1/T) Σ_t |<ψ_ideal|ψ_t>|²`。

### 与 density_matrix 版本的一致性

解开器验证：在相同噪声语义下，`TrajectorySimulator` 与 Aer `density_matrix`
求解保真度统计一致（同通道两种求解器误差 <0.02）。

**注意** 与 `sim.py` 的一个行为差异：旧版 float 模式串扰会把每条边上的 CNOT
退极化替换为 ZZ（Aer specific-错误覆盖 all-qubit 错误）；轨迹模拟器按设计意图
将两者**合成**（`depol ∘ ZZ`），噪声更强也更贴近真实硬件。

### 依赖

```
numpy
qiskit >= 2.5
```

### 快速开始

```python
from sim.sim import NoiseConfig
from sim.trajectory_sim import TrajectorySimulator
from qiskit import QuantumCircuit

config = NoiseConfig(
    t1_times=[50.0, 50.0, 50.0],
    t2_times=[70.0, 70.0, 70.0],
    freq_ghz=[5.0, 5.0, 5.0],
    single_q_gate_error=0.001,
    two_q_gate_error=0.01,
    coupling_map=[(0, 1), (1, 2)],
    readout_error=[0.02, 0.02, 0.02],
)
sim = TrajectorySimulator(config, num_trajectories=256, seed=0)

qc = QuantumCircuit(3)
qc.h(0); qc.cx(0, 1); qc.cx(1, 2)

# 1) 计数（每 shot 一条独立轨迹，量值与 density_matrix 一致）
counts = sim.run(qc, shots=2048)

# 2) 保真度（推荐，O(T * 2^n)，大 n 不会 OOM）
qc2 = qc.copy(); qc2.measure_all()
fid, *_ = 0.0
res = sim.run_trajectories(qc, num_trajectories=256)
fid = res.fidelity(sim.ideal_statevector(qc))

# 3) 兼容旧接口：返回 TrajectoryResult，.data 为平均密度矩阵（大 n 抛错提示改用 fidelity）
result = sim.run_and_get_statevector(qc)
rho = result.data          # 2^n x 2^n
```

### API 参考

| API | 说明 |
|-----|------|
| `run(circuit, shots=None, skip_transpile=False)` | 计数 counts。每个 shot 一条独立噪声轨迹 |
| `run_trajectories(circuit, num_trajectories=None, skip_transpile=False)` | 返回 `TrajectoryResult`（状态向量集合） |
| `run_and_get_statevector(...)` | 兼容旧接口，返回 `TrajectoryResult` |
| `run_batch / run_batch_statevector` | 批量执行 |
| `fidelity(circuit, ideal_sv=None, ...)` | 直接返回平均保真度 |
| `ideal_statevector(circuit)` | 无噪声状态向量（与 Aer statevector 一致） |
| `_transpile(circuit)` | 与 `NoiseSimulator._transpile` 兼容 |

### TrajectoryResult

| 成员 | 说明 |
|------|------|
| `statevectors` | (T, 2^n) 复数数组，每条轨迹的末态 |
| `fidelity(ideal_sv)` | `mean_t |<ψ_ideal|ψ_t>|²` |
| `fidelity_std(ideal_sv)` | 跨轨迹标准差 |
| `.data` | 平均密度矩阵（内存超 2 GiB 抛 `RuntimeError`，提示改用 `fidelity`） |

### 可扩展方向

- **T2 > 2×T1**：当前做硬件约束校验报错；如需支持需引入独立相位阻尼通道。
- **轨迹并行化**：`run_trajectories` 目前是串行 for 循环，可对 `_evolve` 做向量化
  批量演化或用 `numba`/多进程加速。
- **串扰时序**：可为相邻边在 CX 执行时施加邻近比特 ZZ（当前优化为同比特对叠加）。

---

## 示例输出

```
模拟结果: {'000': 935, '111': 992, '001': 21, '110': 21, ...}
```

理想 GHZ 态应只有 `000` 和 `111`，噪声导致其他状态出现。

## GPU 加速

### 安装

GPU 加速需要 `qiskit-aer-gpu` 包（基于 CUDA），该包依赖较旧的 `qiskit<2.0`：

```bash
pip install 'qiskit<2.0'
pip install qiskit-aer-gpu
```

### 使用

在 `NoiseConfig` 中设置 `device='GPU'` 即可启用：

```python
config = NoiseConfig(
    ...
    device='GPU',
)
sim = NoiseSimulator(config)
```

### 性能基准

测试环境: 8× NVIDIA RTX 4090 D, CUDA 13.2  
模拟方法: density_matrix, shots=4096, 3 次平均

| Qubits | Depth | CPU avg | GPU avg | Speedup |
|--------|-------|---------|---------|---------|
| 5      | 10    | 0.0459s | 0.5803s | 0.08x |
| 5      | 50    | 0.0712s | 0.0622s | 1.14x |
| 5      | 100   | 0.0971s | 0.0934s | 1.04x |
| 10     | 10    | 0.2469s | 0.0787s | 3.14x |
| 10     | 50    | 0.7512s | 0.1965s | 3.82x |
| 12     | 10    | 1.6454s | 0.4061s | 4.05x |

**结论**:
- 小规模 (≤5 qubits): GPU 因核启动开销而更慢，建议使用 CPU
- 中大规模 (≥10 qubits): GPU 加速比 3~4x，推荐启用 GPU
- 默认 `device='CPU'` 适用于本项目 RL/GNN 训练中的小规模模拟
