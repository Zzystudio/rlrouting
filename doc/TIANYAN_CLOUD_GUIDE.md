# 天衍量子云平台使用说明

## 1. 基本使用

Cqlib 通过平台类访问天衍云端：

```python
import os
from cqlib import TianYanPlatform
from cqlib.quantum_platform import QuantumLanguage

platform = TianYanPlatform(
    login_key=os.environ["TIANYAN_LOGIN_KEY"],
    machine_name="tianyan24",
    auto_login=True,
)
```

常用接口：

```python
platform.query_quantum_computer_list()       # 查询设备
platform.download_config()                   # 下载机器配置
platform.qcis_check_regular(qcis)            # 检查 QCIS
platform.submit_experiment(...)              # 提交任务
platform.query_experiment(query_ids)         # 查询结果
platform.get_experiment_circuit(query_ids)   # 查看云端编译线路
```

提交 QCIS 并查询结果：

```python
query_ids = platform.submit_experiment(
    circuit=qcis,
    language=QuantumLanguage.QCIS,
    machine_name="tianyan24",
    num_shots=1024,
    is_verify=True,
)
results = platform.query_experiment(query_ids)
```

登录密钥建议使用环境变量：

```bash
export TIANYAN_LOGIN_KEY='你的登录密钥'
```

## 2. 自定义映射和路由

支持自定义初始映射，也支持根据硬件拓扑插入 SWAP 门完成路由。

Cqlib 内置映射接口：

```python
from cqlib.mapping import transpile_qcis

compiled, layout, swap_mapping, final_mapping = transpile_qcis(
    qcis,
    platform,
    initial_layout=None,
    objective="size",
    seed=42,
)
```

参数说明：

- `initial_layout`：逻辑比特到物理比特的初始映射
- `objective="size"`：尽量减少 SWAP 数量
- `objective="depth"`：尽量降低线路深度
- `objective="no_swap"`：尝试寻找不需要 SWAP 的布局

如果使用 Qiskit SABRE：

```python
from qiskit import transpile
from qiskit.transpiler import CouplingMap

transpiled = transpile(
    qiskit_circuit,
    coupling_map=CouplingMap(coupling_edges),
    layout_method="sabre",
    routing_method="sabre",
    optimization_level=0,
    seed_transpiler=42,
)
```

`coupling_edges` 来自机器配置：

```python
config = platform.download_config()
coupler_map = config["overview"]["coupler_map"]
```

路由前要排除：

```python
config["disabledQubits"]
config["disabledCouplers"]
```

Qiskit 线路需要转换为 QCIS 后才能调用 `submit_experiment()`。提交后可以使用 `get_experiment_circuit()` 检查云端最终线路。

## 3. 下载校准数据

```python
config = platform.download_config(machine="tianyan176")
```

主要字段：

```python
config["overview"]["coupler_map"]       # 硬件连接图
config["disabledQubits"]                # 禁用量子比特
config["disabledCouplers"]              # 禁用耦合器
config["qubit"]                         # f01、T1、T2、单比特门参数
config["readout"]["readoutArray"]      # 读出误差和保真度
config["twoQubitGate"]["czGate"]       # CZ 误差、长度、耦合强度
config["twoQubitGate"]["fsim_value"]   # FSIM 参数
```

常见参数：

- `f01`：量子比特频率，单位 `GHz`
- `T1`：能量弛豫时间，单位 `us`
- `T2`：相干时间，单位 `us`
- `gate error`：门误差，通常为百分比
- `readout fidelity`：读出保真度
- `length`：双比特门长度，单位 `ns`
- `coupling strength`：耦合强度，单位 `Hz`

参数通常由 `param_list`、`qubit_used`、`unit` 和 `update_time` 组成，`param_list[i]` 对应 `qubit_used[i]`。

当前校准配置没有显式的串扰参数或多量子比特读出混淆矩阵。

## 4. 云端模拟器和噪声

云端模拟器通过 `machine_name` 选择：

```python
query_ids = platform.submit_experiment(
    circuit=qcis,
    machine_name="tianyan_sw",
    num_shots=1024,
)
```

常见编码：

```text
tianyan_sw   全振幅模拟器
tianyan_s    稳定子模拟器
tianyan_tn   张量网络模拟器
tianyan_sa   单振幅模拟器
tianyan_swn  带噪声密度矩阵模拟器
```

使用噪声模拟器：

```python
noise = [
    {
        "id": 0,
        "noise_type": "depolarizing",
        "data": {"prob": 0.001, "num_qubits": 1},
    },
    {
        "id": 1,
        "noise_type": "depolarizing",
        "data": {"prob": 0.01, "num_qubits": 2},
    },
]
rules = [
    {"noise_id": 0, "add_type": 1, "gates": ["rx", "rz"]},
    {"noise_id": 1, "add_type": 1, "gates": ["cx", "swap"]},
]

query_ids = platform.submit_experiment(
    circuit=qcis,
    language=QuantumLanguage.QCIS,
    name="calibrated-noise-example",
    num_shots=1024,
    machine_name="tianyan_swn",
    noise=noise,
    rules=rules,
)
results = platform.query_experiment(query_ids)
```

当前噪声接口使用 `id`、`noise_type`、`data` 定义噪声，并使用 `rules` 指定应用范围。常见类型包括
`bitflip`、`phaseflip`、`phasebitflip`、`depolarizing`、`pauli`、`damping` 和 `readout`。

校准数据可以帮助估计噪声参数：`T1/T2` 可用于估计退相干，门误差可用于近似去极化，读出保真度适合用于结果校正。校准数据不能直接完整转换为云端噪声模型。

读出校正：

```python
results = platform.query_experiment(
    query_ids,
    readout_calibration=True,
    machine_config=config,
)
```

`tianyan_swn` 标称支持 16 个量子比特，不能直接模拟 `tianyan176` 的完整 66 比特线路。云端噪声参数是全局抽象参数，不包含每个量子比特和每个门的完整校准数组。
