# CqRouting ：噪声感知量子线路路由（推理框架）

本工作包含一个在真机子拓扑 **tianyan176_20q** 上训练完成的噪声感知路由模型、最小化推理框架、标准测试电路与冒烟测试。给定 OpenQASM 2.0 逻辑线路，
框架输出满足目标耦合图约束的物理线路（插入 SWAP 的映射路由），并可附带
调度感知保真度估计。

与 SABRE 基线相比，该模型的特点是**用相对多 SWAP 换取更低的真机执行噪声**：
优先把 SWAP 与门操作放置在低错误率耦合边、以更高并行度压缩执行时间。
在 NAM 标准基准（19 条算术电路）上的验证结果：

| 口径 | R3b beam5 | SABRE | 备注 |
|------|-----------|-------|------|
| 调度感知模拟器（T=16） | **0.3368** | 0.2747 | 优 +22.7% |
| 混合精度（T=64/32/16） | 0.2052 | **0.2110** | 统计打平（SWAP 45.0 vs 45.1） |
| 强异构拓扑（v2 混合精度） | **0.2175** | 0.1962 | 优 +10.9% |

推荐推理模式为 **beam search（--beam 5）**，即验证结果中的 R3b_beam5 配置。

## 目录结构

```
delivery/
├── README.md                 # 本文件
├── run_route.py              # 推理入口 CLI（唯一需要直接调用的脚本）
├── pyproject.toml            # 依赖声明（pip install -e .）
├── models/
│   └── policy_r3b.pt         # 最佳模型权重（R3b，含自动特征维度识别）
├── src/                      # 推理框架源码（routing / sim / utils）
│   ├── routing/              #   DAG、硬件特征、GNN 编码器、PPO、env、CLI
│   ├── sim/                  #   v1 调度感知轨迹模拟器（保真度估计）
│   └── utils/
├── topologies/               # 拓扑定义（JSON）+ Q 标签 sidecar
│   ├── tianyan176_20q.json   #   R3b 的训练/验证拓扑（推荐）
│   └── tianyan287_20q.json   #   实验扩展拓扑（模型未在该拓扑验证）
├── examples/
│   ├── toy/                  # 玩具电路（ghz_5、random_6q，用于冒烟）
│   └── nam_circs/            # NAM 标准基准 19 条（模型训练零重叠）
├── test/
│   └── test_delivery.py      # pytest 冒烟测试（路由有效性/耦合图合法性/指标）
└── routed/                   # 默认输出目录（含两条已生成的示例输出）
```

## 环境依赖

- Python ≥ 3.10
- 核心依赖：`torch`、`torch_geometric`、`qiskit`（≥1.0）、`numpy`
- 测试额外需要：`pytest`

```bash
cd delivery
pip install -e .          # 或手动 pip install torch torch_geometric "qiskit>=1.0" numpy
```

无需 GPU：单条中小电路在 CPU 上毫秒级完成。

## 快速开始

以下命令均在 `delivery/` 目录下执行。

```bash
# 1) argmax 推理（最快）
python run_route.py --circuit examples/toy/ghz_5.qasm \
    --topo topologies/tianyan176_20q.json --out routed/ghz_5

# 2) beam search（推荐模式，对应验证结果中的 R3b_beam5）
python run_route.py --circuit examples/nam_circs/tof_3.qasm \
    --topo topologies/tianyan176_20q.json --beam 5 --out routed/tof_3_beam5

# 3) SABRE 基线对照（同拓扑同参数）
python run_route.py --circuit examples/nam_circs/tof_3.qasm \
    --topo topologies/tianyan176_20q.json --baseline sabre \
    --out routed/tof_3_sabre

# 4) 附带 v1 调度感知保真度估计（16 条噪声轨迹）
python run_route.py --circuit examples/toy/ghz_5.qasm \
    --topo topologies/tianyan176_20q.json --fidelity 16 --out routed/ghz_5

# 5) 批量处理一个目录下的全部电路（配合 shell 循环）
for f in examples/nam_circs/*.qasm; do
  python run_route.py --circuit "$f" --topo topologies/tianyan176_20q.json \
      --beam 5 --out "routed/nam_$(basename "$f" .qasm)"
done
```

## 输出说明

每条电路产生两个文件（`--out` 为前缀）：

- `<out>.qasm`：路由后的物理线路（含 SWAP），可直接用于后续调度/提交；
- `<out>.json`：指标汇总，主要字段：

| 字段 | 含义 |
|------|------|
| `completed` | 是否在步数上限内完成路由（false = 截断失败，见「已知限制」） |
| `num_swaps` | 插入的 SWAP 数（映射阶段虚拟 SWAP 另计于 `mapping_swaps`，不计入线路） |
| `profile` | 自动识别的模型配置（交付模型为 `r3b`） |
| `fidelity_v1` | v1 调度感知轨迹保真度估计（`--fidelity T` 时输出；16 轨迹为报告口径） |
| `initial_mapping` / `final_mapping` | 逻辑→物理初始/最终映射（逻辑 i ↔ 物理 `mapping[i]`） |

注意：`fidelity_v1` 为 **SWAP 免费口径**的轨迹态保真度，绝对值系统性偏高，
适合同口径下的相对比较（与 SABRE 对照）；不宜与 v2 口径或真机结果直接比较。

## CLI 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--circuit` | （必填） | 输入 OpenQASM 2.0 电路 |
| `--topo` | `topologies/tianyan176_20q.json` | 拓扑 JSON |
| `--model` | `models/policy_r3b.pt` | 模型权重（特征维度自动识别） |
| `--beam` | `0` | beam 宽度；0 = argmax，推荐 5 |
| `--baseline` | `none` | `sabre` 时改用 SABRE 基线路由（同拓扑） |
| `--fidelity` | `0` | >0 时用 v1 模拟器估计保真度（值为轨迹数 T） |
| `--out` | `routed/out` | 输出前缀 |
| `--max-episode-steps` | `1000` | 单电路路由步数上限 |
| `--device` | `cpu` | 推理设备 |
| `--seed` | `0` | 随机种子（beam 内部扰动 / SABRE） |

## 运行测试

```bash
cd delivery
python -m pytest test/ -q
```

测试覆盖：模型特征维度自动识别、toy 电路 argmax 路由有效性与耦合图合法性
（所有 2Q 门落在耦合边上）、beam 与 SABRE 输出的 SWAP 一致性区间、v1 保真度
值域、CLI 端到端双输出。已人工验证通过的样例输出见 `routed/`
（ghz\_5：3 SWAP / v1 保真度 0.742；random\_6q：12 SWAP / 0.628）。
