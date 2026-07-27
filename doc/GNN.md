**硬件约束驱动（hardware-constrained）、噪声感知（noise-aware）、GNN 增强的强化学习量子路由框架**。

核心思想：
不要让 RL/GNN 学习硬件规则，而是利用 coupling graph 提供硬约束和合法动作空间；利用 GNN 学习线路状态、错误传播、串扰等复杂因素，在合法 SWAP 中选择最优动作。

**图建模**：采用二分异构图（bipartite heterogeneous graph）。逻辑线路图（gate→gate 依赖边）建模电路结构，物理拓扑图（qubit↔qubit 耦合边）建模硬件约束，跨图映射边（gate→qubit）表示 logical→physical 映射。

---

# 1. 总体框架

目标：解决超导量子计算中的 logical circuit → physical hardware 映射问题。

给定：
- 量子线路
- 当前 logical-physical mapping
- 超导硬件拓扑
- 校准噪声信息

输出：
- SWAP 插入策略

优化目标：
- SWAP 数量
- circuit depth
- execution fidelity

```
                Quantum Circuit
                      |
                      |
    Bipartite Heterogeneous Graph Encoder
    (Logic Circuit + Physical Topology +
         Cross-graph Mapping)
                      |
                      |
               State Embedding
                      |
                      |
       --------------------------------
       |                              |
  Physical Topology           Candidate SWAP
  (Hardware Constraint)       Generation
       |                              |
       --------------------------------
                      |
                RL Decision (PPO)
                      |
                Insert SWAP
                      |
                Executability Check
                      |
                Reward Update
```

---

# 2. 代码模块结构

```
src/routing/
├── __init__.py              # 包声明
├── routing.py               # 顶层路由接口 + 贪心路由 baseline
├── graph/
│   ├── __init__.py
│   ├── circuit_dag.py       # CircuitDAG + build_routing_graph
│   └── features.py          # HardwareFeatures + 特征编码函数
├── gnn/
│   ├── __init__.py
│   └── encoder.py           # GATEncoder + SubGNN（二分异构图编码）
└── rl/
    ├── __init__.py
    ├── env.py               # RoutingEnv (Gymnasium)
    ├── agent.py             # PPOAgent (Actor-Critic + GAE)
    └── train_agent.py       # RL 训练脚本
```

---

# 3. 状态表示设计

状态 = `(C_t, M_t, H)`

| 符号 | 含义 |
|------|------|
| `C_t` | 当前未执行线路 |
| `M_t` | logical → physical mapping |
| `H`   | hardware coupling graph + noise calibration |

---

# 4. 图构建（Graph Construction）

## 4.1 CircuitDAG

文件：`src/routing/graph/circuit_dag.py`

由 Qiskit `QuantumCircuit` 解析得到 `GateRecord` 列表，每个 gate 记录：
- `index`：拓扑位置
- `name`：门名称（`h`, `cx`, `rz`, `sx`, …）
- `qubits`：作用的 logical qubits
- `is_two_qubit`：是否双比特门
- `predecessors`：同一比特上的前驱门索引（DAG 依赖边）

```python
@dataclass
class GateRecord:
    index: int
    name: str
    qubits: List[int]
    clbits: List[int]
    operation: Any
    is_two_qubit: bool
    is_measure: bool
    predecessors: List[int]
```

工厂方法 `CircuitDAG.from_circuit(circuit)` 遍历 `circuit.data`，对每个 logical qubit 追踪上一 gate，建立 `predecessors` 依赖关系。双比特门识别：`cx`, `swap`, `cz`, `ecr`。

## 4.2 RoutingGraphData

构建结果数据结构（二分异构图，包含两个子图与跨图映射边）：

```python
@dataclass
class RoutingGraphData:
    # === Logical Circuit Graph ===
    num_gates: int                # G
    gate_feat: np.ndarray         # (G, 32)  门节点特征
    dep_edge_index: np.ndarray    # (2, E_dep)  gate→gate 依赖边 (COO)
    dep_edge_attr: np.ndarray     # (E_dep, 16)  依赖边特征

    # === Physical Topology Graph ===
    num_physical: int             # P
    qubit_feat: np.ndarray        # (P, 32)  物理比特噪声特征
    coupling_edge_index: np.ndarray   # (2, E_couple) qubit↔qubit 耦合边 (COO)
    coupling_edge_attr: np.ndarray    # (E_couple, 16) 耦合边特征

    # === Cross-graph Mapping Edges ===
    map_edge_index: np.ndarray    # (2, E_map)  gate→qubit 映射边 (COO)
    map_edge_attr: np.ndarray     # (E_map, 16)  映射边特征
```

## 4.3 图结构（Bipartite Heterogeneous Graph）

构建函数 `build_routing_graph(dag, mapping, hw, coupling_map, ...)` 创建二分异构图，包含两个子图与跨图映射边：

### 子图 A：逻辑线路图（Logical Circuit Graph）

**节点**：`0 … G-1` — gate 节点（每个门一个节点）

**边类型 `precedes`** — gate → gate（有向）：

| 边类型 | source → target | 含义 |
|--------|----------------|------|
| `precedes` | gate_i → gate_j | 同一逻辑比特上前驱门 → 后继门（DAG 依赖） |

通过 `GateRecord.predecessors` 在 `CircuitDAG` 上扫描，对每个 gate 与其同一比特上的前驱 gate 建立有向依赖边。

### 子图 B：物理拓扑图（Physical Topology Graph）

**节点**：`0 … P-1` — physical qubit 节点（硬件物理比特）

**边类型 `couples`** — qubit ↔ qubit（无向，双向 COO）：

| 边类型 | source → target | 含义 |
|--------|----------------|------|
| `couples` | qubit_i ↔ qubit_j | 物理耦合连接 |

直接由 `coupling_map` 生成：每对直接相连的物理比特建立两条有向边（COO 格式）。

### 跨图映射边（Cross-graph Mapping Edges）

**边类型 `maps_to`** — gate → qubit（有向）：

| 边类型 | source → target | 含义 |
|--------|----------------|------|
| `maps_to` | gate_i → qubit_p | 逻辑门 gate_i 当前映射在物理比特 qubit_p 上执行 |

对每个 gate node `i`（对应 logical qubit(s) `ql`）：
- 单比特门：`maps_to: i → M_t[ql]`（一条边）
- 双比特门：`maps_to: i → M_t[ql0], i → M_t[ql1]`（两条边）

每次 SWAP 动作更新 mapping `M_t` 后，所有 `maps_to` 边需重新计算，反映最新的 logical→physical 映射。

---

# 5. 特征编码（Feature Encoding）

文件：`src/routing/graph/features.py`

## 5.1 HardwareFeatures

`HardwareFeatures` 从 `NoiseConfig` 构建，将所有噪声参数归一化到相近尺度，作为节点和边特征编码的基础数据源：

| 字段 | 原始参数 | 归一化因子 | 形状 | 说明 |
|------|---------|-----------|------|------|
| `t1` | T1 coherence time (µs) | ÷ 100 | (Q,) | 能量弛豫时间 |
| `t2` | T2 coherence time (µs) | ÷ 100 | (Q,) | 退相干时间 |
| `freq` | qubit frequency (GHz) | ÷ 5.0 | (Q,) | 比特工作频率 |
| `readout_err` | readout error | ÷ 0.1 | (Q,) | 测量读取错误率 |
| `single_q_err` | single-qubit gate error | ÷ 0.05 | (Q,) | 单比特门错误，每比特独立 |
| `two_q_err` | two-qubit gate error | ÷ 0.05 | (Q, Q) | 双比特门错误，按耦合边 |
| `adj` | coupling map | — | (Q, Q) | 邻接矩阵，0/1 |
| `zz` | ZZ crosstalk strength | ÷ 0.05 | (Q, Q) | 串扰强度，按比特对 |
| `dist` | shortest path distance | — | (Q, Q) | BFS 最短路距离 |

归一化因子常量：`_T1_SCALE=100`，`_T2_SCALE=100`，`_FREQ_SCALE=5.0`，`_READOUT_SCALE=0.1`，`_ERROR_SCALE=0.05`。

非耦合比特对的 `two_q_err` 和 `zz` 置 0。`dist` 通过 BFS 计算无权重最短路。

## 5.2 节点特征（32 维）

逻辑线路图门节点（`gate_feat` 每行）和物理拓扑图比特节点（`qubit_feat` 每行）均为 32 维，分别编码门的电路属性（DAG 层数、依赖关系、执行状态）和比特的硬件噪声（相干时间、门错误率、频率）。

### 5.2.1 逻辑线路图 — Gate 节点

| 偏移 | 长度 | 内容 | 说明 |
|------|------|------|------|
| 0 | 12 | gate type one-hot | 12 种：h, cx, rz, sx, x, y, z, s, t, swap, measure, barrier |
| 12 | 1 | `is_two_qubit` | 0=单比特门, 1=双比特门 |
| 13 | 1 | `is_measure` | 0=非测量, 1=测量 |
| 14 | 1 | 归一化门持续时间 | `gate_duration / max_duration` |
| 15 | 1 | 归一化门错误率 | 单比特门取 `single_q_err`；双比特门取受控比特对 `two_q_err` 的最大值 |
| 16 | 1 | DAG 层数（线路所在层数） | `dag_depth / total_dag_depth`，拓扑排序中的层级位置 |
| 17 | 1 | 剩余关键路径深度 | 从该门到电路末端的最长路径层数 / total_dag_depth |
| 18 | 1 | logical qubit 0 索引 | 第一个操作比特编号 / M |
| 19 | 1 | logical qubit 1 索引 | 第二个操作比特编号 / M（单比特门为 0） |
| 20 | 1 | 入度（前驱门数） | `in_degree / max_in_degree` |
| 21 | 1 | 出度（后继门数） | `out_degree / max_out_degree` |
| 22 | 1 | 执行状态 | 0=pending（未就绪）, 1=executable（可执行）, 2=executed（已执行） |
| 23 | 1 | 未完成前驱数 | `remaining_predecessors / total_predecessors` |
| 24 | 1 | 门在电路中的序号 | `gate_index / total_gates` |
| 25 | 1 | 映射物理比特间距离 | 双比特门两操作比特映射后物理距离 / max_physical_dist；单比特门为 0 |
| 26 | 1 | 两操作比特已相邻 | 双比特门判断两映射物理比特是否相邻（0/1） |
| 27 | 1 | 保留 / 0 | |
| 28 | 1 | 保留 / 0 | |
| 29 | 1 | 保留 / 0 | |
| 30 | 1 | 保留 / 0 | |
| 31 | 1 | 保留 / 0 | |

### 5.2.2 物理拓扑图 — Physical Qubit 节点

| 偏移 | 长度 | 内容 | 说明 |
|------|------|------|------|
| 0 | 1 | 物理比特编号 | `qubit_index / P` |
| 1 | 1 | T1 相干时间 | 归一化值 |
| 2 | 1 | T2 相干时间 | 归一化值 |
| 3 | 1 | 工作频率 | 归一化值 |
| 4 | 1 | 测量读取错误率 | `readout_err` 归一化值 |
| 5 | 1 | 单比特门错误率 | `single_q_err` 归一化值 |
| 6 | 1 | 平均双比特门错误率 | 所有耦合邻居的 `two_q_err` 平均值 |
| 7 | 1 | 度（耦合邻居数） | `degree / max_degree` |
| 8 | 1 | 邻居平均 T1 | 所有耦合邻居的 T1 平均值（归一化） |
| 9 | 1 | 邻居平均 T2 | 所有耦合邻居的 T2 平均值（归一化） |
| 10 | 1 | 邻居平均频率 | 所有耦合邻居的频率平均值（归一化） |
| 11 | 1 | 最大 ZZ 串扰 | 所有耦合邻居中 `zz` 的最大值（归一化） |
| 12 | 1 | 邻居平均读取错误率 | 所有耦合邻居的 `readout_err` 平均值（归一化） |
| 13 | 1 | 是否被占用 | 当前是否有 logical qubit 映射到此比特（0/1） |
| 14 | 1 | 占用率 | 映射在该比特的 logical qubit 上 pending 门数 / total_gates，未占用为 0 |
| 15 | 1 | 频率失谐 | `abs(freq_i - median_freq) / median_freq` |
| 16 | 1 | 可执行门依赖数 | 依赖此比特 mapped logical qubit 的可执行门数量 / max_executable_gates |
| 17 | 1 | 最近被占用比特距离 | 到最近已占用物理比特的最短路径长度 / 直径（归一化） |
| 18 | 1 | 占用邻居数 | 已占用的耦合邻居数 / degree |
| 19 | 1 | 保留 / 0 | |
| 20 | 1 | 保留 / 0 | |
| 21 | 1 | 保留 / 0 | |
| 22 | 1 | 保留 / 0 | |
| 23 | 1 | 保留 / 0 | |
| 24 | 1 | 保留 / 0 | |
| 25 | 1 | 保留 / 0 | |
| 26 | 1 | 保留 / 0 | |
| 27 | 1 | 保留 / 0 | |
| 28 | 1 | 保留 / 0 | |
| 29 | 1 | 保留 / 0 | |
| 30 | 1 | 保留 / 0 | |
| 31 | 1 | 保留 / 0 | |

## 5.3 边特征（16 维）

三种边类型在 `edge_attr` 中用前 3 维 one-hot 区分，剩余维度编码各类型特有信息。

### 5.3.1 `precedes`（类型 0）—— gate → gate 依赖边

| 偏移 | 长度 | 内容 | 说明 |
|------|------|------|------|
| 0 | 3 | edge type one-hot | `[1, 0, 0]` |
| 3 | 1 | DAG 层数差 | `(src.depth − tgt.depth) / total_depth` |
| 4 | 1 | 源门层数 | `src.dag_depth / total_depth` |
| 5 | 1 | 目标门层数 | `tgt.dag_depth / total_depth` |
| 6 | 1 | 共享 logical qubit 编号 | 依赖边所在比特编号 / M |
| 7 | 1 | 是否关键路径边 | 若在 DAG 的关键路径上则为 1，否则 0 |
| 8 | 1 | 源门是否可执行 | 0/1 |
| 9 | 1 | 目标门剩余前驱数 | `tgt.remaining_predecessors / tgt.total_predecessors` |
| 10 | 6 | 保留 / 0 | |

### 5.3.2 `couples`（类型 1）—— qubit ↔ qubit 耦合边

| 偏移 | 长度 | 内容 | 说明 |
|------|------|------|------|
| 0 | 3 | edge type one-hot | `[0, 1, 0]` |
| 3 | 1 | 双比特门错误率 | 从 `two_q_err[i][j]` 取值（归一化） |
| 4 | 1 | ZZ 串扰强度 | 从 `zz[i][j]` 取值（归一化） |
| 5 | 1 | T1 几何平均 | `√(t1_i × t1_j)` 归一化 |
| 6 | 1 | T2 几何平均 | `√(t2_i × t2_j)` 归一化 |
| 7 | 1 | 频率差 | `abs(freq_i − freq_j)` 归一化 |
| 8 | 1 | 读取错误率乘积 | `readout_err_i × readout_err_j` 归一化 |
| 9 | 1 | 单比特门错误率几何平均 | `√(1q_err_i × 1q_err_j)` 归一化 |
| 10 | 1 | 两比特是否均被占用 | 0/1 |
| 11 | 1 | 频率碰撞标志 | 若频率差小于碰撞阈值则为 1，否则 0 |
| 12 | 4 | 保留 / 0 | |

### 5.3.3 `maps_to`（类型 2）—— gate → qubit 跨图映射边

| 偏移 | 长度 | 内容 | 说明 |
|------|------|------|------|
| 0 | 3 | edge type one-hot | `[0, 0, 1]` |
| 3 | 1 | 物理比特编号 | `physical_index / P` |
| 4 | 1 | 操作比特角色 | 0=源门的第一操作比特, 1=第二操作比特 |
| 5 | 1 | 该物理比特 T1 | 归一化 |
| 6 | 1 | 该物理比特 T2 | 归一化 |
| 7 | 1 | 该物理比特频率 | 归一化 |
| 8 | 1 | 该物理比特读取错误率 | 归一化 |
| 9 | 1 | 该物理比特单比特门错误率 | 归一化 |
| 10 | 1 | 该物理比特平均双比特门错误率 | 所有耦合邻居 `two_q_err` 均值（归一化） |
| 11 | 1 | 该物理比特是否被占用 | 0/1 |
| 12 | 1 | 到另一操作比特的物理距离 | 双比特门：两映射物理比特间最短路长度 / 直径；单比特门为 0 |
| 13 | 3 | 保留 / 0 | |

---

# 6. GNN 编码器

文件：`src/routing/gnn/encoder.py`

## 6.1 GATEncoder

使用 PyG 的 `GATConv`，支持边特征：

```
GATEncoder(
    node_dim=32, edge_dim=16,
    hidden_dim=64, num_layers=3, heads=4, out_dim=64, dropout=0.1
)
```

- 前 `num_layers-1` 层：每层 `heads` 个头，每头维度 `hidden_dim // heads`，输出拼接为 `hidden_dim`，后接 `LayerNorm → ReLU → Dropout`
- 最后一层：单头，输出 `out_dim` 维，无 concat
- `forward(x, edge_index, edge_attr) → (N, out_dim)`
- `graph_embedding(x, edge_index, edge_attr, batch) → (B, 2*out_dim)`：`global_mean_pool || global_max_pool`

## 6.2 SubGNN

多子网架构中的单个子网，可配置为工作在二分异构图的某个子图上：

```python
SubGNN(node_dim, edge_dim, subgraph='full',
       hidden_dim=48, num_layers=2, heads=3, out_dim=48)
```

- 内部包含一个 `GATEncoder`
- `forward` 根据 `subgraph` 从 `RoutingGraphData` 中选择对应的子图进行编码：
  - `'logic'`：使用 `gate_feat` + `dep_edge_index` + `dep_edge_attr`
  - `'physics'`：使用 `qubit_feat` + `coupling_edge_index` + `coupling_edge_attr`
  - `'mapping'`：使用 `gate_feat` + `qubit_feat`（拼接）+ `map_edge_index` + `map_edge_attr`
  - `'full'`：使用全图（gate_feat + qubit_feat 拼接，三种边合并）
- `graph_embedding` 输出维度 = `out_dim * 2 = 96`

---

# 7. RL 环境

文件：`src/routing/rl/env.py`

基于 Gymnasium 的量子路由环境 `RoutingEnv`。

## 7.1 MDP 定义

**状态**（observation vector）：

```
obs = [GNN embedding (384-d) |  normalized mapping (M-d) |  progress (1-d)]
```

- `GNN embedding`：来自 SubGNN 编码器的图嵌入向量
- `normalized mapping`：`mapping[i] / num_physical_qubits`，长度 = logical qubits 数
- `progress`：`executed_gates / total_gates`

**动作空间**：`Discrete(num_edges + 1)`

| 动作 | 含义 |
|------|------|
| `0 … E-1` | 在 `coupling_map[action]` 上执行 SWAP |
| `E` | "execute"：执行最早可执行的双比特门 |

**奖励函数**（通过 `reward_mode` 配置，支持三阶段训练）：

每种模式的奖励都是步级奖励 `r_step` 与回合终端奖励 `R_T` 之和：

```
R = Σ r_step + R_T
```

**`reward_mode = 'routing'`（Stage 1 — 纯路由先验）**

仅使用步级稠密奖励，不引入模拟器 fidelity。目标：在硬件拓扑约束下学习有效的 SWAP 路径规划。

| 事件 | 奖励 |
|------|------|
| SWAP 动作 | `-swap_penalty`（默认 0.1） |
| 成功 execute | `+α_gate`（默认 0.5） |
| 无 gate 可执行时 execute | `-invalid_penalty`（默认 1.0） |
| 映射距离变化 | `-γ_dist · Δdist`（可选，默认 0） |

回合终端奖励 `R_T = 0`（无 fidelity 信号）。

**`reward_mode = 'noise_aware'`（Stage 2 — 步级路由 + 终端保真度）**

保留 Stage 1 的步级稠密奖励，回合结束时加入 Qiskit noisy simulator 终端保真度。此时 agent 已学会基础路由，现在学习在多个可行路由方案中选择保真度更高的。

步级奖励 `r_step`：

| 事件 | 奖励 |
|------|------|
| SWAP 动作 | `-swap_penalty`（默认 0.1） |
| 成功 execute | `+α_gate`（默认 0.5） |
| 无 gate 可执行时 execute | `-invalid_penalty`（默认 1.0） |
| 串扰惩罚（耦合边被占用） | `-γ_xtalk`（默认 0.02） |
| 高错误率边惩罚 | `-δ_err`（双比特门通过高错误率耦合边时，默认 0.01） |

回合终端奖励：`R_T = λ_fid · F_Qiskit`（λ_fid 默认 5.0，F_Qiskit 为 Qiskit Aer noisy simulator 输出保真度）。

总奖励：`R = Σ r_step + λ_fid · F_Qiskit`

**`reward_mode = 'fidelity_shaping'`（Stage 3 — 保真度塑造）**

在天衍/高精度模拟器或真机场景下使用。在 Stage 2 的基础上，每步额外加入保真度变化奖励，提供更丰富的反馈。

步级奖励扩展：`r_step = r_routing + η·ΔF_t`

其中 `ΔF_t = F(s_{t+1}) − F(s_t)` 是用高精度模拟器估计的单步保真度变化。

回合终端奖励：`R_T = λ_fid · F_final`

总奖励：`R = Σ(r_routing + η·ΔF_t) + λ_fid · F_final`

## 7.2 环境内部逻辑

`_update()` 在每个动作后调用：
1. 自动执行所有 executable 的单比特门（无需动作）
2. 扫描双比特门：依赖满足 且 两 logical qubit 在物理上相邻 → 加入 `executable_2q` 列表

`step(action)`：
- 若 action == `num_edges`（execute）：弹出 `executable_2q` 中最早的门执行；若列表为空则给予 `invalid_penalty`
- 若 action < `num_edges`（SWAP）：交换 `mapping` 中对应物理位置的两个 logical qubit
- 重新调用 `_update()`
- 所有 gate 执行完毕 → `done=True`，添加 fidelity 奖励

## 7.3 初始化

- `random_init=True`：随机 permutation 作为初始 mapping
- `random_init=False`：identity mapping

---

# 8. PPO 智能体

文件：`src/routing/rl/agent.py`

## 8.1 ActorCritic 网络

```python
ActorCritic(obs_dim, action_dim, hidden=128)
├── shared: Linear(obs_dim, 128) → ReLU → Linear(128, 128) → ReLU
├── actor_head: Linear(128, action_dim)      # 输出 logits
└── critic_head: Linear(128, 1)              # 输出标量 value
```

## 8.2 PPOAgent

**超参数**：

| 参数 | 默认值 |
|------|--------|
| `lr` | 3e-4 |
| `gamma` | 0.99 |
| `lam`（GAE λ） | 0.95 |
| `clip_eps` | 0.2 |
| `ent_coef` | 0.01 |
| `vf_coef` | 0.5 |

**核心方法**：
- `act(obs) → (action, log_prob, value)` — 从 categorical 分布采样
- `update(batch, epochs=4, batch_size=64)` — 标准 PPO 更新：
  - GAE 计算 advantage
  - 优势归一化
  - clipped surrogate loss + value MSE + entropy bonus
  - 梯度裁剪（max_norm=0.5）

## 8.3 训练

文件：`src/routing/rl/train_agent.py`

训练分三个阶段进行，每阶段加载上一阶段模型权重微调（fine-tune）：

```
train_agent.py --reward_mode routing             # Stage 1
train_agent.py --reward_mode noise_aware \        # Stage 2
                --load models/policy_s1.pt
train_agent.py --reward_mode fidelity_shaping \   # Stage 3
                --load models/policy_s2.pt
```

**三阶段训练概览**：

| 阶段 | reward_mode | 模拟器 | 核心目标 |
|------|-------------|--------|---------|
| Stage 1 | `routing` | 无 | 学习硬件拓扑下的有效 SWAP 路径规划 |
| Stage 2 | `noise_aware` | Qiskit Aer noisy simulator | 在多个可行路径中选择保真度更高的 |
| Stage 3 | `fidelity_shaping` | 天衍/高精度模拟器/真机 | 利用高精度保真度反馈做精细塑造 |

**Stage 1 — 路由先验训练**：
- 不使用任何模拟器 fidelity
- 仅用步级稠密奖励（`ΔN_exec`、`C_swap`）
- 学习 routing policy prior：哪些 SWAP 合理、如何减少映射距离、如何避免无效搜索
- 不引入 fidelity 的原因：① 目标冲突（agent 可能还没能力理解长期影响）；② 初始噪声模型简化导致的 simulator bias

**Stage 2 — 噪声感知训练**：
- 加载 Stage 1 模型，在 Qiskit Aer noisy simulator 上微调
- 保留步级稠密奖励 + 加入终端保真度信号 `λ_fid · F_Qiskit`
- 此时 agent 已知道如何完成路由，学习在多个可行方案中选择保真度更高的
- 终端保真度作为稀疏信号与步级稠密奖励互补

**Stage 3 — 保真度塑造训练**：
- 加载 Stage 2 模型，在高精度模拟器/真机上微调
- 额外加入每步保真度变化 `η·ΔF_t`，提供逐步反馈
- 终端保真度 `λ_fid · F_final` 仍保留
- 高精度模拟器足够快，可产生丰富反馈支持 fidelity shaping

**训练循环**（每阶段通用）：
1. 创建 `RoutingEnv`（随机电路 + 噪声配置，`reward_mode` 决定奖励函数）
2. 创建/加载 `PPOAgent`
3. Rollout 收集 `rollout_steps`（默认 256）步
4. 计算 GAE，调用 `agent.update()`
5. 跟踪最近 20 个 episode 的平均 reward，保存最佳模型

---

# 9. 贪心路由 Baseline

文件：`src/routing/routing.py`

`greedy_route()` 作为 baseline 路由策略：

1. 确定初始布局：尝试 `num_trials=8` 个随机初始映射，用距离启发式评估选最优
2. 逐门处理：遍历拓扑排序后的门列表
3. 遇到双比特门且两 logical qubit 物理上不相邻时：
   - 扫描所有 coupling edge 作为候选 SWAP
   - 对每个候选 SWAP 模拟执行后两个 logical qubit 的距离，选择使距离最小的 SWAP
4. 重复直至所有门可执行

入口函数 `route_circuit()` 执行贪心路由策略。

---

# 10. 状态更新流程

每一步的循环：

```
Step 1: 检查 front layer 是否有可执行门
  ├── 可执行 → 执行 gate，回到 Step 1
  └── 不可执行 → Step 2

Step 2: 生成合法 SWAP 候选（仅 coupling map 上的边）

Step 3: PPO 选择 SWAP 动作

Step 4: 更新 mapping

Step 5: 回到 Step 1
```

---

# 11. 三阶段训练框架

训练按三个阶段递进，每个阶段的目标和奖励设计不同：

## Stage 1: 纯路由先验（Routing Prior）

**目标**：在硬件拓扑约束下，学习通过 SWAP 有效完成线路映射。不优化最终保真度，仅学习 routing 能力。

**奖励**：

```
r_t = α·ΔN_exec − β·C_swap − γ·C_distance（可选）
R_T = 0（无终端保真度）
```

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `ΔN_exec` | 该步新执行的双比特门数 | — |
| `C_swap` | SWAP 开销 | 0.1 per SWAP |
| `C_distance` | 映射距离变化惩罚（可选） | 0 |

**为什么 Stage 1 不引入 fidelity**：

1. **目标冲突**：agent 可能还没能力理解长期影响——某个 SWAP 立即释放 5 个门但降低估计保真度 vs 暂时无门执行但远期更好
2. **Simulator bias**：初始噪声模型简化，过早优化 `F_sim` 会让 agent 学会适应你的模拟器误差，而非学会好的 routing
3. **信用分配困难**：episode 可能有几十到几百步，terminal fidelity 非常延迟，RL 难以判断哪一步 SWAP 贡献了最终好坏

**本质**：学习 routing policy prior —— 哪些 SWAP 合理，如何减少映射距离，如何避免无效搜索。

## Stage 2: 噪声感知路由（Noise-Aware Routing）

**目标**：在已掌握基础路由的基础上，学习在多个可行方案中选择保真度更高的。

**奖励**：

```
步级：r_t = α·ΔN_exec − β·C_swap − γ·C_xtalk − δ·C_error
终端：R_T = λ_fid · F_Qiskit
总奖励：R = Σ r_t + λ_fid · F_Qiskit
```

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `C_xtalk` | 串扰惩罚（耦合边被占用） | 0.02 |
| `C_error` | 高错误率边惩罚（双比特门通过高错误率耦合边） | 0.01 |
| `F_Qiskit` | Qiskit Aer noisy simulator 终端保真度 | — |
| `λ_fid` | 终端保真度权重 | 5.0 |

**为什么 Stage 2 用步级 + 终端混合**：

- 保留步级稠密奖励保证训练信号充分
- 终端保真度作为稀疏信号与稠密奖励互补
- 类似：Stage 1 学会了"如何到达目的地"，Stage 2 学习"选择哪条路质量最好"

## Stage 3: 保真度塑造（Fidelity Shaping）

**目标**：利用高精度模拟器/真机的丰富反馈，精细优化每一步的路由决策。

**奖励**：

```
步级：r_t = α·ΔN_exec − β·C_swap − γ·C_xtalk + η·ΔF_t
      ΔF_t = F(s_{t+1}) − F(s_t)  （高精度模拟器单步保真度变化）
终端：R_T = λ_fid · F_final
总奖励：R = Σ(r_t + η·ΔF_t) + λ_fid · F_final
```

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `ΔF_t` | 高精度模拟器估计的单步保真度变化 | — |
| `η` | 单步保真度变化权重 | 1.0 |
| `F_final` | 高精度模拟器终端保真度 | — |
| `λ_fid` | 终端保真度权重 | 5.0 |

**与 Stage 2 的区别**：

- Stage 2 只在 episode 结束时获得一次 fidelity 信号
- Stage 3 每步都获得 `ΔF_t`（fidelity shaping），信用分配更精确
- 高精度模拟器足够快，可以产生丰富的 per-step 反馈

## 阶段递进关系

```
Stage 1 (routing)      →    学会怎么路由
    ↓
Stage 2 (noise_aware)  →    学会在可行路径中选择保真度高的
    ↓
Stage 3 (fidelity_shaping) → 精细塑造每步决策
```

每阶段加载上一阶段模型权重进行微调（fine-tune），而非从头训练。

---

# 12. 模块依赖图

```
sim/sim.py ───────────────────────────┐
  │                                   │
  └──> routing/graph/features.py ───> routing/graph/circuit_dag.py
         │                                      │
         └──> routing/gnn/encoder.py            │
                │                               │
                └──> routing/routing.py ────────┘
                └──> routing/rl/env.py
                         │
                         └──> routing/rl/agent.py
                                │
                                └──> routing/rl/train_agent.py
```

---

# 13. 后续实验路线

建议逐步比较：

### Baseline 1
GraphSAGE + RL（原始方法，无噪声感知）

### Baseline 2
GAT + RL（加注意力，无子网分解）

### Model 1
Gate-RGAT：加入 dependency + error propagation 关系

### 评价指标

| 指标 | 意义 |
|------|------|
| SWAP 数量 | routing 效率 |
| Circuit depth | 执行时间 |
| Estimated fidelity | 可靠性（simulator 验证） |
| Success probability | 最终性能（simulator 验证） |
| Runtime | 算法效率 |

---

# 14. 超参数速查

## NoiseConfig（默认值）

| 参数 | 默认值 |
|------|--------|
| `single_q_gate_error` | 0.001 |
| `two_q_gate_error` | 0.01 |
| `single_gate_time` | 0.1 µs |
| `two_gate_time` | 0.3 µs |
| `idle_time` | 0.1 µs |
| `shots` | 1024 |
| `readout_error` | 0.02 |
| `device` | 'CPU' |

## RoutingEnv

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `reward_mode` | 奖励模式: `routing` / `noise_aware` / `fidelity_shaping` | `routing` |
| `swap_penalty` | SWAP 动作惩罚系数 β | 0.1 |
| `gate_reward` | 双比特门执行奖励系数 α | 0.5 |
| `invalid_penalty` | 无效 execute 惩罚 | 1.0 |
| `γ_dist` | 映射距离惩罚系数（Stage 1 可选） | 0.0 |
| `γ_xtalk` | 串扰惩罚系数（Stage 2/3） | 0.02 |
| `δ_err` | 高错误率耦合边惩罚系数（Stage 2/3） | 0.01 |
| `λ_fid` | 终端保真度权重（Stage 2/3） | 5.0 |
| `η` | 单步保真度变化权重（Stage 3） | 1.0 |

## PPOAgent

| 参数 | 默认值 |
|------|--------|
| `lr` | 3e-4 |
| `gamma` | 0.99 |
| `lam` | 0.95 |
| `clip_eps` | 0.2 |
| `ent_coef` | 0.01 |
| `vf_coef` | 0.5 |
| `hidden` | 128 |

## 训练脚本

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `reward_mode` | 奖励模式 | `routing`（Stage 1） |
| `epochs` | 每 rollout PPO 更新轮次 | 4 |
| `batch_size` | PPO mini-batch 大小 | 64 |
| `lr` | 学习率 | 3e-4 |
| `timesteps` | 总训练步数 | 20000（Stage 1）/ 10000（Stage 2/3 fine-tune） |
| `rollout_steps` | 每次 rollout 收集步数 | 256 |
| `load` | 加载预训练模型路径 | — |
