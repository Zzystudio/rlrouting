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

**动作空间**：`Discrete(num_edges)`

| 动作 | 含义 |
|------|------|
| `0 … E-1` | 在 `coupling_map[action]` 上执行 SWAP |

每次 SWAP 后环境自动检查并执行所有可执行的双比特门，直至无可执行门为止。

**奖励函数**（通过 `reward_mode` 配置，支持三阶段训练）：

```
R = Σ (r_execute_t + r_swap_t + r_propagate_t) + R_T
```

每种模式由步级奖励（三组件）与回合终端奖励组成（Stage 3 步级为 0）。

**`reward_mode = 'routing'`（Stage 1 — 纯路由先验）**

仅使用步级稠密奖励，不引入模拟器 fidelity。目标：在硬件拓扑约束下学习有效的 SWAP 路径规划。
奖励由三大组件构成：**门执行收益**、**SWAP 惩罚**、**错误传播信号**。

| 组件 | 条件 / 事件 | 步级奖励 |
|------|------------|---------|
| **门执行收益** | 成功执行 gate g（映射在物理比特 p_a, p_b 上） | `+R_base(type(g)) − ε_err · e_g − ε_xtalk · X_g` |
| **SWAP 惩罚** | 在 coupling edge (p,q) 上插入 SWAP | `−C_swap`（默认等价 3 CNOT 代价） |
| **错误传播** | 门执行后更新 X/Z 错误率 | `−η_xz · Δ_total_XZ` |

其中 `R_base(type)` 按门类型查表，`e_g` 为 gate 的硬件错误率，`X_g` 为邻居串扰代价。
每次 SWAP 后环境自动执行所有可执行的双比特门，累积门执行收益和错误传播惩罚：
`r_step = r_swap + Σ(r_execute_i + r_propagate_i)`。
回合终端奖励 `R_T = 0`（无 fidelity 信号）。

**`reward_mode = 'noise_aware'`（Stage 2 — 步级路由 + 终端分布优化）**

延用 Stage 1 的三组件步级奖励（门执行收益 + SWAP 惩罚 + 错误传播），回合结束时通过 Qiskit Aer noisy simulator 获取理想/噪声输出分布，计算 KL 散度、交叉熵、TVD 加权作为终端奖励。此时 agent 已学会基础路由，现在学习在多个可行路由方案中选择输出分布最接近理想分布的路径。

**`reward_mode = 'fidelity_shaping'`（Stage 3 — 纯终端输出分布优化）**

在天衍高精度模拟器或真机场景下使用。不设任何步级奖励，仅在回合结束时通过模拟器/真机采样获取理想与噪声输出分布，计算 KL 散度、交叉熵、TVD 加权作为终端奖励。此时 agent 已具备完整的路由能力，仅通过输出分布差异信号做最终精调。

步级奖励：`r_step = 0`（无步级信号）

回合终端奖励：`R_T = λ_fid · F_high`（λ_fid 默认 5.0，F_high 为天衍/高精度模拟器输出的终端保真度）。

总奖励：`R = λ_fid · F_high`

## 7.2 环境内部逻辑

`_update()` 在每个动作后调用：
1. 自动执行所有 executable 的单比特门（无需动作）
2. 扫描双比特门：依赖满足 且 两 logical qubit 在物理上相邻 → 加入 `executable_2q` 列表

`_apply_swap(p, q)` 处理物理比特多于逻辑比特的情况（`env.py:115-133`）：

| p 状态 | q 状态 | 行为 |
|--------|--------|------|
| 空闲 | 空闲 | no-op（空 SWAP，不改变 mapping） |
| 空闲 | 占用 | 逻辑比特从 q 移到 p |
| 占用 | 空闲 | 逻辑比特从 p 移到 q |
| 占用 | 占用 | 正常交换两个逻辑比特 |

通过反查表 `inv = {phys: log for log, phys in enumerate(self.mapping)}` 和 `.get()` 安全处理未占用比特。`_swap_counter` 在每次 SWAP 动作后递增，回合结束时写入 `info["num_swaps"]`。

`step(action)`（`env.py:313-327`）—— 宏步 MDP：

1. agent 选择 `action ∈ [0, E-1]`，即一条物理耦合边
2. 在该边上执行 SWAP（`_apply_swap`），`_swap_counter += 1`
3. 执行 `_auto_execute_batch()`：
   - `_update()` 刷新状态，找出当前所有可执行双比特门
   - 循环执行所有可执行门：
     - 执行最早的可执行双比特门（加入 `self.executed`）
     - 累加 `r_execute` 和 `r_propagate`
     - 重新 `_update()`（执行完一个门可能解锁后续门）
   - 直到 `executable_2q` 为空
4. `reward = r_swap + Σ r_execute + Σ r_propagate`
5. 若全部 gate 执行完毕 → `done=True`，叠加 `_terminal_reward(info)`

`_auto_execute_batch()` 在 `reset()` 末尾也被调用，确保初始映射下已相邻的门立即执行。

## 7.3 初始化

- `random_init=True`：随机 permutation 作为初始 mapping
- `random_init=False`：identity mapping

## 7.4 奖励函数实现详解

文件：`src/routing/rl/env.py`。每步仅 SWAP 动作，随后自动执行所有可执行门，步级奖励三组件求和：

```
r_step = r_swap + Σ(r_execute_i + r_propagate_i)
```

SWAP 未解锁任何门时只有 `r_swap`；解锁门则累加对应 `r_execute + r_propagate`。
终端奖励 `R_T` 仅在 episode 结束时叠加。

### 7.4.1 门执行收益（r_execute）

成功执行 gate g 时：

```
r_execute = R_base(type(g)) − ε_err · gate_error_rate − ε_xtalk · crosstalk_cost
```

**R_base(type)** — 按门类型的基础收益查表：

| 门类型 | 基础收益 | 说明 |
|--------|---------|------|
| `cx` / `cz` / `ecr` | 2.0 | 双比特纠缠门，最核心 |
| `swap` | 1.5 | 类似双比特 |
| `h` / `sx` | 0.5 | 常用单比特 |
| `x` / `y` / `z` | 0.3 | 单比特 Pauli |
| `rz` / `s` / `t` | 0.2 | 相位门 |
| `measure` / `barrier` | 0.0 | 不计入 |

**gate_error_rate** — 当前映射物理比特对上的硬件错误率：

```
gate_error_rate = single_q_err[p]                        若单比特门
                  two_q_err[p_a, p_b]                    若双比特门
```

从 `HardwareFeatures` 读取（已归一化）。

**crosstalk_cost** — 执行 gate 时邻居被占用导致的串扰惩罚：

```
crosstalk_cost = ∑_{q ∈ N(p_a) ∪ N(p_b)} zz[q, p_*] · occupied[q]
```

其中 `N(p)` 是物理比特 p 在 coupling map 上的邻居，`zz[]` 是 ZZ 串扰强度矩阵。

### 7.4.2 SWAP 惩罚（r_swap）

在 coupling edge (p,q) 上插入 SWAP 时：

```
r_swap = −C_swap    (默认 C_swap = 0.3，即 3 个 CNOT 的等效惩罚)
```

CNOT 是其基本操作单元，每次 SWAP 的基准惩罚 `C_swap` 按其物理分解开销设定：

```
C_swap = 3 · γ_cnot
γ_cnot = 0.1  (单个 CNOT 的基准开销)
```

SWAP 的插入还额外通过 7.4.3 的错误传播机制产生间接惩罚（误差累加）。

### 7.4.3 错误传播（r_propagate）

为每个 logical qubit 维护一对 `(X_i, Z_i)` 错误率，模拟 Pauli 错误在电路中的传播。

**初始化**：每个 logical qubit 从噪声数据获得初始错误率：

```
X_i⁰ = single_q_err[M⁻¹(p_i)]         — 映射物理比特的单比特错误率
Z_i⁰ = single_q_err[M⁻¹(p_i)]         — 初始 Z 错误率（对称初始化）
```

**门执行后的传播规则**：

| 门类型 | X 传播 | Z 传播 |
|--------|--------|--------|
| **CX** (控制 c, 目标 t) | X_c → X_c ⊗ I, X_t → X_c ⊗ X_t | Z_c → Z_c ⊗ Z_t, Z_t → I ⊗ Z_t |
| 即 | `X_c ← X_c, X_t ← X_c + X_t` | `Z_c ← Z_c + Z_t, Z_t ← Z_t` |
| **H** (h) | `X ← Z, Z ← X` | 交换 |
| **S** / **SX** | `X ← Z` | 不变 |
| **Rz** / **T** | 不变 | 不变 |
| **通用单比特** | `X ← X + g_e, Z ← Z + g_e` | 累加门错误率 |

更新公式（以 error rate 相加模拟概率）：
```
X_t' = X_t + g_e                       # 执行门在目标比特上注入新错误
X_c' = X_c + X_t'                      # CX 把控制门 X 传播到目标
Z_c' = Z_c + g_e
Z_t' = Z_t + Z_c'                      # CX 把目标门 Z 传播到控制
```

**步级错误传播惩罚**：

```
r_propagate = −η_xz · (ΔX + ΔZ)

ΔX = Σ_i (X_i' − X_i)      # 所有比特的 X 错误率增量
ΔZ = Σ_i (Z_i' − Z_i)      # 所有比特的 Z 错误率增量
```

每步执行 gate 后（包括 SWAP 中的 3 个 CNOT），`r_propagate` 累加新增误差，给 agent 一个"该操作引入多少新错误"的稠密信号。

### 7.4.4 终端奖励

```python
def _terminal_reward(self, info: dict) -> float:
```

| 模式 | `info` 写入 | 奖励值 |
|------|------------|--------|
| routing | `num_swaps` | 0 |
| noise_aware | `num_swaps`, `divergence_metrics` | `λ_kl · D_KL + λ_ce · H_cross + λ_tvd · TVD` |
| fidelity_shaping | `num_swaps`, `divergence_metrics` | `λ_kl · D_KL + λ_ce · H_cross + λ_tvd · TVD` |

Stage 1 不使用任何终端奖励，仅依赖步级稠密信号。Stage 2/3 输出分布指标见 §11.2。

### 7.4.5 总奖励

```
R = Σ (r_swap + Σ r_execute + Σ r_propagate) + R_T
```

其中每步只做 SWAP，自动执行批量的门。`R_T` 在 episode 结束时由 `_terminal_reward` 叠加。

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
| Stage 2 | `noise_aware` | Qiskit Aer noisy simulator | 在多个可行路径中选择输出分布最接近理想分布的 |
| Stage 3 | `fidelity_shaping` | 天衍/高精度模拟器/真机 | 纯终端输出分布差异指标，最终精调 |

**Stage 1 — 路由先验训练**：
- 不引入模拟器 fidelity，也不使用任何终端奖励
- 使用三组件步级奖励：门执行收益（按类型 + 错误率 + 串扰）、SWAP 惩罚（3 CNOT 等价）、错误传播（X/Z 跟踪）
- 学习 routing policy prior：哪些 SWAP 合理、如何减少错误累积、如何避免无效搜索

**Stage 2 — 噪声感知训练**：
- 加载 Stage 1 模型，在 Qiskit Aer noisy simulator 上微调
- 步级奖励延用 Stage 1 三组件，终端奖励改为输出分布指标
- 此时 agent 已知道如何完成路由，学习在多个可行方案中选择输出分布最接近理想分布的路径

**Stage 3 — 纯终端输出分布优化**：
- 加载 Stage 2 模型，在高精度模拟器/真机上微调
- 步级奖励为 0，仅通过理想 vs 噪声输出分布的 KL 散度、交叉熵、TVD 给出终端奖励
- 此时 agent 已掌握路由能力，通过纯输出分布差异信号做最终精调

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

**目标**：在硬件拓扑约束下，学习通过 SWAP 有效完成线路映射。不优化终端 fidelity（无模拟器），但引入**错误传播模型**作为稠密替代信号，引导 agent 理解错误累积。

**奖励**：每步 `r_step` 由三部分求和：

```
r_step = r_execute + r_swap + r_propagate
```

### 组件 1：门执行收益（r_execute）

成功执行 gate g（映射在物理比特 p_a, p_b 上）：

```
r_execute = R_base(type(g)) − ε_err · gate_error_rate(p_a, p_b) − ε_xtalk · crosstalk_cost(g)

gate_error_rate = single_q_err[p_a]               若单比特门
                  two_q_err[p_a, p_b]             若双比特门
```

**R_base 查表**：

| 门类型 | 基础收益 | 物理意义 |
|--------|---------|---------|
| `cx` / `cz` / `ecr` | 2.0 | 双比特纠缠门，核心操作 |
| `swap` | 1.5 | 等效 3 CNOT |
| `h` / `sx` | 0.5 | 常用单比特 Clifford |
| `x` / `y` / `z` | 0.3 | 单比特 Pauli |
| `rz` / `s` / `t` | 0.2 | 相位门 |
| `measure` / `barrier` | 0.0 | 不计入 |

**crosstalk_cost** — 执行该 gate 时，其物理比特的邻居若被占用则产生 ZZ 串扰：

```
crosstalk_cost(g) = ∑_{q ∈ N(p_a) ∪ N(p_b)} zz[q, p_*] · occupied_flag[q]

其中 occupied_flag[q] = 1 若 q ∈ {M(ℓ) | ℓ ∈ logical_qubits}，否则 0
```

`N(p)` 为物理比特 p 在 `coupling_map` 上的邻居集合，`zz[]` 为 HardwareFeatures 的串扰强度矩阵。

### 组件 2：SWAP 惩罚（r_swap）

物理 SWAP 门由 3 个 CNOT 门分解实现：

```
SWAP(q₀, q₁) = CNOT(q₀, q₁) · CNOT(q₁, q₀) · CNOT(q₀, q₁)
```

每次插入 SWAP 的步级惩罚：

```
r_swap = −C_swap     (默认 0.3)

C_swap = 3 · γ_cnot
γ_cnot = 0.1        (单 CNOT 的基准开销)
```

SWAP 的 3 个 CNOT 还会在错误传播模型（组件 3）中额外累加 `3 × CNOT` 的误差量。

### 组件 3：错误传播信号（r_propagate）

为每个 logical qubit ℓ 维护 `(X_ℓ, Z_ℓ)` 错误率，模拟 Pauli 错误在电路中的传播。

**初始化**：每个 logical qubit ℓ 映射到物理比特 p = M(ℓ)：

```
X_ℓ⁰ = single_q_err[p]          # 初始 X 错误率
Z_ℓ⁰ = single_q_err[p]          # 初始 Z 错误率（对称初始化）
```

**门传播规则**：执行 gate g（逻辑比特 qa, qb，物理比特 pa, pb）后更新涉及的比特。

假设 error rate ≪ 1，以加法近似概率传播（忽略 O(ε²) 项）：

| 门类型 | X 后 | Z 后 |
|--------|------|------|
| **CNOT** (控制 c, 目标 t) | `X_c ← X_c, X_t ← X_c + X_t + g_e` | `Z_c ← Z_c + Z_t + g_e, Z_t ← Z_t` |
| **H** | `X ← Z, Z ← X` | +g_e 对两者 |
| **S / SX** | `X ← Z + g_e` | Z 不变 |
| **Rz / T / S** | X 不变 | Z 不变 |
| 其他单比特 | `X ← X + g_e, Z ← Z + g_e` | 各累加单比特门错误率 |

其中 `g_e = gate_error_rate(pa, pb)` 是当前 step 执行该 gate 的硬件错误率。

**SWAP 的传播**：SWAP 分解为 3 个 CNOT，依次应用上述规则。

**步级错误传播惩罚**：

```
r_propagate = −η_xz · (ΔX_total + ΔZ_total)

ΔX_total = Σ_ℓ (X_ℓ' − X_ℓ)      # 所有 logical qubit 的 X 错误率增量
ΔZ_total = Σ_ℓ (Z_ℓ' − Z_ℓ)      # 所有 logical qubit 的 Z 错误率增量
```

每步的 `r_propagate` 给 agent 一个稠密信号：该操作引入了多少新错误。

### 总奖励

```
R = Σ (r_execute_t + r_swap_t + r_propagate_t)

  = Σ [ R_base(type(g_t)) − ε_err · e_g_t − ε_xtalk · X_g_t ]
    + Σ [ −C_swap · SWAP_t ]
    + Σ [ −η_xz · (ΔX_t + ΔZ_t) ]
```

不含任何终端奖励项，完全由步级稠密信号驱动。

### 参数表

| 参数 | 含义 | 默认值 | 代码符号 |
|------|------|--------|---------|
| `R_base(type)` | 门类型基础收益 | cx=2.0, h=0.5, … | `gate_base_reward` (dict) |
| `ε_err` | 硬件错误率惩罚系数 | 0.5 | `eta_err` |
| `ε_xtalk` | 串扰惩罚系数 | 0.02 | `eta_xtalk` |
| `γ_cnot` | 单 CNOT 基准开销 | 0.1 | `cnot_cost` |
| `C_swap` | SWAP 惩罚 (= 3·γ_cnot) | 0.3 | `swap_cost` |
| `η_xz` | 步级 X/Z 传播惩罚系数 | 0.1 | `eta_xz_step` |
| `invalid_penalty` | 无效 execute 惩罚 | 1.0 | `invalid_penalty` |

**为什么 Stage 1 引入错误传播而非模拟器 fidelity**：

1. **稠密信号**：每步的 `(ΔX, ΔZ)` 给 agent 即时反馈，而非 fidelity 的稀疏终端信号
2. **无模拟器开销**：错误传播是 O(N) 解析计算，无需运行 Qiskit Aer
3. **可解释性**：`(X, Z)` 错误率直观反映每条 routing 路径的噪声累积量
4. **与 Stage 2 平滑过渡**：Stage 1 低噪声路径 ≈ Stage 2 高保真度路径

## Stage 2: 噪声感知路由（Noise-Aware Routing）

**目标**：在已掌握基础路由的基础上，学习在多个可行方案中选择终端噪声更低的路径。

**步级奖励**：延用 Stage 1 的三组件设计（门执行收益 + SWAP 惩罚 + 错误传播）。
每步 agent 选择一条 coupling edge 执行 SWAP，环境自动执行后续所有可执行门：

```
r_swap       = −C_swap                                     (SWAP 惩罚)
Σ r_execute  = Σ [ R_base(type(g)) − ε_err · e_g − ε_xtalk · X_g ]   (本批门执行收益)
Σ r_propagate = Σ [ −η_xz · (ΔX + ΔZ) ]                              (本批门错误传播)
```

**终端奖励**：不再使用标量保真度，而是通过 Qiskit Aer noisy simulator 同时获取**理想分布**与**噪声分布**的输出结果，计算多个分布差异指标，加权作为终端奖励：

```
R_T = λ_kl · D_KL(P_noisy ‖ P_ideal) + λ_ce · H(P_ideal, P_noisy) + λ_tvd · TVD(P_noisy, P_ideal)
```

其中：

| 指标 | 含义 | 公式 | 范围 | 默认权重 |
|------|------|------|------|---------|
| **KL divergence** `D_KL` | 噪声分布相对于理想分布的 KL 散度 | `Σ_x P_ideal(x) · log(P_ideal(x) / P_noisy(x))` | [0, ∞) | `λ_kl = 0.5` |
| **Cross-entropy** `H_cross` | 理想与噪声分布的交叉熵 | `−Σ_x P_ideal(x) · log(P_noisy(x))` | [0, ∞) | `λ_ce = 0.3` |
| **Total Variation Distance** `TVD` | 全变差距离 | `½ Σ_x |P_ideal(x) − P_noisy(x)|` | [0, 1] | `λ_tvd = 1.0` |

- `P_ideal`：Qiskit Aer 无噪声模拟的输出分布（`shots` 次采样直方图）
- `P_noisy`：Qiskit Aer 含噪声模拟的输出分布（相同 `shots`，相同噪声配置）

**比标量保真度精度更高**：
- 标量保真度 `F = Σ_x √(P_ideal(x) · P_noisy(x))` 只反映分布的部分信息（重叠度）
- KL 散度对分布尾部和大偏差更敏感，惩罚非物理结果的能力更强
- TVD 上限为 1，易于归一化和调参
- 交叉熵等价于负对数似然，在信息论意义上给出分布差异

**权重归一化**：`R_T` 的 scale 通过各指标默认权重调节，实际使用前应通过经验回放计算各指标的实际数值范围，自适应调整 `λ` 使三者的贡献大致均衡。

总奖励：

```
R = Σ (r_execute_t + r_swap_t + r_propagate_t) + λ_kl · D_KL + λ_ce · H_cross + λ_tvd · TVD
```

| 参数 | 含义 | 默认值 | 代码符号 |
|------|------|--------|---------|
| `λ_kl` | KL 散度终端权重 | 0.5 | `lambda_kl` |
| `λ_ce` | 交叉熵终端权重 | 0.3 | `lambda_ce` |
| `λ_tvd` | TVD 终端权重 | 1.0 | `lambda_tvd` |
| 其余步级参数 | 同 Stage 1 | — | — |

**为什么用分布指标替换标量保真度**：

1. **精细度**：TVD 和 KL 散度对不同路由策略的输出差异区分力更强，尤其在保真度接近 0.9–0.99 区间时
2. **端到端优化**：RL agent 直接优化"输出分布与理想分布一致"，而非一个中间标量
3. **与 GNN 预测器互补**：GNN 预测器输出标量保真度作为 state embedding，终端奖励用分布指标作为信号，两者互补

## Stage 3: 纯终端输出分布优化（Terminal-Only Distribution Shaping）

**目标**：在天衍/高精度模拟器或真机场景下，仅通过终端输出分布差异指标对已具备路由能力的 agent 做最终优化。

**奖励**：

```
步级：r_t = 0（无步级奖励）
终端：R_T = λ_kl · D_KL + λ_ce · H_cross + λ_tvd · TVD
总奖励：R = λ_kl · D_KL + λ_ce · H_cross + λ_tvd · TVD
```

使用与 Stage 2 相同的分布指标（KL 散度、交叉熵、TVD），但去掉所有步级奖励。P_ideal 来自理想模拟器（或已知真机精确分布），P_noisy 来自含噪模拟器/真机采样。

| 参数 | 含义 | 默认值 | 代码符号 |
|------|------|--------|---------|
| `D_KL` | KL 散度 `D_KL(P_noisy ‖ P_ideal)` | — | `divergence_metrics['kl']` |
| `H_cross` | 交叉熵 `H(P_ideal, P_noisy)` | — | `divergence_metrics['cross_entropy']` |
| `TVD` | 全变差距离 `½Σ|P_noisy − P_ideal|` | — | `divergence_metrics['tvd']` |
| `λ_kl` | KL 权重 | 0.5 | `lambda_kl` |
| `λ_ce` | 交叉熵权重 | 0.3 | `lambda_ce` |
| `λ_tvd` | TVD 权重 | 1.0 | `lambda_tvd` |

**去掉步级奖励**：

- 前两阶段已教会 agent 路由能力和噪声感知能力
- Step reward 在后期可能产生干扰，让 agent 为了即时收益（如执行更多门、减少 SWAP）而牺牲终端输出分布质量
- 纯终端信号迫使 agent 关注全局最终输出结果，由高精度模拟器/真机给出最真实的分布评价

## 阶段递进关系

```
Stage 1 (routing)      →    学会怎么路由
    ↓
Stage 2 (noise_aware)  →    学会在可行路径中选择输出分布最接近理想分布的
    ↓
Stage 3 (fidelity_shaping) → 纯终端输出分布差异精调
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
| `gate_base_reward` | 门类型基础收益表 | `{"cx":2.0, "h":0.5, "rz":0.2, …}` |
| `eta_err` | 硬件错误率惩罚系数 ε_err | 0.5 |
| `eta_xtalk` | 串扰惩罚系数 ε_xtalk | 0.02 |
| `cnot_cost` | 单 CNOT 基准开销 γ_cnot | 0.1 |
| `swap_cost` | SWAP 惩罚 (= 3·cnot_cost) | 0.3 |
| `eta_xz_step` | 步级 X/Z 传播惩罚系数 η_xz | 0.1 |
| `invalid_penalty` | 无效动作惩罚 | 1.0 |
| `init_x_error` | 初始 X 错误率（若为 None 则从 single_q_err 取） | `None` |
| `init_z_error` | 初始 Z 错误率（若为 None 则从 single_q_err 取） | `None` |
| `lambda_kl` | KL 散度终端权重（Stage 2/3） | 0.5 |
| `lambda_ce` | 交叉熵终端权重（Stage 2/3） | 0.3 |
| `lambda_tvd` | TVD 终端权重（Stage 2/3） | 1.0 |
| `fidelity_fn` | 外部模拟器函数 (dag, mapping) → (P_ideal, P_noisy) | `None` |

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
