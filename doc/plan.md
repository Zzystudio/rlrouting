# Plan: 将初始映射（mapping）纳入 RL 联合训练

> 状态：已确认方案，待实施
> 决策（用户确认）：虚拟免费映射 SWAP、commit 动作结束映射阶段、直接联合训练（不做单独映射预训练）

## 动机

当前 `RoutingEnv.reset()`（env.py:127-144）初始映射只取 identity 或随机 `random_init`，
布局从未被学习。而 SABRE 基线用 8 个随机初始映射按距离启发式挑最优（doc/GNN.md:688），
PPO 在起跑线上就吃亏。

**关键物理事实**：路由前的 k 个初始 SWAP 等价于直接选一个初始布局——它们可以"虚拟化"
（不写入物理线路、不计入 SWAP 数），这正是 SABRE 初始布局的本质。

## 可行性（基于现有代码）

- 观测已含 `map_vec`（归一化映射向量，env.py:191），GNN 嵌入依赖映射（cross-graph map edges）
- 策略 `EdgeActorCritic` 按边打分，输入 `[h_p, h_q, h_p-h_q]`，映射质量天然可被建模
- `_sabre_edge_features()` / `_front_layer_dist()` 已是"如果换个映射会怎样"的启发式信号
- 距离奖励 `r_dist`（env.py:522）可直接复用于映射阶段

## 方案：虚拟映射阶段（mapping phase）+ commit 动作

```
episode = [ 映射阶段: 任意次虚拟 SWAP（重排初始布局，不执行门）]
        → [ commit 动作，结束映射阶段 ]
        → [ 正常路由: SWAP + 自动执行（现有逻辑）]
```

| 设计点 | 方案 |
|---|---|
| 映射阶段 SWAP | **虚拟**：只更新 `self.mapping`，不 append 到 `_phys_circuit`，不计入 `_swap_counter`，单独记 `_mapping_swaps` |
| 阶段终止 | 动作空间扩为 `Discrete(num_edges + 1)`，最后一个动作 = commit；预算上限 `n-1`（任意置换 ≤ n-1 次交换），耗尽自动 commit，防死循环 |
| 映射阶段奖励 | 复用 `r_dist`（front-layer 距离减少，稠密信号）+ 每步 0 惩罚（虚拟交换免费）；Phase 2 终端保真度奖励通过 GAE/value 反向塑造布局 |
| 掩码 | 映射阶段全部边有效（unmapped mask 照常），commit 仅在映射阶段有效；死锁掩码同样适用于映射阶段震荡 |
| 观测 | 追加 1 维 phase 标志（防止 progress=0 时映射阶段与执行前状态混淆） |
| 训练 | 直接联合训练（Phase 1 从映射阶段开始），距离奖励对布局选择是稠密信号 |
| 评估 | argmax/beam 均天然支持（`clone()` 复制 phase 状态）；指标拆成 `mapping_swaps` + `routing_swaps`，对比 SABRE 时只比 routing SWAPs（公平） |

## 代码改动清单

### 1. `src/routing/rl/env.py`

- `__init__`：
  - 新参数 `mapping_budget: int = None`（默认 `num_qubits - 1`）
  - `action_space = gym.spaces.Discrete(self.num_edges + 1)`，`COMMIT = self.num_edges`
  - obs_dim 加 1（phase 标志），注释更新
- `reset`（env.py:127）：
  - `self.mapping_phase = True`，`self._mapping_swaps = 0`，mapping 从 identity（或 random_init 置换）起步
  - **跳过** `_auto_execute_batch()`（门在 commit 前不得执行）
- 新增 `_apply_virtual_swap(p, q)`：仅交换 `self.mapping`（复用 `_apply_swap` 的 inv 反查），
  不 append `_phys_circuit`、不递增 `_swap_counter`
- `step`（env.py:507）映射阶段分支：
  ```
  if mapping_phase:
      if action == COMMIT:
          mapping_phase = False; _auto_execute_batch(); 进入正常执行逻辑
      else:
          _apply_virtual_swap(p, q); _mapping_swaps += 1
          奖励 = r_dist（front_layer 距离减少，复用 env.py:522 逻辑）
          if _mapping_swaps >= mapping_budget: 自动 commit
  else:
      现有逻辑（SWAP + 自动执行 + 死锁掩码）
  ```
- 虚拟 SWAP 同样追加 `_swap_history`（死锁掩码对映射阶段震荡生效）
- `info["mapping_swaps"]` 与 `info["num_swaps"]`（只计 routing SWAP）分开记录
- 截断沿用 `max_episode_steps`（覆盖整个 episode）
- `_obs`（env.py:190）：布局改为 `[edge_feats | map_vec | progress | phase]`，phase 0/1 放最后
- `clone()`（env.py:343）：复制 `mapping_phase`、`_mapping_swaps`

### 2. `src/routing/rl/agent.py`

- `EdgeActorCritic.critic` 输入 `edge_feat_dim + num_qubits + 1 + 1`（v_in 追加 phase）
- `_forward_obs`（agent.py:163）：progress 切片 `obs[n_ef+n_q : n_ef+n_q+1]`，phase 取最后一维
- `_build_edge_obs`（agent.py:196）：batch 追加 phase 缓冲
- `act`（agent.py:144）：掩码长度 `num_edges + 1`，`mask[num_edges] = True` 仅当映射阶段
- `update`：batch 透传 `phase`（与 graph_data/map_vec/progress 同路径，无结构变化）

### 3. `src/routing/rl/train_agent.py`

- `--mapping-phase`（`BooleanOptionalAction`，默认开启）、`--mapping-budget` 参数
- rollout 循环：`env.mapping_phase` 为 True 时构造含 commit 的掩码
  （死锁/未映射掩码 + `mask[num_edges]=True`），否则屏蔽 commit
- ep_buffer 增加 `phase`；metrics.csv 增加 `map_swaps` 列；日志打印 `map_swp`
- `create_env` 透传 mapping 参数

### 4. `src/routing/rl/eval_policy.py`

- `CircuitMetrics` 加 `mapping_swaps: int`
- `evaluate_circuit` / `evaluate_circuit_beam`：与训练相同的掩码逻辑（commit 只在映射阶段有效，
  运行时读取 `env.mapping_phase`）；beam 的 `clone.step(a)` 自动继承 phase
- 报告拆分 routing SWAPs（与 SABRE 对比）与 total（mapping + routing）

### 5. 文档

- `doc/GNN.md`：MDP 定义补充映射阶段（动作空间 +1、虚拟 SWAP 语义、phase 观测）
- `doc/train.md`：记录 mapping 训练实验
- `AGENTS.md`：Commands 中加 `--mapping-phase` 说明，Best Known Results 更新

## 验证

1. `PYTHONPATH=src python3 -m pytest test/ -q`（检查 `test/` 中硬编码 obs_dim/action_dim 处，需适配）
2. 短训练冒烟测试（`--timesteps 5000`，3 个 5q 拓扑），确认 mapping_swaps 非零、学习曲线正常

## 预期收益与风险

- **收益**：布局 + 路由联合优化，直接对标 SABRE 初始布局启发式；虚拟 SWAP 使最终 SWAP 指标与
  SABRE 公平对比；噪声感知模式下布局会主动避开高错误率边
- **风险**：
  - 映射阶段探索空间大 → 距离奖励是稠密信号可缓解
  - commit 动作可能被策略忽略 → 预算上限兜底保证 phase 总会结束
  - obs 布局变更影响面大 → 同步更新 `_forward_obs` 和训练缓冲

## 备选方案（未采纳，供参考）

- **B：固定预算无 commit**——映射阶段固定 k 次虚拟 SWAP 后自动结束，动作空间不变，改动最少，
  但 agent 不能提前 commit
- **C：直接布局打分头**——独立 bipartite matching head（Sinkhorn/Hungarian），可扩展到大拓扑，
  但偏离"SWAP 动作集合"思路，需要新网络头

---

# Plan: 映射阶段改进（mapping phase v2）

> 状态：待实施（写自 2026.08.05，20q 评估后的性能分析）

## 背景与动机

20q 映射评估结论（doc/train.md「20q 映射增益量化」）：

- **grid_5x4**：映射 -15.4%（22.8→19.3，n16 峰值 -6.2）——布局真正有价值
- **ring**：映射 -5.1%（55.4→52.6），仅小规模受益
- **line**：映射 0%（66.1→66.1）——深电路一维链布局无关，反而消耗步数、降低完成率

**瓶颈定位**（基于 env.py 408-578 实现分析）：

| 问题 | 代码位置 | 现象 |
|------|---------|------|
| 映射奖励弱 | `_step_mapping()` 只用 `r_dist` | 深电路 front-layer dist 改善淹没在路由 SWAP 成本中，agent 无"该不该映射"信号 |
| 无法自适应跳过 | `commit_allowed = _mapping_swaps >= mapping_min_swaps`（train_agent.py:435） | 训练强制映射、评估自由 commit，策略从未学到"何时该跳过" |
| 映射/路由同共享 per-edge GNN | `_obs()` 复用跨图 GNN | 布局是全局优化问题，per-edge 打分建模的是局部路由 |
| 深电路布局无关 | grid/ring/line 需求差异大 | 统一 `mapping_budget=n-1` 一刀切，line 深电路浪费步数 |

## 方案 A：奖励信号增强（P0，~50 行）

### A1. skip 基线奖励

映射阶段第一步先结算"直接 commit"的基线：

```
r_baseline = front_dist(直接 commit 后)   # 或路由价值 V(commit 后状态)
每次虚拟 SWAP: r = -η·dist·(dist_after - dist_before)/dist_before  -  r_baseline
```

- r > 0 → 该 SWAP 比直接 commit 好，值得映射
- r ≤ 0 → 应立即 commit
- 直接解决 line/ring 深电路"不该映射"的问题，让 agent 用奖励学会 skip

### A2. front-layer 可执行 bonus

```python
if 虚拟 SWAP 后某 front-layer 门的 qubit 对变为相邻(dist=1):
    reward += lambda_exec * count_of_newly_executable
```

比 `dist_after - dist_before` 更强的稠密信号——agent 确切感知"这个置换产生可立即执行的门"。

### A3. 布局代价预估器（P2，需离线训练）

- 小 MLP：`(circuit_enc, topo_enc, mapping_vec) → 预测最终路由 SWAP 数`
- 映射阶段每个虚拟 SWAP：`r = λ·(cost_before - cost_after)`
- 训练数据：已完成的 episode 记录 `(layout, final_routing_swaps)`，或离线跑 SABRE/greedy 采样
- 终端保真度经 GAE 回溯到布局决策，收益直接

## 方案 B：架构改进

### B1. 两阶段决策头（P1，~100 行）

```
policy = {
  should_map_head(s) → p(map|s) ∈ [0,1]     # 先决定要不要进入映射
  swap_head(s) → logits over edges         # 再选哪个虚拟 SWAP
  commit_head(s)（已有）
}
```

- `should_map_head=0` 直接进路由，reward 无惩罚（agent 自己权衡）
- 解耦"是否映射"与"怎么映射"，两个目标可用不同 reward 训练
- 解决当前 `--mapping-min-swaps` 强制训练导致的"评估时策略不知道何时该跳"

### B2. 布局专用 GNN / 注意力池化（P2）

- 当前 per-edge GNN（`_obs()`）针对局部 SWAP 打分
- 布局优化需要全局理解（qubit 相对位置、拓扑瓶颈、中心度）
- 加 layout head：attention/set2set pooling over qubit embeddings → layout quality scalar，
  作为映射阶段观测的额外特征 + GAE 的信用分配锚

### B3. 布局值函数（P1）

```
V_layout(mapping_vec, circuit_enc) → R
每次虚拟 SWAP: r = V_layout(mapping_new) - V_layout(mapping_old)
```

- 映射决策由"未来预期价值"驱动，而非单步距离改善
- commit 后终端奖励的 GAE 自然回溯到 V_layout

## 方案 C：训练策略

### C1. 按拓扑/规模自适应 mapping budget（P0，~20 行）

```python
budget = {
    "grid": n_q - 1,              # 全用
    "ring": n_q // 2,             # 折半
    "line": n_q // 4 if depth <= T else 0,   # 深电路直接跳过
}
```

或从数据学：统计各 (topo, depth) 下"最优布局需几次虚拟 SWAP"作为自适应上限。

### C2. 映射专项课程学习（P1）

```
Phase 1: 只用 n8/n10 + 强制映射(min-swaps=n-1) → 小空间短 horizon 学布局策略
Phase 2: 混合 n8-n16，min-swaps 递减到 0 → 学习"何时跳过"
Phase 3: n16-n20，min-swaps=0 → 转移经验，自主选择
```

当前一次横跨 n8-20，深电路的噪声信号淹没了浅电路上可学的布局策略。

### C3. replay buffer + hindsight 奖励（P2）

```
buffer: [(circuit, tentative_layout, final_routing_swaps), ...]
每 N 步蒸馏一个 layout→cost 函数 → 用作映射阶段奖励
```

本质是离线强化映射阶段的信用分配。

## 方案 D：推理阶段（P0~P1，~30 行）

### D1. 映射阶段独享 beam search

当前 beam 全程生效导致完成率下降（line 97%→82%，克隆消耗步数预算）。改为：

- **映射阶段**：beam search（布局搜索 step 少、价值高，一条好布局省 10+ SWAP）
- **路由阶段**：argmax（不复用速度慢 + 完成率不降）

### D2. SABRE 初始布局 warm-start（P2）

推理时先用 SABRE 的"试 K 个随机布局挑 front_dist 最小"作为起点，RL 映射阶段只做 ≤2 步微调。
解耦"全局布局搜索"（SABRE 擅长）与"布局精调"（RL 擅长）。

## 优先级总表

| 优先级 | 方案 | 预期效果 | 成本 |
|--------|------|---------|------|
| **P0** | A1 skip 基线奖励 | line/ring 深电路主动跳过映射，完成率回升 | ~30 行 |
| **P0** | A2 front-layer 可执行 bonus | 映射信号显著增强（局部、确切） | ~15 行 |
| **P0** | C1 自适应 budget | 深电路不浪费步数预算 | ~20 行 |
| **P0** | D1 映射阶段独享 beam | 消灭 beam 的完成率掉点、保留布局搜索收益 | ~30 行 eval_policy |
| **P1** | B1 两阶段决策头 | "要不要映射"与"怎么映射"解耦 | ~100 行 |
| **P1** | B3 布局值函数 | 映射由未来价值驱动而非单步距离 | ~60 行 |
| **P1** | C2 映射专项课程 | grid n16 收益扩大，浅电路专项练 | 训练脚本 |
| **P2** | A3 布局代价预估器 | 终端布局 reward 补全 GAE 回溯 | 需训小 MLP |
| **P2** | B2 布局专用 GNN | 从局部打分到全局布局理解 | 架构改动 |
| **P2** | D2 SABRE warm-start | 全局搜索 + 局部精调解耦 | 推理改动 |

## 验证

1. `PYTHONPATH=src python3 -m pytest test/ -q`（改动后全量回归）
2. 短训练冒烟测试（`--timesteps 10000`，5q 三拓扑）：确认
   - line/ring 上平均 `map_swaps` 明显下降（A1 生效后 agent 学会少映射）
   - grid 上 `map_swaps` 保持/上升（映射有价值的场景继续用）
3. 20q 复评（unified_test，`--random-init`，argmax）：对比 v1 的 19.3/66.1/52.6
   - 期望：line/ring 更好或持平，grid 维持或略升；完成率回升到 ~100%

---

# Plan: 效果提升 + 编译时间优化（gap to SABRE）

> 状态：待实施（写自 2026.08.05，20q 评估后的差距分析）

## 背景：与 SABRE 的差距

### 效果差距（20q, unified_test, random-init, routing, argmax+map）

| 拓扑 | PPO | SABRE | gap | gap% |
|------|-----|-------|-----|------|
| grid_5x4 | 19.3 | 17.2 | 2.1 | 12% |
| line | 66.1 | 53.9 | 12.2 | 23% |
| ring | 52.6 | 45.5 | 7.1 | 16% |

**根因**：SABRE 每步对所有候选 SWAP 做精确启发式评估（O(E·|F|) front-layer distance 求和，
本质是确定性 1 步 lookahead）；PPO 的 `forward → argmax → step` 是对 Q(s,a) 的函数近似，
深电路（100+ 步）近似误差逐级累积。

### 编译时间差距（20q, CPU, 60 circuits）

| 方法 | 每电路 | vs SABRE |
|------|--------|----------|
| SABRE | ~5ms | 1x |
| PPO argmax | ~211-683ms | 40-140x |
| PPO beam3 | ~1240-6314ms | 250-1260x |

**单步耗时剖解**（grid_5x4）：
```
_obs()                        ~1.5ms   (build_routing_graph + GNN.node_embeddings + _sabre_edge_features)
agent._forward_obs()          ~0.3ms   (numpy→torch + edge_mlp + critic)
env.step()                    ~0.3ms   (_apply_swap + _auto_execute_batch)
```

---

## 效果提升方案（GPU 反比：效果优先）

### E1. SABRE 行为克隆预训练（P0，闭合 gap 50-80%）

SABRE 策略是最优启发式，PPO 从它初始化 → 起点在最优附近，RL 只做精细优化。

```python
# Phase 0: 收集 SABRE 示范（run_sabre_and_record 记录每个 (obs, sabre_swap)）
# Phase 0.5: 监督预训练
loss = CrossEntropy(PPO.logits, sabre_swap)
# Phase 1-2: PPO 微调（现有流程）
```

### E2. 混合解码：PPO 提案 + SABRE 评估（P1，闭合 gap 70-90%，~40 行 eval_policy）

不在训练做搜索，在推理时结合两者优势：

```
每步推理：
  1. PPO.forward(s) → 对所有 edge 打 logits
  2. 取 top-K=3 候选 SWAP
  3. 对每个候选：计算 SABRE front_layer distance improvement（O(E)，极快）
  4. 选 distance improvement 最大的执行
```

- 无 beam search 的 clone 开销 → 无完成率掉点
- PPO 学"哪些 SWAP 值得考虑"，SABRE 判断"哪个确实更好"
- 最终决策 ≈ SABRE 效果，PPO 过滤低概率候选 → 可能略优

### E3. 深度电路自适应 GAE λ / n-step TD（P1，深电路专项 10-20%）

当前 `GAE(λ=0.95)` + 100 步 episode → 有效 horizon ≈ 20 步，深电路信号衰减到 0.36。

```python
# λ 随 episode 深度自适应
lambda_t = min(0.95 + 0.05 * (t / max_step), 1.0)   # 深电路近 MC return
# 或 n-step TD: G_t = R_t + ... + γ^n R_{t+n} + γ^n V(s_{t+n})，n 随深度
```

### E4. 显式距离矩阵特征（P2，5-10%）

观测中 SABRE 5 维特征已含距离信息，可把 full distance matrix（n×n）经 linear projection
降维加入全局特征，消除 GNN → per-edge 编码的信息损失。

### E5. 有限树搜索 / beam depth=2（P2，30-50%，需 GNN 缓存配套）

当前 beam depth=1；depth=2 需 k² 展开，同 SABRE 的 lookahead。依赖编译时间优化 T2。

---

## 编译时间优化方案（速度优先）

### T1. GPU 推理（P0，5-10x，改命令行）

当前 eval 用 `--device cpu`。换 `--device cuda`：GNN + AC 前向 ~5-10x，20q 小 batch 无
CUDA launch 开销，零代码改动。

### T2. beam search GNN 缓存复用（P0，beam 场景 3x，~20 行 eval_policy）

```python
# 修改前：每个克隆都重算 GNN
for a in topk:
    clone = env.clone(); clone_obs = clone._obs()      # ← GNN 重算
# 修改后：当前状态 GNN 只算一次，克隆只更新 swap 后的局部特征
cached_emb = env._obs()                                  # ← 1 次 GNN
for a in topk:
    clone = env.clone(); clone_obs = clone._obs(emb=cached_emb)   # 跳过 GNN
```

beam3 每步 3 次 GNN forward → 1 次，**耗时减少 ~60%**。

### T3. 批量电路评估（P0，batch≧4 时 2-3x，~50 行）

同一拓扑的 N 个电路批量 GNN forward（一次 CUDA kernel）。env 仍逐电路 step（不等长交互），
只批 GNN 部分。

### T4. SABRE edge features 增量更新（P1，1.3-1.5x，~30 行 env.py）

预计算 `edge→affected_gates` 映射：一条 SWAP 只影响相关 front-layer 门，
增量更新避免 `_sabre_edge_features` 每步遍历 31 条边 × 全部 front-layer 门。

### T5. 模型蒸馏 / 剪枝（P1，2-5x，离线训 student）

```
Teacher: SubGNN(2-layer GAT) + EdgeActorCritic  → 学生: 1-layer GCN 或 MLP-only
```

GNN.node_embeddings 占 ~40% 总时间；或用 `torch.compile(mode="reduce-overhead")` 零改动 ~1.2-1.3x。

### T6. C++ 推理引擎（P2，10-50x，重写）

`build_routing_graph + _update + _auto_execute_batch` 搬到 C++（libtorch 加载权重），
消除 Python 解释器 / GIL 开销。适合部署与大规模评估。

### T7. 预计算静态顶层特征（P2，1.1x）

Topology 距离矩阵/邻接矩阵/最短路径缓存为 numpy，初始化时预载。

---

## 优先级总表

### 效果

| 优先级 | 方案 | 预期 gap 闭合率 | 成本 |
|--------|------|---------------|------|
| **P0** | E1 SABRE 行为克隆预训练 | 50-80% | 记录脚本 + 1 epoch |
| **P1** | E2 混合解码 | 70-90% | ~40 行 eval_policy |
| **P1** | E3 深度电路自适应 λ | 10-20%（深电路） | ~20 行 |
| **P2** | E4 显式距离矩阵特征 | 5-10% | GNN 输入 +1 特征 |
| **P2** | E5 树搜索 / beam depth=2 | 30-50% | ~200 行 + 依赖 T2 |

### 速度

| 优先级 | 方案 | 预期加速 | 成本 |
|--------|------|---------|------|
| **P0** | T1 GPU 推理 | 5-10x | 命令行 |
| **P0** | T2 beam GNN 缓存 | beam 3x | ~20 行 eval_policy |
| **P0** | T3 批量电路 eval | 2-3x | ~50 行 |
| **P1** | T4 SABRE 特征增量更新 | 1.3-1.5x | ~30 行 env.py |
| **P1** | T5 蒸馏 / torch.compile | 2-5x / 1.2-1.3x | 离线训 student / 1 行 |
| **P2** | T6 C++ 推理引擎 | 10-50x | 重写 |
| **P2** | T7 预计算静态特征 | 1.1x | 初始化缓存 |

## 最快见效组合

- **效果**：E1（SABRE 蒸馏）→ E2（混合解码）→ E3（自适应 λ）
- **速度**：T1（GPU）→ T2（beam GNN 缓存）→ T3（批量 eval）
- 建议优先 E2 + T1 + T2：混合解码接近 SABRE 效果且无速度损失，GPU + GNN 缓存解决 60-1000x 差距

## 验证

1. `PYTHONPATH=src python3 -m pytest test/ -q`
2. 5q 三拓扑复评（stage1_phase1）：确认混合解码 SWAPs ≤ argmax 且≈SABRE，完成率 100%
3. 20q 三拓扑复评（unified_test, random-init）：对比 19.3/66.1/52.6 与 per-circuit 耗时
