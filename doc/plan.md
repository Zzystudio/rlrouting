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

---

# Plan: 门时间与线路时间感知（timing-aware 调度 + 并行串扰 + 累计退相干）

> 状态：方案已确认，待实施（写自 2026.08.06）
> 决策（用户确认）：
> - 调度由**环境内部固定策略**执行（Agent 只做路由决策，间接影响调度效果）
> - 并行门串扰用 **reward 惩罚**建模（不改 Aer 噪声通道）
> - 门时间用 **per-gate-type 精细表**（rz=0, sx≈35ns, cx≈300ns）

## 动机与核心问题

当前框架对"门什么时候执行、执行多久、并行还是串行"没有任何建模：

1. **无双量子门并行**：`_auto_execute_batch()`（env.py:513）串行取最小 index 的门逐个执行，
   两个独立门即使在不同 qubit 对上也不并行 → 线路深度虚高、退相干被高估
2. **无累计时序**：噪声模型对每个门独立施加 `thermal_relaxation_error(T1, T2, 固定时长)`
   （sim.py:120-134），qubit 空闲等待时间的 T1/T2 衰减完全没有计入
3. **无调度感知串扰**：`_gate_crosstalk()`（env.py:282）只在 SWAP 步边界检查邻居占用，
   而非在"两个门相同时刻执行"时检查 → 并行串扰成本缺失
4. **门时间粒度粗糙**：所有 1Q 门固定 0.1µs、2Q 门固定 0.3µs（circuit_dag.py:132），
   实际 rz 是虚拟门（~0）、sx ~35ns、cx ~300ns

**核心物理权衡（本次要引入的）**：
- 并行门多 → 线路时间短（退相干少）→ 但相邻耦合对同时执行产生串扰（保真度损失）
- 顺序执行 → 串扰少 → 但线路时间长，idle qubit 累积 T1/T2 衰减（保真度损失）

## 方案总览：round-based 并行调度 + timing 观测/奖励

```
每个 RL step（SWAP 后）:
  while 存在依赖满足的 ready 门:
    Round N:
      1. 收集 ready 1Q + 2Q 门
      2. 贪心选最大独立集（共享 qubit 的互斥；相邻耦合对的并行仅记录串扰，不禁止）
      3. 并行执行 selected 门（同时写入 phys_circuit）
      4. round 耗时 = max(per-gate-type duration)；累计 total_time / qubit_idle_time / crosstalk_events
      5. 更新 qubit_busy_until
```

调度策略固定为 greedy 最大并行（环境内部），Agent 通过**路由决策间接控制调度效果**：
好路由让门均匀分布（并行多、串扰少、idle 少）；差路由让门拥挤（串扰大）或分散（耗时长）。

## 一、新增 `src/routing/timing.py` — 时序与调度模块

```python
GATE_DURATION_TABLE = {
    # 单位 µs（per-gate-type，替代固定 0.1/0.3）
    "cx": 0.30, "cz": 0.30, "ecr": 0.40, "swap": 0.30,
    "sx": 0.035, "x": 0.035, "y": 0.035, "h": 0.035,
    "s": 0.035, "t": 0.035, "sdg": 0.035, "tdg": 0.035,
    "rz": 0.0, "z": 0.0,            # 虚拟门，瞬时
    "barrier": 0.0, "id": 0.0, "measure": 2.0,
}

@dataclass
class CircuitTiming:
    total_time: float              # 累计线路执行时间 (µs)
    qubit_busy_until: np.ndarray   # [n_phys], 每个物理 qubit 的下次可用时刻
    qubit_idle_time: np.ndarray    # [n_phys], 累计空闲时间 (µs)
    rounds: int                    # 已执行并行轮数
    crosstalk_events: int          # 并行门落在相邻耦合对上的累计次数

def schedule_round(ready_gates, timing, hw, coupling_map):
    """选一轮最大独立集并行执行，返回 (selected_idx, round_time, xtalk_count)"""
```

调度要点：
- **互斥约束**（必须）：两个门共享 qubit → 不能同轮
- **串扰记录**（仅 reward，不阻塞）：两个门在相邻耦合对上同轮 → `crosstalk_events += 1`，
  累加 `hw.zz[pa, pb]` 作为加权串扰强度
- **优先级**：DAG 深度大（critical path 上）的门优先入选，减少关键路径阻塞
- **空闲轮**：无任何可执行门（2Q 门不 adjacent）时，qubit 空闲等待计入 `qubit_idle_time`

## 二、观测空间（State）改动 — `env.py`

在现有 `[edge_feats | map_vec | progress | phase]` 末尾追加 timing 特征：

| 特征 | 维度 | 说明 |
|------|------|------|
| `circuit_time_norm` | 1 | `total_time / (max_rounds × two_gate_time)` |
| `crosstalk_ratio` | 1 | 累计串扰事件数 / max(rounds, 1) |
| `parallelism_factor` | 1 | 平均每轮并行门数 / 理论最大并行度 |
| `qubit_idle_norm` | `max_num_qubits` | 每个物理 qubit 的 `idle_time / total_time` |

- `obs_dim += 4 + max_num_qubits`
- 对应修改 `agent.py` 的 `EdgeActorCritic.critic` 输入维度与 `_forward_obs` 切片偏移
- per-edge 特征（`_edge_feat_dim`）保持不变，timing 作为全局信号供 critic/commit head 使用

## 三、动作空间（Action）

**保持不变**：`Discrete(num_edges + 1)`（SWAP 边 + commit）。调度是环境内部确定性策略，
Agent 通过路由间接优化时序权衡。预留中期扩展：增加 `parallelism_mode` 离散维度
（保守/均衡/激进）让 Agent 显式控制并行度（当前不实现）。

## 四、奖励函数（Reward）改动 — `env.py`

新增 reward mode `timing_aware`：

```
r_step = r_exec + r_swap + r_dist + r_time + r_xtalk_par + r_idle + r_prop
r_time      = -η_time × (round_time / two_gate_time)        # 每轮耗时惩罚（线路时间）
r_xtalk_par = -η_xtalk_par × 本轮加权串扰强度                # 并行门串扰惩罚（关键新增）
r_idle      = -η_idle × (本轮 idle_time / max(total_time,1)) # qubit 空闲退相干惩罚
r_terminal  = λ_fid × Aer_fidelity                          # 与 noise_aware 相同
```

新增配置参数（env `__init__`）：

```python
eta_time: float = 0.01        # 每 two_gate_time 单位的时间惩罚
eta_xtalk_par: float = 0.05   # 每对并行串扰惩罚（按 hw.zz 强度加权）
eta_idle: float = 0.005       # 每 qubit·µs 空闲惩罚
```

SWAP 本身也耗时 0.3µs：`_apply_swap` 后更新 `timing.total_time` 与两端 qubit 的
`busy_until`，SWAP 等待期间相邻 qubit 的 idle 同样计入（体现"SWAP 多的线路更慢"）。

## 五、代码改动清单

| 文件 | 改动 | 说明 |
|------|------|------|
| `src/routing/timing.py` | **新建** | `GATE_DURATION_TABLE`、`CircuitTiming`、`schedule_round()`、`apply_gate_timing()` |
| `src/routing/rl/env.py` | 修改 | 集成 `CircuitTiming`；`_auto_execute_batch` 改 round 并行调度；`_obs` 追加 timing 特征；`step` 记录 SWAP 耗时；新 `reward_mode="timing_aware"`；`clone()` 复制 timing |
| `src/routing/graph/circuit_dag.py` | 修改 | `GateRecord` 增加 `duration` 字段（查 `GATE_DURATION_TABLE`）；`gate_duration_norm` 改用 per-type 表 |
| `src/routing/rl/agent.py` | 修改 | `EdgeActorCritic.critic_in` +4+N 维度；`_forward_obs` 切片偏移 |
| `src/routing/rl/train_agent.py` | 修改 | `--reward-mode timing_aware` 支持；`create_env` 透传 timing 参数；metrics 加 `time/xtalk/idle` 列 |
| `src/routing/rl/eval_policy.py` | 修改 | 评估打印 `circuit_time_ms`、`crosstalk_events`、平均并行度 |
| `src/sim/sim.py` | 轻改 | `NoiseConfig` 引用 `GATE_DURATION_TABLE`（Phase 2：`_add_combined_gate_errors` 按门类型取时长） |
| `doc/train.md` | 追加 | 训练记录（待实验后） |

## 六、训练流程

```
Phase 1 (routing)      → Phase 1.5 (timing_aware) → Phase 2 (noise_aware)
纯路由 + SABRE 特征       + 时间/串扰/空闲惩罚        + Aer 保真度终端奖励
```

Phase 1.5 让 Agent 在接触真实 fidelity 前先学会"并行 vs 串行"的时间权衡，再叠加噪声微调。

## 七、验证

1. `PYTHONPATH=src python3 -m pytest test/ -q`
2. 短训练冒烟测试（`--timesteps 5000`，5q 三拓扑，`--reward-mode timing_aware`）：
   - 确认 `crosstalk_ratio`、`parallelism_factor` 收敛（并行度 > 1，串扰非零）
   - 对比 `routing` 模式的线路时间（timing_aware 应更短）
3. 与 SABRE 对比评估：5q + 20q 三拓扑，记录 SWAPs / circuit_time / fidelity 三指标

## 预期收益与风险

- **收益**：
  - 路由结果在"时间维度"上可比较（线路时间成为显式优化目标）
  - 并行执行缩短线路 → 与真实硬件（Aer 时间感知热弛豫）行为一致
  - 并行串扰惩罚让 Agent 学会"错开相邻门"，体现真实调度约束
  - per-gate-type 时长让 rz 虚拟门免耗时、sx/cx 差异体现 → 更接近物理
- **风险**：
  - greedy 调度不可学习，Agent 无法显式控制并行度 → 中期可加 `parallelism_mode` 动作维度
  - `crosstalk_events` 计数若与 `hw.zz` 强度脱节会误导训练 → 用加权强度而非计数
  - obs 维度变更影响面大 → 同步更新 `_forward_obs` / `_build_edge_obs` / 测试断言

## 备选方案（未采纳，供参考）

- **B：Agent 控制调度**——动作空间扩为 `(swap, parallelism_mode)`，Agent 显式决定并行度。
  更灵活但训练复杂、reward 信号需精细设计（当前环境内部固定策略，先验证时序建模本身）
- **C：Aer 噪声通道建模串扰**——在物理电路插入 barrier 标记并行段，让 Aer 在并行门上
  叠加串扰错误通道。更精确但 Aer 对动态并行调度支持有限，实现复杂（reward 惩罚先行，
  终端 fidelity 兜底）
- **D：门时长保持固定**（1Q=0.1/2Q=0.3）——改动最小，但 rz 虚拟门与真实硬件差距大，
  用户已确认采用 per-gate-type 表

---

# Plan: 图编码 timing 扩展（timing-aware graph encoding v1）

> 状态：待实施（写自 2026.08.06）
> 依赖：`src/routing/timing.py` 已创建（`GATE_DURATION_TABLE`、`CircuitTiming` 数据类可用）
> 范围：仅图编码（GNN 输入特征）改动，不含调度器实现本身（见上一节）

## 核心原则

1. **不改维度数**：`NODE_FEATURE_DIM=32`、`EDGE_FEATURE_DIM=16` 保持不变。空闲维度充分
   （qubit dims 19-31、gate dims 27-31、couples edge dims 12-15、maps-to edge dims 13-15）
2. **GNN 架构不改**：`SubGNN`、`GATEncoder` 的 `node_dim`/`edge_dim` 入参不变，
   只改 `build_routing_graph()` 内特征填充逻辑
3. **模板复用**：qubit template（`_build_qubit_template`）与 coupling template
   （`_build_coupling_template`）保持硬件静态预计算不变；新增的 timing 维度在
   `build_routing_graph()` 动态段每一步覆盖
4. **CircuitTiming 来源**：调度器运行后 `self.timing` 已处于最新状态，
   `_obs()` → `build_graph_data()` → `build_routing_graph()` 传入 `timing` 参数读取即可

## 一、门节点特征（Gate Node）改动

### 1.1 dim 14 精炼：per-gate-type duration

**文件**：`src/routing/graph/circuit_dag.py`

**当前代码**（line 132）：
```python
dur = 0.3 / _T1_SCALE if g.is_two_qubit else 0.1 / _T1_SCALE
```

**改为**：
```python
from ..timing import GATE_DURATION_TABLE
dur = GATE_DURATION_TABLE.get(g.name, 0.3) / _T1_SCALE
```

> 注意：`_T1_SCALE` 定义于 `features.py:27`（=100.0）。方案：在 `timing.py` 中
> `from ..graph.features import _T1_SCALE` 复用，避免两处定义不一致。

**影响范围**：`build_gate_template()` 内模板预计算（line 132）；`build_routing_graph()`
内 `gate_template = dag.build_gate_template()`（line 325）自动继承。

### 1.2 dim 27 新增：`scheduling_competition`（调度竞争度）

**含义**：该门与 front-layer 其他 ready 门共享 qubit 的数量 / |front_layer|。
竞争度 0 = 可单独并行执行（无 qubit 冲突）；竞争度 1 = 与所有其他门冲突 → 调度瓶颈。

**伪代码**（在 `build_routing_graph()` gate 循环内，line 324-361 附近）：
```python
front_layer = [gg for gg in dag.gates
               if not executed_mask[gg.index] and gg.is_two_qubit
               and all(executed_mask[p] for p in gg.predecessors)]
if g.is_two_qubit and not executed_mask[g.index]:
    conflicts = sum(
        1 for other in front_layer
        if other.index != g.index and set(g.qubits) & set(other.qubits))
    gate_feat[g.index, 27] = conflicts / max(1, len(front_layer))
else:
    gate_feat[g.index, 27] = 0.0
```

### 1.3 dim 28 新增：`relative_criticality`（相对关键度）

**含义**：该门 `remaining_depth / Σ(front_layer remaining_depths)`，在关键路径上的权重。
高值 = 延迟执行会拖长总线路时间 → 调度器应优先，agent 应优先让该门的 qubit 相邻。

**伪代码**：
```python
if g.is_two_qubit and not executed_mask[g.index]:
    rem = remaining.get(g.index, 0)
    total_rem = sum(remaining.get(gg.index, 0) for gg in front_layer)
    gate_feat[g.index, 28] = rem / max(1, total_rem)
else:
    gate_feat[g.index, 28] = 0.0
```

### 1.4 `GateRecord` 增加 `duration` 字段（可选）

**文件**：`src/routing/graph/circuit_dag.py` line 19-29

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
    predecessors: List[int] = field(default_factory=list)
    duration: float = 0.0   # 新增：查 GATE_DURATION_TABLE 填充
```

`CircuitDAG.from_circuit()` 构造 `GateRecord` 时查表填入；`build_gate_template()`
直接用 `g.duration` 而非重复查表。

## 二、物理 qubit 节点特征（Qubit Node）改动

当前 qubit 节点是模板 + 每步增量更新 dims 13-18（occupied、occupancy、exec_dep、
near_dist、occ_neighbor_norm，line 364-379）。时序信息在**每步调度运行完后**填充
新增动态维度：

### 2.1 dim 19 新增：`idle_ratio`（空闲时间占比）

**数据来源**：`timing.qubit_idle_time` / `max(total_time, 1e-8)`。

**填充**（`build_routing_graph()` qubit 循环内）：
```python
total_time = timing.total_time if timing is not None else 1.0
qubit_feat[pq, 19] = (
    timing.qubit_idle_time[pq] / max(total_time, 1e-8) if timing else 0.0)
```

**物理含义**：高值 = qubit 大部分时间空闲 → 浪费相干资源 → agent 应优先将门路由至此
qubit（降低 idle）。

### 2.2 dim 20 新增：`crosstalk_exposure`（串扰暴露度）

**数据来源**：`CircuitTiming` 新增字段 `qubit_crosstalk: np.ndarray [n_phys]`
（`schedule_round()` 每轮结算时，凡本 qubit 参与并行且相邻 coupling 对上也有门执行，
累加 `hw.zz[pq, neighbor]`）。

**填充**：
```python
max_xtalk = float(timing.qubit_crosstalk.max()) if timing else 1e-8
qubit_feat[pq, 20] = (
    timing.qubit_crosstalk[pq] / max(max_xtalk, 1e-8) if timing else 0.0)
```

**物理含义**：高值 = 该 qubit 反复受邻居并行门串扰 → agent 应减少在此 qubit 附近
同时安排门。

### 2.3 dim 21 新增：`active_ratio`（活跃时间占比）

```python
qubit_feat[pq, 21] = 1.0 - qubit_feat[pq, 19]   # active = 1 - idle_ratio
```

**物理含义**：高值 = 忙碌 qubit → 新门路由到此可能增加等待时间。agent 应平衡各 qubit
负载。（与 dim 19 冗余，保留作显式特征供 GNN 非线性编码提取不同语义。）

## 三、耦合边特征（Couples Edge）改动

当前 couples edge 是 `_build_coupling_template()` 预计算的**硬件静态模板**
（features.py:345-381），每步只在 `build_routing_graph()` 覆盖 dim 10
`both_occupied`（line 398-403）。新增每步动态维度：

### 3.1 dim 12 新增：`parallel_crosstalk_risk`（并行串扰风险）

**含义**：过去几轮调度中，这对 qubit 两端各有门**同时并行执行**的比例。

**数据来源**：`CircuitTiming` 新增字段 `parallel_usage: np.ndarray [n_phys, n_phys]`，
`schedule_round()` 每轮结算时对"该轮同时活跃且相邻的 coupling pair"累加 1.0。

**填充**（line 398-403 附近）：
```python
total_rounds = max(timing.rounds, 1) if timing else 1
risk = timing.parallel_usage[q1, q2] / total_rounds if timing else 0.0
coup_attr_arr[i, 12] = risk
coup_attr_arr[i + 1, 12] = risk
```

**物理含义**：高值 = 这对 qubit 频繁并行使用 → 串扰可能性大 → agent 应避免在此边上
同时安排新操作。

### 3.2 dim 13 新增：`idle_product`（空闲度乘积）

```python
if timing:
    idle_p = timing.qubit_idle_time[q1] / max(total_time, 1e-8)
    idle_q = timing.qubit_idle_time[q2] / max(total_time, 1e-8)
    idle_prod = float(np.sqrt(idle_p * idle_q))
else:
    idle_prod = 0.0
coup_attr_arr[i, 13] = idle_prod
coup_attr_arr[i + 1, 13] = idle_prod
```

**物理含义**：高值 = 两端 qubit 都空闲 → 此 coupling edge 是执行 SWAP 或安排新门的
理想候选；低值 = 至少一端忙 → SWAP 会引入额外等待。

## 三、全局观测向量追加（不进图，供 critic）

**文件**：`src/routing/rl/env.py`，`_obs()`（line 231-269）

在现有 `obs = [edge_feats | map_vec | progress | phase]` 末尾追加：

```python
if self.timing is not None:
    total_time = max(self.timing.total_time, 1e-8)
    circuit_time_norm = np.array(
        [self.timing.total_time / max(self.max_episode_steps, 1)], dtype=np.float32)
    crosstalk_ratio = np.array(
        [self.timing.crosstalk_events / max(self.timing.rounds, 1)], dtype=np.float32)
    parallelism_factor = np.array(
        [self.timing.total_gates_executed / max(self.timing.rounds * self.num_qubits, 1)],
        dtype=np.float32)
    idle_norm = (self.timing.qubit_idle_time / total_time).astype(np.float32)
    if self.max_num_qubits > self.num_qubits:
        idle_norm = np.pad(idle_norm, (0, self.max_num_qubits - self.num_qubits))
    timing_obs = np.concatenate(
        [circuit_time_norm, crosstalk_ratio, parallelism_factor, idle_norm])
    obs = np.concatenate([obs, timing_obs])
```

**obs_dim 变化**：
```
before: n_ef + n_q_max + 1 + 1
after:  n_ef + n_q_max + 1 + 1 + 4 + n_q_max
```

**注意**：为保持 obs_dim 固定（RL buffer 需要），`timing` 需始终存在（非
timing_aware 模式填充零值），避免条件分支导致维度不一致。

## 五、`CircuitTiming` 数据结构扩展

**文件**：`src/routing/timing.py`

在原设计基础上新增 3 个字段：

```python
@dataclass
class CircuitTiming:
    total_time: float = 0.0
    qubit_busy_until: np.ndarray = None          # [n_phys]
    qubit_idle_time: np.ndarray = None           # [n_phys]
    qubit_crosstalk: np.ndarray = None           # [n_phys]          新增
    parallel_usage: np.ndarray = None            # [n_phys, n_phys]   新增
    rounds: int = 0
    crosstalk_events: float = 0.0
    total_gates_executed: int = 0                # 新增

    def clone(self) -> "CircuitTiming": ...      # 供 beam search 用
```

初始化时按 `n_phys` 分配零数组；`schedule_round()` 每轮结算更新。

## 六、`build_routing_graph()` 函数签名改动

**文件**：`src/routing/graph/circuit_dag.py` line 269

**当前签名**：
```python
def build_routing_graph(
    dag: CircuitDAG, mapping: List[int], hw: HardwareFeatures,
    coupling_map: List[Tuple[int, int]],
    single_gate_time: float = 0.1, two_gate_time: float = 0.3,
    executed_mask: Optional[np.ndarray] = None,
    executable_2q: Optional[set] = None,
) -> RoutingGraphData:
```

**改为**：
```python
def build_routing_graph(
    dag: CircuitDAG, mapping: List[int], hw: HardwareFeatures,
    coupling_map: List[Tuple[int, int]],
    executed_mask: Optional[np.ndarray] = None,
    executable_2q: Optional[set] = None,
    timing: Optional["CircuitTiming"] = None,   # 新增，None = 无 timing 信息，填充 0
) -> RoutingGraphData:```
```

- 删除 `single_gate_time` / `two_gate_time`（已由 `GATE_DURATION_TABLE` 替代）
- **所有调用方适配**：`env.py` 的 `_update()` 调用处、`build_graph_data()` 传入
  `self.timing`；`test/` 内调用加 `timing=None`

## 七、每个文件的具体改动量

### 7.1 `src/routing/timing.py`（新建，约 120 行）

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Set
import numpy as np

GATE_DURATION_TABLE = {
    "cx": 0.30, "cz": 0.30, "ecr": 0.40, "swap": 0.30,
    "sx": 0.035, "x": 0.035, "y": 0.035, "h": 0.035,
    "s": 0.035, "t": 0.035, "sdg": 0.035, "tdg": 0.035,
    "rz": 0.0, "z": 0.0,
    "barrier": 0.0, "id": 0.0, "measure": 2.0,
}
FALLBACK_DURATION = 0.3

@dataclass
class CircuitTiming:
    total_time: float = 0.0
    qubit_busy_until: Optional[np.ndarray] = None     # [n_phys]
    qubit_idle_time: Optional[np.ndarray] = None      # [n_phys]
    qubit_crosstalk: Optional[np.ndarray] = None       # [n_phys]
    parallel_usage: Optional[np.ndarray] = None        # [n_phys, n_phys]
    rounds: int = 0
    crosstalk_events: float = 0.0
    total_gates_executed: int = 0

    @classmethod
    def create(cls, n_phys: int) -> "CircuitTiming":
        return cls(
            qubit_busy_until=np.zeros(n_phys, dtype=float),
            qubit_idle_time=np.zeros(n_phys, dtype=float),
            qubit_crosstalk=np.zeros(n_phys, dtype=float),
            parallel_usage=np.zeros((n_phys, n_phys), dtype=float),
        )

    def clone(self) -> "CircuitTiming":
        new = CircuitTiming.create(len(self.qubit_busy_until))
        new.total_time = self.total_time
        new.qubit_busy_until[:] = self.qubit_busy_until
        new.qubit_idle_time[:] = self.qubit_idle_time
        new.qubit_crosstalk[:] = self.qubit_crosstalk
        new.parallel_usage[:] = self.parallel_usage
        new.rounds = self.rounds
        new.crosstalk_events = self.crosstalk_events
        new.total_gates_executed = self.total_gates_executed
        return new


def schedule_round(results, ...):
    """选一轮最大独立集并行执行，返回 (selected_gate_indices, round_time)。"""
    ...
```

（`schedule_round` 完整实现见上一节「一、新增 `src/routing/timing.py`」的调度要点：
互斥约束、串扰记录、criticality 优先级、空闲轮结算。）

### 7.2 `src/routing/graph/circuit_dag.py`（约 +50 行）

| 位置 | 改动 |
|------|------|
| `GateRecord` dataclass | +1 字段 `duration` |
| `CircuitDAG.from_circuit()` | 构造 `GateRecord` 时查表填 `duration` |
| `CircuitDAG.build_gate_template()` line 132 | `dur = GATE_DURATION_TABLE.get(g.name, 0.3) / _T1_SCALE` |
| `build_routing_graph()` 函数签名 | `-single_gate_time, -two_gate_time`，`+timing: Optional[CircuitTiming]` |
| `build_routing_graph()` gate 循环（line 324-361） | 新增 dim 27, 28（见 1.2/1.3） |
| `build_routing_graph()` qubit 循环（line 364-379） | 新增 dim 19, 20, 21（见 2.1/2.2/2.3） |
| `build_routing_graph()` couples 循环（line 398-403） | 新增 dim 12, 13（见 3.1/3.2） |

### 7.3 `src/routing/rl/env.py`（约 +60 行）

| 位置 | 改动 |
|------|------|
| `__init__` | 新增 `eta_time=0.01, eta_xtalk_par=0.05, eta_idle=0.005`；timing_aware 模式实例化 `CircuitTiming.create(n_phys)` |
| `reset` | `self.timing = CircuitTiming.create(self.num_qubits)`（timing_aware 模式） |
| `_update` | 1Q 门**不再直接执行**（当前 line 199-204），改为返回 `ready_1q` 列表供调度器 |
| `_auto_execute_batch` | 完全重写：`while ready_1q or ready_2q: schedule_round(...)` 替代串行 `min(executable_2q)` 循环 |
| `step` | SWAP 耗时 0.3µs：`_apply_swap()` 后 `timing.total_time += 0.3`，`qubit_busy_until[p]=qubit_busy_until[q]=total_time`；SWAP 等待期间 idle 结算 |
| `_step_reward_execute` | 新增 `r_time`, `r_xtalk_par`, `r_idle` 分支（仅 `reward_mode=="timing_aware"`） |
| `_obs` | 末尾追加 timing_obs（见第四节） |
| `build_graph_data` | 传入 `timing=self.timing` |
| `clone` | 新增 `new.timing = self.timing.clone() if self.timing else None` |
| `_get_terminal_reward_value` | 保持原有 `fidelity_fn` / `_compute_aer_fidelity` 不变 |

### 7.4 `src/routing/rl/agent.py`（约 +15 行）

| 位置 | 改动 |
|------|------|
| `EdgeActorCritic.__init__` | `critic_in` 维度 `edge_feat_dim + n_q + 1 + 1` → 追加 `4 + n_q_max` |
| `EdgeActorCritic._forward_obs` | 切片偏移适配：timing_obs 从旧末尾开始截取传入 critic |
| `EdgeActorCritic._build_edge_obs` | batch buffer 扩张容纳 `4 + n_q_max` 维 timing |

### 7.5 `src/routing/rl/train_agent.py`（约 +10 行）

| 位置 | 改动 |
|------|------|
| 命令行参数 | `--reward-mode timing_aware` 加入 choices |
| `create_env` | 透传 `eta_time`, `eta_xtalk_par`, `eta_idle` |
| metrics.csv | 新增列：`avg_time_ms`, `avg_xtalk_events`, `avg_parallelism` |
| rollout 日志 | 打印 `time/xtalk/par` 三指标 |

### 7.6 `src/routing/rl/eval_policy.py`（约 +15 行）

| 位置 | 改动 |
|------|------|
| `CircuitMetrics` | 新增 `circuit_time_us: float`, `crosstalk_events: float`, `avg_parallelism: float` |
| 报告输出 | 打印每电路 timing 指标 + 汇总 |

## 八、`_update` / `_auto_execute_batch` 重写细节

### 8.1 当前问题

当前 `_update()`（env.py:191-213）把 1Q 门直接执行（`executed.add` + `_phys_circuit.append`），
2Q 门收集到 `executable_2q` 后由 `_auto_execute_batch` 串行执行 `min(executable_2q)`：
1Q 门不参与调度（无法与 2Q 门并行）、串行无并行 → 需要重写。

### 8.2 新设计

```python
def _update(self) -> Tuple[List[int], List[int]]:
    """返回 ready_1q 和 ready_2q 列表，不直接执行任何门。"""
    ready_1q, ready_2q = [], []
    for g in self.dag.gates:
        if g.index in self.executed:
            continue
        if not all(p in self.executed for p in g.predecessors):
            continue
        if g.is_measure:
            continue
        if g.is_two_qubit:
            qa, qb = g.qubits
            pa, pb = self.mapping[qa], self.mapping[qb]
            if self.hw.adj[pa, pb] > 0:
                ready_2q.append(g.index)
        else:
            ready_1q.append(g.index)
    return ready_1q, ready_2q


def _auto_execute_batch(self) -> Tuple[float, float]:
    r_exec, r_prop = 0.0, 0.0
    while True:
        ready_1q, ready_2q = self._update()
        if not ready_1q and not ready_2q:
            break
        selected, round_time = schedule_round(
            ready_1q, ready_2q, self.dag, self.mapping, self.timing, self.hw)
        for gate_idx in selected:
            g = self.dag.gates[gate_idx]
            self.executed.add(gate_idx)
            self._last_progress_swap = len(self._swap_history)
            if not g.is_measure:
                pq = [self.mapping[q] for q in g.qubits]
                self._phys_circuit.append(g.operation, pq)
            r_exec += self._step_reward_execute(True, gate_idx)
            r_prop += self._step_reward_propagate(gate_idx)
    return r_exec, r_prop
```

## 九、附录 A：`schedule_round()` 详细实现

```python
def schedule_round(
    ready_1q: List[int], ready_2q: List[int],
    dag, mapping, timing: CircuitTiming, hw,
) -> tuple[List[int], float]:
    n_phys = len(timing.qubit_busy_until)
    current_time = timing.total_time

    entries = []
    for gidx in ready_1q:
        g = dag.gates[gidx]
        pq = mapping[g.qubits[0]]
        if timing.qubit_busy_until[pq] > current_time:
            continue
        dur = GATE_DURATION_TABLE.get(g.name, FALLBACK_DURATION)
        crit = dag.remaining_depths().get(gidx, 0) / max(1, dag.max_depth())
        entries.append((gidx, [pq], dur, crit))
    for gidx in ready_2q:
        g = dag.gates[gidx]
        pa, pb = mapping[g.qubits[0]], mapping[g.qubits[1]]
        if timing.qubit_busy_until[pa] > current_time or \
           timing.qubit_busy_until[pb] > current_time:
            continue
        dur = GATE_DURATION_TABLE.get(g.name, FALLBACK_DURATION)
        crit = dag.remaining_depths().get(gidx, 0) / max(1, dag.max_depth())
        entries.append((gidx, [pa, pb], dur, crit))

    if not entries:
        # 无任何可调度门：2Q 不 adjacent → 本步空转，不累进时间
        return [], 0.0

    # 按 criticality 降序 → 贪心最大独立集（互斥约束）
    entries.sort(key=lambda x: -x[3])
    selected, used_qubits = [], set()
    for gidx, phys_qubits, dur, crit in entries:
        if set(phys_qubits) & used_qubits:
            continue
        selected.append(gidx)
        used_qubits.update(phys_qubits)

    # 串扰检测：并行执行的相邻 coupling 对（仅 reward，不阻塞）
    active_pairs = []
    for i in range(len(selected)):
        pa_list = _qubits_of(selected[i], dag, mapping)
        for j in range(i + 1, len(selected)):
            pb_list = _qubits_of(selected[j], dag, mapping)
            for pa in pa_list:
                for pb in pb_list:
                    if pa != pb and hw.adj[pa, pb] > 0:
                        zz = float(hw.zz[pa, pb])
                        timing.crosstalk_events += zz
                        timing.qubit_crosstalk[pa] += zz
                        timing.qubit_crosstalk[pb] += zz
                        round_pairs.append((pa, pb))

    # 并行使用统计（全活跃 qubit 对的耦合对计数）
    active_qubits = set()
    for _, phys_qs, _, _ in entries:
        active_qubits.update(phys_qs)
    for pa in active_qubits:
        for pb in active_qubits:
            if pa < pb and hw.adj[pa, pb] > 0:
                timing.parallel_usage[pa, pb] += 1.0
                timing.parallel_usage[pb, pa] += 1.0

    # 结算 timing
    round_time = max(e[2] for e in entries if e[0] in selected) if selected else 0.0
    timing.rounds += 1
    timing.total_time += round_time
    timing.total_gates_executed += len(selected)
    for gidx, phys_qubits, _, _ in entries:
        if gidx in selected:
            for pq in phys_qubits:
                timing.qubit_busy_until[pq] = timing.total_time

    # 空闲结算：本轮未执行门的 qubit 空闲累加 round_time
    for pq in range(n_phys):
        if timing.qubit_busy_until[pq] <= current_time:
            timing.qubit_idle_time[pq] += round_time

    return selected, round_time
```

> 注：`_phys_of(entry, ...)` 为取 entry 的物理 qubit 列表的辅助函数；`schedule_round()`
> 亦需对 `used_qubits` 之外的 shared-qubit 互斥做边界处理（1Q 门之间同 qubit 互斥）。

## 十、改动依赖关系

```
timing.py (新建，GATE_DURATION_TABLE)
    ↓
circuit_dag.py (GateRecord.duration, build_gate_template dim 14,
                build_routing_graph 新增 dims 27/28/19/20/21/12/13)
    ↓
env.py (_update 拆分 ready_1q/ready_2q，_auto_execute_batch 重写，
        _obs 追加 timing_obs，clone 复制 timing)
    ↓
agent.py (critic_in / _forward_obs 维度适配)
    ↓
train_agent.py (reward_mode timing_aware + metrics)
    ↓
eval_policy.py (timing 指标输出)
```

**建议实施顺序**：`timing.py` → `circuit_dag.py` → `env.py` → `agent.py` →
`train_agent.py` → `eval_policy.py`

## 十一、向后兼容

- `reward_mode != "timing_aware"` 时 `timing` 填充零（而非 None），obs 维度恒定
- `build_routing_graph(timing=None)` 兼容旧调用方（`test/` 内需显式传 `timing=None`
  或省略默认参数）
- `GATE_DURATION_TABLE` 引入仅替代固定 0.1/0.3 的取值，不改变特征语义
  （rz 由 0.1 → 0 的小幅分布变化在可接受范围）
- `_update` 拆分 `ready_1q` 返回值变动 → 同步修改 `_auto_execute_batch` 和
  `step` 中映射阶段分支的调用

## 十二、验证

1. `PYTHONPATH=src python3 -m pytest test/ -q`（全量回归）
2. 短训练冒烟测试（`--timesteps 5000`，5q 三拓扑，`--reward-mode timing_aware`）：
   - 确认 `crosstalk_ratio`、`parallelism_factor` 收敛（并行度 > 1，串扰非零）
   - 对比 routing 模式线路时间（timing_aware 应更短）
3. 与旧 `routing` 模式对比 GNN 输入分布：dim 14（duration）从 {0.003, 0.001} 变为
   离散多值，dim 19-21/27-28/12-13 均为非零敏感特征，确认无 NaN/极值
4. 20q 三拓扑评估（unified_test，random-init）：记录 SWAPs / circuit_time /
   fidelity 三指标对比 SABRE

---

# Plan: 端到端路由 + 门调度联合优化（routing + scheduling E2E）

> 状态：待实施（写自 2026.08.06）
> 前置：timing-aware 基础版（`timing.py` + 图编码 timing 扩展）已落地
> 动机：当前调度器是固定贪心策略，agent 只能间接影响"哪些门变 adjacent"，
>       无法显式控制"哪些门先执行 / 串行化避串扰 / 关键路径优先"

## 核心问题：调度动作空间设计

调度动作不能是"选一个门子集"——组合爆炸（5q 上 2^10 = 1024，20q 不可行）。
三种候选方案：

| 方案 | 动作 | 组合空间 | GNN 开销 | 调度自由度 | 成本 |
|------|------|---------|---------|-----------|------|
| **A. 优先度连续控制** | 给每个 ready 门打 `priority score ∈ [0,1]`，环境按分数贪心调度 | 连续向量（无爆炸） | 每 SWAP 步 1 次 | 高（近似排序） | ~80 行 |
| **B. 顺序门选择** | SWAP 后反复"选下一个门执行 or 结束" | Discrete(K+1)，K=ready 门数 | 每选 1 个门重算 1 次 | 最高（自适应后续状态） | ~150 行 |
| **C. 分层 RL** | 高层做 SWAP，低层做顺序选门，两层独立训练 | 同上 | 同 B | 高（+层间解耦） | ~200 行 |

## 方案 A：优先度连续控制（推荐起点，~80 行）

### A1. 架构：新增 `gate_score_head`

```
GNN (每 SWAP 步 1 次 forward)
   ├── gate_embeddings [N_gates, d]  →  gate_score_head: Linear(d→32)ReLU→Linear(32→1) → sigmoid
   └── qubit_embeddings [P, d]       →  edge_rawscore_head: existing edge_mlp（不变）
```

- gate_score_head 共享 GNN backbone，独立于 edge policy
- `agent.py: EdgeActorCritic.forward()` 同时输出 `edge_logits` + `gate_scores` + `value`
- 动作仍为 SWAP（edge），gate_scores 只是**额外传给调度器的控制信号**（非 RL 动作）

### A2. `schedule_round` 改造

```python
def schedule_round(ready_1q, ready_2q, dag, mapping, timing, hw,
                   gate_scores=None):
    # 原排序键 criticality 换成:
    #   prio = gate_scores[gidx] if gate_scores is not None else criticality(gidx)
    entries.sort(key=lambda e: -prio(e[0]))
    # 其余（互斥约束 / 串扰记录 / 空闲结算）不变
```

- `gate_scores` 为 None → 退化为 criticality 贪心（向后兼容）
- agent 语义：高分 = 优先执行（关键路径门）；低分 = 延迟执行（串扰避让）
- **局限**：所有 round 共享同一组 scores（GNN 不重跑，gate embedding 不更新）——
  等价于"agent 决定初始优先级，调度器按优先级排多轮"，对大多数场景足够

### A3. 训练信号

`r_time + r_idle + r_xtalk_par` 通过 GAE 同时塑造 edge policy 与 gate_score_head：
- 给高 zz 边上的门打低分 → 串行化 → 无串扰惩罚但多占时间
- 给 critical path 门打高分 → 优先执行 → 缩短总线路时间
- 两信号通过 `η_xtalk_par vs η_time + η_idle` 的权重权衡出最优策略

### A4. 局限

- score 一次性（不随调度推进自适应）
- agent 不能显式表达"round 1 执行 A，round 2 根据 A 结果再定 B"

## 方案 B：顺序门选择（完整控制，~150 行）

```
每个 RL step:
  1. SWAP 动作（现有）
  2. [scheduling 子 episode]:
     while ready_gates 非空:
       obs = [gate_feats 更新 | timing state]
       action ∈ {0..K-1, DONE}   # K = ready 门数 + 结束动作
       execute selected gate → update timing → recompute ready gates
       reward = r_time_step + r_idle_step + r_xtalk_step
  3. 回到 SWAP 决策
```

- 动作 masking：仅 ready 门 + DONE 可选；变长离散动作空间（需 padding mask）
- GNN 每选一个门重跑一次（或复用嵌入 + 轻量 gate_feat 增量更新，参照 beam 缓存）
- 子 episode horizon ≈ 门数 → 信用分配更难，需逐步落地

---

## 推荐路径

```
Phase 0 (routing)      → Phase 1 (timing_aware, 固定调度)
                       → Phase 2 (timing_aware + gate_score_head, 方案 A)
                       → Phase 3 (方案 B，若 A 不够)
```

1. 先实现方案 A（改动最小、无缝接入现有 per-edge 架构）
2. A 验证并行度/串扰指标提升后，评估是否需要 B 的逐门自适应
3. B 可复用 A 的 gate_score_head 作为 warm-start 初始化

---

## 验证（方案 A 完成后）

1. `PYTHONPATH=src python3 -m pytest test/ -q`
2. 5q 三拓扑短训练（`--timesteps 5000`，timing_aware + gate_score）：对比固定调度版
   - 期望：parallelism_factor 提升、crosstalk_events 下降、circuit_time 下降
3. 检查 gate_scores 分布：critical-path 门 avg score 应 > 非 critical 门 avg score
4. 20q 评估，对比三指标（SWAPs / circuit_time / fidelity）vs 固定调度 + SABRE

---

## 十三、60q 截断率优化：自适应 GAE λ + 跨规模课程（已实现）

### 13.1 背景

tianyan 训练（`scripts/train_tianyan.sh`，MAX_STEPS=800）在 ~300k 步时 `trunc_pct≈76.7%`。
分析发现两个根因：

1. **GAE λ 固定导致长电路信号缺失**：`γ·λ = 0.99×0.95 = 0.9405`，有效 horizon ~17 步。
   800 步 episode 中截断终端信号 `0.9405^800 ≈ 2.4e-18`，前 ~780 步完全接收不到
   `unfinished_penalty` 惩罚 → agent 只优化最后 20 步，无长期规划。
2. **无课程学习**：tianyan splits 只含 n30-n60 电路，agent 从第 1 步就面对大电路，
   从未在 n8-n20 电路上学到基础路由先验。

### 13.2 改动

| 文件 | 改动 |
|------|------|
| `agent.py:compute_gae` | `lam` 支持标量或数组（逐时刻 λ_t），数组时在反向 GAE 中使用 `_lam[t]`；形状不符报错 |
| `train_agent.py:curriculum_phase` | 按 `progress·n` 选 prefix，prefix 内局部进度复用 `stage1_phase` 的 depth 递进；支持 routing/noise_aware/fidelity_shaping 三种 split 名 |
| `train_agent.py:build_multi_split_map` | 合并多 prefix 的 split manifest（key 为完整 split 名） |
| `train_agent.py:pick_circuit` | 新增 `split_map` 参数（不传则回退单 prefix），复用预建 map |
| `train_agent.py` CLI | `--curriculum-keys`、`--gae-adaptive`、`--gae-lam-min`（0.95）、`--gae-lam-max`（0.995） |
| `train_agent.py` GAE 调用 | `--gae-adaptive` 时 `λ_t = λ_min + (λ_max−λ_min)·(t/(T−1))`，否则 `λ=agent.lam`（向后兼容） |
| `scripts/train_tianyan_curriculum.sh` | 新脚本：课程 + 自适应 λ，Phase1 默认 500k 步 |

### 13.3 课程设计

| 全局进度 | split_prefix | 局部进度 | 线路规模 |
|---------|-------------|---------|---------|
| 0-33% | large_n10 | 0-1（phase1→3） | n10, 6-50 2q 门 |
| 33-66% | large_n20 | 0-1 | n20, 12-100 |
| 66-100% | tianyan | 0-1 | n30-60, 12-300 |

### 13.4 验证

- `curriculum_phase` 边界单测：0.33→large_n10_phase3、0.34→large_n20_phase1、
  0.66→large_n20_phase3、0.67→tianyan_phase1 均正确
- `compute_gae` 数组 λ：常数数组与标量 λ 结果一致（断言通过）；自适应 λ 使
  episode 开头 advantage 更小、末端更大
- 端到端 smoke：`--curriculum-keys large_n10,large_n20,tianyan --gae-adaptive` 在
  tianyan176 拓扑上 512 步跑通，初始 split 正确选择 `large_n10_phase1`
- 测试套件：`2 failed, 33 passed`。两个失败均为先前会话未提交改动所致（
  `data_gen.py` 的 `max_operands=2` 改变测试用随机电路导致 `test_swap_penalty`
  断言失效；`test_fidelity_shaping_step_zero` 随机动作遇距离奖励而 flaky），
  与本实现无关（stash 掉 agent.py/train_agent.py 后仍失败）

### 13.5 tmux 训练命令

```bash
# 课程 + 自适应 λ，Phase1 500k 步（n10:0-167k / n20:167-333k / tianyan:333-500k）
tmux new-session -d -s curric 'cd /home/zzy/opencode-server/opencode-docker/projects/rlrouting && bash scripts/train_tianyan_curriculum.sh cuda:0 500000 200000 2>&1 | tee logs/train_curric.log'
tmux attach -t curric   # 查看进度；按 Ctrl-B 然后 d 脱离
```
