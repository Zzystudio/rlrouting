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
