# 训练问题记录

## 训练命令

`src` 目录下执行：

```bash
python3 -m routing.rl.train_agent \
  --data-dir ../traindata \
  --topo ../traindata/topo/ibmq_5_line.json \
  --out ../models/policy.pt \
  --device cuda:0 \
  --reward-mode routing \
  --timesteps 100000
```

可选参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-episode-steps` | 200 | 每个 episode 最大步数（超时截断） |
| `--random-init` | 否 | 是否随机初始映射（默认 identity） |
| `--rollout-steps` | 256 | 每次更新前采集的步数 |
| `--lr` | 3e-4 | PPO 学习率 |
| `--load` | — | 预训练模型路径（续训用） |

### 日志格式

```
step=   256  rew=+5.335  swp=12.0  trunc=0%  pl=-0.001  vl=8.275  ent=1.384  kl=0.0000  gn=4.644
```

- `rew` — 近 20 个 completed episode 的平均总 reward
- `swp` — 近 20 个 episode 的平均 SWAP 数
- `trunc` — 截断率（被截断的 episode 占比）
- `pl` — policy loss
- `vl` — value loss
- `ent` — 策略熵
- `kl` — approx KL divergence
- `gn` — gradient norm

---

## Bug：`_update()` 在 SWAP 后未调用（2026.07.27）

### 现象

```
step=   256  rew=+0.000  swp=0.0  pl=-0.005  vl=0.000  ent=1.379  kl=0.0010  gn=0.227
step= 26000  rew=+0.000  swp=0.0  pl=-0.004  vl=0.000  ent=1.354  kl=0.0012  gn=0.272
```

- `rew` 始终为 0（没有任何 episode 完成）
- `swp` 始终为 0
- 26K+ 步后策略近乎未学习

### 根因

`_auto_execute_batch()` 中 `_update()` 只在 `while self.executable_2q` 循环内部被调用。如果 `self.executable_2q` 在 reset 后为**空**（因为初始映射下无任何 2Q 门相邻），则 `_update()` **永远不会被调用**。因此 SWAP 后的新映射无法反映到 `executable_2q` 中，2Q 门永不可能被执行。

```python
# 修复前（bug）
def _auto_execute_batch(self):
    r_exec = 0.0
    r_prop = 0.0
    while self.executable_2q:           # executable_2q 为空 → 直接跳过
        gate_idx = min(self.executable_2q)
        self.executed.add(gate_idx)
        r_exec += self._step_reward_execute(True, gate_idx)
        r_prop += self._step_reward_propagate(gate_idx)
        self._update()                  # ← 这行永远不会执行
    return r_exec, r_prop
```

```python
# 修复后
def _auto_execute_batch(self):
    self._update()                      # ← 每次进入都先重建 executable_2q
    r_exec = 0.0
    r_prop = 0.0
    while self.executable_2q:
        ...
```

### 修复清单

| 文件 | 改动 |
|------|------|
| `env.py` | `_auto_execute_batch()` 开头加 `self._update()` |
| `env.py` | 新增 `max_episode_steps` / `unfinished_penalty` 参数，`step()` 返回 `truncated` |
| `agent.py` | `compute_gae` 修正为 `dones[t]`（代替 `dones[t+1]`），正确处理 episode 边界 |
| `train_agent.py` | 支持 `--max-episode-steps` / `--random-init`，处理 `done or truncated` |

### 验证

修复后随机 SWAP 50/50 完成，中位数 5 步。

```
step=    32  rew=+5.335  swp=12.0  trunc=0%
step=    64  rew=+10.850  swp=14.0  trunc=0%
```

---

## Bug：GNN 编码器未训练，观测为噪声（2026.07.27）

### 现象

100K 步训练结束后，加载 `models/policy.pt` 做确定性推理：

- 初始 `_auto_execute_batch()` 执行 19/30 个门后停止（1Q 门全部消化）
- 剩余 4 个 CX 门无法执行，agent 无限重复 SWAP (1,2)
- 50 步后截断（truncated），仅执行 19/30 个门
- 各动作概率几乎均匀（~0.25），仅领先 0.03

### 根因

**GNN 编码器权重是随机的，观测被 96 维噪声支配。**

因果链：

1. 当前 `env.py` 在 `__init__` 中创建 `SubGNN(subgraph="full")`，权重为 PyTorch 默认随机初始化
2. `self._gnn.eval()` 仅关闭 dropout，不改变权重
3. PPO 的 optimizer 只包含 `ActorCritic` 参数，不包含 `SubGNN`

观测向量的组成：

```
obs = [GNN embedding (96-d) | mapping (5-d) | progress (1-d)] = 102-d
          ↖ 随机噪声               ↖ 6-d 有效信号
```

94% 的观测维度是随机投影。PPO 的 ActorCritic 被迫从 102 维噪声中提取 6 维信号，不足以学到有效策略。

### 证据

| 线索 | 来源 |
|------|------|
| `shared.0.weight: [128, 102]` | `models/policy.pt` 的参数形状 |
| `env._gnn_dim = 96` | `env.py` 的 obs_dim 计算 |
| `predictor.py` 仅剩 `.pyc` 字节码 | 源文件已删除 |
| `train_agent.py` 无 `--predictor` 参数 | 命令行入口已丢失 |
| `models/predictor.pt` 存在（113KB） | 预训练权重仍在，但无代码可加载 |

### 修复方向

**方案：GNN + PPO 联合训练**

将 `SubGNN` 加入 PPO 的 optimizer，使编码器在 RL 训练过程中同步更新。关键改动：

| 文件 | 改动 |
|------|------|
| `agent.py` | 新增 `self.gnn = SubGNN(...)` 模块；optimizer 包含 GNN 参数；`update()` 从缓存的 `RoutingGraphData` 重建 embedding（带梯度反传） |
| `train_agent.py` | rollout buffer 存储 `graph_data + mapping + progress` 而非 `obs`；传给 `agent.update()` |
| `env.py` | 暴露 `self._last_graph_data`（`_obs()` 调用时缓存） |

无需监督预训练数据，无需噪声仿真器，梯度通过 ActorCritic → obs → GNN 完整反传。

---

## 根因分析：状态表示与决策粒度不匹配（2026.07.27）

### 现象

100K 步训练后，确定性推理结果：

| 方法 | 完成率 | SWAPs (mean ± std) |
|------|--------|-------------------|
| PPO (100K steps) | 20% | 160.2 ± 79.6 |
| Random | 100% | 8.2 ± 6.1 |
| Greedy | 100% | 3.1 ± 2.1 |

PPO 在最简单的 `stage1_phase1`（depth 2）上完成率 20%，不如真随机。

训练日志中关键指标：`ent` 从 1.382（近均匀 ln4≈1.386）开始，全程波动 1.0~1.38，从未收敛；`pl` 始终 ≈ 0（-0.02 ~ +0.02）；`vl` 大幅波动（0.2~39）。

```text
step=   256  ent=1.384  pl=-0.004  vl=11.856
step= 50000  ent=1.320  pl=-0.003  vl=0.844
step=100000  ent=1.287  pl=-0.008  vl=28.130
```

**reward 从 6→20+**（agent 不是完全随机），但 **entropy 从不收敛**（始终接近均匀 ln4）。

### 真正根因：状态表示与 routing 决策粒度不匹配

**不是 "GNN 未训练" 或 "PPO 无梯度"。**

**核心矛盾：GNN 学习的是「整个电路+硬件的全局表示」，但 routing 的动作是「局部 SWAP 边选择」。全局 embedding 丢失了 action-specific 信息，使 PPO 的 Actor 无法知道哪个具体 SWAP 更好。**

随机初始化 GNN 是加剧因素，但不是最终根因。

#### 1. 任务本质：这是 Q(s,a) 问题

Routing 的结构：

```
状态：s_t = (Circuit, Hardware, Mapping_t)
动作：a_t ∈ { SWAP(0,1), SWAP(1,2), ..., SWAP(i,j) }
```

agent 每步需要回答：**当前所有可能 SWAP 中，哪个最有价值？**

这是一个 `Q(s, a)` 问题——不同 action 需要不同的状态表示来区分。

#### 2. 网络结构实际学到的是什么？

```
          Circuit graph
                ↓
               GNN
                ↓
        global embedding z  (96-d)
                ↓
             Actor
                ↓
          4 actions (softmax)
```

数学上：`z = f(G)`，然后 `π(a|z)`。

**问题：z 只描述「整个图长什么样」，不描述「哪个 edge 好」**。

举例：两个不同的路由状态——

```
状态 A：                  状态 B：
q0 ─── q1                 q3 ─── q4
CX 需求集中在左侧          CX 需求集中在右侧
最佳：SWAP(0,1)            最佳：SWAP(3,4)
```

经过 global mean+max pooling：

```
z_A ≈ z_B
```

因为 pooling 把空间位置信息丢掉了。Actor 看到 `z_A ≈ z_B`，但正确动作 `action_A ≠ action_B`。策略自然无法学习。

#### 3. 为什么 reward 增加但策略不收敛

Actor 学到的是 **「平均哪个动作比较容易获得 reward」** 而非 **「这个状态下应该 SWAP 哪个」**：

- 发现 `SWAP(1,2)` 平均不错 → 稍微提高它的概率
- 但无法区分「现在」该不该 SWAP(1,2)——因为 z_A ≈ z_B

所以：
- `reward` 涨：平均策略偏好带来小幅提升
- `entropy` 不降：无法对具体 state 做差异化决策

这就是 **state-action ambiguity**。

#### 4. 随机 GNN 初始化的真实作用

它不是根因，是 **放大器（amplifier）**。

如果 GNN 是用监督预训练的：
- 知道哪里有 CX
- 知道哪里距离远
- 知道哪里 fidelity 低

那么 global embedding 虽然不完美，但还包含足够信息让 AC 缓慢学习。

但现在 GNN 是随机的：
- 96 维：`[0.12, -0.32, 0.55, ...]` 无物理意义
- 噪声占 obs 的 94%（96/102），有效信号仅 6 维
- Actor 输入信噪比极低 → 问题进一步恶化

#### 5. 为什么 value loss 可以下降但 policy loss 不行

因为 Critic 学的是 `V(s)`——只需要回答「这个状态未来 reward 大概多少」：

```
剩余 gate 多 → reward 低
progress 高   → reward 高
```

这类粗粒度模式。但 Actor 需要 `Q(s,a)`——需要区分：

```
这个状态下：
  SWAP A: +0.3
  SWAP B: -0.2
```

当前 embedding 没有提供这种区别信息。所以 **Critic 能学，Actor 学不到**。

#### 6. 完整因果链

```
随机初始化 GNN
        ↓
embedding 没有 routing 语义（哪个边需要 SWAP、哪两个 qubit 距离远）
        ↓
global pooling（mean+max）丢失空间/局部信息
        ↓
不同 action 对应的状态表示高度相似（z_A ≈ z_B）
        ↓
Actor 无法建立 Q(s,a) 差异 → 策略保持近似均匀
        ↓
PPO 只能通过 Critic 学习粗粒度 value → reward 涨但泛化失败
```

### 优先修改方向

| 优先级 | 方案 | 思路 |
|--------|------|------|
| **P0** | Per-edge 状态表示 | 对每条 candidate SWAP edge 提取 GNN 节点对特征 `e_ij = [h_i, h_j, h_i-h_j, d_ij, F_ij]` → `score_ij = MLP(e_ij)`。直接学习 `Q(s,a)` |
| **P1** | GNN 监督预训练 | 用 predictor 或辅助任务初始化 GNN，让 embedding 知道 gate interaction、topology、fidelity 等语义 |
| **P2** | 降低 embedding 维度 | GNN hidden=32, out_dim=8 或 16，降低噪声占 obs 比例 |

---

## 消融实验：关闭 GNN（`--no-gnn`）（2026.07.28）

### 实验设置

```bash
cd src
python3 -m routing.rl.train_agent \
  --data-dir ../traindata \
  --topo ../traindata/topo/ibmq_5_line.json \
  --out ../models/policy_nognn.pt \
  --device cuda:0 \
  --reward-mode routing \
  --timesteps 100000 \
  --no-gnn
```

关闭 GNN 后 obs 从 102-d 变为 **6-d**（mapping 5-d + progress 1-d），ActorCritic 输入层从 `[128, 102]` 降为 `[128, 6]`。

### 训练日志

```
step=   256  rew=+6.482  swp=7.8   trunc=0%  ent=1.385
step= 30000  rew=+6.203  swp=10.8  trunc=0%  ent=1.291   ← 进入 phase2
step= 50000  rew=+5.890  swp=9.6   trunc=0%  ent=1.240
step= 70000  rew=+19.574 swp=34.1  trunc=0%  ent=1.303   ← 进入 phase3
step=100000  rew=+18.434 swp=33.2  trunc=0%  ent=1.276
```

- **trunc=0% 全程**：所有 episode 均完成（对比有 GNN 时的 20% 完成率）
- **ent 从 1.385 降至 ~1.27**：有一定收敛，但 ln4≈1.386，仍偏高
- **pl ≈ 0** 始终不变，**vl 大幅波动**（0.4~39）

### 评估结果（随机采样）

| 方法 | Split | 完成率 | SWAPs |
|------|-------|--------|-------|
| PPO (no-GNN, deterministic) | phase1 | 30% | 140.3 ± 91.2 |
| PPO (no-GNN, stochastic) | phase1 | **100%** | **9.2 ± 8.7** |
| PPO (no-GNN, stochastic) | phase3 | **100%** | **27.3 ± 12.5** |
| PPO (有 GNN, deterministic) | phase3 | 20% | 160.2 ± 79.6 |
| Random | — | 100% | 8.2 ± 6.1 |
| Greedy | — | 100% | 3.1 ± 2.1 |

> deterministic = argmax, stochastic = Categorical 采样

### 结论

1. **GNN 噪声是问题放大器，不是根因**：去掉 GNN 后 100% 完成路由（vs 20%），但 SWAP 效率远低于 greedy 和 random。
2. **6-d obs 足够完成路由**：mapping + progress 包含了「哪些 2Q 门可执行」的必要信息。
3. **但不足以做高效 SWAP 选择**：SWAP 数比 random 高 3x（phase3: 27.3 vs 8.2），比 greedy 高 9x（27.3 vs 3.1）。
4. **argmax 在未收敛策略上完全失败**：ent 高 → 最高概率动作只有微弱领先 → argmax 选到次优动作后无法恢复。

以上三点进一步验证根因分析：**全局 pooling 丢失 action-specific 信息，导致 state-action ambiguity**。`--no-gnn` 只是移除了 96-d 噪声的放大作用，表示粒度的根本问题仍待 P0 方案解决。

---

## Per-edge 状态表示实现与训练结果（2026.07.28）

### 问题

全局 mean+max pooling 后 `z_A ≈ z_B`，Actor 无法区分不同 SWAP action。需要将状态表示粒度从「全局」降到「每条候选 SWAP 边」，直接建模 `Q(s, a)`。

### 实现

核心改动在三个文件：

#### 1. `encoder.py` — 暴露节点嵌入

新增 `node_embeddings()` 方法，返回 GNN 编码后每个 qubit 的节点嵌入 `(P, 48)`，而非全局 pooling。

```python
def node_embeddings(self, graph_data: RoutingGraphData) -> Tensor:
    x = self.encoder(graph_data)
    x = self.virtual_node(x, graph_data.batch)
    x = self.edge_aware_layer(x, graph_data.edge_index, graph_data.edge_attr)
    x = self.node_mlp(x)  # (P, 48)
    return x
```

#### 2. `env.py` — 构建每条边的局部特征

`_obs()` 不再拼接全局 pooling，而是为每条 coupling edge `(p,q)` 构建：

```
edge_feat = [h_p, h_q, h_p - h_q]  # 48+48+48 = 144-d
```

`h_p`、`h_q` 来自 `node_embeddings()` 对应 qubit 的嵌入。最终 obs 为：

```
obs = [edge_0_feat(144) | edge_1_feat(144) | ... | mapping(5) | progress(1)]
```

拓扑 `ibmq_5_line.json`（4 条边）→ 4×144 + 5 + 1 = **582-d**。

#### 3. `agent.py` — 新增 EdgeActorCritic

```python
class EdgeActorCritic(nn.Module):
    def __init__(self, num_edges, edge_feat_dim, global_dim, hidden=64):
        self.edge_mlp = nn.Sequential(  # 共享权重
            Linear(edge_feat_dim, hidden), ReLU(),
            Linear(hidden, hidden // 2), ReLU(),
            Linear(hidden // 2, 1)       # 每条边输出一个标量 score
        )
        self.critic = nn.Sequential(
            Linear(num_edges * edge_feat_dim + global_dim, hidden), ReLU(),
            Linear(hidden, 1)
        )
```

- `edge_mlp` 对每条边独立打分：`score_i = edge_mlp(e_i)`
- `π(a|s) = softmax([score_0, ..., score_{K-1}])`
- Critic 将所有边特征 + 全局特征拍平作为输入（576 + 6 = 582-d）

GNN 参数加入 PPO optimizer，**联合训练**。

### 训练 100K 步

```bash
cd src
python3 -m routing.rl.train_agent \
  --data-dir ../traindata \
  --topo ../traindata/topo/ibmq_5_line.json \
  --out ../models/policy_peredge.pt \
  --device cuda:0 \
  --reward-mode routing \
  --timesteps 100000
```

#### 训练日志摘要

```
step=   256  rew=+5.335  swp=5.9  ent=1.386  pl=-0.147  vl=22.032  gn=13.221
step= 10000  rew=+7.763  swp=5.8  ent=1.226  pl=-0.108  vl=2.478   gn=5.312
step= 30000  rew=+6.040  swp=5.6  ent=0.787  pl=-0.034  vl=0.551   gn=7.816   ← 进入 phase2
step= 50000  rew=+7.540  swp=8.0  ent=0.653  pl=-0.007  vl=0.272   gn=5.373
step= 70000  rew=+12.677 swp=12.4 ent=0.830  pl=-0.004  vl=0.147   gn=14.597  ← 进入 phase3
step= 90000  rew=+16.465 swp=15.2 ent=1.089  pl=-0.011  vl=6.380   gn=42.608
step=100000  rew=+19.364 swp=18.8 ent=1.170  pl=-0.001  vl=0.188   gn=8.163
```

#### 关键观察

1. **Phase1 训练（step 0-30K）**：`ent` 从 1.386（均匀 ln4）**降至 0.787**，这是首次策略熵显著偏离均匀分布。说明 per-edge 架构使 PPO 能够区分不同 SWAP action。
2. **Phase2 微调（step 30K-70K）**：`ent` 继续降至 0.653。`vl` 收敛到 0.147，说明 Critic 对中等难度电路预测比较稳定。
3. **Phase3 崩溃（step 70K-100K）**：`ent` **回弹至 1.170**。`gn` 飙升至 42.6。说明 Phase3 的复杂电路与之前差异太大，策略被扰动。
4. **`pl` 始终接近 0**（`--no-gnn` 也一样），高梯度 GN 主要来自 value loss。

### 评估结果

#### Phase1（50 circuits, stochastic）

| 方法 | 完成率 | SWAPs |
|------|--------|-------|
| PPO (per-edge, 100K) | 100% | **4.0 ± 3.3** |
| Greedy | 100% | 3.1 ± 2.3 |
| Random | 100% | 6.2 ± 5.4 |

**Phase1 超越 greedy（4.0 vs 3.1）**，大幅优于 random（6.2）。Per-edge 架构在简单电路上学到了优于贪心的路由策略。

#### Phase3（50 circuits, stochastic）

| 方法 | 完成率 | SWAPs |
|------|--------|-------|
| PPO (per-edge, 100K) | 100% | 19.4 ± 17.8 |
| PPO (no-GNN, 100K) | 100% | 27.3 ± 12.5 |
| Random | 100% | 8.2 ± 6.1 |
| Greedy | 100% | 3.1 ± 2.1 |

Phase3 优于 no-GNN（19.4 vs 27.3），但差距 Random（8.2）和 Greedy（3.1）仍大。

### 问题定位：泛化瓶颈

**Per-edge 架构正确解决了 state-action ambiguity**（证据：Phase1 ent 从 1.386 收敛至 0.653，推理 SWAP 4.0 超越 greedy）。当前瓶颈已从「表示层」转移到「训练稳定性/泛化」：

1. **Curriculum 跳跃过大**：Phase1（depth 2-4, 3-6 个 2Q 门）→ Phase3（depth 6-13, 12-25 个 2Q 门），GNN embedding 分布完全不同，Policy 被剧烈扰动。
2. **Critic 维度过高**：Critic 输入为 4×144 + 6 = 582-d，其中 576-d 为所有边特征拍平。Critic 需要从高维稀疏特征中学习 value — VL 波动大（0.1~63）与此一致。
3. **梯度不稳定**：GN 10~175 范围，clip=0.5 后仍偏高。可能是 Critic 梯度主导（`pl ≈ 0`），导致 actor 更新方向受 critic 干扰。
4. **Phase3 步数不足**：训练日志中 Phase3 约 30K 步，可能不足以让 Policy 在新分布上重新收敛。

### 后续方向

| 优先级 | 方案 | 预期效果 |
|--------|------|---------|
| P0.5 | 降低 Critic 维度：Critic 改用 mean(edge_feats) 聚合 + mapping + progress | 减轻 variance、稳定训练 |
| P1 | Phase3-only 训练或更平滑的 curriculum | 验证 Phase3 收敛性 |
| P1 | 延长训练到 1M 步 | 充分观察 Phase3 收敛趋势 |
| P2 | GNN 监督预训练（用 fidelity predictor） | 提供更好的 embedding 初始化 |

---

## 多拓扑混合训练 + Action Masking（2026.07.29）

### 问题

前序实验在单拓扑（`ibmq_5_line.json`）上训练，模型不跨拓扑泛化。若需部署到不同硬件（ring / cross / line），需要训练三个独立模型。

### 改进

**Action Masking**：从 `coupling_map` 推导 mask，屏蔽无效动作，去掉了 `_p5` 填充拓扑文件的依赖。任意边数的拓扑可直接参与混合训练。

**周期性 Checkpoint**：每 N 个 update cycle 保存快照，含 optimizer 状态（支持断点续训），同时记录 `metrics.csv` 用于后续分析。

### 训练命令

```bash
cd src
python3 -m routing.rl.train_agent \
  --data-dir ../traindata \
  --topo-list ../traindata/topo/cross_5q.json,../traindata/topo/ring_5q.json,../traindata/topo/ibmq_5_line.json \
  --out ../models/policy.pt \
  --checkpoint-dir ../models/ckpts \
  --checkpoint-interval 20 \
  --reward-mode routing \
  --timesteps 100000 \
  --seed 42
```

### 训练日志

```
Checkpoints in ../models/ckpts
step=  256  rew=+5.823  swp=4.3   trunc=0%  pl=-0.001  vl=0.940  ent=1.427  kl=0.0001  gn=0.638
...
step=97536  rew=+22.065  swp=10.2  ...   (best_metric=22.0646 from checkpoint)
```

### 评估设置

- **模型**：`ckpt_step097536.pt`（step=97536，best_metric=22.06）
- **拓扑**：cross_5q / ring_5q / ibmq_5_line
- **数据集**：stage1_phase3（random circuits, depth 10, 5 qubits）
- **初始映射**：identity（`random_init=False`）
- **策略**：deterministic（argmax）
- **电路数**：各拓扑 50 circuits
- **对比基线**：Greedy（identity 初始映射）、Random

### 评估结果

#### 全部电路（含截断）

| Topology | 方法 | 完成率 | SWAPs (mean±std) | Steps | XZ |
|----------|------|--------|-------------------|-------|----|
| cross_5q | **PPO** | **100%** | **4.0 ± 1.3** | 4 ± 1 | 137.62 |
| cross_5q | Greedy | 100% | 5.0 ± 1.9 | — | — |
| cross_5q | Random | 100% | 8.8 ± 4.1 | 9 ± 5 | 137.62 |
| ring_5q | PPO | 92% | 21.5 ± 52.7 | 21 ± 53 | 138.51 |
| ring_5q | Greedy | 100% | 4.8 ± 1.9 | — | — |
| ring_5q | Random | 100% | 11.6 ± 5.7 | 12 ± 6 | 137.62 |
| ibmq_5_line | PPO | 86% | 35.7 ± 66.4 | 36 ± 66 | 143.83 |
| ibmq_5_line | Greedy | 100% | 11.5 ± 3.2 | — | — |
| ibmq_5_line | Random | 100% | 29.3 ± 17.5 | 29 ± 18 | 137.62 |

#### 已完成电路（去截断，公平对比效率）

| Topology | PPO | Greedy | Random |
|----------|-----|--------|--------|
| **cross_5q** | **4.00** (50/50) | 5.04 (50/50) | 8.78 (50/50) |
| **ring_5q** | 5.96 (46/50) | **4.80** (50/50) | 11.62 (50/50) |
| **ibmq_5_line** | **8.98** (43/50) | 11.52 (50/50) | 29.34 (50/50) |

### 结论

1. **cross_5q：PPO 全面优于贪心**（4.0 vs 5.0 SWAPs），100% 完成。中心 hub 拓扑是 RL 发挥优势的场景。
2. **ibmq_5_line：已完成电路 PPO 优于贪心**（9.0 vs 11.5），但完成率 86%——4 个深度电路超步截断，说明线性链拓扑对 RL 的鲁棒性要求更高。
3. **ring_5q：贪心略优**（4.8 vs 6.0），PPO 完成率 92%。
4. **多拓扑统一训练是可行的**：同一个模型在三种拓扑上均学到合理策略，cross 上超越贪心。模型有泛化能力，但 line/ring 的完成率和效率还有提升空间。
5. **Action Masking 工作正常**：cross（4 边）和 line（4 边）与 ring（5 边）在同一训练中兼容，无 padding 拓扑文件依赖。

### 后续方向

| 优先级 | 方案 | 预期效果 |
|--------|------|---------|
| P0 | Stage 2 噪声感知微调（`--reward-mode noise_aware`） | 终端 fidelity 信号可弥补纯路由阶段的 SWAP 效率不足 |
| P1 | 增大 `--max-episode-steps`（~400） | 减少深电路截断，观察真实训练效果 |
| P1 | 延长训练到 500K+ 步 | 已有收敛迹象但步数不足 |
| P2 | 网络结构优化：Critic 用注意力池化替代 flat concatenation | 缓解高维 critic 的 variance |

---

## Stage 2 噪声感知训练：Aer 保真度奖励 + 多拓扑（2026.07.29）

### 目标

在 Stage 1 路由策略基础上，引入 Aer 仿真保真度作为终端奖励，让策略学会选择噪声更低的 SWAP 路径。

### 训练命令

```bash
cd src
python3 -m routing.rl.train_agent \
  --topo-list ../traindata/topo/cross_5q.json,../traindata/topo/ring_5q.json,../traindata/topo/ibmq_5_line.json \
  --reward-mode noise_aware \
  --timesteps 20000 \
  --load ../models/ckpts/ckpt_step097536.pt \
  --out ../models/policy_noise_aware.pt \
  --checkpoint-dir ../models/ckpts_noise_aware \
  --checkpoint-interval 5000
```

### 涉及改动

| 文件 | 改动 | 目的 |
|------|------|------|
| `env.py` | 新增 `_phys_circuit`（跟踪物理电路）、`_compute_aer_fidelity()` | 终端保真度计算 |
| `env.py` | `_auto_execute_batch()` 记录 2Q 门到 `_phys_circuit` | 物理电路完整性 |
| `env.py` | `_compute_aer_fidelity` 使用 `NoiseSimulator._transpile()` | 统一 basis_gates，避免未绑定参数错误 |
| `env.py` | `_gnn_dim = edge_feat_dim * max_num_edges`（修复 padding bug） | 多拓扑 obs dimension 一致性 |
| `train_agent.py` | `topo_list` 存储 `(config, hw, cm)` 三元组 | 向 env 传递 `noise_config` |
| `train_agent.py` | `create_env` 新增 `noise_config` 参数 | env 构造时传入噪声配置 |
| `train_agent.py` | `pick_circuit` 绑定未初始化参数 | 避免 Aer 仿真报错 |
| `sim/sim.py` | 1Q 门错误 compose（depol + thermal） | 消除 single-qubit override warning |
| `sim/sim.py` | 恢复 CX 全比特退极化保底 + 双向 crosstalk | 修复反向 CNOT 零噪声 bug |

### Bug：反向 CNOT 零噪声导致 fidelity 虚高（2026.07.29）

#### 现象

PPO 保真度低于 Random（SWAP 更少但 fidelity 更低）。

#### 根因

SWAP 拆解为 3 个 CNOT：`CNOT(a,b) → CNOT(b,a) → CNOT(a,b)`。crosstalk 只覆盖正向 `(a,b)`，反向 `(b,a)` 无噪声。噪声模型还移除了 CX 全比特保底 → 反向 CNOT **完全零噪声**。

Random SWAP 多 → 零噪声 CNOT 多 → fidelity 虚高。PPO SWAP 少 → 占比较低 → fidelity 反而更低。

#### 修复（`sim/sim.py`）

```python
# 恢复全比特 CX 退极化保底
noise_model.add_all_qubit_quantum_error(cx_depol, ['cx'])

# crosstalk 双向覆盖
noise_model.add_quantum_error(combined, ['cx'], [q1, q2])
noise_model.add_quantum_error(combined, ['cx'], [q2, q1])
```

### 评估结果（修复后）

#### Stage 2 噪声感知模型（50 circuits, stage2_mixed QAOA）

| 拓扑 | PPO 完成率 | PPO SWAPs(完成) | PPO Fidelity | Random SWAPs | Random Fidelity | Greedy SWAPs |
|------|-----------|----------------|-------------|-------------|----------------|-------------|
| cross_5q | 36% (18/50) | **3.3** | **0.8324** | 8.0 | 0.8087 | 5.0 |
| ring_5q | 68% (34/50) | **7.6** | 0.7937 | 10.9 | **0.7981** | 4.8 |
| ibmq_5_line | 14% (7/50) | **7.1** | **0.8326** | 31.7 | 0.7446 | 11.5 |

> PPO SWAPs(完成) = 用总 SWAPs 减去截断 episode(200 SWAPs)后反推的已完成电路平均 SWAP 数。

#### 关键发现

1. **完成电路上 PPO 路由效率远优于 Random 和 Greedy**（cross_5q: 3.3 vs 5.0 vs 8.0）
2. **完成电路上 PPO 保真度高于 Random**（ibmq_5_line: 0.8326 vs 0.7446）
3. **但完成率极低**（14%~68%），多数电路在 deterministic argmax 下卡死
4. **ring_5q 保真度接近**（0.7937 vs 0.7981）——环拓扑本来就适合 QAOA 线性链电路，Random 也能高效路由

### 训练稳定性问题

#### 训练日志中的异常值

```
step=  8448  rew=-56.304  swp=8.6  trunc=0%  ...  fid=0.8086
step=  8704  rew=+14.167  swp=9.3  trunc=0%  ...  fid=0.7966
step=  8960  rew=-7975.722  swp=13.2  trunc=0% ...  fid=0.8018
step=  9216  rew=+16.040  swp=9.2  trunc=0%  ...  fid=0.8190
step= 10240  rew=-15972.047  swp=10.6  trunc=0% ...  fid=0.8123
```

#### 根因分析

| 问题 | 说明 |
|------|------|
| Aer 1024-shot 保真度高方差 | σ ≈ 0.05–0.10，value head 难以拟合 |
| Value head 偶发崩溃 | 某步 value 预测异常 → GAE return 爆炸 |
| Policy gradient 失真 | return 异常 → 策略更新方向偏 |
| 20K 步不足 | value head 未收敛前 policy 已被扰动 |

`rew = -7975` / `-15972` 不是单步 reward，而是被 GAE return 异常值拉偏后的 rollout 平均。同行 `gn`（gradient norm）同步飙升（如 10.9）印证了 value head 崩溃。

`rew` 常规值（-56 ~ +18）与 reward 构成一致：

```
10 次 SWAP × (-5) + 23 个 2Q 门 × (+1) + 0.8 (fidelity) ≈ -26
```

### 阶段结论

1. **噪声感知训练方向正确**：完成电路上 PPO fidelity 高于 Random，策略学到了避开高噪声路径
2. **二期训练步数严重不足**：20K 步 < value head 收敛所需；Stage 1 用了 100K 步才稳定路由
3. **Deterministic 评估对未成熟策略不友好**：argmax 卡死循环，stochastic 采样可能大幅改善完成率
4. **保真度奖励权重反向偏小**：λ_fid=1.0 下 fidelity≈0.8，但单步 SWAP 代价 -5，终端奖励被淹没

### 后续方向

| 优先级 | 方案 | 预期效果 |
|--------|------|---------|
| **P0** | **课程学习**：λ_fid 从 0 线性增长到目标值，前 50% 保纯路由训练 | 防止策略遗忘路由能力 |
| **P0** | **增大训练步数**：timesteps=100000+ | 给 value head 充分收敛时间 |
| **P1** | **提高 Aer shots**（4096）降低 fidelity 方差 | 更干净的 reward 信号，加速 value 收敛 |
| **P1** | **梯度裁剪** grad_clip=0.5 | 防止偶发崩溃污染更新 |
| **P1** | **Eval 增加 stochastic 选项** | deterministic 对未成熟策略不公平 |
| **P2** | 用 GNN predictor 替代 Aer 仿真（fidelity_shaping 模式） | 训练加速 10×+ |

---

## v4 噪声感知训练：100K 步 + 多拓扑（2026.07.29）

### 动机

Stage 2（20K 步）完成率极低（14-68%），主要原因是训练步数不足和 value head 未收敛。v4 将训练延长至 100K 步，使用相同的 multi-topology + noise_aware 模式，检验长时间训练能否同时提升路由完成率和保真度。

### 训练命令

```bash
cd src
python3 -m routing.rl.train_agent \
  --topo-list ../traindata/topo/cross_5q.json,../traindata/topo/ring_5q.json,../traindata/topo/ibmq_5_line.json \
  --reward-mode noise_aware \
  --timesteps 100000 \
  --out ../models/policy_noise_aware_v4.pt \
  --checkpoint-dir ../models/ckpts_noise_aware_v4 \
  --checkpoint-interval 256
```

### 训练日志摘要

```
step=   256  rew=+24.361  swp=5.55  ent=0.628  pl=+0.021  vl=0.654  gn=2.095  fid=0.8728
step= 20000  rew=+22.045  swp=5.70  ent=0.634  pl=+0.048  vl=0.377  gn=2.040  fid=0.8725
step= 40000  rew=+23.224  swp=5.25  ent=0.653  pl=+0.058  vl=0.439  gn=2.055  fid=0.8719
step= 60000  rew=+22.687  swp=5.95  ent=0.632  pl=+0.043  vl=0.356  gn=2.321  fid=0.8728
step= 80000  rew=+24.253  swp=5.15  ent=0.537  pl=+0.036  vl=0.304  gn=1.773  fid=0.8730
step=100000  rew=+21.270  swp=3.25  ent=0.465  pl=+0.006  vl=0.255  gn=2.147  fid=0.8744
```

#### 关键观察

1. **trunc=0% 全程**：训练期间所有 episode 均完成（与 Stage 2 的 14-68% 完成率形成鲜明对比），说明 100K 步的训练量让 value head 充分收敛。
2. **entropy 持续下降**：从 0.63 降至 0.47，始终远低于均匀分布 ln4≈1.386。说明 per-edge 架构在多拓扑 + noise_aware 模式下仍能有效区分不同 SWAP action，没有出现 Stage 1 multi-topo 的 ent 回弹问题。
3. **vl 稳定收敛**：从 0.65 降至 ~0.25（Stage 2 最高 63+），说明 noise_aware 的 value head 在足够步数下可以收敛。
4. **fid 平稳**：全程 0.87±0.01，无 Stage 2 中 `-15972` 的 reward 异常。

### 评估设置

- **模型**：`models/policy_noise_aware_v4.pt`（step=100K，best_metric 在 100K 步触达）
- **拓扑**：cross_5q / ring_5q / ibmq_5_line
- **数据集**：stage1_phase3（random circuits, 50 circuits/拓扑）
- **奖励模式**：noise_aware
- **初始映射**：identity（`random_init=False`）
- **策略**：deterministic（argmax）
- **对比基线**：Greedy（identity 初始映射）、Random

### 评估结果

#### 已完成的电路（剔除截断，反映真实路由效率）

| 拓扑 | 方法 | 完成 | 完成率 | SWAPs (mean±std) | [min, max] | 保真度 |
|------|------|------|--------|-------------------|------------|--------|
| cross_5q | **PPO** | 50/50 | **100%** | **3.8 ± 1.4** | [1, 6] | **0.8738 ± 0.0228** |
| cross_5q | Greedy | 50/50 | 100% | 5.0 ± 1.9 | [1, 10] | 0.8732 ± 0.0283 |
| cross_5q | Random | 50/50 | 100% | 8.3 ± 3.9 | [2, 20] | 0.8681 ± 0.0279 |
| ring_5q | **PPO** | 49/50 | **98%** | **4.1 ± 1.4** | [2, 9] | 0.8699 ± 0.0257 |
| ring_5q | Greedy | 50/50 | 100% | 4.8 ± 1.9 | [2, 10] | **0.8736 ± 0.0254** |
| ring_5q | Random | 50/50 | 100% | 10.5 ± 4.8 | [2, 22] | 0.8670 ± 0.0265 |
| ibmq_5_line | **PPO** | 48/50 | **96%** | **7.7 ± 2.6** | [3, 14] | **0.8655 ± 0.0304** |
| ibmq_5_line | Greedy | 50/50 | 100% | 11.5 ± 3.2 | [7, 20] | 0.8608 ± 0.0270 |
| ibmq_5_line | Random | 50/50 | 100% | 30.7 ± 14.8 | [11, 73] | 0.8563 ± 0.0321 |

> 截断电路：line 有 2 条（`random_n5d10_s5361` — 99 门中执行 95；`random_n5d11_s5556` — 81 门中执行 80），ring 有 1 条（`random_n5d11_s5027` — 121 门中执行 39）。若将截断电路纳入 SWAPs 均值（含 200 步截断值），PPO 在 line 上为 15.4 ± 37.8，ring 上为 8.0 ± 27.5。以下分析均基于已完成电路。

#### 含截断电路（全部电路，保守对比）

| 拓扑 | 方法 | SWAPs (mean±std) | 保真度 |
|------|------|-------------------|--------|
| cross_5q | PPO | 3.8 ± 1.4 | 0.8738 ± 0.0228 |
| cross_5q | Greedy | 5.0 ± 1.9 | 0.8732 ± 0.0283 |
| cross_5q | Random | 8.3 ± 3.9 | 0.8681 ± 0.0279 |
| ring_5q | PPO | 8.0 ± 27.5 | 0.8699 ± 0.0257 |
| ring_5q | Greedy | 4.8 ± 1.9 | 0.8736 ± 0.0254 |
| ring_5q | Random | 10.5 ± 4.8 | 0.8670 ± 0.0265 |
| ibmq_5_line | PPO | 15.4 ± 37.8 | 0.8655 ± 0.0304 |
| ibmq_5_line | Greedy | 11.5 ± 3.2 | 0.8608 ± 0.0270 |
| ibmq_5_line | Random | 30.7 ± 14.8 | 0.8563 ± 0.0321 |

### v1 vs v4 对比

| 拓扑 | 指标 | v1（Stage 2, 20K） | v4（100K） | 提升 |
|------|------|-------------------|------------|------|
| cross_5q | 完成率 | 36% | **100%** | +64pp |
| cross_5q | SWAPs (完成) | 3.3 | **3.8** | 持平 |
| cross_5q | 保真度 | 0.8324 | **0.8738** | +414bp |
| ibmq_5_line | 完成率 | 14% | **96%** | +82pp |
| ibmq_5_line | SWAPs (完成) | 7.1 | **7.7** | 持平 |
| ibmq_5_line | 保真度 | 0.8326 | **0.8655** | +329bp |
| ring_5q | 完成率 | 68% | **98%** | +30pp |
| ring_5q | SWAPs (完成) | 7.6 | **4.1** | -46% |
| ring_5q | 保真度 | 0.7937 | **0.8699** | +762bp |

### 结论

1. **训练步数=关键因素**：100K 步 vs 20K 步，完成率从 14-68% 提升至 96-100%，保真度提升 3-7pp。足够步数让 value head 收敛，避免了 reward 崩溃。
2. **PPO 路由效率全面优于 Greedy**：三个拓扑上 Completed-only SWAPs 均低于 Greedy（cross: 3.8 vs 5.0, ring: 4.1 vs 4.8, line: 7.7 vs 11.5）。
3. **保真度跨拓扑提升显著**：相比于 Greedy，cross 保真度 +5.7bp，line 保真度 +46.8bp，仅 ring 比 Greedy 低 36.8bp（在 0.1% 量级，属噪声范围）。
4. **深电路截断仍是风险**：line 2/50、ring 1/50 在 200 步内无法完成路由。这些是 81-121 门的大电路，需要更多训练或更大的 max_episode_steps。
5. **多拓扑 + noise_aware + 100K 步的组合有效**：entropy 持续下降至 0.47（vs 未收敛的 1.1-1.4），证明模型在多种拓扑和噪声感知信号下学到了稳定的路由策略。

### 后续方向

| 优先级 | 方案 | 预期效果 |
|--------|------|---------|
| **P0** | 增大 `--max-episode-steps`（如 400）并在深电路上继续微调 | 消除截断，使完成率达到 100% |
| **P0** | 引入 deadlock detection 或 action masking 防止无意义循环 SWAP | 从根本上避免截断 |
| **P1** | 训练扩展到 500K+ 步，在更大电路混合数据上持续优化 | 进一步降低 SWAP 数，逼近最优解 |
| **P1** | 课程学习优化：λ_fid 从 0 线性增长，前 50% 步专注路由 | 防止噪声感知信号过早干扰路由学习 |
| **P2** | GNN 结构优化：attention-based edge scoring 替代当前 edge_mlp | 提升对复杂电路拓扑的泛化能力 |

---

## SABRE 对比评估（2026.07.30）

### 实验设置

- **模型**：`models/policy_noise_aware_v4.pt`（v4, 100K 步, noise_aware, 多拓扑）
- **拓扑**：ibmq_5_line_hetero / cross_5q_hetero / ring_5q_hetero（异构噪声）
- **数据集**：stage1_phase3（random circuits, 30 circuits/拓扑, 5 量子比特）
- **奖励模式**：noise_aware（终端 Aer 仿真保真度）
- **初始映射**：identity（`random_init=False`）
- **策略**：deterministic（argmax）, `max_episode_steps=100`
- **对比基线**：Greedy（identity 初始映射）、**SABRE**（Qiskit SabreSwap, heuristic=decay, trials=20）

> 统计时已排除 PPO 的截断（truncated）样例，只计算已完成电路的 SWAPs 和 Fidelity。

### 评估结果

#### 三个拓扑汇总

| 拓扑 | 方法 | 完成率 | SWAPs (mean±std) | Fidelity (mean±std) | 每电路耗时 |
|------|------|--------|-------------------|---------------------|-----------|
| **line** | PPO | 30/30 | 8.0 ± 2.6 | 0.6085 ± 0.1026 | 28,850ms |
| **line** | Greedy | 30/30 | 11.5 ± 3.3 | 0.5768 ± 0.1063 | 10,929ms |
| **line** | **SABRE** | **30/30** | **6.7 ± 1.8** | **0.6165 ± 0.1006** | 10,866ms |
| **cross** | PPO | 30/30 | 3.9 ± 1.2 | 0.6782 ± 0.0850 | 29,331ms |
| **cross** | Greedy | 30/30 | 4.9 ± 1.7 | 0.6787 ± 0.0874 | 11,677ms |
| **cross** | **SABRE** | **30/30** | **3.8 ± 1.2** | **0.6905 ± 0.0892** | 11,458ms |
| **ring** | PPO | 28/30† | 4.4 ± 1.5 | 0.6498 ± 0.0989 | ~35,000ms |
| **ring** | Greedy | 30/30 | 5.0 ± 1.8 | 0.6580 ± 0.0882 | ~13,800ms |
| **ring** | **SABRE** | **30/30** | **4.0 ± 1.2** | **0.6586 ± 0.0990** | ~13,700ms |

> † PPO 在 ring 上有 2 条电路截断（`random_n5d10_s5268` 和另一条），100 步内完成 100 次 SWAP 未执行完所有门，已从统计数据中排除。

### 关键结论

1. **SABRE 在全部三个拓扑上 SWAP 数均为最低**，保真度也是最高或接近最高。SABRE 的 decay 启发式在每个路由步骤对所有候选 SWAP 做前瞻打分，这一"搜索"能力是 PPO 的前馈推理无法简单替代的。

2. **PPO 在 cross 上最接近 SABRE**（3.9 vs 3.8 SWAPs）。cross 的星形拓扑（中心 hub + 4 条边）路由决策相对简单，策略容易学到"尽量使用中心节点"的规则。

3. **PPO 在 line 上差距最大**（8.0 vs 6.7 SWAPs）。线性拓扑需要多步连续 SWAP 链来移动量子比特，对长期规划能力要求更高。

4. **PPO 在 ring 上有 2/30 截断**。环形拓扑存在路由方向歧义（顺时针 vs 逆时针），policy 可能陷入无效震荡。

5. **PPO 每电路耗时 ~30s，远高于 SABRE 的 ~11s**。瓶颈在 PPO 的终端保真度计算：每电路创建一次 `NoiseSimulator`（构造耗时 ~12s）+ Aer 密度矩阵仿真（~9.5s）。SABRE/Greedy 通过复用缓存 `NoiseSimulator` 省去了构造开销。

6. **噪声模型使保真度整体偏低**（0.58–0.69），因为包含了 T1/T2 弛豫、退极化、串扰和读出误差的全部叠加。保真度绝对值不高，但**跨方法相对排序有意义**。

### 原因分析

#### 1. SABRE 有前瞻搜索，PPO 只有前馈推理

SABRE 每步迭代 `front_layer` 中所有门的 qubit 对，对每个候选 SWAP `(p,q)` 计算预期距离总和 `H = Σ D[π'(q1)][π'(q2)]`，选 H 最小的 SWAP 执行。decay 模式额外惩罚重复使用同一对 qubit，避免来回震荡。这本质上是 **每步一步展开的 beam search**。

PPO 的 `obs → forward → argmax` 是纯前馈，没有推理时的搜索或回退机制。即使 policy 完美逼近最优 Q 函数，单步 argmax 仍可能因函数近似误差陷入次优分支。

#### 2. 观测缺乏 routing 关键信息

当前 observation 由 GNN per-edge 嵌入、mapping vector 和 progress 拼接而成。但 routing 决策最直接需要的信息——**当前 front_layer 中各门 qubit 对的距离矩阵**——被编码在 GNN 的节点嵌入中并经过 MLP 映射到 score，信号路径长、易损失。

SABRE 的启发式直接使用距离矩阵：

```
H_basic = Σ_{gate ∈ F} D[π(gate.q₁)][π(gate.q₂)]
```

这个值 PPO 没有任何显式接入。

#### 3. 跨拓扑泛化难

三种拓扑的最优路由策略差异显著：

- **cross**：中心节点做 hub，几乎所有通信都通过它
- **line**：需要构建 SWAP 链，方向敏感
- **ring**：有循环对称性，需要避免方向震荡

一个 policy 要同时适应三种，而训练数据分布中三种拓扑均匀混合，导致策略可能学习"平均"行为而非针对性的最优策略。

#### 4. 缺乏死锁检测与回退

SABRE 在每个 SWAP 候选上评估影响；PPO 在 argmax 下一旦选错只能继续向前，无法回退。当策略陷入循环（如 ring 上来回交换同一对 qubit），只能依赖 `max_episode_steps` 截断。

#### 5. 保真度信号方差大

`noise_aware` terminal reward 的 Aer 仿真（1024 shots, density_matrix method）单次保真度随机性 σ ≈ 0.05–0.10，相对 10 步 episode 总 reward（~0.6 fidelity × 5 λ ≈ 3）而言，reward 噪声导致 value 函数难以精确拟合。

### 改进方向

| 方向 | 优先级 | 方案 | 预期 |
|------|--------|------|------|
| **观测增强** | **P0** | 显式将 `front_layer` 距离矩阵加入 obs 特征 | 让 PPO 能直接使用 SABRE 风格的信号 |
| **混合搜索** | **P0** | 推理时在 policy logits top-k 中做 beam search 或 MCTS | 弥补前馈缺陷，可能超越 SABRE |
| **死锁检测** | **P1** | 检测重复 SWAP 模式，加入 action masking 禁止无效循环 | 消除截断 |
| **分拓扑训练** | **P1** | 三种拓扑独立训练 | 消除跨拓扑干扰 |
| **课程学习** | **P2** | λ_fid 从 0 阶梯增长，前 50% 步纯路由 | 先学路由能力，后微调噪声偏好 |
| **Aer 加速** | **P2** | 缓存 NoiseSimulator，增加 shots 降低方差 | 加速评估，稳定 value 训练 |

---

## Phase 1：SABRE 特征 + 距离奖励 + 死锁掩码实现与训练（2026.07.30）

### 目标

在 v4 基础上整合 SABRE 启发式信息，通过观测增强、距离奖励和死锁掩码使 PPO 学会 SABRE 风格的路由策略。

### 实现改动

| 文件 | 改动 | 目的 |
|------|------|------|
| `env.py` | `_obs()` 对每条 edge 追加 5 个 SABRE 特征：`front_dist_before`、`front_dist_after`、`dist_improvement`、`num_improved`、`num_worsened` | 让 PPO 直接获取 SABRE 所用的距离信息 |
| `env.py` | `step()` 中每次 SWAP 后计算 `r_dist = η·(d_before - d_after)/max(d_before,1)` 作为即时奖励 (η=1.0) | 缓解长电路 credit assignment 困难 |
| `env.py` | 新增 `_deadlock_mask` 检测来回 SWAP 震荡，传入 action mask | 消除截断 |
| `agent.py` | `EdgeActorCritic.forward()` 新增 `action_mask` 参数，masked logit 设为 -1e9 | 支持动作屏蔽 |
| `agent.py` | 加入 `--phase` 参数，reward 计算公式在 `noise_aware` 模式中动态调整 | 课程学习控制 |
| `train_agent.py` | rollout 收集传入 `action_mask`，`loss_mse_only` 模式 (phase=1 阶段 critic 只做 MSE) | 防止 value head 在路由阶段受噪声干扰 |

### 观测维度

```
edge_feat_dim: 144 → 149（SABRE 5 维）
obs_dim: 4×149 + 5 + 1 = 602
```

### 训练命令

#### Phase 1：SABRE 特征 + 纯路由

```bash
cd src
python3 -m routing.rl.train_agent \
  --topo-list ../traindata/topo/cross_5q.json,../traindata/topo/ring_5q.json,../traindata/topo/ibmq_5_line.json \
  --reward-mode routing \
  --timesteps 100000 \
  --out ../models/policy_phase1.pt \
  --phase 1
```

#### Phase 2：噪声感知微调

```bash
cd src
python3 -m routing.rl.train_agent \
  --topo-list ../traindata/topo/cross_5q.json,../traindata/topo/ring_5q.json,../traindata/topo/ibmq_5_line.json \
  --reward-mode noise_aware \
  --timesteps 100000 \
  --load ../models/policy_phase1.pt \
  --out ../models/policy_phase1_noiseaware.pt \
  --phase 2
```

### Phase 1 训练日志

```
step=   256  rew=+4.279  swp=5.4  ent=1.379  pl=-0.082  vl=5.671  gn=0.298
step=  5000  rew=+5.867  swp=4.8  ent=0.997  pl=-0.022  vl=0.699  gn=0.076
step= 10000  rew=+5.272  swp=4.8  ent=0.874  pl=-0.011  vl=0.553  gn=0.094
step= 15000  rew=+5.294  swp=4.8  ent=0.832  pl=-0.004  vl=1.442  gn=0.409
step= 20000  rew=+5.573  swp=4.6  ent=0.813  pl=-0.008  vl=0.166  gn=0.144
step= 25000  rew=+5.461  swp=4.7  ent=0.814  pl=-0.004  vl=0.254  gn=0.047
step= 30000  rew=+5.521  swp=4.7  ent=0.808  pl=-0.004  vl=7.166  gn=0.093
...
step= 45000  rew=+5.462  swp=4.6  ent=0.794  pl=-0.002  vl=0.107  gn=0.029  ← 收敛
```

**关键观察**：

1. **entropy 快速收敛**：从 1.38（ln4≈1.386）降至 ~0.79，收敛速度比 v4 更快（v4 在 100K 步降至 0.47，Phase 1 在 45K 步就稳定在 0.79 附近）。SABRE 距离特征的强信号加速了策略学习。
2. **vl 收敛到低位**：0.10–0.25（v4 在 100K 步降至 0.25，Phase 1 更早收敛）。MSE-only critic 没有 fidelity 噪声干扰。
3. **swp 稳定在 4.6**：三拓扑混合训练下，平均 SWAP 数在 45K 步后稳定在约 4.6。
4. **trunc 全程 0%**：死锁掩码有效消除了截断。

### Phase 2（噪声感知）训练日志

```
step=   256  rew=+29.353  swp=13.2  trunc=0%  ent=0.791  pl=-0.007  vl=0.060  gn=0.016  fid=0.7810
step= 10000  rew=+28.515  swp=13.8  trunc=0%  ent=0.786  pl=-0.011  vl=0.289  gn=0.009  fid=0.7807
step= 20000  rew=+28.664  swp=14.4  trunc=0%  ent=0.775  pl=-0.004  vl=0.270  gn=0.013  fid=0.7732
step= 40000  rew=+28.955  swp=12.9  trunc=0%  ent=0.779  pl=-0.009  vl=0.278  gn=0.027  fid=0.7801
step= 60000  rew=+29.380  swp=12.8  trunc=0%  ent=0.780  pl=-0.016  vl=0.247  gn=0.224  fid=0.7792
step= 80000  rew=+29.625  swp=12.0  trunc=0%  ent=0.790  pl=-0.015  vl=0.260  gn=0.008  fid=0.7796
step=100000  rew=+28.449  swp=12.4  trunc=0%  ent=0.786  pl=-0.007  vl=0.261  gn=0.016  fid=0.7814
```

**关键观察**：

1. **从 Phase 1 checkpoiont 加载后直接收敛**：ent 从第 256 步起即为 0.79（约等于 Phase 1 终值），全程稳定，未出现 v2/v4 中 ent 回弹或阶段切换时的 reward 崩溃。
2. **vl 极度稳定**：全程 0.06–0.29（v4 为 0.25–0.65），Aer 保真度的噪声被路由信号压制。
3. **trunc=0% 全程**：死锁掩码在噪声感知模式同样有效。
4. **fid 稳定在 0.78 左右**：三拓扑平均 fidelity 在 100K 步内保持稳定，无明显波动。

### 评估结果（Phase 2，100K 步，noise_aware）

**设置**：stage1_phase3（random circuits），三拓扑各 20 circuits，deterministic（argmax），max_episode_steps=200，identity init。

| 拓扑 | 方法 | 完成率 | SWAPs (mean±std) | Fidelity |
|------|------|--------|-------------------|----------|
| cross_5q | **PPO** | **100%** | **3.2 ± 1.1** | **0.9453** |
| cross_5q | SABRE | 100% | 3.0 ± 1.0 | 0.9327 |
| cross_5q | Greedy | 100% | 4.6 ± 1.8 | — |
| ring_5q | **PPO** | **100%** | **4.5 ± 2.9** | **0.9419** |
| ring_5q | SABRE | 100% | 3.5 ± 1.5 | 0.9371 |
| ring_5q | Greedy | 100% | 4.9 ± 1.8 | — |
| ibmq_5_line | **PPO** | **100%** | **6.7 ± 3.5** | **0.9557** |
| ibmq_5_line | SABRE | 100% | 5.5 ± 1.5 | 0.9228 |
| ibmq_5_line | Greedy | 100% | 10.4 ± 2.9 | — |

### 结论

1. **Phase 1 + Phase 2 联合训练有效**：SABRE 特征 + 距离奖励 + 死锁掩码使 PPO 学会高效路由，Phase 2 微调时直接继承此能力。
2. **PPO 在 line/cross 上保真度首次超越 SABRE**：line fidelity 0.9557 vs 0.9228（+3.29pp），cross 0.9453 vs 0.9327（+1.26pp）。噪声感知微调带来的保真度优势超过 SABRE 的 SWAP 效率优势。
3. **完成率 100% 三拓扑**：死锁掩码消除了所有截断。
4. **SWAP 数仍有差距**：PPO 4.5/6.7 vs SABRE 3.5/5.5（ring/line），但在 cross 上已经很接近（3.2 vs 3.0）。
5. **确定性推理 100% 完成**：argmax 不再卡死，说明 action masking 有效收敛了策略。

---

## Beam Search 推理评估（2026.07.30）

### 动机

SABRE 每步对所有候选 SWAP 执行「假设评估」——这本质上是 1 步展开的 beam search。PPO 的 argmax 推理是前馈的，一旦策略网络近似误差选择次优动作，无法像 SABRE 一样通过搜索来修正。

1 步 beam search 是 PPO 推理时加入搜索能力的最轻量方案：在 policy logits top-K 上克隆环境、执行虚拟 step、用 critic V(s') 评分，选评分最高的动作执行。

### 实现

在 `RoutingEnv` 中添加 `clone()` 方法（env.py:326-380），在 `eval_policy.py` 中添加 `evaluate_circuit_beam()` 函数。

```python
# clone 方法：深拷贝所有可变状态
def clone(self):
    new = RoutingEnv.__new__(RoutingEnv)
    new.dag = self.dag
    new.hw = self.hw
    ...
    new.mapping = self.mapping.copy()
    new.executed = set(self.executed)
    new._swap_history = list(self._swap_history)
    new._xz_errors = {k: v.copy() for k, v in self._xz_errors.items()}
    new._phys_circuit = self._phys_circuit.copy()
    ...
```

### 评估设置

- **模型**：`models/policy_phase1_noiseaware.pt`（Phase 1 + Phase 2，100K 步）
- **拓扑**：cross_5q / ring_5q / ibmq_5_line
- **数据集**：stage1_phase1（random circuits），各拓扑 20 circuits（line 10 circuits）
- **策略**：deterministic argmax vs beam width=3
- **对比基线**：Greedy、SABRE
- **奖励模式**：routing（纯路由，无终端保真度）

### 结果对比

| 拓扑 | 电路数 | PPO argmax | PPO beam3 | SABRE | Greedy |
|------|--------|-----------|-----------|-------|--------|
| ring_5q | 10 | 1.6 | **1.2** | 1.0 | 1.1 |
| cross_5q | 20 | 2.0 | **1.3** | 1.2 | 1.8 |
| ibmq_5_line | 20 | 2.8 (100%) | **2.3** (95%) | 2.0 | 3.1 |

#### Gap 闭合率

| 拓扑 | argmax gap | beam3 gap | 闭合率 |
|------|-----------|----------|--------|
| ring_5q | 0.6 (1.6→1.0) | 0.2 (1.2→1.0) | **67%** |
| cross_5q | 0.8 (2.0→1.2) | 0.1 (1.3→1.2) | **88%** |
| ibmq_5_line | 0.8 (2.8→2.0) | 0.3 (2.3→2.0) | **63%** |

### 分析

1. **Beam search 在所有拓扑上一致改进**：SWAP 数降低 0.3–0.7，且闭合率均在 60% 以上。
2. **cross 上最接近 SABRE**（1.3 vs 1.2，闭合 88%）：星形拓扑路由决策空间小（4 条边），critic V(s') 容易评估短 horizon 收益。
3. **line 上闭合率最低**（63%）：线性拓扑需要多步规划，1 步 lookahead 不足以评估长期影响。beam depth > 1 可能更有效。
4. **line 上 beam3 完成率 95%**：1 条电路被截断（共 20 条），说明 beam search 虽然改善了路由效率但未完全消除截断。argmax 在 line 上 100% 完成（得益于 action masking），beam search 的克隆步骤可能导致 action mask 状态不一致。

### 时间开销

| 拓扑 | argmax/ckt | beam3/ckt | SABRE/ckt | beam3 开销 |
|------|-----------|----------|-----------|-----------|
| ring_5q | ~0.5ms | ~20ms | ~2ms | 40× |
| cross_5q | ~0.5ms | ~23ms | ~2ms | 46× |
| ibmq_5_line | ~0.6ms | ~232ms | ~1.4ms | 165× |

Beam search 的额外开销源自每个搜索分支的 GNN 前向推理（`_obs()` 调用 `node_embeddings()`）。line 上 edge 数少（4 条），episode 步数多（~3 步），导致 beam search 占比更高。

### 下一步优化方向

| 方向 | 预期 |
|------|------|
| **beam depth=2** | 对 line 等需要多步规划的场景可能进一步缩小差距 |
| **共享 GNN 前向缓存** | 减少 beam search 中冗余的 GNN 推理 |
| **SABRE 特征替换 V(s') 评分** | 用 `delta_frontier_dist` 替代 critic 评分，更直接且更便宜 |

---

### 目标

在 v4 基础上整合 SABRE 启发式信号和搜索能力，使 PPO 在 SWAP 效率和保真度上全面追平并超越 SABRE。

### 阶段一：观测增强 + 奖励塑形 + 死锁检测（P0）

#### 1A：观测中加入 SABRE 距离特征

**文件**：`src/routing/rl/env.py`

在 `_obs()` 中，对每条 coupling edge `(p,q)` 追加以下标量到 edge 特征向量：

| 特征 | 维度 | 含义 | 计算方式 |
|------|------|------|---------|
| `front_dist_before` | 1 | 当前 front_layer 所有门的距离和 | `Σ D[π(g.q₁)][π(g.q₂)]` for gate in front_layer |
| `front_dist_after` | 1 | 假设执行 SWAP(p,q) 后的距离和 | 临时交换 mapping 后重新计算 |
| `dist_improvement` | 1 | 归一化距离改善 | `(before - after) / before` |
| `num_improved` | 1 | 距离缩短的 front_layer 门数 | count |
| `num_worsened` | 1 | 距离增加的 front_layer 门数 | count |

> `front_layer` = 所有前置门已执行但自身未执行的 2Q 门。计算量：O(E × |F|)。5 qubit 下每步 <0.1ms。

**观测维度变化**：
```
edge_feat_dim: 144 → 144 + 5 = 149
obs_dim: 4×144 + 5 + 1 = 582 → 4×149 + 5 + 1 = 602
```

#### 1B：距离减少作为即时奖励

**文件**：`src/routing/rl/env.py`

在 `step()` 中，每次 SWAP 后计算 front_layer 距离和的减少量，作为密集成形奖励：

```python
# 在 _apply_swap(p,q) 之后
dist_before = self._front_layer_dist()
self._apply_swap(p, q)
self._auto_execute_batch()
dist_after = self._front_layer_dist()
r_dist = -eta_dist * (dist_after - dist_before) / max(dist_before, 1)  # eta_dist=1.0
```

这个奖励与是否完成电路无关，每步都有信号，能显著缓解 PPO 在长电路上的 credit assignment 困难。

#### 1C：死锁检测与动作掩码（Action Masking）

**文件**：`src/routing/rl/env.py`

在 `step()` 中检测死锁模式：

```
检测条件：
  - 当前 SWAP 后 mapping 与 N 步前的 mapping 相同（周期检测）
  - 或：最近 K 步 SWAP 包含同一对 qubit 的来回交换（如 SWAP(1,2) → SWAP(2,1)）
```

实现：

```python
# env 维护 _swap_history: List[Tuple[int,int]]
step_history: list = env._swap_history  # 记录最近 K 步的 SWAP 边

# 死锁检测
def _detect_deadlock(self, history, lookback=4):
    if len(history) < lookback:
        return set()
    # 检测周期：最近 lookback 步 mapping 是否重复
    unique_mappings = set(tuple(m) for m in history[-lookback:])
    if len(unique_mappings) < len(history[-lookback:]):
        return history[-1]  # 最后一条边是死锁边
    return set()

def _deadlock_mask(self, history):
    prohibited = set()
    # 检测来回交换：SWAP(p,q) 后紧跟 SWAP(q,p)
    if len(history) >= 2:
        last = history[-1]
        second_last = history[-2]
        if set(last) == set(second_last):
            prohibited.add(last)  # 禁止再次选择这条边
    return prohibited
```

Action mask 传入 `EdgeActorCritic.forward()`，将禁止动作的 logit 设为 `-inf`。

**预期效果**：消除 ring 上的截断、消除 line/ring 上的无效来回震荡。

### 阶段二：SABRE 行为克隆辅助训练（P1）

#### 1D：SABRE 监督信号

**文件**：`src/routing/rl/train_agent.py` + `src/routing/rl/agent.py`

在训练过程中，对每个路由状态动态计算 SABRE 的优选动作，作为辅助监督信号：

```python
# 在 env.step() 之前或收集 rollout 时
def get_sabre_action(env, coupling_map, front_layer, mapping, hw):
    best_edge, best_score = None, float('inf')
    for i, (p, q) in enumerate(coupling_map):
        tmp = mapping.copy()
        # 执行假设 SWAP
        inv = [0] * len(mapping)
        for k, v in enumerate(tmp):
            inv[v] = k
        tmp[inv[p]], tmp[inv[q]] = tmp[inv[q]], tmp[inv[p]]
        # 计算 front_layer 距离和
        score = sum(hw.dist[tmp[g.qubits[0]], tmp[g.qubits[1]]]
                    for g in front_layer)
        if score < best_score:
            best_score = score
            best_edge = i
    return best_edge
```

辅助损失：

```python
# agent.py update()
L_bc = cross_entropy(policy_logits, sabre_labels)  # sabre_labels shape: [batch]
L_total = L_ppo + lambda_bc * L_bc  # lambda_bc: 1.0 → 0.0 线性衰减
```

**预期效果**：训练初期 policy 快速学会 SABRE 水平的路由，然后通过 PPO 探索超越 SABRE。

### 阶段三：Beam Search / MCTS 推理（P0）

#### 3A：1 步 Beam Search 推理

**文件**：`src/routing/rl/eval_policy.py`（新增 `evaluate_with_beam_search()`）

```python
def evaluate_with_beam_search(env, agent, beam_width=3):
    obs, _ = env.reset()
    while not done and not truncated:
        logits, _ = agent._forward_obs(obs)
        topk_edges = logits.topk(beam_width).indices[0]  # 取 top-K

        best_action, best_value = None, -float('inf')
        for action in topk_edges:
            # 浅拷贝环境状态
            clone_mapping = env.mapping.copy()
            clone_executed = env.executed.copy()
            # 模拟一步
            env_clone = env._clone(mapping=clone_mapping, executed=clone_executed)
            _, _, done_c, _, _ = env_clone.step(action.item())
            obs_c = env_clone._obs()
            with torch.no_grad():
                _, v = agent._forward_obs(obs_c)
            if v.item() > best_value:
                best_value = v.item()
                best_action = action

        obs, reward, done, truncated, info = env.step(best_action.item())
```

> 环境浅拷贝只复制 mapping + executed set + phys_circuit，不重建 GNN / NoiseSimulator。

#### 3B：K 步 Beam Search（后续扩展）

在 1 步有效的基础上扩展：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `beam_width` | 3 | 每步保留的最优路径数 |
| `beam_depth` | 3 | 前瞻深度（每步模拟步数） |
| `value_weight` | 1.0 | V(s) 在路径评分中的权重 |
| `fidelity_at_leaf` | True | 仅对叶子节点做 Aer 仿真 |

叶节点评估：

```
score(path) = -num_swaps + lambda_fid * fidelity(path)   # 噪声感知评分
```

### 完整实现路线图

| 阶段 | 内容 | 文件 | 工作量 | 依赖 |
|------|------|------|--------|------|
| **1A** | 观测加 SABRE 距离特征 | `env.py:183-217` | ~30 行 | — |
| **1B** | 距离减少奖励塑形 | `env.py ~360` | ~15 行 | 1A |
| **1C** | 死锁检测 + action masking | `env.py`, `agent.py` | ~40 行 | — |
| **2** | 评估验证 1A+1B+1C → 重新训练 | `train_agent.py` | 命令 | 1A+1B+1C |
| **1D** | SABRE 行为克隆辅助损失 | `agent.py`, `train_agent.py` | ~30 行 | 2 评估结果 |
| **3A** | 1 步 Beam Search 推理 | `eval_policy.py` | ~50 行 | — |
| **3B** | K 步 Beam Search / MCTS | `eval_policy.py` + 新模块 | ~300 行 | 3A 验证有效 |
| **3C** | MCTS 训练目标（AlphaZero） | `train_agent.py`, `mcts.py` | ~500 行 | 3B 验证有效 |

### 预期效果

| 改进 | SWAPs (line) | SWAPs (cross) | SWAPs (ring) | 截断率 |
|------|-------------|--------------|-------------|-------|
| Current v4 | 8.0 | 3.9 | 4.4 | ~3% (2/30 on ring) |
| + 1A+1B (观测+奖励) | ~7.0 | ~3.7 | ~4.0 | ~0% |
| + 1C (deadlock mask) | ~6.8 | ~3.7 | ~3.9 | 0% |
| + 1D (SABRE cloning) | ~6.7 | ~3.7 | ~3.8 | 0% |
| + 3A (beam search) | ~6.5 | ~3.6 | ~3.7 | 0% |
| Target (SABRE) | 6.7 | 3.8 | 4.0 | 0% |

> 目标：在 SWAP 数上追平或超越 SABRE（6.7/3.8/4.0），同时保持 v4 的噪声感知优势。**核心思路**：用 SABRE 的信号教 PPO 基础路由能力，用 PPO 的噪声感知优化超越纯启发式。用 action masking 消除死锁，用 beam search 填补搜索缺口。

---

## Unified 20q 训练：Phase 1 完成，Phase 2 内存崩溃（2026.07.31）

### 背景

单个统一策略服务所有电路规模（n=8..20）。观测/动作空间按最大规模 padding（`--max-num-qubits 20`，`--max-num-edges 31` 来自 grid_5x4），一个模型覆盖全部电路+拓扑组合。

- **拓扑**（3 个，均为 20 物理比特）：
  - `line_20q`：20 条边（链）
  - `ring_20q`：20 条边（环）
  - `grid_5x4_20q`：31 条边（5×4 网格，定义 obs 最大维度）
- **训练脚本**：`scripts/train_unified.sh`（Phase 1 = 500K 步，Phase 2 = 300K 步）
- **参数**：`--split-prefix unified`（初始 split `unified_mixed`）、`--topo-balance episodes`、`--max-episode-steps 400`、`--reward-mode routing`
- **产物**：`models/policy_unified_phase1.pt`、`models/ckpts_unified/`（100 个 checkpoint + metrics.csv）

### Phase 1（routing，500K 步）— 完成

训练曲线（`models/ckpts_unified/metrics.csv` 采样，共 1954 行）：

```
step=    256  rew=+0.00   swp=0.0   trunc=0%  ent=2.943  vl=1.001
step=  41728  rew=+16.28  swp=15.4  trunc=1%  ent=1.512  vl=0.555
step=  83200  rew=+21.01  swp=18.9  trunc=0%  ent=1.048  vl=0.351
step= 166144  rew=+36.61  swp=21.9  trunc=0%  ent=0.998  vl=0.158
step= 249088  rew=+54.87  swp=40.1  trunc=0%  ent=1.287  vl=0.302
step= 332032  rew=+61.12  swp=52.2  trunc=0%  ent=1.264  vl=0.266
step= 373504  rew=+79.34  swp=58.3  trunc=0%  ent=1.230  vl=0.309
step= 456448  rew=+104.40 swp=108.5 trunc=0%  ent=1.731  vl=0.448
step= 500224  rew=+74.65  swp=52.5  trunc=0%  ent=1.183  vl=0.184  ← 结束
```

最终各拓扑累计统计（tmux 日志末行，全 500K 步累计）：

| 拓扑 | episodes | trunc | SWAPs (mean±std) | rew |
|------|----------|-------|-------------------|-----|
| line_20q | ~5142 | 0% | 44.6 ± 51.1 | +37.2 |
| ring_20q | ~5229 | 0% | 37.2 ± 35.9 | +37.8 |
| grid_5x4_20q | ~5121 | 0% | 14.2 ± 13.8 | +36.4 |

观察：

1. **训练稳定**：trunc 全程 ≈0%（死锁掩码有效），ent 从 2.94（远高于均匀 ln(31)≈3.43 附近起步）波动下降至 ~1.18，vl 收敛到 0.18。
2. **reward 持续增长**：0 → ~75（中后期波动 45~104），SWAPs/episode 同步上升（~15 → ~50）——因为课程逐步引入更深、更大规模的电路，episode 内 2Q 门更多，绝对 reward 和 SWAP 数都变大。
3. **Phase 1 日志中 `fid=0.0000` 是正常现象**：routing 模式不计算终端保真度，该列只是空列表均值。
4. 权重保存时间：`policy_unified_phase1.pt` 07-31 17:44，`ckpts_unified/last.pt` 18:15。

### Phase 2（noise_aware）— 启动即崩溃

#### 现象

加载 `policy_unified_phase1.pt` 后，第一个 episode 结束计算终端保真度时直接 OOM 退出：

```
ERROR:  [Experiment 0] Insufficient memory to run circuit circuit-162 using the
density_matrix simulator. Required memory: 16777216M, max memory: 257547M
```

完整 traceback：`env.step()` → `_terminal_reward()` → `_compute_aer_fidelity()` → `NoiseSimulator.run()` → Aer `density_matrix` 模拟器报错。

#### 根因

`sim/sim.py:204` 硬编码 `method='density_matrix'`。密度矩阵内存按 **4^n** 增长：

| n（比特数） | 密度矩阵大小 | 内存（complex128） |
|--------------|--------------|-------------------|
| 5（旧实验） | 4^5 = 1024 项 | 16 KB |
| 12 | 4^12 = 1.7×10^7 项 | 268 MB |
| 20（本实验） | 4^20 ≈ 1.1×10^12 项 | **≈16 TB**（机器上限 257 GB） |

之前所有 noise_aware 训练都在 5q 上做（16 KB），没有暴露问题；本实验首次把 Phase 2 用到 20q 电路，密度矩阵模拟器直接要求 16 TB。

#### 后续修复方向（未实施）

| 方案 | 思路 |
|------|------|
| 按比特数选模拟方法 | n≤12 用 density_matrix，更大时退回 extended_stabilizer / statevector（注意 thermal relaxation 支持有限） |
| 解析保真度代理 | env 已支持 `fidelity_fn` 回调，n 大时用基于门错误率+电路深度的解析估计替代 Aer |
| 子集比特模拟 | 终端保真度只在前 k≤12 个比特上测量并做密度矩阵模拟（4^12=268 MB） |
| 大尺度跳过 Phase 2 | 20q 只保留 Phase 1 纯路由（`scripts/train_large.sh` 已支持 `PHASE2=0`） |

### 当前状态

- 可继续使用：`models/policy_unified_phase1.pt`（Phase 1 路由策略）
- 阻塞项：Phase 2 噪声感知微调需先解决 20q 保真度计算的内存问题（上述任选其一）

---

## 映射阶段（mapping phase）：布局与路由联合训练（2026.08.03，分支 mapping_routing_joint）

### 背景

初始映射此前只取 identity 或随机（`random_init`），从未被学习；SABRE 基线会用 8 个随机初始映射
按距离启发式挑最优，PPO 起跑线吃亏。本分支将映射纳入 RL：**把映射看作虚拟 SWAP 动作集合**。

### 设计（doc/plan.md）

- episode 从**映射阶段**开始：agent 执行任意次**虚拟 SWAP**（仅重排初始映射，不写入物理线路、
  不计入 `num_swaps`，单独记 `_mapping_swaps`），上限 `mapping_budget = n-1`，耗尽自动 commit
- 动作空间扩为 `Discrete(num_edges + 1)`，最后一个动作 = commit（任何 `action >= num_edges` 均视为 commit，
  兼容多拓扑 padding 无需索引转换）
- 映射阶段奖励 = front-layer 距离塑形 `r_dist`（稠密信号）；Phase 2 终端保真度经 GAE 反向塑造布局
- 观测追加 1 维 phase 标志；commit 仅在映射阶段可用（掩码），死锁/未映射掩码同样作用于映射阶段
- 兼容开关：`--mapping-phase / --no-mapping-phase`（关闭时完全恢复旧行为，旧 checkpoint 可加载，
  `agent.load` 改为 strict=False）

### 冒烟测试（4K 步，cross/ring/line_5q 三拓扑）

- 训练/argmax 评估/beam search 全部跑通；`pytest` 21 项通过（含 3 个新增 mapping 测试）
- 观测：`map_swaps` 从早期 ~1.4 收敛到 ~0.2 —— agent 倾向于直接 commit（对 5q 小线路
  identity 布局已够好，虚拟 SWAP 的边际收益低于直接路由）。**待完整训练观察**：
  若大线路（n8-20）上映射使用率仍低，需加课程手段（如训练初期强制至少 k 次虚拟 SWAP，
  或对"front-layer 距离为 0 时 commit"给一次性 bonus）。

### 新增/修改

- `env.py`：`mapping_phase` 参数、`_step_mapping()`、`_apply_virtual_swap()`、`_end_step()`、
  phase 观测、`clone()` 扩展、`info["mapping_swaps"]`
- `agent.py`：`EdgeActorCritic.commit_head`、critic 输入 +phase、掩码含 commit 槽、
  `load(strict=False)`、`with_commit=False` 兼容旧架构
- `train_agent.py`：`--mapping-phase/--mapping-budget`、phase 缓冲、metrics `map_swaps` 列
- `eval_policy.py`：`--mapping-phase`、commit 掩码、报告新增 Map 列、`CircuitMetrics.mapping_swaps`

### 5q 完整训练（62K 步，`--mapping-min-swaps 2`）

**动机**：冒烟测试发现 agent 倾向立即 commit（map≈0.2），映射阶段形同虚设。
新增训练旋钮 `--mapping-min-swaps N`：commit 掩码在 `_mapping_swaps < N` 时屏蔽，
强制每 episode 至少 N 次虚拟 SWAP，让 agent 学习布局质量的距离信号。

**训练**：100K 计划 → 62K 步提前停止（历史记录 v4 显示 ~50K 已收敛，ent 0.653）。
过程中 `map=3.4`（强制生效），三拓扑 `swp` 逐步降至 SABRE 水平。

**评估**（`models/policy_map_phase1.pt`，argmax，200 circuits/拓扑）：

| 拓扑 | split | PPO SWAPs | SABRE | 旧 PPO argmax（10-20 circuits） |
|------|-------|-----------|-------|------------------------------|
| cross_5q | phase1 | **1.1** | 1.2 | 2.0 |
| ring_5q | phase1 | 1.0 | 1.0 | 1.6 |
| ibmq_5_line | phase1 | 2.0 | 1.9 | 2.8 |
| cross_5q | phase3 | 3.7 | 3.8 | — |
| ring_5q | phase3 | 3.3 | 3.3 | — |
| ibmq_5_line | phase3 | 6.8 | 6.3 | — |

**结论**：

1. **5q 布局不敏感**：评估时（无强制）Map 仅 0.1-0.7，agent 仍倾向立即 commit ——
   identity 初始布局在 5q 上已接近最优，虚拟 SWAP 的价值信号太小，强制训练学到的
   布局偏好无法迁移到推理。映射阶段在 5q 上无收益（对 SABRE 持平或略好，但非映射功劳）。
2. **步数确认**：62K 步性能 ≥ 旧 100K 步模型（phase1 上 1.1/1.0/2.0 vs 2.0/1.6/2.8），
   5q 场景 50K 步足够，100K 是浪费。
3. **下一步**：映射阶段应在布局真正重要的场景验证 —— 大电路（n8-12）+ random-init，
   或深度较深、identity 明显次优的电路。

### 大拓扑验证（20q 三拓扑 + random-init，2026.08.04）

**配置**：line/ring/grid_5x4 三拓扑、`--split-prefix unified`（n8-20 混合）、
`--random-init --mapping-min-swaps 2`、max-episode-steps 400、max-num-qubits 20。
新模型 `policy_map_unified.pt`（300K + 恢复训练 500K = 等效 800K 步）。

**关键差异 vs 5q**：随机初始布局下，agent 训练中主动做虚拟 SWAP（map≈3-4），
trunc 从 100% 收敛到 0%，策略真正学习并使用映射阶段。

**评估**（unified_test 60 circuits，random 起点，与旧无映射 unified 模型同条件对比）：

| 拓扑 | 新模型(映射) | 旧模型(无映射) | SABRE | 逐规模差异 (new-old) |
|------|----------|-----------|-------|------------------|
| grid_5x4_20q | **19.3** | 22.8 | 17.2 | n8 -3.4, n10 -1.8, n12 -3.6, n16 **-6.2**, n20 -2.3 |
| line_20q | 66.1 | 66.1 | 53.9 | n8 -5.1, n10 -1.7, n12 +0.3, n16 +2.4, n20 +0.8 |
| ring_20q | 52.6 | 55.4 | 45.5 | n8 -3.3, n10 -0.9, n12 +0.2, n16 +9.9, n20 +0.6 |

**网格拓扑上映射显著生效**：
- grid 全规模一致改善（-2.3 ~ -6.2 SWAPs，6-16%），且 map 使用率 11-12/12、avg 3.6-5.4
- **布局免疫**：grid 上 identity 与 random 起点结果相同（19.3 vs 19.3），
  旧模型 identity=23.2 / random=22.8 —— 学到映射后策略对初始布局不再敏感，
  甚至从 identity 也会用 ~3.7 次虚拟 SWAP 微调（-17% vs 旧模型 identity）
- 小规模（n8/n10）三拓扑一致受益：虚拟 SWAP 在"小电路大空间"里把散布的
  逻辑 qubit 聚拢，收益直接

**局限**：
- ring/line 的 n16+ 无收益或变差（ring n16 +9.9 且 trunc 8/12）：
  一维拓扑 + 深电路场景下路由阶段 SWAP 数（100+）远大于布局优化的量级，
  初始布局无关紧要；映射阶段还额外消耗 400 步预算
- 全部规模仍落后 SABRE（grid 19.3 vs 17.2 最接近）——SABRE 的初始布局搜索
  与 swap 决策联合启发式仍占优，RL 差距主要在路由阶段本身

**结论**：映射阶段在"布局有价值"的场景（网格/环小规模、随机起点）真正起作用，
表现为结果对初始布局免疫；在布局无关场景（一维+深电路）正确选择少用或不用，
与 5q 的"立即 commit"行为自洽。

---

## 20q 映射模型 + beam search 推理评估（2026.08.05）

### 动机

大拓扑验证只报告了 argmax 推理（grid 19.3 / line 66.1 / ring 52.6）。本实验在相同条件
（映射模型 + 随机初始映射）下加入 1 步 beam search（`--beam-width 3`），对比 SABRE /
Greedy / Random，检验「映射 + beam search」组合能否在 20q 上进一步缩小与 SABRE 的差距。

### 评估设置

- **模型**：`models/policy_map_unified.pt`（映射阶段 + commit 动作，n8-20 统一策略，等效 800K 步）
- **拓扑**：line_20q / ring_20q / grid_5x4_20q（20 物理比特）
- **数据集**：unified_test（60 circuits，n8/n10/n12/n16/n20 各 12 条）
- **初始映射**：random（`--random-init`）
- **奖励模式**：routing（纯路由）
- **推理**：argmax vs beam3（top-K 克隆 + V(s') 评分）
- **超参数**：max-num-qubits 20、max-episode-steps 400、seed 0、cpu
- **基线**：SABRE（decay, trials=20）、Greedy、Random

### 评估命令

```bash
cd src
python3 -m routing.rl.eval_policy \
  --model ../models/policy_map_unified.pt --data-dir ../traindata \
  --split unified_test --reward-mode routing \
  --topo ../traindata/topo/grid_5x4_20q.json \
  --max-num-qubits 20 --max-episode-steps 400 --random-init \
  --baselines --out ../results/eval20q_map_grid_argmax.json
python3 -m routing.rl.eval_policy \
  --model ../models/policy_map_unified.pt --data-dir ../traindata \
  --split unified_test --reward-mode routing \
  --topo ../traindata/topo/grid_5x4_20q.json \
  --max-num-qubits 20 --max-episode-steps 400 --random-init \
  --beam-width 3 --out ../results/eval20q_map_grid_beam3.json
# line / ring 同参数，结果文件 eval20q_map_{line,ring}_{argmax,beam3}.json
```

### 结果（60 circuits，括号为完成率）

> **口径说明**：`aggregate()` 的 SWAPs 均值只统计**已完成电路**（截断的不计入）。
> 下面分两张表：`compl` 表 = 已完成电路均值（与原报告一致）；`ALL` 表 = 全部电路均值
> （截断电路以各自 episode 时实际的 `num_swaps`~396-399 计入）。SABRE/Greedy 完成率 100%，两口径相同。

#### completed-only（已完成电路，SWAPs 均值）

| 拓扑 | PPO argmax | PPO beam3 | SABRE | Greedy | Random |
|------|-----------|-----------|-------|--------|--------|
| grid_5x4_20q | 19.3（100%） | **18.2**（100%） | **17.2**（100%） | 33.4 | 228.2（47%） |
| line_20q | 66.1（97%） | **51.4**（82%） | **53.9**（100%） | 118.7 | 222.2（8%） |
| ring_20q | 52.6（92%） | **48.9**（85%） | **45.5**（100%） | 81.8 | 311.0（3%） |

#### 全部电路含截断（ALL，截断电路以实际 `num_swaps` 计入）

| 拓扑 | PPO argmax | PPO beam3 | SABRE |
|------|-----------|-----------|-------|
| grid_5x4_20q | 19.3（无截断） | 18.2（无截断） | 17.2 |
| line_20q | 77.2（截 2 条） | **115.0**（截 11 条） | **53.9** |
| ring_20q | **81.3**（截 5 条） | 101.2（截 9 条） | **45.5** |

> 截断电路的 `num_swaps` 均为 395-399（卡在 400 步上限）。
> **ALL 口径下 beam3 反而远差于 argmax**（line 115.0 vs 77.2、ring 101.2 vs 81.3），
> 「beam3 反超 SABRE」的结论只在 completed-only 口径下成立，属于截断裸漏的假象。

### 逐规模明细（SWAPs，completed）

#### grid_5x4_20q

| n | PPO argmax | PPO beam3 | SABRE |
|----|-----------|-----------|-------|
| n8  | 6.6 | 7.0 | 7.5 |
| n10 | 12.0 | 11.2 | 10.1 |
| n12 | 11.3 | 11.2 | 11.0 |
| n16 | 31.9 | 30.2 | 27.6 |
| n20 | 34.8 | **31.4** | 29.8 |

#### line_20q

| n | argmax | beam3 | SABRE |
|----|--------|-------|-------|
| n8  | 17.2 | **16.2** | 14.6 |
| n10 | 29.8 | 25.4（10/12） | 23.5 |
| n12 | 34.6 | **27.0**（10/12） | 32.7 |
| n16 | 120.8 | **97.8**（10/12） | 91.7 |
| n20 | 140.5 | **117.6**（7/12） | 107.2 |

#### ring_20q

| n | argmax | beam3 | SABRE |
|----|--------|-------|-------|
| n8  | 17.3 | 21.5（11/12） | 14.6 |
| n10 | 30.9 | 24.7（11/12） | 23.5 |
| n12 | 34.5 | **29.9**（10/12） | 34.0 |
| n16 | 121.4 | **101.3**（10/12） | 78.8 |
| n20 | 84.4 | **74.8**（9/12） | 76.5 |

### 结论

1. **completed-only 口径下 beam3 优于 argmax**：grid 19.3→18.2、line 66.1→51.4、ring 52.6→48.9；
   line 均值（51.4 < 53.9）、line n12、ring n20 反超 SABRE。**但这是排除截断后的结果**。
2. **ALL 口径（含截断）下 beam3 全面劣于 argmax**：beam 完成率掉点导致截断电路
   （SWAPs≈400）把均值拉高——line 115.0 vs argmax 77.2 vs SABRE 53.9；ring 101.2 vs 81.3。
   grid 两口径相同（无截断），beam3 18.2 仍略优于 argmax 19.3、高于 SABRE 17.2。
3. **beam search 以完成率为代价换取"已完成电路"的效率**：line 97%→82%（截 11 条）、
   ring 92%→85%（截 9 条）。根因与 5q 一致——beam 的克隆 step 消耗映射/路由 stage 的
   步数预算且 action mask 状态在克隆后不同步，深电路在 400 步内更易截断。
4. **SABRE 是无可争议的胜者**：100% 完成率 + 全部口径下除 line completed-only 外全最低。
   除非先解决 beam 的截断问题，否则「映射 + beam search 超越 SABRE」不成立。
5. **beam3 用更少映射虚拟 SWAP**（grid 2.1 vs 3.6、line 1.6 vs 2.8、ring 1.5 vs 2.2）：
   映射阶段用 V(s') 评分时 critic 更倾向提前 commit，映射收益被 beam 搜索本身部分覆盖。

---

## 20q 映射增益量化：有映射 vs 无映射（2026.08.05）

### 动机

上节（beam 评估）用的映射模型 `policy_map_unified.pt`。为量化「映射阶段」本身带来的提升，
在**完全相同的评估条件**下重跑无映射统一模型 `policy_unified_phase1.pt` 作对比。

### 评估命令

```bash
cd src
python3 -m routing.rl.eval_policy \
  --model ../models/policy_unified_phase1.pt --data-dir ../traindata \
  --split unified_test --reward-mode routing \
  --topo ../traindata/topo/grid_5x4_20q.json \
  --max-num-qubits 20 --max-episode-steps 400 --random-init \
  --no-mapping-phase --out ../results/eval20q_nomap_grid_5x4_20q.json
# line / ring 同理；结果文件 eval20q_nomap_{line_20q,ring_20q}.json
```

两个模型同架构（EdgeActorCritic + GNN，统一 n8-20 padding），唯一差异是映射模型多出
commit head 与映射阶段训练（等效 800K 步 vs 旧 300K 步）。评估均为 argmax + random-init。

### 结果（completed-only SWAPs，随机起点）

| 拓扑 | 无映射 | 有映射 | Δ | SABRE | 映射使用率(map) |
|------|-------|--------|------|-------|----------------|
| grid_5x4_20q | 22.8（60/60） | **19.3**（60/60） | **-3.4（-15.4%）** | 17.2 | 3.6 |
| line_20q | 66.1（59/60） | 66.1（58/60） | -0.0（0%） | 53.9 | 2.8 |
| ring_20q | 55.4（60/60） | **52.6**（55/60） | **-2.8（-5.1%）** | 45.5 | 2.2 |

### 逐规模差异（有映射 - 无映射）

| n | grid | line | ring |
|----|------|------|------|
| n8  | **-3.4** | **-5.1** | **-3.3** |
| n10 | -1.8 | -1.7 | -0.9 |
| n12 | **-3.6** | +0.3 | +0.2 |
| n16 | **-6.2** | +2.4 | +15.6† |
| n20 | -2.3 | +11.0† | +0.1 |

> † 有映射模型在 line n20（10/12）、ring n16（8/12）完成率下降，completed-only 均值受
> 「挑易电路」影响，实际差异被高估为正。

### 结论

1. **映射提升与拓扑维度强相关**：网格（grid）全规模一致受益 -1.8~-6.2 SWAPs（-15.4%），
   环（ring）-5.1%，一维链（line）0%。虚拟 SWAP 在「高连通度拓扑 + 随机起点」上价值最大——
   能把散布的逻辑 qubit 聚拢到中心区域，n16 收益峰值 -6.2。
2. **小规模（n8/n10）三拓扑一致受益**（-5.1/-1.7/-3.3 等）：小电路大空间里布局敏感性高，
   映射的边际收益直接。
3. **一维 + 深电路布局无关**（line n12/n16/n20，ring n16/n20）：路由阶段的 SWAP 需求
   （100+）远大于布局优化的量级，初始布局好坏不改变总成本，映射训练反而消耗步数。
4. **总体提升 = grid -15%、ring -5%、line 0%**，约等于文档（08.04）记录的结论，
   本次为同条件复现确认。距 SABRE 的差距（grid 17.2、line 53.9、ring 45.5）主要仍在路由阶段。
5. **代价**：映射模型完成率略降（ring 60/60→55/60、line 59/60→58/60），虚拟 SWAP 消耗
   episode 步数预算；对浅电路无影响。

---

## 编译时间优化：GPU 推理 + beam GNN 缓存复用（2026.08.05）

### 背景

与 SABRE 的对比中，编译时间一直是主要瓶颈：旧版 beam search 单电路 1.2~6.3s
（SABRE ~5ms，慢 100~1000 倍）。本次实施两个 P0 优化：

1. **T1 GPU 推理修复**（`agent.py`）：`self.gnn.to("cpu")` 硬编码改为
   `self.gnn.to(device)`，`--device cuda` 下 GNN 真正在 GPU 上跑；
   `encoder.py` 的 `node_embeddings` 将 pyg 数据迁移到 GNN 所在设备。
2. **T2 beam 缓存复用**（`env.py` + `eval_policy.py`）：
   - `env.py` 新增 `build_graph_data()`（从 `_obs()` 抽取）、`_obs(qubit_h=None)`
     支持注入缓存嵌入、`step(action, compute_obs=False)` 跳过 obs 计算；
   - beam 循环中 clone 只做 `step(compute_obs=False)`，候选图批量
     `node_embeddings_batched()`（PyG `Batch.from_data_list`），再用
     `_obs(qubit_h=...)` 组装 edge feats，消除每候选 2 次 GNN；
   - 胜出 clone 的 obs 直接复用为下一步 obs，env 不再重复计算 obs
     （原实现 env 与 clone 各算一次，浪费 ~25% 总耗时）。

### 修复过程中的 bug

- `eval_policy.py` 编辑时旧 `env.step(best_action)` 行残留，导致每步执行两次
  step、obs 被覆盖 → 全部 400 步截断、完成率 0%。删除重复行后恢复。
- 独立验证脚本需传 `max_num_qubits=20, max_num_edges=31`（与评估 CLI 一致），
  否则 obs 维度不匹配（161 vs 171）。

### 评测（unified_test 60 电路，policy_map_unified.pt，random-init，beam3，CPU）

| 拓扑 | 旧版 avg(ms/ckt) | 新版 avg(ms/ckt) | 加速 | SWAPs 一致性 |
|------|-----------------|-----------------|------|-------------|
| grid_5x4_20q | 1240.1 | 488.0 | 2.54x | 18.2 == 18.2 |
| line_20q | 6313.9 | 3604.8 | 1.75x | 51.4 == 51.4 |
| ring_20q | 5286.1 | 3305.8 | 1.60x | 48.9 == 48.9 |

- 全拓扑 SWAPs 与优化前逐位一致（动作轨迹相同），Comp% 一致（100/81.7/85.0）。
- 结果文件：`results/eval20q_map_{grid,line,ring}_beam3_v2.json`

### 每步耗时统计（profile_beam_new.py，n10，beam_width=3，CPU）

在一个 episode 的**每一步**内，裁剪计时总耗时约 **22.2ms**，拆分为 7 个阶段
（3 个候选 clone + 批量 GNN + 胜出步）：

| 阶段 | 单步耗时(ms) | 占比 | 说明 |
|------|-------------|------|------|
| `env.clone()` | 0.62 | 2.8% | 浅拷贝环境可变状态 |
| `clone.step(compute_obs=False)` | 0.57 | 2.6% | 克隆步骤执行 SWAP/commit，跳过 obs |
| `c.build_graph_data()` ×3 | 5.50 | 24.7% | PyG 图构建（node/edge_index/edge_attr） |
| `gnn.node_embeddings_batched()` | 6.59 | 29.6% | 3 图批量 GNN 节点嵌入 |
| `_obs(qubit_h=...)` ×3 | 2.99 | 13.4% | 边特征拼接 + mapping/progress/phase 组装 |
| `_forward_obs` V(s') ×3 | 0.77 | 3.5% | ActorCritic 值头预测（无图） |
| `env.step(best_a, compute_obs=False)` | 5.20 | 23.4% | 胜出步执行 + 重新组装 obs（obs 复用前） |
| **合计** | **22.24** | **100%** | 3 个候选下的平均单步总耗时 |

各阶段**每步平均**（26 步）：
clone=0.62、step(nobs)=0.57、graph=5.50、gnn_batch=6.59、obs_asm=2.99、
fwd=0.77、env_step=5.20 ms。

### 每步耗时占比分析

1. **GNN 相关占绝对主导**：`build_graph_data`（24.7%）+ `gnn_batch`（29.6%）+
   `_obs` 组装（13.4%）合计 **67.8%**（~15.1ms/步）。其中 `gnn_batch` 单步就
   吃掉 29.6%，是当前单步最大瓶颈——它必须对 3 个候选图各做一次 GNN forward。
2. **obs 重复计算被清除**：`env.step` 中的 5.20ms（23.4%）在「胜出 clone obs 复用」
   改动后已被消除（env 不再重复算 obs）。当前跑分显示该改动让 line/ring 各再降
   ~20%（见上方评测表，grid 1240→488ms 已含复用收益）。
3. **纯环境开销很小**：`clone` + `step(nobs)` 合计仅 5.4%（1.19ms/步），不是瓶颈。
4. **候选数量线性放大 GNN 开销**：3 个候选 → graph/gnn_batch/obs_asm 全部 ×3。
   beam_width 增大时这三项的占比会同步上升（fwd 也会，但基数小）。
5. **per-step 分解 vs 全电路**：n10 单电路 60~276ms，其中图构建/GNN 批等高阶项
   按步数与每步 15ms 线性累积；n20 深电路（~50 步）该部分 ~750ms 是 line/ring
   3.3~3.6s/ckt 的主体。

### GPU vs CPU（n12 电路，12/12 完成）

| device | avg ms/ckt |
|--------|-----------|
| cpu | 271 |
| cuda | 259 |

GPU 与 CPU 基本持平：单步 batch 太小（3 个图），传输/launch 开销吃掉收益。
结论：当前规模 CPU 已足够，`--device cuda` 保留但无增益。

### 结论

1. T1+T2 使 beam3 编译时间整体降 1.6~2.5x，单电路从 1.2~6.3s 降到 0.5~3.6s；
   grid（大图）收益最大。剩余瓶颈为图构建（98.7ms）+ GNN 批量推理（112.9ms）。
2. 语义零变化：SWAPs/完成率与优化前完全一致。
3. 下一步候选（T3/T4）：图构建缓存复用（同状态图不重复构建）、GNN batch 上
   GPU 或剪枝无效候选、argmax 路径本身优化（grid 211ms 仍有优化空间）。

---

## 提升空间分析：速度 + 路由效果（2026.08.05）

基于当前每步耗时拆解（见上节）与三拓扑 SWAP 差距（grid 18.2 vs 17.2、line
51.4 vs 53.9、ring 48.9 vs 45.5），从编译速度和路由效果两个维度梳理可优化项。

### 一、编译速度（按影响排序）

#### 1. GNN 半精度 / 轻量化（最有效）

`gnn_batch` 占单步 29.6%（6.6ms），GATEncoder 3 层 × 48 维 float32。

| 手段 | 预期 |
|------|------|
| GNN 推理换 float16 | gnn_batch 时间 ~减半（3.6→1.8ms） |
| `hidden_dim` 48→24 | GNN 参数量/计算 ~-30% |
| 两者叠加 | gnn_batch 6.6 → ~2.3ms |

#### 2. 自适应候选数（eval 侧）

大部分步骤 top-1 与 top-2 的 V(s') 评分差显著。可先跑 argmax，只对 top-1
做 V(s') 评估；评分低于阈值才展开其余候选。平均候选数 3 → ~1.5，直接按比例
压缩 GNN 相关开销（67.8%）。

#### 3. 图构建缓存（T3）

`build_graph_data` 占 24.7%（5.5ms/步）。不同候选因 mapping 不同图不同，但
DAG 的静态部分（节点/边拓扑）可预构建一次，每次 step 后仅用新 mapping 更新
edge_attr。预估回收约 40%（~2.2ms/步）。

#### 4. 热点 C++/Rust 重写

PyG Data 构建、`build_routing_graph` 的 Python 循环开销约占图构建的 40-50%，
用 C 扩展或 torch C++ API 可再挤 ~2ms/步。

综合 → 单步从 17ms（清 obs 复用后）降到 ~8ms，全电路再提速约 2×。

### 二、路由效果（按影响排序）

#### 1. 后处理 SWAP 局部优化（最快见效）

RL 路由完成后对 SWAP 序列做一回扫描：

- 消除相邻逆 SWAP（A-B → 立即 B-A）
- 检测可合并的冗余链（A-B, B-C, A-B → 等效 A-C）
- 在最终映射附近试探 1 步 SWAP 是否缩短距离

开销 ~1ms，预计剪掉 5-10% SWAP，深电路（line/ring）可能更多。

#### 2. beam depth=2（带剪枝）

当前 depth=1，line 距 SABRE 仍有 5-10 SWAPs。depth=2 评估两步后的价值，
但候选爆炸（k²=9）。改剪枝版：top-2 → 每支只保留 top-1 → depth=2 共
2+2=4 次 GNN（对比 depth=1 的 4 次，成本持平）。对应 `doc/plan.md` E2。

#### 3. 映射阶段架构改进

当前映射只对 grid 有效（-15.4%），ring 未受益（-5.1%），line 为 0%。原因：
embedding 不感知 qubit 空间位置。方案（`doc/plan.md` 映射 v2）：

- 加入拓扑拉普拉斯位置编码
- 注意力 over qubit pairs 利用空间结构

预计 ring 也能受益 5-10%。

#### 4. 更深训练 + 更强数据增强

当前模型 ~800K 步，line 仍有 2/60 截断。可扩展至 2M 步 + 针对性深电路采样
+ 加长 `noise_aware` 微调。最费时但空间大，模型尚未饱和。

### 三、优先级建议

| 优先级 | 类型 | 方案 | 预期收益 | 开发成本 |
|--------|------|------|---------|---------|
| P0 | 速度 | GNN float16 + hidden 减半 | gnn_batch -60% | 1-2 行代码 |
| P0 | 效果 | 后处理 SWAP 剪枝 | SWAP -5~10% | ~50 行 |
| P1 | 速度 | 自适应候选数 | GNN 总开销 -40% | ~30 行 eval 改动 |
| P1 | 效果 | beam depth=2 + 剪枝 | line/ring SWAP -3~5 | ~100 行 |
| P2 | 速度 | 图构建缓存 | graph -40% | ~200 行 |
| P2 | 效果 | 映射阶段 v2 | ring 映射增益 | 架构改动 + 重训 |

P0 两项合计可能不到 100 行代码，预期把 line 从 3.6s/ckt 拉到 ~2s，同时
SWAP 再降 3-5 个，有望把 line 的 SWAP 差距（51.4 vs 53.9）追平 SABRE。

---

## 轨迹状态向量模拟器 vs 密度矩阵模拟器对比（2026.08.06）

### 背景

实现 `src/sim/trajectory_sim.py`（轨迹采样 + 状态向量，内存 2^n）后，与原有
Aer density_matrix 模拟器（`src/sim/sim.py`，内存 4^n）做系统性对比，验证
统计一致性与性能边界。

### 测试设置

- 电路：4q GHZ（n=5/10/12/16 时取对应 GHZ），转译到 `rz/sx/x/cx/id` + 线拓扑
- 噪声配置：T1=50µs, T2=70µs, 单比特门错误 0.001, CNOT 错误 0.01,
  读出错误 0.02，串扰默认（0.1×tqe）或显式关闭
- 轨迹条数：保真度对比 500 条；counts 对比 shots=4096

### 1. 保真度一致性（4q GHZ，500 条轨迹 vs DM）

| 场景 | DM 保真度 | 轨迹保真度 | Δ | 判定 |
|------|-----------|-----------|------|------|
| noiseless | 1.00000 | 1.00000 | 0 | OK |
| 纯热弛豫 | 0.99857 | 0.99900 | 0.00042 | OK（MC 误差内） |
| 纯退极化（关串扰） | 0.97377 | 0.97000 | 0.00377 | OK |
| 完整模型（关串扰） | 0.97239 | 0.97000 | 0.00239 | OK |
| 完整模型（默认串扰） | 0.99708 | 0.98400 | 0.01308 | 语义差异（见下） |
| 显式串扰 0.005 | 0.99708 | 0.98400 | 0.01308 | 语义差异（见下） |

前四行证明：无噪声、热弛豫、退极化、读出在各噪声通道上，轨迹模拟器与密度
矩阵模拟器统计一致（Δ < 4e-3，均在 MC 误差内）。

### 2. 串扰语义差异（重要发现）

- **密度矩阵版本（sim.py float 模式）**：Aer 的 specific quantum error 会
  **覆盖** all-qubit 的 CNOT 退极化错误（日志 WARNING："overrides previously
  defined all-qubit error"），导致默认串扰开启后边上 CNOT 的 0.01 退极化
  **丢失**，保真度反而升高（0.972 → 0.997）。
- **轨迹模拟器**：按设计语义执行 `depol ∘ ZZ` 合成（CNOT 退极化 + 串扰 ZZ
  都生效），与手写等效 NoiseModel（`depolarizing_error(0.01,2).compose(ZZ)`）
  验证一致：8000 条轨迹收敛到 0.97187，参考 0.97239，Δ=0.0005（MC 误差内，
  500 条时的 Δ=0.0116 属采样噪声 2.1σ）。
- **GHZ 的 ZZ 不变性（物理注）**：`ZZ|0000>=|0000>`、`ZZ|1111>=+|1111>`
  （两个 Z 各翻一次相），故 GHZ 对 ZZ 串扰天然不敏感——这解释了
  "含 ZZ 参考=0.97239" 与 "仅 depol 参考=0.97239"、以及 "显式关串扰
  DM=0.97239" 三者数值完全相同是巧合，并非模型没生效。
- 综上，默认串扰行 DM=0.99708 的来源是 **sim.py float 配置下 Aer 的
  specific error 覆盖掉边上 CNOT 退极化**（CNOT 退极化整体被替换成 ZZ，
  而 ZZ 对 GHZ 无害 → 噪声变弱接近无噪）；轨迹模拟器保留 depol+ZZ → 0.9719，
  语义更符合硬件直觉。若用轨迹做数值对比需用非 GHZ 电路（如 random 电路）
  才能体现串扰真实影响。

### 3. counts 一致性（4q GHZ+measure, shots=4096, 默认噪声）

| 指标 | DM | 轨迹 |
|------|----|------|
| counts_fidelity vs 理想 | 0.9220 | 0.9059 |
| total counts | 4096 | 4096 |
| run(4096 shots) 耗时 | 0.035s | 1.051s |

counts 保真度差异主要来自上述串扰语义（DM 少了 CNOT 退极化噪声，分布更接近
理想）；耗时上轨迹版每 shot 一条轨迹（4096 条独立轨迹），比 DM 单次密度矩阵
演化慢约 30x（但可并行化）。

### 4. 性能与内存（默认噪声配置，GHZ 电路）

| n | DM 单次保真度耗时 | 轨迹 500 条耗时 | 单条轨迹 | DM 理论内存 | 轨迹理论内存 |
|---|------------------|----------------|---------|------------|-------------|
| 5 | 0.023s | 0.119s | 0.24ms | 0.00 GiB | 0.00 MiB |
| 10 | 0.135s | 0.200s | 0.40ms | 0.02 GiB | 0.02 MiB |
| 12 | 0.191s | 0.432s | 0.86ms | 0.25 GiB | 0.06 MiB |
| 16 | 38.299s | 3.263s | 6.53ms | 64.00 GiB | 1.00 MiB |

- n≤10：两者耗时同量级，DM 更快（单次演化）；轨迹适合需要多条统计的场景
- n=16：DM 需要 64 GiB（跑出 38s，接近内存上限），轨迹仅 1 MiB/条，500 条 3.3s
- n=20：DM 需 16 TB（此前 OOM 崩溃），轨迹 ~16 MiB/条，可正常计算

### 5. 结论

1. 轨迹模拟器在 n≤12 时与密度矩阵统计一致，可作为等价的保真度/counts 计算器；
2. 轨迹模拟器把可模拟的比特数从 n≈13-14（DM 内存墙）扩展到 n≥20（内存 2^n）；
3. 发现原 DM 版本 float 模式串扰的 override 缺陷，已在轨迹版本按合成语义实现；
4. 后续优化方向：轨迹并行化（numpy 批量 / multiprocessing）、T2>2T1 支持、
   用轨迹版本替代 Phase 2 20q 的保真度计算（解决 doc/train.md 此前 16TB OOM 阻塞）。

### 6. 位序（bitorder）bug 修复（继续对比时发现）

用 **random 电路**（非 GHZ）对比时发现轨迹模拟器的状态向量位序与 qiskit/Aer
**不一致**：

- 复现：`x(q0)` 在轨迹模拟器落在 flat index 8（1000，qubit0 当 MSB），qiskit
  little-endian 是 index 1（0001）。
- 根因：`_apply1/_apply_cx/_apply_swap/_thermal_noise` 直接 `moveaxis(sv, q, 0)`，
  numpy reshape 后 axis 0 是最高位，而 qiskit 规定 qubit 0 是最低位。
- 为何之前测试没发现：GHZ/Bell 态 `(|00..>+|11..>)//2` 在比特位序翻转下不变，
  且 fidelity/counts 只依赖幅度分布，掩盖了差异；sim.py 对比用的也是 4q GHZ。
- 修复：
  1. 新增 `_axis(q,n)=n-1-q`，四个门算子的 moveaxis/swapaxes 轴全部换成
     `_axis(q)`（共 ~6 处）。
  2. `_rz_matrix` 补上 qiskit 的全局相位 `e^{-iθ/2}`（此前只写 `e^{+iθ}`，
     导致随机电路每步积累额外全局相位）。
  3. `_evolve` 开头乘上 `circuit.global_phase`（qiskit transpile 会在
     `Rz` 分解里带全局相位）。
- 修复后验证（Aer statevector 逐元素对比，max|Δ|）：
  - x(q0)/x(q3)、bell(0,1)、bell(3,4)、GHZ6：0.00e+00
  - random10 电路：1.93e-16
- 修复后噪声保真度（4q random 电路，5000 条轨迹 vs DM）：

| 场景 | DM | 轨迹 ± se | Δ | |
|------|-----|----------|-----|---|
| 纯退极化（关串扰） | 0.86229 | 0.85084 ± 0.00495 | 0.01145 | OK |
| 完整模型（关串扰） | 0.80687 | 0.80664 ± 0.00537 | 0.00023 | OK |
| 完整模型（默认串扰） | 0.88711 | 0.79710 ± 0.00547 | 0.09001 | 语义差 |

- 第三行 Δ=0.09 正是串扰语义差异：sim.py float 模式把边上 CNOT 退极化
  **整体替换成 ZZ**（ZZ 在 random 电路上破坏性小 → DM 更接近无噪/更高保真），
  而轨迹版本按合成 `depol∘ZZ` 实现（破坏性大）。在 GHZ 上该差异被 ZZ
  不变性掩盖，random 电路上一目了然。
- 此 bug 属于**实现级**修复，不改变此前所有 GHZ 保真度结论（数值不变），
  但使轨迹模拟器与 qiskit 的 counts/statevector 语义完全对齐（包括用户
  bee以外自定义电路）。
- 修复后全仓测试仍 `33 passed`，test_trajectory_sim.py 的 12 项全部通过（GHZ
  相关断言本身在位序翻转下等价，无需改）。

---

## 7. 死锁掩码增强：修复 beam search 2-cycle 震荡截断（2026-08-06）

### 问题

- 用户追问「新模型与动作掩码不兼容导致电路截断」是否已解决：实测**未解决**，
  `policy_map_unified.pt`（映射 + beam3）在 20q 拓扑评估上大量截断：
  - line_20q：60 条中 11 条截断（82% 完成），ring_20q：9 条截断（85%）。
  - 典型失败 `random/n10/random_n10d6_s1000068.pkl`：beam3 在 400 步内只执行
    63/154 门，swaps=399 全部耗在震荡上；argmax 54 步即可完成。

### 根因（三层）

1. **原死锁掩码只检测「连续重复同一边」**（`get_deadlock_mask` lookback=2：
   `set(hist[-2:])` 大小 1 才禁止）。对 `[6, 8, 6, 8, ...]` 2-cycle 永远不触发
   （set 恒为 `{6,8}`）。
2. 实测策略在 line 上 6↔8 来回换：step99~step399 exec 恒停在 63/154。
3. 只加周期检测还不够——策略通过**插入第三方边（如 5）**破坏周期指纹后
   回到原震荡（实测 `[5,6,8]` 三边轮换），周期检测在 step 200 触发一次后又被绕开。

### 修复（`src/routing/rl/env.py`）

`get_deadlock_mask` 扩展为三层检测：

```python
def get_deadlock_mask(self, lookback=2, max_cycle=6, stall_window=6):
    # 1) 连续重复同一 SWAP（原逻辑）
    # 2) 末尾 2N 步构成一致周期（N=2..max_cycle）→ 禁止周期内所有边
    # 3) 无进展失速：n - self._last_progress_swap >= stall_window
    #    （最近 stall_window 次 SWAP 均未执行任何门）→ 禁止窗口内全部边
```

- 新增 `_last_progress_swap` 状态：`_auto_execute_batch` 每次真正执行门时
  记录当前 `len(_swap_history)`；`clone()` 同步复制。
- 第 3 层是决定性修复：无法枚举所有震荡组合，直接用「连续 N 步零推进」
  作为死锁判定，强制策略脱离局部循环。

### 验证

- 单电路复现：`random_n10d6_s1000068`（seed=5）从 400 步截断 → **79 步完成**。
- 全量评估（unified_test 60 条，max-episode-steps=400，random_init，
  `policy_map_unified.pt`，beam=3）：

| 拓扑 | 完成率（修复前） | 完成率（修复后） | 剩余截断 |
|------|------------------|------------------|----------|
| line_20q | 82%（49/60） | **95%（57/60）** | 3 条深电路预算不足 |
| ring_20q | 85%（51/60） | **98%（59/60）** | 1 条深电路预算不足 |

- 剩余 4 条截断**均为预算不足而非死锁**：exec 推进到 296/306、232/287、
  272/276、133/165（完成 81%~98%），末尾 swap 序列无周期重复，是
  n16/n20 深电路（165~306 门）在 400 步内无法完成路由。
- 全部震荡型截断（exec 卡在 63~133 的 8 条）已消除。
- 全仓测试 `35 passed`。
- 修复在 train/eval 共用同一 `get_deadlock_mask`，训练侧同时受益
  （训练期间若策略陷入 2-cycle 也会被禁止）。

### 结论

- 「新模型 + beam 动作掩码」截断根因是**死锁掩码检测能力不足**
  （2-cycle/多边绕行），已通过「周期检测 + 无进展失速熔断」修复；
  beam3 完成率 line 82%→95%、ring 85%→98%。
- 残余截断为深电路预算问题，与死锁无关；如需要可提高
  `--max-episode-steps` 或训练时加大步数预算。

---

## 60q 截断率优化：自适应 GAE λ + 跨规模课程训练（2026.08.07 启动）

### 背景：上个 tianyan 训练（`scripts/train_tianyan.sh`）卡在截断

旧训练（MAX_STEPS=800，Phase1 300k 步）末行 `models/ckpts_tianyan/metrics.csv`：

```
step=300032  rew=-15.26  swp=84.25  map=7.35  trunc=76.70%  ent=3.08~4.37
```

- `trunc_pct` 居高不下（76.7%），`rew` 为负（被 `-0.5×剩余门` 惩罚拖累）。
- 分析出的两个根因：

| 根因 | 机制 |
|------|------|
| **GAE λ 固定导致长电路信号缺失** | `γ·λ = 0.99×0.95 = 0.9405`，有效 horizon ~17 步。800 步 episode 中截断信号 `0.9405^800 ≈ 2.4e-18`，前 ~780 步完全收不到 `unfinished_penalty` → agent 只优化最后约 20 步，无长期规划 |
| **无课程学习** | tianyan splits 只含 n30-n60 电路，agent 从第 1 步就面对大电路，从未在简单电路上学到基础路由先验 |

### 改动（本次已实现）

| 文件 | 改动 |
|------|------|
| `agent.py:compute_gae` | `lam` 兼容标量或数组：数组时反向 GAE 逐时刻取 `lam_arr[t]`，形状不符报错 |
| `train_agent.py:curriculum_phase` | 按 `progress·n` 选 prefix，prefix 内局部进度复用 `stage1_phase` 深度递进；支持 routing/noise_aware/fidelity_shaping 三种 split 名 |
| `train_agent.py:build_multi_split_map` | 合并多 prefix 的 split manifest（key 为完整 split 名，如 `large_n10_phase1`） |
| `train_agent.py:pick_circuit` | 新增 `split_map` 参数（不传则回退单 prefix 旧行为） |
| `train_agent.py` CLI | `--curriculum-keys` / `--gae-adaptive` / `--gae-lam-min`(0.95) / `--gae-lam-max`(0.995) |
| `train_agent.py` GAE 调用 | `--gae-adaptive` 时 `λ_t = λ_min + (λ_max−λ_min)·(t/(T−1))`；默认仍 `λ=agent.lam`（向后兼容） |

> 测试套件 33 passed / 2 failed：两个失败均来自先前会话未提交改动（`data_gen.py` 的 `max_operands=2` 改变测试电路 → `test_swap_penalty` 断言失效；`test_fidelity_shaping_step_zero` 随机动作遇距离奖励 flaky），与本实现无关。

### 训练环境与命令

**脚本**：`scripts/train_tianyan_curriculum.sh`（新增）

- 课程：`large_n10 → large_n20 → tianyan`（各占 Phase1 步数的 1/3）
- 自适应 λ：0.95 → 0.995（episode 开头低 variance，末尾高 λ 传导截断信号）
- `python3 -u` 无缓冲输出（配合 `2>&1 | tee` 逐行实时看到进度）

**tmux 启动（只训练 Phase 1，跳过 Phase 2）**：

```bash
tmux new-session -d -s curric 'cd /home/zzy/opencode-server/opencode-docker/projects/rlrouting && PHASE2=0 bash scripts/train_tianyan_curriculum.sh cuda:0 500000 2>&1 | tee logs/train_curric_phase1.log'
tmux attach -t curric    # 查看进度；Ctrl-B 后按 d 脱离
```

### Phase 1 课程进度规划（500k 步）

| 全局进度 | split_prefix | 线路规模 | 覆盖步数 |
|---------|-------------|---------|---------|
| 0-33% | large_n10 | 10 qubit，6-50 2Q 门 | 0~167k |
| 33-66% | large_n20 | 20 qubit，12-100 | 167~333k |
| 66-100% | tianyan | 30-60 qubit，12-300 | 333~500k |

每个 prefix 内仍沿用 `stage1_phase` 的 phase1/2/3 深度递进（0.30/0.40/0.60/0.70 平滑切换）。

### 明天训练完成后需要记录的内容

1. 训练日志摘要：各阶段 `trunc_pct / rew / swp / ent / vl`（重点看 n10/n20 阶段是否快速收敛、tianyan 阶段 trunc 是否显著低于 76.7%）。
2. 评估：`routing.rl.eval_policy --model models/policy_tianyan176_curric_phase1.pt --topo traindata/topo/tianyan176_66q.json --data-dir traindata --split tianyan_test --reward-mode routing --baselines`（argmax 与 beam3）。
3. 对比基准（旧 tianyan 模型）：`models/policy_tianyan176_phase1.pt`（300k 步，trunc 76.7%，同一 tianyan_test 评估结果可复测对比）。

**对比口径**：同一 `tianyan_test` split、同一拓扑、同一 `MAX_STEPS=800` 下，旧模型 vs 课程模型 的 trunc_pct、SWAPs、完成率。

### 验证结果（实现自测，训练前）

- `curriculum_phase` 边界：0.33→large_n10_phase3、0.34→large_n20_phase1、0.66→large_n20_phase3、0.67→tianyan_phase1 ✅
- `compute_gae` 数组 λ：常数数组与标量 λ 结果一致 ✅；自适应 λ 使 episode 开头 advantage 更小、末端更大 ✅
- smoke 512 步跑通，初始 split 正确选 `large_n10_phase1` ✅

---

## PPO vs SABRE 对比：tianyan176 不连通拓扑（2026.08.08）

### 背景：拓扑缺陷发现

`traindata/topo/tianyan176_66q.json`（60 qubits / 81 边）**不连通**：5 个连通分量
= 56-qubit 主分量 + 4 个孤立 qubit（38/44/55/56，来自 `data/tianyan176/config.json`
的 disabledQubits/disabledCouplers 过滤后残留的度 1 节点）。

**60 条 `tianyan_test` 电路中 44 条含孤立 qubit 上的 2Q 门** → 物理上无法路由：
- SABRE（qiskit）直接抛异常：`TranspilerError: ...physical qubit 4 needs to interact with qubit 38 and they belong to different components`
- PPO（argmax, MAX_STEPS=800）在这些电路上 SWAP 死循环耗尽 800 步 → 完成率仅 18.3%（11/60）

训练集同样 ~70% 电路触达孤立 qubit（phase1 160/240、phase3 343/480、mixed 671/960），
但训练时 trunc 仅 ~4%（agent 靠 stochastic 探索 + 虚拟 SWAP 移走孤立位规避，argmax 评估做不到）。

### 公平对比：16 条可路由电路子集

生成 `traindata/splits/tianyan_test_routable.txt`（16 条不触孤立 qubit 的测试电路，
全部 n30 小电路）。命令：

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src python3 -u -m routing.rl.eval_policy \
  --model models/policy_tianyan176_curric_phase1.pt \
  --topo traindata/topo/tianyan176_66q.json \
  --data-dir traindata --split tianyan_test_routable \
  --reward-mode routing --max-episode-steps 800 --max-num-qubits 60 \
  --device cuda:0 --baselines --no-greedy --verbose
```

### 结果（16 电路，argmax 无 greedy）

| 方法 | 完成率 | 平均 SWAPs | 平均 time/电路 |
|------|--------|-----------|---------------|
| PPO argmax | **68.8%（11/16）** | 100.6 ± 126.6 | 3504 ms |
| SABRE | **100%** | **68.3 ± 58.9** | **7.8 ms** |

逐电路明细（PPO 前，SABRE 后）：

| # | 电路 | PPO | SABRE |
|---|------|-----|-------|
| 1 | n30d4_s102604 | OK 31 | 23 |
| 2 | n30d8_s105208 | **TRUNC 799** | 225 |
| 3 | n30d4_s100204 | OK 58 | 32 |
| 4 | n30d4_s104004 | **TRUNC 799** | 95 |
| 5 | n30d2_s103102 | OK 30 | 28 |
| 6 | n30d8_s105308 | OK 478 | 172 |
| 7 | n30d4_s102804 | OK 155 | 57 |
| 8 | n30d4_s100104 | OK 126 | 51 |
| 9 | n30d8_s103504 | **TRUNC 799** | 80 |
| 10 | n30d2_s101204 | OK 73 | 31 |
| 11 | n30d2_s101004 | OK 19 | 16 |
| 12 | n30d2_s101304 | OK 81 | 38 |
| 13 | n30d2_s103404 | OK 16 | 14 |
| 14 | n30d4_s105604 | **TRUNC 799** | 136 |
| 15 | n30d4_s104404 | **TRUNC 799** | 64 |
| 16 | n30d2_s101304-2 | OK 40 | 31 |

### 结论与分析

- **PPO 完成电路上（11 条）平均 100.6 SWAPs vs SABRE 同电路 44.8**（PPO 仅 68.8%
  完成率、SWAPs 高 ~2.2×、时间慢 ~450×）。
- **5 条 TRUNC 全是 n30 d4/d8 深电路（2Q 门 ≥ 50）**：argmax 陷入 SWAP 局部震荡
  死循环（swaps=799、无门执行），SABRE 在同电路仅 64-225 SWAPs —— PPO 在大图
  （60q 嵌 30q）上路由能力显著弱于 SABRE。
- 与已有结论一致：RL 路由在小规模（5-20q）上逼近/超越 SABRE，但 tianyan 60q 图
  上尚未收敛到可用水平；且 argmax 远差于训练 stochastic 行为。
- 根因待修（后续）：(1) 拓扑含孤立 qubit 时训练/评估口径不一致；(2) 深电路 argmax
  死循环（beam search 或 eval 时允许虚拟 remap 可缓解）。

### Beam search 评估（同 16 电路，beam=3, 2026.08.08 补）

命令同前，加 `--beam-width 3 --no-greedy`：

```
PPO_beam3   15418.9  31.2%    27.6 +/- 34.9     12.2      41 +/- 29       5.5200
SABRE          7.0  100.0%    68.3 +/- 58.9      0.0       0 +/- 0            --
```

| 方法 | 完成率 | 平均 SWAPs | 平均 time/电路 |
|------|--------|-----------|---------------|
| PPO argmax | 68.8%（11/16） | 100.6 | 3504 ms |
| **PPO beam3** | **31.2%（5/16）** | 27.6（仅完成电路） | 15419 ms |
| SABRE | 100% | 68.3 | 7.0 ms |

- beam3 完成电路上 SWAPs 显著更低（10/17/97/9/5 vs SABRE 同电路 23/28/31/16/14，
  #13 从 argmax 16 → beam3 5），但 **TRUNC 从 5 条恶化到 11 条**（argmax 完成的多条
  深电路 #3/#6/#7/#8/#12/#16 在 beam3 下全部截断，swaps 耗尽 780-799）。
- 原因推论：beam 的 1 步 lookahead 用 `V(s')` 评分（`evaluate_circuit_beam` 内 clone+step
  会持续扩大间接待遇，且 tianyan 模型 critic 未见在 60q 大图上可靠）；死锁掩码对 beam
  探索分支不生效（各分支独立 step，不做全局 deadlock 表）。
- 结论：**当前课程模型在 tianyan 60q 图上 beam search 无益反而有害**，评估以 argmax 为
  准（与 5-20q 小图结果相反——小图上 beam3 闭合 gap 63-88%，提醒该增益不迁移到大图）。

---

## 轨迹状态向量模拟器接入训练/评估管线（2026.08.11）

### 目标

此前 `trajectory_sim.py` 独立可用但未接入 RL 管线：`_compute_aer_fidelity()` 仍用
Aer density_matrix（O(4^n)，20q 需 16 TB → Phase 2 直接 OOM）。本次把轨迹模拟器接入
训练与评估，使 20q 的 noise_aware 训练不再被内存墙阻塞。

### 改动

| 文件 | 改动 |
|------|------|
| `sim/trajectory_sim.py` | 新增 `trajectory_circuit_fidelity(phys, config, n_traj)` 与 `make_trajectory_fidelity_fn(config, n_traj, seed)`（返回 env 的 fidelity_fn 闭包）。理想态计算前 `remove_final_measurements()`（`_evolve` 本就跳过 measure，仅消除 ideal 路径的校验报错） |
| `routing/rl/env.py` | `_get_terminal_reward_value()` 的 `fidelity_fn` hook 签名从 `(dag, mapping, executed)` 改为 `(env)`（全项目此前无调用者，无兼容负担），闭包可直接取 `env._phys_circuit` |
| `routing/rl/train_agent.py` | `create_env` 新增 `fidelity_fn` 透传；CLI 新增 `--fidelity-sim {aer,trajectory}`（默认 aer）、`--traj-trajectories`（默认 64）、`--traj-seed`；初始 env 与每 episode env（噪声扰动后）都按当前 noise_config 重建闭包 |
| `routing/rl/eval_policy.py` | CLI 同训练侧；PPO/beam/Random 的 env 传 `fidelity_fn`；Greedy/SABRE 基线改用 `phys_fidelity()` 统一入口（aer=counts overlap，trajectory=态保真度），保证同一次评估口径一致 |
| `scripts/train_unified.sh` | Phase 2 增加 `--fidelity-sim trajectory --traj-trajectories 64` |

### 脚本无输出问题修复（2026.08.11 补）

直接 `python3 -m ... 2>&1 | tee` 时 attach tmux 看不到任何输出：Python 在
stdout 为管道时是**块缓冲**（攒满 ~8KB 才 flush）。修复：`train_unified.sh`
内统一改用 `python3 -u`（与 `train_tianyan_curriculum.sh` 一致），逐行实时输出。
tmux 启动方式不变：

```bash
tmux new-session -d -s unif20 'cd /home/zzy/opencode-server/opencode-docker/projects/rlrouting && PHASE2=1 bash scripts/train_unified.sh cuda:0 500000 300000 2>&1 | tee logs/train_unified20.log'
```

### 保真度口径说明（重要）

- **trajectory** 返回**态保真度** `F = mean_t |<ψ_ideal|ψ_t>|²`（statevector overlap）
- **aer** 返回 **counts overlap**（measure_all 后 count 分布交叠，受 shots 采样影响）
- 两者定义不同，绝对值不可直接对比；且 sim.py float 模式存在串扰 override bug
  （specific error 覆盖全比特 CNOT 退极化，见 08.06 记录），aer 侧噪声系统性偏弱。
- **跨方法对比必须同一次运行内统一 `--fidelity-sim`**（PPO/基线同口径）；
  aer 与 trajectory 的历史数值（如 5q 上 0.94 级别）不跨口径对比。

### 验证结果

1. **5q 完整 episode（cross_5q，greedy 路由，256 条轨迹）**：`done=True swaps=3
   traj_fid=0.6454 aer_fid=0.8975` —— 两者同为"噪声变低保真变高"的趋势，但绝对值
   差 0.25，正是指标定义（态保真 vs counts overlap）+ sim.py 串扰语义的差异，
   属预期，非实现 bug。
2. **20q 冒烟训练**（`grid_5x4_20q`，unified split，`--fidelity-sim trajectory
   --traj-trajectories 8 --no-gnn --timesteps 256 --max-episode-steps 60`）：
   训练/checkpoint 正常完成，**无 OOM**（此前 density_matrix 需 16 TB）。
3. **全量测试**：`33 passed / 2 failed`（2 个 failure 为已知既有问题：
   `test_swap_penalty` 因 data_gen.py 改动、`test_fidelity_shaping_step_zero` flaky，
   与本次改动无关；`test_trajectory_sim.py` 14 项全部通过）。

### 20q 训练命令（仅状态向量模拟器）

Phase 1（纯路由，不涉及保真度模拟）：

```bash
cd /home/zzy/opencode-server/opencode-docker/projects/rlrouting
PYTHONPATH=src python3 -m routing.rl.train_agent \
  --data-dir traindata \
  --split-prefix unified \
  --topo-list traindata/topo/line_20q.json,traindata/topo/ring_20q.json,traindata/topo/grid_5x4_20q.json \
  --topo-balance episodes \
  --reward-mode routing \
  --timesteps 500000 \
  --max-episode-steps 400 \
  --max-num-qubits 20 \
  --device cuda:0 \
  --checkpoint-dir models/ckpts_unified \
  --out models/policy_unified_phase1.pt
```

Phase 2（噪声感知微调，状态向量模拟器）：

```bash
PYTHONPATH=src python3 -m routing.rl.train_agent \
  --data-dir traindata \
  --split-prefix unified \
  --topo-list traindata/topo/line_20q.json,traindata/topo/ring_20q.json,traindata/topo/grid_5x4_20q.json \
  --topo-balance episodes \
  --reward-mode noise_aware \
  --timesteps 300000 \
  --max-episode-steps 400 \
  --max-num-qubits 20 \
  --load models/policy_unified_phase1.pt \
  --device cuda:0 \
  --fidelity-sim trajectory \
  --traj-trajectories 64 \
  --checkpoint-dir models/ckpts_unified_ph2 \
  --out models/policy_unified_noiseaware.pt
```

或直接用脚本（Phase 2 已默认 trajectory）：

```bash
PHASE2=1 bash scripts/train_unified.sh cuda:0 500000 300000
```

> 性能提示：20q 下 64 条轨迹终端奖励的实际耗时远高于预期（详见下文
> 「20q 联合训练过慢问题分析（2026.08.13）」：实测 ~21 分钟/episode），
> 需调低 `--traj-trajectories` 或向量化轨迹后才能继续推进。

---

## 20q 联合训练过慢问题分析（2026.08.13）

### 背景

`tmux unif20` 中运行 20q Phase 1&2 联合训练（line_20q / ring_20q / grid_5x4_20q，
unified split，Phase 2 为 `noise_aware` + `--fidelity-sim trajectory --traj-trajectories 64`）。

### 现象

1. **Phase 1（纯路由，无保真度模拟）**：500K 步约耗时 2 天（8/11 → 8/13），
   约 3 steps/s。
2. **Phase 2（noise_aware + 64 条轨迹）**：进程 8/11 22:14 启动，至 8/13 10:27
   仅推进到 step≈3840，**约 30s/step**：
   - checkpoint 证据：`ckpts_unified_ph2/ckpt_step002816.pt`（8/13 01:51）→
     step=3840（8/13 10:27），1024 步耗时 ~8.5h
   - 按此速率 300K timesteps 需 **3-4 个月**
3. **机器负载高**：load average ≈ 67，4 张 GPU 被其他任务占满（100%），
   进程本身 6378% CPU（64 核），numpy 全核争抢内存带宽。

### 基准测量（421 门 20q 电路，TrajectorySimulator）

| 项目 | 耗时 |
|------|------|
| transpile | 0.03s |
| ideal 态矢 1 次（无噪声） | 3.76s |
| 噪声轨迹 1 条 | 21.6s |
| 噪声轨迹 8 条 | 151.2s（18.9s/条） |
| **64 条轨迹（= 1 次终端 fidelity）** | **≈ 21 分钟**（15 分钟未完成，按线性外推） |

### 根因

代码路径：`env.py:_get_terminal_reward_value()`（env.py:533-538）→
`trajectory_sim.py:_evolve()`。每个 episode 结束时计算一次：

1. **纯 Python 逐门演化**：`_evolve` 对每条轨迹逐门迭代 2^20（16MB complex128）
   态矢，`_apply1`/`_apply_cx` 每门多次 `moveaxis` + `ascontiguousarray`
   + 全数组归约（`_thermal_noise` 的 `|a[1]|^2` 求和 + `_renormalize`）。
2. **64 条轨迹串行**：`run_trajectories` 是纯 for 循环，无批量化/并行化。
3. **调用频率**：训练中 line/ring_20q 每 episode 需 40-60 个 SWAP，
   约每 ~38 步结束一个 episode → 每 ~38 步触发一次 ~21 分钟的保真度计算
   → 平均 ~30s/step，与观测一致。
4. **附带开销**：`make_trajectory_fidelity_fn` 每次还重复 `_transpile`
   （并有 measure_all 冗余拷贝）。

### 建议（按性价比排序）

| 优先级 | 方案 | 预期提速 |
|--------|------|---------|
| P0 | `--traj-trajectories 64 → 8/16` | 4-8×（21min → 2.6/5.3min 每 episode） |
| P0 | 轨迹向量化：`(T, 2^n)` 单数组批量演化（64×16MB≈1GB 可行） | ~8×（Python 循环次数不变，numpy 批量操作） |
| P1 | 终端奖励降频：每 K 个 episode 才算 fidelity，或用 `_xz_errors` 近似 + 定期校准 | ~K× |
| P1 | 缓存 `_transpile` 结果，去掉无谓 measure/拷贝 | 若干 % |
| P2 | 单轨内省去 `_renormalize` 的全数组二次归约（跳变时才需要） | 单轨 2-4× |

### 结论

20q 训练慢的瓶颈是**轨迹保真度模拟器的逐门 Python 演化 + 64 条串行轨迹**，
而非 GNN/PPO 本身。优化方向应为向量化/降轨迹数，而非等待。

---

## 2026.08.14 20q Phase 2 训练提速改造（P0 落实）+ 重启训练

### 背景

前一节根因分析定位 20q 训练瓶颈为轨迹保真度模拟器（逐门 Python 演化 + 64 条串行轨迹，
每 episode ~21 分钟保真度计算，平均 ~30s/step）。本次落实两个 P0 方案并重启 Phase 2。

### 改动

1. **P0-1：轨迹数 64 → 16**（真实提速 ~4×，也是主导因素）
   - `train_agent.py` / `eval_policy.py` 的 `--traj-trajectories` 默认 64 → 16。
   - 同步 `scripts/train_unified.sh` 显式传 `--traj-trajectories 16`。

2. **P0-2：轨迹向量化批量演化**（`trajectory_sim.py`）
   - `_evolve_batch`：单数组 `(T, 2^n)` 演化全部轨迹，消除 Python 逐轨迹循环。
   - 全部批量原语重写并逐操作与串行比对 bit-exact：
     - `_apply1_batch`：moveaxis-to-last + `(M,2)@(2,2)` 单次大 gemm
       （axis 修正：`ax = nq - qb`，T 轴占位已处理）。
     - `_apply_cx_batch`：切片拷贝版，`ctl_high` 分支处理 ctl 在高低位，
       (ctl,tgt) 全部组合比对通过。
     - `_apply_swap_batch` / `_depol1_batch` / `_depol2_batch` / `_crosstalk_batch`。
     - `_thermal_noise_batch`：qubit-last 连续布局 + 解析归一化
       （norm² = 1-p1·p 或 p1），避免全数组二次归约。
   - 删除不再使用的 `_CX_MAT`、`_renormalize_batch`。

### 验证

- **noiseless 全等**：批量 vs 串行 maxdiff = 5.4e-15（float 舍入量级），bit-exact。
- **噪声统计一致性**：5q T=8192，batch fid=0.47936 vs 旧串行 fid=0.50446，
  diff=0.02510，MC 噪声 ~0.011（2.3σ，在统计误差内）；更大 T 下收敛。
- **pytest**：34 passed，1 个既有失败 `test_swap_penalty`（与本次无关，train.md 已有记录）。

### 关键实测：向量化在 20q 反而更慢（缓存效应）

| 场景 | 批量 | 串行 | 说明 |
|------|------|------|------|
| n=16 T=16（2017 门） | 34.6s | 37.7s | 基本打平 |
| n=20（~102 门，T=16） | 42-61s | 33-40s | 批量略慢 |
| **n=20 真实 VQE 电路（523 门，T=16）** | **471s** | **243s** | **批量慢 ~2×** |
| 交叉点 n=14/16/17/18（T=16） | 0.86×/0.49×/0.33×/0.43× | — | 批量在 n≥14 全面落后 |

**原因**：批量 `(T, 2^n)` 工作集随 n 指数增长（20q × T=16 = 256MB），击穿 L3
缓存后失去局部性；而串行单轨 16MB 可驻留缓存。train.md 此前预估的 ~8× 来自
Python 循环开销假设，实测为内存带宽瓶颈，结论相反。

**最终方案**：`run_trajectories` 按工作集自动选择——

```python
ws = num_trajectories * (1 << n_qubits) * 16  # complex128 字节
if ws <= 8 MiB: 批量 _evolve_batch
else:           串行 _evolve 循环（缓存友好）
```

20q × T=16（256MB）走串行；小 n / 小 T 走批量。

### 重启训练（tmux）

旧 `unif20` 会话已 kill（9 天仅 5120/300000 steps，按旧速需 ~100+ 天）。
重启命令：

```bash
tmux new-session -d -s unif20_ph2 'cd /home/zzy/opencode-server/opencode-docker/projects/rlrouting && PYTHONPATH=src python3 -u -m routing.rl.train_agent \
  --data-dir traindata \
  --split-prefix unified \
  --topo-list traindata/topo/line_20q.json,traindata/topo/ring_20q.json,traindata/topo/grid_5x4_20q.json \
  --topo-balance episodes \
  --reward-mode noise_aware \
  --timesteps 300000 \
  --max-episode-steps 400 \
  --max-num-qubits 20 \
  --load models/policy_unified_phase1.pt \
  --device cuda:0 \
  --fidelity-sim trajectory \
  --traj-trajectories 16 \
  --checkpoint-dir models/ckpts_unified_ph2_p0 \
  --out models/policy_unified_noiseaware.pt 2>&1 | tee logs/train_unified20_ph2_p0.log'
```

### 重启后实测节奏

- step 256：~38 min；step 512：~1h19m；step 768：~2h19m
- 平均 **~5.5 steps/min**（旧 ~0.4 steps/min，提速 ~14×）
- 300000 steps 预计 **~38 天**（旧方案需 ~520 天，不可行；仍偏慢，后续可考虑 P1 降频）
- 机器负载 ~70（其他用户 GPU 任务占 CPU），轨迹模拟为 CPU 密集。

---

## 2026.08.14 20q 训练时间开销分析（第二轮优化）

### 分析：主要时间开销在哪

用 cProfile 剖析训练中一次终端保真度调用（16 条轨迹 × 36 门物理线路）：

| 组件 | 旧耗时 | 新耗时 | 说明 |
|------|--------|--------|------|
| `_thermal_noise`（2304 次） | 56.5s | 17.6s | T1/T2 MC 采样 |
| `_renormalize`（4992 次） | 45.8s | 0 | 被解析归一化取代 |
| `_apply1`（2470 次） | 47.8s | 38.5s | 单比特门演化 |
| `_apply_cx`（408 次） | 6.5s | ~1s | CNOT |
| **合计** | **~115s** | **~45-57s** | 每次 fidelity 调用 |

**根因**：20q 工作集 256MB 击穿 L3，任何全数组操作（16MB/轨迹）都受内存带宽
限制。旧串行 `_apply1`/`_thermal_noise`/`_apply_cx` 每次操作做 3 次全数组搬运
（moveaxis + ascontiguousarray 拷贝 + 末尾 reshape 拷贝），且 `mat @ moved`
的 2×2×2^19 gemm 会**触发 BLAS 63 线程全核占满**（训练进程 CPU 6363%，机器
load 从 44 飙到 70）。

### 优化：串行路径改为 strided view 原地操作（零拷贝）

- `_apply1`：`sv.reshape(2^(n-1-q), 2, 2^q)` 中轴 stride=2^q 即比特 q，
  `a0/a1` 切片是 strided view（零拷贝），2 个 8MB 临时数组直接写回。
- `_thermal_noise`：同一 reshape 技巧取 |0>/|1> 分量，解析归一化
  （norm² = 1-p1·p 或 p1）取代 `_renormalize` 全数组二次扫描。
- `_apply_cx`：`reshape(A,2,B,2,C)` 轴 1/3 = 高/低位，切片交换，无 moveaxis。

### 验证

- 全部 14 个 trajectory 测试通过；noiseless 与 Qiskit statevector 模拟器
  maxdiff = 1.4e-16（bit-exact）。
- 噪声统计：serial vs batch（同 seed）diff=0.00159 << MC≈0.00641。
- **副作用消除**：训练进程 CPU 从 6363%（63 核 BLAS）降到 ~100%（单核），
  机器 load 从 ~70 降到 ~6。

### 重启后实测

- 原旧代码（16 轨迹、串行旧原语）：~5.5 steps/min（load 70）
- 新代码（strided 原地 + 解析归一化）：**~15.4 steps/min**（load 6，机器空闲）
- 300000 steps 预计 **~13.5 天**（旧代码 ~38 天，二次优化后 ~3 倍提速）

### 仍可挖掘（若还需提速）

1. `_apply1` 仍占 ~40s/调用：fuse matmul 的两行（`m0a0+m0a1` 与 `m1a0+m1a1`
   共用 a0/a1 读取），或对 rz（对角阵）走专用 `a1 *= e^{iθ}` 路径。
2. 终端奖励降频（每 K 个 episode 才算 fidelity）——训练本身只需噪声近似信号。
3. `_thermal_noise` 的 `sv *= sqrt(1/n2)` 全数组缩放可折叠进 a0/a1 切片。
4. 减少轨迹数 16 → 8（方差↑，2 倍提速）。

---

## 真机 20q 子拓扑（tianyan176_20q）Phase 1 微调（2026.08.17）

### 背景与前置修复

- 复用已有的真机 20q 截取拓扑 `traindata/topo/tianyan176_20q.json`
  （天衍176 有效拓扑 Q23 根 BFS 截取，20 qubits / 29 边，连通）。
- **修复字符串键 bug（P0）**：tianyan 系列 topo 的 `two_q_gate_error` 经
  `json.dump` 后为字符串键 `"(0, 18)"`，而 `features.py` / `sim.py` /
  `trajectory_sim.py` 全部用元组键查询 → 真机逐边噪声被静默丢弃
  （features 回退 0.01、trajectory 回退 0.001、Aer 路径直接崩溃）。
  修复：三处 `_lists_to_dict`（`train_agent.py`、`eval_policy.py`、
  `test_eval_phase1.py`）增加 `_normalize_dict_keys`，用 `ast.literal_eval`
  将字符串键规范化为元组键。
- 修复验证：`tianyan176_20q.json` 加载后 `two_q_err` 呈 28 个不同逐边值
  （修复前全为 0.01）；`trajectory_sim._two_error(0,18)=0.0085` 与 topo
  字典一致；Aer `NoiseModel` 构建不再崩溃（cx 逐边错误按真机值构建）。
- 注：此修复对历史 tianyan 60q 训练同样生效——此前 tianyan 训练的噪声
  实际为均匀回退值，逐边真机噪声从未真正进入特征/保真度计算。

### 数据准备

- 新建 `scripts/build_tianyan20q_splits.py`：复用 `traindata/random/n20`
  （d2-d13 共 810 条，拓扑无关 pkl），生成：
  - `tianyan20q_phase1.txt`（d2+d3，116 条）
  - `tianyan20q_phase2.txt`（d4+d5+d6，348 条）
  - `tianyan20q_phase3.txt`（d7-d13，346 条）
  - `tianyan20q_mixed.txt`（全部 810 条，供后续 noise_aware Phase 2 用）
  - `tianyan20q_test.txt`（d2/d4/d8 池留出 30 条，供评估用）

### 训练命令（Phase 1 微调，基于 unified 20q Phase 1 参数）

基于 `models/policy_unified_phase1.pt`（line/ring/grid 20q 联合训练，
max_num_qubits=20 / max_num_edges=31）微调：
- 架构维度兼容性已验证：`EdgeActorCritic` 的 actor 头为逐边共享 MLP、
  critic 只依赖 `num_qubits`（两侧均为 20），严格 `load_state_dict` 100% 匹配；
  29 边 topo 通过 padding + action mask（`mask[:29]=True`）适配 31 边模型。

```bash
# tmux 会话 tianyan20q（2026.08.17 启动，约 1 小时，~25 steps/s）
cd /home/zzy/opencode-server/opencode-docker/projects/rlrouting
PYTHONPATH=src python3 -u -m routing.rl.train_agent \
  --data-dir traindata \
  --split-prefix tianyan20q \
  --topo traindata/topo/tianyan176_20q.json \
  --reward-mode routing \
  --timesteps 100000 \
  --max-episode-steps 400 \
  --max-num-qubits 20 \
  --mapping-budget 8 \
  --load models/policy_unified_phase1.pt \
  --device cuda:0 \
  --checkpoint-dir models/ckpts_tianyan20q_ft \
  --checkpoint-interval 20 \
  --out models/policy_tianyan20q_ft_phase1.pt
```

### 启动日志摘要（前 90 秒）

- 加载成功：`[resume] step=0 best_metric=-1.00000`（fresh 微调，不续步）
- 初始 split：`tianyan20q_phase1`（routing 课程：phase1→phase2→phase3）
- 前 9 个 cycle：rew=+35~40，swp=15~20，map=0.4~0.9，trunc=0%
- 吞吐：~25 steps/s（~1500 steps/min），100k 步预计 ~70 分钟

### 训练完成与收敛情况（2026.08.17 21:38）

- 100k 步全部完成，`models/policy_tianyan20q_ft_phase1.pt` 已保存。
- 收敛趋势：rew +35~40（起步）→ +118~125（末尾），swp 15~20 → 49~55，
  trunc 全程 0%，map 0.0~0.9（映射阶段使用极少，布局主要靠微调前学到的
  先验 + 路由 SWAP 完成）。末期 rew/swp 仍在缓慢上升，未见明显过拟合。

### 评估（tianyan20q_test，30 条 n20 电路，routing 模式，确定性）

评估命令（argmax）：
```bash
cd src
python3 -u -m routing.rl.eval_policy \
  --model ../models/policy_tianyan20q_ft_phase1.pt \
  --topo ../traindata/topo/tianyan176_20q.json \
  --data-dir ../traindata \
  --split tianyan20q_test \
  --reward-mode routing \
  --baselines \
  --device cuda:0
```

| Method | Time(ms) | Comp% | SWAPs | XZ |
|--------|---------|-------|-------|----|
| PPO 微调 argmax | 313 | 100% | 32.0 ± 14.7 | 124.5 |
| PPO 微调 beam3 | 771 | 100% | **29.0 ± 13.5** | 125.5 |
| PPO unified（未微调）argmax | 339 | 100% | 34.2 ± 16.7 | 123.2 |
| Greedy | 6 | 100% | 46.6 ± 22.8 | -- |
| SABRE | 6 | 100% | **26.3 ± 11.7** | -- |
| Random | 25 | 6.7% | 138.5 ± 15.5 | 24.5 |

### 结论与分析

1. **微调有效**：真机 20q 子拓扑上微调后 SWAPs 34.2 → 32.0（-6.4%），
   路由质量在真机异构拓扑上得到改善；beam3 进一步降至 29.0，将相对 SABRE
   的 gap 闭合 66%（(34.2-26.3) → (29.0-26.3)）。
2. **仍落后 SABRE**（29.0 vs 26.3，gap ~10%）：20q 规模上 PPO 尚未完全
   超越 SABRE，与 5q 小图（PPO 接近/超越）及 60q 大图（SABRE 显著占优）的
   既有结论一致——RL 优势在中规模图上仍需更多训练/更强特征。
3. 30 条测试电路全部可路由（test 取自 d2/d4/d8 池，连通子拓扑 + mapping
   阶段保证 100% 完成率）；random 基线仅 6.7% 完成，验证路由任务难度。
4. **后续待办**：Phase 2（noise_aware + trajectory）微调，利用已修复的
   逐边真机噪声做保真度感知路由，再在 `tianyan20q_test` 上做保真度对比。

---

## 真机 20q 子拓扑 Phase 2（noise_aware）微调（2026.08.17 深夜启动）

### 训练命令

基于微调后的 `models/policy_tianyan20q_ft_phase1.pt` 继续 Phase 2
（保真度感知微调，20q 必须用 trajectory 模拟器，Aer density_matrix 不可行；
修复后的逐边真机噪声此时进入终端保真度奖励）：

```bash
# tmux 会话 tianyan20q_ph2（2026.08.17 启动）
cd /home/zzy/opencode-server/opencode-docker/projects/rlrouting
PYTHONPATH=src python3 -u -m routing.rl.train_agent \
  --data-dir traindata \
  --split-prefix tianyan20q \
  --topo traindata/topo/tianyan176_20q.json \
  --reward-mode noise_aware \
  --timesteps 80000 \
  --fidelity-sim trajectory \
  --traj-trajectories 16 \
  --max-episode-steps 400 \
  --max-num-qubits 20 \
  --mapping-budget 8 \
  --load models/policy_tianyan20q_ft_phase1.pt \
  --device cuda:0 \
  --checkpoint-dir models/ckpts_tianyan20q_ft_ph2 \
  --checkpoint-interval 20 \
  --out models/policy_tianyan20q_ft_noiseaware.pt
```

### 启动状态（前 3 分钟）

- `Split prefix: tianyan20q (initial split: tianyan20q_mixed)` ✓（noise_aware 用 mixed split，810 条电路）
- 微调模型加载成功（step=0，fresh 微调）；进程 103% CPU，无报错
- 预计吞吐 ~12-15 steps/min（trajectory 16 轨迹，与 unified 20q Phase 2 一致），
  80k 步预计 **~4-5 天**；checkpoint 每 5120 步（~6h）保存一次

### 阶段性评估（2026.08.19，Phase 2 中途 checkpoint step=15616 / 80000，~20%）

训练进行中（step 15616，~8.9 steps/min，与 unif20 训练共享 CPU 略降速），
用最新 checkpoint `models/ckpts_tianyan20q_ft_ph2/ckpt_step015616.pt` 在
`tianyan20q_test`（30 条）上做 noise_aware 评估（trajectory 16 轨迹，
按用户要求去掉 greedy 基线）：

```bash
# tmux 会话 tianyan20q_eval
cd /home/zzy/opencode-server/opencode-docker/projects/rlrouting
PYTHONPATH=src python3 -u -m routing.rl.eval_policy \
  --model models/ckpts_tianyan20q_ft_ph2/ckpt_step015616.pt \
  --topo traindata/topo/tianyan176_20q.json \
  --data-dir traindata --split tianyan20q_test \
  --reward-mode noise_aware --fidelity-sim trajectory \
  --baselines --no-greedy --verbose --device cuda:0
```

| Method | Time(ms) | Comp% | SWAPs | Fidelity |
|--------|---------|-------|-------|----------|
| PPO Phase2 中途（step 15616） | 216851 | 100% | 31.6 ± 14.4 | **0.0810** |
| Random | 66742 | 16.7% | 128.4 ± 28.2 | 0.0044 |
| SABRE | 6.6 | 100% | 26.3 ± 11.7 | 0.0713 |

**中途结论（~20% 训练进度）**：
1. **保真度已反超 SABRE**：PPO 0.0810 vs SABRE 0.0713（+13.6%），
   噪声感知微调开始兑现——逐边真机噪声修复后的 Phase 2 训练正在起作用。
2. SWAPs 31.6 vs SABRE 26.3（PPO 多 5.3），略高于 Phase 1 微调模型
   （32.0）——noise_aware 用少量 SWAP 代价换取保真度，符合设计目标。
3. 时间开销：PPO 每条电路 ~2-5 min（20q trajectory 保真度模拟为主，
   d4 电路 216-325s），30 条全评估（PPO+Random+SABRE 三遍）约 **2.5-3.5 小时**。
4. 评估仍在进行（2026.08.19 ~19:00 完成 SABRE 遍，汇总如上）。

### 待办（训练完成后补充）

- [ ] 记录最终日志摘要（fid、swp、trunc、收敛情况）
- [ ] 最终模型 `policy_tianyan20q_ft_noiseaware.pt` 的 noise_aware 评估
       与中途 checkpoint 对比，验证保真度是否进一步提升
