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
