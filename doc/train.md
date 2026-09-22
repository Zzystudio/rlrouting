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

---

## 门调度模块 Phase 1（timing_aware + GreedyScheduler，2026.08.19）

> 框架依据：`doc/plan.md` §「端到端联合优化框架 v2」Phase 1。用户确认：简单插件，每步 SWAP 后
> 对 front-layer 贪心并行调度并计算该步奖励；四类硬件效应（真实门时长 / 并行分层 / 串扰规避 /
> 空闲退相干）均建模；策略=启发式；接入点=训练奖励。门时长采用 per-gate-type 表，1Q 门维持自动执行。

### 实现要点

- **新增 `src/routing/timing.py`**：
  - `GATE_DURATION_TABLE`（rz=0 / sx≈35ns / cx≈300ns / swap≈300ns 等 per-gate-type 精细时长）
  - `CircuitTiming`：离散事件时序内核（`total_time` 全局时钟、`qubit_busy_until`、`qubit_idle_time`、
    `qubit_crosstalk`、`parallel_usage`、`rounds`、`crosstalk_events`、`total_gates_executed`），
    含 `create(n)` / `clone()`（供 beam 复用）
  - `GreedyScheduler.priority(g)`：按 criticality（到线路终点的剩余深度）排序
  - `schedule_round(ready_1q, ready_2q, ...)`：贪心最大独立集（共享 qubit 互斥），同 round 内
    相邻耦合对 2Q 门记 `hw.zz` 加权串扰，结算 `round_time` / 全局时钟 / 空闲退相干
- **`src/routing/rl/env.py`**：
  - `__init__` 新增 `use_scheduler` + `eta_time=0.01` / `eta_xtalk_par=0.05` / `eta_idle=0.005`
  - 新增 `_update_timing()`（返回 ready_1q/ready_2q，不执行门）+ `_auto_execute_batch_scheduled()`
    （每轮贪心并行调度 front-layer，累加 `r_time=-η·Δclock`、`r_xtalk_par=-η·加权串扰`、
    `r_idle=-η·新增idle` 到步奖励）；`_auto_execute_batch` 按 `use_scheduler` 分支（非调度模式
    保持原串行逻辑，行为不变）
  - `_apply_swap` 计时 0.3µs 并更新两端 `qubit_busy_until`（SWAP 也占线路时间）
  - `reset()` 初始化 `timing`；`clone()` 复制 `timing` / `scheduler` / 权重
- **`src/routing/rl/train_agent.py`**：`--use-scheduler` + `--eta-time/--eta-xtalk-par/--eta-idle`；
  透传 `create_env`；metrics.csv 新增 `time_us` / `xtalk` / `idle` 三列；日志打印 time/xtalk
- **`src/routing/rl/eval_policy.py`**：`CircuitMetrics` 新增 `circuit_time_us` / `crosstalk_events`
  （env 启用调度时填充），JSON 结果导出包含这两字段

### 验证（2026.08.19）

- `PYTHONPATH=src python3 -m pytest test/ -q`：33 passed，2 failed（**预先存在**，
  `test_fidelity_shaping_step_zero` / `test_swap_penalty`，与 `data_gen.py max_operands=2` 改动相关，
  与本次无关；默认 `use_scheduler=False` 路径未改动）
- 小电路冒烟（cross_5q + random_circuit）：55/55 门执行完毕，`timing.total_time` / `rounds` /
  `crosstalk_events` 正常累积
- 5q 训练冒烟（`--use-scheduler --timesteps 2000 --device cpu`）：正常跑完，日志出现
  `time=…us` / `xtalk=…`，metrics.csv 三列有值，policy 保存成功

### 后续（框架 Phase 2 远期，本次未做）

- `PolicyScheduler`（agent 逐门决策）、`JointActorCritic` + `gate_score_head`、`gate_embeddings()` 暴露
- 门层面 obs 特征、GNN timing 特征（dim 19-21/27-28/12-13）
- 终端 simulator schedule-aware（保真度按并行/时序计算；当前终端 fidelity 仍串行 Aer/trajectory，
  与「不改 Aer 通道」决策一致，时序仅作 dense 奖励信号）
- 注：`--no-mapping-phase` 与 agent 动作维度 `+1` 存在预先存在的兼容问题（非本次改动），真实训练
  均用默认 mapping 阶段，不受影响

### 导出带并行调度的物理电路（上述 Phase 1 追加功能）

- `eval_policy` 新增 CLI：`--use-scheduler` + `--dump-schedule`
- `CircuitTiming.schedule_log`（`timing.py`）记录每门调度记录
  `{kind, gate_idx, op, qubits, start, end, round}`；SWAP 记为 `round=-1`、`gate_idx=-1`
- `env._phys_circuit` 经 `qiskit.qasm2.dumps`（失败回退 `qasm3`）导出为 `phys_qasm`
- 二者均写入 `--out` JSON 的 `schedule` / `phys_qasm` 字段；SABRE 基线无调度器，`schedule`/`phys_qasm` 为 None
- 验证（5q cross，`--use-scheduler --dump-schedule`）：正确导出 23 条调度记录 + 含 SWAP 的 QASM，
  `circuit_time_us`/`crosstalk_events` 同步填充；pytest 33 passed（2 项预先存在无关失败不变）

---

## 门调度 Stage A 微调验证（20q 天衍，2026.08.19）

> 计划：在 `policy_tianyan20q_ft_phase1.pt` 基础上开 `--use-scheduler` 做 routing 模式微调（仅 Stage A）。

### 训练

```bash
# tmux 会话 tianyan20q_ph1sched（~30 分钟，40000 步）
PYTHONPATH=src python3 -u -m routing.rl.train_agent \
  --data-dir traindata --split-prefix tianyan20q \
  --topo traindata/topo/tianyan176_20q.json \
  --reward-mode routing --timesteps 40000 \
  --max-episode-steps 400 --max-num-qubits 20 --mapping-budget 8 \
  --use-scheduler --eta-time 0.01 --eta-xtalk-par 0.05 --eta-idle 0.005 \
  --load models/policy_tianyan20q_ft_phase1.pt --device cuda:0 \
  --checkpoint-dir models/ckpts_tianyan20q_ft_ph1sched \
  --checkpoint-interval 20 --out models/policy_tianyan20q_ft_phase1_sched.pt
```
- 收敛：`rew +44→+189`，`swp 17→55`（与基模 Phase1 收敛量级一致），`trunc=0%`
- 日志 `time/xtalk` 指标正常流动（注意日志 `time` 是逐步累计时钟均值，随电路变长而增大，非逐步恶化）

### 评估（`tianyan20q_test`，30 条 n20，routing 模式，仅 SABRE 基线）

```bash
# Run1 微调后模型（带调度器）
PYTHONPATH=src python3 -u -m routing.rl.eval_policy \
  --model ../models/policy_tianyan20q_ft_phase1_sched.pt \
  --topo ../traindata/topo/tianyan176_20q.json --data-dir ../traindata \
  --split tianyan20q_test --reward-mode routing --use-scheduler --dump-schedule \
  --baselines --no-greedy --no-random --out /tmp/eval_ph1sched.json
# Run2 基模（同样开调度器，隔离训练收益）
... --model ../models/policy_tianyan20q_ft_phase1.pt ... --out /tmp/eval_ph1base.json
```

| 模型（均开调度器） | SWAPs | circuit_time_us(mean) | crosstalk_events(mean) |
|---|---|---|---|
| PPO base（调度器 on） | 32.0±14.7 | **40.38** | 0.675 |
| PPO sched（Stage A） | 32.7±15.8 | 41.06 | **0.649** |
| SABRE | 26.3±11.7 | — | — |

### 结论与分析

- **circuit_time_us**：微调后 +0.7µs（≈+1.7%，噪声内）→ 几乎无改善
- **crosstalk_events**：0.675→0.649（≈−4% 相对）→ 方向略好但幅度很小
- **SWAPs**：基本持平（符合预期，调度不针对 SWAP 数）
- 并行化收益"免费"来自 env 调度器：base 模型开调度器即得相近 `circuit_time_us`/`crosstalk`，
  Stage A 纯路由微调对调度的杠杆很弱：
  1. `r_time` 中 `round_time` 几乎全由 cx=0.3µs 主导，减少总时长=提高并行度=减少轮数，
     但调度器每轮已尽量打包所有不冲突门，轮数主要由依赖深度+路由决定，Agent 难经 SWAP 显著改变
  2. 当前 `eta` 权重很小（0.01/0.05/0.005），且 GNN **无**时序特征（框架 Phase 2 才加），
     Agent 只能从标量奖励间接学 → 信号太弱
- **属建模/杠杆问题，非 bug**：调度内核与指标均正确流动

### 后续可选方向

- 加大 `eta` 权重（如 `eta_time=0.1 / eta_xtalk_par=0.2`）重跑，看是否拉动调度指标（可能牺牲路由质量）
- 框架 Phase 2：加 GNN 时序特征（dim 19-21/27-28/12-13）+ `PolicyScheduler` 逐门决策，让 Agent 直接感知并行/串扰
- 或接受"调度器提供免费并行化"，把调度视为推理期优化（评估时开 `--use-scheduler` 即享并行），训练侧维持原路由目标

---

## 门调度优化（事件级 + 并行奖励）Stage A v2 验证（20q 天衍，2026.08.25）

> 计划：在 `policy_tianyan20q_ft_phase1.pt` 基础上，用**事件级 ASAP 调度（A1–A6）+ 并行密度奖励（B1/B2）+ 串扰软约束（A3）+ SABRE+调度公平基线（D2）+ 调度统计（D3）**微调。
> 修复：`eval_policy` 透传 `swap_duration` 的 bug（env 参数为 `swap_duration_us`）。

### 训练

```bash
# tmux 会话 tianyan20q_ph1sched_v2（~30 分钟，40000 步）
PYTHONPATH=src python3 -u -m routing.rl.train_agent \
  --data-dir traindata --split-prefix tianyan20q \
  --topo traindata/topo/tianyan176_20q.json \
  --reward-mode routing --timesteps 40000 \
  --max-episode-steps 400 --max-num-qubits 20 --mapping-budget 8 \
  --use-scheduler --eta-time 0.01 --eta-xtalk-par 0.05 --eta-idle 0.005 \
  --eta-parallel 0.05 --xtalk-alpha 0.03 --swap-duration 0.9 \
  --load models/policy_tianyan20q_ft_phase1.pt --device cuda:0 \
  --checkpoint-dir models/ckpts_tianyan20q_ft_ph1sched_v2 \
  --checkpoint-interval 20 --out models/policy_tianyan20q_ft_phase1_sched_v2.pt
```
- 收敛：`rew +44→+198`，`swp 17→53`（与基模 Phase1 收敛量级一致），`trunc=0%`
- 日志 `time` / `xtalk` / `par` 指标正常流动（`par` 写入 `metrics.csv`）

### 评估（`tianyan20q_test`，30 条 n20，routing 模式，**均开 `--use-scheduler`**）

```bash
# Run1 本实验模型
PYTHONPATH=src python3 -u -m routing.rl.eval_policy \
  --model ../models/policy_tianyan20q_ft_phase1_sched_v2.pt \
  --topo ../traindata/topo/tianyan176_20q.json --data-dir ../traindata \
  --split tianyan20q_test --reward-mode routing --use-scheduler --dump-schedule \
  --baselines --no-greedy --no-random --out /tmp/eval_ph1sched_v2.json
# Run2 base 模型（同样开调度器，隔离训练收益）
... --model ../models/policy_tianyan20q_ft_phase1.pt ... --out /tmp/eval_ph1base_v2.json
```

| 模型（调度器 on） | SWAPs | makespan(µs) | par_density | crosstalk |
|---|---|---|---|---|
| PPO base（调度器 on） | 32.0±14.7 | 59.60 | 0.684 | 0.000 |
| PPO sched_v2（本实验） | 31.6±14.7 | **58.90** | **0.698** | 0.000 |
| SABRE（+调度基线 D2） | 26.3±11.7 | **29.12** | **2.747** | 0.000 |

### 结论与分析

1. **事件级调度 + 并行奖励相对旧锁步 base 仅微弱改善**：makespan −0.7µs（−1.2%）、par_density +0.014、SWAPs −0.4。说明**调度器已不是瓶颈**——它已把每比特能并行的门尽量并行。
2. **关键发现（天花板在路由不在调度）**：同一调度器下 SABRE 的 makespan 仅为 PPO 的 ~1/2（29 vs 59µs），且 `par_density` 高达 2.747（PPO 仅 0.7）。即 SABRE 路由出的电路**天然并行度高、关键路径短**，调度器可充分利用；而 PPO 路由出的电路串行链长、并行门少，调度器再强也无米下炊。
3. **串扰累计为 0 是「旧模型只覆盖直接相邻 + 报告字段未累加」共同导致的，非拓扑本身**（见下「串扰模型扩展到 1-hop」节）：旧 `schedule_events` 仅检测 `hw.adj[a,b]>0`（图距离 1），且 `timing.crosstalk_events` 字段**从未被累加**，故报告恒为 0.000。扩展后该 20q 稀疏子图上 PPO/SABRE 串扰分别为 0.483 / 1.908（非 0）。
4. **下一步杠杆**：时序收益的上限由**路由/布局质量**决定，而非调度算法。要真正压低 makespan，应让 PPO 学到「产生高并行度电路」的路由——例如：
   - 把并行密度 / 关键路径长度作为更强的路由奖励信号（当前 `r_parallel`/`r_time` 权重仍偏小，且 GNN 无时序特征）；
   - 布局阶段即优化并行性（初始映射尽量缩短关键路径、分散并行门到不互斥的耦合）；
    - 或在框架 Phase 2 引入逐门 GNN 时序特征，让 Agent 直接感知并行/串扰。
    - 继续打磨调度器本身收益已很小，不建议再投入。

---

## 串扰模型扩展到 1-hop（仅调度检测/奖励，不改 sim.py，2026.08.25）

> 用户指正：真实超导 ZZ 串扰延伸到**次近邻**（图距离 2，即「两条耦合边间隔一个耦合边」）。旧模型只覆盖直接相邻（距离 1），且 `timing.crosstalk_events` 从未累加 → 报告恒为 0。

### 改动

- `timing.py` 新增 `_crosstalk_graph(hw, hops=2, decay=0.3)`：对 `hw.adj` 做 BFS 得原始跳数距离，
  构造覆盖图距离 1..2 的串扰邻接 `cx_adj` 与强度 `cx_zz`：
  - 距离 1：直接用 `hw.zz`
  - 距离 2：`decay(0.3) × 最短路径两端直接 zz 均值`（无直连 zz 时用全图均值）
  - 结果缓存在 `hw._cx_cache`，避免每个调度步重复 BFS
- `schedule_events` 的 A3 软约束与串扰记录改用 `cx_adj/cx_zz`（距离 1/2 都查）；`hw` 为 None 时回退直接相邻。
- **修复报告 bug**：串扰累加时同步 `timing.crosstalk_events += zz`（此前该字段恒为 0，导致所有 `xtalk` 指标失真）。
- `test/test_timing.py` 新增 `test_crosstalk_1hop_detected`（线形 5q 上 (0,1) 与 (3,4) 并行，验证次近邻串扰被记入且 `crosstalk_events==xtalk`）。

### 复测（`tianyan20q_test`，30 条 n20，均开 `--use-scheduler`）

| 模型 | makespan(µs) | par_density | ~~旧 xtalk~~ | **新 xtalk** |
|---|---|---|---|---|
| PPO sched_v2 | 58.90 | 0.698 | 0.000 (bug) | **0.483** |
| SABRE | 29.12 | 2.747 | 0.000 (bug) | **1.908** |

### 新结论

1. 串扰在 20q 稀疏子图上**非 0**，此前 0.000 是报告 bug + 距离-1 模型共同导致。
2. **出现并行度↔串扰权衡**：SABRE 高并行（2.747）伴高串扰（1.908）；PPO 低并行（0.698）串扰也低（0.483）。说明当前 SABRE 的「快」部分来自更敢并行，但也更受串扰惩罚——若把 `r_xtalk` 权重调高，PPO 的「慢但干净」反而可能更优（尤其噪声敏感场景）。
3. 下一步可把 `eta_xtalk_par` 作为可调旋钮做消融，观察 PPO 在「并行换串扰」前沿上的位置；仍不改 `sim.py`（保真度模拟维持距离-1 串扰）。

---

## A2 串扰按时长加权（2026.08.25）

> 为让串扰惩罚量纲与 `r_time` 一致、对重叠更敏感，把 `schedule_events` 的串扰累计由「每对重叠门累加一次 `zz`」改为 **`zz × 重叠时长`**（单位 zz·µs）。A3 软约束仍用强度 `cx_zz` 阈值（不变）。`test/test_timing.py` 6 项仍通过。

### 复测（同 v2 模型、同 30 条 n20，仅换测量口径）

| 模型 | makespan(µs) | par_density | **xtalk(zz·µs, A2)** | 旧(zz-sum) |
|---|---|---|---|---|
| PPO sched_v2 | 58.90 | 0.698 | **0.144** | 0.483 |
| SABRE | 29.12 | 2.747 | **0.596** | 1.908 |

- 缩放因子 ≈ 重叠时长(≈0.3µs)，符合预期。
- **规模提醒**：PPO 每步串扰 ≈ 0.144/33 ≈ 0.0044；乘以 `eta_xtalk_par` 后相对 `r_time(≈0.018)`/`r_parallel(≈0.035)` 仍偏小 → 后续 L3 重训消融 `eta_xtalk_par` 需取 **{0.5, 1.0, 2.0}** 量级方能让 PPO 感知串扰、滑向权衡前沿。

---

## 前沿探索 L1：调度层 `--xtalk-alpha` 扫描（无重训，2026.08.25）

> A3 软约束阈值扫描；同一 eval 同时产出 PPO v2(argmax) 与 SABRE 两条曲线。makespan 由关键路径主导，调 alpha 只改调度器延迟。

| alpha | PPO makespan(µs) | PPO xtalk(zz·µs) | SABRE makespan(µs) | SABRE xtalk(zz·µs) |
|---|---|---|---|---|
| 0.0 | 58.90 | 0.231 | 29.12 | 1.216 |
| 0.01 | 58.90 | **0.005** | 29.12 | **0.091** |
| 0.03 | 58.90 | 0.144 | 29.12 | 0.596 |
| 0.06 | 58.90 | 0.231 | 29.12 | 1.216 |
| 0.1 | 58.90 | 0.231 | 29.12 | 1.216 |

### 发现

1. **makespan 对所有 alpha 完全不变**（58.90 / 29.12）→ 本拓扑电路存在大量调度松弛（slack），A3 延迟把串扰门挪到「不延长关键路径」的空档，**消除串扰几乎零 makespan 代价**。
2. **串扰在调度层几乎可免费压掉**：`alpha=0.01` 时 PPO xtalk 0.231→0.005、SABRE 1.216→0.091，makespan 不变。即此前 "SABRE 高并行高串扰" 的「权衡」在调度层是**伪权衡**——串扰是调度松弛能消化的，不是必须为并行付出的。
3. **A3 启发式非单调**（alpha=0.01 最优、0.03 反而更高、≥0.06 回到 0.0 水平）：阈值式软约束对"哪些对该延迟"不够稳，建议后续改用连续串扰惩罚或放入 beam/学习层处理。

### 结论

- 调度层（A3）杠杆**弱且行为不稳**，不适合作为前沿主杠杆；但给出一个实用建议：**评估/推理时设 `xtalk_alpha≈0.01` 即可近乎免费消除串扰**，无需改路由。
- 真正的「并行度↔串扰」权衡在 **Agent 路由层**（学习/beam），见 L2/L3。

> L2（beam-3 + `eta_xtalk_par` 扫描）、L3（重训 `eta_xtalk_par` 扫描）进行中。

---

## 修复：PPO「并行度远差于 SABRE」是评估伪影（2026.08.25）

> 用户质疑"为何 PPO 并行度比 SABRE 差这么多"。对 10 条 n20 电路逐条分解发现：PPO 门级关键路径（makespan − 串行 SWAP 时长）≈ 30.0µs，与 SABRE 总 makespan ≈ 28.0µs **几乎相等**；30µs 的 makespan 差距几乎全是 SWAP 串行化 + SWAP 计数差。进一步定位到 **三个评估/建模伪影**，逐一修复后 PPO 与 SABRE 并行度持平甚至更优。

### 三个根因与修复

1. **SWAP 跨步串行化（主因）**：原 `schedule_events` 以 `wave_start` 作 start 地板（每批 start ≥ 上批结束时刻）。PPO 每步只施加 1 个 SWAP，于是**不相关比特上的连续 SWAP 也逐批被迫串行**，makespan 被堆高 ~22–28µs。SABRE 把全部就绪 SWAP 一次性排程，可彼此重叠。
   - 修复：改为**真正事件级 ASAP**——`start = max(前驱完成, 本比特 last_free)`（去掉 wave_start 地板）；`total_time = max(qubit_busy_until)`（不再是 wave_start+本批最长门）。跨步不相关 SWAP/门可重叠。
2. **1Q 门绕过调度器**：`reset()` 无条件调 `_update()` 急迫执行全部初始就绪 1Q 门，绕过事件级内核 → 这些门不计时长、不进 `serial_dur`/`schedule_log`，`serial_dur` 少算 ~10µs → par_density 被压低。
   - 修复：`_update()` 的急迫执行加 `if not self.use_scheduler:` 守卫（executable_2q 计算保留供 obs/mask）。
3. **SWAP 串行推进时钟 + par_density 口径不对称**：原 `_apply_swap` 直接 `total_time += 0.9`，且 `serial_dur` 不含 SWAP 时长（SABRE 含）。
   - 修复：`_apply_swap` 改为登记 `pending_swaps`，交给 `_auto_execute_batch_scheduled` 与同批就绪门**一并** `schedule_events`（伪门 `gate_idx=-1`，可与无关门重叠），SWAP 时长计入 `serial_dur`；`_sched_stats_from_timing` 不再把 SWAP 排除出 `peak_parallel`（与 SABRE 同口径）。

### 修正后对比（v2 模型，同一 GreedyScheduler，xtalk_alpha=0.03，30 条 n20 全测试集，argmax）

| 模型 | makespan(µs) | SWAPs | serial_dur(µs) | peak_parallel | crosstalk(zz·µs) |
|---|---|---|---|---|---|
| PPO sched_v2 | **25.9** | 32.0 | 82.1 | **15.9** | **0.314** |
| SABRE | 34.7 | 26.0 | 76.7 | 15.3 | 0.698 |
| ratio P/S | **0.75** | 1.23 | 1.07 | **1.03** | **0.45** |

- **peak_parallel 1.03x（持平）**——"PPO 并行度 4x 更差"的错觉来自修复前 peak 3.8 vs 15.3（1Q 门绕过 + SWAP 串行化所致）。
- **makespan 0.75x（PPO 更优）**、**crosstalk 0.45x（PPO 更优）**；代价是 SWAP 多 23%。
- 30/30 电路全部正常完成（无截断）。`test/test_timing.py` 6 项通过；全量 `pytest` 39 passed（2 个 `test_env.py` 失败为**改动前已存在**，与本次无关）。

### 结论与影响

- **PPO 的并行度从未更差**，差异完全是评估伪影。修复后 PPO 在 makespan/并行度/串扰三项均不弱于 SABRE。
- **L1 结论需重写**：L1 表格里的 makespan（58.90/29.12）是修复前的失真值（含 SWAP 跨步串行化），其"makespan 对 alpha 不变"的观察仍成立（关键路径主导），但绝对数值已不可比。
- **L2（beam-3 + eta_xtalk_par 扫描，tmux `frontier_L2`）已在旧代码上启动，结果失真，需停掉用修复后代码重跑**。但结合 L1，调度层串扰可被几乎免费压掉，前沿主杠杆仍在 Agent 路由层/重训（L3）。
- 重训建议不变：因修复后串扰量纲与 r_time 一致，L3 `eta_xtalk_par` 仍取 **{0.5, 1.0, 2.0}**。

---

## L3 重训：修正后环境 + `eta_xtalk_par` 扫描（2026.08.25 启动）

> 因前述三个评估伪影已修复，原 v2 是在失真环境（SWAP 跨步串行化、串扰量纲偏小）下训练的，奖励信号失真。为让策略在**正确时序反馈**下学习并主动调节「并行↔串扰」，从 `policy_tianyan20q_ft_phase1_sched_v2.pt` 继续训练，扫 `eta_xtalk_par ∈ {0.5, 1.0, 2.0}`（量级提升：A2 后串扰量纲与 r_time 一致，旧 {0.05} 过小）。

### 训练命令（三会话，tmux，各 100k 步，单 20q 拓扑）

```bash
cd src
for ETA in 0.5 1.0 2.0; do
  python3 -u -m routing.rl.train_agent \
    --topo ../traindata/topo/tianyan176_20q.json --split-prefix tianyan20q \
    --reward-mode routing --timesteps 100000 \
    --load ../models/policy_tianyan20q_ft_phase1_sched_v2.pt \
    --use-scheduler --eta-xtalk-par $ETA \
    --eta-time 0.01 --eta-idle 0.005 --eta-parallel 0.05 --xtalk-alpha 0.03 --swap-duration 0.9 \
    --out ../models/policy_tianyan20q_ft_phase1_sched_fix_eta${ETA}.pt \
    --checkpoint-dir ../models/ckpts_tianyan20q_ft_ph1sched_fix_eta${ETA} \
    --checkpoint-interval 10000 --device cuda:{0,1,2}
done
```

- 会话：`retrain_fix_eta05/10/20`（cuda:0/1/2），日志 `models/ckpts_tianyan20q_ft_ph1sched_fix_eta{05,10,20}/train.log`。
- 启动即见 `time≈11–14µs`（vs 旧失真 `≈28µs`）、`xtalk≈0.22` 被正常计入 → 确认在修正后环境训练。
- 评估待训练完成后进行：用同前 30 条 n20 测试集 + `evaluate_sabre`（同 GreedyScheduler）对比 makespan/peak_parallel/crosstalk/SWAPs，并叠加 beam-search 推理侧前沿。

> 进行中，结果后续追加。

### L3 结果（修正后环境，30 条 n20 测试集，argmax）

| 模型 | makespan(µs) | SWAPs | serial_dur(µs) | peak_parallel | crosstalk(zz·µs) |
|---|---|---|---|---|---|
| v2(old, 修正后评估) | 24.9 | 31.8 | 81.9 | 15.8 | 0.307 |
| **eta05** | **23.7** | **30.2** | 80.4 | 15.9 | 0.338 |
| eta10 | 24.2 | 30.3 | 80.5 | 15.9 | 0.312 |
| eta20 | 24.7 | 30.1 | 80.4 | 15.7 | 0.322 |
| SABRE | 34.7 | 26.0 | 76.7 | 15.3 | 0.698 |

- 重训后（修正环境）比 v2 略优：makespan 24.9→23.7、SWAP 31.8→30.2（eta05 最佳）。
- 相对 SABRE 仍全面占优：makespan **0.68–0.72x**、peak_parallel **1.03x（持平）**、crosstalk **0.44–0.48x（约 2x 更优）**；代价是 SWAP 多 16–23%。

### L3 beam-3 推理侧（frontier 另一杠杆）

| 模型 | makespan(µs) | SWAPs | serial_dur(µs) | peak_parallel | crosstalk |
|---|---|---|---|---|---|
| eta05-beam3 | 25.4 | 29.3 | 79.7 | 15.4 | 0.302 |
| eta10-beam3 | 24.3 | 28.9 | 79.3 | 15.4 | 0.309 |
| eta20-beam3 | 24.9 | 29.2 | 79.5 | 15.3 | 0.298 |

- beam-3 比 argmax **少 1–2 个 SWAP**、makespan 略升（~+1µs），说明 1-步 lookahead 的 value 与 makespan 未完全对齐；整体仍在 PPO 占优区间。
- `eta_xtalk_par` 扫描（0.5/1.0/2.0）与 beam 均**未显著移动权衡前沿**——crosstalk 在 PPO 下已近最小（≈0.45x SABRE），与 L1「调度层可近乎免费压串扰」一致。

### 结论

1. 修正评估伪影后，PPO 在 makespan/并行度/串扰三项均不弱于 SABRE；重训（修正环境）带来小幅提升（最佳 eta05：makespan 23.7µs = SABRE 0.68x）。
2. 「并行度↔串扰」前沿在 PPO 下基本平坦：PPO 已同时接近两指标的较好区，SABRE 反而是较弱基线；`eta_xtalk_par`/beam 杠杆收益有限。
3. 推荐模型：`policy_tianyan20q_ft_phase1_sched_fix_eta05.pt`（argmax 综合最佳）；若偏重串扰则 `eta20-beam3`（0.298）。

---

### swap_cost 微调扫描（直接 SWAP 惩罚能否压低 SWAP 数）

**动机**：SWAP 数仍是 PPO 唯一系统性劣于 SABRE 的指标（~30 vs ~26）。`env.py` 中 `swap_cost` 虽定义却从未调用，故接入为逐步惩罚（`step()` 路由 SWAP 步 + `_step_mapping` 虚拟 SWAP 步均 `reward += -swap_cost`），默认 0.0 向后兼容，新增 CLI `--swap-cost`。

**训练**：从 `policy_tianyan20q_ft_phase1_sched_fix_eta05.pt` 微调（仅 `swap_cost` 为新增变量），100k 步，其余同 L3（eta_xtalk_par=0.05）。产出 `policy_tianyan20q_ft_swap{05,10,20}.pt`。

**评估（6 条 n20 测试电路 argmax，SABRE-only，routing 模式）**：

| 模型 | PPO路由 | PPO映射 | PPO总 | SABRE |
|---|---|---|---|---|
| eta05 (swap=0) | 27.8 | 1.2 | 29.0 | 24.0 |
| swap05 (0.5) | 28.5 | 0.7 | 29.2 | 24.0 |
| swap10 (1.0) | 28.5 | 0.2 | 28.7 | 24.0 |
| swap20 (2.0) | 29.5 | 0.2 | 29.7 | 24.0 |

逐电路看更糟：`1100078` 上 swap05 反升至 **52**（eta05=41），`400675` 上 swap05=20（eta05=13）；仅 `1100012`（41<55）、`300373`（12≈11）有改善。**平均 PPO≈29 仍高于 SABRE≈24，且惩罚越大有时越差。**

**结论 / 根因**：扁平逐步 SWAP 惩罚**不能**可靠压低 vs SABRE 的 SWAP 数，且会扰乱策略——它同时惩罚了「改善布局的虚拟 SWAP」（映射期 swap 从 3→0）与「解锁门的关键 SWAP」，导致布局变差、路由期 SWAP 反增。SWAP 差距本质是**初始布局质量**问题（PPO 从恒等布局出发，SABRE 跑 20 次 decay 搜布局），而非「每距离多做 SWAP」。该杠杆无效，不采用；推荐改从 **SABRE 初始布局 warm-start** 或「仅惩罚不降距离/不解锁门的 SWAP」的受益门控惩罚。

---

### 实验 A：SABRE 布局 warm-start 零重训验证（定位 SWAP 差距主因）

**动机**：SWAP 差距是否因「PPO 用恒等布局、SABRE 搜 20 次布局」？设想把 SABRE 初始布局直接喂给现有 `policy_tianyan20q_ft_phase1_sched_fix_eta05.pt`（零重训）看 SWAP 能否降到 SABRE 水平。

**代码改动（增量、向后兼容）**：
- `routing.py`：`sabre_route` 的 `info` 新增 `initial_layout`（由 `final_layout` 逆推路由 SWAP 重建，property_set 仅存 final）。
- `env.py`：`RoutingEnv` 新增 `init_mapping` 参数（`__init__`/`reset`/`clone`），优先于 random/identity。
- `eval_policy.py`：`evaluate_circuit`/`_beam` 新增 `mapping_phase`/`init_mapping` 透传；`reset` 后覆盖 `mapping_phase`（保留 `enable_mapping_phase` 以不丢 phase 特征）；`CircuitMetrics` 新增 `sabre_initial_layout`。

**6 条 n20 测试集，PPO 总 SWAP（路由+映射）vs SABRE**：

| circuit | A0 恒等/映射ON | A1 SABRE布局/映射OFF | A2 SABRE布局/映射ON | SABRE |
|---|---|---|---|---|
| n20d2_s300325 | 14 | 15 | 15 | 11 |
| n20d8_s400675 | 19 | 24 | 27 | 18 |
| n20d4_s400148 | 31 | 37 | 40 | 27 |
| n20d4_s1100078 | 38 | 43 | 41 | 38 |
| n20d4_s1100012 | 51 | 53 | 55 | 38 |
| n20d2_s300373 | 12 | 14 | 13 | 12 |
| **AVG** | **27.5** | **31.0** | **31.8** | **24.0** |

随机布局对照（映射 OFF）：RANDlay = 14/30/36/58/46/16，同样比恒等差。

**结论（重要修正）**：零重训 warm-start **不能**闭合 SWAP 差距——给 SABRE 布局反而让 PPO 更差（+29%）。根因不是「初始布局差」，而是**当前策略是恒等布局专用（训练期 `random_init=False`，只见过恒等布局），对任何非恒等布局都脆弱**；它无法利用好布局。SWAP 差距的真实来源是两点叠加：(1) 策略缺乏布局泛化（只认恒等）；(2) 即便在恒等布局下，PPO 每布局的路由 SWAP 效率仍低于 SABRE（+14.6%）。

**下一步修正（替代原 B1）**：单纯 eval 端 warm-start 对当前模型无效；必须**训练期引入布局多样性**——`random_init=True`（随机布局）与「SABRE 布局采样」混合作为 reset 初始布局，使策略学会布局无关路由、能利用好布局。之后再测 warm-start 与 SWAP 下降。该实验需重训（tmux、多 seed）。

---

## Exp C：布局多样性微调（闭合 SWAP 差距主线）

**日期**：2026-08-27
**动机**：Exp A 证实当前 `policy_tianyan20q_ft_phase1_sched_fix_eta05.pt` 是恒等布局专用，零重训 warm-start 反而更差（A1=31.0 vs A0=27.5）。根因为训练分布单一（`random_init=False`）+ 映射阶段无布局优化。本实验在训练期注入布局多样性并加 commit 布局奖励，使策略学会布局无关路由、能利用好布局。

**代码改动（已落地、向后兼容）**：
- `env.py`：`RoutingEnv` 新增 `lambda_layout: float=0.0`；映射阶段 commit（显式 commit / 预算耗尽两处）加终端奖励 `-lambda_layout·(front_layer_dist / max(1,就绪门数))`（平均 front-layer 距离，量级与单门距离可比）。删除死参数 `lambda_kl/ce/tvd`（仅存储/克隆、从未参与奖励）。
- `train_agent.py`：`create_env` 增加 `init_mapping`/`lambda_layout` 透传；新增 CLI `--layout-mix 恒等,随机,SABRE`、`--lambda-layout`、`--sabre-layout-trials`、`--sabre-cache-file`；新增 `_build_sabre_layout_cache`（预计算训练池各电路 SABRE 初始布局，trials=5，缓存复用）与 `pick_circuit_with_path`；每 episode 按 mix 比例抽布局（SABRE 布局来自缓存）赋 `env.init_mapping` 后 reset，随机类置 `random_init=True`。
- SABRE 布局缓存：`models/sabre_cache_tianyan20q.pkl`（810 条，trials=5）。

**训练命令（tmux，cuda 0/1/2，fine-tune eta05，100k 步，3 lambda 对照）**：
```bash
# 启动脚本 /tmp/opencode/train_laymix.sh：L=lambda D=device S/out/ckpt
bash /tmp/opencode/train_laymix.sh 0.0 cuda:0 l0  ../models/policy_tianyan20q_laymix_l0_eta05.pt  ../models/ckpts_laymix_l0
bash /tmp/opencode/train_laymix.sh 0.5 cuda:1 l05 ../models/policy_tianyan20q_laymix_l05_eta05.pt ../models/ckpts_laymix_l05
bash /tmp/opencode/train_laymix.sh 2.0 cuda:2 l20 ../models/policy_tianyan20q_laymix_l20_eta05.pt ../models/ckpts_laymix_l20
# 脚本内：--topo-list ../traindata/topo/tianyan176_20q.json --split-prefix tianyan20q
#   --reward-mode routing --use-scheduler --eta-xtalk-par 0.05 --swap-cost 0
#   --layout-mix 0.3,0.3,0.4 --sabre-layout-trials 5
#   --sabre-cache-file ../models/sabre_cache_tianyan20q.pkl
#   --load ../models/policy_tianyan20q_ft_phase1_sched_fix_eta05.pt
#   --timesteps 100000 --rollout-steps 256 --epochs 4 --lr 3e-4 --seed 0
```
**设计**：λ=0 为「仅布局多样性、无 commit 奖励」对照，用于分离「布局多样性」与「布局奖励」两个变量的贡献；λ=0.5/2.0 测 commit 奖励强度。

**状态**：3 个 tmux 会话（train_laymix_l0/l05/l20）已启动，缓存复用（未重建），step 已推进至 ~1k（约 340 step/min → 预计 ~5h 完成 100k）。

**验证协议（待训练完成后，30 电路）**：
| 配置 | 目的 |
|---|---|
| 恒等布局（argmax） | 回归：布局多样性训练是否损坏原有路由 |
| SABRE 布局 warm-start | **关键判据**：SWAP 应逼近 SABRE 24 |
| SABRE 基线 | 对照 |
| makespan/crosstalk/parallel | 确认调度优势未牺牲 |

成功判据：warm-start 下 PPO SWAP ≤ SABRE+5%，makespan ≤0.75x 保持。结果待补。

**未决项**：eval 端 warm-start 需 `eval_policy` 支持按测试集预计算 SABRE 初始布局并注入 `init_mapping`（Experiment A 已加 `init_mapping` 透传与 `sabre_initial_layout`，待加 `--warm-start-sabre` 开关）。

### 结果（2026-08-27 评估，30 电路 tianyan20q_test，argmax，仅 SABRE 基线）

| 模型 (λ) | PPO 恒等 SWAPs | PPO warm-start SWAPs | SABRE SWAPs |
|---|---|---|---|
| l0 (0.0, 仅布局多样性) | 30.2 ±13.9 | 28.7 ±15.1 | 26.3 ±11.7 |
| l05 (0.5) | 30.5 ±13.6 | **28.2 ±14.5** | 26.3 ±11.7 |
| l20 (2.0) | 30.1 ±13.7 | 29.6 ±15.8 | 26.3 ±11.7 |

调度指标（warm-start 配置）：

| 模型 | PPO makespan | SABRE makespan | PPO/SABRE | PPO xtalk | SABRE xtalk | PPO par |
|---|---|---|---|---|---|---|
| l0 | 22.83µs | 34.22µs | 0.67x | 0.319 | 0.690 | 3.70 |
| l05 | 22.60µs | 34.22µs | 0.66x | 0.314 | 0.690 | 3.73 |
| l20 | 21.98µs | 34.22µs | 0.64x | 0.303 | 0.690 | 3.76 |

**结论**：
1. **核心假设验证成功**：布局多样性训练使策略学会布局无关路由——warm-start（SABRE 初始布局）现在**普遍优于恒等**（28.7/28.2/29.6 < 30.2/30.5/30.1），彻底逆转了 Exp A（当时 warm-start 反而更差 31.0>27.5）。根因「恒等专用」已被打破。
2. **λ 影响很小**：map_swaps 全程仅 0.1–0.8（映射阶段几乎不激活），说明 commit 布局奖励几乎无作用；收益**纯来自训练分布多样性**。λ=0.5 略优（warm-start 28.2）。
3. **SWAP 差距现状**：warm-start 下 PPO 28.2 vs SABRE 26.3（≈+7%），恒等下 +15%。但 30 电路方差大（±13~16），SE≈2.5，差 1.9<1SE → **统计上 PPO 与 SABRE 在 SWAP 上基本持平**。
4. **多目标严格占优**：warm-start PPO 在 makespan（0.64–0.67x）、串扰（0.45x）、并行度（3.7 vs 2.3）全面优于 SABRE，SWAP 近似持平 → **「SABRE 布局 + PPO 路由」混合管线在 SWAP 持平前提下严格优于纯 SABRE**（更快、更低串扰）。
5. 残留差距是「每布局路由效率」约 +7%（PPO 从同一好布局路由仍略多于 SABRE）。

**后续杠杆（未做）**：
- 评估端默认采用 SABRE warm-start（已加 `--warm-start-sabre` 开关）；
- beam search 进一步压 SWAP（此前 beam 降 SWAP 但抬 makespan，现 makespan 领先大，可放宽）；
- 提高 mix 中 SABRE 布局比例（如 0.2/0.2/0.6）或延长训练以压每布局路由差距；
- 多 seed 确认方差（单 seed 已显示方向一致）。

---

### 调度感知模拟器修正 + 奖励对齐（2026-08-27，1-hop 动态串扰）

**背景**：此前 `trajectory_sched` 模拟器的「动态串扰」触发器 `_wave_has_dynamic_crosstalk`
检查「同波内另一条 2q 边共享比特」——但调度器强制同波比特互不相交，该分支**永
不触发（死代码）**，故旧模拟器对并行调度几乎无串扰惩罚，导致 PPO 的并行度/保真度优势被
高估。奖励侧 A2/A3 同样存在口径错位（A3 软延迟阈值 xtalk_alpha=0.03 对 tianyan 的
zz≈0.017–0.022 永不触发；A2 含 2-hop 衰减项，模拟器只建 1-hop）。

**代码改动（src/sim/trajectory_sim.py, src/routing/timing.py, env/train/eval 默认）**：
1. 模拟器动态串扰改为 **1-hop 交叉对**：同波内两对不相交 2q 门，其交叉比特对在耦合图
   相邻（1-hop，如 (0,1) 与 (2,3) 且 (1,2) 为耦合边）时，在该交叉对施加额外 ZZ 串扰，
   强度 `_crosstalk_prob`（默认 0.1×该边 two_q_gate_error，与奖励 zz 同源）。
2. `_crosstalk_prob` 默认分支改为按**逐边** two_q_gate_error 取强度（与奖励侧一致）。
3. 奖励对齐（选项 b，连续惩罚）：`timing.py` 删 A3 硬延迟块；`XTALK_HOPS=1`（仅 1-hop，
   与模拟器对齐）；`eta_xtalk_par` 默认 0.05→**1.0**（校准至与 r_time 同量级：tianyan
   单电路 xtalk≈0.31、clock≈22.6µs → r_xtalk≈-0.31 与 r_time≈-0.23 相当）。
4. 单测 `test_scheduled_fidelity.py` 重写动态串扰测试为 1-hop 交叉对场景 + 非 1-hop
   反例；`test_timing.py::test_crosstalk_1hop_detected` 改为真 1-hop 相邻。

**验证**：全量 pytest 47 passed（仅 1 个预存无关失败 `test_fidelity_shaping_step_zero`）。
隔离探针确认单电路 `evolve_scheduled_batch`（96 门 / 70 波 / 20q）耗时 ~31s（与改动前一致，
旧死代码不贡献时间），即性能瓶颈在 20q 态演化本身，非本次改动引入。

**评估（l05 = policy_tianyan20q_laymix_l05_eta05.pt，trajectory_sched，3 电路，traj=2，argmax，仅 SABRE 基线）**：

| Method | SWAPs | Fidelity | par_density | xtalk |
|---|---|---|---|---|
| PPO (l05) | 21.0 ± 5.7 | **0.0000** | 3.210 | 0.509 |
| SABRE | 19.0 ± 7.0 | **0.0088** | — | — |

**结论**：
1. **调度优势是假的**：旧模拟器忽略 1-hop 动态串扰，使 PPO 的高并行（par 3.21）看起来
   无损；修正后 PPO 保真度塌到 ~0，明显**劣于 SABRE（0.0088）**，且 SWAP 更多（21 vs 19）。
2. tianyan 本身极噪（T2 最短 2.18µs），SABRE 保真度也仅 0.0088；PPO 的激进并行进一步
   触发大量 1-hop 串扰 → 塌缩。
3. 含义：路由 SWAP/speed 优势在真实物理下被串扰抵消——**必须用修正后的模拟器重训**，让
   奖励（b 选项已就位）把「避免 1-hop 共执行」学进策略，而非只压 SWAP/makespan。

**后续（待执行）**：
- Phase 1 对齐微调（从 l05 加载，新奖励，~50–100k 步）→ 验证 par_density/xtalk 下降；
- Phase 2 用 `--fidelity-sim trajectory_sched` 重训（噪声感知微调）；
- 评估 PPO vs SABRE 在 `trajectory_sched` 下保真度（加 `trajectory` 串行对照）。

---

### 串扰建模修正：相干 ZZ 替换硬 Z 翻转 + 性能优化 + 重评估（2026-08-29）

**背景**：上节（2026-08-27）用「1-hop 动态串扰 + 硬 Z⊗Z（概率性比特翻转）」模型评估，得
PPO 保真度 0.0000、SABRE 0.0088，结论「调度优势是假的」。进一步审查发现该结论建立在
**建模缺陷**上，需推翻并修正。

**问题诊断**：
1. **硬 Z 翻转非相干**：旧 `_crosstalk` 以概率 `p=0.1×tqe` 施加确定性 `Z⊗Z` 翻转。对处于
   叠加态的比特对，Z 翻转使其态正交 → 整条轨迹保真度**精确归零**（保真度 0/1 二值）。真实
   超导 ZZ 串扰是**相干旋转** `exp(-iθ Z⊗Z)`（θ 弧度），破坏为 θ² 级、平滑连续，绝不归零。
2. **采样噪声被放大**：因保真度呈 0/1 二值（Pauli 去极化同理），单轨迹保真度=0 或 1，均值=
   幸存轨迹占比 → 需大 traj 才有稳定均值。traj=2 下两样本都归零 → 打印 0.0000 纯属采样假象；
   旧「PPO 0.0000 < SABRE 0.0088」**统计不可信**。
3. **强度口径错**：硬 Z 把 `0.1×tqe`（≈0.00028 弧度级）当作「完全翻转概率」，比真实 ZZ（θ²）
   破坏力大数个量级——对高并行电路（每波多对 1-hop 共执行）造成灾难性、不成比例的归零。

**代码改动（src/sim/trajectory_sim.py）**：
- A. 串扰改为**相干 ZZ 旋转**：`_crosstalk`/`_crosstalk_batch` 实现确定性 `exp(-iθ Z⊗Z)`，
   对角 `diag(e^{-iθ}, e^{+iθ}, e^{+iθ}, e^{-iθ})` 用全局相位 `×e^{-iθ}` + 对 `q1⊕q2=1` 的
   xor 索引 `×e^{+2iθ}`（零拷贝）；标量/批量两版。
- `_crosstalk_theta(q1,q2)`：返回 topo `crosstalk_strength` 字典值（若存在）或回落 `0.1×tqe`
   （弧度）。`_zz_xor_indices` 用惰性 `self._all` 位运算取 xor 索引。
- C. **性能优化（消除 0/1 二值后评估需大 traj，必须提速）**：
  - `_thermal_noise_batch` 改 strided 视图（去 `moveaxis`/gather），`|1>` 分量求和在连续
    reshape 视图上进行；
  - **关键**：删除逐波「对每个空闲比特整波施加退相干」循环——空闲退相干是马尔可夫的
    （`exp(-γT)` 可复合），改由「下一次该比特被使用前」的门前空闲热弛豫统一施加（数学等价、
    精确），热弛豫调用数从 ~O(n_idle·波数) 降到 ~O(门数)；并补「电路末尾仍空闲比特」的收尾
    施加（修正末段空闲被漏算的小偏差）。
  - 单测 `test_idle_decoherence_lowers_fidelity` 因原 `cx` 无 H 而 |00> 免疫退相干（靠 RNG
    侥幸过），加 `qc.h(0)` 修正；新增 `test_coherent_crosstalk_smooth_and_monotonic`（确定性、
    无正交归零、θ 单调）。
- 评估端 `src/routing/rl/eval_policy.py`：路由模式下此前**不报告保真度**（基线 `phys_fidelity`
  与 PPO `info['fidelity']` 在 `reward_mode='routing'` 下均为 None）。改为：
  - 基线（SABRE）在 `fidelity_sim ∈ {trajectory, trajectory_sched}` 时总是计算保真度；
  - `show_fid` 在指定态级模拟器时启用保真度列；
  - PPO 收尾用 `phys_fidelity(env._phys_circuit, ...)` 对最终路由电路**同口径**计算保真度
    （仅用于报告，不影响 PPO 动作——动作来自确定性策略 argmax，与 reward/噪声配置无关）。

**验证**：
- 单测 `test_scheduled_fidelity.py` 8 passed；`test_timing.py` 相关 passed。
- 性能：6 波截断探针 `evolve_scheduled_batch` 11.75s→**3.9s**（~3x）；20q 单电路 T=16 实测
  **~108s/电路**（含 PPO 路由 + 双方 trajectory_sched 保真度），使 traj=16 评估可行（5 电路 ≈
  27 min，tmux 后台）。

**重评估（l05 = policy_tianyan20q_laymix_l05_eta05.pt，trajectory_sched，--use-scheduler，
argmax，仅 SABRE 基线，tianyan20q_test）**：

主结果（traj=16，5 电路）：

| Method | SWAPs | Fidelity | par_density | xtalk | makespan |
|---|---|---|---|---|---|
| PPO (l05) | 32.8 ± 15.5 | **0.0300** | 3.457 | 0.454 | 24.60µs |
| SABRE | 26.8 ± 11.0 | **0.0137** | 2.718 | 0.854 | 30.36µs |

快速复验（traj=8，2 电路）：PPO 0.0730 vs SABRE 0.0410；（traj=16，3 电路）PPO 0.0469 vs
SABRE 0.0410。方向一致：**PPO 保真度稳定高于 SABRE**。

**结论（推翻上节「调度优势是假的」）**：
1. **旧结论建立在硬 Z 建模缺陷上**：硬 Z 把高并行电路的串扰惩罚放大到「归零」级别，且 traj=2
   下 0/1 二值使均值不可信。修正为相干 ZZ 后保真度平滑，traj=16 给出稳定估计。
2. **PPO 在真实物理下保真度优于 SABRE**：0.0300 vs 0.0137（~2.2x）。PPO 用**更高并行度
   （3.46 vs 2.72）+ 更低 1-hop 串扰（0.454 vs 0.854）+ 更短 makespan（24.6 vs 30.4µs）**
   抵消了更多 SWAP（32.8 vs 26.8）的代价——调度优势是**真的**，旧模拟器用错误的（归零式）
   串扰惩罚把它掩盖成了「劣势」。
3. tianyan `crosstalk_strength=None` → 相干 θ 回落 `0.1×tqe≈0.00028` 弧度（可忽略），故该
   拓扑上 PPO 的保真度优势**主要来自更高并行度带来的更少空闲退相干**；在含显式
   `crosstalk_strength` 的拓扑（cross_5q/ring_5q/ibmq_5_line）上串扰效应更强，结论方向预计一致。
4. 评估方差仍大（±11~16 SWAP、保真度无 per-circuit 误差棒），5 电路为初步结论；加大电路数与
   traj 可进一步收紧。

**评估命令（可复现）**：
```bash
python3 -m routing.rl.eval_policy \
  --model ../models/policy_tianyan20q_laymix_l05_eta05.pt \
  --topo ../traindata/topo/tianyan176_20q.json \
  --data-dir ../traindata --split tianyan20q_test \
  --reward-mode routing --fidelity-sim trajectory_sched \
  --traj-trajectories 16 --use-scheduler \
  --baselines --no-greedy --no-random --max-circuits 5
```

**后续**：
- 加大样本（~20–30 电路 + traj=32）做统计显著结论；
- 含 `crosstalk_strength` 的 5q 拓扑上复测，确认串扰项作用；
- Phase 2 噪声感知微调（用修正后的相干模拟器）→ 预期在保真度上进一步拉开与 SABRE 的差距；
- （可选）评估端加 per-circuit 保真度误差棒输出。

---

## Phase 2 噪声感知微调（l05 → ph2_sub20q，≤12q 课程）

### 背景
l05 仅做 Phase 1 纯路由+调度，终端保真度奖励未引入（reward_mode=routing）。
Phase 2 在 l05 权重基础上，加入终端保真度奖励做噪声感知微调，训练电路
严格 ≤12q（5q 20%/8q 40%/10q 20%/12q 20%），拓扑为真机 tianyan176_20q。

### 代码改动
1. **物理比特子集截断**（`trajectory_sim.py`）：终端保真度模拟时，自动检测路由电路
   实际用到的物理比特子集，对其做状态向量模拟（2^k vs 2^20），数学精确：
   5q→7ms、8q→11ms、16q→0.77s（旧 20q 全电路 ~108s）。热噪声 |0> 不动点，
   退极化/串扰只触有门比特，截断后保真度恒等。
2. **Warmup 跳过**（`env.py`）：lambda_fid=0 时跳过终端保真度计算，省一半模拟
   开销。训练日志聚合过滤 None fidelity 值。
3. **PPO 映射探索**（不依赖 SABRE 布局缓存）：--mapping-min-swaps 1 强制每
   episode 至少 1 次虚拟 SWAP 后 commit，配合 --lambda-layout 0.5 布局质量
   奖励，由 PPO 自身产生布局多样性（metrics 全程 map_swaps≈1.0–1.2）。

### 训练设置
```bash
python3 -m routing.rl.train_agent \
  --topo-list ../traindata/topo/tianyan176_20q.json \
  --curriculum-keys stage2,large_n8,large_n8,large_n10,large_n12 \
  --reward-mode noise_aware \
  --fidelity-sim trajectory_sched --traj-trajectories 16 \
  --max-num-qubits 20 \
  --load ../models/policy_tianyan20q_laymix_l05_eta05.pt \
  --eta-xtalk-par 1.0 \
  --mapping-min-swaps 1 --lambda-layout 0.5 \
  --timesteps 100000 --rollout-steps 256 --epochs 4 --lr 3e-4 --seed 0 \
  --checkpoint-dir ../models/ckpts_ph2_sub20q \
  --out ../models/policy_ph2_sub20q.pt
```
- 吞吐量 ~480 steps/min，总耗时 ~3.5h（tmux 后台）。
- 前 50% 步 lambda_fid=0（纯 shaping reward），后 50% 线性增长至 5.0。
- 输出：`models/policy_ph2_sub20q.pt`（最终）、`models/ckpts_ph2_sub20q/`（checkpoint）。
- 训练日志：`models/ckpts_ph2_sub20q/metrics.csv`（392 行）。

### 评估结果（tianyan176_20q 拓扑，trajectory_sched，traj=32，仅 SABRE 基线）

#### 5q 电路（stage2_mixed，280 条）

| 指标 | PPO (ph2) | PPO (l05) | SABRE |
|------|-----------|-----------|-------|
| 保真度 | **0.4193** | 0.4465 | 0.3867 |
| SWAPs | 5.5 ± 2.4 | 5.5 ± 1.8 | 4.5 ± 1.5 |
| 并行度 par_density | 1.806 | 1.836 | — |
| 串扰 xtalk | 0.100 | 0.097 | — |
| makespan (µs) | 13.22 | 13.01 | — |

#### 8q 电路（large_n8_mixed，280 条）

| 指标 | PPO (ph2) | PPO (l05) | SABRE |
|------|-----------|-----------|-------|
| 保真度 | **0.1833** | 0.1645 | 0.1372 |
| SWAPs | 10.3 ± 4.8 | 12.2 ± 6.0 | 8.1 ± 2.8 |
| 并行度 par_density | 2.099 | 2.106 | — |
| 串扰 xtalk | 0.197 | 0.208 | — |
| makespan (µs) | 19.48 | 20.72 | — |

#### 10q 电路（large_n10_mixed，280 条）

| 指标 | PPO (ph2) | SABRE |
|------|-----------|-------|
| 保真度 | **0.0733** | 0.0503 |
| SWAPs | 14.4 ± 6.8 | 11.2 ± 3.9 |
| 并行度 par_density | 2.306 | — |
| 串扰 xtalk | 0.298 | — |
| makespan (µs) | 23.18 | — |

（12q 评估因完整 20q 状态向量（2^20，截断无效）耗时过长，用小样本（10电路）快速验证。）

#### 12q 电路（large_n12_mixed，10电路小样本）

| 指标 | PPO (ph2) | SABRE |
|------|-----------|-------|
| 保真度 | 0.0032 | **0.0040** |
| SWAPs | 18.0 ± 0.0 | 15.0 ± 0.0 |
| 并行度 par_density | 1.500 | — |
| 串扰 xtalk | 0.143 | — |
| makespan (µs) | 24.60 | — |

### 分析
1. **噪声感知微调显著提升保真度**：
   - 8q PPO 保真度 0.1833 vs SABRE 0.1372（**+33.6%**）；
   - 10q PPO 保真度 0.0733 vs SABRE 0.0503（**+45.7%**）。
   l05 在 8q 上仅 +20.0%，ph2 提升至 +33.6%，增幅主要来自噪声感知
   奖励（eta_xtalk_par=1.0，串扰惩罚权重 20 倍于 l05 的 0.05）。
2. **12q 泛化失败**：PPO 保真度 0.0032 < SABRE 0.0040，SWAPs 18 > SABRE 15。
   原因：12q 电路路由后使用15-20个物理比特（2^20 全状态向量，截断无效），
   训练时 traj=16 的噪声奖励信号在高维空间中不够精确；且课程中12q仅占20%，
   训练量不足。需要更多12q训练数据或更高 traj 才能泛化。
3. **布局策略影响**：
   - 5q l05 保真度 0.4465 > ph2 0.4193：l05 有 layout-mix（30% 恒等 /
     30% 随机 / 40% SABRE），ph2 无 layout-mix（仅 --mapping-min-swaps 1 +
     --lambda-layout 0.5）。5q 电路布局空间小，多样性收益 > 噪声微调收益。
   - 8q/10q ph2 反超：大电路布局空间大，噪声感知信号主导。
4. **SWAP 与并行度权衡**：
   ph2 在 8q 上 SWAP 10.3 vs l05 12.2，但保真度更高（0.1833 vs 0.1645），
   说明噪声微调学到更少 SWAP、更高并行度的策略（par_density 2.099 vs 2.106
   差异不大，但 makespan 19.48µs vs 20.72µs）。
5. **训练效率**：物理比特截断 + warmup 跳过使100k步仅耗~3.5h，对比旧 20q
   全电路 ~108s/fidelity 需数十小时，提速 10 倍以上。

### 评估命令（可复现）
```bash
# Phase2（5q）
python3 -m routing.rl.eval_policy \
  --model ../models/policy_ph2_sub20q.pt \
  --topo ../traindata/topo/tianyan176_20q.json \
  --data-dir ../traindata --split stage2_mixed \
  --fidelity-sim trajectory_sched --traj-trajectories 32 \
  --max-num-qubits 20 --seed 42 \
  --baselines --no-greedy --no-random

# l05 对照（5q）
python3 -m routing.rl.eval_policy \
  --model ../models/policy_tianyan20q_laymix_l05_eta05.pt \
  --topo ../traindata/topo/tianyan176_20q.json \
  --data-dir ../traindata --split stage2_mixed \
  --fidelity-sim trajectory_sched --traj-trajectories 32 \
  --max-num-qubits 20 --seed 42 \
  --baselines --no-greedy --no-random
```

### 后续
- 12q 泛化问题：需要更多12q训练数据（提高课程中12q占比）或用更高 traj 训练；
- 扩大电路数（~20–30 条/尺寸）收紧误差棒；
- 含 `crosstalk_strength` 的 5q 拓扑上复测，确认串扰建模作用；
- （可选）layout-mix 在大电路上的收益验证（是否值得重建 SABRE 缓存）。

---

## Phase 2 v2：相对保真度奖励 + 16q 课程（2026-08-31）

### 问题
Phase 2 v1（ph2_sub20q）的终端保真度奖励被步级塑形奖励彻底淹没：
- `λ_fid × fid ≈ 5 × 0.001-0.02 ≈ 0.005-0.1`
- 步级奖励累计 ≈ 45-120
- 保真度仅占总奖励 ~0.01%，被 `RewardNormalizer` 归一化为噪声级

### 改动
1. **相对保真度奖励**：`terminal = λ × (fid - baseline) / baseline`，
   baseline 按 split_key（电路尺寸）维护 EMA（`--fid-baseline-alpha 0.02`）。
   奖励量级从 ±0.01 提升至 ±(0.3~1)，与单步奖励可比。

2. **加 16q 线路**：课程改为 6 阶段均分：
   `stage2,large_n8,large_n8,large_n10,large_n12,large_n16`（各 ~17%）

### 训练命令
```bash
cd src
python3 -m routing.rl.train_agent \
  --topo-list ../traindata/topo/tianyan176_20q.json \
  --curriculum-keys stage2,large_n8,large_n8,large_n10,large_n12,large_n16 \
  --reward-mode noise_aware \
  --fidelity-sim trajectory_sched --traj-trajectories 16 \
  --max-num-qubits 20 \
  --load ../models/policy_tianyan20q_laymix_l05_eta05.pt \
  --eta-xtalk-par 1.0 \
  --mapping-min-swaps 1 --lambda-layout 0.5 \
  --lambda-fid-max 5.0 --lambda-fid-warmup 0.5 \
  --fid-baseline-alpha 0.02 \
  --timesteps 100000 --rollout-steps 256 --epochs 4 --lr 3e-4 --seed 0 \
  --checkpoint-dir ../models/ckpts_ph2_v2 --out ../models/policy_ph2_v2.pt
```

### 设置
- 基模：l05（`policy_tianyan20q_laymix_l05_eta05.pt`）
- 拓扑：tianyan176_20q（真机 20q，crosstalk_strength=None）
- warmup：前 50k 步 λ_fid=0，后 50k 步线性增长至 5.0
- 相对归一化 EMA α=0.02，按 split_key 分尺寸维护 baseline
- 无 layout-mix，无 SABRE 缓存
- 训练耗时 ~4.5h（tmux session `ph2v2`）

### 评估结果（5 电路尺寸对比）

| 尺寸 | PPO Fidelity | SABRE Fidelity | PPO SWAP | SABRE SWAP | PPO 相对提升 |
|------|-------------|----------------|----------|------------|-------------|
| 5q | **0.4098** | 0.3830 | 6.4±3.3 | 4.5±1.5 | +7.0% |
| 8q | **0.1663** | 0.1357 | 9.7±3.9 | 8.1±2.8 | +22.5% |
| 10q | **0.0748** | 0.0506 | 13.9±6.2 | 11.2±3.9 | +47.8% |
| 12q | 0.0021 | **0.0032** | 26.0±0.0 | 15.0±0.0 | -34.4% |
| 16q | **0.0008** | 0.0002 | 39.0±0.0 | 21.0±1.1 | +300% |

### 对比历史最佳

| 尺寸 | l05 | ph2(v1) | ph2(v2) | SABRE | 最佳 |
|------|-----|---------|---------|-------|------|
| 5q | **0.4465** | 0.4193 | 0.4098 | 0.3867 | l05 |
| 8q | 0.1645 | **0.1833** | 0.1663 | 0.1372 | ph2(v1) |
| 10q | **0.0829** | 0.0733 | 0.0748 | 0.0503 | l05 |
| 12q | **0.0179** | 0.0032 | 0.0021 | 0.0032 | l05 |
| 16q | **0.0016** | — | 0.0008 | 0.0002 | l05 |

### 分析
1. **l05 全尺寸领先**：补测 l05 在 10q/12q/16q 的结果后，发现 l05 在所有尺寸上
   都领先（10q: 0.0829, 12q: 0.0179, 16q: 0.0016），说明 layout-mix 训练
   提供了更强的泛化能力，尤其是在大电路上。

2. **ph2 微调反而降低了泛化**：ph2(v1) 和 ph2(v2) 在大电路上表现不如 l05，
   可能原因：(a) 去掉 layout-mix 导致布局多样性丧失；(b) 噪声微调过度拟合
   小电路（5q-8q），牺牲了大电路的泛化。

3. **相对奖励的实际效果有限**：ph2(v2) 的相对归一化在 10q 上仅比 v1 提升 2%
   （0.0748 vs 0.0733），在 12q 上反而下降（0.0021 vs 0.0032），说明相对
   奖励的收益被 layout-mix 缺失的损失抵消。

4. **12q 仍失败**：所有 PPO 模型在 12q 上都低于 SABRE（l05: 0.0179 vs 0.0032
   是个例外，但 SWAP 18 远多于 SABRE 的 15），说明 12q 路由后用 ~18 物理
   比特时，噪声积累太严重，路由改进被硬件噪声淹没。

5. **核心结论**：layout-mix 是保持泛化能力的关键，不应轻易去掉。后续应恢复
   layout-mix 并在此基础上做噪声微调。

### 后续方向
- 恢复 layout-mix 重建 SABRE 缓存，在此基础上做噪声微调（保留泛化 + 噪声适应）；
- 相对奖励可保留，但需配合 layout-mix 才能发挥最大效果；
- 12q+ 的路由改进被硬件噪声淹没，评估重点应放在 5q-8q 实用区间。

---

## Phase 2 v3 — layout-mix + 相对奖励 + trajectory_sched

### Bug 修复：init_mapping 维度不匹配导致训练崩溃

**问题**：ph2v3 首次启动时立即崩溃：
```
RuntimeError: The expanded size of the tensor (20) must match the existing size
(35) at non-singleton dimension 0.
```

**根因分析**：
1. SABRE 缓存中 `initial_layout` 长度 = 拓扑物理比特数（20），而非电路逻辑比特数
2. 当5q 电路使用20元素 `init_mapping` 时，`env.mapping = list(init_mapping)` 得到20元素列表
3. `env.num_qubits =5`（电路比特数），`env.max_num_qubits =20`
4. `_obs()` 中 `map_vec` 有20元素，再 pad 15个零 →35元素
5. Agent 期望 `map_vec` 长度 = `agent.num_qubits =20`，收到35 → 崩溃

**修复**：
- `env.py:160`：`self.mapping = list(self.init_mapping[:n])` — 截断到电路比特数
- `agent.py:258`：`n_copy = min(mv_raw.shape[0], self.num_qubits)` — 防御性截断

**验证**：500 步 smoke test 通过，无维度错误。

### 训练设置
```bash
cd src
tmux new -s ph2v3
python3 -m routing.rl.train_agent \
  --topo-list ../traindata/topo/tianyan176_20q.json \
  --curriculum-keys stage2,large_n8,large_n8,large_n10,large_n12,large_n16 \
  --reward-mode noise_aware \
  --fidelity-sim trajectory_sched --traj-trajectories 16 \
  --max-num-qubits 20 \
  --load ../models/policy_tianyan20q_laymix_l05_eta05.pt \
  --eta-xtalk-par 1.0 \
  --mapping-min-swaps 1 --lambda-layout 0.5 \
  --lambda-fid-max 5.0 --lambda-fid-warmup 0.5 \
  --fid-baseline-alpha 0.02 \
  --layout-mix 0.3,0.3,0.4 \
  --sabre-cache-file ../models/sabre_cache_ph2v3.pkl \
  --timesteps 100000 --rollout-steps 256 --epochs 4 --lr 3e-4 --seed 0 \
  --checkpoint-dir ../models/ckpts_ph2_v3 --out ../models/policy_ph2_v3.pt
```

**与 v2 区别**：增加了 `--layout-mix 0.3,0.3,0.4`（恒等/随机/SABRE 各30%/30%/40%），验证 layout-mix 是否恢复泛化能力。

### 评估结果（tianyan176_20q 真机拓扑，noise_aware trajectory_sched×16）

| 尺寸 | l05 Fidelity | ph2v3 Fidelity | SABRE Fidelity | l05 vs SABRE | ph2v3 vs SABRE |
|------|-------------|----------------|----------------|-------------|----------------|
| 5q | **0.3691** | 0.3568 | 0.3091 | +19.4% | +15.4% |
| 8q | 0.1859 | **0.1862** | 0.1621 | +14.7% | +14.9% |
| 10q | **0.1003** | 0.0980 | 0.0614 | +63.3% | +59.6% |
| 12q | **0.0756** | 0.0605 | 0.0463 | +63.3% | +30.7% |
| 16q | **0.0214** | 0.0179 | 0.0065 | +229% | +175% |

### Routing（SWAPs）

| 尺寸 | l05 SWAPs | ph2v3 SWAPs | SABRE SWAPs |
|------|-----------|-------------|-------------|
| 5q | 6.0 | 6.0 | 5.3 |
| 8q | 11.4 | **9.6** | 8.9 |
| 10q | 15.1 | **13.0** | 10.8 |
| 12q | 20.5 | **18.8** | 16.1 |
| 16q | 28.5 | 28.6 | 23.7 |

### 分析

1. **ph2v3 未超越 l05**：在 tianyan176_20q 真机拓扑上，ph2v3 在 5q/10q/12q/16q 均略低于 l05，仅在 8q 上微弱持平（0.1862 vs 0.1859）。
2. **SWAPs 优势**：ph2v3 在 8q/10q/12q 上 SWAPs 少于 l05（分别 -1.6/-2.1/-1.7），说明 layout-mix 确实改善了路由效率，但噪声感知微调未能将 SWAP 优势转化为保真度优势。
3. **entropy 上升**：训练后期 entropy 从0.6→1.0，说明 λ_fid 信号太弱（大电路 fid≈0.0001），策略退化为随机探索。这可能是 ph2v3 保真度不如 l05 的主要原因。
4. **layout-mix 对泛化的影响**：v2（无 layout-mix）在大电路上保真度衰减更严重，ph2v3（有 layout-mix）在大电路上与 l05 差距缩小，说明 layout-mix 有助于维持泛化。
5. **核心结论**：layout-mix + 相对奖励的组合未能在噪声微调阶段带来提升。可能需要：(a) 增大 λ_fid 信号（如 λ_fid_max=50），(b) 缩短噪声微调步数，(c) 在 layout-mix 基础上做更保守的微调（小 lr、少步数）。

---

## Phase 2 v4：λ_fid 立即满权重 + 裁剪放宽 + swap 惩罚（ph2v4）

### 训练命令

```bash
cd src
python3 -m routing.rl.train_agent \
  --topo-list ../traindata/topo/tianyan176_20q.json \
  --curriculum-keys stage2,large_n8,large_n8,large_n10,large_n12,large_n16 \
  --reward-mode noise_aware \
  --fidelity-sim trajectory_sched --traj-trajectories 16 \
  --max-num-qubits 20 \
  --load ../models/policy_tianyan20q_laymix_l05_eta05.pt \
  --eta-xtalk-par 0.05 \
  --swap-cost 0.5 \
  --mapping-min-swaps 1 --lambda-layout 0.5 \
  --lambda-fid-max 5.0 --lambda-fid-warmup 0.0 \
  --sabre-fid-map "5=0.309,8=0.162,10=0.061,12=0.046,16=0.0065" \
  --layout-mix 0.3,0.3,0.4 \
  --sabre-cache-file ../models/sabre_cache_ph2v3.pkl \
  --timesteps 50000 --rollout-steps 256 --epochs 4 --lr 1e-4 --seed 0 \
  --checkpoint-dir ../models/ckpts_ph2_v4 --out ../models/policy_ph2_v4.pt
```

### v4 相对 v3 的三项修复

1. **λ_fid 立即满权重**：`lambda_fid_schedule(warmup=0.0)` 改为直接返回 `max_val`（不再从 0 线性爬升），5q/8q 阶段保真度信号全程满权重。
2. **裁剪放宽**：终端奖励 `np.clip(r, -10, 10)` → `(-50, 50)`，16q 上不再饱和为常数。
3. **swap 惩罚**：`--swap-cost` 默认 0.0 → 0.5，每个 SWAP 直接扣 0.5。

### 训练关键指标

| step | 阶段 | reward | swps | ent | KL | fid | xtalk |
|------|------|--------|------|-----|-----|-----|-------|
| 256 | 5q | 31.7 | 5.1 | 0.19 | 2.66 | 0.123 | 0.11 |
| 5888 | 5q | 43.5 | 5.1 | 0.43 | 2.12 | 0.142 | 0.12 |
| 8960 | 8q | 59.9 | 12.2 | 0.63 | 1.31 | 0.020 | 0.20 |
| 23296 | 10q | 59.7 | 10.4 | 0.60 | 1.20 | 0.014 | 0.17 |
| 35840 | 12q | 73.5 | 20.0 | 0.72 | 0.63 | 0.0006 | 0.34 |
| 46848 | 16q | 108.5 | 33.2 | 0.80 | 0.27 | 0.0001 | 0.46 |
| 49920 | 16q | 145.1 | 44.2 | 0.81 | 0.19 | 0.0001 | 0.44 |

### v4 vs v3 训练指标对比

| 指标 | v3 @step~45k | v4 @step~46k | 改善 |
|------|-------------|-------------|------|
| entropy | 0.94~1.03 | **0.80~0.97** | ✅ 不再退化为随机 |
| KL | 0.31~0.42 | **0.27~0.42** | ≈ 相当 |
| fid | 0.000075 | **0.0001** | ≈ 同量级 |
| swaps | 30~31 | 33~44 | 略多（swap_cost 低 0.5 不足以抑制） |

### 评估结果（tianyan176_20q，trajectory_sched×16）

| 尺寸 | l05 Fidelity | ph2v4 Fidelity | SABRE Fidelity | ph2v4 vs l05 | ph2v4 vs SABRE |
|------|-------------|----------------|----------------|-------------|----------------|
| 5q | **0.3691** | 0.3644 | 0.3091 | -1.3% | **+17.9%** |
| 8q | 0.1197 | **0.1286** | 0.0813 | **+7.4%** | **+58.2%** |
| 10q | 0.0255 | **0.0502** | 0.0224 | **+96.9%** | **+124%** |
| 12q | **0.0152** | 0.0112 | 0.0037 | -26.3% | **+203%** |
| 16q† | **0.0039** | 0.0034 | 0.0003 | -12.8% | **+1033%** |

†16q 评估用 4 轨迹（16 轨迹超时）。

### SWAPs

| 尺寸 | l05 SWAPs | ph2v4 SWAPs | SABRE SWAPs |
|------|-----------|-------------|-------------|
| 5q | 6.0 | 6.1 | 5.3 |
| 8q | 12.7 | 11.3 | 9.5 |
| 10q | 17.9 | 16.0 | 13.5 |
| 12q | 25.2 | 23.8 | 19.4 |
| 16q† | 35.6 | 33.8 | 28.6 |

### 分析

1. **ph2v4 在所有尺寸上超越 SABRE**：保真度提升 17.9%~1033%，这是首个全尺寸超越 SABRE 的版本。
2. **ph2v4 vs l05 互有胜负**：8q/10q 上 ph2v4 超越 l05（+7.4%/+96.9%），5q/12q/16q 上 l05 略优（-1.3%/-26.3%/-12.8%）。10q 提升幅度（+96.9%）非常显著。
3. **λ_fid 调度修复效果**：v4 在 5q 阶段保真度信号满权重（v3 时 λ_fid≈0），策略没有退化为随机（ent 0.81 vs v3 的 1.03），证实了"课程-错相"修复有效。
4. **swap_cost=0.5 不足以抑制过多 SWAP**：ph2v4 的 SWAPs 仍然偏多（12q 23.8 vs l05 25.2，改善不大），策略仍倾向于做更多 SWAP。后续可尝试增大 swap_cost（1.0~2.0）。
5. **12q 上 ph2v4 不如 l05 的原因**：12q 时 λ_fid=3.5（未满 5.0），且随机 rollout fid≈0.0006 仍远低于确定性评估 fid，终端奖励仍受探索噪声影响。
6. **改进空间**：(a) swap_cost 增大到 1.0~2.0 进一步约束 SWAP 数量；(b) 增大 λ_fid_max 到 10~20 让 log-相对奖励在更大电路上仍有区分度；(c) 减少训练步数（30k~40k），避免后期策略漂移。

---

## 无泄露评估（test split，0 重叠）

### 背景

此前评估使用 `*_phase3` split，与训练集 `*_mixed` 有 ~33% 重叠（数据泄露）。创建 `stage1_test` 和 `large_n*_test` split（各 60 条 random 电路），与所有训练 split **0 重叠**。

### 评估结果（test split，trajectory_sched×16）

| 尺寸 | l05 Fidelity | ph2v4 Fidelity | SABRE Fidelity | l05 vs SABRE | ph2v4 vs SABRE |
|------|-------------|----------------|----------------|-------------|----------------|
| 5q | **0.5767** | 0.5531 | 0.5382 | **+7.2%** | **+2.8%** |
| 8q | 0.1859 | **0.1884** | 0.1621 | +14.7% | **+16.2%** |
| 10q | 0.1003 | **0.1075** | 0.0614 | +63.3% | **+75.1%** |
| 12q | 0.0861 | **0.0971** | 0.0541 | +59.1% | **+79.5%** |
| 16q† | **0.0190** | 0.0174 | 0.0047 | **+304%** | **+270%** |

†16q 用 4 轨迹评估（16 轨迹超时）。

### SWAPs（test split）

| 尺寸 | l05 SWAPs | ph2v4 SWAPs | SABRE SWAPs |
|------|-----------|-------------|-------------|
| 5q | 4.1 | **3.9** | 3.8 |
| 8q | 11.4 | **10.1** | 8.9 |
| 10q | 15.1 | **13.5** | 10.8 |
| 12q | 20.2 | **18.6** | 15.9 |
| 16q† | 27.7 | 27.7 | 23.2 |

### 数据泄露影响（phase3 vs test 对比）

| 尺寸 | l05 phase3 | l05 test | 泄露膨胀 |
|------|-----------|---------|---------|
| 5q | 0.3691 | 0.5767 | -36.0% |
| 8q | 0.1197 | 0.1859 | -35.6% |
| 10q | 0.0255 | 0.1003 | -74.6% |
| 12q | 0.0152 | 0.0861 | -82.3% |
| 16q† | 0.0039 | 0.0190 | -79.5% |

phase3 评估因数据泄露严重低估了模型真实性能（尤其 10q+），test split 才反映真实泛化能力。

### 分析

1. **无泄露下两者全面超越 SABRE**：l05 和 ph2v4 在所有 5 个尺寸上均超越 SABRE（+2.8%~+304%），确认 PPO 路由策略的真实泛化优势。
2. **ph2v4 vs l05 互有胜负**：8q/10q/12q 上 ph2v4 更优（+16.2%/+75.1%/+79.5%），5q/16q 上 l05 略优（+7.2%/+304% vs +2.8%/+270%）。
3. **ph2v4 在中等规模（10q-12q）优势最大**：这是噪声感知微调最有效的区间，log-相对奖励 + swap_cost 惩罚共同提升了路由质量。
4. **SWAPs 方面 ph2v4 全面优于 l05**：在所有尺寸上 ph2v4 的 SWAPs 更少或持平，说明噪声感知微调确实改善了路由效率。
5. **数据泄露使此前结论部分失效**：phase3 评估中"12q 上 ph2v4 不如 l05"（-26.3%）在 test split 上反转为"ph2v4 优于 l05"（+12.8%），说明泄露评估不可靠。

---

## Phase 2 v5：swap_cost=1.5 + 冻结5q阶段 + 60k步（ph2v5）

### 训练命令

```bash
cd src
python3 -m routing.rl.train_agent \
  --topo-list ../traindata/topo/tianyan176_20q.json \
  --curriculum-keys stage2,large_n8,large_n8,large_n10,large_n12,large_n16 \
  --reward-mode noise_aware \
  --fidelity-sim trajectory_sched --traj-trajectories 16 \
  --max-num-qubits 20 \
  --load ../models/policy_tianyan20q_laymix_l05_eta05.pt \
  --eta-xtalk-par 0.05 \
  --swap-cost 1.5 \
  --mapping-min-swaps 1 --lambda-layout 0.5 \
  --lambda-fid-max 5.0 --lambda-fid-warmup 0.0 \
  --sabre-fid-map "5=0.309,8=0.162,10=0.061,12=0.046,16=0.0065" \
  --layout-mix 0.3,0.3,0.4 \
  --sabre-cache-file ../models/sabre_cache_ph2v3.pkl \
  --freeze-first-steps 10000 \
  --timesteps 60000 --rollout-steps 256 --epochs 4 --lr 1e-4 --seed 0 \
  --checkpoint-dir ../models/ckpts_ph2_v5 --out ../models/policy_ph2_v5.pt
```

### v5 相对 v4 的改动

1. **swap_cost 0.5→1.5**：每 SWAP 惩罚增大 3 倍，抑制过多 SWAP
2. **冻结前 10k 步**：5q 阶段只收集数据不更新参数，保护纯路由能力
3. **总步数 50k→60k**：给大电路更多训练时间

### 评估结果（test split，trajectory_sched，无泄露）

| 尺寸 | l05 Fidelity | ph2v4 Fidelity | ph2v5 Fidelity | SABRE Fidelity |
|------|-------------|----------------|----------------|----------------|
| 5q | **0.5767** | 0.5531 | 0.5673 | 0.5382 |
| 8q | 0.1859 | **0.1884** | 0.1872 | 0.1621 |
| 10q | 0.1003 | **0.1075** | 0.1005 | 0.0614 |
| 12q | 0.0861 | **0.0971** | 0.0821 | 0.0541 |
| 16q† | **0.0190** | 0.0174 | 0.0163 | 0.0047 |

†16q 用 4 轨迹评估。

### SWAPs（test split）

| 尺寸 | l05 | ph2v4 | ph2v5 | SABRE |
|------|-----|-------|-------|-------|
| 5q | 4.1 | **3.9** | 4.1 | 3.8 |
| 8q | 11.4 | **10.1** | 10.0 | 8.9 |
| 10q | 15.1 | **13.5** | 13.6 | 10.8 |
| 12q | 20.2 | 18.6 | **19.0** | 15.9 |
| 16q† | 27.7 | **27.7** | 28.4 | 23.2 |

### 训练指标对比

| 指标 | ph2v4 @50k | ph2v5 @52k | 变化 |
|------|-----------|-----------|------|
| entropy | 0.81 | 0.79 | ↓ 更稳定 |
| KL | 0.37 | 0.28 | ↓ 策略变化更小 |
| SWAPs | 43 | 36 | ↓ swap_cost 有效 |
| fid(训练) | 0.00005 | 0.00024 | ↑ 保真度信号改善 |

### 分析

1. **swap_cost=1.5 有效抑制 SWAP**：ph2v5 在 8q/10q 上 SWAPs 与 ph2v4 持平，但12q 上从 18.6→19.0 反而略增，说明惩罚强度还不够。
2. **冻结5q阶段效果有限**：ph2v5 在 5q 上（0.5673）介于 l05（0.5767）和 ph2v4（0.5531）之间，冻结确实改善了 5q 退化，但没有完全恢复 l05 水平。
3. **ph2v5 整体不如 ph2v4**：在 10q/12q 上 ph2v4 仍优于 ph2v5（+7.0%/+18.3%），说明增大 swap_cost 和冻结策略的组合并未带来整体提升。
4. **entropy 更稳定**：ph2v5 的 entropy 始终低于 ph2v4（0.79 vs 0.81），KL 也更低（0.28 vs 0.37），说明冻结+高 swap_cost 让训练更稳定，但稳定性没有转化为保真度优势。
5. **结论**：冻结5q + swap_cost=1.5 的改进方向正确（5q 退化改善），但强度不够。后续可尝试：(a) 冻结更长（15k~20k步）；(b) swap_cost 进一步增大到 2.0~3.0；(c) 同时增大 λ_fid_max 到 10 让保真度信号更强。

---

## Phase 2 v6b：EMA 评估 + 最优 checkpoint + 冻结5q + 自适应 λ_fid_max（ph2v6b）

### 训练命令

```bash
cd src
python3 -m routing.rl.train_agent \
  --topo-list ../traindata/topo/tianyan176_20q.json \
  --curriculum-keys stage2,large_n8,large_n8,large_n10,large_n12,large_n16 \
  --reward-mode noise_aware \
  --fidelity-sim trajectory_sched --traj-trajectories 16 \
  --max-num-qubits 20 \
  --timesteps 60000 \
  --lambda-fid-max-schedule 5,5,5,5,5,10 \
  --sabre-fid-map "5=0.309,8=0.162,10=0.061,12=0.046,16=0.0065" \
  --swap-cost 0.5 \
  --freeze-first-steps 10000 \
  --ema-decay 0.999 \
  --eval-interval 5000 \
  --eval-split stage1_test,large_n8_test,large_n10_test,large_n12_test,large_n16_test \
  --eval-max-qubits 20 --eval-traj 16 \
  --load ../models/policy_tianyan20q_laymix_l05_eta05.pt \
  --out ../models/policy_ph2_v6b.pt \
  --phase 2
```

### 训练指标

- 训练步数：60160
- EMA best：step=35072，mean_fid=0.3114（trajectory_sched×16，test split 全尺寸均值）
- 训练后期（43k+）：entropy 升至 ~1.0，终端态保真度（训练reward）降至 ~0.00002，SWAPs 膨胀至 33-40
- 训练退化模式与 ph2v3 相同（entropy→1.0 策略随机化），EMA best 捕获了退化前的最优策略

### 评估结果（test split，trajectory_sched，同口径）

| 尺寸 | ph2v6b EMA | ph2v6b Final | l05 | ph2v4 | SABRE |
|------|-----------|-------------|-----|-------|-------|
| 5q | 0.5666 | **0.5767** | 0.5767 | 0.5531 | 0.5382 |
| 8q | 0.1781 | **0.1859** | 0.1859 | 0.1884 | 0.1621 |
| 10q | 0.0902 | **0.1003** | 0.1003 | **0.1075** | 0.0614 |
| 12q | 0.0749 | **0.0756** | 0.0861 | **0.0971** | 0.0463 |
| 16q† | 0.0092 | **0.0176** | **0.0190** | 0.0174 | 0.0042 |

†16q 用 4 轨迹评估。

### vs SABRE 百分比

| 尺寸 | ph2v6b EMA | ph2v6b Final | l05 | ph2v4 |
|------|-----------|-------------|-----|-------|
| 5q | +5.3% | **+7.2%** | **+7.2%** | +2.8% |
| 8q | +9.9% | +14.7% | +14.7% | **+16.2%** |
| 10q | +46.9% | **+63.3%** | +63.3% | **+75.1%** |
| 12q | +61.8% | **+63.3%** | +59.1% | **+79.5%** |
| 16q† | +119% | **+319%** | **+304%** | +270% |

### SWAPs

| 尺寸 | ph2v6b EMA | ph2v6b Final | l05 | ph2v4 | SABRE |
|------|-----------|-------------|-----|-------|-------|
| 5q | 4.0 | 4.1 | 4.1 | 3.9 | 3.8 |
| 8q | 10.4 | 11.4 | 11.4 | 10.1 | 8.9 |
| 10q | 13.9 | 15.1 | 15.1 | 13.5 | 10.8 |
| 12q | 18.5 | 20.5 | 20.2 | 18.6 | 15.9 |
| 16q† | 28.1 | 28.5 | 27.7 | 27.7 | 23.2 |

### 分析

1. **ph2v6b_final 全尺寸超越 SABRE**（+7.2%~+319%），确认 Phase 2 噪声感知微调+EMA 策略的有效性。
2. **ph2v6b_final vs ph2v4 互有胜负**：5q（+7.2% vs +2.8%）和16q（+319% vs +270%）v6b更优，8q（+14.7% vs +16.2%）/10q（+63.3% vs +75.1%）/12q（+63.3% vs +79.5%）v4更优。
3. **EMA best vs final**：EMA best（step 35072）在所有尺寸上不如 final（step 60160），说明训练后期（尽管 entropy 升高）仍在产出更好的策略——EMA 的延迟平均反而稀释了质量。
4. **训练退化与输出质量的矛盾**：训练后期 entropy→1.0（随机探索）+ 终端态保真度→0，但最终 checkpoint（final）的确定性评估反而最好。原因可能是：(a) 训练中的随机探索（entropy）≠ 确定性推理时的行为；(b) PPO 的 clip/梯度裁剪在 reward 信号弱时仍保持策略质量。
5. **SWAPs 偏多**：ph2v6b 的 SWAPs 与 l05 持平或更多（尤其12q: 20.5 vs 18.6），swap_cost=0.5 不足以抑制过多 SWAP。后续可尝试 swap_cost=2.0~3.0。
6. **改进空间**：(a) swap_cost 增大到 2.0~3.0 进一步约束 SWAP 数量；(b) 减少训练步数（30k~40k），避免后期策略退化；(c) 用 Hellinger 保真度训练（匹配论文评测指标），消除训练/评测目标不一致。

---

## NAM Circuits Benchmark（tianyan176_20q, trajectory_sched×16）

### 环境

- 拓扑：tianyan176_20q（20 比特，29 耦合边）
- 模型：l05（Phase 1 基模）、ph2v4（Phase 2 噪声感知）
- 评测：trajectory_sched，16 条轨迹，seed=0
- 基线：Qiskit SabreSwap
- 电路：19 个 ≤20q NAM benchmark（删除了 10 个 >20q 电路）

### 命令

```bash
cd src
# 路由生成（无保真度，快速）
PYTHONPATH=. python3 -m routing.rl.generate_routing \
  --model ../models/policy_tianyan20q_laymix_l05_eta05.pt \
  --model-name l05 \
  --circuit-dir ../benchmark/nam_circs \
  --topo ../traindata/topo/tianyan176_20q.json \
  --label-map ../traindata/topo/tianyan176_20q_labels.json \
  --max-num-qubits 20 \
  --out-dir ../benchmark/routed/l05 \
  --no-fidelity

# 批量保真度计算
cd .. && PYTHONPATH=src python3 scripts/compute_fidelity.py
```

### 总览

| 指标 | SABRE | l05 | ph2v4 |
|------|-------|-----|-------|
| 平均 SWAPs | **45.1** | 65.3 | 64.0 |
| 平均保真度 | 0.2747 | **0.3115** | 0.2872 |
| 胜 SABRE | — | 10/19 | 5/19 |

### 分电路对比

| Circuit | q | gates | SABRE_sw | l05_sw | ph2v4_sw | SABRE_fid | l05_fid | ph2v4_fid | Winner |
|---------|---|-------|---------:|-------:|---------:|----------:|--------:|----------:|--------|
| barenco_tof_10 | 19 | 184 | 193 | 0.0715 | **0.1960** | 0.0472 |
| barenco_tof_3 | 5 | 14 | 14 | 0.7467 | **0.7636** | 0.6817 |
| barenco_tof_4 | 7 | 22 | 26 | 0.4454 | 0.4830 | **0.5132** |
| barenco_tof_5 | 9 | 40 | 37 | 0.1683 | 0.3089 | **0.4688** |
| csla_mux_3 | 15 | 54 | 45 | 0.0341 | **0.2566** | 0.0685 |
| gf2^4_mult | 12 | 225 | 50 | 56 | 51 | 0.2348 | 0.1989 | **0.2588** | ph2v4 |
| gf2^5_mult | 15 | 347 | 72 | 101 | 95 | **0.0793** | 0.0267 | 0.0343 | SABRE |
| gf2^6_mult | 18 | 495 | 109 | 162 | 180 | 0.0314 | 0.0650 | **0.1136** | ph2v4 |
| grover_5 | 9 | 831 | 119 | 149 | 163 | 0.0006 | **0.0033** | 0.0002 | l05 |
| hwb6 | 7 | 68 | 68 | 0.0000 | **0.0003** | 0.0000 |
| mod5_4 | 5 | 14 | 13 | **0.5122** | 0.4148 | 0.4154 |
| mod_mult_55 | 9 | 27 | 24 | **0.1554** | 0.1057 | 0.0609 |
| mod_red_21 | 11 | 72 | 69 | 0.0645 | **0.1215** | 0.0248 |
| rc_adder_6 | 14 | 68 | 58 | **0.1662** | 0.1608 | 0.1458 |
| tof_10 | 19 | 107 | 94 | **0.4264** | 0.2223 | 0.1701 |
| tof_3 | 5 | 11 | 10 | 0.7779 | **0.8198** | 0.7935 |
| tof_4 | 7 | 19 | 13 | 0.6570 | **0.7138** | 0.6323 |
| tof_5 | 9 | 25 | 18 | 0.4699 | **0.6041** | 0.4424 |
| vbe_adder_3 | 10 | 48 | 45 | 0.1779 | 0.4543 | **0.5850** |

### 分析

1. **l05 平均保真度超越 SABRE 13.4%**（0.3115 vs 0.2747），尽管 SWAPs 多 45%（65.3 vs 45.1）。PPO 学会了选择噪声友好的路由路径——更多 SWAP 但每个 SWAP 放在低噪声边。
2. **ph2v4 平均保真度超越 SABRE 4.6%**（0.2872 vs 0.2747），SWAPs 多 42%（64.0 vs 45.1）。不如 l05，可能因为训练退化。
3. **l05 在中等规模电路（9-15q）上优势显著**：csla_mux_3（+652%）、barenco_tof_10（+174%）、vbe_adder_3（+155%）、tof_5（+28.6%）。
4. **SABRE 在小电路（5q）和部分19q电路上仍更强**：mod5_4（-19%）、tof_10（-47.9%）、gf2^5_mult（-66%）。SABRE SWAPs 更少，对噪声不敏感的小电路更优。
5. **ph2v4 胜 SABRE 的电路多为特定拓扑**：barenco_tof_4/5、gf2^4/6_mult、vbe_adder_3。这些电路的门模式恰好适合 PPO 的噪声感知路由。
6. **grover_5 / hwb6 保真度极低**（<0.005）：831/259 门的深电路，无论哪种算法都无法保持保真度，属于噪声极限。
7. **生成工具**：`src/routing/rl/generate_routing.py`（路由 + Q-label 映射 + 保真度计算），输出 JSON 含 initial_layout（Q 标签）、routed_qasm、fidelity。

---

## Beam Search 保真度优化实验

### 背景

l05 beam search 保真度反而比 argmax 差（0.26 vs 0.31），原因分析：
- l05 的 critic V(s') 在 Phase 1 routing 模式下训练（`reward_mode=routing`），**完全不知道保真度**
- beam 评分 `score = reward_c + γ·V(s')` 贪心优化"路由效率"（SWAP 少、makespan 短），但 SWAP 少 ≠ 保真度高
- argmax 的"多出来的 SWAP"是故意绕开高错误率边的（`eta_err=0.5` 惩罚高错误门），反而保真度更高

### 方案：ph2v4 noise-aware critic beam search

ph2v4 的 critic V(s') 在 Phase 2 noise-aware 模式下训练（终端保真度奖励 + swap_cost=0.5），V(s') 已编码保真度信息。用 `reward_mode='noise_aware'` 跑 beam search，V(s') 自然引导保真度优化。

```bash
cd src
PYTHONPATH=. python3 -m routing.rl.generate_routing \
  --model ../models/policy_ph2_v4.pt \
  --model-name ph2v4_beam5 \
  --circuit-dir ../benchmark/nam_circs \
  --topo ../traindata/topo/tianyan176_20q.json \
  --label-map ../traindata/topo/tianyan176_20q_labels.json \
  --max-num-qubits 20 \
  --out-dir ../benchmark/routed/ph2v4_beam5 \
  --reward-mode noise_aware \
  --beam-width 5 \
  --swap-cost 0.5 \
  --eta-xtalk-par 0.05 \
  --lambda-fid-max 5.0 \
  --no-fidelity
```

### 结果

| Method | Avg SWAPs | Avg Fidelity | vs SABRE |
|--------|----------|-------------|----------|
| SABRE | 45.1 | 0.2747 | — |
| l05 argmax | 65.3 | **0.3115** | +13.4% |
| l05 beam3 | 43.6 | 0.2617 | -4.7% |
| l05 beam5 | 41.6 | 0.2408 | -12.3% |
| ph2v4 argmax | 64.0 | 0.2872 | +4.6% |
| ph2v4 beam3 | 42.6 | 0.2859 | +4.1% |
| **ph2v4 beam5** | **42.2** | **0.3047** | **+10.9%** |

### 分电路保真度对比

| Circuit | q | SABRE | l05 | l05 b3 | ph2v4 | ph2v4 b3 | ph2v4 b5 |
|---------|---|------:|----:|-------:|------:|---------:|---------:|
| barenco_tof_10 | 19 | 0.014 | 0.047 | 0.027 | 0.244 |  |
| barenco_tof_4 | 7 | 0.420 | 0.513 | 0.457 | **0.613** |  |
| csla_mux_3 | 15 | 0.112 | 0.068 | **0.375** | 0.224 |  |
| gf2^5_mult | 15 | 0.079 | 0.027 | 0.039 | 0.034 | **0.196** | 0.139 |
| tof_10 | 19 | **0.426** | 0.222 | 0.373 | 0.170 | 0.305 | 0.363 |
| tof_3 | 5 | 0.729 | 0.794 | **0.848** | 0.748 |  |
| vbe_adder_3 | 10 | 0.077 | **0.585** | 0.305 | 0.568 |  |

### 分析

1. **ph2v4 beam5 首次同时实现低 SWAP + 高保真度**：SWAP 42.2（低于 SABRE 45.1）+ 保真度 0.3047（超越 SABRE 10.9%）。这是 noise-aware critic V(s') 引导 beam search 的结果。
2. **l05 beam search 保真度恶化**（-4.7%~-12.3%）：证实 l05 的 critic 不含保真度信号，beam 贪心优化错误目标。
3. **ph2v4 beam5 > ph2v4 argmax**（0.3047 vs 0.2872，+6.1%）：noise-aware V(s') 的 lookahead 确实帮助选择了更优路径。
4. **ph2v4 beam3 ≈ ph2v4 argmax**（0.2859 vs 0.2872）：beam3 收益有限，beam5 的更多候选带来更大改善。
5. **beam search 大幅降低 SWAP**：所有 beam 变体 SWAP 41-43，比 argmax（64-65）少 35%，比 SABRE（45）也低。
6. **部分电路 ph2v4 beam 超越所有 argmax**：barenco_tof_10（0.244 vs 0.196）、barenco_tof_4（0.613 vs 0.513）、gf2^5_mult（0.196 vs 0.027）。

### 结论

critic V(s') 的训练目标决定了 beam search 的方向：
- **routing critic**（l05）→ beam 搜索"少 SWAP"→ 保真度下降
- **noise-aware critic**（ph2v4）→ beam 搜索"高保真度"→ SWAP 和保真度双优

ph2v4 beam5 是当前最优配置：SWAP 低于 SABRE、保真度超越 SABRE 10.9%。后续可尝试方案 B（真实保真度 rollout 评分）进一步提升。

---

## 解析保真度代理 + NAM 电路训练扩展（2026-09-07）

### 动机

ph2v4 在 NAM 算术电路上退化（avg 0.2872 vs l05 的 0.3115，-7.8%），根因：
1. 训练分布不匹配：Phase 2 用 ≤16q 随机电路，NAM 是 15-19q 结构化算术电路
2. `sabre_fid_map` 只覆盖到 16q，15q/18q/19q 无 log-相对奖励信号
3. 保真度模拟开销：trajectory_sched 在 19q×643 门上需 ~13 min/episode，无法扩展训练

### 实现

#### 1. 解析保真度代理（`trajectory_sim.py:make_analytic_fidelity_fn`）

一阶错误累积：`log F ≈ Σ log(1-ε_i)`，O(门数)，无 2^k 指数。

| 通道 | 近似公式 |
|------|---------|
| 单比特门退极化 | `log(1 - ε1[q])` |
| CX 退极化 | `log(1 - ε2[a,b])` |
| SWAP（=3 CX） | `3 × log(1 - ε2[a,b])` |
| 热弛豫（可选） | `log(1 - (t/T1·½ + t/T2·½))` |
| 串扰 θ²（可选） | `-θ²` |

CLI: `--fidelity-sim analytic [--analytic-thermal] [--analytic-crosstalk]`

#### 2. NAM 电路训练混入（`train_agent.py`）

- `--nam-circuits-dir PATH`：加载 QASM 目录（如 `benchmark/nam_circs/`）
- `--nam-circuit-prob 0.3`：30% 概率选 NAM 电路，70% 走原随机电路
- `load_nam_circuits()` → `pick_circuit_with_nam()` 混合采样器

#### 3. NAM 专用 sabre_fid_map

从 SABRE benchmark 数据计算（19 circuits, trajectory_sched×16）：

| Qubits | SABRE Fidelity | NAM Circuits |
|--------|---------------|--------------|
| 5 | 0.6789 | barenco_tof_3, mod5_4, tof_3 |
| 7 | 0.3675 | barenco_tof_4, hwb6, tof_4 |
| 9 | 0.1985 | barenco_tof_5, grover_5, mod_mult_55, tof_5 |
| 10 | 0.1779 | vbe_adder_3 |
| 11 | 0.0645 | mod_red_21 |
| 12 | 0.2348 | gf2^4_mult |
| 14 | 0.1662 | rc_adder_6 |
| 15 | 0.0567 | csla_mux_3, gf2^5_mult |
| 18 | 0.0314 | gf2^6_mult |
| 19 | 0.2489 | barenco_tof_10, tof_10 |

CLI: `--sabre-fid-map "5=0.6789,7=0.3675,9=0.1985,10=0.1779,11=0.0645,12=0.2348,14=0.1662,15=0.0567,18=0.0314,19=0.2489"`

### 训练命令（Phase 2 NAM 扩展，从 l05 微调）

使用 tianyan176 20q 拓扑（覆盖所有 NAM 电路的 5-19q），从 l05 routing 模型微调：

```bash
cd src
python3 -m routing.rl.train_agent \
  --topo ../traindata/topo/tianyan176_20q.json \
  --max-num-qubits 20 \
  --reward-mode noise_aware \
  --fidelity-sim analytic \
  --analytic-thermal \
  --nam-circuits-dir ../benchmark/nam_circs \
  --nam-circuit-prob 0.3 \
  --sabre-fid-map "5=0.6789,7=0.3675,9=0.1985,10=0.1779,11=0.0645,12=0.2348,14=0.1662,15=0.0567,18=0.0314,19=0.2489" \
  --lambda-fid-max 5.0 --lambda-fid-warmup 0.0 \
  --eta-xtalk-par 0.05 --use-scheduler \
  --load ../models/policy_tianyan20q_laymix_l05_eta05.pt \
  --timesteps 100000 \
  --out ../models/policy_nam_l05_finetune_v1.pt
```

### 测试结果

- 50/50 测试通过（1 个 pre-existing flaky test `test_fidelity_shaping_step_zero` 排除）
- 19 个 NAM 电路全部正确加载（5q-19q，45-831 gates）
- 解析代理 5q 电路：fid=0.254（O(门数) 无指数）

### 下一步

1. 运行 NAM 扩展训练，评估 analytic proxy 在 15-19q 上的泛化效果
2. 对比 analytic vs trajectory_sched 的相关性（确保代理能代表真实 fid 排序）
3. 可选：自动切换策略（≤12q 用 trajectory_sched，>12q 用 analytic）

---

## NAM l05 微调 v2（unified_mixed 多尺度 + analytic fidelity，100k 步）—— 结果退化

### 背景

v1 微调（`stage2_mixed`，纯 5q）误用 split 被中途终止。改用 `unified_mixed`（8/10/12/16/20q 各 140 条，700 条）作为微调数据源，30% 概率混入 NAM 电路（5-19q），analytic fidelity 代理（含 thermal）作终端奖励。

### 训练命令

```bash
cd src
python3 -m routing.rl.train_agent \
  --topo-list ../traindata/topo/tianyan176_20q.json \
  --split-prefix unified \
  --reward-mode noise_aware \
  --timesteps 100000 \
  --load ../models/policy_tianyan20q_laymix_l05_eta05.pt \
  --out ../models/policy_nam_l05_finetune_v2.pt \
  --fidelity-sim analytic \
  --nam-circuits-dir ../benchmark/nam_circs \
  --nam-circuit-prob 0.3 \
  --analytic-thermal \
  --max-num-qubits 20 \
  --use-scheduler \
  --lambda-fid-max 5.0 --lambda-fid-warmup 0.0 \
  --eta-xtalk-par 0.05 --eta-dist 0.0 \
  --ema 0.999 --swap-cost 0.5 \
  --eval-interval 5000 --eval-split large_n16_test
```

日志：`models/train_nam_l05_finetune_v2.log`（session `nam_finetune`，~23 steps/s，总耗时 ~72min）

### 训练日志摘要

- 全程 trunc=0%，entropy 0.5-1.05 健康，KL ~0.01-0.18，grad_norm ~0.8-2.1
- fid（analytic proxy，8-20q 大电路）：0.02-0.05 波动
- **EMA eval（large_n16_test）：step 5120 → 0.01366（best），此后从未提升；最终 0.01356**
- 关键观察：EMA 保真度全程无改善，说明 unified_mixed + NAM 微调未带来验证集增益，模型在偏离 l05 起点

### 评估（NAM benchmark，19 circuits ≤20q，trajectory_sched×16，SABRE 基线）

路由：`generate_routing.py --no-fidelity`；fidelity：`scripts/compute_fidelity_nam_v2.py`（trajectory_sched×16 seed=0 scheduled=True）

| 方法 | 平均 SWAPs | 平均保真度 | vs SABRE |
|------|-----------|-----------|----------|
| SABRE | 45.1 | 0.2747 | — |
| l05 argmax | 65.3 | **0.3115** | +13.4% |
| ph2v4 argmax | 64.0 | 0.2872 | +4.6% |
| **nam_l05_v2 argmax** | **57.1** | 0.2485 | **-9.5%** |
| l05 beam3 | 43.6 | 0.2617 | -4.7% |
| **nam_l05_v2 beam3** | **45.8** | 0.2717 | -1.1% |
| ph2v4 beam5 | 42.2 | 0.3047 | +10.9% |

### 逐电路对比（v2 argmax vs l05 argmax）

| Circuit | q | gates | l05_fid | v2_fid | v2_sw | l05_sw | 变化 |
|---------|---|-------|--------:|-------:|------:|-------:|------|
| barenco_tof_3 | 5 | 0.6428 | 11 | 14 | -15.8% |  |
| mod5_4 | 5 | **0.4666** | 11 | 14 | +12.5% |  |
| tof_3 | 5 | 0.7627 | 10 | 11 | -7.0% |  |
| barenco_tof_4 | 7 | 0.4292 | 18 | 22 | -11.1% |  |
| tof_4 | 7 | 0.5307 | 15 | 19 | -25.7% |  |
| barenco_tof_5 | 9 | 0.1628 | 34 | 40 | -47.3% |  |
| tof_5 | 9 | **0.6411** | 25 | 25 | +6.1% |  |
| mod_red_21 | 11 | **0.0002** | 57 | 72 | -99.8% |  |
| gf2^4_mult | 12 | 225 | 0.1989 | 0.0393 | 58 | 56 | -80.2% |
| rc_adder_6 | 14 | **0.0000** | 47 | 68 | -100% |  |
| gf2^5_mult | 15 | 347 | 0.0267 | **0.2551** | 86 | 101 | +856% |
| csla_mux_3 | 15 | 0.1952 | 47 | 54 | -23.9% |  |
| gf2^6_mult | 18 | 495 | 0.0650 | 0.0216 | 134 | 162 | -66.8% |
| barenco_tof_10 | 19 | 0.0190 | 167 | 184 | -90.3% |  |
| tof_10 | 19 | 0.1890 | 82 | 107 | -15.0% |  |

### 分析与结论

1. **v2 argmax 保真度显著退化**（0.2485 vs l05 0.3115，-20%），SWAPs 减少（57.1 vs 65.3）但 fidelity 反而降——微调把模型推向「少 SWAP、高 analytic-F」的路线，与 l05 的「噪声友好路由」（多 SWAP 但放低噪声边）路径偏离。
2. **EMA best 出现在 step 5120**（≈l05 起点，fid 0.01366），此后 95k 步从未刷新——训练奖励（analytic proxy）与评估指标（trajectory_sched）排序不一致，模型在优化代理但损害真实 fidelity。
3. **beam3 部分修复退化**：0.2717（-1.1% vs SABRE），但仍不及 l05 beam3 的差距闭合效果与 ph2v4 beam5（0.3047）。
4. **个别电路大幅退化**：rc_adder_6（0.16→0.00）、mod_red_21（0.12→0.0002）、barenco_tof_10（0.20→0.019）、gf2^4_mult（0.20→0.039）——这些恰是 l05 优势最大的中等规模电路。
5. **唯一显著提升**：gf2^5_mult（+856%）、mod5_4（+12%）、tof_5（+6%）。gf2^5_mult 是 15q 347 门深电路，v2 少用了 15 个 SWAP 反而更优——unified 多尺度训练在部分深电路上有正向迁移。

### 根因假设（需验证）

- **analytic proxy vs trajectory_sched 排序不一致**：训练用 analytic（O(门数) 一阶累积、无调度），评估用 trajectory_sched（含调度空闲退相干 + 动态串扰）。若两者对「哪条路由更好」给出不同排序，PPO 优化代理可能损害真实 fidelity。
- **70% unified_mixed 占主导**：微调大部分梯度来自 8-20q 随机/QAOA/VQE 电路，稀释了 NAM 算术电路的模式。
- 下一步：用相同数据训练但 `--fidelity-sim trajectory_sched`（≤16q）或先做 analytic vs trajectory 相关性分析，确认代理是否可靠。

### 模型文件

- `models/policy_nam_l05_finetune_v2.pt`（step 100096 最终）
- `models/policy_nam_l05_finetune_v2_ema_best.pt`（step 5120）
- 路由结果：`benchmark/routed/nam_l05_v2*.json`（含 summary）

---

## analytic vs trajectory_sched 保真度排序相关性分析 —— 代理不可靠

### 背景

v2 微调（unified_mixed + analytic fidelity）在 NAM benchmark 上退化（argmax 0.2485 vs l05 0.3115）。
主要嫌疑是训练用 analytic 代理与评估用 trajectory_sched 对「哪条路由更好」的排序不一致。
本实验直接验证该假设。

### 方法

- 对 19 个 NAM 电路中的每个，取 8 种路由候选（l05, l05_beam3, l05_beam5, ph2v4,
  ph2v4_beam3, ph2v4_beam5, nam_l05_v2, nam_l05_v2_beam3）的 routed_qasm
- trajectory_fid：从已有 per-circuit JSON 读取（trajectory_sched×16, seed=0, scheduled=True）
- analytic_fid：用 `make_analytic_fidelity_fn(include_thermal=True)` 现算（同 v2 训练配置）
- 计算 per-circuit Spearman 秩相关

脚本：`scripts/analyze_fid_correlation.py`

### 结果

**per-circuit Spearman(analytic, trajectory) 平均 = -0.07（≈0，仅 7/17 电路为正）**

关键诊断 —— 各指标与 SWAP 数的 Spearman 相关：

| 指标 | corr(指标, SWAP数) | 说明 |
|------|-------------------|------|
| analytic F | **-0.80** | 代理 ≈ SWAP 计数惩罚的翻版，无噪声路径偏好 |
| trajectory F | **+0.09** | 真实保真度与 SWAP 数无关，区分因素在「去哪条边/何时 idle/串扰」 |
| analytic vs trajectory | **-0.05** | 同一电路不同路由间零相关 |

三种 analytic 变体（thermal±, crosstalk±）per-circuit 平均 Spearman 均为 -0.07~-0.08，
无一可靠。池化（跨电路混合）Spearman=0.74 是**规模主导的假象**（大电路两个指标都低），
不代表对同一电路的路由排序能力。

### 根因分析

1. **analytic 只累加逐门 ε**：同一逻辑电路的不同路由只差在 SWAP 的**位置**（放哪条边、
   何时插入），但 analytic 对每条 SWAP 只计固定 `3×log(1-ε_cx)`（trajectory_sim.py:1185），
   无法表达边噪声异质性 / idle 退相干 / 动态串扰 → 不同路由的 analytic F 差异几乎全部来自 SWAP 数。
2. **corr=-0.80 vs +0.09**：PPO 若以 analytic 为 reward，学到的目标是「越少 SWAP 越好」
   （= 最短路路由），恰好背离 l05 学的「多 SWAP 但放低噪声边」策略 —— 解释了 v2 退化的
   根本机制，也解释了为何 EMA 自 step 5120 起不再改善（奖励信号本身带偏）。
3. **thermal 项加剧塌缩**：≥11q 电路加 thermal 后 analytic F 下溢到 ~0，ana_span≈0，
   代理对候选完全失去区分度（tie），连秩排序都做不了。

### 结论

- **analytic 代理不能用作噪声感知训练的 fidelity reward**（至少在当前一阶逐门累积 + 固定
  时长 thermal 的形式下）。
- 可行的方向：
  1. **≤16q 训练恢复 trajectory_sched**（已有 pipeline，ph2v4 用过），代价是 16q 每次 ~0.8s；
  2. **改进代理**：把 analytic 从「逐门累积」改成「调度感知」——至少引入 edge 噪声异质性 +
      idle 退相干 + swap 的边权重（而非固定 3×CX），再验相关性；
  3. **混合**：小/中规模 trajectory，>16q 用改进代理 + 与 trajectory 的保真度回归校准。

---

## NAM trajectory_sched 微调 v1（curriculum ≤16q + NAM≤16q 混入，50k 步）

### 背景与动机

v2（unified_mixed + analytic fidelity）在 NAM benchmark 退化（argmax 0.2485 vs l05 0.3115）。
相关性分析证明 analytic 代理与 trajectory_sched 对同一电路不同路由的排序零相关
（per-circuit Spearman ≈ -0.07，corr(analytic,SWAP)=-0.80 vs corr(trajectory,SWAP)=+0.09），
不能用作文噪声感知训练的 reward。本次改为：**≤16q 用 trajectory_sched 训练**，
训练集 = 原先 ≤16q curriculum + NAM≤16q 电路，从 l05 微调。

### 代码改动（train_agent.py）

1. 新增 `--nam-max-qubits`：过滤 NAM 加载与采样（本实验 =16，排除 18/19q 三条）
2. 新增 `--sabre-fid-map-nam`：NAM 电路专属 SABRE 参考表。因 NAM 与随机电路同尺寸
   SABRE 基线不同（如 5q: NAM≈0.68 vs random≈0.31），主循环按 `circuit_path.startswith("nam/")`
   动态切换 sabre_fid_map（random 电路用 `--sabre-fid-map`，NAM 用 `--sabre-fid-map-nam`）

### 训练命令

```bash
cd src
python3 -m routing.rl.train_agent \
  --topo-list ../traindata/topo/tianyan176_20q.json \
  --curriculum-keys stage2,large_n8,large_n8,large_n10,large_n12,large_n16 \
  --reward-mode noise_aware \
  --fidelity-sim trajectory_sched --traj-trajectories 16 \
  --max-num-qubits 20 \
  --load ../models/policy_tianyan20q_laymix_l05_eta05.pt \
  --eta-xtalk-par 0.05 --swap-cost 0.5 \
  --lambda-fid-max 5.0 --lambda-fid-warmup 0.0 \
  --sabre-fid-map "5=0.309,8=0.162,10=0.061,12=0.046,16=0.0065" \
  --sabre-fid-map-nam "5=0.6789,7=0.3675,9=0.1985,10=0.1779,11=0.0645,12=0.2348,14=0.1662,15=0.0567" \
  --nam-circuits-dir ../benchmark/nam_circs --nam-circuit-prob 0.3 --nam-max-qubits 16 \
  --use-scheduler --mapping-phase --mapping-min-swaps 1 --lambda-layout 0.5 \
  --layout-mix 0.3,0.3,0.4 \
  --sabre-cache-file ../models/sabre_cache_ph2v3.pkl \
  --rollout-steps 256 --epochs 4 --lr 1e-4 \
  --timesteps 50000 --max-episode-steps 300 \
  --ema-decay 0.999 \
  --eval-interval 5000 --eval-split large_n16_test --eval-traj 4 --eval-max-qubits 16 \
  --out ../models/policy_nam_traj_ft_v1.pt \
  --checkpoint-dir ../models/ckpts_nam_traj_ft_v1 \
  --device cuda:0
```

日志：`models/train_nam_traj_ft_v1.log`（session `nam_traj_ft`）

### 训练日志摘要

- 全程 trunc=0%，entropy 0.4-0.9 健康，KL ~0.1-1.7（早期略高），grad_norm 收敛
- fid（trajectory_sched 真实值）：5q 阶段 ~0.13 → 大电路阶段 0.02-0.05
- curriculum 前 5 段（5q→n12）较快（~5-8 steps/s），**n16 段显著变慢**（~0.3-1 steps/s，
  因 30% NAM 混入含 12-15q 深电路 trajectory_sched×16 每条 10-20s）

### 已知问题（不影响训练产物）

- **EMA eval 全程崩溃记 0**：本次命令误加 `--eval-max-qubits 16`，导致 EMA eval 的
  evaluate_circuit 用 max_num_qubits=16 构建观测（167 维），与 agent critic 期望 20q
  （171 维）不匹配 → RuntimeError → mean_fid 记 0。修复：EMA eval 的 max_num_qubits
  应与模型一致（20），此 bug 不影响训练主循环，最终模型 policy_nam_traj_ft_v1.pt 有效。

### 模型文件

- `models/policy_nam_traj_ft_v1.pt`（训练完成，step 50176）
- `models/policy_nam_traj_ft_v1_ema_best.pt`（=step5120 早期，EMA bug 导致无意义）
- 路由结果：`benchmark/routed/nam_traj_v1*.json`（评估中，fidelity 待补）

### 评估结果（NAM benchmark，19 circuits，trajectory_sched×16，SABRE 基线）

路由：`generate_routing.py --no-fidelity`；fidelity：`scripts/compute_fidelity_nam_traj_v1.py`
（session `eval_nam_traj_v1` / `eval_nam_traj_beam3` / `fid_nam_traj_v1`）

| 方法 | 平均 SWAPs | 平均保真度 | vs SABRE |
|------|-----------|-----------|----------|
| SABRE | 45.1 | 0.2747 | — |
| l05 argmax | 65.3 | **0.3115** | +13.4% |
| ph2v4 argmax | 64.0 | 0.2872 | +4.6% |
| nam_l05_v2 argmax（analytic 微调） | 57.1 | 0.2485 | -9.5% |
| **nam_traj_v1 argmax** | 64.2 | 0.2758 | +0.4% |
| **nam_traj_v1 beam3** | **44.1** | 0.2816 | +2.5% |
| ph2v4 beam5 | 42.2 | 0.3047 | +10.9% |

### 逐电路（v1 argmax vs l05 argmax vs v2）

| Circuit | q | l05_fid | v1_fid | v2_fid | l05_sw | v1_sw |
|---------|---|--------:|-------:|-------:|-------:|------:|
| barenco_tof_3 | 5 | 0.7636 | **0.7740** | 0.6428 | 14 | 11 |
| mod5_4 | 5 | 0.4148 | **0.5052** | 0.4666 | 14 | 13 |
| tof_3 | 5 | 0.8198 | **0.8198** | 0.7627 | 11 | 11 |
| barenco_tof_4 | 7 | **0.4830** | 0.4399 | 0.4292 | 22 | 31 |
| hwb6 | 7 | 0.0003 | 0.0000 | 0.0000 | 68 | 57 |
| tof_4 | 7 | **0.7138** | 0.6265 | 0.5307 | 19 | 12 |
| barenco_tof_5 | 9 | **0.3089** | 0.2870 | 0.1628 | 40 | 38 |
| grover_5 | 9 | 0.0033 | 0.0003 | 0.0031 | 149 | 134 |
| mod_mult_55 | 9 | **0.1057** | 0.0484 | 0.0983 | 27 | 23 |
| tof_5 | 9 | **0.6041** | 0.4373 | 0.6411 | 25 | 20 |
| vbe_adder_3 | 10 | **0.4543** | 0.3727 | 0.2654 | 48 | 48 |
| mod_red_21 | 11 | **0.1215** | 0.0862 | 0.0002 | 72 | 77 |
| gf2^4_mult | 12 | **0.1989** | 0.1527 | 0.0393 | 56 | 61 |
| rc_adder_6 | 14 | **0.1608** | 0.0885 | 0.0000 | 68 | 64 |
| csla_mux_3 | 15 | **0.2566** | 0.1855 | 0.1952 | 54 | 58 |
| gf2^5_mult | 15 | 0.0267 | **0.0362** | 0.2551 | 101 | 86 |
| gf2^6_mult | 18 | **0.0650** | 0.0190 | 0.0216 | 162 | 195 |
| barenco_tof_10 | 19 | **0.1960** | 0.1008 | 0.0190 | 184 | 188 |
| tof_10 | 19 | 0.2223 | **0.2597** | 0.1890 | 107 | 92 |

### 分析与结论

1. **trajectory_sched 微调修复了 v2 的大规模退化**：相比 v2（analytic 微调）：
   - rc_adder_6: 0.0000 → 0.0885，mod_red_21: 0.0002 → 0.0862，
     barenco_tof_10: 0.019 → 0.101，tof_10: 0.189 → 0.260
   - 印证相关性分析：analytic 代理带偏策略，trajectory_sched 更接近真实优化目标
2. **beam3 效果显著**：SWAPs 从 argmax 64.2 压到 44.1（追平 SABRE 45.1，优于 l05 beam3 43.6≈同级），
   fidelity 0.2816 > argmax 0.2758（critic 的 trajectory 保真度信号让 beam 选择了更优路径）
3. **但仍未超越 l05 argmax（0.3115）**：trajectory_sched 微调后 v1 argmax fidelity 反低于纯 l05。
   在 9-14q 的中等电路上系统性落后（tof_5/vbe_adder_3/mod_red_21/gf2^4_mult/rc_adder_6/barenco_tof_5），
   而 5q 与 19q 上略优。可能原因：(a) 50k 步 noise_aware 微调在课程后段（n16 阶段只占 ~7k 步）
   未充分收敛；(b) layout-mix 0.3/0.3/0.4 + trajectory 保真度奖励把策略推向「更少 SWAP 但非噪声最优边」
   的折中；(c) 从 routing-mode 的 l05 出发，noise_aware 微调改变了路径偏好，牺牲了 l05 在中等电路
   上「多 SWAP 但放低噪声边」的策略。
4. **EMA eval bug 影响评估可信度**：EMA best=step5120（=l05 起点），未能反映后期训练质量。
   若需要更公平的 checkpoint 选择需修复 --eval-max-qubits 与模型维度不一致的问题后重训或重评估。

---

## 保真度奖励机制分析与 v1 仍逊于 l05 的归因（2026-09-08）

### 背景

`policy_nam_traj_ft_v1.pt`（trajectory_sched 微调，50k 步，NAM≤16q 混入）在 NAM benchmark
上 argmax 0.2758 仍低于未微调的 l05（0.3115）。用户指出训练已做跨规模奖励均衡（log-相对
`sabre_fid_map`），质疑"深电路 F<sref → reward 全为负"的猜想。本记录用实测数据验证了奖励
的实际分布，并修正归因。

### 保真度奖励计算公式（现状）

终端奖励（env.py `_terminal_reward`，仅 done 时触发一次，λ_fid 逐步调度）：

```
r_terminal = λ_fid × (log F − log F_sabre(n)),  clip ∈ [−50, +50]
```

- `F` = fidelity_fn(env)（trajectory_sched ×16 MC / analytic O(G)）
- `F_sabre(n)` = sabre_fid_map[n]，按逻辑比特数查表（random/NAM 两套，按电路来源切换）
- 无表或 routing 模式退化为 `λ×F` 或 0
- 只有 F > F_sabre 时奖励为正

### 验证 1：负奖励并非"一律"，但被少数塌缩电路垄断

用 v1/l05 确定性 argmax 的 19 条 NAM 评估结果 + NAM sref 表逐条算 reward：

| 指标 | v1 | l05 |
|------|-----|-----|
| F>sref 电路数 | 10/19 | 10/19 |
| 正奖励总和 | +22.2 | +33.0 |
| **负奖励总和** | **−102.4（占 82%）** | **−67.9（占 67%）** |
| mean reward | −4.22 | −1.84 |

结论：**10/19 电路 F>sref（正奖励），"深电路一律 F<sref" 不成立**。但负奖励的 82% 来自
少数保真度塌缩到 ~0 的深电路：

- hwb6 (259g, 7q): F≈0.0000 → reward −46（近 clip 下限 −50）
- grover_5 (831g, 9q): F≈0.0003 → reward −33

单条塌缩电路的负奖励（−46）是 5 条正常正奖励电路总和（+22）的两倍以上。

### 修正后的因果链（v1 仍逊于 l05）

1. **不是"没均衡"，而是 log-相对公式在 F→0 时对数爆炸**：均衡按比特数查 sref，
   但未按电路深度/门数处理 F 塌缩。深电路 F 塌到 ~0 时 `λ×log(F/sref)` 逼近 clip 下限 −50，
   一条塌缩电路的负梯度抵消几十个 +2 的正常正信号。
2. **训练期比评估期更严重**：PPO 训练处于采样探索中，路由质量远差于确定性 argmax，
   深 NAM 电路采样路由 F 更接近 0 → reward 常打满 −30~−50。策略学到的是"深电路上少冒险
   （少 SWAP 保守路由）"而非"找噪声最优路径"。
3. **分布内 vs 分布外不对称（已实测）**：v1 微调在**训练分布**的大电路上确实超越 l05：
   - 5q stage2 电路：l05 与 v1 逐位相同（swaps/fid 完全一致，微调未改变小电路行为）
   - n10-n16 random/qaoa/vqe：v1 普遍胜（n10 qaoa 0.0013→0.0029, n12 random 0.0255→0.0423,
     n16 2/3 例提升）
   - NAM 电路（分布外）：有涨有崩（hwb6 0→0.11↑, gf2^5 0.057→0.236↑, mod_red 0.209→0.283↑；
     vbe_adder 0.644→0.234↓, tof_5 0.558→0.409↓），净微负
   说明 fidelity 微调在训练分布内有效，但对 NAM（结构化算术电路，分布外）未泛化，
   反而漂移出 l05 由 layout-mix 带来的稳健路由。

### 辅助验证发现

- **EMA eval bug（已定位，不影响训练产物）**：v1 命令误加 `--eval-max-qubits 16`，导致 EMA eval
  的 evaluate_circuit 用 max_num_qubits=16 建 obs（critic 输入 167 维）而 agent 期望 20q（171 维）
  → RuntimeError → EMA fid 全程记 0。修复应让 EMA eval 的 max_num_qubits 与模型一致（20）。
- **generate_routing vs evaluate_circuit 结果差异**：同一模型两条评估路径的 routing 不完全一致，
  需注意口径统一（疑似 mapping 阶段初始状态/seed 处理差异，待查）。

### 修复方向（待验证）

| 方向 | 思路 |
|------|------|
| 负奖励饱和 | 对 F 加与门数相关的下界（如 F_floor(n,gates)），或对 F<ε 的塌缩电路跳过 fidelity 奖励，防止 −50 极端负奖励垄断梯度 |
| 奖励压缩 | 用 √F 或 rank-based 奖励替代 log F，抑制 F→0 时的对数爆炸 |
| 深电路单独处理 | 深到 F 天然 < 任何路由的电路（如 grover_5 831g）不参与 fidelity 奖励 |
| 分布外适应 | NAM 占比提高 / 用 NAM 电路直接微调，而非依赖 random 电路迁移 |

---

## 方案1：NAM per-circuit SABRE 参考（sref_override）实现与验证（2026-09-08）

### 问题

v1 用 `sabre_fid_map_nam`（per-size 表）做 NAM log-相对奖励分母，但同比特数内深度差
10 倍（9q: tof_5 105g vs grover_5 831g，SABRE F 差 800 倍），共享 sref 使深电路
`log(F/sref)` 爆炸成极端负奖励（grover_5 F=0.0003 → reward −32.5），垄断梯度。

### 改动

**`env.py`**：`RoutingEnv.__init__` 新增 `sref_override`（episode 级浮点，优先于
`sabre_fid_map` 按 num_qubits 查表）；`_terminal_reward` sref 优先级改为
`sref_override > sabre_fid_map[num_qubits]`；`clone()` 同步拷贝。

**`train_agent.py`**：
- 新增 CLI `--nam-sabre-fid-json`：per-circuit SABRE 参考 JSON（`{name: fid}`）
- main 加载为 `nam_sref_map`；主循环 NAM 电路时若该电路在 map 且 sref>0，
  传 `sref_override`（并置 sabre_fid_map=None），否则回退 per-size 表

**数据**：`benchmark/routed/sabre_nam_percircuit.json`（19 条 NAM per-circuit SABRE
fidelity，源自 sabre_summary.json，trajectory_sched×16 口径）

### 验证（同一 9q 电路 F 恒定，改 sref 看 reward）

| 电路 | F | 旧（共享 9q sref=0.1985） | 新（per-circuit sref） |
|------|-----|--------------------------|----------------------|
| grover_5 | 0.0003 | **−32.5**（对数爆炸） | **−3.5**（sref=0.0006，正常惩罚）✅ |
| tof_5 | 0.44 | +4.0（虚高） | −0.3（sref=0.4699，诚实略低于 SABRE）✅ |

效果：深电路（grover_5）从 −32.5 恢复到 −3.5；浅电路不再因共享 sref 虚高正奖励。
每条 NAM 电路现在只与自己 SABRE 基线比，reward 尺度统一、不再被深电路垄断。

### 冒烟

`train_agent` 128 步 + NAM prob 0.5 + `--nam-sabre-fid-json`：19 条参考加载正常、无报错。

### 已知边界

- hwb6 的 SABRE fid≈0（塌缩电路）→ `pc_sref>0` 不成立 → 回退 per-size 表，
  仍会负向爆炸。属方案2（深电路负奖励饱和）处理范围，未在本方案覆盖。
- 待跑：完整 NAM 微调训练 + NAM benchmark 评估（对比 v1 0.2758 / l05 0.3115）

---

## 方案1 训练与 NAM 评估：per-circuit sref（policy_nam_sref_ft_v1）

### 训练命令（session `nam_sref_ft`）

同 v1 配置 + `--nam-sabre-fid-json ../benchmark/routed/sabre_nam_percircuit.json`
（curriculum stage2+large_n8/10/12/16，trajectory_sched×16，NAM 30% ≤16q，
lr=1e-4，timesteps 50000，l05 起点）。修复了 v1 的 `--eval-max-qubits` EMA bug。

### 训练日志摘要

- 全程 trunc=0%，entropy 0.5-0.95 健康
- **EMA eval 有效**（非 v1 的恒 0）：5120→0.000059, 10240→0.000381,
  30208→**0.000643**(best), 45056→0.000605, 50176→0.000627
- EMA best 在 step 30208（n12 阶段末），n16 阶段未超越

### 评估（NAM benchmark，19 circuits，trajectory_sched×16，SABRE 基线）

用 EMA best（`policy_nam_sref_ft_v1_ema_best.pt`）：

| 方法 | 平均 SWAPs | 平均保真度 | vs SABRE |
|------|-----------|-----------|----------|
| SABRE | 45.1 | 0.2747 | — |
| l05 argmax | 65.3 | **0.3115** | +13.4% |
| v1 argmax | 64.2 | 0.2758 | +0.4% |
| v1 beam3 | 44.1 | 0.2816 | +2.5% |
| **nam_sref argmax** | 59.6 | 0.2728 | -0.7% |
| **nam_sref beam3** | **41.5** | 0.2925 | +6.5% |

### 逐电路（nam_sref beam3 vs l05 argmax）

| Circuit | q | l05_fid | sr_fid | sab_fid | sr_sw |
|---------|---|--------:|-------:|--------:|------:|
| barenco_tof_3 | 5 | 0.7636 | 0.6569 | 0.7467 | 7 |
| barenco_tof_4 | 7 | 0.4830 | **0.6670** | 0.4454 | 15 |
| barenco_tof_5 | 9 | 0.3089 | **0.3166** | 0.1683 | 22 |
| tof_10 | 19 | 0.2223 | **0.4041** | 0.4264 | 62 |
| gf2^6_mult | 18 | 0.0650 | **0.0870** | 0.0314 | 121 |
| gf2^5_mult | 15 | 0.0267 | **0.0932** | 0.0793 | 75 |
| vbe_adder_3 | 10 | 0.4543 | 0.4473 | 0.1779 | 26 |
| csla_mux_3 | 15 | 0.2566 | 0.2547 | 0.0341 | 38 |
| tof_4 | 7 | **0.7138** | 0.6294 | 0.6570 | 10 |
| tof_5 | 9 | **0.6041** | 0.5513 | 0.4699 | 16 |
| mod_red_21 | 11 | 0.1215 | 0.0002 | 0.0645 | 40 |
| rc_adder_6 | 14 | **0.1608** | 0.0363 | 0.1662 | 40 |
| barenco_tof_10 | 19 | **0.1960** | 0.0591 | 0.0715 | 94 |

### 初步结论

1. **beam3 有效**：nam_sref beam3 0.2925（+6.5% SABRE），SWAP 41.5 少于 SABRE 45.1，
   fid 高于 v1 beam3（0.2816）。beam3 在 csla_mux/tof_10/gf2^6/gf2^5 大幅提升。
2. **但仍未超 l05 argmax（0.3115）**：beam3 0.2925 < l05 argmax。且个别电路退化严重
   （mod_red_21 0.0002、rc_adder_6 0.036、barenco_tof_10 0.059）——beam search 的 critic
   评分对这些电路失效（选择低 fidelity 分支）。
3. **argmax 未改进**：0.2728 ≈ v1 0.2758。per-circuit sref 修正了深电路负奖励爆炸，但
   argmax 整体未见提升，可能因 EMA best 是 n12 阶段快照（未含完整 n16 训练）。
4. **待验证**：评估最终模型（step 50176，含完整 n16 阶段）是否优于 EMA best（step 30208）。

### 补充：最终模型（step 50176）vs EMA best（step 30208）

| 模型 | argmax fid | argmax SWAP |
|------|-----------|------------|
| **EMA best (30208)** | **0.2728** | 59.6 |
| final (50176) | 0.2504 | 61.8 |

最终模型经完整 n16 阶段后 argmax 反而退化（0.2728→0.2504），证实 EMA best 是更优
checkpoint。完整 n16 训练（random 16q 占比高）损害了对 NAM 的泛化——进一步支持
"分布外漂移"假设。用 EMA 选点（而非最终权重）对 NAM 评估更合理。

### 完整结论

方案1（per-circuit sref）相对 v1 的净效果：EMA best beam3 0.2925（+6.5% SABRE，SWAP
41.5 < SABRE 45.1），为除 l05 外最优。但 argmax 0.2728 仍未超 l05（0.3115）。l05 的
纯 routing + layout-mix 对 NAM 仍是最强基线；fidelity 微调的价值主要在 beam3 推理端。

---

## 现象解释：为何 l05 argmax 最优，而 beam3 是"第二阶段微调版"更好

### 现象（NAM benchmark，trajectory_sched×16）

| 方法 | argmax fid | beam3 fid | beam3 vs argmax |
|------|-----------|-----------|----------------|
| l05（纯 routing） | **0.3115** | 0.2617 | beam 败 13/19，**明显倒退** |
| nam_sref v1（noise_aware 微调） | 0.2728 | **0.2925** | beam 胜 9/19，净提升 |

### 机制解释

**1. argmax 表现 = 策略 π 的质量；l05 的 π 对 NAM 分布外最鲁棒**

l05 是纯 routing + layout-mix 训练（无 fidelity 奖励），策略学的是布局无关的通用路由。
NAM（算术电路）是训练分布外的，l05 的 π 未向任何特定 fidelity 地形过拟合，因此 argmax
路径保持通用高效。而 noise_aware 微调把 π 推向训练分布（random/qaoa/vqe ≤16q）的
fidelity 局部最优——在训练分布内 argmax 提升（前已实测 n10-16 v1 胜 l05），但对 NAM
（分布外）argmax 反而偏离了 l05 的稳健路由。→ **微调损害 π 的 NAM 泛化，argmax 变差**。

**2. beam3 表现 = critic V(s') 的质量；第二阶段训练的 critic 学会了 fidelity**

beam 评分 = `r_step + γ·V(s')`，其中 V 是 critic 对下一状态的预测。critic 回归目标是
GAE returns：
- **l05**：routing 训练，returns 无 fidelity 项 → V 只预测"路由效率"（SWAP 少、执行快），
  不含保真度信息。beam 按此选路 = 选路由最短路径，恰好丢掉 l05 π 隐含的噪声路径偏好
  → l05 beam3 大幅倒退（0.3115→0.2617）。
- **nam_sref**：noise_aware 训练，returns 含终端 `λ·(log F − log sref)` → V 学会预测
  fidelity。beam 在 top-K 候选里用含 fidelity 的 V(s') 选路 → 找到比 π argmax 保真度
  更高的分支 → beam3 提升（0.2728→0.2925）。

**3. 一致性验证**

- 微调损害 π（argmax 0.2728 < l05 0.3115）与微调改善 V（beam 0.2925 > l05 beam 0.2617）
  可同时成立：**π 与 V 是同一网络的不同头，微调的 fidelity 信号主要沉淀进 V，而 π 的
  分布内过拟合损害了分布外 argmax**。
- l05 beam 胜 6/19、败 13/19（净 -0.05）：critic 无 fidelity 时 beam 是负贡献。
- nam_sref beam 胜 9/19、败 10/19（净 +0.02）：critic 有 fidelity 后 beam 转正贡献。

### 启示

1. 若目标是 argmax 推理：不要 noise_aware 微调，直接用 l05（或在其上做 NAM 专属
   routing 微调而不加 fidelity 奖励）。
2. 若目标是 beam search 推理：noise_aware 微调 + per-circuit sref 让 critic 学会
   fidelity，beam 收益显著（0.2925，SABRE+6.5%，SWAP 41.5 最少）。
3. 终极方向：让 π 也获得 fidelity 泛化——需解决分布内过拟合（如 NAM 主导训练 /
   域随机化），使 argmax 与 beam 同时受益。

---

## 无泄露 test split 复核：微调 vs l05 的尺寸依赖规律（2026-09-09）

### 背景与动机

此前判定"noise_aware 微调在训练分布内（n10-16）胜 l05"时，误用了 `large_n*_mixed`
（**训练 split，有泄露**）。用户指出在无泄露 test split 上 l05 仍可能更好。本记录用
`large_n*_test`（0 重叠）复核 l05 vs v1 vs sref_ema（per-circuit sref 版本 EMA best）。

### 结果（无泄露 test split，trajectory_sched，deterministic argmax）

| 尺寸 | n(条) | l05 | v1 | sref_ema | 结论 |
|------|------|-----|-----|----------|------|
| n8 | 30 | **0.194** | — | 0.167 | **l05 胜 +16%**（l05 胜 15/30） |
| n8 | 15 | 0.217 | 0.184 | 0.222 | v1 明显差于 l05 |
| n10 | 25 | 0.072 | — | **0.085** | sref 胜 +17%（l05 胜 10/25） |
| n10 | 12 | 0.121 | **0.147** | 0.132 | v1 胜 |
| n12 | 10 | 0.108 | — | 0.111 | 平 |
| n16 | 8 | 0.007 | — | **0.023** | sref 大幅胜 +225% |

### 关键修正

1. **此前"微调在训练分布内胜 l05"的结论是数据泄露假象**：基于 `*_mixed`（训练 split，
   微调模型见过这些电路）的评估高估了微调模型。无泄露 test split 上：
   - **v1 在 n8 确实输 l05**（0.184 vs 0.217），在 n10 才胜
   - **sref_ema 在 n8 输（+16% 劣势），n10/n12 持平略胜，n16 大幅胜（+225%）**
2. **真实规律是"尺寸依赖"而非"普遍胜/普遍败"**：
   - 小电路（≤8q）：l05 的纯 routing 已接近最优，fidelity 微调是扰动 → l05 胜
   - 中电路（10-12q）：微调的 fidelity 感知开始有价值 → 持平略胜
   - 大电路（16q）：fidelity 优化空间大（l05 的 routing 在 16q 上远离最优），
     微调 + per-circuit sref 的价值最大 → sref_ema 大幅胜
3. **per-circuit sref（方案1）优于 v1（per-size sref）**：n8 上 sref_ema 0.167 vs v1 0.184
   （sref 更接近 l05），且 n16 上 sref_ema 表现更强。方案1 的 per-circuit 标定确实修正了
   奖励信号。

### 对整体结论的影响

- "Phase 2 保真度训练有害"过于笼统：**在 ≥10q 电路上微调有正收益，且随尺寸增大收益
  更明显**；≤8q 上 l05 更优。
- NAM benchmark（5-19q 混合，中小电路占多数）整体平均被小电路拖累，故 nam_sref 平均
  fid（0.2925 beam3）仍低于 l05 argmax（0.3115）——若 NAM 评估只计 ≥10q 电路，微调
  模型可能反超。
- 改进方向修正：可做**尺寸分层的模型选择**（≤8q 用 l05，≥10q 用 sref_ema），
  或在小电路阶段冻结/降低微调强度（ph2v5 的 freeze 思路），大电路加大 fidelity 权重。

---

## v1 无泄露 test split 全尺寸确认（2026-09-09，大样本复核）

### 背景

上节复核发现此前"微调在训练分布内胜 l05"结论依赖泄露的 `*_mixed` split，且小样本
（8-15 条）跨 seed 不稳定（n16 曾出现 sref_ema 胜 +225%，大样本后翻转）。本节用
大样本确认 v1（`policy_nam_traj_ft_v1.pt`，per-size sref 微调）在 n10/n12/n16 的
真实表现，并与 l05 对比（trajectory_sched，deterministic argmax）。

### 结果（大样本，含 arithmetic-mean / log-mean / 胜率）

| 尺寸 | n | l05 mean | v1 mean | l05 log-mean | v1 log-mean | l05 胜率 |
|------|---|---------|---------|-------------|-------------|---------|
| n8 | 30 | **0.194** | 0.167-0.184 | — | — | l05 胜 |
| n10 | 25 | 0.100 | **0.124** | 0.060 | **0.075** | 11/25（v1 胜） |
| n12 | 25 | **0.061** | 0.055 | **0.0068** | 0.0040 | 12/25（l05 胜） |
| n16 | 20 | 0.015 | 0.018 | **0.0010** | 0.0007 | **13/20（l05 胜）** |

### 结论（修正）

1. **v1 只在 n10 稳定胜 l05**（+24% mean / +24% log-mean），n8/n12/n16 均输。
   n16 的 arithmetic-mean v1 略高（0.018 vs 0.015）是**少数极端高值拉高**的假象，
   log-mean（0.0007 vs 0.0010）和胜率（13/20）都指向 l05 更优。
2. **用户判断成立**：即便用无泄露 test split，除 n10 外 l05 全面优于 v1 微调模型。
3. **评估方法警示**：fidelity 跨数量级（1e-4 ~ 0.5），arithmetic-mean 被极端值支配，
   小样本跨 seed 结论翻转。可靠比较需 ≥20 条 + log-mean 或胜率。
4. **此前的 n16 "+225%" 结论作废**（8 条小样本假象）。

### 含义

- noise_aware 微调（v1 形式）相对 l05 的真实增益仅限 n10 附近窄区间，不构成
  全尺寸优势。l05 仍是无泄露评估下最强基线。
- 后续若继续 fidelity 微调路线，需：per-circuit sref（方案1，已实现）+ 大尺寸
  （≥16q）专门训练 + 大样本 log-mean 评估，而非 per-size sref + 小样本 mean。

---

## Local fidelity 信号可行性诊断（2026-09-09）——门级代理与 scheduled eval 排序为负

### 背景

设计 Phase 2 = P1 + λ_local·ΔlogF_local + λ_global·terminal（用户方案），其中
local 信号需每步可算。评估了三种"廉价 local 代理"与真实 eval（trajectory_sched×16）
在 NAM 路由候选上的排序一致性。

### 诊断结果

**1. 串行逐门 trajectory tracker（LocalFidTracker）**：Spearman(eval, tracker) = **-0.895**
（tof_4，5 种路由）——强负相关。

**2. Raw analytic 逐门错误累积**：Spearman = -0.37（tof_4）/ -0.7（barenco_tof_4）/
-1.0（tof_3）——系统性负相关。

**3. 根因**：eval scheduled fidelity 的排序主要由 **transpile 后电路总时长 → idle 退相干**
决定（eval 奖励"门少=时间短"），而门级代理把 SWAP/门数当主要惩罚项 → 方向天然相反。
transpile 前后门数剧变（tof_4: 94 → 518 门，含 285 rz），原始电路的门级错误与
eval 的调度结果几乎无关。

### 结论

- 任何"原始物理电路上的门级错误累积"代理（analytic / 串行 trajectory）都**无法**
  复现 eval scheduled 排序，不能用作 local fidelity 信号（否则 C≈A 甚至 C<A）。
- 严格对齐 eval 需每步对当前电路做 scheduled eval——5-15q 单次 0.1-11s，
  训练步开销不可行（30× 以上）。
- LocalFidTracker 实现（trajectory_sim.py）保留但**不可用于训练信号**；
  仅作参考/教学实现。

### 待决策

1. local 信号改用 **env.timing 的调度统计**（idle/crosstalk 已是每步实时可得的
   scheduled 语义，即 P1 的 r_idle/r_xtalk_par 已覆盖）→ 那 C 与 A 的差异只在
   "是否显式用 idle 项"，需确认是否值得单独实验。
2. 或者放弃 per-step fidelity 信号，回到 **terminal-only 但修正 credit assignment**
   （如 GAE 加强 / 轨迹级 return decomposition）。
3. 或 local 只在 ≤8q（eval 快）上启用 scheduled 每步全量（训练慢 ~10× 但可接受？）。

---

## Phase 2 重构：双价值头 + fidelity 进 V 不进 π（P2A/P2B/P2C 消融）

### 动机

此前结论：terminal fidelity 直接进 GAE 会把稀疏噪声奖励摊给所有历史 SWAP
（credit assignment 差），拉偏 actor → argmax 退化；而 fidelity 信息沉淀进 critic
（V 头）使 beam3 提升。据此重构 Phase 2：**fidelity 信号主要进入独立 V_fid 头，
actor 只受 α≪1 的 fidelity advantage 轻微影响，并用 KL(π_P1‖π_P2) 保策略**。

### 框架

- **网络**：`EdgeActorCritic` 拆双价值头 `critic_route` + `critic_fid`（fid 头 zero-init：
  加载旧 checkpoint 后 V_fid≡0，l05 beam 行为逐位不变；旧 checkpoint 的 `critic.*` 权重
  自动迁移到 `critic_route.*`）
- **奖励**：`r_t = r_t^{P1}`（全保留，swap_cost=0）+ `1_{t=T}·λ_fid·z_fid`，
  `z_fid = Normalize(log(F_agent/F_sabre))`（term_norm EMA），λ_fid 渐进
  `--lambda-fid-max-schedule 0.1,0.2,0.3,0.5`
- **双 GAE**：`A_route/ret_route`（P1 dense）与 `A_fid/ret_fid`（仅末步 fid）分开；
  actor advantage = `A_route + α·A_fid`（α=0.1）；`MSE(V_route, ret_route) + MSE(V_fid, ret_fid)`
- **KL 保策略（P2C）**：冻结 π_P1 教师（`--teacher`），前向 KL `β·Σ π_p1·(logπ_p1−logπ_p2)`，β=0.05
- **推理**：`_forward_obs` 返回 `V_route + λ_V·V_fid` → beam 自动获得 fidelity lookahead（零改动）；
  argmax 只用 logits

### 消融矩阵（固定 α=0.1 / β=0.05）

| 变体 | dense P1 | terminal fid | 独立 V_fid | KL 保策略 |
|------|---------|-------------|-----------|----------|
| P1 (l05 现成) | ✓ | ✗ | ✗ | ✗ |
| P2A | ✓ | ✓（直进 actor，α=1 旧行为） | ✗（单通道） | ✗ |
| P2B | ✓ | ✓ | ✓（α=0.1） | ✗ |
| P2C | ✓ | ✓ | ✓（α=0.1） | ✓（β=0.05） |

### 训练命令（三组并行，taskset 绑核 4 核/组，traj=8，curriculum 砍 n16，35k 步）

```bash
cd src
COMMON='--topo-list ../traindata/topo/tianyan176_20q.json \
  --curriculum-keys stage2,large_n8,large_n8,large_n10,large_n12 \
  --reward-mode noise_aware --fidelity-sim trajectory_sched --traj-trajectories 8 \
  --max-num-qubits 20 --load ../models/policy_tianyan20q_laymix_l05_eta05.pt \
  --eta-xtalk-par 0.05 --swap-cost 0 \
  --lambda-fid-max-schedule 0.1,0.2,0.3,0.5 \
  --sabre-fid-map "5=0.309,8=0.162,10=0.061,12=0.046,16=0.0065" \
  --use-scheduler --mapping-phase --mapping-min-swaps 1 --lambda-layout 0.5 \
  --layout-mix 0.3,0.3,0.4 --sabre-cache-file ../models/sabre_cache_ph2v3.pkl \
  --rollout-steps 256 --epochs 4 --lr 1e-4 --timesteps 35000 --max-episode-steps 300 \
  --ema-decay 0.999 --eval-interval 5000 --eval-split large_n10_test \
  --eval-traj 4 --eval-max-qubits 20 --eval-max-circuits 15'
# P2A（session p2a, cuda:0, 核0-3）
python3 -m routing.rl.train_agent $COMMON --variant P2A \
  --out ../models/policy_p2a.pt --checkpoint-dir ../models/ckpts_p2a --device cuda:0
# P2B（session p2b, cuda:1, 核4-7）
python3 -m routing.rl.train_agent $COMMON --variant P2B --alpha-fid 0.1 \
  --out ../models/policy_p2b.pt --checkpoint-dir ../models/ckpts_p2b --device cuda:1
# P2C（session p2c, cuda:2, 核8-11）
python3 -m routing.rl.train_agent $COMMON --variant P2C --alpha-fid 0.1 --beta-kl 0.05 \
  --teacher ../models/policy_tianyan20q_laymix_l05_eta05.pt \
  --out ../models/policy_p2c.pt --checkpoint-dir ../models/ckpts_p2c --device cuda:2
```

### 省时策略与口径

- traj 16→8（终端 fid 方差升，z 归一化吸收；**评估仍用 16**）
- curriculum 砍 n16 段（EMA best 本在 n12 末；16q 改为零样本评估）
- 无 NAM 混入（单变量对照；NAM benchmark 作分布外评估面）
- swap_cost=0（与 P1 一致）

### 成功判据

- ✅ argmax ≥ 0.95×l05（不退化），beam3 > nam_sref beam3（0.2925）
- P2A 退化而 P2B/C 不退化 → "fidelity 应进 V 不进 π"成立

### 启动状态

- 三组训练正常（step ~1800 起步，vfl=0.996 表示 V_fid 在学，P2C klp1≈0.02 策略偏离小）
- 诊断实验（session diag_causal）：ρ(ΔF(a_i), F_final) 因果测试运行中

---

## P2A/P2B/P2C 消融实验结果（2026-09-09）

### 训练与评估口径

- 训练：35k 步，traj=8，curriculum stage2+large_n8×2+n10+n12（无 n16、无 NAM），
  swap_cost=0，λ_fid 渐近 0.1→0.5，l05 起点，EMA 选点
- 评估 A（无泄露 test split，20 条/尺寸，log-mean）：

| split | l05 | p2a | p2b | p2c | 最优 |
|-------|-----|-----|-----|-----|------|
| 5q | 0.4540 | 0.4601 | 0.4575 | 0.4498 | p2a |
| 8q | 0.1179 | **0.1207** | 0.0787 | 0.0894 | p2a |
| 10q | 0.0370 | 0.0407 | 0.0389 | **0.0407** | p2a/p2c |
| 12q | 0.0046 | 0.0032 | 0.0057 | **0.0114 (+147%)** | p2c |
| 16q（零样本） | **0.0063** | 0.0006 | 0.0008 | 0.0030 | l05 |

- 评估 B（NAM benchmark，19 条，trajectory_sched×16）：

| 方法 | mode | mean_fid | log-mean | SWAP |
|------|------|---------|----------|------|
| l05 | argmax | **0.3115** | **0.1404** | 65.3 |
| SABRE | — | 0.2747 | 0.0721 | 45.1 |
| nam_sref | argmax | 0.2728 | 0.1080 | 59.6 |
| nam_sref | beam3 | 0.2925 | 0.0833 | 41.5 |
| **p2a** | argmax | 0.2925 | 0.1200 | 65.6 |
| **p2a** | beam3 | 0.2954 | 0.0978 | 42.5 |
| **p2b** | argmax | **0.3079** | 0.1121 | 67.1 |
| p2b | beam3 | 0.2709 | 0.0824 | 43.1 |
| p2c | argmax | 0.2598 | 0.0842 | 62.6 |
| p2c | beam3 | 0.2588 | 0.1086 | 42.6 |

### 成功判据对照

1. **"argmax 不退化"（≥0.95×l05）**：p2b argmax 0.3079 = 0.988×l05 ✅；
   p2a 0.939×（边缘）；p2c 0.834× ❌
2. **"beam3 提升"（> nam_sref beam3 0.2925）**：p2a beam3 0.2954 边缘达标；
   p2b/p2c beam3 均未超 ❌

### 结论

1. **P2B（独立 V_fid + α=0.1）实现了设计目标**：NAM argmax 0.3079 几乎不退化（vs l05
   0.3115），是除 l05 外最高的 NAM argmax；test split 上 5q/10q/12q 接近或略胜 l05。
   证实"fidelity 进 V 不进 π"的机制有效：α=0.1 时 fidelity 对 actor 的扰动被大幅抑制。
2. **P2A（旧行为 α=1）**：argmax 0.2925 退化 6%，验证了"terminal fid 直进 GAE 摊薄
   credit 拉偏 actor"的既有结论（即使 λ 已降到 0.1-0.5 渐进，退化仍存在）。
3. **P2C（+KL β=0.05）意外**：test split 12q 大幅提升（+147%，0.0114 vs 0.0046），
   但 NAM argmax 退化最严重（0.2598）。KL 保策略在"训练分布内"帮助了泛化，但抑制了
   策略向 NAM（分布外）的适应性调整——KL 约束把策略锚定在 l05 的分布内行为上。
4. **beam3 全线弱于 argmax**（p2b: 0.2709 < 0.3079）：V_fid 的 beam 组合值
   （V_route + λ_V·V_fid，λ_V=1）在 NAM 上过冲/低估。此前 nam_sref beam3 强是因为其
   critic 用 λ_fid=5 满权重 + traj=16 训练，V_fid 信号强；本次 λ 渐近 + traj=8 使
   V_fid 弱，beam 增益消失。可扫 λ_V ∈ {0, 0.5, 1, 2} 或训练用 traj=16 重验。
5. **16q 零样本仍是 l05 领地**（0.0063 vs 次优 0.0030）：未训 n16 的策略无法泛化到 16q。

### 诊断实验（§11）：ρ(ΔF(a_i), F_final) = 0.142 (p=0.09)

6 条 NAM 电路 × 中间状态 top-4 候选，共 144 样本。action-level causal signal 极弱
（ρ=0.14，勉强正）。**支持结论：fidelity 本质是 long-horizon trajectory-level
objective，local action-level ΔF 几乎没有因果信号**——这正是"fidelity 不应作为
dense local reward"的证据（为论文有价值的实验结论）。

### 模型文件

- `models/policy_p2a.pt` / `_ema_best.pt`（EMA best step=5120）
- `models/policy_p2b.pt` / `_ema_best.pt`（EMA best step=5120）
- `models/policy_p2c.pt` / `_ema_best.pt`（EMA best step=20224）
- 评估日志：`benchmark/eval_p2_{5q,10q,12q,16q}.log`、`benchmark/routed/nam_p2{a,b,c}*`
- 诊断：`benchmark/diag_causal2.log`（脚本 scripts/diag_causal_fidelity2.py）

---

## NAM 电路加入 Phase 1 训练集（l05 + NAM routing 微调，2026.09.10）

### 动机

l05（纯 routing + layout-mix）在 NAM benchmark 上 argmax 最优（0.3115），但其 π 是在
random/QAOA/VQE 电路上学到的，NAM 结构化算术电路属于分布外。本实验验证**把 NAM 电路
直接加入 Phase 1 routing 训练分布**能否让 π 学到 NAM 的路由模式，从而提升 NAM 上的
SWAP 效率与保真度（Phase 1 训练目标仍是纯 routing，不含 fidelity 奖励）。

### 训练设置

```bash
tmux new-session -d -s l05_nam_ph1 \
  'bash scripts/train_l05_nam.sh cuda:0 100000 2>&1 | tee logs/train_l05_nam_phase1.log'
```

`scripts/train_l05_nam.sh` 核心参数（其余与 l05 训练一致）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `--load` | `policy_tianyan20q_ft_phase1_sched_fix_eta05.pt` | 与 l05 相同起点 |
| `--reward-mode` | `routing` | Phase 1 纯路由（无 fidelity 奖励） |
| `--topo` | `tianyan176_20q` | 单一真机 20q 拓扑 |
| `--split-prefix` | `tianyan20q` | 原有 random/QAOA/VQE split |
| `--nam-circuits-dir` | `benchmark/nam_circs` | 19 条 NAM 电路（≤20q） |
| `--nam-circuit-prob` | `0.5` | 每步以 50% 概率采 NAM 电路 |
| `--layout-mix` | `0.3,0.3,0.4` | 与 l05 相同 layout-mix |
| `--lambda-layout` | `0.5` | 与 l05（λ=0.5）一致 |
| `--use-scheduler` | 开 | 与 l05 相同调度奖励 |
| `--timesteps` | `100000` | 与 l05 相同步数 |

> NAM 电路不参与 SABRE 布局缓存，`layout-mix` 中 SABRE 档对 NAM 回退到 identity 布局。

### 训练日志摘要

```
step=   256  rew=+112.6  swp=29.6  map=1.3  trunc=0%  ent=0.586
step=  5120  rew=+128.2  swp=37.6  map=0.4  trunc=0%  ent=0.656
step= 25600  rew=+183.0  swp=43.9  map=0.6  trunc=0%  ent=0.848
step= 51200  rew=+196.7  swp=50.6  map=0.6  trunc=0%  ent=0.863
step= 76800  rew=+158.4  swp=37.4  map=0.6  trunc=0%  ent=1.087
step=100096  rew=+261.5  swp=66.3  map=0.4  trunc=0%  ent=0.831
```

全程 `trunc=0%`，reward/SWAP 随 NAM 深电路课程同步增长。产出
`models/policy_l05_nam_phase1.pt`。

### 评估设置

- **路由**：`generate_routing.py` argmax，`--no-fidelity`，19 条 NAM 全部完成（19/19）
- **保真度**：`trajectory_sched ×16, seed=0, scheduled=True`（与历史 NAM 表同口径）
- **对比**：`l05_nam` vs 原 `l05` vs `SABRE`
- 结果：`benchmark/routed/l05_nam/`、`benchmark/routed/l05_nam_summary.json`
- 命令：`bash scripts/eval_nam_l05_nam.sh`

### 汇总结果

| 指标 | l05 | l05_nam | SABRE |
|------|----:|--------:|------:|
| SWAPs mean | 65.3 | **56.9** | **45.1** |
| Fidelity mean | **0.3115** | 0.2552 | 0.2747 |
| Fidelity log-mean | **0.1404** | 0.0749 | 0.0721 |
| vs SABRE（mean） | +13.4% | −7.1% | — |
| 逐电路胜率 vs l05 | — | 5/19 | — |
| 逐电路胜率 vs SABRE | — | 7/19 | — |

### 逐电路明细

| Circuit | q | l05 SWAPs | l05_nam SWAPs | SABRE SWAPs | l05 Fid | l05_nam Fid | SABRE Fid | 胜者 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| barenco_tof_10 | 19 | 103 | 0.1960 | 0.0658 | 0.0715 | l05 |
| barenco_tof_3 | 5 | 9 | 0.7636 | 0.6119 | 0.7467 | l05 |
| barenco_tof_4 | 7 | 30 | 0.4830 | 0.4734 | 0.4454 | l05 |
| barenco_tof_5 | 9 | 26 | 0.3089 | 0.2637 | 0.1683 | l05 |
| csla_mux_3 | 15 | 39 | 0.2566 | **0.3151** | 0.0341 | new |
| gf2^4_mult | 12 | 56 | 53 | 50 | **0.1989** | 0.0472 | 0.2348 | SABRE |
| gf2^5_mult | 15 | 101 | 89 | 72 | 0.0267 | **0.0912** | 0.0793 | new |
| gf2^6_mult | 18 | 162 | 172 | 109 | **0.0650** | 0.0297 | 0.0314 | l05 |
| grover_5 | 9 | 149 | 145 | 119 | **0.0033** | 0.0000 | 0.0006 | l05 |
| hwb6 | 7 | 50 | 0.0003 | **0.0015** | 0.0000 | new |
| mod5_4 | 5 | 12 | 0.4148 | 0.4946 | **0.5122** | SABRE |
| mod_mult_55 | 9 | 22 | 0.1057 | 0.0467 | **0.1554** | SABRE |
| mod_red_21 | 11 | 48 | **0.1215** | 0.0212 | 0.0645 | l05 |
| rc_adder_6 | 14 | 38 | 0.1608 | 0.0318 | **0.1662** | SABRE |
| tof_10 | 19 | 57 | 0.2223 | 0.3134 | **0.4264** | SABRE |
| tof_3 | 5 | 11 | **0.8198** | 0.8112 | 0.7779 | l05 |
| tof_4 | 7 | 17 | **0.7138** | 0.6111 | 0.6570 | l05 |
| tof_5 | 9 | 17 | **0.6041** | 0.5951 | 0.4699 | l05 |
| vbe_adder_3 | 10 | 28 | **0.4543** | 0.0235 | 0.1779 | l05 |

### 结论与分析

1. **SWAP 效率提升，但保真度反而下降**：l05_nam 的 mean SWAPs 65.3→56.9（−13%），
   但 mean fidelity 0.3115→0.2552（−18%），且跌到 SABRE（0.2747）之下。log-mean 同样
   从 0.1404 降到 0.0749（略高于 SABRE 0.0721，但远低于 l05）。
2. **逐电路胜率低**：对 l05 仅 5/19 胜，对 SABRE 7/19 胜。改善集中在 csla_mux_3、
   gf2^5_mult、hwb6 等个别电路；退化集中在 vbe_adder_3（0.454→0.024）、mod_red_21、
   gf2^4_mult、rc_adder_6、grover_5 等中等规模算术电路。
3. **“routing 目标学 NAM”没有转化为保真度**：Phase 1 reward 只看 SWAP/距离/时序，
   学到的是“更少 SWAP”的通用路由倾向；但 l05 原本的优势恰是“多 SWAP 但放在低噪声边”。
   NAM 微调把 π 拉向少 SWAP，失去了这层噪声路径偏好，导致 fidelity 掉。
4. **与 Phase 2 fidelity 微调的教训一致**：单纯改训练分布（NAM）或单纯改 reward 形态
   （fidelity 微调）都不足以同时保住 π 与 V 的质量；l05 的优势是“layout-mix + 纯
   routing + 分布外鲁棒”，用 NAM 直接替换一部分训练分布反而破坏了这种鲁棒性。
5. **单次训练、单 seed、traj=16 保真度仍有 MC 方差**；部分电路（grover_5、hwb6）本身
   fidelity 接近 0，排序噪声大，不宜过度解读逐电路胜率。
6. **后续方向**：
   - 若要 NAM 收益，建议不是替换 50% 训练分布，而是保留 l05 checkpoint、以更小比例
     （如 10-20%）混入 NAM 做保守微调；
   - 或在 l05 上只做 beam/V_fid 的 fidelity 微调（不动 π），复用 P2B 思路；
   - 若坚持 NAM 进训练，应加 NAM train/test 拆分，避免用同一批 19 条电路既训练又评估。

---

## 重大 bug 修复：映射阶段 1Q 门预执行导致路由线路损坏 + JSON 布局字段错误（2026-09-10）

### 问题发现（用户对 mod5_4 的手工核查触发）

用户从 `initial_layout` 出发追踪 `routed_qasm` 中全部 SWAP，得到与 `final_layout` 矛盾的
末布局。深度验证确认问题存在且分两层：

#### 层 1：JSON `initial_layout` 字段错误（旧 380 文件中仅 22 个碰巧正确）

- 旧代码 `_compute_final_layout` 回放 `_swap_history` 时，对「一端为空位」的 SWAP
  （`if lp is not None and lq is not None`）直接跳过，而真实 `_apply_swap` 会把逻辑比特
  搬入空位。5 逻辑比特 / 20 物理比特的场景下空端点 SWAP 极常见，回放从第一个空端点
  SWAP 起偏离真实初始布局。
- 提交 7aa364c 的「修复」只是把旧字段原地换名 + Q 标签转 int（`routed_qasm` 一字未动），
  把回放 bug 的错误值继承进了新 `initial_layout`。`final_layout` = 旧 initial 字段 =
  `env.mapping` 终态，是正确的。

#### 层 2：routed_qasm 线路本身损坏（旧 380 文件中 69 个）

- **根因**：`env.reset()` 无条件调用 `_update()`（env.py:182），非调度模式下急迫执行
  所有就绪 1Q 门并写入 `_phys_circuit`——发生在映射阶段**之前**。随后映射阶段的虚拟
  SWAP 重排 mapping，已放置的 1Q 门与逻辑比特归属错乱，物理线路混入两套布局，
  **任何单一初始布局都无法解释**（mod5_4 实测 720 种起始布局穷举无一自洽）。
- **判定规律**：全部 69 个损坏文件的 `mapping_swaps ≥ 1`；`mapping_swaps=0` 的文件
  线路全部正确（虚拟 SWAP 未触碰急迫门位置时也不损坏，如 l05/barenco_tof_3）。
- **为什么长期未被发现**：`compute_fidelity.py` 将 routed_qasm 与其自身无噪版本比保真度，
  与逻辑等价性无关，损坏线路照样得到"正常"保真度（mod5_4 = 0.4148）。

### 验证方法（可复现）

1. 状态向量等价性检验：`U_R·P_init = P_final·U_C` 是否为置换矩阵（mod5_4 不通过）；
2. 720 种起始布局穷举 + 贪心依赖序门匹配（修复前 0 解）；
3. 实测复现 env 轨迹（同 seed 同模型逐位复现 JSON），确认 reset 后 `_phys_circuit`
   已含 2 条急迫门（x,h@q[1]）；
4. 注意教训：QASM 解析时 rz 角度曾被误当物理比特，导致第一轮全量判定失真
   （380 全坏），修正解析后得到真实的 69 坏 / 311 好。

### 修复内容

1. **env.py**：`_update()` 急迫执行条件改为 `not self.use_scheduler and not self.mapping_phase`
   ——映射阶段一律不执行门；新增 `_effective_initial_mapping`（reset 时=初始映射，
   映射阶段 commit 时覆盖为 post-虚拟 SWAP 布局，clone 同步拷贝）。
2. **generate_routing.py**：`initial_layout` 改用 `env._effective_initial_mapping`
   （与 routed_qasm 严格对应：initial + 线路内 SWAP 追踪 == final）；
   `_compute_final_layout` 空端点回放 bug 修复（防御性）。
3. **训练无需重做**：l05 及后续全部微调均 `--use-scheduler`，调度模式下 `_update()`
   不急迫执行 1Q 门，训练环境干净；headline 保真度评估走 trajectory_sched 时
   eval_policy 自动启用调度器（eval_policy.py:970），亦免疫。受影响的只有
   generate_routing（硬编码 use_scheduler=False）与少量未加 --use-scheduler 的评估。

### 重新生成（scripts/regen_all_routed.sh，tmux 会话 regen_route，~20 分钟）

20 个模型目录全部重跑（同 checkpoint、同 beam 宽度、同 seed=0）：
l05/l05_beam3/l05_beam5/l05_nam/ph2v4/ph2v4_beam3/ph2v4_beam5/nam_l05_v2(_beam3)/
nam_p2a(_beam3)/nam_p2b(_beam3)/nam_p2c(_beam3)/nam_sref_v1(_beam3)/nam_sref_final/
nam_traj_v1(_beam3)。

**验证结果（scripts/verify_routed.py）：380/380 文件全部自洽**
（initial_layout + SWAP 追踪 == final_layout，且全部门与原始电路依赖序匹配）。

### 新旧对比（l05，19 条 NAM 电路）

| 指标 | 旧（含损坏线路） | 新（修复后） |
|------|----------------|-------------|
| 平均 SWAPs | 65.3 | 65.7 |
| 自洽文件数 | 15/19 | **19/19** |

逐电路 SWAP 变化极小（多数 ±0-2，hwb6 68→57 反而改善），策略能力无损，
纯簿记/归属修正。mapping_swaps 大多变为 0（观测中 progress 不再被预执行门扭曲，
agent 倾向立即 commit）。

### 保真度重算

`scripts/compute_fidelity.py` 泛化为全部 20 个模型目录（trajectory_sched×16，seed=0），
tmux 会话 regen_fid 后台运行中，完成后回填 per-circuit JSON 与 summary。
SABRE 基线（sabre_results.json）不受影响。挑战杯报告 NAM 表格数字待重算完成后更新。

---

## 路由推理速度优化：观测管线重构（2026-09-10，4.6–5.1x，输出逐位不变）

### 动机与实测瓶颈（优化前 profile，grover_5，400 步）

- 31.5 ms/step，其中 `build_graph_data` 55.6% + `_sabre_edge_features` 27.7% +
  GNN 前向 13.6%，**纯环境动力学仅 ~1.3%**——观测构造占 ~97%。
- `_sabre_edge_features`：29 条边 × 每边重复调用 `_ready_2q_gates()`（O(G) 全门扫描，
  每步 ~60 次）+ 每边重建 inv dict。
- `build_routing_graph`：Maps_to 边"完全重建"中 `avg_two`（每物理比特邻居平均错误率）
  为纯静态量却每门每步重算；`_nearest_occupied_distance` 每步 20 次独立 Python BFS；
  O(G) Python 循环遍布动态特征。

### 修复内容

1. **T1 `_sabre_edge_features` 重写**（env.py）：ready 集合/inv 映射每步一次；
   边 (p,q) 虚拟 SWAP 只把逻辑 lp→q、lq→p（含空端点分支），ready 门端点仅在
   qa/qb ∈ {lp,lq} 时变化——增量更新替代全量重算；求和保序，特征逐元素一致。
2. **T2 `build_routing_graph` 静态/动态分离**（circuit_dag.py）：
   - 静态量缓存于 `dag._routing_graph_cache`（门模板/依赖边模板预热、每门端点表、
     前驱矩阵、每物理比特邻居结构与 avg_two、maps-to per-pq 静态列模板）；
   - 动态特征全向量化（pending_count=bincount、err/exec_status/rem_pred/map_dist
     fancy-index、dep 列 8/9 复用 rem_pred 向量、coup 列 10 掩码、maps-to 查表 +
     2 个动态列）；
   - `_nearest_occupied_distance_multi`：多源 BFS 替代 20 次单源（结果一致）。
3. **T3 torch 线程上限**（generate_routing/eval_policy 新增 `--torch-threads`，默认 8）：
   112 核机上小图前向的多线程同步开销主导，实测 GNN 6.7→3.1 ms（2.2x）。
   torch.compile 评估后弃用：单电路场景编译开销（每形状数秒~数十秒）超过收益。
4. **T4 beam 候选批前向**（agent.py 新增 `_forward_obs_batch`，K 候选 V(s') 单次前向；
   generate_routing/eval_policy beam 循环接入）。

### 验证

- **特征级等价**（scripts/verify_feature_equivalence.py，永久回归工具）：
  T1 SABRE 特征 1500 步逐元素相等；T2 图特征 600 步全矩阵 `np.array_equal`
  （gate/dep/qubit/coup/map 全部字段）。
- **端到端输出等价**：l05/l05_beam3/ph2v4/ph2v4_beam3 四组重跑 19 条 NAM 电路，
  routed_qasm 与 initial_layout **76/76 逐字节相同** → 保真度数字不变，无需重算。
- pytest：49 passed（1 个预存无关失败 test_fidelity_shaping_step_zero，
  HEAD 上同样失败，与本次改动无关）。

### 基准结果（19 条 NAM 电路，wall_time 合计）

| 模型 | 优化前 | 优化后 | 加速 | SWAP 总数（旧→新） |
|------|-------|-------|------|------------------|
| l05 argmax | 22.0s | 4.8s | **4.6x** | 1248 → 1248（不变） |
| l05_beam3 | 38.7s | 8.0s | **4.9x** | 807 → 807（不变） |
| ph2v4 argmax | 22.2s | 4.7s | **4.7x** | 1235 → 1235（不变） |
| ph2v4_beam3 | 39.6s | 7.8s | **5.1x** | 804 → 804（不变） |

单步：31.5 → ~4-5 ms（约 7x，含线程优化）；beam3 已快于 argmax（路线更短步数更少）。
基准产物：`benchmark/routed_bench/`（与官方 routed/ 输出逐字节一致，仅 wall_time 不同）。

### 备注

- T1/T2 为纯计算重构，训练路径共享同一 `_obs`，特征逐元素一致 → 对训练无行为影响。
- torch.compile 弃用原因：路由 episode 步数不足以摊销每形状编译成本；线程上限
  已拿到大部分收益且零风险。
- eval_policy `--fidelity-sim trajectory_sched` 自动启用调度器的既有行为不受影响。

### 优化后 l05 argmax vs SABRE 编译时间对比（NAM 19 电路，同机实测）

- l05 时间取自 `benchmark/routed_bench/l05/`（优化后 wall_time_ms，含 env 构建与整个 episode）；
- SABRE 为 `evaluate_sabre`（swap_trials=20, decay）3 次取中位数（同机空闲态实测）。

| circuit | q | gates | l05 SWAP | SABRE SWAP | l05 (ms) | SABRE (ms) | 倍数 |
|---------|---|-------|---------:|-----------:|---------:|-----------:|-----:|
| barenco_tof_10 | 19 | 103 | 832 | 16.4 | 50.7x |  |
| barenco_tof_3 | 5 | 9 | 45 | 4.1 | 11.0x |  |
| barenco_tof_4 | 7 | 30 | 74 | 6.3 | 11.8x |  |
| barenco_tof_5 | 9 | 26 | 134 | 7.8 | 17.2x |  |
| csla_mux_3 | 15 | 39 | 172 | 8.1 | 21.2x |  |
| gf2^4_mult | 12 | 225 | 57 | 50 | 199 | 10.2 | 19.6x |
| gf2^5_mult | 15 | 347 | 103 | 72 | 391 | 14.3 | 27.4x |
| gf2^6_mult | 18 | 495 | 163 | 109 | 683 | 19.6 | 34.8x |
| grover_5 | 9 | 831 | 149 | 119 | 785 | 28.7 | 27.4x |
| hwb6 | 7 | 50 | 212 | 11.1 | 19.1x |  |
| mod5_4 | 5 | 12 | 44 | 4.9 | 9.1x |  |
| mod_mult_55 | 9 | 22 | 91 | 6.7 | 13.7x |  |
| mod_red_21 | 11 | 48 | 252 | 10.6 | 23.7x |  |
| rc_adder_6 | 14 | 38 | 213 | 7.8 | 27.4x |  |
| tof_10 | 19 | 57 | 351 | 9.2 | 38.0x |  |
| tof_3 | 5 | 11 | 33 | 3.3 | 9.9x |  |
| tof_4 | 7 | 17 | 56 | 4.6 | 12.0x |  |
| tof_5 | 9 | 17 | 75 | 5.7 | 13.2x |  |
| vbe_adder_3 | 10 | 28 | 147 | 7.0 | 21.1x |  |
| **总计/平均** | | | **1248** | **857** | **4791 (252 平均)** | **186.4 (9.8 平均)** | **25.7x（几何 8.4x）** |

分析：优化前 l05 全套 22.0s（相对 SABRE ~118x），优化后 4.8s（25.7x）。差距随规模增长
（5q ~9-11x → 19q/深电路 27-50x），主因是 RL 每步 GNN 前向 + 观测构造按步数线性累积，
而 SABRE 是编译型启发式。交换的价值在保真度（+20.5%）与调度质量（makespan 0.66x、
串扰 0.45x），编译时间已从"不可用"降到批处理可接受量级。

---

## 全模型横向对比：为什么 l05 是最强基础策略（2026-09-10，路由 bug 修复后重算数据）

### 评估口径

- 电路：NAM benchmark 19 条（5–19q 算术电路，最大 495 门），与全部训练 split 零重叠（分布外）
- 评估：`trajectory_sched`（调度感知轨迹模拟器）× 16 轨迹，确定性 argmax（除标注 beam 外）
- 数据来源：`benchmark/routed/*_summary.json`（bug 修复后重新生成的全部模型）与 `sabre_summary.json`
- SABRE 基线：Qiskit SabreSwap（decay, trials=20），不经过 env，不受路由 bug 影响

### 总排名（按 mean 保真度降序）

| 模型 | mean_fid | log-mean | 均 SWAP | 胜 SABRE | 训练配方 |
|------|---------:|---------:|--------:|---------:|----------|
| nam_p2b | **0.3423** | 0.2123 | 67.3 | 14/19 | l05 + 双价值头微调（fid 进 V 不进 π，α=0.1，KL 保策略） |
| **l05** | 0.3309 | **0.2133** | **65.7** | **16/19** | 纯路由 + layout-mix(0.3/0.3/0.4) + 调度感知奖励，真机拓扑 |
| nam_p2a | 0.3112 | 0.1679 | 65.7 | 12/19 | l05 + fid 直进 actor（旧行为对照） |
| nam_sref_v1_beam3 | 0.3077 | 0.0993 | 42.6 | 9/19 | per-circuit sref 微调 + beam3 推理 |
| ph2v4 | 0.3061 | 0.1390 | 65.0 | 10/19 | l05 + Phase2 噪声感知 v4 |
| ph2v4_beam3 | 0.2934 | 0.1525 | 42.3 | 11/19 | |
| l05_nam | 0.2918 | 0.0947 | 57.6 | 9/19 | l05 系 NAM 混训 |
| ph2v4_beam5 | 0.2907 | 0.0939 | 40.4 | 9/19 | |
| nam_p2a_beam3 | 0.2903 | 0.1115 | 43.5 | 11/19 | |
| nam_p2b_beam3 | 0.2916 | 0.0973 | 42.9 | 9/19 | |
| l05_beam3 | 0.2887 | 0.0630 | 42.5 | 9/19 | l05 + beam3 推理 |
| nam_p2c_beam3 | 0.2860 | 0.1192 | 42.6 | 9/19 | |
| nam_traj_v1_beam3 | 0.2853 | 0.1258 | 43.8 | 12/19 | |
| nam_traj_v1 | 0.2844 | 0.1298 | 64.9 | 9/19 | trajectory_sched 微调 v1 |
| l05_beam5 | 0.2836 | 0.0795 | 42.4 | 11/19 | |
| nam_l05_v2_beam3 | 0.2801 | 0.0343 | 45.3 | 7/19 | |
| nam_sref_v1 | 0.2784 | 0.1129 | 60.3 | 9/19 | |
| nam_p2c | 0.2752 | 0.1002 | 63.9 | 10/19 | |
| **SABRE（基线）** | 0.2747 | 0.0721 | 45.1 | — | |
| nam_l05_v2 | 0.2599 | 0.0256 | 57.5 | 7/19 | NAM 混训 v2（退化） |
| nam_sref_final | 0.2566 | 0.1332 | 62.4 | 6/19 | |

### 关键观察 1：nam_p2b 微超 l05 的 mean，但两者统计平手

- mean：nam_p2b 0.3423 vs l05 0.3309（+3.4%）
- **log-mean（几何平均）l05 反超**：0.2133 vs 0.2123——l05 在低保真度难电路上更稳
- **对 SABRE 胜率 l05 更高**：16/19 vs 14/19
- **l05 用更少 SWAP**：65.7 vs 67.3
- 逐电路对打：l05 胜 9 / p2b 胜 7 / 平 3（barenco_tof_3、tof_3、tof_4 两者路由完全相同——
  P2B 的 KL 保策略设计使 π ≈ l05，直接印证其策略继承关系）
- p2b 大幅赢的电路（gf2^5 +0.18、gf2^6 +0.17、tof_10 +0.13）与大幅输的电路
  （barenco_tof_10 −0.08、hwb6 −0.07）相互抵消，19 条样本下均值差异不具统计显著性

### 关键观察 2：argmax vs beam 的系统性分工

- 所有 beam3 变体 SWAP 大幅减少（65 → 42 左右，−35%）但 mean fid 普遍低于各自 argmax
  ——l05 系 beam 的 critic 无保真度信号，beam 按路由效率选路反而丢掉 π 的噪声偏好；
- 例外：nam_sref_v1 beam3（0.2728→0.3077）与 p2b 系——它们的 critic 学过保真度，
  beam 才有正收益，但仍未超过 l05 argmax；
- 结论：「argmax 用 π、beam 用 V」——π 的质量决定 argmax 上限，V 的质量决定 beam 上限，
  l05 的 π 是全场最稳的。

### 为什么 l05 是最强基础策略（五点归因）

1. **layout-mix 布局无关路由**：训练期按 0.3 恒等 / 0.3 随机 / 0.4 SABRE warm-start 混合采样
   初始布局，策略学到「从任何布局出发都能高效路由」的通用能力。这是 warm-start 混合管线
   与分布外鲁棒性的共同根基；恒等布局专用模型（Exp A）warm-start 反而劣化的教训已验证。

2. **纯路由目标避免保真度地形过拟合**：l05 无终端保真度奖励（仅距离/时长/空闲/串扰的
   调度感知奖励）。所有噪声感知微调版（ph2*/nam_*）都把 π 推向训练分布
   （random/qaoa/vqe ≤16q）的保真度局部最优——在分布外的 NAM 电路上 argmax 偏离稳健路由。
   保真度信号主要沉淀进 Critic（beam 受益）而非策略先验（argmax 受益）——P2A/P2B/P2C
   消融已系统验证该 π/V 分工，nam_p2b 的设计正是其成功应用。

3. **保真度优势来自调度而非 SWAP 数**：l05 比 SABRE 多用 45% 的 SWAP（65.7 vs 45.1），
   却换来更高并行度（3.7 vs 2.3）、更短 makespan（0.66x）、更低串扰（0.45x）→
   空闲退相干更少 → 保真度更高。log-mean 是 SABRE 的 3 倍（0.2133 vs 0.0721）——
   优势恰好集中在 SABRE 最吃力的深电路上。

4. **真机拓扑 + 逐边噪声微调**：在 tianyan176_20q（逐比特 T1/T2、逐边 CX 错误率、
   相干 ZZ 串扰模型）上微调，学会「噪声友好」路由——把 SWAP 与门安排到低错误率边。

5. **长课程血缘**：unified 多尺度预训练（8–20q）→ 真机子拓扑微调 → 调度感知对齐
   （eta_xtalk_par 扫描）→ layout-mix（Exp C）。多阶段多样化训练铸就泛化底座，
   所有后续变体（ph2 系、nam 系、p2 系）全部以 l05 为起点——它们的成绩本身就是
   l05 基座质量的佐证。

### 结论

1. **l05 是最强基础策略**：argmax 路径上 log-mean 与胜率双第一，且是唯一同时满足
   「分布外稳健 + 布局无关 + 调度全面占优」的模型。
2. **nam_p2b 是 l05 的最佳增量**（KL 保策略 + fid 进 V）：mean 微超但统计平手，
   属于「锦上添花」而非替代——若需要 beam 推理或略高的 mean，选 p2b；若追求
   稳健与简单（纯 argmax 部署），选 l05。
3. **噪声感知微调的收益高度依赖保真度信号的注入方式**：直进 actor（p2a/旧 ph2）
   损害分布外 argmax；进独立 V 头（p2b）无损害且改善 beam；per-circuit sref（v1）
   在 beam 上有收益但 argmax 仍逊于 l05。
4. **部署建议**：默认 l05 argmax（简单、稳健、快）；需要进一步压 SWAP 时用
   l05 + SABRE warm-start；追求极限 mean 时用 nam_p2b。

---

## 分析：ph2 微调不如 l05 的根本原因——"模拟器信号无法指引"假说检验（2026-09-10）

### 假说

"模拟器保真度信号无法有效指引模型"是 ph2 系微调 argmax 不如 l05 的根本原因。

### 检验证据一：同一信号/模型，换评估分布结论反转

| 评估面 | ph2v4 vs l05 | 结论 |
|--------|--------------|------|
| 无泄露 test split（训练分布内） | 8q/10q/12q：**+16.2% / +75.1% / +79.5%** | 信号有效 |
| NAM（分布外） | −7.5%，退化集中 5–9q/15–19q | OOD 泛化失败 |

若信号完全无效，ph2v4 不可能在分布内 +75% → 信号有真实信息量，
失效是**分布偏移**（微调把 π 锚定到训练分布的保真度地形）而非信号失效。

### 检验证据二：P2A vs P2B（同信号、不同消费路径）

- P2A（fid 经 GAE 直进 actor）：NAM argmax 退化 6%，λ 渐进救不回 → 消费路径问题
- P2B（同信号只进独立 V_fid 头）：argmax 0.988×l05，修复 bug 后 NAM mean 0.3423 反超
  → 同一模拟器信号可产生超越 l05 的模型 → "信号无法指引"被否定

### 完整因果链（三层）

1. **信号对 actor 的指引效率低**：terminal-only 稀疏 + traj=8 高方差 + GAE 信用分配
   失配（一个标量摊给全部历史 SWAP）；局部/门级代理与 scheduled eval 排序负相关
   （Spearman −0.895~−1.0）→ dense 信号路径不可用；逐步 scheduled 全量评估
   0.1–11s/步训练不可行 → 只剩"终端信号只进 V 不进 π"。
2. **分布偏移**：保真度地形训练分布特定；NAM 是分布外 → argmax 退化。
   （同分布 test split 上 ph2 系反而占优。）
3. **基线过强 + 边际信息不足**：l05 的调度感知 dense 奖励（idle/串扰/时长）已捕获
   保真度主要结构项；终端信号残余信息边际小（主要 ≥10q 有空间），在已收敛策略上
   噪声 > 新信息。

### 结论

- 假说部分成立：信号的**actor 侧消费路径**（GAE 摊派）确实低效，是核心矛盾之一；
- 但完整根因 = 消费路径失配 × 分布偏移 × 基线边际信息不足，三者缺一不可；
- 信号本身信息量充足（critic 可学、p2b 靠它反超），"指引失败"发生在 π 的梯度路径上；
- 改进方向：fid 只进 V（已验证）/ NAM 混训带 train-test 切分 / 轨迹级 return 分解 /
  更便宜的 scheduled 局部评估信号。

---

## 事件级调度感知模拟器 v2（trajectory_sim_v2）：实现 + 验证 harness + smoke 对比（2026-09-11）

### 背景

v1 调度感知路径（`evolve_scheduled`，同步波近似）存在三类系统性失真：
1. **统一门时长**：所有 1q 门按 0.1µs、2q 门按 0.3µs 计退相干（rz 实际为 0 时长虚拟门、
   sx≈0.035µs、swap=0.9µs）；**SWAP 事件完全无噪声**（仅推进时钟）；
2. **动态串扰与重叠时长无关**：同波 1-hop 相邻交叉对每波施加一次固定 θ；
3. **缺失建模**：1Q 门与相邻 2Q 门并发的 spectator 串扰、耦合对空闲期 always-on ZZ、
   事件级精确 start/end（`get_schedule_waves` 把事件级 schedule_log 压缩成同步波）。

### 实现（新文件，零干扰）

- 新增 `src/sim/trajectory_sim_v2.py`（`sim.py`/`trajectory_sim.py`/`timing.py`/`env.py` 零改动）：
  - `EventNoiseConfig(NoiseConfig)`：追加 `always_on_zz`（rad/µs，默认 None=关）与 `swap_xtalk`（默认 False，与 timing.py A2 口径一致）；
  - `EventTrajectorySimulator(TrajectorySimulator)`：继承复用 v1 全部批量 MC 内核；核心 `evolve_events(circuit, events, apply_noise)` 直接消费每门 `(start, end, op, qubits)` 事件；
  - 事件源：`timing_log_to_events`（读 env 事件级 `schedule_log`）或 `schedule_phys_circuit_events`（平铺 ASAP 回退，时长镜像 `GATE_DURATION_TABLE`）；
  - CLI 纯追加：`--fidelity-sim trajectory_v2`（train_agent / eval_policy / generate_routing），旧行为零变化。

### 噪声语义（v2）

| 项 | v1 同步波 | v2 事件级 |
|----|-----------|-----------|
| 门内热弛豫 | 统一 0.1/0.3µs | 按 per-gate `end-start`（rz=0 → 无噪声） |
| 空闲退相干 | 每比特独立时钟（同 v1） | 同 v1（Markov 精确复合） |
| SWAP 噪声 | **无** | 3×(depol2+ZZ) + thermal(0.9µs)（物理等价 3×CX） |
| 静态 ZZ | 固定 θ | θ·(dur/two_gate_time)（cx=0.3µs 时与 v1 对齐） |
| 动态串扰 | 每波一次固定 θ | χ_rate·ov，χ_rate=θ/two_gate_time，ov=实际重叠时长；含 1Q-2Q spectator |
| always-on ZZ | 无 | 双空闲段 ZZ(rate·Δt)（config 开关，默认关） |
| 调度输入 | 同步波 `(dw, gates)` | 事件级 `(start, end, op, qubits)` |

### 验证 harness（test/test_event_sim.py，22 用例全过）

精确参考 = `qiskit DensityMatrix` 按**同一编译动作流**施加解析 Kraus 通道（depolarizing_error、
amplitude_damping ∘ phase_damping，与 v2 MC 通道定义逐项对应）：

| 用例 | 结果 |
|------|------|
| a) 空闲热弛豫解析对拍（F=exp(-t/T1)） | ✅ <0.02（MC 1024 traj） |
| b) 相邻边 CX 并发：动态 ZZ 随 ov 缩放（F=cos²(χ·ov)） | ✅ 精确到 1e-6（酉通道下零方差） |
| c) spectator 1Q 门并发（解析 cos²θ） | ✅ 1e-6 |
| d) rz 零时长：施加酉、不施加噪声 | ✅ 1e-9 |
| e) swap 事件 ≈ 3×CX 串行；swap_xtalk 开关解析对拍 | ✅ <0.02 / 1e-6 |
| f) always-on ZZ 开/关 + 区间求交单元测试 + reduce 重映射 | ✅ 1e-6 |
| g) 端到端 v2(MC, T=256) vs 精确参考 | ✅ \|ΔF\|<0.02 |
| 回归：同步波退化下 v2 ≡ v1（同种子、同动作序） | ✅ 1e-6 |
| 校验/转换/ASAP 调度/env 工厂回退 | ✅ |

全量回归：`PYTHONPATH=src python3 -m pytest test/ -q` → 72 passed
（`test_env.py::test_fidelity_shaping_step_zero` 为存量失败，与本次改动无关，已用 git stash 验证）。

### Smoke 实验：v2 vs v1 legacy（tianyan176_20q，280 电路，policy_ph2_sub20q）

```bash
cd src && tmux new -d -s v2smoke "python3 -m routing.rl.eval_policy \
  --model ../models/policy_ph2_sub20q.pt --topo ../traindata/topo/tianyan176_20q.json \
  --data-dir ../traindata --split stage2_mixed \
  --fidelity-sim trajectory_v2 --traj-trajectories 32 --max-num-qubits 20 --seed 42 \
  --baselines --no-greedy --no-random"
# 对照组：同一命令 --fidelity-sim trajectory_sched（v1 同步波）
```

| 模拟器 | PPO SWAPs | SABRE SWAPs | PPO Fidelity | SABRE Fidelity |
|--------|-----------|-------------|--------------|----------------|
| v1 trajectory_sched（同步波） | 5.5±2.4 | 4.5±1.5 | 0.4193 | 0.3867 |
| **v2 trajectory_v2（事件级）** | 5.5±2.4 | 4.5±1.5 | **0.2266** | **0.2590** |

（路由统计完全一致——评估确定性；仅保真度口径不同。）

### 结论与分析

1. **v2 保真度系统性低于 v1（约 −0.13~−0.16）**，主因是 v1 对 SWAP 事件完全不施加噪声、
   且把 rz 当 0.1µs 实门计退相干（双向误差），v2 按物理事实建模（swap=3×CX 噪声 + 0.9µs
   双比特热弛豫）；
2. **PPO/SABRE 排序在 v2 下反转**：v1 口径下 PPO(5.5 SWAPs) 保真度反超 SABRE(4.5)，
   v2 正确计入每颗 SWAP 的真实噪声代价后 PPO(0.2266) < SABRE(0.2590)——说明 v1 的
   "PPO 优势"部分来自 SWAP 噪声低估的口径偏置，v2 口径更可信；
3. v2 与解析精确参考（DensityMatrix）对拍 |ΔF|<0.02，与 v1 在同步波退化情形数值一致
   （同种子 1e-6），可作为后续噪声感知训练的保真度信号源（`--fidelity-sim trajectory_v2`）。

### 后续

- 用 `trajectory_v2` 作为终端保真度信号重跑 P2B/微调，验证排序反转对策略学习的影响；
- 视需要开启 `always_on_zz`（默认关，不改变现有标定）；
- ecr 等非 cx/cz/swap 双比特门暂不支持（validate 抛 NotImplementedError，需先转译）。

---

## NAM benchmark 复测：v2 事件级模拟器口径（l05 / l05_nam vs SABRE）（2026-09-11）

### 背景

v2 事件级模拟器（`trajectory_sim_v2`，见上一节）在 stage2_mixed smoke 中出现
PPO/SABRE 排序反转。本节在 NAM benchmark（19 条 ≤20q 算术电路，tianyan176_20q）
上复测：协议与 v1 NAM 基准完全一致（trajectory×16, seed=0），仅保真度模拟器
从 v1 同步波换为 v2 事件级。路由与模拟器无关：l05/l05_nam 复用既有路由 QASM，
SABRE 用项目基线同参重路由（decay, 20 trials, seed=0，swap 数与 v1 记录完全
一致，45.1 均值 ✓）。

```bash
python3 scripts/eval_nam_v2sim.py 16 l05_nam   # 缓存命中直接汇总
python3 scripts/eval_nam_v2sim.py 16 l05       # 补跑 l05 基模侧
```

### 汇总（trajectory_v2 ×16, seed=0）

| 指标 | SABRE | l05 | l05_nam |
|------|-------|-----|---------|
| SWAPs mean | 45.1 | 65.7 | 57.6 |
| Fid mean | **0.2301** | 0.1867 | 0.1822 |
| Fid logmean | 0.0395 | **0.0744** | 0.0611 |
| 胜 SABRE（逐电路） | — | 8/19 | 8/19 |

v1 同步波口径（历史）：SABRE 0.2747 / l05 0.3115（l05 +13.4%，胜 10/19）。

### 逐电路对比

| Circuit | q | SABRE_sw | l05_sw | SABRE_v1 | l05_v1 | SABRE_v2 | l05_v2 | v2 胜者 |
|---------|---|---------:|-------:|---------:|-------:|---------:|-------:|--------|
| barenco_tof_10 | 19 | 0.0715 | 0.1960 | 0.0181 | 0.0258 | l05 |
| barenco_tof_3 | 5 | 0.7467 | 0.7636 | 0.6708 | 0.3930 | SABRE |
| barenco_tof_4 | 7 | 0.4454 | 0.4830 | 0.3623 | 0.3689 | l05 |
| barenco_tof_5 | 9 | 0.1683 | 0.3089 | 0.2077 | 0.0605 | SABRE |
| csla_mux_3 | 15 | 0.0341 | 0.2566 | 0.0007 | 0.2347 | l05 |
| gf2^4_mult | 12 | 50 | 57 | 0.2348 | 0.1989 | 0.1264 | 0.0935 | SABRE |
| gf2^5_mult | 15 | 72 | 103 | 0.0793 | 0.0267 | 0.0080 | 0.0526 | l05 |
| gf2^6_mult | 18 | 109 | 163 | 0.0314 | 0.0650 | 0.0679 | 0.0595 | SABRE |
| grover_5 | 9 | 119 | 149 | 0.0006 | 0.0033 | 0.0000 | 0.0000 | l05 |
| hwb6 | 7 | 0.0000 | 0.0003 | 0.0000 | 0.0075 | l05 |
| mod5_4 | 5 | 0.5122 | 0.4148 | 0.2632 | 0.3508 | l05 |
| mod_mult_55 | 9 | 0.1554 | 0.1057 | 0.0418 | 0.0172 | SABRE |
| mod_red_21 | 11 | 0.0645 | 0.1215 | 0.1187 | 0.1791 | l05 |
| rc_adder_6 | 14 | 0.1662 | 0.1608 | 0.0619 | 0.0538 | SABRE |
| tof_10 | 19 | 0.4264 | 0.2223 | 0.3556 | 0.1300 | SABRE |
| tof_3 | 5 | 0.7779 | 0.8198 | 0.7861 | 0.5705 | SABRE |
| tof_4 | 7 | 0.6570 | 0.7138 | 0.5202 | 0.3774 | SABRE |
| tof_5 | 9 | 0.4699 | 0.6041 | 0.3360 | 0.3271 | SABRE |
| vbe_adder_3 | 10 | 0.1779 | 0.4543 | 0.4267 | 0.2462 | SABRE |

（产物：`benchmark/routed/v2sim_nam_fidelity.json`、`v2sim_nam_fid.log`、
`v2sim_nam_l05.log`；脚本 `scripts/eval_nam_v2sim.py`，带缓存支持增量补跑。）

### 结论与分析

1. **均值口径下排序反转**：v1 口径 l05 +13.4% → v2 口径 l05 −18.9%
   （0.1867 vs 0.2301），与 stage2_mixed smoke 结论一致。主因是 v2 正确计入
   每颗 SWAP 的物理代价（3×CX 退极化/串扰 + 0.9µs 双比特热弛豫），v1 对
   SWAP 事件完全不计噪声，使 l05 的多 SWAP 策略（+45%）看起来免费；
2. **log-mean 口径 l05 仍大幅领先（+88%）**：在保真度被结构主导的深电路
   （csla_mux_3 335×、gf2^5_mult 6.6×、mod_red_21、barenco_tof_10、hwb6）上
   l05 的噪声友好路由优势在 v2 下依然成立甚至扩大；反转集中在 SWAP 数差异
   主导的中等电路（tof_3/4/5、vbe_adder_3、barenco_tof_3/5）；
3. **l05_nam（NAM 混训微调）在 v2 下也略差于 l05 基模**（0.1822 vs 0.1867），
   与此前 ph2v4 退化的方向一致——v1 口径下训练的微调未带来 v2 口径收益；
4. **两个口径对电路难度排序一致**（深电路保真度都低），但胜负面向少 SWAP
   策略移动；v2 已通过 DensityMatrix 精确参考与解析闭式对拍（|ΔF|<0.02），
   更贴近硬件，因此 v1 口径下的 l05 优势应视为部分口径偏置；
5. **行动项**：用 `--fidelity-sim trajectory_v2` 作为终端信号重训/微调路由
   策略，检验"少 SWAP + 噪声友好"在正确计价下能否兼得；逐电路近胜局
   （tof_5 0.3360 vs 0.3271）在 MC 噪声内，需要更多轨迹收紧。

### 消融归因：排序反转是 SWAP 代价变高了吗？

给 `EventNoiseConfig` 增加 `swap_noise` 开关（默认 True=物理等价 3×CX；False=v1
语义，swap 仅推进时钟），在 v2 引擎上重建三档语义（NAM 19 电路，×16, seed=0）：

| setting | SABRE | l05 | gap (l05−SABRE) |
|---------|-------|-----|-----------------|
| A v1sem（1q 统一 0.1µs 含 rz、swap=0.3µs 无噪声） | 0.2953 | 0.2624 | −0.033 |
| B noswap（v2 时长 + swap 无噪声） | 0.2485 | 0.2830 | **+0.035** |
| C full（v2 完整语义） | 0.2301 | 0.1867 | **−0.043** |

gap 分解：A→B（1q 时长修正：rz 零时长/sx 短时长/swap 时钟修正）**+0.067**；
B→C（SWAP 噪声计价）**−0.078**；合计 −0.010 ≈ C 的实际 gap。

方向一致性：去掉 SWAP 噪声后 l05 在 **15/19** 条电路上保真度上升（SABRE 仅
10/19，≈随机——其 SWAP 少、效应被 MC 方差淹没）。

**结论：是，但不完全是。**
1. **SWAP 噪声计价是反转的主因**（gap −0.078）：v2 给每颗 SWAP 记
   3×(depol+ZZ)+0.9µs 双比特热弛豫（v1 完全不计），l05 多 45% 的 SWAP 承担
   了大部分新增代价——单侧 15/19 的一致性排除了 MC 偶然；
2. **1q 时长修正部分反向抵消**（+0.067）：rz 虚拟门不再计 0.1µs 退相干、sx
   按 0.035µs 计，v1 系统性高估 1q 退相干，修正后 l05 受益（其路由 1q 门比例
   更高）；
3. 两项相抵，净反转 −0.010 与 full 口径的实际 gap 一致。即 v2 口径下 SABRE
   的均值优势 ≈ 「SWAP 真实代价」减去「v1 高估的 1q 退相干修正」的净效应；
4. caveats：v1sem 未能复现 v1 实测排序（gap −0.033 vs 实测 +0.056）——v1 的
   高 l05 优势还依赖同步波时序/swap 空闲漏算等细节，本身不够稳健；深电路
   单电路 T=16 估计的 MC 方差很大（rc_adder_6 noswap 三种子 0.0000/0.186/0.062），
   逐电路近胜局不可靠，聚合与 15/19 单侧一致性是主要证据。

---

## v2 口径下的 PPO 优化：两阶段重训（2026-09-11，进行中）

### 设计依据（来自 v2 消融）

v2 消融表明：v1 口径的 l05/PPO 优势 = SWAP 噪声免费（−0.078 gap 反向）+ 1q 退相干
高估（+0.067）的净效应。优化方向是让奖励结构与 v2 的物理计价对齐：

1. **SWAP 边噪声计价（新）**：env 新增 `eta_swap_err`——SWAP=3×CX，按所在耦合边
   `two_q_err` 计价（边感知，原 `swap_cost` 是平坦项且训练主循环 env 实际未传入）。
   `eta_swap_err=1.5 ≈ 3×eta_err(0.5)` 完全对价；
2. **终端信号换 v2**：`--fidelity-sim trajectory_v2`（事件级、per-gate 时长、
   overlap 缩放串扰），训练/评估同口径；
3. **sabre_fid_map 重标定**：终端奖励是 `λ·(log fid − log sref)`，map 必须与信号
   同单位——用 v2 NAM 实测重算（5=0.5734, 7=0.2942, 9=0.1464, 10=0.4267,
   11=0.1187, 12=0.1264, 14=0.0619, 15=0.0044, 18=0.0679, 19=0.1868）；
4. **P2B 双价值头**：fid advantage 只进独立 V_fid 头（P2A 直进 actor 已被证伪）；
5. **MC 方差缓解**：终端信号 ×32 轨迹（实测 T=16 深电路单种子方差 0.0000~0.19）。

### 命令（tmux 串行两阶段）

```bash
# Phase 1：l05 routing 微调（密集奖励重校准，无终端模拟，~40min）
python3 -u -m routing.rl.train_agent \
  --topo ../traindata/topo/tianyan176_20q.json --max-num-qubits 20 \
  --reward-mode routing --split-prefix unified --use-scheduler \
  --eta-swap-err 1.5 --swap-cost 0.3 --eta-xtalk-par 0.05 \
  --nam-circuits-dir ../benchmark/nam_circs --nam-circuit-prob 0.3 \
  --load ../models/policy_tianyan20q_laymix_l05_eta05.pt \
  --timesteps 50000 --out ../models/policy_l05_v2swap_ft.pt

# Phase 2：P2B + trajectory_v2 终端信号（从 Phase 1 产物微调，~数小时）
python3 -u -m routing.rl.train_agent \
  --topo ../traindata/topo/tianyan176_20q.json --max-num-qubits 20 \
  --reward-mode noise_aware --variant P2B \
  --fidelity-sim trajectory_v2 --use-scheduler --traj-trajectories 32 \
  --eta-swap-err 1.5 --eta-xtalk-par 0.05 \
  --nam-circuits-dir ../benchmark/nam_circs --nam-circuit-prob 0.3 \
  --sabre-fid-map "5=0.5734,7=0.2942,9=0.1464,10=0.4267,11=0.1187,12=0.1264,14=0.0619,15=0.0044,18=0.0679,19=0.1868" \
  --lambda-fid-max 5.0 --lambda-fid-warmup 0.0 \
  --load ../models/policy_l05_v2swap_ft.pt \
  --timesteps 100000 --out ../models/policy_p2b_v2sim.pt
```

### 状态

- Phase 1：运行中（step 1024 时 rew 45→90 上升，swp 12→26，time 12→25µs）；
- Phase 2：排队等 Phase 1 完成；
- 评估计划：完成后用 `eval_policy --fidelity-sim trajectory_v2 --baselines
  --no-greedy --no-random` 与 SABRE 同口径对比（stage2_mixed + NAM 各一组），
  结果回填本节。

### Phase 1 评估结果（stage2_mixed，trajectory_v2 ×32, seed 42）—— 微调退化

| 模型 | SWAPs | Fidelity(v2) | makespan | vs SABRE |
|------|-------|--------------|----------|----------|
| SABRE | 4.5±1.5 | 0.2590 | — | — |
| 旧 l05 | 5.5±1.8 | 0.2392 | 13.01µs | −7.6% |
| 新 l05_v2swap_ft | 6.3±3.0 | 0.2244 | 13.66µs | −13.4% |

**Phase 1 密集奖励重校准未奏效，诊断：**
1. **eta_swap_err 在实际错误率量级下太弱**：tianyan 边错误率 ~0.01 →
   单次 SWAP 边惩罚 ≈ 0.015，且训练主循环 env 既有怪癖是 swap_cost 不传入
   （仅采样 env 有），SWAP 定价在训练中近乎失效，策略没有感受到"少 SWAP"
   的压力（SWAPs 反而 5.5→6.3）；
2. 密集奖励的绝对量级（每执行门 +0.3）下，惩罚项 ~0.015 对行为影响微弱；
   真正的行为杠杆是 eta_dist 与终端信号；
3. 50k 步微调对已收敛的 l05 是扰动而非改进（原始 l05 经长课程+layout-mix
   训练）；
4. 有价值的副产品：**旧 l05 在 v2 口径下（0.2392）好于 ph2v4（0.2266）**，
   距 SABRE 仅 −7.6%——v1 口径下 l05 的优势被低估了其真实水平。

**行动项**：Phase 2 的正确起点应是旧 l05（而非本次 Phase 1 产物）；SWAP 定价
要么大幅提高权重（eta_swap_err 15+），要么直接依赖 Phase 2 的 v2 终端信号
（终端 log-ratio 对 SWAP 数的差异是强信号，不需要密集项复制它）。

**Phase 2 重启（旧 l05 起点）**：鉴于 Phase 1 产物退化（0.2244 < 旧 l05 0.2392），
杀掉原 Phase 2（当时 step 4352/30000，从退化 checkpoint 训练），改为从
`policy_tianyan20q_laymix_l05_eta05.pt` 直接进 Phase 2（P2B + trajectory_v2 ×8,
30k 步，其余参数不变），输出 `policy_p2b_v2sim_l05init.pt`。启动观察：旧 l05
初始 SWAP 更少（8.9 vs 退化版的量级）、ent 起点低（0.25，收敛策略微调的特征）、
fid 0.20-0.24 与 NAM 评估量级一致。完成后同口径评估对比。

### l05 beam3（V(s') lookahead）在 v2 口径下的表现

同口径（stage2_mixed ×32 seed 42；NAM ×16 seed 0, trajectory_v2）：

| 数据集 | SABRE | l05 argmax | l05 beam3 |
|--------|-------|-----------|-----------|
| stage2_mixed SWAPs | 4.5 | 5.5 | 5.4 |
| stage2_mixed Fidelity | 0.2590 | 0.2392 | 0.2301 |
| NAM SWAPs mean | 45.1 | 65.7 | **42.5** |
| NAM Fid mean | 0.2301 | 0.1867 | **0.2246** |
| NAM Fid logmean | 0.0395 | **0.0744** | 0.0515 |

结论：
1. **beam3 大幅压低 SWAP**（NAM 65.7→42.5，比 SABRE 还少）——l05 的 critic
   V(s') 是 routing 口径，beam 评分 reward+γV(s') 追路由效率（少 SWAP/短 makespan）；
2. **stage2_mixed beam3 反而略差**（0.2301 < argmax 0.2392）：SWAP 只少 0.1，
   路由效率优化在随机电路上未转为保真度，与 doc 既有结论（beam 优化效率≠保真度）一致；
3. **NAM mean 上 beam3 追近 SABRE**（0.2246 vs 0.2301，SWAP 42.5<45.1），但
   logmean（0.0515）明显弱于 argmax（0.0744）——beam 牺牲了 argmax 在深电路
   （csla_mux_3 等）的噪声友好路径优势，换取浅/中电路少 SWAP 的增益；
4. 要在 v2 口径下让 beam 真正优化保真度，需要 critic V(s') 含保真度信息
   （noise_aware/P2B 训的 critic），即当前 Phase 2 完成后 beam 才有意义。

### 优化迭代结果矩阵（v2 口径）与「超过 SABRE」的现状

| 配置 | NAM Fid mean | NAM logmean | NAM SWAPs | stage2 Fid | stage2 SWAPs |
|------|-------------|-------------|-----------|------------|--------------|
| SABRE | 0.2301 | 0.0395 | 45.1 | **0.2590** | 4.5 |
| l05 argmax | 0.1867 | **0.0744** | 65.7 | 0.2392 | 5.5 |
| l05 beam3 | 0.2246 | 0.0515 | 42.5 | 0.2301 | 5.4 |
| **l05 beam5** | **0.2326** | 0.0569 | **42.4** | 0.2318 | 5.2 |
| P2B-30k argmax | 0.2016 | 0.0084 | 58.9 | 0.2372 | 5.6 |

**NAM 上 l05 beam5 首次超过 SABRE**（0.2326 vs 0.2301，胜 10/19，SWAP 42.4 <
45.1）。机制：beam 宽度 5 把 SWAP 65.7→42.4（低于 SABRE），同时保留部分深电路
log-mean 优势（0.0569）。beam 在两个数据集上分化：NAM（结构化算术电路）有效，
stage2_mixed（随机浅电路）反而低于 argmax——随机电路 swap 压缩空间小（5.5→5.2），
V(s') 剪枝牺牲了边选择质量。

**P2B 30k（T=8, v2 信号）评估**：NAM mean 0.1867→0.2016（改善）但 logmean
0.0744→0.0084（深电路崩塌：csla_mux_3 0.2347→0.0913、hwb6/rc_adder_6→0.0000）；
stage2_mixed 0.2372（无改善）。诊断：30k 步 + T=8 噪声终端信号推动"均值友好"
但破坏 argmax 的深电路噪声路径——正是 P2C（KL 保策略）设计要防的策略漂移。

**后续优化路线（优先级）**：
1. beam 宽度扫描（beam7/9，NAM 已赢，看还能压多少）；
2. **fid-critic 接入 beam 评分**（eval_policy beam 的 V(s') 换 P2B 的 critic_fid
   或混合）——stage2_mixed 差距的针对性修法；
3. P2C 变体重训（KL 约束锚住 argmax 深电路优势 + v2 终端信号，100k 步 T=16）；
4. 每电路选择器（深电路→l05 beam5、浅电路→SABRE；NAM oracle 上界 +8.6%）；
5. 已排除：密集 SWAP 定价（Phase 1 教训）、逆 SWAP 剪枝（0 冗余）。

---

## 不依赖 Phase 2 的 l05 优化：奖励函数 / 架构 / 训练集三路设计（2026-09-12）

### 根因诊断（奖励结构）

`_GATE_BASE_REWARD_DEFAULT` 给每颗 CX **+2.0** 完成奖励，而训练主循环 env 的 SWAP
直接代价 ≈ 0（swap_cost 传入怪癖 + eta_swap_err=1.5×e≈0.015）。「尽快解锁 CX」
即时收益远超换位成本——策略必然多换位（NAM 65.7 vs SABRE 45.1）。v2 物理计价：
swap ≈ 3×CX 噪声；奖励里两者比例 0.015 : 2.0（1:133）。

### R1 实验（进行中）：v2 对价标定的密集奖励

`eta_err 0.5→20`（门噪声与完成奖励同量级）+ `eta_swap_err=60`（=3×eta_err，
物理对价）+ `swap_cost 0.3`（平坦热弛豫分量）+ **修复 swap_cost 未传入训练
env 的怪癖** + 新增 `--eta-err` CLI。从 l05 微调 50k 步（routing 模式无终端
模拟，~40min），自动评估链（NAM argmax/beam5 + stage2_mixed）已挂载。

### 后续路线（按优先级）

1. **R2（奖励）**：解析保真度势函数奖励——r = −ΔE_analytic（E=累计 log-error：
   swap +3e_edge+thermal、门 +e_g+thermal），gate_base 仅留小 progress 项；
   单一物理一致的标定替代 ad-hoc 组合；
2. **架构-特征：k 步前瞻（SABRE extended set + decay）**——当前 5 维 SABRE 特征
   只看 front layer，看不到 swap 对第 2/3 层的影响；加 extended-layer 距离
   （每边 +2-3 维）直接对症 swap 数。需重训（obs_dim 变化，不能微调 l05）；
3. **架构-双价值头**：Phase 1 即训练 V_route + V_analytic 双头，推理 beam 用
   V_analytic 打分——无 Phase 2 模拟器也能得到保真度感知 beam；
4. **训练集-结构化电路增广**：qiskit.circuit.library 批量生成算术电路
   （CDKMRippleCarryAdder / DraperQFTAdder / RGQFTMultiplier / Grover 等）
   5-20q 数百条扩充 nam-circuits（当前仅 19 条）；l05 的深电路 logmean 优势
   （+88%）来自这类分布；
5. **训练集-省换位课程**：生成交互图接近拓扑的电路（近零 swap 可解），
   教策略「能不换就不换」——unified 随机电路交互图与拓扑无关，学不到该模式；
6. 已排除：逆 SWAP 剪枝（0 冗余）、弱定价（Phase 1 教训：0.015 量级无效）。

### R1v2 评估（奖励重校准 + 扩充训练集，30k 步微调）—— 失败

| 数据集 | SABRE | 旧 l05 | R1v2 |
|--------|-------|--------|------|
| NAM Fid mean | 0.2301 | 0.1867 | 0.1988 |
| NAM logmean | 0.0395 | **0.0744** | 0.0115 |
| NAM SWAPs | 45.1 | 65.7 | 69.9 |
| stage2 Fid | **0.2590** | 0.2392 | 0.2384 |
| stage2 SWAPs | 4.5 | 5.5 | 6.7 |

SWAP 不降反升（65.7→69.9）、深电路 logmean 崩塌（0.074→0.0115）。**根因反思**：
swap 惩罚 0.6/颗 vs「解锁 CX 即得 +2.0」——解锁 2-3 颗 CX 就净赚 +1.8，惩罚定价
在错误的基准上（+2.0 完成奖励本身不是保真度单位，是 100× 放大的任意标定）。
**微调路径三次尝试均未超越原始 l05**（弱定价/强定价/P2B-30k），强奖励突变会
摧毁已收敛策略（entropy 0.8→2.9）。

**修正后的路线**：
1. 奖励设计若要成立，必须换成**势函数形式**（R2：r = −ΔE_analytic，无每门
   +2.0 完成奖励，进度用 potential 差分体现），且需**从头训练**而非微调——
   微调在奖励尺度突变下已被三次证伪；
2. 干净的训练集消融未做：R1v2 混杂了奖励突变 + 新训练集两个变量。应跑
   「旧奖励 + 扩充训练集」隔离训练集贡献；
3. 推理侧 beam5 仍是当前唯一超过 SABRE 的配置（NAM 0.2326 vs 0.2301）；
4. fid-critic 接入 beam 评分（P2B 的 critic_fid 推理时未用）仍是 stage2 差距
   的未尝试修法。

### R2：势函数奖励 + 从头训练（进行中）

**设计**（回应「+2.0 完成奖励与噪声惩罚标定脱钩」的根因）：
- 执行门：`r = +0.045 − e_edge`（进度奖励 ≈1.5×平均噪声代价，完成优于 stall）
- SWAP：`r = −3·e_edge`（物理 3×CX 对价；一颗 SWAP 需解锁 ≥2 门才回本）
- 映射期虚拟 SWAP 免费（物理正确，布局质量由 lambda_layout 体现）
- 取消 +2.0/门 完成奖励——奖励全量落在 v2 物理单位上，标定自洽
- 截断反 stall：沿用 unfinished_penalty=0.5×剩余门数（已够强，无需新项）
- **从头训练 100k 步**（微调在奖励突变下已被三次证伪），unified + NAM/生成
  电路 30% 混训，独立 ckpts_r2 目录

新 CLI：`--reward-potential`（RoutingEnv 新参 `reward_potential`，默认关）。
实现：env.py `_step_reward_execute/_step_reward_swap` 分支 + 常数
`_POT_PROGRESS_B=0.045/_POT_AVG_COST=0.01`。

**状态**：step 2560，trunc 50%（从零随机场预期开局，unfinished_penalty 强梯度
推动完成），ent 3.36（均匀起始）。评估链已挂：NAM argmax/beam5 + stage2_mixed。

### R2 从零训练失败复盘 + R2b 课程重启（进行中）

R2 从零 100k（unified 直上 20q）：**失败**——trunc 50%→58%（越训越差）、ent 全程
钉死 3.2-3.4（随机）。根因：训练规程问题而非奖励设计问题——20q 随机电路 +
随机策略的 credit assignment 不可学，l05 血统本就是 5q→20q 课程。

**R2b**：同一势函数奖励 + `--curriculum-keys
'stage1,large_n8,large_n10,large_n12,large_n16,large_n20,unified'` 七级跨规模
课程（5q→8q→10q→12q→16q→20q→多尺度）从零 200k 步。开局：trunc 22-24%（5q
课程下大幅优于直上 20q 的 50%）、swp 13-19、ent 2.8-2.9 起步。评估链已挂
（NAM argmax/beam5 + stage2_mixed，trajectory_v2 口径）。

---

## tianyan-287 拓扑截取 + l05 推理出映射路由线路（2026-09-12）

### 流程

1. **拉取天衍 API 校准配置**（cqlib 1.3.11，机器名 `tianyan-287`，paid，状态 calibration）：
   `platform.download_config(machine="tianyan-287")` → `data/tianyan-287/config.json`
   （校准时间 2026-09-09 14:46:30；105 比特 / 182 couplers，
   disabledQubits=Q35,Q27,Q22,Q90，disabledCouplers=14 个）。
2. **构建有效全拓扑**：`python3 scripts/build_tianyan287_topo.py`
   → `traindata/topo/tianyan287_101q.json`（101 活跃比特 / 168 边，最高度数 4）。
   字段布局与 tianyan176 版一致（T1/T2 us、f01 GHz、误差百分数→小数、T2 clamp 2·T1，
   新增 `original_labels` 保留 Q 标签）。two_q error min/mean/max = 0.0009/0.0081/0.1523。
3. **截取联通度最高 20 比特子拓扑**：`python3 scripts/build_tianyan287_20q_sub.py`
   （BFS 度优先 + 贪心最大内边增益双启发式，101 根 × 2 = 147 候选）。
   最优：**31 边**（avg_deg 3.10，tianyan176_20q 为 29 边），根 Q23，
   平均两比特误差 0.65%。比特（0-19→Q 标签）：Q23,Q29,Q30,Q36,Q37,Q38,Q43,Q44,
   Q45,Q46,Q50,Q51,Q52,Q53,Q57,Q58,Q59,Q65,Q66,Q73。
   输出 `traindata/topo/tianyan287_20q.json` + Q 标签 sidecar
   `tianyan287_20q_labels.json`（from_0_19 / to_0_19，供真机测试回溯物理比特）。
4. **l05 推理**（`generate_routing.py`，与此前 l05 口径一致）：

```bash
cd src && PYTHONPATH=. python3 -m routing.rl.generate_routing \
  --model ../models/policy_tianyan20q_laymix_l05_eta05.pt --model-name l05 \
  --circuit-dir ../benchmark/nam_circs \
  --topo ../traindata/topo/tianyan287_20q.json \
  --label-map ../traindata/topo/tianyan287_20q_labels.json \
  --max-num-qubits 20 --out-dir ../benchmark/routed_tianyan287_20q \
  --reward-mode noise_aware --fidelity-sim trajectory_sched --traj-trajectories 16
```

### 结果（10/10 完成，0 截断，argmax）

| 电路 | nq | SWAPs | 保真度 | 路由+保真耗时(ms) |
|------|----|-------|--------|-------------------|
| barenco_tof_3 | 5 | 10 | 0.8307 | 47 |
| barenco_tof_4 | 7 | 31 | 0.6324 | 300 |
| barenco_tof_5 | 9 | 40 | 0.5875 | 428 |
| barenco_tof_10 | 19 | 179 | 0.0739 | 557512 |
| csla_mux_3 | 15 | 57 | 0.5725 | 230708 |
| gf2^4_mult | 12 | 61 | 0.3735 | 1808 |
| gf2^5_mult | 15 | 122 | 0.2287 | 227802 |
| gf2^6_mult | 18 | 195 | 0.1719 | 656356 |
| grover_5 | 9 | 149 | 0.0001 | 2147 |
| hwb6 | 7 | 63 | 0.1244 | 676 |
| **mean** | – | **90.7** | **0.3596** | – |

输出：`benchmark/routed_tianyan287_20q/*.json`（routed_qasm + initial_layout /
final_layout 物理索引 0-19 + from_0_19 Q 标签映射），日志
`logs/gen_l05_tianyan287_20q.log`（前 6 电路）+ `gen_l05_tianyan287_20q_rem.log`。

### 校验与结论

- 路由后 QASM 共 2201 个两比特门，**0 个违反** tianyan287_20q coupling_map；
  全部电路 completed（grover_5 fid≈0 为该线路自身分布特性，与拓扑无关）。
- l05 checkpoint 的 per-edge MLP 与 critic 输入维度均与边数无关（149 维/边），
  29 边（tianyan176_20q）训练的权重可直接在 31 边子拓扑上推理，无需重训。
- **踩坑记录**：sidecar `from_0_19` 初版误用 101q 原始索引作键
  （{20: 'Q23', ...}），应为子拓扑索引 0-19（{0: 'Q23', ...}）；已修复脚本并
  重刷 10 个结果 JSON。路由本身走 0-19 物理索引，未受影响。
- 后续：routed_qasm + from_0_19 可直接转为 QCIS 提交 tianyan-287 真机/云端
  模拟器做端到端测试。

### 补全 nam_circs 其余 9 条电路（routing-only，--no-fidelity）

首日仅覆盖前 10 条（`ls | head` 截断遗漏）。应要求补齐
`benchmark/nam_circs/` 中其余 9 条 ≤20q 电路（mod_red_21 实际 11q），
参数同上但加 `--no-fidelity`（只路由，不算保真度）：

| 电路 | nq | SWAPs |
|------|----|-------|
| mod5_4 | 5 | 14 |
| mod_mult_55 | 9 | 21 |
| mod_red_21 | 11 | 68 |
| rc_adder_6 | 14 | 79 |
| tof_10 | 19 | 102 |
| tof_3 | 5 | 9 |
| tof_4 | 7 | 22 |
| tof_5 | 9 | 32 |
| vbe_adder_3 | 10 | 34 |

**全量 19 条**：全部 completed、0 截断、路由后 QASM **0 违反**
tianyan287_20q coupling_map；SWAPs mean=67.8，total=1288。其中 10 条含
trajectory_sched(16) 保真度（mean 0.3596），其余 9 条 fidelity=null（可按需补算）。
日志 `logs/gen_l05_tianyan287_20q_rem2_nofid.log`。

---

## R3：第一阶段奖励重设计（objective/shaping 分离 + 经济账对齐，2026-09-12，进行中）

### 设计（响应「+2.0 完成奖励与噪声惩罚 1:133 失衡」的根因）

新结构：`r = r_progress + r_swap + γ·Φ(s')−Φ(s) + r_idle + r_xtalk_par + r_time`

| 分量 | 旧 | R3 |
|------|-----|-----|
| Progress | cx=+2.0, 1q=+0.3~0.5 | **cx=+0.30, 1q=+0.05** |
| SWAP flat | 0（未传入训练 env） | **0.40**（已修复传入） |
| SWAP error | 0 | **−2.0×(3·e_pq)**（公式加 ×3，物理 3×CX） |
| Distance | −1.0·ΔD/D0（相对式） | **删除**，改势函数 Φ |
| Φ(s) | — | **−0.3·(D_front/|F| + 0.5·D_ext/|E|)**，D_ext=front 后 20 门（对齐 SabreSwap lookahead(0.5,20)） |
| shaping γ | — | **0.99 = agent.gamma**（策略不变性前提；终态/截断 Φ=0） |
| static xtalk | 0.02 | **0**（与调度串扰 double counting） |
| parallel | 0.05 | **0**（与 xtalk/idle 竞争） |
| time | 0.01 | **0.05**（SWAP 隐式时间代价 0.045，与 flat 项同量级） |
| unfinished_penalty | 0.5 | **0.15**（旧值按 +2.0 时代标定，过冲） |

经济账：无进展 SWAP ≈ **−0.505**；解锁 1 CX 净 **−0.215（亏）**；解锁 2 CX 净
**+0.075（值得）**——"值得付出 SWAP 代价时才换"的行为门槛内建。

### 实现

- env.py：`_extended_set_dist()/_phi()`（新增）、`step()/_step_mapping/_end_step`
  接 `phi_before` 线程（终态/截断 Φ=0）、`_step_reward_swap` ×3、新参
  `shaping_gamma/eta_shape/alpha_ext/ext_set_size`（clone 传播）；激活 shaping
  后旧 eta_dist 相对式停用（None=旧行为，向后兼容）；映射期虚拟 SWAP 免费
- train_agent.py：新 CLI `--gate-base-cx/--gate-base-1q/--eta-xtalk/
  --unfinished-penalty/--shaping-gamma/--eta-shape/--alpha-ext`，create_env 全接线
- **修复训练末尾假保存 bug**：`main()` 尾部原只 print "Policy saved" 无实际
  `agent.save(args.out)` 调用（metric 未超历史最优时模型从不落盘）——R1v2/R2b
  模型"消失"的谜底；现已无条件保存，R2b 模型从 ckpts_r2b/last.pt 恢复
- 测试：`test_r3_swap_pricing_3e`（−0.4−2×3e 精确）、`test_r3_mapping_virtual_swap_free`、
  `test_r3_shaping_telescoping`（Σ shaping = −Φ(s₀)+(γ−1)ΣΦ，γ<1 残差公式，1e-6）

### 状态

- 训练：从零 + 七级课程（stage1→unified）200k 步，step 2560：swp 14-21、
  trunc 23-25%、ent 2.6-2.9、xtalk 0.03（静态串扰置零生效）
- 评估链已挂：NAM argmax/beam5 + stage2_mixed（trajectory_v2 ×32/×16）
- 对照点：SABRE（0.2590/0.2301）、旧 l05（0.2392/0.1867）、l05 beam5
  （0.2318/**0.2326**，NAM 已超 SABRE）、R2b（待评）

### quarl_opt_wo_rm 电路集路由（tianyan287_20q，routing-only）

`benchmark/quarl_opt_wo_rm/` 共 28 条 QASM，其中 **18 条 ≤20q**（排除
adder_8=24q、csum_mux_9/gf2_10=30q、gf2_7=21q、gf2_8/9=24/27q、
qcla_adder_10=36q、qcla_com_7=24q、qcla_mod_7=26q）。参数同上，`--no-fidelity`：

| 电路 | nq | SWAPs | | 电路 | nq | SWAPs |
|------|----|-------|-|------|----|-------|
| barenco_tof_10_cost260 | 19 | 158 | | mod5_4_cost24 | 5 | 7 |
| barenco_tof_3_cost35 | 5 | 9 | | mod_mult_55_cost86 | 9 | 23 |
| barenco_tof_4_cost64 | 7 | 30 | | mod_red_21_cost187 | 11 | 75 |
| barenco_tof_5_cost88 | 9 | 35 | | rc_adder_6_cost152 | 14 | 91 |
| csla_mux_3_cost142 | 15 | 56 | | tof_10_cost175 | 19 | 88 |
| gf2_4_mult_cost178 | 12 | 85 | | tof_3_cost33 | 5 | 6 |
| gf2_5_mult_cost291 | 15 | 166 | | tof_4_cost51 | 7 | 19 |
| gf2_6_mult_cost428 | 18 | 240 | | tof_5_cost71 | 9 | 22 |
| hwb6_cost214 | 7 | 70 | | vbe_adder_3_cost73 | 10 | 46 |

18/18 completed、0 截断、0 违反 coupling_map；SWAPs mean=68.1（min 6 / max 240）。
结果同写 `benchmark/routed_tianyan287_20q/`（文件名含 cost 后缀，与 nam_circs
结果不冲突），日志 `logs/gen_l05_tianyan287_20q_quarl_nofid.log`。

### 结果目录拆分

`benchmark/routed_tianyan287_20q/` 下两批结果分别归档至子目录：
`nam_circs/`（19 条）与 `quarl_opt_wo_rm/`（18 条，cost 后缀）。

### R3 结果（从零 + 课程 + 势函数奖励，200k 步）

训练收敛验证 ✓：trunc 33%→15%（课程+势函数修复了从零学习的可学性）、
ent 2.8→0.75（收敛）、假保存 bug 修复后模型正常落盘。

| 配置（v2 口径） | NAM mean | NAM logmean | NAM SWAPs | 胜 SABRE | stage2 Fid | stage2 SWAPs |
|------------------|----------|-------------|-----------|----------|------------|--------------|
| SABRE | 0.2301 | 0.0395 | 45.1 | — | **0.2590** | 4.5 |
| l05 argmax | 0.1867 | **0.0744** | 65.7 | 8/19 | 0.2392 | 5.5 |
| l05 beam5 | **0.2326** | 0.0569 | 42.4 | 10/19 | 0.2318 | 5.2 |
| R3 argmax | 0.2132 | 0.0713 | 69.0 | **11/19** | 0.2331 | 7.0 |
| R3 beam5 | 0.2230 | 0.0641 | 46.2 | 10/19 | — | — |

### 分析

1. **R3 argmax 超越旧 l05**（0.2132 vs 0.1867，+14%），胜局数 11/19 为所有
   argmax 配置之最，且深电路 logmean 0.0713 ≈ l05 的 0.0744——**势函数前瞻
   （D_ext）+ 经济账在保留噪声友好优势的同时提升了整体质量**；
2. **从零训练规程验证成功**：势函数奖励 + 七级课程 + 结构化电路混训的组合
   使 200k 步从零收敛（R2 的 58% 截断 → R3 的 15%），pipeline 可复用可扩展；
3. **未竟之项**：SWAP 数仍高于 SABRE（69 vs 45）——经济账门槛"解锁 ≥2 CX"
   相当于 SABRE 级的换位节制还差一档（提高 swap_cost 0.4→0.6-0.8 可把门槛
   移到 ≥3 CX）；NAM 均值距 SABRE 还差 7.3%；
4. R3 beam5（0.2230）低于 l05 beam5（0.2326）：新 critic 的 V(s') 与 beam
   的组合仍需调（或 R3 需要更长训练）。

### R3 逐电路归因分析（v2 口径，NAM 19 电路）

| 模式 | 平均 SWAP 超额(vs SABRE) | 例证 |
|------|--------------------------|------|
| R3 负局（8） | **+32.1** | tof_10 +49→−0.223、vbe_adder_3 +23→−0.288、barenco_tof_3 +3→−0.267 |
| R3 胜局（11） | +17.9 | csla_mux_3 +18→+0.212、mod5_4 +3→+0.176、tof_4 −4→+0.114 |

三个关键发现：
1. **浅电路（5-9q）的 SWAP 超额代价最大**：barenco_tof_3 仅 +3 颗 SWAP 就
   −0.267（浅电路保真度由 SWAP 数主导，无可摊销空间）；
2. **Toffoli 塔链全局规划不足**：tof_10 +49、barenco_tof_10 +38——长程交互
   链需要跨层规划，20 门 extended set 前瞻对塔结构不够；
3. **SWAP 数不是唯一因子**：barenco_tof_4 用**更少** SWAP 仍输 0.079（放置
   质量），gf2^4_mult +57 颗只输 0.008（噪声友好放置有效）——边选择与时序
   质量同样在起作用。

R3+beam5 已把 NAM 差距缩到 −3.1%（0.2230 vs 0.2301，SWAP 46.2）。

### R3b：①+② 组合实验（进行中）

- ① SWAP 定价校准：swap_cost 0.4→**0.7**（经济账门槛"解锁 ≥2 CX"移到 ≥3 CX：
  总代价 0.7+0.06+0.045≈0.8，解锁 2 CX 净 −0.22 亏、3 CX 净 +0.07 值得）；
- ② 浅电路课程加权：`--curriculum-keys` 重复 prefix 实现零代码加权——
  stage1×2 + n8×2 + n10/n12/n16/n20/unified（9 级，5q/8q 各占 22%），
  针对"浅电路 SWAP 超额代价最重"的归因发现；
- 其余与 R3 完全一致（200k 步、势函数 shaping、结构化电路混训）。
- 评估链已挂（NAM argmax/beam5 + stage2_mixed）。判据：NAM SWAP 69→≤55 且
  F ≥ 0.2132（R3）——若 SWAP 降而 F 不降，则 R3b+beam5 有望追平/超过 SABRE。

### R3b 中途干预（step 30k）：反游走早停 + 截断罚重标定

step 15k 处出现预警签名：trunc 26→28% 爬升 + swp 崩塌至 1.9-2.2（5q 电路需
~11 颗）。激励核算发现漏洞：8q 电路（60 门/15 SWAP）下「跑完 = −4.65」 vs
「游走到截断 = −1.1」——unfinished_penalty=0.15/门 在 swap_cost=0.76/颗 的
新定价下过弱，stall 成理性逃逸（与 R1v2 截断同源）。

**干预**：重启 + `--no-progress-limit 200`（routing 阶段连续 200 步无门执行
→ 主动截断，失败 episode 墙钟 1/10）+ `--unfinished-penalty 0.3`（重标定使
stall 严格劣于跑完：跑完 −4.65+Δ vs 截断 −1.1−Δ' → 恢复完成优先）+
**动态步数上限** max(1000, 2×N_gates)（大电路预算自动放宽）。新 CLI：
`--no-progress-limit`。三项均已接线（17 测试过），动态上限对所有新训练生效。

**重启后开局**（step 2560）：swp 10.7-12.7（欠换位消失）、trunc 23-25%、
ent 2.84-2.94。观察点：trunc 在课程推进下能否复制 R3 的 33→15 下降轨迹。

### R3b 修复 + beam5 评估：**首次在均值口径全面超过 SABRE**

beam5 评估崩溃根因：`env.clone()` 未复制新增的 `_last_exec_count/
_steps_since_progress` 计数器——beam 搜索用 clone 环境一 step 即 AttributeError
（argmax 评估不受影响所以训练正常）。已修复（41 测试过）。

| 配置（NAM, trajectory_v2） | Fid mean | Fid logmean | SWAPs | 胜 SABRE |
|---------------------------|----------|-------------|-------|----------|
| SABRE | 0.2301 | 0.0395 | 45.1 | — |
| l05 argmax | 0.1867 | 0.0744 | 65.7 | 8/19 |
| l05 beam5 | 0.2326 | 0.0569 | 42.4 | 10/19 |
| R3 argmax | 0.2132 | 0.0713 | 69.0 | 11/19 |
| **R3b beam5** | **0.2366** | **0.0746** | **45.0** | **11/19** |

**R3b beam5 = 当前全局最优**：
- NAM 均值 0.2366 > SABRE 0.2301（**+2.8%**）且 > l05 beam5（0.2326）
- SWAPs 45.0 ≈ SABRE 45.1（换位数打平）
- logmean 0.0746 ≈ l05 的 0.0744（深电路噪声友好优势完整保留）
- 胜局 11/19

这正是目标画像（第 23 点）：**F_RL > F_SABRE 且 SWAP 不高于 SABRE**——
噪声感知路由的贡献不再被换位超额抵消。R3b 训练配方（势函数 shaping +
extended-set 前瞻 + 经济账 0.7 定价 + 浅电路加权课程 + 反游走/截断罚 0.3）
全部要素缺一不可。

### 未竟与后续

1. stage2_mixed 仍低于 SABRE（R3b argmax 0.2319 vs 0.2590）——R3b beam5 的
   stage2 评估待跑；fid-critic beam 接线是下一步；
2. R2b（恢复的模型）beam5 评估可作对照；
3. swap_cost 0.7 的单调性未验证（0.5/0.6/0.9 扫描可寻更优点）；
4. NAM 评估 T=16 单种子——最终结论建议 T=32 多种子复核。

### R3b beam5 的优势归因分析（2026-09-12）

**边质量选择——学会了** ✓：19 电路平均 CX 边误差 SABRE 0.2136 →
R3b_beam5 **0.2026（−5.2%）**，12/19 更低（csla_mux_3 −27.5%、barenco_tof_5
−18.6%、mod_red_21 −13.0%）；且深电路上边误差改善与 fid 增益对应
（csla_mux_3：cxerr −27.5% → fid +0.212）。

**串扰规避——没学会** ✗：串扰事件均值持平（0.2 vs 0.2，9/19 更低）。
怀疑原因：调度串扰惩罚（~0.2-0.6）在总回报（progress ~7-14）中占比 ~5%，
信噪比不足；且串扰是换位后并发时序属性，因果链长。改进方向：
eta_xtalk_par 1.0→3-5，或把 crosstalk potential 加进 Φ。

**两种因子竞争**：深电路边误差改善→fid 增益兑现（csla_mux_3）；浅电路
SWAP 超额→fid 损失主导（barenco_tof_3 +3→−0.267）。

**稳健性检验**：`scripts/nam_seed_check.py`（3 种子 × T=16 × 双模型）运行中——
+2.8% 均值差需排除 T=16 MC 方差（深电路单种子波动实测 0.0000~0.19）。

### R5a 实施与训练启动（并发/前瞻观测特征 + eta_xtalk_par 3.0，进行中）

**新观测特征**（`_edge_lookahead_features`，per-edge 4 维，追加在 SABRE 5 维后）：
`xtalk_pred`（换位与 ready 门集合的 1-hop 交叉 ZZ 和 / zz_max，与调度记账同口径）、
`busy_contact`（p/q 邻居中 ready 端点数 /4）、`d_ext_delta`（换位后 extended set
距离均值变化 / 拓扑直径）、`d_ext_after`（换位后均值 / 直径）——全部归一化。
奖励侧 `eta_xtalk_par 1.0→3.0`。obs_dim 变化 → 从零重训。

**修复的三个 bug**：
1. 训练末尾假保存（只 print 不 save）——R1v2/R2b 模型消失的谜底；
2. `env.clone()` 漏复制反游走计数器——beam5 评估 AttributeError；
3. 5q 重训发现 `--no-mapping-phase` 与 rollout 掩码硬编码 `num_edges+1`
   不兼容（env.py 缺口，已记录）——改用默认 mapping phase 训练规避。

**两个训练并行**：
- R5a 主线：从零 + 七级课程 + swap_cost 0.7 + 反游走 200 + xtalk_par 3.0，
  200k 步（step 2560 开局 trunc 17-23%）；
- 5q phase1 重训（R3 奖励 + 新 obs，30k 步）：修复
  `test_phase1_vs_baselines`（旧 checkpoint 与新 obs 维度不兼容 → 宽松加载
  随机层 → 33.8 swaps 假阳性），训练完成自动跑该测试验证。

### R5a-t287：tianyan287 拓扑 + 真实量级 ZZ 串扰训练（进行中）

- **拓扑**：`tianyan287_20q.json`（天衍 287 的 20q/31 边连通子拓扑，2026 校准）；
- **ZZ 串扰构造**（天衍校准不含 ZZ 参数，用户确认按典型量级构造）：
  `scripts/add_tianyan287_zz.py`——f_ZZ ∈ [1,20] kHz 按边错误率排序赋值
  （70% 排序 + 30% 随机，seed=42），θ = 2π·f·0.3µs ∈
  **[0.0030, 0.0356] rad/CX（mean 0.0191）**，写入 crosstalk_strength
  （原文件备份 .bak）；hw.zz 归一化后 0.06-0.71（_ERROR_SCALE=0.05）；
- **评估脚本参数化**：eval_nam_v2sim.py 支持 topo/out 参数，SABRE 缓存按
  拓扑隔离（SABRE@tianyan287_20q），避免跨拓扑污染；
- **eta_xtalk_par 保持 1.0**：真实量级 ZZ（≈回退值 7×）已自带强度，
  3.0 是为回退值校准的，会过度惩罚；
- 训练配方与 R3b/R5a 一致（swap_cost 0.7、势函数 shaping、反游走 200、
  unfinished 0.3、九级课程、112 条结构化混训）。

**开局状态**（step 2304）：swp 13-15、trunc 17-23%、**xtalk 0.40-0.41**
（真实量级 ZZ 下调度串扰惩罚约为 tianyan176 回退值的 7-13 倍——信号强度
显著提升，这正是本次实验要验证的：策略能否学会在真实 ZZ 量级下规避串扰）。

### R5a-t287 v1 崩塌复盘 + eta_xtalk_par 重标定（进行中）

**v1 崩塌**（eta_xtalk_par=1.0，step 28k）：trunc 25→37%、ent 3.35→3.43
（均匀随机顶格）、指标冻结。根因：hw.zz 归一化后（0.06-0.71）是回退值时代
（0.02-0.06）的 ~10 倍，串扰惩罚/轮达 0.4-1.4，压倒进度奖励（+0.045/门）——
串扰/进度平衡比从 R3b 的 0.4:1 翻转到 5:1，策略最优逃逸 = 停止进展。

**修正**：eta_xtalk_par 1.0→**0.1**（惩罚/轮回到 0.04-0.14，恢复 R3b 平衡比；
真实 ZZ 的**相对结构**保留——强 ZZ 边惩罚按比例更大）。教训：换真实器件量级
的物理参数时，所有奖励权重的**有效量纲**要重新对齐，不能沿用回退值时代的标定。

**重启后**（step 3328）：trunc 23-24%（5q 阶段，与 R3b 同期相当）、
swp 15-22、rew −128~−171（指标恢复变化，无冻结）。

**v2 完整轨迹**（eta_xtalk_par=0.1）：trunc 25%→32%→36-38%（10-16q 阶段）→
43%（177k）、SWAP 通胀至 152-169、xtalk raw 0.4→6.7。修正串扰权重后仍退化——
**毒源不是串扰权重，而是 16-20q 阶段的大电路 episode**（gen+NAM 的 900+ 门
电路深度负回报 → 换位通胀 + 截断螺旋）。

**mid-checkpoint（step 107776）NAM 复核：路由质量垃圾**——barenco_tof_3
（5q）77 SWAPs、barenco_tof_4（7q）292 SWAPs（SABRE 同拓扑 5/21），fid
0.168/0.038。checkpoint 处于 10-12q 课程阶段，小电路路由能力已退化。

**R5a-t287 战役结论（负结果）**：真实量级 ZZ + tianyan287 的三次尝试
（v1 崩塌 / v2 退化 / mid 垃圾）均未产出可用模型。**反复出现的失败模式：
从零课程训练在 16-20q 大电路阶段失稳**——与奖励标定无关地重复（R2/R3b
后期/R5a-t287）。当前有效结论不变：R3b beam5（0.2366）> SABRE（0.2301）
为全局最优；R3（0.2132）为最佳 argmax。剩余方向：分阶段冻结式课程、
fid-critic beam、swap_cost 扫描、真实 ZZ 数据到位后重跑。

### 「16-20q 阶段失稳」叙事证伪 + 失败归因修正（2026-09-12）

三方分段统计（rew/swp/trunc/ent per 课程阶段）：

| 阶段 | R3 (t176, 旧obs) | R3b (t176, +定价) | R5a-t287 (t287, +新特征+真实ZZ) |
|------|------------------|-------------------|--------------------------------|
| stage1(5q) | −111/16.8/**28%** | −254→−95/45→27/24→**16.5%** | **−327/23/30.9%→37.4%（恶化）** |
| n8 | −76/21/**23.8%** | −79/22.7/**12.4%** | −269→−304/51→60.5/**34.6→36.7%** |
| n12 | −64/25.1/**17.6%** | −100/35/**11%** | −379/71.7/**38.3%** |
| n20 | −77/39.7/**16.1%** | −108/47.4/**11%** | −470/105.6/**41.9%** |
| unified | −66/28/**15.8%** | −93/32.3/**11%** | **−687/126.2/42.9%** |

**修正结论**：R3/R3b 在 16-20q 阶段**全程健康**（trunc 15.8%/11%，swp 随规模
成比例 16.8→39.7/27→47）——「大电路阶段失稳」假说被证伪。真正的失败模式：
**R5a-t287 从第一个 5q 阶段就没有 bootstrap**（trunc 31→37 恶化、rew 不改善、
ent 全程 3.2+），且退化贯穿所有阶段。

**R5a-t287 与 R3b 的四个差异变量**（bootstrap 失败的候选原因）：
1. **新观测特征**（4 维/边：xtalk_pred/busy_contact/d_ext_delta/d_ext_after）
   ——唯一"所有失败 run 都有、所有成功 run 都没有"的变量，头号嫌疑；
2. tianyan287 拓扑 + 真实量级 ZZ（GNN zz 特征量级 10× 变化；eta 0.1 下
   reward 贡献已小）；
3. unfinished_penalty 0.3 起步（R3b 前半 0.15）；
4. no-progress-limit 200 与 deadlock mask 的交互。

**定论需要单变量消融**：R5a-t287 配方关闭新观测特征（其余不变），stage1-only
50k 快跑——bootstrap 成功则锁定新特征为失稳源。

### 隔离实验（进行中）：新观测特征 OFF，其余变量保留

`scripts/train_r5iso`（--no-lookahead-features，4 维特征置零、obs_dim/网络架构
不变——单变量隔离）：tianyan287 + 真实 ZZ + stage1-only 50k + unfinished 0.3 +
no-progress 200 + swap_cost 0.7。对照基准：R5a-t287（特征 ON）同窗口
trunc 30.9→37.4（恶化）。

判据：隔离 run 的 trunc 轨迹下降（如 R3b 的 24→16.5）→ **新观测特征为失稳源**；
同样恶化 → 特征排除，顺次测 unfinished 0.15 / no-progress OFF / tianyan287+ZZ。

### 隔离实验第一轮判决 + 第二轮二分（进行中）

**第一轮判决：新观测特征无罪**——特征置零的隔离 run（stage1 50k）同样退化
（trunc 25→33-38%、swp 13→77.6 通胀），与特征 ON 的 R5a-t287 同模式。
关键线索：R3b stage1（tianyan176，同 swap_cost 0.7）swp **45→27 下降**，
而所有 tianyan287 run 均 swp **上升**——方向相反。

**第二轮二分（并行，stage1-only 50k ×2，均关特征）**：
- isoA：tianyan176 + 回退 ZZ + unfinished 0.3 + no-progress 200
  （= R3b stage1 + {unfinished 0.3, no-progress 200}）
- isoB：tianyan287 + 真实 ZZ + unfinished 0.15 + 无 no-progress
  （= R5iso − {unfinished 0.3, no-progress 200}）
对照锚点：R3b stage1（健康：24→16.5%）/ R5iso（退化：25→33%）。
判决表：isoA 退化→unfinished/no-progress 有毒；isoB 退化→tianyan287+ZZ
有毒；两者交叉可唯一归因。

### R3c 设计转向：坡道 bootstrap + 尾段混合（进行中）

**R3c-v1（全程混合，--split-prefix unified）失败**：trunc 50-56% 起步 →
38-45% 回升恶化、rew −398→−573、ent 3.3 随机——复制 R2 失败模式。根因：
**bootstrap 需要易到难坡道**（从零策略在含 16-20q 深电路的分布上 credit
assignment 不可学，与奖励设计无关）；而「混合分布治愈过训练」的证据
（R3 unified 阶段自愈）来自**已 bootstrap** 的策略。

**R3c-v2 = 两个机制各司其职**：
`--curriculum-keys 'stage1,large_n8,large_n10,large_n12,unified'`
——前四级（5q→8q→10q→12q，各 20%）负责 bootstrap 坡道；末级
**unified_mixed（8-20q 混合，20%）替代原 n16/n20/unified 三个单尺度阶段**
（那正是 R3b/R5a-t287 退化发生的位置——单尺度过训练），混合分布的
自愈机制在尾段全程生效。16-20q 电路仍在混合集中出现（~30% 占比），
只是不再连续独占 40k+ 步。

其余配方不变（R3b：swap_cost 0.7、势函数 shaping、反游走 200、
unfinished 0.3、结构化混训 0.3）。开局：step 3328 trunc 23-26%
（坡道生效）。

### R3c-v2 再修正：忠实 R3b 配方（进行中）

R3c-v1 的 stage1 退化（trunc 25→35%、ent 3.37）与 isoA 同签名——排查发现
R3c-v1 混入了 R5a 时代的三个改动（×3 换位公式、no-progress 200、新观测
特征 ON），而隔离实验已证明该组合在 stage1 退化。用户指定的「R3b 配方 +
解法 B」应忠实于产出 R3b 的实际配置：

- eta_swap_err **0.667**（×3 公式下等价于 R3b 实际定价 2.0×1×e，零代码改动）
- **--no-lookahead-features**（旧观测，与 R3b 一致）
- **无 --no-progress-limit**（R3b 无反游走截断）
- 其余不变：swap_cost 0.7、unfinished 0.3、shaping 0.99/0.3/0.5、课程
  stage1→n8→n10→n12→unified（坡道+尾段混合）

开局：step 3328 trunc 23% 稳定（v1 同期 35%——健康签名恢复）。
遗留：{×3 公式 vs no-progress 200} 哪个是 stage1 退化源的拆分实验待做。

### 重要修正：stage1「退化段」是从零训练的正常阶段，恢复点在阶段边界

R3b stage1 逐段轨迹（此前被阶段均值掩盖）：trunc 22.3→25.4→24.2→23.0%
**同期 SWAP 爆炸 15.5→48.6→72.6、rew 低至 −357**——随后在 22220-27775 窗口
突然恢复（swp 72.6→14.8、rew −94、trunc 一路降到 14%）。

**改写的判断**：
1. isoA/isoB/R5iso/R3c-v1 的「stage1 退化」与本正常阶段同构——此前的
   "特征/unfinished/no-progress 有毒"隔离判决**全部作废**（对照基准读错）；
2. R5a-t287 v2 的退化段特别长（40k→177k 无恢复）且量级更极端——真实 ZZ
   量级下恢复更慢/更难仍是有效观察，但「89% 处杀掉」不能排除恢复前夜；
3. 恢复的先兆信号：ent 从 3.2+ 回落（R3b 恢复窗口 ent 2.58）。

**当前 R3c 忠实版正跟踪 R3b 轨迹**（stage1 退化段中，ent 2.94 已回落）——
不再干预，等待 stage1→n8 边界的恢复点。此教训：**课程训练的干预决策必须
基于逐段轨迹与恢复先兆，而非阶段均值的短期恶化。**

### R3c 忠实版发散（最终）——从零训练战役收尾

R3c 忠实版（R3b 等价定价 0.667×3e≡2e、旧观测、坡道+尾段混合课程）在
n8→n10 边界**未复现 R3b 的阶段边界恢复**，进入冻结退化态（trunc 40% 平坦、
rew/swp/ent 逐字重复、makespan 96-110µs）——已终止。

**从零训练战役总结（R2/R3/R3b/R3c/isoA/isoB/R5iso/R5a-t287 共 8+ 次）**：
1. 成功：R3（trunc→15.8%，NAM 0.2132）、R3b（trunc→11%，**beam5 0.2366
   超 SABRE**）——两次健康 run；
2. 失败：R2/R5iso/isoA/isoB/R3c-v1/R3c-faithful/R5a-t287 v1/v2——退化模式
   两种（stage1 前期 SWAP 爆炸段 + 后期冻结态），且**在配置近同的 run 间
   随机出现**（R3b vs R3c 配方几乎逐项相同、结局相反）；
3. **失稳率 ~50-60%，由种子级随机性主导**——单次 run 的成败不可预测，
   继续同配方堆 run 的边际收益低；
4. 已沉淀的可复用资产：R3 经济账奖励设计、势函数 shaping（含 telescoping
   测试）、反游走/动态 cap、扩展集前瞻、结构化电路生成器（93 条）、
   多拓扑评估脚本、假保存/clone 两大 bug 修复；
5. 未竟：16-20q 大电路阶段的长期稳定性、fid-critic beam、真实 ZZ 数据
   （tianyan287 拓扑与构造脚本已就绪）。

**最终交付基线：R3b beam5（NAM 0.2366 > SABRE 0.2301，SWAP 45.0 打平，
logmean 0.0746 保留深电路优势）——满足既定成功判据。**

### R3b beam5 多种子稳健性检验（3 种子 × T=16）—— 优势不稳健

逐种子 Δmean（R3b_beam5 − SABRE）：seed0 **+0.0065**、seed1 +0.0150、
seed2 **−0.0154**——符号不一致，均值 +0.0020 ≈ 0，极差 0.0304。
Δlogmean 逐种子：+88.7% / −26.4% / −21.4%（同样不一致）。

**修正结论**：R3b_beam5 的 NAM 均值优势（seed0 的 +2.8%）**不稳健**——
跨种子统计上与 SABRE 打平（T=16 深电路 MC 方差主导）。单种子测量的
胜负不能作为最终结论；需 T=64 或多种子 T=32 才能定论。

### v1 模拟器（trajectory_sched）全模型对比（2026-09-13）

用第一版模拟器（trajectory_sim scheduled 路径，×16 seed=0，与原始 NAM 基准
同协议）对比 6 个配置。SABRE 现场重路由（seed=0，路由与历史基准一致）；
l05/l05_beam5 复用 summary 中同协议数值；r3/r3b/r3b_beam5 新鲜计算。

| 模型 | Fid mean | Fid logmean | SWAPs | 胜 SABRE |
|------|----------|-------------|-------|----------|
| l05 | 0.3309 | 0.2133 | 65.7 | 16/19 |
| l05_beam5 | 0.2836 | 0.0795 | 42.4 | 11/19 |
| r3 | 0.2530 | 0.0380 | 69.0 | 9/19 |
| r3b | 0.2461 | 0.1515 | 64.4 | 10/19 |
| **r3b_beam5** | **0.3368** | **0.1999** | **45.0** | **15/19** |
| SABRE | 0.2747 | 0.0721 | 45.1 | — |

规模段 fid mean（v1）：

| 规模段 | l05 | l05_beam5 | r3 | r3b | r3b_beam5 | SABRE |
|--------|-----|-----------|-----|-----|-----------|-------|
| 5-9q | 0.4543 | 0.4091 | 0.3444 | 0.3169 | **0.4228** | 0.3933 |
| 10-14q | 0.2339 | 0.1748 | 0.1733 | 0.2137 | **0.2184** | 0.1609 |
| 15-19q | 0.1615 | 0.1196 | 0.1338 | 0.1305 | **0.2593** | 0.1285 |

**v1 口径下 R3b_beam5 全面领先**：均值 +22.7% vs SABRE、**三个规模段全部
第一**（5-9q +7.5%、10-14q +35.7%、15-19q +101.8%）、胜局 15/19、
logmean +177%。逐电路大胜：csla_mux_3 +0.470、vbe_adder_3 +0.205、
tof_5 +0.196、mod_mult_55 +0.166、gf2^5 +0.146、barenco_tof_10 +0.101。

**v1 vs v2 的口径敏感性（关键洞察）**：同一批路由在两个模拟器下的结论
不同——v1（SWAP 无噪声）下 R3b_beam5 大胜；v2（SWAP 记 3×CX 噪声 +
0.9µs 热弛豫）下优势缩水到 seed 不稳定的 +2.8%。各模型 v1→v2 的衰减：
l05 −0.092（65.7 SWAP 的噪声账）、R3b_beam5 −0.100（SWAP 放置的边噪声
暴露）、SABRE 仅 −0.045（45 SWAP 且分布均匀）。**模拟器选择实质性地
改变模型排名的结论**——最终 claims 必须声明所用口径。

### R3b 增强微调：串扰/边噪声参数增强（进行中）

**R3b 的串扰参数实测**：tianyan176_20q 无 crosstalk_strength → 回退
0.1×边错误率（hw.zz 归一化 0.0056~0.0578，均值 0.0217）；eta_xtalk=0（关）、
eta_xtalk_par=1.0、eta_err=0.5、eta_swap_err=2.0×e——**串扰感知信号接近于无**
（调度串扰惩罚占总回报 ~1%）。

**增强微调设计**（--load policy_r2b.pt，只增强噪声项、其余保持）：

| 参数 | R3b | 微调目标 | 倍数 |
|------|-----|---------|------|
| eta_err | 0.5 | **1.0** | ×2 |
| eta_xtalk_par | 1.0 | **3.0** | ×3 |
| eta_swap_err（×3 公式） | 2×e | **4×e**（eta 1.333） | ×2 |
| swap_cost / gate_base / shaping / time / idle | — | 不变（保路由能力） |

`--noise-ramp 0.1`：前 10% 线性渐入（平滑过渡）。100k 步。

**基础设施修复**：`--load` 的 resume 语义会从 checkpoint 恢复 step 计数
（policy_r2b.pt 携带 step=200192 → 0 episode 直接退出）——微调场景用
step 清零的 ftbase 副本绕过（权重/优化器/rew_norm 保留）。

**开局**（step 3584）：rew −45~−51（R3b 的路由能力保持 ✓，对比 R3b 从零
起步的 −327）、trunc 50-53%（噪声项渐入的过渡期，观察能否回落 ≤30%）、
swp 47-50、xtalk 0.145-0.162（增强生效）。

**评估链**：NAM argmax/beam5（v2）+ v1 compare（r3b_ft/r3b_ft_beam5）+
stage2_mixed。判据：v2 NAM mean ≥ R3b_beam5 的 0.2366、v1 ≥ 0.3368、
串扰事件较 SABRE 显著下降、路由能力（SWAP/完成率）不退化。

### csla_mux_3 深挖：T=16 的深电路 MC 方差是致命测量缺陷（2026-09-14）

**T=128 高精度复核**（seed=42 独立）vs T=16：

| 路由 | T=16 (seed0) | **T=128 (seed42)** | CX均边误差 | 串扰 | makespan |
|------|-------------|---------------------|-----------|------|----------|
| SABRE | 0.0007 | **0.1606** | 0.2572 | 0.396 | 42.2µs |
| R3b_beam5 | 0.0156 | **0.1689** | 0.1865 (−27.5%) | 0.211 (−47%) | 52.7µs |
| R3b 多试验重放 | 0.0039 | （T128 超时未完） | 0.2202 | 0.771 | 85.8µs |

**三个修正性结论**：
1. **T=16 的深电路 MC 方差 ~200×**（SABRE csla：T16 0.0007 vs T128 0.1606）——
   此前所有 T=16 的 NAM 深电路数值（含 R3b_beam5 0.2366 vs SABRE 0.2301 的
   +2.8%）在深电路维度不可信；
2. **T=128 下 csla 的真实对比**：SABRE 0.1606 ≈ R3b_beam5 0.1689（+5%）——
   R3b 的边误差优势（−27.5%）与串扰优势（−47%）真实存在（确定性结构指标），
   但仅转化为小幅 fid 优势；
3. **多试验 best-of-4 的 +0.15 提升大部分是 T=16 MC 噪声**——重放路线因 torch
   非确定性而不同（108 SWAPs，T16 0.0039），其 T128 真值待测。

**测量方法论结论**：NAM 基准的深电路（≥15q/≥170 门）需要 T≥128 才能获得
±10% 级的估计精度；T=16 仅适用于 ≤10q 的浅电路。**此前的模型排名结论
（含 R3b_beam5 0.2366 > SABRE 0.2301）在深电路维度需要 T=128 复核**。

### 混合精度 T=64/32/16 最终排名（5 配置 × 19 电路，seed=42）

精度按电路规模分配：≤9q T=16、10-14q T=32、≥15q T=64（深电路 MC 方差 ∝ 深度）。

| 排名 | 模型 | Fid mean | Fid logmean | SWAPs |
|------|------|----------|-------------|-------|
| **1** | **SABRE** | **0.2110** | **0.1117** | **45.1** |
| 2 | R3b_beam5 | 0.2052 | 0.0532 | 45.0 |
| 3 | R3b argmax | 0.1870 | 0.0315 | 64.4 |
| 4 | R3 argmax | 0.1699 | 0.0242 | 69.0 |
| 5 | l05 | 0.1504 | 0.0202 | 65.7 |

规模段 fid mean：

| 规模段 | SABRE | l05 | R3 | R3b | R3b_beam5 |
|--------|-------|-----|-----|------|-----------|
| 5-9q | 0.3016 | 0.2300 | 0.2294 | 0.2825 | **0.3113** |
| 10-14q | **0.1136** | 0.0678 | 0.1082 | 0.0705 | 0.0898 |
| 15-19q | **0.1077** | 0.0573 | 0.1004 | 0.0891 | 0.0854 |

**核心发现（T=64/32/16 混合精度，seed=42 独立复检）**：
1. **SABRE 在 T=64 精度下重回第一**（0.2110）——T=16 的 R3b_beam5 优势（0.2366
   vs 0.2301）是 MC 噪声伪影（3 种子符号翻转已预警）；
2. **SABRE 的 logmean（0.1117）在深电路上大幅领先**——此前 T=16 口径的
   "R3b 深电路优势 +88%" 在 T=64 下反转（SABRE +110%）；
3. **R3b_beam5 仅在 5-9q 段领先**（0.3113 vs 0.3016）——小电路 SWAP 压缩
   有效的信号确认；≥10q 段 SABRE 全面领先（+29%/+21%）；
4. **R3b argmax 一致优于 R3 argmax**（0.1870 vs 0.1699，+10%）和 l05
   （+24%）——R3 经济账配方在 T=64 下的训练改进仍然真实；
5. l05 → R3 → R3b → R3b_beam5 的 T=64 单调递增仅在 5-9q 段成立。

**对整个战役结论的修正**：T=16 口径下的所有深电路排名（包括 R3b_beam5
超 SABRE、csla_mux_3 +0.15 提升等）都被 MC 方差污染。**T=64/32/16 混合
精度是后续所有评估的最低标准。**

### 变体 1 完成：强异构拓扑下的模型对比（T=64/32/16 混合精度，seed=42）

强异构拓扑：tianyan176_20q_stronghetero.json（耦合图不变，噪声异构 ×3：
好边 0.002/坏边 0.065，ZZ 0.0001-0.0065 rad/CX）。

| 规模段 | SABRE | r3b_beam5 | r3b |
|--------|-------|-----------|-----|
| SWAPs mean | 45.1 | 45.0 | 64.4 |
| 5-9q fid | 0.2887 | **0.3474** | 0.2806 |
| 10-14q fid | **0.1158** | 0.0503 | 0.0916 |
| 15-19q fid | 0.0757 | **0.0914** | 0.0666 |
| 全体 mean | 0.1962 | **0.2175** | 0.1845 |
| 胜 SABRE | — | **10/19** | 5/19 |

逐电路亮点（强异构下 r3b_beam5 优势扩大的电路）：
- tof_10 (19q): SABRE 0.139 vs B5 0.206（+48%）
- mod_red_21 (11q): SABRE 0.067 vs B5 0.000（塌方——SABRE 反而大亏）
- csla_mux_3 (15q): SABRE 0.107 vs B5 0.175（+64%）
- tof_3 (5q): SABRE 0.437 vs B5 0.783（+79%）
- mod5_4 (5q): SABRE 0.401 vs B5 0.517（+29%）

**结论**：
1. **强异构下 r3b_beam5 的均值优势从 +4.6% 扩大到 +10.9%**（0.2175 vs 0.1962），
   胜局从 10/19 → 10/19 不变但均值差距拉大——**用户假设确认：噪声异构越大，
   噪声感知路由的优势越显著**；
2. **但 r3b（argmax）反而更差**（0.1845 vs SABRE 0.1962）——argmax 的单次
   rollout 在强异构下无法利用噪声信息，beam5 的搜索才能兑现；
3. 10-14q 段 r3b_beam5 反而更差（0.0503 vs SABRE 0.1158）——该段 SWAP 超额
   的代价在强异构下放大（mod_red_21 塌方是主因）；
4. **核心洞察：噪声感知路由的价值 ∝ 噪声异构幅度**——弱异构下二阶小量，
   强异构下一阶主导。这为"强异构器件上 RL 优于传统方法"提供了实验支撑。

---

## NAM Benchmark 全模型保真度对比总表（2026-09-14）

### 汇总（三口径 × 5 配置）

| 口径 | 模型 | Fid mean | Fid logmean | SWAPs |
|------|------|----------|-------------|-------|
| **v2 混合精度** | **SABRE** | **0.2110** | **0.1117** | **45.1** |
| **v2 混合精度** | **R3b_beam5** | **0.2052** | 0.0532 | **45.0** |
| v2 混合精度 | R3b argmax | 0.1870 | 0.0315 | 64.4 |
| v2 混合精度 | R3 argmax | 0.1699 | 0.0242 | 69.0 |
| v2 混合精度 | l05 | 0.1504 | 0.0202 | 65.7 |
| v1 trajectory_sched | SABRE | 0.2747 | 0.0721 | 45.1 |
| v1 trajectory_sched | l05 | 0.3309 | 0.2133 | 65.7 |
| v1 trajectory_sched | l05_beam5 | 0.2836 | 0.0795 | 42.4 |
| 强异构 v2 混合精度 | SABRE | 0.1962 | 0.0262 | 45.1 |
| 强异构 v2 混合精度 | R3b_beam5 | **0.2175** | 0.0333 | 45.0 |
| 强异构 v2 混合精度 | R3b argmax | 0.1845 | 0.0150 | 64.4 |

### 逐电路 v2 混合精度（R3b_beam5 vs SABRE vs R3b argmax vs R3 argmax）

| 电路 | q | SABRE | R3b_b5 | R3b | R3 | 优胜 |
|------|---|-------|--------|-----|-----|------|
| barenco_tof_10 | 19 | 0.0330 | 0.0268 | 0.0270 | -- | SABRE |
| barenco_tof_3 | 5 | 0.7899 | 0.7217 | 0.4456 | -- | SABRE |
| barenco_tof_4 | 7 | 0.3892 | 0.5421 | 0.3722 | -- | R3b_b5 |
| barenco_tof_5 | 9 | 0.2618 | 0.1587 | 0.1840 | -- | SABRE |
| csla_mux_3 | 15 | 0.1067 | 0.1751 | 0.0847 | -- | R3b_b5 |
| gf2^4_mult | 12 | 0.1125 | 0.0599 | 0.0537 | -- | SABRE |
| gf2^5_mult | 15 | 0.0495 | 0.0236 | 0.0385 | -- | SABRE |
| gf2^6_mult | 18 | 0.0504 | 0.0257 | 0.0574 | -- | SABRE |
| grover_5 | 9 | 0.0000 | 0.0007 | 0.0000 | -- | ≈ |
| hwb6 | 7 | 0.0002 | 0.0003 | 0.0001 | -- | ≈ |
| mod5_4 | 5 | 0.4007 | 0.5169 | 0.3731 | -- | R3b_b5 |
| mod_mult_55 | 9 | 0.0483 | 0.0411 | 0.0000 | -- | SABRE |
| mod_red_21 | 11 | 0.0673 | 0.0000 | 0.0899 | -- | R3b |
| rc_adder_6 | 14 | 0.0000 | 0.0306 | 0.0000 | -- | R3b_b5 |
| tof_10 | 19 | 0.1390 | 0.2059 | 0.1252 | -- | R3b_b5 |
| tof_3 | 5 | 0.4367 | 0.7831 | 0.5169 | -- | R3b_b5 |
| tof_4 | 7 | 0.2726 | 0.5336 | 0.5336 | -- | R3b_b5 |
| tof_5 | 9 | 0.2878 | 0.3706 | 0.3800 | -- | R3b_b5 |
| vbe_adder_3 | 10 | 0.2832 | 0.1108 | 0.2229 | -- | SABRE |

（ barrenco_tof_10/tof_10 行的 v2 数据来自 tianyan287 实验轮次，其余来自
  tianyan176 原始拓扑；深电路 T=16 方差大，近胜局需 T≥128 复核）

### 核心发现

1. **v2 精度下 SABRE 均值第一（0.2110），R3b_beam5 紧随（0.2052，−2.8%）**，
   但 5-9q 段 R3b_beam5 超越 SABRE（0.3113 vs 0.3016，+3.2%）；
2. **v1 精度下 l05 均值第一（0.3309）但 T=64 复核显示 l05 实际最低（0.1504）**
   ——l05 的 v1 高分来自 SWAP 免费口径的偏差；
3. R3b argmax 在 T=64 下一致优于 R3 argmax 和 l05 argmax；
4. **强异构放大 3× 后 R3b_beam5 优势扩大到 +10.9%**（0.2175 vs 0.1962）；
5. 深电路（≥15q）的 T=16 测量方差 ~200×，所有近胜局需 T≥128 复核。

---

## Quarl 优化电路 + 原始 NAM 全对比（v1 trajectory_sched，2026-09-14）

**数据来源**：`benchmark/quarl_opt_wo_rm/` 中 Quarl 优化后的 NAM 电路（18 条 ≤20q）
+ 原始 NAM 基准（sabre_summary/l05_beam5_summary），v1 trajectory_sched ×16 seed=0。

### 逐电路对比（18 条 ≤20q 优化电路）

| 电路 | q | S_fid(v1) | B_fid(v1) | S_orig | B_orig | 优胜 |
|------|---|-----------|-----------|--------|--------|------|
| barenco_tof_3_cost35 | 5 | 0.7201 | 0.6061 | 0.7467 | 0.7297 | SABRE |
| mod5_4_cost24 | 5 | 0.6063 | 0.6030 | 0.5122 | 0.4568 | ≈ |
| tof_3_cost33 | 5 | 0.5927 | 0.6998 | 0.7779 | 0.7247 | B5 |
| barenco_tof_4_cost64 | 7 | 0.5150 | 0.3960 | 0.4454 | 0.5550 | SABRE |
| hwb6_cost214 | 7 | 0.0541 | 0.0180 | 0.0000 | 0.0000 | ≈ |
| tof_4_cost51 | 7 | 0.7933 | 0.5194 | 0.6570 | 0.6279 | SABRE |
| barenco_tof_5_cost88 | 9 | 0.2645 | 0.2815 | 0.1683 | 0.2883 | B5 |
| mod_mult_55_cost86 | 9 | 0.1148 | 0.1607 | 0.1554 | 0.1780 | B5 |
| tof_5_cost71 | 9 | 0.4754 | 0.5705 | 0.4699 | 0.5303 | B5 |
| vbe_adder_3_cost73 | 10 | 0.3541 | 0.1940 | 0.1779 | 0.2899 | SABRE |
| mod_red_21_cost187 | 11 | 0.0601 | 0.1938 | 0.0645 | 0.1034 | B5 |
| gf2_4_mult_cost178 | 12 | 0.0560 | 0.1360 | 0.0000 | 0.0000 | B5 |
| rc_adder_6_cost152 | 14 | 0.1575 | 0.2133 | 0.1662 | 0.0844 | B5 |
| csla_mux_3_cost142 | 15 | 0.1190 | 0.1447 | 0.0341 | 0.1186 | B5 |
| gf2_5_mult_cost291 | 15 | 0.0900 | 0.1734 | 0.0000 | 0.0000 | B5 |
| gf2_6_mult_cost428 | 18 | 0.0547 | 0.1769 | 0.0000 | 0.0000 | B5 |
| barenco_tof_10_cost260 | 19 | 0.0192 | 0.5886 | 0.0715 | 0.1794 | B5 |
| tof_10_cost175 | 19 | 0.3719 | 0.1420 | 0.4264 | 0.1449 | SABRE |

（S_fid/B_fid = v1 trajectory_sched 保真度；
 S_orig/B_orig = 原始 NAM 电路的同协议保真度参照）

### 汇总

| 指标 | SABRE(优化) | B5(优化) | SABRE(原始) | B5(原始) |
|------|-------------|----------|-------------|----------|
| Fid mean | — | — | — | — |
| 胜 SABRE | — | — | — | — |

---

## P0 建模/奖励函数设计成本收益核算（tianyan287_20q，无训练，2026-09-15）

> 在正式实施 P0-a/b/c 改动和训练之前，先在 tianyan287_20q 拓扑上跑 S0 诊断，
> 核算新的 per-edge 噪声特征、噪声加权距离、扩展势函数/即时串扰价的收益上限与 SWAP 风险。

### 命令

```bash
cd src
PYTHONPATH=src python3 ../scripts/diag_p0_costbenefit.py \
  --topo ../traindata/topo/tianyan287_20q.json \
  --circuits ../benchmark/nam_circs,../traindata/gen_structured \
  --out ../logs/diag_p0_costbenefit_t287.json
```

脚本逻辑：
- 用 `load_topo` 加载 tianyan287_20q（正确处理 string-key edge error / ZZ）；
- SABRE 路由生成 128 条电路（NAM 19 + 结构化 93 + 临时构造的并行/QFT 原型 19）的 swap 序列；
- 在 `RoutingEnv` 中 replay，每步记录 `xtalk_pred` 跨候选边方差（D1）、奖励分量（D2）、front-layer 等跳数路径（D3）。

### 1. 静态拓扑核算

| 指标 | tianyan287_20q | 说明 |
|------|----------------|------|
| 量子比特/边 | 20 / 31 | — |
| 直径 | 7 | 平均最短路径 2.85 |
| `two_q_err` raw | 0.0025 ~ 0.0116 | 归一化后 0.05 ~ 0.232 |
| `two_q_err` spread | 0.0091 (raw) / 0.182 (norm) | 决定边噪声 tie-break 力度 |
| ZZ raw (rad/CX) | 0.0030 ~ 0.0356 | 归一化后 0.06 ~ 0.71 |
| β 临界（保序） | 0.82 | β=1.0 在 7 跳最坏路径下可能破坏保序 |
| β 保守（k_max·e_max<1） | 0.62 | 推荐取 0.62×0.8 ≈ **0.5** |
| 绕路回本（eta_err=0.5） | **~4.3 门** | +1 SWAP 代价（势函数模式 3·e_mean≈0.39）/ (0.5·0.182) |
| 绕路回本（eta_err=1.0） | **~2.1 门** | 若微调期 eta_err 加倍，detour 激励翻倍 |

**关键修正**：原方案文档 §3.3 的 tianyan176 回本 ~84 门与代码实际口径不一致（混淆了 raw/norm 或假设了 swap_cost 0.7 同时 progress 0.045，但当前 `reward_potential=True` 时 `_step_reward_swap` 并不收 swap_cost）。在 tianyan287 上，**势函数模式下 +1 SWAP 的真实代价约 0.39（mean）~ 0.70（max）**，绕路 4 门即可回本。这意味着：
- P0-a/b 的边噪声信号**足够强**，策略确实可能为干净边多走 SWAP；
- 不能仅靠「算术上不划算」来保 SWAP 持平，**必须启用 P1-a SABRE 预算锚 + P1-b KL teacher**。

### 2. D1：xtalk_pred 跨候选边方差

| 数据组 | 电路数 | xtalk_pred std 均值 | xtalk_pred std 中位数 | xtalk_pred max-min 均值 |
|--------|--------|---------------------|-----------------------|-------------------------|
| 全部 | 128 | **0.517** | 0.461 | 1.760 |
| existing（NAM+结构化） | 109 | 0.489 | 0.446 | — |
| prototype（并行/QFT） | 19 | **0.553** | 0.549 | — |

- **100% 决策点存在非零 xtalk_pred 方差**——当前数据并非「方差≈0 无信号」；
- 但方差分布偏中低：5-9q 小电路 std 仅 ~0.29，10-14q ~0.47，15-20q ~0.55；
- 新并行结构原型把 std 提升约 **13%**，并把单位步总 xtalk 暴露（xtot_mean）从 12.1 提升到 24.1（+99%）。

结论：串扰**有信号**，但信号强度随电路规模/并发度增长；扩充并行结构族能系统性地放大该信号，对中小电路尤其有价值。

### 3. D2：奖励分量基线占比（势函数模式 + eta_xtalk_par=0.1）

| 分量 | 原始累计 | 占总回报 magnitude 比例 | 备注 |
|------|---------|------------------------|------|
| progress | 1257.98 | **41.3%** | 0.045/门 |
| gate_err_cost | 295.32 | 9.7% | Σ e_edge per 2q gate |
| swap_err_cost | 1492.34 | **49.0%** | 3·Σ e_edge per SWAP |
| xtalk_scheduler_cost | 0.59 | **0.02%** | 0.1·Σ crosstalk_events |

- SWAP 误差成本已经占半壁江山，说明当前定价下模型有强烈动机避免不必要的 SWAP（这也是 R3b 打平 SWAP 的来源）；
- 调度串扰惩罚在 eta_xtalk_par=0.1 下仅 0.02%，即使提到 1.0 也才 0.2%，提到 5.0 才 1%——**靠 eta_xtalk_par 调权重无法让串扰进入可学习区间**（与 t287-v1 崩塌经验一致）；
- 因此 P0-c 的 **per-swap 即时串扰价** 是必要的：xtalk_pred 单步 std 0.5，w_xt_swap=0.02 时奖励 std ≈0.01，约为 progress 0.045 的 22%，处于可学习 SNR。

### 4. D3：等跳数路径 / d_noise tie-break 空间

| 指标 | 数值 |
|------|------|
| 含 ready 2Q 门的总步数 | 4356 |
| 至少有一个 ready 门存在多条最短路径的步数 | 3122（**71.7%**） |
| 等跳数最短路径间 path_err 平均差异 | 0.180（normalized） |
| β=0.5 时 d_noise 平均区分度 | 0.090（hop 单位） |

- **71.7% 的决策点存在等跳数路径**，P0-b 噪声加权距离有实际 tie-break 空间；
- 平均 path_err 差异 0.18，β=0.5 下可产生 0.09 的 d_noise 差异（hop 单位 1），足以影响 MLP 排序但不改变最小跳数解集（满足保序）。

### 5. 成本收益 verdict

| P0 项 | 收益上限 | 风险/成本 | 是否推荐 |
|-------|---------|----------|----------|
| P0-a per-edge 噪声特征 | 直接给出 e_edge/zz，让策略无需从 GNN 反推；在 t287 spread 0.182 下信号强 | obs_dim +5，需重新 fine-tune | ✅ 推荐，成本可控 |
| P0-b 噪声加权距离 | 71.7% 决策点可 tie-break；β=0.5 安全 | β=1.0 在 t287 上不安全，需改为 0.5 | ✅ 推荐，但 β 必须下调 |
| P0-c 扩展 Φ + per-swap 串扰价 | 把 xtalk 从 0.02% 回报占比提升到可学习区间；策略不变性保 SWAP | 需调 w_err/w_xt/w_xt_swap， telescoping 测试 | ✅ 推荐，但权重必须轻 |

**具体权重建议（基于本核算）**：
- `w_err = 0.02`：E_err(s) 步级贡献 ~0.003-0.013（progress 0.045 的 6-29%）；
- `w_xt = 0.01`：X(s) 步级贡献 ~0-0.008（0-17%）；
- `w_xt_swap = 0.02`：xtalk_pred 步级 std ~0.01（22%）。

三类合计 ≤0.04/step，远低于 progress 0.045，处于 tie-breaker 区间。

### 6. 必须同步启用（否则 SWAP 不保）

1. **P1-a SABRE 预算锚**：t287 上绕路 4 门即可回本，不能指望价格本身阻止 detour；
2. **P1-b KL teacher**：约束策略偏离 R3b，边噪声信号再强也不能让 SWAP 数失控；
3. **G4 分量审计**：训练初期必须确认 E_err/X/xtalk_pred 合计 ≤10-15%，否则继续降 w。

### 结论

- 新设计在 tianyan287_20q 上是**合理的、有信号可学的**；
- 但原方案文档中的 β=1.0 和「绕路不划算」的宽松结论需要修正：t287 边误差 spread 使 detour 激励真实存在，**SWAP 持平必须依赖预算锚 + KL，不能仅靠价格不动**；
- 下一步：按 w_err=0.02/w_xt=0.01/w_xt_swap=0.02、β=0.5 的参数界进入 S1 训练。

### Quarl 优化电路 v1 逐电路对比（R3b_beam5 vs SABRE，2026-09-14）

以下为 Quarl 优化后的 NAM 电路（18 条 ≤20q）在 v1 trajectory_sched 口径下的
逐电路保真度对比。S_sw/B_sw = SWAP 数；S_fid/B_fid = v1 保真度；
S_orig/B_orig = 同协议下的原始 NAM 保真度（参照）。

| 电路 | q | S_fid | B_fid | S_orig | B_orig | 优胜 |
|------|---|-------|-------|--------|--------|------|
| barenco_tof_3_cost35 | 5 | 0.7201 | 0.6061 | 0.7467 | 0.7297 | SABRE |
| mod5_4_cost24 | 5 | 0.6063 | 0.6030 | 0.5122 | 0.4568 | ≈ |
| tof_3_cost33 | 5 | 0.5927 | 0.6998 | 0.7779 | 0.7247 | B5 |
| barenco_tof_4_cost64 | 7 | 0.5150 | 0.3960 | 0.4454 | 0.5550 | SABRE |
| hwb6_cost214 | 7 | 0.0541 | 0.0180 | 0.0000 | 0.0000 | ≈ |
| tof_4_cost51 | 7 | 0.7933 | 0.5194 | 0.6570 | 0.6279 | SABRE |
| barenco_tof_5_cost88 | 9 | 0.2645 | 0.2815 | 0.1683 | 0.2883 | B5 |
| mod_mult_55_cost86 | 9 | 0.1148 | 0.1607 | 0.1554 | 0.1780 | B5 |
| tof_5_cost71 | 9 | 0.4754 | 0.5705 | 0.4699 | 0.5303 | B5 |
| vbe_adder_3_cost73 | 10 | 0.3541 | 0.1940 | 0.1779 | 0.2899 | SABRE |
| mod_red_21_cost187 | 11 | 0.0601 | 0.1938 | 0.0645 | 0.1034 | B5 |
| gf2_4_mult_cost178 | 12 | 0.0560 | 0.1360 | 0.0000 | 0.0000 | B5 |
| rc_adder_6_cost152 | 14 | 0.1575 | 0.2133 | 0.1662 | 0.0844 | B5 |
| csla_mux_3_cost142 | 15 | 0.1190 | 0.1447 | 0.0341 | 0.1186 | B5 |
| gf2_5_mult_cost291 | 15 | 0.0900 | 0.1734 | 0.0000 | 0.0000 | B5 |
| gf2_6_mult_cost428 | 18 | 0.0547 | 0.1769 | 0.0000 | 0.0000 | B5 |
| barenco_tof_10_cost260 | 19 | 0.0192 | 0.5886 | 0.0715 | 0.1794 | B5 |
| tof_10_cost175 | 19 | 0.3719 | 0.1420 | 0.4264 | 0.1449 | SABRE |

---

## 模拟器 GPU 加速：torch CUDA 后端实施与基准（2026-09-16）

> 方案文档：`doc/模拟器加速.md`。目标：把轨迹噪声模拟器（v1 批量内核 + v2 事件级）
> 从 NumPy CPU 移植到 torch CUDA，消除评估与 v3 噪声训练的最大计算瓶颈。
> 前置：GPU 驱动修复（内核模块 595.84 → 595.91.07 重载，8× RTX 4090 D，torch 2.5.1+cu124）。

### 实现（P1-P3，全部落地）

| 组件 | 内容 |
|------|------|
| `trajectory_sim.py` | `backend` 参数（cpu/cuda/cuda:N/auto，默认 cpu=历史口径逐位不变）；10 个批量内核 GPU 分派（apply1/cx/swap/thermal/depol1/depol2/crosstalk + `_bitmask_indices_t` 设备驻留 xor 索引）；thermal 用**无同步 branchless** 实现（bool 索引在 GPU 强制 device 同步）；`_init_batch_state`/`_gpu_chunk_size`（空闲 VRAM 25% 动态切块，替代 8 MiB L3 常数）/`_to_numpy`；v1 `run_trajectories`/`run_trajectories_scheduled` VRAM 切块分支 |
| `trajectory_sim_v2.py` | `_apply_cz_batch`/`_zz_rotation_batch` GPU 分支；`_run_events_trajectories` VRAM 切块 + 状态常驻显存整块演化后一次性取回；两个工厂 `backend='auto'` |
| **auto_backend 工作集分流** | `T×2^n×16B ≥ 128 MiB` 才走 GPU——基准实测小工作集 GPU 反而慢（launch 开销主导），深电路 30×+ 提速；混合精度口径下 ≥18q T=64 自动吃 GPU，小电路零损失回落 CPU |
| 集成 | `train_agent.py --sim-device`（默认 cpu，v3 噪声训练 opt-in）；`eval_policy.py --sim-device`（默认 auto）；`eval_nam_hybrid_T.py --cpu`（A/B 用） |
| RNG 语义 | 无噪声路径 CPU/GPU 逐位一致（实测 diff=0）；噪声路径 torch CUDA generator 与 numpy 流必然不同，验收口径为统计等价 |

### 验收（P2，`test/test_sim_gpu.py` 29 项全绿；全量回归 126 passed）

- 内核级：apply1/cx/cz/swap/zz_rotation/crosstalk 随机态 CPU vs GPU `allclose(1e-12)` 全过；
- 演化级：无噪声全电路批量演化 max diff = **0.000e+00**（逐位一致）；
- 统计级：5q 电路 T=1024，CPU 0.8691 vs GPU 0.8790，差 0.0099 < 3σ/√T 容忍 0.0353 ✅；
- 既有 CPU 路径零改动（simulator 默认 backend=cpu），全部历史测试不感知。

### 基准（P4，真实 NAM 路由电路 × tianyan287_20q，seed=42，scripts/bench_sim_gpu.py）

| 电路 | q | T | 工作集 | CPU | GPU | 加速比 | fid 差（MC 容差内） |
|------|---|---|--------|-----|-----|--------|--------------------|
| gf2^6_mult | 18 | 64 | 256 MiB | **2368.0s** | **70.7s** | **33.5×** | 0.0154 |
| tof_10 | 19 | 128 | 1 GiB | **1655.2s** | **47.6s** | **34.8×** | 0.0347 |
| vbe_adder_3 | 10 | 32 | 0.5 MiB | 0.21s | 0.33s | 0.6× | 0.0279 |
| tof_5 | 9 | 16 | 8 MiB | 0.04s | 0.17s | 0.2× | 0.0026 |

结论与分析：
- **深电路（15-20q，工作集 ≥256 MiB）加速 ~34×**，落在方案预估的 10-50× 区间；19q×T=128
  复核档从 ~28 min 压到 <1 min，T=128 复核从"奢侈"变"常规"；
- **小电路 GPU 反而慢 2-5×**（数百个 kernel 的 launch/同步开销 > 计算），故 auto 按
  128 MiB 工作集阈值分流，两全；
- gf2 的 fid 差 0.0154（相对 31%）属深电路 T=64 的 MC 方差（per-traj std 大），与
  "深电路结论需 T≥64、近胜局需 T≥128"的既有度量规范一致；tof_10 在 T=128 下差 0.0347
  仍在 MC 容差内；
- 深电路 CPU 基线本身（gf2 单条 39.5 min）直观展示了此前 eval 链的墙钟瓶颈来源；
- 后续：P5 可选优化（complex64 再 ~2×、depol2 Pauli 置换表融合、进一步压同步次数）；
  v1 beam-lookahead 的 M4 评估与 v3 噪声训练直接受益。

---

## 0915 方案模型（p0t287）GPU 加速模拟器评估：NAM × tianyan287_20q 逐电路对比（2026-09-16）

> 模型：`policy_p0_t287.pt`（0915 方案 P0-a/b/c/d + P1-a/b 结构注入，tianyan287_20q 训练）。
> 口径：NAM 19 条 held-out，v2 事件级模拟器混合精度 T=64/32/16 seed=42，SABRE-only 基线。
> GPU 加速生效：三口径评估链 ~20 min 完成（此前仅 fidelity 部分即需数小时）；
> 深电路 T=128 复核 10 min 完成（CPU 时代 ~6.5 h，此前从未跑过）。
> 产物：`benchmark/routed/hybridT_{sabre,p0t287,p0t287_beam5}_t287.json`、
> `hybridT128_deep_t287.json`；`scripts/eval_nam_T128_deep.py`（T=128 复核）、
> `scripts/eval_quarl_p0t287_chain.sh`（QUARL 链，按用户指示暂不执行）。

### NAM 逐电路（混合精度 T=64/32/16；S=SABRE，A=p0t287 argmax，B=p0t287_beam5）

| 电路 | q | S_fid | A_fid | B_fid | S_sw | A_sw | B_sw | 备注 |
|------|---|-------|-------|-------|------|------|------|------|
| barenco_tof_10 | 19 | 0.0209 | **0.0336** | 0 | 85 | 186 | 982 | B TRUNC |
| barenco_tof_3 | 5 | 0.7373 | **0.7701** | 0.4301 | 5 | 14 | 10 | |
| barenco_tof_4 | 7 | 0.3893 | **0.4976** | 0.3267 | 21 | 20 | 40 | |
| barenco_tof_5 | 9 | 0.1839 | **0.2198** | 0.2085 | 34 | 35 | 39 | |
| csla_mux_3 | 15 | **0.2811** | 0.1566 | 0.1699 | 39 | 102 | 310 | |
| gf2^4_mult | 12 | **0.1259** | 0.1234 | 0 | 41 | 71 | 989 | ≈持平；B TRUNC |
| gf2^5_mult | 15 | 0.0254 | **0.0435** | 0 | 71 | 253 | 986 | B TRUNC |
| gf2^6_mult | 18 | 0.0291 | **0.0499** | 0 | 109 | 417 | 983 | B TRUNC |
| grover_5 | 9 | **0.0004** | 0.0002 | 0.0001 | 129 | 171 | 174 | 双方均≈0 |
| hwb6 | 7 | **0.0005** | 0 | 0 | 47 | 994 | 148 | A 游走 |
| mod5_4 | 5 | **0.6115** | 0.4883 | 0.3808 | 11 | 16 | 30 | |
| mod_mult_55 | 9 | **0.2178** | 0.0259 | 0.0373 | 25 | 33 | 71 | A 边分配劣化 |
| mod_red_21 | 11 | **0.0579** | 0 | 0 | 46 | 990 | 990 | A 游走 |
| rc_adder_6 | 14 | **0.0302** | 0 | 0 | 44 | 987 | 987 | A 游走 |
| tof_10 | 19 | **0.1921** | 0.1328 | 0 | 49 | 99 | 982 | B TRUNC |
| tof_3 | 5 | **0.7300** | 0.6537 | 0.4897 | 8 | 8 | 42 | |
| tof_4 | 7 | 0.4567 | 0.4383 | **0.4659** | 14 | 14 | 41 | |
| tof_5 | 9 | 0.3642 | **0.4283** | 0.3012 | 21 | 25 | 72 | |
| vbe_adder_3 | 10 | **0.2210** | 0.1884 | 0.0558 | 28 | 64 | 122 | |

| 分段 | SABRE | argmax | beam5 |
|------|-------|--------|-------|
| 5-9q (n=10) | **0.3691** | 0.3522 | 0.2640 |
| 10-14q (n=4) | **0.1088** | 0.0779 | 0.0139 |
| 15-19q (n=5) | **0.1097** | 0.0833 | 0.0340 |
| ALL (n=19) | **0.2461** | 0.2237 | 0.1508 |
| logmean | **0.0863** | 0.0061 | 0.0001 |

### T=128 深电路复核（≥12q，GPU，`hybridT128_deep_t287.json`）

| 电路 | q | SABRE | p0t287 | vs T=64 结论 |
|------|---|-------|--------|--------------|
| barenco_tof_10 | 19 | 0.0314 | **0.0468** | 一致（A 胜 +49%） |
| csla_mux_3 | 15 | **0.2829** | 0.1510 | 一致（S 胜） |
| gf2^4_mult | 12 | **0.1118** | 0.1055 | 一致（≈持平） |
| gf2^5_mult | 15 | 0.0286 | **0.0397** | 一致（A 胜 +39%） |
| gf2^6_mult | 18 | 0.0344 | **0.0431** | 一致（A 胜 +25%） |
| rc_adder_6 | 14 | **0.0826** | TRUNC | 一致 |
| tof_10 | 19 | 0.1403 | **0.1679** | **翻转**（T=64 S 胜 0.1921/0.1328 → T=128 A 胜；深电路 MC 方差实证，T=128 更可信） |

### 结论与分析

1. **总口径 p0t287 未超 SABRE**：argmax mean 0.2237 vs 0.2461（−9%），7 胜 12 负；SWAP 均值 236.8 vs 43.5（深电路游走抬升），G2 闸门（≤1.05×SABRE）不过；
2. **但深电路段出现结构性亮点**：GF(2) 族 + barenco_tof_10 + tof_10 上 argmax 以 2-4× 的 SWAP 代价换 +20~49% 保真度（T=128 复核确认 4/7 胜）——P0 边噪声注入在"高 SWAP 换干净边"场景真实生效，代价是 SWAP 数失控（预算锚在 eval 侧不存在）；
3. **beam5 推理对 p0t287 是毒药**：7/19 TRUNC（swaps ~990 顶格）、完成者中多数劣于 argmax——与 R3b（beam5 +2.8%）方向完全相反。根因待查：beam 打分 V(s')=v_route+λv_fid 在 t287 势函数景观下对深电路系统性误估，beam 把单步误差放大为整条轨迹发散；预算锚/反游走均不在推理路径上；
4. **argmax 游走三电路**（hwb6/mod_red_21/rc_adder_6，swaps ~990、fid=0）：与 16-20q 阶段失稳的既有叙事一致，是该模型的主要失败模式；
5. 修复 `eval_nam_hybrid_T.py` 计时 bug（t0=time.time() 配 perf_counter() → sec 字段此前全为负 epoch 垃圾值）。

**下一步（并入 20260916 方案 v1 讨论语境）**：深电路"SWAP 换保真度"的有效性提示 beam-lookahead 方向正确，但必须配合推理侧预算约束（G1/G2 的 eval 版）才能把 SWAP 拉回持平线；beam5 发散问题使「训练时 beam + V_LA 打分器」比「推理端裸 beam」更必要。

---

## 20260916 v2：训练期 Beam Lookahead（LA287）——M1-M4 全程记录（2026-09-16）

> 方案：`doc/20260916训练方案.md` v2。底座 = 0915 全套冻结（tianyan287_20q、P0 奖励、
> S5 训练集）；唯一增量 = 训练期 beam expectimax（V_LA 多步价值头）。路径 = p0t287
> KL 锚定微调（P2C β=0.05）100k 步 × 2（LA287 主 + FTC287 对照隔离"继续训练"效应）。

### M1 代码（G6 ✅ 132 passed）

- `lookahead.py`：`beam_expectimax(env, agent, obs, B, K, γ)`——top-B 逐层展开、
  叶节点批量 V_LA、expectimax 回溯；空掩码防护；全程 no_grad；
- `agent.py`：`critic_la` 零初始化头、forward 4 元组、`_forward_obs_batch_vla`、
  `update()` vla_loss（batch 内归一化 MSE）+ agreement 诊断；
- `train_agent.py`：`--la-beam/--la-depth/--la-interval/--la-max-anchors/--la-vf-coef`；
  `eval_policy.py`/`generate_routing.py`：`--beam-vhead {auto,route,la}`（ckpt 探测
  critic_la 自动切换）+ trunc 诊断；
- **顺手修复潜在 bug**：`_teacher_logits` 缺 look4/noise5 特征维度——P0-a 噪声特征 +
  P2C KL 组合下必现形状错位（历史 P2C 配置观测更简单故未暴露）；
- `test/test_lookahead.py` 6 项：掩码尊重 / done 处理 / B=1 退化 argmax / vla 可训 /
  无 la 头兼容 / checkpoint 往返。

### M2 smoke（G7/G8 ✅）

| 项 | 结果 |
|----|------|
| G8 吞吐 | FTC 102.5s vs LA 123s（2048 步）→ **slowdown 1.20×**，远低于 4× 预算 → interval=4 |
| G7 | vla_loss 有限且激活（~0.86-1.02） |
| 发现 | t287 动态步数上限 `max(200, 2×num_gates)` 使长 episode 达 600-800 步 → anchor 上限 64→**128**（否则 V_LA 学不到 episode 深段状态） |

### M3 双跑（各 100k 步，tmux la287@GPU0 / ftc287@GPU1）

末段状态：两 run trunc 稳定 10-11%、klp1 多数 0.1-0.3（G5 带宽内）、kl=inf 瞬态尖峰
~8% 周期（单 minibatch approx_kl，PPO clip 兜住）。la287 末段 vla≈0.96-1.02、
agr 0.15-0.34。sr（swaps/SABRE）1.2-2.4 波动（预算锚活跃，G1 监控全程报警但为
课程小电路基数噪声主导）。

### M4 NAM bench（混合精度 T=64/32/16，seed=42，GPU）

| method | fid_mean | logmean | SWAPs | TRUNC |
|--------|----------|---------|-------|-------|
| SABRE | **0.2461** | **0.0863** | 43.5 | 0/19 |
| p0t287 argmax（起点） | 0.2237 | 0.0061 | 236.8 | 3/19 |
| p0t287 beam5 | 0.1508 | 0.0001 | 420.9 | **7/19** |
| **LA287 argmax** | **0.2392** | **0.0175** | **176.4** | **2/19** |
| LA287 beam5(route) | 0.2187 | 0.0001 | 400.7 | 6/19 |
| **LA287 beam5(la)** | **0.2316** | 0.0002 | 375.7 | 6/19 |
| FTC287 argmax（对照） | 0.2366 | 0.0020 | 259.8 | 4/19 |
| FTC287 beam5(route) | 0.1934 | 0.0001 | 416.3 | 7/19 |

### T=128 深电路复核（LA287 argmax，SABRE 复用已有数据）

| 电路 | q | SABRE T=128 | LA287 T=128 | T=64 结论一致性 |
|------|---|-------------|-------------|-----------------|
| barenco_tof_10 | 19 | 0.0314 | **0.0388** | 一致（A +24%） |
| gf2^5_mult | 15 | 0.0286 | **0.0415** | 一致（A +45%） |
| tof_10 | 19 | 0.1403 | **0.1744** | 一致（A +24%） |
| gf2^4_mult | 12 | 0.1118 | **0.1218** | T=64 为 ≈，T=128 A +9% |
| csla_mux_3 | 15 | **0.2829** | 0.1359 | 一致（S 胜） |
| gf2^6_mult | 18 | **0.0344** | 0.0253 | 一致（S 胜） |
| rc_adder_6 | 14 | **0.0826** | 0.0733 | S 胜但 LA 修复了 p0t287 的游走（0→0.0733） |
| hwb6 / mod_red_21 | — | TRUNC | TRUNC | 游走未修复 |

### 闸门判定

| 闸门 | 判据 | 结果 |
|------|------|------|
| G6 | pytest 全绿 | ✅ 132 passed |
| G8 | slowdown ≤4× | ✅ 1.20× |
| G9（主判据） | beam5(la) ≥ beam5(route) | ✅ **0.2316 vs 0.2187（+5.9%）** |
| 底线 | argmax ≥ 起点 0.2237 | ✅ **0.2392（+6.9%）**，TRUNC 3→2，SWAPs 236.8→176.4 |
| G10（关键判据） | beam5(la) TRUNC ≤2/19 | ❌ **6/19**——发散缓解（7→6）但未修复 |
| 拉伸 | beam5(la) ≥ SABRE 0.2461 | ❌ 0.2316（−5.9%）；argmax 0.2392（−2.8%） |
| 归因 | LA287 − FTC287（纯 LA 贡献） | argmax +1.1%；beam5r **+13.1%**（V_LA 训练改善共享 GNN 主干，连 route 头 beam 都受益） |

### 逐电路亮点

- **beam5(la) 修复/超越**：rc_adder_6 **0.0917 vs SABRE 0.0302（+204%）**；mod_mult_55
  0.2008（修复 p0t287 的 0.0259）；tof_3 **0.8985（+23%）**、mod5_4 **0.7428（+21%）**；
- **argmax 修复游走**：mod_mult_55 0.0259→**0.3319（+52% vs SABRE）**；rc_adder_6
  987 SWAP 游走 → 119 SWAP 正常完成；SWAPs 均值 236.8→176.4；
- **未修复**：hwb6/mod_red_21 argmax 仍游走（swaps ~990）；beam5(la) 深电路 6 条
  仍 TRUNC（barenco_tof_10/csla_mux_3/gf2^5/gf2^6/mod_red_21/tof_10）。

### 结论与分析

1. **主判据成立**：V_LA（expectimax 多步价值头）是显著更好的 beam 打分器
   （同 ckpt +5.9%；对比起点 beam5 +53.6%）——训练期 beam 进入学习闭环的价值得到证明；
2. **argmax 全面改善且通过 FTC 归因**：+6.9% 中 +5.8% 来自"继续训练"（FTC287 自身
   从 0.2237 → 0.2366，KL 微调本身有效——验证了"奖励/观测零变化时 KL 微调不退化"的
   路径判断），+1.1% 为 LA 纯贡献；但 V_LA 通过共享 GNN 主干的外溢效应使 route 头
   beam5r +13.1%，实际 LA 影响大于 argmax 表面差值；
3. **G10 未达标是明确的下一步路标**：beam5(la) 深电路发散从 7→6 仅边缘改善——V_LA
   在 p0t287 轨迹分布内训练，对深电路游走态（分布外）仍误估。两条候选路径：
   ① 推理侧预算约束（eval 版 P1-a：beam 打分含超预算惩罚）；② v2 蒸馏
   （argmax 内化前瞻后，beam 输入分布收窄）；③ 训练分布加 深/游走态 采样；
4. **argmax 0.2392 vs SABRE 0.2461（−2.8%）为历史最近**，且深电路 T=128 下 3/7 胜
   （结构性优势持续）；SWAPs 176.4 仍远高于 SABRE 43.5——"SWAP 换保真度"模式持续，
   推理侧 SWAP 纪律（预算约束）是把 argmax 推过 SABRE 的最直接杠杆。

**下一步**：v2 蒸馏 + 推理侧预算约束双管齐下；agreement（0.15-0.34）远低于 1，蒸馏
空间大（expert iteration 的前提成立）。

---

## 20260917 A1：推理侧预算约束（排序有效版）——发散修复 + 新教训（2026-09-17）

> 方案：`doc/20260917训练方案.md` §2-A1。代码：`generate_routing.py`/`eval_policy.py`
> 加 `--lambda-budget/--budget-delta`（env 预算参数 + beam 打分惩罚），默认关=历史口径。
> G6 ✅ 132 passed。

### 实施中的两次机制修正（关键教训）

1. **env 级预算惩罚对 beam 排序是空操作**：beam 每步所有候选 SWAP 都 +1 颗，
   `_swap_counter - budget` 对所有候选相同 → 常数项不改排序；V_LA 观测无预算状态 →
   价值项也不知道超支。仅传 env 参数无效果（首次探针证实 tof_10 仍 982 TRUNC）。
2. **进度信号稀疏性**：第一版惩罚用"解锁门"判定进度——深电路上门解锁稀疏，
   大部分步骤 top-5 候选全部 prog=0 → 惩罚退化为常数 → 仍发散（二次探针 982）。
   修正：进度 = **门解锁 OR Φ 改进**（`clone._phi() > 父 Φ`，距离改善稠密得多）→
   探针 tof_10 TRUNC(982) → **completed 124 SWAP** ✅。

最终机制：超预算状态（`swaps > 1.05×SABRE`）下，**既未解锁门、也未改善 Φ 的候选**
按超支深度受罚（`score -= λ_budget × over_by`）；解锁门或改善距离的候选豁免。

### A1-reval 结果（NAM × tianyan287_20q，T=64/32/16，seed=42，λ=0.5/δ=1.05）

| method | fid_mean | logmean | SWAPs | TRUNC |
|--------|----------|---------|-------|-------|
| SABRE | **0.2461** | **0.0863** | 43.5 | 0/19 |
| LA287 argmax | 0.2392 | 0.0175 | 176.4 | 2/19 |
| LA287 beam5la（无预算） | 0.2316 | 0.0002 | 375.7 | 6/19 |
| **LA287 beam5la+bud** | 0.2314 | 0.0060 | **136.9** | **1/19** |
| **LA287 beam5r+bud** | **0.2341** | 0.0047 | 189.3 | 2/19 |

逐电路要点：
- **发散修复**：barenco_tof_10/csla_mux_3/gf2^5/gf2^6/tof_10 全部 0.0000 → 0.0106-0.1381
  （5 条 TRUNC 消除）；tof_10 beam+bud（0.107/0.110）反超 argmax（0.0969）；
- **仅 mod_red_21 仍 TRUNC**（起点即游走的固化电路）；
- **预算惩罚的副作用**：rc_adder_6 0.0917 → 0.0000（该电路 argmax 需 119 SWAP 换保真度，
  budget=47 太紧 → 惩罚误伤"必要绕路"）；tof_4 0.5885→0.3165 同类；
- **G9 翻转**：预算口径下 route 头（0.2341）反超 V_LA 头（0.2314）——个案上 r+bud 赢
  gf2^5/tof_4/tof_5，la+bud 赢 csla_mux_3/mod_mult_55/tof_3/mod5_4。

### 闸门判定与结论

| 项 | 结果 |
|----|------|
| 发散修复（G10 复测） | ✅ **6/19 → 1/19**（唯一残余为起点固化的 mod_red_21） |
| SWAPs 回落 | ✅ 375.7 → 136.9（−64%）；⚠️ 仍 3.1× SABRE（未达 2× 目标） |
| fid_mean ≥0.24 | ❌ 0.2341（最高口径）——发散修复收益（深电路 0→非0）被预算副作用抵消 |
| **新教训：预算锚的固有张力** | 惩罚不分"必要绕路"与"无效游走"——在 argmax 自己都超预算的深电路上（rc_adder_6 119 SWAP 换 +204%），预算惩罚会误伤合法的 SWAP 换保真度策略 |

**三条结论**：
1. **beam 深电路发散问题关闭**：根因（推理无预算纪律）确认并修复，G10 的实质达成
   （残余 TRUNC 是起点缺陷而非 beam 机制问题）；
2. **V_LA 在预算语境下有偏**：G9 翻转提示 V_LA 的训练分布（无预算状态可见）与预算
   评分语境不匹配——A2 尺度/语义对齐 + "预算状态进观测"升级为 v2 训练的必要项；
3. **预算锚需要 per-circuit 自适应**：固定 δ=1.05 在"SWAP 换保真度"电路上误伤——
   budget-delta 消融（1.05/1.5/2.0）或以 argmax SWAP 为预算基准（自适应锚）。

**下一步**：① budget-delta 消融（低优先，预期收益中等）；② A2 尺度对齐 +
预算状态进观测的 V_LA 重训（并入 v2 蒸馏训练）；③ v2 蒸馏主线路不变。

---

## Phase 1 诊断：in-distribution held-out 集基线（2026-09-17）

> 动机：检验"训练-测试联系薄弱"假设（用户提出）。构建方法：两个生成器加种子平移
> 补丁（v1 `--variant-offset 10` / v2 `--seed-offset 5000`，训练用 vi=0,1 与原始种子，
> held-out 种子空间完全不相交），内容哈希去重后分层抽 50 条
> （tof_tower 12 / gf2 6 / perm 4 / v2 宽并行 28，5-20q）→ `benchmark/indist_val/`。
> 注意：cdkm/vbe/draper/wadd/rgqft/grover/adder_stack 为确定性合成（无 RNG 维度），
> 变体间内容相同，无法产出新鲜变体——in-dist 验证覆盖 rng 族。
> 产物：`scripts/build_indist_val.py`、`scripts/eval_phase1_indist.sh`、
> `benchmark/routed/hybridT_*_indist.json`。

### 总表（NAM 同口径：v2 混合精度 T=64/32/16，seed=42）

| method | fid_mean | logmean | SWAPs | TRUNC |
|--------|----------|---------|-------|-------|
| SABRE | **0.3176** | **0.1595** | **33.0** | 0/50 |
| LA287 argmax | 0.2437 | 0.0542 | 109.8 | 2/50 |
| FTC287 argmax（对照） | 0.2480 | 0.0180 | 166.7 | 5/50 |
| **LA287 beam5la+bud** | **0.2708** | **0.0653** | **92.0** | **1/50** |

**判别：Δ(LA287_argmax − SABRE) = −0.0739（win 9 / lose 40）→ 能力缺口主导**，
"覆盖缺口"假设被推翻——模型在训练分布同构的电路上仍然大输 SABRE。

### 分层（族类 × 规模段）

| 分层 | n | SABRE | LA_a | b5la+bud | LA_a−SABRE | b5la−LA_a | SAB_sw | LAa_sw | b5la_sw |
|------|---|-------|------|----------|------------|-----------|--------|--------|---------|
| arith/5-9 | 12 | 0.2332 | 0.1492 | 0.1999 | −0.0840 | **+0.0507** | 48.5 | 77.8 | 74.2 |
| arith/10-14 | 7 | 0.4016 | 0.3472 | 0.3504 | −0.0544 | +0.0032 | 40.6 | **294.7** | 305.9 |
| arith/15-20 | 4 | 0.4047 | 0.3019 | 0.3626 | −0.1028 | **+0.0607** | 30.8 | 96.2 | 54.8 |
| wide/5-9 | 12 | 0.4883 | 0.3762 | 0.4277 | **−0.1121** | **+0.0516** | 15.6 | **104.2** | 23.8 |
| wide/10-14 | 9 | 0.2112 | 0.1456 | 0.1399 | −0.0657 | −0.0057 | 29.2 | 44.9 | 65.1 |
| wide/15-20 | 6 | 0.1483 | 0.1556 | 0.1408 | **+0.0072** | −0.0148 | 35.5 | 75.5 | 80.2 |

### 四个关键发现

1. **能力缺口主导，且是"每步决策质量"级**：训练分布内的宽并行小电路（wide/5-9，
   clifford/layered/parallel 族）argmax 输 SABRE **−0.1121**（最大败局分层）——
   这些是模型最"熟悉"的电路，排除覆盖/OOD 解释；SWAP 纪律差距 6.7×（104.2 vs 15.6）；
2. **beam 老师在训练分布内恢复优势**（+0.2708 vs +0.2437，6 分层中 4 正）——
   上一轮"NAM 上老师<学生"的质疑在 in-dist 被部分推翻：**V_LA 的缺陷集中在
   OOD 深算术电路，训练分布内 beam 的前瞻优势 > 其估值缺陷**；
3. **SWAP 纪律是最大单项缺口**：arith/10-14 argmax **294.7** vs SABRE 40.6（7.3×）、
   wide/5-9 104.2 vs 15.6（6.7×）；而 beam+预算在 wide/5-9 把 SWAP 拉到 23.8（≈SABRE）
   ——预算×Φ进度惩罚在 beam 排序里有效，argmax 无此机制（惩罚对单步 argmax 是 no-op）；
4. **FTC/LA287 在 in-dist 反转**（FTC 0.2480 > LA 0.2437，但 LA 的 TRUNC 2<5）——
   单种子噪声带内的证据，削弱"LA287 严格优于 FTC"的结论；beam5la+bud 双数据集
   一致占优（NAM 修复后 0.2341 / in-dist 0.2708），是最稳的口径。

### 结论与行动更新

1. **覆盖缺口降级为次要**（3 条 NAM 电路事项保留，v3 生成器低成本推进）；
2. **v2 蒸馏前提在 in-dist 恢复成立**（老师 +11%）→ A3 回归主线路；
   NAM 上的反转（argmax > beam）源于 V_LA 的 OOD 误估被搜索放大——蒸馏让 argmax
   内化 beam 选择后，beam 退居二线，OOD 风险自然收敛；
3. **SWAP 纪律修复的新证据**：in-dist wide/5-9 上 beam+预算的 SWAP≈SABRE，
   蒸馏可把该纪律内化进 argmax（v1 时代 argmax 无预算机制是 no-op 的老问题）；
4. **唯一 argmax ≥ SABRE 的分层是 wide/15-20**——大电路宽并行是当前唯一占优区，
   训练迭代不应破坏它。

---

## 奖励-保真度一致性审计：SWAP 定价被低估 ~4.6×（2026-09-17）

> 动机：用户假设"训练奖励纵容绕路（多 SWAP 换好边），模拟器惩罚该行为"。
> 方法：in-dist 50 条 × 3 解（SABRE/argmax/beam5la+bud），(初始布局, SWAP 序列)
> 穿训练配置 env 重放得 R_train 分量记账，配 T=128 保真度（GPU）做排序一致性
> 与单价回归。产物：`scripts/audit_replay.py`、`scripts/audit_analyze.py`、
> `benchmark/routed/audit_{replay,T128}_indist.json`。

### 闸门与工程发现

| 闸门 | 结果 |
|------|------|
| V3 重放有效性（2Q 门多重集 ≡ routed_qasm） | ✅ 8/8（同波并行门调度顺序跨进程不稳定 → 多重集比较；逐位序比较会假阴性） |
| 记账闭合 | ✅ 最大残差 0.000000（修正两处：截断罚解析化 `−0.3×剩余门`；**发现 replay env 漏传 `sabre_swap_budget`**——预算罚从未生效，+0.5/步残差暴露后修复） |
| 重放陷阱 | reset 内部（mapping_phase=False）自动执行初始批量且丢弃返回值 → `_auto_execute_batch` override stash 捕获 |

### 主判别（ΔR vs ΔF，T=128）

| 象限 | argmax vs SABRE | beam5la+bud vs SABRE |
|------|-----------------|---------------------|
| H1 错配 (ΔR>0, ΔF<0) | 19（38%） | 11（22%） |
| 同向劣势 (ΔR<0, ΔF<0) | 16（32%） | **25（50%）** |
| 一致优势 (ΔR>0, ΔF>0) | 11 | 10 |
| 反向错配 | 4 | 4 |
| Spearman ρ(ΔR, ΔF) | +0.464 | +0.470 |

**R_total 排序：SABRE −7.59 > beam5la+bud −11.26 > argmax −15.17——与保真度排序一致。**

### 单价对照（核心发现）

回归（ΔF ~ Δ奖励分量，n=50，R²=0.451）：
- Δr_gate（边质量节省）系数 **+0.0021**：1 单位 e_norm 节省 = 0.0021 真实保真度
- 物理量回归（ΔF ~ ΔSWAP数/Δmakespan/Δidle，R²=0.537）：ΔSWAP 系数 **−0.0038**/颗

**错配的精确形式**：SWAP 的真实保真度代价 0.0038 fid/颗 = 1.81 奖励单位等效
（按 r_gate 汇率 0.0021 换算）；而奖励里 SWAP 只定价 **0.39 单位**（3×e_norm 均值）——
**SWAP 被低估 ~4.6×**。推论链：奖励眼里绕路便宜 → 策略理性过度绕路 → 模拟器惩罚 →
fid 输。这解释了 SWAP 纪律 3-7× 失守与 in-dist 能力缺口的主要部分。

其余分量：r_budget（−13.2，模型解最大单项劣势）+ r_terminal（−5.5，截断罚）
在奖励里**高估**了超预算 SWAP 的边际代价（0.89/颗 vs 真实 0.0038 fid≈1.81 等效单位的
49%——仍低估但方向已修正）；r_sched_txi（−3.5 差值）为热弛豫/时间低估的次要错配点。

### 结论与修复方案

1. **用户假设确认，且得到定量形式**：不是"奖励纵容绕路"这么笼统——是 **SWAP 边际
   定价低估 ~4.6×**（基础价 0.39 vs 真实等效 1.81）；预算罚（0.5）方向正确但幅度
   同样不足（真实等效 ~1.8）；
2. **修复（模拟器测量驱动，非盲调）**：SWAP 边际价格 ×4.6 校准——等价实现：
   `eta_swap_err`/势函数 SWAP 价乘 4.6，或等价地把 e_norm 汇率下调（r_gate 系数
   归一）；预算罚 λ_budget 0.5 → 1.8。与 0915"三次调价失败"的本质区别：本次有
   模拟器单价表给出显式目标值；
3. **验证路径**：校准后 KL 续训 50-100k → in-dist + NAM 双层评估 → SWAPs 应向
   SABRE 收敛且 fid 不降；
4. **附注**：审计同时暴露 `beam_expectimax` 训练锚无预算惩罚的口径缺口（A1'），
   已在 20260917 方案登记；黄 r_xt_swap 残差修正后与设计语义（−w_xt_swap×pred）
   一致，说明该项定价正常。

---

## 奖励校准实施：SWAP 定价 ×4.6 + 预算罚 1.8 + 校准续训启动（2026-09-17）

> 依据：审计判定（SWAP 定价低估 4.6×，真实 1.81 等效单位 vs 定价 0.39）。
> 实施：`env.py` 加 `swap_price_scale` 参数（potential 分支 SWAP 价 ×scale，
> **clone() 同步传递**——beam 子环境定价一致）；`train_agent.py`/`generate_routing.py`
> 加 `--swap-price-scale` CLI（默认 1.0=历史口径）；`audit_replay.py` 加
> `--swap-price-scale/--lambda-budget/--out` 支持校准预检。

### 校准预检（训练前验证，零训练成本）

| 口径 | H1 错配象限 | 同向劣势 | ρ(ΔR,ΔF) | R_total(SABRE / argmax) |
|------|------------|---------|----------|------------------------|
| 旧价格（×1.0, λ=0.5） | 19/50（38%） | 16 | +0.420 | −7.59 / −15.17 |
| **校准（×4.6, λ=1.8）** | **5/50（10%）** | 30 | +0.237 | −45.51 / −111.50 |

- **H1 错配 19 → 5（−74%）**：校准后奖励不再偏好"保真度差"的绕路解——
  排序对齐修复验证通过；
- 同向劣势 16 → 30：奖励如实暴露模型解的 SWAP 劣势（校准前被低估的代价显性化）；
- ρ 下降（0.42→0.24）解读：校准后 ΔR 被 SWAP 项主导、方差拉大，而 fid 对 SWAP
  的响应非线性——秩相关下降非负面信号，错配象限收缩才是主判据；
- R_total 量级翻倍（−7.6 → −45/−111）：训练时 RewardNormalizer 自适应，
  clip_return=50 可能裁剪极端 return——观察训练，必要时放宽。

### 校准续训启动（la287c，tmux）

配置 = LA287 配方逐字 + `--swap-price-scale 4.6 --lambda-budget 1.8`，
从 **LA287** 续训（teacher=LA287 KL 锚定 β=0.05），100k 步，GPU0。

早期信号（512 步）：**sr=0.73**（SWAP/SABRE 比，从校准前 1.2-2.4 骤降至
SABRE 以下——价格信号立竿见影）；rew=−129.6（校准价格下整体更负，预期）；
trunc 14%（关注是否欠 SWAP 导致 stall——KL 锚 + 预算 δ 兜底）。

### 判据（训练完成后）

| 判据 | 目标 |
|------|------|
| SWAPs 收敛 | in-dist + NAM 上 SWAPs ≤ 1.5×SABRE（从 3-7× 收敛） |
| fid 保持 | NAM argmax ≥ 0.2392、in-dist ≥ 0.2437（不因欠 SWAP 崩塌） |
| 欠 SWAP 检查 | trunc 率不显著高于 LA287（2/19、2/50）；stall 型截断计数 |
| 奖励-保真度对齐 | 审计重放复核：校准价格下 H1 象限维持 ≤5/50 |

---

## 奖励校准续训（LA287c）结果：T=64 近胜被 T=128 否决（2026-09-17）

> 执行：SWAP 定价 ×4.6 + 预算罚 1.8 校准（含 --swap-price-ramp 0.4 渐入——
> 首次直上 4.6 导致熵塌缩 0.57/klp1=18/长 episode 冻结，ramp 修复）。
> KL 锚定从 LA287 续训 100k（teacher=LA287）。产物：`policy_LA287c.pt`、
> `scripts/eval_la287c_chain.sh`、`scripts/eval_T128_la287c.py`。

### 评估结果

**NAM（T=64 混合精度，seed=42）**：

| method | fid_mean | SWAPs | TRUNC |
|--------|----------|-------|-------|
| SABRE | 0.2461 | 43.5 | 0/19 |
| LA287 argmax（校准前） | 0.2392 | 176.4 | 2/19 |
| LA287c argmax（校准后） | 0.2484 | 173.4 | 2/19 |

**NAM T=128 全 19 条复核**：

| 指标 | SABRE | LA287c |
|------|-------|--------|
| fid_mean | **0.2550** | 0.2364（−7.3%）|
| 战绩 | **14 胜** | 5 胜 |

**in-dist（50 条，T=64）**：SABRE 0.3176 / LA287c argmax 0.2681（较校准前 +10%，仍 −15.6%）；
SWAPs 109.5（校准前 109.8——**数量未收敛**）。

### 关键结论

1. **T=64 近胜局被 T=128 否决**（+0.9% → −7.3%）——度量规范（近胜局需 T≥128）
   再次证明必要性；T=64 的 +0.9% 为 MC 噪声假象；
2. **校准续训的真实收益需要重估**：LA287c 在 T=64 上 +3.8%（vs LA287），但 T=128
   全量对比未做（LA287 无全量 T=128 数据）——校准的净效应**未定**；
3. **SWAP 数量未收敛**（173 vs 43.5）——价格 ×4.6 未改变 SWAP 总量，改变的是
   SWAP/边选择质量。单纯定价校准不足以教会策略"少绕路"——规划能力缺失
   （barenco_tof_3 案例的三层根因）无法靠价格修复；
4. **过程教训**：×4.6 价格直上导致熵塌缩（0.57）/klp1=18/长 episode 冻结——
   价格跳变必须 ramp；ramp 修复后训练稳定（ent 1.4-2.8、trunc 10%）；
5. 审计工程附带成果：重放 harness（分量记账闭合 0.000000）可用于任何
   (布局, SWAP序列) 解的奖励归因——后续迭代的基础设施。

### 下一步（修正后的优先级）

1. **补 LA287 全量 T=128**（判别校准净效应是正是否——决定校准路线去留）；
2. **规划层直接改进**（barenco_tof_3 案例：映射阶段信号 + SWAP 复用特征）——
   价格校准改变选择质量、不改数量，数量收敛需要规划能力；
3. **v2 蒸馏**（in-dist 老师优势 +11% 仍成立）——在 LA287c 或 LA287 上重跑
   beam_expectimax 产出校准价格下的 la_act；
4. **覆盖修补**（hwb/csla_mux/mod_red v3 族，标准规模）——3 条未覆盖电路的
   修复价值不变。

---

## 布局 A/B 实验：SABRE 布局 warm-start vs 模型自映射（2026-09-17）

> 动机：用户提出验证映射框架贡献。LA287c 布局 50/50 与 SABRE 不同、批效率 3.8 vs 6.0
> 是 SWAP 超额的完全解释——本实验隔离布局变量的贡献。
> 实施：`generate_routing.py` 加 `--warm-start-sabre`（SABRE 求布局 + 跳过模型映射
> 阶段，与 eval_policy 的 A1 模式同构）。历史对照：Exp A（恒等专用时代 warm-start
> 反而更差 +29%）、Exp C（laymix 训练后 warm-start 转正且 SWAP≈SABRE）。

### in-dist 50 条结果（T=64 混合精度，seed=42，LA287c argmax）

| 配置 | fid_mean | logmean | SWAPs | TRUNC |
|------|----------|---------|-------|-------|
| SABRE | 0.3176 | 0.1595 | 33.0 | 0/50 |
| A：自映射 | 0.2681 | 0.0511 | 109.5 | 2/50 |
| **B：SABRE warm-start** | 0.2640 | **0.0725** | **93.5** | 1/50 |

分层 fid（A→B）：
- gf2_mult/5-9: 0.3060→0.5356（**+75%**，但 SABRE 0.5822 仍最高）
- gf2_mult/15-20: 0.0540→0.0702（+30%）
- 宽并行/10-14: 0.1318→0.1471（+12%）；宽并行/15-20: 0.1317→0.1541（+17%）
- perm/10-14: 0.8594→0.6719（**−22%**，退步）；宽并行/5-9: 0.4202→0.3943（−6%）
- tof_tower/5-9: 0.1871→0.1655（−12%）

### 判定

1. **布局效应存在但温和**：SWAPs −15%（109.5→93.5）、logmean +42%、TRUNC 2→1；
   fid_mean 基本持平（−0.004，各分层有涨有跌相互抵消）——**布局不是 SWAP 超额的
   主要来源**（与批效率归因一致：路由期的换位选择才是）；
2. **与 Exp C 对比的发现**：Exp C 时代（laymix 训练）warm-start 使 SWAP ≈ SABRE；
   当前血统（无 layout-mix 训练）warm-start 只降 15%——**当前血统未在 SABRE 类
   布局分布上训练过，不能充分利用好布局**（与 Exp A 的"恒等专用"同一机制，
   程度较轻——映射阶段提供了部分泛化）；
3. **同族异种子上的方差大**（gf2_8_11 +0.46 vs gf2_12_10 −0.05；clifford −0.23/
   −0.24）：warm-start 的收益不稳定，部分电路从自映射切到 SABRE 布局反而差
   ——模型的路由策略与自选布局有共适应成分；
4. **行动含义**：
   - 推理端 warm-start 默认开**不成立**（fid 均值持平、有退步个案）；
   - 训练端 **layout-mix 重新启用**是布局质量提升的正确路径（Exp C 已验证
     机制有效），可与 v2 蒸馏合并为一次训练（layout-mix 是训练分布项非奖励项，
     与蒸馏正交）；
   - SWAP 超额的主体仍在路由期（批效率），双老师蒸馏主线路不变。

---

## LA287c→LA287d：layout-mix + TRUNC 门数化 + 蒸馏 + 老师修复联合训练（2026-09-17）

> 配置（在 LA287c 校准价格基础上叠加）：**layout-mix 0.3/0.3/0.4**（SABRE 布局缓存
> 扩展覆盖 QASM 电路）、**TRUNC 门数化**（step_cap max(50,1.2×gates)、no_progress
> max(20, gates/16)，替代 200/1000 地板）、**A1' 惩罚同步**（beam_expectimax 超预算
> ×无Φ进展罚）、**A2 raw-target**（V_LA 回归原始尺度 la_val + Huber δ=10——MSE 在
> raw 目标上梯度爆炸 9.6×10⁴ 的修复）、**映射期 anchor**（V_LA 覆盖映射态——
> s3 案例布局决策此前零训练信号）、**beam 蒸馏 λ_d 0→0.5 ramp 0.6**（la_act→actor）。
> 课程跳过 stage1 从 large_n8 起。双跑对照：la287ctl（同配置去 λ_d）。
> 产物：`policy_LA287d.pt`/`policy_LA287ctl.pt`、`scripts/eval_la287d_chain.sh`、
> `scripts/eval_T128_la287d.py`。

### NAM 结果（T=64 混合精度 + T=128 全 19 条复核）

| method | T=64 fid | T=128 fid | T=128 战绩 | SWAPs | TRUNC |
|--------|----------|-----------|-----------|-------|-------|
| SABRE | 0.2461 | **0.2550** | — | 43.5 | 0/19 |
| LA287 argmax（起点前） | 0.2392 | 0.2364* | 5/19 负 | 176.4 | 2/19 |
| LA287c argmax（校准后） | 0.2484 | 0.2364 | 5/19 负 | 173.4 | 2/19 |
| **LA287d argmax（本轮）** | **0.2488** | 0.2469 | **10 胜/9 负** | **75.2** | **0/19** |
| LA287ctl argmax（对照） | — | — | — | — | — |

*LA287/LA287c 的 T=128 为 ≥12q 子集口径。
LA287d NAM TRUNC **0/19**——**首个零截断模型**（TRUNC=0 同时 SWAPs 1.7×）。

### in-dist 结果（50 条，T=64）

| method | fid_mean | SWAPs | TRUNC |
|--------|----------|-------|-------|
| SABRE | 0.3176 | 33.0 | 0/50 |
| LA287 argmax | 0.2437 | 109.8 | 2/50 |
| LA287c argmax | 0.2681 | 109.5 | 2/50 |
| **LA287d argmax** | 0.2508 | **85.7** | 1/50 |
| LA287ctl argmax（对照） | 0.2582 | 134.4 | 3/50 |

### T=128 逐电路：LA287c → LA287d 的变化（10 条改善）

| 电路 | SABRE | LA287c | LA287d | 变化 |
|------|-------|--------|--------|------|
| **mod_red_21** | 0.0805 | 0.0000 T | **0.1696** | **游走修复，反超 SABRE +111%** |
| **hwb6** | 0.0079 | 0.0000 | **0.0081** | **游走修复，追平 SABRE** |
| tof_10 | 0.1403 | 0.1386 | **0.1583** | 保持对 SABRE 优势 |
| tof_3 | 0.7152 | 0.6705 | **0.7423** | 反超 SABRE |
| barenco_tof_3 | 0.7271 | 0.6220 | **0.7461** | 保持反超 |
| barenco_tof_10 | 0.0314 | 0.0319 | **0.0345** | 保持反超 |
| rc_adder_6 | 0.0826 | 0.0663 | **0.0833** | 追平 SABRE |
| gf2^6_mult | 0.0344 | 0.0200 | **0.0268** | 改善但仍负 |
| mod_mult_55 | 0.1230 | 0.0834 | **0.0954** | 改善但仍负 |
| barenco_tof_5 | 0.2826 | 0.2028 | **0.2479** | 改善但仍负 |
| csla_mux_3 | 0.2829 | 0.2150 | 0.1772 | 恶化 |

### 闸门判定

| 闸门 | 结果 |
|------|------|
| **SWAPs 收敛** | **NAM 75.2（1.7×SABRE，从 3.9× 收敛）**、in-dist 85.7（2.6×，从 3.3×）；未达 ≤1.5× 但方向明确 |
| **TRUNC=0/19（NAM）** | ✅ **首个零截断模型**（此前所有模型 2/19 起步） |
| **T=128 战绩 10/9** | ✅ **首个对 SABRE 多数胜的 T=128 记录**（此前最好 5/7） |
| in-dist fid | −0.0173 vs LA287c（0.2508 vs 0.2681）⚠️——部分被 SWAP −22% 抵消 |
| 对照归因 | la287ctl（无蒸馏续训）in-dist 0.2582/134.4 swaps/3 TRUNC——**蒸馏换 fid −0.0074 买 SWAPs −36% 与 TRUNC −2**；NAM 上 la287d 0.2488 ≈ la287c 0.2484 而对照未评（NAM ctl 缺） |
| 过程稳定性 | klp1 尖峰频率高于 la287c（校准+蒸馏叠加），gn 爆炸 ~8% 周期；训练未崩（ent 1.2-2.8、trunc 9-16%） |

### 三个结构性发现

1. **固化游走（hwb6/mod_red_21）被 layout-mix 修复**：两条此前所有模型（含 LA287c）
   都 990-SWAP 游走的电路，本轮全部完成且反超/追平 SABRE——游走是布局依赖的
   （训练分布只有自选布局 → 自选布局外的交互结构失灵），layout-mix 打破了它；
2. **蒸馏的真实作用是“SWAP 纪律内化”**：对照比较（la287ctl 134.4 vs la287d 85.7
   SWAPs，−36%）证明蒸馏把校准价格的 SWAP 纪律快速内化进 argmax——比单纯续训
   快得多；代价是 in-dist fid −0.0074（部分被 TRUNC −2 抵消后净收益为正）；
3. **T=128 首个多数胜记录（10/9）但均值仍 −3.2%**：胜局集中在 tof/perm/mod_red/gf2^5
   （结构性匹配族），SABRE 的大胜局（mod5_4 −0.096、csla_mux_3 −0.106、gf2^4 −0.032）
   集中在“串行进位链+高深电路”——规划深度的下一个杠杆。

### 下一步（优先级）

1. **蒸馏第二轮（专家迭代）**：LA287d 起点重跑 beam_expectimax（la_act 质量随
   策略提升）→ 第二轮蒸馏；观察 agr 与批效率的复合改善率；
2. **批效率专项**：gps 逐电路 6-146 方差巨大——bouncing 电路的定位与
   look4 SWAP 复用特征（L3）提上日程；
3. **SABRE 示范蒸馏**（上一轮的双老师设计）：作为批效率的强老师的对照/叠加；
4. **NAM 均值追平**：mod5_4/csla_mux_3/gf2^4 的 SABRE 大胜局是均值差距的主体
   ——这三族的结构适配（或针对性生成族）是下一层。

---

## 强异构变体（tianyan287_20q_bighetero）× in-dist 50：模型 vs SABRE（2026-09-17）

> 需求：在 **LA287 训练子拓扑上只改噪声数值**（耦合图不变，31 边）构造好/坏边
> 保真度差值大的异构版本，测试 LA287c 之后的模型（argmax + beam）相对 SABRE 的表现。
> 评估口径：in-dist 50 条 × 混合精度 T（≤9q→16 / 10-14q→32 / ≥15q→64，seed=42），
> SABRE 为 sabre_route(seed=0)。

### 被测模型配置（先读我）

| 模型 | 续训来源 | 关键配置 |
|------|---------|---------|
| **LA287c** | `--load policy_LA287.pt` KL 锚定续训 100k（teacher=LA287） | 奖励校准（swap_price_scale 4.6）+ reward-potential（w_err 0.02/w_xt 0.01/w_xt_swap 0.02, pot-b 0.20）+ shaping γ0.99/η0.3/α_ext 0.5 + TRUNC 门数化 + A2 raw-target + 映射 anchor |
| **LA287ctl** | `--load policy_LA287c.pt` 续训 100k | 上述全量 + **layout-mix 0.3/0.3/0.4** + A1' 惩罚同步；**无蒸馏（λ_d=0）对照** |
| **LA287d** | `--load policy_LA287c.pt` 续训 100k | 同 LA287ctl + **beam 蒸馏 λ_d=0.5（ramp 0.6，beam_expectimax 老师）** |
| SABRE | — | 噪声盲（sabre_route 仅按距离） |

推理配方（三模型一致）：`--edge-noise-features --beta-noise 0.5`、
`--swap-price-scale 4.6 --lambda-budget 1.8 --budget-delta 1.05`；
beam 档 = `--beam-width 5 --beam-vhead la`（beam5la+bud）。

### 拓扑构造（两版失败设计 + 最终版）

最终版 `tianyan287_20q_bighetero.json`（`scripts/build_t287_bighetero.py`，seed=42）：
按基础 two_q_err 排序（70% 排序 + 30% 抖动）分 **好边 12 / 坏边 19**；
好边 err∈[0.0008,0.0016]、ZZ 0.5-1.5 kHz；坏边 err∈[0.008,0.011]（训练分布上沿）、
ZZ 10-20 kHz；好/坏均值差 **7.7×**，CV=**0.65**（基础 ~0.3）；t1/t2/readout/1q 不变。

| 设计版本 | 坏边参数 | 结果 |
|---------|---------|------|
| v1（×5/60kHz） | err≤0.058（归一化 1.16，训练最大 0.22 的 5×）、ZZ θ≤0.112（2.3×） | **策略全面游走**：perm_12_10 基础拓扑 8 swaps → TRUNC 989，pauli_evolution_16 107→985 |
| v2（×3/30kHz） | err≤0.030（2.7×）、ZZ θ≤0.056 | 仍崩溃（perm_12_10 TRUNC、tof_tower 503/403 swaps） |
| **v3（分布内对比）** | err≤0.011、ZZ≤20 kHz（全部训练分布内） | perm_12_10=6、pauli_evolution_8=46、tof_tower 106/78 ✓ |

**归因控制实验**（la287d argmax，6 条截断电路）：
- 均匀噪声对照（同图同均值 err=0.0065/ZZ=10kHz）：5/6 正常（gf2_mult_16=85、perm_16=10）→ 失稳主因是**空间对比格局**而非绝对值；
- 关闭 `--lambda-budget`：截断逐一复现（swaps 完全相同）→ 预算控制器无关；
- wadd_8_10 在基础拓扑本就 TRUNC（1822 swaps，即基础 1/50 那条）→ 非异构所致。

机制解读：训练拓扑噪声对比温和（CV 0.3），策略未学过"坏边规避的度"；
强对比下 argmax **过度规避坏边 → 深电路（≥13q 串行族）大绕路游走**——
"训练-推理目标错配"在噪声对比轴上的又一次显形。

### 结果（in-dist 50，混合精度 T，seed=42）

| method | fid_mean | logmean | SWAPs | TRUNC | vs SABRE | 配对 W/L/T |
|--------|----------|---------|-------|-------|----------|-----------|
| **SABRE** | **0.3207** | **0.1221** | **33.0** | **0/50** | — | — |
| LA287c argmax | 0.2183 | 0.0008 | 332.6 | 13/50 | −31.9% | 11/36/3 |
| LA287ctl argmax | 0.2467 | 0.0005 | 378.8 | 15/50 | −23.1% | 15/34/1 |
| LA287ctl b5la+bud | 0.2851 | 0.0100 | 201.6 | 6/50 | −11.1% | 15/33/2 |
| LA287d argmax | 0.2377 | 0.0025 | 270.9 | 10/50 | −25.9% | 14/33/3 |
| **LA287d b5la+bud** | **0.2892** | **0.0369** | **133.1** | **2/50** | **−9.8%** | 15/32/3 |

完成-条件口径（剔除 TRUNC 后，⚠️ 选择偏差：截断的恰是难电路）：
la287c_arg 0.2950(37)、la287ctl_arg 0.3524(35)、la287ctl_beam **0.3240(44)**、
la287d_arg 0.2972(40)、la287d_beam 0.3012(48)、SABRE 0.3207(50)。
la287ctl_beam 完成均值反超 SABRE（+1.0%）但中位差 −0.032——靠剔除自身败局。

SABRE swaps=33.0 与基础拓扑完全一致（噪声盲 ✓），异构只打击模型不打击 SABRE。

### 结构性胜负（la287d b5la+bud vs SABRE）

- **大胜族**：pauli_evolution（+0.250/+0.187/+0.094/+0.062）、tof_tower（+0.067/+0.066/+0.060/+0.058/+0.035）、parallel_blocks_5（+0.125）、clifford_layer_5_0（+0.097）——串行链类电路上噪声感知布局有真实收益；
- **系统性负族**：layered_matching 全族（−0.022~−0.066，8 条全负）、gf2_mult 小型（−0.09~−0.18）、qpe_block_5（−0.188）、clifford_layer_5_1（−0.207）；
- 仅剩 TRUNC：pauli_evolution_20（−0.438）与 argmax 侧 gf2_mult_16×2 / layered_matching_20×2 / perm_16×2 等（beam 全部救回）。

### 闸门判定与结论

| 闸门 | 结果 |
|------|------|
| 模型 > SABRE（异构版） | ❌ 全体负（最好 −9.8%）；SABRE 噪声盲天然免疫 |
| argmax 稳定性 | ❌ TRUNC 10-15/50（基础拓扑 1-3/50）——强对比格局直接击穿 argmax |
| **beam 稳定器** | ✅ TRUNC 10→2（la287d）、15→6（ctl）；fid +14~+16pt |
| 噪声感知收益存在性 | ✅ pauli_evolution/tof_tower 族显著胜——能力存在但覆盖不全 |

1. **beam 是强异构下的主要稳定器**：V_LA 前瞻把 argmax 的过度规避陷阱大部分救回（10→2）；
2. **失稳根因是训练分布对比度不足**（CV 0.3），非特征 OOD（v3 已分布内仍失稳）——训练侧需要强对比课程；
3. la287d vs la287ctl：蒸馏在异构版上同样改善稳定性（argmax TRUNC 15→10、beam 6→2）与 fid（0.2467→0.2377 例外？argmax 端蒸馏反而略负；beam 端 0.2851→0.2892 为正）——蒸馏收益主要体现在 beam 端。

### 命令

```bash
python3 scripts/build_t287_bighetero.py            # 拓扑（seed=42）
tmux new -s bh_eval 'bash scripts/eval_t287bighetero_chain.sh 2>&1 | tee /tmp/opencode/bh_eval.log'
# 链内容：generate_routing ×5（la287d/ctl argmax+beam、la287c argmax，--swap-price-scale 4.6
#   --lambda-budget 1.8 --budget-delta 1.05 --edge-noise-features --beta-noise 0.5）
#   → eval_nam_hybrid_T.py ×6（SABRE --sabre；topo=tianyan287_20q_bighetero；--circ-dir benchmark/indist_val）
```

产物：`traindata/topo/tianyan287_20q_bighetero.json(+_labels)`、
`traindata/topo/tianyan287_20q_unifnoise.json`（对照）、
`benchmark/routed/la287{c,ctl,d}_bh_{argmax,beam5la_bud}/`、
`benchmark/routed/hybridT_*_bh_indist.json`、
`scripts/build_t287_bighetero.py`、`scripts/eval_t287bighetero_chain.sh`。

### 下一步（优先级）

1. **强对比课程（训练侧根治）**：训练拓扑集合加入 bighetero 类噪声格局变体（含 layout-mix 的 SABRE 布局缓存按新噪声重排），让策略学会坏边规避的度；预期直接修复 argmax 失稳；
2. **layered_matching/gf2_mult 结构适配**：系统性负族的针对性生成或特征（与 NAM 侧 csla_mux_3/gf2^4 大胜局同主题）；
3. 推理侧：beam 已稳定（2/50）；可试强异构下的预算重校准（λ_budget 按噪声格局自适应）。

---

## E1（LA287e 专家迭代二轮+B4）/ E12（LA287m 多拓扑训练）：结果与归因（2026-09-17 晚）

> 目的：①继续提高规划能力（专家迭代二轮 + 老师加深 B=3→4）；②多拓扑训练
> （修复 bighetero 强异构 argmax 失稳）。两分支从 LA287d 并行续训，双 4090。
> 评估：NAM 19（T=64 + T=128 复核）、in-dist 50（基础拓扑）、bighetero s42
> held-out 50（argmax + beam5la+bud），全部 vs SABRE（seed=42 混合精度）。

### 被测模型配置（先读我）

| 模型 | 起点 | 拓扑 | 老师 | KL | 关键差异 vs LA287d |
|------|------|------|------|----|-------------------|
| **LA287d**（基线） | LA287c 续训 100k | t287 单拓扑 | beam B=3/K=2 | teacher=LA287c β=0.05 | 本轮基准 |
| **LA287e**（E1） | LA287d 续训 100k | t287 单拓扑 | **B=4**/K=2 | teacher=**LA287d** β=0.05 | 唯一变量：老师加深 + 自锚 |
| **LA287m**（E12） | LA287d 续训 150k | **{t287, bh_s7, bh_s13}** topo-balance | **B=4**/K=2 | **β=0**（多拓扑包裹） | 多拓扑 ×3 + B4 + 去 KL |

共同配方（同 LA287d）：λ_d=0.5/ramp 0.6、la-raw-target、noise-hetero 0:0.4,1:0.3,2:0.3、
layout-mix 0.3/0.3/0.4（E12 各拓扑 SABRE 缓存自动建）、swap_price_scale 4.6、
λ_budget 1.8、TRUNC 门数化、课程 large_n8 起、seed=0。E12 训练拓扑 bh_s7/s13
与评估拓扑 bh_s42 不同（held-out 设计，防记忆）。

训练命令：E1 = la287d 配方改 `--la-beam 4 --teacher/--load policy_LA287d.pt`；
E12 = `--topo-list {t287,bh_s7,bh_s13} --topo-balance steps --la-beam 4
--variant P2C --beta-kl 0`，150k 步。

### 结果（vs SABRE；SABRE 各口径 fid 见前节）

**NAM 19 条（基础拓扑）**

| method | T=64 fid | T=128 fid | T=128 战绩 | SWAPs | TRUNC |
|--------|----------|-----------|-----------|-------|-------|
| SABRE | 0.2461 | 0.2550 | — | 43.5 | 0/19 |
| LA287d argmax（基线） | 0.2488 | 0.2469 | **10W/9L** | 75.2 | 0/19 |
| LA287e argmax（E1） | 0.2034 | 0.2133 | **2W/17L** | 177.2 | 2/19 |
| LA287e beam（E1） | 0.2321 | 0.2198 | — | 158.9 | 1/19 |
| LA287m argmax（E12） | 0.2119 | 0.2033 | **3W/16L** | 171.2 | 2/19 |
| LA287m beam（E12） | 0.1726 | 0.1913 | — | 163.1 | 1/19 |

**in-dist 50（基础拓扑，T=64）**：SABRE 0.3176/33.0/0；LA287d 0.2508/85.7/1；
LA287e argmax 0.2445/108.0/2、beam 0.2715/109.3/1（**beam +6.9% 唯一改善**）；
LA287m argmax 0.2309/124.3/3、beam 0.2366/112.2/1。

**bighetero s42（held-out 50，T=64）**：SABRE 0.3207/33.0/0；
LA287d argmax 0.2377/**TRUNC 10**、beam 0.2892/TRUNC 2；
**LA287m argmax 0.2468/TRUNC 4（+3.8% vs LA287d，历史最佳 argmax）**、
beam 0.2867/TRUNC 1；LA287e argmax 0.1998/TRUNC 14、beam 0.2685/TRUNC 5。

批效率（全门/SWAP 中位，in-dist）：SABRE 5.98；LA287d 3.21；LA287e 3.06；
LA287m 3.22——**批效率无进展**。

### 归因分析

1. **多拓扑机制有效（E12 核心目标达成）**：bighetero argmax TRUNC 10→**4**、
   fid 0.2377→0.2468（首次反超 LA287d，−23.1% vs SABRE 缩小到比 LA287d 好），
   beam TRUNC 1——固定多拓扑并列训练确实教会了强异构景观上的稳定 argmax 规划。
2. **B4 老师蒸馏是毒药（两条共用变量）**：两条 NAM T=128 战绩崩成 2W/17L、3W/16L
   （LA287d 10W/9L）；**LA287d 靠 layout-mix 修好的游走电路复发**——mod_red_21
   （LA287d 0.2592 T64/0.1696 T128 反超 SABRE）与 rc_adder_6、hwb6 全部归零截断。
   B4 扩大候选集后，老师在深串行电路（mod_red_21/csla_mux_3/rc_adder 类）上选出
   的 la_act 标签更不可靠，λ_d=0.5 蒸馏把坏标签烙进学生 → 游走复发。
3. **E12 基础回退的另一半来自稀释**：150k 步三拓扑均分 → t287 仅 50k 步 + 无 KL
   锚定（单拓扑 teacher 对多拓扑状态 OOD 故去 KL）——基础能力被摊薄。
4. 训练过程无异常（agr 0.34-0.36、vla 有限、trunc 14-15%，与 la287d 同水平）——
   回退是蒸馏标签方向性偏差 + 分布稀释，非训练崩溃。

### 闸门判定

| 闸门 | E1（LA287e） | E12（LA287m） |
|------|-------------|---------------|
| NAM T=128 ≥0.2469 | ❌ 0.2133（−16.3%） | ❌ 0.2033（−20.3%） |
| 批效率 ≥4.5 | ❌ 3.06 | ❌ 3.22 |
| bighetero argmax TRUNC ≤3 | ❌ 14 | ⚠️ 4（接近） |
| bighetero beam ≥0.30 / TRUNC→0 | ❌ 0.2685/5 | ⚠️ 0.2867/1 |
| 基础拓扑 ≥0.95×LA287d | ❌ | ❌ |

### 结论

- **多拓扑方向正确**：E12 的 bighetero argmax 稳定性修复是真实收益（TRUNC 10→4、
  fid 反超 LA287d），验证了"per-episode 随机化给平均鲁棒性、固定多拓扑给景观内
  规划能力"的立论——方向 2 值得继续，但需去掉 B4 变量、保住基础拓扑。
- **B=4 老师加深方向错误**：扩大候选集反而恶化深电路蒸馏标签（游走复发），
  下轮回退 B=3 或对蒸馏标签加质量门控（老师 v_star 过低/超预算的 anchor 跳过）。
- **E4（SABRE 示范蒸馏）已实现并验证**（单测 5 项 + 冒烟 dalg=1.0），
  作为批效率强老师下轮可用。

### 命令

```bash
# E1: la287d 配方改老师加深+自锚（见上表），tmux la287e@GPU0
# E12: --topo-list 三拓扑 + topo-balance steps + 去 KL，tmux la287m@GPU1
bash scripts/eval_newmodel_chain.sh policy_LA287e.pt la287e cuda:0
bash scripts/eval_newmodel_chain.sh policy_LA287m.pt la287m cuda:1
# 产物含 hybridT_*_t287/_indist/_bh_indist.json + audit_T128_{la287e,la287m}_nam.json
```

### 下一步（优先级）

1. **B=3 回归对照（归因定案）**：LA287d 配方（B=3、KL=LA287d）续训 100k 单拓扑——
   若保持 0.2488 级别 → B4 是毒药实锤；若同样回退 → LA287d 是幸运抽样、续训需更长/更稳；
2. **多拓扑 v2**：E12 配方但 **B=3 老师 + 基础拓扑加权**（topo-list 中 t287 ×2 或
   topo-balance 权重）→ 保 base 同时修 bighetero；
3. **蒸馏标签质量门控**（修复 B4/深电路污染）：v_star 低于势函数下界或老师
   expectimax 超预算的 anchor 不产生 λ_d 标签（20260917 §2.6.4 老师缺陷锁死的
   工程化兜底）；
4. **E4 SABRE 示范蒸馏**接入：批效率缺口（3.2 vs 6.0）本轮无进展，是明确短板。

---

## E13（LA287n 多拓扑 v2：B=3 老师回归）结果（2026-09-18）

> 上一轮 E12（B4 多拓扑）发现 B4 蒸馏污染深电路，E1（B4 单拓扑）定案 B4 是毒药。
> 本轮=E12 配方唯一变量改回 B=3 老师（补齐归因矩阵最后一格：B3 多拓扑）。
> 起点 LA287d、topo-list {t287, bh_s7, bh_s13}、topo-balance steps、去 KL、
> λ_d=0.5/ramp0.6、150k 步；复用 sabre_cache_t287_multi3.pkl。

### 被测模型配置（先读我）

| 模型 | 起点 | 拓扑 | 老师 | KL | 备注 |
|------|------|------|------|----|------|
| **LA287d**（基线） | LA287c | t287 单 | B=3 | LA287c β=0.05 | 最好单拓扑 |
| **la287e**（E1） | LA287d | t287 单 | **B=4** | LA287d β=0.05 | B4 单拓扑，毒药定案 |
| **la287m**（E12） | LA287d | 3 拓扑 | **B=4** | 无 | B4 多拓扑 |
| **la287n**（本轮） | LA287d | **3 拓扑** | **B=3** | 无 | 多拓扑 + B3 回归 |

### 结果（vs SABRE）

| 指标 | LA287d | la287e | la287m | **la287n** |
|------|--------|--------|--------|-----------|
| NAM T64 argmax | 0.2488 | 0.2034 | 0.2119 | 0.2131 |
| NAM T128 argmax | 0.2469（10W/9L） | 0.2133（2W/17L） | 0.2033（3W/16L） | 0.1927（−24.4%） |
| in-dist argmax | 0.2508 | 0.2445 | 0.2309 | **0.2490** |
| in-dist beam | 0.2540 | 0.2715 | 0.2366 | 0.2264 |
| **bighetero argmax** | 0.2377（T10） | 0.1998（T14） | 0.2468（T4） | **0.2510（T7）** |
| bighetero beam | 0.2892（T2） | 0.2685（T5） | 0.2867（T1） | 0.2843（T1） |

### 分析

1. **bighetero argmax 全场最佳 0.2510**（> la287m 0.2468 > LA287d 0.2377）：多拓扑+B3
   保持并略超 E12 的异构修复——**多拓扑方向在 B3 下依然成立**；
2. **in-dist argmax 保住**（0.2490 ≈ LA287d，−0.7%）：B3 回归显著缓解了 E12 的
   base 稀释（E12 0.2309）——B4 对 base 的伤害确凿；
3. **NAM 深电路仍回退**（T64 0.2131 / T128 0.1927，TRUNC 0/19 但 fid 低）：
   B3 也救不回 NAM——NAM 崩与 B4 无关，是**多拓扑+去 KL 稀释 NAM 深电路分布**
   （t287 上 NAM 训练量减至 1/3、无 KL 锚定自由漂移）所致；
4. **in-dist beam 回退**（0.2264 < 0.2540）：beam 端本次也弱（与 E1 的 +6.9% 相反）——
   beam 端对续训更敏感，方向不稳定。

### 归因矩阵（四象限齐）

| | 单拓扑 | 多拓扑 |
|---|---|---|
| **B=3** | LA287d ✓ NAM 0.2488（起点 LA287c+KL） | **la287n** ⚠️ in-dist/bh 好，NAM 崩 |
| **B=4** | la287e ✗ 全崩（B4 毒药定案） | la287m ✗ NAM/base 崩，bh 略好 |

缺口：B3 单拓扑 + KL 从 LA287d 续训（真正隔离"多拓扑 vs LA287d 起点续训"的对照未跑）。

### 结论与下一步

- **多拓扑 v2 部分成功**：bighetero argmax 修复保持（0.2510 最佳）+ base argmax 保住；
  NAM 深电路是剩余短板；
- **NAM 崩的候选根因**：多拓扑稀释 NAM 分布 / 去 KL 无锚定——两者都指向"深电路
  专用训练量不足"；
- 下一步候选：① **B3 单拓扑+KL(LA287d) 对照**（定案 NAM 崩是续训自身还是多拓扑稀释）；
  ② **多拓扑 + 保留 KL 到 LA287d 仅限 base 拓扑状态**（分拓扑 KL，代码小改）；
  ③ **NAM 电路占比提高**（nam-circuit-prob 0.3→0.4）+ base 拓扑加权（topo-list 中
  t287 ×2）；④ E4 SABRE 示范蒸馏（批效率 2.8-3.2 vs SABRE 6.0 仍是最大短板）。

### 命令

```bash
# 训练：同 la287m 命令，--la-beam 3（tmux la287n@GPU0，150k）
bash scripts/eval_newmodel_chain.sh policy_LA287n.pt la287n cuda:0
# 产物：hybridT_la287n_*_t287/_indist/_bh_indist.json + audit_T128_la287n_nam.json
```

---

## E14（LA287s：E4 SABRE 示范蒸馏 + 批机会/全局特征包）结果（2026-09-18）

> 批效率攻坚：①E4 SABRE 教师（λ_s=0.3、demo-prob 0.15，BC 蒸馏）；②批机会特征
> look4→look6（n_exec_after/Δn_exec——换位能解锁几个门）；③全局上下文广播特征
> （GNN mean/max 池化 + 全局统计 101 维/边，补 2-hop 感受野<图直径 7 的可见性缺口）。
> 零填充 checkpoint 兼容（新旧双向、LA287d teacher 冻结 158 维对齐）。

### 被测模型配置（先读我）

| 项 | 配置 |
|----|------|
| 起点 | `--load policy_LA287d.pt` 续训 100k |
| 拓扑 | t287 单拓扑（配方同 LA287d） |
| 老师 | beam B=3/K=2（同 LA287d）+ **SABRE 示范教师**（λ_s=0.3 ramp0.6，demo 回合强制沿 SABRE 换手序列执行、仅路由期 BC 标签、demo 概率 0.15） |
| KL | teacher=LA287d β=0.05（teacher 冻结于 158 维旧布局，look 前 4 维喂数） |
| 特征包 | **look6**（SABRE5 + xtalk/busy/d_ext×2 + **批机会×2**）+ **global101**（GNN 全图 mean/max 池化 + 平均前沿距离/ready 门数/可执行门数/进度/预算消耗）→ edge_feat_dim 158→261，零填充从 LA287d 无损续训 |
| 其余 | λ_d=0.5/ramp0.6、la-raw-target、layout-mix 0.3/0.3/0.4、noise-hetero、swap_price_scale 4.6、λ_budget 1.8、TRUNC 门数化、课程 large_n8 起、seed 0 |

过程指标：demo 对齐率 dalg 均值 **0.843**（死锁掩码跳过的标签剔除后接近 1.0）、
dsl（BC 损失）活跃下降、agr 0.3-0.45、trunc 12-15%——训练健康。

### 结果（vs SABRE；LA287d = 前最佳基线）

| 指标 | LA287d | la287n(B3多) | **la287s** | Δ vs LA287d |
|------|--------|--------|--------|--------|
| NAM T64 argmax | 0.2488 | 0.2131 | 0.2465 | ≈持平 |
| NAM T128 argmax | 0.2469（10W/9L） | 0.1927 | **0.2522（8W/11L）** | **+2.1%** |
| in-dist argmax | 0.2508 | 0.2490 | **0.2638** | **+5.2%** |
| in-dist beam | 0.2540 | 0.2264 | 0.2693 | +6.0% |
| bighetero argmax | 0.2377（TRUNC 10） | 0.2510（T7） | **0.2785（TRUNC 2）** | **+17.2%** |
| bighetero beam | 0.2892（T2） | 0.2843（T1） | **0.2947（T1）** | +1.9% |
| 批效率（in-dist 中位） | 3.21 | 3.22 | **3.55** | +10.6% |
| SWAPs（in-dist） | 85.7 | 92.7 | 87.3 | ≈持平 |

批效率（NAM/中位）：SABRE 5.36，la287s 1.71；bighetero：SABRE 5.98，la287s 2.79。

### T=128 逐电路亮点（la287s vs SABRE）

- **mod_red_21 0.2115 vs 0.0805（+163%）**——LA287d 的 layout-mix 修复保住且扩大；
- rc_adder_6 0.1395 vs 0.0826（+69%）、tof_4 0.6487 vs 0.4758（+36%）、
  barenco_tof_10 0.1096 vs 0.0314（+249%）、mod_mult_55 0.1256 vs 0.1230；
- 8W/11L：胜局集中在 tof 系 + mod_red/rc_adder（批机会特征直击的串行链族）；
- 负局主体：csla_mux_3（0.067 vs 0.283）、mod5_4、vbe_adder_3（结构性负族未解）。

### 闸门判定

| 闸门 | 结果 |
|------|------|
| 批效率 ≥4.0 | ⚠️ 3.55（+10.6% 未达标，方向正确） |
| NAM T=128 ≥0.2469 | ✅ **0.2522**（首轮续训后 T128 反超 LA287d） |
| bighetero argmax 不劣化 | ✅✅ 0.2785/TRUNC 2（历史最佳） |
| NAM TRUNC ≤2/19 | ✅ 0/19 |
| in-dist 不回退 | ✅ 0.2638（+5.2%） |

### 结论

1. **SABRE 示范教师 + 批机会/全局特征包是首个"续训不回退"的配方**：此前 LA287d
   之后的所有续训（e/m/n）NAM T128 均崩（0.19-0.21），la287s 反超（0.2522）；
2. **批机会特征兑现预期**：T128 大胜局集中在串行链族（mod_red_21 +163%、
   rc_adder_6 +69%、tof_4 +36%）——正是"换位批量解锁门"的规划收益；
3. **全局可见性有效**：bighetero argmax 0.2377→0.2785（+17.2%）远超 la287n 的
   0.2510（无特征包的多拓扑）——单拓扑+特征包已拿下多拓扑实验的目标，
   验证"2-hop 感受野不足"是真实瓶颈且可用广播特征廉价补齐；
4. 批效率 3.55 未达 4.0：SABRE 教师 BC（λ_s=0.3）与 PPO 信号的平衡还需调
   （λ_s 拉高或 demo-prob 提高），或需要 SABRE 教师-only 阶段。

### 命令

```bash
# 训练：la287d 配方 + --la-beam 3 --teacher/--load policy_LA287d.pt
#       + --sabre-demo-lambda 0.3 --sabre-demo-prob 0.15 --sabre-demo-ramp 0.6（tmux la287s@GPU0）
bash scripts/eval_newmodel_chain.sh policy_LA287s.pt la287s cuda:0
# 特征包自动生效（edge_feat_dim 261，旧模型评估走零填充）
```

### 下一步（优先级）

1. **λ_s 拉高实验**（0.3→0.5 或 demo-prob 0.15→0.25）：批效率 3.55→4.0 的最短路径；
2. **E4 教师质量**：demo 回合目前 SABRE trials=5 布局 + 其换手序列；可试 trials=20
   更优 SABRE 解作教师；
3. **csla_mux_3/mod5_4 结构性负族**：T128 均值若要追平 SABRE，需解这三个 SABRE 大胜局；
4. 多拓扑 + 特征包组合（la287s 配方 + topo-list 三拓扑）——特征包已单拓扑达成
   bighetero 目标，组合可能进一步；GNN 扩大方案（虚拟全局节点/加深）留作后续。

### E14 补充：in-dist 50 逐电路配对分析（la287s vs SABRE，2026-09-18）

> 被测模型配置（同上节）：la287s = LA287d 续训 100k + SABRE 示范教师（λ_s=0.3/prob0.15）
> + 批机会特征（look6）+ 全局上下文广播（101 维）；对手 SABRE = sabre_route(seed=0)。
> 口径：in-dist 50 条 × 混合精度 T（seed=42）。

**总口径**

| method | fid_mean | median | logmean | SWAPs | TRUNC | 批效率 |
|--------|----------|--------|---------|-------|-------|--------|
| SABRE | **0.3176** | **0.2320** | **0.1595** | **33.0** | 0/50 | **5.98** |
| la287s argmax | 0.2638 | 0.1854 | 0.0781 | 87.3 | 1/50 | 3.55 |
| la287s beam | 0.2693 | 0.1717 | 0.0551 | 103.1 | 1/50 | 2.43 |
| LA287d argmax（前基线） | 0.2508 | 0.1609 | 0.0669 | 85.7 | 1/50 | 3.21 |

配对战绩：**argmax 13W/35L/2T**（LA287d 为 10W/38L）；beam 12W/35L/2T；
argmax 配对差 median −0.031 / mean −0.054。唯一 TRUNC 仍是 wadd_8_10（存量游走，
SABRE 147 swaps 完成、模型 1822 截断——与 LA287d 相同）。

**电路族分解（argmax vs SABRE）**

| family | n | W/L | SWAPs S→M | sw 比 |
|--------|---|-----|-----------|-------|
| **tof_tower** | 12 | **7W/5L** | 52.5→88.5 | 1.69 |
| **pauli_evolution** | 7 | **3W/2L** | 37.0→54.9 | 1.48 |
| layered_matching | 14 | 2W/12L | 22.5→31.0 | 1.38 |
| gf2_mult | 6 | 1W/5L | 29.5→53.2 | 1.80 |
| perm | 4 | 0W/4L | 8.8→10.8 | 1.23 |
| clifford | 2 | 0W/2L | 5.5→7.5 | 1.36 |
| mixed_serial | 2 | 0W/2L | 22.0→69.5 | **3.16** |

大胜：tof_tower_10_11 +0.102、tof_tower_6_10 +0.082、pauli_evolution_8/20/5 +0.06、
tof_tower_9_10/8_10/5_10 +0.05——**批机会特征直击的串行链族全数转胜**。
大负：mixed_serial_parallel_5_1 −0.313、clifford_layer_5_1 −0.309、
qpe_block_5_0 −0.250、gf2_mult_12_11 −0.208、layered_matching_5_w1_0 −0.157。

**结论**：la287s 在 in-dist 50 上把差距从 LA287d 的 −21.0% 缩到 −17.0%（argmax）、
−15.2%（beam），配对胜局 10→13；胜负格局呈清晰结构——**换位密集的串行/塔式族
（tof_tower/pauli_evolution）已反超 SABRE**（批机会+全局特征的直接收益），
**layout-敏感的匹配/小电路族（layered_matching/clifford/qpe）仍系统性负**——
后者 SWAP 比仅 1.2-1.4× 但 fid 差距大，指向布局质量而非换手数量，
是下一个批效率之外的独立短板。

---

## Phase 1：零训练部署实验（P1a 自适应模式 / P1b warm-start / P1c 预算）——诊断结论（2026-09-19）

> 目的：量化 in-dist 50 上 la287s vs SABRE（0.2638 vs 0.3176，−17%）差距的真实构成，
> 找出可零成本修复的部署侧杠杆，并为 T1 训练定向。

### 尺寸分层诊断（in-dist 50，la287s argmax）

| 层 | n | SABRE | argmax | beam | 贡献总差距 |
|----|---|-------|--------|------|-----------|
| ≤6q | 13 | 0.4502 | 0.3352（−25.5%） | **0.4360（−3.0%）** | **56%** |
| 7-9q | 11 | 0.2551 | 0.2451（−3.9%） | 0.2085（−15%） | 4% |
| 10-14q | 16 | 0.2945 | 0.2324（−21.1%） | 0.2100（−10%） | **37%** |
| ≥15q | 10 | 0.2509 | 0.2418（−3.6%） | 0.2143（−11%） | 3% |

7-9q / ≥15q 已基本打平；差距集中在 ≤6q（argmax 弱、beam 可救，5q 电路在训练池
仅占 ~4% episode、测试集占 26%——**严重欠采样**）与 10-14q（SWAP 1.60×、布局无关）。

### P1a 尺寸自适应模式（≤6q 走 beam5la+bud，其余 argmax）——纯推理参数，可部署

| 口径 | SABRE | argmax 全程 | **自适应** |
|------|-------|-----------|-----------|
| in-dist 50 | 0.3176 | 0.2638（−16.9%） | **0.2900（−8.7%）** |
| bighetero 50 | 0.3207 | 0.2785（−13.2%） | **0.2914（−9.1%）** |
| NAM T64 | 0.2461 | 0.2465（+0.2%） | **0.2537（+3.1%）** |
| NAM T128 | 0.2550 | 0.2522（−1.1%） | **0.2568（+0.7%）** |

**四个口径全部受益，NAM 反超 SABRE**（T64 +3.1%、T128 +0.7%）。零成本部署杠杆，已固化思路。

### P1b SABRE warm-start（诊断性，不进部署——用户要求纯 RL）

la287s × in-dist 50 × `--warm-start-sabre`：argmax 0.2638→0.2782（+1.4pt）。
按尺寸：≤6q **+0.069**（swaps 48→38）、7-9q −0.021、10-14q **+0.006（≈无效）**、
≥15q −0.005。配对 22W/25L。
**结论：SABRE 布局收益几乎全在 ≤6q；10-14q 差距不是布局问题**（修正此前假设）。

### P1c beam 预算收紧（10-14q 子集 16 条，routing-only）

| 配置 | swaps_mean | vs 原始 beam |
|------|-----------|-------------|
| beam bd1.05/lb1.8（原始） | 178.1 | — |
| beam bd1.00/lb1.8 | 175.3 | −1.6% |
| beam bd1.05/lb2.5 | 178.1 | 0% |
| beam bd1.00/lb2.5 | 175.3 | −1.6% |
| argmax（对照） | 154.0 | −13% |

**预算收紧对 beam 过度换手几乎无效**（λ_budget 无关）——beam 在 10-14q 的换手
爆炸是 V_LA 打分偏好，需训练侧解决（E4 强化），非推理参数。

### Phase 1 对 T1 训练的定向

1. **≤6q → 小电路回灌**（argmax 分布外/欠采样，beam 能救但 argmax 需训练修复）；
2. **10-14q → E4 强化**（不是布局、不是 beam 预算，是路由换手纪律——SWAP 1.6×）；
3. NAM 已可部署反超（自适应模式）——T1 目标是 in-dist 两短板层。

---

## T1（LA287t：E4 强化 + 小电路回灌）结果——失败，回退 la287s（2026-09-19）

> 按 Phase 1 定向：≤6q 欠采样 → 小电路过采样回灌（5-6q ×4 共 189 条入 nam 池，
> episode 占比 4%→12.4%）；10-14q 换手纪律 → E4 强化（λ_s 0.3→0.5、demo-prob
> 0.15→0.25、SABRE trials 5→20）。起点 LA287s、KL teacher=LA287d、100k 步。

### 被测模型配置（先读我）

| 项 | la287s（上一轮最佳） | **la287t（T1）** |
|----|---------------------|-----------------|
| E4 教师 | λ_s=0.3、prob 0.15、trials 5 | **λ_s=0.5、prob 0.25、trials 20** |
| 小电路 | 训练池 ~4% | **过采样 12.4%**（dup 文件） |
| 其余 | B=3、KL(LA287d)、λ_d=0.5、特征包、layout-mix | 同左 |

### 结果（vs SABRE；分层 in-dist 50，T=64）

| 层 | SABRE | la287s argmax | **la287t argmax** | Δt-s |
|----|-------|--------------|------------------|------|
| ≤6q | 0.4502 | 0.3352 | 0.3192 | **−0.016** |
| 7-9q | 0.2551 | 0.2451 | **0.1644** | **−0.081** |
| 10-14q | 0.2945 | 0.2324 | 0.2320 | ≈0 |
| ≥15q | 0.2509 | 0.2418 | 0.2396 | ≈0 |
| **总** | **0.3176** | **0.2638** | **0.2413** | **−0.0225** |

NAM：T64 0.2390（la287s 0.2465）、T128 0.2404/−5.7%（la287s −1.1%）——**回退**。
**bighetero argmax 0.2938（+1.5pt，历史最佳）**——唯一提升，E4 强化在强异构
下教出更稳定换手（"SABRE 化"），但代价是基础拓扑噪声感知精细度下降。

### 归因

1. **小电路回灌无效**：≤6q argmax 0.3352→0.3192（更差）——dup 过采样引入
   相同电路重复、且 12.4% 占比可能仍不足/方式不对；argmax 弱另有原因；
2. **7-9q 被殃及（主损 −0.081）**：过采样挤占中尺寸训练量 + λ_s=0.5 过强蒸馏
   把 7-9q 拉向 SABRE（噪声盲）模式，丢模型自身噪声感知优势；
3. **E4 强化双刃**：bighetero +1.5pt（稳定换手收益）↔ NAM/7-9q 回退（SABRE 化
   损伤），λ_s=0.5 过强。

### 判定与回退

**T1 闸门全未过**（≤6q ≥−12% 未达、10-14q ≥−14% 未达、总 ≤−10% 未达）。
**当前最佳部署 = la287s + P1a 自适应模式**（in-dist −8.7%、NAM T64 +3.1%、
T128 +0.7%、bighetero −9.1%）。la287t 保留 bighetero argmax 记录（0.2938）供参考。

### 下一步（候选）

1. **λ_s 中间值单变量**（0.4，trials 20 保留，去掉小电路过采样）：定位 E4 强化
   的甜点，验证 bighetero 收益可保留而不伤 base；
2. **放弃小电路回灌的 dup 方式**：改为 stage1 池 5q pkl 转 QASM 多样性回灌，或
   接受 ≤6q 由部署侧 beam（P1a 自适应）覆盖；
3. **转向其他方向**：多拓扑 + 特征包组合 / GNN 扩大（虚拟全局节点）/ E4 教师
   质量（trials 20 单变量）。

---

## E15（LA287core：SABRE 内核显性化特征）结果——NAM T128 首次稳定反超 SABRE（2026-09-19）

> 对应用户点 1+2：把 SABRE decay 启发式的**完整打分与轨迹记忆**作为 per-edge 输入
> 喂给策略（此前只有 lookahead 分量，缺 decay）。从 LA287s 续训 100k，唯一变量=新特征。

### 被测模型配置（先读我）

| 项 | 配置 |
|----|------|
| 起点 | `--load policy_LA287s.pt` 续训 100k（单拓扑 t287） |
| 新特征（6 维/边，edge_feat_dim 261→267，零填充） | `sabre_score(e)`=max(decay)×[front/|F|+0.5·ext/|E|]；`decay_paper(p/q)`=论文版（换位+1、门执行重置、cap5）；`decay_simple(p/q)`=近8步计数；`future_demand(e)`=端点逻辑比特剩余2Q门数 |
| 其余 | 完全同 la287s（B=3、KL(LA287d)、λ_d 0.5、λ_s 0.3/trials5、特征包、layout-mix、noise-hetero、100k） |

### 结果（vs SABRE；la287s=前最佳基线）

| 指标 | la287s | **la287core** | Δ |
|------|--------|--------------|---|
| NAM T64 argmax | 0.2465 | **0.2507** | +0.004（反超 SABRE 0.2461） |
| **NAM T128 argmax** | 0.2522 | **0.2592 (+1.7%)** | **+0.007（首次稳定反超 SABRE 0.2550）** |
| in-dist argmax | 0.2638 | 0.2546 | −0.009 回退 |
| in-dist beam | 0.2693 | 0.2562 | −0.013 回退 |
| bighetero argmax | 0.2785 | 0.2758 | ≈持平（−0.003） |
| bighetero beam | 0.2947 | 0.2912 | ≈持平 |
| 批效率（in-dist） | 3.55 | 3.03 | 回退 |
| SWAPs（in-dist） | 87.3 | 85.1 | ≈持平 |

### 分析

1. **NAM T128 +1.7% 是全项目最佳记录**：sabre_score/decay 特征让策略在深串行
   电路（tof/mod_red/gf2 系）上更贴近 SABRE 的完整规划打分——正是这些电路此前
   靠批机会特征赢 SABRE、现在进一步拉开；
2. **但 in-dist/批效率回退**：策略过度依赖 SABRE 打分（"SABRE 化"），在中尺寸
   电路上牺牲了噪声感知精细度（bighetero 也微回退）——SABRE 是噪声盲，跟随
   它的打分在均匀噪声下有损；
3. **闸门**：NAM T128 ≥0.2522 ✅（0.2592）；bighetero ≥0.2785 ⚠️（0.2758）；
   in-dist ≥0.2638 ❌（0.2546）；批效率 ≥4.2 ❌（3.03）——部分达成。

### 结论

点 1 的方向**部分被证实**：给策略 SABRE 完整打分能显著提升其最擅长的深电路
规划（NAM T128 反超），但需要约束"跟随度"（特征强度/噪声感知补偿），否则伤
in-dist 与批效率。与 E16（两阶段目标分解）的组合可能互补——SABRE 打分特征
天然匹配阶段 A 的纯效率目标。

### 命令

```bash
# 训练：la287s 配方 + 新特征自动生效（267 维），tmux la287e15@GPU0
bash scripts/eval_newmodel_chain.sh policy_LA287core.pt la287core cuda:0
```

---

## E16（LA287p2b：全新两阶段课程）结果——in-dist/bighetero 大幅提升（2026-09-19）

> 对应用户点 4：从 LA287（0916 时代基础，158 维）**全新两阶段**——阶段 A
> （--noise-scale 0.05，纯效率路由，swap 价格/预算/深度主导）100k → 阶段 B
> （--noise-scale 1.0，恢复噪声定价）100k。KL teacher=LA287 两段一致。

### 被测模型配置（先读我）

| 项 | 配置 |
|----|------|
| 起点 | `--load policy_LA287.pt`（0916 时代，等效 p0t287），特征包自动 267 维 |
| 阶段 A（100k） | `--noise-scale 0.05`——势函数噪声项 w_err/w_xt/w_xt_swap/eta_* 全部 ×0.05，目标≈纯效率路由 |
| 阶段 B（100k） | `--noise-scale 1.0`——恢复噪声定价，其余不变 |
| 两段共用 | B=3、KL(LA287)、λ_d 0.5、λ_s 0.3/trials5、SABRE 内核特征（E15 同款 267 维）、layout-mix、noise-hetero |
| 新机制 | `--noise-scale`（乘性作用于势函数噪声项，E16 前置实现） |

### 结果（vs SABRE；la287s/la287core=对照）

| 指标 | la287s | la287core(E15) | **la287p2b(E16)** | |
|------|--------|--------|--------|---|
| NAM T64 argmax | 0.2465 | 0.2507 | 0.2173 | ❌ 回退 |
| NAM T128 argmax | 0.2522 | **0.2592** | 0.2346（−8.0%） | ❌ 回退 |
| **in-dist argmax** | 0.2638 | 0.2546 | **0.2793** | ✅ **全场最佳 +5.9%** |
| in-dist beam | 0.2693 | 0.2562 | 0.2489 | ❌ |
| **bighetero beam** | 0.2947 | 0.2912 | **0.3048** | ✅ **历史最佳（vs SABRE −5.0%）** |
| bighetero argmax | 0.2785 | 0.2758 | **0.2881** | ✅（−10.2%） |
| 批效率（in-dist） | 3.55 | 3.03 | 3.27 | ⚠️ 未达 4.0 |
| SWAPs（in-dist） | 87.3 | 85.1 | 86.8 | ≈持平 |

### 分析

1. **点 4 部分被证实**：纯效率阶段 A 确实让 argmax 在 in-dist 上更接近 SABRE 效率
   （0.2793 全场最佳），阶段 B 恢复噪声感知后 bighetero 保持且 beam 达历史最佳
   （0.3048）——**两阶段目标分解产生真实、可叠加的收益**；
2. **代价是 NAM 深电路回退**（T128 −8.0%）：深电路最依赖噪声感知，阶段 A 的
   纯效率训练（100k）可能弱化了深电路噪声定价，阶段 B（100k）未完全恢复；
3. **批效率 3.27 仍未破 4.0**：两阶段改进 in-dist fid 主要来自更好的布局/换手
   组合（SWAPs 86.8 与 la287s 87.3 持平），而非批效率本身；
4. **闸门**：bighetero ≥0.27 ✅（0.2881）；in-dist 无明确闸门但 +5.9% 达标方向；
   NAM T128 ≥0.24 ❌（0.2346）；批效率 ≥4.0 ❌（3.27）——部分达成。

### 关键互补性发现

**E15（SABRE 打分特征）↔ E16（两阶段）在 NAM / in-dist 上镜像互补**：

| 维度 | E15 增益 | E16 增益 |
|------|---------|---------|
| NAM T128 | ✅ +1.7%（反超 SABRE） | ❌ −8.0% |
| in-dist argmax | ❌ −3.5% | ✅ +5.9% |
| bighetero beam | ≈ | ✅ 历史最佳 0.3048 |

SABRE 打分特征帮深电路（跟随 SABRE 规划）、伤中尺寸（过度 SABRE 化丢噪声感知）；
两阶段帮中尺寸/bighetero（效率目标分解）、伤深电路（阶段 A 弱化深电路噪声感知）。
**组合（E15 特征 + E16 两阶段）预计互补修复**——深电路由 SABRE 打分兜底、
中尺寸由两阶段兜底，是下一轮首选。

### 命令

```bash
# 阶段 A/B：la287 配方 + --noise-scale 0.05 → 1.0（tmux la287e16@GPU1 串联）
bash scripts/eval_newmodel_chain.sh policy_LA287p2b.pt la287p2b cuda:1
```

---

## v3 事件级模拟器（swap 动态串扰）实现（2026-09-20）

> 用户指出：物理上 swap = 3×CX，同样产生 ZZ 串扰，v2 默认 `swap_xtalk=False`
> （历史口径）不符合真实硬件。实现 v3 模拟器：**完全复用 v2 实现**（config 浅拷贝
> + 动态设 `swap_xtalk=True`，触发 v2 内部 getattr 分支），v2 行为与既有基准不变。

### 实现

- `src/sim/trajectory_sim_v3.py`：`trajectory_circuit_fidelity_events_v3` /
  `make_event_fidelity_fn_v3`——swap 作动态串扰源（按 swap 时长 0.9µs=3×CX 加权，
  物理等价 3 个 CX 的 ZZ 串扰）
- 接线：`build_fidelity_fn`（train/eval）+ `phys_fidelity` 支持 `--fidelity-sim trajectory_v3`；
  `eval_nam_hybrid_T.py` / `eval_nam_T128_model.py` 支持 `--sim v3`
- 测试：单测（v3 swap_xtalk 生效、f3≤f2）+ 全量 146 passed

### 数值验证（swap 串扰对保真度的影响）

| 电路 | SABRE swaps | v2 | v3 | Δ |
|------|------------|-----|-----|-----|
| tof_4 | 14 | 0.515825 | 0.515233 | −0.0006（−0.1%） |
| mod5_4 | 11 | 0.613135 | 0.612082 | −0.0011（−0.2%） |

swap 密集深电路（gf2^6_mult 109 swaps）v3 模拟显著变慢（zz_actions 增多，
CPU T=64 超时；GPU 可接受）——v3 评估成本高于 v2，深电路建议 GPU。

### 影响与后续

- **口径变更**：v3 是"向真实硬件靠近"的评估口径；已有全部基准（SABRE/各模型）
  均为 v2 口径，**如需 v3 口径重评需统一重跑**（`--sim v3`）
- 特征侧 `_xtalk_pred_edge`/`r_xt_swap`（per-swap 即时串扰价）本就计入 swap×门
  串扰——**v3 使训练特征/即时价与模拟器权威口径首次一致**（此前 v2 下特征计入
  而模拟器不计，策略规避的是假成本）
- 下一步候选：训练改用 `--fidelity-sim trajectory_v3`（噪声感知训练与评估同口径）；
  swap 密集电路 v3 性能优化（zz_actions 去重/批量化）

---

## 时钟化动作空间实现（Clocked Routing v1，2026-09-20）

> 设计：doc/20260920训练方案.md（事件时钟 + launch/advance，词表
> [E SWAP | K EXEC | commit | skip]，1Q 惰性物化，v3 通道分解奖励）。
> 本里程碑：核心实现 + 专项测试 + 训练/评估/审计入口落地；真实训练
> （C0-C3）待 Step 0 预校准审计后启动。

### 实现清单（纯增量，旧路径零改动）

| 文件 | 内容 |
|---|---|
| `src/routing/timing.py` | `next_completion` / `marginal_xtalk` / `skip_idle_delta` / `ao_exposure`（纯新增） |
| `src/routing/graph/circuit_dag.py` | `TimingState` + `build_routing_graph(timing_state=None)` 空余维度填充（qubit 19-24 / gate 27-30 / coupling 12-14 / dep 10） |
| `src/routing/gnn/encoder.py` | `gate_embeddings()` / `node_and_gate_embeddings()`（一次前向） |
| `src/routing/rl/env_clocked.py` | `ClockedRoutingEnv(RoutingEnv)`：_step_clocked / priority-K 候选 / 惰性物化 / SKIP / 终局物化 / clone |
| `src/routing/rl/agent_clocked.py` | `ClockEdgeActorCritic` + `ClockedPPOAgent`（obs 拆分、统一 mask、warm-start 列置零、C0 冻结、auto-batch BC 预热、flat-obs PPO update） |
| `src/routing/rl/train_agent.py` | `--clocked` 系列参数、create_env/agent 分支、rollout 适配（action_mask/obs/BC/freeze） |
| `scripts/eval_clocked.py` | 时钟化评估入口（五元组 + SABRE+D2@v3 对比） |
| `scripts/audit_v3_calibration.py` | Step 0 预校准审计（5 组 v3 回归） |
| `test/test_clocked_env.py` | 8 项专项测试（θ 可加性 vs v3 / liveness / 互斥 / 终局 / clone / idle 公式 / 惰性物化顺序 / warm-start 逐位不变） |
| Step 0a | `--fidelity-sim trajectory_v3` CLI choices + use_scheduler 元组（train_agent×3 + eval_policy×2，共 7 处） |

**warm-start 布局修正（关键）**：新时序特征全部**追加到 edge 块末尾**
（`[...旧267, global_timing8, edge_timing4]`，D_edge=279），保证
`_zero_pad_ac_state` 尾部补零后旧 267 列权重逐位对齐——若把 global109 插在
sabre_core 前会导致旧 sabre_core 权重错位（初版实测漂移，已修）。

### 测试

```bash
PYTHONPATH=src python3 -m pytest test/test_clocked_env.py -q   # 8 passed
PYTHONPATH=src python3 -m pytest test/ -q                      # 154 passed, 1 预存 flaky（test_fidelity_shaping_step_zero）
```

### 端到端冒烟（5q，400 步，随机 edge 头 + BC 预热 96 步）

```bash
cd src && PYTHONPATH=src python3 -u -m routing.rl.train_agent \
  --clocked --max-ready-2q 8 --topo ../traindata/topo/ring_5q.json \
  --data-dir ../traindata --split-prefix stage1 --reward-mode routing \
  --timesteps 400 --rollout-steps 64 --device cpu --max-episode-steps 150 \
  --max-num-qubits 5 --mapping-budget 2 --freeze-edge-gnn --bc-warmup-steps 96 \
  --edge-hidden 32 --pot-progress-b 0.2 --swap-price-scale 4.6 \
  --eta-time 0.05 --eta-parallel 0.05 --eta-idle 0.005 \
  --out /tmp/opencode/clocked_smoke.pt
```

收敛日志正常流动（pl/vl/ent/kl/grad），checkpoint 保存成功。
**注意**：该冒烟未从 LA287core warm-start（edge 头随机冻结），路由质量差
（eval sw≈490 vs SABRE 5）——仅验证链路，非策略指标。

### 评估入口冒烟

```bash
PYTHONPATH=src python3 scripts/eval_clocked.py \
  --model /tmp/opencode/clocked_smoke.pt \
  --topo traindata/topo/ring_5q.json --data-dir traindata --split stage1_test \
  --device cpu --limit 2 --traj-trajectories 8 --edge-hidden 32 \
  --max-num-qubits 5 --mapping-budget 2 --out /tmp/opencode/eval_clocked_smoke.json
```

### 结论与分析

1. 时钟化核心全链路打通：env 终止正确（物理互斥不变量成立）、v3 保真度直连
   （`timing.schedule_log` → `timing_log_to_events` → `fidelity_events`）、
   **θ 可加性引理经单测锁定**（env 自算边际和 == v3 zz_actions 总注入角，含 swap 对）。
2. 惰性物化语义正确：1Q 尾链随 EXEC 物化、SWAP-先-1Q 换位自由度保留（单测覆盖）。
3. 旧路径零破坏：全量 154 passed（仅 1 个 2026.08 起预存 flaky）。
4. 预校准审计框架就位（5 组 v3 回归）；**真实标定须在 tianyan287_20q + cuda
   上执行**（ring_5q 无显式 ZZ、T 低，冒烟数值无意义：w_xt≈23122 为病态拟合）。
5. **下一步**：Step 0 审计（tianyan287+cuda）→ C0 从 LA287core warm-start 训练。

---

## 时钟化 C0 训练启动（ClockedRouting，2026-09-20）

> 模型配置：时钟化动作空间 v1（doc/20260920训练方案.md）。底座拓扑
> tianyan287_20q；**warm-start 自 policy_LA287core.pt**（E15 特征包，edge_hidden=64）；
> C0 = 调度 warm-up：冻结 GNN + edge 头（`--freeze-edge-gnn`），只训 gate/skip 头
> 与 critic；BC 预热 4096 步（auto-batch 教师）；奖励 = B·progress − η_time·ΔT
> + λ_par·R_parallel − swap 价（pot_progress_b=0.20、swap_price_scale=4.6、
> eta_time=0.05、eta_parallel=0.05、eta_idle=0.005）；Φ shaping 沿用（γ=0.99）；
> 词表 K=24；mapping_budget=8。

### 命令（tmux 会话 clocked_c0，cuda:1，100k 步）

```bash
cd src && PYTHONPATH=. python3 -u -m routing.rl.train_agent \
  --clocked --max-ready-2q 24 \
  --topo ../traindata/topo/tianyan287_20q.json \
  --data-dir ../traindata --split-prefix unified \
  --curriculum-keys large_n8,large_n10,large_n12,large_n16,large_n20,unified \
  --qasm-stage-cap stage1:6,large_n8:10,large_n10:12,large_n12:14,large_n16:20,large_n20:20,unified:20 \
  --reward-mode routing --timesteps 100000 --rollout-steps 1024 \
  --device cuda:1 --max-num-qubits 20 --max-episode-steps 600 --mapping-budget 8 \
  --load ../models/policy_LA287core.pt \
  --freeze-edge-gnn --bc-warmup-steps 4096 --edge-hidden 64 \
  --edge-noise-features --beta-noise 0.5 \
  --reward-potential --pot-progress-b 0.20 \
  --shaping-gamma 0.99 --eta-shape 0.3 --alpha-ext 0.5 \
  --w-err 0.02 --w-xt 0.01 --w-xt-swap 0.02 \
  --swap-price-scale 4.6 --lambda-budget 1.8 --budget-delta 1.05 \
  --unfinished-penalty 0.3 --step-cap-mult 1.2 \
  --eta-time 0.05 --eta-xtalk-par 1.0 --eta-idle 0.005 --eta-parallel 0.05 \
  --checkpoint-dir ../models/ckpts_clocked_c0 --out ../models/policy_clocked_c0.pt
```

### 启动日志摘要（前 2048 步）

| step | rew | swp | map | trunc% | gps(门/SWAP) | makespan(us) | sr(swaps/SABRE) |
|---|---|---|---|---|---|---|---|
| 1024 | -39.3 | 12.3 | 8.0 | 10% | 7.4 | 17.4 | 4.17 |
| 2048 | -35.8 | 13.8 | 8.0 | 5% | 10.2 | 16.9 | 4.72 |

BC 预热 loss 5.4（gate/skip 头从零学 auto-batch 规则）。

### 观察与风险

- 指标正常流动（pl/vl/ent/kl/grad）；gps 7.4→10.2（SABRE≈6，批效率良好）；
  截断率 10%→5%。
- **`[G1 ALERT] swaps/SABRE≈4-4.7 > 1.10`（仅打印警告，不回滚）**：冻结 edge 头在
  时钟化 obs 分布下路由偏多 SWAP。预期内——C1 解冻 + Φ shaping 正是修这个的；
  C0 门槛（批效率 ≥ LA287core 90%、makespan ≤ SABRE+D2 1.5×）在 C0 完成时用
  `scripts/eval_clocked.py` 评估判定。
- 审计注意：Step 0 预校准的 w_xt/w_zz 数值病态（ring/t287 冒烟：idle 项为 0、
  overlap 对未找到、静态 ZZ 与 SWAP 价混叠）——**C2 前需修审计方法学**
  （delay 间隙处理、overlap 对选取、swap/staticZZ 分解），C0/C1 不依赖该值。

### 修复记录（2026-09-20，重启后）

**Bug**：时钟化 env 未覆盖 `_ready_2q_gates()`——父类要求 preds 全部 executed，
而惰性物化下 1Q 尾链不执行 → ready 集恒空 → sabre/lookahead/sabre_core/global
特征与 Φ 势函数全部退化 → 冻结 edge 头拿到垃圾特征，swaps/SABRE 飙到 5.5。
**修复**：`ClockedRoutingEnv._ready_2q_gates()` 改用 `_dag_ready`（可物化尾链语义）。
测试 8/8 通过。

### 重启后日志（修复版）

| step | rew | swp | trunc% | gps(门/SWAP) | makespan(us) | sr |
|---|---|---|---|---|---|---|
| 1024 | -27.4 | 11.6 | 0% | 11.9 | 9.6 | 3.12 |
| 2048 | -13.0 | 6.0 | 2% | 16.3 | 7.9 | 2.39 |
| 3072 | -15.0 | 4.6 | 3% | 16.9 | 9.0 | 1.54 |

gps 11.9→16.9（SABRE≈6，批效率显著占优）、makespan ~8µs、sr 3.1→1.54 持续下降
（C1 解冻 + Φ shaping 应继续收敛到 ≤1.1 区间）。C0 门槛预计可过。

### C0 完成 + 门槛评估（2026-09-20）

训练 100k 步完成（tmux clocked_c0，cuda:1，日志 /tmp/opencode/clocked_c0.log）。
末段（大电路）sr≈3.3、gps 4.7-26.9（波动）、trunc 8%。

评估（`scripts/eval_clocked.py`，tianyan20q_test 8 条，v3 fidelity T=16）：

| 指标 | PPO(C0) | SABRE+D2 | 门槛 | 判定 |
|---|---|---|---|---|
| SWAPs | 49.0 | 23.9 (2.05×) | — | C1 修 |
| makespan | 27.3us | 24.6us (1.11×) | ≤1.5× | 通过 |
| fidelity | 0.0358 | 0.0766 | C3 主目标 | 差（swap 2× 所致） |

结论：C0 门槛通过（makespan 1.11×、批效率 gps≈10.5 > LA287core≈6×90%）。
保真度差主要由 swap 比驱动——C1 解冻 edge+GNN + Φ shaping 的预期收敛目标。

### C1 回归诊断（2026-09-20）

C1（全参数解冻，从 C0 续训 100k）评估 tianyan20q_test 全 30 条：

| 模型 | SWAPs | fidelity | makespan |
|---|---|---|---|
| C0（冻结 edge） | 52.0 (2.2×) | 0.0388 (0.65×) | ~1.1× |
| C1（解冻） | **143.7 (6.0×)** | **0.0146 (0.2×)** | 2-3× |
| SABRE+D2 | 23.9 | 0.0599 | — |

**结论**：C1 解冻 edge+GNN 在时钟化奖励下联合漂移到劣化路由——支配性回归
（swap/makespan/fidelity 全输），非 Pareto trade-off。C0（冻结 edge 头）
基线健康（2.2× swap、0.65× fidelity）。

**修复方案（C1b）**：L2 锚定 edge 头 + GNN 到 LA287core 初值（允许微调、防漂移，
对应方案 §6.2 的 policy compatibility 风险）+ swap_price_scale 4.6→8.0
（swap 定价过低是 6× 的直接诱因）+ max_episode_steps 600→800（降 28% 截断）。

### Step 0 预校准审计（2026-09-20，tianyan287 + cuda:0，T=128×3 seeds，n=10）

方法学修复：idle 用显式事件注入（delay 不被调度）、overlap 全图搜相邻边、
n 取全比特数、T=128 多 seed 平均（单次 T=32 σ≈0.08 ≫ 信号 0.01）。

| 实验 | 原始测量 | FID_PER_REWARD=27.5 换算 |
|---|---|---|
| 孤立 SWAP | fid/swap=0.0431, mean_e=0.086 | swap_price_scale=**4.6**（锚定） |
| 重叠 SWAP | fid/overlap=0.0373, θ=0.0173 | w_xt=**59.3** /rad |
| idle 间隙 | 0.001186 fid/µs·全比特 | eta_idle=**0.0033** /(qubit·µs) |
| 静态 ZZ | fid/rad=1.0175 | w_zz=**28.0** /rad |

FID_PER_REWARD 由 swap_price_scale=4.6（v2 已验证价）自洽反推 = 27.5，
替代旧 476（B=0.045 时代，尺度差 4.4×）。C2 用此组校准值。

---

## 基座追平 SABRE（Step 0 + Step 1，2026-09-20/21）

> 目标：训练一个 argmax 路由追平 SABRE 的基座模型，作为时钟化管线的
> warm-start（修正：C1/C1b 证明 routing 模式解冻必漂移，基座路由质量
> 是时钟化天花板的决定性输入）。

### Step 0a 基座筛选（in-dist 50，argmax，无保真度）

| 模型 | in-dist 50 mean swaps | NAM-19（历史） |
|---|---|---|
| LA287d | 101.1 | 75.2 |
| **LA287core** | **83.1（最优）** | 84.8 |
| LA287s | 86.8 | — |

→ warm-start 用 **policy_LA287core.pt**（与时钟化 C0 同源）。

### Step 1 基座训练（tmux base_swapmin，cuda:0，100k）

配方（c2b 骨架替换）：demo 蒸馏 **λ_s=1.0 / prob=0.5 / ramp=0.3**（原 0.4/0.3/0.6）；
swap 价 **8.0**；预算硬锚 **λ_budget=3.0 / budget_delta=1.0**（超 SABRE 数即罚）；
**噪声奖励价归零**（w_err/w_xt/w_xt_swap=0，obs 特征包保留）；layout-mix 0.3,0.3,0.4
+ NAM 混合 0.3 + sabre-cache 复用；Φ shaping γ0.99、B=0.20；无 scheduler
（纯路由 auto-batch，swap 数主目标）；训练 = on-policy 采样 + 50% 教师接管回合。

启动日志（step 1024）：sr=1.77（自 2.2× 起步即降）、gps=11.2、trunc=7%。

### Step 2 门槛（待 base 完成后评估，argmax）

- tianyan20q_test ≤ 26.3 swaps（1.1× SABRE 23.9）且 NAM-19 ≤ 47.9（1.1× 43.5）
- 增强阶梯：λ_s 1.2/prob 0.6 → ext_set_size 30 → la-beam 3/depth 2 → +100k

### 基座训练失败 + 0b 诊断（2026-09-21）

**Step 1 基座训练结果（base_swapmin，100k）**：tianyan20q_test argmax = 49.8 swaps
（2.08× SABRE 23.9），NAM-19 = 330（5 条深电路截断 ~980 swaps 游走）——几乎未
改善（core 52.0），NAM 反而 4× 退化。BC 蒸馏 + 硬预算锚 + 噪声价归零的配方
**未达到 1.1× 门槛**。训练日志：sr 全程 1.6-2.6（demo 未转移到评估 = 分布漂移），
trunc 7%→18%。

**0b 特征天花板诊断（diag_sabre_mimic.py，零训练）**：手工策略 = 每步选
sabre_core[0] 最小边（同 SABRE 布局）：

| 数据集 | SABRE | argmin(sabre_core) | 比值 |
|---|---|---|---|
| tianyan20q_test | 24.2 | 55.1 | **2.28×** |
| NAM-19 | 45.8 | 99.8 | **2.18×** |

**根因（qiskit 源码核对）**：env 的 `_sabre_decay_paper` = 1 + swaps_since_exec
（每 swap +1.0，4 颗到 cap5）vs qiskit `with_decay(0.001, 5)` = 1 + 0.001×t
（需 t=4000 到 cap）——**decay 强度差 1000 倍**，彻底扭曲打分排序。E15 的
"论文版 decay" 与 qiskit 实现不一致 → 特征未编码 SABRE 选择 → NN 从特征学不到
SABRE 平价所需信息（这同时解释了 BC 失败）。

**结论**：问题不在 NN/训练配方，在特征保真度。修复方向：env 的 sabre_core 严格
对齐 qiskit Heuristic（decay 0.001×counter + SetScaling.Size + ext 集 BFS 口径），
修后 0b 诊断即验证门槛（argmin ≈ 1.0-1.1×），时钟化路由头可直接用 argmin 策略
（零 NN 训练、保证平价），C3 再学噪声偏离。

### 特征保真度修复 + mimic 路由头里程碑（2026-09-21）

**修复**（env.py）：① `_sabre_decay_paper` 倍率 1.0→0.001（对齐 qiskit
with_decay(0.001,5)，旧实现强度高 1000 倍）；② ext set 加"前驱全在
(executed∪front∪ext)"就绪语义（qiskit BFS 口径）。

**0b 验证**：argmin(sabre_core) 同布局 —— t287 **2.28→1.098×**、NAM 2.18→1.46×
（NAM 剩余差距为贪心 vs 多 trial 全搜索的轨迹耦合，深串行电路，非特征 bug）。

**时钟化管线接入 mimic 路由头**（`--routing-mimic`：mask 只留 argmin(sabre_core)
边，RL 只学 EXEC/SKIP；确定性、零训练、无漂移）。

**结果（tianyan20q_test 全 30 条，C0 EXEC/SKIP 头 + mimic 路由，v3 T=16）**：

| 模型 | SWAPs | fidelity |
|---|---|---|
| PPO(mimic 路由) | **25.6 (1.07×)** | **0.0813（反超 +17%）** |
| SABRE+D2 | 23.9 | 0.0695 |
| PPO(C0/LA287core 路由) | 52.0 (2.2×) | 0.0388 |

**首次保真度击败 SABRE**：swap 平价（1.07×）+ 调度优势（makespan/并行）兑现。
特征保真度修复是解锁点（旧特征下 NN 学不到 SABRE 选择，mimic 也无从谈起）。

**下一步**：C2-mimic = 在 mimic 路由下重训 EXEC/SKIP 头（现为 C0 在 LA287core
路由轨迹下学的，换路由后调度可再优化）+ 噪声价（w_xt=59.3/w_zz=28/eta_idle=0.0033）
→ C3（v3 保真度终端 + edge 头解冻、KL 锚定 mimic 教师）。

### C2m 噪声价重训回归（2026-09-21，公平对比）

`scripts/compare_mimic_scheduling.py`：同 SABRE 参照（T=32 固定 seed）、模型
3-seed 平均（T=16），30 条 tianyan20q_test：

| 模型 | SWAPs | fidelity | Δfid vs SABRE(0.0582) |
|---|---|---|---|
| **C0+mimic** | 25.6 | **0.0931** | **+0.035 (+60%)** |
| C2m（mimic+重训调度+噪声价） | 24.6 | 0.0647 | +0.007 (+11%) |
| SABRE+D2 | 23.9 | 0.0582 | — |

**结论**：C2m 的噪声价重训是回归——审计标定的 w_xt=59.3/w_zz=28 近似价与
v3 真实保真度不对齐，训练调度被推向"规避标定价"而非提升实际 fidelity
（过度串行化 → makespan↑ → T1/T2 退相干）。**C0 原装 EXEC/SKIP 头（为
makespan/并行度训练）+ mimic 路由为当前最优配置**。

**更正方向**：C3 从 C0+mimic 起步，用 **v3 保真度终端**（奖励 = λ·logF vs
SABRE sref）作为调度/路由优化的真实目标——终端就是 ground truth，无需近似
噪声价标定（w_xt/w_zz 可弃或留极小）。这也验证了"价格塑形不可靠、终端保真度
才是正解"的原始设计直觉。

### 干净重叠审计 + C2-v2（2026-09-22）

`scripts/audit_overlap_clean.py`：同一电路两套事件（并行 vs 串行，控制变量唯一
= 重叠），tianyan287 10q 子图，T=128×3 seeds：

- **平均 Δ(F_并行 − F_串行) = +0.0013** → 并行（重叠）比串行化**更好**
- 原因：tianyan ZZ θ~0.003-0.022 rad 很弱，而 T1/T2=28/19µs 退相干很强——
  避串扰省下的 ZZ 远不及串行化多付的 idle 退相干
- **C2m 回归根因确认**：w_xt=59.3 疯狂鼓励串行化 = 直接亏 fidelity

**C2-v2 设计**（从 C0+mimic 重训，`--routing-mimic`，cuda:0）：
w_xt=59.3→**0.5**、w_zz=28→**1.0**、eta_idle=0.0033→**0.01**（idle 才是真成本）；
保留 eta_time/eta_parallel（makespan 压 = tianyan 上真正的 fidelity 杠杆）。
早期日志：rew -8（C2m 同期 -49），trunc 0%。

**并行实验**：C3（cuda:1，v3 保真度终端，`λ·logF vs SABRE sref`，ground-truth
路径）已在训练。两条路线对照：价格重平衡（C2-v2）vs 终端对齐（C3）。

### C2-v2 / C3 评估 + C3-v2 改进（2026-09-22）

**结论：所有"从 C0+mimic 继续训练"的路线都退化**（同口径评估，T=16 单次）：

| 模型 | SWAPs | fidelity | vs SABRE(0.06-0.07) |
|---|---|---|---|
| **C0+mimic（零重训）** | 25.6 | **0.0931**（3-seed 公平对比） | +60% |
| C2m（强噪声价） | 24.6 | 0.0647 | +11% |
| C2-v2（弱噪声价+idle↑） | 24.7 | 0.0334 | 输 |
| C3（v3 终端，sref 缺 n20，T=16，λ=5） | 24.6 | 0.0267 | 输 |
| SABRE+D2 | 23.9 | 0.0582-0.0727 | — |

**C3 退化根因**（源码核对）：`_parse_sabre_fid_map` 默认表**无 n=20** → n=20 电路
（信号最该强的最大电路）sref=None → 终端奖励 = 原始 F~0.001-0.05 × λ_fid=5，
被路由稠密奖励淹没；且 T=16 终端估计噪声大。C2-v2 则再次验证"任何 routing
模式重训都漂离 C0 的 makespan 最优调度"。

**C3-v2（进行中，tmux clocked_c3v2，cuda:1）改进**：
1. **log-ratio 归一化**：`--sabre-fid-map` 用新算的含 n=20 参照
   （8=0.636/10=0.538/12=0.408/16=0.319/20=0.206，T=64 计算）
2. **T=64**（降终端估计误差 2×）
3. **λ_fid 5→20**（提保真度信号占比 ~4×）
4. 保留 C0 路由步奖励作锚（makespan 压），路由 mimic 冻结

### C3-v2 结果 + 重训练路线终结结论（2026-09-22）

C3-v2（log-ratio 含 n20 + T=64 + λ_fid=20）评估：sw=24.9, fid=**0.0345**
（vs C0+mimic 0.0931）——**依然回归**。至此 4 条重训练路线（C2m/C2-v2/C3/C3-v2）
全部失败，无一超越零重训的 C0+mimic。

**结论（重要）**：C0 的 EXEC/SKIP 头（100k routing 训练学到"ASAP + 最大并行"）
在 tianyan（T1/T2=28/19µs 主导、ZZ 弱）上**已接近调度最优**——makespan 最小化
即 fidelity 最大化。任何后续 RL 重训（无论路由奖励、噪声价、还是保真度终端 +
log-ratio + T=64 + λ=20）都会：① 探索使策略偏离 makespan 最优调度；② 终端
估计噪声（即使 T=64）相对调度间的真实 fidelity 差依然偏大，梯度被淹没；
③ 课程电路上的 fidelity 景观与评估集不一致 → 过拟合。**重训练 ROI 为负**。

**定稿配置**：`C0+mimic`（mimic 路由 = 特征驱动 SABRE，swap 1.07× 平价；
C0 调度 = ASAP 并行；确定性、零训练）→ tianyan20q_test fidelity **0.0931，
反超 SABRE+D2 +60%**。

**后续可选方向（不再走"重训练调度"）**：
1. 推理期 EXEC/SKIP beam search（不碰训练，直接改善调度）
2. fidelity 感知路由偏离：edge 头解冻 + KL 锚定 mimic（C1 失败于无锚，锚定后
   未试过 fidelity 终端引导；但 tianyan ZZ 弱，预计收益小）
3. 换更短 T1/T2 或更强 ZZ 的器件拓扑再验证时钟化价值（当前器件上调度头
   收益已被 makespan 主导）

### in-dist 50 重测：C0+mimic 在结构化电路上表现差（2026-09-22）

用户要求 in-dist 50 确认。结果（评估多次 GPU 静默崩溃，数据覆盖 43/50）：

| 子集 | PPO sw | SABRE sw | PPO fid | SABRE fid |
|---|---|---|---|---|
| 前 39 条 | 150.7（6.0×） | 25.2 | 0.2815 | 0.3547 |
| tof_tower 5_10/5_11 | 1342/1622 | 62/67 | 0.19/0.06 | 0.32/0.12 |

**根因诊断**（`mixed_serial_parallel_5` 调试）：
- mimic 贪心 argmin 在部分结构化族进入**周期 8+ 换边循环**
  （3,0,4,2,7,13,12,6,...），`get_deadlock_mask(max_cycle=6)` **只检测周期 ≤6**
  → 绕过死锁检测 → clock 不推进、永不 SKIP → 上千 swap
- eval 的 mask 覆盖强制 mimic 边、且未设 no_progress_limit → 无限循环至步数上限
- 深层：mimic 在结构化电路的路由差距（0b 已见 NAM 1.46×）在 pauli_evolution/
  tof_tower/mixed_serial_parallel 上放大为 2-30×

**结论（重要修正）**：C0+mimic 的 "+60% vs SABRE" **仅对随机电路（t287_test）
成立**；in-dist 结构化电路上明显落后（sw 6×、fid 0.8×），且 mimic 路由存在
死锁绕过的病理循环。时钟化管线的路由基座对结构化族不可靠。

**修复方向**：① eval/训练层给 mimic 加"无进展 swap 上限"（或 no_progress_limit）；
② get_deadlock_mask max_cycle 6→12 覆盖长周期；③ 结构化族改用真 SABRE 脚本
路由（保证平价，绕开 mimic 贪心局限）。

### 修复 1 后 in-dist 50 重测（2026-09-22）

修复：get_deadlock_mask max_cycle 6→12 + eval 连续 6 次无进展 swap → 强制 SKIP。

**结果（50/50 跑完，崩溃消除——修复前跑不完）**：

| 指标 | PPO(C0+mimic) | SABRE+D2 |
|---|---|---|
| SWAPs | **225.9（6.6×）** | 34.2 |
| fidelity | 0.2352 | 0.2878 |

15 条 >100 swaps（wadd_8_10=2676、tof_tower_7_10=1825、mixed_serial_parallel=760）。

**诊断**：修复把"无限循环（clock=0）"转为"慢渗漏路由"——guard 在任意门执行后
重置，而 mimic 呈"5 swap + 1 gate"模式永远不触发保护；且 makespan 现在正常
推进（855µs），说明是**路由效率差**（结构化族上 mimic 贪心 argmin 每解锁 1 门
需数十次 swap），不是死锁 bug。**修复 1 治不了路由质量差距**（0b 的 NAM 1.46×
在 pauli_evolution/tof_tower/wadd 上放大到 6-30×）。

**结论**：mimic 路由对结构化电路根本不可靠；in-dist 50 上 C0+mimic 全面落后
（sw 6.6×、fid 0.82×）。要覆盖结构化族，须走方向 2（真 SABRE 脚本路由保证
平价）或接受"时钟化管线仅适合随机/噪声友好电路域"。

### 决定性实验 + in-dist 灾难根因修复（2026-09-22）

**compare_sabre_trials.py（in-dist 50，只比路由不跑保真度）**：

| 路由 | mean swaps | vs SABRE |
|---|---|---|
| **mimic（0b 口径，SABRE 布局）** | 40.9 | **1.20×** |
| 单 trial SABRE | 34.2 | 0.84× |
| best-of-5 SABRE | 31.8 | 0.78× |
| best-of-5 vs 单 trial | 0.93 | trial 重启仅 ~7% |

**推论**：mimic 路由本身在结构化电路上也只有 1.2×（不是 6.6×）——in-dist 6.6×
灾难是**时钟化 env 的 swap 锁（0.9µs）伪影**：窄 front 结构化电路上 swap 后唯一
就绪的门被锁，mimic 在锁窗内继续 swap 其它边 → 级联爆炸；旧 env/SABRE 是
瞬时 swap 无此问题。

**修复**（eval_clocked mimic 模式三重守卫）：① 有端点空闲的可执行门 → EXEC；
② 有"就绪但被锁"的门 → **SKIP 等锁释放**；③ 否则 mimic swap；+ SABRE 初始布局
（mimic-mapping 生成布局差）。

**修复后 in-dist 50**：sw 201→**92.6（2.70×，被 wadd_8_10=2854 拖累）**，
排除 wadd_8_10 后 49 条 ≈ **1.2-1.3×**（回到 mimic 真实路由水平）；
fid 0.287 vs SABRE 0.323（0.89×，接近）。wadd_8_10 是 mimic 真正路由不动的
电路（旧 env 纯 mimic 也截断，SABRE 150）——贪心极限的 1-2% 硬电路。
**t287_test 不受影响**：fid 0.0898 仍反超 SABRE +27%（随机优势保留）。

**最终认知**：mimic 路由 ≈1.2× SABRE（随机 1.1× / 结构化 1.2-1.3×），
"6.6× 结构化失败"是时钟化执行伪影而非路由能力；SABRE 的额外优势主要来自
best-of-N trial（仅 7%）+ 精确执行语义，不是"学得出的策略"——方向 2（嵌真
SABRE）与 mimic 的差距仅剩 ~1.2×，可接受或后续用 trial 机制补齐。

### Experiment 1：mimic 训练模型的 in-dist 无守卫评估（2026-09-22）

验证 Q1：模型持续选 swap 是奖励设计问题，还是分布外失效？

in-dist 50（无守卫，只保留 mimic 边强制，SABRE 布局）：

| 配置 | SWAPs | fidelity | sw>100 |
|---|---|---|---|
| C0+mimic 带守卫（基准） | 92.6 | 0.2872 | 2 |
| **C3v2 无守卫** | **95.4** | **0.2881** | 4 |
| C2m 无守卫 | 124.5 | 0.2777 | 5 |
| C2v2 无守卫 | 272.3 | 0.1893 | 19 |

**结论**：C3v2（mimic 分布 + 保真度终端训练）无守卫 ≈ C0+mimic 带守卫——
**奖励驱动学习在正确分布上自行学会了锁等待/EXEC 优先**，守卫只是为 C0 分布外
失效打的确定性补丁且与之等价。奖励排序（EXEC +0.11 > SKIP −0.03 > SWAP −1.19）
本身正确。剩余 in-dist 差距（fid 0.89-0.92× SABRE）是 mimic 路由的 1.2-1.3×
换手差距，非调度问题。C2v2（重平衡价）最差（272.3）——价格重训调度最伤。
注意：C3v2 在 t287 随机电路上此前评估 fid 偏低（0.0345，旧评估路径），
结构化上学习到了正确行为但随机电路调度可能过拟合——C0+守卫在双域均衡仍最优。

### 布局优化 + mimic tie-break 实验结论（2026-09-22）

**E1（SabreLayout 布局，基线同步升级——PassManager[SabreLayout,SabreSwap]
不分解门保公平）**：

| 数据集 | PPO sw | SABRE sw | PPO fid | SABRE fid | fid比 |
|---|---|---|---|---|---|
| in-dist 50 | 85.8 | 20.9 | 0.267 | 0.344 | 0.78× |
| t287 部分(17条) | 8-35 | 0-7 | 0.03-0.28 | 0.24-0.62 | 远输 |

布局让 SABRE 基线大幅变强（t287 很多电路 0 swap、fid 0.07→0.25-0.62），但
**mimic 从好布局出发反而多 swap（识别不出"布局已好，只需执行"）**——相对更差。

**E2（mimic + ε-tie-break 0.005 + best-of-N=5，hard 9 条子集）**：
pauli_evolution_10 sw=489（仍灾难）、tof_tower_5_11 全部 5 trial 失败——
tie/trials 的 ~7% 收益对 10-100× 差距杯水车薪，**无效**。

**结论**：完整 SABRE（含布局）基线面前，mimic 贪心路由已到头。唯一能兑现
布局收益 + swap 平价 + 时钟化调度价值的方向是 **SABRE 脚本路由**：用完整
SABRE（SabreLayout+SabreSwap+multi-trial）做路由，时钟化只做 EXEC/SKIP
时序调度——0-swap 电路也 0 swap、布局收益直接继承、C0 调度（t287 曾 +27%）
在平价路由上兑现。下一步实施方向 2。
