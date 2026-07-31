# MCTS 推理实现方案

## 概述

在 Phase 1（SABRE 特征 + 距离奖励 + 死锁掩码）和 Phase 2（噪声感知微调）的基础上，实现蒙特卡洛树搜索（MCTS）作为推理时的决策增强方法，使 PPO 在路由推理阶段具备多步搜索能力，缩小与 SABRE 之间的 SWAP 效率差距。

## 当前架构回顾

| 组件 | 说明 | 文件 |
|------|------|------|
| `RoutingEnv` | Gymnasium 环境，`step()` 实现 SWAP + 自动执行 + 奖励计算 | `env.py` |
| `env.clone()` | 轻量浅拷贝（`object.__new__` 绕过 `__init__`），供 beam search 模拟使用 | `env.py:329` |
| `env.get_deadlock_mask()` | 检测来回 SWAP 震荡，返回禁止边数组 | `env.py:317` |
| `env._obs()` | 构建 obs：per-edge GNN 嵌入 + SABRE 特征 + mapping + progress | `env.py:188` |
| `EdgeActorCritic` | per-edge 打分 + 注意力池化价值头 | `agent.py:40` |
| `PPOAgent._forward_obs()` | 从 flat obs 解析 edge_feats/map_vec/progress，传给 AC，返回 logits+value | `agent.py:163` |
| `evaluate_circuit_beam()` | 1 步 beam search：top-K → clone → step → V(s') 评分 | `eval_policy.py:200` |

## 实现规划

### 阶段 3B：MCTS 推理（本次实现）

#### 1. 新增文件 `src/routing/rl/mcts.py`（~180 行）

##### 1.1 `MCTSNode` 数据类

```python
@dataclass
class MCTSNode:
    prior_probs: np.ndarray      # (num_edges,) 策略先验 P(s,a) = softmax(logits)
    valid_mask: np.ndarray       # (num_edges,) bool，合法动作（耦合边 + 非死锁）
    N: np.ndarray                # (num_edges,) 访问计数
    W: np.ndarray                # (num_edges,) 累计 value
    Q: np.ndarray                # (num_edges,) 平均 value = W / N
    children: dict[int, MCTSNode] # action → 子节点
    is_expanded: bool = False
```

节点不存储环境状态。每条仿真从根 `env.clone()` 开始沿树遍历。

##### 1.2 `MCTS` 类

```python
class MCTS:
    def __init__(
        self,
        agent: PPOAgent,
        num_simulations: int = 100,
        c_puct: float = 1.4,
        temperature: float = 0.0,
        max_depth: int = 200,
    ):
        ...

    def search(self, env: RoutingEnv) -> tuple[int, np.ndarray]:
        """返回 (best_action, action_probs)"""
```

##### 1.3 核心算法

```
search(env, agent):
  # 1. 构建根节点
  obs = env._obs()
  mask = build_action_mask(env, agent)
  logits, _ = agent._forward_obs(obs, mask.unsqueeze(0))
  priors = softmax(logits[0]).cpu().numpy()
  root = MCTSNode(priors, mask)

  # 2. 执行 num_simulations 次仿真
  for _ in range(num_simulations):
    sim = env.clone()
    node = root
    path = []                     # [(node, action), ...]

    # 2a. Selection
    while node.is_expanded and not terminal(sim):
      a = puct_select(node, c_puct)
      path.append((node, a))
      _, _, done, truncated, _ = sim.step(a)
      if done or truncated:
        break
      node = node.children[a]

    # 2b. Evaluation
    value = evaluate(node, sim, agent, done, truncated)

    # 2c. Backprop
    for n, a in reversed(path):
      n.N[a] += 1
      n.W[a] += value

  # 3. 根据访问次数选动作
  return select_action(root, temperature)
```

##### 1.4 子函数

| 函数 | 逻辑 |
|------|------|
| `build_action_mask(env, agent)` | `合法边 (coupling_map 范围) ∧ ¬死锁边 (get_deadlock_mask)` |
| `puct_select(node, c_puct)` | `a* = argmax_a [Q(a) + c_puct * P(a) * √(ΣN) / (1+N(a))]`，仅遍历 `valid_mask` |
| `evaluate(node, env, agent, done, truncated)` | done→0，truncated→-10，否则 `expand+forward` 取 V(s) |
| `expand(node, env, agent)` | `forward()` 得 logits+value，为每个合法动作创建子节点，设 `is_expanded=True`，返回 value |
| `select_action(root, temperature)` | temp=0 → `argmax N(a)`；temp>0 → `∝ N(a)^(1/τ)` 采样 |
| `terminal(env)` | `executed==全部` 或 `episode_step >= max_steps` |

PUCT 公式细节：

```
Q(a) = W(a) / N(a)           # a 已访问
Q(a) = 0                      # a 未访问（乐观初始化）

U(a) = c_puct * P(a) * √(Σ_b N(b)) / (1 + N(a))
score(a) = Q(a) + U(a)
```

##### 1.5 节点展开说明

展开时重复使用 `forward()` 一次调用获取全部动作的 logits 和 value：
- `logits` → softmax → `prior_probs`（写入子节点）
- `value` → 叶节点评估值（返回用于 backprop）

子节点中的 `prior_probs` 由父节点状态下的策略给出，这是 MCTS 标准做法。子节点被展开前不会进入 selection 的 PUCT 计算，所以 `prior_probs` 仅用于子节点自身的 `is_expanded=False` 阶段，实际不会影响搜索。

---

#### 2. 修改 `src/routing/rl/eval_policy.py`（+60 行）

##### 2.1 新增 `evaluate_circuit_mcts()`

```python
def evaluate_circuit_mcts(
    dag, hw, coupling_map, agent,
    reward_mode='routing',
    max_episode_steps=200,
    seed=0,
    noise_config=None,
    num_simulations=100,
    c_puct=1.4,
    temperature=0.0,
) -> CircuitMetrics:

    env = RoutingEnv(dag, hw, coupling_map, reward_mode=reward_mode,
                     max_episode_steps=max_episode_steps,
                     random_init=False, seed=seed,
                     gnn=agent.gnn, use_gnn=agent.gnn is not None,
                     noise_config=noise_config if reward_mode != 'routing' else None)
    obs, _ = env.reset()

    mcts = MCTS(agent, num_simulations=num_simulations,
                c_puct=c_puct, temperature=temperature,
                max_depth=max_episode_steps)
    t0 = time.perf_counter()

    done, truncated = False, False
    step = 0
    while not done and not truncated:
        best_action, _ = mcts.search(env)
        obs, reward, done, truncated, info = env.step(best_action)
        step += 1

    wall_time_ms = (time.perf_counter() - t0) * 1000
    return CircuitMetrics(...)  # 与 beam search 相同的字段
```

##### 2.2 CLI 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--search` | argmax | 推理搜索方法：`argmax` / `beam` / `mcts` |
| `--mcts-simulations` | 100 | 每步 MCTS 仿真次数 |
| `--mcts-c-puct` | 1.4 | PUCT 探索常数 |
| `--mcts-temperature` | 0.0 | 动作温度（0=确定性 argmax） |
| `--beam-width` | 0 | 兼容旧版，等价于 `--search beam`（当 >0 时） |

##### 2.3 `main()` 分发修改

```python
if args.search == 'mcts':
    m = evaluate_circuit_mcts(dag, hw, coupling_map, agent, ...,
                              num_simulations=args.mcts_simulations,
                              c_puct=args.mcts_c_puct,
                              temperature=args.mcts_temperature)
elif args.search == 'beam' or args.beam_width > 0:
    m = evaluate_circuit_beam(dag, hw, coupling_map, agent, ...)
else:
    m = evaluate_circuit(dag, hw, coupling_map, agent, ...)
```

---

#### 3. 关键技术决策

| 决策 | 选择 | 理由 |
|------|------|------|
| **状态存储** | 不存节点内，每次仿真从根 clone | env.clone() ~0.3ms，100 次仿真 ~30ms，开销可接受 |
| **叶节点评估** | 纯 V(s)，不累加步级奖励 | 与 beam search 一致；V(s) 已编码期望未来回报 |
| **终端值** | done→0，truncated→-10 | 与 beam search 评分体系一致 |
| **Dirichlet 噪声** | 不加（推理场景） | Phase 3C 训练时再引入 |
| **子节点 prior** | 展开时从父节点 forward 计算 | 一次性获取全部动作先验，价值 1 次 forward |
| **批处理** | 每路仿真独立 forward | 5-qubit 动作空间小（4-5），批量收益低 |

---

#### 4. 与 Beam Search 对比

| 维度 | Beam Search (1-step) | MCTS |
|------|---------------------|------|
| 搜索深度 | 1 步 | 多步（直到终端） |
| 分支覆盖 | top-K 固定数 | 全部合法动作，PUCT 自适应分配 |
| 状态评分 | V(s') 单次 | 多次仿真平均 Q(s,a) |
| 探索 | 无（top-K 贪婪） | PUCT （自动平衡 explore/exploit） |
| 每步计算量 | K × depth 次 GNN forward | `simulations × depth` 次 |
| 优势场景 | 短 horizon 收益明确（cross） | 多步规划（line） |

---

#### 5. 预期性能

5-qubit 电路，100 次仿真，拓扑 4 条边：

| 操作 | 单次耗时 | 100 次仿真总耗时 |
|------|---------|-----------------|
| env.clone() | ~0.3ms | ~30ms |
| step() | ~0.05ms | ~30ms（平均 depth 6） |
| GNN forward (in `step → _obs`) | ~0.05ms | ~30ms（复用） |
| GNN forward (in `expand`) | ~0.3ms | ~30ms |
| **合计/步** | — | ~**90ms** |

完整 episode（line 拓扑平均 ~7 SWAP 步）→ 约 **630ms**，beam3 约 ~230ms，argmax 约 ~5ms。

| 调节方式 | 效果 |
|----------|------|
| 降低 `simulations=50` | 每步 ~45ms，episode ~315ms |
| GNN forward 缓存（同状态复用） | 可减少 ~30% 开销 |

---

#### 6. 测试命令

```bash
cd src

# MCTS 推理 + 基线对比
python3 -m routing.rl.eval_policy \
  --model ../models/policy_phase1_noiseaware.pt \
  --topo ../traindata/topo/cross_5q.json \
  --data-dir ../traindata \
  --split stage1_phase1 \
  --search mcts \
  --mcts-simulations 100 \
  --mcts-temperature 0.0 \
  --baselines \
  --reward-mode routing

# 与 beam search 对比
python3 -m routing.rl.eval_policy \
  --model ../models/policy_phase1_noiseaware.pt \
  --topo ../traindata/topo/ring_5q.json \
  --data-dir ../traindata \
  --split stage1_phase1 \
  --search beam \
  --beam-width 3 \
  --reward-mode routing
```

---

### 阶段 3C：AlphaZero 风格训练

#### 总览

在 Phase 3B 推理 MCTS 验证有效的基础上，将 MCTS 引入训练循环。自对弈时用 MCTS 生成更优的访问分布 π_mcts，然后以监督学习方式训练策略和价值网络：

```
MCTS(π_raw) → π_mcts（搜索后的更好分布）→ CE(π_mcts, π_raw) 训练策略 → 下一次 MCTS 更强
                                                          → MSE(z, V(s)) 训练价值
```

与 PPO 的关键区别：

| 维度 | PPO | AlphaZero |
|------|-----|-----------|
| 训练目标 | GAE advantage（单条轨迹+奖励累积） | π_mcts（多路径搜索聚合结果） |
| value target | GAE return（含步级奖励） | 终端结果 z（SWAP 数 / fidelity） |
| 探索机制 | 熵奖励 coefficient | Dirichlet 噪声 + 温度退火 |
| 需要步级奖励 | 是 | **否**（只需终端结果） |

#### A. 改动点 1：`mcts.py`（+25 行）

##### 1. `search()` 增加 Dirichlet 噪声参数

```python
def search(
    self, env: RoutingEnv,
    add_dirichlet_noise: bool = False,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
) -> tuple[int, np.ndarray]:
```

根节点展开后（`_expand` 完成），对 `root.prior_probs` 注入噪声：

```python
if add_dirichlet_noise:
    self._add_dirichlet(root, dirichlet_alpha, dirichlet_eps)
```

##### 2. 新增 `_add_dirichlet()` 方法

```python
def _add_dirichlet(self, node: MCTSNode, alpha: float, eps: float):
    valid = np.where(node.prior_probs > 1e-12)[0]
    if len(valid) == 0:
        return
    # 总 concentration = 0.3，均匀分布在各合法动作
    alpha_vec = [alpha / len(valid)] * len(valid)
    noise = np.random.dirichlet(alpha_vec)
    for i, a in enumerate(valid):
        node.prior_probs[a] = (1 - eps) * node.prior_probs[a] + eps * noise[i]
```

参数来源：AlphaZero 论文标准值 `α=0.3, ε=0.25`。噪声只在自对弈时使用，推理时 `add_dirichlet_noise=False`。

#### B. 改动点 2：`agent.py`（+60 行）

##### 新增 `alphazero_update()` 方法

```python
def alphazero_update(self, batch: dict, batch_size: int = 128) -> dict:
    """AlphaZero 监督训练。
    
    batch keys:
      graph_data, map_vec, progress, sabre_feats, coupling_map  # 状态特征
      pi_mcts   # (N, max_edges) float — MCTS 访问分布（policy target）
      z         # (N,) float — 终端结果（value target）
    
    loss = CE(π_mcts, logits) + vf_coef * MSE(z, value)
    """
```

##### 核心实现

```
alphazero_update(batch, batch_size):
    n = len(batch["pi_mcts"])
    pi_raw = np.array(batch["pi_mcts"])       # (N, max_edges)
    z_raw  = np.array(batch["z"])              # (N,)
    
    # value 归一化: batch z-score
    z = (z_raw - z_raw.mean()) / (z_raw.std() + 1e-8)
    pi_t = torch.tensor(pi_raw, device=self.device)  # (N, E)
    z_t  = torch.tensor(z, dtype=torch.float32, device=self.device)  # (N,)
    
    log_data = {"pl": [], "vl": [], "gn": []}
    indices = np.arange(n)
    
    for _ in range 1:  # epochs=1（监督学习不需多次遍历）
        for start in range(0, n, batch_size):
            sel = indices[start:start + batch_size]
            
            # 1. 重建观测（复用现有 _build_edge_obs）
            cmaps = [batch["coupling_map"][i] for i in sel]
            sblist = [batch["sabre_feats"][i] for i in sel] if "sabre_feats" in batch else None
            ef, mv, pg = self._build_edge_obs(
                [batch["graph_data"][i] for i in sel],
                [batch["map_vec"][i] for i in sel],
                [batch["progress"][i] for i in sel],
                coupling_maps=cmaps,
                sabre_feats_list=sblist,
            )
            
            # 2. Action mask（排除无效边）
            mask = torch.zeros(ef.shape[0], self.num_edges, dtype=torch.bool, device=self.device)
            for i, cmap in enumerate(cmaps):
                mask[i, :len(cmap)] = True
            
            logits, value = self.ac(ef, mv, pg, action_mask=mask)
            
            # 3. CE loss: 合法动作平均
            log_probs = F.log_softmax(logits, dim=-1)                     # (B, E)
            ce_all = -(pi_t[sel] * log_probs)                             # (B, E)
            ce_masked = ce_all[mask]                                       # 只取合法边
            policy_loss = ce_masked.mean()
            
            # 4. Value loss
            value_loss = F.mse_loss(value, z_t[sel])
            
            # 5. 总损失
            loss = policy_loss + self.vf_coef * value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            all_params = list(self.ac.parameters())
            if self.gnn is not None:
                all_params += list(self.gnn.parameters())
            gn = nn.utils.clip_grad_norm_(all_params, 0.5)
            self.optimizer.step()
            
            log_data["pl"].append(policy_loss.item())
            log_data["vl"].append(value_loss.item())
            log_data["gn"].append(gn.item())
    
    return {k: float(np.mean(v)) for k, v in log_data.items()}
```

注意与 PPO 的关键差异：
- **无 `old_logp`**：不需要 PPO clipping，直接监督学习
- **无 `ent_coef`**：π_mcts 自带探索信号，不需要熵奖励
- **无 `approx_kl`**：无重要性采样，无 KL 项
- **value 归一化**：batch 内 z-score，防止 PPO 预训练 scale 与 AlphaZero terminal outcome scale 不匹配

#### C. 改动点 3：`train_agent.py`（+180 行）

##### 1. 新增 CLI 参数

```python
parser.add_argument("--mode", type=str, default="ppo",
                    choices=["ppo", "alphazero"],
                    help="训练模式: ppo / alphazero")
parser.add_argument("--mcts-simulations", type=int, default=50,
                    help="自对弈每步 MCTS 仿真次数")
parser.add_argument("--self-play-episodes", type=int, default=8,
                    help="每次 Update 前自对弈的 episode 数")
parser.add_argument("--self-play-temp", type=float, default=1.0,
                    help="自对弈动作采样温度（初期 1.0，后期退火 0.1）")
parser.add_argument("--dirichlet-alpha", type=float, default=0.3,
                    help="Dirichlet 噪声 concentration")
parser.add_argument("--dirichlet-eps", type=float, default=0.25,
                    help="Dirichlet 噪声混合权重")
parser.add_argument("--alphazero-train-steps", type=int, default=100,
                    help="每次自对弈后的梯度步数")
parser.add_argument("--alphazero-batch-size", type=int, default=128)
parser.add_argument("--alphazero-buffer-size", type=int, default=10000)
```

##### 2. 自对弈数据收集函数 `collect_self_play_data()`

```python
def collect_self_play_data(agent, env, mcts, mcts_simulations, mcts_c_puct,
                           dirichlet_alpha, dirichlet_eps, temperature, max_edges):
    """收集 (state_tuple, π_mcts, z) 三元组。"""
    buffer = []  # dict keys: graph_data, map_vec, progress, sabre_feats, coupling_map, pi_mcts, z

    while not done and not truncated:
        # 1. MCTS 搜索（带 Dirichlet 噪声）
        best_action, pi_mcts = mcts.search(
            env,
            add_dirichlet_noise=True,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
        )
        
        # 2. 存状态 + π_mcts（z 等 episode 结束回填）
        pi_padded = np.zeros(max_edges, dtype=np.float32)
        pi_padded[:len(env.coupling_map)] = pi_mcts[:len(env.coupling_map)]
        
        ep_states.append({
            "graph_data":   env._last_graph_data,
            "map_vec":      env._last_map_vec.copy(),
            "progress":     env._last_progress.copy(),
            "sabre_feats":  env._last_sabre_feats.flatten().copy(),
            "coupling_map": coupling_map,
            "pi_mcts":      pi_padded,
        })
        
        # 3. 按 π_mcts 采样动作（带温度）
        if temperature > 0:
            counts = pi_mcts ** (1.0 / temperature)
            action = np.random.choice(len(pi_mcts), p=counts / counts.sum())
        else:
            action = best_action
        
        obs, reward, done, truncated, info = env.step(action)
    
    # Episode 结束：计算 z
    if truncated:
        z = -unfinished_penalty * remaining_gates
    elif reward_mode == "routing":
        z = float(-info["num_swaps"])
    else:
        z = float(info.get("fidelity", 0.0))
    
    for state_dict in ep_states:
        state_dict["z"] = z
        buffer.append(state_dict)
    
    return buffer
```

##### 3. 主训练循环（AlphaZero 模式分支）

在 `main()` 中，根据 `args.mode` 分派：

```python
if args.mode == "alphazero":
    mcts = MCTS(agent, num_simulations=args.mcts_simulations,
                c_puct=args.mcts_c_puct)
    
    buffer = []  # replay buffer（FIFO）
    
    while total_steps < args.timesteps:
        # --- Phase A: 自对弈 ---
        new_data = collect_self_play_data(
            agent, env, mcts, ...)
        buffer.extend(new_data)
        if len(buffer) > args.alphazero_buffer_size:
            buffer = buffer[-args.alphazero_buffer_size:]
        
        total_steps += len(new_data)
        
        # --- Phase B: 训练 ---
        agent.gnn.train()
        for _ in range(args.alphazero_train_steps):
            batch = random.sample(buffer, min(args.alphazero_batch_size, len(buffer)))
            train_batch = _collate_az_batch(batch)
            losses = agent.alphazero_update(train_batch, batch_size=args.alphazero_batch_size)
        
        # --- Phase C: 日志 & checkpoint ---
        # 日志项: pi_ce, vl, gn, swp, trunc%, fid
```

##### 4. 数据整理辅助函数

```python
def _collate_az_batch(samples):
    return {
        "graph_data":   [s["graph_data"] for s in samples],
        "map_vec":      np.array([s["map_vec"] for s in samples]),
        "progress":     np.array([s["progress"] for s in samples]),
        "sabre_feats":  np.array([s["sabre_feats"] for s in samples]),
        "coupling_map": [s["coupling_map"] for s in samples],
        "pi_mcts":      np.array([s["pi_mcts"] for s in samples]),
        "z":            np.array([s["z"] for s in samples]),
    }
```

##### 5. 日志格式

```
step=  256  pi_ce=1.385  vl=0.873  gn=1.204  swp=5.8  trunc=0%  fid=0.782
```

日志项含义：
- `pi_ce`: cross-entropy loss（越低表示策略分布越接近 MCTS 访问分布）
- `vl`: value MSE loss（越低表示 V(s) 对齐终端结果越好）
- `gn`: gradient norm

#### D. 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| **z 定义** | routing: `-num_swaps`，noise_aware: `fidelity`，同一 episode 共享 | 与 AlphaZero 一致；V(s) 通过 state features（含 progress）区分不同阶段 |
| **replay buffer** | 简单 FIFO，hold 最近 10000 样本 | 避免复杂优先级采样；5-qubit 问题样本量小，FIFO 够用 |
| **训练 epochs** | epochs=1 | 监督学习不再需要 PPO 的 4-epoch oversampling |
| **π_mcts 存储** | pad 到 `max_edges`，loss 时通过 action mask 排除无效边 | 兼容多拓扑统一训练 |
| **value 归一化** | batch 内 z-score | 防止 PPO 预训练 scale 与 terminal outcome scale 不匹配 |
| **GNN 梯度反传** | 通过 `_build_edge_obs → node_embeddings` 自动完成 | 复用 PPO 已有梯度路径 |
| **课程学习** | 保留 `stage1_phase()` 进度函数 | 电路难度渐进 |
| **噪声扰动** | 保留 `perturb_noise_config()` | 提升泛化 |

#### E. 预期效果与风险

| 风险 | 缓解措施 |
|------|---------|
| **自对弈慢**（每步 50 次 MCTS × GNN forward） | 先用 `mcts_simulations=30` 验证流程，后续加大 |
| **value head 尺度漂移**（PPO 预训练 → 切换到 terminal z） | batch z-score 归一化；若不收敛，单独 warmup value head |
| **多拓扑 π padding 导致 CE 不准确** | action mask 只算合法动作，padding 贡献 0 |
| **早期策略未提升时自对弈样本质量低** | Dirichlet ε=0.25 保证一定探索；可采用 temperature 退火 (1.0→0.1) |

#### F. 训练命令

```bash
cd src

# 从 scratch 训练
python3 -m routing.rl.train_agent \
  --mode alphazero \
  --topo-list ../traindata/topo/cross_5q.json,../traindata/topo/ring_5q.json,../traindata/topo/ibmq_5_line.json \
  --reward-mode routing \
  --timesteps 50000 \
  --mcts-simulations 50 \
  --self-play-episodes 8 \
  --alphazero-train-steps 100 \
  --out ../models/policy_alphazero.pt

# 从 PPO 预训练模型微调
python3 -m routing.rl.train_agent \
  --mode alphazero \
  --topo-list ../traindata/topo/cross_5q.json,../traindata/topo/ring_5q.json,../traindata/topo/ibmq_5_line.json \
  --reward-mode noise_aware \
  --timesteps 30000 \
  --mcts-simulations 50 \
  --load ../models/policy_phase1.pt \
  --out ../models/policy_alphazero_noiseaware.pt
```

#### G. 工作量估算

| 文件 | 新增行 | 修改行 | 说明 |
|------|--------|--------|------|
| `mcts.py` | +25 | ~10 | Dirichlet 噪声 + search 签名扩展 |
| `agent.py` | +60 | 0 | `alphazero_update()` |
| `train_agent.py` | +180 | ~30 | AlphaZero mode + self-play + buffer |
| **合计** | **~265** | **~40** | **总 ~300 行改动** |
