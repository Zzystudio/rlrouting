# AGENTS.md - rlrouting

Noise-adaptive quantum circuit routing via reinforcement learning and graph neural
networks. Combines the noise simulator (`src/sim/sim.py`) with per-edge GNN encoding
and PPO with SABRE-inspired features, distance reward, deadlock masking, and beam search.

## Operating Environment / Hard Constraints

- **磁盘 `/dev/sda1`（挂载于 `/data1`）已损坏**：所有操作（读写文件、训练/评估的输入输出、checkpoint、日志、临时文件等）**严禁涉及该硬盘**。不要创建、修改、读取或计划任何位于 `/data1` 下的路径；可用磁盘为 `/dev/nvme0n1p5`（挂载于 `/home`，项目所在）与 `/dev/sdb1`（挂载于 `/data2`）。临时文件请使用 `src/`、`/home` 或 `/tmp`（非 `/data1`）。
- **每次跑完实验后，必须将实验结果写入 `doc/train.md`**：包括训练/评估命令、设置、关键日志摘要、评估结果表格以及结论与分析，按时间顺序追加到该文件末尾，并保持现有格式风格（`---` 分节、中文描述、代码块命令、markdown 表格）。
- **所有训练必须在 `tmux` 会话中后台进行**：使用 `tmux new -s <会话名>` 启动（或用 `tmux attach -t <会话名>` 恢复），训练命令在会话内运行，避免终端断开导致训练中断。
- **评估基线只保留 SABRE**：对比测试一律使用 `--baselines --no-greedy --no-random`（不考虑 greedy 与 random 算法），仅与 SABRE 对比。

## Language / Framework / Package Manager

- **Language**: Python (>=3.10)
- **Package manager**: pip + `pyproject.toml` (setuptools)
- **Key dependencies**: `qiskit`, `qiskit-aer`, `torch`, `torch-geometric`,
  `networkx`, `gymnasium`, `numpy`

## Project Structure

```
rlrouting/
├── src/
│   ├── sim/sim.py              # 噪声模拟器（Aer 仿真保真度计算）
│   ├── routing/
│   │   ├── routing.py          # 顶层路由接口（greedy, SABRE）
│   │   ├── graph/
│   │   │   ├── circuit_dag.py  # CircuitDAG 构建
│   │   │   └── features.py     # HardwareFeatures / RoutingGraphData
│   │   ├── gnn/
│   │   │   ├── encoder.py      # SubGNN 节点嵌入编码器
│   │   │   └── train_predictor.py
│   │   └── rl/
│   │       ├── env.py          # Gymnasium RoutingEnv（SABRE 特征、死锁掩码、clone）
│   │       ├── agent.py        # EdgeActorCritic + PPOAgent
│   │       ├── train_agent.py  # 训练入口（Phase 1/2, curriculum）
│   │       └── eval_policy.py  # 评估入口（argmax / beam search）
│   └── utils/
├── test/                       # pytest 测试
├── models/                     # 训练好的模型权重
├── traindata/                  # 训练数据集 + 拓扑 JSON + split 文件
│   ├── random/
│   ├── vqe/
│   ├── topo/                   # 拓扑定义（cross_5q, ring_5q, ibmq_5_line）
│   └── splits/                 # 数据集划分
├── doc/
│   ├── GNN.md                  # GNN + RL 设计文档
│   ├── sim.md                  # 噪声模拟器文档
│   └── train.md                # 训练记录（bug 修复、消融、各阶段结果）
└── pyproject.toml
```

## Key Design Decisions

| Decision | File | Rationale |
|----------|------|-----------|
| Per-edge 状态表示 | `env.py:_obs()` | 每条 coupling edge 构建局部特征 `[h_p, h_q, h_p-h_q]` 直接建模 `Q(s,a)`，解决全局 pooling 的 state-action ambiguity |
| 5 维 SABRE 特征 | `env.py:_obs()` | `front_dist_before/after, dist_improvement, num_improved/worsened` — 让 PPO 直接获取 SABRE 的距离信息 |
| 距离即时奖励 | `env.py:step()` | `r_dist = η·Δd / max(d_before,1)` 每步都有信号，缓解长电路 credit assignment |
| 死锁掩码 | `env.py:get_deadlock_mask()` | 检测来回 SWAP 震荡 → action mask 禁止无效循环，消除截断 |
| Phase 课程学习 | `train_agent.py` | Phase 1 纯路由 + SABRE 特征 → Phase 2 噪声感知微调 |
| 映射阶段 | `env.py:_step_mapping()` | 虚拟 SWAP 学初始布局 + commit 动作，布局与路由联合训练（动作空间 `num_edges+1`，观测含 phase） |
| Beam search 推理 | `eval_policy.py` | 1 步 lookahead: top-K → clone → step → V(s') 评分 |
| Action masking | `agent.py` | 支持动态屏蔽无效/死锁动作 |

## Training Pipeline

```
               ┌─────────────────────┐
               │   Phase 1: routing  │   SABRE 特征 + 距离奖励 + 死锁掩码
               │   timesteps=100000  │   → policy_phase1.pt
               └─────────┬───────────┘
                         ↓ load
               ┌─────────────────────┐
               │  Phase 2: noise     │   Aer 保真度终端奖励
               │   _aware            │   → policy_phase1_noiseaware.pt
               │   timesteps=100000  │
               └─────────┬───────────┘
                         ↓ eval
               ┌─────────────────────┐
               │  argmax / beam      │   evaluate_circuit / evaluate_circuit_beam
               │  search             │
               └─────────────────────┘
```

## Commands

> 训练 / 运行脚本时需在 `src/` 目录下执行（或将 `src` 加入 `PYTHONPATH`）。

- **Install deps**:
  ```bash
  pip install -e .
  ```
  （可选 `.[test]` 包含 pytest）

- **Run all tests**:
  ```bash
  PYTHONPATH=src python3 -m pytest test/ -q
  ```

- **Train Phase 1**（SABRE 特征 + 纯路由 + 映射阶段）:
  ```bash
  cd src
  python3 -m routing.rl.train_agent \
    --topo-list ../traindata/topo/cross_5q.json,../traindata/topo/ring_5q.json,../traindata/topo/ibmq_5_line.json \
    --reward-mode routing \
    --timesteps 100000 \
    --out ../models/policy_phase1.pt \
    --phase 1
  ```
  默认启用 `--mapping-phase`（映射阶段：虚拟 SWAP 学初始布局 + commit 动作，动作空间 `num_edges+1`）；
  旧行为用 `--no-mapping-phase`（兼容旧 checkpoint）。

- **Train Phase 2**（噪声感知微调）:
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

- **Evaluate PPO（argmax）**:
  ```bash
  cd src
  python3 -m routing.rl.eval_policy \
    --model ../models/policy_phase1_noiseaware.pt \
    --topo ../traindata/topo/cross_5q.json \
    --data-dir ../traindata \
    --split stage1_phase3 \
    --reward-mode routing \
    --baselines --no-greedy --no-random
  ```

- **Evaluate PPO（beam search）**:
  ```bash
  cd src
  python3 -m routing.rl.eval_policy \
    --model ../models/policy_phase1_noiseaware.pt \
    --topo ../traindata/topo/ring_5q.json \
    --data-dir ../traindata \
    --split stage1_phase1 \
    --beam-width 3 \
    --baselines --no-greedy --no-random
  ```

- **Lint / typecheck / fmt**: 尚未配置（建议 ruff / mypy / black）
- **CI/CD**: 尚未配置

## Best Known Results

### Route-only (routing mode, deterministic)

| Topology | PPO argmax | PPO beam3 | SABRE | Greedy |
|----------|-----------|-----------|-------|--------|
| cross_5q | 2.0 SWAPs | **1.3** | **1.2** | 1.8 |
| ring_5q | 1.6 SWAPs | **1.2** | **1.0** | 1.1 |
| ibmq_5_line | 2.8 SWAPs | **2.3** | **2.0** | 3.1 |

### Noise-aware (routing + Aer fidelity, deterministic)

| Topology | PPO SWAPs | SABRE SWAPs | PPO Fidelity | SABRE Fidelity |
|----------|----------|------------|-------------|---------------|
| cross_5q | **3.2** | 3.0 | **0.9453** | 0.9327 |
| ring_5q | **4.5** | 3.5 | **0.9419** | 0.9371 |
| ibmq_5_line | **6.7** | 5.5 | **0.9557** | 0.9228 |

PPO 在 line/cross 上保真度已超越 SABRE，beam search 将路由 SWAP gap 闭合了 63-88%。

## Design References

- `doc/GNN.md` — 完整设计文档（问题定义、图构建、Multi-GNN、RL、API、超参数）
- `doc/sim.md` — 噪声模拟器文档
- `doc/train.md` — 训练记录（bug 修复、消融实验、各阶段结果、beam search 评估）
