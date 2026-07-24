# AGENTS.md - rlrouting

Noise-adaptive quantum circuit routing via reinforcement learning and graph neural
networks. Combines the noise simulator (`src/sim/sim.py`) with a GNN-based fidelity
predictor and a PPO routing policy.

## Language / Framework / Package Manager

- **Language**: Python (>=3.10)
- **Package manager**: pip + `pyproject.toml` (setuptools)
- **Key dependencies**: `qiskit`, `qiskit-aer`, `torch`, `torch-geometric`,
  `networkx`, `gymnasium`, `numpy`

## Project Structure

```
rlrouting/
├── src/
│   ├── sim/sim.py              # 噪声模拟器（已有）
│   ├── routing/                # RL + GNN 路由子系统
│   │   ├── routing.py          # 顶层路由接口
│   │   ├── graph/              # 电路 DAG 构建 + 噪声特征编码
│   │   ├── gnn/                # Multi-GNN 保真度预测器 + 训练脚本
│   │   └── rl/                 # Gymnasium 环境 + PPO 智能体
│   └── utils/                  # 度量 + 训练数据生成
├── test/                       # pytest 测试
├── doc/GNN.md                  # GNN + RL 设计文档
└── pyproject.toml
```

## Commands

> 训练 / 运行脚本时需在 `src/` 目录下执行（或将 `src` 加入 `PYTHONPATH`）。

- **Install deps**: `pip install -e .`  (可选 `.[test]` 包含 pytest)
- **Run all tests**: `PYTHONPATH=src python -m pytest test/ -q`
- **Run a single test**:
  `PYTHONPATH=src python -m pytest test/test_predictor.py -q`
- **Train GNN predictor**:
  `cd src && python -m routing.gnn.train_predictor --out models/predictor.pt`
- **Train RL policy**:
  `cd src && python -m routing.rl.train_agent --predictor models/predictor.pt --out models/policy.pt`
- **Lint / typecheck / fmt**: 尚未配置（建议 ruff / mypy / black）
- **CI/CD**: 尚未配置

## Design References

- `doc/GNN.md` — 完整设计文档（问题定义、图构建、Multi-GNN、RL、API、超参数）
- `doc/sim.md` — 噪声模拟器文档
