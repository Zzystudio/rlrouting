# ============================================================================
# train_predictor.py
# 监督训练 Multi-GNN 保真度预测器。
#
# 用法:
#   python -m routing.gnn.train_predictor --epochs 30 --out models/predictor.pt
# ============================================================================

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch_geometric.data import DataLoader

# 将 src 加入路径，便于以模块方式运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sim.sim import NoiseConfig, NoiseSimulator
from routing.graph.circuit_dag import RoutingGraphData
from routing.gnn.predictor import MultiGNNTidelityPredictor, fidelity_loss
from utils.data_gen import generate_dataset


def to_pyg(sample):
    data = sample.data.to_pyg()
    data.y = torch.tensor([sample.fidelity], dtype=torch.float)
    return data


def build_default_config(num_qubits: int = 5) -> NoiseConfig:
    """构造一个与 sim.md 示例一致、用于训练的默认噪声配置。"""
    # 线性拓扑 0-1-2-3-4
    coupling = [(i, i + 1) for i in range(num_qubits - 1)]
    return NoiseConfig(
        t1_times=[50.0] * num_qubits,
        t2_times=[70.0] * num_qubits,
        freq_ghz=[5.0] * num_qubits,
        single_q_gate_error=0.001,
        two_q_gate_error=0.01,
        coupling_map=coupling,
        readout_error=[0.02] * num_qubits,
        shots=1024,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Multi-GNN fidelity predictor")
    parser.add_argument("--num-qubits", type=int, default=5)
    parser.add_argument("--num-circuits", type=int, default=40)
    parser.add_argument("--layouts", type=int, default=4)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="models/predictor.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = build_default_config(args.num_qubits)
    print(f"生成数据集（电路={args.num_circuits}, 布局/电路={args.layouts}）...")
    samples = generate_dataset(
        config,
        num_circuits=args.num_circuits,
        layouts_per_circuit=args.layouts,
        depth=args.depth,
        seed=args.seed,
    )
    print(f"样本数: {len(samples)}")
    dataset = [to_pyg(s) for s in samples]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = MultiGNNTidelityPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in loader:
            optimizer.zero_grad()
            pred = model(batch)
            loss = fidelity_loss(pred, batch.y)
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.num_graphs
        avg = total / len(dataset)
        print(f"Epoch {epoch + 1}/{args.epochs}  loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), args.out)
    print(f"最佳模型已保存至 {args.out} (loss={best_loss:.4f})")


if __name__ == "__main__":
    main()
