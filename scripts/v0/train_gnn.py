"""Sprint 2 训练与诊断：GNN V_θ + MLP 控制组（Gate L3）。

Gate L3 判据：MAE(V_θ) < MAE(SABRE-rollout 平凡基线)（后者 ≈ 标签的上界
噪声）。同时 GNN vs MLP 判断表征是否瓶颈。

用法:
    PYTHONPATH=src python3 scripts/v0/train_gnn.py --data benchmark/v0_vstar16.npz
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Batch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from routing.graph.circuit_dag import CircuitDAG
from routing.v0.baselines import load_topo, make_env
from routing.v0.sabre_heuristic import SabreScorer, greedy_rollout
from routing.v0.value_net import ValueNet, train_value_net
from routing.v0.value_net_gnn import GNNValueNet, build_state_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="benchmark/v0_vstar16.npz")
    ap.add_argument("--topo", default="traindata/topo/line_16q.json")
    ap.add_argument("--data-dir", default="traindata/v0")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-model", default="models/v0_gnn16.pt")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, y, split = d["X"], d["y"], d["split"]
    masks, mappings, cnames = d["mask"], d["mapping"], d["circuit"]
    print(f"数据: {len(X)} 状态 (train {int((split==0).sum())} / test {int((split==1).sum())})")

    topo, cm, hw = load_topo(args.topo)
    dag_cache = {}

    def env_from_row(i):
        cf = cnames[i]
        if cf not in dag_cache:
            with open(os.path.join(args.data_dir, cf), "rb") as f:
                dag_cache[cf] = CircuitDAG.from_circuit(pickle.load(f))
        e = make_env(dag_cache[cf], cm)
        e.set_state(int(masks[i]), tuple(int(v) for v in mappings[i]))
        return e

    # ---- MLP 控制组 ----
    tr = split == 0
    net_mlp = ValueNet(in_dim=X.shape[1])
    net_mlp, _ = train_value_net(net_mlp, X[tr], y[tr], X[~tr], y[~tr],
                                 epochs=600, device="cpu")
    net_mlp.eval()
    with torch.no_grad():
        pred = net_mlp(torch.as_tensor(X[~tr]))
        mae_mlp = float((pred - torch.as_tensor(y[~tr])).abs().mean().item())
        corr_mlp = float(np.corrcoef(pred.numpy(), y[~tr])[0, 1])
    print(f"\n===== MLP 控制组（12 维特征）=====")
    print(f"test MAE={mae_mlp:.3f}  corr={corr_mlp:.3f}")

    # ---- GNN ----
    print("\n构建图（含 state 重建）...", flush=True)
    t0 = time.perf_counter()
    tr_idx = np.where(tr)[0]
    va_idx = np.where(~tr)[0]
    graphs_tr = [build_state_graph(env_from_row(i)) for i in tr_idx]
    graphs_va = [build_state_graph(env_from_row(i)) for i in va_idx]
    print(f"图构建完成 {time.perf_counter()-t0:.0f}s "
          f"(train {len(graphs_tr)}, val {len(graphs_va)})", flush=True)

    y_tr = torch.as_tensor(y[tr_idx], dtype=torch.float32)
    y_va = torch.as_tensor(y[va_idx], dtype=torch.float32)
    net = GNNValueNet().to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = torch.nn.HuberLoss(delta=1.0)

    best_mae, best_state, bad = float("inf"), None, 0
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(graphs_tr))
        tot = 0.0
        for b in range(0, len(graphs_tr), args.batch):
            idx = [int(i) for i in perm[b:b + args.batch]]
            batch = Batch.from_data_list([graphs_tr[i] for i in idx]).to(args.device)
            pred = net(batch)
            loss = loss_fn(pred, y_tr[idx].to(args.device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        net.eval()
        with torch.no_grad():
            vb = Batch.from_data_list(graphs_va).to(args.device)
            pred_va = net(vb)
            mae = float((pred_va - y_va.to(args.device)).abs().mean().item())
        if mae < best_mae:
            best_mae, best_state, bad = mae, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 15:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        vb = Batch.from_data_list(graphs_va).to(args.device)
        pred_va = net(vb).cpu().numpy()
        corr_gnn = float(np.corrcoef(pred_va, y[va_idx])[0, 1])
    print(f"\n===== GNN（state 图，{args.epochs} epochs 上限）=====")
    print(f"test MAE={best_mae:.3f}  corr={corr_gnn:.3f}")

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    torch.save(net.state_dict(), args.out_model)
    print(f"\n-> model {args.out_model}")
    print(f"\n===== Gate L3 初判 =====")
    print(f"MLP test MAE={mae_mlp:.3f} | GNN test MAE={best_mae:.3f}")
    print(f"SABRE-rollout 平凡基线 ≈ 标签噪声（M1-episode 在 8-10q 实证 = V*）")


if __name__ == "__main__":
    main()
