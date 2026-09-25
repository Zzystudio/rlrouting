"""P2 训练与诊断：V_θ MLP 拟合 V* + OOD MAE + value-greedy 诊断。

问题（Gate B）：
  1. V_θ 能否拟合 V*（in-dist MAE）？
  2. 搜索诱导的 OOD（D_search / 留出电路）上误差多大？
  3. value-greedy（argmin_a V_θ(s')，不带搜索）能否逼近 SABRE/M1？
     —— 若 V_θ 质量足够，MCTS-2/3 才有意义。

用法:
    PYTHONPATH=src python3 scripts/v0/train_value.py --data benchmark/v0_vstar.npz
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from routing.graph.circuit_dag import CircuitDAG
from routing.v0.baselines import load_topo, make_env, run_episode
from routing.v0.exact_solver import ExactSolver
from routing.v0.sabre_heuristic import SabreScorer, greedy_rollout
from routing.v0.value_net import ValueNet, train_value_net, value_greedy_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="benchmark/v0_vstar.npz")
    ap.add_argument("--topo", default="traindata/topo/line_8q.json")
    ap.add_argument("--data-dir", default="traindata/v0")
    ap.add_argument("--nq", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-model", default="models/v0_value_mlp.pt")
    args = ap.parse_args()

    d = np.load(args.data)
    X, y, split, dist, cnames = d["X"], d["y"], d["split"], d["dist"], d["circuit"]
    tr = split == 0
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[~tr], y[~tr]

    net = ValueNet(in_dim=X.shape[1], hidden=args.hidden)
    t0 = time.perf_counter()
    net, hist = train_value_net(net, X_tr, y_tr, X_te, y_te,
                                epochs=args.epochs, device=args.device)
    print(f"训练完成 {hist['epochs_done']} epochs, best val MAE = "
          f"{hist['best_val_mae']:.3f} ({time.perf_counter()-t0:.0f}s)")

    net.eval()
    with torch.no_grad():
        pred = net(torch.as_tensor(X_te))
        yp = pred.numpy()
        mae_all = float(np.abs(yp - y_te).mean())
        corr = float(np.corrcoef(yp, y_te)[0, 1])
    print(f"\n===== 测试集（留出电路 OOD）=====")
    print(f"MAE(all)  = {mae_all:.3f}   corr = {corr:.3f}")
    for dl, name in [(0, "expert"), (1, "random"), (2, "search")]:
        m = (dist[~tr] == dl)
        if m.sum() == 0:
            continue
        mae = float(np.abs(yp[m] - y_te[m]).mean())
        print(f"MAE({name:6s}) = {mae:.3f}  (n={int(m.sum())})")

    # ---- value-greedy 诊断 ----
    topo, cm, hw = load_topo(args.topo)
    manifest = json.load(open(os.path.join(args.data_dir, "manifest.json")))
    test_circuits = sorted(set(cnames[~tr].tolist()))
    scorer = SabreScorer()
    print(f"\n===== value-greedy 诊断（{len(test_circuits)} 个留出电路）=====")
    print(f"{'circuit':<26}{'V*':>4}{'SABRE':>7}{'GSabre':>7}{'VG(V_θ)':>8}")
    vg = value_greedy_policy(net)
    rows = []
    for fn in test_circuits:
        with open(os.path.join(args.data_dir, fn), "rb") as f:
            qc = pickle.load(f)
        dag = CircuitDAG.from_circuit(qc)
        solver = ExactSolver(dag, cm)
        env0 = make_env(dag, cm)
        vstar = solver.solve(env0.executed_mask, env0.mapping)
        if vstar is None:
            continue
        env = make_env(dag, cm)
        ns, _, _ = run_episode(env, lambda e: scorer.best_action(e))
        env = make_env(dag, cm)
        ng, _ = greedy_rollout(env, scorer)
        env = make_env(dag, cm)
        nv, okv, _ = run_episode(env, vg, max_steps=4000)
        rows.append({"circuit": fn, "vstar": vstar, "sabre": ns, "gsabre": ng,
                     "vg": nv if okv else -1})
        print(f"{fn:<26}{vstar:>4}{ns:>7}{ng:>7}{nv if okv else -1:>8}")

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    torch.save(net.state_dict(), args.out_model)
    print(f"\n-> model {args.out_model}")
    return rows


if __name__ == "__main__":
    main()
