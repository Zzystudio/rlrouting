"""Gate L3 端到端判定：learned V vs SABRE-rollout 价值（MCTS 内对比）。

流程：
  1. 分层切分（每族×档 80/20 按电路），V_θ MLP 只在训练电路上训练
     （early-stop 用训练电路内 10% 做 val，不碰留出电路）
  2. 留出电路上跑 M1@100 / M2@100 / M3@100（配对）+ 外部 SABRE 参照
  3. 配对 Wilcoxon + ≤SABRE 率

M1 = MCTS + SABRE prior + SABRE-rollout value（SABRE 价值估计器）
M2 = MCTS + uniform prior + learned V（prior 消融）
M3 = MCTS + SABRE prior + learned V（主方案）
removal ablation：M3 → 去 value → M1；M3 → 去 SABRE prior → M2
"""

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from routing.graph.circuit_dag import CircuitDAG
from routing.v0.baselines import load_topo, make_env, run_episode, sabre_num_swaps
from routing.v0.mcts import MCTSConfig, mcts_episode
from routing.v0.value_net import ValueNet, train_value_net


def _stratified_split(circuits, seed=0, test_frac=0.2):
    strata = {}
    for c in circuits:
        parts = c.replace(".pkl", "").split("_")
        fam = "perm_mix" if parts[0] == "perm" else parts[0]
        tier = parts[3] if parts[0] == "perm" else parts[2]
        strata.setdefault((fam, tier), []).append(c)
    rng = np.random.default_rng(seed)
    test = set()
    for k, cs in sorted(strata.items()):
        cs = sorted(set(cs))
        rng.shuffle(cs)
        test.update(cs[:max(1, int(len(cs) * test_frac))])
    return test


def _eval_circuit(task):
    """worker：单电路 × {sabre, m1@100, m2@100, m3@100}。"""
    file_path, topo_path, model_path, sabre_seeds, sabre_trials = task
    try:
        topo, cm, hw = load_topo(topo_path)
        with open(file_path, "rb") as f:
            qc = pickle.load(f)
        dag = CircuitDAG.from_circuit(qc)

        res = {}
        sabre_list = [sabre_num_swaps(qc, cm, trials=sabre_trials, seed=s,
                                      initial_layout=list(range(qc.num_qubits)))[0]
                      for s in range(sabre_seeds)]
        res["sabre"] = float(np.mean(sabre_list))

        net = ValueNet(in_dim=12)
        net.load_state_dict(torch.load(model_path, map_location="cpu",
                                       weights_only=True))
        net.eval()

        for name, cfg, kw in [
            ("m1@100", MCTSConfig(sims=100, prior="sabre", value="rollout_sabre", seed=0), {}),
            ("m2@100", MCTSConfig(sims=100, prior="uniform", value="learned", seed=0),
             {"value_net": net, "avoid_cycles": False}),
            ("m3@100", MCTSConfig(sims=100, prior="sabre", value="learned", seed=0),
             {"value_net": net, "avoid_cycles": False}),
        ]:
            env = make_env(dag, cm)
            n, ok, st, _ = mcts_episode(env, cfg, max_steps=600, **kw)
            res[name] = n if ok else None
        return {"file": os.path.basename(file_path), **res, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"file": os.path.basename(file_path), "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="benchmark/v0_vstar16.npz")
    ap.add_argument("--topo", default="traindata/topo/line_16q.json")
    ap.add_argument("--data-dir", default="traindata/v0")
    ap.add_argument("--limit", type=int, default=12, help="留出电路数上限")
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default="benchmark/v0_gate_l3.json")
    ap.add_argument("--model-out", default="models/v0_mlp16_strat.pt")
    ap.add_argument("--load-model", default=None,
                    help="跳过训练，直接加载该模型评测（DAgger 对比用）")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, y, cnames = d["X"], d["y"], d["circuit"]
    test_set = _stratified_split(sorted(set(cnames.tolist())))
    print(f"留出电路 {len(test_set)} 个")

    is_test = np.array([c in test_set for c in cnames])
    # early-stop val：训练电路内 10%（不碰留出电路）
    tr_idx = np.where(~is_test)[0]
    rng = np.random.default_rng(1)
    rng.shuffle(tr_idx)
    n_val = max(1, int(len(tr_idx) * 0.1))
    va_idx, tr_idx = tr_idx[:n_val], tr_idx[n_val:]

    if args.load_model:
        net = ValueNet(in_dim=X.shape[1])
        net.load_state_dict(torch.load(args.load_model, map_location="cpu",
                                       weights_only=True))
        net.eval()
        hist = {"epochs_done": 0}
        print(f"加载预训练模型 {args.load_model}（跳过训练）")
    else:
        net = ValueNet(in_dim=X.shape[1])
        net, hist = train_value_net(net, X[tr_idx], y[tr_idx], X[va_idx], y[va_idx],
                                    epochs=800, device="cpu")
        net.eval()
        os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
        torch.save(net.state_dict(), args.model_out)
    with torch.no_grad():
        p_va = net(torch.as_tensor(X[va_idx])).numpy()
    print(f"V_θ 就绪（val MAE={np.abs(p_va - y[va_idx]).mean():.2f}，"
          f"epochs={hist['epochs_done']}）")

    # 留出电路上端到端对比
    test_circuits = sorted(test_set)[: args.limit]
    tasks = [(os.path.join(args.data_dir, c), args.topo, args.model_out,
              5, 20) for c in test_circuits]
    print(f"\n端到端对比：{len(tasks)} 电路 × 4 方法（并行 {args.workers}）")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_eval_circuit, t) for t in tasks]
        for i, fut in enumerate(futs):
            try:
                rows.append(fut.result(timeout=1800))
            except Exception as e:  # noqa: BLE001
                rows.append({"file": "?", "error": str(e)})
            if (i + 1) % 5 == 0 or i + 1 == len(tasks):
                print(f"  [{i+1}/{len(tasks)}]", flush=True)

    ok_rows = [r for r in rows if r.get("error") is None
               and all(r.get(k) is not None for k in ["sabre", "m1@100", "m3@100"])]
    print(f"\n{'circuit':<26}{'SABRE':>7}{'M1':>6}{'M2':>6}{'M3':>6}")
    for r in ok_rows:
        print(f"{r['file']:<26}{r['sabre']:>7.0f}{r['m1@100']:>6.0f}"
              f"{r['m2@100']:>6.0f}{r['m3@100']:>6.0f}")

    from math import erf, sqrt
    def wilcoxon(a, b):
        dd = np.asarray(a, float) - np.asarray(b, float)
        dd = dd[dd != 0]
        n = len(dd)
        if n < 3:
            return 1.0
        absd = np.abs(dd)
        order = np.argsort(np.argsort(absd)) + 1.0
        ranks = order.copy()
        for v in np.unique(absd):
            m = absd == v
            if m.sum() > 1:
                ranks[m] = order[m].mean()
        T = float(ranks[dd > 0].sum())
        T = min(T, n * (n + 1) / 2 - T)
        mu = n * (n + 1) / 4
        sig = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (T - mu) / sig
        return max(2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))), 1e-12)

    print(f"\n===== Gate L3 端到端判定（n={len(ok_rows)}）=====")
    for key, label in [("m1@100", "M1(rollout)"), ("m2@100", "M2(unif+LV)"),
                       ("m3@100", "M3(sabre+LV)")]:
        a = np.array([r["sabre"] for r in ok_rows])
        b = np.array([r[key] for r in ok_rows])
        print(f"{label:<14} mean={b.mean():6.1f}  vs SABRE {a.mean():6.1f}  "
              f"≤SABRE率={float((b <= a).mean()):.2f}  p(vs SABRE)={wilcoxon(b, a):.4f}")
    a1 = np.array([r["m1@100"] for r in ok_rows])
    a3 = np.array([r["m3@100"] for r in ok_rows])
    print(f"\n[removal ablation] M3 vs M1（learned V vs rollout value）: "
          f"mean {a3.mean():.1f} vs {a1.mean():.1f}, p={wilcoxon(a3, a1):.4f}, "
          f"≤M1率={float((a3 <= a1).mean()):.2f}")
    a2 = np.array([r["m2@100"] for r in ok_rows])
    print(f"[prior ablation] M3 vs M2（去 SABRE prior）: mean {a3.mean():.1f} vs "
          f"{a2.mean():.1f}, p={wilcoxon(a3, a2):.4f}")

    with open(args.out, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
