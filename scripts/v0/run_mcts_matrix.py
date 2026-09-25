"""P1 Gate A 实验：MCTS 搜索能力上限 vs SABRE。

方法矩阵（20260922 方案九节）：
  - SABRE（外部，identity 布局，mean/best seeds）
  - Greedy（纯距离 + SABRE-score 两版）
  - MCTS-Oracle（uniform prior + Exact V*）：搜索能力上限
  - MCTS-0（uniform prior + random rollout）：pure search
  - MCTS-1（SABRE prior + SABRE rollout）：heuristic prior

Q1（固定 sims）、Q3（sims 曲线）→ SWAPs vs search budget。
Gate A 判据：Oracle 在 headroom 实例上是否 ≤ SABRE。

用法:
    PYTHONPATH=src python3 scripts/v0/run_mcts_matrix.py \
        --topo traindata/topo/ibmq_5_line.json --nq 5 --family perm_mix \
        --n2q-min 6 --n2q-max 8 --limit 8 --sims 10 50 100 200
"""

import argparse
import json
import os
import pickle
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
import sys
sys.path.insert(0, os.path.join(ROOT, "src"))

from routing.graph.circuit_dag import CircuitDAG
from routing.v0.baselines import (greedy_dist_policy, load_topo, make_env,
                                  run_episode, sabre_num_swaps)
from routing.v0.exact_solver import ExactSolver
from routing.v0.mcts import MCTSConfig, mcts_episode
from routing.v0.sabre_heuristic import SabreScorer, greedy_rollout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default="traindata/topo/ibmq_5_line.json")
    ap.add_argument("--data-dir", default="traindata/v0")
    ap.add_argument("--nq", type=int, default=5)
    ap.add_argument("--n2q-min", type=int, default=6)
    ap.add_argument("--n2q-max", type=int, default=8)
    ap.add_argument("--family", default="perm_mix")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--sims", type=int, nargs="+", default=[10, 50, 100, 200])
    ap.add_argument("--sabre-seeds", type=int, default=5)
    ap.add_argument("--sabre-trials", type=int, default=20)
    ap.add_argument("--methods", default="oracle,0,1",
                    help="逗号分隔：oracle / 0 / 1 / 2 / 3")
    ap.add_argument("--model", default=None, help="learned V_θ 权重 (m2/m3)")
    ap.add_argument("--out", default="benchmark/v0_p1_gateA.json")
    args = ap.parse_args()

    topo, cm, hw = load_topo(args.topo)
    manifest = json.load(open(os.path.join(args.data_dir, "manifest.json")))
    circuits = [x for x in manifest if x["nq"] == args.nq
                and args.n2q_min <= x["n2q"] <= args.n2q_max
                and x["family"] == args.family][:args.limit]
    methods = [m.strip() for m in args.methods.split(",")]
    value_net = None
    if "2" in methods or "3" in methods:
        import torch
        from routing.v0.value_net import ValueNet
        value_net = ValueNet()
        value_net.load_state_dict(torch.load(args.model, map_location="cpu"))
        value_net.eval()

    results = []
    print(f"{'circuit':<28}{'V*':>3}{'SABRE':>7}{'GDist':>6}{'GSabre':>7}" +
          "".join(f"{f'  M{mt}-{s}':>14}" for mt in methods for s in args.sims))
    for meta in circuits:
        with open(os.path.join(args.data_dir, meta["file"]), "rb") as f:
            qc = pickle.load(f)
        dag = CircuitDAG.from_circuit(qc)
        solver = ExactSolver(dag, cm, max_nodes=4_000_000)
        env0 = make_env(dag, cm)
        vstar = solver.solve(env0.executed_mask, env0.mapping)
        if vstar is None or vstar == 0:
            print(f"[skip] {meta['file']} V*={vstar}")
            continue

        # SABRE + Greedy
        sabre_list = [sabre_num_swaps(qc, cm, trials=args.sabre_trials, seed=s,
                                      initial_layout=list(range(args.nq)))[0]
                      for s in range(args.sabre_seeds)]
        sabre_mean = float(np.mean(sabre_list))
        env = make_env(dag, cm)
        gdist, _, _ = run_episode(env, greedy_dist_policy)
        env = make_env(dag, cm)
        gsabre, _ = greedy_rollout(env, SabreScorer())

        row = {"file": meta["file"], "vstar": vstar, "sabre": sabre_mean,
               "gdist": gdist, "gsabre": gsabre, "mcts": {}}
        print(f"{meta['file']:<28}{vstar:>3}{sabre_mean:>7.0f}{gdist:>6}"
              f"{gsabre:>7}", end="")
        for mt in methods:
            for s in args.sims:
                if mt == "oracle":
                    cfg = MCTSConfig(sims=s, prior="uniform", value="oracle", seed=0)
                    kw = {"solver": solver}
                elif mt == "0":
                    cfg = MCTSConfig(sims=s, prior="uniform", value="rollout_random", seed=0)
                    kw = {}
                elif mt == "1":
                    cfg = MCTSConfig(sims=s, prior="sabre", value="rollout_sabre", seed=0)
                    kw = {}
                elif mt == "2":
                    cfg = MCTSConfig(sims=s, prior="uniform", value="learned", seed=0)
                    kw = {"value_net": value_net}
                elif mt == "3":
                    cfg = MCTSConfig(sims=s, prior="sabre", value="learned", seed=0)
                    kw = {"value_net": value_net}
                else:
                    raise ValueError(mt)
                env = make_env(dag, cm)
                n, ok, st, _ = mcts_episode(env, cfg, max_steps=4000, **kw)
                n = n if ok else -1
                row["mcts"][f"{mt}_{s}"] = n
                print(f"{n:>14}", end="")
        print()
        results.append(row)

    # 汇总
    print("\n===== 汇总 =====")
    print(f"{'method':<20}{'mean':>8}{'min':>8}{'max':>8}{'win<=SABRE':>12}")
    n_inst = len(results)
    agg = {"sabre": np.mean([r["sabre"] for r in results])}
    for label, key in [("GreedyDist", "gdist"), ("GreedySabre", "gsabre")]:
        vals = [r[key] for r in results]
        agg[label] = np.mean(vals)
        print(f"{label:<20}{np.mean(vals):>8.2f}")
    print(f"{'SABRE':<20}{agg['sabre']:>8.2f}")
    for mt in methods:
        for s in args.sims:
            k = f"{mt}_{s}"
            vals = [r["mcts"][k] for r in results]
            ok = [v for v in vals if v >= 0]
            if not ok:
                continue
            win = sum(1 for r in results if r["mcts"][k] >= 0 and
                      r["mcts"][k] <= r["sabre"])
            print(f"{k:<20}{np.mean(ok):>8.2f}{min(ok):>8.0f}{max(ok):>8.0f}"
                  f"{f'{win}/{n_inst}':>12}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"topo": args.topo, "args": vars(args), "results": results},
                  f, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
