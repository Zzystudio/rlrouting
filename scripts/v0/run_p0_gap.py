"""P0 交付物：SABRE 最优性 gap 表（搜索 headroom 量化）。

对每个小电路：
  - V*      = ExactSolver(A*) 从起点（identity 映射，executed_mask=0）的最优剩余 SWAP 数
  - SABRE   = Qiskit SabreSwap（identity 初始布局，trials 内取优）；多 seed 平均
  - Greedy  = 纯距离贪心（argmin Σ dist）
  - Random  = 均匀随机（多 seed 平均）
输出: 逐电路表 + 汇总 gap 比率（SABRE/V*, Greedy/V*, Random/V*）+ 求解器展开数校准。

用法:
    PYTHONPATH=src python3 scripts/v0/run_p0_gap.py \
        --topo traindata/topo/ring_5q.json --nq 5 --n2q-max 10 --limit 12
"""

import argparse
import json
import os
import pickle
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys_path = os.path.join(ROOT, "src")
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from routing.graph.circuit_dag import CircuitDAG
from routing.v0.baselines import (greedy_dist_policy, load_topo, make_env,
                                  random_policy, run_episode, sabre_num_swaps)
from routing.v0.exact_solver import ExactSolver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default="traindata/topo/ring_5q.json")
    ap.add_argument("--data-dir", default="traindata/v0")
    ap.add_argument("--nq", type=int, default=5)
    ap.add_argument("--n2q-max", type=int, default=10)
    ap.add_argument("--n2q-min", type=int, default=0)
    ap.add_argument("--family", default=None, help="如 random/chain/staircase/perm_mix")
    ap.add_argument("--seeds", type=int, default=5, help="SABRE/Random 的 seed 数")
    ap.add_argument("--trials", type=int, default=20, help="SABRE trials")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-nodes", type=int, default=2_000_000)
    args = ap.parse_args()

    topo, cm, hw = load_topo(args.topo)
    n_phys = len(topo["device_params"]["t1_times"])
    manifest = json.load(open(os.path.join(args.data_dir, "manifest.json")))
    circuits = [x for x in manifest
                if x["nq"] == args.nq and args.n2q_min <= x["n2q"] <= args.n2q_max]
    if args.family:
        circuits = [x for x in circuits if x["family"] == args.family]
    if args.limit:
        circuits = circuits[:args.limit]

    solver = ExactSolver.__new__(ExactSolver)  # 占位，实际每电路新建
    rows = []
    for i, meta in enumerate(circuits):
        with open(os.path.join(args.data_dir, meta["file"]), "rb") as f:
            qc = pickle.load(f)
        dag = CircuitDAG.from_circuit(qc)
        env = make_env(dag, cm)
        assert env.mapping == tuple(range(args.nq)), "初始映射应为 identity"

        # V*（A*）
        sol = ExactSolver(dag, cm, max_nodes=args.max_nodes)
        t0 = time.perf_counter()
        vstar = sol.solve(env.executed_mask, env.mapping)
        t_v = (time.perf_counter() - t0) * 1000.0
        if vstar is None:
            print(f"[skip] {meta['file']}: A* 超预算（expanded={sol.last_expanded}）")
            continue

        # SABRE（identity 布局, trials 取优, 多 seed 平均）
        sabre_list = []
        for s in range(args.seeds):
            n, ms, _ = sabre_num_swaps(qc, cm, trials=args.trials, seed=s,
                                       initial_layout=list(range(args.nq)))
            sabre_list.append(n)
        sabre_mean = float(np.mean(sabre_list))
        sabre_best_seed = float(min(sabre_list))

        # Greedy（确定性）
        env2 = make_env(dag, cm)
        n_g, ok_g, _ = run_episode(env2, greedy_dist_policy)

        # Random（多 seed 平均）
        rand_list = []
        for s in range(args.seeds):
            env3 = make_env(dag, cm)
            n_r, ok_r, _ = run_episode(env3, random_policy(np.random.default_rng(s)),
                                       max_steps=5000)
            rand_list.append(n_r if ok_r else float("nan"))
        rand_mean = float(np.nanmean(rand_list))

        rows.append({
            "file": meta["file"], "nq": meta["nq"], "n2q": meta["n2q"],
            "vstar": vstar, "sabre_mean": sabre_mean,
            "sabre_best_seed": sabre_best_seed, "greedy": n_g,
            "random_mean": rand_mean,
            "astar_ms": t_v, "expanded": sol.last_expanded,
        })
        print(f"[{i+1}/{len(circuits)}] {meta['file']}: V*={vstar} "
              f"SABRE(mean/best)={sabre_mean:.1f}/{sabre_best_seed:.0f} "
              f"Greedy={n_g} Random={rand_mean:.1f} | A* {t_v:.0f}ms "
              f"expanded={sol.last_expanded}")

    if not rows:
        print("无可用电路（全部超预算或空集）")
        return

    # 汇总
    print("\n===== 汇总（topo=" + os.path.basename(args.topo) + f"）=====")
    print(f"{'metric':<22}{'mean':>8}{'min':>8}{'max':>8}")
    for key, label in [("sabre_mean", "SABRE/V*"), ("sabre_best_seed", "SABRE(bestseed)/V*"),
                       ("greedy", "Greedy/V*"), ("random_mean", "Random/V*")]:
        ratios = [r[key] / max(r["vstar"], 1) for r in rows if r["vstar"] > 0]
        if ratios:
            print(f"{label:<22}{np.mean(ratios):>8.2f}{np.min(ratios):>8.2f}{np.max(ratios):>8.2f}")
    n_solved = sum(1 for r in rows if r["vstar"] == 0)
    print(f"最优(SABRE==V*) 实例数: {sum(1 for r in rows if r['sabre_mean'] <= r['vstar'])}/{len(rows)}")
    print(f"V*=0 实例: {n_solved}")

    out = os.path.join(ROOT, "benchmark", "v0_p0_gap.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"topo": args.topo, "rows": rows}, f, indent=2, default=str)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
