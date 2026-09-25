"""Sprint 1.4 标签阶梯校验（Gate L2）：8→10→12q 上近似标签 vs A* V*。

对每个实例（起点状态 s0）：
  - V*         = A* 精确最优剩余 SWAP
  - V̂_ens      = N=8 随机化 SABRE rollout 集成（mean）
  - V̂_ens_std  = 集成 std（不确定性）
  - V_M1@150   = MCTS-1（SABRE prior+rollout）根价值 @sims=150
  - V_M1@300   = 同上 @sims=300（收敛 gap 代理）
报告：每规模（8/10/12q）MAE/corr vs V*，退化曲线；
Gate L2 判据：8q MAE ≤ 1.0 且 M1@150 vs @300 gap 小。

用法:
    PYTHONPATH=src python3 scripts/v0/validate_labels.py \
        --topo traindata/topo/line_8q.json --nq 8 --family perm_mix --limit 6
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from routing.graph.circuit_dag import CircuitDAG
from routing.v0.baselines import load_topo, make_env
from routing.v0.exact_solver import ExactSolver
from routing.v0.mcts import MCTSConfig, mcts_episode, mcts_search
from routing.v0.sabre_heuristic import SabreScorer, ensemble_rollout_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default="traindata/topo/line_8q.json")
    ap.add_argument("--data-dir", default="traindata/v0")
    ap.add_argument("--nq", type=int, default=8)
    ap.add_argument("--n2q-max", type=int, default=16)
    ap.add_argument("--family", default="perm_mix")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--ens-n", type=int, default=8)
    ap.add_argument("--sims1", type=int, default=150)
    ap.add_argument("--sims2", type=int, default=300)
    ap.add_argument("--max-nodes", type=int, default=4_000_000)
    args = ap.parse_args()

    topo, cm, hw = load_topo(args.topo)
    manifest = json.load(open(os.path.join(args.data_dir, "manifest.json")))
    circuits = [x for x in manifest if x["nq"] == args.nq
                and x["n2q"] <= args.n2q_max and x["family"] == args.family]
    circuits = circuits[:args.limit]

    scorer = SabreScorer()
    rows = []
    for meta in circuits:
        with open(os.path.join(args.data_dir, meta["file"]), "rb") as f:
            qc = pickle.load(f)
        dag = CircuitDAG.from_circuit(qc)
        env = make_env(dag, cm)

        solver = ExactSolver(dag, cm, max_nodes=args.max_nodes)
        vstar = solver.solve(env.executed_mask, env.mapping)
        if vstar is None:
            print(f"[skip A*] {meta['file']}")
            continue

        v_ens, v_std, n_ok = ensemble_rollout_value(
            env, scorer, n=args.ens_n, beta=5.0, epsilon=0.02, seed=0)
        # 主标签：M1 episode 实际结果（v0/10q 实测 = V*，argmax-N 动作选择
        # 比 root-Q 价值估计可靠得多）
        cfg0 = MCTSConfig(sims=args.sims1, prior="sabre", value="rollout_sabre", seed=0)
        e0 = env.clone()
        v_ep, ok_ep, _, _ = mcts_episode(e0, cfg0, max_steps=2000)
        v_ep = float(v_ep) if ok_ep else float("inf")
        # 对比：best-child 价值
        cfg1 = MCTSConfig(sims=args.sims1, prior="sabre", value="rollout_sabre", seed=0)
        e1 = env.clone()
        _, st1 = mcts_search(e1, cfg1)
        v1 = -st1["best_child_Q"] if st1["root_N"] > 0 else float("inf")
        cfg2 = MCTSConfig(sims=args.sims2, prior="sabre", value="rollout_sabre", seed=0)
        e2 = env.clone()
        _, st2 = mcts_search(e2, cfg2)
        v2 = -st2["best_child_Q"] if st2["root_N"] > 0 else float("inf")

        rows.append({"file": meta["file"], "nq": args.nq, "n2q": meta["n2q"],
                     "vstar": vstar, "ens": v_ens, "ens_std": v_std,
                     "ens_n_ok": n_ok, "episode": v_ep, "m1_150": v1, "m1_300": v2})
        print(f"{meta['file']:<26} V*={vstar:>3}  ens={v_ens:>5.1f}±{v_std:>4.1f} "
              f"episode={v_ep:>5.1f} best@150={v1:>5.1f} best@300={v2:>5.1f}")

    if not rows:
        print("无可用实例（A* 全部超预算？）")
        return

    print(f"\n===== 标签质量（topo={os.path.basename(args.topo)}, "
          f"nq={args.nq}, n={len(rows)}）=====")
    for key, name in [("ens", "V̂_ens"), ("episode", "M1-episode"),
                      ("m1_150", "best@150"), ("m1_300", "best@300")]:
        vals = np.array([r[key] for r in rows if np.isfinite(r[key])])
        vs = np.array([r["vstar"] for r in rows if np.isfinite(r[key])])
        if len(vals) == 0:
            continue
        mae = float(np.abs(vals - vs).mean())
        corr = float(np.corrcoef(vals, vs)[0, 1]) if len(vals) > 1 else 0.0
        print(f"{name:<12} MAE={mae:.3f}  corr={corr:.3f}")

    gap = np.array([r["episode"] - r["m1_300"] for r in rows
                    if np.isfinite(r["episode"]) and np.isfinite(r["m1_300"])])
    if len(gap):
        print(f"episode vs best@300 gap: mean={gap.mean():.2f} max={gap.max():.2f}")

    out = f"benchmark/v0_labels_{os.path.basename(args.topo)}.json"
    with open(out, "w") as f:
        json.dump({"topo": args.topo, "rows": rows}, f, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
