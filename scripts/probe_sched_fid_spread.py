#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探针 0a：调度策略间的保真度 spread（方向2 Phase 1 前置判据）。

问题：在 SABRE 路由平价的前提下，纯调度（EXEC 顺序 + SKIP 时机）能在
保真度上做出多大差异？若 spread < 模拟器噪声 σ，则结构化域终端 fid 信号
不可学（train 只用解析奖励），RL 调度收益预期也要下调。

策略：asap（priority 槽序）/ anti（反序）/ random×K（均匀随机槽序），
同一 routed 电路、同一 scheduling_only 环境，fid = trajectory_v3(T=32)。
σ 估计：同一 asap 调度用两个独立 fid 种子评估，σ ≈ |Δ|/√2。

用法（项目根）：
  PYTHONPATH=src python3 scripts/probe_sched_fid_spread.py \
    --topo traindata/topo/tianyan287_20q.json \
    --circuits /tmp/opencode/indist50_list.txt --n-random 8 \
    --out /tmp/opencode/probe0a_indist.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing.rl.eval_policy import load_topo
from sim.trajectory_sim_v3 import make_event_fidelity_fn_v3
from scripts.sched_common import (sabre_route_full, make_sched_env,
                                  run_sched_episode, sched_episode_stats)
from scripts.eval_clocked import qasm_load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default="traindata/topo/tianyan287_20q.json")
    ap.add_argument("--circuits", required=True,
                    help="qasm 路径列表 txt（绝对/相对行）")
    ap.add_argument("--traj", type=int, default=32)
    ap.add_argument("--n-random", type=int, default=8)
    ap.add_argument("--sim-device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/tmp/opencode/probe0a.json")
    args = ap.parse_args()

    config, hw, cm = load_topo(args.topo)
    max_edges = max(19, len(cm))

    paths = []
    with open(args.circuits) as f:
        for line in f:
            line = line.strip()
            if line and os.path.exists(line):
                paths.append(line)
    if args.limit > 0:
        paths = paths[:args.limit]
    print(f"[probe0a] {len(paths)} circuits, traj={args.traj}, "
          f"random={args.n_random}")

    # 两个独立种子的 fid 函数（σ 估计 + 主评估）
    fid_main = make_event_fidelity_fn_v3(config, num_trajectories=args.traj,
                                         seed=0, backend=args.sim_device)
    fid_rep = make_event_fidelity_fn_v3(config, num_trajectories=args.traj,
                                        seed=1234, backend=args.sim_device)

    rows = []
    for i, path in enumerate(paths):
        name = os.path.basename(path)
        try:
            qc = qasm_load(path)
        except Exception as e:
            print(f"  [{i}] {name}: 加载失败 {e}")
            continue
        phys, nsw, _layout = sabre_route_full(qc, config, swap_trials=5,
                                              seed=i)
        res = {}
        env = make_sched_env(phys, hw, cm, max_edges)
        done, _ = run_sched_episode(env, "asap")
        f_asap = float(fid_main(env)) if done else None
        f_rep = float(fid_rep(env)) if done else None
        res["asap"] = f_asap
        sigma = (abs(f_asap - f_rep) / np.sqrt(2)) if done else None
        ms_asap, _, _ = sched_episode_stats(env)

        fids = []
        if done:
            env2 = make_sched_env(phys, hw, cm, max_edges)
            done2, _ = run_sched_episode(env2, "anti")
            if done2:
                res["anti"] = float(fid_main(env2))
                fids.append(res["anti"])
            rng = np.random.default_rng(7)
            r_done = 0
            for k in range(args.n_random):
                envr = make_sched_env(phys, hw, cm, max_edges)
                dk, _ = run_sched_episode(envr, "random", rng=rng)
                if dk:
                    fids.append(float(fid_main(envr)))
                    r_done += 1
            res["random_mean"] = float(np.mean(fids)) if fids else None
            allv = [v for v in [res.get("asap"), res.get("anti")]
                    + ([v for v in fids] if fids else []) if v is not None]
            res["spread"] = float(max(allv) - min(allv)) if allv else None
        res.update({"circuit": name, "sabre_swaps": nsw, "sigma_est": sigma,
                    "makespan_asap": round(ms_asap, 2)})
        rows.append(res)
        sp = res.get("spread")
        print(f"  [{i}] {name:36s} sw={nsw:3d} asap={f_asap if f_asap is None else round(f_asap,4)} "
              f"anti={res.get('anti') if res.get('anti') is None else round(res['anti'],4)} "
              f"rand={res.get('random_mean') if res.get('random_mean') is None else round(res['random_mean'],4)} "
              f"spread={sp if sp is None else round(sp,4)} σ≈{sigma if sigma is None else round(sigma,4)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)

    ok = [r for r in rows if r.get("spread") is not None]
    if ok:
        spreads = np.array([r["spread"] for r in ok])
        sigmas = np.array([r["sigma_est"] for r in ok if r["sigma_est"] is not None])
        print("\n=== 探针 0a 汇总 ===")
        print(f"circuits={len(ok)}")
        print(f"spread:      median={np.median(spreads):.4f} "
              f"mean={spreads.mean():.4f} p90={np.percentile(spreads, 90):.4f} "
              f"max={spreads.max():.4f}")
        if sigmas.size:
            print(f"σ(sim noise): median={np.median(sigmas):.4f} mean={sigmas.mean():.4f}")
            ratio = spreads / (sigmas.mean() + 1e-9)
            print(f"spread/σ:    median={np.median(ratio):.1f} "
                  f"frac(>2σ)={float((ratio > 2).mean()):.0%}")
        print(f"写入 {args.out}")


if __name__ == "__main__":
    main()
