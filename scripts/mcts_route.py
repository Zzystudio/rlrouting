#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCTS 联合搜索式路由推理入口（doc/20260922训练方案.md Phase A）。

纯搜索 + 启发式：先验用 sabre_score/criticality/锁调制，叶子用确定性 rollout
（mimic+ASAP-EXEC+锁等待），无 NN/GNN 依赖。对比 SABRE 基线（SabreLayout）
与 mimic-only（确定性策略无搜索）。指标：SWAPs / makespan / v3 fidelity。

用法（项目根）：PYTHONPATH=src python3 scripts/mcts_route.py \
  --split /tmp/opencode/ind50_a.txt --mcts-sims 128 --out /tmp/opencode/mcts_a.json
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing.graph.circuit_dag import CircuitDAG
from routing.rl.env_clocked import ClockedRoutingEnv
from routing.rl.eval_policy import load_topo, build_fidelity_fn
from routing.rl.mcts import (MCTSConfig, mcts_decide, deterministic_policy_step)
from scripts.eval_clocked import run_sabre, qasm_load, load_split_circuits


def make_env(dag, hw, cm, max_edges, layout):
    n = dag.num_logical_qubits
    init = (list(layout[:n]) if layout and len(layout) >= n
            else list(range(n)))
    return ClockedRoutingEnv(
        dag, hw, cm, reward_mode="routing", reward_potential=True,
        pot_progress_b=0.2, swap_price_scale=4.6,
        mapping_phase=False, init_mapping=init,
        max_ready=24, max_num_edges=max_edges, max_num_qubits=20,
        max_episode_steps=3000, step_cap_factor=2.0, use_gnn=False,
        lookahead_features=True, edge_noise_features=True, beta_noise=0.5,
        shaping_gamma=0.99, eta_shape=0.3, alpha_ext=0.5)


def run_episode(env, decide_fn, max_steps=4000):
    env.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        a = decide_fn(env)
        try:
            _, _, done, trunc, _ = env.step(a)
        except RuntimeError:
            done = True
        done = done or trunc
        steps += 1
    return env._swap_counter, env.clock, done, steps


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default="traindata/topo/tianyan287_20q.json")
    ap.add_argument("--data-dir", default="traindata")
    ap.add_argument("--split", required=True)
    ap.add_argument("--sim-device", default="cuda:0")
    ap.add_argument("--traj-trajectories", type=int, default=16)
    ap.add_argument("--out", default="/tmp/opencode/mcts.json")
    ap.add_argument("--limit", type=int, default=0)
    # MCTS 超参
    ap.add_argument("--mcts-sims", type=int, default=128)
    ap.add_argument("--mcts-c-puct", type=float, default=2.0)
    ap.add_argument("--mcts-alpha", type=float, default=6.0)
    ap.add_argument("--mcts-delta", type=float, default=0.08)
    ap.add_argument("--mcts-max-depth", type=int, default=40)
    ap.add_argument("--mcts-backup", default="avg", choices=["max", "avg"])
    ap.add_argument("--mcts-rollout-mode", default="cap",
                    choices=["cap", "full"],
                    help="cap=短 rollout+余项（快）；full=完整贪心 rollout（准，贵）")
    ap.add_argument("--sabre-trials", type=int, default=5)
    args = ap.parse_args()

    config, hw, cm = load_topo(args.topo)
    max_edges = max(19, len(cm))
    fid_fn = build_fidelity_fn("trajectory_v3", config,
                               num_trajectories=args.traj_trajectories,
                               backend=args.sim_device)
    circuits = load_split_circuits(args.data_dir, args.split)
    if args.limit > 0:
        circuits = circuits[:args.limit]

    rows = []
    for i, path in enumerate(circuits):
        qc = qasm_load(path)
        dag = CircuitDAG.from_circuit(qc)
        phys, nsw, ms_s, par_s, layout = run_sabre(
            qc, config, hw, swap_trials=args.sabre_trials, seed=i)
        # MCTS 策略
        env = make_env(dag, hw, cm, max_edges, layout)
        cfg = MCTSConfig(sims=args.mcts_sims, c_puct=args.mcts_c_puct,
                         swap_alpha=args.mcts_alpha,
                         trigger_delta=args.mcts_delta,
                         max_depth=args.mcts_max_depth,
                         backup=args.mcts_backup,
                         rollout_mode=args.mcts_rollout_mode, seed=i)
        sw_mcts, ms_mcts, done_m, _ = run_episode(env, lambda e: mcts_decide(e, cfg)[0])
        fid_m = fid_fn(env) if done_m else None
        # mimic-only（无搜索对照）
        env2 = make_env(dag, hw, cm, max_edges, layout)
        sw_mimic, ms_mimic, done_mi, _ = run_episode(env2, deterministic_policy_step)
        name = os.path.basename(path)
        rows.append({"circuit": name, "sabre_swaps": nsw, "sabre_ms": ms_s,
                     "mcts_swaps": sw_mcts, "mcts_ms": ms_mcts,
                     "mcts_done": done_m, "mcts_fid": fid_m,
                     "mimic_swaps": sw_mimic, "mimic_ms": ms_mimic,
                     "mimic_done": done_mi,
                     "n_searched": cfg.n_searched,
                     "total_sims": cfg.n_total_sims})
        print(f"{name:32s} SABRE sw={nsw:4d} | mimic sw={sw_mimic:5d}{'' if done_mi else 'T'} "
              f"| MCTS sw={sw_mcts:5d}{'' if done_m else 'T'} "
              f"ms={ms_mcts:6.1f} fid={fid_m if fid_m is None else round(fid_m,3)} "
              f"searched={cfg.n_searched}")

    import json
    json.dump(rows, open(args.out, "w"), indent=2)
    # 汇总
    ok = [r for r in rows if r["mcts_done"] and r["mimic_done"]]
    if ok:
        n = len(ok)
        for col in ("sabre_swaps", "mimic_swaps", "mcts_swaps"):
            print(f"{col:14s} mean={sum(r[col] for r in ok)/n:7.1f}")
        print(f"mcts/mimic = {sum(r['mcts_swaps'] for r in ok)/sum(r['mimic_swaps'] for r in ok):.3f}")
        print(f"mcts/sabre = {sum(r['mcts_swaps'] for r in ok)/sum(r['sabre_swaps'] for r in ok):.3f}")
        print(f"mimic/sabre = {sum(r['mimic_swaps'] for r in ok)/sum(r['sabre_swaps'] for r in ok):.3f}")
    print(f"写入 {args.out}")


if __name__ == "__main__":
    main()
