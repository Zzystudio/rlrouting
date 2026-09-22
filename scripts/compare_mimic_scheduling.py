#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公平对比：C0 头 + mimic vs C2m（mimic + 重训调度 + 噪声价）。

消除跨 run 的 SABRE 基线采样方差：每个电路 SABRE 只路由+保真度一次（固定
seed、T=32），两个模型共享同一参照；模型保真度 3-seed 平均（T=16 each）。
用法（项目根）：PYTHONPATH=src python3 scripts/compare_mimic_scheduling.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing.graph.circuit_dag import CircuitDAG
from routing.gnn.encoder import SubGNN
from routing.rl.env_clocked import ClockedRoutingEnv
from routing.rl.agent_clocked import ClockedPPOAgent, D_EXEC, D_TIMING_GLOB
from routing.rl.eval_policy import load_topo, load_qc, build_fidelity_fn
from routing.routing import sabre_route
from sim.trajectory_sim_v3 import trajectory_circuit_fidelity_events_v3

TOPO = "traindata/topo/tianyan287_20q.json"
SPLIT = "traindata/splits/tianyan20q_test.txt"
MODELS = {
    "C0+mimic": "models/policy_clocked_c0.pt",
    "C3v2": "models/policy_clocked_c3v2.pt",
}


def build_agent(path, gnn, max_edges, device):
    a = ClockedPPOAgent(1, 1, num_qubits=20, num_edges=max_edges, max_ready=24,
                        edge_feat_dim=279, exec_feat_dim=D_EXEC,
                        timing_glob_dim=D_TIMING_GLOB, device=device, gnn=gnn,
                        edge_hidden=64)
    a.load_checkpoint(path)
    a.ac.eval()
    gnn.eval()
    return a


def make_env(dag, hw, cm, max_edges, gnn):
    return ClockedRoutingEnv(
        dag, hw, cm, reward_mode="routing", reward_potential=True,
        pot_progress_b=0.2, swap_price_scale=4.6, mapping_phase=True,
        max_ready=24, max_num_edges=max_edges, max_num_qubits=20,
        mapping_budget=8, use_gnn=True, gnn=gnn, max_episode_steps=1000,
        lookahead_features=True, edge_noise_features=True, beta_noise=0.5,
        shaping_gamma=0.99, eta_shape=0.3, alpha_ext=0.5, step_cap_factor=2.0)


def run_one(agent, env, fid_fn, n_seed=3):
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 4000:
        mask = env.get_action_mask()
        e = env.mimic_swap_index()
        mask[:env.num_edges] = False
        mask[e] = True
        a, *_ = agent.act(obs, deterministic=True, action_mask=mask)
        obs, _, done, truncated, _ = env.step(a)
        done = done or truncated
        steps += 1
    if not done:
        return None, None
    fids = [fid_fn(env) for _ in range(n_seed)]
    return env._swap_counter, float(np.mean(fids))


def main():
    config, hw, cm = load_topo(TOPO)
    max_edges = max(19, len(cm))
    agents = {k: build_agent(v, SubGNN(subgraph="full"), max_edges, "cuda:0")
              for k, v in MODELS.items()}
    circs = [l.strip() for l in open(SPLIT) if l.strip()]
    rows = []
    for i, rel in enumerate(circs):
        qc = load_qc("traindata", rel, seed=0)
        dag = CircuitDAG.from_circuit(qc)
        phys, info = sabre_route(qc, config, swap_trials=5, seed=i)
        s_swaps = info["num_swaps"]
        s_fid = float(trajectory_circuit_fidelity_events_v3(
            phys, config, num_trajectories=32, seed=i + 100, backend="cuda:0"))
        rec = {"circuit": rel, "sabre_swaps": s_swaps, "sabre_fid": s_fid}
        for name, ag in agents.items():
            env = make_env(dag, hw, cm, max_edges, ag.gnn)
            fid_fn = build_fidelity_fn("trajectory_v3", config,
                                       num_trajectories=16,
                                       backend="cuda:0")
            sw, fid = run_one(ag, env, fid_fn, n_seed=3)
            rec[f"{name}_swaps"] = sw
            rec[f"{name}_fid"] = fid
            print(f"[{i}] {rel.split('/')[-1]:28s} SABRE {s_swaps:3d} "
                  f"fid={s_fid:.4f} | {name}: sw={sw} fid={fid and round(fid,4)}")
        rows.append(rec)

    print("\n=== 汇总（30 条，同 SABRE 参照）===")
    print(f"{'model':14s} {'swaps':>7s} {'fid':>8s} {'Δfid vs SABRE':>14s}")
    for name in MODELS:
        sw = [r[f"{name}_swaps"] for r in rows]
        fd = [r[f"{name}_fid"] for r in rows]
        sf = [r["sabre_fid"] for r in rows]
        sw_m = np.mean(sw)
        fd_m = np.mean(fd)
        sf_m = np.mean(sf)
        d = np.mean([a - b for a, b in zip(fd, sf)])
        sw_sabre = np.mean([r["sabre_swaps"] for r in rows])
        print(f"{name:14s} {sw_m:7.1f} {fd_m:8.4f} {d:14.4f} "
              f"(SABRE sw={sw_sabre:.1f} fid={sf_m:.4f})")


if __name__ == "__main__":
    main()
