#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""时钟化模型评估入口（doc/20260920训练方案.md §八）。

用法（在 src/ 下）：
  PYTHONPATH=. python3 scripts/eval_clocked.py \
    --model ../models/policy_clocked_c0.pt \
    --topo ../traindata/topo/tianyan287_20q.json \
    --data-dir ../traindata --split tianyan20q_test \
    --device cuda:1 --out /tmp/eval_clocked.json

指标五元组：SWAPs / makespan(us) / par_density / theta_total / fidelity(v3)。
SABRE 基线配 D2 schedule_routed_circuit 同调度器同 v3 模拟器。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit

from routing.graph.circuit_dag import CircuitDAG
from routing.gnn.encoder import SubGNN
from routing.rl.env_clocked import ClockedRoutingEnv
from routing.rl.agent_clocked import ClockedPPOAgent, D_EXEC, D_TIMING_GLOB
from routing.rl.eval_policy import load_topo, build_fidelity_fn
from routing.routing import sabre_route
from routing.timing import schedule_routed_circuit, GreedyScheduler


def load_split_circuits(data_dir, split_name):
    sp = os.path.join(data_dir, "splits", f"{split_name}.txt")
    if not os.path.exists(sp):
        sp = split_name
    paths = []
    with open(sp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if os.path.isabs(line) and os.path.exists(line):
                paths.append(line)
            else:
                cand = os.path.join(data_dir, line)
                if os.path.exists(cand):
                    paths.append(cand)
    return paths


def run_policy(agent, env, deterministic=True, beam_width=0, routing_mimic=False,
               no_guards=False, tie_eps=0.0, tie_seed=0):
    obs, _ = env.reset()
    E = env.num_edges
    K = env.max_ready
    done = False
    steps = 0
    no_progress_swaps = 0
    exec_mark = len(env.executed)
    tie_rng = np.random.default_rng(tie_seed) if tie_eps > 0 else None
    while not done and steps < 4000:
        try:
            mask = env.get_action_mask()
        except RuntimeError:
            # tie-break/探索可能把 env 推进 liveness 失败状态：该 trial 作废
            return None
        if routing_mimic and not env.mapping_phase:
            if not no_guards:
                legal_exec = [i for i in range(env.num_edges,
                                               env.num_edges + env.max_ready)
                              if mask[i]]
                if legal_exec:
                    a = legal_exec[0]
                    no_progress_swaps = 0
                else:
                    ready_adj = env._ready_2q_adjacent()
                    if ready_adj and mask[env.skip_action]:
                        a = env.skip_action
                        no_progress_swaps = 0
                    else:
                        e = env.mimic_swap_index(tie_eps=tie_eps, rng=tie_rng)
                        mask[:env.num_edges] = False
                        mask[e] = True
                        if beam_width > 1:
                            a, *_ = agent.act(obs, deterministic=True,
                                              action_mask=mask)
                        else:
                            a, *_ = agent.act(obs, deterministic=deterministic,
                                              action_mask=mask)
            else:
                e = env.mimic_swap_index(tie_eps=tie_eps, rng=tie_rng)
                mask[:env.num_edges] = False
                mask[e] = True
                if beam_width > 1:
                    a, *_ = agent.act(obs, deterministic=True,
                                      action_mask=mask)
                else:
                    a, *_ = agent.act(obs, deterministic=deterministic,
                                      action_mask=mask)
        else:
            if beam_width > 1:
                a, *_ = agent.act(obs, deterministic=True, action_mask=mask)
            else:
                a, *_ = agent.act(obs, deterministic=deterministic,
                                  action_mask=mask)
        # 无进展保护：路由期连续 N 次 swap 未执行任何门 → 强制 SKIP 推进时钟，
        # 打破 mimic 贪心换边循环（周期>死锁 max_cycle 的病理，2026-09-22）
        if routing_mimic and not env.mapping_phase and a < env.num_edges:
            if len(env.executed) <= exec_mark:
                no_progress_swaps += 1
            else:
                no_progress_swaps = 0
            exec_mark = len(env.executed)
            if no_progress_swaps >= 6:
                a = env.skip_action
                no_progress_swaps = 0
        obs, _, done, truncated, _ = env.step(a)
        done = done or truncated
        steps += 1
    if not done:
        return None
    makespan = env.clock
    serial = float(getattr(env.timing, "serial_dur", 0.0))
    par = serial / makespan if makespan > 1e-9 else 0.0
    return {
        "swaps": env._swap_counter,
        "makespan_us": round(makespan, 3),
        "par_density": round(par, 3),
        "theta_total": round(env._cum_theta, 5),
        "truncated": False,
    }


def run_sabre(qc, config, hw, swap_trials=5, seed=0):
    """完整 SABRE 基线：SabreLayout（布局）+ SabreSwap（路由）+ D2 ASAP 调度。

    2026-09-22 升级：旧 sabre_route 只跑 SabreSwap 从平凡布局出发（无布局
    pass），n<20 电路 swap 多 ~44%（实测）。用 PassManager([SabreLayout,
    SabreSwap]) 只加布局、**不分解门**（transpile 会把门转成 u3 等，与
    v3 时长表不匹配、污染保真度对比）。返回
    (phys, n_swaps, makespan, density, init_layout)。
    """
    from qiskit import QuantumCircuit
    from qiskit.transpiler import PassManager, CouplingMap
    from qiskit.transpiler.passes import SabreLayout, SabreSwap
    cmap = CouplingMap(list(config.coupling_map))
    c = qc
    if qc.num_qubits < cmap.size():
        c = QuantumCircuit(cmap.size(), qc.num_clbits)
        c.compose(qc, inplace=True)
    pm = PassManager([
        SabreLayout(coupling_map=cmap, seed=seed),
        SabreSwap(coupling_map=cmap, heuristic="decay", trials=swap_trials,
                  seed=seed),
    ])
    phys = pm.run(c)
    n_swaps = sum(1 for inst in phys.data if inst.operation.name == "swap")
    dag = CircuitDAG.from_circuit(phys)
    sched = GreedyScheduler()
    makespan, _, stats = schedule_routed_circuit(dag, hw, scheduler=sched)
    # SabreLayout 输出的初始布局（property_set['layout']）
    init_layout = None
    layout = pm.property_set.get("layout")
    if layout is not None:
        try:
            vb = layout.get_virtual_bits()
            n = qc.num_qubits
            init_layout = [int(vb[qc.qubits[i]].index) for i in range(n)]
        except Exception:
            init_layout = None
    return phys, n_swaps, makespan, stats["density"], init_layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--topo", required=True)
    ap.add_argument("--data-dir", default="traindata")
    ap.add_argument("--split", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="/tmp/eval_clocked.json")
    ap.add_argument("--beam-width", type=int, default=0)
    ap.add_argument("--routing-mimic", action="store_true", default=False,
                    help="路由头用特征驱动 SABRE 模仿（argmin sabre_core）")
    ap.add_argument("--no-eval-guards", action="store_true", default=False,
                    help="关闭 mimic 模式的 EXEC 优先 + 锁等待守卫（观察模型"
                         "学到的原始调度行为，实验用）")
    ap.add_argument("--mimic-tie-eps", type=float, default=0.0,
                    help="mimic 路由 ε-tie-break：min+ε 分数窗口内随机选边"
                         "（近似 SABRE 随机 tie-break，配合 --mimic-trials）")
    ap.add_argument("--mimic-trials", type=int, default=1,
                    help="mimic best-of-N：跑 N 次（不同 tie 种子）取 swap 最少")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-ready-2q", type=int, default=24)
    ap.add_argument("--max-num-qubits", type=int, default=20)
    ap.add_argument("--mapping-budget", type=int, default=8)
    ap.add_argument("--swap-price-scale", type=float, default=4.6)
    ap.add_argument("--pot-progress-b", type=float, default=0.2)
    ap.add_argument("--eta-time", type=float, default=0.05)
    ap.add_argument("--eta-parallel", type=float, default=0.05)
    ap.add_argument("--eta-idle", type=float, default=0.005)
    ap.add_argument("--w-xt-launch", type=float, default=0.0)
    ap.add_argument("--w-zz", type=float, default=0.0)
    ap.add_argument("--sabre-trials", type=int, default=5)
    ap.add_argument("--fidelity-sim", default="trajectory_v3",
                    choices=["trajectory", "trajectory_v2", "trajectory_v3"])
    ap.add_argument("--traj-trajectories", type=int, default=16)
    ap.add_argument("--sim-device", default="auto")
    ap.add_argument("--edge-hidden", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    config, hw, cm = load_topo(args.topo)
    max_edges = max(args.max_num_qubits - 1, len(cm))
    fid_fn = build_fidelity_fn(
        args.fidelity_sim, config, num_trajectories=args.traj_trajectories,
        backend=args.sim_device)

    gnn = SubGNN(subgraph="full")
    agent = ClockedPPOAgent(
        obs_dim=1, action_dim=1, num_qubits=args.max_num_qubits,
        num_edges=max_edges, max_ready=args.max_ready_2q,
        edge_feat_dim=267 + 12,  # 267 旧 + D_GLOBAL_TIMING + D_EDGE_TIMING
        exec_feat_dim=D_EXEC, timing_glob_dim=D_TIMING_GLOB,
        device=args.device, gnn=gnn, edge_hidden=args.edge_hidden)
    state = agent.load_checkpoint(args.model)
    agent.ac.eval()
    gnn.eval()
    print(f"[eval] loaded {args.model} (step={state.get('step', '?')})")

    circuits = load_split_circuits(args.data_dir, args.split)
    if args.limit > 0:
        circuits = circuits[:args.limit]
    print(f"[eval] {len(circuits)} circuits on {args.topo.split('/')[-1]}")

    rows = []
    sabre_rows = []
    for i, path in enumerate(circuits):
        qc = qasm_load(path)
        dag = CircuitDAG.from_circuit(qc)
        # SABRE 基线先行：mimic 模式用其初始布局（布局是 SABRE 能力的一部分，
        # mimic-mapping 生成布局在结构化电路上差——compare_sabre_trials 证实）
        phys, nsw, makespan_s, par_s, sabre_layout = run_sabre(
            qc, config, hw, swap_trials=args.sabre_trials, seed=i)
        env_kw = dict(
            dag=dag, hw=hw, coupling_map=cm, reward_mode="routing",
            reward_potential=True,
            pot_progress_b=args.pot_progress_b,
            swap_price_scale=args.swap_price_scale,
            max_ready=args.max_ready_2q,
            max_num_edges=max_edges, max_num_qubits=args.max_num_qubits,
            use_gnn=True, gnn=gnn,
            max_episode_steps=1000, eta_time=args.eta_time,
            eta_parallel=args.eta_parallel, eta_idle=args.eta_idle,
            w_xt_launch=args.w_xt_launch, w_zz=args.w_zz,
            lookahead_features=True, edge_noise_features=True, beta_noise=0.5,
            fidelity_fn=None, step_cap_factor=2.0)
        r = None
        best_env = None
        layout = None
        if args.routing_mimic:
            n = dag.num_logical_qubits
            layout = (list(sabre_layout[:n]) if sabre_layout
                      and len(sabre_layout) >= n else list(range(n)))
        n_trials = args.mimic_trials if (args.routing_mimic
                                         and args.mimic_trials > 1) else 1
        for k in range(n_trials):
            if args.routing_mimic:
                env = ClockedRoutingEnv(mapping_phase=False,
                                        init_mapping=layout,
                                        mapping_budget=args.mapping_budget,
                                        **env_kw)
            else:
                env = ClockedRoutingEnv(mapping_phase=True,
                                        mapping_budget=args.mapping_budget,
                                        **env_kw)
            rr = run_policy(agent, env, beam_width=args.beam_width,
                            routing_mimic=args.routing_mimic,
                            no_guards=args.no_eval_guards,
                            tie_eps=args.mimic_tie_eps,
                            tie_seed=args.seed + i * 100 + k)
            if rr is None:
                continue
            if r is None or rr["swaps"] < r["swaps"]:
                r = rr
                best_env = env
        name = os.path.basename(path)
        if r is None:
            print(f"  [{i}] {name}: FAILED (unterminated)")
            rows.append({"circuit": name, "ok": False})
            continue
        env = best_env
        fid = fid_fn(env) if fid_fn is not None else None
        r.update({"circuit": name, "ok": True, "fidelity": fid, "trials": n_trials})
        rows.append(r)
        # SABRE + D2 基线（同 v3 模拟器）
        sdag = CircuitDAG.from_circuit(phys)
        senv = ClockedRoutingEnv(
            sdag, hw, cm, reward_mode="routing", reward_potential=True,
            pot_progress_b=args.pot_progress_b, swap_price_scale=1.0,
            mapping_phase=False, max_ready=2, max_num_edges=max_edges,
            max_num_qubits=args.max_num_qubits, use_gnn=False,
            max_episode_steps=1000, fidelity_fn=None)
        # SABRE 物理线路已含 SWAP，用 v3 直接评 events 保真度
        from sim.trajectory_sim_v2 import schedule_phys_circuit_events
        from sim.trajectory_sim_v3 import trajectory_circuit_fidelity_events_v3
        s_fid = trajectory_circuit_fidelity_events_v3(
            phys, config, num_trajectories=args.traj_trajectories,
            backend=args.sim_device)
        sabre_rows.append({
            "circuit": name, "swaps": nsw, "makespan_us": round(makespan_s, 3),
            "par_density": round(par_s, 3), "fidelity": float(s_fid)})
        print(f"  [{i}] {name}: sw={r['swaps']} makespan={r['makespan_us']} "
              f"fid={fid:.4f} | SABRE sw={nsw} ms={makespan_s:.2f} fid={s_fid:.4f}")

    out = {"model": args.model, "topo": args.topo, "split": args.split,
           "policy": rows, "sabre_d2": sabre_rows,
           "summary": {
               "policy_swaps": float(np.mean([r["swaps"] for r in rows if r.get("ok")])),
               "policy_fid": float(np.mean([r["fidelity"] for r in rows if r.get("ok") and r.get("fidelity") is not None])),
               "sabre_swaps": float(np.mean([r["swaps"] for r in sabre_rows])),
               "sabre_fid": float(np.mean([r["fidelity"] for r in sabre_rows])),
           }}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n=== 汇总 ===")
    s = out["summary"]
    print(f"PPO : sw={s['policy_swaps']:.1f}  fid={s['policy_fid']:.4f}")
    print(f"SABRE: sw={s['sabre_swaps']:.1f}  fid={s['sabre_fid']:.4f}")
    print(f"结果写入 {args.out}")


def qasm_load(path):
    from qiskit import qasm2
    if path.endswith(".qasm") or path.endswith(".qas"):
        return qasm2.load(path)
    import pickle
    with open(path, "rb") as f:
        qc = pickle.load(f)
    if qc.num_parameters > 0:
        import numpy as _np
        rng = _np.random.default_rng(0)
        qc = qc.assign_parameters({p: rng.uniform(0, 2 * _np.pi)
                                   for p in qc.parameters})
    return qc


if __name__ == "__main__":
    main()
