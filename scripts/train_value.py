#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1+2：相对价值（advantage）+ 不确定性 ensemble + DAgger 搜索感知数据。

- Stage 1：训练 AdvantageEnsemble（K 个 MLP 预测 A(s,a)=J_guard−J_a，μ/σ），
  MCTS Q = rollout锚 + λ(σ)·(μ_A − β'·σ_A)；σ 触发；守卫动作恒在候选
- Stage 2：DAgger——MCTS 实际访问（触发/高 σ）状态用 oracle rollout 标注
  advantage，并入训练集重训，迭代覆盖搜索分布

用法（项目根）：PYTHONPATH=src python3 scripts/train_value.py \
  --circuits /tmp/opencode/mcts_probe.txt --iters 4 --sims 16
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing.graph.circuit_dag import CircuitDAG
from routing.rl.env_clocked import ClockedRoutingEnv
from routing.rl.eval_policy import load_topo
from routing.rl.mcts import (MCTSConfig, mcts_decide, deterministic_policy_step,
                             is_done)
from routing.rl.value_net import (AdvantageEnsemble, collect_advantage_data,
                                  train_ensemble, make_sabre_expert_policy)
from scripts.eval_clocked import run_sabre, qasm_load, load_split_circuits


def make_env_builder(hw, cm, max_edges):
    def builder(dag, layout):
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
    return builder


def _mcts_policy(env, cfg):
    return mcts_decide(env, cfg)[0]


def eval_gate(env_builder, circuits, cfg, label):
    """MCTS(+当前 cfg) vs mimic 的 swap 对比（评估门控）。"""
    mcts_sw, mimic_sw = [], []
    for c in circuits:
        dag, layout = c[0], c[1]
        env = env_builder(dag, layout)
        env.reset()
        while not is_done(env):
            env.step(mcts_decide(env, cfg)[0])
        mcts_sw.append(env._swap_counter)
        env2 = env_builder(dag, layout)
        env2.reset()
        while not is_done(env2):
            env2.step(deterministic_policy_step(env2))
        mimic_sw.append(env2._swap_counter)
    m, mm = float(np.mean(mcts_sw)), float(np.mean(mimic_sw))
    print(f"  [{label}] MCTS sw={m:.1f}  mimic sw={mm:.1f}  ratio={m/mm:.3f}")
    return m, mm


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default="traindata/topo/tianyan287_20q.json")
    ap.add_argument("--circuits", required=True)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--mcts-alpha", type=float, default=2.0)
    ap.add_argument("--mcts-backup", default="avg", choices=["max", "avg"])
    ap.add_argument("--rollout-mode", default="cap", choices=["cap", "full"])
    ap.add_argument("--ensemble-k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--top-k-adv", type=int, default=5,
                    help="advantage 数据每状态标注的动作数")
    ap.add_argument("--lambda0", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--beta-pess", type=float, default=0.5)
    ap.add_argument("--sigma-trigger", type=float, default=None)
    ap.add_argument("--out", default="/tmp/opencode/adv_ens.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config, hw, cm = load_topo(args.topo)
    max_edges = max(19, len(cm))
    builder = make_env_builder(hw, cm, max_edges)
    circuits = []
    for path in load_split_circuits("", args.circuits):
        qc = qasm_load(path)
        dag = CircuitDAG.from_circuit(qc)
        _, _, _, _, layout = run_sabre(qc, config, hw, swap_trials=5, seed=0)
        circuits.append((dag, layout))
    print(f"[train_value] {len(circuits)} circuits, {args.iters} iters")

    ens = AdvantageEnsemble(K=args.ensemble_k, seed=args.seed)
    cfg = MCTSConfig(sims=args.sims, c_puct=2.0, max_depth=15,
                     swap_alpha=args.mcts_alpha, backup=args.mcts_backup,
                     rollout_mode=args.rollout_mode, seed=args.seed)
    cfg.lambda0 = args.lambda0
    cfg.beta = args.beta
    cfg.beta_pess = args.beta_pess
    cfg.sigma_trigger = args.sigma_trigger
    best = None
    all_data = []
    for it in range(args.iters):
        print(f"=== iter {it} ===")
        # Stage 1 数据：专家（守卫）轨迹状态
        d1 = collect_advantage_data(builder, circuits, cfg,
                                    top_k=args.top_k_adv,
                                    stride=2, max_states=200,
                                    seed=args.seed + it)
        print(f"  专家 advantage 数据: {len(d1)} 对")
        # Stage 2（DAgger）：MCTS 访问状态的 oracle 标注
        cfg.adv_fn = ens.make_adv_fn()
        d2 = collect_advantage_data(builder, circuits, cfg,
                                    top_k=args.top_k_adv,
                                    policy=_mcts_policy,
                                    stride=4, max_states=60,
                                    max_traj_steps=250,
                                    seed=args.seed + it)
        print(f"  DAgger(MCTS 访问) 数据: {len(d2)} 对")
        all_data = (d1 + d2) if it == 0 else (all_data + d1 + d2)
        # 控制数据量（防无限膨胀）
        if len(all_data) > 20000:
            all_data = all_data[-20000:]
        train_ensemble(ens, all_data, epochs=args.epochs, device="cpu")
        cfg.adv_fn = ens.make_adv_fn()
        m, mm = eval_gate(builder, circuits, cfg, f"iter{it}")
        if best is None or m <= best[0]:
            best = (m, it)
            torch.save({"members": [m.state_dict() for m in ens.members]},
                       args.out)
            print(f"  -> saved {args.out} (best MCTS sw={m:.1f})")
    print(f"best: iter {best[1]}, MCTS sw={best[0]:.1f}")


if __name__ == "__main__":
    main()
