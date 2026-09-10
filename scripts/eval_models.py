"""统一评估：多个模型在 test split 上的 fidelity 对比（log-mean + 配对胜率）。

用法: PYTHONPATH=src python3 scripts/eval_models.py --models l05,p2a,p2b,p2c ...
每个模型可配一个 checkpoint 路径。输出 per-circuit 行 + 汇总。
"""
import argparse, os, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from routing.rl.eval_policy import load_topo, evaluate_circuit, load_qc
from routing.graph.circuit_dag import CircuitDAG
from routing.gnn.encoder import SubGNN
from routing.rl.agent import PPOAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True,
                    help="name:ckpt 逗号分隔，如 l05:models/policy_x.pt,p2b:models/policy_p2b_ema_best.pt")
    ap.add_argument("--split", required=True, help="如 large_n10_test")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--traj", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--topo", default="traindata/topo/tianyan176_20q.json")
    args = ap.parse_args()

    config, hw, cm = load_topo(args.topo)
    specs = [(p.split(":", 1)[0], p.split(":", 1)[1])
             for p in args.models.split(",")]

    agents = {}
    for name, ckpt in specs:
        gnn = SubGNN(subgraph='full'); gnn.eval()
        a = PPOAgent(obs_dim=1, action_dim=len(cm) + 1, device='cpu', gnn=gnn,
                     num_qubits=20, num_edges=len(cm), coupling_map=cm,
                     with_commit=True)
        a.load(ckpt)
        a.ac.eval()
        a.gnn.eval()
        agents[name] = a
        print(f"[load] {name}: {ckpt}", flush=True)

    with open(os.path.join(ROOT, 'traindata', 'splits', args.split + '.txt')) as f:
        paths = [l.strip() for l in f if l.strip()]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(paths), size=min(args.n, len(paths)), replace=False)
    paths = [paths[i] for i in sorted(idx)]

    agg = {k: [] for k in agents}
    wins = {k: {o: 0 for o in agents if o != k} for k in agents}
    n_pairs = 0
    for cp in paths:
        qc = load_qc(os.path.join(ROOT, 'traindata'), cp)
        dag = CircuitDAG.from_circuit(qc)
        line = f"{cp.split('/')[-1]:<28} ({dag.num_logical_qubits}q,{dag.num_gates:>3}g)"
        for name, a in agents.items():
            m = evaluate_circuit(dag, hw, cm, a, reward_mode='routing',
                                 max_episode_steps=400, deterministic=True,
                                 seed=0, max_num_qubits=20, use_scheduler=True,
                                 config=config, fidelity_sim='trajectory_sched',
                                 num_trajectories=args.traj)
            f = m.fidelity if m.fidelity is not None else 0.0
            agg[name].append(f)
            line += f" | {name}:{f:.4f}"
        print(line, flush=True)
        for k in agents:
            for o in agents:
                if o != k and agg[k][-1] > agg[o][-1]:
                    wins[k][o] += 1
        n_pairs += 1

    print(f"\n=== 汇总 ({args.split}, {n_pairs} circuits, traj={args.traj}) ===")
    for name in agents:
        fs = np.array([max(f, 1e-9) for f in agg[name]])
        print(f"  {name}: mean={fs.mean():.5f} log-mean={np.exp(np.log(fs).mean()):.6f} "
              f"median={np.median(fs):.5f}")
    for k in agents:
        for o in agents:
            if o != k:
                print(f"  {k} 胜 {o}: {wins[k][o]}/{n_pairs}")


if __name__ == "__main__":
    main()
