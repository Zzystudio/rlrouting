#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""决定性实验：单 trial SABRE vs best-of-5 SABRE vs mimic（in-dist 50）。

隔离"随机重启(trial)"的贡献：
- sabre1 = Qiskit SabreSwap trials=1（单次贪心，随机 tie-break）
- sabre5 = trials=5（best-of-N）
- mimic  = 0b 诊断口径（旧 env + SABRE 布局 + argmin sabre_core）

若 sabre1 ≈ mimic → 差距主要来自 trial 重启机制（学习永远弥合不了，
该嵌入真 SABRE）；若 sabre1 ≪ mimic → 差距在执行语义/特征残差（可修特征）。
用法（项目根）：PYTHONPATH=src python3 scripts/compare_sabre_trials.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import qasm2
from routing.graph.circuit_dag import CircuitDAG
from routing.rl.env import RoutingEnv
from routing.rl.eval_policy import load_topo
from routing.routing import sabre_route

TOPO = "traindata/topo/tianyan287_20q.json"
INDIST = [f"benchmark/indist_val/{f}" for f in sorted(os.listdir(
    "benchmark/indist_val")) if f.endswith(".qasm")]


def mimic_env(dag, hw, cm, layout):
    return RoutingEnv(dag, hw, cm, reward_mode="routing", reward_potential=True,
                      pot_progress_b=0.2, swap_price_scale=4.6, use_gnn=False,
                      mapping_phase=False, init_mapping=layout, random_init=False,
                      max_num_edges=max(len(cm), 20), max_num_qubits=20,
                      edge_noise_features=True, beta_noise=0.5,
                      lookahead_features=True, shaping_gamma=0.99,
                      max_episode_steps=3000, step_cap_mult=2.0)


def mimic_swaps(dag, hw, cm, layout, max_steps=3000):
    env = mimic_env(dag, hw, cm, layout)
    env.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        sc = env._edge_sabre_core_features()
        unmapped = env.get_unmapped_mask()
        dl = env.get_deadlock_mask()
        score = np.where(~unmapped & ~dl, sc[:, 0], np.inf)
        if not np.isfinite(score).any():
            score = np.where(~unmapped, sc[:, 0], np.inf)
        env.step(int(np.argmin(score)))
        done = len(env.executed) == env.dag.num_gates
        steps += 1
    return env._swap_counter, done


def main():
    config, hw, cm = load_topo(TOPO)
    rows = []
    for path in INDIST:
        qc = qasm2.load(path)
        dag = CircuitDAG.from_circuit(qc)
        n = dag.num_logical_qubits
        _, info5 = sabre_route(qc, config, swap_trials=5, seed=0)
        _, info1 = sabre_route(qc, config, swap_trials=1, seed=0)
        layout = info5.get("initial_layout")
        layout = (list(layout[:n]) if layout and len(layout) >= n
                  else list(range(n)))
        msw, done = mimic_swaps(dag, hw, cm, layout)
        name = path.split("/")[-1]
        rows.append({"circuit": name, "n": n,
                     "sabre1": info1["num_swaps"], "sabre5": info5["num_swaps"],
                     "mimic": msw, "mimic_done": done})
        print(f"{name:34s} n={n:2d} | sabre1={info1['num_swaps']:5d} "
              f"sabre5={info5['num_swaps']:5d} mimic={msw:5d}"
              f"({'ok' if done else 'TRUNC'})")

    import json
    json.dump(rows, open("/tmp/opencode/compare_sabre_trials.json", "w"),
              indent=2)
    print("\n=== 汇总（完成电路）===")
    for col in ("sabre1", "sabre5", "mimic"):
        vals = [r[col] for r in rows if r["mimic_done"]]
        print(f"{col:7s} mean={sum(vals)/len(vals):7.1f}")
    ok = [r for r in rows if r["mimic_done"]]
    print(f"\nratio: sabre1/mimic = "
          f"{sum(r['sabre1'] for r in ok)/sum(r['mimic'] for r in ok):.3f}")
    print(f"       sabre5/mimic = "
          f"{sum(r['sabre5'] for r in ok)/sum(r['mimic'] for r in ok):.3f}")
    print(f"       sabre5/sabre1 = "
          f"{sum(r['sabre5'] for r in ok)/sum(r['sabre1'] for r in ok):.3f}")


if __name__ == "__main__":
    main()
