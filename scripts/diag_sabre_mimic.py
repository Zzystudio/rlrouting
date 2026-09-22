#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""0b 特征天花板诊断：手工策略 = 每步选 sabre_core[0]（SABRE 完整打分，低=好）
最小的边。判断特征层能否复制 SABRE 的换手选择（零训练）。

两轮：
  A: SABRE 布局初始化（init_mapping = sabre_route 的 initial_layout），无映射期
     —— 纯换手复制能力测试（同布局，只比 swap 选择）
  B: 恒等布局，无映射期 —— 固定布局参照
对比 SABRE（同电路同布局的完整路由）。

用法（项目根）：PYTHONPATH=src python3 scripts/diag_sabre_mimic.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import qasm2

from routing.graph.circuit_dag import CircuitDAG
from routing.rl.env import RoutingEnv
from routing.rl.eval_policy import load_topo, load_qc
from routing.routing import sabre_route

TOPO = "traindata/topo/tianyan287_20q.json"
T287_SPLIT = "traindata/splits/tianyan20q_test.txt"
NAM_DIR = "benchmark/nam_circs"


def _env_for(dag, hw, cm, layout):
    return RoutingEnv(
        dag, hw, cm, reward_mode="routing", reward_potential=True,
        pot_progress_b=0.2, swap_price_scale=4.6, use_gnn=False,
        mapping_phase=False, init_mapping=layout, random_init=False,
        max_num_edges=max(len(cm), 20), max_num_qubits=20,
        edge_noise_features=True, beta_noise=0.5, lookahead_features=True,
        shaping_gamma=0.99, eta_shape=0.3, alpha_ext=0.5,
        max_episode_steps=1000, step_cap_mult=2.0)


def _front_neighborhood_edges(env):
    """SABRE/LightSABRE 候选边集：前层门物理 qubit 的 1-邻域内耦合边。

    qiskit docstring: "search for SWAPs is restricted to physical qubits in the
    neighborhood of those qubits involved in front_layer"。旧 argmin 全边可选，
    会在所有候选都使打分变差时选中"远端无关边"（打分不变看似低）→ 无进展
    游走（深串行电路最重）。此限制是 SABRE 语义的一部分。
    """
    adj = env.hw.adj
    front_q = set()
    for g in env._ready_2q_gates():
        front_q.add(env.mapping[g.qubits[0]])
        front_q.add(env.mapping[g.qubits[1]])
    if not front_q:
        return set(range(env.num_edges))
    nb = set(front_q)
    for q in list(front_q):
        for r in range(env.hw.num_qubits):
            if adj[q, r] > 0:
                nb.add(r)
    out = set()
    for i, (p, q) in enumerate(env.coupling_map):
        if p in nb or q in nb:
            out.add(i)
    if not out:
        out = set(range(env.num_edges))
    return out


def run_mimic(dag, hw, cm, layout, max_steps=2000, restrict=True):
    env = _env_for(dag, hw, cm, layout)
    env.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        sc = env._edge_sabre_core_features()   # (E, 6), col0=sabre_score 低=好
        unmapped = env.get_unmapped_mask()
        score = np.where(~unmapped, sc[:, 0], np.inf)
        if restrict:
            cand = _front_neighborhood_edges(env)
            for i in range(env.num_edges):
                if i not in cand:
                    score[i] = np.inf
        a = int(np.argmin(score))
        env.step(a)
        done = len(env.executed) == env.dag.num_gates
        steps += 1
    return env._swap_counter, done, steps


def main():
    config, hw, cm = load_topo(TOPO)
    circs = []
    with open(T287_SPLIT) as f:
        for line in f:
            rel = line.strip()
            if rel:
                circs.append(("t287", load_qc("traindata", rel, seed=0), rel))
    for fn in sorted(os.listdir(NAM_DIR)):
        if fn.endswith(".qasm"):
            qc = qasm2.load(os.path.join(NAM_DIR, fn))
            circs.append(("nam", qc, fn))

    rows = []
    for kind, qc, name in circs:
        dag = CircuitDAG.from_circuit(qc)
        phys, info = sabre_route(qc, config, swap_trials=5, seed=0)
        s_swaps = info["num_swaps"]
        sabre_layout = info.get("initial_layout")
        n = dag.num_logical_qubits
        if sabre_layout is None or len(sabre_layout) < n:
            sabre_layout = list(range(n))
        sabre_layout = list(sabre_layout[:n])
        swA, doneA, _ = run_mimic(dag, hw, cm, sabre_layout, restrict=True)
        swB, doneB, _ = run_mimic(dag, hw, cm, list(range(n)), restrict=False)
        rows.append({
            "circuit": name, "kind": kind, "n_logical": n,
            "sabre_swaps": s_swaps,
            "mimic_A_swaps": swA, "mimic_A_done": doneA,
            "mimic_B_swaps": swB, "mimic_B_done": doneB,
        })
        print(f"{name:22s} n={n:2d} | SABRE {s_swaps:5d} | "
              f"mimicA {swA:5d}({'ok' if doneA else 'TRUNC'}) | "
              f"mimicB {swB:5d}({'ok' if doneB else 'TRUNC'})")

    # 汇总
    for kind in ("t287", "nam"):
        rs = [r for r in rows if r["kind"] == kind and r["mimic_A_done"]
              and r["mimic_B_done"]]
        if rs:
            print(f"\n[{kind}] completed circuits: {len(rs)}")
            for col in ("sabre_swaps", "mimic_A_swaps", "mimic_B_swaps"):
                v = [r[col] for r in rs]
                print(f"  {col:16s} mean={sum(v)/len(v):6.1f}")
            a_ratio = sum(r["mimic_A_swaps"] for r in rs) / max(
                1, sum(r["sabre_swaps"] for r in rs))
            print(f"  mimicA/SABRE = {a_ratio:.3f}")
        else:
            print(f"\n[{kind}] no fully-completed circuits (A or B truncated)")

    with open("/tmp/opencode/diag_sabre_mimic.json", "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print("\n写入 /tmp/opencode/diag_sabre_mimic.json")


if __name__ == "__main__":
    main()
