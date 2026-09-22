#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""干净权衡审计：并行 vs 串行相邻 2Q 门的真实 v3 保真度差。

Δ = F(并行) − F(串行)：若 Δ>0，并行更好（重叠 ZZ 成本 < 串行化的 idle 退相干
成本）——奖励应不鼓励串行化（w_xt 应小）；若 Δ<0，串行化更好。
用同一电路的两套事件列表（控制变量唯一 = 重叠），规避上次审计把 SWAP 固有
成本混入的缺陷。输出：每对边的 Δ、典型 θ、以及建议的 w_xt（相对 eta_idle
的平衡点）。

用法（项目根）：PYTHONPATH=src python3 scripts/audit_overlap_clean.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit
from routing.rl.eval_policy import load_topo
from sim.trajectory_sim_v2 import (EventTrajectorySimulator,
                                   _reduce_phys_circuit_for_fidelity_v2)

TOPO = "traindata/topo/tianyan287_20q.json"
T = 128
N_SEED = 3


def _events_serial(circ, n, a, b, c, d, two_time=0.3):
    """CX(a,b) 后 CX(c,d) 串行。返回 (start,end,op,qubits,phys_idx) 事件。"""
    evs = []
    idx = 0
    for op, qs in (("cx", (a, b)), ("cx", (c, d))):
        s = idx * two_time
        evs.append((s, s + two_time, op, qs, None))
        idx += 1
    return evs


def _events_parallel(circ, n, a, b, c, d, two_time=0.3):
    """CX(a,b) 与 CX(c,d) 同时（重叠）。"""
    return [(0.0, two_time, "cx", (a, b), None),
            (0.0, two_time, "cx", (c, d), None)]


def _fid_events(config, circ, events, backend):
    rc, rconfig, remap = _reduce_phys_circuit_for_fidelity_v2(circ, config)
    sim = EventTrajectorySimulator(rconfig, num_trajectories=T, backend=backend)
    evs2 = [(s, e, op, tuple(remap.get(q, q) for q in qs) if remap else qs, pi)
            for (s, e, op, qs, pi) in events]
    return float(sim.fidelity_events(rc, evs2))


def main():
    config, hw, cm = load_topo(TOPO)
    # tianyan 10q 子图：找相邻边对（共享邻居、不相交）
    n = 10
    edges = [e for e in cm if max(e) < n]
    # 相邻边对：e1=(a,b), e2=(c,d)，不相交且 1-hop 相邻
    pairs = []
    for i in range(len(edges)):
        a, b = edges[i]
        for j in range(i + 1, len(edges)):
            c, d = edges[j]
            if {a, b} & {c, d}:
                continue
            if any(hw.adj[x, y] > 0 for x in (a, b) for y in (c, d)):
                pairs.append((edges[i], edges[j]))
    if not pairs:
        print("no adjacent edge pairs found")
        return

    circ = QuantumCircuit(n)
    results = []
    for (e1, e2) in pairs[:8]:
        a, b = e1
        c, d = e2
        circ = QuantumCircuit(n)   # 每对独立电路（只含 2 个 CX）
        circ.cx(a, b)
        circ.cx(c, d)
        f_ser = np.mean([_fid_events(config, circ, _events_serial(
            circ, n, a, b, c, d), "cuda:0") for _ in range(N_SEED)])
        f_par = np.mean([_fid_events(config, circ, _events_parallel(
            circ, n, a, b, c, d), "cuda:0") for _ in range(N_SEED)])
        delta = f_par - f_ser
        th = float(hw.zz[a, c]) * 0.05
        results.append({"e1": e1, "e2": e2, "f_ser": f_ser, "f_par": f_par,
                        "delta": delta, "theta_rad": th})
        print(f"{e1} vs {e2}: F(par)={f_par:.5f} F(ser)={f_ser:.5f} "
              f"Δ={delta:+.5f} θ={th:.4f}")

    # 汇总：Δ 均值（并行是否更好）+ 建议 w_xt
    deltas = [r["delta"] for r in results]
    mean_d = float(np.mean(deltas))
    print(f"\n平均 Δ(F_par − F_ser) = {mean_d:+.5f}")
    if mean_d > 0:
        print("→ 并行更好：重叠 ZZ 成本 < 串行化 idle 退相干成本，"
              "奖励不应过度鼓励串行化（w_xt 应小、eta_idle 应相对大）")
    else:
        print("→ 串行化更好：ZZ 成本占主导（w_xt 可保持较大）")
    # 建议 w_xt：使"避一个重叠"的奖励 ≈ 该重叠的真实 Δ（用 B=0.2 尺度折算）
    # 每重叠 θ 量级 ~0.01-0.02 rad；Δ 为该真实净成本
    import json
    with open("/tmp/opencode/audit_overlap_clean.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
