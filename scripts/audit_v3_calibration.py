#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 0b：v3 预校准审计（doc/20260920训练方案.md §七 Step 0）。

对 v3 模拟器做受控回归，标定时钟化奖励的物理单价：
  - 孤立 SWAP 成本       -> swap_price_scale@v3 重推导（旧 4.6 为 v2 口径）
  - 受控重叠 SWAP        -> w_xt（边际串扰价，rad 单位）
  - 受控 idle 间隙       -> eta_idle（µs·qubit 单位）
  - 单 CX 静态 ZZ/e_edge -> w_zz（rad 单位）
  - 记录项：v2 vs v3 swap 密集族保真度差（headroom 预期）

输出：JSON（/tmp/audit_v3_calibration.json）+ 终端表格。
用法（src/ 下）：python3 scripts/audit_v3_calibration.py \
    --topo ../traindata/topo/tianyan287_20q.json --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit

from routing.rl.eval_policy import load_topo
from sim.trajectory_sim_v2 import schedule_phys_circuit_events
from sim.trajectory_sim_v2 import trajectory_circuit_fidelity_events as _fid_v2
from sim.trajectory_sim_v3 import trajectory_circuit_fidelity_events_v3 as _fid_v3


def _mk_circ(n, gates):
    qc = QuantumCircuit(n)
    for g in gates:
        if len(g) == 1:
            getattr(qc, g[0])(*g[1:])
        else:
            getattr(qc, g[0])(*g[1:])
    return qc


def _fid_with_gap(qc, config, gap_us, T=64, backend="auto", seed=0):
    """v3 保真度，且在所有事件前插入全局 idle 间隙（µs）——用显式事件控制，
    规避 delay 指令不被调度的问题。"""
    from sim.trajectory_sim_v2 import (EventTrajectorySimulator,
                                       _reduce_phys_circuit_for_fidelity_v2)
    rc, rconfig, remap = _reduce_phys_circuit_for_fidelity_v2(qc, config)
    sim = EventTrajectorySimulator(rconfig, num_trajectories=T, seed=seed,
                                   backend=backend)
    events = schedule_phys_circuit_events(rc)
    events2 = [(float(s) + gap_us, float(e) + gap_us, op, qs, pi)
               for (s, e, op, qs, pi) in events]
    return float(sim.fidelity_events(rc, events2))


def _fid(qc, config, sim="v3", T=64, backend="auto", seed=0, n_seed=3):
    """v3/v2 保真度，多 seed 平均降采样方差（校准用，信号 ~0.01 << 单次 σ）。"""
    fn = _fid_v3 if sim == "v3" else _fid_v2
    vals = [float(fn(qc, config, num_trajectories=T, seed=seed + i,
                     backend=backend)) for i in range(n_seed)]
    return float(np.mean(vals))


def _base_gates(n):
    """纠缠化基准线路（确保非平凡态，退相干可见）。"""
    gates = [("h", 0)]
    for i in range(n - 1):
        gates.append(("cx", i, i + 1))
    gates.append(("cx", n - 1, 0))
    return gates


def linear_slope(xs, ys):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    m, b = np.polyfit(xs, ys, 1)
    return m, b


def audit(topo, sim, T, backend, seed):
    config, hw, cm = load_topo(topo)
    n = min(10, len(config.t1_times))   # 校准电路规模（用 tianyan 真实噪声子图）
    cm = [e for e in cm if max(e) < n]  # 只保留子图内的耦合边
    res = {}

    # ---- 1. 孤立 SWAP 成本（无重叠）----
    swaps_cost = []
    e_list = []
    for e in cm[:4]:
        p, q = e
        base = _mk_circ(n, _base_gates(n))
        f0 = _fid(base, config, sim, T, backend, seed)
        base.swap(p, q)
        f1 = _fid(base, config, sim, T, backend, seed)
        swaps_cost.append(f0 - f1)
        e_list.append(float(hw.two_q_err[p, q]))
    fid_per_swap = float(np.mean(swaps_cost))
    mean_e = float(np.mean(e_list))
    # 换算：reward 单位 1 分 ≈ pot_progress_b 的 fid 当量由进度奖励口径定；
    # 给出建议 scale = fid_per_swap / (3*e_edge_norm)（e_edge 已归一化 ~0.2）
    # 再按历史 20260917 转换（1 fid ≈ 476 reward units）折算。
    FID_PER_REWARD = 476.0  # 20260917 审计：1 reward unit ≈ 1/476 fid
    scale_v3 = fid_per_swap * FID_PER_REWARD / (3.0 * max(mean_e, 1e-6))
    res["swap"] = {"fid_per_swap": fid_per_swap, "mean_e_edge": mean_e,
                   "suggested_swap_price_scale_v3": round(scale_v3, 2)}

    # ---- 2. 受控重叠 SWAP：SWAP 与相邻边 CX 并发 ----
    over_cost = []
    th_over = []
    edge01 = cm[0]  # CX 基准边
    # 找与 edge01 相邻（1-hop 交叉对）且不相交的 SWAP 边（全图搜索）
    over_edges = [e for e in cm
                  if set(e) & set(edge01) == set()
                  and any(hw.adj[x, y] > 0 for x in e for y in edge01)]
    for e in over_edges[:4]:
        p, q = e
        base = _mk_circ(n, _base_gates(n))
        f0 = _fid(base, config, sim, T, backend, seed)
        base.cx(*edge01)
        base.swap(p, q)
        f1 = _fid(base, config, sim, T, backend, seed)
        over_cost.append(f0 - f1)
        th_over.append(float(hw.zz[p, q]) * 0.05)
    if over_cost:
        mean_th = float(np.mean(th_over))
        fid_per_overlap = float(np.mean(over_cost))
        w_xt = fid_per_overlap * FID_PER_REWARD / max(mean_th, 1e-6)
        res["overlap_swap"] = {"n_edges": len(over_cost),
                               "fid_per_overlap": fid_per_overlap,
                               "mean_theta_rad": mean_th,
                               "suggested_w_xt": round(w_xt, 1)}
    else:
        res["overlap_swap"] = {"note": "no adjacent edge found"}

    # ---- 3. 受控 idle 间隙（显式事件注入，规避 delay 不被调度）----
    idle_slope = []
    idle_us = []
    for gap in (0.0, 1.0, 5.0, 10.0):
        base = _mk_circ(n, _base_gates(n))
        f0 = _fid_with_gap(base, config, gap, T, backend, seed)
        idle_slope.append(f0)
        idle_us.append(gap)
    m_idle, _ = linear_slope(idle_us, idle_slope)
    eta_idle = m_idle * FID_PER_REWARD / max(1, n)
    res["idle"] = {"fid_per_us_all_qubits": round(m_idle, 6),
                   "suggested_eta_idle_per_qubit_us": round(eta_idle, 4)}

    # ---- 4. 单 CX 静态 ZZ（扫 θ，控 e_edge 相近的边）----
    zz_delta = []
    zz_vals = []
    edge_pool = sorted(cm, key=lambda e: float(hw.two_q_err[e[0], e[1]]))
    ref_e = float(hw.two_q_err[edge_pool[0][0], edge_pool[0][1]])
    pool = [e for e in edge_pool
            if abs(float(hw.two_q_err[e[0], e[1]]) - ref_e) < 0.1]
    for e in pool[:6]:
        p, q = e
        base = _mk_circ(n, _base_gates(n))
        f0 = _fid(base, config, sim, T, backend, seed)
        base.cx(p, q)
        f1 = _fid(base, config, sim, T, backend, seed)
        th = float(hw.zz[p, q]) * 0.05
        zz_delta.append(f0 - f1)
        zz_vals.append(th)
    if len(zz_vals) >= 3 and max(zz_vals) - min(zz_vals) > 1e-4:
        m_zz, _ = linear_slope(zz_vals, zz_delta)
        w_zz = m_zz * FID_PER_REWARD
        res["static_zz"] = {"fid_per_rad": round(m_zz, 4),
                            "suggested_w_zz": round(w_zz, 2)}
    else:
        res["static_zz"] = {"note": "theta 分布过窄无法回归",
                            "n": len(zz_vals),
                            "theta_range": [round(min(zz_vals), 5),
                                            round(max(zz_vals), 5)]}

    # ---- 5. 记录项：v2 vs v3 swap 密集 ----
    base = _mk_circ(n, _base_gates(n))
    f2 = _fid(base, config, "v2", T, backend, seed)
    f3 = _fid(base, config, "v3", T, backend, seed)
    res["v2_vs_v3"] = {"fid_v2": f2, "fid_v3": f3, "delta": f2 - f3}

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default="../traindata/topo/tianyan287_20q.json")
    ap.add_argument("--device", default="cpu", help="cpu/cuda/cuda:N/auto")
    ap.add_argument("--trajectories", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/tmp/audit_v3_calibration.json")
    args = ap.parse_args()
    res = audit(args.topo, "v3", args.trajectories, args.device, args.seed)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"写入 {args.out}")


if __name__ == "__main__":
    main()
