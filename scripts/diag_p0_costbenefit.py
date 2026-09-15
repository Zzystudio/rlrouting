#!/usr/bin/env python3
"""
diag_p0_costbenefit.py
P0 建模/奖励函数设计成本收益核算（无训练）。
基于 tianyan287_20q 拓扑，输出：
  A. 静态拓扑核算（β 约束、绕路边界、权重预算）
  B. D1: 决策点 xtalk_pred / e_edge 跨候选边方差
  C. D2: 奖励分量基线占比
  D. D3: 等跳数路径/噪声加权距离 tie-break 空间

用法（在 src/ 目录外执行）：
  PYTHONPATH=src python3 scripts/diag_p0_costbenefit.py \
    --topo traindata/topo/tianyan287_20q.json \
    --circuits benchmark/nam_circs,traindata/gen_structured \
    --out logs/diag_p0_costbenefit_t287.json
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 路径 / 导入
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from qiskit.qasm2 import load as qasm2_load

from routing.rl.train_agent import load_topo
from routing.rl.env import RoutingEnv
from routing.graph.circuit_dag import CircuitDAG
from routing.routing import sabre_route


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _shortest_path_count_and_err_bounds(
    adj: np.ndarray,
    e_edge: np.ndarray,
    src: int,
    tgt: int,
    path_cap: int = 50,
) -> Tuple[int, float, float]:
    """返回 (最短路径条数（上限 path_cap）, 最短路径上 path_err 最小值, 最大值)。

    若 src==tgt 或相邻，视为 1 条路径且 path_err=0。
    """
    n = adj.shape[0]
    if src == tgt:
        return 1, 0.0, 0.0
    if adj[src, tgt] > 0:
        return 1, float(e_edge[src, tgt]), float(e_edge[src, tgt])

    # BFS 求距离
    dist = np.full(n, n, dtype=int)
    dist[src] = 0
    q = [src]
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        for v in range(n):
            if adj[u, v] > 0 and dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                q.append(v)
    if dist[tgt] >= n:
        return 0, 0.0, 0.0

    # 按 BFS 层构建 DAG（只保留在 shortest path 上的边）
    dag = defaultdict(list)
    for u in range(n):
        for v in range(n):
            if adj[u, v] > 0 and dist[v] == dist[u] + 1:
                dag[u].append(v)

    count = 0
    min_err = float("inf")
    max_err = 0.0

    def dfs(u: int, acc: float):
        nonlocal count, min_err, max_err
        if count >= path_cap:
            return
        if u == tgt:
            count += 1
            min_err = min(min_err, acc)
            max_err = max(max_err, acc)
            return
        for v in dag[u]:
            dfs(v, acc + float(e_edge[u, v]))

    dfs(src, 0.0)
    return count, min_err, max_err


def _load_qasm_circuits(dirs: List[str], max_qubits: int = 20):
    circuits = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".qasm"):
                continue
            fpath = os.path.join(d, fname)
            try:
                qc = qasm2_load(fpath)
            except Exception as e:
                print(f"[skip] {fpath}: {e}")
                continue
            if qc.num_qubits > max_qubits:
                continue
            circuits.append((fname.replace(".qasm", ""), qc))
    return circuits


def _edge_index_of_swap(coupling_map: List[Tuple[int, int]], p: int, q: int) -> int:
    for i, (a, b) in enumerate(coupling_map):
        if (a == p and b == q) or (a == q and b == p):
            return i
    raise ValueError(f"SWAP ({p},{q}) not in coupling_map")


def _prototype_parallel_circuits(max_qubits: int = 20):
    """临时构造几个代表新并行结构族的电路，用于对比 D1 方差。"""
    from qiskit import QuantumCircuit
    out = []

    # 1) 多层独立 CNOT 匹配（高并发）
    for n in [8, 12, 16, 20]:
        for layers in [2, 4]:
            for seed in range(2):
                rng = np.random.default_rng(seed)
                qc = QuantumCircuit(n)
                perm = rng.permutation(n)
                for _ in range(layers):
                    for i in range(0, n - 1, 2):
                        qc.cx(int(perm[i]), int(perm[i + 1]))
                    perm = rng.permutation(n)
                out.append((f"proto_parallel_n{n}_l{layers}_s{seed}", qc))

    # 2) QFT 风格长程 butterfly
    for n in [8, 12, 16]:
        qc = QuantumCircuit(n)
        for i in range(n):
            qc.h(i)
        for k in range(1, n):
            for i in range(n - k):
                qc.cp(np.pi / (2 ** k), i, i + k)
        out.append((f"proto_qft_n{n}", qc))

    return out


# ---------------------------------------------------------------------------
# 静态拓扑核算
# ---------------------------------------------------------------------------
def static_accounting(topo_path: str):
    cfg, hw, cm = load_topo(topo_path)
    n = hw.num_qubits
    adj = hw.adj
    e_edge = hw.two_q_err
    zz = hw.zz

    # 全对最短路径
    dist = np.full((n, n), n, dtype=int)
    for s in range(n):
        dist[s, s] = 0
        q = [s]
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            for v in range(n):
                if adj[u, v] > 0 and dist[s, v] > dist[s, u] + 1:
                    dist[s, v] = dist[s, u] + 1
                    q.append(v)

    kmax = int(dist[dist < n].max())
    mean_sp = float(dist[dist < n].mean())

    errs = [float(e_edge[e]) for e in cm]
    zzs = [float(zz[e]) for e in cm]
    emin, emax = min(errs), max(errs)
    espread = emax - emin
    zmin, zmax = min(zzs), max(zzs)

    # β 临界（等跳数保序）
    denom = kmax * emax - (kmax + 1) * emin
    beta_crit = float("inf") if denom <= 0 else 1.0 / denom
    beta_conservative = 1.0 / (kmax * emax)

    # 绕路回本所需门数（潜在模式：+1 SWAP 代价 ≈ 3*e_edge_mean_norm；保守取 mean+max 平均）
    swap_cost_norm = 3.0 * np.mean(errs)
    payback = {f"eta_err={eta}": float(swap_cost_norm / (eta * espread)) for eta in [0.5, 1.0]}

    return {
        "n_qubits": n,
        "n_edges": len(cm),
        "diameter": kmax,
        "mean_shortest_path": mean_sp,
        "e_norm": {"min": emin, "max": emax, "mean": float(np.mean(errs)), "spread": espread},
        "e_raw": {"min": emin * 0.05, "max": emax * 0.05, "spread": espread * 0.05},
        "zz_norm": {"min": zmin, "max": zmax, "mean": float(np.mean(zzs))},
        "zz_raw_rad_per_cx": {"min": zmin * 0.05, "max": zmax * 0.05, "mean": float(np.mean(zzs)) * 0.05},
        "beta": {"critical": beta_crit, "conservative": beta_conservative, "recommended": round(beta_conservative * 0.8, 2)},
        "swap_cost_norm": swap_cost_norm,
        "detour_payback_gates": payback,
    }


# ---------------------------------------------------------------------------
# 轨迹诊断
# ---------------------------------------------------------------------------
def diagnose_trajectories(topo_path: str, circuit_dirs: List[str], use_prototypes: bool = True):
    cfg, hw, cm = load_topo(topo_path)
    circuits = _load_qasm_circuits(circuit_dirs)
    if use_prototypes:
        circuits += _prototype_parallel_circuits()

    # 环境参数：R3b-like potential mode + scheduler（用于采集 timing/xtalk）
    env_kwargs = dict(
        hw=hw,
        coupling_map=cm,
        reward_mode="routing",
        reward_potential=True,
        shaping_gamma=0.99,
        eta_shape=0.3,
        alpha_ext=0.5,
        use_scheduler=True,
        eta_time=0.01,
        eta_xtalk_par=0.1,
        eta_idle=0.005,
        eta_parallel=0.05,
        xtalk_alpha=0.03,
        swap_duration_us=0.9,
        max_num_qubits=20,
        max_num_edges=31,
        mapping_phase=False,
        random_init=False,
        lookahead_features=True,
        use_gnn=False,
        max_episode_steps=2000,
    )

    records = []
    per_circuit = []

    for name, qc in circuits:
        if qc.num_qubits > 20:
            continue
        try:
            dag = CircuitDAG.from_circuit(qc)
        except Exception as e:
            print(f"[skip] {name}: DAG failed {e}")
            continue

        # SABRE 路由 → swap 动作序列
        try:
            phys, info = sabre_route(qc, cfg, seed=0)
        except Exception as e:
            print(f"[skip] {name}: SABRE failed {e}")
            continue
        init_layout = info.get("initial_layout")
        if init_layout is None:
            continue
        init_layout = init_layout[: qc.num_qubits]
        used_phys = set(init_layout)

        swaps = []
        for inst, qargs, _ in phys.data:
            if inst.name == "swap":
                p, q = qargs[0]._index, qargs[1]._index
                if p in used_phys and q in used_phys:
                    swaps.append(_edge_index_of_swap(cm, p, q))

        # 复现轨迹
        env = RoutingEnv(dag, **env_kwargs, init_mapping=init_layout)
        obs, _ = env.reset()

        step_stats = []
        for action in swaps:
            # 记录当前状态
            valid = np.ones(env.num_edges, dtype=bool)
            if hasattr(env, "get_deadlock_mask"):
                valid &= ~env.get_deadlock_mask()
            if hasattr(env, "get_unmapped_mask"):
                valid &= ~env.get_unmapped_mask()

            xtalk = env._edge_lookahead_features()[:, 0]  # xtalk_pred
            e_per_edge = np.array([float(env.hw.two_q_err[e]) for e in env.coupling_map])
            zz_per_edge = np.array([float(env.hw.zz[e]) for e in env.coupling_map])

            # D3: front-layer 门等跳数路径
            ready_gates = env._ready_2q_gates()
            multi_path = []
            for g in ready_gates:
                pa, pb = env.mapping[g.qubits[0]], env.mapping[g.qubits[1]]
                if env.hw.adj[pa, pb] > 0:
                    continue
                cnt, pmin, pmax = _shortest_path_count_and_err_bounds(
                    env.hw.adj, env.hw.two_q_err, pa, pb, path_cap=50
                )
                if cnt > 1:
                    multi_path.append({
                        "cnt": cnt,
                        "path_err_min": pmin,
                        "path_err_max": pmax,
                        "delta": pmax - pmin,
                    })

            step_stats.append({
                "n_ready": len(ready_gates),
                "valid_edges": int(valid.sum()),
                "xtalk_pred_mean": float(xtalk[valid].mean()) if valid.any() else 0.0,
                "xtalk_pred_std": float(xtalk[valid].std()) if valid.sum() > 1 else 0.0,
                "xtalk_pred_maxmin": float(xtalk[valid].max() - xtalk[valid].min()) if valid.any() else 0.0,
                "e_edge_std": float(e_per_edge[valid].std()) if valid.sum() > 1 else 0.0,
                "e_edge_maxmin": float(e_per_edge[valid].max() - e_per_edge[valid].min()) if valid.any() else 0.0,
                "zz_edge_std": float(zz_per_edge[valid].std()) if valid.sum() > 1 else 0.0,
                "multi_path_count": len(multi_path),
                "multi_path_delta_mean": float(np.mean([m["delta"] for m in multi_path])) if multi_path else 0.0,
            })

            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break

        # 收集 episode 级原始量
        timing = env.timing
        crosstalk_events = float(timing.crosstalk_events) if timing else 0.0
        total_time = float(timing.total_time) if timing else 0.0
        n_gates = len([g for g in dag.gates if not g.is_measure])
        n_2q = len([g for g in dag.gates if g.is_two_qubit])
        n_swaps = len(swaps)

        # 按 executed gates 累计的边误差（从物理线路）
        # 这里用简化估计：每个 2q 门取执行时所在边的 e_edge 均值（实际按映射会变化，但用于量级足够）
        sum_e_gates = float(np.sum([env.hw.two_q_err[env.mapping[g.qubits[0]], env.mapping[g.qubits[1]]]
                                    for g in dag.gates if g.is_two_qubit]))
        sum_e_swaps = 3.0 * float(np.sum([env.hw.two_q_err[env.coupling_map[e]] for e in swaps]))

        per_circuit.append({
            "name": name,
            "n_qubits": qc.num_qubits,
            "n_gates": n_gates,
            "n_2q": n_2q,
            "n_swaps": n_swaps,
            "crosstalk_events": crosstalk_events,
            "total_time_us": total_time,
            "sum_e_gates_norm": sum_e_gates,
            "sum_e_swaps_norm": sum_e_swaps,
            "steps": len(step_stats),
        })
        records.extend([{"circuit": name, **s} for s in step_stats])

    return records, per_circuit


# ---------------------------------------------------------------------------
# 汇总报告
# ---------------------------------------------------------------------------
def summarize(records: List[dict], per_circuit: List[dict], static: dict):
    # D1 方差
    xtalk_std = [r["xtalk_pred_std"] for r in records]
    xtalk_mm = [r["xtalk_pred_maxmin"] for r in records]
    e_std = [r["e_edge_std"] for r in records]
    zz_std = [r["zz_edge_std"] for r in records]

    d1 = {
        "xtalk_pred_std_mean": float(np.mean(xtalk_std)),
        "xtalk_pred_std_median": float(np.median(xtalk_std)),
        "xtalk_pred_maxmin_mean": float(np.mean(xtalk_mm)),
        "xtalk_pred_nonzero_frac": float(np.mean([s > 1e-6 for s in xtalk_std])),
        "e_edge_std_mean": float(np.mean(e_std)),
        "zz_edge_std_mean": float(np.mean(zz_std)),
    }

    # D3 等跳数路径
    mp_counts = [r["multi_path_count"] for r in records]
    mp_deltas = [r["multi_path_delta_mean"] for r in records if r["multi_path_delta_mean"] > 0]
    d3 = {
        "steps_with_ready_2q": len(records),
        "steps_with_multi_path": int(np.sum([c > 0 for c in mp_counts])),
        "frac_steps_multi_path": float(np.mean([c > 0 for c in mp_counts])),
        "mean_path_err_delta_when_tied": float(np.mean(mp_deltas)) if mp_deltas else 0.0,
    }

    # D2 奖励分量占比（潜在模式 + eta_xtalk_par=0.1）
    total_progress = 0.045 * sum(c["n_gates"] for c in per_circuit)
    total_gate_cost = sum(c["sum_e_gates_norm"] for c in per_circuit)
    total_swap_cost = sum(c["sum_e_swaps_norm"] for c in per_circuit)
    total_xtalk_cost = 0.1 * sum(c["crosstalk_events"] for c in per_circuit)
    total_magnitude = total_progress + total_gate_cost + total_swap_cost + total_xtalk_cost
    d2 = {
        "progress": {"raw": total_progress, "share": total_progress / total_magnitude},
        "gate_err_cost": {"raw": total_gate_cost, "share": total_gate_cost / total_magnitude},
        "swap_err_cost": {"raw": total_swap_cost, "share": total_swap_cost / total_magnitude},
        "xtalk_scheduler_cost": {"raw": total_xtalk_cost, "share": total_xtalk_cost / total_magnitude},
        "per_circuit_xtalk_events_mean": float(np.mean([c["crosstalk_events"] for c in per_circuit])),
    }

    return {"static": static, "D1": d1, "D2": d2, "D3": d3, "n_circuits": len(per_circuit), "n_steps": len(records)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default="traindata/topo/tianyan287_20q.json")
    ap.add_argument("--circuits", default="benchmark/nam_circs,traindata/gen_structured")
    ap.add_argument("--out", default="logs/diag_p0_costbenefit_t287.json")
    ap.add_argument("--no-prototypes", action="store_true")
    args = ap.parse_args()

    dirs = [d.strip() for d in args.circuits.split(",") if d.strip()]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print("[1/3] 静态拓扑核算 ...")
    static = static_accounting(args.topo)

    print("[2/3] 轨迹诊断（SABRE 路由） ...")
    records, per_circuit = diagnose_trajectories(args.topo, dirs, use_prototypes=not args.no_prototypes)

    print("[3/3] 汇总 ...")
    report = summarize(records, per_circuit, static)
    report["circuits"] = per_circuit
    report["raw_records_sample"] = records[:200]

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n报告已写入: {args.out}\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("circuits", "raw_records_sample")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
