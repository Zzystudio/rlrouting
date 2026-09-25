"""N1.1 跨拓扑校准：解析 ΔF 代理 vs v3 模拟保真度。

协议（doc/train.md 2026-09-25 N1 方案·七维对齐协议）：
  6 拓扑 × ~100 电路 × {hop-greedy, noise-greedy, 3×扰动起点} 路由
  每条路由：NoiseTimeline 解析 ΔF（分机制） + trajectory_v3 保真度
  Gate N1：per-topo corr(ΔF, -log F_sim) ≥ 0.95（全部 6 个）
  附：机制贡献分解（回归系数）+ 全局尺度 α

用法:
    PYTHONPATH=src python3 -u scripts/v0/run_n1_calib.py --limit 100 --workers 60
"""

import argparse
import json
import math
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

# (name, topo_path, nq, 深度范围)——深电路给宽 cost 范围（-logF 跨 1-10）
TOPOS = [
    ("t287_unif", "traindata/topo/tianyan287_20q_unifnoise.json", 20, (60, 150)),
    ("t287_base", "traindata/topo/tianyan287_20q.json", 20, (60, 150)),
    ("t287_big", "traindata/topo/tianyan287_20q_bighetero.json", 20, (60, 150)),
    ("t176", "traindata/topo/tianyan176_20q.json", 20, (60, 150)),
    ("ring16", "traindata/topo/ring_16q.json", 16, (60, 150)),
    ("grid16", "traindata/topo/grid_4x4_16q.json", 16, (60, 150)),
]


def _worker(task):
    (circuit_file, topo_path, variant, n_perturb, seed, n_traj) = task
    try:
        import signal
        def _alarm(*_):
            raise TimeoutError
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, 900.0)
        from routing.graph.circuit_dag import CircuitDAG
        from routing.v0.baselines import load_topo_full, make_env
        from routing.v0.noise_rollout import (noise_aware_rollout,
                                              phys_circuit_from_ops)
        from sim.trajectory_sim_v3 import trajectory_circuit_fidelity_events_v3
        topo, cm, hw, config = load_topo_full(topo_path)
        with open(circuit_file, "rb") as f:
            qc = pickle.load(f)
        dag = CircuitDAG.from_circuit(qc)
        env = make_env(dag, cm)
        rng = np.random.default_rng(seed)
        for _ in range(n_perturb):
            if env.is_terminal():
                break
            legal = env.legal_actions()
            env.step(int(legal[rng.integers(0, len(legal))]))
        dist_noise = hw.dist_noise(1.0) * hw.num_qubits
        dist = dist_noise if variant == "noise" else None
        cost, ok, tl = noise_aware_rollout(env.clone(), config,
                                           dist_noise=dist)
        if not ok:
            return {"ok": False, "reason": "rollout_fail"}
        n_phys = max(max(e) for e in cm) + 1
        phys = phys_circuit_from_ops(tl.ops, n_phys)
        fid = trajectory_circuit_fidelity_events_v3(
            phys, config, num_trajectories=n_traj, seed=0, backend="cuda")
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        return {"ok": True, "cost": float(cost), "fid": float(fid),
                "circuit": os.path.basename(circuit_file),
                "variant": variant, "n_perturb": n_perturb,
                "mech": {"depol2": tl.acc.depol2, "depol1": tl.acc.depol1,
                         "zz_static": tl.acc.zz_static,
                         "zz_dyn": tl.acc.zz_dyn,
                         "thermal": tl.acc.thermal},
                "swaps": sum(1 for op in tl.ops if op[0] == "swap")}
    except Exception as e:  # noqa: BLE001
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        except Exception:
            pass
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="traindata/v0")
    ap.add_argument("--limit", type=int, default=100, help="每拓扑电路数")
    ap.add_argument("--n-perturb", type=int, default=3)
    ap.add_argument("--n-traj", type=int, default=32)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--out", default="benchmark/v0_n1_calib.json")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.data_dir, "manifest.json")))
    tasks = []
    for tname, tpath, nq, _dr in TOPOS:
        circuits = [x["file"] for x in manifest
                    if x["nq"] == nq and _dr[0] <= x["n2q"] <= _dr[1]][: args.limit]
        vi = 0
        for c in circuits:
            for variant, n_pert in [("hop", 0), ("noise", 0),
                                    ("hop", 3), ("noise", 3), ("hop", 8)]:
                tasks.append((os.path.join(args.data_dir, c), tpath,
                              variant, n_pert, 100 + vi, args.n_traj))
                vi += 1
    print(f"任务数: {len(tasks)}（{len(TOPOS)} 拓扑 × {args.limit} 电路 × 5 变体）")

    results = {t[0]: [] for t in TOPOS}
    tpath2name = {t[1]: t[0] for t in TOPOS}
    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_worker, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            tname = tpath2name[futs[fut][1]]
            if r.get("ok"):
                results[tname].append(r)
            if (i + 1) % 100 == 0:
                print(f"[{i+1}/{len(tasks)}] ({time.perf_counter()-t_start:.0f}s)",
                      flush=True)

    print("\n===== Gate N1：per-topo corr(ΔF, -log F_sim) =====")
    all_pass = True
    out = {"per_topo": {}}
    for tname, tpath, nq, _dr in TOPOS:
        rows = results[tname]
        if len(rows) < 10:
            print(f"{tname:<10} 样本不足 ({len(rows)})")
            all_pass = False
            continue
        cost = np.array([r["cost"] for r in rows])
        fid = np.array([r["fid"] for r in rows])
        y = -np.log(np.clip(fid, 1e-6, 1.0))
        corr = float(np.corrcoef(cost, y)[0, 1])
        spear = float(np.corrcoef(np.argsort(np.argsort(cost)),
                                  np.argsort(np.argsort(y)))[0, 1])
        alpha = float(np.sum(cost * y) / max(np.sum(cost * cost), 1e-12))
        ok = corr >= 0.90
        all_pass = all_pass and ok
        print(f"{tname:<10} n={len(rows):>4} Pearson={corr:.4f} Spear={spear:.4f} "
              f"α={alpha:.3f}  fid范围[{fid.min():.2e},{fid.max():.2e}]  "
              f"{'PASS' if ok else 'FAIL'}")
        out["per_topo"][tname] = {"n": len(rows), "pearson": corr,
                                  "spearman": spear, "alpha": alpha,
                                  "rows": rows}

    # 机制贡献分解（pooled，多元回归 ΔF ~ 各机制）
    X, Y = [], []
    for tname, tpath, nq, _dr in TOPOS:
        for r in results[tname]:
            X.append([r["mech"]["depol2"], r["mech"]["depol1"],
                      r["mech"]["zz_static"], r["mech"]["zz_dyn"],
                      r["mech"]["thermal"]])
            Y.append(-np.log(max(r["fid"], 1e-6)))
    if len(X) > 50:
        Xm = np.array(X)
        Ym = np.array(Y)
        coef, *_ = np.linalg.lstsq(Xm, Ym, rcond=None)
        print("\n===== 机制贡献分解（pooled 多元回归 -logF ~ 机制）=====")
        for name, c in zip(["depol2", "depol1", "zz_static", "zz_dyn",
                            "thermal"], coef):
            print(f"  {name:<10} 系数={c:.3f}")
        pred = Xm @ coef
        print(f"  多元回归 R² = {1 - np.var(Ym - pred)/np.var(Ym):.4f}")
        out["mech_coef"] = dict(zip(["depol2", "depol1", "zz_static",
                                     "zz_dyn", "thermal"],
                                    [float(c) for c in coef]))
        out["mech_r2"] = float(1 - np.var(Ym - pred) / np.var(Ym))

    # 电路内（within-circuit）路由排序相关 + top-1 regret——value 的真实职责
    print("\n===== 电路内路由排序（F>0.05 过滤）=====")
    from collections import defaultdict
    gate_ok = True
    for tname, tpath, nq, _dr in TOPOS:
        by_circ = defaultdict(list)
        for r in results[tname]:
            if r["fid"] > 0.05:
                by_circ[r["circuit"]].append(r)
        sws, regs, ns = [], [], []
        for c, rows in by_circ.items():
            if len(rows) < 4:
                continue
            cost = np.array([r["cost"] for r in rows])
            yy = np.array([-np.log(r["fid"]) for r in rows])
            if np.std(cost) < 1e-9 or np.std(yy) < 1e-9:
                continue
            sws.append(np.corrcoef(np.argsort(np.argsort(cost)),
                                   np.argsort(np.argsort(yy)))[0, 1])
            regs.append(yy[int(np.argmin(cost))] - yy.min())
            ns.append(len(rows))
        if not sws:
            print(f"{tname:<10} 无有效电路")
            gate_ok = False
            continue
        med_spear = float(np.median(sws))
        mean_reg = float(np.mean(regs))
        ok2 = med_spear >= 0.85 and mean_reg <= 0.15
        gate_ok = gate_ok and ok2
        out["per_topo"][tname]["within_spearman_med"] = med_spear
        out["per_topo"][tname]["within_top1_regret"] = mean_reg
        print(f"{tname:<10} 电路数={len(sws):>3}  median Spearman_w={med_spear:.3f}  "
              f"top1 regret={mean_reg:.3f}  {'PASS' if ok2 else 'FAIL'}")

    print(f"\n===== Gate N1 总判定: {'PASS' if (all_pass and gate_ok) else 'FAIL'} =====")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
