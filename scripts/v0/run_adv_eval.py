"""5-8q advantage 头端到端评估：M1-pure vs M2-V(absolute) vs M-A(advantage)。

主指标：optimality gap = swaps − V*(s₀)（ExactSolver 精确真值，5-8q 独有）。
留出电路（与 build_adv_labels 相同的分层切分，seed=0）。
辅指标：叶评估耗时、终止率、配对 Wilcoxon。

用法:
    PYTHONPATH=src python3 -u scripts/v0/run_adv_eval.py --per-topo 8
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
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

ADV_TOPOS = [
    ("cross5", "traindata/topo/cross_5q.json", 5),
    ("ring5", "traindata/topo/ring_5q.json", 5),
    ("line5", "traindata/topo/ibmq_5_line.json", 5),
    ("line8", "traindata/topo/line_8q.json", 8),
    ("ring8", "traindata/topo/ring_8q.json", 8),
]


def _stratified_test(circuits, seed=0, test_frac=0.2):
    strata = {}
    for c in circuits:
        parts = c.replace(".pkl", "").split("_")
        fam = "perm_mix" if parts[0] == "perm" else parts[0]
        strata.setdefault((fam, parts[2]), []).append(c)
    rng = np.random.default_rng(seed)
    test = set()
    for k, cs in sorted(strata.items()):
        cs = sorted(set(cs))
        rng.shuffle(cs)
        test.update(cs[:max(1, int(len(cs) * test_frac))])
    return test


def _worker(task):
    (circuit_file, topo_path, adv_path, abs_path, sims, n_traj) = task
    try:
        import signal
        def _alarm(*_):
            raise TimeoutError
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, 900.0)
        from routing.graph.circuit_dag import CircuitDAG
        from routing.v0.baselines import load_topo, make_env
        from routing.v0.exact_solver import ExactSolver
        from routing.v0.mcts import MCTSConfig, mcts_episode
        from routing.v0.value_net import ValueNet
        topo, cm, hw = load_topo(topo_path)
        with open(circuit_file, "rb") as f:
            qc = pickle.load(f)
        dag = CircuitDAG.from_circuit(qc)
        solver = ExactSolver(dag, cm, max_nodes=2_000_000)
        env0 = make_env(dag, cm)
        vstar = solver.solve(env0.executed_mask, env0.mapping)
        if vstar is None:
            return {"circuit": os.path.basename(circuit_file), "error": "V* 超预算"}

        net_adv = ValueNet(in_dim=24)
        net_adv.load_state_dict(torch.load(adv_path, map_location="cpu",
                                           weights_only=True))
        net_adv.eval()
        net_abs = ValueNet(in_dim=12)
        net_abs.load_state_dict(torch.load(abs_path, map_location="cpu",
                                           weights_only=True))
        net_abs.eval()
        adv_ctx = {"model": net_adv}

        if vstar == 0:
            return {"circuit": os.path.basename(circuit_file), "error": "V*=0 平凡"}
        out = {"circuit": os.path.basename(circuit_file), "vstar": int(vstar)}
        for name, cfg, kw in [
            ("m1pure", MCTSConfig(sims=sims, prior="sabre",
                                  value="rollout_sabre", seed=0), {}),
            ("m1corr", MCTSConfig(sims=sims, prior="sabre",
                                  value="rollout_sabre", seed=0,
                                  depth_corrected=True), {}),
            ("m2v", MCTSConfig(sims=sims, prior="sabre", value="learned",
                               seed=0), {"value_net": net_abs}),
            ("madv", MCTSConfig(sims=sims, prior="sabre", value="adv",
                                seed=0), {"adv_ctx": adv_ctx}),
        ]:
            env = make_env(dag, cm)
            t0 = time.perf_counter()
            n, ok, st, _ = mcts_episode(env, cfg, max_steps=800, **kw)
            dt = time.perf_counter() - t0
            out[f"{name}_swaps"] = float(n) if ok else None
            out[f"{name}_ok"] = ok
            out[f"{name}_wall"] = dt
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        out["error"] = None
        return out
    except Exception as e:  # noqa: BLE001
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        except Exception:
            pass
        return {"circuit": os.path.basename(circuit_file),
                "error": f"{type(e).__name__}: {e}"}


def wilcoxon_p(a, b):
    dd = np.asarray(a, float) - np.asarray(b, float)
    dd = dd[dd != 0]
    n = len(dd)
    if n < 3:
        return 1.0
    absd = np.abs(dd)
    order = np.argsort(np.argsort(absd)) + 1.0
    ranks = order.copy()
    for v in np.unique(absd):
        m = absd == v
        if m.sum() > 1:
            ranks[m] = order[m].mean()
    T = float(ranks[dd > 0].sum())
    T = min(T, n * (n + 1) / 2 - T)
    mu = n * (n + 1) / 4
    sig = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (T - mu) / sig
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return max(p, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="traindata/v0")
    ap.add_argument("--adv-model", default="models/v0_adv8.pt")
    ap.add_argument("--abs-model", default="models/v0_abs8.pt")
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--per-topo", type=int, default=8)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--out", default="benchmark/v0_adv_eval.json")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.data_dir, "manifest.json")))
    tasks = []
    for tname, tpath, nq in ADV_TOPOS:
        circuits = [x["file"] for x in manifest
                    if x["nq"] == nq and x["n2q"] >= 8][: 60]
        test = _stratified_test(circuits)
        eval_circuits = sorted(test)[: args.per_topo]
        print(f"{tname}: 留出评估 {len(eval_circuits)} 电路")
        for c in eval_circuits:
            tasks.append((os.path.join(args.data_dir, c), tpath,
                          args.adv_model, args.abs_model, args.sims, 0))
    print(f"任务: {len(tasks)} × 3 方法（sims={args.sims}）")

    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            rows.append(r)
            if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                print(f"[{i+1}/{len(tasks)}] ({time.perf_counter()-t0:.0f}s)",
                      flush=True)

    print(f"\n===== 端到端（sims={args.sims}，留出电路）=====")
    print(f"{'topo':<8}{'n':>3}{'V*':>5}{'M1':>7}{'M2V':>7}{'MA':>7}  "
          f"gap: M1/M2V/MA (mean)   wall(s): M1/MA")
    all_gaps = {"m1pure": [], "m2v": [], "madv": []}
    all_walls = {"m1pure": [], "madv": []}
    per_topo = {}
    for tname, tpath, nq in ADV_TOPOS:
        rs = [r for r in rows if r.get("error") is None
              and r.get("m1pure_swaps") is not None
              and r.get("m2v_swaps") is not None
              and r.get("madv_swaps") is not None]
        if not rs:
            print(f"{tname:<8}  无有效数据")
            continue
        vs = np.array([r["vstar"] for r in rs])
        g1 = np.array([r["m1pure_swaps"] for r in rs]) - vs
        g2 = np.array([r["m2v_swaps"] for r in rs]) - vs
        g3 = np.array([r["madv_swaps"] for r in rs]) - vs
        w1 = np.mean([r["m1pure_wall"] for r in rs])
        w3 = np.mean([r["madv_wall"] for r in rs])
        print(f"{tname:<8}{len(rs):>3}{vs.mean():>5.1f}"
              f"{np.array([r['m1pure_swaps'] for r in rs]).mean():>7.1f}"
              f"{np.array([r['m2v_swaps'] for r in rs]).mean():>7.1f}"
              f"{np.array([r['madv_swaps'] for r in rs]).mean():>7.1f}"
              f"  {g1.mean():.2f}/{g2.mean():.2f}/{g3.mean():.2f}"
              f"     {w1:.2f}/{w3:.2f}")
        all_gaps["m1pure"] += g1.tolist()
        all_gaps["m2v"] += g2.tolist()
        all_gaps["madv"] += g3.tolist()
        all_walls["m1pure"] += [r["m1pure_wall"] for r in rs]
        all_walls["madv"] += [r["madv_wall"] for r in rs]
        per_topo[tname] = {"rows": rs,
                           "gap": {"m1pure": float(g1.mean()),
                                   "m2v": float(g2.mean()),
                                   "madv": float(g3.mean())}}

    print("\n===== pooled（配对）=====")
    n = min(len(all_gaps["m1pure"]), len(all_gaps["madv"]))
    for k in ["m1pure", "m2v", "madv"]:
        g = all_gaps[k]
        print(f"{k:<8} mean gap={np.mean(g):.3f}  median={np.median(g):.3f}  "
              f"gap=0 率={np.mean([x == 0 for x in g]):.2f}")
    # 配对显著性（M-A vs M1-pure 的 gap）
    g1 = np.array(all_gaps["m1pure"][:n])
    g3 = np.array(all_gaps["madv"][:n])
    print(f"\nM-A vs M1-pure: Δgap mean={np.mean(g3-g1):+.3f}  "
          f"p={wilcoxon_p(g3, g1):.4f}")
    g2 = np.array(all_gaps["m2v"][:n])
    print(f"M-A vs M2-V:   Δgap mean={np.mean(g3-g2):+.3f}  "
          f"p={wilcoxon_p(g3, g2):.4f}")
    if all_walls["m1pure"] and all_walls["madv"]:
        r = np.mean(all_walls["m1pure"]) / max(np.mean(all_walls["madv"]), 1e-9)
        print(f"episode 墙钟: M1={np.mean(all_walls['m1pure']):.2f}s "
          f"M-A={np.mean(all_walls['madv']):.2f}s  加速比={r:.1f}×")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"per_topo": per_topo,
                   "pooled_gaps": all_gaps}, f, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
