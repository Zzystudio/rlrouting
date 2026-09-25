"""N1.2 端到端评估：M1-pure vs M1-noise vs SABRE，双指标（SWAPs + v3 保真度）。

深度匹配（fid 可测区间）：t287 族 40-100 门 / t176 16-40 门 / ring-grid16 10-25 门。
方法：
  sabre    外部 Qiskit SabreSwap（identity 布局，trials=20，5 seeds mean）
  m1pure   MCTS(sims=100) + SABRE prior + hop-rollout value（v16 交付配置）
  m1noise  MCTS(sims=100) + SABRE prior + 噪声感知 rollout value（N1）
保真度：trajectory_v3（CPU 后端，64 traj），identity 布局。

用法:
    PYTHONPATH=src python3 -u scripts/v0/run_n1_eval.py --per-topo 20 --workers 24
"""

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

TOPOS = [
    ("t287_unif", "traindata/topo/tianyan287_20q_unifnoise.json", 20, (40, 100)),
    ("t287_base", "traindata/topo/tianyan287_20q.json", 20, (40, 100)),
    ("t287_big", "traindata/topo/tianyan287_20q_bighetero.json", 20, (40, 100)),
    ("t176", "traindata/topo/tianyan176_20q.json", 20, (16, 40)),
    ("ring16", "traindata/topo/ring_16q.json", 16, (10, 25)),
    ("grid16", "traindata/topo/grid_4x4_16q.json", 16, (10, 25)),
]


def _worker(task):
    (circuit_file, topo_path, n_traj) = task
    try:
        import signal
        def _alarm(*_):
            raise TimeoutError
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, 1200.0)
        from routing.graph.circuit_dag import CircuitDAG
        from routing.v0.baselines import (load_topo_full, make_env,
                                          sabre_num_swaps)
        from routing.v0.mcts import MCTSConfig, mcts_episode
        from routing.v0.noise_rollout import (noise_aware_rollout,
                                              phys_circuit_from_ops)
        from sim.trajectory_sim_v3 import trajectory_circuit_fidelity_events_v3
        topo, cm, hw, config = load_topo_full(topo_path)
        with open(circuit_file, "rb") as f:
            qc = pickle.load(f)
        dag = CircuitDAG.from_circuit(qc)
        n_phys = max(max(e) for e in cm) + 1
        out = {"circuit": os.path.basename(circuit_file)}

        # --- SABRE（外部）---
        sabre_list = [sabre_num_swaps(qc, cm, trials=20, seed=s,
                                      initial_layout=list(range(qc.num_qubits)))[0]
                      for s in range(5)]
        out["sabre_swaps"] = float(np.mean(sabre_list))
        # SABRE 物理电路（用 routing.sabre_route 重建，identity 布局）
        from routing.routing import sabre_route
        from routing.v0.baselines import _topo_to_config

        class _C:  # 轻量 config 适配（sabre_route 只用 coupling_map）
            pass
        _c = _C(); _c.coupling_map = cm
        phys_s, sinfo = sabre_route(qc, _c, swap_trials=20, seed=0)
        fid_s = trajectory_circuit_fidelity_events_v3(
            phys_s, config, num_trajectories=n_traj, seed=0, backend="cpu")
        out["sabre_fid"] = float(fid_s)

        # --- M1-pure ---
        env = make_env(dag, cm)
        cfg_p = MCTSConfig(sims=100, prior="sabre", value="rollout_sabre", seed=0)
        n_p, ok_p, _, _ = mcts_episode(env, cfg_p, max_steps=1500)
        out["m1pure_swaps"] = float(n_p) if ok_p else None
        out["m1pure_ok"] = ok_p
        if ok_p:
            # 重放路由构造物理电路（用噪声 rollout 的 hop 模式重走同一策略？）
            # 简化：用 hop-greedy 重放不可行——直接用 MCTS 的动作序列重放
            # 这里用噪声时间线记录器重跑 MCTS 决策序（确定性同 seed）
            pass
        # --- M1-noise ---
        env = make_env(dag, cm)
        cfg_n = MCTSConfig(sims=100, prior="sabre", value="noise", seed=0)
        dist_noise = hw.dist_noise(1.0) * hw.num_qubits
        n_ctx = {"config": config, "dist_noise": dist_noise}
        n_n, ok_n, _, _ = mcts_episode(env, cfg_n, value_net=None,
                                       noise_ctx=n_ctx, max_steps=1500)
        out["m1noise_swaps"] = float(n_n) if ok_n else None
        out["m1noise_ok"] = ok_n

        # --- 物理电路重建与保真度（M1 两法共用重放器）---
        # 重放：按各自 MCTS 决策序走 env，NoiseTimeline 记账 → 物理电路
        def replay(mcts_cfg, value_kw):
            e = make_env(dag, cm)
            tl_ops = []
            steps = 0
            while not e.is_terminal() and steps < 1500:
                a, _ = mcts_search_wrap(e, mcts_cfg, value_kw)
                pq = e.coupling_map[a]
                tl_ops.append(("swap", (pq[0], pq[1])))
                mask_before = e.executed_mask
                e.step(a)
                steps += 1
                newly = e.executed_mask & ~mask_before
                for g in e._twoq:
                    if newly & (1 << g):
                        gate = e.dag.gates[g]
                        q0, q1 = gate.qubits
                        tl_ops.append(("cx", (e.mapping[q0], e.mapping[q1])))
            return tl_ops, e.is_terminal()

        def mcts_search_wrap(e, cfg, value_kw):
            from routing.v0.mcts import mcts_search
            return mcts_search(e, cfg, **value_kw)

        # M1-pure 物理电路（hop scorer 重放）
        from routing.v0.noise_rollout import NoiseTimeline
        tl_ops_p, ok_replay_p = replay(
            MCTSConfig(sims=100, prior="sabre", value="rollout_sabre", seed=0),
            {})
        if ok_replay_p:
            phys_p = phys_circuit_from_ops(tl_ops_p, n_phys)
            out["m1pure_fid"] = float(trajectory_circuit_fidelity_events_v3(
                phys_p, config, num_trajectories=n_traj, seed=0, backend="cpu"))
        # M1-noise 物理电路
        tl_ops_n, ok_replay_n = replay(
            MCTSConfig(sims=100, prior="sabre", value="noise", seed=0),
            {"noise_ctx": n_ctx})
        if ok_replay_n:
            phys_n = phys_circuit_from_ops(tl_ops_n, n_phys)
            out["m1noise_fid"] = float(trajectory_circuit_fidelity_events_v3(
                phys_n, config, num_trajectories=n_traj, seed=0, backend="cpu"))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="traindata/v0")
    ap.add_argument("--per-topo", type=int, default=20)
    ap.add_argument("--n-traj", type=int, default=64)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default="benchmark/v0_n1_eval.json")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.data_dir, "manifest.json")))
    tasks = []
    for tname, tpath, nq, (dmin, dmax) in TOPOS:
        circuits = [x["file"] for x in manifest
                    if x["nq"] == nq and dmin <= x["n2q"] <= dmax][: args.per_topo]
        for c in circuits:
            tasks.append((os.path.join(args.data_dir, c), tpath, args.n_traj))
    print(f"任务: {len(tasks)}（{len(TOPOS)} 拓扑 × {args.per_topo} 电路）")
    t0 = time.perf_counter()

    all_rows = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_worker, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            tname = [n for n, p, q, d in TOPOS if p == futs[fut][1]][0]
            all_rows.setdefault(tname, []).append(r)
            done += 1
            if done % 10 == 0:
                print(f"[{done}/{len(tasks)}] ({time.perf_counter()-t0:.0f}s)",
                      flush=True)

    # 汇总
    print("\n===== N1.2 端到端（双指标）=====")
    summary = {}
    for tname, tpath, nq, _dr in TOPOS:
        rows = [r for r in all_rows.get(tname, []) if r.get("error") is None]
        if not rows:
            print(f"{tname}: 无数据")
            continue
        def col(key):
            return np.array([r[key] for r in rows if r.get(key) is not None])
        line = f"{tname:<10} n={len(rows):>3}\n"
        for m in ["sabre", "m1pure", "m1noise"]:
            s = col(f"{m}_swaps")
            f = col(f"{m}_fid")
            if len(s) and len(f):
                line += (f"  {m:<8} swaps={s.mean():6.1f}  fid={f.mean():.4f} "
                         f"(median {np.median(f):.4f})\n")
        # 配对比较
        pairs = [("m1noise", "m1pure"), ("m1noise", "sabre"), ("m1pure", "sabre")]
        for a, b in pairs:
            fa = col(f"{a}_fid"); fb = col(f"{b}_fid")
            n = min(len(fa), len(fb))
            if n:
                gain = float(np.mean(fa[:n] - fb[:n]))
                win = float((fa[:n] > fb[:n]).mean())
                line += f"  Δfid {a}−{b} = {gain:+.4f} (win {win:.2f})\n"
        print(line)
        summary[tname] = {"rows": rows}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary}, f, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
