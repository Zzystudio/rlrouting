"""Sprint 1 并行矩阵 runner —— 实例×方法×sims 任务级并行，JSONL 续跑。

方法：
  sabre     : 外部 qiskit SabreSwap（identity 布局，trials=20，多 seed 报 mean/best）
  gdist     : 纯距离贪心
  gsabre    : SABRE-score 贪心 rollout
  m0@sims   : uniform prior + 截断随机 rollout（cap=200）
  m1@sims   : SABRE prior + SABRE rollout
  m1long@sims: 同 m1（高 sims 近似上界）

用法:
    PYTHONPATH=src python3 scripts/v0/run_matrix_parallel.py \
        --topo traindata/topo/line_16q.json --nq 16 \
        --methods "sabre,gdist,gsabre,m0@20,m1@20,m1@100,m1@1000" \
        --out benchmark/v0_s1_line16.jsonl --workers 100
"""

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
import sys
sys.path.insert(0, os.path.join(ROOT, "src"))

from routing.v0.baselines import load_topo, make_env
from routing.v0.mcts import MCTSConfig, mcts_episode


def _run_task(task):
    """worker：单实例×单方法×单sims。task 全部为可 pickle 参数。"""
    file_path, method, sims, topo_path, sabre_seeds, sabre_trials = task
    try:
        with open(file_path, "rb") as f:
            qc = pickle.load(f)
        topo, cm, hw = load_topo(topo_path)
        from routing.graph.circuit_dag import CircuitDAG
        dag = CircuitDAG.from_circuit(qc)
        t0 = time.perf_counter()

        if method == "sabre":
            from routing.v0.baselines import sabre_num_swaps
            vals = []
            for s in range(sabre_seeds):
                n, ms, _ = sabre_num_swaps(qc, cm, trials=sabre_trials, seed=s,
                                           initial_layout=list(range(qc.num_qubits)))
                vals.append(n)
            res = {"swaps": float(sum(vals)) / len(vals), "ok": True,
                   "extra": {"sabre_all": vals,
                             "sabre_best_seed": float(min(vals))}}
        elif method == "gdist":
            from routing.v0.baselines import greedy_dist_policy, run_episode
            env = make_env(dag, cm)
            n, ok, _ = run_episode(env, greedy_dist_policy)
            res = {"swaps": float(n), "ok": ok, "extra": {}}
        elif method == "gsabre":
            from routing.v0.sabre_heuristic import SabreScorer, greedy_rollout
            env = make_env(dag, cm)
            n, ok = greedy_rollout(env, SabreScorer())
            res = {"swaps": float(n), "ok": ok, "extra": {}}
        elif method.startswith("m0@"):
            s = int(method.split("@")[1])
            cfg = MCTSConfig(sims=s, prior="uniform", value="rollout_random",
                             rollout_cap=200, seed=0)
            env = make_env(dag, cm)
            # m0 在 16q 病理（随机 rollout 无价值信号→游走）：有界运行，
            # 超 400 步即记失败（诊断信号本身，v0@8q 已证其不可用）
            n, ok, st, _ = mcts_episode(env, cfg, max_steps=400)
            res = {"swaps": float(n), "ok": ok, "extra": {"expansions": st.get("expansions", -1)}}
        elif method.startswith("m1@"):
            s = int(method.split("@")[1])
            cfg = MCTSConfig(sims=s, prior="sabre", value="rollout_sabre", seed=0)
            env = make_env(dag, cm)
            # cap=1500：循环实例上低 sims 会游走，超限即失败信号（否则 m1@1000 可跑数小时）
            n, ok, st, _ = mcts_episode(env, cfg, max_steps=600)
            res = {"swaps": float(n), "ok": ok, "extra": {"expansions": st.get("expansions", -1)}}
        else:
            raise ValueError(method)

        wall = (time.perf_counter() - t0) * 1000.0
        import re
        m = re.match(r"^(.*)_(\d+)q_(\d+)g_s(\d+)\.pkl$",
                     os.path.basename(file_path))
        fam, nq, n2q = m.group(1), int(m.group(2)), int(m.group(3))
        return {"file": os.path.basename(file_path), "family": fam, "nq": nq,
                "n2q": n2q, "method": method, "sims": sims,
                "swaps": res["swaps"], "ok": res["ok"], "wall_ms": wall,
                "extra": res["extra"], "error": None}
    except Exception as e:  # noqa: BLE001
        return {"file": os.path.basename(file_path), "method": method, "sims": sims,
                "swaps": None, "ok": False, "wall_ms": -1.0, "extra": {},
                "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default="traindata/topo/line_16q.json")
    ap.add_argument("--data-dir", default="traindata/v0")
    ap.add_argument("--nq", type=int, default=16)
    ap.add_argument("--n2q-min", type=int, default=8)
    ap.add_argument("--n2q-max", type=int, default=30)
    ap.add_argument("--family", default=None, help="None=全部族")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--methods", default="sabre,gdist,gsabre,m0@20,m1@20,m1@100")
    ap.add_argument("--sabre-seeds", type=int, default=5)
    ap.add_argument("--sabre-trials", type=int, default=20)
    ap.add_argument("--out", default="benchmark/v0_s1.jsonl")
    ap.add_argument("--workers", type=int, default=100)
    args = ap.parse_args()

    topo, cm, hw = load_topo(args.topo)
    manifest = json.load(open(os.path.join(args.data_dir, "manifest.json")))
    circuits = [x for x in manifest if x["nq"] == args.nq
                and args.n2q_min <= x["n2q"] <= args.n2q_max]
    if args.family:
        circuits = [x for x in circuits if x["family"] == args.family]
    if args.limit:
        circuits = circuits[:args.limit]

    methods = [m.strip() for m in args.methods.split(",")]
    tasks = []
    for c in circuits:
        for m in methods:
            sims = int(m.split("@")[1]) if "@" in m else 0
            tasks.append((os.path.join(args.data_dir, c["file"]), m, sims,
                          args.topo, args.sabre_seeds, args.sabre_trials))
    print(f"{len(circuits)} 实例 × {len(methods)} 方法 = {len(tasks)} 任务")

    # 续跑：读取已有 JSONL，跳过已完成 (file, method, sims)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["file"], r["method"], r["sims"]))
                except Exception:
                    pass
    tasks = [t for t in tasks if (os.path.basename(t[0]), t[1], t[2]) not in done]
    print(f"续跑跳过 {len(done)}，待跑 {len(tasks)}")

    t_start = time.perf_counter()
    n_ok = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_run_task, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            with open(args.out, "a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if r["ok"] and r["swaps"] is not None:
                n_ok += 1
            if (i + 1) % 50 == 0:
                el = time.perf_counter() - t_start
                print(f"[{i+1}/{len(tasks)}] ok={n_ok} elapsed={el:.0f}s "
                      f"({el/(i+1)*1000:.0f}ms/task)", flush=True)
    print(f"完成 {len(tasks)} 任务，成功 {n_ok}，总耗时 "
          f"{time.perf_counter()-t_start:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
