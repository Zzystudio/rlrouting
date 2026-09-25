"""5-8q advantage 标签构建：A*(s,a) = 1 + C*(T(s,a)) - C*(s)（ExactSolver 精确）。

状态采集（与 v16 配方一致）：M1 轨迹（greedy hop）+ 随机游走 + 扰动前缀。
每个状态 s：C*(s) via solver；每个合法动作 a：C*(T(s,a)) via solver → A*。
特征：features(s') ⊕ features(s)（24 维；A* 不是 s' 单独的函数）。
同时输出父状态的 (features12, C*) 供 M2-V 对照训练。

输出: benchmark/v0_adv_labels.npz
  X24 (N,24), A (N,), parent_X12 (M,12), parent_C (M,), circuit, split, pid
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

ADV_TOPOS = [
    ("cross5", "traindata/topo/cross_5q.json", 5),
    ("ring5", "traindata/topo/ring_5q.json", 5),
    ("line5", "traindata/topo/ibmq_5_line.json", 5),
    ("line8", "traindata/topo/line_8q.json", 8),
    ("ring8", "traindata/topo/ring_8q.json", 8),
]


def _worker(task):
    (circuit_file, topo_path, seed) = task
    try:
        import signal
        def _alarm(*_):
            raise TimeoutError
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, 600.0)
        from routing.graph.circuit_dag import CircuitDAG
        from routing.v0.baselines import load_topo, make_env
        from routing.v0.exact_solver import ExactSolver
        from routing.v0.sabre_heuristic import SabreScorer, greedy_rollout
        from routing.v0.value_net import state_features
        topo, cm, hw = load_topo(topo_path)
        with open(circuit_file, "rb") as f:
            qc = pickle.load(f)
        dag = CircuitDAG.from_circuit(qc)
        solver = ExactSolver(dag, cm, max_nodes=2_000_000)
        rng = np.random.default_rng(seed)
        scorer = SabreScorer()

        states = []  # (mask, mapping)
        # M1 轨迹状态（hop-greedy）
        env = make_env(dag, cm)
        seen = {env.state_key()}
        states.append(env.state_key())
        while not env.is_terminal():
            e = scorer.best_action(env)
            env.step(e)
            if env.state_key() not in seen:
                seen.add(env.state_key())
                states.append(env.state_key())
        # 随机游走 3 条 × 4 步
        for w in range(3):
            env = make_env(dag, cm)
            for _ in range(4):
                if env.is_terminal():
                    break
                legal = env.legal_actions()
                env.step(int(legal[rng.integers(0, len(legal))]))
                if env.state_key() not in seen:
                    seen.add(env.state_key())
                    states.append(env.state_key())
        # 扰动前缀（8 步随机 + greedy 恢复路径上的状态）
        env = make_env(dag, cm)
        for _ in range(8):
            if env.is_terminal():
                break
            legal = env.legal_actions()
            env.step(int(legal[rng.integers(0, len(legal))]))
            if env.state_key() not in seen:
                seen.add(env.state_key())
                states.append(env.state_key())
        while not env.is_terminal():
            e = scorer.best_action(env)
            env.step(e)
            if env.state_key() not in seen:
                seen.add(env.state_key())
                states.append(env.state_key())

        # 标注：每个状态的全后继 A*
        X24, A, pid, pX, pC = [], [], [], [], []
        circuit = os.path.basename(circuit_file)
        n_states = 0
        for si, (mask, mapping) in enumerate(states):
            e = make_env(dag, cm)
            e.set_state(mask, mapping)
            cs = solver.solve(e.executed_mask, e.mapping)
            if cs is None:
                continue
            n_states += 1
            pf = state_features(e)
            pX.append(pf)
            pC.append(float(cs))
            for a, (m2, mp2) in e.all_successors():
                e2 = make_env(dag, cm)
                e2.set_state(m2, mp2)
                ct = solver.solve(e2.executed_mask, e2.mapping)
                if ct is None:
                    continue
                a_star = 1.0 + float(ct) - float(cs)
                if a_star < -1e-9:
                    raise ValueError(f"负 A* {a_star}（solver 不一致）")
                X24.append(np.concatenate([state_features(e2), pf]))
                A.append(max(0.0, a_star))
                pid.append(si)
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        return (np.array(X24, dtype=np.float32), np.array(A, dtype=np.float32),
                np.array(pid, dtype=np.int32), np.array(pX, dtype=np.float32),
                np.array(pC, dtype=np.float32), circuit, n_states, None)
    except Exception as e:  # noqa: BLE001
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        except Exception:
            pass
        return (np.zeros((0, 24), dtype=np.float32), np.zeros(0),
                np.zeros(0, dtype=np.int32), np.zeros((0, 12), dtype=np.float32),
                np.zeros(0), os.path.basename(circuit_file), 0,
                f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="traindata/v0")
    ap.add_argument("--per-topo", type=int, default=30, help="每拓扑电路数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--out", default="benchmark/v0_adv_labels.npz")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.data_dir, "manifest.json")))
    tasks = []
    meta = {}
    for tname, tpath, nq in ADV_TOPOS:
        circuits = [x["file"] for x in manifest if x["nq"] == nq][: args.per_topo]
        # 分层留出：每 (family, n2q) 最后 20% 为 test
        strata = {}
        for c in circuits:
            parts = c.replace(".pkl", "").split("_")
            fam = "perm_mix" if parts[0] == "perm" else parts[0]
            strata.setdefault((fam, parts[2]), []).append(c)
        rng = np.random.default_rng(args.seed)
        test = set()
        for k, cs in sorted(strata.items()):
            cs = sorted(set(cs))
            rng.shuffle(cs)
            test.update(cs[:max(1, int(len(cs) * 0.2))])
        for c in circuits:
            tasks.append((os.path.join(args.data_dir, c), tpath, args.seed))
        meta[tname] = {"circuits": circuits, "test": sorted(test)}
        print(f"{tname}: {len(circuits)} 电路（test {len(test)}）")

    X24, A, pid, pX, pC, cnames, splits, psplits = [], [], [], [], [], [], [], []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs)):
            x24, a, pi, px, pc, circuit, n_states, err = fut.result()
            if err:
                print(f"[{i+1}] {circuit}: {err}", flush=True)
                continue
            # 找回拓扑名（按电路的 nq）
            nq_c = int(circuit.split("_")[1].rstrip("q"))
            tname = [n for n, p, q in ADV_TOPOS if q == nq_c][0]
            is_test = circuit in meta[tname]["test"]
            for row in x24:
                X24.append(row)
            for v in a:
                A.append(v)
            for v in pi:
                pid.append(v + len(pC))
            for row in px:
                pX.append(row)
                psplits.append(1 if is_test else 0)
            for v in pc:
                pC.append(v)
            for _ in range(len(a)):
                cnames.append(circuit)
                splits.append(1 if is_test else 0)
            if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                print(f"[{i+1}/{len(tasks)}] X24={len(X24)} 父={len(pC)} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out,
             X24=np.asarray(X24, dtype=np.float32),
             A=np.asarray(A, dtype=np.float32),
             pid=np.asarray(pid, dtype=np.int32),
             parent_X12=np.asarray(pX, dtype=np.float32),
             parent_C=np.asarray(pC, dtype=np.float32),
             circuit=np.asarray(cnames),
             split=np.asarray(splits, dtype=np.int8),
             parent_split=np.asarray(psplits, dtype=np.int8))
    print(f"\n-> {args.out}: {len(X24)} (s,a) 对 / {len(pC)} 父状态")


if __name__ == "__main__":
    main()
