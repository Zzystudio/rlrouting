"""Sprint 2 标签数据集构建（电路级并行）：M1 episode 标签。

标签配方（validate_labels 阶梯结论）：
  - 轨迹状态（episode 路径上）：label(s_i) = D - i（episode 总步 D，恰好=剩余）
  - 树内/偏离状态（搜索访问态）：从该状态重跑 M1 episode @tree-sims 得 label
  - 不确定性特征：N=8 随机化 SABRE rollout 集成 std（β=5, ε=0.02）
  - 状态特征：12 维手工特征（value_net.state_features）

并行：电路级 ProcessPoolExecutor（worker 处理单电路，返回行列表）。

输出: benchmark/v0_vstar16.npz（X, y, std, dist, circuit, split）
      dist: 0=expert(轨迹) 1=random 2=search(树内)
"""

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from routing.graph.circuit_dag import CircuitDAG
from routing.v0.baselines import load_topo, make_env
from routing.v0.mcts import MCTSConfig, mcts_episode, mcts_search
from routing.v0.sabre_heuristic import SabreScorer, ensemble_rollout_value
from routing.v0.value_net import state_features


def _process_circuit(args):
    """worker：单电路 → (rows, stats)。args 全部可 pickle。"""
    file_path, topo_path, traj_sims, tree_sims, tree_states, random_walk, seed = args
    import signal
    def _hard_alarm(*_):
        raise TimeoutError("circuit hard timeout")
    signal.signal(signal.SIGALRM, _hard_alarm)
    signal.setitimer(signal.ITIMER_REAL, 400.0)
    try:
        topo, cm, hw = load_topo(topo_path)
        with open(file_path, "rb") as f:
            qc = pickle.load(f)
        dag = CircuitDAG.from_circuit(qc)
        rng = np.random.default_rng(seed)
        scorer = SabreScorer()
        rows = []

        def relabel(mask, mapping):
            e = make_env(dag, cm)
            e.set_state(mask, mapping)
            cfg_t = MCTSConfig(sims=tree_sims, prior="sabre",
                               value="rollout_sabre", seed=0)
            # 单状态 25s 超时：坏状态上 episode 可拖 140s+，跳过而非阻塞
            import signal
            def _alarm(*_):
                raise TimeoutError("relabel timeout")
            signal.signal(signal.SIGALRM, _alarm)
            signal.setitimer(signal.ITIMER_REAL, 25.0)
            try:
                n_t, ok_t, _, _ = mcts_episode(e.clone(), cfg_t, max_steps=200)
            except TimeoutError:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                return None
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
            if not ok_t:
                return None
            _, sd, _ = ensemble_rollout_value(e, scorer, n=8, beta=5.0,
                                              epsilon=0.02, seed=0)
            return n_t, sd

        # ---- 主 episode：记录状态序列 + 搜索访问态 ----
        cfg = MCTSConfig(sims=traj_sims, prior="sabre",
                         value="rollout_sabre", seed=0)
        env = make_env(dag, cm)
        visited_log = []
        states, steps, t0 = [], 0, time.perf_counter()
        while not env.is_terminal() and steps < 1500:
            if time.perf_counter() - t0 > 150.0:
                return rows, {"file": os.path.basename(file_path), "ok": False,
                              "reason": "timeout"}
            states.append(env.state_key())
            a, _ = mcts_search(env, cfg, visited_log=visited_log)
            env.step(a)
            steps += 1
        if not env.is_terminal():
            return rows, {"file": os.path.basename(file_path), "ok": False,
                          "reason": "episode_cap"}

        # D_expert：轨迹状态 label = 剩余步数
        for i, (mask, mapping) in enumerate(states):
            e = make_env(dag, cm)
            e.set_state(mask, mapping)
            _, sd, _ = ensemble_rollout_value(e, scorer, n=8, beta=5.0,
                                              epsilon=0.02, seed=0)
            rows.append((state_features(e), float(steps - i), sd, 0,
                         os.path.basename(file_path), int(mask), list(mapping)))

        # D_random：随机游走（少数）
        env_r = make_env(dag, cm)
        seen = set()
        n_random = 0
        for _ in range(random_walk):
            if env_r.is_terminal():
                break
            legal = env_r.legal_actions()
            env_r.step(int(legal[rng.integers(0, len(legal))]))
            if env_r.state_key() in seen:
                continue
            seen.add(env_r.state_key())
            lab = relabel(env_r.executed_mask, env_r.mapping)
            if lab is not None:
                rows.append((state_features(env_r), float(lab[0]), lab[1], 1,
                             os.path.basename(file_path),
                             int(env_r.executed_mask), list(env_r.mapping)))
                n_random += 1

        # D_search：树内访问态
        cnt = 0
        seen = set()
        for (mask, mapping), _v in visited_log:
            if cnt >= tree_states:
                break
            if (mask, mapping) in seen:
                continue
            seen.add((mask, mapping))
            lab = relabel(mask, mapping)
            if lab is not None:
                e = make_env(dag, cm)
                e.set_state(mask, mapping)
                rows.append((state_features(e), float(lab[0]), lab[1], 2,
                             os.path.basename(file_path), int(mask), list(mapping)))
                cnt += 1

        signal.setitimer(signal.ITIMER_REAL, 0.0)
        return rows, {"file": os.path.basename(file_path), "ok": True,
                      "steps": steps, "n_rows": len(rows)}
    except Exception as e:  # noqa: BLE001
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        return [], {"file": os.path.basename(file_path), "ok": False,
                    "reason": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default="traindata/topo/line_16q.json")
    ap.add_argument("--data-dir", default="traindata/v0")
    ap.add_argument("--nq", type=int, default=16)
    ap.add_argument("--n2q-min", type=int, default=8)
    ap.add_argument("--n2q-max", type=int, default=30)
    ap.add_argument("--family", default=None)
    ap.add_argument("--train-n", type=int, default=100)
    ap.add_argument("--test-n", type=int, default=20)
    ap.add_argument("--traj-sims", type=int, default=100)
    ap.add_argument("--tree-sims", type=int, default=100)
    ap.add_argument("--tree-states", type=int, default=40)
    ap.add_argument("--random-walk", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--out", default="benchmark/v0_vstar16.npz")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.data_dir, "manifest.json")))
    circuits = [x for x in manifest if x["nq"] == args.nq
                and args.n2q_min <= x["n2q"] <= args.n2q_max]
    if args.family:
        circuits = [x for x in circuits if x["family"] == args.family]
    circuits = circuits[: args.train_n + args.test_n]
    train_set = set(c["file"] for c in circuits[:args.train_n])

    tasks = [(os.path.join(args.data_dir, c["file"]), args.topo,
              args.traj_sims, args.tree_sims, args.tree_states,
              args.random_walk, args.seed) for c in circuits]
    print(f"{len(circuits)} 电路, workers={args.workers}")

    X, y, stds, dist, cnames, split = [], [], [], [], [], []
    masks, mappings = [], []
    t_start = time.perf_counter()
    n_ok = 0
    deadline = time.perf_counter() + 2400.0  # 40 分钟墙钟上限：防止死 worker 卡死
    pool = ProcessPoolExecutor(max_workers=args.workers)
    try:
        futs = [pool.submit(_process_circuit, t) for t in tasks]
        pending = list(futs)
        i = 0
        while pending and time.perf_counter() < deadline:
            done, pending = wait(pending, timeout=30)
            if not done:
                continue
            for fut in done:
                i += 1
                try:
                    rows, stat = fut.result()
                except Exception as e:  # noqa: BLE001 死 worker：记失败不阻塞
                    stat = {"file": "unknown", "ok": False,
                            "reason": f"future_failed: {e}"}
                    rows = []
                if stat["ok"]:
                    n_ok += 1
                    for (feat, lab, sd, dl, cf, msk, mpg) in rows:
                        X.append(feat)
                        y.append(lab)
                        stds.append(sd)
                        dist.append(dl)
                        cnames.append(cf)
                        masks.append(msk)
                        mappings.append(mpg)
                        split.append(0 if cf in train_set else 1)
                if i % 10 == 0 or i == len(tasks):
                    print(f"[{i}/{len(tasks)}] ok={n_ok} 累计 {len(X)} 状态 "
                          f"({time.perf_counter()-t_start:.0f}s)", flush=True)
        if pending:
            print(f"[deadline] 40min 上限，剩余 {len(pending)} 电路未完成，"
                  f"保存部分数据", flush=True)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    stds = np.asarray(stds, dtype=np.float32)
    dist = np.asarray(dist, dtype=np.int8)
    split = np.asarray(split, dtype=np.int8)
    cnames = np.asarray(cnames)
    masks = np.asarray(masks, dtype=np.int64)
    mappings = np.asarray(mappings, dtype=np.int32)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, X=X, y=y, std=stds, dist=dist, circuit=cnames,
             split=split, mask=masks, mapping=mappings)
    print(f"\n-> {args.out}: {len(X)} 状态 | dist: "
          f"expert={int((dist==0).sum())} random={int((dist==1).sum())} "
          f"search={int((dist==2).sum())} | train={int((split==0).sum())} "
          f"test={int((split==1).sum())}")


if __name__ == "__main__":
    main()
