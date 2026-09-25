"""P2 数据集构建：V* 三分布（D_expert ∪ D_random ∪ D_search），A* 精确标注。

每个电路：
  - D_expert:  M1 轨迹（SABRE prior+rollout）+ 最优轨迹（沿 V* 单调下降）的状态
  - D_random:  起点随机 SWAP 游走（k 步）状态
  - D_search:  M1 episode 搜索访问状态（visited_log，含树中/叶子状态）+ 刻意次优前缀
所有状态用共享 memo 的 ExactSolver 标 V*(s)；超预算状态丢弃并计数。

输出: benchmark/v0_vstar.npz（X, y, split, dist, circuit）
      benchmark/v0_vstar_meta.json（每电路求解统计）
"""

import argparse
import json
import os
import pickle
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
import sys
sys.path.insert(0, os.path.join(ROOT, "src"))

from routing.graph.circuit_dag import CircuitDAG
from routing.v0.baselines import load_topo, make_env
from routing.v0.exact_solver import ExactSolver
from routing.v0.mcts import MCTSConfig, mcts_episode
from routing.v0.sabre_heuristic import SabreScorer
from routing.v0.value_net import state_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default="traindata/topo/line_8q.json")
    ap.add_argument("--data-dir", default="traindata/v0")
    ap.add_argument("--nq", type=int, default=8)
    ap.add_argument("--n2q-min", type=int, default=6)
    ap.add_argument("--n2q-max", type=int, default=12)
    ap.add_argument("--family", default="perm_mix")
    ap.add_argument("--train-n", type=int, default=8)
    ap.add_argument("--test-n", type=int, default=3, help="留出电路（OOD）")
    ap.add_argument("--random-walk", type=int, default=20, help="每电路随机游走状态数")
    ap.add_argument("--mcts-sims", type=int, default=50)
    ap.add_argument("--max-nodes", type=int, default=4_000_000)
    ap.add_argument("--out", default="benchmark/v0_vstar.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    topo, cm, hw = load_topo(args.topo)
    manifest = json.load(open(os.path.join(args.data_dir, "manifest.json")))
    circuits = [x for x in manifest if x["nq"] == args.nq
                and args.n2q_min <= x["n2q"] <= args.n2q_max
                and x["family"] == args.family]
    circuits = circuits[: args.train_n + args.test_n]
    if len(circuits) < args.train_n + args.test_n:
        print(f"[warn] 电路不足: {len(circuits)} < train+test")
    train_set = set(c["file"] for c in circuits[:args.train_n])
    test_set = set(c["file"] for c in circuits[args.train_n:])

    rng = np.random.default_rng(args.seed)
    X, y, split, dist, cnames = [], [], [], [], []
    meta = {"skipped": 0, "unsolvable": [], "per_circuit": {}}

    t_start = time.perf_counter()
    for ci, meta_c in enumerate(circuits):
        with open(os.path.join(args.data_dir, meta_c["file"]), "rb") as f:
            qc = pickle.load(f)
        dag = CircuitDAG.from_circuit(qc)
        solver = ExactSolver(dag, cm, max_nodes=args.max_nodes)
        is_train = meta_c["file"] in train_set
        n0 = len(X)
        pc = {"expert": 0, "random": 0, "search": 0, "vstar": None}

        env0 = make_env(dag, cm)
        vstar = solver.solve(env0.executed_mask, env0.mapping)
        if vstar is None:
            meta["unsolvable"].append(meta_c["file"])
            print(f"[skip] {meta_c['file']}: V* 超预算")
            continue
        pc["vstar"] = vstar

        def add_state(mask, mapping, dlabel):
            env = make_env(dag, cm)
            env.set_state(mask, mapping)
            v = solver.solve(mask, mapping)
            if v is None:
                meta["skipped"] += 1
                return
            X.append(state_features(env))
            y.append(float(v))
            split.append(0 if is_train else 1)
            dist.append(dlabel)
            cnames.append(meta_c["file"])

        # -- D_expert: M1 轨迹 --
        env = make_env(dag, cm)
        cfg1 = MCTSConfig(sims=20, prior="sabre", value="rollout_sabre", seed=0)
        n, ok, _, _ = mcts_episode(env, cfg1, max_steps=4000)
        if ok:
            env = make_env(dag, cm)
            seen = set()
            while not env.is_terminal():
                if env.state_key() not in seen:
                    seen.add(env.state_key())
                    add_state(env.executed_mask, env.mapping, 0)
                e = SabreScorer().best_action(env)
                env.step(e)
        # -- D_expert: 最优轨迹（沿 V* 下降） --
        env = make_env(dag, cm)
        v = vstar
        seen = set()
        while v > 0:
            if env.state_key() not in seen:
                seen.add(env.state_key())
                add_state(env.executed_mask, env.mapping, 0)
            chosen = None
            for a, (m2, mp2) in env.all_successors():
                v2 = solver.solve(m2, mp2)
                if v2 is not None and v2 == v - 1:
                    chosen = a
                    break
            if chosen is None:
                break
            env.step(chosen)
            v -= 1

        # -- D_random: 随机游走 --
        env = make_env(dag, cm)
        seen = set()
        for _ in range(args.random_walk):
            if env.is_terminal():
                break
            legal = env.legal_actions()
            env.step(int(legal[rng.integers(0, len(legal))]))
            if env.state_key() not in seen:
                seen.add(env.state_key())
                add_state(env.executed_mask, env.mapping, 1)

        # -- D_search: M1 搜索访问状态 + 次优前缀 --
        # 次优前缀：随机游走前段 3-6 步的"坏状态"
        env = make_env(dag, cm)
        for step in range(6):
            legal = env.legal_actions()
            env.step(int(legal[rng.integers(0, len(legal))]))
            if not env.is_terminal():
                add_state(env.executed_mask, env.mapping, 2)

        # M1 episode 搜索访问状态（树中/叶子状态）
        env = make_env(dag, cm)
        cfg3 = MCTSConfig(sims=args.mcts_sims, prior="sabre", value="rollout_sabre", seed=0)
        _n, _ok, _, vlog = mcts_episode(env, cfg3, max_steps=4000, collect_log=True)
        seen = set()
        for (mask, mapping), _v in (vlog or []):
            if (mask, mapping) in seen:
                continue
            seen.add((mask, mapping))
            add_state(mask, mapping, 2)

        added = len(X) - n0
        pc["total"] = added
        meta["per_circuit"][meta_c["file"]] = pc
        print(f"[{ci+1}/{len(circuits)}] {meta_c['file']}: V*={vstar} "
              f"{'TRAIN' if is_train else 'TEST'} 新增 {added} 状态 "
              f"(累计 {len(X)}), {time.perf_counter()-t_start:.0f}s")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    split = np.asarray(split, dtype=np.int8)
    dist = np.asarray(dist, dtype=np.int8)
    cnames = np.asarray(cnames)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, X=X, y=y, split=split, dist=dist, circuit=cnames)
    with open(args.out.replace(".npz", "_meta.json"), "w") as f:
        json.dump({"topo": args.topo, "args": vars(args), **meta}, f, indent=2,
                  default=str)
    print(f"\n-> {args.out}: {len(X)} 状态 "
          f"(train {int((split==0).sum())}, test {int((split==1).sum())})")
    print(f"   dist 分布: expert={int((dist==0).sum())} "
          f"random={int((dist==1).sum())} search={int((dist==2).sum())}")


if __name__ == "__main__":
    main()
