"""兄弟级 advantage 标注：对训练状态的全部后继打 M1-episode 标签。

动机（Gate L3 诊断）：learned V 绝对精度尚可（corr 0.94）但**动作分辨力缺失**
——兄弟状态间 ΔV(~1-2) 被 V_θ 噪声(4-9)淹没 → MCTS 根选择近随机 → 游走。
修复：给兄弟后继打标签，训练时加 pairwise ranking 损失，直接优化分辨力。

采样：训练电路中 y≥2 的状态，按族×档分层取 800 个父状态 × ~15 后继。
标签：M1-episode@100（8-10q 实证 = V*，16q 主配方）。

输出: benchmark/v0_sibling.npz（X_sib, y_sib, parent_id, parent_y, circuit）
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
from routing.v0.mcts import MCTSConfig, mcts_episode
from routing.v0.value_net import state_features


def _label_siblings(task):
    """worker：单父状态 → 全部后继的 (特征, 标签)。"""
    file_path, topo_path, mask, mapping, parent_y, label_sims = task
    try:
        import signal
        def _alarm(*_):
            raise TimeoutError
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, 360.0)
        topo, cm, hw = load_topo(topo_path)
        with open(file_path, "rb") as f:
            qc = pickle.load(f)
        dag = CircuitDAG.from_circuit(qc)
        env = make_env(dag, cm)
        env.set_state(mask, mapping)

        rows = []
        cfg = MCTSConfig(sims=label_sims, prior="sabre",
                         value="rollout_sabre", seed=0)
        for a, (m2, mp2) in env.all_successors():
            e2 = make_env(dag, cm)
            e2.set_state(m2, mp2)
            if e2.is_terminal():
                lab = 0.0
            else:
                n_t, ok_t, _, _ = mcts_episode(e2.clone(), cfg, max_steps=400)
                if not ok_t:
                    continue
                lab = float(n_t)
            rows.append((state_features(e2), lab, int(m2), list(mp2)))
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        return rows, parent_y, None
    except Exception as e:  # noqa: BLE001
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        except Exception:
            pass
        return [], float(parent_y), f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="benchmark/v0_vstar16.npz")
    ap.add_argument("--topo", default="traindata/topo/line_16q.json")
    ap.add_argument("--data-dir", default="traindata/v0")
    ap.add_argument("--n-parents", type=int, default=800)
    ap.add_argument("--label-sims", type=int, default=100)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--out", default="benchmark/v0_sibling.npz")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, y, cnames = d["X"], d["y"], d["circuit"]
    masks, mappings = d["mask"], d["mapping"]

    # 分层切分（与 run_gate_l3 一致），只取训练电路的状态
    strata = {}
    for c in sorted(set(cnames.tolist())):
        parts = c.replace(".pkl", "").split("_")
        fam = "perm_mix" if parts[0] == "perm" else parts[0]
        tier = parts[3] if parts[0] == "perm" else parts[2]
        strata.setdefault((fam, tier), []).append(c)
    rng = np.random.default_rng(0)
    test_set = set()
    for k, cs in sorted(strata.items()):
        cs = sorted(set(cs))
        rng.shuffle(cs)
        test_set.update(cs[:max(1, int(len(cs) * 0.2))])

    # 候选父状态：训练电路、y≥2，按族×档分层采样
    cand = collections = {}
    from collections import defaultdict
    cand = defaultdict(list)
    for i in range(len(X)):
        c = cnames[i]
        if c in test_set or y[i] < 2:
            continue
        parts = c.replace(".pkl", "").split("_")
        fam = "perm_mix" if parts[0] == "perm" else parts[0]
        tier = parts[3] if parts[0] == "perm" else parts[2]
        cand[(fam, tier)].append(i)
    per = max(1, args.n_parents // max(1, len(cand)))
    parents = []
    for k, idxs in sorted(cand.items()):
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        parents += idxs[:per].tolist()
    parents = parents[: args.n_parents]
    print(f"父状态 {len(parents)} 个（分层 {len(cand)} 组 × ~{per}）")

    tasks = [(os.path.join(args.data_dir, cnames[i]), args.topo,
              int(masks[i]), [int(v) for v in mappings[i]], float(y[i]),
              args.label_sims) for i in parents]

    Xs, ys, pid, py, cn = [], [], [], [], []
    t0 = time.perf_counter()
    n_fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_label_siblings, t) for t in tasks]
        for i, fut in enumerate(futs):
            try:
                rows, parent_y, err = fut.result(timeout=900)
            except Exception as e:  # noqa: BLE001
                rows, parent_y, err = [], -1, str(e)
            if err is not None:
                n_fail += 1
            else:
                for (feat, lab, m2, mp2) in rows:
                    Xs.append(feat)
                    ys.append(lab)
                    pid.append(i)
                    py.append(parent_y)
                    cn.append(cnames[parents[i]])
            if (i + 1) % 50 == 0 or i + 1 == len(tasks):
                print(f"[{i+1}/{len(tasks)}] 兄弟状态 {len(Xs)} "
                      f"(fail {n_fail}) ({time.perf_counter()-t0:.0f}s)", flush=True)

    Xs = np.asarray(Xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    pid = np.asarray(pid, dtype=np.int32)
    py = np.asarray(py, dtype=np.float32)
    cn = np.asarray(cn)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, X=Xs, y=ys, parent_id=pid, parent_y=py, circuit=cn)
    print(f"\n-> {args.out}: {len(Xs)} 兄弟状态 / {len(set(pid.tolist()))} 父 "
          f"(fail {n_fail})")


if __name__ == "__main__":
    main()
