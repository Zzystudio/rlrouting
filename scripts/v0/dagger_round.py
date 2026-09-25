"""DAgger 一轮快速验证：用 M3 自身轨迹状态重打标签，消除价值幻觉循环。

动机（v16 Gate L3 诊断）：M3 的轨迹偏离标签分布（M1 轨迹），V_θ 在偏离状态上
系统性乐观（-9 步）→ argmax 停留在"幻觉好状态"→ episode 游走。
DAgger 教科书场景：用当前策略自身轨迹 + 专家（M1）标签扩充训练分布。

流程：
  1. 用当前 V_θ 跑 M3 episodes（训练电路子集），采集轨迹状态
  2. 对这些状态用 M1-episode@100 打标签（与主数据集同配方）
  3. 合并原 npz 重训 MLP（分层切分不变）
  4. 输出新模型 + 对比旧模型在 M3 轨迹状态上的乐观偏差

用法:
    PYTHONPATH=src python3 scripts/v0/dagger_round.py --circuits 12
"""

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from routing.graph.circuit_dag import CircuitDAG
from routing.v0.baselines import load_topo, make_env
from routing.v0.mcts import MCTSConfig, mcts_episode, mcts_search
from routing.v0.sabre_heuristic import SabreScorer, ensemble_rollout_value
from routing.v0.value_net import ValueNet, state_features, train_value_net


def _collect_traj(task):
    """worker：单电路 M3 轨迹状态采集 + M1 标签。"""
    file_path, topo_path, model_path, traj_sims, label_sims, max_states = task
    try:
        topo, cm, hw = load_topo(topo_path)
        with open(file_path, "rb") as f:
            qc = pickle.load(f)
        dag = CircuitDAG.from_circuit(qc)
        net = ValueNet(in_dim=12)
        net.load_state_dict(torch.load(model_path, map_location="cpu",
                                       weights_only=True))
        net.eval()

        env = make_env(dag, cm)
        cfg = MCTSConfig(sims=traj_sims, prior="sabre", value="learned", seed=0)
        states, steps = [], 0
        while not env.is_terminal() and steps < 400:
            states.append(env.state_key())
            a, _ = mcts_search(env, cfg, value_net=net)
            env.step(a)
            steps += 1

        rows = []
        scorer = SabreScorer()
        seen = set()
        # 采样：起点 + 每 3 步 + 结尾
        idxs = list(range(0, len(states), 3)) + ([len(states) - 1] if states else [])
        for i in sorted(set(idxs)):
            mask, mapping = states[i]
            if (mask, mapping) in seen:
                continue
            seen.add((mask, mapping))
            e = make_env(dag, cm)
            e.set_state(mask, mapping)
            if e.is_terminal():
                lab, sd = 0.0, 0.0
            else:
                cfg_l = MCTSConfig(sims=label_sims, prior="sabre",
                                   value="rollout_sabre", seed=0)
                n_t, ok_t, _, _ = mcts_episode(e.clone(), cfg_l, max_steps=400)
                if not ok_t:
                    continue
                _, sd, _ = ensemble_rollout_value(e, scorer, n=4, beta=5.0,
                                                  epsilon=0.02, seed=0)
                lab = float(n_t)
            rows.append((state_features(e), lab, sd, int(mask), list(mapping),
                         os.path.basename(file_path)))
        return rows, {"file": os.path.basename(file_path), "steps": steps,
                      "n_states": len(rows)}
    except Exception as e:  # noqa: BLE001
        return [], {"file": os.path.basename(file_path), "error": str(e)}


def _stratified_split(circuits, seed=0, test_frac=0.2):
    strata = {}
    for c in circuits:
        parts = c.replace(".pkl", "").split("_")
        fam = "perm_mix" if parts[0] == "perm" else parts[0]
        tier = parts[3] if parts[0] == "perm" else parts[2]
        strata.setdefault((fam, tier), []).append(c)
    rng = np.random.default_rng(seed)
    test = set()
    for k, cs in sorted(strata.items()):
        cs = sorted(set(cs))
        rng.shuffle(cs)
        test.update(cs[:max(1, int(len(cs) * test_frac))])
    return test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="benchmark/v0_vstar16.npz")
    ap.add_argument("--topo", default="traindata/topo/line_16q.json")
    ap.add_argument("--data-dir", default="traindata/v0")
    ap.add_argument("--model", default="models/v0_mlp16_strat.pt")
    ap.add_argument("--circuits", type=int, default=12)
    ap.add_argument("--traj-sims", type=int, default=100)
    ap.add_argument("--label-sims", type=int, default=100)
    ap.add_argument("--max-states", type=int, default=25)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-model", default="models/v0_mlp16_dagger.pt")
    args = ap.parse_args()

    # 分层切分（与 run_gate_l3 一致）
    manifest = json.load(open(os.path.join(args.data_dir, "manifest.json")))
    circuits16 = [x["file"] for x in manifest if x["nq"] == 16 and 8 <= x["n2q"] <= 30]
    test_set = _stratified_split(sorted(set(circuits16)))
    train_circuits = sorted(set(circuits16) - test_set)
    # 排除 chain（V*≈0 无偏离轨迹），聚焦 M3 实际游走的族
    train_circuits = [c for c in train_circuits
                      if not c.startswith("chain")][: args.circuits]
    print(f"DAgger 采集：{len(train_circuits)} 个训练电路")

    tasks = [(os.path.join(args.data_dir, c), args.topo, args.model,
              args.traj_sims, args.label_sims, args.max_states)
             for c in train_circuits]
    new_rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_collect_traj, t) for t in tasks]
        for i, fut in enumerate(futs):
            try:
                rows, stat = fut.result(timeout=1500)
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}] failed: {e}", flush=True)
                continue
            new_rows += rows
            print(f"  [{i+1}/{len(tasks)}] {stat.get('file','?')}: "
                  f"steps={stat.get('steps','?')} +{stat.get('n_states','?')} 状态",
                  flush=True)
    print(f"采集 {len(new_rows)} 个 DAgger 状态 ({time.perf_counter()-t0:.0f}s)")

    # 合并重训
    d = np.load(args.data, allow_pickle=True)
    X = list(d["X"]); y = list(d["y"]); cnames = list(d["circuit"])
    test_set = _stratified_split(sorted(set(cnames)))
    for (feat, lab, sd, mask, mapping, cf) in new_rows:
        X.append(feat); y.append(lab); cnames.append(cf)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    is_test = np.array([c in test_set for c in cnames])
    tr_idx = np.where(~is_test)[0]
    rng = np.random.default_rng(1)
    rng.shuffle(tr_idx)
    n_val = max(1, int(len(tr_idx) * 0.1))
    va_idx, tr_idx = tr_idx[:n_val], tr_idx[n_val:]
    print(f"重训：train {len(tr_idx)}（含 DAgger {len(new_rows)}） val {len(va_idx)}")

    net = ValueNet(in_dim=X.shape[1])
    net, hist = train_value_net(net, X[tr_idx], y[tr_idx], X[va_idx], y[va_idx],
                                epochs=800, device="cpu")
    net.eval()
    with torch.no_grad():
        p_va = net(torch.as_tensor(X[va_idx])).numpy()
    print(f"重训完成 val MAE={np.abs(p_va - y[va_idx]).mean():.2f}")
    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    torch.save(net.state_dict(), args.out_model)
    print(f"-> {args.out_model}")


if __name__ == "__main__":
    main()
