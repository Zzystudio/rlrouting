#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R0：仲裁蒸馏数据集（残差学习，绑定 Layer 1 调度）。

对 routed 电路跑 ASAP episode，单遍收集决策点（|legal_exec|>1）的克隆，
按时钟分位抽样后分支：每个候选 a ∈ legal[:top_cand]
    clone → step(a) → 继续 ASAP 到终点 → fid_fn(同实例、同 seed=0)
  Δ(a) = Q(a) − Q(legal[0])     ← 1-ply 前瞻残差（贪心续基线）
  analytic(a) = 解析 fid(step(a) 后的前缀状态)  ← 基线对照

CRN 配对：所有候选 + Q0 用同一 fid_fn（seed=0 固定）→ 差分降噪。

用法（项目根）：
  PYTHONPATH=src:. python3 scripts/build_residual_dataset.py \
    --topo traindata/topo/tianyan287_20q.json \
    --nam-dirs traindata/gen_structured_v2,traindata/gen_structured_v3 \
    --out-dir traindata/residual --shard 0/3 --sim-device cuda:0 --limit 300
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing.rl.eval_policy import load_topo
from routing.rl.train_agent import _make_sched_analytic_fn
from sim.trajectory_sim_v3 import make_event_fidelity_fn_v3
from scripts.sched_common import (routed_cache_path, load_routed_cache,
                                  make_sched_env)

MAX_STEPS = 4000


def run_asap_from(env, max_steps=MAX_STEPS):
    """从当前 env 状态（不 reset）继续 ASAP 到终点；返回 (done, steps)。"""
    done = False
    steps = 0
    E, K = env.num_edges, env.max_ready
    while not done and steps < max_steps:
        env._update_candidates()
        mask = env.get_action_mask()
        legal = [i for i in range(E, E + K) if mask[i]]
        a = legal[0] if legal else env.skip_action
        try:
            _, _, done, trunc, _ = env.step(a, compute_obs=False)
        except RuntimeError:
            return False, steps
        done = done or trunc
        steps += 1
    return done, steps


def iter_split_circuits(data_dir, split):
    sp = os.path.join(data_dir, "splits", f"{split}.txt")
    if not os.path.exists(sp):
        sp = split
    out = []
    with open(sp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ap = line if os.path.isabs(line) else os.path.join(data_dir, line)
            if os.path.exists(ap):
                out.append((line, ap))
    return out


def collect_decision_points(env):
    """ASAP episode 单遍收集决策点克隆（不分支，主轨迹不受扰）。"""
    dp = []
    steps = 0
    done = False
    E, K = env.num_edges, env.max_ready
    env.reset()
    while not done and steps < MAX_STEPS:
        env._update_candidates()
        mask = env.get_action_mask()
        legal = [i for i in range(E, E + K) if mask[i]]
        if len(legal) > 1:
            dp.append((env.clone(), steps, env.clock, legal))
        a = legal[0] if legal else env.skip_action
        try:
            _, _, done, trunc, _ = env.step(a, compute_obs=False)
        except RuntimeError:
            return dp, steps, False
        done = done or trunc
        steps += 1
    return dp, steps, done


def stratify_select(dp, max_dp):
    """按时钟等距抽样 ≤ max_dp 个决策点。"""
    if len(dp) <= max_dp:
        return dp
    order = sorted(range(len(dp)), key=lambda i: dp[i][2])
    sel = []
    for b in range(max_dp):
        lo = b * len(dp) // max_dp
        hi = (b + 1) * len(dp) // max_dp
        sel.append(order[(lo + hi - 1) // 2])
    return [dp[i] for i in sorted(sel)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default="traindata/topo/tianyan287_20q.json")
    ap.add_argument("--data-dir", default="traindata")
    ap.add_argument("--splits", default="",
                    help="逗号分隔 split 名（splits/*.txt）")
    ap.add_argument("--nam-dirs", default="",
                    help="逗号分隔 QASM 目录（键 nam/<dir>__<名>）")
    ap.add_argument("--cache-dir", default="traindata/routed")
    ap.add_argument("--out-dir", default="traindata/residual")
    ap.add_argument("--traj", type=int, default=16)
    ap.add_argument("--param-seed", type=int, default=0)
    ap.add_argument("--max-dp", type=int, default=8,
                    help="每电路最多采的决策点数")
    ap.add_argument("--top-cand", type=int, default=8,
                    help="每决策点最多评估的候选数")
    ap.add_argument("--sim-device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()

    config, hw, cm = load_topo(args.topo)
    topo_name = os.path.splitext(os.path.basename(args.topo))[0]
    max_edges = max(19, len(cm))
    try:
        si, sn = args.shard.split("/")
        shard_i, shard_n = int(si), max(1, int(sn))
    except ValueError:
        shard_i, shard_n = 0, 1

    fid_fn = make_event_fidelity_fn_v3(config, num_trajectories=args.traj,
                                       seed=0, backend=args.sim_device)
    fa_fn = _make_sched_analytic_fn(config)

    circuits = []
    seen = set()
    for split in filter(None, args.splits.split(",")):
        for rel, apath in iter_split_circuits(args.data_dir, split):
            if rel not in seen:
                seen.add(rel)
                circuits.append((rel, apath))
    for d in filter(None, args.nam_dirs.split(",")):
        d = d.strip()
        if not d or not os.path.isdir(d):
            continue
        tag = os.path.basename(os.path.normpath(d))
        for fname in sorted(os.listdir(d)):
            if fname.endswith(".qasm"):
                rel = f"nam/{tag}__{fname[:-5]}"
                if rel not in seen:
                    seen.add(rel)
                    circuits.append((rel, os.path.join(d, fname)))
    if args.limit > 0:
        circuits = circuits[:args.limit]
    circuits = circuits[shard_i::shard_n]
    print(f"[r0] topo={topo_name} circuits={len(circuits)} shard={args.shard} "
          f"max_dp={args.max_dp} top_cand={args.top_cand} traj={args.traj}")

    rows = []
    n_skip = n_nodes = 0
    for i, (rel, apath) in enumerate(circuits):
        cpath = routed_cache_path(args.cache_dir, topo_name, rel)
        if not os.path.exists(cpath):
            n_skip += 1
            continue
        try:
            entry = load_routed_cache(cpath)
        except Exception as e:
            print(f"  [{i}] {rel}: 缓存读取失败 {e}")
            n_skip += 1
            continue
        phys = entry["phys"]
        if phys.num_parameters > 0:
            rng = np.random.default_rng(args.param_seed)
            phys = phys.assign_parameters(
                {p: rng.uniform(0, 2 * np.pi) for p in phys.parameters})
        env = make_sched_env(phys, hw, cm, max_edges)
        dp_all, steps, done = collect_decision_points(env)
        if not done:
            n_skip += 1
            continue
        for (cenv, step, clock, legal) in stratify_select(dp_all, args.max_dp):
            cands = legal[:args.top_cand]
            deltas = np.full(args.top_cand, np.nan, dtype=np.float32)
            analytics = np.full(args.top_cand, np.nan, dtype=np.float32)
            q0 = None
            qs = []
            for c, a in enumerate(cands):
                b = cenv.clone()
                try:
                    _, _, _, _, _ = b.step(a, compute_obs=False)
                except RuntimeError:
                    continue
                fa = float(fa_fn(b))
                d_ok, _ = run_asap_from(b)
                if not d_ok:
                    continue
                q = float(fid_fn(b))
                if a == cands[0]:
                    q0 = q
                qs.append(q)
                analytics[c] = fa
            if q0 is None:
                continue
            for c in range(len(cands)):
                deltas[c] = qs[c] - q0
            n_nodes += 1
            rows.append((rel, step, float(clock), cands, deltas, analytics))
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(circuits)}] nodes={n_nodes} skip={n_skip}")

    os.makedirs(args.out_dir, exist_ok=True)
    N = len(rows)
    circs = np.array([r[0] for r in rows], dtype=object)
    steps = np.array([r[1] for r in rows], dtype=np.int32)
    clocks = np.array([r[2] for r in rows], dtype=np.float32)
    slots = np.full((N, args.top_cand), -1, dtype=np.int16)
    deltas = np.full((N, args.top_cand), np.nan, dtype=np.float32)
    analytics = np.full((N, args.top_cand), np.nan, dtype=np.float32)
    n_legal = np.zeros(N, dtype=np.int16)
    for k, (rel, step, clock, cands, dl, an) in enumerate(rows):
        m = len(cands)
        n_legal[k] = m
        slots[k, :m] = cands
        deltas[k, :m] = dl[:m]
        analytics[k, :m] = an[:m]

    out_npz = os.path.join(args.out_dir, f"residual_shard{shard_i}of{shard_n}.npz")
    np.savez_compressed(
        out_npz, circuit=circs, step=steps, clock=clocks, action=slots,
        delta=deltas, analytic=analytics, n_legal=n_legal)
    manifest = os.path.join(args.out_dir,
                            f"residual_shard{shard_i}of{shard_n}.json")
    with open(manifest, "w") as f:
        json.dump({
            "shard": args.shard, "topo": topo_name, "traj": args.traj,
            "param_seed": args.param_seed, "max_dp": args.max_dp,
            "top_cand": args.top_cand, "n_samples": int(N),
            "n_nodes": int(n_nodes), "n_skip": int(n_skip),
        }, f)
    # Δ 分布概要
    dv = deltas[~np.isnan(deltas)]
    print(f"\n[r0] 完成 node={n_nodes} skip={n_skip} → {out_npz}")
    if dv.size:
        print(f"  Δ 分布: min={dv.min():+.4f} p25={np.percentile(dv,25):+.4f} "
              f"med={np.median(dv):+.4f} p75={np.percentile(dv,75):+.4f} "
              f"max={dv.max():+.4f}  |Δ|>0.02 占比="
              f"{float((np.abs(dv) > 0.02).mean()):.1%}")


if __name__ == "__main__":
    main()