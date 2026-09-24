#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Layer 0：全数据集 SABRE 脚本路由缓存 + per-circuit sref（方向2）。

对 split/NAM 电路逐条跑 best-of-N 完整 SABRE（SabreLayout+SabreSwap，
不分解门），缓存 routed 物理电路到
  <cache-dir>/<topo名>/<rel>.routed.pkl
并（--with-sref）对 routed 电路跑 ASAP 调度，计算解析/traj_v3 两口径的
参考保真度写 --sref-out JSON（train_agent --sref-cache 消费，做 log-相对
终端奖励的分母）。

用法（项目根）：
  PYTHONPATH=src python3 scripts/export_sabre_routed.py \
    --topo traindata/topo/tianyan287_20q.json \
    --splits unified_train,tianyan20q_test \
    --nam-dir benchmark/nam_circs --with-sref \
    --sref-out traindata/routed/tianyan287_20q_sref.json

resumable：已存在的缓存文件跳过（--force 重跑）。
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing.rl.eval_policy import load_topo
from routing.rl.train_agent import (_make_sched_analytic_fn,
                                    build_fidelity_fn)
from scripts.sched_common import (sabre_route_full, sabre_route_best,
                                  routed_cache_path,
                                  save_routed_cache, parity_check,
                                  make_sched_env, run_sched_episode,
                                  sched_episode_stats)
from scripts.eval_clocked import qasm_load


def load_raw(path: str):
    """加载电路（pkl 不绑定参数——SABRE 路由与角度无关，缓存未绑定结果，
    参数多样性由训练侧每 episode 重绑定保留）。"""
    if path.endswith(".qasm") or path.endswith(".qas"):
        return qasm_load(path)
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def iter_split_circuits(data_dir: str, split: str):
    """split 文件（rel 路径行）→ [(rel_path, abs_path)]，与训练 loop 的
    circuit_path 键完全一致。"""
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
            else:
                print(f"  [skip] 不存在: {line}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default="traindata/topo/tianyan287_20q.json")
    ap.add_argument("--data-dir", default="traindata")
    ap.add_argument("--splits", default="",
                    help="逗号分隔 split 名（splits/*.txt）")
    ap.add_argument("--nam-dir", default="",
                    help="NAM/结构化 QASM 目录（键=nam/<名>）")
    ap.add_argument("--outer-seeds", type=int, default=3)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--cache-dir", default="traindata/routed")
    ap.add_argument("--with-sref", action="store_true", default=False)
    ap.add_argument("--sref-out", default="")
    ap.add_argument("--traj", type=int, default=16)
    ap.add_argument("--sim-device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true", default=False)
    ap.add_argument("--shard", default="0/1",
                    help="分片 i/n：处理 index %% n == i 的电路（并行 tmux 用）")
    ap.add_argument("--fid-select", type=int, default=0,
                    help="Phase 2(b)：跑 N 个 trial 的 SABRE，逐 trial ASAP 调度"
                         "后按 traj fid(T=16) 选优缓存（覆盖 .routed.pkl）；"
                         "0=按 swap 最少选（旧行为）")
    args = ap.parse_args()

    config, hw, cm = load_topo(args.topo)
    topo_name = os.path.splitext(os.path.basename(args.topo))[0]
    max_edges = max(19, len(cm))
    try:
        _si, _sn = args.shard.split("/")
        shard_i, shard_n = int(_si), max(1, int(_sn))
    except ValueError:
        shard_i, shard_n = 0, 1

    circuits = []
    seen = set()
    for split in filter(None, args.splits.split(",")):
        for rel, apath in iter_split_circuits(args.data_dir, split):
            if rel in seen:
                continue
            seen.add(rel)
            circuits.append((rel, apath))
    if args.nam_dir and os.path.isdir(args.nam_dir):
        dir_tag = os.path.basename(os.path.normpath(args.nam_dir))
        for fname in sorted(os.listdir(args.nam_dir)):
            if fname.endswith(".qasm"):
                rel = f"nam/{dir_tag}__{fname[:-5]}"
                if rel not in seen:
                    seen.add(rel)
                    circuits.append((rel, os.path.join(args.nam_dir, fname)))
    if args.limit > 0:
        circuits = circuits[:args.limit]
    circuits = circuits[shard_i::shard_n]
    print(f"[export] topo={topo_name} circuits={len(circuits)} "
          f"shard={args.shard} "
          f"outer_seeds={args.outer_seeds} trials={args.trials} "
          f"with_sref={args.with_sref}")

    fid_a = _make_sched_analytic_fn(config) if args.with_sref else None
    fid_t = (build_fidelity_fn("trajectory_v3", config,
                               num_trajectories=args.traj, seed=0,
                               backend=args.sim_device)
             if (args.with_sref or args.fid_select > 0) else None)

    sref = {}
    if args.with_sref and args.sref_out and os.path.exists(args.sref_out):
        sref = json.load(open(args.sref_out))
        print(f"[sref] 已有 {len(sref)} 条，增量补齐")

    n_new = n_skip = n_bad = 0
    for i, (rel, apath) in enumerate(circuits):
        cpath = routed_cache_path(args.cache_dir, topo_name, rel)
        if os.path.exists(cpath) and not args.force:
            entry = None
            try:
                from scripts.sched_common import load_routed_cache
                entry = load_routed_cache(cpath)
            except Exception:
                entry = None
            if entry is not None:
                n_skip += 1
                phys, nsw = entry["phys"], entry["swaps"]
            else:
                n_skip += 1
                continue
        else:
            try:
                qc = load_raw(apath)
            except Exception as e:
                print(f"  [{i}] {rel}: qasm 加载失败 {e}")
                n_bad += 1
                continue
            if args.fid_select > 0:
                # Phase 2(b)：多 trial × ASAP 调度 × traj fid 选优
                cands = []
                for t in range(args.fid_select):
                    try:
                        phys_t, nsw_t, lay_t = sabre_route_full(
                            qc, config, swap_trials=args.trials, seed=i * 100 + t)
                        env_t = make_sched_env(phys_t, hw, cm, max_edges)
                        done_t, _ = run_sched_episode(env_t, "asap")
                        if not done_t:
                            continue
                        cands.append((float(fid_t(env_t)), phys_t, nsw_t, lay_t))
                    except Exception as e:
                        print(f"  [{i}] {rel} trial{t}: {e}")
                if not cands:
                    print(f"  [{i}] {rel}: 全 trial 失败")
                    n_bad += 1
                    continue
                f_best, phys, nsw, layout = max(cands, key=lambda c: c[0])
                bseed = -1
            else:
                phys, nsw, layout, bseed = sabre_route_best(
                    qc, config, outer_seeds=args.outer_seeds,
                    swap_trials=args.trials, seed0=i)
            if not parity_check(phys, nsw):
                print(f"  [{i}] {rel}: 平价校验失败！swaps={nsw}")
                n_bad += 1
                continue
            save_routed_cache(cpath, phys, layout, nsw, args.trials, bseed)
            n_new += 1
        if not args.with_sref:
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(circuits)}] new={n_new} skip={n_skip}")
            continue
        if rel in sref:
            continue
        if phys.num_parameters > 0:
            # sref 需要数值角度（v3 模拟）；固定 seed=0 绑定（sref 是
            # per-circuit 常数参考，与训练期重绑定正交）
            _rng = np.random.default_rng(0)
            phys = phys.assign_parameters(
                {p: _rng.uniform(0, 2 * np.pi) for p in phys.parameters})
        env = make_sched_env(phys, hw, cm, max_edges)
        done, _ = run_sched_episode(env, "asap")
        if not done:
            print(f"  [{i}] {rel}: ASAP 未完成，跳过 sref")
            continue
        fa = float(fid_a(env))
        ft = float(fid_t(env))
        sref[rel] = {"analytic": fa, "traj": ft}
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(circuits)}] new={n_new} sref={len(sref)} "
                  f"最近: fa={fa:.4f} ft={ft:.4f}")

    print(f"\n[export] 完成：new={n_new} skip={n_skip} bad={n_bad}")
    if args.with_sref and args.sref_out:
        os.makedirs(os.path.dirname(args.sref_out) or ".", exist_ok=True)
        with open(args.sref_out, "w") as f:
            json.dump(sref, f, indent=1)
        if sref:
            fa = np.array([v["analytic"] for v in sref.values()])
            ft = np.array([v["traj"] for v in sref.values()])
            print(f"[sref] {len(sref)} 条 → {args.sref_out}")
            print(f"  analytic: mean={fa.mean():.4f} median={np.median(fa):.4f}")
            print(f"  traj_v3 : mean={ft.mean():.4f} median={np.median(ft):.4f}")


if __name__ == "__main__":
    main()
