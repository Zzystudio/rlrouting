#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""方向2 Phase 1 评估：SABRE 脚本路由 + {ASAP | RL调度器} 对照。

对每条电路：
  1. Layer 0 routed 缓存（无则现场 SABRE 路由）→ scheduling_only 环境
  2. ASAP 基线 episode（priority 槽序 + 锁等待守卫）→ fid/makespan
  3. RL 策略 episode（deterministic, action_mask）→ fid/makespan
指标：swaps_added（必须=0，平价回归）、makespan、fid(trajectory_v3)。
sref 可选：--sref-cache 提供时额外报告 log-ratio（相对 SABRE+ASAP 的增益）。

用法（项目根）：
  PYTHONPATH=src:. python3 scripts/eval_sched.py \
    --model models/sched_armA.pt --arm analytic \
    --topo traindata/topo/tianyan287_20q.json \
    --circuits traindata/splits/indist50_paths.txt \
    --cache-dir traindata/routed --sim-device cuda:0 \
    --out /tmp/opencode/eval_sched_armA_indist.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing.gnn.encoder import SubGNN
from routing.rl.agent_clocked import ClockedPPOAgent, D_EXEC, D_TIMING_GLOB
from routing.rl.eval_policy import load_topo
from sim.trajectory_sim_v3 import make_event_fidelity_fn_v3
from scripts.sched_common import (sabre_route_full, routed_cache_path,
                                  load_routed_cache, save_routed_cache,
                                  make_sched_env, run_sched_episode,
                                  sched_episode_stats, parity_check)
from scripts.eval_clocked import qasm_load


def get_routed(args, rel, apath, config, i):
    """routed 缓存优先；miss 时现场路由并回写。返回 phys。"""
    cpath = routed_cache_path(args.cache_dir, args.topo_name, rel)
    if os.path.exists(cpath):
        try:
            return load_routed_cache(cpath)["phys"]
        except Exception as e:
            print(f"  [cache] {rel}: 读取失败 {e}，重路由")
    qc = qasm_load(apath)
    phys, nsw, layout = sabre_route_full(qc, config, swap_trials=args.trials,
                                         seed=i)
    if not parity_check(phys, nsw):
        raise RuntimeError(f"{rel}: 平价校验失败 swaps={nsw}")
    save_routed_cache(cpath, phys, layout, nsw, args.trials, i)
    return phys


def run_rl_episode(agent, env, max_steps=4000):
    obs, _ = env.reset()
    E = env.num_edges
    done = False
    steps = 0
    while not done and steps < max_steps:
        mask = env.get_action_mask()
        a, _, _, _, _ = agent.act(obs, deterministic=True, action_mask=mask)
        obs, _, done, trunc, _ = env.step(a)
        done = done or trunc
        steps += 1
    return done


def run_residual_episode(model, obs_mean, obs_std, env, margin,
                         max_steps=4000):
    """R2：1-ply margin 门选择算子 episode。

    每步：det = 首合法 EXEC（/SKIP）；若 |legal|>1，Δ̂(legal[:8])，
    max Δ̂ > margin 且 argmax≠det 才偏离。返回 (done, dev_pred_sum, n_dev)。
    dev_pred_sum = 已采偏离的预测 Δ̂ 之和（组合误差对照实测 Δ 用）。
    """
    import torch as _torch
    E, K = env.num_edges, env.max_ready
    obs, _ = env.reset()
    done = False
    steps = 0
    dev_pred_sum = 0.0
    n_dev = 0
    while not done and steps < max_steps:
        mask = env.get_action_mask()
        legal = [i for i in range(E, E + K) if mask[i]]
        det = legal[0] if legal else env.skip_action
        a = det
        if len(legal) > 1:
            xn = (np.asarray(obs, dtype=np.float64) - obs_mean) / obs_std
            x = _torch.tensor(np.asarray(xn, dtype=np.float32)).unsqueeze(0)
            with _torch.no_grad():
                d_hat, _ = model(x)
            d_hat = d_hat[0]
            cands = legal[:8]
            preds = np.array([float(d_hat[a].item()) for a in cands])
            j = int(np.argmax(preds))
            if preds[j] > margin and cands[j] != det:
                a = cands[j]
                dev_pred_sum += float(preds[j])
                n_dev += 1
        obs, _, done, trunc, _ = env.step(a)
        done = done or trunc
        steps += 1
    return done, dev_pred_sum, n_dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="sched-only 训练的 checkpoint")
    ap.add_argument("--no-gnn", action="store_true", default=False)
    ap.add_argument("--edge-hidden", type=int, default=128)
    ap.add_argument("--topo", default="traindata/topo/tianyan287_20q.json")
    ap.add_argument("--topo-name", default=None,
                    help="routed 缓存目录名（默认取 --topo 文件名）")
    ap.add_argument("--circuits", required=True, help="qasm/pkl 路径列表 txt")
    ap.add_argument("--cache-dir", default="traindata/routed")
    ap.add_argument("--sref-cache", default=None)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--traj", type=int, default=32)
    ap.add_argument("--sim-device", default="cuda:0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-num-qubits", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selector", default="none",
                    choices=["none", "analytic"],
                    help="analytic=运行时用解析 fid 在 asap/RL 两调度中选优"
                         "（可部署的 ASAP+RL 选择器，Phase 2b 机制雏形）")
    ap.add_argument("--residual-model", default=None,
                    help="R1 残差模型 checkpoint：1-ply margin 门选择算子"
                         "（与 --model 互斥）")
    ap.add_argument("--margin", type=float, default=0.01,
                    help="残差选择器的偏离安全边际（log-fid）")
    ap.add_argument("--out", default="/tmp/opencode/eval_sched.json")
    args = ap.parse_args()

    args.topo_name = args.topo_name or os.path.splitext(
        os.path.basename(args.topo))[0]
    config, hw, cm = load_topo(args.topo)
    max_edges = max(19, len(cm))
    fid_fn = make_event_fidelity_fn_v3(config, num_trajectories=args.traj,
                                       seed=0, backend=args.sim_device)
    fa_fn = None
    if args.selector == "analytic":
        from routing.rl.train_agent import _make_sched_analytic_fn
        fa_fn = _make_sched_analytic_fn(config)
    sref = json.load(open(args.sref_cache)) if args.sref_cache else {}

    agent = None
    res_model = None
    if args.residual_model:
        from routing.rl.residual_model import ResidualRegressor
        ck = torch.load(args.residual_model, map_location="cpu",
                        weights_only=False)
        res_model = ResidualRegressor(ck["obs_dim"], ck["n_actions"]).to("cpu")
        res_model.load_state_dict(ck["state_dict"])
        res_model.eval()
        print(f"[eval_sched] loaded residual model {args.residual_model} "
              f"(g1_pass={ck.get('g1_pass')})")
    elif args.model:
        gnn = None
        if not args.no_gnn:
            gnn = SubGNN(subgraph="full")
        agent = ClockedPPOAgent(
            obs_dim=1, action_dim=1, num_qubits=args.max_num_qubits,
            num_edges=max_edges, max_ready=24,
            edge_feat_dim=267 + 12,
            exec_feat_dim=D_EXEC, timing_glob_dim=D_TIMING_GLOB,
            device=args.device, gnn=gnn, edge_hidden=args.edge_hidden)
        state = agent.load_checkpoint(args.model)
        agent.ac.eval()
        if gnn is not None:
            gnn.eval()
        print(f"[eval_sched] loaded {args.model} (step={state.get('step', '?')})")

    paths = []
    for line in open(args.circuits):
        line = line.strip()
        if line and os.path.exists(line):
            paths.append(line)
    if args.limit > 0:
        paths = paths[:args.limit]
    print(f"[eval_sched] {len(paths)} circuits, fid=traj_v3(T={args.traj})")

    rows = []
    for i, apath in enumerate(paths):
        name = os.path.basename(apath)
        rel = os.path.splitext(os.path.relpath(apath, "traindata"))[0] \
            if not apath.endswith(".qasm") else None
        # 缓存键与训练侧一致：pkl splits 用 rel 路径（去 .pkl），qasm 用
        # nam/<dir>__<stem>
        if apath.endswith(".qasm"):
            d = os.path.basename(os.path.dirname(apath))
            rel = f"nam/{d}__{name[:-5]}"
        else:
            rel = os.path.splitext(os.path.relpath(apath, "traindata"))[0]
        try:
            phys = get_routed(args, rel, apath, config, i)
        except Exception as e:
            print(f"  [{i}] {name}: 路由失败 {e}")
            continue
        # ASAP 基线
        env0 = make_sched_env(phys, hw, cm, max_edges,
                              max_num_qubits=args.max_num_qubits)
        rng = np.random.default_rng(0)
        done0, _ = run_sched_episode(env0, "asap", rng=rng)
        if not done0:
            print(f"  [{i}] {name}: ASAP 未完成，跳过")
            continue
        sw0, ms0, _ = sched_episode_stats(env0)
        f0 = float(fid_fn(env0))
        row = {"circuit": name, "rel": rel, "sabre_swaps": sw0,
               "asap_ms": round(ms0, 2), "asap_fid": f0,
               "asap_fid_analytic": float(fa_fn(env0)) if fa_fn else None}
        # RL 策略（PPO checkpoint）或 R2 残差 1-ply 选择算子
        if res_model is not None:
            env1 = make_sched_env(phys, hw, cm, max_edges,
                                  max_num_qubits=args.max_num_qubits)
            done1, dev_pred_sum, n_dev = run_residual_episode(
                res_model, ck["obs_mean"], ck["obs_std"], env1, args.margin)
            if done1:
                sw1, ms1, _ = sched_episode_stats(env1)
                f1 = float(fid_fn(env1))
                row.update({"rl_swaps_added": sw1, "rl_ms": round(ms1, 2),
                            "rl_fid": f1, "n_dev": n_dev,
                            "dev_pred_sum": round(dev_pred_sum, 4)})
            else:
                row.update({"rl_swaps_added": None, "rl_fid": None})
        elif agent is not None:
            env1 = make_sched_env(phys, hw, cm, max_edges,
                                  max_num_qubits=args.max_num_qubits)
            done1 = run_rl_episode(agent, env1)
            if done1:
                sw1, ms1, _ = sched_episode_stats(env1)
                f1 = float(fid_fn(env1))
                row.update({"rl_swaps_added": sw1, "rl_ms": round(ms1, 2),
                            "rl_fid": f1,
                            "rl_fid_analytic": float(fa_fn(env1)) if fa_fn else None})
            else:
                row.update({"rl_swaps_added": None, "rl_fid": None})
            # 选择器：解析 fid 选 asap/RL（运行时可得，无 oracle）
            if fa_fn is not None and row.get("rl_fid") is not None:
                pick_rl = row["rl_fid_analytic"] > row["asap_fid_analytic"]
                row["sel_pick"] = "rl" if pick_rl else "asap"
                row["sel_fid"] = row["rl_fid"] if pick_rl else row["asap_fid"]
                row["sel_ms"] = row["rl_ms"] if pick_rl else row["asap_ms"]
        # sref log-ratio（相对 SABRE+ASAP 的调度增益，若提供 per-circuit 参考）
        se = sref.get(rel)
        if se and se.get("traj"):
            row["sref_traj"] = se["traj"]
        rows.append(row)
        rl = row.get("rl_fid")
        print(f"  [{i}] {name:36s} sw={sw0:3d} "
              f"asap: ms={ms0:6.1f} fid={f0:.4f} | "
              f"rl: ms={row.get('rl_ms', '-')} "
              f"fid={rl if rl is None else round(rl, 4)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)

    ok = [r for r in rows if r.get("rl_fid") is not None]
    print("\n=== eval_sched 汇总 ===")
    if ok:
        a = np.array([r["asap_fid"] for r in ok])
        rl = np.array([r["rl_fid"] for r in ok])
        ms_a = np.array([r["asap_ms"] for r in ok])
        ms_r = np.array([r["rl_ms"] for r in ok])
        added = [r["rl_swaps_added"] for r in ok]
        print(f"circuits={len(ok)}  swaps_added>0: "
              f"{sum(1 for x in added if x and x > 0)}（必须为 0）")
        print(f"fid : ASAP={a.mean():.4f}  RL={rl.mean():.4f}  "
              f"RL/ASAP={rl.mean()/max(a.mean(),1e-9):.3f}x  "
              f"win={int((rl > a).sum())}/{len(ok)}")
        print(f"ms  : ASAP={ms_a.mean():.1f}  RL={ms_r.mean():.1f}  "
              f"RL/ASAP={ms_r.mean()/max(ms_a.mean(),1e-9):.3f}x")
        if args.selector == "analytic" and all("sel_fid" in r for r in ok):
            sel = np.array([r["sel_fid"] for r in ok])
            sel_ms = np.array([r["sel_ms"] for r in ok])
            n_rl = sum(1 for r in ok if r["sel_pick"] == "rl")
            print(f"选择器(analytic): fid={sel.mean():.4f} "
                  f"({sel.mean()/max(a.mean(),1e-9):.3f}x vs ASAP)  "
                  f"pick_rl={n_rl}/{len(ok)}  ms={sel_ms.mean():.1f}")
    print(f"写入 {args.out}")


if __name__ == "__main__":
    main()
