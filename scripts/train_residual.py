#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R1-V1：残差回归训练 + G1 报告（排序能力判定）。

数据：R0 仲裁蒸馏 npz（traindata/residual/residual_shard*.npz）。
obs 重建：同 routed 电路 + 同 param seed 的 ASAP 确定性重放，仅在采样
决策步调用 env._obs()（与 R0 的步计数完全一致）。

G1（预先写死）：
  ① held-out Spearman ρ(Δ̂,Δ)（逐决策点 ρ → 每电路中位 / pooled Fisher-z）
     vs 基线（priority-rank、解析 fid）——要求 ρ(Δ̂) > max(基线ρ) + 0.1
  ② "Δ̂>0 精确率"：模型提议偏离（argmax≠首候选）时真 Δ>0 的比例 > 60%，
     且 top-decile 置信更高
  ③ Δ 分布分位曲线

用法（项目根）：
  PYTHONPATH=src:. python3 scripts/train_residual.py \
    --data-dir traindata/residual --out models/residual_v1.pt --device cuda:3
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing.rl.eval_policy import load_topo
from routing.rl.residual_model import ResidualRegressor
from scripts.sched_common import (routed_cache_path, load_routed_cache,
                                  make_sched_env)

MAX_STEPS = 4000


def family_of(rel: str) -> str:
    m = re.search(r"__([a-z]+[a-z0-9]*?)_\d+", rel)
    if m:
        return m.group(1)
    return rel.split("_")[0]


def replay_obs(phys, hw, cm, max_edges, steps_needed):
    """ASAP 确定性重放，仅在 steps_needed 中的步上计算 obs。

    步计数与 R0 一致：s = 已执行动作数；obs(s) = 第 s 步动作前的状态观测。
    注意：observation_space 声明与 _obs() 实际维度不符（存量 bug，agent 用
    block 偏移切片故未触发），obs_dim 从实际 reset 返回测量。
    返回 ({step: obs}, n_actions, obs_dim)。
    """
    env = make_sched_env(phys, hw, cm, max_edges)
    targets = set(steps_needed)
    out = {}
    obs0, _ = env.reset()
    obs_dim = int(np.asarray(obs0).shape[0])
    s = 0
    done = False
    E, K = env.num_edges, env.max_ready
    while not done and s < MAX_STEPS:
        env._update_candidates()
        if s in targets:
            out[s] = np.asarray(env._obs(), dtype=np.float32)
            if len(out) == len(targets):
                break
        mask = env.get_action_mask()
        legal = [i for i in range(E, E + K) if mask[i]]
        a = legal[0] if legal else env.skip_action
        try:
            _, _, done, trunc, _ = env.step(a, compute_obs=False)
        except RuntimeError:
            break
        done = done or trunc
        s += 1
    return out, env.action_space.n, obs_dim


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float((ra * rb).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default="traindata/topo/tianyan287_20q.json")
    ap.add_argument("--cache-dir", default="traindata/routed")
    ap.add_argument("--data-dir", default="traindata/residual")
    ap.add_argument("--out", default="models/residual_v1.pt")
    ap.add_argument("--param-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w-scale", type=float, default=50.0,
                    help="样本权重 w = 1 + w_scale·|Δ|（抗零质量稀疏）")
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config, hw, cm = load_topo(args.topo)
    topo_name = os.path.splitext(os.path.basename(args.topo))[0]
    max_edges = max(19, len(cm))

    # ---- 加载全部 shard ----
    shards = sorted(f for f in os.listdir(args.data_dir)
                    if f.startswith("residual_shard") and f.endswith(".npz"))
    assert shards, f"{args.data_dir} 无 residual_shard*.npz"
    circ, step, act, dl, an, nl = [], [], [], [], [], []
    for f in shards:
        z = np.load(os.path.join(args.data_dir, f), allow_pickle=True)
        circ.extend(z["circuit"].tolist())
        step.append(z["step"])
        act.append(z["action"])
        dl.append(z["delta"])
        an.append(z["analytic"])
        nl.append(z["n_legal"])
        print(f"[load] {f}: {len(z['step'])} 决策点")
    step = np.concatenate(step)
    act = np.concatenate(act)
    dl = np.concatenate(dl)
    an = np.concatenate(an)
    nl = np.concatenate(nl)
    circ = np.array(circ, dtype=object)
    N = len(step)
    print(f"[load] 合计决策点={N} 候选(有限Δ)={int((~np.isnan(dl)).sum())}")

    # ---- 电路划分（80/20 按族分层）----
    circuits = sorted(set(circ.tolist()))
    fam = {}
    for c in circuits:
        fam.setdefault(family_of(str(c)), []).append(c)
    rng = np.random.default_rng(args.seed)
    train_circ, val_circ = set(), set()
    for f, cs in sorted(fam.items()):
        cs = list(cs)
        rng.shuffle(cs)
        k = max(1, int(round(0.8 * len(cs))))
        train_circ.update(cs[:k])
        val_circ.update(cs[k:])
    print(f"[split] 训练电路={len(train_circ)} 验证电路={len(val_circ)} "
          f"（族数={len(fam)}）")

    # ---- obs 重建（每电路一次重放）----
    print("[replay] 重建决策点 obs ...")
    obs_store = {}  # (circuit, step) -> obs
    n_actions = obs_dim = None
    for ci, c in enumerate(circuits):
        need = sorted(set(step[circ == c].tolist()))
        if not need:
            continue
        cpath = routed_cache_path(args.cache_dir, topo_name, str(c))
        if not os.path.exists(cpath):
            continue
        phys = load_routed_cache(cpath)["phys"]
        if phys.num_parameters > 0:
            prng = np.random.default_rng(args.param_seed)
            phys = phys.assign_parameters(
                {p: prng.uniform(0, 2 * np.pi) for p in phys.parameters})
        omap, n_a, o_d = replay_obs(phys, hw, cm, max_edges, need)
        n_actions, obs_dim = n_a, o_d
        for s, o in omap.items():
            obs_store[(str(c), int(s))] = o
        if (ci + 1) % 50 == 0:
            print(f"  [replay] {ci+1}/{len(circuits)} 电路")
    print(f"[replay] obs 数={len(obs_store)}  obs_dim={obs_dim} "
          f"n_actions={n_actions}")

    # ---- 组装训练样本 ----
    Xi, Ai, Yi, Ci, Si = [], [], [], [], []
    for k in range(N):
        c = str(circ[k])
        for j in range(nl[k]):
            if np.isnan(dl[k, j]) or act[k, j] < 0:
                continue
            key = (c, int(step[k]))
            if key not in obs_store:
                continue
            Xi.append(obs_store[key])
            Ai.append(int(act[k, j]))
            Yi.append(float(dl[k, j]))
            Ci.append(k)  # 决策点索引（G1 评估用）
            Si.append(j)  # 候选序（priority 基线用）
    X = np.asarray(Xi, dtype=np.float32)
    A = np.asarray(Ai, dtype=np.int64)
    Y = np.asarray(Yi, dtype=np.float32)
    C = np.asarray(Ci, dtype=np.int64)
    S = np.asarray(Si, dtype=np.int64)
    W = 1.0 + args.w_scale * np.abs(Y)
    is_val = np.array([str(circ[c]) in val_circ for c in C], dtype=bool)
    print(f"[data] 样本={len(X)} 训练={int((~is_val).sum())} "
          f"验证={int(is_val.sum())}")

    mu = X[~is_val].mean(0)
    sd = X[~is_val].std(0) + 1e-6
    Xn = (X - mu) / sd

    dev = torch.device(args.device if torch.cuda.is_available()
                       or args.device.startswith("cuda") else "cpu")
    model = ResidualRegressor(obs_dim, n_actions).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    Xt = torch.tensor(Xn[~is_val], device=dev)
    At = torch.tensor(A[~is_val], device=dev)
    Yt = torch.tensor(Y[~is_val], device=dev)
    Wt = torch.tensor(W[~is_val], device=dev)
    St_sign = (Yt >= 0).float()
    n_tr = len(Xt)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n_tr, device=dev)
        tot = 0.0
        nb = 0
        for i in range(0, n_tr, args.batch):
            sel = perm[i:i + args.batch]
            d_hat, s_logit = model(Xt[sel])
            d_pred = d_hat.gather(1, At[sel].unsqueeze(1)).squeeze(1)
            s_pred = s_logit.gather(1, At[sel].unsqueeze(1)).squeeze(1)
            mse = ((d_pred - Yt[sel]) ** 2 * Wt[sel]).mean()
            bce = F.binary_cross_entropy_with_logits(s_pred, St_sign[sel])
            loss = mse + 0.3 * bce
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item())
            nb += 1
        if (ep + 1) % 10 == 0:
            print(f"  [train] epoch {ep+1}/{args.epochs} loss={tot/max(1,nb):.5f}")

    # ---- G1 评估（held-out 决策点）----
    model.eval()
    Xv = torch.tensor(Xn[is_val], device=dev)
    with torch.no_grad():
        d_all, _ = model(Xv)
    pred_map = {}
    for i, k in enumerate(C[is_val]):
        a = int(A[is_val][i])
        pred_map.setdefault(int(k), {})[a] = float(d_all[i, a].item())

    rows = []
    for k in sorted(pred_map):
        cand_a = act[k][act[k] >= 0]
        cand_d = dl[k][:len(cand_a)]
        cand_an = an[k][:len(cand_a)]
        ok = ~np.isnan(cand_d)
        if ok.sum() < 3:
            continue
        a_ok = cand_a[ok]
        d_ok = cand_d[ok]
        preds = np.array([pred_map[k].get(int(a), np.nan) for a in a_ok])
        if np.isnan(preds).any():
            continue
        rho_m = spearman(preds, d_ok)
        rho_p = spearman(-np.arange(len(a_ok), dtype=float), d_ok)  # priority
        rho_an = (spearman(cand_an[ok], d_ok)
                  if not np.isnan(cand_an[ok]).any() else None)
        am = int(a_ok[np.argmax(preds)])
        a0 = int(cand_a[0])
        propose = am != a0
        true_gain = float(d_ok[np.argmax(preds)]) if propose else 0.0
        rows.append({"rho_m": rho_m, "rho_p": rho_p, "rho_an": rho_an,
                     "propose": propose, "true_gain": true_gain,
                     "conf": float(np.max(preds))})

    n_pts = len(rows)
    print(f"\n=== G1 报告（held-out 决策点 n={n_pts}）===")
    if not n_pts:
        print("无有效验证决策点！")
        return
    rm = [r["rho_m"] for r in rows if r["rho_m"] is not None]
    rp = [r["rho_p"] for r in rows if r["rho_p"] is not None]
    ra = [r["rho_an"] for r in rows if r["rho_an"] is not None]
    print(f"ρ(模型)  : mean={np.mean(rm):+.3f}  median={np.median(rm):+.3f}")
    print(f"ρ(priority 基线): mean={np.mean(rp):+.3f}  median={np.median(rp):+.3f}")
    if ra:
        print(f"ρ(解析 fid 基线): mean={np.mean(ra):+.3f}  "
              f"median={np.median(ra):+.3f}")
    n_prop = sum(1 for r in rows if r["propose"])
    if n_prop:
        good = sum(1 for r in rows if r["propose"] and r["true_gain"] > 0)
        print(f"提议偏离: {n_prop}/{n_pts}（{n_prop/n_pts:.0%}）  "
              f"其中真增益>0: {good}/{n_prop} = "
              f"{good/max(1,n_prop):.0%}（精确率，要求>60%）")
        gains = [r["true_gain"] for r in rows if r["propose"]]
        print(f"提议偏离的平均真增益: {np.mean(gains):+.4f}  "
              f"（负值=有害提议）")
    confs = np.array([r["conf"] for r in rows])
    if n_prop:
        prop_rows = [r for r in rows if r["propose"]]
        th = np.percentile([r["conf"] for r in prop_rows], 90)
        top = [r for r in prop_rows if r["conf"] >= th]
        tg = sum(1 for r in top if r["true_gain"] > 0)
        print(f"top-decile 置信精确率: {tg}/{len(top)} = "
              f"{tg/max(1,len(top)):.0%}")
    g1_pass = (np.mean(rm) > max(np.mean(rp), np.mean(ra) if ra else -1) + 0.1)
    print(f"\n[G1] {'通过' if g1_pass else '不通过'}："
          f"ρ(模型)={np.mean(rm):+.3f} vs max(基线)="
          f"{max(np.mean(rp), np.mean(ra) if ra else -1):+.3f}（要求 +0.1）")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "obs_mean": mu,
                "obs_std": sd, "obs_dim": int(obs_dim),
                "n_actions": int(n_actions), "g1_pass": bool(g1_pass)},
               args.out)
    print(f"模型与归一化统计 → {args.out}")


if __name__ == "__main__":
    main()
