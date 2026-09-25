"""Gate L3-A：排序损失训练（兄弟级 advantage 分辨力）。

损失（对齐 20260922 方案 Phase 3，λ1=1.0 起步 + λ2 排序）：
  L_res  = Huber(V_θ(s), y(s))                    绝对回归（主数据 + 兄弟状态）
  L_rank = mean_pairs max(0, m - (V(worse) - V(better)))   兄弟对排序
  L = L_res + λ2 · L_rank

兄弟对：同一父状态的后继按标签排序，标签差 ≥1 的对参与（差 0 的跳过）。

用法:
    PYTHONPATH=src python3 scripts/v0/train_rank.py --lambda2 0.5
"""

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from routing.v0.value_net import ValueNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="benchmark/v0_vstar16.npz")
    ap.add_argument("--sib", default="benchmark/v0_sibling.npz")
    ap.add_argument("--lambda2", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-model", default="models/v0_mlp16_rank.pt")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, y, cnames = d["X"], d["y"], d["circuit"]
    s = np.load(args.sib, allow_pickle=True)
    Xs, ys, pid = s["X"], s["y"], s["parent_id"]

    # 分层切分（与 run_gate_l3 / dagger_round 一致）
    strata = {}
    for c in sorted(set(cnames.tolist())):
        parts = c.replace(".pkl", "").split("_")
        fam = "perm_mix" if parts[0] == "perm" else parts[0]
        tier = parts[3] if parts[0] == "perm" else parts[2]
        strata.setdefault((fam, tier), []).append(c)
    rng0 = np.random.default_rng(0)
    test_set = set()
    for k, cs in sorted(strata.items()):
        cs = sorted(set(cs))
        rng0.shuffle(cs)
        test_set.update(cs[:max(1, int(len(cs) * 0.2))])
    is_test = np.array([c in test_set for c in cnames])
    tr_idx = np.where(~is_test)[0]
    rng = np.random.default_rng(1)
    rng.shuffle(tr_idx)
    n_val = max(1, int(len(tr_idx) * 0.1))
    va_idx, tr_idx = tr_idx[:n_val], tr_idx[n_val:]

    # 兄弟数据：只取训练电路的父状态
    sib_mask = np.array([c not in test_set for c in s["circuit"]])
    Xs_tr, ys_tr, pid_tr = Xs[sib_mask], ys[sib_mask], pid[sib_mask]
    # 按父分组，构造排序对 (better_idx, worse_idx)
    pairs_b, pairs_w = [], []
    by_parent = {}
    for j, p in enumerate(pid_tr):
        by_parent.setdefault(int(p), []).append(j)
    n_pairs = 0
    for p, idxs in by_parent.items():
        if len(idxs) < 2:
            continue
        order = sorted(idxs, key=lambda j: ys_tr[j])
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                if ys_tr[order[b]] - ys_tr[order[a]] >= 1.0:
                    pairs_b.append(order[a])
                    pairs_w.append(order[b])
                    n_pairs += 1
    pairs_b = np.array(pairs_b)
    pairs_w = np.array(pairs_w)
    print(f"数据: 主 {len(tr_idx)} + 兄弟 {len(Xs_tr)} 状态, 排序对 {n_pairs}")

    Xt = torch.as_tensor(np.concatenate([X[tr_idx], Xs_tr]), dtype=torch.float32)
    yt = torch.as_tensor(np.concatenate([y[tr_idx], ys_tr]), dtype=torch.float32)
    Xb = torch.as_tensor(Xs_tr[pairs_b], dtype=torch.float32)
    Xw = torch.as_tensor(Xs_tr[pairs_w], dtype=torch.float32)
    Xv = torch.as_tensor(X[va_idx], dtype=torch.float32)
    yv = torch.as_tensor(y[va_idx], dtype=torch.float32)

    net = ValueNet(in_dim=X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    huber = torch.nn.HuberLoss(delta=1.0)

    best_mae, best_state, bad = float("inf"), None, 0
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for b0 in range(0, len(Xt), args.batch):
            idx = perm[b0:b0 + args.batch]
            loss_res = huber(net(Xt[idx]), yt[idx])
            # 排序对：随机采样与 batch 同量级
            if len(pairs_b) > 0:
                sel = torch.randint(0, len(pairs_b), (min(len(idx), len(pairs_b)),))
                vb = net(Xb[sel])
                vw = net(Xw[sel])
                loss_rank = torch.clamp(args.margin - (vw - vb), min=0).mean()
            else:
                loss_rank = torch.zeros(1)
            loss = loss_res + args.lambda2 * loss_rank
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        net.eval()
        with torch.no_grad():
            mae = float((net(Xv) - yv).abs().mean().item())
        if mae < best_mae:
            best_mae, best_state, bad = mae, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 60:
                break
    net.load_state_dict(best_state)
    net.eval()

    # 评测：val MAE + 兄弟分辨力（val 不适用——兄弟全是训练父状态；
    # 用训练父状态的兄弟对测 top-1 命中率）
    with torch.no_grad():
        p_va = net(Xv).numpy()
        top1_hits, top1_tot = 0, 0
        for p, idxs in by_parent.items():
            idxs = [j for j in idxs]
            if len(idxs) < 2:
                continue
            vs = net(torch.as_tensor(Xs_tr[idxs])).numpy().tolist()
            ls = [ys_tr[j] for j in idxs]
            best_true = int(np.argmin(ls))
            best_pred = int(np.argmin(vs))
            top1_tot += 1
            top1_hits += int(ls[best_pred] == ls[best_true])
    print(f"\n===== 排序训练结果（λ2={args.lambda2}）=====")
    print(f"val MAE={best_mae:.2f}  兄弟 top-1 命中率={top1_hits}/{top1_tot}="
          f"{top1_hits/max(1,top1_tot):.2f}")
    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    torch.save(net.state_dict(), args.out_model)
    print(f"-> {args.out_model}")


if __name__ == "__main__":
    main()
