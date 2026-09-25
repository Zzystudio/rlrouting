"""5-8q advantage 头训练（Huber + 排序）+ M2-V 对照 + 兄弟 top-1 诊断。

模型：
  adv-MLP: 输入 features(s')⊕features(s) (24 维) → Â(s,a)  （A* ≥ 0，最优=0）
  abs-V  : 输入 features(s) (12 维) → C*(s)              （M2 对照）
损失：
  L_adv  = Huber(Â, A*) + λ·排序（同父兄弟对，A* 差 ≥1）
  L_abs  = Huber(V_θ, C*)
诊断：
  兄弟 top-1 命中率（argmin Â vs argmin A*，随机基线 = 1/|兄弟集|）
  A* 分布、Â-MAE 分桶
"""

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from routing.v0.value_net import ValueNet


def train_generic(net, Xt, yt, Xv, yv, epochs=800, lr=1e-3, batch=256,
                  device="cpu", pairs=None, lambda2=0.3, margin=1.0,
                  patience=80):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    huber = torch.nn.HuberLoss(delta=1.0)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for b0 in range(0, len(Xt), batch):
            idx = perm[b0:b0 + batch]
            loss = huber(net(Xt[idx]).squeeze(-1), yt[idx])
            if pairs is not None and len(pairs[0]) > 0:
                sel = torch.randint(0, len(pairs[0]), (min(len(idx), len(pairs[0])),))
                pb, pw = pairs
                vb = net(Xt[pb[sel]]).squeeze(-1)
                vw = net(Xt[pw[sel]]).squeeze(-1)
                loss = loss + lambda2 * torch.clamp(
                    margin - (vw - vb), min=0).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            mae = float((net(Xv).squeeze(-1) - yv).abs().mean().item())
        if mae < best:
            best, best_state, bad = mae, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="benchmark/v0_adv_labels.npz")
    ap.add_argument("--lambda2", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--out-adv", default="models/v0_adv8.pt")
    ap.add_argument("--out-abs", default="models/v0_abs8.pt")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X24, A, pid = d["X24"], d["A"], d["pid"]
    pX, pC, cnames, split = d["parent_X12"], d["parent_C"], d["circuit"], d["split"]
    print(f"数据: {len(X24)} (s,a) 对 / {len(pC)} 父状态")

    # A* 分布
    vals, cnts = np.unique(np.round(A).astype(int), return_counts=True)
    print("A* 分布:", {int(v): int(c) for v, c in zip(vals[:8], cnts[:8])},
          f"(A*>7: {int((A > 7).sum())})")

    # ===== advantage 头 =====
    tr = split == 0
    tr_idx = np.where(tr)[0]
    rng = np.random.default_rng(1)
    rng.shuffle(tr_idx)
    n_val = max(1, int(len(tr_idx) * 0.1))
    va_idx, tr_idx = tr_idx[:n_val], tr_idx[n_val:]

    # 兄弟对（训练父状态的，A* 差 ≥1）
    by_parent = {}
    for j, p in enumerate(pid[tr_idx]):
        by_parent.setdefault(int(p), []).append(j)
    pb, pw = [], []
    for p, js in by_parent.items():
        js = sorted(js, key=lambda j: A[tr_idx][j])
        for a_i in range(len(js)):
            for b_i in range(a_i + 1, len(js)):
                if A[tr_idx][js[b_i]] - A[tr_idx][js[a_i]] >= 1.0:
                    pb.append(tr_idx[js[a_i]])
                    pw.append(tr_idx[js[b_i]])
    print(f"排序对: {len(pb)}")
    # 绝对索引 → tr_idx 子集内相对索引（train_generic 的 Xt 是切片数组）
    pos = {abs_i: k for k, abs_i in enumerate(tr_idx)}
    pb_rel = np.array([pos[i] for i in pb])
    pw_rel = np.array([pos[i] for i in pw])

    Xt = torch.as_tensor(X24, dtype=torch.float32)
    At = torch.as_tensor(A, dtype=torch.float32)
    net_adv = ValueNet(in_dim=24)
    net_adv, best_adv = train_generic(
        net_adv, Xt[tr_idx], At[tr_idx], Xt[va_idx], At[va_idx],
        epochs=args.epochs, pairs=(pb_rel, pw_rel),
        lambda2=args.lambda2)
    with torch.no_grad():
        p_va = net_adv(Xt[va_idx]).squeeze(-1).numpy()
    print(f"\n===== advantage 头 =====")
    print(f"val MAE={best_adv:.3f}  (A* 均值={A.mean():.2f})")

    # 兄弟 top-1（全部父状态，含 val——诊断用）
    hits, tot, hits_abs = 0, 0, 0
    per_parent_mae = []
    for p, js in by_parent.items():
        if len(js) < 2:
            continue
        idxs = tr_idx[js]
        with torch.no_grad():
            vs = net_adv(Xt[idxs]).squeeze(-1).numpy()
        ls = A[tr_idx][js]
        best_pred = int(np.argmin(vs))
        best_true = int(np.argmin(ls))
        tot += 1
        hits += int(ls[best_pred] == ls[best_true])
        per_parent_mae.append(np.abs(vs - ls).mean())
    print(f"兄弟 top-1 命中率 = {hits}/{tot} = {hits/max(1,tot):.3f} "
          f"(随机基线 ≈ {np.mean([1/len(js) for js in by_parent.values() if len(js)>=2]):.3f})")
    print(f"父内 Â-MAE = {np.mean(per_parent_mae):.3f}")

    os.makedirs(os.path.dirname(args.out_adv), exist_ok=True)
    torch.save(net_adv.state_dict(), args.out_adv)
    print(f"-> {args.out_adv}")

    # ===== M2-V 对照（绝对 C*）=====
    psplit = d["parent_split"]
    ptr = np.where(psplit == 0)[0]
    rng.shuffle(ptr)
    n_v = max(1, int(len(ptr) * 0.1))
    pva, ptr = ptr[:n_v], ptr[n_v:]
    net_abs = ValueNet(in_dim=12)
    net_abs, best_abs = train_generic(
        net_abs,
        torch.as_tensor(pX[ptr], dtype=torch.float32),
        torch.as_tensor(pC[ptr], dtype=torch.float32),
        torch.as_tensor(pX[pva], dtype=torch.float32),
        torch.as_tensor(pC[pva], dtype=torch.float32),
        epochs=args.epochs)
    with torch.no_grad():
        p_abs = net_abs(torch.as_tensor(pX[pva], dtype=torch.float32)).squeeze(-1).numpy()
    print(f"\n===== M2-V 对照（绝对 C*）=====")
    print(f"val MAE={best_abs:.3f}  (C* 均值={pC.mean():.2f})")
    torch.save(net_abs.state_dict(), args.out_abs)
    print(f"-> {args.out_abs}")


if __name__ == "__main__":
    main()
