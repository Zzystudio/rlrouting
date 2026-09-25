"""Sprint 1 Gate L1/L2 决策分析。

Gate L1（搜索有效）：headroom 分层（line×{perm_mix,random}×{中,高门档}）上
  MCTS-1@100 配对 vs SABRE：≤SABRE 率 ≥60% 且 Wilcoxon p<0.05。
用法:
    PYTHONPATH=src python3 scripts/v0/analyze_gate.py \
        --results benchmark/v0_s1_line_16q.jsonl,benchmark/v0_s1_ring_16q.jsonl,benchmark/v0_s1_grid_4x4_16q.jsonl
"""

import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))


def wilcoxon_p(a, b):
    """配对 Wilcoxon signed-rank 双侧 p 值（无 scipy）。"""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n < 3:
        return 1.0
    absd = np.abs(d)
    order = np.argsort(np.argsort(absd)) + 1.0
    ranks = order.copy()
    for v in np.unique(absd):
        mask = absd == v
        if mask.sum() > 1:
            ranks[mask] = order[mask].mean()
    T = float(ranks[d > 0].sum())
    T = min(T, n * (n + 1) / 2 - T)
    mu = n * (n + 1) / 4
    sig = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (T - mu) / sig
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return max(p, 1e-12)


def load(path):
    rows = []
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r["ok"] and r["swaps"] is not None:
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results",
                    default="benchmark/v0_s1_line_16q.jsonl,benchmark/v0_s1_ring_16q.jsonl,benchmark/v0_s1_grid_4x4_16q.jsonl")
    args = ap.parse_args()

    per_topo = []
    for p in args.results.split(","):
        if os.path.exists(p):
            name = os.path.basename(p).replace("v0_s1_", "").replace(".jsonl", "")
            per_topo.append((name, load(p)))
            print(f"载入 {name}: {len(per_topo[-1][1])} 行")

    if not per_topo:
        print("无结果文件")
        return

    # ---- 各拓扑 × 方法 mean swaps + 配对 vs SABRE ----
    print("\n===== 各拓扑方法汇总 =====")
    for topo_name, rows in per_topo:
        methods = {}
        for r in rows:
            methods.setdefault(r["method"], []).append(r["swaps"])
        sabre_map = {r["file"]: r["swaps"] for r in rows if r["method"] == "sabre"}
        print(f"\n[{topo_name}] 实例数≈{len(rows) // max(1, len(methods))}")
        for m in ["sabre", "gdist", "gsabre", "m0@20", "m1@20", "m1@100", "m1@1000"]:
            if m not in methods:
                continue
            v = np.array(methods[m])
            vals = [r["swaps"] for r in rows if r["method"] == m]
            if m == "sabre":
                print(f"  {m:<10} mean={v.mean():6.1f}")
                continue
            common = [f for f in sabre_map if any(r["file"] == f and r["method"] == m for r in rows)]
            if common:
                a = np.array([sabre_map[f] for f in common])
                b = np.array([next(r["swaps"] for r in rows if r["method"] == m and r["file"] == f) for f in common])
                print(f"  {m:<10} mean={v.mean():6.1f}  ≤SABRE率={float((b<=a).mean()):.2f} "
                      f"配对n={len(common)} p={wilcoxon_p(b,a):.4f}")
            else:
                print(f"  {m:<10} mean={v.mean():6.1f}  (无配对)")

    # ---- Gate L1：line_16q headroom 分层 ----
    line_rows = dict(per_topo).get("line_16q", [])
    if line_rows:
        print(f"\n===== Gate L1：line_16q headroom 分层（M1@100 vs SABRE）=====")
        sabre_map = {r["file"]: r["swaps"] for r in line_rows if r["method"] == "sabre"}
        m1_map = {r["file"]: r["swaps"] for r in line_rows if r["method"] == "m1@100"}
        for fam in ["perm_mix", "random"]:
            for tier, lo, hi in [("mid", 13, 20), ("high", 21, 30)]:
                common = [f for f in sabre_map if f in m1_map
                          and any(r["file"] == f and r["family"] == fam and lo <= r["n2q"] <= hi
                                  for r in line_rows)]
                if len(common) < 3:
                    print(f"{fam}/{tier:<5} 配对不足({len(common)})")
                    continue
                a = np.array([sabre_map[f] for f in common])
                b = np.array([m1_map[f] for f in common])
                print(f"{fam}/{tier:<5} n={len(common):>3} SABRE={a.mean():6.1f} "
                      f"M1@100={b.mean():6.1f} ≤SABRE率={float((b<=a).mean()):.2f} "
                      f"p={wilcoxon_p(b,a):.4f}")

    # ---- Gate L2：汇总已生成的阶梯 JSON ----
    print("\n===== Gate L2：标签阶梯（来自 benchmark/v0_labels_*.json）=====")
    for f in sorted(os.listdir("benchmark")):
        if f.startswith("v0_labels_") and f.endswith(".json"):
            data = json.load(open(os.path.join("benchmark", f)))
            rows = data["rows"]
            for key, name in [("episode", "episode"), ("m1_150", "best@150")]:
                vals = np.array([r[key] for r in rows if key in r and np.isfinite(r[key])])
                vs = np.array([r["vstar"] for r in rows if key in r and np.isfinite(r[key])])
                if len(vals):
                    mae = float(np.abs(vals - vs).mean())
                    corr = float(np.corrcoef(vals, vs)[0, 1]) if len(vals) > 1 else 0
                    print(f"  {f.replace('v0_labels_','').replace('.json',''):<10} "
                          f"{name}: MAE={mae:.2f} corr={corr:.2f} n={len(vals)}")


if __name__ == "__main__":
    main()
