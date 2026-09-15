#!/usr/bin/env python3
"""审计训练 QASM 池（V2 验收）：族×规模覆盖矩阵 + front-layer 宽度/并发分布。

对 gen_structured + gen_structured_v2 统一计算（不依赖 manifest），
输出控制台报告与 JSON（logs/audit_training_circuits.json）。

用法: python3 scripts/audit_training_circuits.py
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from qiskit.qasm2 import load as qasm2_load

DIRS = ["traindata/gen_structured", "traindata/gen_structured_v2"]
OUT = "logs/audit_training_circuits.json"


def metrics_for_qc(qc):
    gates = []
    for inst in qc.data:
        qs = [qc.find_bit(q).index for q in inst.qubits]
        gates.append((inst.operation.name, qs))
    q_level = [0] * qc.num_qubits
    waves = {}
    n2 = 0
    cx_dists = []
    for name, qs in gates:
        lv = max(q_level[q] for q in qs) + 1
        for q in qs:
            q_level[q] = lv
        waves.setdefault(lv, []).append((name, qs))
        if len(qs) == 2:
            n2 += 1
            cx_dists.append(abs(qs[0] - qs[1]))
    widths = [len(v) for _, v in sorted(waves.items())]
    two_q_waves = []
    conc_fracs = []
    for _, v in waves.items():
        g2 = [(nm, qs) for nm, qs in v if len(qs) == 2]
        if g2:
            two_q_waves.append(len(g2))
        if len(g2) >= 2:
            tot = disjoint = 0
            for i in range(len(g2)):
                for j in range(i + 1, len(g2)):
                    tot += 1
                    if not set(g2[i][1]) & set(g2[j][1]):
                        disjoint += 1
            conc_fracs.append(disjoint / tot)
    return {
        "front_width_mean": round(float(np.mean(widths)), 2) if widths else 0.0,
        "front_width_max": int(max(widths)) if widths else 0,
        "two_q_wave_mean": round(float(np.mean(two_q_waves)), 2) if two_q_waves else 0.0,
        "concurrency_mean": round(float(np.mean(conc_fracs)), 3) if conc_fracs else 0.0,
        "cx_dist_mean": round(float(np.mean(cx_dists)), 2) if cx_dists else 0.0,
        "cx_dist_max": int(max(cx_dists)) if cx_dists else 0,
    }


def family_of(name: str) -> str:
    parts = name.replace(".qasm", "").split("_")
    # 前缀族名：取前两个 token，剥掉数字/标签
    fam = []
    for p in parts:
        if p.isdigit():
            break
        if p.startswith(("w", "lr", "inv")) and fam:
            break
        fam.append(p)
        if len(fam) == 2:
            break
    return "_".join(fam) if fam else parts[0]


def main():
    rows = []
    for d in DIRS:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".qasm"):
                continue
            qc = qasm2_load(os.path.join(d, fname))
            if qc.num_qubits > 20:
                continue
            m = metrics_for_qc(qc)
            m["file"] = fname
            m["dir"] = os.path.basename(d)
            m["family"] = family_of(fname)
            m["qubits"] = qc.num_qubits
            m["gates"] = qc.size()
            rows.append(m)

    # 覆盖矩阵 family × scale
    matrix = defaultdict(lambda: defaultdict(int))
    for r in rows:
        matrix[r["family"]][r["qubits"]] += 1
    scales = sorted({r["qubits"] for r in rows})
    fams = sorted(matrix.keys())

    print(f"总电路数: {len(rows)}  （目录: {DIRS}）")
    print(f"\n{'family':<22}" + "".join(f"{s:>6}" for s in scales) + f"{'total':>8}")
    print("-" * (22 + 6 * len(scales) + 8))
    for fam in fams:
        cells = "".join(f"{matrix[fam].get(s, 0):>6}" for s in scales)
        total = sum(matrix[fam].values())
        print(f"{fam:<22}{cells}{total:>8}")

    # 宽度/并发分布
    print("\n并发宽度谱（front_width_mean 分布）:")
    all_w = [r["front_width_mean"] for r in rows]
    hist_edges = [0, 2, 4, 6, 8, 10, 15, 20, 100]
    for lo, hi in zip(hist_edges, hist_edges[1:]):
        cnt = sum(1 for w in all_w if lo <= w < hi)
        print(f"  [{lo:>2},{hi:>3}): {cnt:>4}  {'#' * min(60, cnt)}")
    print(f"\nconcurrency_mean: mean={np.mean([r['concurrency_mean'] for r in rows]):.3f} "
          f"max={max(r['concurrency_mean'] for r in rows):.3f}")
    print(f"cx_dist_mean: mean={np.mean([r['cx_dist_mean'] for r in rows]):.2f} "
          f"max={max((r['cx_dist_max'] for r in rows), default=0)}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"n_total": len(rows), "matrix": {k: dict(v) for k, v in matrix.items()},
                   "rows": rows}, f, indent=1)
    print(f"\n已写入 {OUT}")


if __name__ == "__main__":
    main()
