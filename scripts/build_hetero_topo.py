"""构造强异构拓扑：tianyan176_20q_hetero.json

耦合图与 tianyan176_20q 完全一致（29 边），噪声参数按真实器件错误率排序分档：
  好边（后 1/3 低错误）: two_q_gate_error 0.002, ZZ 0.002 rad
  中边（中间 1/3）:      two_q_gate_error 0.010, ZZ 0.010 rad
  坏边（前 1/3 高错误）: two_q_gate_error 0.050, ZZ 0.050 rad
（25× 差异，模拟真实器件的好/坏边分离；边噪声与 ZZ 同源——频率碰撞边
既退极化高又 ZZ 强，故按原始错误率排序赋值。）
seed=42 可复现；原始文件不改动（只输出新文件）。
"""
import argparse
import json
import math
import random

SRC = "traindata/topo/tianyan176_20q.json"
DST = "traindata/topo/tianyan176_20q_hetero.json"
T_CX_US = 0.3

GOOD = (0.002, 0.002)
MID = (0.010, 0.010)
BAD = (0.050, 0.050)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    topo = json.load(open(args.src))
    cm = [tuple(e) for e in topo["coupling_map"]]
    dp = topo["device_params"]
    tqe_orig = dp["two_q_gate_error"]

    def edge_err(p, q):
        v = tqe_orig.get(str((p, q)), tqe_orig.get(str((q, p)), 0.01))
        return v if v is not None else 0.01

    # 按原始器件错误率排序分三档（保持真实结构相关性）
    rng = random.Random(args.seed)
    edges = sorted(cm, key=lambda e: edge_err(*e))
    n = len(edges)
    g = n // 3

    tiers = {}
    for i, e in enumerate(edges):
        if i < g:
            tiers[e] = GOOD
        elif i < 2 * g:
            tiers[e] = MID
        else:
            tiers[e] = BAD

    new_tqe = {}
    new_cts = []
    for (p, q), (err, zz) in tiers.items():
        new_tqe[str((p, q))] = err
        new_tqe[str((q, p))] = err
        theta = 2.0 * math.pi * zz * 1e3 * T_CX_US * 1e-6  # rad/CX
        new_cts.append([p, q, round(theta, 6)])

    topo["device_params"]["two_q_gate_error"] = new_tqe
    topo["crosstalk_strength"] = new_cts
    topo["description"] = (topo.get("description", "") +
        "  [hetero] 强异构变体：按原始错误率分档 好0.002/中0.01/坏0.05，ZZ 同量级")
    with open(args.dst, "w") as f:
        json.dump(topo, f, indent=1)

    import collections
    cnt = collections.Counter(v[0] for v in tiers.values())
    print(f"已写入 {args.dst}: {n} 条边  分档计数: {dict(cnt)}")
    print("坏边（two_q_gate_error=0.05, ZZ=0.05 rad）:")
    for (p, q), (err, zz) in sorted(tiers.items()):
        if err == BAD[0]:
            print(f"  ({p},{q}) 原错误率 {edge_err(p, q):.4f}")


if __name__ == "__main__":
    main()
