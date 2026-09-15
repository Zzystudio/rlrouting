"""为 tianyan287_20q 构造典型量级的 ZZ 串扰强度（crosstalk_strength）。

天衍校准数据不含 ZZ 参数（doc/TIANYAN_CLOUD_GUIDE.md），按超导横向耦合
器件的典型残余 ZZ 量级构造：
  f_ZZ ∈ [1, 20] kHz（未完全回波消除的残余 ZZ 耦合典型范围）
  θ_rad/CX = 2π · f_ZZ · t_CX，t_CX = 0.3 µs → θ ∈ [0.0038, 0.0377] rad
  与合成拓扑 cross_5q_hetero.json 的 0.002-0.012 rad 量级一致。

边相关性：ZZ 耦合与频率碰撞同源，按 two_q_gate_error 排序赋值（70% 确定性
排序 + 30% 随机扰动，seed 固定可复现）——误差大的边倾向更强的 ZZ。

语义（v2 模拟器/奖励）：crosstalk_strength = 每标称 CX 的 ZZ 旋转角（rad），
GNN 的 zz 归一化为 strength / _ERROR_SCALE(0.05)。

用法: python3 scripts/add_tianyan287_zz.py [--fmin 1.0] [--fmax 20.0]
"""
import argparse
import json
import math
import random

TOPO = "traindata/topo/tianyan287_20q.json"
T_CX_US = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default=TOPO)
    ap.add_argument("--fmin", type=float, default=1.0, help="最小残余 ZZ (kHz)")
    ap.add_argument("--fmax", type=float, default=20.0, help="最大残余 ZZ (kHz)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    topo = json.load(open(args.topo))
    cm = [tuple(e) for e in topo["coupling_map"]]
    tqe = topo["device_params"]["two_q_gate_error"]

    def edge_err(p, q):
        return tqe.get(str((p, q)), tqe.get(str((q, p)), 0.01))

    # 按边错误率排序（70% 排序 + 30% 随机 → 与噪声水平弱相关）
    rng = random.Random(args.seed)
    edges = sorted(cm, key=lambda e: edge_err(*e))
    n = len(edges)
    noise = [rng.random() for _ in edges]
    crosstalk = []
    for rank, ((p, q), u) in enumerate(zip(edges, noise)):
        frac = 0.7 * (rank / max(1, n - 1)) + 0.3 * u
        f_khz = args.fmin + (args.fmax - args.fmin) * frac
        theta = 2.0 * math.pi * f_khz * 1e3 * T_CX_US * 1e-6  # rad per CX
        crosstalk.append([p, q, round(theta, 6), round(f_khz, 2)])
    crosstalk.sort(key=lambda r: (r[0], r[1]))

    topo["crosstalk_strength"] = [[p, q, th] for (p, q, th, _f) in crosstalk]
    with open(args.topo, "w") as f:
        json.dump(topo, f, indent=1)

    ths = [r[2] for r in crosstalk]
    print(f"已写入 {args.topo}: {len(crosstalk)} 条边的 crosstalk_strength")
    print(f"θ(rad/CX): min={min(ths):.4f} max={max(ths):.4f} "
          f"mean={sum(ths)/len(ths):.4f}")
    print(f"对应 f_ZZ(kHz): min={min(r[3] for r in crosstalk):.1f} "
          f"max={max(r[3] for r in crosstalk):.1f}")
    for r in crosstalk[:5]:
        print(f"  ({r[0]},{r[1]}): θ={r[2]:.4f} rad  f={r[3]:.1f} kHz")


if __name__ == "__main__":
    main()
