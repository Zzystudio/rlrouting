"""为 16q grid/ring 拓扑补齐 crosstalk_strength（ZZ 串扰，物理模板与 t287 一致）。

物理推导（同 scripts/add_tianyan287_zz.py）：
  残余 ZZ 耦合频率 f_ZZ ∈ [fmin, fmax] kHz（超导横向耦合器件未完全回波消除
  的典型范围）→ θ_rad/CX = 2π·f_ZZ·t_CX（t_CX=0.3µs）→ θ ∈ [0.0038, 0.0377]
  边相关性：ZZ 与频率碰撞同源 → 按边错误率排序赋值（70% 确定性 + 30% 随机，
  seed 固定）——误差大的边倾向更强 ZZ。
适配：本脚本支持 two_q_gate_error 为 list [[q1,q2,err],...] 格式（16q 拓扑）。
原地写入 topo JSON（v16 纯路由从未读噪声字段，零影响）。

用法: PYTHONPATH=src python3 scripts/v0/add_16q_xtalk.py
"""

import argparse
import json
import math
import random

T_CX_US = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topos", nargs="+",
                    default=["traindata/topo/grid_4x4_16q.json",
                             "traindata/topo/ring_16q.json"])
    ap.add_argument("--fmin", type=float, default=1.0, help="最小残余 ZZ (kHz)")
    ap.add_argument("--fmax", type=float, default=20.0, help="最大残余 ZZ (kHz)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for topo_path in args.topos:
        topo = json.load(open(topo_path))
        cm = [tuple(e) for e in topo["coupling_map"]]
        tqe = topo["device_params"]["two_q_gate_error"]

        # list 格式 [[q1,q2,err],...]（双向各一条）
        err_map = {}
        if isinstance(tqe, list):
            for r in tqe:
                err_map[(int(r[0]), int(r[1]))] = float(r[2])
        elif isinstance(tqe, dict):
            for k, v in tqe.items():
                kk = tuple(int(x.strip()) for x in k.strip("()").split(","))
                err_map[kk] = float(v)
        else:
            for (p, q) in cm:
                err_map[(p, q)] = float(tqe)

        def edge_err(p, q):
            return err_map.get((p, q), err_map.get((q, p), 0.01))

        rng = random.Random(args.seed)
        edges = sorted(cm, key=lambda e: edge_err(*e))
        n = len(edges)
        noise = [rng.random() for _ in edges]
        crosstalk = []
        for rank, ((p, q), u) in enumerate(zip(edges, noise)):
            frac = 0.7 * (rank / max(1, n - 1)) + 0.3 * u
            f_khz = args.fmin + (args.fmax - args.fmin) * frac
            theta = 2.0 * math.pi * f_khz * 1e3 * T_CX_US * 1e-6
            crosstalk.append([p, q, round(theta, 6)])
        crosstalk.sort(key=lambda r: (r[0], r[1]))

        topo["crosstalk_strength"] = crosstalk
        with open(topo_path, "w") as f:
            json.dump(topo, f, indent=1)
        ths = [r[2] for r in crosstalk]
        print(f"{topo_path}: {len(crosstalk)} 条 ZZ, θ(rad/CX) "
              f"[{min(ths):.4f}, {max(ths):.4f}]")


if __name__ == "__main__":
    main()
