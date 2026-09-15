"""构造强异构拓扑：tianyan176_20q 的噪声异构幅度放大 3×。

耦合图不变（所有模型兼容），将 two_q_gate_error 和 ZZ 串扰的异构幅度
从 ~6× 拉大到 ~18×，使得边质量选择的一阶贡献显著增大：
  new_err = clip(mean + 3×(orig − mean), 0.001, 0.1)
  ZZ = 0.1 × new_err（与回退公式同比例，但量级更大）
其余参数（T1/T2/freq/readout/single_q_err/shots）不变。

用法: python3 scripts/build_stronghetero_topo.py [--amp 3.0]
"""
import argparse
import json
import statistics

SRC = "traindata/topo/tianyan176_20q.json"
DST = "traindata/topo/tianyan176_20q_stronghetero.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--amp", type=float, default=3.0)
    args = ap.parse_args()

    topo = json.load(open(args.src))
    cm = [tuple(e) for e in topo["coupling_map"]]
    dp = topo["device_params"]
    tqe = dp["two_q_gate_error"]

    def edge_err(p, q):
        return tqe.get(str((p, q)), tqe.get(str((q, p)), 0.01))

    orig_errs = [edge_err(*e) for e in cm]
    mean_err = statistics.mean(orig_errs)

    new_tqe = {}
    new_cts = []
    for (p, q) in cm:
        orig = edge_err(p, q)
        amp = mean_err + args.amp * (orig - mean_err)
        amp = max(0.001, min(0.1, amp))
        new_tqe[str((p, q))] = round(amp, 6)
        new_tqe[str((q, p))] = round(amp, 6)
        zz = round(0.1 * amp, 6)
        new_cts.append([p, q, zz])

    topo["device_params"]["two_q_gate_error"] = new_tqe
    topo["crosstalk_strength"] = new_cts
    topo["description"] = (topo.get("description", "") +
        f"  [strong-hetero] 噪声异构幅度 ×{args.amp}（原始均值 {mean_err:.4f}）")

    with open(args.dst, "w") as f:
        json.dump(topo, f, indent=1)

    new_vals = [new_tqe[str((p, q))] for (p, q) in cm]
    orig_sorted = sorted(orig_errs)
    new_sorted = sorted(new_vals)
    print(f"已写入 {args.dst}")
    print(f"原始 two_q_err: min={min(orig_errs):.4f} max={max(orig_errs):.4f} "
          f"spread={max(orig_errs)/min(orig_errs):.1f}x")
    print(f"增强 two_q_err: min={min(new_vals):.4f} max={max(new_vals):.4f} "
          f"spread={max(new_vals)/min(new_vals):.1f}x")
    print(f"ZZ 串强: min={min(r[2] for r in new_cts):.4f} "
          f"max={max(r[2] for r in new_cts):.4f} rad/CX")
    print(f"原始 vs 增强的边误差排序: "
          f"{'一致' if orig_sorted == sorted(orig_errs) else '不一致'}")


if __name__ == "__main__":
    main()
