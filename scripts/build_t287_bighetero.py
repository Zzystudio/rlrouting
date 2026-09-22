"""在 tianyan287_20q 训练子拓扑上构造强异构噪声变体（不改变耦合图）。

与 tianyan287_20q_stronghetero（重新从 101q 截取、耦合图不同）不同，本脚本
**保留 tianyan287_20q 的耦合图与比特数**，只放大边级噪声对比度，用于隔离
"噪声异构感知/规划"这一个泛化轴：

  - 按基础 two_q_gate_error 排序（70% 排序 + 30% 随机抖动，seed 固定可复现）
    将 31 条边分为好边（12 条）与坏边（19 条）
  - 好边: two_q_err ∈ [0.0008, 0.0016]，ZZ f ∈ [0.5, 1.5] kHz
  - 坏边: two_q_err ∈ [0.008, 0.011]（训练分布上沿），ZZ f ∈ [10, 20] kHz
  - t1/t2/readout/1q 误差保持基础拓扑不变（对比只来自边级噪声）
  - 好/坏边 2q 门误差均值差 ~8×，CV(edge_err) ≈ 0.65（基础拓扑 ≈ 0.3）

设计原则：所有噪声特征值保持在训练分布量级内（往上不超基础最大值，往下仅
轻微低于基础最小值——向下外推无害），只放大空间对比度；避免 OOD 特征值导致
策略崩溃混杂实验结论（首版 ×5/60kHz、×3/30kHz 均实测策略崩溃游走）。

用法: python3 scripts/build_t287_bighetero.py [--seed 42]
"""
import argparse
import json
import math
import random

TOPO = "traindata/topo/tianyan287_20q.json"
DST = "traindata/topo/tianyan287_20q_bighetero.json"
DST_LABELS = "traindata/topo/tianyan287_20q_bighetero_labels.json"
T_CX_US = 0.3
N_GOOD = 12  # 31 条边 → 12 好 / 19 坏
# 分布内对比放大：所有 err/ZZ 值都落在训练分布（基础拓扑 + ZZ 构造）的量级内，
# 只放大好/坏边的空间对比度（均值差 ~8-13×，CV 0.65 vs 基础 ~0.3）。
# 坏边 = 训练分布上沿（err ≤0.011 = 基础最大值，ZZ ≤20 kHz = 基础 ZZ 上限），
# 好边 = 训练分布下沿略下（err 0.0008-0.0016，ZZ 0.5-1.5 kHz）。
# 首版 ×5/60kHz 与 ×3/30kHz 设计均导致策略过度规避坏边 → 大绕路游走
#（perm_12_10 基础拓扑 8 swaps → TRUNC 989），故收紧到分布内。
GOOD_ERR_RANGE = (0.0008, 0.0016)
BAD_ERR_RANGE = (0.008, 0.011)
GOOD_F_KHZ = (0.5, 1.5)
BAD_F_KHZ = (10.0, 20.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=DST, help="输出拓扑路径（labels 自动取同名 _labels.json）")
    args = ap.parse_args()
    dst = args.out
    dst_labels = dst.replace(".json", "_labels.json")

    topo = json.load(open(TOPO))
    cm = [tuple(e) for e in topo["coupling_map"]]
    tqe = topo["device_params"]["two_q_gate_error"]

    def edge_err(p, q):
        return tqe.get(str((p, q)), tqe.get(str((q, p)), 0.01))

    rng = random.Random(args.seed)
    edges = sorted(cm, key=lambda e: edge_err(*e))
    n = len(edges)
    scores = [0.7 * (rank / max(1, n - 1)) + 0.3 * rng.random()
              for rank in range(n)]  # 与 edges 一一对应（已按基础 err 升序）
    order = sorted(range(n), key=lambda i: scores[i])
    good_set = {edges[i] for i in order[:N_GOOD]}

    new_tqe = {}
    crosstalk = []
    errs_good, errs_bad, ths_good, ths_bad = [], [], [], []
    for (p, q) in cm:
        is_good = ((p, q) in good_set) or ((q, p) in good_set)
        lo, hi = GOOD_ERR_RANGE if is_good else BAD_ERR_RANGE
        err = lo + (hi - lo) * rng.random()
        flo, fhi = GOOD_F_KHZ if is_good else BAD_F_KHZ
        f_khz = flo + (fhi - flo) * rng.random()
        theta = 2.0 * math.pi * f_khz * 1e3 * T_CX_US * 1e-6
        new_tqe[str((p, q))] = round(err, 6)
        new_tqe[str((q, p))] = round(err, 6)
        crosstalk.append([p, q, round(theta, 6)])
        (errs_good if is_good else errs_bad).append(err)
        (ths_good if is_good else ths_bad).append(theta)

    topo["name"] = dst.split("/")[-1].replace(".json", "")
    topo["description"] = (
        "Tianyan-287 strong noise-heterogeneity variant of the LA287 training "
        f"sub-topology (coupling graph UNCHANGED, {n} edges). Good edges "
        f"(n={len(errs_good)}): two_q_err in {GOOD_ERR_RANGE}, ZZ 0.5-1.5 kHz. "
        f"Bad edges (n={len(errs_bad)}): two_q_err in {BAD_ERR_RANGE} "
        "(training-range top), ZZ 10-20 kHz. t1/t2/readout/1q unchanged. "
        "CV(edge_err)~0.65 (base ~0.3). Good/bad gap ~8x. seed=42.")
    topo["device_params"]["two_q_gate_error"] = new_tqe
    topo["crosstalk_strength"] = crosstalk

    with open(dst, "w") as f:
        json.dump(topo, f, indent=1)

    import shutil
    shutil.copy("traindata/topo/tianyan287_20q_labels.json", dst_labels)

    all_errs = errs_good + errs_bad
    mean = sum(all_errs) / len(all_errs)
    sd = (sum((e - mean) ** 2 for e in all_errs) / len(all_errs)) ** 0.5
    print(f"已写入 {dst}（labels -> {dst_labels}）")
    print(f"好边 {len(errs_good)} 条: err [{min(errs_good):.5f}, "
          f"{max(errs_good):.5f}]  θ_ZZ [{min(ths_good):.4f}, {max(ths_good):.4f}] rad")
    print(f"坏边 {len(errs_bad)} 条: err [{min(errs_bad):.5f}, "
          f"{max(errs_bad):.5f}]  θ_ZZ [{min(ths_bad):.4f}, {max(ths_bad):.4f}] rad")
    print(f"好/坏边 err 均值比: {(sum(errs_bad)/len(errs_bad))/(sum(errs_good)/len(errs_good)):.1f}x"
          f"  全体 CV={sd/mean:.2f}  mean_err={mean:.5f}")
    print("好边列表:", sorted(good_set))


if __name__ == "__main__":
    main()
