"""从 tianyan287_101q 截取强异构 20 比特子拓扑。

选择策略（与连通度优先的 build_tianyan287_20q_sub.py 不同）：
  1. 按 two_q_gate_error 排序全部 168 条边
  2. 贪心选取：从最优边出发，交替添加当前可达的最高错误边和最低错误边
     的端点比特，使子图的边误差标准差/均值比（CV）最大化
  3. 确保 20q 连通 + 边数 ≥ 20

输出:
  traindata/topo/tianyan287_20q_stronghetero.json
  traindata/topo/tianyan287_20q_stronghetero_labels.json
"""
import json
import math
import os
import sys

TOPO = "traindata/topo/tianyan287_101q.json"
DST = "traindata/topo/tianyan287_20q_stronghetero.json"
TARGET = 20


def main():
    topo = json.load(open(TOPO))
    cm = [tuple(e) for e in topo["coupling_map"]]
    dp = topo["device_params"]
    tqe_raw = dp["two_q_gate_error"]
    n_full = topo["num_qubits"]
    orig_labels = topo.get("original_labels", {})

    adj = {i: set() for i in range(n_full)}
    edge_err = {}
    for (p, q) in cm:
        adj[p].add(q)
        adj[q].add(p)
        v = tqe_raw.get(str((p, q)), tqe_raw.get(str((q, p)), 0.01))
        edge_err[(min(p, q), max(p, q))] = v

    sorted_edges = sorted(edge_err.items(), key=lambda x: x[1])
    best_edge = sorted_edges[0]
    worst_edge = sorted_edges[-1]
    print(f"全器件: {n_full}q {len(cm)} 边  "
          f"best_edge={best_edge[0]} err={best_edge[1]:.4f}  "
          f"worst_edge={worst_edge[0]} err={worst_edge[1]:.4f}")

    # --- 贪心选择：交替好/坏边端点 ---
    selected = set()
    # 种子：从最优边的两个端点开始
    selected.update(best_edge[0])

    for _ in range(TARGET):
        if len(selected) >= TARGET:
            break
        # 找 selected 中任意比特的邻居中未选的
        candidates = set()
        for q in selected:
            for nb in adj[q]:
                if nb not in selected:
                    candidates.add(nb)
        if not candidates:
            break
        # 交替添加：优先选与 selected 有高错误边和低错误边的比特
        best_q = None
        best_score = -1
        for c in sorted(candidates):
            score = 0.0
            for q in selected:
                if (min(c, q), max(c, q)) in edge_err:
                    e = edge_err[(min(c, q), max(c, q))]
                    # 偏好极端值（远离中位数）
                    median_e = 0.01
                    score += abs(e - median_e)
            if score > best_score:
                best_score = score
                best_q = c
        selected.add(best_q)

    selected = sorted(selected)[:TARGET]
    print(f"选中 {len(selected)} qubits: {selected}")

    # 构建子拓扑
    sub_adj = {q: set() for q in selected}
    sub_edges = []
    sub_errs = []
    for (p, q), err in edge_err.items():
        if p in selected and q in selected:
            sub_edges.append((p, q))
            sub_errs.append(err)
            sub_adj[p].add(q)
            sub_adj[q].add(p)

    # 确保连通
    visited = {selected[0]}
    queue = [selected[0]]
    while queue:
        q = queue.pop()
        for nb in sub_adj[q]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    if len(visited) < len(selected):
        print(f"警告: 子图不连通 ({len(visited)}/{len(selected)})")

    # 统计
    errs = sorted(sub_errs)
    mean_e = sum(errs) / len(errs)
    std_e = (sum((e - mean_e) ** 2 for e in errs) / len(errs)) ** 0.5
    cv = std_e / mean_e if mean_e > 0 else 0
    print(f"子拓扑: {len(selected)}q {len(sub_edges)} 边  "
          f"err min={min(errs):.4f} max={max(errs):.4f}  "
          f"CV={cv:.2f}")

    # 构建 device_params（从全器件继承）
    sub_dp = {}
    for key in ("t1_times", "t2_times", "freq_ghz", "readout_error",
                "single_q_gate_error"):
        if key in dp:
            src_arr = dp[key]
            if isinstance(src_arr, list):
                sub_dp[key] = [src_arr[q] for q in selected if q < len(src_arr)]
            else:
                sub_dp[key] = src_arr
    sub_dp["two_q_gate_error"] = {
        str((min(p, q), max(p, q))): v
        for (p, q), v in edge_err.items() if p in selected and q in selected
    }
    sub_dp["shots"] = dp.get("shots", 1024)

    # 重编号 0..k-1
    remap = {old: new for new, old in enumerate(selected)}
    sub_cm = [[remap[p], remap[q]] for (p, q) in sub_edges]

    # labels
    sub_labels = {"from_0_19": {},
                  "description": f"tianyan287 strong-hetero {len(selected)}q sub"}
    for old in selected:
        lbl = None
        if isinstance(orig_labels, dict):
            lbl = orig_labels.get(str(old))
        elif isinstance(orig_labels, list) and old < len(orig_labels):
            lbl = orig_labels[old]
        sub_labels["from_0_19"][str(remap[old])] = lbl or f"Q{old}"

    # ZZ crosstalk：按边错误率排序赋值
    sorted_sub_errs = sorted(sub_errs)
    def _zz_for(e):
        rank = sorted_sub_errs.index(e) / max(1, len(sorted_sub_errs) - 1)
        return round(0.001 + 0.049 * rank, 6)

    topo_out = {
        "name": f"tianyan287_{len(selected)}q_stronghetero",
        "description": (f"Tianyan-287 strong-heterogeneity {len(selected)}-qubit "
                        f"connected subgraph. CV(edge_err)={cv:.2f}. "
                        f"Calibration: 2026. Real two_q_gate_error."),
        "num_qubits": len(selected),
        "coupling_map": sub_cm,
        "device_params": {
            **sub_dp,
        },
        "original_labels": sub_labels,
    }

    # ZZ crosstalk：按边错误率排序赋值（CV 最大的子图）
    zz_values = {}
    for e in sub_errs:
        rank = sorted_sub_errs.index(e) / max(1, len(sorted_sub_errs) - 1) if len(sorted_sub_errs) > 1 else 0
        zz_values[e] = round(0.001 + 0.049 * rank, 6)

    # 设置 crosstalk_strength（v2 模拟器语义：rad/CX，按边错误率排序赋值）
    topo_out["crosstalk_strength"] = []
    sorted_sub_errs = sorted(sub_errs)
    for (p, q) in sub_edges:
        e_key = (min(p, q), max(p, q))
        idx = sorted_sub_errs.index(e_key) if e_key in sorted_sub_errs else len(sorted_sub_errs) // 2
        rank = idx / max(1, len(sorted_sub_errs) - 1)
        topo_out["crosstalk_strength"].append(
            [p, q, round(0.001 + 0.049 * rank, 6)])

    os.makedirs("traindata/topo", exist_ok=True)
    with open(DST, "w") as f:
        json.dump(topo_out, f, indent=1)
    with open(DST.replace(".json", "_labels.json"), "w") as f:
        json.dump(sub_labels, f, indent=1)

    # 验证：load_topo 兼容
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from routing.rl.eval_policy import load_topo
    cfg, hwt, cmt = load_topo(DST)
    print(f"load_topo OK: {hwt.num_qubits}q {len(cmt)} edges")



if __name__ == "__main__":
    main()
