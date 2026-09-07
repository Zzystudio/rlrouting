"""
从 data/tianyan176/config.json 重新推导 tianyan176_20q.json 的 Q 标签 ↔ 0-19 映射，
输出 sidecar 文件 traindata/topo/tianyan176_20q_labels.json。

逻辑与 build_tianyan176_topo.py + build_tianyan176_20q_sub.py 完全一致，
用于追溯 20q 子图的每个物理比特对应的真机原始标签（Qxx）。
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 1. 读 config.json，推导 active qubits（复现 build_tianyan176_topo.py）──
config_path = os.path.join(ROOT, "data/tianyan176/config.json")
with open(config_path) as f:
    c = json.load(f)

dq = set(x.strip() for x in c.get("disabledQubits", "").split(",") if x.strip())
dc = set(x.strip() for x in c.get("disabledCouplers", "").split(",") if x.strip())
cmap = c["overview"]["coupler_map"]

active = sorted(set(q for g in cmap for q in cmap[g]) - dq)
qidx_66 = {q: i for i, q in enumerate(active)}
n_66 = len(active)

# ── 2. 构建 66q 邻接表和耦合边（复现 build_tianyan176_topo.py 的 edges）──
adj_66 = {i: set() for i in range(n_66)}
edges_66 = []
for g_label in sorted(cmap):
    if g_label in dc:
        continue
    p_orig, q_orig = cmap[g_label]
    if p_orig in dq or q_orig in dq:
        continue
    p, q = qidx_66[p_orig], qidx_66[q_orig]
    edges_66.append((p, q))
    adj_66[p].add(q)
    adj_66[q].add(p)

deg_66 = {i: len(adj_66[i]) for i in range(n_66)}

# ── 3. BFS 子图切分（复现 build_tianyan176_20q_sub.py）──
def bfs_subgraph(start, target=20):
    seen = {start}
    queue = list(adj_66[start])
    while len(seen) < target and queue:
        queue.sort(key=lambda x: -deg_66[x])
        nxt = queue.pop(0)
        if nxt not in seen:
            seen.add(nxt)
            queue.extend(adj_66[nxt] - seen)
    if len(seen) < target:
        return None, 0
    edges = sum(1 for p, q in edges_66 if p in seen and q in seen)
    return seen, edges

cands = []
for s in range(n_66):
    seen, edges = bfs_subgraph(s)
    if seen is not None and 24 <= edges <= 31:
        cands.append((edges, s, seen))
cands.sort(reverse=True)
best_edges, best_root, seen = cands[0]
print(f"BFS root index: {best_root}, original label: {active[best_root]}")
print(f"Selected 20 qubits: {sorted(seen)} → labels: {[active[i] for i in sorted(seen)]}")

# ── 4. 建立 0-19 ↔ Q-label 映射 ──
seen_sorted = sorted(seen)
map_0_19_to_q = {i: active[orig_idx] for i, orig_idx in enumerate(seen_sorted)}
map_q_to_0_19 = {v: k for k, v in map_0_19_to_q.items()}

# ── 5. 用现有 tianyan176_20q.json 验证子图一致性 ──
topo_20q_path = os.path.join(ROOT, "traindata/topo/tianyan176_20q.json")
with open(topo_20q_path) as f:
    topo_20q = json.load(f)
num_qubits_20 = topo_20q["num_qubits"]
coupling_map_20 = set(tuple(e) for e in topo_20q["coupling_map"])

sub_edges = sorted((p, q) for p, q in edges_66 if p in seen and q in seen)
qidx_20 = {p: i for i, p in enumerate(seen_sorted)}
reindexed_edges = set()
for p, q in sub_edges:
    reindexed_edges.add((qidx_20[p], qidx_20[q]))

assert num_qubits_20 == 20, f"num_qubits mismatch: {num_qubits_20} != 20"
assert coupling_map_20 == reindexed_edges, (
    f"coupling_map mismatch!\n"
    f"  expected: {sorted(coupling_map_20)[:5]}...\n"
    f"  got:      {sorted(reindexed_edges)[:5]}..."
)

# 比较 device_params 一致性（仅比较 t1 前 5 个值，防止整数组比较）
dp_20 = topo_20q["device_params"]
dp_66 = c["qubit"]
t1_66_used = dp_66["relatime"]["T1"]["qubit_used"]
t1_66_vals = dp_66["relatime"]["T1"]["param_list"]
t1_66_map = dict(zip(t1_66_used, t1_66_vals))
for i, orig_idx in enumerate(seen_sorted):
    q_label = active[orig_idx]
    expected = dp_20["t1_times"][i]
    got = t1_66_map[q_label]
    assert abs(expected - got) < 1e-6, f"t1 mismatch at qubit {i} ({q_label}): {expected} != {got}"

print(f"Verified: regenerated 20q subgraph matches {topo_20q_path}")
print(f"  {num_qubits_20} qubits, {len(coupling_map_20)} edges")

# ── 6. 输出 sidecar ──
out = {
    "description": "Q-label mapping for tianyan176_20q_sub. "
                   f"Root Q23 (index {best_root}), {best_edges} edges. "
                   f"Regenerated from {config_path}.",
    "to_0_19": map_q_to_0_19,
    "from_0_19": map_0_19_to_q,
}
out_path = os.path.join(ROOT, "traindata/topo/tianyan176_20q_labels.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=1, sort_keys=True)
print(f"Saved: {out_path}")
print(f"0-19 → Q-labels: {map_0_19_to_q}")
