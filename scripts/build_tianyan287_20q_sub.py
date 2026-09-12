"""从 tianyan287_101q.json 截取联通度最高的 20 比特连通子拓扑。

两种启发式（对全部 101 个根节点各跑一遍，取并集择优）：
  1) BFS 度优先扩张（与 build_tianyan176_20q_sub.py 完全一致，保证可比）；
  2) 贪心最大内边增益扩张（每步加入使内部边数增加最多的节点）。

选择标准：内部边数最多；并列时取平均两比特门误差更小者。

用法:
    python3 scripts/build_tianyan287_20q_sub.py

输出:
    traindata/topo/tianyan287_20q.json          子拓扑（0-19 重编号）
    traindata/topo/tianyan287_20q_labels.json   0-19 <-> 天衍原始 Q 标签
"""
import json

topo = json.load(open("traindata/topo/tianyan287_101q.json"))
cm = [tuple(e) for e in topo["coupling_map"]]
n = topo["num_qubits"]
orig_labels = topo["original_labels"]
dp = topo["device_params"]

adj = {i: set() for i in range(n)}
for p, q in cm:
    adj[p].add(q)
    adj[q].add(p)
deg = {i: len(adj[i]) for i in range(n)}


def edge_err(p, q):
    v = dp["two_q_gate_error"].get(str((p, q)))
    if v is None:
        v = dp["two_q_gate_error"].get(str((q, p)))
    return v if v is not None else 0.01


def bfs_subgraph(start, target=20):
    seen = {start}
    queue = list(adj[start])
    while len(seen) < target and queue:
        queue.sort(key=lambda x: -deg[x])
        nxt = queue.pop(0)
        if nxt not in seen:
            seen.add(nxt)
            queue.extend(adj[nxt] - seen)
    if len(seen) < target:
        return None
    return seen


def greedy_subgraph(start, target=20):
    seen = {start}
    while len(seen) < target:
        best, best_gain = None, -1
        for node in range(n):
            if node in seen:
                continue
            gain = sum(1 for nb in adj[node] if nb in seen)
            if gain > best_gain:
                best, best_gain = node, gain
        if best is None or best_gain == 0:
            return None
        seen.add(best)
    return seen


def sub_edges_of(seen):
    return sorted((p, q) for p, q in cm if p in seen and q in seen)


def stats(seen):
    sub_edges = sub_edges_of(seen)
    mean_err = sum(edge_err(p, q) for p, q in sub_edges) / len(sub_edges)
    return len(sub_edges), mean_err


cands = {}
for s in range(n):
    for fn in (bfs_subgraph, greedy_subgraph):
        seen = fn(s)
        if seen is None:
            continue
        key = frozenset(seen)
        if key not in cands:
            cands[key] = seen

print(f"candidate subgraphs: {len(cands)}")
scored = sorted(
    ((stats(seen)[0], -stats(seen)[1], min(seen), seen) for seen in cands.values()),
    reverse=True,
)
best_edges, neg_err, _, seen = scored[0]
mean_err = -neg_err
print(f"best: {best_edges} edges, mean 2q err {mean_err:.5f}, "
      f"avg_deg {2 * best_edges / 20.0:.2f}")
print("top 5:")
for e, ne, m, _ in scored[:5]:
    print(f"  edges={e} mean2qerr={-ne:.5f} root={orig_labels[m]}")

sub_edges = sub_edges_of(seen)
qidx = {p: i for i, p in enumerate(sorted(seen))}

twoq = {}
for (p, q) in sub_edges:
    k1, k2 = qidx[p], qidx[q]
    v = edge_err(p, q)
    twoq[str((k1, k2))] = v
    twoq[str((k2, k1))] = v

out = {
    "name": "tianyan287_20q_sub",
    "description": "20-qubit connected subgraph cut from Tianyan-287 effective "
                   "topology (root {}, {} edges, avg_deg {:.2f}, mean 2q err {:.5f}). "
                   "Full topo: traindata/topo/tianyan287_101q.json".format(
                       orig_labels[min(seen)], len(sub_edges),
                       2 * len(sub_edges) / 20.0, mean_err),
    "num_qubits": 20,
    "coupling_map": [[qidx[p], qidx[q]] for p, q in sub_edges],
    "device_params": {
        "t1_times": [dp["t1_times"][i] for i in sorted(seen)],
        "t2_times": [dp["t2_times"][i] for i in sorted(seen)],
        "freq_ghz": [dp["freq_ghz"][i] for i in sorted(seen)],
        "readout_error": [dp["readout_error"][i] for i in sorted(seen)],
        "single_q_gate_error": [dp["single_q_gate_error"][i] for i in sorted(seen)],
        "two_q_gate_error": twoq,
        "shots": 1024,
    },
    "original_labels": [orig_labels[i] for i in sorted(seen)],
}
with open("traindata/topo/tianyan287_20q.json", "w") as f:
    json.dump(out, f, indent=1)

map_0_19_to_q = {k: orig_labels[orig_idx] for k, orig_idx in enumerate(sorted(seen))}
sidecar = {
    "description": "Q-label mapping for tianyan287_20q_sub. "
                   f"Root {orig_labels[min(seen)]}, {len(sub_edges)} edges. "
                   "Regenerated from traindata/topo/tianyan287_101q.json.",
    "to_0_19": {v: k for k, v in map_0_19_to_q.items()},
    "from_0_19": map_0_19_to_q,
}
with open("traindata/topo/tianyan287_20q_labels.json", "w") as f:
    json.dump(sidecar, f, indent=1, sort_keys=True)

print("saved traindata/topo/tianyan287_20q.json + tianyan287_20q_labels.json")
print("qubits:", out["num_qubits"], "edges:", len(sub_edges))
print("0-19 -> Q-labels:", map_0_19_to_q)
print("coupling_map:", out["coupling_map"])
