import json

topo = json.load(open("traindata/topo/tianyan176_66q.json"))
cm = [tuple(e) for e in topo["coupling_map"]]
n = topo["num_qubits"]
adj = {i: set() for i in range(n)}
for p, q in cm:
    adj[p].add(q)
    adj[q].add(p)
deg = {i: len(adj[i]) for i in range(n)}

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
        return None, 0
    edges = sum(1 for p, q in cm if p in seen and q in seen)
    return seen, edges

cands = []
for s in range(n):
    seen, edges = bfs_subgraph(s)
    if seen is not None and 24 <= edges <= 31:
        cands.append((edges, s, seen))
cands.sort(reverse=True)
edges, s, seen = cands[0]
sub_edges = sorted((p, q) for p, q in cm if p in seen and q in seen)
qidx = {p: i for i, p in enumerate(sorted(seen))}

dp = topo["device_params"]
twoq = {}
for (p, q) in sub_edges:
    k1, k2 = qidx[p], qidx[q]
    v = dp["two_q_gate_error"].get(str((p, q)))
    if v is None:
        v = dp["two_q_gate_error"].get(str((q, p)))
    twoq[str((k1, k2))] = v
    twoq[str((k2, k1))] = v

out = {
    "name": "tianyan176_20q_sub",
    "description": "20-qubit connected subgraph cut from Tianyan-176 effective "
                   "topology (root Q23, {} edges, avg_deg {:.2f}). "
                   "Full topo: traindata/topo/tianyan176_66q.json".format(
                       len(sub_edges), 2 * len(sub_edges) / 20.0),
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
}
with open("traindata/topo/tianyan176_20q.json", "w") as f:
    json.dump(out, f, indent=1)
print("saved traindata/topo/tianyan176_20q.json")
print("qubits:", out["num_qubits"], "edges:", len(sub_edges))
print("original qubit labels:", sorted(seen))
print("coupling_map:", out["coupling_map"])
