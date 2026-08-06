import json
import numpy as np

c = json.load(open("data/tianyan176/config.json"))

def parse_csv(v):
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    return list(v)

dq = set(parse_csv(c["disabledQubits"]))
dc = set(parse_csv(c["disabledCouplers"]))
cmap = c["overview"]["coupler_map"]

qu = c["qubit"]
cz = c["twoQubitGate"]["czGate"]
ra = c["readout"]["readoutArray"]

f01_used = qu["frequency"]["f01"]["qubit_used"]
t1_used = qu["relatime"]["T1"]["qubit_used"]
t2_used = qu["relatime"]["T2"]["qubit_used"]
ge_used = qu["singleQubit"]["gate error"]["qubit_used"]
ro_used = ra["Readout Error"]["qubit_used"]

f01 = dict(zip(f01_used, qu["frequency"]["f01"]["param_list"]))
t1 = dict(zip(t1_used, qu["relatime"]["T1"]["param_list"]))
t2 = dict(zip(t2_used, qu["relatime"]["T2"]["param_list"]))
ge = dict(zip(ge_used, qu["singleQubit"]["gate error"]["param_list"]))
ro = dict(zip(ro_used, ra["Readout Error"]["param_list"]))
czerr = dict(zip(cz["gate error"]["qubit_used"], cz["gate error"]["param_list"]))

active = sorted(set(q for g in cmap for q in cmap[g]) - dq)
qidx = {q: i for i, q in enumerate(active)}
n = len(active)

edges = []
for g in sorted(cmap):
    if g in dc:
        continue
    p, q = cmap[g]
    if p in dq or q in dq:
        continue
    edges.append((qidx[p], qidx[q]))

def get(d, q, default):
    return d.get(q, default)

t1_arr = [get(t1, q, 28.4) for q in active]
t2_arr = [get(t2, q, 20.27) for q in active]
# Aer thermal_relaxation_error 要求 T2 <= 2*T1；校准数据偶有 T2 > 2*T1（Q9），clamp
t2_arr = [min(t2_i, 2.0 * t1_i) for t1_i, t2_i in zip(t1_arr, t2_arr)]
viol = sum(1 for t1_i, t2_i in zip(t1_arr, t2_arr) if t2_i > 2 * t1_i)
print("T2>2*T1 violations after clamp:", viol)
freq_arr = [get(f01, q, 4.85) for q in active]
ro_arr = [get(ro, q, 3.49) for q in active]
ge_arr = [get(ge, q, 0.11) for q in active]

two_q = {}
for (p, q) in edges:
    glabel = [g for g in cmap if (cmap[g][0], cmap[g][1]) in ((active[p], active[q]), (active[q], active[p]))]
    gname = glabel[0] if glabel else None
    e = czerr.get(gname, 1.06)
    two_q[(p, q)] = e / 100.0
    two_q[(q, p)] = e / 100.0

out = {
    "name": "tianyan176_66q_effective",
    "description": "Tianyan-176 effective topology (66 qubits, 110 couplers, "
                   "6 disabled qubits + 29 disabled couplers -> 60 active qubits, "
                   "81 edges). Calibration: " + c["calibrationTime"],
    "num_qubits": n,
    "coupling_map": [list(e) for e in edges],
    "device_params": {
        "t1_times": t1_arr,
        "t2_times": t2_arr,
        "freq_ghz": freq_arr,
        "readout_error": [x / 100.0 for x in ro_arr],
        "single_q_gate_error": [x / 100.0 for x in ge_arr],
        "two_q_gate_error": {str(k): v for k, v in two_q.items()},
        "shots": 1024,
    },
}

out_path = "traindata/topo/tianyan176_66q.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=1)
print("saved:", out_path)
print("active qubits:", n, "| edges:", len(edges))
print("two_q dict entries:", len(two_q))
print("sample edge:", edges[0], "->", two_q[(edges[0][0], edges[0][1])])

# 一致性校验
cc = []
for (p, q) in edges:
    cc.append(two_q[(p, q)])
cc = np.array(cc)
print("two_q error: min {:.5f} max {:.5f} mean {:.5f}".format(cc.min(), cc.max(), cc.mean()))
