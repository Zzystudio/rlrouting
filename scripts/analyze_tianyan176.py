import json
import numpy as np

c = json.load(open("data/tianyan176/config.json"))

def parse_csv(v):
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    return list(v)

dq = parse_csv(c["disabledQubits"])
dc = parse_csv(c["disabledCouplers"])

print("=" * 70)
print("1. 禁用集合（解析后）")
print("=" * 70)
print(f"disabledQubits ({len(dq)}): {dq}")
print(f"disabledCouplers ({len(dc)}): {dc}")

ov = c["overview"]
cmap = ov["coupler_map"]
valid_couplers = [g for g in cmap if g not in dc]
active_qubits = sorted(set(q for g in cmap for q in cmap[g]) - set(dq))

print()
print("=" * 70)
print("2. 有效拓扑")
print("=" * 70)
print(f"总 qubit: {len(ov['qubits'])}, 有效（排除 disabled）: {len(active_qubits)}")
print(f"总 coupler: {len(cmap)}, 有效（排除 disabled）: {len(valid_couplers)}")
deg = {q: 0 for q in active_qubits}
for g in valid_couplers:
    for q in cmap[g]:
        if q in deg:
            deg[q] += 1
d = np.array(sorted(deg.values()))
print(f"有效子图度数: min {d.min()}, max {d.max()}, mean {d.mean():.2f}, 中位 {np.median(d):.0f}")
deg1 = [q for q, v in deg.items() if v == 1]
print(f"有效子图度数=1 的 qubit: {deg1}")
isolated = [q for q in active_qubits if deg[q] == 0]
print(f"有效子图孤立 qubit: {isolated}")

print()
print("=" * 70)
print("3. 校准统计（有效 qubit 维度对齐）")
print("=" * 70)
qu = c["qubit"]
f01_used = qu["frequency"]["f01"]["qubit_used"]
t1_used = qu["relatime"]["T1"]["qubit_used"]
t2_used = qu["relatime"]["T2"]["qubit_used"]
ge_used = qu["singleQubit"]["gate error"]["qubit_used"]

def stats_by_qubit(name, used, vals, unit):
    p = np.array(vals)
    print(f"{name:16s} n={len(p):3d}  min {p.min():8.4f}  max {p.max():8.4f}  "
          f"mean {p.mean():8.4f}  median {np.median(p):8.4f}  std {p.std():8.4f}  [{unit}]")
    return dict(zip(used, vals))

f01 = stats_by_qubit("f01", f01_used, qu["frequency"]["f01"]["param_list"], "GHz")
t1 = stats_by_qubit("T1", t1_used, qu["relatime"]["T1"]["param_list"], "us")
t2 = stats_by_qubit("T2", t2_used, qu["relatime"]["T2"]["param_list"], "us")
ge = stats_by_qubit("1q err", ge_used, qu["singleQubit"]["gate error"]["param_list"], "%")

print()
print("=" * 70)
print("4. readout（按 11 根读出线）")
print("=" * 70)
ra = c["readout"]["readoutArray"]
for k, v in ra.items():
    if isinstance(v, dict) and "param_list" in v:
        p = np.array(v["param_list"])
        print(f"{k:28s} n={len(p):3d}  min {p.min():.5f}  max {p.max():.5f}  "
              f"mean {p.mean():.5f}  ({v.get('unit','')})")

print()
print("=" * 70)
print("5. CZ 门（每边）")
print("=" * 70)
cz = c["twoQubitGate"]["czGate"]
for k in ("gate error", "coupling strength", "length"):
    v = cz[k]
    p = np.array(v["param_list"])
    print(f"{k:20s} n={len(p):3d}  min {p.min():12.4g}  max {p.max():12.4g}  "
          f"mean {p.mean():12.4g}  ({v.get('unit','')})")
print("cz 校准覆盖边数:", len(cz["gate error"]["param_list"]), "/", len(cmap))
ge_by_edge = dict(zip(cz["gate error"]["qubit_used"], cz["gate error"]["param_list"]))
print("cz 校准边与有效边交集:", len(set(ge_by_edge) & set(valid_couplers)))

print()
print("=" * 70)
print("6. FSIM 参数（每边）")
print("=" * 70)
fsim = c["twoQubitGate"]["fsim_value"]
thetas, phis = [], []
for g, node in fsim.items():
    if isinstance(node, dict) and node.get("theta"):
        thetas.append(node["theta"])
        phis.append(node["phi"])
print(f"FSIM 有校准的边: {len(thetas)}")
print(f"theta: min {min(thetas):.4f} max {max(thetas):.4f} mean {np.mean(thetas):.4f}")
print(f"phi  : min {min(phis):.4f} max {max(phis):.4f} mean {np.mean(phis):.4f}")
zth = sum(1 for t in thetas if abs(t - np.pi / 2) < 0.05)
print(f"theta≈π/2 (CZ 近似) 的边: {zth}/{len(thetas)}")
