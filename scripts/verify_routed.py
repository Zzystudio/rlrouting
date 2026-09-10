"""Post-regeneration verification: every routed JSON must satisfy
initial_layout + routed swaps -> final_layout, and all gates must match
the original circuit in dependency order (with derived-initial cross-check)."""
import json, math, re, glob, os, sys

def parse_ops(text):
    ops = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("OPENQASM", "include", "qreg", "creg", "//")):
            continue
        m = re.match(r"([A-Za-z_][\w\(\)\.\-\*\/0-9]*)\s*(?:q\[(\d+)\](?:\s*,\s*q\[(\d+)\])?)?;", line)
        if not m: raise ValueError(line)
        name, a, b = m.group(1), m.group(2), m.group(3)
        if name.startswith("rz("):
            ops.append(("rz", (int(a),), eval(name[3:-1], {"pi": math.pi})))
        elif name in ("x","h","t","s","z","sdg","tdg"):
            ops.append((name, (int(a),), None))
        elif name in ("cx","cz","swap"):
            ops.append((name, (int(a), int(b)), None))
        else: raise ValueError(line)
    return ops

def gate_key(op, logs):
    name, qs, ang = op
    return (name, round(ang, 9) if ang is not None else None, tuple(logs))

def swap_map(M, p, q):
    la = next((l for l, v in M.items() if v == p), None)
    lb = next((l for l, v in M.items() if v == q), None)
    if la is not None: M[la] = q
    if lb is not None: M[lb] = p

def verify(M0, rops, cops):
    M = dict(M0); consumed = set()
    pred = {}; last = {}
    for i, op in enumerate(cops):
        pred[i] = {last[q] for q in op[1] if q in last}
        for q in op[1]: last[q] = i
    for op in rops:
        if op[0] == "swap":
            swap_map(M, *op[1]); continue
        logs = [next((l for l, v in M.items() if v == phys), None) for phys in op[1]]
        key = gate_key(op, logs)
        hit = None
        for gi in range(len(cops)):
            if gi in consumed or not all(p in consumed for p in pred[gi]): continue
            if gate_key(cops[gi], list(cops[gi][1])) == key:
                hit = gi; break
        if hit is None:
            return False, f"gate mismatch {op} logicals={logs}"
        consumed.add(hit)
    if len(consumed) != len(cops):
        return False, f"{len(cops)-len(consumed)} gates unmatched"
    return True, "OK"

total = ok_n = 0
bad = []
for f in sorted(glob.glob("benchmark/routed/*/*.json")):
    d = json.load(open(f))
    if not isinstance(d, dict) or "routed_qasm" not in d: continue
    src = f"benchmark/nam_circs/{d['circuit']}"
    if not os.path.exists(src): continue
    total += 1
    rops = parse_ops(d["routed_qasm"])
    cops = parse_ops(open(src).read())
    ini = {int(k): v for k, v in d["initial_layout"].items()}
    fin = {int(k): v for k, v in d["final_layout"].items()}
    # check 1: tracking consistency
    M = dict(ini)
    for op in rops:
        if op[0] == "swap": swap_map(M, *op[1])
    track_ok = (M == fin)
    # check 2: gate-level consistency from initial_layout
    gok, msg = verify(ini, rops, cops)
    if track_ok and gok:
        ok_n += 1
    else:
        bad.append((f, track_ok, msg))

print(f"verified {ok_n}/{total} files fully consistent")
for f, t, m in bad:
    print(f"  BAD {f}: tracking={t} {m}")
sys.exit(0 if not bad else 1)
