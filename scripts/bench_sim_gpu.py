"""模拟器 GPU vs CPU 基准（doc/模拟器加速.md §5-P4）。

用真实 NAM 路由电路（benchmark/routed/p0t287/，tianyan287_20q 物理电路）
按混合精度 T 口径（≤9q T=16 / 10-14q T=32 / ≥15q T=64）对比 CPU/GPU 墙钟，
另加 19q×T=128（深电路复核档）。

用法: python3 scripts/bench_sim_gpu.py [--out benchmark/routed/sim_gpu_bench.json]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qiskit.qasm2 import loads as qasm2_loads
from sim.trajectory_sim_v2 import trajectory_circuit_fidelity_events
from routing.rl.eval_policy import load_topo

SWAP_DEF = 'gate swap a,b { cx a,b; cx b,a; cx a,b; }\n'
ROUTED_DIR = 'benchmark/routed/p0t287'
TOPO = 'tianyan287_20q'

OUT_JSON = 'benchmark/routed/sim_gpu_bench.json'
if '--out' in sys.argv:
    OUT_JSON = sys.argv[sys.argv.index('--out') + 1]

# 19q T=128 复核档在混合精度口径之外，额外测
EXTRA_T = {'tof_10': 128}


def T_for(q):
    return 16 if q <= 9 else (32 if q <= 14 else 64)


config, hw, cm = load_topo(f'traindata/topo/{TOPO}.json')

# 选样：小/中/大/最深 各一
cands = []
for f in sorted(os.listdir(ROUTED_DIR)):
    if not f.endswith('.json'):
        continue
    d = json.load(open(os.path.join(ROUTED_DIR, f)))
    if not d.get('completed', True):
        continue
    cands.append((d.get('num_logical_qubits', 0), d.get('num_swaps', 0),
                  f.removesuffix('.json')))
cands.sort()
small = next(c for c in cands if c[0] >= 8)
mid = next(c for c in cands if 10 <= c[0] <= 14)
big = next(c for c in cands if c[0] >= 19)
deep = next(c for c in cands if c[0] >= 18 and c[1] >= 300)
names = {small[2], mid[2], big[2], deep[2]}

rows = []
if os.path.exists(OUT_JSON):
    try:
        rows = json.load(open(OUT_JSON))
        print(f'[resume] {OUT_JSON}: 已有 {len(rows)} 行', flush=True)
    except Exception:
        rows = []
_done = {(r['circuit'], r['backend'], r['T']) for r in rows}


def run_one(name, d, q, T, backend):
    phys = qasm2_loads(d['routed_qasm'].replace(
        'include "qelib1.inc";', 'include "qelib1.inc";\n' + SWAP_DEF))
    t0 = time.perf_counter()
    fid = trajectory_circuit_fidelity_events(phys, config,
                                             num_trajectories=T,
                                             seed=42, backend=backend)
    dt = round(time.perf_counter() - t0, 2)
    rows.append({'circuit': name, 'qubits': q, 'swaps': d['num_swaps'],
                 'T': T, 'backend': backend, 'fid': round(fid, 6),
                 'sec': dt})
    print(f'{name:<22s} q={q:>2d} T={T:>3d} {backend:>4s}: '
          f'fid={fid:.4f}  {dt:>8.2f}s', flush=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(rows, f, indent=1)


cases = []
for name in sorted(names):
    d = json.load(open(os.path.join(ROUTED_DIR, f'{name}.json')))
    q = d['num_logical_qubits']
    cases.append((name, d, q, EXTRA_T.get(name, T_for(q))))

# 两遍跑：先 GPU（快，快速拿全量数据），再 CPU（深电路极慢，是加速动机本身）
for backend in ('cuda', 'cpu'):
    for name, d, q, T in cases:
        if (name, backend, T) in _done:
            continue
        run_one(name, d, q, T, backend)

for name in sorted(names):
    c = {r['backend']: r for r in rows
         if r['circuit'] == name}
    if 'cpu' in c and 'cuda' in c:
        print(f'{name:<22s} -> speedup = {c["cpu"]["sec"] / c["cuda"]["sec"]:.1f}x  '
              f'(fid diff {abs(c["cpu"]["fid"] - c["cuda"]["fid"]):.4f})', flush=True)

print(f'saved: {OUT_JSON}')
