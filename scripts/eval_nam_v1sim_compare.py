"""NAM benchmark：第一版模拟器（trajectory_sim scheduled 路径）对比
l05 / l05_beam5 / R3 / R3b / R3b_beam5 / SABRE。

协议与原始 NAM benchmark 一致：trajectory_sched ×16, seed=0
（trajectory_circuit_fidelity(..., scheduled=True)，greedy 波次编排）。
SABRE 现场重路由（sabre_route seed=0，与历史基准同路由）。
增量缓存 JSON，支持断点续跑。

用法: python3 scripts/eval_nam_v1sim_compare.py [num_trajectories=16]
"""
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qiskit.qasm2 import loads as qasm2_loads

from sim.trajectory_sim import trajectory_circuit_fidelity
from routing.routing import sabre_route
from routing.rl.eval_policy import load_topo

config, hw, coupling_map = load_topo('traindata/topo/tianyan176_20q.json')
SWAP_DEF = 'gate swap a,b { cx a,b; cx b,a; cx a,b; }\n'
NAM_DIR = 'benchmark/nam_circs'
OUT_JSON = 'benchmark/routed/v1sim_nam_compare.json'
NTRAJ = int(sys.argv[1]) if len(sys.argv) > 1 else 16
SEED = 0
MODELS = ['l05', 'l05_beam5', 'r3', 'r3b', 'r3b_beam5']
if len(sys.argv) > 2:
    MODELS = [m for m in sys.argv[2].split(',') if m]

circuits = sorted(f for f in os.listdir(NAM_DIR) if f.endswith('.qasm'))

# SABRE 路由（与历史基准同参，一次性）
sabre_phys = {}
for fname in circuits:
    qc_raw = qasm2_loads(open(os.path.join(NAM_DIR, fname)).read())
    phys, info = sabre_route(qc_raw, config, seed=SEED)
    sabre_phys[fname] = (phys, info['num_swaps'])

cache = {}
if os.path.exists(OUT_JSON):
    for r in json.load(open(OUT_JSON)):
        cache[(r['model'], r['circuit'])] = r

results = []
for model in MODELS + ['SABRE']:
    for fname in circuits:
        key = (model, fname)
        if key in cache:
            results.append(cache[key])
            continue
        if model == 'SABRE':
            qc = sabre_phys[fname][0]
            swaps = sabre_phys[fname][1]
        else:
            d = json.load(open(f'benchmark/routed/{model}/{fname.removesuffix(".qasm")}.json'))
            qc = qasm2_loads(d['routed_qasm'].replace(
                'include "qelib1.inc";', 'include "qelib1.inc";\n' + SWAP_DEF))
            swaps = d['num_swaps']
        t0 = time.perf_counter()
        fid = trajectory_circuit_fidelity(qc, config, num_trajectories=NTRAJ,
                                          seed=SEED, scheduled=True)
        dt = round(time.perf_counter() - t0, 1)
        row = {'model': model, 'circuit': fname,
               'qubits': qc.num_qubits, 'swaps': swaps,
               'fid_v1': fid, 'sec': dt}
        results.append(row)
        cache[key] = row
        print(f'{fname} [{model}] sw={swaps} fid_v1={fid:.4f} ({dt:.0f}s)',
              flush=True)
        with open(OUT_JSON, 'w') as f:
            json.dump(results, f, indent=1)

# ---- 汇总表 ----
models = MODELS + ['SABRE']
by = {}
for r in results:
    by.setdefault(r['model'], {})[r['circuit']] = r

import math as _m


def logmean(v):
    return _m.exp(statistics.mean(_m.log(max(x, 1e-9)) for x in v))


print(f'\n=== 汇总（v1 trajectory_sched x{NTRAJ}, seed={SEED}）===', flush=True)
hdr = f'{"model":<12} {"Fid mean":>9} {"Fid logmean":>11} {"SWAPs":>7}'
print(hdr)
for m in models:
    fs = [by[m][c]['fid_v1'] for c in circuits]
    sws = [by[m][c]['swaps'] for c in circuits]
    print(f'{m:<12} {statistics.mean(fs):>9.4f} {logmean(fs):>11.4f} '
          f'{statistics.mean(sws):>7.1f}')

# 逐电路规模段（以 SABRE 电路 qubits 为准）
buckets = {}
for c in circuits:
    q = by['SABRE'][c]['qubits']
    b = '5-9q' if q <= 9 else ('10-14q' if q <= 14 else '15-19q')
    buckets.setdefault(b, []).append(c)
print(f'\n{"规模段":<8} ' + ' '.join(f'{m:>9}' for m in models) + '   (fid mean)')
for b in ('5-9q', '10-14q', '15-19q'):
    cs = buckets.get(b, [])
    if not cs:
        continue
    vals = {m: statistics.mean(by[m][c]['fid_v1'] for c in cs) for m in models}
    print(f'{b:<8} ' + ' '.join(f'{vals[m]:>9.4f}' for m in models))

# 逐电路胜者统计（vs SABRE）
print('\n逐电路胜 SABRE 计数（fid_v1）:')
for m in MODELS:
    wins = sum(1 for c in circuits if by[m][c]['fid_v1'] > by['SABRE'][c]['fid_v1'])
    print(f'  {m}: {wins}/{len(circuits)}')
print('ALL DONE', flush=True)
