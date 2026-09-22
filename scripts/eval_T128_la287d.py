"""LA287d argmax NAM 全 19 条 T=128 复核（近胜局判定；SABRE 值复用已有 T=128 数据）。"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qiskit.qasm2 import loads as qasm2_loads
from sim.trajectory_sim_v2 import trajectory_circuit_fidelity_events
from routing.routing import sabre_route
from routing.rl.eval_policy import load_topo

OUT = 'benchmark/routed/audit_T128_la287d_nam.json'
SWAP_DEF = 'gate swap a,b { cx a,b; cx b,a; cx a,b; }\n'
T_DEEP = 128

config, hw, coupling_map = load_topo('traindata/topo/tianyan287_20q.json')
circuits = sorted(f for f in os.listdir('benchmark/nam_circs') if f.endswith('.qasm'))

sabre_phys = {}
for f in circuits:
    qc = qasm2_loads(open(f'benchmark/nam_circs/{f}').read())
    phys, info = sabre_route(qc, config, swap_trials=20, seed=0)
    sabre_phys[f] = (phys, info['num_swaps'])

results = []
if os.path.exists(OUT):
    results = json.load(open(OUT))
_done = {(r['method'], r['circuit']) for r in results}


def run_one(method, fname, phys, swaps, truncated):
    key = (method, fname)
    if key in _done:
        return
    if truncated:
        results.append({'method': method, 'circuit': fname, 'swaps': swaps,
                        'T': 0, 'fid': 0.0, 'truncated': True})
        print(f'{fname} [{method}] TRUNC 跳过', flush=True)
    else:
        t0 = time.perf_counter()
        fid = trajectory_circuit_fidelity_events(phys, config,
                                                 num_trajectories=T_DEEP,
                                                 seed=42, backend='auto')
        dt = round(time.perf_counter() - t0, 1)
        results.append({'method': method, 'circuit': fname, 'swaps': swaps,
                        'T': T_DEEP, 'fid': fid, 'sec': dt})
        print(f'{fname} [{method}] sw={swaps} fid={fid:.4f} ({dt:.0f}s)', flush=True)
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=1)


for fname in circuits:
    name = fname.removesuffix('.qasm')
    run_one('SABRE', fname, sabre_phys[fname][0], sabre_phys[fname][1], False)
    d = json.load(open(f'benchmark/routed/la287d_argmax/{name}.json'))
    phys = qasm2_loads(d['routed_qasm'].replace(
        'include "qelib1.inc";', 'include "qelib1.inc";\n' + SWAP_DEF))
    run_one('LA287d', fname, phys, d['num_swaps'], not d.get('completed', True))

by = {}
for r in results:
    by.setdefault(r['circuit'], {})[r['method']] = r['fid']
print('\n=== T=128 NAM 全 19 条：SABRE vs LA287d ===')
w = l = t = 0
for c in sorted(by):
    s, m = by[c]['SABRE'], by[c]['LA287d']
    mark = 'LA287d' if m > s + 1e-9 else ('SABRE' if s > m + 1e-9 else '≈')
    if m > s + 1e-9: w += 1
    elif s > m + 1e-9: l += 1
    else: t += 1
    print(f'{c:<24s} SABRE={s:.4f}  LA287d={m:.4f}  -> {mark}')
sm = sum(by[c]['SABRE'] for c in by)/len(by)
mm = sum(by[c]['LA287d'] for c in by)/len(by)
print(f'\nfid_mean: SABRE={sm:.4f}  LA287d={mm:.4f}  ({(mm-sm)/sm*100:+.1f}%)')
print(f'战绩: LA287d {w} 胜 / {l} 负 / {t} 平')
