"""审计保真度轴：in-dist 50 条 × 3 解（SABRE/argmax/beam5la+bud）T=128 重模拟。"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qiskit.qasm2 import loads as qasm2_loads
from sim.trajectory_sim_v2 import trajectory_circuit_fidelity_events
from routing.routing import sabre_route
from routing.rl.eval_policy import load_topo

OUT = 'benchmark/routed/audit_T128_indist.json'
SWAP_DEF = 'gate swap a,b { cx a,b; cx b,a; cx a,b; }\n'
T_DEEP = 128
VAL_DIR = 'benchmark/indist_val'

config, hw, coupling_map = load_topo('traindata/topo/tianyan287_20q.json')
circuits = sorted(f for f in os.listdir(VAL_DIR) if f.endswith('.qasm'))

sabre_phys = {}
for f in circuits:
    qc = qasm2_loads(open(os.path.join(VAL_DIR, f)).read())
    phys, info = sabre_route(qc, config, swap_trials=20, seed=0)
    sabre_phys[f] = (phys, info['num_swaps'])

results = []
if os.path.exists(OUT):
    results = json.load(open(OUT))
    print(f'[resume] {len(results)} rows')
_done = {(r['method'], r['circuit']) for r in results}


def run_one(method, fname, phys, swaps, truncated):
    key = (method, fname)
    if key in _done:
        return
    q = qasm2_loads(open(os.path.join(VAL_DIR, fname)).read()).num_qubits
    if truncated:
        results.append({'method': method, 'circuit': fname, 'qubits': q,
                        'swaps': swaps, 'T': 0, 'fid': 0.0, 'truncated': True})
        print(f'{fname} [{method}] TRUNC 跳过', flush=True)
    else:
        t0 = time.perf_counter()
        fid = trajectory_circuit_fidelity_events(phys, config,
                                                 num_trajectories=T_DEEP,
                                                 seed=42, backend='auto')
        dt = round(time.perf_counter() - t0, 1)
        results.append({'method': method, 'circuit': fname, 'qubits': q,
                        'swaps': swaps, 'T': T_DEEP, 'fid': fid, 'sec': dt})
        print(f'{fname} [{method}] q={q} T=128 sw={swaps} fid={fid:.4f} ({dt:.0f}s)',
              flush=True)
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=1)


for fname in circuits:
    name = fname.removesuffix('.qasm')
    # SABRE
    run_one('SABRE', fname, sabre_phys[fname][0], sabre_phys[fname][1], False)
    # argmax
    d = json.load(open(f'benchmark/routed/la287_argmax_indist/{name}.json'))
    phys = qasm2_loads(d['routed_qasm'].replace(
        'include "qelib1.inc";', 'include "qelib1.inc";\n' + SWAP_DEF))
    run_one('argmax', fname, phys, d['num_swaps'], not d.get('completed', True))
    # beam5la+bud
    db = json.load(open(f'benchmark/routed/la287_beam5la_bud_indist/{name}.json'))
    physb = qasm2_loads(db['routed_qasm'].replace(
        'include "qelib1.inc";', 'include "qelib1.inc";\n' + SWAP_DEF))
    run_one('beam5la+bud', fname, physb, db['num_swaps'],
            not db.get('completed', True))
print('saved:', OUT)
