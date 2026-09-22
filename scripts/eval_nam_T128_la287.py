"""LA287 argmax 深电路 T=128 复核（SABRE 的 T=128 值复用 hybridT128_deep_t287.json）。"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qiskit.qasm2 import loads as qasm2_loads
from sim.trajectory_sim_v2 import trajectory_circuit_fidelity_events
from routing.rl.eval_policy import load_topo

OUT_JSON = 'benchmark/routed/hybridT128_la287_t287.json'
SWAP_DEF = 'gate swap a,b { cx a,b; cx b,a; cx a,b; }\n'
T_DEEP = 128
MIN_Q = 12

config, hw, coupling_map = load_topo('traindata/topo/tianyan287_20q.json')
circuits = sorted(f for f in os.listdir('benchmark/nam_circs') if f.endswith('.qasm'))

results = []
if os.path.exists(OUT_JSON):
    results = json.load(open(OUT_JSON))
    print(f'[resume] {len(results)} rows')
_done = {r['circuit'] for r in results}

for fname in circuits:
    name = fname.removesuffix('.qasm')
    if fname in _done:
        continue
    d = json.load(open(f'benchmark/routed/la287_argmax/{name}.json'))
    if not d.get('completed', True):
        results.append({'circuit': fname, 'qubits': d.get('num_logical_qubits', 0),
                        'swaps': d['num_swaps'], 'T': 0, 'fid': 0.0, 'truncated': True})
        print(f'{fname} TRUNC 跳过', flush=True)
        continue
    qc = d.get('num_logical_qubits', 0)
    if qc < MIN_Q:
        continue
    phys = qasm2_loads(d['routed_qasm'].replace(
        'include "qelib1.inc";', 'include "qelib1.inc";\n' + SWAP_DEF))
    t0 = time.perf_counter()
    fid = trajectory_circuit_fidelity_events(phys, config, num_trajectories=T_DEEP,
                                             seed=42, backend='auto')
    dt = round(time.perf_counter() - t0, 1)
    results.append({'circuit': fname, 'qubits': qc, 'swaps': d['num_swaps'],
                    'T': T_DEEP, 'fid': fid, 'sec': dt})
    print(f'{fname} q={qc} T=128 sw={d["num_swaps"]} fid={fid:.4f} ({dt:.0f}s)',
          flush=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=1)
print('saved:', OUT_JSON)
