"""Batch compute fidelity for routed circuits."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from qiskit.qasm2 import loads as qasm2_loads
from sim.trajectory_sim import trajectory_circuit_fidelity
from routing.rl.eval_policy import load_topo

config, hw, coupling_map = load_topo('traindata/topo/tianyan176_20q.json')
SWAP_DEF = 'gate swap a,b { cx a,b; cx b,a; cx a,b; }\n'

for model in ['ph2v4', 'l05']:
    print(f'=== {model} ===', flush=True)
    summary = json.load(open(f'benchmark/routed/{model}_summary.json'))
    for r in summary['results']:
        fname = r['circuit'].replace('.qasm', '.json')
        fpath = f'benchmark/routed/{model}/{fname}'
        d = json.load(open(fpath))
        qasm_str = d['routed_qasm']
        qasm_with_swap = qasm_str.replace(
            'include "qelib1.inc";',
            'include "qelib1.inc";\n' + SWAP_DEF)
        qc = qasm2_loads(qasm_with_swap)
        try:
            fid = trajectory_circuit_fidelity(qc, config, num_trajectories=16, seed=0, scheduled=True)
        except Exception as e:
            fid = None
            print(f'  ERROR {r["circuit"]}: {e}', flush=True)
        r['fidelity'] = fid
        fid_str = f'{fid:.4f}' if fid else '--'
        print(f'  {r["circuit"]}: {r["num_logical_qubits"]}q, swaps={r["num_swaps"]}, fid={fid_str}', flush=True)
        d['fidelity'] = fid
        with open(fpath, 'w') as f:
            json.dump(d, f, indent=1, ensure_ascii=False)
    with open(f'benchmark/routed/{model}_summary.json', 'w') as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(f'=== {model} done ===', flush=True)
print('ALL DONE', flush=True)
