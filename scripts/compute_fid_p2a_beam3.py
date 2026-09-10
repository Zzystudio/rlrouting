"""Batch compute fidelity for nam_l05_v2 routed circuits (argmax + beam3)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from qiskit.qasm2 import loads as qasm2_loads
from sim.trajectory_sim import trajectory_circuit_fidelity
from routing.rl.eval_policy import load_topo

config, hw, coupling_map = load_topo('traindata/topo/tianyan176_20q.json')
SWAP_DEF = 'gate swap a,b { cx a,b; cx b,a; cx a,b; }\n'

for model in ['nam_p2a_beam3']:
    print(f'=== {model} ===', flush=True)
    results = []
    for fname in sorted(os.listdir(f'benchmark/routed/{model}')):
        if not fname.endswith('.json'):
            continue
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
            print(f'  ERROR {fname}: {e}', flush=True)
        d['fidelity'] = fid
        with open(fpath, 'w') as f:
            json.dump(d, f, indent=1, ensure_ascii=False)
        fid_str = f'{fid:.4f}' if fid else '--'
        print(f'  {fname}: {d["num_logical_qubits"]}q, swaps={d["num_swaps"]}, fid={fid_str}', flush=True)
        results.append(d)
    summary = {'model': model, 'results': results}
    with open(f'benchmark/routed/{model}_summary.json', 'w') as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    fids = [r['fidelity'] for r in results if r['fidelity'] is not None]
    if fids:
        print(f'  mean_fid={sum(fids)/len(fids):.4f} ({len(fids)} ctks)', flush=True)
    print(f'=== {model} done ===', flush=True)
print('ALL DONE', flush=True)
