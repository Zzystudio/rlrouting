"""Batch compute trajectory_sched fidelity for a routed model dir.

用法: python3 scripts/compute_fidelity_model.py <model_name>
对 benchmark/routed/<model_name>/ 下每个 * .json 计算 trajectory_sched ×16 保真度，
回写 fidelity 字段并生成 benchmark/routed/<model_name>_summary.json。
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from qiskit.qasm2 import loads as qasm2_loads
from sim.trajectory_sim import trajectory_circuit_fidelity
from routing.rl.eval_policy import load_topo

config, hw, coupling_map = load_topo('traindata/topo/tianyan176_20q.json')
SWAP_DEF = 'gate swap a,b { cx a,b; cx b,a; cx a,b; }\n'

model = sys.argv[1] if len(sys.argv) > 1 else 'l05_nam'
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

with open(f'benchmark/routed/{model}_summary.json', 'w') as f:
    json.dump({'model': model, 'results': results}, f, indent=1, ensure_ascii=False)

fids = [r['fidelity'] for r in results if r['fidelity'] is not None]
if fids:
    import statistics, math
    logmean = math.exp(statistics.mean(math.log(max(f, 1e-9)) for f in fids))
    print(f'  mean_fid={statistics.mean(fids):.4f}  logmean={logmean:.4f} ({len(fids)} ctks)', flush=True)
print(f'=== {model} done ===', flush=True)