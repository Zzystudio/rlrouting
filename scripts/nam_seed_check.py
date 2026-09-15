"""NAM fid 差值的多种子稳健性检验：R3b_beam5 vs SABRE（trajectory_v2）。

对 19 条 NAM 电路 × {r3b_beam5, SABRE} × 3 种子计算 T=16 保真度，
输出每种子均值、配对差值与跨种子稳定性——验证 +2.8% 是否为 MC 噪声。

用法: python3 scripts/nam_seed_check.py [--seeds 0,1,2] [--traj 16]
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qiskit.qasm2 import loads as qasm2_loads

from sim.trajectory_sim_v2 import trajectory_circuit_fidelity_events
from routing.routing import sabre_route
from routing.rl.eval_policy import load_topo

config, hw, coupling_map = load_topo('traindata/topo/tianyan176_20q.json')
SWAP_DEF = 'gate swap a,b { cx a,b; cx b,a; cx a,b; }\n'
NAM_DIR = 'benchmark/nam_circs'
OUT_JSON = 'benchmark/routed/v2sim_nam_seedcheck.json'
SEEDS = [int(s) for s in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['0', '1', '2'])]
NTRAJ = int(sys.argv[2]) if len(sys.argv) > 2 else 16

circuits = sorted(f for f in os.listdir(NAM_DIR) if f.endswith('.qasm'))

# SABRE 路由与保真度种子无关的路由：只重路由一次
sabre_phys = {}
for fname in circuits:
    qc_raw = qasm2_loads(open(os.path.join(NAM_DIR, fname)).read())
    phys, info = sabre_route(qc_raw, config, seed=0)
    sabre_phys[fname] = (phys, info['num_swaps'])

results = []
for seed in SEEDS:
    per_seed = {'seed': seed, 'models': {}}
    for model_dir in ('r3b_beam5', 'SABRE'):
        fids = []
        for fname in circuits:
            name = fname.removesuffix('.qasm')
            if model_dir == 'SABRE':
                fid = trajectory_circuit_fidelity_events(
                    sabre_phys[fname][0], config, num_trajectories=NTRAJ, seed=seed)
            else:
                d = json.load(open(f'benchmark/routed/{model_dir}/{name}.json'))
                qc = qasm2_loads(d['routed_qasm'].replace(
                    'include "qelib1.inc";', 'include "qelib1.inc";\n' + SWAP_DEF))
                fid = trajectory_circuit_fidelity_events(
                    qc, config, num_trajectories=NTRAJ, seed=seed)
            fids.append(fid)
        per_seed['models'][model_dir] = fids
        print(f'seed={seed} {model_dir}: mean={statistics.mean(fids):.4f}', flush=True)
    d_r = per_seed['models']['r3b_beam5']
    d_s = per_seed['models']['SABRE']
    per_seed['delta_mean'] = statistics.mean(d_r) - statistics.mean(d_s)
    per_seed['delta_logmean'] = None
    import math
    lm = lambda v: math.exp(statistics.mean(math.log(max(x, 1e-9)) for x in v))
    per_seed['delta_logmean'] = lm(d_r) / lm(d_s) - 1
    per_seed['wins'] = sum(1 for a, b in zip(d_r, d_s) if a > b)
    results.append(per_seed)
    print(f'seed={seed}: Δmean={per_seed["delta_mean"]:+.4f} '
          f'Δlogmean={per_seed["delta_logmean"]:+.2%} wins={per_seed["wins"]}/19',
          flush=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=1)

deltas = [p['delta_mean'] for p in results]
print(f'\n=== 稳健性结论（{len(SEEDS)} 种子 × T={NTRAJ}）===')
print(f'Δmean 逐种子: {[f"{d:+.4f}" for d in deltas]}')
print(f'Δmean 均值: {statistics.mean(deltas):+.4f}  极差: {max(deltas)-min(deltas):.4f}')
print(f'符号一致: {"是" if all(d > 0 for d in deltas) else ("全部为负" if all(d < 0 for d in deltas) else "不一致！")}')

import math as _m
lmr = [_m.exp(statistics.mean(_m.log(max(x, 1e-9)) for x in p['models']['r3b_beam5'])) for p in results]
lms = [_m.exp(statistics.mean(_m.log(max(x, 1e-9)) for x in p['models']['SABRE'])) for p in results]
print(f'Δlogmean 逐种子: {[f"{a/b-1:+.2%}" for a, b in zip(lmr, lms)]}')
print('ALL DONE', flush=True)
