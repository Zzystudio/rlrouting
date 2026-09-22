"""拓扑泛化测试：in-dist 50 条路由解 × tianyan287_20q_stronghetero（训练未见拓扑）。

所有解的 routed_qasm 已存在（t287 上路由）——本脚本仅换噪声/耦合配置模拟。
注意：stronghetero 与 t287 耦合图不同（29 vs 31 边，20 条共同边）——
使用 t287 路由解在 stronghetero 上模拟需验证布局/交换序列的边有效性。
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qiskit.qasm2 import loads as qasm2_loads
from sim.trajectory_sim_v2 import trajectory_circuit_fidelity_events
from routing.routing import sabre_route
from routing.rl.eval_policy import load_topo

OUT = 'benchmark/routed/hybridT_indist_stronghetero.json'
SWAP_DEF = 'gate swap a,b { cx a,b; cx b,a; cx a,b; }\n'
T_MIX = None  # 由 T_for 决定


def T_for(q):
    return 16 if q <= 9 else (32 if q <= 14 else 64)


SH_CONFIG, SH_HW, SH_CM = load_topo('traindata/topo/tianyan287_20q_stronghetero.json')
circuits = sorted(f for f in os.listdir('benchmark/indist_val') if f.endswith('.qasm'))

results = []
if os.path.exists(OUT):
    results = json.load(open(OUT))
    print(f'[resume] {len(results)} rows')
_done = {(r['method'], r['circuit']) for r in results}


def run_one(method, fname, phys, swaps, truncated):
    key = (method, fname)
    if key in _done:
        return
    q = qasm2_loads(open(os.path.join('benchmark/indist_val', fname)).read()).num_qubits
    T = T_for(q)
    if truncated:
        results.append({'method': method, 'circuit': fname, 'qubits': q,
                        'swaps': swaps, 'T': 0, 'fid': 0.0, 'truncated': True})
        print(f'{fname} [{method}] TRUNC 跳过', flush=True)
    else:
        t0 = time.perf_counter()
        fid = trajectory_circuit_fidelity_events(phys, SH_CONFIG,
                                                 num_trajectories=T,
                                                 seed=42, backend='auto')
        dt = round(time.perf_counter() - t0, 1)
        results.append({'method': method, 'circuit': fname, 'qubits': q,
                        'swaps': swaps, 'T': T, 'fid': fid, 'sec': dt})
        print(f'{fname} [{method}] q={q} T={T} sw={swaps} fid={fid:.4f} ({dt:.0f}s)',
              flush=True)
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=1)


for fname in circuits:
    name = fname.removesuffix('.qasm')
    qc = qasm2_loads(open(os.path.join('benchmark/indist_val', fname)).read())

    # SABRE 在 stronghetero 上现场路由
    phys_s, info_s = sabre_route(qc, SH_CONFIG, swap_trials=20, seed=0)
    run_one('SABRE', fname, phys_s, info_s['num_swaps'], False)

    # 各模型已路由解（t287 上路由的 QASM 物理电路）
    for method, rdir in [('LA287', 'la287_argmax_indist'),
                         ('LA287c', 'la287c_argmax_indist'),
                         ('LA287d', 'la287d_argmax_indist'),
                         ('LA287ctl', 'la287ctl_argmax_indist'),
                         ('LA287d_b5la+bud', 'la287d_beam5la_bud_indist')]:
        jf = f'benchmark/routed/{rdir}/{name}.json'
        if not os.path.exists(jf):
            continue
        d = json.load(open(jf))
        run_one(method, fname,
                qasm2_loads(d['routed_qasm'].replace(
                    'include "qelib1.inc";',
                    'include "qelib1.inc";\n' + SWAP_DEF)),
                d['num_swaps'], not d.get('completed', True))

by = {}
for r in results:
    by.setdefault(r['circuit'], {})[r['method']] = r
print('\n=== stronghetero 拓扑泛化：in-dist 50 逐电路 ===', flush=True)
print(f"{'电路':<30s} {'q':>2s} " + ' '.join(
    f'{m:>9s}' for m in ['SABRE', 'LA287', 'LA287c', 'LA287d', 'LA287ctl']))
for c in sorted(by):
    row = by[c]
    cells = []
    for m in ['SABRE', 'LA287', 'LA287c', 'LA287d', 'LA287ctl']:
        v = row.get(m, {}).get('fid')
        cells.append(f'{v:>9.4f}' if v is not None else '      ---')
    print(f'{c:<30s} {row["SABRE"]["qubits"]:>2d} ' + ' '.join(cells), flush=True)

print('\n=== 分层汇总 ===', flush=True)
mf = {m['name'].removesuffix('.qasm'): m for m in json.load(open('benchmark/indist_val/manifest.json'))}
WIDE = ('clifford_layer', 'layered', 'parallel_blocks', 'pauli_evolution',
        'mixed_serial_parallel', 'qft_butterfly', 'qpe_block', 'mcx_ladder')
cells = {}
for c in by:
    f = mf[c]['family']
    is_wide = any(f.startswith(w) for w in WIDE)
    q = by[c]['SABRE']['qubits']
    band = '5-9' if q <= 9 else ('10-14' if q <= 14 else '15-20')
    cells.setdefault((('宽并行' if is_wide else '算术') + '/' + band), []).append(c)
for k in sorted(cells):
    cs = cells[k]
    n = len(cs)
    line = f'{k:<14s} n={n:>3d} |'
    for m in ['SABRE', 'LA287', 'LA287c', 'LA287d', 'LA287ctl']:
        vals = [by[c][m]['fid'] for c in cs if m in by[c]]
        line += f' {sum(vals)/len(vals):>7.4f}' if vals else '      ---'
    print(line, flush=True)
print('saved:', OUT, flush=True)
