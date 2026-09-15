"""深电路 T=128 复核（doc/20260915训练方案.md 度量规范：近胜局需 T≥128）。

GPU 加速模拟器使 T=128 从"每条 ~28 min 不可行"变为"每条 <2.5 min"。
覆盖 NAM × tianyan287_20q 混合精度口径下 ≥12q 的电路：
  SABRE（现场 sabre_route seed=0） vs p0t287 argmax vs p0t287_beam5（完成者）。
beam5 TRUNC 电路沿用 fid=0 口径（路由级截断，与 T 无关，不重模拟）。

用法: python3 scripts/eval_nam_T128_deep.py
输出: benchmark/routed/hybridT128_deep_t287.json
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

OUT_JSON = 'benchmark/routed/hybridT128_deep_t287.json'
TOPO = 'tianyan287_20q'
NAM_DIR = 'benchmark/nam_circs'
SEED = 42
SWAP_DEF = 'gate swap a,b { cx a,b; cx b,a; cx a,b; }\n'
T_DEEP = 128
MIN_Q = 12

config, hw, coupling_map = load_topo(f'traindata/topo/{TOPO}.json')
circuits = sorted(f for f in os.listdir(NAM_DIR) if f.endswith('.qasm'))

# SABRE 现场路由（与 eval 链同口径：seed=0）
sabre_phys = {}
for fname in circuits:
    qc_raw = qasm2_loads(open(os.path.join(NAM_DIR, fname)).read())
    if qc_raw.num_qubits < MIN_Q:
        continue
    phys, info = sabre_route(qc_raw, config, seed=0)
    sabre_phys[fname] = (phys, info['num_swaps'])

results = []
if os.path.exists(OUT_JSON):
    try:
        results = json.load(open(OUT_JSON))
        print(f'[resume] 已有 {len(results)} 行', flush=True)
    except Exception:
        results = []
_done = {(r['method'], r['circuit']) for r in results}


def run_one(method, fname, phys, swaps, truncated):
    key = (method, fname)
    if key in _done:
        return
    q = sabre_phys[fname][0].num_qubits if False else \
        qasm2_loads(open(os.path.join(NAM_DIR, fname)).read()).num_qubits
    if truncated:
        results.append({'method': method, 'circuit': fname, 'qubits': q,
                        'swaps': swaps, 'T': 0, 'fid': 0.0, 'sec': 0.0,
                        'truncated': True})
        print(f'{fname} [{method}] TRUNC 跳过模拟（fid 记 0）', flush=True)
    else:
        t0 = time.perf_counter()
        fid = trajectory_circuit_fidelity_events(phys, config,
                                                 num_trajectories=T_DEEP,
                                                 seed=SEED, backend='auto')
        dt = round(time.perf_counter() - t0, 1)
        results.append({'method': method, 'circuit': fname, 'qubits': q,
                        'swaps': swaps, 'T': T_DEEP, 'fid': fid, 'sec': dt})
        print(f'{fname} [{method}] q={q} T={T_DEEP} sw={swaps} '
              f'fid={fid:.4f} ({dt:.0f}s)', flush=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=1)


for fname in circuits:
    name = fname.removesuffix('.qasm')
    q = qasm2_loads(open(os.path.join(NAM_DIR, fname)).read()).num_qubits
    if q < MIN_Q:
        continue
    # SABRE
    run_one('SABRE', fname, sabre_phys[fname][0], sabre_phys[fname][1], False)
    # p0t287 argmax（已路由）
    d = json.load(open(f'benchmark/routed/p0t287/{name}.json'))
    run_one('p0t287', fname,
            qasm2_loads(d['routed_qasm'].replace(
                'include "qelib1.inc";', 'include "qelib1.inc";\n' + SWAP_DEF)),
            d['num_swaps'], not d.get('completed', True))
    # p0t287_beam5（已路由；TRUNC 记 0）
    db = json.load(open(f'benchmark/routed/p0t287_beam5/{name}.json'))
    run_one('p0t287_beam5', fname,
            qasm2_loads(db['routed_qasm'].replace(
                'include "qelib1.inc";', 'include "qelib1.inc";\n' + SWAP_DEF)),
            db['num_swaps'], not db.get('completed', True))

# 汇总
print('\n=== T=128 深电路复核汇总 ===', flush=True)
by = {}
for r in results:
    by.setdefault(r['circuit'], {})[r['method']] = r
for c in sorted(by):
    row = by[c]
    parts = [f"{m}={row[m]['fid']:.4f}" for m in ('SABRE', 'p0t287', 'p0t287_beam5')
             if m in row]
    print(f'{c:<24s} {"  ".join(parts)}', flush=True)
print('saved:', OUT_JSON, flush=True)
