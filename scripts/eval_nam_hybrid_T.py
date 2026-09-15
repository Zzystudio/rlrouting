"""NAM benchmark：混合精度 T 重评（深电路高精度、浅电路低精度）。

精度按逻辑比特数分配（MC 方差 ∝ 深度）：
  ≤9q  → T=16     （浅电路，T16 方差可接受）
  10-14q → T=32   （中电路）
  ≥15q → T=64     （深电路，T16 方差 200× 不可信）
seed=42（独立于此前 T=16 seed=0 的测量，无偏复检）。

用法: python3 scripts/eval_nam_hybrid_T.py <model_dir> <out_json> [topo=tianyan176_20q]
SABRE 用 --sabre 标志（现场 sabre_route seed=0）。
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qiskit.qasm2 import loads as qasm2_loads

from sim.trajectory_sim_v2 import trajectory_circuit_fidelity_events
from routing.routing import sabre_route
from routing.rl.eval_policy import load_topo

MODEL_DIR = sys.argv[1]
OUT_JSON = sys.argv[2]
TOPO = sys.argv[3] if len(sys.argv) > 3 else 'tianyan176_20q'
IS_SABRE = len(sys.argv) > 4 and sys.argv[4] == '--sabre'
# GPU 加速：默认 backend=auto（CUDA 可用即 GPU，噪声 MC 统计等价）；
# --cpu 强制历史 CPU 口径（A/B 基准用）
IS_CPU = '--cpu' in sys.argv
SIM_BACKEND = 'cpu' if IS_CPU else 'auto'
# --circ-dir <dir>：评估集目录（默认 benchmark/nam_circs；QUARL 第二评估集用
# benchmark/quarl_opt_wo_rm），路由结果仍从 benchmark/routed/<MODEL_DIR>/ 读取
CIRC_DIR = 'benchmark/nam_circs'
if '--circ-dir' in sys.argv:
    CIRC_DIR = sys.argv[sys.argv.index('--circ-dir') + 1]
SEED = 42
SWAP_DEF = 'gate swap a,b { cx a,b; cx b,a; cx a,b; }\n'


def T_for(q):
    return 16 if q <= 9 else (32 if q <= 14 else 64)


config, hw, coupling_map = load_topo(f'traindata/topo/{TOPO}.json')
NAM_DIR = CIRC_DIR
circuits = sorted(f for f in os.listdir(NAM_DIR) if f.endswith('.qasm'))

# SABRE 路由一次性生成（确定性）
sabre_phys = {}
if IS_SABRE:
    for fname in circuits:
        qc_raw = qasm2_loads(open(os.path.join(NAM_DIR, fname)).read())
        phys, info = sabre_route(qc_raw, config, seed=0)
        sabre_phys[fname] = (phys, info['num_swaps'])

results = []
# 断点续传：已有 OUT_JSON 的行直接复用（增量计算）
if os.path.exists(OUT_JSON):
    try:
        results = json.load(open(OUT_JSON))
        print(f'[resume] {OUT_JSON}: 已有 {len(results)} 行，跳过已完成电路',
              flush=True)
    except Exception as _e:
        print(f'[resume] 读取失败（忽略）: {_e}', flush=True)
        results = []
_done = {r['circuit'] for r in results}

for fname in circuits:
    name = fname.removesuffix('.qasm')
    if fname in _done:
        continue
    if IS_SABRE:
        phys = sabre_phys[fname][0]
        swaps = sabre_phys[fname][1]
        truncated = False
    else:
        d = json.load(open(f'benchmark/routed/{MODEL_DIR}/{name}.json'))
        truncated = not d.get('completed', True)
        if truncated:
            # TRUNC 电路跳过模拟：路由失败的截断电路保真度必然≈0，
            # 记 fid=0 计入均值（诚实口径），节省 ~3000 门电路的模拟时间
            results.append({'model': MODEL_DIR, 'circuit': fname,
                            'qubits': d.get('num_logical_qubits', 0),
                            'swaps': d['num_swaps'], 'T': 0, 'fid': 0.0,
                            'sec': 0.0, 'truncated': True})
            print(f'{fname} [{MODEL_DIR}] TRUNC 跳过模拟（fid 记 0）', flush=True)
            with open(OUT_JSON, 'w') as f:
                json.dump(results, f, indent=1)
            continue
        qc = qasm2_loads(d['routed_qasm'].replace(
            'include "qelib1.inc";', 'include "qelib1.inc";\n' + SWAP_DEF))
        phys = qc
        swaps = d['num_swaps']
    q = qasm2_loads(open(os.path.join(NAM_DIR, fname)).read()).num_qubits
    T = T_for(q)
    t0 = time.perf_counter()
    fid = trajectory_circuit_fidelity_events(phys, config, num_trajectories=T,
                                             seed=SEED, backend=SIM_BACKEND)
    dt = round(time.perf_counter() - t0, 1)
    results.append({'model': MODEL_DIR, 'circuit': fname, 'qubits': q,
                    'swaps': swaps, 'T': T, 'fid': fid, 'sec': dt})
    print(f'{fname} [{MODEL_DIR}] q={q} T={T} sw={swaps} fid={fid:.4f} '
          f'({dt:.0f}s)', flush=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=1)

import math as _m


def _logmean(v):
    return _m.exp(statistics.mean(_m.log(max(x, 1e-9)) for x in v)) if v else 0


fs = [r['fid'] for r in results]
sws = [r['swaps'] for r in results]
n_trunc = sum(1 for r in results if r.get('truncated'))
print(f'=== {MODEL_DIR} 汇总（混合精度 T, seed={SEED}）===', flush=True)
print(f'Fid mean: {statistics.mean(fs):.4f}  logmean: {_logmean(fs):.4f}  '
      f'SWAPs: {statistics.mean(sws):.1f}  (n={len(fs)}, TRUNC跳过={n_trunc})')
fs_ok = [r['fid'] for r in results if not r.get('truncated')]
if fs_ok and n_trunc:
    print(f'Fid mean (completed-only, n={len(fs_ok)}): '
          f'{statistics.mean(fs_ok):.4f}  logmean: {_logmean(fs_ok):.4f}')
print('ALL DONE', flush=True)
