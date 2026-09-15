"""NAM benchmark：l05 vs SABRE，事件级 v2 模拟器（trajectory_sim_v2）保真度。

协议与 v1 NAM 基准（doc/train.md「NAM Circuits Benchmark」）对齐：
trajectory×16, seed=0，仅保真度模拟器从 v1 同步波（trajectory_sched）换为
事件级 v2（trajectory_v2）。路由与保真度模拟器无关：
- l05：<model_dir> 既有路由 QASM（不重新路由）
- SABRE：sabre_route 重路由（decay, 20 trials, seed=0，与项目基线同参）

结果缓存（OUT_JSON）：SABRE 与模型结果分行存储（model_dir='SABRE' / <model_dir>），
已计算的条目自动跳过，支持增量补跑。

用法: python3 scripts/eval_nam_v2sim.py [num_trajectories=16] [model_dir=l05_nam] [topo=tianyan176_20q] [out_json=v2sim_nam_fidelity.json]
"""
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qiskit.qasm2 import loads as qasm2_loads

from sim.trajectory_sim_v2 import trajectory_circuit_fidelity_events
from routing.routing import sabre_route
from routing.rl.eval_policy import load_topo

TOPO_NAME = sys.argv[3] if len(sys.argv) > 3 else 'tianyan176_20q'
config, hw, coupling_map = load_topo(f'traindata/topo/{TOPO_NAME}.json')
SWAP_DEF = 'gate swap a,b { cx a,b; cx b,a; cx a,b; }\n'
NAM_DIR = 'benchmark/nam_circs'
OUT_JSON = (sys.argv[4] if len(sys.argv) > 4
            else f'benchmark/routed/v2sim_nam_fidelity_{TOPO_NAME}.json')
NTRAJ = int(sys.argv[1]) if len(sys.argv) > 1 else 16
MODEL_DIR = sys.argv[2] if len(sys.argv) > 2 else 'l05_nam'
SEED = 0
SAB_KEY = 'fid_v2_sabre'
MOD_KEY = f'fid_v2_{MODEL_DIR}'

# 缓存：{(model_dir, circuit): row}（SABRE 行按拓扑隔离，避免跨拓扑污染）
cache = {}
if os.path.exists(OUT_JSON):
    for r in json.load(open(OUT_JSON)):
        cache[(r.get('model_dir'), r['circuit'])] = r

circuits = sorted(f for f in os.listdir(NAM_DIR) if f.endswith('.qasm'))
merged = []
for fname in circuits:
    name = fname.removesuffix('.qasm')
    row = {'model_dir': MODEL_DIR, 'circuit': fname}

    # --- SABRE 侧（模型无关，命中缓存即复用） ---
    sab = cache.get(('SABRE@' + TOPO_NAME, fname))
    if sab is None:
        qc_raw = qasm2_loads(open(os.path.join(NAM_DIR, fname)).read())
        phys, info = sabre_route(qc_raw, config, seed=SEED)
        t0 = time.perf_counter()
        fid_sab = trajectory_circuit_fidelity_events(
            phys, config, num_trajectories=NTRAJ, seed=SEED)
        dt = round(time.perf_counter() - t0, 1)
        sab = {'model_dir': 'SABRE@' + TOPO_NAME, 'circuit': fname,
               'qubits': qc_raw.num_qubits, 'swaps': info['num_swaps'],
               SAB_KEY: fid_sab, 'sec': dt}
        cache[('SABRE', fname)] = sab
        print(f'{fname}: [sabre] q={qc_raw.num_qubits} '
              f'sw={info["num_swaps"]} fid_v2={fid_sab:.4f} ({dt:.0f}s)',
              flush=True)
        with open(OUT_JSON, 'w') as f:
            json.dump(list(cache.values()), f, indent=1)
    row['sabre_swaps'] = sab['swaps']
    row[SAB_KEY] = sab[SAB_KEY]

    # --- 模型侧（<MODEL_DIR> 既有路由 QASM） ---
    mod = cache.get((MODEL_DIR, fname))
    if mod is None:
        d = json.load(open(os.path.join('benchmark/routed', MODEL_DIR,
                                        f'{name}.json')))
        qc_l05 = qasm2_loads(d['routed_qasm'].replace(
            'include "qelib1.inc";', 'include "qelib1.inc";\n' + SWAP_DEF))
        t0 = time.perf_counter()
        fid_mod = trajectory_circuit_fidelity_events(
            qc_l05, config, num_trajectories=NTRAJ, seed=SEED)
        dt = round(time.perf_counter() - t0, 1)
        mod = {'model_dir': MODEL_DIR, 'circuit': fname,
               'qubits': qc_l05.num_qubits, 'swaps': d['num_swaps'],
               MOD_KEY: fid_mod, 'sec': dt}
        cache[(MODEL_DIR, fname)] = mod
        print(f'{fname}: [{MODEL_DIR}] q={qc_l05.num_qubits} '
              f'sw={d["num_swaps"]} fid_v2={fid_mod:.4f} ({dt:.0f}s)',
              flush=True)
        with open(OUT_JSON, 'w') as f:
            json.dump(list(cache.values()), f, indent=1)
    row['l05_swaps'] = mod['swaps']
    row[MOD_KEY] = mod[MOD_KEY]
    merged.append(row)

sw_s = [r['sabre_swaps'] for r in merged]
sw_l = [r['l05_swaps'] for r in merged]
f_s = [r[SAB_KEY] for r in merged]
f_l = [r[MOD_KEY] for r in merged]


def logmean(vals):
    return math.exp(statistics.mean(math.log(max(v, 1e-9)) for v in vals))


print(f'=== 汇总 [{MODEL_DIR}] (trajectory_v2 x{NTRAJ}, seed={SEED}) ===',
      flush=True)
print(f'SWAPs mean:  sabre={statistics.mean(sw_s):.1f}  '
      f'{MODEL_DIR}={statistics.mean(sw_l):.1f}')
print(f'Fid mean:    sabre={statistics.mean(f_s):.4f}  '
      f'{MODEL_DIR}={statistics.mean(f_l):.4f}')
print(f'Fid logmean: sabre={logmean(f_s):.4f}  '
      f'{MODEL_DIR}={logmean(f_l):.4f}')
wins = sum(1 for r in merged if r[MOD_KEY] > r[SAB_KEY])
print(f'{MODEL_DIR} 胜 SABRE: {wins}/{len(merged)}')
print('ALL DONE', flush=True)
