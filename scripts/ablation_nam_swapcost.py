"""NAM 口径消融：v1→v2 保真度排序反转的归因。

对每条 NAM 电路 × {l05, SABRE} 计算三档保真度（trajectory×16, seed=0）：
  A) v1sem  : v1 噪声语义在 v2 引擎上重建——1q 统一 0.1µs（含 rz）、
              swap=0.3µs 且无噪声（swap_noise=False）
  B) noswap : v2 时长（镜像 GATE_DURATION_TABLE，rz=0/sx=0.035/swap=0.9）
              + swap_noise=False（仍不计 SWAP 噪声）
  C) full   : v2 完整语义（复用 v2sim_nam_fidelity.json 缓存，不重算）

A→B 隔离「1q 时长修正」（rz 虚拟门/短 1q 门不再计退相干）的贡献；
B→C 隔离「SWAP 噪声计价」（3×CX 退极化/串扰 + 0.9µs 热弛豫）的贡献。
sanity：A 应接近 v1 同步波实测值（sabre_summary.json / l05_summary.json）。

用法: python3 scripts/ablation_nam_swapcost.py [num_trajectories=16]
"""
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qiskit.qasm2 import loads as qasm2_loads

from sim.trajectory_sim_v2 import (
    EventNoiseConfig,
    EventTrajectorySimulator,
    _reduce_phys_circuit_for_fidelity_v2,
    schedule_phys_circuit_events,
)
from routing.routing import sabre_route
from routing.rl.eval_policy import load_topo

config, hw, coupling_map = load_topo('traindata/topo/tianyan176_20q.json')
SWAP_DEF = 'gate swap a,b { cx a,b; cx b,a; cx a,b; }\n'
NAM_DIR = 'benchmark/nam_circs'
L05_DIR = 'benchmark/routed/l05'
OUT_JSON = 'benchmark/routed/v2sim_nam_ablation.json'
NTRAJ = int(sys.argv[1]) if len(sys.argv) > 1 else 16
SEED = 0

# v1 语义时长：所有 1q 门 0.1µs（含 rz），2q/swap 0.3µs
DUR_V1 = {'cx': 0.3, 'cz': 0.3, 'swap': 0.3, 'ecr': 0.3,
          'sx': 0.1, 'x': 0.1, 'y': 0.1, 'h': 0.1, 's': 0.1, 't': 0.1,
          'sdg': 0.1, 'tdg': 0.1, 'rz': 0.1, 'z': 0.1, 'id': 0.0}

cfg_v1sem = EventNoiseConfig.from_noise_config(config, swap_noise=False)
cfg_noswap = EventNoiseConfig.from_noise_config(config, swap_noise=False)

# v2 full 缓存（fid_v2_sabre / fid_v2_l05）
full_cache = {}
_v2path = 'benchmark/routed/v2sim_nam_fidelity.json'
if os.path.exists(_v2path):
    for r in json.load(open(_v2path)):
        full_cache[(r['model_dir'], r['circuit'])] = r
# v1 实测值（sanity 对照）
def _v1_fids(path):
    d = json.load(open(path))
    rows = d['results'] if isinstance(d, dict) else d
    return {r['circuit']: r.get('fidelity') for r in rows}


v1_sabre = _v1_fids('benchmark/routed/sabre_summary.json')
v1_l05 = _v1_fids('benchmark/routed/l05_summary.json')

# 消融缓存
ab_cache = {}
if os.path.exists(OUT_JSON):
    for r in json.load(open(OUT_JSON)):
        ab_cache[(r['model_dir'], r['circuit'])] = r


def fid_of(qc, cfg, durations):
    """v2 引擎保真度（比特子集缩减 + 平铺 ASAP 事件）。"""
    rc, rcfg, _ = _reduce_phys_circuit_for_fidelity_v2(qc, cfg)
    sim = EventTrajectorySimulator(rcfg, num_trajectories=NTRAJ, seed=SEED)
    events = schedule_phys_circuit_events(rc, durations)
    return sim.fidelity_events(rc, events)


circuits = sorted(f for f in os.listdir(NAM_DIR) if f.endswith('.qasm'))
for fname in circuits:
    name = fname.removesuffix('.qasm')
    # 路由电路：SABRE 重路由（快）；l05 复用既有 QASM
    qc_raw = qasm2_loads(open(os.path.join(NAM_DIR, fname)).read())
    phys, info = sabre_route(qc_raw, config, seed=SEED)
    d = json.load(open(os.path.join(L05_DIR, f'{name}.json')))
    qc_l05 = qasm2_loads(d['routed_qasm'].replace(
        'include "qelib1.inc";', 'include "qelib1.inc";\n' + SWAP_DEF))

    for method, qc, swaps in (('sabre', phys, info['num_swaps']),
                              ('l05', qc_l05, d['num_swaps'])):
        row = ab_cache.get((method, fname))
        if row is None:
            row = {'model_dir': method, 'circuit': fname, 'swaps': swaps}
            ab_cache[(method, fname)] = row
        row['swaps'] = swaps
        row['fid_v1_recorded'] = (v1_sabre if method == 'sabre'
                                  else v1_l05).get(fname)
        row['fid_full'] = full_cache.get((method, fname), {}).get(
            'fid_v2_' + ('sabre' if method == 'sabre' else 'l05'))
        for tag, cfg, dur in (('v1sem', cfg_v1sem, DUR_V1),
                              ('noswap', cfg_noswap, None)):
            if row.get(f'fid_{tag}') is None:
                t0 = time.perf_counter()
                row[f'fid_{tag}'] = fid_of(qc, cfg, dur)
                dt = round(time.perf_counter() - t0, 1)
                print(f'{fname} [{method}] {tag}: {row[f"fid_{tag}"]:.4f} '
                      f'({dt:.0f}s)', flush=True)
                with open(OUT_JSON, 'w') as f:
                    json.dump(list(ab_cache.values()), f, indent=1)
            else:
                print(f'{fname} [{method}] {tag}: {row[f"fid_{tag}"]:.4f} '
                      f'[cache]', flush=True)

# ---- 汇总 ----
print('\n=== 消融汇总 (NAM 19 电路, trajectory x%d, seed=%d) ===' % (NTRAJ, SEED),
      flush=True)
hdr = f'{"setting":<10} {"SABRE":>8} {"l05":>8} {"l05-SABRE":>10}'
print(hdr)
gaps = {}
for tag in ('v1sem', 'noswap', 'full'):
    fs = [ab_cache[(m, f)].get(f'fid_{tag}') for f in circuits for m in
          ('sabre', 'l05')]
    if any(v is None for v in fs):
        print(f'{tag:<10} (incomplete, skip)')
        continue
    ms = statistics.mean(ab_cache[('sabre', f)][f'fid_{tag}'] for f in circuits)
    ml = statistics.mean(ab_cache[('l05', f)][f'fid_{tag}'] for f in circuits)
    gaps[tag] = (ms, ml, ml - ms)
    print(f'{tag:<10} {ms:>8.4f} {ml:>8.4f} {ml - ms:>+10.4f}')
mv1s = statistics.mean(v for v in v1_sabre.values() if v is not None)
mv1l = statistics.mean(v for v in v1_l05.values() if v is not None)
print(f'{"v1实测":<10} {mv1s:>8.4f} {mv1l:>8.4f} {mv1l - mv1s:>+10.4f}'
      f'   <- sanity：v1sem 应接近此行')
print('\n归因（l05-SABRE 差值分解，v1sem→noswap→full）:')
if 'v1sem' in gaps and 'noswap' in gaps and 'full' in gaps:
    d1 = gaps['noswap'][2] - gaps['v1sem'][2]
    d2 = gaps['full'][2] - gaps['noswap'][2]
    print(f'  1q 时长修正 贡献: {d1:+.4f}')
    print(f'  SWAP 噪声计价 贡献: {d2:+.4f}')
    print(f'  合计: {gaps["full"][2] - gaps["v1sem"][2]:+.4f}')
print('ALL DONE', flush=True)
