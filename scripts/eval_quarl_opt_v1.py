"""quarl_opt 优化电路 vs 原始 NAM：R3b_beam5 与 SABRE 的 v1 逐电路对比。

对 benchmark/quarl_opt_wo_rm/ 中的每条优化电路：
  1. 用 R3b_beam5 模型路由（evaluate_circuit_beam）
  2. 用 SABRE 路由（sabre_route seed=0）
  3. 两条路由均用 v1 trajectory_sched (×16, seed=0) 评估保真度
并匹配原始 NAM 基准结果（sabre_summary / l05_beam5_summary / v1sim_nam_compare），
输出逐电路对比 + 规模段汇总。

用法: python3 scripts/eval_quarl_opt_v1.py [num_trajectories=16]
"""
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qiskit.qasm2 import loads as qasm2_loads

from sim.trajectory_sim import trajectory_circuit_fidelity
from routing.routing import sabre_route
from routing.rl.eval_policy import load_topo, evaluate_circuit_beam
from routing.graph.circuit_dag import CircuitDAG
from routing.rl.agent import PPOAgent
from routing.gnn.encoder import SubGNN
from routing.rl.env import RoutingEnv
from utils.data_gen import random_circuit

import numpy as np
import torch

torch.set_num_threads(8)

config, hw, coupling_map = load_topo('traindata/topo/tianyan176_20q.json')
SWAP_DEF = 'gate swap a,b { cx a,b; cx b,a; cx a,b; }\n'
OPT_DIR = 'benchmark/quarl_opt_wo_rm'
OUT_JSON = 'benchmark/routed/v1sim_quarl_opt.json'
NTRAJ = int(sys.argv[1]) if len(sys.argv) > 1 else 16
SEED = 0

# --- R3b_beam5 模型加载 ---
gnn = SubGNN(subgraph='full')
gnn.eval()
qc0 = random_circuit(5, 6, seed=1000)
dag0 = CircuitDAG.from_circuit(qc0)
env0 = RoutingEnv(dag0, hw, coupling_map, reward_mode='routing',
                  max_episode_steps=100, random_init=False, seed=0,
                  gnn=gnn, use_gnn=True, max_num_qubits=20,
                  max_num_edges=29, lookahead_features=False)
agent = PPOAgent(
    obs_dim=int(np.prod(env0.observation_space.shape)),
    action_dim=29 + 1, device='cpu', gnn=gnn,
    num_qubits=20, num_edges=29, coupling_map=coupling_map,
    with_commit=True)
agent.load('models/policy_r2b.pt')
agent.ac.eval()
if agent.gnn is not None:
    agent.gnn.eval()
print('R3b_beam5 模型加载 OK', flush=True)

# --- 匹配优化电路与原始 NAM ---
opt_files = sorted(f for f in os.listdir(OPT_DIR) if f.endswith('.qasm'))
# 过滤 >20q 的电路（拓扑只有 20q）
valid_opt = []
for f in opt_files:
    qc_check = qasm2_loads(open(os.path.join(OPT_DIR, f)).read())
    if qc_check.num_qubits <= 20:
        valid_opt.append(f)
    else:
        print(f'跳过 {f}: {qc_check.num_qubits}q > 20q', flush=True)
opt_files = valid_opt
print(f'可用优化电路: {len(opt_files)} 条', flush=True)
# name 映射: barenco_tof_3_cost35.qasm → barenco_tof_3
name_map = {}
for f in opt_files:
    base = os.path.basename(f).removesuffix('.qasm')
    orig_name = base.rsplit('_cost', 1)[0]
    name_map[f] = orig_name

# 原始 NAM 基准结果（v1）
orig_v1 = {}
for path in ['benchmark/routed/sabre_summary.json', 'benchmark/routed/l05_beam5_summary.json']:
    if not os.path.exists(path):
        continue
    tag = 'SABRE' if 'sabre' in path else 'l05_beam5'
    d = json.load(open(path))
    rws = d['results'] if isinstance(d, dict) else d
    for r in rws:
        orig_v1.setdefault((tag, r['circuit']), r.get('fidelity'))

# 已有 NAM v1 对比数据
v1c = json.load(open('benchmark/routed/v1sim_nam_compare.json'))
v1_nam = {}
for r in v1c:
    v1_nam.setdefault(r['model'], {})[r['circuit']] = r.get('fid_v1')

results = []
for fname in opt_files:
    orig_name = name_map[fname]
    qc_opt = qasm2_loads(open(os.path.join(OPT_DIR, fname)).read())
    dag_opt = CircuitDAG.from_circuit(qc_opt)

    # --- R3b_beam5 路由优化电路 ---
    t0 = time.perf_counter()
    m_beam = evaluate_circuit_beam(
        dag_opt, hw, coupling_map, agent,
        reward_mode='routing', max_episode_steps=1000, seed=0,
        beam_width=5, max_num_qubits=20, use_scheduler=True,
        config=config, lookahead_features=False, no_progress_limit=200,
        dump_schedule=True,
    )
    t_beam = round(time.perf_counter() - t0, 1)
    phys_beam = None
    if m_beam.phys_qasm:
        phys_beam = qasm2_loads(m_beam.phys_qasm.replace(
            'include "qelib1.inc";', 'include "qelib1.inc";\n' + SWAP_DEF))
        fid_beam = trajectory_circuit_fidelity(
            phys_beam, config, num_trajectories=NTRAJ, seed=SEED, scheduled=True)
    else:
        fid_beam = None

    # --- SABRE 路由优化电路 ---
    t0 = time.perf_counter()
    phys_sab, info_s = sabre_route(qc_opt, config, seed=SEED)
    fid_sab = trajectory_circuit_fidelity(
        phys_sab, config, num_trajectories=NTRAJ, seed=SEED, scheduled=True)
    t_sab = round(time.perf_counter() - t0, 1)

    # 原始 NAM 基准数据
    key = orig_name + '.qasm'
    sab_orig = v1_nam.get('SABRE', {}).get(key)
    b5_orig = v1_nam.get('l05_beam5', {}).get(key)
    l05_orig = v1_nam.get('l05', {}).get(key)

    row = {
        'circuit': fname, 'qubits': qc_opt.num_qubits,
        'sabre_swaps_opt': info_s['num_swaps'],
        'beam_swaps_opt': m_beam.num_swaps,
        'fid_v1_sabre_opt': fid_sab, 'fid_v1_beam5_opt': fid_beam,
        'fid_v1_sabre_orig': sab_orig, 'fid_v1_beam5_orig': b5_orig,
        'fid_v1_l05_orig': l05_orig,
        'sec': dt if 'dt' in dir() else 0,
    }
    results.append(row)
    print(f'{fname}: SABRE sw={info_s["num_swaps"]} fid={fid_sab:.4f} | '
          f'B5 sw={m_beam.num_swaps} fid={fid_beam:.4f}', flush=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=1)

# ---- 汇总 ----
import math as _m


def _logmean(v):
    return _m.exp(statistics.mean(_m.log(max(x, 1e-9)) for x in v)) if v else 0


print(f'\n=== 汇总（v1 trajectory_sched ×{NTRAJ}, seed={SEED}）===', flush=True)
print(f'{"指标":<28} {"SABRE":>9} {"R3b_beam5":>10}')
print("-" * 52)
for tag, sk, bk in [('SABRE(优化电路)', 'fid_v1_sabre_opt', 'fid_v1_sabre_opt'),
                     ('B5(优化电路)', 'fid_v1_beam5_opt', 'fid_v1_beam5_opt')]:
    fs = [r[sk] for r in results if r.get(sk) is not None]
    if fs:
        print(f'{tag:<28} {statistics.mean(fs):>9.4f}')
for tag in ['fid_v1_sabre_orig', 'fid_v1_beam5_orig', 'fid_v1_l05_orig']:
    fs = [r.get(tag) for r in results if r.get(tag) is not None]
    if fs:
        m = tag.replace('fid_v1_', '')
        print(f'{"原始 NAM (" + m + ")":<28} {statistics.mean(fs):>9.4f}')

print('\n逐电路:')
for r in results:
    print(f"  {r['circuit']:<28} SABRE_fid={r.get('fid_v1_sabre_opt')}  "
          f"B5_fid={r.get('fid_v1_beam5_opt')}")
print('ALL DONE', flush=True)
