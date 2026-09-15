"""tianyan287_stronghetero 上 R3b_beam5 vs SABRE 逐电路对比。

R3b 策略路由 NAM 电路 → tianyan287_stronghetero 耦合图上的物理线路；
SABRE 直接 sabre_route → 同一拓扑。
两条路由都用 trajectory_v2 评估保真度（强异构噪声配置）。

用法: python3 scripts/eval_t287hetero_compare.py [num_trajectories=16]
"""
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch

from qiskit import QuantumCircuit
from qiskit.qasm2 import loads as qasm2_loads

from sim.trajectory_sim_v2 import trajectory_circuit_fidelity_events
from routing.routing import sabre_route
from routing.rl.eval_policy import load_topo, evaluate_circuit_beam
from routing.graph.circuit_dag import CircuitDAG
from routing.rl.agent import PPOAgent
from routing.gnn.encoder import SubGNN
from routing.rl.env import RoutingEnv
from utils.data_gen import random_circuit

TOPO = 'traindata/topo/tianyan287_20q_stronghetero.json'
NAM_DIR = 'benchmark/nam_circs'
OUT_JSON = 'benchmark/routed/v2sim_t287hetero_compare.json'
NTRAJ = int(sys.argv[1]) if len(sys.argv) > 1 else 16
SEED = 42

config, hw, coupling_map = load_topo(TOPO)
n_edges = len(coupling_map)
SWAP_DEF = 'gate swap a,b { cx a,b; cx b,a; cx a,b; }\n'

# --- R3b 模型加载 ---
gnn = SubGNN(subgraph='full')
gnn.eval()
qc0 = random_circuit(5, 6, seed=1000)
dag0 = CircuitDAG.from_circuit(qc0)
env0 = RoutingEnv(dag0, hw, coupling_map, reward_mode='routing',
                  max_episode_steps=100, random_init=False, seed=0,
                  gnn=gnn, use_gnn=True, max_num_qubits=20,
                  max_num_edges=n_edges, lookahead_features=False)
agent = PPOAgent(
    obs_dim=int(np.prod(env0.observation_space.shape)),
    action_dim=n_edges + 1, device='cpu', gnn=gnn,
    num_qubits=20, num_edges=n_edges, coupling_map=coupling_map,
    with_commit=True)
agent.load('models/policy_r2b.pt')
agent.ac.eval()
if agent.gnn is not None:
    agent.gnn.eval()
print(f'R3b 模型加载 OK (edges={n_edges})', flush=True)

circuits = sorted(f for f in os.listdir(NAM_DIR) if f.endswith('.qasm'))
out_rows = []
for fname in circuits:
    name = fname.removesuffix('.qasm')
    qc_raw = qasm2_loads(open(os.path.join(NAM_DIR, fname)).read())
    dag_raw = CircuitDAG.from_circuit(qc_raw)

    # --- SABRE 路由 ---
    phys_sab, info_s = sabre_route(qc_raw, config, seed=0)

    # --- R3b beam5 路由 ---
    m_beam = evaluate_circuit_beam(
        dag_raw, hw, coupling_map, agent,
        reward_mode='routing', max_episode_steps=500, seed=0,
        beam_width=5, max_num_qubits=20, use_scheduler=True,
        config=config, lookahead_features=False, no_progress_limit=200,
        fidelity_sim='trajectory_v2', dump_schedule=True,
    )
    phys_beam = m_beam.phys_qasm and qasm2_loads(
        m_beam.phys_qasm.replace('include "qelib1.inc";',
                                 'include "qelib1.inc";\n' + SWAP_DEF))

    # --- 保真度（v2, 强异构噪声配置）---
    t0 = time.perf_counter()
    fid_sab = trajectory_circuit_fidelity_events(
        phys_sab, config, num_trajectories=NTRAJ, seed=SEED)
    fid_beam = m_beam.fidelity if hasattr(m_beam, 'fidelity') and m_beam.fidelity is not None else None
    dt = round(time.perf_counter() - t0, 1)

    row = {'circuit': fname, 'qubits': qc_raw.num_qubits,
           'sabre_swaps': info_s['num_swaps'],
           'beam_swaps': m_beam.num_swaps,
           'fid_sabre': fid_sab, 'fid_beam5': fid_beam, 'sec': dt}
    out_rows.append(row)
    print(f'{fname}: q={qc_raw.num_qubits} S_sw={info_s["num_swaps"]} '
          f'B_sw={m_beam.num_swaps} fid SABRE={fid_sab:.4f} B5={fid_beam:.4f} '
          f'({dt:.0f}s)', flush=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(out_rows, f, indent=1)

# --- 汇总 ---
sw_s = [r['sabre_swaps'] for r in out_rows]
sw_b = [r['beam_swaps'] for r in out_rows]
f_s = [r['fid_sabre'] for r in out_rows]
f_b = [r['fid_beam5'] for r in out_rows]

import math as _m


def _logmean(v):
    return _m.exp(statistics.mean(_m.log(max(x, 1e-9)) for x in v)) if v else 0


print(f'\n=== tianyan287_stronghetero 汇总（trajectory_v2 ×{NTRAJ}, seed={SEED}）===',
      flush=True)
print(f'SWAPs mean:  SABRE={statistics.mean(sw_s):.1f}  B5={statistics.mean(sw_b):.1f}')
print(f'Fid mean:    SABRE={statistics.mean(f_s):.4f}  B5={statistics.mean(f_b):.4f}')
print(f'Fid logmean: SABRE={_logmean(f_s):.4f}  B5={_logmean(f_b):.4f}')
wins = sum(1 for r in out_rows if r['fid_beam5'] > r['fid_sabre'])
print(f'beam5 胜 SABRE: {wins}/{len(out_rows)}')
print('ALL DONE', flush=True)
