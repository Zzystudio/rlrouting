"""csla_mux_3 深挖：为什么多试验 best-of-4 提升 +0.15？

对比三条路由（同一电路，tianyan176_20q）：
  A) SABRE（sabre_route seed=0，decay/20trials）
  B) R3b 单次 beam5（benchmark/routed/r3b_beam5/csla_mux_3.json）
  C) R3b 多试验最优（logit_noise_seed=1000, σ=1.0, beam5——确定性重放）

结构指标：SWAP 数、CX 平均边误差、调度串扰事件、makespan、idle；
保真度：trajectory_v2 T=128（独立 seed=42，排除 T=16 的 MC 方差）。
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
from qiskit.qasm2 import loads as qasm2_loads

from routing.graph.circuit_dag import CircuitDAG
from routing.rl.agent import PPOAgent
from routing.gnn.encoder import SubGNN
from routing.rl.env import RoutingEnv
from routing.rl.eval_policy import evaluate_circuit_beam, load_topo
from routing.routing import sabre_route
from routing.timing import schedule_routed_circuit
from sim.trajectory_sim_v2 import trajectory_circuit_fidelity_events
from utils.data_gen import random_circuit
from utils.metrics import state_fidelity

torch.set_num_threads(8)

config, hw, coupling_map = load_topo('traindata/topo/tianyan176_20q.json')
SWAP_DEF = 'gate swap a,b { cx a,b; cx b,a; cx a,b; }\n'
CIRCUIT = 'csla_mux_3'

# ---- R3b agent（旧 obs）----
gnn = SubGNN(subgraph='full')
gnn.eval()
qc0 = random_circuit(5, 6, seed=1000)
dag0 = CircuitDAG.from_circuit(qc0)
env0 = RoutingEnv(dag0, hw, coupling_map, reward_mode='routing',
                  max_episode_steps=100, random_init=False, seed=0,
                  gnn=gnn, use_gnn=True, max_num_qubits=20,
                  max_num_edges=29, lookahead_features=False)
agent = PPOAgent(obs_dim=int(np.prod(env0.observation_space.shape)),
                 action_dim=30, device='cpu', gnn=gnn,
                 num_qubits=20, num_edges=29, coupling_map=coupling_map,
                 with_commit=True)
agent.load('models/policy_r2b.pt')
agent.ac.eval()
if agent.gnn is not None:
    agent.gnn.eval()

# ---- 三条路由 ----
routes = {}

# A) SABRE
qc_raw = qasm2_loads(open(f'benchmark/nam_circs/{CIRCUIT}.qasm').read())
phys_s, info_s = sabre_route(qc_raw, config, seed=0)
routes['SABRE'] = phys_s

# B) R3b 单次 beam5（既有产物）
d = json.load(open(f'benchmark/routed/r3b_beam5/{CIRCUIT}.json'))
routes['R3b_beam5'] = qasm2_loads(d['routed_qasm'].replace(
    'include "qelib1.inc";', 'include "qelib1.inc";\n' + SWAP_DEF))

# C) R3b 多试验最优（确定性重放：logit_noise_seed=1000, σ=1.0）
qc_raw2 = qasm2_loads(open(f'benchmark/nam_circs/{CIRCUIT}.qasm').read())
dag2 = CircuitDAG.from_circuit(qc_raw2)
m = evaluate_circuit_beam(dag2, hw, coupling_map, agent,
                          reward_mode='routing', max_episode_steps=500,
                          seed=0, beam_width=5, max_num_qubits=20,
                          use_scheduler=True, config=config,
                          lookahead_features=False, no_progress_limit=200,
                          logit_noise_std=1.0, logit_noise_seed=1000,
                          num_trajectories=16, traj_seed=0,
                          dump_schedule=True)
routes['R3b_mt_best'] = qasm2_loads(m.phys_qasm.replace(
    'include "qelib1.inc";',
    'include "qelib1.inc";\n' + SWAP_DEF))
if routes['R3b_mt_best'] is None:
    routes['R3b_mt_best'] = m.phys_qasm
print(f"多试验最优重放: swaps={m.num_swaps} (T=16 口径 fid={m.fidelity:.4f})",
      flush=True)

# ---- 结构指标 ----
def cx_err_stats(phys):
    errs = []
    for inst in phys.data:
        if inst.operation.name.lower() == 'cx':
            p = phys.find_bit(inst.qubits[0]).index
            q = phys.find_bit(inst.qubits[1]).index
            errs.append(float(hw.two_q_err[p, q]))
    return (statistics.mean(errs) if errs else 0.0), len(errs)

def sched_stats(phys):
    dag = CircuitDAG.from_circuit(phys)
    mk, xt, st = schedule_routed_circuit(dag, hw, mapping=None)
    return mk, xt, st.get('density', 0)

print(f"\n=== 结构指标（{CIRCUIT}, 15q/{hw.num_qubits}q 拓扑）===")
print(f"{'路由':<14} {'SWAPs':>6} {'CX均边误差':>10} {'串扰事件':>8} {'makespan':>8} {'并行密度':>7}")
stats_rows = {}
for name, phys in routes.items():
    se, ns = cx_err_stats(phys)
    mk, xt, dens = sched_stats(phys)
    stats_rows[name] = (se, xt, mk, dens)
    print(f"{name:<14} {ns:>6} {se:>10.4f} {xt:>8.3f} {mk:>8.1f} {dens:>7.2f}")

# ---- 高精度保真度（T=128, seed=42 独立）----
print("\n=== 高精度保真度（T=128, seed=42）===", flush=True)
fid_rows = {}
for name, phys in routes.items():
    t0 = time.time()
    fid = trajectory_circuit_fidelity_events(phys, config, num_trajectories=128,
                                             seed=42)
    fid_rows[name] = fid
    print(f"{name}: T128 fid={fid:.4f} ({time.time()-t0:.0f}s)", flush=True)

# ---- 汇总 ----
print("\n=== 结论 ===", flush=True)
f16 = {'SABRE': 0.0007, 'R3b_beam5': 0.0156, 'R3b_mt_best(T16)': 0.1510}
for name in ('SABRE', 'R3b_beam5', 'R3b_mt_best'):
    se, xt, mk, dens = stats_rows[name]
    print(f"{name}: CX均边误差 {se:.4f}  串扰 {xt:.3f}  makespan {mk:.1f}  "
          f"T128 fid {fid_rows[name]:.4f}")

out = {'circuit': CIRCUIT,
       'routes': {n: {'cx_err_mean': stats_rows[n][0], 'xtalk': stats_rows[n][1],
                      'makespan': stats_rows[n][2],
                      'fid_T128_seed42': fid_rows[n]} for n in stats_rows}}
with open(f'benchmark/routed/v1sim_csla_analysis.json', 'w') as f:
    json.dump(out, f, indent=1)
print('saved: benchmark/routed/v1sim_csla_analysis.json')
