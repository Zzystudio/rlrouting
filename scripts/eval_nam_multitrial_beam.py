"""NAM benchmark：多试验 ε-扰动 beam5 评估（best-of-K 搜索多样性）。

对 R3b 模型（policy_r2b.pt，旧 obs——lookahead_features=False）：
  - 每条 NAM 电路跑 K=3 次 ε-扰动 beam5（logit 加 N(0, σ=1.0) 扰动，
    种子 1000/2000/3000——SABRE ε-random tie-break 的 RL 版）
  - 每次试验的物理线路用 trajectory_v2 ×16 (seed=0) 评保真度
  - 取最优试验，并用独立种子 999 复核（避免 winner's-curse）
对照：单次无扰动 beam5（v2sim_nam_fidelity.json 缓存）与 SABRE。

用法: python3 scripts/eval_nam_multitrial_beam.py [--trials 1000,2000,3000] [--sigma 1.0]
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
torch.set_num_threads(8)

from qiskit.qasm2 import loads as qasm2_loads

from routing.graph.circuit_dag import CircuitDAG
from routing.rl.eval_policy import evaluate_circuit_beam, load_topo
from routing.rl.agent import PPOAgent
from routing.gnn.encoder import SubGNN
from sim.trajectory_sim_v2 import trajectory_circuit_fidelity_events
from utils.data_gen import random_circuit

ap = argparse.ArgumentParser()
ap.add_argument('--trials', type=str, default='1000,2000,3000')
ap.add_argument('--sigma', type=float, default=1.0)
ap.add_argument('--model', default='models/policy_r2b.pt')
ap.add_argument('--out', default='benchmark/routed/v2sim_nam_multitrial.json')
args = ap.parse_args()

NTRAJ = 16
TRIAL_SEEDS = [int(s) for s in args.trials.split(',')]
SIGMA = args.sigma

config, hw, coupling_map = load_topo('traindata/topo/tianyan176_20q.json')

# R3b 模型：旧 obs（lookahead_features=False）匹配
qc0 = random_circuit(5, 6, seed=1000)
dag0 = CircuitDAG.from_circuit(qc0)
gnn = SubGNN(subgraph='full')
gnn.eval()
env0 = __import__('routing.rl.env', fromlist=['RoutingEnv']).RoutingEnv(
    dag0, hw, coupling_map, reward_mode='routing', max_episode_steps=100,
    random_init=False, seed=0, gnn=gnn, use_gnn=True,
    max_num_qubits=20, max_num_edges=29,
    lookahead_features=False)
agent = PPOAgent(
    obs_dim=int(np.prod(env0.observation_space.shape)),
    action_dim=29 + 1, device='cpu', gnn=gnn,
    num_qubits=20, num_edges=29, coupling_map=coupling_map,
    with_commit=True)
agent.load(args.model)
agent.ac.eval()
if agent.gnn is not None:
    agent.gnn.eval()
print(f"模型加载 OK: {args.model}", flush=True)

NAM_DIR = 'benchmark/nam_circs'
circuits = sorted(f for f in os.listdir(NAM_DIR) if f.endswith('.qasm'))
_csize = {}
for _f in circuits:
    _dag = CircuitDAG.from_circuit(qasm2_loads(open(os.path.join(NAM_DIR, _f)).read()))
    _csize[_f] = _dag.num_gates
circuits = sorted(circuits, key=lambda f: _csize[f])
print(f'电路按门数升序: ' + ', '.join(f'{f}:{_csize[f]}' for f in circuits[:4]) + ' ...', flush=True)

# 基线：单次 beam5 与 SABRE（v2sim_nam_fidelity.json 缓存，tianyan176）
base = {}
for r in json.load(open('benchmark/routed/v2sim_nam_fidelity.json')):
    md = r.get('model_dir')
    if md == 'r3b_beam5':
        base.setdefault('r3b_beam5', {})[r['circuit']] = r['fid_v2_r3b_beam5']
    elif md == 'SABRE':
        base.setdefault('SABRE', {})[r['circuit']] = r['fid_v2_sabre']

done_fids = {}   # {(circuit, trial): fid}
out_rows = []
_tpath = args.out + '.trials'
if os.path.exists(_tpath):
    for _l in open(_tpath):
        try:
            _r = json.loads(_l)
            done_fids[(_r['circuit'], _r['trial'])] = float(_r['fid'])
        except Exception:
            pass
    print(f'断点续跑: {len(done_fids)} 个已完成试验', flush=True)


def _run_trial(dag, fname, tseed, tsigma):
    m = evaluate_circuit_beam(
        dag, hw, coupling_map, agent,
        reward_mode='routing',
        max_episode_steps=500,
        seed=0,
        beam_width=5,
        max_num_qubits=20,
        use_scheduler=True,
        config=config,
        fidelity_sim='trajectory_v2',
        num_trajectories=NTRAJ,
        traj_seed=0,
        lookahead_features=False,
        logit_noise_std=tsigma,
        logit_noise_seed=(tseed if tsigma > 0 else None),
    )
    with open(_tpath, 'a') as _f:
        _f.write(json.dumps({'circuit': fname, 'trial': tseed,
                             'fid': m.fidelity, 'swaps': m.num_swaps}) + '\n')
    return m

for fname in circuits:
    qc = qasm2_loads(open(os.path.join(NAM_DIR, fname)).read())
    dag = CircuitDAG.from_circuit(qc)

    trial_fids = {}
    trial_pqasms = {}
    trial_list = [(0, 0.0)] + [(t, SIGMA) for t in TRIAL_SEEDS]  # trial 0 无扰动
    for tseed, tsigma in trial_list:
        if (fname, tseed) in done_fids:
            trial_fids[tseed] = done_fids[(fname, tseed)]
            print(f'  {fname} trial {tseed} [cache] fid={trial_fids[tseed]:.4f}',
                  flush=True)
            continue
        print(f'  {fname} trial seed={tseed} sigma={tsigma} 开始...', flush=True)
        m = _run_trial(dag, fname, tseed, tsigma)
        trial_fids[tseed] = m.fidelity
        if m.phys_qasm:
            trial_pqasms[tseed] = m.phys_qasm
    # 最优试验（seed=0 MC 口径下选择；含无扰动 trial 0）
    best_tseed = max(trial_fids, key=lambda t: trial_fids[t])
    # 独立种子复核（避免 winner's curse）：换 999 种子重评
    pq = trial_pqasms.get(best_tseed)
    if pq:
        qc_best = qasm2_loads(pq.replace(
            'gate swap a,b { cx a,b; cx b,a; cx a,b; }\n', ''))
        best_fid_indep = trajectory_circuit_fidelity_events(
            qc_best, config, num_trajectories=16, seed=999)
    else:
        best_fid_indep = max(trial_fids.values())

    # 最优试验的物理线路落盘（供放置质量分析复用）
    pq_best = trial_pqasms.get(best_tseed)
    if pq_best:
        os.makedirs('benchmark/routed/r3b_multitrial', exist_ok=True)
        with open(f'benchmark/routed/r3b_multitrial/{fname.removesuffix(".qasm")}.json', 'w') as _f:
            json.dump({'circuit': fname, 'model': 'r3b_multitrial',
                       'routed_qasm': pq_best.replace(
                           'gate swap a,b { cx a,b; cx b,a; cx a,b; }\n', ''),
                       'fid_best': max(trial_fids.values())}, _f, indent=1)
    row = {
        'circuit': fname, 'qubits_logical': qc.num_qubits,
        'trials': {str(t): f for t, f in trial_fids.items()},
        'best_trial': best_tseed, 'fid_bestof3': max(trial_fids.values()),
        'fid_independent_check': best_fid_indep,
        'swaps_best': None,
    }
    # SABRE / 单次 beam5 基线
    row['fid_single_beam5'] = base.get('r3b_beam5', {}).get(fname)
    row['fid_sabre'] = base.get('SABRE', {}).get(fname)
    out_rows.append(row)
    sb = row.get('fid_sabre')
    sb_str = f'{sb:.4f}' if sb is not None else '--'
    print(f'{fname}: trials={[f"{t:.4f}" for t in trial_fids.values()]} '
          f'best={max(trial_fids.values()):.4f} indep_check={best_fid_indep:.4f} '
          f'SABRE={sb_str}', flush=True)
    with open(args.out, 'w') as f:
        json.dump(out_rows, f, indent=1)

# ---- 汇总 ----
def _mean(v):
    return statistics.mean(v) if v else float('nan')


import math as _m


def _logmean(v):
    return _m.exp(statistics.mean(_m.log(max(x, 1e-9)) for x in v)) if v else float('nan')


print(f'\n=== 汇总（best-of-{len(TRIAL_SEEDS)} ε-扰动 beam5, σ={SIGMA}）===', flush=True)
for tag, keyf in [('单次 beam5（基线）', 'fid_single_beam5'),
                  ('best-of-K (MC同种子)', 'fid_bestof3'),
                  ('best-of-K (独立种子复核)', 'fid_independent_check'),
                  ('SABRE', 'fid_sabre')]:
    vals = [r.get(keyf) for r in out_rows if r.get(keyf) is not None]
    if vals:
        print(f'{tag:<24} mean={_mean(vals):.4f}  logmean={_logmean(vals):.4f}')
sb_better = sum(1 for r in out_rows
                if r.get('fid_sabre') is not None
                and r['fid_independent_check'] > r['fid_sabre'])
print(f'best-of-K 胜 SABRE（独立复核口径）: {sb_better}/{len(out_rows)}')
print('ALL DONE', flush=True)
