"""奖励-保真度一致性审计：重放轴。

对 in-dist 50 条 × 3 解（SABRE / argmax / beam5la+bud）：
  (初始布局, SWAP 序列) 穿训练配置的 env 重放 → R_train 总量 + 分量记账。
分量：r_gate / r_swap价 / r_budget / r_sched_txi(时间+串扰+空闲) / r_prop /
      r_xt_swap(残差) / r_shape(Φ telescoping) / r_terminal。
闸门：V3 重放有效性（模型解重放的 2Q 门序列 ≡ routed_qasm）。

输出: benchmark/routed/audit_replay_indist.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
from collections import Counter
from qiskit.qasm2 import loads as qasm2_loads

from routing.graph.circuit_dag import CircuitDAG
from routing.graph.features import HardwareFeatures
from routing.gnn.encoder import SubGNN
from routing.rl.agent import PPOAgent
from routing.rl.env import RoutingEnv
from routing.routing import sabre_route
from sim.sim import NoiseConfig  # noqa: F401  (load_topo 依赖链)

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--out', default='benchmark/routed/audit_replay_indist.json')
_ap.add_argument('--swap-price-scale', type=float, default=1.0)
_ap.add_argument('--lambda-budget', type=float, default=0.5)
_args = _ap.parse_args()
OUT = _args.out
SWAP_PRICE_SCALE = _args.swap_price_scale
LAMBDA_BUDGET = _args.lambda_budget
print(f'[audit] swap_price_scale={SWAP_PRICE_SCALE} lambda_budget={LAMBDA_BUDGET}')
VAL_DIR = 'benchmark/indist_val'
SWAP_DEF = 'gate swap a,b { cx a,b; cx b,a; cx a,b; }\n'
BUDGET_DELTA = 1.05

from routing.rl.eval_policy import load_topo
config, hw, coupling_map = load_topo('traindata/topo/tianyan287_20q.json')
tqe = config.two_q_gate_error
edge_index = {}
for i, (a, b) in enumerate(coupling_map):
    edge_index[(a, b)] = i
    edge_index[(b, a)] = i

circuits = sorted(f for f in os.listdir(VAL_DIR) if f.endswith('.qasm'))


class RecordingEnv(RoutingEnv):
    """分量记账：分量方法覆写累加 + 批量返回 stash + step 残差归因 r_xt_swap。
    闭合式：r_total ≡ r_gate + r_swap价 + r_prop + r_sched_txi + r_budget
            + r_xt_swap + r_shape + r_terminal（逐项可解释）。"""

    def reset_acc(self):
        self.acc = {'r_gate': 0.0, 'r_swap_price': 0.0, 'r_prop': 0.0,
                    'r_sched_txi': 0.0, 'r_budget': 0.0, 'r_shape': 0.0,
                    'r_terminal': 0.0, 'r_xt_swap': 0.0}
        self._last_batch = None
        self._batch_gate = 0.0

    def reset(self, **kw):
        self.reset_acc()
        return super().reset(**kw)

    def _auto_execute_batch(self):
        g0 = self.acc['r_gate']
        r = super()._auto_execute_batch()
        self._last_batch = r
        self._batch_gate = self.acc['r_gate'] - g0
        return r

    def _step_reward_execute(self, success, gate_idx):
        r = super()._step_reward_execute(success, gate_idx)
        self.acc['r_gate'] += r
        return r

    def _step_reward_swap(self, p=None, q=None):
        r = super()._step_reward_swap(p, q)
        self.acc['r_swap_price'] += r
        return r

    def _end_step(self, reward, info, compute_obs=True, phi_before=None):
        obs, r, done, trunc, info = super()._end_step(
            reward, info, compute_obs=compute_obs, phi_before=phi_before)
        if trunc:
            # 解析式截断罚：-unfinished_penalty × 剩余门数（与 env 内部公式一致）
            self.acc['r_terminal'] += (
                -self.unfinished_penalty * info.get('truncated_remaining', 0))
        if phi_before is not None:
            phi_after = 0.0 if (done or trunc) else self._phi()
            self.acc['r_shape'] += self.shaping_gamma * phi_after - phi_before
        return obs, r, done, trunc, info


ENV_KW = dict(reward_mode='routing', max_episode_steps=200, seed=0,
              use_gnn=False, noise_config=None,
              max_num_edges=len(coupling_map), mapping_phase=False,
              use_scheduler=True, eta_time=0.05, eta_xtalk_par=1.0,
              eta_idle=0.005, eta_xtalk=0.0, eta_parallel=0.0,
              reward_potential=True, pot_progress_b=0.20,
              w_err=0.02, w_xt=0.01, w_xt_swap=0.02,
              shaping_gamma=0.99, eta_shape=0.3, alpha_ext=0.5,
              unfinished_penalty=0.3, no_progress_limit=200,
              lambda_budget=LAMBDA_BUDGET, swap_price_scale=SWAP_PRICE_SCALE,
              edge_noise_features=True,
              beta_noise=0.5)


def extract_swaps(phys):
    return [tuple(phys.find_bit(q).index for q in inst.qubits)
            for inst in phys.data if inst.operation.name.lower() == 'swap']


def two_q_seq(phys):
    return [(inst.operation.name.lower(),
             tuple(phys.find_bit(q).index for q in inst.qubits))
            for inst in phys.data
            if len(inst.qubits) == 2
            and inst.operation.name.lower() not in ('measure', 'barrier')]


def normalize_layout(l, n=20):
    """布局归一化：list（virtual→physical）或 dict（'v'→phys，仅含已用虚位）
    → 定长 list[virtual]=physical；dict 缺失的虚位用恒等补齐（无门，不影响路由）。"""
    if l is None:
        return None
    if isinstance(l, dict):
        out = list(range(n))
        for k, v in l.items():
            out[int(k)] = int(v)
        return out
    return [int(x) for x in l]


def replay(dag, layout, swap_seq, budget):
    env = RecordingEnv(dag, hw, coupling_map, max_num_qubits=20,
                       init_mapping=list(layout), sabre_swap_budget=budget,
                       **ENV_KW)
    obs, _ = env.reset()
    # reset 内部（mapping_phase=False）已执行初始批量，返回值被 stash 捕获
    r0e, r0p = env._last_batch
    total = r0e + r0p
    env.acc['r_prop'] += r0p
    env.acc['r_sched_txi'] += r0e - env._batch_gate
    done = trunc = False
    for (p, q) in swap_seq:
        a = edge_index[(p, q)]
        g0 = env.acc['r_gate']
        s0 = env.acc['r_swap_price']
        sh0 = env.acc['r_shape']
        t0 = env.acc['r_terminal']
        p0 = env.acc['r_prop']
        _, r, done, trunc, info = env.step(a, compute_obs=False)
        batch_r, batch_prop = env._last_batch
        env.acc['r_sched_txi'] += batch_r - (env.acc['r_gate'] - g0)
        env.acc['r_prop'] += batch_prop - p0
        budget_step = (-LAMBDA_BUDGET
                       if (budget is not None
                           and env._swap_counter > budget) else 0.0)
        env.acc['r_budget'] += budget_step
        env.acc['r_xt_swap'] += r - (batch_r + (batch_prop - p0)
                                     + (env.acc['r_swap_price'] - s0)
                                     + budget_step + (env.acc['r_shape'] - sh0)
                                     + (env.acc['r_terminal'] - t0))
        total += r
        if done or trunc:
            break
    completed = len(env.executed) == dag.num_gates
    comp = dict(env.acc)
    comp['r_total'] = total
    comp['swaps'] = env._swap_counter
    comp['completed'] = completed
    comp['truncated'] = trunc
    comp['gates_exec'] = len(env.executed)
    comp['gates_total'] = dag.num_gates
    comp['makespan'] = float(env.timing.total_time) if env.timing else None
    comp['idle'] = float(env.timing.qubit_idle_time.sum()) if env.timing else None
    comp['xtalk'] = float(env.timing.crosstalk_events) if env.timing else None
    return comp, env


results = {'meta': {'lambda_budget': LAMBDA_BUDGET, 'budget_delta': BUDGET_DELTA},
           'circuits': {}}
if os.path.exists(OUT):
    old = json.load(open(OUT))
    results = old if 'circuits' in old else results

shared_gnn = SubGNN(subgraph='full')
shared_gnn.eval()

budget_cache = {}
v3_checks = []
n_done = 0
for fname in circuits:
    name = fname.removesuffix('.qasm')
    if name in results['circuits']:
        continue
    qc = qasm2_loads(open(os.path.join(VAL_DIR, fname)).read())
    dag = CircuitDAG.from_circuit(qc)
    entry = {}

    # ---- SABRE ----
    try:
        phys_s, info_s = sabre_route(qc, config, swap_trials=20, seed=0)
        layout_s = info_s.get('initial_layout')
        swap_seq_s = extract_swaps(phys_s)
        budget_s = int(np.ceil(BUDGET_DELTA * info_s['num_swaps'])) \
            if info_s['num_swaps'] > 0 else None
        comp, env_r = replay(dag, normalize_layout(layout_s), swap_seq_s, budget_s)
        comp['layout'] = list(layout_s) if layout_s is not None else None
        entry['SABRE'] = comp
    except Exception as e:
        import traceback
        print(f'[FAIL SABRE] {fname}: {e}', flush=True)
        traceback.print_exc()
        continue

    # ---- argmax ----
    try:
        d = json.load(open(f'benchmark/routed/la287_argmax_indist/{name}.json'))
        layout_m = d.get('initial_layout')
        phys_m = qasm2_loads(d['routed_qasm'].replace(
            'include "qelib1.inc";', 'include "qelib1.inc";\n' + SWAP_DEF))
        swap_seq_m = extract_swaps(phys_m)
        comp_m, env_m = replay(dag, normalize_layout(layout_m), swap_seq_m, budget_s)
        comp_m['layout'] = list(layout_m) if layout_m is not None else None
        entry['argmax'] = comp_m
    except Exception as e:
        import traceback
        print(f'[FAIL argmax] {fname}: {e}', flush=True)
        traceback.print_exc()
        continue

    # V3：重放 2Q 门多重集 ≡ routed_qasm（前 8 条电路抽检）。
    # 注：同波并行门的调度顺序跨进程可能不稳定（排序 tie），但同波门并行执行、
    # 边奖励与顺序无关 → 多重集等价即奖励语义等价；逐位序比较过严（会假阴性）。
    if len(v3_checks) < 8:
        match = Counter(two_q_seq(env_m._phys_circuit)) == Counter(two_q_seq(phys_m))
        v3_checks.append({'circuit': fname, 'replay_eq_routed': bool(match)})
        if not match:
            print(f'[V3 FAIL] {fname}: 重放 2Q 序列 ≠ routed_qasm', flush=True)

    # ---- beam5la+bud ----
    try:
        db = json.load(open(f'benchmark/routed/la287_beam5la_bud_indist/{name}.json'))
        phys_b = qasm2_loads(db['routed_qasm'].replace(
            'include "qelib1.inc";', 'include "qelib1.inc";\n' + SWAP_DEF))
        swap_seq_b = extract_swaps(phys_b)
        comp_b, env_b = replay(dag, normalize_layout(db.get('initial_layout')), swap_seq_b, budget_s)
        comp_b['layout'] = list(db.get('initial_layout') or [])
        entry['beam5la+bud'] = comp_b
    except Exception as e:
        import traceback
        print(f'[FAIL beam] {fname}: {e}', flush=True)
        traceback.print_exc()
        continue

    results['circuits'][name] = entry
    n_done += 1
    print(f'{name:<28s} SABRE R={comp["r_total"]:+8.2f} sw={comp["swaps"]:>3d} '
          f'| argmax R={comp_m["r_total"]:+8.2f} sw={comp_m["swaps"]:>3d} '
          f'| beam R={comp_b["r_total"]:+8.2f} sw={comp_b["swaps"]:>3d}', flush=True)
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=1)

results['v3_checks'] = v3_checks
with open(OUT, 'w') as f:
    json.dump(results, f, indent=1)
print(f'\nreplay 完成: {n_done} 条新算; V3 抽检: '
      f'{sum(1 for v in v3_checks if v["replay_eq_routed"])}/{len(v3_checks)} 通过')
print('saved:', OUT)
