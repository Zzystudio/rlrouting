"""barenco_tof_3 全流程案例分析：SABRE vs LA287 argmax vs LA287 beam5la。

流程：QASM → CircuitDAG → (SABRE 路由 | agent 路由追踪) → 事件级 v2 模拟器 →
保真度 + 噪声分解（2Q 边误差暴露 / SWAP 价 / 1Q / 热弛豫时长 / ZZ 串扰）。

用法: python3 scripts/analyze_tof3_case.py [--T 512]
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
from qiskit.qasm2 import loads as qasm2_loads

from routing.graph.circuit_dag import CircuitDAG
from routing.rl.agent import PPOAgent
from routing.rl.env import RoutingEnv
from routing.gnn.encoder import SubGNN
from routing.routing import sabre_route
from sim.trajectory_sim_v2 import (EventTrajectorySimulator,
                                   schedule_phys_circuit_events)

T_FID = int(sys.argv[sys.argv.index('--T') + 1]) if '--T' in sys.argv else 512
CIRC = 'barenco_tof_3'

from routing.rl.eval_policy import load_topo
config, hw, coupling_map = load_topo('traindata/topo/tianyan287_20q.json')
tqe = config.two_q_gate_error

qc = qasm2_loads(open(f'benchmark/nam_circs/{CIRC}.qasm').read())
dag = CircuitDAG.from_circuit(qc)
print(f'电路: {CIRC}  logical={qc.num_qubits}q  指令={len(qc.data)}  '
      f'2Q门={sum(1 for i in qc.data if len(i.qubits)==2)}')

# ---------------------------------------------------------------- SABRE
phys_s, info_s = sabre_route(qc, config, swap_trials=20, seed=0)
print(f'\n[SABRE] swaps={info_s["num_swaps"]}  '
      f'initial_layout={info_s.get("initial_layout")}')

# ---------------------------------------------------------------- agent 模型
shared_gnn = SubGNN(subgraph='full')
dummy_env = RoutingEnv(dag, hw, coupling_map, reward_mode='routing',
                       max_episode_steps=1000, seed=0, gnn=shared_gnn,
                       use_gnn=True, max_num_qubits=20,
                       max_num_edges=len(coupling_map), mapping_phase=True,
                       edge_noise_features=True, beta_noise=0.5)
probe = torch.load('models/policy_LA287.pt', map_location='cpu', weights_only=False)
has_la = any(k.startswith('critic_la') for k in probe.get('ac', {}))
agent = PPOAgent(
    obs_dim=int(np.prod(dummy_env.observation_space.shape)),
    action_dim=len(coupling_map) + 1, device='cuda:0', gnn=shared_gnn,
    num_qubits=20, num_edges=len(coupling_map), coupling_map=coupling_map,
    with_commit=True, edge_feat_dim=getattr(dummy_env, '_edge_feat_dim', None),
    with_la_head=has_la)
agent.load('models/policy_LA287.pt')
agent.ac.eval()
shared_gnn.eval()

ENV_KW = dict(reward_mode='routing', max_episode_steps=1000, seed=0,
              gnn=shared_gnn, use_gnn=True, noise_config=None,
              max_num_qubits=20, max_num_edges=len(coupling_map),
              mapping_phase=True, fidelity_fn=None, use_scheduler=False,
              swap_cost=0.0, eta_xtalk_par=1.0, lambda_fid=0.0,
              edge_noise_features=True, beta_noise=0.5,
              reward_potential=True, pot_progress_b=0.20,
              w_err=0.02, w_xt=0.01, w_xt_swap=0.02,
              shaping_gamma=0.99, eta_shape=0.3, alpha_ext=0.5)


def build_mask(env, agent):
    n_a = agent.num_edges + 1
    mask = torch.zeros(n_a, dtype=torch.bool, device=agent.device)
    mask[:len(env.coupling_map)] = True
    dm = env.get_deadlock_mask()
    um = env.get_unmapped_mask()
    for i in range(min(len(dm), n_a)):
        if dm[i] or um[i]:
            mask[i] = False
    mask[agent.num_edges] = env.mapping_phase
    return mask.unsqueeze(0)


def gate_desc(env, exec_prev):
    """本步新执行门的 (name, 物理边)。"""
    out = []
    for i in env.executed:
        if i not in exec_prev:
            g = env.dag.gates[i]
            if g.is_two_qubit:
                pa, pb = env.mapping[g.qubits[0]], env.mapping[g.qubits[1]]
                out.append((g.name, (pa, pb)))
            else:
                out.append((g.name, (env.mapping[g.qubits[0]],)))
    return out


def route_argmax_trace():
    env = RoutingEnv(dag, hw, coupling_map, **ENV_KW)
    obs, _ = env.reset()
    trace, done, trunc = [], False, False
    while not (done or trunc):
        with torch.no_grad():
            mask = build_mask(env, agent)
            logits, _ = agent._forward_obs(obs, action_mask=mask)
            masked = logits[0].clone()
            masked[~mask[0]] = -1e9
            top5 = masked.topk(min(5, int(mask[0].sum()))).indices.tolist()
            a = int(masked.argmax().item())
        snap = dict(phase='map' if env.mapping_phase else 'route',
                    step=env._episode_step, map_before=list(env.mapping),
                    exec_n=len(env.executed), swap_n=env._swap_counter,
                    action=a, top5=top5)
        obs, r, done, trunc, info = env.step(a)
        snap['edge'] = ('COMMIT' if snap['phase'] == 'map' and a >= agent.num_edges
                        else coupling_map[a])
        snap['gates'] = gate_desc(env, set(range(snap['exec_n'])))
        snap['r'] = round(float(r), 3)
        trace.append(snap)
    return trace, env


def route_beam_trace(beam_width=5):
    env = RoutingEnv(dag, hw, coupling_map, **ENV_KW)
    obs, _ = env.reset()
    trace, done, trunc = [], False, False
    gamma = agent.gamma
    while not (done or trunc):
        with torch.no_grad():
            mask = build_mask(env, agent)
            logits, _ = agent._forward_obs(obs, action_mask=mask)
            masked = logits[0].clone()
            masked[~mask[0]] = -1e9
            k = min(beam_width, int(mask[0].sum()))
            topk = masked.topk(k)
            snap = dict(phase='map' if env.mapping_phase else 'route',
                        step=env._episode_step, map_before=list(env.mapping),
                        exec_n=len(env.executed), swap_n=env._swap_counter,
                        action=None, top5=topk.indices.tolist(),
                        cands=[])
            clones = []
            for i in range(topk.indices.shape[0]):
                a = topk.indices[i].item()
                c = env.clone()
                _, rc, dc, tc, _ = c.step(a, compute_obs=False)
                clones.append((c, a, rc, dc, tc))
            gds = [c.build_graph_data() for c, *_ in clones]
            qhs = agent.gnn.node_embeddings_batched(gds)
            obs_list = [t[0]._obs(qubit_h=qh.cpu().numpy())
                        for t, qh in zip(clones, qhs)]
            _, vla = agent._forward_obs_batch_vla(np.stack(obs_list))
            best_a, best_s, best_c = None, -1e18, None
            for (c, a, rc, dc, tc), o, v in zip(clones, obs_list, vla):
                s = rc if (dc or tc) else rc + gamma * float(v)
                snap['cands'].append((int(a), coupling_map[a] if a < agent.num_edges
                                      else 'COMMIT', round(float(rc), 3),
                                      round(float(v), 3), round(float(s), 3)))
                if s > best_s:
                    best_a, best_s, best_c = a, s, o
            obs, r, done, trunc, info = env.step(best_a, compute_obs=False)
            obs = best_c
        snap['action'] = best_a
        snap['edge'] = ('COMMIT' if snap['phase'] == 'map'
                        and best_a >= agent.num_edges else coupling_map[best_a])
        snap['gates'] = gate_desc(env, set(range(snap['exec_n'])))
        snap['r'] = round(float(r), 3)
        trace.append(snap)
    return trace, env


print('\n[agent] argmax 追踪中...')
trace_a, env_a = route_argmax_trace()
print(f'[agent] argmax: steps={len(trace_a)} swaps={env_a._swap_counter} '
      f'map_swaps={env_a._mapping_swaps}')
print('\n[agent] beam5(la) 追踪中...')
trace_b, env_b = route_beam_trace(5)
print(f'[agent] beam5la: steps={len(trace_b)} swaps={env_b._swap_counter} '
      f'map_swaps={env_b._mapping_swaps}')

# 布局（commit 后 mapping：logical -> physical）
print('\n[布局] logical->physical')
print('  SABRE  :', info_s.get('initial_layout'))
print('  argmax :', env_a._effective_initial_mapping)
print('  beam5la:', env_b._effective_initial_mapping)

# ---------------------------------------------------------------- 噪声分解
def theta(a, b):
    if config.crosstalk_strength is not None:
        return float(config.crosstalk_strength.get((a, b),
                      config.crosstalk_strength.get((b, a), 0.0)))
    e = tqe
    base = e.get((a, b), e.get((b, a), 0.001)) if isinstance(e, dict) else float(e)
    return 0.1 * base


def diag(phys, name):
    n_swap = n_2q = 0
    err_2q = err_swap = 0.0
    edge_2q = {}
    n_1q = 0
    err_1q = 0.0
    for inst in phys.data:
        op = inst.operation
        nm = op.name.lower()
        qs = [phys.find_bit(q).index for q in inst.qubits]
        if nm in ('measure', 'barrier'):
            continue
        if len(qs) == 2:
            e = float(tqe.get((qs[0], qs[1]), tqe.get((qs[1], qs[0]), 0.001))
                      if isinstance(tqe, dict) else tqe)
            if nm == 'swap':
                n_swap += 1
                err_swap += 3.0 * e
            else:
                n_2q += 1
                err_2q += e
            edge_2q[(min(qs), max(qs))] = edge_2q.get((min(qs), max(qs)), 0) + 1
        else:
            n_1q += 1
            e1 = config.single_q_gate_error
            err_1q += float(e1[qs[0]]) if isinstance(e1, (list, tuple)) else float(e1)
    events = schedule_phys_circuit_events(phys)
    total_t = max((e[1] for e in events), default=0.0)
    busy = [0.0] * phys.num_qubits
    for (s, e2, op, qs, _i) in events:
        if e2 > s:
            for q in qs:
                busy[q] += e2 - s
    idle = sum(max(0.0, total_t - b) for b in busy)
    zz_static = sum(theta(*qs) * (e2 - s) / max(config.two_gate_time, 1e-9)
                    for (s, e2, op, qs, _i) in events
                    if len(qs) == 2 and op in ('cx', 'cz', 'swap') and e2 > s)
    sim = EventTrajectorySimulator(config, num_trajectories=T_FID, seed=42,
                                   backend='auto')
    actions, _tot = sim._prepare_events(phys, events)
    zz_dyn = sum(a[4] for a in actions if a[0] == 'zz')
    fid = sim.fidelity_events(phys, events)
    top_edges = sorted(edge_2q.items(), key=lambda x: -x[1])[:6]
    print(f'\n[{name}] swaps={n_swap} 2Q门={n_2q} 1Q门={n_1q}  '
          f'makespan={total_t:.2f}us  idle={idle:.1f}q·us')
    print(f'  误差暴露: 2Q门 Σe={err_2q:.4f}  SWAP 3Σe={err_swap:.4f}  '
          f'1Q Σe={err_1q:.4f}  | 合计={err_2q + err_swap + err_1q:.4f}')
    print(f'  ZZ串扰: 静态={zz_static:.4f} rad  动态(重叠)={zz_dyn:.4f} rad')
    print(f'  高频边: ' + ', '.join(f'({a},{b})×{n}' for (a, b), n in top_edges))
    print(f'  v2 fidelity (T={T_FID}): {fid:.4f}')
    return dict(swaps=n_swap, n_2q=n_2q, err_2q=err_2q, err_swap=err_swap,
                err_1q=err_1q, total_t=total_t, idle=idle,
                zz_static=zz_static, zz_dyn=zz_dyn, fid=fid)


print('\n=============== 噪声分解对比 ===============')
d_s = diag(phys_s, 'SABRE')
phys_a = env_a._phys_circuit
d_a = diag(phys_a, 'LA287 argmax')
phys_b = env_b._phys_circuit
d_b = diag(phys_b, 'LA287 beam5la')

print('\n=============== 追踪摘要 ===============')
for tag, tr in (('argmax', trace_a), ('beam5la', trace_b)):
    print(f'\n--- {tag} ({len(tr)} 步) ---')
    for s in tr:
        g = ' '.join(f'{n}{e}' for n, e in s['gates']) or '-'
        if s['phase'] == 'map':
            act = 'COMMIT' if s['edge'] == 'COMMIT' else f'vswap{s["edge"]}'
            print(f"  s{s['step']:>3d} [map] {act:<12s} map={s['map_before']}")
        else:
            extra = ''
            if 'cands' in s:
                extra = ' cands=' + ' '.join(
                    f'{e}:{s_:.2f}' for _a, e, _r, _v, s_ in s['cands'])
            print(f"  s{s['step']:>3d} [route] swap{str(s['edge']):<10s} "
                  f"r={s['r']:+.2f} gates=[{g}]{extra}")

# argmax 的 top5 里是否包含 beam 的选择（前瞻 vs 策略责任分析）
print('\n=============== argmax vs beam 分歧点 ===============')
for i, (sa, sb) in enumerate(zip(trace_a, trace_b)):
    if sa['action'] != sb['action']:
        in_top5 = sb['action'] in sa['top5']
        print(f'  步 {i} (phase={sa["phase"]}): argmax 选 {sa["edge"]} '
              f'top5={sa["top5"]} | beam 选 {sb["edge"]}（在 argmax top5 中: {in_top5}）')
        if len([1 for j, (x, y) in enumerate(zip(trace_a, trace_b))
                if x['action'] != y['action']]) > 12:
            break
print('\nsaved: 控制台输出（分析文本见 doc/案例分析_barenco_tof_3.md）')
