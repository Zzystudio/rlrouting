"""§11 诊断（简化版）：fidelity 是否存在 action-level causal signal？

对每条电路用 l05 argmax 路由，在中间状态对 top-K 候选 SWAP 各克隆+执行一次，
然后用同一策略完成剩余路由，计算最终 F(a_i)；ΔF = F(a_i) - F(a_argmax)。
统计 Spearman ρ(ΔF, F_final)。
"""
import os, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from qiskit.qasm2 import load as qasm2_load
from routing.graph.circuit_dag import CircuitDAG
from routing.gnn.encoder import SubGNN
from routing.rl.agent import PPOAgent
from routing.rl.env import RoutingEnv
from routing.rl.eval_policy import load_topo
from sim.trajectory_sim import trajectory_circuit_fidelity, _reduce_phys_circuit_for_fidelity


def build_mask(env, agent):
    mask = torch.zeros(agent.num_edges + 1, dtype=torch.bool)
    mask[:len(env.coupling_map)] = True
    dm = env.get_deadlock_mask()
    um = env.get_unmapped_mask()
    for i in range(min(len(dm), len(mask))):
        if dm[i] or um[i]:
            mask[i] = False
    if agent.with_commit:
        mask[agent.num_edges] = env.mapping_phase
    return mask


def roll(env, agent, max_steps=600):
    done = truncated = False
    n = 0
    while not done and not truncated and n < max_steps:
        obs = env._obs()
        with torch.no_grad():
            mask = build_mask(env, agent)
            logits, _ = agent._forward_obs(obs, action_mask=mask.unsqueeze(0))
            act = logits.argmax(-1).item()
        _, _, done, truncated, _ = env.step(act)
        n += 1
    return done


def fid_of(env, config, traj):
    rc, rcfg = _reduce_phys_circuit_for_fidelity(env._phys_circuit, config)
    return trajectory_circuit_fidelity(rc, rcfg, num_trajectories=traj,
                                       seed=0, scheduled=True)


def main():
    config, hw, cm = load_topo('traindata/topo/tianyan176_20q.json')
    gnn = SubGNN(subgraph='full'); gnn.eval()
    agent = PPOAgent(obs_dim=1, action_dim=len(cm) + 1, device='cpu', gnn=gnn,
                     num_qubits=20, num_edges=len(cm), coupling_map=cm,
                     with_commit=True)
    agent.load('models/policy_tianyan20q_laymix_l05_eta05.pt')
    agent.ac.eval(); agent.gnn.eval()

    rows = []
    for fname in ['tof_3.qasm', 'tof_4.qasm', 'tof_5.qasm', 'mod5_4.qasm',
                  'barenco_tof_4.qasm', 'barenco_tof_5.qasm']:
        qc = qasm2_load(f'benchmark/nam_circs/{fname}')
        dag = CircuitDAG.from_circuit(qc)
        env = RoutingEnv(dag, hw, cm, reward_mode='routing', max_episode_steps=600,
                         seed=7, gnn=agent.gnn, use_gnn=True,
                         max_num_qubits=20, mapping_phase=True)
        env.reset()
        n_states = 0
        done = truncated = False
        n = 0
        while not done and not truncated and n < 600 and n_states < 8:
            obs = env._obs()
            with torch.no_grad():
                mask = build_mask(env, agent)
                logits, _ = agent._forward_obs(obs, action_mask=mask.unsqueeze(0))
                masked = logits[0].clone()
                masked[~mask[0]] = -1e9
                k = min(4, mask.sum().item())
                topk = masked.topk(k)
            if not env.mapping_phase:
                k = min(4, mask.sum().item())
                if k >= 2:
                    cands = topk.indices.tolist()
                    fids = []
                    for a in cands:
                        c = env.clone()
                        _, _, _, _, _ = c.step(a)
                        ok = roll(c, agent)
                        f = fid_of(c, config, 8) if ok else None
                        fids.append(f)
                        sys.stdout.write(f"    {fname} state{n_states} act{a}: "
                                         f"done={ok} fid={f if f is None else round(f,4)}\n")
                        sys.stdout.flush()
                    if all(f is not None for f in fids):
                        base = fids[0]
                        for i in range(1, len(fids)):
                            rows.append((fname, fids[i] - base, fids[i]))
                        n_states += 1
            act = topk.indices[0].item() if k >= 1 else None
            if act is None:
                _, _, done, truncated, _ = env.step(0)
                n += 1
                continue
            _, _, done, truncated, _ = env.step(act)
            n += 1
        print(f"{fname}: {n_states} states sampled, total rows={len(rows)}")
        sys.stdout.flush()

    if len(rows) < 5:
        print("样本不足，无法统计")
        return
    from scipy.stats import spearmanr
    dF = np.array([r[1] for r in rows])
    FF = np.array([r[2] for r in rows])
    sp, p = spearmanr(dF, FF)
    print(f"\n=== 诊断结果（{len(rows)} 个 (state, action) 样本）===")
    print(f"Spearman ρ(ΔF(a_i), F_final(a_i)) = {sp:.3f} (p={p:.4f})")


if __name__ == "__main__":
    main()
