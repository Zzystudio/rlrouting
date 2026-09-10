"""§11 诊断：fidelity 是否存在 action-level causal signal？

对每个测试电路，在 l05 路由轨迹上的若干中间状态 s_t，对 top-4 候选 SWAP 各执行一次
（clone → step），然后用同一策略继续完成剩余路由，计算最终 F(a_i)；
ΔF(a_i) = F(s, a_i) − F(s, a_0)（相对 argmax 动作）。

统计 ρ(ΔF(a_i), F_final(a_i)) 以及"ΔF 是否预测最终 fidelity 排序"。
若 ρ>0 → fidelity 存在 action-level causal signal；若 ρ≈0 → 是 trajectory-level objective。
"""
import argparse, json, os, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from qiskit.qasm2 import load as qasm2_load
from routing.graph.circuit_dag import CircuitDAG
from routing.gnn.encoder import SubGNN
from routing.rl.agent import PPOAgent
from routing.rl.env import RoutingEnv
from routing.rl.eval_policy import load_topo
from sim.trajectory_sim import trajectory_circuit_fidelity


def roll_complete(env, agent, max_steps=400):
    """从当前 env 状态用 agent argmax 路由至完成，返回 (done, env)。"""
    done = False
    truncated = False
    n = 0
    while not done and not truncated and n < max_steps:
        obs = env._obs()
        with torch.no_grad():
            mask = torch.zeros(agent.num_edges + 1, dtype=torch.bool)
            mask[:len(env.coupling_map)] = True
            dm = env.get_deadlock_mask()
            um = env.get_unmapped_mask()
            for i in range(min(len(dm), len(mask))):
                if dm[i] or um[i]:
                    mask[i] = False
            if agent.with_commit:
                mask[agent.num_edges] = env.mapping_phase
            logits, _ = agent._forward_obs(obs, action_mask=mask.unsqueeze(0))
            act = logits.argmax(-1).item()
        _, _, done, truncated, _ = env.step(act)
        n += 1
    return done, env


def compute_fid(env, config, traj=16):
    from sim.trajectory_sim import _reduce_phys_circuit_for_fidelity
    rc, rcfg = _reduce_phys_circuit_for_fidelity(env._phys_circuit, config)
    try:
        return trajectory_circuit_fidelity(rc, rcfg, num_trajectories=traj,
                                           seed=0, scheduled=True)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--circuit-dir", default="../benchmark/nam_circs")
    ap.add_argument("--topo", default="../traindata/topo/tianyan176_20q.json")
    ap.add_argument("--max-circuits", type=int, default=6)
    ap.add_argument("--max-sample-states", type=int, default=8,
                    help="每电路采样的中间状态数")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--traj", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config, hw, cm = load_topo(args.topo)
    gnn = SubGNN(subgraph='full'); gnn.eval()
    agent = PPOAgent(obs_dim=1, action_dim=len(cm) + 1, device='cpu', gnn=gnn,
                     num_qubits=20, num_edges=len(cm), coupling_map=cm,
                     with_commit=True)
    agent.load(args.model)
    agent.ac.eval(); agent.gnn.eval()

    rng = np.random.default_rng(args.seed)
    qasm_files = sorted(f for f in os.listdir(args.circuit_dir)
                        if f.endswith('.qasm'))
    if args.max_circuits:
        qasm_files = qasm_files[:args.max_circuits]

    all_rows = []
    for fname in qasm_files:
        qc = qasm2_load(os.path.join(args.circuit_dir, fname))
        dag = CircuitDAG.from_circuit(qc)
        # 先完整跑一次 argmax 轨迹记录中间状态
        env = RoutingEnv(dag, hw, cm, reward_mode='routing', max_episode_steps=500,
                         seed=int(rng.integers(0, 1 << 30)), gnn=agent.gnn,
                         use_gnn=True, max_num_qubits=20, mapping_phase=True)
        env.reset()
        states = []
        done = truncated = False
        n = 0
        while not done and not truncated and n < 500:
            obs = env._obs()
            with torch.no_grad():
                mask = torch.zeros(agent.num_edges + 1, dtype=torch.bool)
                mask[:len(cm)] = True
                dm = env.get_deadlock_mask()
                um = env.get_unmapped_mask()
                for i in range(min(len(dm), len(mask))):
                    if dm[i] or um[i]:
                        mask[i] = False
                if agent.with_commit:
                    mask[agent.num_edges] = env.mapping_phase
                logits, _ = agent._forward_obs(obs, action_mask=mask.unsqueeze(0))
                masked = logits[0].clone()
                masked[~mask[0]] = -1e9
                k = min(args.top_k, mask.sum().item())
                topk = masked.topk(k)
            if not env.mapping_phase and len(states) < args.max_sample_states:
                states.append((env.clone(), obs, topk.indices.tolist()))
            act = topk.indices[0].item()
            _, _, done, truncated, _ = env.step(act)
            n += 1
        if not done:
            print(f"  [skip] {fname}: 主轨迹未完成")
            continue

        n_states = 0
        for si, (st_env, obs, cands) in enumerate(states):
            fid_vals = []
            for ai, a in enumerate(cands):
                c = st_env.clone()
                _, _, d_c, t_c, _ = c.step(a, compute_obs=False)
                d2, c2 = roll_complete(c, agent)
                if not d2:
                    print(f"    [{fname} state{si} act{a}] roll 未完成，跳过")
                    fid_vals.append(None)
                else:
                    try:
                        fid_vals.append(compute_fid(c2, config, traj=args.traj))
                    except Exception as e:
                        print(f"    [{fname} state{si} act{a}] fid 异常: {e}")
                        fid_vals.append(None)
            if any(f is None for f in fid_vals) or fid_vals[0] is None:
                continue
            base = fid_vals[0]  # argmax 动作的最终 F
            for ai in range(1, len(fid_vals)):
                df = fid_vals[ai] - fid_vals[0]
                ff = fid_vals[ai]
                all_rows.append({"circuit": fname, "state": si,
                                 "dF": df, "F_final": ff})
            n_states += 1
        print(f"  {fname}: sampled {n_states} states, "
              f"rows={sum(1 for r in all_rows if r['circuit'] == fname)}")

    if not all_rows:
        print("无有效数据")
        return
    from scipy.stats import spearmanr
    dF = np.array([r["dF"] for r in all_rows])
    FF = np.array([r["F_final"] for r in all_rows])
    sp, p = spearmanr(dF, FF)
    print(f"\n=== 诊断结果（{len(all_rows)} 个 (state, action) 样本）===")
    print(f"Spearman ρ(ΔF(a_i), F_final(a_i)) = {sp:.3f}  (p={p:.4f})")
    print(f"ΔF>0 且 F_final 更高的比例: "
          f"{np.mean((dF > 0) == (FF > np.median(FF))):.3f}")
    # 按电路细分
    for fname in sorted({r['circuit'] for r in all_rows}):
        sub = [(r["dF"], r["F_final"]) for r in all_rows if r["circuit"] == fname]
        if len(sub) >= 6:
            s, _ = spearmanr([x[0] for x in sub], [x[1] for x in sub])
            print(f"  {fname}: n={len(sub)} ρ={s:.3f}")


if __name__ == "__main__":
    main()
