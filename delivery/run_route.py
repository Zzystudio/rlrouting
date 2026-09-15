#!/usr/bin/env python3
"""CqRouting 推理入口：对 OpenQASM 2.0 线路执行噪声感知路由。

用法示例（在 delivery/ 目录下运行）：
  # argmax 推理
  python run_route.py --circuit examples/nam_circs/tof_3.qasm \
      --topo topologies/tianyan176_20q.json --out routed/tof_3

  # beam search（推荐；R3b_beam5 即该模式）
  python run_route.py --circuit examples/nam_circs/tof_3.qasm \
      --topo topologies/tianyan176_20q.json --beam 5 --out routed/tof_3

  # 附带 v2 事件级模拟器保真度估计（T 条轨迹，随电路深度耗时增长）
  python run_route.py --circuit examples/toy/ghz_5.qasm \
      --topo topologies/tianyan176_20q.json --fidelity 16 --out routed/ghz_5

  # SABRE 基线对照（同拓扑、同参数）
  python run_route.py --circuit examples/nam_circs/tof_3.qasm \
      --topo topologies/tianyan176_20q.json --baseline sabre --out routed/tof_3

输出：<out>.qasm（路由后物理线路）与 <out>.json（指标汇总）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import torch
from qiskit.qasm2 import load as qasm2_load, dumps as qasm2_dumps

from routing.graph.circuit_dag import CircuitDAG
from routing.graph.features import HardwareFeatures
from routing.gnn.encoder import SubGNN
from routing.rl.agent import PPOAgent
from routing.rl.env import RoutingEnv
from routing.routing import sabre_route
from routing.rl.eval_policy import load_topo


# ---------------------------------------------------------------------------
# 模型档案：按 checkpoint 的 per-edge 特征维度自动识别
#   149 = out(48)*3 + SABRE5            → R3b（本次交付的最佳模型）
#   153 = 149 + look4                   → R5a 观测版
#   158 = 149 + look4 + noise5          → P0（tianyan287 实验版）
# ---------------------------------------------------------------------------
def detect_profile(ckpt_path: str) -> dict:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ac_state = state.get("ac", state)
    w = ac_state.get("edge_mlp.0.weight")
    if w is None:
        raise ValueError("checkpoint 中未找到 edge_mlp.0.weight（非 EdgeActorCritic 模型）")
    dim = int(w.shape[1])
    base = 48 * 3 + 5
    if dim == base:
        return {"profile": "r3b", "lookahead_features": False,
                "edge_noise_features": False, "beta_noise": 0.0,
                "w_err": 0.0, "w_xt": 0.0, "w_xt_swap": 0.0,
                "pot_progress_b": 0.045, "pot_1q_reward": True}
    if dim == base + 4:
        return {"profile": "r5a", "lookahead_features": True,
                "edge_noise_features": False, "beta_noise": 0.0,
                "w_err": 0.0, "w_xt": 0.0, "w_xt_swap": 0.0,
                "pot_progress_b": 0.045, "pot_1q_reward": True}
    if dim == base + 9:
        return {"profile": "p0", "lookahead_features": True,
                "edge_noise_features": True, "beta_noise": 0.5,
                "w_err": 0.02, "w_xt": 0.01, "w_xt_swap": 0.02,
                "pot_progress_b": 0.20, "pot_1q_reward": False}
    raise ValueError(f"未知 per-edge 特征维度 {dim}（期望 {base}/{base+4}/{base+9}）")


# ---------------------------------------------------------------------------
# 路由核心
# ---------------------------------------------------------------------------
def route_with_model(qc, config, hw, coupling_map, model_path,
                     beam_width=0, max_episode_steps=1000, device="cpu",
                     seed=0):
    """用交付模型路由一条逻辑电路，返回 (物理电路, 指标 dict)。"""
    prof = detect_profile(model_path)
    dag = CircuitDAG.from_circuit(qc)
    shared_gnn = SubGNN(subgraph="full")
    shared_gnn.eval()

    probe_env = RoutingEnv(
        dag, hw, coupling_map, reward_mode="routing", max_episode_steps=1,
        seed=seed, gnn=shared_gnn, use_gnn=True,
        max_num_qubits=hw.num_qubits,
        max_num_edges=len(coupling_map),
        mapping_phase=True,
        lookahead_features=prof["lookahead_features"],
        edge_noise_features=prof["edge_noise_features"],
        beta_noise=prof["beta_noise"],
    )
    agent = PPOAgent(
        obs_dim=int(np.prod(probe_env.observation_space.shape)),
        action_dim=len(coupling_map) + 1,
        device=device,
        gnn=shared_gnn,
        num_qubits=hw.num_qubits,
        num_edges=len(coupling_map),
        coupling_map=coupling_map,
        with_commit=True,
        edge_feat_dim=getattr(probe_env, "_edge_feat_dim", None),
    )
    agent.load(model_path)
    agent.ac.eval()
    if agent.gnn is not None:
        agent.gnn.eval()

    # beam 打分的奖励口径须与训练一致（R3b: potential + Φ shaping）
    env = RoutingEnv(
        dag, hw, coupling_map, reward_mode="routing",
        max_episode_steps=max_episode_steps, seed=seed,
        gnn=agent.gnn, use_gnn=True,
        max_num_qubits=hw.num_qubits,
        max_num_edges=len(coupling_map),
        mapping_phase=True,
        lookahead_features=prof["lookahead_features"],
        edge_noise_features=prof["edge_noise_features"],
        beta_noise=prof["beta_noise"],
        w_err=prof["w_err"], w_xt=prof["w_xt"], w_xt_swap=prof["w_xt_swap"],
        pot_progress_b=prof["pot_progress_b"],
        pot_1q_reward=prof["pot_1q_reward"],
        reward_potential=True,
        shaping_gamma=0.99, eta_shape=0.3, alpha_ext=0.5,
    )
    obs, _ = env.reset()
    t0 = time.perf_counter()

    if beam_width > 0:
        done, truncated = False, False
        while not done and not truncated:
            with torch.no_grad():
                n_a = agent.num_edges + 1
                mask = torch.zeros(n_a, dtype=torch.bool, device=agent.device)
                mask[:len(coupling_map)] = True
                dm = env.get_deadlock_mask()
                um = env.get_unmapped_mask()
                for i in range(min(len(dm), n_a)):
                    if dm[i] or um[i]:
                        mask[i] = False
                mask[agent.num_edges] = env.mapping_phase
                mask = mask.unsqueeze(0)
                logits, _ = agent._forward_obs(obs, action_mask=mask)
                ml = logits[0].clone()
                ml[~mask[0]] = -1e9
                k = min(beam_width, int(mask[0].sum().item()))
                if k == 0:  # 全部动作被掩码（病态状态）：安全退出
                    truncated = True
                    break
                _, topk = ml.topk(k)
                clones = []
                for a in topk.tolist():
                    c = env.clone()
                    _, rc, dc, tc, _ = c.step(a, compute_obs=False)
                    clones.append((c, a, rc, dc, tc))
                gds = [c.build_graph_data() for c, *_ in clones]
                qhs = agent.gnn.node_embeddings_batched(gds)
                cobs = [cc._obs(qubit_h=qh.cpu().numpy())
                        for (cc, *_), qh in zip(clones, qhs)]
                _, values = agent._forward_obs_batch(np.stack(cobs))
                best_a, best_i = None, -float("inf")
                for i, ((c, a, rc, dc, tc), v) in enumerate(zip(clones, values)):
                    score = rc if (dc or tc) else rc + agent.gamma * v.item()
                    if score > best_i:
                        best_i, best_a = score, a
                        best_obs = cobs[i]
        # beam 胜出 clone 即最终环境
        obs, reward, done, truncated, info = env.step(best_a, compute_obs=False)
        obs = best_obs
    else:
        done, truncated = False, False
        while not done and not truncated:
            with torch.no_grad():
                n_a = agent.num_edges + 1
                mask = torch.zeros(n_a, dtype=torch.bool, device=agent.device)
                mask[:len(coupling_map)] = True
                dm = env.get_deadlock_mask()
                um = env.get_unmapped_mask()
                for i in range(min(len(dm), n_a)):
                    if dm[i] or um[i]:
                        mask[i] = False
                mask[agent.num_edges] = env.mapping_phase
                mask = mask.unsqueeze(0)
                if int(mask[0].sum().item()) == 0:  # 病态状态安全退出
                    truncated = True
                    break
                logits, _ = agent._forward_obs(obs, action_mask=mask)
                action = logits.argmax(-1).item()
            obs, reward, done, truncated, info = env.step(action)

    wall_ms = (time.perf_counter() - t0) * 1000
    metrics = {
        "completed": bool(done),
        "num_swaps": int(env._swap_counter),
        "mapping_swaps": int(env._mapping_swaps),
        "wall_time_ms": round(wall_ms, 1),
        "profile": prof["profile"],
        "beam_width": beam_width,
        "initial_mapping": list(env._effective_initial_mapping),
        "final_mapping": list(env.mapping),
    }
    return env._phys_circuit, metrics


def route_with_sabre(qc, config, seed=0):
    phys, info = sabre_route(qc, config, seed=seed)
    metrics = {"completed": True, "num_swaps": int(info["num_swaps"]),
               "mapping_swaps": 0, "profile": "sabre", "beam_width": 0}
    return phys, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit", required=True, help="输入 OpenQASM 2.0 电路文件")
    ap.add_argument("--topo", default=os.path.join(ROOT, "topologies",
                                                   "tianyan176_20q.json"),
                    help="拓扑 JSON（默认 tianyan176_20q）")
    ap.add_argument("--out", default="routed/out",
                    help="输出前缀：<out>.qasm 与 <out>.json")
    ap.add_argument("--model", default=os.path.join(ROOT, "models",
                                                    "policy_r3b.pt"))
    ap.add_argument("--beam", type=int, default=0,
                    help="beam search 宽度（0=argmax；推荐 5，即 R3b_beam5）")
    ap.add_argument("--baseline", choices=["none", "sabre"], default="none",
                    help="路由后端：none=交付模型，sabre=基线对照")
    ap.add_argument("--fidelity", type=int, default=0,
                    help="路由完成后用 v2 事件级模拟器估计保真度（T 条轨迹，0=不算）")
    ap.add_argument("--max-episode-steps", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config, hw, coupling_map = load_topo(args.topo)
    qc = qasm2_load(args.circuit)

    if args.baseline == "sabre":
        phys, metrics = route_with_sabre(qc, config, seed=args.seed)
    else:
        phys, metrics = route_with_model(
            qc, config, hw, coupling_map, args.model,
            beam_width=args.beam, max_episode_steps=args.max_episode_steps,
            device=args.device, seed=args.seed)

    if args.fidelity > 0 and metrics["completed"]:
        from sim.trajectory_sim_v2 import trajectory_circuit_fidelity_events
        t0 = time.perf_counter()
        fid = trajectory_circuit_fidelity_events(
            phys, config, num_trajectories=args.fidelity, seed=0)
        metrics["fidelity_v2"] = round(float(fid), 6)
        metrics["fidelity_trajectories"] = args.fidelity
        metrics["fidelity_sec"] = round(time.perf_counter() - t0, 1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out + ".qasm", "w") as f:
        f.write(qasm2_dumps(phys))
    metrics["circuit"] = os.path.basename(args.circuit)
    metrics["topology"] = os.path.basename(args.topo)
    with open(args.out + ".json", "w") as f:
        json.dump(metrics, f, indent=1, ensure_ascii=False)

    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("initial_mapping", "final_mapping")},
                     indent=1, ensure_ascii=False))
    if not metrics["completed"]:
        print("[警告] episode 达到步数上限（TRUNC）：该电路对当前模型可能属于"
              "分布外难例，建议改用 --beam 5 重试。", file=sys.stderr)


if __name__ == "__main__":
    main()
