# ============================================================================
# train_agent.py
# 训练路由策略（PPO）。需要先训练好保真度预测器（Multi-GNN）。
#
# 用法:
#   python -m routing.rl.train_agent \
#       --predictor models/predictor.pt \
#       --out models/policy.pt --timesteps 20000
# ============================================================================

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sim.sim import NoiseConfig
from routing.graph.circuit_dag import CircuitDAG
from routing.graph.features import HardwareFeatures
from routing.gnn.predictor import MultiGNNTidelityPredictor
from routing.rl.env import RoutingEnv
from routing.rl.agent import PPOAgent
from utils.data_gen import random_circuit


def build_default_config(num_qubits: int = 5) -> NoiseConfig:
    coupling = [(i, i + 1) for i in range(num_qubits - 1)]
    return NoiseConfig(
        t1_times=[50.0] * num_qubits,
        t2_times=[70.0] * num_qubits,
        freq_ghz=[5.0] * num_qubits,
        single_q_gate_error=0.001,
        two_q_gate_error=0.01,
        coupling_map=coupling,
        readout_error=[0.02] * num_qubits,
        shots=1024,
    )


def load_predictor(path: str, device: str):
    model = MultiGNNTidelityPredictor()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Train routing policy with PPO")
    parser.add_argument("--predictor", type=str, required=True)
    parser.add_argument("--out", type=str, default="models/policy.pt")
    parser.add_argument("--num-qubits", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = build_default_config(args.num_qubits)
    hw = HardwareFeatures.from_noise_config(config)
    predictor = load_predictor(args.predictor, args.device)

    env = RoutingEnv(
        dag=CircuitDAG.from_circuit(random_circuit(args.num_qubits, 6, seed=args.seed)),
        hw=hw,
        coupling_map=list(config.coupling_map),
        predictor=predictor,
        seed=args.seed,
    )

    agent = PPOAgent(
        obs_dim=int(np.prod(env.observation_space.shape)),
        action_dim=int(env.action_space.n),
        lr=args.lr,
        device=args.device,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    obs, _ = env.reset()
    ep_rewards = []
    ep_fids = []
    buffer = {"obs": [], "act": [], "logp": [], "val": [], "rew": [], "done": []}

    total_steps = 0
    best_fid = -1.0
    while total_steps < args.timesteps:
        for _ in range(args.rollout_steps):
            action, logp, val = agent.act(obs)
            next_obs, reward, done, _, info = env.step(action)
            buffer["obs"].append(obs)
            buffer["act"].append(action)
            buffer["logp"].append(logp)
            buffer["val"].append(val)
            buffer["rew"].append(reward)
            buffer["done"].append(done)
            obs = next_obs
            total_steps += 1
            if done:
                ep_rewards.append(info.get("episode_reward", reward))
                if "fidelity" in info and info["fidelity"] is not None:
                    ep_fids.append(info["fidelity"])
                obs, _ = env.reset()

        # 计算 GAE（用最后一个状态的 value 作 bootstrap）
        with torch.no_grad():
            last_val = agent.model(
                torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            )[1].item()
        adv, ret = PPOAgent.compute_gae(
            buffer["rew"], buffer["val"], buffer["done"],
            bootstrap=last_val, gamma=agent.gamma, lam=agent.lam,
        )
        train_batch = {
            "obs": buffer["obs"],
            "act": buffer["act"],
            "logp": buffer["logp"],
            "adv": adv,
            "ret": ret,
        }
        agent.update(train_batch, epochs=args.epochs)
        buffer = {k: [] for k in buffer}

        avg_fid = np.mean(ep_fids[-20:]) if ep_fids else 0.0
        print(f"step={total_steps}  avg_fid(近20)={avg_fid:.3f}  "
              f"avg_ep_reward={np.mean(ep_rewards[-20:]) if ep_rewards else 0:.2f}")
        if avg_fid > best_fid:
            best_fid = avg_fid
            agent.save(args.out)
    print(f"策略已保存至 {args.out}")


if __name__ == "__main__":
    main()
