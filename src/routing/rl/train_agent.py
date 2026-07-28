from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.sim import NoiseConfig
from routing.graph.circuit_dag import CircuitDAG
from routing.graph.features import HardwareFeatures
from routing.rl.env import RoutingEnv
from routing.rl.agent import PPOAgent
from routing.gnn.encoder import SubGNN


# ---------------------------------------------------------------------------
#  Hardware config from JSON
# ---------------------------------------------------------------------------
def load_topo(path: str) -> tuple:
    with open(path) as f:
        topo = json.load(f)
    coupling_map = [tuple(e) for e in topo["coupling_map"]]
    dp = topo["device_params"]
    config = NoiseConfig(
        t1_times=dp["t1_times"],
        t2_times=dp["t2_times"],
        freq_ghz=dp["freq_ghz"],
        single_q_gate_error=dp["single_q_gate_error"],
        two_q_gate_error=dp["two_q_gate_error"],
        coupling_map=coupling_map,
        readout_error=dp["readout_error"],
        shots=dp.get("shots", 1024),
        crosstalk_strength=topo.get("crosstalk_strength"),
    )
    hw = HardwareFeatures.from_noise_config(config)
    return config, hw, coupling_map


def load_topo_or_default(num_qubits: int = 5, topo_path: str = None) -> tuple:
    if topo_path:
        return load_topo(topo_path)
    coupling = [(i, i + 1) for i in range(num_qubits - 1)]
    config = NoiseConfig(
        t1_times=[50.0] * num_qubits,
        t2_times=[70.0] * num_qubits,
        freq_ghz=[5.0] * num_qubits,
        single_q_gate_error=0.001,
        two_q_gate_error=0.01,
        coupling_map=coupling,
        readout_error=[0.02] * num_qubits,
        shots=1024,
    )
    hw = HardwareFeatures.from_noise_config(config)
    return config, hw, list(config.coupling_map)


# ---------------------------------------------------------------------------
#  Dataset loader
# ---------------------------------------------------------------------------
def load_split(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def build_split_paths(data_dir: str) -> dict:
    return {
        "stage1_phase1": os.path.join(data_dir, "splits", "stage1_phase1.txt"),
        "stage1_phase2": os.path.join(data_dir, "splits", "stage1_phase2.txt"),
        "stage1_phase3": os.path.join(data_dir, "splits", "stage1_phase3.txt"),
        "stage2_mixed": os.path.join(data_dir, "splits", "stage2_mixed.txt"),
        "stage3_alg": os.path.join(data_dir, "splits", "stage3_alg.txt"),
    }


def pick_circuit(data_dir: str, split_name: str):
    split_map = build_split_paths(data_dir)
    split_path = split_map[split_name]
    paths = load_split(split_path)
    path = random.choice(paths)
    with open(os.path.join(data_dir, path), "rb") as f:
        qc = pickle.load(f)
    return CircuitDAG.from_circuit(qc)


# ---------------------------------------------------------------------------
#  Curriculum phase for Stage 1
# ---------------------------------------------------------------------------
def stage1_phase(progress: float) -> str:
    if progress < 0.3:
        return "stage1_phase1"
    elif progress < 0.7:
        return "stage1_phase2"
    else:
        return "stage1_phase3"


def reward_mode_split(reward_mode: str) -> tuple:
    mapping = {
        "routing": ("stage1_phase1", stage1_phase),
        "noise_aware": ("stage2_mixed", lambda _: "stage2_mixed"),
        "fidelity_shaping": ("stage3_alg", lambda _: "stage3_alg"),
    }
    return mapping[reward_mode]


# ---------------------------------------------------------------------------
#  create_env helper
# ---------------------------------------------------------------------------
def create_env(dag, hw, coupling_map, reward_mode, max_episode_steps, random_init, seed, gnn=None, use_gnn=True):
    kw = dict(
        dag=dag, hw=hw, coupling_map=coupling_map,
        reward_mode=reward_mode,
        max_episode_steps=max_episode_steps,
        random_init=random_init,
        seed=seed,
    )
    if gnn is not None:
        kw["gnn"] = gnn
    if not use_gnn:
        kw["use_gnn"] = False
    return RoutingEnv(**kw)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train routing policy with PPO")
    parser.add_argument("--data-dir", type=str, default="../traindata",
                        help="数据集根目录")
    parser.add_argument("--topo", type=str, default=None,
                        help="硬件拓扑 JSON 路径（默认使用线性链）")
    parser.add_argument("--out", type=str, default="../models/policy.pt")
    parser.add_argument("--num-qubits", type=int, default=5,
                        help="硬件比特数（仅无 --topo 时生效）")
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="训练设备 (cpu / cuda:N)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load", type=str, default=None,
                        help="加载预训练模型")
    parser.add_argument("--reward-mode", type=str, default="routing",
                        choices=["routing", "noise_aware", "fidelity_shaping"],
                        help="奖励模式")
    parser.add_argument("--max-episode-steps", type=int, default=200,
                        help="每个 episode 的最大步数（超时截断）")
    parser.add_argument("--random-init", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="是否随机化初始映射")
    parser.add_argument("--no-gnn", action="store_true", default=False,
                        help="禁用 GNN 编码器（不使用 GNN+PPO 联合训练）")
    args = parser.parse_args()

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    # Hardware (fixed across all circuits)
    config, hw, coupling_map = load_topo_or_default(args.num_qubits, args.topo)

    # Dataset
    initial_split_key, phase_fn = reward_mode_split(args.reward_mode)

    use_gnn = not args.no_gnn
    sample_dag = pick_circuit(args.data_dir, initial_split_key)

    if use_gnn:
        shared_gnn = SubGNN(subgraph="full")
        shared_gnn.train()  # trainable for joint training
    else:
        shared_gnn = None

    env = create_env(sample_dag, hw, coupling_map, args.reward_mode,
                     args.max_episode_steps, args.random_init, args.seed,
                     gnn=shared_gnn, use_gnn=use_gnn)

    agent = PPOAgent(
        obs_dim=int(np.prod(env.observation_space.shape)),
        action_dim=int(env.action_space.n),
        lr=args.lr,
        device=args.device,
        gnn=shared_gnn,
        num_qubits=sample_dag.num_logical_qubits,
    )
    if args.load:
        agent.load(args.load)
        print(f"Loaded pretrained model: {args.load}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    obs, _ = env.reset()
    ep_buffer = {"act": [], "logp": [], "val": [], "rew": [], "done": []}
    if use_gnn:
        ep_buffer["graph_data"] = []
        ep_buffer["map_vec"] = []
        ep_buffer["progress"] = []
    else:
        ep_buffer["obs"] = []
    ep_total_reward = 0.0
    ep_rewards = []
    ep_fids = []
    ep_swaps_log = []
    ep_truncated = 0
    ep_completed = 0

    total_steps = 0
    best_metric = -1.0

    while total_steps < args.timesteps:
        for _ in range(args.rollout_steps):
            if use_gnn:
                ep_buffer["graph_data"].append(env._last_graph_data)
                ep_buffer["map_vec"].append(env._last_map_vec)
                ep_buffer["progress"].append(env._last_progress)
            else:
                ep_buffer["obs"].append(obs)

            action, logp, val = agent.act(obs)
            next_obs, reward, done, truncated, info = env.step(action)

            episode_end = done or truncated
            ep_total_reward += reward

            ep_buffer["act"].append(action)
            ep_buffer["logp"].append(logp)
            ep_buffer["val"].append(val)
            ep_buffer["rew"].append(reward)
            ep_buffer["done"].append(episode_end)

            obs = next_obs
            total_steps += 1

            if episode_end:
                ep_rewards.append(ep_total_reward)
                if truncated:
                    ep_truncated += 1
                else:
                    ep_completed += 1
                if "fidelity" in info:
                    ep_fids.append(info["fidelity"])
                ep_swaps_log.append(info.get("num_swaps", 0))

                progress = total_steps / args.timesteps
                split_key = phase_fn(progress)
                new_dag = pick_circuit(args.data_dir, split_key)
                env = create_env(new_dag, hw, coupling_map, args.reward_mode,
                                 args.max_episode_steps, args.random_init,
                                 args.seed + total_steps,
                                 gnn=shared_gnn, use_gnn=use_gnn)
                obs, _ = env.reset()
                ep_total_reward = 0.0

        with torch.no_grad():
            last_val = agent.ac(
                torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            )[1].item()
        adv, ret = PPOAgent.compute_gae(
            ep_buffer["rew"], ep_buffer["val"], ep_buffer["done"],
            bootstrap=last_val, gamma=agent.gamma, lam=agent.lam,
        )
        train_batch = {
            "act": ep_buffer["act"],
            "logp": ep_buffer["logp"],
            "adv": adv,
            "ret": ret,
        }
        if use_gnn:
            train_batch["graph_data"] = ep_buffer["graph_data"]
            train_batch["map_vec"] = ep_buffer["map_vec"]
            train_batch["progress"] = ep_buffer["progress"]
        else:
            train_batch["obs"] = ep_buffer["obs"]
        if use_gnn:
            agent.gnn.train()
        losses = agent.update(train_batch, epochs=args.epochs)
        ep_buffer = {k: [] for k in ep_buffer}

        avg_rew = np.mean(ep_rewards[-20:]) if ep_rewards else 0.0
        avg_swaps = np.mean(ep_swaps_log[-20:]) if ep_swaps_log else 0.0
        total_eps = ep_completed + ep_truncated
        trunc_pct = 100 * ep_truncated / max(1, total_eps)
        parts = [
            f"step={total_steps:>6d}",
            f"rew={avg_rew:+.3f}",
            f"swp={avg_swaps:.1f}",
            f"trunc={trunc_pct:.0f}%",
            f"pl={losses['pl']:.3f}",
            f"vl={losses['vl']:.3f}",
            f"ent={losses['ent']:.3f}",
            f"kl={losses['kl']:.4f}",
            f"gn={losses['grad']:.3f}",
        ]
        if ep_fids:
            avg_fid = np.mean(ep_fids[-20:])
            parts.append(f"fid={avg_fid:.4f}")
        print("  ".join(parts))

        if args.reward_mode == "routing":
            metric = avg_rew
        else:
            metric = np.mean(ep_fids[-20:]) if ep_fids else 0.0
        if metric > best_metric:
            best_metric = metric
            agent.save(args.out)

    print(f"Policy saved to {args.out}")


if __name__ == "__main__":
    main()
