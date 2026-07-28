from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from sim.sim import NoiseConfig
from routing.graph.circuit_dag import CircuitDAG
from routing.graph.features import HardwareFeatures
from routing.rl.env import RoutingEnv
from routing.rl.agent import PPOAgent
from routing.gnn.encoder import SubGNN
from routing.routing import greedy_route


# ---------------------------------------------------------------------------
#  Metrics per circuit
# ---------------------------------------------------------------------------

@dataclass
class CircuitMetrics:
    circuit_path: str
    completed: bool
    num_swaps: int
    gates_executed: int
    total_gates: int
    episode_steps: int
    wall_time_ms: float
    terminal_xz: Optional[float]
    truncated_remaining: int = 0
    fidelity: Optional[float] = None


@dataclass
class SummaryStats:
    n: int
    comp_rate: float
    swaps_mean: float
    swaps_std: float
    swaps_min: int
    swaps_max: int
    steps_mean: float
    steps_std: float
    time_mean_ms: float
    time_std_ms: float
    xz_mean: Optional[float] = None
    fidelity_mean: Optional[float] = None


# ---------------------------------------------------------------------------
#  Hardware helpers
# ---------------------------------------------------------------------------

def load_topo(path: str) -> tuple:
    with open(path) as f:
        topo = json.load(f)
    coupling_map = [tuple(e) for e in topo['coupling_map']]
    dp = topo['device_params']
    config = NoiseConfig(
        t1_times=dp['t1_times'],
        t2_times=dp['t2_times'],
        freq_ghz=dp['freq_ghz'],
        single_q_gate_error=dp['single_q_gate_error'],
        two_q_gate_error=dp['two_q_gate_error'],
        coupling_map=coupling_map,
        readout_error=dp['readout_error'],
        shots=dp.get('shots', 1024),
        crosstalk_strength=topo.get('crosstalk_strength'),
    )
    hw = HardwareFeatures.from_noise_config(config)
    return config, hw, coupling_map


def make_default_hw(num_qubits: int = 5) -> tuple:
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


def load_split(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_qc(data_dir: str, rel_path: str):
    with open(os.path.join(data_dir, rel_path), 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
#  Fidelity stub
# ---------------------------------------------------------------------------

def compute_fidelity(qc, config, mapping, executed) -> Optional[float]:
    return None


# ---------------------------------------------------------------------------
#  Single-circuit evaluation: PPO agent
# ---------------------------------------------------------------------------

def evaluate_circuit(
    dag: CircuitDAG,
    hw: HardwareFeatures,
    coupling_map: list,
    agent: PPOAgent,
    reward_mode: str = 'routing',
    max_episode_steps: int = 200,
    deterministic: bool = True,
    seed: int = 0,
) -> CircuitMetrics:
    import torch
    env = RoutingEnv(
        dag, hw, coupling_map, reward_mode=reward_mode,
        max_episode_steps=max_episode_steps,
        random_init=False, seed=seed,
        gnn=agent.gnn, use_gnn=agent.gnn is not None,
    )

    obs, _ = env.reset()
    t0 = time.perf_counter()

    done, truncated = False, False
    step = 0
    while not done and not truncated:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32,
                                 device=agent.device).unsqueeze(0)
            logits, _ = agent.ac(obs_t)
            action = logits.argmax(-1).item() if deterministic \
                     else torch.distributions.Categorical(logits=logits).sample().item()
        obs, reward, done, truncated, info = env.step(action)
        step += 1

    wall_time_ms = (time.perf_counter() - t0) * 1000

    return CircuitMetrics(
        circuit_path='',
        completed=done,
        num_swaps=env._swap_counter,
        gates_executed=len(env.executed),
        total_gates=dag.num_gates,
        episode_steps=step,
        wall_time_ms=wall_time_ms,
        terminal_xz=info.get('terminal_XZ', None),
        truncated_remaining=info.get('truncated_remaining', 0),
    )


# ---------------------------------------------------------------------------
#  Single-circuit evaluation: Random baseline
# ---------------------------------------------------------------------------

def evaluate_random(
    dag: CircuitDAG,
    hw: HardwareFeatures,
    coupling_map: list,
    reward_mode: str = 'routing',
    max_episode_steps: int = 200,
    seed: int = 0,
) -> CircuitMetrics:
    env = RoutingEnv(
        dag, hw, coupling_map, reward_mode=reward_mode,
        max_episode_steps=max_episode_steps,
        random_init=False, seed=seed,
        use_gnn=False,
    )

    obs, _ = env.reset()
    t0 = time.perf_counter()

    done, truncated = False, False
    step = 0
    while not done and not truncated:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        step += 1

    wall_time_ms = (time.perf_counter() - t0) * 1000

    return CircuitMetrics(
        circuit_path='',
        completed=done,
        num_swaps=env._swap_counter,
        gates_executed=len(env.executed),
        total_gates=dag.num_gates,
        episode_steps=step,
        wall_time_ms=wall_time_ms,
        terminal_xz=info.get('terminal_XZ', None),
        truncated_remaining=info.get('truncated_remaining', 0),
    )


# ---------------------------------------------------------------------------
#  Greedy baseline
# ---------------------------------------------------------------------------

def evaluate_greedy(
    qc,
    config: NoiseConfig,
) -> CircuitMetrics:
    dag = CircuitDAG.from_circuit(qc)
    t0 = time.perf_counter()
    _, info = greedy_route(qc, config)
    wall_time_ms = (time.perf_counter() - t0) * 1000

    return CircuitMetrics(
        circuit_path='',
        completed=True,
        num_swaps=info['num_swaps'],
        gates_executed=dag.num_gates,
        total_gates=dag.num_gates,
        episode_steps=0,
        wall_time_ms=wall_time_ms,
        terminal_xz=None,
    )


# ---------------------------------------------------------------------------
#  Aggregate stats
# ---------------------------------------------------------------------------

def aggregate(metrics: List[CircuitMetrics]) -> SummaryStats:
    n = len(metrics)
    completed = [m for m in metrics if m.completed]
    comp_rate = len(completed) / n if n > 0 else 0.0

    swaps = np.array([m.num_swaps for m in metrics], dtype=float)
    steps = np.array([m.episode_steps for m in metrics], dtype=float)
    times = np.array([m.wall_time_ms for m in metrics], dtype=float)
    xz_vals = [m.terminal_xz for m in metrics if m.terminal_xz is not None]
    fid_vals = [m.fidelity for m in metrics if m.fidelity is not None]

    return SummaryStats(
        n=n,
        comp_rate=comp_rate,
        swaps_mean=float(np.mean(swaps)),
        swaps_std=float(np.std(swaps)),
        swaps_min=int(np.min(swaps)),
        swaps_max=int(np.max(swaps)),
        steps_mean=float(np.mean(steps)),
        steps_std=float(np.std(steps)),
        time_mean_ms=float(np.mean(times)),
        time_std_ms=float(np.std(times)),
        xz_mean=float(np.mean(xz_vals)) if xz_vals else None,
        fidelity_mean=float(np.mean(fid_vals)) if fid_vals else None,
    )


# ---------------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------------

def print_header(show_fidelity: bool = False):
    parts = [
        f"{'Method':<8s}",
        f"{'Time(ms)':>8s}",
        f"{'Comp%':>6s}",
        f"{'SWAPs':>16s}",
        f"{'Steps':>16s}",
        f"{'XZ':>10s}",
    ]
    if show_fidelity:
        parts.append(f"{'Fidelity':>9s}")
    sep = '  '.join(parts)
    print(sep)
    print('-' * len(sep))


def print_report(
    label: str,
    stats: SummaryStats,
    show_fidelity: bool = False,
):
    parts = [
        f"{label:<8s}",
        f"{stats.time_mean_ms:>8.1f}",
        f"{stats.comp_rate * 100:>5.1f}%",
        f"{stats.swaps_mean:>6.1f} +/- {stats.swaps_std:<5.1f}",
        f"{stats.steps_mean:>6.0f} +/- {stats.steps_std:<5.0f}",
    ]
    if stats.xz_mean is not None:
        parts.append(f"{stats.xz_mean:>8.4f}")
    else:
        parts.append('      --   ')
    if show_fidelity:
        if stats.fidelity_mean is not None:
            parts.append(f"{stats.fidelity_mean:>8.4f}")
        else:
            parts.append('      --   ')
    print('  '.join(parts))


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained routing policy')
    parser.add_argument('--model', type=str, required=True,
                        help='policy checkpoint path')
    parser.add_argument('--data-dir', type=str, default='../traindata',
                        help='dataset root directory')
    parser.add_argument('--split', type=str, default='stage1_phase3',
                        choices=['stage1_phase1', 'stage1_phase2',
                                 'stage1_phase3', 'stage2_mixed',
                                 'stage3_alg'],
                        help='split to evaluate on')
    parser.add_argument('--reward-mode', type=str, default='routing',
                        choices=['routing', 'noise_aware', 'fidelity_shaping'])
    parser.add_argument('--topo', type=str, default=None,
                        help='hardware topology JSON')
    parser.add_argument('--num-qubits', type=int, default=5,
                        help='physical qubits (default linear chain)')
    parser.add_argument('--max-episode-steps', type=int, default=200)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-gnn', action='store_true', default=False,
                        help='model was trained without GNN')
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='use argmax for action selection')
    parser.add_argument('--baselines', action='store_true', default=False,
                        help='also run random and greedy baselines')
    parser.add_argument('--max-circuits', type=int, default=None,
                        help='limit number of circuits to evaluate')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='print per-circuit results')
    parser.add_argument('--out', type=str, default=None,
                        help='save per-circuit results as JSON')
    args = parser.parse_args()

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Hardware ----
    if args.topo:
        config, hw, coupling_map = load_topo(args.topo)
    else:
        config, hw, coupling_map = make_default_hw(args.num_qubits)

    # ---- Load circuits from split ----
    split_path = os.path.join(args.data_dir, 'splits', args.split + '.txt')
    if not os.path.exists(split_path):
        print(f'Split file not found: {split_path}')
        sys.exit(1)

    rel_paths = load_split(split_path)
    if args.max_circuits is not None and args.max_circuits < len(rel_paths):
        rel_paths = rel_paths[:args.max_circuits]
    print(f'Split: {args.split} ({len(rel_paths)} circuits)')
    print(f'Model: {args.model}')
    print()

    # ---- Probe circuit to set up agent ----
    sample_qc = load_qc(args.data_dir, rel_paths[0])
    sample_dag = CircuitDAG.from_circuit(sample_qc)

    use_gnn = not args.no_gnn
    shared_gnn = SubGNN(subgraph='full') if use_gnn else None
    if use_gnn:
        shared_gnn.eval()

    sample_env = RoutingEnv(
        sample_dag, hw, coupling_map, reward_mode=args.reward_mode,
        max_episode_steps=args.max_episode_steps,
        random_init=False, seed=args.seed,
        gnn=shared_gnn, use_gnn=use_gnn,
    )

    agent = PPOAgent(
        obs_dim=int(np.prod(sample_env.observation_space.shape)),
        action_dim=int(sample_env.action_space.n),
        device=args.device,
        gnn=shared_gnn,
        num_qubits=sample_dag.num_logical_qubits,
    )
    agent.load(args.model)
    agent.ac.eval()
    if agent.gnn is not None:
        agent.gnn.eval()

    # ---- Evaluate ----
    def _progress(i: int, total: int, method: str):
        if args.verbose:
            print(f'  [{i+1}/{total}] {method}...', end=' ', flush=True)

    def evaluate_agent_on_circuits():
        results = []
        for i, rel_path in enumerate(rel_paths):
            _progress(i, len(rel_paths), 'PPO')
            qc = load_qc(args.data_dir, rel_path)
            dag = CircuitDAG.from_circuit(qc)
            m = evaluate_circuit(
                dag, hw, coupling_map, agent,
                reward_mode=args.reward_mode,
                max_episode_steps=args.max_episode_steps,
                deterministic=args.deterministic,
                seed=args.seed + i,
            )
            m.circuit_path = rel_path
            if args.verbose:
                tag = 'OK' if m.completed else 'TRUNC'
                print(f'{tag} swaps={m.num_swaps} steps={m.episode_steps} {m.wall_time_ms:.0f}ms')
            results.append(m)
        return results

    agent_metrics = evaluate_agent_on_circuits()
    agent_stats = aggregate(agent_metrics)

    print_header()
    print_report('PPO', agent_stats)

    if args.baselines:
        # Random
        random_metrics = []
        for i, rel_path in enumerate(rel_paths):
            _progress(i, len(rel_paths), 'Random')
            qc = load_qc(args.data_dir, rel_path)
            dag = CircuitDAG.from_circuit(qc)
            m = evaluate_random(
                dag, hw, coupling_map,
                reward_mode=args.reward_mode,
                max_episode_steps=args.max_episode_steps,
                seed=args.seed + i + 1000,
            )
            m.circuit_path = rel_path
            if args.verbose:
                tag = 'OK' if m.completed else 'TRUNC'
                print(f'{tag} swaps={m.num_swaps} steps={m.episode_steps} {m.wall_time_ms:.0f}ms')
            random_metrics.append(m)
        random_stats = aggregate(random_metrics)
        print_report('Random', random_stats)

        # Greedy
        greedy_metrics = []
        for i, rel_path in enumerate(rel_paths):
            _progress(i, len(rel_paths), 'Greedy')
            qc = load_qc(args.data_dir, rel_path)
            m = evaluate_greedy(qc, config)
            m.circuit_path = rel_path
            if args.verbose:
                print(f'OK swaps={m.num_swaps} {m.wall_time_ms:.0f}ms')
            greedy_metrics.append(m)
        greedy_stats = aggregate(greedy_metrics)
        print_report('Greedy', greedy_stats)

    print()
    print('Fidelity: (not computed -- simulator stub)')

    # ---- Save per-circuit results ----
    if args.out:
        def _asdict(m: CircuitMetrics) -> dict:
            return {
                'circuit_path': m.circuit_path,
                'completed': m.completed,
                'num_swaps': m.num_swaps,
                'gates_executed': m.gates_executed,
                'total_gates': m.total_gates,
                'episode_steps': m.episode_steps,
                'wall_time_ms': m.wall_time_ms,
                'terminal_xz': m.terminal_xz,
                'truncated_remaining': m.truncated_remaining,
                'fidelity': m.fidelity,
            }

        out = {
            'args': vars(args),
            'agent': [_asdict(m) for m in agent_metrics],
        }
        if args.baselines:
            out['random'] = [_asdict(m) for m in random_metrics]
            out['greedy'] = [_asdict(m) for m in greedy_metrics]
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'Results saved to {args.out}')


if __name__ == '__main__':
    main()
