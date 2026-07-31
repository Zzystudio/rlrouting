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
from routing.routing import greedy_route, sabre_route


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

def _lists_to_dict(raw, coupling_map):
    if raw is None or isinstance(raw, (int, float, dict)):
        return raw
    result = {}
    for item in raw:
        q1, q2, v = int(item[0]), int(item[1]), float(item[2])
        result[(q1, q2)] = v
        result[(q2, q1)] = v
    return result


def load_topo(path: str) -> tuple:
    with open(path) as f:
        topo = json.load(f)
    coupling_map = [tuple(e) for e in topo['coupling_map']]
    dp = topo['device_params']
    tqe = _lists_to_dict(dp.get('two_q_gate_error', 0.01), coupling_map)
    cs = _lists_to_dict(topo.get('crosstalk_strength'), coupling_map)
    config = NoiseConfig(
        t1_times=dp['t1_times'],
        t2_times=dp['t2_times'],
        freq_ghz=dp['freq_ghz'],
        single_q_gate_error=dp.get('single_q_gate_error', 0.001),
        two_q_gate_error=tqe,
        coupling_map=coupling_map,
        readout_error=dp['readout_error'],
        shots=dp.get('shots', 1024),
        crosstalk_strength=cs,
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


def load_qc(data_dir: str, rel_path: str, seed: int = 0):
    with open(os.path.join(data_dir, rel_path), 'rb') as f:
        qc = pickle.load(f)
    if qc.num_parameters > 0:
        import numpy as np
        rng = np.random.default_rng(seed)
        qc = qc.assign_parameters({p: rng.uniform(0, 2 * np.pi) for p in qc.parameters})
    return qc


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
    noise_config: Optional[NoiseConfig] = None,
    use_deadlock_mask: bool = True,
) -> CircuitMetrics:
    import torch
    env = RoutingEnv(
        dag, hw, coupling_map, reward_mode=reward_mode,
        max_episode_steps=max_episode_steps,
        random_init=False, seed=seed,
        gnn=agent.gnn, use_gnn=agent.gnn is not None,
        noise_config=noise_config if reward_mode != 'routing' else None,
    )

    obs, _ = env.reset()
    t0 = time.perf_counter()

    done, truncated = False, False
    step = 0
    while not done and not truncated:
        with torch.no_grad():
            mask = None
            if use_deadlock_mask and hasattr(env, 'get_deadlock_mask') and agent.gnn is not None:
                dm = env.get_deadlock_mask()
                mask = torch.ones(agent.num_edges, dtype=torch.bool, device=agent.device)
                mask[:len(coupling_map)] = True
                for i in range(min(len(dm), len(mask))):
                    if dm[i]:
                        mask[i] = False
                mask = mask.unsqueeze(0)
            logits, _ = agent._forward_obs(obs, action_mask=mask)
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
        fidelity=info.get('fidelity', None),
    )


# ---------------------------------------------------------------------------
#  Single-circuit evaluation: 1-step Beam Search
# ---------------------------------------------------------------------------

def evaluate_circuit_beam(
    dag: CircuitDAG,
    hw: HardwareFeatures,
    coupling_map: list,
    agent: PPOAgent,
    reward_mode: str = 'routing',
    max_episode_steps: int = 200,
    seed: int = 0,
    noise_config: Optional[NoiseConfig] = None,
    beam_width: int = 3,
) -> CircuitMetrics:
    import torch
    env = RoutingEnv(
        dag, hw, coupling_map, reward_mode=reward_mode,
        max_episode_steps=max_episode_steps,
        random_init=False, seed=seed,
        gnn=agent.gnn, use_gnn=agent.gnn is not None,
        noise_config=noise_config if reward_mode != 'routing' else None,
    )

    obs, _ = env.reset()
    t0 = time.perf_counter()

    done, truncated = False, False
    step = 0
    while not done and not truncated:
        with torch.no_grad():
            # Build base action mask (valid edges + deadlock)
            mask = torch.zeros(agent.num_edges, dtype=torch.bool, device=agent.device)
            mask[:len(coupling_map)] = True
            if hasattr(env, 'get_deadlock_mask'):
                dm = env.get_deadlock_mask()
                for i in range(min(len(dm), len(mask))):
                    if dm[i]:
                        mask[i] = False
            mask = mask.unsqueeze(0)

            logits, _ = agent._forward_obs(obs, action_mask=mask)
            masked_logits = logits[0].clone()
            masked_logits[~mask[0]] = -1e9
            k = min(beam_width, (mask[0].sum().item()))
            topk_scores, topk_indices = masked_logits.topk(k)

            best_action, best_score = topk_indices[0].item(), -float('inf')
            for i in range(topk_indices.shape[0]):
                a = topk_indices[i].item()
                clone = env.clone()
                _, _, done_c, truncated_c, info_c = clone.step(a)
                if done_c:
                    score = 10.0
                elif truncated_c:
                    score = -10.0
                else:
                    clone_obs = clone._obs()
                    _, v = agent._forward_obs(clone_obs)
                    score = v.item()
                if score > best_score:
                    best_score = score
                    best_action = a

        obs, reward, done, truncated, info = env.step(best_action)
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
        fidelity=info.get('fidelity', None),
    )


# ---------------------------------------------------------------------------
#  Single-circuit evaluation: MCTS inference
# ---------------------------------------------------------------------------

def evaluate_circuit_mcts(
    dag: CircuitDAG,
    hw: HardwareFeatures,
    coupling_map: list,
    agent: PPOAgent,
    reward_mode: str = 'routing',
    max_episode_steps: int = 200,
    seed: int = 0,
    noise_config: Optional[NoiseConfig] = None,
    num_simulations: int = 100,
    c_puct: float = 1.4,
    temperature: float = 0.0,
) -> CircuitMetrics:
    from routing.rl.mcts import MCTS

    env = RoutingEnv(
        dag, hw, coupling_map, reward_mode=reward_mode,
        max_episode_steps=max_episode_steps,
        random_init=False, seed=seed,
        gnn=agent.gnn, use_gnn=agent.gnn is not None,
        noise_config=noise_config if reward_mode != 'routing' else None,
    )

    obs, _ = env.reset()
    mcts = MCTS(agent, num_simulations=num_simulations,
                c_puct=c_puct, temperature=temperature)
    t0 = time.perf_counter()

    done, truncated = False, False
    step = 0
    while not done and not truncated:
        best_action, _ = mcts.search(env)
        obs, reward, done, truncated, info = env.step(best_action)
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
        fidelity=info.get('fidelity', None),
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
    noise_config: Optional[NoiseConfig] = None,
) -> CircuitMetrics:
    env = RoutingEnv(
        dag, hw, coupling_map, reward_mode=reward_mode,
        max_episode_steps=max_episode_steps,
        random_init=False, seed=seed,
        use_gnn=False,
        noise_config=noise_config if reward_mode != 'routing' else None,
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
        fidelity=info.get('fidelity', None),
    )


# ---------------------------------------------------------------------------
#  Greedy baseline
# ---------------------------------------------------------------------------

def evaluate_greedy(
    qc,
    config: NoiseConfig,
    reward_mode: str = 'routing',
) -> CircuitMetrics:
    from sim.sim import NoiseSimulator
    from qiskit_aer import AerSimulator

    dag = CircuitDAG.from_circuit(qc)
    t0 = time.perf_counter()
    phys, info = greedy_route(qc, config)
    wall_time_ms = (time.perf_counter() - t0) * 1000

    fid = None
    if reward_mode != 'routing':
        meas = phys.copy()
        meas.measure_all()
        noise_sim = NoiseSimulator(config)
        meas_t = noise_sim._transpile(meas)
        shots = config.shots

        ideal_sim = AerSimulator()
        ideal_job = ideal_sim.run(meas_t, shots=shots)
        ideal_counts = ideal_job.result().get_counts()

        noisy_counts = noise_sim.run(meas_t, shots=shots, skip_transpile=True)

        all_outcomes = set(ideal_counts.keys()) | set(noisy_counts.keys())
        overlap = sum(min(ideal_counts.get(k, 0), noisy_counts.get(k, 0)) for k in all_outcomes)
        fid = overlap / shots

    return CircuitMetrics(
        circuit_path='',
        completed=True,
        num_swaps=info['num_swaps'],
        gates_executed=dag.num_gates,
        total_gates=dag.num_gates,
        episode_steps=0,
        wall_time_ms=wall_time_ms,
        terminal_xz=None,
        fidelity=fid,
    )


# ---------------------------------------------------------------------------
#  SABRE baseline
# ---------------------------------------------------------------------------

def evaluate_sabre(
    qc,
    config: NoiseConfig,
    reward_mode: str = 'routing',
    heuristic: str = 'decay',
    swap_trials: int = 20,
    seed: int = 0,
) -> CircuitMetrics:
    dag = CircuitDAG.from_circuit(qc)
    t0 = time.perf_counter()
    phys, info = sabre_route(qc, config, heuristic=heuristic, swap_trials=swap_trials, seed=seed)
    wall_time_ms = (time.perf_counter() - t0) * 1000

    fid = None
    if reward_mode != 'routing':
        from sim.sim import NoiseSimulator
        from qiskit_aer import AerSimulator
        meas = phys.copy()
        meas.measure_all()
        noise_sim = NoiseSimulator(config)
        meas_t = noise_sim._transpile(meas)
        shots = config.shots

        ideal_sim = AerSimulator()
        ideal_job = ideal_sim.run(meas_t, shots=shots)
        ideal_counts = ideal_job.result().get_counts()

        noisy_counts = noise_sim.run(meas_t, shots=shots, skip_transpile=True)

        all_outcomes = set(ideal_counts.keys()) | set(noisy_counts.keys())
        overlap = sum(min(ideal_counts.get(k, 0), noisy_counts.get(k, 0)) for k in all_outcomes)
        fid = overlap / shots

    return CircuitMetrics(
        circuit_path='',
        completed=True,
        num_swaps=info['num_swaps'],
        gates_executed=dag.num_gates,
        total_gates=dag.num_gates,
        episode_steps=0,
        wall_time_ms=wall_time_ms,
        terminal_xz=None,
        fidelity=fid,
    )


# ---------------------------------------------------------------------------
#  Aggregate stats
# ---------------------------------------------------------------------------

def aggregate(metrics: List[CircuitMetrics]) -> SummaryStats:
    n = len(metrics)
    completed = [m for m in metrics if m.completed]
    comp_rate = len(completed) / n if n > 0 else 0.0

    completed_n = len(completed) if completed else 1
    swaps = np.array([m.num_swaps for m in completed], dtype=float) if completed else np.zeros(1)
    steps = np.array([m.episode_steps for m in completed], dtype=float) if completed else np.zeros(1)
    times = np.array([m.wall_time_ms for m in metrics], dtype=float)
    xz_vals = [m.terminal_xz for m in completed if m.terminal_xz is not None]
    fid_vals = [m.fidelity for m in completed if m.fidelity is not None]

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
                        help='also run random, greedy, and SABRE baselines')
    parser.add_argument('--sabre-heuristic', type=str, default='decay',
                        choices=['basic', 'decay', 'lookahead'],
                        help='SABRE heuristic (default: decay)')
    parser.add_argument('--sabre-trials', type=int, default=20,
                        help='SABRE swap trials per circuit (default: 20)')
    parser.add_argument('--max-circuits', type=int, default=None,
                        help='limit number of circuits to evaluate')
    parser.add_argument('--beam-width', type=int, default=0,
                        help='beam width for 1-step lookahead (0 = argmax)')
    parser.add_argument('--search', type=str, default='argmax',
                        choices=['argmax', 'beam', 'mcts'],
                        help='inference search method (default: argmax)')
    parser.add_argument('--mcts-simulations', type=int, default=100,
                        help='MCTS simulations per decision step (default: 100)')
    parser.add_argument('--mcts-c-puct', type=float, default=1.4,
                        help='PUCT exploration constant (default: 1.4)')
    parser.add_argument('--mcts-temperature', type=float, default=0.0,
                        help='MCTS action temperature, 0=deterministic (default: 0.0)')
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
        num_edges=len(coupling_map),
        coupling_map=coupling_map,
    )
    agent.load(args.model)
    agent.ac.eval()
    if agent.gnn is not None:
        agent.gnn.eval()

    # ---- Evaluate ----
    def _progress(i: int, total: int, method: str):
        if args.verbose:
            print(f'  [{i+1}/{total}] {method}...', end=' ', flush=True)

    if args.search == 'mcts':
        label = f'PPO_MCTS({args.mcts_simulations})'
    elif args.beam_width > 0:
        label = f'PPO_beam{args.beam_width}'
    else:
        label = 'PPO'

    def evaluate_agent_on_circuits():
        results = []
        for i, rel_path in enumerate(rel_paths):
            _progress(i, len(rel_paths), label)
            qc = load_qc(args.data_dir, rel_path)
            dag = CircuitDAG.from_circuit(qc)
            if args.search == 'mcts':
                m = evaluate_circuit_mcts(
                    dag, hw, coupling_map, agent,
                    reward_mode=args.reward_mode,
                    max_episode_steps=args.max_episode_steps,
                    seed=args.seed + i,
                    noise_config=config if args.reward_mode != 'routing' else None,
                    num_simulations=args.mcts_simulations,
                    c_puct=args.mcts_c_puct,
                    temperature=args.mcts_temperature,
                )
            elif args.beam_width > 0:
                m = evaluate_circuit_beam(
                    dag, hw, coupling_map, agent,
                    reward_mode=args.reward_mode,
                    max_episode_steps=args.max_episode_steps,
                    seed=args.seed + i,
                    noise_config=config if args.reward_mode != 'routing' else None,
                    beam_width=args.beam_width,
                )
            else:
                m = evaluate_circuit(
                    dag, hw, coupling_map, agent,
                    reward_mode=args.reward_mode,
                    max_episode_steps=args.max_episode_steps,
                    deterministic=args.deterministic,
                    seed=args.seed + i,
                    noise_config=config if args.reward_mode != 'routing' else None,
                )
            m.circuit_path = rel_path
            if args.verbose:
                tag = 'OK' if m.completed else 'TRUNC'
                print(f'{tag} swaps={m.num_swaps} steps={m.episode_steps} {m.wall_time_ms:.0f}ms')
            results.append(m)
        return results

    agent_metrics = evaluate_agent_on_circuits()
    agent_stats = aggregate(agent_metrics)

    show_fid = args.reward_mode != 'routing'
    print_header(show_fidelity=show_fid)
    print_report(label, agent_stats, show_fidelity=show_fid)

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
                noise_config=config if args.reward_mode != 'routing' else None,
            )
            m.circuit_path = rel_path
            if args.verbose:
                tag = 'OK' if m.completed else 'TRUNC'
                print(f'{tag} swaps={m.num_swaps} steps={m.episode_steps} {m.wall_time_ms:.0f}ms')
            random_metrics.append(m)
        random_stats = aggregate(random_metrics)
        print_report('Random', random_stats, show_fidelity=show_fid)

        # Greedy
        greedy_metrics = []
        for i, rel_path in enumerate(rel_paths):
            _progress(i, len(rel_paths), 'Greedy')
            qc = load_qc(args.data_dir, rel_path)
            m = evaluate_greedy(qc, config, reward_mode=args.reward_mode)
            m.circuit_path = rel_path
            if args.verbose:
                print(f'OK swaps={m.num_swaps} {m.wall_time_ms:.0f}ms')
            greedy_metrics.append(m)
        greedy_stats = aggregate(greedy_metrics)
        print_report('Greedy', greedy_stats, show_fidelity=show_fid)

        # SABRE
        sabre_metrics = []
        for i, rel_path in enumerate(rel_paths):
            _progress(i, len(rel_paths), 'SABRE')
            qc = load_qc(args.data_dir, rel_path)
            m = evaluate_sabre(
                qc, config,
                reward_mode=args.reward_mode,
                heuristic=args.sabre_heuristic,
                swap_trials=args.sabre_trials,
                seed=args.seed + i + 2000,
            )
            m.circuit_path = rel_path
            if args.verbose:
                print(f'OK swaps={m.num_swaps} {m.wall_time_ms:.0f}ms')
            sabre_metrics.append(m)
        sabre_stats = aggregate(sabre_metrics)
        print_report('SABRE', sabre_stats, show_fidelity=show_fid)

    print()

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
            out['sabre'] = [_asdict(m) for m in sabre_metrics]
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'Results saved to {args.out}')


if __name__ == '__main__':
    main()
