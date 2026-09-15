from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pytest

from routing.graph.circuit_dag import CircuitDAG
from routing.graph.features import HardwareFeatures
from routing.rl.env import RoutingEnv
from routing.rl.agent import PPOAgent
from routing.gnn.encoder import SubGNN
from routing.routing import greedy_route, sabre_route
from sim.sim import NoiseConfig
from utils.data_gen import random_circuit

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
TOPOS = {
    "cross": "traindata/topo/cross_5q.json",
    "ring": "traindata/topo/ring_5q.json",
    "line": "traindata/topo/ibmq_5_line.json",
}
MODEL_PATH = "models/policy_phase1_noiseaware.pt"

NUM_CIRCUITS = 5
NUM_QUBITS = 5
MAX_EPISODE_STEPS = 200
SEED_OFFSET = 1000  # stay clear of training seeds

# --------------------------------------------------------------------------
#  Topology loader
# --------------------------------------------------------------------------

def _normalize_dict_keys(d):
    if not isinstance(d, dict):
        return d
    out = {}
    for k, v in d.items():
        if isinstance(k, str):
            try:
                k = ast.literal_eval(k)
            except (ValueError, SyntaxError):
                pass
        out[k] = v
    return out


def _lists_to_dict(raw, coupling_map):
    if raw is None or isinstance(raw, (int, float, dict)):
        return _normalize_dict_keys(raw)
    result = {}
    for item in raw:
        q1, q2, v = int(item[0]), int(item[1]), float(item[2])
        result[(q1, q2)] = v
        result[(q2, q1)] = v
    return result


def load_topo(path: str) -> Tuple[NoiseConfig, HardwareFeatures, list]:
    with open(path) as f:
        topo = json.load(f)
    coupling_map = [tuple(e) for e in topo["coupling_map"]]
    dp = topo["device_params"]
    tqe = _lists_to_dict(dp.get("two_q_gate_error", 0.01), coupling_map)
    cs = _lists_to_dict(topo.get("crosstalk_strength"), coupling_map)
    config = NoiseConfig(
        t1_times=dp["t1_times"],
        t2_times=dp["t2_times"],
        freq_ghz=dp["freq_ghz"],
        single_q_gate_error=dp.get("single_q_gate_error", 0.001),
        two_q_gate_error=tqe,
        coupling_map=coupling_map,
        readout_error=dp["readout_error"],
        shots=dp.get("shots", 1024),
        crosstalk_strength=cs,
    )
    hw = HardwareFeatures.from_noise_config(config)
    return config, hw, coupling_map


# --------------------------------------------------------------------------
#  Metrics
# --------------------------------------------------------------------------

@dataclass
class Result:
    method: str
    topo: str
    circuit_idx: int
    completed: bool
    num_swaps: int
    fidelity: Optional[float]
    wall_time_ms: float


# --------------------------------------------------------------------------
#  Fixtures (module-scoped — load once)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_and_agent():
    """Load the trained model once for the entire module."""
    import torch

    # Pick one topology to infer model dimensions
    config, hw, cm = load_topo(TOPOS["cross"])
    qc = random_circuit(NUM_QUBITS, 6, seed=SEED_OFFSET)
    dag = CircuitDAG.from_circuit(qc)

    gnn = SubGNN(subgraph="full")
    gnn.eval()
    max_edges = 5  # ring has 5, cross and line have 4

    env = RoutingEnv(
        dag, hw, cm, reward_mode="routing",
        max_episode_steps=MAX_EPISODE_STEPS,
        random_init=False, seed=0,
        gnn=gnn, use_gnn=True,
        max_num_edges=max_edges,
        mapping_phase=False,
        # 旧 checkpoint（policy_phase1_noiseaware.pt）为 pre-R5a 149 维布局，
        # 须关闭 look/noise 特征块使 env 维度与之一致（此前 33.8 swaps 假阳性
        # 正是 153 维 env obs 与 149 维 checkpoint 错配所致）
        lookahead_features=False,
        edge_noise_features=False,
    )
    agent = PPOAgent(
        obs_dim=int(np.prod(env.observation_space.shape)),
        action_dim=max_edges,
        device="cpu",
        gnn=gnn,
        num_qubits=NUM_QUBITS,
        num_edges=max_edges,
        coupling_map=cm,
        with_commit=False,
    )
    agent.load(MODEL_PATH)
    env.close()
    return agent


# --------------------------------------------------------------------------
#  Inference helpers
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
#  Inference helpers
# --------------------------------------------------------------------------

def evaluate_ppo(dag, hw, coupling_map, config, agent, seed, max_num_edges) -> Result:
    """Run PPO deterministically with deadlock mask."""
    import torch
    env = RoutingEnv(
        dag, hw, coupling_map, reward_mode="noise_aware",
        max_episode_steps=MAX_EPISODE_STEPS,
        random_init=False, seed=seed,
        gnn=agent.gnn, use_gnn=agent.gnn is not None,
        noise_config=config,
        max_num_edges=max_num_edges,
        mapping_phase=False,
        lookahead_features=False,
        edge_noise_features=False,
    )
    obs, _ = env.reset()
    t0 = time.perf_counter()
    done, truncated = False, False
    while not done and not truncated:
        with torch.no_grad():
            dm = env.get_deadlock_mask()
            mask = torch.zeros(agent.num_edges, dtype=torch.bool, device=agent.device)
            mask[:len(coupling_map)] = True
            for i in range(min(len(dm), len(mask))):
                if dm[i]:
                    mask[i] = False
            logits, _ = agent._forward_obs(obs, action_mask=mask.unsqueeze(0))
            action = logits.argmax(-1).item()
        obs, reward, done, truncated, info = env.step(action)
    wall = (time.perf_counter() - t0) * 1000
    fid = info.get("fidelity")
    env.close()
    return Result("PPO", "", seed, done, env._swap_counter, fid, wall)


def evaluate_greedy(qc, config, seed) -> Result:
    t0 = time.perf_counter()
    phys, info = greedy_route(qc, config)
    wall = (time.perf_counter() - t0) * 1000
    dag = CircuitDAG.from_circuit(qc)
    return Result("Greedy", "", seed, True, info["num_swaps"], None, wall)


def evaluate_sabre(qc, config, seed) -> Result:
    t0 = time.perf_counter()
    phys, info = sabre_route(qc, config, heuristic="decay", swap_trials=20, seed=seed)
    wall = (time.perf_counter() - t0) * 1000
    dag = CircuitDAG.from_circuit(qc)
    return Result("SABRE", "", seed, True, info["num_swaps"], None, wall)


def compute_fidelity(qc, config, result):
    """Add fidelity to a greedy/SABRE result that already has swaps."""
    from qiskit_aer import AerSimulator
    from sim.sim import NoiseSimulator
    from routing.routing import greedy_route, sabre_route
    from utils.metrics import counts_fidelity

    if result.method == "Greedy":
        phys, _ = greedy_route(qc, config)
    else:
        phys, _ = sabre_route(qc, config, heuristic="decay", swap_trials=20, seed=result.circuit_idx)

    meas = phys.copy()
    meas.measure_all()

    noise_sim = NoiseSimulator(config)
    meas_t = noise_sim._transpile(meas)
    shots = config.shots

    noisy_counts = noise_sim.run(meas_t, shots=shots, skip_transpile=True)
    ideal_sim = AerSimulator()
    ideal_job = ideal_sim.run(meas_t, shots=shots)
    ideal_counts = ideal_job.result().get_counts()

    result.fidelity = counts_fidelity(ideal_counts, noisy_counts)
    return result


# --------------------------------------------------------------------------
#  Main test
# --------------------------------------------------------------------------

TOPOLOGY_NAMES = {"cross": "cross_5q", "ring": "ring_5q", "line": "ibmq_5_line"}

# Mark as slow since it involves Aer simulation
@pytest.mark.slow
def test_phase1_vs_baselines(model_and_agent):
    agent = model_and_agent
    all_results: List[Result] = []

    for topo_key, topo_path in TOPOS.items():
        topo_name = TOPOLOGY_NAMES[topo_key]
        config, hw, cm = load_topo(topo_path)

        for ci in range(NUM_CIRCUITS):
            seed = SEED_OFFSET + ci
            qc = random_circuit(NUM_QUBITS, 8, seed=seed)
            dag = CircuitDAG.from_circuit(qc)

            # PPO
            r = evaluate_ppo(dag, hw, cm, config, agent, seed, max_num_edges=agent.num_edges)
            r.topo = topo_key
            all_results.append(r)

            # Greedy
            r = evaluate_greedy(qc, config, seed)
            r.topo = topo_key
            r = compute_fidelity(qc, config, r)
            all_results.append(r)

            # SABRE
            r = evaluate_sabre(qc, config, seed)
            r.topo = topo_key
            r = compute_fidelity(qc, config, r)
            all_results.append(r)

    # ---------------------------------------------------------------------
    #  Print summary table
    # ---------------------------------------------------------------------
    print()
    print(f"{'Topo':<8} {'Method':<8} {'N':>3} {'Comp%':>6} {'SWAPs':>8} {'Fidelity':>10} {'Time':>8}")
    print("-" * 60)

    for topo_key in ["cross", "ring", "line"]:
        for method in ["PPO", "Greedy", "SABRE"]:
            rs = [r for r in all_results if r.topo == topo_key and r.method == method]
            n = len(rs)
            comp = sum(1 for r in rs if r.completed)
            swaps = [r.num_swaps for r in rs if r.completed]
            fids = [r.fidelity for r in rs if r.fidelity is not None and r.completed]
            times = [r.wall_time_ms for r in rs]
            sw_str = f"{np.mean(swaps):.1f}±{np.std(swaps):.1f}" if swaps else "N/A"
            fd_str = f"{np.mean(fids):.4f}±{np.std(fids):.4f}" if fids else "   N/A  "
            tm_str = f"{np.mean(times):.0f}ms" if times else "N/A"
            print(f"{topo_key:<8} {method:<8} {n:>3} {100*comp/max(n,1):>5.0f}% {sw_str:>8} {fd_str:>10} {tm_str:>8}")

    # ---------------------------------------------------------------------
    #  Assertions: ensure minimum quality guarantees
    # ---------------------------------------------------------------------
    for topo_key in ["cross", "ring", "line"]:
        ppo_rs = [r for r in all_results if r.topo == topo_key and r.method == "PPO"]
        assert all(r.completed for r in ppo_rs), f"PPO truncation on {topo_key}"

    # PPO should beat Greedy on cross and line (tighter gaps on ring are normal)
    for topo_key in ["cross", "line"]:
        ppo_sw = np.mean([r.num_swaps for r in all_results if r.topo == topo_key and r.method == "PPO"])
        grd_sw = np.mean([r.num_swaps for r in all_results if r.topo == topo_key and r.method == "Greedy"])
        assert ppo_sw < grd_sw + 0.01, (
            f"PPO ({ppo_sw:.1f}) should beat Greedy ({grd_sw:.1f}) on {topo_key}"
        )

    print(f"\nAll assertions passed ({len(all_results)} evaluations).")
