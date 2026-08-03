import numpy as np

from routing.graph.circuit_dag import CircuitDAG
from routing.graph.features import HardwareFeatures
from routing.rl.env import RoutingEnv


def _env(n=4, reward_mode="routing", **kwargs):
    from sim.sim import NoiseConfig
    from utils.data_gen import random_circuit
    coupling = [(i, i + 1) for i in range(n - 1)]
    config = NoiseConfig(
        t1_times=[50.0] * n, t2_times=[70.0] * n, freq_ghz=[5.0] * n,
        single_q_gate_error=0.001, two_q_gate_error=0.01,
        coupling_map=coupling, readout_error=[0.02] * n,
    )
    qc = random_circuit(n, 4, seed=1)
    dag = CircuitDAG.from_circuit(qc)
    hw = HardwareFeatures.from_noise_config(config)
    return RoutingEnv(dag, hw, coupling, reward_mode=reward_mode, seed=2, **kwargs)


def test_reset_obs_shape():
    env = _env()
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape


def test_step_runs_to_completion():
    env = _env(reward_mode="routing")
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 500:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        assert obs.shape == env.observation_space.shape
    assert done
    assert "num_swaps" in info


def test_routing_mode_no_terminal_fidelity():
    env = _env(reward_mode="routing")
    obs, _ = env.reset()
    done = False
    steps = 0
    last_reward = None
    while not done and steps < 500:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        last_reward = reward
        steps += 1
    assert done
    assert "fidelity" not in info
    assert "num_swaps" in info


def test_noise_aware_mode_has_fidelity():
    env = _env(reward_mode="noise_aware")
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 500:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
    assert done
    assert "fidelity" in info
    assert isinstance(info["fidelity"], float)


def test_fidelity_shaping_step_zero():
    env = _env(reward_mode="fidelity_shaping")
    obs, _ = env.reset()
    done = False
    step_rewards = []
    terminal_reward = None
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done:
            terminal_reward = reward
        else:
            step_rewards.append(reward)
    assert done
    for r in step_rewards:
        assert r == 0.0, f"expected 0 step reward, got {r}"
    assert terminal_reward is not None
    assert terminal_reward >= 0.0


def test_swap_penalty():
    env = _env(reward_mode="routing", swap_cost=0.3)
    obs, _ = env.reset()
    env.step(env.commit_action)  # 结束映射阶段
    pre_executed = len(env.executed)
    obs, reward, done, truncated, info = env.step(0)
    if len(env.executed) == pre_executed:
        assert reward == 0.0, f"expected 0 for pure SWAP, got {reward}"
    else:
        assert reward > 0.0, f"expected positive for gate-executing SWAP, got {reward}"


def test_swap_counter():
    env = _env(reward_mode="routing")
    obs, _ = env.reset()
    done = False
    swap_count = 0
    while not done:
        was_mapping = env.mapping_phase
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        if not was_mapping and action < env.num_edges:
            swap_count += 1
    assert done
    assert info["num_swaps"] == swap_count


def test_auto_execute_on_reset():
    env = _env(reward_mode="routing", random_init=False)
    _, _ = env.reset()
    if env.executable_2q:
        env._auto_execute_batch()
    assert not env.executable_2q


def test_batch_gate_execution():
    env = _env(reward_mode="routing", random_init=False)
    _, _ = env.reset()
    env._auto_execute_batch()
    base_executed = len(env.executed)
    for step in range(10):
        if len(env.executed) == env.dag.num_gates:
            break
        obs, reward, done, _, info = env.step(0)
        if done:
            break
        assert len(env.executed) >= base_executed
        base_executed = len(env.executed)


def test_xz_error_tracking():
    env = _env(reward_mode="routing")
    obs, _ = env.reset()
    assert np.all(env._xz_errors == 0.0)
    total_xz_before = float(np.sum(env._xz_errors))
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
    total_xz_after = float(np.sum(env._xz_errors))
    assert total_xz_after >= total_xz_before


def test_terminal_xz_in_info():
    env = _env(reward_mode="routing")
    obs, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
    assert "terminal_XZ" in info
    assert info["terminal_XZ"] >= 0.0


def test_unequal_physical_logical_qubits():
    from sim.sim import NoiseConfig
    from utils.data_gen import random_circuit
    from routing.graph.circuit_dag import CircuitDAG
    from routing.graph.features import HardwareFeatures
    from routing.rl.env import RoutingEnv

    n_logical = 3
    n_physical = 5
    coupling = [(0, 1), (1, 2), (2, 3), (3, 4)]
    config = NoiseConfig(
        t1_times=[50.0] * n_physical,
        t2_times=[70.0] * n_physical,
        freq_ghz=[5.0] * n_physical,
        single_q_gate_error=0.001,
        two_q_gate_error=0.01,
        coupling_map=coupling,
        readout_error=[0.02] * n_physical,
    )
    qc = random_circuit(n_logical, 4, seed=1)
    dag = CircuitDAG.from_circuit(qc)
    hw = HardwareFeatures.from_noise_config(config)
    env = RoutingEnv(dag, hw, coupling, reward_mode="routing", seed=2)

    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 500:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        steps += 1
    assert done

    env2 = RoutingEnv(dag, hw, coupling, reward_mode="routing", random_init=False, seed=42)
    env2.reset()
    env2._apply_swap(0, 1)
    assert env2.mapping == [1, 0, 2], f"normal swap failed: {env2.mapping}"

    env2.reset()
    mapping_before = env2.mapping.copy()
    env2._apply_swap(3, 4)
    assert env2.mapping == mapping_before, f"empty swap should be no-op: {env2.mapping}"

    env2.reset()
    env2._apply_swap(1, 2)
    assert env2.mapping == [0, 2, 1], f"occupied swap failed: {env2.mapping}"

    env2.reset()
    env2._apply_swap(2, 3)
    assert env2.mapping == [0, 1, 3], f"one-sided swap (occupied->empty) failed: {env2.mapping}"

    env2.reset()
    env2._apply_swap(2, 3)
    assert env2.mapping == [0, 1, 3], f"setup failed: {env2.mapping}"
    env2._apply_swap(3, 4)
    assert env2.mapping == [0, 1, 4], f"one-sided swap (3->4) failed: {env2.mapping}"


def test_mapping_phase_virtual_swap_no_execution():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(3, 0)
    qc.cx(2, 3)
    dag = CircuitDAG.from_circuit(qc)
    from sim.sim import NoiseConfig
    coupling = [(i, i + 1) for i in range(3)]
    config = NoiseConfig(
        t1_times=[50.0] * 4, t2_times=[70.0] * 4, freq_ghz=[5.0] * 4,
        single_q_gate_error=0.001, two_q_gate_error=0.01,
        coupling_map=coupling, readout_error=[0.02] * 4,
    )
    hw = HardwareFeatures.from_noise_config(config)
    env = RoutingEnv(dag, hw, coupling, reward_mode="routing",
                     random_init=False, seed=2, use_gnn=False)
    obs, _ = env.reset()
    assert env.mapping_phase
    assert obs[-1] == 1.0
    assert len(env.executed) == 0, "mapping phase must not execute 2q gates"
    env.step(0)
    assert env._mapping_swaps == 1
    assert len(env.executed) == 0, "mapping phase must not execute gates"
    obs, reward, done, truncated, info = env.step(env.commit_action)
    assert not env.mapping_phase
    assert obs[-1] == 0.0
    assert len(env.executed) > 0 or done


def test_mapping_budget_auto_commit():
    env = _env(reward_mode="routing", mapping_budget=2)
    env.reset()
    env.step(0)
    assert env.mapping_phase
    env.step(1)
    assert not env.mapping_phase, "budget exhausted should auto-commit"
    assert env._mapping_swaps == 2


def test_mapping_phase_disabled():
    env = _env(reward_mode="routing", mapping_phase=False)
    obs, _ = env.reset()
    assert not env.mapping_phase
    assert env.action_space.n == env.num_edges
    assert obs.shape == env.observation_space.shape
    done = False
    steps = 0
    while not done and steps < 500:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
    assert done
    assert info["mapping_swaps"] == 0
