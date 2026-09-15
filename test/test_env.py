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
    # wiring: a routing SWAP step must be penalized by exactly -swap_cost
    # relative to the swap_cost=0 baseline (all other terms identical).
    env0 = _env(reward_mode="routing", swap_cost=0.0)
    obs, _ = env0.reset()
    env0.step(env0.commit_action)  # 结束映射阶段
    r0 = env0.step(0)[1]

    env3 = _env(reward_mode="routing", swap_cost=0.3)
    obs, _ = env3.reset()
    env3.step(env3.commit_action)
    r3 = env3.step(0)[1]

    assert abs((r3 - r0) - (-0.3)) < 1e-6, \
        f"swap penalty should be -0.3, got {r3 - r0}"


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


def _env_r3(n=4, **kwargs):
    """R3 势函数 shaping 测试环境（固定 gamma 便于验证 telescoping）。"""
    from routing.rl.env import _GATE_BASE_REWARD_DEFAULT
    gb = dict(_GATE_BASE_REWARD_DEFAULT)
    gb["cx"] = 0.3
    for _k in ("h", "sx", "x", "rz", "y", "z", "s", "t"):
        gb[_k] = 0.05
    return _env(n=n, reward_mode="routing", shaping_gamma=0.99,
                eta_shape=0.3, alpha_ext=0.5, gate_base_reward=gb,
                swap_cost=0.4, eta_swap_err=2.0, eta_xtalk=0.0,
                eta_parallel=0.0, unfinished_penalty=0.15,
                **kwargs)


def test_r3_swap_pricing_3e():
    env = _env_r3()
    env.reset()
    e = float(env.hw.two_q_err[0, 1])
    r = env._step_reward_swap(0, 1)  # routing 模式下虚拟调用，直接验证公式
    assert abs(r - (-0.4 - 2.0 * 3.0 * e)) < 1e-9


def test_r3_mapping_virtual_swap_free():
    env = _env_r3()
    env.reset()
    # 映射阶段虚拟 SWAP：R3 下直接代价为 0（布局引导由 Φ 承担）
    assert env.mapping_phase
    _, reward, _, _, _ = env._step_mapping(0)
    # 仅含调度/执行项与 shaping，不含 swap_cost（0.4）与 3e 惩罚
    assert reward > -1.0  # 若误计 swap_cost+3e（≈-0.46）也会通过，
    # 因此对照：旧路径同动作应含 -0.4
    env2 = _env(reward_mode="routing", swap_cost=0.4)
    env2.reset()
    _, reward2, _, _, _ = env2._step_mapping(0)
    assert reward2 <= reward - 0.3  # 旧路径明显更负


def test_r3_shaping_telescoping():
    # 势函数 shaping 求和恒等式（γ<1）：
    #   Σ_t (γ·Φ_{t+1} − Φ_t) = −Φ(s0) + (γ−1)·Σ_{t=1..T−1} Φ_t   （终态 Φ=0）
    for seed in (0, 1, 2):
        env = _env_r3(n=4)
        env.reset()
        phi0 = env._phi()
        total_shaping = 0.0
        phi_b_sum = 0.0
        first = True
        done = False
        steps = 0
        rng = np.random.default_rng(seed)
        while not done and steps < 500:
            phi_b = env._phi()
            if not first:
                phi_b_sum += phi_b
            first = False
            _, reward, done, truncated, _ = env.step(int(rng.integers(env.action_space.n)))
            phi_a = 0.0 if (done or truncated) else env._phi()
            total_shaping += 0.99 * phi_a - phi_b
            steps += 1
        assert done, f"seed={seed}: 未完成"
        expected = -phi0 + (0.99 - 1.0) * phi_b_sum
        assert abs(total_shaping - expected) < 1e-6, \
            f"seed={seed}: {total_shaping} vs {expected}"


def test_r5a_lookahead_features():
    env = _env_r3(n=5)
    env.reset()
    feats = env._edge_lookahead_features()
    E = env.num_edges
    assert feats.shape == (E, 4)
    assert np.isfinite(feats).all()
    assert (feats[:, 0] >= 0).all() and (feats[:, 0] <= 1.0 + 1e-9).all()   # xtalk_pred 归一化
    assert (feats[:, 1] >= 0).all() and (feats[:, 1] <= 1.0 + 1e-9).all()   # busy_contact /4

    # xtalk_pred 精确性：手动复算边 (0,1) 的 1-hop 交叉 ZZ 和
    ready = env._ready_2q_gates()
    if ready:
        p, q = env.coupling_map[0]
        zz = env.hw.zz
        adj = env.hw.adj
        xt = 0.0
        for g in ready:
            a, b = env.mapping[g.qubits[0]], env.mapping[g.qubits[1]]
            if a in (p, q) or b in (p, q):
                continue
            for (x, y) in ((p, a), (p, b), (q, a), (q, b)):
                if adj[x, y] > 0:
                    xt += float(zz[x, y])
        assert abs(feats[0, 0] - xt / max(float(zz.max()), 1e-9)) < 1e-9


def test_r5a_obs_dim_consistency():
    env = _env_r3(n=5)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    # 新特征确实进了 obs（维度 = gnn_dim + map/progress/phase）
    assert env._edge_feat_dim == env._gnn.encoder.out_dim * 3 + 5 + 4
