# ============================================================================
# test_env.py
# 验证路由环境的 MDP 接口（reset/step/obs 形状/终止）。
# ============================================================================

import numpy as np

from routing.graph.circuit_dag import CircuitDAG
from routing.graph.features import HardwareFeatures
from routing.rl.env import RoutingEnv


def _env(n=4, with_predictor=False):
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
    predictor = None
    if with_predictor:
        from routing.gnn.predictor import MultiGNNTidelityPredictor
        predictor = MultiGNNTidelityPredictor()
        predictor.eval()
    return RoutingEnv(dag, hw, coupling, predictor=predictor, seed=2)


def test_reset_obs_shape():
    env = _env(with_predictor=True)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape


def test_step_runs_to_completion():
    env = _env(with_predictor=True)
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 500:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        assert obs.shape == env.observation_space.shape
    assert done
    assert "fidelity" in info


def test_action_space_includes_execute():
    env = _env()
    # 最后一个动作为「执行」
    assert env.action_space.n == len(env.coupling_map) + 1
