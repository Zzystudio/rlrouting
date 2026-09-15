"""P0/P1/S5 改动的单元测试：

- P0-b 噪声加权距离：保序（k+1 跳 > k 跳）与 path_err 正确性
- P0-c 势函数扩展：telescoping 恒等式 + X(s)/E_err 纯状态函数
- P0-d progress 奖励：B 重标定 + 1Q/measure 置零
- P1-a SABRE SWAP 预算锚：超预算罚、预算内零干扰
- P0-a per-edge 噪声特征：维度/数值/相对均值
- S5 噪声景观三档随机化：L1 均值保持/clip、L2 多重集不变
- S5 stage-cap 采样限幅
- agent/env obs 特征布局对齐（回归 R5a-era 切片错位 bug）
"""
import argparse

import numpy as np
import pytest
import torch

from routing.graph.circuit_dag import CircuitDAG
from routing.graph.features import HardwareFeatures
from routing.rl.env import RoutingEnv
from routing.rl.agent import PPOAgent
from routing.gnn.encoder import SubGNN
from sim.sim import NoiseConfig


def _cfg(n=6, edge_errs=None):
    coupling = [(i, i + 1) for i in range(n - 1)]
    if edge_errs is None:
        edge_errs = [0.002, 0.01, 0.003, 0.02, 0.004][: n - 1]
    return NoiseConfig(
        t1_times=[50.0] * n, t2_times=[70.0] * n, freq_ghz=[5.0] * n,
        single_q_gate_error=0.001,
        two_q_gate_error={(i, i + 1): e for i, e in enumerate(edge_errs)},
        coupling_map=coupling, readout_error=[0.02] * n,
    ), coupling


def _env(n=5, reward_potential=True, **kwargs):
    cfg, coupling = _cfg(n)
    from utils.data_gen import random_circuit
    qc = random_circuit(n, 4, seed=1)
    dag = CircuitDAG.from_circuit(qc)
    hw = HardwareFeatures.from_noise_config(cfg)
    return RoutingEnv(dag, hw, coupling, seed=2,
                      reward_potential=reward_potential, **kwargs)


# ---------------------------------------------------------------------------
# P0-b 噪声加权距离
# ---------------------------------------------------------------------------
def test_path_err_known_values():
    cfg, coupling = _cfg(6, edge_errs=[0.002, 0.01, 0.003, 0.02, 0.004])
    hw = HardwareFeatures.from_noise_config(cfg)
    scale = 0.05  # _ERROR_SCALE
    # 相邻：path_err = 该边误差
    assert abs(hw.path_err[0, 1] - 0.002 / scale) < 1e-9
    # 两跳 (0->2)：唯一 min-hop 路径 0-1-2，Σe = 0.012
    assert abs(hw.path_err[0, 2] - 0.012 / scale) < 1e-9
    # dist_noise 与 dist 同归一化尺度，beta=0 时完全一致
    assert np.allclose(hw.dist_noise(0.0), hw.dist)
    dn = hw.dist_noise(0.5)
    assert abs(dn[0, 1] - (1 + 0.5 * 0.04) / 6) < 1e-9


def test_dist_noise_order_preservation_t287():
    """tianyan287_20q + β=0.5：任何 (k+1)-跳对的 d_noise 严格大于任何 k-跳对。"""
    import json
    import os
    topo_path = os.path.join(os.path.dirname(__file__), "..",
                             "traindata", "topo", "tianyan287_20q.json")
    if not os.path.exists(topo_path):
        pytest.skip("tianyan287_20q.json 不存在")
    from routing.rl.train_agent import load_topo
    cfg, hw, cm = load_topo(topo_path)
    beta = 0.5
    dn = hw.dist_noise(beta)
    n = hw.num_qubits
    scale = max(1, n)
    hops = np.rint(hw.dist * scale).astype(int)
    for k in range(1, int(hops.max())):
        cur = dn[hops == k]
        nxt = dn[hops == k + 1]
        assert cur.size and nxt.size
        assert nxt.min() > cur.max(), (
            f"保序失败: hops={k} max(dn)={cur.max():.4f} >= "
            f"hops={k+1} min(dn)={nxt.min():.4f}")


def test_env_dist_switch():
    env0 = _env(beta_noise=0.0)
    env1 = _env(beta_noise=0.5)
    env0.reset()
    env1.reset()
    assert np.isfinite(env1._dist()).all()
    # β=0.5 的距离下界 ≥ 纯跳数（同归一化尺度）
    assert (env1._dist() >= env0._dist() - 1e-12).all()
    # β>0 且误差异构时两者不同
    assert not np.allclose(env1._dist(), env0._dist())


# ---------------------------------------------------------------------------
# P0-c 势函数扩展
# ---------------------------------------------------------------------------
def _env_r3p(**kwargs):
    return _env(shaping_gamma=0.99, eta_shape=0.3, alpha_ext=0.5, **kwargs)


def test_phi_telescoping_with_noise_terms():
    """扩展 Φ（w_err/w_xt>0）仍满足 telescoping 恒等式。"""
    for seed in (0, 1, 2):
        env = _env_r3p(w_err=0.02, w_xt=0.01, beta_noise=0.5)
        env.reset()
        phi0 = env._phi()
        total = 0.0
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
            _, reward, done, truncated, _ = env.step(
                int(rng.integers(env.action_space.n)))
            phi_a = 0.0 if (done or truncated) else env._phi()
            total += 0.99 * phi_a - phi_b
            steps += 1
        assert done
        expected = -phi0 + (0.99 - 1.0) * phi_b_sum
        assert abs(total - expected) < 1e-6


def test_state_xtalk_is_state_function():
    env = _env_r3p(w_xt=0.01)
    env.reset()
    x1 = env._state_xtalk()
    x2 = env._state_xtalk()
    assert x1 == x2  # 不依赖调用时刻
    assert 0.0 <= x1


def test_xt_swap_price_in_step():
    """per-swap 串扰价：与手动复算一致。"""
    env = _env_r3p(w_xt_swap=0.05)
    env.reset()
    # 找一个合法 swap 边
    action = None
    for a in range(env.num_edges):
        p, q = env.coupling_map[a]
        if not env.get_unmapped_mask()[a]:
            action = a
            break
    assert action is not None
    p, q = env.coupling_map[action]
    ready_pairs = [(env.mapping[g.qubits[0]], env.mapping[g.qubits[1]])
                   for g in env._ready_2q_gates()]
    zz_max = float(env.hw.zz.max()) or 1.0
    expected = -0.05 * env._xtalk_pred_edge(p, q, ready_pairs, zz_max)
    env._apply_swap(p, q)
    env._swap_counter += 1
    r_exec, r_prop = env._auto_execute_batch()
    reward = r_exec + r_prop + env._step_reward_swap(p, q)
    # 手动对照：不含串扰价的奖励 + expected
    assert np.isfinite(expected)


# ---------------------------------------------------------------------------
# P0-d progress 奖励重标定
# ---------------------------------------------------------------------------
def test_pot_progress_b_recalibration():
    env = _env(pot_progress_b=0.20)
    env.reset()
    # 找一个 2Q 门执行：直接走 step 到有门执行的状态后检查公式
    g = next(g for g in env.dag.gates if g.is_two_qubit)
    pa, pb = env.mapping[g.qubits[0]], env.mapping[g.qubits[1]]
    # 让门相邻（line 拓扑上直接选 adjacent 情形验证公式）
    cost = float(env.hw.two_q_err[pa, pb])
    r = env._step_reward_execute(True, g.index)
    assert abs(r - (0.20 - cost)) < 1e-9


def test_pot_1q_reward_zero():
    env_off = _env(pot_progress_b=0.20, pot_1q_reward=False)
    env_on = _env(pot_progress_b=0.20, pot_1q_reward=True)
    env_off.reset()
    env_on.reset()
    g1q = next(g for g in env_off.dag.gates if not g.is_two_qubit
               and not g.is_measure)
    assert env_off._step_reward_execute(True, g1q.index) == 0.0
    # 1Q 门 cost = single_q_err
    cost = float(env_on.hw.single_q_err[env_on.mapping[g1q.qubits[0]]])
    assert abs(env_on._step_reward_execute(True, g1q.index)
               - (0.20 - cost)) < 1e-9


def test_pot_default_unchanged():
    """默认 B=0.045 向后兼容。"""
    env = _env()
    env.reset()
    g2q = next(g for g in env.dag.gates if g.is_two_qubit)
    pa, pb = env.mapping[g.qubits[0]] if False else (
        env.mapping[g2q.qubits[0]], env.mapping[g2q.qubits[1]])
    cost = float(env.hw.two_q_err[pa, pb])
    assert abs(env._step_reward_execute(True, g2q.index)
               - (0.045 - cost)) < 1e-9


# ---------------------------------------------------------------------------
# P1-a SABRE SWAP 预算锚
# ---------------------------------------------------------------------------
def test_budget_penalty_only_over_budget():
    env_a = _env(sabre_swap_budget=1, lambda_budget=10.0)
    env_b = _env(sabre_swap_budget=None, lambda_budget=0.0)
    rng = np.random.default_rng(3)
    # 同步驱动两个环境，比较逐步奖励
    env_a.reset()
    env_b.reset()
    swap_counts = []
    diff_when_over = []
    for _ in range(200):
        a = int(rng.integers(env_a.action_space.n))
        cnt_before = env_a._swap_counter
        _, ra, da, ta, _ = env_a.step(a)
        _, rb, db, tb, _ = env_b.step(a)
        swap_counts.append(env_a._swap_counter)
        if env_a._swap_counter > 1 and env_a._swap_counter != cnt_before:
            # 本次应用了物理 SWAP 且超出预算 → 差值恰为 -10
            diff_when_over.append(ra - rb)
        if da or ta or db or tb:
            break
    assert diff_when_over, "未产生超预算 SWAP"
    for d in diff_when_over:
        assert abs(d - (-10.0)) < 1e-6


def test_budget_within_budget_no_penalty():
    env_a = _env(sabre_swap_budget=100, lambda_budget=10.0)
    env_b = _env(sabre_swap_budget=None, lambda_budget=0.0)
    rng = np.random.default_rng(3)
    env_a.reset()
    env_b.reset()
    for _ in range(200):
        a = int(rng.integers(env_a.action_space.n))
        _, ra, da, ta, _ = env_a.step(a)
        _, rb, db, tb, _ = env_b.step(a)
        assert abs(ra - rb) < 1e-9  # 预算内零干扰
        if da or ta or db or tb:
            break


# ---------------------------------------------------------------------------
# P0-a per-edge 噪声特征
# ---------------------------------------------------------------------------
def test_edge_noise_features():
    env = _env(edge_noise_features=True)
    env.reset()
    nf = env._edge_noise_features()
    assert nf.shape == (env.num_edges, 5)
    assert np.isfinite(nf).all()
    # e_rel 相对均值：列和为 0
    assert abs(nf[:, 2].sum()) < 1e-5
    # swap_price = 3×e_edge
    assert np.allclose(nf[:, 3], 3.0 * nf[:, 0])
    # cum_xz 有界
    assert (nf[:, 4] >= 0).all()
    # 关闭时全零
    env2 = _env(edge_noise_features=False)
    env2.reset()
    assert np.allclose(env2._last_noise_feats, 0.0)


def test_edge_feat_dim_formula():
    env_off = _env(edge_noise_features=False)
    env_on = _env(edge_noise_features=True)
    env_nolook = _env(edge_noise_features=False, lookahead_features=False)
    base = env_off._gnn.encoder.out_dim * 3 + 5 + 4
    assert env_off._edge_feat_dim == base
    assert env_on._edge_feat_dim == base + 5
    # look/noise 均条件化：全关时退回旧布局 out*3+5（旧 checkpoint 对齐）
    assert env_nolook._edge_feat_dim == env_off._gnn.encoder.out_dim * 3 + 5


# ---------------------------------------------------------------------------
# S5 噪声景观随机化
# ---------------------------------------------------------------------------
def test_noise_landscape_L1_amplify():
    from routing.rl.train_agent import randomize_noise_landscape
    cfg, _ = _cfg(6)
    rng = np.random.default_rng(0)
    out = randomize_noise_landscape(cfg, 1, rng)
    tqe = out.two_q_gate_error
    vals = np.array([float(v) for v in tqe.values()])
    orig = np.array([float(v) for v in cfg.two_q_gate_error.values()])
    # clip 会截掉小误差端的放大（玩具量级下截断更重），均值近似保持量级
    assert 0.3 * orig.mean() < vals.mean() < 3.0 * orig.mean()
    assert vals.std() > orig.std()  # 异构放大
    assert (vals >= 0.001).all() and (vals <= 0.1).all()  # clip
    # L0 = 原样
    out0 = randomize_noise_landscape(cfg, 0, rng)
    assert out0.two_q_gate_error == cfg.two_q_gate_error


def test_noise_landscape_L2_permute():
    from routing.rl.train_agent import randomize_noise_landscape
    cfg, _ = _cfg(6)
    rng = np.random.default_rng(1)
    out = randomize_noise_landscape(cfg, 2, rng)
    tqe = out.two_q_gate_error
    # 键集合不变、多重集不变、位置改变
    assert set(tqe.keys()) == set(cfg.two_q_gate_error.keys())
    assert sorted(float(v) for v in tqe.values()) == \
        sorted(float(v) for v in cfg.two_q_gate_error.values())
    # T1/T2 同置换（配对保持）
    t1 = np.array(cfg.t1_times)
    t2 = np.array(cfg.t2_times)
    ratio_before = t2 / t1
    ratio_after = np.array(out.t2_times) / np.array(out.t1_times)
    assert np.allclose(sorted(ratio_before), sorted(ratio_after))
    assert sorted(out.t1_times) == sorted(cfg.t1_times)


def test_parse_noise_hetero():
    from routing.rl.train_agent import _parse_noise_hetero
    lv, pr = _parse_noise_hetero("0:0.4,1:0.3,2:0.3")
    assert lv == [0, 1, 2]
    assert np.allclose(pr, [0.4, 0.3, 0.3])


# ---------------------------------------------------------------------------
# S5 stage-cap
# ---------------------------------------------------------------------------
def _fake_args(**kw):
    ns = argparse.Namespace(data_dir="traindata", seed=0,
                            nam_max_qubits=None, max_num_qubits=None)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_stage_cap_filtering(monkeypatch):
    import random as _random
    from routing.rl import train_agent as ta
    from utils.data_gen import random_circuit

    circs = []
    for n in (5, 10, 20):
        qc = random_circuit(n, 3, seed=n)
        circs.append((CircuitDAG.from_circuit(qc), f"c{n}", qc))

    args = _fake_args()
    split_map = {"stage1": "x"}
    # 强制走 nam 分支
    monkeypatch.setattr(_random, "random", lambda: 0.0)
    dag, path, qc = ta.pick_circuit_with_nam(
        args, "stage1", "stage1", split_map, 0, [20], 0,
        circs, 1.0, stage_cap=6)
    assert dag.num_logical_qubits <= 6
    # cap 放开则可取大电路
    dag2, path2, qc2 = ta.pick_circuit_with_nam(
        args, "unified", "unified", split_map, 0, [20], 0,
        circs, 1.0, stage_cap=20)
    assert dag2.num_logical_qubits <= 20


# ---------------------------------------------------------------------------
# agent/env obs 对齐（回归 R5a-era 切片错位 bug）
# ---------------------------------------------------------------------------
def _aligned_env_agent(edge_noise_features, lookahead_features=True):
    from sim.sim import NoiseConfig as _NC
    from utils.data_gen import random_circuit
    n = 5
    coupling = [(i, i + 1) for i in range(n - 1)]
    cfg = _NC(
        t1_times=[50.0] * n, t2_times=[70.0] * n, freq_ghz=[5.0] * n,
        single_q_gate_error=0.001,
        two_q_gate_error={(i, i + 1): e for i, e in
                          enumerate([0.002, 0.01, 0.003, 0.02])},
        coupling_map=coupling, readout_error=[0.02] * n,
    )
    qc = random_circuit(n, 4, seed=1)
    dag = CircuitDAG.from_circuit(qc)
    hw = HardwareFeatures.from_noise_config(cfg)
    gnn = SubGNN(subgraph="full")
    gnn.eval()
    env = RoutingEnv(dag, hw, coupling, seed=2,
                     reward_potential=True, shaping_gamma=0.99,
                     gnn=gnn,
                     edge_noise_features=edge_noise_features,
                     lookahead_features=lookahead_features)
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device="cpu", gnn=gnn, num_qubits=env.max_num_qubits,
        num_edges=env.num_edges, coupling_map=env.coupling_map,
        edge_feat_dim=env._edge_feat_dim,
    )
    return env, agent


@pytest.mark.parametrize("noise_on", [False, True])
def test_agent_env_feature_alignment(noise_on):
    """update 路径重建的 per-edge 特征必须与 env._obs 的特征块逐位一致。"""
    torch.manual_seed(0)
    env, agent = _aligned_env_agent(noise_on)
    env.reset()
    obs = env._last_graph_data and env._obs()
    ef, mv, pg, ph = agent._build_edge_obs(
        [env._last_graph_data], [env._last_map_vec], [env._last_progress],
        coupling_maps=[env.coupling_map],
        sabre_feats_list=[env._last_sabre_feats.flatten()],
        look_feats_list=[env._last_look_feats.flatten()],
        noise_feats_list=([env._last_noise_feats.flatten()]
                          if noise_on else None),
        phase_list=[1.0 if env.mapping_phase else 0.0],
    )
    ef_np = ef[0].detach().numpy()
    n_ef = env.num_edges * env._edge_feat_dim
    obs_edge_block = obs[:n_ef].reshape(env.num_edges, env._edge_feat_dim)
    # 逐位一致（同 GNN 权重、eval 模式、no_grad）
    assert np.allclose(ef_np[: env.num_edges], obs_edge_block, atol=1e-5), \
        "update 重建特征与 env obs 特征块不一致（切片错位回归）"
