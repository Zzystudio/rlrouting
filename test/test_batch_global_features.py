# ============================================================================
# test_batch_global_features.py
# E13 批效率特征包单元测试：
#   1. 批机会特征（_edge_lookahead_features 第 4/5 维）：虚拟换位后可执行
#      ready 门数与手算一致
#   2. 全局上下文广播特征（_global_context_features）：维度、有限、广播进 obs
#   3. edge_feat_dim 增长（look4→6 + global101）正确
#   4. 零填充 checkpoint 兼容：旧维度权重加载到新架构后，新特征列零权重、
#      logits 逐位不变（旧模型行为不变）
# ============================================================================

import numpy as np
import pytest
import torch

from routing.graph.circuit_dag import CircuitDAG
from routing.graph.features import HardwareFeatures
from routing.gnn.encoder import SubGNN
from routing.rl.agent import PPOAgent, EdgeActorCritic
from routing.rl.env import RoutingEnv, _LOOKAHEAD_FEAT_DIM, _GLOBAL_FEAT_DIM, _SABRE_CORE_FEAT_DIM
from sim.sim import NoiseConfig
from utils.data_gen import random_circuit


def _env(n=4, **kwargs):
    coupling = [(i, i + 1) for i in range(n - 1)]
    config = NoiseConfig(
        t1_times=[50.0] * n, t2_times=[70.0] * n, freq_ghz=[5.0] * n,
        single_q_gate_error=0.001, two_q_gate_error=0.01,
        coupling_map=coupling, readout_error=[0.02] * n,
    )
    qc = random_circuit(n, 4, seed=1)
    dag = CircuitDAG.from_circuit(qc)
    hw = HardwareFeatures.from_noise_config(config)
    return RoutingEnv(dag, hw, coupling, reward_mode="routing", seed=2, **kwargs)


def test_lookahead_dim_is_six():
    env = _env()
    env.reset()
    feats = env._edge_lookahead_features()
    assert feats.shape == (env.num_edges, 6)
    assert _LOOKAHEAD_FEAT_DIM == 6
    # 第 4/5 维（批机会）归一化到 [0,1] / [-1,1]
    assert (feats[:, 4] >= 0).all() and (feats[:, 4] <= 1.0 + 1e-9).all()
    assert (feats[:, 5] >= -1.0 - 1e-9).all() and (feats[:, 5] <= 1.0 + 1e-9).all()


def test_batch_opportunity_matches_manual():
    """对每条边手算虚拟换位后的可执行 ready 门数，与特征第 4 维一致。"""
    env = _env()
    env.reset()
    ready = env._ready_2q_gates()
    if not ready:
        pytest.skip("无 ready 门的状态")
    dist = env._dist()
    ready_pairs = [(env.mapping[g.qubits[0]], env.mapping[g.qubits[1]])
                   for g in ready]
    feats = env._edge_lookahead_features()
    for i, (p, q) in enumerate(env.coupling_map):
        exec_after = 0
        for a, b in ready_pairs:
            na = q if a == p else (p if a == q else a)
            nb = q if b == p else (p if b == q else b)
            if dist[na, nb] <= 1.0:
                exec_after += 1
        assert abs(feats[i, 4] * max(1, len(ready)) - exec_after) < 1e-6


def test_global_context_features_shape_and_broadcast():
    env = _env()
    env.reset()
    gv = env._global_context_features()
    assert gv.shape == (_GLOBAL_FEAT_DIM,)
    assert np.isfinite(gv).all()
    # 前 48 维是 mean 池化、48-95 是 max 池化：有限且非全零（GNN 嵌入非零）
    assert not np.allclose(gv[:48], 0.0)
    assert not np.allclose(gv[48:96], 0.0)


def test_edge_feat_dim_grows_to_expected():
    env = _env(lookahead_features=True, edge_noise_features=True)
    env.reset()
    out3 = env._gnn.encoder.out_dim * 3
    assert env._edge_feat_dim == out3 + 5 + _LOOKAHEAD_FEAT_DIM + 5 + _GLOBAL_FEAT_DIM + _SABRE_CORE_FEAT_DIM
    obs, _ = env.reset()
    assert obs.shape[0] == env._edge_feat_dim * env.max_num_edges + env.max_num_qubits + 2  # map+progress+phase


def _make_ac(edge_feat_dim):
    return EdgeActorCritic(edge_feat_dim=edge_feat_dim, num_edges=4, num_qubits=4,
                           with_commit=True, with_la_head=False)


def test_zero_pad_load_equivalence():
    """旧维度权重 → 新架构：新特征列零权重，前向 logits 与旧网络逐位一致。"""
    torch.manual_seed(7)
    old_ac = _make_ac(edge_feat_dim=10)
    new_ac = _make_ac(edge_feat_dim=13)  # +3 新特征列
    ac_state = {k: v.clone() for k, v in old_ac.state_dict().items()}
    padded = PPOAgent._zero_pad_ac_state(ac_state, new_ac)
    new_ac.load_state_dict(padded)

    # 新特征列必须全零（edge 特征段末尾；critic 输入 = edge_feat_dim+num_qubits+phase，
    # 故 critic 的 edge 段是 [:, 10:13]，其后是 map_vec/progress/phase 保留非零）
    assert torch.all(new_ac.edge_mlp[0].weight[:, 10:] == 0)
    assert torch.all(new_ac.edge_score.weight[:, 10:] == 0)
    assert torch.all(new_ac.commit_head.weight[:, 10:] == 0)
    assert torch.all(new_ac.critic_route[0].weight[:, 10:13] == 0)
    assert torch.all(new_ac.critic_fid[0].weight[:, 10:13] == 0)
    # map_vec/progress/phase 列保留原值（new 中向后移 delta=3 列）
    assert torch.all(new_ac.critic_route[0].weight[:, 13:] == old_ac.critic_route[0].weight[:, 10:])

    # 前向：新特征=0 时 logits 与旧网络一致
    ef_old = torch.randn(1, 4, 10)
    ef_new = torch.zeros(1, 4, 13)
    ef_new[:, :, :10] = ef_old
    mv = torch.randn(1, 4)
    pg = torch.rand(1, 1)
    ph = torch.rand(1, 1)
    with torch.no_grad():
        lo_old = old_ac(ef_old, mv, pg, ph)[0]
        lo_new = new_ac(ef_new, mv, pg, ph)[0]
    assert torch.allclose(lo_old, lo_new, atol=1e-6), "零填充后 logits 应逐位一致"


def test_sabre_core_features_shape_and_bounds():
    env = _env()
    env.reset()
    feats = env._edge_sabre_core_features()
    assert feats.shape == (env.num_edges, 6)
    assert np.isfinite(feats).all()
    # 归一化到 [0,1]
    assert (feats[:, 0] >= 0).all() and (feats[:, 0] <= 1.0 + 1e-9).all()
    assert (feats[:, 1:5] >= 0).all() and (feats[:, 1:5] <= 1.0 + 1e-9).all()
    assert (feats[:, 5] >= 0).all() and (feats[:, 5] <= 1.0 + 1e-9).all()


def test_sabre_decay_paper_after_swap():
    """decay 语义（E15 修复）：per-qubit 换位计数器——swap +1、该 qubit 门执行清零。"""
    env = _env()
    env.reset()
    assert np.allclose(env._sabre_decay_paper(), 1.0)
    p, q = env.coupling_map[0]
    # 模拟一次换位：_apply_swap 递增端点计数器
    env._apply_swap(p, q)
    d1 = env._sabre_decay_paper()
    assert d1[p] == 2.0 and d1[q] == 2.0
    # 门执行（reset helper）→ 该 qubit 清零，其他不受影响
    env._reset_qubit_decay([p])
    d2 = env._sabre_decay_paper()
    assert d2[p] == 1.0 and d2[q] == 2.0


def test_sabre_core_zero_pad_compat():
    """旧 261 维 checkpoint（LA287s era）→ 新 267 维：零填充兼容，sabre_core 列零权重。"""
    torch.manual_seed(11)
    old_ac = EdgeActorCritic(edge_feat_dim=10, num_edges=4, num_qubits=4,
                             with_commit=True, with_la_head=False)
    new_ac = EdgeActorCritic(edge_feat_dim=16, num_edges=4, num_qubits=4,  # +6 sabre_core
                             with_commit=True, with_la_head=False)
    ac_state = {k: v.clone() for k, v in old_ac.state_dict().items()}
    padded = PPOAgent._zero_pad_ac_state(ac_state, new_ac)
    new_ac.load_state_dict(padded)
    assert torch.all(new_ac.edge_mlp[0].weight[:, 10:] == 0)
    assert torch.all(new_ac.critic_route[0].weight[:, 10:16] == 0)
    # extras 保持原位
    assert torch.all(new_ac.critic_route[0].weight[:, 16:] == old_ac.critic_route[0].weight[:, 10:])
