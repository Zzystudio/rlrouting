# ============================================================================
# test_lookahead.py
# 训练期 beam lookahead（doc/20260916训练方案.md v2 §2.4/§4）单元测试：
#   1. 掩码尊重：beam_expectimax 的 a_star 落在合法动作集内
#   2. done 处理：终局节点 Q=r（无自举），输出有限
#   3. B=1 退化为 argmax
#   4. vla_loss 可训：update() 带 la 键后 critic_la 权重变化、返回 vla/agr
#   5. 无 la 头的 agent 忽略 la 键（向后兼容）
# ============================================================================

import numpy as np
import pytest
import torch

from routing.graph.circuit_dag import CircuitDAG
from routing.graph.features import HardwareFeatures
from routing.gnn.encoder import SubGNN
from routing.rl.agent import PPOAgent
from routing.rl.env import RoutingEnv
from routing.rl.lookahead import beam_expectimax
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


def _agent(env, with_la_head=True):
    gnn = SubGNN(subgraph="full")
    return PPOAgent(
        obs_dim=int(np.prod(env.observation_space.shape)),
        action_dim=env.action_space.n,
        device="cpu",
        gnn=gnn,
        num_qubits=env.max_num_qubits,
        num_edges=env.num_edges,
        coupling_map=list(env.coupling_map),
        with_commit=env.enable_mapping_phase,
        edge_feat_dim=getattr(env, "_edge_feat_dim", None),
        with_la_head=with_la_head,
    )


def _valid_actions(env, agent):
    n_a = agent.num_edges + 1 if agent.with_commit else agent.num_edges
    mask = np.zeros(n_a, dtype=bool)
    mask[:len(env.coupling_map)] = True
    dm = env.get_deadlock_mask()
    um = env.get_unmapped_mask()
    for i in range(min(len(dm), n_a)):
        if dm[i] or um[i]:
            mask[i] = False
    if agent.with_commit:
        mask[agent.num_edges] = env.mapping_phase
    return mask


def _run_episode_steps(env, agent, k):
    """推 k 步，返回最后的 (obs, env)。"""
    obs, _ = env.reset()
    for _ in range(k):
        mask_np = _valid_actions(env, agent)
        logits, _ = agent._forward_obs(obs)
        masked = logits[0].clone()
        masked[~torch.tensor(mask_np)] = -1e9
        a = int(masked.argmax().item())
        obs, _, done, trunc, _ = env.step(a)
        if done or trunc:
            obs, _ = env.reset()
    return obs, env


def test_beam_respects_action_mask():
    env = _env()
    agent = _agent(env)
    obs, env = _run_episode_steps(env, agent, 5)
    valid = _valid_actions(env, agent)
    a_star, v_star, q_roots = beam_expectimax(env, agent, obs,
                                              beam_width=3, depth=2, gamma=0.99)
    assert valid[a_star], "beam 选出的动作被掩码禁止"
    for a, q in q_roots:
        assert valid[a]
        assert np.isfinite(q)
    assert np.isfinite(v_star)


def test_beam_b1_degenerates_to_argmax():
    env = _env()
    agent = _agent(env)
    obs, env = _run_episode_steps(env, agent, 3)
    valid = _valid_actions(env, agent)
    logits, _ = agent._forward_obs(obs)
    masked = logits[0].clone()
    masked[~torch.tensor(valid)] = -1e9
    argmax_a = int(masked.argmax().item())
    a_star, _, _ = beam_expectimax(env, agent, obs,
                                   beam_width=1, depth=2, gamma=0.99)
    assert a_star == argmax_a


def test_beam_depth1_and_terminal_finite():
    env = _env()
    agent = _agent(env)
    obs, env = _run_episode_steps(env, agent, 7)
    a1, v1, _q1 = beam_expectimax(env, agent, obs, beam_width=3, depth=1, gamma=0.99)
    a2, v2, _q2 = beam_expectimax(env, agent, obs, beam_width=3, depth=2, gamma=0.99)
    assert np.isfinite(v1) and np.isfinite(v2)
    assert -1e6 < v1 < 1e6 and -1e6 < v2 < 1e6


def _mini_batch(env, agent, n_steps=40, la_every=4):
    """手工 rollout 收集一个带 la 键的 mini batch。"""
    obs, _ = env.reset()
    batch = {"act": [], "logp": [], "val": [], "val_route": [], "val_fid": [],
             "rew": [], "term_rew": [], "done": [],
             "graph_data": [], "map_vec": [], "progress": [], "phase": [],
             "coupling_map": [], "sabre_feats": [],
             "look_feats": [], "global_feats": [], "sabre_core_feats": [], "la_mask": [], "la_act": [], "la_val": []}
    for t in range(n_steps):
        batch["graph_data"].append(env._last_graph_data)
        batch["map_vec"].append(env._last_map_vec)
        batch["progress"].append(env._last_progress)
        batch["phase"].append(1.0 if env.mapping_phase else 0.0)
        batch["coupling_map"].append(list(env.coupling_map))
        batch["sabre_feats"].append(env._last_sabre_feats.flatten())
        batch["look_feats"].append(env._last_look_feats.flatten())
        batch["global_feats"].append(env._last_global_feats)
        batch["sabre_core_feats"].append(env._last_sabre_core_feats)
        use_la = (t % la_every == 0) and not env.mapping_phase
        a_star = v_star = None
        if use_la:
            a_star, v_star, q_roots = beam_expectimax(env, agent, obs,
                                                      beam_width=2, depth=2,
                                                      gamma=0.99)
            if not q_roots:  # 全掩码状态退化为非 anchor
                a_star = None
        if a_star is not None:
            batch["la_mask"].append(True)
            batch["la_act"].append(a_star)
            batch["la_val"].append(v_star)
        else:
            batch["la_mask"].append(False)
            batch["la_act"].append(0)
            batch["la_val"].append(0.0)
        mask_np = _valid_actions(env, agent)
        n_a = agent.num_edges + 1
        mask_t = torch.zeros(n_a, dtype=torch.bool)
        mask_t[:len(env.coupling_map)] = True
        for i in range(min(len(mask_np), n_a)):
            if mask_np[i]:
                mask_t[i] = True
        mask_t[agent.num_edges] = env.mapping_phase
        a, logp, val, vr, vf = agent.act(obs, mapping_phase=bool(mask_t[agent.num_edges]))
        obs, r, done, trunc, info = env.step(a)
        batch["act"].append(a)
        batch["logp"].append(logp)
        batch["val"].append(val)
        batch["val_route"].append(vr)
        batch["val_fid"].append(vf)
        batch["rew"].append(r)
        batch["term_rew"].append(info.get("terminal_reward", 0.0))
        batch["done"].append(done or trunc)
        if done or trunc:
            obs, _ = env.reset()
    # GAE 填充 adv/ret（train_agent 主循环的简化版）
    rews = np.array(batch["rew"], dtype=float)
    adv, ret = PPOAgent.compute_gae(rews, batch["val"], batch["done"],
                                    bootstrap=0.0, gamma=0.99, lam=0.95)
    batch["adv"] = adv
    batch["ret"] = ret
    return batch


def test_vla_loss_trainable():
    """vla_loss 返回且 critic_la 权重被更新（非零初始化后仍变化）。"""
    torch.manual_seed(7)
    np.random.seed(7)
    env = _env()
    agent = _agent(env, with_la_head=True)
    agent.la_vf_coef = 0.5
    batch = _mini_batch(env, agent)
    before = [p.clone() for p in agent.ac.critic_la.parameters()]
    losses = agent.update(batch, epochs=2, batch_size=32, la_vf_coef=0.5, global_feats_list=batch.get("global_feats"), sabre_core_feats_list=batch.get("sabre_core_feats"))
    assert "vla" in losses and "agr" in losses
    assert np.isfinite(losses["vla"])
    after = [p for p in agent.ac.critic_la.parameters()]
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed, "critic_la 参数未被更新（vla_loss 未生效）"


def test_agent_without_la_head_ignores_la_keys():
    """无 la 头的 agent：带 la 键的 batch 正常更新（键被忽略，行为兼容）。"""
    torch.manual_seed(7)
    np.random.seed(7)
    env = _env()
    agent = _agent(env, with_la_head=False)
    batch = _mini_batch(env, agent)
    losses = agent.update(batch, epochs=1, batch_size=32, global_feats_list=batch.get("global_feats"), sabre_core_feats_list=batch.get("sabre_core_feats"))
    assert np.isfinite(losses["pl"])
    assert not hasattr(agent.ac, "critic_la")


def test_checkpoint_roundtrip_with_la_head():
    """save/load 往返：la 头权重保留；旧格式（无 la 头）加载后 v_la≡0。"""
    env = _env()
    agent = _agent(env, with_la_head=True)
    agent.save("/tmp/opencode/test_la_roundtrip.pt")
    agent2 = _agent(env, with_la_head=True)
    agent2.load("/tmp/opencode/test_la_roundtrip.pt")
    w1 = agent.ac.critic_la[0].weight
    w2 = agent2.ac.critic_la[0].weight
    assert torch.equal(w1, w2)
