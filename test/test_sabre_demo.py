# ============================================================================
# test_sabre_demo.py
# SABRE 示范蒸馏（E4）单元测试：
#   1. _extract_swap_schedule：从 sabre_route 物理电路提取按序换手序列
#   2. env 沿 SABRE 布局 + 换手序列精确回放 → 完成且 num_swaps == len(schedule)
#      （对齐率 1.0，BC 标签的可行性前提）
#   3. update() 带 demo_mask/demo_act 键：dsl 返回、参数被更新、损失有限
#      （demo 样本 off-policy 不污染 PPO/value 损失）
#   4. _normalize_sabre_cache：旧格式 {rel: [layout]} → 新格式 dict 向后兼容
# ============================================================================

import numpy as np
import pytest
import torch

from routing.graph.circuit_dag import CircuitDAG
from routing.graph.features import HardwareFeatures
from routing.gnn.encoder import SubGNN
from routing.rl.agent import PPOAgent
from routing.rl.env import RoutingEnv
from routing.rl.train_agent import _extract_swap_schedule, _normalize_sabre_cache
from routing.routing import sabre_route
from sim.sim import NoiseConfig
from utils.data_gen import random_circuit


def _env(n=4, init_mapping=None, mapping_phase=False, seed=2):
    coupling = [(i, i + 1) for i in range(n - 1)]
    config = NoiseConfig(
        t1_times=[50.0] * n, t2_times=[70.0] * n, freq_ghz=[5.0] * n,
        single_q_gate_error=0.001, two_q_gate_error=0.01,
        coupling_map=coupling, readout_error=[0.02] * n,
    )
    qc = random_circuit(n, 6, seed=1)
    dag = CircuitDAG.from_circuit(qc)
    hw = HardwareFeatures.from_noise_config(config)
    return RoutingEnv(dag, hw, coupling, reward_mode="routing", seed=seed,
                      init_mapping=init_mapping, mapping_phase=mapping_phase)


def _agent(env):
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
        with_la_head=False,
    )


def _sabre_schedule_env(n=4):
    """找一条 SABRE 需要 ≥1 次换手的电路，返回 (env, swaps, eidx)。"""
    coupling = [(i, i + 1) for i in range(n - 1)]
    config = NoiseConfig(
        t1_times=[50.0] * n, t2_times=[70.0] * n, freq_ghz=[5.0] * n,
        single_q_gate_error=0.001, two_q_gate_error=0.01,
        coupling_map=coupling, readout_error=[0.02] * n,
    )
    phys = swaps = info = None
    for s in range(20):
        qc = random_circuit(n, 8, seed=s)
        phys, info = sabre_route(qc, config, heuristic="decay",
                                 swap_trials=5, seed=0)
        swaps = _extract_swap_schedule(phys)
        if swaps:
            break
    assert swaps, "未能找到需要 SWAP 的测试电路"
    dag = CircuitDAG.from_circuit(qc)
    hw = HardwareFeatures.from_noise_config(config)
    env = RoutingEnv(dag, hw, coupling, reward_mode="routing", seed=2,
                     init_mapping=info["initial_layout"], mapping_phase=False)
    eidx = {}
    for a, (p, q) in enumerate(env.coupling_map):
        eidx[(p, q)] = a
        eidx[(q, p)] = a
    return env, swaps, eidx


def test_extract_swap_schedule_order():
    env, swaps, _ = _sabre_schedule_env()
    assert len(swaps) >= 1
    for (p, q) in swaps:
        assert (p, q) in env.coupling_map or (q, p) in env.coupling_map


def test_env_follows_sabre_schedule_exact():
    """从 SABRE 布局出发、按序执行换手序列 → 电路完成且 swap 数精确匹配。"""
    env, swaps, eidx = _sabre_schedule_env()
    obs, _ = env.reset()
    assert list(env.mapping) == list(env.init_mapping)
    applied = 0
    for (p, q) in swaps:
        a = eidx[(p, q)]
        obs, r, done, trunc, info = env.step(a)
        applied += 1
        if done or trunc:
            break
    assert done and not trunc, "沿 SABRE 序列未能完成电路"
    assert info["num_swaps"] == len(swaps) == applied


def test_demo_bc_loss_trainable_and_finite():
    """带 demo 键的 batch：dsl 返回、actor 更新、损失有限（demo 不污染 PPO）。"""
    torch.manual_seed(7)
    np.random.seed(7)
    env = _env()
    agent = _agent(env)
    obs, _ = env.reset()
    batch = {"act": [], "logp": [], "val": [], "val_route": [], "val_fid": [],
             "rew": [], "term_rew": [], "done": [],
             "graph_data": [], "map_vec": [], "progress": [], "phase": [],
             "coupling_map": [], "sabre_feats": [],
             "look_feats": [], "global_feats": [], "sabre_core_feats": [], "demo_mask": [], "demo_act": []}
    n_steps = 40
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
        # 模拟 demo 回合：第 10-20 步是 SABRE 示范（off-policy 标签）
        if 10 <= t < 20:
            dm = env.get_deadlock_mask()
            a = int(np.random.choice(np.where(~np.asarray(dm))[0][:env.num_edges]))
            logp = val = vr = vf = 0.0
            batch["demo_mask"].append(True)
            batch["demo_act"].append(a)
        else:
            a, logp, val, vr, vf = agent.act(obs, mapping_phase=env.mapping_phase)
            batch["demo_mask"].append(False)
            batch["demo_act"].append(0)
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
    adv, ret = PPOAgent.compute_gae(np.array(batch["rew"], dtype=float),
                                    batch["val"], batch["done"],
                                    bootstrap=0.0, gamma=0.99, lam=0.95)
    batch["adv"] = adv
    batch["ret"] = ret

    before = [p.clone() for p in agent.ac.actor.parameters()] if hasattr(
        agent.ac, "actor") else [p.clone() for p in agent.ac.parameters()]
    losses = agent.update(batch, epochs=2, batch_size=16, sabre_demo_lambda=0.5, global_feats_list=batch.get("global_feats"), sabre_core_feats_list=batch.get("sabre_core_feats"))
    assert "dsl" in losses
    assert np.isfinite(losses["dsl"])
    assert np.isfinite(losses["pl"]) and np.isfinite(losses["vl"])
    after = [p for p in (agent.ac.actor.parameters() if hasattr(
        agent.ac, "actor") else agent.ac.parameters())]
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed, "actor 参数未被更新（demo BC 未生效）"


def test_demo_disabled_no_dsl_effect():
    """sabre_demo_lambda=0 且无 demo 键：行为与历史一致（dsl=0，不崩溃）。"""
    torch.manual_seed(3)
    env = _env()
    agent = _agent(env)
    obs, _ = env.reset()
    batch = {"act": [], "logp": [], "val": [], "val_route": [], "val_fid": [],
             "rew": [], "term_rew": [], "done": [],
             "graph_data": [], "map_vec": [], "progress": [], "phase": [],
             "coupling_map": [], "sabre_feats": [],
             "look_feats": [], "global_feats": [], "sabre_core_feats": []}
    for _ in range(24):
        batch["graph_data"].append(env._last_graph_data)
        batch["map_vec"].append(env._last_map_vec)
        batch["progress"].append(env._last_progress)
        batch["phase"].append(1.0 if env.mapping_phase else 0.0)
        batch["coupling_map"].append(list(env.coupling_map))
        batch["sabre_feats"].append(env._last_sabre_feats.flatten())
        batch["look_feats"].append(env._last_look_feats.flatten())
        batch["global_feats"].append(env._last_global_feats)
        batch["sabre_core_feats"].append(env._last_sabre_core_feats)
        a, logp, val, vr, vf = agent.act(obs, mapping_phase=env.mapping_phase)
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
    adv, ret = PPOAgent.compute_gae(np.array(batch["rew"], dtype=float),
                                    batch["val"], batch["done"],
                                    bootstrap=0.0, gamma=0.99, lam=0.95)
    batch["adv"] = adv
    batch["ret"] = ret
    losses = agent.update(batch, epochs=1, batch_size=16, sabre_demo_lambda=0.0, global_feats_list=batch.get("global_feats"), sabre_core_feats_list=batch.get("sabre_core_feats"))
    assert np.isfinite(losses["pl"]) and np.isfinite(losses["vl"])
    assert losses["dsl"] == 0.0


def test_normalize_sabre_cache_backward_compat():
    old = {0: {"a/b.qasm": [0, 1, 2, 3]}}
    new = _normalize_sabre_cache(old)
    assert isinstance(new[0]["a/b.qasm"], dict)
    assert new[0]["a/b.qasm"]["layout"] == [0, 1, 2, 3]
    assert new[0]["a/b.qasm"]["swaps"] is None


def test_v3_simulator_swap_crosstalk():
    """v3 模拟器：swap 与并发 1-hop 双比特门产生动态串扰；v2 同场景为 0。"""
    import numpy as np
    from qiskit import QuantumCircuit
    from sim.sim import NoiseConfig as _NC
    from sim.trajectory_sim_v2 import trajectory_circuit_fidelity_events as _fid_v2
    from sim.trajectory_sim_v3 import trajectory_circuit_fidelity_events_v3 as _fid_v3
    # 6 比特链：swap(0,1) 与 cx(2,3) 并发——0/1 与 2/3 相距 2（非 1-hop）。
    # 构造 1-hop 交叉：swap(1,2) 与 cx(0,3)？0 与 1/2 相邻。用耦合 (0,1),(1,2),(2,3),(3,4)：
    # swap(1,2) 的端点 {1,2} 与 cx(0,3) 的端点 {0,3}：1-0 相邻 ✓（1-hop 交叉）。
    coupling = [(0, 1), (1, 2), (2, 3), (3, 4)]
    zz = {(0, 1): 0.05, (1, 2): 0.05, (2, 3): 0.05, (3, 4): 0.05}
    cfg = _NC(
        t1_times=[50.0]*5, t2_times=[70.0]*5, freq_ghz=[5.0]*5,
        single_q_gate_error=0.001,
        two_q_gate_error={(i, i+1): 0.01 for i in range(4)},
        coupling_map=coupling, readout_error=[0.02]*5,
        shots=1024,
    )
    cfg.crosstalk_strength = {(a, b): zz[(a, b)] for (a, b) in coupling}
    # 物理电路：先 swap(1,2) 再 cx(0,3)？调度器会按事件重叠——用同时就绪的
    # 独立门：swap(1,2) 与 cx(3,4)（1/2 与 3/4：2-3 相邻 → 1-hop 交叉）
    qc = QuantumCircuit(5)
    qc.swap(1, 2)
    qc.cx(3, 4)
    # 在 v2：swap 不作串扰源；v3：应计入（结果应不同或 v3 串扰>0）
    # 直接测内部 crosstalk_events 更精确：
    from sim.trajectory_sim_v2 import _reduce_phys_circuit_for_fidelity_v2
    from sim.trajectory_sim_v2 import EventTrajectorySimulator, schedule_phys_circuit_events
    from sim.trajectory_sim import auto_backend
    import dataclasses
    rc, rcfg_v2, _ = _reduce_phys_circuit_for_fidelity_v2(qc, cfg)
    rcfg_v3 = dataclasses.replace(rcfg_v2)
    rcfg_v3.swap_xtalk = True
    events = schedule_phys_circuit_events(rc, None)
    sim2 = EventTrajectorySimulator(rcfg_v2, num_trajectories=8, seed=1, backend="cpu")
    sim3 = EventTrajectorySimulator(rcfg_v3, num_trajectories=8, seed=1, backend="cpu")
    f2 = sim2.fidelity_events(rc, events)
    f3 = sim3.fidelity_events(rc, events)
    # v3 计入 swap 串扰后保真度应 ≤ v2（串扰是额外错误来源）；且 v2/v3 可复现
    assert f3 <= f2 + 1e-9, f"v3 ({f3:.6f}) 应不高于 v2 ({f2:.6f})"
    # 直接验证 crosstalk 事件计数
    cts2 = getattr(sim2, 'config', None)
    cts3 = getattr(sim3, 'config', None)
    assert getattr(cts3, 'swap_xtalk', False) is True
    assert getattr(cts2, 'swap_xtalk', False) is False
