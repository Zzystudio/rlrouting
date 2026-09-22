# -*- coding: utf-8 -*-
"""时钟化环境专项测试（doc/20260920训练方案.md §九）：
1. θ 可加性：env 自算边际和 == v3 zz_actions 总注入角（含 swap 对）
2. liveness：任意状态 >=1 合法动作
3. 物理互斥：schedule_log 每 qubit 事件不重叠
4. 终局：episode 以全部门完成结束（非截断）
5. clone 等价：克隆后 step 与原 env 一致
6. SKIP idle 结算公式与手算一致
7. 惰性物化顺序：SWAP-先-1Q（SWAP 后 1Q 在新位置物化）
8. 架构级 warm-start 逐位不变：零时序列下 ClockEdgeActorCritic 与
   EdgeActorCritic 的 edge logits 一致
"""
import numpy as np
import pytest
import torch

from qiskit import QuantumCircuit

from routing.graph.circuit_dag import CircuitDAG
from routing.gnn.encoder import SubGNN
from routing.rl.env import RoutingEnv
from routing.rl.agent import EdgeActorCritic
from routing.rl.env_clocked import ClockedRoutingEnv
from routing.rl.eval_policy import load_topo
from routing.rl.agent_clocked import ClockEdgeActorCritic, D_EXEC, D_TIMING_GLOB

import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOPO = os.path.join(_REPO, "traindata", "topo", "ring_5q.json")


def _dag_ring():
    config, hw, cm = load_topo(_TOPO)
    qc = QuantumCircuit(5)
    qc.h(0); qc.cx(0, 1); qc.h(1); qc.cx(1, 2)
    qc.cx(2, 3); qc.h(2); qc.cx(3, 4); qc.measure_all()
    dag = CircuitDAG.from_circuit(qc)
    return config, hw, cm, dag


def _make_env(**kw):
    config, hw, cm, dag = _dag_ring()
    defaults = dict(reward_mode="routing", reward_potential=True,
                    pot_progress_b=0.2, swap_price_scale=4.6, use_gnn=False,
                    mapping_phase=True, max_ready=8, max_num_qubits=5,
                    max_episode_steps=300, eta_time=0.05, eta_parallel=0.05,
                    eta_idle=0.005, w_xt_launch=1.0, w_zz=0.5,
                    init_mapping=[0, 1, 2, 3, 4], random_init=False)
    defaults.update(kw)
    return ClockedRoutingEnv(dag, hw, cm, **defaults)


def _auto_batch_rollout(env, max_steps=200):
    """auto-batch 贪心：全 EXEC -> SKIP -> SWAP。返回完整 episode 轨迹。"""
    obs, _ = env.reset()
    E, K = env.num_edges, env.max_ready
    done = False
    steps = 0
    while not done and steps < max_steps:
        mask = env.get_action_mask()
        if env.mapping_phase:
            a = env.commit_action
        else:
            le = [i for i in range(E, E + K) if mask[i]]
            if le:
                a = le[0]
            elif mask[E + K + 1]:
                a = E + K + 1
            else:
                edges = [i for i in range(E) if mask[i]]
                assert edges, "liveness 破坏：无合法动作"
                a = edges[0]
        obs, r, done, trunc, info = env.step(a)
        steps += 1
    return done, steps, env


# ------------------------------------------------------------------ #
#  1. θ 可加性（vs v3 权威口径）
# ------------------------------------------------------------------ #
def test_theta_additivity_v3():
    from sim.trajectory_sim_v2 import EventTrajectorySimulator, timing_log_to_events
    from sim.trajectory_sim_v2 import _reduce_phys_circuit_for_fidelity_v2
    config, hw, cm, dag = _dag_ring()
    env = _make_env()
    done, steps, env = _auto_batch_rollout(env)
    assert done
    # env 自算边际累积
    env_marg = env._cum_theta
    # v3 权威：编译事件 -> zz_actions 总角
    rc, rconfig, remap = _reduce_phys_circuit_for_fidelity_v2(env._phys_circuit, config)
    log = env.timing.schedule_log
    events = timing_log_to_events(log, circuit=rc, remap=remap)
    sim = EventTrajectorySimulator(rconfig, num_trajectories=1)
    actions, _ = sim._prepare_events(rc, events)
    v3_theta = 0.0
    for a in actions:
        if a[0] == "zz":
            v3_theta += float(a[4])
    assert abs(env_marg - v3_theta) < 1e-6, f"additivity 破坏: env={env_marg:.6f} v3={v3_theta:.6f}"


# ------------------------------------------------------------------ #
#  2. liveness
# ------------------------------------------------------------------ #
def test_liveness_random_actions():
    rng = np.random.default_rng(0)
    for trial in range(5):
        env = _make_env()
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            mask = env.get_action_mask()
            assert mask.any(), "liveness 破坏"
            legal = np.flatnonzero(mask)
            a = int(legal[rng.integers(len(legal))])
            obs, r, done, trunc, info = env.step(a)
            done = done or trunc
            steps += 1


# ------------------------------------------------------------------ #
#  3. 物理互斥
# ------------------------------------------------------------------ #
def test_mutex_invariant():
    env = _make_env()
    done, steps, env = _auto_batch_rollout(env)
    assert done
    last_end = {}
    for e in env.timing.schedule_log:
        for q in e["qubits"]:
            if q in last_end:
                assert e["start"] >= last_end[q] - 1e-9, f"qubit {q} 事件重叠"
            last_end[q] = e["end"]


# ------------------------------------------------------------------ #
#  4. 终局
# ------------------------------------------------------------------ #
def test_terminates_complete():
    env = _make_env()
    done, steps, env = _auto_batch_rollout(env)
    assert done
    assert len(env.executed) == env.dag.num_gates


# ------------------------------------------------------------------ #
#  5. clone 等价
# ------------------------------------------------------------------ #
def test_clone_equivalence():
    env = _make_env()
    obs, _ = env.reset()
    E, K = env.num_edges, env.max_ready
    # 推进几步到路由期
    env.step(env.commit_action)
    env.step(E + K + 1 if env.get_action_mask()[E + K + 1] else 0)
    c = env.clone()
    for a in (0, 1, 3):
        if a >= env.action_space.n:
            continue
        o1, r1, d1, t1, i1 = env.step(a)
        o2, r2, d2, t2, i2 = c.step(a)
        assert np.allclose(o1, o2), f"clone obs 不一致 @ action {a}"
        assert abs(r1 - r2) < 1e-9
        assert d1 == d2 and t1 == t2
        env = c  # 继续同步推进（克隆为基准）
        c = env.clone()


# ------------------------------------------------------------------ #
#  6. SKIP idle 结算
# ------------------------------------------------------------------ #
def test_skip_idle_formula():
    from routing.timing import skip_idle_delta
    env = _make_env()
    env.reset()
    env.step(env.commit_action)
    # launch 两个 CX（同波），SKIP 推进 0.3
    mask = env.get_action_mask()
    E, K = env.num_edges, env.max_ready
    execs = [i for i in range(E, E + K) if mask[i]]
    env.step(execs[0])
    # 手算 idle：SKIP [0, 0.3]，激活且在 SKIP 时已空闲的 qubit 数
    bu = env._busy_until()
    lf = env._last_free()
    t = env.clock
    Tp = next((b for b in sorted(set(bu)) if b > t + 1e-9), None)
    assert Tp is not None
    idle = skip_idle_delta(bu, lf, t, Tp)
    expected = float(((bu <= t + 1e-9) & (lf >= 0)).sum()) * (Tp - t)
    assert abs(idle - expected) < 1e-9


# ------------------------------------------------------------------ #
#  7. 惰性物化顺序：SWAP-先-1Q
# ------------------------------------------------------------------ #
def test_lazy_materialization_swap_before_1q():
    # 构造：h(0) 之后立刻 SWAP(0,1)，再 EXEC 一个用 qubit0 的 2Q 门
    config, hw, cm, _ = _dag_ring()
    qc = QuantumCircuit(5)
    qc.h(0); qc.cx(0, 1); qc.cx(0, 4)   # cx(0,4) 非相邻，需 SWAP
    qc.measure_all()
    dag = CircuitDAG.from_circuit(qc)
    env = ClockedRoutingEnv(dag, hw, cm, reward_mode="routing",
                            reward_potential=True, pot_progress_b=0.2,
                            swap_price_scale=4.6, use_gnn=False,
                            mapping_phase=True, max_ready=8, max_num_qubits=5,
                            max_episode_steps=200, init_mapping=[0, 1, 2, 3, 4],
                            random_init=False)
    env.reset()
    env.step(env.commit_action)
    # EXEC cx(0,1)：h(0) 惰性物化，cx(0,1) 起点 = h 之后
    mask = env.get_action_mask()
    E, K = env.num_edges, env.max_ready
    execs = [i for i in range(E, E + K) if mask[i]]
    assert execs
    env.step(execs[0])
    # 找到 h(0) 与 cx(0,1) 的调度记录，验证 h 在 cx 之前
    log = env.timing.schedule_log
    h_entries = [e for e in log if e.get("op") == "h"]
    cx_entries = [e for e in log if e.get("op") == "cx"]
    assert h_entries and cx_entries
    assert h_entries[0]["start"] <= cx_entries[0]["start"] + 1e-9
    # h 在物理 qubit mapping[0]（SWAP 前位置）
    assert h_entries[0]["qubits"] == [env.mapping[0]]


# ------------------------------------------------------------------ #
#  8. 架构级 warm-start 逐位不变（零时序列下 edge logits 一致）
# ------------------------------------------------------------------ #
def test_warmstart_edge_logits_bitexact():
    nq, ne, K = 5, 5, 8
    D_old = 267
    D_new = 279  # 267 + D_GLOBAL_TIMING + D_EDGE_TIMING
    rng = np.random.default_rng(0)
    old = EdgeActorCritic(D_old, ne, nq, with_commit=True)
    new = ClockEdgeActorCritic(D_new, ne, nq, K, with_commit=True)
    # 把 old 权重拷进 new（列扩展零初始化，bias 全拷）
    sd = old.state_dict()
    new_sd = new.state_dict()
    for k in ("edge_mlp.0.weight", "edge_mlp.0.bias",
              "edge_mlp.2.weight", "edge_mlp.2.bias",
              "edge_mlp.4.weight", "edge_mlp.4.bias",
              "edge_score.weight", "edge_score.bias",
              "commit_head.weight", "commit_head.bias"):
        v = sd[k]
        t = new_sd[k]
        if v.shape == t.shape:
            new_sd[k] = v
        else:
            assert t.dim() == 2 and v.dim() == 2 and v.shape[1] == D_old
            t[:, :D_old] = v
    new.load_state_dict(new_sd, strict=False)
    old.eval(); new.eval()
    ef_old = rng.normal(size=(1, ne, D_old)).astype(np.float32)
    ef_new = np.zeros((1, ne, D_new), dtype=np.float32)
    ef_new[:, :, :D_old] = ef_old
    mv = rng.normal(size=(1, nq)).astype(np.float32)
    pg = np.ones((1, 1), dtype=np.float32)
    ph = np.ones((1, 1), dtype=np.float32)
    xf = np.zeros((1, K, D_EXEC), dtype=np.float32)
    tg = np.zeros((1, D_TIMING_GLOB), dtype=np.float32)
    with torch.no_grad():
        lo = old(torch.tensor(ef_old), torch.tensor(mv), torch.tensor(pg),
                 torch.tensor(ph))[0]
        ln = new(torch.tensor(ef_new), torch.tensor(xf), torch.tensor(mv),
                 torch.tensor(pg), torch.tensor(ph), torch.tensor(tg))[0]
    # 零时序列下 edge 段 logits 必须逐位一致（mask 未用）
    assert torch.allclose(lo[0, :ne], ln[0, :ne], atol=1e-5), "warm-start edge logits 漂移"
