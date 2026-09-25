"""test_v0 — v0 纯路由环境 / 精确求解器 / SABRE 打分 / 基线的单元测试。

运行:
    PYTHONPATH=src python3 -m pytest test/test_v0.py -q
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit

from routing.graph.circuit_dag import CircuitDAG
from routing.v0.baselines import greedy_dist_policy, make_env, random_policy, run_episode
from routing.v0.exact_solver import ExactSolver, bfs_solve
from routing.v0.pure_env import PureRoutingEnv, raw_hop_distance
from routing.v0.sabre_heuristic import SabreScorer, greedy_rollout, random_rollout


def make_dag(nq, cx_pairs):
    qc = QuantumCircuit(nq)
    for (a, b) in cx_pairs:
        qc.cx(a, b)
    return CircuitDAG.from_circuit(qc)


LINE4 = [(0, 1), (1, 2), (2, 3)]
RING5 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]


def make_env_for(dag, cm):
    return make_env(dag, cm)


# ---------------------------------------------------------------------------
# PureRoutingEnv
# ---------------------------------------------------------------------------
class TestEnv:
    def test_cascade_fixpoint_and_terminal(self):
        # 一条链上 4q：全邻接可执行，初始即 terminal，0 SWAP
        dag = make_dag(4, [(0, 1), (1, 2), (2, 3)])
        env = make_env_for(dag, LINE4)
        assert env.is_terminal()

    def test_non_adjacent_needs_swap(self):
        # 5q ring 上放 (0,2)：距离 2，需要 SWAP
        dag = make_dag(5, [(0, 2)])
        env = make_env_for(dag, RING5)
        assert not env.is_terminal()

    def test_step_mapping_and_reward(self):
        dag = make_dag(5, [(0, 2)])
        env = make_env_for(dag, RING5)
        # 初始 mapping identity: logical0->phys0, logical2->phys2
        # 对边 (0,1) SWAP：logical0 与 logical1 交换 → mapping[0]=1, mapping[1]=0
        e = RING5.index((0, 1))
        obs, r, done, info = env.step(e)
        assert r == -1.0
        assert env.mapping[0] == 1 and env.mapping[1] == 0
        assert env.swap_count == 1
        assert obs is None

    def test_cascade_executes_unblocked_adjacent(self):
        # (0,1) 后跟 (1,2)，初始都邻接 → 构造时级联全执行 → terminal，0 SWAP
        dag = make_dag(4, [(0, 1), (1, 2)])
        env = make_env_for(dag, LINE4)
        assert env.executed_mask == (1 << 0) | (1 << 1)
        assert env.is_terminal()
        # 再执行一步任意 SWAP，级联保持不动点
        env.step(0)
        assert env.executed_mask == (1 << 0) | (1 << 1)
        assert env.is_terminal()

    def test_clone_independence(self):
        dag = make_dag(5, [(0, 2)])
        env = make_env_for(dag, RING5)
        c = env.clone()
        c.step(RING5.index((0, 1)))
        assert c.swap_count == 1 and env.swap_count == 0
        assert c.mapping != env.mapping

    def test_successor_state_matches_step(self):
        dag = make_dag(5, [(0, 2), (1, 3)])
        env = make_env_for(dag, RING5)
        for e in env.legal_actions():
            c = env.clone()
            c.step(e)
            mask2, map2 = env.successor_state(e)
            assert (mask2, map2) == (c.executed_mask, c.mapping), f"edge {e}"

    def test_ready_2q_derived(self):
        dag = make_dag(4, [(0, 1), (2, 3), (1, 2)])
        env = make_env_for(dag, LINE4)
        # 构造时级联：先执行 (0,1),(2,3)，再解锁 (1,2) → 全部执行
        assert env.executed_mask == (1 << 0) | (1 << 1) | (1 << 2)
        assert env.ready_2q() == []
        assert env.is_terminal()


# ---------------------------------------------------------------------------
# ExactSolver
# ---------------------------------------------------------------------------
class TestSolver:
    def test_astar_equals_bfs(self):
        rng = np.random.default_rng(0)
        for trial in range(6):
            nq = 4 if trial < 3 else 5
            cm = LINE4 if nq == 4 else RING5
            pairs = []
            for _ in range(rng.integers(3, 6)):
                a, b = sorted(rng.choice(nq, size=2, replace=False).tolist())
                pairs.append((a, b))
            dag = make_dag(nq, pairs)
            env = make_env_for(dag, cm)
            solver = ExactSolver(dag, cm)
            v_astar = solver.solve(env.executed_mask, env.mapping, max_nodes=200_000)
            v_bfs = bfs_solve(dag, cm, env.executed_mask, env.mapping, max_nodes=200_000)
            assert v_astar == v_bfs, f"trial {trial} pairs={pairs}: astar={v_astar} bfs={v_bfs}"

    def test_terminal_is_zero(self):
        dag = make_dag(4, [(0, 1), (2, 3)])
        env = make_env_for(dag, LINE4)
        solver = ExactSolver(dag, LINE4)
        assert solver.solve(env.executed_mask, env.mapping) == 0

    def test_memo_reuse_consistency(self):
        # 先解深状态（后段），再解浅状态（前段）：结果自洽
        dag = make_dag(5, [(0, 2), (1, 3), (2, 4)])
        env = make_env_for(dag, RING5)
        solver = ExactSolver(dag, RING5)
        # 构造一个中间状态：执行一步任意 SWAP
        e = env.legal_actions()[0]
        env.step(e)
        mid_key = env.state_key()
        v_mid = solver.solve(env.executed_mask, env.mapping)
        assert v_mid is not None
        # 回起点求解，memo 应包含 mid（浅状态搜索可命中）
        env2 = make_env_for(dag, RING5)
        v_start = solver.solve(env2.executed_mask, env2.mapping)
        assert v_start is not None
        assert mid_key in solver.memo
        # 一致性：起点最优 ≤ 经 mid 的路径
        assert v_start <= v_mid + 1

    def test_budget_exceeded_returns_none(self):
        dag = make_dag(5, [(0, 2), (1, 3), (2, 4), (3, 0)])
        env = make_env_for(dag, RING5)
        solver = ExactSolver(dag, RING5)
        v = solver.solve(env.executed_mask, env.mapping, max_nodes=5)
        assert v is None or v >= 0

    def test_optimal_trajectory_decreases_v(self):
        dag = make_dag(5, [(0, 2), (1, 3), (0, 3)])
        env = make_env_for(dag, RING5)
        solver = ExactSolver(dag, RING5)
        v0 = solver.solve_env(env)
        assert v0 is not None and v0 > 0
        actions = solver.optimal_trajectory(env)
        assert actions is not None
        assert len(actions) == v0
        assert env.is_terminal()


# ---------------------------------------------------------------------------
# SABRE scorer / rollout
# ---------------------------------------------------------------------------
class TestSabre:
    def test_scores_shape_and_finite(self):
        dag = make_dag(5, [(0, 2), (1, 3)])
        env = make_env_for(dag, RING5)
        scorer = SabreScorer()
        sc = scorer.scores(env)
        assert sc.shape == (len(RING5),)
        assert np.isfinite(sc[env.legal_actions()]).all()
        # no-op 边（若有）应 +inf
        for i, e in enumerate(RING5):
            p, q = e
            inv = env._inv
            if inv[p] == -1 and inv[q] == -1:
                assert np.isinf(sc[i])

    def test_prior_normalized(self):
        dag = make_dag(5, [(0, 2), (1, 3)])
        env = make_env_for(dag, RING5)
        scorer = SabreScorer()
        p = scorer.prior(env, beta=2.0)
        assert abs(p.sum() - 1.0) < 1e-9
        assert (p[env.legal_actions()] > 0).all()

    def test_greedy_rollout_terminates(self):
        dag = make_dag(5, [(0, 2), (1, 3), (2, 4), (0, 4)])
        env = make_env_for(dag, RING5)
        n, ok = greedy_rollout(env, SabreScorer())
        assert ok and n >= 0

    def test_random_rollout_terminates(self):
        dag = make_dag(4, [(0, 2), (1, 3)])
        env = make_env_for(dag, LINE4)
        n, ok = random_rollout(env, np.random.default_rng(1), max_steps=2000)
        assert ok and n >= 0


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
class TestBaselines:
    def test_greedy_dist_terminates(self):
        dag = make_dag(5, [(0, 2), (1, 3), (2, 4)])
        env = make_env_for(dag, RING5)
        n, ok, _ = run_episode(env, greedy_dist_policy)
        assert ok and n > 0

    def test_random_terminates(self):
        dag = make_dag(4, [(0, 2), (1, 3)])
        env = make_env_for(dag, LINE4)
        n, ok, _ = run_episode(env, random_policy(np.random.default_rng(2)),
                               max_steps=2000)
        assert ok and n > 0

    def test_raw_hop_distance(self):
        d = raw_hop_distance(LINE4, 4)
        assert d[0, 3] == 3 and d[1, 2] == 1 and d[0, 0] == 0


# ---------------------------------------------------------------------------
# v0 MCTS
# ---------------------------------------------------------------------------
from routing.v0.mcts import MCTSConfig, mcts_episode


class TestMCTS:
    def test_mcts0_uniform_random_rollout_terminates(self):
        dag = make_dag(5, [(0, 2), (1, 3), (2, 4)])
        env = make_env_for(dag, RING5)
        cfg = MCTSConfig(sims=60, prior="uniform", value="rollout_random", seed=0)
        n, ok, st, _ = mcts_episode(env, cfg, max_steps=3000)
        assert ok and n > 0

    def test_mcts1_sabre_prior_sabre_rollout_terminates(self):
        dag = make_dag(5, [(0, 2), (1, 3), (2, 4)])
        env = make_env_for(dag, RING5)
        cfg = MCTSConfig(sims=60, prior="sabre", value="rollout_sabre", seed=0)
        n, ok, st, _ = mcts_episode(env, cfg, max_steps=3000)
        assert ok and n > 0

    def test_mcts_oracle_reaches_optimal(self):
        # line_5q 上远距对：V* 已知，Oracle 搜索应达到最优
        for pairs, cm in [([(0, 2), (1, 3)], RING5), ([(0, 2), (2, 4), (1, 3)], RING5)]:
            dag = make_dag(5, pairs)
            env = make_env_for(dag, cm)
            solver = ExactSolver(dag, cm)
            vstar = solver.solve(env.executed_mask, env.mapping)
            assert vstar is not None and vstar > 0
            cfg = MCTSConfig(sims=400, prior="uniform", value="oracle", seed=0)
            n, ok, st, _ = mcts_episode(env, cfg, solver=solver, max_steps=3000)
            assert ok, f"pairs={pairs}"
            assert n == vstar, f"pairs={pairs}: MCTS={n} V*={vstar}"

    def test_mcts_collects_visited_log(self):
        dag = make_dag(5, [(0, 2), (1, 3)])
        env = make_env_for(dag, RING5)
        cfg = MCTSConfig(sims=40, prior="uniform", value="oracle", seed=0)
        solver = ExactSolver(dag, RING5)
        n, ok, st, log = mcts_episode(env, cfg, solver=solver,
                                      collect_log=True, max_steps=3000)
        assert ok and log is not None and len(log) > 0
