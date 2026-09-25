"""test_v0_noise — N1 噪声感知 rollout 的对齐单测。

对齐协议验证（doc/train.md 2026-09-25 N1 方案·七维对齐协议）：
  1. 调度一致性：rollout 内部时间线 == schedule_phys_circuit_events 的 ASAP
  2. 机制解析对拍：单机制代价 vs 手算（depol/ZZ 二次/热弛豫/动态串扰）
  3. 端到端：rollout → 物理电路 → v3 保真度可运行
"""

import math

import numpy as np
import pytest
from qiskit import QuantumCircuit

from routing.graph.circuit_dag import CircuitDAG
from routing.v0.baselines import load_topo_full, make_env
from routing.v0.noise_rollout import (NoiseTimeline, GATE_DUR,
                                      noise_aware_rollout,
                                      phys_circuit_from_ops)
from sim.sim import NoiseConfig
from sim.trajectory_sim_v2 import schedule_phys_circuit_events


def tiny_config(n=4, e2=0.01, zz=0.01, t1=50.0, t2=70.0):
    cm = [(i, i + 1) for i in range(n - 1)]
    tqe = {(p, q): e2 for (p, q) in cm}
    tqe.update({(q, p): e2 for (p, q) in cm})
    cs = {(p, q): zz for (p, q) in cm}
    cs.update({(q, p): zz for (p, q) in cm})
    return NoiseConfig(
        t1_times=[t1] * n, t2_times=[t2] * n, freq_ghz=[5.0] * n,
        single_q_gate_error=0.001, two_q_gate_error=tqe,
        coupling_map=cm, readout_error=[0.02] * n, shots=1024,
        crosstalk_strength=cs)


def make_dag(nq, pairs):
    qc = QuantumCircuit(nq)
    for (a, b) in pairs:
        qc.cx(a, b)
    return CircuitDAG.from_circuit(qc)


CM4 = [(0, 1), (1, 2), (2, 3)]


class TestMechanisms:
    def test_single_cx_mechanisms(self):
        cfg = tiny_config(e2=0.02, zz=0.01, t1=50.0, t2=70.0)
        tl = NoiseTimeline(4, cfg)
        tl.add_op("cx", (0, 1))
        acc = tl.finish()
        # depol2: (15/16)(3/4)(0.02)
        assert abs(acc.depol2 - (15 / 16) * 0.75 * 0.02) < 1e-12
        # 静态 ZZ: 4 sin²(0.01)/5
        assert abs(acc.zz_static - 4 * math.sin(0.01) ** 2 / 5) < 1e-12
        # 动态串扰 = 0（无并发）
        assert acc.zz_dyn == 0.0
        # 热弛豫: cx 0.3µs + 尾部 (total_end=0.3 起) 两比特 + 其余比特全程
        exp_01 = 0.3 / (3 * 70) + 0.3 / (6 * 50)
        tail01 = (0.3 - 0.3) / (3 * 70) + 0.0
        tail23 = 0.3 / (3 * 70) + 0.3 / (6 * 50)
        assert abs(acc.thermal - (2 * exp_01 + 2 * tail23)) < 1e-9

    def test_zz_quadratic_not_linear(self):
        cfg = tiny_config(zz=0.01)
        tl1 = NoiseTimeline(4, cfg)
        tl1.add_op("cx", (0, 1)); tl1.finish()
        cfg2 = tiny_config(zz=0.02)
        tl2 = NoiseTimeline(4, cfg2)
        tl2.add_op("cx", (0, 1)); tl2.finish()
        # 二次：角度×2 → 代价 ×~4（sin² 的三阶修正使比值略小于 4）
        ratio = tl2.acc.zz_static / tl1.acc.zz_static
        assert abs(ratio - 4.0) < 0.01 * 4.0

    def test_swap_three_cx(self):
        cfg = tiny_config(e2=0.02, zz=0.01)
        tl = NoiseTimeline(4, cfg)
        tl.add_op("swap", (0, 1))
        tl_cx = NoiseTimeline(4, cfg)
        for _ in range(3):
            tl_cx.add_op("cx", (0, 1))
        # depol/ZZ：swap = 3×cx（热弛豫不同：swap dur=0.9 vs cx 3×0.3）
        assert abs(tl.acc.depol2 - tl_cx.acc.depol2) < 1e-12
        assert abs(tl.acc.zz_static - tl_cx.acc.zz_static) < 1e-12

    def test_dynamic_xtalk_overlap(self):
        # line4 上 (0,1) 与 (2,3) 时间重叠：1-hop 相邻对 (1,2) 存在 → 有串扰
        cfg = tiny_config(zz=0.015)
        tl2 = NoiseTimeline(4, cfg)
        tl2.add_op("cx", (0, 1))            # 0-0.3, in_flight
        tl2.add_op("cx", (2, 3))            # 0-0.3 重叠；相邻对 (1,2)
        # ov = min(0.3,0.3)-max(0,0) = 0.3；angle = 0.015/0.3·0.3 = 0.015
        expect = (4 / 5) * 0.015 ** 2
        assert abs(tl2.acc.zz_dyn - expect) < 1e-12

    def test_dynamic_xtalk_shared_qubit_zero(self):
        # 共享比特的并发对不产生串扰（v3 语义：不相交事件对）
        cfg = tiny_config(zz=0.015)
        tl = NoiseTimeline(4, cfg)
        tl.add_op("cx", (0, 1))   # 0-0.3
        tl.add_op("cx", (1, 2))   # start=0.3（共享比特 1 互斥）→ 无重叠
        assert tl.acc.zz_dyn == 0.0

    def test_dynamic_xtalk_adjacent_pair(self):
        """真相邻对：line4 的 (0,1) 与 (2,3) 经耦合边 (1,2) 形成 1-hop 对。"""
        cfg = tiny_config(n=4, zz=0.015)
        tl = NoiseTimeline(4, cfg)
        tl.in_flight = [((0, 1), 0.0, 0.3)]
        sq = tl._marginal_zz_sq((2, 3), 0.0, 0.3)
        # 对 (1,2)：θ=0.015, rate=θ/0.3, ov=0.3 → angle=0.015 → sq=(0.015)²
        assert abs(sq - 0.015 ** 2) < 1e-12
        # 共享比特 → 跳过
        tl.in_flight = [((1, 2), 0.0, 0.3)]
        assert tl._marginal_zz_sq((2, 3), 0.0, 0.3) == 0.0
        tl.in_flight = [((0, 1), 0.0, 0.3)]
        assert tl._marginal_zz_sq((1, 2), 0.0, 0.3) == 0.0


class TestScheduleConsistency:
    def test_rollout_timeline_matches_eval_asap(self):
        """对齐协议维度 3：rollout 内部时间线 == 评估侧 ASAP。"""
        topo, cm, hw, cfg = load_topo_full("traindata/topo/line_4q.json")
        dag = make_dag(4, [(0, 2), (1, 3), (0, 2)])
        env = make_env(dag, cm)
        cost, ok, tl = noise_aware_rollout(env.clone(), cfg)
        assert ok
        # 评估侧：同 op 序列的物理电路 → ASAP
        phys = phys_circuit_from_ops(tl.ops, 4)
        evs = schedule_phys_circuit_events(phys)
        assert len(evs) == len(tl.ops)
        for (op, qs, s_roll, e_roll), (s_ev, e_ev, name, qs_ev, _i) in zip(
                tl.ops, evs):
            assert set(qs) == set(qs_ev), f"{op} qubits mismatch"
            assert abs(s_roll - s_ev) < 1e-9, f"{op} start {s_roll} vs {s_ev}"
            assert abs(e_roll - e_ev) < 1e-9, f"{op} end {e_roll} vs {e_ev}"

    def test_thermal_matches_v2_accounting(self):
        """对齐协议维度 7：空闲/尾部热弛豫记账 == v2 的逐比特口径。"""
        cfg = tiny_config(t1=50.0, t2=70.0)
        tl = NoiseTimeline(4, cfg)
        tl.add_op("cx", (0, 1))   # 0-0.3
        tl.add_op("cx", (2, 3))   # start=0（比特 2,3 空闲）——0-0.3
        tl.add_op("cx", (1, 2))   # start=max(0.3,0.3)=0.3
        acc = tl.finish()
        # 门内：3 个 cx × 2 比特 × 0.3µs = 6×per(0.3)
        # 尾部：total_end=0.6；比特 0 (last 0.3) 尾 0.3；比特 3 (last 0.3) 尾 0.3
        #       比特 1,2 (last 0.6) 无尾
        per = lambda t: t / (3 * 70) + t / (6 * 50)
        expect = 6 * per(0.3) + 2 * per(0.3)
        assert abs(acc.thermal - expect) < 1e-9


class TestEndToEnd:
    def test_rollout_and_v3_fidelity(self):
        topo, cm, hw, cfg = load_topo_full("traindata/topo/line_4q.json")
        dag = make_dag(4, [(0, 2), (1, 3), (0, 2)])
        env = make_env(dag, cm)
        cost, ok, tl = noise_aware_rollout(env.clone(), cfg)
        assert ok and cost > 0
        phys = phys_circuit_from_ops(tl.ops, 4)
        from sim.trajectory_sim_v3 import trajectory_circuit_fidelity_events_v3
        fid = trajectory_circuit_fidelity_events_v3(phys, cfg,
                                                    num_trajectories=8,
                                                    seed=0)
        assert 0.0 < fid <= 1.0
