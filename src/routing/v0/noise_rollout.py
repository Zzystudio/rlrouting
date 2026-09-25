"""N1 噪声感知 rollout —— 增量 ASAP 时间线 + 四机制 ΔF 解析记账。

与 v2/v3 模拟器逐机制对齐（对齐协议见 doc/train.md 2026-09-25 N1 方案）：

  机制            模拟器语义（trajectory_sim_v2._run_actions / _prepare_events）
  ─────────────────────────────────────────────────────────────────────
  边 depol2      每 CX 一次 depol2(p)；SWAP=3×。ΔF=(15/16)·(3/4)·p
  静态 ZZ        每 CX 旋转 θ·(dur/0.3µs)；SWAP=3×θ。ΔF=4sin²(θ)/5（二次！）
  动态串扰       时间重叠的 1-hop 不相交事件对，角=(θ_ab/0.3)·ov；v3 含 swap 源。
                 ΔF=(4/5)·Σ_pair angle²（不同对 ZZ 可交换→一阶 Σ 平方）
  热弛豫 T1/T2   每比特独立时钟：门前空闲 + 门内 + 电路尾收尾。
                 ΔF≈t/(3T2)+t/(6T1)（标准热弛豫一阶）

关键工程点：
  1. rollout 决策序 ≠ 时间序 → overlap 用一般形式
     ov = min(end,b_end) − max(start,b_start)；每个无序对恰计一次。
  2. 增量 start = max(last_free[q]) == 评估侧 schedule_phys_circuit_events
     的 ASAP 语义（同一 op 序列两侧调度一致——单测锁定）。
  3. rollout 副产品 = 物理电路 op 序列（cx 保留逻辑控制方向）→ 直接喂
     trajectory_circuit_fidelity_events_v3 做真值评估。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .pure_env import PureRoutingEnv

GATE_DUR = {"cx": 0.30, "cz": 0.30, "swap": 0.90, "sx": 0.035, "x": 0.035,
            "rz": 0.0, "z": 0.0, "id": 0.0}
T_CX = 0.30


@dataclass
class NoiseAccounting:
    depol2: float = 0.0
    depol1: float = 0.0
    zz_static: float = 0.0
    zz_dyn: float = 0.0
    thermal: float = 0.0

    @property
    def total(self) -> float:
        return (self.depol2 + self.depol1 + self.zz_static
                + self.zz_dyn + self.thermal)


class NoiseTimeline:
    """增量 ASAP 时间线 + 四机制 ΔF 记账（raw 参数，来自 NoiseConfig）。"""

    def __init__(self, n_phys: int, config):
        self.n = n_phys
        self.last_free = [0.0] * n_phys
        self.in_flight: List[Tuple[Tuple[int, ...], float, float]] = []
        self.total_end = 0.0
        self.acc = NoiseAccounting()
        self.ops: List[Tuple[str, Tuple[int, ...], float, float]] = []

        tqe = config.two_q_gate_error
        self.e2: Dict[Tuple[int, int], float] = {}
        if isinstance(tqe, dict):
            for k, v in tqe.items():
                self.e2[k] = float(v)
        elif isinstance(tqe, list):
            for r in tqe:
                self.e2[(int(r[0]), int(r[1]))] = float(r[2])
        else:
            for p in range(n_phys):
                for q in range(n_phys):
                    self.e2[(p, q)] = float(tqe)
        self.zz: Dict[Tuple[int, int], float] = {}
        cs = config.crosstalk_strength
        if isinstance(cs, dict):
            for k, v in cs.items():
                self.zz[k] = float(v)
        elif isinstance(cs, list):
            for r in cs:
                self.zz[(int(r[0]), int(r[1]))] = float(r[2])
        self.t1 = [float(v) for v in config.t1_times]
        self.t2 = [float(v) for v in config.t2_times]
        # single_q_gate_error: 标量或 per-qubit list（20q 真实拓扑为 list）
        sq = getattr(config, "single_q_gate_error", 0.001)
        if isinstance(sq, (list, tuple, np.ndarray)):
            self.p1q = [float(v) for v in sq]
        else:
            self.p1q = [float(sq)] * n_phys
        self.two_time = max(float(getattr(config, "two_gate_time", 0.3)), 1e-9)
        self.adj = set()
        for (a, b) in config.coupling_map:
            self.adj.add((a, b))
            self.adj.add((b, a))

    # -- 单机制代价（一阶通道数学） ----------------------------------------
    @staticmethod
    def _depol2_cost(p: float) -> float:
        return (15.0 / 16.0) * (3.0 / 4.0) * p

    @staticmethod
    def _depol1_cost(p: float) -> float:
        return (3.0 / 4.0) * (1.0 / 2.0) * p

    @staticmethod
    def _zz_cost(angle: float) -> float:
        return 4.0 * math.sin(angle) ** 2 / 5.0

    def _thermal_cost(self, q: int, t: float) -> float:
        if t <= 1e-9:
            return 0.0
        return t / (3.0 * max(self.t2[q], 1e-9)) + t / (6.0 * max(self.t1[q], 1e-9))

    def _e(self, p, q):
        return self.e2.get((p, q), self.e2.get((q, p), 0.01))

    def _theta(self, p, q):
        return self.zz.get((p, q), self.zz.get((q, p), 0.0))

    # -- 动态串扰：v3 同口径逐对角平方累积 ---------------------------------
    def _marginal_zz_sq(self, qs: Tuple[int, ...], start: float,
                        end: float) -> float:
        total_sq = 0.0
        qset = set(qs)
        for (b_qs, b_start, b_end) in self.in_flight:
            ov = min(end, b_end) - max(start, b_start)
            if ov <= 1e-12:
                continue
            if qset & set(b_qs):
                continue
            for a in qs:
                for b in b_qs:
                    if a != b and (a, b) in self.adj:
                        th = self._theta(a, b)
                        if th != 0.0:
                            total_sq += ((th / self.two_time) * ov) ** 2
        return total_sq

    # -- 主入口：记账一个物理操作 ------------------------------------------
    def add_op(self, op: str, qs: Tuple[int, ...]) -> None:
        dur = GATE_DUR.get(op, T_CX)
        start = max(self.last_free[q] for q in qs)
        end = start + dur
        for q in qs:
            self.acc.thermal += self._thermal_cost(q, start - self.last_free[q])
        if op == "swap":
            p, th = self._e(*qs), self._theta(*qs)
            self.acc.depol2 += 3 * self._depol2_cost(p)
            self.acc.zz_static += 3 * self._zz_cost(th)
            self.acc.thermal += self._thermal_cost(qs[0], dur)
            self.acc.thermal += self._thermal_cost(qs[1], dur)
        elif op in ("cx", "cz"):
            p, th = self._e(*qs), self._theta(*qs)
            self.acc.depol2 += self._depol2_cost(p)
            self.acc.zz_static += self._zz_cost(th * dur / self.two_time)
            self.acc.thermal += self._thermal_cost(qs[0], dur)
            self.acc.thermal += self._thermal_cost(qs[1], dur)
        else:
            if op not in ("id", "rz", "z", "barrier"):
                self.acc.depol1 += self._depol1_cost(self.p1q[qs[0]])
            self.acc.thermal += self._thermal_cost(qs[0], dur)
        sq = self._marginal_zz_sq(qs, start, end)
        self.acc.zz_dyn += (4.0 / 5.0) * sq
        for q in qs:
            self.last_free[q] = end
        self.in_flight = [(b, s, e) for (b, s, e) in self.in_flight if e > start]
        self.in_flight.append((tuple(qs), start, end))
        self.total_end = max(self.total_end, end)
        self.ops.append((op, tuple(qs), start, end))

    def finish(self) -> NoiseAccounting:
        """电路尾部：每比特剩余空闲热弛豫（与 v2 收尾一致）。"""
        for q in range(self.n):
            self.acc.thermal += self._thermal_cost(
                q, self.total_end - self.last_free[q])
        return self.acc


def noise_aware_rollout(env: PureRoutingEnv, config,
                        dist_noise: Optional[np.ndarray] = None,
                        max_steps: int = 3000,
                        ) -> Tuple[float, bool, NoiseTimeline]:
    """从 env 当前状态噪声感知贪心走到 terminal。

    策略：SABRE 打分（dist_override=噪声加权距离）；每步在 NoiseTimeline
    记账 swap + 级联执行的门。返回 (ΔF 总代价, ok, timeline)。
    timeline.ops = 物理电路 op 序列（cx 保留逻辑控制方向）。
    """
    from .sabre_heuristic import SabreScorer

    tl = NoiseTimeline(env.n_phys, config)
    scorer = SabreScorer()
    scorer.dist_override = dist_noise
    steps = 0
    seen = set()
    while not env.is_terminal():
        if steps >= max_steps:
            return tl.acc.total, False, tl
        key = env.state_key()
        if key in seen:
            return tl.acc.total, False, tl
        seen.add(key)
        sc = scorer.scores(env)
        legal = env.legal_actions()
        vals = np.nan_to_num(sc[legal], nan=1e9, posinf=1e9)
        e = int(legal[int(np.argmin(vals))])
        p, q = env.coupling_map[e]
        tl.add_op("swap", (p, q))
        mask_before = env.executed_mask
        env.step(e)
        steps += 1
        # 级联新执行的门（按 gate index 序 = env cascade 序）
        newly = env.executed_mask & ~mask_before
        for g in env._twoq:
            bit = 1 << g
            if newly & bit:
                gate = env.dag.gates[g]
                q0, q1 = gate.qubits
                tl.add_op("cx", (env.mapping[q0], env.mapping[q1]))
    tl.finish()
    return tl.acc.total, True, tl


def phys_circuit_from_ops(ops: List[Tuple[str, Tuple[int, ...]]], n_phys: int):
    """op 序列 → qiskit 物理电路（v3 评估入口）。ops 可含 (op, qs) 或 (op, qs, s, e)。"""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n_phys)
    for entry in ops:
        op, qs = entry[0], entry[1]
        if op == "swap":
            qc.swap(qs[0], qs[1])
        elif op == "cx":
            qc.cx(qs[0], qs[1])
        elif op == "cz":
            qc.cz(qs[0], qs[1])
        else:
            getattr(qc, op)(*qs) if hasattr(qc, op) else qc.id(qs[0])
    return qc
