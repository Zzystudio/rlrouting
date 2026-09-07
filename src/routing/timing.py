# ============================================================================
# timing.py — 门时间与线路时间感知模块（事件级 ASAP 调度内核）
#
# 提供 per-gate-type 精细门时长表、离散事件时序内核、增强贪心调度器
# （criticality + 时长 LPT + 串扰代价）、串扰软约束（A3）、SWAP 分解（A4）。
#
# 与「锁步轮」的区别：不再把每批门打包成一轮、统一从当前时钟开始。而是
# 用事件级 ASAP 调度——每门 start = max(前驱完成, 本比特上次空闲)，短门比特
# 在自己的门一结束即可开始下一门，并行度与时长真实反映到 makespan。
#
# 设计依据：doc/plan.md §「端到端联合优化框架 v2」Phase 1/2。
# ============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# per-gate-type 精细时长表（单位 µs），替代原来的固定 1Q=0.1 / 2Q=0.3。
# rz 是虚拟门（瞬时），sx≈35ns，cx/cz≈300ns。
GATE_DURATION_TABLE = {
    "cx": 0.30, "cz": 0.30, "ecr": 0.40, "swap": 0.90,
    "sx": 0.035, "x": 0.035, "y": 0.035, "h": 0.035,
    "s": 0.035, "t": 0.035, "sdg": 0.035, "tdg": 0.035,
    "rz": 0.0, "z": 0.0,
    "barrier": 0.0, "id": 0.0, "measure": 2.0,
}

FALLBACK_DURATION = 0.30
SWAP_DURATION_US = 0.90          # A4：SWAP = 3×CX
MAX_DUR = 0.90                   # 归一化时长用的最大值（SWAP 最长）
# 优先级权重（A2）
W_DUR = 0.30                     # 时长项（LPT）权重
W_XTALK = 1.0                    # 串扰代价项权重


@dataclass
class CircuitTiming:
    """离散事件时序内核状态。

    - total_time: 累计线路执行时间（µs），等价于全局 makespan
    - qubit_busy_until: 每个物理 qubit 的下次可用时刻（互斥判据）
    - qubit_idle_time: 每个物理 qubit 累计空闲时间（µs），用于空闲退相干惩罚
      仅从「该比特首次执行门」起算（未激活比特不计）
    - qubit_crosstalk: 每个物理 qubit 累计受到的并行串扰强度（zz 加权）
    - parallel_usage: 每对耦合边被同时并行使用的累计计数（预留给图编码特征）
    - waves: 调度迭代次数（一层就绪门一批）
    - gate_end: gate_idx → 该门结束时刻（查前驱用）
    - last_free: 每比特上次门结束时刻（-1 哨兵=未使用）
    - schedule_log: 每门时序记录（start/end/wave）
    """

    total_time: float = 0.0
    qubit_busy_until: Optional[np.ndarray] = None      # [n_phys]
    qubit_idle_time: Optional[np.ndarray] = None       # [n_phys]
    qubit_crosstalk: Optional[np.ndarray] = None        # [n_phys]
    parallel_usage: Optional[np.ndarray] = None        # [n_phys, n_phys]
    waves: int = 0
    crosstalk_events: float = 0.0          # 累计串扰（A2：zz×重叠时长，单位 zz·µs）
    serial_dur: float = 0.0
    total_gates_executed: int = 0
    schedule_log: list = field(default_factory=list)
    gate_end: Dict[int, float] = field(default_factory=dict)
    last_free: Optional[np.ndarray] = None              # [n_phys], -1 = 未使用

    @classmethod
    def create(cls, n_phys: int) -> "CircuitTiming":
        return cls(
            qubit_busy_until=np.zeros(n_phys, dtype=float),
            qubit_idle_time=np.zeros(n_phys, dtype=float),
            qubit_crosstalk=np.zeros(n_phys, dtype=float),
            parallel_usage=np.zeros((n_phys, n_phys), dtype=float),
            last_free=np.full(n_phys, -1.0, dtype=float),
            schedule_log=[],
            gate_end={},
        )

    def clone(self) -> "CircuitTiming":
        new = CircuitTiming.create(len(self.qubit_busy_until))
        new.total_time = self.total_time
        new.qubit_busy_until[:] = self.qubit_busy_until
        new.qubit_idle_time[:] = self.qubit_idle_time
        new.qubit_crosstalk[:] = self.qubit_crosstalk
        new.parallel_usage[:] = self.parallel_usage
        new.last_free[:] = self.last_free
        new.waves = self.waves
        new.crosstalk_events = self.crosstalk_events
        new.serial_dur = self.serial_dur
        new.total_gates_executed = self.total_gates_executed
        new.schedule_log = list(self.schedule_log)
        new.gate_end = dict(self.gate_end)
        return new

    def _phys_qubits_idle(self, kind, pq, start, dur):
        """记录 pq 中每比特的空闲 gap 与更新 last_free/busy_until。"""
        end = start + dur
        for q in pq:
            lf = self.last_free[q]
            if lf >= 0.0:
                self.qubit_idle_time[q] += max(0.0, start - lf)
            self.last_free[q] = end
            self.qubit_busy_until[q] = end

    def finalize_idle(self):
        """线路结束时，给已激活比特补上末尾尾段空闲。幂等可重复调用（按 last_free）。"""
        for q in range(len(self.last_free)):
            lf = self.last_free[q]
            if lf >= 0.0:
                self.qubit_idle_time[q] += max(0.0, self.total_time - lf)
                self.last_free[q] = self.total_time


class GreedyScheduler:
    """环境内部固定贪心调度器（框架 v2 的 Phase 1/2 实现）。

    优先级 = criticality + W_DUR·(dur/max_dur) − W_XTALK·xtalk_cost：
    - criticality 高（靠近线路终点）优先
    - 时长长（LPT）优先，使 makespan 近似最优
    - 与同批其他门在相邻耦合对上的串扰代价越大越靠后（让出并行位）
    """

    def _phys_qubits(self, g, mapping) -> List[int]:
        return [mapping[q] for q in g.qubits]

    def _xtalk_cost(self, g, dag, mapping, hw, pending: List[int]) -> float:
        """g 与同批其他候选门在相邻耦合对上的 Σzz。"""
        if hw is None or not pending:
            return 0.0
        qset = set(self._phys_qubits(g, mapping))
        cost = 0.0
        for oidx in pending:
            if oidx == g.index:
                continue
            oq = self._phys_qubits(dag.gates[oidx], mapping)
            for a in qset:
                for b in oq:
                    if a != b and hw.adj[a, b] > 0:
                        zz = float(hw.zz[a, b])
                        if zz > 0:
                            cost += zz
        return cost

    def priority(self, g, dag, timing: "CircuitTiming", ctx=None, hw=None) -> float:
        rem = dag.remaining_depths()
        md = max(1, dag.max_depth())
        crit = float(rem.get(g.index, 0)) / md
        dur = GATE_DURATION_TABLE.get(g.name, FALLBACK_DURATION)
        dur_term = dur / MAX_DUR
        xtalk_cost = 0.0
        if ctx is not None:
            pending = ctx.get("pending", [])
            xtalk_cost = self._xtalk_cost(g, dag, ctx.get("mapping"), hw, pending)
        return crit + W_DUR * dur_term - W_XTALK * xtalk_cost


def _overlap(a_s: float, a_e: float, b_s: float, b_e: float) -> bool:
    return a_s < b_e and b_s < a_e


# 串扰建模：仅 1-hop（两门交叉比特对在某耦合边相邻）。与模拟器动态串扰口径一致；
# 次近邻（图距离 2）不计入（调度互斥下物理不可同时执行，且与模拟器建模对齐）。
XTALK_HOPS = 1
XTALK_DECAY = 0.3        # 预留：若未来需扩展次近邻，距离-2 串扰相对距离-1 的衰减系数


def _crosstalk_graph(hw, hops: int = XTALK_HOPS, decay: float = XTALK_DECAY):
    """返回 (cx_adj, cx_zz)：覆盖图距离 1..hops 的串扰邻接与强度（默认到次近邻）。

    - 距离 1：直接用 hw.zz
    - 距离 k>=2：沿最短路径两端直接 zz 均值 * decay^(k-1)
    结果缓存在 hw 上（按 hops/decay 作 key），避免每个调度步重复 BFS。
    """
    cache = getattr(hw, "_cx_cache", None)
    if cache is not None and cache[0] == hops and cache[1] == decay:
        return cache[2], cache[3]
    n = hw.num_qubits
    adj = hw.adj
    INF = 10 ** 9
    D = [[INF] * n for _ in range(n)]
    parent = [[-1] * n for _ in range(n)]
    for s in range(n):
        D[s][s] = 0
        q = [s]
        while q:
            u = q.pop(0)
            for v in range(n):
                if adj[u, v] > 0 and D[s][v] > D[s][u] + 1:
                    D[s][v] = D[s][u] + 1
                    parent[s][v] = u
                    q.append(v)
    zz_vals = [hw.zz[i, j] for i in range(n) for j in range(n) if adj[i, j] > 0]
    base_zz = float(np.mean(zz_vals)) if zz_vals else 0.0
    cx_adj = np.zeros((n, n), dtype=float)
    cx_zz = np.zeros((n, n), dtype=float)
    for a in range(n):
        for b in range(n):
            d = D[a][b]
            if 0 < d <= hops:
                cx_adj[a, b] = 1.0
                if d == 1:
                    cx_zz[a, b] = float(hw.zz[a, b])
                else:
                    mid = parent[a][b]
                    if mid >= 0 and adj[a, mid] > 0 and adj[mid, b] > 0:
                        path_zz = (float(hw.zz[a, mid]) + float(hw.zz[mid, b])) / 2.0
                    else:
                        path_zz = base_zz
                    cx_zz[a, b] = decay * path_zz
    hw._cx_cache = (hops, decay, cx_adj, cx_zz)
    return cx_adj, cx_zz


def schedule_events(
    ready_1q: List[int],
    ready_2q: List[int],
    dag,
    mapping: List[int],
    timing: "CircuitTiming",
    hw,
    scheduler: Optional[GreedyScheduler] = None,
    xtalk_alpha: float = 0.0,
    pending_swaps: Optional[List[Tuple[int, int, float]]] = None,
) -> Tuple[List[Tuple[int, float, float, str]], float, float]:
    """事件级 ASAP 调度一批就绪门，返回 (placed, clock_advance, xtalk)。

    placed: list of (gate_idx, start, end, kind)；kind ∈ {1q, 2q, measure, swap}。

    - 每门 start = max(total_time, 前驱结束, 本比特 last_free)，end = start + dur
    - 串扰（A2）仅按 1-hop 相邻交叉对记录 zz×重叠时长，供连续惩罚；不做硬延迟
      （硬延迟阈值非物理，且模拟器侧亦只建模 1-hop 同时执行的动态串扰）
    - 互斥约束（必须）：共享比特的门不会在同一位置重叠（last_free 保证）
    """
    n_phys = len(timing.qubit_busy_until)
    wave_start = timing.total_time
    candidates = []
    for gidx in ready_1q:
        g = dag.gates[gidx]
        pq = [mapping[g.qubits[0]]]
        dur = GATE_DURATION_TABLE.get(g.name, FALLBACK_DURATION)
        kind = "measure" if g.is_measure else "1q"
        candidates.append((gidx, pq, dur, kind))
    for gidx in ready_2q:
        g = dag.gates[gidx]
        pq = [mapping[g.qubits[0]], mapping[g.qubits[1]]]
        dur = GATE_DURATION_TABLE.get(g.name, FALLBACK_DURATION)
        candidates.append((gidx, pq, dur, "2q"))
    # SWAP 调度事件（来自 env 的 _pending_swaps）：作为伪门参与 ASAP 调度，
    # 可与不相关比特上的门并行（gate_idx=-1 表示非 DAG 门）。先放最前，
    # 使 SWAP 尽早占位 p,q，其它比特的门与 SWAP 重叠执行（修复串行化失真）。
    swap_cands = []
    for (sp, sq, sd) in (pending_swaps or []):
        swap_cands.append((-1, [sp, sq], sd, "swap"))

    if not candidates and not swap_cands:
        return [], 0.0, 0.0

    cx_adj, cx_zz = _crosstalk_graph(hw) if hw is not None else (None, None)

    ctx = {"mapping": mapping, "pending": ready_1q + ready_2q}
    if scheduler is not None:
        candidates.sort(key=lambda e: -scheduler.priority(
            dag.gates[e[0]], dag, timing, ctx, hw))
    else:
        candidates.sort(key=lambda e: -e[2])
    candidates = swap_cands + candidates

    placed: List[Tuple[int, float, float, str]] = []
    total_xtalk = 0.0
    for gidx, pq, dur, kind in candidates:
        if gidx >= 0:
            g = dag.gates[gidx]
            pred_end = 0.0
            for p in g.predecessors:
                pred_end = max(pred_end, timing.gate_end.get(p, 0.0))
        else:
            g = None
            pred_end = 0.0
        # 真正事件级 ASAP：start = max(前驱完成, 本比特上次空闲)，不再以 wave_start
        # 作地板。这样跨步（PPO 每步一个 SWAP）的不相关 SWAP/门也能彼此重叠，
        # 与 SABRE（同一调度器一次性排程所有就绪门）口径一致。
        start = pred_end
        for q in pq:
            lf = timing.last_free[q]
            if lf >= 0.0:
                start = max(start, lf)
        end = start + dur
        # 应用：空闲 gap + 更新 last_free/busy_until
        timing._phys_qubits_idle(None, pq, start, dur)
        # 串扰记录（A2：按时长加权 zz×重叠时长，量纲与 r_time 一致；含 1-hop 次近邻）
        # + 同时计入 timing.crosstalk_events 供报告 + 并行使用统计
        active = set(pq)
        for (hg, hs, he, hk) in placed:
            if hk == "swap":
                continue
            if _overlap(start, end, hs, he):
                ov = min(end, he) - max(start, hs)
                if ov <= 0:
                    continue
                hpq = [mapping[q] for q in dag.gates[hg].qubits]
                for a in pq:
                    for b in hpq:
                        zz = 0.0
                        if cx_adj is not None and cx_adj[a, b] > 0:
                            zz = float(cx_zz[a, b])
                        elif hw is not None and hw.adj[a, b] > 0:
                            zz = float(hw.zz[a, b])
                        if a != b and zz > 0:
                            w = zz * ov
                            total_xtalk += w
                            timing.crosstalk_events += w
                            timing.qubit_crosstalk[a] += w
                            timing.qubit_crosstalk[b] += w
        for a in active:
            for b in active:
                if a < b and hw.adj[a, b] > 0:
                    timing.parallel_usage[a, b] += 1.0
                    timing.parallel_usage[b, a] += 1.0
        # 日志
        op = "swap" if gidx < 0 else g.name
        timing.schedule_log.append({
            "kind": kind,
            "gate_idx": gidx,
            "op": op,
            "qubits": list(pq),
            "start": start,
            "end": end,
            "wave": timing.waves,
        })
        if gidx >= 0:
            timing.gate_end[gidx] = end
        timing.serial_dur += (end - start)
        placed.append((gidx, start, end, kind))

    timing.waves += 1
    # 真正事件级：total_time = 全局最晚 busy 时刻（而非 wave_start+本批最长门），
    # 使不相关 SWAP/门跨步重叠后 makespan 正确收敛到关键路径。
    new_total = float(timing.qubit_busy_until.max()) if timing.qubit_busy_until is not None else wave_start
    clock_advance = max(0.0, new_total - wave_start)
    timing.total_time = new_total
    return placed, clock_advance, total_xtalk


def schedule_routed_circuit(
    dag,
    hw,
    mapping: Optional[List[int]] = None,
    scheduler: Optional[GreedyScheduler] = None,
    swap_duration: float = SWAP_DURATION_US,
    xtalk_alpha: float = 0.0,
) -> Tuple[float, float, Dict[str, float]]:
    """D2：对已路由（物理）线路做事件级调度，返回 (makespan, xtalk, stats)。

    用于把 SABRE 等基线的输出电路用「同一调度器」排程，得到公平可比的
    makespan / 串扰 / 并行度。mapping 缺省为 identity（电路已是物理线路）。
    """
    n_phys = hw.num_qubits
    if mapping is None:
        mapping = list(range(n_phys))
    timing = CircuitTiming.create(n_phys)
    executed: set = set()
    total_serial = 0.0
    while len(executed) < dag.num_gates:
        ready = []
        for g in dag.gates:
            if g.index in executed:
                continue
            if all(p in executed for p in g.predecessors):
                ready.append(g.index)
        if not ready:
            break
        ready_2q = [i for i in ready if dag.gates[i].is_two_qubit]
        ready_1q = [i for i in ready if i not in ready_2q]
        placed, adv, xt = schedule_events(
            ready_1q, ready_2q, dag, mapping, timing, hw, scheduler, xtalk_alpha)
        if not placed:
            break
        for (gidx, s, e, _) in placed:
            executed.add(gidx)
            total_serial += (e - s)
        timing.finalize_idle()
    makespan = timing.total_time
    density = (total_serial / makespan) if makespan > 1e-9 else 0.0
    pc = {}
    for e in timing.schedule_log:
        if e.get("kind") == "swap":
            continue
        w = e.get("wave", -1)
        pc[w] = pc.get(w, 0) + 1
    peak = max(pc.values()) if pc else 0
    return makespan, timing.crosstalk_events, {
        "makespan_us": makespan,
        "crosstalk_events": timing.crosstalk_events,
        "serial_dur_us": total_serial,
        "density": density,
        "peak_parallel": peak,
        "critical_path_lb_us": makespan,
    }
