# ============================================================================
# trajectory_sim_v2.py
# 事件级调度感知噪声模拟器（v2）
#
# 与 trajectory_sim.py（v1）的关系：
#   v1 的调度感知路径（evolve_scheduled）以「同步波」为输入：同波所有门假设
#   同时开始、统一门时长（single_time/two_time），动态串扰每波一次固定角度。
#   v2 改为直接消费「事件级调度」：每门带精确 (start, end)，时长 = end-start
#   （rz=0 虚拟门无退相干、sx=0.035、cx=0.3、swap=0.9 自动生效），动态串扰按
#   实际重叠时长缩放，并可选建模 always-on 空闲 ZZ 串扰。
#
#   本文件不修改 v1 的任何代码：EventNoiseConfig 以子类扩展 NoiseConfig，
#   EventTrajectorySimulator 以子类复用 v1 全部批量噪声内核（MC Kraus 采样）。
#
# 事件格式：
#   Event = (start, end, op, qubits[, phys_idx])
#     start/end : float，门执行窗口（µs），时长 = end - start
#     op        : 门名（"cx"/"cz"/"swap"/单比特门名）
#     qubits    : 物理比特元组，cx 的 qubits[0] 为 control
#     phys_idx  : 可选，circuit.data 中该指令的位置（rz 等参数化门取参数必需）
#
# 噪声语义（关键标定，详见 doc/sim.md v2 章节）：
#   - 时间：每比特独立时钟 t_last_end[q]；门执行前对该比特空闲区间施加热弛豫
#     （Markov 可复合，精确）；门执行后按 dur 施加门内热弛豫；末尾收尾空闲。
#   - 门噪声：1q = depol1 + thermal(dur)（id 只加 thermal；rz(dur=0) 无噪声）；
#     cx/cz = depol2(edge) + 静态 ZZ θ·(dur/two_gate_time) + thermal 双比特；
#     swap = 酉矩阵 + 3×(depol2 + 静态 ZZ θ) + thermal(0.9)（物理等价 3×CX）。
#   - 动态串扰：时间重叠的不相交事件对，交叉比特对在耦合图上相邻（1-hop）时
#     施加 ZZ(χ_rate·ov)，χ_rate = θ_static/two_gate_time (rad/µs)，ov 为实际
#     重叠时长——同步纯 cx 波（ov=two_gate_time）下与 v1 数值一致。
#     含 1Q-2Q spectator 并发对；swap 默认不作串扰源（与 timing.py A2 口径
#     一致），可用 EventNoiseConfig.swap_xtalk 开启。
#   - always-on 空闲 ZZ（EventNoiseConfig.always_on_zz，默认 None=关）：
#     每对比特在「双空闲」时间段施加 ZZ(rate·Δt)，段边界处应用（ZZ 与门不对
#     易，不跨段合并）。
# ============================================================================

from __future__ import annotations

import heapq
from dataclasses import dataclass, fields as dc_fields
from typing import Dict, List, Optional, Tuple

import numpy as np

from qiskit import QuantumCircuit

from sim.sim import NoiseConfig
from sim.trajectory_sim import TrajectorySimulator, TrajectoryResult, _U1

# 事件元组最短长度 4；第 5 位为可选 phys_idx
Event = Tuple[float, float, str, Tuple[int, ...]]

_EPS = 1e-12

SUPPORTED_2Q_OPS = ("cx", "cz", "swap")

# 镜像 routing/timing.GATE_DURATION_TABLE（µs）。sim 包不反向依赖 routing 包，
# 调用方可显式传入 durations 覆盖（如 GATE_DURATION_TABLE），保持两表同步。
_DEFAULT_GATE_DURATIONS: Dict[str, float] = {
    "cx": 0.30, "cz": 0.30, "ecr": 0.40, "swap": 0.90,
    "sx": 0.035, "x": 0.035, "y": 0.035, "h": 0.035,
    "s": 0.035, "t": 0.035, "sdg": 0.035, "tdg": 0.035,
    "rz": 0.0, "z": 0.0,
    "barrier": 0.0, "id": 0.0, "measure": 2.0,
}
_FALLBACK_DURATION = 0.30


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class EventNoiseConfig(NoiseConfig):
    """v2 事件级模拟器配置：在 NoiseConfig 基础上追加 always-on 串扰开关。"""

    # always-on 空闲 ZZ 串扰：(q1, q2) -> 旋转角速率 (rad/µs)，双空闲时间段
    # 按 ZZ(rate·Δt) 施加；None = 关闭（默认，不改变 v1 噪声标定）
    always_on_zz: Optional[Dict[Tuple[int, int], float]] = None
    # swap 是否作为动态串扰源（默认 False，与 timing.py A2 口径一致）
    swap_xtalk: bool = False
    # swap 是否施加门噪声（depol/ZZ/热弛豫）。False = v1 语义（仅推进时钟），
    # 用于 v1→v2 口径消融；默认 True（物理等价 3×CX）
    swap_noise: bool = True

    @classmethod
    def from_noise_config(cls, cfg: NoiseConfig,
                          **overrides) -> "EventNoiseConfig":
        """由普通 NoiseConfig 构造（复制全部同名字段，可覆盖新增字段）。"""
        kw = {f.name: getattr(cfg, f.name) for f in dc_fields(NoiseConfig)}
        kw.update(overrides)
        return cls(**kw)


# ---------------------------------------------------------------------------
# 事件构建 / 校验
# ---------------------------------------------------------------------------
def timing_log_to_events(schedule_log: List[dict],
                         circuit: Optional[QuantumCircuit] = None,
                         remap: Optional[Dict[int, int]] = None) -> List[Event]:
    """将 CircuitTiming.schedule_log（事件级调度日志）转换为事件列表。

    schedule_log 条目：{kind, gate_idx, op, qubits, start, end, wave}；
    跳过 measure 条目；swap 条目（gate_idx=-1, kind="swap"）保留为 swap 事件。

    circuit 给定时，phys_idx 取「第 i 条非 measure/barrier 指令」在
    circuit.data 中的真实位置（保证 rz 等参数化门能取到参数）；
    remap 给定时（比特子集缩减，old->new），qubits 做重映射。
    """
    # 预计算非 measure/barrier 指令的 data 位置（与日志顺序一一对应）
    data_pos: Optional[List[int]] = None
    if circuit is not None:
        data_pos = [
            i for i, inst in enumerate(circuit.data)
            if inst.operation.name.lower() not in ("measure", "barrier")
        ]
    events: List[Event] = []
    counter = 0
    for entry in schedule_log:
        if entry.get("kind") == "measure":
            continue
        qs = tuple(entry["qubits"])
        if remap is not None:
            qs = tuple(remap[q] for q in qs)
        if data_pos is not None:
            phys_idx = data_pos[counter] if counter < len(data_pos) else None
        else:
            phys_idx = counter
        events.append((float(entry["start"]), float(entry["end"]),
                       entry["op"], qs, phys_idx))
        counter += 1
    return events


def schedule_phys_circuit_events(circuit: QuantumCircuit,
                                 durations: Optional[Dict[str, float]] = None,
                                 ) -> List[Event]:
    """对已映射（物理）平铺电路做事件级 ASAP 调度，返回事件列表。

    电路 data 为拓扑序时，逐门 start = max(各比特上次门结束) 即精确 ASAP
    （DAG 前驱必然共享比特，其结束时刻已包含在 last_free 中）。
    durations 缺省使用模块内镜像表（与 routing/timing.GATE_DURATION_TABLE
    同步维护）；调用方可显式传入（如 GATE_DURATION_TABLE）。
    """
    tbl = durations if durations is not None else _DEFAULT_GATE_DURATIONS
    events: List[Event] = []
    last_free: Dict[int, float] = {}
    for idx, inst in enumerate(circuit.data):
        op = inst.operation
        name = op.name.lower()
        if name in ("measure", "barrier"):
            continue
        qs = tuple(circuit.find_bit(q).index for q in inst.qubits)
        dur = float(tbl.get(name, _FALLBACK_DURATION))
        start = 0.0
        for q in qs:
            lf = last_free.get(q, 0.0)
            if lf > start:
                start = lf
        end = start + dur
        events.append((start, end, name, qs, idx))
        for q in qs:
            last_free[q] = end
    return events


def validate_events(events, circuit: QuantumCircuit) -> None:
    """校验事件列表与电路一致且调度合法（不合法抛 ValueError/NotImplementedError）。

    - 事件数与非 measure/barrier 指令数一致，且 (op, qubits) 按顺序一一对应；
    - 同一比特上的事件互斥（调度互斥约束）；
    - 双比特门仅支持 cx/cz/swap（ecr 等请先转译）。
    """
    ref = []
    for inst in circuit.data:
        name = inst.operation.name.lower()
        if name in ("measure", "barrier"):
            continue
        qs = tuple(circuit.find_bit(q).index for q in inst.qubits)
        ref.append((name, qs))
    if len(events) != len(ref):
        raise ValueError(
            f"事件数 {len(events)} 与电路非测量指令数 {len(ref)} 不一致")
    for i, ((name, qs), ev) in enumerate(zip(ref, events)):
        if len(ev) < 4:
            raise ValueError(f"事件 {i} 至少需要 (start, end, op, qubits) 4 元")
        if ev[2] != name or tuple(ev[3]) != qs:
            raise ValueError(
                f"事件 {i} ({ev[2]}, {tuple(ev[3])}) 与电路指令 "
                f"({name}, {qs}) 不匹配（顺序或比特不一致）")
    # 比特互斥：同比特事件的时间区间不允许重叠（与列表顺序无关，
    # 编译阶段会按 start 排序）
    per_q: Dict[int, List[Tuple[float, float]]] = {}
    for i, ev in enumerate(events):
        s, e, op, qs = float(ev[0]), float(ev[1]), ev[2], tuple(ev[3])
        if e < s - 1e-9:
            raise ValueError(f"事件 {i} end({e}) < start({s})")
        if len(qs) == 2 and op not in SUPPORTED_2Q_OPS:
            raise NotImplementedError(
                f"v2 暂不支持双比特门 '{op}'（请转译为 cx/cz/swap）")
        for q in qs:
            per_q.setdefault(q, []).append((s, e, i))
    for q, ivs in per_q.items():
        ivs.sort()
        for (s0, e0, i0), (s1, e1, i1) in zip(ivs, ivs[1:]):
            if s1 < e0 - 1e-9:
                raise ValueError(
                    f"事件 {i0} 与事件 {i1} 在比特 {q} 上时间重叠"
                    f"（调度互斥被违反）")


# ---------------------------------------------------------------------------
# 空闲区间 / 区间求交（always-on ZZ 用）
# ---------------------------------------------------------------------------
def _free_intervals(busy: List[Tuple[float, float]],
                    total: float) -> List[Tuple[float, float]]:
    """由有序不重叠的忙碌区间求 [0, total] 内的空闲区间。"""
    free: List[Tuple[float, float]] = []
    cur = 0.0
    for (s, e) in busy:
        if s > cur:
            free.append((cur, s))
        cur = max(cur, e)
    if cur < total:
        free.append((cur, total))
    return free


def _intersect_free(fa: List[Tuple[float, float]],
                    fb: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """两个有序空闲区间列表的交集（双空闲时间段）。"""
    segs: List[Tuple[float, float]] = []
    i = j = 0
    while i < len(fa) and j < len(fb):
        s = max(fa[i][0], fb[j][0])
        e = min(fa[i][1], fb[j][1])
        if e > s:
            segs.append((s, e))
        if fa[i][1] < fb[j][1]:
            i += 1
        else:
            j += 1
    return segs


# ---------------------------------------------------------------------------
# v2 模拟器
# ---------------------------------------------------------------------------
class EventTrajectorySimulator(TrajectorySimulator):
    """事件级调度感知轨迹模拟器（复用 v1 全部批量 MC 噪声内核）。"""

    # ---------------- 基础内核补充 ----------------
    def _apply_cz_batch(self, svs: np.ndarray, a: int, b: int) -> np.ndarray:
        """批量 CZ（对角门）：两比特均为 |1> 的基态幅乘 -1。"""
        if self._all is None:
            self._all = np.arange(2 ** self.n_qubits)
        mask = (((self._all >> a) & 1) & ((self._all >> b) & 1)).astype(bool)
        idx = np.nonzero(mask)[0]
        if idx.size:
            svs[:, idx] *= -1.0
        return svs

    def _zz_rotation_batch(self, svs: np.ndarray, q1: int, q2: int,
                           theta: float) -> np.ndarray:
        """批量 ZZ 相干旋转 U = exp(-i·θ·Z⊗Z)（与 v1 _crosstalk_batch 同构，
        角度由调用方给定：静态串扰按门时长缩放，动态串扰按重叠时长缩放）。"""
        if theta == 0.0:
            return svs
        svs *= np.exp(-1j * theta)
        svs[:, self._zz_xor_indices(q1, q2)] *= np.exp(2j * theta)
        return svs

    # ---------------- 时间线编译 ----------------
    def _gate_matrix_for(self, op: str, circuit: QuantumCircuit,
                         phys_idx: Optional[int]) -> np.ndarray:
        """1q 门的酉矩阵：优先查表，参数化门（rz 等）经 phys_idx 取 op。"""
        if op == "id":
            return np.eye(2, dtype=complex)
        if phys_idx is not None:
            inst = circuit.data[phys_idx]
            return self._single_qubit_matrix(inst.operation, op)
        if op in _U1:
            return _U1[op]
        raise ValueError(
            f"门 '{op}' 需要参数（如 rz 旋转角），但事件未提供 phys_idx")

    def _prepare_events(self, circuit: QuantumCircuit,
                        events) -> Tuple[List[tuple], float]:
        """校验并编译事件为统一动作流：
        - ("gate", start, end, op, qubits, mat_or_None)
        - ("zz", close_time, q1, q2, angle)   # 动态/always-on 串扰， ZZ 对角门
        动作按 (时间, 优先级, 序号) 排序：串扰在同时刻门之前应用。
        返回 (actions, total_time)。
        """
        validate_events(events, circuit)
        evs = sorted(events, key=lambda ev: float(ev[0]))
        n = self.n_qubits
        total = max((float(ev[1]) for ev in evs), default=0.0)

        # 每比特忙碌区间（只计正时长的真实门；显式 id/delay 物理上仍是空闲，
        # 计入 always-on 串扰的双空闲窗口）
        busy: List[List[Tuple[float, float]]] = [[] for _ in range(n)]
        for ev in evs:
            s, e = float(ev[0]), float(ev[1])
            if e > s and ev[2] != "id":
                for q in ev[3]:
                    busy[q].append((s, e))

        adj = set()
        for (a, b) in self.config.coupling_map:
            adj.add((a, b))
            adj.add((b, a))
        swap_xt = bool(getattr(self.config, "swap_xtalk", False))
        two_time = max(float(self.config.two_gate_time), 1e-9)
        rate_cache: Dict[Tuple[int, int], float] = {}

        def _rate(a: int, b: int) -> float:
            r = rate_cache.get((a, b))
            if r is None:
                r = self._crosstalk_theta(a, b) / two_time
                rate_cache[(a, b)] = r
            return r

        # --- 动态串扰：重叠不相交事件对的 1-hop 相邻交叉对 ---
        zz_actions: List[Tuple[float, int, int, float]] = []
        heap: List[Tuple[float, int]] = []  # (end, idx)
        for i, ev in enumerate(evs):
            s, e, op, qs = float(ev[0]), float(ev[1]), ev[2], tuple(ev[3])
            while heap and heap[0][0] <= s:
                heapq.heappop(heap)
            for (e_j, j) in heap:
                ev_j = evs[j]
                if not swap_xt and (op == "swap" or ev_j[2] == "swap"):
                    continue
                qs_j = tuple(ev_j[3])
                if set(qs) & set(qs_j):
                    continue  # 防御：共享比特不产生串扰对
                ov = min(e, e_j) - s
                if ov <= 0:
                    continue
                for a in qs:
                    for b in qs_j:
                        if (a, b) in adj:
                            r = _rate(a, b)
                            if r != 0.0:
                                zz_actions.append(
                                    (min(e, e_j), a, b, r * ov))
            heapq.heappush(heap, (e, i))

        # --- always-on 空闲 ZZ：耦合对「双空闲」时间段按 rate·Δt 施加 ---
        ao = getattr(self.config, "always_on_zz", None)
        if ao:
            frees = {q: _free_intervals(busy[q], total) for q in range(n)}
            for pair, rate in ao.items():
                q1, q2 = int(pair[0]), int(pair[1])
                if rate is None or rate <= 0:
                    continue
                if q1 >= n or q2 >= n:
                    raise ValueError(
                        f"always_on_zz 涉及非法量子比特索引: ({q1}, {q2})")
                for (s0, s1) in _intersect_free(frees[q1], frees[q2]):
                    if s1 - s0 > _EPS:
                        zz_actions.append((s1, q1, q2, float(rate) * (s1 - s0)))

        # --- 合并动作流（zz 优先于同时刻门；ZZ 相互对易，序号仅保稳定） ---
        acts: List[tuple] = []
        seq = 0
        for ev in evs:
            s, e, op, qs = float(ev[0]), float(ev[1]), ev[2], tuple(ev[3])
            phys_idx = ev[4] if len(ev) > 4 else None
            if len(qs) == 2:
                mat = None
            else:
                mat = self._gate_matrix_for(op, circuit, phys_idx)
            acts.append((s, 1, seq, ("gate", s, e, op, qs, mat)))
            seq += 1
        for (t, q1, q2, ang) in zz_actions:
            acts.append((t, 0, seq, ("zz", t, q1, q2, ang)))
            seq += 1
        acts.sort(key=lambda x: (x[0], x[1], x[2]))
        return [a[3] for a in acts], total

    # ---------------- 演化 ----------------
    def _run_actions(self, svs: np.ndarray, actions: List[tuple],
                     total: float, apply_noise: bool) -> np.ndarray:
        """按动作流演化 (T, 2^n) 批量状态。返回演化后的 svs。"""
        n = self.n_qubits
        two_time = max(float(self.config.two_gate_time), 1e-9)
        t_last_end = [0.0] * n
        for act in actions:
            if act[0] == "zz":
                if apply_noise:
                    _, _, q1, q2, ang = act
                    svs = self._zz_rotation_batch(svs, q1, q2, ang)
                continue
            _, start, end, op, qs, mat = act
            dur = end - start
            # 门执行前：对该比特的空闲区间施加热弛豫（Markov 可复合）。
            # 与事件自身时长无关——零时长事件（如 rz）前的等待期同样要计，
            # 否则会产生与 v1 swap 漏算同族的空闲丢失
            if apply_noise:
                for q in qs:
                    idle = start - t_last_end[q]
                    if idle > _EPS:
                        svs = self._thermal_noise_batch(svs, q, idle)
            if op == "swap":
                svs = self._apply_swap_batch(svs, qs[0], qs[1])
                if (apply_noise and dur > _EPS
                        and getattr(self.config, "swap_noise", True)):
                    # 物理等价 3×CX：3×(depol2 + 静态 ZZ θ) + 门内热弛豫
                    p = self._two_error(qs[0], qs[1])
                    th = self._crosstalk_theta(qs[0], qs[1])
                    for _ in range(3):
                        svs = self._depol2_batch(svs, qs[0], qs[1], p)
                        svs = self._zz_rotation_batch(svs, qs[0], qs[1], th)
                    svs = self._thermal_noise_batch(svs, qs[0], dur)
                    svs = self._thermal_noise_batch(svs, qs[1], dur)
            elif op in ("cx", "cz"):
                if op == "cx":
                    svs = self._apply_cx_batch(svs, qs[0], qs[1])
                else:
                    svs = self._apply_cz_batch(svs, qs[0], qs[1])
                if apply_noise and dur > _EPS:
                    p = self._two_error(qs[0], qs[1])
                    svs = self._depol2_batch(svs, qs[0], qs[1], p)
                    svs = self._zz_rotation_batch(
                        svs, qs[0], qs[1],
                        self._crosstalk_theta(qs[0], qs[1]) * dur / two_time)
                    svs = self._thermal_noise_batch(svs, qs[0], dur)
                    svs = self._thermal_noise_batch(svs, qs[1], dur)
            else:
                svs = self._apply1_batch(svs, qs[0], mat)
                if apply_noise and dur > _EPS:
                    if op != "id":
                        svs = self._depol1_batch(svs, qs[0],
                                                 self._one_error(qs[0]))
                    svs = self._thermal_noise_batch(svs, qs[0], dur)
            for q in qs:
                t_last_end[q] = end
        # 收尾：电路结束后仍空闲的比特施加剩余退相干
        if apply_noise:
            for q in range(n):
                idle = total - t_last_end[q]
                if idle > _EPS:
                    svs = self._thermal_noise_batch(svs, q, idle)
        return svs

    def _run_events_trajectories(self, circuit: QuantumCircuit, events,
                                 num_trajectories: int,
                                 apply_noise: bool = True) -> np.ndarray:
        """编译事件并演化 num_trajectories 条轨迹，返回 (T, 2^n) 数组。

        工作集超限时按块分批（保持批量内核的缓存局部性），语义与一次性
        批量演化完全一致（同一 rng 流顺序推进）。
        """
        actions, total = self._prepare_events(circuit, events)
        ws_traj = (1 << self.n_qubits) * 16  # complex128 字节 / 轨迹
        chunk = max(1, self._BATCH_WS_LIMIT // ws_traj)
        outs: List[np.ndarray] = []
        done = 0
        while done < num_trajectories:
            t = min(chunk, num_trajectories - done)
            svs = np.tile(self._initial_state(), (t, 1))
            if circuit.global_phase:
                svs = svs * np.exp(1j * float(circuit.global_phase))
            svs = self._run_actions(svs, actions, total, apply_noise)
            outs.append(svs)
            done += t
        return outs[0] if len(outs) == 1 else np.vstack(outs)

    # ---------------- 对外接口 ----------------
    def evolve_events(self, circuit: QuantumCircuit, events,
                      apply_noise: bool = True) -> np.ndarray:
        """事件级演化，返回末态状态向量（无测量）。"""
        svs = self._run_events_trajectories(circuit, events, 1, apply_noise)
        return svs[0]

    def run_trajectories_events(self, circuit: QuantumCircuit, events,
                                num_trajectories: Optional[int] = None
                                ) -> TrajectoryResult:
        """采样 num_trajectories 条事件级噪声轨迹。"""
        if num_trajectories is None:
            num_trajectories = self.num_trajectories
        svs = self._run_events_trajectories(circuit, events,
                                            num_trajectories, True)
        return TrajectoryResult(svs)

    def fidelity_events(self, circuit: QuantumCircuit, events,
                        ideal_sv: Optional[np.ndarray] = None,
                        num_trajectories: Optional[int] = None) -> float:
        """事件级平均态保真度 F = mean_t |<psi_ideal|psi_t>|^2。

        ideal 与调度无关（酉演化与编排无关），用无噪声串行演化作参考。
        """
        if num_trajectories is None:
            num_trajectories = self.num_trajectories
        if ideal_sv is None:
            ideal_sv = self._evolve(circuit, apply_noise=False)
        res = self.run_trajectories_events(circuit, events, num_trajectories)
        return res.fidelity(ideal_sv)

    def run_events(self, circuit: QuantumCircuit, events,
                   shots: Optional[int] = None) -> Dict[str, int]:
        """事件级 counts：每 shot 一条独立噪声轨迹后采样测量结果。"""
        if shots is None:
            shots = self.config.shots
        measured = sorted({
            circuit.find_bit(q).index
            for inst in circuit.data
            if inst.operation.name.lower() == "measure"
            for q in inst.qubits
        })
        if not measured:
            measured = list(range(self.n_qubits))
        actions, total = self._prepare_events(circuit, events)
        counts: Dict[str, int] = {}
        for _ in range(shots):
            svs = np.tile(self._initial_state(), (1, 1))
            if circuit.global_phase:
                svs = svs * np.exp(1j * float(circuit.global_phase))
            svs = self._run_actions(svs, actions, total, True)
            outcome = self._sample(svs[0], measured)
            counts[outcome] = counts.get(outcome, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# 比特子集缩减（镜像 v1 _reduce_phys_circuit_for_fidelity，保留 v2 扩展字段）
# ---------------------------------------------------------------------------
def _reduce_phys_circuit_for_fidelity_v2(phys_circuit: QuantumCircuit,
                                         config: NoiseConfig
                                         ) -> Tuple[QuantumCircuit, NoiseConfig,
                                                    Optional[Dict[int, int]]]:
    """将物理电路 + 噪声配置裁剪到实际被作用的比特子集。

    返回 (reduced_circuit, reduced_config, remap)；remap 为 old->new 比特
    映射（无需缩减或缩减失败时为 None，电路/配置原样返回）。与 v1 版本
    精确等价（未用比特恒为 |0>，对保真度无影响），额外保留
    always_on_zz / swap_xtalk 并对 always_on_zz 的键做重映射。
    """
    cfg = config
    n = len(cfg.t1_times)
    used = set()
    for inst in phys_circuit.data:
        oname = inst.operation.name.lower()
        if oname in ("measure", "barrier"):
            continue
        for q in inst.qubits:
            used.add(q._index)
    if len(used) == n:
        return phys_circuit, cfg, None

    used_list = sorted(used)
    remap = {old: new for new, old in enumerate(used_list)}
    k = len(used_list)
    rc = QuantumCircuit(k)
    rc.global_phase = phys_circuit.global_phase
    for inst in phys_circuit.data:
        oname = inst.operation.name.lower()
        if oname in ("measure", "barrier"):
            continue
        new_qs = [rc.qubits[remap[q._index]] for q in inst.qubits]
        rc.append(inst.operation, new_qs)

    t1 = [cfg.t1_times[i] for i in used_list]
    t2 = [cfg.t2_times[i] for i in used_list]
    freq = [cfg.freq_ghz[i] for i in used_list] if cfg.freq_ghz else None
    sqe = ([cfg.single_q_gate_error[i] for i in used_list]
           if isinstance(cfg.single_q_gate_error, (list, tuple))
           else cfg.single_q_gate_error)
    ro = ([cfg.readout_error[i] for i in used_list]
          if cfg.readout_error is not None else None)

    used_set = set(used_list)
    cm = [(remap[a], remap[b]) for (a, b) in cfg.coupling_map
          if a in used_set and b in used_set]
    cm_qubits = set()
    for a, b in cm:
        cm_qubits.add(a)
        cm_qubits.add(b)
    if not cm_qubits.issuperset(set(range(k))):
        return phys_circuit, cfg, None

    tqe = cfg.two_q_gate_error
    if isinstance(tqe, dict):
        new_tqe: dict = {}
        for (a, b), e in tqe.items():
            if a in used_set and b in used_set:
                new_tqe[(remap[a], remap[b])] = e
                new_tqe[(remap[b], remap[a])] = e
    else:
        new_tqe = tqe

    cts = cfg.crosstalk_strength
    if cts is not None:
        new_cts: dict = {}
        for (a, b), e in cts.items():
            if a in used_set and b in used_set:
                new_cts[(remap[a], remap[b])] = e
                new_cts[(remap[b], remap[a])] = e
    else:
        new_cts = None

    ao = getattr(cfg, "always_on_zz", None)
    if ao is not None:
        new_ao: dict = {}
        for (a, b), r in ao.items():
            if a in used_set and b in used_set:
                new_ao[(remap[a], remap[b])] = r
    else:
        new_ao = None

    reduced = EventNoiseConfig(
        t1_times=t1,
        t2_times=t2,
        freq_ghz=freq,
        single_q_gate_error=sqe,
        two_q_gate_error=new_tqe,
        coupling_map=cm,
        readout_error=ro,
        crosstalk_strength=new_cts,
        single_gate_time=cfg.single_gate_time,
        two_gate_time=cfg.two_gate_time,
        idle_time=cfg.idle_time,
        shots=cfg.shots,
        always_on_zz=new_ao,
        swap_xtalk=bool(getattr(cfg, "swap_xtalk", False)),
        swap_noise=bool(getattr(cfg, "swap_noise", True)),
    )
    return rc, reduced, remap


# ---------------------------------------------------------------------------
# 独立电路入口 / env 工厂
# ---------------------------------------------------------------------------
def trajectory_circuit_fidelity_events(phys_circuit: QuantumCircuit,
                                       config: NoiseConfig,
                                       num_trajectories: int = 16,
                                       seed: Optional[int] = None,
                                       durations: Optional[Dict[str, float]] = None,
                                       ) -> float:
    """对一条物理电路按事件级 ASAP 调度计算轨迹平均态保真度。

    与 v1 trajectory_circuit_fidelity(scheduled=True) 的区别：per-gate 精细
    时长（durations 表，缺省镜像 GATE_DURATION_TABLE）、串扰按重叠时长缩放、
    无需 transpile（swap 原生支持且带 3×CX 等价噪声）。
    用于 SABRE 等基线在公平条件下的调度感知保真度评估。
    """
    rc, rconfig, _remap = _reduce_phys_circuit_for_fidelity_v2(
        phys_circuit, config)
    sim = EventTrajectorySimulator(rconfig, num_trajectories=num_trajectories,
                                   seed=seed)
    events = schedule_phys_circuit_events(rc, durations)
    return sim.fidelity_events(rc, events)


def make_event_fidelity_fn(config: NoiseConfig,
                           num_trajectories: int = 16,
                           seed: Optional[int] = None,
                           durations: Optional[Dict[str, float]] = None):
    """构造 RoutingEnv 的 fidelity_fn hook（事件级调度感知）。

    返回 fn(env) -> float：
    1. 优先使用 env 事件级调度日志（env.timing.schedule_log，事件级 ASAP，
       精确 start/end）；
    2. 日志缺失或与电路校验不一致时，回退到对物理电路的平铺 ASAP 事件调度
       （schedule_phys_circuit_events）。

    训练侧每个 episode 的噪声配置会被扰动，需按 episode 重新构造此函数。
    """
    def event_fidelity(env) -> float:
        rc, rconfig, remap = _reduce_phys_circuit_for_fidelity_v2(
            env._phys_circuit, config)
        sim = EventTrajectorySimulator(rconfig,
                                       num_trajectories=num_trajectories,
                                       seed=seed)
        timing = getattr(env, "timing", None)
        log = getattr(timing, "schedule_log", None) if timing is not None else None
        events = None
        if log:
            cand = timing_log_to_events(log, circuit=rc, remap=remap)
            try:
                validate_events(cand, rc)
                events = cand
            except (ValueError, NotImplementedError):
                events = None
        if events is None:
            events = schedule_phys_circuit_events(rc, durations)
        return sim.fidelity_events(rc, events)

    return event_fidelity


__all__ = [
    "Event",
    "EventNoiseConfig",
    "EventTrajectorySimulator",
    "timing_log_to_events",
    "schedule_phys_circuit_events",
    "validate_events",
    "trajectory_circuit_fidelity_events",
    "make_event_fidelity_fn",
]
