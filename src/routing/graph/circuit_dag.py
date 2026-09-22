from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .features import (
    HardwareFeatures,
    maps_to_edge_feature,
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    GATE_TYPE_INDEX,
    _T1_SCALE,
)


@dataclass
class GateRecord:
    index: int
    name: str
    qubits: List[int]
    clbits: List[int]
    operation: Any
    is_two_qubit: bool
    is_measure: bool
    predecessors: List[int] = field(default_factory=list)


class CircuitDAG:
    def __init__(self, gates: List[GateRecord], num_logical_qubits: int):
        self.gates = gates
        self.num_gates = len(gates)
        self.num_logical_qubits = num_logical_qubits
        self._depths: Optional[Dict[int, int]] = None
        self._remaining: Optional[Dict[int, int]] = None
        self._successors: Optional[Dict[int, List[int]]] = None
        # 模板缓存（惰性构建）
        self.gate_feat_template: Optional[np.ndarray] = None
        self.dep_edge_index: Optional[np.ndarray] = None
        self.dep_edge_attr_template: Optional[np.ndarray] = None

    @classmethod
    def from_circuit(cls, circuit) -> "CircuitDAG":
        gates: List[GateRecord] = []
        last_on_qubit: Dict[int, int] = {}
        for i, instruction in enumerate(circuit.data):
            operation = instruction.operation
            name = operation.name
            qubits = [q._index for q in instruction.qubits]
            is_two = len(qubits) == 2 or name in ("cx", "swap", "cz", "ecr")
            is_measure = name == "measure"
            rec = GateRecord(
                index=i,
                name=name,
                qubits=qubits,
                clbits=[c._index for c in instruction.clbits],
                operation=operation,
                is_two_qubit=is_two,
                is_measure=is_measure,
            )
            preds = []
            for q in qubits:
                if q in last_on_qubit:
                    preds.append(last_on_qubit[q])
            rec.predecessors = preds
            for q in qubits:
                last_on_qubit[q] = i
            gates.append(rec)
        return cls(gates, circuit.num_qubits)

    # ------------------------------------------------------------------
    def _compute_depths(self):
        if self._depths is not None:
            return
        G = self.num_gates
        depths: Dict[int, int] = {}
        for g in self.gates:
            if not g.predecessors:
                depths[g.index] = 0
            else:
                depths[g.index] = max(depths[p] for p in g.predecessors) + 1
        self._depths = depths

        successors: Dict[int, List[int]] = {i: [] for i in range(G)}
        for g in self.gates:
            for p in g.predecessors:
                successors[p].append(g.index)
        self._successors = successors

        remaining: Dict[int, int] = {}
        for g in sorted(self.gates, key=lambda x: -depths[x.index]):
            if not successors[g.index]:
                remaining[g.index] = 0
            else:
                remaining[g.index] = max(remaining[s] for s in successors[g.index]) + 1
        self._remaining = remaining

    def dag_depths(self) -> Dict[int, int]:
        self._compute_depths()
        return self._depths

    def remaining_depths(self) -> Dict[int, int]:
        self._compute_depths()
        return self._remaining

    def successors(self) -> Dict[int, List[int]]:
        self._compute_depths()
        return self._successors

    def max_depth(self) -> int:
        d = self.dag_depths()
        return max(d.values()) if d else 0

    # ------------------------------------------------------------------
    def build_gate_template(self) -> np.ndarray:
        """预计算电路不变的门特征模板（dims 0-14, 16-21, 24）。"""
        if self.gate_feat_template is not None:
            return self.gate_feat_template
        G = self.num_gates
        M = self.num_logical_qubits
        depths = self.dag_depths()
        remaining = self.remaining_depths()
        succ = self.successors()
        max_depth = self.max_depth()
        max_in_deg = max((len(g.predecessors) for g in self.gates), default=1)
        max_out_deg = max((len(succ[g.index]) for g in self.gates), default=1)

        feat = np.zeros((G, NODE_FEATURE_DIM), dtype=float)
        for g in self.gates:
            dur = 0.3 / _T1_SCALE if g.is_two_qubit else 0.1 / _T1_SCALE
            feat[g.index, :12] = 0.0
            idx = GATE_TYPE_INDEX.get(g.name, GATE_TYPE_INDEX["barrier"])
            feat[g.index, idx] = 1.0
            feat[g.index, 12] = 1.0 if g.is_two_qubit else 0.0
            feat[g.index, 13] = 1.0 if g.is_measure else 0.0
            feat[g.index, 14] = dur
            d = depths.get(g.index, 0)
            feat[g.index, 16] = d / max(1, max_depth)
            feat[g.index, 17] = remaining.get(g.index, 0) / max(1, max_depth)
            lq0 = g.qubits[0] / max(1, M) if g.qubits else 0.0
            lq1 = g.qubits[1] / max(1, M) if len(g.qubits) >= 2 else 0.0
            feat[g.index, 18] = lq0
            feat[g.index, 19] = lq1
            feat[g.index, 20] = len(g.predecessors) / max(1, max_in_deg)
            feat[g.index, 21] = len(succ.get(g.index, [])) / max(1, max_out_deg)
            feat[g.index, 24] = g.index / max(1, G)
        self.gate_feat_template = feat
        return feat

    def build_dep_template(self):
        """预计算依赖边索引和不变特征模板（dims 0-7）。"""
        if self.dep_edge_index is not None:
            return self.dep_edge_index, self.dep_edge_attr_template
        G = self.num_gates
        M = self.num_logical_qubits
        depths = self.dag_depths()
        remaining = self.remaining_depths()
        max_depth = self.max_depth()

        src, tgt = [], []
        attrs = []
        for g in self.gates:
            for p in g.predecessors:
                depth_diff = (depths[g.index] - depths[p]) / max(1, max_depth)
                src_d = depths[p] / max(1, max_depth)
                tgt_d = depths[g.index] / max(1, max_depth)
                shared_qubits = set(self.gates[p].qubits) & set(g.qubits)
                sq = list(shared_qubits)[0] / max(1, M) if shared_qubits else 0.0
                is_crit = 1.0 if remaining[g.index] == 0 else 0.0
                src.append(p)
                tgt.append(g.index)
                feat = np.zeros(EDGE_FEATURE_DIM, dtype=float)
                feat[0] = 1.0
                feat[3] = depth_diff
                feat[4] = src_d
                feat[5] = tgt_d
                feat[6] = sq
                feat[7] = is_crit
                attrs.append(feat)

        self.dep_edge_index = np.asarray([src, tgt], dtype=int) if src else np.empty((2, 0), dtype=int)
        self.dep_edge_attr_template = np.asarray(attrs, dtype=float) if attrs else np.empty((0, EDGE_FEATURE_DIM), dtype=float)
        return self.dep_edge_index, self.dep_edge_attr_template

    def two_qubit_gates(self) -> List[GateRecord]:
        return [g for g in self.gates if g.is_two_qubit and not g.is_measure]

    def one_qubit_gates(self) -> List[GateRecord]:
        return [g for g in self.gates if not g.is_two_qubit and not g.is_measure]


@dataclass
class RoutingGraphData:
    num_gates: int                       # G
    num_physical: int                    # P
    gate_feat: np.ndarray                # (G, 32)
    dep_edge_index: np.ndarray           # (2, E_dep)  gate→gate
    dep_edge_attr: np.ndarray            # (E_dep, 16)
    qubit_feat: np.ndarray               # (P, 32)
    coupling_edge_index: np.ndarray      # (2, E_couple)  qubit↔qubit
    coupling_edge_attr: np.ndarray       # (E_couple, 16)
    map_edge_index: np.ndarray           # (2, E_map)  gate→qubit
    map_edge_attr: np.ndarray            # (E_map, 16)
    coupling_edges: List[Tuple[int, int]]  # 物理耦合边（用于 RL 动作空间）
    _pyg_data: Any = None               # PyG Data 缓存

    def to_pyg(self, subgraph: str = "full"):
        from torch_geometric.data import Data

        if subgraph == "logic":
            x = torch.tensor(self.gate_feat, dtype=torch.float)
            edge_index = torch.tensor(self.dep_edge_index, dtype=torch.long)
            edge_attr = torch.tensor(self.dep_edge_attr, dtype=torch.float)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        if subgraph == "physics":
            x = torch.tensor(self.qubit_feat, dtype=torch.float)
            edge_index = torch.tensor(self.coupling_edge_index, dtype=torch.long)
            edge_attr = torch.tensor(self.coupling_edge_attr, dtype=torch.float)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        if subgraph == "mapping":
            x = np.concatenate([self.gate_feat, self.qubit_feat], axis=0)
            x = torch.tensor(x, dtype=torch.float)
            edge_index = torch.tensor(self.map_edge_index, dtype=torch.long)
            edge_attr = torch.tensor(self.map_edge_attr, dtype=torch.float)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        P = self.num_physical
        G = self.num_gates

        if subgraph == "full" and self._pyg_data is not None:
            return self._pyg_data

        gate_x = self.gate_feat
        qubit_x = self.qubit_feat
        x = torch.tensor(np.concatenate([gate_x, qubit_x], axis=0), dtype=torch.float)

        edges = []
        attrs = []
        if self.dep_edge_index.shape[1] > 0:
            edges.append(self.dep_edge_index)
            attrs.append(self.dep_edge_attr)
        if self.coupling_edge_index.shape[1] > 0:
            off = np.full_like(self.coupling_edge_index, G)
            edges.append(self.coupling_edge_index + off)
            attrs.append(self.coupling_edge_attr)
        if self.map_edge_index.shape[1] > 0:
            idx = self.map_edge_index.copy()
            idx[1] += G
            edges.append(idx)
            attrs.append(self.map_edge_attr)

        if edges:
            edge_index = torch.tensor(np.concatenate(edges, axis=1), dtype=torch.long)
            edge_attr = torch.tensor(np.concatenate(attrs, axis=0), dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float)

        result = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        if subgraph == "full":
            self._pyg_data = result
        return result


@dataclass
class TimingState:
    """时钟化 env 的时序状态快照，供 build_routing_graph 填充 GNN 空余维度。

    None（默认）时所有时序列保持 0——旧路径逐位不变（doc/20260920训练方案.md
    §2.1）。维度分配：qubit 19-24 / gate 27-30 / coupling 12-14 / dep 10。
    """

    t: float = 0.0                              # 当前时钟 T（µs）
    busy_until: Optional[np.ndarray] = None     # [P] 每物理比特锁释放时刻
    last_free: Optional[np.ndarray] = None      # [P] 上次门结束时刻（-1=未激活）
    qubit_idle_time: Optional[np.ndarray] = None   # [P] 累计空闲 µs
    qubit_crosstalk: Optional[np.ndarray] = None   # [P] 累计 ZZ 曝露（zz·µs）
    run_kind: Optional[np.ndarray] = None       # [P] 在飞类型：1=swap -1=2q 0=无
    parallel_usage: Optional[np.ndarray] = None # [P,P] 并行使用计数
    gate_in_flight: Optional[set] = None        # 已物化未完成 gate idx
    launchable_2q: Optional[set] = None         # EXEC 候选（launchable）gate idx
    chain_dur: Optional[dict] = None            # gate idx -> 前置未物化 1Q 尾链时长
    ready_age: Optional[dict] = None            # gate idx -> 距首次可 launch 的 SKIP 数
    max_dur: float = 0.9                        # 归一化基准（SWAP 时长）
    ready_age_cap: float = 16.0                 # ready_age 归一化分母


def build_routing_graph(
    dag: CircuitDAG,
    mapping: List[int],
    hw: HardwareFeatures,
    coupling_map: List[Tuple[int, int]],
    single_gate_time: float = 0.1,
    two_gate_time: float = 0.3,
    executed_mask: Optional[np.ndarray] = None,
    executable_2q: Optional[set] = None,
    timing_state: Optional[TimingState] = None,
) -> RoutingGraphData:
    """构建路由图（每步调用）。

    优化：静态量（门/依赖边模板、每物理比特邻居结构、maps-to 端点表等）
    按电路/硬件缓存一次；动态量（executed/executable/mapping 相关特征）
    全部 numpy 向量化，消除逐步 O(G) Python 循环。输出与逐字原实现一致。
    """
    G = dag.num_gates
    M = dag.num_logical_qubits
    P = hw.num_qubits

    if executed_mask is None:
        executed_mask = np.zeros(G, dtype=bool)

    cache = getattr(dag, "_routing_graph_cache", None)
    if cache is None or cache.get("G") != G:
        cache = _build_graph_static_cache(dag, hw)
        dag._routing_graph_cache = cache

    mapping_arr = np.asarray(mapping, dtype=np.int64)
    dist = hw.dist

    # ---- 占用掩码（向量化）----
    occupied_mask = np.zeros(P, dtype=bool)
    occ_vals = mapping_arr[(mapping_arr >= 0) & (mapping_arr < P)]
    occupied_mask[occ_vals] = True

    # ---- pending_count（向量化：未执行门的操作数物理位直方图）----
    unexec = ~executed_mask
    pending_count = np.zeros(P, dtype=float)
    if unexec.any():
        gq0 = cache["gate_q0"][unexec]
        gq1 = cache["gate_q1"][unexec]
        parts = [q[q >= 0] for q in (gq0, gq1) if (q >= 0).any()]
        if parts:
            phys = mapping_arr[np.concatenate(parts)]
            pending_count = np.bincount(phys, minlength=P).astype(float)
    total_pending = float(pending_count.sum())

    # ---- executable_on_qubit（小集合，直接迭代）----
    executable_on_qubit: Dict[int, int] = {}
    if executable_2q:
        gates = dag.gates
        for gi in executable_2q:
            g = gates[gi]
            if not g.is_two_qubit:
                continue
            for q in g.qubits:
                pq = mapping[q]
                executable_on_qubit[pq] = executable_on_qubit.get(pq, 0) + 1
    max_exec = max(executable_on_qubit.values()) if executable_on_qubit else 1

    nearest_dist = _nearest_occupied_distance_multi(hw.adj, occupied_mask)
    diameter = cache["diameter"]
    max_dist = cache["max_dist"]

    # ---- Gate node features（模板拷贝 + 动态列向量化）----
    gate_feat = cache["gate_template"].copy()
    gate_q0, gate_q1 = cache["gate_q0"], cache["gate_q1"]
    has_q0 = gate_q0 >= 0
    has_q1 = gate_q1 >= 0
    phys0 = np.zeros(G, dtype=np.int64)
    phys1 = np.zeros(G, dtype=np.int64)
    phys0[has_q0] = mapping_arr[gate_q0[has_q0]]
    phys1[has_q1] = mapping_arr[gate_q1[has_q1]]

    # err（列 15）：measure→readout，1q→single_q_err，2q→max(two_q_err,0)
    err = np.zeros(G, dtype=float)
    m_meas = cache["is_measure"] & has_q0
    m_1q = cache["is_1q"] & has_q0
    m_2q_full = cache["is_2q"] & has_q0 & has_q1
    if m_meas.any():
        err[m_meas] = hw.readout[phys0[m_meas]]
    if m_1q.any():
        err[m_1q] = hw.single_q_err[phys0[m_1q]]
    if m_2q_full.any():
        err[m_2q_full] = np.maximum(hw.two_q_err[phys0[m_2q_full], phys1[m_2q_full]], 0.0)
    m_2q_partial = cache["is_2q"] & ~has_q1
    if m_2q_partial.any():
        err[m_2q_partial] = hw.two_q_err.max()

    # exec_status（列 22）：2.0 已执行 > 1.0 可执行 2Q / 就绪 1Q（含 measure）> 0
    npred = cache["npred"]
    pred_mat = cache["pred_mat"]
    pred_valid = cache["pred_valid"]
    n_exec_preds = (
        pred_valid & executed_mask[pred_mat.clip(min=0)]
    ).sum(axis=1)
    all_preds_exec = n_exec_preds == npred
    status = np.zeros(G, dtype=float)
    is_2q_exec = np.zeros(G, dtype=bool)
    if executable_2q:
        for gi in executable_2q:
            if cache["is_2q"][gi]:
                is_2q_exec[gi] = True
    status = np.where(executed_mask, 2.0,
                      np.where(is_2q_exec, 1.0,
                               np.where((~cache["is_2q"]) & all_preds_exec, 1.0, 0.0)))

    # rem_pred_norm（列 23）
    rem_pred_norm = (npred - n_exec_preds) / np.maximum(1, npred)

    # map_dist_norm / is_adj（列 25/26）
    map_dist_norm = np.zeros(G, dtype=float)
    is_adj = np.zeros(G, dtype=float)
    if m_2q_full.any():
        map_dist_norm[m_2q_full] = (dist[phys0[m_2q_full], phys1[m_2q_full]] * P) / max_dist
        is_adj[m_2q_full] = (hw.adj[phys0[m_2q_full], phys1[m_2q_full]] > 0).astype(float)

    gate_feat[:, 15] = err
    gate_feat[:, 22] = status
    gate_feat[:, 23] = rem_pred_norm
    gate_feat[:, 25] = map_dist_norm
    gate_feat[:, 26] = is_adj

    # ---- Qubit node features（模板拷贝 + 动态列向量化）----
    qubit_feat = hw.qubit_template.copy()
    occ_f = occupied_mask.astype(float)
    qubit_feat[:, 13] = occ_f
    qubit_feat[:, 14] = pending_count / max(1, total_pending)
    exec_dep = np.zeros(P, dtype=float)
    for pq, c in executable_on_qubit.items():
        exec_dep[pq] = c
    qubit_feat[:, 16] = exec_dep / max(1, max_exec)
    qubit_feat[:, 17] = nearest_dist / diameter
    neigh_mat = cache["neigh_mat"]
    neigh_valid = cache["neigh_valid"]
    occ_neighbors = (occ_f[neigh_mat.clip(min=0)] * neigh_valid).sum(axis=1)
    degree = cache["degree"]
    qubit_feat[:, 18] = occ_neighbors / np.maximum(1, degree)

    # ---- Precedes edges（动态列 8/9 向量化）----
    dep_index, dep_template = dag.build_dep_template()
    dep_attr_arr = dep_template.copy()
    if dep_attr_arr.shape[0] > 0:
        dep_src = dep_index[0]
        dep_tgt = dep_index[1]
        dep_attr_arr[:, 8] = executed_mask[dep_src].astype(float)
        dep_attr_arr[:, 9] = rem_pred_norm[dep_tgt]

    # ---- Couples edges（动态列 10 向量化）----
    coup_index = hw.coupling_index
    coup_attr_arr = hw.coupling_template.copy()
    if coup_attr_arr.shape[0] > 0:
        E = len(coupling_map)
        q1_arr = np.fromiter((e[0] for e in coupling_map), dtype=np.int64, count=E)
        q2_arr = np.fromiter((e[1] for e in coupling_map), dtype=np.int64, count=E)
        both = (occupied_mask[q1_arr] & occupied_mask[q2_arr]).astype(float)
        coup_attr_arr[0::2, 10] = both
        coup_attr_arr[1::2, 10] = both

    # ---- Maps_to edges（静态 per-pq 模板查表 + 2 个动态列）----
    mt_gate = cache["mt_gate"]
    mt_role = cache["mt_role"]
    mt_qubit = cache["mt_qubit"]
    mt_other = cache["mt_other"]
    map_tgt = mapping_arr[mt_qubit]
    if mt_gate.size > 0:
        map_index = np.asarray([mt_gate, map_tgt], dtype=int)
        map_attr_arr = cache["pq_static"][map_tgt].copy()
        map_attr_arr[:, 2] = 1.0
        map_attr_arr[:, 4] = mt_role
        map_attr_arr[:, 11] = occ_f[map_tgt]
        d12 = np.zeros(mt_gate.size, dtype=float)
        m2e = mt_other >= 0
        if m2e.any():
            other_pq = mapping_arr[mt_other[m2e]]
            d12[m2e] = (dist[map_tgt[m2e], other_pq] * P) / diameter
        map_attr_arr[:, 12] = d12
    else:
        map_index = np.empty((2, 0), dtype=int)
        map_attr_arr = np.empty((0, EDGE_FEATURE_DIM), dtype=float)

    # ---- 时钟化时序特征填充（timing_state=None 时全部保持 0，旧路径逐位不变）----
    if timing_state is not None:
        ts = timing_state
        bu = np.asarray(ts.busy_until, dtype=float) if ts.busy_until is not None else np.zeros(P)
        lf = np.asarray(ts.last_free, dtype=float) if ts.last_free is not None else np.full(P, -1.0)
        md = max(float(ts.max_dur), 1e-9)
        t = float(ts.t)
        # --- qubit 节点 19-24 ---
        lock_rem = np.clip((bu - t) / md, 0.0, 3.0)
        running = (bu > t + 1e-9).astype(float)
        if ts.run_kind is not None:
            rk = np.asarray(ts.run_kind, dtype=float) * running
        else:
            rk = np.zeros(P, dtype=float)
        if ts.qubit_idle_time is not None:
            idle_n = np.clip(np.asarray(ts.qubit_idle_time, float) / 100.0, 0.0, 1.0)
        else:
            idle_n = np.zeros(P, dtype=float)
        if ts.qubit_crosstalk is not None:
            xt_n = np.clip(np.asarray(ts.qubit_crosstalk, float) / (md * max(1, P) * 5.0), 0.0, 1.0)
        else:
            xt_n = np.zeros(P, dtype=float)
        activated = (lf >= 0.0).astype(float)
        qubit_feat[:, 19] = lock_rem
        qubit_feat[:, 20] = running
        qubit_feat[:, 21] = rk
        qubit_feat[:, 22] = idle_n
        qubit_feat[:, 23] = xt_n
        qubit_feat[:, 24] = activated
        # --- gate 节点 27-30 ---
        if ts.gate_in_flight:
            in_fl = np.array([1.0 if gi in ts.gate_in_flight else 0.0 for gi in range(G)], dtype=float)
        else:
            in_fl = np.zeros(G, dtype=float)
        if ts.launchable_2q:
            launchable = np.array([1.0 if gi in ts.launchable_2q else 0.0 for gi in range(G)], dtype=float)
        else:
            launchable = np.zeros(G, dtype=float)
        chain_arr = np.zeros(G, dtype=float)
        if ts.chain_dur:
            for gi, d in ts.chain_dur.items():
                if 0 <= gi < G:
                    chain_arr[gi] = min(float(d) / md, 3.0)
        age_arr = np.zeros(G, dtype=float)
        if ts.ready_age:
            cap = max(float(ts.ready_age_cap), 1e-9)
            for gi, a in ts.ready_age.items():
                if 0 <= gi < G:
                    age_arr[gi] = min(float(a) / cap, 1.0)
        gate_feat[:, 27] = in_fl
        gate_feat[:, 28] = launchable
        gate_feat[:, 29] = chain_arr
        gate_feat[:, 30] = age_arr
        # --- coupling 边 12-14（双向同值）---
        if coupling_map:
            q1_arr = np.fromiter((e[0] for e in coupling_map), dtype=np.int64,
                                 count=len(coupling_map))
            q2_arr = np.fromiter((e[1] for e in coupling_map), dtype=np.int64,
                                 count=len(coupling_map))
            bf = ((bu[q1_arr] <= t + 1e-9) & (bu[q2_arr] <= t + 1e-9)).astype(float)
            busy_min = np.clip((np.minimum(bu[q1_arr], bu[q2_arr]) - t) / md, 0.0, 3.0)
            if ts.parallel_usage is not None:
                pu = np.asarray(ts.parallel_usage, dtype=float)
                pu_n = np.clip(pu[q1_arr, q2_arr] / max(1, G), 0.0, 1.0)
            else:
                pu_n = np.zeros(q1_arr.size, dtype=float)
            for off in (0, 1):
                coup_attr_arr[off::2, 12] = bf
                coup_attr_arr[off::2, 13] = busy_min
                coup_attr_arr[off::2, 14] = pu_n
        # --- dep 边 10：src 已物化未完成 ---
        if dep_index.shape[1] > 0:
            dep_src = dep_index[0]
            src_in_fl = np.array(
                [1.0 if int(s) in (ts.gate_in_flight or set()) else 0.0 for s in dep_src],
                dtype=float)
            dep_attr_arr[:, 10] = src_in_fl

    return RoutingGraphData(
        num_gates=G,
        num_physical=P,
        gate_feat=gate_feat,
        dep_edge_index=dep_index,
        dep_edge_attr=dep_attr_arr,
        qubit_feat=qubit_feat,
        coupling_edge_index=coup_index,
        coupling_edge_attr=coup_attr_arr,
        map_edge_index=map_index,
        map_edge_attr=map_attr_arr,
        coupling_edges=list(coupling_map),
    )


def _build_graph_static_cache(dag: CircuitDAG, hw: HardwareFeatures) -> Dict[str, Any]:
    """预计算 build_routing_graph 的每电路/每硬件静态量（每电路一次）。"""
    G = dag.num_gates
    P = hw.num_qubits
    gates = dag.gates

    # 预热 dag 级模板缓存（内部自带 memo）
    dag.build_gate_template()
    dag.build_dep_template()

    gate_q0 = np.full(G, -1, dtype=np.int64)
    gate_q1 = np.full(G, -1, dtype=np.int64)
    is_measure = np.zeros(G, dtype=bool)
    is_2q = np.zeros(G, dtype=bool)
    npred = np.zeros(G, dtype=np.int64)
    mt_gate_list: List[int] = []
    mt_role_list: List[float] = []
    mt_qubit_list: List[int] = []
    mt_other_list: List[int] = []
    max_in = 0
    for g in gates:
        qs = g.qubits
        if len(qs) > 0:
            gate_q0[g.index] = qs[0]
        if len(qs) > 1:
            gate_q1[g.index] = qs[1]
        if g.is_measure:
            is_measure[g.index] = True
        if g.is_two_qubit:
            is_2q[g.index] = True
        npred[g.index] = len(g.predecessors)
        if len(g.predecessors) > max_in:
            max_in = len(g.predecessors)
        for role, q in enumerate(qs):
            mt_gate_list.append(g.index)
            mt_role_list.append(float(role))
            mt_qubit_list.append(q)
            if g.is_two_qubit and len(qs) == 2:
                mt_other_list.append(qs[1 - role])
            else:
                mt_other_list.append(-1)

    width = max(1, max_in)
    pred_mat = np.zeros((G, width), dtype=np.int64)
    pred_valid = np.zeros((G, width), dtype=bool)
    for g in gates:
        for j, p in enumerate(g.predecessors):
            pred_mat[g.index, j] = p
            pred_valid[g.index, j] = True

    dist = hw.dist
    adj = hw.adj
    max_dist = int(max(1, dist.max() * P))
    diameter = max(1, int(dist.max() * P))

    # 每物理比特邻居结构 + 静态平均两比特错误率
    neigh_mat = np.zeros((P, P), dtype=np.int64)
    neigh_valid = np.zeros((P, P), dtype=bool)
    degree = np.zeros(P, dtype=float)
    avg_two = np.zeros(P, dtype=float)
    for pq in range(P):
        neigh = [n for n in range(P) if adj[pq, n] > 0]
        degree[pq] = float(len(neigh))
        if neigh:
            neigh_mat[pq, :len(neigh)] = neigh
            neigh_valid[pq, :len(neigh)] = True
            avg_two[pq] = np.mean([hw.two_q_err[pq, n] for n in neigh])

    # maps-to 边的 per-pq 静态列模板（cols 3,5-10；2/4/11/12 动态或常量）
    pq_static = np.zeros((P, EDGE_FEATURE_DIM), dtype=float)
    pq_static[:, 3] = np.arange(P) / max(1, P - 1)
    pq_static[:, 5] = hw.t1
    pq_static[:, 6] = hw.t2
    pq_static[:, 7] = hw.freq
    pq_static[:, 8] = hw.readout
    pq_static[:, 9] = hw.single_q_err
    pq_static[:, 10] = avg_two

    return {
        "G": G,
        "gate_template": dag.build_gate_template(),
        "gate_q0": gate_q0,
        "gate_q1": gate_q1,
        "is_measure": is_measure,
        "is_1q": ~is_2q & ~is_measure,
        "is_2q": is_2q,
        "npred": npred,
        "pred_mat": pred_mat,
        "pred_valid": pred_valid,
        "max_dist": max_dist,
        "diameter": diameter,
        "neigh_mat": neigh_mat,
        "neigh_valid": neigh_valid,
        "degree": degree,
        "pq_static": pq_static,
        "mt_gate": np.asarray(mt_gate_list, dtype=np.int64),
        "mt_role": np.asarray(mt_role_list, dtype=float),
        "mt_qubit": np.asarray(mt_qubit_list, dtype=np.int64),
        "mt_other": np.asarray(mt_other_list, dtype=np.int64),
    }


def _nearest_occupied_distance_multi(
    adj: np.ndarray, occupied: np.ndarray
) -> np.ndarray:
    """多源 BFS：与逐点 BFS（_nearest_occupied_distance）结果完全一致。"""
    P = len(occupied)
    dist = np.full(P, float(P))
    sources = np.where(occupied)[0]
    if sources.size == 0:
        return dist
    from collections import deque
    visited = occupied.copy()
    dist[sources] = 0.0
    queue = deque(sources.tolist())
    while queue:
        u = queue.popleft()
        du = dist[u]
        row = adj[u]
        for v in range(P):
            if row[v] > 0 and not visited[v]:
                visited[v] = True
                dist[v] = du + 1.0
                queue.append(v)
    return dist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nearest_occupied_distance(
    adj: np.ndarray, occupied: np.ndarray
) -> np.ndarray:
    P = len(occupied)
    dist = np.full(P, P, dtype=float)
    occ_indices = np.where(occupied)[0]
    if len(occ_indices) == 0:
        return dist
    for s in range(P):
        if occupied[s]:
            dist[s] = 0.0
            continue
        # BFS from s to find nearest occupied
        queue = [s]
        d = 0
        visited = {s}
        found = False
        while queue and not found:
            d += 1
            next_q = []
            for u in queue:
                for v in range(P):
                    if adj[u, v] > 0 and v not in visited:
                        if occupied[v]:
                            dist[s] = d
                            found = True
                            break
                        visited.add(v)
                        next_q.append(v)
                if found:
                    break
            queue = next_q
    return dist
