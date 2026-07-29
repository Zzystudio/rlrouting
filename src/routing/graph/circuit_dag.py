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


def build_routing_graph(
    dag: CircuitDAG,
    mapping: List[int],
    hw: HardwareFeatures,
    coupling_map: List[Tuple[int, int]],
    single_gate_time: float = 0.1,
    two_gate_time: float = 0.3,
    executed_mask: Optional[np.ndarray] = None,
    executable_2q: Optional[set] = None,
) -> RoutingGraphData:
    G = dag.num_gates
    M = dag.num_logical_qubits
    P = hw.num_qubits

    if executed_mask is None:
        executed_mask = np.zeros(G, dtype=bool)

    depths = dag.dag_depths()
    remaining = dag.remaining_depths()
    succ = dag.successors()
    max_depth = dag.max_depth()
    max_dist = int(max(1, hw.dist.max() * P))

    max_in_deg = max((len(g.predecessors) for g in dag.gates), default=1)
    max_out_deg = max((len(succ[g.index]) for g in dag.gates), default=1)

    # ---- 占用掩码（多步复用）----
    occupied_mask = np.zeros(P, dtype=bool)
    for lq, pq in enumerate(mapping):
        occupied_mask[pq] = True

    pending_count = np.zeros(P, dtype=float)
    total_pending = 0
    for g in dag.gates:
        if not executed_mask[g.index]:
            for q in g.qubits:
                pq = mapping[q]
                pending_count[pq] += 1.0
                total_pending += 1

    executable_on_qubit: Dict[int, int] = {}
    for g in dag.gates:
        if g.is_two_qubit and executable_2q and g.index in executable_2q:
            for q in g.qubits:
                pq = mapping[q]
                executable_on_qubit[pq] = executable_on_qubit.get(pq, 0) + 1
    max_exec = max(executable_on_qubit.values()) if executable_on_qubit else 1

    nearest_dist = _nearest_occupied_distance(hw.adj, occupied_mask)
    diameter = int(hw.dist.max() * P)
    diameter = max(1, diameter)

    # ---- Gate node features（模板 + 增量更新）----
    gate_template = dag.build_gate_template()
    gate_feat = gate_template.copy()
    for g in dag.gates:
        phys = [mapping[q] for q in g.qubits]
        if g.is_measure:
            err = hw.readout[phys[0]] if phys else 0.0
        elif not g.is_two_qubit:
            err = hw.single_q_err[phys[0]] if phys else 0.0
        else:
            if len(phys) >= 2:
                err = max(hw.two_q_err[phys[0], phys[1]], 0.0)
            else:
                err = hw.two_q_err.max()

        is_executed = bool(executed_mask[g.index])
        if is_executed:
            exec_status = 2.0
        elif g.is_two_qubit and executable_2q and g.index in executable_2q:
            exec_status = 1.0
        elif not g.is_two_qubit and all(executed_mask[p] for p in g.predecessors):
            exec_status = 1.0
        else:
            exec_status = 0.0

        n_rem_pred = sum(1 for p in g.predecessors if not executed_mask[p])
        rem_pred_norm = n_rem_pred / max(1, len(g.predecessors))

        if g.is_two_qubit and len(phys) >= 2:
            map_dist = hw.dist[phys[0], phys[1]] * P
            map_dist_norm = map_dist / max(1, max_dist)
            is_adj = 1.0 if hw.adj[phys[0], phys[1]] > 0 else 0.0
        else:
            map_dist_norm = 0.0
            is_adj = 0.0

        gate_feat[g.index, 15] = err
        gate_feat[g.index, 22] = exec_status
        gate_feat[g.index, 23] = rem_pred_norm
        gate_feat[g.index, 25] = map_dist_norm
        gate_feat[g.index, 26] = is_adj

    # ---- Qubit node features（模板 + 增量更新）----
    qubit_feat = hw.qubit_template.copy()
    for pq in range(P):
        occupied = 1.0 if occupied_mask[pq] else 0.0
        occupancy = pending_count[pq] / max(1, total_pending)
        exec_dep = executable_on_qubit.get(pq, 0) / max(1, max_exec)
        near_dist = nearest_dist[pq] / max(1, diameter)
        neighbors = [n for n in range(P) if hw.adj[pq, n] > 0]
        degree = len(neighbors)
        occ_neighbors = sum(1 for n in neighbors if occupied_mask[n])
        occ_neighbor_norm = occ_neighbors / max(1, degree) if degree > 0 else 0.0

        qubit_feat[pq, 13] = occupied
        qubit_feat[pq, 14] = occupancy
        qubit_feat[pq, 16] = exec_dep
        qubit_feat[pq, 17] = near_dist
        qubit_feat[pq, 18] = occ_neighbor_norm

    # ---- Precedes edges（模板 + 增量 dims 8-9）----
    dep_index, dep_template = dag.build_dep_template()
    dep_attr_arr = dep_template.copy()
    if dep_attr_arr.shape[0] > 0:
        edge_idx = 0
        for g in dag.gates:
            for p in g.predecessors:
                src_exec = 1.0 if executed_mask[p] else 0.0
                n_rem = sum(1 for pp in g.predecessors if not executed_mask[pp])
                tgt_rem = n_rem / max(1, len(g.predecessors))
                dep_attr_arr[edge_idx, 8] = src_exec
                dep_attr_arr[edge_idx, 9] = tgt_rem
                edge_idx += 1

    # ---- Couples edges（模板 + 增量 dim 10）----
    coup_index = hw.coupling_index
    coup_attr_arr = hw.coupling_template.copy()
    if coup_attr_arr.shape[0] > 0:
        for i in range(0, len(coupling_map) * 2, 2):
            q1, q2 = coupling_map[i // 2]
            both_occ = 1.0 if (occupied_mask[q1] and occupied_mask[q2]) else 0.0
            coup_attr_arr[i, 10] = both_occ
            coup_attr_arr[i + 1, 10] = both_occ

    # ---- Maps_to edges (gate → qubit) — 完全重建 ----
    map_src, map_tgt = [], []
    map_attr = []
    for g in dag.gates:
        for role, q in enumerate(g.qubits):
            pq = mapping[q]
            map_src.append(g.index)
            map_tgt.append(pq)
            neighbors = [n for n in range(P) if hw.adj[pq, n] > 0]
            avg_two = np.mean([hw.two_q_err[pq, n] for n in neighbors]) if neighbors else 0.0
            dist_to_other = 0.0
            if g.is_two_qubit and len(g.qubits) == 2:
                other_role = 1 - role
                other_pq = mapping[g.qubits[other_role]]
                dist_to_other = hw.dist[pq, other_pq] * P
                dist_to_other /= max(1, diameter)
            map_attr.append(maps_to_edge_feature(
                phys_index_norm=pq / max(1, P - 1),
                role=float(role),
                t1_norm=hw.t1[pq],
                t2_norm=hw.t2[pq],
                freq_norm=hw.freq[pq],
                readout_err_norm=hw.readout[pq],
                single_q_err_norm=hw.single_q_err[pq],
                avg_two_q_err_norm=avg_two,
                occupied=1.0 if occupied_mask[pq] else 0.0,
                distance_to_other_norm=dist_to_other,
            ))

    map_index = np.asarray([map_src, map_tgt], dtype=int) if map_src else np.empty((2, 0), dtype=int)
    map_attr_arr = np.asarray(map_attr, dtype=float) if map_attr else np.empty((0, EDGE_FEATURE_DIM), dtype=float)

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
