from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .features import (
    HardwareFeatures,
    gate_node_feature,
    qubit_node_feature,
    precedes_edge_feature,
    couples_edge_feature,
    maps_to_edge_feature,
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    GATE_TYPE_INDEX,
    _T1_SCALE,
    _ERROR_SCALE,
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

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


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
    max_pred = max(max_in_deg, 1)

    # ---- Gate node features ----
    logical_to_physical = {l: p for l, p in enumerate(mapping)}
    physical_to_logical = {p: l for l, p in enumerate(mapping)}

    gate_feat = np.zeros((G, NODE_FEATURE_DIM), dtype=float)
    for g in dag.gates:
        phys = [mapping[q] for q in g.qubits]
        err = 0.0
        dur = 0.0
        if g.is_measure:
            err = hw.readout[phys[0]] if phys else 0.0
            dur = 0.0
        elif not g.is_two_qubit:
            err = hw.single_q_err[phys[0]] if phys else 0.0
            dur = single_gate_time / _T1_SCALE
        else:
            if len(phys) >= 2:
                err = max(hw.two_q_err[phys[0], phys[1]], 0.0)
            else:
                err = hw.two_q_err.max()
            dur = two_gate_time / _T1_SCALE

        dag_d = depths.get(g.index, 0)
        rem_d = remaining.get(g.index, 0)
        lq0 = g.qubits[0] / max(1, M) if g.qubits else 0.0
        lq1 = g.qubits[1] / max(1, M) if len(g.qubits) >= 2 else 0.0
        in_deg = len(g.predecessors) / max(1, max_in_deg)
        out_deg = len(succ.get(g.index, [])) / max(1, max_out_deg)

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
        g_idx_norm = g.index / max(1, G)

        if g.is_two_qubit and len(phys) >= 2:
            map_dist = hw.dist[phys[0], phys[1]] * P
            map_dist_norm = map_dist / max(1, max_dist)
            is_adj = 1.0 if hw.adj[phys[0], phys[1]] > 0 else 0.0
        else:
            map_dist_norm = 0.0
            is_adj = 0.0

        gate_feat[g.index] = gate_node_feature(
            gate_name=g.name,
            is_two_qubit=g.is_two_qubit,
            is_measure=g.is_measure,
            gate_duration_norm=dur,
            error_rate_norm=err,
            dag_depth_norm=dag_d / max(1, max_depth),
            remaining_depth_norm=rem_d / max(1, max_depth),
            logical_qubit_0_norm=lq0,
            logical_qubit_1_norm=lq1,
            in_degree_norm=in_deg,
            out_degree_norm=out_deg,
            execution_status=exec_status,
            remaining_predecessors_norm=rem_pred_norm,
            gate_index_norm=g_idx_norm,
            mapping_distance_norm=map_dist_norm,
            is_adjacent=is_adj,
        )

    # ---- Qubit node features ----
    _compute_qubit_neighbor_stats(hw)
    qubit_feat = np.zeros((P, NODE_FEATURE_DIM), dtype=float)

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

    for pq in range(P):
        neighbors = [n for n in range(P) if hw.adj[pq, n] > 0]
        degree = len(neighbors)
        deg_norm = degree / max(1, P - 1)

        na_t1 = np.mean([hw.t1[n] for n in neighbors]) if neighbors else 0.0
        na_t2 = np.mean([hw.t2[n] for n in neighbors]) if neighbors else 0.0
        na_freq = np.mean([hw.freq[n] for n in neighbors]) if neighbors else 0.0
        na_readout = np.mean([hw.readout[n] for n in neighbors]) if neighbors else 0.0

        max_zz = max(hw.zz[pq, n] for n in neighbors) if neighbors else 0.0

        avg_two_q = np.mean([hw.two_q_err[pq, n] for n in neighbors]) if neighbors else 0.0

        occupied = 1.0 if occupied_mask[pq] else 0.0
        occupancy = pending_count[pq] / max(1, total_pending)

        median_freq = np.median(hw.freq)
        freq_det = abs(hw.freq[pq] - median_freq) / max(1e-8, median_freq)

        exec_dep = executable_on_qubit.get(pq, 0) / max(1, max_exec)

        near_dist = nearest_dist[pq] / max(1, diameter)
        occ_neighbors = sum(1 for n in neighbors if occupied_mask[n])
        occ_neighbor_norm = occ_neighbors / max(1, degree) if degree > 0 else 0.0

        qubit_feat[pq] = qubit_node_feature(
            phys_index_norm=pq / max(1, P - 1),
            t1_norm=hw.t1[pq],
            t2_norm=hw.t2[pq],
            freq_norm=hw.freq[pq],
            readout_err_norm=hw.readout[pq],
            single_q_err_norm=hw.single_q_err[pq],
            avg_two_q_err_norm=avg_two_q,
            degree_norm=deg_norm,
            neighbor_avg_t1=na_t1,
            neighbor_avg_t2=na_t2,
            neighbor_avg_freq=na_freq,
            max_zz_norm=max_zz,
            neighbor_avg_readout=na_readout,
            occupied=occupied,
            occupancy_norm=occupancy,
            freq_detuning_norm=freq_det,
            executable_dep_count_norm=exec_dep,
            nearest_occupied_dist_norm=near_dist,
            occupied_neighbor_count_norm=occ_neighbor_norm,
        )

    # ---- Precedes edges (gate → gate) ----
    dep_src, dep_tgt = [], []
    dep_attr = []
    for g in dag.gates:
        for p in g.predecessors:
            depth_diff = (depths[g.index] - depths[p]) / max(1, max_depth)
            src_d = depths[p] / max(1, max_depth)
            tgt_d = depths[g.index] / max(1, max_depth)
            shared_qubits = set(dag.gates[p].qubits) & set(g.qubits)
            sq = list(shared_qubits)[0] / max(1, M) if shared_qubits else 0.0
            is_crit = 1.0 if remaining[g.index] == 0 else 0.0
            src_exec = 1.0 if executed_mask[p] else 0.0
            n_rem = sum(1 for pp in g.predecessors if not executed_mask[pp])
            tgt_rem = n_rem / max(1, len(g.predecessors))
            dep_src.append(p)
            dep_tgt.append(g.index)
            dep_attr.append(precedes_edge_feature(
                depth_diff_norm=depth_diff,
                src_depth_norm=src_d,
                tgt_depth_norm=tgt_d,
                shared_qubit_norm=sq,
                is_critical_path=is_crit,
                src_executable=src_exec,
                tgt_remaining_predecessors_norm=tgt_rem,
            ))

    dep_index = np.asarray([dep_src, dep_tgt], dtype=int) if dep_src else np.empty((2, 0), dtype=int)
    dep_attr_arr = np.asarray(dep_attr, dtype=float) if dep_attr else np.empty((0, EDGE_FEATURE_DIM), dtype=float)

    # ---- Couples edges (qubit ↔ qubit from coupling_map) ----
    coup_src, coup_tgt = [], []
    coup_attr = []
    for q1, q2 in coupling_map:
        t1_gm = np.sqrt(hw.t1[q1] * hw.t1[q2])
        t2_gm = np.sqrt(hw.t2[q1] * hw.t2[q2])
        freq_diff = abs(hw.freq[q1] - hw.freq[q2])
        read_prod = hw.readout[q1] * hw.readout[q2]
        sq_gm = np.sqrt(hw.single_q_err[q1] * hw.single_q_err[q2])
        both_occ = 1.0 if (occupied_mask[q1] and occupied_mask[q2]) else 0.0
        freq_col = 1.0 if freq_diff < 0.02 else 0.0  # collision threshold
        # bidirectional
        for src, tgt in [(q1, q2), (q2, q1)]:
            coup_src.append(src)
            coup_tgt.append(tgt)
            coup_attr.append(couples_edge_feature(
                two_q_err_norm=hw.two_q_err[src, tgt],
                zz_norm=hw.zz[src, tgt],
                t1_geom_mean=t1_gm,
                t2_geom_mean=t2_gm,
                freq_diff_norm=freq_diff,
                readout_product_norm=read_prod,
                single_q_err_geom_mean=sq_gm,
                both_occupied=both_occ,
                freq_collision=freq_col,
            ))

    coup_index = np.asarray([coup_src, coup_tgt], dtype=int) if coup_src else np.empty((2, 0), dtype=int)
    coup_attr_arr = np.asarray(coup_attr, dtype=float) if coup_attr else np.empty((0, EDGE_FEATURE_DIM), dtype=float)

    # ---- Maps_to edges (gate → qubit) ----
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

_neighbor_cache: Dict[int, Optional[Tuple]] = {}


def _compute_qubit_neighbor_stats(hw: HardwareFeatures):
    global _neighbor_cache
    if hw.num_qubits in _neighbor_cache:
        return
    _neighbor_cache[hw.num_qubits] = None


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
