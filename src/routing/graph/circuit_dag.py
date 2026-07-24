# ============================================================================
# circuit_dag.py
# 将 Qiskit 量子电路构建为「门节点 + 量子比特节点」的同质图（PyG Data），
# 并把噪声 / 串扰信息编码进节点与边特征。
#
# 图结构
# ------
# 节点 0 .. G-1         : 门节点（每个电路指令一个）
# 节点 G .. G+M-1       : 量子比特节点（每个逻辑比特一个）
#
# 边：
#   acts_on   : 门节点 -> 其作用的量子比特节点（携带物理比特索引）
#   precedes  : 同一量子比特上前驱门 -> 后继门（DAG 依赖，携带时间间隙）
#   couples   : 双比特门所跨的两个量子比特节点之间（携带 ZZ 串扰 / 距离）
#
# 映射 (mapping: logical -> physical) 改变时，门节点的比特噪声特征与
# couples 边的串扰特征会随之更新 —— 这正是 RL 训练中「状态随动作变化」的来源。
# ============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any

import numpy as np

from .features import (
    HardwareFeatures,
    gate_node_feature,
    qubit_node_feature,
    acts_on_edge_feature,
    couples_edge_feature,
    precedes_edge_feature,
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    GATE_TYPE_INDEX,
    _T1_SCALE,
    _T2_SCALE,
    _ERROR_SCALE,
)


@dataclass
class GateRecord:
    """电路中的单个门指令（拓扑序）。"""
    index: int
    name: str
    qubits: List[int]          # 逻辑比特索引
    clbits: List[int]          # 经典比特索引（测量时非空）
    operation: Any             # Qiskit Operation 对象（用于回放）
    is_two_qubit: bool
    is_measure: bool
    predecessors: List[int] = field(default_factory=list)


class CircuitDAG:
    """电路有向无环图（门 + 比特二部结构）。"""

    def __init__(self, gates: List[GateRecord], num_logical_qubits: int):
        self.gates = gates
        self.num_gates = len(gates)
        self.num_logical_qubits = num_logical_qubits

    @classmethod
    def from_circuit(cls, circuit) -> "CircuitDAG":
        """从 Qiskit QuantumCircuit 解析出门序列与 DAG 依赖边。"""
        gates: List[GateRecord] = []
        # 每个逻辑比特上「最近一次出现的门」索引
        last_on_qubit: dict = {}
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
            # 依赖边：本门依赖于同一比特上的前驱门
            preds = []
            for q in qubits:
                if q in last_on_qubit:
                    preds.append(last_on_qubit[q])
            rec.predecessors = preds
            for q in qubits:
                last_on_qubit[q] = i
            gates.append(rec)

        num_logical = circuit.num_qubits
        return cls(gates, num_logical)

    def two_qubit_gates(self) -> List[GateRecord]:
        return [g for g in self.gates if g.is_two_qubit and not g.is_measure]

    def one_qubit_gates(self) -> List[GateRecord]:
        return [g for g in self.gates if not g.is_two_qubit and not g.is_measure]


@dataclass
class RoutingGraphData:
    """构建好的同质图（可直接转为 PyG Data）。"""
    num_nodes: int
    num_gates: int
    node_features: np.ndarray              # (N, NODE_FEATURE_DIM)
    edge_index: np.ndarray                 # (2, E)
    edge_attr: np.ndarray                  # (E, EDGE_FEATURE_DIM)
    coupling_edges: List[Tuple[int, int]]  # 物理耦合边 (用于 RL 动作空间)

    def to_pyg(self):
        from torch_geometric.data import Data
        import torch
        return Data(
            x=torch.tensor(self.node_features, dtype=torch.float),
            edge_index=torch.tensor(self.edge_index, dtype=torch.long),
            edge_attr=torch.tensor(self.edge_attr, dtype=torch.float),
        )


def build_routing_graph(
    dag: CircuitDAG,
    mapping: List[int],
    hw: HardwareFeatures,
    coupling_map: List[Tuple[int, int]],
    single_gate_time: float = 0.1,
    two_gate_time: float = 0.3,
) -> RoutingGraphData:
    """根据逻辑电路 DAG、当前映射、硬件特征构建路由图。

    Parameters
    ----------
    dag : 电路 DAG
    mapping : 逻辑比特 -> 物理比特 的映射（长度 = 逻辑比特数）
    hw : 硬件噪声特征
    coupling_map : 物理耦合边列表（用于 RL 动作空间）
    """
    G = dag.num_gates
    M = dag.num_logical_qubits
    N = G + M

    node_features = np.zeros((N, NODE_FEATURE_DIM), dtype=float)
    edge_index: List[List[int]] = [[], []]
    edge_attr: List[np.ndarray] = []

    # ---- 门节点特征 ----
    for g in dag.gates:
        phys = [mapping[q] for q in g.qubits]
        if g.name in ("cx", "swap", "cz", "ecr"):
            err = hw.two_q_error
            gtime = two_gate_time / _T1_SCALE
        elif g.is_measure:
            err = hw.readout[phys[0]] if phys else 0.0
            gtime = 0.0
        else:
            err = hw.single_q_error
            gtime = single_gate_time / _T1_SCALE
        pos = g.index / max(1, G)
        node_features[g.index] = gate_node_feature(
            g.name, g.is_two_qubit, err, gtime, pos, phys, hw
        )

    # ---- 量子比特节点特征 ----
    for q in range(M):
        node_features[G + q] = qubit_node_feature(mapping[q], hw)

    # ---- acts_on 边：门 -> 其作用的量子比特节点 ----
    for g in dag.gates:
        for q_logical in g.qubits:
            gi = g.index
            qi = G + q_logical
            edge_index[0].append(gi)
            edge_index[1].append(qi)
            edge_attr.append(acts_on_edge_feature(mapping[q_logical], hw))

    # ---- precedes 边：DAG 依赖 ----
    for g in dag.gates:
        for p in g.predecessors:
            edge_index[0].append(p)
            edge_index[1].append(g.index)
            gap = (g.index - p) / max(1, G)
            qubit_norm = 0.0  # 占位：可扩展为具体比特索引
            edge_attr.append(precedes_edge_feature(gap, qubit_norm))

    # ---- couples 边：双比特门所跨的两物理比特之间 ----
    for g in dag.gates:
        if g.is_two_qubit and len(g.qubits) == 2 and not g.is_measure:
            qa, qb = g.qubits
            pa, pb = mapping[qa], mapping[qb]
            edge_index[0].append(G + qa)
            edge_index[1].append(G + qb)
            edge_attr.append(couples_edge_feature(pa, pb, hw))
            # 无向 -> 反向边
            edge_index[0].append(G + qb)
            edge_index[1].append(G + qa)
            edge_attr.append(couples_edge_feature(pb, pa, hw))

    return RoutingGraphData(
        num_nodes=N,
        num_gates=G,
        node_features=node_features,
        edge_index=np.asarray(edge_index, dtype=int),
        edge_attr=np.asarray(edge_attr, dtype=float),
        coupling_edges=list(coupling_map),
    )
