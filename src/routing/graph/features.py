# ============================================================================
# features.py
# 节点 / 边特征编码：将量子电路的 DAG 结构与硬件噪声信息（含 ZZ 串扰）
# 编码为固定维度的张量，供图神经网络消费。
#
# 设计要点（相对参考论文的扩展）
# ---------------------------------
# 1. 节点同时包含「门节点」与「量子比特节点」，构成二部 + 耦合的同质图；
# 2. ZZ 串扰、耦合距离、双比特门错误率等信息编码在「耦合边」(couples) 上，
#    使得 GNN 在做消息传递时天然感知串扰；
# 3. 所有噪声特征均做归一化，便于网络训练。
# ============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 词表与维度常量
# ---------------------------------------------------------------------------
GATE_TYPES = [
    "h", "cx", "rz", "sx", "x", "y", "z", "s", "t", "swap", "measure", "barrier",
]
GATE_TYPE_INDEX: Dict[str, int] = {g: i for i, g in enumerate(GATE_TYPES)}
NUM_GATE_TYPES = len(GATE_TYPES)

# 边类型（在边特征向量中以 one-hot 占据前 NUM_EDGE_TYPES 列）
EDGE_TYPES = ["acts_on", "couples", "precedes"]
EDGE_TYPE_INDEX: Dict[str, int] = {e: i for i, e in enumerate(EDGE_TYPES)}
NUM_EDGE_TYPES = len(EDGE_TYPES)

NODE_FEATURE_DIM = 32
EDGE_FEATURE_DIM = 16

# ---------------------------------------------------------------------------
# 归一化尺度（用于将物理量映射到 ~[0, 1] 区间）
# ---------------------------------------------------------------------------
_T1_SCALE = 100.0       # µs
_T2_SCALE = 100.0       # µs
_FREQ_SCALE = 5.0       # GHz
_READOUT_SCALE = 0.1    # 概率
_ERROR_SCALE = 0.05     # 概率


@dataclass
class HardwareFeatures:
    """从 NoiseConfig 预计算、归一化后的硬件噪声特征。

    所有数组均为「按物理量子比特索引」的顺序存储，并已归一化。
    """

    num_qubits: int
    t1: np.ndarray            # (Q,) 归一化 T1
    t2: np.ndarray            # (Q,) 归一化 T2
    freq: np.ndarray          # (Q,) 归一化频率
    readout: np.ndarray       # (Q,) 归一化读出错误率
    single_q_error: float     # 归一化单比特门错误率
    two_q_error: float        # 归一化双比特门错误率
    adj: np.ndarray           # (Q, Q) 0/1 邻接矩阵（耦合图）
    zz: np.ndarray            # (Q, Q) 归一化 ZZ 串扰强度
    dist: np.ndarray          # (Q, Q) 最短路径距离（归一化）

    @classmethod
    def from_noise_config(cls, config) -> "HardwareFeatures":
        """从 sim.sim.NoiseConfig 构建 HardwareFeatures。"""
        n = len(config.t1_times)
        t1 = np.asarray(config.t1_times, dtype=float) / _T1_SCALE
        t2 = np.asarray(config.t2_times, dtype=float) / _T2_SCALE
        freq = np.asarray(config.freq_ghz, dtype=float) / _FREQ_SCALE
        if config.readout_error is None:
            ro = np.full(n, 0.02)
        else:
            ro = np.asarray(config.readout_error, dtype=float)
        readout = ro / _READOUT_SCALE

        adj = np.zeros((n, n), dtype=float)
        zz = np.zeros((n, n), dtype=float)
        for (q1, q2) in config.coupling_map:
            adj[q1, q2] = 1.0
            adj[q2, q1] = 1.0
            strength = _crosstalk_strength(config, q1, q2)
            zz[q1, q2] = strength
            zz[q2, q1] = strength

        dist = _shortest_path(adj, n)
        return cls(
            num_qubits=n,
            t1=t1,
            t2=t2,
            freq=freq,
            readout=readout,
            single_q_error=config.single_q_gate_error / _ERROR_SCALE,
            two_q_error=config.two_q_gate_error / _ERROR_SCALE,
            adj=adj,
            zz=zz,
            dist=dist / max(1, n),
        )


def _crosstalk_strength(config, q1: int, q2: int) -> float:
    """读取或自动推导某条耦合边上的 ZZ 串扰强度（归一化前原值）。"""
    if config.crosstalk_strength is not None:
        return config.crosstalk_strength.get((q1, q2), 0.0) or \
               config.crosstalk_strength.get((q2, q1), 0.0)
    return 0.1 * config.two_q_gate_error


def _shortest_path(adj: np.ndarray, n: int) -> np.ndarray:
    """基于邻接矩阵的 BFS 最短路径距离（不可达为 n）。"""
    dist = np.full((n, n), n, dtype=float)
    for s in range(n):
        dist[s, s] = 0.0
        queue = [s]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in range(n):
                if adj[u, v] > 0 and dist[s, v] > dist[s, u] + 1:
                    dist[s, v] = dist[s, u] + 1
                    queue.append(v)
    return dist


# ---------------------------------------------------------------------------
# 特征向量构造
# ---------------------------------------------------------------------------
def gate_node_feature(
    gate_name: str,
    is_two_qubit: bool,
    error_rate_norm: float,
    gate_time_norm: float,
    position_norm: float,
    phys_qubits: List[int],
    hw: HardwareFeatures,
) -> np.ndarray:
    """构造一个门节点的 32 维特征向量。

    Parameters
    ----------
    gate_name : 门名称（如 'h', 'cx'）
    is_two_qubit : 是否为双比特门
    error_rate_norm : 已归一化的门错误率
    gate_time_norm : 已归一化的门时间
    position_norm : 在电路中的相对位置 [0, 1]
    phys_qubits : 该门作用的物理比特索引（长度 1 或 2）
    hw : 硬件噪声特征
    """
    feat = np.zeros(NODE_FEATURE_DIM, dtype=float)
    idx = GATE_TYPE_INDEX.get(gate_name, GATE_TYPE_INDEX["barrier"])
    feat[idx] = 1.0
    feat[NUM_GATE_TYPES] = 1.0 if is_two_qubit else 0.0           # is_two_qubit
    feat[NUM_GATE_TYPES + 1] = error_rate_norm                    # error rate
    feat[NUM_GATE_TYPES + 2] = gate_time_norm                     # gate time
    feat[NUM_GATE_TYPES + 3] = position_norm                      # position
    feat[NUM_GATE_TYPES + 4] = 1.0                                # node_type: gate
    feat[NUM_GATE_TYPES + 5] = 0.0                                # node_type: qubit

    # 第一作用比特的噪声特征
    if len(phys_qubits) >= 1:
        p = phys_qubits[0]
        _fill_qubit_noise(feat, NUM_GATE_TYPES + 6, p, hw)
    # 第二作用比特的噪声特征（单比特门时为 0）
    if len(phys_qubits) >= 2:
        p = phys_qubits[1]
        _fill_qubit_noise(feat, NUM_GATE_TYPES + 10, p, hw)
    return feat


def qubit_node_feature(phys_qubit: int, hw: HardwareFeatures) -> np.ndarray:
    """构造一个量子比特节点的 32 维特征向量。"""
    feat = np.zeros(NODE_FEATURE_DIM, dtype=float)
    feat[NUM_GATE_TYPES + 4] = 0.0   # node_type: gate
    feat[NUM_GATE_TYPES + 5] = 1.0   # node_type: qubit
    _fill_qubit_noise(feat, NUM_GATE_TYPES + 6, phys_qubit, hw)
    return feat


def _fill_qubit_noise(feat: np.ndarray, offset: int, phys_qubit: int, hw: HardwareFeatures):
    feat[offset + 0] = hw.t1[phys_qubit]
    feat[offset + 1] = hw.t2[phys_qubit]
    feat[offset + 2] = hw.freq[phys_qubit]
    feat[offset + 3] = hw.readout[phys_qubit]


def acts_on_edge_feature(phys_qubit: int, hw: HardwareFeatures) -> np.ndarray:
    """acts_on 边特征：门节点 -> 量子比特节点。"""
    feat = np.zeros(EDGE_FEATURE_DIM, dtype=float)
    feat[EDGE_TYPE_INDEX["acts_on"]] = 1.0
    feat[NUM_EDGE_TYPES] = phys_qubit / max(1, hw.num_qubits)
    return feat


def couples_edge_feature(q1: int, q2: int, hw: HardwareFeatures) -> np.ndarray:
    """couples 边特征：两物理比特之间的耦合 / 串扰信息（核心创新点）。"""
    feat = np.zeros(EDGE_FEATURE_DIM, dtype=float)
    feat[EDGE_TYPE_INDEX["couples"]] = 1.0
    feat[NUM_EDGE_TYPES + 0] = hw.zz[q1, q2] / _ERROR_SCALE
    feat[NUM_EDGE_TYPES + 1] = hw.dist[q1, q2]
    feat[NUM_EDGE_TYPES + 2] = (hw.t1[q1] + hw.t1[q2]) / 2.0
    feat[NUM_EDGE_TYPES + 3] = (hw.t2[q1] + hw.t2[q2]) / 2.0
    feat[NUM_EDGE_TYPES + 4] = hw.two_q_error
    feat[NUM_EDGE_TYPES + 5] = hw.adj[q1, q2]
    return feat


def precedes_edge_feature(gap_norm: float, qubit_norm: float) -> np.ndarray:
    """precedes 边特征：DAG 中同一比特上前驱门 -> 后继门。"""
    feat = np.zeros(EDGE_FEATURE_DIM, dtype=float)
    feat[EDGE_TYPE_INDEX["precedes"]] = 1.0
    feat[NUM_EDGE_TYPES + 0] = gap_norm
    feat[NUM_EDGE_TYPES + 1] = qubit_norm
    return feat


def mask_edge_by_type(edge_attr: np.ndarray, edge_type: str) -> np.ndarray:
    """仅保留指定类型的边特征，其余置零（用于 Multi-GNN 子网络）。"""
    keep = EDGE_TYPE_INDEX[edge_type]
    mask = (edge_attr[:, keep] > 0).astype(float).reshape(-1, 1)
    return edge_attr * mask
