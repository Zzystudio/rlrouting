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

EDGE_TYPES = ["precedes", "couples", "maps_to"]
EDGE_TYPE_INDEX: Dict[str, int] = {e: i for i, e in enumerate(EDGE_TYPES)}
NUM_EDGE_TYPES = len(EDGE_TYPES)

NODE_FEATURE_DIM = 32
EDGE_FEATURE_DIM = 16

# ---------------------------------------------------------------------------
# 归一化尺度
# ---------------------------------------------------------------------------
_T1_SCALE = 100.0
_T2_SCALE = 100.0
_FREQ_SCALE = 5.0
_READOUT_SCALE = 0.1
_ERROR_SCALE = 0.05


@dataclass
class HardwareFeatures:
    """从 NoiseConfig 预计算、归一化后的硬件噪声特征。"""

    num_qubits: int
    t1: np.ndarray              # (Q,) 归一化 T1
    t2: np.ndarray              # (Q,) 归一化 T2
    freq: np.ndarray            # (Q,) 归一化频率
    readout: np.ndarray         # (Q,) 归一化读出错误率
    single_q_err: np.ndarray    # (Q,) 归一化单比特门错误率（每比特独立）
    two_q_err: np.ndarray       # (Q, Q) 归一化双比特门错误率（按耦合边）
    adj: np.ndarray             # (Q, Q) 0/1 邻接矩阵
    zz: np.ndarray              # (Q, Q) 归一化 ZZ 串扰强度
    dist: np.ndarray            # (Q, Q) 最短路径距离（归一化）
    path_err: np.ndarray        # (Q, Q) 最短跳数路径上的最小 Σ e_edge（归一化边误差）
    qubit_template: np.ndarray  # (Q, 32) 硬件不变的 qubit 特征模板
    coupling_index: np.ndarray  # (2, 2*|E|) 双向耦合边索引
    coupling_template: np.ndarray  # (2*|E|, 16) 硬件不变的耦合边特征模板

    @classmethod
    def from_noise_config(cls, config) -> "HardwareFeatures":
        n = len(config.t1_times)
        t1 = np.asarray(config.t1_times, dtype=float) / _T1_SCALE
        t2 = np.asarray(config.t2_times, dtype=float) / _T2_SCALE
        freq = np.asarray(config.freq_ghz, dtype=float) / _FREQ_SCALE

        if config.readout_error is None:
            ro = np.full(n, 0.02)
        else:
            ro = np.asarray(config.readout_error, dtype=float)
        readout = ro / _READOUT_SCALE

        if isinstance(config.single_q_gate_error, (list, tuple, np.ndarray)):
            sqe = np.asarray(config.single_q_gate_error, dtype=float) / _ERROR_SCALE
        else:
            sqe = np.full(n, config.single_q_gate_error / _ERROR_SCALE)

        adj = np.zeros((n, n), dtype=float)
        zz = np.zeros((n, n), dtype=float)
        tqe = np.zeros((n, n), dtype=float)
        for (q1, q2) in config.coupling_map:
            adj[q1, q2] = 1.0
            adj[q2, q1] = 1.0
            strength = _crosstalk_strength(config, q1, q2)
            zz[q1, q2] = strength / _ERROR_SCALE
            zz[q2, q1] = strength / _ERROR_SCALE
            if isinstance(config.two_q_gate_error, dict):
                err = config.two_q_gate_error.get((q1, q2), 0.01)
            else:
                err = config.two_q_gate_error
            tqe[q1, q2] = err / _ERROR_SCALE
            tqe[q2, q1] = err / _ERROR_SCALE

        dist = _shortest_path(adj, n)
        hops, path_err = _min_hop_path_err(adj, tqe, n)
        coupling_list = list(config.coupling_map)
        qubit_template = _build_qubit_template(n, t1, t2, freq, readout, sqe, tqe, adj, zz)
        coupling_index, coupling_template = _build_coupling_template(
            coupling_list, tqe, zz, t1, t2, freq, readout, sqe, adj)
        return cls(
            num_qubits=n,
            t1=t1,
            t2=t2,
            freq=freq,
            readout=readout,
            single_q_err=sqe,
            two_q_err=tqe,
            adj=adj,
            zz=zz,
            dist=dist / max(1, n),
            path_err=path_err,
            qubit_template=qubit_template,
            coupling_index=coupling_index,
            coupling_template=coupling_template,
        )

    def dist_noise(self, beta: float) -> np.ndarray:
        """P0-b 噪声加权距离：d_noise = (hops + β·path_err) / max(1, n)。

        - path_err = 最短跳数路径上的最小 Σ e_edge（归一化边误差）；
        - β 满足保序条件 β·k_max·e_max < 1 时，k+1 跳距离严格大于 k 跳，
          噪声只在等跳数路径间做 tie-break；
        - β=0 时与 self.dist 完全一致（向后兼容）。
        """
        key = round(float(beta), 6)
        cache = getattr(self, "_dist_noise_cache", None)
        if cache is None:
            cache = {}
            self._dist_noise_cache = cache
        if key not in cache:
            scale = max(1, self.num_qubits)
            hops = self.dist * scale
            cache[key] = (hops + float(beta) * self.path_err) / scale
        return cache[key]


def _crosstalk_strength(config, q1: int, q2: int) -> float:
    if config.crosstalk_strength is not None:
        return config.crosstalk_strength.get((q1, q2), 0.0) or \
               config.crosstalk_strength.get((q2, q1), 0.0)
    tqe = config.two_q_gate_error
    if isinstance(tqe, dict):
        return 0.1 * (tqe.get((q1, q2), 0.01) or 0.01)
    return 0.1 * tqe


def _shortest_path(adj: np.ndarray, n: int) -> np.ndarray:
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


def _min_hop_path_err(adj: np.ndarray, e_edge: np.ndarray, n: int):
    """每对 (s,t)：最短跳数距离 + 最短跳数路径上的最小 Σ e_edge。

    BFS 分层 + 层内 DP（按跳数非降序处理节点，同层父节点全部处理完后
    子节点的 path_err 才定型）。图不连通时 hops 停留在 n（与 _shortest_path
    的哨兵一致），path_err 保持 inf。
    """
    INF_H = n
    hops = np.full((n, n), INF_H, dtype=float)
    perr = np.full((n, n), np.inf, dtype=float)
    for s in range(n):
        hops[s, s] = 0.0
        perr[s, s] = 0.0
        queue = [s]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            hu = hops[s, u]
            pu = perr[s, u]
            for v in range(n):
                if adj[u, v] <= 0:
                    continue
                alt = pu + float(e_edge[u, v])
                if hops[s, v] > hu + 1:
                    hops[s, v] = hu + 1
                    perr[s, v] = alt
                    queue.append(v)
                elif hops[s, v] == hu + 1 and alt < perr[s, v]:
                    perr[s, v] = alt
    return hops, perr


# ---------------------------------------------------------------------------
# 节点特征构造（32 维）
# ---------------------------------------------------------------------------

def gate_node_feature(
    gate_name: str,
    is_two_qubit: bool,
    is_measure: bool,
    gate_duration_norm: float,
    error_rate_norm: float,
    dag_depth_norm: float,
    remaining_depth_norm: float,
    logical_qubit_0_norm: float,
    logical_qubit_1_norm: float,
    in_degree_norm: float,
    out_degree_norm: float,
    execution_status: float,
    remaining_predecessors_norm: float,
    gate_index_norm: float,
    mapping_distance_norm: float,
    is_adjacent: float,
) -> np.ndarray:
    feat = np.zeros(NODE_FEATURE_DIM, dtype=float)
    idx = GATE_TYPE_INDEX.get(gate_name, GATE_TYPE_INDEX["barrier"])
    feat[idx] = 1.0                                 # 0-11:  gate type one-hot
    feat[12] = 1.0 if is_two_qubit else 0.0         # 12:    is_two_qubit
    feat[13] = 1.0 if is_measure else 0.0            # 13:    is_measure
    feat[14] = gate_duration_norm                    # 14:    gate duration
    feat[15] = error_rate_norm                       # 15:    error rate
    feat[16] = dag_depth_norm                        # 16:    DAG depth
    feat[17] = remaining_depth_norm                  # 17:    remaining depth
    feat[18] = logical_qubit_0_norm                  # 18:    logical qubit 0
    feat[19] = logical_qubit_1_norm                  # 19:    logical qubit 1
    feat[20] = in_degree_norm                        # 20:    in_degree
    feat[21] = out_degree_norm                       # 21:    out_degree
    feat[22] = execution_status                      # 22:    execution status
    feat[23] = remaining_predecessors_norm            # 23:    remaining preds
    feat[24] = gate_index_norm                       # 24:    gate index
    feat[25] = mapping_distance_norm                 # 25:    mapping distance
    feat[26] = is_adjacent                           # 26:    is adjacent
    return feat


def qubit_node_feature(
    phys_index_norm: float,
    t1_norm: float,
    t2_norm: float,
    freq_norm: float,
    readout_err_norm: float,
    single_q_err_norm: float,
    avg_two_q_err_norm: float,
    degree_norm: float,
    neighbor_avg_t1: float,
    neighbor_avg_t2: float,
    neighbor_avg_freq: float,
    max_zz_norm: float,
    neighbor_avg_readout: float,
    occupied: float,
    occupancy_norm: float,
    freq_detuning_norm: float,
    executable_dep_count_norm: float,
    nearest_occupied_dist_norm: float,
    occupied_neighbor_count_norm: float,
) -> np.ndarray:
    feat = np.zeros(NODE_FEATURE_DIM, dtype=float)
    feat[0] = phys_index_norm                        # 0:  physical index
    feat[1] = t1_norm                                 # 1:  T1
    feat[2] = t2_norm                                 # 2:  T2
    feat[3] = freq_norm                               # 3:  freq
    feat[4] = readout_err_norm                        # 4:  readout error
    feat[5] = single_q_err_norm                       # 5:  single-q error
    feat[6] = avg_two_q_err_norm                      # 6:  avg two-q error
    feat[7] = degree_norm                             # 7:  degree
    feat[8] = neighbor_avg_t1                         # 8:  neighbor avg T1
    feat[9] = neighbor_avg_t2                         # 9:  neighbor avg T2
    feat[10] = neighbor_avg_freq                      # 10: neighbor avg freq
    feat[11] = max_zz_norm                            # 11: max ZZ
    feat[12] = neighbor_avg_readout                   # 12: neighbor avg readout
    feat[13] = occupied                               # 13: occupied
    feat[14] = occupancy_norm                         # 14: occupancy
    feat[15] = freq_detuning_norm                     # 15: freq detuning
    feat[16] = executable_dep_count_norm              # 16: executable dep count
    feat[17] = nearest_occupied_dist_norm             # 17: nearest occupied dist
    feat[18] = occupied_neighbor_count_norm           # 18: occupied neighbor count
    return feat


# ---------------------------------------------------------------------------
# 边特征构造（16 维）
# ---------------------------------------------------------------------------

def precedes_edge_feature(
    depth_diff_norm: float,
    src_depth_norm: float,
    tgt_depth_norm: float,
    shared_qubit_norm: float,
    is_critical_path: float,
    src_executable: float,
    tgt_remaining_predecessors_norm: float,
) -> np.ndarray:
    feat = np.zeros(EDGE_FEATURE_DIM, dtype=float)
    feat[0] = 1.0                                     # 0-2: edge type [1,0,0]
    feat[3] = depth_diff_norm                         # 3: DAG depth difference
    feat[4] = src_depth_norm                          # 4: src depth
    feat[5] = tgt_depth_norm                          # 5: tgt depth
    feat[6] = shared_qubit_norm                       # 6: shared qubit
    feat[7] = is_critical_path                        # 7: is critical path
    feat[8] = src_executable                          # 8: src executable
    feat[9] = tgt_remaining_predecessors_norm          # 9: tgt remaining preds
    return feat


def couples_edge_feature(
    two_q_err_norm: float,
    zz_norm: float,
    t1_geom_mean: float,
    t2_geom_mean: float,
    freq_diff_norm: float,
    readout_product_norm: float,
    single_q_err_geom_mean: float,
    both_occupied: float,
    freq_collision: float,
) -> np.ndarray:
    feat = np.zeros(EDGE_FEATURE_DIM, dtype=float)
    feat[1] = 1.0                                     # 0-2: edge type [0,1,0]
    feat[3] = two_q_err_norm                          # 3: two-q error rate
    feat[4] = zz_norm                                 # 4: ZZ crosstalk
    feat[5] = t1_geom_mean                            # 5: T1 geometric mean
    feat[6] = t2_geom_mean                            # 6: T2 geometric mean
    feat[7] = freq_diff_norm                          # 7: freq difference
    feat[8] = readout_product_norm                    # 8: readout product
    feat[9] = single_q_err_geom_mean                  # 9: 1q err geom mean
    feat[10] = both_occupied                          # 10: both occupied
    feat[11] = freq_collision                         # 11: freq collision
    return feat


def maps_to_edge_feature(
    phys_index_norm: float,
    role: float,
    t1_norm: float,
    t2_norm: float,
    freq_norm: float,
    readout_err_norm: float,
    single_q_err_norm: float,
    avg_two_q_err_norm: float,
    occupied: float,
    distance_to_other_norm: float,
) -> np.ndarray:
    feat = np.zeros(EDGE_FEATURE_DIM, dtype=float)
    feat[2] = 1.0                                     # 0-2: edge type [0,0,1]
    feat[3] = phys_index_norm                         # 3: physical index
    feat[4] = role                                    # 4: operand role
    feat[5] = t1_norm                                 # 5: T1
    feat[6] = t2_norm                                 # 6: T2
    feat[7] = freq_norm                               # 7: freq
    feat[8] = readout_err_norm                        # 8: readout error
    feat[9] = single_q_err_norm                       # 9: single-q error
    feat[10] = avg_two_q_err_norm                     # 10: avg two-q error
    feat[11] = occupied                               # 11: occupied
    feat[12] = distance_to_other_norm                 # 12: distance to other
    return feat


# ---------------------------------------------------------------------------
# 模板预计算（硬件不变部分）
# ---------------------------------------------------------------------------

def _build_qubit_template(
    n: int,
    t1: np.ndarray,
    t2: np.ndarray,
    freq: np.ndarray,
    readout: np.ndarray,
    single_q_err: np.ndarray,
    two_q_err: np.ndarray,
    adj: np.ndarray,
    zz: np.ndarray,
) -> np.ndarray:
    feat = np.zeros((n, NODE_FEATURE_DIM), dtype=float)
    median_freq = float(np.median(freq))
    for pq in range(n):
        neighbors = [nb for nb in range(n) if adj[pq, nb] > 0]
        degree = len(neighbors)
        deg_norm = degree / max(1, n - 1)

        na_t1 = float(np.mean([t1[nb] for nb in neighbors])) if neighbors else 0.0
        na_t2 = float(np.mean([t2[nb] for nb in neighbors])) if neighbors else 0.0
        na_freq = float(np.mean([freq[nb] for nb in neighbors])) if neighbors else 0.0
        na_readout = float(np.mean([readout[nb] for nb in neighbors])) if neighbors else 0.0
        max_zz = float(max(zz[pq, nb] for nb in neighbors)) if neighbors else 0.0
        avg_two = float(np.mean([two_q_err[pq, nb] for nb in neighbors])) if neighbors else 0.0
        freq_det = abs(freq[pq] - median_freq) / max(1e-8, median_freq)

        feat[pq, 0] = pq / max(1, n - 1)         # phys_index_norm
        feat[pq, 1] = t1[pq]                     # T1
        feat[pq, 2] = t2[pq]                     # T2
        feat[pq, 3] = freq[pq]                   # freq
        feat[pq, 4] = readout[pq]               # readout
        feat[pq, 5] = single_q_err[pq]           # single_q_err
        feat[pq, 6] = avg_two                   # avg_two_q_err
        feat[pq, 7] = deg_norm                   # degree
        feat[pq, 8] = na_t1                     # neighbor avg T1
        feat[pq, 9] = na_t2                     # neighbor avg T2
        feat[pq, 10] = na_freq                  # neighbor avg freq
        feat[pq, 11] = max_zz                   # max ZZ
        feat[pq, 12] = na_readout               # neighbor avg readout
        feat[pq, 15] = freq_det                # freq_detuning
    return feat


def _build_coupling_template(
    coupling_map: list,
    two_q_err: np.ndarray,
    zz: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    freq: np.ndarray,
    readout: np.ndarray,
    single_q_err: np.ndarray,
    adj: np.ndarray,
) -> tuple:
    src, tgt = [], []
    attrs = []
    for q1, q2 in coupling_map:
        t1_gm = float(np.sqrt(t1[q1] * t1[q2]))
        t2_gm = float(np.sqrt(t2[q1] * t2[q2]))
        freq_diff = abs(freq[q1] - freq[q2])
        read_prod = readout[q1] * readout[q2]
        sq_gm = float(np.sqrt(single_q_err[q1] * single_q_err[q2]))
        freq_col = 1.0 if freq_diff < 0.02 else 0.0
        for s, t in [(q1, q2), (q2, q1)]:
            src.append(s)
            tgt.append(t)
            feat = np.zeros(EDGE_FEATURE_DIM, dtype=float)
            feat[1] = 1.0                     # edge type [0,1,0]
            feat[3] = two_q_err[s, t]         # two-q error rate
            feat[4] = zz[s, t]               # ZZ crosstalk
            feat[5] = t1_gm                  # T1 geom mean
            feat[6] = t2_gm                  # T2 geom mean
            feat[7] = freq_diff              # freq diff
            feat[8] = read_prod              # readout product
            feat[9] = sq_gm                  # 1q err gm
            feat[11] = freq_col              # freq collision
            attrs.append(feat)
    index = np.asarray([src, tgt], dtype=int) if src else np.empty((2, 0), dtype=int)
    attr_arr = np.asarray(attrs, dtype=float) if attrs else np.empty((0, EDGE_FEATURE_DIM), dtype=float)
    return index, attr_arr
