"""PureRoutingEnv — v0 第一性原理路由 MDP（Level 0: min N_SWAP）。

状态: s = (executed_mask: int, mapping: tuple)   —— 可哈希，供精确求解器做 transposition。
动作: a ∈ {0..E-1}，选一条 coupling 边做 SWAP，r = -1。
转移: 确定性 —— SWAP 交换 mapping 两端 → 级联自动执行（frontier ∩ adjacent 执行到不动点）。

级联自动执行的最优性保持引理（写进设计，v0 依赖它把 MDP 纯 SWAP 化）：
  执行相邻 2Q 门零成本、不改 mapping、只缩减 D_rem（剩余门集合）→ 立即执行弱占优，
  任意最优策略都可以"执行到不动点再决策下一个 SWAP"而不损失最优性。

Terminal: 所有剩余 2Q 门在映射下邻接（dist==1）。已执行门不计；依赖阻塞但已邻接的
门永不再需 SWAP（映射此后不变），因此 terminal 判定只看"剩余 2Q 门是否全部邻接"。

无 clock / lock / timing / crosstalk / fidelity / scheduling —— 与 clocked_v1 完全隔离。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..graph.circuit_dag import CircuitDAG


def raw_hop_distance(coupling_map: List[Tuple[int, int]], n_phys: int) -> np.ndarray:
    """原始跳数距离矩阵 (P, P)，int 类型（v0 用未归一化距离做精确求解与启发式）。"""
    dist = np.full((n_phys, n_phys), n_phys + 1, dtype=int)
    adj = np.zeros((n_phys, n_phys), dtype=bool)
    for (p, q) in coupling_map:
        adj[p, q] = True
        adj[q, p] = True
    for s in range(n_phys):
        dist[s, s] = 0
        frontier = [s]
        head = 0
        while head < len(frontier):
            u = frontier[head]
            head += 1
            for v in range(n_phys):
                if adj[u, v] and dist[s, v] > dist[s, u] + 1:
                    dist[s, v] = dist[s, u] + 1
                    frontier.append(v)
    return dist


@dataclass
class PureRoutingEnv:
    """极简路由环境。

    注意：dag / coupling_map / dist 视为不可变；clone 只拷贝可变状态
    （executed_mask / mapping / swaps_since_exec / swap_count）。
    """

    dag: CircuitDAG
    coupling_map: List[Tuple[int, int]]
    dist: np.ndarray                      # (P, P) 原始跳数距离（int）
    executed_mask: int = 0                # bit i = dag.gates[i] 已执行
    mapping: Tuple[int, ...] = ()         # logical -> physical，长度 = 逻辑比特数
    swaps_since_exec: np.ndarray = None   # (P,) 每物理比特自上次执行以来的换位数（SABRE decay）
    swap_count: int = 0

    # -- 预计算（immutable）------------------------------------------------
    _twoq: List[int] = field(default=None, init=False)       # 2Q 门 index 列表
    _twoq_qubits: List[Tuple[int, int]] = field(default=None, init=False)
    _bit: List[int] = field(default=None, init=False)         # gate.index -> 1<<index
    _pred_mask: List[int] = field(default=None, init=False)   # gate.index -> 前驱 bitmask
    _inv: List[int] = field(default=None, init=False)         # physical -> logical (-1 空)
    _n_phys: int = field(default=0, init=False)

    def __post_init__(self):
        self._n_phys = int(self.dist.shape[0])
        if not self.mapping:
            n = self.dag.num_logical_qubits
            self.mapping = tuple(range(n))
        if self.swaps_since_exec is None:
            self.swaps_since_exec = np.zeros(self._n_phys, dtype=np.float32)
        self._twoq = [g.index for g in self.dag.gates
                      if g.is_two_qubit and not g.is_measure]
        self._twoq_qubits = [(self.dag.gates[i].qubits[0], self.dag.gates[i].qubits[1])
                             for i in self._twoq]
        self._bit = [1 << g.index for g in self.dag.gates]
        self._pred_mask = []
        for g in self.dag.gates:
            m = 0
            for p in g.predecessors:
                m |= self._bit[p]
            self._pred_mask.append(m)
        self._refresh_inv()
        # 起点即级联：执行是零成本弱占优操作，构造时执行到不动点，
        # 与 SABRE"任意时刻免费执行相邻门"语义一致（否则 mask=0 状态被
        # 人为卡住，V* 会被高估——2026-09-24 修复）。
        self._cascade()

    # -- 基础访问 -----------------------------------------------------------
    @property
    def n_phys(self) -> int:
        return self._n_phys

    @property
    def num_edges(self) -> int:
        return len(self.coupling_map)

    @property
    def num_logical(self) -> int:
        return len(self.mapping)

    def _refresh_inv(self):
        inv = [-1] * self._n_phys
        for l, p in enumerate(self.mapping):
            inv[p] = l
        self._inv = inv

    def state_key(self) -> Tuple[int, Tuple[int, ...]]:
        return (self.executed_mask, self.mapping)

    def set_state(self, executed_mask: int, mapping: Tuple[int, ...],
                  swaps_since_exec: Optional[np.ndarray] = None) -> None:
        self.executed_mask = int(executed_mask)
        self.mapping = tuple(mapping)
        self._refresh_inv()
        if swaps_since_exec is not None:
            self.swaps_since_exec = np.asarray(swaps_since_exec, dtype=np.float32).copy()
        self._cascade()  # 状态恢复后同样级联到不动点

    def clone(self) -> "PureRoutingEnv":
        env = PureRoutingEnv(
            dag=self.dag,
            coupling_map=self.coupling_map,
            dist=self.dist,
            executed_mask=self.executed_mask,
            mapping=self.mapping,
            swaps_since_exec=self.swaps_since_exec.copy(),
            swap_count=self.swap_count,
        )
        return env

    # -- 核心语义 -----------------------------------------------------------
    def ready_2q(self) -> List[int]:
        """frontier：前驱全部已执行的 2Q 门（派生量，不独立存储）。"""
        out = []
        for idx in self._twoq:
            b = self._bit[idx]
            if (self.executed_mask & b) == 0 and \
               (self.executed_mask & self._pred_mask[idx]) == self._pred_mask[idx]:
                out.append(idx)
        return out

    def remaining_2q(self) -> List[int]:
        return [idx for idx in self._twoq if (self.executed_mask & self._bit[idx]) == 0]

    def is_terminal(self) -> bool:
        """所有剩余 2Q 门在映射下邻接（无需再 SWAP）。"""
        for idx, (q0, q1) in zip(self._twoq, self._twoq_qubits):
            if (self.executed_mask & self._bit[idx]) == 0:
                if int(self.dist[self.mapping[q0], self.mapping[q1]]) != 1:
                    return False
        return True

    def _cascade(self) -> None:
        """执行到不动点：frontier ∩ adjacent 的 2Q 门全部执行，重置其物理比特的 decay。"""
        changed = True
        while changed:
            changed = False
            for idx, (q0, q1) in zip(self._twoq, self._twoq_qubits):
                b = self._bit[idx]
                if (self.executed_mask & b) != 0:
                    continue
                if (self.executed_mask & self._pred_mask[idx]) != self._pred_mask[idx]:
                    continue
                p0, p1 = self.mapping[q0], self.mapping[q1]
                if int(self.dist[p0, p1]) == 1:
                    self.executed_mask |= b
                    self.swaps_since_exec[p0] = 0.0
                    self.swaps_since_exec[p1] = 0.0
                    changed = True

    def legal_actions(self) -> List[int]:
        """全部耦合边（v0 第一版不剪枝，只排除两端均空的 no-op swap）。"""
        out = []
        for i, (p, q) in enumerate(self.coupling_map):
            if self._inv[p] == -1 and self._inv[q] == -1:
                continue
            out.append(i)
        return out

    def apply_swap(self, e: int) -> None:
        """对边 e 做 SWAP（含空端点移动语义），并维护 decay 计数。"""
        p, q = self.coupling_map[e]
        lp, lq = self._inv[p], self._inv[q]
        m = list(self.mapping)
        if lp != -1 and lq != -1:
            m[lp], m[lq] = m[lq], m[lp]
        elif lp != -1:
            m[lp] = q
        elif lq != -1:
            m[lq] = p
        self.mapping = tuple(m)
        self._refresh_inv()
        self.swaps_since_exec[p] += 1.0
        self.swaps_since_exec[q] += 1.0
        self.swap_count += 1

    def step(self, e: int, compute_obs: bool = False):
        """确定性一步：SWAP(e) → 级联执行。返回 (obs, -1, done, info)。"""
        self.apply_swap(e)
        self._cascade()
        done = self.is_terminal()
        info = {"executed": self.executed_mask, "swaps": self.swap_count}
        return (None, -1.0, done, info)

    def successor_state(self, e: int) -> Tuple[int, Tuple[int, ...]]:
        """MCTS/求解器用：返回 SWAP(e) 后的状态 key（不修改自身）。"""
        p, q = self.coupling_map[e]
        lp, lq = self._inv[p], self._inv[q]
        m = list(self.mapping)
        if lp != -1 and lq != -1:
            m[lp], m[lq] = m[lq], m[lp]
        elif lp != -1:
            m[lp] = q
        elif lq != -1:
            m[lq] = p
        mapping2 = tuple(m)
        mask2 = self.executed_mask
        # 级联执行
        changed = True
        while changed:
            changed = False
            for idx, (q0, q1) in zip(self._twoq, self._twoq_qubits):
                b = self._bit[idx]
                if (mask2 & b) != 0:
                    continue
                if (mask2 & self._pred_mask[idx]) != self._pred_mask[idx]:
                    continue
                if int(self.dist[mapping2[q0], mapping2[q1]]) == 1:
                    mask2 |= b
                    changed = True
        return (mask2, mapping2)

    def all_successors(self) -> List[Tuple[int, Tuple[int, Tuple[int, ...]]]]:
        """(action, successor_state_key) 列表，供求解器/MCTS 使用。"""
        out = []
        for e in self.legal_actions():
            out.append((e, self.successor_state(e)))
        return out
