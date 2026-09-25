"""ExactSolver — v0 精确求解器（A* + 可采纳启发式 + 跨查询共享 memo）。

目标: V*(s) = -min_π N_SWAP(s, π)，即从状态 s 到完成路由的最优剩余 SWAP 数。

可采纳启发式:
    h(s) = max_{g ∈ D_rem^2Q} ( d_M(q1^g, q2^g) - 1 )
引理: 单次 SWAP 使任一逻辑对的距离变化 ≤ 1（SWAP 两端同时恰为 u,v 时它们本已
邻接），而每个剩余 2Q 门最终须达距离 1 → 需 ≥ d-1 次 SWAP → max 可采纳。

Memo 集成（h' = max(h, memo)）:
    memo[t] = 精确 V*(t)。A* 弹出状态 s 时若 s ∈ memo，返回 g + memo[s] 即最优
    （一致性: memo 精确 ⇒ |memo[t]-memo[s]| ≤ 1；h 一致 ⇒ h' 一致；A* 按 f 非降
    弹出 ⇒ 首个被弹出的 memo 状态给出全局最优完整路径）。

跨查询共享: solve 轨迹状态时先深后浅，浅状态搜索中命中 memo 的 h' 直接精确引导。
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..graph.circuit_dag import CircuitDAG
from .pure_env import PureRoutingEnv, raw_hop_distance

INF = float("inf")


class ExactSolver:
    def __init__(self, dag: CircuitDAG, coupling_map: List[Tuple[int, int]],
                 memo: Optional[Dict[Tuple[int, Tuple[int, ...]], int]] = None,
                 max_nodes: int = 2_000_000):
        self.dag = dag
        self.coupling_map = list(coupling_map)
        n_phys = max(max(e) for e in self.coupling_map) + 1
        self.dist = raw_hop_distance(self.coupling_map, n_phys)
        self.max_nodes = max_nodes
        self.memo: Dict[Tuple[int, Tuple[int, ...]], int] = memo if memo is not None else {}

        self._twoq = [g.index for g in dag.gates if g.is_two_qubit and not g.is_measure]
        self._twoq_qubits = [(dag.gates[i].qubits[0], dag.gates[i].qubits[1]) for i in self._twoq]
        self._bit = [1 << g.index for g in dag.gates]
        self._pred_mask = []
        for g in dag.gates:
            m = 0
            for p in g.predecessors:
                m |= self._bit[p]
            self._pred_mask.append(m)
        self._inv_cache: Dict[Tuple[int, ...], List[int]] = {}
        self.last_expanded = 0  # 最近一次 solve 的展开节点数（诊断用）

    # -- 辅助 ---------------------------------------------------------------
    def _inverse(self, mapping: Tuple[int, ...]) -> List[int]:
        inv = self._inv_cache.get(mapping)
        if inv is None:
            inv = [-1] * int(self.dist.shape[0])
            for l, p in enumerate(mapping):
                inv[p] = l
            self._inv_cache[mapping] = inv
        return inv

    def h(self, mask: int, mapping: Tuple[int, ...]) -> int:
        best = 0
        for idx, (q0, q1) in zip(self._twoq, self._twoq_qubits):
            if (mask & self._bit[idx]) != 0:
                continue
            d = int(self.dist[mapping[q0], mapping[q1]])
            if d > 1 and d - 1 > best:
                best = d - 1
        return best

    def is_terminal(self, mask: int, mapping: Tuple[int, ...]) -> bool:
        for idx, (q0, q1) in zip(self._twoq, self._twoq_qubits):
            if (mask & self._bit[idx]) == 0 and \
               int(self.dist[mapping[q0], mapping[q1]]) != 1:
                return False
        return True

    def _successors(self, mask: int, mapping: Tuple[int, ...]) -> List[Tuple[int, int, Tuple[int, ...]]]:
        """(action, mask2, mapping2) 列表；mapping2 为 tuple。"""
        inv = self._inverse(mapping)
        out = []
        for e, (p, q) in enumerate(self.coupling_map):
            lp, lq = inv[p], inv[q]
            if lp == -1 and lq == -1:
                continue
            m = list(mapping)
            if lp != -1 and lq != -1:
                m[lp], m[lq] = m[lq], m[lp]
            elif lp != -1:
                m[lp] = q
            else:
                m[lq] = p
            m2 = tuple(m)
            mask2 = self._cascade(mask, m2)
            out.append((e, mask2, m2))
        return out

    def _cascade(self, mask: int, mapping: Tuple[int, ...]) -> int:
        changed = True
        while changed:
            changed = False
            for idx, (q0, q1) in zip(self._twoq, self._twoq_qubits):
                b = self._bit[idx]
                if (mask & b) != 0:
                    continue
                if (mask & self._pred_mask[idx]) != self._pred_mask[idx]:
                    continue
                if int(self.dist[mapping[q0], mapping[q1]]) == 1:
                    mask |= b
                    changed = True
        return mask

    # -- 主求解 -------------------------------------------------------------
    def solve(self, mask: int, mapping: Tuple[int, ...],
              max_nodes: Optional[int] = None) -> Optional[int]:
        """返回 V*(s)；超预算返回 None。"""
        start = (int(mask), tuple(mapping))
        if start in self.memo:
            return self.memo[start]
        if self.is_terminal(*start):
            self.memo[start] = 0
            return 0

        cap = self.max_nodes if max_nodes is None else max_nodes
        h0 = self.h(*start)
        if start in self.memo:
            h0 = max(h0, self.memo[start])
        heap = [(h0, 0, 0, start)]
        gscore = {start: 0}
        expanded = 0
        counter = 1
        self.last_expanded = 0

        while heap:
            f, g, _, s = heapq.heappop(heap)
            if gscore.get(s) != g:
                continue
            if s in self.memo:
                v = g + self.memo[s]
                self.memo[start] = v
                return v
            if self.is_terminal(*s):
                self.memo[s] = 0
                self.memo[start] = g
                return g
            expanded += 1
            self.last_expanded = expanded
            if expanded > cap:
                return None
            mask_s, mapping_s = s
            for _a, mask2, mapping2 in self._successors(mask_s, mapping_s):
                ns = (mask2, mapping2)
                ng = g + 1
                if ng < gscore.get(ns, INF):
                    gscore[ns] = ng
                    hh = self.h(*ns)
                    if ns in self.memo:
                        hh = max(hh, self.memo[ns])
                    heapq.heappush(heap, (ng + hh, ng, counter, ns))
                    counter += 1
        return None

    def solve_env(self, env: PureRoutingEnv, max_nodes: Optional[int] = None) -> Optional[int]:
        return self.solve(env.executed_mask, env.mapping, max_nodes=max_nodes)

    def optimal_trajectory(self, env: PureRoutingEnv) -> Optional[List[int]]:
        """沿 V* 单调递减的最优动作序列（利用 memo 逐格下降）。"""
        actions = []
        v = self.solve_env(env)
        if v is None:
            return None
        while v > 0:
            succ = env.all_successors()
            chosen = None
            for e, (mask2, mapping2) in succ:
                v2 = self.memo.get((mask2, mapping2))
                if v2 is None:
                    v2 = self.solve(mask2, mapping2)
                if v2 is not None and v2 == v - 1:
                    chosen = e
                    break
            if chosen is None:
                return None
            actions.append(chosen)
            env.step(chosen)
            v -= 1
        return actions


def bfs_solve(dag: CircuitDAG, coupling_map: List[Tuple[int, int]],
              start_mask: int, start_mapping: Tuple[int, ...],
              max_nodes: int = 2_000_000) -> Optional[int]:
    """暴力 BFS 真值（仅测试用，小规模）。返回 min SWAP 数，超预算 None。"""
    solver = ExactSolver(dag, coupling_map)
    start = (start_mask, tuple(start_mapping))
    queue = [(start, 0)]
    seen = {start}
    head = 0
    while head < len(queue):
        s, d = queue[head]
        head += 1
        if solver.is_terminal(*s):
            return d
        if head > max_nodes:
            return None
        mask_s, mapping_s = s
        for _a, mask2, mapping2 in solver._successors(mask_s, mapping_s):
            ns = (mask2, mapping2)
            if ns not in seen:
                seen.add(ns)
                queue.append((ns, d + 1))
    return None
