# ============================================================================
# env.py
# 量子电路路由的强化学习环境（Gymnasium 接口）。
#
# MDP 形式化
# ----------
# 状态 (observation):
#   [GNN 图嵌入 | 映射向量(逻辑->物理归一化) | 进度]
# 动作 (action):
#   0 .. E-1  : 在物理耦合边 e 上执行 SWAP（交换两个物理比特上的逻辑比特）
#   E         : 执行当前可执行的最早双比特门（当目标门两比特已相邻时）
# 奖励 (reward):
#   每次 SWAP      : -swap_penalty（鼓励少插入 SWAP）
#   每次执行门     : +gate_reward（鼓励推进）
#   非法「执行」   : -invalid_penalty
#   终止           : +fid_scale * 预测保真度（最大化真机保真度）
# ============================================================================

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from ..graph.circuit_dag import CircuitDAG, build_routing_graph
from ..graph.features import HardwareFeatures


class RoutingEnv(gym.Env):
    """电路路由环境。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dag: CircuitDAG,
        hw: HardwareFeatures,
        coupling_map: List[Tuple[int, int]],
        predictor=None,
        swap_penalty: float = 0.1,
        gate_reward: float = 0.5,
        invalid_penalty: float = 1.0,
        fid_scale: float = 5.0,
        random_init: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.dag = dag
        self.hw = hw
        self.coupling_map = coupling_map
        self.num_edges = len(coupling_map)
        self.predictor = predictor
        self.swap_penalty = swap_penalty
        self.gate_reward = gate_reward
        self.invalid_penalty = invalid_penalty
        self.fid_scale = fid_scale
        self.random_init = random_init
        self._rng = np.random.default_rng(seed)

        self.action_space = gym.spaces.Discrete(self.num_edges + 1)
        self._embed_dim = self._compute_embed_dim()
        obs_dim = self._embed_dim + self.dag.num_logical_qubits + 1
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (obs_dim,), dtype=np.float32
        )
        self.embedding_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def _compute_embed_dim(self) -> int:
        if self.predictor is None:
            return 0
        import torch
        data = build_routing_graph(
            self.dag, list(range(self.dag.num_logical_qubits)),
            self.hw, self.coupling_map,
        ).to_pyg()
        with torch.no_grad():
            emb = self.predictor.graph_embedding(data)
        return int(emb.shape[-1])

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        n = self.dag.num_logical_qubits
        if self.random_init:
            perm = list(range(n))
            self._rng.shuffle(perm)
            self.mapping = perm
        else:
            self.mapping = list(range(n))
        self.executed: set = set()
        self._update()
        return self._obs(), {}

    def _apply_swap(self, p: int, q: int):
        """交换占据物理比特 p、q 的两个逻辑比特。"""
        inv = {phys: log for log, phys in enumerate(self.mapping)}
        lp, lq = inv[p], inv[q]
        self.mapping[lp], self.mapping[lq] = self.mapping[lq], self.mapping[lp]

    def _update(self):
        """标记所有可执行的单比特门（无需 SWAP 自动执行），并计算可执行双比特门。"""
        changed = True
        while changed:
            changed = False
            for g in self.dag.gates:
                if g.index in self.executed:
                    continue
                if all(p in self.executed for p in g.predecessors):
                    if not g.is_two_qubit:
                        self.executed.add(g.index)
                        changed = True
        # 可执行双比特门：依赖已满足且两逻辑比特在物理图上相邻
        self.executable_2q = []
        for g in self.dag.gates:
            if g.index in self.executed or g.is_two_qubit is False:
                continue
            if all(p in self.executed for p in g.predecessors):
                qa, qb = g.qubits
                pa, pb = self.mapping[qa], self.mapping[qb]
                if self.hw.adj[pa, pb] > 0:
                    self.executable_2q.append(g.index)

    def _build_graph(self):
        return build_routing_graph(
            self.dag, self.mapping, self.hw, self.coupling_map
        ).to_pyg()

    def _obs(self):
        obs_parts = []
        if self.predictor is not None:
            import torch
            data = self._build_graph()
            with torch.no_grad():
                emb = self.predictor.graph_embedding(data).cpu().numpy().astype(np.float32)
            obs_parts.append(emb.reshape(-1))
        else:
            obs_parts.append(np.zeros(self._embed_dim, dtype=np.float32))
        map_vec = np.array(
            [m / max(1, self.dag.num_logical_qubits) for m in self.mapping],
            dtype=np.float32,
        )
        progress = np.array(
            [len(self.executed) / max(1, self.dag.num_gates)], dtype=np.float32
        )
        return np.concatenate(obs_parts + [map_vec, progress]).astype(np.float32)

    def _predict_fidelity(self) -> float:
        if self.predictor is None:
            return 0.0
        import torch
        data = self._build_graph()
        with torch.no_grad():
            fid = self.predictor.predict_fidelity(data).item()
        return float(fid)

    # ------------------------------------------------------------------
    def step(self, action: int):
        reward = 0.0
        if action == self.num_edges:
            # 执行最早的可执行双比特门
            if self.executable_2q:
                g = min(self.executable_2q)
                self.executed.add(g)
                reward += self.gate_reward
                self._update()
            else:
                reward -= self.invalid_penalty
        else:
            p, q = self.coupling_map[action]
            self._apply_swap(p, q)
            reward -= self.swap_penalty
            self._update()

        done = len(self.executed) == self.dag.num_gates
        info: Dict = {}
        if done:
            fid = self._predict_fidelity()
            reward += self.fid_scale * fid
            info["fidelity"] = fid
            info["num_swaps"] = self._count_swaps()
        return self._obs(), reward, done, False, info

    def _count_swaps(self) -> int:
        # 以与恒等映射的偏离程度粗略估计（实际 swap 数在路由回放时统计）
        return sum(1 for i, m in enumerate(self.mapping) if m != i)
