from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from qiskit import QuantumCircuit

from ..graph.circuit_dag import CircuitDAG, build_routing_graph
from ..gnn.encoder import SubGNN
from ..graph.features import HardwareFeatures
from ..timing import GreedyScheduler, CircuitTiming, schedule_events, FALLBACK_DURATION, GATE_DURATION_TABLE

_GATE_BASE_REWARD_DEFAULT: Dict[str, float] = {
    "cx": 2.0, "h": 0.5, "sx": 0.3, "x": 0.3,
    "rz": 0.3, "y": 0.3, "z": 0.3, "s": 0.3, "t": 0.3,
    "swap": 0.0, "measure": 0.0, "barrier": 0.0,
}

# R2 势函数奖励标定（v2 物理单位，reward_potential=True 时启用）
_POT_PROGRESS_B = 0.045   # 每执行门进度奖励（≈1.5×平均边噪声代价，保证完成优于 stall）
_POT_AVG_COST = 0.01      # 平均边噪声代价（截断惩罚：按未执行门数计）


class RoutingEnv(gym.Env):
    """电路路由环境，三阶段训练奖励模式。

    Stage 1 (routing):
      步级三组件：门执行收益 + SWAP 惩罚 + 错误传播，无终端奖励。
    Stage 2 (noise_aware):
      步级同 Stage1，终端为输出分布差异指标 (D_KL, H_cross, TVD)。
    Stage 3 (fidelity_shaping):
      步级奖励为 0，纯终端差异指标。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dag: CircuitDAG,
        hw: HardwareFeatures,
        coupling_map: List[Tuple[int, int]],
        reward_mode: str = "routing",

        # --- 步级奖励参数 (Stage 1 & 2) ---
        gate_base_reward: Optional[Dict[str, float]] = None,
        swap_cost: float = 0.0,
        invalid_penalty: float = 1.0,
        eta_err: float = 0.5,
        eta_xtalk: float = 0.02,
        eta_xz_step: float = 0.0,
        cnot_cost: float = 0.1,
        # SWAP 边噪声惩罚（v2 口径）：一次 SWAP = 3×CX，按所在耦合边
        # two_q_err 计价（eta_swap_err ≈ 3×eta_err 时完全对价）；默认 0 = 旧行为
        eta_swap_err: float = 0.0,
        # 势函数奖励（R2）：r = 进度奖励 − 物理噪声代价（v2 一致标定）。
        # 取消 +2.0/门 任意完成奖励；SWAP=3×e_edge；映射期虚拟 SWAP 免费；
        # 截断按未执行门数计惩罚（堵 stall 逃逸）。与 reward_mode=routing 组合用
        reward_potential: bool = False,
        # R3 势函数 shaping（objective 与 shaping 分离）：
        #   Φ(s) = −eta_shape·(D_front + α·D_ext)/|F|,|E| 均值口径
        #   r_shape = γ·Φ(s') − Φ(s)，γ 必须与 PPO discount 一致（策略不变性）
        # 终态/截断 Φ=0；D_ext = front layer 之后按 DAG 序前 ext_set_size 个
        # 2Q 门（对齐 qiskit SabreSwap 的 with_lookahead(0.5, 20)）。激活后
        # 旧 eta_dist 相对式距离塑形停用
        shaping_gamma: Optional[float] = None,
        eta_shape: float = 0.3,
        alpha_ext: float = 0.5,
        ext_set_size: int = 20,

        # --- 终端奖励参数 (Stage 2 & 3) ---
        lambda_fid: float = 5.0,
        fidelity_fn: Optional[Callable] = None,

        # --- 距离塑形奖励 (Phase 1) ---
        eta_dist: float = 1.0,

        # --- 门调度 / 时序感知 (timing_aware, 框架 v2 Phase 1) ---
        use_scheduler: bool = False,
        eta_time: float = 0.01,
        eta_xtalk_par: float = 1.0,
        eta_idle: float = 0.005,
        eta_parallel: float = 0.05,
        xtalk_alpha: float = 0.03,
        swap_duration_us: float = 0.9,

        # --- Episode 截断 ---
        max_episode_steps: int = 200,
        unfinished_penalty: float = 0.5,

        max_num_edges: Optional[int] = None,
        max_num_qubits: Optional[int] = None,
        mapping_budget: Optional[int] = None,
        mapping_phase: bool = True,
        random_init: bool = True,
        init_mapping: Optional[Sequence[int]] = None,
        lambda_layout: float = 0.0,
        use_gnn: bool = True,
        gnn: Optional[SubGNN] = None,
        seed: int = 0,
        noise_config=None,
        sabre_fid_map: Optional[Dict[int, float]] = None,
        sref_override: Optional[float] = None,
    ):
        super().__init__()
        self.dag = dag
        self.hw = hw
        self.coupling_map = coupling_map
        self.num_edges = len(coupling_map)
        self.num_qubits = dag.num_logical_qubits
        self.max_num_qubits = max_num_qubits or self.num_qubits
        self.noise_config = noise_config

        self.reward_mode = reward_mode
        self.gate_base_reward = gate_base_reward or _GATE_BASE_REWARD_DEFAULT.copy()
        self.swap_cost = swap_cost
        self.eta_swap_err = eta_swap_err
        self.reward_potential = reward_potential
        self.shaping_gamma = shaping_gamma
        self.eta_shape = eta_shape
        self.alpha_ext = alpha_ext
        self.ext_set_size = ext_set_size
        self.invalid_penalty = invalid_penalty
        self.eta_err = eta_err
        self.eta_xtalk = eta_xtalk
        self.eta_xz_step = eta_xz_step
        self.eta_dist = eta_dist
        self.cnot_cost = cnot_cost

        self.use_scheduler = use_scheduler
        self.eta_time = eta_time
        self.eta_xtalk_par = eta_xtalk_par
        self.eta_idle = eta_idle
        self.eta_parallel = eta_parallel
        self.xtalk_alpha = xtalk_alpha
        self.swap_duration = swap_duration_us
        self.scheduler = GreedyScheduler() if use_scheduler else None
        self.timing = None

        self.lambda_fid = lambda_fid
        self.fidelity_fn = fidelity_fn
        self.sabre_fid_map = sabre_fid_map
        self.sref_override = sref_override
        self.lambda_layout = lambda_layout

        self.random_init = random_init
        self.init_mapping = list(init_mapping) if init_mapping is not None else None
        self.max_episode_steps = max_episode_steps
        self.unfinished_penalty = unfinished_penalty
        self.max_num_edges = max_num_edges or self.num_edges
        self.mapping_budget = mapping_budget if mapping_budget is not None else max(1, self.num_qubits - 1)
        self.enable_mapping_phase = mapping_phase
        self._rng = np.random.default_rng(seed)

        if gnn is not None:
            self._gnn = gnn
        elif use_gnn:
            self._gnn = SubGNN(subgraph="full")
            self._gnn.eval()
        else:
            self._gnn = None
        if self._gnn is not None:
            self._edge_feat_dim = self._gnn.encoder.out_dim * 3 + 5
            self._gnn_dim = self._edge_feat_dim * self.max_num_edges
        else:
            self._edge_feat_dim = 0
            self._gnn_dim = 0

        if self.enable_mapping_phase:
            self.action_space = gym.spaces.Discrete(self.num_edges + 1)
            self.commit_action = self.num_edges
        else:
            self.action_space = gym.spaces.Discrete(self.num_edges)
            self.commit_action = -1
        obs_dim = self._gnn_dim + self.max_num_qubits + (2 if self.enable_mapping_phase else 1)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (obs_dim,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    #  重置
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        n = self.num_qubits
        if self.init_mapping is not None:
            self.mapping = list(self.init_mapping[:n])
        elif self.random_init:
            perm = list(range(n))
            self._rng.shuffle(perm)
            self.mapping = perm
        else:
            self.mapping = list(range(n))
        self.mapping_phase = self.enable_mapping_phase
        self._mapping_swaps = 0
        self.executed: set = set()
        self._swap_counter = 0
        self._episode_step = 0
        self._swap_history: list = []
        self._last_progress_swap: int = 0
        self._xz_errors = np.zeros((n, 2), dtype=float)
        self._phys_circuit = QuantumCircuit(self.hw.num_qubits)
        self._pending_measures = []
        self._pending_swaps = []
        # 有效初始布局：物理线路第一条门执行时刻的映射（映射阶段 commit 后的布局）。
        # reset 时先等于初始映射；映射阶段发生虚拟 SWAP 后在 commit 时被覆盖。
        self._effective_initial_mapping = list(self.mapping)
        self.timing = CircuitTiming.create(self.hw.num_qubits) if self.use_scheduler else None
        self._update()
        if not self.enable_mapping_phase:
            self._auto_execute_batch()
        # 映射阶段：门在 commit 前不执行（_auto_execute_batch 由 commit 触发）
        return self._obs(), {}

    # ------------------------------------------------------------------
    #  核心更新
    # ------------------------------------------------------------------
    def _apply_swap(self, p: int, q: int):
        inv = {phys: log for log, phys in enumerate(self.mapping)}
        lp = inv.get(p)
        lq = inv.get(q)
        if lp is None and lq is None:
            return
        if lp is None:
            self.mapping[lq] = p
        elif lq is None:
            self.mapping[lp] = q
        else:
            self.mapping[lp], self.mapping[lq] = self.mapping[lq], self.mapping[lp]
        self._phys_circuit.swap(p, q)
        # SWAP 本身也耗时（默认 0.9µs = 3×CX），并占用两端 qubit。
        # 关键修复：不再在 step 内串行推进全局时钟，而是登记为「待调度事件」，
        # 交给 _auto_execute_batch_scheduled 与同批就绪门一起做 ASAP 事件级调度，
        # 从而让与本 SWAP 无关比特上的门可以在 SWAP 期间并行执行（与 SABRE 同口径）。
        # 其它不相关比特的空闲仍由各门 start-last_free 记录，不重复计入。
        if self.use_scheduler and self.timing is not None:
            self._pending_swaps.append((p, q, self.swap_duration))

    def _apply_virtual_swap(self, p: int, q: int):
        """映射阶段虚拟 SWAP：仅重排初始映射，不写入物理线路、不计入 SWAP 数。"""
        inv = {phys: log for log, phys in enumerate(self.mapping)}
        lp = inv.get(p)
        lq = inv.get(q)
        if lp is None and lq is None:
            return
        if lp is None:
            self.mapping[lq] = p
        elif lq is None:
            self.mapping[lp] = q
        else:
            self.mapping[lp], self.mapping[lq] = self.mapping[lq], self.mapping[lp]

    def _update(self):
        # 调度模式下，1Q 门不在此处急迫执行（否则绕过事件级调度内核，导致
        # 时长/并行度统计失真）；所有门统一在 _auto_execute_batch_scheduled 中调度。
        # 仅在非调度模式保留「1Q 门自动执行」的遗留行为。
        # 映射阶段一律不执行门：reset 后急迫执行会把 1Q 门钉在映射前的位置，
        # 随后虚拟 SWAP 重排映射导致门与逻辑比特归属错乱（线路损坏）。
        if not self.use_scheduler and not self.mapping_phase:
            changed = True
            while changed:
                changed = False
                for g in self.dag.gates:
                    if g.index in self.executed:
                        continue
                    if all(p in self.executed for p in g.predecessors):
                        if not g.is_two_qubit:
                            self.executed.add(g.index)
                            if not g.is_measure:
                                pq = [self.mapping[q] for q in g.qubits]
                                self._phys_circuit.append(g.operation, pq)
                            changed = True
        self.executable_2q = []
        for g in self.dag.gates:
            if g.index in self.executed or g.is_two_qubit is False:
                continue
            if all(p in self.executed for p in g.predecessors):
                qa, qb = g.qubits
                pa, pb = self.mapping[qa], self.mapping[qb]
                if self.hw.adj[pa, pb] > 0:
                    self.executable_2q.append(g.index)

    # ------------------------------------------------------------------
    #  观测
    # ------------------------------------------------------------------
    def build_graph_data(self):
        """构建 RoutingGraphData（含 executed / executable 状态），供批量 GNN 推理复用。"""
        executed_mask = np.zeros(self.dag.num_gates, dtype=bool)
        for idx in self.executed:
            executed_mask[idx] = True
        graph_data = build_routing_graph(
            self.dag, self.mapping, self.hw, self.coupling_map,
            executed_mask=executed_mask,
            executable_2q=set(self.executable_2q),
        )
        self._last_graph_data = graph_data
        return graph_data

    def _obs(self, qubit_h=None):
        map_vec = np.array(
            [m / max(1, self.num_qubits) for m in self.mapping],
            dtype=np.float32,
        )
        if self.max_num_qubits > self.num_qubits:
            pad = np.zeros(self.max_num_qubits - self.num_qubits, dtype=np.float32)
            map_vec = np.concatenate([map_vec, pad])

        progress = np.array(
            [len(self.executed) / max(1, self.dag.num_gates)], dtype=np.float32
        )
        phase = np.array([1.0 if self.mapping_phase else 0.0], dtype=np.float32)
        if self._gnn is not None:
            self._last_map_vec = map_vec
            self._last_progress = progress
            if qubit_h is None:
                graph_data = self.build_graph_data()
                with torch.no_grad():
                    qubit_h = self._gnn.node_embeddings(graph_data).cpu().numpy()
            sabre_feats = self._sabre_edge_features()
            self._last_sabre_feats = sabre_feats
            edge_feats_list = []
            for i, (p, q) in enumerate(self.coupling_map):
                h_p = qubit_h[p]
                h_q = qubit_h[q]
                edge_feats_list.extend([h_p, h_q, h_p - h_q, sabre_feats[i]])
            edge_feats = np.concatenate(edge_feats_list).astype(np.float32)
            if self.max_num_edges > self.num_edges:
                pad_len = (self.max_num_edges - self.num_edges) * self._edge_feat_dim
                edge_feats = np.pad(edge_feats, (0, pad_len), constant_values=0)
            if self.enable_mapping_phase:
                obs = np.concatenate([edge_feats, map_vec, progress, phase]).astype(np.float32)
            else:
                obs = np.concatenate([edge_feats, map_vec, progress]).astype(np.float32)
            return obs
        if self.enable_mapping_phase:
            return np.concatenate([map_vec, progress, phase]).astype(np.float32)
        return np.concatenate([map_vec, progress]).astype(np.float32)

    # ------------------------------------------------------------------
    #  辅助方法
    # ------------------------------------------------------------------
    def _gate_two_q_err(self, gate_idx: int) -> float:
        g = self.dag.gates[gate_idx]
        if not g.is_two_qubit:
            return 0.0
        qa, qb = g.qubits
        pa, pb = self.mapping[qa], self.mapping[qb]
        return float(self.hw.two_q_err[pa, pb])

    def _gate_crosstalk(self, gate_idx: int) -> float:
        g = self.dag.gates[gate_idx]
        occupied = set(self.mapping)
        total = 0.0
        for q in g.qubits:
            p = self.mapping[q]
            for nb in range(self.hw.num_qubits):
                if self.hw.adj[p, nb] > 0 and nb in occupied:
                    total += float(self.hw.zz[p, nb])
        return total

    # ------------------------------------------------------------------
    #  SABRE 启发式特征 (Phase 1)
    # ------------------------------------------------------------------
    def _ready_2q_gates(self):
        """返回所有前置门已执行但自身未执行的 2Q 门（front_layer，不论是否相邻）。"""
        ready = []
        for g in self.dag.gates:
            if g.index in self.executed:
                continue
            if not g.is_two_qubit:
                continue
            if all(p in self.executed for p in g.predecessors):
                ready.append(g)
        return ready

    def _front_layer_dist(self, mapping=None):
        """当前 front_layer 各门 qubit 对之间的距离和。"""
        if mapping is None:
            mapping = self.mapping
        ready = self._ready_2q_gates()
        if not ready:
            return 0.0
        total = 0.0
        for g in ready:
            qa, qb = g.qubits
            pa, pb = mapping[qa], mapping[qb]
            total += self.hw.dist[pa, pb]
        return total

    def _extended_set_dist(self, mapping=None):
        """Extended set（front layer 之后按 DAG 序的前 ext_set_size 个未执行
        2Q 门，其前驱均已执行或位于 front layer）的距离和与门数——对齐
        qiskit SabreSwap 的 with_lookahead(W=0.5, |E|=20) 口径。"""
        if mapping is None:
            mapping = self.mapping
        ready = {g.index for g in self._ready_2q_gates()}
        dist = self.hw.dist
        total = 0.0
        count = 0
        for g in self.dag.gates:
            if count >= self.ext_set_size:
                break
            idx = g.index
            if idx in self.executed or idx in ready or not g.is_two_qubit:
                continue
            if not all(p in self.executed or p in ready for p in g.predecessors):
                continue
            pa, pb = mapping[g.qubits[0]], mapping[g.qubits[1]]
            total += float(dist[pa, pb])
            count += 1
        return total, count

    def _phi(self) -> float:
        """R3 势函数：Φ(s) = −eta_shape·(D_front/|F| + α·D_ext/|E|)。

        均值口径天然以拓扑直径为界（跨电路尺度稳定）；终态由调用方置 0。"""
        ready = self._ready_2q_gates()
        n_front = max(len(ready), 1)
        d_front = float(self._front_layer_dist()) / n_front
        d_ext, n_ext = self._extended_set_dist()
        d_ext_avg = (d_ext / n_ext) if n_ext else 0.0
        return -self.eta_shape * (d_front + self.alpha_ext * d_ext_avg)

    def _sabre_edge_features(self):
        """为每条 coupling edge 计算 5 维 SABRE 启发式特征。

        优化：ready 集合与 inv 映射每步只构建一次（旧实现每步约 60 次
        O(G) 全门扫描 + 每边重建 dict）。关键观察：边 (p,q) 上的虚拟
        SWAP 只把逻辑 lp 移到 q、逻辑 lq 移到 p（含空端点分支），因此
        每条 ready 门端点仅在 qa/qb ∈ {lp, lq} 时变化。数值与旧实现
        逐元素一致（求和顺序相同）。
        """
        ready = self._ready_2q_gates()
        n_ready = max(len(ready), 1)
        dist = self.hw.dist
        mapping = self.mapping
        nq = max(self.num_qubits, 1)
        feats = np.zeros((self.num_edges, 5), dtype=np.float32)
        if not ready:
            return feats

        inv = {phys: log for log, phys in enumerate(mapping)}
        pairs = [(mapping[g.qubits[0]], mapping[g.qubits[1]]) for g in ready]

        # dist_before：与旧实现相同的顺序求和
        dist_before = 0.0
        for a, b in pairs:
            dist_before += dist[a, b]
        d0 = dist_before / nq

        for i, (p, q) in enumerate(self.coupling_map):
            lp, lq = inv.get(p), inv.get(q)
            if lp is None and lq is None:
                # 两侧均未被逻辑比特占用：SWAP 不改变任何映射
                feats[i, 0] = d0
                feats[i, 1] = d0
                continue
            dist_after = 0.0
            improved = 0
            worsened = 0
            for g, (a, b) in zip(ready, pairs):
                qa, qb = g.qubits
                na = q if qa == lp else (p if qa == lq else a)
                nb = q if qb == lp else (p if qb == lq else b)
                d_b = dist[a, b]
                d_a = dist[na, nb]
                dist_after += d_a
                if d_a < d_b - 1e-8:
                    improved += 1
                elif d_a > d_b + 1e-8:
                    worsened += 1
            feats[i, 0] = d0
            feats[i, 1] = dist_after / nq
            feats[i, 2] = (dist_before - dist_after) / max(dist_before, 1e-8)
            feats[i, 3] = improved / n_ready
            feats[i, 4] = worsened / n_ready

        return feats

    # ------------------------------------------------------------------
    #  死锁检测 (Phase 1)
    # ------------------------------------------------------------------
    def get_deadlock_mask(self, lookback: int = 2, max_cycle: int = 6, stall_window: int = 6):
        """返回 (num_edges,) bool 数组，True = 该边因死锁被禁止。

        检测三种 SWAP 震荡：
        1. 连续重复同一 SWAP（lookback 步内仅一条边）；
        2. 末尾出现周期 N>=2 的往返震荡（如 [6, 8, 6, 8, ...]），
           一旦最近 2N 步构成一致周期序列，则禁止该周期涉及的所有边；
        3. 无进展失速：最近 stall_window 次 SWAP 均未执行任何门
           （executed 无增长），则禁止该窗口内出现过的所有边，
           强制策略脱离死循环。
        """
        mask = np.zeros(self.num_edges, dtype=bool)
        h = self._swap_history
        n = len(h)
        if n >= lookback:
            recent = set(h[-lookback:])
            if len(recent) == 1:
                mask[list(recent)[0]] = True
        for N in range(2, max_cycle + 1):
            if n < 2 * N:
                break
            last = h[n - N:]
            prev = h[n - 2 * N:n - N]
            if last == prev:
                for e in set(last):
                    mask[e] = True
        if n - self._last_progress_swap >= stall_window:
            for e in set(h[-stall_window:]):
                mask[e] = True
        return mask

    def get_unmapped_mask(self):
        """返回 (num_edges,) bool 数组，True = 该边两端均无映射 qubit（无效换边）。"""
        occupied = {p for p in self.mapping}
        mask = np.zeros(self.num_edges, dtype=bool)
        for i, (p, q) in enumerate(self.coupling_map):
            if p not in occupied and q not in occupied:
                mask[i] = True
        return mask

    # ------------------------------------------------------------------
    #  Env 克隆 (用于 Beam Search 推理)
    # ------------------------------------------------------------------
    def clone(self):
        """轻量浅拷贝当前环境状态，供 beam search 模拟使用。"""
        new = object.__new__(RoutingEnv)
        # 不可变引用（所有 env 共享，不修改）
        new.dag = self.dag
        new.hw = self.hw
        new.coupling_map = self.coupling_map
        new.num_edges = self.num_edges
        new.num_qubits = self.num_qubits
        new.max_num_qubits = self.max_num_qubits
        new.noise_config = self.noise_config
        new.reward_mode = self.reward_mode
        new.gate_base_reward = self.gate_base_reward
        new.swap_cost = self.swap_cost
        new.eta_swap_err = self.eta_swap_err
        new.reward_potential = self.reward_potential
        new.shaping_gamma = self.shaping_gamma
        new.eta_shape = self.eta_shape
        new.alpha_ext = self.alpha_ext
        new.ext_set_size = self.ext_set_size
        new.invalid_penalty = self.invalid_penalty
        new.eta_err = self.eta_err
        new.eta_xtalk = self.eta_xtalk
        new.eta_xz_step = self.eta_xz_step
        new.eta_dist = self.eta_dist
        new.cnot_cost = self.cnot_cost
        new.use_scheduler = self.use_scheduler
        new.eta_time = self.eta_time
        new.eta_xtalk_par = self.eta_xtalk_par
        new.eta_idle = self.eta_idle
        new.eta_parallel = self.eta_parallel
        new.xtalk_alpha = self.xtalk_alpha
        new.swap_duration = self.swap_duration
        new.scheduler = self.scheduler
        new.timing = self.timing.clone() if self.timing is not None else None
        new._pending_measures = list(self._pending_measures)
        new._pending_swaps = list(self._pending_swaps)
        new._effective_initial_mapping = list(self._effective_initial_mapping)
        new.lambda_fid = self.lambda_fid
        new.lambda_layout = self.lambda_layout
        new.fidelity_fn = self.fidelity_fn
        new.sabre_fid_map = self.sabre_fid_map
        new.sref_override = self.sref_override
        new.random_init = self.random_init
        new.init_mapping = self.init_mapping
        new.max_episode_steps = self.max_episode_steps
        new.unfinished_penalty = self.unfinished_penalty
        new.max_num_edges = self.max_num_edges
        new.mapping_budget = self.mapping_budget
        new.commit_action = self.commit_action
        new.enable_mapping_phase = self.enable_mapping_phase
        new._rng = self._rng
        new._gnn = self._gnn
        new._edge_feat_dim = self._edge_feat_dim
        new._gnn_dim = self._gnn_dim
        new.action_space = self.action_space
        new.observation_space = self.observation_space
        # 可变状态（用 copy 隔离）
        new.mapping = self.mapping.copy()
        new.mapping_phase = self.mapping_phase
        new._mapping_swaps = self._mapping_swaps
        new.executed = self.executed.copy()
        new._swap_counter = self._swap_counter
        new._episode_step = self._episode_step
        new._swap_history = self._swap_history.copy()
        new._last_progress_swap = self._last_progress_swap
        new._xz_errors = self._xz_errors.copy()
        new._phys_circuit = self._phys_circuit.copy()
        new.executable_2q = self.executable_2q.copy()
        return new

    # ------------------------------------------------------------------
    #  步级奖励组件
    # ------------------------------------------------------------------
    def _step_reward_execute(self, success: bool, gate_idx: Optional[int]) -> float:
        if self.reward_mode == "fidelity_shaping":
            return 0.0
        if self.reward_potential:
            # R2 势函数：进度奖励 − 物理噪声代价（边感知，无任意完成奖励）
            if not success:
                return -self.invalid_penalty
            if gate_idx is None:
                return 0.0
            g = self.dag.gates[gate_idx]
            if g.is_two_qubit:
                pa, pb = self.mapping[g.qubits[0]], self.mapping[g.qubits[1]]
                cost = float(self.hw.two_q_err[pa, pb])
            else:
                cost = float(self.hw.single_q_err[self.mapping[g.qubits[0]]])
            return _POT_PROGRESS_B - cost
        if not success:
            return -self.invalid_penalty
        if gate_idx is None:
            return 0.0
        g = self.dag.gates[gate_idx]
        r = self.gate_base_reward.get(g.name, 0.3)
        if g.is_two_qubit:
            qa, qb = g.qubits
            pa, pb = self.mapping[qa], self.mapping[qb]
            e_g = float(self.hw.two_q_err[pa, pb])
        else:
            p = self.mapping[g.qubits[0]]
            e_g = float(self.hw.single_q_err[p])
        r -= self.eta_err * e_g
        r -= self.eta_xtalk * self._gate_crosstalk(gate_idx)
        return r

    def _step_reward_swap(self, p: Optional[int] = None,
                          q: Optional[int] = None) -> float:
        if self.reward_mode == "fidelity_shaping":
            return 0.0
        if self.reward_potential:
            # R2 势函数：物理 SWAP = 3×CX 边噪声；映射期虚拟 SWAP 免费
            # （布局质量由 lambda_layout 与后续门代价体现）
            if p is None or q is None:
                return 0.0
            return -3.0 * float(self.hw.two_q_err[p, q])
        r = -self.swap_cost
        # v2 口径：SWAP = 3×CX，按所在耦合边错误率 ×3 计价（边感知，
        # 惩罚"在高噪声边上换位"，与 eta_err 对 2q 门的计价同构）
        if p is not None and q is not None and self.eta_swap_err:
            r -= self.eta_swap_err * 3.0 * float(self.hw.two_q_err[p, q])
        return r

    def _update_xz_after_gate(self, gate_idx: int) -> float:
        g = self.dag.gates[gate_idx]
        old_total = float(np.sum(self._xz_errors))

        if g.is_two_qubit:
            qa, qb = g.qubits
            x_prop = self._xz_errors[qa, 0]
            z_prop = self._xz_errors[qb, 1]
            self._xz_errors[qb, 0] += x_prop
            self._xz_errors[qa, 1] += z_prop
            pa, pb = self.mapping[qa], self.mapping[qb]
            err = float(self.hw.two_q_err[pa, pb])
            self._xz_errors[qa, :] += err
            self._xz_errors[qb, :] += err
        else:
            p = self.mapping[g.qubits[0]]
            err = float(self.hw.single_q_err[p])
            self._xz_errors[g.qubits[0], :] += err

        new_total = float(np.sum(self._xz_errors))
        return new_total - old_total

    def _step_reward_propagate(self, gate_idx: Optional[int]) -> float:
        if gate_idx is None or self.reward_mode == "fidelity_shaping":
            return 0.0
        delta = self._update_xz_after_gate(gate_idx)
        return -self.eta_xz_step * delta

    def _auto_execute_batch(self) -> Tuple[float, float]:
        if self.use_scheduler and self.timing is not None:
            return self._auto_execute_batch_scheduled()
        self._update()
        r_exec = 0.0
        r_prop = 0.0
        while self.executable_2q:
            gate_idx = min(self.executable_2q)
            self.executed.add(gate_idx)
            self._last_progress_swap = len(self._swap_history)
            g = self.dag.gates[gate_idx]
            if not g.is_measure:
                pq = [self.mapping[q] for q in g.qubits]
                self._phys_circuit.append(g.operation, pq)
            r_exec += self._step_reward_execute(True, gate_idx)
            r_prop += self._step_reward_propagate(gate_idx)
            self._update()
        return r_exec, r_prop

    def _update_timing(self) -> Tuple[List[int], List[int]]:
        """返回当前可执行的一量子门 / 双量子门索引列表，不执行任何门。

        供调度器（scheduled 模式）使用：门在 _auto_execute_batch_scheduled 中
        经离散事件时序内核统一执行，以便累积 timing 奖励。
        """
        ready_1q: List[int] = []
        ready_2q: List[int] = []
        ready_meas: List[int] = []
        for g in self.dag.gates:
            if g.index in self.executed:
                continue
            if not all(p in self.executed for p in g.predecessors):
                continue
            if g.is_measure:
                ready_meas.append(g.index)
            elif g.is_two_qubit:
                qa, qb = g.qubits
                pa, pb = self.mapping[qa], self.mapping[qb]
                if self.hw.adj[pa, pb] > 0:
                    ready_2q.append(g.index)
            else:
                ready_1q.append(g.index)
        self.executable_2q = ready_2q
        self._pending_measures = ready_meas
        return ready_1q, ready_2q

    def _auto_execute_batch_scheduled(self) -> Tuple[float, float]:
        """调度版批执行：事件级 ASAP 调度，结算 timing 奖励。

        - 每轮把当前就绪门交给 schedule_events 做事件级调度（A1/A2/A3）
        - 并行密度奖励 r_parallel = eta_parallel·max(0, serial_dur/clock_advance-1)
          而 r_time = -eta_time·clock_advance（B1/B2：优化 clock = 优化时间）
        - 末尾把已就绪的 measure 集中在最终波执行（A5），不入 _phys_circuit
        - 空闲以「相邻门 gap」累计（A6），不重复计入 SWAP 占用
        """
        r_gate_exec = 0.0
        r_prop = 0.0
        r_time = 0.0
        r_xtalk = 0.0
        r_idle = 0.0
        r_parallel = 0.0
        total_clock = 0.0
        total_serial = 0.0
        total_xtalk = 0.0
        while True:
            ready_1q, ready_2q = self._update_timing()
            pending = list(self._pending_swaps)
            if not ready_1q and not ready_2q and not pending:
                break
            idle_before = float(self.timing.qubit_idle_time.sum())
            placed, clock_advance, xtalk = schedule_events(
                ready_1q, ready_2q, self.dag, self.mapping,
                self.timing, self.hw, self.scheduler, self.xtalk_alpha,
                pending_swaps=pending)
            self._pending_swaps = []
            idle_after = float(self.timing.qubit_idle_time.sum())
            if not placed:
                # 安全兜底：保证每轮至少执行一个门，避免死循环
                fb = ready_1q + ready_2q
                gate_idx = min(fb)
                g = self.dag.gates[gate_idx]
                pqs = [self.mapping[q] for q in g.qubits]
                dur = GATE_DURATION_TABLE.get(g.name, FALLBACK_DURATION)
                start = self.timing.total_time
                self.timing._phys_qubits_idle(None, pqs, start, dur)
                self.timing.total_time += dur
                self.timing.schedule_log.append({
                    "kind": "2q" if g.is_two_qubit else "1q",
                    "gate_idx": gate_idx, "op": g.name,
                    "qubits": pqs, "start": start, "end": self.timing.total_time,
                    "wave": self.timing.waves,
                })
                self.timing.waves += 1
                placed = [(gate_idx, start, self.timing.total_time,
                           "2q" if g.is_two_qubit else "1q")]
                xtalk = 0.0
                clock_advance = dur
            for (gate_idx, start, end, kind) in placed:
                if gate_idx >= 0:
                    g = self.dag.gates[gate_idx]
                    self.executed.add(gate_idx)
                    self._last_progress_swap = len(self._swap_history)
                    if not g.is_measure:
                        pq = [self.mapping[q] for q in g.qubits]
                        self._phys_circuit.append(g.operation, pq)
                    r_gate_exec += self._step_reward_execute(True, gate_idx)
                    r_prop += self._step_reward_propagate(gate_idx)
                    total_serial += (end - start)
                else:
                    # SWAP 调度事件（gate_idx==-1）：仅累积时长，不写物理线路/DAG
                    total_serial += (end - start)
            total_clock += clock_advance
            total_xtalk += xtalk
            r_time += -self.eta_time * clock_advance
            r_xtalk += -self.eta_xtalk_par * xtalk
            r_idle += -self.eta_idle * (idle_after - idle_before)
        # 末尾 measure 波（A5）：集中在最终时刻并行执行，不写物理线路
        if getattr(self, "_pending_measures", []):
            meas = self._pending_measures
            self._pending_measures = []
            idle_before = float(self.timing.qubit_idle_time.sum())
            start = self.timing.total_time
            m_dur = GATE_DURATION_TABLE.get("measure", 2.0)
            for gate_idx in meas:
                g = self.dag.gates[gate_idx]
                q = self.mapping[g.qubits[0]]
                self.timing._phys_qubits_idle(None, [q], start, m_dur)
                self.timing.schedule_log.append({
                    "kind": "measure", "gate_idx": gate_idx, "op": "measure",
                    "qubits": [q], "start": start, "end": start + m_dur,
                    "wave": self.timing.waves,
                })
                self.executed.add(gate_idx)
                self._last_progress_swap = len(self._swap_history)
                r_gate_exec += self._step_reward_execute(True, gate_idx)
                r_prop += self._step_reward_propagate(gate_idx)
                total_serial += m_dur
            self.timing.total_time += m_dur
            self.timing.waves += 1
            total_clock += m_dur
            idle_after = float(self.timing.qubit_idle_time.sum())
            r_idle += -self.eta_idle * (idle_after - idle_before)
        r_parallel = self.eta_parallel * max(0.0, (total_serial / total_clock - 1.0)) \
            if total_clock > 1e-9 else 0.0
        return (r_gate_exec + r_time + r_parallel + r_xtalk + r_idle, r_prop)

    # ------------------------------------------------------------------
    #  终端保真度 / 分布差异
    # ------------------------------------------------------------------
    def _get_terminal_reward_value(self) -> float:
        if self.fidelity_fn is not None:
            return self.fidelity_fn(self)
        if self.noise_config is not None:
            return self._compute_aer_fidelity()
        return 0.0

    def get_schedule_waves(self):
        """返回调度波形列表，供调度感知保真度模拟器使用。

        每个波形 = (dw, [(phys_idx, qubits, is_2q), ...])，其中 phys_idx
        对应 self._phys_circuit.data[phys_idx]（跳过末尾 measure 波）。
        dw = 该波内 max(end) - min(start)。无 timing（非调度模式）或空日志返回 None。
        """
        if self.timing is None:
            return None
        log = getattr(self.timing, "schedule_log", None)
        if not log:
            return None
        by_wave = {}
        phys_counter = 0
        for entry in log:
            if entry.get("kind") == "measure":
                continue
            w = entry["wave"]
            by_wave.setdefault(w, []).append((phys_counter, entry))
            phys_counter += 1
        if not by_wave:
            return None
        waves = []
        for w in sorted(by_wave):
            entries = by_wave[w]
            dw = max(e["end"] for _, e in entries) - min(e["start"] for _, e in entries)
            gates = []
            for phys_idx, e in entries:
                qs = tuple(e["qubits"])
                is_2q = (e["kind"] == "2q")
                gates.append((phys_idx, qs, is_2q))
            waves.append((dw, gates))
        return waves

    def _compute_aer_fidelity(self) -> float:
        from qiskit_aer import AerSimulator
        from sim.sim import NoiseSimulator

        shots = self.noise_config.shots

        # 截断到实际使用的量子比特子集，避免 density_matrix OOM
        try:
            from sim.trajectory_sim import _reduce_phys_circuit_for_fidelity
            red_circ, red_cfg = _reduce_phys_circuit_for_fidelity(
                self._phys_circuit, self.noise_config)
        except Exception:
            red_circ, red_cfg = self._phys_circuit, self.noise_config

        meas = red_circ.copy()
        meas.measure_all()

        noise_sim = NoiseSimulator(red_cfg)
        meas_t = noise_sim._transpile(meas)

        noisy_counts = noise_sim.run(meas_t, shots=shots, skip_transpile=True)

        ideal_sim = AerSimulator()
        ideal_job = ideal_sim.run(meas_t, shots=shots)
        ideal_counts = ideal_job.result().get_counts()

        all_outcomes = set(ideal_counts.keys()) | set(noisy_counts.keys())
        overlap = sum(min(ideal_counts.get(k, 0), noisy_counts.get(k, 0)) for k in all_outcomes)
        return overlap / shots

    def _terminal_reward(self, info: dict) -> float:
        info["num_swaps"] = self._swap_counter
        info["mapping_swaps"] = self._mapping_swaps
        info["terminal_reward"] = 0.0
        if self.reward_mode == "routing":
            info["terminal_XZ"] = float(np.sum(self._xz_errors))
            return 0.0
        if self.reward_mode in ("noise_aware", "fidelity_shaping"):
            # warmup：lambda_fid==0 时跳过昂贵的终端保真度模拟（奖励恒为 0）
            if self.lambda_fid == 0:
                info["fidelity"] = None
                return 0.0
            fid = self._get_terminal_reward_value()
            info["fidelity"] = fid
            # sref 优先级：per-circuit 覆盖 > sabre_fid_map 按 num_qubits 查
            sref = self.sref_override
            if sref is None and self.sabre_fid_map is not None:
                sref = self.sabre_fid_map.get(self.num_qubits)
            if sref is not None and sref > 0:
                fid_c = max(fid, sref * 1e-4)
                r = self.lambda_fid * (math.log(fid_c) - math.log(sref))
                r = float(np.clip(r, -50.0, 50.0))
                info["terminal_reward"] = r
                return r
            info["terminal_reward"] = self.lambda_fid * fid
            return self.lambda_fid * fid
        return 0.0

    def _end_step(self, reward: float, info: dict, compute_obs: bool = True,
                  phi_before: Optional[float] = None):
        """统一收尾：done / truncated 判定与终端奖励。"""
        info["mapping_swaps"] = self._mapping_swaps
        done = len(self.executed) == self.dag.num_gates
        truncated = False
        if done:
            reward += self._terminal_reward(info)
        elif self._episode_step >= self.max_episode_steps:
            truncated = True
            remaining = self.dag.num_gates - len(self.executed)
            reward += -self.unfinished_penalty * remaining
            info["truncated_remaining"] = remaining
        if phi_before is not None:
            # R3 势函数 shaping：r += γ·Φ(s') − Φ(s)；终态/截断 Φ(s')=0
            phi_after = 0.0 if (done or truncated) else self._phi()
            reward += self.shaping_gamma * phi_after - phi_before
        obs = self._obs() if compute_obs else None
        return obs, reward, done, truncated, info

    def _step_mapping(self, action: int, compute_obs: bool = True):
        """映射阶段：虚拟 SWAP 重排初始布局；commit 动作（>= num_edges）结束阶段。"""
        info: dict = {}
        reward = 0.0
        phi_before = self._phi() if self.shaping_gamma is not None else None
        legacy_dist = self.eta_dist != 0 and self.shaping_gamma is None
        if action >= self.num_edges:
            self.mapping_phase = False
            self._effective_initial_mapping = list(self.mapping)
            r_exec, r_prop = self._auto_execute_batch()
            reward += r_exec + r_prop
            if self.lambda_layout != 0.0:
                nready = max(1, len(self._ready_2q_gates()))
                reward += -self.lambda_layout * (self._front_layer_dist() / nready)
        else:
            p, q = self.coupling_map[action]
            dist_before = self._front_layer_dist() if legacy_dist else 0.0
            self._apply_virtual_swap(p, q)
            self._mapping_swaps += 1
            self._swap_history.append(action)
            if self.shaping_gamma is None:
                # 旧路径：虚拟 SWAP 也计 swap_cost/eta_swap_err
                reward += self._step_reward_swap(p, q)
            # R3：映射期虚拟 SWAP 物理免费（无噪声、零时长），
            # 布局引导由 Φ shaping（D_front/D_ext 项）承担
            if legacy_dist:
                dist_after = self._front_layer_dist()
                reward += -self.eta_dist * (dist_after - dist_before) / max(dist_before, 1e-8)
            if self._mapping_swaps >= self.mapping_budget:
                self.mapping_phase = False
                self._effective_initial_mapping = list(self.mapping)
                r_exec, r_prop = self._auto_execute_batch()
                reward += r_exec + r_prop
                if self.lambda_layout != 0.0:
                    nready = max(1, len(self._ready_2q_gates()))
                    reward += -self.lambda_layout * (self._front_layer_dist() / nready)
        return self._end_step(reward, info, compute_obs=compute_obs,
                              phi_before=phi_before)

    # ------------------------------------------------------------------
    #  step
    # ------------------------------------------------------------------
    def step(self, action: int, compute_obs: bool = True):
        self._episode_step += 1
        if self.mapping_phase:
            return self._step_mapping(action, compute_obs)

        if action >= self.num_edges:
            # 非映射阶段出现 commit 动作：视为无效（正常流程下会被掩码禁止）
            return self._end_step(-self.invalid_penalty, {}, compute_obs=compute_obs)

        p, q = self.coupling_map[action]

        legacy_dist = self.eta_dist != 0 and self.shaping_gamma is None
        phi_before = self._phi() if self.shaping_gamma is not None else None
        dist_before = self._front_layer_dist() if legacy_dist else 0.0

        self._apply_swap(p, q)
        self._swap_counter += 1
        self._swap_history.append(action)

        r_exec, r_prop = self._auto_execute_batch()

        reward = r_exec + r_prop
        if legacy_dist:
            dist_after = self._front_layer_dist()
            r_dist = -self.eta_dist * (dist_after - dist_before) / max(dist_before, 1e-8)
            reward += r_dist

        reward += self._step_reward_swap(p, q)

        return self._end_step(reward, {}, compute_obs=compute_obs,
                              phi_before=phi_before)
