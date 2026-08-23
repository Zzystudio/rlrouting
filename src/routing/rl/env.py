from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from qiskit import QuantumCircuit

from ..graph.circuit_dag import CircuitDAG, build_routing_graph
from ..gnn.encoder import SubGNN
from ..graph.features import HardwareFeatures

_GATE_BASE_REWARD_DEFAULT: Dict[str, float] = {
    "cx": 2.0, "h": 0.5, "sx": 0.3, "x": 0.3,
    "rz": 0.3, "y": 0.3, "z": 0.3, "s": 0.3, "t": 0.3,
    "swap": 0.0, "measure": 0.0, "barrier": 0.0,
}


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
        swap_cost: float = 0.3,
        invalid_penalty: float = 1.0,
        eta_err: float = 0.5,
        eta_xtalk: float = 0.02,
        eta_xz_step: float = 0.0,
        cnot_cost: float = 0.1,

        # --- 终端奖励参数 (Stage 2 & 3) ---
        lambda_kl: float = 1.0,
        lambda_ce: float = 1.0,
        lambda_tvd: float = 1.0,
        lambda_fid: float = 5.0,
        fidelity_fn: Optional[Callable] = None,

        # --- 距离塑形奖励 (Phase 1) ---
        eta_dist: float = 1.0,

        # --- Episode 截断 ---
        max_episode_steps: int = 200,
        unfinished_penalty: float = 0.5,

        max_num_edges: Optional[int] = None,
        max_num_qubits: Optional[int] = None,
        mapping_budget: Optional[int] = None,
        mapping_phase: bool = True,
        random_init: bool = True,
        use_gnn: bool = True,
        gnn: Optional[SubGNN] = None,
        seed: int = 0,
        noise_config=None,
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
        self.invalid_penalty = invalid_penalty
        self.eta_err = eta_err
        self.eta_xtalk = eta_xtalk
        self.eta_xz_step = eta_xz_step
        self.eta_dist = eta_dist
        self.cnot_cost = cnot_cost

        self.lambda_kl = lambda_kl
        self.lambda_ce = lambda_ce
        self.lambda_tvd = lambda_tvd
        self.lambda_fid = lambda_fid
        self.fidelity_fn = fidelity_fn

        self.random_init = random_init
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
        if self.random_init:
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

    def _sabre_edge_features(self):
        """为每条 coupling edge 计算 5 维 SABRE 启发式特征。"""
        n_ready = max(len(self._ready_2q_gates()), 1)
        dist_before = self._front_layer_dist()
        feats = np.zeros((self.num_edges, 5), dtype=np.float32)

        for i, (p, q) in enumerate(self.coupling_map):
            tmp_map = self.mapping.copy()
            inv = {phys: log for log, phys in enumerate(tmp_map)}
            lp, lq = inv.get(p), inv.get(q)
            if lp is not None and lq is not None:
                tmp_map[lp], tmp_map[lq] = tmp_map[lq], tmp_map[lp]
            elif lp is not None:
                tmp_map[lp] = q
            elif lq is not None:
                tmp_map[lq] = p

            dist_after = self._front_layer_dist(tmp_map)
            feats[i, 0] = dist_before / max(self.num_qubits, 1)
            feats[i, 1] = dist_after / max(self.num_qubits, 1)
            feats[i, 2] = (dist_before - dist_after) / max(dist_before, 1e-8)

            improved = 0
            worsened = 0
            for g in self._ready_2q_gates():
                qa, qb = g.qubits
                d_b = self.hw.dist[self.mapping[qa], self.mapping[qb]]
                d_a = self.hw.dist[tmp_map[qa], tmp_map[qb]]
                if d_a < d_b - 1e-8:
                    improved += 1
                elif d_a > d_b + 1e-8:
                    worsened += 1
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
        new.invalid_penalty = self.invalid_penalty
        new.eta_err = self.eta_err
        new.eta_xtalk = self.eta_xtalk
        new.eta_xz_step = self.eta_xz_step
        new.eta_dist = self.eta_dist
        new.cnot_cost = self.cnot_cost
        new.lambda_kl = self.lambda_kl
        new.lambda_ce = self.lambda_ce
        new.lambda_tvd = self.lambda_tvd
        new.lambda_fid = self.lambda_fid
        new.fidelity_fn = self.fidelity_fn
        new.random_init = self.random_init
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

    def _step_reward_swap(self) -> float:
        if self.reward_mode == "fidelity_shaping":
            return 0.0
        return -self.swap_cost

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

    # ------------------------------------------------------------------
    #  终端保真度 / 分布差异
    # ------------------------------------------------------------------
    def _get_terminal_reward_value(self) -> float:
        if self.fidelity_fn is not None:
            return self.fidelity_fn(self)
        if self.noise_config is not None:
            return self._compute_aer_fidelity()
        return 0.0

    def _compute_aer_fidelity(self) -> float:
        from qiskit_aer import AerSimulator
        from sim.sim import NoiseSimulator

        shots = self.noise_config.shots

        meas = self._phys_circuit.copy()
        meas.measure_all()

        noise_sim = NoiseSimulator(self.noise_config)
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
        if self.reward_mode == "routing":
            info["terminal_XZ"] = float(np.sum(self._xz_errors))
            return 0.0
        if self.reward_mode in ("noise_aware", "fidelity_shaping"):
            fid = self._get_terminal_reward_value()
            info["fidelity"] = fid
            return self.lambda_fid * fid
        return 0.0

    def _end_step(self, reward: float, info: dict, compute_obs: bool = True):
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
        obs = self._obs() if compute_obs else None
        return obs, reward, done, truncated, info

    def _step_mapping(self, action: int, compute_obs: bool = True):
        """映射阶段：虚拟 SWAP 重排初始布局；commit 动作（>= num_edges）结束阶段。"""
        info: dict = {}
        reward = 0.0
        if action >= self.num_edges:
            self.mapping_phase = False
            r_exec, r_prop = self._auto_execute_batch()
            reward += r_exec + r_prop
        else:
            p, q = self.coupling_map[action]
            dist_before = self._front_layer_dist() if self.eta_dist != 0 else 0.0
            self._apply_virtual_swap(p, q)
            self._mapping_swaps += 1
            self._swap_history.append(action)
            if self.eta_dist != 0:
                dist_after = self._front_layer_dist()
                reward += -self.eta_dist * (dist_after - dist_before) / max(dist_before, 1e-8)
            if self._mapping_swaps >= self.mapping_budget:
                self.mapping_phase = False
                r_exec, r_prop = self._auto_execute_batch()
                reward += r_exec + r_prop
        return self._end_step(reward, info, compute_obs=compute_obs)

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

        dist_before = self._front_layer_dist() if self.eta_dist != 0 else 0.0

        self._apply_swap(p, q)
        self._swap_counter += 1
        self._swap_history.append(action)

        r_exec, r_prop = self._auto_execute_batch()

        reward = r_exec + r_prop
        if self.eta_dist != 0:
            dist_after = self._front_layer_dist()
            r_dist = -self.eta_dist * (dist_after - dist_before) / max(dist_before, 1e-8)
            reward += r_dist

        return self._end_step(reward, {}, compute_obs=compute_obs)
