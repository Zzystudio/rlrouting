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

        # --- Episode 截断 ---
        max_episode_steps: int = 200,
        unfinished_penalty: float = 0.5,

        max_num_edges: Optional[int] = None,
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
        self.noise_config = noise_config

        self.reward_mode = reward_mode
        self.gate_base_reward = gate_base_reward or _GATE_BASE_REWARD_DEFAULT.copy()
        self.swap_cost = swap_cost
        self.invalid_penalty = invalid_penalty
        self.eta_err = eta_err
        self.eta_xtalk = eta_xtalk
        self.eta_xz_step = eta_xz_step
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
        self._rng = np.random.default_rng(seed)

        if gnn is not None:
            self._gnn = gnn
        elif use_gnn:
            self._gnn = SubGNN(subgraph="full")
            self._gnn.eval()
        else:
            self._gnn = None
        if self._gnn is not None:
            self._edge_feat_dim = self._gnn.encoder.out_dim * 3
            self._gnn_dim = self._edge_feat_dim * self.max_num_edges
        else:
            self._edge_feat_dim = 0
            self._gnn_dim = 0

        self.action_space = gym.spaces.Discrete(self.num_edges)
        obs_dim = self._gnn_dim + self.num_qubits + 1
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
        self.executed: set = set()
        self._swap_counter = 0
        self._episode_step = 0
        self._xz_errors = np.zeros((n, 2), dtype=float)
        self._phys_circuit = QuantumCircuit(self.hw.num_qubits)
        self._update()
        self._auto_execute_batch()
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
    def _obs(self):
        map_vec = np.array(
            [m / max(1, self.num_qubits) for m in self.mapping],
            dtype=np.float32,
        )
        progress = np.array(
            [len(self.executed) / max(1, self.dag.num_gates)], dtype=np.float32
        )
        if self._gnn is not None:
            executed_mask = np.zeros(self.dag.num_gates, dtype=bool)
            for idx in self.executed:
                executed_mask[idx] = True
            graph_data = build_routing_graph(
                self.dag, self.mapping, self.hw, self.coupling_map,
                executed_mask=executed_mask,
                executable_2q=set(self.executable_2q),
            )
            self._last_graph_data = graph_data
            self._last_map_vec = map_vec
            self._last_progress = progress
            with torch.no_grad():
                qubit_h = self._gnn.node_embeddings(graph_data).cpu().numpy()
            edge_feats_list = []
            for p, q in self.coupling_map:
                h_p = qubit_h[p]
                h_q = qubit_h[q]
                edge_feats_list.extend([h_p, h_q, h_p - h_q])
            edge_feats = np.concatenate(edge_feats_list).astype(np.float32)
            obs = np.concatenate([edge_feats, map_vec, progress]).astype(np.float32)
            # 多拓扑 padding：若当前拓扑边数少于 max，补零
            if self.max_num_edges > self.num_edges:
                pad_len = (self.max_num_edges - self.num_edges) * self._edge_feat_dim
                obs = np.pad(obs, (0, pad_len), constant_values=0)
            return obs
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
            return self.fidelity_fn(self.dag, self.mapping, self.executed)
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
        if self.reward_mode == "routing":
            info["terminal_XZ"] = float(np.sum(self._xz_errors))
            return 0.0
        if self.reward_mode in ("noise_aware", "fidelity_shaping"):
            fid = self._get_terminal_reward_value()
            info["fidelity"] = fid
            return self.lambda_fid * fid
        return 0.0

    # ------------------------------------------------------------------
    #  step
    # ------------------------------------------------------------------
    def step(self, action: int):
        p, q = self.coupling_map[action]
        self._apply_swap(p, q)
        self._swap_counter += 1
        self._episode_step += 1

        r_exec, r_prop = self._auto_execute_batch()

        reward = r_exec + r_prop
        done = len(self.executed) == self.dag.num_gates
        truncated = False
        info: dict = {}
        if done:
            reward += self._terminal_reward(info)
        elif self._episode_step >= self.max_episode_steps:
            truncated = True
            remaining = self.dag.num_gates - len(self.executed)
            reward += -self.unfinished_penalty * remaining
            info["truncated_remaining"] = remaining

        return self._obs(), reward, done, truncated, info
