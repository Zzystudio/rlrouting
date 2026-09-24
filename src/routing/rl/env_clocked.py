# ============================================================================
# env_clocked.py - 时钟化路由环境（doc/20260920训练方案.md）
# 事件时间 + launch/advance：全局时钟 T 连续推进，SKIP 推进到下一完成事件；
# 动作词表 [E SWAP | K EXEC | commit | skip]；锁 = busy_until；
# 1Q 惰性物化；奖励按 v3 通道分解（边际 ZZ / idle / 静态 ZZ）。
# 纯新增：RoutingEnv 子类，旧路径零改动。
# ============================================================================
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch

from ..graph.circuit_dag import build_routing_graph, TimingState
from ..timing import (GATE_DURATION_TABLE, FALLBACK_DURATION, MAX_DUR,
                      next_completion, marginal_xtalk, skip_idle_delta,
                      ao_exposure)
from .env import (RoutingEnv, _SABRE_FEAT_DIM, _LOOKAHEAD_FEAT_DIM,
                  _NOISE_FEAT_DIM, _GLOBAL_FEAT_DIM, _SABRE_CORE_FEAT_DIM)
from .agent_clocked import (D_EXEC, D_TIMING_GLOB, D_EDGE_TIMING,
                            D_GLOBAL_TIMING, D_EXEC_HAND, EXEC_H_DIM)


class ClockedRoutingEnv(RoutingEnv):
    """时钟化路由环境（事件时钟 + launch/advance 语义）。

    状态：self.clock（连续时间 T）、self.in_flight（在飞动作列表）、
    self.timing（CircuitTiming：busy_until/last_free/schedule_log）、
    self.mapping（编译计划布局，SWAP launch 时更新）。
    """

    def __init__(
        self,
        dag,
        hw,
        coupling_map,
        reward_mode: str = "routing",
        gate_base_reward: Optional[Dict[str, float]] = None,
        swap_cost: float = 0.0,
        invalid_penalty: float = 1.0,
        eta_err: float = 0.5,
        eta_xtalk: float = 0.02,
        eta_xz_step: float = 0.0,
        cnot_cost: float = 0.1,
        eta_swap_err: float = 0.0,
        reward_potential: bool = False,
        shaping_gamma: Optional[float] = None,
        eta_shape: float = 0.3,
        alpha_ext: float = 0.5,
        ext_set_size: int = 20,
        lookahead_features: bool = True,
        edge_noise_features: bool = False,
        beta_noise: float = 0.0,
        w_err: float = 0.0,
        w_xt: float = 0.0,
        w_xt_swap: float = 0.0,
        pot_progress_b: float = 0.045,
        pot_1q_reward: bool = True,
        sabre_swap_budget: Optional[int] = None,
        lambda_budget: float = 0.0,
        swap_price_scale: float = 1.0,
        step_cap_mult: float = 2.0,
        no_progress_limit: int = 0,
        lambda_fid: float = 5.0,
        fidelity_fn=None,
        eta_dist: float = 1.0,
        eta_time: float = 0.01,
        eta_xtalk_par: float = 1.0,
        eta_idle: float = 0.005,
        eta_parallel: float = 0.05,
        xtalk_alpha: float = 0.03,
        swap_duration_us: float = 0.9,
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
        gnn=None,
        seed: int = 0,
        noise_config=None,
        sabre_fid_map: Optional[Dict[int, float]] = None,
        sref_override: Optional[float] = None,
        use_scheduler: bool = True,   # 时钟化强制 True（占位兼容 create_env 透传）
        # ---- 时钟化专属参数 ----
        max_ready: int = 24,
        w_xt_launch: float = 0.0,     # 边际 ZZ 价（Step 0 审计标定）
        w_zz: float = 0.0,            # 静态 ZZ 价（Step 0 审计标定）
        eta_ao: float = 0.0,          # always-on ZZ（v2/v3 默认关，联动）
        step_cap_factor: float = 2.0, # 时钟化步数上限系数（相对门数）
        scheduling_only: bool = False,  # 方向2：SABRE 脚本路由，RL 只学调度
    ):
        super().__init__(
            dag, hw, coupling_map, reward_mode=reward_mode,
            gate_base_reward=gate_base_reward, swap_cost=swap_cost,
            invalid_penalty=invalid_penalty, eta_err=eta_err,
            eta_xtalk=eta_xtalk, eta_xz_step=eta_xz_step,
            cnot_cost=cnot_cost, eta_swap_err=eta_swap_err,
            reward_potential=reward_potential,
            shaping_gamma=shaping_gamma, eta_shape=eta_shape,
            alpha_ext=alpha_ext, ext_set_size=ext_set_size,
            lookahead_features=lookahead_features,
            edge_noise_features=edge_noise_features,
            beta_noise=beta_noise, w_err=w_err, w_xt=w_xt,
            w_xt_swap=w_xt_swap, pot_progress_b=pot_progress_b,
            pot_1q_reward=pot_1q_reward,
            sabre_swap_budget=sabre_swap_budget, lambda_budget=lambda_budget,
            swap_price_scale=swap_price_scale, step_cap_mult=step_cap_mult,
            no_progress_limit=no_progress_limit, lambda_fid=lambda_fid,
            fidelity_fn=fidelity_fn, eta_dist=eta_dist,
            use_scheduler=True, eta_time=eta_time,
            eta_xtalk_par=eta_xtalk_par, eta_idle=eta_idle,
            eta_parallel=eta_parallel, xtalk_alpha=xtalk_alpha,
            swap_duration_us=swap_duration_us,
            max_episode_steps=max_episode_steps,
            unfinished_penalty=unfinished_penalty,
            max_num_edges=max_num_edges, max_num_qubits=max_num_qubits,
            mapping_budget=mapping_budget, mapping_phase=mapping_phase,
            random_init=random_init, init_mapping=init_mapping,
            lambda_layout=lambda_layout, use_gnn=use_gnn, gnn=gnn,
            seed=seed, noise_config=noise_config,
            sabre_fid_map=sabre_fid_map, sref_override=sref_override,
        )
        self.max_ready = max_ready
        self.w_xt_launch = w_xt_launch
        self.w_zz = w_zz
        self.eta_ao = eta_ao
        self.step_cap_factor = step_cap_factor
        # 方向2（doc/20260922训练方案.md 方向1）：SABRE 已完成路由（DAG 为
        # 物理电路、SWAP 已物化为 2Q 门），恒等映射、SWAP 动作全屏蔽——
        # RL 只学 EXEC/SKIP 时序调度，swap 数与 SABRE 严格平价。
        self.scheduling_only = scheduling_only
        if scheduling_only:
            self.random_init = False
            self.mapping_phase = False
        # 动作词表：E SWAP | K EXEC | commit(E+K) | skip(E+K+1)
        self.action_space = gym.spaces.Discrete(self.num_edges + self.max_ready + 2)
        self.commit_action = self.num_edges + self.max_ready
        self.skip_action = self.num_edges + self.max_ready + 1
        self._edge_feat_dim = self._edge_feat_dim + D_EDGE_TIMING + D_GLOBAL_TIMING
        self._gnn_dim = self._edge_feat_dim * self.max_num_edges
        obs_dim = (self._gnn_dim
                   + self.max_ready * D_EXEC
                   + self.max_num_qubits + 2
                   + D_TIMING_GLOB)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,),
                                                dtype=np.float32)
        self._clocked_state_init()

    # ------------------------------------------------------------------
    #  时钟化状态
    # ------------------------------------------------------------------
    def _clocked_state_init(self):
        self.clock = 0.0
        self.in_flight: List[dict] = []
        self._ready_since: Dict[int, int] = {}
        self._candidate_slots: List[Optional[int]] = []
        self._cycle_launch_count = 0
        self._cum_theta = 0.0
        self._cum_idle_us = 0.0
        self._skip_count = 0
        self._terminal_done = False

    def reset(self, *, seed=None, options=None):
        if self.scheduling_only:
            # 方向2：物理电路的全部 2Q 门在 t0 即 ready∧adjacent，父类
            # reset 的 _auto_execute_batch 会把整条电路一次执行完——
            # 置位 enable_mapping_phase 阻断 auto-exec，随后恢复标志
            self.enable_mapping_phase = True
        super().reset(seed=seed)
        if self.scheduling_only:
            self.enable_mapping_phase = False
            self.mapping_phase = False
        # barrier 对路由是 no-op：直接视为已执行，避免阻塞 1Q/measure 链
        self.executed |= {g.index for g in self.dag.gates if g.name == "barrier"}
        self._clocked_state_init()
        return self._obs(), {}

    def _busy_until(self):
        return np.asarray(self.timing.qubit_busy_until, dtype=float)

    def _last_free(self):
        return np.asarray(self.timing.last_free, dtype=float)

    # ------------------------------------------------------------------
    #  frontier / 候选集（priority-K）
    # ------------------------------------------------------------------
    def _dag_ready(self, g) -> bool:
        """2Q 门 g 的 DAG 就绪判定（§1.2 三段检查之一）。

        惰性物化语义：g 的 1Q 前驱链可物化即可（未执行但可自动物化）；
        沿每个逻辑比特回溯，若遇到未执行 2Q / measure 则不可就绪，回溯到
        已执行门则通过。
        """
        for lq in g.qubits:
            prev = None
            for p in g.predecessors:
                if lq in self.dag.gates[p].qubits:
                    prev = p
                    break
            while prev is not None:
                gp = self.dag.gates[prev]
                if prev in self.executed:
                    break
                if gp.is_two_qubit or gp.is_measure:
                    return False
                nxt = None
                for p in gp.predecessors:
                    if lq in self.dag.gates[p].qubits:
                        nxt = p
                        break
                prev = nxt
        return True

    def _ready_2q_adjacent(self):
        """ready（可物化语义）且 相邻 的 2Q 门（候选集基础，不含锁检查）。"""
        out = []
        for g in self.dag.gates:
            if g.index in self.executed or not g.is_two_qubit:
                continue
            if not self._dag_ready(g):
                continue
            qa, qb = g.qubits
            pa, pb = self.mapping[qa], self.mapping[qb]
            if self.hw.adj[pa, pb] > 0:
                out.append(g)
        return out

    def _future_demand(self):
        """每逻辑比特剩余 2Q 门数（future_demand 特征）。"""
        rem = np.zeros(self.num_qubits, dtype=float)
        for g in self.dag.gates:
            if g.index in self.executed or not g.is_two_qubit:
                continue
            rem[g.qubits[0]] += 1.0
            rem[g.qubits[1]] += 1.0
        return rem

    def _ready_2q_gates(self):
        """覆盖父类：惰性物化语义下的 ready 2Q 门（§1.2 可物化尾链）。

        父类要求 preds 全部 executed，而 1Q 尾链在 EXEC launch 前不执行 →
        会恒空，导致 sabre/lookahead/sabre_core/global 特征与势函数全部退化。
        """
        return [g for g in self.dag.gates
                if g.index not in self.executed and g.is_two_qubit
                and self._dag_ready(g)]

    def _update_candidates(self):
        """计算候选集（ready∧adjacent，按 priority 排序取前 K）并更新 ready_age。"""
        rem_depth = self.dag.remaining_depths()
        md = max(1, self.dag.max_depth())
        fd = self._future_demand()
        ready = self._ready_2q_adjacent()
        cands = []
        for g in ready:
            gi = g.index
            if gi not in self._ready_since:
                self._ready_since[gi] = 0
            crit = rem_depth.get(gi, 0) / md
            qa, qb = g.qubits
            prio = crit + fd[qa] / 16.0 + fd[qb] / 16.0 \
                + self._ready_since[gi] / 16.0
            cands.append((prio, gi))
        cands.sort(key=lambda x: -x[0])
        slots = [gi for _, gi in cands[:self.max_ready]]
        slots += [None] * (self.max_ready - len(slots))
        self._candidate_slots = slots
        # executable_2q（graph status 特征 + TimingState.launchable）
        self.executable_2q = {gi for gi in slots if gi is not None}

    # ------------------------------------------------------------------
    #  动作 mask（锁 / frontier / skip 合法性 + liveness）
    # ------------------------------------------------------------------
    def _endpoints_free(self, p: int, q: int, t: Optional[float] = None) -> bool:
        if t is None:
            t = self.clock
        bu = self._busy_until()
        return bool(bu[p] <= t + 1e-9 and bu[q] <= t + 1e-9)

    def _is_in_flight_swap(self, p: int, q: int) -> bool:
        return False  # 兼容占位（未使用）

    def get_action_mask(self) -> np.ndarray:
        E = self.num_edges
        K = self.max_ready
        mask = np.zeros(E + K + 2, dtype=bool)
        bu = self._busy_until()
        t = self.clock
        if self.mapping_phase:
            occupied = {int(p) for p in self.mapping}
            for i, (p, q) in enumerate(self.coupling_map):
                mask[i] = (p in occupied and q in occupied)
            mask[E + K] = True                      # commit
            return mask
        # --- routing phase ---
        occupied = {int(p) for p in self.mapping}
        deadlock = self.get_deadlock_mask()
        if not self.scheduling_only:
            for i, (p, q) in enumerate(self.coupling_map):
                if (p in occupied and q in occupied
                        and bu[p] <= t + 1e-9 and bu[q] <= t + 1e-9
                        and not deadlock[i]):
                    mask[i] = True
        for slot, gi in enumerate(self._candidate_slots):
            if gi is None:
                continue
            g = self.dag.gates[gi]
            pa, pb = self.mapping[g.qubits[0]], self.mapping[g.qubits[1]]
            if self._endpoints_free(pa, pb, t):
                mask[E + slot] = True
        mask[E + K + 1] = bool(np.any(bu > t + 1e-9))    # skip
        # liveness：零合法动作时依序放松死锁 mask -> unmapped
        if not mask.any():
            if self.scheduling_only:
                # 调度-only 模式不允许退回 SWAP：无 EXEC 且无在飞 = 真死锁
                raise RuntimeError(
                    "ClockedRoutingEnv(scheduling_only): 无合法动作（liveness 破坏）")
            for i, (p, q) in enumerate(self.coupling_map):
                if p in occupied and q in occupied and bu[p] <= t + 1e-9 \
                        and bu[q] <= t + 1e-9:
                    mask[i] = True
        if not mask.any():
            raise RuntimeError("ClockedRoutingEnv: 无合法动作（liveness 破坏）")
        return mask

    # ------------------------------------------------------------------
    #  1Q 尾链（惰性物化辅助）
    # ------------------------------------------------------------------
    def _chain_1q(self, g, lq: int) -> List[int]:
        """g 在逻辑比特 lq 上的未物化 1Q 尾链（从最下游到最上游）。"""
        chain = []
        prev = None
        for p in g.predecessors:
            if lq in self.dag.gates[p].qubits:
                prev = p
                break
        while prev is not None:
            gp = self.dag.gates[prev]
            if gp.is_two_qubit or gp.is_measure or prev in self.executed:
                break
            chain.append(prev)
            nxt = None
            for p in gp.predecessors:
                if lq in self.dag.gates[p].qubits:
                    nxt = p
                    break
            prev = nxt
        return chain

    def _chain_dur(self, g, lq: int) -> float:
        return float(sum(GATE_DURATION_TABLE.get(self.dag.gates[i].name,
                                                 FALLBACK_DURATION)
                         for i in self._chain_1q(g, lq)))

    # ------------------------------------------------------------------
    #  时序特征
    # ------------------------------------------------------------------
    def _timing_state(self) -> TimingState:
        P = self.hw.num_qubits
        bu = self._busy_until()
        run_kind = np.zeros(P, dtype=float)
        for f in self.in_flight:
            k = 1.0 if f["kind"] == "swap" else -1.0
            for q in f["qubits"]:
                run_kind[q] = k
        gate_in_flight = {f["gate_idx"] for f in self.in_flight
                          if f.get("gate_idx", -1) >= 0}
        launchable = {gi for slot, gi in enumerate(self._candidate_slots)
                      if gi is not None and self.get_exec_slot_free(slot)}
        chain_dur = {}
        for gi in self._candidate_slots:
            if gi is None:
                continue
            g = self.dag.gates[gi]
            chain_dur[gi] = max(self._chain_dur(g, g.qubits[0]),
                                self._chain_dur(g, g.qubits[1]))
        return TimingState(
            t=self.clock,
            busy_until=bu,
            last_free=self._last_free(),
            qubit_idle_time=np.asarray(self.timing.qubit_idle_time, float),
            qubit_crosstalk=np.asarray(self.timing.qubit_crosstalk, float),
            run_kind=run_kind,
            parallel_usage=np.asarray(self.timing.parallel_usage, float),
            gate_in_flight=gate_in_flight,
            launchable_2q=launchable,
            chain_dur=chain_dur,
            ready_age=dict(self._ready_since),
        )

    def get_exec_slot_free(self, slot: int) -> bool:
        gi = self._candidate_slots[slot]
        if gi is None:
            return False
        g = self.dag.gates[gi]
        return self._endpoints_free(self.mapping[g.qubits[0]],
                                    self.mapping[g.qubits[1]])

    def mimic_swap_index(self, tie_eps: float = 0.0,
                         rng: Optional[np.random.Generator] = None) -> int:
        """特征驱动 SABRE 模仿路由头：选 sabre_core[0]（SABRE 完整打分，低=好）
        最小的合法边（映射期=虚拟 swap，路由期=物理 swap）。

        依赖特征保真度修复（decay 0.001 + ext 集前驱就绪语义），0b 诊断：
        tianyan20q_test argmin≈1.098×SABRE。确定性、零训练、无漂移——
        时钟化管线的冻结路由头（doc/train.md 2026-09-21）。
        2026-09-22：排除死锁 mask 边（mimic 贪心 argmin 在 mixed_serial_parallel
        类电路因分数平局反复选同一边，900+ swap 病理震荡）。
        tie_eps>0：在 min+ε 分数窗口内随机 tie-break（近似 SABRE 的随机
        tie-break，配合 best-of-N trial 可再逼近 ~7%，compare_sabre_trials）。
        """
        sc = self._edge_sabre_core_features()
        unmapped = self.get_unmapped_mask()
        deadlock = self.get_deadlock_mask()
        score = np.where(~unmapped & ~deadlock, sc[:, 0], np.inf)
        if not np.isfinite(score).any():
            # 全部被死锁/未映射挡住：放松死锁（仅 unmapped 排除）
            score = np.where(~unmapped, sc[:, 0], np.inf)
        m = float(np.min(score))
        if tie_eps > 0 and np.isfinite(m):
            cand = np.flatnonzero(score <= m + tie_eps)
            if cand.size > 1:
                rng = rng or np.random.default_rng(0)
                return int(cand[rng.integers(len(cand))])
        return int(np.argmin(score))

    def _timing_globals(self) -> np.ndarray:
        """state-level 时序向量（9 维）：skip/critic 输入 + 广播特征来源。"""
        bu = self._busy_until()
        P = max(1, self.hw.num_qubits)
        n_run = float(np.sum(bu > self.clock + 1e-9))
        busy_frac = n_run / P
        serial = float(getattr(self.timing, "serial_dur", 0.0))
        par_density = serial / max(self.clock, 1e-9) if self.clock > 1e-9 else 0.0
        budget = getattr(self, "sabre_swap_budget", None) or 0
        bl = 1.0 - self._swap_counter / max(1, budget) if budget else 0.0
        n_launchable = sum(1 for slot in range(self.max_ready)
                           if self.get_exec_slot_free(slot))
        return np.array([
            self.clock / 100.0,
            n_run / P,
            busy_frac,
            par_density,
            min(self._cum_theta / 0.2, 1.0),
            min(self._cum_idle_us / 200.0, 1.0),
            self._cycle_launch_count / 8.0,
            n_launchable / 8.0,
            bl,
        ], dtype=np.float32)

    def _global_broadcast_timing(self) -> np.ndarray:
        """广播时序特征（8 维，追加到 global101 -> global109）。"""
        g = self._timing_globals()
        return g[:8]

    def _edge_timing_feats(self) -> np.ndarray:
        """每耦合边 4 维时序：both_free/marginal_theta_swap/busy_min/parallel_usage。"""
        E = self.num_edges
        feats = np.zeros((E, D_EDGE_TIMING), dtype=np.float32)
        bu = self._busy_until()
        t = self.clock
        in_fl = [(tuple(f["qubits"]), f["start"], f["end"])
                 for f in self.in_flight]
        pu = np.asarray(self.timing.parallel_usage, float)
        G = max(1, self.dag.num_gates)
        for i, (p, q) in enumerate(self.coupling_map):
            feats[i, 0] = 1.0 if (bu[p] <= t + 1e-9 and bu[q] <= t + 1e-9) else 0.0
            feats[i, 1] = marginal_xtalk(self.hw, (p, q), t, t + self.swap_duration,
                                         in_fl)
            feats[i, 2] = max(0.0, min(bu[p], bu[q]) - t) / max(MAX_DUR, 1e-9)
            feats[i, 3] = min(float(pu[p, q]) / G, 1.0)
        return feats

    # ------------------------------------------------------------------
    #  观测
    # ------------------------------------------------------------------
    def _exec_features(self, gi: int) -> np.ndarray:
        """EXEC 候选手工特征（exec12）。"""
        g = self.dag.gates[gi]
        qa, qb = g.qubits
        pa, pb = self.mapping[qa], self.mapping[qb]
        bu = self._busy_until()
        t = self.clock
        in_fl = [(tuple(f["qubits"]), f["start"], f["end"])
                 for f in self.in_flight]
        ca = self._chain_dur(g, qa)
        cb = self._chain_dur(g, qb)
        chain = max(ca, cb)
        g_start = t + chain
        g_dur = GATE_DURATION_TABLE.get(g.name, FALLBACK_DURATION)
        marg = 0.0
        evs = []
        # 该 launch 物化的全部事件（链 + 2Q）做边际归因（§5.2 可加性）
        for lq, c in ((qa, ca), (qb, cb)):
            q = self.mapping[lq]
            tt = t
            for i in self._chain_1q(g, lq):
                d = GATE_DURATION_TABLE.get(self.dag.gates[i].name,
                                            FALLBACK_DURATION)
                evs.append(((q,), tt, tt + d))
                tt += d
        evs.append(((pa, pb), g_start, g_start + g_dur))
        prev = list(in_fl)
        for qs2, s2, e2 in evs:
            marg += marginal_xtalk(self.hw, qs2, s2, e2, prev)
            prev.append((qs2, s2, e2))
        rem_depth = self.dag.remaining_depths()
        md = max(1, self.dag.max_depth())
        succ = self.dag.successors().get(gi, [])
        fd = self._future_demand()
        neighbors = set()
        for nb in range(self.hw.num_qubits):
            if self.hw.adj[pa, nb] > 0:
                neighbors.add(nb)
            if self.hw.adj[pb, nb] > 0:
                neighbors.add(nb)
        n_busy_nb = sum(1 for nb in neighbors if bu[nb] > t + 1e-9)
        par_win = (float(np.mean([f["end"] for f in self.in_flight])) - t
                   if self.in_flight else 0.0)
        budget = getattr(self, "sabre_swap_budget", None) or 0
        bl = 1.0 - self._swap_counter / max(1, budget) if budget else 0.0
        feats = np.zeros(D_EXEC_HAND, dtype=np.float32)
        feats[0] = float(self.hw.two_q_err[pa, pb])
        feats[1] = marg
        feats[2] = chain / max(MAX_DUR, 1e-9)
        feats[3] = rem_depth.get(gi, 0) / md
        feats[4] = len(succ) / max(1, len(succ))
        feats[5] = (fd[qa] + fd[qb]) / 16.0
        feats[6] = self._ready_since.get(gi, 0) / 16.0
        feats[7] = float(self.hw.zz[pa, pb])
        feats[8] = n_busy_nb / 4.0
        feats[9] = max(0.0, par_win) / max(MAX_DUR, 1e-9)
        feats[10] = bl
        feats[11] = 1.0 if (bu[pa] <= t + 1e-9 and bu[pb] <= t + 1e-9) else 0.0
        return feats

    def _obs(self, qubit_h=None):
        t = self.clock
        self._update_candidates()
        ts = self._timing_state()
        graph_data = build_routing_graph(
            self.dag, self.mapping, self.hw, self.coupling_map,
            executed_mask=np.array([1.0 if i in self.executed else 0.0
                                    for i in range(self.dag.num_gates)],
                                   dtype=bool),
            executable_2q=set(self.executable_2q),
            timing_state=ts,
        )
        if self._gnn is not None:
            with torch.no_grad():
                gate_h, qubit_h = self._gnn.node_and_gate_embeddings(graph_data)
            gate_h = gate_h.cpu().numpy()
            qubit_h = qubit_h.cpu().numpy()
        else:
            gate_h = np.zeros((self.dag.num_gates, EXEC_H_DIM), dtype=np.float32)
            qubit_h = np.zeros((self.hw.num_qubits, EXEC_H_DIM), dtype=np.float32)
        sabre_feats = self._sabre_edge_features()
        look_feats = (self._edge_lookahead_features()
                      if self.lookahead_features
                      else np.zeros((self.num_edges, _LOOKAHEAD_FEAT_DIM),
                                    dtype=np.float32))
        noise_feats = (self._edge_noise_features()
                       if self.edge_noise_features
                       else np.zeros((self.num_edges, _NOISE_FEAT_DIM),
                                     dtype=np.float32))
        if self._gnn is not None:
            global101 = self._global_context_features(qubit_h)
        else:
            global101 = np.zeros(_GLOBAL_FEAT_DIM, dtype=np.float32)
        sabre_core = self._edge_sabre_core_features()
        edge_t = self._edge_timing_feats()
        # --- edge 块（新时序特征全部追加在末尾，保留前 267 列布局不变：
        #     _zero_pad_ac_state 尾部补零 → warm-start 逐位不变）---
        edge_list = []
        for i, (p, q) in enumerate(self.coupling_map):
            edge_list.extend([qubit_h[p], qubit_h[q], qubit_h[p] - qubit_h[q],
                              sabre_feats[i]])
            if self.lookahead_features:
                edge_list.append(look_feats[i])
            if self.edge_noise_features:
                edge_list.append(noise_feats[i])
            edge_list.append(global101)
            edge_list.append(sabre_core[i])
            edge_list.append(self._global_broadcast_timing())
            edge_list.append(edge_t[i])
        edge_feats = np.concatenate(edge_list).astype(np.float32)
        if self.max_num_edges > self.num_edges:
            pad_len = (self.max_num_edges - self.num_edges) * self._edge_feat_dim
            edge_feats = np.pad(edge_feats, (0, pad_len), constant_values=0)
        # --- exec 块 ---
        exec_list = []
        for slot in range(self.max_ready):
            gi = self._candidate_slots[slot]
            if gi is None:
                exec_list.append(np.zeros(D_EXEC, dtype=np.float32))
                continue
            g = self.dag.gates[gi]
            qa, qb = g.qubits
            pa, pb = self.mapping[qa], self.mapping[qb]
            exec_list.append(np.concatenate([
                gate_h[gi], qubit_h[pa], qubit_h[pb],
                qubit_h[pa] - qubit_h[pb], self._exec_features(gi),
            ]).astype(np.float32))
        exec_feats = np.concatenate(exec_list).astype(np.float32)
        # --- 尾部 ---
        map_vec = np.array([m / max(1, self.num_qubits) for m in self.mapping],
                           dtype=np.float32)
        if self.max_num_qubits > self.num_qubits:
            pad = np.zeros(self.max_num_qubits - self.num_qubits,
                           dtype=np.float32)
            map_vec = np.concatenate([map_vec, pad])
        progress = np.array([len(self.executed) / max(1, self.dag.num_gates)],
                            dtype=np.float32)
        phase = np.array([1.0 if self.mapping_phase else 0.0],
                         dtype=np.float32)
        tg = self._timing_globals()
        return np.concatenate([edge_feats, exec_feats, map_vec, progress,
                               phase, tg]).astype(np.float32)

    # ------------------------------------------------------------------
    #  step
    # ------------------------------------------------------------------
    def step(self, action: int, compute_obs: bool = True):
        self._episode_step += 1
        if self.mapping_phase:
            return self._step_mapping_clocked(action, compute_obs)
        if action < self.num_edges:
            return self._launch_swap(action, compute_obs)
        if action < self.num_edges + self.max_ready:
            return self._launch_exec(action - self.num_edges, compute_obs)
        if action == self.skip_action:
            return self._step_skip(compute_obs)
        return self._end_step(-self.invalid_penalty, {}, compute_obs=compute_obs)

    def _update_no_progress(self):
        n_exec = len(self.executed)
        if n_exec > self._last_exec_count:
            self._steps_since_progress = 0
            self._last_exec_count = n_exec
        else:
            self._steps_since_progress += 1

    def _step_mapping_clocked(self, action: int, compute_obs: bool = True):
        info: dict = {}
        reward = 0.0
        phi_before = self._phi() if self.shaping_gamma is not None else None
        if action == self.commit_action:
            if self.mapping_phase:
                self.mapping_phase = False
                self._effective_initial_mapping = list(self.mapping)
                self._clocked_state_init()
                if self.lambda_layout != 0.0:
                    nready = max(1, len(self._ready_2q_gates()))
                    reward += -self.lambda_layout * (self._front_layer_dist() / nready)
        elif action < self.num_edges:
            p, q = self.coupling_map[action]
            self._apply_virtual_swap(p, q)
            self._mapping_swaps += 1
            self._swap_history.append(action)
            if self._mapping_swaps >= self.mapping_budget:
                self.mapping_phase = False
                self._effective_initial_mapping = list(self.mapping)
                self._clocked_state_init()
        else:
            return self._end_step(-self.invalid_penalty, info,
                                  compute_obs=compute_obs, phi_before=phi_before)
        return self._end_step(reward, info, compute_obs=compute_obs,
                              phi_before=phi_before)

    def _launch_swap(self, e: int, compute_obs: bool = True):
        from ..graph.features import _ERROR_SCALE
        p, q = self.coupling_map[e]
        t = self.clock
        dur = self.swap_duration
        end = t + dur
        info: dict = {}
        phi_before = self._phi() if self.shaping_gamma is not None else None
        in_fl = [(tuple(f["qubits"]), f["start"], f["end"])
                 for f in self.in_flight]
        marg = marginal_xtalk(self.hw, (p, q), t, end, in_fl)
        self._apply_swap(p, q)                 # mapping + phys_circuit.swap + decay++
        self._pending_swaps = []               # 时钟化自管 in_flight
        self._swap_counter += 1
        self._swap_history.append(e)
        self.in_flight.append(dict(kind="swap", gate_idx=-1, qubits=(p, q),
                                   start=t, end=end))
        self.timing._phys_qubits_idle(None, [p, q], t, dur)
        self.timing.schedule_log.append({"kind": "swap", "gate_idx": -1,
                                         "op": "swap", "qubits": [p, q],
                                         "start": t, "end": end,
                                         "wave": self.timing.waves})
        self.timing.total_time = max(self.timing.total_time, end)
        self._cycle_launch_count += 1
        self._cum_theta += marg
        e_edge = float(self.hw.two_q_err[p, q])
        theta_pq = float(self.hw.zz[p, q]) * _ERROR_SCALE
        r = -self.swap_price_scale * 3.0 * (e_edge + self.w_zz * theta_pq)
        r -= self.w_xt_launch * marg
        if (self.sabre_swap_budget is not None
                and self._swap_counter > self.sabre_swap_budget):
            r -= self.lambda_budget
        self._update_no_progress()
        return self._end_step(r, info, compute_obs=compute_obs,
                              phi_before=phi_before)

    def _launch_exec(self, slot: int, compute_obs: bool = True):
        from ..graph.features import _ERROR_SCALE
        gi = self._candidate_slots[slot]
        if gi is None:
            return self._end_step(-self.invalid_penalty, {},
                                  compute_obs=compute_obs)
        g = self.dag.gates[gi]
        qa, qb = g.qubits
        pa, pb = self.mapping[qa], self.mapping[qb]
        t = self.clock
        info: dict = {}
        phi_before = self._phi() if self.shaping_gamma is not None else None
        in_fl_before = [(tuple(f["qubits"]), f["start"], f["end"])
                        for f in self.in_flight]

        def sched_chain(chain, lq, s0):
            q = self.mapping[lq]
            s = s0
            evs = []
            for i in reversed(chain):          # 最上游先
                gg = self.dag.gates[i]
                d = GATE_DURATION_TABLE.get(gg.name, FALLBACK_DURATION)
                evs.append((i, (q,), s, s + d))
                s += d
            return evs, s - s0

        ea, ca = sched_chain(self._chain_1q(g, qa), qa, t)
        eb, cb = sched_chain(self._chain_1q(g, qb), qb, t)
        g_start = t + max(ca, cb)
        g_dur = GATE_DURATION_TABLE.get(g.name, FALLBACK_DURATION)
        g_end = g_start + g_dur
        events = ea + eb + [(gi, (pa, pb), g_start, g_end)]
        prev = list(in_fl_before)
        marg = 0.0
        for (_, qs2, s2, e2) in events:
            marg += marginal_xtalk(self.hw, qs2, s2, e2, prev)
            prev.append((qs2, s2, e2))
        for (idx, qs2, s2, e2) in events:
            gg = self.dag.gates[idx]
            dur = e2 - s2
            self.timing._phys_qubits_idle(None, list(qs2), s2, dur)
            kind = "1q" if not gg.is_two_qubit else "2q"
            self.timing.schedule_log.append({
                "kind": kind, "gate_idx": idx, "op": gg.name,
                "qubits": list(qs2), "start": s2, "end": e2,
                "wave": self.timing.waves})
            if not gg.is_measure:
                self._phys_circuit.append(gg.operation, list(qs2))
            self.executed.add(idx)
            self._last_progress_swap = len(self._swap_history)
            self._reset_qubit_decay(list(qs2))
            self.in_flight.append(dict(kind=kind, gate_idx=idx,
                                       qubits=tuple(qs2), start=s2, end=e2))
        self.timing.total_time = max(self.timing.total_time, g_end)
        self.timing.waves += 1
        self._cycle_launch_count += 1
        self._cum_theta += marg
        e_edge = float(self.hw.two_q_err[pa, pb])
        theta_pq = float(self.hw.zz[pa, pb]) * _ERROR_SCALE
        r = self.pot_progress_b - e_edge - self.w_zz * theta_pq \
            - self.w_xt_launch * marg
        r += self._step_reward_propagate(gi)
        self._maybe_materialize_terminal()
        self._update_no_progress()
        return self._end_step(r, info, compute_obs=compute_obs,
                              phi_before=phi_before)

    def _maybe_materialize_terminal(self):
        if getattr(self, "_terminal_done", False):
            return
        n_2q = len(self.dag.two_qubit_gates())
        executed_2q = sum(1 for g in self.dag.gates
                          if g.is_two_qubit and not g.is_measure
                          and g.index in self.executed)
        if executed_2q < n_2q:
            return
        bu = self._busy_until()
        s0 = float(bu.max()) if bu.size else 0.0
        for g in self.dag.gates:
            if g.index in self.executed or g.is_measure:
                continue
            q = self.mapping[g.qubits[0]]
            d = GATE_DURATION_TABLE.get(g.name, FALLBACK_DURATION)
            self.timing._phys_qubits_idle(None, [q], s0, d)
            self.timing.schedule_log.append({"kind": "1q", "gate_idx": g.index,
                                             "op": g.name, "qubits": [q],
                                             "start": s0, "end": s0 + d,
                                             "wave": self.timing.waves})
            self._phys_circuit.append(g.operation, [q])
            self.executed.add(g.index)
            self._reset_qubit_decay([q])
            s0 += d
        s0 = float(self._busy_until().max())
        m_dur = GATE_DURATION_TABLE.get("measure", 2.0)
        for g in self.dag.gates:
            if not g.is_measure or g.index in self.executed:
                continue
            q = self.mapping[g.qubits[0]]
            self.timing.schedule_log.append({"kind": "measure",
                                             "gate_idx": g.index,
                                             "op": "measure", "qubits": [q],
                                             "start": s0, "end": s0 + m_dur,
                                             "wave": self.timing.waves})
            self.executed.add(g.index)
        self.timing.waves += 1
        self._terminal_done = True

    def _step_skip(self, compute_obs: bool = True):
        bu = self._busy_until()
        t = self.clock
        Tp = next_completion(bu, t)
        if not np.isfinite(Tp):
            return self._end_step(-self.invalid_penalty, {},
                                  compute_obs=compute_obs)
        info: dict = {}
        phi_before = self._phi() if self.shaping_gamma is not None else None
        idle_d = skip_idle_delta(bu, self._last_free(), t, Tp)
        ao = (ao_exposure(self.hw.adj, self.hw.zz, bu, t, Tp)
              if self.eta_ao else 0.0)
        dt = Tp - t
        completing = [f for f in self.in_flight if abs(f["end"] - Tp) < 1e-9]
        serial = sum(f["end"] - f["start"] for f in completing)
        self.in_flight = [f for f in self.in_flight
                          if abs(f["end"] - Tp) >= 1e-9]
        density = serial / max(dt, 1e-9) if dt > 1e-9 else 0.0
        r = -self.eta_time * dt \
            + self.eta_parallel * max(0.0, density - 1.0) \
            - self.eta_idle * idle_d - self.eta_ao * ao
        self.clock = Tp
        self.timing.total_time = max(self.timing.total_time, Tp)
        self._cum_idle_us += idle_d
        self._cycle_launch_count = 0
        self._skip_count += 1
        for gi in list(self._ready_since.keys()):
            self._ready_since[gi] += 1
        self._update_no_progress()
        return self._end_step(r, info, compute_obs=compute_obs,
                              phi_before=phi_before)

    # ------------------------------------------------------------------
    #  _end_step（时钟化步数上限）
    # ------------------------------------------------------------------
    def _end_step(self, reward: float, info: dict, compute_obs: bool = True,
                  phi_before: Optional[float] = None):
        import math
        info["mapping_swaps"] = self._mapping_swaps
        done = len(self.executed) == self.dag.num_gates
        truncated = False
        step_cap = max(self.max_episode_steps,
                       math.ceil(self.step_cap_mult * self.dag.num_gates
                                 * self.step_cap_factor))
        no_progress_hit = (self.no_progress_limit > 0
                           and self._steps_since_progress >= self.no_progress_limit)
        if done:
            reward += self._terminal_reward(info)
        elif self._episode_step >= step_cap:
            truncated = True
            remaining = self.dag.num_gates - len(self.executed)
            reward += -self.unfinished_penalty * remaining
            info["truncated_remaining"] = remaining
        elif no_progress_hit:
            truncated = True
            remaining = self.dag.num_gates - len(self.executed)
            reward += -self.unfinished_penalty * remaining
            info["truncated_remaining"] = remaining
            info["truncated_no_progress"] = True
        if phi_before is not None:
            phi_after = 0.0 if (done or truncated) else self._phi()
            reward += self.shaping_gamma * phi_after - phi_before
        obs = self._obs() if compute_obs else None
        return obs, reward, done, truncated, info

    # ------------------------------------------------------------------
    #  clone（beam search）
    # ------------------------------------------------------------------
    def clone(self):
        new = object.__new__(ClockedRoutingEnv)
        for attr in ("dag", "hw", "coupling_map", "num_edges", "num_qubits",
                     "max_num_qubits", "noise_config", "reward_mode",
                     "gate_base_reward", "swap_cost", "eta_swap_err",
                     "reward_potential", "shaping_gamma", "eta_shape",
                     "alpha_ext", "ext_set_size", "lookahead_features",
                     "edge_noise_features", "beta_noise", "w_err", "w_xt",
                     "w_xt_swap", "pot_progress_b", "pot_1q_reward",
                     "sabre_swap_budget", "lambda_budget", "swap_price_scale",
                     "step_cap_mult", "_mean_edge_err", "no_progress_limit",
                     "invalid_penalty", "eta_err", "eta_xtalk", "eta_xz_step",
                     "eta_dist", "cnot_cost", "use_scheduler", "eta_time",
                     "eta_xtalk_par", "eta_idle", "eta_parallel",
                     "xtalk_alpha", "swap_duration", "lambda_fid",
                     "lambda_layout", "fidelity_fn", "sabre_fid_map",
                     "sref_override", "random_init", "init_mapping",
                     "max_episode_steps", "unfinished_penalty",
                     "max_num_edges", "mapping_budget", "enable_mapping_phase",
                     "_rng", "_gnn", "_edge_feat_dim", "_gnn_dim",
                     "action_space", "observation_space", "commit_action",
                     "skip_action", "max_ready", "w_xt_launch", "w_zz",
                     "eta_ao", "step_cap_factor", "scheduling_only"):
            new.__setattr__(attr, getattr(self, attr))
        new.mapping = self.mapping.copy()
        new.mapping_phase = self.mapping_phase
        new._mapping_swaps = self._mapping_swaps
        new.executed = self.executed.copy()
        new._swap_counter = self._swap_counter
        new._episode_step = self._episode_step
        new._swap_history = self._swap_history.copy()
        new._last_progress_swap = self._last_progress_swap
        new._qubit_swap_since_exec = self._qubit_swap_since_exec.copy()
        new._xz_errors = self._xz_errors.copy()
        new._phys_circuit = self._phys_circuit.copy()
        new.executable_2q = self.executable_2q.copy()
        new.timing = self.timing.clone() if self.timing is not None else None
        new._pending_swaps = list(self._pending_swaps)
        new._pending_measures = list(self._pending_measures)
        new._effective_initial_mapping = list(self._effective_initial_mapping)
        new.clock = self.clock
        new.in_flight = [dict(f) for f in self.in_flight]
        new._ready_since = dict(self._ready_since)
        new._candidate_slots = list(self._candidate_slots)
        new._cycle_launch_count = self._cycle_launch_count
        new._cum_theta = self._cum_theta
        new._cum_idle_us = self._cum_idle_us
        new._skip_count = self._skip_count
        new._terminal_done = self._terminal_done
        new._last_exec_count = self._last_exec_count
        new._steps_since_progress = self._steps_since_progress
        return new
