from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from sim.sim import NoiseConfig
from routing.graph.circuit_dag import CircuitDAG
from routing.graph.features import HardwareFeatures
from routing.rl.env import RoutingEnv
from routing.rl.agent import PPOAgent
from routing.gnn.encoder import SubGNN
from routing.routing import greedy_route, sabre_route
from routing.timing import GATE_DURATION_TABLE, FALLBACK_DURATION, schedule_routed_circuit, GreedyScheduler


# ---------------------------------------------------------------------------
#  Metrics per circuit
# ---------------------------------------------------------------------------

@dataclass
class CircuitMetrics:
    circuit_path: str
    completed: bool
    num_swaps: int
    gates_executed: int
    total_gates: int
    episode_steps: int
    wall_time_ms: float
    terminal_xz: Optional[float]
    truncated_remaining: int = 0
    fidelity: Optional[float] = None
    mapping_swaps: int = 0
    circuit_time_us: Optional[float] = None
    crosstalk_events: Optional[float] = None
    schedule: Optional[list] = None        # 带并行时序的门调度序列（启用调度器时）
    phys_qasm: Optional[str] = None        # 含 SWAP 插入的物理电路 QASM
    sched_stats: Optional[dict] = None     # D3: makespan/密度/并行峰值等调度统计
    sabre_initial_layout: Optional[list] = None  # SABRE 初始布局（warm-start 用）


def _sched_stats_from_timing(timing, dag=None) -> Optional[dict]:
    """D3：从事件级时序内核汇总调度统计。"""
    if timing is None:
        return None
    total_time = timing.total_time
    serial = timing.serial_dur
    density = (serial / total_time) if total_time > 1e-9 else 0.0
    pc = {}
    for e in timing.schedule_log:
        # 与 SABRE（schedule_routed_circuit 把 SWAP 当 2Q 门计入波次）同口径：
        # PPO 的 SWAP 调度事件一并计入峰值并行度，避免指标不对称压低 PPO。
        w = e.get("wave", -1)
        pc[w] = pc.get(w, 0) + 1
    peak = max(pc.values()) if pc else 0
    crit_lb = 0.0
    if dag is not None:
        dur = {g.index: GATE_DURATION_TABLE.get(g.name, FALLBACK_DURATION)
               for g in dag.gates}
        best = [0.0] * dag.num_gates
        for g in dag.gates:
            b = dur[g.index]
            for p in g.predecessors:
                b = max(b, dur[g.index] + best[p])
            best[g.index] = b
        crit_lb = max(best) if best else 0.0
    return {
        "makespan_us": total_time,
        "serial_dur_us": serial,
        "density": density,
        "peak_parallel": peak,
        "critical_path_lb_us": crit_lb,
        "crosstalk_events": timing.crosstalk_events,
    }


@dataclass
class SummaryStats:
    n: int
    comp_rate: float
    swaps_mean: float
    swaps_std: float
    swaps_min: int
    swaps_max: int
    steps_mean: float
    steps_std: float
    time_mean_ms: float
    time_std_ms: float
    xz_mean: Optional[float] = None
    fidelity_mean: Optional[float] = None
    mapping_mean: Optional[float] = None
    sched_time_mean: Optional[float] = None     # D3: 平均 makespan (µs)
    sched_density_mean: Optional[float] = None  # D3: 平均并行密度
    sched_crosstalk_mean: Optional[float] = None  # D3: 平均串扰累计


# ---------------------------------------------------------------------------
#  Hardware helpers
# ---------------------------------------------------------------------------

def _normalize_dict_keys(d):
    if not isinstance(d, dict):
        return d
    out = {}
    for k, v in d.items():
        if isinstance(k, str):
            try:
                k = ast.literal_eval(k)
            except (ValueError, SyntaxError):
                pass
        out[k] = v
    return out


def _lists_to_dict(raw, coupling_map):
    if raw is None or isinstance(raw, (int, float, dict)):
        return _normalize_dict_keys(raw)
    result = {}
    for item in raw:
        q1, q2, v = int(item[0]), int(item[1]), float(item[2])
        result[(q1, q2)] = v
        result[(q2, q1)] = v
    return result


def load_topo(path: str) -> tuple:
    with open(path) as f:
        topo = json.load(f)
    coupling_map = [tuple(e) for e in topo['coupling_map']]
    dp = topo['device_params']
    tqe = _lists_to_dict(dp.get('two_q_gate_error', 0.01), coupling_map)
    cs = _lists_to_dict(topo.get('crosstalk_strength'), coupling_map)
    config = NoiseConfig(
        t1_times=dp['t1_times'],
        t2_times=dp['t2_times'],
        freq_ghz=dp['freq_ghz'],
        single_q_gate_error=dp.get('single_q_gate_error', 0.001),
        two_q_gate_error=tqe,
        coupling_map=coupling_map,
        readout_error=dp['readout_error'],
        shots=dp.get('shots', 1024),
        crosstalk_strength=cs,
    )
    hw = HardwareFeatures.from_noise_config(config)
    return config, hw, coupling_map


def make_default_hw(num_qubits: int = 5) -> tuple:
    coupling = [(i, i + 1) for i in range(num_qubits - 1)]
    config = NoiseConfig(
        t1_times=[50.0] * num_qubits,
        t2_times=[70.0] * num_qubits,
        freq_ghz=[5.0] * num_qubits,
        single_q_gate_error=0.001,
        two_q_gate_error=0.01,
        coupling_map=coupling,
        readout_error=[0.02] * num_qubits,
        shots=1024,
    )
    hw = HardwareFeatures.from_noise_config(config)
    return config, hw, list(config.coupling_map)


def load_split(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_qc(data_dir: str, rel_path: str, seed: int = 0):
    with open(os.path.join(data_dir, rel_path), 'rb') as f:
        qc = pickle.load(f)
    if qc.num_parameters > 0:
        import numpy as np
        rng = np.random.default_rng(seed)
        qc = qc.assign_parameters({p: rng.uniform(0, 2 * np.pi) for p in qc.parameters})
    return qc


# ---------------------------------------------------------------------------
#  Fidelity stub
# ---------------------------------------------------------------------------

def compute_fidelity(qc, config, mapping, executed) -> Optional[float]:
    return None

def build_fidelity_fn(fidelity_sim: str, config, num_trajectories: int = 64, seed: Optional[int] = None,
                      backend: str = "auto"):
    """构造 RoutingEnv 的 fidelity_fn（--fidelity-sim trajectory/trajectory_sched/analytic 时启用）。

    backend: 轨迹模拟器后端（auto/cpu/cuda/cuda:N；默认 auto=CUDA 可用即走 GPU）。
    """
    if fidelity_sim == "trajectory":
        from sim.trajectory_sim import make_trajectory_fidelity_fn
        return make_trajectory_fidelity_fn(config, num_trajectories=num_trajectories, seed=seed,
                                           backend=backend)
    if fidelity_sim == "trajectory_sched":
        from sim.trajectory_sim import make_trajectory_fidelity_fn
        return make_trajectory_fidelity_fn(config, num_trajectories=num_trajectories,
                                           seed=seed, scheduled=True, backend=backend)
    if fidelity_sim == "trajectory_v2":
        from sim.trajectory_sim_v2 import make_event_fidelity_fn
        return make_event_fidelity_fn(config, num_trajectories=num_trajectories,
                                      seed=seed, backend=backend)
    if fidelity_sim == "trajectory_v3":
        from sim.trajectory_sim_v3 import make_event_fidelity_fn_v3
        return make_event_fidelity_fn_v3(config, num_trajectories=num_trajectories,
                                         seed=seed, backend=backend)
    if fidelity_sim == "analytic":
        from sim.trajectory_sim import make_analytic_fidelity_fn
        return make_analytic_fidelity_fn(config)
    return None


def phys_fidelity(phys, config, fidelity_sim: str, num_trajectories: int = 64, seed: Optional[int] = None) -> Optional[float]:
    """对路由结果电路计算保真度（基线用）：aer=Hellinger 保真度，trajectory=态保真度，
    trajectory_sched=调度感知态保真度（对已有电路做贪心波次编排）。"""
    if fidelity_sim == "trajectory":
        from sim.trajectory_sim import trajectory_circuit_fidelity
        return trajectory_circuit_fidelity(phys, config, num_trajectories=num_trajectories, seed=seed)
    if fidelity_sim == "trajectory_sched":
        from sim.trajectory_sim import trajectory_circuit_fidelity
        return trajectory_circuit_fidelity(phys, config, num_trajectories=num_trajectories,
                                           seed=seed, scheduled=True)
    if fidelity_sim == "trajectory_v2":
        from sim.trajectory_sim_v2 import trajectory_circuit_fidelity_events
        return trajectory_circuit_fidelity_events(phys, config,
                                                  num_trajectories=num_trajectories,
                                                  seed=seed)
    if fidelity_sim == "trajectory_v3":
        from sim.trajectory_sim_v3 import trajectory_circuit_fidelity_events_v3
        return trajectory_circuit_fidelity_events_v3(phys, config,
                                                     num_trajectories=num_trajectories,
                                                     seed=seed)
    from sim.sim import NoiseSimulator
    from qiskit_aer import AerSimulator
    try:
        from sim.trajectory_sim import _reduce_phys_circuit_for_fidelity
        phys, config = _reduce_phys_circuit_for_fidelity(phys, config)
    except Exception:
        pass
    meas = phys.copy()
    meas.measure_all()
    noise_sim = NoiseSimulator(config)
    meas_t = noise_sim._transpile(meas)
    shots = config.shots
    ideal_sim = AerSimulator()
    ideal_job = ideal_sim.run(meas_t, shots=shots)
    ideal_counts = ideal_job.result().get_counts()
    noisy_counts = noise_sim.run(meas_t, shots=shots, skip_transpile=True)
    from utils.metrics import counts_fidelity
    return counts_fidelity(ideal_counts, noisy_counts)


# ---------------------------------------------------------------------------
#  Single-circuit evaluation: PPO agent
# ---------------------------------------------------------------------------

def evaluate_circuit(
    dag: CircuitDAG,
    hw: HardwareFeatures,
    coupling_map: list,
    agent: PPOAgent,
    reward_mode: str = 'routing',
    max_episode_steps: int = 200,
    deterministic: bool = True,
    seed: int = 0,
    noise_config: Optional[NoiseConfig] = None,
    use_deadlock_mask: bool = True,
    max_num_qubits: Optional[int] = None,
    max_num_edges: Optional[int] = None,
    random_init: bool = False,
    mapping_phase: Optional[bool] = None,
    init_mapping: Optional[list] = None,
    use_scheduler: bool = False,
    eta_time: float = 0.01,
    eta_xtalk_par: float = 1.0,
    eta_idle: float = 0.005,
    eta_parallel: float = 0.05,
    xtalk_alpha: float = 0.03,
    swap_duration: float = 0.9,
    dump_schedule: bool = False,
    fidelity_fn=None,
    config=None,
    fidelity_sim: str = 'aer',
    num_trajectories: int = 16,
    traj_seed: int = 0,
    edge_noise_features: bool = False,
    beta_noise: float = 0.0,
    w_err: float = 0.0,
    w_xt: float = 0.0,
    w_xt_swap: float = 0.0,
    pot_progress_b: float = 0.045,
    pot_1q_reward: bool = True,
    lambda_budget: float = 0.0,
    sabre_swap_budget: Optional[int] = None,
) -> CircuitMetrics:
    import torch
    env = RoutingEnv(
        dag, hw, coupling_map, reward_mode=reward_mode,
        max_episode_steps=max_episode_steps,
        random_init=random_init, seed=seed,
        gnn=agent.gnn, use_gnn=agent.gnn is not None,
        noise_config=noise_config if reward_mode != 'routing' else None,
        max_num_qubits=max_num_qubits,
        max_num_edges=max_num_edges,
        mapping_phase=agent.with_commit,
        init_mapping=init_mapping,
        fidelity_fn=fidelity_fn,
        edge_noise_features=edge_noise_features,
        beta_noise=beta_noise,
        w_err=w_err, w_xt=w_xt, w_xt_swap=w_xt_swap,
        pot_progress_b=pot_progress_b, pot_1q_reward=pot_1q_reward,
        use_scheduler=use_scheduler,
        eta_time=eta_time, eta_xtalk_par=eta_xtalk_par, eta_idle=eta_idle,
        eta_parallel=eta_parallel, xtalk_alpha=xtalk_alpha, swap_duration_us=swap_duration,
        lambda_budget=lambda_budget, sabre_swap_budget=sabre_swap_budget,
    )

    obs, _ = env.reset()
    # 支持「训练带映射阶段、但评估仅路由」：保留 enable_mapping_phase（phase 特征不丢），
    # 仅覆盖运行期 mapping_phase。A1 配置（SABRE 布局 + 关映射）走此分支。
    if mapping_phase is not None and mapping_phase != env.mapping_phase:
        env.mapping_phase = mapping_phase
        if env.mapping_phase:
            env.mapping_swaps = 0
        obs = env._obs()
    t0 = time.perf_counter()

    done, truncated = False, False
    step = 0
    while not done and not truncated:
        with torch.no_grad():
            mask = None
            if use_deadlock_mask and hasattr(env, 'get_deadlock_mask') and agent.gnn is not None:
                dm = env.get_deadlock_mask()
                um = env.get_unmapped_mask()
                combined = dm | um
                n_a = agent.num_edges + 1 if agent.with_commit else agent.num_edges
                mask = torch.zeros(n_a, dtype=torch.bool, device=agent.device)
                mask[:len(coupling_map)] = True
                for i in range(min(len(combined), len(mask))):
                    if combined[i]:
                        mask[i] = False
                if agent.with_commit:
                    mask[agent.num_edges] = env.mapping_phase
                mask = mask.unsqueeze(0)
            logits, _ = agent._forward_obs(obs, action_mask=mask)
            action = logits.argmax(-1).item() if deterministic \
                     else torch.distributions.Categorical(logits=logits).sample().item()
        obs, reward, done, truncated, info = env.step(action)
        step += 1

    wall_time_ms = (time.perf_counter() - t0) * 1000

    sched = None
    pqasm = None
    if dump_schedule and getattr(env, 'timing', None) is not None:
        sched = env.timing.schedule_log
        try:
            from qiskit import qasm2
            pqasm = qasm2.dumps(env._phys_circuit)
        except Exception:
            try:
                from qiskit import qasm3
                pqasm = qasm3.dumps(env._phys_circuit)
            except Exception:
                pqasm = None

    # 报告用保真度：即便 reward_mode='routing'（PPO 环境内不记录保真度），也对最终路由
    # 电路用与 SABRE 基线同口径的模拟器计算一次，便于同口径对比。不影响 PPO 动作
    # （动作来自确定性策略 argmax，与环境 reward/噪声配置无关）。
    # aer 模式下也用 post-hoc Hellinger 保真度（phys_fidelity）覆盖 env 内的 min-count，
    # 确保 PPO 与 SABRE 基线使用同一口径。
    post_fid = None
    if config is not None and fidelity_sim in ('aer', 'trajectory', 'trajectory_sched', 'trajectory_v2'):
        try:
            post_fid = phys_fidelity(
                env._phys_circuit, config, fidelity_sim,
                num_trajectories=num_trajectories, seed=traj_seed,
            )
        except Exception:
            post_fid = None

    return CircuitMetrics(
        circuit_path='',
        completed=done,
        num_swaps=env._swap_counter,
        gates_executed=len(env.executed),
        total_gates=dag.num_gates,
        episode_steps=step,
        wall_time_ms=wall_time_ms,
        terminal_xz=info.get('terminal_XZ', None),
        truncated_remaining=info.get('truncated_remaining', 0),
        fidelity=post_fid if post_fid is not None else info.get('fidelity', None),
        mapping_swaps=env._mapping_swaps,
        circuit_time_us=env.timing.total_time if env.timing is not None else None,
        crosstalk_events=env.timing.crosstalk_events if env.timing is not None else None,
        schedule=sched,
        phys_qasm=pqasm,
        sched_stats=_sched_stats_from_timing(env.timing, dag),
    )


# ---------------------------------------------------------------------------
#  Single-circuit evaluation: 1-step Beam Search
# ---------------------------------------------------------------------------

def evaluate_circuit_beam(
    dag: CircuitDAG,
    hw: HardwareFeatures,
    coupling_map: list,
    agent: PPOAgent,
    reward_mode: str = 'routing',
    max_episode_steps: int = 200,
    seed: int = 0,
    noise_config: Optional[NoiseConfig] = None,
    beam_width: int = 3,
    vhead: str = 'route',
    max_num_qubits: Optional[int] = None,
    max_num_edges: Optional[int] = None,
    random_init: bool = False,
    mapping_phase: Optional[bool] = None,
    init_mapping: Optional[list] = None,
    use_scheduler: bool = False,
    eta_time: float = 0.01,
    eta_xtalk_par: float = 1.0,
    eta_idle: float = 0.005,
    eta_parallel: float = 0.05,
    xtalk_alpha: float = 0.03,
    swap_duration: float = 0.9,
    dump_schedule: bool = False,
    fidelity_fn=None,
    config=None,
    fidelity_sim: str = 'aer',
    num_trajectories: int = 16,
    traj_seed: int = 0,
    lookahead_features: bool = True,
    logit_noise_std: float = 0.0,
    logit_noise_seed: Optional[int] = None,
    no_progress_limit: int = 0,
    edge_noise_features: bool = False,
    beta_noise: float = 0.0,
    w_err: float = 0.0,
    w_xt: float = 0.0,
    w_xt_swap: float = 0.0,
    pot_progress_b: float = 0.045,
    pot_1q_reward: bool = True,
    lambda_budget: float = 0.0,
    sabre_swap_budget: Optional[int] = None,
) -> CircuitMetrics:
    import torch
    env = RoutingEnv(
        dag, hw, coupling_map, reward_mode=reward_mode,
        max_episode_steps=max_episode_steps,
        random_init=random_init, seed=seed,
        gnn=agent.gnn, use_gnn=agent.gnn is not None,
        noise_config=noise_config if reward_mode != 'routing' else None,
        max_num_qubits=max_num_qubits,
        max_num_edges=max_num_edges,
        mapping_phase=agent.with_commit,
        init_mapping=init_mapping,
        fidelity_fn=fidelity_fn,
        lookahead_features=lookahead_features,
        edge_noise_features=edge_noise_features,
        beta_noise=beta_noise,
        w_err=w_err, w_xt=w_xt, w_xt_swap=w_xt_swap,
        pot_progress_b=pot_progress_b, pot_1q_reward=pot_1q_reward,
        no_progress_limit=no_progress_limit,
        use_scheduler=use_scheduler,
        eta_time=eta_time, eta_xtalk_par=eta_xtalk_par, eta_idle=eta_idle,
        eta_parallel=eta_parallel, xtalk_alpha=xtalk_alpha, swap_duration_us=swap_duration,
        lambda_budget=lambda_budget, sabre_swap_budget=sabre_swap_budget,
    )

    obs, _ = env.reset()
    # 支持「训练带映射阶段、但评估仅路由」：保留 enable_mapping_phase（phase 特征不丢），
    # 仅覆盖运行期 mapping_phase。A1 配置（SABRE 布局 + 关映射）走此分支。
    if mapping_phase is not None and mapping_phase != env.mapping_phase:
        env.mapping_phase = mapping_phase
        if env.mapping_phase:
            env.mapping_swaps = 0
        obs = env._obs()
    t0 = time.perf_counter()

    done, truncated = False, False
    step = 0
    # R3b 多试验 beam：logit ε-扰动（每步从固定种子的 generator 采样，
    # 不同 seed 的试验探索不同分支——SABRE ε-random tie-break 的 RL 版）
    noise_gen = None
    if logit_noise_std > 0 and logit_noise_seed is not None:
        noise_gen = torch.Generator(device='cpu').manual_seed(logit_noise_seed)
    while not done and not truncated:
        with torch.no_grad():
            # Build base action mask (valid edges + deadlock + commit)
            n_a = agent.num_edges + 1 if agent.with_commit else agent.num_edges
            mask = torch.zeros(n_a, dtype=torch.bool, device=agent.device)
            mask[:len(coupling_map)] = True
            if hasattr(env, 'get_deadlock_mask'):
                dm = env.get_deadlock_mask()
                um = env.get_unmapped_mask()
                for i in range(min(len(dm), len(mask))):
                    if dm[i] or um[i]:
                        mask[i] = False
            if agent.with_commit:
                mask[agent.num_edges] = env.mapping_phase
            mask = mask.unsqueeze(0)

            logits, _ = agent._forward_obs(obs, action_mask=mask)
            masked_logits = logits[0].clone()
            if noise_gen is not None:
                masked_logits += logit_noise_std * torch.randn(
                    masked_logits.shape, generator=noise_gen)
            masked_logits[~mask[0]] = -1e9
            k = min(beam_width, (mask[0].sum().item()))
            topk_scores, topk_indices = masked_logits.topk(k)

            # 批量：先克隆并 step（跳过无用 obs 计算），再批量 GNN forward
            clones = []
            for i in range(topk_indices.shape[0]):
                a = topk_indices[i].item()
                clone = env.clone()
                _, reward_c, done_c, truncated_c, info_c = clone.step(a, compute_obs=False)
                clones.append((clone, a, reward_c, done_c, truncated_c))

            best_action, best_score = topk_indices[0].item(), -float('inf')
            best_clone_obs = None
            if agent.gnn is not None:
                graph_datas = [c.build_graph_data() for c, *_ in clones]
                qubit_hs = agent.gnn.node_embeddings_batched(graph_datas)
                clone_obs_list = [t[0]._obs(qubit_h=qh.cpu().numpy())
                                  for t, qh in zip(clones, qubit_hs)]
                # K 个候选的 V(s') 批量前向（单次替代 K 次）；
                # vhead='la' 时用 V_LA 多步价值头（训练期 beam expectimax 训得）
                if vhead == 'la':
                    _, values = agent._forward_obs_batch_vla(np.stack(clone_obs_list))
                else:
                    _, values = agent._forward_obs_batch(np.stack(clone_obs_list))
                # P1-a 推理版：父状态势函数（进度判定的基准，每步一次）
                phi_parent = env._phi() if (sabre_swap_budget is not None
                                            and lambda_budget > 0) else None
                for (clone, a, reward_c, done_c, truncated_c), clone_obs, v in zip(
                        clones, clone_obs_list, values):
                    if done_c or truncated_c:
                        score = reward_c
                    else:
                        score = reward_c + agent.gamma * v.item()
                    # P1-a 推理版（排序有效）：常数级预算惩罚不改变同步内排序
                    # （每候选均 +1 swap），必须与候选特异进度交互——超预算
                    # 状态下既未解锁门、也未改善势函数（距离）的候选按超支深度受罚。
                    # 进度用 Φ 改进而非仅门解锁：深电路上门解锁稀疏，Φ 改进稠密
                    if (sabre_swap_budget is not None and not done_c
                            and not truncated_c):
                        over_by = clone._swap_counter - sabre_swap_budget
                        if over_by > 0:
                            prog = ((len(clone.executed) - len(env.executed)) > 0
                                    or (clone._phi() > phi_parent + 1e-9))
                            if not prog:
                                score -= lambda_budget * over_by
                    if score > best_score:
                        best_score = score
                        best_action = a
                        best_clone_obs = clone_obs
            else:
                clone_obs_list = [c._obs() for c, *_ in clones]
                _, values = agent._forward_obs_batch(np.stack(clone_obs_list))
                phi_parent = env._phi() if (sabre_swap_budget is not None
                                            and lambda_budget > 0) else None
                for (clone, a, reward_c, done_c, truncated_c), clone_obs, v in zip(
                        clones, clone_obs_list, values):
                    if done_c or truncated_c:
                        score = reward_c
                    else:
                        score = reward_c + agent.gamma * v.item()
                    if (sabre_swap_budget is not None and not done_c
                            and not truncated_c):
                        over_by = clone._swap_counter - sabre_swap_budget
                        if over_by > 0:
                            prog = ((len(clone.executed) - len(env.executed)) > 0
                                    or (clone._phi() > phi_parent + 1e-9))
                            if not prog:
                                score -= lambda_budget * over_by
                    if score > best_score:
                        best_score = score
                        best_action = a
                        best_clone_obs = clone_obs

        # 胜出 clone 就是 step 后的环境：跳过 env 的重复 GNN obs
        obs, reward, done, truncated, info = env.step(best_action, compute_obs=False)
        if best_clone_obs is not None:
            obs = best_clone_obs
        step += 1

    wall_time_ms = (time.perf_counter() - t0) * 1000

    sched = None
    pqasm = None
    if dump_schedule and getattr(env, 'timing', None) is not None:
        sched = env.timing.schedule_log
        try:
            from qiskit import qasm2
            pqasm = qasm2.dumps(env._phys_circuit)
        except Exception:
            try:
                from qiskit import qasm3
                pqasm = qasm3.dumps(env._phys_circuit)
            except Exception:
                pqasm = None

    # 报告用保真度：即便 reward_mode='routing'（PPO 环境内不记录保真度），也对最终路由
    # 电路用与 SABRE 基线同口径的模拟器计算一次，便于同口径对比。不影响 PPO 动作
    # （动作来自确定性策略 argmax，与环境 reward/噪声配置无关）。
    # aer 模式下也用 post-hoc Hellinger 保真度（phys_fidelity）覆盖 env 内的 min-count，
    # 确保 PPO 与 SABRE 基线使用同一口径。
    post_fid = None
    if config is not None and fidelity_sim in ('aer', 'trajectory', 'trajectory_sched', 'trajectory_v2'):
        try:
            post_fid = phys_fidelity(
                env._phys_circuit, config, fidelity_sim,
                num_trajectories=num_trajectories, seed=traj_seed,
            )
        except Exception:
            post_fid = None

    return CircuitMetrics(
        circuit_path='',
        completed=done,
        num_swaps=env._swap_counter,
        gates_executed=len(env.executed),
        total_gates=dag.num_gates,
        episode_steps=step,
        wall_time_ms=wall_time_ms,
        terminal_xz=info.get('terminal_XZ', None),
        truncated_remaining=info.get('truncated_remaining', 0),
        fidelity=post_fid if post_fid is not None else info.get('fidelity', None),
        mapping_swaps=env._mapping_swaps,
        circuit_time_us=env.timing.total_time if env.timing is not None else None,
        crosstalk_events=env.timing.crosstalk_events if env.timing is not None else None,
        schedule=sched,
        phys_qasm=pqasm,
        sched_stats=_sched_stats_from_timing(env.timing, dag),
    )


# ---------------------------------------------------------------------------
#  Single-circuit evaluation: Random baseline
# ---------------------------------------------------------------------------

def evaluate_random(
    dag: CircuitDAG,
    hw: HardwareFeatures,
    coupling_map: list,
    reward_mode: str = 'routing',
    max_episode_steps: int = 200,
    seed: int = 0,
    noise_config: Optional[NoiseConfig] = None,
    random_init: bool = False,
    fidelity_fn=None,
) -> CircuitMetrics:
    env = RoutingEnv(
        dag, hw, coupling_map, reward_mode=reward_mode,
        max_episode_steps=max_episode_steps,
        random_init=random_init, seed=seed,
        use_gnn=False,
        noise_config=noise_config if reward_mode != 'routing' else None,
        fidelity_fn=fidelity_fn,
    )

    obs, _ = env.reset()
    # 支持「训练带映射阶段、但评估仅路由」：保留 enable_mapping_phase（phase 特征不丢），
    # 仅覆盖运行期 mapping_phase。A1 配置（SABRE 布局 + 关映射）走此分支。
    if mapping_phase is not None and mapping_phase != env.mapping_phase:
        env.mapping_phase = mapping_phase
        if env.mapping_phase:
            env.mapping_swaps = 0
        obs = env._obs()
    t0 = time.perf_counter()

    done, truncated = False, False
    step = 0
    while not done and not truncated:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        step += 1

    wall_time_ms = (time.perf_counter() - t0) * 1000

    return CircuitMetrics(
        circuit_path='',
        completed=done,
        num_swaps=env._swap_counter,
        gates_executed=len(env.executed),
        total_gates=dag.num_gates,
        episode_steps=step,
        wall_time_ms=wall_time_ms,
        terminal_xz=info.get('terminal_XZ', None),
        truncated_remaining=info.get('truncated_remaining', 0),
        fidelity=info.get('fidelity', None),
    )


# ---------------------------------------------------------------------------
#  Greedy baseline
# ---------------------------------------------------------------------------

def evaluate_greedy(
    qc,
    config: NoiseConfig,
    reward_mode: str = 'routing',
    fidelity_sim: str = 'aer',
    num_trajectories: int = 64,
    seed: int = 0,
) -> CircuitMetrics:
    dag = CircuitDAG.from_circuit(qc)
    t0 = time.perf_counter()
    phys, info = greedy_route(qc, config)
    wall_time_ms = (time.perf_counter() - t0) * 1000

    # 即便 reward_mode='routing'，只要指定了态级保真度模拟器（trajectory/
    # trajectory_sched），也对基线计算保真度，便于与 PPO 的 fidelity_fn 结果同口径对比。
    fid = phys_fidelity(phys, config, fidelity_sim, num_trajectories, seed) \
        if (reward_mode != 'routing' or fidelity_sim in ('trajectory', 'trajectory_sched', 'trajectory_v2')) else None

    return CircuitMetrics(
        circuit_path='',
        completed=True,
        num_swaps=info['num_swaps'],
        gates_executed=dag.num_gates,
        total_gates=dag.num_gates,
        episode_steps=0,
        wall_time_ms=wall_time_ms,
        terminal_xz=None,
        fidelity=fid,
    )


# ---------------------------------------------------------------------------
#  SABRE baseline
# ---------------------------------------------------------------------------

def evaluate_sabre(
    qc,
    config: NoiseConfig,
    reward_mode: str = 'routing',
    heuristic: str = 'decay',
    swap_trials: int = 20,
    seed: int = 0,
    fidelity_sim: str = 'aer',
    num_trajectories: int = 64,
    hw=None,
    use_scheduler: bool = False,
    xtalk_alpha: float = 0.0,
) -> CircuitMetrics:
    dag = CircuitDAG.from_circuit(qc)
    t0 = time.perf_counter()
    phys, info = sabre_route(qc, config, heuristic=heuristic, swap_trials=swap_trials, seed=seed)
    wall_time_ms = (time.perf_counter() - t0) * 1000

    # 即便 reward_mode='routing'，只要指定了态级保真度模拟器（trajectory/
    # trajectory_sched），也对基线计算保真度，便于与 PPO 的 fidelity_fn 结果同口径对比。
    fid = phys_fidelity(phys, config, fidelity_sim, num_trajectories, seed) \
        if (reward_mode != 'routing' or fidelity_sim in ('trajectory', 'trajectory_sched', 'trajectory_v2')) else None

    sched_stats = None
    if use_scheduler and hw is not None:
        # D2：用同一事件级调度器给 SABRE 输出电路排程，得到公平可比的 makespan/并行度。
        # 与 PPO 环境使用相同的 GreedyScheduler（criticality+LPT+串扰优先级），
        # 使对比只反映「路由/布局质量」而非调度策略差异。
        phys_dag = CircuitDAG.from_circuit(phys)
        _, _, sched_stats = schedule_routed_circuit(
            phys_dag, hw, mapping=list(range(hw.num_qubits)),
            scheduler=GreedyScheduler(), xtalk_alpha=xtalk_alpha)

    return CircuitMetrics(
        circuit_path='',
        completed=True,
        num_swaps=info['num_swaps'],
        gates_executed=dag.num_gates,
        total_gates=dag.num_gates,
        episode_steps=0,
        wall_time_ms=wall_time_ms,
        terminal_xz=None,
        fidelity=fid,
        sched_stats=sched_stats,
        sabre_initial_layout=info.get('initial_layout'),
    )


# ---------------------------------------------------------------------------
#  Aggregate stats
# ---------------------------------------------------------------------------

def aggregate(metrics: List[CircuitMetrics]) -> SummaryStats:
    n = len(metrics)
    completed = [m for m in metrics if m.completed]
    comp_rate = len(completed) / n if n > 0 else 0.0

    completed_n = len(completed) if completed else 1
    swaps = np.array([m.num_swaps for m in completed], dtype=float) if completed else np.zeros(1)
    steps = np.array([m.episode_steps for m in completed], dtype=float) if completed else np.zeros(1)
    times = np.array([m.wall_time_ms for m in metrics], dtype=float)
    xz_vals = [m.terminal_xz for m in completed if m.terminal_xz is not None]
    fid_vals = [m.fidelity for m in completed if m.fidelity is not None]
    map_vals = [m.mapping_swaps for m in completed]

    sched_times = [m.sched_stats['makespan_us'] for m in completed if m.sched_stats]
    sched_dens = [m.sched_stats['density'] for m in completed if m.sched_stats]
    sched_xt = [m.sched_stats['crosstalk_events'] for m in completed if m.sched_stats]

    return SummaryStats(
        n=n,
        comp_rate=comp_rate,
        swaps_mean=float(np.mean(swaps)),
        swaps_std=float(np.std(swaps)),
        swaps_min=int(np.min(swaps)),
        swaps_max=int(np.max(swaps)),
        steps_mean=float(np.mean(steps)),
        steps_std=float(np.std(steps)),
        time_mean_ms=float(np.mean(times)),
        time_std_ms=float(np.std(times)),
        xz_mean=float(np.mean(xz_vals)) if xz_vals else None,
        fidelity_mean=float(np.mean(fid_vals)) if fid_vals else None,
        mapping_mean=float(np.mean(map_vals)) if map_vals else None,
        sched_time_mean=float(np.mean(sched_times)) if sched_times else None,
        sched_density_mean=float(np.mean(sched_dens)) if sched_dens else None,
        sched_crosstalk_mean=float(np.mean(sched_xt)) if sched_xt else None,
    )


# ---------------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------------

def print_header(show_fidelity: bool = False):
    parts = [
        f"{'Method':<8s}",
        f"{'Time(ms)':>8s}",
        f"{'Comp%':>6s}",
        f"{'SWAPs':>16s}",
        f"{'Map':>6s}",
        f"{'Steps':>16s}",
        f"{'XZ':>10s}",
    ]
    if show_fidelity:
        parts.append(f"{'Fidelity':>9s}")
    sep = '  '.join(parts)
    print(sep)
    print('-' * len(sep))


def print_report(
    label: str,
    stats: SummaryStats,
    show_fidelity: bool = False,
):
    parts = [
        f"{label:<8s}",
        f"{stats.time_mean_ms:>8.1f}",
        f"{stats.comp_rate * 100:>5.1f}%",
        f"{stats.swaps_mean:>6.1f} +/- {stats.swaps_std:<5.1f}",
        f"{stats.mapping_mean:>6.1f}" if stats.mapping_mean is not None else f"{'--':>6s}",
        f"{stats.steps_mean:>6.0f} +/- {stats.steps_std:<5.0f}",
    ]
    if stats.xz_mean is not None:
        parts.append(f"{stats.xz_mean:>8.4f}")
    else:
        parts.append('      --   ')
    if show_fidelity:
        if stats.fidelity_mean is not None:
            parts.append(f"{stats.fidelity_mean:>8.4f}")
        else:
            parts.append('      --   ')
    print('  '.join(parts))
    if stats.sched_time_mean is not None:
        extra = (f"    [sched] makespan={stats.sched_time_mean:.2f}µs  "
                 f"par_density={stats.sched_density_mean:.3f}  "
                 f"xtalk={stats.sched_crosstalk_mean:.3f}")
        print(extra)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained routing policy')
    parser.add_argument("--edge-hidden", type=int, default=64,
                        help="edge_mlp 首层隐藏宽度（E17 容量升级：64→128）")
    parser.add_argument('--model', type=str, required=True,
                        help='policy checkpoint path')
    parser.add_argument('--data-dir', type=str, default='../traindata',
                        help='dataset root directory')
    parser.add_argument('--split', type=str, default='stage1_phase3',
                        help='split file name in <data-dir>/splits/ '
                             '(without .txt; e.g. stage1_phase3 or large_n10_test)')
    parser.add_argument('--reward-mode', type=str, default='routing',
                        choices=['routing', 'noise_aware', 'fidelity_shaping'])
    parser.add_argument('--topo', type=str, default=None,
                        help='hardware topology JSON')
    parser.add_argument('--num-qubits', type=int, default=5,
                        help='physical qubits (default linear chain)')
    parser.add_argument('--max-num-qubits', type=int, default=None,
                        help='fixed obs qubit dim for unified models (default=num-qubits)')
    parser.add_argument('--max-num-edges', type=int, default=None,
                        help='fixed obs/action edge dim for unified models '
                             '(default: topology edge count; required when '
                             'the model was trained with a larger fixed dim)')
    parser.add_argument('--max-episode-steps', type=int, default=200)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--torch-threads', type=int, default=8,
                        help='torch CPU 线程数上限（小图推理多线程同步开销主导，默认 8）')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-gnn', action='store_true', default=False,
                        help='model was trained without GNN')
    parser.add_argument('--mapping-phase', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='model was trained with mapping phase (commit action)')
    parser.add_argument('--random-init', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='start each episode from a random initial layout')
    parser.add_argument('--warm-start-sabre', action='store_true', default=False,
                        help='warm-start: 用每电路 SABRE 初始布局作为 PPO init_mapping')
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='use argmax for action selection')
    parser.add_argument('--baselines', action='store_true', default=False,
                        help='also run random, greedy, and SABRE baselines')
    parser.add_argument('--no-random', action='store_true', default=False,
                        help='skip the random baseline (keep greedy and SABRE)')
    parser.add_argument('--no-greedy', action='store_true', default=False,
                        help='skip the greedy baseline (keep random and SABRE)')
    parser.add_argument('--sabre-heuristic', type=str, default='decay',
                        choices=['basic', 'decay', 'lookahead'],
                        help='SABRE heuristic (default: decay)')
    parser.add_argument('--sabre-trials', type=int, default=20,
                        help='SABRE swap trials per circuit (default: 20)')
    parser.add_argument('--max-circuits', type=int, default=None,
                        help='limit number of circuits to evaluate')
    parser.add_argument('--beam-width', type=int, default=0,
                        help='beam search 宽度（1-step lookahead）；0 表示 argmax')
    parser.add_argument('--beam-vhead', type=str, default='auto',
                        choices=['auto', 'route', 'la'],
                        help='beam 打分价值头：auto=ckpt 含 critic_la 则用 la；'
                             'route=v_route+λv_fid（历史口径）；la=V_LA 多步价值头')
    parser.add_argument('--lambda-budget', type=float, default=0.0,
                        help='P1-a 推理版：SWAP 超预算后每颗惩罚（0=关，默认关=历史口径）。'
                             '使 beam 打分的 r_c 与训练口径一致')
    parser.add_argument('--budget-delta', type=float, default=1.05,
                        help='预算膨胀系数：budget = ceil(delta × SABRE swaps)')
    parser.add_argument('--use-scheduler', action='store_true', default=False,
                        help='启用门调度器（timing_aware），导出并行调度序列')
    parser.add_argument('--eta-time', type=float, default=0.01)
    parser.add_argument('--eta-xtalk-par', type=float, default=0.05)
    parser.add_argument('--eta-idle', type=float, default=0.005)
    parser.add_argument('--eta-parallel', type=float, default=0.05,
                        help='并行密度奖励系数（仅训练用，评估仅透传）')
    parser.add_argument('--swap-duration', type=float, default=0.9,
                        help='SWAP 分解时长（µs，默认 0.9 = 3×CX）')
    parser.add_argument('--xtalk-alpha', type=float, default=0.03,
                        help='串扰软约束阈值（仅调度器内部用；SABRE+调度基线同用）')
    parser.add_argument('--dump-schedule', action='store_true', default=False,
                        help='在 --out JSON 中导出带并时序的门调度序列与物理电路 QASM')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='print per-circuit results')
    parser.add_argument('--fidelity-sim', type=str, default='aer',
                        choices=['aer', 'trajectory', 'trajectory_sched', 'trajectory_v2', 'trajectory_v3', 'analytic'],
                        help='保真度模拟器: aer=density_matrix/counts (n<=12), '
                             'trajectory=轨迹状态向量(串行, O(2^n) 内存), '
                             'trajectory_sched=轨迹状态向量+调度感知(空闲退相干/动态串扰), '
                             'trajectory_v2=事件级调度感知 v2(per-gate 时长/重叠缩放串扰/always-on 可选), '
                             'analytic=解析错误累积代理(O(门数), 无指数)')
    parser.add_argument('--traj-trajectories', type=int, default=16,
                        help='轨迹模拟器采样条数')
    parser.add_argument('--traj-seed', type=int, default=None,
                        help='轨迹模拟器随机种子')
    parser.add_argument('--sim-device', type=str, default='auto',
                        help='轨迹模拟器后端（auto/cpu/cuda/cuda:N；默认 auto=CUDA 可用即 GPU，'
                             '噪声 MC 与 CPU 统计等价；--sim-device cpu 强制历史口径）')
    parser.add_argument('--edge-noise-features', action='store_true', default=False,
                        help='P0-a per-edge 噪声特征（须与训练时一致）')
    parser.add_argument('--beta-noise', type=float, default=0.0,
                        help='P0-b 噪声加权距离 β（须与训练时一致）')
    parser.add_argument('--w-err', type=float, default=0.0,
                        help='P0-c 势函数 E_err 权重（须与训练时一致）')
    parser.add_argument('--w-xt', type=float, default=0.0,
                        help='P0-c 势函数 X(s) 权重（须与训练时一致）')
    parser.add_argument('--w-xt-swap', type=float, default=0.0,
                        help='P0-c per-swap 串扰价权重（须与训练时一致）')
    parser.add_argument('--pot-progress-b', type=float, default=0.045,
                        help='P0-d progress 奖励 B（须与训练时一致）')
    parser.add_argument('--no-pot-1q-reward', dest='pot_1q_reward',
                        action='store_false', default=True,
                        help='P0-d 1Q/measure 门 progress 奖励置零（须与训练时一致）')
    parser.add_argument('--out', type=str, default=None,
                        help='save per-circuit results as JSON')
    args = parser.parse_args()

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # 小图推理：限制 torch CPU 线程数，避免多线程同步开销主导（实测 2x+）
    torch.set_num_threads(max(1, args.torch_threads))

    # ---- Hardware ----
    if args.topo:
        config, hw, coupling_map = load_topo(args.topo)
    else:
        config, hw, coupling_map = make_default_hw(args.num_qubits)

    # ---- Load circuits from split ----
    split_path = os.path.join(args.data_dir, 'splits', args.split + '.txt')
    if not os.path.exists(split_path):
        print(f'Split file not found: {split_path}')
        sys.exit(1)

    rel_paths = load_split(split_path)
    if args.max_circuits is not None and args.max_circuits < len(rel_paths):
        rel_paths = rel_paths[:args.max_circuits]
    print(f'Split: {args.split} ({len(rel_paths)} circuits)')
    print(f'Model: {args.model}')
    print()

    # ---- Probe circuit to set up agent ----
    sample_qc = load_qc(args.data_dir, rel_paths[0])
    sample_dag = CircuitDAG.from_circuit(sample_qc)

    # 探测 checkpoint 是否带 V_LA 头（训练期 beam lookahead 产物）
    _probe = torch.load(args.model, map_location='cpu', weights_only=False)
    _has_la = isinstance(_probe, dict) and any(
        k.startswith('critic_la') for k in _probe.get('ac', {}))
    if args.beam_vhead == 'auto':
        vhead = 'la' if _has_la else 'route'
    elif args.beam_vhead == 'la' and not _has_la:
        print('[warn] --beam-vhead la 但 ckpt 无 critic_la 权重，回退 route 口径')
        vhead = 'route'
    else:
        vhead = args.beam_vhead
    print(f'[vhead] beam 打分价值头 = {vhead}（ckpt critic_la: {_has_la}）')

    use_gnn = not args.no_gnn
    shared_gnn = SubGNN(subgraph='full') if use_gnn else None
    if use_gnn:
        shared_gnn.eval()

    sample_env = RoutingEnv(
        sample_dag, hw, coupling_map, reward_mode=args.reward_mode,
        max_episode_steps=args.max_episode_steps,
        random_init=False, seed=args.seed,
        gnn=shared_gnn, use_gnn=use_gnn,
        max_num_edges=args.max_num_edges,
        max_num_qubits=args.max_num_qubits,
        edge_noise_features=args.edge_noise_features,
        beta_noise=args.beta_noise,
    )

    agent_n_qubits = args.max_num_qubits or sample_dag.num_logical_qubits
    agent_n_edges = args.max_num_edges or len(coupling_map)
    agent = PPOAgent(
        obs_dim=int(np.prod(sample_env.observation_space.shape)),
        action_dim=agent_n_edges + (1 if args.mapping_phase else 0),
        device=args.device,
        gnn=shared_gnn,
        num_qubits=agent_n_qubits,
        num_edges=agent_n_edges,
        coupling_map=coupling_map,
        with_commit=args.mapping_phase,
        edge_feat_dim=(getattr(sample_env, '_edge_feat_dim', None) if use_gnn else None),
        with_la_head=_has_la,
        edge_hidden=args.edge_hidden,
    )
    agent.load(args.model)
    agent.ac.eval()
    if agent.gnn is not None:
        agent.gnn.eval()

    # ---- Evaluate ----
    def _progress(i: int, total: int, method: str):
        if args.verbose:
            print(f'  [{i+1}/{total}] {method}...', end=' ', flush=True)

    label = (f'PPO_beam{args.beam_width}_{vhead}'
             if args.beam_width > 0 else 'PPO')
    traj_seed = args.traj_seed if args.traj_seed is not None else args.seed
    fid_fn = build_fidelity_fn(args.fidelity_sim, config, args.traj_trajectories, traj_seed,
                               backend=args.sim_device) \
        if args.reward_mode != 'routing' else None

    def evaluate_agent_on_circuits():
        results = []
        for i, rel_path in enumerate(rel_paths):
            _progress(i, len(rel_paths), label)
            qc = load_qc(args.data_dir, rel_path)
            dag = CircuitDAG.from_circuit(qc)
            # P1-a 推理版：每电路 SABRE SWAP 预算（超预算每颗罚 lambda_budget，
            # 使 beam 打分的 r_c 与训练口径一致；默认关=历史行为）
            ep_budget = None
            if args.lambda_budget > 0:
                _, sinfo = sabre_route(qc, config, swap_trials=args.sabre_trials,
                                       seed=args.seed)
                s_sw = int(sinfo.get('num_swaps', 0) or 0)
                if s_sw > 0:
                    ep_budget = int(np.ceil(args.budget_delta * s_sw))
            init_mapping = None
            if args.warm_start_sabre:
                _, info = sabre_route(
                    qc, config, heuristic=args.sabre_heuristic,
                    swap_trials=args.sabre_trials, seed=args.seed,
                )
                init_mapping = info.get('initial_layout')
            random_init = args.random_init and not args.warm_start_sabre
            if args.beam_width > 0:
                m = evaluate_circuit_beam(
                    dag, hw, coupling_map, agent,
                    reward_mode=args.reward_mode,
                    max_episode_steps=args.max_episode_steps,
                    seed=args.seed + i,
                    noise_config=config if args.reward_mode != 'routing' else None,
                    beam_width=args.beam_width,
                    vhead=vhead,
                    lambda_budget=args.lambda_budget,
                    sabre_swap_budget=ep_budget,
                    max_num_qubits=args.max_num_qubits,
                    max_num_edges=args.max_num_edges,
                    random_init=random_init,
                    init_mapping=init_mapping,
                    use_scheduler=(args.use_scheduler or args.fidelity_sim in ("trajectory_sched", "trajectory_v2", "trajectory_v3")),
                    eta_time=args.eta_time,
                    eta_xtalk_par=args.eta_xtalk_par,
                    eta_idle=args.eta_idle,
                    eta_parallel=args.eta_parallel,
                    xtalk_alpha=args.xtalk_alpha,
                    swap_duration=args.swap_duration,
                    dump_schedule=args.dump_schedule,
                    fidelity_fn=fid_fn,
                    config=config,
                    fidelity_sim=args.fidelity_sim,
                    num_trajectories=args.traj_trajectories,
                    traj_seed=traj_seed,
                    edge_noise_features=args.edge_noise_features,
                    beta_noise=args.beta_noise,
                    w_err=args.w_err, w_xt=args.w_xt, w_xt_swap=args.w_xt_swap,
                    pot_progress_b=args.pot_progress_b,
                    pot_1q_reward=args.pot_1q_reward,
                )
            else:
                m = evaluate_circuit(
                    dag, hw, coupling_map, agent,
                    reward_mode=args.reward_mode,
                    max_episode_steps=args.max_episode_steps,
                    deterministic=args.deterministic,
                    seed=args.seed + i,
                    noise_config=config if args.reward_mode != 'routing' else None,
                    max_num_qubits=args.max_num_qubits,
                    max_num_edges=args.max_num_edges,
                    random_init=random_init,
                    init_mapping=init_mapping,
                    lambda_budget=args.lambda_budget,
                    sabre_swap_budget=ep_budget,
                    use_scheduler=(args.use_scheduler or args.fidelity_sim in ("trajectory_sched", "trajectory_v2", "trajectory_v3")),
                    eta_time=args.eta_time,
                    eta_xtalk_par=args.eta_xtalk_par,
                    eta_idle=args.eta_idle,
                    eta_parallel=args.eta_parallel,
                    xtalk_alpha=args.xtalk_alpha,
                    swap_duration=args.swap_duration,
                    dump_schedule=args.dump_schedule,
                    fidelity_fn=fid_fn,
                    config=config,
                    fidelity_sim=args.fidelity_sim,
                    num_trajectories=args.traj_trajectories,
                    traj_seed=traj_seed,
                    edge_noise_features=args.edge_noise_features,
                    beta_noise=args.beta_noise,
                    w_err=args.w_err, w_xt=args.w_xt, w_xt_swap=args.w_xt_swap,
                    pot_progress_b=args.pot_progress_b,
                    pot_1q_reward=args.pot_1q_reward,
                )
            m.circuit_path = rel_path
            if args.verbose:
                tag = 'OK' if m.completed else 'TRUNC'
                print(f'{tag} swaps={m.num_swaps} steps={m.episode_steps} {m.wall_time_ms:.0f}ms')
            results.append(m)
        return results

    agent_metrics = evaluate_agent_on_circuits()
    agent_stats = aggregate(agent_metrics)

    # 即便 reward_mode='routing'，只要指定了态级保真度模拟器（trajectory/
    # trajectory_sched），也展示保真度列（PPO 的来自环境 fidelity_fn，SABRE 的来自
    # baseline 的 phys_fidelity），以便做同口径对比。
    show_fid = (args.reward_mode != 'routing') or (args.fidelity_sim in ('trajectory', 'trajectory_sched', 'trajectory_v2'))
    print_header(show_fidelity=show_fid)
    print_report(label, agent_stats, show_fidelity=show_fid)

    # trunc 诊断：预算型（步数耗尽）vs 能力/游走型（no_progress 提前截断）
    _tb = sum(1 for m in agent_metrics
              if not m.completed and m.episode_steps >= args.max_episode_steps)
    _tn = sum(1 for m in agent_metrics
              if not m.completed and m.episode_steps < args.max_episode_steps)
    if _tb + _tn:
        print(f'  [trunc诊断] 预算型={_tb}  能力/游走型={_tn}')

    if args.baselines:
        # Random
        random_metrics = []
        if not args.no_random:
            for i, rel_path in enumerate(rel_paths):
                _progress(i, len(rel_paths), 'Random')
                qc = load_qc(args.data_dir, rel_path)
                dag = CircuitDAG.from_circuit(qc)
                m = evaluate_random(
                    dag, hw, coupling_map,
                    reward_mode=args.reward_mode,
                    max_episode_steps=args.max_episode_steps,
                    seed=args.seed + i + 1000,
                    noise_config=config if args.reward_mode != 'routing' else None,
                    random_init=args.random_init,
                    fidelity_fn=fid_fn,
                )
                m.circuit_path = rel_path
                if args.verbose:
                    tag = 'OK' if m.completed else 'TRUNC'
                    print(f'{tag} swaps={m.num_swaps} steps={m.episode_steps} {m.wall_time_ms:.0f}ms')
                random_metrics.append(m)
            random_stats = aggregate(random_metrics)
            print_report('Random', random_stats, show_fidelity=show_fid)

        # Greedy
        greedy_metrics = []
        if not args.no_greedy:
            for i, rel_path in enumerate(rel_paths):
                _progress(i, len(rel_paths), 'Greedy')
                qc = load_qc(args.data_dir, rel_path)
                m = evaluate_greedy(qc, config, reward_mode=args.reward_mode,
                                    fidelity_sim=args.fidelity_sim,
                                    num_trajectories=args.traj_trajectories,
                                    seed=args.seed + i + 1000)
                m.circuit_path = rel_path
                if args.verbose:
                    print(f'OK swaps={m.num_swaps} {m.wall_time_ms:.0f}ms')
                greedy_metrics.append(m)
            greedy_stats = aggregate(greedy_metrics)
            print_report('Greedy', greedy_stats, show_fidelity=show_fid)

        # SABRE
        sabre_metrics = []
        for i, rel_path in enumerate(rel_paths):
            _progress(i, len(rel_paths), 'SABRE')
            qc = load_qc(args.data_dir, rel_path)
            m = evaluate_sabre(
                qc, config,
                reward_mode=args.reward_mode,
                heuristic=args.sabre_heuristic,
                swap_trials=args.sabre_trials,
                seed=args.seed + i + 2000,
                fidelity_sim=args.fidelity_sim,
                num_trajectories=args.traj_trajectories,
                hw=hw,
                use_scheduler=args.use_scheduler,
                xtalk_alpha=args.xtalk_alpha,
            )
            m.circuit_path = rel_path
            if args.verbose:
                print(f'OK swaps={m.num_swaps} {m.wall_time_ms:.0f}ms')
            sabre_metrics.append(m)
        sabre_stats = aggregate(sabre_metrics)
        print_report('SABRE', sabre_stats, show_fidelity=show_fid)

    print()

    # ---- Save per-circuit results ----
    if args.out:
        def _asdict(m: CircuitMetrics) -> dict:
            return {
                'circuit_path': m.circuit_path,
                'completed': m.completed,
                'num_swaps': m.num_swaps,
                'mapping_swaps': m.mapping_swaps,
                'gates_executed': m.gates_executed,
                'total_gates': m.total_gates,
                'episode_steps': m.episode_steps,
                'wall_time_ms': m.wall_time_ms,
                'terminal_xz': m.terminal_xz,
                'truncated_remaining': m.truncated_remaining,
                'fidelity': m.fidelity,
                'circuit_time_us': m.circuit_time_us,
                'crosstalk_events': m.crosstalk_events,
                'schedule': m.schedule,
                'phys_qasm': m.phys_qasm,
            }

        out = {
            'args': vars(args),
            'agent': [_asdict(m) for m in agent_metrics],
        }
        if args.baselines:
            if not args.no_random:
                out['random'] = [_asdict(m) for m in random_metrics]
            if not args.no_greedy:
                out['greedy'] = [_asdict(m) for m in greedy_metrics]
            out['sabre'] = [_asdict(m) for m in sabre_metrics]
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'Results saved to {args.out}')


if __name__ == '__main__':
    main()

