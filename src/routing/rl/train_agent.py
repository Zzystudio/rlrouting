from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
import os
import pickle
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.sim import NoiseConfig
from routing.graph.circuit_dag import CircuitDAG
from routing.graph.features import HardwareFeatures
from routing.rl.env import RoutingEnv
from routing.rl.agent import PPOAgent, EdgeActorCritic
from routing.gnn.encoder import SubGNN


# ---------------------------------------------------------------------------
#  Hardware config from JSON
# ---------------------------------------------------------------------------
def _normalize_dict_keys(d):
    """Normalize string keys like "(0, 18)" (json round-trip of tuples) to tuple keys."""
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
    """Convert [[q1,q2,val],...] to {(q1,q2): val, (q2,q1): val} dict."""
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


def load_topo_or_default(num_qubits: int = 5, topo_path: str = None) -> tuple:
    if topo_path:
        return load_topo(topo_path)
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


# ---------------------------------------------------------------------------
#  Dataset loader
# ---------------------------------------------------------------------------
def load_split(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def build_split_paths(data_dir: str, prefix: str = "stage1") -> dict:
    """Split manifests; `prefix` allows per-scale datasets (e.g. large_n10)."""
    return {
        f"{prefix}_phase1": os.path.join(data_dir, "splits", f"{prefix}_phase1.txt"),
        f"{prefix}_phase2": os.path.join(data_dir, "splits", f"{prefix}_phase2.txt"),
        f"{prefix}_phase3": os.path.join(data_dir, "splits", f"{prefix}_phase3.txt"),
        f"{prefix}_mixed": os.path.join(data_dir, "splits", f"{prefix}_mixed.txt"),
        f"{prefix}_alg": os.path.join(data_dir, "splits", f"{prefix}_alg.txt"),
    }


def build_multi_split_map(data_dir: str, prefixes: list[str]) -> dict:
    """合并多个 prefix 的 split manifest，key 为完整 split 名（如 large_n10_phase1）。"""
    result = {}
    for prefix in prefixes:
        result.update(build_split_paths(data_dir, prefix))
    return result


def pick_circuit(data_dir: str, split_name: str, seed: Optional[int] = None, split_prefix: str = "stage1",
                 split_map: Optional[dict] = None, max_qubits: Optional[int] = None):
    """Pick a random circuit from a split; optionally filter by logical-qubit count
    (must fit the current topology's physical qubits when multi-topo training)."""
    if split_map is None:
        split_map = build_split_paths(data_dir, split_prefix)
    split_path = split_map[split_name]
    paths = load_split(split_path)
    if max_qubits is not None:
        paths = [p for p in paths if _circuit_qubits(p) <= max_qubits]
        if not paths:
            raise ValueError(f"split {split_name}: no circuit with <= {max_qubits} qubits")
    path = random.choice(paths)
    with open(os.path.join(data_dir, path), "rb") as f:
        qc = pickle.load(f)
    if qc.num_parameters > 0:
        rng = np.random.default_rng(seed)
        param_dict = {p: rng.uniform(0, 2 * np.pi) for p in qc.parameters}
        qc = qc.assign_parameters(param_dict)
    return CircuitDAG.from_circuit(qc)


def _circuit_qubits(path: str) -> int:
    """Extract logical-qubit count from a pkl path without loading the circuit."""
    import re
    name = os.path.basename(path).removesuffix(".pkl")
    m = re.search(r"_n(\d+)", name)
    return int(m.group(1)) if m else 0


def pick_circuit_with_path(data_dir: str, split_name: str, seed: Optional[int] = None, split_prefix: str = "stage1",
                           split_map: Optional[dict] = None, max_qubits: Optional[int] = None):
    """Like pick_circuit but returns (dag, rel_path) so the caller can look up
    a precomputed SABRE initial-layout cache keyed by rel_path."""
    if split_map is None:
        split_map = build_split_paths(data_dir, split_prefix)
    split_path = split_map[split_name]
    paths = load_split(split_path)
    if max_qubits is not None:
        paths = [p for p in paths if _circuit_qubits(p) <= max_qubits]
        if not paths:
            raise ValueError(f"split {split_name}: no circuit with <= {max_qubits} qubits")
    path = random.choice(paths)
    with open(os.path.join(data_dir, path), "rb") as f:
        qc = pickle.load(f)
    if qc.num_parameters > 0:
        rng = np.random.default_rng(seed)
        param_dict = {p: rng.uniform(0, 2 * np.pi) for p in qc.parameters}
        qc = qc.assign_parameters(param_dict)
    return CircuitDAG.from_circuit(qc), path


def load_nam_circuits(nam_dir: str, max_qubits: int = 20):
    """加载目录下所有 QASM 文件为 CircuitDAG 列表（过滤 > max_qubits 的电路）。

    返回 [(dag, name), ...]，name 为不带扩展名的文件名。
    """
    from qiskit.qasm2 import load as qasm2_load
    dags = []
    if not os.path.isdir(nam_dir):
        return dags
    for fname in sorted(os.listdir(nam_dir)):
        if not fname.endswith(".qasm"):
            continue
        fpath = os.path.join(nam_dir, fname)
        try:
            qc = qasm2_load(fpath)
            if qc.num_qubits > max_qubits:
                continue
            dag = CircuitDAG.from_circuit(qc)
            name = fname.removesuffix(".qasm")
            dags.append((dag, name))
        except Exception as e:
            print(f"[nam-circuits] 跳过 {fname}: {e}")
    return dags


def pick_circuit_with_nam(args, split_key, split_prefix, split_map, total_steps, topo_qubits, topo_idx,
                          nam_circuits, nam_prob):
    """带 NAM 电路混入的电路采样器。

    以 nam_prob 概率从 nam_circuits 中均匀选择，否则走原 pick_circuit_with_path。
    NAM 电路取不超过当前拓扑容量者（额外受 nam_max_q 上限约束，保证 ≤16q 训练
    时不用 trajectory_sched 跑大电路）。
    """
    if nam_circuits and random.random() < nam_prob:
        nam_max_q = args.nam_max_qubits or args.max_num_qubits or topo_qubits[topo_idx]
        cap = min(topo_qubits[topo_idx], nam_max_q)
        # 从 NAM 电路中均匀选择，找到能放进当前拓扑的
        candidates = [(d, n) for d, n in nam_circuits if d.num_logical_qubits <= cap]
        if candidates:
            dag, name = random.choice(candidates)
            return dag, f"nam/{name}"
    return pick_circuit_with_path(args.data_dir, split_key, seed=args.seed + total_steps,
                                  split_prefix=split_prefix, split_map=split_map,
                                  max_qubits=topo_qubits[topo_idx])


def _build_sabre_layout_cache(args, topo_list) -> dict:
    """预计算训练池中每个电路在各拓扑下的 SABRE 初始布局。

    返回 {topo_idx: {rel_path: [phys_idx, ...]}}。布局是免费的初始映射
    （虚拟重标号），用于训练期 'sabre' 布局混合，迫使策略学习布局无关路由。
    """
    from routing.routing import sabre_route

    prefix = args.split_prefix or {
        "routing": "stage1",
        "noise_aware": "stage2",
        "fidelity_shaping": "stage3",
    }[args.reward_mode]
    split_names = [f"{prefix}_phase1", f"{prefix}_phase2", f"{prefix}_phase3"]

    # 缓存可复用：从文件加载或构建后保存
    if args.sabre_cache_file and os.path.exists(args.sabre_cache_file):
        with open(args.sabre_cache_file, "rb") as f:
            print(f"[sabre-cache] 复用缓存 {args.sabre_cache_file}")
            return pickle.load(f)

    rels = set()
    for sn in split_names:
        sp = os.path.join(args.data_dir, "splits", f"{sn}.txt")
        if os.path.exists(sp):
            rels.update(load_split(sp))

    cache: dict = {}
    for topo_idx, (config, _cm) in enumerate(topo_list):
        cache[topo_idx] = {}
        for rel in sorted(rels):
            full = os.path.join(args.data_dir, rel)
            if not os.path.exists(full):
                continue
            try:
                with open(full, "rb") as f:
                    qc = pickle.load(f)
                _, info = sabre_route(
                    qc, config, heuristic="decay",
                    swap_trials=args.sabre_layout_trials, seed=args.seed,
                )
                cache[topo_idx][rel] = info.get("initial_layout")
            except Exception as e:
                print(f"[sabre-cache] 跳过 {rel} (topo{topo_idx}): {e}")

    if args.sabre_cache_file:
        os.makedirs(os.path.dirname(args.sabre_cache_file) or ".", exist_ok=True)
        with open(args.sabre_cache_file, "wb") as f:
            pickle.dump(cache, f)
        print(f"[sabre-cache] 保存 {len(rels)} 条缓存 -> {args.sabre_cache_file}")
    return cache


# ---------------------------------------------------------------------------
#  Curriculum phase for Stage 1 — smooth overlap
# ---------------------------------------------------------------------------
def stage1_phase(progress: float, prefix: str = "stage1") -> str:
    if progress < 0.30:
        return f"{prefix}_phase1"
    if progress < 0.40:
        p2 = (progress - 0.30) / 0.10
        if random.random() < p2:
            return f"{prefix}_phase2"
        return f"{prefix}_phase1"
    if progress < 0.60:
        return f"{prefix}_phase2"
    if progress < 0.70:
        p3 = (progress - 0.60) / 0.10
        if random.random() < p3:
            return f"{prefix}_phase3"
        return f"{prefix}_phase2"
    return f"{prefix}_phase3"


def reward_mode_split(reward_mode: str, prefix: str) -> tuple:
    mapping = {
        "routing": (f"{prefix}_phase1", lambda p: stage1_phase(p, prefix)),
        "noise_aware": (f"{prefix}_mixed", lambda _: f"{prefix}_mixed"),
        "fidelity_shaping": (f"{prefix}_alg", lambda _: f"{prefix}_alg"),
    }
    return mapping[reward_mode]


def curriculum_phase(progress: float, prefixes: list[str], reward_mode: str = "routing") -> str:
    """跨规模课程：按训练进度切换到不同 split prefix（如 large_n10 → n20 → tianyan），
    每个 prefix 内部复用 stage1_phase 的深度递进。"""
    n = len(prefixes)
    idx = min(int(progress * n), n - 1)
    prefix = prefixes[idx]
    local_p = progress * n - idx
    if reward_mode == "routing":
        return stage1_phase(local_p, prefix)
    if reward_mode == "noise_aware":
        return f"{prefix}_mixed"
    return f"{prefix}_alg"


def lambda_fid_schedule(progress: float, warmup: float, max_val: float) -> float:
    """λ_fid 退火调度。

    warmup=0.0 表示立即满权重（λ_fid == max_val），避免课程早期（5q/8q，
    保真度信号最强）无保真度反馈的错相问题；warmup>0 时为前 warmup 比例为 0，
    之后线性增长到 max_val。
    """
    if warmup <= 0.0:
        return max_val
    if progress < warmup:
        return 0.0
    return max_val * (progress - warmup) / (1.0 - warmup)


def adaptive_lambda_fid_max(progress: float, schedule_str: str, default: float) -> float:
    """按阶段自适应 λ_fid_max 调度。

    schedule_str 为逗号分隔的值列表，如 "5,5,5,5,5,10"。
    按训练进度等分阶段，返回当前阶段对应的 λ_fid_max。
    """
    if not schedule_str:
        return default
    vals = [float(v) for v in schedule_str.split(",") if v.strip()]
    if not vals:
        return default
    n = len(vals)
    idx = min(int(progress * n), n - 1)
    return vals[idx]


def _eval_ema(agent, eval_circuits, args, gnn, use_gnn, max_edges, topo_list):
    """用 EMA 权重在 test split 上评估平均 fidelity。"""
    import torch
    from .eval_policy import evaluate_circuit, load_qc
    from ..graph.circuit_dag import CircuitDAG
    from ..graph.features import HardwareFeatures

    agent.apply_ema()
    agent.ac.eval()
    if use_gnn:
        agent.gnn.eval()

    noise_config, coupling_map = topo_list[0]
    hw = HardwareFeatures.from_noise_config(noise_config)

    fids = []
    eval_list = eval_circuits[:args.eval_max_circuits] if args.eval_max_circuits else eval_circuits
    for cp in eval_list:
        try:
            qc = load_qc(args.data_dir, cp)
            dag = CircuitDAG.from_circuit(qc)
            if dag.num_logical_qubits > args.eval_max_qubits:
                continue
            fidelity_fn = build_fidelity_fn(
                args.fidelity_sim, noise_config,
                num_trajectories=args.eval_traj, seed=0
            )
            m = evaluate_circuit(
                dag, hw, coupling_map, agent,
                reward_mode=args.reward_mode,
                max_episode_steps=args.max_episode_steps,
                deterministic=True, seed=0,
                noise_config=noise_config if args.reward_mode != "routing" else None,
                max_num_qubits=args.eval_max_qubits,
                use_scheduler=(args.use_scheduler or args.fidelity_sim == "trajectory_sched"),
                eta_time=args.eta_time,
                eta_xtalk_par=args.eta_xtalk_par,
                eta_idle=args.eta_idle,
                eta_parallel=args.eta_parallel,
                xtalk_alpha=args.xtalk_alpha,
                swap_duration=args.swap_duration,
                fidelity_fn=fidelity_fn,
                fidelity_sim=args.fidelity_sim,
                num_trajectories=args.eval_traj,
            )
            if m.fidelity is not None:
                fids.append(m.fidelity)
        except Exception as e:
            import traceback
            traceback.print_exc()
            break

    agent.restore_from_ema()
    agent.ac.train()
    if use_gnn:
        agent.gnn.train()

    return np.mean(fids) if fids else 0.0


# ---------------------------------------------------------------------------
#  create_env helper
# ---------------------------------------------------------------------------
def _parse_sabre_fid_map(spec: Optional[str]) -> dict:
    """解析 --sabre-fid-map 字符串为 {size: fid}，缺省用内置常数。"""
    defaults = {5: 0.309, 8: 0.162, 10: 0.061, 12: 0.046, 16: 0.0065}
    if not spec:
        return defaults
    out: dict = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=")
        out[int(k.strip())] = float(v.strip())
    return out


def create_env(dag, hw, coupling_map, reward_mode, max_episode_steps, random_init, seed, gnn=None, use_gnn=True, max_num_edges=None, max_num_qubits=None, noise_config=None, lambda_fid=None, eta_dist=None, mapping_budget=None, mapping_phase=True, fidelity_fn=None, use_scheduler=None, eta_time=None, eta_xtalk_par=None, eta_idle=None, eta_parallel=None, xtalk_alpha=None, swap_duration=None, swap_cost=None, init_mapping=None, lambda_layout=None, sabre_fid_map=None, sref_override=None):
    kw = dict(
        dag=dag, hw=hw, coupling_map=coupling_map,
        reward_mode=reward_mode,
        max_episode_steps=max_episode_steps,
        random_init=random_init,
        seed=seed,
    )
    if gnn is not None:
        kw["gnn"] = gnn
    if not use_gnn:
        kw["use_gnn"] = False
    if max_num_edges is not None:
        kw["max_num_edges"] = max_num_edges
    if max_num_qubits is not None:
        kw["max_num_qubits"] = max_num_qubits
    if noise_config is not None:
        kw["noise_config"] = noise_config
    if lambda_fid is not None:
        kw["lambda_fid"] = lambda_fid
    if eta_dist is not None:
        kw["eta_dist"] = eta_dist
    if mapping_budget is not None:
        kw["mapping_budget"] = mapping_budget
    if fidelity_fn is not None:
        kw["fidelity_fn"] = fidelity_fn
    if use_scheduler is not None:
        kw["use_scheduler"] = use_scheduler
    if eta_time is not None:
        kw["eta_time"] = eta_time
    if eta_xtalk_par is not None:
        kw["eta_xtalk_par"] = eta_xtalk_par
    if eta_idle is not None:
        kw["eta_idle"] = eta_idle
    if eta_parallel is not None:
        kw["eta_parallel"] = eta_parallel
    if xtalk_alpha is not None:
        kw["xtalk_alpha"] = xtalk_alpha
    if swap_duration is not None:
        kw["swap_duration_us"] = swap_duration
    if swap_cost is not None:
        kw["swap_cost"] = swap_cost
    if init_mapping is not None:
        kw["init_mapping"] = init_mapping
    if lambda_layout is not None:
        kw["lambda_layout"] = lambda_layout
    if sabre_fid_map is not None:
        kw["sabre_fid_map"] = sabre_fid_map
    if sref_override is not None:
        kw["sref_override"] = sref_override
    kw["mapping_phase"] = mapping_phase
    return RoutingEnv(**kw)


def build_fidelity_fn(fidelity_sim: str, noise_config, num_trajectories: int = 64, seed=None,
                      analytic_thermal: bool = True, analytic_crosstalk: bool = False):
    """按 --fidelity-sim 构造 env 终端保真度函数；routing 模式或 aer 模式返回 None。

    aer: 使用 env 内置 NoiseSimulator（density_matrix + counts overlap，n<=12）。
    trajectory: 使用轨迹状态向量模拟器（O(2^n) 内存，20q+ 可用）。
    analytic: 解析错误累积代理（O(门数)，无指数，适用于 16q+ 大电路训练）。
    """
    if fidelity_sim == "trajectory":
        from sim.trajectory_sim import make_trajectory_fidelity_fn
        return make_trajectory_fidelity_fn(noise_config, num_trajectories=num_trajectories, seed=seed)
    if fidelity_sim == "trajectory_sched":
        from sim.trajectory_sim import make_trajectory_fidelity_fn
        return make_trajectory_fidelity_fn(noise_config, num_trajectories=num_trajectories,
                                           seed=seed, scheduled=True)
    if fidelity_sim == "analytic":
        from sim.trajectory_sim import make_analytic_fidelity_fn
        return make_analytic_fidelity_fn(noise_config, include_thermal=analytic_thermal,
                                         include_crosstalk=analytic_crosstalk)
    return None


# ---------------------------------------------------------------------------
#  Noise perturbation
# ---------------------------------------------------------------------------

def perturb_noise_config(
    config: NoiseConfig,
    frac_t1t2: float = 0.05,
    frac_gate: float = 0.15,
    frac_other: float = 0.10,
) -> NoiseConfig:
    """深拷贝并分组扰动 NoiseConfig：T1/T2 用较小扰动，gate errors 用较大扰动。"""
    if max(frac_t1t2, frac_gate, frac_other) <= 0:
        return copy.deepcopy(config)
    cfg = copy.deepcopy(config)
    rng = np.random

    def _p(v, f):
        return v * rng.uniform(1 - f, 1 + f)

    cfg.t1_times = [_p(t, frac_t1t2) for t in cfg.t1_times]
    cfg.t2_times = [min(t1, _p(t2, frac_t1t2)) for t1, t2 in zip(cfg.t1_times, cfg.t2_times)]
    if cfg.freq_ghz:
        cfg.freq_ghz = [_p(f, frac_other) for f in cfg.freq_ghz]
    if cfg.readout_error:
        cfg.readout_error = [_p(r, frac_other) for r in cfg.readout_error]

    sqe = cfg.single_q_gate_error
    if isinstance(sqe, (list, tuple)):
        cfg.single_q_gate_error = [_p(v, frac_gate) for v in sqe]
    else:
        cfg.single_q_gate_error = _p(sqe, frac_gate)

    tqe = cfg.two_q_gate_error
    if isinstance(tqe, dict):
        cfg.two_q_gate_error = {k: _p(v, frac_gate) for k, v in tqe.items()}
    else:
        cfg.two_q_gate_error = _p(tqe, frac_gate) if tqe > 0 else 0.0

    cs = cfg.crosstalk_strength
    if isinstance(cs, dict):
        cfg.crosstalk_strength = {k: _p(v, frac_gate) for k, v in cs.items()}
    return cfg


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train routing policy with PPO")
    parser.add_argument("--data-dir", type=str, default="../traindata",
                        help="数据集根目录")
    parser.add_argument("--split-prefix", type=str, default=None,
                        help="split 文件名前缀，用于按规模区分数据集 "
                             "(如 large_n10；默认按 reward-mode: "
                             "routing->stage1, noise_aware->stage2, fidelity_shaping->stage3)")
    parser.add_argument("--topo", type=str, default=None,
                        help="硬件拓扑 JSON 路径（默认使用线性链）")
    parser.add_argument("--topo-list", type=str, default=None,
                        help="逗号分隔的拓扑 JSON 路径列表，每 episode 随机切换")
    parser.add_argument("--topo-balance", type=str, default="episodes",
                        choices=["episodes", "steps"],
                        help="拓扑采样方式: episodes=等概率, steps=按步数加权均衡")
    parser.add_argument("--out", type=str, default="../models/policy.pt")
    parser.add_argument("--num-qubits", type=int, default=5,
                        help="硬件比特数（仅无 --topo 时生效）")
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="训练设备 (cpu / cuda:N)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load", type=str, default=None,
                        help="加载预训练模型")
    parser.add_argument("--reward-mode", type=str, default="routing",
                        choices=["routing", "noise_aware", "fidelity_shaping"],
                        help="奖励模式")
    parser.add_argument("--max-episode-steps", type=int, default=200,
                        help="每个 episode 的最大步数（超时截断）")
    parser.add_argument("--random-init", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="是否随机化初始映射")
    parser.add_argument("--no-gnn", action="store_true", default=False,
                        help="禁用 GNN 编码器（不使用 GNN+PPO 联合训练）")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="checkpoint 保存目录（默认 --out 同目录下的 ckpts/）")
    parser.add_argument("--checkpoint-interval", type=int, default=10,
                        help="每 N 个 update cycle 保存一次 checkpoint")
    parser.add_argument("--vf-coef", type=float, default=0.1,
                        help="value loss 权重（默认 0.1）")
    parser.add_argument("--lambda-fid-max", type=float, default=5.0,
                        help="终端保真度奖励最大权重（仅 noise_aware/fidelity_shaping）")
    parser.add_argument("--lambda-fid-warmup", type=float, default=0.0,
                        help="lambda_fid 退火比例：前 N%% 步 lambda_fid=0")
    parser.add_argument("--sabre-fid-map", type=str, default=None,
                        help="每尺寸 SABRE 参考保真度（log-相对奖励分母），格式 "
                             "5=0.309,8=0.162,10=0.061,12=0.046,16=0.0065；"
                             "缺省时用内置常数")
    parser.add_argument("--sabre-fid-map-nam", type=str, default=None,
                        help="NAM 电路专用 SABRE 参考保真度表（覆盖 --sabre-fid-map，"
                             "因 NAM 与随机电路同尺寸 SABRE 基线不同），格式同上")
    parser.add_argument("--clip-return", type=float, default=50.0,
                        help="GAE return 裁剪阈值 (0=不裁剪，默认 50.0)")
    parser.add_argument("--noise-perturb", type=float, default=0.15,
                        help="gate errors 扰动幅度（默认 0.15）")
    parser.add_argument("--noise-perturb-t1t2", type=float, default=0.05,
                        help="T1/T2 时间扰动幅度（默认 0.05）")
    parser.add_argument("--noise-perturb-other", type=float, default=0.10,
                        help="freq/readout 等其它参数扰动幅度（默认 0.10）")
    parser.add_argument("--eta-dist", type=float, default=1.0,
                        help="距离减少奖励系数（0=禁用，默认 1.0）")
    parser.add_argument("--max-num-qubits", type=int, default=None,
                        help="统一观测 qubit 维度（多规模混训时设为最大 n，默认=线路自身 n）")
    parser.add_argument("--mapping-phase", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="启用映射阶段（虚拟 SWAP 学初始布局 + commit 动作）")
    parser.add_argument("--mapping-budget", type=int, default=None,
                        help="映射阶段最大虚拟 SWAP 次数（默认 n-1）")
    parser.add_argument("--mapping-min-swaps", type=int, default=0,
                        help="训练时强制每 episode 至少 N 次虚拟 SWAP 才能 commit "
                             "(0=不强制；建议 2-3 让 agent 学习布局质量)")
    parser.add_argument("--use-scheduler", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="启用门调度插件（timing_aware）：每步 SWAP 后对 front-layer "
                             "贪心并行调度，并叠加 r_time / r_xtalk_par / r_idle 奖励")
    parser.add_argument("--eta-time", type=float, default=0.01,
                        help="每 two_gate_time 单位的时间惩罚（调度奖励，默认 0.01）")
    parser.add_argument("--eta-xtalk-par", type=float, default=1.0,
                        help="每对并行串扰（hw.zz 加权）惩罚（默认 0.05）")
    parser.add_argument("--eta-idle", type=float, default=0.005,
                        help="每 qubit·µs 空闲退相干惩罚（默认 0.005）")
    parser.add_argument("--eta-parallel", type=float, default=0.05,
                        help="并行密度奖励 = eta_parallel·max(0, serial_dur/clock-1)（默认 0.05）")
    parser.add_argument("--xtalk-alpha", type=float, default=0.03,
                        help="串扰软约束阈值（zz>alpha 则延迟冲突门；0=关闭，默认 0.03）")
    parser.add_argument("--swap-duration", type=float, default=0.9,
                        help="duration of a SWAP gate (us) in scheduler timing")
    parser.add_argument("--swap-cost", type=float, default=0.5,
                        help="per-SWAP direct penalty (eta_swap); 0 disables the dead-code swap_cost")
    parser.add_argument("--layout-mix", type=str, default=None,
                        help="布局混合比例 恒等/随机/SABRE，逗号分隔如 0.3,0.3,0.4（None=关，即现行为）")
    parser.add_argument("--lambda-layout", type=float, default=0.0,
                        help="commit 时布局质量终端奖励权重（= -λ·平均front-layer距离）")
    parser.add_argument("--sabre-layout-trials", type=int, default=5,
                        help="训练缓存 SABRE 初始布局的 trials（默认 5，省时）")
    parser.add_argument("--sabre-cache-file", type=str, default=None,
                        help="SABRE 布局缓存 pkl 路径（存在则复用，否则构建后保存）")
    parser.add_argument("--freeze-first-steps", type=int, default=0,
                        help="训练前 N 步冻结 GNN+actor/critic 参数（只收集数据不更新，用于固定小电路阶段的路由能力）")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA 衰减系数（默认 0.999）")
    parser.add_argument("--eval-interval", type=int, default=0,
                        help="每 N 步用 EMA 权重在 test split 上评估（0=禁用）")
    parser.add_argument("--eval-split", type=str, default=None,
                        help="EMA 评估用的 test split 路径（逗号分隔多个，如 stage1_test,large_n8_test,...）")
    parser.add_argument("--eval-max-qubits", type=int, default=20,
                        help="EMA 评估电路的最大 qubit 数")
    parser.add_argument("--eval-traj", type=int, default=16,
                        help="EMA 评估的轨迹数（默认 16）")
    parser.add_argument("--eval-max-circuits", type=int, default=None,
                        help="EMA 评估的最大电路数（默认全部）")
    parser.add_argument("--lambda-fid-max-schedule", type=str, default=None,
                        help="λ_fid_max 按阶段自适应调度（逗号分隔，如 5,5,5,5,5,10 表示最后阶段增大到 10）")
    parser.add_argument("--gae-adaptive", action="store_true", default=False,
                        help="启用自适应 GAE λ：随 episode 进度线性增长")
    parser.add_argument("--gae-lam-min", type=float, default=0.95,
                        help="自适应 λ 下限（episode 开头，默认 0.95）")
    parser.add_argument("--gae-lam-max", type=float, default=0.995,
                        help="自适应 λ 上限（episode 末尾，默认 0.995）")
    parser.add_argument("--curriculum-keys", type=str, default=None,
                        help="逗号分隔的 split prefix 列表，按训练进度从小规模到大规模递进 "
                             "(如 large_n10,large_n20,tianyan；默认单 prefix)")
    parser.add_argument("--fidelity-sim", type=str, default="aer",
                        choices=["aer", "trajectory", "trajectory_sched", "analytic"],
                        help="终端保真度模拟器: aer=density_matrix/counts (小比特数), "
                             "trajectory=轨迹状态向量(串行, O(2^n) 内存), "
                             "trajectory_sched=轨迹状态向量+调度感知(空闲退相干/动态串扰, 需 --use-scheduler), "
                             "analytic=解析错误累积代理(O(门数), 无指数, 16q+ 大电路训练)")
    parser.add_argument("--traj-trajectories", type=int, default=16,
                        help="轨迹模拟器采样条数（越大方差越小，训练越慢）")
    parser.add_argument("--traj-seed", type=int, default=None,
                        help="轨迹模拟器随机种子（默认 None=不可复现）")
    parser.add_argument("--analytic-thermal", action="store_true", default=True,
                        help="解析保真度代理包含热弛豫（idle 退相干）项")
    parser.add_argument("--no-analytic-thermal", dest="analytic_thermal", action="store_false",
                        help="解析保真度代理不包含热弛豫项（纯退极化）")
    parser.add_argument("--analytic-crosstalk", action="store_true", default=False,
                        help="解析保真度代理包含串扰 θ² 惩罚项")
    parser.add_argument("--nam-circuits-dir", type=str, default=None,
                        help="NAM 电路 QASM 目录（如 benchmark/nam_circs/），训练时混入 NAM 算术电路")
    parser.add_argument("--nam-circuit-prob", type=float, default=0.3,
                        help="训练时选择 NAM 电路的概率（默认 0.3）")
    parser.add_argument("--nam-max-qubits", type=int, default=None,
                        help="NAM 电路最大逻辑比特数（过滤较大 NAM 电路；默认=--max-num-qubits）")
    parser.add_argument("--nam-sabre-fid-json", type=str, default=None,
                        help="NAM per-circuit SABRE 参考保真度 JSON（{circuit_name: fid}，"
                             "如 benchmark/routed/sabre_nam_percircuit.json）。提供时 NAM 电路用"
                             "自己电路的 SABRE 基线做 log-相对奖励分母，而非按 num_qubits 共享")
    args = parser.parse_args()

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    # Hardware — 支持单拓扑或多拓扑
    topo_list = []
    if args.topo_list:
        for path in args.topo_list.split(","):
            path = path.strip()
            config_i, _, cm_i = load_topo(path)
            topo_list.append((config_i, cm_i))
    elif args.topo:
        config_i, _, cm_i = load_topo(args.topo)
        topo_list = [(config_i, cm_i)]
    else:
        config_i, _, cm_i = load_topo_or_default(args.num_qubits)
        topo_list = [(config_i, cm_i)]
    num_topos = len(topo_list)
    max_edges = max(len(cm) for _, cm in topo_list)
    topo_qubits = []
    for _, cm in topo_list:
        qs = [q for e in cm for q in e]
        topo_qubits.append(max(qs) + 1 if qs else 0)
    topo_names = []
    if args.topo_list:
        for path in args.topo_list.split(","):
            path = path.strip()
            topo_names.append(os.path.splitext(os.path.basename(path))[0])
    else:
        topo_names = [f"topo{i}" for i in range(max(1, num_topos))]

    # Resume: 从上一 checkpoint 恢复 step 计数与课程进度
    resume_step = 0
    resume_best = -1.0
    if args.load:
        try:
            _st = torch.load(args.load, map_location="cpu", weights_only=False)
            resume_step = int(_st.get("step", 0))
            resume_best = float(_st.get("best_metric", -1.0))
            print(f"[resume] {args.load}: step={resume_step} best_metric={resume_best:.5f}")
        except Exception as e:
            print(f"[resume] 无法读取 {args.load} 的 step/best_metric（普通权重或旧格式）：{e}")
    resume_progress = resume_step / args.timesteps if args.timesteps else 0.0

    # Dataset
    split_prefix = args.split_prefix or {
        "routing": "stage1",
        "noise_aware": "stage2",
        "fidelity_shaping": "stage3",
    }[args.reward_mode]
    if args.curriculum_keys:
        curriculum_prefixes = [p.strip() for p in args.curriculum_keys.split(",") if p.strip()]
        split_map = build_multi_split_map(args.data_dir, curriculum_prefixes)
        initial_split_key = curriculum_phase(resume_progress, curriculum_prefixes, args.reward_mode)
        phase_fn = lambda p: curriculum_phase(p, curriculum_prefixes, args.reward_mode)
        print(f"Curriculum prefixes: {curriculum_prefixes}  "
              f"(initial split: {initial_split_key})")
    else:
        split_map = build_split_paths(args.data_dir, split_prefix)
        initial_split_key, phase_fn = reward_mode_split(args.reward_mode, split_prefix)
        print(f"Split prefix: {split_prefix}  (initial split: {initial_split_key})")

    # 布局混合缓存：训练前预计算 SABRE 初始布局（免费初始映射）
    mix = None
    sabre_cache = {}
    if args.layout_mix:
        mix = [float(x) for x in args.layout_mix.split(",")]
        if len(mix) != 3:
            raise SystemExit("--layout-mix 需 3 个比例 (恒等,随机,SABRE)")
        sabre_cache = _build_sabre_layout_cache(args, topo_list)

    # NAM 电路加载
    nam_circuits = []
    nam_sref_map = {}
    if args.nam_circuits_dir:
        nam_max_q = args.nam_max_qubits or args.max_num_qubits or 20
        nam_circuits = load_nam_circuits(args.nam_circuits_dir, max_qubits=nam_max_q)
        if nam_circuits:
            print(f"[nam-circuits] 加载 {len(nam_circuits)} 个 NAM 电路（≤{nam_max_q}q，"
                  f"概率 {args.nam_circuit_prob:.0%}）")
        else:
            print(f"[nam-circuits] 警告：{args.nam_circuits_dir} 无有效 QASM 文件")
        # per-circuit SABRE 参考（方案1：每条 NAM 用自己电路的 SABRE 基线）
        if args.nam_sabre_fid_json:
            try:
                import json as _json
                raw = _json.load(open(args.nam_sabre_fid_json))
                for k, v in raw.items():
                    key = k.removesuffix(".qasm")
                    nam_sref_map[key] = float(v)
                print(f"[nam-sabre] per-circuit SABRE 参考：{len(nam_sref_map)} 条")
            except Exception as e:
                print(f"[nam-sabre] 警告：加载 {args.nam_sabre_fid_json} 失败: {e}")
                nam_sref_map = {}

    use_gnn = not args.no_gnn
    sample_dag = pick_circuit(args.data_dir, initial_split_key, seed=args.seed,
                              split_prefix=split_prefix, split_map=split_map)

    if use_gnn:
        shared_gnn = SubGNN(subgraph="full")
        shared_gnn.train()
    else:
        shared_gnn = None

    # 初始拓扑
    topo_idx = 0
    noise_config, coupling_map = topo_list[topo_idx]
    hw = HardwareFeatures.from_noise_config(noise_config)

    env = create_env(sample_dag, hw, coupling_map, args.reward_mode,
                     args.max_episode_steps, args.random_init, args.seed,
                     gnn=shared_gnn, use_gnn=use_gnn,
                     max_num_edges=max_edges,
                     max_num_qubits=args.max_num_qubits,
                     noise_config=noise_config if args.reward_mode != "routing" else None,
                     lambda_fid=0.0 if args.reward_mode != "routing" else None,
                      eta_dist=args.eta_dist,
                      mapping_budget=args.mapping_budget,
                      mapping_phase=args.mapping_phase,
                       use_scheduler=(args.use_scheduler or args.fidelity_sim == "trajectory_sched"),
                       eta_time=args.eta_time,
                       eta_xtalk_par=args.eta_xtalk_par,
                       eta_idle=args.eta_idle,
                       eta_parallel=args.eta_parallel,
                                    xtalk_alpha=args.xtalk_alpha,
                                     swap_duration=args.swap_duration,
                                     swap_cost=args.swap_cost,
                                     init_mapping=None,
                                     lambda_layout=args.lambda_layout,
                        fidelity_fn=build_fidelity_fn(
                          args.fidelity_sim, noise_config,
                          num_trajectories=args.traj_trajectories, seed=args.traj_seed,
                          analytic_thermal=args.analytic_thermal,
                          analytic_crosstalk=args.analytic_crosstalk,
                      ) if args.reward_mode != "routing" else None)

    agent_n_qubits = args.max_num_qubits or sample_dag.num_logical_qubits
    agent_action_dim = max_edges + (1 if args.mapping_phase else 0)
    agent = PPOAgent(
        obs_dim=int(np.prod(env.observation_space.shape)),
        action_dim=agent_action_dim,
        lr=args.lr,
        device=args.device,
        gnn=shared_gnn,
        num_qubits=agent_n_qubits,
        num_edges=max_edges,
        coupling_map=coupling_map,
        vf_coef=args.vf_coef,
        with_commit=args.mapping_phase,
    )
    if args.load:
        state = agent.load_checkpoint(args.load)
        print(f"Loaded pretrained model: {args.load}")
        if "step" in state:
            resume_step = int(state["step"])
            resume_best = float(state.get("best_metric", -1.0))
            print(f"  -> resume at step {resume_step}, best_metric={resume_best:.5f}")

    # EMA 初始化
    if args.ema_decay > 0:
        agent.init_ema(decay=args.ema_decay)
        print(f"EMA enabled: decay={args.ema_decay}")

    # 加载 EMA 评估用的 test split 电路
    _eval_circuits = []
    if args.eval_split and args.eval_interval > 0:
        for sp in args.eval_split.split(","):
            sp = sp.strip()
            if sp and os.path.exists(sp):
                with open(sp) as f:
                    _eval_circuits.extend([l.strip() for l in f if l.strip()])
            elif sp:
                sp_path = os.path.join(args.data_dir, "splits", f"{sp}.txt")
                if os.path.exists(sp_path):
                    with open(sp_path) as f:
                        _eval_circuits.extend([l.strip() for l in f if l.strip()])
        print(f"EMA eval: {_eval_circuits.__len__()} circuits from {args.eval_split}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # --- checkpoint 初始化 ---
    ckpt_dir = args.checkpoint_dir or os.path.join(os.path.dirname(args.out) or ".", "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    _metrics_path = os.path.join(ckpt_dir, "metrics.csv")
    _metrics_fields = ["step", "reward", "swaps", "map_swaps", "trunc_pct", "pl", "vl", "ent", "kl", "grad", "fid", "time_us", "xtalk", "idle", "par"]
    # 续训时追加而非覆盖；已有行丢到 resume_step 为止，避免旧行与新续训混合
    if resume_step > 0 and os.path.exists(_metrics_path):
        _metrics_fh = open(_metrics_path, "r", newline="")
        _rows = list(csv.DictReader(_metrics_fh))
        _metrics_fh.close()
        kept = [r for r in _rows if int(r["step"] or 0) < resume_step]
        _metrics_fh = open(_metrics_path, "w", newline="")
        _metrics_writer = csv.DictWriter(_metrics_fh, fieldnames=_metrics_fields)
        _metrics_writer.writeheader()
        _metrics_writer.writerows(kept)
        _metrics_fh.flush()
    else:
        _metrics_fh = open(_metrics_path, "w", newline="")
        _metrics_writer = csv.DictWriter(_metrics_fh, fieldnames=_metrics_fields)
        _metrics_writer.writeheader()

    obs, _ = env.reset()
    ep_buffer = {"act": [], "logp": [], "val": [], "rew": [], "term_rew": [], "done": []}
    if use_gnn:
        ep_buffer["graph_data"] = []
        ep_buffer["map_vec"] = []
        ep_buffer["progress"] = []
        ep_buffer["phase"] = []
        ep_buffer["coupling_map"] = []
        ep_buffer["sabre_feats"] = []
    else:
        ep_buffer["obs"] = []
    ep_total_reward = 0.0
    ep_rewards = []
    ep_fids = []
    ep_swaps_log = []
    ep_map_swaps_log = []
    ep_time_log = []
    ep_xtalk_log = []
    ep_idle_log = []
    ep_par_log = []
    ep_truncated = 0
    ep_completed = 0

    # Per-topology tracking
    topo_steps = [0] * num_topos
    topo_episodes = [0] * num_topos
    topo_trunc = [0] * num_topos
    topo_completed = [0] * num_topos
    topo_swaps_lists = [[] for _ in range(num_topos)]
    topo_fids_lists = [[] for _ in range(num_topos)]
    topo_rewards_lists = [[] for _ in range(num_topos)]

    # 每尺寸 SABRE 参考保真度（log-相对奖励分母）；NAM 电路用独立表
    sabre_fid_map = _parse_sabre_fid_map(args.sabre_fid_map)
    sabre_fid_map_nam = _parse_sabre_fid_map(args.sabre_fid_map_nam) if args.sabre_fid_map_nam else None

    total_steps = resume_step
    best_metric = resume_best if resume_best > -1.0 else -1.0
    best_ema_metric = -1.0
    best_ema_step = 0
    cycle_idx = 0

    while total_steps < args.timesteps:
        for _ in range(args.rollout_steps):
            if use_gnn:
                ep_buffer["graph_data"].append(env._last_graph_data)
                ep_buffer["map_vec"].append(env._last_map_vec)
                ep_buffer["progress"].append(env._last_progress)
                ep_buffer["phase"].append(1.0 if env.mapping_phase else 0.0)
                ep_buffer["coupling_map"].append(coupling_map)
                ep_buffer["sabre_feats"].append(env._last_sabre_feats.flatten())
            else:
                ep_buffer["obs"].append(obs)

            deadlock_mask = env.get_deadlock_mask()
            combined_mask = deadlock_mask | env.get_unmapped_mask()
            commit_allowed = env.mapping_phase and (
                env._mapping_swaps >= args.mapping_min_swaps
            )
            action, logp, val = agent.act(obs, deadlock_mask=combined_mask,
                                          mapping_phase=commit_allowed)
            next_obs, reward, done, truncated, info = env.step(action)

            episode_end = done or truncated
            ep_total_reward += reward

            term = info.get("terminal_reward", 0.0)
            ep_buffer["act"].append(action)
            ep_buffer["logp"].append(logp)
            ep_buffer["val"].append(val)
            ep_buffer["rew"].append(reward - term)
            ep_buffer["term_rew"].append(term)
            ep_buffer["done"].append(episode_end)

            obs = next_obs
            total_steps += 1

            if episode_end:
                ep_rewards.append(ep_total_reward)
                if truncated:
                    ep_truncated += 1
                else:
                    ep_completed += 1
                if info.get("fidelity") is not None:
                    ep_fids.append(info["fidelity"])
                ep_swaps_log.append(info.get("num_swaps", 0))
                ep_map_swaps_log.append(info.get("mapping_swaps", 0))
                if env.timing is not None:
                    ep_time_log.append(env.timing.total_time)
                    ep_xtalk_log.append(env.timing.crosstalk_events)
                    ep_idle_log.append(float(env.timing.qubit_idle_time.sum()))
                    tt = env.timing.total_time
                    par = (env.timing.serial_dur / tt - 1.0) if tt > 1e-9 else 0.0
                    ep_par_log.append(max(0.0, par))

                # Per-topology tracking
                topo_steps[topo_idx] += env._episode_step
                topo_episodes[topo_idx] += 1
                if truncated:
                    topo_trunc[topo_idx] += 1
                else:
                    topo_completed[topo_idx] += 1
                topo_swaps_lists[topo_idx].append(info.get("num_swaps", 0))
                topo_rewards_lists[topo_idx].append(ep_total_reward)
                if info.get("fidelity") is not None:
                    topo_fids_lists[topo_idx].append(info["fidelity"])

                progress = total_steps / args.timesteps
                split_key = phase_fn(progress)
                # 多拓扑：按 --topo-balance 策略选拓扑（先选拓扑，电路须适配其容量）
                if num_topos > 1:
                    if args.topo_balance == "steps":
                        max_s = max(topo_steps)
                        w = [max_s - s + float(num_topos) for s in topo_steps]
                        topo_idx = random.choices(range(num_topos), weights=w, k=1)[0]
                    else:
                        topo_idx = random.randrange(num_topos)
                new_dag, circuit_path = pick_circuit_with_nam(
                    args, split_key, split_prefix, split_map, total_steps, topo_qubits, topo_idx,
                    nam_circuits, args.nam_circuit_prob)
                noise_config, coupling_map = topo_list[topo_idx]
                if args.noise_perturb > 0 or args.noise_perturb_t1t2 > 0:
                    noise_config = perturb_noise_config(
                        noise_config,
                        frac_t1t2=args.noise_perturb_t1t2,
                        frac_gate=args.noise_perturb,
                        frac_other=args.noise_perturb_other,
                    )
                hw = HardwareFeatures.from_noise_config(noise_config)
                agent.coupling_map = coupling_map

                # 布局混合：按 mix 比例选 恒等/随机/SABRE 初始布局
                ep_random_init = args.random_init
                ep_init_mapping = None
                if mix is not None:
                    cat = np.random.choice(["id", "rand", "sabre"], p=mix)
                    if cat == "id":
                        ep_random_init = False
                    elif cat == "rand":
                        ep_random_init = True
                    else:  # sabre
                        layout = sabre_cache[topo_idx].get(circuit_path)
                        if layout is not None:
                            ep_random_init = False
                            ep_init_mapping = layout
                        # 缓存缺失则退化恒等
                cur_lambda_fid = lambda_fid_schedule(
                    progress, args.lambda_fid_warmup,
                    adaptive_lambda_fid_max(progress, args.lambda_fid_max_schedule, args.lambda_fid_max)
                ) if args.reward_mode != "routing" else None
                # 方案1：NAM 电路若有 per-circuit SABRE 参考（nam_sref_map）则用它做 log-相对
                # 分母（sref_override 优先于按 num_qubits 共享的 sabre_fid_map_nam），避免
                # 深电路因共享同尺寸 sref 而 log(F/sref) 爆炸成极端负奖励。
                is_nam = circuit_path.startswith("nam/")
                ep_sabre_map = sabre_fid_map
                ep_sref_override = None
                if is_nam:
                    ep_sabre_map = sabre_fid_map_nam if sabre_fid_map_nam is not None else sabre_fid_map
                    nam_name = circuit_path.split("/", 1)[1]
                    pc_sref = nam_sref_map.get(nam_name)
                    if pc_sref is not None and pc_sref > 0:
                        ep_sref_override = float(pc_sref)
                        ep_sabre_map = None  # 用 per-circuit sref，不再需要共享表
                env = create_env(new_dag, hw, coupling_map, args.reward_mode,
                                 args.max_episode_steps, ep_random_init,
                                 args.seed + total_steps,
                                 gnn=shared_gnn, use_gnn=use_gnn,
                                 max_num_edges=max_edges,
                                 max_num_qubits=args.max_num_qubits,
                                 noise_config=noise_config if args.reward_mode != "routing" else None,
                                  lambda_fid=cur_lambda_fid,
                                  eta_dist=args.eta_dist,
                                  mapping_budget=args.mapping_budget,
                                  mapping_phase=args.mapping_phase,
                                   use_scheduler=(args.use_scheduler or args.fidelity_sim == "trajectory_sched"),
                                   eta_time=args.eta_time,
                                   eta_xtalk_par=args.eta_xtalk_par,
                                   eta_idle=args.eta_idle,
                                   eta_parallel=args.eta_parallel,
                                   xtalk_alpha=args.xtalk_alpha,
                                    swap_duration=args.swap_duration,
                                     init_mapping=ep_init_mapping,
                                     lambda_layout=args.lambda_layout,
                                      fidelity_fn=build_fidelity_fn(
                                        args.fidelity_sim, noise_config,
                                        num_trajectories=args.traj_trajectories,
                                        seed=args.traj_seed,
                                        analytic_thermal=args.analytic_thermal,
                                        analytic_crosstalk=args.analytic_crosstalk,
                                     ) if args.reward_mode != "routing" else None,
                                     sabre_fid_map=ep_sabre_map,
                                     sref_override=ep_sref_override)
                obs, _ = env.reset()
                ep_total_reward = 0.0

        with torch.no_grad():
            if use_gnn:
                mask = torch.zeros(agent.num_edges + 1, dtype=torch.bool, device=agent.device)
                mask[:len(agent.coupling_map)] = True
                if env.mapping_phase and env._mapping_swaps >= args.mapping_min_swaps:
                    mask[agent.num_edges] = True
                last_val = agent._forward_obs(obs, action_mask=mask.unsqueeze(0))[1]
            else:
                last_val = agent._forward_obs(obs)[1]
        if use_gnn:
            rew_arr = np.array(ep_buffer["rew"], dtype=float)
            agent.rew_norm.update(rew_arr)
            norm_rew = agent.rew_norm.normalize(rew_arr)

            term_arr = np.array(ep_buffer["term_rew"], dtype=float)
            nz = term_arr[term_arr != 0]
            if nz.size:
                agent.term_norm.update(nz)
            norm_term = np.zeros_like(term_arr)
            m = term_arr != 0
            if m.any():
                norm_term[m] = agent.term_norm.normalize(term_arr[m])
            norm_rew = norm_rew + norm_term
        else:
            norm_rew = ep_buffer["rew"]
        clip_return = args.clip_return if args.clip_return > 0 else None
        if args.gae_adaptive:
            T = len(norm_rew)
            ratio = np.arange(T) / max(T - 1, 1)
            lam_t = args.gae_lam_min + (args.gae_lam_max - args.gae_lam_min) * ratio
        else:
            lam_t = agent.lam
        adv, ret = PPOAgent.compute_gae(
            norm_rew, ep_buffer["val"], ep_buffer["done"],
            bootstrap=last_val, gamma=agent.gamma, lam=lam_t,
            clip_return=clip_return,
        )
        train_batch = {
            "act": ep_buffer["act"],
            "logp": ep_buffer["logp"],
            "adv": adv,
            "ret": ret,
        }
        if use_gnn:
            train_batch["graph_data"] = ep_buffer["graph_data"]
            train_batch["map_vec"] = ep_buffer["map_vec"]
            train_batch["progress"] = ep_buffer["progress"]
            train_batch["phase"] = ep_buffer["phase"]
            if "coupling_map" in ep_buffer and len(ep_buffer["coupling_map"]) > 0:
                train_batch["coupling_map"] = ep_buffer["coupling_map"]
            train_batch["sabre_feats"] = ep_buffer["sabre_feats"]
        else:
            train_batch["obs"] = ep_buffer["obs"]
        if use_gnn:
            agent.gnn.train()
        if total_steps < args.freeze_first_steps:
            losses = {"pl": 0.0, "vl": 0.0, "ent": 0.0, "kl": 0.0, "grad": 0.0}
        else:
            losses = agent.update(train_batch, epochs=args.epochs)
            agent.update_ema()
        ep_buffer = {k: [] for k in ep_buffer}

        avg_rew = np.mean(ep_rewards[-20:]) if ep_rewards else 0.0
        avg_swaps = np.mean(ep_swaps_log[-20:]) if ep_swaps_log else 0.0
        avg_map_swaps = np.mean(ep_map_swaps_log[-20:]) if ep_map_swaps_log else 0.0
        total_eps = ep_completed + ep_truncated
        trunc_pct = 100 * ep_truncated / max(1, total_eps)
        parts = [
            f"step={total_steps:>6d}",
            f"rew={avg_rew:+.3f}",
            f"swp={avg_swaps:.1f}",
            f"map={avg_map_swaps:.1f}",
            f"trunc={trunc_pct:.0f}%",
            f"pl={losses['pl']:.3f}",
            f"vl={losses['vl']:.3f}",
            f"ent={losses['ent']:.3f}",
            f"kl={losses['kl']:.4f}",
            f"gn={losses['grad']:.3f}",
        ]
        if ep_fids:
            avg_fid = np.mean(ep_fids[-20:])
            parts.append(f"fid={avg_fid:.4f}")
        if env.use_scheduler and ep_time_log:
            parts.append(f"time={np.mean(ep_time_log[-20:]):.1f}us")
            parts.append(f"xtalk={np.mean(ep_xtalk_log[-20:]):.3f}")
        print("  ".join(parts))

        if num_topos > 1:
            for i in range(num_topos):
                ep_i = topo_episodes[i]
                if ep_i == 0:
                    continue
                trunc_i = topo_trunc[i]
                swp_i = np.mean(topo_swaps_lists[i]) if topo_swaps_lists[i] else 0.0
                swp_s = np.std(topo_swaps_lists[i]) if len(topo_swaps_lists[i]) > 1 else 0.0
                fid_i = np.mean(topo_fids_lists[i]) if topo_fids_lists[i] else 0.0
                rew_i = np.mean(topo_rewards_lists[i]) if topo_rewards_lists[i] else 0.0
                print(f"  {topo_names[i]:<12s}  {ep_i:>4d}ep | "
                      f"trunc={100*trunc_i/max(1,ep_i):.0f}%  "
                      f"swp={swp_i:.1f}\u00b1{swp_s:.1f}  "
                      f"fid={fid_i:.4f}  rew={rew_i:+.2f}")

        # --- metrics ---
        avg_time = np.mean(ep_time_log[-20:]) if ep_time_log else ""
        avg_xtalk = np.mean(ep_xtalk_log[-20:]) if ep_xtalk_log else ""
        avg_idle = np.mean(ep_idle_log[-20:]) if ep_idle_log else ""
        avg_par = np.mean(ep_par_log[-20:]) if ep_par_log else ""
        row = {
            "step": total_steps,
            "reward": avg_rew,
            "swaps": avg_swaps,
            "map_swaps": avg_map_swaps,
            "trunc_pct": trunc_pct,
            "pl": losses["pl"],
            "vl": losses["vl"],
            "ent": losses["ent"],
            "kl": losses["kl"],
            "grad": losses["grad"],
            "fid": avg_fid if ep_fids else "",
            "time_us": avg_time,
            "xtalk": avg_xtalk,
            "idle": avg_idle,
            "par": avg_par,
        }
        _metrics_writer.writerow(row)
        _metrics_fh.flush()

        # --- periodic checkpoint ---
        if cycle_idx % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_step{total_steps:06d}.pt")
            agent.save_checkpoint(ckpt_path, extra_state={"step": total_steps, "best_metric": best_metric})

        if args.reward_mode == "routing":
            metric = avg_rew
        else:
            metric = np.mean(ep_fids[-20:]) if ep_fids else 0.0
        if metric > best_metric:
            best_metric = metric
            agent.save(args.out)

        # --- EMA 评估 + 最优 checkpoint ---
        if (args.eval_interval > 0 and _eval_circuits
                and total_steps % args.eval_interval < args.rollout_steps
                and total_steps > 0):
            eval_fid = _eval_ema(agent, _eval_circuits, args, shared_gnn, use_gnn, max_edges, topo_list)
            print(f"  [EMA eval] step={total_steps}  mean_fid={eval_fid:.6f}")
            if eval_fid > best_ema_metric:
                best_ema_metric = eval_fid
                best_ema_step = total_steps
                # 保存 EMA 最优 checkpoint
                agent.apply_ema()
                agent.save(args.out.replace(".pt", "_ema_best.pt"))
                agent.save_checkpoint(
                    os.path.join(ckpt_dir, "ema_best.pt"),
                    extra_state={"step": total_steps, "ema_fid": eval_fid}
                )
                agent.restore_from_ema()
                print(f"  [EMA best] step={total_steps}  fid={eval_fid:.6f}")

        cycle_idx += 1

    # --- final checkpoint ---
    last_path = os.path.join(ckpt_dir, "last.pt")
    agent.save_checkpoint(last_path, extra_state={"step": total_steps, "best_metric": best_metric})
    _metrics_fh.close()
    print(f"Checkpoints in {ckpt_dir}")
    print(f"Policy saved to {args.out}")
    if best_ema_metric > -1.0:
        print(f"EMA best: step={best_ema_step}  fid={best_ema_metric:.6f}")
        print(f"EMA best policy: {args.out.replace('.pt', '_ema_best.pt')}")


if __name__ == "__main__":
    main()
