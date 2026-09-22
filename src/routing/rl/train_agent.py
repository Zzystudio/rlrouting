from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
import math
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
    return CircuitDAG.from_circuit(qc), path, qc


def load_nam_circuits(nam_dir: str, max_qubits: int = 20):
    """加载目录下所有 QASM 文件为 CircuitDAG 列表（过滤 > max_qubits 的电路）。

    返回 [(dag, name, qc), ...]，name 为不带扩展名的文件名。
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
            dags.append((dag, name, qc))
        except Exception as e:
            print(f"[nam-circuits] 跳过 {fname}: {e}")
    return dags


def pick_circuit_with_nam(args, split_key, split_prefix, split_map, total_steps, topo_qubits, topo_idx,
                          nam_circuits, nam_prob, stage_cap=None):
    """带 NAM 电路混入的电路采样器。

    以 nam_prob 概率从 nam_circuits 中均匀选择，否则走原 pick_circuit_with_path。
    NAM 电路取不超过当前拓扑容量者（额外受 nam_max_q 上限约束，保证 ≤16q 训练
    时不用 trajectory_sched 跑大电路）。
    stage_cap：S5 课程限幅——QASM 电路在当前课程阶段不超过该量子比特数
    （修复「5q 阶段采到 20q QASM」的缺口）。
    返回 (dag, path, qc)（qc 供 SABRE SWAP 预算缓存用；pkl 分支同样返回）。
    """
    if nam_circuits and random.random() < nam_prob:
        nam_max_q = args.nam_max_qubits or args.max_num_qubits or topo_qubits[topo_idx]
        cap = min(topo_qubits[topo_idx], nam_max_q)
        if stage_cap is not None:
            cap = min(cap, stage_cap)
        # 从 NAM 电路中均匀选择，找到能放进当前拓扑的
        candidates = [(d, n, q) for d, n, q in nam_circuits if d.num_logical_qubits <= cap]
        if candidates:
            dag, name, qc = random.choice(candidates)
            return dag, f"nam/{name}", qc
    return pick_circuit_with_path(args.data_dir, split_key, seed=args.seed + total_steps,
                                  split_prefix=split_prefix, split_map=split_map,
                                  max_qubits=topo_qubits[topo_idx])


def _normalize_sabre_cache(data: dict) -> dict:
    """兼容旧格式 {topo: {rel: [layout]}} → 新格式 {topo: {rel: {"layout":..,"swaps":..}}。"""
    for ti, sub in data.items():
        for k, v in list(sub.items()):
            if isinstance(v, (list, tuple)):
                sub[k] = {"layout": list(v), "swaps": None}
    return data


def _extract_swap_schedule(phys) -> list:
    """从 sabre_route 返回的物理电路中按顺序提取 SWAP 边序列（物理索引对）。"""
    swaps = []
    if phys is None:
        return swaps
    for inst, qargs, _cargs in phys.data:
        if inst.name == "swap" and qargs:
            swaps.append((qargs[0]._index, qargs[1]._index))
    return swaps


def _build_sabre_layout_cache(args, topo_list) -> dict:
    """预计算训练池中每个电路在各拓扑下的 SABRE 初始布局。

    返回 {topo_idx: {rel_path: {"layout": [phys_idx,...], "swaps": [(p,q),...]}}}。
    布局是免费的初始映射（虚拟重标号），用于训练期 'sabre' 布局混合，迫使策略
    学习布局无关路由；swaps 是 SABRE 示范蒸馏（--sabre-demo-lambda）的换手序列。
    """
    from routing.routing import sabre_route

    prefix = args.split_prefix or {
        "routing": "stage1",
        "noise_aware": "stage2",
        "fidelity_shaping": "stage3",
    }[args.reward_mode]
    split_names = [f"{prefix}_phase1", f"{prefix}_phase2", f"{prefix}_phase3"]

    need_schedule = getattr(args, "sabre_demo_lambda", 0.0) > 0

    # 缓存可复用：从文件加载或构建后保存
    if args.sabre_cache_file and os.path.exists(args.sabre_cache_file):
        with open(args.sabre_cache_file, "rb") as f:
            data = _normalize_sabre_cache(pickle.load(f))
        has_schedule = any(
            (entry is not None and isinstance(entry, dict)
             and entry.get("swaps")) or False
            for sub in data.values() for entry in sub.values())
        if not (need_schedule and not has_schedule):
            print(f"[sabre-cache] 复用缓存 {args.sabre_cache_file}"
                  + ("" if not need_schedule else "（含换手序列）"))
            return data
        print(f"[sabre-cache] demo 需换手序列但缓存为旧格式，重建 "
              f"{args.sabre_cache_file}")

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
                phys, info = sabre_route(
                    qc, config, heuristic="decay",
                    swap_trials=args.sabre_layout_trials, seed=args.seed,
                )
                cache[topo_idx][rel] = {
                    "layout": info.get("initial_layout"),
                    "swaps": _extract_swap_schedule(phys),
                }
            except Exception as e:
                print(f"[sabre-cache] 跳过 {rel} (topo{topo_idx}): {e}")

    # 扩展：覆盖 --nam-circuits-dir 的 QASM 电路（20260917 B1：layout-mix
    # 需覆盖 QASM 训练分布，否则 nam 分支缓存缺失退化恒等，mix 不生效）
    if getattr(args, "nam_circuits_dir", None):
        from qiskit.qasm2 import loads as _qasm_load
        for _qd in args.nam_circuits_dir.split(","):
            _qd = _qd.strip()
            if not _qd or not os.path.isdir(_qd):
                continue
            for _f in sorted(os.listdir(_qd)):
                if not _f.endswith(".qasm"):
                    continue
                _key = f"nam/{_f.removesuffix('.qasm')}"
                for _ti in range(len(topo_list)):
                    if _key in cache[_ti]:
                        continue
                    try:
                        with open(os.path.join(_qd, _f)) as _fh:
                            _qc_q = _qasm_load(_fh.read())
                        _phys, _si = sabre_route(_qc_q, topo_list[_ti][0],
                                                 swap_trials=args.sabre_layout_trials,
                                                 seed=args.seed)
                        _il = _si.get("initial_layout")
                        if _il is not None:
                            cache[_ti][_key] = {
                                "layout": list(_il),
                                "swaps": _extract_swap_schedule(_phys),
                            }
                    except Exception as _e:
                        print(f"[sabre-cache] 跳过 {_key} (topo{_ti}): {_e}",
                              flush=True)
        _n_qasm = sum(1 for ti in range(len(topo_list))
                      for k in cache[ti] if k.startswith("nam/"))
        print(f"[sabre-cache] QASM 电路布局扩展: {_n_qasm} 条")

    if args.sabre_cache_file:
        os.makedirs(os.path.dirname(args.sabre_cache_file) or ".", exist_ok=True)
        with open(args.sabre_cache_file, "wb") as f:
            pickle.dump(cache, f)
        print(f"[sabre-cache] 保存 {sum(len(v) for v in cache.values())} 条缓存 -> {args.sabre_cache_file}")
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
                use_scheduler=(args.use_scheduler or args.fidelity_sim in ("trajectory_sched", "trajectory_v2", "trajectory_v3")),
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


def create_env(dag, hw, coupling_map, reward_mode, max_episode_steps, random_init, seed, gnn=None, use_gnn=True, max_num_edges=None, max_num_qubits=None, noise_config=None, lambda_fid=None, eta_dist=None, mapping_budget=None, mapping_phase=True, fidelity_fn=None, use_scheduler=None, eta_time=None, eta_xtalk_par=None, eta_idle=None, eta_parallel=None, xtalk_alpha=None, swap_duration=None, swap_cost=None, eta_swap_err=None, eta_err=None, reward_potential=None, eta_xtalk=None, unfinished_penalty=None, gate_base_cx=None, gate_base_1q=None, shaping_gamma=None, eta_shape=None, alpha_ext=None, no_progress_limit=None, lookahead_features=None, init_mapping=None, lambda_layout=None, sabre_fid_map=None, sref_override=None, edge_noise_features=None, beta_noise=None, w_err=None, w_xt=None, w_xt_swap=None, pot_progress_b=None, pot_1q_reward=None, lambda_budget=None, budget_delta=None, sabre_swap_budget=None,
                    swap_price_scale=None, step_cap_mult=None,
                    clocked=False, max_ready=None, w_xt_launch=None,
                    w_zz=None, eta_ao=None, step_cap_factor=None):
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
    if eta_swap_err is not None:
        kw["eta_swap_err"] = eta_swap_err
    if eta_err is not None:
        kw["eta_err"] = eta_err
    if reward_potential is not None:
        kw["reward_potential"] = reward_potential
    if eta_xtalk is not None:
        kw["eta_xtalk"] = eta_xtalk
    if unfinished_penalty is not None:
        kw["unfinished_penalty"] = unfinished_penalty
    if gate_base_cx is not None or gate_base_1q is not None:
        from routing.rl.env import _GATE_BASE_REWARD_DEFAULT
        gb = dict(_GATE_BASE_REWARD_DEFAULT)
        if gate_base_cx is not None:
            gb["cx"] = gate_base_cx
        if gate_base_1q is not None:
            for _k in ("h", "sx", "x", "rz", "y", "z", "s", "t"):
                gb[_k] = gate_base_1q
        kw["gate_base_reward"] = gb
    if shaping_gamma is not None:
        kw["shaping_gamma"] = shaping_gamma
    if eta_shape is not None:
        kw["eta_shape"] = eta_shape
    if alpha_ext is not None:
        kw["alpha_ext"] = alpha_ext
    if no_progress_limit is not None:
        kw["no_progress_limit"] = no_progress_limit
    if lookahead_features is not None:
        kw["lookahead_features"] = lookahead_features
    if init_mapping is not None:
        kw["init_mapping"] = init_mapping
    if lambda_layout is not None:
        kw["lambda_layout"] = lambda_layout
    if sabre_fid_map is not None:
        kw["sabre_fid_map"] = sabre_fid_map
    if sref_override is not None:
        kw["sref_override"] = sref_override
    if edge_noise_features is not None:
        kw["edge_noise_features"] = edge_noise_features
    if beta_noise is not None:
        kw["beta_noise"] = beta_noise
    if w_err is not None:
        kw["w_err"] = w_err
    if w_xt is not None:
        kw["w_xt"] = w_xt
    if w_xt_swap is not None:
        kw["w_xt_swap"] = w_xt_swap
    if pot_progress_b is not None:
        kw["pot_progress_b"] = pot_progress_b
    if pot_1q_reward is not None:
        kw["pot_1q_reward"] = pot_1q_reward
    if lambda_budget is not None:
        kw["lambda_budget"] = lambda_budget
    if sabre_swap_budget is not None:
        kw["sabre_swap_budget"] = sabre_swap_budget
    if swap_price_scale is not None:
        kw["swap_price_scale"] = swap_price_scale
    if step_cap_mult is not None:
        kw["step_cap_mult"] = step_cap_mult
    kw["mapping_phase"] = mapping_phase
    if clocked:
        from routing.rl.env_clocked import ClockedRoutingEnv
        if max_ready is not None:
            kw["max_ready"] = max_ready
        if w_xt_launch is not None:
            kw["w_xt_launch"] = w_xt_launch
        if w_zz is not None:
            kw["w_zz"] = w_zz
        if eta_ao is not None:
            kw["eta_ao"] = eta_ao
        if step_cap_factor is not None:
            kw["step_cap_factor"] = step_cap_factor
        return ClockedRoutingEnv(**kw)
    return RoutingEnv(**kw)

def build_fidelity_fn(fidelity_sim: str, noise_config, num_trajectories: int = 64, seed=None,
                      analytic_thermal: bool = True, analytic_crosstalk: bool = False,
                      backend: str = "cpu"):
    """按 --fidelity-sim 构造 env 终端保真度函数；routing 模式或 aer 模式返回 None。

    aer: 使用 env 内置 NoiseSimulator（density_matrix + counts overlap，n<=12）。
    trajectory: 使用轨迹状态向量模拟器（O(2^n) 内存，20q+ 可用）。
    analytic: 解析错误累积代理（O(门数)，无指数，适用于 16q+ 大电路训练）。
    backend: 轨迹模拟器后端（cpu/cuda/cuda:N/auto），仅 trajectory/trajectory_v2 生效；
             训练默认 cpu（历史口径），v3 噪声训练可 --sim-device cuda 提速。
    """
    if fidelity_sim == "trajectory":
        from sim.trajectory_sim import make_trajectory_fidelity_fn
        return make_trajectory_fidelity_fn(noise_config, num_trajectories=num_trajectories, seed=seed,
                                           backend=backend)
    if fidelity_sim == "trajectory_sched":
        from sim.trajectory_sim import make_trajectory_fidelity_fn
        return make_trajectory_fidelity_fn(noise_config, num_trajectories=num_trajectories,
                                           seed=seed, scheduled=True, backend=backend)
    if fidelity_sim == "trajectory_v2":
        from sim.trajectory_sim_v2 import make_event_fidelity_fn
        return make_event_fidelity_fn(noise_config, num_trajectories=num_trajectories,
                                      seed=seed, backend=backend)
    if fidelity_sim == "trajectory_v3":
        from sim.trajectory_sim_v3 import make_event_fidelity_fn_v3
        return make_event_fidelity_fn_v3(noise_config,
                                         num_trajectories=num_trajectories,
                                         seed=seed, backend=backend)
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


def _parse_noise_hetero(spec: str):
    """解析 "0:0.4,1:0.3,2:0.3" → (levels, probs)（归一化）。"""
    levels, probs = [], []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split(":")
        levels.append(int(k))
        probs.append(float(v))
    s = sum(probs)
    probs = [p / s for p in probs]
    return levels, probs


def randomize_noise_landscape(config: NoiseConfig, level: int, rng) -> NoiseConfig:
    """P2-b/S5 噪声景观三档随机化（在 L0 乘性扰动之后叠加）：

    - L1 异构放大：new = mean + amp·(orig−mean)，amp∈{2,3} 随机，
      边错误 clip [0.001, 0.1]（与 stronghetero 构造同式）；
    - L2 位置重排：边错误/ZZ 值在耦合边集上随机置换 + T1/T2（同置换保
      T2≤T1 配对）/readout/1q 在比特上置换（边际分布不变、位置随机）。
    """
    cfg = copy.deepcopy(config)
    if level <= 0:
        return cfg

    tqe = cfg.two_q_gate_error
    if isinstance(tqe, dict) and tqe:
        keys = list(tqe.keys())
        vals = np.array([float(tqe[k]) for k in keys])
        if level == 1:
            amp = float(rng.choice([2.0, 3.0]))
            new = np.clip(vals.mean() + amp * (vals - vals.mean()), 0.001, 0.1)
        else:
            new = vals[rng.permutation(len(vals))]
        cfg.two_q_gate_error = {k: float(v) for k, v in zip(keys, new)}

    cs = cfg.crosstalk_strength
    if isinstance(cs, dict) and cs:
        keys = list(cs.keys())
        vals = np.array([float(cs[k]) for k in keys])
        if level == 1:
            amp = float(rng.choice([2.0, 3.0]))
            hi = max(0.1, float(vals.max()) * 3.0)
            new = np.clip(vals.mean() + amp * (vals - vals.mean()), 0.0, hi)
        else:
            new = vals[rng.permutation(len(vals))]
        cfg.crosstalk_strength = {k: float(v) for k, v in zip(keys, new)}

    if level >= 2:
        n = len(cfg.t1_times)
        if n > 1:
            perm = rng.permutation(n)
            cfg.t1_times = [float(cfg.t1_times[j]) for j in perm]
            cfg.t2_times = [float(cfg.t2_times[j]) for j in perm]
            if cfg.readout_error:
                cfg.readout_error = [float(cfg.readout_error[j]) for j in perm]
            if isinstance(cfg.single_q_gate_error, (list, tuple)):
                cfg.single_q_gate_error = [float(cfg.single_q_gate_error[j])
                                           for j in perm]
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
    parser.add_argument("--clocked", action="store_true", default=False,
                        help="时钟化动作空间（doc/20260920训练方案.md）："
                             "词表 [E SWAP | K EXEC | commit | skip]，事件时间")
    parser.add_argument("--max-ready-2q", type=int, default=24,
                        help="时钟化 EXEC 候选槽位数 K（priority-K，pad+mask）")
    parser.add_argument("--w-xt-launch", type=float, default=0.0,
                        help="时钟化边际 ZZ 价（Step 0 v3 审计标定）")
    parser.add_argument("--w-zz", type=float, default=0.0,
                        help="时钟化静态 ZZ 价（Step 0 v3 审计标定）")
    parser.add_argument("--eta-ao", type=float, default=0.0,
                        help="always-on ZZ 价（与模拟器 always_on_zz 联动，默认关）")
    parser.add_argument("--step-cap-factor", type=float, default=2.0,
                        help="时钟化步数上限系数（相对门数，默认 2.0）")
    parser.add_argument("--freeze-edge-gnn", action="store_true", default=False,
                        help="C0：冻结 GNN + edge 头，只训 gate/skip/critic")
    parser.add_argument("--edge-anchor-lambda", type=float, default=0.0,
                        help="C1b：edge 头 + GNN 的 L2 锚定权重（防解冻漂移，"
                             "锚 = 载入时 LA287core 初值）")
    parser.add_argument("--routing-mimic", action="store_true", default=False,
                        help="路由头用特征驱动 SABRE 模仿（argmin sabre_core，"
                             "确定性、零训练、t287 平价）；RL 只学 EXEC/SKIP 调度")
    parser.add_argument("--bc-warmup-steps", type=int, default=0,
                        help="C0 前置：auto-batch 示范蒸馏步数（gate/skip 头 BC）")
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
    parser.add_argument("--eta-swap-err", type=float, default=None,
                        help="SWAP 边噪声惩罚权重（v2 口径：SWAP=3×CX，按所在边 two_q_err 计价；"
                             "≈3×eta_err 时完全对价；默认 None=0 关闭）")
    parser.add_argument("--eta-err", type=float, default=None,
                        help="门边噪声惩罚权重（默认 None=env 默认 0.5；v2 对价标定建议 20，"
                             "使噪声惩罚与 +2.0/门 的完成奖励同量级）")
    parser.add_argument("--reward-potential", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="R2 势函数奖励：r=进度(0.045/门)−物理噪声代价（边感知，"
                             "SWAP=3×e_edge，虚拟SWAP免费，取消 +2.0/门 完成奖励）；"
                             "与 routing 模式组合，默认 None=关闭")
    parser.add_argument("--gate-base-cx", type=float, default=None,
                        help="R3：CX 执行奖励覆盖（env 默认 2.0；R3 建议 0.3）")
    parser.add_argument("--gate-base-1q", type=float, default=None,
                        help="R3：1Q 门执行奖励覆盖（env 默认 0.3；R3 建议 0.05）")
    parser.add_argument("--eta-xtalk", type=float, default=None,
                        help="静态门级串扰惩罚（env 默认 0.02；与调度串扰 double counting，R3 建议 0）")
    parser.add_argument("--unfinished-penalty", type=float, default=None,
                        help="截断时每剩余门惩罚（env 默认 0.5；R3 标定建议 0.15）")
    parser.add_argument("--shaping-gamma", type=float, default=None,
                        help="R3 势函数 shaping 折扣（必须等于 PPO gamma，默认 0.99）；"
                             "设置后激活 Φ(s)=−eta_shape·(D_front+α·D_ext) 前瞻塑形并停用 eta_dist")
    parser.add_argument("--eta-shape", type=float, default=None,
                        help="R3 势函数尺度（env 默认 0.3）")
    parser.add_argument("--alpha-ext", type=float, default=None,
                        help="R3 extended-set 前瞻权重 α（env 默认 0.5，对齐 SabreSwap W）")
    parser.add_argument("--no-progress-limit", type=int, default=None,
                        help="R3b 反游走：routing 阶段连续 N 步无门执行则提前截断"
                             "（按 unfinished_penalty 计；env 默认 0=关闭，建议 200）")
    parser.add_argument("--noise-ramp", type=float, default=0.0,
                        help="噪声项渐入比例（占训练进度；0=立即满值）。两阶段奖励课程："
                             "progress 达该比例前 swap_cost/eta_swap_err/eta_err/"
                             "eta_xtalk_par 从 0 线性升至目标值——先路由能力后噪声感知")
    parser.add_argument("--noise-scale", type=float, default=1.0,
                        help="噪声项总乘性比例（E16 两阶段：阶段 A 用 0.05≈纯效率路由，"
                             "阶段 B 用 1.0 恢复）。乘性作用于势函数噪声项 w_err/w_xt/"
                             "w_xt_swap/eta_xtalk_par/eta_swap_err/eta_err/swap_cost——"
                             "与 --noise-ramp 的进度爬升正交")
    parser.add_argument("--edge-hidden", type=int, default=64,
                        help="edge_mlp 首层隐藏宽度（E17 容量升级：64→128）")
    parser.add_argument("--lookahead-features", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="R5a 并发/前瞻观测特征开关（默认 None=开启；"
                             "消融用 --no-lookahead-features 置零 4 维特征，obs_dim 不变）")
    parser.add_argument("--edge-noise-features", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="P0-a per-edge 直接噪声特征（+5 维 e_edge/zz_edge/e_rel/"
                             "swap_price/cum_xz；关闭时 5 维置零、obs_dim 不变）")
    parser.add_argument("--beta-noise", type=float, default=0.0,
                        help="P0-b 噪声加权距离 β（0=纯跳数；t287 推荐 0.5，"
                             "须满足保序 β·k_max·e_max<1）")
    parser.add_argument("--w-err", type=float, default=0.0,
                        help="P0-c 势函数 E_err 项权重（t287 建议 0.02）")
    parser.add_argument("--w-xt", type=float, default=0.0,
                        help="P0-c 势函数 X(s) 串扰项权重（t287 建议 0.01）")
    parser.add_argument("--w-xt-swap", type=float, default=0.0,
                        help="P0-c per-swap 即时串扰价权重（t287 建议 0.02）")
    parser.add_argument("--pot-progress-b", type=float, default=None,
                        help="P0-d potential 模式进度奖励 B（None=env 默认 0.045；"
                             "t287 建议 0.20 = 1.5×mean(e_norm)）")
    parser.add_argument("--pot-1q-reward", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="P0-d 1Q/measure 门是否发 progress 奖励"
                             "（None=env 默认 True；--no-pot-1q-reward 置零，"
                             "消除 ~60%% 策略不可控事件流）")
    parser.add_argument("--lambda-budget", type=float, default=0.0,
                        help="P1-a SABRE SWAP 预算锚：超出预算后每颗额外 SWAP 罚"
                             "（0=关；建议 0.5；20260917 审计校准值 1.8）")
    parser.add_argument("--swap-price-scale", type=float, default=1.0,
                        help="potential 模式 SWAP 边际价格缩放（20260917 审计："
                             "奖励把 SWAP 低估 ~4.6×，校准值 4.6；默认 1.0=历史口径）")
    parser.add_argument("--swap-price-ramp", type=float, default=0.0,
                        help="价格渐入：占训练进度比例 R，scale/λ_budget 从旧值"
                             "线性爬升到目标值（0=立即满值；校准续训建议 0.4——"
                             "×4.6 价格跳变会导致熵塌缩+KL 爆炸，实测教训）")
    parser.add_argument("--budget-delta", type=float, default=1.05,
                        help="P1-a 预算膨胀系数 δ（budget=ceil(δ×SABRE swaps)）")
    parser.add_argument("--noise-hetero", type=str, default=None,
                        help="S5 噪声景观三档随机化 \"0:0.4,1:0.3,2:0.3\""
                             "（默认关=仅 L0 乘性扰动；L1 异构放大 / L2 位置重排）")
    parser.add_argument("--qasm-stage-cap", type=str, default=None,
                        help="S5 课程限幅 \"stage1:6,large_n8:10,...\"：QASM 混训"
                             "电路在当前课程阶段不超过该 qubit 数（默认关）")
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
                        choices=["aer", "trajectory", "trajectory_sched", "trajectory_v2", "trajectory_v3", "analytic"],
                        help="终端保真度模拟器: aer=density_matrix/counts (小比特数), "
                             "trajectory=轨迹状态向量(串行, O(2^n) 内存), "
                             "trajectory_sched=轨迹状态向量+调度感知(空闲退相干/动态串扰, 需 --use-scheduler), "
                             "trajectory_v2=事件级调度感知 v2(per-gate 时长/重叠缩放串扰/always-on 可选, 需 --use-scheduler), "
                             "analytic=解析错误累积代理(O(门数), 无指数, 16q+ 大电路训练)")
    parser.add_argument("--traj-trajectories", type=int, default=16,
                        help="轨迹模拟器采样条数（越大方差越小，训练越慢）")
    parser.add_argument("--traj-seed", type=int, default=None,
                        help="轨迹模拟器随机种子（默认 None=不可复现）")
    parser.add_argument("--sim-device", type=str, default="cpu",
                        help="轨迹模拟器后端（cpu/cuda/cuda:N/auto；默认 cpu=历史口径）。"
                             "v3 噪声训练建议 --sim-device cuda（20q fidelity 提速 ~10-50x）")
    # ---- 训练期 beam lookahead（doc/20260916训练方案.md v2）----
    parser.add_argument("--la-beam", type=int, default=0,
                        help="训练期 beam expectimax 宽度（0=关；建议 3）")
    parser.add_argument("--la-depth", type=int, default=2,
                        help="beam expectimax 展开深度（默认 2）")
    parser.add_argument("--la-interval", type=int, default=4,
                        help="每 N 个 routing 步一个 LA anchor（默认 4；吞吐超 4x 时调 8）")
    parser.add_argument("--la-max-anchors", type=int, default=64,
                        help="每 episode 的 LA anchor 上限（默认 64）")
    parser.add_argument("--la-vf-coef", type=float, default=0.1,
                        help="V_LA expectimax 回归 loss 权重（默认 0.1）")
    parser.add_argument("--la-raw-target", action="store_true",
                        help="V_LA 回归原始尺度 la_val（20260917 A2：batch 归一化"
                             "使 γ 折扣语义失效，×4.6 价格下放大——校准续训必须开）")
    parser.add_argument("--la-distill-lambda", type=float, default=0.0,
                        help="beam 选择蒸馏强度 λ_d：L += λ_d·(−log π(la_act|s))"
                             "（0=关；v2 建议 0.5）")
    parser.add_argument("--la-distill-ramp", type=float, default=0.6,
                        help="λ_d 渐入：占训练进度比例，进度达该比例时爬满")
    parser.add_argument("--sabre-demo-lambda", type=float, default=0.0,
                        help="SABRE 示范蒸馏强度 λ_s：L += λ_s·(−log π(a_sabre|s))"
                             "（0=关；批效率老师，demo 回合强制沿 SABRE 换手序列"
                             "执行，仅路由期标签，off-policy 仅用于 BC）")
    parser.add_argument("--sabre-demo-prob", type=float, default=0.15,
                        help="每 episode 进入 demo 回合（强制 SABRE 布局+动作）的概率")
    parser.add_argument("--sabre-demo-ramp", type=float, default=0.6,
                        help="λ_s 渐入：占训练进度比例（同 λ_d ramp 语义）")
    parser.add_argument("--step-cap-mult", type=float, default=2.0,
                        help="动态步数上限的门数乘数（TRUNC 门数化，20260917："
                             "1.2 = 完成解 p90 的 1.8 倍余量；默认 2.0=历史口径）")
    parser.add_argument("--no-progress-gates-div", type=int, default=0,
                        help="反游走门数化：no_progress = max(20, gates/D)"
                             "（0=用 --no-progress-limit 固定值；建议 D=16）")
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
    # ---- Phase 2 双价值头（P2A/P2B/P2C）----
    parser.add_argument("--variant", type=str, default=None,
                        choices=["P1", "P2A", "P2B", "P2C"],
                        help="Phase 2 变体：P1=纯路由；P2A=terminal fid 直进 actor（旧行为，α=1）；"
                             "P2B=独立 V_fid 头 + α 混合 advantage；P2C=P2B + KL 保策略")
    parser.add_argument("--alpha-fid", type=float, default=0.1,
                        help="fidelity advantage 混入 actor 的比例 α（P2A 自动=1.0）")
    parser.add_argument("--beta-kl", type=float, default=0.0,
                        help="KL(π_P1‖π_P2) 权重 β（P2C 建议 0.05）")
    parser.add_argument("--lambda-v-fid", type=float, default=1.0,
                        help="beam/rollout 组合值 V = V_route + λ_V·V_fid 的权重")
    parser.add_argument("--teacher", type=str, default=None,
                        help="π_P1 教师 checkpoint（KL 约束用，通常为 P1 模型）")
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
        # 支持逗号分隔多目录（如原始 NAM + 结构化增广集），合并加载
        for _dir in args.nam_circuits_dir.split(","):
            _dir = _dir.strip()
            if _dir:
                nam_circuits.extend(load_nam_circuits(_dir, max_qubits=nam_max_q))
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
                       use_scheduler=(args.use_scheduler or args.fidelity_sim in ("trajectory_sched", "trajectory_v2", "trajectory_v3")),
                       eta_time=args.eta_time,
                                    eta_xtalk_par=args.eta_xtalk_par,
                                    eta_idle=args.eta_idle,
                                    eta_parallel=args.eta_parallel,
                                     xtalk_alpha=args.xtalk_alpha,
                                      swap_duration=args.swap_duration,
                                      swap_cost=args.swap_cost,
                                      eta_swap_err=args.eta_swap_err,
                                      eta_err=args.eta_err,
                                      reward_potential=args.reward_potential,
                                      eta_xtalk=args.eta_xtalk,
                                      unfinished_penalty=args.unfinished_penalty,
                                      gate_base_cx=args.gate_base_cx,
                                      gate_base_1q=args.gate_base_1q,
                                      shaping_gamma=args.shaping_gamma,
                                      eta_shape=args.eta_shape,
                                      alpha_ext=args.alpha_ext,
                                      no_progress_limit=args.no_progress_limit,
                                      lookahead_features=args.lookahead_features,
                                      edge_noise_features=args.edge_noise_features,
                                      beta_noise=args.beta_noise,
                                      w_err=args.w_err,
                                      w_xt=args.w_xt,
                                      w_xt_swap=args.w_xt_swap,
                                      pot_progress_b=args.pot_progress_b,
                                      pot_1q_reward=args.pot_1q_reward,
                                      lambda_budget=args.lambda_budget,
                                      swap_price_scale=args.swap_price_scale,
                                      step_cap_mult=args.step_cap_mult,
                                      init_mapping=None,
                                      lambda_layout=args.lambda_layout,
                          fidelity_fn=build_fidelity_fn(
                           args.fidelity_sim, noise_config,
                          num_trajectories=args.traj_trajectories, seed=args.traj_seed,
                          analytic_thermal=args.analytic_thermal,
                          analytic_crosstalk=args.analytic_crosstalk,
                          backend=args.sim_device,
                      ) if args.reward_mode != "routing" else None,
                      clocked=args.clocked,
                      max_ready=args.max_ready_2q,
                      w_xt_launch=args.w_xt_launch,
                      w_zz=args.w_zz,
                      eta_ao=args.eta_ao,
                      step_cap_factor=args.step_cap_factor)

    agent_n_qubits = args.max_num_qubits or sample_dag.num_logical_qubits
    agent_action_dim = max_edges + (1 if args.mapping_phase else 0)
    if args.clocked:
        from routing.rl.agent_clocked import ClockedPPOAgent, D_EXEC, D_TIMING_GLOB
        agent = ClockedPPOAgent(
            obs_dim=int(np.prod(env.observation_space.shape)),
            action_dim=env.action_space.n,
            num_qubits=agent_n_qubits,
            num_edges=max_edges,
            max_ready=args.max_ready_2q,
            edge_feat_dim=int(getattr(env, "_edge_feat_dim", 0)),
            exec_feat_dim=D_EXEC,
            timing_glob_dim=D_TIMING_GLOB,
            lr=args.lr,
            device=args.device,
            gnn=shared_gnn,
            coupling_map=coupling_map,
            vf_coef=args.vf_coef,
            with_commit=args.mapping_phase,
            lambda_v_fid=args.lambda_v_fid,
            with_la_head=False,
            edge_hidden=args.edge_hidden,
            edge_anchor_lambda=args.edge_anchor_lambda,
        )
    else:
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
        lambda_v_fid=args.lambda_v_fid,
        edge_feat_dim=(getattr(env, "_edge_feat_dim", None) if use_gnn else None),
        with_la_head=(args.la_beam > 0),
        la_vf_coef=args.la_vf_coef,
        la_raw_target=args.la_raw_target,
        edge_hidden=args.edge_hidden,
    )
    if args.load:
        state = agent.load_checkpoint(args.load)
        print(f"Loaded pretrained model: {args.load}")
        if "step" in state:
            resume_step = int(state["step"])
            resume_best = float(state.get("best_metric", -1.0))
            print(f"  -> resume at step {resume_step}, best_metric={resume_best:.5f}")

    # π_P1 教师（P2C KL 约束用）
    if args.teacher and args.variant == "P2C":
        agent.load_teacher(args.teacher)
        print(f"Teacher loaded (KL target): {args.teacher}")

    # EMA 初始化
    if args.ema_decay > 0:
        agent.init_ema(decay=args.ema_decay)
        print(f"EMA enabled: decay={args.ema_decay}")

    # ---- 时钟化 C0：冻结 edge+GNN + auto-batch BC 预热（§6.2/§七）----
    if args.clocked:
        if args.freeze_edge_gnn:
            agent.freeze_edge_gnn(freeze=True)
            print("[clocked-C0] frozen GNN + edge heads")
        if args.bc_warmup_steps > 0:
            from routing.rl.agent_clocked import bc_warmup_auto_batch
            _n, _loss = bc_warmup_auto_batch(agent, env,
                                             num_steps=args.bc_warmup_steps,
                                             seed=args.seed)
            print(f"[clocked-C0] BC warmup done: {_n} samples, loss={_loss:.3f}")

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
    _metrics_fields = ["step", "reward", "swaps", "map_swaps", "trunc_pct", "pl", "vl", "vfl", "vla", "dil", "agr", "agr_map", "agr_rout", "klp1", "ent", "kl", "grad", "fid", "time_us", "xtalk", "idle", "par", "swap_ratio", "gps", "dsl", "dalg"]
    # 续训时追加而非覆盖；已有行丢到 resume_step 为止，避免旧行与新续训混合
    if resume_step > 0 and os.path.exists(_metrics_path):
        _metrics_fh = open(_metrics_path, "r", newline="")
        _rows = list(csv.DictReader(_metrics_fh))
        _metrics_fh.close()
        kept = [r for r in _rows if int(r["step"] or 0) < resume_step]
        _metrics_fh = open(_metrics_path, "w", newline="")
        _metrics_writer = csv.DictWriter(_metrics_fh, fieldnames=_metrics_fields, restval="")
        _metrics_writer.writeheader()
        _metrics_writer.writerows(kept)
        _metrics_fh.flush()
    else:
        _metrics_fh = open(_metrics_path, "w", newline="")
        _metrics_writer = csv.DictWriter(_metrics_fh, fieldnames=_metrics_fields, restval="")
        _metrics_writer.writeheader()

    obs, _ = env.reset()
    ep_buffer = {"act": [], "logp": [], "val": [], "val_route": [], "val_fid": [],
                 "rew": [], "term_rew": [], "done": [],
                 "la_mask": [], "la_act": [], "la_val": [],
                 "demo_mask": [], "demo_act": []}
    if args.clocked:
        ep_buffer["obs"] = []
    look_on = args.lookahead_features is not False
    if use_gnn:
        ep_buffer["graph_data"] = []
        ep_buffer["map_vec"] = []
        ep_buffer["progress"] = []
        ep_buffer["phase"] = []
        ep_buffer["coupling_map"] = []
        ep_buffer["sabre_feats"] = []
        if look_on:
            ep_buffer["look_feats"] = []
        if args.edge_noise_features:
            ep_buffer["noise_feats"] = []
        ep_buffer["global_feats"] = []
        ep_buffer["sabre_core_feats"] = []
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
    ep_demo_align = []

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

    # S5/P1-a 初始化
    qasm_stage_cap = {}
    if args.qasm_stage_cap:
        for part in args.qasm_stage_cap.split(","):
            part = part.strip()
            if not part:
                continue
            k, v = part.split(":")
            qasm_stage_cap[k.strip()] = int(v)
    het_spec = _parse_noise_hetero(args.noise_hetero) if args.noise_hetero else None
    if args.noise_hetero:
        lv, pr = het_spec
        print(f"[noise-hetero] 档位 {lv} 概率 {np.round(pr, 3).tolist()}")
    if qasm_stage_cap:
        print(f"[qasm-stage-cap] {qasm_stage_cap}")
    sabre_swap_cache = {}
    ep_swap_ratio_log = []
    ep_gps_log = []

    total_steps = resume_step
    best_metric = resume_best if resume_best > -1.0 else -1.0
    best_ema_metric = -1.0
    best_ema_step = 0
    cycle_idx = 0
    circuit_path = None  # 当前 episode 的电路路径（首个 episode 为 sample 电路）
    la_anchors_this_ep = 0  # 当前 episode 已消耗的 LA anchor 数（防成本失控）
    ep_demo = False  # SABRE 示范回合标志（E4；首个 episode 为 sample 电路无缓存）
    demo_swaps = []
    demo_ptr = 0
    demo_eidx = {}
    if args.clocked:
        # 时钟化路径暂不启用 LA beam / SABRE demo（C0 用 BC 预热替代）
        args.la_beam = 0
        args.sabre_demo_lambda = 0.0
    if args.la_beam > 0:
        from routing.rl.lookahead import beam_expectimax
        print(f"[LA] beam expectimax 开启：B={args.la_beam} K={args.la_depth} "
              f"interval={args.la_interval} cap={args.la_max_anchors}/ep "
              f"vf_coef={args.la_vf_coef}")

    while total_steps < args.timesteps:
        for _ in range(args.rollout_steps):
            if args.clocked:
                ep_buffer["obs"].append(obs)
            elif use_gnn:
                ep_buffer["graph_data"].append(env._last_graph_data)
                ep_buffer["map_vec"].append(env._last_map_vec)
                ep_buffer["progress"].append(env._last_progress)
                ep_buffer["phase"].append(1.0 if env.mapping_phase else 0.0)
                ep_buffer["coupling_map"].append(coupling_map)
                ep_buffer["sabre_feats"].append(env._last_sabre_feats.flatten())
                if look_on:
                    ep_buffer["look_feats"].append(env._last_look_feats.flatten())
                if args.edge_noise_features:
                    ep_buffer["noise_feats"].append(env._last_noise_feats.flatten())
                ep_buffer["global_feats"].append(env._last_global_feats)
                ep_buffer["sabre_core_feats"].append(env._last_sabre_core_feats)
            else:
                ep_buffer["obs"].append(obs)

            deadlock_mask = env.get_deadlock_mask()
            combined_mask = deadlock_mask | env.get_unmapped_mask()
            commit_allowed = env.mapping_phase and (
                env._mapping_swaps >= args.mapping_min_swaps
            )
            # LA anchor：映射期+路由期（20260917 A3：V_LA 需覆盖映射态——
            # s3 案例布局决策是最高杠杆规划点且评估 beam 在映射期用 V_LA 打分）、
            # 每 interval 步一次、episode 内 cap（demo 回合跳过——纯 SABRE 标签）
            if (args.la_beam > 0 and not ep_demo
                    and env._episode_step % args.la_interval == 0
                    and la_anchors_this_ep < args.la_max_anchors):
                a_star, v_star, _qr = beam_expectimax(
                    env, agent, obs, beam_width=args.la_beam,
                    depth=args.la_depth, gamma=agent.gamma)
                if _qr:  # 全掩码状态无候选 → 退化为非 anchor
                    ep_buffer["la_mask"].append(True)
                    ep_buffer["la_act"].append(a_star)
                    ep_buffer["la_val"].append(v_star)
                    la_anchors_this_ep += 1
                else:
                    ep_buffer["la_mask"].append(False)
                    ep_buffer["la_act"].append(0)
                    ep_buffer["la_val"].append(0.0)
            else:
                ep_buffer["la_mask"].append(False)
                ep_buffer["la_act"].append(0)
                ep_buffer["la_val"].append(0.0)
            if ep_demo and not env.mapping_phase and demo_ptr < len(demo_swaps):
                # SABRE 示范：沿换手序列执行（强制动作，off-policy），
                # 仅当该边未被死锁掩码时才产生 BC 标签（掩码动作教了白教）
                _demo_edge = demo_swaps[demo_ptr]
                demo_ptr += 1
                action = demo_eidx.get(_demo_edge, 0)
                logp, val, v_route, v_fid = 0.0, 0.0, 0.0, 0.0
                demo_label = bool(action < len(deadlock_mask)
                                  and not deadlock_mask[action])
                ep_buffer["demo_mask"].append(demo_label)
                ep_buffer["demo_act"].append(action)
            else:
                # demo 序列耗尽（SABRE 路由已完成、剩余门自动执行中）或非 demo：
                # 回退正常策略采样（demo 尾部不产生 BC 标签）
                if args.clocked:
                    if args.routing_mimic:
                        amask = env.get_action_mask()
                        e = env.mimic_swap_index()
                        amask[:env.num_edges] = False
                        amask[e] = True
                        action, logp, val, v_route, v_fid = agent.act(
                            obs, action_mask=amask)
                    else:
                        action, logp, val, v_route, v_fid = agent.act(
                            obs, action_mask=env.get_action_mask())
                else:
                    action, logp, val, v_route, v_fid = agent.act(
                        obs, deadlock_mask=combined_mask,
                        mapping_phase=commit_allowed)
                ep_buffer["demo_mask"].append(False)
                ep_buffer["demo_act"].append(0)
            next_obs, reward, done, truncated, info = env.step(action)

            episode_end = done or truncated
            ep_total_reward += reward

            term = info.get("terminal_reward", 0.0)
            ep_buffer["act"].append(action)
            ep_buffer["logp"].append(logp)
            ep_buffer["val"].append(val)
            ep_buffer["val_route"].append(v_route)
            ep_buffer["val_fid"].append(v_fid)
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
                # 批效率（20260917：规划内化的主过程判据，SABRE ≈6.0）
                _nsw = max(1, info.get("num_swaps", 0))
                ep_gps_log.append(len(env.executed) / _nsw)
                if ep_demo and demo_swaps:
                    ep_demo_align.append(demo_ptr / max(1, len(demo_swaps)))
                ep_map_swaps_log.append(info.get("mapping_swaps", 0))
                if args.lambda_budget > 0 and circuit_path is not None:
                    s_sw = sabre_swap_cache.get((topo_idx, circuit_path))
                    if s_sw:
                        ep_swap_ratio_log.append(
                            info.get("num_swaps", 0) / max(1, s_sw))
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
                new_dag, circuit_path, ep_qc = pick_circuit_with_nam(
                    args, split_key, split_prefix, split_map, total_steps, topo_qubits, topo_idx,
                    nam_circuits, args.nam_circuit_prob,
                    stage_cap=qasm_stage_cap.get(split_key))
                noise_config, coupling_map = topo_list[topo_idx]
                if args.noise_perturb > 0 or args.noise_perturb_t1t2 > 0:
                    noise_config = perturb_noise_config(
                        noise_config,
                        frac_t1t2=args.noise_perturb_t1t2,
                        frac_gate=args.noise_perturb,
                        frac_other=args.noise_perturb_other,
                    )
                if args.noise_hetero:
                    rng_h = np.random.default_rng((args.seed * 1000003 + total_steps) % (2**31))
                    het_levels, het_probs = het_spec
                    het_level = int(rng_h.choice(het_levels, p=het_probs))
                    noise_config = randomize_noise_landscape(noise_config, het_level, rng_h)
                hw = HardwareFeatures.from_noise_config(noise_config)
                agent.coupling_map = coupling_map

                # P1-a：SABRE SWAP 预算缓存（SABRE 噪声盲，跨 episode 稳定）
                ep_swap_budget = None
                if args.lambda_budget > 0:
                    bkey = (topo_idx, circuit_path)
                    if bkey not in sabre_swap_cache:
                        try:
                            from routing.routing import sabre_route
                            _, sinfo = sabre_route(ep_qc, noise_config,
                                                   swap_trials=20, seed=args.seed)
                            sabre_swap_cache[bkey] = int(sinfo.get("num_swaps", 0) or 0)
                        except Exception as _e:
                            print(f"[budget] SABRE 路由失败 {circuit_path}: {_e}")
                            sabre_swap_cache[bkey] = None
                    s_swaps = sabre_swap_cache[bkey]
                    if s_swaps is not None and s_swaps > 0:
                        ep_swap_budget = int(np.ceil(args.budget_delta * s_swaps))

                # 布局混合：按 mix 比例选 恒等/随机/SABRE 初始布局
                ep_random_init = args.random_init
                ep_init_mapping = None
                ep_demo = False
                demo_swaps = []
                demo_ptr = 0
                demo_eidx = {}
                if mix is not None:
                    cat = np.random.choice(["id", "rand", "sabre"], p=mix)
                    if cat == "id":
                        ep_random_init = False
                    elif cat == "rand":
                        ep_random_init = True
                    else:  # sabre
                        entry = sabre_cache[topo_idx].get(circuit_path)
                        layout = entry["layout"] if isinstance(entry, dict) else entry
                        if layout is not None:
                            ep_random_init = False
                            ep_init_mapping = layout
                        # 缓存缺失则退化恒等
                # SABRE 示范回合（20260917 E4）：强制 SABRE 布局 + 沿换手序列
                # 执行，收集 (obs, a_sabre) 仅用于 BC 蒸馏（off-policy，不进 PPO）。
                # 需换手序列存在；demo 回合关闭映射期以从 SABRE 布局精确进入路由期。
                if (args.sabre_demo_lambda > 0
                        and np.random.rand() < args.sabre_demo_prob):
                    _entry = sabre_cache[topo_idx].get(circuit_path)
                    _swaps = (_entry.get("swaps") if isinstance(_entry, dict)
                              else None)
                    if _swaps:
                        ep_demo = True
                        ep_random_init = False
                        ep_init_mapping = (_entry["layout"] if isinstance(_entry, dict)
                                           else _entry)
                        demo_swaps = list(_swaps)
                        demo_ptr = 0
                        demo_eidx = {}
                        for _a, (_p, _q) in enumerate(coupling_map):
                            demo_eidx[(_p, _q)] = _a
                            demo_eidx[(_q, _p)] = _a
                ep_mapping_phase = args.mapping_phase
                cur_lambda_fid = lambda_fid_schedule(
                    progress, args.lambda_fid_warmup,
                    adaptive_lambda_fid_max(progress, args.lambda_fid_max_schedule, args.lambda_fid_max)
                ) if args.reward_mode != "routing" else None
                # 噪声项渐入（两阶段奖励课程）：progress 达 noise_ramp 前按比例
                # 线性爬升（0→1），使 swap_cost/eta_swap_err/eta_err/eta_xtalk_par
                # 从 0 平滑升至目标值——先学路由能力，再渐入噪声感知
                noise_f = 1.0
                if args.noise_ramp:
                    _nr = max(args.noise_ramp, 1e-9)
                    noise_f = min(1.0, progress / _nr)
                # E16：噪声总比例（--noise-scale，两阶段课程用；与 noise_f 叠乘，
                # 使势函数噪声项 w_err/w_xt/w_xt_swap 也被缩放，达到"纯效率阶段"）
                noise_f = noise_f * args.noise_scale
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
                # 价格渐入：scale/λ_budget 从历史值线性爬升到校准值
                if args.swap_price_ramp > 0:
                    _ramp = min(1.0, progress / args.swap_price_ramp)
                else:
                    _ramp = 1.0
                _ep_scale = 1.0 + (args.swap_price_scale - 1.0) * _ramp
                _ep_lb = 0.5 + (args.lambda_budget - 0.5) * _ramp
                # TRUNC 门数化（20260917）：步数上限 = max(50, mult×gates)、
                # 反游走 = max(20, gates/D)——完成解 sw/g ≤1.24 vs 游走 3.6-8.3×
                if args.step_cap_mult != 2.0 or args.no_progress_gates_div > 0:
                    _g = new_dag.num_gates
                    _cap = max(50, math.ceil(args.step_cap_mult * _g)) \
                        if args.step_cap_mult > 0 else args.max_episode_steps
                    _np = max(20, math.ceil(_g / args.no_progress_gates_div)) \
                        if args.no_progress_gates_div > 0 else args.no_progress_limit
                else:
                    _cap, _np = args.max_episode_steps, args.no_progress_limit
                env = create_env(new_dag, hw, coupling_map, args.reward_mode,
                                 _cap, ep_random_init,
                                 args.seed + total_steps,
                                 gnn=shared_gnn, use_gnn=use_gnn,
                                 max_num_edges=max_edges,
                                 max_num_qubits=args.max_num_qubits,
                                 noise_config=noise_config if args.reward_mode != "routing" else None,
                                   lambda_fid=cur_lambda_fid,
                                   eta_dist=args.eta_dist,
                                   mapping_budget=args.mapping_budget,
                                   mapping_phase=ep_mapping_phase,
                                    use_scheduler=(args.use_scheduler or args.fidelity_sim in ("trajectory_sched", "trajectory_v2", "trajectory_v3")),
                                    eta_time=args.eta_time,
                                    eta_xtalk_par=(args.eta_xtalk_par * noise_f
                                                  if args.eta_xtalk_par is not None else None),
                                   eta_idle=args.eta_idle,
                                   eta_parallel=args.eta_parallel,
                                    xtalk_alpha=args.xtalk_alpha,
                                     swap_duration=args.swap_duration,
                                     swap_cost=(args.swap_cost * noise_f
                                               if args.swap_cost is not None else None),
                                     eta_swap_err=(args.eta_swap_err * noise_f
                                                  if args.eta_swap_err is not None else None),
                                     eta_err=(args.eta_err * noise_f
                                             if args.eta_err is not None else None),
                                     reward_potential=args.reward_potential,
                                     eta_xtalk=args.eta_xtalk,
                                     unfinished_penalty=args.unfinished_penalty,
                                     gate_base_cx=args.gate_base_cx,
                                     gate_base_1q=args.gate_base_1q,
                                     shaping_gamma=args.shaping_gamma,
                                     eta_shape=args.eta_shape,
                                     alpha_ext=args.alpha_ext,
                                     no_progress_limit=_np,
                                     lookahead_features=args.lookahead_features,
                                      init_mapping=ep_init_mapping,
                                     lambda_layout=args.lambda_layout,
                                      fidelity_fn=build_fidelity_fn(
                                        args.fidelity_sim, noise_config,
                                        num_trajectories=args.traj_trajectories,
                                        seed=args.traj_seed,
                                        analytic_thermal=args.analytic_thermal,
                                        analytic_crosstalk=args.analytic_crosstalk,
                                        backend=args.sim_device,
                                     ) if args.reward_mode != "routing" else None,
                                     sabre_fid_map=ep_sabre_map,
                                     sref_override=ep_sref_override,
                                     edge_noise_features=args.edge_noise_features,
                                     beta_noise=args.beta_noise,
                                     w_err=(args.w_err * noise_f
                                            if args.w_err is not None else None),
                                     w_xt=(args.w_xt * noise_f
                                           if args.w_xt is not None else None),
                                     w_xt_swap=(args.w_xt_swap * noise_f
                                                if args.w_xt_swap is not None else None),
                                     pot_progress_b=args.pot_progress_b,
                                     pot_1q_reward=args.pot_1q_reward,
                                     lambda_budget=_ep_lb,
                                     sabre_swap_budget=ep_swap_budget,
                                      swap_price_scale=_ep_scale,
                                      step_cap_mult=args.step_cap_mult,
                                      clocked=args.clocked,
                                      max_ready=args.max_ready_2q,
                                      w_xt_launch=args.w_xt_launch,
                                      w_zz=args.w_zz,
                                      eta_ao=args.eta_ao,
                                      step_cap_factor=args.step_cap_factor)
                obs, _ = env.reset()
                if ep_demo:
                    # demo 回合：保留 mapping-phase 架构维度（obs 形状不变），
                    # 仅翻 phase 标志直接进入路由期（从 SABRE 布局精确对齐换手序列）
                    env.mapping_phase = False
                ep_total_reward = 0.0
                la_anchors_this_ep = 0

        with torch.no_grad():
            if args.clocked:
                last_logits, last_v_route, last_v_fid, _last_v_la = \
                    agent._forward_obs_split(obs)
                last_val = last_v_route + agent.lambda_v_fid * last_v_fid
            elif use_gnn:
                mask = torch.zeros(agent.num_edges + 1, dtype=torch.bool, device=agent.device)
                mask[:len(agent.coupling_map)] = True
                if env.mapping_phase and env._mapping_swaps >= args.mapping_min_swaps:
                    mask[agent.num_edges] = True
                last_logits, last_v_route, last_v_fid, _last_v_la = agent._forward_obs_split(
                    obs, action_mask=mask.unsqueeze(0))
                last_val = last_v_route + agent.lambda_v_fid * last_v_fid
            else:
                last_logits, last_v_route, last_v_fid, _last_v_la = agent._forward_obs_split(obs)
                last_val = last_v_route + agent.lambda_v_fid * last_v_fid
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
        else:
            norm_rew = np.array(ep_buffer["rew"], dtype=float)
            norm_term = np.array(ep_buffer["term_rew"], dtype=float)
        clip_return = args.clip_return if args.clip_return > 0 else None
        if args.gae_adaptive:
            T = len(norm_rew)
            ratio = np.arange(T) / max(T - 1, 1)
            lam_t = args.gae_lam_min + (args.gae_lam_max - args.gae_lam_min) * ratio
        else:
            lam_t = agent.lam

        dual_critic = args.variant in ("P2B", "P2C")
        if dual_critic:
            # 双通道：路由与保真度分开估计 advantage/return
            adv_route, ret_route = PPOAgent.compute_gae(
                norm_rew, ep_buffer["val_route"], ep_buffer["done"],
                bootstrap=last_v_route, gamma=agent.gamma, lam=lam_t,
                clip_return=clip_return,
            )
            adv_fid, ret_fid = PPOAgent.compute_gae(
                norm_term, ep_buffer["val_fid"], ep_buffer["done"],
                bootstrap=last_v_fid, gamma=agent.gamma, lam=lam_t,
                clip_return=clip_return,
            )
            alpha_fid = float(args.alpha_fid)
            adv = adv_route + alpha_fid * adv_fid
            ret = ret_route
        else:
            # 单通道（P1 / P2A 旧行为）：terminal fid 并入总奖励，一次 GAE
            norm_rew = norm_rew + norm_term
            adv, ret = PPOAgent.compute_gae(
                norm_rew, ep_buffer["val"], ep_buffer["done"],
                bootstrap=last_val, gamma=agent.gamma, lam=lam_t,
                clip_return=clip_return,
            )
            ret_fid = None
        train_batch = {
            "act": ep_buffer["act"],
            "logp": ep_buffer["logp"],
            "adv": adv,
            "ret": ret,
        }
        if args.la_beam > 0:
            train_batch["la_mask"] = ep_buffer["la_mask"]
            train_batch["la_act"] = ep_buffer["la_act"]
            train_batch["la_val"] = ep_buffer["la_val"]
        if args.sabre_demo_lambda > 0:
            train_batch["demo_mask"] = ep_buffer["demo_mask"]
            train_batch["demo_act"] = ep_buffer["demo_act"]
        if dual_critic:
            train_batch["ret_fid"] = ret_fid
        if args.clocked:
            train_batch["obs"] = ep_buffer["obs"]
        elif use_gnn:
            train_batch["graph_data"] = ep_buffer["graph_data"]
            train_batch["map_vec"] = ep_buffer["map_vec"]
            train_batch["progress"] = ep_buffer["progress"]
            train_batch["phase"] = ep_buffer["phase"]
            if "coupling_map" in ep_buffer and len(ep_buffer["coupling_map"]) > 0:
                train_batch["coupling_map"] = ep_buffer["coupling_map"]
            train_batch["sabre_feats"] = ep_buffer["sabre_feats"]
            if look_on and "look_feats" in ep_buffer:
                train_batch["look_feats"] = ep_buffer["look_feats"]
            if args.edge_noise_features and "noise_feats" in ep_buffer:
                train_batch["noise_feats"] = ep_buffer["noise_feats"]
            if "global_feats" in ep_buffer and len(ep_buffer["global_feats"]) > 0:
                train_batch["global_feats"] = ep_buffer["global_feats"]
            if "sabre_core_feats" in ep_buffer and len(ep_buffer["sabre_core_feats"]) > 0:
                train_batch["sabre_core_feats"] = ep_buffer["sabre_core_feats"]
        else:
            train_batch["obs"] = ep_buffer["obs"]
        if args.clocked:
            progress = total_steps / max(1, args.timesteps)
        if use_gnn:
            agent.gnn.train()
        if total_steps < args.freeze_first_steps:
            losses = {"pl": 0.0, "vl": 0.0, "vfl": 0.0, "ent": 0.0, "kl": 0.0,
                      "klp1": 0.0, "grad": 0.0}
        else:
            beta_kl = float(args.beta_kl) if args.variant == "P2C" else 0.0
            # λ_d 渐入：进度达 ramp 比例时爬满（0916 §5 v2 蒸馏）
            _lam_d = (args.la_distill_lambda
                      * (min(1.0, progress / args.la_distill_ramp)
                         if args.la_distill_ramp > 0 else 1.0))
            # λ_s 渐入：SABRE 示范蒸馏同 ramp 语义（E4）
            _lam_s = (args.sabre_demo_lambda
                      * (min(1.0, progress / args.sabre_demo_ramp)
                         if args.sabre_demo_ramp > 0 else 1.0))
            losses = agent.update(train_batch, epochs=args.epochs, beta_kl=beta_kl,
                                  la_distill_lambda=_lam_d,
                                  sabre_demo_lambda=_lam_s,
                                  global_feats_list=train_batch.get("global_feats"),
                                  sabre_core_feats_list=train_batch.get("sabre_core_feats"))
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
        if args.variant in ("P2B", "P2C"):
            parts.append(f"vfl={losses['vfl']:.3f}")
            parts.append(f"klp1={losses['klp1']:.4f}")
        if args.la_beam > 0:
            parts.append(f"vla={losses.get('vla', 0.0):.3f}")
            parts.append(f"agr={losses.get('agr', 0.0):.2f}")
        if ep_gps_log:
            parts.append(f"gps={np.mean(ep_gps_log[-20:]):.1f}")
        if os.environ.get("LA_DEBUG"):
            parts.append(f"[dbg eps={len(ep_rewards)} la_ep={la_anchors_this_ep} "
                         f"ep_step={env._episode_step} map_ph={env.mapping_phase}]")
        if ep_fids:
            avg_fid = np.mean(ep_fids[-20:])
            parts.append(f"fid={avg_fid:.4f}")
        if env.use_scheduler and ep_time_log:
            parts.append(f"time={np.mean(ep_time_log[-20:]):.1f}us")
            parts.append(f"xtalk={np.mean(ep_xtalk_log[-20:]):.3f}")
        if args.lambda_budget > 0 and ep_swap_ratio_log:
            swap_ratio = float(np.mean(ep_swap_ratio_log[-20:]))
            parts.append(f"sr={swap_ratio:.2f}")
            if swap_ratio > 1.10:
                print(f"  [G1 ALERT] swaps/SABRE={swap_ratio:.3f} > 1.10 —— 立即回滚检查！")
            elif swap_ratio > 1.05:
                print(f"  [G1 warn] swaps/SABRE={swap_ratio:.3f} > 1.05")
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
            "vfl": losses.get("vfl", ""),
            "vla": losses.get("vla", ""),
            "agr": losses.get("agr", ""),
            "klp1": losses.get("klp1", ""),
            "ent": losses["ent"],
            "kl": losses["kl"],
            "grad": losses["grad"],
            "fid": avg_fid if ep_fids else "",
            "time_us": avg_time,
            "xtalk": avg_xtalk,
            "idle": avg_idle,
            "par": avg_par,
            "swap_ratio": (float(np.mean(ep_swap_ratio_log[-20:]))
                           if ep_swap_ratio_log else ""),
            "gps": (float(np.mean(ep_gps_log[-20:]))
                    if ep_gps_log else ""),
            "dsl": losses.get("dsl", ""),
            "dalg": (round(float(np.mean(ep_demo_align[-20:])), 4)
                     if ep_demo_align else ""),
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
    # 修复：此处此前只打印不保存——metric 未超历史最优时 args.out 从未落盘
    agent.save(args.out)
    _metrics_fh.close()
    print(f"Checkpoints in {ckpt_dir}")
    print(f"Policy saved to {args.out}")
    if best_ema_metric > -1.0:
        print(f"EMA best: step={best_ema_step}  fid={best_ema_metric:.6f}")
        print(f"EMA best policy: {args.out.replace('.pt', '_ema_best.pt')}")


if __name__ == "__main__":
    main()
