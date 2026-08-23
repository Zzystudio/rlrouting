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
    """λ_fid 退火调度：前 warmup 比例为 0，之后线性增长到 max_val。"""
    if progress < warmup:
        return 0.0
    return max_val * (progress - warmup) / (1.0 - warmup)


# ---------------------------------------------------------------------------
#  create_env helper
# ---------------------------------------------------------------------------
def create_env(dag, hw, coupling_map, reward_mode, max_episode_steps, random_init, seed, gnn=None, use_gnn=True, max_num_edges=None, max_num_qubits=None, noise_config=None, lambda_fid=None, eta_dist=None, mapping_budget=None, mapping_phase=True, fidelity_fn=None):
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
    kw["mapping_phase"] = mapping_phase
    return RoutingEnv(**kw)


def build_fidelity_fn(fidelity_sim: str, noise_config, num_trajectories: int = 64, seed=None):
    """按 --fidelity-sim 构造 env 终端保真度函数；routing 模式或 aer 模式返回 None。

    aer: 使用 env 内置 NoiseSimulator（density_matrix + counts overlap，n<=12）。
    trajectory: 使用轨迹状态向量模拟器（O(2^n) 内存，20q+ 可用）。
    """
    if fidelity_sim == "trajectory":
        from sim.trajectory_sim import make_trajectory_fidelity_fn
        return make_trajectory_fidelity_fn(noise_config, num_trajectories=num_trajectories, seed=seed)
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
    parser.add_argument("--lambda-fid-warmup", type=float, default=0.5,
                        help="lambda_fid 退火比例：前 N%% 步 lambda_fid=0")
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
                        choices=["aer", "trajectory"],
                        help="终端保真度模拟器: aer=density_matrix/counts (小比特数), "
                             "trajectory=轨迹状态向量 (O(2^n) 内存，20q+ 必选)")
    parser.add_argument("--traj-trajectories", type=int, default=16,
                        help="轨迹模拟器采样条数（越大方差越小，训练越慢）")
    parser.add_argument("--traj-seed", type=int, default=None,
                        help="轨迹模拟器随机种子（默认 None=不可复现）")
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
                     fidelity_fn=build_fidelity_fn(
                         args.fidelity_sim, noise_config,
                         num_trajectories=args.traj_trajectories, seed=args.traj_seed,
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

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # --- checkpoint 初始化 ---
    ckpt_dir = args.checkpoint_dir or os.path.join(os.path.dirname(args.out) or ".", "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    _metrics_path = os.path.join(ckpt_dir, "metrics.csv")
    _metrics_fields = ["step", "reward", "swaps", "map_swaps", "trunc_pct", "pl", "vl", "ent", "kl", "grad", "fid"]
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
    ep_buffer = {"act": [], "logp": [], "val": [], "rew": [], "done": []}
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

    total_steps = resume_step
    best_metric = resume_best if resume_best > -1.0 else -1.0
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

            ep_buffer["act"].append(action)
            ep_buffer["logp"].append(logp)
            ep_buffer["val"].append(val)
            ep_buffer["rew"].append(reward)
            ep_buffer["done"].append(episode_end)

            obs = next_obs
            total_steps += 1

            if episode_end:
                ep_rewards.append(ep_total_reward)
                if truncated:
                    ep_truncated += 1
                else:
                    ep_completed += 1
                if "fidelity" in info:
                    ep_fids.append(info["fidelity"])
                ep_swaps_log.append(info.get("num_swaps", 0))
                ep_map_swaps_log.append(info.get("mapping_swaps", 0))

                # Per-topology tracking
                topo_steps[topo_idx] += env._episode_step
                topo_episodes[topo_idx] += 1
                if truncated:
                    topo_trunc[topo_idx] += 1
                else:
                    topo_completed[topo_idx] += 1
                topo_swaps_lists[topo_idx].append(info.get("num_swaps", 0))
                topo_rewards_lists[topo_idx].append(ep_total_reward)
                if "fidelity" in info:
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
                new_dag = pick_circuit(args.data_dir, split_key, seed=args.seed + total_steps,
                                       split_prefix=split_prefix,
                                       split_map=split_map,
                                       max_qubits=topo_qubits[topo_idx])
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
                cur_lambda_fid = lambda_fid_schedule(
                    progress, args.lambda_fid_warmup, args.lambda_fid_max
                ) if args.reward_mode != "routing" else None
                env = create_env(new_dag, hw, coupling_map, args.reward_mode,
                                 args.max_episode_steps, args.random_init,
                                 args.seed + total_steps,
                                 gnn=shared_gnn, use_gnn=use_gnn,
                                 max_num_edges=max_edges,
                                 max_num_qubits=args.max_num_qubits,
                                 noise_config=noise_config if args.reward_mode != "routing" else None,
                                 lambda_fid=cur_lambda_fid,
                                 eta_dist=args.eta_dist,
                                 mapping_budget=args.mapping_budget,
                                 mapping_phase=args.mapping_phase,
                                 fidelity_fn=build_fidelity_fn(
                                     args.fidelity_sim, noise_config,
                                     num_trajectories=args.traj_trajectories,
                                     seed=args.traj_seed,
                                 ) if args.reward_mode != "routing" else None)
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
        losses = agent.update(train_batch, epochs=args.epochs)
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

        cycle_idx += 1

    # --- final checkpoint ---
    last_path = os.path.join(ckpt_dir, "last.pt")
    agent.save_checkpoint(last_path, extra_state={"step": total_steps, "best_metric": best_metric})
    _metrics_fh.close()
    print(f"Checkpoints in {ckpt_dir}")
    print(f"Policy saved to {args.out}")


if __name__ == "__main__":
    main()
