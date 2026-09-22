"""
benchmark/nam_circs 中的电路 → PPO/SABRE 路由生成器。

用法:
    cd src
    PYTHONPATH=. python3 -m routing.rl.generate_routing \
        --model ../models/policy_ph2_v4.pt \
        --model-name ph2v4 \
        --circuit-dir ../benchmark/nam_circs \
        --topo ../traindata/topo/tianyan176_20q.json \
        --label-map ../traindata/topo/tianyan176_20q_labels.json \
        --max-num-qubits 20 \
        --out-dir ../benchmark/routed \
        --reward-mode noise_aware \
        --fidelity-sim trajectory_sched --traj-trajectories 16

    PYTHONPATH=. python3 -m routing.rl.generate_routing \
        --model ../models/policy_tianyan20q_laymix_l05_eta05.pt \
        --model-name l05 \
        --circuit-dir ../benchmark/nam_circs \
        --topo ../traindata/topo/tianyan176_20q.json \
        --label-map ../traindata/topo/tianyan176_20q_labels.json \
        --max-num-qubits 20 \
        --out-dir ../benchmark/routed \
        --reward-mode noise_aware \
        --fidelity-sim trajectory_sched --traj-trajectories 16
"""
import argparse, json, os, sys, time
from typing import Optional

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from qiskit.qasm2 import load as qasm2_load
from routing.graph.circuit_dag import CircuitDAG
from routing.rl.agent import PPOAgent
from routing.rl.env import RoutingEnv


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_qasm(path: str):
    """从 QASM2 文件加载为 Qiskit QuantumCircuit。"""
    return qasm2_load(filename=path)


def _remap_qasm_to_labels(phys_circuit, q_label_map: dict) -> str:
    """将路由后的物理电路导出为 QASM2 字符串。

    QASM2 规范要求小写标识符，Q 标签（Q12 等）仅在 initial_layout /
    final_layout 中使用。此处保留 q[0]-q[19] 物理索引。
    """
    from qiskit.qasm2 import dumps
    return dumps(phys_circuit)


def _compute_final_layout(env: RoutingEnv, q_label_map: dict, coupling_map: list) -> dict:
    """从 env.mapping 和 _swap_history 计算最终逻辑→物理 Q 标签布局。

    env.mapping[logical_index] = physical_index。
    遍历 routing 过程中的 SWAP action，跟踪每个逻辑比特最终所在物理位置。
    """
    mapping = list(env.mapping[:env.num_qubits])
    for action in env._swap_history:
        if action < len(coupling_map):
            p, q = coupling_map[action]
            inv = {phys: log for log, phys in enumerate(mapping)}
            lp, lq = inv.get(p), inv.get(q)
            # 与 env._apply_swap 一致：空端点 SWAP 会把逻辑比特搬到空位上，
            # 跳过会导致回放偏离真实映射（历史 bug 根因之一）。
            if lp is None and lq is None:
                continue
            if lp is None:
                mapping[lq] = p
            elif lq is None:
                mapping[lp] = q
            else:
                mapping[lp], mapping[lq] = mapping[lq], mapping[lp]
    return {str(i): q_label_map[mapping[i]] for i in range(len(mapping))}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="对 benchmark 电路生成 PPO/SABRE 映射和路由策略")
    parser.add_argument("--model", required=True, help="模型 checkpoint 路径")
    parser.add_argument("--edge-hidden", type=int, default=64,
                        help="edge_mlp 首层隐藏宽度（E17 容量升级：64→128）")
    parser.add_argument("--model-name", default="ppo",
                        help="模型名称（输出标签，如 l05、ph2v4）")
    parser.add_argument("--circuit-dir", required=True, help="QASM 电路目录")
    parser.add_argument("--topo", required=True, help="拓扑 JSON 路径")
    parser.add_argument("--label-map", required=True, help="Q 标签 sidecar JSON")
    parser.add_argument("--max-num-qubits", type=int, default=20)
    parser.add_argument("--max-num-edges", type=int, default=None)
    parser.add_argument("--out-dir", required=True, help="输出目录（每电路一个 JSON）")
    parser.add_argument("--out-json", default=None, help="汇总输出 JSON（可选）")
    parser.add_argument("--reward-mode", default="noise_aware",
                        choices=["routing", "noise_aware"])
    parser.add_argument("--fidelity-sim", default="trajectory_sched",
                        choices=["aer", "trajectory", "trajectory_sched", "trajectory_v2"])
    parser.add_argument("--traj-trajectories", type=int, default=16)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8,
                        help="torch CPU 线程数上限（小图推理多线程同步开销主导，默认 8）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-fidelity", action="store_true",
                        help="Skip fidelity computation (route only)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-circuits", type=int, default=None)
    parser.add_argument("--no-gnn", action="store_true")
    parser.add_argument("--beam-width", type=int, default=0,
                        help="Beam search width (0=argmax, 3/5 for beam)")
    parser.add_argument("--beam-vhead", type=str, default="auto",
                        choices=["auto", "route", "la"],
                        help="beam 打分价值头：auto=ckpt 含 critic_la 则用 la；"
                             "route=v_route+λv_fid；la=V_LA 多步价值头")
    parser.add_argument("--lambda-budget", type=float, default=0.0,
                        help="P1-a 推理版：SWAP 超预算后每颗惩罚（0=关，默认关=历史口径）。"
                             "使 beam 打分的 r_c 与训练口径一致（训练有预算锚、推理此前裸奔）")
    parser.add_argument("--budget-delta", type=float, default=1.05,
                        help="预算膨胀系数：budget = ceil(delta × SABRE swaps)")
    parser.add_argument("--swap-price-scale", type=float, default=1.0,
                        help="potential 模式 SWAP 边际价格缩放（与训练一致；"
                             "20260917 审计校准值 4.6）")
    parser.add_argument("--warm-start-sabre", action="store_true",
                        help="SABRE 初始布局 warm-start：跳过模型映射阶段，"
                             "直接从 SABRE 布局进入路由（布局 A/B 实验用）")
    parser.add_argument("--swap-cost", type=float, default=0.0,
                        help="Per-SWAP penalty (match training: ph2v4=0.5)")
    parser.add_argument("--eta-xtalk-par", type=float, default=1.0,
                        help="Crosstalk penalty weight (match training: ph2v4=0.05)")
    parser.add_argument("--lambda-fid-max", type=float, default=5.0,
                        help="Terminal fidelity reward weight (match training)")
    parser.add_argument("--lookahead-features", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="R5a look4 特征开关（旧模型评估须 --no-lookahead-features）")
    parser.add_argument("--edge-noise-features", action="store_true", default=False,
                        help="P0-a per-edge 噪声特征（须与训练一致）")
    parser.add_argument("--beta-noise", type=float, default=0.0,
                        help="P0-b 噪声加权距离 β（须与训练一致）")
    parser.add_argument("--w-err", type=float, default=0.0,
                        help="P0-c Φ E_err 权重（beam 打分用，须与训练一致）")
    parser.add_argument("--w-xt", type=float, default=0.0,
                        help="P0-c Φ X(s) 权重（beam 打分用）")
    parser.add_argument("--w-xt-swap", type=float, default=0.0,
                        help="P0-c per-swap 串扰价权重（beam 打分用）")
    parser.add_argument("--pot-progress-b", type=float, default=0.045,
                        help="P0-d progress 奖励 B（beam 打分用）")
    parser.add_argument("--pot-1q-reward", dest="pot_1q_reward",
                        action="store_false", default=True,
                        help="P0-d 1Q/measure 门 progress 置零（beam 打分用）")
    parser.add_argument("--reward-potential", action="store_true", default=False,
                        help="potential 模式奖励（beam 打分用，须与训练一致）")
    parser.add_argument("--shaping-gamma", type=float, default=None,
                        help="R3 Φ shaping γ（beam 打分 reward_c 需与训练同构——"
                             "V(s') 学到的价值含 shaping 流，缺失会带偏 beam 搜索）")
    parser.add_argument("--eta-shape", type=float, default=0.3)
    parser.add_argument("--alpha-ext", type=float, default=0.5)
    args = parser.parse_args()

    # 小图推理：限制 torch CPU 线程数，避免多线程同步开销主导（实测 2x+）
    torch.set_num_threads(max(1, args.torch_threads))

    # ── 加载拓扑 ──
    from routing.rl.eval_policy import load_topo, build_fidelity_fn
    config, hw, coupling_map = load_topo(args.topo)
    print(f"Topo: {args.topo} ({hw.num_qubits} qubits, {len(coupling_map)} edges)")

    # ── 加载标签映射 ──
    with open(args.label_map) as f:
        label_data = json.load(f)
    q_label_map = {int(k): v for k, v in label_data["from_0_19"].items()}
    from_0_19 = label_data["from_0_19"]
    print(f"Label map: {args.label_map}")

    # ── 加载模型 ──
    from routing.gnn.encoder import SubGNN
    use_gnn = not args.no_gnn
    shared_gnn = SubGNN(subgraph="full") if use_gnn else None
    if shared_gnn:
        shared_gnn.eval()

    # 用一个 dummy env 获取 obs_dim
    dummy_dag = CircuitDAG([], 1)
    dummy_env = RoutingEnv(
        dummy_dag, hw, coupling_map, reward_mode="routing",
        max_episode_steps=1, seed=0,
        gnn=shared_gnn, use_gnn=use_gnn,
        max_num_qubits=args.max_num_qubits,
        max_num_edges=args.max_num_edges,
        lookahead_features=(True if args.lookahead_features is None
                            else args.lookahead_features),
        edge_noise_features=args.edge_noise_features,
        beta_noise=args.beta_noise,
    )
    n_edges = args.max_num_edges or len(coupling_map)
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
    agent = PPOAgent(
        obs_dim=int(np.prod(dummy_env.observation_space.shape)),
        action_dim=n_edges + 1,  # +1 for commit action
        device=args.device,
        gnn=shared_gnn,
        num_qubits=args.max_num_qubits,
        num_edges=n_edges,
        coupling_map=coupling_map,
        with_commit=True,
        edge_feat_dim=(getattr(dummy_env, "_edge_feat_dim", None)
                       if use_gnn else None),
        with_la_head=_has_la,
        edge_hidden=args.edge_hidden,
    )
    agent.load(args.model)
    agent.ac.eval()
    if agent.gnn is not None:
        agent.gnn.eval()
    print(f"Model: {args.model} ({args.model_name})")

    # ── 保真度函数 ──
    traj_seed = args.seed
    fid_fn = build_fidelity_fn(
        args.fidelity_sim, config, args.traj_trajectories, traj_seed)

    # ── 扫描电路 ──
    qasm_files = sorted(f for f in os.listdir(args.circuit_dir)
                        if f.endswith(".qasm"))
    if args.max_circuits:
        qasm_files = qasm_files[:args.max_circuits]
    print(f"Circuits: {len(qasm_files)} files in {args.circuit_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    skipped = []
    _budget_cache = {}  # fname -> SABRE swaps（P1-a 推理版预算锚缓存）

    for fname in qasm_files:
        fpath = os.path.join(args.circuit_dir, fname)
        try:
            qc = _load_qasm(fpath)
        except Exception as e:
            print(f"  [SKIP] {fname}: failed to load: {e}")
            skipped.append({"circuit": fname, "reason": f"load_error: {e}"})
            continue

        num_logical = qc.num_qubits
        if num_logical > args.max_num_qubits:
            print(f"  [SKIP] {fname}: {num_logical} qubits > max {args.max_num_qubits}")
            skipped.append({"circuit": fname, "reason": f"{num_logical}q > {args.max_num_qubits}q"})
            continue

        dag = CircuitDAG.from_circuit(qc)

        # ── 路由 ──
        t0 = time.perf_counter()
        # SABRE 布局 warm-start（布局 A/B 实验）：SABRE 求布局，模型只做路由
        ws_layout = None
        if args.warm_start_sabre:
            from routing.routing import sabre_route
            _, ws_info = sabre_route(qc, config, swap_trials=20, seed=args.seed)
            ws_layout = ws_info.get('initial_layout')
        # P1-a 推理版：SABRE SWAP 预算锚——超预算后每颗额外 SWAP 罚 lambda_budget，
        # 使 beam 打分的 r_c 与训练口径一致（doc/20260917训练方案.md §2-A1）
        ep_budget = None
        if args.lambda_budget > 0:
            from routing.routing import sabre_route
            if fname not in _budget_cache:
                _, sinfo = sabre_route(qc, config, swap_trials=20, seed=args.seed)
                _budget_cache[fname] = int(sinfo.get("num_swaps", 0) or 0)
            s_sw = _budget_cache[fname]
            if s_sw > 0:
                ep_budget = int(np.ceil(args.budget_delta * s_sw))
        env = RoutingEnv(
            dag, hw, coupling_map, reward_mode=args.reward_mode,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
            gnn=agent.gnn, use_gnn=agent.gnn is not None,
            noise_config=None,
            max_num_qubits=args.max_num_qubits,
            max_num_edges=n_edges,
            mapping_phase=True,
            init_mapping=ws_layout,
            fidelity_fn=None,
            use_scheduler=False,
            swap_cost=args.swap_cost,
            eta_xtalk_par=args.eta_xtalk_par,
            lambda_fid=args.lambda_fid_max if args.reward_mode != 'routing' else 0.0,
            lambda_budget=args.lambda_budget,
            sabre_swap_budget=ep_budget,
            swap_price_scale=args.swap_price_scale,
            lookahead_features=(True if args.lookahead_features is None
                                else args.lookahead_features),
            edge_noise_features=args.edge_noise_features,
            beta_noise=args.beta_noise,
            w_err=args.w_err, w_xt=args.w_xt, w_xt_swap=args.w_xt_swap,
            pot_progress_b=args.pot_progress_b, pot_1q_reward=args.pot_1q_reward,
            reward_potential=args.reward_potential,
            shaping_gamma=args.shaping_gamma,
            eta_shape=args.eta_shape, alpha_ext=args.alpha_ext,
        )
        obs, _ = env.reset()
        if ws_layout is not None:
            # 覆盖运行期映射阶段（保留 enable_mapping_phase 以不丢 phase 特征，
            # 与 eval_policy 的 A1 模式同构）
            env.mapping_phase = False
            obs = env._obs()

        if args.beam_width > 0:
            # Beam search: inline 1-step lookahead
            bw = args.beam_width
            done, truncated = False, False
            step_count = 0
            while not done and not truncated:
                with torch.no_grad():
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
                    masked_logits[~mask[0]] = -1e9
                    k = min(bw, mask[0].sum().item())
                    topk_scores, topk_indices = masked_logits.topk(k)

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
                        # vhead='la' 用 V_LA 多步价值头（训练期 beam expectimax 训得）
                        if vhead == 'la':
                            _, values = agent._forward_obs_batch_vla(np.stack(clone_obs_list))
                        else:
                            _, values = agent._forward_obs_batch(np.stack(clone_obs_list))
                        # P1-a 推理版：父状态势函数基准（每步一次）
                        phi_parent = env._phi() if (ep_budget is not None
                                                    and args.lambda_budget > 0) else None
                        for (clone, a, reward_c, done_c, truncated_c), clone_obs, v in zip(
                                clones, clone_obs_list, values):
                            score = reward_c if (done_c or truncated_c) else reward_c + agent.gamma * v.item()
                            # P1-a 推理版（排序有效）：超预算下既未解锁门、也未改善
                            # 势函数（距离）的候选按超支深度受罚——常数级惩罚不改变
                            # 同步内排序，必须与候选特异进度交互；进度用 Φ 改进
                            # 而非仅门解锁（深电路上门解锁稀疏、Φ 改进稠密）
                            if (ep_budget is not None and not done_c
                                    and not truncated_c):
                                over_by = clone._swap_counter - ep_budget
                                if over_by > 0:
                                    prog = ((len(clone.executed)
                                             - len(env.executed)) > 0
                                            or (clone._phi() > phi_parent + 1e-9))
                                    if not prog:
                                        score -= args.lambda_budget * over_by
                            if score > best_score:
                                best_score = score
                                best_action = a
                                best_clone_obs = clone_obs
                    else:
                        clone_obs_list = [c._obs() for c, *_ in clones]
                        _, values = agent._forward_obs_batch(np.stack(clone_obs_list))
                        for (clone, a, reward_c, done_c, truncated_c), clone_obs, v in zip(
                                clones, clone_obs_list, values):
                            score = reward_c if (done_c or truncated_c) else reward_c + agent.gamma * v.item()
                            if score > best_score:
                                best_score = score
                                best_action = a
                                best_clone_obs = clone_obs

                obs, reward, done, truncated, info = env.step(best_action, compute_obs=False)
                if best_clone_obs is not None:
                    obs = best_clone_obs
                step_count += 1
        else:
            # Argmax
            done, truncated = False, False
            while not done and not truncated:
                with torch.no_grad():
                    mask = None
                    if agent.gnn is not None and hasattr(env, 'get_deadlock_mask'):
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
                    if mask is not None and int(mask[0].sum().item()) == 0:
                        # 全部动作被掩码（病态状态）：安全退出而非任选动作
                        truncated = True
                        break
                    action = logits.argmax(-1).item()
                obs, reward, done, truncated, info = env.step(action)

        t_route_end = time.perf_counter()

        # 保真度：路由阶段不计算，路由完成后 post-hoc 计算
        fidelity = None
        if done and args.fidelity_sim and not args.no_fidelity:
            from routing.rl.eval_policy import phys_fidelity
            try:
                fidelity = phys_fidelity(
                    env._phys_circuit, config, args.fidelity_sim,
                    num_trajectories=args.traj_trajectories, seed=traj_seed)
            except Exception:
                fidelity = None

        wall_time_ms = (time.perf_counter() - t0) * 1000

        # 有效初始布局 = 映射阶段 commit 后、物理线路第一条门执行时刻的映射。
        # 与 routed_qasm 严格对应：initial_layout + 线路内 SWAP 逐一追踪 == final_layout。
        initial_layout = {str(i): env._effective_initial_mapping[i]
                          for i in range(num_logical)}
        final_layout = {str(i): env.mapping[i]
                        for i in range(num_logical)}
        routed_qasm = _remap_qasm_to_labels(env._phys_circuit, q_label_map)

        result = {
            "circuit": fname,
            "model": args.model_name,
            "num_logical_qubits": num_logical,
            "num_physical_qubits": hw.num_qubits,
            "topology": os.path.basename(args.topo).replace(".json", ""),
            "initial_layout": initial_layout,
            "routed_qasm": routed_qasm,
            "num_swaps": env._swap_counter,
            "mapping_swaps": env._mapping_swaps,
            "completed": done,
            "fidelity": fidelity,
            "final_layout": final_layout,
            "wall_time_ms": round(wall_time_ms, 1),
            "from_0_19": from_0_19,
            "sabre_budget": ep_budget,
            "lambda_budget": args.lambda_budget,
        }
        results.append(result)

        # 写 per-circuit JSON
        out_path = os.path.join(args.out_dir, fname.replace(".qasm", ".json"))
        with open(out_path, "w") as f:
            json.dump(result, f, indent=1, ensure_ascii=False)

        tag = "OK" if done else "TRUNC"
        fid_str = f"{fidelity:.4f}" if fidelity is not None else "--"
        print(f"  {tag} {fname}: {num_logical}q, "
              f"swaps={env._swap_counter}, map_swaps={env._mapping_swaps}, "
              f"fid={fid_str}, {wall_time_ms:.0f}ms")

    # ── 汇总 ──
    print(f"\n=== Summary ({args.model_name}) ===")
    print(f"Routed: {len(results)}/{len(qasm_files)}")
    print(f"Skipped: {len(skipped)}")
    for s in skipped:
        print(f"  - {s['circuit']}: {s['reason']}")

    if results:
        fids = [r["fidelity"] for r in results if r["fidelity"] is not None]
        swaps = [r["num_swaps"] for r in results]
        print(f"Fidelity: mean={np.mean(fids):.4f}, min={np.min(fids):.4f}, "
              f"max={np.max(fids):.4f}" if fids else "Fidelity: N/A")
        print(f"SWAPs: mean={np.mean(swaps):.1f}, min={np.min(swaps)}, "
              f"max={np.max(swaps)}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"model": args.model_name, "results": results,
                        "skipped": skipped}, f, indent=1, ensure_ascii=False)
        print(f"\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()
