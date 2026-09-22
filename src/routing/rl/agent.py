# ============================================================================
# agent.py
# 紧凑自包含的 PPO（Proximal Policy Optimization）实现：
# Actor-Critic + GAE + 截断代理目标。不依赖外部 RL 库。
# ============================================================================

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..gnn.encoder import SubGNN
from .env import _LOOKAHEAD_FEAT_DIM, _SABRE_CORE_FEAT_DIM


class ActorCritic(nn.Module):
    """共享躯干 + 策略头 + 价值头。"""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x, action_mask=None):
        h = self.shared(x)
        logits = self.actor(h)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        value = self.critic(h).squeeze(-1)
        return logits, value


class EdgeActorCritic(nn.Module):
    """Per-edge 动作打分 + 注意力池化价值头。

    edge_mlp 对所有候选 SWAP 边共享参数，输入 e_{pq} = [h_p, h_q, h_p-h_q]。
    价值头用软注意力池化边特征后 + mapping + progress + phase。
    with_commit=True 时追加一个 commit logit（映射阶段提交动作）。
    """

    def __init__(self, edge_feat_dim: int, num_edges: int, num_qubits: int,
                 with_commit: bool = True, with_la_head: bool = False,
                 edge_hidden: int = 64):
        super().__init__()
        self.num_edges = num_edges
        self.with_commit = with_commit
        self.with_la_head = with_la_head
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, edge_hidden // 2),
            nn.ReLU(),
            nn.Linear(edge_hidden // 2, 1),
        )
        self.edge_score = nn.Linear(edge_feat_dim, 1)
        if with_commit:
            self.commit_head = nn.Linear(edge_feat_dim, 1)
        critic_in = edge_feat_dim + num_qubits + (2 if with_commit else 1)
        self.critic_route = nn.Sequential(
            nn.Linear(critic_in, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # 保真度价值头：zero-init（加载旧 checkpoint 后 V_fid ≡ 0，行为逐位不变；
        # P2 训练起步时 fidelity 信息平滑注入）
        self.critic_fid = nn.Sequential(
            nn.Linear(critic_in, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        for p in self.critic_fid.parameters():
            nn.init.zeros_(p)
        # V_LA lookahead 价值头：zero-init（训练期 beam expectimax 目标专用；
        # 老 checkpoint 无此权重，strict=False 加载后 V_LA≡0，行为逐位不变）
        if with_la_head:
            self.critic_la = nn.Sequential(
                nn.Linear(critic_in, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            for p in self.critic_la.parameters():
                nn.init.zeros_(p)

    def forward(self, edge_feats, map_vec, progress, phase=None, action_mask=None):
        B, E, D = edge_feats.shape
        scores = self.edge_mlp(edge_feats).squeeze(-1)
        attn_raw = self.edge_score(edge_feats)

        if action_mask is not None:
            edge_mask = action_mask[..., :E]
            scores = scores.masked_fill(~edge_mask, -1e9)
            attn_raw = attn_raw.masked_fill(~edge_mask.unsqueeze(-1), -1e9)

        attn_w = torch.softmax(attn_raw, dim=1)
        pooled = (edge_feats * attn_w).sum(dim=1)

        if self.with_commit:
            logits = torch.cat([scores, self.commit_head(pooled)], dim=-1)
        else:
            logits = scores
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        if phase is None:
            v_in = torch.cat([pooled, map_vec, progress], dim=-1)
        else:
            v_in = torch.cat([pooled, map_vec, progress, phase], dim=-1)
        v_route = self.critic_route(v_in).squeeze(-1)
        v_fid = self.critic_fid(v_in).squeeze(-1)
        if self.with_la_head:
            v_la = self.critic_la(v_in).squeeze(-1)
        else:
            v_la = torch.zeros_like(v_route)
        return logits, v_route, v_fid, v_la


class RewardNormalizer:
    """EMA running statistics for reward normalization."""

    def __init__(self, eps: float = 1e-8, alpha: float = 0.01):
        self.mean = 0.0
        self.var = 1.0
        self.eps = eps
        self.alpha = alpha

    def update(self, x):
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x)) if len(x) > 1 else 1.0
        self.mean = (1 - self.alpha) * self.mean + self.alpha * batch_mean
        self.var = (1 - self.alpha) * self.var + self.alpha * batch_var

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + self.eps)


class PPOAgent:
    """PPO 智能体，可选 GNN 联合训练。"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.1,
        device: str = "cpu",
        gnn: Optional[SubGNN] = None,
        num_qubits: Optional[int] = None,
        num_edges: Optional[int] = None,
        coupling_map: Optional[list] = None,
        with_commit: bool = True,
        lambda_v_fid: float = 1.0,
        edge_feat_dim: Optional[int] = None,
        with_la_head: bool = False,
        la_vf_coef: float = 0.1,
        la_raw_target: bool = False,
        edge_hidden: int = 64,
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.device = device
        self.with_commit = with_commit
        self.with_la_head = with_la_head
        self.la_vf_coef = la_vf_coef
        self.la_raw_target = la_raw_target
        self.lambda_v_fid = lambda_v_fid
        self.teacher = None          # π_P1 冻结教师（KL policy preservation 用）
        self.teacher_gnn = None

        self.gnn = gnn
        self.num_qubits = num_qubits
        self.num_edges = num_edges
        self.coupling_map = coupling_map
        self.rew_norm = RewardNormalizer()
        self.term_norm = RewardNormalizer()

        params = []
        if gnn is not None:
            self.gnn = gnn
            # edge_feat_dim 必须与 env 的 per-edge 特征布局一致：
            # out*3 + SABRE5 + look4（env 恒发）+ noise5（P0-a 开启时）。
            # 旧默认 out*3+5 仅作向后兼容（R5a-era 代码曾因未同步该值
            # 导致 rollout obs 切片错位——行为/更新策略不一致的存量 bug）。
            self.edge_feat_dim = edge_feat_dim if edge_feat_dim is not None \
                else self.gnn.encoder.out_dim * 3 + 5
            self.gnn.to(device)
            self.ac = EdgeActorCritic(self.edge_feat_dim, num_edges, num_qubits,
                                      with_commit=with_commit,
                                      with_la_head=with_la_head,
                                      edge_hidden=edge_hidden).to(device)
            params += list(self.gnn.parameters())
        else:
            self.ac = ActorCritic(obs_dim, action_dim).to(device)
        params += list(self.ac.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.ema = None

    def init_ema(self, decay=0.999):
        self.ema = EMAModel(self.ac, decay=decay)

    def update_ema(self):
        if self.ema is not None:
            self.ema.update(self.ac)

    def apply_ema(self):
        if self.ema is not None:
            self.ema.apply_shadow(self.ac)

    def restore_from_ema(self):
        if self.ema is not None:
            self.ema.restore(self.ac)

    # ---- 交互 ----
    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False,
            deadlock_mask=None, mapping_phase: bool = False):
        mask = None
        if isinstance(self.ac, EdgeActorCritic):
            n = self.num_edges + 1 if self.with_commit else self.num_edges
            mask = torch.zeros(n, dtype=torch.bool, device=self.device)
            mask[:len(self.coupling_map)] = True
            if deadlock_mask is not None:
                for i in range(min(len(deadlock_mask), len(mask))):
                    if deadlock_mask[i]:
                        mask[i] = False
            if self.with_commit:
                mask[self.num_edges] = mapping_phase
        else:
            action_dim = self.ac.actor.out_features
            mask = torch.zeros(action_dim, dtype=torch.bool, device=self.device)
            mask[:len(self.coupling_map)] = True
            if deadlock_mask is not None:
                for i in range(min(len(deadlock_mask), len(mask))):
                    if deadlock_mask[i]:
                        mask[i] = False
            if self.with_commit and action_dim > len(self.coupling_map):
                mask[-1] = mapping_phase
        logits, v_route, v_fid, _v_la = self._forward_obs_split(
            obs, action_mask=mask.unsqueeze(0) if mask is not None else None)
        value = v_route + self.lambda_v_fid * v_fid
        if deterministic:
            action = logits[0].argmax(-1).item()
            logp = 0.0
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action).item()
        action_i = action.item() if torch.is_tensor(action) else int(action)
        return int(action_i), logp, float(value.item()), \
            float(v_route.item()), float(v_fid.item())

    @torch.no_grad()
    def _forward_obs(self, obs, action_mask=None):
        """对外接口：返回 (logits, v_route + λ_V·v_fid)（beam/rollout 兼容）。"""
        logits, v_route, v_fid, _v_la = self._forward_obs_split(obs, action_mask=action_mask)
        return logits, v_route + self.lambda_v_fid * v_fid

    @torch.no_grad()
    def _forward_obs_batch(self, obs_batch, action_masks=None):
        """批量推理：obs_batch (K, obs_dim) → (logits (K, n_a), value (K,))。

        用于 beam search 的 K 个候选 V(s') 评估，单次前向替代 K 次。
        """
        logits, v_route, v_fid, _v_la = self._forward_obs_batch_full(
            obs_batch, action_masks=action_masks)
        return logits, v_route + self.lambda_v_fid * v_fid

    @torch.no_grad()
    def _forward_obs_batch_vla(self, obs_batch, action_masks=None):
        """批量 V_LA 评估（训练期 beam expectimax 叶节点）：(logits, v_la)。"""
        logits, _v_route, _v_fid, v_la = self._forward_obs_batch_full(
            obs_batch, action_masks=action_masks)
        return logits, v_la

    @torch.no_grad()
    def _forward_obs_batch_full(self, obs_batch, action_masks=None):
        """批量推理全量版：obs_batch (K, obs_dim) → (logits, v_route, v_fid, v_la)。"""
        obs_batch = np.asarray(obs_batch)
        K = obs_batch.shape[0]
        eff_dim = self.edge_feat_dim
        n_ef = self.num_edges * eff_dim
        ef = torch.tensor(obs_batch[:, :n_ef], dtype=torch.float32,
                          device=self.device).reshape(K, self.num_edges, eff_dim)
        mv = torch.tensor(obs_batch[:, n_ef:n_ef + self.num_qubits],
                          dtype=torch.float32, device=self.device)
        if self.with_commit:
            pg = torch.tensor(obs_batch[:, n_ef + self.num_qubits:n_ef + self.num_qubits + 1],
                              dtype=torch.float32, device=self.device)
            ph = torch.tensor(obs_batch[:, n_ef + self.num_qubits + 1:n_ef + self.num_qubits + 2],
                              dtype=torch.float32, device=self.device)
            logits, v_route, v_fid, v_la = self.ac(ef, mv, pg, ph, action_mask=action_masks)
        else:
            pg = torch.tensor(obs_batch[:, -1:], dtype=torch.float32, device=self.device)
            logits, v_route, v_fid, v_la = self.ac(ef, mv, pg, None, action_mask=action_masks)
        return logits, v_route, v_fid, v_la

    @torch.no_grad()
    def _forward_obs_split(self, obs, action_mask=None):
        """返回 (logits, v_route, v_fid, v_la) 四值（训练双通道 GAE + LA 用）。"""
        if isinstance(self.ac, EdgeActorCritic):
            eff_dim = self.edge_feat_dim
            n_ef = self.num_edges * eff_dim
            ef = torch.tensor(obs[:n_ef], dtype=torch.float32, device=self.device).reshape(1, self.num_edges, eff_dim)
            mv = torch.tensor(obs[n_ef:n_ef + self.num_qubits], dtype=torch.float32, device=self.device).unsqueeze(0)
            if self.with_commit:
                pg = torch.tensor(obs[n_ef + self.num_qubits:n_ef + self.num_qubits + 1], dtype=torch.float32, device=self.device).unsqueeze(0)
                ph = torch.tensor(obs[n_ef + self.num_qubits + 1:n_ef + self.num_qubits + 2], dtype=torch.float32, device=self.device).unsqueeze(0)
                return self.ac(ef, mv, pg, ph, action_mask=action_mask)
            pg = torch.tensor(obs[-1:], dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.ac(ef, mv, pg, None, action_mask=action_mask)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.ac(obs_t, action_mask=action_mask)
        return logits, value, torch.zeros_like(value), torch.zeros_like(value)

    # ---- teacher (π_P1 冻结，KL policy preservation) ----
    def load_teacher(self, path: str):
        """加载 Phase 1 检查点为冻结教师（仅 actor 用于 KL 约束）。

        E13：teacher 按 checkpoint 自身的 edge_feat_dim 构建（旧架构，如 LA287d
        158 维），student 可能更大（261 维，含批机会/全局特征）——_teacher_logits
        已按旧布局喂数据（look 前 4 维、无 global），teacher 输入保持对齐即可。
        """
        teacher_gnn = None
        if self.gnn is not None:
            teacher_gnn = SubGNN(subgraph="full")
            teacher_gnn.to(self.device)
            teacher_gnn.eval()
        state = torch.load(path, map_location=self.device, weights_only=False)
        teacher_ef = self.edge_feat_dim
        if "ac" in state:
            ref_w = state["ac"].get("edge_mlp.0.weight")
            if ref_w is not None:
                teacher_ef = ref_w.shape[-1]
        teacher = EdgeActorCritic(teacher_ef, self.num_edges,
                                  self.num_qubits, with_commit=self.with_commit)
        teacher.to(self.device)
        teacher.eval()
        if "ac" in state:
            ac_state = state["ac"]
            if any(k.startswith("critic.") for k in ac_state):
                ac_state = {k.replace("critic.", "critic_route.", 1): v
                            for k, v in ac_state.items()}
            teacher.load_state_dict(ac_state, strict=False)
            if teacher_gnn is not None and "gnn" in state:
                teacher_gnn.load_state_dict({k: v.to(self.device)
                                             for k, v in state["gnn"].items()})
        else:
            teacher.load_state_dict(state, strict=False)
        for p in teacher.parameters():
            p.requires_grad_(False)
        if teacher_gnn is not None:
            for p in teacher_gnn.parameters():
                p.requires_grad_(False)
        self.teacher = teacher
        self.teacher_gnn = teacher_gnn
        return teacher

    def _teacher_logits(self, graph_data_list, map_vec_list, progress_list,
                        coupling_maps, sabre_feats_list, phase_list,
                        look_feats_list=None, noise_feats_list=None):
        """用冻结 π_P1 计算 teacher logits（无梯度）。

        边特征布局必须与 agent 完全一致（sabre5 + look4 + noise5）——
        teacher 网络按 agent 的 edge_feat_dim 构建，缺维度会形状错位
        （P0-a noise5 特征配置下首次暴露）。
        E13：student 的 look 特征扩到 6 维（+批机会 2 维）、追加 global101 维，
        但 teacher 冻结于 158 维时代（look4、无 global）——此处 look 取前 4 维、
        不拼接 global，teacher 输入保持 158 对齐。"""
        with torch.no_grad():
            all_ef, all_mv, all_pg, all_ph = [], [], [], []
            for i, (gd, mv, pg) in enumerate(zip(graph_data_list, map_vec_list,
                                                  progress_list)):
                qubit_h = self.teacher_gnn.node_embeddings(gd)
                cmap = coupling_maps[i]
                n_local = len(cmap)
                ef = self._build_edge_feats_from_h(qubit_h.to(self.device), cmap)
                if sabre_feats_list is not None:
                    sf = torch.tensor(sabre_feats_list[i], dtype=torch.float32,
                                      device=self.device).reshape(n_local, 5)
                    ef = torch.cat([ef, sf], dim=-1)
                if look_feats_list is not None:
                    lf_all = torch.tensor(look_feats_list[i], dtype=torch.float32,
                                          device=self.device).reshape(n_local, _LOOKAHEAD_FEAT_DIM)
                    ef = torch.cat([ef, lf_all[:, :4]], dim=-1)  # teacher 只见旧 4 维
                if noise_feats_list is not None:
                    nf = torch.tensor(noise_feats_list[i], dtype=torch.float32,
                                      device=self.device).reshape(n_local, 5)
                    ef = torch.cat([ef, nf], dim=-1)
                if ef.shape[0] < self.num_edges:
                    pad = torch.zeros(self.num_edges - ef.shape[0], ef.shape[1],
                                      device=ef.device, dtype=ef.dtype)
                    ef = torch.cat([ef, pad], dim=0)
                mv_raw = torch.tensor(mv, dtype=torch.float32, device=self.device)
                mv_t = torch.zeros(1, self.num_qubits, dtype=torch.float32, device=self.device)
                n_copy = min(mv_raw.shape[0], self.num_qubits)
                mv_t[0, :n_copy] = mv_raw[:n_copy]
                pg_t = torch.tensor(pg, dtype=torch.float32, device=self.device).unsqueeze(0)
                ph_t = torch.tensor([[float(phase_list[i])]], dtype=torch.float32, device=self.device)
                all_ef.append(ef.unsqueeze(0)); all_mv.append(mv_t)
                all_pg.append(pg_t); all_ph.append(ph_t)
            ef = torch.cat(all_ef, dim=0); mv = torch.cat(all_mv, dim=0)
            pg = torch.cat(all_pg, dim=0); ph = torch.cat(all_ph, dim=0)
            logits_p1, _, _, _ = self.teacher(ef, mv, pg,
                                              ph if self.with_commit else None,
                                              action_mask=None)
        return logits_p1

    # ---- 训练 ----
    def _build_obs(self, graph_data_list, map_vec_list, progress_vec_list):
        obs_list = []
        for gd, mv, pg in zip(graph_data_list, map_vec_list, progress_vec_list):
            emb = self.gnn(gd)
            mv_t = torch.tensor(mv, dtype=torch.float32, device=self.device).unsqueeze(0)
            pg_t = torch.tensor(pg, dtype=torch.float32, device=self.device).unsqueeze(0)
            obs_list.append(torch.cat([emb.to(self.device), mv_t, pg_t], dim=-1))
        return torch.cat(obs_list, dim=0)

    def _build_edge_feats_from_h(self, qubit_h, coupling_map=None):
        if coupling_map is None:
            coupling_map = self.coupling_map
        edge_list = []
        for p, q in coupling_map:
            h_p = qubit_h[p:p+1]
            h_q = qubit_h[q:q+1]
            diff = h_p - h_q
            edge_list.append(torch.cat([h_p, h_q, diff], dim=-1))
        return torch.cat(edge_list, dim=0)

    def _build_edge_obs(self, graph_data_list, map_vec_list, progress_list,
                        coupling_maps=None, sabre_feats_list=None, phase_list=None,
                        look_feats_list=None, noise_feats_list=None,
                        global_feats_list=None, sabre_core_feats_list=None):
        all_ef, all_mv, all_pg, all_ph = [], [], [], []
        for i, (gd, mv, pg) in enumerate(zip(graph_data_list, map_vec_list, progress_list)):
            qubit_h = self.gnn.node_embeddings(gd)
            cmap = coupling_maps[i] if coupling_maps is not None else self.coupling_map
            n_local = len(cmap)
            ef = self._build_edge_feats_from_h(qubit_h.to(self.device), cmap)
            if sabre_feats_list is not None:
                sf_flat = sabre_feats_list[i]
                sf = torch.tensor(sf_flat, dtype=torch.float32, device=self.device).reshape(n_local, 5)
                ef = torch.cat([ef, sf], dim=-1)
            if look_feats_list is not None:
                lf = torch.tensor(look_feats_list[i], dtype=torch.float32,
                                  device=self.device).reshape(n_local, _LOOKAHEAD_FEAT_DIM)
                ef = torch.cat([ef, lf], dim=-1)
            if noise_feats_list is not None:
                nf = torch.tensor(noise_feats_list[i], dtype=torch.float32,
                                  device=self.device).reshape(n_local, 5)
                ef = torch.cat([ef, nf], dim=-1)
            if global_feats_list is not None:
                gf = torch.tensor(global_feats_list[i], dtype=torch.float32,
                                  device=self.device).reshape(1, -1).expand(n_local, -1)
                ef = torch.cat([ef, gf], dim=-1)
            if sabre_core_feats_list is not None:
                sc = torch.tensor(sabre_core_feats_list[i], dtype=torch.float32,
                                  device=self.device).reshape(n_local, _SABRE_CORE_FEAT_DIM)
                ef = torch.cat([ef, sc], dim=-1)
            # pad to self.num_edges (max_edges) for consistent batching
            if ef.shape[0] < self.num_edges:
                pad = torch.zeros(self.num_edges - ef.shape[0], ef.shape[1],
                                  device=ef.device, dtype=ef.dtype)
                ef = torch.cat([ef, pad], dim=0)
            mv_raw = torch.tensor(mv, dtype=torch.float32, device=self.device)
            mv_t = torch.zeros(1, self.num_qubits, dtype=torch.float32, device=self.device)
            n_copy = min(mv_raw.shape[0], self.num_qubits)
            mv_t[0, :n_copy] = mv_raw[:n_copy]
            pg_t = torch.tensor(pg, dtype=torch.float32, device=self.device).unsqueeze(0)
            if phase_list is not None:
                ph_t = torch.tensor([[float(phase_list[i])]], dtype=torch.float32, device=self.device)
            else:
                ph_t = torch.zeros(1, 1, dtype=torch.float32, device=self.device)
            all_ef.append(ef.unsqueeze(0))
            all_mv.append(mv_t)
            all_pg.append(pg_t)
            all_ph.append(ph_t)
        return torch.cat(all_ef, dim=0), torch.cat(all_mv, dim=0), torch.cat(all_pg, dim=0), torch.cat(all_ph, dim=0)

    def update(self, batch, epochs: int = 4, batch_size: int = 64,
               alpha_fid: float = 0.0, beta_kl: float = 0.0,
               la_vf_coef: float = 0.1, la_distill_lambda: float = 0.0,
               sabre_demo_lambda: float = 0.0, global_feats_list=None,
               sabre_core_feats_list=None):
        """PPO 更新。

        batch["adv"] 已经是混合 advantage（A_route + alpha·A_fid）；
        若 batch 含 "ret_fid"（双通道模式），critic 双头分别回归
        ret（路由回报）与 ret_fid（保真度回报）；否则退回单头行为。
        beta_kl > 0 且 teacher 存在时加入 KL(π_P1‖π_P2) 约束。
        la_vf_coef：V_LA（beam expectimax 多步价值头）value loss 权重，
        需 batch 含 "la_mask"/"la_val"/"la_act" 且 with_la_head=True 才生效。
        sabre_demo_lambda：SABRE 示范蒸馏（E4）——batch 含 "demo_mask"/"demo_act"
        时，demo 样本从 PPO/value/KL 损失中排除（off-policy），仅加 BC 损失
        L += λ_s·(−log π(a_sabre|s))。
        """
        acts = torch.tensor(np.array(batch["act"]), dtype=torch.long, device=self.device)
        old_logp = torch.tensor(np.array(batch["logp"]), dtype=torch.float32, device=self.device)
        adv = torch.tensor(np.array(batch["adv"]), dtype=torch.float32, device=self.device)
        ret = torch.tensor(np.array(batch["ret"]), dtype=torch.float32, device=self.device)
        dual = "ret_fid" in batch and batch["ret_fid"] is not None
        ret_fid = (torch.tensor(np.array(batch["ret_fid"]), dtype=torch.float32, device=self.device)
                   if dual else None)

        # SABRE 示范（E4）：demo 样本不参与 PPO/value/KL，仅 BC 蒸馏
        demo_on = "demo_mask" in batch and batch["demo_mask"] is not None
        if demo_on:
            demo_t = torch.tensor(np.array(batch["demo_mask"]), dtype=torch.bool,
                                  device=self.device)
            demo_act_t = torch.tensor(np.array(batch["demo_act"]), dtype=torch.long,
                                      device=self.device)
            nd_mask = ~demo_t
        else:
            demo_t = demo_act_t = nd_mask = None

        # LA（训练期 beam expectimax）：V_LA 回归目标与 actor-beam agreement
        la_on = self.with_la_head and "la_mask" in batch
        if la_on:
            la_mask = torch.tensor(np.array(batch["la_mask"]), dtype=torch.bool, device=self.device)
            la_val = torch.tensor(np.array(batch["la_val"]), dtype=torch.float32, device=self.device)
            la_act = torch.tensor(np.array(batch["la_act"]), dtype=torch.long, device=self.device)
            n_anchor = int(la_mask.sum().item())
            if n_anchor > 1:
                if self.la_raw_target:
                    # 20260917 A2：原始尺度目标（×4.6 价格放大了归一化失配，
                    # γ 折扣语义失效——raw 让 V_LA 输出与 r_c 同尺度可加）
                    la_val_n = la_val
                else:
                    lv = la_val[la_mask]
                    la_val_n = torch.zeros_like(la_val)
                    la_val_n[la_mask] = (lv - lv.mean()) / (lv.std() + 1e-8)

        # 归一化：demo 样本的 adv/ret 是垃圾（off-policy），统计量只在非 demo 上算
        if nd_mask is not None and int(nd_mask.sum()) > 0:
            adv = (adv - adv[nd_mask].mean()) / (adv[nd_mask].std() + 1e-8)
            ret = (ret - ret[nd_mask].mean()) / (ret[nd_mask].std() + 1e-8)
            if dual:
                ret_fid = ((ret_fid - ret_fid[nd_mask].mean())
                           / (ret_fid[nd_mask].std() + 1e-8))
        else:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ret = (ret - ret.mean()) / (ret.std() + 1e-8)
            if dual:
                ret_fid = (ret_fid - ret_fid.mean()) / (ret_fid.std() + 1e-8)

        is_edge = isinstance(self.ac, EdgeActorCritic)
        n = acts.shape[0]

        log_data = {"pl": [], "vl": [], "vfl": [], "vla": [], "dil": [], "agr": [],
                    "agr_map": [], "agr_rout": [], "ent": [], "kl": [], "klp1": [], "grad": [],
                    "dsl": []}
        if not is_edge and self.gnn is None:
            obs_all = torch.tensor(np.array(batch["obs"]), dtype=torch.float32, device=self.device)
        for _ in range(epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                sel = idx[start:start + batch_size]
                if is_edge:
                    cmaps = ([batch["coupling_map"][i] for i in sel]
                             if "coupling_map" in batch else None)
                    sblist = ([batch["sabre_feats"][i] for i in sel]
                              if "sabre_feats" in batch else None)
                    lflist = ([batch["look_feats"][i] for i in sel]
                              if "look_feats" in batch else None)
                    nflist = ([batch["noise_feats"][i] for i in sel]
                              if "noise_feats" in batch else None)
                    gflist = ([global_feats_list[i] for i in sel]
                              if global_feats_list is not None else None)
                    sclist = ([sabre_core_feats_list[i] for i in sel]
                              if sabre_core_feats_list is not None else None)
                    phlist = ([batch["phase"][i] for i in sel]
                              if "phase" in batch else None)
                    ef, mv, pg, ph = self._build_edge_obs(
                        [batch["graph_data"][i] for i in sel],
                        [batch["map_vec"][i] for i in sel],
                        [batch["progress"][i] for i in sel],
                        coupling_maps=cmaps,
                        sabre_feats_list=sblist,
                        phase_list=phlist,
                        look_feats_list=lflist,
                        noise_feats_list=nflist,
                        global_feats_list=gflist,
                        sabre_core_feats_list=sclist,
                    )
                    mask = None
                    if cmaps is not None:
                        n_a = self.num_edges + 1 if self.with_commit else self.num_edges
                        mask = torch.zeros(ef.shape[0], n_a, dtype=torch.bool, device=self.device)
                        for i, cmap in enumerate(cmaps):
                            mask[i, :len(cmap)] = True
                        if self.with_commit and phlist is not None:
                            for i, phv in enumerate(phlist):
                                if phv:
                                    mask[i, self.num_edges] = True
                    logits, v_route, v_fid, v_la = self.ac(ef, mv, pg,
                                                           ph if self.with_commit else None,
                                                           action_mask=mask)
                elif self.gnn is not None:
                    obs = self._build_obs(
                        [batch["graph_data"][i] for i in sel],
                        [batch["map_vec"][i] for i in sel],
                        [batch["progress"][i] for i in sel],
                    )
                    logits, v_route, v_fid = self.ac(obs)
                    v_la = torch.zeros_like(v_route)
                else:
                    logits, v_route = self.ac(obs_all[sel])
                    v_fid = torch.zeros_like(v_route)

                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(acts[sel])
                entropy = dist.entropy().mean()

                # SABRE demo 样本（E4）：从 PPO/value/KL 排除，仅 BC
                if demo_on:
                    d = demo_t[sel]
                    nd = ~d
                    nd_any = bool(nd.any().item())
                else:
                    d = None
                    nd = None
                    nd_any = True

                ratio = torch.exp(new_logp - old_logp[sel])
                surr1 = ratio * adv[sel]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv[sel]
                if nd_any:
                    policy_loss = -(torch.min(surr1, surr2)[nd]).mean() if nd is not None \
                        else -torch.min(surr1, surr2).mean()
                else:
                    policy_loss = torch.zeros((), device=self.device)

                if nd_any:
                    if nd is not None:
                        value_loss = (F.mse_loss(v_route[nd], ret[sel][nd])
                                      if nd.any() else torch.zeros((), device=self.device))
                        vfid_loss = (F.mse_loss(v_fid[nd], ret_fid[sel][nd])
                                     if dual and nd.any() else torch.zeros((), device=self.device))
                    else:
                        value_loss = F.mse_loss(v_route, ret[sel])
                        vfid_loss = (F.mse_loss(v_fid, ret_fid[sel]) if dual
                                     else torch.zeros((), device=self.device))
                else:
                    value_loss = torch.zeros((), device=self.device)
                    vfid_loss = torch.zeros((), device=self.device)
                if dual:
                    value_loss = value_loss + vfid_loss

                # LA：V_LA 回归 expectimax 目标 + beam 选择蒸馏（v2）
                vla_loss = torch.zeros(())
                dil_loss = torch.zeros(())
                agr = torch.zeros(())
                agr_map = torch.zeros(())
                agr_rout = torch.zeros(())
                if la_on and n_anchor > 1:
                    m = la_mask[sel]
                    if m.any():
                        # Huber（δ=10）：raw 目标（±数百，含 ×4.6 价格）的 MSE
                        # 梯度爆炸——Huber 有界梯度且保持 raw 单位语义
                        vla_loss = F.huber_loss(v_la[m], la_val_n[sel][m],
                                                delta=10.0)
                        agr = (la_act[sel][m] == acts[sel][m]).float().mean()
                        # agr 分列（20260917 A3：映射期 anchor 生效观测）
                        ph = torch.tensor(np.array(batch["phase"])[sel], device=self.device)
                        m_map = m & (ph > 0.5)
                        m_rout = m & (ph <= 0.5)
                        agr_map = ((la_act[sel][m_map] == acts[sel][m_map]).float().mean()
                                   if m_map.any() else torch.zeros(()))
                        agr_rout = ((la_act[sel][m_rout] == acts[sel][m_rout]).float().mean()
                                    if m_rout.any() else torch.zeros(()))
                        if la_distill_lambda > 0:
                            logp_la = torch.log_softmax(logits[m], dim=-1).gather(
                                1, la_act[sel][m].unsqueeze(1)).squeeze(1)
                            dil_loss = -logp_la.mean()

                kl_p1 = torch.zeros(())
                if beta_kl > 0 and self.teacher is not None and is_edge:
                    t_logits = self._teacher_logits(
                        [batch["graph_data"][i] for i in sel],
                        [batch["map_vec"][i] for i in sel],
                        [batch["progress"][i] for i in sel],
                        cmaps, sblist, phlist,
                        look_feats_list=lflist,
                        noise_feats_list=nflist,
                    )
                    if mask is not None:
                        t_logits = t_logits.masked_fill(~mask, -1e9)
                    t_probs = torch.softmax(t_logits, dim=-1)
                    s_probs = torch.softmax(logits, dim=-1).clamp(min=1e-12)
                    t_probs = t_probs.clamp(min=1e-12)
                    kl_per = (t_probs * (t_probs.log() - s_probs.log())).sum(-1)
                    if nd_any and nd is not None:
                        kl_p1 = kl_per[nd].mean() if nd.any() else torch.zeros((), device=self.device)
                    else:
                        kl_p1 = kl_per.mean()

                # SABRE 示范 BC（E4）：仅对未掩码的 demo 状态监督（标签只在
                # 动作未被死锁掩码时采集，故此处 logits 无 -1e9 污染风险）
                demo_loss = torch.zeros((), device=self.device)
                if sabre_demo_lambda > 0 and demo_on and d is not None and d.any():
                    logp_demo = torch.log_softmax(logits[d], dim=-1).gather(
                        1, demo_act_t[sel][d].unsqueeze(1)).squeeze(1)
                    demo_loss = -logp_demo.mean()

                loss = (policy_loss + self.vf_coef * value_loss
                        + self.la_vf_coef * vla_loss
                        + la_distill_lambda * dil_loss
                        + beta_kl * kl_p1 - self.ent_coef * entropy
                        + sabre_demo_lambda * demo_loss)

                self.optimizer.zero_grad()
                loss.backward()
                params = list(self.ac.parameters())
                if self.gnn is not None:
                    params += list(self.gnn.parameters())
                gn = nn.utils.clip_grad_norm_(params, 0.5)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - ratio.log()).mean()
                log_data["pl"].append(policy_loss.item())
                log_data["vl"].append(value_loss.item())
                log_data["vfl"].append(float(vfid_loss.item()))
                log_data["vla"].append(float(vla_loss.item()))
                log_data["agr"].append(float(agr.item()))
                log_data["agr_map"].append(float(agr_map.item()) if isinstance(agr_map, torch.Tensor) and agr_map.dim() >= 0 else 0.0)
                log_data["agr_rout"].append(float(agr_rout.item()) if isinstance(agr_rout, torch.Tensor) and agr_rout.dim() >= 0 else 0.0)
                log_data["dil"].append(float(dil_loss.item()))
                log_data["dsl"].append(float(demo_loss.item()))
                log_data["ent"].append(entropy.item())
                log_data["kl"].append(approx_kl.item())
                log_data["klp1"].append(float(kl_p1.item()))
                log_data["grad"].append(gn.item())

        return {k: float(np.mean(v)) for k, v in log_data.items()}

    def save(self, path: str):
        state = {
            "ac": self.ac.state_dict(),
            "rew_norm": {"mean": self.rew_norm.mean, "var": self.rew_norm.var},
        }
        if self.gnn is not None:
            state["gnn"] = self.gnn.state_dict()
        torch.save(state, path)

    @staticmethod
    def _zero_pad_ac_state(ac_state: dict, module: nn.Module) -> dict:
        """特征维增长兼容（E13）：目标层权重输入维比 checkpoint 大 → 补零。

        新特征（look6 批机会 + global101）追加在 edge 特征向量末尾，补零使旧
        checkpoint（如 LA287d 158 维）加载到新架构（261 维）后新列零权重、
        行为逐位不变。
        关键：critic 系输入 = edge_feats + [map_vec | progress | phase] 拼接，
        新特征位于 edge 段末尾 → 零列必须插在 old_edge_dim 处（否则把
        map_vec/progress/phase 挤到错误位置）；edge_mlp/edge_score/commit_head
        输入即 edge_feats → 零列追加在末尾即可。
        """
        if "edge_mlp.0.weight" not in ac_state:
            return ac_state
        old_ef = ac_state["edge_mlp.0.weight"].shape[-1]
        new_ef = module.edge_mlp[0].in_features
        delta = new_ef - old_ef
        if delta <= 0:
            return ac_state
        zt = torch.zeros
        for key in list(ac_state.keys()):
            if not key.endswith(".weight"):
                continue
            tgt = module
            for part in key.split("."):
                tgt = tgt[int(part)] if part.isdigit() else getattr(tgt, part)
            if tgt.dim() < 2 or tgt.shape[-1] <= ac_state[key].shape[-1]:
                continue
            if key.startswith("critic"):
                # 零列插在 edge 段末尾（old_ef 处），map_vec/progress/phase 保持原位
                ac_state[key] = torch.cat([
                    ac_state[key][:, :old_ef],
                    zt((*ac_state[key].shape[:-1], delta),
                       dtype=ac_state[key].dtype, device=ac_state[key].device),
                    ac_state[key][:, old_ef:],
                ], dim=-1)
            else:
                pad_w = zt((*ac_state[key].shape[:-1],
                            tgt.shape[-1] - ac_state[key].shape[-1]),
                           dtype=ac_state[key].dtype, device=ac_state[key].device)
                ac_state[key] = torch.cat([ac_state[key], pad_w], dim=-1)
        return ac_state

    def load(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if "ac" in state:
            ac_state = state["ac"]
            # 兼容旧 checkpoint：单头 critic.* 权重迁移到 critic_route.*
            if isinstance(self.ac, EdgeActorCritic) and any(
                    k.startswith("critic.") for k in ac_state):
                migrated = {}
                for k, v in ac_state.items():
                    migrated[k.replace("critic.", "critic_route.", 1)] = v
                ac_state = migrated
            ac_state = self._zero_pad_ac_state(ac_state, self.ac)
            missing, unexpected = self.ac.load_state_dict(ac_state, strict=False)
            if missing:
                print(f"[load] missing keys (random init): {list(missing)[:5]}")
            if unexpected:
                print(f"[load] ignored keys: {list(unexpected)[:5]}")
            if "rew_norm" in state:
                self.rew_norm.mean = state["rew_norm"]["mean"]
                self.rew_norm.var = state["rew_norm"]["var"]
            if self.gnn is not None and "gnn" in state:
                self.gnn.load_state_dict({k: v.to("cpu") for k, v in state["gnn"].items()})
        elif "shared.0.weight" in state:
            self.ac.load_state_dict(state)
        else:
            raise ValueError(f"Unknown checkpoint keys: {list(state.keys())[:5]}")

    def save_checkpoint(self, path: str, extra_state: Optional[dict] = None):
        state = {
            "ac": self.ac.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rew_norm": {"mean": self.rew_norm.mean, "var": self.rew_norm.var},
        }
        if self.gnn is not None:
            state["gnn"] = self.gnn.state_dict()
        if extra_state is not None:
            for k, v in extra_state.items():
                state[k] = v.item() if hasattr(v, "item") else v
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=False)
        ac_state = state["ac"]
        if isinstance(self.ac, EdgeActorCritic) and any(
                k.startswith("critic.") for k in ac_state):
            ac_state = {k.replace("critic.", "critic_route.", 1): v
                        for k, v in ac_state.items()}
        ac_state = self._zero_pad_ac_state(ac_state, self.ac)
        self.ac.load_state_dict(ac_state, strict=False)
        if self.gnn is not None and "gnn" in state:
            self.gnn.load_state_dict({k: v.to("cpu") for k, v in state["gnn"].items()})
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "rew_norm" in state:
            self.rew_norm.mean = state["rew_norm"]["mean"]
            self.rew_norm.var = state["rew_norm"]["var"]
        return state

    @staticmethod
    def compute_gae(rewards, values, dones, bootstrap, gamma, lam, clip_return=None):
        """广义优势估计 (GAE-lambda)。

        dones[t] = True 表示 episode 在步 t 结束（无论是 done 还是 truncated），
        此时 λ-bootstrapping 截止，且 δ_t 使用 0 作为 next_value。

        `lam` 可以是标量或与 rewards 等长的数组（自适应 λ：每步使用不同 λ_t，
        例如随 episode 进度从 0.95 增长到 0.998 以覆盖长电路截断信号）。
        """
        T = len(rewards)
        lam_arr = np.asarray(lam, dtype=float)
        if lam_arr.ndim == 0:
            lam_arr = np.full(T, float(lam_arr))
        if lam_arr.shape != (T,):
            raise ValueError(f"lam shape {lam_arr.shape} != rewards shape ({T},)")
        advantages = np.zeros(T, dtype=float)
        last_adv = 0.0
        for t in reversed(range(T)):
            if dones[t]:
                delta = rewards[t] - values[t]
                next_nonterminal = 0.0
            elif t == T - 1:
                delta = rewards[t] + gamma * bootstrap - values[t]
                next_nonterminal = 1.0
            else:
                delta = rewards[t] + gamma * values[t + 1] - values[t]
                next_nonterminal = 1.0
            last_adv = delta + gamma * lam_arr[t] * next_nonterminal * last_adv
            advantages[t] = last_adv
        returns = advantages + np.array(values)
        if clip_return is not None:
            returns = np.clip(returns, -clip_return, clip_return)
        return advantages, returns


class EMAModel:
    """Exponential Moving Average of model parameters for stable evaluation."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
