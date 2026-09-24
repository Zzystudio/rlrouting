# ============================================================================
# agent_clocked.py - 时钟化统一动作空间 agent（doc/20260920训练方案.md §6）
# ClockEdgeActorCritic: logits=[E edge(SWAP)|K gate(EXEC)|commit|skip]
# ClockedPPOAgent: obs 布局 [edge_block(E*D_edge)|exec_block(K*D_exec)|
#                  map_vec|progress|phase|timing_glob(D_TG)] 的拆分、统一
#                  mask、warm-start、C0 冻结、auto-batch BC 预热。
# 纯新增文件：不触碰 agent.py 既有行为。
# ============================================================================
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .agent import EdgeActorCritic, PPOAgent, RewardNormalizer
from ..gnn.encoder import SubGNN

# ---- 特征维度常量（与 env_clocked.py 严格一致）----
D_EDGE_TIMING = 4        # edge 块追加：both_free/marginal_theta_swap/busy_min/parallel_usage
D_GLOBAL_TIMING = 8      # global 广播块 101 -> 109
D_EXEC_HAND = 12         # EXEC 手工特征（exec12）
EXEC_H_DIM = 48          # GNN 嵌入维
D_EXEC = EXEC_H_DIM * 4 + D_EXEC_HAND   # 204
D_TIMING_GLOB = 9        # skip/critic 的 state-level 时序向量

# GNN 第一层中需要置零的输入列（warm-start，§6.2：新特征初始影响为零）
_NODE_ZERO_DIMS = (19, 20, 21, 22, 23, 24, 27, 28, 29, 30)
_EDGE_ZERO_DIMS = (10, 12, 13, 14)


class ClockEdgeActorCritic(EdgeActorCritic):
    """统一词表 actor-critic：SWAP(edge)/EXEC(gate)/COMMIT/SKIP。

    输入 edge_feats[B,E,D_edge]、exec_feats[B,K,D_exec]、map_vec/progress/phase、
    timing_glob[B,D_TIMING_GLOB]；输出 logits[B,E+K+2]（布局
    [E edges, K gates, commit(E+K), skip(E+K+1)]）与 (v_route, v_fid, v_la)。
    """

    def __init__(self, edge_feat_dim: int, num_edges: int, num_qubits: int,
                 max_ready: int, exec_feat_dim: int = D_EXEC,
                 timing_glob_dim: int = D_TIMING_GLOB,
                 with_commit: bool = True, with_la_head: bool = False,
                 edge_hidden: int = 64):
        nn.Module.__init__(self)
        self.num_edges = num_edges
        self.max_ready = max_ready
        self.with_commit = with_commit
        self.with_la_head = with_la_head
        self.edge_feat_dim = edge_feat_dim
        self.exec_feat_dim = exec_feat_dim
        self.timing_glob_dim = timing_glob_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, edge_hidden), nn.ReLU(),
            nn.Linear(edge_hidden, edge_hidden // 2), nn.ReLU(),
            nn.Linear(edge_hidden // 2, 1))
        self.gate_mlp = nn.Sequential(
            nn.Linear(exec_feat_dim, edge_hidden), nn.ReLU(),
            nn.Linear(edge_hidden, edge_hidden // 2), nn.ReLU(),
            nn.Linear(edge_hidden // 2, 1))
        self.edge_score = nn.Linear(edge_feat_dim, 1)
        self.gate_score = nn.Linear(exec_feat_dim, 1)
        if with_commit:
            self.commit_head = nn.Linear(edge_feat_dim, 1)
        skip_in = edge_feat_dim + exec_feat_dim + timing_glob_dim
        self.skip_head = nn.Sequential(nn.Linear(skip_in, 64), nn.ReLU(),
                                       nn.Linear(64, 1))
        critic_in = edge_feat_dim + exec_feat_dim + timing_glob_dim + num_qubits + 2
        self.critic_route = nn.Sequential(nn.Linear(critic_in, 64), nn.ReLU(),
                                          nn.Linear(64, 1))
        self.critic_fid = nn.Sequential(nn.Linear(critic_in, 64), nn.ReLU(),
                                        nn.Linear(64, 1))
        for p in self.critic_fid.parameters():
            nn.init.zeros_(p)
        if with_la_head:
            self.critic_la = nn.Sequential(nn.Linear(critic_in, 64), nn.ReLU(),
                                           nn.Linear(64, 1))
            for p in self.critic_la.parameters():
                nn.init.zeros_(p)

    def forward(self, edge_feats, exec_feats, map_vec, progress, phase=None,
                timing_glob=None, action_mask=None):
        B, E, _ = edge_feats.shape
        _, K, _ = exec_feats.shape
        scores = self.edge_mlp(edge_feats).squeeze(-1)          # [B, E]
        gscores = self.gate_mlp(exec_feats).squeeze(-1)         # [B, K]
        attn_raw = self.edge_score(edge_feats)
        gattn_raw = self.gate_score(exec_feats)

        if action_mask is not None:
            m = action_mask
            if m.dim() == 1:
                m = m.unsqueeze(0)
            scores = scores.masked_fill(~m[:, :E], -1e9)
            gscores = gscores.masked_fill(~m[:, E:E + K], -1e9)
            attn_raw = attn_raw.masked_fill(~m[:, :E].unsqueeze(-1), -1e9)
            gattn_raw = gattn_raw.masked_fill(~m[:, E:E + K].unsqueeze(-1), -1e9)

        attn_w = torch.softmax(attn_raw, dim=1)
        pooled = (edge_feats * attn_w).sum(dim=1)
        gattn_w = torch.softmax(gattn_raw, dim=1)
        gpooled = (exec_feats * gattn_w).sum(dim=1)

        if self.with_commit:
            commit_logit = self.commit_head(pooled).squeeze(-1)
            logits = torch.cat([scores, gscores, commit_logit.unsqueeze(-1)], dim=-1)
        else:
            logits = torch.cat([scores, gscores], dim=-1)
        if timing_glob is None:
            timing_glob = torch.zeros(B, self.timing_glob_dim, device=scores.device)
        skip_logit = self.skip_head(
            torch.cat([pooled, gpooled, timing_glob], dim=-1)).squeeze(-1)
        logits = torch.cat([logits, skip_logit.unsqueeze(-1)], dim=-1)

        if action_mask is not None:
            logits = logits.masked_fill(~m, -1e9)

        if phase is None:
            v_in = torch.cat([pooled, gpooled, timing_glob, map_vec, progress], dim=-1)
        else:
            v_in = torch.cat([pooled, gpooled, timing_glob, map_vec, progress, phase], dim=-1)
        v_route = self.critic_route(v_in).squeeze(-1)
        v_fid = self.critic_fid(v_in).squeeze(-1)
        if self.with_la_head:
            v_la = self.critic_la(v_in).squeeze(-1)
        else:
            v_la = torch.zeros_like(v_route)
        return logits, v_route, v_fid, v_la


def zero_gnn_new_feature_columns(gnn: Optional[SubGNN]):
    """warm-start 用：GNN 第一层对应新特征维度的输入权重列置零。

    节点维 32（置零 19-24/27-30）、边维 16（置零 10/12-14）。仅处理输入层
    （convs[0]），使新特征初始影响为零（§6.2）。
    """
    if gnn is None:
        return
    conv = gnn.encoder.convs[0]
    for p in conv.parameters():
        if p.dim() < 2:
            continue
        if p.shape[1] == 32:      # 节点输入维
            with torch.no_grad():
                p[:, list(_NODE_ZERO_DIMS)] = 0.0
        elif p.shape[1] == 16:    # 边输入维
            with torch.no_grad():
                p[:, list(_EDGE_ZERO_DIMS)] = 0.0


def reinit_critic_heads(ac: ClockEdgeActorCritic):
    """warm-start 后重初始化 critic（旧 critic 输入布局与新 obs 错位，
    零基线起步最安全；策略头 edge_mlp/edge_score/commit 不受影响）。"""
    for name in ("critic_route", "critic_fid"):
        mod = getattr(ac, name)
        for p in mod.parameters():
            nn.init.zeros_(p)
    if getattr(ac, "with_la_head", False) and hasattr(ac, "critic_la"):
        for p in ac.critic_la.parameters():
            nn.init.zeros_(p)


class ClockedPPOAgent(PPOAgent):
    """时钟化 PPO：与 PPOAgent 相同的训练接口，obs 布局/词表/mask 不同。"""

    def __init__(self, obs_dim: int, action_dim: int, *,
                 num_qubits: int, num_edges: int, max_ready: int,
                 edge_feat_dim: int, exec_feat_dim: int = D_EXEC,
                 timing_glob_dim: int = D_TIMING_GLOB,
                 lr: float = 3e-4, gamma: float = 0.99, lam: float = 0.95,
                 clip_eps: float = 0.2, ent_coef: float = 0.01,
                 vf_coef: float = 0.1, device: str = "cpu",
                 gnn: Optional[SubGNN] = None, with_commit: bool = True,
                 coupling_map: Optional[list] = None,
                 lambda_v_fid: float = 1.0, with_la_head: bool = False,
                 la_vf_coef: float = 0.1, la_raw_target: bool = False,
                 edge_hidden: int = 64,
                 edge_anchor_lambda: float = 0.0):
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
        self.teacher = None
        self.teacher_gnn = None
        self.coupling_map = coupling_map

        self.gnn = gnn
        self.num_qubits = num_qubits
        self.num_edges = num_edges
        self.max_ready = max_ready
        self.edge_feat_dim = edge_feat_dim
        self.exec_feat_dim = exec_feat_dim
        self.timing_glob_dim = timing_glob_dim
        self.rew_norm = RewardNormalizer()
        self.term_norm = RewardNormalizer()

        params = []
        if gnn is not None:
            self.gnn.to(device)
        self.ac = ClockEdgeActorCritic(
            edge_feat_dim, num_edges, num_qubits, max_ready,
            exec_feat_dim=exec_feat_dim, timing_glob_dim=timing_glob_dim,
            with_commit=with_commit, with_la_head=with_la_head,
            edge_hidden=edge_hidden).to(device)
        if gnn is not None:
            params += list(self.gnn.parameters())
        params += list(self.ac.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.ema = None
        self.edge_anchor_lambda = float(edge_anchor_lambda)
        self._anchor_params = None
        self._init_ac = None  # R1a：BC(ASAP) 热身后的冻结策略快照

    def snapshot_init_policy(self):
        """冻结当前策略作 KL 锚（在 BC(ASAP) 热身之后调用）。"""
        import copy as _copy
        self._init_ac = _copy.deepcopy(self.ac).eval()
        for _p in self._init_ac.parameters():
            _p.requires_grad_(False)

    # ---- obs 拆分（时钟化布局）----
    def _split_obs(self, obs, batched=False):
        D = self.edge_feat_dim
        D2 = self.exec_feat_dim
        Dt = self.timing_glob_dim
        n_ef = self.num_edges * D
        n_xf = self.max_ready * D2
        nq = self.num_qubits
        obs = np.asarray(obs)
        if batched:
            Kb = obs.shape[0]
            ef = torch.tensor(obs[:, :n_ef], dtype=torch.float32,
                              device=self.device).reshape(Kb, self.num_edges, D)
            xf = torch.tensor(obs[:, n_ef:n_ef + n_xf], dtype=torch.float32,
                              device=self.device).reshape(Kb, self.max_ready, D2)
            mv = torch.tensor(obs[:, n_ef + n_xf:n_ef + n_xf + nq],
                              dtype=torch.float32, device=self.device)
            pg = torch.tensor(obs[:, n_ef + n_xf + nq:n_ef + n_xf + nq + 1],
                              dtype=torch.float32, device=self.device)
            ph = torch.tensor(obs[:, n_ef + n_xf + nq + 1:n_ef + n_xf + nq + 2],
                              dtype=torch.float32, device=self.device)
            tg = torch.tensor(obs[:, -Dt:], dtype=torch.float32, device=self.device)
            return ef, xf, mv, pg, ph, tg
        ef = torch.tensor(obs[:n_ef], dtype=torch.float32,
                          device=self.device).reshape(1, self.num_edges, D)
        xf = torch.tensor(obs[n_ef:n_ef + n_xf], dtype=torch.float32,
                          device=self.device).reshape(1, self.max_ready, D2)
        mv = torch.tensor(obs[n_ef + n_xf:n_ef + n_xf + nq],
                          dtype=torch.float32, device=self.device).unsqueeze(0)
        pg = torch.tensor(obs[n_ef + n_xf + nq:n_ef + n_xf + nq + 1],
                          dtype=torch.float32, device=self.device).unsqueeze(0)
        ph = torch.tensor(obs[n_ef + n_xf + nq + 1:n_ef + n_xf + nq + 2],
                          dtype=torch.float32, device=self.device).unsqueeze(0)
        tg = torch.tensor(obs[-Dt:], dtype=torch.float32,
                          device=self.device).unsqueeze(0)
        return ef, xf, mv, pg, ph, tg

    @torch.no_grad()
    def _forward_obs_split(self, obs, action_mask=None):
        ef, xf, mv, pg, ph, tg = self._split_obs(obs)
        return self.ac(ef, xf, mv, pg, ph, tg, action_mask=action_mask)

    @torch.no_grad()
    def _forward_obs_batch_full(self, obs_batch, action_masks=None):
        ef, xf, mv, pg, ph, tg = self._split_obs(obs_batch, batched=True)
        return self.ac(ef, xf, mv, pg, ph, tg, action_mask=action_masks)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False,
            deadlock_mask=None, mapping_phase: bool = False,
            action_mask=None):
        """统一词表 act：action_mask 由 env 提供（锁/frontier/skip 合法性）。

        未提供时退化为最小 mask（仅耦合边 + commit/skip 按 phase）。
        """
        E = self.num_edges
        K = self.max_ready
        total = E + K + 2
        if action_mask is None:
            mask = torch.zeros(total, dtype=torch.bool, device=self.device)
            mask[:len(self.coupling_map)] = True
            if deadlock_mask is not None:
                for i in range(min(len(deadlock_mask), len(mask))):
                    if deadlock_mask[i]:
                        mask[i] = False
            mask[E + K] = mapping_phase          # commit
            mask[E + K + 1] = True               # skip（env 会真正判定）
        else:
            mask = torch.tensor(np.asarray(action_mask, dtype=bool),
                                device=self.device)
        logits, v_route, v_fid, _v_la = self._forward_obs_split(
            obs, action_mask=mask.unsqueeze(0))
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

    def load(self, path: str):
        """warm-start：丢弃 critic 系权重 + edge 头列扩展零初始化 + GNN 新维
        列置零 + critic 重初始化（§6.2）。"""
        state = torch.load(path, map_location=self.device, weights_only=False)
        if "ac" in state:
            ac_state = dict(state["ac"])
            # 旧 critic.* 前缀迁移到 critic_route.*（兼容最老 checkpoint）
            ac_state = {("critic_route." + k[len("critic."):]) if k.startswith("critic.")
                        else k: v for k, v in ac_state.items()}
            # 丢弃 critic 系：输入布局与时钟化 obs 不兼容，载后零初始化
            ac_state = {k: v for k, v in ac_state.items()
                        if not k.startswith("critic")}
            ac_state = self._zero_pad_ac_state(ac_state, self.ac)
            self.ac.load_state_dict(ac_state, strict=False)
            if "rew_norm" in state:
                self.rew_norm.mean = state["rew_norm"]["mean"]
                self.rew_norm.var = state["rew_norm"]["var"]
            if self.gnn is not None and "gnn" in state:
                self.gnn.load_state_dict({k: v.to("cpu")
                                          for k, v in state["gnn"].items()})
        elif "shared.0.weight" in state:
            self.ac.load_state_dict(state)
        else:
            raise ValueError(f"Unknown checkpoint keys: {list(state.keys())[:5]}")
        zero_gnn_new_feature_columns(self.gnn)
        reinit_critic_heads(self.ac)
        self._capture_anchor()

    def load_checkpoint(self, path: str):
        """同 load，但返回 checkpoint state（train_agent 续训用）。"""
        state = torch.load(path, map_location=self.device, weights_only=False)
        if "ac" in state:
            ac_state = dict(state["ac"])
            ac_state = {("critic_route." + k[len("critic."):]) if k.startswith("critic.")
                        else k: v for k, v in ac_state.items()}
            ac_state = {k: v for k, v in ac_state.items()
                        if not k.startswith("critic")}
            ac_state = self._zero_pad_ac_state(ac_state, self.ac)
            self.ac.load_state_dict(ac_state, strict=False)
            if self.gnn is not None and "gnn" in state:
                self.gnn.load_state_dict({k: v.to("cpu")
                                          for k, v in state["gnn"].items()})
            if "optimizer" in state:
                self.optimizer.load_state_dict(state["optimizer"])
            if "rew_norm" in state:
                self.rew_norm.mean = state["rew_norm"]["mean"]
                self.rew_norm.var = state["rew_norm"]["var"]
        zero_gnn_new_feature_columns(self.gnn)
        reinit_critic_heads(self.ac)
        self._capture_anchor()
        return state

    def _capture_anchor(self):
        """保存 edge 头 + GNN 的当前权重快照（(param 引用, 初值) 对），
        供 L2 锚定（防 C1 解冻漂移，§6.2）。"""
        self._anchor_params = []
        if self.gnn is not None:
            for p in self.gnn.parameters():
                self._anchor_params.append((p, p.detach().clone()))
        for name, p in self.ac.named_parameters():
            if name.startswith(("edge_mlp", "edge_score", "commit_head")):
                self._anchor_params.append((p, p.detach().clone()))

    def freeze_edge_gnn(self, freeze: bool = True):
        """C0：冻结 GNN + edge 头（edge_mlp/edge_score/commit_head），
        只训 gate/skip 头与 critic。"""
        for p in self.gnn.parameters():
            p.requires_grad_(not freeze)
        for mod in (self.ac.edge_mlp, self.ac.edge_score, self.ac.commit_head):
            for p in mod.parameters():
                p.requires_grad_(not freeze)

    def update(self, batch, epochs: int = 4, batch_size: int = 64,
               alpha_fid: float = 0.0, beta_kl: float = 0.0,
               la_vf_coef: float = 0.1, la_distill_lambda: float = 0.0,
               sabre_demo_lambda: float = 0.0, global_feats_list=None,
               sabre_core_feats_list=None, init_kl_beta: float = 0.0):
        """时钟化 PPO 更新：flat obs 前向（_forward_obs_batch_full）。

        时钟化路径不启用 SABRE demo / LA / teacher KL（C0 用 BC 预热替代），
        故仅实现 PPO 核心：clipped policy + value(route/fid 双头) + entropy。
        """
        acts = torch.tensor(np.array(batch["act"]), dtype=torch.long,
                            device=self.device)
        old_logp = torch.tensor(np.array(batch["logp"]), dtype=torch.float32,
                                device=self.device)
        adv = torch.tensor(np.array(batch["adv"]), dtype=torch.float32,
                           device=self.device)
        ret = torch.tensor(np.array(batch["ret"]), dtype=torch.float32,
                           device=self.device)
        dual = "ret_fid" in batch and batch["ret_fid"] is not None
        ret_fid = (torch.tensor(np.array(batch["ret_fid"]),
                                dtype=torch.float32, device=self.device)
                   if dual else None)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = (ret - ret.mean()) / (ret.std() + 1e-8)
        if dual:
            ret_fid = (ret_fid - ret_fid.mean()) / (ret_fid.std() + 1e-8)
        n = acts.shape[0]
        log_data = {"pl": [], "vl": [], "vfl": [], "vla": [], "agr": [],
                    "agr_map": [], "agr_rout": [], "ent": [], "kl": [],
                    "klp1": [], "grad": [], "dsl": [], "kli": []}
        for _ in range(epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                sel = idx[start:start + batch_size]
                obs_sel = np.array([batch["obs"][i] for i in sel])
                ef, xf, mv, pg, ph, tg = self._split_obs(obs_sel, batched=True)
                logits, v_route, v_fid, _v_la = self.ac(ef, xf, mv, pg, ph, tg)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(acts[sel])
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_logp - old_logp[sel])
                surr1 = ratio * adv[sel]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                    1 + self.clip_eps) * adv[sel]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(v_route, ret[sel])
                if dual:
                    vfid_loss = F.mse_loss(v_fid, ret_fid[sel])
                    value_loss = value_loss + vfid_loss
                else:
                    vfid_loss = torch.zeros((), device=self.device)
                loss = (policy_loss + self.vf_coef * value_loss
                        - self.ent_coef * entropy)
                if init_kl_beta > 0 and self._init_ac is not None:
                    # R1a：锚定 BC(ASAP) 初始策略——防漂移出 asap 盆地
                    with torch.no_grad():
                        lf_i, _vi, _vf2, _vla = self._init_ac(
                            ef, xf, mv, pg, ph, tg)
                    logp_i = torch.log_softmax(lf_i, dim=-1)
                    logp_c = torch.log_softmax(logits, dim=-1)
                    p_i = logp_i.exp()
                    kl_init = (p_i * (logp_i - logp_c)).sum(-1).mean()
                    loss = loss + init_kl_beta * kl_init
                    log_data["kli"].append(float(kl_init.item()))
                if self.edge_anchor_lambda > 0 and self._anchor_params:
                    anc = sum(((p - p0) ** 2).sum()
                              for p, p0 in self._anchor_params)
                    loss = loss + self.edge_anchor_lambda * anc
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
                log_data["vl"].append(float(value_loss.item()))
                log_data["vfl"].append(float(vfid_loss.item()))
                log_data["ent"].append(entropy.item())
                log_data["kl"].append(approx_kl.item())
                log_data["grad"].append(float(gn.item()))
        out = {k: float(np.mean(v)) if v else 0.0 for k, v in log_data.items()}
        out["pl"] = out["pl"]; out["vl"] = out["vl"]; out["ent"] = out["ent"]
        out["kl"] = out["kl"]; out["grad"] = out["grad"]
        return out


def bc_warmup_auto_batch(agent: ClockedPPOAgent, env, num_steps: int = 4096,
                         batch_size: int = 256, lr: float = 3e-4,
                         seed: int = 0):
    """C0 前置：auto-batch 示范蒸馏（§6.2）。

    专家轨迹：每状态若存在合法 EXEC → 取最小槽位（依次发射全部合法 EXEC 的
    确定性展开）；无合法 EXEC 但 SKIP 合法 → SKIP；否则取冻结 edge 头的
    argmax SWAP。BC 只更新 gate/skip 头（edge/GNN 保持冻结）。
    返回 (样本数, 平均 loss)。
    """
    params = [p for name, p in agent.ac.named_parameters()
              if name.startswith("gate_mlp") or name.startswith("skip_head")]
    opt = torch.optim.Adam(params, lr=lr)
    E = agent.num_edges
    K = agent.max_ready
    total = 0
    loss_sum = 0.0
    rng = np.random.default_rng(seed)
    obs, _ = env.reset()
    done = False
    for it in range(0, num_steps, batch_size):
        obs_buf, act_buf = [], []
        for _ in range(min(batch_size, num_steps - it)):
            if done:
                obs, _ = env.reset()
                done = False
            mask = env.get_action_mask()
            if env.mapping_phase:
                a = env.commit_action
            else:
                legal_exec = [i for i in range(E, E + K) if mask[i]]
                if legal_exec:
                    a = legal_exec[0]
                elif mask[E + K + 1]:
                    a = E + K + 1
                else:
                    edge_mask = mask[:E]
                    legal_edges = [i for i in range(E) if edge_mask[i]]
                    if not legal_edges:
                        raise RuntimeError("BC 专家陷入无合法动作状态")
                    a = int(legal_edges[rng.integers(len(legal_edges))])
            obs_buf.append(obs)
            act_buf.append(a)
            obs, r, done, truncated, info = env.step(a)
            done = done or truncated
        obs_arr = np.asarray(obs_buf)
        acts = torch.tensor(act_buf, dtype=torch.long, device=agent.device)
        ef, xf, mv, pg, ph, tg = agent._split_obs(obs_arr, batched=True)
        logits, _vr, _vf, _vl = agent.ac(ef, xf, mv, pg, ph, tg)
        loss = F.cross_entropy(logits, acts)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum += float(loss.item()) * len(obs_buf)
        total += len(obs_buf)
        if (it // batch_size) % 4 == 0:
            print(f"[bc] step {total}/{num_steps} loss={loss.item():.3f}")
    return total, loss_sum / max(1, total)

