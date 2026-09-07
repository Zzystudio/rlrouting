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
                 with_commit: bool = True):
        super().__init__()
        self.num_edges = num_edges
        self.with_commit = with_commit
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.edge_score = nn.Linear(edge_feat_dim, 1)
        if with_commit:
            self.commit_head = nn.Linear(edge_feat_dim, 1)
        critic_in = edge_feat_dim + num_qubits + (2 if with_commit else 1)
        self.critic = nn.Sequential(
            nn.Linear(critic_in, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

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
        value = self.critic(v_in).squeeze(-1)
        return logits, value


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
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.device = device
        self.with_commit = with_commit

        self.gnn = gnn
        self.num_qubits = num_qubits
        self.num_edges = num_edges
        self.coupling_map = coupling_map
        self.rew_norm = RewardNormalizer()
        self.term_norm = RewardNormalizer()

        params = []
        if gnn is not None:
            self.gnn = gnn
            edge_feat_dim = self.gnn.encoder.out_dim * 3 + 5
            self.edge_feat_dim = edge_feat_dim
            self.gnn.to(device)
            self.ac = EdgeActorCritic(edge_feat_dim, num_edges, num_qubits,
                                      with_commit=with_commit).to(device)
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
        logits, value = self._forward_obs(obs, action_mask=mask.unsqueeze(0) if mask is not None else None)
        if deterministic:
            action = logits[0].argmax(-1).item()
            logp = 0.0
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action).item()
        return int(action.item()), logp, float(value.item())

    @torch.no_grad()
    def _forward_obs(self, obs, action_mask=None):
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
        return self.ac(obs_t, action_mask=action_mask)

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
                        coupling_maps=None, sabre_feats_list=None, phase_list=None):
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

    def update(self, batch, epochs: int = 4, batch_size: int = 64):
        acts = torch.tensor(np.array(batch["act"]), dtype=torch.long, device=self.device)
        old_logp = torch.tensor(np.array(batch["logp"]), dtype=torch.float32, device=self.device)
        adv = torch.tensor(np.array(batch["adv"]), dtype=torch.float32, device=self.device)
        ret = torch.tensor(np.array(batch["ret"]), dtype=torch.float32, device=self.device)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = (ret - ret.mean()) / (ret.std() + 1e-8)

        is_edge = isinstance(self.ac, EdgeActorCritic)
        n = acts.shape[0]

        log_data = {"pl": [], "vl": [], "ent": [], "kl": [], "grad": []}
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
                    phlist = ([batch["phase"][i] for i in sel]
                              if "phase" in batch else None)
                    ef, mv, pg, ph = self._build_edge_obs(
                        [batch["graph_data"][i] for i in sel],
                        [batch["map_vec"][i] for i in sel],
                        [batch["progress"][i] for i in sel],
                        coupling_maps=cmaps,
                        sabre_feats_list=sblist,
                        phase_list=phlist,
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
                    logits, value = self.ac(ef, mv, pg,
                                            ph if self.with_commit else None,
                                            action_mask=mask)
                elif self.gnn is not None:
                    obs = self._build_obs(
                        [batch["graph_data"][i] for i in sel],
                        [batch["map_vec"][i] for i in sel],
                        [batch["progress"][i] for i in sel],
                    )
                    logits, value = self.ac(obs)
                else:
                    logits, value = self.ac(obs_all[sel])

                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(acts[sel])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - old_logp[sel])
                surr1 = ratio * adv[sel]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv[sel]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value, ret[sel])
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

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
                log_data["ent"].append(entropy.item())
                log_data["kl"].append(approx_kl.item())
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

    def load(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if "ac" in state:
            missing, unexpected = self.ac.load_state_dict(state["ac"], strict=False)
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
        self.ac.load_state_dict(state["ac"])
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
