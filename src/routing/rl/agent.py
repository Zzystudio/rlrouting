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

    def forward(self, x):
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value


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
        vf_coef: float = 0.5,
        device: str = "cpu",
        gnn: Optional[SubGNN] = None,
        num_qubits: Optional[int] = None,
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.device = device

        self.gnn = gnn
        if gnn is not None:
            self.gnn.to("cpu")
            ac_in = self.gnn.out_dim + num_qubits + 1
        else:
            ac_in = obs_dim

        self.ac = ActorCritic(ac_in, action_dim).to(device)

        params = list(self.ac.parameters())
        if self.gnn is not None:
            params += list(self.gnn.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    # ---- 交互 ----
    @torch.no_grad()
    def act(self, obs: np.ndarray):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.ac(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action).item(), float(value.item())

    # ---- 训练 ----
    def _build_obs(self, graph_data_list, map_vec_list, progress_vec_list):
        obs_list = []
        for gd, mv, pg in zip(graph_data_list, map_vec_list, progress_vec_list):
            emb = self.gnn(gd)
            mv_t = torch.tensor(mv, dtype=torch.float32, device=self.device).unsqueeze(0)
            pg_t = torch.tensor(pg, dtype=torch.float32, device=self.device).unsqueeze(0)
            obs_list.append(torch.cat([emb.to(self.device), mv_t, pg_t], dim=-1))
        return torch.cat(obs_list, dim=0)

    def update(self, batch, epochs: int = 4, batch_size: int = 64):
        acts = torch.tensor(np.array(batch["act"]), dtype=torch.long, device=self.device)
        old_logp = torch.tensor(np.array(batch["logp"]), dtype=torch.float32, device=self.device)
        adv = torch.tensor(np.array(batch["adv"]), dtype=torch.float32, device=self.device)
        ret = torch.tensor(np.array(batch["ret"]), dtype=torch.float32, device=self.device)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        if self.gnn is not None:
            obs_all = self._build_obs(batch["graph_data"], batch["map_vec"], batch["progress"])
            obs_all = obs_all.detach()
        else:
            obs_all = torch.tensor(np.array(batch["obs"]), dtype=torch.float32, device=self.device)

        n = obs_all.shape[0]
        log_data = {"pl": [], "vl": [], "ent": [], "kl": [], "grad": []}
        for _ in range(epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                sel = idx[start:start + batch_size]
                if self.gnn is not None:
                    obs = self._build_obs(
                        [batch["graph_data"][i] for i in sel],
                        [batch["map_vec"][i] for i in sel],
                        [batch["progress"][i] for i in sel],
                    )
                else:
                    obs = obs_all[sel]

                logits, value = self.ac(obs)
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
        state = {"ac": self.ac.state_dict()}
        if self.gnn is not None:
            state["gnn"] = self.gnn.state_dict()
        torch.save(state, path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        if "ac" in state:
            self.ac.load_state_dict(state["ac"])
            if self.gnn is not None and "gnn" in state:
                self.gnn.load_state_dict({k: v.to("cpu") for k, v in state["gnn"].items()})
        elif "shared.0.weight" in state:
            self.ac.load_state_dict(state)
        else:
            raise ValueError(f"Unknown checkpoint keys: {list(state.keys())[:5]}")

    @staticmethod
    def compute_gae(rewards, values, dones, bootstrap, gamma, lam):
        """广义优势估计 (GAE-lambda)。

        dones[t] = True 表示 episode 在步 t 结束（无论是 done 还是 truncated），
        此时 λ-bootstrapping 截止，且 δ_t 使用 0 作为 next_value。
        """
        T = len(rewards)
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
            last_adv = delta + gamma * lam * next_nonterminal * last_adv
            advantages[t] = last_adv
        returns = advantages + np.array(values)
        return advantages, returns
