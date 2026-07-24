# ============================================================================
# agent.py
# 紧凑自包含的 PPO（Proximal Policy Optimization）实现：
# Actor-Critic + GAE + 截断代理目标。不依赖外部 RL 库。
# ============================================================================

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """PPO 智能体。"""

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
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.device = device
        self.model = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    # ---- 交互 ----
    @torch.no_grad()
    def act(self, obs: np.ndarray):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.model(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action).item(), float(value.item())

    # ---- 训练 ----
    def update(self, batch, epochs: int = 4, batch_size: int = 64):
        obs = torch.tensor(np.array(batch["obs"]), dtype=torch.float32, device=self.device)
        acts = torch.tensor(np.array(batch["act"]), dtype=torch.long, device=self.device)
        old_logp = torch.tensor(np.array(batch["logp"]), dtype=torch.float32, device=self.device)
        adv = torch.tensor(np.array(batch["adv"]), dtype=torch.float32, device=self.device)
        ret = torch.tensor(np.array(batch["ret"]), dtype=torch.float32, device=self.device)

        # 归一化优势
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.shape[0]
        for _ in range(epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                sel = idx[start:start + batch_size]
                logits, value = self.model(obs[sel])
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
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    @staticmethod
    def compute_gae(rewards, values, dones, bootstrap, gamma, lam):
        """广义优势估计 (GAE-lambda)。"""
        T = len(rewards)
        advantages = np.zeros(T, dtype=float)
        last_adv = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_nonterminal = 1.0 - dones[t]
                next_value = bootstrap
            else:
                next_nonterminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            last_adv = delta + gamma * lam * next_nonterminal * last_adv
            advantages[t] = last_adv
        returns = advantages + np.array(values)
        return advantages, returns
