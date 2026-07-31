from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from routing.rl.env import RoutingEnv
from routing.rl.agent import PPOAgent


def episode_outcome(env: RoutingEnv, done: bool, truncated: bool, info: dict) -> float:
    """终局价值，与训练目标 z 保持一致（AlphaZero 目标 = 回合结果）。

    - truncated: 与 collect_self_play_data 的 z=-10.0 对齐
    - routing:   z = -num_swaps
    - 噪声感知:   z = fidelity
    """
    if truncated:
        return -10.0
    if env.reward_mode == "routing":
        return float(-info.get("num_swaps", env._swap_counter))
    return float(info.get("fidelity", 0.0))


class MCTSNode:
    """Monte Carlo Tree Search node.

    每个节点对应一个路由状态，存储从该状态出发的边统计信息：
      - P(s,a): 策略先验（来自 policy network softmax）
      - N(s,a): 访问次数
      - W(s,a): 累计叶节点价值
      - children: 动作 → 子节点映射（仅包含合法动作）
    """

    def __init__(self, num_edges: int):
        self.prior_probs = np.zeros(num_edges, dtype=np.float64)
        self.N = np.zeros(num_edges, dtype=np.float64)
        self.W = np.zeros(num_edges, dtype=np.float64)
        self.children: dict[int, MCTSNode] = {}
        self.is_expanded = False

    def q(self, action: int) -> float:
        if self.N[action] == 0:
            return 0.0
        return self.W[action] / self.N[action]


class MCTS:
    """蒙特卡洛树搜索推理器。

    在每次路由决策时，从当前环境状态出发，执行 num_simulations 次
    MCTS 仿真。每路仿真从根克隆环境，沿树下降（PUCT selection）、
    展开并评估叶节点（policy+value），然后将结果回传。
    最终根据根节点访问分布选择动作。
    """

    def __init__(
        self,
        agent: PPOAgent,
        num_simulations: int = 100,
        c_puct: float = 1.4,
        temperature: float = 0.0,
    ):
        self.agent = agent
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature

    def search(
        self, env: RoutingEnv,
        add_dirichlet_noise: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
    ) -> tuple[int, np.ndarray]:
        num_edges = self.agent.num_edges
        root = MCTSNode(num_edges)

        # Expand root
        self._expand(root, env)

        if add_dirichlet_noise:
            self._add_dirichlet(root, dirichlet_alpha, dirichlet_eps)

        for _ in range(self.num_simulations):
            sim = env.clone()
            node = root
            path: list[tuple[MCTSNode, int]] = []

            # Selection（中间节点不需要 obs，只在叶节点展开时计算一次）
            done, truncated = False, False
            info: dict = {}
            while node.is_expanded and not done and not truncated:
                a = self._select(node)
                path.append((node, a))
                _, _, done, truncated, info = sim.step(a, compute_obs=False)
                if done or truncated:
                    break
                node = node.children[a]

            # Terminal leaf during selection
            if done or truncated:
                value = episode_outcome(sim, done, truncated, info)
            else:
                value = self._expand(node, sim)

            # Backpropagation
            for n, a in path:
                n.N[a] += 1.0
                n.W[a] += value

        return self._select_action(root)

    def _add_dirichlet(self, node: MCTSNode, alpha: float, eps: float):
        valid = np.where(node.prior_probs > 1e-12)[0]
        if len(valid) == 0:
            return
        alpha_vec = [alpha / len(valid)] * len(valid)
        noise = np.random.dirichlet(alpha_vec)
        for i, a in enumerate(valid):
            node.prior_probs[a] = (1 - eps) * node.prior_probs[a] + eps * noise[i]

    # ------------------------------------------------------------------
    #  PUCT selection
    # ------------------------------------------------------------------
    def _select(self, node: MCTSNode) -> int:
        total_N = float(node.N.sum())
        sqrt_total = math.sqrt(max(total_N, 1e-8))
        best_score = -float("inf")
        best_action = -1
        for a in node.children:
            u = (
                self.c_puct
                * node.prior_probs[a]
                * sqrt_total
                / (1.0 + node.N[a])
            )
            score = node.q(a) + u
            if score > best_score:
                best_score = score
                best_action = a
        return best_action

    # ------------------------------------------------------------------
    #  Expand leaf node with policy + value forward
    # ------------------------------------------------------------------
    def _expand(self, node: MCTSNode, env: RoutingEnv) -> float:
        obs = env._obs()
        mask = self._build_action_mask(env)
        with torch.no_grad():
            logits, value = self.agent._forward_obs(
                obs, action_mask=mask.unsqueeze(0)
            )

        logits_np = logits[0].cpu().numpy()
        mask_np = mask.cpu().numpy()

        logits_masked = np.where(mask_np, logits_np, -1e9)
        logits_masked -= logits_masked.max()
        exp_l = np.exp(logits_masked)
        priors = exp_l / (exp_l.sum() + 1e-8)
        priors[~mask_np] = 0.0

        node.prior_probs[:] = priors[:]
        node.N[:] = 0.0
        node.W[:] = 0.0
        node.children = {}

        for a_idx in np.where(mask_np)[0]:
            a = int(a_idx)
            node.children[a] = MCTSNode(len(priors))

        node.is_expanded = True
        return float(value.item())

    # ------------------------------------------------------------------
    #  Action mask: valid edges (within coupling_map) + no deadlock
    # ------------------------------------------------------------------
    def _build_action_mask(self, env: RoutingEnv) -> torch.Tensor:
        num_edges = self.agent.num_edges
        mask = torch.zeros(num_edges, dtype=torch.bool, device=self.agent.device)
        mask[: len(env.coupling_map)] = True
        dm = env.get_deadlock_mask()
        for i in range(min(len(dm), num_edges)):
            if dm[i]:
                mask[i] = False
        return mask

    # ------------------------------------------------------------------
    #  Action selection from root visit counts
    # ------------------------------------------------------------------
    def _select_action(self, root: MCTSNode) -> tuple[int, np.ndarray]:
        if self.temperature == 0:
            best = int(np.argmax(root.N))
            probs = np.zeros_like(root.N)
            probs[best] = 1.0
            return best, probs

        counts = root.N ** (1.0 / max(self.temperature, 1e-8))
        total = counts.sum()
        if total < 1e-8:
            probs = np.ones_like(root.N) / max(len(root.N), 1)
        else:
            probs = counts / total
        best = int(np.argmax(probs))
        return best, probs
