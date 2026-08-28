"""Minimal from-scratch PPO with a shared-trunk actor-critic (GAE, clipped objective)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_space: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.policy_head = nn.Linear(64, action_space)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value


class PPO:
    """Clipped-objective PPO with GAE and combined policy/value/entropy loss."""

    def __init__(self, policy: ActorCritic, gamma: float = 0.99, lr: float = 1e-3,
                 epsilon: float = 0.2, lam: float = 0.95, c1: float = 1.0,
                 c2: float = 0.01, max_grad_norm: float = 0.5,
                 target_kl: float = None,
                 device: torch.device = None):
        self.policy = policy
        self.gamma = gamma
        self.epsilon = epsilon
        self.lam = lam
        self.c1 = c1
        self.c2 = c2
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.device = device or torch.device("cpu")
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.losses: list[float] = []

    def compute_advantages(self, rewards, values, dones, next_values):
        """GAE advantage estimation. Runs on CPU; result moved to device in update()."""
        vals = values.cpu().tolist() if hasattr(values, "cpu") else list(values)
        advantages, gae = [], 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_values[t] * nonterminal - vals[t]
            gae = delta + self.gamma * self.lam * nonterminal * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages, dtype=torch.float32)

    def update(self, states, actions, rewards, dones, old_log_probs,
               epochs: int = 4, minibatch_size: int = 64):
        dev = self.device
        states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=dev)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=dev)
        old_lp_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=dev)

        with torch.no_grad():
            _, values = self.policy(states_t)
            values = values.squeeze(-1)
        old_values = values.detach()

        # next_values[t] = values[t+1] for non-terminal steps, 0 for the last step.
        # Terminal steps use nonterminal=0 in GAE so their next_values don't matter.
        next_vals = torch.cat([values[1:], torch.zeros(1, device=dev)])
        advantages = self.compute_advantages(rewards, values, dones, next_vals).to(dev)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(states_t)
        loss_sum = 0.0
        loss_count = 0
        for _ in range(epochs):
            idx = torch.randperm(n, device=dev)
            early_stop = False
            for start in range(0, n, minibatch_size):
                mb = idx[start:start + minibatch_size]
                logits, value = self.policy(states_t[mb])
                value = value.squeeze(-1)
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(actions_t[mb])
                # Clamp log-ratio before exp to prevent +inf overflow (inf*0 = NaN).
                log_ratio = (new_lp - old_lp_t[mb]).clamp(-10.0, 10.0)
                ratio = torch.exp(log_ratio)

                # Early stopping: if KL divergence is too high, skip remaining minibatches.
                if self.target_kl is not None:
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    if approx_kl > 1.5 * self.target_kl:
                        early_stop = True
                        break

                clipped = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                adv_mb = advantages[mb]
                policy_loss = -torch.min(ratio * adv_mb, clipped * adv_mb).mean()

                # Value function clipping: mirrors policy clip to stabilise value head.
                v_old = old_values[mb]
                v_clipped = v_old + torch.clamp(value - v_old, -self.epsilon, self.epsilon)
                value_loss = torch.max(
                    (value - returns[mb]) ** 2,
                    (v_clipped - returns[mb]) ** 2,
                ).mean()

                entropy = dist.entropy().mean()
                loss = policy_loss + self.c1 * value_loss - self.c2 * entropy
                if not torch.isfinite(loss):
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                loss_sum += float(loss.item())
                loss_count += 1
            if early_stop:
                break

        avg_loss = loss_sum / loss_count if loss_count > 0 else 0.0
        self.losses.append(avg_loss)
        return avg_loss
