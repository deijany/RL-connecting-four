"""PettingZoo to Stable-Baselines3 bridges for action-masked classic games."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pettingzoo.utils


class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper):
    """Expose a PettingZoo AEC env to SB3 as a single-agent, action-masked env."""

    def __init__(self, env):
        super().__init__(env)
        agent0 = self.possible_agents[0]
        self.observation_space = super().observation_space(agent0)["observation"]
        self.action_space = super().action_space(agent0)

    def reset(self, seed=None, options=None):
        super().reset(seed, options)
        self.observation_space = super().observation_space(self.possible_agents[0])["observation"]
        self.action_space = super().action_space(self.possible_agents[0])
        return self.observe(self.agent_selection), {}

    def step(self, action):
        current_agent = self.agent_selection
        super().step(action)
        obs, reward, term, trunc, info = super().last()
        if term or trunc:
            # last() returns the NEXT agent's reward, which is always -1 at terminal
            # regardless of who won. Return the acting agent's actual reward instead.
            reward = float(self.rewards.get(current_agent, 0.0))
        return obs, reward, term, trunc, info

    def observe(self, agent):
        return super().observe(agent)["observation"]

    def action_mask(self):
        return super().observe(self.agent_selection)["action_mask"]


def mask_fn(env):
    """Return the current legal-action mask (used by ActionMasker)."""
    return env.action_mask()


class PettingZooToGymEnv(gym.Env):
    """Adapt an AEC env to a plain Gym single-agent interface (no train-time masking).
    Used by vanilla PPO. Masking is reapplied only at evaluation time.
    """

    def __init__(self, env_fn, **env_kwargs):
        self._aec = env_fn.env(**env_kwargs)
        self._aec.reset()
        agent0 = self._aec.possible_agents[0]
        self.observation_space = self._aec.observation_space(agent0)["observation"]
        self.action_space = self._aec.action_space(agent0)

    def reset(self, *, seed=None, options=None):
        self._aec.reset(seed=seed)
        obs = self._aec.observe(self._aec.agent_selection)["observation"]
        return obs, {}

    def step(self, action):
        current_agent = self._aec.agent_selection
        self._aec.step(action)
        obs, reward, term, trunc, info = self._aec.last()
        if term or trunc:
            reward = float(self._aec.rewards.get(current_agent, 0.0))
        return obs["observation"], reward, term, trunc, info
