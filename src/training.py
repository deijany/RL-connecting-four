"""Training entry points for the three PPO variants on a PettingZoo classic env."""
from __future__ import annotations

import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.callbacks import BaseCallback
from torch.distributions import Categorical

from .custom_ppo import ActorCritic, PPO
from .env_wrappers import PettingZooToGymEnv, SB3ActionMaskWrapper, mask_fn


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Top-level env factory functions (must be at module level to be picklable
# for SubprocVecEnv's spawn context on macOS)
# ---------------------------------------------------------------------------

def _masked_env_factory(env_fn, env_kwargs):
    env = env_fn.env(**env_kwargs)
    env = SB3ActionMaskWrapper(env)
    return ActionMasker(env, mask_fn)


def _vanilla_env_factory(env_fn, env_kwargs):
    return PettingZooToGymEnv(env_fn, **env_kwargs)


# ---------------------------------------------------------------------------
# SB3 loss logger callback
# ---------------------------------------------------------------------------

class _LossLogger(BaseCallback):
    """Records (env_steps, loss) and optionally prints win-rate checkpoints."""

    def __init__(self, env_fn=None, eval_fn=None, eval_every: int = None,
                 eval_games: int = 200, label: str = ""):
        super().__init__(verbose=0)
        self.steps: list[int] = []
        self.losses: list[float] = []
        self._env_fn = env_fn
        self._eval_fn = eval_fn
        self._eval_every = eval_every
        self._eval_games = eval_games
        self._label = label
        self._next_eval = eval_every or 0

    def _record(self) -> None:
        val = self.model.logger.name_to_value.get("train/loss")
        if val is not None:
            self.steps.append(int(self.model.num_timesteps))
            self.losses.append(float(val))

    def _on_rollout_start(self) -> None:
        self._record()

    def _on_training_end(self) -> None:
        self._record()

    def _on_step(self) -> bool:
        if self._eval_every and self.model.num_timesteps >= self._next_eval:
            scores = self._eval_fn(
                self._env_fn, self.model, num_games=self._eval_games,
            )
            total = sum(scores.values()) or 1
            print(
                f"  [{self._label}] step {self.model.num_timesteps:>8,} | "
                f"win {scores['win']/total*100:5.1f}%  "
                f"loss {scores['loss']/total*100:5.1f}%  "
                f"tie {scores['tie']/total*100:4.1f}%  "
                f"({self._eval_games} games)"
            )
            self._next_eval += self._eval_every
        return True


# ---------------------------------------------------------------------------
# train_masked
# ---------------------------------------------------------------------------

def train_masked(
    env_fn,
    steps: int = 10_000,
    seed: int = 0,
    save_dir: str = ".",
    return_losses: bool = False,
    n_steps: int = 256,
    n_epochs: int = 4,
    lr: float = 3e-4,
    gamma: float = 0.99,
    clip_range: float = 0.2,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.01,
    vf_coef: float = 1.0,
    n_cores: int = 1,
    minibatch_size: int = 64,
    eval_every: int = None,
    eval_games: int = 200,
    **env_kwargs,
):
    """Train MaskablePPO with self-play.

    Returns path, or (path, env_steps_list, losses_list) when return_losses=True.
    With n_cores > 1 uses SubprocVecEnv for parallel collection.
    """
    device = _get_device()

    if n_cores > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        env = SubprocVecEnv([
            partial(_masked_env_factory, env_fn, env_kwargs)
            for _ in range(n_cores)
        ])
    else:
        env = _masked_env_factory(env_fn, env_kwargs)
        env.reset(seed=seed)

    model = MaskablePPO(
        MaskableActorCriticPolicy, env,
        n_steps=max(1, n_steps // n_cores),
        n_epochs=n_epochs,
        batch_size=minibatch_size,
        learning_rate=lr,
        gamma=gamma,
        clip_range=clip_range,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        verbose=0,
        device=device,
    )
    model.set_random_seed(seed)
    from .evaluation import evaluate_vs_random
    cb = _LossLogger(env_fn=env_fn, eval_fn=evaluate_vs_random,
                     eval_every=eval_every, eval_games=eval_games,
                     label="Masked PPO")
    model.learn(total_timesteps=steps, callback=cb)

    os.makedirs(save_dir, exist_ok=True)
    name = f"{env.unwrapped.metadata.get('name', 'env')}_{time.strftime('%Y%m%d-%H%M%S')}"
    path = os.path.join(save_dir, name)
    model.save(path)
    env.close()

    if return_losses:
        return path + ".zip", cb.steps, cb.losses
    return path + ".zip"


# ---------------------------------------------------------------------------
# train_vanilla
# ---------------------------------------------------------------------------

def train_vanilla(
    env_fn,
    steps: int = 10_000,
    seed: int = 0,
    save_dir: str = ".",
    return_losses: bool = False,
    n_steps: int = 256,
    n_epochs: int = 4,
    lr: float = 3e-4,
    gamma: float = 0.99,
    clip_range: float = 0.2,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.01,
    vf_coef: float = 1.0,
    n_cores: int = 1,
    minibatch_size: int = 64,
    eval_every: int = None,
    eval_games: int = 200,
    **env_kwargs,
):
    """Train vanilla SB3 PPO (no train-time masking).

    Returns path, or (path, env_steps_list, losses_list) when return_losses=True.
    With n_cores > 1 uses SubprocVecEnv for parallel collection.
    """
    device = _get_device()

    if n_cores > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        env = SubprocVecEnv([
            partial(_vanilla_env_factory, env_fn, env_kwargs)
            for _ in range(n_cores)
        ])
    else:
        env = _vanilla_env_factory(env_fn, env_kwargs)

    model = SB3PPO(
        "MlpPolicy", env,
        n_steps=max(1, n_steps // n_cores),
        n_epochs=n_epochs,
        batch_size=minibatch_size,
        learning_rate=lr,
        gamma=gamma,
        clip_range=clip_range,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        seed=seed,
        verbose=0,
        device=device,
    )
    from .evaluation import evaluate_vanilla_vs_random
    cb = _LossLogger(env_fn=env_fn, eval_fn=evaluate_vanilla_vs_random,
                     eval_every=eval_every, eval_games=eval_games,
                     label="Vanilla PPO")
    model.learn(total_timesteps=steps, callback=cb)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"vanilla_{time.strftime('%Y%m%d-%H%M%S')}")
    model.save(path)
    env.close()

    if return_losses:
        return path + ".zip", cb.steps, cb.losses
    return path + ".zip"


# ---------------------------------------------------------------------------
# train_custom
# ---------------------------------------------------------------------------

def train_custom(
    env_fn,
    steps: int = 2_000,
    seed: int = 0,
    n_steps: int = 256,
    n_epochs: int = 4,
    lr: float = 3e-4,
    gamma: float = 0.99,
    clip_range: float = 0.2,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.01,
    vf_coef: float = 1.0,
    n_cores: int = 1,
    minibatch_size: int = 64,
    target_kl: float = None,
    episodes: int = None,
    eval_every: int = None,
    eval_games: int = 200,
    **env_kwargs,
):
    """Self-play training loop driving the custom PPO.

    With n_cores > 1 uses ParallelAECCollector: each worker subprocess plays
    complete games until it has collected n_steps // n_cores steps, then the
    main process aggregates all workers' complete-game trajectories and updates
    the network -- no game is ever cut mid-way.

    With n_cores == 1 uses a single-env rolling buffer (original behaviour).

    Terminal rewards (+1/-1) are applied retroactively so the network sees
    the game outcome. Returns (ppo, env_steps_list, losses_list).
    """
    env = env_fn.env(**env_kwargs)
    env.reset(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent0 = env.possible_agents[0]
    obs_size = int(np.prod(env.observation_space(agent0)["observation"].shape))
    n_actions = env.action_space(agent0).n
    env.close()

    device = _get_device()
    net = ActorCritic(obs_size, n_actions).to(device)
    ppo = PPO(net, gamma=gamma, lr=lr, epsilon=clip_range, lam=gae_lambda,
              c1=vf_coef, c2=ent_coef, target_kl=target_kl, device=device)

    cumulative_steps = 0
    steps_log: list[int] = []
    losses_log: list[float] = []
    _next_eval = eval_every or 0

    def _maybe_eval_custom(label: str = "Custom PPO") -> None:
        nonlocal _next_eval
        if not eval_every or cumulative_steps < _next_eval:
            return
        from .evaluation import evaluate_custom_vs_random
        scores = evaluate_custom_vs_random(env_fn, net, num_games=eval_games)
        total = sum(scores.values()) or 1
        print(
            f"  [{label}] step {cumulative_steps:>8,} | "
            f"win {scores['win']/total*100:5.1f}%  "
            f"loss {scores['loss']/total*100:5.1f}%  "
            f"tie {scores['tie']/total*100:4.1f}%  "
            f"({eval_games} games)"
        )
        _next_eval += eval_every

    # -- Parallel path --
    if n_cores > 1:
        from .parallel_aec import ParallelAECCollector
        repo_root = str(Path(__file__).resolve().parent.parent)
        collector = ParallelAECCollector(
            env_fn, n_cores, obs_size, n_actions,
            seed=seed, repo_root=repo_root, **env_kwargs,
        )
        ep = 0
        try:
            while (steps is None or cumulative_steps < steps) and \
                  (episodes is None or ep < episodes):
                # Each worker plays complete games until it has n_steps // n_cores steps.
                # All steps from all workers (all complete games) are used for the update.
                states, actions, rewards, dones, log_probs = collector.collect(
                    net, min_steps_per_worker=max(1, n_steps // n_cores),
                )
                if not states:
                    break
                ep += sum(1 for d in dones if d) // 2
                cumulative_steps += len(states)
                loss = ppo.update(
                    states=states, actions=actions, rewards=rewards,
                    dones=dones, old_log_probs=log_probs,
                    epochs=n_epochs, minibatch_size=minibatch_size,
                )
                steps_log.append(cumulative_steps)
                losses_log.append(loss)
                _maybe_eval_custom()
        finally:
            collector.close()
        return ppo, steps_log, losses_log

    # -- Single-env rolling buffer path --
    env = env_fn.env(**env_kwargs)
    env.reset(seed=seed)

    ep = 0
    buf_states: list = []
    buf_actions: list = []
    buf_rewards: list = []
    buf_dones: list = []
    buf_log_probs: list = []

    while True:
        if steps is not None and cumulative_steps >= steps:
            break
        if episodes is not None and ep >= episodes:
            break
        ep += 1

        env.reset()
        last_buf_idx: dict[str, int] = {}

        for agent in env.agent_iter():
            obs, reward, term, trunc, _ = env.last()
            done = term or trunc

            if done:
                # Mark terminal; only apply +1 for winner, loser keeps reward=0.
                idx = last_buf_idx.get(agent)
                if idx is not None:
                    buf_rewards[idx] = max(0.0, float(reward))
                    buf_dones[idx] = True
                action = None
            else:
                flat = torch.as_tensor(obs["observation"].flatten(), dtype=torch.float32)
                mask = torch.as_tensor(obs["action_mask"], dtype=torch.bool)
                with torch.no_grad():
                    logits, _ = net(flat.to(device).unsqueeze(0))
                    logits = logits.squeeze(0).cpu()
                    logits[~mask] = float("-inf")
                dist = Categorical(logits=logits)
                action = int(dist.sample().item())

                last_buf_idx[agent] = len(buf_states)
                buf_states.append(flat.numpy())
                buf_actions.append(action)
                buf_rewards.append(float(reward))
                buf_dones.append(False)
                buf_log_probs.append(dist.log_prob(torch.tensor(action)).detach().item())
                cumulative_steps += 1

            env.step(action)

        # Update when buffer has enough steps (all complete games, variable size)
        if len(buf_states) >= n_steps:
            loss = ppo.update(
                states=buf_states,
                actions=buf_actions,
                rewards=buf_rewards,
                dones=buf_dones,
                old_log_probs=buf_log_probs,
                epochs=n_epochs,
                minibatch_size=minibatch_size,
            )
            steps_log.append(cumulative_steps)
            losses_log.append(loss)
            _maybe_eval_custom()
            buf_states = []
            buf_actions = []
            buf_rewards = []
            buf_dones = []
            buf_log_probs = []

    env.close()
    return ppo, steps_log, losses_log
