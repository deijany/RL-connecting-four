"""Evaluation: trained policies vs a random opponent, with masking applied at play time."""
from __future__ import annotations

import numpy as np


def evaluate_vs_random(env_fn, model, num_games: int = 100, seed: int = 0, **env_kwargs):
    """Play num_games with model as player 0 and random (legal) as player 1.
    Returns counts "win", "loss", "tie" from the model's perspective.
    """
    env = env_fn.env(**env_kwargs)
    scores = {"win": 0, "loss": 0, "tie": 0}

    for g in range(num_games):
        env.reset(seed=seed + g)
        model_agent = env.possible_agents[0]
        for agent in env.agent_iter():
            obs, reward, term, trunc, _ = env.last()
            if term or trunc:
                if agent == model_agent:
                    if reward > 0:
                        scores["win"] += 1
                    elif reward < 0:
                        scores["loss"] += 1
                    else:
                        scores["tie"] += 1
                action = None
            elif agent == model_agent:
                action = int(model.predict(
                    obs["observation"], action_masks=obs["action_mask"],
                    deterministic=True)[0])
            else:
                legal = np.flatnonzero(obs["action_mask"])
                action = int(np.random.choice(legal))
            env.step(action)
    env.close()
    return scores


def evaluate_vanilla_vs_random(env_fn, model, num_games: int = 100, seed: int = 0,
                                **env_kwargs):
    """Vanilla SB3 PPO vs random. Predicted illegal moves fall back to a random legal action.
    Returns counts "win", "loss", "tie" from the model's perspective.
    """
    env = env_fn.env(**env_kwargs)
    scores = {"win": 0, "loss": 0, "tie": 0}

    for g in range(num_games):
        env.reset(seed=seed + g)
        model_agent = env.possible_agents[0]
        for agent in env.agent_iter():
            obs, reward, term, trunc, _ = env.last()
            if term or trunc:
                if agent == model_agent:
                    if reward > 0:
                        scores["win"] += 1
                    elif reward < 0:
                        scores["loss"] += 1
                    else:
                        scores["tie"] += 1
                action = None
            elif agent == model_agent:
                legal = np.flatnonzero(obs["action_mask"])
                predicted = int(model.predict(obs["observation"], deterministic=True)[0])
                action = predicted if predicted in legal else int(np.random.choice(legal))
            else:
                legal = np.flatnonzero(obs["action_mask"])
                action = int(np.random.choice(legal))
            env.step(action)
    env.close()
    return scores


def evaluate_custom_vs_random(env_fn, net, num_games: int = 100, seed: int = 0,
                               **env_kwargs):
    """Custom ActorCritic vs random. Mask applied to probs before greedy selection.
    Returns counts "win", "loss", "tie" from the model's perspective.
    """
    import torch

    env = env_fn.env(**env_kwargs)
    scores = {"win": 0, "loss": 0, "tie": 0}
    net.eval()
    device = next(net.parameters()).device

    for g in range(num_games):
        env.reset(seed=seed + g)
        model_agent = env.possible_agents[0]
        for agent in env.agent_iter():
            obs, reward, term, trunc, _ = env.last()
            if term or trunc:
                if agent == model_agent:
                    if reward > 0:
                        scores["win"] += 1
                    elif reward < 0:
                        scores["loss"] += 1
                    else:
                        scores["tie"] += 1
                action = None
            elif agent == model_agent:
                flat = torch.as_tensor(obs["observation"].flatten(), dtype=torch.float32, device=device)
                mask = torch.as_tensor(obs["action_mask"], dtype=torch.bool)
                with torch.no_grad():
                    logits, _ = net(flat.unsqueeze(0))
                    logits = logits.squeeze(0).cpu()
                    logits[~mask] = float("-inf")
                action = int(logits.argmax().item())
            else:
                legal = np.flatnonzero(obs["action_mask"])
                action = int(np.random.choice(legal))
            env.step(action)
    env.close()
    return scores
