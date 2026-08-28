"""Parallel AEC rollout collector for custom PPO.

Each worker subprocess runs one AEC env independently. The main process
sends the current network weights + number of games to each worker; workers
play those games with their local copy of the network and return the full
trajectory. Main aggregates all trajectories into one rollout buffer.

Terminal rewards are applied retroactively: PettingZoo AEC reports the
reward for action a_t on the agent's NEXT call to env.last(), so the
winning/losing reward (+1/-1) appears at the terminal step. When a terminal
step is encountered the reward of the last non-terminal buffer entry for
that agent is overwritten with the correct terminal reward.
"""
from __future__ import annotations

import importlib
import multiprocessing as mp
import sys
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Worker (runs in a subprocess)
# ---------------------------------------------------------------------------

def _aec_worker(
    conn: Any,
    env_module: str,
    obs_size: int,
    n_actions: int,
    seed: int,
    repo_root: str,
    env_kwargs: dict,
) -> None:
    """Subprocess entry point. Protocol:
    - recv: ("stop",)            -> clean exit
    - recv: (state_dict, min_steps) -> play complete games until min_steps collected, send back list of trajectories
    Each trajectory is a list of (flat_obs, action, reward, done, log_prob) tuples.
    """
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import torch
    from torch.distributions import Categorical
    from src.custom_ppo import ActorCritic

    mod = importlib.import_module(env_module)
    net = ActorCritic(obs_size, n_actions)
    net.eval()
    game_counter = 0

    while True:
        msg = conn.recv()
        if msg == "stop":
            conn.close()
            return

        state_dict, min_steps = msg
        # state_dict values are numpy arrays (serialized in collect() for Pipe compat)
        net.load_state_dict({k: torch.from_numpy(v) for k, v in state_dict.items()})
        net.eval()

        all_trajs = []
        steps_collected = 0
        while steps_collected < min_steps:
            env = mod.env(**env_kwargs)
            env.reset(seed=seed + game_counter)
            game_counter += 1

            traj: list[tuple] = []
            last_buf_idx: dict[str, int] = {}

            for agent in env.agent_iter():
                obs_dict, reward, term, trunc, _ = env.last()
                done = term or trunc

                if done:
                    # Mark the agent's last entry as terminal.
                    # Only apply +1 for the winner; loser keeps reward=0.
                    # This matches SB3 self-play where the game ends at the
                    # winner's step and the loser's -1 is never added to the buffer.
                    idx = last_buf_idx.get(agent)
                    if idx is not None:
                        s, a, _, _, lp = traj[idx]
                        traj[idx] = (s, a, max(0.0, float(reward)), True, lp)
                    env.step(None)
                else:
                    flat = obs_dict["observation"].flatten().astype(np.float32)
                    mask = obs_dict["action_mask"].astype(bool)
                    with torch.no_grad():
                        logits, _ = net(torch.as_tensor(flat).unsqueeze(0))
                        logits = logits.squeeze(0)
                        logits[~torch.as_tensor(mask)] = float("-inf")
                        dist = Categorical(logits=logits)
                        a_t = dist.sample()
                        lp = float(dist.log_prob(a_t).item())
                        action = int(a_t.item())

                    last_buf_idx[agent] = len(traj)
                    traj.append((flat, action, float(reward), False, lp))
                    env.step(action)

            env.close()
            all_trajs.append(traj)
            steps_collected += len(traj)

        conn.send(all_trajs)


# ---------------------------------------------------------------------------
# Main-process collector
# ---------------------------------------------------------------------------

class ParallelAECCollector:
    """Manages N worker subprocesses for parallel AEC rollout collection.

    Usage:
        collector = ParallelAECCollector(env_fn, n_cores=8, obs_size=84,
                                         n_actions=7, seed=0, repo_root=".")
        states, actions, rewards, dones, log_probs = collector.collect(net, min_steps_per_worker=256)
        collector.close()
    """

    def __init__(
        self,
        env_fn,
        n_cores: int,
        obs_size: int,
        n_actions: int,
        seed: int = 0,
        repo_root: str = ".",
        **env_kwargs,
    ) -> None:
        self.n_cores = n_cores
        ctx = mp.get_context("spawn")
        self._workers: list[mp.Process] = []
        self._conns: list[Any] = []
        env_module = env_fn.__name__

        for i in range(n_cores):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(
                target=_aec_worker,
                args=(child_conn, env_module, obs_size, n_actions,
                      seed + i * 10_000, repo_root, env_kwargs),
                daemon=True,
            )
            p.start()
            child_conn.close()
            self._workers.append(p)
            self._conns.append(parent_conn)

    def collect(self, net, min_steps_per_worker: int = 512):
        """Broadcast current weights to all workers, collect trajectories.

        Each worker plays complete games until it has >= min_steps_per_worker steps,
        so no game is ever cut mid-way. All steps from all workers are returned as-is.
        """
        # Convert to numpy so tensors (including MPS) are picklable over Pipe
        state_dict = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        if any(np.isnan(v).any() for v in state_dict.values()):
            raise RuntimeError("NaN detected in network weights before collection")

        for conn in self._conns:
            conn.send((state_dict, min_steps_per_worker))

        buf_s: list = []
        buf_a: list = []
        buf_r: list = []
        buf_d: list = []
        buf_lp: list = []

        for conn in self._conns:
            for traj in conn.recv():
                for flat, action, reward, done, lp in traj:
                    buf_s.append(flat)
                    buf_a.append(action)
                    buf_r.append(reward)
                    buf_d.append(done)
                    buf_lp.append(lp)

        return buf_s, buf_a, buf_r, buf_d, buf_lp

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send("stop")
            except Exception:
                pass
        for w in self._workers:
            w.join(timeout=5)
            if w.is_alive():
                w.terminate()
