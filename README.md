# RL Connecting Four

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deijany/RL-connecting-four/blob/main/connect_four_ppo.ipynb)

Three PPO variants trained via self-play on Connect Four (`connect_four_v3`, [PettingZoo Classic](https://pettingzoo.farama.org/environments/classic/)):

| Variant | Train-time masking | Library |
|---------|--------------------|---------|
| Masked PPO | Yes | `sb3_contrib.MaskablePPO` |
| Vanilla PPO | No | `stable_baselines3.PPO` |
| Custom PPO | Yes (manual) | Pure PyTorch, hand-written PPO loop |

All three share the same network shape at inference (84 -> 64 -> 64 -> 7) and the same hyperparameters, so the comparison isolates the effect of action masking and implementation (SB3 vs. from-scratch).

Masked and Custom PPO both mask illegal moves during training, but not the same way. Masked PPO relies on `sb3_contrib`'s built-in masking layer and collects fixed-length rollouts through `SubprocVecEnv`, standard SB3 behaviour, so an episode can end mid-rollout. Custom PPO masks by hand (illegal logits set to `-inf` before the softmax) and collects through `ParallelAECCollector`, where each worker finishes a full game before returning, so no episode is ever cut mid-way.

## Structure

```
connect_four_ppo.ipynb   # main notebook: train, evaluate, compare
src/
  training.py            # train_masked / train_vanilla / train_custom
  custom_ppo.py           # ActorCritic net + from-scratch PPO
  evaluation.py           # vs-random evaluation for each variant
  env_wrappers.py         # PettingZoo AEC -> single-agent Gym adapters
  parallel_aec.py         # multi-worker rollout collector for Custom PPO
  plotting.py             # win-rate / training-curve plots
```

## Running

Click the Colab badge above and run all cells. Locally:

```bash
pip install "pettingzoo[classic]>=1.24.0" "stable-baselines3>=2.0.0" "sb3-contrib>=2.0.0" "gymnasium<=0.29.1" torch supersuit
jupyter notebook connect_four_ppo.ipynb
```
