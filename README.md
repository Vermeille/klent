# Revisiting Regularized Policy Optimization for Stable and Efficient Reinforcement Learning in Two-Player Games

[![Project Page](https://img.shields.io/badge/project-page-1f6feb)](https://kazukiohta.github.io/klent/)
[![Paper](https://img.shields.io/badge/paper-arXiv-b31b1b)](https://arxiv.org/abs/2602.10894)
[![Poster](https://img.shields.io/badge/poster-PDF-2ea44f)](https://kazukiohta.github.io/klent/assets/poster.pdf)

This repository provides supplementary code for the ICML 2026 paper
[**"Revisiting Regularized Policy Optimization for Stable and Efficient Reinforcement Learning in Two-Player Games."**](https://kazukiohta.github.io/klent/)

For an overview of the method, figures, poster, and BibTeX, please see the
[project page](https://kazukiohta.github.io/klent/).

The code contains a compact implementation of our algorithm KLENT for training and evaluating agents in board game environments.

## Installation

Please run the following commands to install the dependencies.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Execution

Please run the following command to train and evaluate an agent in a 9x9 Go environment without Weights & Biases logging.

```
python3 main.py env_id=go_9x9 wandb_on=false
```

Choose an environment from the following list: `connect_four`, `animal_shogi`, `gardner_chess`, `go_9x9`, `hex`, and `othello`. If you do not specify, `connect_four` will be chosen by default.

To enable Weights & Biases logging, set `wandb_on=true`.

Please make sure that you have activated the virtual environment before the execution. The execution may require a GPU with CUDA 12.

## Contents
* `main.py` is the main file that trains and evaluates the agent.
* `resnet.py` provides the residual network architecture with policy and action-value heads.
* `util.py` provides a useful function for handling `jax.random.key`.
* `README.md` is this file.
* `requirements.txt` provides the information of dependencies. 

## Century asymmetric PPO comparison

`ours.py` is a second, self-contained training entrypoint implementing the
[Century asymmetric advantage-shaped PPO trainer](https://github.com/Vermeille/century-RL/blob/master/trainers/adversarial-advshape.py)
on the **same PGX environments, ResNet, default self-play size and simulator
budget** as KLENT's `main.py`. The original `main.py` is unchanged.

```bash
python3 ours.py env_id=connect_four wandb_on=false
# Quick smoke run (CPU or GPU, with the dependencies above installed):
python3 ours.py env_id=connect_four selfplay_vmap=8 selfplay_step=64 \
  fitting_batch_size=64 evaluation_vmap=8 num_channels=16 num_blocks=1 \
  limit_simulator_evaluations=512 wandb_on=false
```

Each episode randomly assigns the **agent** and **environment** policies to the
two seats. The policies have independent model weights, critics, PPO updates,
perplexity thermostats, and adaptive reverse-KL penalties. For each policy,
normalized GAE is multiplied by `threshold - rollout_win_rate` (defaults:
agent `0.98`, environment `0.70`), including the sign reversal above the
threshold. TD(lambda) value targets are *not* shaped. Returns/advantages flip
sign across opposing-player turns and cut traces at episode boundaries.

For architectural parity, PPO uses the original PQNet's action-value head:
`V(s) = sum_a pi(a|s) Q(s,a)`, with the Q head fitted to the sampled
TD(lambda) targets. BatchNorm uses inference statistics for both rollout and
PPO fitting, so unchanged weights really give the same behavior policy.

The cost axis `cost/simulator_evaluations/total` counts **training** calls
to `env.step` and retains KLENT's evaluation-excluded convention. This matches
simulator evaluations, **not wall-clock time, forward-pass count, or total
parameters**: `ours.py` executes two full policy forward passes per rollout
step and has two parameter sets. It evaluates the deployable agent against the
same PGX baseline on player 0, with KLENT-compatible `eval/...` and `score/...`
logging names. Checkpoints are written as paired `-agent.keras` and
`-environment.keras` models under `checkpoints/<env_id>_<seed>/ours/`.
These are model exports, not full optimizer/controller resumable snapshots.

Exploration and PPO hyperparameters are CLI configurable (OmegaConf, as with
`main.py`); `lr_schedule=true` enables optional LR warmup/cosine decay.
