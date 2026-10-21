"""Century's asymmetric, advantage-shaped PPO on the KLENT PGX benchmark.

This is deliberately a sibling of main.py: it uses the same games, PQNet,
rollout/evaluation sizes and simulator-evaluation budget, but *not* KLENT's
regularized policy improvement / supervised policy fitting.

The two roles have independent policies and critics. They play each other with
randomized seats. Each role's PPO advantage is multiplied by its target win
rate minus its observed batch win rate; its critic target is NOT shaped.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import itertools
import math
from time import time
from typing import get_args

import jax
import jax.numpy as jnp
import keras
import numpy as np
import optax
import pgx
import wandb
from omegaconf import OmegaConf
from pydantic import BaseModel

from resnet import PQNet
from util import KeyGenerator


class Config(BaseModel):
    env_id: str = "connect_four"
    seed: int = 0
    # Same simulator budget / batch construction as KLENT's main.py.
    selfplay_vmap: int = 1024
    selfplay_step: int = 2048
    fitting_batch_size: int = 4096
    fitting_epochs: int = 1
    limit_simulator_evaluations: int = 10**9
    # Same network architecture.
    num_channels: int = 128
    num_blocks: int = 6
    num_params: int = 0
    zero_init: bool = True
    # PPO / TD(lambda), matching the Century trainer where possible.
    learning_rate: float = 1e-3
    gradient_clip: float = 5.0
    ppo_clip: float = 0.2
    discount: float = 1.0
    gae_lambda: float = 0.1
    value_lambda: float = 0.9
    value_strength: float = 1.0
    agent_threshold: float = 0.98
    environment_threshold: float = 0.70
    # Role-specific scheduled *raw* perplexity targets, exp(entropy).
    agent_perplexity_start: float = 2.5
    agent_perplexity_end: float = 1.5
    environment_perplexity_start: float = 3.0
    environment_perplexity_end: float = 1.0
    agent_entropy_strength: float = 0.1
    environment_entropy_strength: float = 1.0
    entropy_baseline_ratio: float = 0.05
    perplexity_adaptation_rate: float = 0.004
    perplexity_beta: float = 0.98
    perplexity_deadband: float = 0.02
    kl_target: float = 0.05
    kl_strength: float = 0.05
    kl_adaptation_rate: float = 0.05
    kl_deadband: float = 0.001
    # Match the original constant-LR benchmark by default. Can be overridden.
    lr_schedule: bool = False
    min_lr_scale: float = 0.0
    warmup_iterations: int = 20
    host_name: str = os.uname().nodename
    device_kind: str = jax.local_devices()[0].device_kind
    # Evaluation: the same seat-0 PGX baseline as main.py.
    evaluation_vmap: int = 1024
    save_interval: int = 50
    checkpoints_dir: str = "./checkpoints"
    wandb_on: bool = True
    comment: str = ""


config = Config(**OmegaConf.from_cli())
if config.selfplay_vmap < 1 or config.selfplay_step < 1 or config.fitting_batch_size < 1:
    raise ValueError("Rollout and fitting batch sizes must be positive")
if config.fitting_epochs < 1 or config.gradient_clip <= 0:
    raise ValueError("fitting_epochs and gradient_clip must be positive")
if not (0.0 <= config.agent_threshold <= 1.0 and 0.0 <= config.environment_threshold <= 1.0):
    raise ValueError("Thresholds must be in [0, 1]")
if not (0.0 <= config.discount <= 1.0 and 0.0 <= config.gae_lambda <= 1.0 and 0.0 <= config.value_lambda <= 1.0):
    raise ValueError("discount and lambda values must be in [0, 1]")
if not (0.0 <= config.entropy_baseline_ratio <= 1.0):
    raise ValueError("entropy_baseline_ratio must be in [0, 1]")

env = pgx.make(config.env_id)
if env.num_players != 2:
    raise ValueError("Asymmetric training currently requires two-player games")
prod_observation_shape = int(np.prod(env.observation_shape))

keras.utils.set_random_seed(config.seed)
agent_model = PQNet(env.observation_shape, env.num_actions, config.zero_init,
                    config.num_channels, config.num_blocks)
environment_model = PQNet(env.observation_shape, env.num_actions, config.zero_init,
                          config.num_channels, config.num_blocks)
config.num_params = 2 * agent_model.count_params()  # Two separately trained models.


def get_params(model):
    return {
        "trainable_variables": tuple(jnp.asarray(v.numpy()) for v in model.trainable_variables),
        "non_trainable_variables": tuple(jnp.asarray(v.numpy()) for v in model.non_trainable_variables),
    }


def encode(observation):
    return jnp.packbits(observation.astype(jnp.bool_).flatten())


def decode(code):
    return jnp.unpackbits(code)[:prod_observation_shape].reshape(env.observation_shape).astype(jnp.float32)


def network(observations, params):
    # In PPO, sampling and updating must use the same policy distribution.
    # Inference-mode BN avoids per-minibatch BN statistics changing the PPO
    # behavior log-probabilities and KL even before any optimizer step.
    observations = jax.vmap(lambda obs: decode(encode(obs)))(observations)
    outputs, _ = agent_model.stateless_call(**params, inputs=observations, training=False)
    return outputs["logits"], outputs["qvalue"]


def policy(logits, legal):
    return jax.nn.log_softmax(jnp.where(legal, logits, -1e9), axis=-1)


def state_value(log_probs, qvalues):
    # The unmodified KLENT PQNet has Q(s,a), not a scalar critic head.
    # V(s) = E_{a ~ policy}[Q(s,a)]; the critic loss fits sampled Q(s,a).
    return jnp.sum(jnp.exp(log_probs) * qvalues, axis=-1)


def signed_gae(reward, done, values, players, bootstrap_values, bootstrap_players, trace_lambda):
    """Two-player zero-sum GAE from the *acting* player's perspective.

    Values and rewards refer to the player whose turn it is in that state.
    A change of current_player reverses the sign. Terminals cut the trace, so
    independently reset episodes do not leak into one another.
    """
    next_v = jnp.concatenate([values[1:], bootstrap_values[None]], axis=0)
    next_player = jnp.concatenate([players[1:], bootstrap_players[None]], axis=0)
    sign = jnp.where(players == next_player, 1.0, -1.0)
    ongoing = (~done).astype(values.dtype)
    delta = reward + config.discount * ongoing * sign * next_v - values

    def step(carry, row):
        d, s, cont = row
        advantage = d + config.discount * trace_lambda * s * cont * carry
        return advantage, advantage

    _, advantages = jax.lax.scan(step, jnp.zeros_like(bootstrap_values),
                                 (delta, sign, ongoing), reverse=True)
    return advantages


@jax.jit
def selfplay(rng_key, agent_params, environment_params):
    """Collect two-role behavior distributions and signed TD(lambda) targets."""
    n = config.selfplay_vmap
    rng_key, init_key, seat_key = jax.random.split(rng_key, 3)
    states = jax.vmap(env.init)(jax.random.split(init_key, n))
    agent_seats = jax.random.randint(seat_key, (n,), 0, 2)

    def step(carry, _):
        states, agent_seats, key = carry
        key, reset_key, seat_key, action_key, step_key = jax.random.split(key, 5)
        was_done = states.terminated | states.truncated
        reset_keys = jax.random.split(reset_key, n)
        states = jax.vmap(lambda state, k: jax.lax.cond(
            (state.terminated | state.truncated), lambda: env.init(k), lambda: state
        ))(states, reset_keys)
        agent_seats = jnp.where(was_done,
                                jax.random.randint(seat_key, (n,), 0, 2),
                                agent_seats)
        agent_turn = states.current_player == agent_seats
        agent_logits, agent_q = network(states.observation, agent_params)
        opponent_logits, opponent_q = network(states.observation, environment_params)
        logits = jnp.where(agent_turn[:, None], agent_logits, opponent_logits)
        qvalues = jnp.where(agent_turn[:, None], agent_q, opponent_q)
        legal = states.legal_action_mask
        log_probs = policy(logits, legal)
        actions = jax.random.categorical(action_key, log_probs, axis=-1)
        values = state_value(log_probs, qvalues)

        next_states = jax.vmap(env.step)(states, actions, jax.random.split(step_key, n))
        row = jnp.arange(n)
        rewards = next_states.rewards[row, states.current_player]
        rewards_agent = next_states.rewards[row, agent_seats]
        transition = (
            jax.vmap(encode)(states.observation), legal, actions, log_probs,
            rewards, (next_states.terminated | next_states.truncated), values,
            states.current_player, jnp.where(agent_turn, 0, 1), rewards_agent,
        )
        return (next_states, agent_seats, key), transition

    (last_states, last_agent_seats, _), trajectory = jax.lax.scan(
        step, (states, agent_seats, rng_key), None, length=config.selfplay_step
    )
    (C, L, A, LOGP, R, D, V, PLAYER, ROLE, AGENT_REWARD) = trajectory

    last_agent_turn = last_states.current_player == last_agent_seats
    a_logits, a_q = network(last_states.observation, agent_params)
    e_logits, e_q = network(last_states.observation, environment_params)
    last_logits = jnp.where(last_agent_turn[:, None], a_logits, e_logits)
    last_q = jnp.where(last_agent_turn[:, None], a_q, e_q)
    last_v = state_value(policy(last_logits, last_states.legal_action_mask), last_q)
    last_v = jnp.where(last_states.terminated | last_states.truncated, 0.0, last_v)
    gae = signed_gae(R, D, V, PLAYER, last_v, last_states.current_player,
                     config.gae_lambda)
    value_targets = V + signed_gae(R, D, V, PLAYER, last_v,
                                   last_states.current_player, config.value_lambda)

    # Flatten step x environment, retaining the role and behavior distribution.
    def flatten(x):
        return x.reshape((-1, *x.shape[2:]))

    return {
        "C": flatten(C), "L": flatten(L), "A": flatten(A),
        "LOGP": flatten(LOGP), "ADV": flatten(gae),
        "TARGET": flatten(value_targets), "ROLE": flatten(ROLE),
        "D": flatten(D), "AGENT_REWARD": flatten(AGENT_REWARD),
    }


# One optimizer and one independent controller state per policy.
optimizer = optax.chain(
    optax.clip_by_global_norm(config.gradient_clip),
    optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-7),
)


def init_controller(initial_entropy_strength):
    # [entropy coefficient, smoothed perplexity, KL coefficient]
    return jnp.array([initial_entropy_strength, jnp.nan, config.kl_strength])


@jax.jit
def train_step(params, opt_state, controller, batch, target_ppl,
               init_entropy_strength, learning_rate):
    """One PPO minibatch step; adapt both controllers without host sync."""
    def objective(trainable):
        observations = jax.vmap(decode)(batch["C"])
        outputs, _ = agent_model.stateless_call(
            trainable_variables=trainable,
            non_trainable_variables=params["non_trainable_variables"],
            inputs=observations, training=False,
        )
        log_probs = policy(outputs["logits"], batch["L"])
        new_logp = jnp.take_along_axis(log_probs, batch["A"][:, None], axis=-1)[:, 0]
        old_logp = jnp.take_along_axis(batch["LOGP"], batch["A"][:, None], axis=-1)[:, 0]
        ratio = jnp.exp(new_logp - old_logp)
        unclipped = ratio * batch["ADV"]
        clipped = jnp.clip(ratio, 1.0 - config.ppo_clip,
                           1.0 + config.ppo_clip) * batch["ADV"]
        policy_loss = -jnp.minimum(unclipped, clipped).mean()
        q_chosen = jnp.take_along_axis(outputs["qvalue"], batch["A"][:, None], axis=-1)[:, 0]
        value_loss = jnp.square(q_chosen - batch["TARGET"]).mean()

        probs = jnp.exp(log_probs)
        safe_log_probs = jnp.where(batch["L"], log_probs, 0.0)
        entropy_each = -jnp.sum(probs * safe_log_probs, axis=-1)
        entropy = entropy_each.mean()
        kl = jnp.sum(probs * jnp.where(batch["L"],
                       log_probs - batch["LOGP"], 0.0), axis=-1).mean()
        valid = (batch["L"].sum(axis=-1) > 1).astype(probs.dtype)
        ppl = jnp.sum(jnp.exp(entropy_each) * valid) / jnp.maximum(valid.sum(), 1)

        # Same one-sided thermostat and adaptive reverse KL as Century's
        # ScheduledPerplexity and AdaptiveKLPenalty (on this PPO minibatch).
        old_ema = controller[1]
        ema = jnp.where(jnp.isnan(old_ema), ppl,
                        config.perplexity_beta * old_ema +
                        (1.0 - config.perplexity_beta) * ppl)
        entropy_error = target_ppl - ema
        baseline_strength = init_entropy_strength * config.entropy_baseline_ratio
        new_entropy_strength = jnp.where(
            entropy_error > config.perplexity_deadband,
            controller[0] + config.perplexity_adaptation_rate *
            init_entropy_strength * entropy_error,
            controller[0] + 0.5 * config.perplexity_adaptation_rate *
            (baseline_strength - controller[0]),
        )
        new_entropy_strength = jnp.clip(
            new_entropy_strength, baseline_strength * 0.1,
            init_entropy_strength * 10.0,
        )
        kl_error = kl - config.kl_target
        new_kl_strength = jnp.where(
            kl_error > config.kl_deadband,
            controller[2] + config.kl_adaptation_rate * kl_error,
            controller[2] + 0.1 * config.kl_adaptation_rate *
            (config.kl_strength - controller[2]),
        )
        new_kl_strength = jnp.clip(new_kl_strength,
                                    config.kl_strength, config.kl_strength * 100.0)
        new_controller = jax.lax.stop_gradient(
            jnp.stack([new_entropy_strength, ema, new_kl_strength])
        )

        loss = (policy_loss + config.value_strength * value_loss
                - new_controller[0] * entropy + new_controller[2] * kl)
        clip_fraction = (jnp.abs(ratio - 1.0) > config.ppo_clip).mean()
        metrics = jnp.stack([loss, policy_loss, value_loss, entropy, ppl,
                             kl, clip_fraction])
        return loss, (new_controller, metrics)

    (_, (new_controller, metrics)), grads = jax.value_and_grad(
        objective, has_aux=True
    )(params["trainable_variables"])
    updates, opt_state = optimizer.update(grads, opt_state, params["trainable_variables"])
    trainable = optax.apply_updates(
        params["trainable_variables"],
        jax.tree.map(lambda x: -learning_rate * x, updates),
    )
    params = {**params, "trainable_variables": trainable}
    return params, opt_state, new_controller, metrics


@jax.jit
def evaluate(rng_key, params, opp_coef):
    """Keep main.py's player-0 baseline protocol (including greedy policy)."""
    our_player = 0
    rng_key, sub_key = jax.random.split(rng_key)
    n = config.evaluation_vmap
    states = jax.vmap(env.init)(jax.random.split(sub_key, n))

    def body_fn(loop_state):
        key, states, rewards = loop_state
        key, sub_key = jax.random.split(key)
        logits, _ = network(states.observation, params)
        our_logits = 10000 * logits
        opp_logits = opp_coef * baseline(states.observation)[0]
        logits = jnp.where((states.current_player == our_player)[:, None],
                           our_logits, opp_logits)
        logits = logits + jnp.log(states.legal_action_mask)
        actions = jax.random.categorical(sub_key, logits, axis=-1)
        # Only count live games. Ended trajectories must not accumulate a
        # terminal reward again while the other vectorized games play.
        active = ~(states.terminated | states.truncated)
        states = jax.vmap(lambda s, a: jax.lax.cond(
            (s.terminated | s.truncated), lambda: s, lambda: env.step(s, a)
        ))(states, actions)
        rewards = rewards + jnp.where(active, states.rewards[:, our_player], 0.0)
        return key, states, rewards

    _, _, reward = jax.lax.while_loop(
        lambda x: ~(x[1].terminated | x[1].truncated).all(), body_fn,
        (rng_key, states, jnp.zeros(n)),
    )
    return jnp.mean(reward == 1), jnp.mean(reward == 0), jnp.mean(reward == -1)


def cosine_target(iteration, start, end):
    total = max(1, math.ceil(config.limit_simulator_evaluations /
                             (config.selfplay_vmap * config.selfplay_step)))
    progress = np.clip((iteration / max(total - 1, 1) - 0.01) / 0.98, 0.0, 1.0)
    progress = (1.0 - math.cos(math.pi * progress)) / 2.0
    return float(start + (end - start) * progress)


def learning_rate_at(iteration):
    if not config.lr_schedule:
        return config.learning_rate
    total = max(1, math.ceil(config.limit_simulator_evaluations /
                             (config.selfplay_vmap * config.selfplay_step)))
    warmup = min((iteration + 1) / max(config.warmup_iterations, 1), 1.0)
    decay = np.clip((iteration / max(total - 1, 1) - 0.5) / 0.5, 0.0, 1.0)
    scale = config.min_lr_scale + (1.0 - config.min_lr_scale) * (
        1.0 + math.cos(math.pi * decay)) / 2.0
    return config.learning_rate * warmup * scale


def update_role(params, opt_state, controller, data, role, threshold, entropy_init,
                target_ppl, lr, rng):
    indices = np.flatnonzero(data["ROLE"] == role)
    if len(indices) == 0:
        raise RuntimeError(f"No transitions collected for role {role}")

    # Negative multipliers are intentional, as in Century's AdvShape.
    # Only the policy receives the multiplier; the critic remains unshaped.
    batch_size = min(config.fitting_batch_size, len(indices))
    batches_per_epoch = max(1, len(indices) // batch_size)
    metrics = []
    for _ in range(config.fitting_epochs):
        rng.shuffle(indices)
        # Look up advantages by original flat index after each shuffle.
        for start in range(0, batches_per_epoch * batch_size, batch_size):
            idx = indices[start:start + batch_size]
            batch = {
                "C": jnp.asarray(data["C"][idx]),
                "L": jnp.asarray(data["L"][idx]),
                "A": jnp.asarray(data["A"][idx]),
                "LOGP": jnp.asarray(data["LOGP"][idx]),
                "ADV": jnp.asarray(data["NORMALIZED_ADV"][idx] * threshold),
                "TARGET": jnp.asarray(data["TARGET"][idx]),
            }
            params, opt_state, controller, stats = train_step(
                params, opt_state, controller, batch,
                jnp.asarray(target_ppl), jnp.asarray(entropy_init), jnp.asarray(lr),
            )
            metrics.append(stats)
    avg_stats = np.asarray(jax.device_get(jnp.mean(jnp.stack(metrics), axis=0)))
    return params, opt_state, controller, avg_stats, batches_per_epoch * batch_size


def attach_normalized_advantages(data):
    # Each role is normalized independently BEFORE advantage shaping.
    normalized = np.empty_like(data["ADV"])
    for role in (0, 1):
        ids = data["ROLE"] == role
        if not ids.any():
            raise RuntimeError(f"Missing transitions for role {role}")
        adv = data["ADV"][ids]
        normalized[ids] = (adv - adv.mean()) / (adv.std() + 1e-8)
    data["NORMALIZED_ADV"] = normalized


def save_models(path, agent_params, environment_params):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for role, model, params in (
        ("agent", agent_model, agent_params),
        ("environment", environment_model, environment_params),
    ):
        for variable, value in zip(model.trainable_variables,
                                   params["trainable_variables"]):
            variable.assign(value)
        for variable, value in zip(model.non_trainable_variables,
                                   params["non_trainable_variables"]):
            variable.assign(value)
        model.save(f"{path}-{role}.keras")


baseline_id = config.env_id + "_v0"
if baseline_id in get_args(pgx.BaselineModelId):
    baseline = pgx.make_baseline_model(baseline_id)
else:
    def baseline(obs):
        return jnp.zeros((obs.shape[0], env.num_actions)), None


def main():
    if config.wandb_on:
        wandb.init(project="klent", name="ours", config=config.model_dump())
    print(config)
    keys = KeyGenerator(config.seed)
    rng = np.random.default_rng(config.seed)
    a_params, e_params = get_params(agent_model), get_params(environment_model)
    a_opt = optimizer.init(a_params["trainable_variables"])
    e_opt = optimizer.init(e_params["trainable_variables"])
    a_control = init_controller(config.agent_entropy_strength)
    e_control = init_controller(config.environment_entropy_strength)
    ckpt_dir = os.path.join(config.checkpoints_dir,
                            f"{config.env_id}_{config.seed}", "ours")
    os.makedirs(ckpt_dir, exist_ok=True)
    selfplay_hours = preprocess_hours = fit_hours = 0.0
    frames_used = 0
    frames_per_iteration = config.selfplay_vmap * config.selfplay_step

    for iteration in itertools.count():
        t0 = time()
        trajectory = jax.device_get(selfplay(keys(), a_params, e_params))
        t1 = time()
        done = trajectory["D"]
        outcomes = trajectory["AGENT_REWARD"]
        n_finished = int(done.sum())
        if n_finished:
            agent_wins = float(np.count_nonzero(done & (outcomes > 0)) / n_finished)
            env_wins = float(np.count_nonzero(done & (outcomes < 0)) / n_finished)
            agent_factor = config.agent_threshold - agent_wins
            env_factor = config.environment_threshold - env_wins
        else:
            # No observed outcome means the win-rate controller has no signal.
            agent_wins = env_wins = float("nan")
            agent_factor = env_factor = 1.0
        attach_normalized_advantages(trajectory)
        t2 = time()

        lr = learning_rate_at(iteration)
        a_target = cosine_target(iteration, config.agent_perplexity_start,
                                 config.agent_perplexity_end)
        e_target = cosine_target(iteration, config.environment_perplexity_start,
                                 config.environment_perplexity_end)
        a_params, a_opt, a_control, a_metrics, a_used = update_role(
            a_params, a_opt, a_control, trajectory, 0, agent_factor,
            config.agent_entropy_strength, a_target, lr, rng,
        )
        e_params, e_opt, e_control, e_metrics, e_used = update_role(
            e_params, e_opt, e_control, trajectory, 1, env_factor,
            config.environment_entropy_strength, e_target, lr, rng,
        )
        jax.block_until_ready(e_params["trainable_variables"][0])
        t3 = time()
        frames_used += a_used + e_used
        selfplay_hours += (t1 - t0) / 3600
        preprocess_hours += (t2 - t1) / 3600
        fit_hours += (t3 - t2) / 3600

        W, D, L = map(float, evaluate(keys(), a_params, 1.0))
        simulator_evaluations = frames_per_iteration * (iteration + 1)
        total_loss = float(a_metrics[0] + e_metrics[0])
        log = {
            "cost/iteration": iteration + 1,
            "cost/simulator_evaluations/planning": 0,
            "cost/simulator_evaluations/playing": simulator_evaluations,
            "cost/simulator_evaluations/total": simulator_evaluations,
            "cost/simulator_evaluations/total [million]": simulator_evaluations / 1e6,
            "cost/hours/selfplay": selfplay_hours,
            "cost/hours/preprocess": preprocess_hours,
            "cost/hours/fit": fit_hours,
            "cost/hours/total": selfplay_hours + preprocess_hours + fit_hours,
            "cost/gpu_hours/selfplay": selfplay_hours,
            "cost/gpu_hours/fit": fit_hours,
            "cost/gpu_hours/total": selfplay_hours + preprocess_hours + fit_hours,
            "cost/frames/total": simulator_evaluations,
            "cost/frames/used": frames_used,
            "stats/sample_util_ratio": frames_used / simulator_evaluations,
            "rollout/finished_games": n_finished,
            "rollout/agent_win_rate": agent_wins,
            "rollout/environment_win_rate": env_wins,
            "rollout/agent_advantage_multiplier": agent_factor,
            "rollout/environment_advantage_multiplier": env_factor,
            "eval/vs_baseline_100/win_rate": W,
            "eval/vs_baseline_100/draw_rate": D,
            "eval/vs_baseline_100/lose_rate": L,
            "eval/vs_baseline_100/avg_R": W - L,
            "eval/vs_baseline_100/score": W + 0.5 * D,
            "score/vs_baseline_100": W + 0.5 * D,
            "train/total_loss": total_loss,
            "train/policy_loss": float(a_metrics[1] + e_metrics[1]),
            "train/qvalue_loss": float(a_metrics[2] + e_metrics[2]),
            "train/learning_rate": lr,
        }
        for prefix, metrics, control, target in (
            ("agent", a_metrics, a_control, a_target),
            ("environment", e_metrics, e_control, e_target),
        ):
            coef, ppl_ema, kl_coef = map(float, control)
            log.update({
                f"train/{prefix}/total_loss": float(metrics[0]),
                f"train/{prefix}/policy_loss": float(metrics[1]),
                f"train/{prefix}/qvalue_loss": float(metrics[2]),
                f"train/{prefix}/entropy": float(metrics[3]),
                f"train/{prefix}/perplexity": float(metrics[4]),
                f"train/{prefix}/kl": float(metrics[5]),
                f"train/{prefix}/clip_fraction": float(metrics[6]),
                f"train/{prefix}/perplexity_target": target,
                f"train/{prefix}/perplexity_ema": ppl_ema,
                f"train/{prefix}/entropy_strength": coef,
                f"train/{prefix}/kl_strength": kl_coef,
            })
        print(f"Iteration {iteration + 1}: {log}")
        if config.wandb_on:
            wandb.log(log)
        if (iteration + 1) % config.save_interval == 0:
            save_models(os.path.join(ckpt_dir, f"{iteration+1:04}"), a_params, e_params)
        if simulator_evaluations >= config.limit_simulator_evaluations:
            save_models(os.path.join(ckpt_dir, "final"), a_params, e_params)
            break


if __name__ == "__main__":
    main()