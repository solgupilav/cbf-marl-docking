"""Monte Carlo evaluation of a trained 6-DOF docking policy.

Runs independent episodes from a checkpoint and reports the per-agent and
all-agents success rates (95% Wilson interval), target and inter-agent
contacts, delta-v and filter intervention.

Contact is a geometric test made in this script:
  target contact       box overlap of the chaser and the target
  inter-agent contact  centre distance < 2 * bounding-sphere radius
The first contact ends the episode as a failure. Each episode is a success,
a collision or a timeout.

If config.pkl is two levels above the checkpoint (the BenchMARL run layout),
the run is reloaded with its own configuration.

Usage:
    python scripts/mc_eval_6dof.py --checkpoint <run>/checkpoints/<file>.pt
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import torch

# Make src/ and the repository root importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from torchrl.envs.utils import ExplorationType, set_exploration_type  # noqa: E402

# Capture gate for scoring (m). It does not depend on the curriculum state of
# the checkpoint. Change it with --goal-tolerance.
EVAL_GOAL_TOLERANCE = 0.05

# Filter during evaluation. "auto" uses the training setting. "on"/"off"
# force it (for ablations). Change it with --cbf.
EVAL_CBF_MODE = "auto"

# Sensor noise on the observation. Same auto/on/off rule as EVAL_CBF_MODE.
# Reward, physics and the filter do not change.
EVAL_SENSOR_NOISE_MODE = "auto"


def apply_cbf_mode(task, mode: str) -> bool:
    """Set enable_cbf on the task config and return it. "auto" keeps the training
    value. The cbf_* parameters and enable_safety_obs do not change.
    """
    trained = bool(task.config.get("enable_cbf", False))
    if mode == "auto":
        effective = trained
    elif mode in ("on", "off"):
        effective = mode == "on"
    else:
        raise ValueError(f"Unknown CBF mode: {mode!r} (expected auto/on/off)")
    task.config["enable_cbf"] = effective
    if effective != trained:
        print(
            f"[Config] WARNING: filter forced {'ON' if effective else 'OFF'}. "
            f"Training used enable_cbf={trained}."
        )
    return effective


def apply_sensor_noise_mode(task, mode: str) -> bool:
    """Set enable_sensor_noise on the task config. Same rule as apply_cbf_mode.
    The noise magnitudes do not change.
    """
    trained = bool(task.config.get("enable_sensor_noise", False))
    if mode == "auto":
        effective = trained
    elif mode in ("on", "off"):
        effective = mode == "on"
    else:
        raise ValueError(f"Unknown sensor-noise mode: {mode!r} (expected auto/on/off)")
    task.config["enable_sensor_noise"] = effective
    if effective != trained:
        print(
            f"[Config] WARNING: sensor noise forced {'ON' if effective else 'OFF'}. "
            f"Training used enable_sensor_noise={trained}."
        )
    return effective


# Actuator uncertainty presets. The default is "off" (exact actuation).
# "typical" and "high" are the cells of the actuator table.
ACTUATOR_PRESETS = {
    "high":    dict(scale_bias=0.10, scale_noise=0.05, align=0.0349),   # 10%, 2 deg
    "typical": dict(scale_bias=0.025, scale_noise=0.01, align=0.0087),  # 2.5%, 0.5 deg
}


def apply_actuator_mode(task, mode: str) -> bool:
    """Set actuator uncertainty and return True if it is on.
    This changes the plant, so the filter output is no longer the applied
    control.
    """
    if mode == "off":
        task.config["enable_actuator_uncertainty"] = False
        return False
    if mode not in ACTUATOR_PRESETS:
        raise ValueError(f"Unknown actuator mode: {mode!r} (expected off/typical/high)")
    q = ACTUATOR_PRESETS[mode]
    task.config["enable_actuator_uncertainty"] = True
    task.config["actuator_lin_scale_bias_std"] = q["scale_bias"]
    task.config["actuator_ang_scale_bias_std"] = q["scale_bias"]
    task.config["actuator_lin_scale_noise_std"] = q["scale_noise"]
    task.config["actuator_ang_scale_noise_std"] = q["scale_noise"]
    task.config["actuator_lin_align_std"] = q["align"]
    task.config["actuator_ang_align_std"] = q["align"]
    print(
        f"[Config] Actuator uncertainty '{mode}': scale bias "
        f"{q['scale_bias']:.1%}, per-step jitter {q['scale_noise']:.1%}, "
        f"misalignment {q['align'] * 180 / 3.14159265:.2f} deg (1-sigma)."
    )
    return True


def apply_keepout_mode(task, mode: str) -> bool:
    """Set the box-covering target keep-out and return the setting.
    Same auto/on/off rule as apply_cbf_mode. "on" also sets the superellipsoid
    exponent to 16, because a lower exponent makes the keep-out cover the ports.
    """
    trained = bool(task.config.get("cbf_target_keepout_cover_box", False))
    if mode == "auto":
        effective = trained
    elif mode in ("on", "off"):
        effective = mode == "on"
    else:
        raise ValueError(f"Unknown keep-out mode: {mode!r} (expected auto/on/off)")
    task.config["cbf_target_keepout_cover_box"] = effective
    if effective != trained:
        task.config["cbf_target_keepout_exponent"] = 16 if effective else 8
        print(
            f"[Config] WARNING: box-covering keep-out forced "
            f"{'ON' if effective else 'OFF'} (exponent "
            f"{task.config['cbf_target_keepout_exponent']}). "
            f"Training used cbf_target_keepout_cover_box={trained}."
        )
    return effective


# Statistics.
def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + (z * z) / n
    centre = (p + (z * z) / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + (z * z) / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


# Policy and task loading.
def _register_train_callbacks_for_unpickle() -> None:
    """Make the training callbacks available as __main__.<Name> for unpickling.
    config.pkl stores them under "__main__", because training runs the train
    script as the main module.
    """
    import __main__

    try:
        import scripts.train_benchmarl_6dof as train_mod
    except Exception as exc:
        print(f"[Load] Could not import train callbacks for unpickle: {exc}")
        return
    # Register every Callback and Experiment subclass in the train script.
    # A callback keeps a reference to the experiment, so both are in the pickle.
    from benchmarl.experiment import Experiment
    from benchmarl.experiment.callback import Callback

    for name in dir(train_mod):
        obj = getattr(train_mod, name)
        if not isinstance(obj, type):
            continue
        if issubclass(obj, (Callback, Experiment)) and not hasattr(__main__, name):
            setattr(__main__, name, obj)


def load_policy_and_task(checkpoint: str, *, force_train_config: bool = False):
    """Return (policy, task). Use the run's config.pkl if it exists."""
    ckpt_path = Path(checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    config_pkl = ckpt_path.parent.parent / "config.pkl"
    if config_pkl.exists() and not force_train_config:
        from benchmarl.experiment import Experiment

        _register_train_callbacks_for_unpickle()

        # BenchMARL writes into the run folder when it loads a run. Load from a
        # temporary copy, so the checkpoint folder does not change.
        tmp_run = Path(tempfile.mkdtemp(prefix="mc_eval_")) / ckpt_path.parent.parent.name
        atexit.register(shutil.rmtree, tmp_run.parent, True)
        (tmp_run / "checkpoints").mkdir(parents=True)
        shutil.copy2(config_pkl, tmp_run / "config.pkl")
        tmp_ckpt = tmp_run / "checkpoints" / ckpt_path.name
        shutil.copy2(ckpt_path, tmp_ckpt)
        print(f"[Load] Reloading experiment from run config: {config_pkl}")
        experiment = Experiment.reload_from_file(str(tmp_ckpt))
    else:
        print("[Load] No config.pkl. Rebuilding the config from "
              "scripts/train_benchmarl_6dof.py.")
        from scripts.train_benchmarl_6dof import build_experiment

        experiment = build_experiment(save_eval_videos=False)
        experiment.config.restore_file = str(ckpt_path)
        experiment.config.restore_map_location = "cpu"
        experiment._load_experiment()

    policy = experiment.policy
    if policy is None:
        raise RuntimeError("Could not extract policy from the experiment.")
    policy.eval()
    # init_near_goal_fraction is a training aid. Set it to 0 for scoring, so all
    # agents start in the spawn box.
    task = experiment.task
    if float(task.config.get("init_near_goal_fraction", 0.0)) != 0.0:
        print("[Config] init_near_goal_fraction set to 0 for scoring "
              f"(training used {task.config['init_near_goal_fraction']}).")
        task.config["init_near_goal_fraction"] = 0.0
    return policy, task


# Contact checks (no smoothing).
def discrete_target_contact(env) -> torch.Tensor:
    """[B, N] bool: chaser box overlaps target box on all three axes."""
    agent_half = 0.5 * env.agent_size.view(1, 1, 3)
    target_half = 0.5 * env.target_size.view(1, 1, 3)
    target_center = env.target_center.unsqueeze(1)
    overlap = (
        torch.minimum(env.pos + agent_half, target_center + target_half)
        - torch.maximum(env.pos - agent_half, target_center - target_half)
    )
    return (overlap > 0.0).all(dim=-1)


def discrete_agent_contact(env) -> torch.Tensor:
    """[B, N] bool: centre distance to the closest teammate < 2 * r_bound.
    r_bound is the bounding-sphere radius, so the test does not depend on attitude.
    """
    n = env.n_agents
    if n < 2:
        return torch.zeros(env.num_envs, n, dtype=torch.bool, device=env.device)
    diff = env.pos.unsqueeze(2) - env.pos.unsqueeze(1)  # [B, N, N, 3]
    dist = diff.norm(dim=-1)
    eye = torch.eye(n, dtype=torch.bool, device=env.device)
    dist = dist.masked_fill(eye.unsqueeze(0), float("inf"))
    contact_dist = 2.0 * 0.5 * torch.linalg.norm(env.agent_size)
    return dist.min(dim=-1).values < contact_dist


def closest_pair_distance(env) -> torch.Tensor:
    """[B] float: centre distance of the closest agent pair at this step.
    Docked agents are included.
    """
    n = env.n_agents
    if n < 2:
        return torch.full((env.num_envs,), float("inf"), device=env.device)
    diff = env.pos.unsqueeze(2) - env.pos.unsqueeze(1)
    dist = diff.norm(dim=-1)
    eye = torch.eye(n, dtype=torch.bool, device=env.device)
    return dist.masked_fill(eye.unsqueeze(0), float("inf")).amin(dim=(1, 2))


def fake_quantize_weights(policy, mode: str) -> dict:
    """Round the policy weights to int8 or fp16, then convert back to fp32.
    int8 uses symmetric scaling per output channel (scale = max|w| / 127).
    Only weight matrices change. Biases and 1-D parameters stay fp32.
    """
    import torch as _t

    n_q = n_kept = 0
    max_err = 0.0
    with _t.no_grad():
        for name, prm in policy.named_parameters():
            if prm.dim() < 2:
                n_kept += prm.numel()
                continue
            w = prm.data
            if mode == "fp16":
                q = w.half().float()
            elif mode == "int8":
                flat = w.reshape(w.shape[0], -1)
                scale = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
                q = (flat / scale).round().clamp(-127, 127).mul(scale).reshape(w.shape)
            else:
                raise ValueError(mode)
            max_err = max(max_err, float((q - w).abs().max()))
            prm.data.copy_(q)
            n_q += prm.numel()
    return {"quantized": n_q, "kept_fp32": n_kept, "max_abs_weight_error": max_err}


class LQRNominal:
    """LQR and quaternion PD in place of the learned policy.

    The controller uses the same filter, scoring and delta-v count as the policy.
    It reads the observation, so it sees the same noisy estimate.
    Gains are the same as scripts/lqr_baseline_6dof.py, including the plus sign
    on the attitude proportional term.
    """

    def __init__(self, task, kp_att: float, kd_att: float, r_max: float, v_max: float):
        import numpy as _np
        from scripts.lqr_baseline_6dof import cw_matrices, dlqr

        c = task.config
        dt = float(c["dt"])
        self.a_bar = float(c["linear_accel_scale"])
        self.al_bar = float(c["angular_accel_scale"])
        self.pos_scale = float(c.get("obs_pos_scale", 10.0))
        self.vel_scale = float(c.get("obs_vel_scale", 1.0))
        self.kp, self.kd = float(kp_att), float(kd_att)

        probe = task.get_env_fun(num_envs=1, continuous_actions=True, seed=0, device="cpu")()
        n_mm = float(probe.mean_motion)
        Ad, Bd = cw_matrices(n_mm, dt)
        Q = _np.diag([1.0 / r_max**2] * 3 + [1.0 / v_max**2] * 3)
        R = _np.eye(3) * (1.0 / self.a_bar**2)
        self.K = torch.tensor(dlqr(Ad, Bd, Q, R), dtype=torch.float32)
        self._group = None
        print(f"[LQR] mean motion {n_mm:.3e} rad/s, dt {dt} s, |a| <= {self.a_bar} m/s^2, "
              f"attitude PD kp={self.kp} kd={self.kd}")

    def __call__(self, td):
        if self._group is None:
            keys = td.keys(include_nested=True)
            self._group = next(k[0] for k in keys if isinstance(k, tuple)
                               and len(k) == 2 and k[1] == "observation")
        obs = td[(self._group, "observation")]
        # obs = [(goal-pos)/pos_scale, vel/vel_scale, 2*q_err_vec, ang_err, ...]
        rel_goal = obs[..., 0:3] * self.pos_scale       # goal - pos
        vel = obs[..., 3:6] * self.vel_scale
        err = torch.cat([-rel_goal, vel], dim=-1)       # [pos - goal, vel]
        a_cmd = (-(err @ self.K.T)).clamp(-self.a_bar, self.a_bar)
        att_vec, ang_err = obs[..., 6:9], obs[..., 9:12]
        al_cmd = (self.kp * att_vec - self.kd * ang_err).clamp(-self.al_bar, self.al_bar)
        act = torch.cat([a_cmd / self.a_bar, al_cmd / self.al_bar], dim=-1).clamp(-1.0, 1.0)
        td[(self._group, "action")] = act
        return td


# Monte Carlo rollout.
@torch.no_grad()
def run_monte_carlo(
    *,
    policy,
    task,
    episodes: int,
    batch_size: int,
    seed: int,
    deterministic: bool = True,
) -> dict:
    """Run `episodes` independent episodes, one per env slot, without auto-reset."""
    n_agents = int(task.config["n_agents"])
    max_steps = int(task.config["max_steps"])

    # Totals over all episodes.
    agent_success = torch.zeros(0, n_agents, dtype=torch.bool)
    agent_target_coll = torch.zeros(0, n_agents, dtype=torch.bool)
    agent_agent_coll = torch.zeros(0, n_agents, dtype=torch.bool)
    episode_len = torch.zeros(0, dtype=torch.long)
    min_separation = torch.zeros(0, dtype=torch.float32)
    dock_step = torch.zeros(0, n_agents, dtype=torch.float32)
    cbf_interv_mean = torch.zeros(0, n_agents, dtype=torch.float32)
    cbf_infeas_mean = torch.zeros(0, n_agents, dtype=torch.float32)
    cbf_relaxed_mean = torch.zeros(0, n_agents, dtype=torch.float32)
    delta_v_total = torch.zeros(0, n_agents, dtype=torch.float32)
    ang_delta_v_total = torch.zeros(0, n_agents, dtype=torch.float32)
    # Filter intervention against true range to the own port, for agents not yet
    # docked. Saved with --arrays-out.
    iv_samples: list[torch.Tensor] = []
    range_samples: list[torch.Tensor] = []

    remaining = episodes
    round_idx = 0
    group = None
    while remaining > 0:
        b = min(batch_size, remaining)
        env = task.get_env_fun(
            num_envs=b, continuous_actions=True, seed=seed + round_idx, device="cpu"
        )()
        group = env.group
        td = env.reset()

        env_active = torch.ones(b, dtype=torch.bool)  # episode still running
        ep_len = torch.full((b,), max_steps, dtype=torch.long)
        docked = torch.zeros(b, n_agents, dtype=torch.bool)
        t_dock = torch.full((b, n_agents), float("nan"))
        hit_target = torch.zeros(b, n_agents, dtype=torch.bool)
        hit_agent = torch.zeros(b, n_agents, dtype=torch.bool)
        min_sep = torch.full((b,), float("inf"))
        iv_sum = torch.zeros(b, n_agents)  # CBF intervention, summed over steps
        infeas_sum = torch.zeros(b, n_agents)  # CBF hard-infeasible substep fraction
        relaxed_sum = torch.zeros(b, n_agents)  # CBF soft-row-dropped substep fraction
        dv_sum = torch.zeros(b, n_agents)  # linear delta-v (m/s), summed over the episode
        adv_sum = torch.zeros(b, n_agents)  # angular delta-v (rad/s), summed

        with set_exploration_type(
            ExplorationType.DETERMINISTIC if deterministic else ExplorationType.RANDOM
        ):
            for step in range(max_steps):
                td = policy(td)
                nxt = env.step(td)["next"]
                active = env_active.unsqueeze(-1)  # [B, 1]

                # Capture per agent, counted only while the episode runs.
                on_goal = nxt[(group, "info", "on_goal")][..., 0].bool()  # [B, N]
                newly = on_goal & ~docked & active
                t_dock[newly] = float(step + 1)
                docked |= newly

                # Contact events for running episodes. The first target or inter-agent
                # contact ends the episode as a failure.
                tc = discrete_target_contact(env) & active  # [B, N]
                ac = discrete_agent_contact(env) & active
                hit_target |= tc
                hit_agent |= ac

                # Running minimum of the closest-pair distance.
                min_sep = torch.minimum(
                    min_sep,
                    torch.where(env_active, closest_pair_distance(env),
                                torch.full_like(min_sep, float("inf"))),
                )
                contact_now = (tc | ac).any(dim=1)  # [B]

                # Filter intervention ||u_safe - u_nom|| (substep mean, normalised units).
                # It is zero when the filter is off.
                iv_step = nxt[(group, "info", "cbf_intervention")][..., 0]
                iv_sum += iv_step * active.float()
                live = active & ~on_goal
                if bool(live.any()):
                    iv_samples.append(iv_step[live].clone())
                    range_samples.append((env.goals - env.pos).norm(dim=-1)[live].clone())
                info_td = nxt[(group, "info")]
                if "cbf_infeasible" in info_td.keys():
                    infeas_sum += info_td["cbf_infeasible"][..., 0] * active.float()
                if "cbf_relaxed" in info_td.keys():
                    relaxed_sum += info_td["cbf_relaxed"][..., 0] * active.float()

                # Delta-v of running episodes. Docked agents add zero.
                if "delta_v" in info_td.keys():
                    dv_sum += info_td["delta_v"][..., 0] * active.float()
                    adv_sum += info_td["angular_delta_v"][..., 0] * active.float()

                done_now = (nxt["done"][..., 0].bool() | contact_now) & env_active
                ep_len[done_now] = step + 1
                env_active &= ~done_now
                td = nxt
                if not env_active.any():
                    break

        agent_success = torch.cat([agent_success, docked])
        agent_target_coll = torch.cat([agent_target_coll, hit_target])
        agent_agent_coll = torch.cat([agent_agent_coll, hit_agent])
        episode_len = torch.cat([episode_len, ep_len])
        min_separation = torch.cat([min_separation, min_sep])
        dock_step = torch.cat([dock_step, t_dock])
        cbf_interv_mean = torch.cat(
            [cbf_interv_mean, iv_sum / ep_len.unsqueeze(-1).float().clamp(min=1)]
        )
        cbf_infeas_mean = torch.cat(
            [cbf_infeas_mean, infeas_sum / ep_len.unsqueeze(-1).float().clamp(min=1)]
        )
        cbf_relaxed_mean = torch.cat(
            [cbf_relaxed_mean, relaxed_sum / ep_len.unsqueeze(-1).float().clamp(min=1)]
        )
        delta_v_total = torch.cat([delta_v_total, dv_sum])
        ang_delta_v_total = torch.cat([ang_delta_v_total, adv_sum])

        remaining -= b
        round_idx += 1
        print(f"[MC] round {round_idx}: {episodes - remaining}/{episodes} episodes done")

    return {
        "n_agents": n_agents,
        "episodes": episodes,
        "agent_success": agent_success,
        "agent_target_coll": agent_target_coll,
        "agent_agent_coll": agent_agent_coll,
        "episode_len": episode_len,
        "min_separation": min_separation,
        "dock_step": dock_step,
        "cbf_intervention_mean": cbf_interv_mean,
        "cbf_infeasible_mean": cbf_infeas_mean,
        "cbf_relaxed_mean": cbf_relaxed_mean,
        "delta_v_total": delta_v_total,
        "ang_delta_v_total": ang_delta_v_total,
        "cbf_enabled": bool(task.config.get("enable_cbf", False)),
        "iv_flat": torch.cat(iv_samples) if iv_samples else torch.zeros(0),
        "range_flat": torch.cat(range_samples) if range_samples else torch.zeros(0),
    }


# Reporting.
def _per_agent_line(flags: torch.Tensor) -> str:
    rates = flags.float().mean(dim=0)
    return "  ".join(f"a{i}: {r:6.1%}" for i, r in enumerate(rates))


def _per_agent_line_f(values: torch.Tensor) -> str:
    means = values.float().mean(dim=0)
    return "  ".join(f"a{i}: {v:.4f}" for i, v in enumerate(means))


def print_report(stats: dict) -> dict:
    n = stats["episodes"]
    success = stats["agent_success"]
    t_coll = stats["agent_target_coll"]
    a_coll = stats["agent_agent_coll"]

    any_target = t_coll.any(dim=1)
    any_agent = a_coll.any(dim=1)
    any_collision = any_target | any_agent  # episode ended in a collision

    # Success means all agents docked and no contact. Success, collision and
    # timeout add up to n.
    complete = success.all(dim=1) & ~any_collision
    n_complete = int(complete.sum())
    ci_lo, ci_hi = wilson_ci(n_complete, n)
    n_collision = int(any_collision.sum())
    n_timeout = n - n_complete - n_collision

    print("\n" + "=" * 72)
    print(f"MONTE CARLO REPORT  ({n} episodes, {stats['n_agents']} agents)")
    print("=" * 72)
    print("Episode outcome (a collision ends the episode):")
    print(f"  SUCCESS (all agents docked, no collision): {n_complete}/{n} = "
          f"{n_complete / n:.1%}   [95% CI {ci_lo:.1%} - {ci_hi:.1%}]")
    print(f"  FAILED - collision:                        {n_collision}/{n} = "
          f"{n_collision / n:.1%}")
    print(f"  FAILED - timeout (no collision, not all docked): {n_timeout}/{n} = "
          f"{n_timeout / n:.1%}")
    print("\nSuccess (docking capture):")
    print(f"  per-agent : {_per_agent_line(success)}")
    print(f"  mean agent: {success.float().mean():.1%}")
    print("\nCollisions (discrete contact, no smoothing; episode-terminal):")
    print("  with target:")
    print(f"    per-agent : {_per_agent_line(t_coll)}")
    print(f"    mean agent: {t_coll.float().mean():.1%}")
    print(f"    episodes with any target contact: {any_target.float().mean():.1%}")
    print("  inter-agent:")
    print(f"    per-agent : {_per_agent_line(a_coll)}")
    print(f"    mean agent: {a_coll.float().mean():.1%}")
    print(f"    episodes with any inter-agent contact: {any_agent.float().mean():.1%}")

    if stats.get("cbf_enabled", False):
        iv = stats["cbf_intervention_mean"]
        any_iv = (iv > 1e-6).any(dim=1)
        print("\nCBF safety filter (see [Config] lines for training parity):")
        print(f"  per-agent mean intervention ||u_safe - u_nom||: {_per_agent_line_f(iv)}")
        print(f"  mean intervention: {iv.mean():.4f} (normalised action units/step)")
        print(f"  episodes with any intervention: {any_iv.float().mean():.1%}")
        infeas = stats["cbf_infeasible_mean"]
        any_infeas = (infeas > 1e-9).any(dim=1)
        print(
            f"  episodes with a hard-infeasible QP: "
            f"{any_infeas.float().mean():.1%}"
        )
        relaxed = stats.get("cbf_relaxed_mean", None)
        if relaxed is not None and relaxed.numel() > 0:
            any_relaxed = (relaxed > 1e-9).any(dim=1)
            print(
                f"  episodes with a soft row dropped: {any_relaxed.float().mean():.1%}  "
                f"fraction of steps per agent: {relaxed.mean():.4f}"
            )

    # Delta-v per episode: linear in m/s, angular in rad/s. Also given for
    # successful episodes only.
    dv = stats["delta_v_total"]  # [episodes, n_agents], m/s
    adv = stats["ang_delta_v_total"]  # [episodes, n_agents], rad/s
    print("\nFuel use (delta-v = integral of applied |acceleration|):")
    print(f"  linear  per agent : mean {dv.mean():.4f}  median "
          f"{dv.median():.4f}  max {dv.max():.4f} m/s")
    print(f"  linear  per episode (sum of {stats['n_agents']} agents): "
          f"mean {dv.sum(dim=1).mean():.4f} m/s")
    if adv.abs().sum() > 0:
        print(f"  angular per agent : mean {adv.mean():.4f}  median "
              f"{adv.median():.4f}  max {adv.max():.4f} rad/s")
    if int(complete.sum()) > 0:
        dv_succ = dv[complete].mean()
        print(f"  linear  per agent, successful episodes only: mean {dv_succ:.4f} m/s")

    dock_steps = stats["dock_step"][success]
    print("\nEpisode stats:")
    print(f"  mean episode length: {stats['episode_len'].float().mean():.1f} steps")
    if dock_steps.numel() > 0:
        print(
            f"  time-to-dock (successful agents): mean {dock_steps.mean():.1f}, "
            f"median {dock_steps.median():.0f} steps"
        )

    return {
        "cbf_enabled": bool(stats.get("cbf_enabled", False)),
        "mean_cbf_intervention": float(stats["cbf_intervention_mean"].mean()),
        "any_cbf_relaxed_rate": float(
            (stats["cbf_relaxed_mean"] > 1e-9).any(dim=1).float().mean()
        ) if stats.get("cbf_relaxed_mean", None) is not None and stats["cbf_relaxed_mean"].numel() > 0 else 0.0,
        "any_cbf_infeasible_rate": float(
            (stats["cbf_infeasible_mean"] > 1e-9).any(dim=1).float().mean()
        ),
        "episodes": n,
        "per_agent_success": success.float().mean(dim=0).tolist(),
        "mean_agent_success": float(success.float().mean()),
        "complete_success_rate": n_complete / n,
        "complete_success_ci95": [ci_lo, ci_hi],
        "collision_failure_rate": n_collision / n,
        "timeout_failure_rate": n_timeout / n,
        "mean_linear_delta_v_per_agent": float(stats["delta_v_total"].mean()),
        "mean_linear_delta_v_per_episode": float(stats["delta_v_total"].sum(dim=1).mean()),
        "mean_angular_delta_v_per_agent": float(stats["ang_delta_v_total"].mean()),
        "mean_linear_delta_v_successful": (
            float(stats["delta_v_total"][complete].mean())
            if int(complete.sum()) > 0
            else None
        ),
        "per_agent_target_collision": t_coll.float().mean(dim=0).tolist(),
        "mean_agent_target_collision": float(t_coll.float().mean()),
        "any_target_collision_rate": float(any_target.float().mean()),
        "per_agent_interagent_collision": a_coll.float().mean(dim=0).tolist(),
        "mean_agent_interagent_collision": float(a_coll.float().mean()),
        "any_interagent_collision_rate": float(any_agent.float().mean()),
        "mean_episode_len": float(stats["episode_len"].float().mean()),
    }


def _retarget_policy_agent_count(policy, n_agents: int) -> None:
    """Set a shared-parameter policy to a different agent count.
    With share_params=True the weights have no agent axis, so changing n_agents
    is exact. Raise an error if a module does not share parameters.
    """
    import torchrl.modules.models.multiagent as ma

    n_patched = 0
    for module in policy.modules():
        if isinstance(module, ma.MultiAgentMLP):
            if not getattr(module, "share_params", False):
                raise ValueError(
                    "--n-agents needs a shared-parameter policy "
                    "(share_policy_params=True)."
                )
            module.n_agents = int(n_agents)
            n_patched += 1
    if n_patched == 0:
        raise ValueError(
            "--n-agents: no MultiAgentMLP found in the policy to retarget."
        )


# Main.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint .pt "
                                                            "file.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of Monte Carlo "
                                                                  "episodes.")
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Number of parallel envs per "
                                                    "round."
    )
    parser.add_argument("--seed", type=int, default=1234, help="Evaluation seed.")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy instead of taking the mode.",
    )
    parser.add_argument(
        "--force-train-config",
        action="store_true",
        help="Ignore the run's config.pkl. Build the configuration from "
             "scripts/train_benchmarl_6dof.py.",
    )
    parser.add_argument(
        "--goal-tolerance",
        type=float,
        default=EVAL_GOAL_TOLERANCE,
        help="Capture radius for scoring (m). Default: %(default)s.",
    )
    parser.add_argument(
        "--cbf",
        choices=("auto", "on", "off"),
        default=EVAL_CBF_MODE,
        help="Safety filter. 'auto': use the training setting. 'on'/'off': "
             "force it.",
    )
    parser.add_argument(
        "--sensor-noise",
        choices=("auto", "on", "off"),
        default=EVAL_SENSOR_NOISE_MODE,
        help="Sensor noise on the observation. 'auto': use the training "
             "setting. 'on'/'off': force it. The noise magnitudes come from "
             "the run configuration.",
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=None,
        help="Number of chasers. The spawn box, ports and target size are "
             "made as in training. Default: the training value.",
    )
    parser.add_argument(
        "--keepout-mode",
        choices=("auto", "superellipsoid", "halfspace"),
        default="auto",
        help="Target keep-out geometry. 'halfspace': h_T = p_x - rho_x. This "
             "changes only the filter, so it works with any checkpoint.",
    )
    parser.add_argument(
        "--cbf-sensed-state",
        choices=("off", "on"),
        default="off",
        help="Give the filter the noisy navigation estimate instead of the "
             "true state. Needs sensor noise on.",
    )
    parser.add_argument(
        "--attitude-gate",
        type=float,
        default=None,
        help=(
            "Attitude capture gate (rad). Default: the config value (0.175). "
            "Set it for runs trained with --attitude-curriculum."
        ),
    )
    parser.add_argument(
        "--target-lambda",
        type=float,
        default=None,
        help=(
            "Set both target keep-out poles. Default: the trained value "
            "(0.15)."
        ),
    )
    parser.add_argument(
        "--actuator-uncertainty",
        choices=("off", "typical", "high"),
        default="off",
        help="Actuator error on the applied command: scale bias, misalignment"
             " and jitter. 'typical': 2.5%% and 0.5 deg. 'high': 10%% and 2 "
             "deg. Default: 'off'.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Episode length (steps). Default: the training value.",
    )
    parser.add_argument(
        "--keepout-cover-box",
        choices=("auto", "on", "off"),
        default="auto",
        help="Box-covering target keep-out (exponent 16). 'auto': use the "
             "training setting. 'on'/'off': force it.",
    )
    parser.add_argument(
        "--quantize-weights",
        choices=("none", "fp16", "int8"),
        default="none",
        help="Round the policy weights to this precision. The arithmetic "
             "stays fp32.",
    )
    parser.add_argument(
        "--nominal",
        choices=("policy", "lqr"),
        default="policy",
        help="Nominal controller before the filter. 'lqr': LQR and quaternion"
             " PD instead of the policy.",
    )
    parser.add_argument("--lqr-kp-att", type=float, default=0.02)
    parser.add_argument("--lqr-kd-att", type=float, default=0.25)
    parser.add_argument("--lqr-r-max", type=float, default=12.0)
    parser.add_argument("--lqr-v-max", type=float, default=0.5)
    parser.add_argument("--json-out", default="", help="Path for the results JSON.")
    parser.add_argument(
        "--arrays-out",
        default="",
        help="Path for the per-agent-episode arrays (.npz).",
    )
    args = parser.parse_args()

    policy, task = load_policy_and_task(
        args.checkpoint, force_train_config=args.force_train_config
    )

    if args.quantize_weights != "none":
        _qi = fake_quantize_weights(policy, args.quantize_weights)
        _tot = _qi["quantized"] + _qi["kept_fp32"]
        print(
            f"[Config] Weights quantised to {args.quantize_weights}: "
            f"{_qi['quantized']:,} of {_tot:,} parameters "
            f"({100 * _qi['quantized'] / max(_tot, 1):.1f}%), max change "
            f"{_qi['max_abs_weight_error']:.3e}. Biases stay fp32."
        )

    # Agent-count override. Make the spawn box the same way as build_experiment.
    # The environment lays out the ports and enlarges the target face.
    if args.n_agents is not None:
        n = int(args.n_agents)
        prev = int(task.config["n_agents"])
        task.config["n_agents"] = n
        lat = 2.0 * max(1.0, (n / 4.0) ** 0.5)
        task.config["init_agent_pos_means"] = [[10.0, 0.0, 0.0] for _ in range(n)]
        task.config["init_agent_pos_ranges"] = [[1.5, lat, lat] for _ in range(n)]
        # With share_params=True the weights have no agent axis, so setting n_agents
        # lets the policy run any agent count.
        _retarget_policy_agent_count(policy, n)
        print(
            f"[Config] Evaluating {n} chasers (trained with {prev})."
        )

    # Score at a fixed capture radius, not the curriculum value in the
    # checkpoint. get_env_fun copies task.config on each call, so this change
    # applies to every evaluation env.
    effective_tol = float(args.goal_tolerance)
    cfg_tol = float(task.config.get("goal_tolerance"))
    if abs(effective_tol - cfg_tol) > 1e-9:
        print(
            f"[Config] Scoring at goal_tolerance={effective_tol} m "
            f"(config had {cfg_tol} m)"
        )
    task.config["goal_tolerance"] = effective_tol

    # The same for the attitude gate. The curriculum changes max_attitude_error
    # on the env, not on task.config, so the config still holds 0.175 rad.
    # None keeps the config value.
    cfg_att = float(task.config.get("max_attitude_error", 0.175))
    effective_att = cfg_att if args.attitude_gate is None else float(args.attitude_gate)
    task.config["max_attitude_error"] = effective_att
    print(
        f"[Config] Attitude capture gate {effective_att:.4f} rad "
        f"({effective_att * 180.0 / 3.141592653589793:.2f} deg)"
        + ("" if args.attitude_gate is None
           else f" (config had {cfg_att:.4f} rad)")
    )

    # Filter: use the training setting ("auto") or force it.
    enable_cbf = apply_cbf_mode(task, args.cbf)
    enable_sensor_noise = apply_sensor_noise_mode(task, args.sensor_noise)
    apply_keepout_mode(task, args.keepout_cover_box)
    if args.keepout_mode != "auto":
        trained_mode = task.config.get("cbf_target_keepout_mode", "superellipsoid")
        task.config["cbf_target_keepout_mode"] = args.keepout_mode
        if args.keepout_mode != trained_mode:
            print(f"[Config] Target keep-out set to '{args.keepout_mode}' "
                  f"(trained with '{trained_mode}').")
    apply_actuator_mode(task, args.actuator_uncertainty)
    if args.cbf_sensed_state == "on":
        task.config["cbf_use_sensed_state"] = True
        if not task.config.get("enable_sensor_noise", False):
            print("[Config] WARNING: --cbf-sensed-state has no effect without "
                  "sensor noise.")
        else:
            print("[Config] Filter uses the sensed state.")
    if args.target_lambda is not None:
        lam = float(args.target_lambda)
        prev = float(task.config.get("cbf_target_lambda1", 0.15))
        task.config["cbf_target_lambda1"] = lam
        task.config["cbf_target_lambda2"] = lam
        print(
            f"[Config] Target keep-out poles set to -{lam} (trained with -{prev})."
        )
    if args.max_steps is not None:
        trained_steps = int(task.config["max_steps"])
        task.config["max_steps"] = int(args.max_steps)
        if int(args.max_steps) != trained_steps:
            print(
                f"[Config] Episode horizon overridden: {trained_steps} -> "
                f"{args.max_steps} steps (trained at {trained_steps})."
            )
    if enable_sensor_noise:
        c = task.config
        print(
            "[Config] Sensor noise ON: pos floor="
            f"{c.get('sensor_pos_noise_floor', 0.004)} m "
            f"(+{c.get('sensor_pos_noise_per_m', 0.001)} m/m), "
            f"vel floor={c.get('sensor_vel_noise_floor', 0.001)} m/s "
            f"(+{c.get('sensor_vel_noise_per_m', 0.0003)} m/s/m), "
            f"att floor={c.get('sensor_att_noise_floor', 0.0028)} rad "
            f"(+{c.get('sensor_att_noise_per_m', 0.0005)} rad/m), "
            f"gyro noise={c.get('sensor_angvel_noise_std', 1e-4)} rad/s, "
            f"dropout p={c.get('sensor_dropout_prob', 0.01)}/step"
        )
    if enable_cbf:
        c = task.config
        print(
            "[Config] Filter ON: "
            f"D_s={c.get('cbf_agent_keepout', 0.55)} m, "
            f"lambda=({c.get('cbf_lambda1', 0.05)}, {c.get('cbf_lambda2', 0.075)}) 1/s, "
            f"|v| <= {c.get('cbf_speed_limit_intercept', 0.15)} + "
            f"{c.get('cbf_speed_limit_slope', 0.015)}*r m/s, "
            f"target keep-out={'ON' if c.get('cbf_enable_target_keepout', True) else 'OFF'}, "
            f"w_max={c.get('cbf_max_ang_speed', 0.2)} rad/s"
        )
    else:
        print("[Config] Filter OFF")

    cfg = task.config
    print(
        "[Config] Initial conditions: "
        f"spawn means={cfg.get('init_agent_pos_means')}, "
        f"ranges={cfg.get('init_agent_pos_ranges')}, "
        f"attitude={cfg.get('enable_attitude')} "
        f"(init angle<= {cfg.get('init_max_attitude_angle')} rad), "
        f"max_steps={cfg.get('max_steps')}, tol={cfg.get('goal_tolerance')} m, "
        f"v_max={cfg.get('max_docking_speed')} m/s"
    )

    if args.nominal == "lqr":
        print("[Config] Nominal controller: LQR with quaternion PD. "
              "The policy is not used.")
        policy = LQRNominal(task, args.lqr_kp_att, args.lqr_kd_att,
                            args.lqr_r_max, args.lqr_v_max)

    stats = run_monte_carlo(
        policy=policy,
        task=task,
        episodes=args.episodes,
        batch_size=args.batch_size,
        seed=args.seed,
        deterministic=not args.stochastic,
    )
    results = print_report(stats)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\n[MC] Results written to {out}")

    if args.arrays_out:
        import numpy as _np

        aout = Path(args.arrays_out)
        aout.parent.mkdir(parents=True, exist_ok=True)
        _np.savez(
            aout,
            dock_step=stats["dock_step"].numpy(),
            agent_success=stats["agent_success"].numpy(),
            agent_target_coll=stats["agent_target_coll"].numpy(),
            agent_agent_coll=stats["agent_agent_coll"].numpy(),
            episode_len=stats["episode_len"].numpy(),
            min_separation=stats["min_separation"].numpy(),
            iv_flat=stats["iv_flat"].numpy().astype(_np.float32),
            range_flat=stats["range_flat"].numpy().astype(_np.float32),
            agent_keepout=_np.float32(task.config.get("cbf_agent_keepout", 0.55)),
            dt=_np.float32(task.config.get("dt", 1.0)),
            episodes=_np.int32(args.episodes),
            n_agents=_np.int32(stats["n_agents"]),
        )
        print(f"[MC] Per-agent-episode arrays written to {aout}")


if __name__ == "__main__":
    main()
