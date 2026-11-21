"""Train the docking policy with MAPPO (BenchMARL) and the HOCBF safety filter.

The recipe of the paper is in reproduce/01_train.sh. Run with --help for the
options.
"""
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.experiment.callback import Callback
from torchrl.envs.utils import ExplorationType, set_exploration_type
from benchmarl.models.mlp import MlpConfig
from tensordict.nn import add_custom_mapping

from marl_docking_6dof.envs import Docking6DoFEnv
from marl_docking_6dof.tasks import MarlDocking6DoFTask
from marl_docking_6dof.visualization import save_3d_episode_animation


# Bounded policy noise: sigma = lo + (hi - lo) * sigmoid(raw).
# The lower bound (0.1) prevents a collapse of the std. The upper bound stops
# tanh saturation, which gives an infinite log_prob and NaN weights.
# PolicySharpening anneals the upper bound during training.
SIGMA_MIN, SIGMA_MAX = 0.1, 1.0
SIGMA_MAX_FINAL = 0.15
SCALE_MAPPING = f"bounded_sigma_{SIGMA_MIN}_{SIGMA_MAX}"

# Current upper bound. A one-element list, so a callback can change it.
_SIGMA_CEILING = [SIGMA_MAX]


def _bounded_sigma(x: torch.Tensor) -> torch.Tensor:
    hi = _SIGMA_CEILING[0]
    return SIGMA_MIN + (hi - SIGMA_MIN) * torch.sigmoid(x)


add_custom_mapping(SCALE_MAPPING, _bounded_sigma)


def _unwrap_docking_env(env) -> Docking6DoFEnv | None:
    """Get the raw Docking6DoFEnv from TransformedEnv and wrapper chains.
    Curriculum updates must go to the base env, because setattr on a wrapper
    does not reach it.
    """
    seen: set[int] = set()
    while env is not None and id(env) not in seen:
        if isinstance(env, Docking6DoFEnv):
            return env
        seen.add(id(env))
        env = getattr(env, "base_env", None) or getattr(env, "env", None)
    return None


class PolicySharpening(Callback):
    """Anneal the upper bound of the policy noise, so the policy becomes precise.

    The bound stays at SIGMA_MAX until `start_frac` of the training frames.
    Then it decreases linearly to SIGMA_MAX_FINAL at `end_frac`. The lower
    bound does not change. The schedule comes from `total_frames`, so it
    continues correctly after a restore.
    """

    def __init__(self, *, start_frac: float = 0.5, end_frac: float = 0.9):
        self.start_frac = float(start_frac)
        self.end_frac = float(end_frac)
        self._last_logged = None

    def _ceiling_for(self, frac: float) -> float:
        if frac <= self.start_frac:
            return SIGMA_MAX
        if frac >= self.end_frac:
            return SIGMA_MAX_FINAL
        t = (frac - self.start_frac) / max(self.end_frac - self.start_frac, 1e-9)
        return SIGMA_MAX + t * (SIGMA_MAX_FINAL - SIGMA_MAX)

    def _update(self) -> None:
        total = float(getattr(self.experiment.config, "max_n_frames", 0) or 0)
        if total <= 0:
            return
        frac = float(getattr(self.experiment, "total_frames", 0)) / total
        ceiling = self._ceiling_for(frac)
        _SIGMA_CEILING[0] = ceiling
        bucket = round(ceiling, 2)
        if self._last_logged != bucket:
            self._last_logged = bucket
            print(
                f"[Sharpening] progress {frac:.0%} -> policy-noise ceiling "
                f"sigma_max = {ceiling:.3f} (floor stays {SIGMA_MIN})"
            )

    def on_setup(self):
        self._update()

    def on_batch_collected(self, batch):
        self._update()

    # The upper bound is a module global, and a new process resets it. Save it
    # with the checkpoint, so a reloaded policy samples with the annealed value.
    def on_state_dict(self, state_dict: Dict[str, Any]):
        state_dict["policy_sharpening"] = {"sigma_ceiling": float(_SIGMA_CEILING[0])}

    def on_load_state_dict(self, state_dict: Dict[str, Any]):
        saved = state_dict.get("policy_sharpening", None)
        if saved is not None and "sigma_ceiling" in saved:
            _SIGMA_CEILING[0] = float(saved["sigma_ceiling"])
            print(
                f"[Sharpening] Restored policy-noise ceiling "
                f"sigma_max = {_SIGMA_CEILING[0]:.3f}"
            )


class GoalToleranceCurriculum(Callback):
    """Capture-gate curriculum: tighten goal_tolerance when the agents dock.

    Training starts at `start_tolerance`, so agents can find the goal bonus.
    When the dock rate is above `success_threshold`, the tolerance decreases
    by `step`, down to `end_tolerance`. The evaluation env is kept in sync.
    """

    def __init__(
        self,
        *,
        start_tolerance: float = 0.15,
        end_tolerance: float = 0.05,
        step: float = 0.02,
        success_threshold: float = 0.6,
        ema_alpha: float = 0.8,
        # Minimum number of batches between two steps. Without it the gate can step
        # too fast for the policy to follow.
        min_batches_between_steps: int = 8,
        # Number of consecutive deterministic evaluations above success_threshold
        # before the gate tightens.
        eval_confirmations: int = 2,
        # capture_shaping_radius = max(factor * tolerance, floor), so the shaping
        # reward stays near the current gate.
        shape_radius_factor: float = 4.0,
        shape_radius_floor: float = 0.2,
        # Attitude gate curriculum (optional). None keeps max_attitude_error fixed.
        # The attitude gate steps with the position gate.
        start_attitude: float | None = None,
        end_attitude: float | None = None,
        attitude_step: float = 0.014,
        # Evaluation rate used to step the gate.
        # "per_agent": per-agent dock rate (default).
        # "complete": all-agents-docked rate. Use it when the end gate is tight,
        # because the complete rate is about p^N of the per-agent rate.
        success_metric: str = "per_agent",
    ):
        super().__init__()
        self.start_tolerance = float(start_tolerance)
        self.end_tolerance = float(end_tolerance)
        self.step = float(step)
        self.success_threshold = float(success_threshold)
        self.ema_alpha = float(ema_alpha)
        self.min_batches_between_steps = max(int(min_batches_between_steps), 1)
        self.eval_confirmations = max(int(eval_confirmations), 1)
        self._eval_rates: list[float] = []
        self._eval_seen = -1
        self.shape_radius_factor = float(shape_radius_factor)
        self.shape_radius_floor = float(shape_radius_floor)
        self.tolerance = self.start_tolerance
        self.dock_rate_ema = 0.0
        self.batches_since_step = 0
        self.start_attitude = None if start_attitude is None else float(start_attitude)
        self.end_attitude = None if end_attitude is None else float(end_attitude)
        self.attitude_step = float(attitude_step)
        self.attitude = self.start_attitude
        if success_metric not in ("per_agent", "complete"):
            raise ValueError(
                f"success_metric must be 'per_agent' or 'complete', got "
                f"{success_metric!r}"
            )
        self.success_metric = str(success_metric)

    # Default values for fields added later, so an older pickle loads.
    _PICKLE_DEFAULTS = {
        "min_batches_between_steps": 8,
        "shape_radius_factor": 4.0,
        "shape_radius_floor": 0.2,
        "batches_since_step": 0,
        "eval_confirmations": 2,
        "_eval_rates": [],
        "_eval_seen": -1,
        "start_attitude": None,
        "end_attitude": None,
        "attitude_step": 0.014,
        "attitude": None,
        "success_metric": "per_agent",
    }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # Add the missing fields of an older pickle, then apply the saved state.
        merged = {**self._PICKLE_DEFAULTS, **state}
        self.__dict__.update(merged)

    def _apply(self) -> None:
        radius = max(
            self.shape_radius_factor * self.tolerance, self.shape_radius_floor
        )
        for env in (
            getattr(getattr(self.experiment, "collector", None), "env", None),
            getattr(self.experiment, "test_env", None),
        ):
            base = _unwrap_docking_env(env)
            if base is not None:
                base.goal_tolerance = self.tolerance
                if self.attitude is not None:
                    base.max_attitude_error = self.attitude
                if abs(base.capture_shaping_radius - radius) > 1e-9:
                    base.capture_shaping_radius = radius
                    # Update the stored potential, so the next shaping term uses one radius.
                    base.prev_capture_potential = base._capture_potential()

    def on_setup(self):
        self._apply()

    def on_batch_collected(self, batch):
        try:
            group = next(iter(self.experiment.group_map.keys()))
            done = batch.get(("next", "done")).reshape(-1).bool()
            on_goal = batch.get(("next", group, "info", "on_goal"))
            # [*, N, 1]: per-agent dock fraction at terminal steps.
            dock = on_goal[..., 0].float().mean(dim=-1).reshape(-1)
            if not done.any():
                return
            batch_rate = float(dock[done].mean().item())
        except Exception as exc:  # do not stop training if the curriculum fails
            print(f"[Curriculum] Failed to read batch dock rate: {exc}")
            return

        self.dock_rate_ema = (
            self.ema_alpha * self.dock_rate_ema + (1.0 - self.ema_alpha) * batch_rate
        )
        self.batches_since_step += 1

        # Use the deterministic evaluation rate when it is available. The rate on
        # the collection batch comes from the exploratory policy and is lower.
        # The gate steps after `eval_confirmations` consecutive evaluations above
        # the threshold.
        eval_rate = getattr(
            self.experiment,
            "_last_eval_per_agent_rate"
            if self.success_metric == "per_agent"
            else "_last_success_rate",
            None,
        )
        eval_idx = getattr(self.experiment, "_eval_report_index", -1)
        using_eval = eval_rate is not None
        if using_eval and eval_idx != self._eval_seen:
            self._eval_seen = eval_idx
            self._eval_rates.append(float(eval_rate))
            del self._eval_rates[: -self.eval_confirmations]

        if using_eval:
            ready = len(self._eval_rates) >= self.eval_confirmations and all(
                r >= self.success_threshold for r in self._eval_rates
            )
        else:
            ready = self.dock_rate_ema >= self.success_threshold

        attitude_pending = (
            self.attitude is not None
            and self.end_attitude is not None
            and self.attitude > self.end_attitude + 1e-12
        )
        if (
            (self.tolerance > self.end_tolerance or attitude_pending)
            and ready
            and self.batches_since_step >= self.min_batches_between_steps
        ):
            self.tolerance = max(self.end_tolerance, self.tolerance - self.step)
            if self.attitude is not None and self.end_attitude is not None:
                self.attitude = max(
                    self.end_attitude, self.attitude - self.attitude_step
                )
            self.dock_rate_ema = 0.0
            self._eval_rates.clear()
            self.batches_since_step = 0
            self._apply()
            print(
                f"[Curriculum] {self.success_metric} dock rate hit "
                f"{self.success_threshold:.0%} -> "
                f"goal_tolerance tightened to {self.tolerance:.3f} m "
                f"(capture_shaping_radius -> "
                f"{max(self.shape_radius_factor * self.tolerance, self.shape_radius_floor):.2f} m)"
                + (
                    f"; attitude gate -> {self.attitude:.4f} rad "
                    f"({self.attitude * 180.0 / math.pi:.2f} deg)"
                    if self.attitude is not None
                    else ""
                )
            )
        else:
            # Apply the current gate to all envs (also after a restore).
            self._apply()

    # Save and restore with the checkpoint.
    def on_state_dict(self, state_dict: Dict[str, Any]):
        state_dict["goal_tolerance_curriculum"] = {
            "tolerance": self.tolerance,
            "dock_rate_ema": self.dock_rate_ema,
            "batches_since_step": self.batches_since_step,
            "attitude": self.attitude,
        }

    def on_load_state_dict(self, state_dict: Dict[str, Any]):
        saved = state_dict.get("goal_tolerance_curriculum", None)
        if saved:
            self.tolerance = float(saved["tolerance"])
            self.dock_rate_ema = float(saved["dock_rate_ema"])
            self.batches_since_step = int(saved.get("batches_since_step", 0))
            if saved.get("attitude", None) is not None:
                self.attitude = float(saved["attitude"])
            self._apply()
            print(
                f"[Curriculum] Restored goal_tolerance={self.tolerance:.3f} m"
                + (
                    f", attitude gate={self.attitude:.4f} rad"
                    if self.attitude is not None
                    else ""
                )
            )


class VideoEvalExperiment(Experiment):
    """Experiment that saves one video for each evaluation."""

    def __init__(
        self,
        *args,
        eval_video_dir: str | None = None,
        eval_video_fps: int = 100,
        eval_video_ext: str = ".mp4",
        save_eval_videos: bool = True,
        **kwargs,
    ):
        self.eval_video_dir = (
            None if eval_video_dir is None or eval_video_dir == "" else Path(eval_video_dir)
        )
        self.eval_video_fps = int(eval_video_fps)
        self.eval_video_ext = eval_video_ext if eval_video_ext.startswith(".") else f".{eval_video_ext}"
        self.save_eval_videos = bool(save_eval_videos)
        self._eval_video_count = 0
        super().__init__(*args, **kwargs)

    def _save_experiment(self) -> None:
        """Save a checkpoint and add the last complete success rate to its name:
        checkpoint_{frames}_succ<pct>.pt. The folder layout does not change.
        """
        super()._save_experiment()
        rate = getattr(self, "_last_success_rate", None)
        if rate is None or not self._checkpointed_files:
            return
        try:
            saved = self._checkpointed_files[-1]
            new_path = saved.with_name(f"{saved.stem}_succ{int(round(rate * 100)):03d}{saved.suffix}")
            saved.rename(new_path)
            self._checkpointed_files[-1] = new_path
        except Exception as exc:
            print(f"[Checkpoint] Could not append success rate to name: {exc}")

    def _evaluation_loop(self) -> None:
        super()._evaluation_loop()
        # One evaluation rollout for the success report and the video.
        rollout = self._collect_eval_rollout()
        if rollout is None:
            return
        self._report_eval_success(rollout)
        self._report_reward_decomposition(rollout)
        if self.save_eval_videos:
            self._save_eval_gif(rollout)

    def _eval_group(self) -> str:
        test_env = self.test_env
        if hasattr(test_env, "group_map") and test_env.group_map:
            return next(iter(test_env.group_map.keys()))
        return "sixdof"

    @torch.no_grad()
    def _collect_eval_rollout(self):
        """One batched rollout over `evaluation_episodes` parallel envs.

        Envs auto-reset, as in the BenchMARL evaluation. Actions are deterministic
        if configured. Must run under no_grad, or the rollout keeps the full
        autograd graph (tens of GB).
        """
        try:
            policy = self.policy
            if policy is None and hasattr(self, "policies") and self.policies:
                policy = next(iter(self.policies.values()))
            if policy is None:
                print("[Eval] No policy found; skipping success report and video.")
                return None
            deterministic = getattr(self.config, "evaluation_deterministic_actions", True)
            with set_exploration_type(
                ExplorationType.DETERMINISTIC if deterministic else ExplorationType.RANDOM
            ):
                return self.test_env.rollout(
                    max_steps=self.max_steps,
                    policy=policy,
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                )
        except Exception as exc:
            print(f"[Eval] Failed to collect evaluation rollout: {exc}")
            return None

    def _report_eval_success(self, rollout) -> None:
        """Print the docking success rate over all completed episodes.
        An episode that does not end in the window is not counted.
        """
        try:
            group = self._eval_group()
            done = rollout.get(("next", "done"))
            info = rollout.get(("next", group, "info"))
            on_goal = info.get("on_goal")  # [B, T, N, 1] latched per-agent capture
            coll = info.get("collision_terminal")  # [B, T, N, 1]
            numerr = info.get("numerical_error")  # [B, T, N, 1]
            agent_coll = info.get("agent_collision_terminal", None)  # [B, T, N, 1]

            b, t = done.shape[0], done.shape[1]
            done = done.reshape(b, t).bool()
            all_docked = on_goal[..., 0].bool().all(dim=-1)  # [B, T] all agents docked
            dock_frac = on_goal[..., 0].float().mean(dim=-1)  # [B, T] fraction docked
            target_coll_any = coll[..., 0].bool().any(dim=-1)  # [B, T] chaser<->target
            if agent_coll is not None:
                agent_coll_any = agent_coll[..., 0].bool().any(dim=-1)  # [B, T] inter-agent
            else:
                agent_coll_any = torch.zeros_like(target_coll_any)
            coll_any = target_coll_any | agent_coll_any  # [B, T] any collision
            num_any = numerr[..., 0].bool().any(dim=-1)  # [B, T]

            n_episodes = int(done.sum().item())
            # An env whose last step is not terminal has an unfinished episode.
            n_unfinished = int((~done[:, -1]).sum().item())
            if n_episodes == 0:
                print(
                    f"[EvalSuccess] frames={int(self.total_frames)} | "
                    f"no episodes completed within the {t}-step window "
                    f"({n_unfinished} unfinished, excluded)."
                )
                return

            success = done & all_docked
            n_success = int(success.sum().item())
            failed_coll = done & coll_any & ~all_docked
            n_collision = int(failed_coll.sum().item())
            # Target contact has priority, so the two counts add up to n_collision.
            n_collision_target = int((failed_coll & target_coll_any).sum().item())
            n_collision_agent = n_collision - n_collision_target
            n_numerical = int((done & num_any & ~all_docked & ~coll_any).sum().item())
            n_timeout = n_episodes - n_success - n_collision - n_numerical

            success_rate = n_success / n_episodes
            per_agent_rate = float(dock_frac[done].mean().item())
            # Last complete success rate, added to the next checkpoint name.
            self._last_success_rate = success_rate
            # Deterministic per-agent dock rate, used by GoalToleranceCurriculum.
            self._last_eval_per_agent_rate = per_agent_rate
            self._eval_report_index = getattr(self, "_eval_report_index", -1) + 1

            base_env = _unwrap_docking_env(self.test_env)
            tol_str = (
                f" | goal_tolerance={base_env.goal_tolerance:.3f} m"
                if base_env is not None
                else ""
            )
            print(
                f"[EvalSuccess] frames={int(self.total_frames)} | "
                f"episodes={n_episodes} (excluded {n_unfinished} unfinished) | "
                f"all-agents docked: {n_success}/{n_episodes} = {success_rate:.1%} | "
                f"per-agent dock: {per_agent_rate:.1%}{tol_str}"
            )
            print(
                f"              failures -> timeout: {n_timeout}, "
                f"collision: {n_collision} (target: {n_collision_target}, "
                f"inter-agent: {n_collision_agent}), numerical: {n_numerical}"
            )
        except Exception as exc:
            print(f"[EvalSuccess] Failed to compute success report: {exc}")

    def _report_reward_decomposition(self, rollout) -> None:
        """Split the mean return per episode into its reward terms.
        The terms are the same as in _compute_reward of the env.
        """
        try:
            group = self._eval_group()
            done = rollout.get(("next", "done"))
            info = rollout.get(("next", group, "info"))

            b, t = done.shape[0], done.shape[1]
            n_episodes = int(done.reshape(b, t).bool().sum().item())
            if n_episodes == 0:
                return

            # (info key -> label), in print order.
            components = [
                ("reward_progress", "position (progress)"),
                ("reward_velocity", "velocity"),
                ("reward_time", "time penalty"),
                ("reward_capture", "capture shaping"),
                ("reward_bonus", "goal bonus"),
                ("reward_collision", "target collision"),
                ("reward_agent_collision", "inter-agent collision"),
                ("reward_attitude", "attitude"),
                ("reward_cbf_intervention", "CBF intervention"),
                ("reward_fuel", "fuel (delta-v)"),
            ]

            lines = []
            total = 0.0
            for key, label in components:
                comp = info.get(key, None)
                if comp is None:
                    continue
                # [B, T, N, 1] -> mean over agents, sum over (B, T), per episode.
                contrib = float(comp[..., 0].mean(dim=-1).sum().item()) / n_episodes
                total += contrib
                lines.append(f"{label}: {contrib:+.3f}")

            print(
                f"[EvalReward] frames={int(self.total_frames)} | mean return "
                f"~= {total:+.3f} over {n_episodes} episodes (per agent)"
            )
            print("              sources -> " + ", ".join(lines))
        except Exception as exc:
            print(f"[EvalReward] Failed to compute reward decomposition: {exc}")

    def _save_eval_gif(self, rollout) -> None:
        try:
            group = self._eval_group()
            pos = self._get_agent_positions(rollout, group)
            goals = self._get_goal_positions(rollout, group)
            if pos is None or goals is None:
                print("[EvalVideo] Position/goal info keys not found in rollout; skipping.")
                return
            target_centers = self._get_target_centers(rollout, group)
            target_sizes = self._get_target_sizes(rollout, group)
            agent_sizes = self._get_agent_sizes(rollout, group)
            attitude_errors_deg = self._get_attitude_errors_deg(rollout, group)
            if not torch.isfinite(pos).all() or not torch.isfinite(goals).all():
                print("[EvalVideo] Non-finite rollout detected; skipping this evaluation video.")
                return

            video_dir = self._resolve_eval_video_dir()
            video_dir.mkdir(parents=True, exist_ok=True)
            path = video_dir / (
                f"eval_{self._eval_video_count:05d}_frames_{int(self.total_frames)}{self.eval_video_ext}"
            )
            saved = save_3d_episode_animation(
                positions=pos.numpy(),
                goals=goals.numpy(),
                target_centers=(
                    np.nan_to_num(target_centers, nan=0.0, posinf=1e6, neginf=-1e6)
                    if target_centers is not None
                    else None
                ),
                target_sizes=(
                    np.nan_to_num(target_sizes, nan=1.0, posinf=1e3, neginf=1e-3)
                    if target_sizes is not None
                    else None
                ),
                agent_sizes=(
                    np.nan_to_num(agent_sizes, nan=0.1, posinf=1e3, neginf=1e-3)
                    if agent_sizes is not None
                    else None
                ),
                attitude_errors_deg=(
                    np.nan_to_num(attitude_errors_deg, nan=0.0, posinf=0.0, neginf=0.0)
                    if attitude_errors_deg is not None
                    else None
                ),
                output_path=path,
                fps=self.eval_video_fps,
                title=f"Eval {self._eval_video_count} @ frames={int(self.total_frames)}",
            )
            self._eval_video_count += 1
            print(f"[EvalVideo] Saved: {saved}")
        except Exception as exc:
            print(f"[EvalVideo] Failed to save evaluation video: {exc}")

    def _resolve_eval_video_dir(self) -> Path:
        if self.eval_video_dir is not None:
            return self.eval_video_dir
        # Default: inside the current run folder.
        run_folder = Path(self.folder_name) if hasattr(self, "folder_name") else Path.cwd()
        return run_folder / "eval_videos"

    @staticmethod
    def _get_agent_positions(rollout, group: str):
        for key in (
            (group, "info", "agent_position"),
            ("next", group, "info", "agent_position"),
        ):
            try:
                # [B, T, N, 3] -> env 0 => [T, N, 3]
                return rollout.get(key)[0].detach().cpu()
            except Exception:
                continue
        return None

    @staticmethod
    def _get_goal_positions(rollout, group: str):
        for key in (
            (group, "info", "goal_position"),
            ("next", group, "info", "goal_position"),
        ):
            try:
                # [B, T, N, 3] -> env 0 => [T, N, 3]
                return rollout.get(key)[0].detach().cpu()
            except Exception:
                continue
        return None

    @staticmethod
    def _get_target_centers(rollout, group: str):
        for key in (
            (group, "info", "target_center"),
            ("next", group, "info", "target_center"),
        ):
            try:
                data = rollout.get(key)
                # [B, T, N, 3] -> env 0, agent 0 => [T, 3]
                return data[0, :, 0, :].detach().cpu().numpy()
            except Exception:
                continue
        return None

    @staticmethod
    def _get_target_sizes(rollout, group: str):
        for key in (
            (group, "info", "target_size"),
            ("next", group, "info", "target_size"),
        ):
            try:
                data = rollout.get(key)
                # Use the first available size vector.
                return data[0, 0, 0, :].detach().cpu().numpy()
            except Exception:
                continue
        return None

    @staticmethod
    def _get_agent_sizes(rollout, group: str):
        for key in (
            (group, "info", "agent_size"),
            ("next", group, "info", "agent_size"),
        ):
            try:
                data = rollout.get(key)
                # [B, T, N, 3] -> first frame of env 0: [N, 3]
                return data[0, 0].detach().cpu().numpy()
            except Exception:
                continue
        return None

    @staticmethod
    def _get_attitude_errors_deg(rollout, group: str):
        for key in (
            (group, "info", "attitude_error_angle"),
            ("next", group, "info", "attitude_error_angle"),
        ):
            try:
                data = rollout.get(key)
                # [B, T, N, 1] -> env 0: [T, N], radians to degrees.
                angle = data[0, ..., 0].detach().cpu().numpy()
                return np.degrees(angle)
            except Exception:
                continue
        return None


def build_experiment(
    *,
    save_eval_videos: bool = True,
    eval_video_dir: str | None = None,
    eval_video_fps: int = 20,
    eval_video_ext: str = ".mp4",
    restore_file: str | None = None,
    save_folder: str | None = None,
    max_frames: int | None = None,
    disable_cbf: bool = False,
    enable_sensor_noise: bool = False,
    enable_fuel_cost: bool = False,
    fuel_lin_scale: float = 1.0,
    fuel_ang_scale: float = 8.0,
    seed: int = 0,
    safety_obs_features: str | None = None,
    keepout_mode: str | None = None,
    target_lambda: float | None = None,
    attitude_curriculum: bool = False,
    end_attitude: float = 0.0349,
    end_tolerance: float = 0.05,
    curriculum_metric: str = "per_agent",
    checkpoint_every: int = 10,
    restore_map_location: str | None = None,
) -> Experiment:
    task = MarlDocking6DoFTask.DOCKING_6DOF.get_from_yaml()
    # Number of chasers. The spawn box and the port grid are made for n_agents.
    n_agents = 9
    task.config.update(
        {
            "n_agents": n_agents,
            # None: obs_dim and action_dim are set from the feature switches.
            "obs_dim": None,
            "action_dim": None,
            # Feature switches.
            # Full 6-DOF: quaternion attitude, torque actions, attitude error in the
            # observation and reward, attitude gate on capture.
            "enable_attitude": True,
            # Inter-agent collisions: penalty and terminal contact.
            "enable_agent_collisions": True,
            # Each agent observes the relative position and velocity of its two nearest
            # teammates (nearest first).
            "enable_neighbor_obs": True,
            "neighbor_obs_count": 2,
            # Scales for the neighbour features. The base scale (10 m) makes separations
            # of 0.4-1 m too small for the policy.
            "obs_neighbor_pos_scale": 2.0,
            "obs_neighbor_vel_scale": 0.5,
            # CBF safety filter (safety/cbf.py). The policy action is the nominal. The
            # filter projects it onto the safe set inside the actuator box. Collision
            # terminations stay on as a check.
            "enable_cbf": True,
            # Keep-apart distance D_s: contact 2r (about 0.442 m) plus margin.
            # Keep it well below the port spacing (1.1 m), or the pair rows push docked
            # agents off their ports.
            "cbf_agent_keepout": 0.55,
            # HOCBF gains. Lower lambda1 brakes earlier. lambda1 = 0.1 keeps the
            # required braking inside the shared authority for gaps up to about 1 m.
            "cbf_lambda1": 0.1,
            "cbf_lambda2": 0.15,
            # Keep this at inf. With a finite radius, a pair can enter range too fast to
            # stop, and the QP becomes infeasible.
            "cbf_activation_radius": float("inf"),
            # Speed limit ||v|| <= nu0 + nu1*range (range from the target centre).
            # This row is never relaxed, so it is the main target protection.
            # nu0 = 0.13 is just above the 0.1 m/s capture speed.
            "cbf_speed_limit_intercept": 0.13,
            "cbf_speed_limit_slope": 0.012,
            "cbf_speed_limit_gain": 0.5,
            # Target keep-out superellipsoid (anisotropic, so the docked pose stays
            # outside). Clearance must be > 0, or the surface is inside the target box
            # at the port corners.
            "cbf_enable_target_keepout": True,
            "cbf_target_clearance": 0.08,
            # Keep-out gains. Lower values brake earlier.
            "cbf_target_lambda1": 0.15,
            "cbf_target_lambda2": 0.15,
            # Superellipsoid exponent. At n = 4 the surface is inside the box at the
            # corner ports. n = 8 covers the corners.
            "cbf_target_keepout_exponent": 8,
            # Rate limit equal to the capture gate max_docking_ang_speed (0.05 rad/s),
            # so the rate condition of capture is always satisfied.
            "cbf_max_ang_speed": 0.05,
            "cbf_ang_speed_gain": 0.1,
            # Solve the QP once per control step and hold the result for the 1 s step.
            # The D_s buffer (D_s - 2r = 0.108 m) covers the drift in the hold. If you
            # increase lambda or the closing speed, increase D_s.
            "cbf_filter_substep_interval": 10,
            # Intervention penalty c_int.
            "cbf_intervention_penalty_scale": 0.05,
            # Safety features in the observation: gap to keep-out and last intervention
            # (see --safety-obs-features). enable_terminal_obs is an option that is off.
            "enable_terminal_obs": False,
            # Near-goal respawn (training only; mc_eval sets it to 0). 25% of agents
            # start within 0.25 m of their port, about half of them behind it. This
            # teaches recovery from behind the port.
            "init_near_goal_fraction": 0.25,
            "init_near_goal_radius": 0.25,
            "enable_safety_obs": True,
            # Episode length (steps of 1 s).
            "max_steps": 300,
            "dt": 1.0,
            "dynamics_substeps": 10,
            # 6U CubeSat: about 50 mN thrust on about 10 kg, so a_max = 0.005 m/s^2.
            "linear_accel_scale": 0.005,
            # Capture radius (m). Training starts at 0.15 m. GoalToleranceCurriculum
            # makes it smaller.
            "goal_tolerance": 0.15,
            # Maximum closing speed at capture (m/s).
            "max_docking_speed": 0.1,
            # Attitude parameters.
            # Small reaction wheels on a 6U bus: about 0.02 rad/s^2.
            "angular_accel_scale": 0.02,
            # Capture also needs attitude error < max_attitude_error and rate < 0.05
            # rad/s. Set attitude_capture_gate False to train translation first.
            "attitude_capture_gate": True,
            "max_attitude_error": 0.175,
            "max_docking_ang_speed": 0.05,
            # Diagonal of Q in -alpha^T Q alpha, alpha = [2*q_err_vec, w_err].
            "reward_att_scale": 0.05,
            "reward_angvel_scale": 0.02,
            # Spawn attitude: up to 45 deg error, at rest.
            "init_max_attitude_angle": 0.785,
            "init_max_ang_vel": 0.0,
            # Inter-agent collision parameters.
            # Contact distance 2r is about 0.442 m. The warning band (0.10 m) ends at
            # 0.542 m, inside D_s = 0.55 m, so an agent held at the filter boundary has
            # no penalty. The terminal contact penalty must not occur.
            "agent_collision_penalty_scale": 1.0,
            "agent_collision_warning_distance": 0.10,
            "agent_collision_terminal_on_contact": True,
            "agent_collision_terminal_penalty": 20.0,
            "agent_collision_smoothing_distance": 0.1,  # unused with terminal_on_contact
            "agent_collision_margin": 0.0,
            # Contact with the target ends the episode. Its penalty is larger than one
            # step of the contact ramp.
            "collision_terminal_penalty": 20.0,
            # Dense potential-based progress reward, small time penalty and sparse bonus.
            "reward_progress_scale": 1.5,
            # Time penalty per step. It makes hovering near the port cost more, and
            # stays below the crash penalties.
            "reward_time_penalty": 0.01,
            # Small velocity penalty. The capture gate already requires a slow arrival.
            "reward_vel_scale": 4.0,
            "reward_goal_bonus": 40.0,
            # Depth weight of the capture bonus: an edge capture gives 40 and the centre
            # gives 60. This moves the aim point inside the capture ball. The premium is
            # added, so it never lowers the reward.
            "reward_goal_bonus_depth_weight": 0.5,
            # Potential-based arrive-and-stop shaping: reward the change in
            # Phi = near*slow. It gives a gradient to brake into the capture region,
            # and hovering cannot collect it. radius is where the shaping starts (m).
            "reward_capture_scale": 10.0,
            "capture_shaping_radius": 0.6,
            # Contact with the target ends the episode, so the policy must brake.
            "collision_penalty_scale": 2.0,
            "collision_smoothing_distance": 0.1,
            "obs_pos_scale": 10.0,
            "obs_vel_scale": 1.0,
            # Spawn box (mean position in m, relative to the target centre). All agents
            # share one box. The lateral extent grows with sqrt(n_agents / 4), so the
            # spawn density stays about constant.
            "init_agent_pos_means": [[10.0, 0.0, 0.0] for _ in range(n_agents)],
            # Range around each mean (scalar or [rx, ry, rz] per agent).
            "init_agent_pos_ranges": [
                [1.5, 2.0 * max(1.0, (n_agents / 4.0) ** 0.5), 2.0 * max(1.0, (n_agents / 4.0) ** 0.5)]
                for _ in range(n_agents)
            ],
            # Minimum spawn separation, outside the collision warning band.
            "min_spawn_separation": 0.9,
            # Target spacecraft: a box of about 2 m.
            "target_size_x": 2.0,
            "target_size_y": 2.0,
            "target_size_z": 2.2,
            # Chaser: 6U CubeSat (0.1 x 0.2263 x 0.366 m). It docks on its broad face,
            # so the depth along the approach axis (x) is the short side.
            "agent_size_x": 0.10,
            "agent_size_y": 0.2263,
            "agent_size_z": 0.366,
            "target_center_jitter": 0.0,
            "docking_offset_x": 0.2,
            # Ports at (+/-0.55, +/-0.55) m on the +X face, 1.1 m apart. Docked agents
            # are outside D_s, so the pair rows are slack at the ports.
            "port_spread": 0.55,
        }
    )

    # PPO stability. A small policy std makes the PPO ratio very sensitive
    # (about 1/sigma^2), which causes KL spikes and a collapse of success.
    # The guards are: the sigma floor of 0.1 (scale mapping above) and 4
    # minibatch iterations instead of 8.
    algo = MappoConfig(
        share_param_critic=True,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        critic_coef=1.0,
        loss_critic_type="l2",
        lmbda=0.95,
        scale_mapping=SCALE_MAPPING,
        use_tanh_normal=True,
        minibatch_advantage=False,
    )

    # Ablation switches. The policy settings do not change.
    if disable_cbf:
        # Without the filter, the gap and psi1 features stay valid and the
        # intervention features are zero. obs_dim does not change, so the policy
        # can be evaluated with the filter on.
        task.config["enable_cbf"] = False
        task.config["cbf_intervention_penalty_scale"] = 0.0
        print("[Ablation] Safety filter off.")
    if keepout_mode:
        task.config["cbf_target_keepout_mode"] = keepout_mode
    if target_lambda is not None:
        # Target keep-out poles. For the half-space, lambda sets where braking starts
        # (h = 2s/lambda). Safety needs lambda <= 4 a_bar / s. Below about 0.03 the
        # agent does not reach the port in an episode.
        task.config["cbf_target_lambda1"] = float(target_lambda)
        task.config["cbf_target_lambda2"] = float(target_lambda)
        print(f"[CBF] Target keep-out poles set to -{target_lambda}.")
        print(f"[Keepout] Target keep-out geometry: {keepout_mode}.")
    if safety_obs_features:
        feats = tuple(int(x) for x in safety_obs_features.replace(",", " ").split())
        task.config["safety_obs_features"] = feats
        labels = {0: "gap-to-keepout", 1: "tanh(psi1)", 2: "intervention", 3: "active flag"}
        print("[Obs] Safety block reduced to "
              + ", ".join(f"{i}:{labels[i]}" for i in feats)
              + f" (obs_dim 28 -> {24 + len(feats)}).")
    if enable_fuel_cost:
        # Fuel cost on the applied (post-filter) control. See reward_fuel_scale in
        # Docking6DoFEnv.
        task.config["reward_fuel_scale"] = float(fuel_lin_scale)
        task.config["reward_fuel_ang_scale"] = float(fuel_ang_scale)
        print(
            "[Fuel] Delta-v cost on: linear "
            f"{task.config['reward_fuel_scale']}/(m/s), angular "
            f"{task.config['reward_fuel_ang_scale']}/(rad/s)."
        )
    if enable_sensor_noise:
        # Sensor noise on the observation only. See enable_sensor_noise in
        # Docking6DoFEnv. Reward, capture, physics and the filter use the true state.
        task.config["enable_sensor_noise"] = True
        print("[Sensor] Sensor noise on.")

    model = MlpConfig.get_from_yaml()
    critic_model = MlpConfig.get_from_yaml()
    model.num_cells = [256, 256]
    critic_model.num_cells = [256, 256]

    max_steps = int(task.config["max_steps"])
    n_envs = 32

    exp_config = ExperimentConfig.get_from_yaml()
    exp_config.prefer_continuous_actions = True
    exp_config.max_n_frames = int(max_frames) if max_frames else 20_000_000
    exp_config.on_policy_n_envs_per_worker = n_envs
    # About two full episodes per env per batch.
    exp_config.on_policy_collected_frames_per_batch = n_envs * max_steps * 2
    # 4 iterations (see the PPO stability note above).
    exp_config.on_policy_n_minibatch_iters = 4
    exp_config.on_policy_minibatch_size = 4096
    exp_config.gamma = 0.99
    exp_config.lr = 3e-4
    exp_config.evaluation = True
    exp_config.evaluation_episodes = 200
    exp_config.evaluation_interval = n_envs * max_steps * 2 * 10
    # Save often, so each curriculum gate has at least one checkpoint (the dwell
    # is 8 batches).
    exp_config.checkpoint_interval = n_envs * max_steps * 2 * int(checkpoint_every)
    # Keep every checkpoint (the default keeps the last 3).
    exp_config.keep_checkpoints_num = None
    # The observations are relative, so a shared policy works for all agents.
    exp_config.share_policy_params = True
    exp_config.loggers = ["csv"]
    exp_config.render = False

    # Resume: BenchMARL loads the checkpoint in Experiment.__init__, so set
    # restore_file before the experiment is made. Without --save-folder,
    # BenchMARL writes into the folder of the restored run.
    if save_folder:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        exp_config.save_folder = save_folder
    if restore_file:
        exp_config.restore_file = restore_file
        if restore_map_location is not None:
            exp_config.restore_map_location = restore_map_location

    # Curriculum pacing: small steps, a high threshold and a dwell, so each gate
    # holds until the policy docks at it.
    sharpening = PolicySharpening(start_frac=0.5, end_frac=0.9)

    curriculum = GoalToleranceCurriculum(
        start_tolerance=float(task.config["goal_tolerance"]),  # 0.15 m
        end_tolerance=float(end_tolerance),
        step=0.01,
        # Step threshold for the curriculum.
        success_threshold=0.95,
        ema_alpha=0.8,
        min_batches_between_steps=8,
        shape_radius_factor=4.0,  # 0.6 m at the start, 0.2 m at 0.05 m
        shape_radius_floor=0.2,
        # Attitude gate: 0.175 -> 0.0349 rad (10 -> 2 deg).
        success_metric=str(curriculum_metric),
        start_attitude=(
            float(task.config["max_attitude_error"]) if attitude_curriculum else None
        ),
        end_attitude=float(end_attitude) if attitude_curriculum else None,
        # Use the same number of steps as the position gate, so both end together.
        attitude_step=(
            (float(task.config["max_attitude_error"]) - float(end_attitude))
            / max(
                1.0,
                round((float(task.config["goal_tolerance"]) - float(end_tolerance))
                      / 0.01),
            )
        ),
    )

    return VideoEvalExperiment(
        task=task,
        algorithm_config=algo,
        model_config=model,
        critic_model_config=critic_model,
        seed=int(seed),
        config=exp_config,
        callbacks=[curriculum, sharpening],
        save_eval_videos=save_eval_videos,
        eval_video_dir=eval_video_dir,
        eval_video_fps=eval_video_fps,
        eval_video_ext=eval_video_ext,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore-file", default="", help="Checkpoint .pt to resume from.")
    parser.add_argument("--restore-map-location", default=None, help="torch.load map_location, for "
                                                                     "example 'cpu'.")
    parser.add_argument(
        "--no-cbf",
        action="store_true",
        help="Ablation: train without the safety filter. obs_dim does not "
             "change, so the policy can be evaluated with the filter on.",
    )
    parser.add_argument(
        "--sensor-noise",
        action="store_true",
        help="Sensor noise on the observation: range-dependent noise, bias "
             "per episode and dropout. Reward, physics and the filter do not "
             "change.",
    )
    parser.add_argument(
        "--keepout-mode",
        choices=("superellipsoid", "halfspace"),
        default=None,
        help="Target keep-out geometry: 'superellipsoid' or 'halfspace'.",
    )
    parser.add_argument(
        "--safety-obs-features",
        default=None,
        help="Safety features to keep, by index (0 gap, 1 tanh(psi1), 2 "
             "intervention, 3 active flag). Changes obs_dim. Default: all "
             "four.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Training seed.",
    )
    parser.add_argument(
        "--fuel-lin-scale",
        type=float,
        default=1.0,
        help="Weight on translational delta-v (per m/s). Used only with "
             "--fuel.",
    )
    parser.add_argument(
        "--fuel-ang-scale",
        type=float,
        default=8.0,
        help="Weight on rotational delta-v (per rad/s). Used only with "
             "--fuel.",
    )
    parser.add_argument(
        "--fuel",
        action="store_true",
        help="Add a delta-v cost on the applied (post-filter) control. See "
             "--fuel-lin-scale and --fuel-ang-scale.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help=(
            "Total training frames (default 20M). The policy-noise anneal "
            "uses a fraction of this."
        ),
    )
    parser.add_argument(
        "--save-folder",
        default="",
        help=(
            "Output folder for the run. Set it when --restore-file starts a "
            "new experiment, or BenchMARL writes into the folder of the "
            "restored run."
        ),
    )
    parser.add_argument(
        "--save-eval-videos",
        action="store_true",
        default=True,
        help="Save one video per evaluation.",
    )
    parser.add_argument(
        "--no-save-eval-videos",
        action="store_false",
        dest="save_eval_videos",
        help="Do not save evaluation videos.",
    )
    parser.add_argument(
        "--eval-video-dir",
        default="",
        help="Folder for evaluation videos. Empty: "
             "'<run_folder>/eval_videos'.",
    )
    parser.add_argument(
        "--target-lambda",
        type=float,
        default=None,
        help=(
            "Set both target keep-out poles (default 0.15). For --keepout-"
            "mode halfspace use 0.03-0.05."
        ),
    )
    parser.add_argument(
        "--attitude-curriculum",
        action="store_true",
        help=(
            "Tighten the attitude gate together with the position gate, from "
            "10 deg to --end-attitude."
        ),
    )
    parser.add_argument(
        "--end-attitude",
        type=float,
        default=0.0349,
        help="Final attitude capture gate (rad). Default: 0.0349 (2 deg).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help=(
            "Save a checkpoint every N batches (default 10)."
        ),
    )
    parser.add_argument(
        "--curriculum-metric",
        choices=("per_agent", "complete"),
        default="per_agent",
        help=(
            "Evaluation rate for the curriculum steps. 'per_agent' (default):"
            " per-agent dock rate. 'complete': all-agents-docked rate. Use "
            "'complete' when the end gate is tight."
        ),
    )
    parser.add_argument(
        "--end-tolerance",
        type=float,
        default=0.05,
        help=(
            "Final position capture gate (m). Default: 0.05. The gate "
            "tightens after two consecutive evaluations at or above 95%% at "
            "the current gate."
        ),
    )
    parser.add_argument("--eval-video-fps", type=int, default=20, help="Frames per second of the "
                                                                       "evaluation videos.")
    parser.add_argument("--eval-video-ext", default=".mp4", help="Video format (.mp4 or .gif).")
    args = parser.parse_args()

    if args.restore_file:
        print(f"[Resume] Restoring from {args.restore_file}")
    experiment = build_experiment(
        save_eval_videos=args.save_eval_videos,
        eval_video_dir=args.eval_video_dir,
        eval_video_fps=args.eval_video_fps,
        eval_video_ext=args.eval_video_ext,
        restore_file=args.restore_file or None,
        save_folder=args.save_folder or None,
        max_frames=args.max_frames or None,
        disable_cbf=args.no_cbf,
        enable_sensor_noise=args.sensor_noise,
        enable_fuel_cost=args.fuel,
        fuel_lin_scale=args.fuel_lin_scale,
        fuel_ang_scale=args.fuel_ang_scale,
        seed=args.seed,
        safety_obs_features=args.safety_obs_features,
        keepout_mode=args.keepout_mode,
        target_lambda=args.target_lambda,
        attitude_curriculum=args.attitude_curriculum,
        end_attitude=args.end_attitude,
        end_tolerance=args.end_tolerance,
        curriculum_metric=args.curriculum_metric,
        checkpoint_every=args.checkpoint_every,
        restore_map_location=args.restore_map_location,
    )

    interrupted = False
    try:
        experiment.run()
    except KeyboardInterrupt:
        # BenchMARL already closed the experiment. Catch the interrupt here, so the
        # exit does not print false "leaked semaphore" warnings.
        interrupted = True
    if interrupted:
        import gc

        del experiment
        gc.collect()
        print("[Shutdown] Training stopped (Ctrl-C).")
        raise SystemExit(130)
