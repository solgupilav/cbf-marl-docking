from typing import Any, Callable, Dict, List, Optional, Type

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import _read_yaml_config
from benchmarl.utils import DEVICE_TYPING
from torchrl.data import Composite
from torchrl.envs import EnvBase

from marl_docking_6dof.envs import Docking6DoFEnv


class MarlDocking6DoFTaskClass(TaskClass):
    """BenchMARL task for the 6-DOF docking environment."""

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = dict(self.config)

        def _opt_int(key: str) -> int | None:
            value = config.get(key, None)
            return None if value is None else int(value)

        return lambda: Docking6DoFEnv(
            num_envs=num_envs,
            n_agents=int(config.get("n_agents", 4)),
            # None: set from the feature switches.
            obs_dim=_opt_int("obs_dim"),
            action_dim=_opt_int("action_dim"),
            max_steps=int(config.get("max_steps", 120)),
            dt=float(config.get("dt", 1.0)),
            dynamics_substeps=int(config.get("dynamics_substeps", 10)),
            linear_accel_scale=float(config.get("linear_accel_scale", 0.01)),
            goal_tolerance=float(config.get("goal_tolerance", 0.4)),
            max_docking_speed=float(config.get("max_docking_speed", 0.1)),
            enable_attitude=bool(config.get("enable_attitude", False)),
            angular_accel_scale=float(config.get("angular_accel_scale", 0.02)),
            attitude_capture_gate=bool(config.get("attitude_capture_gate", True)),
            max_attitude_error=float(config.get("max_attitude_error", 0.175)),
            max_docking_ang_speed=float(config.get("max_docking_ang_speed", 0.05)),
            reward_att_scale=float(config.get("reward_att_scale", 0.05)),
            reward_angvel_scale=float(config.get("reward_angvel_scale", 0.02)),
            init_max_attitude_angle=float(config.get("init_max_attitude_angle", 0.785)),
            init_max_ang_vel=float(config.get("init_max_ang_vel", 0.0)),
            enable_agent_collisions=bool(config.get("enable_agent_collisions", False)),
            agent_collision_penalty_scale=float(
                config.get("agent_collision_penalty_scale", 1.0)
            ),
            agent_collision_smoothing_distance=float(
                config.get("agent_collision_smoothing_distance", 0.1)
            ),
            agent_collision_margin=float(config.get("agent_collision_margin", 0.0)),
            agent_collision_warning_distance=float(
                config.get("agent_collision_warning_distance", 0.0)
            ),
            agent_collision_terminal_on_contact=bool(
                config.get("agent_collision_terminal_on_contact", False)
            ),
            agent_collision_terminal_penalty=float(
                config.get("agent_collision_terminal_penalty", 0.0)
            ),
            collision_terminal_penalty=float(config.get("collision_terminal_penalty", 0.0)),
            enable_neighbor_obs=bool(config.get("enable_neighbor_obs", False)),
            neighbor_obs_count=int(config.get("neighbor_obs_count", 1)),
            obs_neighbor_pos_scale=config.get("obs_neighbor_pos_scale", None),
            obs_neighbor_vel_scale=config.get("obs_neighbor_vel_scale", None),
            enable_cbf=bool(config.get("enable_cbf", False)),
            safety_filter=config.get("safety_filter", None),
            cbf_agent_keepout=float(config.get("cbf_agent_keepout", 0.55)),
            cbf_lambda1=float(config.get("cbf_lambda1", 0.05)),
            cbf_lambda2=float(config.get("cbf_lambda2", 0.075)),
            cbf_activation_radius=float(config.get("cbf_activation_radius", 1.5)),
            cbf_neighbor_limit=int(config.get("cbf_neighbor_limit", 0)),
            cbf_speed_limit_intercept=float(config.get("cbf_speed_limit_intercept", 0.15)),
            cbf_speed_limit_slope=float(config.get("cbf_speed_limit_slope", 0.015)),
            cbf_speed_limit_gain=float(config.get("cbf_speed_limit_gain", 0.5)),
            cbf_enable_target_keepout=bool(config.get("cbf_enable_target_keepout", True)),
            cbf_target_clearance=float(config.get("cbf_target_clearance", 0.05)),
            cbf_target_lambda1=float(config.get("cbf_target_lambda1", 0.15)),
            cbf_target_lambda2=float(config.get("cbf_target_lambda2", 0.15)),
            cbf_max_ang_speed=float(config.get("cbf_max_ang_speed", 0.2)),
            cbf_ang_speed_gain=float(config.get("cbf_ang_speed_gain", 0.5)),
            cbf_filter_substep_interval=int(config.get("cbf_filter_substep_interval", 1)),
            cbf_intervention_penalty_scale=float(
                config.get("cbf_intervention_penalty_scale", 0.0)
            ),
            cbf_target_keepout_exponent=int(config.get("cbf_target_keepout_exponent", 4)),
            cbf_target_keepout_mode=str(
                config.get("cbf_target_keepout_mode", "superellipsoid")
            ),
            cbf_target_keepout_cover_box=bool(
                config.get("cbf_target_keepout_cover_box", False)
            ),
            enable_terminal_obs=bool(config.get("enable_terminal_obs", False)),
            terminal_obs_scale=float(config.get("terminal_obs_scale", 0.1)),
            init_near_goal_fraction=float(config.get("init_near_goal_fraction", 0.0)),
            init_near_goal_radius=float(config.get("init_near_goal_radius", 0.25)),
            enable_safety_obs=bool(config.get("enable_safety_obs", False)),
            safety_obs_features=config.get("safety_obs_features", None),
            enable_sensor_noise=bool(config.get("enable_sensor_noise", False)),
            sensor_pos_noise_floor=float(config.get("sensor_pos_noise_floor", 0.004)),
            sensor_pos_noise_per_m=float(config.get("sensor_pos_noise_per_m", 0.001)),
            sensor_vel_noise_floor=float(config.get("sensor_vel_noise_floor", 0.001)),
            sensor_vel_noise_per_m=float(config.get("sensor_vel_noise_per_m", 0.0003)),
            sensor_att_noise_floor=float(config.get("sensor_att_noise_floor", 0.0028)),
            sensor_att_noise_per_m=float(config.get("sensor_att_noise_per_m", 0.0005)),
            sensor_angvel_noise_std=float(config.get("sensor_angvel_noise_std", 1.0e-4)),
            sensor_pos_bias_std=float(config.get("sensor_pos_bias_std", 0.003)),
            sensor_vel_bias_std=float(config.get("sensor_vel_bias_std", 0.0005)),
            sensor_att_bias_std=float(config.get("sensor_att_bias_std", 0.0017)),
            sensor_angvel_bias_std=float(config.get("sensor_angvel_bias_std", 2.0e-4)),
            sensor_dropout_prob=float(config.get("sensor_dropout_prob", 0.01)),
            cbf_use_sensed_state=bool(config.get("cbf_use_sensed_state", False)),
            enable_actuator_uncertainty=bool(config.get("enable_actuator_uncertainty", False)),
            actuator_lin_scale_bias_std=float(config.get("actuator_lin_scale_bias_std", 0.10)),
            actuator_ang_scale_bias_std=float(config.get("actuator_ang_scale_bias_std", 0.10)),
            actuator_lin_scale_noise_std=float(config.get("actuator_lin_scale_noise_std", 0.05)),
            actuator_ang_scale_noise_std=float(config.get("actuator_ang_scale_noise_std", 0.05)),
            actuator_lin_align_std=float(config.get("actuator_lin_align_std", 0.0349)),
            actuator_ang_align_std=float(config.get("actuator_ang_align_std", 0.0349)),
            reward_progress_scale=float(config.get("reward_progress_scale", 1.0)),
            reward_time_penalty=float(config.get("reward_time_penalty", 0.03)),
            reward_vel_scale=float(config.get("reward_vel_scale", 0.0)),
            reward_goal_bonus=float(config.get("reward_goal_bonus", 10.0)),
            reward_goal_bonus_depth_weight=float(
                config.get("reward_goal_bonus_depth_weight", 0.0)
            ),
            reward_fuel_scale=float(config.get("reward_fuel_scale", 0.0)),
            reward_fuel_ang_scale=float(config.get("reward_fuel_ang_scale", 0.0)),
            reward_capture_scale=float(config.get("reward_capture_scale", 0.0)),
            capture_shaping_radius=float(config.get("capture_shaping_radius", 0.5)),
            collision_penalty_scale=float(config.get("collision_penalty_scale", 0.0)),
            collision_smoothing_distance=float(config.get("collision_smoothing_distance", 0.2)),
            obs_pos_scale=float(config.get("obs_pos_scale", 10.0)),
            obs_vel_scale=float(config.get("obs_vel_scale", 1.0)),
            init_pos_radius=float(config.get("init_pos_radius", 1.0)),
            init_agent_pos_means=config.get("init_agent_pos_means", None),
            init_agent_pos_ranges=config.get("init_agent_pos_ranges", None),
            min_spawn_separation=float(config.get("min_spawn_separation", 0.0)),
            max_spawn_resample_iters=int(config.get("max_spawn_resample_iters", 100)),
            target_size_x=float(config.get("target_size_x", 2.0)),
            target_size_y=float(config.get("target_size_y", 2.0)),
            target_size_z=float(config.get("target_size_z", 2.2)),
            agent_size_x=float(config.get("agent_size_x", 0.10)),
            agent_size_y=float(config.get("agent_size_y", 0.2263)),
            agent_size_z=float(config.get("agent_size_z", 0.366)),
            target_center_jitter=float(config.get("target_center_jitter", 0.0)),
            docking_offset_x=float(config.get("docking_offset_x", 0.2)),
            port_spread=float(config.get("port_spread", 0.35)),
            seed=seed,
            device=device,
        )

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return False

    def has_render(self, env: EnvBase) -> bool:
        return False

    def max_steps(self, env: EnvBase) -> int:
        return int(self.config["max_steps"])

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        if hasattr(env, "group_map"):
            return env.group_map
        return {"sixdof": [f"sixdof_{i}" for i in range(int(self.config.get("n_agents", 2)))]}

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def observation_spec(self, env: EnvBase) -> Composite:
        observation_spec = env.full_observation_spec_unbatched.clone()
        for group in self.group_map(env):
            if "info" in observation_spec[group]:
                del observation_spec[(group, "info")]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        info_spec = env.full_observation_spec_unbatched.clone()
        for group in self.group_map(env):
            if "observation" in info_spec[group]:
                del info_spec[(group, "observation")]
        for group in self.group_map(env):
            if "info" in info_spec[group]:
                return info_spec
        return None

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec_unbatched

    @staticmethod
    def env_name() -> str:
        return "sixdof_custom"


class MarlDocking6DoFTask(Task):
    """Task enum for 6-DOF docking."""

    DOCKING_6DOF = None

    @staticmethod
    def associated_class() -> Type[TaskClass]:
        return MarlDocking6DoFTaskClass

    def get_from_yaml(self, path: Optional[str] = None) -> MarlDocking6DoFTaskClass:
        config = self._default_config()
        if path is not None:
            loaded = _read_yaml_config(path)
            config.update(loaded)
        return self.get_task(config=config)

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return {
            "n_agents": 4,
            # None: obs and action dims are set from the feature switches.
            "obs_dim": None,
            "action_dim": None,
            "max_steps": 120,
            "dt": 1.0,
            "dynamics_substeps": 10,
            "linear_accel_scale": 0.01,
            "goal_tolerance": 0.4,
            "max_docking_speed": 0.1,
            # Feature switches. All are off by default (translation only).
            "enable_attitude": False,
            "angular_accel_scale": 0.02,
            "attitude_capture_gate": True,
            "max_attitude_error": 0.175,
            "max_docking_ang_speed": 0.05,
            "reward_att_scale": 0.05,
            "reward_angvel_scale": 0.02,
            "init_max_attitude_angle": 0.785,
            "init_max_ang_vel": 0.0,
            "enable_agent_collisions": False,
            "agent_collision_penalty_scale": 1.0,
            "agent_collision_smoothing_distance": 0.1,
            "agent_collision_margin": 0.0,
            "agent_collision_warning_distance": 0.0,
            "agent_collision_terminal_on_contact": False,
            "agent_collision_terminal_penalty": 0.0,
            "collision_terminal_penalty": 0.0,
            "enable_neighbor_obs": False,
            "neighbor_obs_count": 1,
            "obs_neighbor_pos_scale": None,
            "obs_neighbor_vel_scale": None,
            "enable_cbf": False,
            # None: the environment makes the default HOCBFSafetyFilter from the
            # cbf_* parameters when enable_cbf=True.
            "safety_filter": None,
            "cbf_agent_keepout": 0.55,
            "cbf_lambda1": 0.05,
            "cbf_lambda2": 0.075,
            # inf: apply the inter-agent row to all pairs. Distant rows are slack and
            # cost little.
            "cbf_activation_radius": float("inf"),
            "cbf_neighbor_limit": 0,
            "cbf_speed_limit_intercept": 0.15,
            "cbf_speed_limit_slope": 0.015,
            "cbf_speed_limit_gain": 0.5,
            "cbf_enable_target_keepout": True,
            "cbf_target_clearance": 0.05,
            "cbf_target_lambda1": 0.15,
            "cbf_target_lambda2": 0.15,
            "cbf_max_ang_speed": 0.2,
            "cbf_ang_speed_gain": 0.5,
            "cbf_filter_substep_interval": 1,
            "cbf_intervention_penalty_scale": 0.0,
            "cbf_target_keepout_exponent": 4,
            "cbf_target_keepout_mode": "superellipsoid",
            "cbf_target_keepout_cover_box": False,
            "enable_terminal_obs": False,
            "terminal_obs_scale": 0.1,
            "init_near_goal_fraction": 0.0,
            "init_near_goal_radius": 0.25,
            "enable_safety_obs": False,
            # None: use all four features. A tuple keeps only those indices.
            "safety_obs_features": None,
            "enable_sensor_noise": False,
            "sensor_pos_noise_floor": 0.004,
            "sensor_pos_noise_per_m": 0.001,
            "sensor_vel_noise_floor": 0.001,
            "sensor_vel_noise_per_m": 0.0003,
            "sensor_att_noise_floor": 0.0028,
            "sensor_att_noise_per_m": 0.0005,
            "sensor_angvel_noise_std": 1.0e-4,
            "sensor_pos_bias_std": 0.003,
            "sensor_vel_bias_std": 0.0005,
            "sensor_att_bias_std": 0.0017,
            "sensor_angvel_bias_std": 2.0e-4,
            "sensor_dropout_prob": 0.01,
            # Actuator uncertainty. The defaults are the high case. The feature is
            # off unless enable_actuator_uncertainty is set.
            # If cbf_use_sensed_state is set, the filter reads the noisy estimate.
            "cbf_use_sensed_state": False,
            "enable_actuator_uncertainty": False,
            "actuator_lin_scale_bias_std": 0.10,
            "actuator_ang_scale_bias_std": 0.10,
            "actuator_lin_scale_noise_std": 0.05,
            "actuator_ang_scale_noise_std": 0.05,
            "actuator_lin_align_std": 0.0349,
            "actuator_ang_align_std": 0.0349,
            "reward_progress_scale": 1.0,
            "reward_time_penalty": 0.03,
            "reward_vel_scale": 0.0,
            "reward_goal_bonus": 10.0,
            "reward_goal_bonus_depth_weight": 0.0,
            # Delta-v weights. 0.0 gives no fuel cost.
            "reward_fuel_scale": 0.0,
            "reward_fuel_ang_scale": 0.0,
            "reward_capture_scale": 0.0,
            "capture_shaping_radius": 0.5,
            "collision_penalty_scale": 0.0,
            "collision_smoothing_distance": 0.2,
            "obs_pos_scale": 10.0,
            "obs_vel_scale": 1.0,
            "init_pos_radius": 1.0,
            "init_agent_pos_means": None,
            "init_agent_pos_ranges": None,
            # 0.0: off. A value > 0 sets a minimum spawn separation (rejection sampling).
            "min_spawn_separation": 0.0,
            "max_spawn_resample_iters": 100,
            "target_size_x": 2.0,
            "target_size_y": 2.0,
            "target_size_z": 2.2,
            "agent_size_x": 0.10,
            "agent_size_y": 0.2263,
            "agent_size_z": 0.366,
            "target_center_jitter": 0.0,
            "docking_offset_x": 0.2,
            "port_spread": 0.35,
        }
