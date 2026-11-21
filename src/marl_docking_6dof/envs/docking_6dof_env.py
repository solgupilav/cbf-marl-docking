from __future__ import annotations

import math
from typing import Optional

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import Bounded, Categorical, Composite, Unbounded
from torchrl.envs.common import EnvBase

from marl_docking_6dof.safety.cbf import CBFSafetyFilter, HOCBFSafetyFilter


class Docking6DoFEnv(EnvBase):
    """Multi-agent spacecraft docking environment (TorchRL).

    Each agent moves to its own docking port on the +X face of a target
    spacecraft. The dynamics are Clohessy-Wiltshire translation and, with
    enable_attitude, quaternion attitude with torque control.

    Feature switches (all off by default):
      enable_attitude          attitude state, torque actions, attitude error in
                               the observation and reward, attitude capture gate.
      enable_agent_collisions  inter-agent bounding-sphere collision penalty and
                               terminal contact.
      enable_neighbor_obs      relative position and velocity of the nearest
                               teammates in the observation.
      enable_cbf               send the action through the HOCBF-QP safety filter
                               (safety/cbf.py). The policy action is the nominal.
      enable_safety_obs        add the safety features to the observation.
      enable_sensor_noise      add noise, bias and dropout to the observed relative
                               state. Reward, capture, physics and the filter use
                               the true state unless cbf_use_sensed_state is set.

    Observation (obs_dim is set from the switches):
        [rel_goal(3), vel(3)]                                   6
        + [2*q_err_vec(3), ang_vel_err(3)]  enable_attitude     +6
        + [rel_pos_j(3), rel_vel_j(3)] per neighbour            +6 each
        + safety features                   enable_safety_obs   +1 to +4
        + [tanh(rel_goal / terminal_obs_scale)(3)]              +3

    Action (action_dim is set from the switches):
        [linear_accel(3)] + [angular_accel(3) if enable_attitude]

    Reward: potential-based progress, a small time penalty, potential-based
    arrive-and-stop shaping, a capture bonus, the attitude penalty
    -alpha^T Q alpha, collision penalties, filter intervention and delta-v cost.
    """

    batch_locked = True

    def __init__(
        self,
        *,
        num_envs: int,
        n_agents: int = 4,
        # None: obs_dim and action_dim are set from the feature switches.
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        max_steps: int = 120,
        dt: float = 1.0,
        dynamics_substeps: int = 10,
        linear_accel_scale: float = 0.01,
        goal_tolerance: float = 0.4,
        # Maximum relative speed at capture (m/s).
        max_docking_speed: float = 0.1,
        # Attitude.
        enable_attitude: bool = False,
        # Maximum angular acceleration (rad/s^2) at |action| = 1. Small reaction
        # wheels on a 6U CubeSat give about 0.02 rad/s^2.
        angular_accel_scale: float = 0.02,
        # Attitude capture gate (only with enable_attitude). Angle in radians
        # (0.175 rad is about 10 deg).
        attitude_capture_gate: bool = True,
        max_attitude_error: float = 0.175,
        max_docking_ang_speed: float = 0.05,
        # Diagonal of Q in alpha^T Q alpha, alpha = [2*q_err_vec, ang_vel_err]:
        # Q = diag(reward_att_scale*I3, reward_angvel_scale*I3).
        reward_att_scale: float = 0.05,
        reward_angvel_scale: float = 0.02,
        # Initial attitude: rotation about a random axis by an angle uniform in
        # [0, init_max_attitude_angle] rad. Initial rate: uniform per axis in
        # +/- init_max_ang_vel rad/s.
        init_max_attitude_angle: float = 0.785,
        init_max_ang_vel: float = 0.0,
        reward_progress_scale: float = 1.0,
        reward_time_penalty: float = 0.03,
        reward_vel_scale: float = 0.001,
        reward_goal_bonus: float = 10.0,
        reward_goal_bonus_depth_weight: float = 0.0,
        # Arrive-and-stop shaping: reward for being near the port and slow at the
        # same time. Active only inside capture_shaping_radius.
        # Fuel: -reward_fuel_scale * ||dv_lin|| (m/s) and
        # -reward_fuel_ang_scale * ||dv_ang|| (rad/s) of the applied (post-filter)
        # control. The linear weight is small. The angular weight mainly removes
        # torque commands after the agent is aligned. Both are 0.0 by default.
        reward_fuel_scale: float = 0.0,
        reward_fuel_ang_scale: float = 0.0,
        reward_capture_scale: float = 0.0,
        capture_shaping_radius: float = 0.5,
        collision_penalty_scale: float = 0.0,
        collision_smoothing_distance: float = 0.2,
        # Inter-agent collisions.
        enable_agent_collisions: bool = False,
        # Bounding-sphere collision model. The penalty ramps over
        # agent_collision_smoothing_distance of overlap.
        agent_collision_penalty_scale: float = 1.0,
        agent_collision_smoothing_distance: float = 0.1,
        # Extra margin (m) on top of the two bounding radii.
        agent_collision_margin: float = 0.0,
        # Width (m) of a warning band before contact. The penalty starts at
        # contact_dist + warning. 0.0: penalty after contact only.
        agent_collision_warning_distance: float = 0.0,
        # True: any contact is terminal. False: terminal only when the penetration
        # is >= smoothing_distance.
        agent_collision_terminal_on_contact: bool = False,
        # One-time penalty on the step of a terminal crash. 0.0: off.
        agent_collision_terminal_penalty: float = 0.0,
        collision_terminal_penalty: float = 0.0,
        # Neighbour observation: relative position and velocity of the nearest
        # teammates.
        enable_neighbor_obs: bool = False,
        # Number of neighbours (nearest first, 6 obs each). Missing neighbours are
        # zero.
        neighbor_obs_count: int = 1,
        # Scales for the neighbour features. None: use obs_pos_scale and
        # obs_vel_scale.
        obs_neighbor_pos_scale: Optional[float] = None,
        obs_neighbor_vel_scale: Optional[float] = None,
        # CBF safety filter. If True, the action goes through
        # safety_filter.filter_action before integration. The default filter is
        # HOCBFSafetyFilter, made from the cbf_* parameters. A custom safety_filter
        # can be given.
        enable_cbf: bool = False,
        safety_filter: Optional[CBFSafetyFilter] = None,
        # Inter-agent keep-apart distance D_s (m). Contact is 2r (about 0.442 m).
        cbf_agent_keepout: float = 0.55,
        # HOCBF poles -lambda1 and -lambda2.
        cbf_lambda1: float = 0.05,
        cbf_lambda2: float = 0.075,
        # Pairs beyond this range get no row. Default inf: a finite radius lets a
        # pair enter range too fast to stop.
        cbf_activation_radius: float = float("inf"),
        cbf_neighbor_limit: int = 0,
        # Target speed limit ||v|| <= nu0 + nu1*range. nu0 is above
        # max_docking_speed, so the limit does not block capture.
        cbf_speed_limit_intercept: float = 0.15,
        cbf_speed_limit_slope: float = 0.015,
        cbf_speed_limit_gain: float = 0.5,
        # Target keep-out: superellipsoid around the target body. The clearance is
        # anisotropic, so the docked pose stays outside.
        cbf_enable_target_keepout: bool = True,
        cbf_target_clearance: float = 0.05,
        cbf_target_lambda1: float = 0.15,
        cbf_target_lambda2: float = 0.15,
        # Angular-rate limit (rad/s) and its gain.
        cbf_max_ang_speed: float = 0.2,
        cbf_ang_speed_gain: float = 0.5,
        # Solve interval of the QP in dynamics substeps. 1: every substep.
        # >= dynamics_substeps: once per control step. With the longer interval,
        # the D_s buffer (D_s - 2r) must cover the drift between solves.
        cbf_filter_substep_interval: int = 1,
        # Intervention penalty c_int: r_int = -c_int * ||u_safe - u_nom||^2
        # (normalised units). 0.0: off.
        cbf_intervention_penalty_scale: float = 0.0,
        cbf_target_keepout_exponent: int = 4,
        # "superellipsoid" or "halfspace" (see HOCBFSafetyFilter).
        cbf_target_keepout_mode: str = "superellipsoid",
        # Make the superellipsoid contain the whole contact box (see where the axes
        # are made below). Off by default.
        cbf_target_keepout_cover_box: bool = False,
        enable_terminal_obs: bool = False,
        terminal_obs_scale: float = 0.1,
        init_near_goal_fraction: float = 0.0,
        init_near_goal_radius: float = 0.25,
        # Add the safety features [gap-to-keepout, tanh(psi1), last intervention,
        # filter-active flag] to the observation (+4 obs).
        enable_safety_obs: bool = False,
        # Safety features to keep, by index (0 gap, 1 tanh(psi1), 2 intervention,
        # 3 active flag). None keeps all four.
        safety_obs_features: Optional[tuple[int, ...]] = None,
        # Sensor noise (observation only).
        # The model stands for a vision-based relative-pose sensor (camera and
        # fiducial markers, as in Pirat et al. 2018 and Stesina 2021) and a MEMS
        # gyro. The error model is bias + noise + dropout on the delivered relative
        # pose and rate. The gyro is not affected by vision dropout.
        enable_sensor_noise: bool = False,
        # Vision noise (1-sigma) = floor + slope * range. The floors are the
        # demonstrated accuracy in Pirat et al. 2018 (3.7 mm, 0.16 deg). The slope is
        # an assumption: the error stays below 1 cm / 2 deg at the 10 m spawn range.
        sensor_pos_noise_floor: float = 0.004,
        sensor_pos_noise_per_m: float = 0.001,
        sensor_vel_noise_floor: float = 0.001,
        sensor_vel_noise_per_m: float = 0.0003,
        sensor_att_noise_floor: float = 0.0028,
        sensor_att_noise_per_m: float = 0.0005,
        # Gyro white noise, from the angle random walk of a COTS MEMS gyro
        # (PG400, 0.23 deg/sqrt(hr)): about 6.7e-5 rad/s at 1 Hz, rounded up.
        sensor_angvel_noise_std: float = 1.0e-4,
        # Constant bias per episode, drawn at reset. Same order as the noise floors.
        sensor_pos_bias_std: float = 0.003,
        sensor_vel_bias_std: float = 0.0005,
        sensor_att_bias_std: float = 0.0017,
        # Gyro turn-on bias. This is a conservative estimate (about 40 times the
        # in-run bias stability), because each episode is a new power-on.
        sensor_angvel_bias_std: float = 2.0e-4,
        # Probability per step that the vision data (position, velocity, attitude)
        # is lost. The last reading is then held. Independent for each tracked object.
        # The gyro is not affected.
        sensor_dropout_prob: float = 0.01,
        # Actuator uncertainty: the applied control differs from the filter output.
        #
        #     u_exec = R(delta) [ (1 + s + eta) .* u ]
        #
        # s      scale-factor error per episode and axis (calibration).
        # eta    scale jitter per step (valve repeatability, torque ripple).
        # R      small rotation per episode about a random axis (misalignment).
        #        This gives off-axis force.
        # The defaults are the high case (10% scale, 2 deg). A typical system is
        # about 2-3% and 0.5 deg.
        #
        # cbf_use_sensed_state: give the filter the same noisy estimate as the policy.
        # The filter uses pos, vel and ang_vel. Position is p_hat = goal -
        # sensed_rel_goal. The estimate is from the start of the control step.
        # Both agents of a pair see the same separation estimate.
        cbf_use_sensed_state: bool = False,
        enable_actuator_uncertainty: bool = False,
        actuator_lin_scale_bias_std: float = 0.10,
        actuator_ang_scale_bias_std: float = 0.10,
        actuator_lin_scale_noise_std: float = 0.05,
        actuator_ang_scale_noise_std: float = 0.05,
        # Misalignment 1-sigma in radians (0.0349 = 2 deg high case,
        # 0.0087 = 0.5 deg typical).
        actuator_lin_align_std: float = 0.0349,
        actuator_ang_align_std: float = 0.0349,
        obs_pos_scale: float = 10.0,
        obs_vel_scale: float = 1.0,
        init_pos_radius: float = 1.0,
        init_agent_pos_means: Optional[list[list[float]]] = None,
        init_agent_pos_ranges: Optional[list[list[float] | float]] = None,
        # Minimum spawn separation (m), by rejection sampling of the whole env.
        # 0.0: off.
        min_spawn_separation: float = 0.0,
        max_spawn_resample_iters: int = 100,
        target_size_x: float = 2.0,
        target_size_y: float = 2.0,
        target_size_z: float = 2.2,
        # Chaser: 6U CubeSat (0.1 x 0.2263 x 0.366 m). It docks on its broad face,
        # so the depth along the approach axis (x) is about 0.1 m.
        agent_size_x: float = 0.10,
        agent_size_y: float = 0.2263,
        agent_size_z: float = 0.366,
        target_center_jitter: float = 0.0,
        docking_offset_x: float = 0.2,
        # Port spread on the +X face, as a fraction of min(half_y, half_z).
        port_spread: float = 0.35,
        seed: Optional[int] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.n_agents = int(n_agents)
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.dynamics_substeps = max(int(dynamics_substeps), 1)
        self.linear_accel_scale = float(linear_accel_scale)
        self.goal_tolerance = float(goal_tolerance)
        self.max_docking_speed = max(float(max_docking_speed), 0.0)

        # Feature switches.
        self.enable_attitude = bool(enable_attitude)
        self.enable_agent_collisions = bool(enable_agent_collisions)
        self.enable_neighbor_obs = bool(enable_neighbor_obs)
        self.neighbor_obs_count = max(int(neighbor_obs_count), 1)
        self.enable_cbf = bool(enable_cbf)
        self.enable_safety_obs = bool(enable_safety_obs)
        self.safety_obs_features: tuple[int, ...] = (
            (0, 1, 2, 3) if safety_obs_features is None
            else tuple(sorted({int(i) for i in safety_obs_features}))
        )
        if any(i < 0 or i > 3 for i in self.safety_obs_features):
            raise ValueError("safety_obs_features indices must lie in 0..3")
        if not self.safety_obs_features:
            raise ValueError("safety_obs_features cannot be empty; "
                             "use enable_safety_obs=False to drop the block")
        self.enable_sensor_noise = bool(enable_sensor_noise)
        self.sensor_pos_noise_floor = max(float(sensor_pos_noise_floor), 0.0)
        self.sensor_pos_noise_per_m = max(float(sensor_pos_noise_per_m), 0.0)
        self.sensor_vel_noise_floor = max(float(sensor_vel_noise_floor), 0.0)
        self.sensor_vel_noise_per_m = max(float(sensor_vel_noise_per_m), 0.0)
        self.sensor_att_noise_floor = max(float(sensor_att_noise_floor), 0.0)
        self.sensor_att_noise_per_m = max(float(sensor_att_noise_per_m), 0.0)
        self.sensor_angvel_noise_std = max(float(sensor_angvel_noise_std), 0.0)
        self.sensor_pos_bias_std = max(float(sensor_pos_bias_std), 0.0)
        self.sensor_vel_bias_std = max(float(sensor_vel_bias_std), 0.0)
        self.sensor_att_bias_std = max(float(sensor_att_bias_std), 0.0)
        self.sensor_angvel_bias_std = max(float(sensor_angvel_bias_std), 0.0)
        self.sensor_dropout_prob = min(max(float(sensor_dropout_prob), 0.0), 1.0)
        self.cbf_use_sensed_state = bool(cbf_use_sensed_state)
        self.enable_actuator_uncertainty = bool(enable_actuator_uncertainty)
        self.actuator_lin_scale_bias_std = max(float(actuator_lin_scale_bias_std), 0.0)
        self.actuator_ang_scale_bias_std = max(float(actuator_ang_scale_bias_std), 0.0)
        self.actuator_lin_scale_noise_std = max(float(actuator_lin_scale_noise_std), 0.0)
        self.actuator_ang_scale_noise_std = max(float(actuator_ang_scale_noise_std), 0.0)
        self.actuator_lin_align_std = max(float(actuator_lin_align_std), 0.0)
        self.actuator_ang_align_std = max(float(actuator_ang_align_std), 0.0)

        # CBF parameters (used by the safety features and the default filter).
        self.cbf_agent_keepout = float(cbf_agent_keepout)
        self.cbf_lambda1 = float(cbf_lambda1)
        self.cbf_lambda2 = float(cbf_lambda2)
        self.cbf_activation_radius = max(float(cbf_activation_radius), 1e-6)
        self.cbf_neighbor_limit = int(cbf_neighbor_limit)
        # Scale of the gap feature: the activation radius if finite, else 1.5 m.
        self._cbf_gap_obs_scale = (
            self.cbf_activation_radius
            if self.cbf_activation_radius != float("inf")
            else 1.5
        )
        self.cbf_speed_limit_intercept = float(cbf_speed_limit_intercept)
        self.cbf_speed_limit_slope = float(cbf_speed_limit_slope)
        self.cbf_speed_limit_gain = float(cbf_speed_limit_gain)
        self.cbf_max_ang_speed = float(cbf_max_ang_speed)
        self.cbf_ang_speed_gain = float(cbf_ang_speed_gain)
        self.cbf_filter_substep_interval = max(int(cbf_filter_substep_interval), 1)
        self.cbf_intervention_penalty_scale = float(cbf_intervention_penalty_scale)
        self.cbf_target_keepout_exponent = int(cbf_target_keepout_exponent)
        self.cbf_target_keepout_mode = str(cbf_target_keepout_mode)
        self.cbf_target_keepout_cover_box = bool(cbf_target_keepout_cover_box)
        # Gate-relative position feature: tanh(rel_goal / terminal_obs_scale). The
        # scale is a fixed length, not goal_tolerance, so the feature does not change
        # when the curriculum tightens the gate.
        self.enable_terminal_obs = bool(enable_terminal_obs)
        self.terminal_obs_scale = max(float(terminal_obs_scale), 1e-6)
        # Near-goal respawn (training aid; 0 for evaluation). This fraction of agents
        # starts in a ball of init_near_goal_radius around its own port. About half
        # of them start behind the port, so the policy learns to recover from there.
        self.init_near_goal_fraction = min(max(float(init_near_goal_fraction), 0.0), 1.0)
        self.init_near_goal_radius = max(float(init_near_goal_radius), 0.0)
        self.cbf_enable_target_keepout = bool(cbf_enable_target_keepout)
        self.cbf_target_clearance = float(cbf_target_clearance)
        self.cbf_target_lambda1 = float(cbf_target_lambda1)
        self.cbf_target_lambda2 = float(cbf_target_lambda2)

        # Port grid and target size. Ports are on a centred rows x cols grid on the
        # +X face at a fixed pitch. The face (y, z) grows if the grid does not fit.
        self.port_cols = int(math.ceil(math.sqrt(max(self.n_agents, 1))))
        self.port_rows = int(math.ceil(self.n_agents / self.port_cols))
        _half_y0 = 0.5 * float(target_size_y)
        _half_z0 = 0.5 * float(target_size_z)
        self.port_pitch = 2.0 * float(port_spread) * min(_half_y0, _half_z0)
        # Keep the ports half an agent from the edge of the face.
        _port_margin = 0.5 * float(agent_size_y)
        _req_half_y = (self.port_cols - 1) / 2.0 * self.port_pitch + _port_margin
        _req_half_z = (self.port_rows - 1) / 2.0 * self.port_pitch + _port_margin
        _tgt_y = max(float(target_size_y), 2.0 * _req_half_y)
        _tgt_z = max(float(target_size_z), 2.0 * _req_half_z)
        # The enlarged sizes are used for the target body and the keep-out.
        target_size_y, target_size_z = _tgt_y, _tgt_z

        # Keep-out semi-axes: half-depth of the chaser along the approach axis,
        # bounding-sphere radius laterally.
        r_bound = 0.5 * (
            float(agent_size_x) ** 2 + float(agent_size_y) ** 2 + float(agent_size_z) ** 2
        ) ** 0.5
        # Contact half-extents: the smallest box that contains the chaser when it
        # touches the target. x uses the half-depth, y and z the bounding radius.
        contact_half = (
            0.5 * float(target_size_x) + 0.5 * float(agent_size_x),
            0.5 * float(target_size_y) + r_bound,
            0.5 * float(target_size_z) + r_bound,
        )
        keepout_axes = tuple(c + self.cbf_target_clearance for c in contact_half)

        if self.cbf_target_keepout_cover_box:
            # Coverage. Over the contact box, the sum in h_T is largest at the corner.
            # With rho = contact_half + clearance at n = 8, the corner has h_T > 0.
            # rho_k = contact_half_k * 3^(1/n) puts the corner on the surface:
            #
            #     rho_i = max(contact_half_i + clearance, contact_half_i * 3^(1/n))
            #
            # At n = 8 this puts the docked pose inside the keep-out. At n = 16 the x
            # axis does not change and the lateral axes grow by about 2 cm.
            grow = 3.0 ** (1.0 / float(self.cbf_target_keepout_exponent))
            keepout_axes = tuple(
                max(axis, c * grow) for axis, c in zip(keepout_axes, contact_half)
            )

        self.cbf_target_keepout_axes: Optional[tuple[float, float, float]] = (
            keepout_axes if self.cbf_enable_target_keepout else None
        )

        # Attitude parameters.
        self.angular_accel_scale = float(angular_accel_scale)
        self.attitude_capture_gate = bool(attitude_capture_gate)
        self.max_attitude_error = float(max_attitude_error)
        self.max_docking_ang_speed = float(max_docking_ang_speed)
        self.reward_att_scale = float(reward_att_scale)
        self.reward_angvel_scale = float(reward_angvel_scale)
        self.init_max_attitude_angle = float(init_max_attitude_angle)
        self.init_max_ang_vel = float(init_max_ang_vel)

        # Inter-agent collision parameters.
        self.agent_collision_penalty_scale = float(agent_collision_penalty_scale)
        self.agent_collision_smoothing_distance = max(
            float(agent_collision_smoothing_distance), 1e-6
        )
        self.agent_collision_margin = float(agent_collision_margin)
        self.agent_collision_warning_distance = max(
            float(agent_collision_warning_distance), 0.0
        )
        self.agent_collision_terminal_on_contact = bool(agent_collision_terminal_on_contact)
        self.agent_collision_terminal_penalty = float(agent_collision_terminal_penalty)
        self.collision_terminal_penalty = float(collision_terminal_penalty)

        # obs_dim and action_dim (explicit values are used if given).
        derived_obs = (
            6
            + (6 if self.enable_attitude else 0)
            + (6 * self.neighbor_obs_count if self.enable_neighbor_obs else 0)
            + (len(self.safety_obs_features) if self.enable_safety_obs else 0)
            + (3 if self.enable_terminal_obs else 0)
        )
        derived_act = 3 + (3 if self.enable_attitude else 0)
        self.obs_dim = int(obs_dim) if obs_dim is not None else derived_obs
        self.action_dim = int(action_dim) if action_dim is not None else derived_act

        # Mean motion of the target orbit, for the CW dynamics and the filter model.
        mu = 3.986e14
        rad = 42000e3
        self.mean_motion = (mu / rad**3) ** 0.5

        # The filter object always exists, but it runs only when enable_cbf=True.
        if safety_filter is not None:
            self.safety_filter = safety_filter
        elif self.enable_cbf:
            self.safety_filter = HOCBFSafetyFilter(
                linear_accel_scale=float(linear_accel_scale),
                angular_accel_scale=float(angular_accel_scale),
                agent_keepout=self.cbf_agent_keepout,
                lambda1=self.cbf_lambda1,
                lambda2=self.cbf_lambda2,
                activation_radius=self.cbf_activation_radius,
                neighbor_limit=self.cbf_neighbor_limit,
                speed_limit_intercept=self.cbf_speed_limit_intercept,
                speed_limit_slope=self.cbf_speed_limit_slope,
                speed_limit_gain=self.cbf_speed_limit_gain,
                target_keepout_axes=self.cbf_target_keepout_axes,
                target_lambda1=self.cbf_target_lambda1,
                target_lambda2=self.cbf_target_lambda2,
                target_keepout_exponent=self.cbf_target_keepout_exponent,
                target_keepout_mode=self.cbf_target_keepout_mode,
                max_ang_speed=self.cbf_max_ang_speed,
                ang_speed_gain=self.cbf_ang_speed_gain,
                mean_motion=self.mean_motion,
            )
        else:
            self.safety_filter = CBFSafetyFilter(linear_accel_scale=float(linear_accel_scale))
        self.reward_progress_scale = float(reward_progress_scale)
        self.reward_time_penalty = float(reward_time_penalty)
        self.reward_vel_scale = float(reward_vel_scale)
        self.reward_goal_bonus = float(reward_goal_bonus)
        # Capture bonus depth weight w. bonus = B * (1 + w * depth),
        # depth = 1 - dist/goal_tolerance at capture. An edge capture gives B and the
        # centre gives B*(1+w). 0: flat bonus. The premium is added, so it never
        # lowers the reward.
        self.reward_goal_bonus_depth_weight = max(
            float(reward_goal_bonus_depth_weight), 0.0
        )
        self.reward_fuel_scale = max(float(reward_fuel_scale), 0.0)
        self.reward_fuel_ang_scale = max(float(reward_fuel_ang_scale), 0.0)
        self.reward_capture_scale = float(reward_capture_scale)
        self.capture_shaping_radius = max(float(capture_shaping_radius), 1e-6)
        self.collision_penalty_scale = float(collision_penalty_scale)
        self.collision_smoothing_distance = max(float(collision_smoothing_distance), 1e-6)
        self.obs_pos_scale = max(float(obs_pos_scale), 1e-6)
        self.obs_vel_scale = max(float(obs_vel_scale), 1e-6)
        self.obs_neighbor_pos_scale = (
            self.obs_pos_scale
            if obs_neighbor_pos_scale is None
            else max(float(obs_neighbor_pos_scale), 1e-6)
        )
        self.obs_neighbor_vel_scale = (
            self.obs_vel_scale
            if obs_neighbor_vel_scale is None
            else max(float(obs_neighbor_vel_scale), 1e-6)
        )
        self.init_pos_radius = float(init_pos_radius)
        self.min_spawn_separation = max(float(min_spawn_separation), 0.0)
        self.max_spawn_resample_iters = max(int(max_spawn_resample_iters), 1)
        self._warned_spawn_infeasible = False
        self._init_agent_pos_means_raw = init_agent_pos_means
        self._init_agent_pos_ranges_raw = init_agent_pos_ranges
        self.init_agent_pos_means: Optional[torch.Tensor] = None
        self.init_agent_pos_ranges: Optional[torch.Tensor] = None
        self.target_center_jitter = float(target_center_jitter)
        self.docking_offset_x = float(docking_offset_x)
        self.port_spread = float(port_spread)
        self.target_size = torch.tensor(
            [float(target_size_x), float(target_size_y), float(target_size_z)],
            dtype=torch.float32,
            device=device,
        )
        self.agent_size = torch.tensor(
            [float(agent_size_x), float(agent_size_y), float(agent_size_z)],
            dtype=torch.float32,
            device=device,
        )
        self.group = "sixdof"
        self.group_map = {
            self.group: [f"{self.group}_{i}" for i in range(self.n_agents)]
        }

        super().__init__(device=device, batch_size=torch.Size([self.num_envs]))
        self._make_spec()
        self.set_seed(seed if seed is not None else 0)
        self._allocate_state()
        self._set_agent_spawn_specs(
            means=self._init_agent_pos_means_raw,
            ranges=self._init_agent_pos_ranges_raw,
        )

    # Specs.
    def _make_spec(self) -> None:
        device = self.device
        observation_spec = Composite(
            {
                self.group: Composite(
                    {
                        "observation": Unbounded(
                            shape=(self.n_agents, self.obs_dim),
                            device=device,
                            dtype=torch.float32,
                        ),
                        "info": Composite(
                            {
                                "distance_to_goal": Unbounded(
                                    shape=(self.n_agents, 1), device=device, dtype=torch.float32
                                ),
                                "relative_speed": Unbounded(
                                    shape=(self.n_agents, 1), device=device, dtype=torch.float32
                                ),
                                "on_goal": Categorical(
                                    n=2, shape=(self.n_agents, 1), device=device, dtype=torch.bool
                                ),
                                "complete_success": Categorical(
                                    n=2, shape=(self.n_agents, 1), device=device, dtype=torch.bool
                                ),
                                "goal_bonus_awarded": Categorical(
                                    n=2, shape=(self.n_agents, 1), device=device, dtype=torch.bool
                                ),
                                "reward_shaped": Unbounded(
                                    shape=(self.n_agents, 1), device=device, dtype=torch.float32
                                ),
                                "reward_progress": Unbounded(
                                    shape=(self.n_agents, 1), device=device, dtype=torch.float32
                                ),
                                "reward_time": Unbounded(
                                    shape=(self.n_agents, 1), device=device, dtype=torch.float32
                                ),
                                "reward_velocity": Unbounded(
                                    shape=(self.n_agents, 1), device=device, dtype=torch.float32
                                ),
                                "reward_capture": Unbounded(
                                    shape=(self.n_agents, 1), device=device, dtype=torch.float32
                                ),
                                "reward_bonus": Unbounded(
                                    shape=(self.n_agents, 1), device=device, dtype=torch.float32
                                ),
                                "reward_collision": Unbounded(
                                    shape=(self.n_agents, 1), device=device, dtype=torch.float32
                                ),
                                "target_center": Unbounded(
                                    shape=(self.n_agents, 3), device=device, dtype=torch.float32
                                ),
                                "target_size": Unbounded(
                                    shape=(self.n_agents, 3), device=device, dtype=torch.float32
                                ),
                                "agent_position": Unbounded(
                                    shape=(self.n_agents, 3), device=device, dtype=torch.float32
                                ),
                                "goal_position": Unbounded(
                                    shape=(self.n_agents, 3), device=device, dtype=torch.float32
                                ),
                                "collision": Categorical(
                                    n=2, shape=(self.n_agents, 1), device=device, dtype=torch.bool
                                ),
                                "collision_penetration": Unbounded(
                                    shape=(self.n_agents, 1), device=device, dtype=torch.float32
                                ),
                                "collision_terminal": Categorical(
                                    n=2, shape=(self.n_agents, 1), device=device, dtype=torch.bool
                                ),
                                "numerical_error": Categorical(
                                    n=2, shape=(self.n_agents, 1), device=device, dtype=torch.bool
                                ),
                                "agent_size": Unbounded(
                                    shape=(self.n_agents, 3), device=device, dtype=torch.float32
                                ),
                                # Attitude (identity or zero when disabled).
                                "quaternion": Unbounded(
                                    shape=(self.n_agents, 4), device=device, dtype=torch.float32
                                ),
                                "angular_velocity": Unbounded(
                                    shape=(self.n_agents, 3), device=device, dtype=torch.float32
                                ),
                                "attitude_error_angle": Unbounded(
                                    shape=(self.n_agents, 1), device=device, dtype=torch.float32
                                ),
                                "reward_attitude": Unbounded(
                                    shape=(self.n_agents, 1), device=device, dtype=torch.float32
                                ),
                                # Inter-agent collisions (zero when disabled).
                                "agent_collision": Categorical(
                                    n=2, shape=(self.n_agents, 1), device=device, dtype=torch.bool
                                ),
                                "agent_collision_penetration": Unbounded(
                                    shape=(self.n_agents, 1), device=device, dtype=torch.float32
                                ),
                                "agent_collision_terminal": Categorical(
                                    n=2, shape=(self.n_agents, 1), device=device, dtype=torch.bool
                                ),
                                "min_agent_separation": Unbounded(
                                    shape=(self.n_agents, 1), device=device, dtype=torch.float32
                                ),
                                "reward_agent_collision": Unbounded(
                                    shape=(self.n_agents, 1), device=device, dtype=torch.float32
                                ),
                                # CBF safety filter (zero when disabled).
                                "cbf_intervention": Unbounded(
                                    shape=(self.n_agents, 1), device=device, dtype=torch.float32
                                ),
                                "reward_cbf_intervention": Unbounded(
                                    shape=(self.n_agents, 1), device=device, dtype=torch.float32
                                ),
                                # Fraction of substeps with a hard-infeasible QP
                                # (backup used). Should be 0.
                                "cbf_infeasible": Unbounded(
                                    shape=(self.n_agents, 1), device=device, dtype=torch.float32
                                ),
                                # Fraction of substeps with a soft row
                                # (speed limit or keep-out) dropped.
                                "cbf_relaxed": Unbounded(
                                    shape=(self.n_agents, 1), device=device, dtype=torch.float32
                                ),
                                # Fuel penalty per step.
                                "reward_fuel": Unbounded(
                                    shape=(self.n_agents, 1), device=device, dtype=torch.float32
                                ),
                                # Delta-v per step: linear (m/s) and
                                # angular (rad/s).
                                "delta_v": Unbounded(
                                    shape=(self.n_agents, 1), device=device, dtype=torch.float32
                                ),
                                "angular_delta_v": Unbounded(
                                    shape=(self.n_agents, 1), device=device, dtype=torch.float32
                                ),
                            },
                            device=device,
                            shape=(self.n_agents,),
                        ),
                    },
                    device=device,
                    shape=(self.n_agents,),
                )
            },
            device=device,
        )
        action_spec = Composite(
            {
                self.group: Composite(
                    {
                        "action": Bounded(
                            low=-1.0,
                            high=1.0,
                            shape=(self.n_agents, self.action_dim),
                            device=device,
                            dtype=torch.float32,
                        )
                    },
                    device=device,
                    shape=(self.n_agents,),
                )
            },
            device=device,
        )
        reward_spec = Composite(
            {
                self.group: Composite(
                    {
                        "reward": Unbounded(
                            shape=(self.n_agents, 1), device=device, dtype=torch.float32
                        )
                    },
                    device=device,
                    shape=(self.n_agents,),
                )
            },
            device=device,
        )
        done_leaf = Categorical(n=2, shape=(1,), device=device, dtype=torch.bool)
        done_spec = Composite(
            {
                "done": done_leaf,
                "terminated": done_leaf.clone(),
            },
            device=device,
        )
        self.full_observation_spec_unbatched = observation_spec
        self.full_action_spec_unbatched = action_spec
        self.full_reward_spec_unbatched = reward_spec
        self.full_done_spec_unbatched = done_spec

    def _allocate_state(self) -> None:
        b = self.batch_size[0]
        d = self.device
        self.pos = torch.zeros((b, self.n_agents, 3), dtype=torch.float32, device=d)
        self.vel = torch.zeros((b, self.n_agents, 3), dtype=torch.float32, device=d)
        # Attitude state: unit quaternion [w, x, y, z] and body rate. Allocated also
        # when attitude is off (identity and zero).
        self.quat = self._identity_quat((b, self.n_agents))
        self.ang_vel = torch.zeros((b, self.n_agents, 3), dtype=torch.float32, device=d)
        self.on_goal_latched = torch.zeros((b, self.n_agents, 1), dtype=torch.bool, device=d)
        self.goals = torch.zeros((b, self.n_agents, 3), dtype=torch.float32, device=d)
        self.target_center = torch.zeros((b, 3), dtype=torch.float32, device=d)
        self.step_count = torch.zeros((b, 1), dtype=torch.long, device=d)
        self.prev_distance = torch.zeros((b, self.n_agents, 1), dtype=torch.float32, device=d)
        self.prev_capture_potential = torch.zeros(
            (b, self.n_agents, 1), dtype=torch.float32, device=d
        )
        self.last_goal_bonus_awarded = torch.zeros((b, self.n_agents, 1), dtype=torch.bool, device=d)
        self.last_reward_shaped = torch.zeros((b, self.n_agents, 1), dtype=torch.float32, device=d)
        self.last_reward_progress = torch.zeros((b, self.n_agents, 1), dtype=torch.float32, device=d)
        self.last_reward_time = torch.zeros((b, self.n_agents, 1), dtype=torch.float32, device=d)
        self.last_reward_velocity = torch.zeros((b, self.n_agents, 1), dtype=torch.float32, device=d)
        self.last_reward_capture = torch.zeros((b, self.n_agents, 1), dtype=torch.float32, device=d)
        self.last_reward_bonus = torch.zeros((b, self.n_agents, 1), dtype=torch.float32, device=d)
        self.last_reward_collision = torch.zeros((b, self.n_agents, 1), dtype=torch.float32, device=d)
        self.last_reward_attitude = torch.zeros((b, self.n_agents, 1), dtype=torch.float32, device=d)
        self.last_reward_agent_collision = torch.zeros(
            (b, self.n_agents, 1), dtype=torch.float32, device=d
        )
        self.last_cbf_intervention = torch.zeros(
            (b, self.n_agents, 1), dtype=torch.float32, device=d
        )
        self.last_cbf_infeasible = torch.zeros(
            (b, self.n_agents, 1), dtype=torch.float32, device=d
        )
        self.last_cbf_relaxed = torch.zeros(
            (b, self.n_agents, 1), dtype=torch.float32, device=d
        )
        self.last_reward_cbf = torch.zeros((b, self.n_agents, 1), dtype=torch.float32, device=d)
        self.last_reward_fuel = torch.zeros((b, self.n_agents, 1), dtype=torch.float32, device=d)
        # Delta-v per step: integral of the applied (post-filter) acceleration
        # magnitude over the substeps. Linear in m/s, angular in rad/s.
        self.last_delta_v = torch.zeros((b, self.n_agents, 1), dtype=torch.float32, device=d)
        self.last_angular_delta_v = torch.zeros(
            (b, self.n_agents, 1), dtype=torch.float32, device=d
        )
        # Sensor-noise state. Allocated also when the flag is off.
        # Per-episode biases (own channel and one per neighbour slot).
        # Actuator-uncertainty state: scale error and misalignment rotation per
        # episode. Identity and zero when the feature is off.
        # Navigation estimate for the filter. Set in _build_observation each step.
        self._nav_pos = torch.zeros((b, self.n_agents, 3), dtype=torch.float32, device=d)
        self._nav_vel = torch.zeros((b, self.n_agents, 3), dtype=torch.float32, device=d)
        self._nav_ang_vel = torch.zeros((b, self.n_agents, 3), dtype=torch.float32, device=d)
        self._act_scale_lin = torch.zeros((b, self.n_agents, 3), dtype=torch.float32, device=d)
        self._act_scale_ang = torch.zeros((b, self.n_agents, 3), dtype=torch.float32, device=d)
        eye = torch.eye(3, dtype=torch.float32, device=d).expand(b, self.n_agents, 3, 3)
        self._act_rot_lin = eye.clone()
        self._act_rot_ang = eye.clone()
        self._sensor_bias_pos = torch.zeros((b, self.n_agents, 3), dtype=torch.float32, device=d)
        self._sensor_bias_vel = torch.zeros((b, self.n_agents, 3), dtype=torch.float32, device=d)
        self._sensor_bias_att = torch.zeros((b, self.n_agents, 3), dtype=torch.float32, device=d)
        self._sensor_bias_ang_vel = torch.zeros(
            (b, self.n_agents, 3), dtype=torch.float32, device=d
        )
        _K = max(int(self.neighbor_obs_count), 1)
        self._sensor_bias_nbr_pos = torch.zeros(
            (b, self.n_agents, _K, 3), dtype=torch.float32, device=d
        )
        self._sensor_bias_nbr_vel = torch.zeros(
            (b, self.n_agents, _K, 3), dtype=torch.float32, device=d
        )
        # Last delivered reading for each channel, held on a dropout step.
        self._sensor_hold_pos = torch.zeros((b, self.n_agents, 3), dtype=torch.float32, device=d)
        self._sensor_hold_vel = torch.zeros((b, self.n_agents, 3), dtype=torch.float32, device=d)
        self._sensor_hold_att = torch.zeros((b, self.n_agents, 3), dtype=torch.float32, device=d)
        self._sensor_hold_nbr_pos = torch.zeros(
            (b, self.n_agents, _K, 3), dtype=torch.float32, device=d
        )
        self._sensor_hold_nbr_vel = torch.zeros(
            (b, self.n_agents, _K, 3), dtype=torch.float32, device=d
        )

    def _set_seed(self, seed: Optional[int]) -> None:
        self._seed = int(0 if seed is None else seed)
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self._seed)

    # Reset (uses the TorchRL partial-reset mask).
    def _reset(self, tensordict: Optional[TensorDictBase] = None) -> TensorDictBase:
        reset_mask = self._resolve_reset_mask(tensordict)
        m_agent = reset_mask.view(self.num_envs, 1, 1)
        m_env = reset_mask.view(self.num_envs, 1)

        new_pos = self._sample_initial_positions()
        new_target = self._rand_uniform(
            (-self.target_center_jitter, self.target_center_jitter),
            (self.num_envs, 3),
        )

        self.pos = torch.where(m_agent, new_pos, self.pos)
        self.vel = torch.where(m_agent, torch.zeros_like(self.vel), self.vel)
        new_quat, new_ang_vel = self._sample_initial_attitude()
        self.quat = torch.where(m_agent, new_quat, self.quat)
        self.ang_vel = torch.where(m_agent, new_ang_vel, self.ang_vel)
        self.target_center = torch.where(m_env, new_target, self.target_center)
        # Goals depend only on target_center, so other envs do not change.
        self.goals = self._compute_goals_from_target()
        self.on_goal_latched = torch.where(
            m_agent, torch.zeros_like(self.on_goal_latched), self.on_goal_latched
        )
        # New per-episode bias for each reset env. The hold buffers are set to the
        # true initial state, as for a sensor that has just acquired lock.
        m_agent_nbr = m_agent.unsqueeze(2)  # [B,1,1,1], broadcasts over neighbour slots
        if self.enable_sensor_noise:
            true_rel_goal = self._relative_goal_vector()
            true_att_vec, _ = self._attitude_error_terms()
            self._sensor_hold_pos = torch.where(m_agent, true_rel_goal, self._sensor_hold_pos)
            self._sensor_hold_vel = torch.where(m_agent, self.vel, self._sensor_hold_vel)
            self._sensor_hold_att = torch.where(m_agent, true_att_vec, self._sensor_hold_att)
            nbr_pos, nbr_vel = self._nearest_neighbor_features(self.neighbor_obs_count)
            self._sensor_hold_nbr_pos = torch.where(
                m_agent_nbr, nbr_pos, self._sensor_hold_nbr_pos
            )
            self._sensor_hold_nbr_vel = torch.where(
                m_agent_nbr, nbr_vel, self._sensor_hold_nbr_vel
            )
        # Actuator errors are properties of the vehicle, so they are drawn once per
        # episode.
        if self.enable_actuator_uncertainty:
            self._act_scale_lin = torch.where(
                m_agent,
                self.actuator_lin_scale_bias_std * self._rand_normal(self._act_scale_lin.shape),
                self._act_scale_lin,
            )
            self._act_scale_ang = torch.where(
                m_agent,
                self.actuator_ang_scale_bias_std * self._rand_normal(self._act_scale_ang.shape),
                self._act_scale_ang,
            )
            new_lin = self._random_small_rotation(self.actuator_lin_align_std)
            new_ang = self._random_small_rotation(self.actuator_ang_align_std)
            m_rot = m_agent.unsqueeze(-1)  # [B,N,1,1] over the 3x3 block
            self._act_rot_lin = torch.where(m_rot, new_lin, self._act_rot_lin)
            self._act_rot_ang = torch.where(m_rot, new_ang, self._act_rot_ang)

        self._sensor_bias_pos = torch.where(
            m_agent,
            self.sensor_pos_bias_std * self._rand_normal(self._sensor_bias_pos.shape),
            self._sensor_bias_pos,
        )
        self._sensor_bias_vel = torch.where(
            m_agent,
            self.sensor_vel_bias_std * self._rand_normal(self._sensor_bias_vel.shape),
            self._sensor_bias_vel,
        )
        self._sensor_bias_att = torch.where(
            m_agent,
            self.sensor_att_bias_std * self._rand_normal(self._sensor_bias_att.shape),
            self._sensor_bias_att,
        )
        self._sensor_bias_ang_vel = torch.where(
            m_agent,
            self.sensor_angvel_bias_std * self._rand_normal(self._sensor_bias_ang_vel.shape),
            self._sensor_bias_ang_vel,
        )
        self._sensor_bias_nbr_pos = torch.where(
            m_agent_nbr,
            self.sensor_pos_bias_std * self._rand_normal(self._sensor_bias_nbr_pos.shape),
            self._sensor_bias_nbr_pos,
        )
        self._sensor_bias_nbr_vel = torch.where(
            m_agent_nbr,
            self.sensor_vel_bias_std * self._rand_normal(self._sensor_bias_nbr_vel.shape),
            self._sensor_bias_nbr_vel,
        )
        self.step_count = torch.where(m_env, torch.zeros_like(self.step_count), self.step_count)
        self.prev_distance = torch.where(m_agent, self._current_distance(), self.prev_distance)
        self.prev_capture_potential = torch.where(
            m_agent, self._capture_potential(), self.prev_capture_potential
        )
        self.last_goal_bonus_awarded = torch.where(
            m_agent, torch.zeros_like(self.last_goal_bonus_awarded), self.last_goal_bonus_awarded
        )
        self.last_reward_shaped = torch.where(
            m_agent, torch.zeros_like(self.last_reward_shaped), self.last_reward_shaped
        )
        self.last_reward_progress = torch.where(
            m_agent, torch.zeros_like(self.last_reward_progress), self.last_reward_progress
        )
        self.last_reward_time = torch.where(
            m_agent, torch.zeros_like(self.last_reward_time), self.last_reward_time
        )
        self.last_reward_velocity = torch.where(
            m_agent, torch.zeros_like(self.last_reward_velocity), self.last_reward_velocity
        )
        self.last_reward_capture = torch.where(
            m_agent, torch.zeros_like(self.last_reward_capture), self.last_reward_capture
        )
        self.last_reward_bonus = torch.where(
            m_agent, torch.zeros_like(self.last_reward_bonus), self.last_reward_bonus
        )
        self.last_reward_collision = torch.where(
            m_agent, torch.zeros_like(self.last_reward_collision), self.last_reward_collision
        )
        self.last_reward_attitude = torch.where(
            m_agent, torch.zeros_like(self.last_reward_attitude), self.last_reward_attitude
        )
        self.last_reward_agent_collision = torch.where(
            m_agent,
            torch.zeros_like(self.last_reward_agent_collision),
            self.last_reward_agent_collision,
        )
        self.last_cbf_intervention = torch.where(
            m_agent, torch.zeros_like(self.last_cbf_intervention), self.last_cbf_intervention
        )
        self.last_cbf_infeasible = torch.where(
            m_agent, torch.zeros_like(self.last_cbf_infeasible), self.last_cbf_infeasible
        )
        self.last_cbf_relaxed = torch.where(
            m_agent, torch.zeros_like(self.last_cbf_relaxed), self.last_cbf_relaxed
        )
        self.last_reward_fuel = torch.where(
            m_agent, torch.zeros_like(self.last_reward_fuel), self.last_reward_fuel
        )
        self.last_reward_cbf = torch.where(
            m_agent, torch.zeros_like(self.last_reward_cbf), self.last_reward_cbf
        )
        self.last_delta_v = torch.where(
            m_agent, torch.zeros_like(self.last_delta_v), self.last_delta_v
        )
        self.last_angular_delta_v = torch.where(
            m_agent, torch.zeros_like(self.last_angular_delta_v), self.last_angular_delta_v
        )
        return self._build_state_tensordict(reward=None)

    def _resolve_reset_mask(self, tensordict: Optional[TensorDictBase]) -> torch.Tensor:
        if tensordict is not None and "_reset" in tensordict.keys():
            return tensordict.get("_reset").reshape(self.num_envs).to(torch.bool)
        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    # Step.
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict[(self.group, "action")]
        # The policy action is the nominal for the control step. The QP is solved
        # inside _apply_dynamics.
        if self.enable_cbf:
            # Update the target position in the filter before the projection.
            self.safety_filter.target_center = self.target_center
            self.safety_filter.target_size = self.target_size
        self._apply_dynamics(action)
        self.step_count = self.step_count + 1
        reward = self._compute_reward()
        return self._build_state_tensordict(reward=reward)

    def _apply_dynamics(self, action: torch.Tensor) -> None:
        """Clohessy-Wiltshire translational dynamics.

        action[..., :3] is the linear acceleration in [-1, 1], scaled by
        linear_accel_scale. Docked agents are frozen. With enable_cbf=True the
        action goes through the safety filter first.
        """
        omega = self.mean_motion

        dtype = self.pos.dtype
        device = self.pos.device
        A = torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [3 * omega**2, 0.0, 0.0, 0.0, 2 * omega, 0.0],
                [0.0, 0.0, 0.0, -2 * omega, 0.0, 0.0],
                [0.0, 0.0, -omega**2, 0.0, 0.0, 0.0],
            ],
            dtype=dtype,
            device=device,
        )
        B = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=dtype,
            device=device,
        )

        action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        active_mask = ~self.on_goal_latched  # [B, N, 1]
        sub_dt = self.dt / float(self.dynamics_substeps)

        if self.enable_cbf:
            self._apply_dynamics_filtered(action, active_mask, A, B, sub_dt)
            return

        linear_acc = action[..., :3] * self.linear_accel_scale
        # Apply the actuator model before the delta-v and the integration.
        linear_acc = self._apply_actuator_error(linear_acc, angular=False)
        linear_acc = linear_acc * active_mask.float()

        x_k = torch.cat([self.pos, self.vel], dim=-1)  # [..., 6]
        for _ in range(self.dynamics_substeps):
            x_dot = torch.matmul(x_k, A.T) + torch.matmul(linear_acc, B.T)
            x_k = x_k + sub_dt * x_dot
            x_k = torch.nan_to_num(x_k, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)

        pos_next = x_k[..., 0:3].clamp(-1e5, 1e5)
        vel_next = x_k[..., 3:6].clamp(-1e4, 1e4)
        self.pos = torch.where(active_mask, pos_next, self.pos)
        self.vel = torch.where(active_mask, vel_next, torch.zeros_like(self.vel))
        # The acceleration is constant over the step, so delta-v = ||a|| * dt.
        self.last_delta_v = linear_acc.norm(dim=-1, keepdim=True) * self.dt

        if self.enable_attitude:
            self._apply_attitude_dynamics(action, active_mask)
        else:
            self.last_angular_delta_v = torch.zeros_like(self.last_angular_delta_v)

    def _apply_dynamics_filtered(
        self,
        action: torch.Tensor,
        active_mask: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        sub_dt: float,
    ) -> None:
        """Substep integration with the filter solved every
        cbf_filter_substep_interval substeps. The safe action is held between
        solves. A pass-through filter gives the unfiltered result.
        last_cbf_intervention is the substep mean of ||u_safe - u_nom||.
        """
        pos, vel = self.pos, self.vel
        quat, ang_vel = self.quat, self.ang_vel
        active = active_mask.float()
        intervention_sum = torch.zeros_like(self.last_cbf_intervention)
        infeasible_sum = torch.zeros_like(self.last_cbf_infeasible)
        relaxed_sum = torch.zeros_like(self.last_cbf_relaxed)
        delta_v_sum = torch.zeros_like(self.last_delta_v)
        ang_delta_v_sum = torch.zeros_like(self.last_angular_delta_v)
        # Docked agents do not move, so the filter treats them as obstacles.
        static_mask = self.on_goal_latched.squeeze(-1)
        interval = self.cbf_filter_substep_interval
        safe_action = action
        intervention = torch.zeros_like(self.last_cbf_intervention)
        infeasible_step: Optional[torch.Tensor] = None
        relaxed_step: Optional[torch.Tensor] = None

        for sub in range(self.dynamics_substeps):
            # Solve the QP only at the interval; hold the result in between.
            if sub % interval == 0:
                # State given to the filter: the true state by default, or the navigation
                # estimate with cbf_use_sensed_state. The integration always uses the true
                # state.
                if self.cbf_use_sensed_state and self.enable_sensor_noise:
                    f_pos, f_vel, f_ang_vel = (
                        self._nav_pos, self._nav_vel, self._nav_ang_vel
                    )
                else:
                    f_pos, f_vel, f_ang_vel = pos, vel, ang_vel
                safe_action, intervention = self.safety_filter.filter_action(
                    pos=f_pos,
                    vel=f_vel,
                    action=action,
                    quat=quat if self.enable_attitude else None,
                    ang_vel=f_ang_vel if self.enable_attitude else None,
                    static_mask=static_mask,
                )
                safe_action = torch.nan_to_num(
                    safe_action, nan=0.0, posinf=0.0, neginf=0.0
                ).clamp(-1.0, 1.0)
                infeasible_step = getattr(self.safety_filter, "last_infeasible_mask", None)
                relaxed_step = getattr(self.safety_filter, "last_relaxed_mask", None)
            intervention_sum = intervention_sum + intervention
            if infeasible_step is not None:
                infeasible_sum = infeasible_sum + infeasible_step.float()
            if relaxed_step is not None:
                relaxed_sum = relaxed_sum + relaxed_step.float()

            linear_acc = safe_action[..., :3] * self.linear_accel_scale
            # The plant applies this, not the filter output.
            linear_acc = self._apply_actuator_error(linear_acc, angular=False) * active
            # Delta-v: integrate the applied (filtered) acceleration over the substeps.
            delta_v_sum = delta_v_sum + linear_acc.norm(dim=-1, keepdim=True) * sub_dt
            x_k = torch.cat([pos, vel], dim=-1)
            x_dot = torch.matmul(x_k, A.T) + torch.matmul(linear_acc, B.T)
            x_k = x_k + sub_dt * x_dot
            x_k = torch.nan_to_num(x_k, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
            pos = x_k[..., 0:3]
            vel = x_k[..., 3:6]

            if self.enable_attitude:
                if safe_action.shape[-1] < 6:
                    ang_cmd = torch.zeros_like(ang_vel)
                else:
                    ang_cmd = safe_action[..., 3:6] * self.angular_accel_scale
                    ang_cmd = self._apply_actuator_error(ang_cmd, angular=True) * active
                ang_delta_v_sum = (
                    ang_delta_v_sum + ang_cmd.norm(dim=-1, keepdim=True) * sub_dt
                )
                ang_vel = ang_vel + sub_dt * ang_cmd
                omega_quat = torch.cat(
                    [torch.zeros_like(ang_vel[..., :1]), ang_vel], dim=-1
                )
                quat = quat + sub_dt * 0.5 * self._quat_mul(quat, omega_quat)
                quat = torch.nan_to_num(quat, nan=0.0, posinf=1.0, neginf=-1.0)
                quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        pos_next = pos.clamp(-1e5, 1e5)
        vel_next = vel.clamp(-1e4, 1e4)
        self.pos = torch.where(active_mask, pos_next, self.pos)
        self.vel = torch.where(active_mask, vel_next, torch.zeros_like(self.vel))
        if self.enable_attitude:
            ang_vel = torch.nan_to_num(ang_vel, nan=0.0, posinf=1e3, neginf=-1e3).clamp(
                -1e3, 1e3
            )
            self.quat = torch.where(active_mask, quat, self.quat)
            self.ang_vel = torch.where(active_mask, ang_vel, torch.zeros_like(self.ang_vel))
        self.last_cbf_intervention = (
            intervention_sum / float(self.dynamics_substeps)
        ) * active
        self.last_cbf_infeasible = (
            infeasible_sum / float(self.dynamics_substeps)
        ) * active
        self.last_cbf_relaxed = (relaxed_sum / float(self.dynamics_substeps)) * active
        self.last_delta_v = delta_v_sum
        self.last_angular_delta_v = ang_delta_v_sum

    def _apply_attitude_dynamics(self, action: torch.Tensor, active_mask: torch.Tensor) -> None:
        """Quaternion attitude kinematics with commanded angular acceleration.

        action[..., 3:6] is the body angular acceleration in [-1, 1], scaled by
        angular_accel_scale. Gyroscopic coupling is ignored (small at these rates).
        q_dot = 0.5 * q (x) [0, omega_body], normalised at each substep.
        Docked agents are frozen with zero rate.
        """
        if action.shape[-1] < 6:
            ang_cmd = torch.zeros_like(self.ang_vel)
        else:
            ang_cmd = action[..., 3:6] * self.angular_accel_scale
        ang_cmd = self._apply_actuator_error(ang_cmd, angular=True)
        ang_cmd = ang_cmd * active_mask.float()

        sub_dt = self.dt / float(self.dynamics_substeps)
        q = self.quat
        w = self.ang_vel
        for _ in range(self.dynamics_substeps):
            w = w + sub_dt * ang_cmd
            omega_quat = torch.cat([torch.zeros_like(w[..., :1]), w], dim=-1)
            q = q + sub_dt * 0.5 * self._quat_mul(q, omega_quat)
            q = torch.nan_to_num(q, nan=0.0, posinf=1.0, neginf=-1.0)
            q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        w = torch.nan_to_num(w, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
        self.quat = torch.where(active_mask, q, self.quat)
        self.ang_vel = torch.where(active_mask, w, torch.zeros_like(self.ang_vel))
        # Angular delta-v: ||alpha|| * dt.
        self.last_angular_delta_v = ang_cmd.norm(dim=-1, keepdim=True) * self.dt

    # Reward.
    def _compute_reward(self) -> torch.Tensor:
        distance = self._current_distance()
        active = (~self.on_goal_latched).float()

        # Potential-based progress: reward for a smaller distance to the goal.
        progress = self.prev_distance - distance
        vel = torch.linalg.norm(self.vel, dim=-1, keepdim=True)

        # Potential-based arrive-and-stop shaping. Phi = near * slow is 1 at
        # capture. The reward is the change in Phi, so hovering gives nothing
        # (Ng et al.).
        capture_potential = self._capture_potential(distance=distance, speed=vel)
        capture_shaping = self.reward_capture_scale * (
            capture_potential - self.prev_capture_potential
        )

        # Shaping terms, kept separately for the reward breakdown. Their sum is
        # `shaped`.
        progress_term = (self.reward_progress_scale * progress) * active
        time_term = (-self.reward_time_penalty) * active
        velocity_term = (-self.reward_vel_scale * vel**2) * active
        capture_term = capture_shaping * active
        shaped = progress_term + time_term + velocity_term + capture_term

        # Attitude penalty -alpha^T Q alpha, alpha = [2*q_err_vec, omega_err],
        # q_err = q_des (x) q^-1.
        if self.enable_attitude:
            att_vec, ang_err = self._attitude_error_terms()
            attitude_penalty = -(
                self.reward_att_scale * (att_vec**2).sum(dim=-1, keepdim=True)
                + self.reward_angvel_scale * (ang_err**2).sum(dim=-1, keepdim=True)
            ) * active
        else:
            attitude_penalty = torch.zeros_like(shaped)

        captured = self._capture_condition(distance, vel)
        newly_on_goal = captured & (~self.on_goal_latched)
        w_depth = self.reward_goal_bonus_depth_weight
        if w_depth > 0.0:
            depth = (1.0 - distance / max(self.goal_tolerance, 1e-9)).clamp(0.0, 1.0)
            bonus_scale = 1.0 + w_depth * depth
        else:
            bonus_scale = torch.ones_like(distance)
        bonus = newly_on_goal.float() * self.reward_goal_bonus * bonus_scale

        collision_progress, _, collision_penetration = self._collision_progress_and_flag()
        collision_penalty = -self.collision_penalty_scale * torch.sin(
            0.5 * torch.pi * collision_progress
        )
        # One-time penalty when a target crash ends the episode.
        if self.collision_terminal_penalty != 0.0:
            target_terminal = (
                collision_penetration >= self.collision_smoothing_distance
            ).float()
            collision_penalty = collision_penalty - self.collision_terminal_penalty * target_terminal

        # Inter-agent collision penalty.
        if self.enable_agent_collisions:
            (
                agent_coll_progress,
                agent_coll,
                agent_coll_penetration,
                _,
            ) = self._agent_collision_progress_and_flag()
            agent_collision_penalty = -self.agent_collision_penalty_scale * torch.sin(
                0.5 * torch.pi * agent_coll_progress
            )
            if self.agent_collision_terminal_penalty != 0.0:
                agent_terminal = self._agent_collision_terminal_mask(
                    agent_coll, agent_coll_penetration
                ).float()
                agent_collision_penalty = (
                    agent_collision_penalty
                    - self.agent_collision_terminal_penalty * agent_terminal
                )
        else:
            agent_collision_penalty = torch.zeros_like(collision_penalty)

        # Filter intervention penalty c_int * ||u_safe - u_nom||^2.
        if self.enable_cbf and self.cbf_intervention_penalty_scale != 0.0:
            cbf_penalty = (
                -self.cbf_intervention_penalty_scale
                * self.last_cbf_intervention**2
                * active
            )
        else:
            cbf_penalty = torch.zeros_like(collision_penalty)

        # Delta-v cost of the applied (post-filter) control, including the filter
        # correction. Docked agents have zero delta-v.
        if self.reward_fuel_scale != 0.0 or self.reward_fuel_ang_scale != 0.0:
            fuel_penalty = -(
                self.reward_fuel_scale * self.last_delta_v
                + self.reward_fuel_ang_scale * self.last_angular_delta_v
            ) * active
        else:
            fuel_penalty = torch.zeros_like(collision_penalty)

        reward = (
            shaped
            + attitude_penalty
            + bonus
            + collision_penalty
            + agent_collision_penalty
            + cbf_penalty
            + fuel_penalty
        )
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=-1e6)

        # Update the potentials for the next step.
        self.prev_distance = distance
        self.prev_capture_potential = capture_potential
        self.last_goal_bonus_awarded = newly_on_goal
        self.last_reward_shaped = torch.nan_to_num(shaped, nan=0.0, posinf=0.0, neginf=-1e6)
        self.last_reward_progress = torch.nan_to_num(
            progress_term, nan=0.0, posinf=0.0, neginf=-1e6
        )
        self.last_reward_time = torch.nan_to_num(time_term, nan=0.0, posinf=0.0, neginf=-1e6)
        self.last_reward_velocity = torch.nan_to_num(
            velocity_term, nan=0.0, posinf=0.0, neginf=-1e6
        )
        self.last_reward_capture = torch.nan_to_num(
            capture_term, nan=0.0, posinf=0.0, neginf=-1e6
        )
        self.last_reward_bonus = bonus
        self.last_reward_collision = torch.nan_to_num(
            collision_penalty, nan=0.0, posinf=0.0, neginf=-1e6
        )
        self.last_reward_attitude = torch.nan_to_num(
            attitude_penalty, nan=0.0, posinf=0.0, neginf=-1e6
        )
        self.last_reward_agent_collision = torch.nan_to_num(
            agent_collision_penalty, nan=0.0, posinf=0.0, neginf=-1e6
        )
        self.last_reward_cbf = torch.nan_to_num(cbf_penalty, nan=0.0, posinf=0.0, neginf=-1e6)
        self.last_reward_fuel = torch.nan_to_num(fuel_penalty, nan=0.0, posinf=0.0, neginf=-1e6)
        return reward

    # State and observation.
    def _build_state_tensordict(self, *, reward: Optional[torch.Tensor]) -> TensorDictBase:
        distance = self._current_distance()
        speed = self._current_speed()
        _, collision, penetration = self._collision_progress_and_flag()
        instant_on_goal = self._capture_condition(distance, speed)
        self.on_goal_latched = torch.logical_or(self.on_goal_latched, instant_on_goal)
        on_goal = self.on_goal_latched
        complete_success = on_goal.all(dim=1, keepdim=True).all(dim=2)
        complete_success_agents = complete_success.unsqueeze(1).expand(
            self.num_envs, self.n_agents, 1
        )
        timeout = self.step_count >= self.max_steps
        collision_terminal = penetration >= self.collision_smoothing_distance
        collision_terminal_any = collision_terminal.any(dim=1)

        # Inter-agent collision metrics (zero when the switch is off).
        (
            _,
            agent_collision,
            agent_penetration,
            min_agent_sep,
        ) = self._agent_collision_progress_and_flag()
        agent_collision_terminal = self._agent_collision_terminal_mask(
            agent_collision, agent_penetration
        )
        agent_collision_terminal_any = agent_collision_terminal.any(dim=1)

        finite_mask = (
            torch.isfinite(self.pos).all(dim=(1, 2))
            & torch.isfinite(self.vel).all(dim=(1, 2))
            & torch.isfinite(self.quat).all(dim=(1, 2))
            & torch.isfinite(self.ang_vel).all(dim=(1, 2))
        ).unsqueeze(-1)
        numerical_error = ~finite_mask.unsqueeze(1).expand(self.num_envs, self.n_agents, 1)
        numerical_error_any = numerical_error.any(dim=1)
        done = torch.logical_or(
            torch.logical_or(complete_success, timeout),
            torch.logical_or(
                torch.logical_or(collision_terminal_any, agent_collision_terminal_any),
                numerical_error_any,
            ),
        )

        obs = self._build_observation()
        info_td = TensorDict(
            {
                "distance_to_goal": distance,
                "relative_speed": speed,
                "on_goal": on_goal,
                "complete_success": complete_success_agents,
                "goal_bonus_awarded": self.last_goal_bonus_awarded,
                "reward_shaped": self.last_reward_shaped,
                "reward_progress": self.last_reward_progress,
                "reward_time": self.last_reward_time,
                "reward_velocity": self.last_reward_velocity,
                "reward_capture": self.last_reward_capture,
                "reward_bonus": self.last_reward_bonus,
                "reward_collision": self.last_reward_collision,
                "target_center": self.target_center.unsqueeze(1).expand(
                    self.num_envs, self.n_agents, 3
                ),
                "target_size": self.target_size.view(1, 1, 3).expand(
                    self.num_envs, self.n_agents, 3
                ),
                "agent_position": self.pos,
                "goal_position": self.goals,
                "collision": collision,
                "collision_penetration": penetration,
                "collision_terminal": collision_terminal,
                "numerical_error": numerical_error,
                "agent_size": self.agent_size.view(1, 1, 3).expand(
                    self.num_envs, self.n_agents, 3
                ),
                "quaternion": self.quat,
                "angular_velocity": self.ang_vel,
                "attitude_error_angle": self._attitude_error_angle(),
                "reward_attitude": self.last_reward_attitude,
                "agent_collision": agent_collision,
                "agent_collision_penetration": agent_penetration,
                "agent_collision_terminal": agent_collision_terminal,
                "min_agent_separation": min_agent_sep,
                "reward_agent_collision": self.last_reward_agent_collision,
                "cbf_intervention": self.last_cbf_intervention,
                "reward_cbf_intervention": self.last_reward_cbf,
                "cbf_infeasible": self.last_cbf_infeasible,
                "cbf_relaxed": self.last_cbf_relaxed,
                "reward_fuel": self.last_reward_fuel,
                "delta_v": self.last_delta_v,
                "angular_delta_v": self.last_angular_delta_v,
            },
            batch_size=torch.Size([self.num_envs, self.n_agents]),
            device=self.device,
        )
        group_payload = {"observation": obs, "info": info_td}
        if reward is not None:
            group_payload["reward"] = reward
        group_td = TensorDict(
            group_payload,
            batch_size=torch.Size([self.num_envs, self.n_agents]),
            device=self.device,
        )
        out = TensorDict(
            {
                "done": done,
                "terminated": done.clone(),
                self.group: group_td,
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        return out

    def _build_observation(self) -> torch.Tensor:
        """Per-agent observation (see the class docstring for the layout)."""
        # Read the sensor model once per step. The terminal feature reuses
        # sensed_rel_goal.
        sensed_rel_goal, sensed_vel, sensed_att_vec, sensed_ang_err = self._sensed_own_state()
        # Keep this navigation estimate for the filter (cbf_use_sensed_state).
        # p_hat = goal - rel_goal.
        self._nav_pos = self.goals - sensed_rel_goal
        self._nav_vel = sensed_vel
        self._nav_ang_vel = sensed_ang_err
        rel_goal = sensed_rel_goal / self.obs_pos_scale
        vel = sensed_vel / self.obs_vel_scale
        features = [rel_goal, vel]
        if self.enable_attitude:
            features.extend([sensed_att_vec, sensed_ang_err])
        if self.enable_neighbor_obs:
            nn_rel_pos, nn_rel_vel = self._nearest_neighbor_features(self.neighbor_obs_count)
            nn_rel_pos, nn_rel_vel = self._sensed_neighbor_state(nn_rel_pos, nn_rel_vel)
            # Blocks [pos_j, vel_j], nearest first.
            for j in range(self.neighbor_obs_count):
                features.extend(
                    [
                        nn_rel_pos[:, :, j] / self.obs_neighbor_pos_scale,
                        nn_rel_vel[:, :, j] / self.obs_neighbor_vel_scale,
                    ]
                )
        if self.enable_safety_obs:
            # True state, not the estimate: these features show what the filter sees.
            features.append(self._safety_obs_block())
        if self.enable_terminal_obs:
            features.append(torch.tanh(sensed_rel_goal / self.terminal_obs_scale))
        obs = torch.cat(features, dim=-1)
        if obs.shape[-1] < self.obs_dim:
            pad = torch.zeros(
                (*obs.shape[:-1], self.obs_dim - obs.shape[-1]),
                device=self.device,
                dtype=torch.float32,
            )
            obs = torch.cat([obs, pad], dim=-1)
        elif obs.shape[-1] > self.obs_dim:
            obs = obs[..., : self.obs_dim]
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
        return obs

    # Helpers.
    def _rand_uniform(self, bounds: tuple[float, float], shape: tuple[int, ...]) -> torch.Tensor:
        low, high = bounds
        if high <= low:
            return torch.full(shape, float(low), device=self.device, dtype=torch.float32)
        return low + (high - low) * torch.rand(shape, generator=self.rng, device=self.device)

    def _rand_normal(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=self.rng, device=self.device)

    def _random_small_rotation(self, angle_std: float) -> torch.Tensor:
        """Rotation per agent by a small random angle about a uniform random axis
        (Rodrigues' formula).
        """
        shape = (self.num_envs, self.n_agents, 3)
        axis = self._rand_normal(shape)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        ang = angle_std * self._rand_normal((self.num_envs, self.n_agents, 1))
        kx, ky, kz = axis[..., 0], axis[..., 1], axis[..., 2]
        zero = torch.zeros_like(kx)
        K = torch.stack(
            [torch.stack([zero, -kz, ky], dim=-1),
             torch.stack([kz, zero, -kx], dim=-1),
             torch.stack([-ky, kx, zero], dim=-1)], dim=-2
        )  # [B,N,3,3]
        eye = torch.eye(3, dtype=K.dtype, device=K.device).expand_as(K)
        sin = torch.sin(ang).unsqueeze(-1)
        cos = torch.cos(ang).unsqueeze(-1)
        return eye + sin * K + (1.0 - cos) * (K @ K)

    def _apply_actuator_error(
        self, cmd: torch.Tensor, *, angular: bool
    ) -> torch.Tensor:
        """Apply the actuator model to the post-filter command."""
        if not self.enable_actuator_uncertainty:
            return cmd
        if angular:
            bias, noise_std, rot = (
                self._act_scale_ang, self.actuator_ang_scale_noise_std, self._act_rot_ang
            )
        else:
            bias, noise_std, rot = (
                self._act_scale_lin, self.actuator_lin_scale_noise_std, self._act_rot_lin
            )
        gain = 1.0 + bias + noise_std * self._rand_normal(cmd.shape)
        return torch.matmul(rot, (gain * cmd).unsqueeze(-1)).squeeze(-1)

    def _sensor_dropout_mask(self, shape: tuple[int, ...]) -> torch.Tensor:
        """True where the vision data is lost this step and the last reading is held.
        All False when noise or dropout is off.
        """
        if not self.enable_sensor_noise or self.sensor_dropout_prob <= 0.0:
            return torch.zeros(shape, dtype=torch.bool, device=self.device)
        return (
            torch.rand(shape, generator=self.rng, device=self.device)
            < self.sensor_dropout_prob
        )

    def _sensed_own_state(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sensed (rel_goal, vel, att_vec, ang_vel_err), each [B, N, 3].

        When noise is off, this is the true state. It updates the bias and hold
        buffers, so call it only once per step.
        """
        true_rel_goal = self._relative_goal_vector()
        att_vec_true, ang_err_true = self._attitude_error_terms()
        if not self.enable_sensor_noise:
            return true_rel_goal, self.vel, att_vec_true, ang_err_true

        # Vision data: range-dependent noise and the episode bias.
        range_own = true_rel_goal.norm(dim=-1, keepdim=True)
        pos_std = self.sensor_pos_noise_floor + self.sensor_pos_noise_per_m * range_own
        vel_std = self.sensor_vel_noise_floor + self.sensor_vel_noise_per_m * range_own
        att_std = self.sensor_att_noise_floor + self.sensor_att_noise_per_m * range_own
        fresh_pos = (
            true_rel_goal + self._sensor_bias_pos + pos_std * self._rand_normal(true_rel_goal.shape)
        )
        fresh_vel = self.vel + self._sensor_bias_vel + vel_std * self._rand_normal(self.vel.shape)
        fresh_att = (
            att_vec_true + self._sensor_bias_att + att_std * self._rand_normal(att_vec_true.shape)
        )
        drop = self._sensor_dropout_mask((self.num_envs, self.n_agents, 1))
        sensed_pos = torch.where(drop, self._sensor_hold_pos, fresh_pos)
        sensed_vel = torch.where(drop, self._sensor_hold_vel, fresh_vel)
        sensed_att = torch.where(drop, self._sensor_hold_att, fresh_att)
        self._sensor_hold_pos = sensed_pos
        self._sensor_hold_vel = sensed_vel
        self._sensor_hold_att = sensed_att

        # Gyro rate: not affected by vision dropout.
        sensed_ang_vel = (
            ang_err_true
            + self._sensor_bias_ang_vel
            + self.sensor_angvel_noise_std * self._rand_normal(ang_err_true.shape)
        )
        return sensed_pos, sensed_vel, sensed_att, sensed_ang_vel

    def _sensed_neighbor_state(
        self, nn_rel_pos_true: torch.Tensor, nn_rel_vel_true: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sensed neighbour (position, velocity), [B, N, count, 3]. Same model as
        _sensed_own_state. Bias and hold belong to the neighbour slot, not to a
        specific agent.
        """
        if not self.enable_sensor_noise:
            return nn_rel_pos_true, nn_rel_vel_true
        range_nbr = nn_rel_pos_true.norm(dim=-1, keepdim=True)
        pos_std = self.sensor_pos_noise_floor + self.sensor_pos_noise_per_m * range_nbr
        vel_std = self.sensor_vel_noise_floor + self.sensor_vel_noise_per_m * range_nbr
        fresh_pos = (
            nn_rel_pos_true
            + self._sensor_bias_nbr_pos
            + pos_std * self._rand_normal(nn_rel_pos_true.shape)
        )
        fresh_vel = (
            nn_rel_vel_true
            + self._sensor_bias_nbr_vel
            + vel_std * self._rand_normal(nn_rel_vel_true.shape)
        )
        drop = self._sensor_dropout_mask(nn_rel_pos_true.shape[:-1] + (1,))
        sensed_pos = torch.where(drop, self._sensor_hold_nbr_pos, fresh_pos)
        sensed_vel = torch.where(drop, self._sensor_hold_nbr_vel, fresh_vel)
        self._sensor_hold_nbr_pos = sensed_pos
        self._sensor_hold_nbr_vel = sensed_vel
        return sensed_pos, sensed_vel

    def _sample_initial_positions(self) -> torch.Tensor:
        pos = self._draw_raw_positions()
        if self.min_spawn_separation > 0.0 and self.n_agents > 1:
            pos = self._enforce_min_separation(pos)
        pos = self._apply_near_goal_starts(pos)
        return pos

    def _apply_near_goal_starts(self, pos: torch.Tensor) -> torch.Tensor:
        """Respawn a fraction of agents around their own port (training aid).
        Ports are far apart, so the jitter cannot put a pair inside D_s.
        About half of the agents start behind their port.
        """
        if self.init_near_goal_fraction <= 0.0 or self.init_near_goal_radius <= 0.0:
            return pos
        goals = self._compute_goals_from_target()
        pick = (
            self._rand_uniform((0.0, 1.0), (self.num_envs, self.n_agents, 1))
            < self.init_near_goal_fraction
        )
        # Uniform in the ball: direction * r^(1/3).
        direction = torch.randn(
            (self.num_envs, self.n_agents, 3),
            generator=self.rng,
            device=self.device,
            dtype=pos.dtype,
        )
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        radius = self._rand_uniform((0.0, 1.0), (self.num_envs, self.n_agents, 1)) ** (1.0 / 3.0)
        near = goals + direction * radius * self.init_near_goal_radius
        return torch.where(pick, near, pos)

    def _draw_raw_positions(self) -> torch.Tensor:
        """One draw from the spawn distribution."""
        if self.init_agent_pos_means is not None and self.init_agent_pos_ranges is not None:
            eps = self._rand_uniform((-1.0, 1.0), (self.num_envs, self.n_agents, 3))
            means = self.init_agent_pos_means.unsqueeze(0).expand(self.num_envs, self.n_agents, 3)
            ranges = self.init_agent_pos_ranges.unsqueeze(0).expand(self.num_envs, self.n_agents, 3)
            return means + eps * ranges
        return self._rand_uniform(
            (-self.init_pos_radius, self.init_pos_radius),
            (self.num_envs, self.n_agents, 3),
        )

    def _min_pairwise_separation(self, pos: torch.Tensor) -> torch.Tensor:
        """[B] smallest distance between any agent pair in each env."""
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B, N, N, 3]
        dist = diff.norm(dim=-1)  # [B, N, N]
        eye = torch.eye(self.n_agents, dtype=torch.bool, device=self.device)
        dist = dist.masked_fill(eye.unsqueeze(0), float("inf"))
        return dist.amin(dim=(1, 2))

    def _enforce_min_separation(self, pos: torch.Tensor) -> torch.Tensor:
        """Draw again all agents of each env whose closest pair is too close.
        Stop after max_spawn_resample_iters and warn once.
        """
        for _ in range(self.max_spawn_resample_iters):
            violating = self._min_pairwise_separation(pos) < self.min_spawn_separation  # [B]
            if not violating.any():
                return pos
            redraw = self._draw_raw_positions()
            pos = torch.where(violating.view(self.num_envs, 1, 1), redraw, pos)

        if not self._warned_spawn_infeasible and bool(
            (self._min_pairwise_separation(pos) < self.min_spawn_separation).any()
        ):
            self._warned_spawn_infeasible = True
            import warnings

            warnings.warn(
                f"min_spawn_separation={self.min_spawn_separation} m not met in "
                f"every env after {self.max_spawn_resample_iters} resamples. "
                "Using the closest draw.",
                RuntimeWarning,
            )
        return pos

    def _set_agent_spawn_specs(
        self,
        *,
        means: Optional[list[list[float]]],
        ranges: Optional[list[list[float] | float]],
    ) -> None:
        if means is None or ranges is None:
            self.init_agent_pos_means = None
            self.init_agent_pos_ranges = None
            return
        self.init_agent_pos_means = self._to_means_tensor(means)
        self.init_agent_pos_ranges = self._to_ranges_tensor(ranges)

    def _to_means_tensor(self, means: list[list[float]]) -> torch.Tensor:
        means_t = torch.as_tensor(means, dtype=torch.float32, device=self.device)
        if means_t.shape != (self.n_agents, 3):
            raise ValueError(
                f"init_agent_pos_means must have shape [{self.n_agents}, 3], got {tuple(means_t.shape)}"
            )
        return means_t

    def _to_ranges_tensor(self, ranges: list[list[float] | float]) -> torch.Tensor:
        range_rows: list[list[float]] = []
        for item in ranges:
            if isinstance(item, (int, float)):
                scalar = float(item)
                range_rows.append([scalar, scalar, scalar])
            else:
                values = [float(v) for v in item]
                if len(values) != 3:
                    raise ValueError(
                        "Each init_agent_pos_ranges entry must be a scalar or length-3 list."
                    )
                range_rows.append(values)
        ranges_t = torch.as_tensor(range_rows, dtype=torch.float32, device=self.device)
        if ranges_t.shape != (self.n_agents, 3):
            raise ValueError(
                f"init_agent_pos_ranges must have shape [{self.n_agents}, 3], got {tuple(ranges_t.shape)}"
            )
        if (ranges_t < 0).any():
            raise ValueError("init_agent_pos_ranges must be non-negative.")
        return ranges_t

    def _compute_goals_from_target(self) -> torch.Tensor:
        """Goal of each agent's centre of mass.

        The goal is the port location moved out along +X by
            com_standoff = docking_offset_x + 0.5 * agent_size_x
        When the centre reaches the goal, the docking face of the chaser is
        docking_offset_x in front of the target face.
        """
        goals = torch.zeros(
            (self.num_envs, self.n_agents, 3), dtype=torch.float32, device=self.device
        )
        half = 0.5 * self.target_size
        com_standoff = self.docking_offset_x + 0.5 * self.agent_size[0]
        face_x = self.target_center[:, 0].unsqueeze(1) + half[0] + com_standoff

        # Centred rows x cols port grid on the +X face at port_pitch
        # (agent i -> column i // rows, row i % rows).
        p = self.port_pitch
        cols, rows = self.port_cols, self.port_rows
        cy = self.target_center[:, 1].unsqueeze(1)  # [B, 1]
        cz = self.target_center[:, 2].unsqueeze(1)
        goals[..., 0] = face_x
        for i in range(self.n_agents):
            col = i // rows
            row = i % rows
            y = (col - (cols - 1) / 2.0) * p
            z = (row - (rows - 1) / 2.0) * p
            goals[:, i, 1] = (cy + y).squeeze(1)
            goals[:, i, 2] = (cz + z).squeeze(1)
        return goals

    def _relative_goal_vector(self) -> torch.Tensor:
        """Vector from each agent to its own docking point."""
        return self.goals - self.pos

    def _current_distance(self) -> torch.Tensor:
        rel_goal = torch.nan_to_num(self._relative_goal_vector(), nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.linalg.norm(rel_goal, dim=-1, keepdim=True)

    def _current_speed(self) -> torch.Tensor:
        vel = torch.nan_to_num(self.vel, nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.linalg.norm(vel, dim=-1, keepdim=True)

    def _capture_condition(self, distance: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """Capture: inside the position tolerance and slower than max_docking_speed.
        With the attitude gate, the attitude error must be below
        max_attitude_error and the rate below max_docking_ang_speed.
        """
        within_position = distance <= self.goal_tolerance
        within_speed = speed <= self.max_docking_speed
        captured = within_position & within_speed
        if self.enable_attitude and self.attitude_capture_gate:
            within_attitude = self._attitude_error_angle() <= self.max_attitude_error
            within_ang_speed = (
                torch.linalg.norm(self.ang_vel, dim=-1, keepdim=True)
                <= self.max_docking_ang_speed
            )
            captured = captured & within_attitude & within_ang_speed
        return captured

    # Quaternion and attitude helpers.
    def _identity_quat(self, batch_shape: tuple[int, ...]) -> torch.Tensor:
        q = torch.zeros((*batch_shape, 4), dtype=torch.float32, device=self.device)
        q[..., 0] = 1.0
        return q

    @staticmethod
    def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Hamilton product q1 (x) q2 for [w, x, y, z] quaternions."""
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )

    @staticmethod
    def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
        return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

    def _desired_attitude(self) -> torch.Tensor:
        """Desired attitude per agent. All ports are on the +X face, so q_des is the
        identity quaternion.
        """
        return self._identity_quat((self.num_envs, self.n_agents))

    def _attitude_error_quat(self) -> torch.Tensor:
        """Error quaternion q_err = q_des (x) q^-1, with w >= 0 (shortest rotation)."""
        q = self.quat / self.quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        q_err = self._quat_mul(self._desired_attitude(), self._quat_conjugate(q))
        sign = torch.where(q_err[..., :1] < 0.0, -1.0, 1.0)
        return q_err * sign

    def _attitude_error_terms(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (2*q_err_vec, ang_vel_err), each [B, N, 3]. The desired rate is zero."""
        q_err = self._attitude_error_quat()
        att_vec = 2.0 * q_err[..., 1:4]
        ang_err = self.ang_vel  # omega_des = 0
        return att_vec, ang_err

    def _attitude_error_angle(self) -> torch.Tensor:
        """Attitude error angle in radians, [B, N, 1]."""
        q_err = self._attitude_error_quat()
        vec_norm = q_err[..., 1:4].norm(dim=-1, keepdim=True)
        w = q_err[..., :1].clamp(-1.0, 1.0)
        return 2.0 * torch.atan2(vec_norm, w)

    def _sample_initial_attitude(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Random initial attitude and rate. Identity and zero when attitude is off."""
        shape = (self.num_envs, self.n_agents)
        if not self.enable_attitude:
            return self._identity_quat(shape), torch.zeros(
                (*shape, 3), dtype=torch.float32, device=self.device
            )
        axis = self._rand_uniform((-1.0, 1.0), (*shape, 3))
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        angle = self._rand_uniform((0.0, self.init_max_attitude_angle), (*shape, 1))
        quat = torch.cat(
            [torch.cos(0.5 * angle), torch.sin(0.5 * angle) * axis], dim=-1
        )
        ang_vel = self._rand_uniform(
            (-self.init_max_ang_vel, self.init_max_ang_vel), (*shape, 3)
        )
        return quat, ang_vel

    # Nearest-neighbour observation.
    def _nearest_neighbor_features(
        self, count: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Relative position and velocity of the `count` nearest teammates, nearest
        first. Returns two [B, N, count, 3] tensors. Missing slots are zero.
        """
        rel_pos = torch.zeros(
            (self.num_envs, self.n_agents, count, 3), dtype=torch.float32, device=self.device
        )
        rel_vel = torch.zeros_like(rel_pos)
        k = min(count, self.n_agents - 1)
        if k < 1:
            return rel_pos, rel_vel
        diff = self.pos.unsqueeze(2) - self.pos.unsqueeze(1)  # [B, N(self), N(other), 3]
        dist = diff.norm(dim=-1)  # [B, N, N]
        eye = torch.eye(self.n_agents, dtype=torch.bool, device=self.device)
        dist = dist.masked_fill(eye.unsqueeze(0), float("inf"))
        nearest = dist.topk(k, dim=-1, largest=False).indices  # [B, N, k], nearest first

        batch_idx = torch.arange(self.num_envs, device=self.device).view(-1, 1, 1)
        nn_pos = self.pos[batch_idx, nearest]  # [B, N, k, 3]
        nn_vel = self.vel[batch_idx, nearest]
        rel_pos[:, :, :k] = nn_pos - self.pos.unsqueeze(2)
        rel_vel[:, :, :k] = nn_vel - self.vel.unsqueeze(2)
        return rel_pos, rel_vel

    def _safety_obs_block(self) -> torch.Tensor:
        """[B, N, 4] safety features:
          1. gap: (||dp_nearest|| - D_s) / R_act.
          2. tanh(psi1), psi1 = 2 dp.dv + lambda1 (||dp||^2 - D_s^2).
             psi1 < 0: the pair closes faster than the barrier allows.
          3. filter intervention of the previous step (normalised units).
          4. filter-active flag of the previous step.
        Features 3-4 are zero when enable_cbf is off.
        """
        if self.n_agents < 2:
            gap = torch.ones((self.num_envs, self.n_agents, 1), device=self.device)
            psi1 = torch.ones_like(gap)
        else:
            rel_pos, rel_vel = self._nearest_neighbor_features(1)
            rel_pos, rel_vel = rel_pos[:, :, 0], rel_vel[:, :, 0]
            dist = rel_pos.norm(dim=-1, keepdim=True)
            gap = (dist - self.cbf_agent_keepout) / self._cbf_gap_obs_scale
            psi1 = 2.0 * (rel_pos * rel_vel).sum(dim=-1, keepdim=True) + self.cbf_lambda1 * (
                dist**2 - self.cbf_agent_keepout**2
            )
        intervention = self.last_cbf_intervention
        flag = (intervention > 1e-6).float()
        block = torch.cat([gap, torch.tanh(psi1), intervention, flag], dim=-1)
        if len(self.safety_obs_features) == 4:
            return block
        idx = torch.tensor(self.safety_obs_features, device=block.device)
        return block.index_select(-1, idx)

    def _capture_potential(
        self,
        distance: Optional[torch.Tensor] = None,
        speed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Potential Phi = near * slow, 1 at capture. near goes 0 -> 1 inside
        capture_shaping_radius. slow goes 0 -> 1 below max_docking_speed.
        """
        if distance is None:
            distance = self._current_distance()
        if speed is None:
            speed = self._current_speed()
        near = (1.0 - distance / self.capture_shaping_radius).clamp(min=0.0, max=1.0)
        slow = (1.0 - speed / max(self.max_docking_speed, 1e-6)).clamp(min=0.0, max=1.0)
        return near * slow

    def _collision_progress_and_flag(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Box collision flag with the target and penetration along x."""
        agent_half = 0.5 * self.agent_size.view(1, 1, 3)
        target_half = 0.5 * self.target_size.view(1, 1, 3)
        agent_min = self.pos - agent_half
        agent_max = self.pos + agent_half
        target_center = self.target_center.unsqueeze(1)
        target_min = target_center - target_half
        target_max = target_center + target_half

        overlap = torch.minimum(agent_max, target_max) - torch.maximum(agent_min, target_min)
        overlap_pos = overlap.clamp(min=0.0)
        yz_overlap = (overlap_pos[..., 1:3] > 0.0).all(dim=-1, keepdim=True)
        face_x = target_center[..., 0:1] + target_half[..., 0:1]
        x_penetration = (face_x - agent_min[..., 0:1]).clamp(min=0.0)
        collision = yz_overlap & (x_penetration > 0.0)
        # TorchRL needs done=False from reset(), so this starts after the first step.
        collision = collision & (self.step_count > 0).unsqueeze(1)
        penetration = torch.where(collision, x_penetration, torch.zeros_like(x_penetration))
        progress = (penetration / self.collision_smoothing_distance).clamp(0.0, 1.0)
        return progress, collision, penetration

    def _agent_collision_progress_and_flag(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inter-agent bounding-sphere collision.

        r_bound is half the box diagonal, so the test does not depend on attitude.
        For each agent and its closest teammate:

            contact_dist = 2 * r_bound + agent_collision_margin
            penetration  = max(0, contact_dist - min_separation)
            band_start   = contact_dist + agent_collision_warning_distance
            ramp_len     = warning_distance                (terminal_on_contact)
                         = warning_distance + smoothing    (otherwise)
            progress     = clamp((band_start - min_sep) / ramp_len, 0, 1)

        Return (progress, collision, penetration, min_separation), all [B, N, 1].
        All zero (separation +inf) when the switch is off or N = 1.
        """
        zeros = torch.zeros(
            (self.num_envs, self.n_agents, 1), dtype=torch.float32, device=self.device
        )
        if not self.enable_agent_collisions or self.n_agents < 2:
            return zeros, zeros.bool(), zeros, torch.full_like(zeros, float("inf"))

        diff = self.pos.unsqueeze(2) - self.pos.unsqueeze(1)  # [B, N, N, 3]
        dist = diff.norm(dim=-1)  # [B, N, N]
        eye = torch.eye(self.n_agents, dtype=torch.bool, device=self.device)
        dist = dist.masked_fill(eye.unsqueeze(0), float("inf"))
        min_sep = dist.min(dim=-1).values.unsqueeze(-1)  # [B, N, 1]

        r_bound = 0.5 * torch.linalg.norm(self.agent_size)
        contact_dist = 2.0 * r_bound + self.agent_collision_margin
        # No collision at step 0 (TorchRL needs done=False from reset()).
        step_active = (self.step_count > 0).unsqueeze(1)  # [B, 1, 1]

        penetration = (contact_dist - min_sep).clamp(min=0.0)
        collision = (penetration > 0.0) & step_active
        penetration = torch.where(collision, penetration, torch.zeros_like(penetration))

        band_start = contact_dist + self.agent_collision_warning_distance
        if self.agent_collision_terminal_on_contact:
            ramp_len = max(self.agent_collision_warning_distance, 1e-6)
        else:
            ramp_len = (
                self.agent_collision_warning_distance
                + self.agent_collision_smoothing_distance
            )
        proximity = (band_start - min_sep).clamp(min=0.0) * step_active.float()
        progress = (proximity / ramp_len).clamp(0.0, 1.0)
        return progress, collision, penetration, min_sep

    def _agent_collision_terminal_mask(
        self, collision: torch.Tensor, penetration: torch.Tensor
    ) -> torch.Tensor:
        """Terminal-crash mask [B, N, 1]."""
        if not self.enable_agent_collisions:
            return torch.zeros_like(collision)
        if self.agent_collision_terminal_on_contact:
            return collision
        return penetration >= self.agent_collision_smoothing_distance
