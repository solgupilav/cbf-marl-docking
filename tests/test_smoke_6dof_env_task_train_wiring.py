import torch
from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

from marl_docking_6dof.tasks import MarlDocking6DoFTask


def test_6dof_env_task_and_mappo_wiring_smoke():
    task = MarlDocking6DoFTask.DOCKING_6DOF.get_from_yaml()
    task.config.update(
        {
            "n_agents": 2,
            "max_steps": 30,
            "goal_tolerance": 0.4,
            "reward_progress_scale": 1.0,
            "reward_goal_bonus": 10.0,
        }
    )

    env = task.get_env_fun(
        num_envs=4,
        continuous_actions=True,
        seed=0,
        device="cpu",
    )()
    td = env.reset()
    assert td is not None

    rollout = env.rollout(max_steps=5)
    assert rollout is not None

    algo = MappoConfig(
        share_param_critic=True,
        clip_epsilon=0.2,
        entropy_coef=0.001,
        critic_coef=1.0,
        loss_critic_type="l2",
        lmbda=0.95,
        scale_mapping="biased_softplus_1.0",
        use_tanh_normal=True,
        minibatch_advantage=False,
    )
    model = MlpConfig.get_from_yaml()
    critic_model = MlpConfig.get_from_yaml()
    model.num_cells = [64, 64]
    critic_model.num_cells = [64, 64]

    exp_config = ExperimentConfig.get_from_yaml()
    exp_config.prefer_continuous_actions = True
    exp_config.max_n_frames = 64
    exp_config.on_policy_n_envs_per_worker = 4
    exp_config.on_policy_collected_frames_per_batch = 32
    exp_config.on_policy_n_minibatch_iters = 1
    exp_config.on_policy_minibatch_size = 16
    exp_config.evaluation = False
    exp_config.render = False
    exp_config.loggers = ["csv"]

    experiment = Experiment(
        task=task,
        algorithm_config=algo,
        model_config=model,
        critic_model_config=critic_model,
        seed=0,
        config=exp_config,
    )
    experiment.run()


def test_6dof_env_is_solvable_with_pd_controller():
    """A PD controller must dock with a soft arrival. If it cannot, RL cannot."""
    task = MarlDocking6DoFTask.DOCKING_6DOF.get_from_yaml()
    task.config.update(
        {
            "n_agents": 4,
            "max_steps": 300,
            "goal_tolerance": 0.4,
            "max_docking_speed": 0.1,
            "init_agent_pos_means": [
                [10.0, -0.8, -0.8],
                [8.0, -0.8, 0.8],
                [10.0, 0.8, -0.8],
                [9.0, 0.8, 0.8],
            ],
            "init_agent_pos_ranges": [0.15, 0.15, 0.15, 0.15],
        }
    )
    num_envs = 8
    env = task.get_env_fun(num_envs=num_envs, continuous_actions=True, seed=0, device="cpu")()

    td = env.reset()
    total_reward = torch.zeros(num_envs, env.n_agents, 1)
    reached = torch.zeros(num_envs, env.n_agents, 1, dtype=torch.bool)

    # PD gains: the proportional term saturates far away. The large derivative
    # term makes the arrival soft.
    kp, kd = 2.0, 30.0
    for _ in range(int(task.config["max_steps"])):
        rel_goal = env.goals - env.pos  # [B, N, 3]
        action = (kp * rel_goal - kd * env.vel).clamp(-1.0, 1.0)

        td.set((env.group, "action"), action)
        td_step = env.step(td)
        nxt = td_step["next"]
        total_reward += nxt[(env.group, "reward")]
        reached |= nxt[(env.group, "info", "on_goal")]
        td = nxt
        if bool(nxt["done"].all()):
            break

    success_rate = reached.float().mean().item()
    mean_return = total_reward.mean().item()
    assert success_rate > 0.95, f"controller only docked {success_rate:.2%} of agents"
    assert mean_return > 0.0, f"episode return should be positive, got {mean_return:.3f}"


def _make_env(**overrides):
    task = MarlDocking6DoFTask.DOCKING_6DOF.get_from_yaml()
    task.config.update({"n_agents": 4, "max_steps": 50, **overrides})
    return task.get_env_fun(num_envs=8, continuous_actions=True, seed=0, device="cpu")()


def test_feature_switches_derive_obs_and_action_dims():
    # Translation only: 6 obs, 3 actions.
    env = _make_env()
    assert env.obs_dim == 6 and env.action_dim == 3

    # Attitude adds 6 obs [2*q_err_vec, w_err] and 3 torque actions.
    env = _make_env(enable_attitude=True)
    assert env.obs_dim == 12 and env.action_dim == 6

    # Neighbour observation adds the relative position and velocity (6 obs).
    env = _make_env(enable_neighbor_obs=True)
    assert env.obs_dim == 12 and env.action_dim == 3

    # All features on.
    env = _make_env(enable_attitude=True, enable_neighbor_obs=True)
    assert env.obs_dim == 18 and env.action_dim == 6
    td = env.reset()
    assert td[(env.group, "observation")].shape == (8, 4, 18)
    rollout = env.rollout(max_steps=5)
    assert rollout is not None


def test_attitude_dynamics_and_error_terms():
    env = _make_env(enable_attitude=True, init_max_attitude_angle=0.5)
    env.reset()
    # Quaternions have unit norm after reset and after a step.
    assert torch.allclose(env.quat.norm(dim=-1), torch.ones(8, 4), atol=1e-5)
    angle0 = env._attitude_error_angle().clone()
    assert (angle0 <= 0.5 + 1e-4).all() and (angle0 >= 0).all()

    # A torque command changes the attitude. Quaternions stay unit.
    td = env.reset()
    action = torch.zeros(8, 4, 6)
    action[..., 3] = 1.0
    td.set((env.group, "action"), action)
    env.step(td)
    assert torch.allclose(env.quat.norm(dim=-1), torch.ones(8, 4), atol=1e-4)
    assert (env.ang_vel.abs().sum() > 0).item()


def test_agent_collision_flags_when_overlapping():
    env = _make_env(
        enable_agent_collisions=True,
        # Spawn all four agents at almost the same point.
        init_agent_pos_means=[[5.0, 0.0, 0.0]] * 4,
        init_agent_pos_ranges=[0.01] * 4,
    )
    td = env.reset()
    td.set((env.group, "action"), torch.zeros(8, 4, 3))
    nxt = env.step(td)["next"]
    info = nxt[(env.group, "info")]
    assert info["agent_collision"].all(), "overlapping agents must flag collision"
    assert (info["agent_collision_penetration"] > 0).all()
    assert info["agent_collision_terminal"].all()
    assert bool(nxt["done"].all()), "deep inter-agent overlap must terminate"
    assert (nxt[(env.group, "reward")] < 0).all(), "collision must be penalised"

    # Same geometry with the switch off: no flags and no termination.
    env = _make_env(
        enable_agent_collisions=False,
        init_agent_pos_means=[[5.0, 0.0, 0.0]] * 4,
        init_agent_pos_ranges=[0.01] * 4,
    )
    td = env.reset()
    td.set((env.group, "action"), torch.zeros(8, 4, 3))
    nxt = env.step(td)["next"]
    assert not nxt[(env.group, "info")]["agent_collision"].any()
    assert not bool(nxt["done"].any())


def test_cbf_placeholder_is_exact_passthrough():
    # With the pass-through filter, enable_cbf=True must give the same result
    # as enable_cbf=False and zero intervention. This also checks the
    # filtered integration at the substep rate.
    from marl_docking_6dof.safety.cbf import CBFSafetyFilter

    trajs = []
    for enable_cbf in (False, True):
        env = _make_env(enable_cbf=enable_cbf, safety_filter=CBFSafetyFilter())
        td = env.reset()
        torch.manual_seed(0)
        for _ in range(10):
            action = torch.rand(8, 4, 3) * 2 - 1
            td.set((env.group, "action"), action)
            td = env.step(td)["next"]
        trajs.append((env.pos.clone(), td[(env.group, "info")]["cbf_intervention"].clone()))
    assert torch.equal(trajs[0][0], trajs[1][0]), "placeholder CBF must not alter dynamics"
    assert (trajs[1][1] == 0).all(), "placeholder CBF must report zero intervention"


def test_6dof_full_task_solvable_with_pd_attitude_controller():
    """With attitude, agent collisions and neighbour obs on, a PD translation
    controller and a quaternion attitude controller must still dock.
    """
    env = _make_env(
        enable_attitude=True,
        enable_agent_collisions=True,
        enable_neighbor_obs=True,
        max_steps=120,
        goal_tolerance=0.4,
        max_docking_speed=0.1,
        init_max_attitude_angle=0.5,
        init_agent_pos_means=[
            [6.0, -0.8, -0.8],
            [5.0, -0.8, 0.8],
            [6.0, 0.8, -0.8],
            [5.5, 0.8, 0.8],
        ],
        init_agent_pos_ranges=[0.15] * 4,
        linear_accel_scale=0.01,
    )
    td = env.reset()
    reached = torch.zeros(8, env.n_agents, 1, dtype=torch.bool)
    for _ in range(120):
        rel = env.goals - env.pos
        lin = (2.0 * rel - 30.0 * env.vel).clamp(-1.0, 1.0)
        # Quaternion attitude controller: torque ~ -kp*q_err_vec - kd*w
        att_vec, ang_err = env._attitude_error_terms()
        # att_vec = 2*q_err_vec of q_des (x) q^-1. In this body-frame convention
        # the correction needs negative feedback.
        ang = (0.5 * att_vec - 10.0 * ang_err).clamp(-1.0, 1.0)
        td.set((env.group, "action"), torch.cat([lin, ang], dim=-1))
        nxt = env.step(td)["next"]
        reached |= nxt[(env.group, "info", "on_goal")]
        td = nxt
        if bool(nxt["done"].all()):
            break
    success = reached.float().mean().item()
    assert success > 0.95, f"6-DoF PD controller only docked {success:.2%}"
