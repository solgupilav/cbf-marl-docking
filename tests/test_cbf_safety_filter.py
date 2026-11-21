"""Tests for the HOCBF-QP safety filter and related environment features."""

import itertools
import math

import pytest
import torch

from marl_docking_6dof.envs import Docking6DoFEnv
from marl_docking_6dof.safety.cbf import CBFSafetyFilter, HOCBFSafetyFilter

CONTACT_DIST = 0.4425  # 2 * bounding-sphere radius of the 6U chaser (m)


# QP solver.
def _reference_qp(u_nom, rows_c, rows_d, bound):
    """Exact projection onto {C x <= d, |x|_inf <= bound} by enumeration of active
    sets (float64). Return None if the set is empty.
    """
    a_full = torch.cat([rows_c, torch.eye(3, dtype=torch.float64), -torch.eye(3, dtype=torch.float64)])
    b_full = torch.cat(
        [rows_d, bound * torch.ones(3, dtype=torch.float64), bound * torch.ones(3, dtype=torch.float64)]
    )
    m = a_full.shape[0]
    best_x, best_val = None, float("inf")
    for r in range(0, 4):
        for subset in itertools.combinations(range(m), r):
            idx = list(subset)
            a_s = a_full[idx]
            if r == 0:
                x = u_nom.clone()
            else:
                gram = a_s @ a_s.T
                try:
                    lam = torch.linalg.solve(gram, a_s @ u_nom - b_full[idx])
                except RuntimeError:
                    continue
                if (lam < -1e-9).any():
                    continue
                x = u_nom - a_s.T @ lam
            if (a_full @ x <= b_full + 1e-9).all():
                val = float(((x - u_nom) ** 2).sum())
                if val < best_val:
                    best_val, best_x = val, x
    return best_x


def test_qp_solver_matches_active_set_reference():
    torch.manual_seed(7)
    flt = HOCBFSafetyFilter()
    bound = 0.005
    n_prob, n_rows = 120, 4
    # float32 is the working precision of the filter. Promote to float64 for the
    # reference, so both solve the same problem.
    u_nom = ((torch.rand(n_prob, 3) * 2 - 1) * 2 * bound).float()
    rows_c = torch.randn(n_prob, n_rows, 3).float()
    # A mix of slack, active and sometimes infeasible rows.
    rows_d = ((torch.rand(n_prob, n_rows) * 3 - 1) * bound).float()

    x_batch, infeasible = flt._solve_box_qp(u_nom, rows_c, rows_d, bound)
    u_nom, rows_c, rows_d = u_nom.double(), rows_c.double(), rows_d.double()
    n_checked = 0
    for i in range(n_prob):
        x_ref = _reference_qp(u_nom[i], rows_c[i], rows_d[i], bound)
        if x_ref is None:
            assert bool(infeasible[i]), f"problem {i}: reference infeasible, solver disagreed"
            continue
        assert not bool(infeasible[i]), f"problem {i}: reference feasible, solver flagged infeasible"
        x = x_batch[i].double()
        # Feasible and objective not above the optimum, so the point is optimal.
        viol = (rows_c[i] @ x - rows_d[i]).max()
        assert float(viol) < 1e-3 * bound, f"problem {i}: violation {float(viol):.2e}"
        assert float(x.abs().max()) <= bound + 1e-12, f"problem {i}: box violated"
        obj = float(((x - u_nom[i]) ** 2).sum())
        obj_ref = float(((x_ref - u_nom[i]) ** 2).sum())
        # Feasible with objective within 0.01% of the optimum. By strong convexity,
        # ||x - x*||^2 <= obj - obj_ref. The tolerance allows for float32 rounding.
        assert obj <= obj_ref * (1.0 + 1e-4) + 1e-8 * bound**2, (
            f"problem {i}: objective {obj:.3e} > reference {obj_ref:.3e}"
        )
        n_checked += 1
    assert n_checked > 50, "test degenerate: almost every random problem was infeasible"


def test_qp_solver_flags_infeasible_and_stays_in_box():
    flt = HOCBFSafetyFilter()
    bound = 0.005
    # x_1 <= -2*bound is outside the box: empty set.
    u_nom = torch.zeros(1, 3)
    rows_c = torch.tensor([[[1.0, 0.0, 0.0]]])
    rows_d = torch.tensor([[-2.0 * bound]])
    x, infeasible = flt._solve_box_qp(u_nom, rows_c, rows_d, bound)
    assert bool(infeasible[0])
    assert float(x.abs().max()) <= bound + 1e-12


# Actuator limits.
def test_filter_action_always_respects_actuator_limits():
    torch.manual_seed(0)
    flt = HOCBFSafetyFilter(linear_accel_scale=0.005, angular_accel_scale=0.02)
    b_sz, n_agents = 16, 4
    pos = torch.randn(b_sz, n_agents, 3) * 2.0
    vel = torch.randn(b_sz, n_agents, 3) * 0.3
    ang_vel = torch.randn(b_sz, n_agents, 3) * 0.5
    flt.target_center = torch.zeros(b_sz, 3)
    # Extreme, out-of-range and non-finite nominal actions.
    action = torch.randn(b_sz, n_agents, 6) * 10.0
    action[0, 0, 0] = float("nan")
    action[1, 2, 4] = float("inf")
    safe, intervention = flt.filter_action(
        pos=pos, vel=vel, action=action, ang_vel=ang_vel
    )
    assert torch.isfinite(safe).all()
    assert float(safe.abs().max()) <= 1.0 + 1e-6, "normalised action left [-1, 1]"
    assert intervention.shape == (b_sz, n_agents, 1)
    assert torch.isfinite(intervention).all()


# Inter-agent collision avoidance.
def _head_on_env(enable_cbf: bool) -> Docking6DoFEnv:
    return Docking6DoFEnv(
        num_envs=4,
        n_agents=2,
        max_steps=400,
        linear_accel_scale=0.005,
        enable_cbf=enable_cbf,
        # No terminal crash, so the episode runs to the end.
        enable_agent_collisions=False,
        init_agent_pos_means=[[5.0, -1.6, 0.0], [5.0, 1.6, 0.0]],
        init_agent_pos_ranges=[0.0, 0.0],
        seed=0,
        device="cpu",
    )


def _drive_head_on(env: Docking6DoFEnv, steps: int) -> float:
    """Command the two agents toward each other. Return the minimum separation."""
    td = env.reset()
    action = torch.zeros(env.num_envs, 2, 3)
    action[:, 0, 1] = 1.0  # agent 0 accelerates +y (toward agent 1)
    action[:, 1, 1] = -1.0  # agent 1 accelerates -y (toward agent 0)
    min_sep = float("inf")
    for _ in range(steps):
        td.set((env.group, "action"), action)
        td = env.step(td)["next"]
        sep = (env.pos[:, 0] - env.pos[:, 1]).norm(dim=-1)
        min_sep = min(min_sep, float(sep.min()))
    return min_sep


def test_head_on_agents_collide_without_filter_but_not_with_it():
    # Without the filter, the head-on command causes contact.
    min_sep_free = _drive_head_on(_head_on_env(enable_cbf=False), steps=60)
    assert min_sep_free < CONTACT_DIST, (
        f"scenario not stressing: unfiltered min separation {min_sep_free:.3f} m"
    )
    # With the filter, the separation stays above D_s (0.55 m) minus a small
    # sampling allowance, and the agents do not touch.
    env = _head_on_env(enable_cbf=True)
    min_sep_cbf = _drive_head_on(env, steps=400)
    assert min_sep_cbf >= env.cbf_agent_keepout - 0.01, (
        f"keep-apart violated: min separation {min_sep_cbf:.3f} m"
    )
    # After 400 s the pair is near D_s, so the constraint was active.
    assert min_sep_cbf < 0.8, (
        "constraint never pressed - test is vacuous "
        f"(min separation {min_sep_cbf:.3f} m)"
    )
    assert isinstance(env.safety_filter, HOCBFSafetyFilter)
    assert env.safety_filter.last_infeasible == 0


def test_head_on_intervention_is_logged_and_penalised():
    env = Docking6DoFEnv(
        num_envs=2,
        n_agents=2,
        max_steps=400,
        linear_accel_scale=0.005,
        enable_cbf=True,
        cbf_intervention_penalty_scale=0.05,
        enable_agent_collisions=False,
        init_agent_pos_means=[[5.0, -0.4, 0.0], [5.0, 0.4, 0.0]],
        init_agent_pos_ranges=[0.0, 0.0],
        seed=0,
        device="cpu",
    )
    td = env.reset()
    action = torch.zeros(2, 2, 3)
    action[:, 0, 1] = 1.0
    action[:, 1, 1] = -1.0
    td.set((env.group, "action"), action)
    nxt = env.step(td)["next"]
    info = nxt[(env.group, "info")]
    assert (info["cbf_intervention"] > 0).all(), "filter should intervene at 0.8 m gap"
    assert (info["reward_cbf_intervention"] < 0).all(), "intervention must be penalised"


# Target speed limit.
def test_speed_limit_is_enforced_on_full_thrust_approach():
    env = Docking6DoFEnv(
        num_envs=2,
        n_agents=2,
        max_steps=400,
        linear_accel_scale=0.005,
        enable_cbf=True,
        # Far apart in y, so only the speed-limit row is active.
        init_agent_pos_means=[[10.0, -3.0, 0.0], [10.0, 3.0, 0.0]],
        init_agent_pos_ranges=[0.0, 0.0],
        seed=0,
        device="cpu",
    )
    td = env.reset()
    action = torch.zeros(2, 2, 3)
    action[..., 0] = -1.0  # full thrust toward the target
    max_ratio = 0.0
    for _ in range(150):
        td.set((env.group, "action"), action)
        td = env.step(td)["next"]
        speed = env.vel.norm(dim=-1)
        rng = (env.pos - env.target_center.unsqueeze(1)).norm(dim=-1)
        limit = env.cbf_speed_limit_intercept + env.cbf_speed_limit_slope * rng
        assert float((speed - limit).max()) < 0.02, (
            f"speed limit violated: {float(speed.max()):.3f} vs {float(limit.min()):.3f}"
        )
        max_ratio = max(max_ratio, float((speed / limit).max()))
    assert max_ratio > 0.9, f"speed limit never pressed (max ratio {max_ratio:.2f})"


# Angular-rate limit.
def test_angular_rate_limit_is_enforced_on_full_spin_command():
    env = Docking6DoFEnv(
        num_envs=2,
        n_agents=2,
        max_steps=400,
        enable_attitude=True,
        enable_cbf=True,
        init_agent_pos_means=[[10.0, -3.0, 0.0], [10.0, 3.0, 0.0]],
        init_agent_pos_ranges=[0.0, 0.0],
        init_max_attitude_angle=0.3,
        seed=0,
        device="cpu",
    )
    td = env.reset()
    action = torch.zeros(2, 2, 6)
    action[..., 3:6] = 1.0  # full spin-up command on all axes
    max_rate = 0.0
    for _ in range(100):
        td.set((env.group, "action"), action)
        td = env.step(td)["next"]
        rate = float(env.ang_vel.norm(dim=-1).max())
        max_rate = max(max_rate, rate)
        assert rate <= env.cbf_max_ang_speed + 5e-3, f"rate limit violated: {rate:.3f} rad/s"
    assert max_rate > 0.9 * env.cbf_max_ang_speed, "rate limit never pressed"


# Docking is still possible with the filter.
def test_full_6dof_docking_succeeds_with_filter_and_zero_contacts():
    env = Docking6DoFEnv(
        num_envs=8,
        n_agents=4,
        max_steps=200,
        goal_tolerance=0.4,
        max_docking_speed=0.1,
        enable_attitude=True,
        enable_agent_collisions=True,
        agent_collision_terminal_on_contact=True,
        enable_neighbor_obs=True,
        enable_cbf=True,
        enable_safety_obs=True,
        init_max_attitude_angle=0.5,
        init_agent_pos_means=[
            [6.0, -0.8, -0.8],
            [5.0, -0.8, 0.8],
            [6.0, 0.8, -0.8],
            [5.5, 0.8, 0.8],
        ],
        init_agent_pos_ranges=[0.15] * 4,
        min_spawn_separation=0.9,
        linear_accel_scale=0.01,
        port_spread=0.45,
        seed=0,
        device="cpu",
    )
    assert env.obs_dim == 22  # 6 base + 6 attitude + 6 neighbour + 4 safety
    td = env.reset()
    assert td[(env.group, "observation")].shape == (8, 4, 22)
    reached = torch.zeros(8, env.n_agents, 1, dtype=torch.bool)
    any_contact = False
    for _ in range(200):
        rel = env.goals - env.pos
        lin = (2.0 * rel - 30.0 * env.vel).clamp(-1.0, 1.0)
        att_vec, ang_err = env._attitude_error_terms()
        ang = (0.5 * att_vec - 10.0 * ang_err).clamp(-1.0, 1.0)
        td.set((env.group, "action"), torch.cat([lin, ang], dim=-1))
        nxt = env.step(td)["next"]
        info = nxt[(env.group, "info")]
        reached |= info["on_goal"]
        any_contact = any_contact or bool(info["agent_collision"].any())
        td = nxt
        if bool(nxt["done"].all()):
            break
    success = reached.float().mean().item()
    assert success > 0.95, f"PD controller under the CBF filter only docked {success:.2%}"
    assert not any_contact, "bounding-sphere contact occurred despite the filter"


def test_safety_obs_dim_derivation_and_rollout():
    env = Docking6DoFEnv(
        num_envs=4,
        n_agents=4,
        max_steps=30,
        enable_cbf=True,
        enable_safety_obs=True,
        seed=0,
        device="cpu",
    )
    assert env.obs_dim == 6 + 4
    rollout = env.rollout(max_steps=5)
    obs = rollout[(env.group, "observation")]
    assert obs.shape[-1] == 10
    assert torch.isfinite(obs).all()


# Docked neighbours: full-responsibility rows.
def test_static_neighbor_gets_full_pair_condition():
    flt = HOCBFSafetyFilter()
    pos = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    vel = torch.tensor([[[0.05, 0.0, 0.0], [0.0, 0.0, 0.0]]])  # agent 0 closing
    _, d_shared, _, _ = flt._translational_rows(pos, vel, static_mask=None)
    static = torch.tensor([[False, True]])  # agent 1 is docked/frozen
    _, d_static, _, _ = flt._translational_rows(pos, vel, static_mask=static)
    # The row of agent 0 against docked agent 1 has the full b (two times the
    # shared half). The rows of a docked agent are dropped (0.a <= 1).
    assert d_static[0, 0, 1] == pytest.approx(2.0 * float(d_shared[0, 0, 1]))
    assert float(d_static[0, 1, 0]) == 1.0 and float(d_static[0, 1, 1]) == 1.0


def test_closing_on_docked_agent_never_reaches_contact():
    env = Docking6DoFEnv(
        num_envs=2,
        n_agents=2,
        max_steps=400,
        linear_accel_scale=0.005,
        enable_cbf=True,
        enable_agent_collisions=False,
        init_agent_pos_means=[[5.0, -1.5, 0.0], [5.0, 1.5, 0.0]],
        init_agent_pos_ranges=[0.0, 0.0],
        seed=0,
        device="cpu",
    )
    td = env.reset()
    # Freeze agent 1 as if docked. The filter must treat it as an obstacle.
    env.on_goal_latched[:, 1] = True
    action = torch.zeros(2, 2, 3)
    action[:, 0, 1] = 1.0  # agent 0 accelerates straight at the frozen agent
    min_sep = float("inf")
    for _ in range(300):
        td.set((env.group, "action"), action)
        td = env.step(td)["next"]
        min_sep = min(min_sep, float((env.pos[:, 0] - env.pos[:, 1]).norm(dim=-1).min()))
    assert min_sep >= env.cbf_agent_keepout - 0.01, (
        f"frozen-neighbour keep-apart violated: min separation {min_sep:.3f} m"
    )
    assert min_sep < 0.9, "constraint never pressed - test is vacuous"


# Staged relaxation of soft rows.
def test_soft_rows_relaxed_before_backup_pair_rows_kept():
    flt = HOCBFSafetyFilter()
    bound = 0.005
    # A hard pair row needs full +x thrust and a soft row needs full -x thrust.
    # Together they are infeasible. The staged solve must drop the soft row and
    # satisfy the pair row.
    c_pair = torch.tensor([-1.0, 0.0, 0.0])
    d_pair = -0.9 * bound  # -a_x <= -0.9*bound => a_x >= 0.9*bound
    c_soft = torch.tensor([1.0, 0.0, 0.0])
    d_soft = -0.9 * bound  # a_x <= -0.9*bound
    rows_c = torch.stack([c_pair, c_soft], dim=0).unsqueeze(0)  # [1, 2, 3]
    rows_d = torch.tensor([[d_pair, d_soft]])  # [1, 2]
    a_nom = torch.zeros(1, 3)
    a_star, infeasible, n_relaxed, relaxed = flt._solve_translational(
        a_nom, rows_c, rows_d, soft_cols=[1], bound=bound
    )
    assert bool(relaxed[0]), "relaxed mask must flag the agent whose row was dropped"
    assert not bool(infeasible[0]), "relaxation should restore feasibility"
    assert n_relaxed == 1
    assert float(a_star[0, 0]) >= 0.9 * bound - 1e-9, "hard pair row must be satisfied"
    # With no soft rows, the same conflict is hard-infeasible.
    _, infeasible2, _, relaxed2 = flt._solve_translational(
        a_nom, rows_c, rows_d, soft_cols=[], bound=bound
    )
    assert bool(infeasible2[0])
    assert not bool(relaxed2[0])


# k-nearest-neighbour observations.
def test_two_neighbor_obs_dims_and_ordering():
    env = Docking6DoFEnv(
        num_envs=1,
        n_agents=3,
        max_steps=30,
        enable_neighbor_obs=True,
        neighbor_obs_count=2,
        init_agent_pos_means=[[5.0, 0.0, 0.0], [5.0, 1.0, 0.0], [5.0, 3.0, 0.0]],
        init_agent_pos_ranges=[0.0, 0.0, 0.0],
        seed=0,
        device="cpu",
    )
    assert env.obs_dim == 6 + 12
    env.reset()
    rel_pos, rel_vel = env._nearest_neighbor_features(2)
    assert rel_pos.shape == (1, 3, 2, 3)
    # Agent 0: nearest is agent 1 (dy=+1), then agent 2 (dy=+3).
    assert rel_pos[0, 0, 0, 1] == pytest.approx(1.0)
    assert rel_pos[0, 0, 1, 1] == pytest.approx(3.0)
    # Agent 2: nearest is agent 1 (dy=-2), then agent 0 (dy=-3).
    assert rel_pos[0, 2, 0, 1] == pytest.approx(-2.0)
    assert rel_pos[0, 2, 1, 1] == pytest.approx(-3.0)
    # More neighbours than exist gives zero padding.
    rel_pos3, _ = env._nearest_neighbor_features(3)
    assert rel_pos3.shape == (1, 3, 3, 3)
    assert (rel_pos3[:, :, 2] == 0).all()
    rollout = env.rollout(max_steps=3)
    assert rollout[(env.group, "observation")].shape[-1] == 18


# Filter solved once per control step.
def test_control_rate_passthrough_is_bit_identical():
    # A pass-through filter must give the unfiltered dynamics at any solve
    # interval.
    trajs = []
    for interval in (1, 10):
        env = Docking6DoFEnv(
            num_envs=8, n_agents=4, max_steps=40, enable_cbf=True,
            safety_filter=CBFSafetyFilter(), cbf_filter_substep_interval=interval,
            seed=0, device="cpu",
        )
        td = env.reset()
        torch.manual_seed(0)
        for _ in range(10):
            td.set((env.group, "action"), torch.rand(8, 4, 3) * 2 - 1)
            td = env.step(td)["next"]
        trajs.append(env.pos.clone())
    assert torch.equal(trajs[0], trajs[1])


def test_control_rate_still_prevents_inter_agent_contact():
    # With one solve per control step, the separation must stay above D_s.
    # The D_s margin covers the 1 s zero-order hold.
    env = Docking6DoFEnv(
        num_envs=4, n_agents=2, max_steps=400, linear_accel_scale=0.005,
        enable_cbf=True, enable_agent_collisions=False, cbf_agent_keepout=0.55,
        cbf_filter_substep_interval=10, dynamics_substeps=10,
        init_agent_pos_means=[[5.0, -1.6, 0.0], [5.0, 1.6, 0.0]],
        init_agent_pos_ranges=[0.0, 0.0], seed=0, device="cpu",
    )
    td = env.reset()
    action = torch.zeros(4, 2, 3)
    action[:, 0, 1] = 1.0
    action[:, 1, 1] = -1.0
    min_sep = float("inf")
    for _ in range(300):
        td.set((env.group, "action"), action)
        td = env.step(td)["next"]
        min_sep = min(min_sep, float((env.pos[:, 0] - env.pos[:, 1]).norm(dim=-1).min()))
    assert min_sep >= env.cbf_agent_keepout - 0.02, (
        f"control-rate keep-apart violated: min separation {min_sep:.3f} m"
    )
    assert min_sep < 0.9, "constraint never pressed - test is vacuous"


def test_default_filter_selection():
    env_off = Docking6DoFEnv(num_envs=2, n_agents=2, seed=0, device="cpu")
    assert type(env_off.safety_filter) is CBFSafetyFilter
    env_on = Docking6DoFEnv(num_envs=2, n_agents=2, enable_cbf=True, seed=0, device="cpu")
    assert isinstance(env_on.safety_filter, HOCBFSafetyFilter)
    # The filter mean motion must match the simulator.
    assert env_on.safety_filter.mean_motion == pytest.approx(env_on.mean_motion)


# Evaluation filter option: same filter as training.
def _filter_signature(flt: HOCBFSafetyFilter) -> dict:
    return {
        k: getattr(flt, k)
        for k in (
            "linear_accel_scale", "angular_accel_scale", "agent_keepout",
            "lambda1", "lambda2", "activation_radius",
            "speed_limit_intercept", "speed_limit_slope", "speed_limit_gain",
            "target_keepout_axes", "target_lambda1", "target_lambda2",
            "max_ang_speed", "ang_speed_gain", "mean_motion", "vel_eps",
        )
    }


def test_mc_eval_cbf_mode_yields_identical_filter_to_training():
    from marl_docking_6dof.tasks import MarlDocking6DoFTask
    from scripts.mc_eval_6dof import apply_cbf_mode

    train_overrides = {
        "n_agents": 4, "max_steps": 30, "enable_attitude": True,
        "enable_cbf": True, "enable_safety_obs": True,
        "linear_accel_scale": 0.005, "cbf_lambda1": 0.06,  # not the default
    }
    task_train = MarlDocking6DoFTask.DOCKING_6DOF.get_from_yaml()
    task_train.config.update(train_overrides)
    env_train = task_train.get_env_fun(
        num_envs=2, continuous_actions=True, seed=0, device="cpu"
    )()

    # Evaluation uses the same task config (from config.pkl).
    task_eval = MarlDocking6DoFTask.DOCKING_6DOF.get_from_yaml()
    task_eval.config.update(train_overrides)

    # auto: training setting, same filter parameters.
    assert apply_cbf_mode(task_eval, "auto") is True
    env_eval = task_eval.get_env_fun(
        num_envs=2, continuous_actions=True, seed=0, device="cpu"
    )()
    assert type(env_eval.safety_filter) is type(env_train.safety_filter)
    assert _filter_signature(env_eval.safety_filter) == _filter_signature(
        env_train.safety_filter
    )
    # The eval switch must not change enable_safety_obs (policy input size).
    assert env_eval.obs_dim == env_train.obs_dim

    # off: filter disabled. Everything else, including obs size, is unchanged.
    assert apply_cbf_mode(task_eval, "off") is False
    env_eval_off = task_eval.get_env_fun(
        num_envs=2, continuous_actions=True, seed=0, device="cpu"
    )()
    assert env_eval_off.enable_cbf is False
    assert env_eval_off.obs_dim == env_train.obs_dim

    # on, for a run trained without the filter: HOCBF filter forced on.
    task_plain = MarlDocking6DoFTask.DOCKING_6DOF.get_from_yaml()
    task_plain.config.update({**train_overrides, "enable_cbf": False,
                              "enable_safety_obs": False})
    assert apply_cbf_mode(task_plain, "on") is True
    env_forced = task_plain.get_env_fun(
        num_envs=2, continuous_actions=True, seed=0, device="cpu"
    )()
    assert isinstance(env_forced.safety_filter, HOCBFSafetyFilter)


# Keep-out row dropped: maximum outward thrust, not the policy command.
def test_keepout_drop_substitutes_outward_push():
    flt = HOCBFSafetyFilter()
    bound = 0.005
    # Row 0: pair row, slack. Row 1: speed row, slack. Row 2: keep-out row with
    # grad = +x. Row 2 alone is infeasible in the box (needs a_x >= 2*bound).
    c_pair = torch.tensor([0.0, 1.0, 0.0]); d_pair = 10.0
    c_speed = torch.tensor([0.0, 0.0, 1.0]); d_speed = 10.0
    c_keep = torch.tensor([-1.0, 0.0, 0.0]); d_keep = -2.0 * bound  # -a_x <= -2b
    rows_c = torch.stack([c_pair, c_speed, c_keep]).unsqueeze(0)
    rows_d = torch.tensor([[d_pair, d_speed, d_keep]])
    # The policy commands full thrust into the wall.
    a_nom = torch.tensor([[-bound, 0.0, 0.0]])
    a_star, infeasible, n_relaxed, relaxed = flt._solve_translational(
        a_nom, rows_c, rows_d, soft_cols=[1, 2], bound=bound, keepout_col=2
    )
    assert not bool(infeasible[0])
    assert bool(relaxed[0]) and n_relaxed == 1
    # The result must not be the nominal (-x). The re-solve gives full +x thrust
    # (outward = -sign(c_keep) = +x).
    assert float(a_star[0, 0]) >= bound - 1e-9, a_star
    # Without keepout_col, the nominal is returned.
    a_leg, _, _, _ = flt._solve_translational(
        a_nom, rows_c, rows_d, soft_cols=[1, 2], bound=bound, keepout_col=None
    )
    assert float(a_leg[0, 0]) <= -bound + 1e-9


# Keep-out exponent: protection of the corner ports on the 3x3 grid.
def _keepout_h(env, pos):
    n = env.cbf_target_keepout_exponent
    rho = torch.tensor(env.cbf_target_keepout_axes)
    y = (pos - env.target_center[0]) / rho
    return float((y**n).sum() - 1.0)


@pytest.mark.parametrize("n_exp,expect_covered", [(4, False), (8, True)])
def test_keepout_exponent_corner_port_coverage(n_exp, expect_covered):
    env = Docking6DoFEnv(
        num_envs=1, n_agents=9, max_steps=10, enable_cbf=True,
        cbf_target_keepout_exponent=n_exp, cbf_target_clearance=0.08,
        port_spread=0.55, seed=0, device="cpu",
    )
    env.reset()
    goals = env.goals[0]  # [9, 3]
    face_x = float(env.target_center[0, 0] + 0.5 * env.target_size[0])
    contact_x = face_x + 0.5 * float(env.agent_size[0])  # chaser -X face on the target face
    # Least protected port: the largest h_T at contact.
    worst = max(
        _keepout_h(env, torch.tensor([contact_x, float(g[1]), float(g[2])])) for g in goals
    )
    # Every docked pose must be in the safe set, so docking is possible.
    assert all(_keepout_h(env, g) > 0.0 for g in goals)
    if expect_covered:
        assert worst < 0.0, f"n={n_exp}: contact at some port has h_T={worst:+.3f} >= 0 (unprotected)"
    else:
        # n=4 does not protect the corner ports (h_T > 0 at contact).
        assert worst > 0.0


# Depth-weighted capture bonus.
def _capture_bonus(depth_weight, offset):
    env = Docking6DoFEnv(
        num_envs=1, n_agents=1, max_steps=5, goal_tolerance=0.1, max_docking_speed=0.1,
        reward_goal_bonus=40.0, reward_goal_bonus_depth_weight=depth_weight,
        reward_progress_scale=0.0, reward_time_penalty=0.0, reward_vel_scale=0.0,
        reward_capture_scale=0.0, collision_penalty_scale=0.0,
        init_agent_pos_means=[[5.0, 0.0, 0.0]], init_agent_pos_ranges=[0.0],
        seed=0, device="cpu",
    )
    td = env.reset()
    env.pos = env.goals.clone(); env.pos[..., 0] += offset; env.vel.zero_()
    td[(env.group, "action")] = torch.zeros(1, 1, env.action_dim)
    out = env.step(td)["next"]
    assert bool(out[(env.group, "info", "on_goal")][0, 0, 0])
    return float(out[(env.group, "reward")][0, 0, 0])


def test_depth_premium_never_lowers_the_legacy_bonus():
    # Flat bonus: the same at any point inside the capture ball.
    assert _capture_bonus(0.0, 0.0) == pytest.approx(40.0, abs=1e-4)
    assert _capture_bonus(0.0, 0.09) == pytest.approx(40.0, abs=1e-4)
    # w=0.5 adds a premium: an edge capture gives 40 and the centre gives 60.
    assert _capture_bonus(0.5, 0.09) == pytest.approx(40.0 * (1.0 + 0.5 * 0.1), abs=1e-3)
    assert _capture_bonus(0.5, 0.0) == pytest.approx(60.0, abs=1e-4)
    for off in (0.0, 0.03, 0.06, 0.09):
        assert _capture_bonus(0.5, off) >= _capture_bonus(0.0, off) - 1e-6


def test_cbf_relaxed_info_key_present_and_zero_when_benign():
    env = Docking6DoFEnv(
        num_envs=2, n_agents=2, max_steps=5, enable_cbf=True,
        init_agent_pos_means=[[6.0, -1.0, 0.0], [6.0, 1.0, 0.0]], init_agent_pos_ranges=[0.0, 0.0],
        seed=0, device="cpu",
    )
    td = env.reset()
    td[(env.group, "action")] = torch.zeros(2, 2, env.action_dim)
    out = env.step(td)["next"]
    rel = out[(env.group, "info", "cbf_relaxed")]
    assert rel.shape == (2, 2, 1)
    assert float(rel.abs().max()) == 0.0


# Row normalisation: very large rows must not give false infeasibility.
def test_huge_magnitude_row_is_not_spuriously_infeasible():
    """A feasible row scaled by 1e7 must give the same answer.
    This is the n=8 keep-out row at spawn range (gradient about 3e7).
    """
    flt = HOCBFSafetyFilter()
    bound = 0.005
    torch.manual_seed(0)
    for trial in range(25):
        c = torch.randn(1, 2, 3, dtype=torch.float32)
        # The box-clamped nominal violates row 0, so the slow KKT path runs.
        u_nom = torch.randn(1, 3) * bound
        d_small = (c * u_nom.unsqueeze(1)).sum(-1) - 0.3 * bound * c.norm(dim=-1)
        u_a, inf_a = flt._solve_box_qp(u_nom, c.clone(), d_small.clone(), bound)
        k = 3.0e7
        u_b, inf_b = flt._solve_box_qp(u_nom, c * k, d_small * k, bound)
        assert bool(inf_a[0]) == bool(inf_b[0]), (
            f"trial {trial}: scaling a row by {k:g} flipped feasibility "
            f"({bool(inf_a[0])} -> {bool(inf_b[0])})"
        )
        if not bool(inf_a[0]):
            assert torch.allclose(u_a, u_b, atol=1e-6), (
                f"trial {trial}: scaled row changed the solution {u_a} vs {u_b}"
            )


def test_row_normalisation_preserves_disabled_rows():
    """Disabled rows (c = 0, d = 1) must stay valid after normalisation."""
    flt = HOCBFSafetyFilter()
    bound = 0.005
    rows_c = torch.zeros(1, 2, 3)
    rows_d = torch.ones(1, 2)
    u_nom = torch.tensor([[bound, -bound, bound]])
    u, inf = flt._solve_box_qp(u_nom, rows_c, rows_d, bound)
    assert not bool(inf[0])
    assert torch.allclose(u, u_nom)


# Policy-noise annealing (PolicySharpening).
def _load_train_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "tb_sharp", "scripts/train_benchmarl_6dof.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_sigma_ceiling_anneals_and_floor_never_moves():
    m = _load_train_module()
    sh = m.PolicySharpening(start_frac=0.5, end_frac=0.9)
    assert sh._ceiling_for(0.0) == m.SIGMA_MAX
    assert sh._ceiling_for(0.5) == m.SIGMA_MAX          # holds through exploration
    assert sh._ceiling_for(0.9) == m.SIGMA_MAX_FINAL    # fully sharpened
    assert sh._ceiling_for(1.0) == m.SIGMA_MAX_FINAL    # clamped past the end
    mid = sh._ceiling_for(0.7)
    assert m.SIGMA_MAX_FINAL < mid < m.SIGMA_MAX, mid
    # Does not increase.
    prev = float("inf")
    for i in range(21):
        cur = sh._ceiling_for(i / 20)
        assert cur <= prev + 1e-12
        prev = cur


def test_bounded_sigma_respects_the_live_ceiling():
    m = _load_train_module()
    big = torch.tensor(30.0)   # saturates the sigmoid -> takes the ceiling
    small = torch.tensor(-30.0)  # -> takes the floor
    try:
        m._SIGMA_CEILING[0] = m.SIGMA_MAX
        assert float(m._bounded_sigma(big)) == pytest.approx(m.SIGMA_MAX, abs=1e-4)
        assert float(m._bounded_sigma(small)) == pytest.approx(m.SIGMA_MIN, abs=1e-4)
        # Anneal the ceiling. The floor does not change.
        m._SIGMA_CEILING[0] = m.SIGMA_MAX_FINAL
        assert float(m._bounded_sigma(big)) == pytest.approx(m.SIGMA_MAX_FINAL, abs=1e-4)
        assert float(m._bounded_sigma(small)) == pytest.approx(m.SIGMA_MIN, abs=1e-4)
        # 1/sigma_min^2 stays at most 100.
        assert 1.0 / m.SIGMA_MIN**2 == pytest.approx(100.0)
    finally:
        m._SIGMA_CEILING[0] = m.SIGMA_MAX


def test_mc_eval_registers_every_train_callback_for_unpickle():
    """config.pkl stores the training callbacks under __main__. All must load."""
    import __main__

    import scripts.mc_eval_6dof as mc
    import scripts.train_benchmarl_6dof as train_mod
    from benchmarl.experiment import Experiment
    from benchmarl.experiment.callback import Callback

    expected = {
        n
        for n in dir(train_mod)
        if isinstance(getattr(train_mod, n), type)
        and issubclass(getattr(train_mod, n), (Callback, Experiment))
    }
    assert "PolicySharpening" in expected and "GoalToleranceCurriculum" in expected
    saved = {n: getattr(__main__, n, None) for n in expected}
    try:
        for n in expected:
            if hasattr(__main__, n):
                delattr(__main__, n)
        mc._register_train_callbacks_for_unpickle()
        missing = [n for n in expected if not hasattr(__main__, n)]
        assert not missing, f"not registered for unpickle: {missing}"
    finally:
        for n, v in saved.items():
            if v is not None:
                setattr(__main__, n, v)


def test_sigma_ceiling_survives_save_and_restore():
    """A new process resets the module global. The ceiling must be restored."""
    m = _load_train_module()
    sh = m.PolicySharpening(start_frac=0.5, end_frac=0.9)
    try:
        m._SIGMA_CEILING[0] = 0.22
        sd = {}
        sh.on_state_dict(sd)
        assert sd["policy_sharpening"]["sigma_ceiling"] == pytest.approx(0.22)
        # New process: the global returns to the default.
        m._SIGMA_CEILING[0] = m.SIGMA_MAX
        sh.on_load_state_dict(sd)
        assert m._SIGMA_CEILING[0] == pytest.approx(0.22)
        # A checkpoint without this callback must still load.
        sh.on_load_state_dict({})
        assert m._SIGMA_CEILING[0] == pytest.approx(0.22)
    finally:
        m._SIGMA_CEILING[0] = m.SIGMA_MAX


# Terminal observation relative to the gate.
def test_terminal_obs_dim_and_gate_relative_scaling():
    kw = dict(
        num_envs=1, n_agents=1, max_steps=10, goal_tolerance=0.05,
        init_agent_pos_means=[[5.0, 0.0, 0.0]], init_agent_pos_ranges=[0.0],
        seed=0, device="cpu",
    )
    base = Docking6DoFEnv(**kw)
    zoom = Docking6DoFEnv(enable_terminal_obs=True, **kw)
    assert zoom.obs_dim == base.obs_dim + 3

    zoom.reset()
    # Put the agent a known distance before the goal along +x. The features use
    # rel_goal = goal - pos, so the x component is negative.
    for off in (0.05, 0.10):
        zoom.pos = zoom.goals.clone()
        zoom.pos[..., 0] += off
        obs = zoom._build_observation()
        base_x = float(obs[0, 0, 0])          # rel_goal_x / obs_pos_scale
        term_x = float(obs[0, 0, -3])         # tanh(rel_goal_x / (2*tol))
        assert base_x == pytest.approx(-off / zoom.obs_pos_scale, abs=1e-6)
        assert term_x == pytest.approx(
            math.tanh(-off / zoom.terminal_obs_scale), abs=1e-5
        )
        # Much higher resolution at the gate than the base feature.
        assert abs(term_x) > 20 * abs(base_x)


def test_terminal_obs_is_stationary_across_curriculum_steps():
    """The feature must not change when the curriculum tightens the gate.
    A feature scaled by goal_tolerance changes the policy input at each step.
    """
    env = Docking6DoFEnv(
        num_envs=1, n_agents=1, max_steps=10, goal_tolerance=0.15,
        enable_terminal_obs=True,
        init_agent_pos_means=[[5.0, 0.0, 0.0]], init_agent_pos_ranges=[0.0],
        seed=0, device="cpu",
    )
    env.reset()
    env.pos = env.goals.clone()
    env.pos[..., 0] += 0.05
    wide = float(env._build_observation()[0, 0, -3])
    env.goal_tolerance = 0.05          # curriculum tightens the gate
    tight = float(env._build_observation()[0, 0, -3])
    assert tight == pytest.approx(wide, abs=1e-9), (wide, tight)
    assert wide == pytest.approx(math.tanh(-0.05 / env.terminal_obs_scale), abs=1e-5)


def test_terminal_obs_scale_spans_the_whole_curriculum_range():
    """tanh(d/scale) must not saturate at any gate."""
    env = Docking6DoFEnv(
        num_envs=1, n_agents=1, max_steps=10, enable_terminal_obs=True,
        init_agent_pos_means=[[5.0, 0.0, 0.0]], init_agent_pos_ranges=[0.0],
        seed=0, device="cpu",
    )
    sc = env.terminal_obs_scale
    reads = [math.tanh(d / sc) for d in (0.02, 0.05, 0.10, 0.15)]
    # Increasing and not saturated, so each gate can be resolved.
    assert all(b > a for a, b in zip(reads, reads[1:])), reads
    assert reads[-1] < 0.99, reads
    # Much more sensitive at the tightest gate than the base feature.
    d = 0.05
    assert abs(math.tanh(d / sc)) > 20 * (d / env.obs_pos_scale)


def test_terminal_obs_off_by_default_preserves_layout():
    env = Docking6DoFEnv(num_envs=1, n_agents=2, max_steps=5, seed=0, device="cpu")
    assert env.enable_terminal_obs is False
    assert env.obs_dim == 6


# Near-goal respawn.
def _near_goal_env(fraction, radius=0.25, n_agents=9, num_envs=64):
    return Docking6DoFEnv(
        num_envs=num_envs, n_agents=n_agents, max_steps=10,
        enable_cbf=True, enable_attitude=True,
        init_near_goal_fraction=fraction, init_near_goal_radius=radius,
        port_spread=0.55, seed=0, device="cpu",
    )


def test_near_goal_starts_off_by_default():
    env = Docking6DoFEnv(num_envs=8, n_agents=4, max_steps=5, seed=0, device="cpu")
    assert env.init_near_goal_fraction == 0.0
    env.reset()
    # With the feature off, no agent spawns near a port.
    d = (env.pos - env.goals).norm(dim=-1)
    assert float(d.min()) > 0.5, float(d.min())


def test_near_goal_starts_place_roughly_the_requested_fraction():
    env = _near_goal_env(0.5)
    env.reset()
    d = (env.pos - env.goals).norm(dim=-1)
    frac = float((d <= env.init_near_goal_radius + 1e-6).float().mean())
    assert 0.35 < frac < 0.65, frac
    # About half of those are behind the port.
    near = d <= env.init_near_goal_radius + 1e-6
    behind = ((env.pos - env.goals)[..., 0] < 0) & near
    share = float(behind.sum()) / max(float(near.sum()), 1.0)
    assert 0.3 < share < 0.7, share


def test_near_goal_starts_never_violate_the_inter_agent_keepout():
    """Ports are far enough apart that the jitter cannot go below D_s."""
    env = _near_goal_env(1.0, radius=0.25)
    d_s = env.safety_filter.agent_keepout
    for _ in range(6):
        env.reset()
        sep = torch.cdist(env.pos, env.pos)
        sep = sep + torch.eye(env.n_agents).unsqueeze(0) * 99.0
        assert float(sep.min()) > d_s, (float(sep.min()), d_s)


def test_near_goal_starts_are_reproducible_under_the_env_seed():
    a = _near_goal_env(0.5); a.reset()
    b = _near_goal_env(0.5); b.reset()
    assert torch.allclose(a.pos, b.pos), "near-goal draws must use the seeded rng"


def test_mc_eval_forces_near_goal_starts_off_for_scoring():
    """The training aid must not change a reported success rate."""
    import inspect

    import scripts.mc_eval_6dof as mc

    src = inspect.getsource(mc.load_policy_and_task)
    assert 'task.config["init_near_goal_fraction"] = 0.0' in src
    assert "init_near_goal_fraction" in src


# Sensor noise (enable_sensor_noise).
def _noise_env(enable, **overrides):
    kw = dict(
        num_envs=16, n_agents=4, max_steps=10, enable_attitude=True,
        enable_neighbor_obs=True, neighbor_obs_count=2,
        init_agent_pos_means=[[8.0, 0.0, 0.0], [8.0, 1.0, 0.0], [8.0, 0.0, 1.0], [8.0, -1.0, -1.0]],
        init_agent_pos_ranges=[0.3] * 4,
        seed=0, device="cpu",
    )
    kw.update(overrides)
    return Docking6DoFEnv(enable_sensor_noise=enable, **kw)


def test_sensor_noise_off_by_default_matches_pre_feature_observation():
    """The default env must be the same as enable_sensor_noise=False and the same
    as the direct true-state functions.
    """
    kw = dict(
        num_envs=4, n_agents=2, max_steps=5, enable_attitude=True, seed=0, device="cpu",
        init_agent_pos_means=[[6.0, -1.0, 0.0], [6.0, 1.0, 0.0]], init_agent_pos_ranges=[0.0, 0.0],
    )
    default_env = Docking6DoFEnv(**kw)
    explicit_env = Docking6DoFEnv(enable_sensor_noise=False, **kw)
    td1 = default_env.reset(); td2 = explicit_env.reset()
    assert torch.allclose(
        td1[(default_env.group, "observation")], td2[(explicit_env.group, "observation")]
    )
    rel_goal_true = default_env._relative_goal_vector() / default_env.obs_pos_scale
    assert torch.allclose(
        td1[(default_env.group, "observation")][..., :3], rel_goal_true
    )


def test_sensor_noise_on_changes_observation_but_not_shape():
    env_off = _noise_env(False)
    env_on = _noise_env(True)
    obs_off = env_off.reset()[(env_off.group, "observation")]
    obs_on = env_on.reset()[(env_on.group, "observation")]
    assert obs_off.shape == obs_on.shape == (16, 4, env_off.obs_dim)
    assert not torch.allclose(obs_off, obs_on)
    assert torch.isfinite(obs_on).all()


def test_sensor_noise_does_not_change_obs_dim():
    """A checkpoint trained without noise must load with noise on."""
    kw = dict(num_envs=1, n_agents=2, max_steps=5, enable_attitude=True, seed=0, device="cpu")
    assert Docking6DoFEnv(enable_sensor_noise=False, **kw).obs_dim == \
        Docking6DoFEnv(enable_sensor_noise=True, **kw).obs_dim


def test_sensor_noise_never_affects_reward_or_truth():
    """With the same actions, reward, true state and capture must be the same with
    noise on and off. Noise is on the observation only.
    """
    torch.manual_seed(0)
    kw = dict(
        num_envs=8, n_agents=2, max_steps=15, enable_attitude=True,
        goal_tolerance=0.1, max_docking_speed=0.1,
        init_agent_pos_means=[[5.5, -0.3, 0.0], [5.5, 0.3, 0.0]],
        init_agent_pos_ranges=[0.1, 0.1], seed=0, device="cpu",
    )
    env_off = Docking6DoFEnv(enable_sensor_noise=False, **kw)
    env_on = Docking6DoFEnv(enable_sensor_noise=True, **kw)
    td_off = env_off.reset(); td_on = env_on.reset()
    actions = [torch.rand(8, 2, env_off.action_dim) * 2 - 1 for _ in range(15)]
    for act in actions:
        td_off[(env_off.group, "action")] = act.clone()
        td_on[(env_on.group, "action")] = act.clone()
        out_off = env_off.step(td_off)["next"]
        out_on = env_on.step(td_on)["next"]
        assert torch.allclose(env_off.pos, env_on.pos)
        assert torch.allclose(env_off.vel, env_on.vel)
        assert torch.allclose(out_off[(env_off.group, "reward")], out_on[(env_on.group, "reward")])
        assert torch.equal(
            out_off[(env_off.group, "info", "on_goal")], out_on[(env_on.group, "info", "on_goal")]
        )
        td_off, td_on = out_off, out_on


def test_sensor_noise_grows_with_range():
    """A far agent must get a noisier reading than a near agent."""
    env = _noise_env(True, num_envs=300, n_agents=2,
                      init_agent_pos_means=[[11.0, 0.0, 0.0], [5.05, 0.0, 0.0]],
                      init_agent_pos_ranges=[0.0, 0.0])
    env.reset()
    obs = env._build_observation()
    far_pos_obs = obs[:, 0, :3] * env.obs_pos_scale  # about 11 m from its port
    near_pos_obs = obs[:, 1, :3] * env.obs_pos_scale  # about 0.05 m from its port
    true_far = (env.goals[:, 0] - env.pos[:, 0]).norm(dim=-1)
    true_near = (env.goals[:, 1] - env.pos[:, 1]).norm(dim=-1)
    resid_far = (obs[:, 0, :3] * env.obs_pos_scale - (env.goals[:, 0] - env.pos[:, 0])).norm(dim=-1)
    resid_near = (obs[:, 1, :3] * env.obs_pos_scale - (env.goals[:, 1] - env.pos[:, 1])).norm(dim=-1)
    assert float(true_far.mean()) > float(true_near.mean())
    assert float(resid_far.std()) > float(resid_near.std())


def test_sensor_noise_bias_is_constant_within_an_episode_and_resamples_on_reset():
    """The bias must be constant in an episode and change after a reset."""
    env = _noise_env(True, num_envs=4, n_agents=1,
                      init_agent_pos_means=[[8.0, 0.0, 0.0]], init_agent_pos_ranges=[0.0])
    td = env.reset()
    bias0 = env._sensor_bias_pos.clone()
    act = torch.zeros(4, 1, env.action_dim)
    for _ in range(5):
        td[(env.group, "action")] = act
        td = env.step(td)["next"]
        assert torch.equal(env._sensor_bias_pos, bias0), "bias must not change within an episode"
    seen = [bias0.clone()]
    for _ in range(5):
        env.reset()
        seen.append(env._sensor_bias_pos.clone())
    assert any(not torch.equal(seen[0], b) for b in seen[1:]), "bias should resample across resets"


def test_sensor_dropout_holds_the_previous_reading():
    """With dropout probability 1, the reading must not change after the first step."""
    env = _noise_env(True, num_envs=6, n_agents=1, sensor_dropout_prob=1.0,
                      init_agent_pos_means=[[8.0, 0.0, 0.0]], init_agent_pos_ranges=[0.0])
    td = env.reset()
    first_obs = td[(env.group, "observation")][..., :3].clone()
    act = torch.rand(6, 1, env.action_dim) * 2 - 1
    for _ in range(5):
        td[(env.group, "action")] = act
        td = env.step(td)["next"]
        held = td[(env.group, "info")]  # only to advance the env
        assert torch.allclose(td[(env.group, "observation")][..., :3], first_obs), \
            "dropout_prob=1.0 must hold the reading fixed forever"


def test_sensor_dropout_zero_never_holds():
    """With dropout probability 0, the reading must follow the true state."""
    env = _noise_env(True, num_envs=6, n_agents=1, sensor_dropout_prob=0.0,
                      init_agent_pos_means=[[8.0, 0.0, 0.0]], init_agent_pos_ranges=[0.0])
    td = env.reset()
    act = torch.ones(6, 1, env.action_dim) * 0.5  # move, so the true state changes
    changed = 0
    prev = td[(env.group, "observation")][..., :3].clone()
    for _ in range(8):
        td[(env.group, "action")] = act
        td = env.step(td)["next"]
        cur = td[(env.group, "observation")][..., :3]
        if not torch.allclose(cur, prev):
            changed += 1
        prev = cur
    assert changed >= 6, f"expected the reading to change almost every step, changed {changed}/8"


def test_sensor_noise_gyro_has_no_dropout():
    """The gyro reading must not hold during vision dropout.
    With vision dropout = 1, ang_vel must still follow the true rate.
    """
    env = _noise_env(True, num_envs=8, n_agents=1, enable_attitude=True,
                      sensor_dropout_prob=1.0, sensor_angvel_noise_std=1e-9,
                      sensor_angvel_bias_std=0.0,
                      init_agent_pos_means=[[8.0, 0.0, 0.0]], init_agent_pos_ranges=[0.0])
    td = env.reset()
    act = torch.zeros(8, 1, env.action_dim)
    act[..., 3] = 1.0  # angular acceleration, so the true rate changes
    ang_slice = slice(9, 12)  # [rel_goal(3), vel(3), att_vec(3), ang_err(3), ...]
    prev = td[(env.group, "observation")][..., ang_slice].clone()
    moved = False
    for _ in range(5):
        td[(env.group, "action")] = act
        td = env.step(td)["next"]
        cur = td[(env.group, "observation")][..., ang_slice]
        if not torch.allclose(cur, prev, atol=1e-6):
            moved = True
        prev = cur
    assert moved, "gyro-derived ang_vel must keep updating even while the vision bundle is dropped"


def test_sensor_noise_bias_scale_zero_isolates_white_noise():
    """With bias std = 0, resets at the same pose must give a mean error near zero."""
    env = _noise_env(
        True, num_envs=2000, n_agents=1,
        sensor_pos_bias_std=0.0, sensor_vel_bias_std=0.0,
        sensor_att_bias_std=0.0, sensor_angvel_bias_std=0.0,
        init_agent_pos_means=[[8.0, 0.0, 0.0]], init_agent_pos_ranges=[0.0],
    )
    env.reset()
    obs = env._build_observation()
    resid = obs[:, 0, :3] * env.obs_pos_scale - (env.goals[:, 0] - env.pos[:, 0])
    assert float(resid.mean().abs()) < 0.001, "zero-bias residual should average near zero"


def test_mc_eval_sensor_noise_mode_helper():
    import scripts.mc_eval_6dof as mc

    class FakeTask:
        def __init__(self, trained):
            self.config = {"enable_sensor_noise": trained}

    t = FakeTask(True)
    assert mc.apply_sensor_noise_mode(t, "auto") is True
    assert t.config["enable_sensor_noise"] is True

    t = FakeTask(False)
    assert mc.apply_sensor_noise_mode(t, "on") is True
    assert t.config["enable_sensor_noise"] is True

    t = FakeTask(True)
    assert mc.apply_sensor_noise_mode(t, "off") is False
    assert t.config["enable_sensor_noise"] is False

    with pytest.raises(ValueError):
        mc.apply_sensor_noise_mode(FakeTask(False), "bogus")


# Target keep-out geometry.
def _paper_target_env(**overrides):
    """The 9-agent geometry of the paper, with the keep-out settings exposed."""
    cfg = dict(
        num_envs=2,
        n_agents=9,
        max_steps=10,
        enable_attitude=True,
        enable_cbf=True,
        enable_safety_obs=True,
        enable_neighbor_obs=True,
        cbf_enable_target_keepout=True,
        cbf_target_clearance=0.08,
        cbf_target_keepout_exponent=8,
        target_size_x=2.0,
        target_size_y=2.0,
        target_size_z=2.2,
        agent_size_x=0.10,
        agent_size_y=0.2263,
        agent_size_z=0.366,
        port_spread=0.55,
        docking_offset_x=0.2,
        seed=0,
        device="cpu",
    )
    cfg.update(overrides)
    env = Docking6DoFEnv(**cfg)
    env.reset()
    return env


def _keepout_h(env, points):
    """h_T = sum((p_k / rho_k)^n) - 1 for points relative to the target centre."""
    rho = torch.tensor(env.cbf_target_keepout_axes, dtype=torch.float64)
    n = env.cbf_target_keepout_exponent
    return ((points.double() / rho) ** n).sum(-1) - 1.0


def _contact_box_corner(env):
    """Corner of the chaser box at contact: half-depth along the approach axis,
    bounding-sphere radius laterally.
    """
    r_bound = 0.5 * float((env.agent_size**2).sum().sqrt())
    return 0.5 * env.target_size.double() + torch.tensor(
        [0.5 * float(env.agent_size[0]), r_bound, r_bound], dtype=torch.float64
    )


def test_default_keepout_leaves_the_contact_box_corner_uncovered():
    """At n=8 the corner of the contact box has h_T > 0, so the keep-out row does
    not engage there. The cover-box option fixes this.
    """
    env = _paper_target_env()
    assert _keepout_h(env, _contact_box_corner(env)) > 0.0


def test_cover_box_keepout_contains_contact_box_and_keeps_ports_reachable():
    """rho_i = max(half_i + clearance, contact_half_i * 3^(1/n)) at n=16.
    Two conditions must hold: the contact box is inside the keep-out
    (corner h_T <= 0), and every docked pose stays safe (h_T > 0) with a margin
    at least as large as the default.
    """
    base = _paper_target_env()
    env = _paper_target_env(
        cbf_target_keepout_exponent=16, cbf_target_keepout_cover_box=True
    )

    # The contact box is inside the keep-out. The corner is the worst point of
    # the box.
    assert _keepout_h(env, _contact_box_corner(env)) <= 0.0

    # Every port is still outside, with a larger margin. Only the lateral axes grow.
    ports = env.goals[0] - env.target_center[0]
    h_ports = _keepout_h(env, ports)
    assert bool((h_ports > 0.0).all())
    assert float(h_ports.min()) > float(_keepout_h(base, base.goals[0] - base.target_center[0]).min())

    # The keep-out only grows, and not along the approach axis.
    assert env.cbf_target_keepout_axes[0] == base.cbf_target_keepout_axes[0]
    for new, old in zip(env.cbf_target_keepout_axes, base.cbf_target_keepout_axes):
        assert new >= old


def test_cover_box_option_is_off_by_default():
    """The default geometry must not change."""
    env = _paper_target_env()
    assert env.cbf_target_keepout_cover_box is False
    # Default: half-depth along the approach axis, bounding-sphere radius
    # laterally, plus the clearance. The env enlarges the +X face for the 3x3
    # grid, so read y and z from the env.
    r_bound = 0.5 * (0.10**2 + 0.2263**2 + 0.366**2) ** 0.5
    face_half = 0.5 * float(env.target_size[1])
    assert env.cbf_target_keepout_axes == pytest.approx(
        (
            0.5 * 2.0 + 0.5 * 0.10 + 0.08,
            face_half + r_bound + 0.08,
            face_half + r_bound + 0.08,
        ),
        abs=1e-6,
    )
    # No growth factor: 3^(1/n) would change every axis.
    assert env.cbf_target_keepout_axes[0] == 0.5 * 2.0 + 0.5 * 0.10 + 0.08


# Fuel (delta-v) reward.
def test_fuel_penalty_prices_applied_delta_v_without_changing_dynamics():
    """The fuel term must be -(w_lin*dv_lin + w_ang*dv_ang). The trajectory must
    not change.
    """
    def rollout(fuel):
        torch.manual_seed(0)
        env = Docking6DoFEnv(
            num_envs=4,
            n_agents=3,
            max_steps=8,
            enable_attitude=True,
            enable_cbf=True,
            enable_safety_obs=True,
            enable_agent_collisions=True,
            reward_fuel_scale=1.0 if fuel else 0.0,
            reward_fuel_ang_scale=8.0 if fuel else 0.0,
            seed=0,
            device="cpu",
        )
        td = env.reset()
        torch.manual_seed(1)
        total = fuel_term = dv = adv = 0.0
        for _ in range(8):
            td.set((env.group, "action"), torch.rand(4, 3, 6) * 2 - 1)
            td = env.step(td)["next"]
            info = td[(env.group, "info")]
            total += float(td[(env.group, "reward")].mean())
            fuel_term += float(info["reward_fuel"].mean())
            dv += float(info["delta_v"].mean())
            adv += float(info["angular_delta_v"].mean())
        return total, fuel_term, dv, adv

    off_total, off_fuel, off_dv, off_adv = rollout(False)
    on_total, on_fuel, on_dv, on_adv = rollout(True)

    assert off_fuel == 0.0
    assert on_fuel == pytest.approx(-(1.0 * on_dv + 8.0 * on_adv), abs=1e-4)
    # Only the reward changes...
    assert on_total - off_total == pytest.approx(on_fuel, abs=1e-3)
    # ...and the trajectory is the same.
    assert on_dv == pytest.approx(off_dv, abs=1e-9)
    assert on_adv == pytest.approx(off_adv, abs=1e-9)


# Actuator uncertainty.
def _actuator_env(**overrides):
    cfg = dict(
        num_envs=2000, n_agents=1, max_steps=5, enable_attitude=True, enable_cbf=False,
        init_agent_pos_means=[[10.0, 0.0, 0.0]], init_agent_pos_ranges=[[1.5, 2.0, 2.0]],
        seed=0, device="cpu",
    )
    cfg.update(overrides)
    env = Docking6DoFEnv(**cfg)
    env.reset()
    return env


def test_actuator_uncertainty_off_is_a_bit_exact_no_op():
    env = _actuator_env()
    cmd = torch.randn(2000, 1, 3) * 1e-3
    assert torch.equal(env._apply_actuator_error(cmd, angular=False), cmd)
    assert torch.equal(env._apply_actuator_error(cmd, angular=True), cmd)


def test_actuator_misalignment_is_a_proper_rotation():
    """The misalignment must rotate the command without a change of magnitude.
    Only the scale-factor term can change the magnitude.
    """
    env = _actuator_env(enable_actuator_uncertainty=True)
    for R in (env._act_rot_lin, env._act_rot_ang):
        R = R.reshape(-1, 3, 3)
        eye = torch.eye(3).expand_as(R)
        assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5)
        assert torch.allclose(torch.linalg.det(R), torch.ones(R.shape[0]), atol=1e-5)


def test_actuator_error_magnitude_matches_the_declared_sigmas():
    """The scale spread is the root-sum-square of the per-episode and per-step terms."""
    env = _actuator_env(enable_actuator_uncertainty=True,
                        actuator_lin_scale_bias_std=0.10,
                        actuator_lin_scale_noise_std=0.05,
                        actuator_lin_align_std=1e-9)  # only the scale term
    u = torch.zeros(2000, 1, 3); u[..., 0] = 1.0
    out = env._apply_actuator_error(u, angular=False).norm(dim=-1)
    assert out.mean() == pytest.approx(1.0, abs=0.02)
    assert float(out.std()) == pytest.approx(math.hypot(0.10, 0.05), rel=0.15)


def test_actuator_bias_is_per_episode_and_jitter_is_per_step():
    """A calibration offset is constant in a flight. Valve repeatability changes
    each step.
    """
    u = torch.zeros(2000, 1, 3); u[..., 0] = 1.0
    fixed = _actuator_env(enable_actuator_uncertainty=True,
                          actuator_lin_scale_noise_std=0.0)
    assert torch.allclose(fixed._apply_actuator_error(u, angular=False),
                          fixed._apply_actuator_error(u, angular=False))
    jittery = _actuator_env(enable_actuator_uncertainty=True,
                            actuator_lin_scale_noise_std=0.05)
    assert not torch.allclose(jittery._apply_actuator_error(u, angular=False),
                              jittery._apply_actuator_error(u, angular=False))
