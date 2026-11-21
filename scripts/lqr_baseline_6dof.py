"""Collision-agnostic LQR fuel reference for the 6-DOF docking task.

The LQR ignores the other agents, so its delta-v is a lower bound for the
RL policy. The script uses the environment as the simulator, with the
filter and collision model off, so the dynamics are the same.

Translation: discrete LQR on the CW model, weights from Bryson's rule.
Attitude: quaternion PD, alpha = +kp*(2 q_err_vec) - kd*omega.
The proportional term has a plus sign because of the error convention of the
environment (q_err = q_des^-1 (x) q). A minus sign makes the attitude diverge.
Both outputs are saturated to the policy limits. Success uses the capture
test of the environment.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch


def cw_matrices(n: float, dt: float):
    A = np.zeros((6, 6))
    A[0, 3] = A[1, 4] = A[2, 5] = 1.0
    A[3, 0] = 3 * n * n; A[3, 4] = 2 * n
    A[4, 3] = -2 * n
    A[5, 2] = -n * n
    B = np.zeros((6, 3)); B[3, 0] = B[4, 1] = B[5, 2] = 1.0
    # Series discretisation. This is accurate at dt = 1 s and n ~ 7e-5.
    M = np.zeros((9, 9)); M[:6, :6] = A; M[:6, 6:] = B
    E = np.eye(9); T = np.eye(9)
    for k in range(1, 30):
        T = T @ (M * dt) / k
        E = E + T
    return E[:6, :6], E[:6, 6:]


def dlqr(Ad, Bd, Q, R, iters=20000, tol=1e-11):
    P = Q.copy()
    for _ in range(iters):
        BtP = Bd.T @ P
        K = np.linalg.solve(R + BtP @ Bd, BtP @ Ad)
        Pn = Q + Ad.T @ P @ (Ad - Bd @ K)
        Pn = 0.5 * (Pn + Pn.T)
        if np.max(np.abs(Pn - P)) < tol:
            P = Pn; break
        P = Pn
    BtP = Bd.T @ P
    return np.linalg.solve(R + BtP @ Bd, BtP @ Ad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Checkpoint whose task configuration is used "
                         "(geometry, spawn and capture test).")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--goal-tolerance", type=float, default=0.05)
    ap.add_argument("--attitude-gate", type=float, default=None,
                    help="Attitude capture gate (rad). Default: the config "
                         "value (0.175). Use 0.0349 for the 2 deg gate of the"
                         " paper.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--kp-att", type=float, default=0.02)
    ap.add_argument("--kd-att", type=float, default=0.25)
    ap.add_argument("--r-max", type=float, default=12.0,
                    help="Bryson scale for position (m): the operating range,"
                         " not the docking tolerance.")
    ap.add_argument("--v-max", type=float, default=0.5,
                    help="Bryson scale for velocity (m/s): maximum cruise "
                         "speed.")
    args = ap.parse_args()

    from scripts.mc_eval_6dof import load_policy_and_task
    _, task = load_policy_and_task(args.checkpoint)
    c = task.config
    tol = float(args.goal_tolerance)
    c["goal_tolerance"] = tol
    if args.attitude_gate is not None:
        c["max_attitude_error"] = float(args.attitude_gate)
    att_gate = float(c.get("max_attitude_error", 0.175))
    print(f"[LQR] scoring gate: {tol} m, attitude {att_gate:.4f} rad "
          f"({att_gate * 180 / 3.141592653589793:.2f} deg)")
    c["enable_cbf"] = False                 # collision-agnostic reference
    c["enable_agent_collisions"] = False
    c["agent_collision_terminal_on_contact"] = False
    c["init_near_goal_fraction"] = 0.0      # score on the real far-spawn task

    dt = float(c["dt"]); a_bar = float(c["linear_accel_scale"])
    al_bar = float(c["angular_accel_scale"]); T = int(c["max_steps"])

    probe = task.get_env_fun(num_envs=1, continuous_actions=True, seed=0, device="cpu")()
    n_mm = float(probe.mean_motion); N = probe.n_agents
    Ad, Bd = cw_matrices(n_mm, dt)

    # Bryson's rule: weight each state by 1/(max acceptable)^2 and each input
    # by 1/u_max^2. Scale the velocity block up to make the arrival slow.
    q_pos = 1.0 / args.r_max**2
    q_vel = 1.0 / args.v_max**2
    Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])
    R = np.eye(3) * (1.0 / a_bar**2)
    K = dlqr(Ad, Bd, Q, R)
    print(f"[LQR] mean motion {n_mm:.3e} rad/s, dt {dt} s, |a|<= {a_bar} m/s^2")
    print(f"[LQR] gain matrix K:\n{np.array2string(K, precision=4)}")

    tot = succ = 0
    dv_all, adv_all, agent_dock = [], [], []
    rounds = max(1, args.episodes // args.batch_size)
    for r in range(rounds):
        B = args.batch_size
        env = task.get_env_fun(num_envs=B, continuous_actions=True,
                               seed=args.seed + r, device="cpu")()
        env.goal_tolerance = tol
        env.max_attitude_error = att_gate
        td = env.reset()
        Kt = torch.tensor(K, dtype=torch.float32)
        active = torch.ones(B, dtype=torch.bool)
        docked = torch.zeros(B, N, dtype=torch.bool)
        dv = torch.zeros(B, N); adv = torch.zeros(B, N)
        with torch.no_grad():
            for _ in range(T):
                err = torch.cat([env.pos - env.goals, env.vel], dim=-1)     # [B,N,6]
                a_cmd = -(err @ Kt.T)                                        # [B,N,3]
                a_cmd = a_cmd.clamp(-a_bar, a_bar)
                att_vec, ang_err = env._attitude_error_terms()
                al_cmd = (args.kp_att * att_vec - args.kd_att * ang_err).clamp(-al_bar, al_bar)
                act = torch.cat([a_cmd / a_bar, al_cmd / al_bar], dim=-1).clamp(-1, 1)
                td[(env.group, "action")] = act
                nxt = env.step(td)["next"]; info = nxt[(env.group, "info")]
                m = active.unsqueeze(-1)
                docked |= info["on_goal"][..., 0].bool() & m
                dv += info["delta_v"][..., 0] * m
                adv += info["angular_delta_v"][..., 0] * m
                active &= ~(nxt["done"][..., 0].bool() & active)
                td = nxt
                if not active.any():
                    break
        tot += B; succ += int(docked.all(1).sum())
        dv_all.append(dv); adv_all.append(adv); agent_dock.append(docked)
        print(f"[LQR] round {r+1}/{rounds}: {tot} episodes", flush=True)

    dv = torch.cat(dv_all); adv = torch.cat(adv_all); dk = torch.cat(agent_dock)
    print("\n" + "=" * 68)
    print(f"COLLISION-AGNOSTIC LQR REFERENCE ({tot} episodes, {N} agents, tol {tol} m, "
          f"attitude {att_gate * 180 / 3.141592653589793:.2f} deg)")
    print("=" * 68)
    print(f"  all agents docked      : {succ}/{tot} = {succ/tot*100:.1f}%")
    print(f"  per-agent capture rate : {dk.float().mean()*100:.2f}%")
    print(f"  linear  dv per agent   : mean {dv.mean():.4f}  median {dv.median():.4f} m/s")
    print(f"  linear  dv per episode : mean {dv.sum(1).mean():.4f} m/s (sum of {N})")
    print(f"  angular dv per agent   : mean {adv.mean():.4f} rad/s")
    if dk.all(1).any():
        print(f"  linear dv, successful episodes only: {dv[dk.all(1)].mean():.4f} m/s")
    print("\n  Neighbours are ignored. This is a lower bound on fuel.")


if __name__ == "__main__":
    main()
