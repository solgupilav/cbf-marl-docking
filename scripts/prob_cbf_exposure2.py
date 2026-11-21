"""Extended exposure log to check the Gaussian contact model.

Same configuration as prob_cbf_exposure.py. For each close pair-step it also
records:
  pair ids (i, j)
  vision dropout age of each agent and the dead-reckoned estimate
  persistent errors along the line of sight:
    x_p  position-bias part of the separation error
    x_v  velocity-bias part of the closing-speed error
    x_a  actuation error of the relative acceleration (no jitter)
  filter activity and hard infeasibility of each agent
  true and estimated closing speed

Usage:
  python scripts/prob_cbf_exposure2.py --seeds 0,1,2,3,4 --out results/probabilistic_safety/exposure2
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import mc_eval_6dof as mc  # noqa: E402
from prob_cbf_exposure import configure  # noqa: E402
from torchrl.envs.utils import ExplorationType, set_exploration_type  # noqa: E402

COLS = ["ep", "step", "i", "j", "d", "dh", "dhp", "sig", "sigp", "marg",
        "stale_i", "stale_j", "age_i", "age_j", "rho_i", "rho_j",
        "x_p", "x_v", "x_a", "ddot", "ddot_hat", "act_i", "act_j", "inf_i", "inf_j", "Eq"]
EPCOLS = ["S_los_all", "S_los_prop_all", "n_pairsteps", "infeas_steps", "min_d", "contact", "tcontact", "stale_steps"]


def phi_cdf(x):
    return torch.special.ndtr(x)


def run_seed(ckpt, episodes, batch, seed, actuator, near, noise_scale):
    policy, task = mc.load_policy_and_task(ckpt)
    configure(task, actuator, noise_scale)
    c = task.config
    sbp, sp0, kp = float(c["sensor_pos_bias_std"]), float(c["sensor_pos_noise_floor"]), float(c["sensor_pos_noise_per_m"])
    sbv, sv0, kv = float(c["sensor_vel_bias_std"]), float(c["sensor_vel_noise_floor"]), float(c["sensor_vel_noise_per_m"])
    a_bar = float(c.get("linear_accel_scale", 0.005))
    if bool(c.get("enable_actuator_uncertainty", False)):
        g_max = 1.0 + 5.0 * float(c["actuator_lin_scale_bias_std"]) + 5.0 * float(c["actuator_lin_scale_noise_std"])
    else:
        g_max = 1.0
    a_max = math.sqrt(3.0) * a_bar * g_max                  # per agent, m/s^2
    dt = 1.0
    delta_between = 2.0 * a_max * dt * dt / 8.0             # inter-sample dip bound for a pair
    D_s = float(c.get("cbf_agent_keepout", 0.55))
    n = int(c["n_agents"])
    T = int(c["max_steps"])
    iu = torch.triu_indices(n, n, 1)
    I, J = iu[0], iu[1]

    rows, eprows = [], []
    remaining, rnd = episodes, 0
    while remaining > 0:
        b = min(batch, remaining)
        env = task.get_env_fun(num_envs=b, continuous_actions=True, seed=seed + rnd, device="cpu")()
        group = env.group
        orig_mask = env._sensor_dropout_mask
        env._last_own_drop = torch.zeros(b, n, 1, dtype=torch.bool)

        def patched(shape, _orig=orig_mask, _env=env, _b=b, _n=n):
            m = _orig(shape)
            if tuple(shape) == (_b, _n, 1):
                _env._last_own_drop = m.clone()
            return m

        env._sensor_dropout_mask = patched
        sf = env.safety_filter
        orig_filter = sf.filter_action
        env._last_safe = torch.zeros(b, n, 6)

        def pfilter(*a, _orig=orig_filter, _env=env, **k):
            out = _orig(*a, **k)
            _env._last_safe = out[0].detach().clone()
            return out

        sf.filter_action = pfilter
        r_c = float(2.0 * 0.5 * torch.linalg.norm(env.agent_size))
        td = env.reset()

        active = torch.ones(b, dtype=torch.bool)
        age = torch.zeros(b, n)
        S_all = torch.zeros(b, dtype=torch.float64)
        S_prop = torch.zeros(b, dtype=torch.float64)
        n_ps = torch.zeros(b, dtype=torch.long)
        infeas = torch.zeros(b, dtype=torch.long)
        stale_steps = torch.zeros(b, dtype=torch.long)
        min_d = torch.full((b,), float("inf"))
        contact = torch.zeros(b, dtype=torch.bool)
        tcontact = torch.zeros(b, dtype=torch.bool)

        with set_exploration_type(ExplorationType.DETERMINISTIC):
            for step in range(T):
                td = policy(td)
                safe = env._last_safe.clone()               # command applied during this step
                nxt = env.step(td)["next"]
                pos, vel = env.pos.detach(), env.vel.detach()
                nav, navv = env._nav_pos.detach(), env._nav_vel.detach()
                rho = (env.goals - pos).norm(dim=-1)        # [B, N]
                drop = env._last_own_drop[..., 0]
                age = torch.where(drop, age + 1.0, torch.zeros_like(age))

                s2p = sbp**2 + (sp0 + kp * rho) ** 2         # per agent, per axis
                s2v = sbv**2 + (sv0 + kv * rho) ** 2
                tau = age * dt
                s2prop = s2p + tau**2 * s2v
                marg_a = 0.5 * a_max * tau**2
                prop = nav + navv * tau.unsqueeze(-1)

                dp = pos[:, I] - pos[:, J]
                d = dp.norm(dim=-1)
                uh = dp / d.clamp(min=1e-9).unsqueeze(-1)
                dph = nav[:, I] - nav[:, J]
                dh = dph.norm(dim=-1)
                dhp = (prop[:, I] - prop[:, J]).norm(dim=-1)
                sig = torch.sqrt(s2p[:, I] + s2p[:, J])
                sigp = torch.sqrt(s2prop[:, I] + s2prop[:, J])
                marg = marg_a[:, I] + marg_a[:, J] + delta_between

                eb = -env._sensor_bias_pos                  # bias part of the position error
                x_p = (uh * (eb[:, I] - eb[:, J])).sum(-1)
                bv = env._sensor_bias_vel
                x_v = (uh * (bv[:, I] - bv[:, J])).sum(-1)
                u_cmd = safe[..., :3].clamp(-1, 1) * a_bar
                if bool(c.get("enable_actuator_uncertainty", False)):
                    da = torch.matmul(env._act_rot_lin, ((1.0 + env._act_scale_lin) * u_cmd).unsqueeze(-1)).squeeze(-1) - u_cmd
                else:
                    da = torch.zeros_like(u_cmd)
                x_a = (uh * (da[:, I] - da[:, J])).sum(-1)
                dv = vel[:, I] - vel[:, J]
                ddot = (uh * dv).sum(-1)
                dvh = navv[:, I] - navv[:, J]
                ddot_hat = (dph * dvh).sum(-1) / dh.clamp(min=1e-9)
                interv = env.last_cbf_intervention[..., 0]
                infe = env.last_cbf_infeasible[..., 0]
                Eq = 3.0 * (s2v[:, I] + s2v[:, J])

                a = active.unsqueeze(1)
                S_all += (phi_cdf(-(dh - r_c) / sig).double() * a).sum(1)
                S_prop += (phi_cdf(-(dhp - (r_c + marg)) / sigp).double() * a).sum(1)
                n_ps += active.long() * d.shape[1]
                stale_steps += (drop.any(1) & active).long()
                info = nxt[(group, "info")]
                if "cbf_infeasible" in info.keys():
                    inf_now = (info["cbf_infeasible"][..., 0] > 0).any(1)
                    infeas += (inf_now & active).long()
                min_d = torch.minimum(min_d, torch.where(active, d.min(1).values, torch.full_like(min_d, float("inf"))))
                ac = mc.discrete_agent_contact(env) & a
                contact |= ac.any(1)

                m = ((dh < near) | (dhp < near) | (d < near)) & a
                if m.any():
                    e_idx, p_idx = m.nonzero(as_tuple=True)
                    ii, jj = I[p_idx], J[p_idx]
                    cols = [
                        (rnd * batch + e_idx).float(), torch.full_like(e_idx, step).float(), ii.float(), jj.float(),
                        d[e_idx, p_idx], dh[e_idx, p_idx], dhp[e_idx, p_idx], sig[e_idx, p_idx], sigp[e_idx, p_idx],
                        marg[e_idx, p_idx], drop[e_idx, ii].float(), drop[e_idx, jj].float(),
                        age[e_idx, ii], age[e_idx, jj], rho[e_idx, ii], rho[e_idx, jj],
                        x_p[e_idx, p_idx], x_v[e_idx, p_idx], x_a[e_idx, p_idx],
                        ddot[e_idx, p_idx], ddot_hat[e_idx, p_idx],
                        (interv[e_idx, ii] > 1e-6).float(), (interv[e_idx, jj] > 1e-6).float(),
                        (infe[e_idx, ii] > 0).float(), (infe[e_idx, jj] > 0).float(), Eq[e_idx, p_idx],
                    ]
                    rows.append(torch.stack([x.float() for x in cols], 1).numpy())

                tc = mc.discrete_target_contact(env) & a
                tcontact |= tc.any(1)
                done = (nxt["done"][..., 0].bool() | ac.any(1) | tc.any(1)) & active
                active &= ~done
                td = nxt
                if not active.any():
                    break

        eprows.append(np.stack([S_all.numpy(), S_prop.numpy(), n_ps.numpy(), infeas.numpy(), min_d.numpy(),
                                contact.numpy().astype(float), tcontact.numpy().astype(float),
                                stale_steps.numpy()], 1))
        remaining -= b
        rnd += 1
        print(f"  round {rnd}: {episodes - remaining}/{episodes}", flush=True)

    meta = dict(r_c=r_c, D_s=D_s, sbp=sbp, sp0=sp0, kp=kp, sbv=sbv, sv0=sv0, kv=kv, a_max=a_max,
                delta_between=delta_between, dropout=float(c["sensor_dropout_prob"]), n_agents=n,
                lam1=float(c.get("cbf_lambda1", 0.10)), lam2=float(c.get("cbf_lambda2", 0.15)),
                act_sb=float(c.get("actuator_lin_scale_bias_std", 0.0)) if bool(c.get("enable_actuator_uncertainty", False)) else 0.0,
                act_al=float(c.get("actuator_lin_align_std", 0.0)) if bool(c.get("enable_actuator_uncertainty", False)) else 0.0)
    return np.concatenate(eprows).astype(np.float64), np.concatenate(rows).astype(np.float32), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--seed", type=int, default=24680)
    ap.add_argument("--actuator", default="typical")
    ap.add_argument("--near", type=float, default=1.2)
    ap.add_argument("--out", default="results/probabilistic_safety/exposure2")
    ap.add_argument("--noise-scale", type=float, default=1.0)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    for s in [int(x) for x in a.seeds.split(",")]:
        ck = json.load(open(ROOT / f"results/main/seed{s}.json"))["checkpoint"]
        print(f"=== seed {s}: {ck}", flush=True)
        ep, rows, meta = run_seed(str(ROOT / ck), a.episodes, a.batch, a.seed, a.actuator, a.near, a.noise_scale)
        meta.update(noise_scale=a.noise_scale, seed=s, cols=COLS, epcols=EPCOLS)
        np.savez_compressed(out / f"s{s}.npz", ep=ep, rows=rows, meta=json.dumps(meta))
        print(f"  saved {out / f's{s}.npz'}  episodes={len(ep)} rows={len(rows)} "
              f"contacts={int(ep[:, 5].sum())} min_d={ep[:, 4].min():.4f}", flush=True)
    (out / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
