"""Log the inter-agent exposure at each step for the probabilistic safety
analysis. Uses the reported Monte Carlo configuration.

For each step and agent pair it records:
  d      true separation        ||p_i - p_j||
  dhat   estimated separation   ||p_hat_i - p_hat_j||
  sigma  1-sigma error of the relative position per axis:
         sigma^2 = s_i^2 + s_j^2, s_k^2 = sigma_b^2 + (sigma0 + kappa*rho_k)^2

It sums the per-step violation probabilities (Luo et al. 2020) over pairs
and steps:
  P_los = Phi(-(dhat - r_c)/sigma)
  P_chi = P(||e|| > dhat - r_c), with ||e||/sigma ~ chi_3
Samples with dhat < 1.0 m are saved to test the Gaussian error model.
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
from torchrl.envs.utils import ExplorationType, set_exploration_type  # noqa: E402


def phi(x: torch.Tensor) -> torch.Tensor:
    return torch.special.ndtr(x)


def chi3_sf(k: torch.Tensor) -> torch.Tensor:
    """P(chi_3 > k) = erfc(k/sqrt2) + sqrt(2/pi) k exp(-k^2/2), k >= 0."""
    k = k.clamp(min=0.0)
    return torch.special.erfc(k / math.sqrt(2.0)) + math.sqrt(2.0 / math.pi) * k * torch.exp(-0.5 * k * k)


NOISE_KEYS = ("sensor_pos_noise_floor", "sensor_pos_noise_per_m", "sensor_vel_noise_floor",
              "sensor_vel_noise_per_m", "sensor_att_noise_floor", "sensor_att_noise_per_m",
              "sensor_angvel_noise_std", "sensor_pos_bias_std", "sensor_vel_bias_std",
              "sensor_att_bias_std", "sensor_angvel_bias_std")


def configure(task, actuator: str, noise_scale: float = 1.0):
    task.config["goal_tolerance"] = 0.04
    task.config["max_attitude_error"] = 0.0349
    mc.apply_cbf_mode(task, "on")
    mc.apply_sensor_noise_mode(task, "on")
    mc.apply_keepout_mode(task, "off")
    task.config["cbf_target_keepout_mode"] = "halfspace"
    mc.apply_actuator_mode(task, actuator)
    task.config["cbf_use_sensed_state"] = True
    task.config["cbf_target_lambda1"] = 0.04
    task.config["cbf_target_lambda2"] = 0.04
    if noise_scale != 1.0:
        # Scale all sensing errors (noise, range slope and bias).
        # The dropout probability does not change.
        for k in NOISE_KEYS:
            task.config[k] = float(task.config[k]) * noise_scale
        print(f"[Config] Sensor noise scaled x{noise_scale:g} (dropout unchanged)", flush=True)


def run_seed(ckpt: str, episodes: int, batch: int, seed: int, actuator: str, near: float, noise_scale: float = 1.0):
    policy, task = mc.load_policy_and_task(ckpt)
    configure(task, actuator, noise_scale)
    c = task.config
    s_b = float(c["sensor_pos_bias_std"])
    s0 = float(c["sensor_pos_noise_floor"])
    kap = float(c["sensor_pos_noise_per_m"])
    D_s = float(c.get("cbf_agent_keepout", 0.55))
    n = int(c["n_agents"])
    T = int(c["max_steps"])
    iu = torch.triu_indices(n, n, 1)

    ep_rows, near_rows = [], []
    remaining, rnd = episodes, 0
    while remaining > 0:
        b = min(batch, remaining)
        env = task.get_env_fun(num_envs=b, continuous_actions=True, seed=seed + rnd, device="cpu")()
        group = env.group
        # Record the vision dropout mask of the own state (the (B, N, 1) call).
        orig_mask = env._sensor_dropout_mask
        env._last_own_drop = torch.zeros(b, n, 1, dtype=torch.bool)

        def patched(shape, _orig=orig_mask, _env=env, _b=b, _n=n):
            m = _orig(shape)
            if tuple(shape) == (_b, _n, 1):
                _env._last_own_drop = m.clone()
            return m

        env._sensor_dropout_mask = patched
        r_c = float(2.0 * 0.5 * torch.linalg.norm(env.agent_size))
        td = env.reset()

        active = torch.ones(b, dtype=torch.bool)
        S_rc_los = torch.zeros(b, dtype=torch.float64)
        S_rc_chi = torch.zeros(b, dtype=torch.float64)
        S_ds_los = torch.zeros(b, dtype=torch.float64)
        n_below_ds = torch.zeros(b, dtype=torch.long)      # observed pair-steps with d < D_s
        n_hat_below_ds = torch.zeros(b, dtype=torch.long)  # pair-steps with dhat < D_s
        n_pairsteps = torch.zeros(b, dtype=torch.long)
        n_near = torch.zeros(b, dtype=torch.long)          # pair-steps with dhat < D_s + 0.1
        infeas_steps = torch.zeros(b, dtype=torch.long)
        min_d = torch.full((b,), float("inf"))
        contact = torch.zeros(b, dtype=torch.bool)
        tcontact = torch.zeros(b, dtype=torch.bool)

        with set_exploration_type(ExplorationType.DETERMINISTIC):
            for step in range(T):
                td = policy(td)
                nxt = env.step(td)["next"]
                pos = env.pos.detach()
                nav = env._nav_pos.detach()
                goals = env.goals.detach()
                d = (pos[:, iu[0]] - pos[:, iu[1]]).norm(dim=-1).double()
                dh = (nav[:, iu[0]] - nav[:, iu[1]]).norm(dim=-1).double()
                rho = (goals - pos).norm(dim=-1).double()                 # [B, N]
                s2 = s_b ** 2 + (s0 + kap * rho) ** 2                      # per agent, per axis
                sig = torch.sqrt(s2[:, iu[0]] + s2[:, iu[1]])              # [B, P]
                drop = env._last_own_drop[..., 0]
                stale = drop[:, iu[0]] | drop[:, iu[1]]
                a = active.unsqueeze(1).double()

                S_rc_los += (phi(-(dh - r_c) / sig) * a).sum(1)
                S_rc_chi += (chi3_sf((dh - r_c) / sig) * a).sum(1)
                S_ds_los += (phi(-(dh - D_s) / sig) * a).sum(1)
                n_below_ds += ((d < D_s) & active.unsqueeze(1)).sum(1)
                n_hat_below_ds += ((dh < D_s) & active.unsqueeze(1)).sum(1)
                n_near += ((dh < D_s + 0.1) & active.unsqueeze(1)).sum(1)
                n_pairsteps += active.long() * d.shape[1]
                info = nxt[(group, "info")]
                if "cbf_infeasible" in info.keys():
                    inf_now = (info["cbf_infeasible"][..., 0] > 0).any(1)
                    infeas_steps += (inf_now & active).long()
                min_d = torch.minimum(min_d, torch.where(active, d.min(1).values.float(),
                                                        torch.full_like(min_d, float("inf"))))
                ac = mc.discrete_agent_contact(env) & active.unsqueeze(1)
                contact |= ac.any(1)

                m = (dh < near) & active.unsqueeze(1)
                if m.any():
                    e_idx, p_idx = m.nonzero(as_tuple=True)
                    near_rows.append(np.stack([
                        d[e_idx, p_idx].numpy(), dh[e_idx, p_idx].numpy(), sig[e_idx, p_idx].numpy(),
                        stale[e_idx, p_idx].numpy().astype(float),
                        (rnd * batch + e_idx).numpy().astype(float),
                        np.full(len(e_idx), step, float),
                    ], 1))

                tc = mc.discrete_target_contact(env) & active.unsqueeze(1)
                tcontact |= tc.any(1)
                done = (nxt["done"][..., 0].bool() | ac.any(1) | tc.any(1)) & active
                active &= ~done
                td = nxt
                if not active.any():
                    break

        ep_rows.append(np.stack([
            S_rc_los.numpy(), S_rc_chi.numpy(), S_ds_los.numpy(), n_below_ds.numpy(),
            n_hat_below_ds.numpy(), n_near.numpy(), n_pairsteps.numpy(), infeas_steps.numpy(),
            min_d.numpy(), contact.numpy().astype(float), tcontact.numpy().astype(float)], 1))
        remaining -= b
        rnd += 1
        print(f"  round {rnd}: {episodes - remaining}/{episodes}", flush=True)

    meta = dict(r_c=r_c, D_s=D_s, sigma_b=s_b, sigma0=s0, kappa=kap, n_agents=n, max_steps=T,
                dropout=float(c["sensor_dropout_prob"]))
    return np.concatenate(ep_rows), np.concatenate(near_rows), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--seed", type=int, default=24680)
    ap.add_argument("--actuator", default="typical")
    ap.add_argument("--near", type=float, default=1.0)
    ap.add_argument("--out", default="results/probabilistic_safety/exposure")
    ap.add_argument("--noise-scale", type=float, default=1.0)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    for s in [int(x) for x in a.seeds.split(",")]:
        ck = json.load(open(ROOT / f"results/main/seed{s}.json"))["checkpoint"]
        print(f"=== seed {s}: {ck}", flush=True)
        ep, near, meta = run_seed(str(ROOT / ck), a.episodes, a.batch, a.seed, a.actuator, a.near, a.noise_scale)
        meta["noise_scale"] = a.noise_scale
        np.savez(out / f"s{s}.npz", ep=ep, near=near, **{k: np.array(v) for k, v in meta.items()})
        print(f"  saved {out / f's{s}.npz'}  episodes={len(ep)} near-samples={len(near)} "
              f"contacts={int(ep[:, 9].sum())} min_d={ep[:, 8].min():.4f}", flush=True)
    (out / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
