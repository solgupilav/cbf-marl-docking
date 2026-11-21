"""Compute the numbers in the probabilistic part of the Discussion (Eq. 45 and
the two paragraphs after it) from the stored pair-encounter logs.

Input:  results/probabilistic_safety/exposure2/s*.npz (from prob_cbf_exposure2.py)
Output: results/probabilistic_safety/paper_numbers.json and a printed table.

Model (1-sigma, metres):
  sensing          sigma_ij^2 = s_i^2 + s_j^2, s_k^2 = sigma_b^2 + (sigma0 + kappa rho_k)^2
  inter-step dip   delta = 2 sqrt(3) a_bar g / 8 * dt^2, g = 1 + 5 sigma_s + 5 sigma_eta
  actuation        sigma_a = sqrt(2) a_bar sqrt(sigma_s^2 + 3 sigma_delta^2) / (lambda1 lambda2)
                   (both agents brake at full thrust; worst-case axis, so factor 3)
  held-pair shift  x_p + (lambda1 + lambda2)/(lambda1 lambda2) x_v
  white noise      sigma_noise^2 = (l1+l2)/(2 l1 l2) sigma_nv^2 + l1 l2/(2 (l1+l2)) sigma_np^2
  encounter sum    E = sum_pairs Phi(-(d_clean - r_c - delta) / sigma),
                   sigma^2 = sigma_bias^2 + sigma_noise^2 + sigma_a^2

The model does not include heavy tails, so the values are estimates, not bounds.
"""
import glob
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def Phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))


PhiV = np.vectorize(Phi)

# Geometry and filter (Table 9, Table 5)
D_S = 0.55
R_C = 2 * 0.5 * math.sqrt(0.1**2 + 0.2263**2 + 0.366**2)   # contact distance, 0.4418 m
L1, L2 = 0.10, 0.15
LAM = L1 * L2
GV = (L1 + L2) / LAM                                        # velocity-bias gain, 16.7 s
A_BAR = 0.005
DT = 1.0
N_EPISODES = 5000

# Sensing (Table 1)
SIG0_P, KAP_P, SIGB_P = 0.004, 0.001, 0.003
SIG0_V, KAP_V, SIGB_V = 0.001, 0.0003, 0.0005

# Actuation (Table 2): scale, jitter, misalignment [deg]
CASES = {"typical": (0.025, 0.01, 0.5), "high": (0.10, 0.05, 2.0)}

# Worst range: far corner of the spawn box to the far port.
RHO_MAX = math.sqrt((11.5 - 1.25) ** 2 + (3 + 1.1) ** 2 + (3 + 1.1) ** 2)


def s_k(rho):
    return math.sqrt(SIGB_P**2 + (SIG0_P + KAP_P * rho) ** 2)


def delta(ss, se):
    return 2 * math.sqrt(3) * A_BAR * (1 + 5 * ss + 5 * se) * DT**2 / 8


def sigma_a(ss, sd_deg):
    return math.sqrt(2) * A_BAR * math.sqrt(ss**2 + 3 * math.radians(sd_deg) ** 2) / LAM


def load_encounters():
    """Closest approach of every pair encounter, with its own bias shift removed."""
    d_clean, sig_noise = [], []
    files = sorted(glob.glob(os.path.join(RES, "probabilistic_safety", "exposure2", "s*.npz")))
    if not files:
        raise SystemExit("results/probabilistic_safety/exposure2/s*.npz not found. Run prob_cbf_exposure2.py first.")
    for k, f in enumerate(files):
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        C = {c: i for i, c in enumerate(m["cols"])}
        R = z["rows"].astype(np.float64)
        key = ((R[:, C["ep"]].astype(np.int64) + k * 1000) * 100
               + R[:, C["i"]].astype(np.int64) * 10 + R[:, C["j"]].astype(np.int64))
        clean = R[:, C["d"]] + R[:, C["x_p"]] + GV * R[:, C["x_v"]]
        o = np.argsort(key, kind="stable")
        st = np.flatnonzero(np.r_[True, key[o][1:] != key[o][:-1]])
        d_clean.append(np.minimum.reduceat(clean[o], st))
        # White noise at the true closest step of each encounter.
        oo = np.lexsort((R[:, C["d"]], key))
        idx = oo[np.r_[True, key[oo][1:] != key[oo][:-1]]]
        ri, rj = R[idx, C["rho_i"]], R[idx, C["rho_j"]]
        snv2 = (SIG0_V + KAP_V * ri) ** 2 + (SIG0_V + KAP_V * rj) ** 2
        snp2 = (SIG0_P + KAP_P * ri) ** 2 + (SIG0_P + KAP_P * rj) ** 2
        sig_noise.append(np.sqrt((L1 + L2) / (2 * LAM) * snv2 + LAM / (2 * (L1 + L2)) * snp2))
    return np.concatenate(d_clean), np.concatenate(sig_noise)


def main():
    out = {"r_c": R_C, "D_s": D_S, "rho_max": RHO_MAX}

    # Eq. 45, sensing only.
    s_ij = math.sqrt(2) * s_k(RHO_MAX)
    d0 = delta(0.025, 0.01)
    k0 = (D_S - R_C - d0) / s_ij
    out["sensing"] = {"sigma_ij": s_ij, "delta": d0, "buffer_sigma": k0, "p_step": Phi(-k0)}

    # Pair encounters for the per-episode estimate.
    d_clean, sig_noise = load_encounters()
    sig_bias = math.sqrt(2 * SIGB_P**2 + GV**2 * 2 * SIGB_V**2)
    i_min = int(np.argmin(d_clean))
    out["encounters"] = {"n": int(d_clean.size), "sigma_bias": sig_bias,
                         "closest_clean": float(d_clean[i_min]),
                         "sigma_noise_at_closest": float(sig_noise[i_min])}

    for lab, (ss, se, sd) in CASES.items():
        dl = delta(ss, se)
        sa = sigma_a(ss, sd)
        st = math.hypot(s_ij, sa)
        kb = (D_S - R_C - dl) / st
        sig = np.sqrt(sig_bias**2 + sig_noise**2 + sa**2)
        e_contacts = float(PhiV(-(d_clean - R_C - dl) / sig).sum())
        out[lab] = {
            "delta": dl, "sigma_a": sa, "buffer_sigma": kb, "p_step": Phi(-kb),
            "keepout_for_sensing_only_p_step": R_C + dl + (D_S - R_C - d0) / s_ij * st,
            "sigma_at_closest_pair": float(sig[i_min]),
            "expected_contacts_5000": e_contacts,
            "p_episode": e_contacts / N_EPISODES,
        }

    with open(os.path.join(RES, "probabilistic_safety", "paper_numbers.json"), "w") as f:
        json.dump(out, f, indent=2)

    s = out["sensing"]
    print(f"r_c {R_C*1e3:.1f} mm, worst range {RHO_MAX:.2f} m")
    print(f"sensing only: sigma_ij {s['sigma_ij']*1e3:.1f} mm, delta {s['delta']*1e3:.2f} mm, "
          f"buffer {s['buffer_sigma']:.2f} sigma, per-step {s['p_step']:.2e}")
    e = out["encounters"]
    print(f"{e['n']} pair encounters, sigma_bias {e['sigma_bias']*1e3:.1f} mm")
    for lab in CASES:
        c = out[lab]
        print(f"{lab:8s} sigma_a {c['sigma_a']*1e3:5.1f} mm  buffer {c['buffer_sigma']:.2f} sigma  "
              f"per-step {c['p_step']:.2e}  sigma at closest pair {c['sigma_at_closest_pair']*1e3:.0f} mm  "
              f"expected contacts / 5000 {c['expected_contacts_5000']:.3g}  per episode {c['p_episode']:.2e}")


if __name__ == "__main__":
    main()
