"""Pool the 5-seed results of the actuator and swarm-size tables (Tables 15
and 16 of the paper) and print them.

The exact-actuation and N = 9 cells are the main results, so
they are read from results/main/seed*.json. All other cells are in
results/sensitivity/.
"""
import json
import os
from math import comb

import numpy as np

R = "results"
CONTACT = 0.44177788496017456
D_S = 0.55


def fisher_p(a, na, b, nb):
    """Two-sided Fisher exact test for a/na against b/nb."""
    tot, k = na + nb, a + b
    pr = lambda x: comb(na, x) * comb(nb, k - x) / comb(tot, k)
    p0 = pr(a)
    return min(1.0, sum(pr(x) for x in range(max(0, k - nb), min(k, na) + 1)
                        if pr(x) <= p0 * (1 + 1e-9)))


def cell(name):
    """Five per-seed result paths for a cell name."""
    if name in ("actuator_off", "scale_n9"):
        return [os.path.join(R, "main", f"seed{i}.json") for i in range(5)]
    return [os.path.join(R, "sensitivity", f"{name}_seed{i}.json") for i in range(5)]


def pool(name):
    rs = [json.load(open(p)) for p in cell(name) if os.path.exists(p)]
    if len(rs) < 5:
        return None
    cnt = lambda k: int(round(sum(r[k] * r["episodes"] for r in rs)))
    return dict(
        n=sum(r["episodes"] for r in rs),
        success=100 * np.mean([r["complete_success_rate"] for r in rs]),
        seeds=[100 * r["complete_success_rate"] for r in rs],
        per_agent=100 * np.mean([r["mean_agent_success"] for r in rs]),
        inter=cnt("any_interagent_collision_rate"),
        inter_seeds=[int(round(r["any_interagent_collision_rate"] * r["episodes"])) for r in rs],
        target=cnt("any_target_collision_rate"),
        target_seeds=[int(round(r["any_target_collision_rate"] * r["episodes"])) for r in rs],
        timeout=cnt("timeout_failure_rate"),
        qp=100 * np.mean([r["any_cbf_infeasible_rate"] for r in rs]),
    )


def line(label, c):
    if c is None:
        print(f"  {label:<22} INCOMPLETE")
        return
    rng = f"{min(c['seeds']):.1f}-{max(c['seeds']):.1f}"
    print(f"  {label:<22} {c['success']:6.2f}% [{rng:>11}]  per-agent {c['per_agent']:6.2f}%  "
          f"inter {c['inter']:>4}  target {c['target']:>2}  timeout {c['timeout']:>3}  QP {c['qp']:5.2f}%")


def header(title):
    print("=" * 100)
    print(f"{title}  (5 seeds x 1000 episodes; contacts are episode counts)")
    print("=" * 100)


header("Actuator uncertainty")
A = {m: pool("actuator_" + m) for m in ("off", "typical", "high")}
for m in A:
    line(m, A[m])
if all(A.values()):
    print(f"\n  inter-agent contacts per seed, high: {A['high']['inter_seeds']}")
    print(f"  Fisher, inter-agent, off vs high:    p = {fisher_p(A['high']['inter'], 5000, A['off']['inter'], 5000):.3f}")
    print(f"  Fisher, inter-agent, off vs typical: p = {fisher_p(A['typical']['inter'], 5000, A['off']['inter'], 5000):.3f}")
    # Separation margin, if the per-episode arrays are present.
    for m in A:
        paths = ([os.path.join(R, "main", f"seed{i}.npz") for i in range(5)] if m == "off" else
                 [os.path.join(R, "sensitivity", f"actuator_{m}_seed{i}.npz") for i in range(5)])
        if all(os.path.exists(p) for p in paths):
            s = np.concatenate([np.load(p)["min_separation"] for p in paths])
            print(f"  min separation {m:<8} p1 {np.percentile(s, 1):.4f}  min {s.min():.4f}  "
                  f"inside D_s {int((s < D_S).sum()):>3}  at contact {int((s < CONTACT).sum())}")

print()
header("Swarm size")
for N in (4, 6, 9, 12, 16, 20):
    line(f"N={N}", pool(f"scale_n{N}"))
c20 = pool("scale_n20")
if c20:
    pa = c20["per_agent"] / 100
    print(f"\n  N=20: per-agent {100 * pa:.2f}%^20 = {100 * pa ** 20:.1f}% if independent, "
          f"measured {c20['success']:.1f}%")
