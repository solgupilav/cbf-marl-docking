"""Write results/PAPER_NUMBERS.txt: every result in the paper, computed from
the stored files in results/.

Usage: python scripts/paper_numbers.py
"""
import json
import math
import os

import numpy as np

R = "results"
OUT = os.path.join(R, "PAPER_NUMBERS.txt")
D_S = 0.55


def load(*parts):
    return json.load(open(os.path.join(R, *parts)))


def seeds(fmt):
    return [load(*fmt.format(i=i).split("/")) for i in range(5)]


def pooled(rs):
    """Pooled rates and counts over a list of per-seed result files."""
    n = sum(r["episodes"] for r in rs)
    cnt = lambda k: int(round(sum(r[k] * r["episodes"] for r in rs)))
    mean = lambda k: sum(r[k] * r["episodes"] for r in rs) / n
    return dict(n=n, success=cnt("complete_success_rate"), timeout=cnt("timeout_failure_rate"),
                target=cnt("any_target_collision_rate"), inter=cnt("any_interagent_collision_rate"),
                per_agent=mean("mean_agent_success"), lin=mean("mean_linear_delta_v_per_agent"),
                ang=mean("mean_angular_delta_v_per_agent"), steps=mean("mean_episode_len"),
                qp=mean("any_cbf_infeasible_rate"), collision=cnt("collision_failure_rate"))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def pct(k, n):
    return 100.0 * k / n


lines = []
w = lines.append


def section(title):
    w("")
    w(title)
    w("-" * len(title))


w("Results reported in the paper, computed from the files in results/.")
w("Scoring: 4 cm / 2 deg capture gate, half-space keep-out (pole 0.04), filter")
w("on the sensed state, evaluation seed 24680, 1000 episodes per seed.")

# Table 10 and Figure 7.
main = seeds("main/seed{i}.json")
m = pooled(main)
lo, hi = wilson(m["success"], m["n"])
section("Table 10: main result (5 seeds x 1000 episodes)")
w(f"All agents docked, no collision   {pct(m['success'], m['n']):.2f}%  (Wilson 95% CI {lo:.2f}-{hi:.2f}%)")
w(f"Failed, collision                 {pct(m['collision'], m['n']):.2f}%")
w(f"Failed, timeout                   {pct(m['timeout'], m['n']):.2f}%")
w(f"Per-agent capture                 {100 * m['per_agent']:.2f}%")
w(f"Episodes with inter-agent contact {m['inter']}/{m['n']}")
w(f"Episodes with target contact      {m['target']}/{m['n']}")
w(f"Mean linear delta-v per agent     {m['lin']:.4f} m/s")
w(f"Mean angular delta-v per agent    {m['ang']:.4f} rad/s")
w(f"Mean episode length               {m['steps']:.1f} steps")
w(f"Hard-infeasible QP (episodes)     {100 * m['qp']:.2f}%")
w("Per-seed success (Figure 7)       " + ", ".join(f"{100 * r['complete_success_rate']:.1f}%" for r in main))
w("Per-seed linear delta-v           " + ", ".join(f"{r['mean_linear_delta_v_per_agent']:.4f}" for r in main))
w(f"Contact rate bound (rule of three) {pct(3, m['n']):.2f}% at 95% confidence")

# Table 11.
section("Table 11: safety-filter ablation (1000 episodes each)")
w("Trained with the filter: seed 3. Trained without the filter: training seed 0.")
cols = [("trained with, eval on", "ablations/filter_on.json"),
        ("trained with, eval off", "ablations/filter_off.json"),
        ("trained without, eval off", "ablations/trained_without_filter_eval_off.json"),
        ("trained without, eval on", "ablations/trained_without_filter_eval_on.json")]
w(f"{'':<27}{'success':>9}{'per-agent':>11}{'inter':>9}{'target':>9}{'timeout':>9}{'lin dv':>8}")
for label, f in cols:
    d = load(*f.split("/"))
    w(f"{label:<27}{100 * d['complete_success_rate']:>8.2f}%{100 * d['mean_agent_success']:>10.2f}%"
      f"{100 * d['any_interagent_collision_rate']:>8.2f}%{100 * d['any_target_collision_rate']:>8.2f}%"
      f"{100 * d['timeout_failure_rate']:>8.2f}%{d['mean_linear_delta_v_per_agent']:>8.3f}")

# Table 12.
section("Table 12: LQR nominal through the same filter (1000 episodes each)")
w(f"{'':<16}{'success':>9}{'inter':>9}{'target':>9}{'timeout':>9}")
for label, d in (("LQR, filter off", load("lqr", "lqr_no_filter.json")),
                 ("LQR, filter on", load("lqr", "lqr_with_filter.json"))):
    w(f"{label:<16}{100 * d['complete_success_rate']:>8.2f}%{100 * d['any_interagent_collision_rate']:>8.2f}%"
      f"{100 * d['any_target_collision_rate']:>8.2f}%{100 * d['timeout_failure_rate']:>8.2f}%")
w(f"{'RL, filter on':<16}{pct(m['success'], m['n']):>8.2f}%{pct(m['inter'], m['n']):>8.2f}%"
  f"{pct(m['target'], m['n']):>8.2f}%{pct(m['timeout'], m['n']):>8.2f}%  (Table 10)")

# Table 13.
lqr_lin = lqr_ang = None
for line in open(os.path.join(R, "lqr", "fuel_reference.txt")):
    if "linear  dv per agent" in line:
        lqr_lin = float(line.split("mean")[1].split()[0])
    if "angular dv per agent" in line:
        lqr_ang = float(line.split("mean")[1].split()[0])
section("Table 13: fuel against the LQR reference (lqr/fuel_reference.txt)")
w(f"Linear delta-v per agent   RL {m['lin']:.4f}  LQR {lqr_lin:.4f} m/s    {100 * (m['lin'] / lqr_lin - 1):+.0f}%")
w(f"Angular delta-v per agent  RL {m['ang']:.4f}  LQR {lqr_ang:.4f} rad/s  {100 * (m['ang'] / lqr_ang - 1):+.0f}%")

# Table 14.
section("Table 14: target keep-out geometry (5 seeds x 1000 episodes)")
w(f"{'keep-out':<12}{'horizon':>8}{'success':>9}{'target':>8}{'timeout':>9}{'steps':>7}")
for label, g in (("m = 8", "m8"), ("m = 16", "m16"), ("half-space", "halfspace")):
    for h, sfx in ((300, "_h300"), (600, "")):
        k = pooled(seeds(f"keepout/seed{{i}}_{g}{sfx}.json"))
        w(f"{label:<12}{h:>8}{pct(k['success'], k['n']):>8.2f}%{k['target']:>8}{k['timeout']:>9}{k['steps']:>7.0f}")

# Table 15.
section("Table 15: actuator uncertainty (5 seeds x 1000 episodes)")
w(f"{'actuation':<10}{'success':>9}{'target':>8}{'inter':>7}{'QP infeas.':>12}")
for label, rs in (("exact", main), ("typical", seeds("sensitivity/actuator_typical_seed{i}.json")),
                  ("high", seeds("sensitivity/actuator_high_seed{i}.json"))):
    a = pooled(rs)
    w(f"{label:<10}{pct(a['success'], a['n']):>8.2f}%{a['target']:>8}{a['inter']:>7}{100 * a['qp']:>11.2f}%")

# Table 16.
section("Table 16: swarm size (5 seeds x 1000 episodes)")
w(f"{'N':<4}{'success':>9}{'per-agent':>11}{'target':>9}{'inter':>8}{'QP infeas.':>12}")
for n in (4, 6, 9, 12, 16, 20):
    s = pooled(main if n == 9 else seeds(f"sensitivity/scale_n{n}_seed{{i}}.json"))
    w(f"{n:<4}{pct(s['success'], s['n']):>8.1f}%{100 * s['per_agent']:>10.2f}%"
      f"{pct(s['target'], s['n']):>8.2f}%{pct(s['inter'], s['n']):>7.2f}%{100 * s['qp']:>11.2f}%")

# Table 17 and quantisation.
section("Table 17: compute cost (compute/flop_count.txt)")
for line in open(os.path.join(R, "compute", "flop_count.txt")):
    if line.strip() and not line.startswith(("=", "-")):
        w(line.rstrip())
q8 = load("quantisation", "seed3_int8.json")
q16 = load("quantisation", "seed3_fp16.json")
w(f"Seed 3 with int8 weights: success {100 * q8['complete_success_rate']:.2f}%, "
  f"linear delta-v {q8['mean_linear_delta_v_per_agent']:.4f} (fp32 {main[3]['mean_linear_delta_v_per_agent']:.4f})")
w(f"Seed 3 with fp16 weights: success {100 * q16['complete_success_rate']:.2f}%")

# Figures 4 and 5.
arr = [np.load(os.path.join(R, "main", f"seed{i}.npz")) for i in range(5)]
sep = np.concatenate([a["min_separation"] for a in arr])
dock = np.concatenate([a["dock_step"].ravel() * float(a["dt"]) for a in arr])
cap = dock[dock >= 0]
section("Figures 4 and 5 (main/seed*.npz)")
w(f"Minimum separation  {sep.min():.4f} m; {int((sep < D_S).sum())} of {sep.size} episodes inside D_s = {D_S} m")
w(f"Time to capture     median {np.median(cap):.0f} s, 95th percentile {np.percentile(cap, 95):.0f} s, "
  f"max {cap.max():.0f} s; {cap.size} of {dock.size} agent-episodes captured")

# Discussion.
p = load("probabilistic_safety", "paper_numbers.json")
section("Discussion: probabilistic safety (probabilistic_safety/paper_numbers.json)")
w(f"Sensing only: sigma_ij {1000 * p['sensing']['sigma_ij']:.1f} mm at {p['rho_max']:.1f} m, "
  f"buffer {p['sensing']['buffer_sigma']:.2f} sigma, {p['sensing']['p_step']:.1e} per pair and step")
for case in ("typical", "high"):
    c = p[case]
    w(f"{case:<8} actuation: sigma_a {1000 * c['sigma_a']:.1f} mm, per episode {c['p_episode']:.1e}")

open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
