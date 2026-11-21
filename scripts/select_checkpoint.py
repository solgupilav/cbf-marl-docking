"""Select the checkpoint of a training run to report.

1. Evaluate evenly spaced checkpoints of the run at the reporting gate.
2. Evaluate the best of them and their neighbours again with more episodes.
3. Score each checkpoint by the mean success of itself and its two
   neighbours, and select the best. This prefers a stable plateau to a
   single lucky checkpoint.
4. Evaluate the selected checkpoint with 1000 episodes on the final seed.

Steps 1-3 use the screening seed (default 1234). Report the results on a
different seed (24680 in the paper), so the selection does not bias them.
"""

from __future__ import annotations

import argparse, json, re, subprocess, sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
PATTERN = re.compile(r"checkpoint_(\d+)(?:_succ(\d+))?\.pt$")


def all_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    dirs = [d for d in (run_dir / "checkpoints", *run_dir.glob("*/checkpoints")) if d.is_dir()]
    out = []
    for d in dirs:
        for p in d.glob("checkpoint_*.pt"):
            m = PATTERN.match(p.name)
            if m:
                out.append((int(m.group(1)), p))
    return sorted(out)


def mc_eval(ckpt: Path, episodes: int, json_out: Path, extra: list[str], seed: int):
    cmd = [sys.executable, str(_REPO_ROOT / "scripts" / "mc_eval_6dof.py"),
           "--checkpoint", str(ckpt), "--episodes", str(episodes),
           "--seed", str(seed), "--json-out", str(json_out), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not json_out.exists():
        print(f"    !! FAILED {ckpt.name}: {r.stderr.strip().splitlines()[-1:] or r.returncode}")
        return None
    return json.loads(json_out.read_text())


def rank_key(d: dict):
    # Complete success first, then fewer collisions, then less fuel.
    return (d["complete_success_rate"],
            -d.get("any_target_collision_rate", 0.0),
            -d.get("any_interagent_collision_rate", 0.0),
            -d.get("mean_linear_delta_v_per_agent", 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Training run folder.")
    ap.add_argument("--label", required=True, help="Name for the output files.")
    ap.add_argument("--out-dir", default="outputs/select", help="Output folder.")
    ap.add_argument("--coarse-points", type=int, default=16,
                    help="Number of evenly spaced checkpoints in the first "
                         "pass.")
    ap.add_argument("--coarse-episodes", type=int, default=150)
    ap.add_argument("--refine-top", type=int, default=4,
                    help="Number of best checkpoints to evaluate again, with "
                         "their neighbours.")
    ap.add_argument("--refine-episodes", type=int, default=400)
    ap.add_argument("--final-episodes", type=int, default=1000)
    ap.add_argument("--screen-seed", type=int, default=1234)
    ap.add_argument("--final-seed", type=int, default=97531)
    ap.add_argument("--min-frames", type=int, default=6_000_000,
                    help="Skip checkpoints before this frame count.")
    ap.add_argument("--mc-arg", action="append", default=[],
                    help="Argument for mc_eval_6dof.py. Repeat for each argument.")
    args = ap.parse_args()
    if args.screen_seed == args.final_seed:
        raise SystemExit("--final-seed must differ from --screen-seed")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cks = [(f, p) for f, p in all_checkpoints(Path(args.run_dir)) if f >= args.min_frames]
    if not cks:
        raise SystemExit(f"No checkpoints >= {args.min_frames} under {args.run_dir}")

    step = max(1, len(cks) // args.coarse_points)
    coarse = cks[::step]
    print(f"[select] {len(cks)} checkpoints above {args.min_frames/1e6:.0f}M frames; "
          f"coarse pass over {len(coarse)} of them at {args.coarse_episodes} episodes\n")
    print(f"{'frames':>12}{'success':>10}{'target':>9}{'inter':>8}")

    scored = {}
    for f, p in coarse:
        r = mc_eval(p, args.coarse_episodes, out / f"sweep_{args.label}_{p.stem}.json",
                    args.mc_arg, args.screen_seed)
        if r is None:
            continue
        r["checkpoint"] = str(p); r["frames"] = f
        scored[f] = r
        print(f"{f:>12,}{r['complete_success_rate']:>10.3f}"
              f"{r.get('any_target_collision_rate',0):>9.3f}"
              f"{r.get('any_interagent_collision_rate',0):>8.3f}")

    if not scored:
        raise SystemExit("Every coarse screen failed")

    # Evaluate the best coarse points and their neighbours again, because the
    # coarse grid can miss the peak.
    idx = {f: i for i, (f, _) in enumerate(cks)}
    top = sorted(scored.values(), key=rank_key, reverse=True)[:args.refine_top]
    want: set[int] = set()
    for r in top:
        i = idx[r["frames"]]
        for j in range(max(0, i - step // 2), min(len(cks), i + step // 2 + 1)):
            want.add(cks[j][0])
    want -= set()  # keep all
    print(f"\n[select] refining {len(want)} checkpoints around the top "
          f"{args.refine_top} at {args.refine_episodes} episodes\n")
    print(f"{'frames':>12}{'success':>10}{'target':>9}{'inter':>8}")
    refined = []
    for f in sorted(want):
        p = dict(cks)[f]
        r = mc_eval(p, args.refine_episodes, out / f"refine_{args.label}_{p.stem}.json",
                    args.mc_arg, args.screen_seed)
        if r is None:
            continue
        r["checkpoint"] = str(p); r["frames"] = f
        refined.append(r)
        print(f"{f:>12,}{r['complete_success_rate']:>10.4f}"
              f"{r.get('any_target_collision_rate',0):>9.4f}"
              f"{r.get('any_interagent_collision_rate',0):>8.4f}")

    # Score each checkpoint by the mean of itself and its neighbours. Many
    # checkpoints have 100% on the screening seed, so the maximum alone selects
    # the luckiest one. Only screening-seed data is used here.
    pool = refined or list(scored.values())
    pool.sort(key=lambda r: r["frames"])
    if len(pool) >= 3:
        succ = [r["complete_success_rate"] for r in pool]
        for i, r in enumerate(pool):
            lo, hi = max(0, i - 1), min(len(pool), i + 2)
            r["_smoothed"] = sum(succ[lo:hi]) / (hi - lo)
        best = max(pool, key=lambda r: (r["_smoothed"], rank_key(r)))
        print(f"\n[select] plateau-smoothed selection over {len(pool)} refined "
              f"checkpoints (neighbour-mean, screening seed only)")
    else:
        best = max(pool, key=rank_key)
    bp = Path(best["checkpoint"])
    print(f"[select] best: {bp.name} at {best['frames']:,} frames "
          f"({best['complete_success_rate']:.2%} raw, "
          f"{best.get('_smoothed', best['complete_success_rate']):.2%} smoothed, "
          f"screening seed)")
    print(f"[select] final evaluation at {args.final_episodes} episodes, "
          f"held-out seed {args.final_seed}")
    final = mc_eval(bp, args.final_episodes, Path(args.out_dir) / f"{args.label}_selected.json",
                    args.mc_arg, args.final_seed)
    if final is None:
        raise SystemExit("Final evaluation failed")
    print(f"[select] REPORTED {args.label}: {final['complete_success_rate']:.2%} "
          f"complete success, target {final.get('any_target_collision_rate',0):.4f}, "
          f"inter-agent {final.get('any_interagent_collision_rate',0):.4f}")
    (Path(args.out_dir) / f"{args.label}_selected.json").write_text(
        json.dumps({**final, "checkpoint": str(bp), "frames": best["frames"]}, indent=2))


if __name__ == "__main__":
    main()
