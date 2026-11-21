"""Make the rollout data for the paper figures.

curve.npz    training curve and curriculum gate schedule, from a training log.
figdata.npz  rollout of one checkpoint: trajectories, separation, outcomes
             and filter intervention.

Intervention samples do not include docked agents, because the environment
freezes them.

Usage:
    python scripts/make_figure_data.py --checkpoint <ckpt.pt> --episodes 400
    (add --train-log <train.log> to also write curve.npz)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torchrl.envs.utils import ExplorationType, set_exploration_type  # noqa: E402
from scripts.mc_eval_6dof import load_policy_and_task  # noqa: E402

EVAL_LINE = re.compile(
    r"frames=(\d+).*?all-agents docked:\s*\d+/\d+\s*=\s*([\d.]+)%.*?goal_tolerance=([\d.]+)"
)


def build_curve(train_log: Path, out: Path) -> None:
    frames, succ, gate_f, gate_v = [], [], [], []
    last_gate = None
    for line in train_log.read_text(errors="ignore").splitlines():
        m = EVAL_LINE.search(line)
        if not m:
            continue
        f, s, g = int(m.group(1)), float(m.group(2)), float(m.group(3))
        frames.append(f); succ.append(s)
        if last_gate is None or g != last_gate:
            gate_f.append(f); gate_v.append(g); last_gate = g
    np.savez(out, frames=np.array(frames), succ=np.array(succ),
             gate_f=np.array(gate_f), gate_v=np.array(gate_v))
    print(f"  wrote {out}  ({len(frames)} eval points, {len(gate_f)} gate steps, "
          f"final gate {gate_v[-1] if gate_v else float('nan')} m)")


def build_rollout(checkpoint: str, episodes: int, seed: int, out: Path) -> None:
    policy, task = load_policy_and_task(checkpoint)
    env = task.get_env_fun(num_envs=episodes, continuous_actions=True,
                           seed=seed, device="cpu")()
    raw = env
    while not hasattr(raw, "safety_filter"):
        raw = getattr(raw, "base_env", None) or getattr(raw, "env")
    N, T = raw.n_agents, int(raw.max_steps)

    a_half = 0.5 * raw.agent_size.view(1, 1, 3)
    t_half = 0.5 * raw.target_size.view(1, 1, 3)
    r_contact = float(torch.linalg.norm(raw.agent_size))          # 2 * bounding radius

    td = env.reset()
    traj = np.full((T, episodes, N, 3), np.nan, dtype=np.float32)
    minsep = np.full(episodes, np.inf, dtype=np.float32)
    docked = torch.zeros(episodes, N, dtype=torch.bool)
    tgt_hit = torch.zeros(episodes, N, dtype=torch.bool)
    ag_hit = torch.zeros(episodes, N, dtype=torch.bool)
    dock_step = np.full((episodes, N), np.nan, dtype=np.float32)
    iv_s, dist_s = [], []
    active = torch.ones(episodes, 1, dtype=torch.bool)

    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for t in range(T):
            td = policy(td)
            nxt = env.step(td)["next"]
            info = nxt[(raw.group, "info")]
            am = active.squeeze(-1)
            traj[t, am.numpy()] = raw.pos[am].numpy()

            # Closest pair this step (inf when only one agent is in flight).
            dp = raw.pos.unsqueeze(2) - raw.pos.unsqueeze(1)
            dd = dp.norm(dim=-1) + torch.eye(N).unsqueeze(0) * 1e6
            step_min = dd.amin(dim=(1, 2))
            minsep = np.minimum(minsep, np.where(am.numpy(), step_min.numpy(), np.inf))
            ag_hit |= (dd.amin(dim=2) < r_contact) & active

            # Box overlap with the target, the same test as mc_eval.
            # target_center is [B, 3]; broadcast over agents.
            rel = (raw.pos - raw.target_center.unsqueeze(1)).abs()
            tgt_hit |= ((rel < (t_half + a_half)).all(-1)) & active

            newly = info["on_goal"][..., 0].bool() & (~docked) & active
            dock_step[newly.numpy()] = t
            docked |= newly

            # Filter intervention against range, for agents not yet docked.
            live = (~raw.on_goal_latched.squeeze(-1)) & active
            if bool(live.any()):
                iv_s.append(info["cbf_intervention"][..., 0][live].numpy())
                dist_s.append((raw.goals - raw.pos).norm(dim=-1)[live].numpy())

            active &= ~nxt["done"].bool()
            td = nxt
            if not bool(active.any()):
                break

    np.savez(
        out,
        traj=traj, goals=raw.goals.numpy(),
        target_center=raw.target_center[0].numpy(), target_size=raw.target_size.numpy(),
        agent_size=raw.agent_size.numpy(),
        minsep=minsep, D_s=np.float32(raw.cbf_agent_keepout),
        iv_flat=np.concatenate(iv_s) if iv_s else np.zeros(0, np.float32),
        dist_flat=np.concatenate(dist_s) if dist_s else np.zeros(0, np.float32),
        docked=docked.numpy(), tgt=tgt_hit.numpy(), ag=ag_hit.numpy(),
        dock_step=dock_step.reshape(-1),
        keepout_axes=np.asarray(raw.cbf_target_keepout_axes, dtype=np.float32),
        keepout_exp=np.int32(raw.cbf_target_keepout_exponent),
    )
    print(f"  wrote {out}  ({episodes} episodes, {N} agents, "
          f"{np.isfinite(dock_step).sum()} captures, "
          f"min separation {np.nanmin(minsep[np.isfinite(minsep)]):.3f} m)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-log", default="")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if args.train_log:
        build_curve(Path(args.train_log), out / "curve.npz")
    build_rollout(args.checkpoint, args.episodes, args.seed, out / "figdata.npz")


if __name__ == "__main__":
    main()
