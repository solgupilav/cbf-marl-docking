"""Count how often each path of the translational QP runs in the reported
evaluation: fast path, tiers 1-2, tier 3 and relaxation.

Output: results/compute/filter_path_stats.json
"""
import json, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]
import torch
from marl_docking_6dof.safety.cbf import HOCBFSafetyFilter as F

S = {"agent_steps": 0, "stage": {1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0]},
     "rot": [0, 0, 0]}          # [problems, enumerated, reached tier 3]
_ctx = {"where": None, "stage": 0}

_orig_trans, _orig_box, _orig_kkt = F._solve_translational, F._solve_box_qp, F._project_kkt

def _trans(self, a_nom, *a, **k):
    _ctx["where"], _ctx["stage"] = "trans", 0
    S["agent_steps"] += a_nom.shape[0]
    try:
        return _orig_trans(self, a_nom, *a, **k)
    finally:
        _ctx["where"] = "rot"          # the rotational solve follows

def _box(self, u_nom, *a, **k):
    if _ctx["where"] == "trans":
        _ctx["stage"] += 1
        S["stage"][_ctx["stage"]][0] += u_nom.shape[0]
    else:
        S["rot"][0] += u_nom.shape[0]
    return _orig_box(self, u_nom, *a, **k)

def _kkt(self, u, *a, **k):
    bucket = S["stage"][_ctx["stage"]] if _ctx["where"] == "trans" else S["rot"]
    bucket[1] += u.shape[0]
    x, inf = _orig_kkt(self, u, *a, **k)
    return x, inf

# Only tier 3 calls torch.cross, so a probe on it counts tier-3 problems.
_orig_cross = torch.cross
def _cross(*a, **k):
    out = _orig_cross(*a, **k)
    if _ctx.get("in_kkt"):
        _ctx["t3"] = out.shape[0]
    return out

def _kkt_t3(self, u, *a, **k):
    _ctx["in_kkt"], _ctx["t3"] = True, 0
    try:
        r = _kkt(self, u, *a, **k)
    finally:
        _ctx["in_kkt"] = False
    bucket = S["stage"][_ctx["stage"]] if _ctx["where"] == "trans" else S["rot"]
    bucket[2] += _ctx["t3"]
    return r

F._solve_translational, F._solve_box_qp, F._project_kkt = _trans, _box, _kkt_t3
torch.cross = _cross

from scripts import mc_eval_6dof  # noqa: E402
if __name__ == "__main__":
    ck = json.load(open(_ROOT / "results/main/seed3.json"))["checkpoint"]
    sys.argv = ["mc_eval", "--checkpoint", ck, "--episodes", "200", "--seed", "24680",
                "--goal-tolerance", "0.04", "--attitude-gate", "0.0349",
                "--keepout-mode", "halfspace", "--target-lambda", "0.04",
                "--cbf-sensed-state", "on"]
    mc_eval_6dof.main()
    out = _ROOT / "results/compute/filter_path_stats.json"
    out.write_text(json.dumps(S, indent=1))
    n = S["agent_steps"]
    s1, s2, s3 = S["stage"][1], S["stage"][2], S["stage"][3]
    print("\n" + "=" * 66)
    print(f"agent-steps (translational QP calls)     {n:,}")
    print(f"  fast path, no enumeration              {100*(n-s1[1])/n:6.2f}%")
    print(f"  enumerated, tiers 1-2 sufficient       {100*(s1[1]-s1[2])/n:6.2f}%")
    print(f"  reached tier 3                         {100*s1[2]/n:6.2f}%")
    print(f"  2nd solve (speed row dropped)          {100*s2[0]/n:6.3f}%")
    print(f"  3rd solve (keep-out row dropped)       {100*s3[0]/n:6.3f}%")
    print(f"rotational: enumerated {100*S['rot'][1]/max(S['rot'][0],1):.2f}%  tier 3 {100*S['rot'][2]/max(S['rot'][0],1):.3f}%")
    print(f"wrote {out}")
