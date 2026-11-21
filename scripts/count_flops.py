"""Count the FLOPs of one control step by recording every ATen op that runs.

One FLOP per output element for elementwise ops. One per input element for a
sum. 2*M*N*K for a matrix product. 3 per output element for a 3-D cross
product. Comparisons, selects and data movement are counted separately.

The worst case runs the fast-path check, tiers 1-3 and the relaxation. The
script forces it with an input whose optimum is a box vertex.
"""
import sys, json, io, contextlib
from collections import Counter
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode

_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

ARITH = {"add", "sub", "rsub", "mul", "div", "neg", "pow", "sqrt", "rsqrt",
         "tanh", "abs", "reciprocal", "exp", "log", "square", "sign"}
REDUCE_SUM = {"sum", "mean", "prod"}
COMPARE = {"ge", "le", "gt", "lt", "eq", "ne", "minimum", "maximum", "clamp",
           "clamp_min", "clamp_max", "where", "amax", "amin", "max", "min",
           "any", "all", "isfinite", "logical_not", "logical_and", "logical_or",
           "bitwise_and", "bitwise_or", "bitwise_not"}


def _numel(x):
    return x.numel() if isinstance(x, torch.Tensor) else 0


class FlopCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.flops = Counter()
        self.compares = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = func.overloadpacket.__name__.rstrip("_")
        o = out[0] if isinstance(out, (tuple, list)) else out
        if name in ARITH:
            self.flops[name] += _numel(o)
        elif name in REDUCE_SUM:
            self.flops["sum"] += _numel(args[0])
        elif name in ("linalg_vector_norm", "norm"):
            self.flops["norm"] += 2 * _numel(args[0]) + _numel(o)
        elif name in ("mm", "addmm", "bmm", "matmul"):
            a, b = (args[1], args[2]) if name == "addmm" else (args[0], args[1])
            m, k = a.shape[-2], a.shape[-1]
            n = b.shape[-1]
            batch = int(torch.tensor(a.shape[:-2]).prod()) if a.dim() > 2 else 1
            self.flops["matmul"] += 2 * m * n * k * batch
            if name == "addmm":
                self.flops["matmul_bias"] += m * n
        elif name == "linalg_cross":
            self.flops["cross"] += 3 * _numel(o)
        elif name in COMPARE:
            self.compares += _numel(o) if name != "where" else 0
        return out

    @property
    def total(self):
        return sum(self.flops.values())


def count(fn):
    c = FlopCounter()
    with torch.no_grad(), c:
        fn()
    return c


DUMP = _ROOT / "results" / "flops_state.pt"


def dump():
    """Stage 1: save the weights, the filter and a real state.

    The counter runs in a separate process. BenchMARL changes how ATen ops are
    dispatched, so the counter cannot run in the same process.
    """
    import torch.nn as nn
    from scripts.mc_eval_6dof import load_policy_and_task
    ck = json.load(open(_ROOT / "results/main/seed3.json"))["checkpoint"]
    with contextlib.redirect_stdout(io.StringIO()):
        policy, task = load_policy_and_task(ck)
        task.config["cbf_target_keepout_mode"] = "halfspace"
        env = task.get_env_fun(num_envs=1, continuous_actions=True, seed=0, device="cpu")()
        env.reset()
    raw = env
    while not hasattr(raw, "safety_filter"):
        raw = getattr(raw, "base_env", None) or getattr(raw, "env")
    f = raw.safety_filter
    f.target_center = raw.target_center.clone()   # the env sets this every step
    lin = [m for m in policy.modules() if isinstance(m, nn.Linear)]
    torch.save({
        "W": [l.weight.detach().clone() for l in lin],
        "b": [l.bias.detach().clone() for l in lin],
        "filter": f,
        "pos": raw.pos.clone(), "vel": raw.vel.clone(), "N": raw.n_agents,
    }, DUMP)
    print(f"dumped to {DUMP}")


def main():
    from math import comb
    st = torch.load(DUMP, weights_only=False)
    f, N = st["filter"], st["N"]

    # Policy: one forward pass for one agent.
    # The deployed weights are meta placeholders, so make CPU tensors of the same
    # shapes. FLOPs depend only on shapes.
    st["W"] = [torch.randn(w.shape) for w in st["W"]]
    st["b"] = [torch.randn(b.shape) for b in st["b"]]
    x0 = torch.zeros(1, st["W"][0].shape[1])
    def fwd():
        h = x0
        for i, (W, b) in enumerate(zip(st["W"], st["b"])):
            h = torch.mm(h, W.t()) + b
            if i < len(st["W"]) - 1:
                h = torch.tanh(h)
    cp = count(fwd)

    # Constraint rows: built for all N agents, divided by N.
    rc, _, _, soft = f._translational_rows(st["pos"], st["vel"], None)
    K = rc.shape[2]
    cr = count(lambda: f._translational_rows(st["pos"], st["vel"], None))
    rows_pa = cr.total / N

    # Rotational solve: 1 angular-rate row and 6 box rows.
    al = float(f.angular_accel_scale)
    ua = torch.full((1, 3), 10.0 * al, dtype=torch.float64)
    ca = count(lambda: f._project_kkt(ua, torch.zeros(1, 1, 3, dtype=torch.float64),
                                      torch.ones(1, 1, dtype=torch.float64), al))
    stages = 1 + len(soft)

    def solve_costs(k):
        """FLOPs of one translational solve with k rows: (all tiers, tiers 1-2, fast path)."""
        bound = float(f.linear_accel_scale)
        # The nominal is far outside the box, so the optimum is a box vertex
        # and tier 3 runs.
        u = torch.full((1, 3), 10.0 * bound, dtype=torch.float64)
        c0 = torch.zeros(1, k, 3, dtype=torch.float64)
        d0 = torch.ones(1, k, dtype=torch.float64)
        x, inf = f._project_kkt(u, c0, d0, bound)
        assert not bool(inf.any()) and torch.allclose(x.abs(), torch.full_like(x, bound)), \
            "worst-case construction did not land on a box vertex"
        ct = count(lambda: f._project_kkt(u, c0, d0, bound))
        # Tiers 1-2 only: optimum on one row.
        u12 = torch.zeros(1, 3, dtype=torch.float64); u12[0, 0] = 10.0 * bound
        c12 = c0.clone(); c12[0, 0, 0] = 1.0; d12 = d0.clone(); d12[0, 0] = 0.5 * bound
        c12tier = count(lambda: f._project_kkt(u12, c12, d12, bound))
        # _solve_box_qp wrapper: normalisation and fast-path residual.
        cfast = count(lambda: f._solve_box_qp(torch.zeros(1, 3), c0.float(), d0.float(), bound))
        return ct.total, c12tier.total, cfast.total

    # Column 1: rows as coded. The code keeps a zero row for the agent itself,
    # so K includes it. Column 2: without that row, as reported in Table 17.
    cols = []
    for k in (K, K - 1):
        full, t12, fast = solve_costs(k)
        R = k + 6
        one = fast + full
        cols.append(dict(
            K=k, R=R, sets=R + comb(R, 2) + comb(R, 3),
            t12=t12 + fast, one=one,
            typ=rows_pa + fast + fast,                     # both QPs take the fast path
            filt_one=rows_pa + one + (fast + ca.total),
            filt_worst=rows_pa + stages * one + (fast + ca.total),
            rot=ca.total + fast))

    def row(label, key, fmt="{:>12,.0f}"):
        print(f"{label:<36}" + "".join(fmt.format(c[key]) for c in cols))

    n_params = sum(w.numel() for w in st["W"]) + sum(b.numel() for b in st["b"])
    print("=" * 74)
    print(f"policy parameters                   {n_params:>12,}       ({n_params * 4 / 1024:.0f} KiB fp32, "
          f"{n_params / 1024:.0f} KiB int8)")
    print(f"policy forward, one agent           {cp.total:>12,} FLOP  {dict(cp.flops)}")
    print(f"{'':<36}{'as coded':>12}{'Table 17':>12}")
    row("translational rows K", "K", "{:>12}")
    row("  enumerated R = K + 6 box", "R", "{:>12}")
    row("  candidate active sets", "sets", "{:>12}")
    row("  one solve, tiers 1-2 only", "t12")
    row("  one solve, tiers 1-3 (full)", "one")
    print(f"  staged relaxation: up to {stages} solves per step (soft cols {soft})")
    row("rotational solve (R=7, 63 sets)", "rot")
    print(f"constraint-row construction/agent   {rows_pa:>12,.0f}")
    print("-" * 74)
    row("filter, fast path (typical)", "typ")
    row("filter, one full solve", "filt_one")
    row(f"filter, worst case ({stages} solves)", "filt_worst")
    print("-" * 74)
    for c in cols:
        c["tot_one"] = cp.total + c["filt_one"]
        c["tot_worst"] = cp.total + c["filt_worst"]
    row("TOTAL, one full solve", "tot_one")
    row("TOTAL, worst case", "tot_worst")


if __name__ == "__main__":
    dump() if "--dump" in sys.argv else main()
