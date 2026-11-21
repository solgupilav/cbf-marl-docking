"""Control barrier function (CBF) safety filters for the 6-DOF docking env.

CBFSafetyFilter is a pass-through filter. It is the default when
enable_cbf=False.

HOCBFSafetyFilter is the decentralised high-order CBF filter. Its rows are:

1. Inter-agent separation (hard, relative degree 2). For each pair (i, j),
   h_ij = ||dp||^2 - D_s^2. Each agent takes half of the pair condition:

       -2 dp . a_i <= b_ij / 2
       b_ij = 2||dv||^2 + 2 dp.(A_p dp + A_v dv)
              + 2(lambda1+lambda2) dp.dv + lambda1*lambda2 (||dp||^2 - D_s^2)

   A_p and A_v are the CW drift terms, so the filter model matches the plant.

2. Target speed limit (relative degree 1):

       h_v = nu0 + nu1 ||p - c_T|| - ||v||

3. Target keep-out (relative degree 2): a superellipsoid
   h_T = sum_k ((p_k - c_k)/rho_k)^n - 1, or the half-space h_T = p_x - rho_x.

4. Angular-rate limit (relative degree 1):

       2 w . alpha <= gamma_w (w_max^2 - ||w||^2)

The translational rows use only the linear acceleration and the rotational
row only the angular acceleration. So the QP splits into two 3-variable QPs:

    u* = argmin ||u - u_nom||^2   s.t.  C u <= d,   |u|_inf <= u_bar

Each QP is solved exactly by enumeration of active sets of size 1, 2 and 3
(KKT conditions), as batched tensor operations. The result is always clamped
to the actuator box.

If the hard rows have no solution in the box, the filter applies a backup
action: maximum thrust away from the nearest threat and maximum despin.

Inputs are physical states (m, m/s) and normalised actions in [-1, 1].
The output is a normalised action.
"""

from __future__ import annotations

import itertools
from typing import Optional

import torch


class CBFSafetyFilter:
    """Base class for CBF safety filters. This class does not change the action.

    pos and vel are [B, N, 3]. action is [B, N, action_dim]; its first 3
    components are the linear acceleration in [-1, 1].
    """

    def __init__(
        self,
        *,
        linear_accel_scale: float = 0.005,
        target_center: Optional[torch.Tensor] = None,
        target_size: Optional[torch.Tensor] = None,
        agent_radius: float = 0.25,
        min_agent_separation: float = 0.0,
        alpha_gain: float = 1.0,
    ) -> None:
        self.linear_accel_scale = float(linear_accel_scale)
        self.target_center = target_center
        self.target_size = target_size
        self.agent_radius = float(agent_radius)
        self.min_agent_separation = float(min_agent_separation)
        self.alpha_gain = float(alpha_gain)

    # Barrier functions (pass-through).
    def h_target(self, pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        """Target barrier, [B, N, 1]. Safe when h >= 0. Pass-through: +inf."""
        return torch.full_like(pos[..., :1], float("inf"))

    def h_agents(self, pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        """Inter-agent barrier, [B, N, 1]. Safe when h >= 0. Pass-through: +inf."""
        return torch.full_like(pos[..., :1], float("inf"))

    # Action filter (pass-through).
    def filter_action(
        self,
        *,
        pos: torch.Tensor,
        vel: torch.Tensor,
        action: torch.Tensor,
        quat: Optional[torch.Tensor] = None,
        ang_vel: Optional[torch.Tensor] = None,
        static_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the action onto the safe set. Return (safe_action, intervention).

        intervention is [B, N, 1], ||u_safe - u_RL|| in normalised units.
        static_mask [B, N] marks docked agents, which do not move.
        Pass-through: return the action unchanged and zero intervention.
        """
        intervention = torch.zeros_like(action[..., :1])
        return action, intervention


class HOCBFSafetyFilter(CBFSafetyFilter):
    """Decentralised HOCBF-QP safety filter (see the module docstring).

    Gains are per second. The environment solves the filter at each dynamics
    substep.
    """

    def __init__(
        self,
        *,
        linear_accel_scale: float = 0.005,
        angular_accel_scale: float = 0.02,
        # Inter-agent keep-apart distance D_s (m): contact distance 2*r_bound
        # (about 0.442 m) plus margins.
        agent_keepout: float = 0.55,
        # HOCBF poles at -lambda1 and -lambda2. Keep lambda1 <= sqrt(2 a_rel / d_max),
        # so the shared braking can stop any closing speed the barrier allows.
        lambda1: float = 0.05,
        lambda2: float = 0.075,
        # Pairs farther than this get no row. Default inf: a finite radius lets a pair
        # enter range already too fast to stop. Distant rows are slack and cost little.
        activation_radius: float = float("inf"),
        neighbor_limit: int = 0,
        # Speed limit ||v|| <= nu0 + nu1*||p - c_T||. nu0 is above max_docking_speed,
        # so the limit does not block capture.
        speed_limit_intercept: float = 0.15,
        speed_limit_slope: float = 0.015,
        speed_limit_gain: float = 0.5,
        # Target keep-out semi-axes rho (m). They are anisotropic, so the docked pose
        # stays outside the surface. None disables the row.
        target_keepout_axes: Optional[tuple[float, float, float]] = None,
        target_lambda1: float = 0.15,
        target_lambda2: float = 0.15,
        target_keepout_exponent: int = 4,
        # "superellipsoid" (default) or "halfspace". The half-space
        # h_T = p_x - rho_x keeps the chaser in front of the docking face. It is exact,
        # because the target is at p_x <= rho_x. A superellipsoid cannot cover the box
        # corners. The approach is along +X, so the space beside the target is not used.
        target_keepout_mode: str = "superellipsoid",
        # Angular-rate limit ||w|| <= w_max (rad/s) and its gain.
        max_ang_speed: float = 0.2,
        ang_speed_gain: float = 0.5,
        # Mean motion n of the target orbit (rad/s) for the CW drift terms.
        mean_motion: float = 0.0,
        # Regulariser for ||v|| in the speed-limit row (smooth at v = 0).
        vel_eps: float = 1e-3,
        target_center: Optional[torch.Tensor] = None,
        target_size: Optional[torch.Tensor] = None,
        agent_radius: float = 0.25,
    ) -> None:
        super().__init__(
            linear_accel_scale=linear_accel_scale,
            target_center=target_center,
            target_size=target_size,
            agent_radius=agent_radius,
            min_agent_separation=agent_keepout,
            alpha_gain=lambda1,
        )
        self.angular_accel_scale = float(angular_accel_scale)
        self.agent_keepout = float(agent_keepout)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.activation_radius = float(activation_radius)
        # Keep only the pair rows of the k nearest teammates (0 = no limit).
        self.neighbor_limit = int(neighbor_limit)
        self.speed_limit_intercept = float(speed_limit_intercept)
        self.speed_limit_slope = float(speed_limit_slope)
        self.speed_limit_gain = float(speed_limit_gain)
        self.target_keepout_axes = (
            None
            if target_keepout_axes is None
            else tuple(float(v) for v in target_keepout_axes)
        )
        self.target_lambda1 = float(target_lambda1)
        self.target_lambda2 = float(target_lambda2)
        # Superellipsoid exponent n (even). At n = 4 the surface is inside the box at
        # the corner ports, so the row does not engage there. n = 8 covers the corners.
        self.target_keepout_mode = str(target_keepout_mode)
        if self.target_keepout_mode not in ("superellipsoid", "halfspace"):
            raise ValueError(
                f"target_keepout_mode must be 'superellipsoid' or 'halfspace', "
                f"got {target_keepout_mode!r}"
            )
        self.target_keepout_exponent = int(target_keepout_exponent)
        if self.target_keepout_mode == "superellipsoid" and (
            self.target_keepout_exponent < 2 or self.target_keepout_exponent % 2
        ):
            raise ValueError("target_keepout_exponent must be an even integer >= 2")
        self.max_ang_speed = float(max_ang_speed)
        self.ang_speed_gain = float(ang_speed_gain)
        self.mean_motion = float(mean_motion)
        self.vel_eps = float(vel_eps)
        # Cache of constraint-subset index tensors, keyed by (n, r, device).
        self._combo_cache: dict[tuple[int, int, str], torch.Tensor] = {}
        # Diagnostics of the last call: hard-infeasible count (backup used), count of
        # problems with a relaxed soft row, and the hard-infeasible mask [B, N, 1].
        self.last_infeasible: int = 0
        self.last_relaxed: int = 0
        self.last_infeasible_mask: Optional[torch.Tensor] = None
        # [B, N, 1] bool: agents whose QP needed a soft row dropped.
        self.last_relaxed_mask: Optional[torch.Tensor] = None

    # Barrier values (for logging and the observation).
    def h_agents(self, pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        """min_j (||p_i - p_j||^2 - D_s^2) per agent, [B, N, 1]."""
        n_agents = pos.shape[-2]
        if n_agents < 2:
            return torch.full_like(pos[..., :1], float("inf"))
        dp = pos.unsqueeze(-2) - pos.unsqueeze(-3)  # [B, N(self), N(other), 3]
        dist2 = (dp * dp).sum(-1)
        eye = torch.eye(n_agents, dtype=torch.bool, device=pos.device)
        dist2 = dist2.masked_fill(eye, float("inf"))
        return (dist2.amin(dim=-1) - self.agent_keepout**2).unsqueeze(-1)

    def h_target(self, pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        """min(speed-limit barrier, keep-out barrier) per agent, [B, N, 1].
        Both are safe when >= 0.
        """
        c_t = self._target_center_like(pos)
        if c_t is None:
            return torch.full_like(pos[..., :1], float("inf"))
        p_r = pos - c_t
        range_norm = p_r.norm(dim=-1, keepdim=True)
        speed = torch.sqrt((vel * vel).sum(-1, keepdim=True) + self.vel_eps**2)
        h_v = self.speed_limit_intercept + self.speed_limit_slope * range_norm - speed
        if self.target_keepout_axes is None:
            return h_v
        rho = torch.tensor(self.target_keepout_axes, dtype=pos.dtype, device=pos.device)
        h_keepout = (p_r / rho).pow(4).sum(-1, keepdim=True) - 1.0
        return torch.minimum(h_v, h_keepout)

    def _target_center_like(self, pos: torch.Tensor) -> Optional[torch.Tensor]:
        """Target centre as [B, 1, 3], or None."""
        if self.target_center is None:
            return None
        c_t = torch.as_tensor(self.target_center, dtype=pos.dtype, device=pos.device)
        if c_t.dim() == 1:
            return c_t.view(1, 1, 3)
        return c_t.reshape(-1, 1, 3)

    # Constraint rows (affine in u: C u <= d).
    def _cwh_drift(self, dp: torch.Tensor, dv: torch.Tensor) -> torch.Tensor:
        """A_p dp + A_v dv: CW drift, linear, so exact for pairs."""
        n = self.mean_motion
        return torch.stack(
            [
                3.0 * n * n * dp[..., 0] + 2.0 * n * dv[..., 1],
                -2.0 * n * dv[..., 0],
                -(n * n) * dp[..., 2],
            ],
            dim=-1,
        )

    def _translational_rows(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        static_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        """Make C [B,N,K,3] and d [B,N,K] for the linear-acceleration QP.

        Rows: one shared inter-agent row per pair, the target speed limit and the
        target keep-out. An inactive row is 0.a <= 1.

        static_mask [B, N] marks docked agents. A docked neighbour cannot do its half,
        so the active agent takes the full pair condition. Rows of a docked agent
        are dropped.

        Return (C, d, backup_dir, soft_cols). backup_dir is the escape direction.
        soft_cols are the rows that can be relaxed, in drop order (speed limit, then
        keep-out). Pair rows are never relaxed.
        """
        b_sz, n_agents, _ = pos.shape
        device, dtype = pos.device, pos.dtype
        rows_c: list[torch.Tensor] = []
        rows_d: list[torch.Tensor] = []
        soft_cols: list[int] = []
        backup_dir = torch.zeros_like(pos)

        n_pair_cols = 0
        if n_agents > 1:
            n_pair_cols = n_agents
            dp = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B, N(self), N(other), 3]
            dv = vel.unsqueeze(2) - vel.unsqueeze(1)
            dist2 = (dp * dp).sum(-1)  # [B, N, N]
            drift = self._cwh_drift(dp, dv)
            b_ij = (
                2.0 * (dv * dv).sum(-1)
                + 2.0 * (dp * drift).sum(-1)
                + 2.0 * (self.lambda1 + self.lambda2) * (dp * dv).sum(-1)
                + self.lambda1 * self.lambda2 * (dist2 - self.agent_keepout**2)
            )
            eye = torch.eye(n_agents, dtype=torch.bool, device=device).unsqueeze(0)
            inactive = eye | (dist2 > self.activation_radius**2) | (dist2 < 1e-12)
            if 0 < self.neighbor_limit < n_agents - 1:
                # Keep the k nearest teammates. Set self to +inf so it is never selected.
                d_rank = dist2.masked_fill(eye, float("inf"))
                keep_idx = d_rank.topk(
                    self.neighbor_limit, dim=-1, largest=False
                ).indices                                   # [B, N, k]
                keep = torch.zeros_like(inactive)
                keep.scatter_(-1, keep_idx, True)
                inactive = inactive | ~keep
            c_pair = (-2.0 * dp).masked_fill(inactive.unsqueeze(-1), 0.0)
            d_pair = torch.where(inactive, torch.ones_like(b_ij), 0.5 * b_ij)
            if static_mask is not None:
                neighbor_static = static_mask.unsqueeze(1).expand(-1, n_agents, -1)
                d_pair = torch.where(neighbor_static & ~inactive, b_ij, d_pair)
            rows_c.append(c_pair)
            rows_d.append(d_pair)

            # Backup direction: away from the nearest agent.
            dist2_masked = dist2.masked_fill(eye, float("inf"))
            nearest = dist2_masked.argmin(dim=-1)  # [B, N]
            batch_idx = torch.arange(b_sz, device=device).unsqueeze(1)
            agent_idx = torch.arange(n_agents, device=device).unsqueeze(0)
            backup_dir = torch.sign(dp[batch_idx, agent_idx, nearest])

        c_t = self._target_center_like(pos)
        if c_t is not None:
            p_r = pos - c_t
            range_norm = p_r.norm(dim=-1, keepdim=True)
            p_hat = p_r / range_norm.clamp(min=1e-9)
            speed = torch.sqrt((vel * vel).sum(-1, keepdim=True) + self.vel_eps**2)
            v_hat = vel / speed
            h_v = self.speed_limit_intercept + self.speed_limit_slope * range_norm - speed
            drift_i = self._cwh_drift(pos, vel)
            rhs = (
                self.speed_limit_slope * (p_hat * vel).sum(-1)
                - (v_hat * drift_i).sum(-1)
                + self.speed_limit_gain * h_v.squeeze(-1)
            )
            soft_cols.append(n_pair_cols)  # speed row: relaxed first
            rows_c.append(v_hat.unsqueeze(2))
            rows_d.append(rhs.unsqueeze(2))

            # Superellipsoid keep-out, relative degree 2, poles -mu1 and -mu2:
            # -grad(h).a <= v'H v + grad(h).drift + (mu1+mu2) hdot + mu1 mu2 h
            if self.target_keepout_axes is not None:
                rho = torch.tensor(
                    self.target_keepout_axes, dtype=dtype, device=device
                ).view(1, 1, 3)
                if self.target_keepout_mode == "halfspace":
                    # Half-space h_T = p_x - rho_x. The gradient is e_x and the curvature is 0.
                    h_t = p_r[..., 0] - float(rho[0, 0, 0])
                    grad = torch.zeros_like(p_r)
                    grad[..., 0] = 1.0
                    h_dot = vel[..., 0]
                    hess_quad = torch.zeros_like(h_t)
                    near_wall_thresh = 0.10          # metres
                else:
                    y = p_r / rho
                    n_exp = self.target_keepout_exponent
                    h_t = (y**n_exp).sum(-1) - 1.0  # [B, N]
                    grad = n_exp * y ** (n_exp - 1) / rho  # [B, N, 3]
                    h_dot = (grad * vel).sum(-1)
                    hess_quad = (
                        n_exp * (n_exp - 1) * y ** (n_exp - 2) / rho**2 * vel**2
                    ).sum(-1)
                    near_wall_thresh = 0.25          # dimensionless h_T
                mu1, mu2 = self.target_lambda1, self.target_lambda2
                d_t = (
                    hess_quad
                    + (grad * drift_i).sum(-1)
                    + (mu1 + mu2) * h_dot
                    + mu1 * mu2 * h_t
                )
                # The gradient is zero near the centre, so drop the row there.
                degenerate = (grad * grad).sum(-1) < 1e-12
                c_keep = (-grad).masked_fill(degenerate.unsqueeze(-1), 0.0)
                d_keep = torch.where(degenerate, torch.ones_like(d_t), d_t)
                soft_cols.append(n_pair_cols + 1)  # keep-out: relaxed last
                rows_c.append(c_keep.unsqueeze(2))
                rows_d.append(d_keep.unsqueeze(2))
                # If the keep-out is the nearest threat, the backup moves outward along
                # grad(h_T). The threshold depends on the mode: h_T is a distance for the
                # half-space and has no unit for the superellipsoid.
                near_wall = (h_t < near_wall_thresh).unsqueeze(-1)
                backup_dir = torch.where(near_wall, torch.sign(grad), backup_dir)

        if not rows_c:
            zero_c = torch.zeros((b_sz, n_agents, 1, 3), dtype=dtype, device=device)
            one_d = torch.ones((b_sz, n_agents, 1), dtype=dtype, device=device)
            return zero_c, one_d, backup_dir, soft_cols
        c_all = torch.cat(rows_c, dim=2)
        d_all = torch.cat(rows_d, dim=2)
        if static_mask is not None:
            # The env discards the action of a docked agent, so drop all its rows.
            self_static = static_mask.view(b_sz, n_agents, 1)
            c_all = c_all.masked_fill(self_static.unsqueeze(-1), 0.0)
            d_all = torch.where(self_static.expand_as(d_all), torch.ones_like(d_all), d_all)
        return c_all, d_all, backup_dir, soft_cols

    # Relaxation for infeasible translational QPs.
    def _solve_translational(
        self,
        a_nom: torch.Tensor,
        rows_c: torch.Tensor,
        rows_d: torch.Tensor,
        soft_cols: list[int],
        bound: float,
        keepout_col: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Solve in stages. Soft rows can be relaxed. Pair rows are never relaxed.

        Stage 1 uses all rows. If there is no solution, solve again without the speed
        row, then without the keep-out row. The backup is used only if the pair rows
        and the box have no solution.

        When the keep-out row is dropped, the nominal becomes the maximum thrust
        outward from the wall. Return (a_star [M,3], hard_infeasible [M],
        n_relaxed, relaxed_mask [M]).
        """
        a_star, infeasible = self._solve_box_qp(a_nom, rows_c, rows_d, bound)
        n_relaxed = 0
        relaxed = torch.zeros_like(infeasible)
        for stage in range(len(soft_cols)):
            if not bool(infeasible.any()):
                break
            idx = infeasible.nonzero(as_tuple=True)[0]
            c_sub = rows_c[idx].clone()
            d_sub = rows_d[idx].clone()
            dropped = soft_cols[: stage + 1]
            for col in dropped:
                c_sub[:, col] = 0.0
                d_sub[:, col] = 1.0
            a_nom_sub = a_nom[idx]
            if keepout_col is not None and keepout_col in dropped:
                # The row is -grad(h_T).a <= d, so outward = -sign(c).
                outward = -torch.sign(rows_c[idx, keepout_col])
                has_wall = (outward != 0.0).any(dim=-1, keepdim=True)
                a_nom_sub = torch.where(has_wall, bound * outward, a_nom_sub)
            a_sub, infeasible_sub = self._solve_box_qp(a_nom_sub, c_sub, d_sub, bound)
            a_star[idx] = a_sub
            n_relaxed += int((~infeasible_sub).sum())
            relaxed = relaxed.clone()
            relaxed[idx] = ~infeasible_sub
            infeasible = infeasible.clone()
            infeasible[idx] = infeasible_sub
        return a_star, infeasible, n_relaxed, relaxed

    # Batched QP solve by enumeration of active sets.
    def _solve_box_qp(
        self,
        u_nom: torch.Tensor,
        rows_c: torch.Tensor,
        rows_d: torch.Tensor,
        bound: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """argmin ||u - u_nom||^2 s.t. rows_c u <= rows_d, |u|_inf <= bound.

        u_nom [M,3], rows_c [M,K,3], rows_d [M,K]. Return (u_star [M,3],
        infeasible [M] bool). u_star is always in the box. The caller applies the
        backup if infeasible is True.
        """
        # Divide each row by ||c_i||. This does not change the feasible set, but it
        # is necessary for conditioning. The keep-out gradient can reach 3e7 at 10 m
        # (n = 8), which gives false infeasible results. Zero rows (c = 0, d = 1)
        # do not change.
        row_norm = rows_c.norm(dim=-1, keepdim=True)  # [M, K, 1]
        scale = torch.where(row_norm > 1e-12, 1.0 / row_norm, torch.ones_like(row_norm))
        rows_c = rows_c * scale
        rows_d = rows_d * scale.squeeze(-1)

        u0 = u_nom.clamp(-bound, bound)
        # Fast path: if the box-clamped nominal satisfies every row, it is the optimum.
        resid0 = (rows_c * u0.unsqueeze(1)).sum(-1) - rows_d  # [M, K]
        need = (resid0 > 0.0).any(dim=-1)  # [M]
        if not bool(need.any()):
            return u0, torch.zeros_like(need)

        idx = need.nonzero(as_tuple=True)[0]
        # Use float64 for the closed-form KKT solves.
        x, infeasible_sub = self._project_kkt(
            u_nom[idx].double(), rows_c[idx].double(), rows_d[idx].double(), float(bound)
        )
        u_star = u0.clone()
        u_star[idx] = x.to(u_nom.dtype)
        infeasible = torch.zeros_like(need)
        infeasible[idx] = infeasible_sub
        # Clamp to the actuator box.
        return u_star.clamp(-bound, bound), infeasible

    def _combinations(self, n: int, r: int, device: torch.device) -> torch.Tensor:
        """[C(n,r), r] long tensor of constraint-index subsets, cached."""
        key = (n, r, str(device))
        combos = self._combo_cache.get(key)
        if combos is None:
            combos = torch.tensor(
                list(itertools.combinations(range(n), r)),
                dtype=torch.long,
                device=device,
            )
            self._combo_cache[key] = combos
        return combos

    def _project_kkt(
        self, u: torch.Tensor, rows_c: torch.Tensor, rows_d: torch.Tensor, bound: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact projection of u onto {C x <= d} and the box (float64).

        The optimum satisfies the KKT system x = u - A_S^T lam, A_S x = b_S,
        lam >= 0 for an active set S with |S| <= 3. A candidate that is primal and
        dual feasible is the optimum. Tiers 1 and 2 run for all problems. Tier 3
        (Cramer's rule) runs only for problems not solved by tiers 1-2.
        Return (x [m,3], infeasible [m] bool).
        """
        m = u.shape[0]
        device = u.device
        f64 = torch.float64
        eye = torch.eye(3, dtype=f64, device=device)
        a_full = torch.cat([rows_c, eye.expand(m, 3, 3), -eye.expand(m, 3, 3)], dim=1)
        b_full = torch.cat(
            [rows_d, torch.full((m, 6), bound, dtype=f64, device=device)], dim=1
        )
        norms = a_full.norm(dim=-1)
        a_n = a_full / norms.clamp_min(1e-12).unsqueeze(-1)  # [m, R, 3]
        b_n = b_full / norms.clamp_min(1e-12)  # [m, R]
        n_con = a_n.shape[1]
        # Tolerances (m).
        tol_feas = 1e-9
        tol_dual = 1e-11
        det_tol = 1e-12
        inf = float("inf")

        # Do not use torch.einsum here. It calls batched BLAS, which crashes on macOS
        # for some large batch shapes. Multiply and sum gives the same result.
        def evaluate(a_sub, b_sub, u_sub, x_cand, lam_ok):
            """Best valid candidate per problem: (obj [q], x [q,3])."""
            # resid[q,c,r] = sum_x x_cand[q,c,x] * a_sub[q,r,x]
            resid = (x_cand.unsqueeze(2) * a_sub.unsqueeze(1)).sum(-1) - b_sub.unsqueeze(1)
            valid = lam_ok & (resid.amax(dim=-1) <= tol_feas)
            obj = ((x_cand - u_sub.unsqueeze(1)) ** 2).sum(-1)
            obj = torch.where(valid, obj, torch.full_like(obj, inf))
            o, j = obj.min(dim=1)
            xb = x_cand.gather(1, j.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
            return o, xb

        # Tier 1: one active constraint (unit rows, so Gram = 1).
        lam1 = (a_n * u.unsqueeze(1)).sum(-1) - b_n  # [m, R]
        x1 = u.unsqueeze(1) - lam1.unsqueeze(-1) * a_n  # [m, R, 3]
        best_obj, best_x = evaluate(a_n, b_n, u, x1, lam1 >= -tol_dual)

        # Tier 2: pairs (closed-form 2x2 KKT).
        pairs = self._combinations(n_con, 2, device)  # [P, 2]
        a2 = a_n[:, pairs]  # [m, P, 2, 3]
        b2 = b_n[:, pairs]  # [m, P, 2]
        g01 = (a2[..., 0, :] * a2[..., 1, :]).sum(-1)  # [m, P] (diagonal = 1)
        r2 = (a2 * u.unsqueeze(1).unsqueeze(1)).sum(-1) - b2  # [m, P, 2]
        det2 = 1.0 - g01 * g01
        det_ok2 = det2.abs() > det_tol
        det2_safe = torch.where(det_ok2, det2, torch.ones_like(det2))
        lam2 = torch.stack(
            [
                (r2[..., 0] - g01 * r2[..., 1]) / det2_safe,
                (r2[..., 1] - g01 * r2[..., 0]) / det2_safe,
            ],
            dim=-1,
        )  # [m, P, 2]
        x2 = u.unsqueeze(1) - (lam2.unsqueeze(-1) * a2).sum(-2)
        o2, xb2 = evaluate(a_n, b_n, u, x2, det_ok2 & (lam2 >= -tol_dual).all(-1))
        upd = o2 < best_obj
        best_x = torch.where(upd.unsqueeze(-1), xb2, best_x)
        best_obj = torch.minimum(best_obj, o2)

        # Tier 3: triples (Cramer's rule), only for unsolved problems.
        pending = ~torch.isfinite(best_obj)
        if bool(pending.any()):
            pidx = pending.nonzero(as_tuple=True)[0]
            a_p, b_p, u_p = a_n[pidx], b_n[pidx], u[pidx]
            triples = self._combinations(n_con, 3, device)  # [T, 3]
            a3 = a_p[:, triples]  # [q, T, 3, 3]
            b3 = b_p[:, triples]
            # gram[q,t,i,j] = sum_x a3[q,t,i,x] * a3[q,t,j,x]
            gram = (a3.unsqueeze(3) * a3.unsqueeze(2)).sum(-1)  # [q, T, 3, 3]
            r3 = (a3 * u_p.unsqueeze(1).unsqueeze(1)).sum(-1) - b3  # [q, T, 3]
            g0, g1, g2 = gram[..., 0], gram[..., 1], gram[..., 2]  # columns
            cross12 = torch.cross(g1, g2, dim=-1)
            det3 = (g0 * cross12).sum(-1)
            det_ok3 = det3.abs() > det_tol
            det3_safe = torch.where(det_ok3, det3, torch.ones_like(det3))
            lam3 = torch.stack(
                [
                    (r3 * cross12).sum(-1) / det3_safe,
                    (g0 * torch.cross(r3, g2, dim=-1)).sum(-1) / det3_safe,
                    (g0 * torch.cross(g1, r3, dim=-1)).sum(-1) / det3_safe,
                ],
                dim=-1,
            )  # [q, T, 3]
            x3 = u_p.unsqueeze(1) - (lam3.unsqueeze(-1) * a3).sum(-2)
            o3, xb3 = evaluate(a_p, b_p, u_p, x3, det_ok3 & (lam3 >= -tol_dual).all(-1))
            upd3 = o3 < best_obj[pidx]
            best_x[pidx] = torch.where(upd3.unsqueeze(-1), xb3, best_x[pidx])
            best_obj[pidx] = torch.minimum(best_obj[pidx], o3)

        infeasible = ~torch.isfinite(best_obj)
        # Value for infeasible problems. The caller applies the backup.
        best_x = torch.where(infeasible.unsqueeze(-1), u.clamp(-bound, bound), best_x)
        return best_x, infeasible

    # Main entry point.
    def filter_action(
        self,
        *,
        pos: torch.Tensor,
        vel: torch.Tensor,
        action: torch.Tensor,
        quat: Optional[torch.Tensor] = None,
        ang_vel: Optional[torch.Tensor] = None,
        static_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the action onto the safe set with the smallest change.

        The QP splits into a translational and a rotational 3-variable QP.
        static_mask [B, N] marks docked agents. Return (safe_action in [-1, 1],
        intervention [B, N, 1] = ||u_safe - u_nom|| in normalised units).

        Runs under no_grad: the filter is part of the environment.
        """
        with torch.no_grad():
            return self._filter_action_impl(
                pos=pos,
                vel=vel,
                action=action,
                quat=quat,
                ang_vel=ang_vel,
                static_mask=static_mask,
            )

    def _filter_action_impl(
        self,
        *,
        pos: torch.Tensor,
        vel: torch.Tensor,
        action: torch.Tensor,
        quat: Optional[torch.Tensor] = None,
        ang_vel: Optional[torch.Tensor] = None,
        static_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b_sz, n_agents, act_dim = action.shape
        action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        pos = torch.nan_to_num(pos, nan=0.0, posinf=1e6, neginf=-1e6)
        vel = torch.nan_to_num(vel, nan=0.0, posinf=1e6, neginf=-1e6)
        safe_action = action.clone()
        self.last_infeasible = 0
        self.last_relaxed = 0
        infeasible_mask = torch.zeros(
            (b_sz, n_agents, 1), dtype=torch.bool, device=action.device
        )

        # Translational QP (3 variables).
        a_bar = self.linear_accel_scale
        a_nom = action[..., :3] * a_bar  # m/s^2, inside the box
        rows_c, rows_d, backup_dir, soft_cols = self._translational_rows(
            pos, vel, static_mask
        )
        # soft_cols is [speed_col] or [speed_col, keepout_col].
        keepout_col = soft_cols[1] if len(soft_cols) >= 2 else None
        a_star, infeasible_t, n_relaxed, relaxed_t = self._solve_translational(
            a_nom.reshape(-1, 3),
            rows_c.reshape(b_sz * n_agents, -1, 3),
            rows_d.reshape(b_sz * n_agents, -1),
            soft_cols,
            a_bar,
            keepout_col=keepout_col,
        )
        self.last_relaxed += n_relaxed
        self.last_relaxed_mask = relaxed_t.reshape(b_sz, n_agents, 1)
        a_star = a_star.reshape(b_sz, n_agents, 3)
        infeasible_t = infeasible_t.reshape(b_sz, n_agents, 1)
        if bool(infeasible_t.any()):
            # Backup: maximum thrust away from the nearest threat (neighbour or keep-out).
            # Used only if the pair rows and the box have no solution.
            a_star = torch.where(infeasible_t, a_bar * backup_dir, a_star)
            self.last_infeasible += int(infeasible_t.sum())
            infeasible_mask |= infeasible_t
        safe_action[..., :3] = a_star / a_bar

        # Rotational QP (3 variables).
        if act_dim >= 6 and ang_vel is not None:
            al_bar = self.angular_accel_scale
            w = torch.nan_to_num(ang_vel, nan=0.0, posinf=1e6, neginf=-1e6)
            al_nom = action[..., 3:6] * al_bar
            c_w = (2.0 * w).unsqueeze(2)  # [B, N, 1, 3]
            d_w = (
                self.ang_speed_gain
                * (self.max_ang_speed**2 - (w * w).sum(-1, keepdim=True))
            )  # [B, N, 1]
            al_star, infeasible_r = self._solve_box_qp(
                al_nom.reshape(-1, 3),
                c_w.reshape(b_sz * n_agents, 1, 3),
                d_w.reshape(b_sz * n_agents, 1),
                al_bar,
            )
            al_star = al_star.reshape(b_sz, n_agents, 3)
            infeasible_r = infeasible_r.reshape(b_sz, n_agents, 1)
            if bool(infeasible_r.any()):
                # Backup: maximum despin.
                al_star = torch.where(infeasible_r, -al_bar * torch.sign(w), al_star)
                self.last_infeasible += int(infeasible_r.sum())
                infeasible_mask |= infeasible_r
            safe_action[..., 3:6] = al_star / al_bar

        self.last_infeasible_mask = infeasible_mask
        safe_action = safe_action.clamp(-1.0, 1.0)
        intervention = (safe_action - action).norm(dim=-1, keepdim=True)
        return safe_action, intervention
