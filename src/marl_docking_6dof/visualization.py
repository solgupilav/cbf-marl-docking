from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def save_3d_episode_animation(
    *,
    positions: np.ndarray,
    goals: np.ndarray,
    target_centers: np.ndarray | None = None,
    target_sizes: np.ndarray | None = None,
    agent_sizes: np.ndarray | None = None,
    attitude_errors_deg: np.ndarray | None = None,
    output_path: str | Path,
    fps: int = 20,
    title: str = "6-DoF Docking Evaluation",
    dynamic_zoom: bool = True,
    zoom_padding: float = 0.25,
    min_view_radius: float = 0.35,
    zoom_smoothing: float = 0.7,
) -> Path:
    """Make a 3-D animation of the agent trajectories and save it.

    Args:
        positions: [T, N, 3].
        goals: [T, N, 3].
        target_centers: optional [T, 3].
        target_sizes: optional [3] or [T, 3].
        agent_sizes: optional [3], [N, 3] or [T, N, 3].
        attitude_errors_deg: optional [T, N] attitude error in degrees. If given,
            a time-series plot is added at the bottom.
        output_path: .gif or .mp4 path.
        fps: frames per second.
        title: plot title.
        dynamic_zoom: if True, the camera limits change each frame.
        zoom_padding: margin around the tracked objects, as a fraction.
        min_view_radius: minimum camera half-width.
        zoom_smoothing: exponential smoothing of the camera, in [0, 1).
    """
    positions = np.asarray(positions, dtype=float)
    goals = np.asarray(goals, dtype=float)
    if target_centers is not None:
        target_centers = np.asarray(target_centers, dtype=float)
    if target_sizes is not None:
        target_sizes = np.asarray(target_sizes, dtype=float)
    if agent_sizes is not None:
        agent_sizes = np.asarray(agent_sizes, dtype=float)
    positions = np.nan_to_num(positions, nan=0.0, posinf=1e6, neginf=-1e6)
    goals = np.nan_to_num(goals, nan=0.0, posinf=1e6, neginf=-1e6)
    if target_centers is not None:
        target_centers = np.nan_to_num(target_centers, nan=0.0, posinf=1e6, neginf=-1e6)
    if target_sizes is not None:
        target_sizes = np.nan_to_num(target_sizes, nan=1.0, posinf=1e3, neginf=1e-3)
    if agent_sizes is not None:
        agent_sizes = np.nan_to_num(agent_sizes, nan=0.1, posinf=1e3, neginf=1e-3)
    positions = np.clip(positions, -1e6, 1e6)
    goals = np.clip(goals, -1e6, 1e6)
    if target_centers is not None:
        target_centers = np.clip(target_centers, -1e6, 1e6)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t_steps, n_agents, _ = positions.shape
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))

    if attitude_errors_deg is not None:
        attitude_errors_deg = np.asarray(attitude_errors_deg, dtype=float)
        if attitude_errors_deg.ndim == 1:
            attitude_errors_deg = attitude_errors_deg[:, None]
        attitude_errors_deg = np.nan_to_num(
            attitude_errors_deg, nan=0.0, posinf=0.0, neginf=0.0
        )
        # Pad or cut each array to the number of frames.
        if attitude_errors_deg.shape[0] < t_steps:
            pad = t_steps - attitude_errors_deg.shape[0]
            attitude_errors_deg = np.concatenate(
                [attitude_errors_deg, np.repeat(attitude_errors_deg[-1:], pad, axis=0)],
                axis=0,
            )
        elif attitude_errors_deg.shape[0] > t_steps:
            attitude_errors_deg = attitude_errors_deg[:t_steps]

    show_attitude = attitude_errors_deg is not None
    if show_attitude:
        fig = plt.figure(figsize=(14, 9))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
        ax = fig.add_subplot(gs[0, 0], projection="3d")
        ax_face = fig.add_subplot(gs[0, 1])
        ax_att = fig.add_subplot(gs[1, :])
    else:
        fig = plt.figure(figsize=(14, 7))
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax_face = fig.add_subplot(1, 2, 2)
        ax_att = None
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax_face.set_title("Docking Face View (Y-Z)")
    ax_face.set_xlabel("y")
    ax_face.set_ylabel("z")
    ax_face.set_aspect("equal", adjustable="box")

    center0, radius0 = _compute_frame_bounds(
        positions=positions[0],
        goals=goals[0],
        target_center=(target_centers[0] if target_centers is not None else None),
        target_size=_target_size_at_frame(target_sizes, 0),
        zoom_padding=zoom_padding,
        min_view_radius=min_view_radius,
    )
    _set_axis_limits_from_center_radius(
        ax=ax,
        center=center0,
        radius=radius0,
    )
    face_center0, face_radius0 = _compute_face_bounds(
        positions=positions[0],
        goals=goals[0],
        target_center=(target_centers[0] if target_centers is not None else None),
        target_size=_target_size_at_frame(target_sizes, 0),
        zoom_padding=zoom_padding,
        min_view_radius=min_view_radius,
    )
    _set_face_limits(ax=ax_face, center=face_center0, radius=face_radius0)

    goal_scatter = ax.scatter(
        goals[0, :, 0],
        goals[0, :, 1],
        goals[0, :, 2],
        marker="x",
        s=80,
        c=colors,
        label="goals",
    )
    agent_scatter = ax.scatter(
        positions[0, :, 0],
        positions[0, :, 1],
        positions[0, :, 2],
        marker="o",
        s=60,
        c=colors,
        label="agents",
    )
    goal_face_scatter = ax_face.scatter(
        goals[0, :, 1],
        goals[0, :, 2],
        marker="x",
        s=80,
        c=colors,
        label="goals",
    )
    agent_face_scatter = ax_face.scatter(
        positions[0, :, 1],
        positions[0, :, 2],
        marker="o",
        s=60,
        c=colors,
        label="agents",
    )

    trails = [ax.plot([], [], [], color=colors[i], alpha=0.6)[0] for i in range(n_agents)]
    face_trails = [ax_face.plot([], [], color=colors[i], alpha=0.6)[0] for i in range(n_agents)]
    target_edges = None
    target_face_rect = None
    if target_centers is not None:
        target_edges = []
        for edge_idx, _ in enumerate(_BOX_EDGES):
            label = "target spacecraft" if edge_idx == 0 else None
            line = ax.plot([], [], [], color="black", linewidth=1.5, alpha=0.9, label=label)[0]
            target_edges.append(line)
        target_face_rect = ax_face.plot([], [], color="black", linewidth=1.5, alpha=0.9, label="target face")[0]
    chaser_edges = []
    chaser_face_rects = []
    if agent_sizes is not None:
        for agent_idx in range(n_agents):
            edges = []
            for edge_idx, _ in enumerate(_BOX_EDGES):
                label = f"chaser {agent_idx}" if edge_idx == 0 else None
                line = ax.plot(
                    [],
                    [],
                    [],
                    color=colors[agent_idx],
                    linewidth=1.2,
                    alpha=0.85,
                    label=label,
                )[0]
                edges.append(line)
            chaser_edges.append(edges)
            rect = ax_face.plot([], [], color=colors[agent_idx], linewidth=1.2, alpha=0.85)[0]
            chaser_face_rects.append(rect)
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
    ax.legend(loc="upper right")
    ax_face.legend(loc="upper right")

    att_lines = []
    att_marker = None
    if ax_att is not None:
        ax_att.set_title("Attitude error to target")
        ax_att.set_xlabel("t (steps)")
        ax_att.set_ylabel("attitude error (deg)")
        ax_att.set_xlim(0, max(t_steps - 1, 1))
        finite_att = attitude_errors_deg[np.isfinite(attitude_errors_deg)]
        y_max = float(finite_att.max()) if finite_att.size else 1.0
        ax_att.set_ylim(0.0, max(y_max * 1.1, 1.0))
        ax_att.grid(True, alpha=0.3)
        for i in range(n_agents):
            att_lines.append(
                ax_att.plot([], [], color=colors[i], linewidth=1.5, label=f"agent {i}")[0]
            )
        att_marker = ax_att.axvline(0, color="grey", linestyle="--", alpha=0.6)
        ax_att.legend(loc="upper right", ncol=min(n_agents, 4), fontsize=8)

    prev_center = center0
    prev_radius = radius0
    prev_face_center = face_center0
    prev_face_radius = face_radius0

    def _update(frame_idx: int):
        nonlocal prev_center, prev_radius, prev_face_center, prev_face_radius
        current_pos = positions[frame_idx]
        current_goals = goals[frame_idx]
        agent_scatter._offsets3d = (
            current_pos[:, 0],
            current_pos[:, 1],
            current_pos[:, 2],
        )
        agent_face_scatter.set_offsets(current_pos[:, 1:3])
        goal_scatter._offsets3d = (
            current_goals[:, 0],
            current_goals[:, 1],
            current_goals[:, 2],
        )
        goal_face_scatter.set_offsets(current_goals[:, 1:3])
        for i, line in enumerate(trails):
            line.set_data(positions[: frame_idx + 1, i, 0], positions[: frame_idx + 1, i, 1])
            line.set_3d_properties(positions[: frame_idx + 1, i, 2])
            face_trails[i].set_data(positions[: frame_idx + 1, i, 1], positions[: frame_idx + 1, i, 2])
        if target_edges is not None:
            corners = _box_corners(
                center=target_centers[frame_idx],
                size=_target_size_at_frame(target_sizes, frame_idx),
            )
            for edge_line, (start, end) in zip(target_edges, _BOX_EDGES):
                pts = corners[[start, end]]
                edge_line.set_data(pts[:, 0], pts[:, 1])
                edge_line.set_3d_properties(pts[:, 2])
            face_poly = _rect_face_polygon(
                center_yz=target_centers[frame_idx, 1:3],
                size_yz=_target_size_at_frame(target_sizes, frame_idx)[1:3],
            )
            target_face_rect.set_data(face_poly[:, 0], face_poly[:, 1])
        if chaser_edges:
            for agent_idx, edges in enumerate(chaser_edges):
                agent_size = _agent_size_at_frame(agent_sizes, frame_idx, agent_idx)
                corners = _box_corners(
                    center=current_pos[agent_idx],
                    size=agent_size,
                )
                for edge_line, (start, end) in zip(edges, _BOX_EDGES):
                    pts = corners[[start, end]]
                    edge_line.set_data(pts[:, 0], pts[:, 1])
                    edge_line.set_3d_properties(pts[:, 2])
                face_poly = _rect_face_polygon(
                    center_yz=current_pos[agent_idx, 1:3],
                    size_yz=agent_size[1:3],
                )
                chaser_face_rects[agent_idx].set_data(face_poly[:, 0], face_poly[:, 1])
        if dynamic_zoom:
            center_t, radius_t = _compute_frame_bounds(
                positions=current_pos,
                goals=current_goals,
                target_center=(target_centers[frame_idx] if target_centers is not None else None),
                target_size=_target_size_at_frame(target_sizes, frame_idx),
                zoom_padding=zoom_padding,
                min_view_radius=min_view_radius,
            )
            alpha = float(np.clip(zoom_smoothing, 0.0, 0.95))
            prev_center = alpha * prev_center + (1.0 - alpha) * center_t
            prev_radius = alpha * prev_radius + (1.0 - alpha) * radius_t
            _set_axis_limits_from_center_radius(ax=ax, center=prev_center, radius=prev_radius)
            face_center_t, face_radius_t = _compute_face_bounds(
                positions=current_pos,
                goals=current_goals,
                target_center=(target_centers[frame_idx] if target_centers is not None else None),
                target_size=_target_size_at_frame(target_sizes, frame_idx),
                zoom_padding=zoom_padding,
                min_view_radius=min_view_radius,
            )
            prev_face_center = alpha * prev_face_center + (1.0 - alpha) * face_center_t
            prev_face_radius = alpha * prev_face_radius + (1.0 - alpha) * face_radius_t
            _set_face_limits(ax=ax_face, center=prev_face_center, radius=prev_face_radius)
        if ax_att is not None:
            ts = np.arange(frame_idx + 1)
            for i, line in enumerate(att_lines):
                line.set_data(ts, attitude_errors_deg[: frame_idx + 1, i])
            att_marker.set_xdata([frame_idx, frame_idx])
        time_text.set_text(f"t = {frame_idx}")
        artists = [
            agent_scatter,
            goal_scatter,
            agent_face_scatter,
            goal_face_scatter,
            *trails,
            *face_trails,
            time_text,
        ]
        if ax_att is not None:
            artists.extend(att_lines)
            artists.append(att_marker)
        if target_edges is not None:
            artists.extend(target_edges)
            artists.append(target_face_rect)
        if chaser_edges:
            for edges in chaser_edges:
                artists.extend(edges)
            artists.extend(chaser_face_rects)
        return artists

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=t_steps,
        interval=1000 / max(fps, 1),
        blit=False,
    )

    suffix = output_path.suffix.lower()
    if suffix == ".mp4":
        try:
            import imageio.v2 as imageio

            frames = []
            for i in range(t_steps):
                _update(i)
                fig.canvas.draw()
                rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
                frames.append(rgba[..., :3].copy())
            with imageio.get_writer(
                str(output_path),
                fps=fps,
            ) as writer:
                for frame in frames:
                    writer.append_data(frame)
        except Exception as exc:
            print(f"[Visualization] MP4 export failed, falling back to GIF: {exc}")
            fallback = output_path.with_suffix(".gif")
            writer = animation.PillowWriter(fps=fps)
            anim.save(str(fallback), writer=writer)
            output_path = fallback
    elif suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
        anim.save(str(output_path), writer=writer)
    else:
        try:
            writer = animation.FFMpegWriter(fps=fps)
            anim.save(str(output_path), writer=writer)
        except Exception:
            # Use GIF if ffmpeg is not available.
            fallback = output_path.with_suffix(".gif")
            writer = animation.PillowWriter(fps=fps)
            anim.save(str(fallback), writer=writer)
            output_path = fallback

    plt.close(fig)
    return output_path


def _compute_frame_bounds(
    *,
    positions: np.ndarray,
    goals: np.ndarray,
    target_center: np.ndarray | None,
    target_size: np.ndarray,
    zoom_padding: float,
    min_view_radius: float,
) -> tuple[np.ndarray, float]:
    pts = [positions.reshape(-1, 3), goals.reshape(-1, 3)]
    if target_center is not None:
        pts.append(_box_corners(center=target_center, size=target_size))
    pts = np.concatenate(pts, axis=0)
    finite_rows = np.isfinite(pts).all(axis=1)
    if not np.any(finite_rows):
        return np.zeros(3, dtype=float), float(max(min_view_radius, 1.0))
    pts = pts[finite_rows]
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins) + 1e-6)
    radius = max(min_view_radius, 0.5 * span * (1.0 + max(zoom_padding, 0.0)))
    return center, radius


def _set_axis_limits_from_center_radius(*, ax, center: np.ndarray, radius: float) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _compute_face_bounds(
    *,
    positions: np.ndarray,
    goals: np.ndarray,
    target_center: np.ndarray | None,
    target_size: np.ndarray,
    zoom_padding: float,
    min_view_radius: float,
) -> tuple[np.ndarray, float]:
    pts = [positions[:, 1:3], goals[:, 1:3]]
    if target_center is not None:
        face_poly = _rect_face_polygon(center_yz=target_center[1:3], size_yz=target_size[1:3])
        pts.append(face_poly[:, 0:2])
    pts2d = np.concatenate(pts, axis=0)
    finite_rows = np.isfinite(pts2d).all(axis=1)
    if not np.any(finite_rows):
        return np.zeros(2, dtype=float), float(max(min_view_radius, 1.0))
    pts2d = pts2d[finite_rows]
    mins = pts2d.min(axis=0)
    maxs = pts2d.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins) + 1e-6)
    radius = max(min_view_radius, 0.5 * span * (1.0 + max(zoom_padding, 0.0)))
    return center, radius


def _set_face_limits(*, ax, center: np.ndarray, radius: float) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)


_BOX_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def _target_size_at_frame(target_sizes: np.ndarray | None, frame_idx: int) -> np.ndarray:
    if target_sizes is None:
        return np.array([0.8, 0.4, 0.3], dtype=float)
    target_sizes = np.asarray(target_sizes)
    if target_sizes.ndim == 1:
        return target_sizes.astype(float)
    return target_sizes[frame_idx].astype(float)


def _agent_size_at_frame(
    agent_sizes: np.ndarray | None,
    frame_idx: int,
    agent_idx: int,
) -> np.ndarray:
    if agent_sizes is None:
        return np.array([0.26, 0.23, 0.45], dtype=float)
    agent_sizes = np.asarray(agent_sizes)
    if agent_sizes.ndim == 1:
        return agent_sizes.astype(float)
    if agent_sizes.ndim == 2:
        return agent_sizes[agent_idx].astype(float)
    return agent_sizes[frame_idx, agent_idx].astype(float)


def _box_corners(*, center: np.ndarray, size: np.ndarray) -> np.ndarray:
    cx, cy, cz = center.astype(float)
    sx, sy, sz = size.astype(float)
    hx, hy, hz = 0.5 * sx, 0.5 * sy, 0.5 * sz
    return np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
        dtype=float,
    )


def _rect_face_polygon(*, center_yz: np.ndarray, size_yz: np.ndarray) -> np.ndarray:
    cy, cz = center_yz.astype(float)
    sy, sz = size_yz.astype(float)
    hy, hz = 0.5 * sy, 0.5 * sz
    return np.array(
        [
            [cy - hy, cz - hz],
            [cy + hy, cz - hz],
            [cy + hy, cz + hz],
            [cy - hy, cz + hz],
            [cy - hy, cz - hz],
        ],
        dtype=float,
    )
