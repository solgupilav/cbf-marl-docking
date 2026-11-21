"""Draw Figures 1-7 of the paper as PDF and PNG.

Data:
  results/figure_data/curve.npz                  training curve and curriculum gates (Fig. 2)
  results/figure_data/figdata.npz                rollout of seed 3 (Fig. 3)
  results/main/seed*.npz                         per-episode arrays, 5 seeds (Figs. 4, 5)
  results/figure_data/intervention_seed*.npz     per-step filter intervention (Fig. 6)
  results/main/seed*.json                        per-seed results (Fig. 7)
"""
import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle

# Plot style: Times text, STIX maths, closed frame with inward ticks, no grid,
# colour-blind safe palette. Fonts are embedded as Type 42 for PDF eXpress.
INK = "#1A1A1A"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 10, "axes.labelsize": 10, "legend.fontsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.5, "axes.grid": False, "axes.axisbelow": True,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
    "lines.linewidth": 1.0, "lines.solid_capstyle": "butt", "lines.dash_capstyle": "butt",
    "patch.linewidth": 0.5,
    "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": INK,
    "legend.framealpha": 1.0, "legend.borderpad": 0.35, "legend.borderaxespad": 0.6,
    "legend.handlelength": 1.8, "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.28, "legend.columnspacing": 1.0,
    "figure.dpi": 200, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
# Printed widths in the AeroConf class: column 243 bp, text 504 bp.
# save() fits each figure to these widths, so FONT is the printed size.
COL, DBL = 3.375, 7.0
FONT = 10
C = {"blue": "#1F4E79", "red": "#A51C30", "green": "#2E6B45", "orange": "#B8651B",
     "purple": "#5E4B8B", "grey": "#4D4D4D", "lgrey": "#B5B5B5", "teal": "#23707A"}
PRIMARY, ACCENT, SECOND = C["blue"], C["red"], C["orange"]
FILL = "#C9D6E6"                      # light tint of PRIMARY for filled areas
SEED_COLS = ["#1F4E79", "#B5452B", "#2E7D4F", "#6A4C93", "#B8860B"]
# Nine agents: Paul Tol 'muted' scheme.
AGENT_COLS = ["#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
              "#DDCC77", "#CC6677", "#882255", "#AA4499"]
SP = os.environ.get("FIGDATA_DIR", "results/figure_data")
OUTDIR = os.environ.get("FIGDIR", "figures")
os.makedirs(OUTDIR, exist_ok=True)


def fit_width(fig, target, bbox_fn, scale_height=False, iters=10):
    """Resize the figure until its cropped width is `target` inches."""
    goal = target * 0.999
    for _ in range(iters):
        fig.canvas.draw()
        bb = bbox_fn(fig)
        ratio = goal / bb.width
        if abs(ratio - 1.0) < 2e-4:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * ratio, h * ratio if scale_height else h, forward=True)
    fig.canvas.draw()
    return bbox_fn(fig)


def save(fig, name, pad=None, bbox=None, width=None, scale_height=False):
    """Save as PDF and PNG. `width` fits the cropped figure to COL or DBL.
    `bbox` can be a function fig -> Bbox (inches) for a custom crop.
    """
    pad = plt.rcParams["savefig.pad_inches"] if pad is None else pad
    if bbox is None:
        bbox_fn = lambda f: f.get_tightbbox(f.canvas.get_renderer()).padded(pad)
    elif callable(bbox):
        bbox_fn = bbox
    else:
        bbox_fn = lambda f: bbox
    if width is not None:
        bb = fit_width(fig, width, bbox_fn, scale_height=scale_height)
    else:
        fig.canvas.draw(); bb = bbox_fn(fig)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"), bbox_inches=bb, pad_inches=0)
    plt.close(fig)
    print(f"  wrote {name}.pdf  ({bb.width:.3f} x {bb.height:.3f} in)")


RESULTS = os.environ.get("RESULTS_DIR", "results")


def load_json(*parts):
    """Read a results JSON. Return None if the file does not exist."""
    import json
    p = os.path.join(RESULTS, *parts)
    return json.load(open(p)) if os.path.exists(p) else None


def seed_results():
    """Results of the five baseline seeds on the evaluation seed."""
    out = []
    for i in range(5):
        d = load_json("main", f"seed{i}.json")
        if d is not None:
            out.append(d)
    return out


def panel(ax, label, x=0.0, y=1.02):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=FONT, ha="left", va="bottom")


def legend(ax, *args, **kw):
    """Square legend with a hairline frame matching the axes."""
    leg = ax.legend(*args, **kw)
    leg.get_frame().set_linewidth(0.5)
    return leg


def sharp_hist(ax, data, bins, color=PRIMARY, fill=FILL):
    """Histogram as a light fill under an outline."""
    _, edges, _ = ax.hist(data, bins=bins, histtype="stepfilled", color=fill, lw=0)
    ax.hist(data, bins=edges, histtype="step", color=color, lw=0.7)
    return edges


def vline_label(ax, x, text, y=0.97, side="right", color=INK, pad=3):
    """Label a vertical reference line along the line."""
    off = pad if side == "right" else -pad
    ax.annotate(text, (x, y), xycoords=("data", "axes fraction"), xytext=(off, 0),
                textcoords="offset points", rotation=90, ha="left" if side == "right" else "right",
                va="top", fontsize=FONT, color=color)


def smooth(y, k):
    if y is None or len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="valid")


def fig1_system():
    FH = 2.9
    fig = plt.figure(figsize=(DBL, FH))
    # (a) Target face and port grid.
    ax = fig.add_axes([0.07, 0.15, 0.31, 0.77])
    npz = os.path.join(SP, "figdata.npz")
    if os.path.exists(npz):
        d = np.load(npz); ts = d["target_size"]; th_y, th_z = ts[1] / 2, ts[2] / 2
        ports = [(float(g[1]), float(g[2])) for g in d["goals"][0]]
    else:
        th_y, th_z = 1.213, 1.213
        ports = [(sy * 1.10, sz * 1.10) for sy in (-1, 0, 1) for sz in (-1, 0, 1)]
    ax.add_patch(Rectangle((-th_y, -th_z), 2 * th_y, 2 * th_z, fc="#E8E8E8",
                           ec=INK, lw=0.6, joinstyle="miter", zorder=1))
    rng = np.random.default_rng(5)
    for (py, pz) in ports:
        r = 0.9 + rng.uniform(0, 0.45)
        ang = (math.atan2(pz, py) if (py or pz) else rng.uniform(0, 2 * math.pi)) + rng.normal(0, 0.45)
        ax.annotate("", xy=(py, pz), xytext=(py + r * math.cos(ang), pz + r * math.sin(ang)),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.16", lw=0.7,
                                    color=C["orange"], shrinkA=0, shrinkB=4),
                    zorder=2)
        ax.add_patch(Circle((py, pz), 0.10, fc=C["blue"], ec="white", lw=0.6, zorder=3))
    lim = max(th_y, th_z) + 1.45
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim + 0.8); ax.set_aspect("equal")
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xlabel(r"$y$ [m]"); ax.set_ylabel(r"$z$ [m]")
    ax.plot([], [], "o", color=C["blue"], ms=5, label="docking port")
    ax.plot([], [], "-", color=C["orange"], lw=0.7, label="chaser approach")
    legend(ax, loc="upper right", fontsize=FONT)
    panel(ax, "(a)")

    # (b) Block diagram. Coordinates are draw.io px, y down.
    W, H = 500, 250
    bx = fig.add_axes([0.43, 0.06, 0.56, 0.86]); bx.axis("off")
    bx.set_xlim(0, W); bx.set_ylim(H, 0); bx.set_aspect("equal", anchor="NW")
    s = min(0.56 * DBL / W, 0.86 * FH / H) * 72            # pt per draw.io px
    lw = max(1.0 * s, 0.5)                                  # IEEE floor 0.5 pt
    sans = dict(family="Arial", math_fontfamily="custom")   # Helvetica metrics
    def box(x, y, label, w=120, h=60):
        bx.add_patch(Rectangle((x, y), w, h, fc="white", ec="black", lw=lw,
                               joinstyle="miter", clip_on=False))
        bx.text(x + w / 2, y + h / 2 + 0.358 * FONT / s, label, ha="center",
                va="baseline", fontsize=FONT, **sans)
    def edge(pts, label=None, at=None):
        # Arrow head in the draw.io 'classic' style.
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        n = math.hypot(x2 - x1, y2 - y1); ux, uy = (x2 - x1) / n, (y2 - y1) / n
        tx, ty = x2 - 1.118 * ux, y2 - 1.118 * uy
        bxx, byy = tx - 7 * ux, ty - 7 * uy
        notch = (tx - 5.25 * ux, ty - 5.25 * uy)
        bx.add_patch(Polygon([(tx, ty), (bxx - 3.5 * uy, byy + 3.5 * ux), notch,
                              (bxx + 3.5 * uy, byy - 3.5 * ux)], closed=True,
                             fc="black", ec="black", lw=lw, joinstyle="miter", clip_on=False))
        xs, ys = zip(*(list(pts[:-1]) + [notch]))
        bx.plot(xs, ys, color="black", lw=lw, solid_joinstyle="miter",
                solid_capstyle="butt", clip_on=False)
        if label:
            bx.text(*at, label, ha="center", va="center", fontsize=FONT, **sans,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    # Row 1: observation -> policy -> filter. Row 2: plant and feedback.
    box(0, 30, "observation")
    box(160, 30, "policy")
    box(350, 30, "HOCBF-QP filter", w=150)
    box(365, 180, "plant")
    edge([(120, 60), (160, 60)])
    edge([(280, 60), (350, 60)], r"$\mathbf{u}^{\mathrm{nom}}$", (315, 60))
    edge([(425, 90), (425, 180)], r"$\mathbf{u}^{\mathrm{safe}}$", (425, 135))
    edge([(365, 210), (60, 210), (60, 90)], "state feedback", (212, 210))
    panel(bx, "(b)", x=0.0, y=0.98)
    # Set the maths font before the save, because it is resolved at draw time.
    with plt.rc_context({"mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                         "mathtext.bf": "Arial:bold"}):
        save(fig, "fig1_system", width=DBL)


def fig2_training():
    cur = os.path.join(SP, "curve.npz")
    if not os.path.exists(cur):
        print("  [skip] fig2"); return
    d = np.load(cur)
    fr = d["frames"] / 1e6; su = d["succ"]
    # Top: success. Bottom: curriculum gate.
    fig, (ax, axg) = plt.subplots(2, 1, figsize=(COL, 3.1), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2.1, 1], hspace=0.08))
    k = 5
    ax.plot(fr, su, color=C["lgrey"], lw=0.6, label="per evaluation")
    ax.plot(fr[k - 1:], smooth(su, k), color=PRIMARY, lw=1.1, label=f"{k}-evaluation\nmoving average")
    ax.set_ylabel("all agents docked [%]")
    ax.set_ylim(-4, 104)
    # X limit from the data, so a longer run is not clipped.
    ax.set_xlim(0, fr[-1] * 1.01)
    legend(ax, loc="lower right")
    if d["gate_f"].size:
        gf = np.append(d["gate_f"] / 1e6, fr[-1]); gv = np.append(d["gate_v"], d["gate_v"][-1]) * 100
        axg.step(gf, gv, where="post", color=INK, lw=0.9)
        axg.set_ylim(0, 18); axg.set_yticks([0, 5, 10, 15])
    axg.set_ylabel("capture\ntolerance [cm]")
    axg.set_xlabel("environment frames [millions]")
    fig.align_ylabels([ax, axg])
    save(fig, "fig2_training", width=COL)


def fig3_trajectories(d):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    traj = d["traj"]; goals = d["goals"]; tc = d["target_center"]; ts = d["target_size"]
    fig = plt.figure(figsize=(COL, 2.3))
    ax = fig.add_axes([-0.05, -0.2, 1.0, 1.4], projection="3d")
    ax.computed_zorder = False
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((1, 1, 1, 0)); a.pane.set_edgecolor("0.55"); a.pane.set_linewidth(0.4)
        a._axinfo["grid"].update(color="0.88", linewidth=0.35)
        a.line.set_color(INK); a.line.set_linewidth(0.5)
        a._axinfo["tick"]["inward_factor"] = 0.0; a._axinfo["tick"]["outward_factor"] = 0.25
        a.set_minor_locator(matplotlib.ticker.NullLocator())
    n_ep = min(5, traj.shape[1]); N = traj.shape[2]
    for e in range(n_ep):
        for i in range(N):
            p = traj[:, e, i, :]; good = ~np.isnan(p[:, 0])
            ax.plot(p[good, 0], p[good, 1], p[good, 2], lw=0.6,
                    color=AGENT_COLS[i % len(AGENT_COLS)], alpha=0.9, zorder=2)
    cx, cy, cz = tc; hx, hy, hz = ts / 2
    X = [cx - hx, cx + hx]; Y = [cy - hy, cy + hy]; Z = [cz - hz, cz + hz]
    faces = []
    for xi in X:
        faces.append([(xi, Y[0], Z[0]), (xi, Y[1], Z[0]), (xi, Y[1], Z[1]), (xi, Y[0], Z[1])])
    for yi in Y:
        faces.append([(X[0], yi, Z[0]), (X[1], yi, Z[0]), (X[1], yi, Z[1]), (X[0], yi, Z[1])])
    for zi in Z:
        faces.append([(X[0], Y[0], zi), (X[1], Y[0], zi), (X[1], Y[1], zi), (X[0], Y[1], zi)])
    ax.add_collection3d(Poly3DCollection(faces, facecolor=(0.55, 0.55, 0.55, 0.16),
                                         edgecolor=INK, linewidths=0.5, zorder=1))
    g = goals[0]
    ax.scatter(g[:, 0], g[:, 1], g[:, 2], s=9, facecolor="white", edgecolor=INK,
               linewidths=0.6, depthshade=False, zorder=5)
    ax.set_xlabel(r"$x$ [m]", labelpad=-2); ax.set_ylabel(r"$y$ [m]", labelpad=-2)
    ax.set_zlabel("")                        # auto z-label is clipped by the tight bbox;
    fig.text(0.985, 0.56, r"$z$ [m]", rotation=90, ha="center", va="center", fontsize=FONT)
    ax.tick_params(pad=-1.5, labelsize=FONT)
    ax.set_xticks([0, 4, 8, 12]); ax.set_yticks([-2, 0, 2]); ax.set_zticks([-2, 0, 2])
    ax.set_box_aspect((2.0, 1.2, 1.0)); ax.view_init(elev=22, azim=-70)
    handles = [Line2D([], [], marker="o", ls="none", ms=3.6, mfc="white", mec=INK, mew=0.6,
                      label="docking port"),
               Patch(facecolor=(0.55, 0.55, 0.55, 0.16), edgecolor=INK, lw=0.5, label="target"),
               Line2D([], [], color=AGENT_COLS[0], lw=0.8, label="chaser (colour per agent)")]
    leg = legend(ax, handles=handles, loc="upper left", bbox_to_anchor=(0.04, 0.86), fontsize=FONT)
    # Crop to the drawn content. 'tight' keeps the empty 3-D axes box.
    from matplotlib.transforms import Bbox
    ztxt = fig.texts[-1]

    def crop(f):
        r = f.canvas.get_renderer()
        arts = [leg, ztxt] + list(ax.lines) + list(ax.collections)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            arts += [a.label, a.line] + [t for t in a.get_ticklabels() if t.get_text()]
        # Ignore hidden tick labels outside the canvas. Keep the bottom labels.
        W, H = f.bbox.width, f.bbox.height
        frame = Bbox.from_extents(-0.3 * W, -0.3 * H, 1.3 * W, 1.3 * H)
        ext = [a.get_window_extent(r) for a in arts]
        bb = Bbox.union([e for e in ext if np.all(np.isfinite(e.extents)) and e.width > 0
                         and e.height > 0 and frame.contains(*e.min) and frame.contains(*e.max)])
        return Bbox.from_extents(*(bb.transformed(f.dpi_scale_trans.inverted()).extents
                                   + np.array([-0.04, -0.04, 0.04, 0.04])))
    # 3-D: scale both dimensions to keep the box proportions.
    save(fig, "fig3_trajectories", bbox=crop, width=COL, scale_height=True)


def fig5_separation(_unused=None):
    arrs = mc_arrays()
    if not arrs:
        print("  [skip] fig5 (no results/main/seed*.npz)"); return

    per_seed = [a["min_separation"].reshape(-1) for a in arrs]
    per_seed = [v[np.isfinite(v) & (v < 9)] for v in per_seed]
    ms = np.concatenate(per_seed)
    Ds = float(arrs[0]["agent_keepout"])
    if ms.size == 0:
        print("  [skip] fig5"); return

    lo = float(ms.min())
    n_below = int((ms < Ds).sum())
    # Contact is the bounding-sphere distance, not D_s. D_s is the filter buffer.
    r_contact = float(np.linalg.norm(np.load(os.path.join(SP, "figdata.npz"))["agent_size"])) \
        if os.path.exists(os.path.join(SP, "figdata.npz")) else float("nan")
    fig, axes = plt.subplots(2, 1, figsize=(COL, 4.5))
    fig.subplots_adjust(hspace=0.42)

    # (a) Pooled distribution of the per-episode minimum separation.
    ax = axes[0]
    hmax = np.histogram(ms, bins=60)[0].max()
    sharp_hist(ax, ms, bins=60)
    ax.axvline(Ds, color=ACCENT, lw=0.9, ls=(0, (4, 2)))
    ax.annotate(f"keep-out $D_s$ = {Ds:.2f} m", (Ds, 0.955), xycoords=("data", "axes fraction"),
                xytext=(3, 0), textcoords="offset points", ha="left", va="top",
                fontsize=FONT, color=ACCENT)
    if np.isfinite(r_contact):
        ax.axvline(r_contact, color=INK, lw=0.9, ls=(0, (5, 1.5, 1, 1.5)))
        vline_label(ax, r_contact, f"contact {r_contact:.3f} m", side="right")
    ax.text(0.975, 0.84,
            f"min. observed {lo:.3f} m\n{ms.size:,} episodes\n"
            f"{n_below} inside $D_s$\n0 at contact",
            transform=ax.transAxes, ha="right", va="top", fontsize=FONT, linespacing=1.2)
    ax.set_xlabel("minimum inter-agent separation per episode [m]")
    ax.set_ylabel("episodes")
    ax.set_ylim(0, hmax * 1.62)
    # Show the contact line as well as D_s.
    ax.set_xlim(left=(r_contact - 0.025 if np.isfinite(r_contact)
                      else min(Ds * 0.97, lo * 0.99)))
    panel(ax, "(a)")

    # (b) Lower tail of each seed against the boundary.
    ax = axes[1]
    for i, v in enumerate(per_seed):
        w = np.sort(v)
        k = max(1, int(round(0.05 * w.size)))
        ax.plot(np.arange(1, k + 1) / w.size * 100, w[:k],
                color=SEED_COLS[i % len(SEED_COLS)], lw=0.9, label=f"s{i}")
    ax.axhline(Ds, color=ACCENT, lw=0.9, ls=(0, (4, 2)))
    ax.annotate(r"keep-out $D_s$", (0.02, Ds), xycoords=("axes fraction", "data"),
                xytext=(0, 2), textcoords="offset points", ha="left", va="bottom",
                fontsize=FONT, color=ACCENT)
    ax.set_xlabel("episode percentile (lowest 5%)")
    ax.set_ylabel("minimum separation [m]")
    ax.set_xlim(0, 5.1)
    ax.set_ylim(lo - 0.004, None)
    legend(ax, loc="lower right", ncol=3, handlelength=1.5, columnspacing=0.8)
    panel(ax, "(b)")

    save(fig, "fig5_separation", width=COL)
    print(f"    pooled over {len(arrs)} seeds: {ms.size:,} episodes, "
          f"min {lo:.4f} m vs D_s {Ds:.3f} m, "
          f"{n_below} inside D_s, 1st pct {np.percentile(ms, 1):.3f} m")


def fig6_intervention(_unused=None):
    """Filter intervention against range, pooled over the five reported
    evaluations (results/figure_data/intervention_seed*.npz).
    """
    ps = [os.path.join(RESULTS, "figure_data", f"intervention_seed{i}.npz") for i in range(5)]
    ps = [p for p in ps if os.path.exists(p)]
    if len(ps) < 5:
        print(f"  [skip] fig6 (only {len(ps)}/5 results/figure_data/intervention_seed*.npz)"); return
    zs = [np.load(p) for p in ps]
    iv = np.concatenate([z["iv_flat"] for z in zs])
    dist = np.concatenate([z["range_flat"] for z in zs])
    m = np.isfinite(iv) & np.isfinite(dist) & (dist < 12); iv, dist = iv[m], dist[m]
    bins = np.linspace(0, 12, 49); idx = np.digitize(dist, bins) - 1
    med = np.full(48, np.nan); p90 = np.full(48, np.nan); frac = np.full(48, np.nan)
    for b in range(48):
        s = iv[idx == b]
        if s.size > 30:
            med[b] = np.median(s); p90[b] = np.percentile(s, 90); frac[b] = np.mean(s > 1e-6) * 100
    ctr = 0.5 * (bins[1:] + bins[:-1])
    # Magnitude and frequency on stacked axes with a shared range axis.
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(COL, 3.4), sharex=True,
                                  gridspec_kw=dict(height_ratios=[1.6, 1], hspace=0.08))
    ax.fill_between(ctr, med, p90, color=FILL, lw=0)
    ax.plot(ctr, p90, color=PRIMARY, lw=0.8, ls=(0, (4, 2)), label="90th percentile")
    ax.plot(ctr, med, color=PRIMARY, lw=1.1, label="median")
    ax.set_ylabel(r"$\|\mathbf{u}^{\mathrm{safe}}-\mathbf{u}^{\mathrm{nom}}\|$ [norm.]")
    ax.set_ylim(0, 1.6); ax.set_yticks([0, 0.5, 1.0, 1.5])
    legend(ax, loc="upper left")
    ax2.plot(ctr, frac, color=INK, lw=1.0)
    ax2.set_ylim(0, 108); ax2.set_yticks([0, 50, 100])
    ax2.set_ylabel("steps with\nintervention [%]")
    ax2.set_xlabel("range to own docking port [m]")
    ax2.set_xlim(0, 12)
    fig.align_ylabels([ax, ax2])
    save(fig, "fig6_intervention", width=COL)


def mc_arrays():
    """Per-agent-episode arrays from the five reported evaluations."""
    out = []
    for i in range(5):
        p = os.path.join(RESULTS, "main", f"seed{i}.npz")
        if os.path.exists(p):
            out.append(np.load(p))
    return out


def fig4_timetodock(_unused=None):
    arrs = mc_arrays()
    if not arrs:
        print("  [skip] fig4 (no results/main/seed*.npz)"); return

    per_seed = [a["dock_step"].reshape(-1) for a in arrs]
    allsteps = np.concatenate(per_seed)
    ds = allsteps[np.isfinite(allsteps)]
    n_total, n_cap = allsteps.size, ds.size
    if n_cap == 0:
        print("  [skip] fig4"); return

    med = float(np.median(ds)); p95 = float(np.percentile(ds, 95)); mx = float(ds.max())

    fig, axes = plt.subplots(2, 1, figsize=(COL, 4.3), sharex=True)
    fig.subplots_adjust(hspace=0.12)

    # (a) Pooled distribution.
    ax = axes[0]
    hmax = np.histogram(ds, bins=60)[0].max()
    sharp_hist(ax, ds, bins=60)
    ax.axvline(med, color=INK, ls=(0, (4, 2)), lw=0.8)
    ax.axvline(p95, color=ACCENT, ls=(0, (1, 1.5)), lw=0.9)
    ax.set_ylim(0, hmax * 1.42)
    ax.annotate(f"median\n{med:.0f} s", (med, 0.955), xycoords=("data", "axes fraction"),
                xytext=(-3, 0), textcoords="offset points", ha="right", va="top", fontsize=FONT)
    ax.annotate(f"95th percentile {p95:.0f} s", (p95, 0.955), xycoords=("data", "axes fraction"),
                xytext=(3, 0), textcoords="offset points", ha="left", va="top", fontsize=FONT,
                color=ACCENT)
    ax.text(0.975, 0.78, f"max. {mx:.0f} s\n{n_cap:,} of {n_total:,}\nagents captured",
            transform=ax.transAxes, ha="right", va="top", fontsize=FONT, linespacing=1.2)
    ax.set_ylabel("agent-episodes")
    x0, x1 = float(ds.min()) - 5.0, mx + 5.0
    ax.set_xlim(x0, x1)
    panel(ax, "(a)")

    # (b) ECDF of each seed.
    ax = axes[1]
    ax.axhline(50, color=C["lgrey"], lw=0.5, ls=(0, (1, 1.5)))
    for i, v in enumerate(per_seed):
        # Capture times are whole steps, so plot the ECDF at the distinct times.
        w = v[np.isfinite(v)]
        t, cnt = np.unique(w, return_counts=True)
        ax.plot(t, np.cumsum(cnt) / w.size * 100,
                color=SEED_COLS[i % len(SEED_COLS)], lw=0.9, label=f"s{i}")
    ax.set_xlabel("time to capture [s]"); ax.set_ylabel("agents captured [%]")
    ax.set_xlim(x0, x1); ax.set_ylim(0, 102)
    legend(ax, loc="lower right", ncol=2, handlelength=1.5)
    panel(ax, "(b)")

    save(fig, "fig4_timetodock", width=COL)
    print(f"    pooled over {len(arrs)} seeds: {n_cap:,}/{n_total:,} agent-episodes "
          f"captured, median {med:.0f} s, 95th {p95:.0f} s, max {mx:.0f} s")


def fig7_seeds():
    """Per-seed outcome with the across-seed mean."""
    rs = seed_results()
    if not rs:
        print("  skip fig7_seeds (no seed results)"); return
    import numpy as _np
    comp = [100 * r["complete_success_rate"] for r in rs]
    ang = [r["mean_angular_delta_v_per_agent"] for r in rs]
    x = _np.arange(len(rs))

    fig, axes = plt.subplots(2, 1, figsize=(COL, 4.1), sharex=True)
    fig.subplots_adjust(hspace=0.18)

    def label_bars(ax, vals, fmt):
        for xi, v in zip(x, vals):
            ax.annotate(fmt.format(v), (xi, v), xytext=(0, 2.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=FONT,
                        bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"))

    def label_mean(ax, vals, text):
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color=C["grey"], ls=(0, (4, 2)), lw=0.8)],
                  labels=[text], loc="lower right", bbox_to_anchor=(1.0, 1.0),
                  frameon=False, borderaxespad=0.0, borderpad=0.0,
                  handlelength=2.2, handletextpad=0.5, labelcolor=C["grey"])

    def frame(ax):
        ax.set_xlim(-0.6, len(rs) - 0.4)
        ax.set_xticks(x); ax.set_xticklabels([f"s{i}" for i in range(len(rs))])
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)

    ax = axes[0]
    ax.bar(x, comp, 0.56, color=FILL, edgecolor=PRIMARY, lw=0.7)
    m = float(_np.mean(comp))
    ax.axhline(m, color=C["grey"], ls=(0, (4, 2)), lw=0.8)
    frame(ax); ax.set_ylim(97, 100.5)
    ax.set_ylabel("complete success [%]")
    label_mean(ax, comp, f"mean {m:.2f}%")
    label_bars(ax, comp, "{:.1f}")
    panel(ax, "(a)")

    ax = axes[1]
    ax.bar(x, ang, 0.56, color=FILL, edgecolor=PRIMARY, lw=0.7)
    m2 = float(_np.mean(ang))
    ax.axhline(m2, color=C["grey"], ls=(0, (4, 2)), lw=0.8)
    frame(ax)
    pad = max(max(ang) - min(ang), 1e-3)
    ax.set_ylim(min(ang) - 1.2 * pad, max(ang) + 0.6 * pad)
    ax.set_ylabel(r"angular $\Delta v$ [rad/s]"); ax.set_xlabel("training seed")
    label_mean(ax, ang, f"mean {m2:.3f}")
    label_bars(ax, ang, "{:.3f}")
    panel(ax, "(b)")
    fig.align_ylabels(axes)
    save(fig, "fig7_seeds", width=COL)


if __name__ == "__main__":
    npz = os.path.join(SP, "figdata.npz")
    print("generating figures ->", OUTDIR)
    fig1_system(); fig2_training()
    if os.path.exists(npz):
        fig3_trajectories(np.load(npz))
    else:
        print(f"  [skip] fig3 (no {npz})")
    fig4_timetodock(); fig5_separation(); fig6_intervention()
    fig7_seeds()
    print("done")
