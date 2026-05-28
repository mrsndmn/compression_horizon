"""Animate the progressive-cramming optimization trajectory over the cached
static accuracy regions (talk/demo MP4 + looping GIF).

Phase 1: cache-only. Reads the same ``landscape_pca_pairs.npz`` that
``visual_abstract.py`` consumes (produced by ``visualize_landscale_2pca.py``) and
reveals the trajectory's PC1-PC2 points one-by-one (moving cursor + growing trail)
on top of the fixed >threshold accuracy regions. No model forward passes.
"""

import argparse
import os
import subprocess
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

# Reuse the polished region-rendering helpers from the static-figure script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visual_abstract import (  # noqa: E402
    _boost_saturation,
    _ensure_2d,
    _estimate_cell_area,
    _find_pair_index,
    _load_npz,
)

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

DEFAULT_NPZ = "artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/" "visualizations/landscape_pca_pairs.npz"


def _make_colors(n: int) -> np.ndarray:
    n = max(int(n), 1)
    if sns is not None:
        cols = np.array(sns.color_palette("rocket_r", n_colors=n))
        if cols.shape[1] == 3:
            cols = np.concatenate([cols, np.ones((cols.shape[0], 1), dtype=cols.dtype)], axis=1)
        return cols
    return plt.get_cmap("viridis")(np.linspace(0.05, 0.95, num=n))


def _region_rgba(acc_map: np.ndarray, gx: np.ndarray, gy: np.ndarray, center_xy: Tuple[float, float], color, thr: float):
    """Replicate visual_abstract._draw_panel region shading for one anchor.

    Returns (rgba_img [H,W,4], extent [x0,x1,y0,y1], near_perfect_area) or None if empty.
    """
    max_region_alpha = 0.7
    mask = acc_map > thr
    cell_area = _estimate_cell_area(gx, gy)
    near_perfect_area = float(mask.sum()) * cell_area if cell_area > 0 else float(mask.sum())
    if not np.any(mask):
        return None, None, near_perfect_area

    alpha_grid = (max_region_alpha * mask.astype(np.float32)).astype(np.float32)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    dist = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2).astype(np.float32)
    dist_in = dist[mask]
    d_min = float(np.min(dist_in)) if dist_in.size else 0.0
    d_max = float(np.max(dist_in)) if dist_in.size else 0.0
    if d_max > d_min:
        whiten = (dist - d_min) / (d_max - d_min + 1e-12)
    else:
        whiten = np.zeros_like(dist, dtype=np.float32)
    whiten = np.clip(whiten, 0.0, 1.0) * mask.astype(np.float32) * 0.75

    anchor_rgb = _boost_saturation(np.array(color[:3], dtype=np.float32), factor=1.35)
    rgb = anchor_rgb[None, None, :] * (1.0 - whiten[..., None]) + 1.0 * whiten[..., None]
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    rgba = np.zeros((acc_map.shape[0], acc_map.shape[1], 4), dtype=np.float32)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = alpha_grid
    extent = [float(gx.min()), float(gx.max()), float(gy.min()), float(gy.max())]
    return rgba, extent, near_perfect_area


def _size_from_area(area: float, a_min: float, a_max: float) -> float:
    if not np.isfinite(area) or area <= 0 or a_max <= a_min:
        return 140.0
    t = float(np.clip((area - a_min) / (a_max - a_min + 1e-12), 0.0, 1.0))
    return 120.0 + 320.0 * (t**0.5)


def _fit_to_aspect(rect: Tuple[float, float, float, float], aspect: float) -> Tuple[float, float, float, float]:
    """Expand a (x0,x1,y0,y1) rect about its center until dx/dy == aspect.

    Keeping every camera rect at one aspect ratio lets us interpolate limits
    frame-to-frame without the axes box (and thus the trajectory's proportions)
    changing shape during a smooth zoom.
    """
    x0, x1, y0, y1 = rect
    dx = max(x1 - x0, 1e-9)
    dy = max(y1 - y0, 1e-9)
    cur = dx / dy
    if cur < aspect:  # too narrow -> widen x
        new_dx = aspect * dy
        cx = 0.5 * (x0 + x1)
        x0, x1 = cx - 0.5 * new_dx, cx + 0.5 * new_dx
    elif cur > aspect:  # too wide -> heighten y
        new_dy = dx / aspect
        cy = 0.5 * (y0 + y1)
        y0, y1 = cy - 0.5 * new_dy, cy + 0.5 * new_dy
    return (x0, x1, y0, y1)


def _smoothstep(z: float) -> float:
    z = float(np.clip(z, 0.0, 1.0))
    return z * z * (3.0 - 2.0 * z)


def _lerp_rect(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float], t: float):
    return tuple(float(ai + (bi - ai) * t) for ai, bi in zip(a, b))


def _load_steps_per_point(npz: Dict, n_expected: int):
    """Per-stage optimizer ``steps_taken`` aligned with the trajectory ``coords`` order.

    Reads only light columns from the cached dataset (avoids loading the heavy
    ``embedding`` column, which OOMs). Auto-detects whether ``steps_taken`` is
    cumulative or per-stage and returns per-stage steps. Returns None on any failure.
    """
    try:
        from datasets import Dataset
    except Exception:
        return None
    dpath = str(npz["dataset_path"][0]) if "dataset_path" in npz else None
    sid = int(npz["sample_id"][0]) if "sample_id" in npz else 0
    if not dpath or not os.path.isdir(dpath):
        print(f"[pace=steps] dataset dir not found ({dpath!r}); falling back to uniform pacing.")
        return None
    try:
        ds = Dataset.load_from_disk(dpath)
        cols = [c for c in ["sample_id", "stage_index", "steps_taken"] if c in ds.column_names]
        if "steps_taken" not in cols:
            print("[pace=steps] no 'steps_taken' column; uniform pacing.")
            return None
        df = ds.select_columns(cols).to_pandas()
    except Exception as e:  # pragma: no cover
        print(f"[pace=steps] failed to read steps ({e}); uniform pacing.")
        return None

    df = df[df["sample_id"] == sid].sort_values("stage_index")
    steps = df["steps_taken"].to_numpy().astype(np.float64)
    if steps.size == 0:
        return None
    if steps.shape[0] != n_expected:
        if steps.shape[0] > n_expected:
            steps = steps[:n_expected]
        else:
            steps = np.pad(steps, (0, n_expected - steps.shape[0]), mode="edge")

    # Auto-detect cumulative vs per-stage (steps_taken meaning varies by run).
    diffs = np.diff(steps)
    nondecreasing = float(np.mean(diffs >= -1e-9)) if diffs.size else 1.0
    med = float(np.median(np.clip(steps, 1.0, None)))
    if steps.shape[0] > 2 and nondecreasing > 0.97 and steps[-1] > 3.0 * med:
        steps = np.diff(steps, prepend=0.0)  # cumulative -> per-stage
        kind = "cumulative->per-stage"
    else:
        kind = "per-stage"
    steps = np.clip(steps, 1.0, None)
    print(f"[pace=steps] {kind}: min={steps.min():.0f} max={steps.max():.0f} sum={steps.sum():.0f}")
    return steps


def _write_gif_via_ffmpeg(mp4_path: str, gif_path: str, fps: int, width: int) -> bool:
    """High-quality looping GIF from the MP4 using ffmpeg palettegen. Returns success."""
    if not mp4_path or not os.path.exists(mp4_path):
        return False
    palette = gif_path + ".palette.png"
    vf = f"fps={int(fps)},scale={int(width)}:-1:flags=lanczos"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", mp4_path, "-vf", f"{vf},palettegen", palette],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                mp4_path,
                "-i",
                palette,
                "-lavfi",
                f"{vf}[x];[x][1:v]paletteuse",
                "-loop",
                "0",
                gif_path,
            ],
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    finally:
        if os.path.exists(palette):
            os.remove(palette)


def main() -> None:
    ap = argparse.ArgumentParser(description="Animate progressive-cramming trajectory over cached accuracy regions.")
    ap.add_argument("--npz_path", type=str, default=DEFAULT_NPZ, help="Path to landscape_pca_pairs.npz")
    ap.add_argument("--output", type=str, default=None, help="Output basename (no ext). Default: alongside npz.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=12.0, help="Duration of the reveal (before end-hold).")
    ap.add_argument(
        "--hold-end",
        "--hold_end",
        dest="hold_end",
        type=float,
        default=1.5,
        help="Freeze the final frame for this many seconds.",
    )
    ap.add_argument("--threshold", type=float, default=0.9, help='Accuracy threshold for "near-ideal" regions.')
    ap.add_argument(
        "--regions-upfront",
        "--regions_upfront",
        dest="regions_upfront",
        action="store_true",
        help="Show all accuracy regions from frame 0 (default: each fades in as the cursor reaches its anchor).",
    )
    ap.add_argument("--trail-alpha", "--trail_alpha", dest="trail_alpha", type=float, default=0.45)
    ap.add_argument(
        "--opacity",
        choices=["steps", "uniform"],
        default="steps",
        help="'steps' (default): each trail point's opacity scales (log) with the optimizer steps that "
        "token took to converge -- hard tokens render darker, surfacing the dwell-and-leap basin edges. "
        "'uniform': every point uses --trail-alpha.",
    )
    ap.add_argument(
        "--trail-alpha-min",
        "--trail_alpha_min",
        dest="trail_alpha_min",
        type=float,
        default=0.12,
        help="Trail-point alpha for the fewest-step token when --opacity steps.",
    )
    ap.add_argument(
        "--trail-alpha-max",
        "--trail_alpha_max",
        dest="trail_alpha_max",
        type=float,
        default=0.9,
        help="Trail-point alpha for the most-step token when --opacity steps.",
    )
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--gif-width", "--gif_width", dest="gif_width", type=int, default=900)
    ap.add_argument(
        "--pace",
        choices=["steps", "uniform"],
        default="steps",
        help="'steps' (default): each token's dwell time is proportional to the optimizer "
        "steps it took to converge (read from the cached dataset). 'uniform': equal time per token.",
    )
    ap.add_argument(
        "--zoom-start",
        "--zoom_start",
        dest="zoom_start",
        type=int,
        default=0,
        help="If >0, smoothly zoom the camera into the END of the trajectory (tokens [zoom_start:] and their "
        "anchors/regions) as the cursor reaches token zoom_start. 0 = no zoom (full view throughout).",
    )
    ap.add_argument(
        "--zoom-seconds",
        "--zoom_seconds",
        dest="zoom_seconds",
        type=float,
        default=1.5,
        help="Duration (s) of the smooth camera zoom transition.",
    )
    ap.add_argument("--no-mp4", dest="no_mp4", action="store_true", help="Skip MP4 output.")
    ap.add_argument("--no-gif", dest="no_gif", action="store_true", help="Skip GIF output.")
    args = ap.parse_args()

    npz = _load_npz(args.npz_path)
    required = [
        "pair_indices",
        "grid_x",
        "grid_y",
        "accuracy",
        "coords",
        "explained_variance_ratio",
        "sampled_indices",
        "anchor_coords",
    ]
    missing = [k for k in required if k not in npz]
    if missing:
        raise ValueError(f"Missing keys in npz: {missing}. Available: {sorted(npz.keys())}")

    pair_indices = npz["pair_indices"].astype(np.int64)
    pair_idx = _find_pair_index(pair_indices, (0, 1))

    coords = npz["coords"].astype(np.float32)
    coords_xy = coords[:, :2]
    n_traj = int(coords_xy.shape[0])

    zoom_start = int(args.zoom_start)
    if zoom_start < 0 or zoom_start >= n_traj:
        raise ValueError(f"--zoom-start out of range: {zoom_start} (valid: 0..{n_traj - 1})")
    zoom_enabled = zoom_start > 0

    explained = npz["explained_variance_ratio"].astype(np.float64)
    ev_cum_2 = float(explained[0] + explained[1])

    sampled_indices = npz["sampled_indices"].astype(np.int64).reshape(-1)
    anchor_xy_all = _ensure_2d(npz["anchor_coords"].astype(np.float32))[:, :2]
    stage_seq_len = npz["stage_seq_len"].astype(np.int64).reshape(-1) if "stage_seq_len" in npz else None

    acc = npz["accuracy"]  # [F,P,H,W]
    grid_x_all = npz["grid_x"]
    grid_y_all = npz["grid_y"]
    if acc.ndim != 4:
        raise ValueError(f"Expected accuracy [F,P,H,W], got {acc.shape}")
    acc_per_anchor = acc[:, pair_idx]
    n_anchors = int(acc_per_anchor.shape[0])

    thr = float(args.threshold)
    colors = _make_colors(n_anchors)

    # Precompute region images + areas (static; only their visibility animates).
    regions: List[Dict] = []
    areas = np.zeros(n_anchors, dtype=np.float64)
    for a in range(n_anchors):
        gx = grid_x_all[a, pair_idx]
        gy = grid_y_all[a, pair_idx]
        rgba, extent, area = _region_rgba(acc_per_anchor[a], gx, gy, anchor_xy_all[a], colors[a], thr)
        areas[a] = area
        # Bounds of the >threshold cells only (the grid can be much larger than the
        # near-ideal region, e.g. radius-300 first anchor); used to frame the camera.
        mask = acc_per_anchor[a] > thr
        if np.any(mask):
            mask_bounds = (float(gx[mask].min()), float(gx[mask].max()), float(gy[mask].min()), float(gy[mask].max()))
        else:
            mask_bounds = None
        regions.append({"rgba": rgba, "extent": extent, "reach": int(sampled_indices[a]), "mask_bounds": mask_bounds})
    a_min, a_max = float(np.nanmin(areas)), float(np.nanmax(areas))

    # Static camera limits: full trajectory + anchors + the near-ideal (masked) regions
    # only -- NOT the full accuracy grid, which is far larger than the >thr area.
    x_mins = [float(coords_xy[:, 0].min())]
    x_maxs = [float(coords_xy[:, 0].max())]
    y_mins = [float(coords_xy[:, 1].min())]
    y_maxs = [float(coords_xy[:, 1].max())]
    x_mins.append(float(anchor_xy_all[:, 0].min()))
    x_maxs.append(float(anchor_xy_all[:, 0].max()))
    y_mins.append(float(anchor_xy_all[:, 1].min()))
    y_maxs.append(float(anchor_xy_all[:, 1].max()))
    for r in regions:
        mb = r["mask_bounds"]
        if mb is not None:
            x_mins.append(mb[0])
            x_maxs.append(mb[1])
            y_mins.append(mb[2])
            y_maxs.append(mb[3])
    x_min, x_max = min(x_mins), max(x_maxs)
    y_min, y_max = min(y_mins), max(y_maxs)
    dx = max(x_max - x_min, 1e-6)
    dy = max(y_max - y_min, 1e-6)
    pad_x, pad_bot, pad_top = 0.06, 0.07, 0.20  # extra headroom so top region/label clears the title

    full_rect = (x_min - pad_x * dx, x_max + pad_x * dx, y_min - pad_bot * dy, y_max + pad_top * dy)
    cam_aspect = (full_rect[1] - full_rect[0]) / max(full_rect[3] - full_rect[2], 1e-9)

    # Zoom target: the end of the trajectory (tokens [zoom_start:]) plus the anchors/regions
    # at index >= zoom_start. Fit to the full view's aspect so the camera move doesn't distort.
    zoom_rect = full_rect
    if zoom_enabled:
        zx0 = [float(coords_xy[zoom_start:, 0].min())]
        zx1 = [float(coords_xy[zoom_start:, 0].max())]
        zy0 = [float(coords_xy[zoom_start:, 1].min())]
        zy1 = [float(coords_xy[zoom_start:, 1].max())]
        for a, r in enumerate(regions):
            if int(sampled_indices[a]) < zoom_start:
                continue
            zx0.append(float(anchor_xy_all[a, 0]))
            zx1.append(float(anchor_xy_all[a, 0]))
            zy0.append(float(anchor_xy_all[a, 1]))
            zy1.append(float(anchor_xy_all[a, 1]))
            mb = r["mask_bounds"]
            if mb is not None:
                zx0.append(mb[0])
                zx1.append(mb[1])
                zy0.append(mb[2])
                zy1.append(mb[3])
        zx_min, zx_max = min(zx0), max(zx1)
        zy_min, zy_max = min(zy0), max(zy1)
        zdx = max(zx_max - zx_min, 1e-6)
        zdy = max(zy_max - zy_min, 1e-6)
        zoom_rect = _fit_to_aspect(
            (zx_min - pad_x * zdx, zx_max + pad_x * zdx, zy_min - pad_bot * zdy, zy_max + pad_top * zdy),
            cam_aspect,
        )

    # Styling (talk: large fonts).
    if sns is not None:
        sns.set_theme(style="whitegrid")
    matplotlib.rcParams.update(
        {
            "font.size": 22,
            "axes.titlesize": 26,
            "axes.labelsize": 24,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(12.0, 9.0))
    ax.set_xlim(full_rect[0], full_rect[1])
    ax.set_ylim(full_rect[2], full_rect[3])
    ax.set_xlabel(f"PC1  ({float(explained[0]):.1%})")
    ax.set_ylabel(f"PC2  ({float(explained[1]):.1%})")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.15)

    # Region imshow artists (toggle visibility), anchor markers + labels.
    anchor_centroid = anchor_xy_all.mean(axis=0)
    region_artists = []
    marker_artists = []
    label_artists = []
    for a, r in enumerate(regions):
        if r["rgba"] is None:
            region_artists.append(None)
        else:
            im = ax.imshow(r["rgba"], origin="lower", extent=r["extent"], interpolation="nearest", zorder=1 + 0.001 * a)
            im.set_visible(bool(args.regions_upfront))
            region_artists.append(im)
        size = _size_from_area(areas[a], a_min, a_max)
        mk = ax.scatter(
            [anchor_xy_all[a, 0]], [anchor_xy_all[a, 1]], s=size, c=[colors[a]], edgecolors="black", linewidths=0.9, zorder=5
        )
        mk.set_visible(bool(args.regions_upfront))
        marker_artists.append(mk)
        # Push the label radially outward from the anchor cloud so close anchors
        # (50/100, 800/1000) don't collide; white bbox + thin leader for legibility.
        d = anchor_xy_all[a] - anchor_centroid
        nrm = float(np.linalg.norm(d))
        u = d / nrm if nrm > 1e-6 else np.array([0.0, 1.0])
        off = (float(u[0]) * 46.0, float(u[1]) * 46.0)
        ha = "left" if u[0] > 0.2 else ("right" if u[0] < -0.2 else "center")
        va = "bottom" if u[1] > 0.2 else ("top" if u[1] < -0.2 else "center")
        lab = ax.annotate(
            f"{int(sampled_indices[a])}",
            xy=(float(anchor_xy_all[a, 0]), float(anchor_xy_all[a, 1])),
            xytext=off,
            textcoords="offset points",
            fontsize=18,
            ha=ha,
            va=va,
            color="black",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75),
            arrowprops=dict(arrowstyle="-", color="black", alpha=0.45, lw=0.8),
        )
        lab.set_visible(bool(args.regions_upfront))
        label_artists.append(lab)

    (path_line,) = ax.plot([], [], color="black", alpha=0.30, linewidth=1.6, zorder=1.9)
    trail = ax.scatter([], [], s=16, linewidths=0, zorder=2)
    head = ax.scatter([], [], s=300, marker="*", c="red", edgecolors="black", linewidths=1.0, zorder=8)
    title = ax.set_title("", pad=16)

    # Frame schedule: reveal across `seconds`, then hold.
    anim_frames = max(int(round(args.fps * args.seconds)), 2)
    hold_frames = max(int(round(args.fps * args.hold_end)), 0)
    # Pace the reveal so each token's on-screen dwell is proportional to the optimizer
    # steps it took to converge: walk a cumulative "optimizer-step time" axis at constant
    # real-time speed, so hard-to-cram tokens (many steps) linger and easy ones flash by.
    steps_per_point = _load_steps_per_point(npz, n_traj)
    if args.pace == "steps" and steps_per_point is not None:
        w = np.clip(steps_per_point.astype(np.float64), 1.0, None)
        cum = np.cumsum(w)
        t = (np.arange(1, anim_frames + 1) / float(anim_frames)) * float(cum[-1])
        reveal_counts = np.clip(np.searchsorted(cum, t, side="left") + 1, 1, n_traj).astype(int)
    else:
        reveal_counts = np.clip(np.linspace(1, n_traj, anim_frames).round().astype(int), 1, n_traj)
    reveal_counts = np.concatenate([reveal_counts, np.full(hold_frames, n_traj, dtype=int)])
    total_frames = int(reveal_counts.shape[0])

    # Per-point trail opacity: scale (log) with the optimizer steps each token took to
    # converge, so hard tokens render darker and the dwell-and-leap basin edges stand out.
    if args.opacity == "steps" and steps_per_point is not None:
        la = np.log(np.clip(steps_per_point.astype(np.float64), 1.0, None))
        span = float(la.max() - la.min())
        rng01 = (la - la.min()) / span if span > 1e-12 else np.full(n_traj, 0.5)
        pt_alpha = float(args.trail_alpha_min) + (float(args.trail_alpha_max) - float(args.trail_alpha_min)) * rng01
    else:
        pt_alpha = np.full(n_traj, float(args.trail_alpha))
    trail_rgba = np.zeros((n_traj, 4), dtype=np.float64)
    trail_rgba[:, 3] = np.clip(pt_alpha, 0.0, 1.0)  # black points, per-token alpha

    # Smooth camera-zoom schedule: start easing when the cursor reaches token zoom_start.
    heads = reveal_counts - 1
    if zoom_enabled:
        reached = np.where(heads >= zoom_start)[0]
        f_zoom_start = int(reached[0]) if reached.size else total_frames
        zoom_span = max(int(round(args.zoom_seconds * args.fps)), 1)
    else:
        f_zoom_start = total_frames
        zoom_span = 1

    def init():
        ax.set_xlim(full_rect[0], full_rect[1])
        ax.set_ylim(full_rect[2], full_rect[3])
        path_line.set_data([], [])
        trail.set_offsets(np.empty((0, 2)))
        head.set_offsets(np.empty((0, 2)))
        return [path_line, trail, head, title]

    def update(f):
        k = int(reveal_counts[f])
        h = k - 1
        if zoom_enabled:
            z = _smoothstep((f - f_zoom_start) / float(zoom_span))
            cx0, cx1, cy0, cy1 = _lerp_rect(full_rect, zoom_rect, z)
            ax.set_xlim(cx0, cx1)
            ax.set_ylim(cy0, cy1)
        path_line.set_data(coords_xy[:k, 0], coords_xy[:k, 1])
        trail.set_offsets(coords_xy[:k])
        trail.set_facecolors(trail_rgba[:k])
        head.set_offsets(coords_xy[h : h + 1])
        for a, r in enumerate(regions):
            show = bool(args.regions_upfront) or (h >= r["reach"])
            if region_artists[a] is not None:
                region_artists[a].set_visible(show)
            marker_artists[a].set_visible(show)
            label_artists[a].set_visible(show)
        if stage_seq_len is not None and 0 <= h < stage_seq_len.shape[0]:
            plen = int(stage_seq_len[h])
        else:
            plen = h + 1
        if steps_per_point is not None and 0 <= h < steps_per_point.shape[0]:
            title.set_text(f"Llama-3.1-8B  ·  prefix {plen} tokens  ·  {int(steps_per_point[h])} optimizer steps")
        else:
            title.set_text(f"Llama-3.1-8B  ·  prefix length: {plen} tokens   (PC1+PC2 = {ev_cum_2:.1%})")
        return [path_line, trail, head, title] + list(marker_artists) + list(label_artists)

    anim = FuncAnimation(
        fig, update, init_func=init, frames=total_frames, blit=False, interval=1000.0 / args.fps, cache_frame_data=False
    )

    out_base = args.output
    if out_base is None:
        name = "visual_abstract_trajectory" + (f"_zoom{zoom_start}" if zoom_enabled else "")
        out_base = os.path.join(os.path.dirname(args.npz_path), name)
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)

    mp4_path = out_base + ".mp4"
    gif_path = out_base + ".gif"

    if not args.no_mp4:
        anim.save(mp4_path, writer=FFMpegWriter(fps=args.fps, bitrate=6000), dpi=args.dpi)
        print(f"Saved MP4: {mp4_path}  ({total_frames} frames @ {args.fps}fps)")

    if not args.no_gif:
        ok = (not args.no_mp4) and _write_gif_via_ffmpeg(mp4_path, gif_path, fps=min(args.fps, 25), width=args.gif_width)
        if ok:
            print(f"Saved GIF (ffmpeg palette, looping): {gif_path}")
        else:
            anim.save(gif_path, writer=PillowWriter(fps=min(args.fps, 25)), dpi=max(args.dpi // 2, 80))
            print(f"Saved GIF (PillowWriter, looping): {gif_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
