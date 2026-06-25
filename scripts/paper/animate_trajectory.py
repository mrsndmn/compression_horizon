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
from matplotlib.patches import Rectangle
from matplotlib.patheffects import withStroke

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


def _anchor_grid(grid_all: np.ndarray, a: int, pair_idx: int) -> np.ndarray:
    """Return anchor ``a``'s 2D meshgrid slice for the given PCA pair.

    Supports both cached schemas: per-anchor grids ``[n_anchors, n_pairs, H, W]``
    (each anchor evaluated on its own local neighborhood) and a single shared grid
    ``[n_pairs, H, W]`` (all anchors evaluated on one global PCA-plane grid).
    """
    if grid_all.ndim == 4:
        return grid_all[a, pair_idx]
    if grid_all.ndim == 3:
        return grid_all[pair_idx]
    raise ValueError(f"Unexpected grid ndim={grid_all.ndim}; expected 3 ([P,H,W]) or 4 ([F,P,H,W]).")


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


def _respace_by_steps(coords_xy: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Step-warped trajectory layout: keep each PCA hop's DIRECTION but set its on-screen
    length proportional to the optimizer steps that converged the landing token.

    Hard (many-step) tokens spread far apart; trivial tokens bunch up. This is NOT a PCA
    projection -- the accuracy field (defined on the PCA grid) no longer aligns and should
    not be drawn in this view. The hop scale is chosen so the median hop length matches the
    median PCA jump, keeping the overall extent familiar.
    """
    P = coords_xy.astype(np.float64)
    if P.shape[0] < 2:
        return coords_xy
    delta = np.diff(P, axis=0)  # [N-1, 2] PCA jump vectors
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    unit = np.divide(delta, norm, out=np.zeros_like(delta), where=norm > 1e-12)
    seg_len = np.clip(steps[1:].astype(np.float64), 0.0, None)  # steps for the landing token
    pos_steps = seg_len[seg_len > 0]
    pos_jumps = norm[norm > 1e-12]
    scale = float(np.median(pos_jumps)) / float(np.median(pos_steps)) if pos_steps.size and pos_jumps.size else 1.0
    seg = unit * (seg_len[:, None] * scale)
    out = np.empty_like(P)
    out[0] = P[0]
    out[1:] = P[0] + np.cumsum(seg, axis=0)
    return out.astype(np.float32)


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
        help="Duration (s) of the smooth camera zoom transition. Ignored if --zoom-to is set.",
    )
    ap.add_argument(
        "--zoom-from",
        "--zoom_from",
        dest="zoom_from",
        type=int,
        default=-1,
        help="Cursor token at which the zoom transition BEGINS (default: zoom_start). "
        "Lets the camera start moving before the framed region is reached.",
    )
    ap.add_argument(
        "--zoom-to",
        "--zoom_to",
        dest="zoom_to",
        type=int,
        default=-1,
        help="Cursor token at which the zoom transition COMPLETES. If set, the zoom span runs "
        "from token --zoom-from to token --zoom-to (overrides --zoom-seconds).",
    )
    ap.add_argument(
        "--zoom-out-seconds",
        "--zoom_out_seconds",
        dest="zoom_out_seconds",
        type=float,
        default=0.0,
        help="If >0 (and zoom is enabled), after the end-hold pause the camera eases back out "
        "to the full view over this many seconds.",
    )
    ap.add_argument(
        "--normalize-by-steps",
        "--normalize_by_steps",
        dest="normalize_by_steps",
        action="store_true",
        help="Step-warped view: re-space the trajectory so each hop's length is proportional to "
        "the optimizer steps that converged the landing token (PCA jump DIRECTION preserved). "
        "Drops the PCA accuracy-field overlay (it no longer aligns). Default off (= PCA view).",
    )
    ap.add_argument(
        "--zoom-out-start",
        "--zoom_out_start",
        dest="zoom_out_start_k",
        type=int,
        default=0,
        help="If >0, START the camera zoomed in on tokens [0:K] (the small early hops) and do ONE "
        "smooth eased zoom-OUT to the full view as the cursor advances (timing via --zoom-from/"
        "--zoom-to or --zoom-seconds). Mutually exclusive with --zoom-start. Pairs with "
        "--normalize-by-steps for the step-warped view.",
    )
    ap.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Hide the top progress bar (converged tokens, colored by per-token optimizer-step cost).",
    )
    ap.add_argument("--no-mp4", dest="no_mp4", action="store_true", help="Skip MP4 output.")
    ap.add_argument("--no-gif", dest="no_gif", action="store_true", help="Skip GIF output.")
    ap.add_argument(
        "--fill-canvas",
        "--fill_canvas",
        dest="fill_canvas",
        action="store_true",
        help="Make the axes fill the whole (landscape) frame while KEEPING equal (1:1) PC1/PC2 scaling: "
        "the camera rect is padded out to the frame's aspect so the extra room shows as empty plot area "
        "(no-data space) rather than stretching/distorting the geometry. Default off (= letterboxed equal view).",
    )
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

    # Per-token optimizer steps (used for pacing, opacity, the progress bar, and the
    # optional step-warped layout). Loaded once here so the transform can use it too.
    steps_per_point = _load_steps_per_point(npz, n_traj)
    normalize_by_steps = bool(args.normalize_by_steps)
    if normalize_by_steps:
        if steps_per_point is None:
            raise SystemExit("--normalize-by-steps needs per-token 'steps_taken' but none could be loaded.")
        coords_xy = _respace_by_steps(coords_xy, steps_per_point)

    zoom_start = int(args.zoom_start)
    if zoom_start < 0 or zoom_start >= n_traj:
        raise ValueError(f"--zoom-start out of range: {zoom_start} (valid: 0..{n_traj - 1})")
    zoom_enabled = zoom_start > 0
    # Zoom-OUT-from-start: begin framing tokens [0:K] and ease out to the full view.
    zoom_out_start_k = int(args.zoom_out_start_k)
    if zoom_out_start_k < 0 or zoom_out_start_k >= n_traj:
        raise ValueError(f"--zoom-out-start out of range: {zoom_out_start_k} (valid: 0..{n_traj - 1})")
    zoomout_start_enabled = zoom_out_start_k > 0
    if zoomout_start_enabled and zoom_enabled:
        raise ValueError("--zoom-out-start and --zoom-start are mutually exclusive (zoom OUT from start vs zoom IN to end).")
    # Schedule tokens: the camera moves between zoom_from and (optionally) zoom_to. For zoom-in this
    # is tied to --zoom-start's region; for zoom-out-start it defaults to begin at token 0.
    zoom_from = int(args.zoom_from) if int(args.zoom_from) >= 0 else zoom_start
    zoom_to = int(args.zoom_to)
    if (zoom_enabled or zoomout_start_enabled) and not (0 <= zoom_from < n_traj):
        raise ValueError(f"--zoom-from out of range: {zoom_from} (valid: 0..{n_traj - 1})")
    if zoom_to >= 0 and not (zoom_from < zoom_to < n_traj):
        raise ValueError(f"--zoom-to must satisfy zoom_from({zoom_from}) < zoom_to < {n_traj}; got {zoom_to}")

    explained = npz["explained_variance_ratio"].astype(np.float64)
    ev_cum_2 = float(explained[0] + explained[1])

    sampled_indices = npz["sampled_indices"].astype(np.int64).reshape(-1)
    anchor_xy_all = _ensure_2d(npz["anchor_coords"].astype(np.float32))[:, :2]
    if normalize_by_steps:
        # Anchors are trajectory points; place them at their step-warped positions so the
        # markers sit on the transformed path.
        anchor_xy_all = coords_xy[np.clip(sampled_indices, 0, n_traj - 1)].astype(np.float32)

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
        gx = _anchor_grid(grid_x_all, a, pair_idx)
        gy = _anchor_grid(grid_y_all, a, pair_idx)
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

    if normalize_by_steps:
        # The accuracy field lives on the PCA grid and can't be re-spaced; drop it so it is
        # neither drawn nor used to frame the camera. Anchor markers/labels remain.
        for r in regions:
            r["rgba"] = None
            r["mask_bounds"] = None

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

    # --fill-canvas: keep equal (1:1) scaling but pad the camera rect out to the landscape AXES box
    # aspect so the axes fills the frame, with the extra room showing as empty (no-data) plot area
    # instead of stretching the geometry. The padded aspect (cam_aspect) is then shared by the full
    # and zoom views, so framing stays consistent through the camera move. Margins below are also used
    # for subplots_adjust at figure-build time, so the equal-aspect box fills the position rectangle.
    fill_canvas = bool(args.fill_canvas)
    FILL_FIGSIZE = (12.0, 9.0)
    # left=0.14 keeps the rotated "PC2 (..)" axis title + the wide full-view tick labels ("-6000")
    # on-canvas; at the tight 0.11 they were pushed off the left edge at the start (full view).
    FILL_MARGINS = (0.14, 0.985, 0.085, 0.87 if bool(args.progress) else 0.93)  # (left, right, bottom, top)
    if fill_canvas:
        ml, mr, mb, mt = FILL_MARGINS
        box_aspect = (FILL_FIGSIZE[0] * (mr - ml)) / max(FILL_FIGSIZE[1] * (mt - mb), 1e-9)
        full_rect = _fit_to_aspect(full_rect, box_aspect)
        cam_aspect = box_aspect

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

    # Zoom-OUT-from-start origin: frame the first K tokens (the small early hops) + any anchors in
    # [0:K], fit to the full view's aspect so the eased start_rect -> full_rect move doesn't distort.
    start_rect = full_rect
    if zoomout_start_enabled:
        k0 = zoom_out_start_k
        sx0 = [float(coords_xy[:k0, 0].min())]
        sx1 = [float(coords_xy[:k0, 0].max())]
        sy0 = [float(coords_xy[:k0, 1].min())]
        sy1 = [float(coords_xy[:k0, 1].max())]
        for a, r in enumerate(regions):
            if int(sampled_indices[a]) >= k0:
                continue
            sx0.append(float(anchor_xy_all[a, 0]))
            sx1.append(float(anchor_xy_all[a, 0]))
            sy0.append(float(anchor_xy_all[a, 1]))
            sy1.append(float(anchor_xy_all[a, 1]))
            mb = r["mask_bounds"]
            if mb is not None:
                sx0.append(mb[0])
                sx1.append(mb[1])
                sy0.append(mb[2])
                sy1.append(mb[3])
        sx_min, sx_max = min(sx0), max(sx1)
        sy_min, sy_max = min(sy0), max(sy1)
        sdx = max(sx_max - sx_min, 1e-6)
        sdy = max(sy_max - sy_min, 1e-6)
        start_rect = _fit_to_aspect(
            (sx_min - pad_x * sdx, sx_max + pad_x * sdx, sy_min - pad_bot * sdy, sy_max + pad_top * sdy),
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

    fig, ax = plt.subplots(1, 1, figsize=FILL_FIGSIZE)
    ax.set_xlim(full_rect[0], full_rect[1])
    ax.set_ylim(full_rect[2], full_rect[3])
    if normalize_by_steps:
        ax.set_xlabel("PC1 direction · step-warped")
        ax.set_ylabel("PC2 direction · step-warped")
    else:
        ax.set_xlabel(f"PC1  ({float(explained[0]):.1%})")
        ax.set_ylabel(f"PC2  ({float(explained[1]):.1%})")
    # PCA view keeps true geometry (equal). The step-warped view is tall and not a metric
    # space, so let it fill the (landscape) figure like the PCA videos instead of letterboxing.
    ax.set_aspect("auto" if normalize_by_steps else "equal", adjustable="box")
    if fill_canvas:
        # full_rect was padded to FILL_MARGINS' box aspect above, so the equal-aspect box fills the
        # position rectangle exactly (no letterbox); the padding shows as empty plot area. Realign pbar.
        ml, mr, mb, mt = FILL_MARGINS
        fig.subplots_adjust(left=ml, right=mr, bottom=mb, top=mt)
        pbar_box = (ml, 0.905, mr - ml, 0.030)
    else:
        pbar_box = (0.12, 0.905, 0.76, 0.030)
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
    title = fig.text(0.5, 0.965, "", ha="center", va="center", fontsize=24)

    # Frame schedule: reveal across `seconds`, then hold.
    anim_frames = max(int(round(args.fps * args.seconds)), 2)
    hold_frames = max(int(round(args.fps * args.hold_end)), 0)
    # Pace the reveal so each token's on-screen dwell is proportional to the optimizer
    # steps it took to converge: walk a cumulative "optimizer-step time" axis at constant
    # real-time speed, so hard-to-cram tokens (many steps) linger and easy ones flash by.
    if args.pace == "steps" and steps_per_point is not None:
        w = np.clip(steps_per_point.astype(np.float64), 1.0, None)
        cum = np.cumsum(w)
        t = (np.arange(1, anim_frames + 1) / float(anim_frames)) * float(cum[-1])
        reveal_counts = np.clip(np.searchsorted(cum, t, side="left") + 1, 1, n_traj).astype(int)
    else:
        reveal_counts = np.clip(np.linspace(1, n_traj, anim_frames).round().astype(int), 1, n_traj)
    reveal_counts = np.concatenate([reveal_counts, np.full(hold_frames, n_traj, dtype=int)])
    # End zoom-out: after the reveal + end-hold pause, ease the camera back to the full view
    # (+ a short settle hold). The trajectory stays fully revealed throughout.
    zoom_out_frames = max(int(round(args.fps * args.zoom_out_seconds)), 0) if zoom_enabled else 0
    zoom_out_settle = max(int(round(args.fps * 0.5)), 1) if zoom_out_frames > 0 else 0
    if zoom_out_frames > 0:
        reveal_counts = np.concatenate([reveal_counts, np.full(zoom_out_frames + zoom_out_settle, n_traj, dtype=int)])
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

    # Top progress bar: converged-token count, with the bar itself colored by each token's
    # optimizer-step cost (easy=light, hard=red). Because the reveal is paced in real time by
    # `reveal_counts`, the fill races across the cheap early tokens and crawls through the
    # expensive later ones -- making "fast early, slow late" convergence visible at a glance.
    show_progress = bool(args.progress)
    cover = cursor = pbar_text = None
    if show_progress:
        fig.subplots_adjust(top=0.87)  # title sits at the very top; bar goes just beneath it
        if steps_per_point is not None:
            rib = np.log(np.clip(steps_per_point.astype(np.float64), 1.0, None))
            rmin, rmax = float(rib.min()), float(rib.max())
            rib_norm = (rib - rmin) / (rmax - rmin) if rmax > rmin else np.full(n_traj, 0.5)
        else:
            rib_norm = np.full(n_traj, 0.5)
        pbar_ax = fig.add_axes(pbar_box)
        pbar_ax.set_xlim(0.0, float(n_traj))
        pbar_ax.set_ylim(0.0, 1.0)
        pbar_ax.set_xticks([])
        pbar_ax.set_yticks([])
        pbar_ax.imshow(
            rib_norm[None, :],
            aspect="auto",
            extent=(0.0, float(n_traj), 0.0, 1.0),
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            origin="lower",
            zorder=1,
        )
        cover = Rectangle((0.0, 0.0), float(n_traj), 1.0, facecolor="0.92", edgecolor="none", zorder=2)
        pbar_ax.add_patch(cover)
        pbar_ax.add_patch(Rectangle((0.0, 0.0), float(n_traj), 1.0, fill=False, edgecolor="0.4", lw=1.2, zorder=3))
        cursor = pbar_ax.axvline(0.0, color="black", lw=1.4, zorder=4)
        pbar_text = pbar_ax.text(
            0.5,
            0.5,
            "",
            transform=pbar_ax.transAxes,
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="white",
            zorder=5,
            path_effects=[withStroke(linewidth=2.6, foreground="black")],
        )

    # Smooth camera-zoom schedule: start easing when the cursor reaches token zoom_from, and
    # either run for zoom_seconds or finish exactly when the cursor reaches token zoom_to. Shared by
    # zoom-IN (full -> zoom_rect) and zoom-OUT-from-start (start_rect -> full).
    heads = reveal_counts - 1
    if zoom_enabled or zoomout_start_enabled:
        reached = np.where(heads >= zoom_from)[0]
        f_zoom_start = int(reached[0]) if reached.size else total_frames
        if zoom_to >= 0:
            reached_end = np.where(heads >= zoom_to)[0]
            f_zoom_end = int(reached_end[0]) if reached_end.size else total_frames
            zoom_span = max(f_zoom_end - f_zoom_start, 1)
        else:
            zoom_span = max(int(round(args.zoom_seconds * args.fps)), 1)
    else:
        f_zoom_start = total_frames
        zoom_span = 1
    # Zoom-out window begins right after the reveal + end-hold pause.
    zoom_out_enabled = zoom_out_frames > 0
    f_zoomout_start = anim_frames + hold_frames
    zoom_out_span = max(zoom_out_frames, 1)

    init_rect = start_rect if zoomout_start_enabled else full_rect

    def init():
        ax.set_xlim(init_rect[0], init_rect[1])
        ax.set_ylim(init_rect[2], init_rect[3])
        path_line.set_data([], [])
        trail.set_offsets(np.empty((0, 2)))
        head.set_offsets(np.empty((0, 2)))
        if show_progress:
            cover.set_bounds(0.0, 0.0, float(n_traj), 1.0)
            cursor.set_xdata([0.0, 0.0])
            pbar_text.set_text("")
        return [path_line, trail, head, title]

    def update(f):
        k = int(reveal_counts[f])
        h = k - 1
        if zoomout_start_enabled:
            # One smooth eased zoom-OUT: start_rect (tokens [0:K]) -> full view.
            z = _smoothstep((f - f_zoom_start) / float(zoom_span))
            cx0, cx1, cy0, cy1 = _lerp_rect(start_rect, full_rect, z)
            ax.set_xlim(cx0, cx1)
            ax.set_ylim(cy0, cy1)
        elif zoom_enabled:
            if zoom_out_enabled and f >= f_zoomout_start:
                zo = _smoothstep((f - f_zoomout_start) / float(zoom_out_span))
                cx0, cx1, cy0, cy1 = _lerp_rect(zoom_rect, full_rect, zo)
            else:
                z = _smoothstep((f - f_zoom_start) / float(zoom_span))
                cx0, cx1, cy0, cy1 = _lerp_rect(full_rect, zoom_rect, z)
            ax.set_xlim(cx0, cx1)
            ax.set_ylim(cy0, cy1)
        path_line.set_data(coords_xy[:k, 0], coords_xy[:k, 1])
        trail.set_offsets(coords_xy[:k])
        trail.set_facecolors(trail_rgba[:k])
        head.set_offsets(coords_xy[h : h + 1])
        if show_progress:
            cover.set_bounds(float(k), 0.0, float(max(n_traj - k, 0)), 1.0)
            cursor.set_xdata([float(k), float(k)])
            pbar_text.set_text(f"{k:,} / {n_traj:,} tokens converged  ({k / float(n_traj):.0%})")
        for a, r in enumerate(regions):
            show = bool(args.regions_upfront) or (h >= r["reach"])
            if region_artists[a] is not None:
                region_artists[a].set_visible(show)
            marker_artists[a].set_visible(show)
            label_artists[a].set_visible(show)
        if steps_per_point is not None and 0 <= h < steps_per_point.shape[0]:
            title.set_text(f"Llama-3.1-8B  ·  {int(steps_per_point[h])} optimizer steps")
        else:
            title.set_text(f"Llama-3.1-8B   (PC1+PC2 = {ev_cum_2:.1%})")
        return [path_line, trail, head, title] + list(marker_artists) + list(label_artists)

    anim = FuncAnimation(
        fig, update, init_func=init, frames=total_frames, blit=False, interval=1000.0 / args.fps, cache_frame_data=False
    )

    out_base = args.output
    if out_base is None:
        name = (
            "visual_abstract_trajectory"
            + ("_stepspace" if normalize_by_steps else "")
            + (f"_zoom{zoom_start}" if zoom_enabled else "")
        )
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
