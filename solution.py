#!/usr/bin/env python3
"""Part A - ego-vehicle ground-plane trajectory from a fixed traffic light.

Pipeline (see PLAN.md for the derivation):

    bboxes CSV  ->  bbox centre pixel (u, v)
                ->  xyz[v, u]  (3D point in CAMERA coords, metres)
                ->  car -> light vector per frame
                ->  vehicle heading psi_t  (non-holonomic solve, below)
                ->  ego position p_t = -R(psi_t) @ c_t  in the WORLD frame
                ->  trajectory.png / trajectory.mp4 / diagnostics.png

Two coordinate frames, never to be conflated:

  * camera frame - moves with the car, origin at the camera (top of the car,
    centred on vehicle width).  +X forward.  +Z up.  The sign of +Y is
    *measured from the data* by `detect_axis_convention` rather than assumed:
    the challenge text says "+Y right" but also calls the frame right-handed,
    which is self-contradictory, and this dataset is in fact +Y **left**.
  * world frame - fixed for the whole clip.  Origin on the ground under the
    traffic light, +Z up through the light, and by definition the car->light
    line at the reference frame lies along +X.  Right-handed: X fwd, Y left.

Run:  python solution.py [--dataset dataset]
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

# --------------------------------------------------------------------------
# Palette (validated for CVD separation on a light surface; see PLAN.md §6).
# --------------------------------------------------------------------------
C_PATH = "#2a78d6"    # categorical slot 1 - the ego trajectory
C_START = "#eb6834"   # categorical slot 2 - start marker
C_END = "#1baf7a"     # categorical slot 3 - end / current-position marker
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d6d5d1"
AXIS = "#8a8984"

DEFAULT_FPS = 30.0

# --------------------------------------------------------------------------
# 1. Loading and pre-processing
# --------------------------------------------------------------------------

_COLUMN_ALIASES = {
    "frame": "frame_id", "frame_id": "frame_id", "frameid": "frame_id",
    "x1": "x_min", "x_min": "x_min", "xmin": "x_min",
    "y1": "y_min", "y_min": "y_min", "ymin": "y_min",
    "x2": "x_max", "x_max": "x_max", "xmax": "x_max",
    "y2": "y_max", "y_max": "y_max", "ymax": "y_max",
}

_BBOX_CSV_NAMES = ("bboxes_light.csv", "bbox_light.csv", "bboxes.csv")


def find_bbox_csv(dataset_dir):
    """Locate the traffic-light bbox CSV, tolerating the two names seen so far
    (`bboxes_light.csv` in the spec, `bbox_light.csv` in the shipped data) and
    falling back to any single *.csv in the dataset root."""
    for name in _BBOX_CSV_NAMES:
        path = os.path.join(dataset_dir, name)
        if os.path.isfile(path):
            return path
    loose = sorted(glob.glob(os.path.join(dataset_dir, "*.csv")))
    if len(loose) == 1:
        return loose[0]
    raise FileNotFoundError(
        f"No traffic-light bbox CSV in {dataset_dir!r} "
        f"(looked for {', '.join(_BBOX_CSV_NAMES)})"
    )


def load_bbox_csv(path):
    """Parse the bbox CSV into dicts keyed frame_id/x_min/y_min/x_max/y_max.

    Accepts either known header spelling (`frame,x1,y1,x2,y2` as shipped, or
    `frame_id,x_min,y_min,x_max,y_max` as documented) and tolerates blank
    trailing lines."""
    rows = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")
        header_map = {}
        for col in reader.fieldnames:
            if col is None:
                continue
            key = col.strip().lower().replace(" ", "")
            if key not in _COLUMN_ALIASES:
                raise ValueError(f"Unrecognised bbox CSV column {col!r} in {path}")
            header_map[col] = _COLUMN_ALIASES[key]
        missing = {"frame_id", "x_min", "y_min", "x_max", "y_max"} - set(header_map.values())
        if missing:
            raise ValueError(f"{path} is missing column(s): {sorted(missing)}")

        for raw in reader:
            values = {header_map[k]: v for k, v in raw.items() if k in header_map}
            if any(v is None or v == "" for v in values.values()):
                continue  # blank / ragged line
            rows.append({
                "frame_id": int(float(values["frame_id"])),
                "x_min": float(values["x_min"]),
                "y_min": float(values["y_min"]),
                "x_max": float(values["x_max"]),
                "y_max": float(values["y_max"]),
            })
    return rows


def frame_id_from_name(name):
    """Pull the frame index out of a filename. Uses the *last* run of digits so
    that `left000038.png`, `depth000038.npz` and `frame_0038.npz` all agree even
    when a prefix happens to contain digits (`cam2_frame_0038.png`)."""
    runs = re.findall(r"\d+", os.path.basename(name))
    return int(runs[-1]) if runs else None


def build_file_index(directory, extensions=None):
    """frame id -> path for every file in `directory` with a numeric suffix."""
    index = {}
    if not os.path.isdir(directory):
        return index
    for path in sorted(glob.glob(os.path.join(directory, "*"))):
        if not os.path.isfile(path):
            continue
        if extensions and os.path.splitext(path)[1].lower() not in extensions:
            continue
        fid = frame_id_from_name(path)
        if fid is not None:
            index[fid] = path
    return index


def get_valid_frames(rows, dataset_dir):
    """Drop degenerate bboxes, attach rgb/xyz paths, sort chronologically.

    Returns `(frames, stats)`; `stats` counts why rows were dropped so `main`
    can report the data yield instead of silently swallowing frames."""
    rgb_index = build_file_index(os.path.join(dataset_dir, "rgb"), {".png", ".jpg", ".jpeg"})
    xyz_index = build_file_index(os.path.join(dataset_dir, "xyz"), {".npz", ".npy"})

    frames, stats = [], {"degenerate_bbox": 0, "missing_xyz": 0, "kept": 0}
    for row in rows:
        if row["x_max"] <= row["x_min"] or row["y_max"] <= row["y_min"]:
            stats["degenerate_bbox"] += 1
            continue
        xyz_path = xyz_index.get(row["frame_id"])
        if xyz_path is None:
            stats["missing_xyz"] += 1
            continue
        entry = dict(row)
        entry["rgb_path"] = rgb_index.get(row["frame_id"])
        entry["xyz_path"] = xyz_path
        frames.append(entry)
        stats["kept"] += 1

    frames.sort(key=lambda e: e["frame_id"])
    return frames, stats


# --------------------------------------------------------------------------
# 2. Per-frame 3D localisation of the traffic light
# --------------------------------------------------------------------------

def load_xyz(path):
    """Load one frame's point cloud as (H, W, >=3), float64.

    The shipped files store the array under the key `"xyz"` with a 4th channel
    that is only 0/NaN/inf; the spec says key `"points"` with 3 channels. Both
    are handled: whichever key exists is used and only X/Y/Z are returned."""
    if path.endswith(".npy"):
        array = np.load(path)
    else:
        with np.load(path) as data:
            key = "points" if "points" in data.files else ("xyz" if "xyz" in data.files else data.files[0])
            array = data[key]
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"{path}: expected (H, W, >=3) point array, got {array.shape}")
    return np.asarray(array[:, :, :3], dtype=np.float64)


def detect_axis_convention(xyz):
    """Measure the camera frame's Y and Z sign conventions from the data itself.

    The challenge text says "+Y right, +Z up, right-handed", which cannot all be
    true at once, so guessing is not safe. Instead correlate each channel with
    pixel position: if Y falls as the column index u rises, +Y points left; if Z
    falls as the row index v rises, +Z points up. Returns
    `{"y_is_left": bool, "z_is_up": bool}`."""
    h, w = xyz.shape[:2]
    vs, us = np.mgrid[0:h, 0:w]
    valid = np.isfinite(xyz).all(axis=2) & ~np.all(xyz == 0, axis=2) & (xyz[:, :, 0] > 0)
    if valid.sum() < 100:
        raise ValueError("Too few valid points to detect the axis convention")
    u_flat, v_flat = us[valid].astype(float), vs[valid].astype(float)
    y_flat, z_flat = xyz[:, :, 1][valid], xyz[:, :, 2][valid]
    return {
        "y_is_left": float(np.corrcoef(u_flat, y_flat)[0, 1]) < 0,
        "z_is_up": float(np.corrcoef(v_flat, z_flat)[0, 1]) < 0,
    }


def light_center_pixel(row):
    """Bbox -> integer centre pixel (u, v)."""
    return int(round((row["x_min"] + row["x_max"]) / 2.0)), int(round((row["y_min"] + row["y_max"]) / 2.0))


def estimate_light_position(xyz, row, inner_fraction=0.5, min_samples=8, mad_scale=3.0):
    """Robust (X, Y, Z) of the traffic light in camera coords, or None.

    Rather than trusting the single centre pixel (noisy, and sometimes a stereo
    hole on the light's dark housing) this averages a patch, as the challenge
    suggests: take the central `inner_fraction` of the bbox, keep only finite,
    non-zero, in-front-of-the-camera points, then reject range outliers with a
    median-absolute-deviation gate before taking the component-wise median. The
    MAD gate is what removes sky/background pixels that leak in around the
    light's silhouette - those sit at a wildly different range from the light.

    NOTE the index order: the array is (H, W, 3), so pixel (u, v) is `xyz[v, u]`.
    """
    h, w = xyz.shape[:2]
    u, v = light_center_pixel(row)
    u = int(np.clip(u, 0, w - 1))
    v = int(np.clip(v, 0, h - 1))

    half_u = max(1, int(round(inner_fraction * (row["x_max"] - row["x_min"]) / 2.0)))
    half_v = max(1, int(round(inner_fraction * (row["y_max"] - row["y_min"]) / 2.0)))
    patch = xyz[max(0, v - half_v):min(h, v + half_v + 1),
                max(0, u - half_u):min(w, u + half_u + 1)].reshape(-1, 3)

    keep = np.isfinite(patch).all(axis=1) & ~np.all(patch == 0, axis=1) & (patch[:, 0] > 0.0)
    points = patch[keep]
    if len(points) < min_samples:
        return None

    ranges = np.linalg.norm(points, axis=1)
    median_range = np.median(ranges)
    mad = np.median(np.abs(ranges - median_range))
    # 1.4826 * MAD approximates sigma for normal data; the 0.25 m floor keeps the
    # gate from collapsing when the patch is already tight.
    gate = max(mad_scale * 1.4826 * mad, 0.25)
    inliers = points[np.abs(ranges - median_range) <= gate]
    if len(inliers) < min_samples:
        return None

    return {
        "xyz": np.median(inliers, axis=0),
        "n_samples": int(len(points)),
        "n_inliers": int(len(inliers)),
        "range_mad": float(mad),
    }


def to_ground_vector(xyz_cam, y_is_left):
    """Camera (X, Y, Z) -> the car->light vector on the ground plane, expressed
    with the world frame's handedness (X forward, Y **left**). Height is dropped
    here: only the ground projection is asked for."""
    y = xyz_cam[1] if y_is_left else -xyz_cam[1]
    return np.array([xyz_cam[0], y], dtype=float)


# --------------------------------------------------------------------------
# 3. Temporal cleaning
# --------------------------------------------------------------------------

def reject_height_outliers(heights, mad_scale=4.0, floor=0.30):
    """Boolean keep-mask over frames, using the fact that a fixed overhead light
    must stay at a constant height above the (locally flat) road. A frame whose
    measured Z jumps away from the clip median is a mis-measured patch, not a
    moving traffic light, so it is dropped."""
    heights = np.asarray(heights, dtype=float)
    median = np.median(heights)
    mad = np.median(np.abs(heights - median))
    gate = max(mad_scale * 1.4826 * mad, floor)
    return np.abs(heights - median) <= gate


def local_linear_fit(times, values, half_window):
    """Local linear regression evaluated at each sample; returns
    `(fitted_value, fitted_slope)`.

    A boxcar average would be wrong twice over here: it shrinks curvature at the
    turn, and it treats samples as evenly spaced when this dataset has gaps
    (frames 38-127 are sparse). Fitting a line over a *time* window and
    evaluating it at the sample's own timestamp handles both, and the fit's slope
    is a far steadier derivative than a two-sample difference. Falls back to the
    raw value (and a zero slope) where a window holds fewer than two distinct
    timestamps."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    fitted = np.empty_like(values)
    slope = np.zeros_like(values)
    for i, t_i in enumerate(times):
        window = np.abs(times - t_i) <= half_window
        if window.sum() < 2 or np.ptp(times[window]) == 0:
            fitted[i] = values[i]
            continue
        dt = times[window] - t_i
        design = np.vstack([dt, np.ones_like(dt)]).T
        coef, *_ = np.linalg.lstsq(design, values[window], rcond=None)
        slope[i], fitted[i] = coef[0], coef[1]  # slope, and the line at dt = 0
    return fitted, slope


def local_linear_smooth(times, values, half_window):
    """Just the fitted values from `local_linear_fit`."""
    return local_linear_fit(times, values, half_window)[0]


def smooth_vectors(times, vectors, half_window):
    """`local_linear_smooth` applied column-wise to an (N, 2) array."""
    return np.stack([local_linear_smooth(times, vectors[:, d], half_window)
                     for d in range(vectors.shape[1])], axis=1)


def speed_profile(times, positions, half_window):
    """Instantaneous speed from the local-linear slope of x(t) and y(t).

    Differencing consecutive samples would work out to metres of travel divided
    by 1/30 s, so a centimetre of depth noise becomes a metre per second of
    phantom speed; the windowed fit keeps the estimate readable."""
    _, vx = local_linear_fit(times, positions[:, 0], half_window)
    _, vy = local_linear_fit(times, positions[:, 1], half_window)
    return np.hypot(vx, vy)


# --------------------------------------------------------------------------
# 4. World frame + the two heading models
# --------------------------------------------------------------------------

def rotation(psi):
    """2x2 rotation by `psi` (right-handed, +psi turns +X toward +Y = left)."""
    cos_p, sin_p = np.cos(psi), np.sin(psi)
    return np.array([[cos_p, -sin_p], [sin_p, cos_p]])


def reference_heading(c0):
    """Vehicle heading at the reference frame, in world radians.

    The world frame is *defined* so the car->light line at t0 lies along +X. The
    car->light vector in camera coords is c0, so rotating the camera frame by
    psi0 = -atan2(c0_y, c0_x) puts it on +X. This is a definitional choice, not a
    measurement - there is no independent heading source in the dataset."""
    return -np.arctan2(c0[1], c0[0])


def solve_nonholonomic_yaw(vectors, max_yaw_step=np.radians(8.0), iterations=80):
    """Recover a per-frame heading from the one piece of physics available for
    free in this dataset: **a car cannot drive sideways.**

    Unknowns are the headings psi_t; psi_0 is pinned by the world-frame
    definition. For each consecutive pair the ego positions are fully determined
    by the headings (`p_t = -R(psi_t) c_t`), and the non-holonomic constraint says
    the chord `p_{t+1} - p_t` must point along the vehicle's heading. For motion
    on a constant-curvature arc the chord direction is exactly the *mean* of the
    two headings, so

        cross( p_{t+1}(psi_{t+1}) - p_t ,  u((psi_t + psi_{t+1}) / 2) ) = 0

    is one scalar equation in the single unknown psi_{t+1}, solved by bisection
    inside +-`max_yaw_step`. That makes the whole heading sequence a forward
    recursion with no extra sensor and no extra dependency.

    Returns `(headings, positions)`."""
    vectors = np.asarray(vectors, dtype=float)
    n = len(vectors)
    headings = np.zeros(n)
    positions = np.zeros((n, 2))
    headings[0] = reference_heading(vectors[0])
    positions[0] = -(rotation(headings[0]) @ vectors[0])

    for t in range(n - 1):
        def residual(delta):
            nxt = -(rotation(headings[t] + delta) @ vectors[t + 1])
            chord = nxt - positions[t]
            mean_heading = headings[t] + delta / 2.0
            return chord[0] * np.sin(mean_heading) - chord[1] * np.cos(mean_heading)

        lo, hi = -max_yaw_step, max_yaw_step
        f_lo, f_hi = residual(lo), residual(hi)
        if f_lo * f_hi > 0:
            delta = 0.0  # no bracketed root (near-stationary frame) - hold heading
        else:
            for _ in range(iterations):
                mid = 0.5 * (lo + hi)
                f_mid = residual(mid)
                if f_lo * f_mid <= 0:
                    hi, f_hi = mid, f_mid
                else:
                    lo, f_lo = mid, f_mid
            delta = 0.5 * (lo + hi)

        headings[t + 1] = headings[t] + delta
        positions[t + 1] = -(rotation(headings[t + 1]) @ vectors[t + 1])

    return headings, positions


# --------------------------------------------------------------------------
# 5. Metrics
# --------------------------------------------------------------------------

def sideslip_angles(headings, positions):
    """Per-step angle (radians) between the travelled chord and the vehicle's
    mean heading over that step - the residual of the constraint that
    `solve_nonholonomic_yaw` drives to zero.

    A real car holds this near zero, so it should come out at ~0 for every step
    the solver bracketed. It is not decoration: a step where bisection found no
    sign change falls back to holding the heading, and that shows up here as a
    non-zero residual. Watch the p95, not the median."""
    chords = np.diff(positions, axis=0)
    mean_heading = 0.5 * (headings[:-1] + headings[1:])
    error = np.arctan2(chords[:, 1], chords[:, 0]) - mean_heading
    # Wrap to (-pi, pi], take the magnitude, then fold onto [0, pi/2]: driving in
    # reverse is still motion along the axle, not sideslip.
    wrapped = np.abs(np.arctan2(np.sin(error), np.cos(error)))
    slip = np.minimum(wrapped, np.pi - wrapped)
    return np.where(np.linalg.norm(chords, axis=1) > 1e-6, slip, 0.0)


SPEED_HALF_WINDOW_S = 10.0 / DEFAULT_FPS


def trajectory_metrics(times, headings, positions, speed_half_window=SPEED_HALF_WINDOW_S):
    step_len = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    speed = speed_profile(times, positions, speed_half_window)
    slip = sideslip_angles(headings, positions)
    duration = times[-1] - times[0]
    return {
        "path_length_m": float(step_len.sum()),
        "net_displacement_m": float(np.linalg.norm(positions[-1] - positions[0])),
        "mean_speed_mps": float(step_len.sum() / duration) if duration > 0 else 0.0,
        "max_speed_mps": float(speed.max()) if len(speed) else 0.0,
        "total_yaw_deg": float(np.degrees(headings[-1] - headings[0])),
        "median_sideslip_deg": float(np.degrees(np.median(slip))) if len(slip) else 0.0,
        "p95_sideslip_deg": float(np.degrees(np.percentile(slip, 95))) if len(slip) else 0.0,
        "speed": speed,
    }


# --------------------------------------------------------------------------
# 6. Rendering
#
# Laid out after the sample BEV plot in `instructions.md`: framed axes, a dashed
# grid, thin axis lines through the origin, an X at the start, a filled dot at
# the end, a star on the traffic light with an "Origin" label, and a boxed
# legend. The hues are the validated categorical slots rather than the sample's
# red/green pair - the marker shapes already carry the distinction, so swapping
# in a colour-vision-safe pair costs nothing.
# --------------------------------------------------------------------------

def _style_axes(ax, title=None, xlabel=None, ylabel=None, origin_lines=False):
    ax.set_facecolor(SURFACE)
    ax.grid(True, linestyle="--", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=10)
    if origin_lines:
        ax.axhline(0.0, color=INK, linewidth=0.8, zorder=1)
        ax.axvline(0.0, color=INK, linewidth=0.8, zorder=1)
    if title:
        # Generous pad: the metrics subtitle is drawn into this gap.
        ax.set_title(title, color=INK, fontsize=15, pad=30)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=12)


def _legend(ax, loc="upper right"):
    legend = ax.legend(loc=loc, frameon=True, fontsize=11)
    frame = legend.get_frame()
    frame.set_edgecolor(AXIS)
    frame.set_facecolor(SURFACE)
    for text in legend.get_texts():
        text.set_color(INK)
    return legend


def _set_equal_aspect_limits(fig, ax, points, pad=3.0):
    """Equal-aspect limits that contain `points` and fill the axes box.

    A BEV has to be equal-aspect or distances lie, but matplotlib's own aspect
    handling is free to *shrink* one axis to satisfy the ratio - which crops the
    origin or the start of the path out of frame - or to collapse the axes into a
    short strip that the legend then runs into. Sizing the limits to the box
    ourselves only ever expands. Call it after `tight_layout`, so the box
    position is final."""
    box = ax.get_position()
    width_in = box.width * fig.get_figwidth()
    height_in = box.height * fig.get_figheight()
    low = np.min(points, axis=0) - pad
    high = np.max(points, axis=0) + pad
    span = high - low
    target = width_in / height_in
    if span[0] / span[1] < target:
        span[0] = span[1] * target
    else:
        span[1] = span[0] / target
    centre = (low + high) / 2.0
    ax.set_xlim(centre[0] - span[0] / 2, centre[0] + span[0] / 2)
    ax.set_ylim(centre[1] - span[1] / 2, centre[1] + span[1] / 2)
    ax.set_aspect("equal", adjustable="box")


def _fit_layout(fig, ax, points, pad=3.0):
    """Lay out the figure and fit the equal-aspect view to it.

    Two passes, because the two steps depend on each other: `tight_layout` sizes
    the axes box from the current tick labels, and fitting the view changes those
    labels' widths. One pass leaves the y-label clipped off the canvas."""
    for _ in range(2):
        fig.tight_layout()
        _set_equal_aspect_limits(fig, ax, points, pad=pad)


def _draw_landmark(ax, label_side="left"):
    """The traffic light at the world origin, with its direct label.

    The label goes on whichever side has room: the car approaches the light from
    -X, so the origin sits at the right-hand edge of the view and a right-hand
    label would clip."""
    ax.plot([0], [0], marker="*", markersize=17, color=INK, linestyle="None",
            zorder=5, label="Traffic light (origin)")
    offset, align = ((-12, 5), "right") if label_side == "left" else ((12, 5), "left")
    ax.annotate("Origin", xy=(0, 0), xytext=offset, textcoords="offset points",
                ha=align, color=INK, fontsize=11, zorder=6)


def plot_trajectory_png(positions, out_path, subtitle=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6.0), facecolor=SURFACE)
    ax.plot(positions[:, 0], positions[:, 1], "-", linewidth=2.0, color=C_PATH,
            zorder=3, label="Ego trajectory")
    ax.plot([positions[0, 0]], [positions[0, 1]], marker="x", markersize=12,
            markeredgewidth=3, color=C_START, linestyle="None", zorder=5, label="Start")
    ax.plot([positions[-1, 0]], [positions[-1, 1]], marker="o", markersize=11,
            color=C_END, linestyle="None", zorder=5, label="End")
    _draw_landmark(ax)

    _style_axes(ax, title="Ego Trajectory in the Traffic-Light Ground Frame",
                xlabel="Forward (X, m)", ylabel="Lateral (Y, m)", origin_lines=True)
    if subtitle:
        ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center",
                color=INK_MUTED, fontsize=10.5)
    _legend(ax)

    _fit_layout(fig, ax, np.vstack([positions, np.zeros((1, 2))]))
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_diagnostics_png(frame_ids, times, vectors, headings, metrics, heights, out_path):
    """Extra plot (the brief invites them): the raw measurements the trajectory
    rests on, so a reader can judge the result rather than take it on trust."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranges = np.linalg.norm(vectors, axis=1)
    bearings = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor=SURFACE)
    panels = [
        (axes[0][0], times, ranges, "Range to the traffic light", "distance (m)"),
        (axes[0][1], times, bearings, "Light bearing in the camera frame (+ = left)", "bearing (deg)"),
        (axes[1][0], times, metrics["speed"], "Ego speed (local-linear fit)", "speed (m/s)"),
        (axes[1][1], times, np.degrees(headings - headings[0]), "Recovered heading change", "yaw (deg)"),
    ]
    for ax, x, y, title, ylabel in panels:
        ax.plot(x, y, "-", linewidth=2.0, color=C_PATH)
        _style_axes(ax, title=title, xlabel="time (s)", ylabel=ylabel)
        ax.title.set_fontsize(12)

    fig.suptitle(
        f"Measurement diagnostics  -  light height above the camera: "
        f"{np.median(heights):.2f} m (sd {np.std(heights):.2f} m over {len(heights)} frames)",
        color=INK_MUTED, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def animate_trajectory_mp4(frame_ids, positions, out_path, fps):
    """Animate on the *source* frame grid, not the sample index, so the playback
    pacing matches real time even though frames 38-127 are sparsely sampled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    fig, ax = plt.subplots(figsize=(11, 6.0), facecolor=SURFACE)
    (trail,) = ax.plot([], [], "-", linewidth=2.0, color=C_PATH, zorder=3,
                       label="Ego trajectory")
    ax.plot([positions[0, 0]], [positions[0, 1]], marker="x", markersize=12,
            markeredgewidth=3, color=C_START, linestyle="None", zorder=5, label="Start")
    (car,) = ax.plot([], [], marker="o", markersize=11, color=C_END,
                     markeredgecolor=SURFACE, markeredgewidth=1.2, linestyle="None",
                     zorder=6, label="Current position")
    _draw_landmark(ax)
    clock = ax.text(0.99, 0.02, "", transform=ax.transAxes, ha="right",
                    color=INK_MUTED, fontsize=11)

    _style_axes(ax, title="Ego Trajectory in the Traffic-Light Ground Frame",
                xlabel="Forward (X, m)", ylabel="Lateral (Y, m)", origin_lines=True)
    _legend(ax)

    # Fix the view once, up front: the path grows during the animation and the
    # frame must not rescale under it.
    _fit_layout(fig, ax, np.vstack([positions, np.zeros((1, 2))]))
    ax.autoscale(False)

    first, last = int(frame_ids[0]), int(frame_ids[-1])
    grid = np.arange(first, last + 1)
    # For each source frame, how many trajectory samples exist up to it.
    counts = np.searchsorted(frame_ids, grid, side="right")

    def init():
        trail.set_data([], [])
        car.set_data([], [])
        clock.set_text("")
        return trail, car, clock

    def update(step):
        count = max(1, int(counts[step]))
        trail.set_data(positions[:count, 0], positions[:count, 1])
        car.set_data([positions[count - 1, 0]], [positions[count - 1, 1]])
        clock.set_text(f"frame {grid[step]}   t = {grid[step] / fps:5.2f} s")
        return trail, car, clock

    anim = FuncAnimation(fig, update, frames=len(grid), init_func=init, blit=True)
    anim.save(out_path, writer=FFMpegWriter(fps=fps, bitrate=2400))
    plt.close(fig)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def build_trajectory(dataset_dir, fps=DEFAULT_FPS, smooth_frames=6.0, log=print):
    """Run the whole estimation pipeline and return everything the renderers and
    the report need. Kept separate from `main` so it is importable and testable."""
    csv_path = find_bbox_csv(dataset_dir)
    rows = load_bbox_csv(csv_path)
    log(f"Loaded {len(rows)} bbox rows from {csv_path}")

    frames, stats = get_valid_frames(rows, dataset_dir)
    log(f"  {stats['degenerate_bbox']} rows dropped: degenerate bbox (light not detected)")
    log(f"  {stats['missing_xyz']} rows dropped: no matching .npz depth file")
    log(f"  {stats['kept']} frames carried forward")
    if len(frames) < 2:
        raise RuntimeError("Fewer than two usable frames - cannot build a trajectory")

    convention = detect_axis_convention(load_xyz(frames[0]["xyz_path"]))
    log(f"Measured camera convention: +Y is "
        f"{'LEFT' if convention['y_is_left'] else 'RIGHT'}, +Z is "
        f"{'UP' if convention['z_is_up'] else 'DOWN'}")
    if not convention["z_is_up"]:
        raise RuntimeError("Camera +Z is not up; the ground-plane projection assumes it is")

    frame_ids, raw_vectors, heights, dropped_patch = [], [], [], 0
    for entry in frames:
        estimate = estimate_light_position(load_xyz(entry["xyz_path"]), entry)
        if estimate is None:
            dropped_patch += 1
            continue
        frame_ids.append(entry["frame_id"])
        raw_vectors.append(to_ground_vector(estimate["xyz"], convention["y_is_left"]))
        heights.append(estimate["xyz"][2])
    log(f"  {dropped_patch} frames dropped: no usable depth inside the light's bbox")

    frame_ids = np.array(frame_ids, dtype=float)
    raw_vectors = np.array(raw_vectors)
    heights = np.array(heights)

    keep = reject_height_outliers(heights)
    log(f"  {int((~keep).sum())} frames dropped: light height inconsistent with the clip median")
    frame_ids, raw_vectors, heights = frame_ids[keep], raw_vectors[keep], heights[keep]
    if len(frame_ids) < 2:
        raise RuntimeError("Fewer than two usable frames after filtering")

    times = frame_ids / fps
    vectors = smooth_vectors(times, raw_vectors, half_window=smooth_frames / fps)
    log(f"{len(frame_ids)} usable frames "
        f"(source frames {int(frame_ids[0])}-{int(frame_ids[-1])}, {times[-1] - times[0]:.2f} s)")

    headings, positions = solve_nonholonomic_yaw(vectors)
    metrics = trajectory_metrics(times, headings, positions)

    log(f"Reference frame {int(frame_ids[0])}: light at camera (X={raw_vectors[0][0]:.2f} m, "
        f"Y={raw_vectors[0][1]:.2f} m), i.e. {np.degrees(np.arctan2(*raw_vectors[0][::-1])):.1f}° to the left")
    log(f"Light height above the camera: median {np.median(heights):.2f} m, sd {np.std(heights):.2f} m "
        f"(a fixed light should hold this constant - it is the main independent sanity check)")
    log(f"Path {metrics['path_length_m']:.1f} m, total yaw {metrics['total_yaw_deg']:+.1f}°, "
        f"mean speed {metrics['mean_speed_mps']:.2f} m/s "
        f"({metrics['mean_speed_mps'] * 2.237:.1f} mph)")
    log(f"Non-holonomic residual: p95 sideslip {metrics['p95_sideslip_deg']:.3f}° "
        f"(should be ~0 - every step the solver bracketed satisfies the constraint)")
    log(f"Trajectory X ∈ [{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}] m, "
        f"Y ∈ [{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}] m; "
        f"the car ends {np.linalg.norm(positions[-1]):.1f} m from the light")

    return {
        "frame_ids": frame_ids,
        "times": times,
        "raw_vectors": raw_vectors,
        "vectors": vectors,
        "heights": heights,
        "headings": headings,
        "positions": positions,
        "metrics": metrics,
        "convention": convention,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="dataset", help="dataset directory (default: dataset)")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="source frame rate")
    parser.add_argument("--smooth-frames", type=float, default=6.0,
                        help="half-width, in source frames, of the local-linear smoother")
    parser.add_argument("--png", default="trajectory.png")
    parser.add_argument("--mp4", default="trajectory.mp4")
    parser.add_argument("--diagnostics", default="diagnostics.png")
    parser.add_argument("--no-video", action="store_true", help="skip the mp4 (no ffmpeg needed)")
    args = parser.parse_args(argv)

    result = build_trajectory(args.dataset, fps=args.fps, smooth_frames=args.smooth_frames)

    metrics = result["metrics"]
    subtitle = (f"{len(result['frame_ids'])} frames  ·  {metrics['path_length_m']:.1f} m driven  "
                f"·  {metrics['total_yaw_deg']:+.0f}° net yaw  "
                f"·  {metrics['mean_speed_mps']:.1f} m/s mean speed")
    plot_trajectory_png(result["positions"], args.png, subtitle=subtitle)
    print(f"Wrote {args.png}")

    plot_diagnostics_png(result["frame_ids"], result["times"], result["vectors"],
                         result["headings"], metrics, result["heights"], args.diagnostics)
    print(f"Wrote {args.diagnostics}")

    if not args.no_video:
        animate_trajectory_mp4(result["frame_ids"], result["positions"], args.mp4, fps=args.fps)
        print(f"Wrote {args.mp4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
