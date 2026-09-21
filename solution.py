"""
Part A: Ego-vehicle trajectory reconstruction using the traffic light as a
fixed world reference. See PLAN.md for the full algorithm write-up.
"""

import csv
import glob
import os
import re

import numpy as np


# ---------------------------------------------------------------------------
# 2. Pre-processing pipeline
# ---------------------------------------------------------------------------

_COLUMN_ALIASES = {
    "frame": "frame_id",
    "frame_id": "frame_id",
    "x1": "x_min",
    "x_min": "x_min",
    "y1": "y_min",
    "y_min": "y_min",
    "x2": "x_max",
    "x_max": "x_max",
    "y2": "y_max",
    "y_max": "y_max",
}


def load_bbox_csv(path):
    """Parse the traffic-light bbox CSV, normalizing either known column set
    (README spec: frame_id,x_min,y_min,x_max,y_max; sample: frame,x1,y1,x2,y2)
    into a list of dicts with keys frame_id, x_min, y_min, x_max, y_max."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header_map = {}
        for col in reader.fieldnames:
            key = col.strip().lower()
            if key not in _COLUMN_ALIASES:
                raise ValueError(f"Unrecognized bbox CSV column: {col!r}")
            header_map[col] = _COLUMN_ALIASES[key]

        for raw_row in reader:
            row = {header_map[k]: v for k, v in raw_row.items()}
            rows.append(
                {
                    "frame_id": int(row["frame_id"]),
                    "x_min": float(row["x_min"]),
                    "y_min": float(row["y_min"]),
                    "x_max": float(row["x_max"]),
                    "y_max": float(row["y_max"]),
                }
            )
    return rows


def _build_file_index(directory):
    """Map frame id (int) -> file path for every file in `directory` whose
    name contains a run of digits."""
    index = {}
    if not os.path.isdir(directory):
        return index
    for path in glob.glob(os.path.join(directory, "*")):
        name = os.path.basename(path)
        match = re.search(r"(\d+)", name)
        if match is None:
            continue
        index[int(match.group(1))] = path
    return index


def get_valid_frames(rows, dataset_dir):
    """Filter degenerate bbox rows, match each remaining frame to its rgb and
    xyz files, and return a chronologically sorted list of dicts:
    {frame_id, x_min, y_min, x_max, y_max, rgb_path, xyz_path}."""
    rgb_index = _build_file_index(os.path.join(dataset_dir, "rgb"))
    xyz_index = _build_file_index(os.path.join(dataset_dir, "xyz"))

    valid = []
    for row in rows:
        if row["x_max"] <= row["x_min"] or row["y_max"] <= row["y_min"]:
            continue  # degenerate / not detected this frame
        frame_id = row["frame_id"]
        xyz_path = xyz_index.get(frame_id)
        if xyz_path is None:
            continue  # no depth for this frame
        entry = dict(row)
        entry["rgb_path"] = rgb_index.get(frame_id)
        entry["xyz_path"] = xyz_path
        valid.append(entry)

    valid.sort(key=lambda e: e["frame_id"])
    return valid


# ---------------------------------------------------------------------------
# 3. Per-frame traffic-light 3D localization
# ---------------------------------------------------------------------------

def load_xyz(path):
    """Load a per-frame point-cloud array, shape (H, W, >=3)."""
    data = np.load(path)
    key = "points" if "points" in data.files else data.files[0]
    return data[key]


def light_center_pixel(row):
    """Bbox -> integer pixel center (u, v)."""
    u = round((row["x_min"] + row["x_max"]) / 2.0)
    v = round((row["y_min"] + row["y_max"]) / 2.0)
    return int(u), int(v)


def estimate_light_position(xyz, u, v, bbox_w, bbox_h):
    """Patch-average around (u, v) -> median (X, Y, Z) in camera coords, or
    None if no valid points are found. xyz is indexed [row=v, col=u]."""
    h, w = xyz.shape[0], xyz.shape[1]
    u = int(np.clip(u, 0, w - 1))
    v = int(np.clip(v, 0, h - 1))

    half = max(1, int(round(0.25 * min(bbox_w, bbox_h))))
    u0, u1 = max(0, u - half), min(w, u + half + 1)
    v0, v1 = max(0, v - half), min(h, v + half + 1)

    patch = xyz[v0:v1, u0:u1, :3].reshape(-1, 3)

    valid_mask = np.isfinite(patch).all(axis=1)
    valid_mask &= ~np.all(patch == 0, axis=1)
    valid_mask &= patch[:, 0] > 0  # Z<=0 in README frame == X<=0 here (+X forward)

    valid = patch[valid_mask]
    if valid.shape[0] == 0:
        return None

    return tuple(np.median(valid, axis=0))


# ---------------------------------------------------------------------------
# 4. World-frame construction
# ---------------------------------------------------------------------------

def compute_reference_rotation(x0, y0):
    """2x2 rotation matrix R0 that rotates (x0, y0) onto (r0, 0)."""
    theta0 = np.arctan2(y0, x0)
    cos_t, sin_t = np.cos(theta0), np.sin(theta0)
    # Rotating a vector by -theta0 aligns it with +X.
    return np.array([[cos_t, sin_t], [-sin_t, cos_t]])


def compute_trajectory(frames_data, r0):
    """frames_data: list of (frame_id, X, Y, Z) camera-frame light positions.
    Returns (frame_ids, traj) where traj is an (N, 2) array of (x_m, y_m)."""
    frame_ids = []
    traj = []
    for frame_id, x, y, _z in frames_data:
        world_vec = r0 @ np.array([x, y])
        car_pos = -world_vec
        frame_ids.append(frame_id)
        traj.append(car_pos)
    return np.array(frame_ids), np.array(traj)


def smooth_trajectory(traj, window=9):
    """Centered moving-average smoothing of the (x, y) trajectory, to reduce
    per-frame depth-estimation jitter without shifting the path's overall
    shape. Edge-padded so the smoothed array has the same length and the
    start/end points don't get pulled toward zero. No-op if `traj` is
    shorter than `window`."""
    window = window if window % 2 == 1 else window + 1  # keep it centered
    if window < 3 or len(traj) < window:
        return traj.copy()

    pad = window // 2
    padded = np.pad(traj, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.empty_like(traj, dtype=float)
    for dim in range(traj.shape[1]):
        smoothed[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return smoothed


def _estimate_velocity(points):
    """Least-squares linear fit of x(t) and y(t) over `points` (t = index),
    returning the per-step velocity vector (dx/dt, dy/dt). Far more stable
    than a two-point difference since it uses every point in the window to
    minimize squared error, rather than being fully determined by whichever
    two points happen to be noisiest."""
    t = np.arange(len(points))
    design = np.vstack([t, np.ones_like(t)]).T
    vx = np.linalg.lstsq(design, points[:, 0], rcond=None)[0][0]
    vy = np.linalg.lstsq(design, points[:, 1], rcond=None)[0][0]
    return np.array([vx, vy])


def predict_path_to_origin(traj, velocity_window=20, max_extra_steps=1000):
    """Extrapolate the observed trajectory forward under a constant-velocity
    assumption, stepping until the predicted position reaches its closest
    approach to the origin (the traffic light). Velocity is a least-squares
    fit over the last `velocity_window` points (see `_estimate_velocity`),
    not a raw two-point difference, so a single noisy point near the end of
    the clip doesn't swing the predicted direction.

    The recorded clip ends before the car reaches the light, so without this
    the plotted path stops short; this fills in the (unobserved) remainder
    of the trip so the full path to the origin is visible. Returns an
    (M, 2) array of predicted points *after* the last observed point (not
    including it), or an empty array if the recent trend isn't actually
    heading toward the origin.
    """
    n = min(velocity_window, len(traj))
    if n < 2:
        return np.empty((0, 2))

    velocity = _estimate_velocity(traj[-n:])
    speed = np.linalg.norm(velocity)
    if speed < 1e-9:
        return np.empty((0, 2))

    last = traj[-1]
    dist_to_origin = np.linalg.norm(last)
    direction_to_origin = -last / dist_to_origin if dist_to_origin > 1e-9 else np.zeros(2)
    if np.dot(velocity / speed, direction_to_origin) <= 0:
        return np.empty((0, 2))  # current trend isn't heading toward the light

    predicted = []
    pos = last.copy()
    prev_dist = dist_to_origin
    for _ in range(max_extra_steps):
        pos = pos + velocity
        dist = np.linalg.norm(pos)
        if dist > prev_dist:
            break  # passed closest approach to the origin
        predicted.append(pos.copy())
        prev_dist = dist

    return np.array(predicted) if predicted else np.empty((0, 2))


# ---------------------------------------------------------------------------
# 5. Outputs
# ---------------------------------------------------------------------------

def plot_trajectory_png(traj, out_path, predicted=None):
    import matplotlib.pyplot as plt

    if predicted is None:
        predicted = np.empty((0, 2))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(traj[:, 0], traj[:, 1], "-o", color="tab:blue", markersize=3, linewidth=1, label="Observed trajectory")
    if len(predicted):
        extension = np.vstack([traj[-1:], predicted])
        ax.plot(extension[:, 0], extension[:, 1], "--", color="tab:orange", linewidth=1.5,
                label="Predicted (extrapolated) path")
        ax.plot(predicted[-1, 0], predicted[-1, 1], marker="x", color="tab:orange", markersize=10,
                label="Predicted closest approach")
    ax.plot(0, 0, marker="*", color="tab:red", markersize=18, linestyle="None", label="Traffic light (origin)")
    ax.plot(traj[0, 0], traj[0, 1], marker="^", color="tab:green", markersize=10, linestyle="None", label="Start")
    ax.plot(traj[-1, 0], traj[-1, 1], marker="s", color="tab:blue", markersize=10, linestyle="None",
            label="Last observed")

    ax.set_xlabel("X (m, world frame)")
    ax.set_ylabel("Y (m, world frame)")
    ax.set_title("Ego-Vehicle Ground-Plane Trajectory")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def animate_trajectory_mp4(traj, out_path, fps, predicted=None):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    if predicted is None:
        predicted = np.empty((0, 2))

    all_points = np.vstack([traj, predicted]) if len(predicted) else traj

    fig, ax = plt.subplots(figsize=(8, 8))
    pad = 5.0
    x_min = min(all_points[:, 0].min(), 0.0)
    x_max = max(all_points[:, 0].max(), 0.0)
    y_min = min(all_points[:, 1].min(), 0.0)
    y_max = max(all_points[:, 1].max(), 0.0)
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel("X (m, world frame)")
    ax.set_ylabel("Y (m, world frame)")
    ax.set_title("Ego-Vehicle Ground-Plane Trajectory")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.plot(0, 0, marker="*", color="tab:red", markersize=18, linestyle="None", label="Traffic light (origin)")

    (trail_line,) = ax.plot([], [], "-", color="tab:blue", linewidth=1, label="Observed")
    (pred_line,) = ax.plot([], [], "--", color="tab:orange", linewidth=1.5, label="Predicted (extrapolated)")
    (marker,) = ax.plot([], [], marker="o", color="tab:blue", markersize=6)
    ax.legend(loc="upper right")

    n_real = len(traj)
    n_total = n_real + len(predicted)

    def init():
        trail_line.set_data([], [])
        pred_line.set_data([], [])
        marker.set_data([], [])
        return trail_line, pred_line, marker

    def update(frame_idx):
        if frame_idx < n_real:
            trail_line.set_data(traj[: frame_idx + 1, 0], traj[: frame_idx + 1, 1])
            marker.set_data([traj[frame_idx, 0]], [traj[frame_idx, 1]])
        else:
            trail_line.set_data(traj[:, 0], traj[:, 1])
            k = frame_idx - n_real + 1
            pred_segment = np.vstack([traj[-1:], predicted[:k]])
            pred_line.set_data(pred_segment[:, 0], pred_segment[:, 1])
            marker.set_data([predicted[k - 1, 0]], [predicted[k - 1, 1]])
        return trail_line, pred_line, marker

    anim = FuncAnimation(fig, update, frames=n_total, init_func=init, blit=True)
    writer = FFMpegWriter(fps=fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(dataset_dir="dataset", fps=30):
    bbox_csv_path = None
    for candidate in ("bboxes_light.csv", "bbox_light.csv"):
        path = os.path.join(dataset_dir, candidate)
        if os.path.isfile(path):
            bbox_csv_path = path
            break
    if bbox_csv_path is None:
        raise FileNotFoundError(f"No bbox CSV found in {dataset_dir}")

    rows = load_bbox_csv(bbox_csv_path)
    print(f"Loaded {len(rows)} bbox rows from {bbox_csv_path}")

    frames = get_valid_frames(rows, dataset_dir)
    print(f"{len(frames)} frames have a valid bbox + matching xyz file")

    frames_data = []
    dropped_no_points = 0
    for entry in frames:
        xyz = load_xyz(entry["xyz_path"])
        u, v = light_center_pixel(entry)
        bbox_w = entry["x_max"] - entry["x_min"]
        bbox_h = entry["y_max"] - entry["y_min"]
        pos = estimate_light_position(xyz, u, v, bbox_w, bbox_h)
        if pos is None:
            dropped_no_points += 1
            continue
        frames_data.append((entry["frame_id"], pos[0], pos[1], pos[2]))

    print(f"{len(frames_data)} frames have a valid 3D light position "
          f"({dropped_no_points} dropped: no valid points in patch)")

    if len(frames_data) < 2:
        raise RuntimeError("Not enough valid frames to build a trajectory")

    x0, y0 = frames_data[0][1], frames_data[0][2]
    r0 = compute_reference_rotation(x0, y0)
    print(f"Reference frame: {frames_data[0][0]}, "
          f"light at camera-frame (X={x0:.2f}, Y={y0:.2f})")

    frame_ids, traj = compute_trajectory(frames_data, r0)

    print(f"Trajectory spans {len(traj)} points (frames {frame_ids[0]}-{frame_ids[-1]}), "
          f"X range [{traj[:,0].min():.2f}, {traj[:,0].max():.2f}] m, "
          f"Y range [{traj[:,1].min():.2f}, {traj[:,1].max():.2f}] m")

    traj_smooth = smooth_trajectory(traj)

    predicted = predict_path_to_origin(traj_smooth)
    if len(predicted):
        print(f"Extrapolated {len(predicted)} predicted points toward the origin "
              f"(closest approach {np.linalg.norm(predicted[-1]):.2f} m from the light)")
    else:
        print("No predicted extension: recent trend isn't heading toward the origin")

    plot_trajectory_png(traj_smooth, "trajectory.png", predicted=predicted)
    print("Wrote trajectory.png")

    animate_trajectory_mp4(traj_smooth, "trajectory.mp4", fps=fps, predicted=predicted)
    print("Wrote trajectory.mp4")


if __name__ == "__main__":
    main()
