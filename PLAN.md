# Implementation Plan — Part A: Ego-Vehicle Trajectory

This document plans the implementation of **Part A only** (required): estimating the ego-vehicle's ground-plane trajectory `(x_m, y_m)` using the traffic light as a fixed world reference. Part B (tracking additional objects) is intentionally out of scope here.

Code will live in `solution.py`. The full dataset isn't downloaded yet — the current `dataset/` folder is a partial subset (48 RGB frames vs. ~300 depth/CSV rows) — so this plan is written to be robust to whatever the full dataset's exact file naming/columns turn out to be, rather than hardcoded to the partial sample.

---

## 1. Dependencies / Setup

The project venv currently only has `numpy` installed. We need one more library:

- **`matplotlib`** — for both the static plot (`trajectory.png`) and the animation (`trajectory.mp4`, via `FuncAnimation` + `FFMpegWriter`).
- **`ffmpeg`** is already available on the system PATH, so `matplotlib` can encode the mp4 directly — no need for `opencv-python` or `imageio` as extra dependencies.
- The bbox CSV will be read with Python's stdlib `csv` module — `pandas` isn't necessary for a single small file.

Action: `pip install matplotlib` into `.venv`.

---

## 2. Pre-processing Pipeline

Before any geometry, the raw inputs need to be cleaned and aligned:

1. **Load `bboxes_light.csv`.** Read the header and normalize column names defensively (the partial sample uses `frame,x1,y1,x2,y2`; the README spec uses `frame_id,x_min,y_min,x_max,y_max` — the loader should accept either).
2. **Drop degenerate detections.** Any row where `x2 <= x1` or `y2 <= y1` (this covers the all-zero `0,0,0,0` rows seen in the sample) means the traffic light wasn't detected that frame. These frames are excluded from the trajectory rather than interpolated — the README explicitly allows a trajectory made of discrete, possibly-gappy points.
3. **Match frames to files.** For each remaining `frame_id`, locate the corresponding RGB and `.npz` files by globbing the `rgb/` and `xyz/` folders (not by assuming one exact filename pattern), since the full dataset's naming may differ from the partial sample's (`leftNNNNNN.png` / `depthNNNNNN.npz`). Skip any frame missing its `.npz` file — no depth means no 3D position.
4. **Sort chronologically.** Order the remaining frames numerically by frame id.
5. **Pick the reference frame `t0`.** This is the *first valid* frame in the filtered, sorted list — not necessarily literal frame index 0, since early frames could have an invalid detection.

---

## 3. Per-Frame Traffic-Light 3D Localization

For each valid frame `t`:

1. **Bounding-box center pixel:** `u = (x1+x2)/2`, `v = (y1+y2)/2`, rounded to the nearest integer and clipped to image bounds `[0, W-1] x [0, H-1]`.
2. **Depth lookup:** load `xyz = np.load(path)["points"]`, shape `(H, W, 3)`. Index as `xyz[v, u]` — **row is `v` (y), column is `u` (x)**. This axis order is a common source of bugs and is called out explicitly here.
3. **Patch averaging for noise robustness** (per the README's suggestion): instead of trusting a single pixel, sample a small window centered on `(u, v)`, sized relative to the bbox (e.g., a fraction of `min(bbox_width, bbox_height)`, capped so the window stays inside the light and doesn't pick up background).
4. **Validity filtering within the patch:** discard points that are `NaN`, exactly zero (sensor's invalid-value marker), or outside a plausible range (e.g., `Z <= 0` or distance implausibly large for this scene).
5. **Aggregate:** take the **median** (more outlier-robust than mean) of the valid points in the patch → `(X_t, Y_t, Z_t)`, the traffic light's position in **camera coordinates** at frame `t`.
6. If a frame's patch has zero valid points after filtering, drop that frame from the trajectory entirely (consistent with the "gappy trajectory is OK" guidance).

---

## 4. World-Frame Construction (Core Algorithm)

**Key simplifying assumption, stated explicitly:** no IMU or independent heading data is provided — only the light's position relative to the camera each frame. Part A therefore assumes the vehicle's **heading stays constant** across the 10-second clip (translation-only ego-motion; no yaw correction). This is the plan's main documented limitation, not a hidden shortcut.

Given that assumption:

1. **Reference bearing at `t0`:** compute `theta0 = atan2(Y_t0, X_t0)` — the angular offset between the camera's forward axis and the actual direction to the light at the reference frame.
2. **Fixed rotation `R0`:** build the 2×2 rotation matrix that rotates `(X_t0, Y_t0)` onto `(r0, 0)`. This operationalizes the README's world-frame definition: *"at t=0, the line joining the car and the traffic light is aligned with the +X axis."* Because this is a definitional choice about the world frame (not something we can independently measure), the rotation is derived directly from the data at `t0`.
3. **Apply `R0` to every valid frame:** `(Xw_t, Yw_t) = R0 @ (X_t, Y_t)` — the car→light vector in world-aligned axes, under the constant-heading assumption.
4. **Invert to get car position:** since the light's ground projection defines the world origin `(0, 0)`, and `(Xw_t, Yw_t)` is the vector *from* the car *to* the light, the car's world position is:
   ```
   (x_m, y_m) = -(Xw_t, Yw_t)
   ```
5. Height (`Z`) is dropped at this stage — the README only requires the ground-plane `(X, Y)` projection.

---

## 5. Outputs

- **`trajectory.png`** (required): static matplotlib plot of all valid `(x_m, y_m)` points. Mark the origin (traffic light) with a distinct marker, mark the start and end of the path, use equal aspect ratio, label axes in meters, add a grid/title.
- **`trajectory.mp4`** (required per submission section): `matplotlib.animation.FuncAnimation` + `FFMpegWriter`, progressively drawing the path over time (point-by-point, or a moving marker with a fading trail). Frame rate is currently assumed at ~30 fps (consistent with the ~300 frames over 10 seconds seen in the partial dataset) and should be confirmed once the full dataset arrives.

---

## 6. Suggested Code Structure (`solution.py`)

A small set of single-purpose functions:

- `load_bbox_csv(path)` — parse + normalize the CSV
- `get_valid_frames(rows, dataset_dir)` — filter degenerate rows, match files, sort
- `load_xyz(frame_id)` — load the `.npz` point array
- `light_center_pixel(row)` — bbox → `(u, v)`
- `estimate_light_position(xyz, u, v, bbox_w, bbox_h)` — patch + median → `(X, Y, Z)` or `None`
- `compute_reference_rotation(X0, Y0)` — → `R0`
- `compute_trajectory(frames_data, R0)` — → array of `(x_m, y_m)` + frame ids
- `plot_trajectory_png(traj, out_path)`
- `animate_trajectory_mp4(traj, out_path, fps)`
- `main()` — orchestrates the pipeline; takes the dataset directory as a parameter so it's trivial to point at the full dataset once downloaded

---

## 7. Assumptions & Limitations

- **Constant-heading assumption** is the biggest potential source of trajectory error — if the car turns significantly during the clip, this will distort the reconstructed path. No independent heading source is available to correct for this in Part A.
- **Patch size and outlier thresholds** are heuristic and will likely need tuning once the real, full-resolution data is visible.
- **Uniform-fps assumption** only affects mp4 playback timing, not the computed `(x, y)` values themselves.
- **Invalid frames are dropped, not interpolated** — by design, matching the README's tolerance for a discrete/gappy trajectory.

---

## 8. Verification Plan

Once the full dataset is available:

1. Run the script and check the printed count of valid vs. dropped frames (sanity check that most frames survive filtering).
2. Visually inspect `trajectory.png` for a plausible path shape (e.g., smooth-ish approach toward/past the origin, no wild jumps).
3. Manually print raw `(X, Y, Z)` at a couple of frames (e.g., first and middle) and confirm the values are in a plausible metric range for this scene before trusting the full plot.
4. Play `trajectory.mp4` to confirm the animation renders and timing looks reasonable.
