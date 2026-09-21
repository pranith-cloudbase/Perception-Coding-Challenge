# Submission: Ego-Trajectory Reconstruction (Part A)

Code: [`solution.py`](solution.py). Full algorithm write-up: [`PLAN.md`](PLAN.md).

## Method

The ego-vehicle's ground-plane trajectory is reconstructed by treating the fixed traffic
light as a world anchor and tracking its *apparent* motion in the camera frame:

1. **Load & clean detections.** Parse `bboxes_light.csv`, dropping degenerate rows
   (`x_max <= x_min` or `y_max <= y_min`) where the light wasn't detected, and match each
   remaining frame to its `.npz` depth file by frame id.
2. **Localize the light in 3D (camera frame).** For each frame, take the bbox center pixel
   `(u, v)`, sample a small patch around it (sized ~25% of the bbox's shorter side) in the
   `(H, W, 3)` point array (indexed `xyz[v, u]`, not `[u, v]`), discard invalid points
   (NaN, exactly zero, or non-positive forward distance), and take the **median** of the
   remaining points as a noise-robust `(X, Y, Z)` estimate. Frames with zero valid points
   are dropped.
3. **Build the world frame.** At the first valid frame `t0`, the car→light vector
   `(X_t0, Y_t0)` defines the world's `+X` axis by construction (per the challenge spec).
   A single 2×2 rotation `R0` is computed once from `theta0 = atan2(Y_t0, X_t0)` and applied
   to every frame's `(X, Y)`.
4. **Recover ego position.** Since the light sits at the world origin and `R0 @ (X, Y)` is
   the car→light vector in world axes, the car's position is simply the negation:
   `(x_m, y_m) = -R0 @ (X, Y)`.
5. **Render.** `trajectory.png` (static, matplotlib) and `trajectory.mp4` (animated,
   `FuncAnimation` + `FFMpegWriter`, drawn frame-by-frame at ~30 fps) plot the accumulated
   `(x_m, y_m)` points with the light, start, and end marked.

## Assumptions & Limitations

- **Constant heading (the main limitation).** No IMU or independent heading source is
  provided — only the light's position relative to the camera. `R0` is derived once from
  `t0` and reused for every frame, i.e. the vehicle's yaw is assumed constant across the
  clip. Any real turning shows up as apparent curvature/drift in the reconstructed path
  rather than being corrected for.
- **Gappy trajectory by design.** Frames with a degenerate bbox, missing depth file, or an
  empty valid patch are dropped outright rather than interpolated, per the challenge's
  tolerance for a discrete/possibly-gappy path.
- **Patch size and outlier thresholds are heuristic** (fixed fraction of bbox size, simple
  NaN/zero/sign filtering) and may need retuning on the full-resolution dataset.
- **Height is discarded** — only the ground-plane `(X, Y)` projection is reported, as
  required.

## Results

Run against the currently available local data (198 of 299 bbox rows have a matching depth
file — the full ~300-frame dataset was not fully downloaded at submission time):

- 198/198 frames with a matched depth file yielded a valid 3D light position (0 dropped for
  lack of valid patch points) — the patch-median filtering is well within the light's
  detection range for this data.
- Reference frame: frame 38 (first valid detection), light at camera-frame
  `(X=32.65 m, Y=15.38 m)`.
- Reconstructed path spans `X ∈ [-36.1, -7.1] m`, `Y ∈ [-0.3, 6.2] m` relative to the light
  at the origin — a smooth, gently arcing approach with no discontinuities or outlier
  jumps, consistent with the vehicle driving toward the light while the constant-heading
  assumption absorbs the car's real (mild) turning as path curvature.
- See `trajectory.png` / `trajectory.mp4` for the rendered output.

Part B (tracking the golf cart, barrels, and pedestrians into a richer BEV) was not
attempted in this submission.
