# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A submission for the Wisconsin Autonomous "Computer Vision Challenge: Ego-Trajectory & Bird's-Eye View Mapping" (see `instructions.md` for the full spec — that's the challenge prompt, not `README.md`; see below). The task: given a 10-second ego-vehicle video with per-frame RGB, per-pixel 3D point clouds, and traffic-light bounding boxes, reconstruct the ego-vehicle's ground-plane trajectory using the traffic light as a fixed world reference (Part A, required), and optionally place other tracked objects (golf cart, barrels, pedestrians) into a richer BEV scene (Part B, extra credit).

**Part A is implemented** in `solution.py` and has been run successfully against the current local dataset (see Results in `README.md`). Part B has not been attempted.

`PLAN.md` contains the detailed algorithm/implementation plan `solution.py` was built from — read it before modifying trajectory logic, since it documents the coordinate-frame math and the deliberate simplifying assumptions (e.g. constant vehicle heading, since no IMU/heading data is provided). `README.md` is the submission write-up (method/assumptions/results, max 1 page) — keep it in sync with `solution.py` if the algorithm changes.

## Commands

The project uses a local venv at `.venv`; dependencies are pinned in `requirements.txt` (`numpy`, `matplotlib` — deliberately no OpenCV/SciPy/pandas). There is no build system or linter.

```bash
# run the solution (writes trajectory.png, trajectory.mp4, diagnostics.png)
.venv/bin/python solution.py

# useful flags
.venv/bin/python solution.py --no-video --method constant-heading --dataset dataset

# tests: 25 synthetic cases, no dataset needed, ~0.2 s
.venv/bin/python -m unittest -v test_solution
```

## Architecture

**Data flow (per `PLAN.md`):** `bboxes_light.csv` (traffic-light bbox per frame) → bbox center pixel → lookup into that frame's `xyz/*.npz` point array → traffic-light position in **camera coordinates** → per-frame vehicle heading → ego-vehicle `(x_m, y_m)` in the **world frame** anchored at the traffic light → `trajectory.png` / `trajectory.mp4` / `diagnostics.png`.

**Two coordinate frames matter and must not be conflated:**
- **Camera frame** (per-frame, moves with the car): +X forward, +Z up, origin at the camera (top of car, centered on vehicle width). This is what the `.npz` point clouds are expressed in. **The sign of +Y is measured from the data, not assumed** — `instructions.md` says "+Y right, right-handed", which is self-contradictory (forward × right = down), and this dataset is in fact **+Y left**. `detect_axis_convention()` correlates channel 1 against the column index and channel 2 against the row index to read it off; `to_ground_vector()` flips Y only if a dataset turns out to be +Y-right.
- **World frame** (fixed, defined once): origin on the ground directly under the traffic light, Z through the light, and by definition the car–light line at the reference frame lies along +X (X forward, Y left, Z up). With the light at the origin, the ego position is `p_t = -R(psi_t) @ c_t`.

**Heading estimation (`PLAN.md` §5).** There is no IMU, so `solve_nonholonomic_yaw()` uses the one free piece of physics: a car cannot drive sideways. With headings as unknowns (`psi_0` pinned by the world-frame definition), the constraint "the chord `p_{t+1} - p_t` points along the mean of the two headings" gives one scalar equation per step in one unknown, solved by bisection as a forward recursion. Recovers the clip's real +44° left turn, and returns a constant heading by itself on a straight drive.

**Do not reintroduce the constant-heading model.** An earlier version assumed a single fixed rotation `R0` for the whole clip. It was removed as inaccurate, not simplified away: on this clip the light's bearing sweeps 25° left to 5° right while its range drops 36 m to 8 m, which no straight-line drive can produce, and forcing a fixed heading made the reconstruction slide 6 m sideways (15° median sideslip, 55° at p95). `sideslip_angles()` survives that removal as a *solver convergence check* — a step bisection failed to bracket falls back to holding the heading and shows up as a non-zero p95 residual.

**`dataset/` layout** — the local copy present does **not** match the naming/shape `instructions.md` describes, so `solution.py`'s loader handles both/is written defensively rather than hardcoding one convention:
- `instructions.md` spec: `dataset/rgb/frame_XXXX.png`, `dataset/xyz/frame_XXXX.npz` (key `"points"`, shape `(H, W, 3)` float32 meters), `dataset/bboxes_light.csv` with columns `frame_id, x_min, y_min, x_max, y_max`.
- Current local dataset instead uses `dataset/rgb/leftNNNNNN.png` (299 files, complete), `dataset/xyz/depthNNNNNN.npz` (198 of 299 files present — frames 0–37 have no depth yet), and `dataset/bbox_light.csv` with columns `frame,x1,y1,x2,y2` (299 rows, 4 degenerate/all-zero). The full dataset is downloaded separately from the Google Drive link in `instructions.md`.
- The `.npz` key is actually `"xyz"` (not `"points"`), shape `(1200, 1920, 4)` — the 4th channel is always `0`/`NaN`/`inf` and is ignored; only the first 3 channels (X, Y, Z in camera coords) are used. `load_xyz()` in `solution.py` picks up whichever key is present rather than hardcoding `"points"`.
- Degenerate/all-zero bbox rows (light not detected that frame) are filtered out in `get_valid_frames()` rather than treated as a valid detection.

**`.npz` indexing gotcha:** the point array is `(H, W, ...)`, so a pixel at `(u, v)` (x, y) is read as `xyz[v, u]`, not `xyz[u, v]`.

**Validation, since there is no ground truth** (`PLAN.md` §7): the light's measured height must stay constant (3.59 m ± 0.14 m here — the strongest check that the patch estimator locks onto the light, and `reject_height_outliers()` enforces it); sideslip must stay near zero for a real car; the bbox must track one light throughout. `test_solution.py` recovers synthetic constant-curvature arcs in closed form and pins the `xyz[v, u]` index order.

**Required deliverables** (per `instructions.md`): `trajectory.png`, `trajectory.mp4`, the code, and a submission `README.md` (max 1 page) describing method/assumptions/results. `diagnostics.png` is an extra plot (the brief invites them). `README.md` at the repo root *is* that submission doc (there was no pre-existing top-level README to conflict with — the challenge prompt lives in `instructions.md`); keep it, not this file, as the place to update method/assumptions/results for the grader.
