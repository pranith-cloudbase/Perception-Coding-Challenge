# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A submission for the Wisconsin Autonomous "Computer Vision Challenge: Ego-Trajectory & Bird's-Eye View Mapping" (see `instructions.md` for the full spec — that's the challenge prompt, not `README.md`; see below). The task: given a 10-second ego-vehicle video with per-frame RGB, per-pixel 3D point clouds, and traffic-light bounding boxes, reconstruct the ego-vehicle's ground-plane trajectory using the traffic light as a fixed world reference (Part A, required), and optionally place other tracked objects (golf cart, barrels, pedestrians) into a richer BEV scene (Part B, extra credit).

**Both parts are implemented and run against the current local dataset**: Part A in `solution.py`, Part B in `part_b.py` (see Results in `README.md`).

`PLAN.md` is the detailed design document for both — sections 1-8 cover Part A, section 9 covers Part B. Read it before modifying trajectory or detection logic, since it documents the coordinate-frame math, the measured stereo-error law behind Part B's range gates, and the detectors that were tried and rejected (with the reasons, which are properties of the scene rather than bugs). `README.md` is the submission write-up (method/assumptions/results, max 1 page) — keep it in sync if either algorithm changes.

## Commands

The project uses a local venv at `.venv`; dependencies are pinned in `requirements.txt` (`numpy`, `matplotlib` — deliberately no OpenCV/SciPy/pandas). There is no build system or linter.

```bash
# Part A (writes trajectory.png, trajectory.mp4, diagnostics.png)
.venv/bin/python solution.py

# Part B (writes bev.png, bev.mp4, detections.png)
.venv/bin/python part_b.py

# useful flags
.venv/bin/python solution.py --no-video --dataset dataset --smooth-frames 6
.venv/bin/python part_b.py --no-video --no-overlay --refresh

# tests: 60 synthetic cases across both suites, no dataset needed, ~0.2 s
.venv/bin/python -m unittest -v test_solution test_part_b
```

Part B caches its per-frame scan in `part_b_cache.npz` (gitignored, ~17 MB) and reuses it
unless `--refresh` is passed; a cold scan is ~90 s, a re-render from cache ~20 s. Only the
raw scan is cached — `barrel_objects` is recomputed on load, so re-tuning the locator's
gates does not require re-scanning imagery.

## Architecture

**Data flow (per `PLAN.md`):** `bboxes_light.csv` (traffic-light bbox per frame) → bbox center pixel → lookup into that frame's `xyz/*.npz` point array → traffic-light position in **camera coordinates** → per-frame vehicle heading → ego-vehicle `(x_m, y_m)` in the **world frame** anchored at the traffic light → `trajectory.png` / `trajectory.mp4` / `diagnostics.png`.

**Two coordinate frames matter and must not be conflated:**
- **Camera frame** (per-frame, moves with the car): +X forward, +Z up, origin at the camera (top of car, centered on vehicle width). This is what the `.npz` point clouds are expressed in. **The sign of +Y is measured from the data, not assumed** — `instructions.md` says "+Y right, right-handed", which is self-contradictory (forward × right = down), and this dataset is in fact **+Y left**. `detect_axis_convention()` correlates channel 1 against the column index and channel 2 against the row index to read it off; `to_ground_vector()` flips Y only if a dataset turns out to be +Y-right.
- **World frame** (fixed, defined once): origin on the ground directly under the traffic light, Z through the light, and by definition the car–light line at the reference frame lies along +X (X forward, Y left, Z up). With the light at the origin, the ego position is `p_t = -R(psi_t) @ c_t`.

**Heading estimation (`PLAN.md` §5).** There is no IMU, so `solve_nonholonomic_yaw()` uses the one free piece of physics: a car cannot drive sideways. With headings as unknowns (`psi_0` pinned by the world-frame definition), the constraint "the chord `p_{t+1} - p_t` points along the mean of the two headings" gives one scalar equation per step in one unknown, solved by bisection as a forward recursion. Recovers the clip's real +44° left turn, and returns a constant heading by itself on a straight drive.

**Do not reintroduce the constant-heading model.** An earlier version assumed a single fixed rotation `R0` for the whole clip. It was removed as inaccurate, not simplified away: on this clip the light's bearing sweeps 25° left to 5° right while its range drops 36 m to 8 m, which no straight-line drive can produce, and forcing a fixed heading made the reconstruction slide 6 m sideways (15° median sideslip, 55° at p95). `sideslip_angles()` survives that removal as a *solver convergence check* — a step bisection failed to bracket falls back to holding the heading and shows up as a non-zero p95 residual.

**`dataset/` layout** — the local copy present does **not** match the naming/shape `instructions.md` describes, so `solution.py`'s loader handles both/is written defensively rather than hardcoding one convention:
- `instructions.md` spec: `dataset/rgb/frame_XXXX.png`, `dataset/xyz/frame_XXXX.npz` (key `"points"`, shape `(H, W, 3)` float32 meters), `dataset/bboxes_light.csv` with columns `frame_id, x_min, y_min, x_max, y_max`.
- Current local dataset instead uses `dataset/rgb/leftNNNNNN.png` (299 files, complete), `dataset/xyz/depthNNNNNN.npz` (198 of 299 files present — frames 0–37 are missing entirely, 38–126 are sparse, and 127–298 are complete apart from frame 141), and `dataset/bbox_light.csv` with columns `frame,x1,y1,x2,y2` (299 rows, 4 degenerate/all-zero). The full dataset is downloaded separately from the Google Drive link in `instructions.md`.
- The `.npz` key is actually `"xyz"` (not `"points"`), shape `(1200, 1920, 4)` — the 4th channel is always `0`/`NaN`/`inf` and is ignored; only the first 3 channels (X, Y, Z in camera coords) are used. `load_xyz()` in `solution.py` picks up whichever key is present rather than hardcoding `"points"`.
- Degenerate/all-zero bbox rows (light not detected that frame) are filtered out in `get_valid_frames()` rather than treated as a valid detection.

**Part B (`part_b.py`, `PLAN.md` §9).** Consumes Part A's pose: `to_world()` maps any camera-frame ground vector into the world frame as `p_t + R(psi_t) @ v`, and feeding it the car→light vector must return `(0, 0)` — that round-trip is the consistency condition tying the two files together and `test_part_b.py` pins it. Four detectors, all colour-threshold + depth-geometry, no learned model:
- `barrel_points()` — orange, **height-gated 0.15–1.8 m above the road**. The height gate is not tidying: the traffic-light *housings* are the same amber and would be mapped as road furniture, but hang at 5.4 m.
- `detect_golf_cart()` — the pale tan canopy, scored big-and-near with a 6 m gate against the previous detection. **Do not replace this with a driving-corridor detector** ("nearest obstacle within ±2.5 m of the camera axis"): it was tried and fails, because the ego turns 44° and a fixed camera-frame corridor loses the cart in the first third where it sits 25° off-axis.
- `detect_pedestrians()` — deliberately the weak layer, drawn in neutral ink and labelled low-confidence. Torso-band clustering and background-standoff were both tried and both fail for scene reasons documented in the docstring and `PLAN.md` §9.5; the surviving cue (both workers wear blue) is clip-specific in a way the other detectors are not. Do not promote it to a confident layer.
- `traffic_light_state()` — isolates the lit lamp on **brightness before hue** (V ≈ 0.85 vs the amber housing's 0.56). Hue-first gets it backwards, since the housing fills more of the bbox than the lamp.

**`locate_static_objects()` is peak-picking, not connected components,** and that is deliberate. Each accumulated object is a dense head with a faint radial tail pointing away from wherever the car stood; neighbouring tails touch, so components swallow a whole barrel row into one blob while density maxima stay one per object. The non-max suppression pass exists for exact ties (integer counts make plateaus real), which is what `test_tied_adjacent_cells_are_suppressed_to_one_peak` covers.

**Part B's range gates are measured, not tuned** (`PLAN.md` §9.3): re-observing one isolated barrel gives per-frame centroid scatter of 0.10 m at 9–12 m, ~0.30 m to 22 m, 0.58 m at 22–30 m and 0.81 m beyond — stereo error growing with range² and spent along the viewing ray. Hence `OBJECT_RANGE_GATE = 20 m` and `PED_RANGE_GATE = 12 m`. The faint cloud is still drawn at every range on purpose; the streaks are honest evidence of that error.

**`.npz` indexing gotcha:** the point array is `(H, W, ...)`, so a pixel at `(u, v)` (x, y) is read as `xyz[v, u]`, not `xyz[u, v]`.

**Validation, since there is no ground truth** (`PLAN.md` §7, §9.7): the light's measured height must stay constant (3.59 m ± 0.14 m here — the strongest check that the patch estimator locks onto the light, and `reject_height_outliers()` enforces it); sideslip must stay near zero for a real car; the bbox must track one light throughout. Part B adds the check with the most teeth, and it is a check on **Part A**: a barrel cannot move, so `static_consistency()` measures the spread of its *per-frame* centroids (which isolates trajectory drift from object size) — median 0.39 m, p90 0.57 m here. `detections.png` projects every mask back onto the RGB and is the first figure to check when a BEV marker looks wrong. `test_solution.py` recovers synthetic constant-curvature arcs in closed form and pins the `xyz[v, u]` index order; `test_part_b.py` pins the world-transform round-trip and every detector gate (the suite was mutation-tested — breaking a gate must fail a test).

**Required deliverables** (per `instructions.md`): `trajectory.png`, `trajectory.mp4`, the code, and a submission `README.md` (max 1 page) describing method/assumptions/results. `diagnostics.png`, `bev.png`, `bev.mp4` and `detections.png` are extra plots/videos (submission requirement #4 invites them). `README.md` at the repo root *is* that submission doc (there was no pre-existing top-level README to conflict with — the challenge prompt lives in `instructions.md`); keep it, not this file, as the place to update method/assumptions/results for the grader.
