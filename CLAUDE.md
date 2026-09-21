# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A submission for the Wisconsin Autonomous "Computer Vision Challenge: Ego-Trajectory & Bird's-Eye View Mapping" (see `instructions.md` for the full spec — that's the challenge prompt, not `README.md`; see below). The task: given a 10-second ego-vehicle video with per-frame RGB, per-pixel 3D point clouds, and traffic-light bounding boxes, reconstruct the ego-vehicle's ground-plane trajectory using the traffic light as a fixed world reference (Part A, required), and optionally place other tracked objects (golf cart, barrels, pedestrians) into a richer BEV scene (Part B, extra credit).

**Part A is implemented** in `solution.py` and has been run successfully against the current local dataset (see Results in `README.md`). Part B has not been attempted.

`PLAN.md` contains the detailed algorithm/implementation plan `solution.py` was built from — read it before modifying trajectory logic, since it documents the coordinate-frame math and the deliberate simplifying assumptions (e.g. constant vehicle heading, since no IMU/heading data is provided). `README.md` is the submission write-up (method/assumptions/results, max 1 page) — keep it in sync with `solution.py` if the algorithm changes.

## Commands

The project uses a local venv at `.venv` with `numpy` and `matplotlib` installed; there is no requirements file, build system, linter, or test suite yet.

```bash
# activate the venv
source .venv/bin/activate

# run the solution (writes trajectory.png and trajectory.mp4)
.venv/bin/python solution.py
```

## Architecture

**Data flow (per `PLAN.md`):** `bboxes_light.csv` (traffic-light bbox per frame) → bbox center pixel → lookup into that frame's `xyz/*.npz` point array → traffic-light position in **camera coordinates** → rotate into a **world frame** anchored at the traffic light → invert to get the ego-vehicle's `(x_m, y_m)` at that frame → accumulate across frames into the trajectory plotted in `trajectory.png` / `trajectory.mp4`.

**Two coordinate frames matter and must not be conflated:**
- **Camera frame** (per-frame, moves with the car): +X forward, +Y right, +Z up, origin at the camera (top of car, centered on vehicle width). This is what the `.npz` point clouds are expressed in.
- **World frame** (fixed, defined once): origin on the ground directly under the traffic light, Z through the light, and by definition the car–light line at t=0 lies along +X (X forward, Y left, Z up). The rotation from camera frame to this world frame is derived from the traffic light's apparent position at the reference frame (see `PLAN.md` §4) — there is no independently measured heading/IMU, so the plan explicitly assumes constant vehicle heading across the clip as a documented limitation.

**`dataset/` layout** — the local copy present does **not** match the naming/shape `instructions.md` describes, so `solution.py`'s loader handles both/is written defensively rather than hardcoding one convention:
- `instructions.md` spec: `dataset/rgb/frame_XXXX.png`, `dataset/xyz/frame_XXXX.npz` (key `"points"`, shape `(H, W, 3)` float32 meters), `dataset/bboxes_light.csv` with columns `frame_id, x_min, y_min, x_max, y_max`.
- Current local dataset instead uses `dataset/rgb/leftNNNNNN.png` (299 files, complete), `dataset/xyz/depthNNNNNN.npz` (198 of 299 files present — frames 0–37 have no depth yet), and `dataset/bbox_light.csv` with columns `frame,x1,y1,x2,y2` (299 rows, 4 degenerate/all-zero). The full dataset is downloaded separately from the Google Drive link in `instructions.md`.
- The `.npz` key is actually `"xyz"` (not `"points"`), shape `(1200, 1920, 4)` — the 4th channel is always `0`/`NaN`/`inf` and is ignored; only the first 3 channels (X, Y, Z in camera coords) are used. `load_xyz()` in `solution.py` picks up whichever key is present rather than hardcoding `"points"`.
- Degenerate/all-zero bbox rows (light not detected that frame) are filtered out in `get_valid_frames()` rather than treated as a valid detection.

**`.npz` indexing gotcha:** the point array is `(H, W, ...)`, so a pixel at `(u, v)` (x, y) is read as `xyz[v, u]`, not `xyz[u, v]`.

**Required deliverables** (per `instructions.md`): `trajectory.png`, `trajectory.mp4`, the code, and a submission `README.md` (max 1 page) describing method/assumptions/results. `README.md` at the repo root *is* that submission doc (there was no pre-existing top-level README to conflict with — the challenge prompt lives in `instructions.md`); keep it, not this file, as the place to update method/assumptions/results for the grader.
