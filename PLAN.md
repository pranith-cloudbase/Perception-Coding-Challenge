# Implementation Plan / Design Notes — Part A: Ego-Vehicle Trajectory

Design document for `solution.py`. It covers **Part A only** (required): recovering the
ego-vehicle's ground-plane trajectory `(x_m, y_m)` using the traffic light as the fixed
world reference. Part B (placing other objects into a richer BEV) is out of scope.

`README.md` is the one-page submission write-up. This file is the longer "why", and is the
thing to read before changing any of the geometry.

---

## 1. Dependencies

`numpy` for everything numeric, `matplotlib` for `trajectory.png` / `diagnostics.png` and
for `trajectory.mp4` (`FuncAnimation` + `FFMpegWriter`, which shells out to the `ffmpeg`
already on PATH). The CSV is read with the stdlib `csv` module. Pinned in
`requirements.txt`; no OpenCV, SciPy or pandas, deliberately — every method below is a
handful of lines of NumPy, and keeping the dependency list short keeps the geometry
auditable.

```bash
pip install -r requirements.txt
python solution.py                 # writes trajectory.png, diagnostics.png, trajectory.mp4
python -m unittest -v test_solution
```

---

## 2. Pre-processing

1. **Find and parse the bbox CSV.** `find_bbox_csv` tolerates `bboxes_light.csv` (the
   spec's name) and `bbox_light.csv` (the shipped name). `load_bbox_csv` normalises either
   header spelling (`frame,x1,y1,x2,y2` or `frame_id,x_min,y_min,x_max,y_max`) onto one
   internal set of keys, and raises rather than guessing on an unknown column.
2. **Drop degenerate detections.** `x_max <= x_min` or `y_max <= y_min` means the light was
   not detected that frame (the shipped CSV carries 4 all-zero rows). These frames are
   dropped, not interpolated — the brief explicitly allows a discrete, gappy trajectory.
3. **Match frames to files by globbing**, not by assuming a filename template: the spec
   says `frame_XXXX.png` / `frame_XXXX.npz`, the shipped data uses `leftNNNNNN.png` /
   `depthNNNNNN.npz`. `frame_id_from_name` takes the **last** run of digits in the
   basename so a digit in the prefix can't be mistaken for the frame index.
4. **Skip frames with no `.npz`.** In the currently downloaded subset only 198 of 299
   frames have depth, so this is the single largest source of dropped frames.
5. **Sort by frame id**, and take the reference frame `t0` to be the first *surviving*
   frame — frame 38 here, not frame 0.

---

## 3. Per-frame 3D localisation of the light

For each surviving frame (`estimate_light_position`):

1. **Centre pixel** `u = (x_min+x_max)/2`, `v = (y_min+y_max)/2`, rounded and clipped to
   the image.
2. **Index as `xyz[v, u]`.** The array is `(H, W, 3)`, so the row index is `v`. This is the
   classic trap in this challenge and `test_index_order_is_row_major` pins it down by
   encoding the row index in one channel and the column index in another.
3. **Patch, not a single pixel** (as the brief suggests): the central 50 % of the bbox.
   One pixel on the light's dark housing is often a stereo hole.
4. **Validity mask**: finite, not exactly `(0,0,0)`, and `X > 0` (in front of the camera).
5. **MAD gate on range**, then a component-wise **median**. The gate is what removes the
   sky and background pixels that leak in around the light's silhouette — they sit at a
   completely different range from the light, so `|r - median(r)| > max(3·1.4826·MAD, 0.25 m)`
   discards them while leaving the light's own spread alone.
6. Frames with fewer than 8 surviving samples are dropped.

The `.npz` files store the array under key `"xyz"` with a 4th channel that only ever holds
`0`/`NaN`/`inf`; the spec says key `"points"` with 3 channels. `load_xyz` accepts either and
returns only X/Y/Z.

---

## 4. Axis conventions — measured, not assumed

The brief says the camera frame is "+X forward, +Y **right**, +Z up, right-handed". Those
cannot all be true at once: forward × right = *down*, so that triple is left-handed. The
world frame it then defines ("X forward, Y **left**, Z up") is the right-handed one.

So the sign of camera +Y is not something to assume. `detect_axis_convention` measures it
from the data: correlate channel 1 against the column index `u` and channel 2 against the
row index `v` over all valid pixels. In this dataset Y falls as `u` rises and Z falls as
`v` rises, i.e. the point clouds really are **+X forward, +Y left, +Z up** — right-handed,
and already matching the world frame's handedness. `to_ground_vector` flips Y only if the
measurement says the data is +Y-right, so the code stays correct either way.

Height is dropped at this point: only the ground projection is asked for. The measured
height is kept for validation (§7).

---

## 5. World frame and heading recovery

The world frame is fixed once: origin on the ground under the light, +Z up through it, and
**by definition** the car→light line at `t0` lies along +X. With the light at the origin and
`c_t` the car→light vector in camera coords,

```
p_t = -R(psi_t) @ c_t
```

where `psi_t` is the vehicle's world heading. `psi_0 = -atan2(c0_y, c0_x)` follows directly
from the frame definition. Everything then hinges on what we do about `psi_t` for `t > 0`.

> **Removed, on purpose.** An earlier version assumed `psi_t = psi_0` for all `t` — one
> rotation `R0` reused for every frame, i.e. the vehicle never yaws. On this clip that is not
> an approximation, it is wrong: the light's bearing sweeps 25° left to 5° right while its
> range drops 36 m to 8 m, which no straight-line drive can produce. Forcing a fixed heading
> onto it made the reconstruction slide 6 m sideways (15° median sideslip, 55° at p95). Do
> not reintroduce it.

### 5a. Non-holonomic yaw recovery (`solve_nonholonomic_yaw`)

There is no IMU in this dataset, but there is one piece of physics available for free:
**a car cannot drive sideways.** Its velocity is along its heading.

Treat the headings `psi_t` as the unknowns. `psi_0` is pinned by the frame definition, and
each `p_t` is a known function of `psi_t`. For a step along a constant-curvature arc the
chord direction is exactly the *mean* of the two end headings, which gives one scalar
equation per consecutive pair in one unknown:

```
cross( p_{t+1}(psi_{t+1}) - p_t ,  u((psi_t + psi_{t+1}) / 2) ) = 0
```

solved by bisection inside ±8° of yaw per step. The whole heading sequence falls out as a
forward recursion — no extra sensor, no extra dependency, no optimiser.

Two useful properties: on a straight drive it returns a constant heading by itself, so
nothing is assumed that the data does not support; and because the constraint is written
between *samples* rather than consecutive video frames, it is unaffected by the dataset's
frame gaps.

`test_solution.py` checks it against synthetic constant-curvature arcs with known
ground truth (left turn, right turn, and no turn) and recovers positions to ~1e-4 m.

### 5b. Smoothing

`local_linear_fit` does a local linear regression in a ±6-frame *time* window and evaluates
it at each sample's own timestamp. A boxcar average would be wrong twice over: it shrinks
curvature through the turn, and it assumes even spacing when frames 38–127 are sparsely
sampled. The same fit's slope gives the speed profile, which is far steadier than
differencing consecutive samples (at 30 fps, 1 cm of depth noise becomes 0.3 m/s).

---

## 6. Outputs

- **`trajectory.png`** (required) — BEV plot, equal aspect, metres, laid out to match the
  sample in `instructions.md`: framed axes, dashed grid, thin axis lines through the origin,
  an X at the start, a filled dot at the end, a star on the light labelled "Origin", and a
  boxed legend.
- **`trajectory.mp4`** (required) — the path drawn over time. Animated on the **source frame
  grid**, not the sample index, so playback pacing matches real time despite the gaps.
- **`diagnostics.png`** (extra) — range, bearing, speed and recovered yaw against time, plus
  the light-height check. This is the evidence for §7.

Colours are the first three slots of a categorical palette validated for
colour-vision-deficiency separation, in place of the sample's red/green start/end pair. The
marker shapes (X, dot, star) already carry the distinction, so the swap costs nothing and
helps readers who cannot separate red from green.

**On orientation:** the sample's trajectory runs vertically, because its ego vehicle
approaches along the plot's lateral axis. Ours runs horizontally, and that is the spec being
followed rather than a styling difference: `instructions.md` defines the world frame so that
*"at t = 0, the line joining the car and the traffic light is aligned with the +X axis"*,
which puts the car on the -X axis at `t0` and the light dead ahead along +X. Rotating the
plot to match the sample would mean abandoning the frame definition the result is graded
against.

---

## 7. Validation

Nothing here has ground truth, so the checks are internal-consistency ones:

1. **Light height must be constant.** A fixed overhead light cannot change height. Measured:
   median 3.59 m, sd 0.14 m over 198 frames. This is the strongest independent check that
   the patch estimator is locking onto the light and not onto background.
   `reject_height_outliers` also drops any frame that fails it.
2. **Solver residual.** `sideslip_angles` measures the angle between the travelled chord
   and the vehicle heading — exactly the constraint §5a solves. A step where bisection finds
   no sign change falls back to holding the heading, and that failure would show up here, so
   this is a convergence check rather than decoration. Measured: **p95 0.000°**, i.e. every
   step on the real data was bracketed and solved.
3. **The recovered turn is visible in the RGB.** At frame 38 the road bends left ahead and
   the light cluster sits at the far left of the image; by frame 298 the car faces the
   intersection square-on. A +44° left turn is what the footage shows.
4. **Bbox continuity.** The bbox centre moves smoothly and its width grows monotonically
   20 → 88 px, so the CSV tracks one light throughout — no identity switch between the
   several lights on the span wire.
5. **Speed plausibility.** 3.4 m/s mean (7.6 mph), decelerating to a near-stop 8.2 m short
   of the light. Consistent with the video, in which the car pulls up behind a golf cart.
6. **Unit tests** (`test_solution.py`, 24 cases, no dataset needed) cover CSV parsing, frame
   matching, the `[v, u]` index order, outlier rejection, the smoother, and closed-form
   recovery of synthetic arcs.

---

## 8. Assumptions and limitations

- **Planar motion.** The ground is assumed locally flat and the camera's pitch/roll constant;
  only yaw is estimated. Height is dropped from the trajectory, as the brief allows.
- **Constant curvature within a step.** The non-holonomic constraint uses the mean-heading
  chord rule, exact for a circular arc and a very good approximation at 30 fps.
- **Range-dependent stereo noise.** Depth error grows roughly with range², so the earliest
  samples (~36 m out) are the least reliable. This is visible as the speed spike in the
  first ~0.3 s of `diagnostics.png`, where the sparse sampling gives the smoother little to
  work with. The estimate is left in rather than trimmed — it is a real measurement, and
  its uncertainty is part of the result.
- **Sparse early frames.** Only 198 of 299 frames have a `.npz` in the downloaded subset,
  and frames 38–127 are sparse. Re-running against the complete dataset needs no code
  change.
- **Dropped, not interpolated.** Frames that fail any filter are simply absent.
- **Uniform 30 fps** affects the speed/time axes and mp4 pacing, not the `(x, y)` geometry.
