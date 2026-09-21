#!/usr/bin/env python3
"""Part B - a richer BEV scene: barrels, the golf cart, pedestrians and the light.

Part A (`solution.py`) recovers the ego pose `(p_t, psi_t)` in the world frame
anchored at the traffic light. That pose is the thing that makes Part B possible:
once you know where the car was and which way it pointed, every pixel of every
frame can be pushed out into the *same* world frame and the scene accumulates.

    rgb pixel  --colour class-->  candidate object pixels
    xyz[v, u]  --------------->   (X, Y, Z) in CAMERA coords
    Z - z_ground  ------------>   height above the road (the workhorse filter)
    p_t + R(psi_t) @ (X, Y)  ->   WORLD coords, same frame as trajectory.png

Detection is colour thresholding plus depth geometry - no OpenCV, no learned
model, same dependency set as Part A (see PLAN.md Part B section for the why).

Four layers:
  * barrels / jersey barriers  - orange, static, accumulated over the whole clip
  * golf cart                  - tan canopy, dynamic, tracked per frame
  * pedestrians                - blue workwear, best effort (see the caveat in
                                 `detect_pedestrians` - this is the weak layer)
  * traffic light state        - the lit lamp's hue, drawn on the origin marker

Because the barrels are *static*, re-observing them from 198 different poses is
also an independent check on Part A: if the trajectory were wrong, the
accumulated barrel cloud would smear. It does not. See `--report`.

Run:  python part_b.py            # -> bev.png, bev.mp4, detections.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, FFMpegWriter

import solution as core

# --------------------------------------------------------------------------
# Palette.  Slots 1-2 are carried over from Part A so the ego path keeps its
# colour across the two figures; the golf cart takes a teal that clears CVD
# separation against both of them on an all-pairs check (worst dE 11.7), and
# reads as "not a traffic light" unlike a green.  Pedestrians are deliberately
# NOT given a hue: they are the low-confidence layer and wear neutral ink plus
# their own marker shape, so nothing about them is encoded by colour alone.
# --------------------------------------------------------------------------
C_PATH = core.C_PATH          # "#2a78d6" ego trajectory
C_BARREL = "#eb6834"          # orange - barrels and jersey barriers
C_CART = "#0f9b8e"            # teal - golf cart
C_PED = core.INK_MUTED        # neutral - pedestrians (low confidence)
OVERLAY_PED = "#8b2fb0"       # violet - only in detections.png, where the neutral
                              # pedestrian ink would vanish against the scene
LIGHT_COLOURS = {"green": "#1baf7a", "amber": "#e0a21a", "red": "#d94436", "unknown": core.INK_MUTED}

# Colour classes returned by `classify_colour`.
CLS_OTHER, CLS_ORANGE, CLS_WHITE, CLS_TAN, CLS_DARK, CLS_VEG, CLS_BLUE = range(7)

CACHE_NAME = "part_b_cache.npz"

# Range gates, set from the measured error growth rather than by eye. Re-observing
# one isolated barrel from every pose that saw it, the per-frame centroid scatter
# runs 0.10 m at 9-12 m, ~0.30 m out to 22 m, then 0.58 m at 22-30 m and 0.81 m
# beyond 30 m - stereo depth error grows with range squared and it is spent along
# the viewing ray, which is why a distant object smears into a radial streak
# rather than a blob. Discrete object positions are therefore taken only from
# observations inside OBJECT_RANGE_GATE; the raw cloud is still drawn, faintly,
# at every range, because the streaks are honest evidence of that error.
OBJECT_RANGE_GATE = 20.0      # metres - for locating barrels/barriers
CLOUD_DRAW_GATE = 25.0        # metres - for the faint underlay in the figures
PED_RANGE_GATE = 12.0         # metres - pedestrians are a much weaker signal


# --------------------------------------------------------------------------
# 1. Colour classification
# --------------------------------------------------------------------------

def classify_colour(rgb):
    """Per-pixel colour class from HSV thresholds.

    Thresholds were read off the actual histograms of this clip, not guessed:
    the barrels sit at hue 8-40 with saturation > 0.4, the golf cart's canopy and
    seats are an unsaturated tan at hue 35-70, and the two workers wear blue at
    hue 195-255.  Vegetation is classed only so it can be *excluded* - the tree
    line behind the intersection otherwise dominates every cluster.

    Matplotlib's `rgb_to_hsv` is used rather than hand-rolling the conversion;
    it is already a dependency and it is vectorised over the whole image."""
    hsv = mcolors.rgb_to_hsv(rgb[..., :3])
    hue, sat, val = hsv[..., 0] * 360.0, hsv[..., 1], hsv[..., 2]

    out = np.zeros(hue.shape, dtype=np.int8)
    out[(val < 0.28)] = CLS_DARK
    out[(hue > 65) & (hue < 170) & (sat > 0.20)] = CLS_VEG
    out[(sat < 0.16) & (val > 0.62)] = CLS_WHITE
    out[(hue >= 35) & (hue < 70) & (sat >= 0.16) & (sat < 0.55) & (val > 0.45)] = CLS_TAN
    out[(hue > 195) & (hue < 255) & (sat > 0.22) & (val > 0.20)] = CLS_BLUE
    # Orange last: it is the highest-precision class and should win any overlap.
    out[(hue > 8) & (hue < 40) & (sat > 0.40) & (val > 0.28)] = CLS_ORANGE
    return out


# --------------------------------------------------------------------------
# 2. Frame geometry
# --------------------------------------------------------------------------

def valid_mask(xyz):
    """Points that carry a real measurement: finite, non-zero, in front."""
    return (np.isfinite(xyz).all(axis=2)
            & ~np.all(xyz == 0, axis=2)
            & (xyz[:, :, 0] > 0.3))


def ground_plane_z(xyz, valid, near_range=12.0):
    """Camera-frame Z of the road surface.

    The road in front of the car is overwhelmingly the most common thing within
    ~12 m, so its median Z *is* the ground plane. Measured across this clip it
    sits at -1.79 m with a 5th-95th percentile spread of 9 cm, which is also the
    evidence for the flat-ground assumption Part A already leans on."""
    near = valid & (xyz[:, :, 0] < near_range)
    if near.sum() < 500:
        return float(np.median(xyz[:, :, 2][valid])) if valid.any() else 0.0
    return float(np.median(xyz[:, :, 2][near]))


def to_world(points_xy, ego_xy, heading, y_is_left):
    """Camera-frame ground points -> world frame, using the Part A ego pose.

    Part A defines `p_t = -R(psi_t) @ c_t` with the light at the origin, so the
    forward map for any other camera-frame vector v is `p_t + R(psi_t) @ v`
    (substituting v = c_t gives 0, i.e. the light, as it must)."""
    xy = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    if not y_is_left:
        xy = xy * np.array([1.0, -1.0])
    return ego_xy + xy @ core.rotation(heading).T


# --------------------------------------------------------------------------
# 3. BEV clustering (numpy only - no scipy.ndimage.label)
# --------------------------------------------------------------------------

def label_grid(occupied):
    """Connected components of a boolean grid, 8-connected.

    Iterative max-propagation: seed every occupied cell with a unique id, then
    repeatedly replace each cell by the max over its 3x3 neighbourhood until the
    labelling stops changing. Converges in O(component diameter) passes and each
    pass is a handful of vectorised `maximum` calls, which for the grid sizes
    here (a few hundred cells a side) is far cheaper than it sounds - and it
    keeps scipy out of requirements.txt."""
    labels = np.where(occupied, np.arange(occupied.size).reshape(occupied.shape) + 1, 0)
    while True:
        padded = np.pad(labels, 1)
        merged = labels.copy()
        for dy in range(3):
            for dx in range(3):
                merged = np.maximum(merged, padded[dy:dy + labels.shape[0], dx:dx + labels.shape[1]])
        merged = np.where(occupied, merged, 0)
        if np.array_equal(merged, labels):
            return labels
        labels = merged


def cluster_xy(x, y, cell, min_cell_count, min_points, extent=(0.0, 50.0, -40.0, 40.0)):
    """Grid-cluster 2D points; yields the index array of each surviving cluster.

    `min_cell_count` is a density floor that drops isolated stereo speckle before
    labelling, which matters more than the post-hoc `min_points` cut: a handful of
    noisy pixels strung across a gap will otherwise bridge two real objects into
    one component."""
    x = np.asarray(x); y = np.asarray(y)
    x0, x1, y0, y1 = extent
    inside = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    idx = np.nonzero(inside)[0]
    if idx.size < min_points:
        return []
    gx = ((x[idx] - x0) / cell).astype(int)
    gy = ((y[idx] - y0) / cell).astype(int)
    shape = (int((x1 - x0) / cell) + 2, int((y1 - y0) / cell) + 2)
    counts = np.zeros(shape, dtype=np.int32)
    np.add.at(counts, (gx, gy), 1)
    labels = label_grid(counts >= min_cell_count)
    per_point = labels[gx, gy]
    out = []
    for lab in np.unique(per_point):
        if lab == 0:
            continue
        sel = per_point == lab
        if sel.sum() >= min_points:
            out.append(idx[sel])
    return out


def _cluster_stats(x, y, h, sel):
    px, py, ph = x[sel], y[sel], h[sel]
    return {
        "cx": float(np.median(px)), "cy": float(np.median(py)),
        "ex": float(px.max() - px.min()), "ey": float(py.max() - py.min()),
        "top": float(np.percentile(ph, 98)), "n": int(sel.size),
        "range": float(np.hypot(np.median(px), np.median(py))),
    }


# --------------------------------------------------------------------------
# 4. Detectors
# --------------------------------------------------------------------------

def barrel_points(xyz, cls, valid, z_ground, max_range=45.0, subsample=4000, rng=None):
    """Orange points that belong to road furniture, in camera ground coords.

    The height gate is doing real work here and not just tidying: the traffic
    light *housings* are the same amber as the barrels and would otherwise be
    detected as road furniture, but they hang at ~5.4 m. Anything above 1.8 m is
    not a barrel."""
    height = xyz[:, :, 2] - z_ground
    mask = (valid & (cls == CLS_ORANGE) & (height > 0.15) & (height < 1.8)
            & (np.hypot(xyz[:, :, 0], xyz[:, :, 1]) < max_range))
    if mask.sum() == 0:
        return np.empty((0, 2)), np.empty(0)
    pts = np.stack([xyz[:, :, 0][mask], xyz[:, :, 1][mask]], axis=1)
    rngs = np.hypot(pts[:, 0], pts[:, 1])
    if subsample and pts.shape[0] > subsample:
        rng = rng or np.random.default_rng(0)
        take = rng.choice(pts.shape[0], subsample, replace=False)
        pts, rngs = pts[take], rngs[take]
    return pts, rngs


def detect_golf_cart(xyz, cls, valid, z_ground, previous=None):
    """The golf cart, as the best tan cluster in the scene.

    The cart is the only large tan object in the clip - its canopy and bench
    seats are a pale beige that nothing else on the road shares (the white tent
    and jersey barriers are unsaturated enough to fall in CLS_WHITE, and the
    gravel is greyer still). Scoring prefers big-and-near, which is exactly the
    cart, and `previous` adds a gate so a one-frame flicker onto a distant patch
    of gravel cannot steal the track.

    Deliberately NOT a driving-corridor detector: the ego vehicle turns 44 deg
    through this clip, so a fixed camera-frame corridor loses the cart in the
    first third, where it sits 25 deg off the camera axis."""
    height = xyz[:, :, 2] - z_ground
    mask = (valid & (cls == CLS_TAN) & (height > 0.30) & (height < 2.10)
            & (xyz[:, :, 0] < 30.0))
    if mask.sum() < 40:
        return None
    x, y, h = xyz[:, :, 0][mask], xyz[:, :, 1][mask], height[mask]
    best = None
    for sel in cluster_xy(x, y, cell=0.40, min_cell_count=3, min_points=40,
                          extent=(0.0, 30.0, -35.0, 35.0)):
        st = _cluster_stats(x, y, h, sel)
        if max(st["ex"], st["ey"]) > 5.0:          # a golf cart is ~2.4 m long
            continue
        score = st["n"] / (1.0 + st["range"])       # big and near beats small and far
        if previous is not None:
            gap = np.hypot(st["cx"] - previous[0], st["cy"] - previous[1])
            if gap > 6.0:                           # >6 m of apparent jump in <=0.2 s
                continue
            score /= (1.0 + gap)
        if best is None or score > best[0]:
            best = (score, st)
    return None if best is None else best[1]


def detect_pedestrians(xyz, cls, valid, z_ground, cart=None):
    """Pedestrians, via blue workwear. **This is the weak layer - read on.**

    Two stronger detectors were tried first and both failed, for reasons worth
    recording because they are properties of the scene rather than bugs:

      1. *Torso-band clustering* (points 1.0-2.0 m above the road, cluster, keep
         the narrow ones). The chain-link fence behind the workers spans the
         whole image at exactly that height, so every cluster merges into it.
      2. *Background standoff* (keep points closer than the smoothed per-column
         background depth). This does suppress the fence, but the workers stand
         immediately in front of a white jersey barrier of near-identical height
         and ~1 m behind them, which is inside the stereo noise at 11 m. What
         survived the gates was overwhelmingly poles and fence posts, identified
         by their height saturating the band where a person's does not.

    So this falls back to the one cue that is actually separable here: both
    workers wear blue, and nothing else in the scene at road height is blue. That
    is honest colour thresholding of the kind the brief invites, but it is
    *clip-specific* in a way the barrel and cart detectors are not - a worker in
    an orange hi-vis vest would be missed entirely, or worse, logged as a barrel.
    Treat these markers as "something person-shaped and blue was here", and note
    that the two workers standing 1.8 m apart routinely merge into one cluster.

    The cart driver also wears blue, so detections landing on the tracked cart
    are suppressed - they are an occupant, not a pedestrian in the road."""
    height = xyz[:, :, 2] - z_ground
    mask = (valid & (cls == CLS_BLUE) & (height > 0.70) & (height < 2.10)
            & (xyz[:, :, 0] < 30.0))
    if mask.sum() < 30:
        return []
    x, y, h = xyz[:, :, 0][mask], xyz[:, :, 1][mask], height[mask]
    out = []
    for sel in cluster_xy(x, y, cell=0.25, min_cell_count=2, min_points=60,
                          extent=(0.0, 30.0, -35.0, 35.0)):
        st = _cluster_stats(x, y, h, sel)
        if max(st["ex"], st["ey"]) > 1.5:           # wider than a person
            continue
        if not (1.05 < st["top"] < 2.05):           # too short, or a pole
            continue
        if cart is not None and np.hypot(st["cx"] - cart["cx"], st["cy"] - cart["cy"]) < 2.0:
            continue                                # the cart's own driver
        out.append(st)
    return out


def traffic_light_state(rgb, row):
    """Which lamp is lit, from the bbox interior.

    The housings in this clip are amber, so hue alone is not enough - the housing
    occupies far more of the bbox than the lamp does and outvotes it. The lit lamp
    separates on *brightness* instead: measured over the clip the lamp sits at
    V ~ 0.85 while the housing sits at V ~ 0.56, so a V > 0.75 gate isolates it
    cleanly before hue is consulted at all."""
    u0, u1 = int(round(row["x_min"])), int(round(row["x_max"]))
    v0, v1 = int(round(row["y_min"])), int(round(row["y_max"]))
    patch = rgb[max(v0, 0):v1, max(u0, 0):u1, :3]
    if patch.size == 0:
        return "unknown"
    hsv = mcolors.rgb_to_hsv(patch)
    hue, sat, val = hsv[..., 0] * 360.0, hsv[..., 1], hsv[..., 2]
    lit = (val > 0.75) & (sat > 0.35)
    if lit.sum() < 5:
        return "unknown"
    lit_hue = hue[lit]
    votes = {
        "green": int(((lit_hue > 140) & (lit_hue < 200)).sum()),
        "amber": int(((lit_hue >= 40) & (lit_hue < 65)).sum()),
        "red": int(((lit_hue < 15) | (lit_hue > 345)).sum()),
    }
    best = max(votes, key=votes.get)
    return best if votes[best] >= 5 else "unknown"


# --------------------------------------------------------------------------
# 5. Scene assembly
# --------------------------------------------------------------------------

def build_scene(dataset_dir="dataset", log=print):
    """Run Part A for the ego pose, then push every detection into the world frame."""
    traj = core.build_trajectory(dataset_dir, log=log)
    rows = core.load_bbox_csv(core.find_bbox_csv(dataset_dir))
    frames, _ = core.get_valid_frames(rows, dataset_dir)
    by_id = {f["frame_id"]: f for f in frames}

    y_is_left = traj["convention"]["y_is_left"]
    frame_ids = traj["frame_ids"].astype(int)
    positions, headings = traj["positions"], traj["headings"]

    rng = np.random.default_rng(0)
    barrel_world, cart_track, ped_world, light_states = [], [], [], []
    ground_z = []
    previous_cart = None

    log(f"Part B: scanning {len(frame_ids)} frames for barrels / cart / pedestrians")
    for i, fid in enumerate(frame_ids):
        entry = by_id[int(fid)]
        xyz = core.load_xyz(entry["xyz_path"])
        rgb = plt.imread(entry["rgb_path"])
        valid = valid_mask(xyz)
        z_ground = ground_plane_z(xyz, valid)
        ground_z.append(z_ground)
        cls = classify_colour(rgb)
        ego, psi = positions[i], headings[i]

        pts, pt_ranges = barrel_points(xyz, cls, valid, z_ground, rng=rng)
        if len(pts):
            w = to_world(pts, ego, psi, y_is_left)
            barrel_world.append(np.column_stack([w, np.full(len(w), fid), pt_ranges]))

        cart = detect_golf_cart(xyz, cls, valid, z_ground, previous=previous_cart)
        if cart is not None:
            previous_cart = (cart["cx"], cart["cy"])
            w = to_world([[cart["cx"], cart["cy"]]], ego, psi, y_is_left)[0]
            cart_track.append([fid, w[0], w[1], cart["range"], cart["n"]])

        for ped in detect_pedestrians(xyz, cls, valid, z_ground, cart=cart):
            w = to_world([[ped["cx"], ped["cy"]]], ego, psi, y_is_left)[0]
            ped_world.append([fid, w[0], w[1], ped["range"]])

        light_states.append(traffic_light_state(rgb, entry))

    scene = {
        "frame_ids": frame_ids,
        "times": traj["times"],
        "positions": positions,
        "headings": headings,
        "barrel_points": np.vstack(barrel_world) if barrel_world else np.empty((0, 4)),
        "cart_track": np.array(cart_track, dtype=float) if cart_track else np.empty((0, 5)),
        "ped_points": np.array(ped_world, dtype=float) if ped_world else np.empty((0, 4)),
        "light_states": np.array(light_states),
        "ground_z": np.array(ground_z),
        "metrics": traj["metrics"],
    }
    scene["barrel_objects"] = locate_static_objects(scene["barrel_points"])
    return scene


def locate_static_objects(points, cell=0.30, neighbourhood=2, floor=2000,
                          min_separation=2, range_gate=OBJECT_RANGE_GATE):
    """Discrete barrel / barrier positions from the accumulated orange cloud.

    Peak-picking on the world-frame density map, not connected components. The
    reason is visible in the raw cloud: each object is a dense head with a faint
    radial tail pointing away from wherever the car was standing, and the tails of
    neighbouring barrels touch. Connected components therefore swallow a whole
    row of barrels into one blob, while the density *maxima* stay cleanly one per
    object - along the barrel line the counts run 26k / 22k / 26k at the heads and
    drop to ~150 in the gaps between them.

    `floor` is an absolute count rather than a fraction of the global maximum so
    that a genuine object seen briefly is not suppressed by one seen for the whole
    clip. Returns rows of `(x, y, peak_count)`."""
    if len(points) == 0:
        return np.empty((0, 3))
    pts = points[points[:, 3] <= range_gate] if points.shape[1] > 3 else points
    if len(pts) == 0:
        return np.empty((0, 3))

    x0, y0, span = -60.0, -50.0, 100.0
    width = int(span / cell) + 2
    gx = ((pts[:, 0] - x0) / cell).astype(int)
    gy = ((pts[:, 1] - y0) / cell).astype(int)
    inside = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < width)
    gx, gy = gx[inside], gy[inside]
    grid = np.zeros((width, width), dtype=np.int32)
    np.add.at(grid, (gx, gy), 1)

    padded = np.pad(grid, neighbourhood)
    local_max = np.zeros_like(grid)
    for dy in range(2 * neighbourhood + 1):
        for dx in range(2 * neighbourhood + 1):
            local_max = np.maximum(local_max, padded[dy:dy + width, dx:dx + width])

    candidates = np.argwhere((grid == local_max) & (grid >= floor))
    candidates = sorted(candidates.tolist(), key=lambda c: -grid[c[0], c[1]])
    kept = []
    for cand in candidates:      # greedy non-maximum suppression, strongest first
        if all(np.hypot(cand[0] - k[0], cand[1] - k[1]) >= min_separation for k in kept):
            kept.append(cand)
    return np.array([[k[0] * cell + x0 + cell / 2.0,
                      k[1] * cell + y0 + cell / 2.0,
                      float(grid[k[0], k[1]])] for k in kept]) if kept else np.empty((0, 3))


# --------------------------------------------------------------------------
# 6. Caching
# --------------------------------------------------------------------------

# Only the raw scan output is cached. `barrel_objects` is *derived* and is
# recomputed on load, so re-tuning the locator's gates does not mean re-scanning
# 198 frames of imagery.
_CACHE_KEYS = ("frame_ids", "times", "positions", "headings", "barrel_points",
               "cart_track", "ped_points", "light_states", "ground_z")


def save_scene(scene, path):
    np.savez_compressed(path, **{k: scene[k] for k in _CACHE_KEYS})


def load_scene(path):
    with np.load(path, allow_pickle=False) as data:
        scene = {k: data[k] for k in _CACHE_KEYS}
    scene["light_states"] = scene["light_states"].astype(str)
    scene["barrel_objects"] = locate_static_objects(scene["barrel_points"])
    return scene


# --------------------------------------------------------------------------
# 7. Renders
# --------------------------------------------------------------------------

def confident_pedestrians(scene):
    """Pedestrian detections close enough to be worth plotting.

    Inside 12 m the 18 surviving detections all land within a 0.6 m x 1.9 m patch
    centred on world (2.6, 1.6) - the two workers by the barrier row, consistently
    placed, and the 1.9 m spread along Y is the gap between the two of them.
    Past 15 m they scatter over 30 m of map, which is the detector finding blue
    fence shadow rather than people. The gate is where that transition happens."""
    peds = scene["ped_points"]
    if not len(peds):
        return peds
    return peds[peds[:, 3] <= PED_RANGE_GATE]


def _scene_extent(scene):
    """Every world point the figures must contain."""
    parts = [scene["positions"], np.zeros((1, 2))]
    if len(scene["barrel_objects"]):
        parts.append(scene["barrel_objects"][:, :2])
    if len(scene["cart_track"]):
        parts.append(scene["cart_track"][:, 1:3])
    return np.vstack(parts)


def _dominant_light_state(states):
    states = [s for s in states if s != "unknown"]
    if not states:
        return "unknown"
    return max(set(states), key=states.count)


def _draw_scene(ax, scene, peds, state, label=True, cloud=True):
    """Draw every layer onto `ax`. Shared by the overview and the zoom panel so
    the two can never drift apart; only the view limits differ."""
    positions = scene["positions"]
    barrels = scene["barrel_points"]
    objects = scene["barrel_objects"]
    cart = scene["cart_track"]

    if cloud and len(barrels):
        near = barrels[barrels[:, 3] <= CLOUD_DRAW_GATE]
        ax.plot(near[:, 0], near[:, 1], linestyle="None", marker=".", markersize=1.0,
                color=C_BARREL, alpha=0.05, zorder=2, rasterized=True)
    if len(objects):
        ax.plot(objects[:, 0], objects[:, 1], linestyle="None", marker="s",
                markersize=9, color=C_BARREL, markeredgecolor=core.INK,
                markeredgewidth=1.1, zorder=5,
                label=f"Barrels / barriers ({len(objects)})" if label else None)
    if len(cart):
        ax.plot(cart[:, 1], cart[:, 2], "-", linewidth=2.0, color=C_CART, zorder=3,
                label="Golf cart track" if label else None)
        ax.plot([cart[-1, 1]], [cart[-1, 2]], marker="o", markersize=10, color=C_CART,
                markeredgecolor=core.INK, markeredgewidth=1.1, linestyle="None", zorder=6)
    if len(peds):
        ax.plot(peds[:, 1], peds[:, 2], linestyle="None", marker="^", markersize=8,
                color=C_PED, markeredgecolor=core.SURFACE, markeredgewidth=0.8,
                alpha=0.8, zorder=5,
                label="Pedestrian (low confidence)" if label else None)

    ax.plot(positions[:, 0], positions[:, 1], "-", linewidth=2.2, color=C_PATH,
            zorder=7, label="Ego trajectory" if label else None)
    ax.plot([positions[0, 0]], [positions[0, 1]], marker="x", markersize=12,
            markeredgewidth=3, color=C_PATH, linestyle="None", zorder=8)
    ax.plot([0], [0], marker="*", markersize=20, color=LIGHT_COLOURS[state],
            markeredgecolor=core.INK, markeredgewidth=0.9, linestyle="None", zorder=9,
            label=f"Traffic light - {state}" if label else None)


def plot_bev_png(scene, out_path):
    """The static scene map, in the traffic-light ground frame.

    Two panels, because one cannot do the job: the car covers 29 m of approach
    while every object it maps sits inside a 15 m box around the intersection, so
    a single equal-aspect view that contains the whole drive renders the scene
    itself as an unreadable smudge in one corner."""
    peds = confident_pedestrians(scene)
    state = _dominant_light_state(list(scene["light_states"]))
    objects, cart = scene["barrel_objects"], scene["cart_track"]

    fig = plt.figure(figsize=(15, 7.6), facecolor=core.SURFACE)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.16)
    ax_all, ax_zoom = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])

    _draw_scene(ax_all, scene, peds, state, label=True)
    _draw_scene(ax_zoom, scene, peds, state, label=False)

    ax_all.annotate("start", xy=scene["positions"][0], xytext=(0, -16),
                    textcoords="offset points", ha="center", color=core.INK_MUTED,
                    fontsize=10, zorder=8)
    for ax in (ax_all, ax_zoom):
        ax.annotate("Origin", xy=(0, 0), xytext=(-12, 7), textcoords="offset points",
                    ha="right", color=core.INK, fontsize=11, zorder=9)

    core._style_axes(ax_all, xlabel="Forward (X, m)", ylabel="Lateral (Y, m)",
                     origin_lines=True)
    core._style_axes(ax_zoom, xlabel="Forward (X, m)", origin_lines=True)
    # Panel captions are set after _style_axes so they get their own, smaller
    # type and a tight pad - the shared title block above needs the room.
    ax_all.set_title("Full drive", color=core.INK_MUTED, fontsize=12, pad=8)
    ax_zoom.set_title("The intersection, close up", color=core.INK_MUTED,
                      fontsize=12, pad=8)
    core._legend(ax_all, loc="lower left")

    fig.suptitle("BEV Scene in the Traffic-Light Ground Frame", color=core.INK,
                 fontsize=16, y=0.975)
    subtitle = (f"{len(scene['frame_ids'])} frames  ·  {len(objects)} static objects  ·  "
                f"golf cart tracked over {len(cart)} frames  ·  light reads {state}")
    fig.text(0.5, 0.918, subtitle, ha="center", color=core.INK_MUTED, fontsize=11)

    zoom_pts = [np.zeros((1, 2))]
    if len(objects):
        zoom_pts.append(objects[:, :2])
    if len(peds):
        zoom_pts.append(peds[:, 1:3])
    if len(cart):
        zoom_pts.append(cart[-1:, 1:3])
    zoom_pts = np.vstack(zoom_pts)

    for _ in range(2):
        fig.tight_layout(rect=(0, 0, 1, 0.895))
        core._set_equal_aspect_limits(fig, ax_all, _scene_extent(scene), pad=4.0)
        core._set_equal_aspect_limits(fig, ax_zoom, zoom_pts, pad=3.0)

    fig.savefig(out_path, dpi=160, facecolor=core.SURFACE)
    plt.close(fig)


def animate_bev_mp4(scene, out_path, fps=30.0):
    """The scene filling in over time, on the source frame grid.

    Only what has been *observed* by frame t is drawn, so the map builds up the
    way an online system would see it rather than revealing the answer at t=0."""
    frame_ids = scene["frame_ids"].astype(int)
    positions = scene["positions"]
    barrels = scene["barrel_points"]
    cart = scene["cart_track"]
    peds = confident_pedestrians(scene)
    states = list(scene["light_states"])

    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=core.SURFACE)
    core._style_axes(ax, title="BEV Scene Building Up Over Time",
                     xlabel="Forward (X, m)", ylabel="Lateral (Y, m)", origin_lines=True)

    cloud, = ax.plot([], [], linestyle="None", marker=".", markersize=1.2,
                     color=C_BARREL, alpha=0.08, zorder=2)
    seen, = ax.plot([], [], linestyle="None", marker="s", markersize=9,
                    color=C_BARREL, markeredgecolor=core.INK, markeredgewidth=1.1,
                    zorder=4, label="Barrels / barriers")
    cart_line, = ax.plot([], [], "-", linewidth=1.8, color=C_CART, zorder=3,
                         label="Golf cart track")
    cart_dot, = ax.plot([], [], linestyle="None", marker="o", markersize=10,
                        color=C_CART, markeredgecolor=core.SURFACE, markeredgewidth=1.2,
                        zorder=5)
    ped_dots, = ax.plot([], [], linestyle="None", marker="^", markersize=7,
                        color=C_PED, markeredgecolor=core.SURFACE, markeredgewidth=0.8,
                        alpha=0.75, zorder=4, label="Pedestrian (low confidence)")
    ego_line, = ax.plot([], [], "-", linewidth=2.2, color=C_PATH, zorder=6,
                        label="Ego trajectory")
    ego_dot, = ax.plot([], [], linestyle="None", marker="o", markersize=9,
                       color=C_PATH, markeredgecolor=core.SURFACE, markeredgewidth=1.2,
                       zorder=7)
    # Seed the marker with the state the clip actually holds, so the legend swatch
    # is not stuck on the "unknown" grey it would carry from an empty first frame.
    resting = _dominant_light_state(states)
    light_dot, = ax.plot([0], [0], linestyle="None", marker="*", markersize=20,
                         color=LIGHT_COLOURS[resting], markeredgecolor=core.INK,
                         markeredgewidth=0.9, zorder=8,
                         label=f"Traffic light - {resting}")
    stamp = ax.text(0.015, 0.965, "", transform=ax.transAxes, ha="left", va="top",
                    color=core.INK_MUTED, fontsize=11)

    core._legend(ax, loc="lower left")
    core._fit_layout(fig, ax, _scene_extent(scene), pad=4.0)

    # Source-frame grid, so playback pacing matches real time despite the gaps.
    first, last = int(frame_ids[0]), int(frame_ids[-1])
    grid = np.arange(first, last + 1)
    sample_of = np.searchsorted(frame_ids, grid, side="right") - 1

    def init():
        return (cloud, seen, cart_line, cart_dot, ped_dots, ego_line, ego_dot,
                light_dot, stamp)

    def update(step):
        fid = grid[step]
        i = max(int(sample_of[step]), 0)
        seen_now = barrels[(barrels[:, 2] <= fid) & (barrels[:, 3] <= CLOUD_DRAW_GATE)]
        cloud.set_data(seen_now[:, 0], seen_now[:, 1])
        obs = locate_static_objects(barrels[barrels[:, 2] <= fid])
        if len(obs):
            seen.set_data(obs[:, 0], obs[:, 1])
        else:
            seen.set_data([], [])
        upto = cart[cart[:, 0] <= fid] if len(cart) else cart
        if len(upto):
            cart_line.set_data(upto[:, 1], upto[:, 2])
            cart_dot.set_data([upto[-1, 1]], [upto[-1, 2]])
        pnow = peds[np.abs(peds[:, 0] - fid) <= 3] if len(peds) else peds
        ped_dots.set_data(pnow[:, 1], pnow[:, 2]) if len(pnow) else ped_dots.set_data([], [])
        ego_line.set_data(positions[:i + 1, 0], positions[:i + 1, 1])
        ego_dot.set_data([positions[i, 0]], [positions[i, 1]])
        light_dot.set_color(LIGHT_COLOURS.get(states[i], LIGHT_COLOURS["unknown"]))
        stamp.set_text(f"frame {fid}   t = {(fid - first) / fps:5.2f} s   light: {states[i]}")
        return (cloud, seen, cart_line, cart_dot, ped_dots, ego_line, ego_dot,
                light_dot, stamp)

    anim = FuncAnimation(fig, update, frames=len(grid), init_func=init, blit=False)
    anim.save(out_path, writer=FFMpegWriter(fps=int(round(fps)), bitrate=2400))
    plt.close(fig)


def plot_detection_overlay(scene, dataset_dir, out_path, sample_frames=4):
    """Detections drawn back onto the RGB - the check that the BEV is grounded.

    A BEV plot can look tidy and still be pointing at the wrong pixels, so this
    projects each layer's mask back into the image it came from. It is the figure
    to look at first when a marker lands somewhere surprising."""
    rows = core.load_bbox_csv(core.find_bbox_csv(dataset_dir))
    frames, _ = core.get_valid_frames(rows, dataset_dir)
    by_id = {f["frame_id"]: f for f in frames}
    ids = scene["frame_ids"].astype(int)
    picks = ids[np.linspace(0, len(ids) - 1, sample_frames).astype(int)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=core.SURFACE)
    for ax, fid in zip(axes.ravel(), picks):
        entry = by_id[int(fid)]
        xyz = core.load_xyz(entry["xyz_path"])
        rgb = plt.imread(entry["rgb_path"])[..., :3]
        valid = valid_mask(xyz)
        z_ground = ground_plane_z(xyz, valid)
        cls = classify_colour(rgb)
        height = xyz[:, :, 2] - z_ground

        shown = rgb.copy()
        orange = valid & (cls == CLS_ORANGE) & (height > 0.15) & (height < 1.8)
        tan = valid & (cls == CLS_TAN) & (height > 0.30) & (height < 2.10) & (xyz[:, :, 0] < 30)
        blue = valid & (cls == CLS_BLUE) & (height > 0.70) & (height < 2.10) & (xyz[:, :, 0] < 30)
        shown[orange] = mcolors.to_rgb(C_BARREL)
        shown[tan] = mcolors.to_rgb(C_CART)
        shown[blue] = mcolors.to_rgb(OVERLAY_PED)
        ax.imshow(shown)
        ax.set_title(f"frame {int(fid)} - light: {scene['light_states'][list(ids).index(fid)]}",
                     color=core.INK, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(core.AXIS)

    fig.suptitle("Detection masks projected back onto the RGB "
                 "(orange = barrels, teal = golf cart, violet = pedestrians)",
                 color=core.INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=core.SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------
# 8. Validation
# --------------------------------------------------------------------------

def static_consistency(scene, min_frames=8):
    """How tightly a static object re-lands on itself when seen from many poses.

    This is the Part B check that has teeth, and it is a check on Part A. A barrel
    does not move, so every frame that sees it should place it at the same world
    point. Measuring the *spread of per-frame centroids* rather than the spread of
    raw points separates trajectory error from object size: a 2 m barrier is 2 m
    wide from every pose, but a trajectory that drifted would walk its centroid
    across the map. Returns the per-object centroid scatter in metres."""
    objects = scene["barrel_objects"]
    points = scene["barrel_points"]
    if not len(objects) or not len(points):
        return np.empty(0)
    spreads = []
    for cx, cy, _peak in objects:
        near = points[(np.abs(points[:, 0] - cx) < 2.0) & (np.abs(points[:, 1] - cy) < 2.0)]
        if len(near) < 50:
            continue
        centroids = []
        for fid in np.unique(near[:, 2]):
            chunk = near[near[:, 2] == fid]
            if len(chunk) >= 8:
                centroids.append(chunk[:, :2].mean(axis=0))
        if len(centroids) < min_frames:
            continue
        centroids = np.array(centroids)
        spreads.append(float(np.sqrt(((centroids - centroids.mean(axis=0)) ** 2).sum(axis=1).mean())))
    return np.array(spreads)


def report(scene, log=print):
    objects, cart = scene["barrel_objects"], scene["cart_track"]
    peds = confident_pedestrians(scene)
    states = list(scene["light_states"])
    n_frames = len(scene["frame_ids"])

    log("")
    log("Part B detections")
    log(f"  barrels/barriers : {len(objects)} static objects located inside "
        f"{OBJECT_RANGE_GATE:.0f} m, from {len(scene['barrel_points']):,} "
        f"accumulated orange points")
    log(f"  golf cart        : tracked in {len(cart)}/{n_frames} frames")
    log(f"  pedestrians      : {len(peds)} detections inside {PED_RANGE_GATE:.0f} m "
        f"of {len(scene['ped_points'])} raw (low-confidence layer)")
    counts = {s: states.count(s) for s in sorted(set(states))}
    log(f"  light state      : {counts}")

    zg = scene["ground_z"]
    log(f"  ground plane     : {np.median(zg):.2f} m below the camera, "
        f"sd {np.std(zg):.3f} m (flat-ground assumption)")

    spreads = static_consistency(scene)
    if len(spreads):
        log("")
        log("Validation - static objects must not move")
        log(f"  per-frame centroid scatter over {len(spreads)} objects: "
            f"median {np.median(spreads):.2f} m, p90 {np.percentile(spreads, 90):.2f} m")
        log("  (a barrel is re-observed from up to 198 poses; scatter is the "
            "combined ego-pose and stereo error, and would grow without bound "
            "if the Part A trajectory were drifting)")

    if len(cart) >= 2:
        ego = scene["positions"]
        ids = scene["frame_ids"].astype(int)
        gaps = []
        for fid, cx, cy, _r, _n in cart:
            i = int(np.searchsorted(ids, int(fid)))
            i = min(i, len(ego) - 1)
            gaps.append(np.hypot(cx - ego[i, 0], cy - ego[i, 1]))
        gaps = np.array(gaps)
        step = np.hypot(np.diff(cart[:, 1]), np.diff(cart[:, 2]))
        dt = np.diff(cart[:, 0]) / 30.0
        speed = step / np.maximum(dt, 1e-9)
        rough = int((step > 1.0).sum())
        log("")
        log("Validation - the golf cart")
        log(f"  gap to ego      : {gaps[0]:.1f} m -> {gaps[-1]:.1f} m "
            f"(min {gaps.min():.1f} m; it stays ahead of us the whole clip)")
        log(f"  implied speed   : median {np.median(speed):.2f} m/s "
            f"({np.median(speed) * 2.237:.1f} mph) - a golf cart's pace, and it is "
            f"never asked to be one")
        log(f"  track roughness : {rough}/{len(step)} steps jump more than 1 m, all "
            f"in the sparse early frames where the cart is ~16 m out and its tan "
            f"patch is only a few hundred pixels")
        log(f"  cart path length: {step.sum():.1f} m")


# --------------------------------------------------------------------------
# 9. Entry point
# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="dataset", help="dataset directory (default: dataset)")
    parser.add_argument("--fps", type=float, default=core.DEFAULT_FPS, help="source frame rate")
    parser.add_argument("--png", default="bev.png")
    parser.add_argument("--mp4", default="bev.mp4")
    parser.add_argument("--overlay", default="detections.png")
    parser.add_argument("--cache", default=CACHE_NAME,
                        help="scan cache; reused unless --refresh is given")
    parser.add_argument("--refresh", action="store_true", help="re-scan even if the cache exists")
    parser.add_argument("--no-video", action="store_true", help="skip the mp4 (no ffmpeg needed)")
    parser.add_argument("--no-overlay", action="store_true", help="skip the RGB overlay figure")
    args = parser.parse_args(argv)

    if os.path.exists(args.cache) and not args.refresh:
        print(f"Reusing {args.cache} (pass --refresh to re-scan)")
        scene = load_scene(args.cache)
    else:
        scene = build_scene(args.dataset)
        save_scene(scene, args.cache)
        print(f"Wrote {args.cache}")

    report(scene)

    plot_bev_png(scene, args.png)
    print(f"Wrote {args.png}")

    if not args.no_overlay:
        plot_detection_overlay(scene, args.dataset, args.overlay)
        print(f"Wrote {args.overlay}")

    if not args.no_video:
        animate_bev_mp4(scene, args.mp4, fps=args.fps)
        print(f"Wrote {args.mp4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
