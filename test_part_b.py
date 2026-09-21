#!/usr/bin/env python3
"""Unit tests for part_b.py.

Run with:  python -m unittest -v test_part_b
Synthetic only - no dataset needed. Same reasoning as test_solution.py: the
detectors are checked against scenes whose answer is known by construction, not
against the one clip we happen to have, so a threshold that drifts shows up here
rather than as a plausible-looking but wrong marker on the BEV.
"""

import unittest

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors

import part_b as pb
import solution as core


def synth_frame(shape=(60, 80), blocks=()):
    """Build a synthetic (xyz, rgb) pair.

    `blocks` is a sequence of `(rows, cols, (X, Y, Z), hsv)`: every pixel in that
    slice gets that camera-frame point and that colour. Everything else is NaN
    depth (a stereo hole) over mid-grey."""
    h, w = shape
    xyz = np.full((h, w, 3), np.nan, dtype=float)
    rgb = np.full((h, w, 3), 0.5, dtype=float)
    for rows, cols, point, hsv in blocks:
        xyz[rows, cols] = point
        rgb[rows, cols] = mcolors.hsv_to_rgb(np.array(hsv))
    return xyz, rgb


class TestColourClassification(unittest.TestCase):
    def test_known_colours_land_in_their_class(self):
        cases = {
            pb.CLS_ORANGE: (20 / 360.0, 0.85, 0.80),   # construction orange
            pb.CLS_WHITE: (0.0, 0.03, 0.90),           # white jersey barrier
            pb.CLS_TAN: (50 / 360.0, 0.30, 0.75),      # golf cart canopy
            pb.CLS_DARK: (0.0, 0.10, 0.10),            # shadow / tyre
            pb.CLS_VEG: (110 / 360.0, 0.60, 0.45),     # foliage
            pb.CLS_BLUE: (220 / 360.0, 0.55, 0.50),    # workwear
        }
        for expected, hsv in cases.items():
            rgb = mcolors.hsv_to_rgb(np.array(hsv)).reshape(1, 1, 3)
            self.assertEqual(int(pb.classify_colour(rgb)[0, 0]), expected,
                             msg=f"hsv {hsv} should classify as {expected}")

    def test_orange_wins_over_tan_on_overlap(self):
        # A saturated orange sits inside the tan hue window too; orange is the
        # higher-precision class and is applied last so it must win.
        rgb = mcolors.hsv_to_rgb(np.array([38 / 360.0, 0.70, 0.70])).reshape(1, 1, 3)
        self.assertEqual(int(pb.classify_colour(rgb)[0, 0]), pb.CLS_ORANGE)

    def test_sky_is_not_an_object_class(self):
        sky = mcolors.hsv_to_rgb(np.array([210 / 360.0, 0.25, 0.95])).reshape(1, 1, 3)
        self.assertNotIn(int(pb.classify_colour(sky)[0, 0]),
                         (pb.CLS_ORANGE, pb.CLS_TAN))


class TestFrameGeometry(unittest.TestCase):
    def test_valid_mask_rejects_holes_zeros_and_behind_camera(self):
        xyz = np.array([[[10.0, 1.0, -1.0],      # good
                         [np.nan, 1.0, -1.0],    # hole
                         [0.0, 0.0, 0.0],        # exact zero
                         [-5.0, 1.0, -1.0]]])    # behind the camera
        self.assertEqual(pb.valid_mask(xyz).ravel().tolist(), [True, False, False, False])

    def test_ground_plane_ignores_the_far_field(self):
        """The road is measured from what is close, because that is where it is
        actually the dominant surface. Let a distant hillside outvote it and the
        height gates - which every detector depends on - all shift together."""
        xyz, _ = synth_frame(blocks=[
            (slice(40, 60), slice(None), (8.0, 0.0, -1.80), (0.0, 0.0, 0.5)),
            (slice(0, 40), slice(None), (35.0, 0.0, 2.50), (0.0, 0.0, 0.5)),
        ])
        self.assertAlmostEqual(pb.ground_plane_z(xyz, pb.valid_mask(xyz)), -1.80, places=6)

    def test_ground_plane_is_the_near_field_median(self):
        xyz, _ = synth_frame(blocks=[
            (slice(0, 50), slice(None), (8.0, 0.0, -1.75), (0.0, 0.0, 0.5)),   # road
            (slice(50, 60), slice(None), (8.0, 0.0, 3.0), (0.0, 0.0, 0.5)),    # sign
        ])
        z = pb.ground_plane_z(xyz, pb.valid_mask(xyz))
        self.assertAlmostEqual(z, -1.75, places=6)


class TestWorldTransform(unittest.TestCase):
    def test_identity_pose_is_a_no_op(self):
        out = pb.to_world([[3.0, 4.0]], np.array([0.0, 0.0]), 0.0, y_is_left=True)
        np.testing.assert_allclose(out, [[3.0, 4.0]])

    def test_rotation_and_translation(self):
        out = pb.to_world([[1.0, 0.0]], np.array([1.0, 2.0]), np.pi / 2, y_is_left=True)
        np.testing.assert_allclose(out, [[1.0, 3.0]], atol=1e-12)

    def test_y_right_data_is_flipped(self):
        out = pb.to_world([[0.0, 5.0]], np.array([0.0, 0.0]), 0.0, y_is_left=False)
        np.testing.assert_allclose(out, [[0.0, -5.0]])

    def test_light_vector_maps_back_onto_the_origin(self):
        """The consistency condition that ties Part B to Part A.

        Part A puts the light at the world origin via p = -R(psi) @ c. Pushing that
        same camera vector c forward through `to_world` from that pose must land on
        (0, 0) again - if it does not, the two files disagree about the frame."""
        for psi in (0.0, 0.4, -1.1, 2.7):
            c = np.array([12.0, -3.5])
            ego = -core.rotation(psi) @ c
            np.testing.assert_allclose(pb.to_world([c], ego, psi, True)[0],
                                       [0.0, 0.0], atol=1e-12)


class TestLabelGrid(unittest.TestCase):
    def test_separate_blobs_get_separate_labels(self):
        occ = np.zeros((9, 9), dtype=bool)
        occ[1:3, 1:3] = True
        occ[6:8, 6:8] = True
        labels = pb.label_grid(occ)
        self.assertEqual(len(np.unique(labels[occ])), 2)

    def test_diagonal_touch_is_one_component(self):
        occ = np.zeros((6, 6), dtype=bool)
        occ[1, 1] = occ[2, 2] = True          # 8-connected
        self.assertEqual(len(np.unique(pb.label_grid(occ)[occ])), 1)

    def test_empty_grid_is_all_background(self):
        self.assertEqual(pb.label_grid(np.zeros((4, 4), bool)).sum(), 0)


class TestClusterXy(unittest.TestCase):
    def test_two_clusters_are_separated(self):
        rng = np.random.default_rng(0)
        a = rng.normal([5.0, 0.0], 0.1, size=(300, 2))
        b = rng.normal([15.0, 6.0], 0.1, size=(300, 2))
        pts = np.vstack([a, b])
        got = pb.cluster_xy(pts[:, 0], pts[:, 1], cell=0.5, min_cell_count=5,
                            min_points=50)
        self.assertEqual(len(got), 2)

    def test_sparse_speckle_is_dropped_by_the_density_floor(self):
        rng = np.random.default_rng(1)
        speckle = rng.uniform([0, -20], [40, 20], size=(200, 2))
        got = pb.cluster_xy(speckle[:, 0], speckle[:, 1], cell=0.3,
                            min_cell_count=10, min_points=40)
        self.assertEqual(got, [])


class TestLocateStaticObjects(unittest.TestCase):
    def _cloud(self, centres, n=4000, rng_value=10.0, spread=0.12):
        rng = np.random.default_rng(3)
        rows = []
        for cx, cy in centres:
            xy = rng.normal([cx, cy], spread, size=(n, 2))
            rows.append(np.column_stack([xy, np.full(n, 100.0), np.full(n, rng_value)]))
        return np.vstack(rows)

    def test_finds_one_peak_per_object(self):
        centres = [(1.0, -1.5), (1.0, -3.5), (-2.0, 6.0)]
        got = pb.locate_static_objects(self._cloud(centres), floor=200)
        self.assertEqual(len(got), 3)
        for cx, cy in centres:
            self.assertTrue(np.any(np.hypot(got[:, 0] - cx, got[:, 1] - cy) < 0.5),
                            msg=f"no peak near ({cx}, {cy})")

    def test_observations_beyond_the_range_gate_are_ignored(self):
        far = self._cloud([(1.0, -1.5)], rng_value=pb.OBJECT_RANGE_GATE + 5.0)
        self.assertEqual(len(pb.locate_static_objects(far, floor=200)), 0)

    def test_tied_adjacent_cells_are_suppressed_to_one_peak(self):
        """What the non-maximum suppression pass is actually for.

        The local-max filter uses `==`, so two *equal* neighbouring cells are both
        "maxima" and both survive it - a plateau reports one object twice. Here two
        adjacent cells (0.3 m apart, one cell) carry identical counts by
        construction; only the suppression pass can collapse them."""
        a = np.tile([1.00, -1.50, 100.0, 10.0], (500, 1))
        b = np.tile([1.00, -1.20, 100.0, 10.0], (500, 1))
        got = pb.locate_static_objects(np.vstack([a, b]), floor=200)
        self.assertEqual(len(got), 1)

    def test_empty_input(self):
        self.assertEqual(len(pb.locate_static_objects(np.empty((0, 4)))), 0)


class TestTrafficLightState(unittest.TestCase):
    def _head(self, lamp_hsv=None, lamp_rows=slice(30, 40), housing_hue=48 / 360.0):
        """A 48x16 signal head: amber housing, optionally one bright lamp.

        The housing hue defaults to 48 deg, i.e. squarely inside the detector's
        own amber vote band, so the only thing keeping it from being reported as
        a lit amber lamp is the brightness gate. That is the property under test."""
        head = np.tile(mcolors.hsv_to_rgb(np.array([housing_hue, 0.75, 0.56])), (48, 16, 1))
        if lamp_hsv is not None:
            head[lamp_rows, 5:11] = mcolors.hsv_to_rgb(np.array(lamp_hsv))
        return head

    def _row(self):
        return {"x_min": 0, "y_min": 0, "x_max": 16, "y_max": 48}

    def test_green_lamp(self):
        rgb = self._head((170 / 360.0, 0.80, 0.90))
        self.assertEqual(pb.traffic_light_state(rgb, self._row()), "green")

    def test_red_lamp(self):
        rgb = self._head((2 / 360.0, 0.85, 0.90), lamp_rows=slice(4, 14))
        self.assertEqual(pb.traffic_light_state(rgb, self._row()), "red")

    def test_amber_housing_alone_is_not_a_lit_lamp(self):
        """The trap this detector exists to avoid: the housing is amber and fills
        most of the bbox, so hue alone would report amber on every frame."""
        self.assertEqual(pb.traffic_light_state(self._head(), self._row()), "unknown")

    def test_a_lit_green_lamp_beats_a_much_larger_amber_housing(self):
        """Every pixel of the housing votes amber and only a few dozen vote green,
        so a majority over all coloured pixels would get this backwards."""
        rgb = self._head((170 / 360.0, 0.80, 0.90))
        self.assertEqual(pb.traffic_light_state(rgb, self._row()), "green")

    def test_empty_bbox(self):
        rgb = np.zeros((10, 10, 3))
        self.assertEqual(pb.traffic_light_state(rgb, {"x_min": 5, "y_min": 5,
                                                      "x_max": 5, "y_max": 5}), "unknown")


class TestGolfCartDetector(unittest.TestCase):
    def _scene(self, cart_xy=(10.0, 1.0)):
        x, y = cart_xy
        xyz, rgb = synth_frame(shape=(60, 80), blocks=[
            (slice(40, 60), slice(None), (6.0, 0.0, -1.8), (0.0, 0.0, 0.45)),   # road
            (slice(24, 34), slice(30, 44), (x, y, -0.6), (50 / 360.0, 0.30, 0.75)),
        ])
        return xyz, rgb

    def test_finds_the_tan_object(self):
        xyz, rgb = self._scene()
        valid = pb.valid_mask(xyz)
        got = pb.detect_golf_cart(xyz, pb.classify_colour(rgb), valid,
                                  pb.ground_plane_z(xyz, valid))
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got["cx"], 10.0, places=6)
        self.assertAlmostEqual(got["cy"], 1.0, places=6)

    def test_a_tan_object_above_head_height_is_not_the_cart(self):
        """A golf cart is on the road. Tan signage and cladding up on a building
        are not, and the height gate is the only thing separating them."""
        xyz, rgb = synth_frame(shape=(60, 80), blocks=[
            (slice(40, 60), slice(None), (6.0, 0.0, -1.8), (0.0, 0.0, 0.45)),
            (slice(4, 14), slice(30, 44), (12.0, 1.0, 4.5), (50 / 360.0, 0.30, 0.75)),
        ])
        valid = pb.valid_mask(xyz)
        self.assertIsNone(pb.detect_golf_cart(xyz, pb.classify_colour(rgb), valid,
                                              pb.ground_plane_z(xyz, valid)))

    def test_no_tan_object_means_no_detection(self):
        xyz, rgb = synth_frame(blocks=[
            (slice(40, 60), slice(None), (6.0, 0.0, -1.8), (0.0, 0.0, 0.45))])
        valid = pb.valid_mask(xyz)
        self.assertIsNone(pb.detect_golf_cart(xyz, pb.classify_colour(rgb), valid,
                                              pb.ground_plane_z(xyz, valid)))

    def test_a_distant_flicker_cannot_steal_an_established_track(self):
        xyz, rgb = self._scene(cart_xy=(25.0, 12.0))
        valid = pb.valid_mask(xyz)
        got = pb.detect_golf_cart(xyz, pb.classify_colour(rgb), valid,
                                  pb.ground_plane_z(xyz, valid), previous=(8.0, 0.5))
        self.assertIsNone(got)


    def test_a_tan_surface_too_large_to_be_a_cart_is_rejected(self):
        """A golf cart is about 2.4 m long. A tan building facade or a stretch of
        gravel spanning 12 m is not one, however well it matches the colour."""
        xyz, rgb = synth_frame(shape=(60, 80), blocks=[
            (slice(40, 60), slice(None), (6.0, 0.0, -1.8), (0.0, 0.0, 0.45)),
        ])
        # A wide tan slab: same colour, same height band, twelve metres of it.
        cols = np.arange(10, 70)
        xyz[24:34, cols] = np.stack(
            [np.full(cols.size, 14.0), np.linspace(-6.0, 6.0, cols.size),
             np.full(cols.size, -0.6)], axis=1)
        rgb[24:34, cols] = mcolors.hsv_to_rgb(np.array([50 / 360.0, 0.30, 0.75]))
        valid = pb.valid_mask(xyz)
        self.assertIsNone(pb.detect_golf_cart(xyz, pb.classify_colour(rgb), valid,
                                              pb.ground_plane_z(xyz, valid)))


class TestPedestrianDetector(unittest.TestCase):
    """The weak layer still gets pinned down, because a detector that is known to
    be fragile is exactly the one whose gates must not drift silently."""

    def _scene(self, point=(10.0, 2.0, -0.2), rows=slice(20, 32), cols=slice(30, 40)):
        xyz, rgb = synth_frame(shape=(60, 80), blocks=[
            (slice(40, 60), slice(None), (6.0, 0.0, -1.8), (0.0, 0.0, 0.45)),
            (rows, cols, point, (220 / 360.0, 0.55, 0.50)),
        ])
        return xyz, rgb

    def test_finds_a_person_sized_blue_object(self):
        xyz, rgb = self._scene()
        valid = pb.valid_mask(xyz)
        got = pb.detect_pedestrians(xyz, pb.classify_colour(rgb), valid,
                                    pb.ground_plane_z(xyz, valid))
        self.assertEqual(len(got), 1)
        self.assertAlmostEqual(got[0]["cx"], 10.0, places=6)

    def test_a_blue_object_shorter_than_a_person_is_rejected(self):
        """A blue bin: same colour, same place, top 0.9 m up.

        The height is chosen to sit *inside* the detector's search band, so the
        thing rejecting it is the person-height test and not the band - there is a
        blue recycling bin beside the barriers in this clip that this is aimed at."""
        xyz, rgb = self._scene(point=(10.0, 2.0, -0.9))
        valid = pb.valid_mask(xyz)
        self.assertEqual(pb.detect_pedestrians(xyz, pb.classify_colour(rgb), valid,
                                               pb.ground_plane_z(xyz, valid)), [])

    def test_the_carts_own_driver_is_not_a_pedestrian(self):
        xyz, rgb = self._scene()
        valid = pb.valid_mask(xyz)
        cart = {"cx": 10.4, "cy": 2.2}
        self.assertEqual(pb.detect_pedestrians(xyz, pb.classify_colour(rgb), valid,
                                               pb.ground_plane_z(xyz, valid),
                                               cart=cart), [])


class TestPedestrianGate(unittest.TestCase):
    def test_range_gate_keeps_only_near_detections(self):
        scene = {"ped_points": np.array([
            [100.0, 3.0, 1.0, pb.PED_RANGE_GATE - 1.0],
            [101.0, -20.0, -9.0, pb.PED_RANGE_GATE + 8.0],
        ])}
        kept = pb.confident_pedestrians(scene)
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(kept[0, 1], 3.0)

    def test_empty(self):
        self.assertEqual(len(pb.confident_pedestrians({"ped_points": np.empty((0, 4))})), 0)


class TestDominantLightState(unittest.TestCase):
    def test_majority_wins_and_unknown_is_ignored(self):
        self.assertEqual(pb._dominant_light_state(
            ["unknown", "green", "green", "red", "unknown"]), "green")

    def test_all_unknown(self):
        self.assertEqual(pb._dominant_light_state(["unknown", "unknown"]), "unknown")


if __name__ == "__main__":
    unittest.main()
