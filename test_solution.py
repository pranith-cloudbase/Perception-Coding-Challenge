#!/usr/bin/env python3
"""Unit tests for solution.py.

Run with:  python -m unittest -v test_solution
No pytest / no dataset required - everything here is synthetic, which is the
point: the geometry is checked against cases whose answer is known in closed
form, not against the one clip we happen to have.
"""

import os
import tempfile
import unittest

import numpy as np

import solution as sol


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

class TestBboxCsv(unittest.TestCase):
    def _write(self, text):
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        handle.write(text)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_accepts_shipped_header(self):
        path = self._write("frame,x1,y1,x2,y2\n7,10,20,30,40\n")
        self.assertEqual(sol.load_bbox_csv(path), [
            {"frame_id": 7, "x_min": 10.0, "y_min": 20.0, "x_max": 30.0, "y_max": 40.0}])

    def test_accepts_documented_header(self):
        path = self._write("frame_id,x_min,y_min,x_max,y_max\n7,10,20,30,40\n")
        self.assertEqual(sol.load_bbox_csv(path)[0]["frame_id"], 7)

    def test_rejects_unknown_column(self):
        path = self._write("frame,left,top,right,bottom\n1,1,1,2,2\n")
        with self.assertRaises(ValueError):
            sol.load_bbox_csv(path)

    def test_skips_blank_lines(self):
        path = self._write("frame,x1,y1,x2,y2\n1,1,1,2,2\n\n2,1,1,2,2\n")
        self.assertEqual(len(sol.load_bbox_csv(path)), 2)


class TestFrameIds(unittest.TestCase):
    def test_uses_the_last_digit_run(self):
        for name, expected in [("left000038.png", 38), ("depth000038.npz", 38),
                               ("frame_0001.npz", 1), ("cam2_frame_0038.png", 38)]:
            self.assertEqual(sol.frame_id_from_name(name), expected, name)

    def test_none_when_no_digits(self):
        self.assertIsNone(sol.frame_id_from_name("README.md"))


class TestValidFrames(unittest.TestCase):
    def test_degenerate_bboxes_are_dropped(self):
        rows = [{"frame_id": 0, "x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0},
                {"frame_id": 1, "x_min": 5, "y_min": 5, "x_max": 1, "y_max": 9}]
        frames, stats = sol.get_valid_frames(rows, tempfile.mkdtemp())
        self.assertEqual(frames, [])
        self.assertEqual(stats["degenerate_bbox"], 2)


# --------------------------------------------------------------------------
# Per-frame localisation
# --------------------------------------------------------------------------

def _synthetic_cloud(h=200, w=260, y_is_left=True):
    """A clean pinhole-ish cloud: X constant forward, Y and Z linear in pixel."""
    vs, us = np.mgrid[0:h, 0:w]
    sign = -1.0 if y_is_left else 1.0
    cloud = np.empty((h, w, 3))
    cloud[:, :, 0] = 10.0
    cloud[:, :, 1] = sign * (us - w / 2.0) * 0.01
    cloud[:, :, 2] = -(vs - h / 2.0) * 0.01
    return cloud


class TestAxisConvention(unittest.TestCase):
    def test_detects_left_handed_y(self):
        self.assertEqual(sol.detect_axis_convention(_synthetic_cloud(y_is_left=True)),
                         {"y_is_left": True, "z_is_up": True})

    def test_detects_right_handed_y(self):
        self.assertEqual(sol.detect_axis_convention(_synthetic_cloud(y_is_left=False)),
                         {"y_is_left": False, "z_is_up": True})


class TestLightPosition(unittest.TestCase):
    def setUp(self):
        # Deliberately non-square bbox and light region: a (u, v) transpose bug
        # would drag background pixels into the patch and shift the answer.
        self.cloud = np.full((300, 400, 3), 0.0)
        self.cloud[:, :, 0] = 80.0   # distant background everywhere
        self.cloud[110:131, 100:161] = (10.0, 1.0, 3.0)   # rows 110-130, cols 100-160
        self.row = {"x_min": 100.0, "y_min": 110.0, "x_max": 160.0, "y_max": 130.0}

    def test_recovers_the_light(self):
        got = sol.estimate_light_position(self.cloud, self.row)
        np.testing.assert_allclose(got["xyz"], (10.0, 1.0, 3.0))

    def test_survives_nan_zero_and_outlier_pixels(self):
        self.cloud[115, 120] = np.nan
        self.cloud[116, 121] = 0.0
        self.cloud[117, 122] = (60.0, 1.0, 3.0)   # background leak, far range
        got = sol.estimate_light_position(self.cloud, self.row)
        np.testing.assert_allclose(got["xyz"], (10.0, 1.0, 3.0))
        self.assertLess(got["n_inliers"], got["n_samples"])

    def test_returns_none_when_nothing_is_valid(self):
        self.cloud[:] = np.nan
        self.assertIsNone(sol.estimate_light_position(self.cloud, self.row))

    def test_index_order_is_row_major(self):
        """The (u, v) vs (v, u) trap: the array is (H, W, 3), so the bbox centre
        pixel (u, v) must be read as xyz[v, u]. Encode the row index in Y and the
        column index in Z, then a transposed read swaps the two."""
        probe = np.zeros((300, 400, 3))
        rows_idx, cols_idx = np.mgrid[0:300, 0:400]
        probe[:, :, 0] = 10.0
        probe[:, :, 1] = rows_idx * 0.001   # encodes v
        probe[:, :, 2] = cols_idx * 0.001   # encodes u
        row = {"x_min": 196.0, "y_min": 96.0, "x_max": 204.0, "y_max": 104.0}
        got = sol.estimate_light_position(probe, row)["xyz"]
        self.assertAlmostEqual(got[1], 0.100, places=6)  # v = 100
        self.assertAlmostEqual(got[2], 0.200, places=6)  # u = 200


class TestGroundVector(unittest.TestCase):
    def test_flips_y_when_camera_y_points_right(self):
        np.testing.assert_allclose(sol.to_ground_vector(np.array([5.0, 2.0, 3.0]), True), [5.0, 2.0])
        np.testing.assert_allclose(sol.to_ground_vector(np.array([5.0, 2.0, 3.0]), False), [5.0, -2.0])


# --------------------------------------------------------------------------
# Temporal cleaning
# --------------------------------------------------------------------------

class TestTemporalCleaning(unittest.TestCase):
    def test_height_outliers_rejected(self):
        heights = np.array([3.6, 3.5, 3.7, 3.6, 12.0, 3.55])
        keep = sol.reject_height_outliers(heights)
        self.assertFalse(keep[4])
        self.assertTrue(keep[[0, 1, 2, 3, 5]].all())

    def test_local_linear_fit_is_exact_on_a_line(self):
        times = np.array([0.0, 0.1, 0.4, 0.5, 0.9, 1.0])  # deliberately uneven
        values = 3.0 * times - 2.0
        fitted, slope = sol.local_linear_fit(times, values, half_window=0.5)
        np.testing.assert_allclose(fitted, values, atol=1e-9)
        np.testing.assert_allclose(slope, 3.0, atol=1e-9)

    def test_local_linear_fit_attenuates_noise(self):
        rng = np.random.default_rng(0)
        times = np.arange(200) / 30.0
        clean = 2.0 * times
        noisy = clean + rng.normal(0.0, 0.05, times.shape)
        fitted = sol.local_linear_smooth(times, noisy, half_window=10 / 30.0)
        self.assertLess(np.std(fitted - clean), np.std(noisy - clean) / 2.0)


# --------------------------------------------------------------------------
# World frame and the heading models
# --------------------------------------------------------------------------

def _arc(psi0, curvature, speed, n=200, fps=30.0, start=(-30.0, 0.0)):
    """Ground-truth constant-curvature drive, and the car->light vectors a
    perfect sensor would report for a light at the world origin."""
    dt = 1.0 / fps
    times = np.arange(n) * dt
    headings = psi0 + curvature * speed * times
    positions = np.empty((n, 2))
    positions[0] = start
    for i in range(1, n):
        mean_heading = 0.5 * (headings[i - 1] + headings[i])
        positions[i] = positions[i - 1] + speed * dt * np.array(
            [np.cos(mean_heading), np.sin(mean_heading)])
    vectors = np.array([sol.rotation(-psi) @ (-p) for psi, p in zip(headings, positions)])
    return times, headings, positions, vectors


class TestWorldFrame(unittest.TestCase):
    def test_reference_heading_puts_the_light_on_plus_x(self):
        c0 = np.array([30.0, 14.0])
        psi0 = sol.reference_heading(c0)
        rotated = sol.rotation(psi0) @ c0
        self.assertAlmostEqual(rotated[1], 0.0, places=9)
        self.assertGreater(rotated[0], 0.0)

    def test_reference_frame_car_sits_on_negative_x(self):
        """The world frame is defined so the car->light line at t0 is along +X,
        which puts the car itself on -X with y = 0."""
        vectors = np.array([[30.0, 14.0], [29.0, 13.0]])
        _, positions = sol.solve_nonholonomic_yaw(vectors)
        self.assertAlmostEqual(positions[0][1], 0.0, places=9)
        self.assertLess(positions[0][0], 0.0)


class TestNonholonomicSolver(unittest.TestCase):
    def test_recovers_a_left_turn(self):
        _, headings, truth, vectors = _arc(psi0=-0.44, curvature=0.03, speed=3.4)
        est_headings, est = sol.solve_nonholonomic_yaw(vectors)
        np.testing.assert_allclose(est, truth, atol=1e-4)
        np.testing.assert_allclose(est_headings, headings, atol=1e-6)

    def test_recovers_a_right_turn(self):
        _, headings, truth, vectors = _arc(psi0=0.3, curvature=-0.04, speed=5.0)
        _, est = sol.solve_nonholonomic_yaw(vectors)
        np.testing.assert_allclose(est, truth, atol=1e-4)

    def test_recovers_a_straight_drive(self):
        _, headings, truth, vectors = _arc(psi0=0.1, curvature=0.0, speed=4.0)
        est_headings, est = sol.solve_nonholonomic_yaw(vectors)
        np.testing.assert_allclose(est, truth, atol=1e-6)
        np.testing.assert_allclose(est_headings, headings, atol=1e-8)

    def test_sideslip_residual_is_driven_to_zero(self):
        """Solver health: every bracketed step must satisfy the constraint it was
        solved for, so the residual is ~0 at the p95 and not just the median."""
        _, _, _, vectors = _arc(psi0=-0.44, curvature=0.03, speed=3.4)
        headings, positions = sol.solve_nonholonomic_yaw(vectors)
        slip = np.degrees(sol.sideslip_angles(headings, positions))
        self.assertLess(np.percentile(slip, 95), 1e-3)


class TestMetrics(unittest.TestCase):
    def test_speed_and_path_length_on_a_known_arc(self):
        times, headings, truth, vectors = _arc(psi0=0.0, curvature=0.02, speed=4.0, n=150)
        metrics = sol.trajectory_metrics(times, headings, truth)
        self.assertAlmostEqual(metrics["path_length_m"], 4.0 * (times[-1] - times[0]), places=6)
        np.testing.assert_allclose(metrics["speed"], 4.0, atol=0.02)


if __name__ == "__main__":
    unittest.main()
