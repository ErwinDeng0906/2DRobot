"""Offline tests for seed-guided full-wafer centre refinement."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.wafer_center_refinement import refine_wafer_geometry_center


def _purple(hue: int = 140, value: int = 120) -> tuple[int, int, int]:
    hsv = np.array([[[hue, 210, value]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(channel) for channel in bgr)


def _patch(size: int = 256) -> np.ndarray:
    return np.full((size, size, 3), 185, dtype=np.uint8)


class WaferCenterRefinementTests(unittest.TestCase):
    def test_recovers_complete_off_center_square_geometry(self) -> None:
        patch = _patch()
        expected_center = (82.0, 151.0)
        box = cv2.boxPoints((expected_center, (78.0, 78.0), 7.0))
        cv2.fillConvexPoly(
            patch, np.round(box).astype(np.int32), _purple(), cv2.LINE_AA
        )

        result = refine_wafer_geometry_center(patch, (91.0, 148.0))

        self.assertTrue(result.success, result.reason)
        self.assertIsNotNone(result.center_patch_px)
        self.assertAlmostEqual(result.center_patch_px[0], expected_center[0], delta=1.0)
        self.assertAlmostEqual(result.center_patch_px[1], expected_center[1], delta=1.0)
        self.assertEqual(len(result.box_patch_px), 4)
        self.assertFalse(result.touches_boundary)
        self.assertEqual(result.to_json()["reason"], "ok")

    def test_larger_distant_component_cannot_steal_seed(self) -> None:
        patch = _patch()
        target_center = (69.0, 72.0)
        cv2.rectangle(patch, (41, 44), (97, 100), _purple(), -1)
        cv2.rectangle(patch, (135, 118), (238, 221), _purple(147), -1)

        result = refine_wafer_geometry_center(patch, (68.0, 71.0))

        self.assertTrue(result.success, result.reason)
        self.assertAlmostEqual(result.center_patch_px[0], target_center[0], delta=1.0)
        self.assertAlmostEqual(result.center_patch_px[1], target_center[1], delta=1.0)
        self.assertEqual(result.seed_distance_px, 0.0)

    def test_relative_lab_contrast_recovers_hue_shifted_tilted_wafer(self) -> None:
        patch = _patch()
        expected_center = (121.0, 137.0)
        box = cv2.boxPoints((expected_center, (82.0, 82.0), 27.0))
        cv2.fillConvexPoly(
            patch,
            np.round(box).astype(np.int32),
            _purple(hue=175, value=105),
            cv2.LINE_AA,
        )
        # A low-saturation glare stripe removes colour evidence through the
        # middle; the physical four-edge fit must retain the wafer yaw.
        cv2.line(patch, (94, 121), (148, 153), (205, 205, 205), 4, cv2.LINE_AA)

        result = refine_wafer_geometry_center(patch, expected_center)

        self.assertTrue(result.success, result.reason)
        self.assertAlmostEqual(result.center_patch_px[0], expected_center[0], delta=1.5)
        self.assertAlmostEqual(result.center_patch_px[1], expected_center[1], delta=1.5)
        self.assertAlmostEqual(result.yaw_deg, 27.0, delta=2.0)
        self.assertNotEqual(result.quadrilateral_fit_method, "none")
        self.assertGreater(result.quadrilateral_fit_iou, 0.70)

    def test_neutral_tray_shadow_is_not_silicon_contrast(self) -> None:
        patch = _patch()
        cv2.rectangle(patch, (70, 70), (186, 186), (75, 75, 75), -1)

        result = refine_wafer_geometry_center(patch, (128.0, 128.0))

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "no_plausible_chromatic_candidate")

    def test_component_touching_expanded_roi_boundary_is_rejected(self) -> None:
        patch = _patch()
        cv2.rectangle(patch, (0, 80), (78, 158), _purple(), -1)

        result = refine_wafer_geometry_center(patch, (44.0, 119.0))

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "touches_expanded_roi_boundary")
        self.assertTrue(result.touches_boundary)
        self.assertIsNone(result.center_patch_px)

    def test_no_candidate_is_rejected(self) -> None:
        patch = _patch()

        result = refine_wafer_geometry_center(patch, (128.0, 128.0))

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "no_plausible_chromatic_candidate")
        self.assertIsNone(result.center_patch_px)
        self.assertEqual(result.box_patch_px, ())


if __name__ == "__main__":
    unittest.main()
