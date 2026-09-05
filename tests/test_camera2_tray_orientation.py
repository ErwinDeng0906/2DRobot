from __future__ import annotations

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.camera2_tray_orientation import (  # noqa: E402
    normalize_tray_axis_angle_deg,
    observe_camera2_tray_orientation,
)


class Camera2TrayOrientationTests(unittest.TestCase):
    def test_normalizes_all_square_edges_to_same_axis(self) -> None:
        self.assertAlmostEqual(-4.5, normalize_tray_axis_angle_deg(-4.5))
        self.assertAlmostEqual(-4.5, normalize_tray_axis_angle_deg(85.5))
        self.assertAlmostEqual(-4.5, normalize_tray_axis_angle_deg(175.5))

    def test_decoded_rotated_marker_reports_visual_tray_angle(self) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(dictionary, 25, 180)
        canvas = np.full((420, 420), 255, dtype=np.uint8)
        canvas[120:300, 120:300] = marker
        transform = cv2.getRotationMatrix2D((210.0, 210.0), 4.25, 1.0)
        rotated = cv2.warpAffine(
            canvas,
            transform,
            (420, 420),
            flags=cv2.INTER_LINEAR,
            borderValue=255,
        )
        image = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
        observation = observe_camera2_tray_orientation(
            image,
            frame_sequence=7,
            captured_monotonic_s=12.5,
        )
        self.assertTrue(observation.accepted, observation.rejection_reasons)
        self.assertEqual((25,), observation.marker_ids)
        self.assertAlmostEqual(-4.25, observation.angle_error_deg, delta=0.35)

    def test_blank_frame_fails_closed(self) -> None:
        observation = observe_camera2_tray_orientation(
            np.full((240, 320, 3), 220, dtype=np.uint8),
            frame_sequence=8,
            captured_monotonic_s=13.0,
        )
        self.assertFalse(observation.accepted)
        self.assertIsNone(observation.angle_error_deg)


if __name__ == "__main__":
    unittest.main()
