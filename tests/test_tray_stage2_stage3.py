"""Offline tests for Tray Frame geometry and ^C T_T estimation."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.tray_board_geometry import (
    MARKER_LABEL_TO_ID,
    MARKER_SIDE_MM,
    build_tray_board_geometry,
)
from scara.vision.tray_pose_estimator import (
    CameraIntrinsics,
    TrayBoardPoseEstimator,
    invert_transform,
    load_camera_intrinsics,
    make_transform_C_T,
    transform_points,
)
from scara.vision.tray_pose_tracker import (
    TrayPoseTracker,
    TrayPoseTrackerConfig,
)


def _complete_preset_path() -> Path:
    candidates = [
        PROJECT_ROOT / "scara_presets.json",
        Path(
            r"C:\Users\Admin\Desktop\2D robot\RobotArm_SCARA_Control 0812\scara_presets.json"
        ),
    ]
    required = {"P00 float", "P05 float", "P50 float", "P55 float"}
    for label in "ABCDEFGH":
        required.update((label, f"{label}ul", f"{label}dl"))
    for path in candidates:
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if required <= set(raw):
            return path
    raise RuntimeError("No complete Stage-2 scara_presets.json found")


def _synthetic_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        K=np.array(
            [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        dist_coeffs=np.zeros((5, 1), dtype=np.float64),
        image_size=(1280, 720),
        source_path="synthetic",
        calibration_status="success",
        global_rms_px=0.0,
    )


class _SyntheticDetector:
    def __init__(self, corners: list[np.ndarray], ids: list[int]) -> None:
        self._corners = corners
        self._ids = np.asarray(ids, dtype=np.int32).reshape(-1, 1)

    def detectMarkers(self, _image: np.ndarray):
        return self._corners, self._ids, []


class TrayGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = build_tray_board_geometry(_complete_preset_path())

    def test_frame_slots_and_marker_squares(self) -> None:
        self.assertTrue(self.geometry["validation"]["valid"])
        rotation = np.asarray(
            self.geometry["tray_frame"]["rotation_mechanical_from_tray"]
        )
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=10)
        self.assertEqual(len(self.geometry["slots"]), 36)
        self.assertEqual(self.geometry["slots"]["P00"], [0.0, 0.0, 0.0])
        self.assertEqual(self.geometry["slots"]["P55"], [-125.0, -125.0, 0.0])
        self.assertEqual(
            {label: row["id"] for label, row in self.geometry["markers"].items()},
            MARKER_LABEL_TO_ID,
        )
        for marker in self.geometry["markers"].values():
            corners = np.asarray(marker["corners_T_mm"])
            edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
            np.testing.assert_allclose(edges, MARKER_SIDE_MM, atol=1e-9)
            np.testing.assert_allclose(corners.mean(axis=0), marker["center_T_mm"])
            self.assertGreater(marker["center_T_mm"][2], 0.0)

    def test_transform_round_trip(self) -> None:
        rvec = np.array([[0.1], [-0.2], [0.3]])
        tvec = np.array([[10.0], [20.0], [300.0]])
        transform = make_transform_C_T(rvec, tvec)
        inverse = invert_transform(transform)
        np.testing.assert_allclose(inverse @ transform, np.eye(4), atol=1e-12)
        points = np.array([[0.0, 0.0, 0.0], [-125.0, -125.0, 0.0]])
        np.testing.assert_allclose(
            transform_points(inverse, transform_points(transform, points)),
            points,
            atol=1e-10,
        )

    def test_unapproved_intrinsics_are_rejected_by_default(self) -> None:
        payload = {
            "status": "rejected_pose_diversity",
            "K": [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]],
            "distCoeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
            "image_resolution": {"width": 1280, "height": 720},
            "global_rms_px": 0.3,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "camera1_intrinsics.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不是已批准的 success"):
                load_camera_intrinsics(path)
            loaded = load_camera_intrinsics(
                path, allow_unapproved_status=True
            )
            self.assertEqual(loaded.calibration_status, "rejected_pose_diversity")


class TrayPoseEstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = build_tray_board_geometry(_complete_preset_path())
        cls.intrinsics = _synthetic_intrinsics()

    def _synthetic_estimator(
        self,
        noise_px: float = 0.0,
        corrupt_marker_id: int | None = None,
    ) -> tuple[TrayBoardPoseEstimator, np.ndarray, np.ndarray]:
        estimator = TrayBoardPoseEstimator(self.geometry, self.intrinsics)
        # Down-looking camera: +Z_T points from the tray toward the camera, so
        # it maps approximately to -Z_C.  Compose small tilt/yaw with Rx(pi).
        base_down = np.diag([1.0, -1.0, -1.0])
        small_rvec = np.array(
            [[math.radians(4.0)], [math.radians(-7.0)], [0.2]]
        )
        small_rotation, _ = cv2.Rodrigues(small_rvec)
        rvec_true, _ = cv2.Rodrigues(small_rotation @ base_down)
        tvec_true = np.array([[20.0], [-15.0], [430.0]])
        rng = np.random.default_rng(20260814)
        corners = []
        ids = []
        for marker in self.geometry["markers"].values():
            marker_id = int(marker["id"])
            projected, _ = cv2.projectPoints(
                np.asarray(marker["corners_T_mm"], dtype=np.float64),
                rvec_true,
                tvec_true,
                self.intrinsics.K,
                self.intrinsics.dist_coeffs,
            )
            projected = projected.reshape(4, 2)
            if noise_px:
                projected += rng.normal(0.0, noise_px, projected.shape)
            if marker_id == corrupt_marker_id:
                projected += np.array([35.0, -25.0])
            corners.append(projected.astype(np.float32).reshape(1, 4, 2))
            ids.append(marker_id)
        estimator.detector = _SyntheticDetector(corners, ids)
        return estimator, rvec_true, tvec_true

    def test_recovers_transform(self) -> None:
        estimator, rvec_true, tvec_true = self._synthetic_estimator(noise_px=0.05)
        result = estimator.estimate(np.zeros((720, 1280, 3), dtype=np.uint8))
        self.assertTrue(result.success)
        self.assertTrue(result.quality_passed, result.failure_reason)
        expected = make_transform_C_T(rvec_true, tvec_true)
        self.assertLess(np.linalg.norm(result.T_C_T[:3, 3] - expected[:3, 3]), 0.2)
        relative = expected[:3, :3].T @ result.T_C_T[:3, :3]
        angle = math.degrees(
            math.acos(float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
        )
        self.assertLess(angle, 0.1)

    def test_rejects_corrupt_whole_marker(self) -> None:
        estimator, _rvec, _tvec = self._synthetic_estimator(
            noise_px=0.05, corrupt_marker_id=4
        )
        result = estimator.estimate(np.zeros((720, 1280, 3), dtype=np.uint8))
        self.assertTrue(result.success)
        self.assertTrue(result.quality_passed, result.failure_reason)
        self.assertIn(4, result.rejected_marker_ids)
        self.assertNotIn(4, result.used_marker_ids)

    def test_tracker_smooths_and_rejects_jump(self) -> None:
        estimator, _rvec, _tvec = self._synthetic_estimator(noise_px=0.0)
        tracker = TrayPoseTracker(
            estimator,
            TrayPoseTrackerConfig(
                translation_alpha=0.5,
                rotation_alpha=0.5,
                maximum_translation_jump_mm=5.0,
                maximum_rotation_jump_deg=5.0,
            ),
        )
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        first = tracker.update(image)
        self.assertTrue(first.accepted_by_tracker)
        # Shift every detected corner strongly; raw pose may pass geometry, but
        # its camera translation jump must be rejected by the temporal gate.
        shifted = [corner + np.array([[[120.0, 0.0]]], dtype=np.float32) for corner in estimator.detector._corners]
        estimator.detector = _SyntheticDetector(
            shifted, list(range(1, 9))
        )
        second = tracker.update(image)
        self.assertFalse(second.accepted_by_tracker)
        self.assertIsNotNone(second.filtered_T_C_T)


if __name__ == "__main__":
    unittest.main()
