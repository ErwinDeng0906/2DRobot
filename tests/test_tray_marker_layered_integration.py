"""Offline tests for the layered tray-marker integration."""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.slot_marker_observation import (
    SlotMarkerEvidence,
    SlotProjection,
    build_slot_projections,
    detect_aruco_observations,
    load_slot_marker_layout,
)
from scara.vision.tray_occupancy import SlotState, decide_slot_state
from scara.vision.tray_pose_estimator import (
    CameraIntrinsics,
    TrayBoardPoseEstimator,
    TrayPoseEstimate,
    invert_transform,
    make_transform_C_T,
)
from scara.vision.tray_vision_fusion import (
    TrayVisionAnalyzer,
    image_pixel_to_tray_plane,
    nearest_metric_slot,
)
from scara.vision.wafer_shape_quality import WaferObservation, analyze_wafer_patch


GEOMETRY_PATH = PROJECT_ROOT / "src/scara/calib/tray_board_geometry.json"
LAYOUT_PATH = PROJECT_ROOT / "tools/tray_marker_detector_v2/tray_marker_layout.json"


def _geometry() -> dict:
    return json.loads(GEOMETRY_PATH.read_text(encoding="utf-8"))


def _intrinsics() -> CameraIntrinsics:
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


def _valid_pose(image: np.ndarray) -> TrayPoseEstimate:
    rotation_C_T = np.diag([1.0, -1.0, -1.0])
    rvec, _ = cv2.Rodrigues(rotation_C_T)
    tvec = np.array([[60.0], [-60.0], [450.0]], dtype=np.float64)
    transform_C_T = make_transform_C_T(rvec, tvec)
    transform_T_C = invert_transform(transform_C_T)
    return TrayPoseEstimate(
        success=True,
        quality_passed=True,
        failure_reason=None,
        visible_marker_ids=(1, 2, 3, 4),
        used_marker_ids=(1, 2, 3, 4),
        rejected_marker_ids=(),
        ransac_inlier_corner_count=16,
        object_span_mm=180.0,
        reprojection_rms_px=0.2,
        per_marker_rms_px={1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2},
        rvec_C_T=rvec,
        tvec_C_T_mm=tvec,
        T_C_T=transform_C_T,
        T_T_C=transform_T_C,
        camera_position_T_mm=transform_T_C[:3, 3].copy(),
        minimum_object_depth_C_mm=440.0,
        annotated_image=image.copy(),
    )


def _failed_pose(image: np.ndarray) -> TrayPoseEstimate:
    return TrayPoseEstimate(
        success=False,
        quality_passed=False,
        failure_reason="synthetic pose rejection",
        visible_marker_ids=(),
        used_marker_ids=(),
        rejected_marker_ids=(),
        ransac_inlier_corner_count=0,
        object_span_mm=0.0,
        reprojection_rms_px=None,
        per_marker_rms_px={},
        rvec_C_T=None,
        tvec_C_T_mm=None,
        T_C_T=None,
        T_T_C=None,
        camera_position_T_mm=None,
        minimum_object_depth_C_mm=None,
        annotated_image=image.copy(),
    )


def _purple(hue: int = 140, value: int = 115) -> tuple[int, int, int]:
    hsv = np.array([[[hue, 210, value]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(channel) for channel in bgr)


def _draw_square(
    image: np.ndarray,
    center: tuple[float, float],
    side: float,
    angle_deg: float,
    color: tuple[int, int, int],
) -> None:
    box = cv2.boxPoints((center, (side, side), angle_deg))
    cv2.fillConvexPoly(image, np.round(box).astype(np.int32), color, cv2.LINE_AA)


class SlotProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = _geometry()
        cls.intrinsics = _intrinsics()
        cls.estimator = TrayBoardPoseEstimator(cls.geometry, cls.intrinsics)
        cls.layout = load_slot_marker_layout(LAYOUT_PATH)

    def test_legacy_layout_is_a_fixed_identity_table(self) -> None:
        self.assertEqual(self.layout.dictionary_name, "DICT_4X4_50")
        self.assertEqual(self.layout.metric_slot_transform, "rot270")
        self.assertEqual(len(self.layout.marker_id_by_slot), 36)
        self.assertEqual(self.layout.marker_id_by_slot["P00"], 9)
        self.assertEqual(self.layout.marker_id_by_slot["P55"], 44)

    def test_metric_pose_projects_all_slots_without_resorting(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        pose = _valid_pose(image)
        projected = build_slot_projections(
            self.geometry, self.estimator, pose, image.shape
        )
        self.assertEqual(len(projected), 36)
        self.assertEqual(projected["P00"].row, 0)
        self.assertEqual(projected["P55"].column, 5)
        self.assertTrue(all(slot.image_coverage_ratio > 0.999 for slot in projected.values()))

    def test_pixel_to_tray_round_trip_and_nearest_slot(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        pose = _valid_pose(image)
        target_T = np.asarray(self.geometry["slots"]["P23"], dtype=np.float64)
        pixel = self.estimator.project_tray_points(target_T.reshape(1, 3), pose)[0]
        recovered = image_pixel_to_tray_plane(pixel, pose, self.intrinsics)
        np.testing.assert_allclose(recovered, target_T, atol=1e-7)
        slot_key, distance = nearest_metric_slot(recovered, self.geometry)
        self.assertEqual(slot_key, "P23")
        self.assertLess(distance, 1e-7)

    def test_multiscale_marker_observation(self) -> None:
        image = np.full((360, 480), 220, dtype=np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(dictionary, 39, 72)
        image[130:202, 210:282] = marker
        observations = detect_aruco_observations(image, "DICT_4X4_50")
        self.assertIn(39, observations)
        self.assertGreater(observations[39].square_quality, 0.95)


class WaferQualityTests(unittest.TestCase):
    def test_centered_rotated_square_is_normal_and_angle_is_accurate(self) -> None:
        patch = np.full((192, 192, 3), 185, dtype=np.uint8)
        _draw_square(patch, (96.0, 96.0), 92.0, 6.0, _purple())
        result = analyze_wafer_patch(patch)
        self.assertTrue(result.found)
        self.assertEqual(result.quality, "normal", result.flags)
        self.assertAlmostEqual(result.yaw_relative_to_tray_deg, 6.0, delta=1.5)

    def test_overlapping_squares_are_abnormal(self) -> None:
        patch = np.full((192, 192, 3), 185, dtype=np.uint8)
        _draw_square(patch, (75.0, 96.0), 82.0, 0.0, _purple(140, 100))
        _draw_square(patch, (119.0, 96.0), 82.0, 0.0, _purple(150, 135))
        result = analyze_wafer_patch(patch)
        self.assertTrue(result.found)
        self.assertEqual(result.quality, "abnormal")
        self.assertTrue(
            {"non_square_aspect", "irregular_outline", "internal_overlap_edges"}
            & set(result.flags),
            result.flags,
        )

    def test_black_white_marker_is_not_a_wafer(self) -> None:
        patch = np.full((192, 192, 3), 185, dtype=np.uint8)
        cv2.rectangle(patch, (48, 48), (144, 144), (5, 5, 5), -1)
        cv2.rectangle(patch, (66, 66), (92, 92), (245, 245, 245), -1)
        cv2.rectangle(patch, (104, 104), (130, 130), (245, 245, 245), -1)
        result = analyze_wafer_patch(patch)
        self.assertFalse(result.found)
        self.assertIn("no_chromatic_candidate", result.flags)


class OccupancyDecisionTests(unittest.TestCase):
    @staticmethod
    def _projection(coverage: float = 1.0) -> SlotProjection:
        return SlotProjection(
            slot_key="P00",
            row=0,
            column=0,
            center_T_mm=(0.0, 0.0, -2.0),
            center_px=(100.0, 100.0),
            polygon_T_mm=((1.0, 1.0, -2.0), (1.0, -1.0, -2.0), (-1.0, -1.0, -2.0), (-1.0, 1.0, -2.0)),
            polygon_px=((80.0, 80.0), (120.0, 80.0), (120.0, 120.0), (80.0, 120.0)),
            image_coverage_ratio=coverage,
            projected_area_px=1600.0,
        )

    @staticmethod
    def _marker(decoded: bool = False, marker_like: bool = False) -> SlotMarkerEvidence:
        return SlotMarkerEvidence(
            slot_key="P00",
            expected_marker_id=39,
            decoded=decoded,
            decoded_marker_id=39 if decoded else None,
            marker_like_visible=marker_like,
            center_error_px=0.0 if decoded else None,
            detection_quality=1.0 if decoded else None,
            pattern_features={},
        )

    def test_partial_view_is_not_called_missing(self) -> None:
        decision = decide_slot_state(
            self._projection(0.55), self._marker(), WaferObservation.not_found()
        )
        self.assertEqual(decision.state, SlotState.OUT_OF_VIEW)
        self.assertFalse(decision.safe_to_use_as_empty)

    def test_marker_and_wafer_states_remain_separate(self) -> None:
        empty = decide_slot_state(
            self._projection(), self._marker(decoded=True), WaferObservation.not_found()
        )
        self.assertEqual(empty.state, SlotState.EMPTY)
        patch = np.full((192, 192, 3), 185, dtype=np.uint8)
        _draw_square(patch, (96.0, 96.0), 92.0, 3.0, _purple())
        wafer = analyze_wafer_patch(patch)
        occupied = decide_slot_state(self._projection(), self._marker(), wafer)
        self.assertEqual(occupied.state, SlotState.OCCUPIED)

    def test_no_evidence_is_unknown(self) -> None:
        decision = decide_slot_state(
            self._projection(), self._marker(), WaferObservation.not_found()
        )
        self.assertEqual(decision.state, SlotState.UNKNOWN)
        self.assertFalse(decision.safe_to_use_as_empty)


class FusionFailClosedTests(unittest.TestCase):
    def test_pose_rejection_stops_slot_analysis(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)

        class RejectedEstimator:
            intrinsics = _intrinsics()

            @staticmethod
            def estimate(_image: np.ndarray) -> TrayPoseEstimate:
                return _failed_pose(_image)

        analyzer = TrayVisionAnalyzer(
            RejectedEstimator(), _geometry(), load_slot_marker_layout(LAYOUT_PATH)
        )
        result = analyzer.analyze(image)
        self.assertFalse(result.quality_passed)
        self.assertFalse(result.coordinate_mapping_allowed)
        self.assertEqual(len(result.slots), 0)
        self.assertEqual(result.summary["unknown"], 36)


if __name__ == "__main__":
    unittest.main()
