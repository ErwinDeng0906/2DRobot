"""Offline tests for the layered tray-marker integration."""

from __future__ import annotations

import json
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.slot_marker_observation import (
    ArucoObservation,
    SlotMarkerEvidence,
    SlotProjection,
    apply_slot_marker_registration,
    build_slot_projections,
    detect_aruco_observations,
    estimate_slot_marker_registration,
    load_slot_marker_layout,
    patch_points_to_image,
    warp_slot_patch,
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
    tracked_pose_estimate,
    wafer_patch_center_to_tray,
)
from scara.vision.tray_pose_tracker import TrackedTrayPose
from scara.vision.wafer_shape_quality import (
    DEFAULT_WAFER_QUALITY,
    WaferObservation,
    analyze_wafer_patch,
)
from scara.vision.wafer_correction_target import (
    SLOT_QUADRILATERAL_CENTER_SOURCE,
    extract_outside_wafer_candidates,
)


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

    def test_decoded_slot_markers_correct_current_frame_projection(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        projected = build_slot_projections(
            self.geometry, self.estimator, _valid_pose(image), image.shape
        )
        expected_homography = np.asarray(
            [
                [0.985, -0.012, 5.0],
                [0.004, 0.979, 4.0],
                [-0.00001, -0.00002, 1.0],
            ],
            dtype=np.float64,
        )
        observations = {}
        for slot_key in ("P00", "P02", "P11", "P30", "P41", "P55"):
            marker_id = self.layout.marker_id_by_slot[slot_key]
            source = np.asarray(
                [[projected[slot_key].center_px]], dtype=np.float32
            )
            target = cv2.perspectiveTransform(
                source, expected_homography
            ).reshape(2)
            observations[marker_id] = ArucoObservation(
                marker_id=marker_id,
                center_px=(float(target[0]), float(target[1])),
                corners_px=(),
                angle_deg=0.0,
                perimeter_px=40.0,
                area_px=100.0,
                square_quality=1.0,
            )

        fitted, diagnostics = estimate_slot_marker_registration(
            projected, self.layout, observations
        )
        self.assertTrue(diagnostics["applied"], diagnostics)
        corrected = apply_slot_marker_registration(
            projected, fitted, image.shape
        )
        for slot_key in ("P00", "P23", "P55"):
            source = np.asarray(
                [[projected[slot_key].center_px]], dtype=np.float32
            )
            expected = cv2.perspectiveTransform(
                source, expected_homography
            ).reshape(2)
            np.testing.assert_allclose(
                corrected[slot_key].center_px, expected, atol=0.05
            )


class WaferQualityTests(unittest.TestCase):
    def test_rejected_oversized_candidate_keeps_measured_area_for_diagnostics(self) -> None:
        patch = np.full((192, 192, 3), (80, 20, 80), dtype=np.uint8)
        observation = analyze_wafer_patch(patch)
        self.assertFalse(observation.found)
        self.assertIn("candidate_area_out_of_range", observation.flags)
        self.assertGreater(observation.area_ratio, DEFAULT_WAFER_QUALITY.maximum_area_ratio)

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
        self.assertNotIn("stacked_geometry_confirmed", result.flags)
        self.assertEqual(result.secondary_boxes_patch_px, ())

    def test_reflection_edges_are_warning_not_confirmed_stacking(self) -> None:
        patch = np.full((192, 192, 3), 185, dtype=np.uint8)
        _draw_square(patch, (96.0, 96.0), 92.0, 0.0, _purple(140, 100))
        cv2.rectangle(patch, (91, 50), (101, 142), _purple(140, 220), -1)
        result = analyze_wafer_patch(
            patch,
            replace(DEFAULT_WAFER_QUALITY, stacking_detection_enabled=True),
        )
        self.assertTrue(result.found)
        self.assertEqual(result.quality, "warning", result.flags)
        self.assertIn("internal_overlap_edges", result.flags)
        self.assertNotIn("stacked_geometry_confirmed", result.flags)
        self.assertEqual(result.secondary_boxes_patch_px, ())

    def test_l_corner_confirms_stacking_and_produces_second_outline(self) -> None:
        patch = np.full((192, 192, 3), 185, dtype=np.uint8)
        _draw_square(patch, (78.0, 78.0), 80.0, 0.0, _purple(140, 100))
        _draw_square(patch, (112.0, 112.0), 80.0, 0.0, _purple(150, 145))
        result = analyze_wafer_patch(
            patch,
            replace(DEFAULT_WAFER_QUALITY, stacking_detection_enabled=True),
        )
        self.assertTrue(result.found)
        self.assertEqual(result.quality, "abnormal", result.flags)
        self.assertIn("l_shaped_overlap_corner", result.flags)
        self.assertIn("stacked_geometry_confirmed", result.flags)
        self.assertEqual(len(result.secondary_boxes_patch_px), 1)
        self.assertEqual(len(result.secondary_boxes_patch_px[0]), 4)

    def test_second_quadrilateral_confirms_stacking(self) -> None:
        patch = np.full((192, 192, 3), 185, dtype=np.uint8)
        _draw_square(patch, (58.0, 96.0), 52.0, 0.0, _purple(140, 100))
        _draw_square(patch, (134.0, 96.0), 52.0, 0.0, _purple(150, 145))
        result = analyze_wafer_patch(
            patch,
            replace(DEFAULT_WAFER_QUALITY, stacking_detection_enabled=True),
        )
        self.assertTrue(result.found)
        self.assertIn("second_quadrilateral", result.flags)
        self.assertIn("stacked_geometry_confirmed", result.flags)
        self.assertEqual(len(result.secondary_boxes_patch_px), 1)
        self.assertEqual(len(result.to_json()["secondary_boxes_patch_px"]), 1)

    def test_stacking_evidence_is_dormant_by_default(self) -> None:
        patch = np.full((192, 192, 3), 185, dtype=np.uint8)
        _draw_square(patch, (78.0, 78.0), 80.0, 0.0, _purple(140, 100))
        _draw_square(patch, (112.0, 112.0), 80.0, 0.0, _purple(150, 145))
        result = analyze_wafer_patch(patch)
        self.assertNotIn("l_shaped_overlap_corner", result.flags)
        self.assertNotIn("second_quadrilateral", result.flags)
        self.assertNotIn("stacked_geometry_confirmed", result.flags)
        self.assertEqual(0, result.internal_line_count)
        self.assertEqual(0.0, result.internal_line_score)
        self.assertEqual((), result.secondary_boxes_patch_px)

    def test_boundary_crossing_wafer_records_outside_slot_evidence(self) -> None:
        patch = np.full((192, 192, 3), 185, dtype=np.uint8)
        _draw_square(patch, (170.0, 96.0), 72.0, 0.0, _purple())
        result = analyze_wafer_patch(patch)
        self.assertTrue(result.found)
        self.assertTrue(result.outside_slot)
        self.assertIn("outside_slot", result.flags)
        self.assertTrue(result.to_json()["outside_slot"])

    def test_black_white_marker_is_not_a_wafer(self) -> None:
        patch = np.full((192, 192, 3), 185, dtype=np.uint8)
        cv2.rectangle(patch, (48, 48), (144, 144), (5, 5, 5), -1)
        cv2.rectangle(patch, (66, 66), (92, 92), (245, 245, 245), -1)
        cv2.rectangle(patch, (104, 104), (130, 130), (245, 245, 245), -1)
        result = analyze_wafer_patch(patch)
        self.assertFalse(result.found)
        self.assertIn("no_chromatic_candidate", result.flags)


class WaferMetricCentreTests(unittest.TestCase):
    @staticmethod
    def _projection() -> SlotProjection:
        return SlotProjection(
            slot_key="P23",
            row=2,
            column=3,
            center_T_mm=(10.0, 20.0, -2.0),
            center_px=(320.0, 240.0),
            polygon_T_mm=(
                (21.5, 31.5, -2.0),
                (21.5, 8.5, -2.0),
                (-1.5, 8.5, -2.0),
                (-1.5, 31.5, -2.0),
            ),
            polygon_px=((200.0, 120.0), (440.0, 120.0), (440.0, 360.0), (200.0, 360.0)),
            image_coverage_ratio=1.0,
            projected_area_px=57600.0,
        )

    def test_patch_centre_to_tray_uses_fixed_metric_orientation(self) -> None:
        projection = self._projection()
        center_T, offset_T, distance = wafer_patch_center_to_tray(
            (120.0, 80.0), projection
        )
        expected = np.array(
            [
                11.5 * (1.0 - 2.0 * 80.0 / 191.0),
                11.5 * (1.0 - 2.0 * 120.0 / 191.0),
            ]
        )
        np.testing.assert_allclose(offset_T, expected, atol=1e-12)
        np.testing.assert_allclose(center_T[:2], np.array([10.0, 20.0]) + expected)
        self.assertAlmostEqual(distance, float(np.linalg.norm(expected)), places=12)

    def test_fusion_exports_metric_and_image_wafer_centres(self) -> None:
        image = np.full((720, 1280, 3), 185, dtype=np.uint8)
        geometry = _geometry()
        intrinsics = _intrinsics()
        estimator = TrayBoardPoseEstimator(geometry, intrinsics)
        provisional_pose = _valid_pose(image)
        projection = build_slot_projections(
            geometry, estimator, provisional_pose, image.shape
        )["P23"]
        _patch, image_to_patch = warp_slot_patch(image, projection)
        patch_box = cv2.boxPoints(((120.0, 80.0), (82.0, 82.0), 4.0))
        image_box = patch_points_to_image(patch_box, image_to_patch)
        cv2.fillConvexPoly(
            image,
            np.round(image_box).astype(np.int32),
            _purple(),
            cv2.LINE_AA,
        )
        pose = _valid_pose(image)
        analyzer = TrayVisionAnalyzer(
            estimator, geometry, load_slot_marker_layout(LAYOUT_PATH)
        )
        result = analyzer.analyze(image, pose=pose)
        analysis = next(
            slot for slot in result.slots if slot.projection.slot_key == "P23"
        )
        self.assertTrue(analysis.wafer.found, analysis.wafer.flags)
        self.assertIsNotNone(analysis.wafer_center_image_px)
        self.assertIsNotNone(analysis.wafer_center_T_mm)
        self.assertIsNotNone(analysis.wafer_offset_T_mm)
        self.assertIsNotNone(analysis.wafer_offset_distance_mm)
        expected_image_center = patch_points_to_image(
            np.asarray([analysis.wafer.center_patch_px], dtype=np.float32),
            image_to_patch,
        )[0]
        np.testing.assert_allclose(
            analysis.wafer_center_image_px, expected_image_center, atol=0.25
        )
        payload = analysis.to_json()
        self.assertEqual(payload["wafer_center_image_px"], list(analysis.wafer_center_image_px))
        self.assertEqual(payload["wafer_center_T_mm"], list(analysis.wafer_center_T_mm))
        self.assertEqual(payload["wafer_offset_T_mm"], list(analysis.wafer_offset_T_mm))
        self.assertAlmostEqual(
            payload["wafer_offset_distance_mm"],
            analysis.wafer_offset_distance_mm,
        )
        empty_slot = next(
            slot for slot in result.slots if slot.projection.slot_key == "P00"
        )
        self.assertFalse(empty_slot.wafer.found)
        self.assertIsNone(empty_slot.wafer_center_image_px)
        self.assertIsNone(empty_slot.wafer_center_T_mm)
        self.assertIsNone(empty_slot.wafer_offset_T_mm)
        self.assertIsNone(empty_slot.wafer_offset_distance_mm)

    def test_outside_wafer_uses_expanded_full_contour_center_for_correction(self) -> None:
        image = np.full((720, 1280, 3), 185, dtype=np.uint8)
        geometry = _geometry()
        intrinsics = _intrinsics()
        estimator = TrayBoardPoseEstimator(geometry, intrinsics)
        provisional_pose = _valid_pose(image)
        projection = build_slot_projections(
            geometry, estimator, provisional_pose, image.shape
        )["P23"]
        _patch, image_to_patch = warp_slot_patch(image, projection)
        expected_patch_center = (174.0, 96.0)
        patch_box = cv2.boxPoints(
            (expected_patch_center, (78.0, 78.0), 0.0)
        )
        image_box = patch_points_to_image(patch_box, image_to_patch)
        cv2.fillConvexPoly(
            image,
            np.round(image_box).astype(np.int32),
            _purple(),
            cv2.LINE_AA,
        )
        pose = _valid_pose(image)
        analyzer = TrayVisionAnalyzer(
            estimator, geometry, load_slot_marker_layout(LAYOUT_PATH)
        )
        result = analyzer.analyze(image, pose=pose)
        analysis = next(
            slot for slot in result.slots if slot.projection.slot_key == "P23"
        )
        self.assertEqual(analysis.decision.state, SlotState.OUTSIDE_SLOT)
        self.assertTrue(
            analysis.wafer_correction_center_valid,
            analysis.wafer_correction_center_reason,
        )
        self.assertTrue(analysis.wafer_center_refinement["success"])
        expected_center_T, _offset, _distance = wafer_patch_center_to_tray(
            expected_patch_center, projection
        )
        np.testing.assert_allclose(
            analysis.wafer_center_T_mm,
            expected_center_T,
            atol=0.30,
        )
        self.assertEqual(
            analysis.to_json()["wafer_correction_center_reason"], "ok"
        )
        self.assertTrue(analysis.wafer_correction_outside_slot)

    def test_cross_slot_low_area_fragments_trigger_independent_refinement(self) -> None:
        image = np.full((720, 1280, 3), 185, dtype=np.uint8)
        geometry = _geometry()
        intrinsics = _intrinsics()
        estimator = TrayBoardPoseEstimator(geometry, intrinsics)
        provisional_pose = _valid_pose(image)
        projection = build_slot_projections(
            geometry, estimator, provisional_pose, image.shape
        )["P23"]
        _patch, image_to_patch = warp_slot_patch(image, projection)
        # The centre is halfway between P23/P24.  Each 23 mm slot patch sees
        # less than the legacy 7.5% minimum, while the expanded ROI sees the
        # complete wafer.
        expected_patch_center = (199.3, 96.0)
        image_box = patch_points_to_image(
            cv2.boxPoints((expected_patch_center, (72.0, 72.0), 0.0)),
            image_to_patch,
        )
        cv2.fillConvexPoly(
            image,
            np.round(image_box).astype(np.int32),
            _purple(),
            cv2.LINE_AA,
        )

        analyzer = TrayVisionAnalyzer(
            estimator, geometry, load_slot_marker_layout(LAYOUT_PATH)
        )
        result = analyzer.analyze(image, pose=_valid_pose(image))
        adjacent = [
            slot
            for slot in result.slots
            if slot.projection.slot_key in {"P23", "P24"}
        ]
        self.assertEqual(2, len(adjacent))
        self.assertTrue(all(not slot.wafer.found for slot in adjacent))
        self.assertTrue(
            all(slot.decision.state is SlotState.UNKNOWN for slot in adjacent)
        )
        self.assertTrue(
            all(slot.wafer.area_ratio < 0.075 for slot in adjacent)
        )
        self.assertTrue(
            all(slot.wafer_correction_outside_slot for slot in adjacent)
        )
        self.assertTrue(
            all(slot.wafer_correction_center_valid for slot in adjacent)
        )
        expected_center_T, _offset, _distance = wafer_patch_center_to_tray(
            expected_patch_center, projection
        )
        for slot in adjacent:
            np.testing.assert_allclose(
                slot.wafer_center_T_mm,
                expected_center_T,
                atol=0.35,
            )
            gates = slot.wafer_center_refinement["full_contour_gates"]
            self.assertTrue(gates["physical_size_passed"])
            self.assertTrue(gates["outside_nearest_slot"])

    def test_normal_wafer_in_adjacent_slot_is_not_correction_outside(self) -> None:
        image = np.full((720, 1280, 3), 185, dtype=np.uint8)
        geometry = _geometry()
        intrinsics = _intrinsics()
        estimator = TrayBoardPoseEstimator(geometry, intrinsics)
        provisional_pose = _valid_pose(image)
        projection = build_slot_projections(
            geometry, estimator, provisional_pose, image.shape
        )["P24"]
        _patch, image_to_patch = warp_slot_patch(image, projection)
        image_box = patch_points_to_image(
            cv2.boxPoints(((96.0, 96.0), (78.0, 78.0), 0.0)),
            image_to_patch,
        )
        cv2.fillConvexPoly(
            image,
            np.round(image_box).astype(np.int32),
            _purple(),
            cv2.LINE_AA,
        )

        analyzer = TrayVisionAnalyzer(
            estimator, geometry, load_slot_marker_layout(LAYOUT_PATH)
        )
        result = analyzer.analyze(image, pose=_valid_pose(image))
        occupied = next(
            slot for slot in result.slots if slot.projection.slot_key == "P24"
        )
        self.assertEqual(SlotState.OCCUPIED, occupied.decision.state)
        self.assertFalse(
            any(slot.wafer_correction_outside_slot for slot in result.slots)
        )

    def test_physical_19_9mm_slot_boundary_is_not_the_23mm_crop(self) -> None:
        image = np.full((720, 1280, 3), 185, dtype=np.uint8)
        geometry = _geometry()
        intrinsics = _intrinsics()
        estimator = TrayBoardPoseEstimator(geometry, intrinsics)
        provisional_pose = _valid_pose(image)
        projection = build_slot_projections(
            geometry, estimator, provisional_pose, image.shape
        )["P23"]
        _patch, image_to_patch = warp_slot_patch(image, projection)
        # The expanded quadrilateral is evaluated against the calibrated
        # 19.9 mm slot, independently of the larger 23 mm analysis crop.
        image_box = patch_points_to_image(
            cv2.boxPoints(((120.0, 96.0), (148.0, 148.0), 0.0)),
            image_to_patch,
        )
        cv2.fillConvexPoly(
            image,
            np.round(image_box).astype(np.int32),
            _purple(),
            cv2.LINE_AA,
        )

        analyzer = TrayVisionAnalyzer(
            estimator, geometry, load_slot_marker_layout(LAYOUT_PATH)
        )
        result = analyzer.analyze(image, pose=_valid_pose(image))
        analysis = next(
            slot for slot in result.slots if slot.projection.slot_key == "P23"
        )

        self.assertTrue(analysis.wafer.found)
        self.assertEqual(SlotState.OUTSIDE_SLOT, analysis.decision.state)
        self.assertTrue(analysis.wafer_correction_outside_slot)
        gates = analysis.wafer_center_refinement["full_contour_gates"]
        self.assertAlmostEqual(11.5, gates["patch_half_extent_mm"])
        self.assertAlmostEqual(9.95, gates["slot_boundary_half_extent_mm"])
        self.assertEqual("outside", gates["boundary_classification"])
        self.assertGreater(gates["maximum_corner_overflow_mm"], 0.0)

    def test_expanded_refinement_rejects_oversize_physical_contour(self) -> None:
        image = np.full((720, 1280, 3), 185, dtype=np.uint8)
        geometry = _geometry()
        intrinsics = _intrinsics()
        estimator = TrayBoardPoseEstimator(geometry, intrinsics)
        provisional_pose = _valid_pose(image)
        projection = build_slot_projections(
            geometry, estimator, provisional_pose, image.shape
        )["P23"]
        _patch, image_to_patch = warp_slot_patch(image, projection)
        image_box = patch_points_to_image(
            cv2.boxPoints(((190.0, 96.0), (170.0, 170.0), 0.0)),
            image_to_patch,
        )
        cv2.fillConvexPoly(
            image,
            np.round(image_box).astype(np.int32),
            _purple(),
            cv2.LINE_AA,
        )

        analyzer = TrayVisionAnalyzer(
            estimator, geometry, load_slot_marker_layout(LAYOUT_PATH)
        )
        result = analyzer.analyze(image, pose=_valid_pose(image))
        analysis = next(
            slot for slot in result.slots if slot.projection.slot_key == "P23"
        )
        self.assertEqual(SlotState.OUTSIDE_SLOT, analysis.decision.state)
        self.assertFalse(analysis.wafer_correction_outside_slot)
        self.assertFalse(analysis.wafer_correction_center_valid)
        self.assertEqual(
            "refined_physical_size_out_of_range",
            analysis.wafer_correction_center_reason,
        )
        refinement = analysis.wafer_center_refinement
        self.assertFalse(refinement["success"])
        gates = refinement["full_contour_gates"]
        self.assertFalse(gates["physical_size_passed"])
        self.assertGreater(gates["long_side_mm"], gates["maximum_side_mm"])

        # The general detector has nevertheless made an exact OUT decision
        # and drawn a fitted quadrilateral.  Wafer correction may therefore
        # use its diagonal intersection instead of treating expanded-ROI
        # refinement as a mandatory gate.
        candidates = extract_outside_wafer_candidates(result, geometry)
        candidate = next(
            row for row in candidates if "P23" in row["source_slot_keys"]
        )
        self.assertEqual(
            SLOT_QUADRILATERAL_CENTER_SOURCE,
            candidate["center_source"],
        )
        box = np.asarray(analysis.wafer_box_image_px, dtype=np.float64)
        p0, p1, p2, p3 = box
        diagonal_matrix = np.column_stack((p2 - p0, -(p3 - p1)))
        diagonal_scale = np.linalg.solve(diagonal_matrix, p1 - p0)[0]
        expected_image_center = p0 + diagonal_scale * (p2 - p0)
        np.testing.assert_allclose(
            candidate["center_image_px"], expected_image_center, atol=1e-6
        )


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

    def test_abnormal_state_is_split_into_stacked_outside_and_both(self) -> None:
        contained_patch = np.full((192, 192, 3), 185, dtype=np.uint8)
        _draw_square(contained_patch, (78.0, 78.0), 80.0, 0.0, _purple(140, 100))
        _draw_square(contained_patch, (112.0, 112.0), 80.0, 0.0, _purple(150, 145))
        contained = analyze_wafer_patch(
            contained_patch,
            replace(DEFAULT_WAFER_QUALITY, stacking_detection_enabled=True),
        )
        self.assertFalse(contained.outside_slot)
        self.assertEqual(
            decide_slot_state(self._projection(), self._marker(), contained).state,
            SlotState.STACKED,
        )

        outside_patch = np.full((192, 192, 3), 185, dtype=np.uint8)
        _draw_square(outside_patch, (170.0, 96.0), 72.0, 0.0, _purple())
        outside = analyze_wafer_patch(outside_patch)
        self.assertEqual(
            decide_slot_state(self._projection(), self._marker(), outside).state,
            SlotState.OUTSIDE_SLOT,
        )

        both = replace(
            outside,
            quality="abnormal",
            flags=tuple(outside.flags) + ("stacked_geometry_confirmed",),
        )
        self.assertEqual(
            decide_slot_state(self._projection(), self._marker(), both).state,
            SlotState.STACKED_OUTSIDE_SLOT,
        )

    def test_contained_severe_shape_and_reflection_do_not_become_stacked(self) -> None:
        distorted_patch = np.full((192, 192, 3), 185, dtype=np.uint8)
        cv2.rectangle(distorted_patch, (45, 70), (147, 122), _purple(), -1)
        distorted = analyze_wafer_patch(distorted_patch)
        self.assertEqual(distorted.quality, "abnormal")
        self.assertEqual(
            decide_slot_state(self._projection(), self._marker(), distorted).state,
            SlotState.WARNING,
        )

        reflection_patch = np.full((192, 192, 3), 185, dtype=np.uint8)
        _draw_square(reflection_patch, (96.0, 96.0), 92.0, 0.0, _purple(140, 100))
        cv2.rectangle(reflection_patch, (91, 50), (101, 142), _purple(140, 220), -1)
        reflection = analyze_wafer_patch(
            reflection_patch,
            replace(DEFAULT_WAFER_QUALITY, stacking_detection_enabled=True),
        )
        self.assertIn("internal_overlap_edges", reflection.flags)
        self.assertEqual(
            decide_slot_state(self._projection(), self._marker(), reflection).state,
            SlotState.WARNING,
        )

    def test_no_evidence_is_unknown(self) -> None:
        decision = decide_slot_state(
            self._projection(), self._marker(), WaferObservation.not_found()
        )
        self.assertEqual(decision.state, SlotState.UNKNOWN)
        self.assertFalse(decision.safe_to_use_as_empty)


class CameraOverlayTests(unittest.TestCase):
    def test_primary_and_second_wafer_outlines_are_both_drawn(self) -> None:
        canvas = np.zeros((220, 220, 3), dtype=np.uint8)
        analysis = SimpleNamespace(
            projection=SimpleNamespace(
                polygon_px=((20.0, 20.0), (200.0, 20.0), (200.0, 200.0), (20.0, 200.0)),
                center_px=(110.0, 110.0),
                projected_area_px=32400.0,
                slot_key="P32",
            ),
            decision=SimpleNamespace(state=SlotState.STACKED),
            wafer_box_image_px=((40.0, 40.0), (100.0, 40.0), (100.0, 100.0), (40.0, 100.0)),
            wafer_secondary_boxes_image_px=(
                ((110.0, 80.0), (170.0, 80.0), (170.0, 140.0), (110.0, 140.0)),
            ),
            wafer_center_image_px=None,
            wafer_offset_distance_mm=None,
        )
        TrayVisionAnalyzer._draw_slot(
            canvas,
            analysis,
            stacking_detection_enabled=True,
        )
        self.assertGreater(int(canvas[40, 70, 2]), 200)  # W1/state-colour outline
        self.assertGreater(int(canvas[80, 140, 0]), 200)  # W2 cyan outline
        self.assertGreater(int(canvas[80, 140, 1]), 200)

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

    def test_tracked_analysis_uses_filtered_pose(self) -> None:
        image = np.full((720, 1280, 3), 185, dtype=np.uint8)
        raw = _valid_pose(image)
        filtered = raw.T_C_T.copy()
        filtered[0, 3] += 6.0
        filtered_inverse = np.linalg.inv(filtered)
        tracked = TrackedTrayPose(
            raw=raw,
            accepted_by_tracker=True,
            tracker_reason=None,
            filtered_T_C_T=filtered,
            filtered_T_T_C=filtered_inverse,
            translation_jump_mm=6.0,
            rotation_jump_deg=0.0,
            lost_frame_count=0,
        )
        geometry = _geometry()
        analyzer = TrayVisionAnalyzer(
            TrayBoardPoseEstimator(geometry, _intrinsics()),
            geometry,
            load_slot_marker_layout(LAYOUT_PATH),
        )
        result = analyzer.analyze_tracked(image, tracked)
        self.assertTrue(result.quality_passed)
        np.testing.assert_allclose(result.pose.T_C_T, filtered, atol=1e-12)
        converted = tracked_pose_estimate(tracked)
        np.testing.assert_allclose(converted.tvec_C_T_mm.reshape(3), filtered[:3, 3])

    def test_tracker_rejection_is_authoritative(self) -> None:
        image = np.full((720, 1280, 3), 185, dtype=np.uint8)
        raw = _valid_pose(image)
        tracked = TrackedTrayPose(
            raw=raw,
            accepted_by_tracker=False,
            tracker_reason="synthetic temporal jump",
            filtered_T_C_T=raw.T_C_T.copy(),
            filtered_T_T_C=raw.T_T_C.copy(),
            translation_jump_mm=40.0,
            rotation_jump_deg=0.0,
            lost_frame_count=1,
        )
        geometry = _geometry()
        analyzer = TrayVisionAnalyzer(
            TrayBoardPoseEstimator(geometry, _intrinsics()),
            geometry,
            load_slot_marker_layout(LAYOUT_PATH),
        )
        result = analyzer.analyze_tracked(image, tracked)
        self.assertFalse(result.quality_passed)
        self.assertEqual(result.failure_reason, "synthetic temporal jump")
        self.assertEqual(result.slots, ())
        self.assertFalse(result.coordinate_mapping_allowed)


if __name__ == "__main__":
    unittest.main()
