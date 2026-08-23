"""Regression tests for observation-only Task14 tray registration."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.planar_tray_registration import (
    build_planar_slot_projections,
    estimate_planar_tray_registration,
)
from scara.vision.slot_marker_observation import (
    ArucoObservation,
    detect_aruco_observations,
    load_slot_marker_layout,
)
from scara.vision.silicon_detection_config import load_silicon_detection_config
from scara.vision.tray_pose_estimator import (
    TrayBoardPoseEstimator,
    load_camera_intrinsics,
    load_tray_board_geometry,
)
from scara.vision.tray_vision_fusion import TrayVisionAnalyzer


GEOMETRY_PATH = PROJECT_ROOT / "src/scara/calib/tray_board_geometry.json"
INTRINSICS_PATH = PROJECT_ROOT / "src/scara/calib/camera1_intrinsics.json"
LAYOUT_PATH = PROJECT_ROOT / "tools/tray_marker_detector_v2/tray_marker_layout.json"
TASK14_REGRESSION_IMAGE = (
    PROJECT_ROOT
    / "tests/fixtures/task14_planar_registration/raw_task14_1_077.jpg"
)
P52_BOUNDARY_REGRESSION_IMAGES = (
    PROJECT_ROOT
    / "tests/fixtures/task14_planar_registration/raw_task14_1_013.jpg",
    PROJECT_ROOT
    / "tests/fixtures/task14_planar_registration/raw_task14_1_014.jpg",
)
SILICON_CONFIG_PATH = PROJECT_ROOT / "src/scara/calib/silicon_detection_0818.json"


def _transform(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float32).reshape(1, -1, 2), homography
    ).reshape(-1, 2)


def _observation(
    marker_id: int,
    center: np.ndarray,
    corners: np.ndarray | None = None,
    *,
    square_quality: float = 0.98,
) -> ArucoObservation:
    center = np.asarray(center, dtype=float).reshape(2)
    if corners is None:
        corners = np.asarray(
            [
                center + (-5.0, -5.0),
                center + (5.0, -5.0),
                center + (5.0, 5.0),
                center + (-5.0, 5.0),
            ]
        )
    corners = np.asarray(corners, dtype=float).reshape(4, 2)
    return ArucoObservation(
        marker_id=marker_id,
        center_px=tuple(float(value) for value in center),
        corners_px=tuple(
            tuple(float(value) for value in point) for point in corners
        ),
        angle_deg=0.0,
        perimeter_px=40.0,
        area_px=100.0,
        square_quality=square_quality,
    )


class PlanarTrayRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = json.loads(GEOMETRY_PATH.read_text(encoding="utf-8"))
        cls.layout = load_slot_marker_layout(LAYOUT_PATH)
        cls.homography = np.asarray(
            [[3.2, 0.28, 710.0], [0.18, 2.85, 430.0], [0.0004, -0.0002, 1.0]],
            dtype=np.float64,
        )

    def _slot_observations(self, slot_keys: tuple[str, ...]) -> dict[int, ArucoObservation]:
        result = {}
        for slot_key in slot_keys:
            marker_id = self.layout.marker_id_by_slot[slot_key]
            center_T = np.asarray(self.geometry["slots"][slot_key], dtype=float)[:2]
            center_px = _transform(self.homography, center_T)[0]
            result[marker_id] = _observation(marker_id, center_px)
        return result

    def _outer_observations(self, marker_ids: tuple[int, ...]) -> dict[int, ArucoObservation]:
        markers = {
            int(marker["id"]): marker
            for marker in self.geometry["markers"].values()
        }
        result = {}
        for marker_id in marker_ids:
            marker = markers[marker_id]
            corners_T = np.asarray(marker["corners_T_mm"], dtype=float)[:, :2]
            corners_px = _transform(self.homography, corners_T)
            center_T = np.asarray(marker["center_T_mm"], dtype=float)[:2]
            center_px = _transform(self.homography, center_T)[0]
            result[marker_id] = _observation(marker_id, center_px, corners_px)
        return result

    def test_six_distributed_slot_centres_form_read_only_grid(self) -> None:
        observations = self._slot_observations(
            ("P00", "P05", "P22", "P33", "P50", "P55")
        )
        registration = estimate_planar_tray_registration(
            np.zeros((720, 1280, 3), dtype=np.uint8),
            self.geometry,
            self.layout,
            observations,
        )
        self.assertTrue(registration.success, registration.failure_reason)
        self.assertEqual(registration.method, "marker_grid_homography")
        self.assertGreaterEqual(registration.span_x_mm, 60.0)
        self.assertGreaterEqual(registration.span_y_mm, 60.0)
        projections = build_planar_slot_projections(
            self.geometry, registration, (720, 1280, 3), half_extent_mm=15.5
        )
        self.assertEqual(len(projections), 36)

    def test_two_cross_axis_outer_markers_require_all_corners(self) -> None:
        observations = self._outer_observations((4, 8))
        registration = estimate_planar_tray_registration(
            np.zeros((720, 1280, 3), dtype=np.uint8),
            self.geometry,
            self.layout,
            observations,
            allow_partial_corners=False,
        )
        self.assertTrue(registration.success, registration.failure_reason)
        self.assertEqual(registration.method, "two_outer_marker_homography")
        self.assertEqual(registration.inlier_correspondence_count, 8)

    def test_same_edge_two_outer_markers_are_rejected(self) -> None:
        registration = estimate_planar_tray_registration(
            np.zeros((720, 1280, 3), dtype=np.uint8),
            self.geometry,
            self.layout,
            self._outer_observations((1, 2)),
        )
        self.assertFalse(registration.success)
        self.assertIn("spanning both tray axes", registration.failure_reason)

    def test_two_slot_centres_cannot_authorize_a_plane(self) -> None:
        registration = estimate_planar_tray_registration(
            np.zeros((720, 1280, 3), dtype=np.uint8),
            self.geometry,
            self.layout,
            self._slot_observations(("P00", "P55")),
        )
        self.assertFalse(registration.success)
        self.assertEqual(registration.homography_image_from_tray_xy, None)

    def test_low_quality_two_outer_candidate_is_rejected(self) -> None:
        observations = self._outer_observations((4, 8))
        observations[8] = _observation(
            8,
            np.asarray(observations[8].center_px),
            np.asarray(observations[8].corners_px),
            square_quality=0.60,
        )
        registration = estimate_planar_tray_registration(
            np.zeros((720, 1280, 3), dtype=np.uint8),
            self.geometry,
            self.layout,
            observations,
        )
        self.assertFalse(registration.success)
        self.assertIn("square quality", registration.failure_reason)

    def test_high_residual_marker_grid_is_rejected(self) -> None:
        slot_keys = ("P00", "P05", "P22", "P33", "P50", "P55")
        observations = self._slot_observations(slot_keys)
        bad_centres = (
            (80.0, 90.0),
            (1180.0, 120.0),
            (210.0, 610.0),
            (1040.0, 570.0),
            (650.0, 80.0),
            (600.0, 660.0),
        )
        for slot_key, center in zip(slot_keys, bad_centres):
            marker_id = self.layout.marker_id_by_slot[slot_key]
            observations[marker_id] = _observation(
                marker_id, np.asarray(center, dtype=float)
            )
        registration = estimate_planar_tray_registration(
            np.zeros((720, 1280, 3), dtype=np.uint8),
            self.geometry,
            self.layout,
            observations,
        )
        self.assertFalse(registration.success)
        self.assertTrue(
            "residual rejected" in (registration.failure_reason or "")
            or "homography failed" in (registration.failure_reason or "")
        )

    def test_bad_partial_corner_texture_is_never_standalone_evidence(self) -> None:
        registration = estimate_planar_tray_registration(
            np.random.default_rng(4).integers(
                0, 255, size=(720, 1280, 3), dtype=np.uint8
            ),
            self.geometry,
            self.layout,
            self._slot_observations(("P00", "P55")),
        )
        self.assertFalse(registration.success)
        self.assertEqual(registration.partial_marker_ids, ())

    def test_two_identified_partial_outer_corners_only_refine_full_grid(self) -> None:
        observations = self._slot_observations(
            ("P00", "P05", "P22", "P33", "P50", "P55")
        )
        marker = next(
            item
            for item in self.geometry["markers"].values()
            if int(item["id"]) == 4
        )
        predicted = _transform(
            self.homography,
            np.asarray(marker["corners_T_mm"], dtype=float)[:, :2],
        )
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        for point in predicted[:2]:
            x, y = (int(round(value)) for value in point)
            cv2.rectangle(image, (x, y), (x + 7, y + 7), (255, 255, 255), -1)
        registration = estimate_planar_tray_registration(
            image,
            self.geometry,
            self.layout,
            observations,
        )
        self.assertTrue(registration.success, registration.failure_reason)
        self.assertIn(4, registration.partial_marker_ids)
        self.assertTrue(
            registration.partial_marker_diagnostics[4]["retained"]
        )

    def test_image_edge_partial_corners_are_not_forced_when_residual_worsens(self) -> None:
        edge_homography = np.asarray(
            [[2.0, 0.0, 305.0], [0.0, 2.0, 486.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        observations = {}
        for slot_key in ("P00", "P05", "P22", "P33", "P50", "P55"):
            marker_id = self.layout.marker_id_by_slot[slot_key]
            center_T = np.asarray(self.geometry["slots"][slot_key], dtype=float)[:2]
            observations[marker_id] = _observation(
                marker_id, _transform(edge_homography, center_T)[0]
            )
        marker = next(
            item
            for item in self.geometry["markers"].values()
            if int(item["id"]) == 8
        )
        predicted = _transform(
            edge_homography,
            np.asarray(marker["corners_T_mm"], dtype=float)[:, :2],
        )
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        visible_count = 0
        for point in predicted:
            x, y = (int(round(value)) for value in point)
            if 9 <= x < image.shape[1] - 9 and 9 <= y < image.shape[0] - 9:
                cv2.rectangle(
                    image, (x, y), (x + 7, y + 7), (255, 255, 255), -1
                )
                visible_count += 1
        self.assertEqual(visible_count, 2)
        registration = estimate_planar_tray_registration(
            image,
            self.geometry,
            self.layout,
            observations,
        )
        self.assertTrue(registration.success, registration.failure_reason)
        diagnostic = registration.partial_marker_diagnostics[8]
        self.assertTrue(diagnostic["candidate_accepted_for_refinement"])
        self.assertFalse(diagnostic["retained"])
        self.assertNotIn(8, registration.partial_marker_ids)

    def test_native_scale_candidate_wins_multiscale_duplicates(self) -> None:
        image = np.full((240, 320), 220, dtype=np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(dictionary, 39, 72)
        image[90:162, 120:192] = marker
        observations = detect_aruco_observations(
            image,
            "DICT_4X4_50",
            scales=(2.0, 1.5, 1.0),
            include_clahe=True,
        )
        self.assertEqual(observations[39].detection_scale, 1.0)
        self.assertEqual(observations[39].preprocessing, "native")
        self.assertTrue(observations[39].complete_decoded)

    def test_small_blurred_marker_remains_decodable(self) -> None:
        image = np.full((128, 128), 220, dtype=np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(dictionary, 39, 18)
        image[54:72, 55:73] = marker
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        observations = detect_aruco_observations(
            blurred, "DICT_4X4_50", scales=(1.0, 1.5, 2.0)
        )
        self.assertIn(39, observations)
        self.assertGreater(observations[39].square_quality, 0.80)


@unittest.skipUnless(
    TASK14_REGRESSION_IMAGE.is_file(),
    "Task14 1_077 regression fixture is not present",
)
class Task14ImageRegressionTests(unittest.TestCase):
    def test_raw_1_077_uses_read_only_grid_while_strict_pose_stays_rejected(self) -> None:
        geometry = load_tray_board_geometry(GEOMETRY_PATH)
        estimator = TrayBoardPoseEstimator(
            geometry, load_camera_intrinsics(INTRINSICS_PATH)
        )
        analyzer = TrayVisionAnalyzer(
            estimator, geometry, load_slot_marker_layout(LAYOUT_PATH)
        )
        image = cv2.imread(str(TASK14_REGRESSION_IMAGE), cv2.IMREAD_COLOR)
        pose = estimator.estimate(image)
        result = analyzer.analyze(image, pose=pose)
        self.assertFalse(pose.quality_passed)
        self.assertFalse(result.quality_passed)
        self.assertTrue(result.analysis_quality_passed)
        self.assertEqual(result.projection_source, "marker_grid_homography")
        self.assertEqual(len(result.slots), 36)
        self.assertFalse(result.coordinate_mapping_allowed)
        self.assertFalse(result.robot_correction_allowed)

    @unittest.skipUnless(
        all(path.is_file() for path in P52_BOUNDARY_REGRESSION_IMAGES),
        "Task14 P52 paired regression fixtures are not present",
    )
    def test_p52_adjacent_frames_remain_in_the_outside_state_family(self) -> None:
        geometry = load_tray_board_geometry(GEOMETRY_PATH)
        estimator = TrayBoardPoseEstimator(
            geometry, load_camera_intrinsics(INTRINSICS_PATH)
        )
        config = load_silicon_detection_config(SILICON_CONFIG_PATH).fusion_config
        analyzer = TrayVisionAnalyzer(
            estimator,
            geometry,
            load_slot_marker_layout(LAYOUT_PATH),
            config=config,
        )
        states = []
        for image_path in P52_BOUNDARY_REGRESSION_IMAGES:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            pose = estimator.estimate(image)
            result = analyzer.analyze(image, pose=pose)
            self.assertTrue(result.quality_passed)
            self.assertEqual("strict_pnp_slot_marker_refined", result.projection_source)
            p52 = next(
                slot for slot in result.slots if slot.projection.slot_key == "P52"
            )
            states.append(p52.decision.state.value)
            self.assertIn(
                p52.decision.state.value,
                {"outside_slot", "stacked_outside_slot"},
            )
            self.assertEqual("strong_outside", p52.wafer.boundary_evidence)
            self.assertIsNotNone(p52.wafer.base_projection_clearance_px)
            self.assertIsNotNone(p52.wafer.refined_projection_clearance_px)
        self.assertNotIn("occupied", states)


if __name__ == "__main__":
    unittest.main()
