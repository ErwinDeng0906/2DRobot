from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.marker_grid_wafer_review import (
    _independent_states_for_review,
    review_marker_grid_image,
)
from scara.vision.silicon_detection_config import load_silicon_detection_config
from scara.vision.slot_marker_observation import load_slot_marker_layout
from scara.vision.tray_pose_estimator import load_tray_board_geometry
from scara.vision.wafer_shape_quality import (
    WaferObservation,
    analyze_dark_wafer_patch,
    analyze_wafer_patch,
)
from tools.evaluate_wafer_dataset import (
    _placement_consensus,
    _stacking_consensus,
)


FIXTURE_DIR = ROOT / "tests/fixtures/silicon_detection"
IMAGE_PATH = FIXTURE_DIR / "0820_upper_normal_lower_outside.jpg"
LABEL_PATH = FIXTURE_DIR / "0820_upper_normal_lower_outside.json"


class MarkerGridWaferReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
        assert cls.image is not None
        cls.labels = json.loads(LABEL_PATH.read_text(encoding="utf-8"))
        cls.geometry = load_tray_board_geometry(
            ROOT / "src/scara/calib/tray_board_geometry.json"
        )
        cls.layout = load_slot_marker_layout(
            ROOT / "tools/tray_marker_detector_v2/tray_marker_layout.json"
        )
        cls.config = load_silicon_detection_config(
            ROOT / "src/scara/calib/silicon_detection_0818.json"
        ).fusion_config

    def _analyze(self, image):
        return review_marker_grid_image(
            image,
            self.geometry,
            self.layout,
            self.config,
        )

    def test_operator_labelled_normal_and_outside_sets_are_exact(self) -> None:
        result = self._analyze(self.image)
        self.assertTrue(result.success, result.failure_reason)
        states = {slot.slot_key: slot.state for slot in result.slots}
        self.assertEqual(
            set(self.labels["normal_slots"]),
            {slot for slot, state in states.items() if state == "occupied"},
        )
        self.assertEqual(
            set(self.labels["outside_slots"]),
            {slot for slot, state in states.items() if state == "outside_slot"},
        )
        self.assertEqual(8, result.summary.get("occupied"))
        self.assertEqual(8, result.summary.get("outside_slot"))
        self.assertEqual(0, result.to_json()["robot_motion_authorized"])
        for slot in result.slots:
            self.assertIn(
                slot.placement_state,
                {"inside", "outside", "uncertain", "empty", "unknown", "unobservable"},
            )
            self.assertIn(
                slot.stacking_state,
                {
                    "single",
                    "suspected",
                    "confirmed",
                    "uncertain",
                    "not_applicable",
                    "unknown",
                    "unobservable",
                },
            )

    def test_zoom_change_keeps_the_same_labelled_sets(self) -> None:
        resized = cv2.resize(
            self.image,
            None,
            fx=0.75,
            fy=0.75,
            interpolation=cv2.INTER_AREA,
        )
        result = self._analyze(resized)
        self.assertTrue(result.success, result.failure_reason)
        states = {slot.slot_key: slot.state for slot in result.slots}
        self.assertEqual(
            set(self.labels["normal_slots"]),
            {slot for slot, state in states.items() if state == "occupied"},
        )
        expected_outside = set(self.labels["outside_slots"])
        self.assertEqual(
            expected_outside,
            {
                slot
                for slot in expected_outside
                if states[slot] in {"outside_slot", "warning"}
            },
        )
        self.assertNotIn(
            "occupied", {states[slot] for slot in expected_outside}
        )

    def test_known_0805_normal_and_outside_patches_remain_distinct(self) -> None:
        normal_patch = cv2.imread(
            str(FIXTURE_DIR / "0805_P25_expected_normal.png"),
            cv2.IMREAD_COLOR,
        )
        outside_patch = cv2.imread(
            str(FIXTURE_DIR / "0805_P35_expected_outside.png"),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(normal_patch)
        self.assertIsNotNone(outside_patch)
        normal = analyze_wafer_patch(normal_patch, self.config.wafer_quality)
        outside = analyze_wafer_patch(outside_patch, self.config.wafer_quality)
        self.assertEqual("normal", normal.quality, normal.flags)
        self.assertFalse(normal.outside_slot)
        self.assertIn("thin_bridge_removed", normal.flags)
        self.assertGreater(normal.minimum_slot_clearance_ratio or 0.0, 0.0)
        self.assertTrue(outside.outside_slot, outside.flags)
        self.assertIn("outside_slot", outside.flags)
        self.assertLess(outside.minimum_slot_clearance_ratio or 0.0, 0.0)

    def test_oversize_merged_colour_region_cannot_be_confirmed_outside(self) -> None:
        patch = np.zeros((192, 192, 3), dtype=np.uint8)
        cv2.rectangle(patch, (25, 25), (166, 166), (90, 20, 90), -1)
        observation = analyze_wafer_patch(
            patch, self.config.wafer_quality
        )
        self.assertTrue(observation.found)
        self.assertGreater(
            observation.side_ratio,
            self.config.wafer_quality.maximum_normal_side_ratio,
        )
        self.assertIn("oversize_footprint", observation.flags)
        self.assertIn("boundary_crossing_unconfirmed", observation.flags)
        self.assertFalse(observation.outside_slot)

    def test_uniform_low_chroma_square_is_a_warning_only_fallback_candidate(self) -> None:
        patch = np.full((192, 192, 3), 205, dtype=np.uint8)
        cv2.rectangle(patch, (49, 49), (142, 142), (35, 32, 36), -1)
        observation = analyze_dark_wafer_patch(
            patch, self.config.wafer_quality
        )
        self.assertTrue(observation.found, observation.flags)
        self.assertEqual("warning", observation.quality, observation.flags)
        self.assertIn("dark_low_chroma_fallback", observation.flags)
        self.assertFalse(observation.outside_slot)
        placement, _stacking = _independent_states_for_review(
            observation,
            marker_visible=False,
            coverage=1.0,
            minimum_coverage=0.90,
        )
        self.assertEqual("uncertain", placement)

    def test_insufficient_marker_geometry_rejects_without_coordinates(self) -> None:
        crop = self.image[:220, :220]
        result = self._analyze(crop)
        self.assertFalse(result.success)
        payload = result.to_json()
        self.assertFalse(payload["coordinate_mapping_allowed"])
        self.assertFalse(payload["robot_motion_authorized"])
        self.assertEqual({"unknown": 36}, result.summary)

    def test_one_axis_local_view_rejects_instead_of_extrapolating(self) -> None:
        old_partial_view = Path(
            "/Users/chenge/Desktop/二维机器人/images 0805/"
            "6-36 flake 2-stacked 1-crooked/"
            "camera_1_20260805_151721_319747.png"
        )
        if not old_partial_view.is_file():
            self.skipTest("optional laboratory archive is not available")
        image = cv2.imread(str(old_partial_view), cv2.IMREAD_COLOR)
        self.assertIsNotNone(image)
        result = self._analyze(image)
        self.assertFalse(result.success)
        self.assertIn("do not span both tray axes", result.failure_reason or "")
        self.assertFalse(result.to_json()["coordinate_mapping_allowed"])
        self.assertFalse(result.to_json()["robot_motion_authorized"])

    def test_three_row_view_recovers_operator_checked_slots(self) -> None:
        old_partial_view = Path(
            "/Users/chenge/Desktop/二维机器人/images 0805/"
            "8-36 flake 2-stacked 6-crooked/"
            "camera_1_20260805_152338_835838.png"
        )
        if not old_partial_view.is_file():
            self.skipTest("optional laboratory archive is not available")
        image = cv2.imread(str(old_partial_view), cv2.IMREAD_COLOR)
        self.assertIsNotNone(image)
        result = self._analyze(image)
        self.assertTrue(result.success, result.failure_reason)
        slots = {slot.slot_key: slot for slot in result.slots}
        self.assertEqual("inside", slots["P25"].placement_state)
        self.assertEqual("outside", slots["P35"].placement_state)
        self.assertLessEqual(result.fit.reprojection_rms_px or 99.0, 4.0)
        self.assertFalse(result.to_json()["coordinate_mapping_allowed"])
        self.assertFalse(result.to_json()["robot_motion_authorized"])

    def test_three_marker_corner_fit_is_review_only_and_conservative(self) -> None:
        image_path = Path(
            "/Users/chenge/Desktop/二维机器人/images 0805/"
            "5-36 flake normal/"
            "camera_1_20260805_150850_913398.png"
        )
        if not image_path.is_file():
            self.skipTest("optional laboratory archive is not available")
        result = self._analyze(cv2.imread(str(image_path), cv2.IMREAD_COLOR))
        self.assertTrue(result.success, result.failure_reason)
        self.assertEqual("three_marker_corner_geometry", result.fit.fit_method)
        self.assertNotIn("outside_slot", result.summary)
        self.assertNotIn("stacked", result.summary)
        self.assertFalse(result.to_json()["coordinate_mapping_allowed"])
        self.assertFalse(result.to_json()["robot_motion_authorized"])

    def test_two_same_edge_outer_markers_remain_rejected(self) -> None:
        image_path = Path(
            "/Users/chenge/Desktop/二维机器人/images 0805/"
            "5-36 flake normal/"
            "camera_1_20260805_150826_632468.png"
        )
        if not image_path.is_file():
            self.skipTest("optional laboratory archive is not available")
        result = self._analyze(cv2.imread(str(image_path), cv2.IMREAD_COLOR))
        self.assertFalse(result.success)
        self.assertIn("insufficient", result.failure_reason or "")
        self.assertFalse(result.to_json()["coordinate_mapping_allowed"])
        self.assertFalse(result.to_json()["robot_motion_authorized"])

    def test_two_diagonal_outer_markers_recover_review_only(self) -> None:
        image_path = Path(
            "/Users/chenge/Desktop/二维机器人/images 0805/"
            "5-36 flake normal/"
            "camera_1_20260805_151016_210585.png"
        )
        if not image_path.is_file():
            self.skipTest("optional laboratory archive is not available")
        result = self._analyze(cv2.imread(str(image_path), cv2.IMREAD_COLOR))
        self.assertTrue(result.success, result.failure_reason)
        self.assertEqual(
            "two_outer_marker_corner_geometry", result.fit.fit_method
        )
        self.assertNotIn("outside_slot", result.summary)
        self.assertNotIn("stacked", result.summary)
        self.assertFalse(result.to_json()["coordinate_mapping_allowed"])
        self.assertFalse(result.to_json()["robot_motion_authorized"])

    def test_normal_full_tray_reflections_are_not_suspected_stacks(self) -> None:
        image_path = Path(
            "/Users/chenge/Desktop/二维机器人/images 0805/"
            "5-36 flake normal/full view 1.png"
        )
        if not image_path.is_file():
            self.skipTest("optional laboratory archive is not available")
        result = self._analyze(cv2.imread(str(image_path), cv2.IMREAD_COLOR))
        self.assertTrue(result.success, result.failure_reason)
        found = [slot for slot in result.slots if slot.wafer.found]
        self.assertTrue(found)
        self.assertTrue(
            all(slot.stacking_state == "single" for slot in found),
            [(slot.slot_key, slot.stacking_state) for slot in found],
        )

    def test_known_stack_remains_suspected_in_a_single_frame(self) -> None:
        image_path = Path(
            "/Users/chenge/Desktop/二维机器人/images 0805/"
            "9-36 flake 6-stacked/"
            "camera_1_20260805_154549_147868.png"
        )
        if not image_path.is_file():
            self.skipTest("optional laboratory archive is not available")
        result = self._analyze(cv2.imread(str(image_path), cv2.IMREAD_COLOR))
        self.assertTrue(result.success, result.failure_reason)
        slot = next(slot for slot in result.slots if slot.slot_key == "P43")
        self.assertEqual("suspected", slot.stacking_state)
        self.assertIn("oversize_footprint", slot.wafer.flags)

    def test_old_empty_tray_marker_fringes_are_not_wafers(self) -> None:
        image_path = Path(
            "/Users/chenge/Desktop/二维机器人/images/1-0 flake/"
            "camera_0_20260804_051743_582729.png"
        )
        if not image_path.is_file():
            self.skipTest("optional laboratory archive is not available")
        result = self._analyze(cv2.imread(str(image_path), cv2.IMREAD_COLOR))
        self.assertTrue(result.success, result.failure_reason)
        states = {slot.slot_key: slot.state for slot in result.slots}
        self.assertEqual("empty", states["P50"])
        self.assertEqual("empty", states["P55"])

    def test_sparse_marker_extrapolation_cannot_confirm_normal_wafers_outside(self) -> None:
        image_path = Path(
            "/Users/chenge/Desktop/二维机器人/images 0805/"
            "5-36 flake normal/camera_1_20260805_150956_663354.png"
        )
        if not image_path.is_file():
            self.skipTest("optional laboratory archive is not available")
        result = self._analyze(cv2.imread(str(image_path), cv2.IMREAD_COLOR))
        self.assertTrue(result.success, result.failure_reason)
        slots = {slot.slot_key: slot for slot in result.slots}
        for slot_key in ("P11", "P21", "P22", "P32", "P43"):
            self.assertNotEqual("outside", slots[slot_key].placement_state)
            self.assertIn(
                "boundary_geometry_extrapolated", slots[slot_key].wafer.flags
            )

    def test_one_l_shaped_reflection_does_not_confirm_a_stack(self) -> None:
        image_path = Path(
            "/Users/chenge/Desktop/二维机器人/images 0805/"
            "10-36 flake 6-crooked/camera_1_20260805_160042_929363.png"
        )
        if not image_path.is_file():
            self.skipTest("optional laboratory archive is not available")
        result = self._analyze(cv2.imread(str(image_path), cv2.IMREAD_COLOR))
        self.assertTrue(result.success, result.failure_reason)
        slot = next(slot for slot in result.slots if slot.slot_key == "P54")
        self.assertEqual("single", slot.stacking_state)
        self.assertNotEqual("stacked", slot.state)

    def test_marker_center_fallback_recovers_checked_old_frame(self) -> None:
        image_path = Path(
            "/Users/chenge/Desktop/二维机器人/images/2-1 flake normal/"
            "camera_0_20260804_053100_270203.png"
        )
        if not image_path.is_file():
            self.skipTest("optional laboratory archive is not available")
        result = self._analyze(cv2.imread(str(image_path), cv2.IMREAD_COLOR))
        self.assertTrue(result.success, result.failure_reason)
        self.assertEqual(
            "marker_centres_only_fallback", result.fit.fit_method
        )
        self.assertGreaterEqual(result.fit.inlier_marker_count, 4)
        self.assertLessEqual(result.fit.reprojection_rms_px or 99.0, 3.5)
        self.assertFalse(result.to_json()["coordinate_mapping_allowed"])

    def test_sequence_consensus_recovers_normal_from_one_obstructed_frame(self) -> None:
        evidence = [
            {
                "file": "clear_1.png",
                "image_coverage_ratio": 1.0,
                "wafer_found": True,
                "marker_visible": False,
                "minimum_slot_clearance_ratio": 0.05,
                "side_ratio": 0.48,
                "aspect_ratio": 1.02,
                "flags": [],
                "stacking_state": "single",
            },
            {
                "file": "clear_2.png",
                "image_coverage_ratio": 1.0,
                "wafer_found": True,
                "marker_visible": False,
                "minimum_slot_clearance_ratio": 0.03,
                "side_ratio": 0.50,
                "aspect_ratio": 1.05,
                "flags": [],
                "stacking_state": "single",
            },
            {
                "file": "clear_3.png",
                "image_coverage_ratio": 1.0,
                "wafer_found": True,
                "marker_visible": False,
                "minimum_slot_clearance_ratio": 0.04,
                "side_ratio": 0.49,
                "aspect_ratio": 1.03,
                "flags": [],
                "stacking_state": "single",
            },
            {
                "file": "cable_obstruction.png",
                "image_coverage_ratio": 1.0,
                "wafer_found": True,
                "marker_visible": False,
                "minimum_slot_clearance_ratio": -0.04,
                "side_ratio": 0.70,
                "aspect_ratio": 1.10,
                "flags": ["oversize_footprint", "boundary_crossing_unconfirmed"],
                "stacking_state": "suspected",
            },
        ]
        placement = _placement_consensus(evidence)
        stacking = _stacking_consensus(evidence)
        self.assertEqual("inside", placement["decision"])
        self.assertEqual(3, placement["boundary_sample_count"])
        self.assertEqual("single", stacking["decision"])

    def test_sequence_consensus_requires_repeated_outside_evidence(self) -> None:
        evidence = []
        for index, clearance in enumerate((-0.12, -0.10, -0.08, 0.01)):
            evidence.append(
                {
                    "file": f"view_{index}.png",
                    "image_coverage_ratio": 1.0,
                    "wafer_found": True,
                    "marker_visible": False,
                    "minimum_slot_clearance_ratio": clearance,
                    "side_ratio": 0.51,
                    "aspect_ratio": 1.08,
                    "flags": ["outside_slot"] if clearance < 0 else [],
                    "stacking_state": "single",
                }
            )
        consensus = _placement_consensus(evidence)
        self.assertEqual("outside", consensus["decision"])
        self.assertEqual(3, consensus["outside_evidence_count"])

    def test_sequence_consensus_does_not_promote_one_false_outside(self) -> None:
        evidence = [
            {
                "file": "only_view.png",
                "image_coverage_ratio": 1.0,
                "wafer_found": True,
                "marker_visible": False,
                "minimum_slot_clearance_ratio": -0.10,
                "side_ratio": 0.50,
                "aspect_ratio": 1.03,
                "flags": ["outside_slot"],
                "stacking_state": "single",
            }
        ]
        self.assertEqual(
            "unknown", _placement_consensus(evidence)["decision"]
        )

    def test_sequence_consensus_confirms_repeatable_stacked_envelope(self) -> None:
        evidence = []
        for index in range(5):
            evidence.append(
                {
                    "file": f"stack_view_{index}.png",
                    "image_coverage_ratio": 1.0,
                    "wafer_found": True,
                    "stacking_state": "suspected",
                    "side_ratio": 0.67,
                    "rectangularity": 0.99,
                    "solidity": 0.94,
                }
            )
        consensus = _stacking_consensus(evidence)
        self.assertEqual("confirmed", consensus["decision"])
        self.assertEqual(
            "repeatable_oversized_square_envelope",
            consensus["decision_reason"],
        )

    def test_sequence_consensus_rejects_one_reflection_envelope(self) -> None:
        evidence = [
            {
                "file": "reflection.png",
                "image_coverage_ratio": 1.0,
                "wafer_found": True,
                "stacking_state": "suspected",
                "side_ratio": 0.68,
                "rectangularity": 1.0,
                "solidity": 0.95,
            },
            {
                "file": "clear_1.png",
                "image_coverage_ratio": 1.0,
                "wafer_found": True,
                "stacking_state": "single",
                "side_ratio": 0.50,
                "rectangularity": 0.98,
                "solidity": 0.94,
            },
            {
                "file": "clear_2.png",
                "image_coverage_ratio": 1.0,
                "wafer_found": True,
                "stacking_state": "single",
                "side_ratio": 0.49,
                "rectangularity": 0.99,
                "solidity": 0.95,
            },
        ]
        self.assertEqual("single", _stacking_consensus(evidence)["decision"])

    def test_clean_oversized_square_is_only_suspected_stack(self) -> None:
        observation = self._wafer_observation(
            side_ratio=0.66,
            center_offset_ratio=0.08,
            rectangularity=0.98,
            solidity=0.94,
            flags=("oversize_footprint", "boundary_crossing_unconfirmed"),
        )
        placement, stacking = _independent_states_for_review(
            observation,
            marker_visible=False,
            coverage=1.0,
            minimum_coverage=0.90,
        )
        self.assertEqual("uncertain", placement)
        self.assertEqual("suspected", stacking)

    def test_irregular_oversized_shape_has_uncertain_layer_count(self) -> None:
        observation = self._wafer_observation(
            side_ratio=0.69,
            center_offset_ratio=0.10,
            rectangularity=1.0,
            solidity=0.82,
            flags=(
                "oversize_footprint",
                "internal_overlap_edges",
                "irregular_outline",
                "boundary_crossing_unconfirmed",
            ),
        )
        _, stacking = _independent_states_for_review(
            observation,
            marker_visible=False,
            coverage=1.0,
            minimum_coverage=0.90,
        )
        self.assertEqual("uncertain", stacking)

    def test_boundary_uncertain_composite_is_fail_closed_for_both_axes(self) -> None:
        observation = self._wafer_observation(
            side_ratio=0.56,
            center_offset_ratio=0.12,
            rectangularity=0.90,
            solidity=0.76,
            flags=(
                "multiple_components",
                "irregular_outline",
                "boundary_uncertain",
            ),
        )
        placement, stacking = _independent_states_for_review(
            observation,
            marker_visible=False,
            coverage=1.0,
            minimum_coverage=0.90,
        )
        self.assertEqual("uncertain", placement)
        self.assertEqual("uncertain", stacking)

    @staticmethod
    def _wafer_observation(
        *,
        side_ratio: float,
        center_offset_ratio: float,
        rectangularity: float,
        solidity: float,
        flags: tuple[str, ...],
    ) -> WaferObservation:
        return WaferObservation(
            found=True,
            quality="warning",
            center_patch_px=(96.0, 96.0),
            box_patch_px=((64.0, 64.0), (128.0, 64.0), (128.0, 128.0), (64.0, 128.0)),
            area_ratio=0.42,
            side_ratio=side_ratio,
            aspect_ratio=1.0,
            rectangularity=rectangularity,
            solidity=solidity,
            center_offset_ratio=center_offset_ratio,
            yaw_relative_to_tray_deg=0.0,
            polygon_vertices=4,
            component_count=1,
            second_component_area_ratio=0.0,
            internal_line_count=0,
            internal_line_score=0.0,
            chromatic_fraction=0.80,
            confidence=0.70,
            flags=flags,
            outside_slot=False,
            minimum_slot_clearance_ratio=-0.02,
            secondary_boxes_patch_px=(),
        )


if __name__ == "__main__":
    unittest.main()
