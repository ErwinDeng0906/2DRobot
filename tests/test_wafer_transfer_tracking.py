from __future__ import annotations

import time
import unittest
from dataclasses import replace
from pathlib import Path
import sys
from typing import Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.close_range_slot_observation import (  # noqa: E402
    CloseRangeOperation,
    CloseRangeSlotObservation,
    CloseRangeState,
)
from scara.vision.slot_marker_observation import (  # noqa: E402
    SlotMarkerEvidence,
    SlotProjection,
)
from scara.vision.tray_occupancy import SlotDecision, SlotState  # noqa: E402
from scara.vision.tray_pose_estimator import TrayPoseEstimate  # noqa: E402
from scara.vision.tray_vision_fusion import (  # noqa: E402
    SlotAnalysis,
    TrayVisionResult,
)
from scara.vision.wafer_shape_quality import WaferObservation  # noqa: E402
from scara.vision.wafer_transfer_tracking import (  # noqa: E402
    TransferPhase,
    WaferTransferSession,
)
from scara.vision.wafer_transfer_runtime import LiveWaferTransferRuntime  # noqa: E402


def geometry() -> dict:
    return {
        "slots": {
            f"P{row}{column}": [25.0 * row, 25.0 * column, -2.0]
            for row in range(6)
            for column in range(6)
        }
    }


def pose(image: np.ndarray) -> TrayPoseEstimate:
    transform = np.eye(4, dtype=np.float64)
    return TrayPoseEstimate(
        success=True,
        quality_passed=True,
        failure_reason=None,
        visible_marker_ids=(1, 2, 3, 4),
        used_marker_ids=(1, 2, 3, 4),
        rejected_marker_ids=(),
        ransac_inlier_corner_count=16,
        object_span_mm=150.0,
        reprojection_rms_px=0.25,
        per_marker_rms_px={1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2},
        rvec_C_T=np.zeros((3, 1), dtype=np.float64),
        tvec_C_T_mm=np.asarray([[0.0], [0.0], [400.0]]),
        T_C_T=transform,
        T_T_C=transform,
        camera_position_T_mm=np.asarray([0.0, 0.0, 400.0]),
        minimum_object_depth_C_mm=400.0,
        annotated_image=image.copy(),
    )


def analysis(slot: str, state: SlotState) -> SlotAnalysis:
    row, column = int(slot[1]), int(slot[2])
    center = (25.0 * row, 25.0 * column, -2.0)
    projection = SlotProjection(
        slot_key=slot,
        row=row,
        column=column,
        center_T_mm=center,
        center_px=(100.0 + row * 40.0, 100.0 + column * 40.0),
        polygon_T_mm=(center, center, center, center),
        polygon_px=((80.0, 80.0), (120.0, 80.0), (120.0, 120.0), (80.0, 120.0)),
        image_coverage_ratio=1.0,
        projected_area_px=1600.0,
    )
    occupied = state in {
        SlotState.OCCUPIED,
        SlotState.WARNING,
        SlotState.STACKED,
        SlotState.OUTSIDE_SLOT,
        SlotState.STACKED_OUTSIDE_SLOT,
    }
    safe_empty = state in {SlotState.EMPTY, SlotState.EMPTY_UNREAD_MARKER}
    decision = SlotDecision(
        slot,
        state,
        occupied=occupied,
        safe_to_use_as_empty=safe_empty,
        reason=state.value,
        flags=(),
    )
    wafer = WaferObservation.not_found()
    if occupied:
        wafer = replace(
            wafer,
            found=True,
            quality=("normal" if state is SlotState.OCCUPIED else state.value),
            yaw_relative_to_tray_deg=0.0,
            confidence=0.95,
        )
    marker = SlotMarkerEvidence(
        slot,
        expected_marker_id=row * 6 + column,
        decoded=safe_empty,
        decoded_marker_id=(row * 6 + column if safe_empty else None),
        marker_like_visible=safe_empty,
        center_error_px=0.0 if safe_empty else None,
        detection_quality=1.0 if safe_empty else None,
        pattern_features={},
    )
    return SlotAnalysis(
        projection=projection,
        marker=marker,
        wafer=wafer,
        decision=decision,
        wafer_box_image_px=(),
        wafer_secondary_boxes_image_px=(),
        wafer_center_image_px=None,
        wafer_center_T_mm=None,
        wafer_offset_T_mm=None,
        wafer_offset_distance_mm=None,
        explicit_occlusion_ratio=0.0,
    )


def result(source_state: SlotState = SlotState.OCCUPIED) -> TrayVisionResult:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    rows = []
    for row in range(6):
        for column in range(6):
            slot = f"P{row}{column}"
            state = source_state if slot == "P11" else SlotState.EMPTY
            rows.append(analysis(slot, state))
    return TrayVisionResult(
        success=True,
        quality_passed=True,
        failure_reason=None,
        coordinate_mapping_allowed=True,
        robot_correction_allowed=False,
        pose=pose(image),
        slot_markers={},
        slots=tuple(rows),
        summary={},
        annotated_image=image,
    )


def registration() -> dict:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = 100.0
    transform[1, 3] = 200.0
    return {
        "status": "success",
        "transform_W_T": transform.tolist(),
        "origin_world_xy_mm": [100.0, 200.0],
        "yaw_world_from_tray_deg": 0.0,
    }


def robot_state(x: float, y: float, *, captured: Optional[float] = None) -> dict:
    return {
        "captured_monotonic_s": time.monotonic() if captured is None else captured,
        "joints": [20.0, 30.0, -27.0, 10.0],
        "pose": [x, y, -27.0, 180.0, 0.0, 20.0],
    }


def square(center_x: float, center_y: float, half_size: float = 10.0):
    return (
        (center_x - half_size, center_y - half_size),
        (center_x + half_size, center_y - half_size),
        (center_x + half_size, center_y + half_size),
        (center_x - half_size, center_y + half_size),
    )


def pick_observation(
    *,
    state: CloseRangeState = CloseRangeState.ALIGNED,
    captured: Optional[float] = None,
    robot_j3_mm: float = -27.0,
) -> CloseRangeSlotObservation:
    return CloseRangeSlotObservation(
        operation=CloseRangeOperation.PICK,
        state=state,
        target_slot="P11",
        measurement_id="camera2-pick",
        captured_monotonic_s=time.monotonic() if captured is None else captured,
        quality_passed=True,
        suction_center_px=(320.0, 240.0),
        wafer_center_px=(320.2, 239.9),
        wafer_corners_px=square(320.2, 239.9),
        center_error_px=(0.2, -0.1),
        angle_error_deg=0.3,
        edge_fit_rms_px=0.25,
        robot_j3_mm=robot_j3_mm,
        calibration_profile="camera2-close-j3--27",
        valid_j3_range_mm=(-27.5, -26.5),
        maximum_center_error_px=2.0,
        maximum_angle_error_deg=1.0,
        maximum_edge_fit_rms_px=0.8,
    )


def place_observation(
    *,
    state: CloseRangeState,
    captured: Optional[float] = None,
    complete: bool = True,
) -> CloseRangeSlotObservation:
    return CloseRangeSlotObservation(
        operation=CloseRangeOperation.PLACE,
        state=state,
        target_slot="P22",
        measurement_id="camera2-place",
        captured_monotonic_s=time.monotonic() if captured is None else captured,
        quality_passed=True,
        slot_center_px=(300.0, 220.0) if complete else None,
        slot_corners_px=square(300.0, 220.0, 12.0) if complete else (),
        wafer_center_px=(300.1, 220.2) if complete else None,
        wafer_corners_px=square(300.1, 220.2) if complete else (),
        center_error_px=(0.1, 0.2) if complete else None,
        angle_error_deg=0.2 if complete else None,
        minimum_clearance_px=2.0 if complete else None,
        edge_fit_rms_px=0.3 if complete else None,
        robot_j3_mm=-27.0 if complete else None,
        calibration_profile="camera2-close-j3--27" if complete else "",
        valid_j3_range_mm=(-27.5, -26.5) if complete else None,
        maximum_center_error_px=2.0 if complete else None,
        maximum_angle_error_deg=1.0 if complete else None,
        maximum_edge_fit_rms_px=0.8 if complete else None,
        minimum_required_clearance_px=1.0 if complete else None,
        lifted_wafer_detected=True if complete else None,
        wafer_attached_to_suction=True if complete else None,
    )


class WaferTransferSessionTests(unittest.TestCase):
    @staticmethod
    def _update_overview_frames(
        session: WaferTransferSession,
        *,
        state: SlotState = SlotState.OCCUPIED,
        count: int = 3,
        x: float = 120.0,
        y: float = 220.0,
        robot_captured_offset_s: float = 0.0,
    ) -> None:
        start_sequence = int(session.latest_frame_sequence or 0) + 1
        for sequence in range(start_sequence, start_sequence + count):
            captured = time.monotonic()
            session.update_overview(
                result(state),
                frame_sequence=sequence,
                frame_captured_monotonic_s=captured,
                robot_state=robot_state(
                    x,
                    y,
                    captured=captured + robot_captured_offset_s,
                ),
            )

    def _ready_session(self) -> WaferTransferSession:
        session = WaferTransferSession(geometry())
        self._update_overview_frames(session)
        session.select_source("P11")
        session.select_destination("P22")
        session.set_registration(registration())
        self.assertEqual(session.phase, TransferPhase.ROUTE_READY)
        return session

    def test_source_and_destination_require_independent_visual_evidence(self) -> None:
        session = WaferTransferSession(geometry())
        now = time.monotonic()
        session.update_overview(
            result(SlotState.WARNING),
            frame_sequence=1,
            frame_captured_monotonic_s=now,
            robot_state=robot_state(0.0, 0.0, captured=now),
        )
        with self.assertRaises(ValueError):
            session.select_source("P11")
        self._update_overview_frames(session, x=0.0, y=0.0)
        session.select_source("P11")
        with self.assertRaises(ValueError):
            session.select_destination("P11")

    def test_world_delta_updates_with_robot_motion(self) -> None:
        session = self._ready_session()
        # P11 is (25,25) in T, so it is (125,225) in W.
        np.testing.assert_allclose(session.active_delta_world_xy(), [5.0, 5.0])
        session.start_tracking()
        now = time.monotonic()
        session.update_overview(
            result(),
            frame_sequence=2,
            frame_captured_monotonic_s=now,
            robot_state=robot_state(124.5, 224.5, captured=now),
        )
        np.testing.assert_allclose(session.active_delta_world_xy(), [0.5, 0.5])
        self.assertEqual(session.phase, TransferPhase.WAITING_PICK_ALIGNMENT)

    def test_source_only_click_can_start_pickup_navigation(self) -> None:
        session = WaferTransferSession(geometry())
        self._update_overview_frames(session)
        session.select_source("P11")
        session.set_registration(registration())
        snapshot = session.snapshot()
        self.assertEqual(TransferPhase.SOURCE_READY, session.phase)
        self.assertTrue(snapshot["tracking_ready"])
        self.assertFalse(snapshot["route_ready"])
        session.start_tracking()
        self.assertEqual(TransferPhase.TRACKING_PICK, session.phase)
        np.testing.assert_allclose(session.active_delta_world_xy(), [5.0, 5.0])

    def test_source_selection_requires_three_of_five_current_normal_frames(self) -> None:
        session = WaferTransferSession(geometry())
        self._update_overview_frames(session, count=2)
        with self.assertRaisesRegex(ValueError, "temporally stable"):
            session.select_source("P11")
        self._update_overview_frames(session, count=1)
        session.select_source("P11")
        consensus = session.snapshot()["source_consensus"]
        self.assertEqual(3, consensus["occupied_frame_count"])
        self.assertTrue(consensus["passed"])

    def test_repeated_frame_sequence_cannot_satisfy_consensus(self) -> None:
        session = WaferTransferSession(geometry())
        captured = time.monotonic()
        for _ in range(5):
            session.update_overview(
                result(),
                frame_sequence=7,
                frame_captured_monotonic_s=captured,
                robot_state=robot_state(120.0, 220.0, captured=captured),
            )
        consensus = session.source_consensus("P11")
        self.assertEqual(1, consensus["observed_frame_count"])
        self.assertFalse(consensus["passed"])
        with self.assertRaisesRegex(ValueError, "temporally stable"):
            session.select_source("P11")

    def test_close_range_contract_blocks_until_real_evidence_arrives(self) -> None:
        session = self._ready_session()
        session.start_tracking()
        now = time.monotonic()
        session.update_overview(
            result(),
            frame_sequence=2,
            frame_captured_monotonic_s=now,
            robot_state=robot_state(125.0, 225.0, captured=now),
        )
        unavailable = CloseRangeSlotObservation.unavailable(
            CloseRangeOperation.PICK,
            "P11",
            "not calibrated",
        )
        session.update_close_range(unavailable, robot_state=robot_state(125.0, 225.0))
        self.assertEqual(session.phase, TransferPhase.WAITING_PICK_ALIGNMENT)

        captured = time.monotonic()
        aligned = pick_observation(captured=captured)
        session.update_close_range(
            aligned,
            robot_state=robot_state(125.0, 225.0, captured=captured),
        )
        self.assertEqual(session.phase, TransferPhase.VERIFYING_PICK)
        session.record_pick_verification(
            lifted_wafer_detected=True,
            source_now_empty=True,
        )
        self.assertEqual(session.phase, TransferPhase.PICKED)

    def test_place_requires_insertability_and_two_sided_verification(self) -> None:
        session = self._ready_session()
        session.start_tracking()
        now = time.monotonic()
        session.update_overview(
            result(),
            frame_sequence=2,
            frame_captured_monotonic_s=now,
            robot_state=robot_state(125.0, 225.0, captured=now),
        )
        pick_time = time.monotonic()
        session.update_close_range(
            pick_observation(captured=pick_time),
            robot_state=robot_state(125.0, 225.0, captured=pick_time),
        )
        session.record_pick_verification(
            lifted_wafer_detected=True,
            source_now_empty=True,
        )
        session.start_place_tracking()
        # P22 is (50,50) in T, therefore (150,250) in W.
        now = time.monotonic()
        session.update_overview(
            result(),
            frame_sequence=3,
            frame_captured_monotonic_s=now,
            robot_state=robot_state(150.0, 250.0, captured=now),
        )
        self.assertEqual(session.phase, TransferPhase.WAITING_PLACE_ALIGNMENT)
        place_time = time.monotonic()
        session.update_close_range(
            place_observation(
                state=CloseRangeState.ALIGNED,
                captured=place_time,
            ),
            robot_state=robot_state(150.0, 250.0, captured=place_time),
        )
        self.assertEqual(session.phase, TransferPhase.WAITING_PLACE_ALIGNMENT)
        incomplete_time = time.monotonic()
        session.update_close_range(
            place_observation(
                state=CloseRangeState.INSERTABLE,
                captured=incomplete_time,
                complete=False,
            ),
            robot_state=robot_state(150.0, 250.0, captured=incomplete_time),
        )
        self.assertEqual(session.phase, TransferPhase.WAITING_PLACE_ALIGNMENT)
        self.assertFalse(
            session.close_range_gates["affirmative_visual_evidence"]["passed"]
        )
        over_limit_time = time.monotonic()
        session.update_close_range(
            replace(
                place_observation(
                    state=CloseRangeState.INSERTABLE,
                    captured=over_limit_time,
                ),
                maximum_center_error_px=0.05,
            ),
            robot_state=robot_state(150.0, 250.0, captured=over_limit_time),
        )
        self.assertEqual(session.phase, TransferPhase.WAITING_PLACE_ALIGNMENT)
        self.assertIn(
            "center_error_exceeds_limit",
            session.close_range.evidence_errors(),
        )
        insertable_time = time.monotonic()
        session.update_close_range(
            place_observation(
                state=CloseRangeState.INSERTABLE,
                captured=insertable_time,
            ),
            robot_state=robot_state(150.0, 250.0, captured=insertable_time),
        )
        self.assertEqual(session.phase, TransferPhase.READY_TO_PLACE)
        session.record_place_verification(
            lifted_wafer_absent=True,
            destination_now_occupied=True,
        )
        self.assertEqual(session.phase, TransferPhase.COMPLETE)

    def test_stale_or_unsynchronised_robot_state_disables_tracking(self) -> None:
        session = WaferTransferSession(geometry())
        self._update_overview_frames(
            session,
            robot_captured_offset_s=-2.0,
        )
        session.select_source("P11")
        session.select_destination("P22")
        session.set_registration(registration())
        snapshot = session.snapshot()
        self.assertFalse(snapshot["tracking_ready"])
        self.assertIsNone(snapshot["active_delta_world_xy_mm"])
        with self.assertRaises(RuntimeError):
            session.start_tracking()

    def test_camera1_invalidation_discards_stale_pass_and_blocks_active_lock(
        self,
    ) -> None:
        session = self._ready_session()
        session.start_tracking()
        session.invalidate_overview("synthetic camera timeout")
        snapshot = session.snapshot()
        self.assertEqual(TransferPhase.BLOCKED, session.phase)
        self.assertFalse(snapshot["tracking_ready"])
        self.assertIsNone(snapshot["source_state"])
        self.assertIsNone(snapshot["robot_state"])
        self.assertEqual([], snapshot["source_consensus"]["states"])
        self.assertIn("camera1 overview invalidated", snapshot["block_reason"])

    def test_runtime_camera1_invalidation_clears_registration(self) -> None:
        runtime = LiveWaferTransferRuntime(PROJECT_ROOT)
        runtime.session.set_registration(registration())
        runtime.invalidate_camera1("synthetic stale frame")
        snapshot = runtime.snapshot()
        self.assertIsNone(snapshot["registration"])
        self.assertFalse(snapshot["coordinate_ready"])
        self.assertFalse(snapshot["tracking_ready"])
        self.assertFalse(snapshot["robot_motion_authorized"])

    def test_close_range_requires_fresh_timestamp_and_matching_j3(self) -> None:
        session = self._ready_session()
        session.start_tracking()
        now = time.monotonic()
        session.update_overview(
            result(),
            frame_sequence=2,
            frame_captured_monotonic_s=now,
            robot_state=robot_state(125.0, 225.0, captured=now),
        )
        stale = now - 2.0
        session.update_close_range(
            pick_observation(captured=stale),
            robot_state=robot_state(125.0, 225.0, captured=stale),
        )
        self.assertEqual(session.phase, TransferPhase.WAITING_PICK_ALIGNMENT)
        self.assertFalse(session.close_range_gates["fresh_close_range_frame"]["passed"])

        current = time.monotonic()
        session.update_close_range(
            pick_observation(captured=current, robot_j3_mm=-27.0),
            robot_state=robot_state(125.0, 225.0, captured=current)
            | {"joints": [20.0, 30.0, -26.0, 10.0]},
        )
        self.assertEqual(session.phase, TransferPhase.WAITING_PICK_ALIGNMENT)
        self.assertFalse(
            session.close_range_gates["close_range_j3_consistency"]["passed"]
        )

    def test_live_runtime_uses_checked_silicon_profile_and_never_authorizes_motion(
        self,
    ) -> None:
        runtime = LiveWaferTransferRuntime(PROJECT_ROOT)
        self.assertEqual(
            "silicon_detection_0820_geometry_robust",
            runtime.silicon_detection_config.profile_name,
        )
        snapshot = runtime.snapshot()
        self.assertFalse(snapshot["robot_motion_authorized"])
        self.assertEqual(
            "silicon_detection_0820_geometry_robust",
            snapshot["locked_inputs"]["silicon_detection_profile"],
        )


if __name__ == "__main__":
    unittest.main()
