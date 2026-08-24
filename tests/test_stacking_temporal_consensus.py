"""Boundary and five-frame regressions for second-wafer evidence."""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.stacking_temporal_consensus import (
    FiveFrameLShapeStackingTracker,
    LShapeFrameEvidence,
    evaluate_l_shape_window,
)
from scara.vision.tray_occupancy import SlotDecision, SlotState
from scara.vision.wafer_shape_quality import (
    DEFAULT_WAFER_QUALITY,
    SecondaryWaferCandidate,
    WaferObservation,
    _secondary_candidate_geometry,
)


PRIMARY = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))


def _shifted_box(dx: float, *, side: float = 100.0):
    return (
        (dx, 0.0),
        (dx + side, 0.0),
        (dx + side, side),
        (dx, side),
    )


def _frame(frame_id: int, *, dx: float = 20.0, dy: float = 10.0, box_dx: float = 0.0):
    return LShapeFrameEvidence(
        frame_id=frame_id,
        candidate_box_patch_px=(
            (box_dx, 0.0),
            (box_dx + 40.0, 0.0),
            (box_dx + 40.0, 40.0),
            (box_dx, 40.0),
        ),
        relative_center_offset_px=(dx, dy),
    )


@dataclass(frozen=True)
class _ProjectionStub:
    slot_key: str = "P22"
    center_px: tuple[float, float] = (64.0, 64.0)


@dataclass(frozen=True)
class _AnalysisStub:
    projection: _ProjectionStub
    wafer: WaferObservation
    decision: SlotDecision
    wafer_secondary_boxes_image_px: tuple[tuple[tuple[float, float], ...], ...]


@dataclass(frozen=True)
class _ResultStub:
    analysis_quality_passed: bool
    quality_passed: bool
    slots: tuple[_AnalysisStub, ...]
    summary: dict[str, int]
    annotated_image: np.ndarray
    stacking_temporal: dict[str, object]
    coordinate_mapping_allowed: bool = False
    robot_correction_allowed: bool = False


def _wafer_with_l_candidate(*, outside: bool = False) -> WaferObservation:
    candidate = SecondaryWaferCandidate(
        source="l_shape",
        box_patch_px=((40.0, 40.0), (80.0, 40.0), (80.0, 80.0), (40.0, 80.0)),
        rectangularity=None,
        solidity=None,
        aspect_ratio=None,
        overlap_ratio=0.60,
        protrusion_depth_px=10.0,
        relative_center_offset_px=(20.0, 10.0),
        accepted=True,
        rejection_reason=None,
    )
    return replace(
        WaferObservation.not_found(),
        found=True,
        quality="warning",
        flags=("l_shape_stacking_candidate", "l_shaped_overlap_corner"),
        outside_slot=outside,
        secondary_boxes_patch_px=(candidate.box_patch_px,),
        secondary_candidates=(candidate,),
    )


def _tracker_result(*, with_candidate: bool, outside: bool = False) -> _ResultStub:
    wafer = _wafer_with_l_candidate(outside=outside) if with_candidate else replace(
        _wafer_with_l_candidate(outside=outside),
        flags=("outside_slot",) if outside else (),
        secondary_boxes_patch_px=(),
        secondary_candidates=(),
    )
    state = SlotState.OUTSIDE_SLOT if outside else SlotState.WARNING
    decision = SlotDecision("P22", state, True, False, "test", ())
    analysis = _AnalysisStub(
        projection=_ProjectionStub(),
        wafer=wafer,
        decision=decision,
        wafer_secondary_boxes_image_px=(
            (((40.0, 40.0), (80.0, 40.0), (80.0, 80.0), (40.0, 80.0)),)
            if with_candidate
            else ()
        ),
    )
    return _ResultStub(
        analysis_quality_passed=True,
        quality_passed=True,
        slots=(analysis,),
        summary={},
        annotated_image=np.zeros((128, 128, 3), dtype=np.uint8),
        stacking_temporal={},
    )


class SecondaryCandidateGateTests(unittest.TestCase):
    def test_overlap_boundaries_are_inclusive(self) -> None:
        at_minimum = _secondary_candidate_geometry(
            "quadrilateral", _shifted_box(80.0), PRIMARY, DEFAULT_WAFER_QUALITY
        )
        at_maximum = _secondary_candidate_geometry(
            "quadrilateral", _shifted_box(8.0), PRIMARY, DEFAULT_WAFER_QUALITY
        )
        self.assertAlmostEqual(0.20, at_minimum.overlap_ratio, places=6)
        self.assertAlmostEqual(0.92, at_maximum.overlap_ratio, places=6)
        self.assertTrue(at_minimum.accepted)
        self.assertTrue(at_maximum.accepted)

    def test_overlap_outside_limits_has_specific_rejection_reason(self) -> None:
        adjacent = _secondary_candidate_geometry(
            "quadrilateral", _shifted_box(80.01), PRIMARY, DEFAULT_WAFER_QUALITY
        )
        reflection = _secondary_candidate_geometry(
            "quadrilateral", _shifted_box(7.99), PRIMARY, DEFAULT_WAFER_QUALITY
        )
        self.assertEqual("adjacent_slot_interference", adjacent.rejection_reason)
        self.assertEqual("contained_reflection", reflection.rejection_reason)

    def test_three_pixel_protrusion_boundary_is_inclusive(self) -> None:
        config = replace(
            DEFAULT_WAFER_QUALITY,
            stacked_candidate_max_overlap_ratio=1.0,
        )
        below = _secondary_candidate_geometry(
            "l_shape", _shifted_box(2.99), PRIMARY, config
        )
        boundary = _secondary_candidate_geometry(
            "l_shape", _shifted_box(3.0), PRIMARY, config
        )
        self.assertEqual("insufficient_protrusion", below.rejection_reason)
        self.assertTrue(boundary.accepted)


class FiveFrameLShapeConsensusTests(unittest.TestCase):
    def test_stable_three_of_five_confirms(self) -> None:
        rows = [_frame(1), LShapeFrameEvidence(2), _frame(3), LShapeFrameEvidence(4), _frame(5)]
        result = evaluate_l_shape_window("P22", rows, DEFAULT_WAFER_QUALITY)
        self.assertTrue(result.confirmed)
        self.assertEqual("confirmed", result.status)
        self.assertEqual(3, result.l_shape_support_count)
        self.assertIsNotNone(result.max_relative_center_jitter_px)
        self.assertAlmostEqual(0.0, float(result.max_relative_center_jitter_px))
        self.assertAlmostEqual(1.0, result.median_pairwise_iou or 0.0)

    def test_common_motion_does_not_change_relative_center_jitter(self) -> None:
        rows = [
            _frame(index, dx=20.0, dy=10.0, box_dx=float(2 * index))
            for index in range(1, 6)
        ]
        result = evaluate_l_shape_window("P22", rows, DEFAULT_WAFER_QUALITY)
        self.assertTrue(result.confirmed)
        self.assertIsNotNone(result.max_relative_center_jitter_px)
        self.assertAlmostEqual(0.0, float(result.max_relative_center_jitter_px))

    def test_relative_center_motion_over_five_pixels_rejects(self) -> None:
        rows = [
            _frame(1, dx=20.0),
            _frame(2, dx=20.0),
            _frame(3, dx=30.5),
            LShapeFrameEvidence(4),
            LShapeFrameEvidence(5),
        ]
        result = evaluate_l_shape_window("P50", rows, DEFAULT_WAFER_QUALITY)
        self.assertFalse(result.confirmed)
        self.assertEqual("relative_center_unstable", result.status)
        self.assertGreater(result.max_relative_center_jitter_px or 0.0, 5.0)

    def test_low_candidate_box_iou_rejects(self) -> None:
        rows = [
            _frame(1, box_dx=0.0),
            _frame(2, box_dx=30.0),
            _frame(3, box_dx=60.0),
            LShapeFrameEvidence(4),
            LShapeFrameEvidence(5),
        ]
        result = evaluate_l_shape_window("P50", rows, DEFAULT_WAFER_QUALITY)
        self.assertFalse(result.confirmed)
        self.assertEqual("candidate_box_unstable", result.status)
        self.assertLess(result.median_pairwise_iou or 1.0, 0.60)

    def test_support_shortage_and_duplicate_frames_do_not_confirm(self) -> None:
        insufficient = evaluate_l_shape_window(
            "P50",
            [_frame(1), _frame(2), LShapeFrameEvidence(3), LShapeFrameEvidence(4), LShapeFrameEvidence(5)],
            DEFAULT_WAFER_QUALITY,
        )
        duplicate = evaluate_l_shape_window(
            "P50",
            [_frame(1), _frame(1), _frame(3), LShapeFrameEvidence(4), LShapeFrameEvidence(5)],
            DEFAULT_WAFER_QUALITY,
        )
        self.assertEqual("insufficient_support", insufficient.status)
        self.assertFalse(insufficient.confirmed)
        self.assertEqual("duplicate_frame", duplicate.status)
        self.assertFalse(duplicate.confirmed)

    def test_incomplete_window_stays_pending(self) -> None:
        result = evaluate_l_shape_window(
            "P22", [_frame(1), _frame(2), _frame(3), _frame(4)], DEFAULT_WAFER_QUALITY
        )
        self.assertEqual("window_pending", result.status)
        self.assertFalse(result.confirmed)

    def test_tracker_promotes_only_after_complete_window_without_changing_safety(self) -> None:
        tracker = FiveFrameLShapeStackingTracker(DEFAULT_WAFER_QUALITY)
        result = None
        for frame_id, candidate in enumerate((True, False, True, False, True), 1):
            result = tracker.update(
                _tracker_result(with_candidate=candidate), frame_id=frame_id
            )
        assert result is not None
        self.assertEqual(SlotState.STACKED, result.slots[0].decision.state)
        self.assertTrue(result.stacking_temporal["P22"].confirmed)
        self.assertFalse(result.coordinate_mapping_allowed)
        self.assertFalse(result.robot_correction_allowed)

    def test_tracker_uses_current_outside_evidence_after_confirmation(self) -> None:
        tracker = FiveFrameLShapeStackingTracker(DEFAULT_WAFER_QUALITY)
        result = None
        for frame_id, candidate in enumerate((True, False, True, False), 1):
            result = tracker.update(
                _tracker_result(with_candidate=candidate), frame_id=frame_id
            )
        result = tracker.update(
            _tracker_result(with_candidate=True, outside=True), frame_id=5
        )
        self.assertEqual(
            SlotState.STACKED_OUTSIDE_SLOT, result.slots[0].decision.state
        )


if __name__ == "__main__":
    unittest.main()
