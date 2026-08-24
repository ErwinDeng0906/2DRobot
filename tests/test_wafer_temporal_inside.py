"""Safety regressions for temporal warning-to-occupied recovery."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.wafer_shape_quality import DEFAULT_WAFER_QUALITY
from scara.vision.wafer_temporal_inside import (
    TemporalInsideFrameEvidence,
    evaluate_multiview_inside_latch,
    evaluate_temporal_inside_window,
)


BOX = ((45.0, 45.0), (145.0, 45.0), (145.0, 145.0), (45.0, 145.0))


def _frame(
    index: int,
    *,
    base: float = 5.0,
    refined: float | None = 6.0,
    refined_evidence: str = "inside",
    raw_state: str = "warning",
    flags: tuple[str, ...] = (),
) -> TemporalInsideFrameEvidence:
    return TemporalInsideFrameEvidence(
        frame_id=index,
        raw_state=raw_state,
        found=True,
        center_patch_px=(95.0 + 0.1 * index, 95.0),
        box_patch_px=BOX,
        yaw_deg=1.0 + 0.1 * index,
        flags=flags,
        boundary_evidence=refined_evidence,
        base_clearance_px=base,
        refined_clearance_px=refined,
        base_boundary_evidence="inside",
        refined_boundary_evidence=refined_evidence,
        base_contour_depth_px=0.5,
        base_contour_support_px=2,
        base_contour_area_ratio=0.001,
        refined_contour_depth_px=0.5,
        refined_contour_support_px=2,
        refined_contour_area_ratio=0.001,
    )


class TemporalInsideWindowTests(unittest.TestCase):
    def test_stable_refined_projection_confirms_locally(self) -> None:
        result = evaluate_temporal_inside_window(
            "P20", [_frame(index) for index in range(1, 6)], DEFAULT_WAFER_QUALITY
        )
        self.assertTrue(result.candidate)
        self.assertTrue(result.locally_confirmed)
        self.assertEqual("refined", result.projection.source)
        self.assertEqual(5, result.projection.weak_contour_frame_count)

    def test_stable_refined_outside_cannot_be_overruled_by_positive_base(self) -> None:
        rows = [
            _frame(
                index,
                base=4.0,
                refined=-6.0,
                refined_evidence="strong_outside",
            )
            for index in range(1, 6)
        ]
        result = evaluate_temporal_inside_window(
            "P50", rows, DEFAULT_WAFER_QUALITY
        )
        self.assertFalse(result.candidate)
        self.assertEqual("refined", result.projection.source)
        self.assertEqual(5, result.projection.strong_outside_frame_count)

    def test_unstable_refined_projection_uses_stable_base(self) -> None:
        refined = (8.0, -1.0, -0.7, 4.0, -0.1)
        rows = [
            _frame(index, base=4.0 + 0.05 * index, refined=value)
            for index, value in enumerate(refined, 1)
        ]
        result = evaluate_temporal_inside_window(
            "P01", rows, DEFAULT_WAFER_QUALITY
        )
        self.assertTrue(result.candidate)
        self.assertTrue(result.locally_confirmed)
        self.assertEqual("base_after_refined_unstable", result.projection.source)

    def test_base_only_small_margin_requires_multiview_latch(self) -> None:
        result = evaluate_temporal_inside_window(
            "P20",
            [_frame(index, base=2.5, refined=None) for index in range(1, 6)],
            DEFAULT_WAFER_QUALITY,
        )
        self.assertTrue(result.candidate)
        self.assertFalse(result.locally_confirmed)
        self.assertEqual("base_only_requires_multiview_latch", result.status)


class MultiViewLatchTests(unittest.TestCase):
    def _group(self, slot: str, index: int, *, outside: bool = False):
        rows = [
            _frame(
                frame_index,
                refined=-6.0 if outside else 6.0,
                refined_evidence="strong_outside" if outside else "inside",
                raw_state="outside_slot" if outside else "occupied",
            )
            for frame_index in range(index * 5, index * 5 + 5)
        ]
        return evaluate_temporal_inside_window(slot, rows, DEFAULT_WAFER_QUALITY)

    def test_inside_supported_multiview_history_authorizes(self) -> None:
        groups = [self._group("P20", index) for index in range(5)]
        result = evaluate_multiview_inside_latch(
            "P20", groups, DEFAULT_WAFER_QUALITY
        )
        self.assertTrue(result.authorized)
        self.assertEqual("authorized", result.status)

    def test_outside_dominant_multiview_history_blocks(self) -> None:
        groups = [
            self._group("P50", index, outside=index < 4) for index in range(5)
        ]
        result = evaluate_multiview_inside_latch(
            "P50", groups, DEFAULT_WAFER_QUALITY
        )
        self.assertFalse(result.authorized)
        self.assertGreater(result.strong_outside_group_ratio, 0.20)


if __name__ == "__main__":
    unittest.main()
