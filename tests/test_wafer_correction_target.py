from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.tray_occupancy import SlotState
from scara.vision.wafer_correction_target import (
    SLOT_QUADRILATERAL_CENTER_SOURCE,
    aggregate_nearest_outside_wafer,
    extract_outside_wafer_candidates,
)


GEOMETRY = {
    "slots": {
        "P00": [10.0, -5.0, -2.0],
        "P01": [10.0, -30.0, -2.0],
        "P10": [-15.0, -5.0, -2.0],
    }
}


def _slot(
    slot_key: str,
    state: SlotState,
    center: object = (0.0, 0.0, -2.0),
    *,
    found: object = True,
    correction_center_valid: object = True,
    correction_outside: object | None = None,
    box_image: object = (
        (40.0, 40.0),
        (100.0, 40.0),
        (100.0, 100.0),
        (40.0, 100.0),
    ),
    center_image: object = (70.0, 70.0),
) -> SimpleNamespace:
    if correction_outside is None:
        correction_outside = state is SlotState.OUTSIDE_SLOT
    return SimpleNamespace(
        projection=SimpleNamespace(slot_key=slot_key),
        decision=SimpleNamespace(state=state),
        wafer=SimpleNamespace(found=found),
        wafer_box_image_px=box_image,
        wafer_center_image_px=center_image,
        wafer_center_T_mm=center,
        wafer_correction_outside_slot=correction_outside,
        wafer_correction_center_valid=correction_center_valid,
        wafer_correction_center_reason=(
            "ok" if correction_center_valid is True else "refinement_failed"
        ),
        wafer_center_refinement={
            "success": correction_center_valid is True,
            "reason": (
                "ok" if correction_center_valid is True else "refinement_failed"
            ),
        },
    )


def _result(*slots: SimpleNamespace, **gates: object) -> SimpleNamespace:
    values = {
        "success": True,
        "quality_passed": True,
        "coordinate_mapping_allowed": True,
        "failure_reason": None,
    }
    values.update(gates)
    return SimpleNamespace(slots=slots, **values)


def _candidate(slot_key: str, x: float, y: float, z: float = -2.0) -> dict:
    return {
        "slot_key": slot_key,
        "center_T_mm": [x, y, z],
        # Deliberately wrong: aggregation must recompute this from P00.
        "distance_to_p00_mm": 9999.0,
        "center_source": "expanded_roi_full_contour_min_area_rect",
        "refinement": {"success": True, "reason": "ok"},
    }


class ExtractOutsideWaferCandidateTests(unittest.TestCase):
    def test_accepts_correction_outside_candidates_and_sorts_from_p00(self) -> None:
        tray_result = _result(
            _slot("P10", SlotState.OUTSIDE_SLOT, (13.0, -1.0, -2.0)),
            _slot("P01", SlotState.OUTSIDE_SLOT, (7.0, -1.0, -2.0)),
            _slot("P00", SlotState.OCCUPIED, (10.0, -5.0, -2.0)),
        )

        candidates = extract_outside_wafer_candidates(tray_result, GEOMETRY)

        self.assertEqual(["P01", "P10"], [row["slot_key"] for row in candidates])
        self.assertTrue(
            all(row["distance_to_p00_mm"] == 5.0 for row in candidates)
        )
        json.dumps(candidates, allow_nan=False)

    def test_any_stacked_outside_slot_blocks_the_whole_frame(self) -> None:
        tray_result = _result(
            _slot("P12", SlotState.STACKED_OUTSIDE_SLOT, (10.1, -5.0, -2.0)),
            _slot("P01", SlotState.OUTSIDE_SLOT, (7.0, -1.0, -2.0)),
        )
        with self.assertRaisesRegex(ValueError, "P12.*P00"):
            extract_outside_wafer_candidates(tray_result, GEOMETRY)

    def test_requires_all_three_vision_quality_gates(self) -> None:
        for gate in ("success", "quality_passed", "coordinate_mapping_allowed"):
            with self.subTest(gate=gate):
                tray_result = _result(
                    _slot("P00", SlotState.OUTSIDE_SLOT), **{gate: False}
                )
                with self.assertRaisesRegex(ValueError, gate):
                    extract_outside_wafer_candidates(tray_result, GEOMETRY)

        missing_gate = _result(_slot("P00", SlotState.OUTSIDE_SLOT))
        del missing_gate.coordinate_mapping_allowed
        with self.assertRaisesRegex(ValueError, "coordinate_mapping_allowed"):
            aggregate_nearest_outside_wafer([missing_gate] * 5, GEOMETRY)

    def test_correction_specific_outside_and_exact_out_fallback_are_authoritative(self) -> None:
        tray_result = _result(
            _slot(
                "P01",
                SlotState.UNKNOWN,
                (11.0, -5.0, -2.0),
                found=False,
                correction_outside=True,
            )
        )
        candidates = extract_outside_wafer_candidates(tray_result, GEOMETRY)
        self.assertEqual(["P01"], candidates[0]["source_slot_keys"])

        legacy_out = _result(
            _slot(
                "P01",
                SlotState.OUTSIDE_SLOT,
                (12.0, -5.0, -2.0),
                correction_center_valid=False,
                correction_outside=False,
            )
        )
        fallback = extract_outside_wafer_candidates(legacy_out, GEOMETRY)
        self.assertEqual(
            SLOT_QUADRILATERAL_CENTER_SOURCE,
            fallback[0]["center_source"],
        )
        self.assertEqual([70.0, 70.0], fallback[0]["center_image_px"])
        self.assertEqual(
            "ok_outside_slot_fitted_quadrilateral",
            fallback[0]["refinement"]["reason"],
        )

    def test_cross_slot_candidate_still_requires_successful_full_contour_refinement(self) -> None:
        with self.assertRaisesRegex(ValueError, "完整轮廓中心精修未通过"):
            extract_outside_wafer_candidates(
                _result(
                    _slot(
                        "P01",
                        SlotState.UNKNOWN,
                        correction_center_valid=False,
                        correction_outside=True,
                    )
                ),
                GEOMETRY,
            )
        with self.assertRaisesRegex(ValueError, "wafer_center_T_mm"):
            extract_outside_wafer_candidates(
                _result(
                    _slot("P01", SlotState.OUTSIDE_SLOT, (math.nan, 0.0, -2.0))
                ),
                GEOMETRY,
            )

    def test_exact_out_fallback_rejects_bad_quadrilateral_or_center_mismatch(self) -> None:
        for slot, expected in (
            (
                _slot(
                    "P01",
                    SlotState.OUTSIDE_SLOT,
                    correction_center_valid=False,
                    correction_outside=False,
                    box_image=((0.0, 0.0),) * 4,
                ),
                "退化",
            ),
            (
                _slot(
                    "P01",
                    SlotState.OUTSIDE_SLOT,
                    correction_center_valid=False,
                    correction_outside=False,
                    center_image=(75.0, 70.0),
                ),
                "不一致",
            ),
        ):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    extract_outside_wafer_candidates(
                        _result(slot), GEOMETRY
                    )

    def test_exact_out_uses_true_perspective_diagonal_intersection(self) -> None:
        # This trapezoid's diagonal intersection is (5, 5), while the mean of
        # its four vertices is (5, 4).  The fallback must use the actual
        # projective diagonal intersection requested by the operator.
        candidates = extract_outside_wafer_candidates(
            _result(
                _slot(
                    "P01",
                    SlotState.OUTSIDE_SLOT,
                    correction_center_valid=False,
                    correction_outside=False,
                    box_image=(
                        (0.0, 0.0),
                        (10.0, 0.0),
                        (8.0, 8.0),
                        (2.0, 8.0),
                    ),
                    center_image=(5.0, 5.0),
                )
            ),
            GEOMETRY,
        )
        self.assertEqual([5.0, 5.0], candidates[0]["center_image_px"])
        self.assertNotEqual(
            [5.0, 4.0], candidates[0]["center_image_px"]
        )

    def test_serialized_state_string_is_not_accepted_as_authoritative_enum(self) -> None:
        tray_result = _result(
            _slot("P01", "outside_slot", (10.0, -6.0, -2.0))  # type: ignore[arg-type]
        )
        self.assertEqual([], extract_outside_wafer_candidates(tray_result, GEOMETRY))

    def test_geometry_p00_is_required_and_finite(self) -> None:
        tray_result = _result(_slot("P01", SlotState.OUTSIDE_SLOT))
        for geometry in ({}, {"slots": {}}, {"slots": {"P00": [0, math.inf, 0]}}):
            with self.subTest(geometry=geometry):
                with self.assertRaisesRegex(ValueError, "P00"):
                    extract_outside_wafer_candidates(tray_result, geometry)


class AggregateNearestOutsideWaferTests(unittest.TestCase):
    def test_locks_mean_center_when_slot_labels_change_across_frames(self) -> None:
        frames = [
            [_candidate("P01", 11.00, -5.00), _candidate("P10", 30.0, 30.0)],
            [_candidate("P10", 11.10, -5.10)],
            [_candidate("P01", 10.90, -4.95)],
            [_candidate("P10", 11.05, -5.05)],
            [_candidate("P01", 10.95, -4.90)],
        ]

        target = aggregate_nearest_outside_wafer(frames, GEOMETRY)

        self.assertEqual([11.0, -5.0, -2.0], target["center_T_mm"])
        self.assertAlmostEqual(1.0, target["distance_to_p00_mm"])
        self.assertEqual(
            ["P01", "P10", "P01", "P10", "P01"],
            target["source_slot_keys"],
        )
        self.assertEqual(["P01", "P10"], target["unique_source_slot_keys"])
        self.assertLessEqual(
            target["stability"]["maximum_center_residual_mm"], 0.75
        )
        # Both public APIs return JSON-safe simple mappings/lists.
        json.dumps(target, allow_nan=False)

    def test_selects_each_frames_nearest_candidate_with_slot_key_tie_break(self) -> None:
        frames = [
            [_candidate("P10", 13.0, -1.0), _candidate("P01", 7.0, -1.0)]
            for _ in range(5)
        ]
        target = aggregate_nearest_outside_wafer(frames, GEOMETRY)
        self.assertEqual(["P01"] * 5, target["source_slot_keys"])
        self.assertEqual([7.0, -1.0, -2.0], target["center_T_mm"])

    def test_accepts_raw_tray_results_and_rejects_any_frame_without_candidate(self) -> None:
        frames = [
            _result(_slot("P01", SlotState.OUTSIDE_SLOT, (11.0, -5.0, -2.0)))
            for _ in range(5)
        ]
        target = aggregate_nearest_outside_wafer(frames, GEOMETRY)
        self.assertEqual([11.0, -5.0, -2.0], target["center_T_mm"])

        frames[3] = _result(
            _slot("P01", SlotState.STACKED_OUTSIDE_SLOT, (11.0, -5.0, -2.0))
        )
        with self.assertRaisesRegex(ValueError, "叠片"):
            aggregate_nearest_outside_wafer(frames, GEOMETRY)

    def test_rejects_wrong_frame_count_and_averages_noisy_centres(self) -> None:
        stable = [[_candidate("P01", 11.0, -5.0)] for _ in range(5)]
        with self.assertRaisesRegex(ValueError, "恰好5帧"):
            aggregate_nearest_outside_wafer(stable[:4], GEOMETRY)

        unstable = list(stable)
        unstable[4] = [_candidate("P01", 12.0, -5.0)]
        target = aggregate_nearest_outside_wafer(unstable, GEOMETRY)
        self.assertEqual([11.2, -5.0, -2.0], target["center_T_mm"])
        self.assertEqual(
            "five_frame_arithmetic_mean", target["aggregation_method"]
        )
        self.assertFalse(target["stability"]["residual_gate_enforced"])
        self.assertAlmostEqual(
            0.8, target["stability"]["maximum_center_residual_mm"]
        )

        with self.assertRaisesRegex(ValueError, "中心不稳定"):
            aggregate_nearest_outside_wafer(
                unstable,
                GEOMETRY,
                maximum_center_residual_mm=0.75,
            )

        with self.assertRaisesRegex(ValueError, "required_frame_count"):
            aggregate_nearest_outside_wafer(
                stable,
                GEOMETRY,
                required_frame_count="five",  # type: ignore[arg-type]
            )

    def test_accepts_sample_mapping_that_contains_extracted_candidates(self) -> None:
        frames = [
            {"outside_wafer_candidates": [_candidate("P01", 11.0, -5.0)]}
            for _ in range(5)
        ]
        target = aggregate_nearest_outside_wafer(frames, GEOMETRY)
        self.assertEqual([11.0, -5.0, -2.0], target["center_T_mm"])

    def test_recomputes_distance_from_geometry_instead_of_trusting_mapping(self) -> None:
        frames = [[_candidate("P01", 13.0, -1.0)] for _ in range(5)]
        target = aggregate_nearest_outside_wafer(frames, GEOMETRY)
        self.assertEqual(5.0, target["distance_to_p00_mm"])
        self.assertEqual(
            [5.0] * 5,
            [
                row["distance_to_p00_mm"]
                for row in target["selected_frame_candidates"]
            ],
        )

    def test_serialized_candidate_requires_allowed_geometry_provenance(self) -> None:
        valid = _candidate("P01", 11.0, -5.0)
        for mutation, expected in (
            ({"center_source": "legacy_slot_patch"}, "center_source"),
            ({"refinement": {"success": False}}, "refinement"),
            ({"refinement": None}, "refinement"),
        ):
            with self.subTest(mutation=mutation):
                candidate = dict(valid)
                candidate.update(mutation)
                with self.assertRaisesRegex(ValueError, expected):
                    aggregate_nearest_outside_wafer(
                        [[candidate] for _ in range(5)], GEOMETRY
                    )

        fallback = _candidate("P01", 11.0, -5.0)
        fallback.update(
            {
                "center_source": SLOT_QUADRILATERAL_CENTER_SOURCE,
                "center_image_px": [70.0, 70.0],
                "refinement": {
                    "success": True,
                    "reason": "ok_outside_slot_fitted_quadrilateral",
                    "source": SLOT_QUADRILATERAL_CENTER_SOURCE,
                    "box_image_px": [
                        [40.0, 40.0],
                        [100.0, 40.0],
                        [100.0, 100.0],
                        [40.0, 100.0],
                    ],
                },
            }
        )
        target = aggregate_nearest_outside_wafer(
            [[fallback] for _ in range(5)], GEOMETRY
        )
        self.assertEqual(
            SLOT_QUADRILATERAL_CENTER_SOURCE, target["center_source"]
        )

    def test_cross_slot_duplicates_are_merged_and_sources_preserved(self) -> None:
        tray_result = _result(
            _slot("P01", SlotState.OUTSIDE_SLOT, (11.0, -5.0, -2.0)),
            _slot(
                "P10",
                SlotState.UNKNOWN,
                (11.2, -5.1, -2.0),
                correction_outside=True,
            ),
        )
        candidates = extract_outside_wafer_candidates(tray_result, GEOMETRY)
        self.assertEqual(1, len(candidates))
        self.assertEqual(["P01", "P10"], candidates[0]["source_slot_keys"])
        target = aggregate_nearest_outside_wafer(
            [candidates for _ in range(5)], GEOMETRY
        )
        self.assertEqual(
            ["P01", "P10"], target["unique_source_slot_keys"]
        )


if __name__ == "__main__":
    unittest.main()
