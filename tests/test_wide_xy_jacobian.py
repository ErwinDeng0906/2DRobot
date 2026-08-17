"""Offline tests for the Task11 wide-area image-error model."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.wide_xy_jacobian import (
    WideXYJacobianQualityConfig,
    evaluate_wide_error_and_jacobian,
    fit_wide_xy_image_model,
    wide_correction_command_xy_mm,
)


TRAIN = [(x, y) for x in (-10, -5, 0, 5, 10) for y in (-10, -5, 0, 5, 10)]
VALID = [
    (x, y)
    for x in (-7.5, -2.5, 2.5, 7.5)
    for y in (-7.5, -2.5, 2.5, 7.5)
]


def synthetic_error(x: float, y: float) -> np.ndarray:
    return np.array(
        [
            2.2 + 3.65 * x + 1.84 * y + 0.010 * x * x - 0.006 * x * y,
            -0.56 + 1.86 * x - 3.76 * y + 0.005 * x * y + 0.008 * y * y,
        ],
        dtype=np.float64,
    )


def samples(*, break_validation: bool = False) -> list[dict]:
    rows: list[dict] = []
    noise = [(-0.04, 0.02), (0.03, -0.01), (0.0, 0.0), (0.02, 0.03), (-0.01, -0.02)]
    for pass_index in (1, 2):
        for x, y in TRAIN:
            for du, dv in noise:
                value = synthetic_error(x, y) + np.array([du, dv])
                rows.append(
                    {
                        "phase": "train",
                        "pass_index": pass_index,
                        "command_offset_xy_mm": [x, y],
                        "image_error_px": value.tolist(),
                        "accepted": True,
                    }
                )
    for x, y in VALID:
        for du, dv in noise:
            value = synthetic_error(x, y) + np.array([du, dv])
            if break_validation and x == 7.5 and y == 7.5:
                value += np.array([4.0, -4.0])
            rows.append(
                {
                    "phase": "validation",
                    "pass_index": 1,
                    "command_offset_xy_mm": [x, y],
                    "image_error_px": value.tolist(),
                    "accepted": True,
                }
            )
    return rows


class WideXYJacobianTests(unittest.TestCase):
    def test_quadratic_model_is_selected_and_recovers_local_jacobian(self) -> None:
        result = fit_wide_xy_image_model(samples(), TRAIN, VALID)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["selected_model_type"], "quadratic")
        prediction, jacobian = evaluate_wide_error_and_jacobian(
            result["selected_model"], [6.0, -4.0]
        )
        expected_j = np.array(
            [
                [3.65 + 0.020 * 6.0 - 0.006 * -4.0, 1.84 - 0.006 * 6.0],
                [1.86 + 0.005 * -4.0, -3.76 + 0.005 * 6.0 + 0.016 * -4.0],
            ]
        )
        self.assertTrue(np.allclose(prediction, synthetic_error(6.0, -4.0), atol=0.05))
        self.assertTrue(np.allclose(jacobian, expected_j, atol=0.02))

    def test_wide_correction_uses_position_dependent_jacobian(self) -> None:
        result = fit_wide_xy_image_model(samples(), TRAIN, VALID)
        offset = np.array([6.0, -4.0])
        error = synthetic_error(*offset)
        correction = wide_correction_command_xy_mm(error, offset, result)
        self.assertIsNotNone(correction)
        _predicted, jacobian = evaluate_wide_error_and_jacobian(
            result["selected_model"], offset
        )
        self.assertTrue(
            np.allclose(error + jacobian @ np.asarray(correction), [0.0, 0.0], atol=1e-8)
        )

    def test_missing_validation_node_fails_closed(self) -> None:
        rows = [
            row
            for row in samples()
            if not (
                row["phase"] == "validation"
                and row["command_offset_xy_mm"] == [7.5, 7.5]
            )
        ]
        result = fit_wide_xy_image_model(rows, TRAIN, VALID)
        self.assertEqual(result["status"], "failure")
        self.assertIn("validation_node_coverage", result["failure_reasons"])

    def test_bad_independent_validation_never_installs_model(self) -> None:
        strict = WideXYJacobianQualityConfig(
            maximum_validation_rms_px=0.75,
            maximum_validation_error_px=1.5,
        )
        result = fit_wide_xy_image_model(
            samples(break_validation=True), TRAIN, VALID, strict
        )
        self.assertEqual(result["status"], "failure")
        self.assertIsNone(result["selected_model_type"])
        self.assertIsNone(
            wide_correction_command_xy_mm([2.0, 1.0], [0.0, 0.0], result)
        )

    def test_all_rejected_second_visit_cannot_be_hidden_by_first_visit(self) -> None:
        rows = samples()
        for row in rows:
            if (
                row["phase"] == "train"
                and row["pass_index"] == 2
                and row["command_offset_xy_mm"] == [5, -5]
            ):
                row["accepted"] = False
                row["image_error_px"] = None
        result = fit_wide_xy_image_model(rows, TRAIN, VALID)
        self.assertEqual(result["status"], "failure")
        self.assertIn("visit_frame_coverage", result["failure_reasons"])
        failed_visit = next(
            visit
            for visit in result["visit_aggregates"]
            if visit["phase"] == "train"
            and visit["pass_index"] == 2
            and visit["command_offset_xy_mm"] == [5.0, -5.0]
        )
        self.assertEqual(failed_visit["frame_count"], 5)
        self.assertEqual(failed_visit["accepted_frame_count"], 0)
        self.assertFalse(failed_visit["usable"])


if __name__ == "__main__":
    unittest.main()
