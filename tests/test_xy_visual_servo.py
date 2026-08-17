from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.xy_image_jacobian import (  # noqa: E402
    REQUIRED_XY_JACOBIAN_QUALITY_GATES,
)
from scara.vision.xy_visual_servo import (  # noqa: E402
    DEFAULT_STAGE7A_CONFIG,
    REQUIRED_STAGE7A_EXTERNAL_GATES,
    build_stage7a_proposal,
    evaluate_stage7a_response,
)


J = np.asarray([[4.0, 0.0], [0.0, -4.0]], dtype=np.float64)
J_INV = np.linalg.inv(J)
ANCHOR_XY = np.asarray([100.0, 200.0], dtype=np.float64)


def _quality_gates() -> dict:
    return {
        name: {"passed": True}
        for name in REQUIRED_XY_JACOBIAN_QUALITY_GATES
    }


def _jacobian_payload() -> dict:
    return {
        "schema_version": 2,
        "status": "success",
        "anchor_target_name": "P22",
        "valid_target_names": ["P22"],
        "coordinate_definition": {
            "anchor_robot_xy_mm": ANCHOR_XY.tolist(),
            "offset_extent_mm": 2.0,
        },
        "fit": {
            "status": "success",
            "j_error_px_per_command_mm": J.tolist(),
            "j_command_mm_per_error_px": J_INV.tolist(),
            "quality_gates": _quality_gates(),
        },
    }


def _external(*, consent: bool = True, **overrides: bool) -> dict:
    result = {name: True for name in REQUIRED_STAGE7A_EXTERNAL_GATES}
    result["operator_consent"] = consent
    result.update(overrides)
    return result


def _measurements(
    center_error_px: tuple[float, float] = (2.0, -1.0),
    *,
    current_delta_xy_mm: tuple[float, float] = (0.5, -0.5),
    after_command_xy_mm: tuple[float, float] = (0.0, 0.0),
    noise_px: float = 0.04,
    id_prefix: str = "pre",
) -> list[dict]:
    offsets = (-2.0, -1.0, 0.0, 1.0, 2.0)
    current_xy = (
        ANCHOR_XY
        + np.asarray(current_delta_xy_mm, dtype=np.float64)
        + np.asarray(after_command_xy_mm, dtype=np.float64)
    )
    rows: list[dict] = []
    for index, factor in enumerate(offsets, start=1):
        rows.append(
            {
                "measurement_id": f"{id_prefix}-frame-{index}",
                "target_name": "P22",
                "accepted": True,
                "image_error_px": [
                    center_error_px[0] + noise_px * factor,
                    center_error_px[1] - noise_px * factor,
                ],
                "jacobian_domain_passed": True,
                "robot_state_age_s": 0.04 + index * 0.005,
                "current_robot_xy_mm": (
                    current_xy
                    + np.asarray([factor * 0.001, -factor * 0.001])
                ).tolist(),
                "current_joints": [30.0, 40.0, -27.0, -4.0],
                "reason": "ok",
            }
        )
    return rows


class Stage7AProposalTests(unittest.TestCase):
    def test_valid_window_uses_gain_vector_limit_and_predicts_endpoint(self) -> None:
        report = build_stage7a_proposal(
            _measurements(),
            _jacobian_payload(),
            external_safety_gates=_external(consent=True),
        )

        self.assertTrue(report["overall_passed"])
        self.assertTrue(report["motion_authorized"])
        self.assertEqual(report["decision"], "authorized_single_step")
        self.assertEqual(report["measurement"]["accepted_frame_count"], 5)
        np.testing.assert_allclose(
            report["measurement"]["median_error_px"], [2.0, -1.0]
        )
        np.testing.assert_allclose(
            report["calculation"]["full_cancellation_correction_xy_mm"],
            [-0.5, -0.25],
        )
        np.testing.assert_allclose(
            report["calculation"]["unclipped_correction_xy_mm"],
            [-0.3, -0.15],
        )
        self.assertTrue(report["calculation"]["was_clamped"])
        command = np.asarray(
            report["calculation"]["commanded_correction_xy_mm"]
        )
        self.assertAlmostEqual(np.linalg.norm(command), 0.25, places=12)
        expected_error = np.asarray([2.0, -1.0]) + J @ command
        np.testing.assert_allclose(
            report["calculation"]["predicted_error_px"], expected_error
        )
        np.testing.assert_allclose(
            report["calculation"]["predicted_endpoint_xy_mm"],
            ANCHOR_XY + [0.5, -0.5] + command,
            atol=1e-12,
        )
        self.assertTrue(
            all(
                {"name", "passed", "actual", "limit", "note"}
                <= set(gate)
                for gate in report["safety_gates"].values()
            )
        )

    def test_operator_must_confirm_and_missing_external_gate_fails_closed(self) -> None:
        awaiting = build_stage7a_proposal(
            _measurements(),
            _jacobian_payload(),
            external_safety_gates=_external(consent=False),
        )
        self.assertTrue(awaiting["ready_for_operator_confirmation"])
        self.assertFalse(awaiting["overall_passed"])
        self.assertFalse(awaiting["motion_authorized"])
        self.assertEqual(awaiting["decision"], "awaiting_operator_confirmation")
        self.assertIn("operator_consent", awaiting["failure_reasons"])

        incomplete = _external(consent=True)
        incomplete.pop("alarm_clear")
        rejected = build_stage7a_proposal(
            _measurements(),
            _jacobian_payload(),
            external_safety_gates=incomplete,
        )
        self.assertFalse(rejected["motion_authorized"])
        self.assertIn("alarm_clear", rejected["failure_reasons"])

        duplicate = _measurements()
        duplicate[1]["measurement_id"] = duplicate[0]["measurement_id"]
        frozen = build_stage7a_proposal(
            duplicate,
            _jacobian_payload(),
            external_safety_gates=_external(consent=True),
        )
        self.assertFalse(frozen["motion_authorized"])
        self.assertIn("unique_measurement_ids", frozen["failure_reasons"])

    def test_measurement_and_domain_failures_are_explained(self) -> None:
        rows = _measurements()
        rows[0]["accepted"] = False
        rows[1]["accepted"] = False
        rows[2]["accepted"] = False
        rows[3]["robot_state_age_s"] = 0.60
        rows[4]["jacobian_domain_passed"] = False
        report = build_stage7a_proposal(
            rows,
            _jacobian_payload(),
            external_safety_gates=_external(),
        )
        self.assertFalse(report["motion_authorized"])
        self.assertIn("minimum_accepted_frames", report["failure_reasons"])
        self.assertIn("stage3_and_jacobian_domain", report["failure_reasons"])
        self.assertIn("robot_state_freshness", report["failure_reasons"])

        outside = build_stage7a_proposal(
            _measurements(
                center_error_px=(-2.0, -1.0),
                current_delta_xy_mm=(1.79, 0.0),
            ),
            _jacobian_payload(),
            external_safety_gates=_external(),
        )
        self.assertFalse(outside["motion_authorized"])
        self.assertIn(
            "predicted_endpoint_inside_local_domain", outside["failure_reasons"]
        )

    def test_directshow_snapshot_latency_below_half_second_is_fresh(self) -> None:
        rows = _measurements()
        for index, age in enumerate((0.296, 0.268, 0.289, 0.277, 0.286)):
            rows[index]["robot_state_age_s"] = age
        report = build_stage7a_proposal(
            rows,
            _jacobian_payload(),
            external_safety_gates=_external(consent=True),
        )
        self.assertTrue(report["safety_gates"]["robot_state_freshness"]["passed"])

    def test_unstable_window_and_bad_model_fail_closed(self) -> None:
        noisy = _measurements(noise_px=1.0)
        unstable = build_stage7a_proposal(
            noisy,
            _jacobian_payload(),
            external_safety_gates=_external(),
        )
        self.assertFalse(unstable["motion_authorized"])
        self.assertIn("error_window_dispersion", unstable["failure_reasons"])

        payload = copy.deepcopy(_jacobian_payload())
        payload["fit"]["quality_gates"][
            next(iter(REQUIRED_XY_JACOBIAN_QUALITY_GATES))
        ]["passed"] = False
        failed = build_stage7a_proposal(
            _measurements(), payload, external_safety_gates=_external()
        )
        self.assertFalse(failed["motion_authorized"])
        self.assertIsNone(
            failed["calculation"]["commanded_correction_xy_mm"]
        )
        self.assertIn("stage5_model_quality", failed["failure_reasons"])

    def test_already_aligned_never_authorizes_a_nonzero_move(self) -> None:
        report = build_stage7a_proposal(
            _measurements(center_error_px=(0.4, -0.2), noise_px=0.01),
            _jacobian_payload(),
            external_safety_gates=_external(),
        )
        self.assertTrue(report["overall_passed"])
        self.assertFalse(report["motion_required"])
        self.assertFalse(report["motion_authorized"])
        self.assertEqual(report["decision"], "already_aligned")
        np.testing.assert_allclose(
            report["calculation"]["commanded_correction_xy_mm"], [0.0, 0.0]
        )


class Stage7AResponseTests(unittest.TestCase):
    def _proposal(self) -> dict:
        return build_stage7a_proposal(
            _measurements(),
            _jacobian_payload(),
            external_safety_gates=_external(),
        )

    def test_response_validation_passes_matching_observed_change(self) -> None:
        proposal = self._proposal()
        command = np.asarray(
            proposal["calculation"]["commanded_correction_xy_mm"]
        )
        before = np.asarray(proposal["measurement"]["median_error_px"])
        after = before + J @ command + np.asarray([0.05, -0.04])
        response = evaluate_stage7a_response(
            proposal,
            _measurements(
                center_error_px=tuple(after.tolist()),
                after_command_xy_mm=tuple(command.tolist()),
                noise_px=0.02,
                id_prefix="post",
            ),
            _jacobian_payload(),
        )
        self.assertTrue(response["passed"])
        np.testing.assert_allclose(
            response["actual_delta_error_px"], after - before
        )
        np.testing.assert_allclose(
            response["predicted_delta_error_px"], J @ command
        )
        np.testing.assert_allclose(response["innovation_px"], [0.05, -0.04])
        self.assertGreater(response["improvement_ratio"], 0.20)

    def test_response_validation_rejects_wrong_direction(self) -> None:
        proposal = self._proposal()
        before = np.asarray(proposal["measurement"]["median_error_px"])
        response = evaluate_stage7a_response(
            proposal,
            _measurements(
                center_error_px=tuple((before * 1.2).tolist()),
                noise_px=0.02,
                id_prefix="post",
            ),
            _jacobian_payload(),
        )
        self.assertFalse(response["passed"])
        self.assertLess(response["improvement_ratio"], 0.0)
        self.assertIn("actual_error_improvement", response["failure_reasons"])
        self.assertIn("response_innovation", response["failure_reasons"])
        self.assertIn("command_tracking", response["failure_reasons"])

    def test_response_validation_rejects_changed_jacobian(self) -> None:
        proposal = self._proposal()
        command = np.asarray(
            proposal["calculation"]["commanded_correction_xy_mm"]
        )
        before = np.asarray(proposal["measurement"]["median_error_px"])
        after = before + J @ command
        changed = _jacobian_payload()
        changed["fit"]["j_error_px_per_command_mm"][0][0] = 4.1
        changed["fit"]["j_command_mm_per_error_px"] = np.linalg.inv(
            np.asarray(changed["fit"]["j_error_px_per_command_mm"])
        ).tolist()
        response = evaluate_stage7a_response(
            proposal,
            _measurements(
                center_error_px=tuple(after.tolist()),
                after_command_xy_mm=tuple(command.tolist()),
                noise_px=0.02,
                id_prefix="post",
            ),
            changed,
        )
        self.assertFalse(response["passed"])
        self.assertIn("proposal_model_unchanged", response["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
