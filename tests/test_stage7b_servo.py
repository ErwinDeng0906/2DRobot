from __future__ import annotations

import unittest

import numpy as np

from scara.pipeline.kinematics import rz_of, solve_joints
from scara.vision.stage7b_servo import (
    DEFAULT_STAGE7B_CONFIG,
    Stage7BConfig,
    build_stage7b_iteration,
)
from scara.vision.wide_xy_jacobian import REQUIRED_WIDE_XY_JACOBIAN_QUALITY_GATES
from scara.vision.xy_image_jacobian import REQUIRED_XY_JACOBIAN_QUALITY_GATES


ANCHOR = [118.333, 272.783]
JOINTS = [30.6646, 84.7845, -27.0046, -4.6268]
RZ = rz_of(JOINTS[0], JOINTS[1], JOINTS[3])


def calibration_payloads():
    J = [[3.65, 1.84], [1.86, -3.76]]
    fine_gates = {name: {"passed": True} for name in REQUIRED_XY_JACOBIAN_QUALITY_GATES}
    fine = {
        "status": "success", "anchor_target_name": "P22", "valid_target_names": ["P22"],
        "coordinate_definition": {"anchor_robot_xy_mm": ANCHOR, "offset_extent_mm": 2.0, "imaging_j3_mm": JOINTS[2], "rz_deg": RZ},
        "fit": {"status": "success", "quality_gates": fine_gates, "j_error_px_per_command_mm": J, "j_command_mm_per_error_px": np.linalg.inv(J).tolist()},
    }
    wide_gates = {name: {"passed": True} for name in REQUIRED_WIDE_XY_JACOBIAN_QUALITY_GATES}
    coefficients = [[2.2, -0.5], [3.65, 1.86], [1.84, -3.76]]
    wide_fit = {"status": "success", "quality_gates": wide_gates, "selected_model": {"model_type": "global_affine", "coefficients_feature_by_error": coefficients}}
    wide = {"status": "success", "coordinate_definition": {"anchor_robot_xy_mm": ANCHOR}, "fit": wide_fit}
    return fine, wide


def samples(xy, error, start=1):
    joints = solve_joints(xy[0], xy[1], JOINTS[2], RZ, JOINTS)
    pose = [xy[0], xy[1], JOINTS[2], 180.0, 0.0, RZ]
    return [{"measurement_id": f"f{start+i}", "accepted": True, "target_name": "P22", "image_error_px": error, "current_robot_xy_mm": xy, "current_joints": joints, "current_pose": pose, "robot_state_age_s": 0.1} for i in range(5)]


def request(xy):
    joints = solve_joints(xy[0], xy[1], JOINTS[2], RZ, JOINTS)
    return {"target_name": "P22", "controller_state": {"joints": joints, "pose": [xy[0], xy[1], JOINTS[2], 180, 0, RZ]}, "external_safety_gates": {name: True for name in ("controller_connected", "controller_enabled", "alarm_clear", "estop_clear", "soft_estop_clear", "controller_idle")}}


class Stage7BServoTests(unittest.TestCase):
    def test_operator_relaxed_gate_policy_is_explicit(self):
        config = DEFAULT_STAGE7B_CONFIG
        self.assertEqual(config.maximum_total_path_mm, 50.0)
        self.assertEqual(config.maximum_error_dispersion_px, 1.5)
        self.assertEqual(config.maximum_peak_deviation_px, 3.0)
        self.assertEqual(config.maximum_robot_xy_spread_mm, 0.2)
        self.assertEqual(config.maximum_joint_spread, 0.2)
        self.assertEqual(config.maximum_robot_state_age_s, 1.0)
        self.assertEqual(config.maximum_request_state_xy_mismatch_mm, 0.2)
        self.assertEqual(config.maximum_request_state_joint_mismatch, 0.2)
        self.assertEqual(config.maximum_anchor_mismatch_mm, 0.5)
        self.assertEqual(config.maximum_response_innovation_px, 2.0)
        self.assertEqual(config.maximum_command_tracking_error_mm, 0.15)
        self.assertEqual(config.arrival_distance_threshold_mm, 1.0)
        self.assertEqual(config.fine_planning_step_limit_mm, 0.24)
        self.assertEqual(config.fine_execution_step_limit_mm, 0.25)
        self.assertEqual(config.coarse_planning_step_limit_mm, 0.74)
        self.assertEqual(config.coarse_execution_step_limit_mm, 0.75)

    def test_selects_wide_outside_local_domain(self):
        fine, wide = calibration_payloads()
        xy = [ANCHOR[0] + 5.0, ANCHOR[1]]
        report = build_stage7b_iteration(samples(xy, [20.0, 8.0]), request(xy), fine, wide, iteration_index=1, cumulative_path_mm=0.0)
        self.assertEqual(report["model_tier"], "coarse_task11")
        self.assertEqual(report["decision"], "move")
        self.assertAlmostEqual(
            np.linalg.norm(report["commanded_correction_xy_mm"]),
            0.74,
            places=9,
        )
        planner_gates = report["planner"]["audit"]["gates"]
        self.assertNotIn(
            "sequential_intermediate_inside_local_domain", planner_gates
        )

    def test_selects_fine_inside_local_domain(self):
        fine, wide = calibration_payloads()
        xy = [ANCHOR[0] + 1.0, ANCHOR[1] - 1.0]
        J = np.asarray(fine["fit"]["j_error_px_per_command_mm"], dtype=float)
        error = -(J @ np.asarray([1.2, 0.0], dtype=float))
        report = build_stage7b_iteration(samples(xy, error.tolist()), request(xy), fine, wide, iteration_index=2, cumulative_path_mm=0.7)
        self.assertEqual(report["model_tier"], "fine_task9")
        self.assertEqual(report["decision"], "move")
        self.assertLessEqual(np.linalg.norm(report["commanded_correction_xy_mm"]), 0.240001)

    def test_remaining_visual_correction_below_one_mm_completes(self):
        fine, wide = calibration_payloads()
        J = np.asarray(fine["fit"]["j_error_px_per_command_mm"], dtype=float)
        desired_full_correction = np.asarray([0.8, 0.0], dtype=float)
        error = -(J @ desired_full_correction)
        xy = list(ANCHOR)
        report = build_stage7b_iteration(
            samples(xy, error.tolist()),
            request(xy),
            fine,
            wide,
            iteration_index=1,
            cumulative_path_mm=0.0,
        )
        self.assertGreater(report["error_norm_px"], 1.0)
        self.assertAlmostEqual(report["remaining_alignment_distance_mm"], 0.8)
        self.assertEqual(report["convergence_reason"], "within_1mm")
        self.assertEqual(report["decision"], "converged")
        self.assertIsNone(report["commanded_correction_xy_mm"])
        self.assertFalse(report["motion_authorized"])

    def test_convergence_completes_without_motion(self):
        fine, wide = calibration_payloads()
        xy = list(ANCHOR)
        report = build_stage7b_iteration(samples(xy, [0.3, -0.2]), request(xy), fine, wide, iteration_index=3, cumulative_path_mm=1.0)
        self.assertEqual(report["decision"], "converged")
        self.assertFalse(report["motion_authorized"])

    def test_previous_divergent_response_rejects(self):
        fine, wide = calibration_payloads()
        xy = list(ANCHOR)
        previous = {"median_error_px": [6.0, 0.0], "current_robot_xy_mm": xy, "commanded_correction_xy_mm": [0.1, 0.0], "predicted_delta_error_px": [-1.0, 0.0]}
        current_xy = [xy[0] + 0.1, xy[1]]
        report = build_stage7b_iteration(samples(current_xy, [7.0, 0.0]), request(current_xy), fine, wide, iteration_index=2, cumulative_path_mm=0.1, previous_iteration=previous)
        self.assertEqual(report["decision"], "reject")
        self.assertFalse(report["safety_gates"]["previous_error_improved"]["passed"])

    def test_outside_wide_domain_rejects(self):
        fine, wide = calibration_payloads()
        xy = [ANCHOR[0] + 9.6, ANCHOR[1]]
        report = build_stage7b_iteration(samples(xy, [20, 5]), request(xy), fine, wide, iteration_index=1, cumulative_path_mm=0.0)
        self.assertEqual(report["decision"], "reject")
        self.assertFalse(report["safety_gates"]["current_inside_supported_domain"]["passed"])

    def test_final_observation_can_converge_but_cannot_exceed_move_budget(self):
        fine, wide = calibration_payloads()
        xy = list(ANCHOR)
        converged = build_stage7b_iteration(
            samples(xy, [0.2, 0.1]), request(xy), fine, wide,
            iteration_index=33, cumulative_path_mm=4.0,
        )
        self.assertEqual(converged["decision"], "converged")
        not_converged = build_stage7b_iteration(
            samples(xy, [8.0, 3.0]), request(xy), fine, wide,
            iteration_index=33, cumulative_path_mm=4.0,
        )
        self.assertEqual(not_converged["decision"], "reject")
        self.assertFalse(
            not_converged["safety_gates"]["motion_iteration_budget"]["passed"]
        )

    def test_ideal_corner_case_switches_tiers_and_converges_within_budget(self):
        fine, wide = calibration_payloads()
        xy = np.asarray([ANCHOR[0] + 9.4, ANCHOR[1] + 9.4], dtype=float)
        # Use the fitted wide surface for the first observation.  Subsequent
        # observations follow exactly the model response predicted by the
        # previous iteration, which isolates the finite-loop geometry/budget.
        coefficients = np.asarray(
            wide["fit"]["selected_model"]["coefficients_feature_by_error"],
            dtype=float,
        )
        offset = xy - np.asarray(ANCHOR)
        error = np.array([1.0, offset[0], offset[1]]) @ coefficients
        previous = None
        cumulative = 0.0
        tiers = []
        final = None
        for iteration in range(1, 34):
            report = build_stage7b_iteration(
                samples(xy.tolist(), error.tolist(), start=iteration * 10),
                request(xy.tolist()),
                fine,
                wide,
                iteration_index=iteration,
                cumulative_path_mm=cumulative,
                previous_iteration=previous,
            )
            tiers.append(report["model_tier"])
            final = report
            if report["decision"] == "converged":
                break
            self.assertEqual(
                report["decision"],
                "move",
                msg=f"iteration={iteration}, failures={report['failure_reasons']}, "
                    f"error={report['error_norm_px']}, offset={report['current_offset_xy_mm']}",
            )
            xy += np.asarray(report["commanded_correction_xy_mm"], dtype=float)
            error = np.asarray(report["predicted_error_px"], dtype=float)
            cumulative = float(report["cumulative_path_after_mm"])
            previous = report
        self.assertIsNotNone(final)
        self.assertEqual(final["decision"], "converged")
        self.assertIn("coarse_task11", tiers)
        self.assertIn("fine_task9", tiers)
        self.assertLessEqual(cumulative, 50.0)


if __name__ == "__main__":
    unittest.main()
