from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scara.pipeline.xy_correction_planner import plan_fixed_rz_xy_step
from scara.ui.action_worker import ActionWorker, normalize_action_task

from tests.test_stage7a_action_framework import (
    FakeController,
    P22_JOINTS,
    P22_RZ,
    P22_XY,
)


WIDE_HASH = "B" * 64
FINE_HASH = "C" * 64


def stage7b_step(**overrides):
    step = {
        "type": "runtime_move_joints",
        "name": "Stage7B finite iteration",
        "request_key": "stage7b_p22_finite_loop",
        "target_name": "P22",
        "calibration_sha256": WIDE_HASH,
        "fine_calibration_sha256": FINE_HASH,
        "anchor_robot_xy_mm": list(P22_XY),
        "local_extent_mm": 10.0,
        "domain_margin_mm": 0.5,
        "required_j3_mm": P22_JOINTS[2],
        "required_rz_deg": P22_RZ,
        "max_xy_step_norm_mm": 0.75,
        "max_xy_axis_mm": 0.75,
        "j3_tolerance_mm": 0.15,
        "rz_tolerance_deg": 0.20,
        "target_rz_tolerance_deg": 0.15,
        "max_sequential_transient_rz_deg": 0.30,
        "precompensate_rz": True,
        "max_state_drift_xy_mm": 0.05,
        "max_state_drift_joint": 0.05,
        "max_sequential_transient_xy_mm": 1.5,
        "move_tolerance": 0.01,
        "proposal_max_age_s": 8.0,
        "fk_pose_xy_tolerance_mm": 0.20,
    }
    step.update(overrides)
    return step


def task(actions):
    return {
        "api_version": 1,
        "name": "Stage7B offline",
        "description": "offline",
        "camera_model": {
            "offset_mm": 0,
            "angle_reference": "world_negative_y",
            "positive_rotation": "counter_clockwise_from_above",
        },
        "actions": list(actions),
    }


class Stage7BActionIntegrationTests(unittest.TestCase):
    def test_normalization_has_separate_wide_ceiling_and_preserves_stage7a(self):
        normalized = normalize_action_task(task([stage7b_step()]))
        step = normalized["actions"][0]
        self.assertEqual(step["local_extent_mm"], 10.0)
        self.assertEqual(step["max_xy_step_norm_mm"], 0.75)
        self.assertEqual(step["fine_calibration_sha256"], FINE_HASH)
        self.assertFalse(step["enforce_sequential_intermediate_domain"])
        stage7a = normalize_action_task(
            task(
                [
                    stage7b_step(
                        request_key="stage7a_p22_single_step",
                        fine_calibration_sha256="",
                        local_extent_mm=2.0,
                        domain_margin_mm=0.2,
                        max_xy_step_norm_mm=0.25,
                        max_xy_axis_mm=0.25,
                        max_sequential_transient_xy_mm=0.5,
                    )
                ]
            )
        )["actions"][0]
        self.assertTrue(stage7a["enforce_sequential_intermediate_domain"])
        for override in (
            {"local_extent_mm": 10.001},
            {"max_xy_step_norm_mm": 0.751},
            {"max_sequential_transient_xy_mm": 1.501},
            {"fine_calibration_sha256": "bad"},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                normalize_action_task(task([stage7b_step(**override)]))

    def test_complete_decision_stops_remaining_iterations_without_motion(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = FakeController()
            normalized = normalize_action_task(task([stage7b_step(), stage7b_step()]))
            worker = ActionWorker(controller, normalized, Path(temporary) / "run")

            def complete(request):
                worker.respond_runtime_move_joints(
                    {
                        "request_id": request["request_id"],
                        "decision": "complete",
                        "reason": "converged",
                        "calibration_sha256": WIDE_HASH,
                        "fine_calibration_sha256": FINE_HASH,
                    }
                )

            worker.runtime_move_joints_requested.connect(complete)
            result = []
            worker.run_finished.connect(lambda ok, message, folder: result.append((ok, message, folder)))
            worker.run()
            self.assertTrue(result[0][0])
            self.assertEqual(controller.goto_calls, [])
            manifest = json.loads(worker.manifest_path.read_text("utf-8"))
            self.assertEqual(len(manifest["runtime_moves"]), 1)
            self.assertEqual(manifest["runtime_moves"][0]["status"], "session_completed_no_motion")

    def test_wide_planning_reserve_survives_four_decimal_joint_readback(self):
        normalized_step = normalize_action_task(task([stage7b_step()]))[
            "actions"
        ][0]
        direction = np.asarray([-0.5488990380, -0.5110869261], dtype=float)
        command = (direction / np.linalg.norm(direction) * 0.74).tolist()
        pose = [P22_XY[0], P22_XY[1], P22_JOINTS[2], 180.0, 0.0, P22_RZ]
        planned = plan_fixed_rz_xy_step(
            P22_JOINTS,
            pose,
            command,
            anchor_robot_xy_mm=normalized_step["anchor_robot_xy_mm"],
            local_extent_mm=normalized_step["local_extent_mm"],
            domain_margin_mm=normalized_step["domain_margin_mm"],
            required_j3_mm=normalized_step["required_j3_mm"],
            j3_tolerance_mm=normalized_step["j3_tolerance_mm"],
            required_rz_deg=normalized_step["required_rz_deg"],
            rz_tolerance_deg=normalized_step["rz_tolerance_deg"],
            target_rz_tolerance_deg=normalized_step["target_rz_tolerance_deg"],
            max_xy_step_norm_mm=0.74,
            max_xy_axis_mm=0.74,
            max_sequential_transient_xy_mm=normalized_step[
                "max_sequential_transient_xy_mm"
            ],
            max_sequential_transient_rz_deg=normalized_step[
                "max_sequential_transient_rz_deg"
            ],
            precompensate_rz=True,
            enforce_sequential_intermediate_domain=False,
        )
        quantized_target = [round(value, 4) for value in planned["target_joints"]]
        actual_audit = ActionWorker._audit_runtime_target(
            {"joints": list(P22_JOINTS), "pose": pose},
            quantized_target,
            normalized_step,
        )
        self.assertTrue(actual_audit["passed"])
        self.assertLess(actual_audit["step_norm_mm"], 0.75)

    def test_fine_planning_reserve_survives_four_decimal_joint_readback(self):
        normalized_step = normalize_action_task(task([stage7b_step()]))[
            "actions"
        ][0]
        fine_step = dict(normalized_step)
        fine_step.update(
            local_extent_mm=2.0,
            domain_margin_mm=0.2,
            max_xy_step_norm_mm=0.25,
            max_xy_axis_mm=0.25,
            max_sequential_transient_xy_mm=0.5,
        )
        direction = np.asarray([-0.18, -0.16], dtype=float)
        command = (direction / np.linalg.norm(direction) * 0.24).tolist()
        pose = [P22_XY[0], P22_XY[1], P22_JOINTS[2], 180.0, 0.0, P22_RZ]
        planned = plan_fixed_rz_xy_step(
            P22_JOINTS,
            pose,
            command,
            anchor_robot_xy_mm=fine_step["anchor_robot_xy_mm"],
            local_extent_mm=fine_step["local_extent_mm"],
            domain_margin_mm=fine_step["domain_margin_mm"],
            required_j3_mm=fine_step["required_j3_mm"],
            j3_tolerance_mm=fine_step["j3_tolerance_mm"],
            required_rz_deg=fine_step["required_rz_deg"],
            rz_tolerance_deg=fine_step["rz_tolerance_deg"],
            target_rz_tolerance_deg=fine_step["target_rz_tolerance_deg"],
            max_xy_step_norm_mm=0.24,
            max_xy_axis_mm=0.24,
            max_sequential_transient_xy_mm=fine_step[
                "max_sequential_transient_xy_mm"
            ],
            max_sequential_transient_rz_deg=fine_step[
                "max_sequential_transient_rz_deg"
            ],
            precompensate_rz=True,
            enforce_sequential_intermediate_domain=False,
        )
        quantized_target = [round(value, 4) for value in planned["target_joints"]]
        actual_audit = ActionWorker._audit_runtime_target(
            {"joints": list(P22_JOINTS), "pose": pose},
            quantized_target,
            fine_step,
        )
        self.assertTrue(actual_audit["passed"])
        self.assertLess(actual_audit["step_norm_mm"], 0.25)

    def test_stage7b_failure_message_uses_single_point_loop_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = FakeController()
            normalized = normalize_action_task(task([stage7b_step()]))
            step = normalized["actions"][0]
            output = Path(temporary) / "run"
            output.mkdir()
            worker = ActionWorker(controller, normalized, output)
            worker._manifest = {
                "points": [], "photos": [], "videos": [], "runtime_moves": []
            }
            worker._save_manifest()

            def approve_with_mismatched_displayed_command(request):
                worker.respond_runtime_move_joints(
                    {
                        "request_id": request["request_id"],
                        "decision": "approve",
                        "calibration_sha256": WIDE_HASH,
                        "proposal": {
                            "target_name": "P22",
                            "proposal_id": "bad-command-label-test",
                            "motion_authorized": True,
                            "model_tier": "coarse_task11",
                            "wide_calibration_sha256": WIDE_HASH,
                            "fine_calibration_sha256": FINE_HASH,
                            "calculation": {
                                "commanded_correction_xy_mm": [0.1, 0.0]
                            },
                        },
                        "target_joints": list(P22_JOINTS),
                    }
                )

            worker.runtime_move_joints_requested.connect(
                approve_with_mismatched_displayed_command
            )
            with self.assertRaises(RuntimeError) as caught:
                worker._runtime_move_joints(step)
            self.assertIn("单点有限闭环", str(caught.exception))
            self.assertNotIn("Stage7A", str(caught.exception))

    def test_fine_tier_is_reaudited_with_task9_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = FakeController()
            normalized = normalize_action_task(task([stage7b_step()]))
            step = normalized["actions"][0]
            output = Path(temporary) / "run"
            output.mkdir()
            worker = ActionWorker(controller, normalized, output)
            worker._manifest = {"points": [], "photos": [], "videos": [], "runtime_moves": []}
            worker._save_manifest()
            command = [0.20, -0.10]
            planned = plan_fixed_rz_xy_step(
                controller.joints,
                controller.pose,
                command,
                anchor_robot_xy_mm=step["anchor_robot_xy_mm"],
                local_extent_mm=2.0,
                domain_margin_mm=0.2,
                required_j3_mm=step["required_j3_mm"],
                j3_tolerance_mm=step["j3_tolerance_mm"],
                required_rz_deg=step["required_rz_deg"],
                rz_tolerance_deg=step["rz_tolerance_deg"],
                target_rz_tolerance_deg=step["target_rz_tolerance_deg"],
                max_xy_step_norm_mm=0.25,
                max_xy_axis_mm=0.25,
                max_sequential_transient_xy_mm=0.5,
                max_sequential_transient_rz_deg=0.3,
                precompensate_rz=True,
            )

            def approve(request):
                worker.respond_runtime_move_joints(
                    {
                        "request_id": request["request_id"],
                        "decision": "approve",
                        "calibration_sha256": WIDE_HASH,
                        "proposal": {
                            "target_name": "P22",
                            "proposal_id": "fine-1",
                            "motion_authorized": True,
                            "model_tier": "fine_task9",
                            "wide_calibration_sha256": WIDE_HASH,
                            "fine_calibration_sha256": FINE_HASH,
                            "calculation": {"commanded_correction_xy_mm": command},
                        },
                        "target_joints": planned["target_joints"],
                    }
                )

            worker.runtime_move_joints_requested.connect(approve)
            worker._runtime_move_joints(step)
            self.assertEqual(len(controller.goto_calls), 1)
            manifest = json.loads(worker.manifest_path.read_text("utf-8"))
            self.assertEqual(manifest["runtime_moves"][0]["status"], "motion_completed")


if __name__ == "__main__":
    unittest.main()
