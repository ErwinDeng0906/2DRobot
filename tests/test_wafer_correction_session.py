from __future__ import annotations

import copy
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.ui.action_worker import ActionWorker, normalize_action_task
from scara.vision.handeye_interaction import load_latest_suction_target
from scara.pipeline.kinematics import fk_wrist, solve_joints
from scara.vision.wafer_correction_session import (
    WAFER_CORRECTION_RUNTIME_REQUEST_KEY,
    WAFER_CORRECTION_TARGET_NAME,
    WaferCorrectionSession,
)
from tests.test_stage7a_action_framework import FakeController


def _initial_state() -> dict:
    suction = load_latest_suction_target(ROOT)
    return {
        "joints": [0.0, 90.0, suction.imaging_j3_mm, 0.0],
        "pose": [
            225.0,
            175.0,
            suction.imaging_j3_mm,
            0.0,
            0.0,
            suction.rz_mean_deg,
        ],
        "captured_monotonic_s": time.monotonic(),
    }


def _candidate(slot_key: str = "P00") -> dict:
    return {
        "slot_key": slot_key,
        "center_T_mm": [1.0, 2.0, -2.0],
        "distance_to_p00_mm": 2.2360679775,
        "center_source": "expanded_roi_full_contour_min_area_rect",
        "refinement": {"success": True, "reason": "ok"},
    }


class WaferCorrectionActionContractTests(unittest.TestCase):
    def test_action_contract_is_runtime_xy_only_and_normalizes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = WaferCorrectionSession(
                ROOT,
                Path(temporary) / "evidence",
                _initial_state(),
            )
            raw = session.action_task()
            normalized = normalize_action_task(raw)

        action_types = {step["type"] for step in normalized["actions"]}
        self.assertEqual(
            {"assert_joints", "wait", "runtime_move_joints"}, action_types
        )
        self.assertNotIn("move_xyzr", action_types)
        self.assertNotIn("set_do", action_types)
        runtime_steps = [
            step
            for step in normalized["actions"]
            if step["type"] == "runtime_move_joints"
        ]
        self.assertTrue(runtime_steps)
        self.assertTrue(
            all(
                step["request_key"]
                == WAFER_CORRECTION_RUNTIME_REQUEST_KEY
                and step["target_name"] == WAFER_CORRECTION_TARGET_NAME
                and step["required_j3_mm"] == session.required_j3_mm
                and step["max_xy_step_norm_mm"] == 10.0
                for step in runtime_steps
            )
        )

        wrong_target = copy.deepcopy(raw)
        wrong_target["actions"][2]["target_name"] = "P22"
        with self.assertRaisesRegex(ValueError, WAFER_CORRECTION_TARGET_NAME):
            normalize_action_task(wrong_target)

    def test_first_five_frames_freeze_mean_target_then_transform_to_world(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = WaferCorrectionSession(
                ROOT,
                Path(temporary) / "evidence",
                _initial_state(),
            )
            registration = {
                "status": "success",
                "transform_W_T": [
                    [1.0, 0.0, 0.0, 300.0],
                    [0.0, 1.0, 0.0, 50.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "origin_world_xy_mm": [300.0, 50.0],
                "yaw_world_from_tray_deg": 0.0,
            }
            request = {
                "request_id": "wafer-test-001",
                "request_key": WAFER_CORRECTION_RUNTIME_REQUEST_KEY,
                "target_name": WAFER_CORRECTION_TARGET_NAME,
                "calibration_sha256": session.calibration_hash,
                "requested_monotonic_s": time.monotonic() - 0.1,
            }
            samples = []
            for index, center_x in enumerate((0.0, 1.0, 1.0, 1.0, 2.0)):
                candidate = _candidate()
                candidate["center_T_mm"] = [center_x, 2.0, -2.0]
                samples.append(
                    {
                        "outside_wafer_candidates": [candidate],
                        "measurement_id": f"frame-{index}",
                    }
                )
            with patch.object(
                session, "_build_registration", return_value=registration
            ):
                response = session.build_response(request, samples)

            self.assertEqual("observe", response["decision"])
            self.assertEqual("approach", session.phase)
            self.assertEqual([1.0, 2.0, -2.0], session.locked_target["center_T_mm"])
            np.testing.assert_allclose(
                session.target_world_xy_mm, [301.0, 52.0], atol=1e-12
            )
            self.assertEqual(
                "expanded_roi_full_contour_min_area_rect",
                session.locked_target["center_source"],
            )
            self.assertEqual(
                "five_frame_arithmetic_mean",
                session.locked_target["aggregation_method"],
            )
            self.assertAlmostEqual(
                1.0,
                session.locked_target["stability"][
                    "maximum_center_residual_mm"
                ],
            )
            self.assertFalse(
                session.locked_target["stability"][
                    "residual_gate_enforced"
                ]
            )
            self.assertTrue(session.report_path.exists())
            self.assertTrue(session.registration_path.exists())

    def test_registration_that_requires_probe_aborts_without_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = WaferCorrectionSession(
                ROOT,
                Path(temporary) / "evidence",
                _initial_state(),
            )
            request = {
                "request_id": "wafer-test-002",
                "request_key": WAFER_CORRECTION_RUNTIME_REQUEST_KEY,
                "target_name": WAFER_CORRECTION_TARGET_NAME,
                "calibration_sha256": session.calibration_hash,
                "requested_monotonic_s": time.monotonic() - 0.1,
            }
            samples = [
                {"outside_wafer_candidates": [_candidate()]}
                for _index in range(5)
            ]
            registration = {
                "status": "requires_three_pose_probe",
                "transform_W_T": np.eye(4).tolist(),
            }
            with patch.object(
                session, "_build_registration", return_value=registration
            ):
                response = session.build_response(request, samples)
            self.assertEqual("abort", response["decision"])
            self.assertEqual("safety_rejected", session.status)
            self.assertEqual([], session.iterations)

    def test_continuity_registration_must_also_be_exact_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = WaferCorrectionSession(
                ROOT,
                Path(temporary) / "evidence",
                _initial_state(),
            )
            session.phase = "approach"
            session.registration = {
                "status": "success",
                "transform_W_T": np.eye(4).tolist(),
                "origin_world_xy_mm": [0.0, 0.0],
                "yaw_world_from_tray_deg": 0.0,
            }
            session.locked_target = {
                "center_T_mm": [1.0, 2.0, -2.0]
            }
            session.target_world_xy_mm = [1.0, 2.0]
            request = {
                "request_id": "wafer-test-continuity-probe",
                "request_key": WAFER_CORRECTION_RUNTIME_REQUEST_KEY,
                "target_name": WAFER_CORRECTION_TARGET_NAME,
                "calibration_sha256": session.calibration_hash,
                "requested_monotonic_s": time.monotonic() - 0.1,
            }
            samples = [
                {"outside_wafer_candidates": [_candidate()]}
                for _index in range(5)
            ]
            registration = {
                "status": "requires_three_pose_probe",
                "transform_W_T": np.eye(4).tolist(),
            }
            with patch.object(
                session, "_build_registration", return_value=registration
            ):
                response = session.build_response(request, samples)

            self.assertEqual("abort", response["decision"])
            self.assertIn("不是success", response["reason"])
            self.assertEqual([], session.iterations)

    def test_approach_reprojects_target_and_keeps_exact_current_j3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = WaferCorrectionSession(
                ROOT,
                Path(temporary) / "evidence",
                _initial_state(),
            )
            anchor = np.asarray(session.static_audit_anchor_xy, dtype=float)
            joints = solve_joints(
                float(anchor[0]),
                float(anchor[1]),
                session.required_j3_mm,
                rz_deg=session.required_rz_deg,
            )
            self.assertIsNotNone(joints)
            joints = list(joints)
            current_xy = fk_wrist(joints[0], joints[1])
            pose = [
                current_xy[0],
                current_xy[1],
                session.required_j3_mm,
                180.0,
                0.0,
                session.required_rz_deg,
            ]
            locked = {
                "center_T_mm": [1.0, 2.0, -2.0],
                "distance_to_p00_mm": 2.236,
                "source_slot_keys": ["P00"] * 5,
                "unique_source_slot_keys": ["P00"],
            }
            target_world = np.asarray(current_xy) + [5.0, 0.0]
            translation = target_world - np.asarray(locked["center_T_mm"][:2])
            registration = {
                "status": "success",
                "transform_W_T": [
                    [1.0, 0.0, 0.0, float(translation[0])],
                    [0.0, 1.0, 0.0, float(translation[1])],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "origin_world_xy_mm": translation.astype(float).tolist(),
                "yaw_world_from_tray_deg": 0.0,
            }
            session.phase = "approach"
            session.status = "target_locked"
            session.registration = registration
            session.locked_target = locked
            session.target_world_xy_mm = target_world.astype(float).tolist()
            current_registration = copy.deepcopy(registration)
            current_registration["origin_world_xy_mm"][0] += 0.5
            current_registration["transform_W_T"][0][3] += 0.5
            request = {
                "request_id": "wafer-test-approach",
                "request_key": WAFER_CORRECTION_RUNTIME_REQUEST_KEY,
                "target_name": WAFER_CORRECTION_TARGET_NAME,
                "calibration_sha256": session.calibration_hash,
                "requested_monotonic_s": time.monotonic() - 0.1,
            }
            samples = []
            for index in range(5):
                observed_candidate = _candidate()
                observed_candidate["center_T_mm"] = [2.0, 2.0, -2.0]
                samples.append(
                    {
                        "outside_wafer_candidates": [observed_candidate],
                        "current_joints": joints,
                        "current_pose": pose,
                        "measurement_id": f"approach-{index}",
                    }
                )
            with patch.object(
                session,
                "_build_registration",
                return_value=current_registration,
            ):
                response = session.build_response(request, samples)

            self.assertEqual("approve", response["decision"], response)
            proposal = response["proposal"]
            self.assertAlmostEqual(
                5.5,
                np.linalg.norm(proposal["commanded_correction_xy_mm"]),
                places=9,
            )
            np.testing.assert_allclose(
                session.target_world_xy_mm,
                target_world + [0.5, 0.0],
                atol=1e-12,
            )
            self.assertEqual(
                joints[2], proposal["planner"]["target_joints"][2]
            )
            self.assertTrue(proposal["planner"]["audit"]["passed"])
            centre_gate = proposal["safety_gates"][
                "locked_wafer_center_still_present"
            ]
            self.assertTrue(centre_gate["passed"])
            self.assertAlmostEqual(1.0, centre_gate["actual"])
            self.assertIn("diagnostic only", centre_gate["limit"])

    def test_continuity_rejects_any_frame_with_bad_j3_or_rz(self) -> None:
        for fault in ("j3", "pose_rz", "joint_rz"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary:
                session = WaferCorrectionSession(
                    ROOT,
                    Path(temporary) / "evidence",
                    _initial_state(),
                )
                anchor = np.asarray(session.static_audit_anchor_xy, dtype=float)
                joints = list(
                    solve_joints(
                        float(anchor[0]),
                        float(anchor[1]),
                        session.required_j3_mm,
                        rz_deg=session.required_rz_deg,
                    )
                )
                current_xy = fk_wrist(joints[0], joints[1])
                pose = [
                    current_xy[0],
                    current_xy[1],
                    session.required_j3_mm,
                    180.0,
                    0.0,
                    session.required_rz_deg,
                ]
                registration = {
                    "status": "success",
                    "transform_W_T": np.eye(4).tolist(),
                    "origin_world_xy_mm": [0.0, 0.0],
                    "yaw_world_from_tray_deg": 0.0,
                }
                session.phase = "approach"
                session.registration = copy.deepcopy(registration)
                session.locked_target = {
                    "center_T_mm": [1.0, 2.0, -2.0]
                }
                session.target_world_xy_mm = [1.0, 2.0]
                request = {
                    "request_id": f"wafer-test-{fault}",
                    "request_key": WAFER_CORRECTION_RUNTIME_REQUEST_KEY,
                    "target_name": WAFER_CORRECTION_TARGET_NAME,
                    "calibration_sha256": session.calibration_hash,
                    "requested_monotonic_s": time.monotonic() - 0.1,
                }
                samples = [
                    {
                        "outside_wafer_candidates": [_candidate()],
                        "current_joints": list(joints),
                        "current_pose": list(pose),
                        "measurement_id": f"{fault}-{index}",
                    }
                    for index in range(5)
                ]
                if fault == "j3":
                    samples[2]["current_joints"][2] += 0.201
                elif fault == "pose_rz":
                    samples[2]["current_pose"][5] += 0.301
                else:
                    samples[2]["current_joints"][3] += 0.301

                with patch.object(
                    session,
                    "_build_registration",
                    return_value=registration,
                ):
                    response = session.build_response(request, samples)

                self.assertEqual("abort", response["decision"])
                gates = response["evaluation"]["continuity_gates"]
                expected_gate = {
                    "j3": "all_frames_j3_at_required_height",
                    "pose_rz": "all_frames_pose_rz_matches_calibration",
                    "joint_rz": "all_frames_joint_rz_matches_calibration",
                }[fault]
                self.assertFalse(gates[expected_gate]["passed"])

    def test_worker_accepts_wafer_observe_then_complete_without_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = WaferCorrectionSession(
                ROOT,
                Path(temporary) / "unused-session-output",
                _initial_state(),
            )
            raw = session.action_task()
            runtime_step = next(
                step
                for step in raw["actions"]
                if step["type"] == "runtime_move_joints"
            )
            raw["actions"] = [runtime_step, copy.deepcopy(runtime_step)]
            worker = ActionWorker(
                FakeController(),
                normalize_action_task(raw),
                Path(temporary) / "worker-output",
            )
            response_count = 0

            def respond(request: dict) -> None:
                nonlocal response_count
                response_count += 1
                worker.respond_runtime_move_joints(
                    {
                        "request_id": request["request_id"],
                        "decision": (
                            "observe" if response_count == 1 else "complete"
                        ),
                        "calibration_sha256": session.calibration_hash,
                        "reason": (
                            "target locked" if response_count == 1 else "held"
                        ),
                    }
                )

            worker.runtime_move_joints_requested.connect(respond)
            result: list[bool] = []
            worker.run_finished.connect(
                lambda ok, _message, _folder: result.append(bool(ok))
            )
            worker.run()

            self.assertEqual([True], result)
            self.assertEqual([], worker._controller.goto_calls)
            manifest = json.loads(
                worker.manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                [row["status"] for row in manifest["runtime_moves"]],
                [
                    "observation_completed_no_motion",
                    "session_completed_no_motion",
                ],
            )

    def test_worker_rechecks_controller_j3_rz_and_safety_before_complete(self) -> None:
        for fault in ("j3", "pose_rz", "joint_rz", "alarm"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary:
                session = WaferCorrectionSession(
                    ROOT,
                    Path(temporary) / "unused-session-output",
                    _initial_state(),
                )
                raw = session.action_task()
                runtime_step = next(
                    step
                    for step in raw["actions"]
                    if step["type"] == "runtime_move_joints"
                )
                raw["actions"] = [runtime_step]
                controller = FakeController()
                if fault == "j3":
                    controller.joints[2] += 0.201
                    controller.pose[2] = controller.joints[2]
                elif fault == "pose_rz":
                    controller.pose[5] += 0.301
                elif fault == "joint_rz":
                    controller.joints[3] += 0.301
                else:
                    controller.warn = 9
                worker = ActionWorker(
                    controller,
                    normalize_action_task(raw),
                    Path(temporary) / "worker-output",
                )

                def respond(request: dict) -> None:
                    worker.respond_runtime_move_joints(
                        {
                            "request_id": request["request_id"],
                            "decision": "complete",
                            "calibration_sha256": session.calibration_hash,
                            "reason": "held",
                        }
                    )

                worker.runtime_move_joints_requested.connect(respond)
                result: list[tuple[bool, str]] = []
                worker.run_finished.connect(
                    lambda ok, message, _folder: result.append(
                        (bool(ok), str(message))
                    )
                )
                worker.run()

                self.assertFalse(result[0][0])
                self.assertIn("完成前控制器/J3/Rz复核失败", result[0][1])
                self.assertEqual([], controller.goto_calls)
                manifest = json.loads(
                    worker.manifest_path.read_text(encoding="utf-8")
                )
                self.assertEqual(
                    "rejected_no_motion",
                    manifest["runtime_moves"][0]["status"],
                )
                self.assertIn(
                    "completion_fresh_gates",
                    manifest["runtime_moves"][0],
                )


if __name__ == "__main__":
    unittest.main()
