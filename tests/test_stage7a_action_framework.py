"""Offline safety tests for the generic Stage-7A action handshake."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.pipeline.kinematics import fk_wrist, j4_for_rz, rz_of
from scara.pipeline.xy_correction_planner import plan_fixed_rz_xy_step
from scara.ui.action_worker import ActionWorker, normalize_action_task
from scara.ui.control_widget import ScaraControlWidget


P22_JOINTS = [30.6646, 84.7845, -27.0046, -4.6268]
P22_RZ = rz_of(P22_JOINTS[0], P22_JOINTS[1], P22_JOINTS[3])
P22_XY = fk_wrist(P22_JOINTS[0], P22_JOINTS[1])
CALIBRATION_SHA256 = "A" * 64


class FakeController:
    def __init__(self) -> None:
        self.joints = list(P22_JOINTS)
        self.pose = [P22_XY[0], P22_XY[1], -27.0046, 180.0, 0.0, P22_RZ]
        self.goto_calls: list[list[float]] = []
        self.emergency_stop_count = 0
        self._motion_sequence_lock = threading.Lock()
        self.connected = True
        self.enabled = True
        self.warn = 0
        self.estop = False
        self.soft_estop = False
        self.fault_after_move = False
        self.fail_on_goto_call: int | None = None

    def is_connected(self) -> bool:
        return self.connected

    def read_all_sync(self) -> dict:
        return {
            "joints": list(self.joints),
            "pose": list(self.pose),
            "enable": 1 if self.enabled else 0,
            "effectively_enabled": self.enabled,
            "warn": self.warn,
            "estop": self.estop,
            "soft_estop": self.soft_estop,
            "need_clear": bool(self.warn or self.estop or self.soft_estop),
            "mode": "T1",
        }

    def goto_joints_sync(
        self,
        _name: str,
        target: list[float],
        *,
        should_stop,
        tolerance: float,
    ) -> bool:
        if should_stop() or tolerance <= 0.0:
            return False
        self.goto_calls.append([float(value) for value in target])
        if self.fail_on_goto_call == len(self.goto_calls):
            return False
        self.joints = [float(value) for value in target]
        xy = fk_wrist(self.joints[0], self.joints[1])
        self.pose = [
            float(xy[0]),
            float(xy[1]),
            float(self.joints[2]),
            180.0,
            0.0,
            float(rz_of(self.joints[0], self.joints[1], self.joints[3])),
        ]
        if self.fault_after_move:
            self.warn = 7
        return True

    def emergency_stop(self) -> None:
        self.emergency_stop_count += 1


def runtime_step(**overrides) -> dict:
    step = {
        "type": "runtime_move_joints",
        "name": "Stage7A P22 supervised correction",
        "request_key": "stage7a_p22_single_step",
        "target_name": "P22",
        "calibration_sha256": CALIBRATION_SHA256,
        "anchor_robot_xy_mm": [P22_XY[0], P22_XY[1]],
        "local_extent_mm": 2.0,
        "domain_margin_mm": 0.20,
        "required_j3_mm": P22_JOINTS[2],
        "required_rz_deg": P22_RZ,
        "max_xy_step_norm_mm": 0.25,
        "max_xy_axis_mm": 0.25,
        "j3_tolerance_mm": 0.15,
        "rz_tolerance_deg": 0.20,
        "target_rz_tolerance_deg": 0.15,
        "max_sequential_transient_rz_deg": 0.30,
        "precompensate_rz": True,
        "max_state_drift_xy_mm": 0.05,
        "max_state_drift_joint": 0.05,
        "max_sequential_transient_xy_mm": 0.50,
        "move_tolerance": 0.01,
        "proposal_max_age_s": 60.0,
        "fk_pose_xy_tolerance_mm": 0.20,
    }
    step.update(overrides)
    return step


def task_for(step: dict) -> dict:
    return {
        "name": "stage7a framework test",
        "description": "offline only",
        "camera_model": {
            "offset_mm": 0.0,
            "angle_reference": "world_negative_y",
            "positive_rotation": "counter_clockwise_from_above",
        },
        "actions": [step],
    }


def proposal_for(command_xy_mm: list[float]) -> dict:
    return {
        "schema_version": 1,
        "stage": "7A_supervised_single_step",
        "target_name": "P22",
        "proposal_id": "TEST-PROPOSAL-001",
        "motion_authorized": True,
        "calculation": {
            "commanded_correction_xy_mm": [
                float(command_xy_mm[0]),
                float(command_xy_mm[1]),
            ]
        },
    }


class Stage7AActionNormalizationTests(unittest.TestCase):
    def test_runtime_step_normalizes_with_hard_safety_envelope(self) -> None:
        normalized = normalize_action_task(task_for(runtime_step()))
        step = normalized["actions"][0]
        self.assertEqual(step["type"], "runtime_move_joints")
        self.assertEqual(step["target_name"], "P22")
        self.assertEqual(step["calibration_sha256"], CALIBRATION_SHA256)
        self.assertEqual(step["max_xy_step_norm_mm"], 0.25)
        self.assertEqual(step["domain_margin_mm"], 0.20)
        self.assertEqual(step["rz_tolerance_deg"], 0.20)
        self.assertEqual(step["target_rz_tolerance_deg"], 0.15)
        self.assertEqual(step["max_sequential_transient_rz_deg"], 0.30)
        self.assertTrue(step["precompensate_rz"])

    def test_runtime_step_cannot_relax_engine_ceilings(self) -> None:
        unsafe = (
            {"max_xy_step_norm_mm": 0.251},
            {"local_extent_mm": 2.001},
            {"domain_margin_mm": 0.199},
            {"j3_tolerance_mm": 0.201},
            {"rz_tolerance_deg": 0.201},
            {"target_rz_tolerance_deg": 0.151},
            {"max_sequential_transient_rz_deg": 0.301},
            {"precompensate_rz": "true"},
            {"max_sequential_transient_xy_mm": 0.501},
            {"proposal_max_age_s": 60.001},
            {"target_name": "P21"},
        )
        for override in unsafe:
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    normalize_action_task(task_for(runtime_step(**override)))


class Stage7AWorkerHandshakeTests(unittest.TestCase):
    def _worker(self, output: Path, controller: FakeController) -> tuple[ActionWorker, dict]:
        normalized = normalize_action_task(task_for(runtime_step()))
        worker = ActionWorker(controller, normalized, output)
        worker._manifest = {
            "points": [],
            "photos": [],
            "videos": [],
            "runtime_moves": [],
        }
        output.mkdir(parents=True, exist_ok=True)
        worker._save_manifest()
        return worker, normalized["actions"][0]

    def test_decline_is_default_safe_and_does_not_move(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = FakeController()
            worker, step = self._worker(Path(temporary) / "run", controller)

            def decline(request: dict) -> None:
                worker.respond_runtime_move_joints(
                    {
                        "request_id": request["request_id"],
                        "decision": "decline",
                        "reason": "operator chose no motion",
                    }
                )

            worker.runtime_move_joints_requested.connect(decline)
            worker._runtime_move_joints(step)

            self.assertEqual(controller.goto_calls, [])
            manifest = json.loads(worker.manifest_path.read_text("utf-8"))
            self.assertEqual(
                manifest["runtime_moves"][0]["status"],
                "declined_no_motion",
            )

    def test_approved_target_moves_exactly_once_and_is_audited(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = FakeController()
            worker, step = self._worker(Path(temporary) / "run", controller)
            command = [0.12, -0.08]
            planned = plan_fixed_rz_xy_step(
                controller.joints,
                controller.pose,
                command,
                anchor_robot_xy_mm=step["anchor_robot_xy_mm"],
                local_extent_mm=step["local_extent_mm"],
                domain_margin_mm=step["domain_margin_mm"],
                required_j3_mm=step["required_j3_mm"],
                j3_tolerance_mm=step["j3_tolerance_mm"],
                required_rz_deg=step["required_rz_deg"],
                rz_tolerance_deg=step["rz_tolerance_deg"],
                max_xy_step_norm_mm=step["max_xy_step_norm_mm"],
                max_xy_axis_mm=step["max_xy_axis_mm"],
                max_sequential_transient_xy_mm=step[
                    "max_sequential_transient_xy_mm"
                ],
            )

            def approve(request: dict) -> None:
                worker.respond_runtime_move_joints(
                    {
                        "request_id": request["request_id"],
                        "decision": "approve",
                        "calibration_sha256": CALIBRATION_SHA256,
                        "proposal": proposal_for(command),
                        "target_joints": planned["target_joints"],
                    }
                )

            worker.runtime_move_joints_requested.connect(approve)
            worker._runtime_move_joints(step)

            self.assertEqual(len(controller.goto_calls), 1)
            manifest = json.loads(worker.manifest_path.read_text("utf-8"))
            record = manifest["runtime_moves"][0]
            self.assertEqual(record["status"], "motion_completed")
            self.assertTrue(record["fresh_kinematic_audit"]["passed"])
            self.assertTrue(record["actual_motion_audit"]["passed"])
            self.assertTrue(all(record["final_controller_gates"].values()))
            self.assertLess(record["proposal_target_command_mismatch_mm"], 0.002)

    def test_controller_fault_after_move_cannot_be_reported_completed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = FakeController()
            controller.fault_after_move = True
            worker, step = self._worker(Path(temporary) / "run", controller)
            command = [0.10, 0.0]
            planned = plan_fixed_rz_xy_step(
                controller.joints,
                controller.pose,
                command,
                anchor_robot_xy_mm=step["anchor_robot_xy_mm"],
                local_extent_mm=step["local_extent_mm"],
                domain_margin_mm=step["domain_margin_mm"],
                required_j3_mm=step["required_j3_mm"],
                j3_tolerance_mm=step["j3_tolerance_mm"],
                required_rz_deg=step["required_rz_deg"],
                rz_tolerance_deg=step["rz_tolerance_deg"],
                max_xy_step_norm_mm=step["max_xy_step_norm_mm"],
                max_xy_axis_mm=step["max_xy_axis_mm"],
                max_sequential_transient_xy_mm=step[
                    "max_sequential_transient_xy_mm"
                ],
            )

            def approve(request: dict) -> None:
                worker.respond_runtime_move_joints(
                    {
                        "request_id": request["request_id"],
                        "decision": "approve",
                        "calibration_sha256": CALIBRATION_SHA256,
                        "proposal": proposal_for(command),
                        "target_joints": planned["target_joints"],
                    }
                )

            worker.runtime_move_joints_requested.connect(approve)
            with self.assertRaisesRegex(RuntimeError, "到位后控制器安全门失败"):
                worker._runtime_move_joints(step)
            self.assertEqual(len(controller.goto_calls), 1)
            manifest = json.loads(worker.manifest_path.read_text("utf-8"))
            record = manifest["runtime_moves"][0]
            self.assertEqual(record["status"], "final_verification_failed")
            self.assertFalse(record["final_controller_gates"]["alarm_clear"])

    def test_off_angle_start_precompensates_j4_before_xy_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = FakeController()
            controller.joints = [30.8957, 84.3851, -27.0074, -4.2862]
            current_xy = fk_wrist(controller.joints[0], controller.joints[1])
            controller.pose = [
                float(current_xy[0]),
                float(current_xy[1]),
                controller.joints[2],
                180.0,
                0.0,
                rz_of(
                    controller.joints[0],
                    controller.joints[1],
                    controller.joints[3],
                ),
            ]
            worker, step = self._worker(Path(temporary) / "run", controller)
            command = [-0.07193199871756309, -0.23942804255244723]
            planned = plan_fixed_rz_xy_step(
                controller.joints,
                controller.pose,
                command,
                anchor_robot_xy_mm=step["anchor_robot_xy_mm"],
                local_extent_mm=step["local_extent_mm"],
                domain_margin_mm=step["domain_margin_mm"],
                required_j3_mm=step["required_j3_mm"],
                j3_tolerance_mm=step["j3_tolerance_mm"],
                required_rz_deg=step["required_rz_deg"],
                rz_tolerance_deg=step["rz_tolerance_deg"],
                target_rz_tolerance_deg=step["target_rz_tolerance_deg"],
                max_sequential_transient_rz_deg=step[
                    "max_sequential_transient_rz_deg"
                ],
                precompensate_rz=step["precompensate_rz"],
                max_xy_step_norm_mm=step["max_xy_step_norm_mm"],
                max_xy_axis_mm=step["max_xy_axis_mm"],
                max_sequential_transient_xy_mm=step[
                    "max_sequential_transient_xy_mm"
                ],
            )

            def approve(request: dict) -> None:
                worker.respond_runtime_move_joints(
                    {
                        "request_id": request["request_id"],
                        "decision": "approve",
                        "calibration_sha256": CALIBRATION_SHA256,
                        "proposal": proposal_for(command),
                        "target_joints": planned["target_joints"],
                    }
                )

            worker.runtime_move_joints_requested.connect(approve)
            worker._runtime_move_joints(step)

            self.assertEqual(len(controller.goto_calls), 2)
            precomp_target = controller.goto_calls[0]
            self.assertEqual(precomp_target[:3], [30.8957, 84.3851, -27.0074])
            self.assertAlmostEqual(
                precomp_target[3],
                j4_for_rz(30.8957, 84.3851, P22_RZ),
                places=9,
            )
            self.assertEqual(controller.goto_calls[1], planned["target_joints"])
            manifest = json.loads(worker.manifest_path.read_text("utf-8"))
            record = manifest["runtime_moves"][0]
            self.assertEqual(record["status"], "motion_completed")
            self.assertTrue(record["rz_precompensation"]["executed"])
            self.assertTrue(record["post_precompensation_kinematic_audit"]["passed"])

    def test_xy_failure_after_j4_precompensation_is_logged_as_partial_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = FakeController()
            controller.joints = [30.8957, 84.3851, -27.0074, -4.2862]
            current_xy = fk_wrist(controller.joints[0], controller.joints[1])
            controller.pose = [
                float(current_xy[0]),
                float(current_xy[1]),
                controller.joints[2],
                180.0,
                0.0,
                rz_of(
                    controller.joints[0],
                    controller.joints[1],
                    controller.joints[3],
                ),
            ]
            controller.fail_on_goto_call = 2
            worker, step = self._worker(Path(temporary) / "run", controller)
            command = [-0.07193199871756309, -0.23942804255244723]
            planned = plan_fixed_rz_xy_step(
                controller.joints,
                controller.pose,
                command,
                anchor_robot_xy_mm=step["anchor_robot_xy_mm"],
                local_extent_mm=step["local_extent_mm"],
                domain_margin_mm=step["domain_margin_mm"],
                required_j3_mm=step["required_j3_mm"],
                j3_tolerance_mm=step["j3_tolerance_mm"],
                required_rz_deg=step["required_rz_deg"],
                rz_tolerance_deg=step["rz_tolerance_deg"],
                target_rz_tolerance_deg=step["target_rz_tolerance_deg"],
                max_sequential_transient_rz_deg=step[
                    "max_sequential_transient_rz_deg"
                ],
                precompensate_rz=step["precompensate_rz"],
                max_xy_step_norm_mm=step["max_xy_step_norm_mm"],
                max_xy_axis_mm=step["max_xy_axis_mm"],
                max_sequential_transient_xy_mm=step[
                    "max_sequential_transient_xy_mm"
                ],
            )

            def approve(request: dict) -> None:
                worker.respond_runtime_move_joints(
                    {
                        "request_id": request["request_id"],
                        "decision": "approve",
                        "calibration_sha256": CALIBRATION_SHA256,
                        "proposal": proposal_for(command),
                        "target_joints": planned["target_joints"],
                    }
                )

            worker.runtime_move_joints_requested.connect(approve)
            with self.assertRaisesRegex(RuntimeError, "移动到"):
                worker._runtime_move_joints(step)

            self.assertEqual(len(controller.goto_calls), 2)
            manifest = json.loads(worker.manifest_path.read_text("utf-8"))
            record = manifest["runtime_moves"][0]
            self.assertEqual(record["status"], "motion_failed")
            self.assertTrue(record["physical_motion_started"])
            self.assertTrue(record["rz_precompensation"]["executed"])

    def test_response_target_must_match_the_displayed_correction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = FakeController()
            worker, step = self._worker(Path(temporary) / "run", controller)
            actual_command = [0.12, -0.08]
            planned = plan_fixed_rz_xy_step(
                controller.joints,
                controller.pose,
                actual_command,
                anchor_robot_xy_mm=step["anchor_robot_xy_mm"],
                local_extent_mm=step["local_extent_mm"],
                domain_margin_mm=step["domain_margin_mm"],
                required_j3_mm=step["required_j3_mm"],
                j3_tolerance_mm=step["j3_tolerance_mm"],
                required_rz_deg=step["required_rz_deg"],
                rz_tolerance_deg=step["rz_tolerance_deg"],
                max_xy_step_norm_mm=step["max_xy_step_norm_mm"],
                max_xy_axis_mm=step["max_xy_axis_mm"],
                max_sequential_transient_xy_mm=step[
                    "max_sequential_transient_xy_mm"
                ],
            )

            def mismatched(request: dict) -> None:
                worker.respond_runtime_move_joints(
                    {
                        "request_id": request["request_id"],
                        "decision": "approve",
                        "calibration_sha256": CALIBRATION_SHA256,
                        "proposal": proposal_for([-0.12, 0.08]),
                        "target_joints": planned["target_joints"],
                    }
                )

            worker.runtime_move_joints_requested.connect(mismatched)
            with self.assertRaisesRegex(RuntimeError, "不一致"):
                worker._runtime_move_joints(step)
            self.assertEqual(controller.goto_calls, [])

    def test_fresh_read_rejects_state_changed_during_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = FakeController()
            worker, step = self._worker(Path(temporary) / "run", controller)
            command = [0.10, 0.0]
            planned = plan_fixed_rz_xy_step(
                controller.joints,
                controller.pose,
                command,
                anchor_robot_xy_mm=step["anchor_robot_xy_mm"],
                local_extent_mm=step["local_extent_mm"],
                domain_margin_mm=step["domain_margin_mm"],
                required_j3_mm=step["required_j3_mm"],
                j3_tolerance_mm=step["j3_tolerance_mm"],
                required_rz_deg=step["required_rz_deg"],
                rz_tolerance_deg=step["rz_tolerance_deg"],
                max_xy_step_norm_mm=step["max_xy_step_norm_mm"],
                max_xy_axis_mm=step["max_xy_axis_mm"],
                max_sequential_transient_xy_mm=step[
                    "max_sequential_transient_xy_mm"
                ],
            )

            def approve_after_manual_change(request: dict) -> None:
                controller.pose[0] += 0.06
                worker.respond_runtime_move_joints(
                    {
                        "request_id": request["request_id"],
                        "decision": "approve",
                        "calibration_sha256": CALIBRATION_SHA256,
                        "proposal": proposal_for(command),
                        "target_joints": planned["target_joints"],
                    }
                )

            worker.runtime_move_joints_requested.connect(
                approve_after_manual_change
            )
            with self.assertRaisesRegex(RuntimeError, "旧提案已作废"):
                worker._runtime_move_joints(step)
            self.assertEqual(controller.goto_calls, [])

    def test_stop_interrupts_waiting_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = FakeController()
            worker, step = self._worker(Path(temporary) / "run", controller)
            failures: list[BaseException] = []

            def run_wait() -> None:
                try:
                    worker._runtime_move_joints(step)
                except BaseException as exc:  # noqa: BLE001 - test capture
                    failures.append(exc)

            thread = threading.Thread(target=run_wait)
            thread.start()
            deadline = time.monotonic() + 2.0
            while (
                worker._runtime_move_pending_request_id is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            worker.request_stop()
            thread.join(timeout=2.0)

            self.assertFalse(thread.is_alive())
            self.assertEqual(len(failures), 1)
            self.assertIn("取消", str(failures[0]))
            self.assertEqual(controller.goto_calls, [])
            self.assertEqual(controller.emergency_stop_count, 1)

    def test_record_point_persists_controller_safety_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = FakeController()
            worker, _step = self._worker(Path(temporary) / "run", controller)
            worker._record_point("before-01")
            point = worker._manifest["points"][0]
            safety = point["controller_safety"]
            self.assertTrue(safety["connected"])
            self.assertTrue(safety["effectively_enabled"])
            self.assertFalse(safety["estop"])
            self.assertFalse(safety["soft_estop"])
            self.assertEqual(safety["warn"], 0)
            self.assertFalse(safety["need_clear"])
            self.assertEqual(safety["mode"], "T1")
            self.assertTrue(math.isfinite(safety["captured_monotonic_s"]))


class Stage7AControlWidgetRoutingTests(unittest.TestCase):
    class _Worker:
        def __init__(self) -> None:
            self.responses: list[dict] = []

        @staticmethod
        def isRunning() -> bool:  # noqa: N802 - Qt compatibility
            return True

        def respond_runtime_move_joints(self, response: dict) -> None:
            self.responses.append(response)

    def test_ui_routes_runtime_response_without_calling_controller(self) -> None:
        worker = self._Worker()
        runtime = SimpleNamespace(
            on_runtime_move_joints_requested=lambda request: {
                "request_id": request["request_id"],
                "decision": "decline",
            }
        )
        owner = SimpleNamespace(
            _action_worker=worker,
            _action_runtime=runtime,
            _append=lambda *_args: None,
        )
        ScaraControlWidget._on_action_runtime_move_joints(
            owner,
            {"request_id": "request-1"},
        )
        self.assertEqual(
            worker.responses,
            [{"request_id": "request-1", "decision": "decline"}],
        )

    def test_ui_callback_failure_returns_abort_and_releases_worker(self) -> None:
        worker = self._Worker()

        def fail(_request: dict) -> dict:
            raise RuntimeError("synthetic popup failure")

        messages: list[str] = []
        owner = SimpleNamespace(
            _action_worker=worker,
            _action_runtime=SimpleNamespace(
                on_runtime_move_joints_requested=fail
            ),
            _append=lambda _title, message, _color: messages.append(message),
        )
        ScaraControlWidget._on_action_runtime_move_joints(
            owner,
            {"request_id": "request-2"},
        )
        self.assertEqual(worker.responses[0]["decision"], "abort")
        self.assertEqual(worker.responses[0]["request_id"], "request-2")
        self.assertIn("synthetic popup failure", messages[0])


if __name__ == "__main__":
    unittest.main()
