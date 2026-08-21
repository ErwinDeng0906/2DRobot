"""Offline integration tests for Task10 and the Stage-7A GUI runtime.

No test in this module opens a camera, connects a controller, or invokes a
hardware command.  Saved photos, Stage3 output, and the operator decision are
all synthetic.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from scara.pipeline.kinematics import fk_wrist, rz_of  # noqa: E402
from scara.ui.action_worker import ActionWorker, normalize_action_task  # noqa: E402
from scara.ui.stage7a_dialog import Stage7AOperatorDialog  # noqa: E402
from scara.vision import stage7a_runtime as runtime_module  # noqa: E402
from scara.vision.xy_image_jacobian import (  # noqa: E402
    REQUIRED_XY_JACOBIAN_QUALITY_GATES,
)


P22_JOINTS = [30.6646, 84.7845, -27.0046, -4.6268]
P22_XY = fk_wrist(P22_JOINTS[0], P22_JOINTS[1])
P22_RZ = rz_of(P22_JOINTS[0], P22_JOINTS[1], P22_JOINTS[3])
CALIBRATION_SHA256 = "A" * 64
IMAGE_SIZE = (8, 6)


def _jacobian_payload() -> dict:
    matrix = np.asarray([[4.0, 0.0], [0.0, -4.0]], dtype=np.float64)
    return {
        "schema_version": 2,
        "status": "success",
        "anchor_target_name": "P22",
        "valid_target_names": ["P22"],
        "locked_inputs": {"synthetic": True},
        "coordinate_definition": {
            "anchor_robot_xy_mm": list(P22_XY),
            "offset_extent_mm": 2.0,
        },
        "fit": {
            "status": "success",
            "j_error_px_per_command_mm": matrix.tolist(),
            "j_command_mm_per_error_px": np.linalg.inv(matrix).tolist(),
            "quality_gates": {
                name: {"passed": True}
                for name in REQUIRED_XY_JACOBIAN_QUALITY_GATES
            },
        },
    }


def _contract() -> dict:
    return {
        "stage5_path": "src/scara/calib/camera1_xy_image_jacobian.json",
        "stage5_sha256": CALIBRATION_SHA256,
        "anchor_robot_xy_mm": list(P22_XY),
        "local_extent_mm": 2.0,
        "required_j3_mm": P22_JOINTS[2],
        "required_rz_deg": P22_RZ,
    }


def _controller_request(*, request_key: str = runtime_module.REQUEST_KEY) -> dict:
    return {
        "schema_version": 1,
        "request_id": "runtime-move-test-001",
        "request_key": request_key,
        "target_name": "P22",
        "requested_at": "2026-08-14T22:00:00+08:00",
        "requested_monotonic_s": 100.0,
        "calibration_sha256": CALIBRATION_SHA256,
        "limits": {
            "max_state_drift_xy_mm": 0.05,
            "max_state_drift_joint": 0.05,
        },
        "controller_state": {
            "joints": list(P22_JOINTS),
            "pose": [P22_XY[0], P22_XY[1], P22_JOINTS[2], 180.0, 0.0, P22_RZ],
            "controller_connected": True,
            "controller_enabled": True,
            "alarm_clear": True,
            "estop_clear": True,
            "soft_estop_clear": True,
            "controller_idle": True,
        },
        "external_safety_gates": {
            "controller_connected": True,
            "controller_enabled": True,
            "alarm_clear": True,
            "estop_clear": True,
            "soft_estop_clear": True,
            "controller_idle": True,
            # The runtime must replace these two untrusted placeholders.
            "camera_fresh": False,
            "operator_consent": False,
        },
    }


class _FakeDialog:
    def __init__(self, approved: bool) -> None:
        self.approved = bool(approved)
        self.frames: list[dict] = []
        self.proposals: list[tuple[dict, dict | None]] = []
        self.final_text = ""

    def show(self) -> None:
        return None

    def add_frame(self, record: dict, _image) -> None:
        self.frames.append(dict(record))

    def request_decision(self, proposal: dict, plan: dict | None) -> bool:
        self.proposals.append((proposal, plan))
        return self.approved

    def set_final_text(self, text: str) -> None:
        self.final_text = str(text)


class _FakeTracker:
    def __init__(self, _estimator) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def update(self, _image):
        raw = SimpleNamespace(
            visible_marker_ids=[1, 2, 3, 4],
            used_marker_ids=[1, 2, 3, 4],
            ransac_inlier_corner_count=16,
        )
        return SimpleNamespace(
            raw=raw,
            translation_jump_mm=0.01,
            rotation_jump_deg=0.01,
        )


class _FakeController:
    """In-memory ActionWorker target; it has no transport or hardware API."""

    def __init__(self) -> None:
        self.joints = list(P22_JOINTS)
        self.pose = [P22_XY[0], P22_XY[1], P22_JOINTS[2], 180.0, 0.0, P22_RZ]
        self.goto_calls: list[list[float]] = []
        self._motion_sequence_lock = threading.Lock()

    @staticmethod
    def is_connected() -> bool:
        return True

    def read_all_sync(self) -> dict:
        return {
            "joints": list(self.joints),
            "pose": list(self.pose),
            "enable": 1,
            "effectively_enabled": True,
            "warn": 0,
            "estop": False,
            "soft_estop": False,
            "need_clear": False,
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
        self.joints = [float(value) for value in target]
        xy = fk_wrist(self.joints[0], self.joints[1])
        self.pose = [
            xy[0],
            xy[1],
            self.joints[2],
            180.0,
            0.0,
            rz_of(self.joints[0], self.joints[1], self.joints[3]),
        ]
        return True

    @staticmethod
    def emergency_stop() -> None:
        return None


def _fake_handeye_evaluation(image, tracked, *_args, **_kwargs):
    return SimpleNamespace(
        annotated_bgr=image.copy(),
        accepted=True,
        reason="ok",
        visible_marker_count=4,
        used_marker_count=4,
        reprojection_rms_px=0.30,
        slot_pixel_px=[5.0, 2.0],
        suction_target_pixel_px=[3.0, 3.0],
        image_error_px=[2.0, -1.0],
        image_error_norm_px=float(np.hypot(2.0, -1.0)),
        correction_xy_mm=[-0.5, -0.25],
        jacobian_domain_passed=True,
        jacobian_domain_note="synthetic pass",
    )


def _point(sequence: int, phase: str, index: int) -> dict:
    return {
        "sequence": sequence,
        "name": f"TASK10|target=P22|phase={phase}|frame={index:02d}/05",
        "recorded_at": f"2026-08-14T22:00:{sequence:02d}.000+08:00",
        "joints": {
            "J1_deg": P22_JOINTS[0],
            "J2_deg": P22_JOINTS[1],
            "J3_mm": P22_JOINTS[2],
            "J4_deg": P22_JOINTS[3],
        },
        "mechanical_center": {
            "x_mm": P22_XY[0],
            "y_mm": P22_XY[1],
            "z_mm": P22_JOINTS[2],
            "Rx_deg": 180.0,
            "Ry_deg": 0.0,
            "Rz_deg": P22_RZ,
        },
        "controller_safety": {
            "connected": True,
            "effectively_enabled": True,
            "warn": 0,
            "estop": False,
            "soft_estop": False,
            "need_clear": False,
            "mode": "T1",
        },
    }


def _write_manifest(run_dir: Path, phases: tuple[str, ...]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    points: list[dict] = []
    photos: list[dict] = []
    sequence = 0
    for phase in phases:
        for index in range(1, 6):
            sequence += 1
            filename = f"1_{sequence:03d}.jpg"
            points.append(_point(sequence, phase, index))
            photos.append(
                {
                    "filename": filename,
                    "source": 1,
                    "point_sequence": sequence,
                    "captured_at": f"2026-08-14T22:00:{sequence:02d}.050+08:00",
                }
            )
            image = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)
            if not cv2.imwrite(str(run_dir / filename), image):
                raise RuntimeError("synthetic photo write failed")
    payload = {
        "points": points,
        "photos": photos,
        "videos": [],
        "runtime_moves": [
            {
                "request_key": runtime_module.REQUEST_KEY,
                "status": "declined_no_motion",
            }
        ],
    }
    (run_dir / "points.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@contextmanager
def _runtime_fixture(run_dir: Path, *, approved: bool):
    dialog = _FakeDialog(approved)
    contract = _contract()
    suction = SimpleNamespace(resolution=IMAGE_SIZE)
    with (
        patch.object(runtime_module, "load_stage7a_motion_contract", return_value=contract),
        patch.object(runtime_module, "load_latest_suction_target", return_value=suction),
        patch.object(runtime_module, "load_local_xy_jacobian", return_value=_jacobian_payload()),
        patch.object(runtime_module, "load_camera_intrinsics", return_value=object()),
        patch.object(runtime_module, "load_tray_board_geometry", return_value=object()),
        patch.object(runtime_module, "TrayBoardPoseEstimator", return_value=object()),
        patch.object(runtime_module, "TrayPoseTracker", _FakeTracker),
        patch.object(runtime_module, "Stage7AOperatorDialog", return_value=dialog),
        patch.object(runtime_module, "evaluate_handeye_frame", side_effect=_fake_handeye_evaluation),
    ):
        runtime = runtime_module.Stage7ASingleStepRuntime(
            run_dir,
            PROJECT_ROOT,
            parent=None,
        )
        yield runtime, dialog


def _load_task10_module():
    path = PROJECT_ROOT / "Tasks" / "task10.py"
    spec = importlib.util.spec_from_file_location("task10_stage7a_contract_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load task10.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Task10ContractTests(unittest.TestCase):
    def test_task_contains_only_bounded_basic_actions_and_after_burst(self) -> None:
        module = _load_task10_module()
        with patch.object(
            module, "load_stage7a_motion_contract", return_value=_contract()
        ):
            task = module.build_action()

        normalized = normalize_action_task(task)
        actions = normalized["actions"]
        action_types = [action["type"] for action in actions]
        self.assertLessEqual(
            set(action_types),
            {"wait", "record_point", "capture", "runtime_move_joints"},
        )
        self.assertNotIn("assert_joints", action_types)
        self.assertNotIn("move_xyzr", action_types)
        self.assertNotIn("set_do", action_types)
        self.assertNotIn("move_joints", action_types)
        self.assertEqual(action_types.count("runtime_move_joints"), 1)
        self.assertEqual(action_types.count("record_point"), 10)
        self.assertEqual(action_types.count("capture"), 10)
        self.assertTrue(
            all(action["source"] == 1 for action in actions if action["type"] == "capture")
        )
        move_index = action_types.index("runtime_move_joints")
        self.assertEqual(actions[move_index]["move_tolerance"], 0.01)
        self.assertEqual(actions[move_index]["rz_tolerance_deg"], 0.20)
        self.assertEqual(actions[move_index]["target_rz_tolerance_deg"], 0.15)
        self.assertEqual(
            actions[move_index]["max_sequential_transient_rz_deg"], 0.30
        )
        self.assertTrue(actions[move_index]["precompensate_rz"])
        after_records = [
            action
            for action in actions[move_index + 1 :]
            if action["type"] == "record_point"
        ]
        self.assertEqual(len(after_records), 5)
        self.assertTrue(all("phase=after" in row["name"] for row in after_records))
        self.assertEqual(actions[move_index]["request_key"], runtime_module.REQUEST_KEY)


class Stage7ARuntimeTests(unittest.TestCase):
    def test_decline_records_before_and_after_evidence_without_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            _write_manifest(run_dir, ("before",))
            with _runtime_fixture(run_dir, approved=False) as (runtime, dialog):
                response = runtime.on_runtime_move_joints_requested(
                    _controller_request()
                )
                self.assertEqual(response["decision"], "decline")
                self.assertEqual(response["request_id"], "runtime-move-test-001")
                self.assertEqual(len(dialog.frames), 5)

                # This models ActionWorker continuing the static Task10 action
                # list after a decline and saving all five audit-only frames.
                _write_manifest(run_dir, ("before", "after"))
                runtime.on_task_finished(True, "done", str(run_dir))

            report = json.loads(
                (run_dir / runtime_module.RESULT_FILENAME).read_text("utf-8")
            )
            manifest = json.loads((run_dir / "points.json").read_text("utf-8"))
            self.assertEqual(report["status"], "operator_declined")
            self.assertFalse(report["action_result"]["motion_executed"])
            self.assertEqual(report["operator_decision"], "operator_declined")
            self.assertEqual(len(report["frame_records"]), 10)
            self.assertEqual(len(list((run_dir / "annotated_stage7a").glob("*.jpg"))), 10)
            self.assertEqual(manifest["stage7a"]["status"], "operator_declined")
            self.assertTrue(
                all("stage7a_evaluation" in point for point in manifest["points"])
            )
            for record in report["frame_records"]:
                self.assertTrue(
                    {
                        "point_sequence",
                        "point_name",
                        "filename",
                        "annotated_filename",
                        "robot_joints",
                        "robot_pose",
                        "robot_state_age_s",
                        "controller_safety",
                        "image_error_px",
                        "reprojection_rms_px",
                        "accepted",
                    }
                    <= set(record)
                )

    def test_valid_operator_consent_returns_worker_compatible_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            _write_manifest(run_dir, ("before",))
            with _runtime_fixture(run_dir, approved=True) as (runtime, dialog):
                response = runtime.on_runtime_move_joints_requested(
                    _controller_request()
                )

            self.assertEqual(response["decision"], "approve")
            self.assertEqual(response["request_id"], "runtime-move-test-001")
            self.assertEqual(response["calibration_sha256"], CALIBRATION_SHA256)
            self.assertEqual(len(response["target_joints"]), 4)
            self.assertTrue(response["proposal"]["motion_authorized"])
            self.assertTrue(response["planner"]["audit"]["passed"])
            self.assertEqual(
                response["proposal_id"], response["proposal"]["proposal_id"]
            )
            gates = response["proposal"]["safety_gates"]
            self.assertTrue(gates["measurement_matches_request_state"]["passed"])
            self.assertEqual(len(dialog.frames), 5)
            self.assertEqual(len(dialog.proposals), 1)
            incremental = json.loads(
                (run_dir / runtime_module.RESULT_FILENAME).read_text("utf-8")
            )
            self.assertEqual(
                incremental["status"], "motion_approved_pending_worker_preflight"
            )
            self.assertEqual(
                incremental["proposal"]["calculation"]["commanded_correction_xy_mm"],
                response["proposal"]["calculation"]["commanded_correction_xy_mm"],
            )

    def test_runtime_approval_is_accepted_by_actual_action_worker_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            _write_manifest(run_dir, ("before",))
            module = _load_task10_module()
            with patch.object(
                module,
                "load_stage7a_motion_contract",
                return_value=_contract(),
            ):
                task = normalize_action_task(module.build_action())
            step = next(
                action
                for action in task["actions"]
                if action["type"] == "runtime_move_joints"
            )
            controller = _FakeController()
            worker = ActionWorker(controller, task, run_dir)
            worker._manifest = json.loads((run_dir / "points.json").read_text("utf-8"))
            worker._manifest["runtime_moves"] = []
            worker._save_manifest()

            with _runtime_fixture(run_dir, approved=True) as (runtime, _dialog):
                worker.runtime_move_joints_requested.connect(
                    lambda request: worker.respond_runtime_move_joints(
                        runtime.on_runtime_move_joints_requested(request)
                    )
                )
                worker._runtime_move_joints(step)

            self.assertEqual(len(controller.goto_calls), 1)
            manifest = json.loads((run_dir / "points.json").read_text("utf-8"))
            self.assertEqual(manifest["runtime_moves"][0]["status"], "motion_completed")
            self.assertTrue(
                manifest["runtime_moves"][0]["fresh_kinematic_audit"]["passed"]
            )

    def test_bad_request_and_stale_measurement_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "bad-key"
            _write_manifest(run_dir, ("before",))
            with _runtime_fixture(run_dir, approved=True) as (runtime, dialog):
                response = runtime.on_runtime_move_joints_requested(
                    _controller_request(request_key="unexpected")
                )
                self.assertEqual(response["decision"], "abort")
                self.assertEqual(dialog.proposals, [])
                self.assertTrue(runtime._processing_failed)

            stale_dir = Path(temporary) / "stale"
            _write_manifest(stale_dir, ("before",))
            request = _controller_request()
            request["controller_state"]["pose"][0] += 0.06
            with _runtime_fixture(stale_dir, approved=True) as (runtime, _dialog):
                stale_response = runtime.on_runtime_move_joints_requested(request)
                self.assertEqual(stale_response["decision"], "abort")
                self.assertIn("measurement_matches_request_state", stale_response["reason"])
                self.assertTrue(runtime._processing_failed)

            timestamp_dir = Path(temporary) / "bad-timestamp"
            _write_manifest(timestamp_dir, ("before",))
            timestamp_manifest = json.loads(
                (timestamp_dir / "points.json").read_text("utf-8")
            )
            timestamp_manifest["points"][0].pop("recorded_at")
            (timestamp_dir / "points.json").write_text(
                json.dumps(timestamp_manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with _runtime_fixture(timestamp_dir, approved=True) as (runtime, _dialog):
                timestamp_response = runtime.on_runtime_move_joints_requested(
                    _controller_request()
                )
                self.assertEqual(timestamp_response["decision"], "abort")
                self.assertIn("时间戳", timestamp_response["reason"])
                self.assertTrue(runtime._processing_failed)

    def test_safety_rejection_is_not_mislabeled_as_operator_decline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "safety-rejected"
            _write_manifest(run_dir, ("before",))
            request = _controller_request()
            request["controller_state"]["pose"][0] += 0.06
            with _runtime_fixture(run_dir, approved=False) as (runtime, _dialog):
                response = runtime.on_runtime_move_joints_requested(request)
                self.assertEqual(response["decision"], "decline")
                self.assertEqual(runtime._operator_decision, "safety_rejected")
                _write_manifest(run_dir, ("before", "after"))
                runtime.on_task_finished(True, "done", str(run_dir))

            report = json.loads(
                (run_dir / runtime_module.RESULT_FILENAME).read_text("utf-8")
            )
            self.assertEqual(report["status"], "safety_rejected")
            self.assertEqual(report["operator_decision"], "safety_rejected")
            self.assertFalse(report["action_result"]["motion_executed"])

    def test_post_controller_fault_rejects_response_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "post-fault"
            _write_manifest(run_dir, ("before",))
            with _runtime_fixture(run_dir, approved=True) as (runtime, _dialog):
                response = runtime.on_runtime_move_joints_requested(
                    _controller_request()
                )
                self.assertEqual(response["decision"], "approve")
                _write_manifest(run_dir, ("before", "after"))
                manifest = json.loads((run_dir / "points.json").read_text("utf-8"))
                manifest["runtime_moves"][0]["status"] = "motion_completed"
                manifest["points"][-1]["controller_safety"]["warn"] = 7
                (run_dir / "points.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                runtime.on_task_finished(True, "done", str(run_dir))

            report = json.loads(
                (run_dir / runtime_module.RESULT_FILENAME).read_text("utf-8")
            )
            gate = report["response_validation"]["quality_gates"][
                "post_controller_safety"
            ]
            self.assertFalse(gate["passed"])
            self.assertIn(
                "post_controller_safety",
                report["response_validation"]["failure_reasons"],
            )
            self.assertEqual(report["status"], "response_rejected")


class Stage7ADialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_white_black_style_position_columns_and_default_decline(self) -> None:
        dialog = Stage7AOperatorDialog()
        try:
            self.assertIn("background:#FFFFFF", dialog.styleSheet())
            self.assertIn("color:#111827", dialog.styleSheet())
            self.assertEqual(dialog.frame_table.columnCount(), 10)
            self.assertTrue(dialog.decline_button.isDefault())
            self.assertFalse(dialog.approve_button.isDefault())

            dialog.add_frame(
                {
                    "phase": "before",
                    "filename": "1_001.jpg",
                    "robot_pose": [P22_XY[0], P22_XY[1], -27.0, 180.0, 0.0, P22_RZ],
                    "accepted": True,
                    "used_marker_count": 4,
                    "visible_marker_count": 5,
                    "reprojection_rms_px": 0.3,
                    "image_error_px": [2.0, -1.0],
                    "image_error_norm_px": float(np.hypot(2.0, -1.0)),
                },
                None,
            )
            self.assertEqual(
                dialog.frame_table.item(0, 2).text(), f"{P22_XY[0]:.3f}"
            )
            self.assertEqual(
                dialog.frame_table.item(0, 3).text(), f"{P22_XY[1]:.3f}"
            )
            proposal = {
                "ready_for_operator_confirmation": True,
                "motion_required": True,
                "measurement": {
                    "current_robot_xy_mm": list(P22_XY),
                    "median_error_px": [2.0, -1.0],
                },
                "calculation": {
                    "full_cancellation_correction_xy_mm": [-0.5, -0.25],
                    "commanded_correction_xy_mm": [-0.2, -0.1],
                    "predicted_endpoint_xy_mm": [P22_XY[0] - 0.2, P22_XY[1] - 0.1],
                    "predicted_error_px": [1.2, -0.6],
                    "was_clamped": True,
                },
                "safety_gates": {
                    "synthetic": {"passed": True, "actual": True, "limit": True},
                    "operator_consent": {
                        "passed": False,
                        "actual": False,
                        "limit": True,
                    },
                },
            }
            plan = {
                "target_joints": list(P22_JOINTS),
                "audit": {
                    "passed": True,
                    "sequential_transient_max_mm": 0.2,
                    "gates": {
                        "planner.synthetic": {
                            "passed": True,
                            "actual": 0.2,
                            "limit": "<=0.5 mm",
                        }
                    },
                },
            }
            dialog.set_proposal(proposal, plan)
            self.assertIn("world XY", dialog.summary.toPlainText())
            self.assertIn(f"{P22_XY[0]}", dialog.summary.toPlainText())
            self.assertIn("J4 Rz预补偿", dialog.summary.toPlainText())
            consent_rows = [
                row
                for row in range(dialog.gate_table.rowCount())
                if dialog.gate_table.item(row, 0).text() == "operator_consent"
            ]
            self.assertEqual(len(consent_rows), 1)
            self.assertEqual(
                dialog.gate_table.item(consent_rows[0], 1).text(), "PENDING"
            )

            QTimer.singleShot(0, dialog.decline_button.click)
            self.assertFalse(dialog.request_decision(proposal, plan))
        finally:
            dialog.close()


if __name__ == "__main__":
    unittest.main()
