from __future__ import annotations

import copy
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path

from scara.pipeline.kinematics import fk_wrist, j4_for_rz, rz_of, solve_joints
from scara.pipeline.xy_correction_planner import (
    audit_j4_only_orientation_target,
    plan_fixed_rz_xy_step,
)

try:
    from PyQt6 import QtCore as _qt_core  # noqa: F401
except ImportError:  # Allow protocol-only CI to run without the desktop wheel.
    qt_core = types.ModuleType("PyQt6.QtCore")

    class _Signal:
        def __init__(self):
            self._callbacks = []

        def connect(self, *_args, **_kwargs):
            self._callbacks.extend(_args[:1])
            return None

        def emit(self, *args, **kwargs):
            for callback in list(self._callbacks):
                callback(*args, **kwargs)

    class _SignalDescriptor:
        def __init__(self):
            self._storage_name = ""

        def __set_name__(self, _owner, name):
            self._storage_name = f"__qt_signal_{name}"

        def __get__(self, instance, _owner):
            if instance is None:
                return self
            signal = instance.__dict__.get(self._storage_name)
            if signal is None:
                signal = _Signal()
                instance.__dict__[self._storage_name] = signal
            return signal

    class _QtBase:
        def __init__(self, *_args, **_kwargs):
            pass

    class _Thread(_QtBase):
        pass

    class _Pool:
        @staticmethod
        def globalInstance():
            return _Pool()

    qt_core.QObject = _QtBase
    qt_core.QThread = _Thread
    qt_core.QTimer = _QtBase
    qt_core.QRunnable = _QtBase
    qt_core.QThreadPool = _Pool
    qt_core.pyqtSignal = lambda *_args, **_kwargs: _SignalDescriptor()
    pyqt_package = types.ModuleType("PyQt6")
    pyqt_package.QtCore = qt_core
    sys.modules["PyQt6"] = pyqt_package
    sys.modules["PyQt6.QtCore"] = qt_core

from scara.ui.action_worker import (
    ActionWorker,
    MOVED_TRAY_RUNTIME_REQUEST_KEY,
    WAFER_PICK_XY_RUNTIME_REQUEST_KEY,
    normalize_action_task,
)


HASH = "A" * 64


def runtime_step(**overrides):
    step = {
        "type": "runtime_move_joints",
        "name": "P31 XY overhead",
        "request_key": WAFER_PICK_XY_RUNTIME_REQUEST_KEY,
        "target_name": "P31",
        "calibration_sha256": HASH,
        "anchor_robot_xy_mm": [100.0, 250.0],
        "local_extent_mm": 70.0,
        "domain_margin_mm": 5.0,
        "required_j3_mm": -27.0046,
        "required_rz_deg": 20.8223,
        "final_rz_deg": -1.0,
        "max_j4_rotation_deg": 30.0,
        "max_xy_step_norm_mm": 10.0,
        "max_xy_axis_mm": 10.0,
        "j3_tolerance_mm": 0.20,
        "rz_tolerance_deg": 0.30,
        "target_rz_tolerance_deg": 0.15,
        "max_sequential_transient_rz_deg": 5.0,
        "precompensate_rz": True,
        "max_state_drift_xy_mm": 0.10,
        "max_state_drift_joint": 0.10,
        "max_sequential_transient_xy_mm": 8.0,
        "move_tolerance": 0.02,
        "proposal_max_age_s": 5.0,
        "fk_pose_xy_tolerance_mm": 0.20,
    }
    step.update(overrides)
    return step


def task(step):
    return {
        "api_version": 1,
        "name": "wafer pick XY test",
        "description": "XY only",
        "camera_model": {
            "offset_mm": 0.0,
            "angle_reference": "world_negative_y",
            "positive_rotation": "counter_clockwise_from_above",
        },
        "actions": [copy.deepcopy(step)],
    }


class WaferPickXYActionIntegrationTests(unittest.TestCase):
    def test_new_request_allows_any_valid_tray_slot(self) -> None:
        normalized = normalize_action_task(task(runtime_step()))
        step = normalized["actions"][0]
        self.assertEqual("P31", step["target_name"])
        self.assertEqual(10.0, step["max_xy_step_norm_mm"])
        self.assertEqual(-1.0, step["final_rz_deg"])
        self.assertEqual(30.0, step["max_j4_rotation_deg"])
        self.assertEqual(70.0, step["local_extent_mm"])
        self.assertFalse(step["enforce_sequential_intermediate_domain"])

    def test_existing_p22_request_cannot_be_relabelled(self) -> None:
        with self.assertRaisesRegex(ValueError, "P22"):
            normalize_action_task(
                task(
                    runtime_step(
                        request_key=MOVED_TRAY_RUNTIME_REQUEST_KEY,
                    )
                )
            )

    def test_worker_hard_limits_cannot_be_relaxed_by_task(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\(0, 10\]"):
            normalize_action_task(
                task(runtime_step(max_xy_step_norm_mm=10.01))
            )
        with self.assertRaisesRegex(ValueError, r"\(0, 30\]"):
            normalize_action_task(
                task(runtime_step(max_j4_rotation_deg=30.01))
            )
        with self.assertRaisesRegex(ValueError, r"\(0, 70\]"):
            normalize_action_task(task(runtime_step(local_extent_mm=70.01)))

    def test_invalid_slot_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "P00到P55"):
            normalize_action_task(task(runtime_step(target_name="P66")))

    def test_xy_overhead_requires_t1_and_speed_at_most_twenty_percent(self) -> None:
        state = {
            "controller_connected": True,
            "controller_enabled": True,
            "alarm_clear": True,
            "estop_clear": True,
            "soft_estop_clear": True,
            "controller_idle": True,
            "mode": "T1",
            "speed_percent": 20.0,
        }
        gates = ActionWorker._runtime_controller_gates(
            state, WAFER_PICK_XY_RUNTIME_REQUEST_KEY
        )
        self.assertTrue(all(gates.values()))

        gates = ActionWorker._runtime_controller_gates(
            {**state, "mode": "T2"}, WAFER_PICK_XY_RUNTIME_REQUEST_KEY
        )
        self.assertFalse(gates["controller_mode_is_t1"])
        gates = ActionWorker._runtime_controller_gates(
            {**state, "speed_percent": 20.01},
            WAFER_PICK_XY_RUNTIME_REQUEST_KEY,
        )
        self.assertFalse(gates["controller_speed_at_most_20_percent"])

    def test_wafer_pick_observe_timeout_continues_without_motion(self) -> None:
        joints = solve_joints(
            117.4353,
            271.2880,
            -27.0046,
            rz_deg=20.8223,
            ref_joints=[30.0, 85.0, -27.0046, -5.0],
        )
        self.assertIsNotNone(joints)
        xy = fk_wrist(joints[0], joints[1])

        class FakeController:
            def __init__(self):
                self._motion_sequence_lock = threading.Lock()
                self.goto_calls = 0

            def is_connected(self):
                return True

            def read_all_sync(self):
                return {
                    "joints": list(joints),
                    "pose": [
                        float(xy[0]),
                        float(xy[1]),
                        float(joints[2]),
                        180.0,
                        0.0,
                        rz_of(joints[0], joints[1], joints[3]),
                    ],
                    "enable": 1,
                    "effectively_enabled": True,
                    "warn": 0,
                    "need_clear": False,
                    "estop": False,
                    "soft_estop": False,
                    "mode": "T1",
                    "speed": 10.0,
                }

            def goto_joints_sync(self, *_args, **_kwargs):
                self.goto_calls += 1
                return True

            def emergency_stop(self, **_kwargs):
                return None

        controller = FakeController()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            output.mkdir()
            worker = ActionWorker(controller, task(runtime_step()), output)
            worker._manifest = {"runtime_moves": []}

            def respond(request):
                worker.respond_runtime_move_joints(
                    {
                        "request_id": request["request_id"],
                        "decision": "observe",
                        "calibration_sha256": HASH,
                        "reason": "synthetic camera dropout; retry next window",
                    }
                )

            worker.runtime_move_joints_requested.connect(respond)
            worker._runtime_move_joints(
                normalize_action_task(task(runtime_step()))["actions"][0]
            )

        self.assertEqual(0, controller.goto_calls)
        self.assertEqual(
            "observation_completed_no_motion",
            worker._manifest["runtime_moves"][0]["status"],
        )

    def test_action_worker_rejects_forbidden_z_or_vacuum_authorization(self) -> None:
        joints = solve_joints(
            117.4353,
            271.2880,
            -27.0046,
            rz_deg=20.8223,
            ref_joints=[30.0, 85.0, -27.0046, -5.0],
        )
        self.assertIsNotNone(joints)
        xy = fk_wrist(joints[0], joints[1])

        class FakeController:
            def __init__(self):
                self._motion_sequence_lock = threading.Lock()
                self.goto_calls = 0

            def is_connected(self):
                return True

            def read_all_sync(self):
                return {
                    "joints": list(joints),
                    "pose": [
                        float(xy[0]),
                        float(xy[1]),
                        -27.0046,
                        180.0,
                        0.0,
                        rz_of(joints[0], joints[1], joints[3]),
                    ],
                    "effectively_enabled": True,
                    "warn": 0,
                    "need_clear": False,
                    "estop": False,
                    "soft_estop": False,
                    "mode": "T1",
                    "speed": 10.0,
                }

            def goto_joints_sync(self, *_args, **_kwargs):
                self.goto_calls += 1
                return True

            def emergency_stop(self, **_kwargs):
                return None

        controller = FakeController()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            output.mkdir()
            worker = ActionWorker(controller, task(runtime_step()), output)
            worker._manifest = {"runtime_moves": []}

            def respond(request):
                worker.respond_runtime_move_joints(
                    {
                        "request_id": request["request_id"],
                        "decision": "approve",
                        "calibration_sha256": HASH,
                        "target_joints": list(joints),
                        "proposal": {
                            "proposal_id": "malicious-z-authorization",
                            "target_name": "P31",
                            "phase": "wafer_pick_xy_overhead",
                            "motion_authorized": True,
                            "xy_only": True,
                            "z_motion_authorized": True,
                            "vacuum_authorized": False,
                            "do_authorized": False,
                            "locked_j3_mm": -27.0046,
                            "locked_rz_deg": 20.8223,
                            "calculation": {
                                "commanded_correction_xy_mm": [0.0, 0.0]
                            },
                            "safety_gates": {"synthetic": {"passed": True}},
                        },
                    }
                )

            worker.runtime_move_joints_requested.connect(respond)
            with self.assertRaisesRegex(RuntimeError, "下降、真空或DO授权"):
                worker._runtime_move_joints(normalize_action_task(task(runtime_step()))["actions"][0])
        self.assertEqual(0, controller.goto_calls)

    def test_action_worker_executes_valid_xy_only_target_with_j3_unchanged(self) -> None:
        joints = solve_joints(
            117.4353,
            271.2880,
            -27.0046,
            rz_deg=20.8223,
            ref_joints=[30.0, 85.0, -27.0046, -5.0],
        )
        self.assertIsNotNone(joints)
        start_xy = fk_wrist(joints[0], joints[1])
        pose = [
            float(start_xy[0]),
            float(start_xy[1]),
            -27.0046,
            180.0,
            0.0,
            rz_of(joints[0], joints[1], joints[3]),
        ]
        plan = plan_fixed_rz_xy_step(
            joints,
            pose,
            [0.25, 0.0],
            anchor_robot_xy_mm=[100.0, 250.0],
            local_extent_mm=70.0,
            domain_margin_mm=5.0,
            required_j3_mm=-27.0046,
            j3_tolerance_mm=0.20,
            required_rz_deg=20.8223,
            rz_tolerance_deg=0.30,
            max_xy_step_norm_mm=2.0,
            max_xy_axis_mm=2.0,
            max_sequential_transient_xy_mm=5.0,
            target_rz_tolerance_deg=0.15,
            max_sequential_transient_rz_deg=1.0,
            precompensate_rz=True,
            enforce_sequential_intermediate_domain=False,
        )

        class FakeController:
            def __init__(self):
                self._motion_sequence_lock = threading.Lock()
                self.joints = list(joints)
                self.pose = list(pose)
                self.goto_calls: list[list[float]] = []

            def is_connected(self):
                return True

            def read_all_sync(self):
                return {
                    "joints": list(self.joints),
                    "pose": list(self.pose),
                    "enable": 1,
                    "effectively_enabled": True,
                    "warn": 0,
                    "need_clear": False,
                    "estop": False,
                    "soft_estop": False,
                    "mode": "T1",
                    "speed": 10.0,
                }

            def goto_joints_sync(self, _name, target, *, should_stop, tolerance):
                if should_stop() or tolerance <= 0.0:
                    return False
                self.goto_calls.append([float(value) for value in target])
                self.joints = [float(value) for value in target]
                xy = fk_wrist(self.joints[0], self.joints[1])
                self.pose = [
                    float(xy[0]),
                    float(xy[1]),
                    float(self.joints[2]),
                    180.0,
                    0.0,
                    rz_of(self.joints[0], self.joints[1], self.joints[3]),
                ]
                return True

            def emergency_stop(self, **_kwargs):
                return None

        controller = FakeController()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            output.mkdir()
            worker = ActionWorker(controller, task(runtime_step()), output)
            worker._manifest = {"runtime_moves": []}

            def respond(request):
                worker.respond_runtime_move_joints(
                    {
                        "request_id": request["request_id"],
                        "decision": "approve",
                        "calibration_sha256": HASH,
                        "target_joints": list(plan["target_joints"]),
                        "proposal": {
                            "proposal_id": "valid-xy-only",
                            "target_name": "P31",
                            "phase": "wafer_pick_xy_overhead",
                            "motion_authorized": True,
                            "xy_only": True,
                            "z_motion_authorized": False,
                            "vacuum_authorized": False,
                            "do_authorized": False,
                            "locked_j3_mm": -27.0046,
                            "locked_rz_deg": 20.8223,
                            "calculation": {
                                "commanded_correction_xy_mm": [0.25, 0.0]
                            },
                            "safety_gates": {"synthetic": {"passed": True}},
                        },
                    }
                )

            worker.runtime_move_joints_requested.connect(respond)
            worker._runtime_move_joints(
                normalize_action_task(task(runtime_step()))["actions"][0]
            )

        self.assertEqual(1, len(controller.goto_calls))
        self.assertAlmostEqual(joints[2], controller.goto_calls[0][2], places=9)
        self.assertAlmostEqual(joints[2], controller.joints[2], places=9)

    def test_action_worker_executes_final_j4_only_tray_alignment(self) -> None:
        joints = solve_joints(
            117.4353,
            271.2880,
            -27.0046,
            rz_deg=20.8223,
            ref_joints=[30.0, 85.0, -27.0046, -5.0],
        )
        self.assertIsNotNone(joints)
        start_xy = fk_wrist(joints[0], joints[1])
        target_rz = 16.3223
        target_joints = list(joints)
        target_joints[3] = j4_for_rz(joints[0], joints[1], target_rz)

        class FakeController:
            def __init__(self):
                self._motion_sequence_lock = threading.Lock()
                self.joints = list(joints)
                self.goto_calls: list[list[float]] = []

            def is_connected(self):
                return True

            def read_all_sync(self):
                xy = fk_wrist(self.joints[0], self.joints[1])
                return {
                    "joints": list(self.joints),
                    "pose": [
                        float(xy[0]),
                        float(xy[1]),
                        float(self.joints[2]),
                        180.0,
                        0.0,
                        rz_of(self.joints[0], self.joints[1], self.joints[3]),
                    ],
                    "enable": 1,
                    "effectively_enabled": True,
                    "warn": 0,
                    "need_clear": False,
                    "estop": False,
                    "soft_estop": False,
                    "mode": "T1",
                    "speed": 10.0,
                }

            def goto_joints_sync(self, _name, target, *, should_stop, tolerance):
                if should_stop() or tolerance <= 0.0:
                    return False
                self.goto_calls.append([float(value) for value in target])
                self.joints = [float(value) for value in target]
                return True

            def emergency_stop(self, **_kwargs):
                return None

        controller = FakeController()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            output.mkdir()
            worker = ActionWorker(controller, task(runtime_step()), output)
            worker._manifest = {"runtime_moves": []}

            def respond(request):
                worker.respond_runtime_move_joints(
                    {
                        "request_id": request["request_id"],
                        "decision": "approve",
                        "calibration_sha256": HASH,
                        "target_joints": target_joints,
                        "proposal": {
                            "proposal_id": "valid-final-j4-only",
                            "target_name": "P31",
                            "phase": "wafer_pick_final_tray_orientation",
                            "motion_authorized": True,
                            "xy_only": False,
                            "j4_only": True,
                            "z_motion_authorized": False,
                            "vacuum_authorized": False,
                            "do_authorized": False,
                            "locked_j3_mm": -27.0046,
                            "locked_rz_deg": target_rz,
                            "calculation": {
                                "commanded_correction_xy_mm": [0.0, 0.0],
                                "camera2_median_angle_error_deg": 4.5,
                                "commanded_j4_correction_deg": -4.5,
                                "current_absolute_rz_deg": 20.8223,
                                "target_absolute_rz_deg": target_rz,
                            },
                            "safety_gates": {"synthetic": {"passed": True}},
                        },
                    }
                )

            worker.runtime_move_joints_requested.connect(respond)
            worker._runtime_move_joints(
                normalize_action_task(task(runtime_step()))["actions"][0]
            )

        self.assertEqual(1, len(controller.goto_calls))
        for index in range(3):
            self.assertAlmostEqual(
                joints[index], controller.goto_calls[0][index], places=9
            )
        self.assertAlmostEqual(float(start_xy[0]), fk_wrist(*controller.joints[:2])[0])
        self.assertAlmostEqual(
            target_rz, rz_of(*controller.joints[:2], controller.joints[3])
        )

    def test_j4_only_audit_rejects_wrapped_full_turn_command(self) -> None:
        joints = solve_joints(
            117.4353,
            271.2880,
            -27.0046,
            rz_deg=20.8223,
            ref_joints=[30.0, 85.0, -27.0046, -5.0],
        )
        self.assertIsNotNone(joints)
        xy = fk_wrist(joints[0], joints[1])
        target = list(joints)
        target[3] += 360.0
        audit = audit_j4_only_orientation_target(
            joints,
            [float(xy[0]), float(xy[1]), -27.0046, 180.0, 0.0, 20.8223],
            target,
            anchor_robot_xy_mm=[100.0, 250.0],
            local_extent_mm=70.0,
            domain_margin_mm=5.0,
            required_j3_mm=-27.0046,
            j3_tolerance_mm=0.20,
            required_start_rz_deg=20.8223,
            start_rz_tolerance_deg=0.30,
            target_rz_deg=20.8223,
            target_rz_tolerance_deg=0.15,
            maximum_j4_rotation_deg=30.0,
        )
        self.assertFalse(audit["passed"])
        self.assertFalse(audit["gates"]["j4_rotation_limit"]["passed"])


if __name__ == "__main__":
    unittest.main()
