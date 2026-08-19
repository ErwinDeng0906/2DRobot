"""Offline safety/overlay tests for the Stage-6 hand-eye demonstration.

These tests never construct a robot controller, open a physical camera, send a
motion command, or write a DO.  A fake camera and temporary calibration tree
exercise the Qt dialog without showing a blocking modal window.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import sys
import tempfile
import threading
import time
import unittest
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

from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication, QGraphicsView, QMessageBox

from scara.ui.camera_view import (
    DIRECTSHOW_EXPOSURE_DEFAULT,
    DIRECTSHOW_EXPOSURE_MAX,
    DIRECTSHOW_EXPOSURE_MIN,
    ScaraCameraThread,
    _validated_exposure_value,
)
from scara.ui.handeye_demo_dialog import CameraImageView, HandEyeDemoDialog
from scara.vision.handeye_interaction import (
    SuctionTargetModel,
    evaluate_handeye_frame,
    load_local_xy_jacobian,
    project_tray_points_from_transform,
)
from scara.vision.tray_pose_estimator import CameraIntrinsics, TrayPoseEstimate
from scara.vision.tray_pose_tracker import TrackedTrayPose
from scara.vision.xy_image_jacobian import (
    REQUIRED_XY_JACOBIAN_QUALITY_GATES,
    correction_command_xy_mm,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        K=np.array(
            [[618.6, 0.0, 635.5], [0.0, 619.9, 355.9], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        dist_coeffs=np.zeros((5, 1), dtype=np.float64),
        image_size=(1280, 720),
        source_path="synthetic",
        calibration_status="success",
        global_rms_px=0.47,
    )


def _raw_pose(
    image: np.ndarray,
    *,
    passed: bool,
    transform: np.ndarray | None,
) -> TrayPoseEstimate:
    inverse = None if transform is None else np.linalg.inv(transform)
    return TrayPoseEstimate(
        success=passed,
        quality_passed=passed,
        failure_reason=None if passed else "synthetic Stage3 rejection",
        visible_marker_ids=(1, 2, 3, 4) if passed else (1, 2),
        used_marker_ids=(1, 2, 3, 4) if passed else (),
        rejected_marker_ids=(),
        ransac_inlier_corner_count=16 if passed else 0,
        object_span_mm=100.0 if passed else 0.0,
        reprojection_rms_px=0.5 if passed else None,
        per_marker_rms_px={1: 0.5} if passed else {},
        rvec_C_T=None,
        tvec_C_T_mm=None,
        T_C_T=transform,
        T_T_C=inverse,
        camera_position_T_mm=None,
        minimum_object_depth_C_mm=400.0 if passed else None,
        annotated_image=image.copy(),
    )


def _tracked_pose(
    image: np.ndarray,
    *,
    accepted: bool,
    transform: np.ndarray | None,
    retained_filtered_transform: np.ndarray | None = None,
) -> TrackedTrayPose:
    raw = _raw_pose(image, passed=accepted, transform=transform)
    filtered = transform if accepted else retained_filtered_transform
    return TrackedTrayPose(
        raw=raw,
        accepted_by_tracker=accepted,
        tracker_reason=None if accepted else "synthetic tracker rejection",
        filtered_T_C_T=filtered,
        filtered_T_T_C=None if filtered is None else np.linalg.inv(filtered),
        translation_jump_mm=0.0 if accepted else None,
        rotation_jump_deg=0.0 if accepted else None,
        lost_frame_count=0 if accepted else 1,
    )


def _suction(
    target_pixel_px: tuple[float, float] = (625.5, 218.5),
) -> SuctionTargetModel:
    return SuctionTargetModel(
        source_path=Path("synthetic-camera1-suction-target.json"),
        source_sha256="A" * 64,
        camera_source=1,
        resolution=(1280, 720),
        p_C_S_mm=(-2.4, -33.0, 149.5),
        target_pixel_px=target_pixel_px,
        working_plane_z_T_mm=-2.0,
        imaging_j3_mm=-27.0046,
        rz_mean_deg=20.8209,
    )


def _all_pass_gates() -> dict[str, dict[str, bool]]:
    return {
        name: {"passed": True}
        for name in REQUIRED_XY_JACOBIAN_QUALITY_GATES
    }


def _valid_fit() -> dict[str, object]:
    return {
        "status": "success",
        "j_error_px_per_command_mm": [[2.0, 0.2], [-0.1, 1.8]],
        "j_command_mm_per_error_px": np.linalg.inv(
            np.array([[2.0, 0.2], [-0.1, 1.8]], dtype=np.float64)
        ).tolist(),
        "quality_gates": _all_pass_gates(),
    }


ANCHOR_ROBOT_XY_MM = (300.0, 100.0)
EXPECTED_J3_MM = -27.0046
EXPECTED_RZ_DEG = 20.8209


def _valid_jacobian_payload() -> dict[str, object]:
    return {
        "schema_version": 2,
        "status": "success",
        "anchor_target_name": "P22",
        "valid_target_names": ["P22"],
        "coordinate_definition": {
            "command_frame": "robot_controller_world_XY",
            "image_error": (
                "slot_pixel_distorted - suction_target_pixel_distorted"
            ),
            "anchor_robot_xy_mm": list(ANCHOR_ROBOT_XY_MM),
            "imaging_j3_mm": EXPECTED_J3_MM,
            "rz_deg": EXPECTED_RZ_DEG,
            "offset_extent_mm": 2.0,
        },
        "fit": _valid_fit(),
    }


def _fresh_robot_state(
    *,
    delta_xy_mm: tuple[float, float] = (0.0, 0.0),
    j3_mm: float = EXPECTED_J3_MM,
    rz_deg: float = EXPECTED_RZ_DEG,
    age_s: float = 0.0,
) -> dict[str, object]:
    # rz_of(J1,J2,J4) = J1 + J2 + J4 - 90 deg.
    j1_deg = 10.0
    j2_deg = 20.0
    j4_deg = rz_deg + 90.0 - j1_deg - j2_deg
    return {
        "joints": [j1_deg, j2_deg, j3_mm, j4_deg],
        "pose": [
            ANCHOR_ROBOT_XY_MM[0] + delta_xy_mm[0],
            ANCHOR_ROBOT_XY_MM[1] + delta_xy_mm[1],
            j3_mm,
            0.0,
            0.0,
            rz_deg,
        ],
        "captured_monotonic_s": time.monotonic() - age_s,
    }


class FakeCamera:
    """Minimal camera-1 API; it has no motion or digital-output methods."""

    source_index = 1

    def __init__(self, *, running: bool = True) -> None:
        self.running = running
        self.latest_frame_calls = 0

    def isRunning(self) -> bool:  # noqa: N802 - Qt-compatible spelling
        return self.running

    def latest_frame(self, max_age_s: float = 1.0):
        del max_age_s
        self.latest_frame_calls += 1
        return None


class ExposureCamera(FakeCamera):
    def __init__(self, *, running: bool = True) -> None:
        super().__init__(running=running)
        self.exposure_requests: list[int] = []
        self.restore_calls = 0
        self.recovery_calls = 0

    def request_exposure_value(self, exposure: int) -> bool:
        self.exposure_requests.append(int(exposure))
        return self.running

    def restore_original_exposure(self) -> bool:
        self.restore_calls += 1
        return self.running

    def request_auto_exposure_recovery(self) -> bool:
        self.recovery_calls += 1
        return self.running


class FrozenPacketCamera(FakeCamera):
    """Return one captured frame forever, with an unchanged sequence number."""

    def __init__(self) -> None:
        super().__init__(running=True)
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.captured_at = time.monotonic()

    def latest_frame(self, max_age_s: float = 1.0):
        del max_age_s
        self.latest_frame_calls += 1
        return self.frame.copy()

    def latest_frame_packet(self, max_age_s: float = 1.0):
        del max_age_s
        self.latest_frame_calls += 1
        return self.frame.copy(), 1, self.captured_at


class ReadOnlySourceAuditTests(unittest.TestCase):
    def test_hardware_exposure_accepts_only_supported_integer_stops(self) -> None:
        for value in range(DIRECTSHOW_EXPOSURE_MIN, DIRECTSHOW_EXPOSURE_MAX + 1):
            self.assertEqual(_validated_exposure_value(value), value)
            self.assertEqual(_validated_exposure_value(float(value)), value)
        for invalid in (-14, -5.5, -0.5, 0, float("nan")):
            with self.assertRaises(ValueError):
                _validated_exposure_value(invalid)

        camera = ScaraCameraThread(index=1)
        with self.assertRaises(ValueError):
            camera.request_exposure_value(-5.5)

    def test_camera_thread_applies_driver_exposure_not_pixel_brightness(self) -> None:
        class FakeHardwareCapture:
            def __init__(self) -> None:
                self.values = {
                    cv2.CAP_PROP_EXPOSURE: -5.0,
                    cv2.CAP_PROP_AUTO_EXPOSURE: 0.75,
                }
                self.set_calls: list[tuple[int, float]] = []

            def get(self, prop: int) -> float:
                return float(self.values[prop])

            def set(self, prop: int, value: float) -> bool:
                self.set_calls.append((prop, float(value)))
                self.values[prop] = float(value)
                return True

        camera = ScaraCameraThread(index=1)
        capture = FakeHardwareCapture()
        success, _message = camera._apply_hardware_exposure(capture, cv2, -6)
        self.assertTrue(success)
        self.assertIn((cv2.CAP_PROP_AUTO_EXPOSURE, 0.25), capture.set_calls)
        self.assertIn((cv2.CAP_PROP_EXPOSURE, -6.0), capture.set_calls)
        success, _message = camera._restore_hardware_exposure(capture, cv2)
        self.assertTrue(success)
        self.assertIn((cv2.CAP_PROP_AUTO_EXPOSURE, 0.75), capture.set_calls)

    def test_fractional_exposure_is_rejected_before_touching_driver(self) -> None:
        class FakeHardwareCapture:
            def __init__(self) -> None:
                self.values = {
                    cv2.CAP_PROP_EXPOSURE: -5.0,
                    cv2.CAP_PROP_AUTO_EXPOSURE: 0.75,
                }
                self.set_calls: list[tuple[int, float]] = []

            def get(self, prop: int) -> float:
                return float(self.values[prop])

            def set(self, prop: int, value: float) -> bool:
                self.set_calls.append((prop, float(value)))
                self.values[prop] = float(value)
                return True

        camera = ScaraCameraThread(index=1)
        capture = FakeHardwareCapture()
        with self.assertRaises(ValueError):
            camera._apply_hardware_exposure(capture, cv2, -5.5)
        self.assertEqual(capture.set_calls, [])

    def test_fixed_one_auto_mode_readback_does_not_block_integer_exposure(self) -> None:
        class FixedOneModeCapture:
            def __init__(self) -> None:
                self.exposure = -5.0
                self.set_calls: list[tuple[int, float]] = []

            def get(self, prop: int) -> float:
                if prop == cv2.CAP_PROP_AUTO_EXPOSURE:
                    return 1.0
                return float(self.exposure)

            def set(self, prop: int, value: float) -> bool:
                self.set_calls.append((prop, float(value)))
                if prop == cv2.CAP_PROP_EXPOSURE:
                    self.exposure = float(value)
                # This driver accepts mode writes but always reads back 1.0.
                return True

        camera = ScaraCameraThread(index=1)
        capture = FixedOneModeCapture()
        success, message = camera._apply_hardware_exposure(capture, cv2, -6)
        self.assertTrue(success)
        self.assertEqual(capture.exposure, -6.0)
        self.assertIn("自动模式读回 1.000 不可靠", message)
        self.assertIn("稳定后的整数曝光读回确认", message)

    def test_unsupported_minus_one_mode_readback_uses_exposure_as_evidence(self) -> None:
        class UnsupportedModeReadbackCapture:
            def __init__(self) -> None:
                self.exposure = -5.0

            def get(self, prop: int) -> float:
                if prop == cv2.CAP_PROP_AUTO_EXPOSURE:
                    return -1.0
                return float(self.exposure)

            def set(self, prop: int, value: float) -> bool:
                if prop == cv2.CAP_PROP_EXPOSURE:
                    self.exposure = float(value)
                return True

        camera = ScaraCameraThread(index=1)
        capture = UnsupportedModeReadbackCapture()
        success, message = camera._apply_hardware_exposure(capture, cv2, -6)
        self.assertTrue(success)
        self.assertEqual(capture.exposure, -6.0)
        self.assertIn("自动模式读回 -1.000 不可靠", message)
        restored, restore_message = camera._restore_hardware_exposure(capture, cv2)
        self.assertTrue(restored)
        self.assertEqual(capture.exposure, -5.0)
        self.assertIn("已恢复原整数曝光值", restore_message)

    def test_unknown_mode_readback_accepts_default_auto_without_fixed_write(self) -> None:
        class UnsupportedAutoReadbackCapture:
            def __init__(self) -> None:
                self.set_calls: list[tuple[int, float]] = []

            def get(self, prop: int) -> float:
                if prop == cv2.CAP_PROP_AUTO_EXPOSURE:
                    return -1.0
                return -5.0

            def set(self, prop: int, value: float) -> bool:
                self.set_calls.append((prop, float(value)))
                return True

            def read(self):
                return True, np.full((20, 20, 3), 80, dtype=np.uint8)

        camera = ScaraCameraThread(index=1)
        capture = UnsupportedAutoReadbackCapture()
        with patch("scara.ui.camera_view.time.sleep", return_value=None):
            success, message = camera._recover_auto_exposure(capture, cv2)
        self.assertTrue(success, message)
        self.assertIn("驱动已接受请求", message)
        self.assertTrue(camera._original_auto_exposure_enabled)
        self.assertEqual(
            [(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)],
            capture.set_calls,
        )
        self.assertFalse(
            any(prop == cv2.CAP_PROP_EXPOSURE for prop, _value in capture.set_calls)
        )

    def test_restore_requires_confirmed_auto_mode_instead_of_exposure_only(self) -> None:
        class AutoRestoreFailureCapture:
            def __init__(self) -> None:
                self.values = {
                    cv2.CAP_PROP_EXPOSURE: -6.0,
                    cv2.CAP_PROP_AUTO_EXPOSURE: 0.25,
                }

            def get(self, prop: int) -> float:
                return float(self.values[prop])

            def set(self, prop: int, value: float) -> bool:
                if prop == cv2.CAP_PROP_AUTO_EXPOSURE:
                    return False
                self.values[prop] = float(value)
                return True

        camera = ScaraCameraThread(index=1)
        camera._original_exposure_raw = -5.0
        camera._original_auto_exposure_raw = 0.75
        camera._original_auto_exposure_enabled = True
        success, message = camera._restore_hardware_exposure(
            AutoRestoreFailureCapture(), cv2
        )
        self.assertFalse(success)
        self.assertIn("未确认恢复自动曝光", message)

    def test_black_frame_guard_rolls_back_to_auto_exposure(self) -> None:
        class BlackAfterManualCapture:
            def __init__(self) -> None:
                self.values = {
                    cv2.CAP_PROP_EXPOSURE: -5.0,
                    cv2.CAP_PROP_AUTO_EXPOSURE: 0.75,
                }
                self.set_calls: list[tuple[int, float]] = []

            def get(self, prop: int) -> float:
                return float(self.values[prop])

            def set(self, prop: int, value: float) -> bool:
                self.set_calls.append((prop, float(value)))
                self.values[prop] = float(value)
                return True

            def read(self):
                return True, np.zeros((20, 20, 3), dtype=np.uint8)

        camera = ScaraCameraThread(index=1)
        camera._last_frame = np.full((20, 20, 3), 120, dtype=np.uint8)
        capture = BlackAfterManualCapture()
        success, message = camera._apply_hardware_exposure(capture, cv2, -6)
        self.assertFalse(success)
        self.assertIn("画面亮度异常下降", message)
        self.assertEqual(capture.values[cv2.CAP_PROP_AUTO_EXPOSURE], 0.75)

    def test_read_only_modules_do_not_import_or_call_motion_backends(self) -> None:
        sources = [
            SRC / "scara/ui/handeye_demo_dialog.py",
            SRC / "scara/vision/handeye_interaction.py",
            SRC / "scara/vision/tray_vision_fusion.py",
        ]
        forbidden_import_fragments = (
            "controller",
            "action_worker",
            "snrobot",
            "digital_output",
        )
        forbidden_call_names = {
            "cmd_goto_preset",
            "jog_start",
            "jog_stop",
            "move_joints",
            "move_linear",
            "set_do",
            "write_do",
            "vacuum_on",
            "vacuum_off",
            "emergency_stop",
        }
        for source_path in sources:
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            imports = []
            call_names = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or "")
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        call_names.append(node.func.attr)
                    elif isinstance(node.func, ast.Name):
                        call_names.append(node.func.id)
            self.assertFalse(
                any(
                    fragment in imported.lower()
                    for imported in imports
                    for fragment in forbidden_import_fragments
                ),
                f"{source_path} imports a motion/DO backend: {imports}",
            )
            self.assertTrue(
                forbidden_call_names.isdisjoint(call_names),
                f"{source_path} contains a forbidden hardware call",
            )

    def test_control_widget_open_demo_has_no_controller_calls(self) -> None:
        source = (SRC / "scara/ui/control_widget.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_open_handeye_demo"
        )
        controller_calls = []
        for node in ast.walk(method):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            value = node.func.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "_ctrl"
                and isinstance(value.value, ast.Name)
                and value.value.id == "self"
            ):
                controller_calls.append(node.func.attr)
        self.assertEqual(controller_calls, [])

    def test_overlay_and_ui_do_not_claim_placement_alignment(self) -> None:
        interaction_source = (
            SRC / "scara/vision/handeye_interaction.py"
        ).read_text(encoding="utf-8")
        dialog_source = (SRC / "scara/ui/handeye_demo_dialog.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("ALIGNED", interaction_source)
        self.assertNotIn("已对准", dialog_source)
        self.assertIn("COMPUTE ONLY - NO ROBOT MOTION", interaction_source)
        self.assertIn('f"{target_name} e=(', interaction_source)
        self.assertIn('f"VISUAL |e|<=', interaction_source)
        self.assertIn("NOT PLACEMENT ACCEPTANCE", interaction_source)
        self.assertIn("非放置精度验收", dialog_source)

    def test_control_widget_status_cache_fails_closed_after_disconnect(self) -> None:
        from scara.ui.control_widget import ScaraControlWidget

        owner = SimpleNamespace(
            _handeye_state_lock=threading.Lock(),
            _handeye_controller_connected=False,
            _latest_handeye_robot_state={"sentinel": True},
        )
        status = {
            "joints": [10.0, 20.0, EXPECTED_J3_MM, 80.8209],
            "pose": [300.0, 100.0, EXPECTED_J3_MM, 0.0, 0.0, EXPECTED_RZ_DEG],
        }

        # A queued status arriving after disconnect must not revive the cache.
        ScaraControlWidget._cache_handeye_robot_state(owner, status)
        self.assertIsNone(owner._latest_handeye_robot_state)
        self.assertIsNone(ScaraControlWidget._handeye_robot_state_snapshot(owner))

        owner._handeye_controller_connected = True
        ScaraControlWidget._cache_handeye_robot_state(owner, status)
        snapshot = ScaraControlWidget._handeye_robot_state_snapshot(owner)
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["joints"], status["joints"])
        self.assertEqual(snapshot["pose"], status["pose"])
        self.assertTrue(math.isfinite(snapshot["captured_monotonic_s"]))

        owner._handeye_controller_connected = False
        self.assertIsNone(ScaraControlWidget._handeye_robot_state_snapshot(owner))

    def test_validation_rechecks_robot_state_age_at_click_time(self) -> None:
        current = SimpleNamespace(
            jacobian_domain_passed=True,
            correction_available=True,
            robot_state_age_s=0.80,
        )
        self.assertTrue(
            HandEyeDemoDialog._current_domain_still_fresh(current, 0.19)
        )
        self.assertFalse(
            HandEyeDemoDialog._current_domain_still_fresh(current, 0.21)
        )
        current.robot_state_age_s = None
        self.assertFalse(
            HandEyeDemoDialog._current_domain_still_fresh(current, 0.0)
        )


class HandEyeEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = json.loads(
            (SRC / "scara/calib/tray_board_geometry.json").read_text(
                encoding="utf-8-sig"
            )
        )

    def test_geometry_exposes_exactly_all_36_slots(self) -> None:
        expected = {f"P{row}{column}" for row in range(6) for column in range(6)}
        self.assertEqual(set(self.geometry["slots"]), expected)

    def test_accepted_pose_draws_required_overlay_and_computes_error(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        rotation, _ = cv2.Rodrigues(
            np.radians(np.array([22.0, -9.0, 4.0], dtype=np.float64))
        )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = [45.0, 30.0, 500.0]
        tracked = _tracked_pose(frame, accepted=True, transform=transform)
        slot_pixel = project_tray_points_from_transform(
            [self.geometry["slots"]["P22"]], transform, _intrinsics()
        )[0]
        desired_error = np.array([1.0, -1.0], dtype=np.float64)
        suction = _suction(
            tuple((slot_pixel - desired_error).astype(float).tolist())
        )

        evaluation = evaluate_handeye_frame(
            frame,
            tracked,
            "P22",
            self.geometry,
            _intrinsics(),
            suction,
            _valid_jacobian_payload(),
            _fresh_robot_state(),
        )

        self.assertTrue(evaluation.accepted)
        self.assertIsNotNone(evaluation.slot_pixel_px)
        self.assertIsNotNone(evaluation.image_error_px)
        self.assertIsNotNone(evaluation.correction_xy_mm)
        self.assertTrue(evaluation.jacobian_domain_passed)
        self.assertEqual(evaluation.alignment_threshold_px, 3.0)
        self.assertTrue(evaluation.aligned)
        self.assertEqual(evaluation.annotated_bgr.shape, frame.shape)
        self.assertIsNotNone(evaluation.tray_transform_C_T)
        self.assertIsNotNone(evaluation.suction_point_T_mm)
        self.assertIsNotNone(evaluation.target_point_T_mm)
        self.assertIsNotNone(evaluation.metric_error_T_mm)
        np.testing.assert_allclose(
            np.asarray(evaluation.target_point_T_mm)
            - np.asarray(evaluation.suction_point_T_mm),
            evaluation.metric_error_T_mm,
            atol=1e-9,
        )

        # Exact drawing colours survive OpenCV anti-aliasing at the line core.
        colours = evaluation.annotated_bgr.reshape(-1, 3)
        for bgr, label in (
            ((0, 0, 255), "red suction target / T-X"),
            ((0, 255, 0), "green slot target / T-Y"),
            ((0, 255, 255), "yellow error arrow"),
            ((255, 255, 0), "cyan A-H reprojected corners"),
            ((255, 0, 0), "blue T-Z axis"),
        ):
            self.assertTrue(
                np.any(np.all(colours == np.asarray(bgr), axis=1)),
                f"missing overlay colour for {label}",
            )

        marker = next(iter(self.geometry["markers"].values()))
        projected = project_tray_points_from_transform(
            marker["corners_T_mm"], transform, _intrinsics()
        )
        self.assertEqual(projected.shape, (4, 2))

    def test_rejected_stage3_never_uses_retained_tracker_pose(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        stale_transform = np.eye(4, dtype=np.float64)
        stale_transform[:3, 3] = [45.0, 30.0, 500.0]
        tracked = _tracked_pose(
            frame,
            accepted=False,
            transform=None,
            retained_filtered_transform=stale_transform,
        )

        evaluation = evaluate_handeye_frame(
            frame,
            tracked,
            "P22",
            self.geometry,
            _intrinsics(),
            _suction(),
            _valid_jacobian_payload(),
            _fresh_robot_state(),
        )

        self.assertFalse(evaluation.accepted)
        self.assertIsNone(evaluation.slot_pixel_px)
        self.assertIsNone(evaluation.image_error_px)
        self.assertEqual(evaluation.alignment_threshold_px, 3.0)
        self.assertIsNone(evaluation.aligned)
        self.assertIsNone(evaluation.correction_xy_mm)
        self.assertFalse(evaluation.correction_available)

    def test_optional_tray_overlay_is_preserved_under_handeye_drawing(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        base = np.full_like(frame, (17, 33, 49))
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = [45.0, 30.0, 500.0]
        tracked = _tracked_pose(frame, accepted=True, transform=transform)
        slot_pixel = project_tray_points_from_transform(
            [self.geometry["slots"]["P22"]], transform, _intrinsics()
        )[0]
        evaluation = evaluate_handeye_frame(
            frame,
            tracked,
            "P22",
            self.geometry,
            _intrinsics(),
            _suction(tuple((slot_pixel - np.array([1.0, -1.0])).tolist())),
            base_annotated_bgr=base,
        )
        preserved = np.all(
            evaluation.annotated_bgr == np.asarray((17, 33, 49)), axis=2
        )
        self.assertGreater(int(np.count_nonzero(preserved)), frame.shape[0] * frame.shape[1] // 2)

        with self.assertRaisesRegex(ValueError, "must match"):
            evaluate_handeye_frame(
                frame,
                tracked,
                "P22",
                self.geometry,
                _intrinsics(),
                _suction(),
                base_annotated_bgr=np.zeros((10, 10, 3), dtype=np.uint8),
            )

    def _evaluate_domain_case(
        self,
        *,
        image_error_px: tuple[float, float] = (1.0, -1.0),
        robot_state: object = ...,
        target_name: str = "P22",
    ):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = [25.0, 15.0, 500.0]
        tracked = _tracked_pose(frame, accepted=True, transform=transform)
        slot_pixel = project_tray_points_from_transform(
            [self.geometry["slots"][target_name]], transform, _intrinsics()
        )[0]
        suction = _suction(
            tuple((slot_pixel - np.asarray(image_error_px)).astype(float).tolist())
        )
        state = _fresh_robot_state() if robot_state is ... else robot_state
        return evaluate_handeye_frame(
            frame,
            tracked,
            target_name,
            self.geometry,
            _intrinsics(),
            suction,
            _valid_jacobian_payload(),
            state,
        )

    def test_jacobian_domain_requires_fresh_matching_robot_state(self) -> None:
        cases = (
            ("missing", None, "缺少"),
            ("stale", _fresh_robot_state(age_s=1.01), "过期"),
            (
                "xy outside",
                _fresh_robot_state(delta_xy_mm=(2.002, 0.0)),
                "超出Task9局部域",
            ),
            (
                "j3 mismatch",
                _fresh_robot_state(j3_mm=EXPECTED_J3_MM + 0.201),
                "J3偏离",
            ),
            (
                "rz mismatch",
                _fresh_robot_state(rz_deg=EXPECTED_RZ_DEG + 0.201),
                "Rz偏离",
            ),
        )
        for label, state, expected_note in cases:
            with self.subTest(label=label):
                evaluation = self._evaluate_domain_case(robot_state=state)
                self.assertTrue(evaluation.accepted)
                self.assertFalse(evaluation.jacobian_domain_passed)
                self.assertFalse(evaluation.correction_available)
                self.assertIsNone(evaluation.correction_xy_mm)
                self.assertIn(expected_note, evaluation.jacobian_domain_note)

    def test_jacobian_domain_is_p22_only(self) -> None:
        evaluation = self._evaluate_domain_case(target_name="P20")
        self.assertFalse(evaluation.jacobian_domain_passed)
        self.assertIsNone(evaluation.correction_xy_mm)
        self.assertIn("只验证于P22", evaluation.jacobian_domain_note)

    def test_jacobian_domain_rejects_large_correction_and_endpoint(self) -> None:
        large = self._evaluate_domain_case(image_error_px=(10.0, 0.0))
        self.assertFalse(large.jacobian_domain_passed)
        self.assertIsNone(large.correction_xy_mm)
        self.assertIn("候选XY修正超出", large.jacobian_domain_note)

        # This error asks for +0.5 mm X.  The command itself is local, but from
        # current delta +1.8 mm its predicted endpoint would be +2.3 mm.
        endpoint = self._evaluate_domain_case(
            image_error_px=(-1.0, 0.05),
            robot_state=_fresh_robot_state(delta_xy_mm=(1.8, 0.0)),
        )
        self.assertFalse(endpoint.jacobian_domain_passed)
        self.assertIsNone(endpoint.correction_xy_mm)
        self.assertIn("预测终点越出", endpoint.jacobian_domain_note)

    def test_correction_fails_closed_for_missing_or_failed_quality_gate(self) -> None:
        valid = _valid_fit()
        self.assertIsNotNone(correction_command_xy_mm([4.0, -2.0], valid))

        missing = json.loads(json.dumps(valid))
        missing["quality_gates"].pop(next(iter(REQUIRED_XY_JACOBIAN_QUALITY_GATES)))
        self.assertIsNone(correction_command_xy_mm([4.0, -2.0], missing))

        failed = json.loads(json.dumps(valid))
        failed_gate = next(iter(REQUIRED_XY_JACOBIAN_QUALITY_GATES))
        failed["quality_gates"][failed_gate]["passed"] = False
        self.assertIsNone(correction_command_xy_mm([4.0, -2.0], failed))


class HandEyeDialogOfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _temporary_project(root: Path) -> None:
        calib = root / "src/scara/calib"
        calib.mkdir(parents=True)
        geometry_path = calib / "tray_board_geometry.json"
        geometry_path.write_bytes(
            (SRC / "scara/calib/tray_board_geometry.json").read_bytes()
        )
        (calib / "silicon_detection_0818.json").write_bytes(
            (SRC / "scara/calib/silicon_detection_0818.json").read_bytes()
        )
        intrinsics_path = calib / "camera1_intrinsics.json"
        intrinsics_path.write_text(
            json.dumps(
                {
                    "status": "success",
                    "K": _intrinsics().K.tolist(),
                    "distCoeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "image_resolution": {"width": 1280, "height": 720},
                    "global_rms_px": 0.47,
                }
            ),
            encoding="utf-8",
        )
        result_path = (
            root
            / "Trajectory Photos/260814171235/camera1_suction_target.json"
        )
        result_path.parent.mkdir(parents=True)
        result_path.write_text(
            json.dumps(
                {
                    "status": "success",
                    "fit": {
                        "status": "success",
                        "p_C_S_mm": [-2.4, -33.0, 149.5],
                        "target_pixel_distorted_px": [625.5, 218.5],
                    },
                    "locked_inputs": {
                        "camera_intrinsics_sha256": _sha256(intrinsics_path),
                        "tray_geometry_sha256": _sha256(geometry_path),
                    },
                    "camera": {
                        "source_index": 1,
                        "resolution": {"width": 1280, "height": 720},
                    },
                    "coordinate_definition": {
                        "working_plane_z_T_mm": -2.0,
                        "imaging_j3_mm": -27.0046,
                        "rz_mean_deg": 20.8209,
                    },
                }
            ),
            encoding="utf-8",
        )
        layout_path = root / "tools/tray_marker_detector_v2/tray_marker_layout.json"
        layout_path.parent.mkdir(parents=True)
        layout_path.write_bytes(
            (
                PROJECT_ROOT
                / "tools/tray_marker_detector_v2/tray_marker_layout.json"
            ).read_bytes()
        )

    @staticmethod
    def _tray_result_for_table():
        slots = []
        summary = {
            "empty": 34,
            "empty_unread_marker": 0,
            "occupied": 1,
            "warning": 0,
            "stacked": 0,
            "outside_slot": 0,
            "stacked_outside_slot": 0,
            "out_of_view": 0,
            "occluded": 0,
            "unknown": 1,
            "analyzed": 36,
        }
        for row in range(6):
            for column in range(6):
                slot_name = f"P{row}{column}"
                if slot_name == "P00":
                    state = "occupied"
                    offset = (1.25, -0.50)
                    distance = math.hypot(*offset)
                    wafer = SimpleNamespace(
                        found=True,
                        flags=("synthetic_wafer",),
                        confidence=0.91,
                    )
                elif slot_name == "P01":
                    state = "unknown"
                    offset = None
                    distance = None
                    wafer = SimpleNamespace(found=False, flags=(), confidence=0.0)
                else:
                    state = "empty"
                    offset = None
                    distance = None
                    wafer = SimpleNamespace(found=False, flags=(), confidence=0.0)
                slots.append(
                    SimpleNamespace(
                        projection=SimpleNamespace(slot_key=slot_name),
                        decision=SimpleNamespace(
                            state=SimpleNamespace(value=state),
                            reason=f"synthetic {state}",
                            flags=(),
                        ),
                        wafer=wafer,
                        wafer_offset_T_mm=offset,
                        wafer_offset_distance_mm=distance,
                    )
                )
        return SimpleNamespace(
            quality_passed=True,
            failure_reason=None,
            slots=tuple(slots),
            summary=summary,
        )

    def test_dialog_has_36_targets_local_calibration_and_positioning_buttons(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._temporary_project(root)
            camera = FakeCamera(running=True)
            dialog = HandEyeDemoDialog(root, camera)
            try:
                self.assertEqual(dialog.target_combo.count(), 36)
                self.assertEqual(
                    {dialog.target_combo.itemText(i) for i in range(36)},
                    {f"P{row}{column}" for row in range(6) for column in range(6)},
                )
                self.assertIn("只计算", dialog.windowTitle())
                self.assertEqual(dialog.slot_table.rowCount(), 36)
                self.assertEqual(dialog.slot_table.columnCount(), 6)
                self.assertEqual(
                    dialog.silicon_parameter_button.text(), "硅片判定参数"
                )
                self.assertEqual(
                    dialog.silicon_detection_config.source_path.name,
                    "silicon_detection_0818.json",
                )
                self.assertEqual(
                    [
                        dialog.slot_table.horizontalHeaderItem(column).text()
                        for column in range(6)
                    ],
                    [
                        "槽位",
                        "占用",
                        "ΔX_T (mm)",
                        "ΔY_T (mm)",
                        "距离 (mm)",
                        "硅片状态",
                    ],
                )
                self.assertEqual(
                    [dialog.slot_table.item(row, 0).text() for row in range(36)],
                    [f"P{row}{column}" for row in range(6) for column in range(6)],
                )
                safety_labels = [
                    label.text() for label in dialog.findChildren(type(dialog.status))
                ]
                self.assertTrue(any("机械臂不会移动" in text for text in safety_labels))

                self.assertFalse(hasattr(dialog, "validation_button"))
                self.assertEqual(
                    dialog.local_jacobian_button.text(),
                    "local Jacobian标定",
                )
                self.assertEqual(dialog.stage7b_button.text(), "单点有限闭环")
                self.assertEqual(dialog.full_tray_button.text(), "全盘定位")
                controls = None
                root_layout = dialog.content_layout
                for index in range(root_layout.count()):
                    candidate = root_layout.itemAt(index).layout()
                    if candidate is not None and any(
                        candidate.itemAt(item).widget() is dialog.stage7b_button
                        for item in range(candidate.count())
                    ):
                        controls = candidate
                        break
                self.assertIsNotNone(controls)
                control_widgets = [
                    controls.itemAt(index).widget()
                    for index in range(controls.count())
                ]
                self.assertEqual(
                    control_widgets.index(dialog.local_jacobian_button),
                    control_widgets.index(dialog.stage7b_button) - 1,
                )
                self.assertEqual(
                    control_widgets.index(dialog.full_tray_button),
                    control_widgets.index(dialog.stage7b_button) + 1,
                )
                safety_text = "\n".join(safety_labels)
                self.assertIn("确认左侧选取的是P22", safety_text)

                calibrated: list[str] = []
                dialog.local_jacobian_calibration_requested.connect(
                    calibrated.append
                )
                dialog.target_combo.setCurrentText("P31")
                dialog.local_jacobian_button.click()
                self.assertEqual(calibrated, ["P31"])

                emitted: list[bool] = []
                dialog.full_tray_start_requested.connect(
                    lambda: emitted.append(True)
                )
                dialog.target_combo.setCurrentText("P22")
                with patch.object(
                    QMessageBox,
                    "exec",
                    return_value=QMessageBox.StandardButton.Yes,
                ):
                    dialog.full_tray_button.click()
                self.assertEqual(emitted, [True])

                # A stale PASS image/result must be explicitly discarded.
                dialog._last_image = QImage(8, 8, QImage.Format.Format_RGB888)
                dialog._last_evaluation = object()  # type: ignore[assignment]
                dialog._last_evaluation_at = time.monotonic()
                dialog._invalidate_current("synthetic camera timeout")
                self.assertIsNone(dialog._last_image)
                self.assertIsNone(dialog._last_evaluation)
                self.assertIsNone(dialog._last_evaluation_at)
                self.assertIn("判断已失效", dialog.status.text())
                self.assertTrue(
                    all(
                        dialog.slot_table.item(row, 1).text() == "不确定"
                        and dialog.slot_table.item(row, 5).text() == "当前帧不可用"
                        for row in range(36)
                    )
                )
            finally:
                camera.running = False
                dialog.close()
                self.app.processEvents()
                if dialog.monitor.isRunning():
                    self.assertTrue(dialog.monitor.stop())

    def test_hardware_exposure_slider_uses_integer_directshow_stops_and_restores(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._temporary_project(root)
            camera = ExposureCamera(running=True)
            dialog = HandEyeDemoDialog(root, camera)
            try:
                self.assertTrue(dialog.monitor.stop())
                self.assertTrue(dialog.exposure_slider.isEnabled())
                self.assertEqual(
                    dialog.exposure_slider.minimum(), DIRECTSHOW_EXPOSURE_MIN
                )
                self.assertEqual(
                    dialog.exposure_slider.maximum(), DIRECTSHOW_EXPOSURE_MAX
                )
                self.assertEqual(
                    dialog.exposure_slider.value(), DIRECTSHOW_EXPOSURE_DEFAULT
                )
                dialog.exposure_apply_button.click()
                self.assertEqual(
                    camera.exposure_requests[-1], DIRECTSHOW_EXPOSURE_DEFAULT
                )

                dialog.exposure_slider.setValue(-7)
                self.assertEqual(camera.exposure_requests[-1], -7)
                self.assertEqual(dialog.exposure_value_label.text(), "-7")

                dialog.exposure_slider.setValue(-13)
                self.assertEqual(camera.exposure_requests[-1], -13)
                self.assertEqual(dialog.exposure_value_label.text(), "-13")

                dialog.exposure_slider.setValue(-1)
                self.assertEqual(camera.exposure_requests[-1], -1)
                self.assertEqual(dialog.exposure_value_label.text(), "-1")

            finally:
                dialog.close()
                self.app.processEvents()

            self.assertEqual(camera.restore_calls, 1)

    def test_silicon_parameter_picker_hot_switches_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._temporary_project(root)
            selected = root / "lower_side_ratio.json"
            payload = json.loads(
                (
                    root / "src/scara/calib/silicon_detection_0818.json"
                ).read_text(encoding="utf-8-sig")
            )
            payload["profile_name"] = "lower_side_ratio"
            payload["wafer_quality"]["maximum_normal_side_ratio"] = 0.77
            selected.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            camera = FakeCamera(running=True)
            dialog = HandEyeDemoDialog(root, camera)
            try:
                self.assertTrue(dialog.monitor.stop())
                with patch(
                    "scara.ui.handeye_demo_dialog.QFileDialog.getOpenFileName",
                    return_value=(str(selected), "JSON 配置文件 (*.json)"),
                ):
                    dialog.silicon_parameter_button.click()
                self.assertEqual(
                    dialog.silicon_detection_config.profile_name,
                    "lower_side_ratio",
                )
                self.assertAlmostEqual(
                    dialog.monitor._tray_vision_config.wafer_quality.maximum_normal_side_ratio,
                    0.77,
                )
                self.assertIn("lower_side_ratio.json", dialog.silicon_parameter_label.text())
                self.assertIsNone(dialog._last_image)
            finally:
                dialog.close()
                self.app.processEvents()

            reopened = HandEyeDemoDialog(root, camera)
            try:
                self.assertTrue(reopened.monitor.stop())
                self.assertEqual(
                    "lower_side_ratio",
                    reopened.silicon_detection_config.profile_name,
                )
                self.assertAlmostEqual(
                    0.77,
                    reopened.monitor._tray_vision_config.wafer_quality.maximum_normal_side_ratio,
                )
                self.assertIn(
                    "默认：lower_side_ratio.json",
                    reopened.silicon_parameter_label.text(),
                )
            finally:
                reopened.close()
                self.app.processEvents()

    def test_invalid_silicon_parameter_file_keeps_current_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._temporary_project(root)
            selected = root / "partial.json"
            selected.write_text('{"schema_version": 1}', encoding="utf-8")
            camera = FakeCamera(running=True)
            dialog = HandEyeDemoDialog(root, camera)
            try:
                self.assertTrue(dialog.monitor.stop())
                before = dialog.silicon_detection_config
                with (
                    patch(
                        "scara.ui.handeye_demo_dialog.QFileDialog.getOpenFileName",
                        return_value=(str(selected), "JSON 配置文件 (*.json)"),
                    ),
                    patch(
                        "scara.ui.handeye_demo_dialog.QMessageBox.critical"
                    ) as critical,
                ):
                    dialog.silicon_parameter_button.click()
                self.assertIs(dialog.silicon_detection_config, before)
                critical.assert_called_once()
                self.assertFalse(
                    (root / "local_silicon_detection_selection.json").exists()
                )
            finally:
                dialog.close()
                self.app.processEvents()

    def test_hardware_exposure_recovery_button_forces_auto_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._temporary_project(root)
            camera = ExposureCamera(running=True)
            dialog = HandEyeDemoDialog(root, camera)
            try:
                self.assertTrue(dialog.monitor.stop())
                dialog.exposure_slider.setValue(-7)
                dialog.exposure_recovery_button.click()
                self.assertEqual(camera.recovery_calls, 1)
                self.assertEqual(dialog.exposure_slider.value(), -7)
                self.assertEqual(
                    dialog.exposure_value_label.text(),
                    "AUTO（恢复中）",
                )
            finally:
                dialog.close()
                self.app.processEvents()
            self.assertEqual(camera.restore_calls, 0)

    def test_slot_table_maps_occupancy_offsets_and_states(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._temporary_project(root)
            camera = FakeCamera(running=True)
            dialog = HandEyeDemoDialog(root, camera)
            try:
                self.assertTrue(dialog.monitor.stop())
                dialog._update_slot_table(self._tray_result_for_table())
                p00 = dialog._slot_row_by_name["P00"]
                self.assertEqual(dialog.slot_table.item(p00, 1).text(), "是")
                self.assertEqual(dialog.slot_table.item(p00, 2).text(), "+1.25")
                self.assertEqual(dialog.slot_table.item(p00, 3).text(), "-0.50")
                self.assertEqual(
                    dialog.slot_table.item(p00, 4).text(),
                    f"{math.hypot(1.25, -0.50):.2f}",
                )
                self.assertEqual(dialog.slot_table.item(p00, 5).text(), "正常")
                self.assertIn(
                    "synthetic_wafer", dialog.slot_table.item(p00, 5).toolTip()
                )

                p01 = dialog._slot_row_by_name["P01"]
                self.assertEqual(dialog.slot_table.item(p01, 1).text(), "不确定")
                self.assertEqual(dialog.slot_table.item(p01, 2).text(), "—")
                self.assertEqual(dialog.slot_table.item(p01, 5).text(), "证据不足")

                p02 = dialog._slot_row_by_name["P02"]
                self.assertEqual(dialog.slot_table.item(p02, 1).text(), "否")
                self.assertEqual(dialog.slot_table.item(p02, 5).text(), "空槽")
                self.assertIn("占用=1", dialog.tray_summary.text())
                self.assertIn("空槽=34", dialog.tray_summary.text())
                self.assertIn("不确定=1", dialog.tray_summary.text())
            finally:
                camera.running = False
                dialog.close()
                self.app.processEvents()

    def test_camera1_preview_uses_full_width_aspect_ratio_and_zoom_pan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._temporary_project(root)
            camera = FakeCamera(running=True)
            dialog = HandEyeDemoDialog(root, camera)
            try:
                self.assertTrue(dialog.monitor.stop())
                dialog.resize(1200, 960)
                dialog.show()
                image = QImage(1280, 720, QImage.Format.Format_RGB888)
                image.fill(0x305070)
                dialog._last_image = image
                dialog._refresh_preview()
                self.app.processEvents()

                self.assertIsInstance(dialog.preview, CameraImageView)
                self.assertTrue(dialog.preview.has_image)
                self.assertLessEqual(dialog.width() - dialog.preview.width(), 40)
                expected_height = dialog.preview.heightForWidth(
                    dialog.preview.width()
                )
                self.assertLessEqual(abs(dialog.preview.height() - expected_height), 2)
                self.assertEqual(dialog.zoom_label.text(), "100%")
                self.assertTrue(dialog.zoom_in_button.isEnabled())
                self.assertFalse(dialog.zoom_out_button.isEnabled())

                dialog.zoom_in_button.click()
                self.app.processEvents()
                self.assertAlmostEqual(dialog.preview.zoom_factor, 1.25)
                self.assertEqual(dialog.zoom_label.text(), "125%")
                self.assertEqual(
                    dialog.preview.dragMode(),
                    QGraphicsView.DragMode.ScrollHandDrag,
                )
                center_before = dialog.preview.mapToScene(
                    dialog.preview.viewport().rect().center()
                )
                dialog._last_image = image.copy()
                dialog._refresh_preview()
                center_after = dialog.preview.mapToScene(
                    dialog.preview.viewport().rect().center()
                )
                self.assertAlmostEqual(center_before.x(), center_after.x(), places=3)
                self.assertAlmostEqual(center_before.y(), center_after.y(), places=3)
                self.assertAlmostEqual(dialog.preview.zoom_factor, 1.25)

                dialog.zoom_fit_button.click()
                self.app.processEvents()
                self.assertAlmostEqual(dialog.preview.zoom_factor, 1.0)
                self.assertEqual(
                    dialog.preview.dragMode(), QGraphicsView.DragMode.NoDrag
                )
                dialog._invalidate_current("synthetic stale frame")
                self.assertFalse(dialog.preview.has_image)
                self.assertFalse(dialog.zoom_in_button.isEnabled())
            finally:
                camera.running = False
                dialog.close()
                self.app.processEvents()

    def test_slot_table_distinguishes_stacked_outside_and_both(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._temporary_project(root)
            camera = FakeCamera(running=True)
            dialog = HandEyeDemoDialog(root, camera)
            try:
                self.assertTrue(dialog.monitor.stop())
                result = self._tray_result_for_table()
                expected = {
                    "P02": ("stacked", "叠片"),
                    "P03": ("outside_slot", "槽外"),
                    "P04": ("stacked_outside_slot", "叠片且槽外"),
                }
                result.summary["empty"] -= len(expected)
                for slot_name, (state, _text) in expected.items():
                    result.summary[state] = 1
                    analysis = next(
                        slot
                        for slot in result.slots
                        if slot.projection.slot_key == slot_name
                    )
                    analysis.decision.state.value = state
                dialog._update_slot_table(result)
                for slot_name, (_state, text) in expected.items():
                    row = dialog._slot_row_by_name[slot_name]
                    self.assertEqual(dialog.slot_table.item(row, 1).text(), "是")
                    self.assertEqual(dialog.slot_table.item(row, 5).text(), text)
                self.assertIn("占用=4", dialog.tray_summary.text())
                self.assertIn("叠片=1、槽外=1、叠片且槽外=1", dialog.tray_summary.text())
            finally:
                camera.running = False
                dialog.close()
                self.app.processEvents()

    def test_validation_cannot_pass_without_current_robot_domain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._temporary_project(root)
            camera = FakeCamera(running=True)
            dialog = HandEyeDemoDialog(root, camera)
            try:
                self.assertTrue(dialog.monitor.stop())
                dialog.target_combo.setCurrentText("P22")
                dialog._last_evaluation = SimpleNamespace(
                    accepted=True,
                    jacobian_domain_passed=False,
                    correction_available=False,
                    robot_state_age_s=None,
                    jacobian_domain_note="缺少最新机械臂只读状态",
                    correction_xy_mm=None,
                )
                dialog._last_evaluation_at = time.monotonic()
                captured_text: list[str] = []

                def capture_modal(message_box: QMessageBox) -> int:
                    captured_text.append(message_box.text())
                    return 0

                with (
                    patch.object(
                        dialog,
                        "_reload_jacobian",
                        return_value=_valid_jacobian_payload(),
                    ),
                    patch.object(QMessageBox, "exec", new=capture_modal),
                ):
                    dialog._show_jacobian_validation()

                self.assertEqual(len(captured_text), 1)
                self.assertIn("结果：当前条件未全部通过", captured_text[0])
                self.assertIn("目标名匹配：是", captured_text[0])
                self.assertIn("当前机器人状态/Jacobian局部域：FAIL", captured_text[0])
            finally:
                camera.running = False
                dialog.close()
                self.app.processEvents()

    def test_jacobian_loader_rejects_failed_and_missing_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._temporary_project(root)
            suction_path = next(
                (root / "Trajectory Photos").glob("*/camera1_suction_target.json")
            )
            suction = _suction()
            suction = SuctionTargetModel(
                **{
                    **suction.__dict__,
                    "source_path": suction_path,
                    "source_sha256": _sha256(suction_path),
                }
            )
            calib = root / "src/scara/calib"
            jacobian_path = calib / "camera1_xy_image_jacobian.json"
            locked = {
                "camera_intrinsics_sha256": _sha256(
                    calib / "camera1_intrinsics.json"
                ),
                "tray_geometry_sha256": _sha256(
                    calib / "tray_board_geometry.json"
                ),
                "suction_target_sha256": suction.source_sha256,
            }
            payload = {
                "schema_version": 2,
                "status": "success",
                "anchor_target_name": "P22",
                "valid_target_names": ["P22"],
                "camera": {
                    "source_index": 1,
                    "resolution": {"width": 1280, "height": 720},
                },
                "coordinate_definition": {
                    "command_frame": "robot_controller_world_XY",
                    "image_error": (
                        "slot_pixel_distorted - suction_target_pixel_distorted"
                    ),
                    "imaging_j3_mm": suction.imaging_j3_mm,
                    "rz_deg": suction.rz_mean_deg,
                    "offset_extent_mm": 2.0,
                    "anchor_robot_xy_mm": list(ANCHOR_ROBOT_XY_MM),
                },
                "locked_inputs": locked,
                "fit": _valid_fit(),
            }
            jacobian_path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNotNone(load_local_xy_jacobian(root, suction))

            p31_payload = json.loads(json.dumps(payload))
            p31_payload["anchor_target_name"] = "P31"
            p31_payload["valid_target_names"] = ["P31"]
            p31_path = (
                calib
                / "Jacobians"
                / "camera1_xy_image_jacobian_P31.json"
            )
            p31_path.parent.mkdir(parents=True, exist_ok=True)
            p31_path.write_text(json.dumps(p31_payload), encoding="utf-8")
            self.assertIsNotNone(
                load_local_xy_jacobian(root, suction, "P31")
            )
            self.assertIsNone(
                load_local_xy_jacobian(root, suction, "P30")
            )

            old_schema = json.loads(json.dumps(payload))
            old_schema["schema_version"] = 1
            jacobian_path.write_text(json.dumps(old_schema), encoding="utf-8")
            self.assertIsNone(load_local_xy_jacobian(root, suction))

            missing_anchor = json.loads(json.dumps(payload))
            missing_anchor["coordinate_definition"].pop("anchor_robot_xy_mm")
            jacobian_path.write_text(json.dumps(missing_anchor), encoding="utf-8")
            self.assertIsNone(load_local_xy_jacobian(root, suction))

            missing = json.loads(json.dumps(payload))
            missing["fit"]["quality_gates"].pop(
                next(iter(REQUIRED_XY_JACOBIAN_QUALITY_GATES))
            )
            jacobian_path.write_text(json.dumps(missing), encoding="utf-8")
            self.assertIsNone(load_local_xy_jacobian(root, suction))

            failed = json.loads(json.dumps(payload))
            failed_gate = next(iter(REQUIRED_XY_JACOBIAN_QUALITY_GATES))
            failed["fit"]["quality_gates"][failed_gate]["passed"] = False
            jacobian_path.write_text(json.dumps(failed), encoding="utf-8")
            self.assertIsNone(load_local_xy_jacobian(root, suction))

            wrong_source = json.loads(json.dumps(payload))
            wrong_source["camera"]["source_index"] = 2
            jacobian_path.write_text(json.dumps(wrong_source), encoding="utf-8")
            self.assertIsNone(load_local_xy_jacobian(root, suction))

            wrong_j3 = json.loads(json.dumps(payload))
            wrong_j3["coordinate_definition"]["imaging_j3_mm"] += 0.21
            jacobian_path.write_text(json.dumps(wrong_j3), encoding="utf-8")
            self.assertIsNone(load_local_xy_jacobian(root, suction))

            wrong_rz = json.loads(json.dumps(payload))
            wrong_rz["coordinate_definition"]["rz_deg"] += 0.51
            jacobian_path.write_text(json.dumps(wrong_rz), encoding="utf-8")
            self.assertIsNone(load_local_xy_jacobian(root, suction))

    def test_repeated_camera_buffer_is_invalidated_by_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._temporary_project(root)
            camera = FrozenPacketCamera()
            dialog = HandEyeDemoDialog(root, camera)
            started = time.monotonic()
            try:
                deadline = started + 2.0
                while (
                    "判断已失效" not in dialog.status.text()
                    and time.monotonic() < deadline
                ):
                    self.app.processEvents()
                    time.sleep(0.02)
                self.assertIn("判断已失效", dialog.status.text())
                self.assertIn("超过1秒未更新", dialog.status.text())
                self.assertLess(time.monotonic() - started, 2.0)
                self.assertIsNone(dialog._last_image)
                self.assertIsNone(dialog._last_evaluation)
            finally:
                camera.running = False
                dialog.close()
                self.app.processEvents()
                if dialog.monitor.isRunning():
                    self.assertTrue(dialog.monitor.stop())


if __name__ == "__main__":
    unittest.main()
