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
from PyQt6.QtWidgets import QApplication, QMessageBox

from scara.ui.handeye_demo_dialog import HandEyeDemoDialog
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
    def test_read_only_modules_do_not_import_or_call_motion_backends(self) -> None:
        sources = [
            SRC / "scara/ui/handeye_demo_dialog.py",
            SRC / "scara/vision/handeye_interaction.py",
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
                root_layout = dialog.layout()
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
            finally:
                camera.running = False
                dialog.close()
                self.app.processEvents()
                if dialog.monitor.isRunning():
                    self.assertTrue(dialog.monitor.stop())

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
