"""Guided runtime for Task10 / Stage-7A supervised single-step correction.

The imported Task10 file owns only the basic action sequence.  This runtime
reuses Stage3, Stage4, the approved Stage5 Jacobian, the pure Stage7A policy,
and the shared fixed-Rz IK planner.  It never imports or calls the controller;
the generic ActionWorker remains the sole hardware owner.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QWidget

from scara.file_io import atomic_write_text, read_text_snapshot
from scara.pipeline.xy_correction_planner import (
    angular_difference_deg,
    load_stage7a_motion_contract,
    plan_fixed_rz_xy_step,
)
from scara.ui.stage7a_dialog import Stage7AOperatorDialog

from .handeye_interaction import (
    evaluate_handeye_frame,
    load_latest_suction_target,
    load_local_xy_jacobian,
)
from .tray_pose_estimator import (
    TrayBoardPoseEstimator,
    load_camera_intrinsics,
    load_tray_board_geometry,
)
from .tray_pose_tracker import TrayPoseTracker
from .xy_visual_servo import (
    DEFAULT_STAGE7A_CONFIG,
    aggregate_stable_measurements,
    build_stage7a_proposal,
    evaluate_stage7a_response,
)


RESULT_FILENAME = "stage7a_single_step.json"
ANNOTATED_DIRECTORY = "annotated_stage7a"
TARGET_NAME = "P22"
REQUEST_KEY = "stage7a_p22_single_step"
CAMERA_SOURCE = 1
BEFORE_FRAME_COUNT = 5
AFTER_FRAME_COUNT = 5
DOMAIN_MARGIN_MM = 0.20
J3_TOLERANCE_MM = 0.15
CURRENT_RZ_TOLERANCE_DEG = 0.20
TARGET_RZ_TOLERANCE_DEG = 0.15
MAXIMUM_SEQUENTIAL_TRANSIENT_RZ_DEG = 0.30
PRECOMPENSATE_RZ = True
MAXIMUM_STEP_NORM_MM = 0.25
MAXIMUM_STEP_AXIS_MM = 0.25
MAXIMUM_SEQUENTIAL_TRANSIENT_MM = 0.50


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(
        Path(path),
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _finite_list(value: Any, length: int, label: str) -> list[float]:
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}必须包含{length}个有限数值") from exc
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label}必须包含{length}个有限数值")
    return result


def _state_to_photo_age_seconds(
    point: Mapping[str, Any], photo: Mapping[str, Any]
) -> float:
    """Return the recorded-state to completed-snapshot latency.

    ActionWorker writes both wall-clock timestamps.  ``captured_at`` is saved
    after the snapshot succeeds, so this is a conservative upper bound on the
    state/exposure separation.  Missing, reversed, or naive timestamps fail
    closed instead of being reported as a fabricated zero-age sample.
    """

    try:
        recorded = datetime.fromisoformat(str(point.get("recorded_at")))
        captured = datetime.fromisoformat(str(photo.get("captured_at")))
        if recorded.tzinfo is None or captured.tzinfo is None:
            raise ValueError("timestamps must include timezone")
        age = float((captured - recorded).total_seconds())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Task10途径点/照片时间戳无效，无法验证状态新鲜度") from exc
    if not math.isfinite(age) or age < 0.0:
        raise ValueError("Task10照片时间早于途径点状态，拒绝使用该帧")
    return age


def _phase_from_point_name(name: str) -> Optional[str]:
    text = str(name)
    if not text.startswith("TASK10|") or "target=P22" not in text:
        return None
    if "phase=before" in text:
        return "before"
    if "phase=after" in text:
        return "after"
    return None


def _qimage_from_bgr(image_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    return QImage(
        rgb.data,
        width,
        height,
        channels * width,
        QImage.Format.Format_RGB888,
    ).copy()


class Stage7ASingleStepRuntime(QObject):
    """Process Task10 evidence and return one default-deny joint proposal."""

    fatal_error = pyqtSignal(str)

    def __init__(
        self,
        output_dir: Path,
        project_root: Path,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.output_dir = Path(output_dir)
        self.project_root = Path(project_root)
        self.result_path = self.output_dir / RESULT_FILENAME
        self.annotated_dir = self.output_dir / ANNOTATED_DIRECTORY
        self.config = DEFAULT_STAGE7A_CONFIG

        self.contract = load_stage7a_motion_contract(self.project_root)
        self.suction = load_latest_suction_target(self.project_root)
        self.jacobian = load_local_xy_jacobian(self.project_root, self.suction)
        if self.jacobian is None:
            raise ValueError("Task10拒绝启动：当前正式Stage5 Jacobian不可用")
        intrinsics_path = self.project_root / "src/scara/calib/camera1_intrinsics.json"
        geometry_path = self.project_root / "src/scara/calib/tray_board_geometry.json"
        self.intrinsics = load_camera_intrinsics(intrinsics_path)
        self.geometry = load_tray_board_geometry(geometry_path)
        self.estimator = TrayBoardPoseEstimator(self.geometry, self.intrinsics)
        self.tracker = TrayPoseTracker(self.estimator)

        self.dialog = Stage7AOperatorDialog(parent)
        self.dialog.show()
        self._records: list[dict[str, Any]] = []
        self._measurements: dict[str, list[dict[str, Any]]] = {
            "before": [],
            "after": [],
        }
        self._processed_paths: set[Path] = set()
        self._proposal: Optional[dict[str, Any]] = None
        self._plan: Optional[dict[str, Any]] = None
        self._operator_decision = "pending"
        self._motion_request: Optional[dict[str, Any]] = None
        self._worker_motion_audit: Optional[dict[str, Any]] = None
        self._response_validation: Optional[dict[str, Any]] = None
        self._processing_failed = False
        self._fatal_messages: list[str] = []
        self._fatal_emitted = False
        self._last_phase: Optional[str] = None
        self._started_at = _now()

    def _report_fatal(self, message: str) -> None:
        text = str(message) or "Task10运行时发生未知错误"
        self._processing_failed = True
        self._fatal_messages.append(text)
        try:
            self._write_incremental("failure")
        except Exception:
            pass
        if not self._fatal_emitted:
            self._fatal_emitted = True
            self.fatal_error.emit(text)

    def _load_manifest(self) -> dict[str, Any]:
        raw = json.loads(
            read_text_snapshot(self.output_dir / "points.json", encoding="utf-8-sig")
        )
        if not isinstance(raw, dict):
            raise ValueError("points.json顶层必须是对象")
        return raw

    def _photo_context(
        self, path: Path, manifest: Optional[Mapping[str, Any]] = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        data = dict(manifest or self._load_manifest())
        photos = data.get("photos") or []
        points = data.get("points") or []
        photo = next(
            (
                item
                for item in photos
                if isinstance(item, Mapping) and item.get("filename") == path.name
            ),
            None,
        )
        if photo is None:
            raise ValueError(f"points.json找不到照片记录：{path.name}")
        sequence = int(photo.get("point_sequence"))
        point = next(
            (
                item
                for item in points
                if isinstance(item, Mapping) and int(item.get("sequence", -1)) == sequence
            ),
            None,
        )
        if point is None:
            raise ValueError(f"points.json找不到照片{path.name}对应的途径点")
        return dict(photo), dict(point)

    def _point_state(self, point: Mapping[str, Any]) -> tuple[list[float], list[float]]:
        joints = point.get("joints") or {}
        centre = point.get("mechanical_center") or {}
        return (
            _finite_list(
                [
                    joints.get("J1_deg"),
                    joints.get("J2_deg"),
                    joints.get("J3_mm"),
                    joints.get("J4_deg"),
                ],
                4,
                "途径点关节",
            ),
            _finite_list(
                [
                    centre.get("x_mm"),
                    centre.get("y_mm"),
                    centre.get("z_mm"),
                    centre.get("Rx_deg"),
                    centre.get("Ry_deg"),
                    centre.get("Rz_deg"),
                ],
                6,
                "途径点位姿",
            ),
        )

    def _process_photo(self, path: Path, *, update_ui: bool = True) -> None:
        resolved = Path(path).resolve()
        if resolved in self._processed_paths:
            return
        if resolved.parent != self.output_dir.resolve():
            raise ValueError(f"Task10照片不在当前输出目录：{resolved}")
        photo, point = self._photo_context(resolved)
        if int(photo.get("source", -1)) != CAMERA_SOURCE:
            raise ValueError(f"Task10只允许相机源1：{resolved.name}")
        phase = _phase_from_point_name(str(point.get("name") or ""))
        if phase is None:
            raise ValueError(f"Task10途径点命名不符合约定：{point.get('name')}")
        image = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"无法读取Task10照片：{resolved}")
        if (image.shape[1], image.shape[0]) != self.suction.resolution:
            raise ValueError(
                f"Task10照片分辨率{image.shape[1]}x{image.shape[0]}与标定"
                f"{self.suction.resolution[0]}x{self.suction.resolution[1]}不一致"
            )
        if self._last_phase != phase:
            self.tracker.reset()
            self._last_phase = phase

        joints, pose = self._point_state(point)
        robot_state_age_s = _state_to_photo_age_seconds(point, photo)
        robot_state = {
            "joints": joints,
            "pose": pose,
            "captured_monotonic_s": time.monotonic(),
        }
        tracked = self.tracker.update(image)
        evaluation = evaluate_handeye_frame(
            image,
            tracked,
            TARGET_NAME,
            self.geometry,
            self.intrinsics,
            self.suction,
            self.jacobian,
            robot_state,
            alignment_threshold_px=1.0,
            maximum_robot_state_age_s=self.config.maximum_robot_state_age_s,
        )
        cv2.putText(
            evaluation.annotated_bgr,
            f"STAGE7A {phase.upper()} | {resolved.name}",
            (18, evaluation.annotated_bgr.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = self.annotated_dir / resolved.name
        if not cv2.imwrite(str(annotated_path), evaluation.annotated_bgr):
            raise ValueError(f"保存Task10标注图失败：{annotated_path}")

        raw = tracked.raw
        record = {
            "phase": phase,
            "filename": resolved.name,
            "annotated_filename": str(
                Path(ANNOTATED_DIRECTORY) / annotated_path.name
            ).replace("\\", "/"),
            "point_sequence": int(point.get("sequence")),
            "point_name": str(point.get("name")),
            "captured_at": photo.get("captured_at"),
            "processed_at": _now(),
            "accepted": bool(evaluation.accepted),
            "reason": str(evaluation.reason),
            "visible_marker_count": int(evaluation.visible_marker_count),
            "used_marker_count": int(evaluation.used_marker_count),
            "visible_marker_ids": list(raw.visible_marker_ids),
            "used_marker_ids": list(raw.used_marker_ids),
            "ransac_inlier_corner_count": int(raw.ransac_inlier_corner_count),
            "reprojection_rms_px": evaluation.reprojection_rms_px,
            "translation_jump_mm": tracked.translation_jump_mm,
            "rotation_jump_deg": tracked.rotation_jump_deg,
            "slot_pixel_px": evaluation.slot_pixel_px,
            "suction_target_pixel_px": evaluation.suction_target_pixel_px,
            "image_error_px": evaluation.image_error_px,
            "image_error_norm_px": evaluation.image_error_norm_px,
            "per_frame_full_correction_xy_mm": evaluation.correction_xy_mm,
            "jacobian_domain_passed": bool(evaluation.jacobian_domain_passed),
            "jacobian_domain_note": evaluation.jacobian_domain_note,
            "robot_joints": joints,
            "robot_pose": pose,
            "controller_safety": dict(point.get("controller_safety") or {}),
            "robot_state_age_s": robot_state_age_s,
        }
        measurement = {
            "measurement_id": f"{phase}:{resolved.name}:{point.get('sequence')}",
            "target_name": TARGET_NAME,
            "accepted": bool(evaluation.accepted),
            "image_error_px": evaluation.image_error_px,
            "current_robot_xy_mm": [pose[0], pose[1]],
            "current_joints": joints,
            "robot_state_age_s": robot_state_age_s,
            "jacobian_domain_passed": bool(evaluation.jacobian_domain_passed),
            "reason": str(evaluation.reason),
        }
        self._records.append(record)
        self._measurements[phase].append(measurement)
        self._processed_paths.add(resolved)
        if update_ui:
            self.dialog.add_frame(record, _qimage_from_bgr(evaluation.annotated_bgr))
        self._write_incremental("collecting")

    def _synchronise_saved_photos(self, phase: str) -> None:
        manifest = self._load_manifest()
        points = {
            int(item.get("sequence")): item
            for item in (manifest.get("points") or [])
            if isinstance(item, Mapping)
        }
        for photo in manifest.get("photos") or []:
            if not isinstance(photo, Mapping) or int(photo.get("source", -1)) != CAMERA_SOURCE:
                continue
            point = points.get(int(photo.get("point_sequence", -1)))
            if point is None or _phase_from_point_name(str(point.get("name") or "")) != phase:
                continue
            self._process_photo(self.output_dir / str(photo.get("filename")))

    def _external_gates(
        self, request: Mapping[str, Any], *, operator_consent: bool
    ) -> dict[str, Any]:
        supplied = request.get("external_safety_gates") or {}
        gates = {
            name: supplied.get(name)
            for name in (
                "controller_connected",
                "controller_enabled",
                "alarm_clear",
                "estop_clear",
                "soft_estop_clear",
                "controller_idle",
            )
        }
        try:
            aggregate = aggregate_stable_measurements(
                self._measurements["before"], self.config
            )
            measured_xy = _finite_list(
                aggregate.get("current_robot_xy_mm"),
                2,
                "视觉窗口机械臂XY",
            )
            measured_joints = _finite_list(
                aggregate.get("current_joints"),
                4,
                "视觉窗口机械臂关节",
            )
            state = request.get("controller_state") or {}
            requested_pose = _finite_list(
                state.get("pose"), 6, "运动请求机械臂位姿"
            )
            requested_joints = _finite_list(
                state.get("joints"), 4, "运动请求机械臂关节"
            )
            xy_difference = math.hypot(
                requested_pose[0] - measured_xy[0],
                requested_pose[1] - measured_xy[1],
            )
            joint_differences = [
                angular_difference_deg(requested_joints[0], measured_joints[0]),
                angular_difference_deg(requested_joints[1], measured_joints[1]),
                abs(requested_joints[2] - measured_joints[2]),
                angular_difference_deg(requested_joints[3], measured_joints[3]),
            ]
            maximum_joint_difference = max(joint_differences)
            gates["measurement_matches_request_state"] = {
                "passed": bool(
                    xy_difference <= 0.05 + 1e-12
                    and maximum_joint_difference <= 0.05 + 1e-12
                ),
                "actual": {
                    "xy_difference_mm": float(xy_difference),
                    "joint_differences_deg_or_mm": [
                        float(value) for value in joint_differences
                    ],
                },
                "limit": {
                    "maximum_xy_difference_mm": 0.05,
                    "maximum_joint_difference_deg_or_mm": 0.05,
                },
                "note": (
                    "the controller state used for planning must still match "
                    "the state paired with the five vision frames"
                ),
            }
        except Exception as exc:  # fail closed; the popup still explains why
            gates["measurement_matches_request_state"] = {
                "passed": False,
                "actual": None,
                "limit": {
                    "maximum_xy_difference_mm": 0.05,
                    "maximum_joint_difference_deg_or_mm": 0.05,
                },
                "note": f"cannot compare visual and request states: {exc}",
            }
        gates["camera_fresh"] = (
            len(self._measurements["before"]) == BEFORE_FRAME_COUNT
            and len({row["measurement_id"] for row in self._measurements["before"]})
            == BEFORE_FRAME_COUNT
        )
        gates["operator_consent"] = bool(operator_consent)
        return gates

    def _assert_locked_inputs_unchanged(self) -> None:
        """Revalidate the formal Stage5 file and every upstream hash.

        Task10 normally lasts only seconds, but an external calibration edit
        during the popup must invalidate the displayed proposal instead of
        silently mixing old in-memory geometry with a new approved file.
        """

        current = load_stage7a_motion_contract(self.project_root)
        if str(current.get("stage5_sha256") or "").upper() != str(
            self.contract["stage5_sha256"]
        ).upper():
            raise ValueError("Task10运行期间正式Stage5文件发生变化")

    def _build_plan(
        self, request: Mapping[str, Any], proposal: Mapping[str, Any]
    ) -> dict[str, Any]:
        state = request.get("controller_state") or {}
        command = (proposal.get("calculation") or {}).get(
            "commanded_correction_xy_mm"
        )
        return plan_fixed_rz_xy_step(
            _finite_list(state.get("joints"), 4, "请求时关节"),
            _finite_list(state.get("pose"), 6, "请求时位姿"),
            _finite_list(command, 2, "Stage7A命令"),
            anchor_robot_xy_mm=self.contract["anchor_robot_xy_mm"],
            local_extent_mm=self.contract["local_extent_mm"],
            domain_margin_mm=DOMAIN_MARGIN_MM,
            required_j3_mm=self.contract["required_j3_mm"],
            j3_tolerance_mm=J3_TOLERANCE_MM,
            required_rz_deg=self.contract["required_rz_deg"],
            rz_tolerance_deg=CURRENT_RZ_TOLERANCE_DEG,
            max_xy_step_norm_mm=MAXIMUM_STEP_NORM_MM,
            max_xy_axis_mm=MAXIMUM_STEP_AXIS_MM,
            max_sequential_transient_xy_mm=MAXIMUM_SEQUENTIAL_TRANSIENT_MM,
            target_rz_tolerance_deg=TARGET_RZ_TOLERANCE_DEG,
            max_sequential_transient_rz_deg=(
                MAXIMUM_SEQUENTIAL_TRANSIENT_RZ_DEG
            ),
            precompensate_rz=PRECOMPENSATE_RZ,
            allow_rejected_audit=True,
        )

    @pyqtSlot(str)
    def on_photo_saved(self, path_text: str) -> None:
        try:
            self._process_photo(Path(path_text))
        except Exception as exc:  # noqa: BLE001 - fail closed via Qt signal
            self._report_fatal(f"Task10照片处理失败：{exc}")

    def on_runtime_move_joints_requested(
        self, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Called on the Qt thread; display one default-deny motion proposal."""

        try:
            if self._processing_failed:
                raise ValueError("Task10此前照片处理已经失败")
            if str(request.get("request_key")) != REQUEST_KEY:
                raise ValueError("未知的Task10运行时运动请求")
            if str(request.get("target_name")) != TARGET_NAME:
                raise ValueError("Task10运行时运动请求目标不是P22")
            if str(request.get("calibration_sha256") or "").upper() != str(
                self.contract["stage5_sha256"]
            ).upper():
                raise ValueError("运动请求绑定的Stage5 hash与当前正式文件不一致")
            self._assert_locked_inputs_unchanged()
            self._synchronise_saved_photos("before")
            if len(self._measurements["before"]) != BEFORE_FRAME_COUNT:
                raise ValueError(
                    f"修正前证据不是{BEFORE_FRAME_COUNT}帧："
                    f"{len(self._measurements['before'])}"
                )
            display_proposal = build_stage7a_proposal(
                self._measurements["before"],
                self.jacobian,
                external_safety_gates=self._external_gates(
                    request, operator_consent=False
                ),
                config=self.config,
            )
            plan: Optional[dict[str, Any]] = None
            if display_proposal.get("ready_for_operator_confirmation") and display_proposal.get(
                "motion_required"
            ):
                plan = self._build_plan(request, display_proposal)
            proposal_can_be_approved = bool(
                display_proposal.get("ready_for_operator_confirmation")
                and display_proposal.get("motion_required")
                and plan is not None
                and (plan.get("audit") or {}).get("passed") is True
            )
            approved = self.dialog.request_decision(display_proposal, plan)
            self._motion_request = dict(request)
            if not approved:
                if (
                    display_proposal.get("ready_for_operator_confirmation")
                    and not display_proposal.get("motion_required")
                ):
                    self._operator_decision = "no_motion_required"
                    incremental_status = "no_motion_required"
                    decline_reason = (
                        "visual error is already below the Stage7A motion threshold "
                        "or the effective command is below the minimum step"
                    )
                elif not proposal_can_be_approved:
                    self._operator_decision = "safety_rejected"
                    incremental_status = "safety_rejected"
                    decline_reason = "one or more Stage7A safety gates rejected motion"
                else:
                    self._operator_decision = "operator_declined"
                    incremental_status = "operator_declined"
                    decline_reason = "operator declined the supervised XY motion"
                self._proposal = display_proposal
                self._plan = plan
                self._write_incremental(incremental_status)
                return {
                    "request_id": str(request.get("request_id")),
                    "decision": "decline",
                    "reason": decline_reason,
                }

            self._operator_decision = "approved"
            # The operator may spend several seconds reviewing the proposal.
            # Re-read all formal calibration hashes immediately after consent.
            self._assert_locked_inputs_unchanged()
            authorised = build_stage7a_proposal(
                self._measurements["before"],
                self.jacobian,
                external_safety_gates=self._external_gates(
                    request, operator_consent=True
                ),
                config=self.config,
            )
            if not authorised.get("motion_authorized"):
                raise ValueError(
                    "操作员确认后Stage7A仍未授权运动，安全门失败："
                    + ", ".join(authorised.get("failure_reasons") or [])
                )
            plan = self._build_plan(request, authorised)
            if (plan.get("audit") or {}).get("passed") is not True:
                raise ValueError("操作员确认后的Stage7A规划仍有安全门未通过")
            self._proposal = authorised
            self._plan = plan
            self._write_incremental("motion_approved_pending_worker_preflight")
            return {
                "request_id": str(request.get("request_id")),
                "decision": "approve",
                "calibration_sha256": self.contract["stage5_sha256"],
                "proposal": authorised,
                "proposal_id": authorised["proposal_id"],
                "target_joints": plan["target_joints"],
                "planner": plan,
            }
        except Exception as exc:  # noqa: BLE001
            self._report_fatal(f"Task10运动候选生成失败：{exc}")
            return {
                "request_id": str(request.get("request_id")),
                "decision": "abort",
                "reason": str(exc),
            }

    def _base_report(self, status: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "stage": "7A_supervised_single_step",
            "status": str(status),
            "started_at": self._started_at,
            "updated_at": _now(),
            "target_name": TARGET_NAME,
            "source_run_folder": str(self.output_dir),
            "camera": {
                "source_index": CAMERA_SOURCE,
                "resolution": {
                    "width": self.suction.resolution[0],
                    "height": self.suction.resolution[1],
                },
            },
            "locked_inputs": dict(self.jacobian.get("locked_inputs") or {}),
            "stage5_calibration": {
                "path": self.contract["stage5_path"],
                "sha256": self.contract["stage5_sha256"],
            },
            "configuration": {
                "before_frames": BEFORE_FRAME_COUNT,
                "after_frames": AFTER_FRAME_COUNT,
                "gain": self.config.gain,
                "maximum_step_norm_mm": MAXIMUM_STEP_NORM_MM,
                "maximum_step_axis_mm": MAXIMUM_STEP_AXIS_MM,
                "domain_margin_mm": DOMAIN_MARGIN_MM,
                "maximum_sequential_transient_mm": MAXIMUM_SEQUENTIAL_TRANSIENT_MM,
                "j3_tolerance_mm": J3_TOLERANCE_MM,
                "current_rz_tolerance_deg": CURRENT_RZ_TOLERANCE_DEG,
                "target_rz_tolerance_deg": TARGET_RZ_TOLERANCE_DEG,
                "maximum_sequential_transient_rz_deg": (
                    MAXIMUM_SEQUENTIAL_TRANSIENT_RZ_DEG
                ),
                "precompensate_rz": PRECOMPENSATE_RZ,
                "no_z_motion": True,
                "no_do_or_vacuum": True,
                "automatic_repeat": False,
            },
            "frame_records": list(self._records),
            "proposal": self._proposal,
            "planner": self._plan,
            "operator_decision": self._operator_decision,
            "motion_request": self._motion_request,
            "worker_motion_audit": self._worker_motion_audit,
            "response_validation": self._response_validation,
            "runtime_processing": {
                "failed": bool(self._processing_failed),
                "fatal_messages": list(self._fatal_messages),
            },
        }

    def _write_incremental(self, status: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.result_path, self._base_report(status))

    def _matching_runtime_move(
        self, manifest: Mapping[str, Any]
    ) -> Optional[dict[str, Any]]:
        rows = manifest.get("runtime_moves") or manifest.get("interactive_moves") or []
        matches = [
            dict(row)
            for row in rows
            if isinstance(row, Mapping)
            and str(row.get("request_key")) == REQUEST_KEY
        ]
        return matches[-1] if matches else None

    def _motion_executed(self, manifest: Mapping[str, Any]) -> bool:
        row = self._matching_runtime_move(manifest)
        return bool(
            row is not None
            and (
                row.get("executed") is True
                or row.get("physical_motion_started") is True
                or row.get("status") in {"completed", "motion_completed"}
            )
        )

    def _post_controller_safety_gate(self) -> dict[str, Any]:
        after_records = [
            record for record in self._records if record.get("phase") == "after"
        ]
        rows: list[dict[str, Any]] = []
        for record in after_records:
            safety = record.get("controller_safety") or {}
            passed = bool(
                safety.get("connected") is True
                and safety.get("effectively_enabled") is True
                and int(safety.get("warn", -1)) == 0
                and safety.get("need_clear") is False
                and safety.get("estop") is False
                and safety.get("soft_estop") is False
            )
            rows.append(
                {
                    "filename": record.get("filename"),
                    "passed": passed,
                    "controller_safety": dict(safety),
                }
            )
        return {
            "name": "post_controller_safety",
            "passed": bool(
                len(rows) == AFTER_FRAME_COUNT
                and all(row["passed"] for row in rows)
            ),
            "actual": rows,
            "limit": (
                f"exactly {AFTER_FRAME_COUNT} post states; connected/enabled; "
                "warn=0; no clear request or emergency stop"
            ),
            "note": "a response cannot pass after a controller safety fault",
        }

    def _enrich_manifest(
        self, manifest: dict[str, Any], final_report: Mapping[str, Any]
    ) -> None:
        by_sequence = {
            int(record["point_sequence"]): record for record in self._records
        }
        for point in manifest.get("points") or []:
            if not isinstance(point, dict):
                continue
            record = by_sequence.get(int(point.get("sequence", -1)))
            if record is not None:
                point["stage7a_evaluation"] = dict(record)
        manifest["stage7a"] = {
            "status": final_report.get("status"),
            "result_file": RESULT_FILENAME,
            "operator_decision": self._operator_decision,
            "proposal_id": (self._proposal or {}).get("proposal_id"),
            "commanded_correction_xy_mm": (
                (self._proposal or {}).get("calculation") or {}
            ).get("commanded_correction_xy_mm"),
            "response_passed": (
                None
                if self._response_validation is None
                else self._response_validation.get("passed")
            ),
        }

    def on_task_finished(self, ok: bool, message: str, output_dir_text: str) -> None:
        try:
            if Path(output_dir_text).resolve() != self.output_dir.resolve():
                raise ValueError("Task10结束目录与运行时目录不一致")
            manifest = self._load_manifest()
            self._synchronise_saved_photos("after")
            self._worker_motion_audit = self._matching_runtime_move(manifest)
            motion_executed = self._motion_executed(manifest)
            if (
                self._proposal is not None
                and motion_executed
                and len(self._measurements["after"]) == AFTER_FRAME_COUNT
            ):
                self._response_validation = evaluate_stage7a_response(
                    self._proposal,
                    self._measurements["after"],
                    self.jacobian,
                    config=self.config,
                )
                post_safety_gate = self._post_controller_safety_gate()
                self._response_validation.setdefault("quality_gates", {})[
                    "post_controller_safety"
                ] = post_safety_gate
                if not post_safety_gate["passed"]:
                    self._response_validation["passed"] = False
                    failures = self._response_validation.setdefault(
                        "failure_reasons", []
                    )
                    if "post_controller_safety" not in failures:
                        failures.append("post_controller_safety")
            elif self._operator_decision in {
                "operator_declined",
                "safety_rejected",
                "no_motion_required",
            }:
                self._response_validation = {
                    "passed": None,
                    "status": "not_executed",
                    "reason": (
                        f"{self._operator_decision}; post frames are audit-only"
                    ),
                }

            if self._processing_failed or not ok:
                status = "failure"
            elif self._operator_decision == "operator_declined":
                status = "operator_declined"
            elif self._operator_decision == "safety_rejected":
                status = "safety_rejected"
            elif self._operator_decision == "no_motion_required":
                status = "no_motion_required"
            elif not motion_executed:
                status = "failure"
                self._fatal_messages.append("worker did not record an executed Stage7A move")
            elif not isinstance(self._response_validation, Mapping) or not bool(
                self._response_validation.get("passed")
            ):
                status = "response_rejected"
            else:
                status = "success"
            report = self._base_report(status)
            report["finished_at"] = _now()
            report["action_result"] = {
                "ok": bool(ok),
                "message": str(message),
                "motion_executed": bool(motion_executed),
            }
            _atomic_json(self.result_path, report)
            self._enrich_manifest(manifest, report)
            _atomic_json(self.output_dir / "points.json", manifest)
            if status == "success":
                final_text = "Stage7A完成：一次XY修正已执行，复测响应门全部通过。"
            elif status == "operator_declined":
                final_text = "Stage7A完成：人员选择不运动；修正前后审计数据均已保存。"
            elif status == "safety_rejected":
                final_text = "Stage7A安全拒绝：未运动；失败门和前后审计数据均已保存。"
            elif status == "no_motion_required":
                final_text = "Stage7A未下发运动：视觉误差已低于本阶段动作阈值或步长过小。"
            else:
                final_text = f"Stage7A未通过：{status}。不会执行第二次自动修正。"
            self.dialog.set_final_text(final_text)
        except Exception as exc:  # noqa: BLE001
            self._report_fatal(f"Task10结束处理失败：{exc}")
            raise


def create_stage7a_runtime(
    output_dir: Path,
    project_root: Path,
    parent: Optional[QWidget] = None,
) -> Stage7ASingleStepRuntime:
    return Stage7ASingleStepRuntime(output_dir, project_root, parent)


__all__ = [
    "AFTER_FRAME_COUNT",
    "BEFORE_FRAME_COUNT",
    "REQUEST_KEY",
    "RESULT_FILENAME",
    "Stage7ASingleStepRuntime",
    "create_stage7a_runtime",
]
