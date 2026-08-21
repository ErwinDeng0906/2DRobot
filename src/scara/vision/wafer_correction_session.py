"""Fail-closed XY-only session for the nearest outside wafer to P00.

The dialog supplies five fresh Stage-3/wafer-analysis frames per request.  This
module freezes one non-stacked outside-wafer centre in Tray coordinates,
registers the current Tray into robot world coordinates, and returns bounded
fixed-J3/fixed-Rz joint proposals.  ``ActionWorker`` remains the only hardware
owner and independently re-audits every proposal immediately before motion.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from scara.file_io import atomic_write_text
from scara.pipeline.kinematics import rz_of
from scara.pipeline.xy_correction_planner import plan_fixed_rz_xy_step

from .handeye_interaction import load_latest_suction_target, sha256_file
from .moved_tray_servo import registration_transform_W_T
from .runtime_tray_registration import (
    build_runtime_tray_registration,
    load_planar_handeye,
)
from .tray_pose_estimator import load_tray_board_geometry
from .wafer_correction_target import aggregate_nearest_outside_wafer


WAFER_CORRECTION_RUNTIME_REQUEST_KEY = "nearest_p00_outside_wafer_xy"
WAFER_CORRECTION_TARGET_NAME = "nearest_outside_wafer"
RESULT_FILENAME = "wafer_correction.json"
REGISTRATION_FILENAME = "wafer_correction_runtime_registration.json"
RUNTIME_ACTION_SLOTS = 64
MAXIMUM_NOMINAL_STEP_MM = 9.99
ARRIVAL_TOLERANCE_MM = 0.20
J3_TOLERANCE_MM = 0.20
RZ_TOLERANCE_DEG = 0.30


def _vector(value: Any, length: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != int(length) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label}必须包含{length}个有限数值")
    return result


def _angle_delta_deg(left: float, right: float) -> float:
    return (float(left) - float(right) + 180.0) % 360.0 - 180.0


def _json_safe_evaluation(
    value: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        encoded = json.dumps(
            dict(value), ensure_ascii=False, allow_nan=False
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError, OverflowError):
        return {"diagnostic_omitted": "contains_nonfinite_or_nonjson_value"}
    return decoded if isinstance(decoded, dict) else None


def _previous_registration_drifted(project_root: Path) -> bool:
    reports = sorted(
        (Path(project_root) / "Trajectory Photos").glob(
            "*/full_tray_positioning.json"
        ),
        key=lambda path: path.parent.name,
        reverse=True,
    )
    for path in reports:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if (
            payload.get("stage")
            != "moved_tray_runtime_registration_and_metric_xy_positioning"
        ):
            continue
        return "登记发生漂移" in str(
            payload.get("result_message") or ""
        )
    return False


class WaferCorrectionSession:
    """Freeze, register, approach and independently hold one wafer centre."""

    def __init__(
        self,
        project_root: Path,
        output_dir: Path,
        initial_robot_state: Mapping[str, Any],
        *,
        camera_reconnected: bool = False,
    ) -> None:
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)
        self.initial_joints = _vector(
            initial_robot_state.get("joints"), 4, "initial joints"
        )
        self.initial_pose = _vector(
            initial_robot_state.get("pose"), 6, "initial pose"
        )
        captured = float(
            initial_robot_state.get("captured_monotonic_s", math.nan)
        )
        import time

        age = time.monotonic() - captured
        if not math.isfinite(age) or age < 0.0 or age > 1.0:
            raise RuntimeError("启动硅片纠错时机械臂状态必须是1秒内的新鲜读数")

        self.suction = load_latest_suction_target(self.project_root)
        self.calibration = load_planar_handeye(
            self.project_root, self.suction
        )
        self.calibration_hash = str(
            self.calibration["_source_sha256"]
        ).upper()
        self.geometry_path = (
            self.project_root / "src/scara/calib/tray_board_geometry.json"
        )
        self.geometry = load_tray_board_geometry(self.geometry_path)
        self.geometry_hash = sha256_file(self.geometry_path)
        self.required_j3_mm = float(self.suction.imaging_j3_mm)
        self.required_rz_deg = float(self.suction.rz_mean_deg)
        if (
            abs(float(self.initial_joints[2]) - self.required_j3_mm)
            > J3_TOLERANCE_MM
        ):
            raise RuntimeError(
                "硅片纠错只保持当前标定成像高度；"
                f"当前J3={float(self.initial_joints[2]):.3f}mm，"
                f"要求{self.required_j3_mm:.3f}±0.20mm"
            )
        if (
            abs(
                _angle_delta_deg(
                    float(self.initial_pose[5]), self.required_rz_deg
                )
            )
            > RZ_TOLERANCE_DEG
        ):
            raise RuntimeError(
                "硅片纠错要求保持标定绝对Rz；"
                f"当前Rz={float(self.initial_pose[5]):.3f}°，"
                f"要求{self.required_rz_deg:.3f}±0.30°"
            )

        frame = self.geometry.get("tray_frame") or {}
        old_origin = _vector(
            frame.get("origin_mechanical_xy_mm"), 2, "Stage2 origin"
        )
        old_rotation = np.asarray(
            frame.get("rotation_mechanical_from_tray"), dtype=np.float64
        )
        p22_T = _vector(
            (self.geometry.get("slots") or {}).get("P22"), 3, "P22"
        )
        if old_rotation.shape != (3, 3) or not np.all(
            np.isfinite(old_rotation)
        ):
            raise ValueError("Stage2 rotation必须是有限3x3矩阵")
        self.static_audit_anchor_xy = (
            old_origin + old_rotation[:2, :2] @ p22_T[:2]
        ).astype(float).tolist()

        self.started_at = datetime.now().astimezone().isoformat(
            timespec="milliseconds"
        )
        self.status = "waiting_for_runtime_registration"
        self.phase = "registration"
        self.result_message = ""
        self.registration: dict[str, Any] | None = None
        self.locked_target: dict[str, Any] | None = None
        self.target_world_xy_mm: list[float] | None = None
        self.iterations: list[dict[str, Any]] = []
        self.final_hold: dict[str, Any] | None = None
        self.evidence_images: list[dict[str, Any]] = []
        self.photo_sequence = 0
        self.camera_reconnected = bool(camera_reconnected)
        self.installation_check_expired = bool(
            self.calibration.get("_installation_check_expired") is True
        )
        self.previous_registration_drifted = _previous_registration_drifted(
            self.project_root
        )

    @property
    def report_path(self) -> Path:
        return self.output_dir / RESULT_FILENAME

    @property
    def registration_path(self) -> Path:
        return self.output_dir / REGISTRATION_FILENAME

    def action_task(self) -> dict[str, Any]:
        actions: list[dict[str, Any]] = [
            {
                "type": "assert_joints",
                "name": "槽外硅片纠错启动状态绑定",
                "joints": self.initial_joints.astype(float).tolist(),
                "tolerance": 0.20,
            }
        ]
        for index in range(1, RUNTIME_ACTION_SLOTS + 1):
            actions.extend(
                [
                    {"type": "wait", "seconds": 0.30},
                    {
                        "type": "runtime_move_joints",
                        "name": f"槽外硅片中心XY纠错窗口 {index:03d}",
                        "request_key": WAFER_CORRECTION_RUNTIME_REQUEST_KEY,
                        "target_name": WAFER_CORRECTION_TARGET_NAME,
                        "calibration_sha256": self.calibration_hash,
                        "anchor_robot_xy_mm": list(
                            self.static_audit_anchor_xy
                        ),
                        "local_extent_mm": 130.0,
                        "domain_margin_mm": 5.0,
                        "required_j3_mm": self.required_j3_mm,
                        "required_rz_deg": self.required_rz_deg,
                        "max_xy_step_norm_mm": 10.0,
                        "max_xy_axis_mm": 10.0,
                        "j3_tolerance_mm": 0.20,
                        "rz_tolerance_deg": 0.30,
                        "target_rz_tolerance_deg": 0.15,
                        "max_sequential_transient_rz_deg": 15.0,
                        "precompensate_rz": True,
                        "enforce_sequential_intermediate_domain": False,
                        "max_state_drift_xy_mm": 0.20,
                        "max_state_drift_joint": 0.20,
                        "max_sequential_transient_xy_mm": 130.0,
                        "move_tolerance": 0.01,
                        "proposal_max_age_s": 8.0,
                        "fk_pose_xy_tolerance_mm": 0.20,
                    },
                ]
            )
        return {
            "api_version": 1,
            "name": "离P00最近的槽外硅片中心XY纠错",
            "description": (
                "仅接受非叠片槽外硅片；完整轮廓中心优先，明确OUT时允许"
                "拟合四边形对角线交点回退；取5张新鲜帧中心的算术平均值，"
                "登记本次Tray→World后按不超过9.99mm的固定J3/Rz航点移动。"
                "最终再用独立5帧确认；不执行Z、DO或真空。"
            ),
            "camera_model": {
                "offset_mm": 0.0,
                "angle_reference": "world_negative_y",
                "positive_rotation": "counter_clockwise_from_above",
            },
            "actions": actions,
        }

    def _save_images(
        self, samples: Sequence[Mapping[str, Any]], stage: str
    ) -> list[str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        names: list[str] = []
        for sample in samples:
            image = sample.get("annotated_bgr")
            if image is None:
                continue
            self.photo_sequence += 1
            name = f"wafer_{self.photo_sequence:03d}.jpg"
            if not cv2.imwrite(str(self.output_dir / name), image):
                raise RuntimeError(f"无法保存硅片纠错证据图片{name}")
            names.append(name)
            self.evidence_images.append(
                {
                    "filename": name,
                    "stage": str(stage),
                    "measurement_id": str(
                        sample.get("measurement_id") or ""
                    ),
                }
            )
        return names

    def _build_registration(
        self,
        request: Mapping[str, Any],
        samples: Sequence[Mapping[str, Any]],
        method: str,
    ) -> dict[str, Any]:
        return build_runtime_tray_registration(
            samples,
            self.calibration,
            self.suction,
            self.geometry,
            requested_monotonic_s=float(
                request.get("requested_monotonic_s")
            ),
            method=method,
            camera_reconnected=(
                self.camera_reconnected and self.phase == "registration"
            ),
            installation_check_expired=(
                self.installation_check_expired
                and self.phase == "registration"
            ),
            previous_registration_drifted=(
                self.previous_registration_drifted
                and self.phase == "registration"
            ),
        )

    def _registration_drift(
        self, current: Mapping[str, Any]
    ) -> tuple[float, float]:
        if self.registration is None:
            return math.inf, math.inf
        current_origin = _vector(
            current.get("origin_world_xy_mm"), 2, "current origin"
        )
        locked_origin = _vector(
            self.registration.get("origin_world_xy_mm"), 2, "locked origin"
        )
        translation = float(np.linalg.norm(current_origin - locked_origin))
        yaw = abs(
            _angle_delta_deg(
                float(current.get("yaw_world_from_tray_deg")),
                float(self.registration.get("yaw_world_from_tray_deg")),
            )
        )
        return translation, yaw

    def _aggregate_target(
        self, samples: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        candidate_lists = [
            list(sample.get("outside_wafer_candidates") or [])
            for sample in samples
        ]
        return aggregate_nearest_outside_wafer(
            candidate_lists,
            self.geometry,
            required_frame_count=5,
        )

    def _target_world_xy(
        self,
        target: Mapping[str, Any],
        registration: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        selected_registration = (
            self.registration if registration is None else registration
        )
        if selected_registration is None:
            raise RuntimeError("Tray→World登记尚未完成")
        center_T = _vector(
            target.get("center_T_mm"), 3, "locked wafer centre"
        )
        transform = registration_transform_W_T(selected_registration)
        point_W = transform @ np.concatenate((center_T, [1.0]))
        if abs(float(point_W[3])) <= 1e-12:
            raise RuntimeError("硅片中心Tray→World齐次尺度为0")
        return point_W[:2] / point_W[3]

    def _continuity_motion_state(
        self, samples: Sequence[Mapping[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, Any]]]:
        """Validate fixed-height/fixed-Rz evidence for every fresh frame."""

        pose_rows = np.asarray(
            [
                _vector(sample.get("current_pose"), 6, "sample pose")
                for sample in samples
            ],
            dtype=np.float64,
        )
        joint_rows = np.asarray(
            [
                _vector(sample.get("current_joints"), 4, "sample joints")
                for sample in samples
            ],
            dtype=np.float64,
        )
        j3_errors = np.abs(joint_rows[:, 2] - self.required_j3_mm)
        pose_rz_errors = np.asarray(
            [
                abs(_angle_delta_deg(value, self.required_rz_deg))
                for value in pose_rows[:, 5]
            ],
            dtype=np.float64,
        )
        joint_rz_values = np.asarray(
            [
                rz_of(float(row[0]), float(row[1]), float(row[3]))
                for row in joint_rows
            ],
            dtype=np.float64,
        )
        joint_rz_errors = np.asarray(
            [
                abs(_angle_delta_deg(value, self.required_rz_deg))
                for value in joint_rz_values
            ],
            dtype=np.float64,
        )
        gates = {
            "all_frames_j3_at_required_height": {
                "passed": bool(np.all(j3_errors <= J3_TOLERANCE_MM + 1e-12)),
                "actual": j3_errors.astype(float).tolist(),
                "limit": f"each <={J3_TOLERANCE_MM:.2f} mm",
            },
            "all_frames_pose_rz_matches_calibration": {
                "passed": bool(
                    np.all(pose_rz_errors <= RZ_TOLERANCE_DEG + 1e-12)
                ),
                "actual": pose_rz_errors.astype(float).tolist(),
                "limit": f"each <={RZ_TOLERANCE_DEG:.2f} deg",
            },
            "all_frames_joint_rz_matches_calibration": {
                "passed": bool(
                    np.all(joint_rz_errors <= RZ_TOLERANCE_DEG + 1e-12)
                ),
                "actual": joint_rz_errors.astype(float).tolist(),
                "limit": f"each <={RZ_TOLERANCE_DEG:.2f} deg",
            },
        }
        return pose_rows, joint_rows, gates

    def _approve(
        self, request_id: str, proposal: Mapping[str, Any]
    ) -> dict[str, Any]:
        target_joints = list(
            ((proposal.get("planner") or {}).get("target_joints") or [])
        )
        return {
            "request_id": request_id,
            "decision": "approve",
            "calibration_sha256": self.calibration_hash,
            "proposal": dict(proposal),
            "target_joints": target_joints,
        }

    def _observe(
        self,
        request_id: str,
        reason: str,
        evaluation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "decision": "observe",
            "calibration_sha256": self.calibration_hash,
            "reason": str(reason),
            "evaluation": _json_safe_evaluation(evaluation),
        }

    def _abort(
        self,
        request_id: str,
        reason: str,
        evaluation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.status = "safety_rejected"
        self.result_message = str(reason)
        self._save()
        return {
            "request_id": request_id,
            "decision": "abort",
            "reason": str(reason),
            "evaluation": _json_safe_evaluation(evaluation),
        }

    def build_response(
        self,
        request: Mapping[str, Any],
        samples: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        request_id = str(request.get("request_id") or "")
        if (
            str(request.get("request_key") or "")
            != WAFER_CORRECTION_RUNTIME_REQUEST_KEY
        ):
            return self._abort(request_id, "硅片纠错会话收到未知请求")
        if (
            str(request.get("target_name") or "")
            != WAFER_CORRECTION_TARGET_NAME
        ):
            return self._abort(request_id, "硅片纠错会话目标名不匹配")
        if (
            str(request.get("calibration_sha256") or "").upper()
            != self.calibration_hash
        ):
            return self._abort(request_id, "平面手眼标定hash在会话中变化")

        image_names = self._save_images(samples, self.phase)
        try:
            observed_target = self._aggregate_target(samples)
        except Exception as exc:  # noqa: BLE001 - fail closed on target evidence
            return self._abort(
                request_id, f"槽外硅片5帧中心锁定失败：{exc}"
            )

        if self.phase == "registration":
            registration = self._build_registration(
                request, samples, "wafer_correction_initial_5_frames"
            )
            registration["evidence_image_filenames"] = image_names
            self._save_registration(registration)
            if registration.get("status") != "success":
                return self._abort(
                    request_id,
                    "硅片纠错要求一次通过的Tray→World登记；当前结果需要复核或已拒绝",
                    registration,
                )
            self.registration = registration
            self.locked_target = observed_target
            world_xy = self._target_world_xy(observed_target, registration)
            self.target_world_xy_mm = world_xy.astype(float).tolist()
            self.phase = "approach"
            self.status = "target_locked"
            self.result_message = (
                "已取5张有效帧的算术平均位置并冻结为槽外硅片中心；"
                "下一窗口开始固定J3的XY接近"
            )
            self._save()
            return self._observe(
                request_id,
                self.result_message,
                {
                    "runtime_registration": registration,
                    "locked_target": observed_target,
                    "target_world_xy_mm": self.target_world_xy_mm,
                },
            )

        if self.registration is None or self.locked_target is None:
            return self._abort(request_id, "硅片纠错锁定信息丢失")
        current_registration = self._build_registration(
            request, samples, "wafer_correction_continuity_5_frames"
        )
        if current_registration.get("status") != "success":
            return self._abort(
                request_id,
                "当前5帧Tray→World登记不是success，不能继续硅片纠错",
                current_registration,
            )
        drift_xy, drift_yaw = self._registration_drift(
            current_registration
        )
        locked_center = _vector(
            self.locked_target.get("center_T_mm"), 3, "locked centre"
        )
        observed_center = _vector(
            observed_target.get("center_T_mm"), 3, "observed centre"
        )
        target_drift = float(
            np.linalg.norm(observed_center[:2] - locked_center[:2])
        )
        try:
            pose_rows, joint_rows, motion_state_gates = (
                self._continuity_motion_state(samples)
            )
        except Exception as exc:  # noqa: BLE001 - fail closed on state evidence
            return self._abort(
                request_id,
                f"当前5帧固定J3/Rz状态证据无效：{exc}",
                current_registration,
            )
        continuity_gates = {
            "tray_registration_translation_drift": {
                "passed": drift_xy <= 0.75,
                "actual": drift_xy,
                "limit": "<=0.75 mm",
            },
            "tray_registration_yaw_drift": {
                "passed": drift_yaw <= 0.25,
                "actual": drift_yaw,
                "limit": "<=0.25 deg",
            },
            "locked_wafer_center_still_present": {
                "passed": True,
                "actual": target_drift,
                "limit": (
                    "diagnostic only; initial five-frame arithmetic mean "
                    "remains frozen"
                ),
            },
            **motion_state_gates,
        }
        if not all(gate["passed"] for gate in continuity_gates.values()):
            current_registration["continuity_gates"] = continuity_gates
            current_registration["observed_target"] = observed_target
            return self._abort(
                request_id,
                "托盘或已锁定槽外硅片在纠错期间发生漂移",
                current_registration,
            )

        current_pose = np.median(pose_rows, axis=0)
        current_joints = np.median(joint_rows, axis=0)
        try:
            target_world = self._target_world_xy(
                self.locked_target, current_registration
            )
        except Exception as exc:  # noqa: BLE001 - fail closed on reprojection
            return self._abort(
                request_id,
                f"当前success登记无法重投影锁定硅片中心：{exc}",
                current_registration,
            )
        self.target_world_xy_mm = target_world.astype(float).tolist()
        remaining_vector = target_world - current_pose[:2]
        remaining = float(np.linalg.norm(remaining_vector))

        window = {
            "phase": self.phase,
            "locked_target": self.locked_target,
            "observed_target": observed_target,
            "target_world_xy_mm": target_world.astype(float).tolist(),
            "current_world_xy_mm": current_pose[:2].astype(float).tolist(),
            "remaining_distance_mm": remaining,
            "continuity_gates": continuity_gates,
            "runtime_registration": current_registration,
            "evidence_image_filenames": image_names,
        }
        if self.phase == "hold":
            self.final_hold = window
            if remaining <= ARRIVAL_TOLERANCE_MM:
                self.status = "converged"
                self.result_message = (
                    "吸盘XY已到离P00最近的槽外硅片拟合几何中心上方；"
                    "独立5帧hold通过，J3未改变"
                )
                self._save()
                return {
                    "request_id": request_id,
                    "decision": "complete",
                    "calibration_sha256": self.calibration_hash,
                    "reason": self.result_message,
                    "evaluation": window,
                }
            self.phase = "approach"

        if remaining <= ARRIVAL_TOLERANCE_MM:
            self.phase = "hold"
            self.status = "awaiting_independent_hold"
            self.result_message = (
                "当前XY已进入0.20mm终止范围；下一独立5帧只验收不运动"
            )
            self._save()
            return self._observe(request_id, self.result_message, window)

        step_norm = min(remaining, MAXIMUM_NOMINAL_STEP_MM)
        command = remaining_vector * (step_norm / remaining)
        try:
            plan = plan_fixed_rz_xy_step(
                current_joints.astype(float).tolist(),
                current_pose.astype(float).tolist(),
                command.astype(float).tolist(),
                anchor_robot_xy_mm=self.static_audit_anchor_xy,
                local_extent_mm=130.0,
                domain_margin_mm=5.0,
                required_j3_mm=self.required_j3_mm,
                j3_tolerance_mm=0.20,
                required_rz_deg=self.required_rz_deg,
                rz_tolerance_deg=0.30,
                target_rz_tolerance_deg=0.15,
                max_xy_step_norm_mm=10.0,
                max_xy_axis_mm=10.0,
                max_sequential_transient_xy_mm=130.0,
                max_sequential_transient_rz_deg=15.0,
                precompensate_rz=True,
                enforce_sequential_intermediate_domain=False,
            )
        except Exception as exc:  # noqa: BLE001 - represented in session audit
            return self._abort(
                request_id, f"槽外硅片XY航点规划失败：{exc}", window
            )
        planner_passed = bool(
            (plan.get("audit") or {}).get("passed") is True
        )
        safety_gates = {
            **continuity_gates,
            "fixed_j3_rz_kinematic_planner": {
                "passed": planner_passed,
                "actual": (plan.get("audit") or {}).get("passed"),
                "limit": "true",
            },
            "nominal_xy_step_below_worker_hard_limit": {
                "passed": step_norm <= MAXIMUM_NOMINAL_STEP_MM + 1e-9,
                "actual": step_norm,
                "limit": f"<={MAXIMUM_NOMINAL_STEP_MM:.2f} mm",
            },
        }
        proposal = {
            "proposal_id": (
                f"wafer-correction-{len(self.iterations):03d}-{request_id}"
            ),
            "target_name": WAFER_CORRECTION_TARGET_NAME,
            "phase": "wafer_center_xy_approach",
            "motion_authorized": all(
                gate["passed"] for gate in safety_gates.values()
            ),
            "locked_target": self.locked_target,
            "observed_target": observed_target,
            "target_world_xy_mm": target_world.astype(float).tolist(),
            "remaining_distance_mm": remaining,
            "calculation": {
                "commanded_correction_xy_mm": command.astype(float).tolist()
            },
            "commanded_correction_xy_mm": command.astype(float).tolist(),
            "predicted_endpoint_xy_mm": (
                current_pose[:2] + command
            ).astype(float).tolist(),
            "planner": plan,
            "safety_gates": safety_gates,
            "evidence_image_filenames": image_names,
        }
        if proposal["motion_authorized"] is not True:
            return self._abort(
                request_id, "槽外硅片XY候选安全门拒绝", proposal
            )
        self.iterations.append(proposal)
        self.status = "approach_active"
        self.result_message = "槽外硅片中心XY分段接近中"
        self._save()
        return self._approve(request_id, proposal)

    def _save_registration(
        self, payload: Mapping[str, Any] | None = None
    ) -> None:
        report = dict(
            payload if payload is not None else (self.registration or {})
        )
        if not report:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self.registration_path,
            json.dumps(
                report, ensure_ascii=False, indent=2, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "stage": "nearest_p00_outside_wafer_xy_correction",
            "status": self.status,
            "phase": self.phase,
            "started_at": self.started_at,
            "updated_at": datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            ),
            "result_message": self.result_message,
            "locked_target": self.locked_target,
            "target_world_xy_mm": self.target_world_xy_mm,
            "runtime_registration": self.registration,
            "iterations": self.iterations,
            "final_hold": self.final_hold,
            "evidence_images": self.evidence_images,
            "locked_inputs": {
                "planar_handeye_path": str(
                    self.calibration.get("_source_path") or ""
                ),
                "planar_handeye_sha256": self.calibration_hash,
                "tray_geometry_path": str(self.geometry_path.resolve()),
                "tray_geometry_sha256": self.geometry_hash,
                "suction_target_path": str(
                    self.suction.source_path.resolve()
                ),
                "suction_target_sha256": self.suction.source_sha256,
            },
            "safety_boundary": {
                "candidate_state": "outside_slot only; stacked rejected",
                "full_contour_refinement_required": False,
                "full_contour_refinement_preferred": True,
                "outside_slot_fitted_quadrilateral_fallback_allowed": True,
                "five_frame_target_lock": True,
                "five_frame_center_method": "arithmetic_mean",
                "center_residual_gate": False,
                "xy_only": True,
                "fixed_j3_mm": self.required_j3_mm,
                "fixed_absolute_rz_deg": self.required_rz_deg,
                "maximum_nominal_xy_step_mm": MAXIMUM_NOMINAL_STEP_MM,
                "z_motion": False,
                "do_or_vacuum": False,
                "hardware_owner": "ActionWorker only",
            },
        }

    def _save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self.report_path,
            json.dumps(
                self._payload(), ensure_ascii=False, indent=2, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )

    def finish(self, ok: bool, message: str) -> None:
        completed_message = (
            self.result_message if self.status == "converged" else ""
        )
        if self.status != "converged":
            if ok:
                self.status = "not_converged"
            elif self.status not in {"safety_rejected", "failure"}:
                self.status = "stopped"
        self.result_message = completed_message or str(message)
        self._save()


__all__ = [
    "REGISTRATION_FILENAME",
    "RESULT_FILENAME",
    "WAFER_CORRECTION_RUNTIME_REQUEST_KEY",
    "WAFER_CORRECTION_TARGET_NAME",
    "WaferCorrectionSession",
]
