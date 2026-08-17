"""Controller-free moved-tray P22 registration and finite XY session.

The class only consumes fresh Stage-3/Stage-4 measurements and returns audited
joint targets.  ``ActionWorker`` remains the sole hardware owner and performs a
fresh independent kinematic/controller check immediately before each move.
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
from scara.pipeline.xy_correction_planner import plan_fixed_rz_xy_step

from .handeye_interaction import load_latest_suction_target, sha256_file
from .moved_tray_servo import (
    COARSE_ENDPOINT_HARD_LIMIT_MM,
    COARSE_VISUAL_GAIN,
    COARSE_VISUAL_MAXIMUM_STEP_MM,
    COARSE_SEQUENTIAL_TRANSIENT_RZ_LIMIT_DEG,
    COARSE_SEQUENTIAL_TRANSIENT_XY_LIMIT_MM,
    aggregate_metric_window,
    build_registered_control_proposal,
    final_hold_gates,
    registered_slot_world_xy_mm,
    registered_tray_workspace,
)
from .runtime_tray_registration import (
    build_runtime_tray_registration,
    fuse_three_pose_registrations,
    load_planar_handeye,
)
from .tray_pose_estimator import load_tray_board_geometry


MOVED_TRAY_RUNTIME_REQUEST_KEY = "moved_tray_p22_runtime_positioning"
RESULT_FILENAME = "full_tray_positioning.json"
REGISTRATION_FILENAME = "runtime_tray_registration.json"
RUNTIME_ACTION_SLOTS = 400


def _vector(value: Any, length: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != int(length) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label}必须包含{length}个有限数值")
    return result


def _angle_delta_deg(left: float, right: float) -> float:
    return (float(left) - float(right) + 180.0) % 360.0 - 180.0


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
        if payload.get("stage") != "moved_tray_runtime_registration_and_metric_xy_positioning":
            continue
        return "登记发生漂移" in str(payload.get("result_message") or "")
    return False


class MovedTrayPositioningSession:
    """Single-session runtime registration, coarse route, metric loop and hold."""

    def __init__(
        self,
        project_root: Path,
        output_dir: Path,
        initial_robot_state: Mapping[str, Any],
        *,
        target_name: str,
        camera_reconnected: bool = False,
    ) -> None:
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)
        self.target_name = str(target_name)
        if self.target_name != "P22":
            raise RuntimeError("可移动托盘第一版只授权P22")
        self.initial_joints = _vector(initial_robot_state.get("joints"), 4, "initial joints")
        self.initial_pose = _vector(initial_robot_state.get("pose"), 6, "initial pose")
        captured = float(initial_robot_state.get("captured_monotonic_s", math.nan))
        import time

        age = time.monotonic() - captured
        if not math.isfinite(age) or age < 0.0 or age > 1.0:
            raise RuntimeError("启动全盘定位时机械臂状态必须是1秒内的新鲜读数")
        self.suction = load_latest_suction_target(self.project_root)
        self.calibration = load_planar_handeye(self.project_root, self.suction)
        self.calibration_hash = str(self.calibration["_source_sha256"]).upper()
        self.geometry_path = self.project_root / "src/scara/calib/tray_board_geometry.json"
        self.geometry = load_tray_board_geometry(self.geometry_path)
        self.geometry_hash = sha256_file(self.geometry_path)
        self.required_j3_mm = float(self.suction.imaging_j3_mm)
        self.required_rz_deg = float(self.suction.rz_mean_deg)
        frame = self.geometry.get("tray_frame") or {}
        old_origin = _vector(frame.get("origin_mechanical_xy_mm"), 2, "Stage2 origin")
        slots = self.geometry.get("slots") or {}
        p22_T = _vector(slots.get("P22"), 3, "P22")
        old_rotation = np.asarray(frame.get("rotation_mechanical_from_tray"), dtype=np.float64)
        self.static_audit_anchor_xy = (old_origin + old_rotation[:2, :2] @ p22_T[:2]).tolist()
        self.started_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
        self.status = "waiting_for_runtime_registration"
        self.phase = "registration"
        self.result_message = ""
        self.registration: dict[str, Any] | None = None
        self.registration_candidate: dict[str, Any] | None = None
        self.probe_authorized = False
        self.probe_targets = list(self.calibration.get("prevalidated_probe_poses") or [])
        self.probe_target_index = 0
        self.probe_registrations: list[dict[str, Any]] = []
        self.coarse_movements = 0
        self.coarse_route_world_xy_mm: tuple[tuple[float, float], ...] = ()
        self.coarse_route_index = 0
        self.metric_movements = 0
        self.cumulative_metric_path_mm = 0.0
        self.iterations: list[dict[str, Any]] = []
        self.previous_iteration: dict[str, Any] | None = None
        self.final_hold: dict[str, Any] | None = None
        self.evidence_images: list[dict[str, Any]] = []
        self.photo_sequence = 0
        self.registration_context = {
            "camera_reconnected": bool(camera_reconnected),
            "installation_check_expired": bool(
                self.calibration.get("_installation_check_expired") is True
            ),
            "previous_registration_drifted": _previous_registration_drifted(
                self.project_root
            ),
        }

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
                "name": "可移动托盘全盘定位启动状态绑定",
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
                        "name": f"可移动托盘P22定位窗口 {index:03d}",
                        "request_key": MOVED_TRAY_RUNTIME_REQUEST_KEY,
                        "target_name": "P22",
                        "calibration_sha256": self.calibration_hash,
                        "anchor_robot_xy_mm": list(self.static_audit_anchor_xy),
                        "local_extent_mm": 130.0,
                        "domain_margin_mm": 5.0,
                        "required_j3_mm": self.required_j3_mm,
                        "required_rz_deg": self.required_rz_deg,
                        "max_xy_step_norm_mm": COARSE_ENDPOINT_HARD_LIMIT_MM,
                        "max_xy_axis_mm": COARSE_ENDPOINT_HARD_LIMIT_MM,
                        "j3_tolerance_mm": 0.20,
                        "rz_tolerance_deg": 0.30,
                        "target_rz_tolerance_deg": 0.15,
                        "max_sequential_transient_rz_deg": COARSE_SEQUENTIAL_TRANSIENT_RZ_LIMIT_DEG,
                        "precompensate_rz": True,
                        "enforce_sequential_intermediate_domain": False,
                        "max_state_drift_xy_mm": 0.20,
                        "max_state_drift_joint": 0.20,
                        "max_sequential_transient_xy_mm": COARSE_SEQUENTIAL_TRANSIENT_XY_LIMIT_MM,
                        "move_tolerance": 0.01,
                        "proposal_max_age_s": 8.0,
                        "fk_pose_xy_tolerance_mm": 0.20,
                    },
                ]
            )
        return {
            "api_version": 1,
            "name": "可移动托盘P22动态登记与全盘XY定位",
            "description": (
                "先用5张新鲜Stage3帧登记本次W←T，再按0.8增益和9.99mm名义上限粗定位，"
                "执行器继续保持10.00mm硬上限，"
                "10mm仅限制本轮终点净位移，J1/J2按控制器原有顺序自然运动，"
                "随后执行Stage3+Stage4毫米闭环和独立5帧hold验收。"
                "不加载Task9/Task11，不执行Z、DO或真空。"
            ),
            "camera_model": {
                "offset_mm": 0.0,
                "angle_reference": "world_negative_y",
                "positive_rotation": "counter_clockwise_from_above",
            },
            "actions": actions,
        }

    def authorize_three_pose_probe(self) -> None:
        if self.registration_candidate is None or self.registration_candidate.get("status") != "requires_three_pose_probe":
            raise RuntimeError("当前没有待确认的三姿态异常复核")
        if len(self.probe_targets) < 3:
            raise RuntimeError("平面手眼标定未保存3个预验证观察姿态")
        self.probe_authorized = True
        self.phase = "probe"
        self.status = "three_pose_probe_authorized"
        self.result_message = "人员已确认三姿态异常复核；尚未完成复核"
        self._save()

    def _save_images(self, samples: Sequence[Mapping[str, Any]], stage: str) -> list[str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        names: list[str] = []
        for sample in samples:
            image = sample.get("annotated_bgr")
            if image is None:
                continue
            self.photo_sequence += 1
            name = f"1_{self.photo_sequence:03d}.jpg"
            if not cv2.imwrite(str(self.output_dir / name), image):
                raise RuntimeError(f"无法保存全盘定位证据图片{name}")
            names.append(name)
            self.evidence_images.append(
                {
                    "filename": name,
                    "stage": str(stage),
                    "measurement_id": str(sample.get("measurement_id") or ""),
                }
            )
        return names

    def _build_registration(self, request: Mapping[str, Any], samples: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
        initial = method == "single_stationary_pose_5_frames"
        return build_runtime_tray_registration(
            samples,
            self.calibration,
            self.suction,
            self.geometry,
            requested_monotonic_s=float(request.get("requested_monotonic_s")),
            method=method,
            camera_reconnected=bool(
                initial and self.registration_context["camera_reconnected"]
            ),
            installation_check_expired=bool(
                initial
                and self.registration_context["installation_check_expired"]
            ),
            previous_registration_drifted=bool(
                initial
                and self.registration_context["previous_registration_drifted"]
            ),
        )

    def _registration_drift(self, current: Mapping[str, Any]) -> tuple[float, float]:
        if self.registration is None:
            return math.inf, math.inf
        origin = _vector(current.get("origin_world_xy_mm"), 2, "current registration origin")
        session_origin = _vector(self.registration.get("origin_world_xy_mm"), 2, "session registration origin")
        yaw = float(current.get("yaw_world_from_tray_deg"))
        session_yaw = float(self.registration.get("yaw_world_from_tray_deg"))
        return float(np.linalg.norm(origin - session_origin)), abs(_angle_delta_deg(yaw, session_yaw))

    def _initialize_coarse_route(self, samples: Sequence[Mapping[str, Any]]) -> None:
        """Freeze the 0.8-gain, 10 mm-capped visual route after registration."""

        if self.registration is None or self.registration.get("status") != "success":
            raise RuntimeError("不能在运行时托盘登记完成前创建粗定位路线")
        start = np.median(
            np.asarray(
                [
                    _vector(sample.get("current_robot_xy_mm"), 2, "sample XY")
                    for sample in samples
                ]
            ),
            axis=0,
        )
        target = registered_slot_world_xy_mm(self.geometry, "P22", self.registration)
        route: list[tuple[float, float]] = []
        cursor = start.astype(np.float64).copy()
        # Coarse vision ends once the runtime-registered world target is within
        # 2 mm.  Each outer iteration closes 80% of the remaining distance but
        # never requests more than 9.99 mm, leaving 0.01 mm below the worker's
        # immutable 10.00 mm endpoint ceiling.  This is an endpoint-displacement
        # limit only; the controller keeps its natural sequential J1/J2 motion.
        for _ in range(128):
            delta = target - cursor
            remaining = float(np.linalg.norm(delta))
            if remaining <= 2.0 + 1e-9:
                break
            step_norm = min(
                COARSE_VISUAL_GAIN * remaining,
                COARSE_VISUAL_MAXIMUM_STEP_MM,
            )
            point = cursor + delta * (step_norm / remaining)
            route.append((float(point[0]), float(point[1])))
            cursor = point
        else:  # pragma: no cover - finite contraction guarantees termination
            raise RuntimeError("粗定位0.8增益路线未能在128步内进入2mm邻域")
        self.coarse_route_world_xy_mm = tuple(route)
        self.coarse_route_index = 0

    def _approve(self, request_id: str, proposal: Mapping[str, Any]) -> dict[str, Any]:
        target_joints = list(((proposal.get("planner") or {}).get("target_joints") or []))
        return {
            "request_id": request_id,
            "decision": "approve",
            "calibration_sha256": self.calibration_hash,
            "proposal": dict(proposal),
            "target_joints": target_joints,
        }

    def _observe(self, request_id: str, reason: str, evaluation: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "decision": "observe",
            "calibration_sha256": self.calibration_hash,
            "reason": str(reason),
            "evaluation": None if evaluation is None else dict(evaluation),
        }

    def _abort(self, request_id: str, reason: str, evaluation: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self.status = "safety_rejected"
        self.result_message = str(reason)
        self._save()
        return {
            "request_id": request_id,
            "decision": "abort",
            "reason": str(reason),
            "evaluation": None if evaluation is None else dict(evaluation),
        }

    def _probe_step(self, request: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        request_id = str(request.get("request_id") or "")
        target = self.probe_targets[self.probe_target_index]
        tray_xy = _vector(target.get("tray_xy_mm"), 2, "probe target Tray XY")
        base_registration = self.registration_candidate
        if base_registration is None:
            return self._abort(request_id, "三姿态复核缺少初始登记")
        transform_W_T = np.asarray(base_registration.get("transform_W_T"), dtype=np.float64)
        if transform_W_T.shape != (4, 4) or not np.all(np.isfinite(transform_W_T)):
            return self._abort(request_id, "三姿态复核缺少有效的初始W←T")
        # Probe poses are fixed relative to the rigid Tray, not to the old
        # robot world.  Recompute them from the current session registration.
        target_xy = (
            transform_W_T
            @ np.asarray([float(tray_xy[0]), float(tray_xy[1]), 0.0, 1.0])
        )[:2]
        current_xy = np.median(
            np.asarray([_vector(sample.get("current_robot_xy_mm"), 2, "sample XY") for sample in samples]),
            axis=0,
        )
        delta = target_xy - current_xy
        distance = float(np.linalg.norm(delta))
        if distance > 0.10:
            command = delta if distance <= 2.0 else delta * (2.0 / distance)
            plan = plan_fixed_rz_xy_step(
                np.median(
                    np.asarray(
                        [
                            _vector(sample.get("current_joints"), 4, "sample joints")
                            for sample in samples
                        ]
                    ),
                    axis=0,
                ).astype(float).tolist(),
                np.median(
                    np.asarray(
                        [
                            _vector(sample.get("current_pose"), 6, "sample pose")
                            for sample in samples
                        ]
                    ),
                    axis=0,
                ).astype(float).tolist(),
                command.astype(float).tolist(),
                anchor_robot_xy_mm=self.static_audit_anchor_xy,
                local_extent_mm=130.0,
                domain_margin_mm=5.0,
                required_j3_mm=self.required_j3_mm,
                j3_tolerance_mm=0.20,
                required_rz_deg=self.required_rz_deg,
                rz_tolerance_deg=0.30,
                target_rz_tolerance_deg=0.15,
                max_xy_step_norm_mm=2.000001,
                max_xy_axis_mm=2.000001,
                max_sequential_transient_xy_mm=4.0,
                max_sequential_transient_rz_deg=1.0,
                precompensate_rz=True,
                enforce_sequential_intermediate_domain=False,
            )
            proposal = {
                "proposal_id": f"moved-tray-probe-{self.probe_target_index}-{request_id}",
                "target_name": "P22",
                "phase": "moved_tray_three_pose_probe_route",
                "motion_authorized": bool((plan.get("audit") or {}).get("passed") is True),
                "probe_target_index": self.probe_target_index,
                "probe_target": dict(target),
                "calculation": {"commanded_correction_xy_mm": command.astype(float).tolist()},
                "commanded_correction_xy_mm": command.astype(float).tolist(),
                "predicted_endpoint_xy_mm": (current_xy + command).astype(float).tolist(),
                "planner": plan,
                "safety_gates": {
                    "kinematic_planner": {
                        "passed": bool((plan.get("audit") or {}).get("passed") is True),
                        "actual": (plan.get("audit") or {}).get("passed"),
                        "limit": "true",
                    }
                },
            }
            if proposal["motion_authorized"] is not True:
                return self._abort(request_id, "三姿态复核航点运动学门拒绝", proposal)
            self.status = "three_pose_probe_moving"
            self._save()
            return self._approve(request_id, proposal)

        registration = self._build_registration(
            request,
            samples,
            f"three_pose_probe_{self.probe_target_index + 1}",
        )
        if registration.get("status") == "rejected":
            return self._abort(request_id, "三姿态复核单姿态硬门拒绝", registration)
        self.probe_registrations.append(registration)
        self.probe_target_index += 1
        if self.probe_target_index < 3:
            self.status = "three_pose_probe_collecting"
            self._save()
            return self._observe(request_id, f"三姿态复核已完成{self.probe_target_index}/3，继续下一姿态", registration)
        fused = fuse_three_pose_registrations(self.probe_registrations, self.geometry)
        if fused.get("status") != "success":
            return self._abort(request_id, "三姿态融合质量门未通过", fused)
        fused["calibration"] = dict((self.registration_candidate or {}).get("calibration") or {})
        self.registration = fused
        self._initialize_coarse_route(samples)
        self.phase = "coarse"
        self.status = "runtime_registration_success"
        self.result_message = "三姿态异常复核通过，准备动态P22粗定位"
        self._save_registration()
        self._save()
        return self._observe(request_id, self.result_message, fused)

    def build_response(self, request: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        request_id = str(request.get("request_id") or "")
        if str(request.get("request_key") or "") != MOVED_TRAY_RUNTIME_REQUEST_KEY:
            return self._abort(request_id, "可移动托盘会话收到未知请求")
        if str(request.get("target_name") or "") != "P22":
            return self._abort(request_id, "可移动托盘会话目标不是P22")
        if str(request.get("calibration_sha256") or "").upper() != self.calibration_hash:
            return self._abort(request_id, "平面手眼标定hash在会话中发生变化")
        image_names = self._save_images(samples, self.phase)

        if self.phase == "registration":
            registration = self._build_registration(request, samples, "single_stationary_pose_5_frames")
            registration["evidence_image_filenames"] = image_names
            self.registration_candidate = registration
            self._save_registration(registration)
            if registration.get("status") == "rejected":
                return self._abort(request_id, "单姿态托盘登记硬门拒绝", registration)
            if registration.get("requires_three_pose_probe") is True:
                self.status = "three_pose_probe_confirmation_required"
                self.result_message = "单姿态登记触发三姿态异常复核，等待人员确认"
                self._save()
                return {
                    "request_id": request_id,
                    "decision": "probe_required",
                    "reason": self.result_message,
                    "evaluation": registration,
                }
            self.registration = registration
            self._initialize_coarse_route(samples)
            self.phase = "coarse"
            self.status = "runtime_registration_success"
            self.result_message = "单姿态5帧托盘登记通过；准备动态P22粗定位"
            self._save()
            return self._observe(request_id, self.result_message, registration)

        if self.phase == "probe":
            return self._probe_step(request, samples)

        if self.registration is None:
            return self._abort(request_id, "运行时登记结果丢失")
        current_registration = self._build_registration(request, samples, "continuity_check_5_frames")
        if current_registration.get("status") == "rejected":
            return self._abort(request_id, "当前5帧无法继续验证托盘登记", current_registration)
        drift_xy, drift_yaw = self._registration_drift(current_registration)
        drift_gates = {
            "session_registration_translation_drift": {
                "passed": drift_xy <= 0.75,
                "actual": drift_xy,
                "limit": "<=0.75 mm",
            },
            "session_registration_yaw_drift": {
                "passed": drift_yaw <= 0.25,
                "actual": drift_yaw,
                "limit": "<=0.25 deg",
            },
        }
        if not all(gate["passed"] for gate in drift_gates.values()):
            current_registration["continuity_gates"] = drift_gates
            return self._abort(request_id, "闭环期间托盘登记发生漂移", current_registration)

        if self.phase == "coarse":
            current_xy = np.median(
                np.asarray([_vector(sample.get("current_robot_xy_mm"), 2, "sample XY") for sample in samples]), axis=0
            )
            while self.coarse_route_index < len(self.coarse_route_world_xy_mm):
                waypoint = np.asarray(
                    self.coarse_route_world_xy_mm[self.coarse_route_index],
                    dtype=np.float64,
                )
                if float(np.linalg.norm(waypoint - current_xy)) > 0.20:
                    break
                self.coarse_route_index += 1
            target_xy = registered_slot_world_xy_mm(self.geometry, "P22", self.registration)
            if self.coarse_route_index >= len(self.coarse_route_world_xy_mm):
                if float(np.linalg.norm(target_xy - current_xy)) > 2.0:
                    return self._abort(
                        request_id,
                        "不可变粗路线耗尽但尚未进入P22的2mm邻域",
                        current_registration,
                    )
                self.phase = "metric"
                self.previous_iteration = None
                self.status = "coarse_route_complete"
                self.result_message = "已进入P22附近2mm，下一窗口开始Stage3+Stage4毫米闭环"
                self._save()
                return self._observe(request_id, self.result_message, current_registration)
            proposal = build_registered_control_proposal(
                samples,
                request,
                self.registration,
                self.geometry,
                phase="coarse",
                movement_index=0,
                cumulative_path_mm=0.0,
                previous_iteration=self.previous_iteration,
                coarse_goal_world_xy_mm=self.coarse_route_world_xy_mm[
                    self.coarse_route_index
                ],
            )
            proposal["coarse_route_index"] = self.coarse_route_index
            proposal["coarse_route_count"] = len(self.coarse_route_world_xy_mm)
            proposal["registration_continuity_gates"] = drift_gates
            proposal["safety_gates"].update(drift_gates)
            proposal["motion_authorized"] = all(gate["passed"] for gate in proposal["safety_gates"].values())
            proposal["evidence_image_filenames"] = image_names
            if proposal["motion_authorized"] is not True:
                return self._abort(request_id, "动态粗定位安全门拒绝", proposal)
            self.coarse_movements += 1
            self.previous_iteration = proposal
            self.iterations.append(proposal)
            self.status = "coarse_route_active"
            self._save()
            return self._approve(request_id, proposal)

        window = aggregate_metric_window(samples)
        if self.phase == "hold":
            gates = final_hold_gates(window)
            self.final_hold = {
                "window": window,
                "quality_gates": gates,
                "measurement_ids": [str(sample.get("measurement_id") or "") for sample in samples],
                "evidence_image_filenames": image_names,
            }
            if all(gate["passed"] for gate in gates.values()):
                self.status = "converged"
                self.result_message = "视觉估计进入1 mm；独立5帧hold全部通过"
                self._save()
                return {
                    "request_id": request_id,
                    "decision": "complete",
                    "calibration_sha256": self.calibration_hash,
                    "reason": self.result_message,
                    "evaluation": self.final_hold,
                }
            self.phase = "metric"
            self.result_message = "独立hold未通过，恢复毫米闭环"

        if float(window["median_error_norm_mm"]) <= 0.60:
            self.phase = "hold"
            self.status = "awaiting_independent_hold"
            self.result_message = "控制窗口中位误差已≤0.60mm；下一独立5帧只验收不控制"
            self._save()
            return self._observe(request_id, self.result_message, window)

        proposal = build_registered_control_proposal(
            samples,
            request,
            self.registration,
            self.geometry,
            phase="metric",
            movement_index=self.metric_movements,
            cumulative_path_mm=self.cumulative_metric_path_mm,
            previous_iteration=self.previous_iteration,
        )
        proposal["registration_continuity_gates"] = drift_gates
        proposal["safety_gates"].update(drift_gates)
        proposal["motion_authorized"] = all(gate["passed"] for gate in proposal["safety_gates"].values())
        proposal["evidence_image_filenames"] = image_names
        if proposal["motion_authorized"] is not True:
            return self._abort(request_id, "Stage3毫米闭环安全门拒绝", proposal)
        self.metric_movements += 1
        self.cumulative_metric_path_mm = float(proposal["cumulative_path_after_mm"])
        self.previous_iteration = proposal
        self.iterations.append(proposal)
        self.status = "metric_loop_active"
        self._save()
        return self._approve(request_id, proposal)

    def _save_registration(self, payload: Mapping[str, Any] | None = None) -> None:
        report = dict(payload if payload is not None else (self.registration or {}))
        if not report:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self.registration_path,
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def _payload(self) -> dict[str, Any]:
        workspace = None
        if self.registration is not None and self.registration.get("status") == "success":
            workspace = registered_tray_workspace(self.geometry, self.registration)
        return {
            "schema_version": 2,
            "stage": "moved_tray_runtime_registration_and_metric_xy_positioning",
            "status": self.status,
            "phase": self.phase,
            "started_at": self.started_at,
            "updated_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "result_message": self.result_message,
            "target_name": self.target_name,
            "currently_authorized_target_names": ["P22"],
            "locked_inputs": {
                "planar_handeye_path": str(self.calibration.get("_source_path") or ""),
                "planar_handeye_sha256": self.calibration_hash,
                "tray_geometry_path": str(self.geometry_path.resolve()),
                "tray_geometry_sha256": self.geometry_hash,
                "suction_target_path": str(self.suction.source_path.resolve()),
                "suction_target_sha256": self.suction.source_sha256,
            },
            "runtime_registration": self.registration,
            "registration_context": self.registration_context,
            "registration_candidate": self.registration_candidate,
            "three_pose_probe": {
                "operator_authorized": self.probe_authorized,
                "target_index": self.probe_target_index,
                "targets": self.probe_targets,
                "registrations": self.probe_registrations,
            },
            "runtime_workspace": workspace,
            "coarse_phase": {
                "movement_count": self.coarse_movements,
                "immutable_route_created_after_registration": True,
                "route_world_xy_mm": [list(point) for point in self.coarse_route_world_xy_mm],
                "route_waypoint_count": len(self.coarse_route_world_xy_mm),
                "next_route_index": self.coarse_route_index,
                "maximum_nominal_waypoint_spacing_mm": 2.0,
            },
            "metric_closed_loop": {
                "movement_count": self.metric_movements,
                "maximum_movement_count": 32,
                "cumulative_path_mm": self.cumulative_metric_path_mm,
                "maximum_cumulative_path_mm": 50.0,
                "iterations": self.iterations,
            },
            "final_hold": self.final_hold,
            "evidence_images": self.evidence_images,
            "safety_boundary": {
                "xy_only": True,
                "fixed_j3_mm": self.required_j3_mm,
                "fixed_absolute_rz_deg": self.required_rz_deg,
                "z_motion": False,
                "do_or_vacuum": False,
                "visual_claim": "视觉估计进入1 mm；不是硅片物理放置精度",
                "hardware_owner": "ActionWorker only",
            },
        }

    def _save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self.report_path,
            json.dumps(self._payload(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def finish(self, ok: bool, message: str) -> None:
        completed_message = self.result_message if self.status == "converged" else ""
        if self.status != "converged":
            if ok:
                self.status = "not_converged"
            elif self.status not in {"safety_rejected", "failure"}:
                self.status = "stopped"
        # ActionWorker's generic completion text must not overwrite the actual
        # independent-hold acceptance claim saved by this session.
        self.result_message = completed_message or str(message)
        manifest_path = self.output_dir / "points.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            manifest["full_tray_positioning"] = {
                "status": self.status,
                "target_name": "P22",
                "result_file": RESULT_FILENAME,
                "registration_file": REGISTRATION_FILENAME,
                "coarse_movement_count": self.coarse_movements,
                "metric_movement_count": self.metric_movements,
                "evidence_image_count": len(self.evidence_images),
                "visual_one_mm_hold_passed": self.status == "converged",
            }
            atomic_write_text(
                manifest_path,
                json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        self._save()


__all__ = [
    "MOVED_TRAY_RUNTIME_REQUEST_KEY",
    "MovedTrayPositioningSession",
    "REGISTRATION_FILENAME",
    "RESULT_FILENAME",
]
