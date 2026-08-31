"""Fail-closed XY-only positioning above a selected tray wafer.

This module owns no camera, Qt widget, or robot controller.  It consumes five
fresh camera-1 observations for each request and may return one candidate joint
target.  ``ActionWorker`` remains the sole hardware owner and repeats the
controller and kinematic checks immediately before every physical move.
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
from scara.pipeline.xy_correction_planner import (
    angular_difference_deg,
    plan_fixed_rz_xy_step,
)

from .handeye_interaction import load_latest_suction_target, sha256_file
from .moved_tray_servo import registered_slot_world_xy_mm
from .runtime_tray_registration import load_planar_handeye
from .tray_pose_estimator import load_tray_board_geometry


WAFER_PICK_XY_RUNTIME_REQUEST_KEY = "wafer_pick_xy_overhead_positioning"
RESULT_FILENAME = "wafer_pick_xy_positioning.json"
RUNTIME_ACTION_SLOTS = 120
OBSERVATION_WINDOW_SIZE = 5
MAXIMUM_MOVEMENT_COUNT = 100
MAXIMUM_CUMULATIVE_PATH_MM = 200.0
MAXIMUM_STEP_MM = 2.0
LOCAL_EXTENT_MM = 70.0
DOMAIN_MARGIN_MM = 5.0
ARRIVAL_MEDIAN_MM = 0.50
ARRIVAL_MAXIMUM_MM = 0.80
ARRIVAL_RMS_MM = 0.25


def _vector(value: Any, length: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != int(length) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label}必须包含{length}个有限数值")
    return result


def _transform(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label}必须是有限4x4矩阵")
    if abs(float(np.linalg.det(result[:3, :3]))) < 1e-9:
        raise ValueError(f"{label}旋转部分不可逆")
    return result


def _gate(passed: bool, actual: Any, limit: str, note: str = "") -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "actual": actual,
        "limit": str(limit),
        "note": str(note),
    }


def _yaw_deg(transform_W_T: np.ndarray) -> float:
    return math.degrees(
        math.atan2(float(transform_W_T[1, 0]), float(transform_W_T[0, 0]))
    )


def _registered_tray_center_world_xy(
    geometry: Mapping[str, Any], registration: Mapping[str, Any]
) -> np.ndarray:
    slots = geometry.get("slots") or {}
    points = np.asarray(list(slots.values()), dtype=np.float64)
    if points.shape != (36, 3) or not np.all(np.isfinite(points)):
        raise ValueError("托盘几何必须包含36个有限槽中心")
    center_T = np.mean(points, axis=0)
    transform_W_T = _transform(registration.get("transform_W_T"), "transform_W_T")
    center_W = transform_W_T @ np.asarray([*center_T, 1.0], dtype=np.float64)
    return center_W[:2] / center_W[3]


def _step_policy(distance_mm: float) -> tuple[float, float]:
    distance = float(distance_mm)
    if distance > 5.0:
        return 0.85, 2.0
    if distance > 2.0:
        return 0.75, 1.0
    if distance > 0.75:
        return 0.65, 0.50
    return 0.50, 0.25


class WaferPickXYPositioningSession:
    """Generate bounded XY waypoints above one locked occupied slot."""

    def __init__(
        self,
        project_root: Path,
        output_dir: Path,
        initial_robot_state: Mapping[str, Any],
        initial_snapshot: Mapping[str, Any],
        *,
        target_name: str,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.output_dir = Path(output_dir)
        self.target_name = str(target_name)
        if not (
            len(self.target_name) == 3
            and self.target_name[0] == "P"
            and self.target_name[1] in "012345"
            and self.target_name[2] in "012345"
        ):
            raise ValueError("拾取目标必须是P00到P55中的一个槽位")

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
        if not math.isfinite(age) or age < -0.05 or age > 1.0:
            raise RuntimeError("启动XY悬空定位时机械臂状态必须是1秒内的新鲜读数")

        source_slot = str(initial_snapshot.get("source_slot") or "")
        if source_slot != self.target_name:
            raise RuntimeError("启动时锁定的拾取槽与当前点击目标不一致")
        if initial_snapshot.get("tracking_ready") is not True:
            raise RuntimeError("当前视觉质量门未全部通过，不能启动XY悬空定位")
        source_state = initial_snapshot.get("source_state") or {}
        if source_state.get("state") != "occupied":
            raise RuntimeError("拾取目标必须持续识别为正常单片 occupied")
        registration = initial_snapshot.get("registration") or {}
        if registration.get("status") != "success":
            raise RuntimeError("缺少成功的本次 W<-T 登记")

        self.geometry_path = (
            self.project_root / "src/scara/calib/tray_board_geometry.json"
        )
        self.geometry = load_tray_board_geometry(self.geometry_path)
        if self.target_name not in (self.geometry.get("slots") or {}):
            raise ValueError(f"托盘几何中没有目标槽 {self.target_name}")
        self.suction = load_latest_suction_target(self.project_root)
        self.handeye = load_planar_handeye(self.project_root, self.suction)
        self.calibration_hash = str(self.handeye["_source_sha256"]).upper()
        self.geometry_hash = sha256_file(self.geometry_path)
        self.required_j3_mm = float(self.suction.imaging_j3_mm)
        self.required_rz_deg = float(self.suction.rz_mean_deg)
        self.anchor_robot_xy_mm = _registered_tray_center_world_xy(
            self.geometry, registration
        ).astype(float).tolist()
        self.initial_registration_origin_xy_mm = _vector(
            registration.get("origin_world_xy_mm"),
            2,
            "initial registration origin",
        )
        self.initial_registration_yaw_deg = float(
            registration.get("yaw_world_from_tray_deg", math.nan)
        )
        if not math.isfinite(self.initial_registration_yaw_deg):
            raise RuntimeError("启动时W<-T缺少有效yaw")

        if abs(self.initial_joints[2] - self.required_j3_mm) > 0.20:
            raise RuntimeError(
                "当前J3不在相机1安全观察高度；本功能不会自动升降机械臂"
            )
        current_rz = rz_of(
            self.initial_joints[0], self.initial_joints[1], self.initial_joints[3]
        )
        if angular_difference_deg(current_rz, self.required_rz_deg) > 0.30:
            raise RuntimeError("当前绝对Rz不在标定姿态容差内")

        self.started_at = datetime.now().astimezone().isoformat(
            timespec="milliseconds"
        )
        self.status = "armed_waiting_for_fresh_frames"
        self.result_message = ""
        self.movement_count = 0
        self.cumulative_path_mm = 0.0
        self.previous_distance_mm: float | None = None
        self.previous_command_norm_mm: float | None = None
        self.iterations: list[dict[str, Any]] = []
        self.final_hold: dict[str, Any] | None = None
        self.evidence_images: list[dict[str, Any]] = []

    @property
    def report_path(self) -> Path:
        return self.output_dir / RESULT_FILENAME

    def action_task(self) -> dict[str, Any]:
        actions: list[dict[str, Any]] = [
            {
                "type": "assert_joints",
                "name": f"{self.target_name} XY悬空定位启动状态绑定",
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
                        "name": f"{self.target_name} XY悬空定位窗口 {index:03d}",
                        "request_key": WAFER_PICK_XY_RUNTIME_REQUEST_KEY,
                        "target_name": self.target_name,
                        "calibration_sha256": self.calibration_hash,
                        "anchor_robot_xy_mm": list(self.anchor_robot_xy_mm),
                        "local_extent_mm": LOCAL_EXTENT_MM,
                        "domain_margin_mm": DOMAIN_MARGIN_MM,
                        "required_j3_mm": self.required_j3_mm,
                        "required_rz_deg": self.required_rz_deg,
                        "max_xy_step_norm_mm": MAXIMUM_STEP_MM,
                        "max_xy_axis_mm": MAXIMUM_STEP_MM,
                        "j3_tolerance_mm": 0.20,
                        "rz_tolerance_deg": 0.30,
                        "target_rz_tolerance_deg": 0.15,
                        "max_sequential_transient_rz_deg": 1.0,
                        "precompensate_rz": True,
                        "max_state_drift_xy_mm": 0.10,
                        "max_state_drift_joint": 0.10,
                        "max_sequential_transient_xy_mm": 5.0,
                        "move_tolerance": 0.02,
                        "proposal_max_age_s": 5.0,
                        "fk_pose_xy_tolerance_mm": 0.20,
                    },
                ]
            )
        return {
            "api_version": 1,
            "name": f"吸盘移动到{self.target_name}硅片正上方",
            "description": (
                "使用相机1、本次W<-T和五帧复测做分段XY悬空定位。"
                "固定J3安全观察高度与绝对Rz；不执行下降、吸取、DO或真空。"
            ),
            "camera_model": {
                "offset_mm": 0.0,
                "angle_reference": "world_negative_y",
                "positive_rotation": "counter_clockwise_from_above",
            },
            "actions": actions,
        }

    def _save_images(
        self, samples: Sequence[Mapping[str, Any]], request_id: str
    ) -> list[str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        names: list[str] = []
        for index, sample in enumerate(samples, start=1):
            image = sample.get("annotated_bgr")
            if not isinstance(image, np.ndarray) or image.ndim != 3:
                continue
            name = (
                f"pick_xy_{self.movement_count:03d}_{request_id[-10:]}_"
                f"frame_{index:02d}.png"
            )
            path = self.output_dir / name
            if not cv2.imwrite(str(path), image):
                raise RuntimeError(f"无法保存XY悬空定位证据图 {name}")
            names.append(name)
            self.evidence_images.append(
                {
                    "filename": name,
                    "measurement_id": str(sample.get("measurement_id") or ""),
                    "request_id": request_id,
                }
            )
        return names

    def _window(self, request: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        rows = list(samples)
        if len(rows) != OBSERVATION_WINDOW_SIZE:
            raise ValueError("XY悬空定位每轮必须恰好使用5张新鲜画面")
        requested_at = float(request.get("requested_monotonic_s", math.nan))
        if not math.isfinite(requested_at):
            raise ValueError("执行请求缺少有效时间戳")

        transforms: list[np.ndarray] = []
        target_world: list[np.ndarray] = []
        robot_xy: list[np.ndarray] = []
        distances: list[float] = []
        origins: list[np.ndarray] = []
        yaws: list[float] = []
        rms_values: list[float] = []
        all_gates_pass = True
        all_accepted = True
        all_occupied = True
        all_fresh = True
        sequences: list[int] = []
        captured_times: list[float] = []
        all_target_locked = True
        all_states_valid = True

        for sample in rows:
            all_accepted = all_accepted and sample.get("accepted") is True
            captured = float(sample.get("captured_monotonic_s", math.nan))
            all_fresh = all_fresh and math.isfinite(captured) and captured >= requested_at
            captured_times.append(captured)
            try:
                sequences.append(int(sample.get("frame_sequence")))
            except (TypeError, ValueError, OverflowError):
                sequences.append(-1)
            all_target_locked = all_target_locked and (
                str(sample.get("target_name") or "") == self.target_name
            )
            all_occupied = all_occupied and (
                (sample.get("source_state") or {}).get("state") == "occupied"
            )
            gates = sample.get("selection_gates") or {}
            all_gates_pass = all_gates_pass and bool(gates) and all(
                isinstance(gate, Mapping) and gate.get("passed") is True
                for gate in gates.values()
            )
            registration = sample.get("registration") or {}
            transform = _transform(
                registration.get("transform_W_T"), "sample transform_W_T"
            )
            if registration.get("status") != "success":
                all_states_valid = False
            state = sample.get("robot_state") or {}
            pose = _vector(state.get("pose"), 6, "sample robot pose")
            joints = _vector(state.get("joints"), 4, "sample robot joints")
            all_states_valid = all_states_valid and (
                abs(joints[2] - self.required_j3_mm) <= 0.20
                and angular_difference_deg(
                    rz_of(joints[0], joints[1], joints[3]), self.required_rz_deg
                )
                <= 0.30
            )
            point = registered_slot_world_xy_mm(
                self.geometry, self.target_name, registration
            )
            transforms.append(transform)
            target_world.append(point)
            robot_xy.append(pose[:2])
            distances.append(float(np.linalg.norm(point - pose[:2])))
            origins.append(transform[:2, 3])
            yaws.append(_yaw_deg(transform))
            rms_values.append(float(sample.get("reprojection_rms_px", math.inf)))

        target_array = np.asarray(target_world, dtype=np.float64)
        robot_array = np.asarray(robot_xy, dtype=np.float64)
        origin_array = np.asarray(origins, dtype=np.float64)
        target_median = np.median(target_array, axis=0)
        robot_median = np.median(robot_array, axis=0)
        distance_array = np.asarray(distances, dtype=np.float64)
        target_deviation = np.linalg.norm(target_array - target_median, axis=1)
        robot_deviation = np.linalg.norm(robot_array - robot_median, axis=1)
        origin_deviation = np.linalg.norm(
            origin_array - np.median(origin_array, axis=0), axis=1
        )
        yaw_median = float(np.median(yaws))
        yaw_deviation = np.asarray(
            [angular_difference_deg(value, yaw_median) for value in yaws],
            dtype=np.float64,
        )
        distance_median = float(np.median(distance_array))
        distance_rms = float(
            math.sqrt(float(np.mean((distance_array - distance_median) ** 2)))
        )
        request_state = request.get("controller_state") or {}
        request_joints = _vector(
            request_state.get("joints"), 4, "request controller joints"
        )
        request_pose = _vector(
            request_state.get("pose"), 6, "request controller pose"
        )
        frame_joint_rows = np.asarray(
            [
                _vector(
                    (row.get("robot_state") or {}).get("joints"),
                    4,
                    "frame robot joints",
                )
                for row in rows
            ],
            dtype=np.float64,
        )
        request_xy_difference = float(
            np.linalg.norm(robot_median - request_pose[:2])
        )
        request_joint_difference = float(
            np.max(np.abs(np.median(frame_joint_rows, axis=0) - request_joints))
        )
        registration_translation_drift = float(
            np.linalg.norm(
                np.median(origin_array, axis=0)
                - self.initial_registration_origin_xy_mm
            )
        )
        registration_yaw_drift = angular_difference_deg(
            yaw_median, self.initial_registration_yaw_deg
        )

        gates = {
            "all_frames_explicitly_accepted": _gate(
                all_accepted, all_accepted, "true for all five frames"
            ),
            "five_fresh_post_request_frames": _gate(
                all_fresh, [float(row.get("captured_monotonic_s", math.nan)) for row in rows], ">= request timestamp"
            ),
            "five_distinct_ordered_frames": _gate(
                len(set(sequences)) == OBSERVATION_WINDOW_SIZE
                and all(value >= 0 for value in sequences)
                and all(
                    captured_times[index] > captured_times[index - 1]
                    for index in range(1, len(captured_times))
                ),
                {
                    "frame_sequences": sequences,
                    "captured_monotonic_s": captured_times,
                },
                "five unique frame sequences with strictly increasing capture times",
            ),
            "target_name_locked": _gate(
                all_target_locked,
                [str(row.get("target_name") or "") for row in rows],
                f"all {self.target_name}",
            ),
            "source_remains_normal_occupied": _gate(
                all_occupied,
                [(row.get("source_state") or {}).get("state") for row in rows],
                "all occupied",
            ),
            "overview_selection_gates": _gate(
                all_gates_pass, all_gates_pass, "all PASS in all five frames"
            ),
            "robot_height_and_rz_locked": _gate(
                all_states_valid,
                {
                    "required_j3_mm": self.required_j3_mm,
                    "required_rz_deg": self.required_rz_deg,
                },
                "J3 <=0.20 mm and Rz <=0.30 deg",
            ),
            "target_world_repeatability": _gate(
                float(np.max(target_deviation)) <= 0.60,
                {
                    "rms_mm": float(math.sqrt(float(np.mean(target_deviation ** 2)))),
                    "maximum_mm": float(np.max(target_deviation)),
                },
                "maximum <=0.60 mm",
            ),
            "robot_stationary_during_window": _gate(
                float(np.max(robot_deviation)) <= 0.20,
                float(np.max(robot_deviation)),
                "<=0.20 mm",
            ),
            "registration_origin_stability": _gate(
                float(np.max(origin_deviation)) <= 0.60,
                float(np.max(origin_deviation)),
                "<=0.60 mm",
            ),
            "registration_yaw_stability": _gate(
                float(np.max(yaw_deviation)) <= 0.20,
                float(np.max(yaw_deviation)),
                "<=0.20 deg",
            ),
            "registration_locked_to_armed_session": _gate(
                registration_translation_drift <= 0.75
                and registration_yaw_drift <= 0.25,
                {
                    "translation_drift_mm": registration_translation_drift,
                    "yaw_drift_deg": registration_yaw_drift,
                },
                "translation <=0.75 mm and yaw <=0.25 deg",
            ),
            "request_controller_state_matches_frames": _gate(
                request_xy_difference <= 0.20
                and request_joint_difference <= 0.20,
                {
                    "xy_difference_mm": request_xy_difference,
                    "maximum_joint_difference_deg_or_mm": request_joint_difference,
                },
                "XY <=0.20 mm and every joint <=0.20",
            ),
            "finite_pose_rms": _gate(
                all(math.isfinite(value) for value in rms_values),
                rms_values,
                "all finite and existing pose gates PASS",
            ),
        }
        return {
            "gates": gates,
            "target_world_xy_mm": target_median.astype(float).tolist(),
            "robot_world_xy_mm": robot_median.astype(float).tolist(),
            "distance_median_mm": distance_median,
            "distance_maximum_mm": float(np.max(distance_array)),
            "distance_rms_mm": distance_rms,
            "target_world_maximum_deviation_mm": float(np.max(target_deviation)),
            "registration_origin_world_xy_mm": np.median(origin_array, axis=0).astype(float).tolist(),
            "registration_yaw_world_from_tray_deg": yaw_median,
        }

    def _progress_gate(self, distance_mm: float) -> dict[str, Any]:
        if self.previous_distance_mm is None or self.previous_command_norm_mm is None:
            return _gate(True, None, "not applicable before first motion")
        required_decrease = min(0.05, 0.10 * self.previous_command_norm_mm)
        actual_decrease = self.previous_distance_mm - float(distance_mm)
        return _gate(
            actual_decrease >= required_decrease,
            actual_decrease,
            f">={required_decrease:.3f} mm",
            "每次实际移动后，下一独立五帧必须证明距离继续减小",
        )

    def build_response(
        self,
        request: Mapping[str, Any],
        samples: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        request_id = str(request.get("request_id") or "")
        if str(request.get("request_key") or "") != WAFER_PICK_XY_RUNTIME_REQUEST_KEY:
            return self._abort(request_id, "XY悬空定位收到未知请求")
        if str(request.get("target_name") or "") != self.target_name:
            return self._abort(request_id, "执行请求目标与锁定硅片不一致")
        if str(request.get("calibration_sha256") or "").upper() != self.calibration_hash:
            return self._abort(request_id, "平面手眼标定hash在会话中发生变化")
        if self.movement_count >= MAXIMUM_MOVEMENT_COUNT:
            return self._abort(request_id, "达到最大XY移动次数，未继续运动")

        try:
            window = self._window(request, samples)
            image_names = self._save_images(samples, request_id)
        except Exception as exc:
            return self._abort(request_id, f"五帧视觉证据无效：{exc}")
        window["evidence_image_filenames"] = image_names
        gates = dict(window["gates"])
        gates["post_motion_progress"] = self._progress_gate(
            float(window["distance_median_mm"])
        )
        if not all(gate.get("passed") is True for gate in gates.values()):
            window["safety_gates"] = gates
            return self._abort(request_id, "XY悬空定位五帧质量门拒绝", window)

        distance = float(window["distance_median_mm"])
        if (
            distance <= ARRIVAL_MEDIAN_MM
            and float(window["distance_maximum_mm"]) <= ARRIVAL_MAXIMUM_MM
            and float(window["distance_rms_mm"]) <= ARRIVAL_RMS_MM
        ):
            self.status = "arrived_above_selected_wafer"
            self.result_message = (
                f"吸盘XY已到达{self.target_name}正上方；J3未下降，真空未开启"
            )
            self.final_hold = {**window, "safety_gates": gates}
            self._save()
            return {
                "request_id": request_id,
                "decision": "complete",
                "calibration_sha256": self.calibration_hash,
                "reason": self.result_message,
                "evaluation": self.final_hold,
            }

        state = request.get("controller_state") or {}
        current_joints = _vector(state.get("joints"), 4, "request controller joints")
        current_pose = _vector(state.get("pose"), 6, "request controller pose")
        target_xy = _vector(window["target_world_xy_mm"], 2, "target world XY")
        error = target_xy - current_pose[:2]
        error_norm = float(np.linalg.norm(error))
        gain, maximum_step = _step_policy(error_norm)
        command = error * gain
        command_norm = float(np.linalg.norm(command))
        if command_norm > maximum_step:
            command *= maximum_step / command_norm
            command_norm = maximum_step
        if self.cumulative_path_mm + command_norm > MAXIMUM_CUMULATIVE_PATH_MM:
            return self._abort(request_id, "达到累计XY路径上限，未继续运动", window)

        try:
            plan = plan_fixed_rz_xy_step(
                current_joints.astype(float).tolist(),
                current_pose.astype(float).tolist(),
                command.astype(float).tolist(),
                anchor_robot_xy_mm=self.anchor_robot_xy_mm,
                local_extent_mm=LOCAL_EXTENT_MM,
                domain_margin_mm=DOMAIN_MARGIN_MM,
                required_j3_mm=self.required_j3_mm,
                j3_tolerance_mm=0.20,
                required_rz_deg=self.required_rz_deg,
                rz_tolerance_deg=0.30,
                target_rz_tolerance_deg=0.15,
                max_xy_step_norm_mm=MAXIMUM_STEP_MM,
                max_xy_axis_mm=MAXIMUM_STEP_MM,
                max_sequential_transient_xy_mm=5.0,
                max_sequential_transient_rz_deg=1.0,
                precompensate_rz=True,
                enforce_sequential_intermediate_domain=False,
            )
        except Exception as exc:
            return self._abort(request_id, f"XY关节规划安全门拒绝：{exc}", window)

        gates["kinematic_planner"] = _gate(
            (plan.get("audit") or {}).get("passed") is True,
            (plan.get("audit") or {}).get("passed"),
            "true",
        )
        proposal = {
            "proposal_id": f"wafer-pick-{self.movement_count + 1:03d}-{request_id}",
            "target_name": self.target_name,
            "phase": "wafer_pick_xy_overhead",
            "motion_authorized": all(
                gate.get("passed") is True for gate in gates.values()
            ),
            "xy_only": True,
            "z_motion_authorized": False,
            "vacuum_authorized": False,
            "do_authorized": False,
            "locked_j3_mm": self.required_j3_mm,
            "locked_rz_deg": self.required_rz_deg,
            "calculation": {
                "commanded_correction_xy_mm": command.astype(float).tolist(),
                "target_world_xy_mm": target_xy.astype(float).tolist(),
                "current_world_xy_mm": current_pose[:2].astype(float).tolist(),
                "distance_before_mm": error_norm,
            },
            "commanded_correction_xy_mm": command.astype(float).tolist(),
            "predicted_endpoint_xy_mm": (
                current_pose[:2] + command
            ).astype(float).tolist(),
            "window": window,
            "safety_gates": gates,
            "planner": plan,
            "movement_index": self.movement_count + 1,
            "cumulative_path_after_mm": self.cumulative_path_mm + command_norm,
        }
        if proposal["motion_authorized"] is not True:
            return self._abort(request_id, "XY悬空定位综合安全门拒绝", proposal)

        self.movement_count += 1
        self.cumulative_path_mm += command_norm
        self.previous_distance_mm = distance
        self.previous_command_norm_mm = command_norm
        self.iterations.append(proposal)
        self.status = "xy_motion_active"
        self.result_message = (
            f"第{self.movement_count}步候选已生成；等待ActionWorker独立复核"
        )
        self._save()
        return {
            "request_id": request_id,
            "decision": "approve",
            "calibration_sha256": self.calibration_hash,
            "proposal": proposal,
            "target_joints": list(plan["target_joints"]),
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
        response: dict[str, Any] = {
            "request_id": str(request_id),
            "decision": "abort",
            "reason": str(reason),
        }
        if evaluation is not None:
            response["evaluation"] = dict(evaluation)
        return response

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "stage": "selected_wafer_xy_overhead_positioning",
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            ),
            "result_message": self.result_message,
            "target_name": self.target_name,
            "locked_inputs": {
                "planar_handeye_path": str(self.handeye.get("_source_path") or ""),
                "planar_handeye_sha256": self.calibration_hash,
                "tray_geometry_path": str(self.geometry_path),
                "tray_geometry_sha256": self.geometry_hash,
                "suction_target_path": str(self.suction.source_path),
                "suction_target_sha256": self.suction.source_sha256,
            },
            "proposal_count": self.movement_count,
            "maximum_proposal_count": MAXIMUM_MOVEMENT_COUNT,
            "cumulative_path_mm": self.cumulative_path_mm,
            "maximum_cumulative_path_mm": MAXIMUM_CUMULATIVE_PATH_MM,
            "iterations": self.iterations,
            "final_hold": self.final_hold,
            "evidence_images": self.evidence_images,
            "safety_boundary": {
                "xy_only": True,
                "fixed_j3_mm": self.required_j3_mm,
                "fixed_absolute_rz_deg": self.required_rz_deg,
                "maximum_xy_step_mm": MAXIMUM_STEP_MM,
                "tray_center_domain_half_width_mm": (
                    LOCAL_EXTENT_MM - DOMAIN_MARGIN_MM
                ),
                "z_motion": False,
                "descent": False,
                "do_or_vacuum": False,
                "hardware_owner": "ActionWorker only",
            },
        }

    def _save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self.report_path,
            json.dumps(self._payload(), ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )

    def finish(self, ok: bool, message: str) -> tuple[bool, str]:
        if self.status == "arrived_above_selected_wafer" and bool(ok):
            final_ok = True
            final_message = self.result_message
        else:
            final_ok = False
            if self.status == "arrived_above_selected_wafer":
                self.status = "worker_failed_after_visual_arrival"
                self.result_message = str(message)
            elif self.status not in {"safety_rejected", "stopped"}:
                self.status = "stopped" if not ok else "not_converged"
            self.result_message = self.result_message or str(message)
            final_message = self.result_message
        manifest_path = self.output_dir / "points.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            manifest["wafer_pick_xy_positioning"] = {
                "status": self.status,
                "target_name": self.target_name,
                "result_file": RESULT_FILENAME,
                "proposal_count": self.movement_count,
                "arrived_above_selected_wafer": final_ok,
                "descent_executed": False,
                "vacuum_or_do_executed": False,
            }
            atomic_write_text(
                manifest_path,
                json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        self._save()
        return final_ok, final_message


__all__ = [
    "RESULT_FILENAME",
    "WAFER_PICK_XY_RUNTIME_REQUEST_KEY",
    "WaferPickXYPositioningSession",
]
