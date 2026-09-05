"""Fail-closed XY positioning and camera-2 J4 tray alignment.

This module owns no camera, Qt widget, or robot controller.  Camera 1 proves the
source and ``W<-T`` before arming.  During XY motion it still supplies two
current marker-pose observations, but temporary wafer occlusion cannot change
the selected slot and the accepted ``W<-T`` is never overwritten.  The
``ActionWorker`` remains the sole hardware owner and repeats controller and
kinematic checks immediately before every physical move.  After XY arrival,
camera 2 supplies a new two-frame visual angle window; controller Rz alone is
never accepted as proof that the tray is visually square.
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
from scara.pipeline.kinematics import j4_for_rz, rz_of
from scara.pipeline.xy_correction_planner import (
    angular_difference_deg,
    audit_j4_only_orientation_target,
    plan_fixed_rz_xy_step,
)

from .handeye_interaction import load_latest_suction_target, sha256_file
from .moved_tray_servo import registered_slot_world_xy_mm
from .runtime_tray_registration import estimate_transform_W_T, load_planar_handeye
from .tray_pose_estimator import load_tray_board_geometry


WAFER_PICK_XY_RUNTIME_REQUEST_KEY = "wafer_pick_xy_overhead_positioning"
RESULT_FILENAME = "wafer_pick_xy_positioning.json"
RUNTIME_ACTION_SLOTS = 120
OBSERVATION_WINDOW_SIZE = 2
MAXIMUM_OBSERVATION_GAP_S = 1.75
MAXIMUM_OBSERVATION_WINDOW_SPAN_S = 2.0
MAXIMUM_MOVEMENT_COUNT = 100
MAXIMUM_CUMULATIVE_PATH_MM = 200.0
MAXIMUM_STEP_MM = 10.0
CONTROLLER_QUANTIZATION_HEADROOM_MM = 0.01
MAXIMUM_SEQUENTIAL_TRANSIENT_XY_MM = 8.0
MAXIMUM_SEQUENTIAL_TRANSIENT_RZ_DEG = 5.0
MAXIMUM_FINAL_J4_ROTATION_DEG = 30.0
MAXIMUM_CAMERA2_J4_STEP_DEG = 10.0
MAXIMUM_CAMERA2_J4_CORRECTIONS = 3
CAMERA2_ALIGNMENT_MEDIAN_DEG = 0.50
CAMERA2_ALIGNMENT_MAXIMUM_DEG = 0.75
CAMERA2_MAXIMUM_TWO_FRAME_SPREAD_DEG = 0.60
LOCAL_EXTENT_MM = 70.0
DOMAIN_MARGIN_MM = 5.0
ARRIVAL_MEDIAN_MM = 0.50
ARRIVAL_MAXIMUM_MM = 0.80
ARRIVAL_RMS_MM = 0.25
TRAY_MOVEMENT_TRANSLATION_LIMIT_MM = 3.0
TRAY_MOVEMENT_YAW_LIMIT_DEG = 1.0


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


def _signed_angle_deg(value: float) -> float:
    return float((float(value) + 180.0) % 360.0 - 180.0)


def _square_axis_angle_deg(value: float) -> float:
    return float((float(value) + 45.0) % 90.0 - 45.0)


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
    if distance > 15.0:
        return 0.85, 10.0
    if distance > 5.0:
        return 0.80, 5.0
    if distance > 2.0:
        return 0.75, 2.0
    if distance > 0.75:
        return 0.65, 0.50
    return 0.50, 0.25


def select_bounded_observation_window(
    samples: Sequence[Mapping[str, Any]],
    *,
    requested_monotonic_s: float,
) -> list[Mapping[str, Any]]:
    """Select two accepted observations while tolerating brief dropouts.

    Rejected frames are never used as evidence.  They may occur between good
    frames only while the accepted-to-accepted gap and total window span stay
    tightly bounded.  A longer interruption starts a new evidence window.
    """

    requested = float(requested_monotonic_s)
    if not math.isfinite(requested):
        return []
    accepted: list[Mapping[str, Any]] = []
    for sample in samples:
        try:
            captured = float(sample.get("captured_monotonic_s", math.nan))
        except (TypeError, ValueError, OverflowError):
            captured = math.nan
        if not math.isfinite(captured) or captured < requested:
            continue
        if accepted:
            previous = float(accepted[-1]["captured_monotonic_s"])
            if (
                captured <= previous
                or captured - previous > MAXIMUM_OBSERVATION_GAP_S
            ):
                accepted.clear()
        if sample.get("accepted") is not True:
            continue
        accepted.append(sample)
        while (
            accepted
            and captured - float(accepted[0]["captured_monotonic_s"])
            > MAXIMUM_OBSERVATION_WINDOW_SPAN_S
        ):
            accepted.pop(0)
        if len(accepted) >= OBSERVATION_WINDOW_SIZE:
            return accepted[-OBSERVATION_WINDOW_SIZE:]
    return accepted[-OBSERVATION_WINDOW_SIZE:]


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
        self.locked_target_world_xy_mm = registered_slot_world_xy_mm(
            self.geometry,
            self.target_name,
            registration,
        )
        self.initial_registration_yaw_deg = float(
            registration.get("yaw_world_from_tray_deg", math.nan)
        )
        if not math.isfinite(self.initial_registration_yaw_deg):
            raise RuntimeError("启动时W<-T缺少有效yaw")
        # This is only a normalization placeholder required by the task schema.
        # The real final Rz is measured after XY arrival from two fresh camera-2
        # frames; camera-1 world yaw is not camera-2 visual alignment evidence.
        self.final_tray_rz_deg = self.initial_registration_yaw_deg
        source_consensus = initial_snapshot.get("source_consensus") or {}
        selection_gates = initial_snapshot.get("selection_gates") or {}
        source_proof_latched = bool(
            source_consensus.get("qualification_latched") is True
            or (
                source_state.get("state") == "occupied"
                and source_consensus.get("passed") is True
            )
        )
        self.source_confirmed_at_arm = bool(
            source_proof_latched
            and selection_gates
            and all(
                isinstance(gate, Mapping) and gate.get("passed") is True
                for gate in selection_gates.values()
            )
        )
        if not self.source_confirmed_at_arm:
            raise RuntimeError("启动时没有已锁存的两帧occupied硅片证据")
        self.locked_acquisition_evidence = {
            "frame_sequence": initial_snapshot.get("latest_frame_sequence"),
            "frame_captured_monotonic_s": initial_snapshot.get(
                "latest_frame_captured_monotonic_s"
            ),
            "source_slot": source_slot,
            "source_state": source_state.get("state"),
            "occupied_frame_count": source_consensus.get("occupied_frame_count"),
            "required_occupied_frame_count": source_consensus.get(
                "required_occupied_frame_count"
            ),
            "qualification_latched": source_consensus.get(
                "qualification_latched"
            ),
            "qualification_frame_sequence": source_consensus.get(
                "qualification_frame_sequence"
            ),
            "all_start_gates_passed": True,
        }

        if abs(self.initial_joints[2] - self.required_j3_mm) > 0.20:
            raise RuntimeError(
                "当前J3不在相机1安全观察高度；本功能不会自动升降机械臂"
            )
        current_rz = rz_of(
            self.initial_joints[0], self.initial_joints[1], self.initial_joints[3]
        )
        self.initial_rz_deg = float(current_rz)
        self.prealignment_required = bool(
            angular_difference_deg(current_rz, self.required_rz_deg) > 0.30
        )
        self.prealignment_target_joints = self.initial_joints.astype(float).copy()
        self.prealignment_target_joints[3] = j4_for_rz(
            self.initial_joints[0],
            self.initial_joints[1],
            self.required_rz_deg,
        )
        self.prealignment_audit: dict[str, Any] | None = None
        if self.prealignment_required:
            self.prealignment_audit = audit_j4_only_orientation_target(
                self.initial_joints.astype(float).tolist(),
                self.initial_pose.astype(float).tolist(),
                self.prealignment_target_joints.astype(float).tolist(),
                anchor_robot_xy_mm=self.anchor_robot_xy_mm,
                local_extent_mm=LOCAL_EXTENT_MM,
                domain_margin_mm=DOMAIN_MARGIN_MM,
                required_j3_mm=self.required_j3_mm,
                j3_tolerance_mm=0.20,
                required_start_rz_deg=current_rz,
                start_rz_tolerance_deg=0.15,
                target_rz_deg=self.required_rz_deg,
                target_rz_tolerance_deg=0.15,
                maximum_j4_rotation_deg=MAXIMUM_FINAL_J4_ROTATION_DEG,
            )
            if self.prealignment_audit.get("passed") is not True:
                failed = [
                    name
                    for name, gate in self.prealignment_audit["gates"].items()
                    if gate.get("passed") is not True
                ]
                raise RuntimeError(
                    "当前Rz需要先做J4标定姿态预对齐，但安全复核失败："
                    + ", ".join(failed)
                )

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
        self.final_orientation_commanded = False
        self.camera2_alignment_active = False
        self.camera2_j4_correction_count = 0
        self.camera2_cumulative_j4_correction_deg = 0.0
        self.camera2_alignment_windows: list[dict[str, Any]] = []
        self.camera2_orientation_lock_joints: list[float] | None = None
        self.final_orientation_audit: dict[str, Any] | None = None
        self.final_orientation_proposal: dict[str, Any] | None = None
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
        if self.prealignment_required:
            actions.extend(
                [
                    {
                        "type": "move_joints",
                        "name": f"{self.target_name} 相机1标定Rz预对齐（仅J4）",
                        "joints": self.prealignment_target_joints.astype(
                            float
                        ).tolist(),
                        "tolerance": 0.10,
                        "require_current_j3_mm": self.required_j3_mm,
                        "j3_tolerance_mm": 0.20,
                    },
                    {
                        "type": "assert_joints",
                        "name": f"{self.target_name} J4预对齐到位复核",
                        "joints": self.prealignment_target_joints.astype(
                            float
                        ).tolist(),
                        "tolerance": 0.15,
                    },
                ]
            )
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
                        "final_rz_deg": self.final_tray_rz_deg,
                        "max_j4_rotation_deg": MAXIMUM_FINAL_J4_ROTATION_DEG,
                        "max_xy_step_norm_mm": MAXIMUM_STEP_MM,
                        "max_xy_axis_mm": MAXIMUM_STEP_MM,
                        "j3_tolerance_mm": 0.20,
                        "rz_tolerance_deg": 0.30,
                        "target_rz_tolerance_deg": 0.15,
                        "max_sequential_transient_rz_deg": MAXIMUM_SEQUENTIAL_TRANSIENT_RZ_DEG,
                        "precompensate_rz": True,
                        "max_state_drift_xy_mm": 0.10,
                        "max_state_drift_joint": 0.10,
                        "max_sequential_transient_xy_mm": MAXIMUM_SEQUENTIAL_TRANSIENT_XY_MM,
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
                "相机1在启动前确认硅片并锁定本次W<-T；运动中两帧只复核当前"
                "托盘位姿与真实位移，不因机械臂短暂遮挡硅片撤销目标。"
                "远距离单步最多9.99mm，接近目标时自动减小；XY阶段固定J3安全"
                "观察高度；必要时先仅转J4对齐标定Rz。XY到位后由相机2连续两帧"
                "测量画面方向，只转J4闭环修正并再次用相机2复核；不执行"
                "下降、吸取、DO或真空。"
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
            raise ValueError("XY悬空定位每轮必须恰好使用2张新鲜画面")
        requested_at = float(request.get("requested_monotonic_s", math.nan))
        if not math.isfinite(requested_at):
            raise ValueError("执行请求缺少有效时间戳")
        request_state = request.get("controller_state") or {}
        request_joints = _vector(
            request_state.get("joints"), 4, "request controller joints"
        )
        request_pose = _vector(
            request_state.get("pose"), 6, "request controller pose"
        )

        observed_target_world: list[np.ndarray] = []
        robot_xy: list[np.ndarray] = []
        distances: list[float] = []
        origins: list[np.ndarray] = []
        yaws: list[float] = []
        rms_values: list[float] = []
        all_gates_pass = True
        all_accepted = True
        source_locked_at_arm = self.source_confirmed_at_arm
        all_fresh = True
        sequences: list[int] = []
        captured_times: list[float] = []
        all_target_locked = True
        all_states_valid = bool(
            abs(request_joints[2] - self.required_j3_mm) <= 0.20
            and angular_difference_deg(
                rz_of(request_joints[0], request_joints[1], request_joints[3]),
                self.required_rz_deg,
            )
            <= 0.30
        )

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
            gates = sample.get("motion_gates") or sample.get("selection_gates") or {}
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
            # The ActionWorker request is captured immediately before it waits
            # for this observation window, so the robot is stationary here.
            # Recombine that authoritative state with each current C<-T pose
            # to monitor genuine tray movement without using the lagging UI
            # robot-state cache.  Legacy recorded samples without C<-T retain
            # their stored registration only for offline compatibility.
            tray_transform = sample.get("tray_transform_C_T")
            if tray_transform is None:
                candidate = registration
            else:
                candidate = estimate_transform_W_T(
                    tray_transform,
                    request_joints,
                    request_pose[:2],
                    self.handeye.get("R_F_C"),
                    self.suction.p_C_S_mm,
                )
                candidate = {"status": "success", **candidate}
            candidate_transform = _transform(
                candidate.get("transform_W_T"), "observed transform_W_T"
            )
            point = registered_slot_world_xy_mm(
                self.geometry, self.target_name, candidate
            )
            observed_target_world.append(point)
            robot_xy.append(request_pose[:2])
            # The robot follows the one W<-T transform accepted when this
            # session was armed.  Later frames are independent drift evidence,
            # not permission to move the target every time PnP jitters.
            distances.append(
                float(
                    np.linalg.norm(
                        self.locked_target_world_xy_mm - request_pose[:2]
                    )
                )
            )
            origins.append(candidate_transform[:2, 3])
            yaws.append(_yaw_deg(candidate_transform))
            rms_values.append(float(sample.get("reprojection_rms_px", math.inf)))

        target_array = np.asarray(observed_target_world, dtype=np.float64)
        robot_array = np.asarray(robot_xy, dtype=np.float64)
        origin_array = np.asarray(origins, dtype=np.float64)
        observed_target_median = np.median(target_array, axis=0)
        robot_median = np.median(robot_array, axis=0)
        distance_array = np.asarray(distances, dtype=np.float64)
        target_deviation = np.linalg.norm(
            target_array - observed_target_median, axis=1
        )
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
        observation_gaps = [
            captured_times[index] - captured_times[index - 1]
            for index in range(1, len(captured_times))
        ]
        observation_span = captured_times[-1] - captured_times[0]

        gates = {
            "all_frames_explicitly_accepted": _gate(
                all_accepted, all_accepted, "true for both frames"
            ),
            "fresh_post_request_frames": _gate(
                all_fresh, [float(row.get("captured_monotonic_s", math.nan)) for row in rows], ">= request timestamp"
            ),
            "distinct_ordered_frames": _gate(
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
                "two unique frame sequences with strictly increasing capture times",
            ),
            "bounded_observation_timing": _gate(
                observation_span <= MAXIMUM_OBSERVATION_WINDOW_SPAN_S
                and all(
                    0.0 < gap <= MAXIMUM_OBSERVATION_GAP_S
                    for gap in observation_gaps
                ),
                {
                    "adjacent_accepted_frame_gaps_s": observation_gaps,
                    "window_span_s": observation_span,
                },
                (
                    f"each accepted gap <= {MAXIMUM_OBSERVATION_GAP_S:.2f}s and "
                    f"window span <= {MAXIMUM_OBSERVATION_WINDOW_SPAN_S:.1f}s"
                ),
                "failed frames may be skipped only inside this bounded interval",
            ),
            "target_name_locked": _gate(
                all_target_locked,
                [str(row.get("target_name") or "") for row in rows],
                f"all {self.target_name}",
            ),
            "source_locked_normal_occupied_at_arm": _gate(
                source_locked_at_arm,
                self.locked_acquisition_evidence,
                "occupied and all source acquisition gates PASS before motion",
                "移动中允许硅片离开视野或被机械臂遮挡，不重复改变已锁定槽位",
            ),
            "current_overview_motion_gates": _gate(
                all_gates_pass, all_gates_pass, "all PASS in both frames"
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
            "target_world_xy_mm": self.locked_target_world_xy_mm.astype(
                float
            ).tolist(),
            "observed_target_world_xy_mm": observed_target_median.astype(
                float
            ).tolist(),
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
            "每次实际移动后，下一独立两帧必须证明距离继续减小",
        )

    def _camera2_window(
        self,
        request: Mapping[str, Any],
        samples: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        rows = list(samples)
        if len(rows) != OBSERVATION_WINDOW_SIZE:
            raise ValueError("相机2方向复核必须恰好使用2张新鲜画面")
        requested_at = float(request.get("requested_monotonic_s", math.nan))
        sequences = [int(row.get("frame_sequence", -1)) for row in rows]
        timestamps = [
            float(row.get("captured_monotonic_s", math.nan)) for row in rows
        ]
        angles = [float(row.get("angle_error_deg", math.nan)) for row in rows]
        source_ok = all(int(row.get("camera_source", -1)) == 2 for row in rows)
        accepted = all(row.get("accepted") is True for row in rows)
        finite = bool(
            math.isfinite(requested_at)
            and all(math.isfinite(value) for value in timestamps + angles)
        )
        ordered = bool(
            len(set(sequences)) == OBSERVATION_WINDOW_SIZE
            and sequences[1] > sequences[0]
            and finite
            and timestamps[1] > timestamps[0]
        )
        fresh = bool(finite and all(value >= requested_at for value in timestamps))
        span = timestamps[-1] - timestamps[0] if finite else math.inf
        spread = abs(_square_axis_angle_deg(angles[1] - angles[0])) if finite else math.inf
        median_angle = float(np.median(angles)) if finite else math.nan
        maximum_abs_angle = max(abs(value) for value in angles) if finite else math.inf
        gates = {
            "camera2_source_identity": _gate(source_ok, source_ok, "true"),
            "all_frames_explicitly_accepted": _gate(accepted, accepted, "true"),
            "distinct_ordered_frames": _gate(
                ordered, sequences, "two unique increasing frame sequences"
            ),
            "frames_captured_after_request": _gate(
                fresh,
                timestamps,
                f">={requested_at:.6f}",
            ),
            "two_frame_window_span": _gate(
                finite and 0.0 < span <= MAXIMUM_OBSERVATION_WINDOW_SPAN_S,
                span,
                f"0..{MAXIMUM_OBSERVATION_WINDOW_SPAN_S:.2f} s",
            ),
            "two_frame_angle_consistency": _gate(
                finite and spread <= CAMERA2_MAXIMUM_TWO_FRAME_SPREAD_DEG,
                spread,
                f"<={CAMERA2_MAXIMUM_TWO_FRAME_SPREAD_DEG:.2f} deg",
            ),
        }
        return {
            "camera_source": 2,
            "measurement_ids": [str(row.get("measurement_id") or "") for row in rows],
            "frame_sequences": sequences,
            "captured_monotonic_s": timestamps,
            "angle_error_deg": angles,
            "median_angle_error_deg": median_angle,
            "maximum_absolute_angle_error_deg": maximum_abs_angle,
            "two_frame_spread_deg": spread,
            "marker_ids": [list(row.get("marker_ids") or []) for row in rows],
            "gates": gates,
        }

    def _build_camera2_response(
        self,
        request: Mapping[str, Any],
        samples: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        request_id = str(request.get("request_id") or "")
        try:
            window = self._camera2_window(request, samples)
            image_names = self._save_images(samples, f"cam2-{request_id}")
            current_state = request.get("controller_state") or {}
            current_joints = _vector(
                current_state.get("joints"), 4, "camera2 request controller joints"
            )
            current_pose = _vector(
                current_state.get("pose"), 6, "camera2 request controller pose"
            )
        except Exception as exc:
            return self._abort(request_id, f"相机2两帧方向证据无效：{exc}")
        window["evidence_image_filenames"] = image_names
        gates = dict(window["gates"])
        locked = _vector(
            self.camera2_orientation_lock_joints,
            4,
            "camera2 orientation lock joints",
        )
        locked_axis_errors = np.abs(current_joints[:3] - locked[:3])
        gates["j1_j2_j3_remain_locked"] = _gate(
            bool(np.all(locked_axis_errors <= 0.20)),
            locked_axis_errors.astype(float).tolist(),
            "each <=0.20 deg/mm",
        )
        if not all(gate.get("passed") is True for gate in gates.values()):
            window["safety_gates"] = gates
            self.camera2_alignment_windows.append(window)
            self.status = "camera2_alignment_waiting_for_consistent_frames"
            self.result_message = "相机2两帧方向暂不一致；未运动，自动重新观察"
            self._save()
            return {
                "request_id": request_id,
                "decision": "observe",
                "calibration_sha256": self.calibration_hash,
                "camera2_required": True,
                "reason": self.result_message,
                "evaluation": window,
            }

        median_angle = float(window["median_angle_error_deg"])
        maximum_abs_angle = float(window["maximum_absolute_angle_error_deg"])
        current_rz = float(rz_of(current_joints[0], current_joints[1], current_joints[3]))
        aligned = bool(
            abs(median_angle) <= CAMERA2_ALIGNMENT_MEDIAN_DEG
            and maximum_abs_angle <= CAMERA2_ALIGNMENT_MAXIMUM_DEG
        )
        alignment_gate = _gate(
            aligned,
            {
                "median_deg": median_angle,
                "maximum_deg": maximum_abs_angle,
            },
            (
                f"median <={CAMERA2_ALIGNMENT_MEDIAN_DEG:.2f} deg and "
                f"maximum <={CAMERA2_ALIGNMENT_MAXIMUM_DEG:.2f} deg"
            ),
        )
        # Being outside the final angle tolerance is the reason to move J4, not
        # a failed authorization gate.  Keep it as a diagnostic/termination
        # gate on the evidence window; only evidence quality and the J4-only
        # planner belong to the proposal's all-PASS safety gates.
        window["quality_gates"] = dict(gates)
        window["camera2_visual_alignment_gate"] = alignment_gate
        self.camera2_alignment_windows.append(window)

        if aligned:
            final_audit = audit_j4_only_orientation_target(
                current_joints.astype(float).tolist(),
                current_pose.astype(float).tolist(),
                current_joints.astype(float).tolist(),
                anchor_robot_xy_mm=self.anchor_robot_xy_mm,
                local_extent_mm=LOCAL_EXTENT_MM,
                domain_margin_mm=DOMAIN_MARGIN_MM,
                required_j3_mm=self.required_j3_mm,
                j3_tolerance_mm=0.20,
                required_start_rz_deg=current_rz,
                start_rz_tolerance_deg=0.15,
                target_rz_deg=current_rz,
                target_rz_tolerance_deg=0.15,
                maximum_j4_rotation_deg=MAXIMUM_CAMERA2_J4_STEP_DEG,
            )
            if final_audit.get("passed") is not True:
                return self._abort(
                    request_id,
                    "相机2显示已转正，但最终J4-only控制器安全复核失败",
                    {"camera2_window": window, "audit": final_audit},
                )
            self.final_tray_rz_deg = current_rz
            self.final_orientation_audit = {
                "camera2_visual_window": window,
                "controller_j4_only_audit": final_audit,
            }
            self.status = "arrived_above_selected_wafer_and_tray_aligned"
            self.result_message = (
                f"吸盘XY已到达{self.target_name}正上方；相机2连续两帧确认托盘横平竖直"
                f"（中值误差{median_angle:+.3f}°，最大误差{maximum_abs_angle:.3f}°）；"
                "J1/J2/J3保持不动，真空未开启"
            )
            self._save()
            return {
                "request_id": request_id,
                "decision": "complete",
                "calibration_sha256": self.calibration_hash,
                "reason": self.result_message,
                "evaluation": self.final_orientation_audit,
            }

        if self.camera2_j4_correction_count >= MAXIMUM_CAMERA2_J4_CORRECTIONS:
            return self._abort(
                request_id,
                "相机2经过最大J4修正次数后仍未确认托盘转正",
                window,
            )
        remaining_budget = (
            MAXIMUM_FINAL_J4_ROTATION_DEG
            - self.camera2_cumulative_j4_correction_deg
        )
        correction = max(
            -MAXIMUM_CAMERA2_J4_STEP_DEG,
            min(MAXIMUM_CAMERA2_J4_STEP_DEG, -median_angle),
        )
        if remaining_budget <= 0.0 or abs(correction) > remaining_budget + 1e-9:
            return self._abort(request_id, "相机2 J4累计修正超过安全上限", window)
        target_rz = _signed_angle_deg(current_rz + correction)
        target_joints = current_joints.copy()
        target_joints[3] = j4_for_rz(
            current_joints[0], current_joints[1], target_rz
        )
        orientation_audit = audit_j4_only_orientation_target(
            current_joints.astype(float).tolist(),
            current_pose.astype(float).tolist(),
            target_joints.astype(float).tolist(),
            anchor_robot_xy_mm=self.anchor_robot_xy_mm,
            local_extent_mm=LOCAL_EXTENT_MM,
            domain_margin_mm=DOMAIN_MARGIN_MM,
            required_j3_mm=self.required_j3_mm,
            j3_tolerance_mm=0.20,
            required_start_rz_deg=current_rz,
            start_rz_tolerance_deg=0.15,
            target_rz_deg=target_rz,
            target_rz_tolerance_deg=0.15,
            maximum_j4_rotation_deg=MAXIMUM_CAMERA2_J4_STEP_DEG,
        )
        motion_gates = dict(gates)
        motion_gates["camera2_correction_required"] = _gate(
            not aligned,
            {
                "median_deg": median_angle,
                "maximum_deg": maximum_abs_angle,
            },
            "outside final visual tolerance",
        )
        motion_gates["camera2_j4_only_planner"] = _gate(
            orientation_audit.get("passed") is True,
            orientation_audit.get("passed"),
            "true",
        )
        if orientation_audit.get("passed") is not True:
            return self._abort(
                request_id,
                "相机2 J4-only修正规划未通过安全复核",
                {"camera2_window": window, "audit": orientation_audit},
            )
        proposal = {
            "proposal_id": f"wafer-pick-camera2-j4-{request_id}",
            "target_name": self.target_name,
            "phase": "wafer_pick_final_tray_orientation",
            "motion_authorized": True,
            "xy_only": False,
            "j4_only": True,
            "z_motion_authorized": False,
            "vacuum_authorized": False,
            "do_authorized": False,
            "locked_j3_mm": self.required_j3_mm,
            "locked_rz_deg": target_rz,
            "calculation": {
                "commanded_correction_xy_mm": [0.0, 0.0],
                "camera2_median_angle_error_deg": median_angle,
                "camera2_control_law": "target_rz = current_rz - image_angle",
                "commanded_j4_correction_deg": correction,
                "current_absolute_rz_deg": current_rz,
                "target_absolute_rz_deg": target_rz,
            },
            "commanded_correction_xy_mm": [0.0, 0.0],
            "predicted_endpoint_xy_mm": current_pose[:2].astype(float).tolist(),
            "window": window,
            "safety_gates": motion_gates,
            "planner": {
                "target_joints": target_joints.astype(float).tolist(),
                "audit": orientation_audit,
            },
            "movement_index": self.movement_count + self.camera2_j4_correction_count + 1,
            "cumulative_path_after_mm": self.cumulative_path_mm,
        }
        self.camera2_j4_correction_count += 1
        self.camera2_cumulative_j4_correction_deg += abs(correction)
        self.final_tray_rz_deg = target_rz
        self.final_orientation_commanded = True
        self.final_orientation_proposal = proposal
        self.status = "camera2_j4_correction_pending_visual_verification"
        self.result_message = (
            f"相机2两帧测得托盘方向误差{median_angle:+.3f}°；等待仅J4修正"
            f"{correction:+.3f}°，随后必须再次通过相机2两帧复核"
        )
        self._save()
        return {
            "request_id": request_id,
            "decision": "approve",
            "calibration_sha256": self.calibration_hash,
            "camera2_required": True,
            "proposal": proposal,
            "target_joints": target_joints.astype(float).tolist(),
        }

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

        if self.camera2_alignment_active:
            return self._build_camera2_response(request, samples)
        if self.movement_count >= MAXIMUM_MOVEMENT_COUNT:
            return self._abort(request_id, "达到最大XY移动次数，未继续运动")

        try:
            window = self._window(request, samples)
            image_names = self._save_images(samples, request_id)
        except Exception as exc:
            return self._abort(request_id, f"两帧视觉证据无效：{exc}")
        window["evidence_image_filenames"] = image_names
        gates = dict(window["gates"])
        gates["post_motion_progress"] = self._progress_gate(
            float(window["distance_median_mm"])
        )
        if not all(gate.get("passed") is True for gate in gates.values()):
            window["safety_gates"] = gates
            return self._abort(request_id, "XY悬空定位两帧质量门拒绝", window)

        distance = float(window["distance_median_mm"])
        if (
            distance <= ARRIVAL_MEDIAN_MM
            and float(window["distance_maximum_mm"]) <= ARRIVAL_MAXIMUM_MM
            and float(window["distance_rms_mm"]) <= ARRIVAL_RMS_MM
        ):
            self.final_hold = {**window, "safety_gates": gates}
            state = request.get("controller_state") or {}
            try:
                current_joints = _vector(
                    state.get("joints"), 4, "request controller joints"
                )
            except Exception as exc:
                return self._abort(
                    request_id, f"切换相机2前的关节锁定失败：{exc}", window
                )
            self.camera2_orientation_lock_joints = current_joints.astype(float).tolist()
            self.camera2_alignment_active = True
            self.status = "xy_arrived_waiting_for_camera2_alignment"
            self.result_message = (
                f"{self.target_name}的XY已到位；保持J1/J2/J3不动，切换相机2并等待"
                "连续两帧托盘方向证据；当前尚未授权J4运动"
            )
            self._save()
            return {
                "request_id": request_id,
                "decision": "observe",
                "calibration_sha256": self.calibration_hash,
                "camera2_required": True,
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
        maximum_step = min(
            maximum_step,
            MAXIMUM_STEP_MM - CONTROLLER_QUANTIZATION_HEADROOM_MM,
        )
        command = error * gain
        command_norm = float(np.linalg.norm(command))
        if command_norm > maximum_step:
            command *= maximum_step / command_norm
            command_norm = maximum_step
        base_command = command.copy()
        plan: dict[str, Any] | None = None
        planner_backoff_scale = 1.0
        planning_failures: list[str] = []
        # A 10 mm Cartesian endpoint can cause a larger J1-then-J2 transient
        # sweep in some arm configurations.  Keep the strict 8 mm transient
        # gate and automatically shorten only that candidate instead of
        # aborting the whole positioning session.
        for scale in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.25):
            command = base_command * scale
            command_norm = float(np.linalg.norm(command))
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
                    max_sequential_transient_xy_mm=MAXIMUM_SEQUENTIAL_TRANSIENT_XY_MM,
                    max_sequential_transient_rz_deg=MAXIMUM_SEQUENTIAL_TRANSIENT_RZ_DEG,
                    precompensate_rz=True,
                    enforce_sequential_intermediate_domain=False,
                )
                planner_backoff_scale = scale
                break
            except Exception as exc:
                planning_failures.append(f"scale={scale:.2f}: {exc}")
        if plan is None:
            reason = planning_failures[-1] if planning_failures else "未知规划失败"
            return self._abort(request_id, f"XY关节规划安全门拒绝：{reason}", window)
        if self.cumulative_path_mm + command_norm > MAXIMUM_CUMULATIVE_PATH_MM:
            return self._abort(request_id, "达到累计XY路径上限，未继续运动", window)

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
                "planner_backoff_scale": planner_backoff_scale,
                "rejected_larger_candidate_count": len(planning_failures),
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
            "final_orientation_proposal": self.final_orientation_proposal,
            "final_orientation_audit": self.final_orientation_audit,
            "camera2_alignment_active": self.camera2_alignment_active,
            "camera2_j4_correction_count": self.camera2_j4_correction_count,
            "camera2_cumulative_j4_correction_deg": (
                self.camera2_cumulative_j4_correction_deg
            ),
            "camera2_alignment_windows": self.camera2_alignment_windows,
            "evidence_images": self.evidence_images,
            "safety_boundary": {
                "xy_only_during_positioning": True,
                "fixed_j3_mm": self.required_j3_mm,
                "fixed_absolute_rz_during_xy_deg": self.required_rz_deg,
                "initial_absolute_rz_deg": self.initial_rz_deg,
                "j4_only_calibration_rz_prealignment": (
                    self.prealignment_required
                ),
                "prealignment_audit": self.prealignment_audit,
                "final_j4_only_alignment": True,
                "final_tray_absolute_rz_deg": self.final_tray_rz_deg,
                "final_alignment_evidence": "two fresh camera2 frames",
                "camera2_alignment_median_tolerance_deg": (
                    CAMERA2_ALIGNMENT_MEDIAN_DEG
                ),
                "camera2_alignment_maximum_tolerance_deg": (
                    CAMERA2_ALIGNMENT_MAXIMUM_DEG
                ),
                "maximum_camera2_j4_step_deg": MAXIMUM_CAMERA2_J4_STEP_DEG,
                "maximum_camera2_j4_corrections": MAXIMUM_CAMERA2_J4_CORRECTIONS,
                "maximum_final_j4_rotation_deg": MAXIMUM_FINAL_J4_ROTATION_DEG,
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
        successful_status = "arrived_above_selected_wafer_and_tray_aligned"
        if self.status == successful_status and bool(ok):
            final_ok = True
            final_message = self.result_message
        else:
            final_ok = False
            if self.status in {
                "final_j4_tray_alignment_pending",
                "xy_arrived_waiting_for_camera2_alignment",
                "camera2_alignment_waiting_for_consistent_frames",
                "camera2_j4_correction_pending_visual_verification",
                successful_status,
            }:
                self.status = "worker_failed_during_final_j4_alignment"
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
                "tray_orientation_aligned_by_j4": final_ok,
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
