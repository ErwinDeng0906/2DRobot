"""Execute imported SCARA action files without blocking the Qt UI.

The action format deliberately separates movement, dwell, still capture, video
recording, and state recording.  An imported Python file only *describes* those
operations; hardware access happens here after the operator confirms the run.
"""

from __future__ import annotations

import json
import math
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

from PyQt6.QtCore import QThread, pyqtSignal

from scara.config.camera_config import ResolvedCameraSource, resolve_camera_sources
from scara.controller.scara_controller import ScaraController
from scara.file_io import atomic_write_text


ACTION_API_VERSION = 1
SUPPORTED_ACTION_TYPES = {
    "assert_joints",
    "move_joints",
    "runtime_move_joints",
    "move_xyzr",
    "set_do",
    "wait",
    "capture",
    "start_video",
    "stop_video",
    "record_point",
    "operator_checkpoint",
}
# ``runtime_move_joints`` is intentionally much narrower than an ordinary
# imported ``move_joints`` step.  These are engine-enforced Stage-7A ceilings,
# not defaults that an imported task may relax.
RUNTIME_MOVE_MAXIMUM_STEP_NORM_MM = 0.25
RUNTIME_MOVE_MAXIMUM_AXIS_STEP_MM = 0.25
RUNTIME_MOVE_MAXIMUM_LOCAL_EXTENT_MM = 2.0
RUNTIME_MOVE_MINIMUM_DOMAIN_MARGIN_MM = 0.20
RUNTIME_MOVE_MAXIMUM_J3_TOLERANCE_MM = 0.20
RUNTIME_MOVE_MAXIMUM_RZ_TOLERANCE_DEG = 0.20
RUNTIME_MOVE_MAXIMUM_TARGET_RZ_TOLERANCE_DEG = 0.15
RUNTIME_MOVE_MAXIMUM_SEQUENTIAL_TRANSIENT_RZ_DEG = 0.30
RUNTIME_MOVE_MAXIMUM_STATE_DRIFT_XY_MM = 0.05
RUNTIME_MOVE_MAXIMUM_STATE_DRIFT_JOINT = 0.05
RUNTIME_MOVE_MAXIMUM_SEQUENTIAL_TRANSIENT_MM = 0.50
RUNTIME_MOVE_MAXIMUM_MOVE_TOLERANCE = 0.05
RUNTIME_MOVE_MAXIMUM_PROPOSAL_AGE_S = 60.0
RUNTIME_MOVE_MAXIMUM_FK_POSE_MISMATCH_MM = 0.20
STAGE7B_RUNTIME_REQUEST_KEY = "stage7b_p22_finite_loop"
STAGE7B_MAXIMUM_STEP_NORM_MM = 0.75
STAGE7B_MAXIMUM_AXIS_STEP_MM = 0.75
STAGE7B_MAXIMUM_LOCAL_EXTENT_MM = 10.0
STAGE7B_MAXIMUM_SEQUENTIAL_TRANSIENT_MM = 1.50
FULL_TRAY_GEOMETRY_REQUEST_KEY = "full_tray_p22_metric_geometry_correction"
FULL_TRAY_GEOMETRY_MAXIMUM_STEP_NORM_MM = 3.0
FULL_TRAY_GEOMETRY_MAXIMUM_AXIS_STEP_MM = 3.0
FULL_TRAY_GEOMETRY_MAXIMUM_LOCAL_EXTENT_MM = 5.0
FULL_TRAY_GEOMETRY_MAXIMUM_SEQUENTIAL_TRANSIENT_MM = 6.0
FULL_TRAY_GEOMETRY_MAXIMUM_RZ_TOLERANCE_DEG = 0.30
FULL_TRAY_GEOMETRY_MAXIMUM_SEQUENTIAL_TRANSIENT_RZ_DEG = 1.0
FULL_TRAY_GEOMETRY_MAXIMUM_STATE_DRIFT_XY_MM = 0.20
FULL_TRAY_GEOMETRY_MAXIMUM_STATE_DRIFT_JOINT = 0.20
MOVED_TRAY_RUNTIME_REQUEST_KEY = "moved_tray_p22_runtime_positioning"
MOVED_TRAY_MAXIMUM_STEP_NORM_MM = 10.0
MOVED_TRAY_MAXIMUM_AXIS_STEP_MM = 10.0
MOVED_TRAY_MAXIMUM_LOCAL_EXTENT_MM = 130.0
# In moved-tray coarse positioning, 10 mm constrains only the net endpoint.
# J1/J2 retain the controller's sequential motion, so their intermediate TCP
# sweep is recorded and bounded by broad anomaly ceilings rather than by the
# endpoint limit.
MOVED_TRAY_MAXIMUM_SEQUENTIAL_TRANSIENT_MM = 130.0
MOVED_TRAY_MAXIMUM_RZ_TOLERANCE_DEG = 0.30
MOVED_TRAY_MAXIMUM_SEQUENTIAL_TRANSIENT_RZ_DEG = 15.0
MOVED_TRAY_MAXIMUM_STATE_DRIFT_XY_MM = 0.20
MOVED_TRAY_MAXIMUM_STATE_DRIFT_JOINT = 0.20
WAFER_PICK_XY_RUNTIME_REQUEST_KEY = "wafer_pick_xy_overhead_positioning"
WAFER_PICK_XY_MAXIMUM_STEP_NORM_MM = 10.0
WAFER_PICK_XY_MAXIMUM_AXIS_STEP_MM = 10.0
WAFER_PICK_XY_MAXIMUM_LOCAL_EXTENT_MM = 70.0
WAFER_PICK_XY_MAXIMUM_SEQUENTIAL_TRANSIENT_MM = 8.0
WAFER_PICK_XY_MAXIMUM_RZ_TOLERANCE_DEG = 0.30
WAFER_PICK_XY_MAXIMUM_SEQUENTIAL_TRANSIENT_RZ_DEG = 5.0
WAFER_PICK_XY_MAXIMUM_J4_ROTATION_DEG = 30.0
WAFER_PICK_XY_MAXIMUM_STATE_DRIFT_XY_MM = 0.10
WAFER_PICK_XY_MAXIMUM_STATE_DRIFT_JOINT = 0.10
WAFER_PICK_XY_MAXIMUM_SPEED_PERCENT = 20.0


def _finite(value: object, label: str) -> float:
    """Convert one action parameter to a finite float or raise a useful error."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是数字") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} 必须是有限数字")
    return result


def _joint_values(value: object, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{label} 必须包含 J1/J2/J3/J4 四个值")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _vector_values(value: object, length: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != int(length):
        raise ValueError(f"{label} 必须包含 {length} 个数值")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _positive_at_most(
    value: object,
    label: str,
    maximum: float,
) -> float:
    result = _finite(value, label)
    if result <= 0.0 or result > float(maximum) + 1e-12:
        raise ValueError(f"{label} 必须在 (0, {maximum:g}] 范围内")
    return result


def _json_safe_mapping(
    value: object,
    label: str,
    *,
    maximum_characters: int = 1_000_000,
) -> dict:
    """Return a detached JSON mapping or fail closed.

    Runtime proposal objects are persisted into ``points.json``.  Round-trip
    serialization both rejects NaN/custom objects and prevents a GUI runtime
    from mutating the audit payload after it has authorized a request.
    """

    if not isinstance(value, dict):
        raise ValueError(f"{label} 必须是JSON对象")
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} 包含不可序列化或非有限值") from exc
    if len(encoded) > int(maximum_characters):
        raise ValueError(f"{label} 过大，拒绝写入运行记录")
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - guarded above
        raise ValueError(f"{label} 必须是JSON对象")
    return decoded


def _normalize_camera_capture_settings(value: object) -> dict[int, dict[str, object]]:
    """Allow tasks to request auto mode, never a task-owned fixed exposure."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("camera_capture_settings 必须是按相机源编号索引的字典")
    normalized: dict[int, dict[str, object]] = {}
    for raw_source, raw_setting in value.items():
        if isinstance(raw_source, bool):
            raise ValueError("camera_capture_settings 的相机源必须是 0 到 8 的整数")
        try:
            source = int(raw_source)
        except (TypeError, ValueError) as exc:
            raise ValueError("camera_capture_settings 的相机源必须是 0 到 8 的整数") from exc
        if source < 0 or source > 8 or str(raw_source).strip() not in {str(source), f"{source}.0"}:
            raise ValueError("camera_capture_settings 的相机源必须是 0 到 8 的整数")
        if not isinstance(raw_setting, dict) or set(raw_setting) != {"auto_exposure"}:
            raise ValueError(
                f"camera_capture_settings[{source}] 只能指定 auto_exposure=true；"
                "固定曝光只允许由动态演示UI中的人工操作临时设置"
            )
        if raw_setting["auto_exposure"] is not True:
            raise ValueError(
                f"camera_capture_settings[{source}].auto_exposure 必须为 true"
            )
        normalized[source] = {"auto_exposure": True}
    return normalized


def normalize_action_task(raw_task: object) -> dict:
    """Validate an ``ACTION_API_VERSION = 1`` task and return a safe copy."""
    if not isinstance(raw_task, dict):
        raise ValueError("build_action() 必须返回字典")

    task_name = str(raw_task.get("name") or "未命名动作").strip()
    if not task_name:
        raise ValueError("动作 name 不能为空")

    raw_camera = raw_task.get("camera_model") or {}
    if not isinstance(raw_camera, dict):
        raise ValueError("camera_model 必须是字典")
    camera_model = {
        "offset_mm": _finite(raw_camera.get("offset_mm", 20.0), "camera_model.offset_mm"),
        "angle_reference": str(
            raw_camera.get("angle_reference", "world_negative_y")
        ).strip(),
        "positive_rotation": str(
            raw_camera.get("positive_rotation", "counter_clockwise_from_above")
        ).strip(),
    }
    if camera_model["offset_mm"] < 0:
        raise ValueError("camera_model.offset_mm 不能为负数")
    if camera_model["angle_reference"] != "world_negative_y":
        raise ValueError("camera_model.angle_reference 必须是 world_negative_y")
    if camera_model["positive_rotation"] != "counter_clockwise_from_above":
        raise ValueError(
            "camera_model.positive_rotation 必须是 counter_clockwise_from_above"
        )

    raw_actions = raw_task.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError("actions 必须是非空列表")

    actions: list[dict] = []
    active_video_sources: set[int] = set()
    video_filenames: set[str] = set()
    for index, raw in enumerate(raw_actions, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"第 {index} 步必须是字典")
        kind = str(raw.get("type") or "").strip()
        if kind not in SUPPORTED_ACTION_TYPES:
            allowed = ", ".join(sorted(SUPPORTED_ACTION_TYPES))
            raise ValueError(f"第 {index} 步 type={kind!r} 不受支持；允许：{allowed}")

        step: dict = {"type": kind}
        if kind in {
            "assert_joints",
            "move_joints",
            "runtime_move_joints",
            "move_xyzr",
            "set_do",
            "record_point",
            "operator_checkpoint",
        }:
            name = str(raw.get("name") or f"步骤 {index}").strip()
            if not name:
                raise ValueError(f"第 {index} 步 name 不能为空")
            step["name"] = name

        if kind in {"assert_joints", "move_joints"}:
            step["joints"] = _joint_values(raw.get("joints"), f"第 {index} 步 joints")
            step["tolerance"] = _finite(raw.get("tolerance", 0.2), f"第 {index} 步 tolerance")
            if step["tolerance"] <= 0:
                raise ValueError(f"第 {index} 步 tolerance 必须大于 0")
            if kind == "move_joints" and "require_current_j3_mm" in raw:
                step["require_current_j3_mm"] = _finite(
                    raw["require_current_j3_mm"],
                    f"第 {index} 步 require_current_j3_mm",
                )
                step["j3_tolerance_mm"] = _finite(
                    raw.get("j3_tolerance_mm", 0.2),
                    f"第 {index} 步 j3_tolerance_mm",
                )
                if step["j3_tolerance_mm"] <= 0:
                    raise ValueError(f"第 {index} 步 j3_tolerance_mm 必须大于 0")
        elif kind == "runtime_move_joints":
            # A runtime may calculate a target only inside this immutable,
            # normalized envelope.  ActionWorker independently re-reads the
            # controller and audits the returned target before any command.
            request_key = str(raw.get("request_key") or "").strip()
            if not request_key:
                raise ValueError(
                    f"第 {index} 步 runtime_move_joints.request_key 不能为空"
                )
            is_stage7b = request_key == STAGE7B_RUNTIME_REQUEST_KEY
            is_full_tray_geometry = (
                request_key == FULL_TRAY_GEOMETRY_REQUEST_KEY
            )
            is_moved_tray = request_key == MOVED_TRAY_RUNTIME_REQUEST_KEY
            is_wafer_pick_xy = request_key == WAFER_PICK_XY_RUNTIME_REQUEST_KEY
            target_name = str(raw.get("target_name") or "P22").strip()
            is_valid_tray_slot = bool(
                len(target_name) == 3
                and target_name[0] == "P"
                and target_name[1] in "012345"
                and target_name[2] in "012345"
            )
            if is_wafer_pick_xy:
                if not is_valid_tray_slot:
                    raise ValueError(
                        f"第 {index} 步XY悬空定位target_name必须是P00到P55"
                    )
            elif target_name != "P22":
                raise ValueError(
                    f"第 {index} 步该runtime_move_joints请求只允许target_name=P22"
                )
            calibration_sha256 = str(
                raw.get("calibration_sha256") or ""
            ).strip().upper()
            if len(calibration_sha256) != 64 or any(
                character not in "0123456789ABCDEF"
                for character in calibration_sha256
            ):
                raise ValueError(
                    f"第 {index} 步 calibration_sha256 必须是64位SHA-256十六进制"
                )
            fine_calibration_sha256 = str(
                raw.get("fine_calibration_sha256") or ""
            ).strip().upper()
            if is_stage7b and (
                len(fine_calibration_sha256) != 64
                or any(character not in "0123456789ABCDEF" for character in fine_calibration_sha256)
            ):
                raise ValueError(
                    f"第 {index} 步 fine_calibration_sha256 必须是64位SHA-256十六进制"
                )
            maximum_extent = (
                STAGE7B_MAXIMUM_LOCAL_EXTENT_MM
                if is_stage7b
                else (
                    MOVED_TRAY_MAXIMUM_LOCAL_EXTENT_MM
                    if is_moved_tray
                    else (
                        WAFER_PICK_XY_MAXIMUM_LOCAL_EXTENT_MM
                        if is_wafer_pick_xy
                        else (
                            FULL_TRAY_GEOMETRY_MAXIMUM_LOCAL_EXTENT_MM
                            if is_full_tray_geometry
                            else RUNTIME_MOVE_MAXIMUM_LOCAL_EXTENT_MM
                        )
                    )
                )
            )
            maximum_step_norm = (
                STAGE7B_MAXIMUM_STEP_NORM_MM
                if is_stage7b
                else (
                    MOVED_TRAY_MAXIMUM_STEP_NORM_MM
                    if is_moved_tray
                    else (
                        WAFER_PICK_XY_MAXIMUM_STEP_NORM_MM
                        if is_wafer_pick_xy
                        else (
                            FULL_TRAY_GEOMETRY_MAXIMUM_STEP_NORM_MM
                            if is_full_tray_geometry
                            else RUNTIME_MOVE_MAXIMUM_STEP_NORM_MM
                        )
                    )
                )
            )
            maximum_axis_step = (
                STAGE7B_MAXIMUM_AXIS_STEP_MM
                if is_stage7b
                else (
                    MOVED_TRAY_MAXIMUM_AXIS_STEP_MM
                    if is_moved_tray
                    else (
                        WAFER_PICK_XY_MAXIMUM_AXIS_STEP_MM
                        if is_wafer_pick_xy
                        else (
                            FULL_TRAY_GEOMETRY_MAXIMUM_AXIS_STEP_MM
                            if is_full_tray_geometry
                            else RUNTIME_MOVE_MAXIMUM_AXIS_STEP_MM
                        )
                    )
                )
            )
            maximum_transient_xy = (
                STAGE7B_MAXIMUM_SEQUENTIAL_TRANSIENT_MM
                if is_stage7b
                else (
                    MOVED_TRAY_MAXIMUM_SEQUENTIAL_TRANSIENT_MM
                    if is_moved_tray
                    else (
                        WAFER_PICK_XY_MAXIMUM_SEQUENTIAL_TRANSIENT_MM
                        if is_wafer_pick_xy
                        else (
                            FULL_TRAY_GEOMETRY_MAXIMUM_SEQUENTIAL_TRANSIENT_MM
                            if is_full_tray_geometry
                            else RUNTIME_MOVE_MAXIMUM_SEQUENTIAL_TRANSIENT_MM
                        )
                    )
                )
            )
            maximum_rz_tolerance = (
                MOVED_TRAY_MAXIMUM_RZ_TOLERANCE_DEG
                if is_moved_tray
                else RUNTIME_MOVE_MAXIMUM_RZ_TOLERANCE_DEG
            )
            if is_full_tray_geometry:
                maximum_rz_tolerance = FULL_TRAY_GEOMETRY_MAXIMUM_RZ_TOLERANCE_DEG
            if is_wafer_pick_xy:
                maximum_rz_tolerance = WAFER_PICK_XY_MAXIMUM_RZ_TOLERANCE_DEG
            maximum_transient_rz = (
                MOVED_TRAY_MAXIMUM_SEQUENTIAL_TRANSIENT_RZ_DEG
                if is_moved_tray
                else RUNTIME_MOVE_MAXIMUM_SEQUENTIAL_TRANSIENT_RZ_DEG
            )
            if is_full_tray_geometry:
                maximum_transient_rz = FULL_TRAY_GEOMETRY_MAXIMUM_SEQUENTIAL_TRANSIENT_RZ_DEG
            if is_wafer_pick_xy:
                maximum_transient_rz = WAFER_PICK_XY_MAXIMUM_SEQUENTIAL_TRANSIENT_RZ_DEG
            maximum_state_drift_xy = (
                MOVED_TRAY_MAXIMUM_STATE_DRIFT_XY_MM
                if is_moved_tray
                else RUNTIME_MOVE_MAXIMUM_STATE_DRIFT_XY_MM
            )
            if is_full_tray_geometry:
                maximum_state_drift_xy = FULL_TRAY_GEOMETRY_MAXIMUM_STATE_DRIFT_XY_MM
            if is_wafer_pick_xy:
                maximum_state_drift_xy = WAFER_PICK_XY_MAXIMUM_STATE_DRIFT_XY_MM
            maximum_state_drift_joint = (
                MOVED_TRAY_MAXIMUM_STATE_DRIFT_JOINT
                if is_moved_tray
                else RUNTIME_MOVE_MAXIMUM_STATE_DRIFT_JOINT
            )
            if is_full_tray_geometry:
                maximum_state_drift_joint = FULL_TRAY_GEOMETRY_MAXIMUM_STATE_DRIFT_JOINT
            if is_wafer_pick_xy:
                maximum_state_drift_joint = WAFER_PICK_XY_MAXIMUM_STATE_DRIFT_JOINT
            extent = _positive_at_most(
                raw.get("local_extent_mm"),
                f"第 {index} 步 local_extent_mm",
                maximum_extent,
            )
            margin = _finite(
                raw.get("domain_margin_mm"),
                f"第 {index} 步 domain_margin_mm",
            )
            if (
                margin < RUNTIME_MOVE_MINIMUM_DOMAIN_MARGIN_MM - 1e-12
                or margin >= extent
            ):
                raise ValueError(
                    f"第 {index} 步 domain_margin_mm 必须至少为"
                    f" {RUNTIME_MOVE_MINIMUM_DOMAIN_MARGIN_MM:g} 且小于local_extent_mm"
                )
            raw_precompensate_rz = raw.get("precompensate_rz", False)
            if not isinstance(raw_precompensate_rz, bool):
                raise ValueError(
                    f"第 {index} 步 precompensate_rz 必须是布尔值"
                )
            step.update(
                {
                    "request_key": request_key,
                    "target_name": target_name,
                    "calibration_sha256": calibration_sha256,
                    "fine_calibration_sha256": fine_calibration_sha256,
                    "anchor_robot_xy_mm": _vector_values(
                        raw.get("anchor_robot_xy_mm"),
                        2,
                        f"第 {index} 步 anchor_robot_xy_mm",
                    ),
                    "local_extent_mm": extent,
                    "domain_margin_mm": margin,
                    "required_j3_mm": _finite(
                        raw.get("required_j3_mm"),
                        f"第 {index} 步 required_j3_mm",
                    ),
                    "required_rz_deg": _finite(
                        raw.get("required_rz_deg"),
                        f"第 {index} 步 required_rz_deg",
                    ),
                    "max_xy_step_norm_mm": _positive_at_most(
                        raw.get(
                            "max_xy_step_norm_mm",
                            maximum_step_norm,
                        ),
                        f"第 {index} 步 max_xy_step_norm_mm",
                        maximum_step_norm,
                    ),
                    "max_xy_axis_mm": _positive_at_most(
                        raw.get(
                            "max_xy_axis_mm",
                            maximum_axis_step,
                        ),
                        f"第 {index} 步 max_xy_axis_mm",
                        maximum_axis_step,
                    ),
                    "j3_tolerance_mm": _positive_at_most(
                        raw.get(
                            "j3_tolerance_mm",
                            RUNTIME_MOVE_MAXIMUM_J3_TOLERANCE_MM,
                        ),
                        f"第 {index} 步 j3_tolerance_mm",
                        RUNTIME_MOVE_MAXIMUM_J3_TOLERANCE_MM,
                    ),
                    "rz_tolerance_deg": _positive_at_most(
                        raw.get(
                            "rz_tolerance_deg",
                            maximum_rz_tolerance,
                        ),
                        f"第 {index} 步 rz_tolerance_deg",
                        maximum_rz_tolerance,
                    ),
                    "target_rz_tolerance_deg": _positive_at_most(
                        raw.get(
                            "target_rz_tolerance_deg",
                            RUNTIME_MOVE_MAXIMUM_TARGET_RZ_TOLERANCE_DEG,
                        ),
                        f"第 {index} 步 target_rz_tolerance_deg",
                        RUNTIME_MOVE_MAXIMUM_TARGET_RZ_TOLERANCE_DEG,
                    ),
                    "max_sequential_transient_rz_deg": _positive_at_most(
                        raw.get(
                            "max_sequential_transient_rz_deg",
                            maximum_transient_rz,
                        ),
                        f"第 {index} 步 max_sequential_transient_rz_deg",
                        maximum_transient_rz,
                    ),
                    "precompensate_rz": raw_precompensate_rz,
                    # Stage7B uses a measured 20 x 20 mm endpoint domain and
                    # explicitly permits the controller's natural sequential
                    # J1/J2 path to leave that model domain.  Stage7A keeps the
                    # stricter intermediate-domain gate.  This policy is
                    # derived from the recognized request key, not trusted
                    # from an imported task field.
                    "enforce_sequential_intermediate_domain": not (
                        is_stage7b
                        or is_full_tray_geometry
                        or is_moved_tray
                        or is_wafer_pick_xy
                    ),
                    "max_state_drift_xy_mm": _positive_at_most(
                        raw.get(
                            "max_state_drift_xy_mm",
                            maximum_state_drift_xy,
                        ),
                        f"第 {index} 步 max_state_drift_xy_mm",
                        maximum_state_drift_xy,
                    ),
                    "max_state_drift_joint": _positive_at_most(
                        raw.get(
                            "max_state_drift_joint",
                            maximum_state_drift_joint,
                        ),
                        f"第 {index} 步 max_state_drift_joint",
                        maximum_state_drift_joint,
                    ),
                    "max_sequential_transient_xy_mm": _positive_at_most(
                        raw.get(
                            "max_sequential_transient_xy_mm",
                            maximum_transient_xy,
                        ),
                        f"第 {index} 步 max_sequential_transient_xy_mm",
                        maximum_transient_xy,
                    ),
                    "move_tolerance": _positive_at_most(
                        raw.get(
                            "move_tolerance",
                            RUNTIME_MOVE_MAXIMUM_MOVE_TOLERANCE,
                        ),
                        f"第 {index} 步 move_tolerance",
                        RUNTIME_MOVE_MAXIMUM_MOVE_TOLERANCE,
                    ),
                    "proposal_max_age_s": _positive_at_most(
                        raw.get(
                            "proposal_max_age_s",
                            RUNTIME_MOVE_MAXIMUM_PROPOSAL_AGE_S,
                        ),
                        f"第 {index} 步 proposal_max_age_s",
                        RUNTIME_MOVE_MAXIMUM_PROPOSAL_AGE_S,
                    ),
                    "fk_pose_xy_tolerance_mm": _positive_at_most(
                        raw.get(
                            "fk_pose_xy_tolerance_mm",
                            RUNTIME_MOVE_MAXIMUM_FK_POSE_MISMATCH_MM,
                        ),
                        f"第 {index} 步 fk_pose_xy_tolerance_mm",
                        RUNTIME_MOVE_MAXIMUM_FK_POSE_MISMATCH_MM,
                    ),
                }
            )
            if is_wafer_pick_xy:
                step["final_rz_deg"] = _finite(
                    raw.get("final_rz_deg"),
                    f"第 {index} 步 final_rz_deg",
                )
                step["max_j4_rotation_deg"] = _positive_at_most(
                    raw.get(
                        "max_j4_rotation_deg",
                        WAFER_PICK_XY_MAXIMUM_J4_ROTATION_DEG,
                    ),
                    f"第 {index} 步 max_j4_rotation_deg",
                    WAFER_PICK_XY_MAXIMUM_J4_ROTATION_DEG,
                )
        elif kind == "move_xyzr":
            for key in ("x_mm", "y_mm", "z_mm", "r_deg"):
                step[key] = _finite(raw.get(key, 0.0), f"第 {index} 步 {key}")
            if not any(abs(step[key]) > 1e-12 for key in ("x_mm", "y_mm", "z_mm", "r_deg")):
                raise ValueError(f"第 {index} 步 move_xyzr 至少要有一个非零增量")
        elif kind == "set_do":
            channel = raw.get("channel")
            level = raw.get("level")
            if (
                isinstance(channel, bool)
                or not isinstance(channel, int)
                or not 1 <= channel <= 16
            ):
                raise ValueError(f"第 {index} 步 set_do.channel 必须是1到16的整数")
            if (
                isinstance(level, bool)
                or not isinstance(level, int)
                or level not in {0, 1}
            ):
                raise ValueError(f"第 {index} 步 set_do.level 必须是整数0或1")
            step["channel"] = int(channel)
            step["level"] = int(level)
        elif kind == "wait":
            step["seconds"] = _finite(raw.get("seconds"), f"第 {index} 步 seconds")
            if step["seconds"] < 0:
                raise ValueError(f"第 {index} 步 seconds 不能为负数")
        elif kind == "operator_checkpoint":
            if active_video_sources:
                raise ValueError(
                    f"第 {index} 步人工确认前必须先停止全部录像"
                )
            message = str(raw.get("message") or "").strip()
            if not message:
                raise ValueError(f"第 {index} 步 operator_checkpoint.message 不能为空")
            continue_text = str(
                raw.get("continue_text") or "继续采集"
            ).strip()
            finish_text = str(raw.get("finish_text") or "结束采集").strip()
            if not continue_text or not finish_text:
                raise ValueError(
                    f"第 {index} 步人工确认的两个按钮文字都不能为空"
                )
            repeat_from_index = raw.get("repeat_from_index")
            if (
                isinstance(repeat_from_index, bool)
                or not isinstance(repeat_from_index, int)
            ):
                raise ValueError(
                    f"第 {index} 步 repeat_from_index 必须是从0开始的整数"
                )
            if repeat_from_index < 0 or repeat_from_index >= index - 1:
                raise ValueError(
                    f"第 {index} 步 repeat_from_index 必须指向此前的动作"
                )
            if str(raw_actions[repeat_from_index].get("type") or "").strip() == (
                "operator_checkpoint"
            ):
                raise ValueError(
                    f"第 {index} 步不能跳回另一个 operator_checkpoint"
                )
            step.update(
                {
                    "message": message,
                    "continue_text": continue_text,
                    "finish_text": finish_text,
                    "repeat_from_index": repeat_from_index,
                }
            )
        elif kind in {"capture", "start_video", "stop_video"}:
            source = raw.get("source")
            if isinstance(source, bool) or not isinstance(source, int) or not 0 <= source <= 8:
                raise ValueError(f"第 {index} 步 source 必须是 0 到 8 的整数")
            step["source"] = source
            if kind == "start_video":
                if source in active_video_sources:
                    raise ValueError(f"第 {index} 步 相机源#{source}已在录像")
                filename = str(raw.get("filename") or f"{source}_video.avi").strip()
                suffix = Path(filename).suffix.lower()
                if (
                    not filename
                    or Path(filename).name != filename
                    or suffix not in {".avi", ".mp4"}
                ):
                    raise ValueError(
                        f"第 {index} 步 filename 必须是当前实验文件夹内的 .avi 或 .mp4 文件名"
                    )
                if filename in video_filenames:
                    raise ValueError(f"第 {index} 步 录像文件名重复：{filename}")
                fps = _finite(raw.get("fps", 20.0), f"第 {index} 步 fps")
                if not 0.0 < fps <= 120.0:
                    raise ValueError(f"第 {index} 步 fps 必须在 (0, 120] 范围内")
                step["filename"] = filename
                step["fps"] = fps
                active_video_sources.add(source)
                video_filenames.add(filename)
            elif kind == "stop_video":
                if source not in active_video_sources:
                    raise ValueError(f"第 {index} 步 相机源#{source}尚未开始录像")
                active_video_sources.remove(source)

        actions.append(step)

    if active_video_sources:
        sources = ", ".join(f"#{source}" for source in sorted(active_video_sources))
        raise ValueError(f"录像步骤缺少 stop_video：相机源 {sources}")

    camera_capture_settings = _normalize_camera_capture_settings(
        raw_task.get("camera_capture_settings")
    )
    used_camera_sources = {
        int(step["source"])
        for step in actions
        if step["type"] in {"capture", "start_video", "stop_video"}
    }
    unused_settings = sorted(set(camera_capture_settings) - used_camera_sources)
    if unused_settings:
        sources = ", ".join(f"#{source}" for source in unused_settings)
        raise ValueError(f"camera_capture_settings 指定了任务未使用的相机源：{sources}")

    return {
        "api_version": ACTION_API_VERSION,
        "name": task_name,
        "description": str(raw_task.get("description") or "").strip(),
        "camera_model": camera_model,
        "camera_capture_settings": camera_capture_settings,
        "actions": actions,
    }


def calculate_camera_position(pose: Sequence[float], camera_model: dict) -> dict:
    """Calculate the camera centre from the measured TCP pose.

    Coordinate/equation annotations:

    * ``rz_rad = Rz*pi/180`` converts the controller's degree value to radians.
    * Rz is measured counter-clockwise from world -Y.  Rotating the base vector
      ``(0, -d)`` gives ``offset_x = d*sin(rz_rad)`` and
      ``offset_y = -d*cos(rz_rad)``.
    * ``camera_x = centre_x + offset_x`` and
      ``camera_y = centre_y + offset_y`` translate that 20 mm radial vector
      from the suction-cup/wafer centre into world coordinates.
    * ``camera_z = centre_z`` assumes no measured vertical camera offset.
    """
    if len(pose) != 6:
        raise ValueError("位姿必须包含 x/y/z/Rx/Ry/Rz 六个值")
    values = [_finite(value, f"pose[{index}]") for index, value in enumerate(pose)]
    centre_x_mm, centre_y_mm, centre_z_mm, _rx_deg, _ry_deg, rz_deg = values
    offset_mm = _finite(camera_model["offset_mm"], "camera_model.offset_mm")
    rz_rad = math.radians(rz_deg)
    camera_offset_x_mm = offset_mm * math.sin(rz_rad)
    camera_offset_y_mm = -offset_mm * math.cos(rz_rad)
    camera_x_mm = centre_x_mm + camera_offset_x_mm
    camera_y_mm = centre_y_mm + camera_offset_y_mm
    camera_z_mm = centre_z_mm
    return {
        "x_mm": camera_x_mm,
        "y_mm": camera_y_mm,
        "z_mm": camera_z_mm,
        "angle_from_negative_y_deg": rz_deg,
        "offset_mm": offset_mm,
    }


class CameraSourcePool:
    """Keep all required OpenCV camera sources open for one action run."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        source_resolver: Callable[
            [Sequence[int]], dict[int, ResolvedCameraSource]
        ] = resolve_camera_sources,
    ):
        self._width = int(width)
        self._height = int(height)
        self._source_resolver = source_resolver
        self._captures: dict[int, object] = {}
        self._resolved_sources: dict[int, ResolvedCameraSource] = {}
        self._cv2 = None
        self._capture_setting_reports: dict[int, dict] = {}
        self._video_sessions: dict[int, dict] = {}
        self._video_error_lock = threading.Lock()
        self._video_error: Optional[str] = None

    @staticmethod
    def _auto_exposure_enabled(actual: float) -> Optional[bool]:
        """Normalize DirectShow 0.25/0.75 and UVC 0/1 mode readbacks."""

        raw = float(actual)
        if not math.isfinite(raw) or raw < 0.0:
            return None
        if abs(raw - 0.75) <= 0.13 or abs(raw - 1.0) <= 0.13:
            return True
        if abs(raw - 0.25) <= 0.13 or abs(raw) <= 0.13:
            return False
        return None

    def _restore_default_auto_exposure(self, source: int) -> None:
        """Restore the application policy default: automatic exposure.

        Fixed exposure is deliberately not restored across task/camera
        lifetimes.  It is a temporary, operator-owned setting in the hand-eye
        UI only.
        """
        cap = self._captures.get(int(source))
        if cap is None or self._cv2 is None:
            return
        try:
            for requested_mode in (0.75, 1.0):
                accepted = bool(
                    cap.set(self._cv2.CAP_PROP_AUTO_EXPOSURE, requested_mode)
                )
                readback = float(cap.get(self._cv2.CAP_PROP_AUTO_EXPOSURE))
                state = self._auto_exposure_enabled(readback)
                if accepted and state is not False:
                    break
        except Exception:
            pass

    def _apply_capture_setting(self, source: int, setting: dict[str, object]) -> tuple[bool, str]:
        cap = self._captures[int(source)]
        cv2 = self._cv2
        assert cv2 is not None
        original_exposure = float(cap.get(cv2.CAP_PROP_EXPOSURE))
        original_auto_exposure = float(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
        if setting.get("auto_exposure") is not True:
            return False, (
                f"相机源#{source}拒绝任务固定曝光；"
                "固定曝光只允许由动态演示UI中的人工操作临时设置"
            )
        report = {
            "source": int(source),
            "requested": {"auto_exposure": True},
            "original": {
                "exposure": (
                    original_exposure if math.isfinite(original_exposure) else None
                ),
                "auto_exposure": (
                    original_auto_exposure
                    if math.isfinite(original_auto_exposure)
                    else None
                ),
            },
            "auto_mode_requests": [],
            "applied": None,
        }
        self._capture_setting_reports[int(source)] = report
        for requested_mode in (0.75, 1.0):
            accepted = bool(
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, requested_mode)
            )
            readback = float(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
            state = self._auto_exposure_enabled(readback)
            report["auto_mode_requests"].append(
                {
                    "requested": requested_mode,
                    "accepted": accepted,
                    "readback": readback if math.isfinite(readback) else None,
                    "normalized_state": state,
                }
            )
            if not accepted or state is False:
                continue
            exposure_readback = float(cap.get(cv2.CAP_PROP_EXPOSURE))
            report["applied"] = {
                "auto_mode_request_accepted": True,
                "auto_mode_confirmed": state is True,
                "auto_mode_effective": True,
                "auto_mode_readback_is_advisory": state is None,
                "auto_mode_immediate_readback": (
                    readback if math.isfinite(readback) else None
                ),
                "exposure_immediate_readback": (
                    exposure_readback if math.isfinite(exposure_readback) else None
                ),
            }
            return True, ""
        # Never switch to manual exposure as a rollback.  One last auto request
        # is harmless and close() will repeat it before releasing the device.
        self._restore_default_auto_exposure(source)
        return False, f"相机源#{source}驱动拒绝自动曝光请求"

    def capture_settings_report(self) -> dict[str, dict]:
        return {
            str(source): json.loads(json.dumps(report, allow_nan=False))
            for source, report in sorted(self._capture_setting_reports.items())
        }

    def camera_sources_report(self) -> dict[str, dict]:
        return {
            str(source): row.to_json()
            for source, row in sorted(self._resolved_sources.items())
        }

    def open_sources(
        self,
        sources: Sequence[int],
        camera_capture_settings: Optional[dict[int, dict[str, object]]] = None,
    ) -> tuple[bool, str]:
        try:
            import cv2
        except Exception as exc:
            return False, f"未安装 opencv-python: {exc}"
        self._cv2 = cv2
        requested_sources = sorted(set(int(value) for value in sources))
        try:
            self._resolved_sources = self._source_resolver(requested_sources)
        except Exception as exc:
            self.close()
            return False, f"相机USB身份检查失败：{exc}"
        for source in requested_sources:
            resolved = self._resolved_sources[source]
            cap = cv2.VideoCapture(resolved.physical_index, resolved.backend)
            if not cap.isOpened():
                cap.release()
                self.close()
                return False, (
                    f"无法打开逻辑相机源#{source}（物理Index "
                    f"{resolved.physical_index}）"
                )
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._captures[source] = cap
            setting = (camera_capture_settings or {}).get(
                source, {"auto_exposure": True}
            )
            applied, reason = self._apply_capture_setting(source, setting)
            self._capture_setting_reports[source]["camera"] = resolved.to_json()
            if not applied:
                self.close()
                return False, reason

        # Opening a DirectShow device does not guarantee that it can already
        # return frames.  Validate every required source before the first robot
        # movement so a cold/unavailable source #2 cannot stop the experiment
        # after source #0/#1 have already saved their P00 images.
        for source in sorted(self._captures):
            if self._read_fresh_frame(source, attempts=12) is None:
                self.close()
                resolved = self._resolved_sources[source]
                return False, (
                    f"逻辑相机源#{source}/物理Index {resolved.physical_index}"
                    "已打开，但预热后仍无法取帧"
                )
            assert self._capture_setting_reports[source]["applied"] is not None
            settled_mode = float(
                self._captures[source].get(cv2.CAP_PROP_AUTO_EXPOSURE)
            )
            settled_state = self._auto_exposure_enabled(settled_mode)
            if settled_state is False:
                self._restore_default_auto_exposure(source)
                self.close()
                return False, (
                    f"相机源#{source}预热后明确处于手动曝光模式："
                    f"读回 {settled_mode:.3f}"
                )
            settled_exposure = float(
                self._captures[source].get(cv2.CAP_PROP_EXPOSURE)
            )
            self._capture_setting_reports[source]["applied"].update(
                {
                    "auto_mode_settled_confirmed": settled_state is True,
                    "auto_mode_settled_effective": True,
                    "auto_mode_settled_readback_is_advisory": (
                        settled_state is None
                    ),
                    "auto_mode_settled_readback": (
                        settled_mode if math.isfinite(settled_mode) else None
                    ),
                    "exposure_settled_readback": (
                        settled_exposure
                        if math.isfinite(settled_exposure)
                        else None
                    ),
                }
            )
        return True, ""

    def _read_fresh_frame(self, source: int, *, attempts: int = 8):
        """Read through buffered frames with bounded retries; return the newest."""
        cap = self._captures.get(int(source))
        if cap is None:
            return None
        for _attempt in range(max(1, int(attempts))):
            latest = None
            # Multiple reads drain frames buffered during the two-second dwell.
            for _ in range(4):
                ok, frame = cap.read()
                if not ok or frame is None or getattr(frame, "size", 1) == 0:
                    latest = None
                    break
                latest = frame
            if latest is not None:
                return latest
            time.sleep(0.05)
        return None

    def snapshot(self, source: int, path: Path) -> bool:
        if self._cv2 is None:
            return False
        frame = self._read_fresh_frame(source)
        if frame is None:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        return bool(self._cv2.imwrite(str(path), frame))

    def start_video(self, source: int, path: Path, fps: float) -> None:
        """Start a background AVI/MJPG or MP4/mp4v recording."""
        source = int(source)
        if self._cv2 is None or source not in self._captures:
            raise RuntimeError(f"相机源#{source}尚未打开")
        if source in self._video_sessions:
            raise RuntimeError(f"相机源#{source}已在录像")
        frame = self._read_fresh_frame(source)
        if frame is None:
            raise RuntimeError(f"相机源#{source}录像前无法取帧")
        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            raise RuntimeError(f"相机源#{source}录像画面尺寸无效")
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        codecs = {".avi": "MJPG", ".mp4": "mp4v"}
        codec = codecs.get(suffix)
        if codec is None:
            raise RuntimeError(f"不支持的录像格式：{suffix or '无扩展名'}")
        writer = self._cv2.VideoWriter(
            str(path),
            self._cv2.VideoWriter_fourcc(*codec),
            float(fps),
            (int(width), int(height)),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"无法创建相机源#{source}录像文件：{path.name}")

        stop_event = threading.Event()
        session = {
            "source": source,
            "path": Path(path),
            "fps": float(fps),
            "writer": writer,
            "stop_event": stop_event,
            "frame_count": 0,
            "started_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        }

        def record_loop() -> None:
            period_s = 1.0 / float(fps)
            deadline = time.monotonic()
            try:
                while not stop_event.is_set():
                    ok, next_frame = self._captures[source].read()
                    if not ok or next_frame is None or getattr(next_frame, "size", 1) == 0:
                        raise RuntimeError(f"相机源#{source}录像期间取帧失败")
                    writer.write(next_frame)
                    session["frame_count"] += 1
                    deadline += period_s
                    stop_event.wait(max(0.0, deadline - time.monotonic()))
            except Exception as exc:  # noqa: BLE001 - reported on worker thread
                with self._video_error_lock:
                    self._video_error = str(exc) or exc.__class__.__name__

        thread = threading.Thread(
            target=record_loop,
            name=f"scara-video-source-{source}",
            daemon=True,
        )
        session["thread"] = thread
        self._video_sessions[source] = session
        thread.start()

    def check_video_error(self) -> None:
        """Raise a background recorder error on the action worker thread."""
        with self._video_error_lock:
            message = self._video_error
            self._video_error = None
        if message:
            raise RuntimeError(message)

    def stop_video(self, source: int) -> dict:
        """Stop one recording and return JSON-serializable session metadata."""
        source = int(source)
        session = self._video_sessions.get(source)
        if session is None:
            raise RuntimeError(f"相机源#{source}尚未开始录像")
        session["stop_event"].set()
        session["thread"].join(timeout=3.0)
        if session["thread"].is_alive():
            raise RuntimeError(f"相机源#{source}录像线程无法停止")
        session["writer"].release()
        self._video_sessions.pop(source, None)
        self.check_video_error()
        frame_count = int(session["frame_count"])
        if frame_count < 1:
            raise RuntimeError(f"相机源#{source}录像没有写入任何画面")
        return {
            "source": source,
            "filename": session["path"].name,
            "fps": float(session["fps"]),
            "frame_count": frame_count,
            "started_at": session["started_at"],
            "finished_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        }

    def close(self) -> None:
        for source in list(self._video_sessions):
            try:
                self.stop_video(source)
            except Exception:
                session = self._video_sessions.pop(source, None)
                if session is not None:
                    try:
                        session["stop_event"].set()
                        session["writer"].release()
                    except Exception:
                        pass
        for source in list(self._capture_setting_reports):
            self._restore_default_auto_exposure(source)
        for cap in self._captures.values():
            try:
                cap.release()
            except Exception:
                pass
        self._captures.clear()


class ActionWorker(QThread):
    """Execute one validated action and persist photos/videos/point JSON."""

    progress = pyqtSignal(str)
    photo_saved = pyqtSignal(str)
    video_saved = pyqtSignal(str)
    point_recorded = pyqtSignal(str)
    operator_checkpoint_requested = pyqtSignal(str, str, str, str)
    runtime_move_joints_requested = pyqtSignal(dict)
    run_finished = pyqtSignal(bool, str, str)

    def __init__(
        self,
        controller: ScaraController,
        task: dict,
        output_dir: Path,
        *,
        snapshot_source: Optional[Callable[[int, Path], bool]] = None,
        camera_position_calculator: Optional[Callable[[Sequence[float]], dict]] = None,
        source_position_calculators: Optional[
            dict[int, Callable[[Sequence[float], Sequence[float]], dict]]
        ] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._controller = controller
        self._task = normalize_action_task(task)
        self._output_dir = Path(output_dir)
        self._snapshot_source = snapshot_source
        self._camera_position_calculator = camera_position_calculator
        self._source_position_calculators = dict(source_position_calculators or {})
        self._camera_pool: Optional[CameraSourcePool] = None
        self._stop_requested = threading.Event()
        self._photo_counts: dict[int, int] = {}
        self._active_video_sources: set[int] = set()
        self._manifest: dict = {}
        self._operator_decision_event = threading.Event()
        self._operator_decision: Optional[bool] = None
        self._runtime_move_response_event = threading.Event()
        self._runtime_move_response_lock = threading.Lock()
        self._runtime_move_pending_request_id: Optional[str] = None
        self._runtime_move_response: Optional[object] = None
        self._runtime_move_request_sequence = 0
        self._runtime_session_complete = False
        self._declared_do_channels = sorted(
            {
                int(step["channel"])
                for step in self._task["actions"]
                if step["type"] == "set_do"
            }
        )
        self._touched_do_channels: set[int] = set()
        self._uncertain_do_channels: set[int] = set()
        self._known_do_levels: dict[int, int] = {}
        self._repeatable = any(
            step["type"] == "operator_checkpoint"
            for step in self._task["actions"]
        )
        self._collection_round = 1

    def request_stop(self) -> None:
        self._stop_requested.set()
        try:
            self._controller.emergency_stop(
                do_channels=list(self._declared_do_channels)
            )
        except TypeError:
            self._controller.emergency_stop()

    def respond_operator_checkpoint(self, continue_collection: bool) -> None:
        """Release a paused checkpoint from the Qt UI thread.

        ``True`` repeats the configured acquisition block.  ``False`` ends the
        acquisition normally, allowing the task runtime to calibrate all images
        already accumulated in the current output folder.
        """
        self._operator_decision = bool(continue_collection)
        self._operator_decision_event.set()

    def respond_runtime_move_joints(self, response: object) -> None:
        """Return one UI/runtime decision to the paused action thread.

        The response is deliberately treated as untrusted input.  This method
        only transfers it across threads; validation, a fresh controller read,
        and the independent kinematic audit all happen on the worker thread.
        Stale or duplicate responses cannot release a later request.
        """

        with self._runtime_move_response_lock:
            pending = self._runtime_move_pending_request_id
            if pending is None or self._runtime_move_response_event.is_set():
                return
            if isinstance(response, dict) and str(
                response.get("request_id") or ""
            ) not in {"", pending}:
                return
            try:
                detached: object = _json_safe_mapping(
                    response,
                    "runtime_move_joints response",
                )
            except ValueError as exc:
                detached = {
                    "request_id": pending,
                    "decision": "abort",
                    "reason": str(exc),
                }
            self._runtime_move_response = detached
            self._runtime_move_response_event.set()

    def _interruptible_wait(self, seconds: float) -> bool:
        remaining = max(0.0, float(seconds))
        while remaining > 1e-9:
            if self._stop_requested.is_set():
                return False
            interval = min(0.1, remaining)
            time.sleep(interval)
            remaining -= interval
        return not self._stop_requested.is_set()

    @property
    def manifest_path(self) -> Path:
        return self._output_dir / "points.json"

    def _save_manifest(self) -> None:
        atomic_write_text(
            self.manifest_path,
            json.dumps(self._manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _read_state(self, context: str) -> dict:
        status = self._controller.read_all_sync()
        if not isinstance(status, dict):
            raise RuntimeError(f"{context}：读取机械臂状态失败")
        joints = status.get("joints")
        pose = status.get("pose")
        if not isinstance(joints, (list, tuple)) or len(joints) != 4:
            raise RuntimeError(f"{context}：状态中缺少 J1/J2/J3/J4")
        if not isinstance(pose, (list, tuple)) or len(pose) != 6:
            raise RuntimeError(f"{context}：状态中缺少 x/y/z/Rx/Ry/Rz")
        connected_reader = getattr(self._controller, "is_connected", None)
        connected = bool(connected_reader()) if callable(connected_reader) else False
        controller_safety = {
            "captured_monotonic_s": time.monotonic(),
            "connected": connected,
            "effectively_enabled": status.get("effectively_enabled") is True,
            "enable": int(status.get("enable", 0)),
            "estop": status.get("estop") is True,
            "soft_estop": status.get("soft_estop") is True,
            "warn": int(status.get("warn", -1)),
            "need_clear": status.get("need_clear") is True,
            "mode": str(status.get("mode") or "?"),
        }
        return {
            "joints": [_finite(value, f"{context}.joints") for value in joints],
            "pose": [_finite(value, f"{context}.pose") for value in pose],
            "controller_safety": controller_safety,
        }

    def _read_runtime_motion_state(self, context: str) -> dict:
        """Read joints, pose, and authoritative controller safety flags.

        Missing flags are false/failing by design.  A Stage-7A runtime gets
        this detached snapshot for display and calculation, but ActionWorker
        always performs another read after the operator confirms.
        """

        status = self._controller.read_all_sync()
        if not isinstance(status, dict):
            raise RuntimeError(f"{context}：读取机械臂状态失败")
        joints = _joint_values(status.get("joints"), f"{context}.joints")
        pose = _vector_values(status.get("pose"), 6, f"{context}.pose")
        connected_reader = getattr(self._controller, "is_connected", None)
        connected = bool(connected_reader()) if callable(connected_reader) else False

        sequence_lock = getattr(self._controller, "_motion_sequence_lock", None)
        lock_reader = getattr(sequence_lock, "locked", None)
        if callable(lock_reader):
            controller_idle = not bool(lock_reader())
        else:
            ready_reader = getattr(self._controller, "motion_ready", None)
            controller_idle = bool(ready_reader()) if callable(ready_reader) else False

        warn = int(status.get("warn", -1))
        estop = status.get("estop") is True
        soft_estop = status.get("soft_estop") is True
        need_clear = status.get("need_clear") is True
        enabled = status.get("effectively_enabled") is True
        try:
            speed_percent = float(status.get("speed"))
        except (TypeError, ValueError, OverflowError):
            speed_percent = None
        if speed_percent is not None and not math.isfinite(speed_percent):
            speed_percent = None
        return {
            "captured_monotonic_s": time.monotonic(),
            "joints": joints,
            "pose": pose,
            "controller_connected": connected,
            "controller_enabled": enabled,
            "warn": warn,
            "need_clear": need_clear,
            "estop": estop,
            "soft_estop": soft_estop,
            "alarm_clear": warn == 0 and not need_clear,
            "estop_clear": not estop,
            "soft_estop_clear": not soft_estop,
            "controller_idle": controller_idle,
            "mode": str(status.get("mode") or "?"),
            "speed_percent": speed_percent,
        }

    def _record_point(self, name: str) -> None:
        state = self._read_state(name)
        joints = state["joints"]
        pose = state["pose"]
        if self._camera_position_calculator is None:
            raw_camera = calculate_camera_position(pose, self._task["camera_model"])
        else:
            raw_camera = self._camera_position_calculator(list(pose))
        if not isinstance(raw_camera, dict):
            raise RuntimeError(f"{name}：相机位置函数必须返回字典")
        camera = dict(raw_camera)
        for key in ("x_mm", "y_mm", "z_mm"):
            camera[key] = _finite(camera.get(key), f"{name}.camera_position.{key}")
        point = {
            "sequence": len(self._manifest["points"]) + 1,
            "name": name,
            "recorded_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "joints": {
                "J1_deg": joints[0],
                "J2_deg": joints[1],
                "J3_mm": joints[2],
                "J4_deg": joints[3],
            },
            "mechanical_center": {
                "x_mm": pose[0],
                "y_mm": pose[1],
                "z_mm": pose[2],
                "Rx_deg": pose[3],
                "Ry_deg": pose[4],
                "Rz_deg": pose[5],
            },
            "camera_position": camera,
            "controller_safety": dict(state["controller_safety"]),
        }
        if self._repeatable:
            point["collection_round"] = self._collection_round
        self._manifest["points"].append(point)
        self._save_manifest()
        self.point_recorded.emit(name)

    def _source_position_for_last_point(self, source: int) -> Optional[dict]:
        """Calculate optional source-specific XYZ for the last recorded point."""
        calculator = self._source_position_calculators.get(int(source))
        if calculator is None:
            return None
        if not self._manifest["points"]:
            raise RuntimeError(f"相机源#{source}位置计算前缺少 record_point 途径点")

        point = self._manifest["points"][-1]
        point_joints = point["joints"]
        centre = point["mechanical_center"]
        joints = [
            point_joints["J1_deg"],
            point_joints["J2_deg"],
            point_joints["J3_mm"],
            point_joints["J4_deg"],
        ]
        pose = [
            centre["x_mm"],
            centre["y_mm"],
            centre["z_mm"],
            centre["Rx_deg"],
            centre["Ry_deg"],
            centre["Rz_deg"],
        ]
        raw_position = calculator(joints, pose)
        if not isinstance(raw_position, dict):
            raise RuntimeError(f"相机源#{source}位置函数必须返回字典")
        position = dict(raw_position)
        for key in ("x_mm", "y_mm", "z_mm"):
            position[key] = _finite(
                position.get(key),
                f"途径点 {point['sequence']}.camera{source}_position.{key}",
            )
        return position

    def _capture(self, source: int) -> None:
        point_sequence = len(self._manifest["points"])
        if point_sequence < 1:
            raise RuntimeError(f"相机源#{source}拍照前缺少 record_point 途径点")
        number = self._photo_counts.get(source, 0) + 1
        # The suffix is the global route-point sequence, not this source's own
        # photo counter.  All cameras captured at one physical point therefore
        # share the same suffix (for example 0_001, 1_001, 2_001).
        photo_path = self._output_dir / f"{source}_{point_sequence:03d}.jpg"
        if photo_path.exists():
            raise RuntimeError(f"途径点照片已存在，拒绝覆盖：{photo_path.name}")
        source_position = self._source_position_for_last_point(source)
        if self._snapshot_source is not None:
            ok = bool(self._snapshot_source(source, photo_path))
        elif self._camera_pool is not None:
            ok = self._camera_pool.snapshot(source, photo_path)
        else:  # pragma: no cover - constructor/run invariant
            ok = False
        if not ok:
            raise RuntimeError(f"保存相机源#{source}照片失败")
        if source_position is not None:
            # Only a point at which this source actually saved a photo receives
            # the source-specific position annotation in points.json.
            self._manifest["points"][-1][f"camera{source}_position"] = source_position
        self._photo_counts[source] = number
        photo_record = {
            "source": source,
            "sequence_for_source": number,
            "filename": photo_path.name,
            "captured_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "point_sequence": point_sequence,
        }
        if self._repeatable:
            photo_record["collection_round"] = self._collection_round
        self._manifest["photos"].append(photo_record)
        self._save_manifest()
        self.photo_saved.emit(str(photo_path))

    def _wait_for_operator_checkpoint(self, step: dict) -> bool:
        """Pause after a complete scan and wait for Continue or Finish.

        The robot has already returned to the taught centre before Task 7 emits
        this action.  No motion command is issued while this method waits.
        """
        self._operator_decision = None
        self._operator_decision_event.clear()
        photo_total = sum(self._photo_counts.values())
        message = (
            f"已完成第 {self._collection_round} 个标定板姿态，"
            f"本任务累计保存 {photo_total} 张照片。\n\n"
            + step["message"]
        )
        self.progress.emit(
            f"第 {self._collection_round} 个姿态采集完成；等待人员确认"
        )
        self.operator_checkpoint_requested.emit(
            step["name"],
            message,
            step["continue_text"],
            step["finish_text"],
        )
        while not self._operator_decision_event.wait(0.1):
            if self._stop_requested.is_set():
                raise RuntimeError("动作已取消")
        if self._stop_requested.is_set():
            raise RuntimeError("动作已取消")
        if self._operator_decision is None:
            raise RuntimeError("人工确认没有返回有效选择")
        continue_collection = bool(self._operator_decision)
        self._manifest.setdefault("operator_checkpoints", []).append(
            {
                "collection_round": self._collection_round,
                "decided_at": datetime.now().astimezone().isoformat(
                    timespec="milliseconds"
                ),
                "decision": "continue" if continue_collection else "finish",
                "point_count": len(self._manifest["points"]),
                "photo_count": photo_total,
            }
        )
        self._save_manifest()
        return continue_collection

    def _start_video(self, step: dict) -> None:
        """Start one source recording inside the current timestamp folder."""
        source = int(step["source"])
        if self._camera_pool is None:
            raise RuntimeError(f"相机源#{source}录像池未初始化")
        video_path = self._output_dir / step["filename"]
        if video_path.exists():
            raise RuntimeError(f"录像文件已存在，拒绝覆盖：{video_path.name}")
        self._camera_pool.start_video(source, video_path, step["fps"])
        self._active_video_sources.add(source)
        self._manifest["video_recording"] = {
            "source": source,
            "filename": video_path.name,
            "fps": step["fps"],
            "status": "recording",
        }
        self._save_manifest()

    def _stop_video(self, source: int) -> None:
        """Finish one recording, save its metadata, and emit its path."""
        source = int(source)
        if self._camera_pool is None:
            raise RuntimeError(f"相机源#{source}录像池未初始化")
        metadata = self._camera_pool.stop_video(source)
        self._active_video_sources.discard(source)
        self._manifest["videos"].append(metadata)
        self._manifest.pop("video_recording", None)
        self._save_manifest()
        self.video_saved.emit(str(self._output_dir / metadata["filename"]))

    @staticmethod
    def _runtime_move_limits(step: dict) -> dict:
        keys = (
            "anchor_robot_xy_mm",
            "local_extent_mm",
            "domain_margin_mm",
            "required_j3_mm",
            "required_rz_deg",
            "final_rz_deg",
            "max_j4_rotation_deg",
            "max_xy_step_norm_mm",
            "max_xy_axis_mm",
            "j3_tolerance_mm",
            "rz_tolerance_deg",
            "target_rz_tolerance_deg",
            "max_sequential_transient_rz_deg",
            "precompensate_rz",
            "enforce_sequential_intermediate_domain",
            "max_state_drift_xy_mm",
            "max_state_drift_joint",
            "max_sequential_transient_xy_mm",
            "move_tolerance",
            "proposal_max_age_s",
            "fk_pose_xy_tolerance_mm",
        )
        return {key: step[key] for key in keys if key in step}

    @staticmethod
    def _runtime_controller_gates(
        state: dict,
        request_key: str | None = None,
    ) -> dict[str, bool]:
        gates = {
            "controller_connected": state.get("controller_connected") is True,
            "controller_enabled": state.get("controller_enabled") is True,
            "alarm_clear": state.get("alarm_clear") is True,
            "estop_clear": state.get("estop_clear") is True,
            "soft_estop_clear": state.get("soft_estop_clear") is True,
            "controller_idle": state.get("controller_idle") is True,
            "worker_not_stopped": True,
        }
        if request_key == WAFER_PICK_XY_RUNTIME_REQUEST_KEY:
            try:
                speed = float(state.get("speed_percent"))
            except (TypeError, ValueError, OverflowError):
                speed = math.nan
            gates.update(
                {
                    "controller_mode_is_t1": state.get("mode") == "T1",
                    "controller_speed_at_most_20_percent": (
                        math.isfinite(speed)
                        and speed > 0.0
                        and speed <= WAFER_PICK_XY_MAXIMUM_SPEED_PERCENT
                    ),
                }
            )
        return gates

    @staticmethod
    def _runtime_state_drift(
        displayed: dict,
        fresh: dict,
    ) -> tuple[float, float]:
        displayed_pose = _vector_values(displayed.get("pose"), 6, "displayed.pose")
        fresh_pose = _vector_values(fresh.get("pose"), 6, "fresh.pose")
        displayed_joints = _joint_values(
            displayed.get("joints"), "displayed.joints"
        )
        fresh_joints = _joint_values(fresh.get("joints"), "fresh.joints")
        xy_drift = math.hypot(
            fresh_pose[0] - displayed_pose[0],
            fresh_pose[1] - displayed_pose[1],
        )
        joint_drift = max(
            abs(actual - previous)
            for actual, previous in zip(fresh_joints, displayed_joints)
        )
        return float(xy_drift), float(joint_drift)

    @staticmethod
    def _audit_runtime_target(
        current_state: dict,
        target_joints: Sequence[float],
        step: dict,
    ) -> dict:
        """Call the shared, controller-free kinematic safety audit."""

        from scara.pipeline.xy_correction_planner import (
            audit_fixed_rz_xy_target,
            audit_j4_only_orientation_target,
        )

        if step.get("_audit_mode") == "j4_only":
            audit = audit_j4_only_orientation_target(
                current_state["joints"],
                current_state["pose"],
                target_joints,
                anchor_robot_xy_mm=step["anchor_robot_xy_mm"],
                local_extent_mm=step["local_extent_mm"],
                domain_margin_mm=step["domain_margin_mm"],
                required_j3_mm=step["required_j3_mm"],
                j3_tolerance_mm=step["j3_tolerance_mm"],
                required_start_rz_deg=step["required_rz_deg"],
                start_rz_tolerance_deg=step["rz_tolerance_deg"],
                target_rz_deg=step["final_rz_deg"],
                target_rz_tolerance_deg=step["target_rz_tolerance_deg"],
                maximum_j4_rotation_deg=step["max_j4_rotation_deg"],
            )
        else:
            audit = audit_fixed_rz_xy_target(
                current_state["joints"],
                current_state["pose"],
                target_joints,
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
                target_rz_tolerance_deg=step["target_rz_tolerance_deg"],
                max_sequential_transient_rz_deg=step[
                    "max_sequential_transient_rz_deg"
                ],
                precompensate_rz=step["precompensate_rz"],
                enforce_sequential_intermediate_domain=step[
                    "enforce_sequential_intermediate_domain"
                ],
            )
        closure_gate = (audit.get("gates") or {}).get(
            "controller_pose_matches_fk"
        )
        if isinstance(closure_gate, dict):
            try:
                closure = float(closure_gate.get("actual"))
            except (TypeError, ValueError, OverflowError):
                closure = math.inf
            closure_gate["passed"] = bool(
                math.isfinite(closure)
                and closure <= step["fk_pose_xy_tolerance_mm"] + 1e-12
            )
            closure_gate["limit"] = (
                f"<={step['fk_pose_xy_tolerance_mm']:.3f} mm"
            )
            audit["passed"] = all(
                isinstance(gate, dict) and gate.get("passed") is True
                for gate in (audit.get("gates") or {}).values()
            )
        return audit

    def _runtime_move_joints(self, step: dict) -> None:
        """Pause for one supervised target and execute it only after re-audit."""

        if step["request_key"] == STAGE7B_RUNTIME_REQUEST_KEY:
            runtime_label = "单点有限闭环"
        elif step["request_key"] == MOVED_TRAY_RUNTIME_REQUEST_KEY:
            runtime_label = "可移动托盘P22全盘定位"
        elif step["request_key"] == FULL_TRAY_GEOMETRY_REQUEST_KEY:
            runtime_label = "全盘定位Stage3毫米几何修正"
        elif step["request_key"] == WAFER_PICK_XY_RUNTIME_REQUEST_KEY:
            runtime_label = f"{step['target_name']}硅片XY悬空定位"
        else:
            runtime_label = "Stage7A"
        displayed_state = self._read_runtime_motion_state(
            f"{step['name']} 提案前检查"
        )
        self._runtime_move_request_sequence += 1
        request_id = (
            f"runtime-move-{self._runtime_move_request_sequence:03d}-"
            f"{time.monotonic_ns()}"
        )
        requested_monotonic_s = time.monotonic()
        request = {
            "schema_version": 1,
            "request_id": request_id,
            "request_key": step["request_key"],
            "target_name": step["target_name"],
            "name": step["name"],
            "requested_at": datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            ),
            "requested_monotonic_s": requested_monotonic_s,
            "calibration_sha256": step["calibration_sha256"],
            "fine_calibration_sha256": step.get("fine_calibration_sha256", ""),
            "limits": self._runtime_move_limits(step),
            "controller_state": displayed_state,
            "external_safety_gates": {
                **self._runtime_controller_gates(
                    displayed_state, step["request_key"]
                ),
                "camera_fresh": False,
                "operator_consent": False,
            },
        }
        audit_entry: dict = {
            "request_id": request_id,
            "request_key": step["request_key"],
            "target_name": step["target_name"],
            "name": step["name"],
            "status": "awaiting_operator_decision",
            "request": request,
        }
        self._manifest.setdefault("runtime_moves", []).append(audit_entry)
        self._save_manifest()

        with self._runtime_move_response_lock:
            self._runtime_move_pending_request_id = request_id
            self._runtime_move_response = None
            self._runtime_move_response_event.clear()
        self.progress.emit(f"{step['name']}：等待{runtime_label}计算与确认")
        self.runtime_move_joints_requested.emit(
            _json_safe_mapping(request, "runtime_move_joints request")
        )

        try:
            while not self._runtime_move_response_event.wait(0.1):
                if self._stop_requested.is_set():
                    audit_entry["status"] = "stopped_while_waiting"
                    self._save_manifest()
                    raise RuntimeError("动作已取消")
            if self._stop_requested.is_set():
                audit_entry["status"] = "stopped_while_waiting"
                self._save_manifest()
                raise RuntimeError("动作已取消")
            with self._runtime_move_response_lock:
                raw_response = self._runtime_move_response
            response = _json_safe_mapping(
                raw_response,
                "runtime_move_joints response",
            )
            audit_entry["response"] = response
            audit_entry["responded_at"] = datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            )

            if str(response.get("request_id") or "") != request_id:
                raise RuntimeError(f"{runtime_label}响应request_id不匹配，已拒绝运动")
            decision = str(response.get("decision") or "").strip().lower()
            allowed_decisions = {"approve", "decline", "abort"}
            if step["request_key"] in {
                STAGE7B_RUNTIME_REQUEST_KEY,
                MOVED_TRAY_RUNTIME_REQUEST_KEY,
                WAFER_PICK_XY_RUNTIME_REQUEST_KEY,
            }:
                allowed_decisions.update({"complete", "observe"})
            if decision not in allowed_decisions:
                raise RuntimeError(
                    "运行时响应decision必须是" + "/".join(sorted(allowed_decisions))
                )
            audit_entry["operator_decision"] = decision
            if decision == "observe":
                if step["request_key"] not in {
                    MOVED_TRAY_RUNTIME_REQUEST_KEY,
                    WAFER_PICK_XY_RUNTIME_REQUEST_KEY,
                }:
                    raise RuntimeError(f"{runtime_label}不允许observe响应")
                if str(response.get("calibration_sha256") or "").upper() != step["calibration_sha256"]:
                    raise RuntimeError(f"{runtime_label}观察响应的平面手眼hash不匹配")
                audit_entry["status"] = "observation_completed_no_motion"
                audit_entry["observation_reason"] = str(
                    response.get("reason") or "完成观察窗口，继续下一窗口"
                )
                self._save_manifest()
                self.progress.emit(f"{step['name']}：{audit_entry['observation_reason']}")
                return
            if decision == "complete":
                if str(response.get("calibration_sha256") or "").upper() != step["calibration_sha256"]:
                    raise RuntimeError(f"{runtime_label}完成响应的标定hash不匹配")
                if step["request_key"] == STAGE7B_RUNTIME_REQUEST_KEY and str(
                    response.get("fine_calibration_sha256") or ""
                ).upper() != step["fine_calibration_sha256"]:
                    raise RuntimeError(f"{runtime_label}完成响应的精细Jacobian hash不匹配")
                audit_entry["status"] = "session_completed_no_motion"
                audit_entry["completion_reason"] = str(
                    response.get("reason") or f"{runtime_label}达到终止条件"
                )
                self._runtime_session_complete = True
                self._save_manifest()
                self.progress.emit(f"{step['name']}：{runtime_label}已到达，正常结束且不再运动")
                return
            if decision == "decline":
                audit_entry["status"] = "declined_no_motion"
                self._save_manifest()
                self.progress.emit(f"{step['name']}：人员拒绝，本步不运动")
                return
            if decision == "abort":
                audit_entry["status"] = "aborted_no_motion"
                self._save_manifest()
                raise RuntimeError(
                    str(response.get("reason") or f"{runtime_label}运行时终止任务")
                )

            response_hash = str(
                response.get("calibration_sha256") or ""
            ).strip().upper()
            if response_hash != step["calibration_sha256"]:
                raise RuntimeError(f"{runtime_label}响应使用的Jacobian hash与任务锁定值不一致")
            proposal = _json_safe_mapping(
                response.get("proposal"),
                f"{runtime_label} proposal",
            )
            if proposal.get("motion_authorized") is not True:
                raise RuntimeError(f"{runtime_label} proposal未明确授权单步运动")
            if not str(proposal.get("proposal_id") or "").strip():
                raise RuntimeError(f"{runtime_label} proposal缺少proposal_id")
            if str(proposal.get("target_name") or "") != step["target_name"]:
                raise RuntimeError(f"{runtime_label} proposal目标不是本任务锁定槽位")
            audit_step = step
            if step["request_key"] == STAGE7B_RUNTIME_REQUEST_KEY:
                tier = str(proposal.get("model_tier") or "")
                if tier not in {"coarse_task11", "fine_task9"}:
                    raise RuntimeError(f"{runtime_label} proposal缺少有效的两级模型标识")
                if str(proposal.get("wide_calibration_sha256") or "").upper() != step["calibration_sha256"]:
                    raise RuntimeError(f"{runtime_label}宽域Jacobian hash与任务锁定值不一致")
                if str(proposal.get("fine_calibration_sha256") or "").upper() != step["fine_calibration_sha256"]:
                    raise RuntimeError(f"{runtime_label}精细Jacobian hash与任务锁定值不一致")
                audit_step = dict(step)
                if tier == "fine_task9":
                    audit_step.update(
                        local_extent_mm=2.0,
                        domain_margin_mm=0.2,
                        max_xy_step_norm_mm=min(step["max_xy_step_norm_mm"], 0.25),
                        max_xy_axis_mm=min(step["max_xy_axis_mm"], 0.25),
                        max_sequential_transient_xy_mm=min(
                            step["max_sequential_transient_xy_mm"], 0.50
                        ),
                    )
            if step["request_key"] == WAFER_PICK_XY_RUNTIME_REQUEST_KEY:
                phase = str(proposal.get("phase") or "")
                if phase not in {
                    "wafer_pick_xy_overhead",
                    "wafer_pick_final_tray_orientation",
                }:
                    raise RuntimeError(f"{runtime_label} proposal阶段标识无效")
                if phase == "wafer_pick_xy_overhead":
                    if proposal.get("xy_only") is not True:
                        raise RuntimeError(f"{runtime_label} proposal未声明XY-only")
                    if proposal.get("j4_only") is True:
                        raise RuntimeError(f"{runtime_label} XY阶段错误声明J4-only")
                    expected_locked_rz = step["required_rz_deg"]
                else:
                    if proposal.get("xy_only") is not False:
                        raise RuntimeError(f"{runtime_label} 最终方向阶段不得声明XY-only")
                    if proposal.get("j4_only") is not True:
                        raise RuntimeError(f"{runtime_label} 最终方向阶段未声明J4-only")
                    expected_locked_rz = step["final_rz_deg"]
                    audit_step = dict(step)
                    audit_step["_audit_mode"] = "j4_only"
                    audit_step["precompensate_rz"] = False
                forbidden_authorizations = {
                    "z_motion_authorized": proposal.get("z_motion_authorized"),
                    "vacuum_authorized": proposal.get("vacuum_authorized"),
                    "do_authorized": proposal.get("do_authorized"),
                }
                if any(value is not False for value in forbidden_authorizations.values()):
                    raise RuntimeError(
                        f"{runtime_label} proposal包含下降、真空或DO授权，已拒绝"
                    )
                try:
                    locked_j3 = float(proposal.get("locked_j3_mm"))
                    locked_rz = float(proposal.get("locked_rz_deg"))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RuntimeError(
                        f"{runtime_label} proposal缺少有效J3/Rz锁定值"
                    ) from exc
                if abs(locked_j3 - step["required_j3_mm"]) > 1e-9:
                    raise RuntimeError(f"{runtime_label} proposal改变了J3安全高度")
                if (
                    abs(
                        (locked_rz - expected_locked_rz + 180.0)
                        % 360.0
                        - 180.0
                    )
                    > 1e-9
                ):
                    raise RuntimeError(f"{runtime_label} proposal的目标绝对Rz与阶段不一致")
                proposal_gates = proposal.get("safety_gates") or {}
                if not proposal_gates or any(
                    not isinstance(gate, dict) or gate.get("passed") is not True
                    for gate in proposal_gates.values()
                ):
                    raise RuntimeError(f"{runtime_label} proposal存在未通过的视觉安全门")
            proposal_command_xy = _vector_values(
                (proposal.get("calculation") or {}).get(
                    "commanded_correction_xy_mm"
                ),
                2,
                f"{runtime_label} proposal commanded_correction_xy_mm",
            )
            target_joints = _joint_values(
                response.get("target_joints"),
                f"{runtime_label} target_joints",
            )
            displayed_target_audit = self._audit_runtime_target(
                displayed_state,
                target_joints,
                audit_step,
            )

            audit_entry["displayed_kinematic_audit"] = displayed_target_audit
            if displayed_target_audit.get("passed") is not True:
                failed = [
                    name
                    for name, gate in (
                        displayed_target_audit.get("gates") or {}
                    ).items()
                    if not isinstance(gate, dict) or gate.get("passed") is not True
                ]
                raise RuntimeError(
                    f"{runtime_label}目标未通过显示状态下的独立审计："
                    + (", ".join(failed) if failed else "未知门")
                )
            displayed_step_xy = _vector_values(
                displayed_target_audit.get("step_xy_mm"),
                2,
                f"{runtime_label} displayed step_xy_mm",
            )
            command_mismatch = math.hypot(
                displayed_step_xy[0] - proposal_command_xy[0],
                displayed_step_xy[1] - proposal_command_xy[1],
            )
            audit_entry["proposal_target_command_mismatch_mm"] = (
                command_mismatch
            )
            if command_mismatch > 0.002:
                raise RuntimeError(
                    f"{runtime_label}目标关节与已显示的XY修正不一致"
                    f"（差值{command_mismatch:.4f}mm）"
                )

            proposal_age_s = time.monotonic() - requested_monotonic_s
            audit_entry["proposal_age_s_at_revalidation"] = proposal_age_s
            if (
                proposal_age_s < 0.0
                or proposal_age_s > audit_step["proposal_max_age_s"]
            ):
                raise RuntimeError(
                    f"{runtime_label}提案已过期（{proposal_age_s:.2f}s > "
                    f"{audit_step['proposal_max_age_s']:.2f}s）"
                )

            fresh_state = self._read_runtime_motion_state(
                f"{step['name']} 下发前重新检查"
            )
            audit_entry["fresh_controller_state"] = fresh_state
            fresh_gates = self._runtime_controller_gates(
                fresh_state, step["request_key"]
            )
            audit_entry["fresh_controller_gates"] = fresh_gates
            failed_controller = [
                name for name, passed in fresh_gates.items() if not passed
            ]
            if self._stop_requested.is_set():
                failed_controller.append("worker_not_stopped")
            if failed_controller:
                raise RuntimeError(
                    f"{runtime_label}下发前控制器安全门失败："
                    + ", ".join(sorted(set(failed_controller)))
                )

            xy_drift, joint_drift = self._runtime_state_drift(
                displayed_state,
                fresh_state,
            )
            audit_entry["state_drift_since_display"] = {
                "xy_mm": xy_drift,
                "maximum_joint_deg_or_mm": joint_drift,
            }
            if xy_drift > audit_step["max_state_drift_xy_mm"] + 1e-12:
                raise RuntimeError(
                    f"{runtime_label}确认期间机械臂XY变化{xy_drift:.4f}mm，旧提案已作废"
                )
            if joint_drift > audit_step["max_state_drift_joint"] + 1e-12:
                raise RuntimeError(
                    f"{runtime_label}确认期间关节变化{joint_drift:.4f}，旧提案已作废"
                )

            kinematic_audit = self._audit_runtime_target(
                fresh_state,
                target_joints,
                audit_step,
            )
            audit_entry["fresh_kinematic_audit"] = kinematic_audit
            if kinematic_audit.get("passed") is not True:
                failed = [
                    name
                    for name, gate in (kinematic_audit.get("gates") or {}).items()
                    if not isinstance(gate, dict) or gate.get("passed") is not True
                ]
                raise RuntimeError(
                    f"{runtime_label}候选关节目标安全审计失败："
                    + (", ".join(failed) if failed else "未知门")
                )

            precompensation = kinematic_audit.get("rz_precompensation")
            if audit_step["precompensate_rz"]:
                if not isinstance(precompensation, dict):
                    raise RuntimeError(f"{runtime_label}缺少J4 Rz预补偿审计数据")
                precompensation_target = _joint_values(
                    precompensation.get("target_joints"),
                    f"{runtime_label} Rz precompensation target_joints",
                )
                precompensation_delta = abs(
                    float(precompensation.get("delta_j4_deg"))
                )
                precompensation_required = bool(
                    precompensation_delta > audit_step["move_tolerance"] + 1e-12
                )
            else:
                precompensation_target = list(fresh_state["joints"])
                precompensation_delta = 0.0
                precompensation_required = False
            audit_entry["rz_precompensation"] = {
                **(dict(precompensation) if isinstance(precompensation, dict) else {}),
                "required_by_tolerance": precompensation_required,
                "move_tolerance_deg": audit_step["move_tolerance"],
                "executed": False,
            }
            # Persist the complete authorization evidence before the first
            # physical command.  A crash or emergency stop therefore cannot
            # leave an unaudited movement behind.
            audit_entry["status"] = (
                "authorized_pending_rz_precompensation"
                if precompensation_required
                else "authorized_pending_motion"
            )
            audit_entry["target_joints"] = target_joints
            audit_entry["proposal"] = proposal
            self._save_manifest()

            motion_start_state = fresh_state
            if precompensation_required:
                audit_entry["physical_motion_started"] = True
                if not self._controller.goto_joints_sync(
                    f"{step['name']} / J4 Rz预补偿",
                    precompensation_target,
                    should_stop=self._stop_requested.is_set,
                    tolerance=audit_step["move_tolerance"],
                ):
                    audit_entry["status"] = "rz_precompensation_failed"
                    self._save_manifest()
                    raise RuntimeError(f"{runtime_label} J4 Rz预补偿运动失败")

                precompensated_state = self._read_runtime_motion_state(
                    f"{step['name']} J4 Rz预补偿后检查"
                )
                audit_entry["rz_precompensation"]["executed"] = True
                audit_entry["rz_precompensation"]["final_controller_state"] = (
                    precompensated_state
                )
                precompensation_controller_gates = self._runtime_controller_gates(
                    precompensated_state, step["request_key"]
                )
                precompensation_controller_gates["worker_not_stopped"] = (
                    not self._stop_requested.is_set()
                )
                audit_entry["rz_precompensation"]["controller_gates"] = (
                    precompensation_controller_gates
                )
                if not all(precompensation_controller_gates.values()):
                    audit_entry["status"] = "rz_precompensation_verification_failed"
                    self._save_manifest()
                    raise RuntimeError(f"{runtime_label} J4 Rz预补偿后控制器安全门失败")
                precompensation_joint_error = max(
                    abs(actual - target)
                    for actual, target in zip(
                        precompensated_state["joints"], precompensation_target
                    )
                )
                audit_entry["rz_precompensation"]["joint_error_max"] = (
                    precompensation_joint_error
                )
                if precompensation_joint_error > audit_step["move_tolerance"] + 1e-12:
                    audit_entry["status"] = "rz_precompensation_verification_failed"
                    self._save_manifest()
                    raise RuntimeError(
                        f"{runtime_label} J4 Rz预补偿到位误差"
                        f"{precompensation_joint_error:.4f}超过容差"
                    )
                post_precompensation_audit = self._audit_runtime_target(
                    precompensated_state,
                    target_joints,
                    audit_step,
                )
                audit_entry["post_precompensation_kinematic_audit"] = (
                    post_precompensation_audit
                )
                if post_precompensation_audit.get("passed") is not True:
                    audit_entry["status"] = "rz_precompensation_verification_failed"
                    self._save_manifest()
                    raise RuntimeError(
                        f"{runtime_label} J4 Rz预补偿后XY目标运动学复核失败"
                    )
                motion_start_state = precompensated_state
                audit_entry["status"] = (
                    "rz_precompensation_completed_pending_xy_motion"
                )
                self._save_manifest()

            audit_entry["physical_motion_started"] = True
            if not self._controller.goto_joints_sync(
                step["name"],
                target_joints,
                should_stop=self._stop_requested.is_set,
                tolerance=audit_step["move_tolerance"],
            ):
                audit_entry["status"] = "motion_failed"
                self._save_manifest()
                raise RuntimeError(f"移动到 {step['name']} 失败")
            final_state = self._read_runtime_motion_state(
                f"{step['name']} 到位后检查"
            )

            audit_entry["final_controller_state"] = final_state
            final_controller_gates = self._runtime_controller_gates(
                final_state, step["request_key"]
            )
            final_controller_gates["worker_not_stopped"] = (
                not self._stop_requested.is_set()
            )
            audit_entry["final_controller_gates"] = final_controller_gates
            failed_final_controller = [
                name
                for name, passed in final_controller_gates.items()
                if not passed
            ]
            if failed_final_controller:
                audit_entry["status"] = "final_verification_failed"
                self._save_manifest()
                raise RuntimeError(
                    f"{runtime_label}到位后控制器安全门失败："
                    + ", ".join(sorted(failed_final_controller))
                )
            final_joint_error = max(
                abs(actual - target)
                for actual, target in zip(final_state["joints"], target_joints)
            )
            audit_entry["final_joint_error_max"] = final_joint_error
            if final_joint_error > audit_step["move_tolerance"] + 1e-12:
                audit_entry["status"] = "final_verification_failed"
                self._save_manifest()
                raise RuntimeError(
                    f"{runtime_label}最终关节误差{final_joint_error:.4f}超过容差"
                )
            final_audit = self._audit_runtime_target(
                motion_start_state,
                final_state["joints"],
                audit_step,
            )
            audit_entry["actual_motion_audit"] = final_audit
            if final_audit.get("passed") is not True:
                audit_entry["status"] = "final_verification_failed"
                self._save_manifest()
                raise RuntimeError(f"{runtime_label}实际到位状态未通过运动安全复核")
            audit_entry["status"] = "motion_completed"
            self._save_manifest()
        except Exception as exc:
            if audit_entry.get("status") not in {
                "declined_no_motion",
                "aborted_no_motion",
                "stopped_while_waiting",
                "motion_failed",
                "final_verification_failed",
                "rz_precompensation_failed",
                "rz_precompensation_verification_failed",
            }:
                audit_entry["status"] = "rejected_no_motion"
            audit_entry["failure_reason"] = str(exc) or exc.__class__.__name__
            self._save_manifest()
            raise
        finally:
            with self._runtime_move_response_lock:
                if self._runtime_move_pending_request_id == request_id:
                    self._runtime_move_pending_request_id = None
                self._runtime_move_response = None
                self._runtime_move_response_event.clear()

    def _assert_joints(self, step: dict) -> None:
        state = self._read_state(step["name"])
        errors = [
            abs(actual - target)
            for actual, target in zip(state["joints"], step["joints"])
        ]
        if any(error > step["tolerance"] for error in errors):
            detail = ", ".join(f"J{i + 1}={error:.3f}" for i, error in enumerate(errors))
            raise RuntimeError(f"{step['name']} 起点检查失败（偏差 {detail}）")

    def _move_joints(self, step: dict) -> None:
        if "require_current_j3_mm" in step:
            state = self._read_state(f"{step['name']} 高度安全检查")
            difference = abs(state["joints"][2] - step["require_current_j3_mm"])
            if difference > step["j3_tolerance_mm"]:
                raise RuntimeError(
                    f"{step['name']} 已阻止 XY/R 移动：当前 J3 未回到浮动高度，"
                    f"高度偏差 {difference:.3f} mm"
                )
        if not self._controller.goto_joints_sync(
            step["name"],
            step["joints"],
            should_stop=self._stop_requested.is_set,
            tolerance=step["tolerance"],
        ):
            raise RuntimeError(f"移动到 {step['name']} 失败")

    def _set_do(self, step: dict) -> None:
        """Synchronously write one task-owned DO and persist the evidence.

        Turning an output on is fail-closed on controller connection, enable,
        alarm and E-stop state. Turning it off is always attempted so a fault
        cannot prevent the task from releasing an energized output.
        """
        channel = int(step["channel"])
        level = int(step["level"])
        event = {
            "sequence": len(self._manifest.setdefault("do_events", [])) + 1,
            "name": step["name"],
            "requested_at": datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            ),
            "channel": channel,
            "level": level,
            "status": "requested",
        }
        self._manifest["do_events"].append(event)
        self._touched_do_channels.add(channel)
        self._uncertain_do_channels.add(channel)

        if level == 1:
            state = self._read_state(f"{step['name']} DO开启前安全检查")
            safety = state["controller_safety"]
            failed = []
            if safety.get("connected") is not True:
                failed.append("controller_connected")
            if safety.get("effectively_enabled") is not True:
                failed.append("controller_enabled")
            if safety.get("estop") is True:
                failed.append("estop_clear")
            if safety.get("soft_estop") is True:
                failed.append("soft_estop_clear")
            if int(safety.get("warn", -1)) != 0 or safety.get("need_clear") is True:
                failed.append("alarm_clear")
            event["preflight"] = {
                "passed": not failed,
                "failed_gates": failed,
                "controller_safety": safety,
            }
            if failed:
                event["status"] = "rejected"
                event["failure_reason"] = ", ".join(failed)
                self._save_manifest()
                raise RuntimeError(
                    f"{step['name']} 被DO开启安全门拒绝：{', '.join(failed)}"
                )

        self._save_manifest()
        writer = getattr(self._controller, "set_do_sync", None)
        if not callable(writer):
            event["status"] = "failed"
            event["failure_reason"] = "控制器不支持同步set_do_sync"
            self._save_manifest()
            raise RuntimeError(f"{step['name']}：控制器不支持同步DO写入")
        try:
            ok, detail = writer(channel, level)
        except Exception as exc:
            ok, detail = False, str(exc)
        event["completed_at"] = datetime.now().astimezone().isoformat(
            timespec="milliseconds"
        )
        event["controller_message"] = str(detail or "")[:1000]
        if not ok:
            event["status"] = "failed"
            self._save_manifest()
            raise RuntimeError(
                f"{step['name']} 写入失败：{str(detail or '未知错误')[:200]}"
            )
        event["status"] = "completed"
        self._known_do_levels[channel] = level
        self._uncertain_do_channels.discard(channel)
        self._save_manifest()

    def _cleanup_task_outputs(self, *, task_ok: bool) -> tuple[bool, str]:
        """Clear outputs that may remain on after an interrupted task."""
        active = {
            channel
            for channel, level in self._known_do_levels.items()
            if int(level) != 0
        }
        cleanup_channels = sorted(active | self._uncertain_do_channels)
        if not task_ok:
            cleanup_channels = sorted(
                set(cleanup_channels) | self._touched_do_channels
            )
        if not cleanup_channels:
            if self._manifest:
                self._manifest["do_cleanup"] = {
                    "required": False,
                    "passed": True,
                    "channels": [],
                }
            return True, "not required"

        cleaner = getattr(self._controller, "zero_do_channels_sync", None)
        if not callable(cleaner):
            ok, detail = False, "控制器不支持zero_do_channels_sync"
        else:
            try:
                ok, detail = cleaner(cleanup_channels)
            except Exception as exc:
                ok, detail = False, str(exc)
        if ok:
            for channel in cleanup_channels:
                self._known_do_levels[channel] = 0
                self._uncertain_do_channels.discard(channel)
        if self._manifest:
            self._manifest["do_cleanup"] = {
                "required": True,
                "passed": bool(ok),
                "channels": cleanup_channels,
                "finished_at": datetime.now().astimezone().isoformat(
                    timespec="milliseconds"
                ),
                "controller_message": str(detail or "")[:1000],
            }
        return bool(ok), str(detail or "")

    def _execute_step(self, step: dict) -> None:
        kind = step["type"]
        if kind == "assert_joints":
            self._assert_joints(step)
        elif kind == "move_joints":
            self._move_joints(step)
        elif kind == "runtime_move_joints":
            self._runtime_move_joints(step)
        elif kind == "move_xyzr":
            if not self._controller.move_xyzr_sync(
                step["name"],
                x_mm=step["x_mm"],
                y_mm=step["y_mm"],
                z_mm=step["z_mm"],
                r_deg=step["r_deg"],
                should_stop=self._stop_requested.is_set,
            ):
                raise RuntimeError(f"执行 {step['name']} 失败")
        elif kind == "set_do":
            self._set_do(step)
        elif kind == "wait":
            if not self._interruptible_wait(step["seconds"]):
                raise RuntimeError("动作已取消")
        elif kind == "capture":
            self._capture(step["source"])
        elif kind == "start_video":
            self._start_video(step)
        elif kind == "stop_video":
            self._stop_video(step["source"])
        elif kind == "record_point":
            self._record_point(step["name"])

    def run(self) -> None:
        ok = False
        message = "动作未开始"
        try:
            self._output_dir.mkdir(parents=True, exist_ok=False)
            self._manifest = {
                "schema_version": 1,
                "task_name": self._task["name"],
                "description": self._task["description"],
                "started_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "status": "running",
                "coordinate_convention": {
                    "world_x_positive": "机械臂控制器返回的世界 +X",
                    "world_y_positive": "P05 到 P00",
                    "world_z_positive": "机械臂升高、远离平台",
                    "rz_definition": "相机方向相对世界 -Y 的有符号角度",
                    "positive_rotation": "从上方看逆时针（从世界 -Y 转向 +X）",
                },
                "camera_model": dict(self._task["camera_model"]),
                "camera_capture_settings_requested": {
                    str(source): dict(setting)
                    for source, setting in sorted(
                        self._task["camera_capture_settings"].items()
                    )
                },
                "points": [],
                "photos": [],
                "videos": [],
                "runtime_moves": [],
                "do_events": [],
            }
            if self._repeatable:
                self._manifest["collection_mode"] = "operator_repeated_scan"
                self._manifest["operator_checkpoints"] = []
            self._save_manifest()

            capture_sources = {
                step["source"]
                for step in self._task["actions"]
                if step["type"] == "capture"
            }
            video_sources = {
                step["source"]
                for step in self._task["actions"]
                if step["type"] in {"start_video", "stop_video"}
            }
            sources = sorted(capture_sources | video_sources)
            pool_sources = sorted(
                video_sources | (capture_sources if self._snapshot_source is None else set())
            )
            if self._task["camera_capture_settings"] and self._snapshot_source is not None:
                raise RuntimeError(
                    "任务要求硬件相机设置，但当前使用外部 snapshot_source，无法在运动前验证曝光"
                )
            if pool_sources:
                # Application-wide policy: every task-owned camera starts in
                # automatic exposure.  Imported tasks cannot opt into a fixed
                # value; the only manual path is the hand-eye UI slider.
                effective_capture_settings = {
                    source: {"auto_exposure": True} for source in pool_sources
                }
                self._manifest["camera_capture_settings_requested"] = {
                    str(source): dict(setting)
                    for source, setting in sorted(effective_capture_settings.items())
                }
                self._save_manifest()
                self._camera_pool = CameraSourcePool()
                opened, error = self._camera_pool.open_sources(
                    pool_sources,
                    effective_capture_settings,
                )
                self._manifest["camera_sources_resolved"] = (
                    self._camera_pool.camera_sources_report()
                )
                # Persist attempted mode writes even when a driver explicitly
                # rejects auto mode, so a stopped run contains the exact
                # request/readback evidence instead of only a generic error.
                self._manifest["camera_capture_settings_applied"] = (
                    self._camera_pool.capture_settings_report()
                )
                self._save_manifest()
                if not opened:
                    raise RuntimeError(error)
            self.progress.emit(
                "相机源检查完成，动作即将开始" if sources else "动作即将开始（无拍照步骤）"
            )

            total = len(self._task["actions"])
            index = 0
            while index < total:
                step = self._task["actions"][index]
                if self._stop_requested.is_set():
                    raise RuntimeError("动作已取消")
                if self._camera_pool is not None:
                    self._camera_pool.check_video_error()
                label = step.get("name") or step["type"]
                round_prefix = (
                    f"姿态{self._collection_round} " if self._repeatable else ""
                )
                self.progress.emit(f"{round_prefix}{index + 1}/{total} {label}")
                if step["type"] == "operator_checkpoint":
                    if self._wait_for_operator_checkpoint(step):
                        self._collection_round += 1
                        index = int(step["repeat_from_index"])
                        continue
                    break
                self._execute_step(step)
                if self._runtime_session_complete:
                    break
                index += 1

            ok = True
            photo_total = sum(self._photo_counts.values())
            message = (
                f"动作完成，共记录 {len(self._manifest['points'])} 个点、"
                f"保存 {photo_total} 张照片、{len(self._manifest['videos'])} 段录像、"
                f"执行 {len(self._manifest['do_events'])} 次DO写入"
            )
        except FileExistsError:
            message = f"输出文件夹已存在：{self._output_dir}"
        except Exception as exc:  # noqa: BLE001 - displayed safely in the UI
            message = str(exc) or exc.__class__.__name__
        finally:
            for source in list(self._active_video_sources):
                try:
                    self._stop_video(source)
                except Exception:
                    self._active_video_sources.discard(source)
            if self._camera_pool is not None:
                self._camera_pool.close()
            if self._manifest:
                cleanup_ok, cleanup_detail = self._cleanup_task_outputs(task_ok=ok)
                if not cleanup_ok:
                    ok = False
                    message = (
                        f"{message}；安全清零DO失败："
                        f"{str(cleanup_detail or '未知错误')[:200]}"
                    )
                self._manifest["status"] = "completed" if ok else "stopped"
                self._manifest["finished_at"] = datetime.now().astimezone().isoformat(
                    timespec="milliseconds"
                )
                self._manifest["result"] = message
                try:
                    self._save_manifest()
                except Exception:
                    pass
            self.run_finished.emit(ok, message, str(self._output_dir))
