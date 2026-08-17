"""Runtime Tray-to-world registration for moved-tray fixed-height XY control.

The installed planar hand-eye provides the fixed camera-to-forearm rotation.
Each new Stage-3 pose and Stage-4 suction point then produces one independent
planar ``^W T_T`` estimate.  Five unique, post-request measurements are robustly
aggregated.  This module is pure computation: it imports no Qt, controller, DO,
vacuum, or camera backend.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scara.pipeline.kinematics import forearm_pose_W_F

from .handeye_interaction import SuctionTargetModel, sha256_file


@dataclass(frozen=True)
class RuntimeRegistrationConfig:
    required_frame_count: int = 5
    minimum_marker_count: int = 3
    recommended_marker_count: int = 4
    maximum_state_age_s: float = 1.0
    maximum_stationary_xy_spread_mm: float = 0.20
    maximum_stationary_joint_spread: float = 0.20
    normal_origin_rms_mm: float = 0.30
    anomaly_origin_rms_mm: float = 0.50
    normal_yaw_rms_deg: float = 0.10
    anomaly_yaw_rms_deg: float = 0.20
    normal_p22_uncertainty_mm: float = 0.50
    anomaly_p22_uncertainty_mm: float = 0.80
    supported_translation_mm: float = 10.0
    supported_yaw_deg: float = 5.0
    hard_translation_mm: float = 10.5
    hard_yaw_deg: float = 5.25
    anomaly_translation_mm: float = 9.0
    anomaly_yaw_deg: float = 4.5
    probe_maximum_origin_disagreement_mm: float = 0.75
    probe_maximum_yaw_disagreement_deg: float = 0.25
    probe_maximum_p22_uncertainty_mm: float = 0.60


DEFAULT_REGISTRATION_CONFIG = RuntimeRegistrationConfig()


def _finite_vector(value: Any, length: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != int(length) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label}必须包含{length}个有限数值")
    return result


def _finite_matrix(value: Any, shape: tuple[int, int], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{label}必须是有限{shape[0]}x{shape[1]}矩阵")
    return result


def _angle_delta_deg(left: float, right: float) -> float:
    return (float(left) - float(right) + 180.0) % 360.0 - 180.0


def _circular_mean_deg(values: Sequence[float]) -> float:
    radians = np.radians(np.asarray(values, dtype=np.float64))
    sine = float(np.mean(np.sin(radians)))
    cosine = float(np.mean(np.cos(radians)))
    if abs(sine) + abs(cosine) < 1e-12:
        raise ValueError("yaw样本退化，无法求圆均值")
    return math.degrees(math.atan2(sine, cosine))


def _rotation_z(degrees: float) -> np.ndarray:
    angle = math.radians(float(degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _yaw_deg(rotation: np.ndarray) -> float:
    return math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))


def _gate(passed: bool, actual: Any, limit: str, note: str = "") -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "actual": actual,
        "limit": str(limit),
        "note": str(note),
    }


def _current_input_paths(project_root: Path) -> tuple[Path, Path]:
    root = Path(project_root)
    return (
        root / "src/scara/calib/camera1_intrinsics.json",
        root / "src/scara/calib/tray_board_geometry.json",
    )


def load_planar_handeye(
    project_root: Path,
    suction: SuctionTargetModel,
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    """Load the installed model and reject every stale or broadened condition."""

    root = Path(project_root)
    calibration_path = (
        Path(path)
        if path is not None
        else root / "src/scara/calib/camera1_forearm_planar_handeye.json"
    )
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise RuntimeError(
            "缺少camera1_forearm_planar_handeye.json；请先用现有数据拟合，"
            "若质量门失败则运行Task13"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"平面手眼标定JSON损坏：{exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("status") != "success":
        raise RuntimeError("平面手眼标定不是success")
    if int(payload.get("schema_version", 0)) < 1:
        raise RuntimeError("平面手眼标定schema过旧")
    scope = payload.get("scope") or {}
    if scope.get("planar_xy_supported") is not True or scope.get("z_supported") is not False:
        raise RuntimeError("平面手眼标定适用范围声明不正确")
    if abs(float(scope.get("required_j3_mm")) - float(suction.imaging_j3_mm)) > 0.05:
        raise RuntimeError("平面手眼固定J3与当前Stage4不一致")
    camera = payload.get("camera") or {}
    resolution = camera.get("resolution") or {}
    if int(camera.get("source_index", -1)) != int(suction.camera_source):
        raise RuntimeError("平面手眼相机源与当前Stage4不一致")
    if (int(resolution.get("width", 0)), int(resolution.get("height", 0))) != tuple(suction.resolution):
        raise RuntimeError("平面手眼分辨率与当前Stage4不一致")
    quality_gates = payload.get("quality_gates") or {}
    if not quality_gates or not all(
        isinstance(gate, Mapping) and gate.get("passed") is True
        for gate in quality_gates.values()
    ):
        raise RuntimeError("平面手眼质量门不完整或未全部通过")
    intrinsics_path, geometry_path = _current_input_paths(root)
    locked = payload.get("locked_inputs") or {}
    expected = {
        "camera_intrinsics_sha256": sha256_file(intrinsics_path),
        "tray_geometry_sha256": sha256_file(geometry_path),
        "suction_target_sha256": suction.source_sha256,
    }
    for key, value in expected.items():
        if str(locked.get(key) or "").upper() != str(value).upper():
            raise RuntimeError(f"平面手眼锁定输入已变化：{key}")
    rotation = _finite_matrix(payload.get("R_F_C"), (3, 3), "R_F_C")
    if np.linalg.norm(rotation.T @ rotation - np.eye(3)) > 1e-5 or abs(np.linalg.det(rotation) - 1.0) > 1e-5:
        raise RuntimeError("R_F_C不是有效旋转矩阵")
    probe_poses = payload.get("prevalidated_probe_poses") or []
    if not isinstance(probe_poses, list) or len(probe_poses) < 3:
        raise RuntimeError("平面手眼标定缺少3个预验证观察姿态")
    for index, probe in enumerate(probe_poses[:3], start=1):
        if not isinstance(probe, Mapping):
            raise RuntimeError(f"第{index}个预验证观察姿态不是对象")
        _finite_vector(probe.get("tray_xy_mm"), 2, f"probe {index} tray_xy_mm")
        _finite_vector(probe.get("world_xy_mm"), 2, f"probe {index} world_xy_mm")
        _finite_vector(probe.get("joints"), 4, f"probe {index} joints")
        if not str(probe.get("slot") or ""):
            raise RuntimeError(f"第{index}个预验证观察姿态缺少slot")
    result = dict(payload)
    installation_check = payload.get("installation_check") or {}
    expired = installation_check.get("expired") is True
    valid_until = installation_check.get("valid_until")
    if isinstance(valid_until, str) and valid_until.strip():
        try:
            deadline = datetime.fromisoformat(valid_until.strip())
            if deadline.tzinfo is None:
                deadline = deadline.astimezone()
            expired = expired or datetime.now().astimezone() > deadline
        except ValueError as exc:
            raise RuntimeError("平面手眼installation_check.valid_until格式无效") from exc
    result["_installation_check_expired"] = bool(expired)
    result["_source_path"] = str(calibration_path.resolve())
    result["_source_sha256"] = sha256_file(calibration_path)
    return result


def estimate_transform_W_T(
    transform_C_T: Sequence[Sequence[float]],
    joints: Sequence[float],
    world_suction_xy_mm: Sequence[float],
    R_F_C: Sequence[Sequence[float]],
    suction_point_C_mm: Sequence[float],
) -> dict[str, Any]:
    """Estimate one planar ``^W T_T`` from one stationary observation."""

    T_C_T = _finite_matrix(transform_C_T, (4, 4), "T_C_T")
    joint_values = _finite_vector(joints, 4, "joints")
    world_xy = _finite_vector(world_suction_xy_mm, 2, "world suction XY")
    rotation_F_C = _finite_matrix(R_F_C, (3, 3), "R_F_C")
    suction_C = _finite_vector(suction_point_C_mm, 3, "p_C_S")
    rotation_W_F_planar = np.asarray(
        forearm_pose_W_F(float(joint_values[0]), float(joint_values[1])),
        dtype=np.float64,
    )[:2, :2]
    rotation_W_F = np.eye(3, dtype=np.float64)
    rotation_W_F[:2, :2] = rotation_W_F_planar
    raw_rotation_W_T = rotation_W_F @ rotation_F_C @ T_C_T[:3, :3]
    yaw_world_from_tray = _yaw_deg(raw_rotation_W_T)
    rotation_W_T = _rotation_z(yaw_world_from_tray)
    suction_T_h = np.linalg.inv(T_C_T) @ np.concatenate((suction_C, [1.0]))
    if abs(float(suction_T_h[3])) < 1e-12:
        raise ValueError("p_C_S逆变换齐次尺度为0")
    suction_T = suction_T_h[:3] / suction_T_h[3]
    origin_world_xy = world_xy - rotation_W_T[:2, :2] @ suction_T[:2]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_W_T
    transform[:2, 3] = origin_world_xy
    return {
        "transform_W_T": transform.astype(float).tolist(),
        "origin_world_xy_mm": origin_world_xy.astype(float).tolist(),
        "yaw_world_from_tray_deg": float(yaw_world_from_tray),
        "suction_point_T_mm": suction_T.astype(float).tolist(),
        "forearm_alpha_deg": float(joint_values[0] + joint_values[1]),
    }


def _stage2_registration(geometry: Mapping[str, Any]) -> tuple[np.ndarray, float]:
    frame = geometry.get("tray_frame") or {}
    origin = _finite_vector(frame.get("origin_mechanical_xy_mm"), 2, "Stage2 origin")
    rotation = _finite_matrix(frame.get("rotation_mechanical_from_tray"), (3, 3), "Stage2 rotation")
    return origin, _yaw_deg(rotation)


def _slot_point(geometry: Mapping[str, Any], target_name: str) -> np.ndarray:
    slots = geometry.get("slots") or {}
    if target_name not in slots:
        raise ValueError(f"Tray geometry缺少{target_name}")
    return _finite_vector(slots[target_name], 3, target_name)


def build_runtime_tray_registration(
    samples: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    suction: SuctionTargetModel,
    geometry: Mapping[str, Any],
    *,
    requested_monotonic_s: float,
    method: str = "single_stationary_pose_5_frames",
    installation_check_expired: bool = False,
    camera_reconnected: bool = False,
    previous_registration_drifted: bool = False,
    config: RuntimeRegistrationConfig = DEFAULT_REGISTRATION_CONFIG,
) -> dict[str, Any]:
    """Aggregate five new measurements and classify pass/probe/reject."""

    rows = list(samples)
    ids = [str(row.get("measurement_id") or "") for row in rows]
    accepted = [row.get("accepted") is True for row in rows]
    captured = [float(row.get("captured_monotonic_s", math.nan)) for row in rows]
    state_ages = [float(row.get("robot_state_age_s", math.inf)) for row in rows]
    marker_counts = [int(row.get("used_marker_count", 0)) for row in rows]
    joints_rows: list[np.ndarray] = []
    xy_rows: list[np.ndarray] = []
    estimates: list[dict[str, Any]] = []
    estimate_error: str | None = None
    for row in rows:
        try:
            joints = _finite_vector(row.get("current_joints"), 4, "sample joints")
            xy = _finite_vector(row.get("current_robot_xy_mm"), 2, "sample world XY")
            estimate = estimate_transform_W_T(
                row.get("tray_transform_C_T"),
                joints,
                xy,
                calibration.get("R_F_C"),
                suction.p_C_S_mm,
            )
            joints_rows.append(joints)
            xy_rows.append(xy)
            estimates.append(estimate)
        except Exception as exc:  # noqa: BLE001 - represented by fail-closed gate
            estimate_error = str(exc)
            break

    exact_frame_count = len(rows) == config.required_frame_count
    unique_and_post_request = (
        exact_frame_count
        and all(ids)
        and len(set(ids)) == config.required_frame_count
        and all(math.isfinite(value) and value >= float(requested_monotonic_s) for value in captured)
    )
    robot_xy_spread = (
        float(np.max(np.linalg.norm(np.asarray(xy_rows) - np.median(np.asarray(xy_rows), axis=0), axis=1)))
        if len(xy_rows) == len(rows) and xy_rows else math.inf
    )
    joint_spread = (
        float(np.max(np.ptp(np.asarray(joints_rows), axis=0)))
        if len(joints_rows) == len(rows) and joints_rows else math.inf
    )
    origins = np.asarray([item["origin_world_xy_mm"] for item in estimates], dtype=np.float64) if estimates else np.empty((0, 2))
    yaws = [float(item["yaw_world_from_tray_deg"]) for item in estimates]
    origin_centre = np.median(origins, axis=0) if len(origins) else np.asarray([math.nan, math.nan])
    yaw_centre = _circular_mean_deg(yaws) if yaws else math.nan
    origin_rms = (
        float(math.sqrt(float(np.mean(np.sum((origins - origin_centre) ** 2, axis=1)))))
        if len(origins) else math.inf
    )
    yaw_residuals = [abs(_angle_delta_deg(value, yaw_centre)) for value in yaws]
    yaw_rms = (
        float(math.sqrt(float(np.mean(np.square(yaw_residuals)))))
        if yaw_residuals else math.inf
    )
    p22_T = _slot_point(geometry, "P22")
    p22_predictions = np.asarray(
        [
            np.asarray(item["origin_world_xy_mm"])
            + _rotation_z(item["yaw_world_from_tray_deg"])[:2, :2] @ p22_T[:2]
            for item in estimates
        ],
        dtype=np.float64,
    ) if estimates else np.empty((0, 2))
    p22_centre = np.median(p22_predictions, axis=0) if len(p22_predictions) else np.asarray([math.nan, math.nan])
    p22_uncertainty = (
        float(math.sqrt(float(np.mean(np.sum((p22_predictions - p22_centre) ** 2, axis=1)))))
        if len(p22_predictions) else math.inf
    )
    old_origin, old_yaw = _stage2_registration(geometry)
    translation_delta = origin_centre - old_origin
    translation_norm = float(np.linalg.norm(translation_delta))
    yaw_delta = float(_angle_delta_deg(yaw_centre, old_yaw))

    hard_gates = {
        "required_unique_post_request_frames": _gate(
            unique_and_post_request,
            {"count": len(rows), "unique": len(set(ids)), "ids": ids},
            f"exactly {config.required_frame_count} unique frames captured after request",
        ),
        "stage3_pass": _gate(all(accepted) and exact_frame_count, accepted, "all true"),
        "minimum_marker_count": _gate(
            exact_frame_count and all(value >= config.minimum_marker_count for value in marker_counts),
            marker_counts,
            f"each >={config.minimum_marker_count}; recommend >={config.recommended_marker_count}",
        ),
        "robot_state_freshness": _gate(
            exact_frame_count and all(0.0 <= value <= config.maximum_state_age_s for value in state_ages),
            state_ages,
            f"each 0..{config.maximum_state_age_s:.2f} s",
        ),
        "robot_stationary_xy": _gate(
            robot_xy_spread <= config.maximum_stationary_xy_spread_mm,
            robot_xy_spread,
            f"<={config.maximum_stationary_xy_spread_mm:.2f} mm",
        ),
        "robot_stationary_joints": _gate(
            joint_spread <= config.maximum_stationary_joint_spread,
            joint_spread,
            f"<={config.maximum_stationary_joint_spread:.2f} deg/mm",
        ),
        "registration_computable": _gate(
            estimate_error is None and len(estimates) == config.required_frame_count,
            estimate_error or len(estimates),
            f"{config.required_frame_count} finite transforms",
        ),
        "translation_hard_scope": _gate(
            translation_norm <= config.hard_translation_mm,
            translation_norm,
            f"<={config.hard_translation_mm:.2f} mm",
        ),
        "yaw_hard_scope": _gate(
            abs(yaw_delta) <= config.hard_yaw_deg,
            abs(yaw_delta),
            f"<={config.hard_yaw_deg:.2f} deg",
        ),
        "origin_dispersion_hard": _gate(
            origin_rms <= config.anomaly_origin_rms_mm,
            origin_rms,
            f"<={config.anomaly_origin_rms_mm:.2f} mm",
        ),
        "yaw_dispersion_hard": _gate(
            yaw_rms <= config.anomaly_yaw_rms_deg,
            yaw_rms,
            f"<={config.anomaly_yaw_rms_deg:.2f} deg",
        ),
        "p22_uncertainty_hard": _gate(
            p22_uncertainty <= config.anomaly_p22_uncertainty_mm,
            p22_uncertainty,
            f"<={config.anomaly_p22_uncertainty_mm:.2f} mm",
        ),
    }
    hard_passed = all(gate["passed"] for gate in hard_gates.values())
    normal_gates = {
        "origin_rms": _gate(origin_rms <= config.normal_origin_rms_mm, origin_rms, f"<={config.normal_origin_rms_mm:.2f} mm"),
        "yaw_rms": _gate(yaw_rms <= config.normal_yaw_rms_deg, yaw_rms, f"<={config.normal_yaw_rms_deg:.2f} deg"),
        "p22_position_uncertainty": _gate(p22_uncertainty <= config.normal_p22_uncertainty_mm, p22_uncertainty, f"<={config.normal_p22_uncertainty_mm:.2f} mm"),
        "supported_translation": _gate(translation_norm <= config.supported_translation_mm, translation_norm, f"<={config.supported_translation_mm:.2f} mm"),
        "supported_yaw": _gate(abs(yaw_delta) <= config.supported_yaw_deg, abs(yaw_delta), f"<={config.supported_yaw_deg:.2f} deg"),
    }
    anomaly_reasons: list[str] = []
    if origin_rms > config.normal_origin_rms_mm:
        anomaly_reasons.append("origin_rms_borderline")
    if yaw_rms > config.normal_yaw_rms_deg:
        anomaly_reasons.append("yaw_rms_borderline")
    if p22_uncertainty > config.normal_p22_uncertainty_mm:
        anomaly_reasons.append("p22_uncertainty_borderline")
    if translation_norm >= config.anomaly_translation_mm:
        anomaly_reasons.append("translation_near_scope_limit")
    if abs(yaw_delta) >= config.anomaly_yaw_deg:
        anomaly_reasons.append("yaw_near_scope_limit")
    if camera_reconnected:
        anomaly_reasons.append("camera_reconnected")
    if installation_check_expired:
        anomaly_reasons.append("installation_check_expired")
    if previous_registration_drifted:
        anomaly_reasons.append("previous_registration_drifted")
    normal_passed = hard_passed and all(gate["passed"] for gate in normal_gates.values())
    requires_probe = bool(hard_passed and (anomaly_reasons or not normal_passed))
    status = "success" if normal_passed and not requires_probe else ("requires_three_pose_probe" if requires_probe else "rejected")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rotation_z(yaw_centre) if math.isfinite(yaw_centre) else np.eye(3)
    if np.all(np.isfinite(origin_centre)):
        transform[:2, 3] = origin_centre
    return {
        "schema_version": 1,
        "status": status,
        "registered_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "method": str(method),
        "session_only": True,
        "transform_W_T": transform.astype(float).tolist(),
        "origin_world_xy_mm": origin_centre.astype(float).tolist(),
        "yaw_world_from_tray_deg": float(yaw_centre),
        "relative_to_stage2": {
            "translation_xy_mm": translation_delta.astype(float).tolist(),
            "translation_norm_mm": translation_norm,
            "yaw_deg": yaw_delta,
        },
        "measurement_ids": ids,
        "measurement_count": len(rows),
        "per_frame_estimates": estimates,
        "dispersion": {
            "origin_rms_mm": origin_rms,
            "yaw_rms_deg": yaw_rms,
            "p22_position_uncertainty_mm": p22_uncertainty,
            "p22_world_xy_mm": p22_centre.astype(float).tolist(),
        },
        "calibration": {
            "path": str(calibration.get("_source_path") or ""),
            "sha256": str(calibration.get("_source_sha256") or ""),
            "locked_inputs": dict(calibration.get("locked_inputs") or {}),
        },
        "hard_gates": hard_gates,
        "normal_gates": normal_gates,
        "requires_three_pose_probe": requires_probe,
        "anomaly_reasons": anomaly_reasons,
        "configuration": asdict(config),
    }


def fuse_three_pose_registrations(
    registrations: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Any],
    *,
    config: RuntimeRegistrationConfig = DEFAULT_REGISTRATION_CONFIG,
) -> dict[str, Any]:
    """Fuse three independently observed registrations after operator consent."""

    rows = list(registrations)
    if len(rows) != 3:
        raise ValueError("三姿态复核必须恰好包含3个登记结果")
    if any(row.get("status") == "rejected" for row in rows):
        raise ValueError("三姿态复核包含硬门拒绝结果")
    origins = np.asarray([_finite_vector(row.get("origin_world_xy_mm"), 2, "probe origin") for row in rows])
    yaws = [float(row.get("yaw_world_from_tray_deg")) for row in rows]
    centre_origin = np.median(origins, axis=0)
    centre_yaw = _circular_mean_deg(yaws)
    origin_disagreement = float(np.max(np.linalg.norm(origins - centre_origin, axis=1)))
    yaw_disagreement = max(abs(_angle_delta_deg(value, centre_yaw)) for value in yaws)
    p22_T = _slot_point(geometry, "P22")
    predictions = np.asarray(
        [origin + _rotation_z(yaw)[:2, :2] @ p22_T[:2] for origin, yaw in zip(origins, yaws)]
    )
    p22_centre = np.median(predictions, axis=0)
    p22_uncertainty = float(math.sqrt(float(np.mean(np.sum((predictions - p22_centre) ** 2, axis=1)))))
    gates = {
        "probe_origin_agreement": _gate(origin_disagreement <= config.probe_maximum_origin_disagreement_mm, origin_disagreement, f"<={config.probe_maximum_origin_disagreement_mm:.2f} mm"),
        "probe_yaw_agreement": _gate(yaw_disagreement <= config.probe_maximum_yaw_disagreement_deg, yaw_disagreement, f"<={config.probe_maximum_yaw_disagreement_deg:.2f} deg"),
        "fused_p22_uncertainty": _gate(p22_uncertainty <= config.probe_maximum_p22_uncertainty_mm, p22_uncertainty, f"<={config.probe_maximum_p22_uncertainty_mm:.2f} mm"),
    }
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rotation_z(centre_yaw)
    transform[:2, 3] = centre_origin
    old_origin, old_yaw = _stage2_registration(geometry)
    translation_delta = centre_origin - old_origin
    return {
        "schema_version": 1,
        "status": "success" if all(gate["passed"] for gate in gates.values()) else "rejected",
        "registered_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "method": "operator_confirmed_three_pose_fusion",
        "session_only": True,
        "transform_W_T": transform.astype(float).tolist(),
        "origin_world_xy_mm": centre_origin.astype(float).tolist(),
        "yaw_world_from_tray_deg": float(centre_yaw),
        "relative_to_stage2": {
            "translation_xy_mm": translation_delta.astype(float).tolist(),
            "translation_norm_mm": float(np.linalg.norm(translation_delta)),
            "yaw_deg": float(_angle_delta_deg(centre_yaw, old_yaw)),
        },
        "measurement_ids": [identifier for row in rows for identifier in (row.get("measurement_ids") or [])],
        "dispersion": {
            "maximum_origin_disagreement_mm": origin_disagreement,
            "maximum_yaw_disagreement_deg": yaw_disagreement,
            "p22_position_uncertainty_mm": p22_uncertainty,
            "p22_world_xy_mm": p22_centre.astype(float).tolist(),
        },
        "quality_gates": gates,
        "component_registrations": rows,
    }


__all__ = [
    "DEFAULT_REGISTRATION_CONFIG",
    "RuntimeRegistrationConfig",
    "build_runtime_tray_registration",
    "estimate_transform_W_T",
    "fuse_three_pose_registrations",
    "load_planar_handeye",
]
