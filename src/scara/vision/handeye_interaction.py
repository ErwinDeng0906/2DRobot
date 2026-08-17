"""Read-only hand-eye evaluation and overlay rendering for camera 1.

This module consumes the approved intrinsics, Tray geometry, Stage-3 tracked
pose and Task-8 suction target.  It projects a selected slot and reports the
image error.  No controller or motion backend is imported here.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from scara.pipeline.kinematics import rz_of

from .tray_pose_estimator import CameraIntrinsics
from .tray_pose_tracker import TrackedTrayPose
from .full_tray_positioning import metric_suction_error_in_tray
from .xy_image_jacobian import (
    REQUIRED_XY_JACOBIAN_QUALITY_GATES,
    correction_command_xy_mm,
)


@dataclass(frozen=True)
class SuctionTargetModel:
    source_path: Path
    source_sha256: str
    camera_source: int
    resolution: tuple[int, int]
    p_C_S_mm: tuple[float, float, float]
    target_pixel_px: tuple[float, float]
    working_plane_z_T_mm: float
    imaging_j3_mm: float
    rz_mean_deg: float


@dataclass(frozen=True)
class HandEyeEvaluation:
    target_name: str
    accepted: bool
    reason: str
    slot_pixel_px: Optional[tuple[float, float]]
    suction_target_pixel_px: tuple[float, float]
    image_error_px: Optional[tuple[float, float]]
    image_error_norm_px: Optional[float]
    alignment_threshold_px: float
    aligned: Optional[bool]
    correction_xy_mm: Optional[tuple[float, float]]
    correction_available: bool
    correction_note: str
    jacobian_domain_passed: bool
    jacobian_domain_note: str
    robot_state_age_s: Optional[float]
    current_robot_xy_mm: Optional[tuple[float, float]]
    current_robot_delta_xy_mm: Optional[tuple[float, float]]
    visible_marker_count: int
    used_marker_count: int
    reprojection_rms_px: Optional[float]
    annotated_bgr: np.ndarray
    measurement_id: Optional[str] = None
    frame_captured_monotonic_s: Optional[float] = None
    current_joints: Optional[tuple[float, float, float, float]] = None
    current_pose: Optional[tuple[float, float, float, float, float, float]] = None
    tray_transform_C_T: Optional[
        tuple[tuple[float, float, float, float], ...]
    ] = None
    suction_point_T_mm: Optional[tuple[float, float, float]] = None
    target_point_T_mm: Optional[tuple[float, float, float]] = None
    metric_error_T_mm: Optional[tuple[float, float, float]] = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _finite_vector(value: Any, length: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != length or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain {length} finite values")
    return array


def _angular_difference_deg(a_deg: float, b_deg: float) -> float:
    """Return the smallest absolute circular difference in degrees."""

    return abs((float(a_deg) - float(b_deg) + 180.0) % 360.0 - 180.0)


def _candidate_suction_files(project_root: Path) -> list[Path]:
    project_root = Path(project_root)
    candidates: list[Path] = []
    approved = project_root / "src/scara/calib/camera1_suction_target.json"
    if approved.is_file():
        candidates.append(approved)
    run_root = project_root / "Trajectory Photos"
    if run_root.is_dir():
        candidates.extend(
            sorted(
                run_root.glob("*/camera1_suction_target.json"),
                key=lambda path: (path.parent.name, path.stat().st_mtime),
                reverse=True,
            )
        )
    return candidates


def load_latest_suction_target(project_root: Path) -> SuctionTargetModel:
    """Load the newest successful Task-8 result matching current inputs."""

    project_root = Path(project_root)
    intrinsics_path = project_root / "src/scara/calib/camera1_intrinsics.json"
    geometry_path = project_root / "src/scara/calib/tray_board_geometry.json"
    current_intrinsics_hash = sha256_file(intrinsics_path)
    current_geometry_hash = sha256_file(geometry_path)
    failures: list[str] = []
    for path in _candidate_suction_files(project_root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if payload.get("status") != "success":
                raise ValueError("status is not success")
            fit = payload.get("fit") or {}
            if fit.get("status") != "success":
                raise ValueError("fit status is not success")
            locked = payload.get("locked_inputs") or {}
            if str(locked.get("camera_intrinsics_sha256", "")).upper() != (
                current_intrinsics_hash
            ):
                raise ValueError("camera intrinsics hash does not match current file")
            if str(locked.get("tray_geometry_sha256", "")).upper() != (
                current_geometry_hash
            ):
                raise ValueError("Tray geometry hash does not match current file")
            camera = payload.get("camera") or {}
            resolution = camera.get("resolution") or {}
            width = int(resolution.get("width"))
            height = int(resolution.get("height"))
            point = _finite_vector(fit.get("p_C_S_mm"), 3, "p_C_S_mm")
            pixel = _finite_vector(
                fit.get("target_pixel_distorted_px"), 2, "target pixel"
            )
            coordinate = payload.get("coordinate_definition") or {}
            if not 0.0 <= pixel[0] < width or not 0.0 <= pixel[1] < height:
                raise ValueError("target pixel is outside calibrated resolution")
            return SuctionTargetModel(
                source_path=path,
                source_sha256=sha256_file(path),
                camera_source=int(camera.get("source_index", 1)),
                resolution=(width, height),
                p_C_S_mm=(float(point[0]), float(point[1]), float(point[2])),
                target_pixel_px=(float(pixel[0]), float(pixel[1])),
                working_plane_z_T_mm=float(coordinate["working_plane_z_T_mm"]),
                imaging_j3_mm=float(coordinate["imaging_j3_mm"]),
                rz_mean_deg=float(coordinate["rz_mean_deg"]),
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{path}: {exc}")
    detail = "\n".join(failures[-3:]) if failures else "no Task-8 result found"
    raise RuntimeError(f"No usable camera1 suction target calibration:\n{detail}")


def load_local_xy_jacobian(
    project_root: Path,
    suction: Optional[SuctionTargetModel] = None,
    target_name: str = "P22",
) -> Optional[dict[str, Any]]:
    """Load Stage-5 only when all locked upstream calibrations still match."""
    from scara.vision.local_jacobian_registry import local_jacobian_path

    try:
        path = local_jacobian_path(project_root, target_name)
    except ValueError:
        return None
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    try:
        schema_version = int(payload.get("schema_version"))
    except (TypeError, ValueError, OverflowError):
        return None
    if schema_version < 2 or payload.get("status") != "success":
        return None
    fit = payload.get("fit") or payload
    if fit.get("status") != "success":
        return None
    gates = fit.get("quality_gates")
    if not isinstance(gates, Mapping):
        return None
    if not REQUIRED_XY_JACOBIAN_QUALITY_GATES.issubset(gates):
        return None
    if any(
        not isinstance(gates.get(name), Mapping)
        or not bool(gates[name].get("passed"))
        for name in REQUIRED_XY_JACOBIAN_QUALITY_GATES
    ):
        return None
    locked = payload.get("locked_inputs") or {}
    try:
        intrinsics_hash = sha256_file(
            Path(project_root) / "src/scara/calib/camera1_intrinsics.json"
        )
        geometry_hash = sha256_file(
            Path(project_root) / "src/scara/calib/tray_board_geometry.json"
        )
    except OSError:
        return None
    if str(locked.get("camera_intrinsics_sha256", "")).upper() != intrinsics_hash:
        return None
    if str(locked.get("tray_geometry_sha256", "")).upper() != geometry_hash:
        return None
    if suction is not None and str(
        locked.get("suction_target_sha256", "")
    ).upper() != suction.source_sha256:
        return None

    # A local image Jacobian is valid only at the acquisition condition that
    # produced it.  Fail closed when the file is copied from another camera,
    # height, wrist angle, error convention, or an unsupported motion range.
    camera = payload.get("camera") or {}
    coordinate = payload.get("coordinate_definition") or {}
    valid_targets = payload.get("valid_target_names")
    anchor_target = payload.get("anchor_target_name")
    if (
        not isinstance(valid_targets, list)
        or anchor_target not in valid_targets
        or str(target_name) != str(anchor_target)
    ):
        return None
    if coordinate.get("command_frame") != "robot_controller_world_XY":
        return None
    if coordinate.get("image_error") != (
        "slot_pixel_distorted - suction_target_pixel_distorted"
    ):
        return None
    try:
        source_index = int(camera["source_index"])
        resolution = camera["resolution"]
        width = int(resolution["width"])
        height = int(resolution["height"])
        imaging_j3_mm = float(coordinate["imaging_j3_mm"])
        rz_deg = float(coordinate["rz_deg"])
        offset_extent_mm = float(coordinate["offset_extent_mm"])
        anchor_robot_xy_mm = _finite_vector(
            coordinate["anchor_robot_xy_mm"], 2, "anchor_robot_xy_mm"
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not all(
        math.isfinite(value)
        for value in (
            imaging_j3_mm,
            rz_deg,
            offset_extent_mm,
            *anchor_robot_xy_mm,
        )
    ):
        return None
    if source_index != 1 or (width, height) != (1280, 720):
        return None
    if offset_extent_mm <= 0.0 or offset_extent_mm > 2.0 + 1e-9:
        return None
    if suction is not None:
        if source_index != suction.camera_source:
            return None
        if (width, height) != suction.resolution:
            return None
        if abs(imaging_j3_mm - suction.imaging_j3_mm) > 0.20:
            return None
        if _angular_difference_deg(rz_deg, suction.rz_mean_deg) > 0.50:
            return None
    return payload


ROBOT_STATE_MAXIMUM_AGE_S = 1.0
JACOBIAN_DOMAIN_DECIMAL_TOLERANCE_MM = 1e-3
JACOBIAN_DOMAIN_J3_TOLERANCE_MM = 0.20
JACOBIAN_DOMAIN_RZ_TOLERANCE_DEG = 0.20


def _jacobian_correction_in_current_domain(
    image_error_px: tuple[float, float],
    target_name: str,
    jacobian_payload: Mapping[str, Any],
    robot_state: Optional[Mapping[str, Any]],
    *,
    maximum_state_age_s: float,
) -> tuple[
    Optional[tuple[float, float]],
    bool,
    str,
    Optional[float],
    Optional[tuple[float, float]],
    Optional[tuple[float, float]],
]:
    """Return a correction only inside the selected slot's Task-9 domain.

    This is a read-only gate.  ``robot_state`` is a cached UI status snapshot;
    no controller object or hardware method is accepted by this module.
    """

    fit = jacobian_payload.get("fit") or jacobian_payload
    coordinate = jacobian_payload.get("coordinate_definition") or {}
    anchor_target = str(jacobian_payload.get("anchor_target_name") or "")
    valid_targets = jacobian_payload.get("valid_target_names") or []
    if anchor_target != target_name or target_name not in valid_targets:
        note = (
            "阶段5局部Jacobian只验证于P22；其他槽只显示视觉像素误差"
            if anchor_target == "P22"
            else f"没有加载与{target_name}匹配的局部Jacobian；只显示视觉像素误差"
        )
        return (
            None,
            False,
            note,
            None,
            None,
            None,
        )
    if robot_state is None:
        return None, False, "缺少最新机械臂只读状态", None, None, None

    try:
        captured_at = float(robot_state["captured_monotonic_s"])
        joints = _finite_vector(robot_state["joints"], 4, "robot joints")
        pose = _finite_vector(robot_state["pose"], 6, "robot pose")
        anchor_xy = _finite_vector(
            coordinate["anchor_robot_xy_mm"], 2, "anchor_robot_xy_mm"
        )
        extent = float(coordinate["offset_extent_mm"])
        expected_j3 = float(coordinate["imaging_j3_mm"])
        expected_rz = float(coordinate["rz_deg"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, False, "机械臂状态或Jacobian适用域字段无效", None, None, None
    if not all(
        math.isfinite(value)
        for value in (captured_at, extent, expected_j3, expected_rz)
    ) or extent <= 0.0:
        return None, False, "机械臂状态或Jacobian适用域字段无效", None, None, None

    age = time.monotonic() - captured_at
    current_xy_array = pose[:2]
    delta_array = current_xy_array - anchor_xy
    current_xy = (float(current_xy_array[0]), float(current_xy_array[1]))
    delta_xy = (float(delta_array[0]), float(delta_array[1]))
    if age < -0.05 or age > float(maximum_state_age_s):
        return (
            None,
            False,
            f"机械臂只读状态已过期（age={age:.2f}s）",
            float(age),
            current_xy,
            delta_xy,
        )

    extent_with_tolerance = extent + JACOBIAN_DOMAIN_DECIMAL_TOLERANCE_MM
    if np.any(np.abs(delta_array) > extent_with_tolerance):
        return (
            None,
            False,
            (
                "当前world XY超出Task9局部域："
                f"Δ=({delta_xy[0]:+.3f},{delta_xy[1]:+.3f})mm，"
                f"每轴限制±{extent:.3f}mm"
            ),
            float(age),
            current_xy,
            delta_xy,
        )
    j3_drift = abs(float(joints[2]) - expected_j3)
    if j3_drift > JACOBIAN_DOMAIN_J3_TOLERANCE_MM:
        return (
            None,
            False,
            f"当前J3偏离Task9高度{j3_drift:.3f}mm（限制0.20mm）",
            float(age),
            current_xy,
            delta_xy,
        )
    current_rz = rz_of(float(joints[0]), float(joints[1]), float(joints[3]))
    rz_drift = _angular_difference_deg(current_rz, expected_rz)
    if rz_drift > JACOBIAN_DOMAIN_RZ_TOLERANCE_DEG:
        return (
            None,
            False,
            f"当前Rz偏离Task9姿态{rz_drift:.3f}°（限制0.20°）",
            float(age),
            current_xy,
            delta_xy,
        )

    correction = correction_command_xy_mm(image_error_px, fit)
    if correction is None:
        return (
            None,
            False,
            "阶段5Jacobian不可逆或未通过质量门",
            float(age),
            current_xy,
            delta_xy,
        )
    correction_array = np.asarray(correction, dtype=np.float64)
    if np.any(np.abs(correction_array) > extent_with_tolerance):
        return (
            None,
            False,
            (
                "候选XY修正超出Task9单步局部域："
                f"({correction[0]:+.3f},{correction[1]:+.3f})mm，"
                f"每轴限制±{extent:.3f}mm"
            ),
            float(age),
            current_xy,
            delta_xy,
        )
    predicted_delta = delta_array + correction_array
    if np.any(np.abs(predicted_delta) > extent_with_tolerance):
        return (
            None,
            False,
            (
                "候选修正的预测终点越出Task9标定域："
                f"Δnext=({predicted_delta[0]:+.3f},"
                f"{predicted_delta[1]:+.3f})mm"
            ),
            float(age),
            current_xy,
            delta_xy,
        )
    return (
        correction,
        True,
        (
            "当前机械臂状态与候选修正均位于Task9 "
            f"{target_name}局部标定域；仅计算，未下发"
        ),
        float(age),
        current_xy,
        delta_xy,
    )


def project_tray_points_from_transform(
    points_T_mm: Any,
    transform_C_T: Any,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    points = np.asarray(points_T_mm, dtype=np.float64).reshape(-1, 3)
    transform = np.asarray(transform_C_T, dtype=np.float64).reshape(4, 4)
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    projected, _ = cv2.projectPoints(
        points,
        rvec,
        transform[:3, 3].reshape(3, 1),
        intrinsics.K,
        intrinsics.dist_coeffs,
    )
    return projected.reshape(-1, 2)


def _draw_cross(
    image: np.ndarray,
    point: tuple[float, float],
    color: tuple[int, int, int],
    size: int = 16,
    thickness: int = 3,
) -> None:
    x, y = int(round(point[0])), int(round(point[1]))
    cv2.line(image, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)
    cv2.circle(image, (x, y), 4, color, thickness, cv2.LINE_AA)


def _draw_reprojected_markers_and_axes(
    image: np.ndarray,
    transform_C_T: np.ndarray,
    intrinsics: CameraIntrinsics,
    geometry: Mapping[str, Any],
) -> None:
    height, width = image.shape[:2]
    for label, marker in geometry["markers"].items():
        pixels = project_tray_points_from_transform(
            marker["corners_T_mm"], transform_C_T, intrinsics
        )
        for pixel in pixels:
            x, y = int(round(pixel[0])), int(round(pixel[1]))
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(image, (x, y), 4, (255, 255, 0), 2, cv2.LINE_AA)
        centre = np.mean(pixels, axis=0)
        x, y = int(round(centre[0])), int(round(centre[1]))
        if 0 <= x < width and 0 <= y < height:
            cv2.putText(
                image,
                f"T-{label}",
                (x + 7, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

    axis_length = 35.0
    axes = project_tray_points_from_transform(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        transform_C_T,
        intrinsics,
    )
    origin = tuple(np.round(axes[0]).astype(int))
    for endpoint, color, label in zip(
        axes[1:],
        ((0, 0, 255), (0, 255, 0), (255, 0, 0)),
        ("T-X", "T-Y", "T-Z"),
    ):
        end = tuple(np.round(endpoint).astype(int))
        cv2.arrowedLine(image, origin, end, color, 3, cv2.LINE_AA, tipLength=0.18)
        cv2.putText(
            image,
            label,
            (end[0] + 5, end[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def evaluate_handeye_frame(
    frame_bgr: np.ndarray,
    tracked: TrackedTrayPose,
    target_name: str,
    geometry: Mapping[str, Any],
    intrinsics: CameraIntrinsics,
    suction: SuctionTargetModel,
    jacobian_payload: Optional[Mapping[str, Any]] = None,
    robot_state: Optional[Mapping[str, Any]] = None,
    *,
    alignment_threshold_px: float = 3.0,
    maximum_robot_state_age_s: float = ROBOT_STATE_MAXIMUM_AGE_S,
) -> HandEyeEvaluation:
    """Evaluate one tracked frame and draw the requested read-only overlay."""

    alignment_threshold_px = float(alignment_threshold_px)
    maximum_robot_state_age_s = float(maximum_robot_state_age_s)
    if not math.isfinite(alignment_threshold_px) or alignment_threshold_px <= 0.0:
        raise ValueError("alignment_threshold_px must be a positive finite value")
    if (
        not math.isfinite(maximum_robot_state_age_s)
        or maximum_robot_state_age_s <= 0.0
    ):
        raise ValueError("maximum_robot_state_age_s must be positive and finite")

    raw = tracked.raw
    annotated = (
        raw.annotated_image.copy()
        if raw.annotated_image is not None
        else np.asarray(frame_bgr).copy()
    )
    target_pixel = suction.target_pixel_px
    _draw_cross(annotated, target_pixel, (0, 0, 255))
    cv2.putText(
        annotated,
        "SUCTION TARGET",
        (int(target_pixel[0]) + 18, int(target_pixel[1]) - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    accepted = bool(
        tracked.accepted_by_tracker and tracked.filtered_T_C_T is not None
    )
    if not accepted:
        reason = tracked.tracker_reason or raw.failure_reason or "pose rejected"
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 64), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            "COMPUTE ONLY - NO ROBOT MOTION | CURRENT FRAME REJECTED",
            (18, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        return HandEyeEvaluation(
            target_name=target_name,
            accepted=False,
            reason=reason,
            slot_pixel_px=None,
            suction_target_pixel_px=target_pixel,
            image_error_px=None,
            image_error_norm_px=None,
            alignment_threshold_px=alignment_threshold_px,
            aligned=None,
            correction_xy_mm=None,
            correction_available=False,
            correction_note="当前帧未通过Stage3/时序质量门",
            jacobian_domain_passed=False,
            jacobian_domain_note="Stage3无合格位姿，未评估Jacobian适用域",
            robot_state_age_s=None,
            current_robot_xy_mm=None,
            current_robot_delta_xy_mm=None,
            visible_marker_count=len(raw.visible_marker_ids),
            used_marker_count=len(raw.used_marker_ids),
            reprojection_rms_px=raw.reprojection_rms_px,
            annotated_bgr=annotated,
        )

    slots = geometry.get("slots") or {}
    if target_name not in slots:
        raise KeyError(f"unknown Tray slot {target_name}")
    transform = np.asarray(tracked.filtered_T_C_T, dtype=np.float64)
    metric = metric_suction_error_in_tray(
        transform,
        suction.p_C_S_mm,
        slots[target_name],
    )
    slot_pixel_array = project_tray_points_from_transform(
        [slots[target_name]], transform, intrinsics
    )[0]
    slot_pixel = (float(slot_pixel_array[0]), float(slot_pixel_array[1]))
    error = (
        slot_pixel[0] - target_pixel[0],
        slot_pixel[1] - target_pixel[1],
    )
    error_norm = float(math.hypot(error[0], error[1]))
    aligned = error_norm <= alignment_threshold_px

    _draw_reprojected_markers_and_axes(
        annotated, transform, intrinsics, geometry
    )
    _draw_cross(annotated, slot_pixel, (0, 255, 0))
    cv2.putText(
        annotated,
        f"SLOT {target_name}",
        (int(slot_pixel[0]) + 18, int(slot_pixel[1]) + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.arrowedLine(
        annotated,
        tuple(np.round(target_pixel).astype(int)),
        tuple(np.round(slot_pixel).astype(int)),
        (0, 255, 255),
        3,
        cv2.LINE_AA,
        tipLength=0.12,
    )
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 76), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        (
            "COMPUTE ONLY - NO ROBOT MOTION | "
            f"{target_name} e=({error[0]:+.1f},{error[1]:+.1f})px"
        ),
        (18, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        (
            f"VISUAL |e|<={alignment_threshold_px:g}px ONLY - "
            f"NOT PLACEMENT ACCEPTANCE | "
            f"{'WITHIN' if aligned else 'OUTSIDE'}"
        ),
        (18, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 0) if aligned else (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    correction: Optional[tuple[float, float]] = None
    correction_available = False
    correction_note = "阶段5局部雅可比尚未标定"
    jacobian_domain_passed = False
    jacobian_domain_note = correction_note
    robot_state_age_s: Optional[float] = None
    current_robot_xy_mm: Optional[tuple[float, float]] = None
    current_robot_delta_xy_mm: Optional[tuple[float, float]] = None
    if jacobian_payload is not None:
        (
            correction,
            jacobian_domain_passed,
            jacobian_domain_note,
            robot_state_age_s,
            current_robot_xy_mm,
            current_robot_delta_xy_mm,
        ) = _jacobian_correction_in_current_domain(
            error,
            target_name,
            jacobian_payload,
            robot_state,
            maximum_state_age_s=maximum_robot_state_age_s,
        )
        correction_available = correction is not None and jacobian_domain_passed
        correction_note = jacobian_domain_note

    return HandEyeEvaluation(
        target_name=target_name,
        accepted=True,
        reason="ok",
        slot_pixel_px=slot_pixel,
        suction_target_pixel_px=target_pixel,
        image_error_px=error,
        image_error_norm_px=error_norm,
        alignment_threshold_px=alignment_threshold_px,
        aligned=aligned,
        correction_xy_mm=correction,
        correction_available=correction_available,
        correction_note=correction_note,
        jacobian_domain_passed=jacobian_domain_passed,
        jacobian_domain_note=jacobian_domain_note,
        robot_state_age_s=robot_state_age_s,
        current_robot_xy_mm=current_robot_xy_mm,
        current_robot_delta_xy_mm=current_robot_delta_xy_mm,
        visible_marker_count=len(raw.visible_marker_ids),
        used_marker_count=len(raw.used_marker_ids),
        reprojection_rms_px=raw.reprojection_rms_px,
        annotated_bgr=annotated,
        tray_transform_C_T=tuple(
            tuple(float(value) for value in row) for row in transform
        ),
        suction_point_T_mm=tuple(
            float(value) for value in metric["suction_point_T_mm"]
        ),
        target_point_T_mm=tuple(
            float(value) for value in metric["target_point_T_mm"]
        ),
        metric_error_T_mm=tuple(
            float(value) for value in metric["metric_error_T_mm"]
        ),
    )


__all__ = [
    "HandEyeEvaluation",
    "ROBOT_STATE_MAXIMUM_AGE_S",
    "SuctionTargetModel",
    "evaluate_handeye_frame",
    "load_latest_suction_target",
    "load_local_xy_jacobian",
    "project_tray_points_from_transform",
    "sha256_file",
]
