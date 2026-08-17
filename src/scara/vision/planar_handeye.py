"""Fixed-height planar hand-eye calibration for camera 1 on the SCARA forearm.

This module deliberately solves only the observable planar problem.  Frame
``F`` is centred at the J4 axis and has yaw ``J1 + J2``.  For every accepted
Stage-3 observation we know ``^C T_T`` and the Stage-4 suction/J4-axis point
``^C p_S``.  Therefore the suction point expressed in Tray coordinates is::

    ^T p_S = inv(^C T_T) ^C p_S

The controller readback supplies the same J4-axis point in world XY.  A robust
2-D rigid fit over spatially separated poses yields one ``^W T_T`` per source
run.  With that run transform, the fixed camera rotation follows from::

    ^F R_C = (^W R_F)^T ^W R_T (^C R_T)^T

Repeated images at one slot are aggregated before fitting and never counted as
independent robot poses.  Validation holds out complete slots.  No Z extrinsic
or full 6-DoF hand-eye claim is made.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from scara.file_io import atomic_write_text


DEFAULT_EXCLUDED_SLOTS = frozenset({"P00", "P05", "P25", "P45"})
DEFAULT_STRESS_SLOTS = frozenset({"P01", "P10"})
DEFAULT_TASK12_HOLDOUT = frozenset(
    {"P02", "P04", "P15", "P24", "P35", "P40", "P53", "P55"}
)
DEFAULT_TASK8_HOLDOUT = frozenset({"P22", "P52"})
DEFAULT_TASK13_HOLDOUT = frozenset({"P04", "P15", "P35", "P53"})


@dataclass(frozen=True)
class PlanarHandEyeQualityConfig:
    minimum_frames_per_pose: int = 12
    minimum_task13_frames_per_pose: int = 8
    minimum_independent_training_poses: int = 12
    minimum_world_xy_span_mm: float = 100.0
    minimum_forearm_alpha_span_deg: float = 30.0
    maximum_holdout_xy_rms_mm: float = 0.50
    maximum_holdout_xy_p95_mm: float = 0.80
    maximum_holdout_xy_mm: float = 1.20
    maximum_holdout_yaw_rms_deg: float = 0.20
    maximum_holdout_yaw_deg: float = 0.50
    maximum_cross_run_rotation_difference_deg: float = 1.00


DEFAULT_QUALITY = PlanarHandEyeQualityConfig()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"找不到{label}：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}不是有效JSON：{path}（{exc}）") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label}顶层必须是JSON对象")
    return payload


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


def _rotation_z(degrees: float) -> np.ndarray:
    angle = math.radians(float(degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _yaw_deg(rotation: np.ndarray) -> float:
    matrix = _finite_matrix(rotation, (3, 3), "rotation")
    return math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))


def _angle_delta_deg(left: float, right: float) -> float:
    return (float(left) - float(right) + 180.0) % 360.0 - 180.0


def _angle_magnitude_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _average_rotation(rotations: Sequence[np.ndarray]) -> np.ndarray:
    if not rotations:
        raise ValueError("至少需要一个旋转矩阵")
    total = np.zeros((3, 3), dtype=np.float64)
    for rotation in rotations:
        total += _finite_matrix(rotation, (3, 3), "rotation")
    u, _singular, vt = np.linalg.svd(total)
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def _circular_span_deg(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    wrapped = np.mod(np.asarray(values, dtype=np.float64), 360.0)
    if wrapped.size <= 1:
        return 0.0
    ordered = np.sort(wrapped)
    gaps = np.diff(np.concatenate((ordered, ordered[:1] + 360.0)))
    return float(360.0 - np.max(gaps))


def _robust_keep_mask(points: np.ndarray, floor_mm: float = 0.50) -> np.ndarray:
    centre = np.median(points, axis=0)
    distance = np.linalg.norm(points - centre, axis=1)
    median = float(np.median(distance))
    mad = float(np.median(np.abs(distance - median)))
    threshold = max(float(floor_mm), median + 3.0 * 1.4826 * mad)
    return distance <= threshold + 1e-12


def _fit_rigid_xy(source_xy: np.ndarray, target_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_xy, dtype=np.float64)
    target = np.asarray(target_xy, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2 or target.shape != source.shape:
        raise ValueError("二维刚体拟合要求Nx2对应点")
    if source.shape[0] < 2:
        raise ValueError("二维刚体拟合至少需要两个点")
    source_centre = np.mean(source, axis=0)
    target_centre = np.mean(target, axis=0)
    covariance = (source - source_centre).T @ (target - target_centre)
    u, _singular, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_centre - rotation @ source_centre
    return rotation, translation


def _slot_name(point: Mapping[str, Any]) -> str | None:
    for key in ("task8_target_name", "task12_target_name", "target_name"):
        value = point.get(key)
        if isinstance(value, str) and len(value) == 3 and value.startswith("P"):
            return value
    name = str(point.get("name") or "")
    for token in name.replace("|", " ").split():
        if len(token) == 3 and token.startswith("P") and token[1:].isdigit():
            return token
    return None


def _accepted_transform(point: Mapping[str, Any]) -> np.ndarray | None:
    pose = point.get("stage3_pose")
    temporal = point.get("stage3_temporal_quality")
    if not isinstance(pose, Mapping) or pose.get("quality_passed") is not True:
        return None
    if not isinstance(temporal, Mapping) or temporal.get("accepted_by_tracker") is not True:
        return None
    value = temporal.get("filtered_T_C_T") or pose.get("T_C_T")
    try:
        return _finite_matrix(value, (4, 4), "Stage3 T_C_T")
    except ValueError:
        return None


def _extract_frame_rows(points_path: Path, suction_point_C_mm: Sequence[float]) -> list[dict[str, Any]]:
    manifest = _load_json(points_path, "运行points.json")
    points = manifest.get("points")
    if not isinstance(points, list):
        raise ValueError(f"{points_path}缺少points数组")
    suction_C = _finite_vector(suction_point_C_mm, 3, "Stage4 suction point")
    rows: list[dict[str, Any]] = []
    for point in points:
        if not isinstance(point, Mapping):
            continue
        slot = _slot_name(point)
        transform_C_T = _accepted_transform(point)
        joints = point.get("joints")
        mechanical = point.get("mechanical_center")
        if slot is None or transform_C_T is None or not isinstance(joints, Mapping) or not isinstance(mechanical, Mapping):
            continue
        try:
            joint_values = np.asarray(
                [joints["J1_deg"], joints["J2_deg"], joints["J3_mm"], joints["J4_deg"]],
                dtype=np.float64,
            )
            world_xy = np.asarray([mechanical["x_mm"], mechanical["y_mm"]], dtype=np.float64)
            if not np.all(np.isfinite(joint_values)) or not np.all(np.isfinite(world_xy)):
                continue
            suction_T_h = np.linalg.inv(transform_C_T) @ np.concatenate((suction_C, [1.0]))
            suction_T = suction_T_h[:3] / suction_T_h[3]
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            continue
        rows.append(
            {
                "slot": slot,
                "sequence": int(point.get("sequence") or 0),
                "suction_T_mm": suction_T,
                "world_xy_mm": world_xy,
                "joints": joint_values,
                "rotation_C_T": transform_C_T[:3, :3].copy(),
            }
        )
    return rows


def _aggregate_run(
    run_name: str,
    points_path: Path,
    suction_point_C_mm: Sequence[float],
    quality: PlanarHandEyeQualityConfig,
) -> list[dict[str, Any]]:
    frame_rows = _extract_frame_rows(points_path, suction_point_C_mm)
    minimum_frames = (
        quality.minimum_task13_frames_per_pose
        if "task13" in run_name.lower()
        else quality.minimum_frames_per_pose
    )
    aggregates: list[dict[str, Any]] = []
    for slot in sorted({row["slot"] for row in frame_rows}):
        group = [row for row in frame_rows if row["slot"] == slot]
        suction_points = np.asarray([row["suction_T_mm"][:2] for row in group])
        world_points = np.asarray([row["world_xy_mm"] for row in group])
        keep = _robust_keep_mask(suction_points) & _robust_keep_mask(world_points, 0.10)
        kept = [row for row, accepted in zip(group, keep) if bool(accepted)]
        if len(kept) < minimum_frames:
            continue
        aggregates.append(
            {
                "run_name": str(run_name),
                "slot": slot,
                "accepted_frame_count": len(kept),
                "source_frame_count": len(group),
                "suction_T_xy_mm": np.median(
                    np.asarray([row["suction_T_mm"][:2] for row in kept]), axis=0
                ),
                "world_xy_mm": np.median(
                    np.asarray([row["world_xy_mm"] for row in kept]), axis=0
                ),
                "joints": np.median(
                    np.asarray([row["joints"] for row in kept]), axis=0
                ),
                "rotation_C_T": _average_rotation(
                    [row["rotation_C_T"] for row in kept]
                ),
            }
        )
    return aggregates


def _select_holdout(run_name: str, available: set[str]) -> set[str]:
    lower_name = run_name.lower()
    if "task8" in lower_name:
        preferred = DEFAULT_TASK8_HOLDOUT
    elif "task13" in lower_name:
        preferred = DEFAULT_TASK13_HOLDOUT
    else:
        preferred = DEFAULT_TASK12_HOLDOUT
    selected = set(preferred) & available
    if selected:
        return selected
    ordered = sorted(available)
    return {name for index, name in enumerate(ordered) if index % 4 == 3}


def _gate(passed: bool, actual: Any, limit: str) -> dict[str, Any]:
    return {"passed": bool(passed), "actual": actual, "limit": str(limit)}


def _metric(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "rms": None, "p95": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "rms": float(math.sqrt(float(np.mean(array * array)))),
        "p95": float(np.percentile(array, 95.0)),
        "maximum": float(np.max(array)),
    }


def _max_pairwise_span(points: Sequence[Sequence[float]]) -> float:
    array = np.asarray(points, dtype=np.float64)
    if array.shape[0] < 2:
        return 0.0
    delta = array[:, None, :] - array[None, :, :]
    return float(np.max(np.linalg.norm(delta, axis=2)))


def _independent_pose_count(
    poses: Sequence[Mapping[str, Any]],
    *,
    xy_separation_mm: float = 0.50,
    alpha_separation_deg: float = 0.10,
) -> int:
    """Count robot configurations without double-counting repeated runs.

    A pose is new when either its world XY or forearm yaw differs materially
    from every existing cluster representative.  This prevents Task12 and
    Task13 observations at the same robot configuration from artificially
    doubling the excitation count.
    """

    representatives: list[tuple[np.ndarray, float]] = []
    for pose in poses:
        xy = np.asarray(pose["world_xy_mm"], dtype=np.float64)
        joints = np.asarray(pose["joints"], dtype=np.float64)
        alpha = float(joints[0] + joints[1])
        if any(
            float(np.linalg.norm(xy - existing_xy)) <= xy_separation_mm
            and abs(_angle_delta_deg(alpha, existing_alpha)) <= alpha_separation_deg
            for existing_xy, existing_alpha in representatives
        ):
            continue
        representatives.append((xy, alpha))
    return len(representatives)


def _choose_probe_poses(training: Sequence[Mapping[str, Any]], count: int = 3) -> list[dict[str, Any]]:
    candidates = list(training)
    if not candidates:
        return []
    selected = [max(candidates, key=lambda item: float(item["world_xy_mm"][0] + item["world_xy_mm"][1]))]
    while len(selected) < min(int(count), len(candidates)):
        def score(item: Mapping[str, Any]) -> float:
            xy = np.asarray(item["world_xy_mm"], dtype=np.float64)
            alpha = float(item["joints"][0] + item["joints"][1])
            return min(
                float(np.linalg.norm(xy - np.asarray(existing["world_xy_mm"])))
                + 0.5 * abs(_angle_delta_deg(alpha, float(existing["joints"][0] + existing["joints"][1])))
                for existing in selected
            )
        remaining = [item for item in candidates if item not in selected]
        selected.append(max(remaining, key=score))
    return [
        {
            "run_name": str(item["run_name"]),
            "slot": str(item["slot"]),
            # Runtime probe locations must move with the Tray.  The historical
            # world coordinate is audit evidence only; routing uses this Tray
            # coordinate together with the session's W<-T registration.
            "tray_xy_mm": np.asarray(item["suction_T_xy_mm"], dtype=float).tolist(),
            "world_xy_mm": np.asarray(item["world_xy_mm"], dtype=float).tolist(),
            "joints": np.asarray(item["joints"], dtype=float).tolist(),
            "forearm_alpha_deg": float(item["joints"][0] + item["joints"][1]),
        }
        for item in selected
    ]


def fit_planar_handeye(
    project_root: Path,
    source_runs: Sequence[tuple[str, Path]],
    *,
    suction_target_path: Path,
    quality: PlanarHandEyeQualityConfig = DEFAULT_QUALITY,
) -> dict[str, Any]:
    """Fit and independently validate one fixed-height camera-1 planar model."""

    root = Path(project_root)
    suction_path = Path(suction_target_path)
    suction = _load_json(suction_path, "Stage4 suction target")
    if suction.get("status") != "success":
        raise ValueError("Stage4 suction target不是success")
    fit = suction.get("fit") or {}
    suction_point_C = _finite_vector(
        fit.get("p_C_S_mm", fit.get("suction_point_C_mm")),
        3,
        "Stage4 p_C_S_mm",
    )
    coordinate = suction.get("coordinate_definition") or {}
    required_j3 = float(coordinate.get("imaging_j3_mm"))
    if not math.isfinite(required_j3):
        raise ValueError("Stage4缺少固定观察J3")
    camera = suction.get("camera") or {}
    source_index = int(camera.get("source_index", -1))
    resolution = camera.get("resolution") or {}
    image_resolution = {
        "width": int(resolution.get("width", 0)),
        "height": int(resolution.get("height", 0)),
    }
    locked = suction.get("locked_inputs") or {}
    intrinsics_path = root / "src/scara/calib/camera1_intrinsics.json"
    geometry_path = root / "src/scara/calib/tray_board_geometry.json"
    current_intrinsics_hash = _sha256(intrinsics_path)
    current_geometry_hash = _sha256(geometry_path)
    if str(locked.get("camera_intrinsics_sha256") or "").upper() != current_intrinsics_hash:
        raise ValueError("Stage4锁定的相机内参hash与当前项目文件不一致")
    if str(locked.get("tray_geometry_sha256") or "").upper() != current_geometry_hash:
        raise ValueError("Stage4锁定的Tray geometry hash与当前项目文件不一致")

    run_aggregates: dict[str, list[dict[str, Any]]] = {}
    source_records: list[dict[str, Any]] = []
    for run_name, points_path in source_runs:
        path = Path(points_path)
        aggregates = _aggregate_run(run_name, path, suction_point_C, quality)
        run_aggregates[str(run_name)] = aggregates
        source_records.append(
            {
                "run_name": str(run_name),
                "points_path": str(path.resolve()),
                "points_sha256": _sha256(path),
                "aggregate_pose_count": len(aggregates),
            }
        )

    run_models: dict[str, dict[str, Any]] = {}
    all_training: list[dict[str, Any]] = []
    all_holdout: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []
    for run_name, aggregates in run_aggregates.items():
        normal = [
            item for item in aggregates
            if item["slot"] not in DEFAULT_EXCLUDED_SLOTS
            and item["slot"] not in DEFAULT_STRESS_SLOTS
        ]
        available = {str(item["slot"]) for item in normal}
        holdout_names = _select_holdout(run_name, available)
        training = [item for item in normal if item["slot"] not in holdout_names]
        holdout = [item for item in normal if item["slot"] in holdout_names]
        if len(training) < 2:
            raise ValueError(f"{run_name}没有足够训练槽位拟合独立Tray到World登记")
        rotation_2d, translation = _fit_rigid_xy(
            np.asarray([item["suction_T_xy_mm"] for item in training]),
            np.asarray([item["world_xy_mm"] for item in training]),
        )
        yaw = math.degrees(math.atan2(float(rotation_2d[1, 0]), float(rotation_2d[0, 0])))
        run_models[run_name] = {
            "rotation_2d": rotation_2d,
            "translation": translation,
            "yaw_world_from_tray_deg": float(yaw),
            "training_slots": sorted(str(item["slot"]) for item in training),
            "holdout_slots": sorted(str(item["slot"]) for item in holdout),
        }
        all_training.extend(training)
        all_holdout.extend(holdout)
        stress_rows.extend(
            item for item in aggregates if item["slot"] in DEFAULT_STRESS_SLOTS
        )

    rfc_by_run: dict[str, np.ndarray] = {}
    rfc_training: list[np.ndarray] = []
    for run_name, model in run_models.items():
        R_W_T = _rotation_z(model["yaw_world_from_tray_deg"])
        candidates: list[np.ndarray] = []
        for item in all_training:
            if item["run_name"] != run_name:
                continue
            joints = item["joints"]
            R_W_F = _rotation_z(float(joints[0] + joints[1]))
            candidate = R_W_F.T @ R_W_T @ np.asarray(item["rotation_C_T"]).T
            candidates.append(candidate)
            rfc_training.append(candidate)
        rfc_by_run[run_name] = _average_rotation(candidates)
    R_F_C = _average_rotation(rfc_training)

    xy_errors: list[float] = []
    yaw_errors: list[float] = []
    validation_rows: list[dict[str, Any]] = []
    for item in all_holdout:
        model = run_models[str(item["run_name"])]
        predicted_xy = model["rotation_2d"] @ item["suction_T_xy_mm"] + model["translation"]
        xy_error = float(np.linalg.norm(predicted_xy - item["world_xy_mm"]))
        joints = item["joints"]
        predicted_R_W_T = _rotation_z(float(joints[0] + joints[1])) @ R_F_C @ item["rotation_C_T"]
        yaw_error = abs(
            _angle_delta_deg(_yaw_deg(predicted_R_W_T), model["yaw_world_from_tray_deg"])
        )
        xy_errors.append(xy_error)
        yaw_errors.append(yaw_error)
        validation_rows.append(
            {
                "run_name": str(item["run_name"]),
                "slot": str(item["slot"]),
                "actual_world_xy_mm": np.asarray(item["world_xy_mm"], dtype=float).tolist(),
                "predicted_world_xy_mm": predicted_xy.astype(float).tolist(),
                "xy_error_mm": xy_error,
                "yaw_error_deg": float(yaw_error),
                "accepted_frame_count": int(item["accepted_frame_count"]),
            }
        )

    stress_validation: list[dict[str, Any]] = []
    for item in stress_rows:
        model = run_models[str(item["run_name"])]
        predicted = model["rotation_2d"] @ item["suction_T_xy_mm"] + model["translation"]
        stress_validation.append(
            {
                "run_name": str(item["run_name"]),
                "slot": str(item["slot"]),
                "xy_error_mm": float(np.linalg.norm(predicted - item["world_xy_mm"])),
                "accepted_frame_count": int(item["accepted_frame_count"]),
            }
        )

    xy_metric = _metric(xy_errors)
    yaw_metric = _metric(yaw_errors)
    cross_run_differences: list[float] = []
    run_names = sorted(rfc_by_run)
    for left_index, left in enumerate(run_names):
        for right in run_names[left_index + 1 :]:
            cross_run_differences.append(
                _angle_magnitude_deg(rfc_by_run[left].T @ rfc_by_run[right])
            )
    cross_run_max = max(cross_run_differences, default=0.0)
    training_world = [item["world_xy_mm"] for item in all_training]
    alphas = [float(item["joints"][0] + item["joints"][1]) for item in all_training]
    world_span = _max_pairwise_span(training_world)
    alpha_span = _circular_span_deg(alphas)
    independent_pose_count = _independent_pose_count(all_training)
    j3_deviations = [
        abs(float(item["joints"][2]) - required_j3) for item in all_training + all_holdout
    ]
    maximum_j3_deviation = max(j3_deviations, default=math.inf)
    gates = {
        "independent_robot_poses": _gate(
            independent_pose_count >= quality.minimum_independent_training_poses,
            independent_pose_count,
            f">={quality.minimum_independent_training_poses}",
        ),
        "world_xy_excitation": _gate(
            world_span >= quality.minimum_world_xy_span_mm,
            world_span,
            f">={quality.minimum_world_xy_span_mm:.2f} mm",
        ),
        "forearm_alpha_excitation": _gate(
            alpha_span >= quality.minimum_forearm_alpha_span_deg,
            alpha_span,
            f">={quality.minimum_forearm_alpha_span_deg:.2f} deg",
        ),
        "fixed_imaging_j3_consistency": _gate(
            maximum_j3_deviation <= 0.05,
            maximum_j3_deviation,
            "<=0.05 mm from Stage4 imaging J3",
        ),
        "holdout_xy_rms": _gate(
            xy_metric["rms"] is not None and float(xy_metric["rms"]) <= quality.maximum_holdout_xy_rms_mm,
            xy_metric["rms"],
            f"<={quality.maximum_holdout_xy_rms_mm:.2f} mm",
        ),
        "holdout_xy_p95": _gate(
            xy_metric["p95"] is not None and float(xy_metric["p95"]) <= quality.maximum_holdout_xy_p95_mm,
            xy_metric["p95"],
            f"<={quality.maximum_holdout_xy_p95_mm:.2f} mm",
        ),
        "holdout_xy_maximum": _gate(
            xy_metric["maximum"] is not None and float(xy_metric["maximum"]) <= quality.maximum_holdout_xy_mm,
            xy_metric["maximum"],
            f"<={quality.maximum_holdout_xy_mm:.2f} mm",
        ),
        "holdout_yaw_rms": _gate(
            yaw_metric["rms"] is not None and float(yaw_metric["rms"]) <= quality.maximum_holdout_yaw_rms_deg,
            yaw_metric["rms"],
            f"<={quality.maximum_holdout_yaw_rms_deg:.2f} deg",
        ),
        "holdout_yaw_maximum": _gate(
            yaw_metric["maximum"] is not None and float(yaw_metric["maximum"]) <= quality.maximum_holdout_yaw_deg,
            yaw_metric["maximum"],
            f"<={quality.maximum_holdout_yaw_deg:.2f} deg",
        ),
        "cross_run_rotation_consistency": _gate(
            len(run_names) >= 2 and cross_run_max <= quality.maximum_cross_run_rotation_difference_deg,
            cross_run_max if len(run_names) >= 2 else None,
            f"<={quality.maximum_cross_run_rotation_difference_deg:.2f} deg with >=2 runs",
        ),
    }
    status = "success" if all(gate["passed"] for gate in gates.values()) else "failure"
    run_payload = {
        name: {
            "transform_W_T_planar": [
                [float(model["rotation_2d"][0, 0]), float(model["rotation_2d"][0, 1]), float(model["translation"][0])],
                [float(model["rotation_2d"][1, 0]), float(model["rotation_2d"][1, 1]), float(model["translation"][1])],
                [0.0, 0.0, 1.0],
            ],
            "yaw_world_from_tray_deg": float(model["yaw_world_from_tray_deg"]),
            "training_slots": list(model["training_slots"]),
            "holdout_slots": list(model["holdout_slots"]),
            "R_F_C_run_estimate": rfc_by_run[name].astype(float).tolist(),
        }
        for name, model in run_models.items()
    }
    calibrated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    return {
        "schema_version": 1,
        "status": status,
        "calibrated_at": calibrated_at,
        "model": "fixed_height_planar_camera1_to_forearm",
        "frame_definition": {
            "world_frame": "SCARA controller world XY",
            "forearm_frame": "F origin at J4 axis; yaw alpha=J1+J2",
            "camera_frame": "OpenCV camera1 frame",
            "tray_frame": "Stage2 rigid Tray frame T",
            "forearm_angle_formula": "alpha_deg = J1_deg + J2_deg",
            "j4_or_terminal_rz_used_for_camera_yaw": False,
        },
        "scope": {
            "planar_xy_supported": True,
            "fixed_height_only": True,
            "z_supported": False,
            "full_6dof_supported": False,
            "required_j3_mm": required_j3,
        },
        "installation_check": {
            "checked_at": calibrated_at,
            "valid_until": None,
            "expired": False,
            "policy": "no automatic calendar expiry configured; camera reconnect still triggers runtime probe",
        },
        "camera": {
            "source_index": source_index,
            "resolution": image_resolution,
        },
        "R_F_C": R_F_C.astype(float).tolist(),
        "locked_inputs": {
            "camera_intrinsics_path": str(intrinsics_path.resolve()),
            "camera_intrinsics_sha256": current_intrinsics_hash,
            "tray_geometry_path": str(geometry_path.resolve()),
            "tray_geometry_sha256": current_geometry_hash,
            "suction_target_path": str(suction_path.resolve()),
            "suction_target_sha256": _sha256(suction_path),
            "source_runs": source_records,
        },
        "source_run_models": run_payload,
        "training": {
            "aggregate_training_pose_count": len(all_training),
            "independent_pose_count": independent_pose_count,
            "world_xy_span_mm": world_span,
            "forearm_alpha_span_deg": alpha_span,
            "maximum_imaging_j3_deviation_mm": maximum_j3_deviation,
            "excluded_slots": sorted(DEFAULT_EXCLUDED_SLOTS),
            "stress_only_slots": sorted(DEFAULT_STRESS_SLOTS),
        },
        "validation": {
            "whole_slot_holdout": True,
            "random_frame_split_used": False,
            "xy_error_mm": xy_metric,
            "yaw_error_deg": yaw_metric,
            "rows": validation_rows,
            "stress_rows": stress_validation,
            "cross_run_rotation_difference_deg": {
                "values": cross_run_differences,
                "maximum": cross_run_max if cross_run_differences else None,
            },
        },
        "quality_policy": {
            "maximum_cross_run_rotation_difference_deg": quality.maximum_cross_run_rotation_difference_deg,
            "policy_note": (
                "Operator-authorized 1.00 deg cross-run limit; values near the "
                "limit have low margin and remain visible in the report."
            ),
        },
        "prevalidated_probe_poses": _choose_probe_poses(
            [
                item
                for item in all_training
                if item["run_name"] == str(source_runs[-1][0])
            ]
            or all_training,
            3,
        ),
        "quality_gates": gates,
    }


def install_planar_handeye(
    report: Mapping[str, Any], destination: Path
) -> Path:
    """Atomically install a successful report; failed fits never replace it."""

    if report.get("status") != "success":
        failed = [
            name for name, gate in (report.get("quality_gates") or {}).items()
            if not isinstance(gate, Mapping) or gate.get("passed") is not True
        ]
        raise ValueError("平面手眼质量门未全部通过，不安装：" + ", ".join(failed))
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        path,
        json.dumps(dict(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


__all__ = [
    "DEFAULT_EXCLUDED_SLOTS",
    "DEFAULT_QUALITY",
    "DEFAULT_STRESS_SLOTS",
    "PlanarHandEyeQualityConfig",
    "fit_planar_handeye",
    "install_planar_handeye",
]
