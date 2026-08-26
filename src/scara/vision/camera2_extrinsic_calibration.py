"""Fail-closed camera-2 to J4 extrinsic calibration.

The close-range camera is rigidly mounted on the SCARA J4/suction assembly.
This module estimates the rigid transform ``T_J4_C2`` from synchronized robot
states and ChArUco observations of a board whose world pose was measured
independently.  It contains no robot commands and never installs a calibration
unless every quality gate passes.

Transform notation follows the rest of the vision package: ``T_A_B`` maps a
homogeneous point expressed in frame B into frame A.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np

from scara.pipeline.kinematics import fk_wrist, rz_of

from .charuco_calibration import CharucoBoardSpec, DEFAULT_BOARD_SPEC


@dataclass(frozen=True)
class Camera2ExtrinsicQualityConfig:
    """Detection, pose-diversity and final-residual gates."""

    minimum_charuco_corners: int = 12
    minimum_pnp_inlier_ratio: float = 0.75
    maximum_pnp_rms_px: float = 1.0
    maximum_pnp_residual_px: float = 2.5
    minimum_observation_count: int = 12
    minimum_unique_robot_poses: int = 6
    minimum_world_xy_span_mm: float = 40.0
    minimum_j3_span_mm: float = 10.0
    minimum_rz_span_deg: float = 50.0
    maximum_translation_rms_mm: float = 1.0
    maximum_translation_residual_mm: float = 2.5
    maximum_rotation_rms_deg: float = 0.75
    maximum_rotation_residual_deg: float = 2.0
    minimum_inlier_ratio: float = 0.75
    maximum_board_pose_uncertainty_mm: float = 0.5
    maximum_board_pose_uncertainty_deg: float = 0.5


DEFAULT_QUALITY = Camera2ExtrinsicQualityConfig()


@dataclass(frozen=True)
class Camera2ExtrinsicObservation:
    """One accepted camera-2 board pose synchronized with a robot state."""

    measurement_id: str
    image_path: str
    point_sequence: int
    captured_at: str
    transform_W_J4: np.ndarray
    transform_C2_B: np.ndarray
    charuco_corner_count: int
    pnp_inlier_count: int
    pnp_inlier_ratio: float
    reprojection_rms_px: float
    reprojection_max_px: float

    def to_json(self) -> dict[str, Any]:
        return {
            "measurement_id": self.measurement_id,
            "image_path": self.image_path,
            "point_sequence": self.point_sequence,
            "captured_at": self.captured_at,
            "transform_W_J4": self.transform_W_J4.astype(float).tolist(),
            "transform_C2_B": self.transform_C2_B.astype(float).tolist(),
            "charuco_corner_count": self.charuco_corner_count,
            "pnp_inlier_count": self.pnp_inlier_count,
            "pnp_inlier_ratio": self.pnp_inlier_ratio,
            "reprojection_rms_px": self.reprojection_rms_px,
            "reprojection_max_px": self.reprojection_max_px,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _finite_matrix(value: object, shape: tuple[int, int], label: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64).reshape(shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是{shape[0]}x{shape[1]}矩阵") from exc
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label}包含NaN或Inf")
    return matrix


def _rigid_transform(value: object, label: str) -> np.ndarray:
    transform = _finite_matrix(value, (4, 4), label)
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{label}最后一行不是[0,0,0,1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{label}旋转部分不正交")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError(f"{label}旋转部分行列式不是+1")
    return transform


def _finite_scalar(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是有限数值") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label}必须是有限数值")
    return result


def _rotation_x(angle_deg: float) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def _rotation_y(angle_deg: float) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _rotation_z(angle_deg: float) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def transform_from_controller_pose(pose: Mapping[str, Any]) -> np.ndarray:
    """Build ``T_W_J4`` from ActionWorker's mechanical-center record.

    The controller reports extrinsic XYZ Euler angles.  The established SCARA
    convention is represented by ``Rz @ Ry @ Rx``; normal downward poses have
    Rx close to 180 degrees.
    """

    x = _finite_scalar(pose.get("x_mm"), "mechanical_center.x_mm")
    y = _finite_scalar(pose.get("y_mm"), "mechanical_center.y_mm")
    z = _finite_scalar(pose.get("z_mm"), "mechanical_center.z_mm")
    rx = _finite_scalar(pose.get("Rx_deg"), "mechanical_center.Rx_deg")
    ry = _finite_scalar(pose.get("Ry_deg"), "mechanical_center.Ry_deg")
    rz = _finite_scalar(pose.get("Rz_deg"), "mechanical_center.Rz_deg")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rotation_z(rz) @ _rotation_y(ry) @ _rotation_x(rx)
    transform[:3, 3] = [x, y, z]
    return transform


def invert_transform(transform: Sequence[Sequence[float]]) -> np.ndarray:
    value = _rigid_transform(transform, "transform")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = value[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ value[:3, 3]
    return result


def rotation_error_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left[:3, :3]).T @ np.asarray(right[:3, :3])
    cosine = float((np.trace(relative) - 1.0) * 0.5)
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


def _average_rotations(rotations: Sequence[np.ndarray]) -> np.ndarray:
    if not rotations:
        raise ValueError("无法平均空的旋转集合")
    accumulator = np.sum(np.asarray(rotations, dtype=np.float64), axis=0)
    left, _singular, right_t = np.linalg.svd(accumulator)
    result = left @ right_t
    if np.linalg.det(result) < 0.0:
        left[:, -1] *= -1.0
        result = left @ right_t
    return result


def _circular_span_deg(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    normalized = np.sort(np.mod(np.asarray(values, dtype=np.float64), 360.0))
    gaps = np.diff(np.concatenate((normalized, [normalized[0] + 360.0])))
    return float(360.0 - np.max(gaps))


def _rotation_to_euler_zyx_deg(rotation: np.ndarray) -> list[float]:
    value = _finite_matrix(rotation, (3, 3), "rotation")
    sy = math.hypot(float(value[0, 0]), float(value[1, 0]))
    if sy > 1e-9:
        rx = math.atan2(float(value[2, 1]), float(value[2, 2]))
        ry = math.atan2(-float(value[2, 0]), sy)
        rz = math.atan2(float(value[1, 0]), float(value[0, 0]))
    else:
        rx = math.atan2(-float(value[1, 2]), float(value[1, 1]))
        ry = math.atan2(-float(value[2, 0]), sy)
        rz = 0.0
    return [math.degrees(rx), math.degrees(ry), math.degrees(rz)]


def load_camera2_intrinsics(path: Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"找不到相机2内参：{source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"相机2内参不是有效JSON：{source}") from exc
    if payload.get("status") != "success":
        raise ValueError("相机2内参status不是success")
    camera = payload.get("camera") or {}
    if int(camera.get("source_index", -1)) != 2:
        raise ValueError("相机2内参的source_index不是2")
    resolution = payload.get("image_resolution") or camera.get("resolution") or {}
    width = int(resolution.get("width", 0))
    height = int(resolution.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("相机2内参缺少有效分辨率")
    board = dict(payload.get("board") or {})
    validate_board_identity(board, DEFAULT_BOARD_SPEC, "相机2内参board")
    distortion = np.asarray(payload.get("distCoeffs"), dtype=np.float64).reshape(-1, 1)
    if distortion.size < 4 or not np.all(np.isfinite(distortion)):
        raise ValueError("相机2内参distCoeffs无效")
    return {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "K": _finite_matrix(payload.get("K"), (3, 3), "K"),
        "distCoeffs": distortion,
        "resolution": (width, height),
        "board": board,
        "global_rms_px": _finite_scalar(payload.get("global_rms_px"), "global_rms_px"),
    }


def load_board_pose_world(path: Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"找不到ChArUco板世界位姿：{source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"ChArUco板世界位姿不是有效JSON：{source}") from exc
    if payload.get("status") not in {"measured", "success"}:
        raise ValueError("ChArUco板世界位姿尚未测量，status必须是measured或success")
    uncertainty = payload.get("uncertainty") or {}
    board = dict(payload.get("board") or {})
    validate_board_identity(board, DEFAULT_BOARD_SPEC, "板世界位姿board")
    translation_uncertainty = _finite_scalar(
        uncertainty.get("translation_mm"), "uncertainty.translation_mm"
    )
    rotation_uncertainty = _finite_scalar(
        uncertainty.get("rotation_deg"), "uncertainty.rotation_deg"
    )
    if translation_uncertainty < 0.0 or rotation_uncertainty < 0.0:
        raise ValueError("标定板位姿不确定度不能为负数")
    return {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "transform_W_B": _rigid_transform(payload.get("transform_W_B"), "transform_W_B"),
        "translation_uncertainty_mm": translation_uncertainty,
        "rotation_uncertainty_deg": rotation_uncertainty,
        "measurement_method": str(payload.get("measurement_method") or ""),
        "board": board,
    }


def validate_board_identity(
    payload: Mapping[str, Any],
    expected: CharucoBoardSpec,
    label: str,
) -> None:
    reference = expected.to_json()
    keys = (
        "type",
        "squares_x",
        "squares_y",
        "square_length_mm",
        "marker_length_mm",
        "dictionary",
        "legacy_pattern",
    )
    for key in keys:
        actual = payload.get(key)
        wanted = reference[key]
        if isinstance(wanted, float):
            try:
                matches = math.isclose(float(actual), wanted, abs_tol=1e-6)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == wanted
        if not matches:
            raise ValueError(
                f"{label}.{key}={actual!r}与要求的{wanted!r}不一致"
            )


def board_pose_from_three_world_points(
    origin_world_mm: Sequence[float],
    x_reference_world_mm: Sequence[float],
    y_reference_world_mm: Sequence[float],
) -> tuple[np.ndarray, dict[str, float]]:
    """Construct ``T_W_B`` from measured board origin/+X/+Y points.

    The +Y measurement is orthogonalized against +X.  Diagnostics retain the
    raw angle so a poorly placed or incorrectly identified reference point is
    visible rather than silently accepted by the caller.
    """

    origin = np.asarray(origin_world_mm, dtype=np.float64).reshape(-1)
    x_reference = np.asarray(x_reference_world_mm, dtype=np.float64).reshape(-1)
    y_reference = np.asarray(y_reference_world_mm, dtype=np.float64).reshape(-1)
    if any(value.size != 3 for value in (origin, x_reference, y_reference)):
        raise ValueError("origin/x-reference/y-reference都必须包含XYZ三个数值")
    if not all(np.all(np.isfinite(value)) for value in (origin, x_reference, y_reference)):
        raise ValueError("三点测量包含NaN或Inf")
    x_vector = x_reference - origin
    y_vector = y_reference - origin
    x_length = float(np.linalg.norm(x_vector))
    y_length = float(np.linalg.norm(y_vector))
    if x_length < 20.0 or y_length < 20.0:
        raise ValueError("板坐标参考点距离原点必须至少20 mm")
    x_axis = x_vector / x_length
    raw_cosine = float(np.dot(x_vector, y_vector) / (x_length * y_length))
    raw_angle = math.degrees(math.acos(min(1.0, max(-1.0, raw_cosine))))
    y_orthogonal = y_vector - np.dot(y_vector, x_axis) * x_axis
    y_orthogonal_length = float(np.linalg.norm(y_orthogonal))
    if y_orthogonal_length < 20.0:
        raise ValueError("板+X和+Y参考方向近似共线")
    y_axis = y_orthogonal / y_orthogonal_length
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    transform[:3, 3] = origin
    return transform, {
        "x_reference_distance_mm": x_length,
        "y_reference_distance_mm": y_length,
        "raw_xy_angle_deg": raw_angle,
        "orthogonalization_correction_deg": abs(90.0 - raw_angle),
    }


class Camera2CharucoPoseDetector:
    """Estimate ``T_C2_B`` with locked camera intrinsics."""

    def __init__(
        self,
        intrinsics: Mapping[str, Any],
        board_spec: CharucoBoardSpec = DEFAULT_BOARD_SPEC,
        quality: Camera2ExtrinsicQualityConfig = DEFAULT_QUALITY,
    ) -> None:
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoDetector"):
            raise RuntimeError("当前OpenCV缺少aruco/CharucoDetector")
        self.intrinsics = intrinsics
        self.board_spec = board_spec
        self.quality = quality
        dictionary_id = getattr(cv2.aruco, board_spec.dictionary_name, None)
        if dictionary_id is None:
            raise ValueError(f"OpenCV不支持字典{board_spec.dictionary_name}")
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.board = cv2.aruco.CharucoBoard(
            (board_spec.squares_x, board_spec.squares_y),
            board_spec.square_length_mm,
            board_spec.marker_length_mm,
            dictionary,
        )
        self.board.setLegacyPattern(board_spec.legacy_pattern)
        self.detector = cv2.aruco.CharucoDetector(self.board)

    def detect(self, image: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("相机2图像必须是有效BGR三通道图像")
        expected_width, expected_height = self.intrinsics["resolution"]
        height, width = image.shape[:2]
        if (width, height) != (expected_width, expected_height):
            raise ValueError(
                f"相机2图像分辨率为{width}x{height}，标定要求"
                f"{expected_width}x{expected_height}"
            )
        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            self.detector.detectBoard(image)
        )
        corner_count = 0 if charuco_ids is None else int(len(charuco_ids))
        marker_count = 0 if marker_ids is None else int(len(marker_ids))
        if corner_count < self.quality.minimum_charuco_corners:
            raise ValueError(
                f"ChArUco角点只有{corner_count}个，至少需要"
                f"{self.quality.minimum_charuco_corners}个"
            )
        if self.board.checkCharucoCornersCollinear(charuco_ids):
            raise ValueError("ChArUco角点近似共线，无法求位姿")
        object_points, image_points = self.board.matchImagePoints(
            charuco_corners, charuco_ids
        )
        object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
        image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            self.intrinsics["K"],
            self.intrinsics["distCoeffs"],
            iterationsCount=200,
            reprojectionError=self.quality.maximum_pnp_residual_px,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or inliers is None or len(inliers) < 4:
            raise ValueError("相机2 ChArUco solvePnPRansac失败")
        inlier_indices = np.asarray(inliers, dtype=np.int32).reshape(-1)
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points[inlier_indices],
            image_points[inlier_indices],
            self.intrinsics["K"],
            self.intrinsics["distCoeffs"],
            rvec,
            tvec,
        )
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.intrinsics["K"],
            self.intrinsics["distCoeffs"],
        )
        residuals = np.linalg.norm(
            projected.reshape(-1, 2) - image_points,
            axis=1,
        )
        inlier_ratio = float(len(inlier_indices) / len(object_points))
        rms = float(math.sqrt(float(np.mean(np.square(residuals[inlier_indices])))))
        maximum = float(np.max(residuals[inlier_indices]))
        if inlier_ratio < self.quality.minimum_pnp_inlier_ratio:
            raise ValueError(
                f"PnP内点率{inlier_ratio:.3f}低于"
                f"{self.quality.minimum_pnp_inlier_ratio:.3f}"
            )
        if rms > self.quality.maximum_pnp_rms_px:
            raise ValueError(
                f"PnP重投影RMS {rms:.3f}px超过"
                f"{self.quality.maximum_pnp_rms_px:.3f}px"
            )
        if maximum > self.quality.maximum_pnp_residual_px:
            raise ValueError(
                f"PnP最大内点残差{maximum:.3f}px超过"
                f"{self.quality.maximum_pnp_residual_px:.3f}px"
            )
        rotation, _ = cv2.Rodrigues(rvec)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
        annotated = image.copy()
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)
        cv2.drawFrameAxes(
            annotated,
            self.intrinsics["K"],
            self.intrinsics["distCoeffs"],
            rvec,
            tvec,
            board_spec_axis_length(self.board_spec),
            3,
        )
        return {
            "transform_C2_B": transform,
            "charuco_corner_count": corner_count,
            "marker_count": marker_count,
            "pnp_inlier_count": int(len(inlier_indices)),
            "pnp_inlier_ratio": inlier_ratio,
            "reprojection_rms_px": rms,
            "reprojection_max_px": maximum,
        }, annotated


def board_spec_axis_length(board_spec: CharucoBoardSpec) -> float:
    return float(min(board_spec.squares_x, board_spec.squares_y) * board_spec.square_length_mm * 0.25)


def _gate(passed: bool, actual: object, limit: str) -> dict[str, Any]:
    return {"passed": bool(passed), "actual": actual, "limit": limit}


def _count_distinct_robot_poses(
    observations: Sequence[Camera2ExtrinsicObservation],
    *,
    translation_tolerance_mm: float = 1.0,
    rotation_tolerance_deg: float = 1.0,
) -> int:
    """Cluster repeated stationary frames instead of counting numeric jitter."""

    representatives: list[np.ndarray] = []
    for observation in observations:
        pose = _rigid_transform(observation.transform_W_J4, "transform_W_J4")
        if any(
            np.linalg.norm(pose[:3, 3] - reference[:3, 3])
            <= translation_tolerance_mm
            and rotation_error_deg(reference, pose) <= rotation_tolerance_deg
            for reference in representatives
        ):
            continue
        representatives.append(pose)
    return len(representatives)


def _candidate_transform(
    observation: Camera2ExtrinsicObservation,
    transform_W_B: np.ndarray,
) -> np.ndarray:
    # T_W_C2 = T_W_B * T_B_C2; T_J4_C2 = T_J4_W * T_W_C2.
    return (
        invert_transform(observation.transform_W_J4)
        @ transform_W_B
        @ invert_transform(observation.transform_C2_B)
    )


def solve_known_board_extrinsic(
    observations: Sequence[Camera2ExtrinsicObservation],
    board_pose: Mapping[str, Any],
    *,
    quality: Camera2ExtrinsicQualityConfig = DEFAULT_QUALITY,
) -> dict[str, Any]:
    """Robustly estimate ``T_J4_C2`` when ``T_W_B`` is independently known."""

    transform_W_B = _rigid_transform(board_pose.get("transform_W_B"), "transform_W_B")
    candidates = [
        _candidate_transform(observation, transform_W_B)
        for observation in observations
    ]
    active = np.ones(len(candidates), dtype=bool)
    for _iteration in range(4):
        active_indices = np.flatnonzero(active)
        if len(active_indices) < 3:
            break
        translations = np.asarray(
            [candidates[index][:3, 3] for index in active_indices],
            dtype=np.float64,
        )
        translation_center = np.median(translations, axis=0)
        rotation_center = _average_rotations(
            [candidates[index][:3, :3] for index in active_indices]
        )
        translation_errors = np.asarray(
            [
                np.linalg.norm(candidate[:3, 3] - translation_center)
                for candidate in candidates
            ],
            dtype=np.float64,
        )
        rotation_errors = np.asarray(
            [
                rotation_error_deg(
                    np.block(
                        [
                            [rotation_center, np.zeros((3, 1))],
                            [np.zeros((1, 3)), np.ones((1, 1))],
                        ]
                    ),
                    candidate,
                )
                for candidate in candidates
            ],
            dtype=np.float64,
        )
        t_active = translation_errors[active]
        r_active = rotation_errors[active]
        t_median = float(np.median(t_active))
        r_median = float(np.median(r_active))
        t_sigma = max(0.10, 1.4826 * float(np.median(np.abs(t_active - t_median))))
        r_sigma = max(0.05, 1.4826 * float(np.median(np.abs(r_active - r_median))))
        new_active = (
            (translation_errors <= max(quality.maximum_translation_residual_mm, t_median + 3.0 * t_sigma))
            & (rotation_errors <= max(quality.maximum_rotation_residual_deg, r_median + 3.0 * r_sigma))
        )
        if np.array_equal(new_active, active):
            break
        active = new_active

    inlier_indices = np.flatnonzero(active)
    if len(inlier_indices):
        translation = np.median(
            np.asarray([candidates[index][:3, 3] for index in inlier_indices]),
            axis=0,
        )
        rotation = _average_rotations(
            [candidates[index][:3, :3] for index in inlier_indices]
        )
    else:
        translation = np.asarray([math.nan, math.nan, math.nan])
        rotation = np.eye(3, dtype=np.float64)
    estimate = np.eye(4, dtype=np.float64)
    estimate[:3, :3] = rotation
    estimate[:3, 3] = translation

    translation_residuals: list[float] = []
    rotation_residuals: list[float] = []
    sample_rows: list[dict[str, Any]] = []
    for index, (observation, candidate) in enumerate(zip(observations, candidates)):
        translation_error = float(np.linalg.norm(candidate[:3, 3] - translation))
        rotation_error = rotation_error_deg(estimate, candidate)
        reconstructed_board = (
            observation.transform_W_J4
            @ estimate
            @ observation.transform_C2_B
        )
        board_translation_error = float(
            np.linalg.norm(reconstructed_board[:3, 3] - transform_W_B[:3, 3])
        )
        board_rotation_error = rotation_error_deg(transform_W_B, reconstructed_board)
        if active[index]:
            translation_residuals.append(board_translation_error)
            rotation_residuals.append(board_rotation_error)
        sample_rows.append(
            {
                "measurement_id": observation.measurement_id,
                "image_path": observation.image_path,
                "inlier": bool(active[index]),
                "candidate_translation_J4_C2_mm": candidate[:3, 3].astype(float).tolist(),
                "candidate_rotation_error_deg": rotation_error,
                "candidate_translation_error_mm": translation_error,
                "reconstructed_board_translation_error_mm": board_translation_error,
                "reconstructed_board_rotation_error_deg": board_rotation_error,
                "pnp_rms_px": observation.reprojection_rms_px,
            }
        )

    translations_W = np.asarray(
        [observation.transform_W_J4[:3, 3] for observation in observations],
        dtype=np.float64,
    ) if observations else np.empty((0, 3), dtype=np.float64)
    j3_values = translations_W[:, 2] if len(translations_W) else np.asarray([])
    rz_values = [
        math.degrees(
            math.atan2(
                float(observation.transform_W_J4[1, 0]),
                float(observation.transform_W_J4[0, 0]),
            )
        )
        for observation in observations
    ]
    unique_robot_pose_count = _count_distinct_robot_poses(observations)
    translation_rms = (
        float(math.sqrt(float(np.mean(np.square(translation_residuals)))))
        if translation_residuals
        else math.inf
    )
    rotation_rms = (
        float(math.sqrt(float(np.mean(np.square(rotation_residuals)))))
        if rotation_residuals
        else math.inf
    )
    translation_max = max(translation_residuals, default=math.inf)
    rotation_max = max(rotation_residuals, default=math.inf)
    inlier_ratio = len(inlier_indices) / max(len(observations), 1)
    xy_span = (
        float(np.linalg.norm(np.ptp(translations_W[:, :2], axis=0)))
        if len(translations_W)
        else 0.0
    )
    j3_span = float(np.ptp(j3_values)) if len(j3_values) else 0.0
    rz_span = _circular_span_deg(rz_values)
    board_t_uncertainty = float(board_pose.get("translation_uncertainty_mm", math.inf))
    board_r_uncertainty = float(board_pose.get("rotation_uncertainty_deg", math.inf))
    gates = {
        "minimum_observation_count": _gate(
            len(observations) >= quality.minimum_observation_count,
            len(observations),
            f">={quality.minimum_observation_count}",
        ),
        "minimum_unique_robot_poses": _gate(
            unique_robot_pose_count >= quality.minimum_unique_robot_poses,
            unique_robot_pose_count,
            f">={quality.minimum_unique_robot_poses}",
        ),
        "world_xy_span": _gate(
            xy_span >= quality.minimum_world_xy_span_mm,
            xy_span,
            f">={quality.minimum_world_xy_span_mm:.3f} mm",
        ),
        "j3_span": _gate(
            j3_span >= quality.minimum_j3_span_mm,
            j3_span,
            f">={quality.minimum_j3_span_mm:.3f} mm",
        ),
        "rz_span": _gate(
            rz_span >= quality.minimum_rz_span_deg,
            rz_span,
            f">={quality.minimum_rz_span_deg:.3f} deg",
        ),
        "inlier_ratio": _gate(
            inlier_ratio >= quality.minimum_inlier_ratio,
            inlier_ratio,
            f">={quality.minimum_inlier_ratio:.3f}",
        ),
        "translation_rms": _gate(
            translation_rms <= quality.maximum_translation_rms_mm,
            translation_rms,
            f"<={quality.maximum_translation_rms_mm:.3f} mm",
        ),
        "translation_max": _gate(
            translation_max <= quality.maximum_translation_residual_mm,
            translation_max,
            f"<={quality.maximum_translation_residual_mm:.3f} mm",
        ),
        "rotation_rms": _gate(
            rotation_rms <= quality.maximum_rotation_rms_deg,
            rotation_rms,
            f"<={quality.maximum_rotation_rms_deg:.3f} deg",
        ),
        "rotation_max": _gate(
            rotation_max <= quality.maximum_rotation_residual_deg,
            rotation_max,
            f"<={quality.maximum_rotation_residual_deg:.3f} deg",
        ),
        "board_translation_uncertainty": _gate(
            board_t_uncertainty <= quality.maximum_board_pose_uncertainty_mm,
            board_t_uncertainty,
            f"<={quality.maximum_board_pose_uncertainty_mm:.3f} mm",
        ),
        "board_rotation_uncertainty": _gate(
            board_r_uncertainty <= quality.maximum_board_pose_uncertainty_deg,
            board_r_uncertainty,
            f"<={quality.maximum_board_pose_uncertainty_deg:.3f} deg",
        ),
    }
    passed = bool(gates) and all(gate["passed"] for gate in gates.values())
    return {
        "schema_version": 1,
        "status": "success" if passed else "rejected",
        "method": "known-board-pose robust per-frame T_J4_C2 composition",
        "coordinate_definition": {
            "transform_J4_C2": "maps camera2 coordinates into the controller J4/TCP frame",
            "transform_C2_J4": "inverse transform; maps J4/TCP coordinates into camera2",
            "controller_euler": "R_W_J4 = Rz(Rz) @ Ry(Ry) @ Rx(Rx)",
        },
        "transform_J4_C2": estimate.astype(float).tolist(),
        "transform_C2_J4": invert_transform(estimate).astype(float).tolist(),
        "translation_J4_C2_mm": estimate[:3, 3].astype(float).tolist(),
        "rotation_J4_C2_euler_zyx_deg": _rotation_to_euler_zyx_deg(estimate[:3, :3]),
        "quality_gates": gates,
        "metrics": {
            "observation_count": len(observations),
            "inlier_count": int(len(inlier_indices)),
            "inlier_ratio": float(inlier_ratio),
            "unique_robot_pose_count": unique_robot_pose_count,
            "world_xy_span_mm": xy_span,
            "j3_span_mm": j3_span,
            "rz_span_deg": rz_span,
            "translation_rms_mm": translation_rms,
            "translation_max_mm": translation_max,
            "rotation_rms_deg": rotation_rms,
            "rotation_max_deg": rotation_max,
        },
        "samples": sample_rows,
        "robot_motion_authorized": False,
        "installation_allowed": passed,
    }


def observations_from_run(
    run_dir: Path,
    detector: Camera2CharucoPoseDetector,
    *,
    annotated_dir: Optional[Path] = None,
) -> tuple[list[Camera2ExtrinsicObservation], list[dict[str, Any]]]:
    """Read one ActionWorker run and pair source-2 photos with robot states."""

    root = Path(run_dir)
    manifest_path = root / "points.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"运行目录缺少points.json：{root}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"points.json不是有效JSON：{manifest_path}") from exc
    task_name = str(manifest.get("task_name") or "").strip()
    if not _is_task18_manifest(manifest):
        raise ValueError(
            f"运行目录不是Task18相机2外参采集：{task_name or '缺少task_name'}"
        )
    points = {
        int(point.get("sequence")): point
        for point in manifest.get("points", [])
        if isinstance(point, dict) and point.get("sequence") is not None
    }
    observations: list[Camera2ExtrinsicObservation] = []
    rejected: list[dict[str, Any]] = []
    if annotated_dir is not None:
        Path(annotated_dir).mkdir(parents=True, exist_ok=True)
    for photo in manifest.get("photos", []):
        if not isinstance(photo, dict) or int(photo.get("source", -1)) != 2:
            continue
        filename = str(photo.get("filename") or "")
        point_sequence = int(photo.get("point_sequence", -1))
        image_path = root / filename
        point = points.get(point_sequence)
        measurement_id = f"{root.name}:{filename}"
        try:
            if point is None:
                raise ValueError(f"找不到point_sequence={point_sequence}的机器人状态")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"无法读取照片{image_path}")
            detected, annotated = detector.detect(image)
            if annotated_dir is not None:
                output_path = Path(annotated_dir) / f"{root.name}_{filename}"
                if not cv2.imwrite(str(output_path), annotated):
                    raise ValueError(f"无法写入标注图{output_path}")
            validate_recorded_robot_state(point)
            observations.append(
                Camera2ExtrinsicObservation(
                    measurement_id=measurement_id,
                    image_path=str(image_path.resolve()),
                    point_sequence=point_sequence,
                    captured_at=str(photo.get("captured_at") or ""),
                    transform_W_J4=transform_from_controller_pose(
                        point.get("mechanical_center") or {}
                    ),
                    transform_C2_B=detected["transform_C2_B"],
                    charuco_corner_count=int(detected["charuco_corner_count"]),
                    pnp_inlier_count=int(detected["pnp_inlier_count"]),
                    pnp_inlier_ratio=float(detected["pnp_inlier_ratio"]),
                    reprojection_rms_px=float(detected["reprojection_rms_px"]),
                    reprojection_max_px=float(detected["reprojection_max_px"]),
                )
            )
        except Exception as exc:  # represented in the report, never guessed
            rejected.append(
                {
                    "measurement_id": measurement_id,
                    "image_path": str(image_path),
                    "reason": str(exc) or exc.__class__.__name__,
                }
            )
    return observations, rejected


def validate_recorded_robot_state(point: Mapping[str, Any]) -> None:
    """Reject manifests whose mechanical center is not the declared J4 axis."""

    joints = point.get("joints") or {}
    mechanical = point.get("mechanical_center") or {}
    j1 = _finite_scalar(joints.get("J1_deg"), "joints.J1_deg")
    j2 = _finite_scalar(joints.get("J2_deg"), "joints.J2_deg")
    j3 = _finite_scalar(joints.get("J3_mm"), "joints.J3_mm")
    j4 = _finite_scalar(joints.get("J4_deg"), "joints.J4_deg")
    expected_x, expected_y = fk_wrist(j1, j2)
    actual_x = _finite_scalar(mechanical.get("x_mm"), "mechanical_center.x_mm")
    actual_y = _finite_scalar(mechanical.get("y_mm"), "mechanical_center.y_mm")
    actual_z = _finite_scalar(mechanical.get("z_mm"), "mechanical_center.z_mm")
    actual_rx = _finite_scalar(mechanical.get("Rx_deg"), "mechanical_center.Rx_deg")
    actual_ry = _finite_scalar(mechanical.get("Ry_deg"), "mechanical_center.Ry_deg")
    actual_rz = _finite_scalar(mechanical.get("Rz_deg"), "mechanical_center.Rz_deg")
    xy_error = math.hypot(actual_x - expected_x, actual_y - expected_y)
    z_error = abs(actual_z - j3)
    rx_error = abs(((actual_rx - 180.0 + 180.0) % 360.0) - 180.0)
    ry_error = abs(((actual_ry + 180.0) % 360.0) - 180.0)
    rz_error = abs(((actual_rz - rz_of(j1, j2, j4) + 180.0) % 360.0) - 180.0)
    if xy_error > 0.5:
        raise ValueError(f"机械中心与J4正运动学XY相差{xy_error:.3f} mm")
    if z_error > 0.2:
        raise ValueError(f"mechanical_center.z与J3相差{z_error:.3f} mm")
    if rx_error > 0.2 or ry_error > 0.2:
        raise ValueError(
            "mechanical_center不是SCARA标准向下姿态："
            f"Rx error={rx_error:.3f} deg, Ry error={ry_error:.3f} deg"
        )
    if rz_error > 0.2:
        raise ValueError(f"mechanical_center.Rz与J1+J2+J4约定相差{rz_error:.3f} deg")


def collect_run_directories(paths: Iterable[Path]) -> list[Path]:
    """Return unique Task18 run folders from explicit runs or a parent."""

    result: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        candidates = [path] if (path / "points.json").is_file() else sorted(
            item.parent for item in path.rglob("points.json")
        )
        for candidate in candidates:
            manifest_path = candidate / "points.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                continue
            if not _is_task18_manifest(manifest):
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                result.append(resolved)
    return result


def _is_task18_manifest(manifest: Mapping[str, Any]) -> bool:
    task_name = str(manifest.get("task_name") or "").strip().lower()
    return task_name.startswith("task18 ") or task_name.startswith("task18_")


__all__ = [
    "Camera2CharucoPoseDetector",
    "Camera2ExtrinsicObservation",
    "Camera2ExtrinsicQualityConfig",
    "DEFAULT_QUALITY",
    "board_pose_from_three_world_points",
    "collect_run_directories",
    "invert_transform",
    "load_board_pose_world",
    "load_camera2_intrinsics",
    "observations_from_run",
    "rotation_error_deg",
    "sha256_file",
    "solve_known_board_extrinsic",
    "transform_from_controller_pose",
    "validate_board_identity",
    "validate_recorded_robot_state",
]
