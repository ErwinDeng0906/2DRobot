"""Stage 3: estimate the Tray pose ``^C T_T`` from the rigid A-H board.

``^C T_T`` maps a point expressed in Tray Frame ``T`` into camera coordinates:

    ``p_C = R_CT @ p_T + t_CT``

The estimator detects only configured perimeter marker IDs, joins their image
corners to the measured 3D corners from ``tray_board_geometry.json``, runs
RANSAC PnP, removes whole-marker reprojection outliers, refines the pose, and
reports both global and per-marker pixel errors.  It never controls the robot.

OpenCV ArUco detection returns each decoded marker's corners in canonical order
``[UL, UR, DR, DL]``.  Stage-2 geometry uses exactly the same order.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

from .tray_board_geometry import validate_geometry


@dataclass(frozen=True)
class TrayPoseQualityConfig:
    """Quality gates for one independently estimated camera frame."""

    minimum_visible_markers: int = 3
    minimum_inlier_markers: int = 3
    ransac_reprojection_threshold_px: float = 3.0
    ransac_confidence: float = 0.999
    ransac_iterations: int = 200
    marker_outlier_threshold_px: float = 3.0
    maximum_global_rms_px: float = 3.0
    minimum_ransac_inlier_corner_ratio: float = 0.60
    minimum_object_span_mm: float = 50.0
    minimum_camera_height_above_tray_mm: float = 20.0
    minimum_object_depth_C_mm: float = 20.0
    maximum_refinement_iterations: int = 3


DEFAULT_POSE_QUALITY = TrayPoseQualityConfig()


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics and distortion valid at one image resolution."""

    K: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    source_path: str
    calibration_status: str
    global_rms_px: Optional[float]


@dataclass(frozen=True)
class TrayPoseEstimate:
    """Complete Stage-3 result for one image, including rejected frames."""

    success: bool
    quality_passed: bool
    failure_reason: Optional[str]
    visible_marker_ids: tuple[int, ...]
    used_marker_ids: tuple[int, ...]
    rejected_marker_ids: tuple[int, ...]
    ransac_inlier_corner_count: int
    object_span_mm: float
    reprojection_rms_px: Optional[float]
    per_marker_rms_px: dict[int, float]
    rvec_C_T: Optional[np.ndarray]
    tvec_C_T_mm: Optional[np.ndarray]
    T_C_T: Optional[np.ndarray]
    T_T_C: Optional[np.ndarray]
    camera_position_T_mm: Optional[np.ndarray]
    minimum_object_depth_C_mm: Optional[float]
    annotated_image: np.ndarray

    def to_json(self) -> dict[str, Any]:
        """Return a serialization-safe pose/quality record."""
        return {
            "success": self.success,
            "quality_passed": self.quality_passed,
            "failure_reason": self.failure_reason,
            "visible_marker_ids": list(self.visible_marker_ids),
            "used_marker_ids": list(self.used_marker_ids),
            "rejected_marker_ids": list(self.rejected_marker_ids),
            "ransac_inlier_corner_count": self.ransac_inlier_corner_count,
            "object_span_mm": self.object_span_mm,
            "reprojection_rms_px": self.reprojection_rms_px,
            "per_marker_rms_px": {
                str(key): value for key, value in self.per_marker_rms_px.items()
            },
            "rvec_C_T": (
                None
                if self.rvec_C_T is None
                else self.rvec_C_T.reshape(3).astype(float).tolist()
            ),
            "tvec_C_T_mm": (
                None
                if self.tvec_C_T_mm is None
                else self.tvec_C_T_mm.reshape(3).astype(float).tolist()
            ),
            "T_C_T": (
                None
                if self.T_C_T is None
                else self.T_C_T.astype(float).tolist()
            ),
            "T_T_C": (
                None
                if self.T_T_C is None
                else self.T_T_C.astype(float).tolist()
            ),
            "camera_position_T_mm": (
                None
                if self.camera_position_T_mm is None
                else self.camera_position_T_mm.reshape(3).astype(float).tolist()
            ),
            "minimum_object_depth_C_mm": self.minimum_object_depth_C_mm,
        }


def load_camera_intrinsics(
    path: Path,
    *,
    allow_unapproved_status: bool = False,
) -> CameraIntrinsics:
    """Load Task-6 intrinsics and enforce approval status by default.

    A report with ``status != success`` can be used only for offline diagnostics
    when ``allow_unapproved_status=True``.  Real placement must use approved
    intrinsics because low ChArUco pose diversity can bias K/distCoeffs despite
    a low calibration RMS.
    """
    path = Path(path).resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"找不到相机内参：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"相机内参不是有效JSON：{path}（{exc}）") from exc
    status = str(raw.get("status") or "unknown")
    if status != "success" and not allow_unapproved_status:
        raise ValueError(
            f"相机内参状态为 {status!r}，不是已批准的 success。"
            "请重新完成姿态多样性合格的Task6标定；离线诊断可显式允许未批准参数。"
        )
    K = np.asarray(raw.get("K"), dtype=np.float64)
    dist = np.asarray(raw.get("distCoeffs"), dtype=np.float64).reshape(-1, 1)
    resolution = raw.get("image_resolution") or raw.get("camera", {}).get(
        "resolution"
    )
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        raise ValueError("相机内参 K 必须是有限3x3矩阵")
    if dist.size < 4 or not np.all(np.isfinite(dist)):
        raise ValueError("distCoeffs 至少需要4个有限参数")
    if not isinstance(resolution, dict):
        raise ValueError("相机内参缺少 image_resolution")
    width = int(resolution.get("width", 0))
    height = int(resolution.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("相机内参分辨率无效")
    return CameraIntrinsics(
        K=K,
        dist_coeffs=dist,
        image_size=(width, height),
        source_path=str(path),
        calibration_status=status,
        global_rms_px=(
            None
            if raw.get("global_rms_px") is None
            else float(raw["global_rms_px"])
        ),
    )


def load_tray_board_geometry(path: Path) -> dict[str, Any]:
    """Load Stage-2 geometry and rerun its structural validation."""
    path = Path(path).resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"找不到Tray Board几何：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Tray Board几何不是有效JSON：{path}（{exc}）") from exc
    validation = validate_geometry(raw)
    if not validation["valid"]:
        raise ValueError("Tray Board几何无效：" + "; ".join(validation["errors"]))
    return raw


def make_transform_C_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Create homogeneous ``^C T_T`` from OpenCV PnP rotation/translation."""
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid homogeneous transform analytically."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("齐次变换必须是4x4")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply one homogeneous transform to N x 3 points."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T[:, :3]


class TrayBoardPoseEstimator:
    """Detect A-H and robustly estimate ``^C T_T`` for independent frames."""

    def __init__(
        self,
        geometry: Mapping[str, Any],
        intrinsics: CameraIntrinsics,
        quality: TrayPoseQualityConfig = DEFAULT_POSE_QUALITY,
    ) -> None:
        validation = validate_geometry(geometry)
        if not validation["valid"]:
            raise ValueError("Tray Board几何无效：" + "; ".join(validation["errors"]))
        self.geometry = dict(geometry)
        self.intrinsics = intrinsics
        self.quality = quality
        dictionary_name = str(geometry["aruco_board"]["dictionary"])
        dictionary_id = getattr(cv2.aruco, dictionary_name, None)
        if dictionary_id is None:
            raise ValueError(f"OpenCV不支持字典 {dictionary_name}")
        self.dictionary_name = dictionary_name
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        parameters = cv2.aruco.DetectorParameters()
        # Sub-pixel refinement reduces corner quantization noise for PnP.
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, parameters)
        self.object_corners_by_id: dict[int, np.ndarray] = {}
        self.label_by_id: dict[int, str] = {}
        for label, marker in geometry["markers"].items():
            marker_id = int(marker["id"])
            self.object_corners_by_id[marker_id] = np.asarray(
                marker["corners_T_mm"], dtype=np.float64
            ).reshape(4, 3)
            self.label_by_id[marker_id] = str(label)
        self.configured_ids = frozenset(self.object_corners_by_id)

    def _empty_result(
        self,
        image: np.ndarray,
        visible_ids: Sequence[int],
        reason: str,
    ) -> TrayPoseEstimate:
        annotated = image.copy()
        cv2.putText(
            annotated,
            reason,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return TrayPoseEstimate(
            success=False,
            quality_passed=False,
            failure_reason=reason,
            visible_marker_ids=tuple(sorted(set(int(x) for x in visible_ids))),
            used_marker_ids=(),
            rejected_marker_ids=(),
            ransac_inlier_corner_count=0,
            object_span_mm=0.0,
            reprojection_rms_px=None,
            per_marker_rms_px={},
            rvec_C_T=None,
            tvec_C_T_mm=None,
            T_C_T=None,
            T_T_C=None,
            camera_position_T_mm=None,
            minimum_object_depth_C_mm=None,
            annotated_image=annotated,
        )

    @staticmethod
    def _object_span(object_points: np.ndarray) -> float:
        xy = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)[:, :2]
        if len(xy) < 2:
            return 0.0
        delta = xy[:, None, :] - xy[None, :, :]
        return float(np.sqrt(np.max(np.sum(delta * delta, axis=2))))

    def _solve(
        self,
        observations: Mapping[int, np.ndarray],
    ) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        ids = list(observations)
        object_points = np.concatenate(
            [self.object_corners_by_id[marker_id] for marker_id in ids], axis=0
        ).astype(np.float64)
        image_points = np.concatenate(
            [observations[marker_id] for marker_id in ids], axis=0
        ).astype(np.float64)
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                self.intrinsics.K,
                self.intrinsics.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                iterationsCount=self.quality.ransac_iterations,
                reprojectionError=self.quality.ransac_reprojection_threshold_px,
                confidence=self.quality.ransac_confidence,
            )
        except cv2.error:
            return False, None, None, np.empty((0,), dtype=np.int32)
        if not ok or rvec is None or tvec is None:
            return False, None, None, np.empty((0,), dtype=np.int32)
        inlier_indices = (
            np.empty((0,), dtype=np.int32)
            if inliers is None
            else np.asarray(inliers, dtype=np.int32).reshape(-1)
        )
        # Preserve RANSAC robustness: refine on point inliers only.  Refining
        # immediately on all points would reintroduce the outliers RANSAC just
        # rejected.  Once whole-marker pruning stabilizes, estimate() performs
        # one final refinement using all remaining marker corners.
        refine_object = (
            object_points[inlier_indices]
            if len(inlier_indices) >= 4
            else object_points
        )
        refine_image = (
            image_points[inlier_indices]
            if len(inlier_indices) >= 4
            else image_points
        )
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                refine_object,
                refine_image,
                self.intrinsics.K,
                self.intrinsics.dist_coeffs,
                rvec,
                tvec,
            )
        except cv2.error:
            pass
        return True, rvec, tvec, inlier_indices

    def _refine_all_active(
        self,
        observations: Mapping[int, np.ndarray],
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Refine on all corners only after marker-level outliers are removed."""
        ids = list(observations)
        object_points = np.concatenate(
            [self.object_corners_by_id[marker_id] for marker_id in ids], axis=0
        ).astype(np.float64)
        image_points = np.concatenate(
            [observations[marker_id] for marker_id in ids], axis=0
        ).astype(np.float64)
        try:
            return cv2.solvePnPRefineLM(
                object_points,
                image_points,
                self.intrinsics.K,
                self.intrinsics.dist_coeffs,
                rvec,
                tvec,
            )
        except cv2.error:
            return rvec, tvec

    def _marker_errors(
        self,
        observations: Mapping[int, np.ndarray],
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> tuple[dict[int, float], float]:
        per_marker: dict[int, float] = {}
        all_residuals: list[np.ndarray] = []
        for marker_id, detected in observations.items():
            projected, _ = cv2.projectPoints(
                self.object_corners_by_id[marker_id],
                rvec,
                tvec,
                self.intrinsics.K,
                self.intrinsics.dist_coeffs,
            )
            residual = projected.reshape(-1, 2) - detected.reshape(-1, 2)
            all_residuals.append(residual)
            per_marker[marker_id] = math.sqrt(
                float(np.mean(np.sum(residual * residual, axis=1)))
            )
        stacked = np.concatenate(all_residuals, axis=0)
        global_rms = math.sqrt(
            float(np.mean(np.sum(stacked * stacked, axis=1)))
        )
        return per_marker, global_rms

    def estimate(self, image: np.ndarray) -> TrayPoseEstimate:
        """Estimate ``^C T_T`` and quality metrics from one BGR image."""
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("输入必须是有效BGR三通道图像")
        height, width = image.shape[:2]
        if (width, height) != self.intrinsics.image_size:
            return self._empty_result(
                image,
                (),
                (
                    f"分辨率{width}x{height}与内参"
                    f"{self.intrinsics.image_size[0]}x{self.intrinsics.image_size[1]}不一致"
                ),
            )

        marker_corners, marker_ids, _rejected = self.detector.detectMarkers(image)
        all_detected_ids = (
            [] if marker_ids is None else [int(x) for x in marker_ids.reshape(-1)]
        )
        observations: dict[int, np.ndarray] = {}
        display_corners: list[np.ndarray] = []
        display_ids: list[int] = []
        for corners, marker_id in zip(marker_corners, all_detected_ids):
            if marker_id not in self.configured_ids:
                continue
            # If an ID somehow appears twice, keep the larger image quadrilateral.
            candidate = np.asarray(corners, dtype=np.float64).reshape(4, 2)
            if marker_id in observations:
                old_area = abs(float(cv2.contourArea(observations[marker_id].astype(np.float32))))
                new_area = abs(float(cv2.contourArea(candidate.astype(np.float32))))
                if new_area <= old_area:
                    continue
            observations[marker_id] = candidate

        visible_ids = tuple(sorted(observations))
        if len(visible_ids) < self.quality.minimum_visible_markers:
            return self._empty_result(
                image,
                visible_ids,
                (
                    f"外围Marker不足：{len(visible_ids)}/"
                    f"{self.quality.minimum_visible_markers}"
                ),
            )
        object_span = self._object_span(
            np.concatenate(
                [self.object_corners_by_id[marker_id] for marker_id in visible_ids]
            )
        )
        if object_span < self.quality.minimum_object_span_mm:
            return self._empty_result(
                image,
                visible_ids,
                f"可见Board跨度不足：{object_span:.1f}mm",
            )

        active = dict(observations)
        rejected_ids: set[int] = set()
        ransac_inliers = np.empty((0,), dtype=np.int32)
        rvec: Optional[np.ndarray] = None
        tvec: Optional[np.ndarray] = None
        per_marker: dict[int, float] = {}
        global_rms: Optional[float] = None
        for _iteration in range(self.quality.maximum_refinement_iterations):
            ok, candidate_rvec, candidate_tvec, ransac_inliers = self._solve(active)
            if not ok or candidate_rvec is None or candidate_tvec is None:
                return self._empty_result(image, visible_ids, "solvePnPRansac失败")
            rvec, tvec = candidate_rvec, candidate_tvec
            per_marker, global_rms = self._marker_errors(active, rvec, tvec)
            bad = {
                marker_id
                for marker_id, error in per_marker.items()
                if error > self.quality.marker_outlier_threshold_px
            }
            if not bad:
                break
            if len(active) - len(bad) < self.quality.minimum_inlier_markers:
                break
            rejected_ids.update(bad)
            active = {
                marker_id: corners
                for marker_id, corners in active.items()
                if marker_id not in bad
            }

        # Always solve once more on the final active set.  This also covers the
        # case where the last allowed pruning iteration removed a marker.
        ok, final_rvec, final_tvec, ransac_inliers = self._solve(active)
        if not ok or final_rvec is None or final_tvec is None:
            return self._empty_result(image, visible_ids, "最终PnP没有产生位姿")
        rvec, tvec = final_rvec, final_tvec
        used_ids = tuple(sorted(active))
        rvec, tvec = self._refine_all_active(active, rvec, tvec)
        # Report errors only for the final active set; rejected IDs are explicit.
        per_marker, global_rms = self._marker_errors(active, rvec, tvec)
        transform_C_T = make_transform_C_T(rvec, tvec)
        transform_T_C = invert_transform(transform_C_T)
        camera_position_T = transform_T_C[:3, 3].copy()
        used_object_points = np.concatenate(
            [self.object_corners_by_id[marker_id] for marker_id in used_ids]
        )
        depths_C = transform_points(transform_C_T, used_object_points)[:, 2]
        minimum_depth_C = float(np.min(depths_C))
        inlier_ratio = len(ransac_inliers) / float(4 * len(used_ids))
        failure_reason: Optional[str] = None
        quality_passed = True
        if len(used_ids) < self.quality.minimum_inlier_markers:
            quality_passed = False
            failure_reason = (
                f"有效Marker不足：{len(used_ids)}/"
                f"{self.quality.minimum_inlier_markers}"
            )
        elif global_rms > self.quality.maximum_global_rms_px:
            quality_passed = False
            failure_reason = (
                f"重投影RMS过高：{global_rms:.3f}px > "
                f"{self.quality.maximum_global_rms_px:.3f}px"
            )
        elif max(per_marker.values(), default=0.0) > self.quality.marker_outlier_threshold_px:
            quality_passed = False
            worst_id = max(per_marker, key=per_marker.get)
            failure_reason = (
                f"Marker {worst_id} 重投影误差过高："
                f"{per_marker[worst_id]:.3f}px > "
                f"{self.quality.marker_outlier_threshold_px:.3f}px"
            )
        elif inlier_ratio < self.quality.minimum_ransac_inlier_corner_ratio:
            quality_passed = False
            failure_reason = (
                f"RANSAC角点内点比例过低：{inlier_ratio:.3f} < "
                f"{self.quality.minimum_ransac_inlier_corner_ratio:.3f}"
            )
        elif minimum_depth_C < self.quality.minimum_object_depth_C_mm:
            quality_passed = False
            failure_reason = (
                f"Board不在相机前方：最小深度{minimum_depth_C:.1f}mm"
            )
        elif camera_position_T[2] < self.quality.minimum_camera_height_above_tray_mm:
            quality_passed = False
            failure_reason = (
                f"相机位于Tray目标平面的错误一侧："
                f"z_T={camera_position_T[2]:.1f}mm"
            )

        annotated = image.copy()
        for marker_id in visible_ids:
            display_corners.append(
                np.asarray(observations[marker_id], dtype=np.float32).reshape(1, 4, 2)
            )
            display_ids.append(marker_id)
        if display_ids:
            cv2.aruco.drawDetectedMarkers(
                annotated,
                display_corners,
                np.asarray(display_ids, dtype=np.int32).reshape(-1, 1),
            )
        axis_length = 35.0
        try:
            axis_points = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [axis_length, 0.0, 0.0],
                    [0.0, axis_length, 0.0],
                    [0.0, 0.0, axis_length],
                ],
                dtype=np.float64,
            )
            projected_axes, _ = cv2.projectPoints(
                axis_points,
                rvec,
                tvec,
                self.intrinsics.K,
                self.intrinsics.dist_coeffs,
            )
            axis_pixels = projected_axes.reshape(-1, 2)
            if np.all(np.isfinite(axis_pixels)):
                origin = tuple(np.round(axis_pixels[0]).astype(int))
                # OpenCV convention: X red, Y green, Z blue in BGR drawing.
                for endpoint, axis_color in zip(
                    axis_pixels[1:],
                    ((0, 0, 255), (0, 255, 0), (255, 0, 0)),
                ):
                    cv2.line(
                        annotated,
                        origin,
                        tuple(np.round(endpoint).astype(int)),
                        axis_color,
                        2,
                        cv2.LINE_AA,
                    )
        except cv2.error:
            pass
        color = (0, 160, 0) if quality_passed else (0, 0, 255)
        cv2.putText(
            annotated,
            (
                f"A-H used={list(used_ids)} RMS={global_rms:.3f}px "
                f"{'PASS' if quality_passed else 'REJECT'}"
            ),
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
        return TrayPoseEstimate(
            success=True,
            quality_passed=quality_passed,
            failure_reason=failure_reason,
            visible_marker_ids=visible_ids,
            used_marker_ids=used_ids,
            rejected_marker_ids=tuple(sorted(rejected_ids)),
            ransac_inlier_corner_count=int(len(ransac_inliers)),
            object_span_mm=object_span,
            reprojection_rms_px=global_rms,
            per_marker_rms_px=per_marker,
            rvec_C_T=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            tvec_C_T_mm=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            T_C_T=transform_C_T,
            T_T_C=transform_T_C,
            camera_position_T_mm=camera_position_T,
            minimum_object_depth_C_mm=minimum_depth_C,
            annotated_image=annotated,
        )

    def project_tray_points(
        self,
        points_T_mm: np.ndarray,
        estimate: TrayPoseEstimate,
    ) -> np.ndarray:
        """Project arbitrary Tray-frame targets into the current image."""
        if estimate.rvec_C_T is None or estimate.tvec_C_T_mm is None:
            raise ValueError("位姿估计不包含可用的 ^C T_T")
        points = np.asarray(points_T_mm, dtype=np.float64).reshape(-1, 3)
        projected, _ = cv2.projectPoints(
            points,
            estimate.rvec_C_T,
            estimate.tvec_C_T_mm,
            self.intrinsics.K,
            self.intrinsics.dist_coeffs,
        )
        return projected.reshape(-1, 2)


__all__ = [
    "CameraIntrinsics",
    "DEFAULT_POSE_QUALITY",
    "TrayBoardPoseEstimator",
    "TrayPoseEstimate",
    "TrayPoseQualityConfig",
    "invert_transform",
    "load_camera_intrinsics",
    "load_tray_board_geometry",
    "make_transform_C_T",
    "transform_points",
]
