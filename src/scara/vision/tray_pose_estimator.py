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
from itertools import combinations
from dataclasses import dataclass, field
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
    detected_corners_px: dict[int, np.ndarray] = field(default_factory=dict)

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


class _EdgeRefinedDetector:
    """Locate globally, decode/refine metric markers from local edge pixels.

    AprilTag edge fitting avoids the observed SUBPIX corner bias. Applying it
    to bounded marker ROIs avoids repeating expensive quad segmentation over
    the entire camera image. An insufficient seed always tries the full image.
    """

    def __init__(self, dictionary, metric_ids, minimum_count):
        self.metric_ids = frozenset(metric_ids)
        self.minimum_count = minimum_count
        native = cv2.aruco.DetectorParameters()
        native.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        edge = cv2.aruco.DetectorParameters()
        edge.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.native = cv2.aruco.ArucoDetector(dictionary, native)
        self.edge = cv2.aruco.ArucoDetector(dictionary, edge)

    def detectMarkers(self, image):
        corners, ids, rejected = self.native.detectMarkers(image)
        if ids is None:
            return self.edge.detectMarkers(image)
        observations = {}
        height, width = image.shape[:2]
        for raw_id, raw_points in zip(ids.reshape(-1), corners):
            key = int(raw_id)
            points = np.asarray(raw_points, dtype=np.float32).reshape(4, 2)
            if key not in self.metric_ids:
                observations[key] = points.reshape(1, 4, 2)
                continue
            pad = max(10, int(.25 * np.max(np.ptp(points, axis=0))))
            low = np.maximum(np.floor(points.min(axis=0)).astype(int) - pad, [0, 0])
            high = np.minimum(np.ceil(points.max(axis=0)).astype(int) + pad + 1, [width, height])
            local, local_ids, _ = self.edge.detectMarkers(image[low[1]:high[1], low[0]:high[0]])
            if local_ids is not None:
                for candidate_id, candidate in zip(local_ids.reshape(-1), local):
                    if int(candidate_id) == key:
                        observations[key] = np.asarray(candidate, np.float32) + low.astype(np.float32)
                        break
        if len(self.metric_ids & observations.keys()) < self.minimum_count:
            return self.edge.detectMarkers(image)
        ordered = sorted(observations)
        return [observations[key] for key in ordered], np.asarray(ordered, np.int32).reshape(-1, 1), rejected


class TrayBoardPoseEstimator:
    """Detect A-H and robustly estimate ``^C T_T`` for independent frames."""

    def __init__(
        self,
        geometry: Mapping[str, Any],
        intrinsics: CameraIntrinsics,
        quality: TrayPoseQualityConfig = DEFAULT_POSE_QUALITY,
        *,
        edge_refinement: bool = False,
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
        parameters.cornerRefinementMethod = (
            cv2.aruco.CORNER_REFINE_APRILTAG if edge_refinement
            else cv2.aruco.CORNER_REFINE_SUBPIX
        )
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
        if edge_refinement:
            self.detector = _EdgeRefinedDetector(self.dictionary, self.configured_ids,
                                                self.quality.minimum_visible_markers)

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

    def _fit_marker_consensus(self, observations: Mapping[int, np.ndarray]):
        """Fit complete-marker hypotheses, then validate the FINAL fit.

        A point-RANSAC fit can initially blame two of four markers even when
        three agree. Removing every blamed marker at once loses that solution.
        Search largest sets first, without blacklisting an ID across frames.
        Refinement is part of each hypothesis, never an unchecked last step.
        """
        ids = tuple(sorted(observations))
        fallback = None
        attempts = 0
        for count in range(len(ids), self.quality.minimum_inlier_markers - 1, -1):
            candidates = []
            for subset in combinations(ids, count):
                attempts += 1
                if attempts > 64:
                    return fallback, 'Marker consensus search budget exhausted'
                active = {key: observations[key] for key in subset}
                ok, rvec, tvec, seed_inliers = self._solve(active)
                if not ok or rvec is None or tvec is None:
                    continue
                rvec, tvec = self._refine_all_active(active, rvec, tvec)
                errors, rms = self._marker_errors(active, rvec, tvec)
                points = np.concatenate([self.object_corners_by_id[key] for key in subset])
                pixels = np.concatenate([active[key] for key in subset])
                projected, _ = cv2.projectPoints(points, rvec, tvec,
                    self.intrinsics.K, self.intrinsics.dist_coeffs)
                residuals = np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1)
                # Inlier diagnostics and the gate must describe this fit, not
                # the now-obsolete point-RANSAC seed from before LM refinement.
                final_inliers = np.flatnonzero(residuals <= self.quality.ransac_reprojection_threshold_px)
                inliers = np.intersect1d(seed_inliers, final_inliers)
                transform = make_transform_C_T(rvec, tvec)
                finite = bool(np.isfinite(rms) and np.all(np.isfinite(transform)))
                span = self._object_span(points)
                fit = (active, rvec, tvec, inliers, errors, rms, span)
                if fallback is None:
                    fallback = fit
                if not finite:
                    continue
                camera = invert_transform(transform)[:3, 3]
                if (
                    max(errors.values()) > self.quality.marker_outlier_threshold_px
                    or rms > self.quality.maximum_global_rms_px
                    or len(inliers) / len(points) < self.quality.minimum_ransac_inlier_corner_ratio
                    or span < self.quality.minimum_object_span_mm
                    or camera[2] < self.quality.minimum_camera_height_above_tray_mm
                    or np.min(transform_points(transform, points)[:, 2]) < self.quality.minimum_object_depth_C_mm
                ):
                    continue
                # Evaluate excluded markers too: a low-error small subset must
                # not win over a larger valid consensus, and ties use evidence
                # from ALL detections rather than just the convenient subset.
                all_errors, _ = self._marker_errors(observations, rvec, tvec)
                score = sum(min(value, 2 * self.quality.marker_outlier_threshold_px) ** 2
                            for value in all_errors.values())
                candidates.append((score, rms, subset, fit))
            if candidates:
                candidates.sort(key=lambda row: row[:3])
                winner = candidates[0][3]
                # Equally supported but incompatible poses are not permission
                # to move. Compare their projection across the full board.
                check_points = np.concatenate(list(self.object_corners_by_id.values()))
                reference, _ = cv2.projectPoints(check_points, winner[1], winner[2],
                    self.intrinsics.K, self.intrinsics.dist_coeffs)
                for candidate in candidates[1:]:
                    alternative = candidate[3]
                    pixels, _ = cv2.projectPoints(check_points, alternative[1], alternative[2],
                        self.intrinsics.K, self.intrinsics.dist_coeffs)
                    if float(np.max(np.linalg.norm(pixels - reference, axis=2))) > self.quality.marker_outlier_threshold_px:
                        return fallback, 'Marker consensus ambiguous: equally supported poses disagree'
                return winner, None
        return fallback, None

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
        decoded_corners: dict[int, np.ndarray] = {}
        display_corners: list[np.ndarray] = []
        display_ids: list[int] = []
        for corners, marker_id in zip(marker_corners, all_detected_ids):
            candidate = np.asarray(corners, dtype=np.float64).reshape(4, 2)
            previous = decoded_corners.get(marker_id)
            if previous is None or abs(cv2.contourArea(candidate.astype(np.float32))) > abs(cv2.contourArea(previous.astype(np.float32))):
                decoded_corners[marker_id] = candidate.copy()
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

        fitted, consensus_error = self._fit_marker_consensus(observations)
        if fitted is None:
            return self._empty_result(image, visible_ids, "最终PnP没有产生位姿")
        active, rvec, tvec, ransac_inliers, per_marker, global_rms, object_span = fitted
        used_ids = tuple(sorted(active))
        rejected_ids = set(observations) - set(active)
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
        if consensus_error or not np.isfinite(global_rms) or not np.all(np.isfinite(transform_C_T)):
            quality_passed = False
            failure_reason = consensus_error or 'non-finite final pose'
        elif object_span < self.quality.minimum_object_span_mm:
            quality_passed = False
            failure_reason = f"有效Board跨度不足：{object_span:.1f}mm"
        elif len(used_ids) < self.quality.minimum_inlier_markers:
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
            detected_corners_px=decoded_corners,
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
