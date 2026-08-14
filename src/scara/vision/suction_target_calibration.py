"""Stage 4 fixed-plane camera-to-suction calibration (numerical core).

Stage 3 remains the sole owner of A-H detection and ``^C T_T`` estimation.
This module consumes only quality-passed Stage-3 transforms.  At taught slot
``Q_i`` the suction/J4 axis is known to intersect the declared working plane,
therefore one camera-frame candidate is obtained by applying the existing
Stage-3 transform to the known Tray point.  Repeated frames are robustly
aggregated on SE(3), then the constant camera-frame suction target is fitted
across locations.  No robot, Qt, camera, or filesystem access occurs here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .tray_pose_estimator import invert_transform, transform_points


@dataclass(frozen=True)
class SuctionCalibrationQualityConfig:
    """Fail-closed quality limits for one Task-8 run."""

    frames_per_location: int = 20
    minimum_accepted_frames_per_location: int = 12
    within_location_translation_floor_mm: float = 1.0
    within_location_rotation_floor_deg: float = 1.0
    robust_sigma_multiplier: float = 3.0
    minimum_fit_locations: int = 6
    location_outlier_floor_mm: float = 1.5
    maximum_fit_xy_rms_mm: float = 1.0
    maximum_fit_3d_rms_mm: float = 1.5
    maximum_cross_validation_xy_rms_mm: float = 1.5
    maximum_cross_validation_xy_mm: float = 3.0
    maximum_target_pixel_rms_px: float = 3.0

    def __post_init__(self) -> None:
        if self.frames_per_location < 1:
            raise ValueError("frames_per_location must be positive")
        if not 1 <= self.minimum_accepted_frames_per_location <= self.frames_per_location:
            raise ValueError("minimum accepted frames must be within the burst size")
        if self.minimum_fit_locations < 3:
            raise ValueError("minimum_fit_locations must be at least three")
        positive = (
            self.within_location_translation_floor_mm,
            self.within_location_rotation_floor_deg,
            self.robust_sigma_multiplier,
            self.location_outlier_floor_mm,
            self.maximum_fit_xy_rms_mm,
            self.maximum_fit_3d_rms_mm,
            self.maximum_cross_validation_xy_rms_mm,
            self.maximum_cross_validation_xy_mm,
            self.maximum_target_pixel_rms_px,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("all suction calibration quality limits must be positive")


DEFAULT_SUCTION_QUALITY = SuctionCalibrationQualityConfig()


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _project_to_so3(matrix: np.ndarray) -> np.ndarray:
    u, _singular, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def so3_geodesic_mean(rotations: Sequence[np.ndarray]) -> np.ndarray:
    """Return the iterative geodesic mean of valid rotation matrices."""

    if not rotations:
        raise ValueError("at least one rotation is required")
    current = _project_to_so3(
        np.mean(
            np.stack(
                [np.asarray(rotation, dtype=np.float64).reshape(3, 3) for rotation in rotations]
            ),
            axis=0,
        )
    )
    for _iteration in range(50):
        increments = []
        for rotation in rotations:
            relative = current.T @ np.asarray(rotation, dtype=np.float64).reshape(3, 3)
            rvec, _ = cv2.Rodrigues(relative)
            increments.append(rvec.reshape(3))
        mean_increment = np.mean(np.stack(increments), axis=0)
        if float(np.linalg.norm(mean_increment)) < 1e-12:
            break
        update, _ = cv2.Rodrigues(mean_increment.reshape(3, 1))
        current = _project_to_so3(current @ update)
    return current


def _robust_threshold(
    values: np.ndarray,
    floor: float,
    sigma_multiplier: float,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    median = float(np.median(values)) if len(values) else 0.0
    mad = float(np.median(np.abs(values - median))) if len(values) else 0.0
    robust_sigma = 1.4826 * mad
    threshold = max(float(floor), median + sigma_multiplier * robust_sigma)
    return threshold, median, mad


def _rms(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return math.sqrt(float(np.mean(array * array))) if len(array) else math.nan


def aggregate_location_poses(
    target_name: str,
    point_T_mm: Sequence[float],
    frame_transforms_C_T: Sequence[np.ndarray],
    frame_indices: Sequence[int],
    quality: SuctionCalibrationQualityConfig = DEFAULT_SUCTION_QUALITY,
) -> dict[str, Any]:
    """Robustly combine one stationary burst into one stable ``^C T_T``.

    Translation starts from a component-wise median.  Rotation starts from an
    SO(3) geodesic mean.  Translation/rotation residual MAD gates remove burst
    outliers, then the retained transforms are recombined.  This batch result
    is independent from the online tracker output used to accept each frame.
    """

    point = np.asarray(point_T_mm, dtype=np.float64).reshape(3)
    transforms = [np.asarray(value, dtype=np.float64).reshape(4, 4) for value in frame_transforms_C_T]
    indices = [int(value) for value in frame_indices]
    if len(transforms) != len(indices):
        raise ValueError("frame transforms and indices must have equal length")
    result: dict[str, Any] = {
        "target_name": str(target_name),
        "known_point_T_mm": point.astype(float).tolist(),
        "accepted_before_batch_filter": len(transforms),
        "required_accepted_frames": quality.minimum_accepted_frames_per_location,
        "success": False,
        "failure_reason": None,
        "used_frame_indices": [],
        "batch_rejected_frame_indices": [],
        "stable_T_C_T": None,
        "stable_T_T_C": None,
        "suction_candidate_C_mm": None,
    }
    if len(transforms) < quality.minimum_accepted_frames_per_location:
        result["failure_reason"] = (
            f"only {len(transforms)}/{quality.minimum_accepted_frames_per_location} "
            "frames passed Stage-3 and temporal gates"
        )
        return result

    translations = np.stack([transform[:3, 3] for transform in transforms])
    rotations = [transform[:3, :3] for transform in transforms]
    initial_translation = np.median(translations, axis=0)
    initial_rotation = so3_geodesic_mean(rotations)
    translation_residuals = np.linalg.norm(
        translations - initial_translation.reshape(1, 3), axis=1
    )
    rotation_residuals = np.asarray(
        [
            _rotation_angle_deg(initial_rotation.T @ rotation)
            for rotation in rotations
        ],
        dtype=np.float64,
    )
    translation_threshold, translation_median, translation_mad = _robust_threshold(
        translation_residuals,
        quality.within_location_translation_floor_mm,
        quality.robust_sigma_multiplier,
    )
    rotation_threshold, rotation_median, rotation_mad = _robust_threshold(
        rotation_residuals,
        quality.within_location_rotation_floor_deg,
        quality.robust_sigma_multiplier,
    )
    mask = (translation_residuals <= translation_threshold) & (
        rotation_residuals <= rotation_threshold
    )
    used_positions = np.flatnonzero(mask)
    rejected_positions = np.flatnonzero(~mask)
    result.update(
        {
            "translation_residual_gate": {
                "threshold_mm": translation_threshold,
                "median_mm": translation_median,
                "mad_mm": translation_mad,
            },
            "rotation_residual_gate": {
                "threshold_deg": rotation_threshold,
                "median_deg": rotation_median,
                "mad_deg": rotation_mad,
            },
            "used_frame_indices": [indices[index] for index in used_positions],
            "batch_rejected_frame_indices": [
                indices[index] for index in rejected_positions
            ],
        }
    )
    if len(used_positions) < quality.minimum_accepted_frames_per_location:
        result["failure_reason"] = (
            f"only {len(used_positions)} frames remain after within-location filtering"
        )
        return result

    used_translations = translations[used_positions]
    used_rotations = [rotations[index] for index in used_positions]
    stable = np.eye(4, dtype=np.float64)
    stable[:3, :3] = so3_geodesic_mean(used_rotations)
    stable[:3, 3] = np.median(used_translations, axis=0)
    final_translation_residuals = np.linalg.norm(
        used_translations - stable[:3, 3].reshape(1, 3), axis=1
    )
    final_rotation_residuals = [
        _rotation_angle_deg(stable[:3, :3].T @ rotation)
        for rotation in used_rotations
    ]
    suction_candidate = transform_points(stable, point.reshape(1, 3))[0]
    result.update(
        {
            "success": True,
            "stable_T_C_T": stable.astype(float).tolist(),
            "stable_T_T_C": invert_transform(stable).astype(float).tolist(),
            "translation_rms_mm": _rms(final_translation_residuals),
            "translation_max_mm": float(np.max(final_translation_residuals)),
            "rotation_rms_deg": _rms(final_rotation_residuals),
            "rotation_max_deg": float(np.max(final_rotation_residuals)),
            "suction_candidate_C_mm": suction_candidate.astype(float).tolist(),
        }
    )
    return result


def project_camera_point(
    point_C_mm: Sequence[float],
    K: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """Project one already camera-frame point into the raw distorted image."""

    point = np.asarray(point_C_mm, dtype=np.float64).reshape(1, 3)
    if point[0, 2] <= 0.0:
        raise ValueError("camera-frame suction point must have positive depth")
    projected, _ = cv2.projectPoints(
        point,
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        np.asarray(K, dtype=np.float64).reshape(3, 3),
        np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1),
    )
    return projected.reshape(2)


def fit_suction_target(
    location_aggregates: Sequence[Mapping[str, Any]],
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    quality: SuctionCalibrationQualityConfig = DEFAULT_SUCTION_QUALITY,
) -> dict[str, Any]:
    """Fit constant ``^C p_S`` and perform location-wise leave-one-out CV."""

    usable = [row for row in location_aggregates if bool(row.get("success"))]
    report: dict[str, Any] = {
        "status": "rejected_quality",
        "failure_reasons": [],
        "location_count": len(location_aggregates),
        "usable_location_count": len(usable),
        "used_location_names": [],
        "rejected_location_names": [],
        "p_C_S_mm": None,
        "target_pixel_distorted_px": None,
    }
    if len(usable) < quality.minimum_fit_locations:
        report["failure_reasons"].append(
            f"only {len(usable)}/{quality.minimum_fit_locations} locations are stable"
        )
        return report

    names = [str(row["target_name"]) for row in usable]
    candidates = np.asarray(
        [row["suction_candidate_C_mm"] for row in usable], dtype=np.float64
    ).reshape(-1, 3)
    initial = np.median(candidates, axis=0)
    location_residuals = np.linalg.norm(candidates - initial.reshape(1, 3), axis=1)
    location_threshold, location_median, location_mad = _robust_threshold(
        location_residuals,
        quality.location_outlier_floor_mm,
        quality.robust_sigma_multiplier,
    )
    inlier_mask = location_residuals <= location_threshold
    if int(np.count_nonzero(inlier_mask)) < quality.minimum_fit_locations:
        # Keep the nearest locations instead of allowing a noisy MAD estimate
        # to make the solve singular; the quality residuals still fail closed.
        nearest = np.argsort(location_residuals)[: quality.minimum_fit_locations]
        inlier_mask = np.zeros(len(candidates), dtype=bool)
        inlier_mask[nearest] = True
    used_candidates = candidates[inlier_mask]
    fitted = np.mean(used_candidates, axis=0)
    fit_delta = used_candidates - fitted.reshape(1, 3)
    fit_xy = np.linalg.norm(fit_delta[:, :2], axis=1)
    fit_3d = np.linalg.norm(fit_delta, axis=1)
    target_pixel = project_camera_point(fitted, K, dist_coeffs)
    candidate_pixels = np.stack(
        [project_camera_point(point, K, dist_coeffs) for point in used_candidates]
    )
    pixel_residuals = np.linalg.norm(
        candidate_pixels - target_pixel.reshape(1, 2), axis=1
    )

    # Each validation row is predicted from every *other* stable location.
    # The held-out location never contributes to its own predicted target.
    cross_validation = []
    for index, (name, observed) in enumerate(zip(names, candidates)):
        other = np.delete(candidates, index, axis=0)
        predicted = np.median(other, axis=0)
        delta = observed - predicted
        observed_pixel = project_camera_point(observed, K, dist_coeffs)
        predicted_pixel = project_camera_point(predicted, K, dist_coeffs)
        cross_validation.append(
            {
                "held_out_location": name,
                "predicted_p_C_S_mm": predicted.astype(float).tolist(),
                "observed_p_C_S_mm": observed.astype(float).tolist(),
                "error_xyz_mm": delta.astype(float).tolist(),
                "error_xy_mm": float(np.linalg.norm(delta[:2])),
                "error_3d_mm": float(np.linalg.norm(delta)),
                "error_px": float(np.linalg.norm(observed_pixel - predicted_pixel)),
            }
        )
    cv_xy = [row["error_xy_mm"] for row in cross_validation]
    cv_3d = [row["error_3d_mm"] for row in cross_validation]
    cv_px = [row["error_px"] for row in cross_validation]

    fit_xy_rms = _rms(fit_xy)
    fit_3d_rms = _rms(fit_3d)
    pixel_rms = _rms(pixel_residuals)
    cv_xy_rms = _rms(cv_xy)
    cv_xy_max = float(np.max(cv_xy))
    gates = {
        "fit_xy_rms": {
            "value_mm": fit_xy_rms,
            "maximum_mm": quality.maximum_fit_xy_rms_mm,
            "passed": fit_xy_rms <= quality.maximum_fit_xy_rms_mm,
        },
        "fit_3d_rms": {
            "value_mm": fit_3d_rms,
            "maximum_mm": quality.maximum_fit_3d_rms_mm,
            "passed": fit_3d_rms <= quality.maximum_fit_3d_rms_mm,
        },
        "target_pixel_rms": {
            "value_px": pixel_rms,
            "maximum_px": quality.maximum_target_pixel_rms_px,
            "passed": pixel_rms <= quality.maximum_target_pixel_rms_px,
        },
        "cross_validation_xy_rms": {
            "value_mm": cv_xy_rms,
            "maximum_mm": quality.maximum_cross_validation_xy_rms_mm,
            "passed": cv_xy_rms <= quality.maximum_cross_validation_xy_rms_mm,
        },
        "cross_validation_xy_max": {
            "value_mm": cv_xy_max,
            "maximum_mm": quality.maximum_cross_validation_xy_mm,
            "passed": cv_xy_max <= quality.maximum_cross_validation_xy_mm,
        },
    }
    failures = [name for name, gate in gates.items() if not gate["passed"]]
    report.update(
        {
            "status": "success" if not failures else "rejected_quality",
            "failure_reasons": failures,
            "location_outlier_gate": {
                "threshold_mm": location_threshold,
                "median_mm": location_median,
                "mad_mm": location_mad,
            },
            "used_location_names": [
                name for name, keep in zip(names, inlier_mask) if keep
            ],
            "rejected_location_names": [
                name for name, keep in zip(names, inlier_mask) if not keep
            ],
            "p_C_S_mm": fitted.astype(float).tolist(),
            "target_pixel_distorted_px": target_pixel.astype(float).tolist(),
            "fit_residuals": [
                {
                    "target_name": name,
                    "candidate_p_C_S_mm": point.astype(float).tolist(),
                    "used": bool(keep),
                    "residual_to_initial_mm": float(residual),
                }
                for name, point, keep, residual in zip(
                    names, candidates, inlier_mask, location_residuals
                )
            ],
            "fit_xy_rms_mm": fit_xy_rms,
            "fit_3d_rms_mm": fit_3d_rms,
            "target_pixel_rms_px": pixel_rms,
            "cross_validation": {
                "method": "leave-one-location-out; held-out location excluded from its predictor",
                "rows": cross_validation,
                "xy_rms_mm": cv_xy_rms,
                "xy_max_mm": cv_xy_max,
                "three_d_rms_mm": _rms(cv_3d),
                "pixel_rms_px": _rms(cv_px),
            },
            "quality_gates": gates,
        }
    )
    return report


__all__ = [
    "DEFAULT_SUCTION_QUALITY",
    "SuctionCalibrationQualityConfig",
    "aggregate_location_poses",
    "fit_suction_target",
    "project_camera_point",
    "so3_geodesic_mean",
]
