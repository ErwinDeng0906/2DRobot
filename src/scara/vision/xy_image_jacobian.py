"""Robust local Jacobian calibration for camera-1 image error.

The calibrated model is deliberately local and task-specific::

    delta_error_px = J_local @ delta_command_xy_mm

The image error is ``slot_pixel - suction_target_pixel``.  A correction that
would cancel a current error is therefore ``-inv(J_local) @ error_px``.  This
module only estimates and evaluates that mapping; it never talks to hardware.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class XYImageJacobianQualityConfig:
    minimum_frames_per_offset: int = 5
    minimum_distinct_offsets: int = 7
    robust_sigma_multiplier: float = 3.0
    residual_floor_px: float = 0.75
    maximum_fit_rms_px: float = 1.5
    maximum_cross_validation_rms_px: float = 2.5
    maximum_condition_number: float = 25.0
    minimum_singular_value_px_per_mm: float = 0.25


DEFAULT_XY_IMAGE_JACOBIAN_QUALITY = XYImageJacobianQualityConfig()

REQUIRED_XY_JACOBIAN_QUALITY_GATES = frozenset(
    {
        "minimum_offset_count",
        "fit_rms",
        "cross_validation_rms",
        "condition_number",
        "minimum_singular_value",
        "invertible",
    }
)


def _finite_pair(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain two finite values")
    return array


def _fit_affine(
    offsets_xy_mm: np.ndarray,
    errors_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = np.column_stack(
        [offsets_xy_mm, np.ones(len(offsets_xy_mm), dtype=np.float64)]
    )
    # The affine model has two slopes plus one intercept.  Raw offsets can have
    # rank 2 even when every sample lies on an affine line that does not pass
    # through the origin; require the complete design to have rank 3.
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("XY command offsets do not span two independent axes")
    coefficients, _residuals, _rank, _singular = np.linalg.lstsq(
        design,
        errors_px,
        rcond=None,
    )
    jacobian = coefficients[:2, :].T
    intercept = coefficients[2, :]
    predicted = design @ coefficients
    return jacobian, intercept, predicted


def _aggregate_samples(
    samples: Iterable[Mapping[str, Any]],
    quality: XYImageJacobianQualityConfig,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], list[np.ndarray]] = {}
    raw_counts: dict[tuple[float, float], int] = {}
    for index, sample in enumerate(samples, start=1):
        if not bool(sample.get("accepted", True)):
            continue
        offset = _finite_pair(sample.get("command_offset_xy_mm"), f"sample {index} offset")
        error = _finite_pair(sample.get("image_error_px"), f"sample {index} error")
        key = (round(float(offset[0]), 6), round(float(offset[1]), 6))
        grouped.setdefault(key, []).append(error)
        raw_counts[key] = raw_counts.get(key, 0) + 1

    aggregates: list[dict[str, Any]] = []
    for key in sorted(grouped):
        values = np.asarray(grouped[key], dtype=np.float64)
        frame_count = int(len(values))
        median = np.median(values, axis=0)
        deviations = np.linalg.norm(values - median, axis=1)
        deviation_median = float(np.median(deviations))
        mad = float(np.median(np.abs(deviations - deviation_median)))
        threshold = max(
            quality.residual_floor_px,
            deviation_median
            + quality.robust_sigma_multiplier * 1.4826 * mad,
        )
        keep = deviations <= threshold + 1e-12
        kept_values = values[keep]
        usable = frame_count >= quality.minimum_frames_per_offset and len(kept_values) >= (
            quality.minimum_frames_per_offset
        )
        aggregates.append(
            {
                "command_offset_xy_mm": [float(key[0]), float(key[1])],
                "frame_count": frame_count,
                "used_frame_count": int(len(kept_values)) if usable else 0,
                "usable": bool(usable),
                "median_image_error_px": (
                    np.median(kept_values, axis=0).astype(float).tolist()
                    if usable
                    else None
                ),
                "within_offset_residual_gate_px": float(threshold),
                "within_offset_rms_px": (
                    float(np.sqrt(np.mean(np.square(deviations[keep]))))
                    if usable and len(kept_values)
                    else None
                ),
                "raw_frame_count": int(raw_counts[key]),
            }
        )
    return aggregates


def fit_local_xy_image_jacobian(
    samples: Iterable[Mapping[str, Any]],
    quality: XYImageJacobianQualityConfig = DEFAULT_XY_IMAGE_JACOBIAN_QUALITY,
) -> dict[str, Any]:
    """Fit a robust affine local Jacobian and return a JSON-ready report."""

    aggregates = _aggregate_samples(samples, quality)
    usable = [row for row in aggregates if row["usable"]]
    failure_reasons: list[str] = []
    if len(usable) < quality.minimum_distinct_offsets:
        failure_reasons.append(
            f"usable offsets {len(usable)}/{quality.minimum_distinct_offsets}"
        )

    if usable:
        offsets = np.asarray(
            [row["command_offset_xy_mm"] for row in usable], dtype=np.float64
        )
        errors = np.asarray(
            [row["median_image_error_px"] for row in usable], dtype=np.float64
        )
    else:
        offsets = np.empty((0, 2), dtype=np.float64)
        errors = np.empty((0, 2), dtype=np.float64)

    if (
        len(usable) >= 3
        and np.linalg.matrix_rank(offsets - np.mean(offsets, axis=0)) < 2
    ):
        failure_reasons.append("command offsets do not span both X and Y")

    result: dict[str, Any] = {
        "status": "failure",
        "failure_reasons": failure_reasons,
        "quality_configuration": asdict(quality),
        "offset_aggregates": aggregates,
        "used_offset_count": 0,
        "rejected_offsets_xy_mm": [],
        "j_error_px_per_command_mm": None,
        "j_command_mm_per_error_px": None,
        "intercept_error_px": None,
        "fit_rms_px": None,
        "fit_max_px": None,
        "determinant_px2_per_mm2": None,
        "condition_number": None,
        "singular_values_px_per_mm": None,
        "cross_validation": None,
        "quality_gates": {},
    }
    if failure_reasons:
        return result

    keep = np.ones(len(offsets), dtype=bool)
    for _iteration in range(5):
        jacobian, intercept, predicted = _fit_affine(offsets[keep], errors[keep])
        all_predicted = offsets @ jacobian.T + intercept
        residual_norms = np.linalg.norm(errors - all_predicted, axis=1)
        active = residual_norms[keep]
        median = float(np.median(active))
        mad = float(np.median(np.abs(active - median)))
        threshold = max(
            quality.residual_floor_px,
            median + quality.robust_sigma_multiplier * 1.4826 * mad,
        )
        candidate = residual_norms <= threshold + 1e-12
        if np.array_equal(candidate, keep):
            break
        if (
            int(np.count_nonzero(candidate)) < quality.minimum_distinct_offsets
            or np.linalg.matrix_rank(
                offsets[candidate] - np.mean(offsets[candidate], axis=0)
            )
            < 2
        ):
            break
        keep = candidate

    jacobian, intercept, predicted_used = _fit_affine(offsets[keep], errors[keep])
    residual_vectors = errors[keep] - predicted_used
    residual_norms = np.linalg.norm(residual_vectors, axis=1)
    fit_rms = float(np.sqrt(np.mean(np.square(residual_norms))))
    fit_max = float(np.max(residual_norms))
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    condition = float(np.linalg.cond(jacobian))
    determinant = float(np.linalg.det(jacobian))
    inverse: Optional[np.ndarray]
    try:
        inverse = np.linalg.inv(jacobian)
    except np.linalg.LinAlgError:
        inverse = None

    cv_rows: list[dict[str, Any]] = []
    used_indices = np.flatnonzero(keep)
    for held_out in used_indices:
        train = keep.copy()
        train[held_out] = False
        if (
            np.count_nonzero(train) < 3
            or np.linalg.matrix_rank(
                offsets[train] - np.mean(offsets[train], axis=0)
            )
            < 2
        ):
            continue
        cv_j, cv_b, _ = _fit_affine(offsets[train], errors[train])
        prediction = cv_j @ offsets[held_out] + cv_b
        error_vector = errors[held_out] - prediction
        cv_rows.append(
            {
                "held_out_offset_xy_mm": offsets[held_out].astype(float).tolist(),
                "observed_error_px": errors[held_out].astype(float).tolist(),
                "predicted_error_px": prediction.astype(float).tolist(),
                "prediction_error_px": error_vector.astype(float).tolist(),
                "prediction_error_norm_px": float(np.linalg.norm(error_vector)),
            }
        )
    cv_rms = (
        float(
            math.sqrt(
                sum(row["prediction_error_norm_px"] ** 2 for row in cv_rows)
                / len(cv_rows)
            )
        )
        if cv_rows
        else math.inf
    )

    gates = {
        "minimum_offset_count": {
            "value": int(np.count_nonzero(keep)),
            "minimum": quality.minimum_distinct_offsets,
            "passed": int(np.count_nonzero(keep)) >= quality.minimum_distinct_offsets,
        },
        "fit_rms": {
            "value_px": fit_rms,
            "maximum_px": quality.maximum_fit_rms_px,
            "passed": fit_rms <= quality.maximum_fit_rms_px,
        },
        "cross_validation_rms": {
            "value_px": cv_rms,
            "maximum_px": quality.maximum_cross_validation_rms_px,
            "passed": cv_rms <= quality.maximum_cross_validation_rms_px,
        },
        "condition_number": {
            "value": condition,
            "maximum": quality.maximum_condition_number,
            "passed": condition <= quality.maximum_condition_number,
        },
        "minimum_singular_value": {
            "value_px_per_mm": float(np.min(singular_values)),
            "minimum_px_per_mm": quality.minimum_singular_value_px_per_mm,
            "passed": float(np.min(singular_values))
            >= quality.minimum_singular_value_px_per_mm,
        },
        "invertible": {"passed": inverse is not None},
    }
    failures = [name for name, gate in gates.items() if not gate["passed"]]
    result.update(
        {
            "status": "success" if not failures else "failure",
            "failure_reasons": failures,
            "used_offset_count": int(np.count_nonzero(keep)),
            "rejected_offsets_xy_mm": offsets[~keep].astype(float).tolist(),
            "j_error_px_per_command_mm": jacobian.astype(float).tolist(),
            "j_command_mm_per_error_px": (
                None if inverse is None else inverse.astype(float).tolist()
            ),
            "intercept_error_px": intercept.astype(float).tolist(),
            "fit_rms_px": fit_rms,
            "fit_max_px": fit_max,
            "determinant_px2_per_mm2": determinant,
            "condition_number": condition,
            "singular_values_px_per_mm": singular_values.astype(float).tolist(),
            "cross_validation": {"rows": cv_rows, "rms_px": cv_rms},
            "quality_gates": gates,
        }
    )
    return result


def correction_command_xy_mm(
    image_error_px: Any,
    calibration: Mapping[str, Any],
) -> Optional[tuple[float, float]]:
    """Return ``-J^-1 error`` for a successful calibration, else ``None``."""

    if calibration.get("status") != "success":
        return None
    gates = calibration.get("quality_gates")
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
    inverse = calibration.get("j_command_mm_per_error_px")
    if inverse is None:
        return None
    matrix = np.asarray(inverse, dtype=np.float64).reshape(2, 2)
    error = _finite_pair(image_error_px, "image_error_px")
    if not np.all(np.isfinite(matrix)):
        return None
    correction = -(matrix @ error)
    return float(correction[0]), float(correction[1])


__all__ = [
    "DEFAULT_XY_IMAGE_JACOBIAN_QUALITY",
    "REQUIRED_XY_JACOBIAN_QUALITY_GATES",
    "XYImageJacobianQualityConfig",
    "correction_command_xy_mm",
    "fit_local_xy_image_jacobian",
]
