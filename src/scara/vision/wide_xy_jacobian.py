"""Wide-area P22 image-error model for Stage 7B.

This module is deliberately controller-free.  Task11 supplies repeated
camera-1 measurements at a 5 x 5 training grid and a shifted 4 x 4 validation
grid.  The fitter compares a single global affine Jacobian with a quadratic
error surface and installs the simplest model that passes independent
validation.  At run time the selected surface provides a position-dependent
Jacobian for coarse visual servoing::

    e(q) = [u_slot - u_suction, v_slot - v_suction]
    delta_q = -gain * inverse(J(q)) @ e(q)
    J(q) = partial e / partial q

The existing Task9 Jacobian remains the fine model inside its measured
``+/-2 mm`` neighbourhood.  This file never sends a robot command.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np


GLOBAL_FEATURES = ("1", "x", "y")
QUADRATIC_FEATURES = ("1", "x", "y", "x2", "xy", "y2")


@dataclass(frozen=True)
class WideXYJacobianQualityConfig:
    minimum_frames_per_visit: int = 3
    required_training_nodes: int = 25
    required_validation_nodes: int = 16
    robust_sigma_multiplier: float = 3.0
    within_visit_floor_px: float = 0.75
    # Task11 pass-to-pass repeatability is evaluated at the same physical
    # node.  The operational acceptance limit is 1.50 px; all fit and
    # independent-validation limits below remain unchanged.
    maximum_node_repeatability_px: float = 1.50
    maximum_training_rms_px: float = 0.75
    maximum_validation_rms_px: float = 0.75
    maximum_validation_error_px: float = 1.50
    maximum_condition_number: float = 10.0
    minimum_singular_value_px_per_mm: float = 0.25
    maximum_validation_jacobian_relative_error: float = 0.25
    minimum_predicted_improvement_ratio: float = 0.05
    global_model_simplicity_allowance_px: float = 0.10


DEFAULT_WIDE_XY_JACOBIAN_QUALITY = WideXYJacobianQualityConfig()

REQUIRED_WIDE_XY_JACOBIAN_QUALITY_GATES = frozenset(
    {
        "training_node_coverage",
        "validation_node_coverage",
        "visit_frame_coverage",
        "node_repeatability",
        "selected_training_rms",
        "independent_validation_rms",
        "independent_validation_max",
        "jacobian_condition_number",
        "minimum_singular_value",
        "validation_jacobian_consistency",
        "predicted_descent",
        "invertible",
    }
)


def _pair(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain two finite values")
    return array


def _offset_key(value: Any) -> tuple[float, float]:
    pair = _pair(value, "offset")
    return round(float(pair[0]), 6), round(float(pair[1]), 6)


def _normalise_expected(
    values: Sequence[Sequence[float]], label: str
) -> set[tuple[float, float]]:
    result = {_offset_key(value) for value in values}
    if not result:
        raise ValueError(f"{label} cannot be empty")
    return result


def _aggregate_visits(
    samples: Iterable[Mapping[str, Any]],
    quality: WideXYJacobianQualityConfig,
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, int, float, float], dict[str, Any]
    ] = {}
    for index, sample in enumerate(samples, start=1):
        phase = str(sample.get("phase") or "").strip().lower()
        if phase not in {"train", "validation"}:
            raise ValueError(f"sample {index} phase must be train/validation")
        try:
            pass_index = int(sample.get("pass_index", 1))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"sample {index} pass_index is invalid") from exc
        if pass_index < 1:
            raise ValueError(f"sample {index} pass_index must be positive")
        offset = _offset_key(sample.get("command_offset_xy_mm"))
        group = grouped.setdefault(
            (phase, pass_index, *offset),
            {"frame_count": 0, "accepted_errors": []},
        )
        group["frame_count"] += 1
        if bool(sample.get("accepted", True)):
            error = _pair(sample.get("image_error_px"), f"sample {index} error")
            group["accepted_errors"].append(error)

    visits: list[dict[str, Any]] = []
    for (phase, pass_index, x_mm, y_mm), group in sorted(grouped.items()):
        values_list = group["accepted_errors"]
        values = np.asarray(values_list, dtype=np.float64)
        if len(values):
            median = np.median(values, axis=0)
            deviations = np.linalg.norm(values - median, axis=1)
            deviation_median = float(np.median(deviations))
            mad = float(np.median(np.abs(deviations - deviation_median)))
            threshold = max(
                quality.within_visit_floor_px,
                deviation_median
                + quality.robust_sigma_multiplier * 1.4826 * mad,
            )
            keep = deviations <= threshold + 1e-12
            used = values[keep]
        else:
            deviations = np.asarray([], dtype=np.float64)
            threshold = quality.within_visit_floor_px
            keep = np.asarray([], dtype=bool)
            used = np.empty((0, 2), dtype=np.float64)
        usable = len(used) >= quality.minimum_frames_per_visit
        visits.append(
            {
                "phase": phase,
                "pass_index": int(pass_index),
                "command_offset_xy_mm": [float(x_mm), float(y_mm)],
                "frame_count": int(group["frame_count"]),
                "accepted_frame_count": int(len(values)),
                "used_frame_count": int(len(used)) if usable else 0,
                "usable": bool(usable),
                "median_image_error_px": (
                    np.median(used, axis=0).astype(float).tolist()
                    if usable
                    else None
                ),
                "within_visit_gate_px": float(threshold),
                "within_visit_rms_px": (
                    float(np.sqrt(np.mean(np.square(deviations[keep]))))
                    if usable and len(used)
                    else None
                ),
            }
        )
    return visits


def _aggregate_nodes(
    visits: Sequence[Mapping[str, Any]], phase: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], list[Mapping[str, Any]]] = {}
    for visit in visits:
        if visit.get("phase") != phase or not bool(visit.get("usable")):
            continue
        grouped.setdefault(_offset_key(visit["command_offset_xy_mm"]), []).append(visit)

    nodes: list[dict[str, Any]] = []
    for key in sorted(grouped):
        node_visits = grouped[key]
        medians = np.asarray(
            [visit["median_image_error_px"] for visit in node_visits],
            dtype=np.float64,
        )
        node_median = np.median(medians, axis=0)
        repeatability = float(
            np.max(np.linalg.norm(medians - node_median, axis=1))
        )
        nodes.append(
            {
                "command_offset_xy_mm": [float(key[0]), float(key[1])],
                "visit_count": int(len(node_visits)),
                "pass_indices": sorted(
                    {int(visit["pass_index"]) for visit in node_visits}
                ),
                "median_image_error_px": node_median.astype(float).tolist(),
                "repeatability_max_px": repeatability,
            }
        )
    return nodes


def _design(offsets: np.ndarray, model_type: str) -> np.ndarray:
    x = offsets[:, 0]
    y = offsets[:, 1]
    if model_type == "global_affine":
        return np.column_stack([np.ones(len(offsets)), x, y])
    if model_type == "quadratic":
        return np.column_stack([np.ones(len(offsets)), x, y, x * x, x * y, y * y])
    raise ValueError(f"unknown wide model type {model_type}")


def _fit_model(
    train_offsets: np.ndarray,
    train_errors: np.ndarray,
    validation_offsets: np.ndarray,
    validation_errors: np.ndarray,
    model_type: str,
) -> dict[str, Any]:
    train_design = _design(train_offsets, model_type)
    if np.linalg.matrix_rank(train_design) < train_design.shape[1]:
        raise ValueError(f"{model_type} design matrix is rank deficient")
    coefficients, _residuals, _rank, _singular = np.linalg.lstsq(
        train_design, train_errors, rcond=None
    )
    train_prediction = train_design @ coefficients
    validation_prediction = _design(validation_offsets, model_type) @ coefficients
    train_vectors = train_errors - train_prediction
    validation_vectors = validation_errors - validation_prediction
    train_norms = np.linalg.norm(train_vectors, axis=1)
    validation_norms = np.linalg.norm(validation_vectors, axis=1)
    return {
        "model_type": model_type,
        "feature_order": list(
            GLOBAL_FEATURES if model_type == "global_affine" else QUADRATIC_FEATURES
        ),
        # One row per feature and one column per image-error component.
        "coefficients_feature_by_error": coefficients.astype(float).tolist(),
        "training_rms_px": float(np.sqrt(np.mean(np.square(train_norms)))),
        "training_max_px": float(np.max(train_norms)),
        "validation_rms_px": float(
            np.sqrt(np.mean(np.square(validation_norms)))
        ),
        "validation_max_px": float(np.max(validation_norms)),
        "training_rows": [
            {
                "offset_xy_mm": train_offsets[index].astype(float).tolist(),
                "observed_error_px": train_errors[index].astype(float).tolist(),
                "predicted_error_px": train_prediction[index].astype(float).tolist(),
                "residual_error_px": train_vectors[index].astype(float).tolist(),
                "residual_norm_px": float(train_norms[index]),
            }
            for index in range(len(train_offsets))
        ],
        "validation_rows": [
            {
                "offset_xy_mm": validation_offsets[index].astype(float).tolist(),
                "observed_error_px": validation_errors[index].astype(float).tolist(),
                "predicted_error_px": validation_prediction[index].astype(float).tolist(),
                "residual_error_px": validation_vectors[index].astype(float).tolist(),
                "residual_norm_px": float(validation_norms[index]),
            }
            for index in range(len(validation_offsets))
        ],
    }


def evaluate_wide_error_and_jacobian(
    model: Mapping[str, Any], offset_xy_mm: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate ``e(q)`` and ``J(q)`` for a fitted wide model."""

    offset = _pair(offset_xy_mm, "offset_xy_mm")
    model_type = str(model.get("model_type") or "")
    coefficients = np.asarray(
        model.get("coefficients_feature_by_error"), dtype=np.float64
    )
    if model_type == "global_affine":
        if coefficients.shape != (3, 2) or not np.all(np.isfinite(coefficients)):
            raise ValueError("global affine coefficients must have shape 3x2")
        feature = np.array([1.0, offset[0], offset[1]], dtype=np.float64)
        prediction = feature @ coefficients
        jacobian = np.array(
            [
                [coefficients[1, 0], coefficients[2, 0]],
                [coefficients[1, 1], coefficients[2, 1]],
            ],
            dtype=np.float64,
        )
    elif model_type == "quadratic":
        if coefficients.shape != (6, 2) or not np.all(np.isfinite(coefficients)):
            raise ValueError("quadratic coefficients must have shape 6x2")
        x_mm, y_mm = float(offset[0]), float(offset[1])
        feature = np.array(
            [1.0, x_mm, y_mm, x_mm * x_mm, x_mm * y_mm, y_mm * y_mm],
            dtype=np.float64,
        )
        prediction = feature @ coefficients
        derivative_x = coefficients[1] + 2.0 * x_mm * coefficients[3] + y_mm * coefficients[4]
        derivative_y = coefficients[2] + x_mm * coefficients[4] + 2.0 * y_mm * coefficients[5]
        jacobian = np.column_stack([derivative_x, derivative_y])
    else:
        raise ValueError(f"unsupported wide model type {model_type!r}")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(jacobian)):
        raise ValueError("wide model produced non-finite values")
    return prediction.astype(np.float64), jacobian.astype(np.float64)


def _empirical_jacobians(nodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    table = {
        _offset_key(node["command_offset_xy_mm"]): _pair(
            node["median_image_error_px"], "validation error"
        )
        for node in nodes
    }
    xs = sorted({key[0] for key in table})
    ys = sorted({key[1] for key in table})
    if len(xs) < 2 or len(ys) < 2:
        return []

    def derivative(x: float, y: float, axis: int) -> Optional[np.ndarray]:
        values = xs if axis == 0 else ys
        coordinate = x if axis == 0 else y
        index = values.index(coordinate)
        lower = values[index - 1] if index > 0 else None
        upper = values[index + 1] if index + 1 < len(values) else None
        if lower is not None and upper is not None:
            low_key = (lower, y) if axis == 0 else (x, lower)
            high_key = (upper, y) if axis == 0 else (x, upper)
            if low_key in table and high_key in table:
                return (table[high_key] - table[low_key]) / (upper - lower)
        neighbour = upper if upper is not None else lower
        if neighbour is None:
            return None
        neighbour_key = (neighbour, y) if axis == 0 else (x, neighbour)
        centre_key = (x, y)
        if neighbour_key not in table or centre_key not in table:
            return None
        return (table[neighbour_key] - table[centre_key]) / (neighbour - coordinate)

    rows: list[dict[str, Any]] = []
    for x_mm, y_mm in sorted(table):
        dx = derivative(x_mm, y_mm, 0)
        dy = derivative(x_mm, y_mm, 1)
        if dx is None or dy is None:
            continue
        jacobian = np.column_stack([dx, dy])
        rows.append(
            {
                "offset_xy_mm": [float(x_mm), float(y_mm)],
                "empirical_jacobian_px_per_mm": jacobian.astype(float).tolist(),
            }
        )
    return rows


def _model_diagnostics(
    model: Mapping[str, Any],
    train_nodes: Sequence[Mapping[str, Any]],
    validation_nodes: Sequence[Mapping[str, Any]],
    quality: WideXYJacobianQualityConfig,
) -> dict[str, Any]:
    offsets = [
        node["command_offset_xy_mm"] for node in (*train_nodes, *validation_nodes)
    ]
    conditions: list[float] = []
    singular_minima: list[float] = []
    invertible = True
    descent_rows: list[dict[str, Any]] = []
    for offset in offsets:
        _predicted, jacobian = evaluate_wide_error_and_jacobian(model, offset)
        singular = np.linalg.svd(jacobian, compute_uv=False)
        conditions.append(float(np.linalg.cond(jacobian)))
        singular_minima.append(float(np.min(singular)))
        if abs(float(np.linalg.det(jacobian))) <= 1e-12:
            invertible = False

    for node in validation_nodes:
        offset = _pair(node["command_offset_xy_mm"], "validation offset")
        error = _pair(node["median_image_error_px"], "validation error")
        _model_error, jacobian = evaluate_wide_error_and_jacobian(model, offset)
        try:
            full = -(np.linalg.solve(jacobian, error))
        except np.linalg.LinAlgError:
            invertible = False
            continue
        norm = float(np.linalg.norm(full))
        step = full if norm <= 0.75 else full * (0.75 / norm)
        predicted_next = error + jacobian @ step
        before_norm = float(np.linalg.norm(error))
        after_norm = float(np.linalg.norm(predicted_next))
        improvement = (
            1.0 if before_norm <= 1e-12 else (before_norm - after_norm) / before_norm
        )
        descent_rows.append(
            {
                "offset_xy_mm": offset.astype(float).tolist(),
                "limited_test_step_xy_mm": step.astype(float).tolist(),
                "before_error_norm_px": before_norm,
                "predicted_after_error_norm_px": after_norm,
                "predicted_improvement_ratio": float(improvement),
                "passed": bool(
                    before_norm <= 1.0
                    or improvement >= quality.minimum_predicted_improvement_ratio
                ),
            }
        )

    empirical = _empirical_jacobians(validation_nodes)
    consistency_rows: list[dict[str, Any]] = []
    for row in empirical:
        empirical_j = np.asarray(
            row["empirical_jacobian_px_per_mm"], dtype=np.float64
        )
        _prediction, model_j = evaluate_wide_error_and_jacobian(
            model, row["offset_xy_mm"]
        )
        relative = float(
            np.linalg.norm(model_j - empirical_j)
            / max(float(np.linalg.norm(empirical_j)), 1e-9)
        )
        consistency_rows.append(
            {
                **row,
                "model_jacobian_px_per_mm": model_j.astype(float).tolist(),
                "relative_frobenius_error": relative,
            }
        )
    return {
        "maximum_condition_number": max(conditions, default=math.inf),
        "minimum_singular_value_px_per_mm": min(
            singular_minima, default=0.0
        ),
        "invertible_everywhere": bool(invertible),
        "predicted_descent_rows": descent_rows,
        "all_validation_nodes_predict_descent": bool(
            descent_rows and all(row["passed"] for row in descent_rows)
        ),
        "validation_jacobian_consistency_rows": consistency_rows,
        "maximum_validation_jacobian_relative_error": max(
            (row["relative_frobenius_error"] for row in consistency_rows),
            default=math.inf,
        ),
    }


def fit_wide_xy_image_model(
    samples: Iterable[Mapping[str, Any]],
    training_offsets_xy_mm: Sequence[Sequence[float]],
    validation_offsets_xy_mm: Sequence[Sequence[float]],
    quality: WideXYJacobianQualityConfig = DEFAULT_WIDE_XY_JACOBIAN_QUALITY,
) -> dict[str, Any]:
    """Fit and independently validate the wide P22 image-error model."""

    expected_train = _normalise_expected(training_offsets_xy_mm, "training offsets")
    expected_validation = _normalise_expected(
        validation_offsets_xy_mm, "validation offsets"
    )
    visits = _aggregate_visits(samples, quality)
    train_nodes = _aggregate_nodes(visits, "train")
    validation_nodes = _aggregate_nodes(visits, "validation")
    train_keys = {_offset_key(node["command_offset_xy_mm"]) for node in train_nodes}
    validation_keys = {
        _offset_key(node["command_offset_xy_mm"]) for node in validation_nodes
    }
    usable_visits = [visit for visit in visits if visit["usable"]]
    every_visit_usable = bool(visits and len(usable_visits) == len(visits))
    training_coverage = expected_train.issubset(train_keys)
    validation_coverage = expected_validation.issubset(validation_keys)
    repeatability_max = max(
        (node["repeatability_max_px"] for node in train_nodes), default=math.inf
    )

    base: dict[str, Any] = {
        "status": "failure",
        "failure_reasons": [],
        "quality_configuration": asdict(quality),
        "visit_aggregates": visits,
        "training_nodes": train_nodes,
        "validation_nodes": validation_nodes,
        "models": {},
        "selected_model_type": None,
        "selected_model": None,
        "quality_gates": {},
    }
    if not training_coverage or not validation_coverage or not every_visit_usable:
        if not training_coverage:
            base["failure_reasons"].append("training_node_coverage")
        if not validation_coverage:
            base["failure_reasons"].append("validation_node_coverage")
        if not every_visit_usable:
            base["failure_reasons"].append("visit_frame_coverage")
        return base

    train_offsets = np.asarray(
        [node["command_offset_xy_mm"] for node in train_nodes], dtype=np.float64
    )
    train_errors = np.asarray(
        [node["median_image_error_px"] for node in train_nodes], dtype=np.float64
    )
    validation_offsets = np.asarray(
        [node["command_offset_xy_mm"] for node in validation_nodes], dtype=np.float64
    )
    validation_errors = np.asarray(
        [node["median_image_error_px"] for node in validation_nodes], dtype=np.float64
    )
    models: dict[str, dict[str, Any]] = {}
    for model_type in ("global_affine", "quadratic"):
        model = _fit_model(
            train_offsets,
            train_errors,
            validation_offsets,
            validation_errors,
            model_type,
        )
        model["diagnostics"] = _model_diagnostics(
            model, train_nodes, validation_nodes, quality
        )
        models[model_type] = model

    def model_passes(model: Mapping[str, Any]) -> bool:
        diagnostics = model["diagnostics"]
        return bool(
            model["training_rms_px"] <= quality.maximum_training_rms_px
            and model["validation_rms_px"] <= quality.maximum_validation_rms_px
            and model["validation_max_px"] <= quality.maximum_validation_error_px
            and diagnostics["maximum_condition_number"]
            <= quality.maximum_condition_number
            and diagnostics["minimum_singular_value_px_per_mm"]
            >= quality.minimum_singular_value_px_per_mm
            and diagnostics["maximum_validation_jacobian_relative_error"]
            <= quality.maximum_validation_jacobian_relative_error
            and diagnostics["all_validation_nodes_predict_descent"]
            and diagnostics["invertible_everywhere"]
        )

    global_ok = model_passes(models["global_affine"])
    quadratic_ok = model_passes(models["quadratic"])
    selected_type: Optional[str]
    if global_ok and (
        not quadratic_ok
        or models["global_affine"]["validation_rms_px"]
        <= models["quadratic"]["validation_rms_px"]
        + quality.global_model_simplicity_allowance_px
    ):
        selected_type = "global_affine"
    elif quadratic_ok:
        selected_type = "quadratic"
    else:
        selected_type = None

    base["models"] = models
    base["selected_model_type"] = selected_type
    base["selected_model"] = None if selected_type is None else models[selected_type]
    if selected_type is None:
        base["failure_reasons"] = ["no_wide_model_passed_independent_validation"]
        return base

    selected = models[selected_type]
    diagnostics = selected["diagnostics"]
    gates = {
        "training_node_coverage": {
            "value": len(train_keys & expected_train),
            "required": len(expected_train),
            "passed": training_coverage,
        },
        "validation_node_coverage": {
            "value": len(validation_keys & expected_validation),
            "required": len(expected_validation),
            "passed": validation_coverage,
        },
        "visit_frame_coverage": {
            "value": len(usable_visits),
            "required": len(visits),
            "passed": every_visit_usable,
        },
        "node_repeatability": {
            "value_px": repeatability_max,
            "maximum_px": quality.maximum_node_repeatability_px,
            "passed": repeatability_max <= quality.maximum_node_repeatability_px,
        },
        "selected_training_rms": {
            "value_px": selected["training_rms_px"],
            "maximum_px": quality.maximum_training_rms_px,
            "passed": selected["training_rms_px"] <= quality.maximum_training_rms_px,
        },
        "independent_validation_rms": {
            "value_px": selected["validation_rms_px"],
            "maximum_px": quality.maximum_validation_rms_px,
            "passed": selected["validation_rms_px"] <= quality.maximum_validation_rms_px,
        },
        "independent_validation_max": {
            "value_px": selected["validation_max_px"],
            "maximum_px": quality.maximum_validation_error_px,
            "passed": selected["validation_max_px"] <= quality.maximum_validation_error_px,
        },
        "jacobian_condition_number": {
            "value": diagnostics["maximum_condition_number"],
            "maximum": quality.maximum_condition_number,
            "passed": diagnostics["maximum_condition_number"]
            <= quality.maximum_condition_number,
        },
        "minimum_singular_value": {
            "value_px_per_mm": diagnostics["minimum_singular_value_px_per_mm"],
            "minimum_px_per_mm": quality.minimum_singular_value_px_per_mm,
            "passed": diagnostics["minimum_singular_value_px_per_mm"]
            >= quality.minimum_singular_value_px_per_mm,
        },
        "validation_jacobian_consistency": {
            "value_relative": diagnostics[
                "maximum_validation_jacobian_relative_error"
            ],
            "maximum_relative": quality.maximum_validation_jacobian_relative_error,
            "passed": diagnostics["maximum_validation_jacobian_relative_error"]
            <= quality.maximum_validation_jacobian_relative_error,
        },
        "predicted_descent": {
            "passed": diagnostics["all_validation_nodes_predict_descent"]
        },
        "invertible": {"passed": diagnostics["invertible_everywhere"]},
    }
    failures = [name for name, gate in gates.items() if gate.get("passed") is not True]
    base["quality_gates"] = gates
    base["failure_reasons"] = failures
    base["status"] = "success" if not failures else "failure"
    return base


def wide_correction_command_xy_mm(
    image_error_px: Any,
    current_offset_xy_mm: Any,
    calibration: Mapping[str, Any],
) -> Optional[tuple[float, float]]:
    """Return full Newton correction for a successful wide calibration."""

    if calibration.get("status") != "success":
        return None
    gates = calibration.get("quality_gates")
    if not isinstance(gates, Mapping):
        return None
    if not REQUIRED_WIDE_XY_JACOBIAN_QUALITY_GATES.issubset(gates):
        return None
    if any(
        not isinstance(gates.get(name), Mapping)
        or gates[name].get("passed") is not True
        for name in REQUIRED_WIDE_XY_JACOBIAN_QUALITY_GATES
    ):
        return None
    model = calibration.get("selected_model")
    if not isinstance(model, Mapping):
        return None
    error = _pair(image_error_px, "image_error_px")
    _prediction, jacobian = evaluate_wide_error_and_jacobian(
        model, current_offset_xy_mm
    )
    try:
        correction = -np.linalg.solve(jacobian, error)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(correction)):
        return None
    return float(correction[0]), float(correction[1])


__all__ = [
    "DEFAULT_WIDE_XY_JACOBIAN_QUALITY",
    "REQUIRED_WIDE_XY_JACOBIAN_QUALITY_GATES",
    "WideXYJacobianQualityConfig",
    "evaluate_wide_error_and_jacobian",
    "fit_wide_xy_image_model",
    "wide_correction_command_xy_mm",
]
