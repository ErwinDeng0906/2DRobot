"""Pure Stage-7A visual-servo calculations and fail-closed safety gates.

Stage 5 identifies the local image Jacobian at P22::

    delta_e_px = J_px_per_mm @ delta_q_world_xy_mm

where ``e = slot_pixel - suction_target_pixel``.  Stage 7A reuses that
approved model; it does *not* repeat the nine-point calibration.  A stable
window of fresh Stage-6 measurements is aggregated with a component-wise
median and one deliberately damped correction is proposed::

    delta_q_full = -inv(J) @ median(e)
    delta_q_raw  = gain * delta_q_full
    delta_q_cmd  = norm_limit(delta_q_raw, maximum_step_norm_mm)
    e_predicted  = median(e) + J @ delta_q_cmd

This module contains no controller, Qt, camera, file-system, or motion API.
It only returns JSON-ready reports.  In particular, ``motion_authorized`` can
be true only when every required external controller gate *and* explicit
operator consent are supplied as true by the caller.  The worker must still
re-read the controller and rebuild the proposal immediately before motion.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .xy_image_jacobian import (
    REQUIRED_XY_JACOBIAN_QUALITY_GATES,
    correction_command_xy_mm,
)


REQUIRED_STAGE7A_EXTERNAL_GATES = (
    "controller_connected",
    "controller_enabled",
    "alarm_clear",
    "estop_clear",
    "soft_estop_clear",
    "controller_idle",
    "measurement_matches_request_state",
    "camera_fresh",
    "operator_consent",
)


@dataclass(frozen=True)
class Stage7AConfig:
    """Conservative defaults for the first supervised P22 correction."""

    target_name: str = "P22"
    requested_frame_count: int = 5
    minimum_accepted_frames: int = 3
    gain: float = 0.60
    maximum_step_norm_mm: float = 0.25
    domain_margin_mm: float = 0.20
    maximum_error_dispersion_px: float = 0.75
    maximum_error_peak_deviation_px: float = 1.50
    maximum_robot_xy_spread_mm: float = 0.05
    maximum_joint_spread: float = 0.05
    maximum_command_tracking_error_mm: float = 0.05
    # CameraSourcePool deliberately drains several DirectShow buffers before
    # saving a snapshot.  Real Task10 evidence shows a normal state-to-file
    # latency of 0.254--0.296 s while every joint sample is identical.  Keep
    # this as a bounded gate, but leave enough headroom for that acquisition
    # path; the worker still performs a fresh controller read and state-match
    # audit immediately before any motion.
    maximum_robot_state_age_s: float = 0.50
    convergence_error_norm_px: float = 1.00
    minimum_effective_step_mm: float = 0.005
    minimum_predicted_improvement_ratio: float = 0.05
    maximum_response_innovation_px: float = 0.75
    minimum_response_improvement_ratio: float = 0.20


DEFAULT_STAGE7A_CONFIG = Stage7AConfig()


def _validate_config(config: Stage7AConfig) -> None:
    if not config.target_name:
        raise ValueError("target_name must not be empty")
    if config.requested_frame_count < 1:
        raise ValueError("requested_frame_count must be positive")
    if not 1 <= config.minimum_accepted_frames <= config.requested_frame_count:
        raise ValueError("minimum_accepted_frames must be within the window")
    positive = {
        "gain": config.gain,
        "maximum_step_norm_mm": config.maximum_step_norm_mm,
        "maximum_error_dispersion_px": config.maximum_error_dispersion_px,
        "maximum_error_peak_deviation_px": (
            config.maximum_error_peak_deviation_px
        ),
        "maximum_robot_xy_spread_mm": config.maximum_robot_xy_spread_mm,
        "maximum_joint_spread": config.maximum_joint_spread,
        "maximum_command_tracking_error_mm": (
            config.maximum_command_tracking_error_mm
        ),
        "maximum_robot_state_age_s": config.maximum_robot_state_age_s,
        "convergence_error_norm_px": config.convergence_error_norm_px,
        "maximum_response_innovation_px": config.maximum_response_innovation_px,
    }
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive.values()):
        raise ValueError("Stage7A positive configuration values must be finite")
    if not 0.0 < config.gain <= 1.0:
        raise ValueError("gain must be in (0, 1]")
    if not math.isfinite(config.domain_margin_mm) or config.domain_margin_mm < 0.0:
        raise ValueError("domain_margin_mm must be finite and non-negative")
    if (
        not math.isfinite(config.minimum_effective_step_mm)
        or config.minimum_effective_step_mm < 0.0
    ):
        raise ValueError("minimum_effective_step_mm must be finite and non-negative")
    for name, value in (
        (
            "minimum_predicted_improvement_ratio",
            config.minimum_predicted_improvement_ratio,
        ),
        (
            "minimum_response_improvement_ratio",
            config.minimum_response_improvement_ratio,
        ),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite_vector(value: Any, length: int) -> Optional[np.ndarray]:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None
    if result.size != length or not np.all(np.isfinite(result)):
        return None
    return result


def _json_value(value: Any) -> Any:
    """Convert common numpy/scalar values to deterministic JSON values."""

    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _gate(
    name: str,
    passed: bool,
    actual: Any,
    *,
    limit: Any = None,
    note: str = "",
) -> dict[str, Any]:
    return {
        "name": str(name),
        "passed": bool(passed),
        "actual": _json_value(actual),
        "limit": _json_value(limit),
        "note": str(note),
    }


def _normalise_external_gate(name: str, source: Any) -> dict[str, Any]:
    if isinstance(source, Mapping):
        # A missing ``passed`` is never interpreted as truthy.
        passed = source.get("passed") is True
        return _gate(
            name,
            passed,
            source.get("actual", source.get("passed")),
            limit=source.get("limit", True),
            note=str(source.get("note", "supplied by Stage7A worker")),
        )
    return _gate(
        name,
        source is True,
        source,
        limit=True,
        note="supplied by Stage7A worker",
    )


def aggregate_stable_measurements(
    measurements: Sequence[Any],
    config: Stage7AConfig = DEFAULT_STAGE7A_CONFIG,
) -> dict[str, Any]:
    """Aggregate a five-frame Stage-6 window without hiding rejected frames.

    A frame enters the numerical median only when Stage 3 accepted it and its
    error, robot XY, four joints, and state age are finite.  Rejected or
    malformed frames remain in ``raw_measurements`` so the JSON audit trail
    explains exactly why a proposal was refused.
    """

    _validate_config(config)
    rows: list[dict[str, Any]] = []
    used_errors: list[np.ndarray] = []
    used_xy: list[np.ndarray] = []
    used_joints: list[np.ndarray] = []
    used_ages: list[float] = []
    used_domain_flags: list[bool] = []
    used_target_flags: list[bool] = []
    used_source_ids: list[str] = []

    for index, measurement in enumerate(measurements):
        accepted = _field(measurement, "accepted") is True
        target_name = str(_field(measurement, "target_name", ""))
        error = _finite_vector(_field(measurement, "image_error_px"), 2)
        robot_xy = _finite_vector(
            _field(measurement, "current_robot_xy_mm"), 2
        )
        joints = _finite_vector(
            _field(
                measurement,
                "current_joints",
                _field(measurement, "joints"),
            ),
            4,
        )
        try:
            state_age = float(_field(measurement, "robot_state_age_s"))
        except (TypeError, ValueError, OverflowError):
            state_age = math.nan
        domain_passed = _field(measurement, "jacobian_domain_passed") is True
        raw_source_id = _field(
            measurement,
            "measurement_id",
            _field(measurement, "frame_id"),
        )
        source_id_present = raw_source_id is not None and str(raw_source_id) != ""
        source_id = (
            str(raw_source_id) if source_id_present else f"missing-{index + 1}"
        )
        finite = (
            error is not None
            and robot_xy is not None
            and joints is not None
            and math.isfinite(state_age)
            and state_age >= 0.0
        )
        used = bool(accepted and finite)
        row = {
            "source_id": source_id,
            "source_id_present": source_id_present,
            "target_name": target_name,
            "accepted": bool(accepted),
            "used": used,
            "image_error_px": None if error is None else error.astype(float).tolist(),
            "current_robot_xy_mm": (
                None if robot_xy is None else robot_xy.astype(float).tolist()
            ),
            "current_joints": (
                None if joints is None else joints.astype(float).tolist()
            ),
            "robot_state_age_s": (
                float(state_age) if math.isfinite(state_age) else None
            ),
            "jacobian_domain_passed": bool(domain_passed),
            "reason": str(
                _field(
                    measurement,
                    "reason",
                    "ok" if used else "rejected or incomplete measurement",
                )
            ),
        }
        rows.append(row)
        if used:
            used_errors.append(error)
            used_xy.append(robot_xy)
            used_joints.append(joints)
            used_ages.append(float(state_age))
            used_domain_flags.append(domain_passed)
            used_target_flags.append(target_name == config.target_name)
            used_source_ids.append(source_id)

    result: dict[str, Any] = {
        "requested_frame_count": int(config.requested_frame_count),
        "captured_frame_count": int(len(rows)),
        "accepted_frame_count": int(len(used_errors)),
        "raw_measurements": rows,
        "raw_error_px": [value.astype(float).tolist() for value in used_errors],
        "source_ids": used_source_ids,
        "all_source_ids_present": bool(
            rows and all(bool(row["source_id_present"]) for row in rows)
        ),
        "source_ids_unique": len(used_source_ids) == len(set(used_source_ids)),
        "median_error_px": None,
        "median_error_norm_px": None,
        "error_dispersion_px": {
            "rms_about_median": None,
            "maximum_about_median": None,
            "component_mad": None,
        },
        "current_robot_xy_mm": None,
        "robot_xy_spread_mm": None,
        "current_joints": None,
        "joint_spread": None,
        "maximum_robot_state_age_s": (
            max(used_ages) if used_ages else None
        ),
        "all_used_frames_in_jacobian_domain": bool(
            used_domain_flags and all(used_domain_flags)
        ),
        "all_used_frames_match_target": bool(
            used_target_flags and all(used_target_flags)
        ),
    }
    if not used_errors:
        return result

    errors = np.asarray(used_errors, dtype=np.float64)
    xy_values = np.asarray(used_xy, dtype=np.float64)
    joint_values = np.asarray(used_joints, dtype=np.float64)
    median_error = np.median(errors, axis=0)
    error_deviations = np.linalg.norm(errors - median_error, axis=1)
    median_xy = np.median(xy_values, axis=0)
    xy_deviations = np.linalg.norm(xy_values - median_xy, axis=1)
    median_joints = np.median(joint_values, axis=0)
    joint_spread = np.ptp(joint_values, axis=0)
    result.update(
        {
            "median_error_px": median_error.astype(float).tolist(),
            "median_error_norm_px": float(np.linalg.norm(median_error)),
            "error_dispersion_px": {
                "rms_about_median": float(
                    np.sqrt(np.mean(np.square(error_deviations)))
                ),
                "maximum_about_median": float(np.max(error_deviations)),
                "component_mad": np.median(
                    np.abs(errors - median_error), axis=0
                ).astype(float).tolist(),
            },
            "current_robot_xy_mm": median_xy.astype(float).tolist(),
            "robot_xy_spread_mm": float(np.max(xy_deviations)),
            "current_joints": median_joints.astype(float).tolist(),
            "joint_spread": joint_spread.astype(float).tolist(),
        }
    )
    return result


def _model_components(
    jacobian_payload: Mapping[str, Any],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Mapping[str, Any], list[str]]:
    fit = jacobian_payload.get("fit") or jacobian_payload
    failures: list[str] = []
    if jacobian_payload.get("status", fit.get("status")) != "success":
        failures.append("Stage5 top-level status is not success")
    if fit.get("status") != "success":
        failures.append("Stage5 fit status is not success")
    quality = fit.get("quality_gates")
    if not isinstance(quality, Mapping):
        failures.append("Stage5 quality_gates is missing")
    else:
        for name in REQUIRED_XY_JACOBIAN_QUALITY_GATES:
            if not isinstance(quality.get(name), Mapping) or quality[name].get(
                "passed"
            ) is not True:
                failures.append(f"Stage5 quality gate failed or missing: {name}")
    J = _finite_vector(fit.get("j_error_px_per_command_mm"), 4)
    inverse = _finite_vector(fit.get("j_command_mm_per_error_px"), 4)
    J_matrix = None if J is None else J.reshape(2, 2)
    inverse_matrix = None if inverse is None else inverse.reshape(2, 2)
    if J_matrix is None or inverse_matrix is None:
        failures.append("Stage5 Jacobian or inverse is missing/non-finite")
    elif not np.allclose(J_matrix @ inverse_matrix, np.eye(2), atol=1e-3):
        failures.append("Stage5 Jacobian and inverse are inconsistent")
    return J_matrix, inverse_matrix, fit, failures


def _proposal_id(payload: Mapping[str, Any]) -> str:
    serialised = json.dumps(
        _json_value(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest().upper()[:24]


def _model_fingerprint(jacobian_payload: Mapping[str, Any]) -> str:
    fit = jacobian_payload.get("fit") or jacobian_payload
    return _proposal_id(
        {
            "anchor_target_name": jacobian_payload.get("anchor_target_name"),
            "valid_target_names": jacobian_payload.get("valid_target_names"),
            "coordinate_definition": jacobian_payload.get(
                "coordinate_definition"
            ),
            "locked_inputs": jacobian_payload.get("locked_inputs"),
            "J": fit.get("j_error_px_per_command_mm"),
            "J_inverse": fit.get("j_command_mm_per_error_px"),
            "quality_gates": fit.get("quality_gates"),
        }
    )


def build_stage7a_proposal(
    measurements: Sequence[Any],
    jacobian_payload: Mapping[str, Any],
    *,
    external_safety_gates: Optional[Mapping[str, Any]] = None,
    config: Stage7AConfig = DEFAULT_STAGE7A_CONFIG,
) -> dict[str, Any]:
    """Build one supervised correction proposal; never send motion.

    ``operator_consent`` is intentionally one of the required external gates.
    The popup should first call this function with consent false to display the
    proposal.  After the operator confirms, the worker must re-read the robot,
    reacquire/revalidate freshness as appropriate, and rebuild it with consent
    true.  A previously displayed dictionary is not a motion token.
    """

    _validate_config(config)
    measurement = aggregate_stable_measurements(measurements, config)
    external = external_safety_gates or {}
    J, _inverse, fit, model_failures = _model_components(jacobian_payload)
    coordinate = jacobian_payload.get("coordinate_definition") or {}
    valid_targets = jacobian_payload.get("valid_target_names") or []
    anchor_target = str(jacobian_payload.get("anchor_target_name") or "")

    median_error = _finite_vector(measurement.get("median_error_px"), 2)
    current_xy = _finite_vector(measurement.get("current_robot_xy_mm"), 2)
    anchor_xy = _finite_vector(coordinate.get("anchor_robot_xy_mm"), 2)
    try:
        extent = float(coordinate.get("offset_extent_mm"))
    except (TypeError, ValueError, OverflowError):
        extent = math.nan
    allowed_extent = extent - config.domain_margin_mm
    domain_definition_valid = (
        anchor_xy is not None
        and math.isfinite(extent)
        and extent > 0.0
        and allowed_extent > 0.0
    )
    current_delta = (
        current_xy - anchor_xy
        if current_xy is not None and anchor_xy is not None
        else None
    )

    full_correction: Optional[np.ndarray] = None
    raw_correction: Optional[np.ndarray] = None
    commanded: Optional[np.ndarray] = None
    was_clamped = False
    if median_error is not None and not model_failures:
        correction = correction_command_xy_mm(median_error, fit)
        if correction is not None:
            full_correction = np.asarray(correction, dtype=np.float64)
            raw_correction = config.gain * full_correction
            raw_norm = float(np.linalg.norm(raw_correction))
            if raw_norm > config.maximum_step_norm_mm:
                commanded = raw_correction * (
                    config.maximum_step_norm_mm / raw_norm
                )
                was_clamped = True
            else:
                commanded = raw_correction.copy()

    before_norm = (
        float(np.linalg.norm(median_error)) if median_error is not None else None
    )
    already_aligned = bool(
        before_norm is not None
        and before_norm <= config.convergence_error_norm_px
    )
    motion_required = bool(
        commanded is not None
        and not already_aligned
        and np.linalg.norm(commanded) >= config.minimum_effective_step_mm
    )
    # Never command a nonzero step when the convergence criterion is already
    # satisfied.  Keep the calculated raw values in the report for diagnosis.
    effective_command = (
        np.zeros(2, dtype=np.float64)
        if commanded is not None and not motion_required
        else commanded
    )
    predicted_error = (
        median_error + J @ effective_command
        if median_error is not None and J is not None and effective_command is not None
        else None
    )
    predicted_xy = (
        current_xy + effective_command
        if current_xy is not None and effective_command is not None
        else None
    )
    predicted_delta = (
        predicted_xy - anchor_xy
        if predicted_xy is not None and anchor_xy is not None
        else None
    )
    predicted_norm = (
        float(np.linalg.norm(predicted_error))
        if predicted_error is not None
        else None
    )
    predicted_improvement_ratio = (
        (before_norm - predicted_norm) / before_norm
        if before_norm is not None
        and predicted_norm is not None
        and before_norm > 1e-12
        else None
    )

    dispersion = measurement["error_dispersion_px"]
    gates: dict[str, dict[str, Any]] = {}

    def add_gate(item: dict[str, Any]) -> None:
        gates[item["name"]] = item

    add_gate(
        _gate(
            "requested_frame_count",
            measurement["captured_frame_count"] == config.requested_frame_count,
            measurement["captured_frame_count"],
            limit={"exactly": config.requested_frame_count},
            note="pre-correction window must contain exactly five captures",
        )
    )
    add_gate(
        _gate(
            "unique_measurement_ids",
            measurement["all_source_ids_present"]
            and measurement["source_ids_unique"],
            {
                "all_present": measurement["all_source_ids_present"],
                "unique": measurement["source_ids_unique"],
                "source_ids": measurement["source_ids"],
            },
            limit="five present and unique frame/measurement identifiers",
            note="a frozen or replayed camera buffer cannot authorize motion",
        )
    )
    add_gate(
        _gate(
            "minimum_accepted_frames",
            measurement["accepted_frame_count"] >= config.minimum_accepted_frames,
            measurement["accepted_frame_count"],
            limit={"minimum": config.minimum_accepted_frames},
            note="accepted means Stage3 passed and robot fields are finite",
        )
    )
    add_gate(
        _gate(
            "target_consistency",
            measurement["all_used_frames_match_target"]
            and config.target_name == "P22"
            and anchor_target == config.target_name
            and config.target_name in valid_targets,
            {
                "requested": config.target_name,
                "anchor": anchor_target,
                "valid_targets": valid_targets,
                "all_frames_match": measurement[
                    "all_used_frames_match_target"
                ],
            },
            limit="P22 only",
            note="the installed Stage5 model is locally approved only for P22",
        )
    )
    add_gate(
        _gate(
            "stage3_and_jacobian_domain",
            measurement["all_used_frames_in_jacobian_domain"],
            measurement["all_used_frames_in_jacobian_domain"],
            limit=True,
            note="every numerically used frame must pass Stage3 and Stage6 domain gates",
        )
    )
    add_gate(
        _gate(
            "error_window_dispersion",
            dispersion["rms_about_median"] is not None
            and dispersion["rms_about_median"]
            <= config.maximum_error_dispersion_px
            and dispersion["maximum_about_median"]
            <= config.maximum_error_peak_deviation_px,
            dispersion,
            limit={
                "maximum_rms_px": config.maximum_error_dispersion_px,
                "maximum_peak_px": config.maximum_error_peak_deviation_px,
            },
            note="robust median is used, but a moving/noisy window still fails",
        )
    )
    add_gate(
        _gate(
            "robot_state_freshness",
            measurement["maximum_robot_state_age_s"] is not None
            and 0.0 <= measurement["maximum_robot_state_age_s"]
            <= config.maximum_robot_state_age_s,
            measurement["maximum_robot_state_age_s"],
            limit={"maximum_s": config.maximum_robot_state_age_s},
            note="Stage7A is stricter than the read-only Stage6 display",
        )
    )
    joint_spread = measurement.get("joint_spread")
    stationary = (
        measurement["robot_xy_spread_mm"] is not None
        and measurement["robot_xy_spread_mm"]
        <= config.maximum_robot_xy_spread_mm
        and isinstance(joint_spread, list)
        and len(joint_spread) == 4
        and max(joint_spread) <= config.maximum_joint_spread
    )
    add_gate(
        _gate(
            "robot_stationary_window",
            stationary,
            {
                "maximum_xy_deviation_mm": measurement[
                    "robot_xy_spread_mm"
                ],
                "joint_peak_to_peak": joint_spread,
            },
            limit={
                "maximum_xy_deviation_mm": config.maximum_robot_xy_spread_mm,
                "maximum_joint_peak_to_peak": config.maximum_joint_spread,
            },
            note="motion-process camera frames must not enter a proposal",
        )
    )
    add_gate(
        _gate(
            "stage5_model_quality",
            not model_failures,
            {"failure_reasons": model_failures},
            limit="all required Stage5 gates pass and J*inv(J) ~= I",
            note="correction_command_xy_mm is reused for the sign convention",
        )
    )
    add_gate(
        _gate(
            "local_domain_definition",
            domain_definition_valid,
            {
                "extent_mm": extent if math.isfinite(extent) else None,
                "margin_mm": config.domain_margin_mm,
                "allowed_each_axis_mm": (
                    allowed_extent if domain_definition_valid else None
                ),
            },
            limit="positive extent after safety margin",
            note="the 0.20 mm margin keeps commands away from the Task9 boundary",
        )
    )
    current_inside = bool(
        domain_definition_valid
        and current_delta is not None
        and np.all(np.abs(current_delta) <= allowed_extent + 1e-9)
    )
    add_gate(
        _gate(
            "current_point_inside_local_domain",
            current_inside,
            None if current_delta is None else current_delta,
            limit={"absolute_each_axis_mm": allowed_extent if domain_definition_valid else None},
            note="delta is relative to Stage5 anchor_robot_xy_mm",
        )
    )
    full_local = bool(
        full_correction is not None
        and domain_definition_valid
        and np.all(np.abs(full_correction) <= extent + 1e-9)
    )
    add_gate(
        _gate(
            "full_cancellation_within_measured_scale",
            full_local,
            full_correction,
            limit={"absolute_each_axis_mm": extent if math.isfinite(extent) else None},
            note="large visual errors are not rescued merely by step clamping",
        )
    )
    step_norm = (
        float(np.linalg.norm(effective_command))
        if effective_command is not None
        else None
    )
    add_gate(
        _gate(
            "command_step_norm",
            step_norm is not None and step_norm <= config.maximum_step_norm_mm + 1e-9,
            step_norm,
            limit={"maximum_mm": config.maximum_step_norm_mm},
            note="norm limiting preserves direction; it does not clip X/Y independently",
        )
    )
    endpoint_inside = bool(
        domain_definition_valid
        and predicted_delta is not None
        and np.all(np.abs(predicted_delta) <= allowed_extent + 1e-9)
    )
    add_gate(
        _gate(
            "predicted_endpoint_inside_local_domain",
            endpoint_inside,
            predicted_delta,
            limit={"absolute_each_axis_mm": allowed_extent if domain_definition_valid else None},
            note="both current point and predicted endpoint must stay in the shrunken domain",
        )
    )
    prediction_ok = bool(
        already_aligned
        or (
            predicted_improvement_ratio is not None
            and predicted_improvement_ratio
            >= config.minimum_predicted_improvement_ratio
        )
    )
    add_gate(
        _gate(
            "predicted_model_response",
            prediction_ok,
            {
                "before_norm_px": before_norm,
                "predicted_norm_px": predicted_norm,
                "predicted_improvement_ratio": predicted_improvement_ratio,
            },
            limit={
                "minimum_improvement_ratio": (
                    config.minimum_predicted_improvement_ratio
                )
            },
            note="a command whose own Stage5 model predicts no improvement is rejected",
        )
    )

    for name in REQUIRED_STAGE7A_EXTERNAL_GATES:
        add_gate(_normalise_external_gate(name, external.get(name)))

    non_consent_gates = [
        value for key, value in gates.items() if key != "operator_consent"
    ]
    ready_for_confirmation = bool(
        non_consent_gates and all(item["passed"] for item in non_consent_gates)
    )
    overall_passed = bool(gates and all(item["passed"] for item in gates.values()))
    motion_authorized = bool(overall_passed and motion_required)
    if overall_passed and already_aligned:
        decision = "already_aligned"
    elif motion_authorized:
        decision = "authorized_single_step"
    elif ready_for_confirmation and motion_required:
        decision = "awaiting_operator_confirmation"
    else:
        decision = "rejected"
    failures = [name for name, item in gates.items() if not item["passed"]]

    report: dict[str, Any] = {
        "schema_version": 1,
        "stage": "7A_supervised_single_step",
        "target_name": config.target_name,
        "jacobian_model_fingerprint": _model_fingerprint(jacobian_payload),
        "configuration": asdict(config),
        "measurement": measurement,
        "calculation": {
            "equation": "delta_q_cmd = norm_limit(gain * (-inv(J) * e_median))",
            "full_cancellation_correction_xy_mm": _json_value(full_correction),
            "unclipped_correction_xy_mm": _json_value(raw_correction),
            "commanded_correction_xy_mm": _json_value(effective_command),
            # Alias retained because the Stage7A UI/runtime names this value
            # explicitly in its JSON contract.
            "correction_xy_mm_raw": _json_value(raw_correction),
            "gain": float(config.gain),
            "maximum_step_norm_mm": float(config.maximum_step_norm_mm),
            "was_clamped": bool(was_clamped),
            "predicted_endpoint_xy_mm": _json_value(predicted_xy),
            "predicted_endpoint_delta_xy_mm": _json_value(predicted_delta),
            "predicted_error_px": _json_value(predicted_error),
            "predicted_error_norm_px": predicted_norm,
            "predicted_improvement_ratio": predicted_improvement_ratio,
        },
        "safety_gates": gates,
        "quality_gates": gates,
        "current_joints": measurement["current_joints"],
        "current_robot_xy_mm": measurement["current_robot_xy_mm"],
        "current_robot_delta_xy_mm": _json_value(current_delta),
        "image_error_px": measurement["median_error_px"],
        "correction_xy_mm_raw": _json_value(raw_correction),
        "commanded_correction_xy_mm": _json_value(effective_command),
        "predicted_endpoint_xy_mm": _json_value(predicted_xy),
        "predicted_endpoint_delta_xy_mm": _json_value(predicted_delta),
        "predicted_error_px": _json_value(predicted_error),
        "ready_for_operator_confirmation": ready_for_confirmation,
        "operator_confirmation_required": True,
        "motion_required": motion_required,
        "motion_authorized": motion_authorized,
        "overall_passed": overall_passed,
        "decision": decision,
        "failure_reasons": failures,
    }
    report["proposal_id"] = _proposal_id(
        {
            "target_name": report["target_name"],
            "source_ids": measurement["source_ids"],
            "median_error_px": measurement["median_error_px"],
            "current_robot_xy_mm": measurement["current_robot_xy_mm"],
            "current_joints": measurement["current_joints"],
            "commanded_correction_xy_mm": report["calculation"][
                "commanded_correction_xy_mm"
            ],
            "safety_gates": gates,
            "jacobian_model_fingerprint": report[
                "jacobian_model_fingerprint"
            ],
        }
    )
    return report


def evaluate_stage7a_response(
    proposal: Mapping[str, Any],
    post_measurements: Sequence[Any],
    jacobian_payload: Mapping[str, Any],
    *,
    config: Stage7AConfig = DEFAULT_STAGE7A_CONFIG,
) -> dict[str, Any]:
    """Compare the post-move window with Stage5's predicted response.

    The innovation is

    ``r = (e_after - e_before) - J @ delta_q_commanded``.

    Stage7A passes only when the post window is stable, the error improves by
    at least the configured ratio, and the innovation remains small.  This is
    an empirical response check, not permission for a second automatic step.
    """

    _validate_config(config)
    post = aggregate_stable_measurements(post_measurements, config)
    before = _finite_vector(
        (proposal.get("measurement") or {}).get("median_error_px"), 2
    )
    commanded = _finite_vector(
        (proposal.get("calculation") or {}).get(
            "commanded_correction_xy_mm"
        ),
        2,
    )
    after = _finite_vector(post.get("median_error_px"), 2)
    J, _inverse, _fit, model_failures = _model_components(jacobian_payload)
    current_model_fingerprint = _model_fingerprint(jacobian_payload)

    actual_delta = after - before if after is not None and before is not None else None
    predicted_delta = J @ commanded if J is not None and commanded is not None else None
    innovation = (
        actual_delta - predicted_delta
        if actual_delta is not None and predicted_delta is not None
        else None
    )
    before_norm = float(np.linalg.norm(before)) if before is not None else None
    after_norm = float(np.linalg.norm(after)) if after is not None else None
    innovation_norm = (
        float(np.linalg.norm(innovation)) if innovation is not None else None
    )
    before_xy = _finite_vector(
        (proposal.get("measurement") or {}).get("current_robot_xy_mm"), 2
    )
    after_xy = _finite_vector(post.get("current_robot_xy_mm"), 2)
    actual_command_delta = (
        after_xy - before_xy
        if after_xy is not None and before_xy is not None
        else None
    )
    command_tracking_error = (
        float(np.linalg.norm(actual_command_delta - commanded))
        if actual_command_delta is not None and commanded is not None
        else None
    )
    before_source_ids = set(
        str(value)
        for value in (proposal.get("measurement") or {}).get("source_ids", [])
    )
    post_source_ids = set(str(value) for value in post.get("source_ids", []))
    reused_source_ids = sorted(before_source_ids.intersection(post_source_ids))
    improvement_ratio = (
        (before_norm - after_norm) / before_norm
        if before_norm is not None
        and after_norm is not None
        and before_norm > 1e-12
        else None
    )
    dispersion = post["error_dispersion_px"]
    gates = {
        "proposal_was_motion_authorized": _gate(
            "proposal_was_motion_authorized",
            proposal.get("motion_authorized") is True,
            proposal.get("motion_authorized"),
            limit=True,
            note="response validation is meaningful only after an authorized step",
        ),
        "proposal_model_unchanged": _gate(
            "proposal_model_unchanged",
            bool(proposal.get("jacobian_model_fingerprint"))
            and proposal.get("jacobian_model_fingerprint")
            == current_model_fingerprint,
            {
                "proposal": proposal.get("jacobian_model_fingerprint"),
                "current": current_model_fingerprint,
            },
            limit="exact fingerprint match",
            note="the installed Jacobian must not change between prediction and response",
        ),
        "post_window_frame_count": _gate(
            "post_window_frame_count",
            post["captured_frame_count"] == config.requested_frame_count,
            post["captured_frame_count"],
            limit={"exactly": config.requested_frame_count},
            note="exactly five new frames are required after settling",
        ),
        "post_window_unique_measurement_ids": _gate(
            "post_window_unique_measurement_ids",
            post["all_source_ids_present"] and post["source_ids_unique"],
            {
                "all_present": post["all_source_ids_present"],
                "unique": post["source_ids_unique"],
                "source_ids": post["source_ids"],
            },
            limit="five present and unique identifiers",
            note="post evidence must consist of genuinely new camera frames",
        ),
        "post_frames_newer_than_proposal": _gate(
            "post_frames_newer_than_proposal",
            bool(before_source_ids)
            and bool(post_source_ids)
            and not reused_source_ids,
            {"reused_source_ids": reused_source_ids},
            limit={"maximum_reused_count": 0},
            note="no post frame may be reused from the pre-move proposal",
        ),
        "post_window_accepted_frames": _gate(
            "post_window_accepted_frames",
            post["accepted_frame_count"] >= config.minimum_accepted_frames,
            post["accepted_frame_count"],
            limit={"minimum": config.minimum_accepted_frames},
            note="post frames must independently pass Stage3",
        ),
        "post_window_domain": _gate(
            "post_window_domain",
            post["all_used_frames_in_jacobian_domain"]
            and post["all_used_frames_match_target"],
            {
                "domain": post["all_used_frames_in_jacobian_domain"],
                "target": post["all_used_frames_match_target"],
            },
            limit=True,
            note="post-move evidence cannot reuse the pre-move pose",
        ),
        "post_window_dispersion": _gate(
            "post_window_dispersion",
            dispersion["rms_about_median"] is not None
            and dispersion["rms_about_median"] <= config.maximum_error_dispersion_px
            and dispersion["maximum_about_median"]
            <= config.maximum_error_peak_deviation_px,
            dispersion,
            limit={
                "maximum_rms_px": config.maximum_error_dispersion_px,
                "maximum_peak_px": config.maximum_error_peak_deviation_px,
            },
            note="post-move result must be stable before judging the model",
        ),
        "post_robot_state_freshness": _gate(
            "post_robot_state_freshness",
            post["maximum_robot_state_age_s"] is not None
            and 0.0 <= post["maximum_robot_state_age_s"]
            <= config.maximum_robot_state_age_s,
            post["maximum_robot_state_age_s"],
            limit={"maximum_s": config.maximum_robot_state_age_s},
            note="post-move controller state must be fresh",
        ),
        "post_robot_stationary_window": _gate(
            "post_robot_stationary_window",
            post["robot_xy_spread_mm"] is not None
            and post["robot_xy_spread_mm"]
            <= config.maximum_robot_xy_spread_mm
            and isinstance(post.get("joint_spread"), list)
            and len(post["joint_spread"]) == 4
            and max(post["joint_spread"]) <= config.maximum_joint_spread,
            {
                "maximum_xy_deviation_mm": post["robot_xy_spread_mm"],
                "joint_peak_to_peak": post.get("joint_spread"),
            },
            limit={
                "maximum_xy_deviation_mm": config.maximum_robot_xy_spread_mm,
                "maximum_joint_peak_to_peak": config.maximum_joint_spread,
            },
            note="the arm must have settled before response validation",
        ),
        "command_tracking": _gate(
            "command_tracking",
            command_tracking_error is not None
            and command_tracking_error
            <= config.maximum_command_tracking_error_mm,
            {
                "commanded_delta_xy_mm": _json_value(commanded),
                "actual_delta_xy_mm": _json_value(actual_command_delta),
                "tracking_error_mm": command_tracking_error,
            },
            limit={
                "maximum_error_mm": config.maximum_command_tracking_error_mm
            },
            note="image response is meaningful only if the commanded XY step occurred",
        ),
        "stage5_model_quality": _gate(
            "stage5_model_quality",
            not model_failures,
            {"failure_reasons": model_failures},
            limit="approved finite invertible Stage5 model",
            note="the same installed model must predict and verify the step",
        ),
        "response_innovation": _gate(
            "response_innovation",
            innovation_norm is not None
            and innovation_norm <= config.maximum_response_innovation_px,
            innovation_norm,
            limit={"maximum_px": config.maximum_response_innovation_px},
            note="innovation = actual delta error - predicted delta error",
        ),
        "actual_error_improvement": _gate(
            "actual_error_improvement",
            improvement_ratio is not None
            and improvement_ratio >= config.minimum_response_improvement_ratio,
            improvement_ratio,
            limit={
                "minimum_ratio": config.minimum_response_improvement_ratio
            },
            note="a wrong-sign, stagnant, or divergent response fails",
        ),
    }
    passed = bool(gates and all(gate["passed"] for gate in gates.values()))
    return {
        "schema_version": 1,
        "stage": "7A_response_validation",
        "proposal_id": proposal.get("proposal_id"),
        "post_measurement": post,
        "actual_delta_error_px": _json_value(actual_delta),
        "predicted_delta_error_px": _json_value(predicted_delta),
        "innovation_px": _json_value(innovation),
        "innovation_norm_px": innovation_norm,
        "commanded_delta_xy_mm": _json_value(commanded),
        "actual_command_delta_xy_mm": _json_value(actual_command_delta),
        "command_tracking_error_mm": command_tracking_error,
        "before_error_norm_px": before_norm,
        "after_error_norm_px": after_norm,
        "improvement_ratio": improvement_ratio,
        "quality_gates": gates,
        "passed": passed,
        "failure_reasons": [
            name for name, gate in gates.items() if not gate["passed"]
        ],
    }


__all__ = [
    "DEFAULT_STAGE7A_CONFIG",
    "REQUIRED_STAGE7A_EXTERNAL_GATES",
    "Stage7AConfig",
    "aggregate_stable_measurements",
    "build_stage7a_proposal",
    "evaluate_stage7a_response",
]
