"""Controller-free two-tier visual-servo calculations for Stage 7B.

Stage 7B does not recalibrate a Jacobian while running.  It consumes the
Task11 wide-area model and the approved Task9 local model.  With
``e = slot_pixel - suction_target_pixel`` and command-space offset ``d`` from
the measured P22 anchor, one finite iteration is::

    J_k = d e / d q | d_k
    dq_full = -inverse(J_k) @ median(e_k)
    dq_cmd = norm_limit(gain * dq_full, step_limit)
    e_pred = median(e_k) + J_k @ dq_cmd

The Task11 model is selected outside the shrunken Task9 domain.  The Task9
model is selected when both command axes are within ``2.0 - 0.2 = 1.8 mm``.
No camera, Qt, controller, file-system, DO, vacuum, or Z-motion API is imported
by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from scara.pipeline.xy_correction_planner import plan_fixed_rz_xy_step

from .wide_xy_jacobian import (
    REQUIRED_WIDE_XY_JACOBIAN_QUALITY_GATES,
    evaluate_wide_error_and_jacobian,
    wide_correction_command_xy_mm,
)
from .xy_image_jacobian import (
    REQUIRED_XY_JACOBIAN_QUALITY_GATES,
    correction_command_xy_mm,
)


@dataclass(frozen=True)
class Stage7BConfig:
    target_name: str = "P22"
    stable_window_size: int = 5
    minimum_accepted_frames: int = 3
    convergence_error_norm_px: float = 1.0
    arrival_distance_threshold_mm: float = 1.0
    local_extent_mm: float = 2.0
    local_margin_mm: float = 0.2
    wide_extent_mm: float = 10.0
    wide_margin_mm: float = 0.5
    fine_gain: float = 0.60
    coarse_gain: float = 0.65
    fine_planning_step_limit_mm: float = 0.24
    fine_execution_step_limit_mm: float = 0.25
    # Keep the wide-model proposal below the ActionWorker hard ceiling.  The
    # 0.01 mm reserve absorbs controller joint quantization and FK round-off
    # when the completed move is audited from the returned joint state.
    coarse_planning_step_limit_mm: float = 0.74
    coarse_execution_step_limit_mm: float = 0.75
    minimum_effective_step_mm: float = 0.005
    # A 20 x 20 mm square has a 14.14 mm corner-to-centre diagonal.  With
    # 0.74 mm coarse steps and the final Task9 fine phase, 25 moves can stop
    # before convergence.  Thirty-two remains a finite iteration ceiling;
    # the cumulative-path ceiling is deliberately a separate operator policy.
    maximum_iterations: int = 32
    maximum_total_path_mm: float = 50.0
    maximum_error_dispersion_px: float = 1.50
    maximum_peak_deviation_px: float = 3.00
    maximum_robot_xy_spread_mm: float = 0.20
    maximum_joint_spread: float = 0.20
    maximum_robot_state_age_s: float = 1.00
    maximum_request_state_xy_mismatch_mm: float = 0.20
    maximum_request_state_joint_mismatch: float = 0.20
    maximum_anchor_mismatch_mm: float = 0.50
    maximum_response_innovation_px: float = 2.00
    minimum_response_drop_px: float = 0.10
    maximum_command_tracking_error_mm: float = 0.15
    j3_tolerance_mm: float = 0.15
    rz_tolerance_deg: float = 0.20
    target_rz_tolerance_deg: float = 0.15
    maximum_sequential_transient_rz_deg: float = 0.30
    fine_maximum_sequential_transient_xy_mm: float = 0.50
    coarse_maximum_sequential_transient_xy_mm: float = 1.50


DEFAULT_STAGE7B_CONFIG = Stage7BConfig()


def _finite(value: Any, length: int) -> Optional[np.ndarray]:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None
    if array.size != length or not np.all(np.isfinite(array)):
        return None
    return array


def _json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _gate(passed: bool, actual: Any, limit: Any, note: str) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "actual": _json(actual),
        "limit": _json(limit),
        "note": str(note),
    }


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _all_required_gates(
    payload: Mapping[str, Any], required: frozenset[str], *, fit_nested: bool
) -> tuple[bool, list[str]]:
    fit = payload.get("fit") if fit_nested else payload
    if not isinstance(fit, Mapping):
        return False, ["missing_fit"]
    gates = fit.get("quality_gates")
    failures: list[str] = []
    if payload.get("status") != "success" or fit.get("status") != "success":
        failures.append("status")
    if not isinstance(gates, Mapping):
        failures.append("quality_gates")
        return False, failures
    for name in sorted(required):
        gate = gates.get(name)
        if not isinstance(gate, Mapping) or gate.get("passed") is not True:
            failures.append(name)
    return not failures, failures


def aggregate_stage7b_window(
    samples: Sequence[Mapping[str, Any]],
    config: Stage7BConfig = DEFAULT_STAGE7B_CONFIG,
) -> dict[str, Any]:
    """Aggregate one sequence-bound, stationary burst using medians."""

    rows = list(samples)
    used: list[dict[str, Any]] = []
    ids: list[str] = []
    for row in rows:
        source_id = str(row.get("measurement_id") or "").strip()
        ids.append(source_id)
        error = _finite(row.get("image_error_px"), 2)
        xy = _finite(row.get("current_robot_xy_mm"), 2)
        joints = _finite(row.get("current_joints"), 4)
        pose = _finite(row.get("current_pose"), 6)
        try:
            age = float(row.get("robot_state_age_s"))
        except (TypeError, ValueError, OverflowError):
            age = math.inf
        if (
            row.get("accepted") is True
            and str(row.get("target_name") or "") == config.target_name
            and source_id
            and error is not None
            and xy is not None
            and joints is not None
            and pose is not None
            and math.isfinite(age)
            and age >= 0.0
        ):
            used.append(
                {
                    "measurement_id": source_id,
                    "error": error,
                    "xy": xy,
                    "joints": joints,
                    "pose": pose,
                    "age": age,
                }
            )
    result: dict[str, Any] = {
        "captured_frame_count": len(rows),
        "accepted_frame_count": len(used),
        "source_ids": ids,
        "all_source_ids_present": bool(ids and all(ids)),
        "source_ids_unique": len(ids) == len(set(ids)),
        "median_error_px": None,
        "current_robot_xy_mm": None,
        "current_joints": None,
        "current_pose": None,
        "maximum_robot_state_age_s": None,
        "error_dispersion_px": {"rms": None, "maximum": None},
        "robot_xy_spread_mm": None,
        "joint_spread": None,
    }
    if not used:
        return result
    errors = np.asarray([row["error"] for row in used])
    xys = np.asarray([row["xy"] for row in used])
    joints = np.asarray([row["joints"] for row in used])
    poses = np.asarray([row["pose"] for row in used])
    median_error = np.median(errors, axis=0)
    median_xy = np.median(xys, axis=0)
    median_joints = np.median(joints, axis=0)
    median_pose = np.median(poses, axis=0)
    error_deviation = np.linalg.norm(errors - median_error, axis=1)
    result.update(
        {
            "median_error_px": median_error.astype(float).tolist(),
            "current_robot_xy_mm": median_xy.astype(float).tolist(),
            "current_joints": median_joints.astype(float).tolist(),
            "current_pose": median_pose.astype(float).tolist(),
            "maximum_robot_state_age_s": max(row["age"] for row in used),
            "error_dispersion_px": {
                "rms": float(np.sqrt(np.mean(np.square(error_deviation)))),
                "maximum": float(np.max(error_deviation)),
            },
            "robot_xy_spread_mm": float(
                np.max(np.linalg.norm(xys - median_xy, axis=1))
            ),
            "joint_spread": np.ptp(joints, axis=0).astype(float).tolist(),
        }
    )
    return result


def build_stage7b_iteration(
    samples: Sequence[Mapping[str, Any]],
    request: Mapping[str, Any],
    local_calibration: Mapping[str, Any],
    wide_calibration: Mapping[str, Any],
    *,
    iteration_index: int,
    cumulative_path_mm: float,
    previous_iteration: Optional[Mapping[str, Any]] = None,
    config: Stage7BConfig = DEFAULT_STAGE7B_CONFIG,
) -> dict[str, Any]:
    """Build one fail-closed Stage7B decision and kinematic target."""

    measurement = aggregate_stage7b_window(samples, config)
    controller = request.get("controller_state") or {}
    external = request.get("external_safety_gates") or {}
    current_xy = _finite(measurement.get("current_robot_xy_mm"), 2)
    current_joints = _finite(measurement.get("current_joints"), 4)
    current_pose = _finite(measurement.get("current_pose"), 6)
    request_xy = _finite(controller.get("pose"), 6)
    request_joints = _finite(controller.get("joints"), 4)
    local_ok, local_failures = _all_required_gates(
        local_calibration, REQUIRED_XY_JACOBIAN_QUALITY_GATES, fit_nested=True
    )
    wide_fit = wide_calibration.get("fit") or wide_calibration
    wide_ok, wide_failures = _all_required_gates(
        wide_fit, REQUIRED_WIDE_XY_JACOBIAN_QUALITY_GATES, fit_nested=False
    )
    local_coordinate = local_calibration.get("coordinate_definition") or {}
    wide_coordinate = wide_calibration.get("coordinate_definition") or {}
    anchor = _finite(local_coordinate.get("anchor_robot_xy_mm"), 2)
    wide_anchor = _finite(wide_coordinate.get("anchor_robot_xy_mm"), 2)
    anchors_match = bool(
        anchor is not None
        and wide_anchor is not None
        and np.linalg.norm(anchor - wide_anchor)
        <= config.maximum_anchor_mismatch_mm
    )
    offset = current_xy - anchor if current_xy is not None and anchor is not None else None
    error = _finite(measurement.get("median_error_px"), 2)
    error_norm = float(np.linalg.norm(error)) if error is not None else None

    local_allowed = config.local_extent_mm - config.local_margin_mm
    wide_allowed = config.wide_extent_mm - config.wide_margin_mm
    tier: Optional[str] = None
    if offset is not None and np.max(np.abs(offset)) <= local_allowed + 1e-9:
        tier = "fine_task9"
    elif offset is not None and np.max(np.abs(offset)) <= wide_allowed + 1e-9:
        tier = "coarse_task11"

    gates: dict[str, dict[str, Any]] = {
        "target_is_p22": _gate(
            str(request.get("target_name") or "") == config.target_name,
            request.get("target_name"), config.target_name, "Stage7B is approved for P22 only"
        ),
        "session_iteration_limit": _gate(
            1 <= int(iteration_index) <= config.maximum_iterations + 1,
            iteration_index,
            {
                "maximum_motion_iterations": config.maximum_iterations,
                "one_final_observation_only_iteration": config.maximum_iterations + 1,
            },
            "the final extra request measures the response to motion 20 and cannot move",
        ),
        "session_path_limit": _gate(
            0.0 <= float(cumulative_path_mm) <= config.maximum_total_path_mm,
            cumulative_path_mm, {"maximum_mm": config.maximum_total_path_mm}, "cumulative XY path"
        ),
        "stable_window_count": _gate(
            measurement["captured_frame_count"] == config.stable_window_size
            and measurement["accepted_frame_count"] >= config.minimum_accepted_frames,
            {"captured": measurement["captured_frame_count"], "accepted": measurement["accepted_frame_count"]},
            {"captured": config.stable_window_size, "accepted_minimum": config.minimum_accepted_frames},
            "new Stage3 frames only",
        ),
        "unique_measurement_ids": _gate(
            measurement["all_source_ids_present"] and measurement["source_ids_unique"],
            measurement["source_ids"], "all present and unique", "reject frozen/replayed frames"
        ),
        "error_window_stable": _gate(
            measurement["error_dispersion_px"]["rms"] is not None
            and measurement["error_dispersion_px"]["rms"] <= config.maximum_error_dispersion_px
            and measurement["error_dispersion_px"]["maximum"] <= config.maximum_peak_deviation_px,
            measurement["error_dispersion_px"],
            {"maximum_rms_px": config.maximum_error_dispersion_px, "maximum_peak_px": config.maximum_peak_deviation_px},
            "median is used, but an unstable burst cannot move",
        ),
        "robot_stationary_window": _gate(
            measurement["robot_xy_spread_mm"] is not None
            and measurement["robot_xy_spread_mm"] <= config.maximum_robot_xy_spread_mm
            and isinstance(measurement["joint_spread"], list)
            and max(measurement["joint_spread"]) <= config.maximum_joint_spread,
            {"xy_mm": measurement["robot_xy_spread_mm"], "joint_spread": measurement["joint_spread"]},
            {"xy_mm": config.maximum_robot_xy_spread_mm, "joint": config.maximum_joint_spread},
            "do not mix moving frames into a proposal",
        ),
        "robot_state_freshness": _gate(
            measurement["maximum_robot_state_age_s"] is not None
            and measurement["maximum_robot_state_age_s"] <= config.maximum_robot_state_age_s,
            measurement["maximum_robot_state_age_s"], config.maximum_robot_state_age_s, "fresh camera/state pairing"
        ),
        "request_state_matches_measurement": _gate(
            current_xy is not None and current_joints is not None and request_xy is not None and request_joints is not None
            and np.linalg.norm(current_xy - request_xy[:2]) <= config.maximum_request_state_xy_mismatch_mm
            and np.max(np.abs(current_joints - request_joints)) <= config.maximum_request_state_joint_mismatch,
            None if current_xy is None or current_joints is None or request_xy is None or request_joints is None else {
                "xy_mm": float(np.linalg.norm(current_xy - request_xy[:2])),
                "joint_max": float(np.max(np.abs(current_joints - request_joints))),
            },
            {"xy_mm": config.maximum_request_state_xy_mismatch_mm, "joint": config.maximum_request_state_joint_mismatch},
            "bind the camera burst to this worker request",
        ),
        "local_model_quality": _gate(local_ok, local_failures, [], "Task9 quality/hash validation"),
        "wide_model_quality": _gate(wide_ok, wide_failures, [], "Task11 independent validation"),
        "calibration_anchors_match": _gate(
            anchors_match,
            None
            if anchor is None or wide_anchor is None
            else float(np.linalg.norm(anchor - wide_anchor)),
            config.maximum_anchor_mismatch_mm,
            "Task9 and Task11 P22 anchors must agree within the configured tolerance",
        ),
        "current_inside_supported_domain": _gate(tier is not None, _json(offset), {"absolute_each_axis_mm": wide_allowed}, "select fine at <=1.8mm, otherwise wide"),
    }
    for name in (
        "controller_connected", "controller_enabled", "alarm_clear",
        "estop_clear", "soft_estop_clear", "controller_idle",
    ):
        gates[name] = _gate(external.get(name) is True, external.get(name), True, "authoritative worker snapshot")

    response_validation: Optional[dict[str, Any]] = None
    if previous_iteration is not None:
        before_error = _finite(previous_iteration.get("median_error_px"), 2)
        before_xy = _finite(previous_iteration.get("current_robot_xy_mm"), 2)
        command = _finite(previous_iteration.get("commanded_correction_xy_mm"), 2)
        predicted_delta = _finite(previous_iteration.get("predicted_delta_error_px"), 2)
        actual_delta = error - before_error if error is not None and before_error is not None else None
        innovation = actual_delta - predicted_delta if actual_delta is not None and predicted_delta is not None else None
        tracking = float(np.linalg.norm((current_xy-before_xy)-command)) if current_xy is not None and before_xy is not None and command is not None else None
        before_norm = float(np.linalg.norm(before_error)) if before_error is not None else None
        drop = before_norm - error_norm if before_norm is not None and error_norm is not None else None
        response_validation = {
            "actual_delta_error_px": _json(actual_delta),
            "predicted_delta_error_px": _json(predicted_delta),
            "innovation_px": _json(innovation),
            "innovation_norm_px": None if innovation is None else float(np.linalg.norm(innovation)),
            "command_tracking_error_mm": tracking,
            "error_drop_px": drop,
        }
        gates["previous_response_innovation"] = _gate(
            innovation is not None and np.linalg.norm(innovation) <= config.maximum_response_innovation_px,
            response_validation["innovation_norm_px"], config.maximum_response_innovation_px,
            "stop if the real image response contradicts the selected model",
        )
        gates["previous_command_tracking"] = _gate(
            tracking is not None and tracking <= config.maximum_command_tracking_error_mm,
            tracking, config.maximum_command_tracking_error_mm, "actual XY must match the last command",
        )
        gates["previous_error_improved"] = _gate(
            bool(error_norm is not None and error_norm <= config.convergence_error_norm_px)
            or (drop is not None and drop >= config.minimum_response_drop_px),
            drop, {"minimum_drop_px": config.minimum_response_drop_px, "or_converged_px": config.convergence_error_norm_px},
            "no ineffective or divergent automatic repetition",
        )

    pixel_converged = bool(
        error_norm is not None
        and error_norm <= config.convergence_error_norm_px
    )
    # The remaining suction-to-slot distance is estimated from the selected
    # visual model as the full XY cancellation command ||-J^-1 e||.  This is
    # intentionally not the robot-anchor distance: the calibrated P22 anchor
    # can retain a non-zero visual intercept.
    model_inputs_passed = bool(
        gates
        and all(
            gate["passed"]
            for name, gate in gates.items()
            if name != "previous_error_improved"
        )
    )
    full: Optional[np.ndarray] = None
    jacobian: Optional[np.ndarray] = None
    gain = config.fine_gain if tier == "fine_task9" else config.coarse_gain
    step_limit = (
        config.fine_planning_step_limit_mm
        if tier == "fine_task9"
        else config.coarse_planning_step_limit_mm
    )
    if model_inputs_passed and error is not None and offset is not None:
        if tier == "fine_task9":
            correction = correction_command_xy_mm(error, local_calibration.get("fit") or local_calibration)
            jacobian = np.asarray((local_calibration.get("fit") or local_calibration).get("j_error_px_per_command_mm"), dtype=np.float64)
        else:
            correction = wide_correction_command_xy_mm(error, offset, wide_fit)
            _predicted, jacobian = evaluate_wide_error_and_jacobian(wide_fit["selected_model"], offset)
        if correction is not None:
            full = np.asarray(correction, dtype=np.float64)
    remaining_alignment_distance_mm = (
        float(np.linalg.norm(full)) if full is not None else None
    )
    distance_converged = bool(
        remaining_alignment_distance_mm is not None
        and remaining_alignment_distance_mm
        <= config.arrival_distance_threshold_mm + 1e-12
    )
    converged = bool(pixel_converged or distance_converged)
    convergence_reason: Optional[str] = None
    if distance_converged:
        convergence_reason = "within_1mm"
    elif pixel_converged:
        convergence_reason = "pixel_error_threshold"
    if previous_iteration is not None and "previous_error_improved" in gates:
        drop = (
            response_validation.get("error_drop_px")
            if isinstance(response_validation, Mapping)
            else None
        )
        gates["previous_error_improved"] = _gate(
            converged
            or (drop is not None and drop >= config.minimum_response_drop_px),
            drop,
            {
                "minimum_drop_px": config.minimum_response_drop_px,
                "or_pixel_converged_px": config.convergence_error_norm_px,
                "or_remaining_distance_mm": config.arrival_distance_threshold_mm,
            },
            "no ineffective or divergent automatic repetition",
        )
    gates["motion_iteration_budget"] = _gate(
        converged or int(iteration_index) <= config.maximum_iterations,
        iteration_index,
        {"maximum_motion_iterations": config.maximum_iterations},
        "the final observation-only iteration may only report convergence or stop",
    )
    pre_calculation_passed = bool(
        gates and all(gate["passed"] for gate in gates.values())
    )
    command: Optional[np.ndarray] = None
    if full is not None and pre_calculation_passed and not converged:
        command = gain * full
        norm = float(np.linalg.norm(command))
        if norm > step_limit:
            command = command * (step_limit / norm)
    predicted_delta_error = jacobian @ command if jacobian is not None and command is not None else None
    predicted_error = error + predicted_delta_error if error is not None and predicted_delta_error is not None else None
    predicted_norm = float(np.linalg.norm(predicted_error)) if predicted_error is not None else None
    remaining_path = config.maximum_total_path_mm - float(cumulative_path_mm)
    command_norm = float(np.linalg.norm(command)) if command is not None else None
    gates["correction_computable"] = _gate(converged or command is not None, command, "finite invertible correction", "selected Jacobian must solve")
    gates["step_and_total_path_limit"] = _gate(
        converged or (command_norm is not None and command_norm <= step_limit + 1e-9 and command_norm <= remaining_path + 1e-9),
        {"step_mm": command_norm, "remaining_path_mm": remaining_path},
        {"step_mm": step_limit, "total_path_mm": config.maximum_total_path_mm}, "bounded automatic motion"
    )
    gates["predicted_error_decreases"] = _gate(
        converged or (predicted_norm is not None and error_norm is not None and predicted_norm < error_norm - 1e-6),
        {"before_px": error_norm, "predicted_px": predicted_norm}, "predicted < before", "reject wrong-sign commands"
    )

    planner: Optional[dict[str, Any]] = None
    planner_failures: list[str] = []
    if all(gate["passed"] for gate in gates.values()) and not converged and command is not None and current_joints is not None and current_pose is not None and anchor is not None:
        tier_extent = config.local_extent_mm if tier == "fine_task9" else config.wide_extent_mm
        tier_margin = config.local_margin_mm if tier == "fine_task9" else config.wide_margin_mm
        transient_xy = config.fine_maximum_sequential_transient_xy_mm if tier == "fine_task9" else config.coarse_maximum_sequential_transient_xy_mm
        original_command = command.copy()
        selected_scale: Optional[float] = None
        # Keep bounded backoff for the remaining endpoint, transient-distance,
        # transient-Rz and reachability gates.  Stage7B intentionally does not
        # reject a target solely because the controller's natural J1/J2
        # intermediate path leaves the Jacobian model domain; only the
        # measured start and commanded endpoint must remain in-domain.
        for scale in (1.0, 0.5, 0.25, 0.125):
            candidate = original_command * scale
            if float(np.linalg.norm(candidate)) < config.minimum_effective_step_mm:
                continue
            try:
                candidate_planner = plan_fixed_rz_xy_step(
                    current_joints.astype(float).tolist(),
                    current_pose.astype(float).tolist(),
                    candidate.astype(float).tolist(),
                    anchor_robot_xy_mm=anchor.astype(float).tolist(),
                    local_extent_mm=tier_extent,
                    domain_margin_mm=tier_margin,
                    required_j3_mm=float(local_coordinate["imaging_j3_mm"]),
                    j3_tolerance_mm=config.j3_tolerance_mm,
                    required_rz_deg=float(local_coordinate["rz_deg"]),
                    rz_tolerance_deg=config.rz_tolerance_deg,
                    target_rz_tolerance_deg=config.target_rz_tolerance_deg,
                    max_xy_step_norm_mm=step_limit,
                    max_xy_axis_mm=step_limit,
                    max_sequential_transient_xy_mm=transient_xy,
                    max_sequential_transient_rz_deg=config.maximum_sequential_transient_rz_deg,
                    precompensate_rz=True,
                    enforce_sequential_intermediate_domain=False,
                )
            except (ValueError, ArithmeticError) as exc:
                planner_failures.append(f"scale={scale:g}: {exc}")
                continue
            planner = candidate_planner
            command = candidate
            selected_scale = scale
            break
        if planner is not None:
            planner["stage7b_command_backoff_scale"] = selected_scale
            planner["stage7b_rejected_larger_scales"] = planner_failures
            command_norm = float(np.linalg.norm(command))
            predicted_delta_error = jacobian @ command
            predicted_error = error + predicted_delta_error
            predicted_norm = float(np.linalg.norm(predicted_error))
            gates["step_and_total_path_limit"] = _gate(
                command_norm <= step_limit + 1e-9
                and command_norm <= remaining_path + 1e-9,
                {"step_mm": command_norm, "remaining_path_mm": remaining_path},
                {"step_mm": step_limit, "total_path_mm": config.maximum_total_path_mm},
                "bounded automatic motion after any sequential-path backoff",
            )
            gates["predicted_error_decreases"] = _gate(
                predicted_norm < error_norm - 1e-6,
                {"before_px": error_norm, "predicted_px": predicted_norm},
                "predicted < before",
                "reject wrong-sign commands",
            )
        planner_audit = (planner or {}).get("audit") or {}
        gates["kinematic_planner"] = _gate(
            planner_audit.get("passed") is True,
            {
                "gates": planner_audit.get("gates"),
                "rejected_larger_scales": planner_failures,
                "selected_scale": selected_scale,
            },
            "all pass",
            "shared endpoint and sequential-axis audit",
        )
    elif not converged:
        gates["kinematic_planner"] = _gate(False, None, "planner available", "upstream gate prevented planning")

    passed = bool(gates and all(gate["passed"] for gate in gates.values()))
    if converged and passed:
        decision = "converged"
    elif passed and planner is not None:
        decision = "move"
    else:
        decision = "reject"
    report = {
        "schema_version": 1,
        "stage": "7B_finite_two_tier_closed_loop",
        "iteration_index": int(iteration_index),
        "target_name": config.target_name,
        "configuration": asdict(config),
        "measurement": measurement,
        "model_tier": tier,
        "model_fingerprints": {"fine_task9": _fingerprint(local_calibration), "coarse_task11": _fingerprint(wide_calibration)},
        "current_robot_xy_mm": _json(current_xy),
        "current_offset_xy_mm": _json(offset),
        "median_error_px": _json(error),
        "error_norm_px": error_norm,
        "full_correction_xy_mm": _json(full),
        "remaining_alignment_distance_mm": remaining_alignment_distance_mm,
        "arrival_distance_threshold_mm": config.arrival_distance_threshold_mm,
        "convergence_reason": convergence_reason,
        "commanded_correction_xy_mm": _json(command),
        "predicted_delta_error_px": _json(predicted_delta_error),
        "predicted_error_px": _json(predicted_error),
        "predicted_error_norm_px": predicted_norm,
        "predicted_endpoint_xy_mm": _json(current_xy + command) if current_xy is not None and command is not None else None,
        "cumulative_path_before_mm": float(cumulative_path_mm),
        "cumulative_path_after_mm": float(cumulative_path_mm + (command_norm or 0.0)),
        "previous_response_validation": response_validation,
        "planner": planner,
        "safety_gates": gates,
        "decision": decision,
        "motion_authorized": decision == "move",
        "failure_reasons": [name for name, gate in gates.items() if not gate["passed"]],
    }
    report["proposal_id"] = _fingerprint(report)
    return report


__all__ = [
    "DEFAULT_STAGE7B_CONFIG",
    "Stage7BConfig",
    "aggregate_stage7b_window",
    "build_stage7b_iteration",
]
