"""Pure equations for moved-tray P22 coarse and metric XY positioning.

Unlike the fixed-tray Stage-5 Jacobians, every world target and direction in
this module is derived from the current session's runtime ``^W T_T``.  No
static Stage-2 world origin is used for control.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from scara.pipeline.kinematics import rz_of
from scara.pipeline.xy_correction_planner import plan_fixed_rz_xy_step


COARSE_VISUAL_GAIN = 0.8
# Leave 0.01 mm of command/readback/FK quantization reserve below the
# ActionWorker's immutable 10.00 mm endpoint ceiling.
COARSE_VISUAL_MAXIMUM_STEP_MM = 9.99
COARSE_ENDPOINT_HARD_LIMIT_MM = 10.0
COARSE_SEQUENTIAL_TRANSIENT_XY_LIMIT_MM = 130.0
COARSE_SEQUENTIAL_TRANSIENT_RZ_LIMIT_DEG = 15.0


def _vector(value: Any, length: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != int(length) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain {length} finite values")
    return result


def _matrix(value: Any, shape: tuple[int, int], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite {shape[0]}x{shape[1]} matrix")
    return result


def _gate(passed: bool, actual: Any, limit: str, note: str = "") -> dict[str, Any]:
    return {"passed": bool(passed), "actual": actual, "limit": str(limit), "note": str(note)}


def registration_transform_W_T(registration: Mapping[str, Any]) -> np.ndarray:
    if registration.get("status") != "success":
        raise ValueError("runtime Tray registration is not successful")
    transform = _matrix(registration.get("transform_W_T"), (4, 4), "transform_W_T")
    if abs(np.linalg.det(transform[:3, :3])) < 1e-9:
        raise ValueError("runtime Tray registration rotation is singular")
    return transform


def registered_slot_world_xy_mm(
    geometry: Mapping[str, Any],
    target_name: str,
    registration: Mapping[str, Any],
) -> np.ndarray:
    """Transform one Stage-2 *internal* slot coordinate with runtime ``^W T_T``."""

    slots = geometry.get("slots") or {}
    if target_name not in slots:
        raise KeyError(f"unknown Tray slot {target_name}")
    point_T = _vector(slots[target_name], 3, f"slot {target_name}")
    transform = registration_transform_W_T(registration)
    point_W = transform @ np.concatenate((point_T, [1.0]))
    return point_W[:2] / point_W[3]


def registered_tray_delta_to_world_xy_mm(
    delta_T_xy_mm: Sequence[float], registration: Mapping[str, Any]
) -> np.ndarray:
    delta = _vector(delta_T_xy_mm, 2, "delta_T_xy_mm")
    transform = registration_transform_W_T(registration)
    return transform[:2, :2] @ delta


def registered_tray_workspace(
    geometry: Mapping[str, Any],
    registration: Mapping[str, Any],
    *,
    margin_mm: float = 10.0,
) -> dict[str, Any]:
    names = sorted((geometry.get("slots") or {}).keys())
    if len(names) != 36:
        raise ValueError("Tray geometry must contain exactly 36 slots")
    world = np.asarray(
        [registered_slot_world_xy_mm(geometry, name, registration) for name in names]
    )
    low = np.min(world, axis=0) - float(margin_mm)
    high = np.max(world, axis=0) + float(margin_mm)
    return {
        "slot_world_xy_mm": {
            name: world[index].astype(float).tolist() for index, name in enumerate(names)
        },
        "minimum_world_xy_mm": low.astype(float).tolist(),
        "maximum_world_xy_mm": high.astype(float).tolist(),
        "margin_mm": float(margin_mm),
    }


def metric_step_policy(error_norm_mm: float) -> dict[str, float]:
    norm = float(error_norm_mm)
    if not math.isfinite(norm) or norm < 0.0:
        raise ValueError("error_norm_mm must be finite and non-negative")
    if norm > 5.0:
        return {"gain": 0.8, "maximum_step_mm": 2.0}
    if norm > 1.5:
        return {"gain": 0.7, "maximum_step_mm": 0.75}
    return {"gain": 0.5, "maximum_step_mm": 0.25}


def aggregate_metric_window(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(samples)
    errors = np.asarray(
        [_vector(row.get("metric_error_T_mm"), 3, "metric_error_T_mm") for row in rows]
    )
    if errors.shape != (5, 3):
        raise ValueError("metric window requires exactly five finite errors")
    median = np.median(errors, axis=0)
    deviations = np.linalg.norm(errors[:, :2] - median[:2], axis=1)
    norms = np.linalg.norm(errors[:, :2], axis=1)
    return {
        "median_error_T_mm": median.astype(float).tolist(),
        "median_error_norm_mm": float(np.linalg.norm(median[:2])),
        "window_rms_dispersion_mm": float(math.sqrt(float(np.mean(deviations * deviations)))),
        "window_max_deviation_mm": float(np.max(deviations)),
        "per_frame_error_norm_mm": norms.astype(float).tolist(),
        "maximum_frame_error_norm_mm": float(np.max(norms)),
    }


def final_hold_gates(window: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    median_norm = float(window.get("median_error_norm_mm", math.inf))
    rms = float(window.get("window_rms_dispersion_mm", math.inf))
    maximum = float(window.get("maximum_frame_error_norm_mm", math.inf))
    return {
        "median_tray_xy_error": _gate(median_norm <= 0.60, median_norm, "<=0.60 mm"),
        "window_rms_dispersion": _gate(rms <= 0.30, rms, "<=0.30 mm"),
        "all_five_frame_error": _gate(maximum <= 1.00, maximum, "<=1.00 mm"),
    }


def plan_registered_xy_step(
    current_joints: Sequence[float],
    current_pose: Sequence[float],
    command_world_xy_mm: Sequence[float],
    registration: Mapping[str, Any],
    geometry: Mapping[str, Any],
    *,
    required_j3_mm: float,
    required_rz_deg: float,
    maximum_step_mm: float,
) -> dict[str, Any]:
    """Plan one fixed-J3/Rz endpoint inside the runtime-transformed Tray envelope."""

    target_world = registered_slot_world_xy_mm(geometry, "P22", registration)
    return plan_fixed_rz_xy_step(
        _vector(current_joints, 4, "current_joints").astype(float).tolist(),
        _vector(current_pose, 6, "current_pose").astype(float).tolist(),
        _vector(command_world_xy_mm, 2, "command_world_xy_mm").astype(float).tolist(),
        anchor_robot_xy_mm=target_world.astype(float).tolist(),
        local_extent_mm=130.0,
        domain_margin_mm=5.0,
        required_j3_mm=float(required_j3_mm),
        j3_tolerance_mm=0.20,
        required_rz_deg=float(required_rz_deg),
        rz_tolerance_deg=0.30,
        target_rz_tolerance_deg=0.15,
        max_xy_step_norm_mm=float(maximum_step_mm) + 1e-6,
        max_xy_axis_mm=float(maximum_step_mm) + 1e-6,
        max_sequential_transient_xy_mm=(
            COARSE_SEQUENTIAL_TRANSIENT_XY_LIMIT_MM
            if float(maximum_step_mm) > 2.0
            else 4.0
        ),
        max_sequential_transient_rz_deg=(
            COARSE_SEQUENTIAL_TRANSIENT_RZ_LIMIT_DEG
            if float(maximum_step_mm) > 2.0
            else 1.0
        ),
        precompensate_rz=True,
        enforce_sequential_intermediate_domain=False,
    )


def build_registered_control_proposal(
    samples: Sequence[Mapping[str, Any]],
    request: Mapping[str, Any],
    registration: Mapping[str, Any],
    geometry: Mapping[str, Any],
    *,
    phase: str,
    movement_index: int,
    cumulative_path_mm: float,
    previous_iteration: Mapping[str, Any] | None = None,
    coarse_goal_world_xy_mm: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Build one coarse or metric proposal plus every calculation gate."""

    rows = list(samples)
    ids = [str(row.get("measurement_id") or "") for row in rows]
    accepted = [row.get("accepted") is True for row in rows]
    targets = [str(row.get("target_name") or "") for row in rows]
    state_ages = [float(row.get("robot_state_age_s", math.inf)) for row in rows]
    xy = np.asarray([_vector(row.get("current_robot_xy_mm"), 2, "sample XY") for row in rows])
    joints = np.asarray([_vector(row.get("current_joints"), 4, "sample joints") for row in rows])
    poses = np.asarray([_vector(row.get("current_pose"), 6, "sample pose") for row in rows])
    requested_at = float(request.get("requested_monotonic_s", math.inf))
    limits = request.get("limits")
    if not isinstance(limits, Mapping):
        raise ValueError("runtime request must contain a limits mapping")
    required_j3_mm = float(limits.get("required_j3_mm"))
    required_rz_deg = float(limits.get("required_rz_deg"))
    if not math.isfinite(required_j3_mm) or not math.isfinite(required_rz_deg):
        raise ValueError("runtime request limits must contain finite required_j3_mm and required_rz_deg")
    captures = [float(row.get("captured_monotonic_s", math.nan)) for row in rows]
    window = aggregate_metric_window(rows)
    median_xy = np.median(xy, axis=0)
    median_joints = np.median(joints, axis=0)
    median_pose = np.median(poses, axis=0)
    xy_spread = float(np.max(np.linalg.norm(xy - median_xy, axis=1)))
    joint_spread = float(np.max(np.ptp(joints, axis=0)))
    unique_post = (
        len(rows) == 5 and all(ids) and len(set(ids)) == 5
        and all(math.isfinite(value) and value >= requested_at for value in captures)
    )
    target_world = registered_slot_world_xy_mm(geometry, "P22", registration)
    coarse_goal = (
        target_world
        if coarse_goal_world_xy_mm is None
        else _vector(coarse_goal_world_xy_mm, 2, "coarse_goal_world_xy_mm")
    )
    coarse_delta = coarse_goal - median_xy
    coarse_distance = float(np.linalg.norm(target_world - median_xy))
    coarse_waypoint_distance = float(np.linalg.norm(coarse_delta))
    error_T = np.asarray(window["median_error_T_mm"], dtype=np.float64)
    error_norm = float(window["median_error_norm_mm"])
    if phase == "coarse":
        maximum_step = COARSE_VISUAL_MAXIMUM_STEP_MM
        command = coarse_delta.copy()
        if coarse_waypoint_distance > maximum_step:
            command *= maximum_step / coarse_waypoint_distance
        gain = COARSE_VISUAL_GAIN
        predicted_error = None
    elif phase == "metric":
        policy = metric_step_policy(error_norm)
        gain = float(policy["gain"])
        maximum_step = float(policy["maximum_step_mm"])
        raw = gain * registered_tray_delta_to_world_xy_mm(error_T[:2], registration)
        raw_norm = float(np.linalg.norm(raw))
        command = raw if raw_norm <= maximum_step else raw * (maximum_step / raw_norm)
        applied_T = registration_transform_W_T(registration)[:2, :2].T @ command
        predicted_error = error_T[:2] - applied_T
    else:
        raise ValueError("phase must be coarse or metric")
    command_norm = float(np.linalg.norm(command))
    previous_error = None if previous_iteration is None else float(previous_iteration.get("median_error_norm_mm", math.inf))
    previous_endpoint = None if previous_iteration is None else previous_iteration.get("predicted_endpoint_xy_mm")
    tracking_error = (
        0.0
        if previous_endpoint is None
        else float(np.linalg.norm(median_xy - _vector(previous_endpoint, 2, "previous endpoint")))
    )
    improvement = None if previous_error is None or not math.isfinite(previous_error) else previous_error - error_norm
    predicted_endpoint = median_xy + command
    workspace = registered_tray_workspace(geometry, registration)
    workspace_low = _vector(workspace["minimum_world_xy_mm"], 2, "workspace minimum")
    workspace_high = _vector(workspace["maximum_world_xy_mm"], 2, "workspace maximum")
    current_inside_workspace = bool(
        np.all(median_xy >= workspace_low - 1e-9)
        and np.all(median_xy <= workspace_high + 1e-9)
    )
    endpoint_inside_workspace = bool(
        np.all(predicted_endpoint >= workspace_low - 1e-9)
        and np.all(predicted_endpoint <= workspace_high + 1e-9)
    )
    plan = plan_registered_xy_step(
        median_joints,
        median_pose,
        command,
        registration,
        geometry,
        required_j3_mm=required_j3_mm,
        required_rz_deg=required_rz_deg,
        maximum_step_mm=maximum_step,
    )
    gates = {
        "five_unique_post_request_frames": _gate(unique_post, {"ids": ids, "captures": captures}, "5 unique after request"),
        "stage3_pass": _gate(len(rows) == 5 and all(accepted), accepted, "all true"),
        "target_p22": _gate(len(rows) == 5 and all(value == "P22" for value in targets), targets, "all P22"),
        "robot_state_freshness": _gate(len(rows) == 5 and all(0.0 <= value <= 1.0 for value in state_ages), state_ages, "each <=1.0 s"),
        "robot_stationary_xy": _gate(xy_spread <= 0.20, xy_spread, "<=0.20 mm"),
        "robot_stationary_joints": _gate(joint_spread <= 0.20, joint_spread, "<=0.20 deg/mm"),
        "current_inside_registered_tray_workspace": _gate(
            current_inside_workspace,
            median_xy.astype(float).tolist(),
            {
                "minimum_world_xy_mm": workspace_low.astype(float).tolist(),
                "maximum_world_xy_mm": workspace_high.astype(float).tolist(),
            },
        ),
        "endpoint_inside_registered_tray_workspace": _gate(
            endpoint_inside_workspace,
            predicted_endpoint.astype(float).tolist(),
            {
                "minimum_world_xy_mm": workspace_low.astype(float).tolist(),
                "maximum_world_xy_mm": workspace_high.astype(float).tolist(),
            },
        ),
        "error_window_stable": _gate(float(window["window_rms_dispersion_mm"]) <= 0.50, window["window_rms_dispersion_mm"], "<=0.50 mm RMS"),
        "movement_count_limit": _gate(
            phase == "coarse" or int(movement_index) < 32,
            int(movement_index),
            "coarse route exempt; metric movement index <32",
        ),
        "session_path_limit": _gate(
            phase == "coarse" or float(cumulative_path_mm) + command_norm <= 50.0,
            float(cumulative_path_mm) + command_norm,
            "coarse route exempt; metric cumulative path <=50.0 mm",
        ),
        "previous_command_tracking": _gate(previous_endpoint is None or tracking_error <= 0.15, tracking_error, "<=0.15 mm"),
        "previous_error_decreased": _gate(previous_error is None or improvement > 0.0, improvement, ">0 mm improvement"),
        "predicted_error_decreases": _gate(phase == "coarse" or predicted_error is not None and float(np.linalg.norm(predicted_error)) < error_norm, None if predicted_error is None else float(np.linalg.norm(predicted_error)), f"<{error_norm:.6f} mm"),
        "kinematic_planner": _gate((plan.get("audit") or {}).get("passed") is True, (plan.get("audit") or {}).get("passed"), "true"),
    }
    authorized = all(gate["passed"] for gate in gates.values())
    return {
        "proposal_id": f"moved-tray-{phase}-{movement_index + 1}-" + "-".join(ids),
        "target_name": "P22",
        "phase": f"moved_tray_{phase}",
        "motion_authorized": bool(authorized),
        "measurement_ids": ids,
        "median_metric_error_T_mm": error_T.astype(float).tolist(),
        "median_error_norm_mm": error_norm,
        "error_window": window,
        "current_robot_xy_mm": median_xy.astype(float).tolist(),
        "dynamic_p22_world_xy_mm": target_world.astype(float).tolist(),
        "coarse_distance_to_p22_world_mm": coarse_distance,
        "coarse_waypoint_world_xy_mm": coarse_goal.astype(float).tolist(),
        "coarse_distance_to_waypoint_mm": coarse_waypoint_distance,
        "gain": gain,
        "maximum_step_mm": maximum_step,
        "calculation": {
            "commanded_correction_xy_mm": command.astype(float).tolist(),
            "predicted_error_T_xy_mm": None if predicted_error is None else predicted_error.astype(float).tolist(),
        },
        "commanded_correction_xy_mm": command.astype(float).tolist(),
        "predicted_endpoint_xy_mm": predicted_endpoint.astype(float).tolist(),
        "previous_command_tracking_error_mm": tracking_error,
        "previous_error_improvement_mm": improvement,
        "movement_index": int(movement_index),
        "cumulative_path_before_mm": float(cumulative_path_mm),
        "cumulative_path_after_mm": float(cumulative_path_mm + command_norm),
        "planner": plan,
        "safety_gates": gates,
        "failure_reasons": [name for name, gate in gates.items() if not gate["passed"]],
        "current_rz_deg": float(rz_of(median_joints[0], median_joints[1], median_joints[3])),
    }


__all__ = [
    "COARSE_ENDPOINT_HARD_LIMIT_MM",
    "COARSE_VISUAL_GAIN",
    "COARSE_VISUAL_MAXIMUM_STEP_MM",
    "aggregate_metric_window",
    "build_registered_control_proposal",
    "final_hold_gates",
    "metric_step_policy",
    "registered_slot_world_xy_mm",
    "registered_tray_delta_to_world_xy_mm",
    "registered_tray_workspace",
    "registration_transform_W_T",
]
