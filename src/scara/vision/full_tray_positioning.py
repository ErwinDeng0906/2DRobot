"""Pure geometry and planning equations for full-tray XY positioning.

The module has no Qt, camera-thread, controller, DO, or vacuum dependency.
It connects three existing coordinate systems:

``T``
    Tray coordinates defined by Stage 2.
``C``
    Camera-1 coordinates estimated by Stage 3 as ``T_C_T`` (Tray to camera).
``W``
    The SCARA controller/world XY plane used by inverse kinematics.

For a fixed-height Stage-4 suction point ``p_C_S`` and one Stage-3 pose,

    p_T_S = inv(T_C_T) @ p_C_S
    e_T   = p_T_target - p_T_S
    delta_W_xy = R_W_T[:2,:2] @ e_T_xy

The first move is feed-forward geometry: one absolute slot target is computed
from P00/Tray geometry.  Because the current controller executes J1 then J2,
the single coarse phase is internally subdivided into short, audited IK
waypoints; this avoids a large unobserved sweep between two valid endpoints.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from scara.pipeline.kinematics import fk_wrist, rz_of, solve_joints
from scara.pipeline.xy_correction_planner import (
    angular_difference_deg,
    audit_fixed_rz_xy_target,
    plan_fixed_rz_xy_step,
)


GEOMETRY_COARSE_ENDPOINT_STEP_MM = 2.0
GEOMETRY_COARSE_MAX_SEQUENTIAL_XY_MM = 4.0
GEOMETRY_COARSE_MAX_SEQUENTIAL_RZ_DEG = 1.0
GEOMETRY_CORRECTION_MAX_STEP_MM = 3.0
GEOMETRY_CORRECTION_LOCAL_EXTENT_MM = 5.0
GEOMETRY_CORRECTION_DOMAIN_MARGIN_MM = 0.2
GEOMETRY_CORRECTION_MAX_TRANSIENT_XY_MM = 6.0
GEOMETRY_CORRECTION_MAX_TRANSIENT_RZ_DEG = 1.0


def _vector(value: Any, length: int, label: str) -> np.ndarray:
    candidate = np.asarray(value, dtype=np.float64).reshape(-1)
    if candidate.size != int(length) or not np.all(np.isfinite(candidate)):
        raise ValueError(f"{label} must contain {length} finite values")
    return candidate


def _matrix(value: Any, shape: tuple[int, int], label: str) -> np.ndarray:
    candidate = np.asarray(value, dtype=np.float64)
    if candidate.shape != shape or not np.all(np.isfinite(candidate)):
        raise ValueError(f"{label} must be a finite {shape[0]}x{shape[1]} matrix")
    return candidate


def _gate(
    passed: bool,
    *,
    actual: Any = None,
    limit: Any = None,
    note: str = "",
) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "actual": actual,
        "limit": limit,
        "note": str(note),
    }


def slot_world_xy_mm(
    geometry: Mapping[str, Any], target_name: str
) -> np.ndarray:
    """Convert one Stage-2 slot centre from Tray coordinates to world XY."""

    frame = geometry.get("tray_frame") or {}
    rotation = _matrix(
        frame.get("rotation_mechanical_from_tray"),
        (3, 3),
        "rotation_mechanical_from_tray",
    )
    origin = _vector(
        frame.get("origin_mechanical_xy_mm"), 2, "origin_mechanical_xy_mm"
    )
    slots = geometry.get("slots") or {}
    if str(target_name) not in slots:
        raise KeyError(f"unknown Tray slot {target_name}")
    point_T = _vector(slots[str(target_name)], 3, f"slot {target_name}")
    return origin + rotation[:2, :2] @ point_T[:2]


def tray_delta_to_world_xy_mm(
    geometry: Mapping[str, Any], delta_T_xy_mm: Sequence[float]
) -> np.ndarray:
    """Rotate a Tray-frame displacement into controller/world XY."""

    rotation = _matrix(
        (geometry.get("tray_frame") or {}).get(
            "rotation_mechanical_from_tray"
        ),
        (3, 3),
        "rotation_mechanical_from_tray",
    )
    delta = _vector(delta_T_xy_mm, 2, "delta_T_xy_mm")
    return rotation[:2, :2] @ delta


def metric_suction_error_in_tray(
    transform_C_T: Sequence[Sequence[float]],
    suction_point_C_mm: Sequence[float],
    target_point_T_mm: Sequence[float],
) -> dict[str, list[float]]:
    """Return the Stage-3/Stage-4 suction point and target error in Tray mm.

    ``transform_C_T`` is the Stage-3 homogeneous transform ``^C T_T``.
    The Stage-4 point is converted back into Tray coordinates with its inverse.
    """

    transform = _matrix(transform_C_T, (4, 4), "transform_C_T")
    if abs(float(np.linalg.det(transform[:3, :3]))) < 1e-9:
        raise ValueError("transform_C_T rotation is singular")
    suction_C = _vector(suction_point_C_mm, 3, "suction_point_C_mm")
    target_T = _vector(target_point_T_mm, 3, "target_point_T_mm")
    suction_h = np.concatenate((suction_C, [1.0]))
    suction_T = np.linalg.inv(transform) @ suction_h
    if abs(float(suction_T[3])) < 1e-12:
        raise ValueError("homogeneous suction point has zero scale")
    suction_T = suction_T[:3] / suction_T[3]
    error_T = target_T - suction_T
    return {
        "suction_point_T_mm": suction_T.astype(float).tolist(),
        "target_point_T_mm": target_T.astype(float).tolist(),
        "metric_error_T_mm": error_T.astype(float).tolist(),
    }


def _tray_xy_of_world(
    geometry: Mapping[str, Any], world_xy_mm: Sequence[float]
) -> np.ndarray:
    frame = geometry.get("tray_frame") or {}
    rotation = _matrix(
        frame.get("rotation_mechanical_from_tray"),
        (3, 3),
        "rotation_mechanical_from_tray",
    )
    origin = _vector(
        frame.get("origin_mechanical_xy_mm"), 2, "origin_mechanical_xy_mm"
    )
    world = _vector(world_xy_mm, 2, "world_xy_mm")
    return rotation[:2, :2].T @ (world - origin)


def plan_geometry_coarse_route(
    initial_joints: Sequence[float],
    initial_pose: Sequence[float],
    geometry: Mapping[str, Any],
    target_name: str,
    *,
    required_j3_mm: float,
    required_rz_deg: float,
    endpoint_step_mm: float = GEOMETRY_COARSE_ENDPOINT_STEP_MM,
    tray_workspace_margin_mm: float = 10.0,
) -> dict[str, Any]:
    """Plan one feed-forward slot target as short fixed-height IK waypoints."""

    joints = _vector(initial_joints, 4, "initial_joints")
    pose = _vector(initial_pose, 6, "initial_pose")
    endpoint_step_mm = float(endpoint_step_mm)
    if not math.isfinite(endpoint_step_mm) or endpoint_step_mm <= 0.0:
        raise ValueError("endpoint_step_mm must be positive and finite")
    target_world = slot_world_xy_mm(geometry, target_name)
    current_world = pose[:2]
    fk_current = np.asarray(fk_wrist(joints[0], joints[1]), dtype=np.float64)
    start_tray = _tray_xy_of_world(geometry, current_world)
    slot_points = np.asarray(
        [value[:2] for value in (geometry.get("slots") or {}).values()],
        dtype=np.float64,
    )
    if slot_points.shape != (36, 2):
        raise ValueError("Tray geometry must contain exactly 36 finite slot centres")
    low = np.min(slot_points, axis=0) - float(tray_workspace_margin_mm)
    high = np.max(slot_points, axis=0) + float(tray_workspace_margin_mm)
    start_in_workspace = bool(np.all(start_tray >= low) and np.all(start_tray <= high))
    start_rz = rz_of(joints[0], joints[1], joints[3])
    top_gates = {
        "controller_pose_matches_fk": _gate(
            float(np.linalg.norm(fk_current - current_world)) <= 0.20,
            actual=float(np.linalg.norm(fk_current - current_world)),
            limit="<=0.20 mm",
        ),
        "current_j3_at_imaging_height": _gate(
            abs(float(joints[2]) - float(required_j3_mm)) <= 0.20,
            actual=abs(float(joints[2]) - float(required_j3_mm)),
            limit="<=0.20 mm",
        ),
        "current_rz_matches_calibration": _gate(
            angular_difference_deg(start_rz, required_rz_deg) <= 0.30,
            actual=angular_difference_deg(start_rz, required_rz_deg),
            limit="<=0.30 deg",
        ),
        "current_xy_inside_full_tray_workspace": _gate(
            start_in_workspace,
            actual=start_tray.astype(float).tolist(),
            limit={
                "minimum_T_xy_mm": low.astype(float).tolist(),
                "maximum_T_xy_mm": high.astype(float).tolist(),
            },
        ),
    }
    if not all(gate["passed"] for gate in top_gates.values()):
        failed = ", ".join(name for name, gate in top_gates.items() if not gate["passed"])
        raise ValueError(f"geometry coarse start failed safety audit: {failed}")

    displacement = target_world - current_world
    distance = float(np.linalg.norm(displacement))
    segment_count = max(1, int(math.ceil(distance / endpoint_step_mm)))
    # The broad domain is used only to exercise the common endpoint audit.  It
    # covers the complete 6x6 Tray plus the explicit 10 mm start margin.
    anchor = target_world
    domain_extent = 120.0
    previous_joints = joints.astype(float).tolist()
    previous_pose = pose.astype(float).tolist()
    waypoints: list[dict[str, Any]] = []
    for index in range(1, segment_count + 1):
        fraction = float(index) / float(segment_count)
        endpoint = current_world + fraction * displacement
        target_joints = solve_joints(
            float(endpoint[0]),
            float(endpoint[1]),
            float(joints[2]),
            rz_deg=float(required_rz_deg),
            ref_joints=previous_joints,
        )
        if target_joints is None:
            raise ValueError(f"coarse route waypoint {index} has no IK solution")
        audit = audit_fixed_rz_xy_target(
            previous_joints,
            previous_pose,
            target_joints,
            anchor_robot_xy_mm=anchor.astype(float).tolist(),
            local_extent_mm=domain_extent,
            domain_margin_mm=1.0,
            required_j3_mm=required_j3_mm,
            j3_tolerance_mm=0.20,
            required_rz_deg=required_rz_deg,
            rz_tolerance_deg=0.30,
            target_rz_tolerance_deg=0.15,
            max_xy_step_norm_mm=endpoint_step_mm + 0.01,
            max_xy_axis_mm=endpoint_step_mm + 0.01,
            max_sequential_transient_xy_mm=GEOMETRY_COARSE_MAX_SEQUENTIAL_XY_MM,
            max_sequential_transient_rz_deg=GEOMETRY_COARSE_MAX_SEQUENTIAL_RZ_DEG,
            precompensate_rz=False,
            enforce_sequential_intermediate_domain=False,
        )
        if audit.get("passed") is not True:
            failed = ", ".join(
                name
                for name, gate in (audit.get("gates") or {}).items()
                if gate.get("passed") is not True
            )
            raise ValueError(f"coarse route waypoint {index} failed audit: {failed}")
        waypoints.append(
            {
                "index": index,
                "fraction": fraction,
                "target_world_xy_mm": audit["target_xy_mm"],
                "target_joints": audit["target_joints"],
                "endpoint_step_norm_mm": audit["step_norm_mm"],
                "sequential_transient_max_mm": audit[
                    "sequential_transient_max_mm"
                ],
                "sequential_transient_rz_max_deg": audit[
                    "sequential_transient_rz_max_deg"
                ],
                "audit": audit,
            }
        )
        previous_joints = list(audit["target_joints"])
        previous_pose = [
            float(audit["target_xy_mm"][0]),
            float(audit["target_xy_mm"][1]),
            float(pose[2]),
            float(pose[3]),
            float(pose[4]),
            float(required_rz_deg),
        ]
    return {
        "target_name": str(target_name),
        "initial_world_xy_mm": current_world.astype(float).tolist(),
        "initial_tray_xy_mm": start_tray.astype(float).tolist(),
        "geometry_target_world_xy_mm": target_world.astype(float).tolist(),
        "coarse_distance_mm": distance,
        "endpoint_step_limit_mm": endpoint_step_mm,
        "waypoint_count": len(waypoints),
        "top_level_gates": top_gates,
        "waypoints": waypoints,
    }


def build_metric_geometry_correction(
    samples: Sequence[Mapping[str, Any]],
    request: Mapping[str, Any],
    geometry: Mapping[str, Any],
    *,
    target_name: str,
    transition_anchor_xy_mm: Sequence[float],
) -> dict[str, Any]:
    """Aggregate five Stage-3 metric measurements and plan one correction."""

    request_state = request.get("controller_state") or {}
    request_joints = _vector(request_state.get("joints"), 4, "request joints")
    request_pose = _vector(request_state.get("pose"), 6, "request pose")
    requested_at = float(request.get("requested_monotonic_s", math.nan))
    if not math.isfinite(requested_at):
        raise ValueError("request is missing requested_monotonic_s")
    rows = list(samples)
    measurement_ids = [str(row.get("measurement_id") or "") for row in rows]
    accepted = [row for row in rows if row.get("accepted") is True]
    errors = []
    robot_xy = []
    robot_joints = []
    state_ages = []
    captured = []
    for row in accepted:
        errors.append(_vector(row.get("metric_error_T_mm"), 3, "metric error"))
        robot_xy.append(_vector(row.get("current_robot_xy_mm"), 2, "sample robot XY"))
        robot_joints.append(_vector(row.get("current_joints"), 4, "sample joints"))
        state_ages.append(float(row.get("robot_state_age_s", math.inf)))
        captured.append(float(row.get("captured_monotonic_s", -math.inf)))
    error_array = np.asarray(errors, dtype=np.float64) if errors else np.empty((0, 3))
    xy_array = np.asarray(robot_xy, dtype=np.float64) if robot_xy else np.empty((0, 2))
    joints_array = np.asarray(robot_joints, dtype=np.float64) if robot_joints else np.empty((0, 4))
    if error_array.shape[0] >= 1:
        median_error = np.median(error_array, axis=0)
        deviations = np.linalg.norm(error_array[:, :2] - median_error[:2], axis=1)
        stable_rms = float(math.sqrt(float(np.mean(deviations**2))))
        stable_max = float(np.max(deviations))
    else:
        median_error = np.array([math.nan, math.nan, math.nan])
        stable_rms = math.inf
        stable_max = math.inf
    xy_dispersion = (
        float(np.max(np.linalg.norm(xy_array - np.median(xy_array, axis=0), axis=1)))
        if len(xy_array)
        else math.inf
    )
    joint_dispersion = (
        float(np.max(np.abs(joints_array - np.median(joints_array, axis=0))))
        if len(joints_array)
        else math.inf
    )
    median_xy = np.median(xy_array, axis=0) if len(xy_array) else np.array([math.nan, math.nan])
    median_joints = (
        np.median(joints_array, axis=0)
        if len(joints_array)
        else np.full(4, math.nan)
    )
    state_xy_delta = float(np.linalg.norm(median_xy - request_pose[:2]))
    state_joint_delta = float(np.max(np.abs(median_joints - request_joints)))
    gates = {
        "five_fresh_stage3_measurements": _gate(
            len(rows) == 5
            and len(accepted) == 5
            and len(set(measurement_ids)) == 5
            and all(measurement_ids)
            and all(value >= requested_at for value in captured),
            actual={"rows": len(rows), "accepted": len(accepted), "unique": len(set(measurement_ids))},
            limit="exactly 5 accepted, unique, post-request frames",
        ),
        "target_consistency": _gate(
            all(str(row.get("target_name") or "") == str(target_name) for row in rows),
            actual=sorted({str(row.get("target_name") or "") for row in rows}),
            limit=str(target_name),
        ),
        "robot_state_freshness": _gate(
            bool(state_ages) and all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in state_ages),
            actual=max(state_ages, default=math.inf),
            limit="<=1.0 s",
        ),
        "metric_error_window_stable": _gate(
            stable_rms <= 1.5 and stable_max <= 3.0,
            actual={"rms_mm": stable_rms, "maximum_deviation_mm": stable_max},
            limit="RMS<=1.5 mm and maximum<=3.0 mm",
        ),
        "robot_stationary_window": _gate(
            xy_dispersion <= 0.20 and joint_dispersion <= 0.20,
            actual={"xy_dispersion_mm": xy_dispersion, "joint_dispersion": joint_dispersion},
            limit="XY<=0.20 mm and joints<=0.20",
        ),
        "request_state_matches_measurement": _gate(
            state_xy_delta <= 0.20 and state_joint_delta <= 0.20,
            actual={"xy_mm": state_xy_delta, "maximum_joint": state_joint_delta},
            limit="XY<=0.20 mm and joints<=0.20",
        ),
        "metric_error_is_finite": _gate(
            bool(np.all(np.isfinite(median_error))),
            actual=median_error.astype(float).tolist(),
            limit="finite XYZ error",
        ),
        "suction_on_declared_working_plane": _gate(
            math.isfinite(float(median_error[2])) and abs(float(median_error[2])) <= 2.0,
            actual=float(median_error[2]),
            limit="absolute Z residual<=2.0 mm; XY-only session",
        ),
    }
    world_command = tray_delta_to_world_xy_mm(geometry, median_error[:2])
    command_norm = float(np.linalg.norm(world_command))
    gates["one_geometry_correction_limit"] = _gate(
        command_norm <= GEOMETRY_CORRECTION_MAX_STEP_MM
        and float(np.max(np.abs(world_command))) <= GEOMETRY_CORRECTION_MAX_STEP_MM,
        actual={"world_xy_mm": world_command.astype(float).tolist(), "norm_mm": command_norm},
        limit=f"norm and each axis<={GEOMETRY_CORRECTION_MAX_STEP_MM:.1f} mm",
    )
    planner: dict[str, Any] | None = None
    planner_error = ""
    if all(gate["passed"] for gate in gates.values()):
        try:
            limits = request.get("limits") or {}
            planner = plan_fixed_rz_xy_step(
                request_joints.astype(float).tolist(),
                request_pose.astype(float).tolist(),
                world_command.astype(float).tolist(),
                anchor_robot_xy_mm=transition_anchor_xy_mm,
                local_extent_mm=float(limits["local_extent_mm"]),
                domain_margin_mm=float(limits["domain_margin_mm"]),
                required_j3_mm=float(limits["required_j3_mm"]),
                j3_tolerance_mm=float(limits["j3_tolerance_mm"]),
                required_rz_deg=float(limits["required_rz_deg"]),
                rz_tolerance_deg=float(limits["rz_tolerance_deg"]),
                target_rz_tolerance_deg=float(limits["target_rz_tolerance_deg"]),
                max_xy_step_norm_mm=float(limits["max_xy_step_norm_mm"]),
                max_xy_axis_mm=float(limits["max_xy_axis_mm"]),
                max_sequential_transient_xy_mm=float(limits["max_sequential_transient_xy_mm"]),
                max_sequential_transient_rz_deg=float(limits["max_sequential_transient_rz_deg"]),
                precompensate_rz=bool(limits.get("precompensate_rz")),
                enforce_sequential_intermediate_domain=False,
                allow_rejected_audit=True,
            )
        except Exception as exc:  # noqa: BLE001 - returned as fail-closed evidence
            planner_error = str(exc)
    planner_passed = bool(planner and (planner.get("audit") or {}).get("passed") is True)
    gates["kinematic_planner"] = _gate(
        planner_passed,
        actual=planner_error or ((planner or {}).get("audit") or {}).get("passed"),
        limit="shared fixed-J3/fixed-Rz planner PASS",
    )
    predicted_endpoint = (
        np.asarray((planner or {}).get("audit", {}).get("target_xy_mm"), dtype=np.float64)
        if planner_passed
        else np.array([math.nan, math.nan])
    )
    transition_anchor = _vector(
        transition_anchor_xy_mm, 2, "transition_anchor_xy_mm"
    )
    endpoint_delta = predicted_endpoint - transition_anchor
    gates["enters_task9_local_domain"] = _gate(
        planner_passed and float(np.max(np.abs(endpoint_delta))) <= 1.8,
        actual=endpoint_delta.astype(float).tolist(),
        limit="each axis<=1.8 mm from Task9 P22 anchor",
    )
    authorized = all(gate["passed"] for gate in gates.values())
    proposal_seed = json.dumps(
        {
            "request_id": request.get("request_id"),
            "measurements": measurement_ids,
            "command": world_command.astype(float).tolist(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "proposal_id": "full-tray-geometry-" + hashlib.sha256(proposal_seed).hexdigest()[:20],
        "target_name": str(target_name),
        "phase": "stage3_metric_geometry_correction",
        "motion_authorized": bool(authorized),
        "measurement_ids": measurement_ids,
        "median_metric_error_T_mm": median_error.astype(float).tolist(),
        "commanded_correction_xy_mm": world_command.astype(float).tolist(),
        "current_robot_xy_mm": request_pose[:2].astype(float).tolist(),
        "predicted_endpoint_xy_mm": predicted_endpoint.astype(float).tolist(),
        "transition_anchor_delta_xy_mm": endpoint_delta.astype(float).tolist(),
        "safety_gates": gates,
        "failure_reasons": [name for name, gate in gates.items() if not gate["passed"]],
        "planner": planner,
        "calculation": {
            "commanded_correction_xy_mm": world_command.astype(float).tolist(),
            "predicted_endpoint_xy_mm": predicted_endpoint.astype(float).tolist(),
            "equations": [
                "p_T_S = inverse(T_C_T) * p_C_S",
                "e_T = p_T_target - p_T_S",
                "delta_W_xy = R_W_T[:2,:2] * median(e_T_xy)",
            ],
        },
    }


__all__ = [
    "GEOMETRY_COARSE_ENDPOINT_STEP_MM",
    "GEOMETRY_CORRECTION_MAX_STEP_MM",
    "build_metric_geometry_correction",
    "metric_suction_error_in_tray",
    "plan_geometry_coarse_route",
    "slot_world_xy_mm",
    "tray_delta_to_world_xy_mm",
]
