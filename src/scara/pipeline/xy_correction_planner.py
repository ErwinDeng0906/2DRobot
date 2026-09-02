"""Pure planning and safety audit for small fixed-pose SCARA XY corrections.

This module is the shared mathematical boundary between Task 9 calibration
motion and the Stage-7A supervised single-step correction.  It contains no Qt,
controller, camera, or vacuum dependency.  A caller first creates a target with
``plan_fixed_rz_xy_step`` and the hardware-owning worker independently repeats
the checks with ``audit_fixed_rz_xy_target`` immediately before motion.

The endpoint equations are::

    W_target = W_current + [delta_x, delta_y]
    J4_target = Rz_fixed - J1_target - J2_target + 90 deg

Stage7A may first align J4 to the calibrated absolute Rz, after which the
controller executes J1 -> J2 -> J3 -> final J4 sequentially.  The audit
enumerates the selected sequence and independently bounds transient XY and Rz.
"""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .kinematics import (
    fk_wrist,
    ik_wrist,
    j4_for_rz,
    reach_ok,
    rz_of,
    solve_joints,
)


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} must contain {length} values")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain finite values")
    return result


def angular_difference_deg(left: float, right: float) -> float:
    """Smallest absolute circular angle difference in degrees."""

    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def load_p22_float_preset(project_root: Path) -> tuple[str, list[float]]:
    """Load the manually taught P22 observation-height preset fail-closed."""

    path = Path(project_root) / "scara_presets.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"找不到预设文件：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"预设文件不是有效JSON：{path}（{exc}）") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("scara_presets.json 顶层必须是对象")
    for name in ("P22 float", "P22_float"):
        if name in raw:
            return name, _finite_vector(raw[name], 4, f"预设点 {name}")
    raise ValueError(
        "缺少手工示教的 P22 float（也接受 P22_float）。"
        "请先在安全观察高度保存该预设；程序不会从其他槽位插值。"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_stage7a_motion_contract(project_root: Path) -> dict[str, Any]:
    """Load the approved P22 Stage-5 domain used by Task10 action fields."""

    from scara.vision.handeye_interaction import (
        load_latest_suction_target,
        load_local_xy_jacobian,
    )

    root = Path(project_root)
    suction = load_latest_suction_target(root)
    payload = load_local_xy_jacobian(root, suction)
    if payload is None:
        raise ValueError(
            "找不到与当前内参、Tray几何和Stage4一致的成功Stage5 Jacobian"
        )
    if payload.get("anchor_target_name") != "P22" or "P22" not in (
        payload.get("valid_target_names") or []
    ):
        raise ValueError("当前Stage5 Jacobian未批准用于P22")
    coordinate = payload.get("coordinate_definition") or {}
    anchor = _finite_vector(
        coordinate.get("anchor_robot_xy_mm"), 2, "anchor_robot_xy_mm"
    )
    try:
        extent = float(coordinate["offset_extent_mm"])
        imaging_j3 = float(coordinate["imaging_j3_mm"])
        rz_deg = float(coordinate["rz_deg"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Stage5 Jacobian缺少有效P22适用域字段") from exc
    if not all(math.isfinite(value) for value in (extent, imaging_j3, rz_deg)):
        raise ValueError("Stage5 Jacobian适用域字段必须为有限数值")
    stage5_path = root / "src/scara/calib/camera1_xy_image_jacobian.json"
    return {
        "target_name": "P22",
        "anchor_robot_xy_mm": [float(anchor[0]), float(anchor[1])],
        "local_extent_mm": float(extent),
        "required_j3_mm": float(imaging_j3),
        "required_rz_deg": float(rz_deg),
        "stage5_path": str(stage5_path),
        "stage5_sha256": _sha256(stage5_path),
    }


def _gate(passed: bool, *, actual: Any = None, limit: Any = None, note: str = "") -> dict:
    return {
        "passed": bool(passed),
        "actual": actual,
        "limit": limit,
        "note": str(note),
    }


def audit_fixed_rz_xy_target(
    current_joints: Sequence[float],
    current_pose: Sequence[float],
    target_joints: Sequence[float],
    *,
    anchor_robot_xy_mm: Sequence[float],
    local_extent_mm: float,
    domain_margin_mm: float,
    required_j3_mm: float,
    j3_tolerance_mm: float,
    required_rz_deg: float,
    rz_tolerance_deg: float,
    max_xy_step_norm_mm: float,
    max_xy_axis_mm: float,
    max_sequential_transient_xy_mm: float,
    target_rz_tolerance_deg: float | None = None,
    max_sequential_transient_rz_deg: float | None = None,
    precompensate_rz: bool = False,
    enforce_sequential_intermediate_domain: bool = True,
) -> dict[str, Any]:
    """Independently audit one proposed absolute joint target.

    The function treats all inputs as untrusted.  Malformed values raise
    ``ValueError``; geometrically valid but unsafe targets return ``passed``
    false with every gate recorded for the JSON audit trail.
    """

    start = _finite_vector(current_joints, 4, "current_joints")
    pose = _finite_vector(current_pose, 6, "current_pose")
    target = _finite_vector(target_joints, 4, "target_joints")
    anchor = _finite_vector(anchor_robot_xy_mm, 2, "anchor_robot_xy_mm")
    numeric = {
        "local_extent_mm": float(local_extent_mm),
        "domain_margin_mm": float(domain_margin_mm),
        "required_j3_mm": float(required_j3_mm),
        "j3_tolerance_mm": float(j3_tolerance_mm),
        "required_rz_deg": float(required_rz_deg),
        "rz_tolerance_deg": float(rz_tolerance_deg),
        "target_rz_tolerance_deg": float(
            rz_tolerance_deg
            if target_rz_tolerance_deg is None
            else target_rz_tolerance_deg
        ),
        "max_sequential_transient_rz_deg": float(
            rz_tolerance_deg
            if max_sequential_transient_rz_deg is None
            else max_sequential_transient_rz_deg
        ),
        "max_xy_step_norm_mm": float(max_xy_step_norm_mm),
        "max_xy_axis_mm": float(max_xy_axis_mm),
        "max_sequential_transient_xy_mm": float(
            max_sequential_transient_xy_mm
        ),
    }
    if not all(math.isfinite(value) for value in numeric.values()):
        raise ValueError("motion audit limits must be finite")
    if (
        numeric["local_extent_mm"] <= 0.0
        or numeric["domain_margin_mm"] < 0.0
        or numeric["domain_margin_mm"] >= numeric["local_extent_mm"]
        or numeric["j3_tolerance_mm"] <= 0.0
        or numeric["rz_tolerance_deg"] <= 0.0
        or numeric["target_rz_tolerance_deg"] <= 0.0
        or numeric["max_sequential_transient_rz_deg"] <= 0.0
        or numeric["max_xy_step_norm_mm"] <= 0.0
        or numeric["max_xy_axis_mm"] <= 0.0
        or numeric["max_sequential_transient_xy_mm"] <= 0.0
    ):
        raise ValueError("motion audit limits are not positive/consistent")

    fk_start = fk_wrist(start[0], start[1])
    fk_target = fk_wrist(target[0], target[1])
    current_xy = (float(pose[0]), float(pose[1]))
    target_xy = (float(fk_target[0]), float(fk_target[1]))
    fk_pose_closure = math.hypot(
        fk_start[0] - current_xy[0], fk_start[1] - current_xy[1]
    )
    step_xy = (
        target_xy[0] - current_xy[0],
        target_xy[1] - current_xy[1],
    )
    step_norm = math.hypot(*step_xy)
    start_delta = (
        current_xy[0] - anchor[0],
        current_xy[1] - anchor[1],
    )
    target_delta = (
        target_xy[0] - anchor[0],
        target_xy[1] - anchor[1],
    )
    allowed_domain = numeric["local_extent_mm"] - numeric["domain_margin_mm"]
    start_rz = rz_of(start[0], start[1], start[3])
    target_rz = rz_of(target[0], target[1], target[3])
    reachable, reach_note = reach_ok(target_xy[0], target_xy[1])

    precompensation_target = [
        float(start[0]),
        float(start[1]),
        float(start[2]),
        float(j4_for_rz(start[0], start[1], numeric["required_rz_deg"])),
    ]
    precompensation_delta_j4 = (
        precompensation_target[3] - float(start[3])
    )
    if bool(precompensate_rz):
        # Stage7A first aligns J4 at the current XY, then moves J1/J2 and
        # applies the final J4 compensation.  The initial Rz mismatch is
        # checked separately by ``current_rz_matches_calibration``; the
        # transient gate covers only commanded states after precompensation.
        sequential_states = [
            list(start),
            list(precompensation_target),
            [target[0], start[1], start[2], precompensation_target[3]],
            [target[0], target[1], start[2], precompensation_target[3]],
            [target[0], target[1], target[2], precompensation_target[3]],
            list(target),
        ]
        transient_rz_states = sequential_states[1:]
        sequential_order_note = "J4 precomp -> J1 -> J2 -> J3 -> final J4"
    else:
        sequential_states = [
            list(start),
            [target[0], start[1], start[2], start[3]],
            [target[0], target[1], start[2], start[3]],
            [target[0], target[1], target[2], start[3]],
            list(target),
        ]
        transient_rz_states = sequential_states
        sequential_order_note = "J1 -> J2 -> J3 -> J4"
    sequential_xy = [fk_wrist(state[0], state[1]) for state in sequential_states]
    sequential_anchor_deltas = [
        (xy[0] - anchor[0], xy[1] - anchor[1]) for xy in sequential_xy
    ]
    sequential_domain_max_axis = max(
        (
            max(abs(delta[0]), abs(delta[1]))
            for delta in sequential_anchor_deltas
        ),
        default=0.0,
    )
    sequential_legs = [
        math.hypot(
            sequential_xy[index][0] - sequential_xy[index - 1][0],
            sequential_xy[index][1] - sequential_xy[index - 1][1],
        )
        for index in range(1, len(sequential_xy))
    ]
    transient_max = max(sequential_legs, default=0.0)
    sequential_rz = [
        rz_of(state[0], state[1], state[3]) for state in sequential_states
    ]
    transient_rz = [
        rz_of(state[0], state[1], state[3]) for state in transient_rz_states
    ]
    transient_rz_max = max(
        (
            angular_difference_deg(value, numeric["required_rz_deg"])
            for value in transient_rz
        ),
        default=0.0,
    )

    gates = {
        "controller_pose_matches_fk": _gate(
            fk_pose_closure <= 0.20 + 1e-12,
            actual=fk_pose_closure,
            limit="<=0.20 mm",
        ),
        "current_j3_at_imaging_height": _gate(
            abs(start[2] - numeric["required_j3_mm"])
            <= numeric["j3_tolerance_mm"] + 1e-12,
            actual=abs(start[2] - numeric["required_j3_mm"]),
            limit=f"<={numeric['j3_tolerance_mm']:.3f} mm",
        ),
        "target_j3_unchanged": _gate(
            abs(target[2] - start[2]) <= 1e-6,
            actual=abs(target[2] - start[2]),
            limit="<=0.000001 mm",
        ),
        "current_rz_matches_calibration": _gate(
            angular_difference_deg(start_rz, numeric["required_rz_deg"])
            <= numeric["rz_tolerance_deg"] + 1e-12,
            actual=angular_difference_deg(start_rz, numeric["required_rz_deg"]),
            limit=f"<={numeric['rz_tolerance_deg']:.3f} deg",
        ),
        "target_rz_preserved": _gate(
            angular_difference_deg(target_rz, numeric["required_rz_deg"])
            <= numeric["target_rz_tolerance_deg"] + 1e-12,
            actual=angular_difference_deg(target_rz, numeric["required_rz_deg"]),
            limit=f"<={numeric['target_rz_tolerance_deg']:.3f} deg",
        ),
        "target_reachable": _gate(reachable, actual=math.hypot(*target_xy), note=reach_note),
        "xy_step_axis_limit": _gate(
            max(abs(step_xy[0]), abs(step_xy[1]))
            <= numeric["max_xy_axis_mm"] + 1e-12,
            actual=max(abs(step_xy[0]), abs(step_xy[1])),
            limit=f"<={numeric['max_xy_axis_mm']:.3f} mm",
        ),
        "xy_step_norm_limit": _gate(
            step_norm <= numeric["max_xy_step_norm_mm"] + 1e-12,
            actual=step_norm,
            limit=f"<={numeric['max_xy_step_norm_mm']:.3f} mm",
        ),
        "current_inside_local_domain": _gate(
            max(abs(start_delta[0]), abs(start_delta[1])) <= allowed_domain + 1e-12,
            actual=[float(start_delta[0]), float(start_delta[1])],
            limit=f"each axis <= {allowed_domain:.3f} mm",
        ),
        "target_inside_local_domain": _gate(
            max(abs(target_delta[0]), abs(target_delta[1])) <= allowed_domain + 1e-12,
            actual=[float(target_delta[0]), float(target_delta[1])],
            limit=f"each axis <= {allowed_domain:.3f} mm",
        ),
        "sequential_transient_xy_limit": _gate(
            transient_max <= numeric["max_sequential_transient_xy_mm"] + 1e-12,
            actual=transient_max,
            limit=f"<={numeric['max_sequential_transient_xy_mm']:.3f} mm",
            note=f"controller order {sequential_order_note}",
        ),
        "sequential_transient_rz_limit": _gate(
            transient_rz_max
            <= numeric["max_sequential_transient_rz_deg"] + 1e-12,
            actual=transient_rz_max,
            limit=f"<={numeric['max_sequential_transient_rz_deg']:.3f} deg",
            note=f"maximum absolute-Rz departure after {sequential_order_note}",
        ),
    }
    if bool(enforce_sequential_intermediate_domain):
        gates["sequential_intermediate_inside_local_domain"] = _gate(
            sequential_domain_max_axis <= allowed_domain + 1e-12,
            actual={
                "maximum_absolute_axis_mm": float(sequential_domain_max_axis),
                "anchor_deltas_xy_mm": [
                    [float(delta[0]), float(delta[1])]
                    for delta in sequential_anchor_deltas
                ],
            },
            limit=f"each axis <= {allowed_domain:.3f} mm",
            note=f"all {sequential_order_note} intermediate states remain in-domain",
        )
    passed = all(bool(gate["passed"]) for gate in gates.values())
    return {
        "passed": bool(passed),
        "gates": gates,
        "current_xy_mm": [float(current_xy[0]), float(current_xy[1])],
        "target_xy_mm": [float(target_xy[0]), float(target_xy[1])],
        "step_xy_mm": [float(step_xy[0]), float(step_xy[1])],
        "step_norm_mm": float(step_norm),
        "anchor_delta_current_xy_mm": [float(start_delta[0]), float(start_delta[1])],
        "anchor_delta_target_xy_mm": [float(target_delta[0]), float(target_delta[1])],
        "sequential_anchor_deltas_xy_mm": [
            [float(delta[0]), float(delta[1])]
            for delta in sequential_anchor_deltas
        ],
        "sequential_transient_xy_legs_mm": [float(value) for value in sequential_legs],
        "sequential_transient_max_mm": float(transient_max),
        "sequential_transient_rz_max_deg": float(transient_rz_max),
        "sequential_intermediate_domain_enforced": bool(
            enforce_sequential_intermediate_domain
        ),
        "sequential_rz_deg": [float(value) for value in sequential_rz],
        "rz_precompensation": {
            "enabled": bool(precompensate_rz),
            "required": bool(
                precompensate_rz
                and abs(precompensation_delta_j4) > 1e-9
            ),
            "target_joints": [float(value) for value in precompensation_target],
            "current_j4_deg": float(start[3]),
            "target_j4_deg": float(precompensation_target[3]),
            "delta_j4_deg": float(precompensation_delta_j4),
            "current_rz_deg": float(start_rz),
            "target_rz_deg": float(numeric["required_rz_deg"]),
        },
        "current_rz_deg": float(start_rz),
        "target_rz_deg": float(target_rz),
        "target_joints": [float(value) for value in target],
    }


def audit_j4_only_orientation_target(
    current_joints: Sequence[float],
    current_pose: Sequence[float],
    target_joints: Sequence[float],
    *,
    anchor_robot_xy_mm: Sequence[float],
    local_extent_mm: float,
    domain_margin_mm: float,
    required_j3_mm: float,
    j3_tolerance_mm: float,
    required_start_rz_deg: float,
    start_rz_tolerance_deg: float,
    target_rz_deg: float,
    target_rz_tolerance_deg: float,
    maximum_j4_rotation_deg: float,
) -> dict[str, Any]:
    """Audit a final orientation move that may change J4 and nothing else."""

    start = _finite_vector(current_joints, 4, "current_joints")
    pose = _finite_vector(current_pose, 6, "current_pose")
    target = _finite_vector(target_joints, 4, "target_joints")
    anchor = _finite_vector(anchor_robot_xy_mm, 2, "anchor_robot_xy_mm")
    numeric = {
        "local_extent_mm": float(local_extent_mm),
        "domain_margin_mm": float(domain_margin_mm),
        "required_j3_mm": float(required_j3_mm),
        "j3_tolerance_mm": float(j3_tolerance_mm),
        "required_start_rz_deg": float(required_start_rz_deg),
        "start_rz_tolerance_deg": float(start_rz_tolerance_deg),
        "target_rz_deg": float(target_rz_deg),
        "target_rz_tolerance_deg": float(target_rz_tolerance_deg),
        "maximum_j4_rotation_deg": float(maximum_j4_rotation_deg),
    }
    if not all(math.isfinite(value) for value in numeric.values()):
        raise ValueError("J4 orientation audit limits must be finite")
    if (
        numeric["local_extent_mm"] <= 0.0
        or numeric["domain_margin_mm"] < 0.0
        or numeric["domain_margin_mm"] >= numeric["local_extent_mm"]
        or numeric["j3_tolerance_mm"] <= 0.0
        or numeric["start_rz_tolerance_deg"] <= 0.0
        or numeric["target_rz_tolerance_deg"] <= 0.0
        or numeric["maximum_j4_rotation_deg"] <= 0.0
    ):
        raise ValueError("J4 orientation audit limits are not positive/consistent")

    current_xy = np.asarray(fk_wrist(start[0], start[1]), dtype=np.float64)
    target_xy = np.asarray(fk_wrist(target[0], target[1]), dtype=np.float64)
    pose_xy = np.asarray(pose[:2], dtype=np.float64)
    step_xy = target_xy - pose_xy
    current_delta = current_xy - anchor
    target_delta = target_xy - anchor
    allowed_domain = numeric["local_extent_mm"] - numeric["domain_margin_mm"]
    start_rz = rz_of(start[0], start[1], start[3])
    actual_target_rz = rz_of(target[0], target[1], target[3])
    # Joint commands are absolute controller values.  Do not treat +179 and
    # -179 as a harmless 2-degree move: a controller may execute the raw
    # 358-degree difference instead of choosing a wrapped shortest path.
    j4_rotation = abs(target[3] - start[3])
    reachable, reach_note = reach_ok(float(target_xy[0]), float(target_xy[1]))
    gates = {
        "controller_pose_matches_fk": _gate(
            float(np.linalg.norm(current_xy - pose_xy)) <= 0.20 + 1e-12,
            actual=float(np.linalg.norm(current_xy - pose_xy)),
            limit="<=0.20 mm",
        ),
        "current_j3_at_imaging_height": _gate(
            abs(start[2] - numeric["required_j3_mm"])
            <= numeric["j3_tolerance_mm"] + 1e-12,
            actual=abs(start[2] - numeric["required_j3_mm"]),
            limit=f"<={numeric['j3_tolerance_mm']:.3f} mm",
        ),
        "current_rz_matches_xy_stage": _gate(
            angular_difference_deg(start_rz, numeric["required_start_rz_deg"])
            <= numeric["start_rz_tolerance_deg"] + 1e-12,
            actual=angular_difference_deg(start_rz, numeric["required_start_rz_deg"]),
            limit=f"<={numeric['start_rz_tolerance_deg']:.3f} deg",
        ),
        "j1_unchanged": _gate(
            abs(target[0] - start[0]) <= 0.002 + 1e-12,
            actual=abs(target[0] - start[0]),
            limit="<=0.002 deg",
        ),
        "j2_unchanged": _gate(
            abs(target[1] - start[1]) <= 0.002 + 1e-12,
            actual=abs(target[1] - start[1]),
            limit="<=0.002 deg",
        ),
        "j3_unchanged": _gate(
            abs(target[2] - start[2]) <= 0.002 + 1e-12,
            actual=abs(target[2] - start[2]),
            limit="<=0.002 mm",
        ),
        "xy_held_during_j4_rotation": _gate(
            float(np.linalg.norm(step_xy)) <= 0.01 + 1e-12,
            actual=float(np.linalg.norm(step_xy)),
            limit="<=0.010 mm",
        ),
        "target_rz_aligned_to_tray": _gate(
            angular_difference_deg(actual_target_rz, numeric["target_rz_deg"])
            <= numeric["target_rz_tolerance_deg"] + 1e-12,
            actual=angular_difference_deg(actual_target_rz, numeric["target_rz_deg"]),
            limit=f"<={numeric['target_rz_tolerance_deg']:.3f} deg",
        ),
        "j4_rotation_limit": _gate(
            j4_rotation <= numeric["maximum_j4_rotation_deg"] + 1e-12,
            actual=j4_rotation,
            limit=f"<={numeric['maximum_j4_rotation_deg']:.3f} deg",
        ),
        "target_reachable": _gate(
            reachable,
            actual=float(np.linalg.norm(target_xy)),
            note=reach_note,
        ),
        "current_inside_local_domain": _gate(
            float(np.max(np.abs(current_delta))) <= allowed_domain + 1e-12,
            actual=current_delta.astype(float).tolist(),
            limit=f"each axis <= {allowed_domain:.3f} mm",
        ),
        "target_inside_local_domain": _gate(
            float(np.max(np.abs(target_delta))) <= allowed_domain + 1e-12,
            actual=target_delta.astype(float).tolist(),
            limit=f"each axis <= {allowed_domain:.3f} mm",
        ),
    }
    return {
        "passed": all(gate["passed"] is True for gate in gates.values()),
        "gates": gates,
        "current_xy_mm": pose_xy.astype(float).tolist(),
        "target_xy_mm": target_xy.astype(float).tolist(),
        "step_xy_mm": step_xy.astype(float).tolist(),
        "step_norm_mm": float(np.linalg.norm(step_xy)),
        "current_rz_deg": float(start_rz),
        "target_rz_deg": float(actual_target_rz),
        "target_joints": [float(value) for value in target],
        "rz_precompensation": {
            "enabled": False,
            "required": False,
            "target_joints": [float(value) for value in start],
            "current_j4_deg": float(start[3]),
            "target_j4_deg": float(start[3]),
            "delta_j4_deg": 0.0,
            "current_rz_deg": float(start_rz),
            "target_rz_deg": float(start_rz),
        },
    }


def plan_fixed_rz_xy_step(
    current_joints: Sequence[float],
    current_pose: Sequence[float],
    command_xy_mm: Sequence[float],
    *,
    anchor_robot_xy_mm: Sequence[float],
    local_extent_mm: float,
    domain_margin_mm: float,
    required_j3_mm: float,
    j3_tolerance_mm: float,
    required_rz_deg: float,
    rz_tolerance_deg: float,
    max_xy_step_norm_mm: float,
    max_xy_axis_mm: float,
    max_sequential_transient_xy_mm: float,
    target_rz_tolerance_deg: float | None = None,
    max_sequential_transient_rz_deg: float | None = None,
    precompensate_rz: bool = False,
    enforce_sequential_intermediate_domain: bool = True,
    allow_rejected_audit: bool = False,
) -> dict[str, Any]:
    """Plan an exact fixed-J3/fixed-Rz endpoint on the current IK branch.

    The normal/default contract raises when any audit gate rejects the target.
    A read-only UI may set ``allow_rejected_audit`` so it can display every
    failed gate; such a returned target remains unapproved and the hardware
    worker always repeats the strict audit before accepting a command.
    """

    start = _finite_vector(current_joints, 4, "current_joints")
    pose = _finite_vector(current_pose, 6, "current_pose")
    command = _finite_vector(command_xy_mm, 2, "command_xy_mm")
    target_x = float(pose[0] + command[0])
    target_y = float(pose[1] + command[1])
    reachable, reason = reach_ok(target_x, target_y)
    if not reachable:
        raise ValueError(f"Stage7A XY target is unreachable: {reason}")

    rounded = solve_joints(
        target_x,
        target_y,
        start[2],
        rz_deg=float(required_rz_deg),
        ref_joints=list(start),
    )
    if rounded is None:
        raise ValueError("Stage7A XY target has no continuous IK solution")
    branches = ik_wrist(target_x, target_y)
    if not branches:
        raise ValueError("Stage7A XY target has no IK branches")
    chosen_j1, chosen_j2 = min(
        branches,
        key=lambda branch: (
            angular_difference_deg(branch[0], rounded[0])
            + angular_difference_deg(branch[1], rounded[1])
        ),
    )
    target = [
        float(chosen_j1),
        float(chosen_j2),
        float(start[2]),
        float(j4_for_rz(chosen_j1, chosen_j2, required_rz_deg)),
    ]
    audit = audit_fixed_rz_xy_target(
        start,
        pose,
        target,
        anchor_robot_xy_mm=anchor_robot_xy_mm,
        local_extent_mm=local_extent_mm,
        domain_margin_mm=domain_margin_mm,
        required_j3_mm=required_j3_mm,
        j3_tolerance_mm=j3_tolerance_mm,
        required_rz_deg=required_rz_deg,
        rz_tolerance_deg=rz_tolerance_deg,
        max_xy_step_norm_mm=max_xy_step_norm_mm,
        max_xy_axis_mm=max_xy_axis_mm,
        max_sequential_transient_xy_mm=max_sequential_transient_xy_mm,
        target_rz_tolerance_deg=target_rz_tolerance_deg,
        max_sequential_transient_rz_deg=max_sequential_transient_rz_deg,
        precompensate_rz=precompensate_rz,
        enforce_sequential_intermediate_domain=(
            enforce_sequential_intermediate_domain
        ),
    )
    if not audit["passed"] and not bool(allow_rejected_audit):
        failed = ", ".join(
            name for name, gate in audit["gates"].items() if not gate["passed"]
        )
        raise ValueError(f"Stage7A planned target failed safety audit: {failed}")
    return {
        "target_joints": [float(value) for value in target],
        "requested_command_xy_mm": [float(command[0]), float(command[1])],
        "audit": audit,
    }


__all__ = [
    "angular_difference_deg",
    "audit_fixed_rz_xy_target",
    "audit_j4_only_orientation_target",
    "load_p22_float_preset",
    "load_stage7a_motion_contract",
    "plan_fixed_rz_xy_step",
]
