"""Task 9: calibrate the local robot-XY to image-error Jacobian.

The equations, coordinate definitions, quality gates, and validation boundary
are documented in ``docs/stage5_xy_image_jacobian.md``.  This action only owns
the small, explicitly bounded robot motion and camera-1 acquisition.  Stage-3
pose estimation and the Stage-5 solve live in
``scara.vision.xy_image_jacobian_runtime``.

Safety contract
---------------
* The UI locks one selected ``P00``-``P55`` slot into the task.  The operator
  must first place the arm at that slot's computed fixed-height anchor; Task9
  deliberately does not perform a cross-tray positioning move.
* J3 and absolute end-effector Rz remain fixed at the approved imaging values.
* The J4/suction axis visits only a 3 x 3 grid with X/Y components in
  ``{-2, 0, +2} mm`` and then returns to the exact selected-slot anchor. Every 2 mm
  grid/return edge is split into Cartesian targets no farther than 1 mm apart
  because the controller executes J1, J2, J3, and J4 sequentially.
* Only camera source 1 is opened.  There is no Z command, vacuum command, or
  contact-height motion in this file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ACTION_API_VERSION = 1

ANCHOR_TARGET_NAME = "P22"
CAMERA_SOURCE = 1
FRAMES_PER_OFFSET = 12
SETTLE_SECONDS = 1.5
START_TOLERANCE_DEG_OR_MM = 0.15
MOVE_TOLERANCE_DEG_OR_MM = 0.05
J3_SAFETY_TOLERANCE_MM = 0.15
IMAGING_J3_NOMINAL_MM = -27.0119
IMAGING_J3_ALLOWED_ERROR_MM = 0.50
EXPECTED_RZ_DEG = 20.82
MAXIMUM_RZ_ERROR_DEG = 1.0
MAX_CARTESIAN_TRANSIT_STEP_MM = 1.0
MAX_SEQUENTIAL_TRANSIENT_XY_MM = 2.0001
MAX_SEQUENTIAL_TRANSIENT_RZ_DEG = 1.0

# These are the nine acquisition locations. The centre is acquired first;
# movement between locations is subdivided separately and transit points are
# never sampled.
GRID_OFFSETS_XY_MM: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (-2.0, 0.0),
    (-2.0, -2.0),
    (0.0, -2.0),
    (2.0, -2.0),
    (2.0, 0.0),
    (2.0, 2.0),
    (0.0, 2.0),
    (-2.0, 2.0),
)

# Return through the already validated left-middle grid point. Both return
# edges are also subdivided into <= 1 mm Cartesian targets and are not sampled.
RETURN_OFFSETS_XY_MM: tuple[tuple[float, float], ...] = (
    (-2.0, 0.0),
    (0.0, 0.0),
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _finite_joints(values: object, label: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"{label} 必须包含 J1/J2/J3/J4 四个数值")
    joints = [float(value) for value in values]
    if not all(math.isfinite(value) for value in joints):
        raise ValueError(f"{label} 包含非有限数值")
    return joints


def _load_presets() -> Mapping[str, Any]:
    path = _project_root() / "scara_presets.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"找不到预设文件：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"预设文件不是有效JSON：{path}（{exc}）") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("scara_presets.json 顶层必须是对象")
    return raw


def _load_p22_preset() -> tuple[str, list[float]]:
    raw = _load_presets()
    for name in ("P22 float", "P22_float"):
        if name in raw:
            return name, _finite_joints(raw[name], f"预设点 {name}")
    raise ValueError(
        "Task9缺少手工示教的 P22 float（也接受 P22_float）。"
        "请先在安全观察高度保存该预设。"
    )


def _load_geometry() -> Mapping[str, Any]:
    path = _project_root() / "src/scara/calib/tray_board_geometry.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"找不到Tray几何：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Tray几何不是有效JSON：{path}（{exc}）") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("tray_board_geometry.json 顶层必须是对象")
    return raw


def _slot_point_T_mm(target_name: str) -> list[float]:
    from scara.vision.local_jacobian_registry import validate_slot_name

    target = validate_slot_name(target_name)
    geometry = _load_geometry()
    slots = geometry.get("slots")
    if not isinstance(slots, Mapping) or target not in slots:
        raise ValueError(f"Tray几何缺少目标槽 {target}")
    point = [float(value) for value in slots[target]]
    if len(point) != 3 or not all(math.isfinite(value) for value in point):
        raise ValueError(f"Tray几何槽位 {target} 必须包含三个有限坐标")
    return point


def _geometry_anchor_joints(target_name: str) -> tuple[str, list[float]]:
    """Compute one slot-centre anchor from Stage-2 geometry and the P00 pose.

    The P00 preset supplies only the approved imaging J3, absolute Rz and the
    continuous IK branch.  Target world XY comes from the calibrated Tray frame.
    """

    from scara.pipeline.kinematics import fk_wrist, ik_wrist, j4_for_rz, rz_of
    from scara.vision.full_tray_positioning import slot_world_xy_mm
    from scara.vision.local_jacobian_registry import validate_slot_name

    target = validate_slot_name(target_name)
    presets = _load_presets()
    p00_name = next(
        (name for name in ("P00 float", "P00_float") if name in presets),
        None,
    )
    if p00_name is None:
        raise ValueError(
            "任意槽local Jacobian标定需要 P00 float，"
            "用于锁定观察高度、绝对Rz和IK分支"
        )
    reference = _finite_joints(presets[p00_name], f"预设点 {p00_name}")
    _validated_anchor_context(reference, p00_name)
    geometry = _load_geometry()
    target_xy = slot_world_xy_mm(geometry, target)
    fixed_rz = float(rz_of(reference[0], reference[1], reference[3]))
    branches = ik_wrist(float(target_xy[0]), float(target_xy[1]))
    if not branches:
        raise ValueError(f"目标槽 {target} 的几何中心不可达")
    j1, j2 = min(
        branches,
        key=lambda branch: (
            _angular_error_deg(branch[0], reference[0])
            + _angular_error_deg(branch[1], reference[1])
        ),
    )
    anchor = [
        float(j1),
        float(j2),
        float(reference[2]),
        float(j4_for_rz(j1, j2, fixed_rz)),
    ]
    actual_xy = fk_wrist(anchor[0], anchor[1])
    if math.dist(actual_xy, target_xy) > 0.02:
        raise ValueError(f"目标槽 {target} 的IK回代误差超过0.02mm")
    return f"{target} geometry anchor", anchor


def load_target_anchor(target_name: str) -> tuple[str, list[float], list[float]]:
    """Return ``(anchor label, joints, point_T)`` for one selected slot.

    P22 keeps the existing taught-preset behavior whenever that preset exists.
    Other slots use the Stage-2 Tray geometry, so teaching all 36 presets is not
    required.  The task still asserts the computed anchor before any motion.
    """

    from scara.vision.local_jacobian_registry import validate_slot_name

    target = validate_slot_name(target_name)
    point_T = _slot_point_T_mm(target)
    if target == "P22":
        try:
            label, joints = _load_p22_preset()
            return label, joints, point_T
        except ValueError:
            pass
    label, joints = _geometry_anchor_joints(target)
    return label, joints, point_T


def _angular_error_deg(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def _validated_anchor_context(
    anchor_joints: Sequence[float],
    anchor_label: str = "P22 float",
) -> tuple[list[float], float, float, float]:
    """Validate one anchor and return joints, wrist XY and absolute Rz."""

    from scara.pipeline.kinematics import fk_wrist, rz_of

    anchor = _finite_joints(list(anchor_joints), anchor_label)
    if abs(anchor[2] - IMAGING_J3_NOMINAL_MM) > IMAGING_J3_ALLOWED_ERROR_MM:
        raise ValueError(
            f"{anchor_label} J3={anchor[2]:.4f} mm is outside the approved imaging "
            f"height {IMAGING_J3_NOMINAL_MM:.4f} +/- "
            f"{IMAGING_J3_ALLOWED_ERROR_MM:.2f} mm"
        )
    anchor_rz = rz_of(anchor[0], anchor[1], anchor[3])
    if _angular_error_deg(anchor_rz, EXPECTED_RZ_DEG) > MAXIMUM_RZ_ERROR_DEG:
        raise ValueError(
            f"{anchor_label} Rz={anchor_rz:.4f} deg differs from the approved Stage-4 "
            f"pose {EXPECTED_RZ_DEG:.2f} deg by more than "
            f"{MAXIMUM_RZ_ERROR_DEG:.2f} deg"
        )
    anchor_x, anchor_y = fk_wrist(anchor[0], anchor[1])
    return anchor, float(anchor_x), float(anchor_y), float(anchor_rz)


def _solve_offset_joint_target(
    anchor: Sequence[float],
    anchor_xy: tuple[float, float],
    anchor_rz: float,
    offset_xy_mm: tuple[float, float],
    reference_joints: Sequence[float],
) -> list[float]:
    """Solve one exact Cartesian target on the continuous anchor IK branch.

    ``solve_joints`` is called for every acquisition or transit target. Its
    selected branch is retained, while the unrounded ``ik_wrist`` angles keep
    the <= 1 mm Cartesian subdivision geometrically exact.
    """

    from scara.pipeline.kinematics import (
        fk_wrist,
        ik_wrist,
        j4_for_rz,
        reach_ok,
        rz_of,
        solve_joints,
    )

    dx_mm, dy_mm = (float(value) for value in offset_xy_mm)
    target_x = float(anchor_xy[0]) + dx_mm
    target_y = float(anchor_xy[1]) + dy_mm
    reachable, reason = reach_ok(target_x, target_y)
    if not reachable:
        raise ValueError(
            f"Task9 offset ({dx_mm:+.3f}, {dy_mm:+.3f}) mm is unreachable: "
            f"{reason}"
        )
    rounded_solution = solve_joints(
        target_x,
        target_y,
        float(anchor[2]),
        rz_deg=float(anchor_rz),
        ref_joints=list(reference_joints),
    )
    if rounded_solution is None:
        raise ValueError(
            f"Task9 offset ({dx_mm:+.3f}, {dy_mm:+.3f}) mm has no "
            "continuous IK solution"
        )

    branches = ik_wrist(target_x, target_y)
    chosen_j1, chosen_j2 = min(
        branches,
        key=lambda branch: (
            _angular_error_deg(branch[0], rounded_solution[0])
            + _angular_error_deg(branch[1], rounded_solution[1])
        ),
    )
    joints = [
        float(chosen_j1),
        float(chosen_j2),
        float(anchor[2]),
        float(j4_for_rz(chosen_j1, chosen_j2, anchor_rz)),
    ]
    actual_x, actual_y = fk_wrist(joints[0], joints[1])
    position_error = math.hypot(actual_x - target_x, actual_y - target_y)
    if position_error > 0.02:
        raise ValueError(f"Task9 IK closure error is {position_error:.4f} mm")
    if abs(joints[2] - float(anchor[2])) > 1e-9:
        raise ValueError("Task9 IK unexpectedly changed J3")
    if _angular_error_deg(rz_of(joints[0], joints[1], joints[3]), anchor_rz) > 0.01:
        raise ValueError("Task9 IK failed to preserve endpoint Rz")
    return joints


def _cartesian_substeps(
    start_offset_xy_mm: tuple[float, float],
    end_offset_xy_mm: tuple[float, float],
    max_step_mm: float = MAX_CARTESIAN_TRANSIT_STEP_MM,
) -> list[tuple[float, float]]:
    """Return equally spaced offsets excluding start and including end."""

    max_step = float(max_step_mm)
    if not math.isfinite(max_step) or max_step <= 0.0:
        raise ValueError("Task9 Cartesian max step must be finite and positive")
    start_x, start_y = (float(value) for value in start_offset_xy_mm)
    end_x, end_y = (float(value) for value in end_offset_xy_mm)
    dx = end_x - start_x
    dy = end_y - start_y
    distance = math.hypot(dx, dy)
    if distance <= 1e-12:
        return []
    count = max(1, int(math.ceil(distance / max_step - 1e-12)))
    return [
        (start_x + dx * index / count, start_y + dy * index / count)
        for index in range(1, count + 1)
    ]


def generate_grid_joint_targets(
    anchor_joints: Sequence[float],
) -> dict[tuple[float, float], list[float]]:
    """Generate absolute joint targets while preserving anchor J3 and Rz.

    For an offset ``(dx, dy)``, the desired J4-axis position is
    ``W_target = FK(anchor.J1, anchor.J2) + [dx, dy]``.  ``solve_joints`` then finds
    the continuous IK branch and selects J4 so that
    ``Rz = J1 + J2 + J4 - 90`` remains exactly equal to the anchor value.
    """

    anchor, anchor_x, anchor_y, anchor_rz = _validated_anchor_context(anchor_joints)
    targets: dict[tuple[float, float], list[float]] = {
        (0.0, 0.0): list(anchor)
    }
    reference = list(anchor)
    for dx_mm, dy_mm in GRID_OFFSETS_XY_MM[1:]:
        joints = _solve_offset_joint_target(
            anchor,
            (anchor_x, anchor_y),
            anchor_rz,
            (dx_mm, dy_mm),
            reference,
        )
        key = (float(dx_mm), float(dy_mm))
        targets[key] = [float(value) for value in joints]
        reference = targets[key]
    return targets


def _record_name(
    dx_mm: float,
    dy_mm: float,
    frame_index: int,
    target_name: str = ANCHOR_TARGET_NAME,
) -> str:
    return (
        f"TASK9|target={target_name}|dx={dx_mm:+.3f}|dy={dy_mm:+.3f}|"
        f"frame={frame_index:02d}/{FRAMES_PER_OFFSET:02d}"
    )


def _append_frame_burst(
    actions: list[dict],
    target_name: str,
    dx_mm: float,
    dy_mm: float,
) -> None:
    actions.append({"type": "wait", "seconds": SETTLE_SECONDS})
    for frame_index in range(1, FRAMES_PER_OFFSET + 1):
        actions.append(
            {
                "type": "record_point",
                "name": _record_name(
                    dx_mm, dy_mm, frame_index, target_name=target_name
                ),
            }
        )
        actions.append({"type": "capture", "source": CAMERA_SOURCE})


def _sequential_motion_audit(
    current_joints: Sequence[float],
    target_joints: Sequence[float],
    required_rz_deg: float,
) -> tuple[float, float]:
    """Audit the controller's real J1 -> J2 -> J3 -> J4 execution order."""

    from scara.pipeline.kinematics import fk_wrist, rz_of

    state = [float(value) for value in current_joints]
    states = [list(state)]
    for axis in range(4):
        state = list(state)
        state[axis] = float(target_joints[axis])
        states.append(state)
    xy = [fk_wrist(value[0], value[1]) for value in states]
    maximum_xy_step = max(
        math.dist(left, right) for left, right in zip(xy, xy[1:])
    )
    maximum_rz_departure = max(
        _angular_error_deg(
            rz_of(value[0], value[1], value[3]), required_rz_deg
        )
        for value in states
    )
    return float(maximum_xy_step), float(maximum_rz_departure)


def build_action_for_anchor(
    target_name: str,
    preset_name: str,
    anchor_joints: Sequence[float],
) -> dict:
    """Build a validated Task9 action for one immutable selected slot."""

    from scara.vision.local_jacobian_registry import validate_slot_name

    target = validate_slot_name(target_name)
    anchor = _finite_joints(list(anchor_joints), preset_name)
    anchor, anchor_x, anchor_y, anchor_rz = _validated_anchor_context(
        anchor, preset_name
    )
    actions: list[dict] = [
        {
            "type": "assert_joints",
            "name": f"确认Task9起点 {preset_name}",
            "joints": list(anchor),
            "tolerance": START_TOLERANCE_DEG_OR_MM,
        }
    ]

    current_offset = (0.0, 0.0)
    reference_joints = list(anchor)

    def append_cartesian_edge(
        end_offset: tuple[float, float],
        endpoint_name: str,
        *,
        final_exact_anchor: bool = False,
    ) -> None:
        """Append unsampled <=1 mm targets for one grid or return edge."""

        nonlocal current_offset, reference_joints
        # Keep the already-tested P22 route byte-for-byte compatible.  Other
        # tray locations can have a less favourable J1/J2 sequential sweep,
        # so use 0.5 mm endpoints and still audit every actual controller leg.
        substeps = _cartesian_substeps(
            current_offset,
            end_offset,
            max_step_mm=(
                MAX_CARTESIAN_TRANSIT_STEP_MM
                if target == "P22"
                else min(0.5, MAX_CARTESIAN_TRANSIT_STEP_MM)
            ),
        )
        for substep_index, offset in enumerate(substeps, start=1):
            # Solve every intermediate target with the previous target as the
            # IK reference, preserving branch continuity throughout the route.
            solved_joints = _solve_offset_joint_target(
                anchor,
                (anchor_x, anchor_y),
                anchor_rz,
                offset,
                reference_joints,
            )
            is_endpoint = substep_index == len(substeps)
            joints = (
                list(anchor)
                if final_exact_anchor and is_endpoint
                else solved_joints
            )
            transient_xy, transient_rz = _sequential_motion_audit(
                reference_joints, joints, anchor_rz
            )
            if transient_xy > MAX_SEQUENTIAL_TRANSIENT_XY_MM + 1e-9:
                raise ValueError(
                    f"Task9 {target}逐轴中转XY={transient_xy:.3f}mm，"
                    f"超过{MAX_SEQUENTIAL_TRANSIENT_XY_MM:.4f}mm"
                )
            if transient_rz > MAX_SEQUENTIAL_TRANSIENT_RZ_DEG + 1e-9:
                raise ValueError(
                    f"Task9 {target}逐轴中转Rz偏差={transient_rz:.3f}°，"
                    f"超过{MAX_SEQUENTIAL_TRANSIENT_RZ_DEG:.2f}°"
                )
            actions.append(
                {
                    "type": "move_joints",
                    "name": (
                        endpoint_name
                        if is_endpoint
                        else (
                            "TASK9 Cartesian transit "
                            f"dx={offset[0]:+.3f}mm, dy={offset[1]:+.3f}mm"
                        )
                    ),
                    "joints": list(joints),
                    "tolerance": MOVE_TOLERANCE_DEG_OR_MM,
                    "require_current_j3_mm": anchor[2],
                    "j3_tolerance_mm": J3_SAFETY_TOLERANCE_MM,
                }
            )
            reference_joints = list(joints)
        current_offset = (float(end_offset[0]), float(end_offset[1]))

    for index, (dx_mm, dy_mm) in enumerate(GRID_OFFSETS_XY_MM):
        if index > 0:
            append_cartesian_edge(
                (dx_mm, dy_mm),
                (
                    f"TASK9小幅移动到 dx={dx_mm:+.1f}mm, "
                    f"dy={dy_mm:+.1f}mm"
                ),
            )
        _append_frame_burst(actions, target, dx_mm, dy_mm)

    for return_index, (dx_mm, dy_mm) in enumerate(RETURN_OFFSETS_XY_MM):
        is_final = return_index == len(RETURN_OFFSETS_XY_MM) - 1
        append_cartesian_edge(
            (dx_mm, dy_mm),
            (
                f"Task9结束，返回 {preset_name}"
                if is_final
                else (
                    f"Task9返回中转 dx={dx_mm:+.1f}mm, "
                    f"dy={dy_mm:+.1f}mm"
                )
            ),
            final_exact_anchor=is_final,
        )
    return {
        "name": f"task9 {target} 相机1局部XY图像Jacobian标定",
        "description": (
            f"从{target}固定高度槽中心锚点开始，只执行±2mm的3×3局部XY网格；"
            "每条2mm网格边拆成两个不超过1mm的Cartesian中转目标；"
            f"每个偏移采集12张相机1照片并记录实际机械臂状态，结束后回到{target}。"
            "所有目标保持相同J3和末端Rz；不执行Z移动、不执行旋转扫描、不触碰托盘、"
            "不控制真空。该任务会实际小幅移动机械臂，请保持低速和物理急停可用。"
        ),
        "camera_model": {
            "offset_mm": 0.0,
            "angle_reference": "world_negative_y",
            "positive_rotation": "counter_clockwise_from_above",
        },
        "actions": actions,
    }


def build_action_for_preset(
    preset_name: str,
    p22_joints: Sequence[float],
) -> dict:
    """Backward-compatible P22 entry point used by existing tests/imports."""

    return build_action_for_anchor(ANCHOR_TARGET_NAME, preset_name, p22_joints)


def build_action_for_target(target_name: str) -> dict:
    """Build Task9 for the slot locked by the hand-eye target selector."""

    preset_name, anchor_joints, _point_T = load_target_anchor(target_name)
    return build_action_for_anchor(target_name, preset_name, anchor_joints)


def build_action() -> dict:
    preset_name, p22_joints = _load_p22_preset()
    return build_action_for_preset(preset_name, p22_joints)


def create_task_runtime_for_target(
    output_dir: Path,
    target_name: str,
    parent=None,
):
    """Attach Stage-3 processing and install one target-named Jacobian."""

    from scara.vision.local_jacobian_registry import validate_slot_name
    from scara.vision.xy_image_jacobian_runtime import (
        create_camera1_xy_image_jacobian_runtime,
    )

    target = validate_slot_name(target_name)
    _anchor_label, anchor_joints, point_T = load_target_anchor(target)
    return create_camera1_xy_image_jacobian_runtime(
        output_dir=output_dir,
        project_root=_project_root(),
        anchor_target_name=target,
        anchor_point_T_mm=point_T,
        anchor_preset_joints=anchor_joints,
        command_offsets_xy_mm=GRID_OFFSETS_XY_MM,
        frames_per_offset=FRAMES_PER_OFFSET,
        parent=parent,
    )


def create_task_runtime(output_dir: Path, parent=None):
    """Attach Stage-3 processing and the Stage-5 robust batch fit."""

    return create_task_runtime_for_target(
        output_dir, ANCHOR_TARGET_NAME, parent=parent
    )


if __name__ == "__main__":
    preview = build_action()
    point_count = sum(step["type"] == "record_point" for step in preview["actions"])
    photo_count = sum(step["type"] == "capture" for step in preview["actions"])
    move_count = sum(step["type"] == "move_joints" for step in preview["actions"])
    print(preview["name"])
    print(
        f"offsets={len(GRID_OFFSETS_XY_MM)}; moves={move_count}; "
        f"points={point_count}; photos={photo_count}"
    )
    print("Z motion=none; rotation scan=none; vacuum commands=none; final=P22 float")
