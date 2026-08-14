"""Task 9: calibrate the local robot-XY to image-error Jacobian.

The equations, coordinate definitions, quality gates, and validation boundary
are documented in ``docs/stage5_xy_image_jacobian.md``.  This action only owns
the small, explicitly bounded robot motion and camera-1 acquisition.  Stage-3
pose estimation and the Stage-5 solve live in
``scara.vision.xy_image_jacobian_runtime``.

Safety contract
---------------
* The operator must place the arm at the manually taught ``P22 float`` preset.
* J3 and absolute end-effector Rz remain fixed at their P22 values.
* The J4/suction axis visits only a 3 x 3 grid with X/Y components in
  ``{-2, 0, +2} mm`` and then returns to the exact P22 preset. Every 2 mm
  grid/return edge is split into Cartesian targets no farther than 1 mm apart
  because the controller executes J1, J2, J3, and J4 sequentially.
* Only camera source 1 is opened.  There is no Z command, vacuum command, or
  contact-height motion in this file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence


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


def _load_p22_preset() -> tuple[str, list[float]]:
    path = _project_root() / "scara_presets.json"
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
            return name, _finite_joints(raw[name], f"预设点 {name}")
    raise ValueError(
        "Task9缺少手工示教的 P22 float（也接受 P22_float）。"
        "请先在安全观察高度保存该预设；程序不会从其他槽位插值。"
    )


def _angular_error_deg(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def _validated_anchor_context(
    p22_joints: Sequence[float],
) -> tuple[list[float], float, float, float]:
    """Validate P22 and return ``(joints, wrist_x, wrist_y, absolute_rz)``."""

    from scara.pipeline.kinematics import fk_wrist, rz_of

    anchor = _finite_joints(list(p22_joints), "P22 float")
    if abs(anchor[2] - IMAGING_J3_NOMINAL_MM) > IMAGING_J3_ALLOWED_ERROR_MM:
        raise ValueError(
            f"P22 J3={anchor[2]:.4f} mm is outside the approved imaging "
            f"height {IMAGING_J3_NOMINAL_MM:.4f} +/- "
            f"{IMAGING_J3_ALLOWED_ERROR_MM:.2f} mm"
        )
    anchor_rz = rz_of(anchor[0], anchor[1], anchor[3])
    if _angular_error_deg(anchor_rz, EXPECTED_RZ_DEG) > MAXIMUM_RZ_ERROR_DEG:
        raise ValueError(
            f"P22 Rz={anchor_rz:.4f} deg differs from the approved Stage-4 "
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
    """Solve one exact Cartesian target on the continuous P22 IK branch.

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
    p22_joints: Sequence[float],
) -> dict[tuple[float, float], list[float]]:
    """Generate absolute joint targets while preserving P22 J3 and Rz.

    For an offset ``(dx, dy)``, the desired J4-axis position is
    ``W_target = FK(P22.J1, P22.J2) + [dx, dy]``.  ``solve_joints`` then finds
    the continuous IK branch and selects J4 so that
    ``Rz = J1 + J2 + J4 - 90`` remains exactly equal to the P22 value.
    """

    anchor, anchor_x, anchor_y, anchor_rz = _validated_anchor_context(p22_joints)
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
) -> str:
    return (
        f"TASK9|target={ANCHOR_TARGET_NAME}|dx={dx_mm:+.3f}|dy={dy_mm:+.3f}|"
        f"frame={frame_index:02d}/{FRAMES_PER_OFFSET:02d}"
    )


def _append_frame_burst(
    actions: list[dict],
    dx_mm: float,
    dy_mm: float,
) -> None:
    actions.append({"type": "wait", "seconds": SETTLE_SECONDS})
    for frame_index in range(1, FRAMES_PER_OFFSET + 1):
        actions.append(
            {
                "type": "record_point",
                "name": _record_name(dx_mm, dy_mm, frame_index),
            }
        )
        actions.append({"type": "capture", "source": CAMERA_SOURCE})


def build_action_for_preset(
    preset_name: str,
    p22_joints: Sequence[float],
) -> dict:
    """Build a validated Task9 action from one explicit P22 preset."""

    anchor = _finite_joints(list(p22_joints), preset_name)
    anchor, anchor_x, anchor_y, anchor_rz = _validated_anchor_context(anchor)
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
        substeps = _cartesian_substeps(current_offset, end_offset)
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
        _append_frame_burst(actions, dx_mm, dy_mm)

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
        "name": "task9 相机1局部XY图像Jacobian标定",
        "description": (
            "从P22 float开始，只在安全观察高度执行±2mm的3×3局部XY网格；"
            "每条2mm网格边拆成两个不超过1mm的Cartesian中转目标；"
            "每个偏移采集12张相机1照片并记录实际机械臂状态，结束后回到P22。"
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


def build_action() -> dict:
    preset_name, p22_joints = _load_p22_preset()
    return build_action_for_preset(preset_name, p22_joints)


def create_task_runtime(output_dir: Path, parent=None):
    """Attach Stage-3 processing and the Stage-5 robust batch fit."""

    from scara.vision.xy_image_jacobian_runtime import (
        create_camera1_xy_image_jacobian_runtime,
    )

    _preset_name, p22_joints = _load_p22_preset()
    return create_camera1_xy_image_jacobian_runtime(
        output_dir=output_dir,
        project_root=_project_root(),
        anchor_target_name=ANCHOR_TARGET_NAME,
        anchor_point_T_mm=[-50.0, -50.0, -2.0],
        anchor_preset_joints=p22_joints,
        command_offsets_xy_mm=GRID_OFFSETS_XY_MM,
        frames_per_offset=FRAMES_PER_OFFSET,
        parent=parent,
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
