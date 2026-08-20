"""Task11: acquire and validate the 20 x 20 mm P22 wide Jacobian field.

This imported task contains only deterministic joint moves, waits, robot-state
records, and camera-1 captures.  Stage3 projection, robust aggregation,
global/quadratic model fitting, independent validation, hash locking, and
installation are implemented in ``scara.vision.wide_xy_jacobian_runtime``.

Acquisition contract
--------------------
* 5 x 5 training nodes at X/Y = {-10,-5,0,+5,+10} mm.
* Two visits from different route directions, five frames per visit.
* A shifted 4 x 4 validation grid at {-7.5,-2.5,+2.5,+7.5} mm which is never
  used for fitting.
* Every Cartesian route edge is subdivided to <=1 mm before IK.
* J3 and absolute Rz are fixed.  No Z motion, DO, vacuum, or contact action.
* The route starts and ends at the exact taught ``P22 float`` preset.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ACTION_API_VERSION = 1
TARGET_NAME = "P22"
CAMERA_SOURCE = 1
FRAMES_PER_VISIT = 5
SETTLE_SECONDS = 0.8
FRAME_INTERVAL_SECONDS = 0.12
START_TOLERANCE_DEG_OR_MM = 0.15
MOVE_TOLERANCE_DEG_OR_MM = 0.02
J3_TOLERANCE_MM = 0.15
# Use 0.90 mm rather than the nominal 1.00 mm ceiling so the controller's
# sequential J1/J2 execution also keeps transient absolute-Rz below 0.30 deg.
MAX_CARTESIAN_TRANSIT_STEP_MM = 0.90
MAX_SEQUENTIAL_TRANSIENT_XY_MM = 1.50
MAX_SEQUENTIAL_TRANSIENT_RZ_DEG = 0.30

TRAIN_AXIS_MM = (-10.0, -5.0, 0.0, 5.0, 10.0)
VALIDATION_AXIS_MM = (-7.5, -2.5, 2.5, 7.5)


def _snake(axis: Sequence[float]) -> list[tuple[float, float]]:
    route: list[tuple[float, float]] = []
    for row_index, y_mm in enumerate(axis):
        xs = list(axis) if row_index % 2 == 0 else list(reversed(axis))
        route.extend((float(x_mm), float(y_mm)) for x_mm in xs)
    return route


TRAINING_OFFSETS_XY_MM = tuple(_snake(TRAIN_AXIS_MM))
VALIDATION_OFFSETS_XY_MM = tuple(_snake(VALIDATION_AXIS_MM))

# The first visit must be P22 so the runtime can bind the measured world-XY
# anchor.  The remainder follows a complete snake.  Pass 2 approaches the same
# nodes in the reverse order after an unsampled return through P22.
PASS1_OFFSETS_XY_MM = (
    (0.0, 0.0),
    *tuple(offset for offset in TRAINING_OFFSETS_XY_MM if offset != (0.0, 0.0)),
)
PASS2_OFFSETS_XY_MM = tuple(reversed(TRAINING_OFFSETS_XY_MM))

VISITS: tuple[tuple[str, int, tuple[float, float]], ...] = tuple(
    [("train", 1, offset) for offset in PASS1_OFFSETS_XY_MM]
    + [("train", 2, offset) for offset in PASS2_OFFSETS_XY_MM]
    + [("validation", 1, offset) for offset in VALIDATION_OFFSETS_XY_MM]
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _finite_joints(value: object, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{label} 必须包含J1/J2/J3/J4")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} 包含非有限数值")
    return result


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
            return name, _finite_joints(raw[name], name)
    raise ValueError("Task11要求已经示教的 P22 float 预设")


def _angle_error(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def _solve_offset(
    anchor_joints: Sequence[float],
    anchor_xy: tuple[float, float],
    anchor_rz_deg: float,
    offset_xy_mm: tuple[float, float],
    reference_joints: Sequence[float],
) -> list[float]:
    from scara.pipeline.kinematics import (
        fk_wrist,
        ik_wrist,
        j4_for_rz,
        rz_of,
        solve_joints,
    )

    target_x = float(anchor_xy[0]) + float(offset_xy_mm[0])
    target_y = float(anchor_xy[1]) + float(offset_xy_mm[1])
    rounded = solve_joints(
        target_x,
        target_y,
        float(anchor_joints[2]),
        rz_deg=float(anchor_rz_deg),
        ref_joints=list(reference_joints),
    )
    if rounded is None:
        raise ValueError(f"Task11偏移{offset_xy_mm}不可达")
    branches = ik_wrist(target_x, target_y)
    chosen_j1, chosen_j2 = min(
        branches,
        key=lambda branch: (
            _angle_error(branch[0], rounded[0])
            + _angle_error(branch[1], rounded[1])
        ),
    )
    joints = [
        float(chosen_j1),
        float(chosen_j2),
        float(anchor_joints[2]),
        float(j4_for_rz(chosen_j1, chosen_j2, anchor_rz_deg)),
    ]
    actual_xy = fk_wrist(joints[0], joints[1])
    if math.hypot(actual_xy[0] - target_x, actual_xy[1] - target_y) > 0.02:
        raise ValueError(f"Task11偏移{offset_xy_mm} IK闭合误差过大")
    if abs(joints[2] - float(anchor_joints[2])) > 1e-9:
        raise ValueError("Task11 IK改变了J3")
    if _angle_error(rz_of(joints[0], joints[1], joints[3]), anchor_rz_deg) > 0.01:
        raise ValueError("Task11 IK未保持Rz")
    return [float(value) for value in joints]


def _substeps(
    start: tuple[float, float], end: tuple[float, float]
) -> list[tuple[float, float]]:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    distance = math.hypot(dx, dy)
    if distance <= 1e-12:
        return []
    count = max(
        1,
        int(math.ceil(distance / MAX_CARTESIAN_TRANSIT_STEP_MM - 1e-12)),
    )
    return [
        (float(start[0]) + dx * index / count, float(start[1]) + dy * index / count)
        for index in range(1, count + 1)
    ]


def _record_name(
    phase: str,
    pass_index: int,
    visit_index: int,
    offset: tuple[float, float],
    frame_index: int,
) -> str:
    return (
        f"TASK11|target={TARGET_NAME}|phase={phase}|pass={pass_index}|"
        f"visit={visit_index:02d}|dx={offset[0]:+.3f}|dy={offset[1]:+.3f}|"
        f"frame={frame_index:02d}/{FRAMES_PER_VISIT:02d}"
    )


def _append_capture_burst(
    actions: list[dict[str, Any]],
    phase: str,
    pass_index: int,
    visit_index: int,
    offset: tuple[float, float],
) -> None:
    actions.append({"type": "wait", "seconds": SETTLE_SECONDS})
    for frame_index in range(1, FRAMES_PER_VISIT + 1):
        actions.append(
            {
                "type": "record_point",
                "name": _record_name(
                    phase, pass_index, visit_index, offset, frame_index
                ),
            }
        )
        actions.append({"type": "capture", "source": CAMERA_SOURCE})
        if frame_index < FRAMES_PER_VISIT:
            actions.append({"type": "wait", "seconds": FRAME_INTERVAL_SECONDS})


def build_action_for_preset(preset_name: str, p22_joints: Sequence[float]) -> dict:
    from scara.pipeline.kinematics import fk_wrist, rz_of

    anchor = _finite_joints(list(p22_joints), preset_name)
    anchor_xy = tuple(float(value) for value in fk_wrist(anchor[0], anchor[1]))
    anchor_rz = float(rz_of(anchor[0], anchor[1], anchor[3]))
    actions: list[dict[str, Any]] = [
        {
            "type": "assert_joints",
            "name": f"确认Task11起点 {preset_name}",
            "joints": list(anchor),
            "tolerance": START_TOLERANCE_DEG_OR_MM,
        }
    ]
    current_offset = (0.0, 0.0)
    reference = list(anchor)

    def move_to(end: tuple[float, float], name: str, *, exact_anchor: bool = False) -> None:
        nonlocal current_offset, reference
        route = _substeps(current_offset, end)
        for index, offset in enumerate(route, start=1):
            solved = _solve_offset(anchor, anchor_xy, anchor_rz, offset, reference)
            final = index == len(route)
            target = list(anchor) if exact_anchor and final else solved
            actions.append(
                {
                    "type": "move_joints",
                    "name": name if final else (
                        f"TASK11中转 dx={offset[0]:+.3f},dy={offset[1]:+.3f}"
                    ),
                    "joints": target,
                    "tolerance": MOVE_TOLERANCE_DEG_OR_MM,
                    "require_current_j3_mm": anchor[2],
                    "j3_tolerance_mm": J3_TOLERANCE_MM,
                }
            )
            reference = target
        current_offset = (float(end[0]), float(end[1]))

    previous_phase_pass: tuple[str, int] | None = None
    visit_index = 0
    for phase, pass_index, offset in VISITS:
        phase_pass = (phase, pass_index)
        if previous_phase_pass is not None and phase_pass != previous_phase_pass:
            move_to((0.0, 0.0), "Task11阶段切换返回P22")
        previous_phase_pass = phase_pass
        visit_index += 1
        move_to(
            offset,
            f"Task11 {phase} pass={pass_index} visit={visit_index} "
            f"dx={offset[0]:+.1f},dy={offset[1]:+.1f}",
        )
        _append_capture_burst(
            actions, phase, pass_index, visit_index, offset
        )

    move_to((0.0, 0.0), f"Task11结束返回{preset_name}", exact_anchor=True)
    return {
        "name": "task11 P22 20x20mm宽域XY图像Jacobian标定",
        "description": (
            "在P22 float观察高度执行世界XY每轴±10mm的5×5双向训练网格，"
            "并采集独立4×4半格验证点。每条路径拆成≤1mm中转，固定J3和绝对Rz，"
            "共330张相机1照片；结束返回P22。无Z下降、无DO、无真空。"
        ),
        "camera_model": {
            "offset_mm": 0.0,
            "angle_reference": "world_negative_y",
            "positive_rotation": "counter_clockwise_from_above",
        },
        "actions": actions,
    }


def build_action() -> dict:
    return build_action_for_preset(*_load_p22_preset())


def create_task_runtime(output_dir: Path, parent=None):
    from scara.vision.wide_xy_jacobian_runtime import (
        create_camera1_wide_xy_jacobian_runtime,
    )

    _name, joints = _load_p22_preset()
    return create_camera1_wide_xy_jacobian_runtime(
        output_dir=output_dir,
        project_root=_project_root(),
        anchor_target_name=TARGET_NAME,
        anchor_point_T_mm=[-50.0, -50.0, -2.0],
        anchor_preset_joints=joints,
        visits=VISITS,
        training_offsets_xy_mm=TRAINING_OFFSETS_XY_MM,
        validation_offsets_xy_mm=VALIDATION_OFFSETS_XY_MM,
        frames_per_visit=FRAMES_PER_VISIT,
        parent=parent,
    )


if __name__ == "__main__":
    task = build_action()
    kinds = [step["type"] for step in task["actions"]]
    print(task["name"])
    print(
        f"visits={len(VISITS)}; moves={kinds.count('move_joints')}; "
        f"points={kinds.count('record_point')}; photos={kinds.count('capture')}"
    )
    print("Z=none; DO/vacuum=none; final=P22 float")
