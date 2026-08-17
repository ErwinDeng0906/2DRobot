"""Task12: scan Stage-3 observability at every P00-P55 slot.

The imported task deliberately contains only deterministic high-plane motion,
robot-state records, waits, and camera-1 captures.  It starts from the taught
``P00 float`` preset, visits all 36 slots through an adjacent-cell closed snake,
and returns to the exact taught start.  Stage-3 detection, temporal tracking,
quality aggregation, annotations, and report persistence live in
``scara.vision.stage3_visibility_scan``.

There is no contact-height motion, Z descent, DO, vacuum, Jacobian correction,
or closed-loop motion in this task.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ACTION_API_VERSION = 1
CAMERA_SOURCE = 1
FRAMES_PER_SLOT = 20
SETTLE_SECONDS = 0.8
FRAME_INTERVAL_SECONDS = 0.10
START_TOLERANCE_DEG_OR_MM = 0.25
MOVE_TOLERANCE_DEG_OR_MM = 0.02
J3_TOLERANCE_MM = 0.15

# A 25 mm slot-to-slot edge is subdivided.  The bound is intentionally the
# same conservative scale already exercised by Task11.
# Near the P00 edge, the controller's sequential J1 then J2 execution can move
# the wrist roughly twice the simultaneous-Cartesian endpoint step.  0.60 mm
# keeps the audited sequential transient below the independent 1.50 mm ceiling
# over the complete 6 x 6 route.
MAX_CARTESIAN_TRANSIT_STEP_MM = 0.60
MAX_SEQUENTIAL_TRANSIENT_XY_MM = 1.50
MAX_SEQUENTIAL_TRANSIENT_RZ_DEG = 0.35


def _closed_snake_route() -> tuple[str, ...]:
    """Visit all 36 cells through adjacent 25 mm edges and finish near P00.

    Row 0 is traversed completely.  Rows 1-5 snake over columns 1-5, then the
    remaining column 0 is followed upward.  The final P10 -> P00 return is one
    adjacent edge, avoiding a long 125 mm unsampled return move.
    """

    route = [f"P0{column}" for column in range(6)]
    for row in range(1, 6):
        columns = range(5, 0, -1) if row % 2 else range(1, 6)
        route.extend(f"P{row}{column}" for column in columns)
    route.extend(f"P{row}0" for row in range(5, 0, -1))
    assert len(route) == 36 and len(set(route)) == 36
    return tuple(route)


TARGET_ORDER = _closed_snake_route()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _finite_joints(value: object, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{label} 必须包含J1/J2/J3/J4四个数值")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} 包含NaN或Inf")
    return result


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"找不到{label}：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}不是有效JSON：{path}（{exc}）") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label}顶层必须是对象")
    return payload


def _load_scan_inputs() -> tuple[list[float], dict[str, list[float]], list[list[float]]]:
    root = _project_root()
    presets = _load_json(root / "scara_presets.json", "预设点文件")
    p00_value = presets.get("P00 float", presets.get("P00_float"))
    if p00_value is None:
        raise ValueError(
            "Task12要求已示教的 P00 float 作为固定高度安全起点；"
            "请先将吸盘中心置于P00上方的观察高度并保存该preset。"
        )
    p00_joints = _finite_joints(p00_value, "P00 float")

    geometry = _load_json(
        root / "src/scara/calib/tray_board_geometry.json",
        "Tray Board几何",
    )
    slots = geometry.get("slots")
    tray_frame = geometry.get("tray_frame")
    if not isinstance(slots, Mapping) or not isinstance(tray_frame, Mapping):
        raise ValueError("Tray Board几何缺少slots或tray_frame")
    required = {f"P{row}{column}" for row in range(6) for column in range(6)}
    if set(slots) != required:
        raise ValueError("Task12要求Tray几何完整包含P00-P55共36个槽")
    slot_points = {
        name: [float(value) for value in slots[name]] for name in sorted(required)
    }
    if any(len(point) != 3 or not all(math.isfinite(v) for v in point) for point in slot_points.values()):
        raise ValueError("Tray槽中心必须是有限的三维坐标")

    rotation = tray_frame.get("rotation_mechanical_from_tray")
    if (
        not isinstance(rotation, list)
        or len(rotation) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in rotation)
    ):
        raise ValueError("Tray几何缺少3x3 rotation_mechanical_from_tray")
    rotation_values = [[float(value) for value in row] for row in rotation]
    if not all(math.isfinite(value) for row in rotation_values for value in row):
        raise ValueError("Tray旋转矩阵包含NaN或Inf")
    return p00_joints, slot_points, rotation_values


def _angle_error(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def _tray_to_mechanical_xy(
    p00_mechanical_xy: Sequence[float],
    rotation_mechanical_from_tray: Sequence[Sequence[float]],
    point_T_mm: Sequence[float],
) -> tuple[float, float]:
    """Apply the measured Tray XY axes while retaining taught P00 as anchor."""

    tx, ty = float(point_T_mm[0]), float(point_T_mm[1])
    rotation = rotation_mechanical_from_tray
    return (
        float(p00_mechanical_xy[0]) + float(rotation[0][0]) * tx + float(rotation[0][1]) * ty,
        float(p00_mechanical_xy[1]) + float(rotation[1][0]) * tx + float(rotation[1][1]) * ty,
    )


def _solve_xy(
    mechanical_xy: Sequence[float],
    imaging_j3_mm: float,
    absolute_rz_deg: float,
    reference_joints: Sequence[float],
) -> list[float]:
    from scara.pipeline.kinematics import fk_wrist, ik_wrist, j4_for_rz, rz_of

    x_mm, y_mm = float(mechanical_xy[0]), float(mechanical_xy[1])
    branches = ik_wrist(x_mm, y_mm)
    if not branches:
        raise ValueError(f"Task12目标机械XY=({x_mm:.3f},{y_mm:.3f})mm不可达")
    j1, j2 = min(
        branches,
        key=lambda branch: (
            _angle_error(branch[0], reference_joints[0])
            + _angle_error(branch[1], reference_joints[1])
            + _angle_error(
                j4_for_rz(branch[0], branch[1], absolute_rz_deg),
                reference_joints[3],
            )
        ),
    )
    target = [
        float(j1),
        float(j2),
        float(imaging_j3_mm),
        float(j4_for_rz(j1, j2, absolute_rz_deg)),
    ]
    actual_xy = fk_wrist(target[0], target[1])
    if math.dist(actual_xy, (x_mm, y_mm)) > 0.02:
        raise ValueError("Task12 IK回代误差超过0.02mm")
    if _angle_error(rz_of(target[0], target[1], target[3]), absolute_rz_deg) > 0.01:
        raise ValueError("Task12 IK未保持固定绝对Rz")
    return target


def _substeps(
    start_T_xy_mm: Sequence[float],
    end_T_xy_mm: Sequence[float],
) -> list[tuple[float, float]]:
    dx = float(end_T_xy_mm[0]) - float(start_T_xy_mm[0])
    dy = float(end_T_xy_mm[1]) - float(start_T_xy_mm[1])
    distance = math.hypot(dx, dy)
    if distance <= 1e-12:
        return []
    count = max(1, int(math.ceil(distance / MAX_CARTESIAN_TRANSIT_STEP_MM - 1e-12)))
    return [
        (
            float(start_T_xy_mm[0]) + dx * index / count,
            float(start_T_xy_mm[1]) + dy * index / count,
        )
        for index in range(1, count + 1)
    ]


def _sequential_motion_audit(
    current: Sequence[float],
    target: Sequence[float],
    required_rz_deg: float,
) -> tuple[float, float]:
    """Audit the controller's real J1 -> J2 -> J3 -> J4 execution order."""

    from scara.pipeline.kinematics import fk_wrist, rz_of

    states = [list(float(value) for value in current)]
    state = list(states[0])
    for axis in range(4):
        state = list(state)
        state[axis] = float(target[axis])
        states.append(state)
    xy = [fk_wrist(state[0], state[1]) for state in states]
    maximum_xy_step = max(math.dist(left, right) for left, right in zip(xy, xy[1:]))
    maximum_rz_departure = max(
        _angle_error(rz_of(state[0], state[1], state[3]), required_rz_deg)
        for state in states
    )
    return float(maximum_xy_step), float(maximum_rz_departure)


def _record_name(target: str, frame_index: int) -> str:
    return f"TASK12|{target}|frame={frame_index:02d}/{FRAMES_PER_SLOT:02d}"


def _append_capture_burst(actions: list[dict[str, Any]], target: str) -> None:
    actions.append({"type": "wait", "seconds": SETTLE_SECONDS})
    for frame_index in range(1, FRAMES_PER_SLOT + 1):
        actions.append({"type": "record_point", "name": _record_name(target, frame_index)})
        actions.append({"type": "capture", "source": CAMERA_SOURCE})
        if frame_index < FRAMES_PER_SLOT:
            actions.append({"type": "wait", "seconds": FRAME_INTERVAL_SECONDS})


def build_action() -> dict[str, Any]:
    from scara.pipeline.kinematics import fk_wrist, rz_of

    p00_joints, slot_points, rotation = _load_scan_inputs()
    p00_xy = fk_wrist(p00_joints[0], p00_joints[1])
    imaging_j3 = float(p00_joints[2])
    fixed_rz = float(rz_of(p00_joints[0], p00_joints[1], p00_joints[3]))
    actions: list[dict[str, Any]] = [
        {
            "type": "assert_joints",
            "name": "确认Task12固定高度起点 P00 float",
            "joints": list(p00_joints),
            "tolerance": START_TOLERANCE_DEG_OR_MM,
        }
    ]
    current_T_xy = (0.0, 0.0)
    reference = list(p00_joints)

    def move_to(target_name: str, *, exact_p00: bool = False) -> None:
        nonlocal current_T_xy, reference
        target_T = slot_points[target_name]
        route = _substeps(current_T_xy, target_T[:2])
        for index, intermediate_T_xy in enumerate(route, start=1):
            intermediate_point_T = [intermediate_T_xy[0], intermediate_T_xy[1], target_T[2]]
            mechanical_xy = _tray_to_mechanical_xy(p00_xy, rotation, intermediate_point_T)
            solved = _solve_xy(mechanical_xy, imaging_j3, fixed_rz, reference)
            is_final = index == len(route)
            target_joints = list(p00_joints) if exact_p00 and is_final else solved
            transient_xy, transient_rz = _sequential_motion_audit(reference, target_joints, fixed_rz)
            if transient_xy > MAX_SEQUENTIAL_TRANSIENT_XY_MM + 1e-9:
                raise ValueError(
                    f"Task12到{target_name}的逐轴中转XY={transient_xy:.3f}mm，"
                    f"超过{MAX_SEQUENTIAL_TRANSIENT_XY_MM:.2f}mm"
                )
            if transient_rz > MAX_SEQUENTIAL_TRANSIENT_RZ_DEG + 1e-9:
                raise ValueError(
                    f"Task12到{target_name}的逐轴中转Rz偏差={transient_rz:.3f}°，"
                    f"超过{MAX_SEQUENTIAL_TRANSIENT_RZ_DEG:.2f}°"
                )
            actions.append(
                {
                    "type": "move_joints",
                    "name": (
                        f"Task12观察点 {target_name}"
                        if is_final
                        else f"Task12安全中转至{target_name} {index:02d}/{len(route):02d}"
                    ),
                    "joints": target_joints,
                    "tolerance": MOVE_TOLERANCE_DEG_OR_MM,
                    "require_current_j3_mm": imaging_j3,
                    "j3_tolerance_mm": J3_TOLERANCE_MM,
                }
            )
            reference = list(target_joints)
        current_T_xy = (float(target_T[0]), float(target_T[1]))

    for index, target in enumerate(TARGET_ORDER):
        if index:
            move_to(target)
        _append_capture_burst(actions, target)
    move_to("P00", exact_p00=True)

    return {
        "name": "task12_code visibility scan — 36槽Stage3可观测性扫描",
        "description": (
            "从已示教P00 float固定观察高度出发，以相邻槽回环蛇形路线扫描P00-P55共36槽；"
            "每槽等待稳定后由相机1采20帧，共720张。槽间路径拆成≤0.60mm小步，固定J3和绝对Rz，"
            "结束返回P00。只评估A-H、RANSAC、重投影及帧间稳定性；不做Jacobian修正、不下降Z、"
            "不触发DO或真空。"
        ),
        "camera_model": {
            "offset_mm": 0.0,
            "angle_reference": "world_negative_y",
            "positive_rotation": "counter_clockwise_from_above",
        },
        "actions": actions,
    }


def create_task_runtime(output_dir: Path, parent=None):
    from scara.vision.stage3_visibility_scan import create_stage3_visibility_scan_runtime

    _p00, slot_points, _rotation = _load_scan_inputs()
    return create_stage3_visibility_scan_runtime(
        output_dir=output_dir,
        project_root=_project_root(),
        target_order=TARGET_ORDER,
        slot_points_T_mm=slot_points,
        frames_per_slot=FRAMES_PER_SLOT,
        parent=parent,
    )


if __name__ == "__main__":
    preview = build_action()
    kinds = [step["type"] for step in preview["actions"]]
    print(preview["name"])
    print(
        f"slots={len(TARGET_ORDER)}; moves={kinds.count('move_joints')}; "
        f"points={kinds.count('record_point')}; photos={kinds.count('capture')}"
    )
    print("J3=fixed; Rz=fixed; Z descent=none; DO/vacuum=none; final=P00 float")
