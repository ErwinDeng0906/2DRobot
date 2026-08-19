"""Task 7: repeated nine-point ChArUco acquisition for camera 1.

One pass scans a 120 x 120 mm square around taught preset ``X`` using a 3 x 3
serpentine route.  After nine images the robot returns to ``X`` and the generic
action runner pauses at an ``operator_checkpoint``.  The operator can then:

* choose ``继续采集`` after fixing a different rigid board pose, which repeats
  the same nine points in the same output folder; or
* choose ``结束采集``, which finishes normally and calibrates once from every
  image accumulated by this task.

The action never changes Z or R.  All robot points, photos, collection-round
numbers, and operator decisions are appended to one ``points.json`` manifest.
The reusable ChArUco runtime saves the final calculation separately as
``camera1_intrinsics.json`` and only updates the approved project copy when its
pose-diversity quality gate passes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence


ACTION_API_VERSION = 1

START_PRESET_NAME = "X"
CAMERA_SOURCE = 1
GRID_ROWS = 3
GRID_COLUMNS = 3
SQUARE_SIZE_MM = 120.0
HALF_SPAN_MM = SQUARE_SIZE_MM / 2.0
TARGET_DWELL_SECONDS = 2.0
START_TOLERANCE = 0.2
SCARA_LINK1_MM = 225.0
SCARA_LINK2_MM = 175.0


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _finite_values(
    values: Sequence[float], expected: int, label: str
) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != expected:
        raise ValueError(f"{label} 必须包含 {expected} 个数值")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} 包含非有限数值")
    return result


def _load_start_preset() -> list[float]:
    """Load the taught four-joint centre ``X`` from scara_presets.json."""
    preset_path = _project_root() / "scara_presets.json"
    try:
        raw = json.loads(preset_path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"找不到预设文件：{preset_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"预设文件不是有效JSON：{preset_path}（{exc}）"
        ) from exc
    joints = raw.get(START_PRESET_NAME)
    if joints is None:
        raise ValueError(
            f"{preset_path.name} 中没有预设点 {START_PRESET_NAME!r}。"
            "请先把机械臂移动到120×120mm采集范围中心并保存为 X。"
        )
    return _finite_values(joints, 4, f"预设点 {START_PRESET_NAME}")


def _fk_wrist_xy(joints: Sequence[float]) -> tuple[float, float]:
    """Return J4-axis world XY from J1/J2 planar forward kinematics."""
    j1_deg, j2_deg, _j3_mm, _j4_deg = _finite_values(
        joints, 4, "关节值"
    )
    j1 = math.radians(j1_deg)
    forearm = math.radians(j1_deg + j2_deg)
    return (
        SCARA_LINK1_MM * math.cos(j1)
        + SCARA_LINK2_MM * math.cos(forearm),
        SCARA_LINK1_MM * math.sin(j1)
        + SCARA_LINK2_MM * math.sin(forearm),
    )


def build_grid_targets(start_joints: Sequence[float]) -> list[dict]:
    """Return a reachable rotated 120 mm square in row-wise zigzag order.

    A world-axis-aligned 120 mm square around the currently taught ``X`` reaches
    about 407.9 mm and exceeds the 400 mm arm span.  Rotation does not change
    the requested square: use one local axis along the base-to-X radial
    direction and the other perpendicular to it.  For the current X, the
    farthest corner is then about 391.8 mm from the base.
    """
    coordinates = (-HALF_SPAN_MM, 0.0, HALF_SPAN_MM)
    centre_x, centre_y = _fk_wrist_xy(start_joints)
    centre_radius = math.hypot(centre_x, centre_y)
    if centre_radius < 1e-9:
        raise ValueError("预设X位于机器人基座原点，无法定义安全扫描方向")
    radial_x = centre_x / centre_radius
    radial_y = centre_y / centre_radius
    tangent_x = -radial_y
    tangent_y = radial_x
    targets: list[dict] = []
    sequence = 0
    for row, grid_v_mm in enumerate(coordinates):
        columns = (
            range(GRID_COLUMNS)
            if row % 2 == 0
            else range(GRID_COLUMNS - 1, -1, -1)
        )
        for column in columns:
            sequence += 1
            grid_u_mm = coordinates[column]
            offset_x_mm = grid_u_mm * radial_x + grid_v_mm * tangent_x
            offset_y_mm = grid_u_mm * radial_y + grid_v_mm * tangent_y
            targets.append(
                {
                    "sequence": sequence,
                    "row": row,
                    "column": column,
                    "name": f"CAL3_R{row + 1:02d}_C{column + 1:02d}",
                    "grid_u_mm": float(grid_u_mm),
                    "grid_v_mm": float(grid_v_mm),
                    "offset_x_mm": float(offset_x_mm),
                    "offset_y_mm": float(offset_y_mm),
                }
            )
    return targets


def _validate_planar_reach(
    start_joints: Sequence[float], targets: Sequence[dict]
) -> None:
    """Sample every transfer, including the final return to X, for reach."""
    centre_x, centre_y = _fk_wrist_xy(start_joints)
    min_radius = abs(SCARA_LINK1_MM - SCARA_LINK2_MM)
    max_radius = SCARA_LINK1_MM + SCARA_LINK2_MM
    offsets = [
        (float(target["offset_x_mm"]), float(target["offset_y_mm"]))
        for target in targets
    ]
    offsets.append((0.0, 0.0))
    previous = (0.0, 0.0)
    for current in offsets:
        for sample in range(21):
            fraction = sample / 20.0
            offset_x = previous[0] + fraction * (current[0] - previous[0])
            offset_y = previous[1] + fraction * (current[1] - previous[1])
            radius = math.hypot(centre_x + offset_x, centre_y + offset_y)
            if radius < min_radius - 1e-6 or radius > max_radius + 1e-6:
                raise ValueError(
                    "以X为中心的120×120mm路径超出SCARA理论平面臂展："
                    f"偏移({offset_x:+.1f}, {offset_y:+.1f})mm处半径为"
                    f"{radius:.1f}mm，允许范围为"
                    f"{min_radius:.1f}–{max_radius:.1f}mm。请重新示教X。"
                )
        previous = current


def _append_capture(actions: list[dict], target: dict) -> None:
    name = (
        f"{target['name']} / X{target['offset_x_mm']:+.3f}mm "
        f"Y{target['offset_y_mm']:+.3f}mm"
    )
    actions.append({"type": "wait", "seconds": TARGET_DWELL_SECONDS})
    actions.append({"type": "record_point", "name": name})
    actions.append({"type": "capture", "source": CAMERA_SOURCE})


def build_action() -> dict:
    """Build one nine-point pass plus an operator-controlled repeat point."""
    start_joints = _load_start_preset()
    targets = build_grid_targets(start_joints)
    _validate_planar_reach(start_joints, targets)

    actions: list[dict] = [
        {
            "type": "assert_joints",
            "name": "确认标定中心 X",
            "joints": list(start_joints),
            "tolerance": START_TOLERANCE,
        }
    ]
    repeat_from_index = len(actions)
    previous_x = 0.0
    previous_y = 0.0
    for target in targets:
        target_x = float(target["offset_x_mm"])
        target_y = float(target["offset_y_mm"])
        delta_x = target_x - previous_x
        delta_y = target_y - previous_y
        if math.hypot(delta_x, delta_y) > 1e-9:
            actions.append(
                {
                    "type": "move_xyzr",
                    "name": f"移动到 {target['name']}",
                    "x_mm": delta_x,
                    "y_mm": delta_y,
                    "z_mm": 0.0,
                    "r_deg": 0.0,
                }
            )
        _append_capture(actions, target)
        previous_x, previous_y = target_x, target_y

    actions.append(
        {
            "type": "move_xyzr",
            "name": "返回标定中心 X 并暂停",
            "x_mm": -previous_x,
            "y_mm": -previous_y,
            "z_mm": 0.0,
            "r_deg": 0.0,
        }
    )
    actions.append(
        {
            "type": "operator_checkpoint",
            "name": "本姿态九点采集完成",
            "message": (
                "机械臂已经返回预设中心X，并停止下发运动。\n"
                "如需继续：先保持机械臂停止，使用刚性支架改变并固定标定板"
                "姿态，检查120×120mm范围及高度间隙，人员完全离开工作区后"
                "点击“继续采集”。\n"
                "如已完成所有计划姿态，点击“结束采集”，程序将使用本次任务"
                "累计的全部照片统一计算内参。至少需要两个姿态（18张），"
                "推荐5–8个多方向姿态。"
            ),
            "continue_text": "继续采集",
            "finish_text": "结束采集",
            "repeat_from_index": repeat_from_index,
        }
    )

    return {
        "name": "task7 相机1 多姿态九点循环内参标定",
        "description": (
            "以预设X为中心执行120×120mm、3×3蛇形九点采集；路径保持正方形"
            "尺寸并按X的径向/切向旋转，以避开当前X的400mm臂展边界。每轮"
            "返回X后暂停，由人员选择改变标定板姿态后继续，或结束并使用"
            "本次全部姿态统一计算相机内参。全程只运动世界XY，不改变Z/R。"
        ),
        "camera_model": {
            "offset_mm": 0.0,
            "angle_reference": "world_negative_y",
            "positive_rotation": "counter_clockwise_from_above",
        },
        "actions": actions,
    }


def create_task_runtime(output_dir: Path, parent=None):
    """Create one open-ended runtime shared by every repeated board pose."""
    from scara.vision.charuco_calibration_runtime import (
        create_camera1_charuco_runtime,
    )

    return create_camera1_charuco_runtime(
        output_dir,
        _project_root(),
        parent,
        planned_images=None,
        scan_size_mm=SQUARE_SIZE_MM,
        images_per_pose=GRID_ROWS * GRID_COLUMNS,
        operator_reposition_between_passes=True,
    )


if __name__ == "__main__":
    task = build_action()
    targets = build_grid_targets(_load_start_preset())
    print(task["name"])
    print(f"points per pose: {len(targets)}; action templates: {len(task['actions'])}")
    print("square: 120.0 x 120.0 mm around preset X")
    print("camera source: 1; every movement has z_mm=0 and r_deg=0")
    print("operator checkpoint: continue repeats; finish calibrates all images")
