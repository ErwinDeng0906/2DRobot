"""Task 4: record source-2 video while scanning task2 row P00..P05.

At each station the arm descends 5 mm five times, then rises 25 mm once to the
station's float height.  This module is import-safe; the SCARA UI performs
camera/robot access only after the operator confirms execution.
"""

from __future__ import annotations

import importlib.util
import math
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Sequence


ACTION_API_VERSION = 1

TARGET_DWELL_SECONDS = 2.0
Z_STEP_MM = 5.0
DESCENT_STEP_COUNT = 5
TOTAL_DESCENT_MM = Z_STEP_MM * DESCENT_STEP_COUNT
VIDEO_SOURCE = 2
VIDEO_FPS = 20.0
VIDEO_FILENAME = "2_video.mp4"


@lru_cache(maxsize=1)
def _task2_module() -> ModuleType:
    """Load the sibling task2 geometry so row 0 cannot drift from task2.

    ``task2.py`` is loaded by absolute sibling path because the directory name
    contains a space and is not a Python package.  Importing task2 is safe: its
    hardware-free preview is protected by ``if __name__ == '__main__'``.
    """
    task2_path = Path(__file__).resolve().with_name("task2.py")
    spec = importlib.util.spec_from_file_location("_scara_task4_task2_geometry", task2_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法载入task2网格定义：{task2_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def camera_position_from_pose(pose: Sequence[float]) -> dict[str, float]:
    """Calculate source-2 XYZ using the same 20 mm/Rz model as task2.

    Delegating to task2 guarantees that every point recorded in ``points.json``
    uses exactly the same coordinate convention and equations as the 6x6 scan.
    In that model Rz is measured counter-clockwise from world -Y::

        camera2_x = centre_x + 20*sin(Rz)
        camera2_y = centre_y - 20*cos(Rz)
        camera2_z = centre_z
    """
    return dict(_task2_module().camera_position_from_pose(pose))


def build_row_targets() -> list[dict]:
    """Return task2's first zigzag row in P00,P01,...,P05 order.

    Task2 bilinearly interpolates the taught P00/P05/P50/P55 world coordinates
    and solves a continuous SCARA IK branch.  Filtering ``row == 0`` therefore
    reuses its exact first six targets rather than independently approximating
    them here.
    """
    row = [
        dict(target)
        for target in _task2_module().build_grid_targets()
        if int(target["row"]) == 0
    ]
    expected_names = [f"P0{column}" for column in range(6)]
    if [target["name"] for target in row] != expected_names:
        raise ValueError(
            "task2第一行顺序异常："
            + " -> ".join(target["name"] for target in row)
        )
    return row


def _append_settle_and_record(
    actions: list[dict],
    point_name: str,
) -> None:
    """Wait two seconds at one reached target, then record live robot state."""
    actions.append({"type": "wait", "seconds": TARGET_DWELL_SECONDS})
    actions.append({"type": "record_point", "name": point_name})


def _append_vertical_cycle(actions: list[dict], station: str) -> None:
    """Append five -5 mm moves followed by one +25 mm float-height return.

    Downward target ``i`` is::

        depth_i_mm = i * 5 mm,  i = 1..5

    Hence the fifth target is 25 mm below float.  The single return displacement
    is the exact inverse of the accumulated descent::

        return_delta_z_mm = 5 * 5 mm = +25 mm

    World +Z raises the arm away from the platform.  Every reached target waits
    two seconds and is recorded in ``points.json``; no still-photo step is added.
    """
    for step_index in range(1, DESCENT_STEP_COUNT + 1):
        depth_mm = step_index * Z_STEP_MM
        point_name = f"{station} / float-{depth_mm:.0f}mm"
        actions.append(
            {
                "type": "move_xyzr",
                "name": point_name,
                "x_mm": 0.0,
                "y_mm": 0.0,
                "z_mm": -Z_STEP_MM,
                "r_deg": 0.0,
            }
        )
        _append_settle_and_record(actions, point_name)

    actions.append(
        {
            "type": "move_xyzr",
            "name": f"{station} / 返回float",
            "x_mm": 0.0,
            "y_mm": 0.0,
            "z_mm": TOTAL_DESCENT_MM,
            "r_deg": 0.0,
        }
    )
    _append_settle_and_record(actions, f"{station} float / 返回")


def build_action() -> dict:
    """Build the six-station vertical scan and source-2 video session."""
    row = build_row_targets()
    p00 = row[0]
    actions: list[dict] = [
        {
            "type": "assert_joints",
            "name": "确认起点 P00 float",
            "joints": list(p00["joints"]),
            "tolerance": 0.2,
        },
        # The action executor creates the timestamp folder before this step,
        # opens/warms source 2 before any motion, and writes this MP4 inside it.
        {
            "type": "start_video",
            "source": VIDEO_SOURCE,
            "filename": VIDEO_FILENAME,
            "fps": VIDEO_FPS,
        },
    ]

    _append_settle_and_record(actions, "P00 float / 初始")
    _append_vertical_cycle(actions, "P00")

    previous_float_j3_mm = float(p00["joints"][2])
    for target in row[1:]:
        target_joints = [float(value) for value in target["joints"]]
        if not all(math.isfinite(value) for value in target_joints):
            raise ValueError(f"{target['name']} 关节目标含非有限数字")
        actions.append(
            {
                "type": "move_joints",
                "name": f"安全转移到 {target['name']}",
                "joints": target_joints,
                "tolerance": 0.2,
                # This blocks lateral/J4 motion unless the preceding +25 mm
                # return actually restored the previous station's float height.
                "require_current_j3_mm": previous_float_j3_mm,
                "j3_tolerance_mm": 0.2,
            }
        )
        _append_settle_and_record(actions, f"{target['name']} float / 到达")
        _append_vertical_cycle(actions, target["name"])
        previous_float_j3_mm = target_joints[2]

    # No ``capture`` steps exist: stopping finalizes the single source-2 MP4.
    actions.append({"type": "stop_video", "source": VIDEO_SOURCE})

    return {
        "name": "task4 P00到P05源2录像",
        "description": (
            "沿task2第一行P00→P01→P02→P03→P04→P05移动；"
            "每站分五次下降25mm，再一次上升25mm回到float；不拍摄JPG，"
            "源2全程录制2_video.mp4，并记录全部float/深度/返回点。"
        ),
        "camera_model": {
            "offset_mm": 20.0,
            "angle_reference": "world_negative_y",
            "positive_rotation": "counter_clockwise_from_above",
        },
        "actions": actions,
    }


if __name__ == "__main__":
    # Hardware-free preview.
    targets = build_row_targets()
    preview = build_action()
    actions = preview["actions"]
    print("route:", " -> ".join(target["name"] for target in targets))
    print(f"task4: {len(actions)} actions")
    print(
        "recorded points:",
        sum(step["type"] == "record_point" for step in actions),
        "; photos:",
        sum(step["type"] == "capture" for step in actions),
        "; videos:",
        sum(step["type"] == "start_video" for step in actions),
    )
