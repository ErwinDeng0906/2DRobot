"""Task 5: Task 2's 6 x 6 scan with a 4.3 mm vertical step.

This module deliberately reuses Task 2's corner interpolation, inverse
kinematics, serpentine route, camera geometry, capture schedule, file naming,
and points.json recording contract.  Its only experimental change is the Z
cycle at every grid station:

    float -> -4.3 -> -8.6 -> -12.9 -> -17.2 -> -21.5 mm
          -> -17.2 -> -12.9 -> -8.6 -> -4.3 -> float

Importing this file does not connect to the robot or any camera.  The SCARA UI
executes the returned action only after its normal validation and confirmation.
"""

from __future__ import annotations

import importlib.util
import math
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Iterable, Sequence


ACTION_API_VERSION = 1

# Safety-critical Task 5 height parameters.  Five equal moves descend exactly
# 21.5 mm, and five equal inverse moves return exactly to the station's float
# height before any lateral transfer is allowed.
Z_STEP_MM = 4.3
DESCENT_STEP_COUNT = 5
TOTAL_DESCENT_MM = Z_STEP_MM * DESCENT_STEP_COUNT
TARGET_DWELL_SECONDS = 2.0


@lru_cache(maxsize=1)
def _task2_module() -> ModuleType:
    """Load sibling ``task2.py`` without causing camera or robot activity."""
    task2_path = Path(__file__).with_name("task2.py")
    spec = importlib.util.spec_from_file_location(
        "_scara_task5_task2_geometry",
        task2_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 Task 2：{task2_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def camera_position_from_pose(pose: Sequence[float]) -> dict[str, float]:
    """Return source-2 XYZ using exactly the camera model defined by Task 2."""
    return _task2_module().camera_position_from_pose(pose)


def camera1_position_from_state(
    joints: Sequence[float],
    pose: Sequence[float],
) -> dict[str, float]:
    """Return source-1 XYZ using exactly the camera model defined by Task 2."""
    return _task2_module().camera1_position_from_state(joints, pose)


def build_grid_targets() -> list[dict]:
    """Return independent copies of Task 2's 36 serpentine float targets."""
    targets: list[dict] = []
    for original in _task2_module().build_grid_targets():
        target = dict(original)
        target["joints"] = list(original["joints"])
        targets.append(target)
    return targets


def _wait_record_and_capture(
    actions: list[dict],
    point_name: str,
    sources: Iterable[int],
) -> None:
    """Dwell two seconds, record live robot state, then take requested photos."""
    actions.append({"type": "wait", "seconds": TARGET_DWELL_SECONDS})
    actions.append({"type": "record_point", "name": point_name})
    for source in sources:
        actions.append({"type": "capture", "source": int(source)})


def _depth_label(station: str, depth_mm: float) -> str:
    """Name a station/depth without rounding any 4.3 mm intermediate height."""
    if math.isclose(float(depth_mm), 0.0, abs_tol=1e-9):
        return f"{station} / float"
    return f"{station} / float-{float(depth_mm):.1f}mm"


def _append_vertical_cycle(actions: list[dict], station: str) -> None:
    """Append five -4.3 mm moves and five +4.3 mm moves at one grid cell.

    Every target height has a two-second dwell, a live-state JSON record, and
    one source-2 photo, exactly as in Task 2.  The down sequence reaches
    ``5 * 4.3 = 21.5 mm`` below float.  The up sequence is its exact inverse,
    so the accumulated relative Z displacement is zero before XY/R movement.
    """
    for step_index in range(1, DESCENT_STEP_COUNT + 1):
        depth_mm = step_index * Z_STEP_MM
        point_name = _depth_label(station, depth_mm)
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
        _wait_record_and_capture(actions, point_name, (2,))

    for upward_step in range(1, DESCENT_STEP_COUNT + 1):
        remaining_depth_mm = TOTAL_DESCENT_MM - upward_step * Z_STEP_MM
        point_name = _depth_label(station, remaining_depth_mm)
        actions.append(
            {
                "type": "move_xyzr",
                "name": point_name,
                "x_mm": 0.0,
                "y_mm": 0.0,
                "z_mm": Z_STEP_MM,
                "r_deg": 0.0,
            }
        )
        _wait_record_and_capture(actions, point_name, (2,))


def _append_safe_grid_transfer(
    actions: list[dict],
    target: dict,
    previous_float_j3_mm: float,
) -> None:
    """Permit the next XY/R transfer only after return to the prior float Z."""
    actions.append(
        {
            "type": "move_joints",
            "name": f"安全转移到 {target['name']}",
            "joints": list(target["joints"]),
            "tolerance": 0.2,
            "require_current_j3_mm": float(previous_float_j3_mm),
            "j3_tolerance_mm": 0.2,
        }
    )


def build_action() -> dict:
    """Build Task 2's complete acquisition plan using 4.3 mm Z increments."""
    if not math.isclose(TOTAL_DESCENT_MM, 21.5, abs_tol=1e-12):
        raise AssertionError("Task 5 总下降量必须严格等于 21.5 mm")

    targets = build_grid_targets()
    p00 = targets[0]
    actions: list[dict] = [
        {
            "type": "assert_joints",
            "name": "确认起点 P00 float",
            "joints": list(p00["joints"]),
            "tolerance": 0.2,
        }
    ]

    # This is exactly Task 2's capture schedule: source 0 is used once; source
    # 1 photographs the float arrival at all 36 stations; source 2 photographs
    # every recorded point, including the starting P00 float point.
    _wait_record_and_capture(actions, "P00 float / 初始", (0, 1, 2))
    _append_vertical_cycle(actions, "P00")
    previous_float_j3_mm = float(p00["joints"][2])

    for target in targets[1:]:
        _append_safe_grid_transfer(actions, target, previous_float_j3_mm)
        _wait_record_and_capture(
            actions,
            f"{target['name']} float / 到达",
            (1, 2),
        )
        _append_vertical_cycle(actions, target["name"])
        previous_float_j3_mm = float(target["joints"][2])

    return {
        "name": "task5 6x6蛇形多源分层拍照（4.3mm步长）",
        "description": (
            "完全沿用task2的6x6蛇形路线、拍照和坐标记录；每格从float"
            "分五次各下降4.3mm至总下降21.5mm，再分五次各上升4.3mm"
            "精确回到float后才允许转移到下一格。"
        ),
        "camera_model": dict(_task2_module().build_action()["camera_model"]),
        "actions": actions,
    }


if __name__ == "__main__":
    # Side-effect-free preview: no camera or robot access.
    grid = build_grid_targets()
    preview = build_action()
    actions = preview["actions"]
    points = sum(step["type"] == "record_point" for step in actions)
    photos = sum(step["type"] == "capture" for step in actions)
    by_source = {
        source: sum(
            step["type"] == "capture" and step["source"] == source
            for step in actions
        )
        for source in (0, 1, 2)
    }
    print(f"task5: {len(actions)} actions")
    print(f"grid cells: {len(grid)}; recorded points: {points}; photos: {photos}")
    print(f"photos by source: {by_source}")
    print(f"Z cycle: 5 x -{Z_STEP_MM:.1f} mm, then 5 x +{Z_STEP_MM:.1f} mm")
    print(f"maximum descent: {TOTAL_DESCENT_MM:.1f} mm; net Z: 0.0 mm")
    print("route:", " -> ".join(target["name"] for target in grid))
