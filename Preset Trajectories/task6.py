"""Task 6: acquire camera-1 images for ChArUco intrinsic calibration.

This action file intentionally owns only robot-side acquisition:

* read taught preset ``X``;
* generate a 105 x 105 mm, 7 x 7 serpentine XY scan;
* reject a route outside the theoretical two-link SCARA reach annulus;
* at each target wait, record live robot state, and capture camera source 1.

Reusable image-quality, ChArUco calibration, outlier rejection, pose-diversity,
GUI guidance, and JSON persistence live under ``src/scara/vision``.  The thin
``create_task_runtime`` hook at the bottom only connects that reusable runtime
to the existing SCARA action UI.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


ACTION_API_VERSION = 1

# Route and robot geometry.  X is the centre and unchanged-height reference.
START_PRESET_NAME = "X"
CAMERA_SOURCE = 1
GRID_ROWS = 7
GRID_COLUMNS = 7
SQUARE_SIZE_MM = 105.0
HALF_SPAN_MM = SQUARE_SIZE_MM / 2.0
TARGET_DWELL_SECONDS = 2.0
START_TOLERANCE = 0.2
SCARA_LINK1_MM = 225.0
SCARA_LINK2_MM = 175.0


def _finite_values(
    values: Sequence[float], expected: int, label: str
) -> list[float]:
    """Validate a fixed-length sequence and reject NaN/Inf robot data."""
    if not isinstance(values, (list, tuple)) or len(values) != expected:
        raise ValueError(f"{label} 必须包含 {expected} 个数值")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} 包含非有限数值")
    return result


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_start_preset() -> list[float]:
    """Read the four taught joints of preset X from scara_presets.json."""
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
            "请先把机械臂移动到105×105mm采集范围的中心，并保存为 X。"
        )
    return _finite_values(joints, 4, f"预设点 {START_PRESET_NAME}")


def _fk_wrist_xy(joints: Sequence[float]) -> tuple[float, float]:
    """Return J4-axis XY using planar two-link forward kinematics.

    With q1=J1, q2=J2 and link lengths L1/L2:

        x = L1*cos(q1) + L2*cos(q1 + q2)
        y = L1*sin(q1) + L2*sin(q1 + q2)
    """
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


def _validate_planar_reach(
    start_joints: Sequence[float], targets: list[dict]
) -> None:
    """Verify sampled route points satisfy |L1-L2| <= r <= L1+L2.

    Twenty-one samples along every straight transfer also catch a segment that
    crosses the inner unreachable circle despite having reachable endpoints.
    This is a theoretical geometry check, not collision checking.
    """
    centre_x, centre_y = _fk_wrist_xy(start_joints)
    min_radius = abs(SCARA_LINK1_MM - SCARA_LINK2_MM)
    max_radius = SCARA_LINK1_MM + SCARA_LINK2_MM
    previous = (0.0, 0.0)
    for target in targets:
        current = (
            float(target["offset_x_mm"]),
            float(target["offset_y_mm"]),
        )
        for sample in range(21):
            fraction = sample / 20.0
            # p(s) = p_previous + s*(p_current - p_previous), 0 <= s <= 1.
            offset_x = previous[0] + fraction * (current[0] - previous[0])
            offset_y = previous[1] + fraction * (current[1] - previous[1])
            radius = math.hypot(centre_x + offset_x, centre_y + offset_y)
            if radius < min_radius - 1e-6 or radius > max_radius + 1e-6:
                raise ValueError(
                    "以 X 为中心的105×105mm路径超出SCARA理论平面臂展："
                    f"点({offset_x:+.1f}, {offset_y:+.1f})mm的半径为"
                    f"{radius:.1f}mm，允许范围为{min_radius:.1f}–"
                    f"{max_radius:.1f}mm。请重新示教X。"
                )
        previous = current


def build_grid_targets() -> list[dict]:
    """Return 49 equally spaced offsets in row-wise serpentine order.

    With half-span H and C columns:

        x_c = -H + c*(2H)/(C-1),  c=0..C-1

    The same equation generates y.  Odd rows reverse direction to form the
    zigzag route without a long empty return at each row end.
    """
    x_values = np.linspace(-HALF_SPAN_MM, HALF_SPAN_MM, GRID_COLUMNS)
    y_values = np.linspace(-HALF_SPAN_MM, HALF_SPAN_MM, GRID_ROWS)
    targets: list[dict] = []
    sequence = 0
    for row, y_mm in enumerate(y_values):
        columns = (
            range(GRID_COLUMNS)
            if row % 2 == 0
            else range(GRID_COLUMNS - 1, -1, -1)
        )
        for column in columns:
            sequence += 1
            targets.append(
                {
                    "sequence": sequence,
                    "row": row,
                    "column": column,
                    "name": f"CAL_R{row + 1:02d}_C{column + 1:02d}",
                    "offset_x_mm": round(float(x_values[column]), 9),
                    "offset_y_mm": round(float(y_mm), 9),
                }
            )
    return targets


def _append_capture(actions: list[dict], name: str) -> None:
    """At one target: settle, record live robot state, then save camera-1 JPG."""
    actions.append({"type": "wait", "seconds": TARGET_DWELL_SECONDS})
    actions.append({"type": "record_point", "name": name})
    actions.append({"type": "capture", "source": CAMERA_SOURCE})


def build_action() -> dict:
    """Build the complete fixed-height acquisition route from current X."""
    start_joints = _load_start_preset()
    targets = build_grid_targets()
    _validate_planar_reach(start_joints, targets)
    actions: list[dict] = [
        {
            "type": "assert_joints",
            "name": "确认标定中心 X",
            "joints": list(start_joints),
            "tolerance": START_TOLERANCE,
        }
    ]

    previous_x = 0.0
    previous_y = 0.0
    for target in targets:
        # The runner accepts relative movement: Delta p = target - previous.
        delta_x = float(target["offset_x_mm"]) - previous_x
        delta_y = float(target["offset_y_mm"]) - previous_y
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
        _append_capture(
            actions,
            (
                f"{target['name']} / X{target['offset_x_mm']:+.3f}mm "
                f"Y{target['offset_y_mm']:+.3f}mm"
            ),
        )
        previous_x = float(target["offset_x_mm"])
        previous_y = float(target["offset_y_mm"])

    # Return to X without taking a duplicate calibration image.
    actions.append(
        {
            "type": "move_xyzr",
            "name": "返回标定中心 X",
            "x_mm": -previous_x,
            "y_mm": -previous_y,
            "z_mm": 0.0,
            "r_deg": 0.0,
        }
    )
    return {
        "name": "task6 相机1 ChArUco 内参标定采集",
        "description": (
            "读取预设X作为中心，在其±52.5mm范围内执行7×7蛇形扫描；"
            "共49个点，仅世界XY运动，Z和R始终为0；每点记录机械臂状态并"
            "由相机源1拍照。"
        ),
        # Action API requires this legacy field.  Zero offset makes its generic
        # camera_position equal to the recorded mechanical centre.
        "camera_model": {
            "offset_mm": 0.0,
            "angle_reference": "world_negative_y",
            "positive_rotation": "counter_clockwise_from_above",
        },
        "actions": actions,
    }


def create_task_runtime(output_dir: Path, parent=None):
    """Thin UI hook; all reusable calibration behavior lives in scara.vision."""
    from scara.vision.charuco_calibration_runtime import (
        create_camera1_charuco_runtime,
    )

    return create_camera1_charuco_runtime(
        output_dir,
        _project_root(),
        parent,
        planned_images=GRID_ROWS * GRID_COLUMNS,
        scan_size_mm=SQUARE_SIZE_MM,
    )


if __name__ == "__main__":
    # Safe preview only: validate X and print route facts; no camera or motion.
    task = build_action()
    targets = build_grid_targets()
    print(task["name"])
    print(f"grid points: {len(targets)}; actions: {len(task['actions'])}")
    print(
        f"square: {SQUARE_SIZE_MM:.1f} x {SQUARE_SIZE_MM:.1f} mm "
        "around preset X"
    )
    print("camera source: 1; all z_mm=0 and r_deg=0")
