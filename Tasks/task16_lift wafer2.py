"""Task16: pick from the current pose, place at P00, and return.

The action runner captures the live J1-J4 values in the first step and keeps
that snapshot immutable.  The task file therefore never reads hardware while
being imported, while the final move can still return to the actual task-start
pose rather than to a preset approximation.

Sequence:
* current pose: Z -23.3 mm, wait 1 s, DO1 on, Z +23.3 mm, wait 2 s;
* move to the taught ``P00 float`` joint target;
* P00: Z -23.0 mm, DO1 off, DO2 on, wait 5 s, Z +23.0 mm, DO2 off;
* return to the captured task-start J1-J4 pose.

No camera, XY-relative, rotation-relative, or visual correction action is used.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence


ACTION_API_VERSION = 1

PUMP_DO_CHANNEL = 1
VALVE_DO_CHANNEL = 2
PICKUP_Z_MM = 23.3
P00_PLACE_Z_MM = 23.0
P00_PRESET_NAME = "P00 float"
P00_MOVE_TOLERANCE = 0.20
RETURN_TOLERANCE = 0.20


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _finite_joints(values: object, label: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"{label}必须包含J1/J2/J3/J4四个值")
    joints = [float(value) for value in values]
    if not all(math.isfinite(value) for value in joints):
        raise ValueError(f"{label}包含NaN或Inf")
    return joints


def _load_p00_float() -> list[float]:
    path = _project_root() / "scara_presets.json"
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or P00_PRESET_NAME not in payload:
        raise ValueError(f"{path}缺少预设点{P00_PRESET_NAME!r}")
    return _finite_joints(payload[P00_PRESET_NAME], P00_PRESET_NAME)


def _z_move(name: str, delta_z_mm: float) -> dict:
    return {
        "type": "move_xyzr",
        "name": name,
        "x_mm": 0.0,
        "y_mm": 0.0,
        "z_mm": float(delta_z_mm),
        "r_deg": 0.0,
    }


def _set_do(name: str, channel: int, level: int) -> dict:
    return {
        "type": "set_do",
        "name": name,
        "channel": int(channel),
        "level": int(level),
    }


def _assert_balanced_vertical_pair(values: Sequence[float], label: str) -> None:
    total = sum(float(value) for value in values)
    if abs(total) > 1e-9:
        raise AssertionError(f"{label}累计Z位移必须为0，当前为{total:.6f} mm")


def build_action() -> dict:
    """Build the requested current-pose -> P00 -> current-pose cycle."""
    _assert_balanced_vertical_pair((-PICKUP_Z_MM, PICKUP_Z_MM), "吸取段")
    _assert_balanced_vertical_pair((-P00_PLACE_Z_MM, P00_PLACE_Z_MM), "P00放置段")
    p00_joints = _load_p00_float()

    actions = [
        {
            "type": "remember_start_joints",
            "name": "记录并锁定Task16当前起始J1-J4",
        },
        _z_move("当前位置下降23.3 mm至吸取高度", -PICKUP_Z_MM),
        {"type": "record_point", "name": "Task16 / 当前点吸取高度"},
        {"type": "wait", "seconds": 1.0},
        _set_do("打开真空泵 DO1", PUMP_DO_CHANNEL, 1),
        _z_move("吸住wafer后上升23.3 mm", PICKUP_Z_MM),
        {"type": "record_point", "name": "Task16 / 当前点抬起高度"},
        {"type": "wait", "seconds": 2.0},
        {
            "type": "move_joints",
            "name": "携带wafer移动到P00 float",
            "joints": p00_joints,
            "tolerance": P00_MOVE_TOLERANCE,
        },
        {"type": "record_point", "name": "Task16 / 到达P00 float"},
        _z_move("P00下降23.0 mm至放置高度", -P00_PLACE_Z_MM),
        {"type": "record_point", "name": "Task16 / P00放置高度"},
        _set_do("关闭真空泵 DO1", PUMP_DO_CHANNEL, 0),
        _set_do("打开释放电磁阀 DO2", VALVE_DO_CHANNEL, 1),
        {"type": "wait", "seconds": 5.0},
        _z_move("P00释放完成后上升23.0 mm", P00_PLACE_Z_MM),
        {"type": "record_point", "name": "Task16 / P00安全高度"},
        _set_do("关闭释放电磁阀 DO2", VALVE_DO_CHANNEL, 0),
        {
            "type": "return_to_start_joints",
            "name": "返回Task16实际起始J1-J4",
            "tolerance": RETURN_TOLERANCE,
        },
        {"type": "record_point", "name": "Task16 / 已返回实际原位"},
    ]

    return {
        "name": "task16 — 当前点吸取wafer、P00放置并返回实际原位",
        "description": (
            "任务第一步读取并锁定当前J1-J4；当前位置下降23.3 mm，等待1秒后"
            "开启DO1，回升23.3 mm并等待2秒；携带wafer移动到P00 float，"
            "下降23.0 mm后关闭DO1并开启DO2，等待5秒；回升23.0 mm后"
            "关闭DO2，最后返回任务开始时实际记录的J1-J4。开始前须确认当前位置"
            "正好是安全吸取起点、P00 float及两处下降行程无碰撞、DO接线/逻辑"
            "正确且物理急停可用。"
        ),
        "camera_model": {
            "offset_mm": 0.0,
            "angle_reference": "world_negative_y",
            "positive_rotation": "counter_clockwise_from_above",
        },
        "actions": actions,
    }


if __name__ == "__main__":
    task = build_action()
    print(task["name"])
    for index, action in enumerate(task["actions"], start=1):
        print(f"{index:02d}. {action}")
