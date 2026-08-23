"""Task15: lift and replace wafer 1 with a pump/vent sequence.

The route is deliberately relative to the robot pose at task start.  World-Z
positive raises the SCARA away from the tray, so descent commands are negative.
The four vertical moves sum to zero and do not change X, Y, or R.

DO contract:
* DO1 = vacuum pump;
* DO2 = release solenoid valve;
* level 1 is ON and level 0 is OFF.

The action runner records every DO write and clears touched outputs if the task
is stopped or fails.  This file never opens a camera and never moves XY/R.
"""

from __future__ import annotations


ACTION_API_VERSION = 1

PUMP_DO_CHANNEL = 1
VALVE_DO_CHANNEL = 2
FIRST_DESCENT_MM = 23.3
LIFT_MM = 23.3
PLACEMENT_DESCENT_MM = 23.3
FINAL_RETRACT_MM = 23.3


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


def build_action() -> dict:
    """Return the exact requested lift/release sequence."""
    actions = [
        _z_move("下降23.3 mm至吸取高度", -FIRST_DESCENT_MM),
        {"type": "record_point", "name": "wafer1 / 吸取高度"},
        {"type": "wait", "seconds": 1.0},
        _set_do("打开真空泵 DO1", PUMP_DO_CHANNEL, 1),
        _z_move("吸住wafer1后上升23.3 mm", LIFT_MM),
        {"type": "record_point", "name": "wafer1 / 抬起高度"},
        {"type": "wait", "seconds": 2.0},
        _z_move("放回wafer1并下降23.3 mm", -PLACEMENT_DESCENT_MM),
        {"type": "record_point", "name": "wafer1 / 放置高度"},
        _set_do("关闭真空泵 DO1", PUMP_DO_CHANNEL, 0),
        _set_do("打开释放电磁阀 DO2", VALVE_DO_CHANNEL, 1),
        {"type": "wait", "seconds": 5.0},
        _z_move("释放完成后上升23.3 mm", FINAL_RETRACT_MM),
        {"type": "record_point", "name": "wafer1 / 最终安全高度"},
        _set_do("关闭释放电磁阀 DO2", VALVE_DO_CHANNEL, 0),
    ]

    net_z_mm = sum(
        float(step.get("z_mm", 0.0))
        for step in actions
        if step["type"] == "move_xyzr"
    )
    if abs(net_z_mm) > 1e-9:
        raise AssertionError(f"Task15累计Z位移必须为0，当前为{net_z_mm:.6f} mm")

    return {
        "name": "task15_lift wafer1 — 真空吸取、抬起、放回和吹气释放",
        "description": (
            "从当前XY/R姿态相对下降23.3 mm，等待1秒后开启DO1真空泵；"
            "上升23.3 mm并等待2秒，再下降23.3 mm；关闭DO1并开启DO2"
            "释放阀，等待5秒后上升23.3 mm并关闭DO2。累计Z位移为0；"
            "不移动XY/R、不拍照。开始前须确认当前高度正好是wafer1的安全起点，"
            "DO1/DO2接线和0/1逻辑正确，行程无碰撞且物理急停可用。"
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
