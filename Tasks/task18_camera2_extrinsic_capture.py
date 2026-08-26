"""Task 18: stationary camera-2/J4 extrinsic calibration capture.

This task deliberately issues no robot motion, digital-output, vacuum, or
runtime-motion action.  The operator manually places the robot at one safe
calibration pose before importing/running the task.  Five fresh robot-state
records are paired with camera 1 and camera 2 images.  Move to a different safe
XY/J3/J4 pose only after the task has finished, then run it again.  The offline
solver combines multiple timestamp folders.
"""

from __future__ import annotations

from typing import Sequence


ACTION_API_VERSION = 1

CAMERA1_SOURCE = 1
CAMERA2_SOURCE = 2
FRAME_COUNT = 5
SETTLE_SECONDS = 1.0
INTER_FRAME_SECONDS = 0.35


def camera_position_from_pose(pose: Sequence[float]) -> dict:
    """Record the J4/TCP origin without pretending it is camera-2 position."""

    if not isinstance(pose, (list, tuple)) or len(pose) != 6:
        raise ValueError("pose必须包含x/y/z/Rx/Ry/Rz六个值")
    return {
        "x_mm": float(pose[0]),
        "y_mm": float(pose[1]),
        "z_mm": float(pose[2]),
        "status": "j4_reference_only_camera2_extrinsic_not_yet_known",
    }


def build_action() -> dict:
    actions: list[dict] = [{"type": "wait", "seconds": SETTLE_SECONDS}]
    for frame_index in range(1, FRAME_COUNT + 1):
        if frame_index > 1:
            actions.append({"type": "wait", "seconds": INTER_FRAME_SECONDS})
        actions.append(
            {
                "type": "record_point",
                "name": f"TASK18|stationary_pose|frame={frame_index:02d}",
            }
        )
        # Camera 1 is retained as an independent world/board cross-check.  The
        # extrinsic solver consumes only source 2 unless explicitly extended.
        actions.append({"type": "capture", "source": CAMERA1_SOURCE})
        actions.append({"type": "capture", "source": CAMERA2_SOURCE})
    return {
        "name": "task18 相机2-J4外参静止采集（无机械臂运动）",
        "description": (
            "在运行前由现场人员把机械臂放到一个安全静止姿态并固定ChArUco板。"
            "本任务只记录5组J1-J4/TCP状态并依次拍摄相机1和相机2，不发送任何"
            "机械臂、J3、J4、DO或真空指令。任务结束后才能人工移动到下一个姿态"
            "并重新运行。多个时间戳文件夹由离线外参工具统一分析。"
        ),
        "camera_model": {
            "offset_mm": 0.0,
            "angle_reference": "unavailable_until_camera2_extrinsic_calibration",
            "positive_rotation": "controller_world_convention",
        },
        "camera_capture_settings": {
            CAMERA1_SOURCE: {"auto_exposure": True},
            CAMERA2_SOURCE: {"auto_exposure": True},
        },
        "actions": actions,
    }


if __name__ == "__main__":
    task = build_action()
    kinds = [step["type"] for step in task["actions"]]
    print(task["name"])
    print(f"points={kinds.count('record_point')}; photos={kinds.count('capture')}")
    print("motion=0; DO=0; vacuum=0")
