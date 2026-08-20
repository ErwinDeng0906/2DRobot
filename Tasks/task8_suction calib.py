"""Task 8: fixed-plane camera-1 suction-target calibration.

The operator first teaches nine safe ``PXX float`` presets after manually
centering the current soft suction cup over each slot marker and raising only
J3 to the imaging height.  This action then revisits those presets in a
serpentine route.  It never performs the manual contact descent and never
turns vacuum on.

At each location the task records live robot state and saves twenty camera-1
frames.  The task runtime calls the existing Stage-3 estimator/tracker for the
``^C T_T`` equation and quality gates; Task 8 deliberately does not duplicate
PnP or reprojection equations.  Batch Stage-4 processing enriches points.json
and writes camera1_suction_target.json in the timestamped run folder.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence


ACTION_API_VERSION = 1

CAMERA_SOURCE = 1
FRAMES_PER_TARGET = 20
TARGET_DWELL_SECONDS = 2.0
JOINT_TOLERANCE_DEG_OR_MM = 0.25
IMAGING_J3_NOMINAL_MM = -27.0119
IMAGING_J3_ALLOWED_ERROR_MM = 0.50
MAXIMUM_PRESET_J3_SPREAD_MM = 0.20
EXPECTED_RZ_DEG = 20.82
MAXIMUM_RZ_ERROR_DEG = 1.0
MAXIMUM_RZ_SPREAD_DEG = 0.50
WORKING_PLANE_Z_T_MM = -2.0

# Serpentine path over the requested 3x3 selection.
TARGET_ORDER = (
    "P00",
    "P02",
    "P05",
    "P25",
    "P22",
    "P20",
    "P50",
    "P52",
    "P55",
)

TARGET_POINTS_T_MM = {
    "P00": [0.0, 0.0, WORKING_PLANE_Z_T_MM],
    "P02": [0.0, -50.0, WORKING_PLANE_Z_T_MM],
    "P05": [0.0, -125.0, WORKING_PLANE_Z_T_MM],
    "P20": [-50.0, 0.0, WORKING_PLANE_Z_T_MM],
    "P22": [-50.0, -50.0, WORKING_PLANE_Z_T_MM],
    "P25": [-50.0, -125.0, WORKING_PLANE_Z_T_MM],
    "P50": [-125.0, 0.0, WORKING_PLANE_Z_T_MM],
    "P52": [-125.0, -50.0, WORKING_PLANE_Z_T_MM],
    "P55": [-125.0, -125.0, WORKING_PLANE_Z_T_MM],
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _finite_joints(values: object, label: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"{label} 必须包含J1/J2/J3/J4四个数值")
    joints = [float(value) for value in values]
    if not all(math.isfinite(value) for value in joints):
        raise ValueError(f"{label} 包含非有限数值")
    return joints


def _find_preset(raw: Mapping[str, object], target: str) -> tuple[str, list[float]]:
    aliases = (f"{target} float", f"{target}_float")
    for name in aliases:
        if name in raw:
            return name, _finite_joints(raw[name], f"预设点 {name}")
    raise ValueError(
        f"scara_presets.json 缺少 {aliases[0]!r}（也接受 {aliases[1]!r}）。"
        "请先按Task8手工流程对准该槽中心、只抬高J3到安全观察高度，"
        "再保存该四关节预设；程序不会用理论IK代替人工示教。"
    )


def load_target_presets() -> tuple[dict[str, list[float]], dict[str, str]]:
    """Load all nine manually taught float presets and fail closed if absent."""

    path = _project_root() / "scara_presets.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"找不到预设文件：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"预设文件不是有效JSON：{path}（{exc}）") from exc
    if not isinstance(raw, dict):
        raise ValueError("scara_presets.json顶层必须是对象")
    presets: dict[str, list[float]] = {}
    resolved_names: dict[str, str] = {}
    missing: list[str] = []
    for target in TARGET_ORDER:
        try:
            resolved, joints = _find_preset(raw, target)
        except ValueError:
            missing.append(f"{target} float")
            continue
        resolved_names[target] = resolved
        presets[target] = joints
    if missing:
        raise ValueError(
            "Task8缺少手工示教的安全观察高度预设："
            + ", ".join(missing)
            + "。当前程序按安全原则拒绝从四角插值或自动下降。"
        )

    j3_values = [joints[2] for joints in presets.values()]
    j3_median = sorted(j3_values)[len(j3_values) // 2]
    if abs(j3_median - IMAGING_J3_NOMINAL_MM) > IMAGING_J3_ALLOWED_ERROR_MM:
        raise ValueError(
            f"九点J3中位数为{j3_median:.4f}mm，不是已确认的观察高度"
            f"{IMAGING_J3_NOMINAL_MM:.4f}±{IMAGING_J3_ALLOWED_ERROR_MM:.2f}mm"
        )
    if max(j3_values) - min(j3_values) > MAXIMUM_PRESET_J3_SPREAD_MM:
        raise ValueError(
            "九个PXX float预设的J3不在同一观察高度："
            f"范围{min(j3_values):.4f}–{max(j3_values):.4f}mm"
        )

    rz_values = [
        joints[0] + joints[1] + joints[3] - 90.0
        for joints in presets.values()
    ]
    rz_mean = sum(rz_values) / len(rz_values)
    if abs(rz_mean - EXPECTED_RZ_DEG) > MAXIMUM_RZ_ERROR_DEG:
        raise ValueError(
            f"九点末端平均Rz={rz_mean:.3f}°，偏离已确认的{EXPECTED_RZ_DEG:.2f}°"
        )
    if max(rz_values) - min(rz_values) > MAXIMUM_RZ_SPREAD_DEG:
        raise ValueError(
            "九点末端Rz没有保持不变："
            f"范围{min(rz_values):.3f}–{max(rz_values):.3f}°"
        )
    return presets, resolved_names


def _append_burst(actions: list[dict], target: str) -> None:
    actions.append({"type": "wait", "seconds": TARGET_DWELL_SECONDS})
    for frame_index in range(1, FRAMES_PER_TARGET + 1):
        actions.append(
            {
                "type": "record_point",
                "name": (
                    f"TASK8|{target}|frame={frame_index:02d}/"
                    f"{FRAMES_PER_TARGET:02d}"
                ),
            }
        )
        actions.append({"type": "capture", "source": CAMERA_SOURCE})


def build_action() -> dict:
    """Build the nine-preset acquisition without any contact-height motion."""

    presets, resolved_names = load_target_presets()
    p00 = presets["P00"]
    imaging_j3 = float(sorted(joints[2] for joints in presets.values())[4])
    actions: list[dict] = [
        {
            "type": "assert_joints",
            "name": f"确认起点 {resolved_names['P00']}",
            "joints": list(p00),
            "tolerance": JOINT_TOLERANCE_DEG_OR_MM,
        }
    ]
    for index, target in enumerate(TARGET_ORDER):
        if index > 0:
            actions.append(
                {
                    "type": "move_joints",
                    "name": f"低速安全移动到 {resolved_names[target]}",
                    "joints": list(presets[target]),
                    "tolerance": JOINT_TOLERANCE_DEG_OR_MM,
                    "require_current_j3_mm": imaging_j3,
                    "j3_tolerance_mm": JOINT_TOLERANCE_DEG_OR_MM,
                }
            )
        _append_burst(actions, target)
    actions.append(
        {
            "type": "move_joints",
            "name": f"采集完成后返回 {resolved_names['P00']}",
            "joints": list(p00),
            "tolerance": JOINT_TOLERANCE_DEG_OR_MM,
            "require_current_j3_mm": imaging_j3,
            "j3_tolerance_mm": JOINT_TOLERANCE_DEG_OR_MM,
        }
    )
    return {
        "name": "task8 相机1固定平面吸盘target标定",
        "description": (
            "调用九个已手工示教的PXX float安全观察点，按蛇形路线移动；"
            "每点读取20次实时机械臂状态并连续保存20张相机1照片。"
            "不下降到接触高度、不改变已示教Rz、不打开真空。照片由现有"
            "阶段3模块求^C T_T并经过全部单帧/时序质量门，结束后计算"
            "固定工作平面z_T=-2mm上的吸盘轴target。"
        ),
        "camera_model": {
            "offset_mm": 0.0,
            "angle_reference": "world_negative_y",
            "positive_rotation": "counter_clockwise_from_above",
        },
        "actions": actions,
    }


def create_task_runtime(output_dir: Path, parent=None):
    """Create Stage-4 processing attached to the existing action callbacks."""

    from scara.vision.suction_target_calibration_runtime import (
        create_camera1_suction_runtime,
    )

    presets, _resolved_names = load_target_presets()
    ordered_points = {
        target: list(TARGET_POINTS_T_MM[target]) for target in TARGET_ORDER
    }
    return create_camera1_suction_runtime(
        output_dir,
        _project_root(),
        ordered_points,
        presets,
        parent,
    )


if __name__ == "__main__":
    preview = build_action()
    point_count = sum(step["type"] == "record_point" for step in preview["actions"])
    photo_count = sum(step["type"] == "capture" for step in preview["actions"])
    print(preview["name"])
    print(f"targets={len(TARGET_ORDER)}; points={point_count}; photos={photo_count}")
    print("route=" + " -> ".join(TARGET_ORDER))
    print("contact descent=manual only; automatic Task8 motion remains at PXX float")
