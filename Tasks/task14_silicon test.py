"""Task14: automatic-exposure silicon-wafer scan over all 36 tray slots.

The operator-provided wafers are expected at sixteen slots: eight normal and
eight outside-slot.  The action
visits every P00-P55 location through Task12's already-audited adjacent-cell
closed snake, captures five camera-1 frames per slot with automatic exposure,
and returns to the exact taught ``P00 float`` pose.  Detection statistics use
all 180 source frames for every slot; the five frames captured directly over
that slot are excluded from its statistics as high-risk tool occlusion.  Root-level ``1_XXX.jpg``
files are published with the TrayVision wafer/slot overlay; untouched captures
are retained under ``raw_task14/`` for report-time reprocessing.

This task never descends Z, operates DO/vacuum, or applies a visual correction.
Detection and JSON report generation live in
``scara.vision.task14_silicon_detection``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


ACTION_API_VERSION = 1
CAMERA_SOURCE = 1
CAMERA_EXPOSURE_MODE = "auto"
FRAMES_PER_SLOT = 5
SETTLE_SECONDS = 0.6
START_TOLERANCE_DEG_OR_MM = 0.25
MOVE_TOLERANCE_DEG_OR_MM = 0.02
J3_TOLERANCE_MM = 0.15

EXPECTED_NORMAL_WAFER_SLOTS = (
    "P01",
    "P03",
    "P04",
    "P12",
    "P15",
    "P20",
    "P22",
    "P23",
)
EXPECTED_OUTSIDE_WAFER_SLOTS = (
    "P31",
    "P33",
    "P35",
    "P42",
    "P44",
    "P50",
    "P52",
    "P54",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_route_helpers():
    """Load Task12's audited motion math without executing a robot action."""
    path = Path(__file__).with_name("task12_code visibility scan.py")
    spec = importlib.util.spec_from_file_location("_task14_task12_route_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载Task12安全路径工具：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ROUTE = _load_route_helpers()
TARGET_ORDER = tuple(_ROUTE.TARGET_ORDER)


def _record_name(target: str, frame_index: int) -> str:
    return f"TASK14|{target}|frame={frame_index:02d}/{FRAMES_PER_SLOT:02d}"


def _append_capture_burst(actions: list[dict[str, Any]], target: str) -> None:
    actions.append({"type": "wait", "seconds": SETTLE_SECONDS})
    for frame_index in range(1, FRAMES_PER_SLOT + 1):
        actions.append({"type": "record_point", "name": _record_name(target, frame_index)})
        actions.append({"type": "capture", "source": CAMERA_SOURCE})


def build_action() -> dict[str, Any]:
    from scara.pipeline.kinematics import fk_wrist, rz_of

    p00_joints, slot_points, rotation = _ROUTE._load_scan_inputs()
    p00_xy = fk_wrist(p00_joints[0], p00_joints[1])
    imaging_j3 = float(p00_joints[2])
    fixed_rz = float(rz_of(p00_joints[0], p00_joints[1], p00_joints[3]))
    actions: list[dict[str, Any]] = [
        {
            "type": "assert_joints",
            "name": "确认Task14固定高度起点 P00 float",
            "joints": list(p00_joints),
            "tolerance": START_TOLERANCE_DEG_OR_MM,
        }
    ]
    reference = list(p00_joints)

    def move_to(target_name: str, *, exact_p00: bool = False) -> None:
        nonlocal reference
        target_T = slot_points[target_name]
        mechanical_xy = _ROUTE._tray_to_mechanical_xy(
            p00_xy,
            rotation,
            target_T,
        )
        solved = _ROUTE._solve_xy(
            mechanical_xy,
            imaging_j3,
            fixed_rz,
            reference,
        )
        target_joints = list(p00_joints) if exact_p00 else solved
        actions.append(
            {
                "type": "move_joints",
                "name": f"Task14自然移动到观察点 {target_name}",
                "joints": target_joints,
                "tolerance": MOVE_TOLERANCE_DEG_OR_MM,
                "require_current_j3_mm": imaging_j3,
                "j3_tolerance_mm": J3_TOLERANCE_MM,
            }
        )
        reference = list(target_joints)

    for index, target in enumerate(TARGET_ORDER):
        if index:
            move_to(target)
        _append_capture_burst(actions, target)
    move_to("P00", exact_p00=True)

    return {
        "name": "task14_silicon detection — 36槽自动曝光硅片扫描",
        "description": (
            "从P00 float固定观察高度出发，沿闭合相邻槽蛇形路径扫描全部36槽；"
            "相机1使用自动曝光，每槽拍5帧，共180张，结束返回P00。"
            "相邻槽之间各发送一次直接关节目标，不拆分微小中转步；"
            "运动速度沿用主UI当前选定速度。"
            "报告对每个槽使用全部180张照片，并排除吸盘正对该槽的5张及其他无效观测。"
            "根目录1_XXX.jpg保存槽位/硅片标注图，原始图保留在raw_task14。"
            "预期正常硅片槽为P01/P03/P04/P12/P15/P20/P22/P23；"
            "预期槽外硅片槽为P31/P33/P35/P42/P44/P50/P52/P54。"
            "不下降Z、不触发DO/真空、不执行视觉修正。"
        ),
        "camera_model": {
            "offset_mm": 0.0,
            "angle_reference": "world_negative_y",
            "positive_rotation": "counter_clockwise_from_above",
        },
        "camera_capture_settings": {
            CAMERA_SOURCE: {"auto_exposure": True},
        },
        "actions": actions,
    }


def create_task_runtime(output_dir: Path, parent=None):
    from scara.vision.task14_silicon_detection import (
        create_task14_silicon_detection_runtime,
    )

    _p00, slot_points, _rotation = _ROUTE._load_scan_inputs()
    return create_task14_silicon_detection_runtime(
        output_dir=output_dir,
        project_root=_project_root(),
        target_order=TARGET_ORDER,
        slot_points_T_mm=slot_points,
        expected_normal_wafer_slots=EXPECTED_NORMAL_WAFER_SLOTS,
        frames_per_slot=FRAMES_PER_SLOT,
        exposure_mode=CAMERA_EXPOSURE_MODE,
        parent=parent,
        expected_outside_wafer_slots=EXPECTED_OUTSIDE_WAFER_SLOTS,
    )


if __name__ == "__main__":
    preview = build_action()
    kinds = [step["type"] for step in preview["actions"]]
    print(preview["name"])
    print(
        f"slots={len(TARGET_ORDER)}; frames/slot={FRAMES_PER_SLOT}; "
        f"photos={kinds.count('capture')}; exposure={CAMERA_EXPOSURE_MODE}"
    )
    print("J3=fixed; Rz=fixed; Z descent=none; DO/vacuum=none; final=P00 float")
