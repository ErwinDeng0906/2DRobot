"""Task13: dedicated fixed-height planar camera1-to-forearm calibration.

The task starts and ends at the explicit ``P22 float`` preset.  Sixteen
spatially dispersed observation targets are generated from the rigid Tray
geometry relative to that preset; Task12 data or recorded poses are not read.
J3 and absolute Rz remain fixed and camera1 captures ten images per target.
All fitting/validation/install equations live in
``scara.vision.planar_handeye`` and its Task13 runtime.

No Z descent, DO, vacuum, wafer action, or online visual correction is present.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ACTION_API_VERSION = 1
CAMERA_SOURCE = 1
POSE_COUNT = 16
FRAMES_PER_POSE = 10
SETTLE_SECONDS = 0.8
FRAME_INTERVAL_SECONDS = 0.10
MAX_CARTESIAN_TRANSIT_STEP_MM = 0.60
MAX_SEQUENTIAL_TRANSIENT_XY_MM = 1.50
MAX_SEQUENTIAL_TRANSIENT_RZ_DEG = 0.35
MOVE_TOLERANCE = 0.02
CALIBRATION_SLOTS = (
    "P22",
    "P02", "P04", "P11", "P14", "P15",
    "P20", "P24", "P30", "P35", "P40",
    "P44", "P50", "P52", "P53", "P55",
)


def _angle_delta(left: float, right: float) -> float:
    return (float(left) - float(right) + 180.0) % 360.0 - 180.0


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON顶层必须是对象：{path}")
    return payload


def _p22_float_preset(root: Path) -> list[float]:
    payload = _load(root / "scara_presets.json")
    value = payload.get("P22 float")
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("scara_presets.json缺少四轴预设点P22 float")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("P22 float包含非有限数值")
    return result


def _geometry_poses(root: Path, p22_joints: Sequence[float]) -> list[dict[str, Any]]:
    import numpy as np

    from scara.pipeline.kinematics import fk_wrist, rz_of

    geometry = _load(root / "src/scara/calib/tray_board_geometry.json")
    slots = geometry.get("slots") or {}
    missing = [name for name in CALIBRATION_SLOTS if name not in slots]
    if missing:
        raise ValueError("Tray geometry缺少Task13槽位：" + ", ".join(missing))
    rotation = np.asarray(
        (geometry.get("tray_frame") or {}).get("rotation_mechanical_from_tray"),
        dtype=float,
    )
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("Tray geometry缺少有效rotation_mechanical_from_tray")
    p22_T = np.asarray(slots["P22"], dtype=float)[:2]
    p22_world = np.asarray(fk_wrist(float(p22_joints[0]), float(p22_joints[1])))
    j3 = float(p22_joints[2])
    rz = float(rz_of(p22_joints[0], p22_joints[1], p22_joints[3]))
    result: list[dict[str, Any]] = []
    reference = list(p22_joints)
    for name in CALIBRATION_SLOTS:
        point_T = np.asarray(slots[name], dtype=float)[:2]
        world_xy = p22_world + rotation[:2, :2] @ (point_T - p22_T)
        joints = _solve_xy(world_xy, j3, rz, reference)
        if name == "P22":
            joints = [float(value) for value in p22_joints]
        reference = joints
        result.append(
            {
                "slot": name,
                "joints": joints,
                "world_xy_mm": world_xy.astype(float).tolist(),
                "alpha_deg": float(joints[0] + joints[1]),
                "source": "P22 float + rigid Tray geometry",
            }
        )
    return result


def _ordered_route(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_name = {str(item["slot"]): dict(item) for item in candidates}
    if len(by_name) != POSE_COUNT or "P22" not in by_name:
        raise ValueError(f"Task13几何观察姿态必须为{POSE_COUNT}个且包含P22")
    route = [by_name["P22"]]
    remaining = [item for name, item in by_name.items() if name != "P22"]
    while remaining:
        current = route[-1]
        next_item = min(remaining, key=lambda item: math.dist(current["world_xy_mm"], item["world_xy_mm"]))
        route.append(next_item)
        remaining.remove(next_item)
    return route


def _solve_xy(xy: Sequence[float], j3: float, rz: float, reference: Sequence[float]) -> list[float]:
    from scara.pipeline.kinematics import solve_joints

    result = solve_joints(float(xy[0]), float(xy[1]), float(j3), rz_deg=float(rz), ref_joints=list(reference))
    if result is None:
        raise ValueError(f"Task13观察点XY={list(xy)}不可达")
    return [float(value) for value in result]


def _sequential_audit(current: Sequence[float], target: Sequence[float], rz: float) -> tuple[float, float]:
    from scara.pipeline.kinematics import fk_wrist, rz_of

    state = list(current)
    states = [list(state)]
    for axis in range(4):
        state = list(state)
        state[axis] = float(target[axis])
        states.append(state)
    xy = [fk_wrist(row[0], row[1]) for row in states]
    max_xy = max(math.dist(left, right) for left, right in zip(xy, xy[1:]))
    max_rz = max(_angle_delta(rz_of(row[0], row[1], row[3]), rz) for row in states)
    return float(max_xy), float(max_rz)


def _moves_between(start: Mapping[str, Any], end: Mapping[str, Any], reference: Sequence[float], j3: float, rz: float) -> tuple[list[dict[str, Any]], list[float]]:
    start_xy = start["world_xy_mm"]
    end_xy = end["world_xy_mm"]
    distance = math.dist(start_xy, end_xy)
    count = max(1, int(math.ceil(distance / MAX_CARTESIAN_TRANSIT_STEP_MM)))
    actions: list[dict[str, Any]] = []
    current = list(reference)
    for index in range(1, count + 1):
        fraction = index / count
        xy = [
            float(start_xy[0]) + fraction * (float(end_xy[0]) - float(start_xy[0])),
            float(start_xy[1]) + fraction * (float(end_xy[1]) - float(start_xy[1])),
        ]
        target = _solve_xy(xy, j3, rz, current)
        max_xy, max_rz = _sequential_audit(current, target, rz)
        if max_xy > MAX_SEQUENTIAL_TRANSIENT_XY_MM + 1e-9:
            raise ValueError(f"Task13逐轴中转XY={max_xy:.3f}mm超过{MAX_SEQUENTIAL_TRANSIENT_XY_MM:.2f}mm")
        if max_rz > MAX_SEQUENTIAL_TRANSIENT_RZ_DEG + 1e-9:
            raise ValueError(f"Task13逐轴中转Rz={max_rz:.3f}°超过{MAX_SEQUENTIAL_TRANSIENT_RZ_DEG:.2f}°")
        actions.append(
            {
                "type": "move_joints",
                "name": f"Task13到{end['slot']}安全中转 {index:03d}/{count:03d}",
                "joints": target,
                "tolerance": MOVE_TOLERANCE,
                "require_current_j3_mm": j3,
                "j3_tolerance_mm": 0.15,
            }
        )
        current = target
    return actions, current


def build_action() -> dict[str, Any]:
    from scara.pipeline.kinematics import rz_of

    root = _root()
    p22_preset = _p22_float_preset(root)
    route = _ordered_route(_geometry_poses(root, p22_preset))
    p22 = route[0]
    start_joints = list(p22_preset)
    j3 = float(start_joints[2])
    rz = float(rz_of(start_joints[0], start_joints[1], start_joints[3]))
    actions: list[dict[str, Any]] = [
        {
            "type": "assert_joints",
            "name": "确认Task13起点为预设点P22 float",
            "joints": start_joints,
            "tolerance": 0.25,
        }
    ]
    current_pose = p22
    reference = start_joints
    for target in route:
        if target is not p22:
            moves, reference = _moves_between(current_pose, target, reference, j3, rz)
            actions.extend(moves)
        actions.append({"type": "wait", "seconds": SETTLE_SECONDS})
        for frame in range(1, FRAMES_PER_POSE + 1):
            actions.append({"type": "record_point", "name": f"TASK13|{target['slot']}|frame={frame:02d}/{FRAMES_PER_POSE:02d}"})
            actions.append({"type": "capture", "source": CAMERA_SOURCE})
            if frame < FRAMES_PER_POSE:
                actions.append({"type": "wait", "seconds": FRAME_INTERVAL_SECONDS})
        current_pose = target
    moves, reference = _moves_between(current_pose, p22, reference, j3, rz)
    actions.extend(moves)
    actions.append(
        {
            "type": "move_joints",
            "name": "Task13精确返回P22观察姿态",
            "joints": start_joints,
            "tolerance": MOVE_TOLERANCE,
            "require_current_j3_mm": j3,
            "j3_tolerance_mm": 0.15,
        }
    )
    return {
        "api_version": 1,
        "name": "Task13 — camera1固定高度平面手眼专用采集",
        "description": (
            f"从预设点P22 float开始；按刚体Tray geometry生成{len(route)}个分散观察姿态，每姿态10帧；"
            "同槽帧只作聚合，不冒充独立姿态。固定J3/Rz，无Z下降、DO、真空或视觉修正。"
        ),
        "camera_model": {
            "offset_mm": 0.0,
            "angle_reference": "world_negative_y",
            "positive_rotation": "counter_clockwise_from_above",
        },
        "actions": actions,
    }


def create_task_runtime(output_dir: Path, parent=None):
    from scara.vision.planar_handeye_runtime import create_task13_runtime

    return create_task13_runtime(_root(), output_dir, parent)


if __name__ == "__main__":
    task = build_action()
    kinds = [step["type"] for step in task["actions"]]
    print(task["name"])
    print(f"moves={kinds.count('move_joints')} photos={kinds.count('capture')} points={kinds.count('record_point')}")
