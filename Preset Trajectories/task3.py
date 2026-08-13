"""Task 3: trace the four taught corners in <=5 mm XY increments.

Route: P00 float -> P05 -> P55 -> P50 -> P00.  Importing this module never
accesses a robot or camera; the UI calls :func:`build_action` and executes the
validated action only after operator confirmation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence


ACTION_API_VERSION = 1

XY_STEP_MM = 5.0
TARGET_DWELL_SECONDS = 2.0

# SCARA planar geometry shared with ``src/scara/pipeline/kinematics.py``.
SCARA_LINK1_MM = 225.0
SCARA_LINK2_MM = 175.0

# Camera geometry retained from task1/task2.
CAMERA_OFFSET_MM = 20.0
CAMERA1_OFFSET_FROM_J4_MM = 33.55
CAMERA1_Z_OFFSET_FROM_J4_MM = 0.0


def _finite_values(values: Sequence[float], expected: int, label: str) -> list[float]:
    """Convert a fixed-length sequence to finite floats."""
    if len(values) != expected:
        raise ValueError(f"{label} 必须包含 {expected} 个值")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} 必须全部是有限数字")
    return result


def camera_position_from_pose(pose: Sequence[float]) -> dict[str, float]:
    """Calculate the J4-rotating camera XYZ from measured centre XYZ/Rz.

    Rz is the counter-clockwise angle from world -Y to the vector from the J4
    axis/mechanical centre toward this camera.  Resolving the 20 mm offset into
    world coordinates gives::

        theta = Rz_deg * pi / 180
        camera_x = centre_x + 20*sin(theta)
        camera_y = centre_y - 20*cos(theta)
        camera_z = centre_z

    The Z equality retains the existing zero vertical-offset assumption.
    """
    values = _finite_values(pose, 6, "pose x/y/z/Rx/Ry/Rz")
    centre_x_mm, centre_y_mm, centre_z_mm, _rx, _ry, rz_deg = values
    theta = math.radians(rz_deg)
    return {
        "x_mm": centre_x_mm + CAMERA_OFFSET_MM * math.sin(theta),
        "y_mm": centre_y_mm - CAMERA_OFFSET_MM * math.cos(theta),
        "z_mm": centre_z_mm,
        "angle_from_negative_y_deg": rz_deg,
        "offset_mm": CAMERA_OFFSET_MM,
    }


def camera1_position_from_state(
    joints: Sequence[float],
    pose: Sequence[float],
) -> dict[str, float]:
    """Calculate source-1 XYZ from J1/J2; J4 has no influence.

    Source 1 is collinear with the J2 and J4 axes and lies 33.55 mm beyond J4
    away from J2.  Its world-frame direction is therefore the forearm angle
    ``alpha = J1 + J2``.  With first/second link lengths 225/175 mm::

        camera1_x = 225*cos(J1) + (175+33.55)*cos(J1+J2)
        camera1_y = 225*sin(J1) + (175+33.55)*sin(J1+J2)
        camera1_z = measured_J4_axis_z + fixed_z_offset

    Angles are converted from degrees to radians before ``sin``/``cos``.  The
    fixed Z offset is currently zero because no measured vertical offset has
    been supplied.
    """
    joint_values = _finite_values(joints, 4, "joints J1/J2/J3/J4")
    pose_values = _finite_values(pose, 6, "pose x/y/z/Rx/Ry/Rz")
    j1_deg, j2_deg = joint_values[0], joint_values[1]
    j1_rad = math.radians(j1_deg)
    forearm_angle_deg = j1_deg + j2_deg
    forearm_rad = math.radians(forearm_angle_deg)
    effective_forearm_mm = SCARA_LINK2_MM + CAMERA1_OFFSET_FROM_J4_MM
    return {
        "x_mm": (
            SCARA_LINK1_MM * math.cos(j1_rad)
            + effective_forearm_mm * math.cos(forearm_rad)
        ),
        "y_mm": (
            SCARA_LINK1_MM * math.sin(j1_rad)
            + effective_forearm_mm * math.sin(forearm_rad)
        ),
        "z_mm": pose_values[2] + CAMERA1_Z_OFFSET_FROM_J4_MM,
        "forearm_angle_deg": forearm_angle_deg,
        "offset_from_j4_axis_mm": CAMERA1_OFFSET_FROM_J4_MM,
        "z_offset_from_j4_axis_mm": CAMERA1_Z_OFFSET_FROM_J4_MM,
    }


def _project_root() -> Path:
    """Return the repository root containing ``scara_presets.json``."""
    return Path(__file__).resolve().parents[1]


def _load_corner_presets() -> dict[str, list[float]]:
    """Load the four float-height joint targets used to define the XY loop."""
    preset_path = _project_root() / "scara_presets.json"
    raw = json.loads(preset_path.read_text(encoding="utf-8"))
    corners: dict[str, list[float]] = {}
    for name in ("P00", "P05", "P55", "P50"):
        key = f"{name} float"
        values = raw.get(key)
        if not isinstance(values, list):
            raise ValueError(f"{preset_path} 缺少预设点 {key!r}")
        corners[name] = _finite_values(values, 4, f"预设点 {key}")
    return corners


def _fk_wrist_xy(joints: Sequence[float]) -> tuple[float, float]:
    """Calculate the taught point's J4-axis world XY from its J1/J2.

    For the 225/175 mm planar SCARA::

        x = 225*cos(J1) + 175*cos(J1+J2)
        y = 225*sin(J1) + 175*sin(J1+J2)

    J3 and J4 do not change the J4-axis XY position.
    """
    values = _finite_values(joints, 4, "关节目标")
    j1_rad = math.radians(values[0])
    forearm_rad = math.radians(values[0] + values[1])
    return (
        SCARA_LINK1_MM * math.cos(j1_rad)
        + SCARA_LINK2_MM * math.cos(forearm_rad),
        SCARA_LINK1_MM * math.sin(j1_rad)
        + SCARA_LINK2_MM * math.sin(forearm_rad),
    )


def build_xy_waypoints() -> list[dict]:
    """Build the closed P00->P05->P55->P50->P00 route.

    For an edge from ``A`` to ``B``::

        distance = hypot(Bx-Ax, By-Ay)
        unit = ((Bx-Ax)/distance, (By-Ay)/distance)
        waypoint(k) = A + min(5*k, distance)*unit

    Thus every regular displacement has Euclidean length exactly 5 mm.  Only
    the final residual displacement of an edge can be shorter than 5 mm, so the
    route reaches each taught corner exactly rather than overshooting it.
    """
    corners = _load_corner_presets()
    corner_xy = {name: _fk_wrist_xy(joints) for name, joints in corners.items()}
    route = ("P00", "P05", "P55", "P50", "P00")
    waypoints: list[dict] = [
        {
            "name": "P00",
            "edge": "起点",
            "distance_on_edge_mm": 0.0,
            "x_mm": corner_xy["P00"][0],
            "y_mm": corner_xy["P00"][1],
            "is_corner": True,
        }
    ]

    for start_name, end_name in zip(route, route[1:]):
        start_x, start_y = corner_xy[start_name]
        end_x, end_y = corner_xy[end_name]
        edge_dx = end_x - start_x
        edge_dy = end_y - start_y
        edge_length_mm = math.hypot(edge_dx, edge_dy)
        if edge_length_mm <= 1e-9:
            raise ValueError(f"{start_name} 与 {end_name} 的XY位置重合")
        unit_x = edge_dx / edge_length_mm
        unit_y = edge_dy / edge_length_mm
        full_steps = int(math.floor(edge_length_mm / XY_STEP_MM))

        distances = [XY_STEP_MM * index for index in range(1, full_steps + 1)]
        if not distances or edge_length_mm - distances[-1] > 1e-9:
            distances.append(edge_length_mm)
        else:
            # An exactly divisible edge must use the exact taught endpoint,
            # avoiding a tiny floating-point mismatch at the corner.
            distances[-1] = edge_length_mm

        for distance_mm in distances:
            is_corner = abs(distance_mm - edge_length_mm) <= 1e-9
            x_mm = end_x if is_corner else start_x + distance_mm * unit_x
            y_mm = end_y if is_corner else start_y + distance_mm * unit_y
            waypoints.append(
                {
                    "name": end_name if is_corner else f"{start_name}->{end_name}",
                    "edge": f"{start_name}->{end_name}",
                    "distance_on_edge_mm": distance_mm,
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "is_corner": is_corner,
                }
            )
    return waypoints


def _append_settle_record_capture(
    actions: list[dict],
    name: str,
    sources: Sequence[int],
) -> None:
    """Wait two seconds, record live robot state, then save requested photos."""
    actions.append({"type": "wait", "seconds": TARGET_DWELL_SECONDS})
    actions.append({"type": "record_point", "name": name})
    for source in sources:
        actions.append({"type": "capture", "source": int(source)})


def build_action() -> dict:
    """Build Task 3 using only relative world-X/Y motion steps."""
    corners = _load_corner_presets()
    waypoints = build_xy_waypoints()
    p00 = corners["P00"]
    actions: list[dict] = [
        {
            "type": "assert_joints",
            "name": "确认起点 P00 float",
            "joints": list(p00),
            "tolerance": 0.2,
        }
    ]

    # Match task1: source 0 is the one-off overview; source 1 photographs each
    # distinct taught corner once; source 2 photographs every recorded point.
    _append_settle_record_capture(actions, "P00 float / 初始", (0, 1, 2))

    previous = waypoints[0]
    visited_source1_corners = {"P00"}
    for point_index, waypoint in enumerate(waypoints[1:], start=2):
        delta_x_mm = waypoint["x_mm"] - previous["x_mm"]
        delta_y_mm = waypoint["y_mm"] - previous["y_mm"]
        step_length_mm = math.hypot(delta_x_mm, delta_y_mm)
        if step_length_mm <= 1e-9 or step_length_mm > XY_STEP_MM + 1e-8:
            raise ValueError(
                f"第{point_index}点XY步长 {step_length_mm:.9f} mm 不在 (0, 5] mm"
            )

        # z_mm and r_deg are always exactly zero.  The controller therefore
        # receives no Z or explicit J4/R command anywhere in task3.
        actions.append(
            {
                "type": "move_xyzr",
                "name": f"XY步进到 {waypoint['name']}",
                "x_mm": delta_x_mm,
                "y_mm": delta_y_mm,
                "z_mm": 0.0,
                "r_deg": 0.0,
            }
        )

        is_final_p00 = waypoint["is_corner"] and waypoint["name"] == "P00"
        if waypoint["is_corner"]:
            record_name = (
                "P00 float / 最终"
                if is_final_p00
                else f"{waypoint['name']} float / 到达"
            )
        else:
            record_name = (
                f"{waypoint['edge']} / {waypoint['distance_on_edge_mm']:.3f}mm"
            )

        sources = [2]
        if (
            waypoint["is_corner"]
            and waypoint["name"] not in visited_source1_corners
        ):
            sources.insert(0, 1)
            visited_source1_corners.add(waypoint["name"])
        _append_settle_record_capture(actions, record_name, sources)
        previous = waypoint

    return {
        "name": "task3 四角5mm XY闭环拍照",
        "description": (
            "P00→P05→P55→P50→P00；沿每条边以5mm世界XY步长移动，"
            "每边末步允许小于5mm以准确到角点，全程无Z和显式R/J4运动。"
        ),
        "camera_model": {
            "offset_mm": CAMERA_OFFSET_MM,
            "angle_reference": "world_negative_y",
            "positive_rotation": "counter_clockwise_from_above",
        },
        "actions": actions,
    }


if __name__ == "__main__":
    # Side-effect-free preview: no robot or camera access.
    waypoints = build_xy_waypoints()
    preview = build_action()
    actions = preview["actions"]
    points = sum(step["type"] == "record_point" for step in actions)
    photos_by_source = {
        source: sum(
            step["type"] == "capture" and step["source"] == source
            for step in actions
        )
        for source in (0, 1, 2)
    }
    move_steps = [step for step in actions if step["type"] == "move_xyzr"]
    print(f"task3: {len(actions)} actions")
    print(f"XY move steps: {len(move_steps)}; recorded points: {points}")
    print(f"photos by source: {photos_by_source}")
    print(
        "max XY step (mm):",
        max(math.hypot(step["x_mm"], step["y_mm"]) for step in move_steps),
    )
