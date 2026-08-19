"""Task 2: 6 x 6 serpentine grid scan with three camera sources.

This module is import-safe: importing it only reads geometry when
``build_action()`` is called.  It never connects to a camera or robot.  The
SCARA UI validates the returned action and executes it only after the operator
confirms the run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence


ACTION_API_VERSION = 1

GRID_SIZE = 6
TARGET_DWELL_SECONDS = 2.0
Z_STEP_MM = 5.0
DESCENT_STEP_COUNT = 5
TOTAL_DESCENT_MM = Z_STEP_MM * DESCENT_STEP_COUNT

# Geometry shared with ``src/scara/pipeline/kinematics.py``.
SCARA_LINK1_MM = 225.0
SCARA_LINK2_MM = 175.0

# The Rz-rotating camera centre is 20 mm from the J4 axis in the XY plane.
CAMERA_OFFSET_MM = 20.0

# Source 1 is fixed to the forearm centreline, 33.55 mm beyond J4 away from
# J2.  No vertical optical-centre offset has been measured, so Z offset is
# explicitly assumed to be zero until this constant is replaced by a measured
# value.
CAMERA1_OFFSET_FROM_J4_MM = 33.55
CAMERA1_Z_OFFSET_FROM_J4_MM = 0.0


def _finite_values(values: Sequence[float], expected: int, label: str) -> list[float]:
    """Return ``expected`` finite floats or fail while the action is built."""
    if len(values) != expected:
        raise ValueError(f"{label} 必须包含 {expected} 个值")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} 必须全部是有限数字")
    return result


def _norm_deg(angle_deg: float) -> float:
    """Normalize an angle to ``(-180, 180]`` degrees."""
    angle = math.fmod(float(angle_deg), 360.0)
    if angle <= -180.0:
        angle += 360.0
    elif angle > 180.0:
        angle -= 360.0
    return angle


def _angle_error_deg(a_deg: float, b_deg: float) -> float:
    """Return the smallest absolute distance between two degree angles."""
    return abs(_norm_deg(float(a_deg) - float(b_deg)))


def camera_position_from_pose(pose: Sequence[float]) -> dict[str, float]:
    """Calculate the J4-rotating camera position from measured XYZ/Rz.

    Rz is the signed counter-clockwise angle from world -Y to the vector from
    the J4/mechanical centre to this camera.  Therefore::

        rz_rad = Rz_deg * pi / 180
        camera_x = centre_x + 20*sin(rz_rad)
        camera_y = centre_y - 20*cos(rz_rad)
        camera_z = centre_z

    The final equality declares the existing zero vertical-offset assumption.
    """
    values = _finite_values(pose, 6, "pose x/y/z/Rx/Ry/Rz")
    centre_x_mm, centre_y_mm, centre_z_mm, _rx, _ry, rz_deg = values
    rz_rad = math.radians(rz_deg)
    return {
        "x_mm": centre_x_mm + CAMERA_OFFSET_MM * math.sin(rz_rad),
        "y_mm": centre_y_mm - CAMERA_OFFSET_MM * math.cos(rz_rad),
        "z_mm": centre_z_mm,
        "angle_from_negative_y_deg": rz_deg,
        "offset_mm": CAMERA_OFFSET_MM,
    }


def camera1_position_from_state(
    joints: Sequence[float],
    pose: Sequence[float],
) -> dict[str, float]:
    """Calculate source-1 XYZ; J4 and Rz intentionally have no influence.

    Let ``alpha = J1 + J2`` be the forearm direction measured from world +X.
    The J2 and J4 axes are::

        E_x = 225*cos(J1)              E_y = 225*sin(J1)
        W_x = E_x + 175*cos(alpha)     W_y = E_y + 175*sin(alpha)

    Source 1 lies another 33.55 mm along the same J2-to-J4 direction, hence::

        camera1_x = 225*cos(J1) + (175 + 33.55)*cos(J1+J2)
        camera1_y = 225*sin(J1) + (175 + 33.55)*sin(J1+J2)
        camera1_z = measured_J4_axis_z + fixed_z_offset

    ``fixed_z_offset`` is currently zero because no vertical offset has been
    supplied.  Changing J4 while J1/J2 remain fixed cannot change this result.
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
    """Load and validate the four taught float-height corner joint sets."""
    preset_path = _project_root() / "scara_presets.json"
    raw = json.loads(preset_path.read_text(encoding="utf-8"))
    corners: dict[str, list[float]] = {}
    for name in ("P00", "P05", "P50", "P55"):
        key = f"{name} float"
        values = raw.get(key)
        if not isinstance(values, list):
            raise ValueError(f"{preset_path} 缺少预设点 {key!r}")
        corners[name] = _finite_values(values, 4, f"预设点 {key}")
    return corners


def _fk_wrist_xy(joints: Sequence[float]) -> tuple[float, float]:
    """Return J4-axis world XY from J1/J2 using the project SCARA model."""
    values = _finite_values(joints, 4, "关节目标")
    j1_rad = math.radians(values[0])
    forearm_rad = math.radians(values[0] + values[1])
    return (
        SCARA_LINK1_MM * math.cos(j1_rad)
        + SCARA_LINK2_MM * math.cos(forearm_rad),
        SCARA_LINK1_MM * math.sin(j1_rad)
        + SCARA_LINK2_MM * math.sin(forearm_rad),
    )


def _rz_from_joints(joints: Sequence[float]) -> float:
    """Return absolute end yaw: ``Rz = J1 + J2 + J4 - 90 deg``."""
    values = _finite_values(joints, 4, "关节目标")
    return _norm_deg(values[0] + values[1] + values[3] - 90.0)


def _bilinear(
    p00: float,
    p05: float,
    p50: float,
    p55: float,
    row_fraction: float,
    column_fraction: float,
) -> float:
    """Interpolate one scalar inside the quadrilateral defined by four corners.

    With ``u = row/5`` and ``v = column/5``::

        value(u,v) = (1-u)(1-v)*P00 + (1-u)v*P05
                   + u(1-v)*P50     + uv*P55

    Consequently every row and column contains six uniformly parameterized
    positions, while all four measured corner values remain exact.
    """
    u = float(row_fraction)
    v = float(column_fraction)
    return (
        (1.0 - u) * (1.0 - v) * float(p00)
        + (1.0 - u) * v * float(p05)
        + u * (1.0 - v) * float(p50)
        + u * v * float(p55)
    )


def _ik_wrist(
    x_mm: float,
    y_mm: float,
    j3_mm: float,
    rz_deg: float,
    reference_joints: Sequence[float],
) -> list[float]:
    """Solve the two SCARA elbow branches and select the continuous branch.

    The cosine rule gives::

        cos(J2) = (x^2+y^2-L1^2-L2^2) / (2*L1*L2)

    Each sign of ``acos`` is a possible elbow branch.  For each branch::

        J1 = atan2(y,x) - atan2(L2*sin(J2), L1+L2*cos(J2))
        J4 = Rz - J1 - J2 + 90 deg

    The candidate with the smallest wrapped J1/J2/J4 change from the previous
    serpentine target is selected, preventing an unnecessary elbow flip.
    """
    x = float(x_mm)
    y = float(y_mm)
    cos_j2 = (
        x * x
        + y * y
        - SCARA_LINK1_MM * SCARA_LINK1_MM
        - SCARA_LINK2_MM * SCARA_LINK2_MM
    ) / (2.0 * SCARA_LINK1_MM * SCARA_LINK2_MM)
    if cos_j2 < -1.0 - 1e-9 or cos_j2 > 1.0 + 1e-9:
        raise ValueError(f"网格点 ({x:.3f}, {y:.3f}) 超出SCARA臂展")
    cos_j2 = max(-1.0, min(1.0, cos_j2))
    base_rad = math.atan2(y, x)
    reference = _finite_values(reference_joints, 4, "参考关节")
    candidates: list[tuple[float, list[float]]] = []
    for sign in (1.0, -1.0):
        j2_rad = sign * math.acos(cos_j2)
        j1_rad = base_rad - math.atan2(
            SCARA_LINK2_MM * math.sin(j2_rad),
            SCARA_LINK1_MM + SCARA_LINK2_MM * math.cos(j2_rad),
        )
        j1_deg = _norm_deg(math.degrees(j1_rad))
        j2_deg = _norm_deg(math.degrees(j2_rad))
        j4_deg = _norm_deg(float(rz_deg) - j1_deg - j2_deg + 90.0)
        joints = [j1_deg, j2_deg, float(j3_mm), j4_deg]
        cost = (
            _angle_error_deg(j1_deg, reference[0])
            + _angle_error_deg(j2_deg, reference[1])
            + _angle_error_deg(j4_deg, reference[3])
        )
        candidates.append((cost, joints))
    return [round(value, 4) for value in min(candidates, key=lambda item: item[0])[1]]


def build_grid_targets() -> list[dict]:
    """Build all 36 float targets in row-wise serpentine order.

    Point naming follows the four taught corners: row increases P00→P50 and
    column increases P00→P05.  Even rows traverse columns 0→5; odd rows return
    5→0.  The resulting route starts at P00 and ends at P50.
    """
    corners = _load_corner_presets()
    corner_xy = {name: _fk_wrist_xy(joints) for name, joints in corners.items()}
    corner_rz = {name: _rz_from_joints(joints) for name, joints in corners.items()}
    corner_j3 = {name: joints[2] for name, joints in corners.items()}

    targets: list[dict] = []
    reference = list(corners["P00"])
    for row in range(GRID_SIZE):
        columns = range(GRID_SIZE) if row % 2 == 0 else range(GRID_SIZE - 1, -1, -1)
        for column in columns:
            name = f"P{row}{column}"
            u = row / (GRID_SIZE - 1)
            v = column / (GRID_SIZE - 1)
            x_mm = _bilinear(
                corner_xy["P00"][0],
                corner_xy["P05"][0],
                corner_xy["P50"][0],
                corner_xy["P55"][0],
                u,
                v,
            )
            y_mm = _bilinear(
                corner_xy["P00"][1],
                corner_xy["P05"][1],
                corner_xy["P50"][1],
                corner_xy["P55"][1],
                u,
                v,
            )
            j3_mm = _bilinear(
                corner_j3["P00"],
                corner_j3["P05"],
                corner_j3["P50"],
                corner_j3["P55"],
                u,
                v,
            )
            rz_deg = _bilinear(
                corner_rz["P00"],
                corner_rz["P05"],
                corner_rz["P50"],
                corner_rz["P55"],
                u,
                v,
            )

            # Preserve the four physically taught corner joint sets exactly;
            # use IK only for the 32 interpolated interior/edge targets.
            if name in corners:
                joints = list(corners[name])
            else:
                joints = _ik_wrist(x_mm, y_mm, j3_mm, rz_deg, reference)
            solved_x_mm, solved_y_mm = _fk_wrist_xy(joints)
            interpolation_error_mm = math.hypot(
                solved_x_mm - x_mm,
                solved_y_mm - y_mm,
            )
            if interpolation_error_mm > 0.01:
                raise ValueError(
                    f"{name} IK回代误差 {interpolation_error_mm:.4f} mm 超过0.01 mm"
                )
            targets.append(
                {
                    "name": name,
                    "row": row,
                    "column": column,
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "rz_deg": rz_deg,
                    "joints": joints,
                }
            )
            reference = joints
    return targets


def _wait_record_and_capture(
    actions: list[dict],
    point_name: str,
    sources: Iterable[int],
) -> None:
    """Wait two seconds, record live state, then capture requested sources."""
    actions.append({"type": "wait", "seconds": TARGET_DWELL_SECONDS})
    actions.append({"type": "record_point", "name": point_name})
    for source in sources:
        actions.append({"type": "capture", "source": int(source)})


def _append_vertical_cycle(actions: list[dict], station: str) -> None:
    """At one grid cell, descend 5x5 mm and return upward in five 5 mm steps.

    Every reached depth is recorded after a two-second dwell.  Source 2 takes
    one photo at every vertical point; source 0 and source 1 are not repeated
    during the vertical cycle.
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
        _wait_record_and_capture(actions, point_name, (2,))

    for upward_step in range(1, DESCENT_STEP_COUNT + 1):
        remaining_depth_mm = TOTAL_DESCENT_MM - upward_step * Z_STEP_MM
        point_name = (
            f"{station} / float"
            if remaining_depth_mm == 0.0
            else f"{station} / float-{remaining_depth_mm:.0f}mm"
        )
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
    """Move to the next grid XY/R only after verifying float-height return."""
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
    """Build the complete 36-cell serpentine acquisition action."""
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

    # Initial P00: source 0 is used once for the entire experiment, source 1
    # records this cell's float point, and source 2 begins its all-point series.
    _wait_record_and_capture(actions, "P00 float / 初始", (0, 1, 2))
    _append_vertical_cycle(actions, "P00")
    previous_float_j3_mm = float(p00["joints"][2])

    # Remaining 35 grid cells: source 1 captures each float arrival; source 2
    # captures that arrival plus all ten vertical-cycle targets.
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
        "name": "task2 6x6蛇形多源分层拍照",
        "description": (
            "由P00/P05/P50/P55双线性插值得到6x6世界XY网格，"
            "P00起始逐行蛇形扫描至P50；每格从float分五次下降25mm，"
            "再分五次回升至float。"
        ),
        "camera_model": {
            "offset_mm": CAMERA_OFFSET_MM,
            "angle_reference": "world_negative_y",
            "positive_rotation": "counter_clockwise_from_above",
        },
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
    print(f"task2: {len(actions)} actions")
    print(f"grid cells: {len(grid)}; recorded points: {points}; photos: {photos}")
    print(f"photos by source: {by_source}")
    print("route:", " -> ".join(target["name"] for target in grid))
