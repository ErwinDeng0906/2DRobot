"""Task 1: four-station vertical scan with three camera sources.

This module is import-safe: importing it never connects to a camera or robot.
The SCARA UI calls :func:`build_action` only after the operator imports it and
explicitly confirms execution.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence


ACTION_API_VERSION = 1

# The camera centre is 20 mm from the J4 rotation axis in the world XY plane.
CAMERA_OFFSET_MM = 20.0

# SCARA planar geometry used by ``src/scara/pipeline/kinematics.py``.  Camera
# source 1 is mounted on the forearm centreline and therefore rotates with
# J1+J2, not with J4.
SCARA_LINK1_MM = 225.0
SCARA_LINK2_MM = 175.0
CAMERA1_OFFSET_FROM_J4_MM = 33.55

# No camera-1-to-J4 vertical offset has been supplied.  Until one is measured,
# its optical centre is explicitly assumed to have the same Z as the J4 axis.
CAMERA1_Z_OFFSET_FROM_J4_MM = 0.0

# Every target is allowed two seconds to settle before state recording/photos.
TARGET_DWELL_SECONDS = 2.0

# One vertical increment is 5 mm; five downward increments therefore give
# TOTAL_DESCENT_MM = 5 mm/step * 5 steps = 25 mm.
Z_STEP_MM = 5.0
DESCENT_STEP_COUNT = 5
TOTAL_DESCENT_MM = Z_STEP_MM * DESCENT_STEP_COUNT


def camera_position_from_pose(
    pose: Sequence[float],
) -> dict[str, float]:
    """Calculate camera XYZ from one measured x/y/z/Rx/Ry/Rz centre pose.

    Coordinate conventions supplied for this experiment:

    * world +Y points from P05 toward P00;
    * world +Z raises the arm away from the platform;
    * positive J4/Rz is counter-clockwise when viewed from above.
    * Rz is the signed counter-clockwise angle from world -Y to the vector that
      starts at the mechanical centre (suction cup/wafer centre) and ends at
      the camera.

    Equation 1 — degree-to-radian conversion::

        rz_rad = rz_deg * pi / 180

    Python ``sin`` and ``cos`` require radians, so the Rz value returned by the
    controller must be converted before resolving the 20 mm offset.

    Equation 2 — camera offset along world X::

        camera_offset_x_mm = 20 mm * sin(rz_rad)

    At Rz=0° the camera vector points exactly along -Y, so its X component is
    zero.  Positive counter-clockwise rotation moves that vector toward +X;
    therefore the X projection is positive ``sin(Rz)``.

    Equation 3 — camera offset along world Y::

        camera_offset_y_mm = -20 mm * cos(rz_rad)

    At Rz=0° the full 20 mm offset points along world -Y, hence the leading
    minus sign.  At ±90° the Y projection becomes zero.

    Equation 4 — camera world X coordinate::

        camera_x_mm = centre_x_mm + camera_offset_x_mm

    The camera's X coordinate is the measured mechanical-centre X plus the
    rotated camera offset's X projection from Equation 2.

    Equation 5 — camera world Y coordinate::

        camera_y_mm = centre_y_mm + camera_offset_y_mm

    The camera's Y coordinate is the measured mechanical-centre Y plus the
    rotated camera offset's Y projection from Equation 3.

    Equation 6 — camera world Z coordinate::

        camera_z_mm = centre_z_mm

    Only the known 20 mm in-plane radial distance was supplied.  Therefore the
    model assumes zero camera-to-axis Z offset.  Add a measured Z offset later
    if the camera optical centre is vertically displaced from the TCP/axis.
    """
    if len(pose) != 6:
        raise ValueError("pose 必须包含 x/y/z/Rx/Ry/Rz 六个值")
    centre_x_mm, centre_y_mm, centre_z_mm, _rx_deg, _ry_deg, rz_deg = [
        float(value) for value in pose
    ]
    if not all(
        math.isfinite(value)
        for value in (centre_x_mm, centre_y_mm, centre_z_mm, rz_deg)
    ):
        raise ValueError("pose 的 x/y/z/Rz 必须是有限数字")

    rz_rad = math.radians(rz_deg)
    camera_offset_x_mm = CAMERA_OFFSET_MM * math.sin(rz_rad)
    camera_offset_y_mm = -CAMERA_OFFSET_MM * math.cos(rz_rad)
    camera_x_mm = centre_x_mm + camera_offset_x_mm
    camera_y_mm = centre_y_mm + camera_offset_y_mm
    camera_z_mm = centre_z_mm
    return {
        "x_mm": camera_x_mm,
        "y_mm": camera_y_mm,
        "z_mm": camera_z_mm,
        "angle_from_negative_y_deg": rz_deg,
    }


def camera1_position_from_state(
    joints: Sequence[float],
    pose: Sequence[float],
) -> dict[str, float]:
    """Calculate source-1 camera XYZ from J1/J2 and the measured J4-axis Z.

    Camera source 1 is fixed to the forearm, does not rotate with J4, and its
    optical centre lies on the line from the J2 axis through the J4 axis.  It
    is 33.55 mm beyond J4, on the side farther from J2.  The project SCARA
    model has a 225 mm first link and a 175 mm second link.

    Equation 1 — degree-to-radian conversion::

        j1_rad = J1_deg * pi / 180
        forearm_rad = (J1_deg + J2_deg) * pi / 180

    Trigonometric functions use radians.  ``forearm_rad`` is the world-frame
    direction from the J2 axis toward the J4 axis; J3 and J4 cannot change this
    planar direction.

    Equation 2 — J2-axis world position::

        j2_axis_x_mm = 225 mm * cos(j1_rad)
        j2_axis_y_mm = 225 mm * sin(j1_rad)

    The base is the world origin and the first link is 225 mm long, matching
    ``scara.pipeline.kinematics.fk_wrist``.

    Equation 3 — J4-axis world position::

        j4_axis_x_mm = j2_axis_x_mm + 175 mm * cos(forearm_rad)
        j4_axis_y_mm = j2_axis_y_mm + 175 mm * sin(forearm_rad)

    The J4 axis is at the end of the 175 mm forearm.  J4 changes only the tool
    yaw, so J4 does not appear in these equations.

    Equation 4 — camera-1 world XY::

        camera1_x_mm = j4_axis_x_mm + 33.55 mm * cos(forearm_rad)
        camera1_y_mm = j4_axis_y_mm + 33.55 mm * sin(forearm_rad)

    The plus sign follows from the measured mounting direction: J2, J4 and the
    camera are collinear, with the camera beyond J4 away from J2.  Equivalently::

        camera1_x_mm = 225*cos(J1) + 208.55*cos(J1+J2)
        camera1_y_mm = 225*sin(J1) + 208.55*sin(J1+J2)

    Equation 5 — camera-1 world Z::

        camera1_z_mm = measured_j4_axis_z_mm + camera1_z_offset_mm

    The controller's measured pose supplies the current J4-axis Z.  Because no
    vertical optical-centre offset has yet been measured, the declared offset
    is currently 0 mm and is included in the returned JSON for traceability.
    """
    if len(joints) != 4:
        raise ValueError("joints 必须包含 J1/J2/J3/J4 四个值")
    if len(pose) != 6:
        raise ValueError("pose 必须包含 x/y/z/Rx/Ry/Rz 六个值")

    joint_values = [float(value) for value in joints]
    pose_values = [float(value) for value in pose]
    if not all(math.isfinite(value) for value in joint_values + pose_values):
        raise ValueError("joints 和 pose 必须全部是有限数字")

    j1_deg, j2_deg = joint_values[0], joint_values[1]
    j1_rad = math.radians(j1_deg)
    forearm_angle_deg = j1_deg + j2_deg
    forearm_rad = math.radians(forearm_angle_deg)

    j2_axis_x_mm = SCARA_LINK1_MM * math.cos(j1_rad)
    j2_axis_y_mm = SCARA_LINK1_MM * math.sin(j1_rad)
    j4_axis_x_mm = j2_axis_x_mm + SCARA_LINK2_MM * math.cos(forearm_rad)
    j4_axis_y_mm = j2_axis_y_mm + SCARA_LINK2_MM * math.sin(forearm_rad)
    camera1_x_mm = (
        j4_axis_x_mm + CAMERA1_OFFSET_FROM_J4_MM * math.cos(forearm_rad)
    )
    camera1_y_mm = (
        j4_axis_y_mm + CAMERA1_OFFSET_FROM_J4_MM * math.sin(forearm_rad)
    )
    camera1_z_mm = pose_values[2] + CAMERA1_Z_OFFSET_FROM_J4_MM

    return {
        "x_mm": camera1_x_mm,
        "y_mm": camera1_y_mm,
        "z_mm": camera1_z_mm,
        "forearm_angle_deg": forearm_angle_deg,
        "offset_from_j4_axis_mm": CAMERA1_OFFSET_FROM_J4_MM,
        "z_offset_from_j4_axis_mm": CAMERA1_Z_OFFSET_FROM_J4_MM,
    }


def _project_root() -> Path:
    """Return the repository root containing ``scara_presets.json``."""
    return Path(__file__).resolve().parents[1]


def _load_required_presets() -> dict[str, list[float]]:
    """Load the four float-height joint targets at action-build time."""
    preset_path = _project_root() / "scara_presets.json"
    raw = json.loads(preset_path.read_text(encoding="utf-8"))
    required = ("P00 float", "P50 float", "P55 float", "P05 float")
    presets: dict[str, list[float]] = {}
    for name in required:
        values = raw.get(name)
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError(f"{preset_path} 缺少四关节预设点 {name!r}")
        joints = [float(value) for value in values]
        if not all(math.isfinite(value) for value in joints):
            raise ValueError(f"预设点 {name!r} 含非有限关节值")
        presets[name] = joints
    return presets


def _wait_record_and_capture(
    actions: list[dict],
    point_name: str,
    sources: Iterable[int],
) -> None:
    """Append the common arrival sequence for one physical target point.

    Time equation::

        capture_time >= arrival_time + TARGET_DWELL_SECONDS

    The explicit wait is placed immediately after motion (or the initial point
    assertion), then live joints/pose are read into JSON, and only then are the
    requested sources captured.  Thus every recorded target has settled for at
    least two seconds before its state and photos are sampled.
    """
    actions.append({"type": "wait", "seconds": TARGET_DWELL_SECONDS})
    actions.append({"type": "record_point", "name": point_name})
    for source in sources:
        actions.append({"type": "capture", "source": int(source)})


def _append_vertical_cycle(
    actions: list[dict],
    station: str,
    down_sources: Iterable[int],
    up_sources: Iterable[int],
) -> None:
    """Append five 5 mm descents followed by five 5 mm ascents.

    Depth after downward step ``i`` is::

        depth_i_mm = i * Z_STEP_MM,  i = 1..5

    so step 5 reaches 25 mm below float height.  The reverse loop applies five
    ``+5 mm`` increments.  A single +5 mm could not undo a -25 mm descent; five
    upward targets are therefore required to return exactly to float height.
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
                # World +Z raises the arm, hence descending 5 mm is ΔZ = -5 mm.
                "z_mm": -Z_STEP_MM,
                "r_deg": 0.0,
            }
        )
        _wait_record_and_capture(actions, point_name, down_sources)

    for upward_step in range(1, DESCENT_STEP_COUNT + 1):
        remaining_depth_mm = TOTAL_DESCENT_MM - upward_step * Z_STEP_MM
        point_name = (
            f"{station} / float"
            if remaining_depth_mm == 0
            else f"{station} / float-{remaining_depth_mm:.0f}mm"
        )
        actions.append(
            {
                "type": "move_xyzr",
                "name": point_name,
                "x_mm": 0.0,
                "y_mm": 0.0,
                # World +Z raises the arm, hence each return step is ΔZ = +5 mm.
                "z_mm": Z_STEP_MM,
                "r_deg": 0.0,
            }
        )
        _wait_record_and_capture(actions, point_name, up_sources)


def _append_safe_station_transfer(
    actions: list[dict],
    station: str,
    target_joints: list[float],
    p00_float_j3_mm: float,
) -> None:
    """Append a float-height-only XY/R station transfer.

    The preceding vertical cycle already applies five upward 5 mm moves.  The
    executor additionally verifies::

        abs(current_J3 - P00_float_J3) <= 0.2 mm

    before it permits this joint transfer.  Since all four stored float targets
    have essentially the same J3, J1/J2/J4 (the XY/R placement) are never sent
    while the arm remains below float height.  A failed height check aborts the
    action instead of attempting lateral motion.
    """
    actions.append(
        {
            "type": "move_joints",
            "name": f"安全转移到 {station}",
            "joints": list(target_joints),
            "tolerance": 0.2,
            "require_current_j3_mm": float(p00_float_j3_mm),
            "j3_tolerance_mm": 0.2,
        }
    )


def build_action() -> dict:
    """Build Task 1 from the current four float-height presets."""
    presets = _load_required_presets()
    p00 = presets["P00 float"]
    actions: list[dict] = []

    # The experiment must start physically at P00 float.  This read-only check
    # happens before every movement; it aborts instead of silently moving first,
    # so the required source 0/1/2 initial photos truly precede all robot motion.
    actions.append(
        {
            "type": "assert_joints",
            "name": "确认起点 P00 float",
            "joints": list(p00),
            "tolerance": 0.2,
        }
    )
    # Source 0 is the one-off experiment overview: it is used exactly once at
    # the initial P00 point.  Source 1 records the four float station arrivals.
    # Source 2 is the measurement camera and must record every one of the 45
    # route points, starting here before any robot movement.
    _wait_record_and_capture(actions, "P00 float / 初始", (0, 1, 2))
    _append_vertical_cycle(actions, "P00", down_sources=(2,), up_sources=(2,))

    # At P50/P55/P05, source 1 records only the float-height arrival while
    # source 2 records that arrival and all ten vertical-cycle points.
    for station in ("P50", "P55", "P05"):
        target = presets[f"{station} float"]
        _append_safe_station_transfer(actions, station, target, p00[2])
        _wait_record_and_capture(actions, f"{station} float / 到达", (1, 2))
        _append_vertical_cycle(actions, station, down_sources=(2,), up_sources=(2,))

    # Final transfer also enforces float height.  The 45th and final route point
    # is recorded by source 2 after its two-second dwell; source 0 is not reused.
    _append_safe_station_transfer(actions, "P00", p00, p00[2])
    _wait_record_and_capture(actions, "P00 float / 最终", (2,))

    return {
        "name": "task1 四工位多源分层拍照",
        "description": (
            "P00→P50→P55→P05→P00；每站从 float 分五次下降 25 mm，再分五次回升。"
        ),
        "camera_model": {
            "offset_mm": CAMERA_OFFSET_MM,
            "angle_reference": "world_negative_y",
            "positive_rotation": "counter_clockwise_from_above",
        },
        "actions": actions,
    }


if __name__ == "__main__":
    # Standalone mode is a side-effect-free preview; it never controls hardware.
    preview = build_action()
    capture_count = sum(step["type"] == "capture" for step in preview["actions"])
    point_count = sum(step["type"] == "record_point" for step in preview["actions"])
    print(f"task1: {len(preview['actions'])} actions")
    print(f"recorded points: {point_count}; photos: {capture_count}")
