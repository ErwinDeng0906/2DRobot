"""Pure geometry utilities for wafer tray pick/place poses.

No hardware access, no network calls — only numpy + stdlib.
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


__all__ = [
    "Frame",
    "load_tray_geometry",
    "solve_frame",
    "pocket_pose",
    "compose_place_orientation",
    "save_frame",
    "load_frame",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Frame:
    """Tray frame expressed in the robot base coordinate system.

    Attributes:
        origin: 3-D base coordinates of slot (0, 0) center (mm).
        xhat: Unit vector along the tray +col direction in the base XY plane.
        yhat: Unit vector along the tray +row direction in the base XY plane.
        yaw_rad: Angle of xhat measured from the base +X axis (rad).
        z: Z height associated with the frame (mm).  Usually the pocket bottom.
        pitch_x: Grid pitch along the tray +col direction (mm).
        pitch_y: Grid pitch along the tray +row direction (mm).
    """
    origin: np.ndarray
    xhat: np.ndarray
    yhat: np.ndarray
    yaw_rad: float
    z: float
    pitch_x: float = 25.0
    pitch_y: float = 25.0

    def __post_init__(self):
        # Accept list/tuple input and force float arrays.
        object.__setattr__(self, "origin", np.asarray(self.origin, dtype=float).reshape(3))
        object.__setattr__(self, "xhat", np.asarray(self.xhat, dtype=float).reshape(2))
        object.__setattr__(self, "yhat", np.asarray(self.yhat, dtype=float).reshape(2))

    def to_dict(self) -> dict:
        return {
            "origin": self.origin.tolist(),
            "xhat": self.xhat.tolist(),
            "yhat": self.yhat.tolist(),
            "yaw_rad": float(self.yaw_rad),
            "z": float(self.z),
            "pitch_x": float(self.pitch_x),
            "pitch_y": float(self.pitch_y),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Frame":
        return cls(
            origin=d["origin"],
            xhat=d["xhat"],
            yhat=d["yhat"],
            yaw_rad=float(d["yaw_rad"]),
            z=float(d["z"]),
            pitch_x=float(d.get("pitch_x", 25.0)),
            pitch_y=float(d.get("pitch_y", 25.0)),
        )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _to_xy(p: Sequence[float]) -> np.ndarray:
    """Return the first two elements of *p* as a float array."""
    return np.asarray(p, dtype=float)[:2]


def _to_xyz(p: Sequence[float], default_z: float | None = None) -> np.ndarray:
    """Return *p* as a 3-D float array, using default_z if no z is supplied."""
    arr = np.asarray(p, dtype=float)
    if arr.size == 2:
        z = 0.0 if default_z is None else default_z
        return np.array([arr[0], arr[1], z], dtype=float)
    return arr.reshape(3)


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        raise ValueError("Cannot normalize a zero-length vector")
    return v / norm


# ---------------------------------------------------------------------------
# Tray geometry loading
# ---------------------------------------------------------------------------

def load_tray_geometry(path: str | Path) -> dict:
    """Load and normalize the tray geometry JSON.

    Returns a dict with:
      - rows, cols, pitch_x, pitch_y (floats)
      - slots: list of {"row": int, "col": int, "rel_xy": [x, y]} for all slots
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    grid = data["grid"]
    slots_in = data["slots"]

    slots_out = []
    for s in slots_in:
        rel = s["center_relative_to_origin_mm"]
        slots_out.append(
            {
                "row": int(s["row"]),
                "col": int(s["col"]),
                "rel_xy": [float(rel[0]), float(rel[1])],
            }
        )

    return {
        "rows": int(grid["n_rows"]),
        "cols": int(grid["n_cols"]),
        "pitch_x": float(grid["pitch_x_mm"]),
        "pitch_y": float(grid["pitch_y_mm"]),
        "origin_slot_center_mm": [float(v) for v in grid["origin_slot_center_mm"]],
        "slots": slots_out,
    }


# ---------------------------------------------------------------------------
# Frame solving from 3 probed corner pockets
# ---------------------------------------------------------------------------

def solve_frame(
    p00: Sequence[float],
    p_col: Sequence[float],
    p_row: Sequence[float],
    col_n: int,
    row_m: int,
    pitch_x: float = 25.0,
    pitch_y: float = 25.0,
    z: float | None = None,
) -> Frame:
    """Compute a tray Frame from three probed pocket centers.

    p00  : base coords of slot (0, 0)
    p_col: base coords of slot (0, col_n)
    p_row: base coords of slot (row_m, 0)

    Each input may be (x, y) or (x, y, z).  The z component defaults to the
    z of p00 or to the explicitly supplied *z*.
    """
    if col_n <= 0 or row_m <= 0:
        raise ValueError("col_n and row_m must be positive integers")

    p00 = _to_xyz(p00)
    p_col = _to_xyz(p_col, default_z=p00[2])
    p_row = _to_xyz(p_row, default_z=p00[2])

    dx = p_col[:2] - p00[:2]
    dy = p_row[:2] - p00[:2]

    len_x = float(np.linalg.norm(dx))
    len_y = float(np.linalg.norm(dy))
    expected_x = col_n * pitch_x
    expected_y = row_m * pitch_y

    rel_tol = 0.02  # 2 %
    abs_tol = 0.5   # mm

    if not math.isclose(len_x, expected_x, rel_tol=rel_tol, abs_tol=abs_tol):
        warnings.warn(
            f"X-side length {len_x:.3f} mm differs from expected {expected_x:.3f} mm "
            f"(col_n={col_n}, pitch_x={pitch_x})",
            UserWarning,
        )
    if not math.isclose(len_y, expected_y, rel_tol=rel_tol, abs_tol=abs_tol):
        warnings.warn(
            f"Y-side length {len_y:.3f} mm differs from expected {expected_y:.3f} mm "
            f"(row_m={row_m}, pitch_y={pitch_y})",
            UserWarning,
        )

    xhat = _unit(dx)
    yhat = _unit(dy)
    yaw_rad = math.atan2(float(xhat[1]), float(xhat[0]))

    final_z = p00[2] if z is None else float(z)

    return Frame(
        origin=p00,
        xhat=xhat,
        yhat=yhat,
        yaw_rad=yaw_rad,
        z=final_z,
        pitch_x=float(pitch_x),
        pitch_y=float(pitch_y),
    )


# ---------------------------------------------------------------------------
# Pocket pose from frame
# ---------------------------------------------------------------------------

def pocket_pose(frame: Frame, row: int, col: int) -> dict:
    """Return the base-frame pocket pose for slot (row, col).

    Result: {"x": float, "y": float, "z": float, "yaw_rad": float}
    """
    origin_xy = np.asarray(frame.origin[:2], dtype=float)
    xy = (
        origin_xy
        + float(col) * frame.pitch_x * frame.xhat
        + float(row) * frame.pitch_y * frame.yhat
    )
    return {
        "x": float(xy[0]),
        "y": float(xy[1]),
        "z": float(frame.z),
        "yaw_rad": float(frame.yaw_rad),
    }


# ---------------------------------------------------------------------------
# Orientation composition (matrix-only, no scalar rz addition)
# ---------------------------------------------------------------------------

def _zyx_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """ZYX Euler angles -> rotation matrix.  R = Rz(rz) Ry(ry) Rx(rx)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=float,
    )


def _matrix_to_zyx(R: np.ndarray) -> tuple[float, float, float]:
    """Rotation matrix -> ZYX Euler angles (rx, ry, rz)."""
    R = np.asarray(R, dtype=float)
    # ry from -sin(ry) = R[2,0]
    ry = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))

    cos_ry = math.cos(ry)
    if abs(cos_ry) < 1e-10:
        # Gimbal lock: Ry = +/- 90 deg
        rx = 0.0
        # When cos(ry)==0, R reduces to a function of (rz +/- rx).
        # Choose rz from R[1,1], R[0,1] as the remaining degree.
        rz = math.atan2(-R[0, 1], R[1, 1])
    else:
        rx = math.atan2(R[2, 1], R[2, 2])
        rz = math.atan2(R[1, 0], R[0, 0])

    return rx, ry, rz


def compose_place_orientation(
    R_down_euler_zyx: tuple[float, float, float],
    target_yaw_rad: float,
) -> tuple[float, float, float]:
    """Compose a tray-aligned place orientation.

    R_down_euler_zyx : (rx, ry, rz) of the gripper "pointing down" orientation.
    target_yaw_rad   : desired world-frame yaw for this pocket.

    The resulting rotation is R = Rz(target_yaw) @ R_down, converted back to
    ZYX Euler angles for cal_ikine.  No scalar addition of rz is performed.
    """
    rx, ry, rz = R_down_euler_zyx
    R_down = _zyx_to_matrix(rx, ry, rz)

    cz, sz = math.cos(target_yaw_rad), math.sin(target_yaw_rad)
    R_z = np.array(
        [
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    R = R_z @ R_down
    return _matrix_to_zyx(R)


# ---------------------------------------------------------------------------
# Frame persistence
# ---------------------------------------------------------------------------

def save_frame(frame: Frame, path: str | Path) -> None:
    """Serialize a Frame to JSON."""
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(frame.to_dict(), f, indent=2, ensure_ascii=False)


def load_frame(path: str | Path) -> Frame:
    """Deserialize a Frame from JSON."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return Frame.from_dict(d)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _assert_close(a, b, tol=1e-6, msg=""):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.linalg.norm(a - b) > tol:
        raise AssertionError(f"{msg}: {a} != {b} (diff {np.linalg.norm(a - b)})")


def _run_tests():
    import tempfile

    print("=" * 60)
    print("wafer_tray.py self tests")
    print("=" * 60)

    # 1. Load real tray geometry and print summary.
    tray_path = Path("C:/Users/Lenovo/tray_geometry.json")
    geom = load_tray_geometry(tray_path)
    print(
        f"[1] tray_geometry.json: rows={geom['rows']}, cols={geom['cols']}, "
        f"pitch_x={geom['pitch_x']}, pitch_y={geom['pitch_y']}, "
        f"slots={len(geom['slots'])}"
    )
    assert geom["rows"] == 6
    assert geom["cols"] == 6
    assert len(geom["slots"]) == 36
    assert all(s["row"] in range(6) and s["col"] in range(6) for s in geom["slots"])

    # 2. Aligned grid test.
    p00 = (0.0, 0.0, 42.0)
    p_col = (125.0, 0.0, 42.0)
    p_row = (0.0, 125.0, 42.0)
    frame = solve_frame(p00, p_col, p_row, col_n=5, row_m=5)
    assert math.isclose(frame.yaw_rad, 0.0, abs_tol=1e-9)
    pose_00 = pocket_pose(frame, 0, 0)
    pose_55 = pocket_pose(frame, 5, 5)
    _assert_close([pose_00["x"], pose_00["y"], pose_00["z"]], [0.0, 0.0, 42.0], tol=1e-9)
    _assert_close([pose_55["x"], pose_55["y"], pose_55["z"]], [125.0, 125.0, 42.0], tol=1e-9)
    print("[2] aligned grid: yaw=0, pocket(0,0)=(0,0,42), pocket(5,5)=(125,125,42)  OK")

    # 3. Rotated 30 deg grid test.
    yaw = math.radians(30.0)
    cy, sy = math.cos(yaw), math.sin(yaw)
    p0 = np.array([100.0, 200.0, 42.0])
    p0_xy = p0[:2]
    xh = np.array([cy, sy])
    yh = np.array([-sy, cy])
    p_c = np.append(p0_xy + 5 * 25.0 * xh, 42.0)
    p_r = np.append(p0_xy + 5 * 25.0 * yh, 42.0)
    frame_rot = solve_frame(p0, p_c, p_r, col_n=5, row_m=5)
    assert math.isclose(frame_rot.yaw_rad, yaw, abs_tol=1e-9)

    expected_55_xy = p0_xy + 5 * 25.0 * xh + 5 * 25.0 * yh
    pose_rot_55 = pocket_pose(frame_rot, 5, 5)
    _assert_close(
        [pose_rot_55["x"], pose_rot_55["y"], pose_rot_55["z"]],
        [expected_55_xy[0], expected_55_xy[1], 42.0],
        tol=1e-6,
    )
    print("[3] rotated 30 deg grid: yaw and pocket(5,5) OK")

    # 4. Orientation composition tests.
    R_down = (math.pi, 0.0, 0.0)  # gripper pointing straight down, base x/y aligned

    # yaw = 0 -> identical to R_down
    orient0 = compose_place_orientation(R_down, 0.0)
    _assert_close(orient0, R_down, tol=1e-9)
    print(f"[4a] compose yaw=0 -> {orient0}  OK")

    # yaw = 90 deg -> verify by matrix reconstruction, not scalar addition
    yaw90 = math.radians(90.0)
    orient90 = compose_place_orientation(R_down, yaw90)
    R_expected = _zyx_to_matrix(0.0, 0.0, yaw90) @ _zyx_to_matrix(*R_down)
    R_actual = _zyx_to_matrix(*orient90)
    _assert_close(R_actual, R_expected, tol=1e-9)
    # Also confirm the scalar-add shortcut would be wrong for a non-trivial R_down.
    R_down2 = (math.pi / 4, math.pi / 6, math.pi / 3)
    for test_yaw in [0.0, math.radians(45.0), math.radians(90.0), math.radians(-30.0)]:
        orient = compose_place_orientation(R_down2, test_yaw)
        R_exp = _zyx_to_matrix(0.0, 0.0, test_yaw) @ _zyx_to_matrix(*R_down2)
        R_act = _zyx_to_matrix(*orient)
        _assert_close(R_act, R_exp, tol=1e-9)
    print("[4b] compose yaw=90 deg and arbitrary R_down matrix checks OK")

    # 5. Frame serialization round-trip.
    with tempfile.TemporaryDirectory() as tmp:
        fpath = Path(tmp) / "frame.json"
        save_frame(frame, fpath)
        frame2 = load_frame(fpath)
        _assert_close(frame2.origin, frame.origin, tol=1e-9)
        _assert_close(frame2.xhat, frame.xhat, tol=1e-9)
        _assert_close(frame2.yhat, frame.yhat, tol=1e-9)
        assert math.isclose(frame2.yaw_rad, frame.yaw_rad, abs_tol=1e-9)
        assert math.isclose(frame2.z, frame.z, abs_tol=1e-9)
    print("[5] frame save/load round-trip OK")

    # 6. Back-project all 36 slots from the real geometry JSON.
    geom = load_tray_geometry(tray_path)
    slots_by_key = {(s["row"], s["col"]): s for s in geom["slots"]}

    # Build a frame from corner probes on the CAD data.
    p00 = slots_by_key[(0, 0)]["rel_xy"] + [geom["origin_slot_center_mm"][2]]
    p_col = slots_by_key[(0, 5)]["rel_xy"] + [geom["origin_slot_center_mm"][2]]
    p_row = slots_by_key[(5, 0)]["rel_xy"] + [geom["origin_slot_center_mm"][2]]
    frame_real = solve_frame(p00, p_col, p_row, col_n=5, row_m=5, z=p00[2])
    print(f"[6] real tray frame yaw={math.degrees(frame_real.yaw_rad):.4f} deg")
    assert math.isclose(frame_real.yaw_rad, 0.0, abs_tol=1e-6)

    max_err = 0.0
    for s in geom["slots"]:
        pose = pocket_pose(frame_real, s["row"], s["col"])
        err = math.hypot(pose["x"] - s["rel_xy"][0], pose["y"] - s["rel_xy"][1])
        max_err = max(max_err, err)
    print(f"[6] max back-projection error vs JSON rel_xy = {max_err:.6f} mm")
    assert max_err < 0.02, f"back-projection error too large: {max_err} mm"

    print("=" * 60)
    print("ALL SELF TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()
