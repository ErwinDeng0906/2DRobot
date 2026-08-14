"""Stage 2: construct the Tray Frame and rigid A-H ArUco board geometry.

The taught values in ``scara_presets.json`` are SCARA joints
``[J1_deg, J2_deg, J3_mm, J4_deg]``.  They are never treated as Cartesian XY.
J1/J2 first pass through two-link forward kinematics.  Those mechanical-plane
coordinates are then used only to establish the independently named Tray Frame
``T``:

* origin: centre of slot P00;
* ``+X_T``: direction from P50 to P00;
* ``+Y_T``: direction from P05 to P00, orthogonalized against ``+X_T``;
* ``+Z_T = +X_T x +Y_T``: from the slot-bottom target plane toward markers.

The 6 x 6 slot centres are exact design geometry with 25 mm pitch.  Each A-H
marker is a measured 13.27 mm square.  Its measured centre and two labelled
left corners (UL/DL) determine orientation; the final four corners are forced
to be a rigid square centred at the taught centre.  Each marker retains its own
measured J3 height relative to the slot-bottom target plane.

No camera, Qt, or robot command is used in this module.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCARA_LINK1_MM = 225.0
SCARA_LINK2_MM = 175.0
SLOT_BOTTOM_J3_MM = -52.01
SLOT_GRID_SIZE = 6
SLOT_PITCH_MM = 25.0
MARKER_SIDE_MM = 13.27
ARUCO_DICTIONARY_NAME = "DICT_4X4_1000"

MARKER_LABEL_TO_ID = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "H": 8,
}

SLOT_PRESET_KEYS = {
    "P00": "P00 float",
    "P05": "P05 float",
    "P50": "P50 float",
    "P55": "P55 float",
}


@dataclass(frozen=True)
class TrayFrameDefinition:
    """Tray origin and axes represented in the mechanical planar frame."""

    origin_mechanical_xy_mm: np.ndarray
    x_axis_mechanical: np.ndarray
    y_axis_mechanical: np.ndarray
    z_axis_mechanical: np.ndarray

    @property
    def rotation_mechanical_from_tray(self) -> np.ndarray:
        """Return columns [X_T, Y_T, Z_T] expressed mechanically."""
        return np.column_stack(
            (
                self.x_axis_mechanical,
                self.y_axis_mechanical,
                self.z_axis_mechanical,
            )
        )

    def mechanical_xy_to_tray_xy(self, xy_mm: Sequence[float]) -> np.ndarray:
        """Project a mechanical-plane point onto the independent Tray axes."""
        xy = np.asarray(xy_mm, dtype=np.float64).reshape(2)
        delta = xy - self.origin_mechanical_xy_mm
        return np.array(
            [
                float(np.dot(delta, self.x_axis_mechanical[:2])),
                float(np.dot(delta, self.y_axis_mechanical[:2])),
            ],
            dtype=np.float64,
        )


def _finite_joint_values(value: Any, label: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        candidate = value.reshape(-1)
    elif isinstance(value, (list, tuple)):
        candidate = np.asarray(value)
    else:
        candidate = np.asarray([])
    if candidate.size != 4:
        raise ValueError(f"{label} 必须包含 J1/J2/J3/J4 四个值")
    result = np.asarray(candidate, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} 包含 NaN/Inf")
    return result


def load_presets(path: Path) -> dict[str, np.ndarray]:
    """Load and validate all preset rows without changing their semantics."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"找不到预设文件：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"预设文件不是有效JSON：{path}（{exc}）") from exc
    if not isinstance(raw, dict):
        raise ValueError("scara_presets.json 顶层必须是对象")
    return {
        str(key): _finite_joint_values(value, str(key))
        for key, value in raw.items()
    }


def scara_planar_fk(
    joints: Sequence[float],
    link1_mm: float = SCARA_LINK1_MM,
    link2_mm: float = SCARA_LINK2_MM,
) -> np.ndarray:
    """Convert J1/J2 to mechanical XY with two-link forward kinematics.

    ``x = L1*cos(q1) + L2*cos(q1+q2)``
    ``y = L1*sin(q1) + L2*sin(q1+q2)``
    """
    values = _finite_joint_values(joints, "关节值")
    q1 = math.radians(float(values[0]))
    q12 = math.radians(float(values[0] + values[1]))
    return np.array(
        [
            link1_mm * math.cos(q1) + link2_mm * math.cos(q12),
            link1_mm * math.sin(q1) + link2_mm * math.sin(q12),
        ],
        dtype=np.float64,
    )


def build_tray_frame(presets: Mapping[str, np.ndarray]) -> tuple[TrayFrameDefinition, dict[str, Any]]:
    """Build a strictly orthonormal Tray Frame from P00/P05/P50/P55.

    ``+X_T`` is exactly P50 -> P00.  ``+Y_T`` starts as P05 -> P00 and is
    Gram-Schmidt orthogonalized.  This preserves the user's sign definition
    while preventing taught-point noise from creating a skew coordinate frame.
    """
    points = {
        label: scara_planar_fk(presets[key])
        for label, key in SLOT_PRESET_KEYS.items()
    }
    origin = points["P00"]
    raw_x = origin - points["P50"]
    raw_y = origin - points["P05"]
    norm_x = float(np.linalg.norm(raw_x))
    norm_y = float(np.linalg.norm(raw_y))
    if norm_x < 1e-6 or norm_y < 1e-6:
        raise ValueError("P00/P05/P50 无法定义Tray方向")
    x2 = raw_x / norm_x
    y2_raw = raw_y / norm_y
    y2 = y2_raw - x2 * float(np.dot(y2_raw, x2))
    y_norm = float(np.linalg.norm(y2))
    if y_norm < 1e-6:
        raise ValueError("P50→P00 与 P05→P00 近似共线，无法定义Tray Frame")
    y2 /= y_norm
    # Keep +Y aligned with the user's P05 -> P00 direction after orthogonalizing.
    if float(np.dot(y2, raw_y)) < 0.0:
        y2 = -y2
    x3 = np.array([x2[0], x2[1], 0.0], dtype=np.float64)
    y3 = np.array([y2[0], y2[1], 0.0], dtype=np.float64)
    z3 = np.cross(x3, y3)
    z3 /= np.linalg.norm(z3)
    frame = TrayFrameDefinition(origin, x3, y3, z3)

    measured_tray_xy = {
        label: frame.mechanical_xy_to_tray_xy(xy).tolist()
        for label, xy in points.items()
    }
    ideal_tray_xy = {
        "P00": [0.0, 0.0],
        "P05": [0.0, -5.0 * SLOT_PITCH_MM],
        "P50": [-5.0 * SLOT_PITCH_MM, 0.0],
        "P55": [-5.0 * SLOT_PITCH_MM, -5.0 * SLOT_PITCH_MM],
    }
    residuals = {
        label: (
            np.asarray(measured_tray_xy[label]) - np.asarray(ideal_tray_xy[label])
        ).tolist()
        for label in ideal_tray_xy
    }
    raw_angle_deg = math.degrees(
        math.acos(
            float(
                np.clip(
                    np.dot(raw_x, raw_y) / (norm_x * norm_y),
                    -1.0,
                    1.0,
                )
            )
        )
    )
    diagnostics = {
        "measured_mechanical_xy_mm": {
            label: xy.tolist() for label, xy in points.items()
        },
        "measured_tray_xy_mm": measured_tray_xy,
        "ideal_tray_xy_mm": ideal_tray_xy,
        "residual_measured_minus_ideal_mm": residuals,
        "raw_axis_angle_deg": raw_angle_deg,
        "raw_p50_to_p00_length_mm": norm_x,
        "raw_p05_to_p00_length_mm": norm_y,
    }
    return frame, diagnostics


def build_slot_centres() -> dict[str, list[float]]:
    """Return exact 6 x 6 slot-bottom targets in Tray coordinates.

    Name ``Prc`` uses the existing convention: ``r`` increases from P00 toward
    P50, hence negative X_T; ``c`` increases toward P05, hence negative Y_T.
    """
    return {
        f"P{row}{column}": [
            -row * SLOT_PITCH_MM,
            -column * SLOT_PITCH_MM,
            0.0,
        ]
        for row in range(SLOT_GRID_SIZE)
        for column in range(SLOT_GRID_SIZE)
    }


def _marker_orientation_from_taught_points(
    centre_xy: np.ndarray,
    ul_xy: np.ndarray,
    dl_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fit marker right/down axes from centre plus taught UL and DL.

    OpenCV marker corner order is ``[UL, UR, DR, DL]``.  Let marker unit axes
    ``u`` point UL->UR and ``v`` point UL->DL.  The two left corners should
    satisfy ``DL-UL = side*v`` and their midpoint should be ``C-side/2*u``.
    Taught points are noisy, so both cues are fused, orthogonalized, and the
    final square is reconstructed with exact measured side length.
    """
    left_down = dl_xy - ul_xy
    left_norm = float(np.linalg.norm(left_down))
    if left_norm < 1e-6:
        raise ValueError("UL和DL示教点重合，无法确定Marker朝向")
    v = left_down / left_norm
    left_mid = 0.5 * (ul_xy + dl_xy)
    outward = centre_xy - left_mid
    u_from_mid = outward - v * float(np.dot(outward, v))
    u_norm = float(np.linalg.norm(u_from_mid))
    if u_norm < 1e-6:
        # In 2D there are two perpendiculars; choose the one whose dot product
        # with centre-left_mid is positive whenever that cue exists.
        u_from_mid = np.array([v[1], -v[0]], dtype=np.float64)
        u_norm = 1.0
    u = u_from_mid / u_norm
    if float(np.dot(u, outward)) < 0.0:
        u = -u
    # Recompute v as the perpendicular closest to the measured UL->DL vector.
    v_candidate = np.array([-u[1], u[0]], dtype=np.float64)
    if float(np.dot(v_candidate, left_down)) < 0.0:
        v_candidate = -v_candidate
    v = v_candidate
    diagnostics = {
        "taught_left_edge_length_mm": left_norm,
        "taught_left_midpoint_to_centre_mm": float(np.linalg.norm(outward)),
        "taught_left_edge_minus_side_mm": left_norm - MARKER_SIDE_MM,
        "taught_centre_offset_minus_half_side_mm": (
            float(np.linalg.norm(outward)) - MARKER_SIDE_MM / 2.0
        ),
    }
    return u, v, diagnostics


def build_marker_geometry(
    presets: Mapping[str, np.ndarray], frame: TrayFrameDefinition
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build exact marker squares in T while retaining individual heights."""
    markers: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    half = MARKER_SIDE_MM / 2.0
    for label, marker_id in MARKER_LABEL_TO_ID.items():
        centre_joint = presets[label]
        ul_joint = presets[f"{label}ul"]
        dl_joint = presets[f"{label}dl"]
        centre_xy = frame.mechanical_xy_to_tray_xy(scara_planar_fk(centre_joint))
        taught_ul_xy = frame.mechanical_xy_to_tray_xy(scara_planar_fk(ul_joint))
        taught_dl_xy = frame.mechanical_xy_to_tray_xy(scara_planar_fk(dl_joint))
        u, v, fit = _marker_orientation_from_taught_points(
            centre_xy, taught_ul_xy, taught_dl_xy
        )
        marker_z = float(centre_joint[2] - SLOT_BOTTOM_J3_MM)
        if marker_z <= 0.0:
            raise ValueError(f"Marker {label} 不在槽底目标平面上方")
        ul = centre_xy - half * u - half * v
        ur = centre_xy + half * u - half * v
        dr = centre_xy + half * u + half * v
        dl = centre_xy - half * u + half * v
        corners_xy = np.vstack((ul, ur, dr, dl))
        corners_xyz = np.column_stack(
            (corners_xy, np.full(4, marker_z, dtype=np.float64))
        )
        predicted_left = np.vstack((ul, dl))
        taught_left = np.vstack((taught_ul_xy, taught_dl_xy))
        fit.update(
            {
                "taught_ul_tray_xy_mm": taught_ul_xy.tolist(),
                "taught_dl_tray_xy_mm": taught_dl_xy.tolist(),
                "predicted_ul_tray_xy_mm": ul.tolist(),
                "predicted_dl_tray_xy_mm": dl.tolist(),
                "left_corner_fit_rms_mm": math.sqrt(
                    float(np.mean(np.sum((predicted_left - taught_left) ** 2, axis=1)))
                ),
            }
        )
        markers[label] = {
            "label": label,
            "id": marker_id,
            "side_length_mm": MARKER_SIDE_MM,
            "center_T_mm": [float(centre_xy[0]), float(centre_xy[1]), marker_z],
            "surface_height_above_slot_target_mm": marker_z,
            "corner_order": ["UL", "UR", "DR", "DL"],
            "corners_T_mm": corners_xyz.tolist(),
            "u_axis_T": [float(u[0]), float(u[1]), 0.0],
            "v_axis_T": [float(v[0]), float(v[1]), 0.0],
            "source_presets": {
                "center": label,
                "upper_left": f"{label}ul",
                "down_left": f"{label}dl",
            },
        }
        diagnostics[label] = fit
    return markers, diagnostics


def validate_geometry(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate schema, rigid squares, unique IDs, axes, slots, and heights."""
    errors: list[str] = []
    warnings: list[str] = []
    frame = payload.get("tray_frame", {})
    rotation = np.asarray(
        frame.get("rotation_mechanical_from_tray"), dtype=np.float64
    )
    if rotation.shape != (3, 3):
        errors.append("Tray rotation must be 3x3")
    else:
        orth_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
        det = float(np.linalg.det(rotation))
        if orth_error > 1e-9:
            errors.append(f"Tray axes are not orthonormal: {orth_error}")
        if abs(det - 1.0) > 1e-9:
            errors.append(f"Tray rotation determinant is {det}")

    slots = payload.get("slots", {})
    if len(slots) != SLOT_GRID_SIZE**2:
        errors.append(f"Expected 36 slots, got {len(slots)}")
    markers = payload.get("markers", {})
    ids: list[int] = []
    for label in MARKER_LABEL_TO_ID:
        marker = markers.get(label)
        if not isinstance(marker, dict):
            errors.append(f"Missing marker {label}")
            continue
        ids.append(int(marker.get("id", -1)))
        corners = np.asarray(marker.get("corners_T_mm"), dtype=np.float64)
        if corners.shape != (4, 3):
            errors.append(f"Marker {label} corners must be 4x3")
            continue
        edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        if float(np.max(np.abs(edges - MARKER_SIDE_MM))) > 1e-6:
            errors.append(f"Marker {label} is not a {MARKER_SIDE_MM}mm square")
        centre = np.asarray(marker.get("center_T_mm"), dtype=np.float64)
        if float(np.linalg.norm(corners.mean(axis=0) - centre)) > 1e-9:
            errors.append(f"Marker {label} corners are not centered")
        if centre[2] <= 0:
            errors.append(f"Marker {label} is not above slot target plane")
    if len(ids) != len(set(ids)):
        errors.append("Marker IDs are not unique")

    fit = payload.get("diagnostics", {}).get("marker_corner_fit", {})
    for label, row in fit.items():
        rms = float(row.get("left_corner_fit_rms_mm", 0.0))
        if rms > 1.0:
            warnings.append(
                f"Marker {label} taught UL/DL fit RMS is {rms:.3f} mm"
            )
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def build_tray_board_geometry(presets_path: Path) -> dict[str, Any]:
    """Construct complete serializable Stage-2 geometry from taught joints."""
    presets_path = Path(presets_path).resolve()
    presets = load_presets(presets_path)
    required = set(SLOT_PRESET_KEYS.values())
    for label in MARKER_LABEL_TO_ID:
        required.update((label, f"{label}ul", f"{label}dl"))
    missing = sorted(required - set(presets))
    if missing:
        raise ValueError("scara_presets.json 缺少：" + ", ".join(missing))

    frame, frame_diagnostics = build_tray_frame(presets)
    markers, marker_diagnostics = build_marker_geometry(presets, frame)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_presets": str(presets_path),
        "units": {"length": "mm", "angle": "degree"},
        "scara_model_used_only_for_teach_conversion": {
            "link1_mm": SCARA_LINK1_MM,
            "link2_mm": SCARA_LINK2_MM,
        },
        "tray_frame": {
            "name": "T",
            "origin": "slot P00 center on slot-bottom target plane",
            "x_positive_definition": "P50 slot center toward P00 slot center",
            "y_positive_definition": "P05 slot center toward P00 slot center, orthogonalized",
            "z_positive_definition": "X_T cross Y_T; slot-bottom target plane toward markers",
            "origin_mechanical_xy_mm": frame.origin_mechanical_xy_mm.tolist(),
            "rotation_mechanical_from_tray": (
                frame.rotation_mechanical_from_tray.tolist()
            ),
            "slot_target_plane_z_T_mm": 0.0,
            "slot_bottom_j3_mm_used_for_height_difference": SLOT_BOTTOM_J3_MM,
            "warning": (
                "Mechanical coordinates are retained only as traceable teach-data "
                "diagnostics; all board and target geometry below is expressed in T."
            ),
        },
        "slot_grid": {
            "rows": SLOT_GRID_SIZE,
            "columns": SLOT_GRID_SIZE,
            "pitch_x_mm": SLOT_PITCH_MM,
            "pitch_y_mm": SLOT_PITCH_MM,
            "naming": "Prc; r=0..5 toward P50 (-X_T), c=0..5 toward P05 (-Y_T)",
        },
        "slots": build_slot_centres(),
        "aruco_board": {
            "dictionary": ARUCO_DICTIONARY_NAME,
            "marker_labels": list(MARKER_LABEL_TO_ID),
            "marker_ids": list(MARKER_LABEL_TO_ID.values()),
            "measured_marker_side_mm": MARKER_SIDE_MM,
            "opencv_detected_corner_order": ["UL", "UR", "DR", "DL"],
            "non_coplanar": True,
        },
        "markers": markers,
        "diagnostics": {
            "tray_frame_from_taught_slots": frame_diagnostics,
            "marker_corner_fit": marker_diagnostics,
        },
    }
    payload["validation"] = validate_geometry(payload)
    if not payload["validation"]["valid"]:
        raise ValueError(
            "生成的Tray/Board几何无效："
            + "; ".join(payload["validation"]["errors"])
        )
    return payload


def atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def generate_geometry_file(presets_path: Path, output_path: Path) -> dict[str, Any]:
    payload = build_tray_board_geometry(presets_path)
    atomic_json_write(output_path, payload)
    return payload


__all__ = [
    "ARUCO_DICTIONARY_NAME",
    "MARKER_LABEL_TO_ID",
    "MARKER_SIDE_MM",
    "SLOT_BOTTOM_J3_MM",
    "SLOT_PITCH_MM",
    "TrayFrameDefinition",
    "atomic_json_write",
    "build_marker_geometry",
    "build_slot_centres",
    "build_tray_board_geometry",
    "build_tray_frame",
    "generate_geometry_file",
    "load_presets",
    "scara_planar_fk",
    "validate_geometry",
]
