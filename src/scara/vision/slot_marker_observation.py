"""Observe the 36 slot markers inside a pose-projected tray grid.

This module is the bridge between the metric Tray Frame and the stable parts
of ``tools/tray_marker_detector_v2``.  The Tray Frame owns slot identity and
geometry.  The legacy layout contributes only the fixed marker ID printed in
each slot; it is never allowed to reorder the grid at runtime.

No robot command is issued here.  A missing marker is reported as evidence,
not immediately interpreted as an occupied or empty slot.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

from .tray_pose_estimator import TrayBoardPoseEstimator, TrayPoseEstimate


LEGACY_SLOT_DICTIONARY = "DICT_4X4_50"
DEFAULT_SLOT_HALF_EXTENT_MM = 15.5
DEFAULT_CANONICAL_PATCH_SIZE = 192


@dataclass(frozen=True)
class SlotMarkerLayout:
    """Fixed mapping from metric slot names (P00..P55) to printed IDs."""

    dictionary_name: str
    marker_id_by_slot: dict[str, int]
    metric_slot_transform: str
    source_path: str

    def to_json(self) -> dict[str, Any]:
        return {
            "dictionary": self.dictionary_name,
            "marker_id_by_slot": dict(self.marker_id_by_slot),
            "metric_slot_transform": self.metric_slot_transform,
            "source_path": self.source_path,
        }


@dataclass(frozen=True)
class ArucoObservation:
    """One decoded marker in image coordinates."""

    marker_id: int
    center_px: tuple[float, float]
    corners_px: tuple[tuple[float, float], ...]
    angle_deg: float
    perimeter_px: float
    area_px: float
    square_quality: float

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.marker_id,
            "center_px": list(self.center_px),
            "corners_px": [list(point) for point in self.corners_px],
            "angle_deg": self.angle_deg,
            "perimeter_px": self.perimeter_px,
            "area_px": self.area_px,
            "square_quality": self.square_quality,
        }


@dataclass(frozen=True)
class SlotProjection:
    """One design slot projected from Tray millimetres into the image."""

    slot_key: str
    row: int
    column: int
    center_T_mm: tuple[float, float, float]
    center_px: tuple[float, float]
    polygon_T_mm: tuple[tuple[float, float, float], ...]
    polygon_px: tuple[tuple[float, float], ...]
    image_coverage_ratio: float
    projected_area_px: float

    def to_json(self) -> dict[str, Any]:
        return {
            "slot_key": self.slot_key,
            "row": self.row,
            "column": self.column,
            "center_T_mm": list(self.center_T_mm),
            "center_px": list(self.center_px),
            "polygon_T_mm": [list(point) for point in self.polygon_T_mm],
            "polygon_px": [list(point) for point in self.polygon_px],
            "image_coverage_ratio": self.image_coverage_ratio,
            "projected_area_px": self.projected_area_px,
        }


@dataclass(frozen=True)
class SlotMarkerEvidence:
    """Marker evidence associated with one metric slot."""

    slot_key: str
    expected_marker_id: Optional[int]
    decoded: bool
    decoded_marker_id: Optional[int]
    marker_like_visible: bool
    center_error_px: Optional[float]
    detection_quality: Optional[float]
    pattern_features: dict[str, float]

    def to_json(self) -> dict[str, Any]:
        return {
            "slot_key": self.slot_key,
            "expected_marker_id": self.expected_marker_id,
            "decoded": self.decoded,
            "decoded_marker_id": self.decoded_marker_id,
            "marker_like_visible": self.marker_like_visible,
            "center_error_px": self.center_error_px,
            "detection_quality": self.detection_quality,
            "pattern_features": dict(self.pattern_features),
        }


def load_slot_marker_layout(path: Path) -> SlotMarkerLayout:
    """Load the old 6x6 layout without importing its image-plane geometry."""
    path = Path(path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"slot marker layout not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"slot marker layout is not valid JSON: {path}") from exc
    rows = int(payload.get("rows", 0))
    columns = int(payload.get("cols", 0))
    grid = payload.get("slot_id_grid")
    if rows != 6 or columns != 6 or not isinstance(grid, list) or len(grid) != 6:
        raise ValueError("slot marker layout must contain a 6x6 slot_id_grid")
    transform_name = str(payload.get("metric_slot_transform") or "identity")
    transforms = {
        "identity": lambda row, column: (row, column),
        "rot90": lambda row, column: (5 - column, row),
        "rot180": lambda row, column: (5 - row, 5 - column),
        "rot270": lambda row, column: (column, 5 - row),
        "flip_columns": lambda row, column: (row, 5 - column),
        "flip_rows": lambda row, column: (5 - row, column),
        "transpose": lambda row, column: (column, row),
        "anti_transpose": lambda row, column: (5 - column, 5 - row),
    }
    if transform_name not in transforms:
        raise ValueError(f"unsupported metric_slot_transform: {transform_name}")
    marker_id_by_slot: dict[str, int] = {}
    seen_ids: set[int] = set()
    for row, values in enumerate(grid):
        if not isinstance(values, list) or len(values) != 6:
            raise ValueError("each slot_id_grid row must contain six IDs")
    transform = transforms[transform_name]
    for row in range(6):
        for column in range(6):
            layout_row, layout_column = transform(row, column)
            marker_id = int(grid[layout_row][layout_column])
            if marker_id in seen_ids:
                raise ValueError(f"duplicate slot marker ID: {marker_id}")
            seen_ids.add(marker_id)
            marker_id_by_slot[f"P{row}{column}"] = marker_id
    dictionary_name = str(payload.get("dictionary") or LEGACY_SLOT_DICTIONARY)
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"OpenCV does not support marker dictionary {dictionary_name}")
    return SlotMarkerLayout(
        dictionary_name,
        marker_id_by_slot,
        transform_name,
        str(path),
    )


def _marker_metrics(corners: np.ndarray) -> tuple[float, float, float, float]:
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    edges = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    perimeter = float(np.sum(edges))
    area = abs(float(cv2.contourArea(points.astype(np.float32))))
    longest = max(float(np.max(edges)), 1e-9)
    edge_ratio = float(np.min(edges)) / longest
    diagonal = np.linalg.norm(points[[0, 1]] - points[[2, 3]], axis=1)
    diagonal_ratio = float(np.min(diagonal)) / max(float(np.max(diagonal)), 1e-9)
    return perimeter, area, edge_ratio, diagonal_ratio


def _make_aruco_detector(dictionary_name: str) -> Any:
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(f"OpenCV does not support marker dictionary {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def detect_aruco_observations(
    image: np.ndarray,
    dictionary_name: str,
    *,
    scales: Sequence[float] = (1.0, 1.5, 2.0),
) -> dict[int, ArucoObservation]:
    """Detect IDs at several scales and retain the largest valid duplicate.

    The multi-scale policy and duplicate selection are retained from the old
    tray marker detector because they are useful when the overview camera is
    zoomed out.  Marker corner orientation is recorded but is not used to set
    the Tray Frame angle.
    """
    if image is None or image.ndim not in (2, 3):
        raise ValueError("image must be a valid grayscale or BGR array")
    detector = _make_aruco_detector(dictionary_name)
    observations: dict[int, ArucoObservation] = {}
    for raw_scale in scales:
        scale = float(raw_scale)
        if scale <= 0.0:
            raise ValueError("marker detection scales must be positive")
        scaled = image if abs(scale - 1.0) < 1e-9 else cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        corners, ids, _rejected = detector.detectMarkers(scaled)
        if ids is None:
            continue
        for raw_id, raw_corners in zip(ids.reshape(-1), corners):
            marker_id = int(raw_id)
            points = np.asarray(raw_corners, dtype=np.float64).reshape(4, 2) / scale
            perimeter, area, edge_ratio, diagonal_ratio = _marker_metrics(points)
            top_edge = points[1] - points[0]
            angle = math.degrees(math.atan2(float(top_edge[1]), float(top_edge[0])))
            quality = max(0.0, min(1.0, edge_ratio * diagonal_ratio))
            candidate = ArucoObservation(
                marker_id=marker_id,
                center_px=tuple(float(x) for x in np.mean(points, axis=0)),
                corners_px=tuple(tuple(float(x) for x in point) for point in points),
                angle_deg=float(angle),
                perimeter_px=perimeter,
                area_px=area,
                square_quality=quality,
            )
            previous = observations.get(marker_id)
            if previous is None or candidate.perimeter_px > previous.perimeter_px:
                observations[marker_id] = candidate
    return observations


def _polygon_coverage_ratio(points: np.ndarray, image_shape: tuple[int, ...]) -> float:
    polygon = cv2.convexHull(np.asarray(points, dtype=np.float32).reshape(-1, 2))
    area = abs(float(cv2.contourArea(polygon)))
    if area <= 1e-9:
        return 0.0
    height, width = image_shape[:2]
    image_polygon = np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )
    overlap, _ = cv2.intersectConvexConvex(polygon, image_polygon)
    return float(max(0.0, min(1.0, overlap / area)))


def build_slot_projections(
    geometry: Mapping[str, Any],
    estimator: TrayBoardPoseEstimator,
    estimate: TrayPoseEstimate,
    image_shape: tuple[int, ...],
    *,
    half_extent_mm: float = DEFAULT_SLOT_HALF_EXTENT_MM,
) -> dict[str, SlotProjection]:
    """Project all 36 fixed metric slot cells into the current frame."""
    if not estimate.success or not estimate.quality_passed:
        raise ValueError("a quality-passed tray pose is required")
    if half_extent_mm <= 0.0:
        raise ValueError("slot half extent must be positive")
    raw_slots = geometry.get("slots")
    if not isinstance(raw_slots, Mapping) or len(raw_slots) != 36:
        raise ValueError("tray geometry must contain 36 slots")
    result: dict[str, SlotProjection] = {}
    for slot_key in sorted(raw_slots):
        if len(slot_key) != 3 or not slot_key.startswith("P"):
            raise ValueError(f"invalid metric slot key: {slot_key}")
        row = int(slot_key[1])
        column = int(slot_key[2])
        center = np.asarray(raw_slots[slot_key], dtype=np.float64).reshape(3)
        x, y, z = (float(value) for value in center)
        half = float(half_extent_mm)
        # Canonical patch x follows increasing column (-Y_T); patch y follows
        # increasing row (-X_T).  The order is TL, TR, BR, BL.
        polygon_T = np.array(
            [
                [x + half, y + half, z],
                [x + half, y - half, z],
                [x - half, y - half, z],
                [x - half, y + half, z],
            ],
            dtype=np.float64,
        )
        projected = estimator.project_tray_points(
            np.vstack((center.reshape(1, 3), polygon_T)), estimate
        )
        center_px = projected[0]
        polygon_px = projected[1:]
        area = abs(float(cv2.contourArea(polygon_px.astype(np.float32))))
        result[slot_key] = SlotProjection(
            slot_key=slot_key,
            row=row,
            column=column,
            center_T_mm=tuple(float(value) for value in center),
            center_px=tuple(float(value) for value in center_px),
            polygon_T_mm=tuple(tuple(float(value) for value in point) for point in polygon_T),
            polygon_px=tuple(tuple(float(value) for value in point) for point in polygon_px),
            image_coverage_ratio=_polygon_coverage_ratio(polygon_px, image_shape),
            projected_area_px=area,
        )
    return result


def warp_slot_patch(
    image: np.ndarray,
    projection: SlotProjection,
    *,
    output_size: int = DEFAULT_CANONICAL_PATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Perspective-normalize one slot and return patch plus image->patch H."""
    if output_size < 32:
        raise ValueError("canonical slot patch must be at least 32 pixels")
    source = np.asarray(projection.polygon_px, dtype=np.float32).reshape(4, 2)
    edge = float(output_size - 1)
    target = np.array([[0.0, 0.0], [edge, 0.0], [edge, edge], [0.0, edge]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, target)
    patch = cv2.warpPerspective(
        image,
        transform,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return patch, transform


def patch_points_to_image(points: np.ndarray, image_to_patch: np.ndarray) -> np.ndarray:
    """Map canonical-patch points back into the original camera image."""
    inverse = np.linalg.inv(np.asarray(image_to_patch, dtype=np.float64))
    source = np.asarray(points, dtype=np.float64).reshape(1, -1, 2)
    return cv2.perspectiveTransform(source.astype(np.float32), inverse.astype(np.float32)).reshape(-1, 2)


def marker_like_pattern_features(patch: np.ndarray) -> tuple[bool, dict[str, float]]:
    """Recognize an undecoded black/white marker pattern in a canonical slot."""
    if patch is None or patch.size == 0:
        return False, {}
    bgr = patch if patch.ndim == 3 else cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    height, width = bgr.shape[:2]
    margin_x = max(1, int(round(width * 0.20)))
    margin_y = max(1, int(round(height * 0.20)))
    core = bgr[margin_y : height - margin_y, margin_x : width - margin_x]
    gray = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    contrast = float(np.percentile(gray, 90) - np.percentile(gray, 10))
    dark_ratio = float(np.mean(gray < 82))
    bright_ratio = float(np.mean(gray > 128))
    colorful_ratio = float(np.mean(hsv[:, :, 1] > 55))
    edges = cv2.Canny(gray, 50, 140)
    edge_ratio = float(np.mean(edges > 0))
    visible = bool(
        contrast >= 52.0
        and dark_ratio >= 0.16
        and bright_ratio >= 0.10
        and edge_ratio >= 0.035
        and colorful_ratio <= 0.22
    )
    return visible, {
        "contrast": contrast,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "edge_ratio": edge_ratio,
        "colorful_ratio": colorful_ratio,
    }


def associate_marker_to_slot(
    projection: SlotProjection,
    layout: SlotMarkerLayout,
    observations: Mapping[int, ArucoObservation],
    patch: np.ndarray,
) -> SlotMarkerEvidence:
    """Associate only the slot's configured ID and reject a far-away duplicate."""
    expected_id = layout.marker_id_by_slot.get(projection.slot_key)
    observation = observations.get(expected_id) if expected_id is not None else None
    center_error: Optional[float] = None
    decoded = False
    if observation is not None:
        center_error = math.dist(projection.center_px, observation.center_px)
        characteristic = max(1.0, math.sqrt(max(projection.projected_area_px, 1.0)))
        decoded = center_error <= 0.46 * characteristic
    marker_like = False
    features: dict[str, float] = {}
    if not decoded and projection.image_coverage_ratio >= 0.90:
        marker_like, features = marker_like_pattern_features(patch)
    return SlotMarkerEvidence(
        slot_key=projection.slot_key,
        expected_marker_id=expected_id,
        decoded=decoded,
        decoded_marker_id=expected_id if decoded else None,
        marker_like_visible=marker_like,
        center_error_px=center_error,
        detection_quality=observation.square_quality if decoded and observation is not None else None,
        pattern_features=features,
    )


__all__ = [
    "ArucoObservation",
    "DEFAULT_CANONICAL_PATCH_SIZE",
    "DEFAULT_SLOT_HALF_EXTENT_MM",
    "SlotMarkerEvidence",
    "SlotMarkerLayout",
    "SlotProjection",
    "associate_marker_to_slot",
    "build_slot_projections",
    "detect_aruco_observations",
    "load_slot_marker_layout",
    "marker_like_pattern_features",
    "patch_points_to_image",
    "warp_slot_patch",
]
