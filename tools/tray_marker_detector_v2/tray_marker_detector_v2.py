#!/usr/bin/env python3
"""
Detect ArUco markers on a 6x6 chip tray and estimate the tray grid angle.

The tray used in this project has 36 slot markers plus 8 outer locator
markers. Marker print orientation is intentionally not trusted; the tray angle
is estimated from detected marker center positions and fitted grid lines.

Typical calibration command:
    python3 tray_marker_detector.py \
        --image empty_tray.png \
        --save-json tray_analysis.json \
        --save-layout tray_layout.json \
        --annotate tray_annotated.png

Typical runtime command after an empty-tray layout has been saved:
    python3 tray_marker_detector.py --image live_frame.png --layout tray_layout.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


DEFAULT_DICT = "DICT_4X4_50"
DEFAULT_ROWS = 6
DEFAULT_COLS = 6
DEFAULT_OCCLUSION_BOTTOM_RATIO = 0.0
DEFAULT_LOWER_PURPLE = (110, 20, 20)
DEFAULT_UPPER_PURPLE = (165, 255, 255)
DEFAULT_FLAKE_MIN_AREA = 5000.0
DEFAULT_FLAKE_MAX_ASPECT = 2.2
DEFAULT_EDGE_OCCLUSION_MARGIN_RATIO = 0.55
DEFAULT_ARM_OCCLUSION_POLYGON_NORM = [
    (0.354, 1.000),
    (0.377, 0.948),
    (0.393, 0.895),
    (0.407, 0.823),
    (0.418, 0.766),
    (0.448, 0.738),
    (0.486, 0.719),
    (0.528, 0.713),
    (0.569, 0.719),
    (0.594, 0.738),
    (0.609, 0.778),
    (0.620, 0.845),
    (0.642, 0.920),
    (0.667, 1.000),
]
DEFAULT_TRAY_ROI_PADDING_RATIO = 0.18
DEFAULT_CHIP_MAX_ASPECT_OK = 1.28
DEFAULT_CHIP_MAX_ASPECT_WARN = 1.45
DEFAULT_CHIP_MAX_ANGLE_REL_OK = 8.0
DEFAULT_CHIP_MAX_CENTER_OFFSET_RATIO = 0.30
DEFAULT_CHIP_MIN_SIDE_RATIO = 0.48
DEFAULT_CHIP_MAX_SIDE_RATIO = 0.92
DEFAULT_CHIP_STACKED_SIDE_RATIO = 0.91
DEFAULT_CHIP_STACKED_MAX_FILL_RATIO = 0.82
DEFAULT_CHIP_PARTIAL_EDGE_RATIO = 0.22
DEFAULT_CHIP_REFINEMENT_RADIUS_RATIO = 0.58
DEFAULT_CHIP_INTERNAL_LINE_MIN_RATIO = 0.28
DEFAULT_SILVER_GAP_MIN_SPACING_PX = 80.0
DEFAULT_SILVER_GAP_BOUNDARY_HALF_RATIO = 0.05
DEFAULT_SILVER_GAP_PROBE_HALF_RATIO = 0.14
DEFAULT_SILVER_GAP_MIN_COVERAGE_RATIO = 0.55
DEFAULT_SILVER_GAP_MIN_COMPONENT_RATIO = 0.45
DEFAULT_SILVER_GAP_MIN_CHIP_SIDE_RATIO = 0.735


@dataclass
class Marker:
    marker_id: int
    corners: np.ndarray
    center: np.ndarray
    marker_angle_deg: float
    u: float = 0.0
    v: float = 0.0
    u_cluster: int = -1
    v_cluster: int = -1


def normalize_axis_angle(theta: float) -> float:
    """Normalize an axis angle to [-45 deg, 45 deg)."""
    period = math.pi / 2.0
    while theta < -period / 2.0:
        theta += period
    while theta >= period / 2.0:
        theta -= period
    return theta


def circular_mean(angles: Iterable[float], period: float) -> float:
    values = list(angles)
    if not values:
        raise ValueError("cannot average an empty angle list")
    scale = 2.0 * math.pi / period
    s = sum(math.sin(a * scale) for a in values)
    c = sum(math.cos(a * scale) for a in values)
    return math.atan2(s, c) / scale


def get_aruco_dictionary(name: str) -> Any:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("This OpenCV build does not include cv2.aruco")
    if not hasattr(cv2.aruco, name):
        known = sorted(x for x in dir(cv2.aruco) if x.startswith("DICT_"))
        raise ValueError(f"Unknown ArUco dictionary {name!r}. Known values: {known}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def make_detector(dictionary_name: str) -> Any:
    aruco_dict = get_aruco_dictionary(dictionary_name)
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(aruco_dict, params)
    return aruco_dict, params


def detect_markers(image: np.ndarray, dictionary_name: str = DEFAULT_DICT) -> list[Marker]:
    detector = make_detector(dictionary_name)
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        aruco_dict, params = detector
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    if ids is None:
        return []

    markers: list[Marker] = []
    for marker_id, corner in zip(ids.flatten(), corners):
        pts = corner.reshape(4, 2).astype(float)
        center = pts.mean(axis=0)
        top_edge = pts[1] - pts[0]
        marker_angle = math.degrees(math.atan2(float(top_edge[1]), float(top_edge[0])))
        markers.append(
            Marker(
                marker_id=int(marker_id),
                corners=pts,
                center=center,
                marker_angle_deg=marker_angle,
            )
        )
    markers.sort(key=lambda m: m.marker_id)
    return markers


def marker_perimeter(marker: Marker) -> float:
    pts = marker.corners.reshape(4, 2)
    return float(sum(np.linalg.norm(pts[(idx + 1) % 4] - pts[idx]) for idx in range(4)))


def scaled_marker(marker: Marker, scale: float) -> Marker:
    if abs(float(scale) - 1.0) < 1e-9:
        return marker
    factor = 1.0 / float(scale)
    return Marker(
        marker_id=marker.marker_id,
        corners=marker.corners * factor,
        center=marker.center * factor,
        marker_angle_deg=marker.marker_angle_deg,
    )


def detect_markers_multiscale(
    image: np.ndarray,
    dictionary_name: str = DEFAULT_DICT,
    *,
    min_initial_markers: int = 12,
) -> list[Marker]:
    markers_by_id: dict[int, Marker] = {marker.marker_id: marker for marker in detect_markers(image, dictionary_name)}
    if len(markers_by_id) >= min_initial_markers:
        return sorted(markers_by_id.values(), key=lambda marker: marker.marker_id)

    for scale in (1.5, 2.0, 3.0):
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        for marker in detect_markers(resized, dictionary_name):
            candidate = scaled_marker(marker, scale)
            existing = markers_by_id.get(candidate.marker_id)
            if existing is None or marker_perimeter(candidate) > marker_perimeter(existing):
                markers_by_id[candidate.marker_id] = candidate
    return sorted(markers_by_id.values(), key=lambda marker: marker.marker_id)


def nearest_neighbor_spacing(centers: np.ndarray) -> float:
    if len(centers) < 2:
        raise ValueError("at least two markers are required to estimate spacing")
    nearest: list[float] = []
    for idx, point in enumerate(centers):
        distances = np.linalg.norm(centers - point, axis=1)
        distances = distances[distances > 1e-6]
        if len(distances):
            nearest.append(float(np.min(distances)))
    if not nearest:
        raise ValueError("marker centers are degenerate")
    return float(np.median(nearest))


def estimate_pairwise_axis(centers: np.ndarray) -> tuple[float, float, int]:
    """Estimate tray axis from near-neighbor center vectors.

    The angle has a 90-degree period because horizontal and vertical grid
    vectors describe the same tray orientation. This avoids using individual
    marker corner angles.
    """
    spacing = nearest_neighbor_spacing(centers)
    pair_angles: list[float] = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            vector = centers[j] - centers[i]
            distance = float(np.linalg.norm(vector))
            if 0.65 * spacing <= distance <= 1.45 * spacing:
                pair_angles.append(math.atan2(float(vector[1]), float(vector[0])))
    if not pair_angles:
        raise ValueError("no near-neighbor marker pairs found")
    theta = normalize_axis_angle(circular_mean(pair_angles, math.pi / 2.0))
    return theta, spacing, len(pair_angles)


def cluster_1d(values: np.ndarray, gap: float) -> list[dict[str, Any]]:
    order = np.argsort(values)
    clusters: list[dict[str, Any]] = []
    for raw_idx in order:
        idx = int(raw_idx)
        value = float(values[idx])
        if not clusters or value - float(clusters[-1]["max"]) > gap:
            clusters.append({"indices": [idx], "min": value, "max": value})
        else:
            clusters[-1]["indices"].append(idx)
            clusters[-1]["max"] = value
    for cluster in clusters:
        cluster["center"] = float(np.mean([values[i] for i in cluster["indices"]]))
    return clusters


def candidate_spacing_penalty(clusters: list[dict[str, Any]], start: int, count: int) -> float:
    centers = [float(clusters[i]["center"]) for i in range(start, start + count)]
    if len(centers) < 3:
        return 0.0
    gaps = np.diff(centers)
    median_gap = float(np.median(gaps))
    if median_gap <= 1e-6:
        return 0.0
    return float(np.mean(np.abs(gaps - median_gap)) / median_gap)


def choose_slot_window(
    markers: list[Marker],
    u_clusters: list[dict[str, Any]],
    v_clusters: list[dict[str, Any]],
    rows: int,
    cols: int,
) -> tuple[int, int]:
    if len(u_clusters) < cols or len(v_clusters) < rows:
        raise ValueError(
            f"not enough projected clusters for a {rows}x{cols} grid: "
            f"{len(v_clusters)} rows, {len(u_clusters)} columns"
        )

    best_score: float | None = None
    best_start = (0, 0)
    for u_start in range(len(u_clusters) - cols + 1):
        for v_start in range(len(v_clusters) - rows + 1):
            occupied: set[tuple[int, int]] = set()
            count = 0
            for marker in markers:
                col = marker.u_cluster - u_start
                row = marker.v_cluster - v_start
                if 0 <= row < rows and 0 <= col < cols:
                    occupied.add((row, col))
                    count += 1
            u_penalty = candidate_spacing_penalty(u_clusters, u_start, cols)
            v_penalty = candidate_spacing_penalty(v_clusters, v_start, rows)
            score = len(occupied) * 1000.0 + count * 10.0 - (u_penalty + v_penalty) * 100.0
            if best_score is None or score > best_score:
                best_score = score
                best_start = (u_start, v_start)
    return best_start


def pca_line_angle(points: np.ndarray, target_angle: float) -> float:
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    angle = math.atan2(float(vh[0, 1]), float(vh[0, 0]))
    if math.cos(angle - target_angle) < 0:
        angle += math.pi
    while angle - target_angle > math.pi / 2.0:
        angle -= math.pi
    while angle - target_angle < -math.pi / 2.0:
        angle += math.pi
    return angle


def estimate_grid_line_angle(
    slot_grid: list[list[Marker | None]],
    fallback_theta: float,
) -> tuple[float, list[float], list[float]]:
    row_angles: list[float] = []
    col_angles_as_rows: list[float] = []
    rows = len(slot_grid)
    cols = len(slot_grid[0]) if rows else 0

    for row in slot_grid:
        points = np.array([marker.center for marker in row if marker is not None], dtype=float)
        if len(points) >= 2:
            row_angles.append(pca_line_angle(points, fallback_theta))

    for col_idx in range(cols):
        points = np.array(
            [slot_grid[row_idx][col_idx].center for row_idx in range(rows) if slot_grid[row_idx][col_idx] is not None],
            dtype=float,
        )
        if len(points) >= 2:
            col_angle = pca_line_angle(points, fallback_theta + math.pi / 2.0)
            col_angles_as_rows.append(col_angle - math.pi / 2.0)

    usable = row_angles + col_angles_as_rows
    if not usable:
        return fallback_theta, row_angles, col_angles_as_rows
    angle = normalize_axis_angle(circular_mean(usable, math.pi))
    return angle, row_angles, col_angles_as_rows


def fit_affine_from_slots(slot_grid: list[list[Marker | None]]) -> list[list[float]] | None:
    image_points: list[np.ndarray] = []
    grid_points: list[list[float]] = []
    for row_idx, row in enumerate(slot_grid):
        for col_idx, marker in enumerate(row):
            if marker is not None:
                grid_points.append([float(col_idx), float(row_idx), 1.0])
                image_points.append(marker.center)
    if len(grid_points) < 3:
        return None
    a = np.array(grid_points, dtype=float)
    b = np.array(image_points, dtype=float)
    transform, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    return transform.tolist()


def apply_affine(transform: list[list[float]], col: float, row: float) -> tuple[float, float]:
    matrix = np.array(transform, dtype=float)
    point = np.array([float(col), float(row), 1.0], dtype=float) @ matrix
    return float(point[0]), float(point[1])


def apply_point_transform(transform: list[list[float]] | np.ndarray, point: tuple[float, float] | list[float]) -> tuple[float, float]:
    matrix = np.array(transform, dtype=float)
    if matrix.shape == (2, 3):
        x, y = float(point[0]), float(point[1])
        out = matrix @ np.array([x, y, 1.0], dtype=float)
        return float(out[0]), float(out[1])
    if matrix.shape == (3, 3):
        x, y = float(point[0]), float(point[1])
        out = matrix @ np.array([x, y, 1.0], dtype=float)
        if abs(float(out[2])) > 1e-9:
            out = out / float(out[2])
        return float(out[0]), float(out[1])
    raise ValueError(f"unsupported point transform shape {matrix.shape}")


def fit_grid_affine_from_points(points_by_slot: dict[tuple[int, int], tuple[float, float] | list[float]]) -> list[list[float]] | None:
    if len(points_by_slot) < 3:
        return None
    grid_points: list[list[float]] = []
    image_points: list[tuple[float, float] | list[float]] = []
    for (row_idx, col_idx), point in points_by_slot.items():
        grid_points.append([float(col_idx), float(row_idx), 1.0])
        image_points.append(point)
    a = np.array(grid_points, dtype=float)
    b = np.array(image_points, dtype=float)
    transform, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    return transform.tolist()


def layout_marker_reference_centers(layout: dict[str, Any] | None) -> dict[int, list[float]]:
    if not layout:
        return {}
    raw = layout.get("marker_reference_centers_px") or layout.get("reference_marker_centers_px") or {}
    centers: dict[int, list[float]] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                centers[int(key)] = [float(value[0]), float(value[1])]
            except Exception:
                continue
    return centers


def layout_slot_reference_centers(layout: dict[str, Any] | None) -> dict[tuple[int, int], list[float]]:
    if not layout:
        return {}
    raw = layout.get("slot_reference_centers_px") or []
    centers: dict[tuple[int, int], list[float]] = {}
    if isinstance(raw, list):
        for row_idx, row in enumerate(raw):
            if not isinstance(row, list):
                continue
            for col_idx, value in enumerate(row):
                if value is None:
                    continue
                try:
                    centers[(row_idx, col_idx)] = [float(value[0]), float(value[1])]
                except Exception:
                    continue
    return centers


def estimate_reference_transform(
    layout: dict[str, Any] | None,
    marker_by_id: dict[int, Marker],
) -> tuple[list[list[float]] | None, list[int]]:
    reference_centers = layout_marker_reference_centers(layout)
    if not reference_centers:
        return None, []
    source: list[list[float]] = []
    target: list[list[float]] = []
    used_ids: list[int] = []
    for marker_id, marker in marker_by_id.items():
        if marker_id in reference_centers:
            source.append(reference_centers[marker_id])
            target.append(marker.center.astype(float).tolist())
            used_ids.append(marker_id)
    if len(source) < 2:
        return None, used_ids
    src = np.array(source, dtype=np.float32)
    dst = np.array(target, dtype=np.float32)
    if len(source) == 2:
        src_vec = src[1] - src[0]
        dst_vec = dst[1] - dst[0]
        src_len = float(np.linalg.norm(src_vec))
        dst_len = float(np.linalg.norm(dst_vec))
        if src_len <= 1e-6 or dst_len <= 1e-6:
            return None, used_ids
        angle = math.atan2(float(dst_vec[1]), float(dst_vec[0])) - math.atan2(float(src_vec[1]), float(src_vec[0]))
        scale = dst_len / src_len
        cos_a = math.cos(angle) * scale
        sin_a = math.sin(angle) * scale
        matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=float)
        translation = dst[0].astype(float) - matrix @ src[0].astype(float)
        transform = np.array(
            [
                [matrix[0, 0], matrix[0, 1], translation[0]],
                [matrix[1, 0], matrix[1, 1], translation[1]],
            ],
            dtype=float,
        )
        return transform.tolist(), used_ids
    transform, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=8.0)
    if transform is None:
        transform, inliers = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=8.0)
    if transform is None:
        return None, used_ids
    if inliers is not None:
        used_ids = [marker_id for marker_id, keep in zip(used_ids, inliers.ravel()) if int(keep)]
    return transform.astype(float).tolist(), used_ids


def transformed_slot_reference_points(
    layout: dict[str, Any] | None,
    reference_transform: list[list[float]] | None,
) -> dict[tuple[int, int], tuple[float, float]]:
    if reference_transform is None:
        return {}
    transformed: dict[tuple[int, int], tuple[float, float]] = {}
    for slot_key, point in layout_slot_reference_centers(layout).items():
        transformed[slot_key] = apply_point_transform(reference_transform, point)
    return transformed


def layout_reference_grid_spacing(layout: dict[str, Any] | None) -> float | None:
    centers = layout_slot_reference_centers(layout)
    if not centers:
        return None
    distances: list[float] = []
    for (row_idx, col_idx), point in centers.items():
        right = centers.get((row_idx, col_idx + 1))
        down = centers.get((row_idx + 1, col_idx))
        for neighbor in (right, down):
            if neighbor is None:
                continue
            distances.append(math.hypot(float(point[0]) - float(neighbor[0]), float(point[1]) - float(neighbor[1])))
    if not distances:
        return None
    return float(np.median(np.array(distances, dtype=float)))


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"could not encode image as {ext}: {path}")
    encoded.tofile(str(path))


def normalize_angle_180_deg(angle: float) -> float:
    angle = angle % 180.0
    if angle < 0:
        angle += 180.0
    return angle


def rect_angle_from_min_area_rect(rect: tuple) -> float:
    (_, _), (width, height), angle = rect
    if width < height:
        angle += 90.0
    return normalize_angle_180_deg(angle)


def square_box_from_center_angle(
    center: tuple[float, float] | list[float],
    side: float,
    angle_deg: float,
) -> list[list[float]]:
    cx, cy = float(center[0]), float(center[1])
    half_side = max(1.0, float(side)) / 2.0
    theta = math.radians(float(angle_deg))
    ux = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    uy = np.array([-math.sin(theta), math.cos(theta)], dtype=float)
    points = [
        np.array([cx, cy]) - ux * half_side - uy * half_side,
        np.array([cx, cy]) + ux * half_side - uy * half_side,
        np.array([cx, cy]) + ux * half_side + uy * half_side,
        np.array([cx, cy]) - ux * half_side + uy * half_side,
    ]
    return [point.astype(float).tolist() for point in points]


def square_side_from_rect_and_area(width: float, height: float, area: float) -> float:
    area_side = math.sqrt(max(1.0, float(area)))
    rect_side = 0.5 * (float(width) + float(height))
    return float(np.median([area_side, rect_side, rect_side]))


def circular_mean_deg(angles_deg: list[float], weights: list[float] | np.ndarray | None = None) -> float | None:
    if not angles_deg:
        return None
    angles = np.deg2rad(np.array(angles_deg, dtype=np.float64) * 2.0)
    if weights is None:
        weights_array = np.ones(len(angles), dtype=np.float64)
    else:
        weights_array = np.array(weights, dtype=np.float64)
    x = np.sum(weights_array * np.cos(angles))
    y = np.sum(weights_array * np.sin(angles))
    return normalize_angle_180_deg(0.5 * np.rad2deg(np.arctan2(y, x)))


def angle_delta_period_deg(angle: float, reference: float, period: float = 90.0) -> float:
    return ((float(angle) - float(reference) + period / 2.0) % period) - period / 2.0


def equivalent_angle_near_reference_deg(angle: float, reference: float, period: float = 90.0) -> float:
    return float(reference) + angle_delta_period_deg(angle, reference, period)


def weighted_circular_mean_period_deg(values_deg: list[float], weights: list[float], period: float) -> float:
    if not values_deg:
        raise ValueError("cannot average an empty angle list")
    scale = 2.0 * math.pi / float(period)
    values = np.deg2rad(np.array(values_deg, dtype=np.float64))
    weight_array = np.array(weights, dtype=np.float64)
    x = float(np.sum(weight_array * np.cos(values * scale)))
    y = float(np.sum(weight_array * np.sin(values * scale)))
    return float(np.rad2deg(math.atan2(y, x) / scale))


def square_angle_from_edges(
    rect_angle_deg: float,
    texture_angle_deg: float | None,
    texture_confidence: float,
    reference_angle_deg: float,
    edge_angle_deg: float | None = None,
    edge_confidence: float = 0.0,
) -> tuple[float, float, str]:
    angles = [float(rect_angle_deg)]
    weights = [1.0]
    source = "square_edge"

    trusted_edge_angle = edge_angle_deg is not None and float(edge_confidence) >= 2.0
    if trusted_edge_angle:
        angles.append(float(edge_angle_deg))
        weights.append(min(3.2, 1.2 + float(edge_confidence) / 3.0))
        source = "slot_boundary"

    if texture_angle_deg is not None and float(texture_confidence) >= 3.0:
        texture_delta_from_rect = abs(angle_delta_period_deg(texture_angle_deg, rect_angle_deg, period=90.0))
        texture_consistent_with_edge = True
        if trusted_edge_angle:
            texture_consistent_with_edge = abs(angle_delta_period_deg(texture_angle_deg, edge_angle_deg, period=90.0)) <= 12.0
        if texture_delta_from_rect <= 18.0 and texture_consistent_with_edge:
            angles.append(float(texture_angle_deg))
            weights.append(min(1.2, 0.4 + float(texture_confidence) / 24.0))
            source = f"{source}+texture"

    rel_values = [angle_delta_period_deg(angle, reference_angle_deg, period=90.0) for angle in angles]
    rel_angle = weighted_circular_mean_period_deg(rel_values, weights, period=90.0)
    rel_angle = angle_delta_period_deg(rel_angle, 0.0, period=90.0)
    return float(reference_angle_deg) + rel_angle, rel_angle, source


def apply_square_angle_model(flakes: list[dict[str, Any]], reference_angle_deg: float) -> None:
    for flake in flakes:
        rect_angle = float(flake.get("rect_angle_deg", flake.get("final_angle_deg", reference_angle_deg)))
        texture_angle = flake.get("texture_angle_deg")
        texture_confidence = float(flake.get("texture_confidence", 0.0))
        edge_angle = flake.get("edge_angle_deg")
        edge_confidence = float(flake.get("edge_angle_confidence", 0.0))
        raw_final = float(flake.get("final_angle_deg", rect_angle))
        square_angle, relative_angle, source = square_angle_from_edges(
            rect_angle,
            float(texture_angle) if texture_angle is not None else None,
            texture_confidence,
            reference_angle_deg,
            float(edge_angle) if edge_angle is not None else None,
            edge_confidence,
        )
        width, height = flake.get("rect_size_px", [0.0, 0.0])
        side = float(flake.get("square_side_px") or square_side_from_rect_and_area(float(width), float(height), float(flake["area"])))
        square_box = square_box_from_center_angle(flake["center_px"], side, square_angle)
        flake["raw_final_angle_deg"] = raw_final
        flake["square_angle_deg"] = float(square_angle)
        flake["angle_relative_to_tray_deg"] = float(relative_angle)
        flake["final_angle_deg"] = float(square_angle)
        flake["angle_source"] = source
        flake["square_side_px"] = side
        flake["square_box_px"] = square_box
        flake["box_px"] = square_box


def update_chip_quality(
    flake: dict[str, Any],
    *,
    spacing_px: float,
    slot_center: tuple[float, float] | list[float] | None = None,
    image_shape: tuple[int, ...] | None = None,
) -> None:
    width, height = [float(x) for x in flake.get("rect_size_px", [0.0, 0.0])]
    short_side = max(1.0, min(width, height))
    long_side = max(width, height)
    aspect = long_side / short_side
    side = float(flake.get("square_side_px") or square_side_from_rect_and_area(width, height, float(flake.get("area", 0.0))))
    side_ratio = side / max(1.0, float(spacing_px))
    fill_ratio = float(flake.get("area", 0.0)) / max(1.0, side * side)
    rel_angle = float(flake.get("angle_relative_to_tray_deg", 0.0))
    center_offset_ratio = 0.0
    if slot_center is not None:
        fx, fy = flake["center_px"]
        center_offset_ratio = math.hypot(float(fx) - float(slot_center[0]), float(fy) - float(slot_center[1])) / max(1.0, float(spacing_px))

    flags: list[str] = []
    status = "ok"
    partial_view = bool(flake.get("partial_chip_view", False))
    severe_partial_view = bool(flake.get("chip_occlusion_overlap", False))
    if image_shape is not None and slot_center is not None:
        image_h, image_w = image_shape[:2]
        margin = max(8.0, DEFAULT_CHIP_PARTIAL_EDGE_RATIO * float(spacing_px))
        severe_margin = max(8.0, 0.30 * float(spacing_px))
        sx, sy = float(slot_center[0]), float(slot_center[1])
        center_near_edge = sx < margin or sy < margin or sx > float(image_w) - margin or sy > float(image_h) - margin
        severe_partial_view = severe_partial_view or (
            sx < severe_margin
            or sy < severe_margin
            or sx > float(image_w) - severe_margin
            or sy > float(image_h) - severe_margin
        )
        partial_view = partial_view or center_near_edge
        square_box = flake.get("square_box_px") or flake.get("box_px")
        if square_box is not None:
            partial_view = partial_view or square_box_near_or_outside_image(square_box, image_shape, margin=3.0)

    solidity = float(flake.get("chip_solidity", 1.0))
    rectangularity = float(flake.get("chip_rectangularity", fill_ratio))
    vertex_count = int(flake.get("chip_polygon_vertices", 4))
    long_segment_count = int(flake.get("chip_long_segment_count", vertex_count))
    component_count = int(round(float(flake.get("component_count", 1.0))))
    second_component_ratio = float(flake.get("second_component_area_ratio", 0.0))
    internal_line_count = int(round(float(flake.get("internal_line_count", 0.0))))
    internal_line_score = float(flake.get("internal_line_score", 0.0))
    internal_any_line_count = int(round(float(flake.get("internal_any_line_count", 0.0))))
    internal_any_line_score = float(flake.get("internal_any_line_score", 0.0))
    internal_oblique_line_count = int(round(float(flake.get("internal_oblique_line_count", 0.0))))
    internal_oblique_line_score = float(flake.get("internal_oblique_line_score", 0.0))
    silver_gap_count = int(round(float(flake.get("silver_gap_blocked_count", 0.0))))
    silver_gap_score = float(flake.get("silver_gap_blocked_score", 0.0))
    geometry_reliable = float(spacing_px) >= 80.0 and side >= 55.0
    stack_size_plausible = side_ratio >= 0.780
    strong_silver_gap_evidence = (
        silver_gap_count >= 2
        and silver_gap_score >= 0.80
        and side_ratio >= 0.700
        and center_offset_ratio >= 0.075
    )
    silver_gap_evidence = (
        geometry_reliable
        and (side_ratio >= DEFAULT_SILVER_GAP_MIN_CHIP_SIDE_RATIO or strong_silver_gap_evidence)
        and silver_gap_count >= 1
        and silver_gap_score >= DEFAULT_SILVER_GAP_MIN_COVERAGE_RATIO
    )

    oversized_low_fill = side_ratio > DEFAULT_CHIP_STACKED_SIDE_RATIO and fill_ratio < DEFAULT_CHIP_STACKED_MAX_FILL_RATIO
    component_evidence = geometry_reliable and component_count >= 2 and second_component_ratio >= 0.16
    cross_stack_edge = geometry_reliable and stack_size_plausible and (
        internal_oblique_line_count >= 2
        and internal_oblique_line_score >= 0.85
        and internal_any_line_score >= 1.80
        and (
            (not partial_view and (rectangularity >= 0.875 or side_ratio >= 0.820))
            or (partial_view and bool(flake.get("chip_occlusion_overlap", False)))
            or (partial_view and rectangularity < 0.845 and side_ratio >= 0.820)
        )
    )
    mild_partial_stack_outline = stack_size_plausible and (
        partial_view
        and not severe_partial_view
        and rectangularity < 0.845
        and vertex_count >= 8
        and fill_ratio > 0.800
    )
    outline_context_evidence = (
        aspect > 1.18
        or center_offset_ratio > 0.100
        or internal_any_line_score >= 1.50
        or internal_oblique_line_count >= 2
        or silver_gap_evidence
        or component_evidence
    )
    complex_outline = geometry_reliable and stack_size_plausible and (
        (
            rectangularity < 0.835 and vertex_count >= 8 and outline_context_evidence
        ) or (
            rectangularity < 0.790 and vertex_count >= 6 and (outline_context_evidence or solidity < 0.860)
        ) or (
            vertex_count >= 11 and rectangularity < 0.870 and solidity < 0.950 and outline_context_evidence
        ) or mild_partial_stack_outline
    )
    compressed_complex_outline = geometry_reliable and side_ratio >= 0.700 and (
        rectangularity < 0.825
        and solidity < 0.910
        and vertex_count >= 8
        and (aspect > 1.18 or center_offset_ratio > 0.100 or long_segment_count >= 6)
    )
    concave_outline = (
        geometry_reliable
        and stack_size_plausible
        and solidity < 0.900
        and rectangularity < 0.820
        and vertex_count >= 6
        and (outline_context_evidence or rectangularity < 0.760)
    )
    internal_stack_edge = geometry_reliable and stack_size_plausible and (
        internal_line_count >= 2
        and internal_line_score >= 0.80
        and (component_evidence or complex_outline or rectangularity < 0.805 or (vertex_count >= 9 and solidity < 0.910))
    )
    if partial_view and rectangularity < 0.805 and internal_line_count == 0 and not component_evidence and not oversized_low_fill:
        severe_partial_view = True
    if fill_ratio < 0.600 and side_ratio < 0.600 and center_offset_ratio > 0.200:
        partial_view = True
        severe_partial_view = True
    stacked_like = oversized_low_fill or component_evidence or complex_outline or compressed_complex_outline or concave_outline or internal_stack_edge or cross_stack_edge

    can_call_geometric_error = geometry_reliable and (not severe_partial_view or cross_stack_edge or silver_gap_evidence)
    if can_call_geometric_error and stacked_like:
        flags.append("stacked_or_non_square")
        if oversized_low_fill:
            flags.append("oversized_low_fill")
        if component_evidence:
            flags.append("multiple_chip_regions")
        if complex_outline:
            flags.append("complex_outline")
        if compressed_complex_outline and "complex_outline" not in flags:
            flags.append("complex_outline")
        if concave_outline:
            flags.append("concave_outline")
        if internal_stack_edge:
            flags.append("internal_stack_edge")
        if cross_stack_edge:
            flags.append("internal_cross_edge")
        status = "abnormal"
    if silver_gap_evidence:
        if "silver_gap_blocked" not in flags:
            flags.append("silver_gap_blocked")
        if "off_slot_cover" not in flags:
            flags.append("off_slot_cover")
        status = "abnormal"
    elif status != "abnormal" and can_call_geometric_error and (
        aspect > DEFAULT_CHIP_MAX_ASPECT_WARN
        or side_ratio > 1.05
        or side_ratio < 0.42
    ):
        flags.append("stacked_or_non_square")
        status = "abnormal"
    elif status != "abnormal" and can_call_geometric_error and (aspect > DEFAULT_CHIP_MAX_ASPECT_OK or side_ratio > DEFAULT_CHIP_MAX_SIDE_RATIO or side_ratio < DEFAULT_CHIP_MIN_SIDE_RATIO):
        flags.append("not_square_enough")
        status = "warning"
    if can_call_geometric_error and abs(rel_angle) > DEFAULT_CHIP_MAX_ANGLE_REL_OK:
        flags.append("crooked")
        if status == "ok":
            status = "warning"
    if center_offset_ratio > DEFAULT_CHIP_MAX_CENTER_OFFSET_RATIO:
        flags.append("off_center")
        if status == "ok":
            status = "warning"
    if can_call_geometric_error and (fill_ratio < 0.55 or fill_ratio > 1.22):
        flags.append("bad_square_fill")
        status = "abnormal"
    elif can_call_geometric_error and fill_ratio < 0.66 and status == "ok":
        flags.append("low_square_fill")
        status = "warning"
    if partial_view and status != "abnormal":
        flags = [flag for flag in flags if flag not in {"stacked_or_non_square", "not_square_enough", "crooked", "bad_square_fill"}]
        if "partial_view" not in flags:
            flags.append("partial_view")
        status = "warning"
    elif partial_view and "partial_view" not in flags:
        flags.append("partial_view")

    flake["chip_status"] = status
    flake["chip_flags"] = flags
    flake["shape_aspect_ratio"] = float(aspect)
    flake["square_fill_ratio"] = float(fill_ratio)
    flake["side_to_grid_ratio"] = float(side_ratio)
    flake["center_offset_grid_ratio"] = float(center_offset_ratio)


def is_marker_artifact_like(flake: dict[str, Any]) -> bool:
    """Reject missed ArUco/QR patterns before they become occupied chips."""
    purple_ratio = float(flake.get("purple_pixel_ratio", 0.0))
    white_ratio = float(flake.get("white_pixel_ratio", 0.0))
    colorful_ratio = float(flake.get("colorful_pixel_ratio", 0.0))
    fill_ratio = float(flake.get("square_fill_ratio", 1.0))
    side_ratio = float(flake.get("side_to_grid_ratio", 0.0))
    aspect = float(flake.get("shape_aspect_ratio", flake.get("aspect_ratio", 1.0)))

    marker_has_black_white_pattern = purple_ratio < 0.08 and white_ratio > 0.16 and fill_ratio < 0.74
    low_color_broken_shape = purple_ratio < 0.04 and colorful_ratio < 0.14 and fill_ratio < 0.70
    tiny_broken_shape = side_ratio < 0.38 and fill_ratio < 0.80
    long_marker_fragment = aspect > 1.70 and purple_ratio < 0.08 and white_ratio > 0.10
    return bool(marker_has_black_white_pattern or low_color_broken_shape or tiny_broken_shape or long_marker_fragment)


def texture_angle_from_roi(roi_bgr: np.ndarray) -> tuple[float | None, float]:
    if roi_bgr.size == 0:
        return None, 0.0
    if roi_bgr.ndim == 2:
        gray = roi_bgr
    else:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    min_len = max(20, min(roi_bgr.shape[:2]) // 4)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=30,
        minLineLength=min_len,
        maxLineGap=10,
    )
    if lines is None:
        return None, 0.0

    angles: list[float] = []
    weights: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = line.ravel()
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)
        if length < 10.0:
            continue
        angles.append(normalize_angle_180_deg(math.degrees(math.atan2(dy, dx))))
        weights.append(length)
    if not angles:
        return None, 0.0
    return circular_mean_deg(angles, weights), float(len(angles))


def pad_and_crop(img: np.ndarray, x: int, y: int, width: int, height: int, pad: int = 10) -> np.ndarray:
    image_h, image_w = img.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image_w, x + width + pad)
    y1 = min(image_h, y + height + pad)
    return img[y0:y1, x0:x1].copy()


def contour_points(contour: np.ndarray, offset: tuple[int, int] = (0, 0)) -> list[list[float]]:
    epsilon = max(1.0, 0.006 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(float)
    if offset != (0, 0):
        approx += np.array(offset, dtype=float)
    return approx.tolist()


def polygon_pixel_mask(image_shape: tuple[int, ...], polygon_points: list[list[float]] | np.ndarray) -> np.ndarray:
    image_h, image_w = image_shape[:2]
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    polygon = np.array(polygon_points, dtype=np.int32).reshape(-1, 2)
    if len(polygon) >= 3:
        cv2.fillPoly(mask, [polygon], 255, lineType=cv2.LINE_AA)
    return mask


def chip_color_features(
    image: np.ndarray,
    polygon_points: list[list[float]] | np.ndarray,
    *,
    lower_hsv: tuple[int, int, int] = DEFAULT_LOWER_PURPLE,
    upper_hsv: tuple[int, int, int] = DEFAULT_UPPER_PURPLE,
) -> dict[str, float]:
    bgr = as_bgr_image(image)
    mask = polygon_pixel_mask(bgr.shape, polygon_points)
    pixels = bgr[mask > 0]
    if pixels.size == 0:
        return {
            "purple_pixel_ratio": 0.0,
            "dark_pixel_ratio": 0.0,
            "white_pixel_ratio": 0.0,
            "colorful_pixel_ratio": 0.0,
            "median_hsv_h": 0.0,
            "median_hsv_s": 0.0,
            "median_hsv_v": 0.0,
        }
    hsv_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h = hsv_pixels[:, 0]
    s = hsv_pixels[:, 1]
    v = hsv_pixels[:, 2]
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    purple = (
        (h >= int(lower[0]))
        & (h <= int(upper[0]))
        & (s >= max(20, int(lower[1])))
        & (v >= max(20, int(lower[2])))
    )
    return {
        "purple_pixel_ratio": float(np.mean(purple)),
        "dark_pixel_ratio": float(np.mean(v < 82)),
        "white_pixel_ratio": float(np.mean((v > 128) & (s < 48))),
        "colorful_pixel_ratio": float(np.mean(s > 45)),
        "median_hsv_h": float(np.median(h)),
        "median_hsv_s": float(np.median(s)),
        "median_hsv_v": float(np.median(v)),
    }


def as_bgr_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def chip_candidate_mask_from_crop(
    crop_bgr: np.ndarray,
    *,
    lower_hsv: tuple[int, int, int] = DEFAULT_LOWER_PURPLE,
    upper_hsv: tuple[int, int, int] = DEFAULT_UPPER_PURPLE,
) -> np.ndarray:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    purple_mask = cv2.inRange(
        hsv,
        np.array(lower_hsv, dtype=np.uint8),
        np.array(upper_hsv, dtype=np.uint8),
    )
    dark_chip = (((val < 82) & ((sat > 14) | (val < 48))).astype(np.uint8)) * 255
    blue_purple_dark = (((hue >= 85) & (hue <= 158) & (sat > 22) & (val < 126)).astype(np.uint8)) * 255
    mask = cv2.bitwise_or(purple_mask, dark_chip)
    mask = cv2.bitwise_or(mask, blue_purple_dark)

    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=1)
    return mask


def connected_component_features(
    mask: np.ndarray,
    *,
    spacing_px: float,
    center_local: tuple[float, float],
) -> tuple[np.ndarray, dict[str, float]]:
    min_component_area = max(35.0, 0.012 * float(spacing_px) * float(spacing_px))
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    components: list[dict[str, Any]] = []
    for label in range(1, n_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        ccx, ccy = float(centroids[label][0]), float(centroids[label][1])
        distance = math.hypot(ccx - float(center_local[0]), ccy - float(center_local[1]))
        components.append(
            {
                "label": int(label),
                "area": area,
                "distance": distance,
                "bbox": stats[label].astype(int).tolist(),
            }
        )
    if not components:
        return np.zeros(mask.shape, dtype=np.uint8), {
            "component_count": 0.0,
            "selected_component_count": 0.0,
            "second_component_area_ratio": 0.0,
        }

    components.sort(key=lambda item: (item["area"] - item["distance"] * float(spacing_px) * 0.08), reverse=True)
    largest_area = max(item["area"] for item in components)
    substantial = [item for item in components if item["area"] >= max(min_component_area, 0.10 * largest_area)]
    second_ratio = 0.0
    if len(substantial) > 1:
        sorted_areas = sorted((item["area"] for item in substantial), reverse=True)
        second_ratio = float(sorted_areas[1] / max(1.0, sorted_areas[0]))

    selected_labels: set[int] = {int(components[0]["label"])}
    for item in components[1:]:
        close_to_slot = item["distance"] <= 0.42 * float(spacing_px)
        large_enough = item["area"] >= max(min_component_area, 0.16 * largest_area)
        if close_to_slot and large_enough:
            selected_labels.add(int(item["label"]))

    selected_mask = np.zeros(mask.shape, dtype=np.uint8)
    for label in selected_labels:
        selected_mask[labels == label] = 255
    selected_mask = cv2.morphologyEx(selected_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return selected_mask, {
        "component_count": float(len(substantial)),
        "selected_component_count": float(len(selected_labels)),
        "second_component_area_ratio": float(second_ratio),
    }


def contour_edge_angle_from_mask(
    mask: np.ndarray,
    contour: np.ndarray,
    *,
    reference_angle_deg: float,
    spacing_px: float,
) -> tuple[float | None, float]:
    angles: list[float] = []
    weights: list[float] = []
    perimeter = cv2.arcLength(contour, True)
    if perimeter > 1.0:
        approx = cv2.approxPolyDP(contour, max(1.0, 0.010 * perimeter), True).reshape(-1, 2)
        for p0, p1 in zip(approx, np.roll(approx, -1, axis=0)):
            dx = float(p1[0] - p0[0])
            dy = float(p1[1] - p0[1])
            length = math.hypot(dx, dy)
            if length < max(14.0, 0.14 * float(spacing_px)):
                continue
            angles.append(normalize_angle_180_deg(math.degrees(math.atan2(dy, dx))))
            weights.append(length)

    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    lines = cv2.HoughLinesP(
        boundary,
        rho=1,
        theta=np.pi / 180.0,
        threshold=12,
        minLineLength=max(18, int(0.16 * float(spacing_px))),
        maxLineGap=6,
    )
    if lines is not None:
        for line in lines:
            x0, y0, x1, y1 = line.ravel()
            dx = float(x1 - x0)
            dy = float(y1 - y0)
            length = math.hypot(dx, dy)
            if length < max(18.0, 0.16 * float(spacing_px)):
                continue
            angles.append(normalize_angle_180_deg(math.degrees(math.atan2(dy, dx))))
            weights.append(0.55 * length)

    if not angles:
        return None, 0.0
    rel_angles = [angle_delta_period_deg(angle, reference_angle_deg, period=90.0) for angle in angles]
    rel_angle = weighted_circular_mean_period_deg(rel_angles, weights, period=90.0)
    rel_angle = angle_delta_period_deg(rel_angle, 0.0, period=90.0)
    confidence = min(12.0, float(sum(weights)) / max(1.0, float(spacing_px)))
    return float(reference_angle_deg) + float(rel_angle), confidence


def internal_line_features(
    crop_bgr: np.ndarray,
    object_mask: np.ndarray,
    *,
    center_local: tuple[float, float],
    angle_deg: float,
    side_px: float,
    spacing_px: float,
) -> dict[str, float]:
    if crop_bgr.size == 0 or object_mask.size == 0:
        return {"internal_line_count": 0.0, "internal_line_score": 0.0}
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 32, 104)
    interior = cv2.erode(object_mask, np.ones((5, 5), np.uint8), iterations=2)
    edges = cv2.bitwise_and(edges, interior)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=14,
        minLineLength=max(18, int(DEFAULT_CHIP_INTERNAL_LINE_MIN_RATIO * float(spacing_px))),
        maxLineGap=7,
    )
    if lines is None:
        return {"internal_line_count": 0.0, "internal_line_score": 0.0}

    theta = math.radians(float(angle_deg))
    ux = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    uy = np.array([-math.sin(theta), math.cos(theta)], dtype=float)
    center = np.array([float(center_local[0]), float(center_local[1])], dtype=float)
    half_side = max(1.0, float(side_px) / 2.0)
    internal_lengths: list[float] = []
    for line in lines:
        x0, y0, x1, y1 = line.ravel()
        dx = float(x1 - x0)
        dy = float(y1 - y0)
        length = math.hypot(dx, dy)
        if length < max(18.0, 0.24 * float(side_px)):
            continue
        angle = normalize_angle_180_deg(math.degrees(math.atan2(dy, dx)))
        aligned_delta = abs(angle_delta_period_deg(angle, angle_deg, period=90.0))
        if aligned_delta > 16.0:
            continue
        midpoint = np.array([(float(x0) + float(x1)) / 2.0, (float(y0) + float(y1)) / 2.0], dtype=float) - center
        u = float(np.dot(midpoint, ux)) / half_side
        v = float(np.dot(midpoint, uy)) / half_side
        if max(abs(u), abs(v)) >= 0.74:
            continue
        internal_lengths.append(length)
    return {
        "internal_line_count": float(len(internal_lengths)),
        "internal_line_score": float(sum(internal_lengths) / max(1.0, float(side_px))),
    }


def internal_cross_line_features(
    crop_bgr: np.ndarray,
    object_mask: np.ndarray,
    *,
    center_local: tuple[float, float],
    angle_deg: float,
    side_px: float,
    spacing_px: float,
) -> dict[str, float]:
    if crop_bgr.size == 0 or object_mask.size == 0:
        return {
            "internal_any_line_count": 0.0,
            "internal_any_line_score": 0.0,
            "internal_oblique_line_count": 0.0,
            "internal_oblique_line_score": 0.0,
        }
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 26, 92)
    interior = cv2.erode(object_mask, np.ones((5, 5), np.uint8), iterations=1)
    edges = cv2.bitwise_and(edges, interior)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=10,
        minLineLength=max(18, int(0.18 * float(spacing_px))),
        maxLineGap=9,
    )
    if lines is None:
        return {
            "internal_any_line_count": 0.0,
            "internal_any_line_score": 0.0,
            "internal_oblique_line_count": 0.0,
            "internal_oblique_line_score": 0.0,
        }

    theta = math.radians(float(angle_deg))
    ux = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    uy = np.array([-math.sin(theta), math.cos(theta)], dtype=float)
    center = np.array([float(center_local[0]), float(center_local[1])], dtype=float)
    half_side = max(1.0, float(side_px) / 2.0)
    all_lengths: list[float] = []
    oblique_lengths: list[float] = []
    for line in lines:
        x0, y0, x1, y1 = line.ravel()
        dx = float(x1 - x0)
        dy = float(y1 - y0)
        length = math.hypot(dx, dy)
        if length < max(18.0, 0.20 * float(side_px)):
            continue
        midpoint = np.array([(float(x0) + float(x1)) / 2.0, (float(y0) + float(y1)) / 2.0], dtype=float) - center
        u = float(np.dot(midpoint, ux)) / half_side
        v = float(np.dot(midpoint, uy)) / half_side
        if max(abs(u), abs(v)) > 0.82:
            continue
        angle = normalize_angle_180_deg(math.degrees(math.atan2(dy, dx)))
        aligned_delta = abs(angle_delta_period_deg(angle, angle_deg, period=90.0))
        all_lengths.append(length)
        if 20.0 <= aligned_delta <= 70.0:
            oblique_lengths.append(length)
    return {
        "internal_any_line_count": float(len(all_lengths)),
        "internal_any_line_score": float(sum(all_lengths) / max(1.0, float(side_px))),
        "internal_oblique_line_count": float(len(oblique_lengths)),
        "internal_oblique_line_score": float(sum(oblique_lengths) / max(1.0, float(side_px))),
    }


def square_box_near_or_outside_image(square_box: list[list[float]], image_shape: tuple[int, ...], margin: float) -> bool:
    image_h, image_w = image_shape[:2]
    for point in square_box:
        x, y = float(point[0]), float(point[1])
        if x < margin or y < margin or x > float(image_w) - margin or y > float(image_h) - margin:
            return True
    return False


def refine_flake_with_slot_context(
    flake: dict[str, Any],
    image: np.ndarray,
    *,
    slot_center: tuple[float, float] | list[float] | None,
    slot_polygon: list[list[float]] | None,
    spacing_px: float,
    reference_angle_deg: float,
    lower_hsv: tuple[int, int, int] = DEFAULT_LOWER_PURPLE,
    upper_hsv: tuple[int, int, int] = DEFAULT_UPPER_PURPLE,
) -> bool:
    if slot_center is None:
        return False
    bgr = as_bgr_image(image)
    image_h, image_w = bgr.shape[:2]
    cx, cy = float(slot_center[0]), float(slot_center[1])
    radius = int(max(42.0, min(190.0, DEFAULT_CHIP_REFINEMENT_RADIUS_RATIO * float(spacing_px))))
    x0 = max(0, int(round(cx)) - radius)
    y0 = max(0, int(round(cy)) - radius)
    x1 = min(image_w, int(round(cx)) + radius)
    y1 = min(image_h, int(round(cy)) + radius)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return False

    crop = bgr[y0:y1, x0:x1]
    mask = chip_candidate_mask_from_crop(crop, lower_hsv=lower_hsv, upper_hsv=upper_hsv)
    if slot_polygon is not None:
        local_polygon = np.array(slot_polygon, dtype=np.float32).reshape(-1, 2) - np.array([x0, y0], dtype=np.float32)
        slot_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(slot_mask, [np.round(local_polygon).astype(np.int32)], 255, lineType=cv2.LINE_AA)
        mask = cv2.bitwise_and(mask, slot_mask)

    center_local = (cx - float(x0), cy - float(y0))
    object_mask, component_features = connected_component_features(mask, spacing_px=spacing_px, center_local=center_local)
    outline_kernel_size = int(max(5.0, min(17.0, round(0.070 * float(spacing_px)))))
    if outline_kernel_size % 2 == 0:
        outline_kernel_size += 1
    outline_kernel = np.ones((outline_kernel_size, outline_kernel_size), np.uint8)
    outline_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, outline_kernel, iterations=1)
    outline_mask = cv2.morphologyEx(outline_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(outline_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    if contour_area < max(45.0, 0.018 * float(spacing_px) * float(spacing_px)):
        return False

    rect = cv2.minAreaRect(contour)
    (local_x, local_y), (width, height), _ = rect
    rect_angle = rect_angle_from_min_area_rect(rect)
    edge_angle, edge_confidence = contour_edge_angle_from_mask(
        outline_mask,
        contour,
        reference_angle_deg=reference_angle_deg,
        spacing_px=spacing_px,
    )
    preliminary_angle = float(edge_angle) if edge_angle is not None and edge_confidence >= 2.0 else float(rect_angle)
    square_side = square_side_from_rect_and_area(float(width), float(height), contour_area)
    square_box = square_box_from_center_angle((local_x + x0, local_y + y0), square_side, preliminary_angle)

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    rect_area = max(1.0, float(width) * float(height))
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, max(1.0, 0.010 * perimeter), True).reshape(-1, 2)
    long_segment_count = 0
    for p0, p1 in zip(approx, np.roll(approx, -1, axis=0)):
        if math.hypot(float(p1[0] - p0[0]), float(p1[1] - p0[1])) >= 0.16 * float(spacing_px):
            long_segment_count += 1

    x, y, bw, bh = cv2.boundingRect(contour)
    local_partial = (
        (x <= 1 and x0 <= 1)
        or (y <= 1 and y0 <= 1)
        or (x + bw >= object_mask.shape[1] - 2 and x1 >= image_w - 1)
        or (y + bh >= object_mask.shape[0] - 2 and y1 >= image_h - 1)
        or square_box_near_or_outside_image(square_box, image.shape, margin=2.0)
    )
    line_features = internal_line_features(
        crop,
        object_mask,
        center_local=(float(local_x), float(local_y)),
        angle_deg=preliminary_angle,
        side_px=square_side,
        spacing_px=spacing_px,
    )
    cross_line_features = internal_cross_line_features(
        crop,
        object_mask,
        center_local=(float(local_x), float(local_y)),
        angle_deg=preliminary_angle,
        side_px=square_side,
        spacing_px=spacing_px,
    )
    contour_global = contour + np.array([[[x0, y0]]], dtype=contour.dtype)
    box_global = cv2.boxPoints(rect) + np.array([x0, y0], dtype=np.float32)
    color_features = chip_color_features(
        bgr,
        square_box,
        lower_hsv=lower_hsv,
        upper_hsv=upper_hsv,
    )

    flake.update(
        {
            "area": contour_area,
            "center_px": [float(local_x + x0), float(local_y + y0)],
            "rect_size_px": [float(width), float(height)],
            "aspect_ratio": float(max(width, height) / max(1.0, min(width, height))),
            "box_px": square_box,
            "square_box_px": square_box,
            "square_side_px": float(square_side),
            "raw_rect_box_px": box_global.astype(float).tolist(),
            "contour_px": contour_points(contour_global),
            "bounding_rect_px": [int(x + x0), int(y + y0), int(bw), int(bh)],
            "rect_angle_deg": float(rect_angle),
            "edge_angle_deg": float(edge_angle) if edge_angle is not None else None,
            "edge_angle_confidence": float(edge_confidence),
            "slot_refined": True,
            "chip_solidity": float(contour_area / max(1.0, hull_area)),
            "chip_rectangularity": float(contour_area / rect_area),
            "chip_polygon_vertices": int(len(approx)),
            "chip_long_segment_count": int(long_segment_count),
            "partial_chip_view": bool(local_partial),
            **component_features,
            **line_features,
            **cross_line_features,
            **color_features,
        }
    )
    return True


def detect_purple_flakes(
    image: np.ndarray,
    *,
    lower_hsv: tuple[int, int, int] = DEFAULT_LOWER_PURPLE,
    upper_hsv: tuple[int, int, int] = DEFAULT_UPPER_PURPLE,
    min_area: float = DEFAULT_FLAKE_MIN_AREA,
    max_aspect: float = DEFAULT_FLAKE_MAX_ASPECT,
    allowed_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    bgr = as_bgr_image(image)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(lower_hsv, dtype=np.uint8),
        np.array(upper_hsv, dtype=np.uint8),
    )
    if allowed_mask is not None:
        if allowed_mask.shape[:2] != mask.shape[:2]:
            raise ValueError("allowed_mask shape does not match image")
        mask = cv2.bitwise_and(mask, (allowed_mask > 0).astype(np.uint8) * 255)
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flakes: list[dict[str, Any]] = []
    for contour_idx, contour in enumerate(contours):
        area = float(cv2.contourArea(contour))
        if area < float(min_area):
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        (cx, cy), (width, height), _ = rect
        short_side = max(1.0, min(float(width), float(height)))
        long_side = max(float(width), float(height))
        aspect = long_side / short_side
        if aspect > float(max_aspect):
            continue
        rect_angle = rect_angle_from_min_area_rect(rect)

        x, y, bw, bh = cv2.boundingRect(contour)
        roi = pad_and_crop(bgr, x, y, bw, bh, pad=12)
        texture_angle, texture_confidence = texture_angle_from_roi(roi)
        if texture_angle is not None and texture_confidence >= 3.0:
            final_angle = texture_angle
            angle_source = "texture"
        else:
            final_angle = rect_angle
            angle_source = "rect"
        square_side = square_side_from_rect_and_area(width, height, area)
        square_box = square_box_from_center_angle((cx, cy), square_side, final_angle)
        color_features = chip_color_features(
            bgr,
            square_box,
            lower_hsv=lower_hsv,
            upper_hsv=upper_hsv,
        )

        flakes.append(
            {
                "idx": len(flakes),
                "contour_idx": int(contour_idx),
                "area": area,
                "center_px": [float(cx), float(cy)],
                "rect_size_px": [float(width), float(height)],
                "aspect_ratio": float(aspect),
                "box_px": square_box,
                "square_box_px": square_box,
                "square_side_px": float(square_side),
                "raw_rect_box_px": box.astype(float).tolist(),
                "contour_px": contour_points(contour),
                "bounding_rect_px": [int(x), int(y), int(bw), int(bh)],
                "rect_angle_deg": float(rect_angle),
                "texture_angle_deg": float(texture_angle) if texture_angle is not None else None,
                "texture_confidence": float(texture_confidence),
                "final_angle_deg": float(final_angle),
                "angle_source": angle_source,
                "kind": "purple",
                **color_features,
            }
        )
    flakes.sort(key=lambda item: item["area"], reverse=True)
    for idx, flake in enumerate(flakes):
        flake["idx"] = idx
    return flakes


def detect_slot_chip_candidate(
    image: np.ndarray,
    *,
    center: tuple[float, float] | list[float] | None,
    spacing_px: float,
    slot_key: tuple[int, int],
    mask_polygon_px: list[tuple[float, float]] | list[list[float]] | None = None,
    lower_hsv: tuple[int, int, int] = DEFAULT_LOWER_PURPLE,
    upper_hsv: tuple[int, int, int] = DEFAULT_UPPER_PURPLE,
    min_area: float = DEFAULT_FLAKE_MIN_AREA,
    max_aspect: float = DEFAULT_FLAKE_MAX_ASPECT,
) -> dict[str, Any] | None:
    if center is None:
        return None
    bgr = as_bgr_image(image)
    image_h, image_w = bgr.shape[:2]
    cx, cy = float(center[0]), float(center[1])
    radius = int(max(38.0, min(160.0, 0.46 * float(spacing_px))))
    x0 = max(0, int(round(cx)) - radius)
    y0 = max(0, int(round(cy)) - radius)
    x1 = min(image_w, int(round(cx)) + radius)
    y1 = min(image_h, int(round(cy)) + radius)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None

    crop = bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    purple_mask = cv2.inRange(
        hsv,
        np.array(lower_hsv, dtype=np.uint8),
        np.array(upper_hsv, dtype=np.uint8),
    )
    dark_mask = (((val < 72) & ((sat > 16) | (val < 50))).astype(np.uint8)) * 255
    blue_purple_dark = (((hue >= 85) & (hue <= 150) & (sat > 28) & (val < 112)).astype(np.uint8)) * 255
    mask = cv2.bitwise_or(purple_mask, dark_mask)
    mask = cv2.bitwise_or(mask, blue_purple_dark)

    local_center = (int(round(cx)) - x0, int(round(cy)) - y0)
    central = np.zeros(mask.shape, dtype=np.uint8)
    if mask_polygon_px is not None:
        polygon = np.array(mask_polygon_px, dtype=np.float32).reshape(-1, 2)
        polygon -= np.array([x0, y0], dtype=np.float32)
        cv2.fillPoly(central, [np.round(polygon).astype(np.int32)], 255, lineType=cv2.LINE_AA)
    else:
        axes = (
            int(max(18.0, min(radius, 0.42 * float(spacing_px)))),
            int(max(18.0, min(radius, 0.42 * float(spacing_px)))),
        )
        cv2.ellipse(central, local_center, axes, 0, 0, 360, 255, -1)
    mask = cv2.bitwise_and(mask, central)

    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = max(
        45.0,
        min(1200.0, 0.040 * float(spacing_px) * float(spacing_px)),
        min(0.10 * float(min_area), 900.0),
    )
    best: dict[str, Any] | None = None
    best_score = -float("inf")
    for contour_idx, contour in enumerate(contours):
        area = float(cv2.contourArea(contour))
        if area < min_contour_area:
            continue
        rect = cv2.minAreaRect(contour)
        (local_x, local_y), (width, height), _ = rect
        short_side = max(1.0, min(float(width), float(height)))
        long_side = max(float(width), float(height))
        aspect = long_side / short_side
        if aspect > max(3.0, float(max_aspect)):
            continue
        if short_side < 0.16 * float(spacing_px) or long_side < 0.25 * float(spacing_px):
            continue
        box_local = cv2.boxPoints(rect)
        inside = cv2.pointPolygonTest(box_local.astype(np.float32), (float(local_center[0]), float(local_center[1])), False) >= 0
        distance = math.hypot(float(local_x) - local_center[0], float(local_y) - local_center[1])
        if not inside and distance > 0.38 * float(spacing_px):
            continue
        box_global = box_local + np.array([x0, y0], dtype=np.float32)
        rect_angle = rect_angle_from_min_area_rect(rect)
        x, y, bw, bh = cv2.boundingRect(contour)
        roi = pad_and_crop(crop, x, y, bw, bh, pad=12)
        texture_angle, texture_confidence = texture_angle_from_roi(roi)
        if texture_angle is not None and texture_confidence >= 3.0:
            final_angle = texture_angle
            angle_source = "texture"
        else:
            final_angle = rect_angle
            angle_source = "rect"
        square_side = square_side_from_rect_and_area(width, height, area)
        square_box = square_box_from_center_angle((local_x + x0, local_y + y0), square_side, final_angle)
        color_features = chip_color_features(
            bgr,
            square_box,
            lower_hsv=lower_hsv,
            upper_hsv=upper_hsv,
        )

        score = area - distance * 8.0
        if score > best_score:
            best_score = score
            best = {
                "idx": -1,
                "contour_idx": int(contour_idx),
                "area": area,
                "center_px": [float(local_x + x0), float(local_y + y0)],
                "rect_size_px": [float(width), float(height)],
                "aspect_ratio": float(aspect),
                "box_px": square_box,
                "square_box_px": square_box,
                "square_side_px": float(square_side),
                "raw_rect_box_px": box_global.astype(float).tolist(),
                "contour_px": contour_points(contour, offset=(x0, y0)),
                "bounding_rect_px": [int(x + x0), int(y + y0), int(bw), int(bh)],
                "rect_angle_deg": float(rect_angle),
                "texture_angle_deg": float(texture_angle) if texture_angle is not None else None,
                "texture_confidence": float(texture_confidence),
                "final_angle_deg": float(final_angle),
                "angle_source": angle_source,
                "kind": "slot_chip",
                "slot_key": [int(slot_key[0]), int(slot_key[1])],
                "slot_row": int(slot_key[0]) + 1,
                "slot_col": int(slot_key[1]) + 1,
                **color_features,
            }
    return best


def scaled_polygon_points(
    image_shape: tuple[int, ...],
    points_norm: list[tuple[float, float]],
) -> list[list[float]]:
    image_h, image_w = image_shape[:2]
    return [[float(x) * float(image_w), float(y) * float(image_h)] for x, y in points_norm]


def occlusion_regions_for_image(
    image_shape: tuple[int, ...],
    bottom_ratio: float,
    *,
    use_fixed_arm_mask: bool = True,
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    if use_fixed_arm_mask:
        regions.append(
            {
                "type": "polygon",
                "label": "fixed_robot_arm",
                "points": scaled_polygon_points(image_shape, DEFAULT_ARM_OCCLUSION_POLYGON_NORM),
            }
        )
    ratio = max(0.0, min(0.95, float(bottom_ratio)))
    if ratio > 0.0:
        image_h, image_w = image_shape[:2]
        y0 = float(image_h) * (1.0 - ratio)
        regions.append({"type": "bottom_band", "x0": 0.0, "y0": y0, "x1": float(image_w), "y1": float(image_h)})
    return regions


def point_in_occlusion(point: tuple[float, float] | list[float] | None, regions: list[dict[str, Any]]) -> bool:
    if point is None:
        return False
    x, y = float(point[0]), float(point[1])
    for region in regions:
        if region.get("type") == "bottom_band":
            if float(region["x0"]) <= x <= float(region["x1"]) and float(region["y0"]) <= y <= float(region["y1"]):
                return True
        elif region.get("type") == "polygon":
            polygon = np.array(region.get("points", []), dtype=np.float32).reshape(-1, 2)
            if len(polygon) >= 3 and cv2.pointPolygonTest(polygon, (x, y), False) >= 0:
                return True
    return False


def polygon_overlaps_occlusion(polygon_points: list[list[float]] | None, regions: list[dict[str, Any]]) -> bool:
    if not polygon_points or not regions:
        return False
    polygon = np.array(polygon_points, dtype=np.float32).reshape(-1, 2)
    if len(polygon) < 3:
        return False
    for point in polygon:
        if point_in_occlusion(point.astype(float).tolist(), regions):
            return True
    polygon_contour = polygon.reshape(-1, 1, 2)
    center = polygon.mean(axis=0)
    if point_in_occlusion(center.astype(float).tolist(), regions):
        return True
    for region in regions:
        if region.get("type") != "polygon":
            continue
        region_points = np.array(region.get("points", []), dtype=np.float32).reshape(-1, 2)
        for point in region_points:
            if cv2.pointPolygonTest(polygon_contour, (float(point[0]), float(point[1])), False) >= 0:
                return True
    return False


def expanded_convex_hull(points: list[tuple[float, float] | list[float]], image_shape: tuple[int, ...], padding_ratio: float) -> list[list[float]] | None:
    if len(points) < 3:
        return None
    arr = np.array(points, dtype=np.float32).reshape(-1, 2)
    hull = cv2.convexHull(arr).reshape(-1, 2)
    if len(hull) < 3:
        return None
    center = hull.mean(axis=0)
    expanded = center + (hull - center) * (1.0 + max(0.0, float(padding_ratio)))
    image_h, image_w = image_shape[:2]
    expanded[:, 0] = np.clip(expanded[:, 0], 0.0, float(image_w - 1))
    expanded[:, 1] = np.clip(expanded[:, 1], 0.0, float(image_h - 1))
    return expanded.astype(float).tolist()


def tray_roi_polygon(
    image_shape: tuple[int, ...],
    *,
    markers: list[Marker],
    locator_ids: list[int],
    layout: dict[str, Any] | None,
    reference_transform: list[list[float]] | None,
    transformed_slot_points: dict[tuple[int, int], tuple[float, float]],
    padding_ratio: float = DEFAULT_TRAY_ROI_PADDING_RATIO,
) -> tuple[list[list[float]] | None, str]:
    points: list[tuple[float, float] | list[float]] = []
    reference_centers = layout_marker_reference_centers(layout)
    if reference_transform is not None and transformed_slot_points:
        points.extend(transformed_slot_points.values())
        for marker_id in locator_ids:
            ref_point = reference_centers.get(int(marker_id))
            if ref_point is not None:
                points.append(apply_point_transform(reference_transform, ref_point))
        polygon = expanded_convex_hull(points, image_shape, padding_ratio)
        if polygon is not None:
            return polygon, "layout_reference"

    locator_set = set(int(x) for x in locator_ids)
    locator_corner_points: list[list[float]] = []
    known_corner_points: list[list[float]] = []
    for marker in markers:
        marker_points = marker.corners.astype(float).tolist()
        known_corner_points.extend(marker_points)
        if marker.marker_id in locator_set:
            locator_corner_points.extend(marker_points)
    polygon = expanded_convex_hull(locator_corner_points, image_shape, padding_ratio * 1.5)
    if polygon is not None:
        return polygon, "locator_hull"
    polygon = expanded_convex_hull(known_corner_points, image_shape, padding_ratio * 2.0)
    if polygon is not None:
        return polygon, "marker_hull"
    return None, "none"


def mask_from_polygon(image_shape: tuple[int, ...], polygon: list[list[float]] | None) -> np.ndarray | None:
    if polygon is None:
        return None
    image_h, image_w = image_shape[:2]
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 255, lineType=cv2.LINE_AA)
    return mask


def point_in_polygon(point: tuple[float, float] | list[float] | None, polygon: list[list[float]] | None) -> bool:
    if point is None or polygon is None:
        return False
    pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return False
    return cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False) >= 0


def grid_spacing_from_affine(affine: list[list[float]] | None, fallback_spacing: float) -> float:
    if affine is None:
        return float(fallback_spacing)
    matrix = np.array(affine, dtype=float)
    if matrix.shape != (3, 2):
        return float(fallback_spacing)
    col_step = float(np.linalg.norm(matrix[0]))
    row_step = float(np.linalg.norm(matrix[1]))
    usable = [value for value in (col_step, row_step) if value > 1e-6]
    return float(np.median(usable)) if usable else float(fallback_spacing)


def point_near_or_outside_image(
    point: tuple[float, float] | list[float] | None,
    image_shape: tuple[int, ...],
    margin_px: float,
) -> bool:
    if point is None:
        return True
    image_h, image_w = image_shape[:2]
    x, y = float(point[0]), float(point[1])
    return (
        x < margin_px
        or y < margin_px
        or x > float(image_w) - margin_px
        or y > float(image_h) - margin_px
    )


def slot_cell_near_or_outside_image(
    affine: list[list[float]] | None,
    col: int,
    row: int,
    image_shape: tuple[int, ...],
) -> bool:
    if affine is None:
        return False
    image_h, image_w = image_shape[:2]
    corners = [
        apply_affine(affine, col - 0.5, row - 0.5),
        apply_affine(affine, col + 0.5, row - 0.5),
        apply_affine(affine, col + 0.5, row + 0.5),
        apply_affine(affine, col - 0.5, row + 0.5),
    ]
    for x, y in corners:
        if x < 0.0 or y < 0.0 or x > float(image_w) or y > float(image_h):
            return True
    return False


def slot_detection_polygon(
    affine: list[list[float]] | None,
    col: int,
    row: int,
    half_size: float = 0.42,
) -> list[tuple[float, float]] | None:
    if affine is None:
        return None
    return [
        apply_affine(affine, col - half_size, row - half_size),
        apply_affine(affine, col + half_size, row - half_size),
        apply_affine(affine, col + half_size, row + half_size),
        apply_affine(affine, col - half_size, row + half_size),
    ]


def grid_rect_polygon(
    affine: list[list[float]] | None,
    col0: float,
    row0: float,
    col1: float,
    row1: float,
) -> list[tuple[float, float]] | None:
    if affine is None:
        return None
    return [
        apply_affine(affine, col0, row0),
        apply_affine(affine, col1, row0),
        apply_affine(affine, col1, row1),
        apply_affine(affine, col0, row1),
    ]


def image_point_to_grid(
    affine: list[list[float]] | None,
    point: tuple[float, float] | list[float] | np.ndarray,
) -> tuple[float, float] | None:
    if affine is None:
        return None
    matrix = np.array(affine, dtype=float)
    if matrix.shape != (3, 2):
        return None
    linear = np.array(
        [
            [matrix[0, 0], matrix[1, 0]],
            [matrix[0, 1], matrix[1, 1]],
        ],
        dtype=float,
    )
    if abs(float(np.linalg.det(linear))) < 1e-9:
        return None
    target = np.array([float(point[0]) - matrix[2, 0], float(point[1]) - matrix[2, 1]], dtype=float)
    try:
        col, row = np.linalg.solve(linear, target)
    except np.linalg.LinAlgError:
        return None
    return float(col), float(row)


def occlusion_mask_from_regions(image_shape: tuple[int, ...], regions: list[dict[str, Any]]) -> np.ndarray:
    image_h, image_w = image_shape[:2]
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    for region in regions:
        if region.get("type") == "polygon":
            points = np.array(region.get("points", []), dtype=np.int32).reshape(-1, 2)
            if len(points) >= 3:
                cv2.fillPoly(mask, [points], 255, lineType=cv2.LINE_AA)
        elif region.get("type") == "bottom_band":
            x0 = int(max(0, min(image_w, round(float(region["x0"])))))
            y0 = int(max(0, min(image_h, round(float(region["y0"])))))
            x1 = int(max(0, min(image_w, round(float(region["x1"])))))
            y1 = int(max(0, min(image_h, round(float(region["y1"])))))
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    return mask


def inter_slot_chip_like_mask(image: np.ndarray) -> np.ndarray:
    bgr = as_bgr_image(image)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    saturated_purple = (hue >= 95) & (hue <= 168) & (sat >= 50) & (val >= 20)
    saturated_dark = (val < 82) & (sat >= 30)
    very_dark_colored = (val < 45) & (sat >= 10)
    blue_purple_dark = (hue >= 82) & (hue <= 160) & (sat >= 42) & (val < 135)
    mask = ((saturated_purple | saturated_dark | very_dark_colored | blue_purple_dark).astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def detect_silver_gap_obstructions(
    image: np.ndarray,
    *,
    affine: list[list[float]] | None,
    rows: int,
    cols: int,
    spacing_px: float,
    tray_roi_mask: np.ndarray | None,
    occlusion_regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if affine is None or float(spacing_px) < DEFAULT_SILVER_GAP_MIN_SPACING_PX:
        return []
    chip_mask = inter_slot_chip_like_mask(image)
    occlusion_mask = occlusion_mask_from_regions(image.shape, occlusion_regions)
    usable_tray_mask = tray_roi_mask if tray_roi_mask is not None and tray_roi_mask.shape[:2] == chip_mask.shape[:2] else None
    boundary_half = DEFAULT_SILVER_GAP_BOUNDARY_HALF_RATIO
    probe_half = DEFAULT_SILVER_GAP_PROBE_HALF_RATIO
    obstructions: list[dict[str, Any]] = []

    def inspect_gap(
        *,
        polygon: list[tuple[float, float]] | None,
        gap_type: str,
        boundary_grid_value: float,
        neighbor_slots: list[tuple[int, int]],
    ) -> None:
        if polygon is None:
            return
        gap_mask = polygon_pixel_mask(image.shape, polygon)
        total_pixels = int(np.count_nonzero(gap_mask))
        if total_pixels < 30:
            return
        usable_mask = gap_mask
        if usable_tray_mask is not None:
            usable_mask = cv2.bitwise_and(usable_mask, usable_tray_mask)
        usable_mask = cv2.bitwise_and(usable_mask, cv2.bitwise_not(occlusion_mask))
        usable_pixels = int(np.count_nonzero(usable_mask))
        usable_ratio = usable_pixels / max(1.0, float(total_pixels))
        if usable_ratio < 0.75:
            return
        chip_pixels = ((chip_mask > 0) & (usable_mask > 0)).astype(np.uint8)
        chip_count = int(np.count_nonzero(chip_pixels))
        coverage_ratio = chip_count / max(1.0, float(usable_pixels))
        if coverage_ratio < DEFAULT_SILVER_GAP_MIN_COVERAGE_RATIO:
            return

        label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(chip_pixels, 8)
        if label_count <= 1:
            return
        component_labels = range(1, label_count)
        best_label = max(component_labels, key=lambda label: int(stats[label, cv2.CC_STAT_AREA]))
        largest_component_area = int(stats[best_label, cv2.CC_STAT_AREA])
        largest_component_ratio = largest_component_area / max(1.0, float(usable_pixels))
        if largest_component_ratio < DEFAULT_SILVER_GAP_MIN_COMPONENT_RATIO:
            return
        component_center = centroids[best_label]
        grid_center = image_point_to_grid(affine, component_center)
        if grid_center is None:
            return
        grid_col, grid_row = grid_center
        if gap_type == "row_gap":
            assigned_slot = neighbor_slots[1] if grid_row >= boundary_grid_value else neighbor_slots[0]
        else:
            assigned_slot = neighbor_slots[1] if grid_col >= boundary_grid_value else neighbor_slots[0]
        obstructions.append(
            {
                "type": gap_type,
                "between_slots": [[int(row) + 1, int(col) + 1] for row, col in neighbor_slots],
                "assigned_slot": [int(assigned_slot[0]) + 1, int(assigned_slot[1]) + 1],
                "polygon_px": [[float(x), float(y)] for x, y in polygon],
                "chip_pixel_ratio": float(coverage_ratio),
                "largest_component_ratio": float(largest_component_ratio),
                "usable_pixel_ratio": float(usable_ratio),
                "component_center_px": [float(component_center[0]), float(component_center[1])],
                "component_center_grid": [float(grid_col), float(grid_row)],
            }
        )

    for row in range(rows):
        for col in range(cols - 1):
            inspect_gap(
                polygon=grid_rect_polygon(
                    affine,
                    col + 0.5 - boundary_half,
                    row - probe_half,
                    col + 0.5 + boundary_half,
                    row + probe_half,
                ),
                gap_type="col_gap",
                boundary_grid_value=col + 0.5,
                neighbor_slots=[(row, col), (row, col + 1)],
            )
    for row in range(rows - 1):
        for col in range(cols):
            inspect_gap(
                polygon=grid_rect_polygon(
                    affine,
                    col - probe_half,
                    row + 0.5 - boundary_half,
                    col + probe_half,
                    row + 0.5 + boundary_half,
                ),
                gap_type="row_gap",
                boundary_grid_value=row + 0.5,
                neighbor_slots=[(row, col), (row + 1, col)],
            )
    return obstructions


def detect_marker_like_pattern_at_slot(
    image: np.ndarray,
    *,
    center: tuple[float, float] | list[float] | None,
    spacing_px: float,
    lower_hsv: tuple[int, int, int] = DEFAULT_LOWER_PURPLE,
    upper_hsv: tuple[int, int, int] = DEFAULT_UPPER_PURPLE,
) -> tuple[bool, dict[str, float]]:
    if center is None:
        return False, {}
    bgr = as_bgr_image(image)
    image_h, image_w = bgr.shape[:2]
    cx, cy = float(center[0]), float(center[1])
    radius = int(max(10.0, min(70.0, 0.33 * float(spacing_px))))
    x0 = max(0, int(round(cx)) - radius)
    y0 = max(0, int(round(cy)) - radius)
    x1 = min(image_w, int(round(cx)) + radius)
    y1 = min(image_h, int(round(cy)) + radius)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return False, {}

    crop = bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    contrast = float(np.percentile(gray, 90) - np.percentile(gray, 10))
    dark_ratio = float(np.mean(gray < 82))
    bright_ratio = float(np.mean(gray > 128))
    white_ratio = float(np.mean((v > 128) & (s < 55)))
    colorful_ratio = float(np.mean(s > 55))
    purple_ratio = float(
        np.mean(
            (h >= int(lower_hsv[0]))
            & (h <= int(upper_hsv[0]))
            & (s >= max(20, int(lower_hsv[1])))
            & (v >= max(20, int(lower_hsv[2])))
        )
    )
    edges = cv2.Canny(gray, 50, 140)
    edge_ratio = float(np.mean(edges > 0))
    marker_like = (
        contrast > 58.0
        and 0.08 <= dark_ratio <= 0.72
        and bright_ratio > 0.10
        and white_ratio > 0.08
        and colorful_ratio < 0.42
        and purple_ratio < 0.22
        and edge_ratio > 0.045
    )
    features = {
        "marker_like_contrast": contrast,
        "marker_like_dark_ratio": dark_ratio,
        "marker_like_bright_ratio": bright_ratio,
        "marker_like_white_ratio": white_ratio,
        "marker_like_colorful_ratio": colorful_ratio,
        "marker_like_purple_ratio": purple_ratio,
        "marker_like_edge_ratio": edge_ratio,
    }
    return bool(marker_like), features


def flake_slot_match_score(
    point: tuple[float, float] | list[float] | None,
    flake: dict[str, Any],
    spacing_px: float,
) -> float | None:
    if point is None:
        return None
    x, y = float(point[0]), float(point[1])
    max_distance = max(25.0, 0.65 * float(spacing_px))
    polygon_points = flake.get("square_box_px") or flake["box_px"]
    polygon = np.array(polygon_points, dtype=np.float32).reshape(-1, 2)
    signed_distance = cv2.pointPolygonTest(polygon, (x, y), True)
    fx, fy = flake["center_px"]
    center_distance = math.hypot(float(fx) - x, float(fy) - y)
    if signed_distance >= 0:
        return center_distance * 0.25
    if center_distance <= max_distance:
        return center_distance
    if signed_distance >= -0.18 * float(spacing_px):
        return max_distance + abs(float(signed_distance))
    return None


def assign_flakes_to_slots(
    slot_points: dict[tuple[int, int], tuple[float, float] | list[float] | None],
    flakes: list[dict[str, Any]],
    spacing_px: float,
) -> dict[tuple[int, int], dict[str, Any]]:
    candidates: list[tuple[float, tuple[int, int], int]] = []
    for slot_key, point in slot_points.items():
        for flake in flakes:
            target_slot = flake.get("slot_key")
            if target_slot is not None and tuple(int(x) for x in target_slot) != slot_key:
                continue
            score = flake_slot_match_score(point, flake, spacing_px)
            if score is not None:
                candidates.append((float(score), slot_key, int(flake["idx"])))

    candidates.sort(key=lambda item: item[0])
    used_slots: set[tuple[int, int]] = set()
    used_flakes: set[int] = set()
    flake_by_idx = {int(flake["idx"]): flake for flake in flakes}
    matches: dict[tuple[int, int], dict[str, Any]] = {}
    for _score, slot_key, flake_idx in candidates:
        if slot_key in used_slots or flake_idx in used_flakes:
            continue
        used_slots.add(slot_key)
        used_flakes.add(flake_idx)
        matches[slot_key] = flake_by_idx[flake_idx]
    return matches


def analyze_image(
    image: np.ndarray,
    *,
    image_path: str | None = None,
    dictionary_name: str = DEFAULT_DICT,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    layout: dict[str, Any] | None = None,
    occlusion_bottom_ratio: float = DEFAULT_OCCLUSION_BOTTOM_RATIO,
    detect_flakes: bool = True,
    flake_lower_hsv: tuple[int, int, int] = DEFAULT_LOWER_PURPLE,
    flake_upper_hsv: tuple[int, int, int] = DEFAULT_UPPER_PURPLE,
    flake_min_area: float = DEFAULT_FLAKE_MIN_AREA,
    flake_max_aspect: float = DEFAULT_FLAKE_MAX_ASPECT,
    edge_occlusion_margin_ratio: float = DEFAULT_EDGE_OCCLUSION_MARGIN_RATIO,
    use_fixed_arm_mask: bool = True,
) -> dict[str, Any]:
    flakes: list[dict[str, Any]] = []
    markers = detect_markers_multiscale(image, dictionary_name)
    if not markers:
        raise RuntimeError("no ArUco markers were detected")

    centers = np.array([marker.center for marker in markers], dtype=float)
    if len(centers) >= 2:
        try:
            pairwise_theta, spacing, pair_count = estimate_pairwise_axis(centers)
        except ValueError:
            pairwise_theta = math.radians(float(layout.get("reference_tray_angle_deg", 0.0))) if layout else 0.0
            spacing = layout_reference_grid_spacing(layout) or 1.0
            pair_count = 0
    else:
        pairwise_theta = math.radians(float(layout.get("reference_tray_angle_deg", 0.0))) if layout else 0.0
        spacing = layout_reference_grid_spacing(layout) or 1.0
        pair_count = 0
    ex = np.array([math.cos(pairwise_theta), math.sin(pairwise_theta)], dtype=float)
    ey = np.array([-math.sin(pairwise_theta), math.cos(pairwise_theta)], dtype=float)

    for marker in markers:
        marker.u = float(marker.center @ ex)
        marker.v = float(marker.center @ ey)

    u_values = np.array([marker.u for marker in markers], dtype=float)
    v_values = np.array([marker.v for marker in markers], dtype=float)
    u_clusters = cluster_1d(u_values, spacing * 0.38)
    v_clusters = cluster_1d(v_values, spacing * 0.38)
    for marker in markers:
        marker.u_cluster = min(range(len(u_clusters)), key=lambda i: abs(marker.u - float(u_clusters[i]["center"])))
        marker.v_cluster = min(range(len(v_clusters)), key=lambda i: abs(marker.v - float(v_clusters[i]["center"])))

    marker_by_id = {marker.marker_id: marker for marker in markers}
    reference_transform, reference_transform_marker_ids = estimate_reference_transform(layout, marker_by_id)
    transformed_slot_points = transformed_slot_reference_points(layout, reference_transform)
    auto_slot_grid: list[list[Marker | None]] | None = None
    auto_locator_markers: list[Marker] = []
    auto_u_start: int | None = None
    auto_v_start: int | None = None

    try:
        auto_u_start, auto_v_start = choose_slot_window(markers, u_clusters, v_clusters, rows, cols)
        auto_slot_grid = [[None for _ in range(cols)] for _ in range(rows)]
        for marker in markers:
            col = marker.u_cluster - auto_u_start
            row = marker.v_cluster - auto_v_start
            if 0 <= row < rows and 0 <= col < cols and auto_slot_grid[row][col] is None:
                auto_slot_grid[row][col] = marker
            else:
                auto_locator_markers.append(marker)
    except ValueError:
        auto_slot_grid = None
        auto_locator_markers = []

    if layout is not None:
        slot_id_grid = layout.get("slot_id_grid")
        if not slot_id_grid:
            raise ValueError("layout file does not contain slot_id_grid")
        rows = int(layout.get("rows", len(slot_id_grid)))
        cols = int(layout.get("cols", len(slot_id_grid[0])))
        slot_id_set = {int(x) for row in slot_id_grid for x in row}
        locator_ids = [int(x) for x in layout.get("locator_ids", [])]
        locator_id_set = set(locator_ids)
        layout_reference_centers = layout_marker_reference_centers(layout)
        reference_affine_for_filter = fit_grid_affine_from_points(transformed_slot_points)
        reference_spacing_for_filter = grid_spacing_from_affine(reference_affine_for_filter, spacing)

        def marker_is_consistent_with_layout(marker: Marker) -> bool:
            if reference_transform is None:
                return True
            ref_point = layout_reference_centers.get(int(marker.marker_id))
            if ref_point is None:
                return True
            predicted = apply_point_transform(reference_transform, ref_point)
            distance = float(np.linalg.norm(marker.center - np.array(predicted, dtype=float)))
            return distance <= max(14.0, 0.45 * float(reference_spacing_for_filter))

        slot_grid: list[list[Marker | None]] = []
        for row_ids in slot_id_grid:
            row_markers: list[Marker | None] = []
            for marker_id in row_ids:
                marker = marker_by_id.get(int(marker_id))
                if marker is not None and not marker_is_consistent_with_layout(marker):
                    marker = None
                row_markers.append(marker)
            slot_grid.append(row_markers)
        locator_markers = [
            marker_by_id[x]
            for x in locator_ids
            if x in marker_by_id and marker_is_consistent_with_layout(marker_by_id[x])
        ]
        unknown_markers = [
            marker
            for marker in markers
            if (
                marker.marker_id not in slot_id_set
                and marker.marker_id not in locator_id_set
            )
            or (
                (marker.marker_id in slot_id_set or marker.marker_id in locator_id_set)
                and not marker_is_consistent_with_layout(marker)
            )
        ]
    else:
        if auto_slot_grid is None:
            raise RuntimeError("could not derive the slot grid from detected markers")
        slot_grid = auto_slot_grid
        slot_id_grid = [
            [marker.marker_id if marker is not None else None for marker in row]
            for row in slot_grid
        ]
        locator_markers = auto_locator_markers
        locator_ids = [marker.marker_id for marker in locator_markers]
        unknown_markers = []

    visible_slot_count = sum(1 for row in slot_grid for marker in row if marker is not None)
    visible_grid_theta, row_angles, col_angles_as_rows = estimate_grid_line_angle(slot_grid, pairwise_theta)
    visible_affine = fit_affine_from_slots(slot_grid)
    reference_affine = fit_grid_affine_from_points(transformed_slot_points)
    visible_affine_reliable = visible_affine is not None and visible_slot_count >= max(8, min(rows * cols, 12))
    affine = visible_affine if visible_affine_reliable else reference_affine or visible_affine

    reference_grid_theta: float | None = None
    if reference_affine is not None:
        reference_matrix = np.array(reference_affine, dtype=float)
        if reference_matrix.shape == (3, 2) and float(np.linalg.norm(reference_matrix[0])) > 1e-6:
            reference_grid_theta = normalize_axis_angle(
                math.atan2(float(reference_matrix[0][1]), float(reference_matrix[0][0]))
            )
    if visible_affine_reliable:
        grid_theta = visible_grid_theta
        grid_angle_source = "visible_slot_markers"
    elif reference_grid_theta is not None:
        grid_theta = reference_grid_theta
        grid_angle_source = "layout_reference"
    else:
        grid_theta = visible_grid_theta
        grid_angle_source = "pairwise_markers"

    tray_angle_deg = math.degrees(grid_theta)
    occlusion_regions = occlusion_regions_for_image(
        image.shape,
        occlusion_bottom_ratio,
        use_fixed_arm_mask=use_fixed_arm_mask,
    )
    grid_spacing_px = grid_spacing_from_affine(affine, spacing)
    edge_margin_px = max(0.0, float(edge_occlusion_margin_ratio)) * grid_spacing_px
    tray_roi_poly, tray_roi_source = tray_roi_polygon(
        image.shape,
        markers=markers,
        locator_ids=locator_ids,
        layout=layout,
        reference_transform=reference_transform,
        transformed_slot_points=transformed_slot_points,
        padding_ratio=DEFAULT_TRAY_ROI_PADDING_RATIO,
    )
    tray_roi_mask = mask_from_polygon(image.shape, tray_roi_poly)

    if detect_flakes:
        adaptive_global_min_area = min(
            float(flake_min_area),
            max(80.0, 0.10 * float(grid_spacing_px) * float(grid_spacing_px)),
        )
        flakes = detect_purple_flakes(
            image,
            lower_hsv=flake_lower_hsv,
            upper_hsv=flake_upper_hsv,
            min_area=adaptive_global_min_area,
            max_aspect=max(float(flake_max_aspect), 4.0),
            allowed_mask=tray_roi_mask,
        )
        for flake in flakes:
            flake["kind"] = flake.get("kind", "purple")
            flake["detection_scope"] = "tray_roi" if tray_roi_mask is not None else "full_image"
        apply_square_angle_model(flakes, tray_angle_deg)

    slot_geometry: list[dict[str, Any]] = []
    flake_match_points: dict[tuple[int, int], tuple[float, float] | list[float] | None] = {}
    for row_idx, row in enumerate(slot_grid):
        for col_idx, marker in enumerate(row):
            expected_id = slot_id_grid[row_idx][col_idx]
            reference_center = transformed_slot_points.get((row_idx, col_idx))
            predicted_center = apply_affine(affine, col_idx, row_idx) if affine is not None else reference_center
            center_for_occlusion = marker.center.tolist() if marker is not None else list(predicted_center) if predicted_center is not None else None
            slot_polygon = slot_detection_polygon(affine, col_idx, row_idx)
            outside_tray_roi = (
                tray_roi_poly is not None
                and center_for_occlusion is not None
                and not point_in_polygon(center_for_occlusion, tray_roi_poly)
            )
            marker_like_visible = False
            marker_like_features: dict[str, float] = {}
            known_unknown_area = (
                point_in_occlusion(center_for_occlusion, occlusion_regions)
                or slot_cell_near_or_outside_image(affine, col_idx, row_idx, image.shape)
                or point_near_or_outside_image(center_for_occlusion, image.shape, edge_margin_px)
            )
            if marker is None and center_for_occlusion is not None and not outside_tray_roi and not known_unknown_area:
                marker_like_visible, marker_like_features = detect_marker_like_pattern_at_slot(
                    image,
                    center=center_for_occlusion,
                    spacing_px=grid_spacing_px,
                    lower_hsv=flake_lower_hsv,
                    upper_hsv=flake_upper_hsv,
                )
            slot_geometry.append(
                {
                    "row_idx": row_idx,
                    "col_idx": col_idx,
                    "marker": marker,
                    "expected_id": expected_id,
                    "predicted_center": predicted_center,
                    "center_for_occlusion": center_for_occlusion,
                    "slot_polygon": slot_polygon,
                    "outside_tray_roi": outside_tray_roi,
                    "marker_like_visible": marker_like_visible,
                    "marker_like_features": marker_like_features,
                }
            )
            if marker is None and not marker_like_visible:
                flake_match_points[(row_idx, col_idx)] = center_for_occlusion

    initial_flake_matches = assign_flakes_to_slots(flake_match_points, flakes, grid_spacing_px)
    if detect_flakes:
        for item in slot_geometry:
            marker = item["marker"]
            if marker is not None:
                continue
            if item.get("marker_like_visible"):
                continue
            row_idx = int(item["row_idx"])
            col_idx = int(item["col_idx"])
            if (row_idx, col_idx) in initial_flake_matches:
                continue
            center_for_occlusion = item["center_for_occlusion"]
            if point_in_occlusion(center_for_occlusion, occlusion_regions):
                continue
            if item["outside_tray_roi"]:
                continue
            if slot_cell_near_or_outside_image(affine, col_idx, row_idx, image.shape):
                continue
            if point_near_or_outside_image(center_for_occlusion, image.shape, edge_margin_px):
                continue
            candidate = detect_slot_chip_candidate(
                image,
                center=center_for_occlusion,
                spacing_px=grid_spacing_px,
                slot_key=(row_idx, col_idx),
                mask_polygon_px=item["slot_polygon"],
                lower_hsv=flake_lower_hsv,
                upper_hsv=flake_upper_hsv,
                min_area=flake_min_area,
                max_aspect=max(float(flake_max_aspect), 4.0),
            )
            if candidate is not None:
                candidate["idx"] = len(flakes)
                flakes.append(candidate)
        apply_square_angle_model(flakes, tray_angle_deg)

    flake_matches = assign_flakes_to_slots(flake_match_points, flakes, grid_spacing_px)
    slot_geometry_by_key = {(int(item["row_idx"]), int(item["col_idx"])): item for item in slot_geometry}
    silver_gap_obstructions = (
        detect_silver_gap_obstructions(
            image,
            affine=affine,
            rows=rows,
            cols=cols,
            spacing_px=grid_spacing_px,
            tray_roi_mask=tray_roi_mask,
            occlusion_regions=occlusion_regions,
        )
        if detect_flakes
        else []
    )
    silver_gap_by_slot: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for gap in silver_gap_obstructions:
        assigned_row, assigned_col = [int(value) - 1 for value in gap.get("assigned_slot", [0, 0])]
        assigned_key = (assigned_row, assigned_col)
        target_key: tuple[int, int] | None = assigned_key if assigned_key in flake_matches else None
        if target_key is None:
            neighbor_keys = [
                (int(row) - 1, int(col) - 1)
                for row, col in gap.get("between_slots", [])
                if (int(row) - 1, int(col) - 1) in flake_matches
            ]
            if len(neighbor_keys) == 1:
                target_key = neighbor_keys[0]
            elif len(neighbor_keys) > 1:
                component_center = gap.get("component_center_px")
                if component_center is not None:
                    target_key = min(
                        neighbor_keys,
                        key=lambda key: math.hypot(
                            float(flake_matches[key]["center_px"][0]) - float(component_center[0]),
                            float(flake_matches[key]["center_px"][1]) - float(component_center[1]),
                        ),
                    )
                else:
                    target_key = neighbor_keys[0]
        if target_key is None:
            continue
        gap["applied_slot"] = [int(target_key[0]) + 1, int(target_key[1]) + 1]
        silver_gap_by_slot.setdefault(target_key, []).append(gap)
    for slot_key, matched_flake in flake_matches.items():
        slot_gaps = silver_gap_by_slot.get(slot_key, [])
        if not slot_gaps:
            continue
        matched_flake["silver_gap_blocked_count"] = int(len(slot_gaps))
        matched_flake["silver_gap_blocked_score"] = float(max(float(gap["chip_pixel_ratio"]) for gap in slot_gaps))
        matched_flake["silver_gap_blocked_gaps"] = [
            {
                "type": gap.get("type"),
                "between_slots": gap.get("between_slots"),
                "chip_pixel_ratio": gap.get("chip_pixel_ratio"),
                "largest_component_ratio": gap.get("largest_component_ratio"),
            }
            for gap in slot_gaps
        ]
    pre_match_artifacts: list[dict[str, Any]] = []
    verified_flake_matches: dict[tuple[int, int], dict[str, Any]] = {}
    pre_match_artifact_ids: set[int] = set()
    for slot_key, matched_flake in flake_matches.items():
        slot_item = slot_geometry_by_key.get(slot_key, {})
        refine_flake_with_slot_context(
            matched_flake,
            image,
            slot_center=flake_match_points.get(slot_key),
            slot_polygon=slot_item.get("slot_polygon"),
            spacing_px=grid_spacing_px,
            reference_angle_deg=tray_angle_deg,
            lower_hsv=flake_lower_hsv,
            upper_hsv=flake_upper_hsv,
        )
        apply_square_angle_model([matched_flake], tray_angle_deg)
        if polygon_overlaps_occlusion(matched_flake.get("square_box_px"), occlusion_regions):
            matched_flake["partial_chip_view"] = True
            matched_flake["chip_occlusion_overlap"] = True
        matched_flake["quality_slot_key"] = [int(slot_key[0]), int(slot_key[1])]
        update_chip_quality(
            matched_flake,
            spacing_px=grid_spacing_px,
            slot_center=flake_match_points.get(slot_key),
            image_shape=image.shape,
        )
        if is_marker_artifact_like(matched_flake):
            flags = list(matched_flake.get("chip_flags", []))
            if "marker_artifact" not in flags:
                flags.append("marker_artifact")
            matched_flake["chip_status"] = "artifact"
            matched_flake["chip_flags"] = flags
            pre_match_artifacts.append(matched_flake)
            pre_match_artifact_ids.add(int(matched_flake["idx"]))
        else:
            verified_flake_matches[slot_key] = matched_flake
    flake_matches = verified_flake_matches
    if pre_match_artifact_ids:
        flakes = [flake for flake in flakes if int(flake["idx"]) not in pre_match_artifact_ids]

    slot_entries: list[dict[str, Any]] = []
    missing_slots: list[dict[str, Any]] = []
    occluded_slots: list[dict[str, Any]] = []
    occupied_slots: list[dict[str, Any]] = []
    warning_slots: list[dict[str, Any]] = []
    abnormal_slots: list[dict[str, Any]] = []
    visible_unread_slots: list[dict[str, Any]] = []
    for item in slot_geometry:
        row_idx = int(item["row_idx"])
        col_idx = int(item["col_idx"])
        marker = item["marker"]
        expected_id = item["expected_id"]
        predicted_center = item["predicted_center"]
        center_for_occlusion = item["center_for_occlusion"]
        marker_like_visible = bool(item.get("marker_like_visible"))
        matched_flake = None if marker is not None else flake_matches.get((row_idx, col_idx))
        if matched_flake is not None:
            if matched_flake.get("quality_slot_key") != [row_idx, col_idx]:
                refine_flake_with_slot_context(
                    matched_flake,
                    image,
                    slot_center=center_for_occlusion,
                    slot_polygon=item.get("slot_polygon"),
                    spacing_px=grid_spacing_px,
                    reference_angle_deg=tray_angle_deg,
                    lower_hsv=flake_lower_hsv,
                    upper_hsv=flake_upper_hsv,
                )
                apply_square_angle_model([matched_flake], tray_angle_deg)
                if polygon_overlaps_occlusion(matched_flake.get("square_box_px"), occlusion_regions):
                    matched_flake["partial_chip_view"] = True
                    matched_flake["chip_occlusion_overlap"] = True
                matched_flake["quality_slot_key"] = [row_idx, col_idx]
            update_chip_quality(
                matched_flake,
                spacing_px=grid_spacing_px,
                slot_center=center_for_occlusion,
                image_shape=image.shape,
            )
        occlusion_reason = None
        if marker is None and marker_like_visible:
            occlusion_reason = None
        elif marker is None and point_in_occlusion(center_for_occlusion, occlusion_regions):
            occlusion_reason = "arm_occlusion_region"
        elif marker is None and item["outside_tray_roi"]:
            occlusion_reason = "outside_tray_roi"
        elif marker is None and slot_cell_near_or_outside_image(affine, col_idx, row_idx, image.shape):
            occlusion_reason = "partly_outside_camera_view"
        elif marker is None and point_near_or_outside_image(center_for_occlusion, image.shape, edge_margin_px):
            occlusion_reason = "near_or_outside_image_edge"
        occluded = marker is None and not marker_like_visible and matched_flake is None and occlusion_reason is not None
        occupied = marker is None and not marker_like_visible and matched_flake is not None
        chip_status = matched_flake.get("chip_status") if matched_flake is not None else None
        chip_flags = matched_flake.get("chip_flags", []) if matched_flake is not None else []
        abnormal = occupied and chip_status == "abnormal"
        warning = occupied and chip_status == "warning"
        state = (
            "visible"
            if marker is not None
            else "visible_unread_marker"
            if marker_like_visible
            else "abnormal"
            if abnormal
            else "warning"
            if warning
            else "occupied"
            if occupied
            else "occluded"
            if occluded
            else "missing"
        )
        entry = {
            "row": row_idx + 1,
            "col": col_idx + 1,
            "id": expected_id,
            "visible": marker is not None or marker_like_visible,
            "decoded_marker": marker is not None,
            "marker_like_visible": marker_like_visible,
            "marker_like_features": item.get("marker_like_features", {}),
            "occluded": occluded,
            "occupied": occupied,
            "warning": warning,
            "abnormal": abnormal,
            "state": state,
            "occlusion_reason": occlusion_reason if occluded else None,
            "matched_flake_idx": int(matched_flake["idx"]) if matched_flake is not None else None,
            "chip_status": chip_status,
            "chip_flags": chip_flags,
            "center_px": marker.center.tolist() if marker is not None else None,
            "predicted_center_px": list(predicted_center) if predicted_center is not None else None,
            "marker_angle_deg": marker.marker_angle_deg if marker is not None else None,
        }
        slot_entries.append(entry)
        if occupied:
            occupied_slots.append(entry)
            if abnormal:
                abnormal_slots.append(entry)
            elif warning:
                warning_slots.append(entry)
        elif marker_like_visible:
            visible_unread_slots.append(entry)
        elif occluded:
            occluded_slots.append(entry)
        elif marker is None:
            missing_slots.append(entry)

    used_flake_ids = {int(flake["idx"]) for flake in flake_matches.values()}
    slot_points = {
        (int(item["row_idx"]), int(item["col_idx"])): item["center_for_occlusion"]
        for item in slot_geometry
        if item["center_for_occlusion"] is not None
    }
    slot_entry_by_key = {(int(entry["row"]) - 1, int(entry["col"]) - 1): entry for entry in slot_entries}
    for gap in silver_gap_obstructions:
        applied_slot = gap.get("applied_slot")
        active = False
        if applied_slot is not None and len(applied_slot) == 2:
            entry = slot_entry_by_key.get((int(applied_slot[0]) - 1, int(applied_slot[1]) - 1))
            active = bool(entry and "silver_gap_blocked" in entry.get("chip_flags", []))
        gap["active"] = active
    silver_gap_obstructions = [gap for gap in silver_gap_obstructions if gap.get("active")]
    off_grid_chips: list[dict[str, Any]] = []
    extra_chips: list[dict[str, Any]] = []
    ignored_chip_artifacts: list[dict[str, Any]] = list(pre_match_artifacts)
    ignored_flake_ids: set[int] = set()
    for flake in flakes:
        flake_idx = int(flake["idx"])
        if flake_idx in used_flake_ids:
            continue
        center = flake.get("center_px")
        if point_in_occlusion(center, occlusion_regions):
            continue
        if tray_roi_poly is not None and not point_in_polygon(center, tray_roi_poly):
            continue
        nearest_slot: tuple[int, int] | None = None
        nearest_distance = float("inf")
        for slot_key, point in slot_points.items():
            if point is None:
                continue
            distance = math.hypot(float(center[0]) - float(point[0]), float(center[1]) - float(point[1]))
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_slot = slot_key
        if nearest_slot is not None:
            flake["nearest_slot"] = [int(nearest_slot[0]) + 1, int(nearest_slot[1]) + 1]
            flake["nearest_slot_distance_px"] = float(nearest_distance)
            flake["nearest_slot_distance_ratio"] = float(nearest_distance / max(1.0, grid_spacing_px))
        update_chip_quality(
            flake,
            spacing_px=grid_spacing_px,
            slot_center=slot_points.get(nearest_slot) if nearest_slot is not None else None,
            image_shape=image.shape,
        )
        flags = list(flake.get("chip_flags", []))
        nearest_entry = slot_entry_by_key.get(nearest_slot) if nearest_slot is not None else None
        if nearest_entry is not None and nearest_entry.get("visible") and nearest_distance <= max(12.0, 0.22 * grid_spacing_px):
            if "visible_marker_artifact" not in flags:
                flags.append("visible_marker_artifact")
            flake["chip_status"] = "artifact"
            flake["chip_flags"] = flags
            ignored_chip_artifacts.append(flake)
            ignored_flake_ids.add(flake_idx)
            continue
        if is_marker_artifact_like(flake):
            if "marker_artifact" not in flags:
                flags.append("marker_artifact")
            flake["chip_status"] = "artifact"
            flake["chip_flags"] = flags
            ignored_chip_artifacts.append(flake)
            ignored_flake_ids.add(flake_idx)
            continue
        severe_size_artifact = (
            float(flake.get("side_to_grid_ratio", 0.0)) > 1.80
            or float(flake.get("side_to_grid_ratio", 0.0)) < 0.32
            or float(flake.get("square_fill_ratio", 1.0)) < 0.35
            or float(flake.get("square_fill_ratio", 1.0)) > 1.45
        )
        if severe_size_artifact and (nearest_slot is None or nearest_distance > max(18.0, 0.38 * grid_spacing_px)):
            if "artifact_like_shape" not in flags:
                flags.append("artifact_like_shape")
            flake["chip_status"] = "artifact"
            flake["chip_flags"] = flags
            ignored_chip_artifacts.append(flake)
            ignored_flake_ids.add(flake_idx)
            continue
        if nearest_slot is not None and nearest_distance <= max(18.0, 0.38 * grid_spacing_px):
            if "extra_chip_near_slot" not in flags:
                flags.append("extra_chip_near_slot")
            flake["chip_status"] = "abnormal"
            flake["chip_flags"] = flags
            extra_chips.append(flake)
        else:
            if "off_grid" not in flags:
                flags.append("off_grid")
            flake["chip_status"] = "off_grid"
            flake["chip_flags"] = flags
            off_grid_chips.append(flake)
    if ignored_flake_ids:
        flakes = [flake for flake in flakes if int(flake["idx"]) not in ignored_flake_ids]

    marker_entries = [
        {
            "id": marker.marker_id,
            "center_px": marker.center.tolist(),
            "corners_px": marker.corners.tolist(),
            "marker_angle_deg": marker.marker_angle_deg,
            "projected_u": marker.u,
            "projected_v": marker.v,
            "u_cluster": marker.u_cluster,
            "v_cluster": marker.v_cluster,
        }
        for marker in markers
    ]

    return {
        "image": image_path,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dictionary": dictionary_name,
        "detected_marker_count": len(markers),
        "detected_ids": [marker.marker_id for marker in markers],
        "tray_angle_deg": tray_angle_deg,
        "pairwise_angle_deg": math.degrees(pairwise_theta),
        "nearest_neighbor_spacing_px": spacing,
        "near_neighbor_pair_count": pair_count,
        "grid": {
            "rows": rows,
            "cols": cols,
            "visible_slot_count": sum(1 for entry in slot_entries if entry["visible"]),
            "decoded_visible_slot_count": sum(1 for entry in slot_entries if entry.get("decoded_marker")),
            "visible_unread_slot_count": len(visible_unread_slots),
            "missing_slot_count": len(missing_slots),
            "occluded_slot_count": len(occluded_slots),
            "occupied_slot_count": len(occupied_slots),
            "warning_slot_count": len(warning_slots),
            "abnormal_slot_count": len(abnormal_slots),
            "off_grid_chip_count": len(off_grid_chips),
            "extra_chip_count": len(extra_chips),
            "ignored_chip_artifact_count": len(ignored_chip_artifacts),
            "silver_gap_obstruction_count": len(silver_gap_obstructions),
            "slot_id_grid": slot_id_grid,
            "row_line_angles_deg": [math.degrees(x) for x in row_angles],
            "column_line_angles_as_row_deg": [math.degrees(x) for x in col_angles_as_rows],
            "affine_grid_to_image": affine,
            "visible_affine_grid_to_image": visible_affine,
            "reference_affine_grid_to_image": reference_affine,
            "visible_affine_reliable": visible_affine_reliable,
            "angle_source": grid_angle_source,
            "u_cluster_start": auto_u_start,
            "v_cluster_start": auto_v_start,
            "u_clusters": [float(cluster["center"]) for cluster in u_clusters],
            "v_clusters": [float(cluster["center"]) for cluster in v_clusters],
            "spacing_px": grid_spacing_px,
            "edge_occlusion_margin_px": edge_margin_px,
        },
        "slots": slot_entries,
        "missing_slots": missing_slots,
        "occluded_slots": occluded_slots,
        "occupied_slots": occupied_slots,
        "visible_unread_slots": visible_unread_slots,
        "warning_slots": warning_slots,
        "abnormal_slots": abnormal_slots,
        "off_grid_chips": off_grid_chips,
        "extra_chips": extra_chips,
        "ignored_chip_artifacts": ignored_chip_artifacts,
        "silver_gap_obstructions": silver_gap_obstructions,
        "tray_roi": {
            "enabled": tray_roi_poly is not None,
            "source": tray_roi_source,
            "padding_ratio": DEFAULT_TRAY_ROI_PADDING_RATIO,
            "polygon_px": tray_roi_poly,
        },
        "reference_transform": {
            "available": reference_transform is not None,
            "marker_ids": reference_transform_marker_ids,
            "affine_reference_to_image": reference_transform,
        },
        "occlusion": {
            "bottom_ratio": max(0.0, min(0.95, float(occlusion_bottom_ratio))),
            "fixed_arm_mask_enabled": bool(use_fixed_arm_mask),
            "fixed_arm_polygon_norm": [list(point) for point in DEFAULT_ARM_OCCLUSION_POLYGON_NORM],
            "edge_margin_ratio": max(0.0, float(edge_occlusion_margin_ratio)),
            "edge_margin_px": edge_margin_px,
            "regions_px": occlusion_regions,
        },
        "flakes": flakes,
        "flake_detection": {
            "enabled": bool(detect_flakes),
            "count": len(flakes),
            "matched_slot_count": len(flake_matches),
            "warning_count": len(warning_slots),
            "abnormal_count": len(abnormal_slots),
            "off_grid_count": len(off_grid_chips),
            "extra_count": len(extra_chips),
            "ignored_artifact_count": len(ignored_chip_artifacts),
            "silver_gap_obstruction_count": len(silver_gap_obstructions),
            "lower_hsv": list(flake_lower_hsv),
            "upper_hsv": list(flake_upper_hsv),
            "min_area": float(flake_min_area),
            "adaptive_global_min_area": float(adaptive_global_min_area) if detect_flakes else None,
            "max_aspect": float(flake_max_aspect),
        },
        "locator_ids": locator_ids,
        "locator_markers": [
            {
                "id": marker.marker_id,
                "center_px": marker.center.tolist(),
                "marker_angle_deg": marker.marker_angle_deg,
            }
            for marker in locator_markers
        ],
        "unknown_markers": [
            {
                "id": marker.marker_id,
                "center_px": marker.center.tolist(),
                "marker_angle_deg": marker.marker_angle_deg,
            }
            for marker in unknown_markers
        ],
        "markers": marker_entries,
    }


def make_layout(result: dict[str, Any]) -> dict[str, Any]:
    marker_reference_centers = {
        str(marker["id"]): marker["center_px"]
        for marker in result.get("markers", [])
        if marker.get("center_px") is not None
    }
    marker_reference_corners = {
        str(marker["id"]): marker["corners_px"]
        for marker in result.get("markers", [])
        if marker.get("corners_px") is not None
    }
    rows = int(result["grid"]["rows"])
    cols = int(result["grid"]["cols"])
    slot_reference_centers: list[list[list[float] | None]] = [[None for _ in range(cols)] for _ in range(rows)]
    for slot in result.get("slots", []):
        row_idx = int(slot["row"]) - 1
        col_idx = int(slot["col"]) - 1
        center = slot.get("center_px") or slot.get("predicted_center_px")
        if 0 <= row_idx < rows and 0 <= col_idx < cols and center is not None:
            slot_reference_centers[row_idx][col_idx] = [float(center[0]), float(center[1])]
    locator_reference_centers = {
        str(marker["id"]): marker["center_px"]
        for marker in result.get("locator_markers", [])
        if marker.get("center_px") is not None
    }
    return {
        "version": 2,
        "created_at": result["timestamp"],
        "created_from_image": result.get("image"),
        "dictionary": result["dictionary"],
        "rows": rows,
        "cols": cols,
        "slot_id_grid": result["grid"]["slot_id_grid"],
        "locator_ids": result["locator_ids"],
        "reference_tray_angle_deg": result["tray_angle_deg"],
        "marker_reference_centers_px": marker_reference_centers,
        "marker_reference_corners_px": marker_reference_corners,
        "slot_reference_centers_px": slot_reference_centers,
        "locator_reference_centers_px": locator_reference_centers,
    }


def draw_result(image: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        canvas = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        canvas = image.copy()

    tray_roi = result.get("tray_roi", {})
    tray_polygon = tray_roi.get("polygon_px")
    if tray_polygon is not None:
        points = np.array(tray_polygon, dtype=np.int32).reshape(-1, 2)
        if len(points) >= 3:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [points], (255, 220, 0), lineType=cv2.LINE_AA)
            canvas = cv2.addWeighted(overlay, 0.10, canvas, 0.90, 0)
            cv2.polylines(canvas, [points], True, (255, 220, 0), 3, lineType=cv2.LINE_AA)
            x, y, _w, _h = cv2.boundingRect(points)
            cv2.putText(
                canvas,
                f"tray ROI: {tray_roi.get('source', 'unknown')}",
                (x + 10, max(y + 30, 34)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.86,
                (255, 245, 80),
                3,
                cv2.LINE_AA,
            )

    display_spacing = float(result.get("grid", {}).get("spacing_px") or 80.0)
    show_slot_labels = display_spacing >= 58.0
    show_chip_labels = display_spacing >= 72.0
    marker_text_scale = 0.78 if display_spacing >= 58.0 else 0.52
    marker_text_thickness = 3 if display_spacing >= 58.0 else 2

    for region in result.get("occlusion", {}).get("regions_px", []):
        overlay = canvas.copy()
        if region.get("type") == "polygon":
            points = np.array(region.get("points", []), dtype=np.int32).reshape(-1, 2)
            if len(points) >= 3:
                cv2.fillPoly(overlay, [points], (0, 215, 255), lineType=cv2.LINE_AA)
                canvas = cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0)
                cv2.polylines(canvas, [points], True, (0, 165, 255), 4, lineType=cv2.LINE_AA)
                x, y, w, h = cv2.boundingRect(points)
                cv2.putText(
                    canvas,
                    "fixed robot arm mask",
                    (x + 12, max(y + 34, 38)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.92,
                    (0, 120, 180),
                    3,
                    cv2.LINE_AA,
                )
        elif region.get("type") == "bottom_band":
            p0 = (int(round(float(region["x0"]))), int(round(float(region["y0"]))))
            p1 = (int(round(float(region["x1"]))), int(round(float(region["y1"]))))
            cv2.rectangle(overlay, p0, p1, (0, 215, 255), -1)
            canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0)
            cv2.rectangle(canvas, p0, p1, (0, 180, 230), 3, lineType=cv2.LINE_AA)
            cv2.putText(
                canvas,
                "robot arm occlusion region",
                (p0[0] + 16, max(p0[1] + 28, 28)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.92,
                (0, 120, 180),
                3,
                cv2.LINE_AA,
            )

    slot_ids = {entry["id"] for entry in result["slots"] if entry["id"] is not None}
    locator_ids = set(result["locator_ids"])

    for marker in result["markers"]:
        marker_id = int(marker["id"])
        corners = np.array(marker["corners_px"], dtype=np.int32).reshape(4, 2)
        if marker_id in slot_ids:
            color = (60, 180, 75)
        elif marker_id in locator_ids:
            color = (240, 160, 40)
        else:
            color = (180, 180, 180)
        cv2.polylines(canvas, [corners], True, color, 3, lineType=cv2.LINE_AA)
        center = tuple(np.round(marker["center_px"]).astype(int))
        cv2.circle(canvas, center, 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(marker_id),
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            marker_text_scale,
            color,
            marker_text_thickness,
            cv2.LINE_AA,
        )

    rows = int(result["grid"]["rows"])
    cols = int(result["grid"]["cols"])
    affine = result["grid"].get("affine_grid_to_image")
    if affine is not None:
        for row in range(rows):
            points = [tuple(round(x) for x in apply_affine(affine, col, row)) for col in range(cols)]
            cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, (30, 160, 255), 2, cv2.LINE_AA)
        for col in range(cols):
            points = [tuple(round(x) for x in apply_affine(affine, col, row)) for row in range(rows)]
            cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, (30, 160, 255), 2, cv2.LINE_AA)

    for gap in result.get("silver_gap_obstructions", []):
        polygon = gap.get("polygon_px")
        if not polygon:
            continue
        points = np.array(polygon, dtype=np.int32).reshape(-1, 2)
        if len(points) < 3:
            continue
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [points], (0, 0, 255), lineType=cv2.LINE_AA)
        canvas = cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0)
        cv2.polylines(canvas, [points], True, (0, 0, 255), 4, lineType=cv2.LINE_AA)
        applied_slot = gap.get("applied_slot") or gap.get("assigned_slot")
        if show_slot_labels and applied_slot:
            x, y, _w, _h = cv2.boundingRect(points)
            cv2.putText(
                canvas,
                f"gap->R{int(applied_slot[0])}C{int(applied_slot[1])}",
                (x + 4, max(22, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

    for slot in result["slots"]:
        center = slot["center_px"] or slot["predicted_center_px"]
        if center is None:
            continue
        x, y = (int(round(center[0])), int(round(center[1])))
        label = f"R{slot['row']}C{slot['col']}"
        if slot.get("marker_like_visible") and not slot.get("decoded_marker"):
            cv2.circle(canvas, (x, y), 18, (120, 255, 180), 3, lineType=cv2.LINE_AA)
            if show_slot_labels:
                cv2.putText(canvas, f"{label} QR", (x - 42, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (120, 255, 180), 2, cv2.LINE_AA)
        elif slot["visible"]:
            if show_slot_labels:
                cv2.putText(canvas, label, (x - 34, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
        elif slot.get("occupied"):
            if slot.get("abnormal"):
                color = (0, 0, 255)
                text = f"{label} ABNORMAL"
            elif slot.get("warning"):
                color = (0, 165, 255)
                text = f"{label} WARN"
            else:
                color = (255, 0, 255)
                text = f"{label} OCCUPIED"
            cv2.circle(canvas, (x, y), 26, color, 4, lineType=cv2.LINE_AA)
            if show_slot_labels:
                cv2.putText(canvas, text, (x - 78, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 3, cv2.LINE_AA)
        elif slot.get("occluded"):
            cv2.circle(canvas, (x, y), 22, (0, 215, 255), 3, lineType=cv2.LINE_AA)
            if show_slot_labels:
                cv2.putText(canvas, f"{label} OCC", (x - 48, y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 150, 220), 3, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (x, y), 22, (0, 0, 255), 3, lineType=cv2.LINE_AA)
            if show_slot_labels:
                cv2.putText(canvas, label, (x - 34, y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 3, cv2.LINE_AA)

    for flake in result.get("flakes", []):
        chip_status = flake.get("chip_status", "ok")
        if chip_status in {"abnormal", "off_grid"}:
            color = (0, 0, 255)
        elif chip_status == "warning":
            color = (0, 165, 255)
        else:
            color = (255, 0, 255)
        box = np.array(flake.get("square_box_px") or flake["box_px"], dtype=np.int32).reshape(4, 2)
        cv2.polylines(canvas, [box], True, color, 4, lineType=cv2.LINE_AA)
        cx, cy = flake["center_px"]
        center = (int(round(cx)), int(round(cy)))
        cv2.circle(canvas, center, 7, color, -1, lineType=cv2.LINE_AA)
        angle = float(flake["final_angle_deg"])
        length = max(25.0, math.sqrt(float(flake["area"])) * 0.35)
        theta = math.radians(angle)
        p1 = (int(round(cx - math.cos(theta) * length)), int(round(cy - math.sin(theta) * length)))
        p2 = (int(round(cx + math.cos(theta) * length)), int(round(cy + math.sin(theta) * length)))
        cv2.line(canvas, p1, p2, color, 3, lineType=cv2.LINE_AA)
        rel = flake.get("angle_relative_to_tray_deg")
        rel_text = f" rel {float(rel):+.1f}" if rel is not None else ""
        status_text = "" if chip_status == "ok" else f" {chip_status}"
        label = f"chip {flake['idx']}{status_text} {angle:.1f} deg{rel_text}"
        if show_chip_labels:
            cv2.putText(canvas, label, (center[0] + 10, center[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.82, color, 3, cv2.LINE_AA)

    angle_text = (
        f"tray angle: {result['tray_angle_deg']:.3f} deg | "
        f"markers: {result['detected_marker_count']} | "
        f"missing: {result['grid']['missing_slot_count']} | "
        f"occluded: {result['grid'].get('occluded_slot_count', 0)} | "
        f"occupied: {result['grid'].get('occupied_slot_count', 0)} | "
        f"warn: {result['grid'].get('warning_slot_count', 0)} | "
        f"bad: {result['grid'].get('abnormal_slot_count', 0)} | "
        f"off-grid: {result['grid'].get('off_grid_chip_count', 0)} | "
        f"flakes: {result.get('flake_detection', {}).get('count', 0)}"
    )
    text_size, _baseline = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.90, 3)
    cv2.rectangle(canvas, (12, 12), (min(canvas.shape[1] - 12, text_size[0] + 42), 64), (0, 0, 0), -1)
    cv2.putText(canvas, angle_text, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 255, 255), 3, cv2.LINE_AA)
    return canvas


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(result: dict[str, Any]) -> None:
    print(f"Detected markers: {result['detected_marker_count']}")
    print(f"Tray angle: {result['tray_angle_deg']:.3f} deg")
    print(f"Tray angle source: {result['grid'].get('angle_source', 'unknown')}")
    print(f"Pairwise center angle: {result['pairwise_angle_deg']:.3f} deg")
    print(f"Visible slots: {result['grid']['visible_slot_count']}/{result['grid']['rows'] * result['grid']['cols']}")
    if result["grid"].get("visible_unread_slot_count", 0):
        unread = [f"R{x['row']}C{x['col']} id={x['id']}" for x in result.get("visible_unread_slots", [])]
        print("Visible marker pattern but unread ID: " + ", ".join(unread))
    if result["grid"].get("occupied_slot_count", 0):
        occupied = [
            f"R{x['row']}C{x['col']} id={x['id']} flake={x['matched_flake_idx']} status={x.get('chip_status', 'ok')}"
            for x in result["occupied_slots"]
        ]
        print("Occupied chip/flake-covered slot markers: " + ", ".join(occupied))
    if result["grid"].get("warning_slot_count", 0):
        warnings = [f"R{x['row']}C{x['col']} id={x['id']} flags={','.join(x.get('chip_flags', []))}" for x in result["warning_slots"]]
        print("Warning chip slots: " + ", ".join(warnings))
    if result["grid"].get("abnormal_slot_count", 0):
        abnormal = [f"R{x['row']}C{x['col']} id={x['id']} flags={','.join(x.get('chip_flags', []))}" for x in result["abnormal_slots"]]
        print("Abnormal chip slots: " + ", ".join(abnormal))
    if result["grid"].get("silver_gap_obstruction_count", 0):
        gaps = [
            f"{x.get('type')} {x.get('between_slots')} -> R{x.get('applied_slot', x.get('assigned_slot'))[0]}C{x.get('applied_slot', x.get('assigned_slot'))[1]}"
            for x in result.get("silver_gap_obstructions", [])
            if x.get("applied_slot") or x.get("assigned_slot")
        ]
        print("Blocked silver inter-slot gaps: " + ", ".join(gaps))
    if result["grid"].get("off_grid_chip_count", 0):
        off_grid = [f"chip {x['idx']} nearest={x.get('nearest_slot')} flags={','.join(x.get('chip_flags', []))}" for x in result["off_grid_chips"]]
        print("Off-grid chip candidates: " + ", ".join(off_grid))
    if result["grid"].get("extra_chip_count", 0):
        extra = [f"chip {x['idx']} nearest={x.get('nearest_slot')} flags={','.join(x.get('chip_flags', []))}" for x in result["extra_chips"]]
        print("Extra chip candidates near a slot: " + ", ".join(extra))
    if result["grid"].get("occluded_slot_count", 0):
        occluded = [f"R{x['row']}C{x['col']} id={x['id']} reason={x['occlusion_reason']}" for x in result["occluded_slots"]]
        print("Occluded/unknown slot markers: " + ", ".join(occluded))
    if result["missing_slots"]:
        missing = [f"R{x['row']}C{x['col']} id={x['id']}" for x in result["missing_slots"]]
        print("Missing/unexplained slot markers: " + ", ".join(missing))
    if result.get("flakes"):
        print("Chip/flake candidates:")
        for flake in result["flakes"]:
            cx, cy = flake["center_px"]
            rel = flake.get("angle_relative_to_tray_deg")
            rel_text = f" rel={rel:+.1f}" if rel is not None else ""
            print(
                f"  chip {flake['idx']}: center=({cx:.1f}, {cy:.1f}) "
                f"area={flake['area']:.1f} angle={flake['final_angle_deg']:.1f} "
                f"{rel_text} status={flake.get('chip_status', 'unassigned')} "
                f"flags={','.join(flake.get('chip_flags', [])) or '-'} "
                f"kind={flake.get('kind', 'chip')} source={flake['angle_source']}"
            )
    print("Slot marker ID grid:")
    for row in result["grid"]["slot_id_grid"]:
        print("  " + " ".join(f"{int(x):>3}" if x is not None else "  ." for x in row))
    if result["locator_ids"]:
        print("Locator marker IDs: " + ", ".join(str(x) for x in result["locator_ids"]))


def load_layout(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_hsv_triplet(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("HSV values must be H,S,V")
    try:
        hsv = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("HSV values must be integers") from exc
    if not all(0 <= item <= 255 for item in hsv):
        raise argparse.ArgumentTypeError("HSV values must be in 0..255")
    return hsv  # type: ignore[return-value]


def ratio_from_percent(value: float) -> float:
    number = float(value)
    if number > 1.0:
        number /= 100.0
    return max(0.0, min(0.95, number))


def analyze_static_image(args: argparse.Namespace) -> int:
    image_path = Path(args.image)
    image = read_image(image_path)

    layout = load_layout(Path(args.layout)) if args.layout else None
    dictionary_name = layout.get("dictionary", args.dictionary) if layout else args.dictionary
    result = analyze_image(
        image,
        image_path=str(image_path),
        dictionary_name=dictionary_name,
        rows=args.rows,
        cols=args.cols,
        layout=layout,
        occlusion_bottom_ratio=ratio_from_percent(args.occlusion_bottom),
        detect_flakes=not args.no_flake_detect,
        flake_lower_hsv=args.flake_lower_hsv,
        flake_upper_hsv=args.flake_upper_hsv,
        flake_min_area=args.flake_min_area,
        flake_max_aspect=args.flake_max_aspect,
        edge_occlusion_margin_ratio=args.edge_occlusion_margin,
        use_fixed_arm_mask=not args.no_fixed_arm_mask,
    )
    print_summary(result)

    if args.save_json:
        write_json(Path(args.save_json), result)
        print(f"Wrote analysis JSON: {args.save_json}")
    if args.save_layout:
        layout_payload = make_layout(result)
        write_json(Path(args.save_layout), layout_payload)
        print(f"Wrote tray layout JSON: {args.save_layout}")
    if args.annotate:
        annotated = draw_result(image, result)
        out_path = Path(args.annotate)
        write_image(out_path, annotated)
        print(f"Wrote annotated image: {args.annotate}")
    if args.show:
        annotated = draw_result(image, result)
        cv2.imshow("tray marker detector", annotated)
        cv2.waitKey(0)
    return 0


def camera_backend(name: str) -> int:
    if name == "auto":
        return 0
    if not hasattr(cv2, name):
        raise ValueError(f"unknown cv2 backend {name!r}")
    return int(getattr(cv2, name))


def analyze_camera(args: argparse.Namespace) -> int:
    backend = camera_backend(args.backend)
    if backend:
        cap = cv2.VideoCapture(args.camera, backend)
    else:
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {args.camera}")

    layout = load_layout(Path(args.layout)) if args.layout else None
    dictionary_name = layout.get("dictionary", args.dictionary) if layout else args.dictionary
    last_print = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            try:
                result = analyze_image(
                    frame,
                    image_path=f"camera:{args.camera}",
                    dictionary_name=dictionary_name,
                    rows=args.rows,
                    cols=args.cols,
                    layout=layout,
                    occlusion_bottom_ratio=ratio_from_percent(args.occlusion_bottom),
                    detect_flakes=not args.no_flake_detect,
                    flake_lower_hsv=args.flake_lower_hsv,
                    flake_upper_hsv=args.flake_upper_hsv,
                    flake_min_area=args.flake_min_area,
                    flake_max_aspect=args.flake_max_aspect,
                    edge_occlusion_margin_ratio=args.edge_occlusion_margin,
                    use_fixed_arm_mask=not args.no_fixed_arm_mask,
                )
                annotated = draw_result(frame, result)
                now = time.monotonic()
                if now - last_print > args.print_interval:
                    print(
                        f"angle={result['tray_angle_deg']:.3f} deg, "
                        f"markers={result['detected_marker_count']}, "
                        f"visible_slots={result['grid']['visible_slot_count']}, "
                        f"missing={result['grid']['missing_slot_count']}, "
                        f"occluded={result['grid'].get('occluded_slot_count', 0)}, "
                        f"occupied={result['grid'].get('occupied_slot_count', 0)}, "
                        f"warn={result['grid'].get('warning_slot_count', 0)}, "
                        f"bad={result['grid'].get('abnormal_slot_count', 0)}, "
                        f"off_grid={result['grid'].get('off_grid_chip_count', 0)}, "
                        f"flakes={result.get('flake_detection', {}).get('count', 0)}"
                    )
                    last_print = now
            except Exception as exc:
                annotated = frame.copy()
                cv2.putText(
                    annotated,
                    str(exc),
                    (24, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow("tray marker detector", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Path to a still image to analyze.")
    source.add_argument("--camera", type=int, help="OpenCV camera index for live detection.")
    parser.add_argument("--dictionary", default=DEFAULT_DICT, help=f"ArUco dictionary name, default {DEFAULT_DICT}.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help=f"Slot grid rows, default {DEFAULT_ROWS}.")
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS, help=f"Slot grid columns, default {DEFAULT_COLS}.")
    parser.add_argument("--layout", help="Existing tray layout JSON from an empty-tray calibration.")
    parser.add_argument("--save-layout", help="Write an empty-tray layout JSON containing the slot ID grid.")
    parser.add_argument("--save-json", help="Write full detection and grid analysis JSON.")
    parser.add_argument("--annotate", help="Write an annotated image.")
    parser.add_argument("--show", action="store_true", help="Show an OpenCV preview window.")
    parser.add_argument("--backend", default="auto", help="Camera backend name, or auto.")
    parser.add_argument("--print-interval", type=float, default=0.5, help="Seconds between live camera status prints.")
    parser.add_argument("--occlusion-bottom", type=float, default=DEFAULT_OCCLUSION_BOTTOM_RATIO, help="Optional extra bottom occlusion band as fraction or percent. Default is off; the fixed arm polygon is used instead.")
    parser.add_argument("--no-fixed-arm-mask", action="store_true", help="Disable the built-in fixed bottom-center robot-arm occlusion polygon.")
    parser.add_argument("--edge-occlusion-margin", type=float, default=DEFAULT_EDGE_OCCLUSION_MARGIN_RATIO, help="Image-edge occlusion margin as a fraction of grid spacing.")
    parser.add_argument("--no-flake-detect", action="store_true", help="Disable chip/flake detection.")
    parser.add_argument("--flake-lower-hsv", type=parse_hsv_triplet, default=DEFAULT_LOWER_PURPLE, help="Lower purple HSV threshold H,S,V.")
    parser.add_argument("--flake-upper-hsv", type=parse_hsv_triplet, default=DEFAULT_UPPER_PURPLE, help="Upper purple HSV threshold H,S,V.")
    parser.add_argument("--flake-min-area", type=float, default=DEFAULT_FLAKE_MIN_AREA, help="Minimum global purple candidate contour area in pixels. Slot-local dark chips can use a lower derived threshold.")
    parser.add_argument("--flake-max-aspect", type=float, default=DEFAULT_FLAKE_MAX_ASPECT, help="Maximum long/short side ratio for chip/flake candidates.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.image:
        return analyze_static_image(args)
    return analyze_camera(args)


if __name__ == "__main__":
    raise SystemExit(main())
