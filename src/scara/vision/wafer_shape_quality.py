"""Wafer shape and stacking evidence in a perspective-normalized slot patch.

The thresholds originate from ``tray_marker_detector_v2`` but are expressed
as ratios of a canonical slot patch.  Production entry points load their
``WaferQualityConfig`` from ``src/scara/calib/silicon_detection_0818.json``;
the dataclass defaults remain API/test fallbacks.  Perspective normalization
removes camera zoom and most perspective effects before measuring square
shape, centre offset and yaw.  Black/white ArUco patterns are excluded by
requiring chromatic wafer pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class WaferQualityConfig:
    lower_hsv: tuple[int, int, int] = (110, 20, 20)
    upper_hsv: tuple[int, int, int] = (165, 255, 255)
    dark_value_max: int = 155
    dark_saturation_min: int = 28
    relative_contrast_enabled: bool = True
    relative_lab_lightness_weight: float = 0.22
    relative_lab_distance_min: float = 18.0
    relative_lab_chroma_min: float = 10.0
    relative_reference_max_saturation: int = 90
    relative_reference_min_value: int = 45
    minimum_area_ratio: float = 0.075
    maximum_area_ratio: float = 0.82
    minimum_chromatic_fraction: float = 0.62
    normal_max_aspect_ratio: float = 1.20
    warning_max_aspect_ratio: float = 1.38
    normal_min_rectangularity: float = 0.80
    warning_min_rectangularity: float = 0.64
    normal_min_solidity: float = 0.90
    warning_min_solidity: float = 0.82
    normal_max_center_offset_ratio: float = 0.18
    warning_max_center_offset_ratio: float = 0.32
    normal_max_yaw_deg: float = 8.0
    warning_max_yaw_deg: float = 15.0
    maximum_normal_side_ratio: float = 0.86
    stacking_detection_enabled: bool = False
    stacked_second_component_ratio: float = 0.10
    stacked_internal_line_count: int = 2
    stacked_internal_line_score: float = 0.45
    irregular_outline_vertex_threshold: int = 6
    irregular_outline_max_solidity: float = 0.95
    stacked_second_quadrilateral_ratio: float = 0.18
    stacked_quadrilateral_max_aspect_ratio: float = 1.35
    stacked_quadrilateral_min_rectangularity: float = 0.72
    stacked_quadrilateral_min_solidity: float = 0.86
    stacked_l_min_leg_ratio: float = 0.22
    stacked_l_angle_tolerance_deg: float = 20.0
    slot_boundary_margin_ratio: float = 0.010
    quadrilateral_min_mask_iou: float = 0.70


DEFAULT_WAFER_QUALITY = WaferQualityConfig()


@dataclass(frozen=True)
class WaferObservation:
    found: bool
    quality: str
    center_patch_px: Optional[tuple[float, float]]
    box_patch_px: tuple[tuple[float, float], ...]
    area_ratio: float
    side_ratio: float
    aspect_ratio: float
    rectangularity: float
    solidity: float
    center_offset_ratio: float
    yaw_relative_to_tray_deg: Optional[float]
    polygon_vertices: int
    component_count: int
    second_component_area_ratio: float
    internal_line_count: int
    internal_line_score: float
    chromatic_fraction: float
    confidence: float
    flags: tuple[str, ...]
    outside_slot: bool
    secondary_boxes_patch_px: tuple[tuple[tuple[float, float], ...], ...]
    quadrilateral_fit_method: str
    quadrilateral_fit_iou: float

    @classmethod
    def not_found(
        cls, *flags: str, area_ratio: float = 0.0
    ) -> "WaferObservation":
        return cls(
            found=False,
            quality="none",
            center_patch_px=None,
            box_patch_px=(),
            area_ratio=float(area_ratio),
            side_ratio=0.0,
            aspect_ratio=0.0,
            rectangularity=0.0,
            solidity=0.0,
            center_offset_ratio=0.0,
            yaw_relative_to_tray_deg=None,
            polygon_vertices=0,
            component_count=0,
            second_component_area_ratio=0.0,
            internal_line_count=0,
            internal_line_score=0.0,
            chromatic_fraction=0.0,
            confidence=0.0,
            flags=tuple(flags),
            outside_slot=False,
            secondary_boxes_patch_px=(),
            quadrilateral_fit_method="none",
            quadrilateral_fit_iou=0.0,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "quality": self.quality,
            "center_patch_px": None if self.center_patch_px is None else list(self.center_patch_px),
            "box_patch_px": [list(point) for point in self.box_patch_px],
            "area_ratio": self.area_ratio,
            "side_ratio": self.side_ratio,
            "aspect_ratio": self.aspect_ratio,
            "rectangularity": self.rectangularity,
            "solidity": self.solidity,
            "center_offset_ratio": self.center_offset_ratio,
            "yaw_relative_to_tray_deg": self.yaw_relative_to_tray_deg,
            "polygon_vertices": self.polygon_vertices,
            "component_count": self.component_count,
            "second_component_area_ratio": self.second_component_area_ratio,
            "internal_line_count": self.internal_line_count,
            "internal_line_score": self.internal_line_score,
            "chromatic_fraction": self.chromatic_fraction,
            "confidence": self.confidence,
            "flags": list(self.flags),
            "outside_slot": self.outside_slot,
            "secondary_boxes_patch_px": [
                [list(point) for point in box]
                for box in self.secondary_boxes_patch_px
            ],
            "quadrilateral_fit_method": self.quadrilateral_fit_method,
            "quadrilateral_fit_iou": self.quadrilateral_fit_iou,
        }


@dataclass(frozen=True)
class WaferQuadrilateralFit:
    """Four physical wafer corners recovered from one segmented contour."""

    success: bool
    reason: str
    center_px: Optional[tuple[float, float]]
    corners_px: tuple[tuple[float, float], ...]
    short_side_px: float
    long_side_px: float
    area_px: float
    mask_iou: float
    yaw_deg: Optional[float]
    method: str

    @classmethod
    def failed(cls, reason: str) -> "WaferQuadrilateralFit":
        return cls(
            success=False,
            reason=str(reason),
            center_px=None,
            corners_px=(),
            short_side_px=0.0,
            long_side_px=0.0,
            area_px=0.0,
            mask_iou=0.0,
            yaw_deg=None,
            method="none",
        )


def normalize_square_angle_deg(angle_deg: float) -> float:
    """Normalize a square axis to [-45, 45) degrees."""
    return float((float(angle_deg) + 45.0) % 90.0 - 45.0)


def _local_tray_lab_reference(
    bgr: np.ndarray,
    hsv: np.ndarray,
    config: WaferQualityConfig,
) -> np.ndarray:
    """Estimate neutral tray colour from a robust outer-ring sample.

    The expanded wafer ROI normally contains exposed grey tray around the
    silicon.  Low-saturation, non-black ring pixels form the primary sample;
    the whole patch is only used when the ring is too small.  A median keeps
    marker cells, glare and the wafer itself from steering the reference.
    """

    height, width = bgr.shape[:2]
    ring_width = max(2, int(round(0.12 * float(min(height, width)))))
    ring = np.zeros((height, width), dtype=bool)
    ring[:ring_width, :] = True
    ring[-ring_width:, :] = True
    ring[:, :ring_width] = True
    ring[:, -ring_width:] = True
    neutral = (
        (hsv[:, :, 1] <= int(config.relative_reference_max_saturation))
        & (hsv[:, :, 2] >= int(config.relative_reference_min_value))
    )
    sample_mask = ring & neutral
    minimum_samples = max(64, int(round(0.01 * float(height * width))))
    if int(np.count_nonzero(sample_mask)) < minimum_samples:
        sample_mask = neutral
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    samples = lab[sample_mask]
    if samples.shape[0] < minimum_samples:
        samples = lab.reshape(-1, 3)
    return np.median(samples, axis=0).astype(np.float32)


def _wafer_mask(patch: np.ndarray, config: WaferQualityConfig) -> tuple[np.ndarray, np.ndarray]:
    if patch.ndim == 2:
        bgr = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    elif patch.ndim == 3 and patch.shape[2] == 4:
        bgr = cv2.cvtColor(patch, cv2.COLOR_BGRA2BGR)
    else:
        bgr = patch
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    purple = cv2.inRange(
        hsv,
        np.asarray(config.lower_hsv, dtype=np.uint8),
        np.asarray(config.upper_hsv, dtype=np.uint8),
    )
    dark_chromatic = cv2.inRange(
        hsv,
        np.asarray((0, config.dark_saturation_min, 0), dtype=np.uint8),
        np.asarray((179, 255, config.dark_value_max), dtype=np.uint8),
    )
    if config.relative_contrast_enabled:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        tray_lab = _local_tray_lab_reference(bgr, hsv, config)
        difference = lab - tray_lab.reshape(1, 1, 3)
        lightness = difference[:, :, 0]
        chroma = np.hypot(difference[:, :, 1], difference[:, :, 2])
        weighted_distance = np.sqrt(
            float(config.relative_lab_lightness_weight) * np.square(lightness)
            + np.square(difference[:, :, 1])
            + np.square(difference[:, :, 2])
        )
        relative_chromatic = np.where(
            (
                (weighted_distance >= float(config.relative_lab_distance_min))
                & (chroma >= float(config.relative_lab_chroma_min))
                & (hsv[:, :, 1] >= int(config.dark_saturation_min))
            ),
            255,
            0,
        ).astype(np.uint8)
        # Fixed purple is now only a prior.  The relative term admits silicon
        # whose hue moves under oblique illumination while rejecting neutral
        # marker cells and grey shadows.
        chromatic_support = cv2.bitwise_or(purple, relative_chromatic)
        mask = cv2.bitwise_or(
            purple,
            cv2.bitwise_and(dark_chromatic, chromatic_support),
        )
        mask = cv2.bitwise_or(mask, relative_chromatic)
    else:
        chromatic_support = purple
        mask = cv2.bitwise_or(purple, dark_chromatic)
    scale = max(1.0, min(mask.shape[:2]) / 192.0)
    open_size = max(3, int(round(3.0 * scale)) | 1)
    close_size = max(5, int(round(7.0 * scale)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    return mask, chromatic_support


def _select_components(mask: np.ndarray) -> tuple[np.ndarray, list[tuple[float, np.ndarray]]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask), []
    height, width = mask.shape[:2]
    patch_area = float(height * width)
    patch_center = np.array([0.5 * (width - 1), 0.5 * (height - 1)], dtype=np.float64)
    components: list[tuple[float, float, int]] = []
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < max(18.0, 0.002 * patch_area):
            continue
        distance = float(np.linalg.norm(np.asarray(centroids[label]) - patch_center))
        score = area / (1.0 + 0.018 * distance * distance)
        components.append((score, area, label))
    if not components:
        return np.zeros_like(mask), []
    components.sort(reverse=True)
    primary_label = components[0][2]
    primary_mask = np.where(labels == primary_label, 255, 0).astype(np.uint8)
    details = [(area, np.asarray(centroids[label], dtype=np.float64)) for _score, area, label in components]
    return primary_mask, details


def _edge_axis_angle(contour: np.ndarray, reference_deg: float = 0.0) -> tuple[Optional[float], float]:
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 1e-6:
        return None, 0.0
    approx = cv2.approxPolyDP(contour, max(1.0, 0.012 * perimeter), True).reshape(-1, 2)
    angles: list[float] = []
    weights: list[float] = []
    for start, end in zip(approx, np.roll(approx, -1, axis=0)):
        delta = end.astype(np.float64) - start.astype(np.float64)
        length = float(np.linalg.norm(delta))
        if length < 0.10 * perimeter:
            continue
        angle = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
        angles.append(normalize_square_angle_deg(angle - reference_deg))
        weights.append(length)
    if not angles:
        return None, 0.0
    radians = np.deg2rad(np.asarray(angles, dtype=np.float64) * 4.0)
    weight_array = np.asarray(weights, dtype=np.float64)
    x = float(np.sum(weight_array * np.cos(radians)))
    y = float(np.sum(weight_array * np.sin(radians)))
    if math.hypot(x, y) <= 1e-9:
        return None, 0.0
    relative = math.degrees(math.atan2(y, x)) / 4.0
    concentration = math.hypot(x, y) / max(float(np.sum(weight_array)), 1e-9)
    return normalize_square_angle_deg(reference_deg + relative), concentration


def _cyclic_quadrilateral(points: np.ndarray) -> Optional[np.ndarray]:
    """Return four finite convex vertices in cyclic order."""

    try:
        raw = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if raw.shape != (4, 2) or not np.all(np.isfinite(raw)):
        return None
    hull = cv2.convexHull(raw.astype(np.float32), clockwise=False).reshape(-1, 2)
    if hull.shape != (4, 2):
        return None
    if abs(float(cv2.contourArea(hull.astype(np.float32)))) <= 1.0:
        return None
    return hull.astype(np.float64)


def _quadrilateral_diagonal_center(
    corners: np.ndarray,
) -> Optional[tuple[float, float]]:
    points = _cyclic_quadrilateral(corners)
    if points is None:
        return None
    p0, p1, p2, p3 = points
    first = p2 - p0
    second = p3 - p1
    matrix = np.column_stack((first, -second))
    determinant = float(np.linalg.det(matrix))
    scale = max(1.0, float(np.linalg.norm(first)), float(np.linalg.norm(second)))
    if abs(determinant) <= 1e-9 * scale * scale:
        return None
    factors = np.linalg.solve(matrix, p1 - p0)
    if not np.all(np.isfinite(factors)) or not np.all(
        (-1e-6 <= factors) & (factors <= 1.0 + 1e-6)
    ):
        return None
    center = p0 + float(factors[0]) * first
    return float(center[0]), float(center[1])


def _quadrilateral_mask_iou(
    contour: np.ndarray,
    corners: np.ndarray,
    image_shape: tuple[int, int],
) -> float:
    height, width = image_shape
    contour_mask = np.zeros((height, width), dtype=np.uint8)
    polygon_mask = np.zeros_like(contour_mask)
    cv2.drawContours(contour_mask, [contour.astype(np.int32)], -1, 255, -1)
    cv2.fillConvexPoly(
        polygon_mask,
        np.round(corners).astype(np.int32),
        255,
        cv2.LINE_AA,
    )
    intersection = int(np.count_nonzero((contour_mask > 0) & (polygon_mask > 0)))
    union = int(np.count_nonzero((contour_mask > 0) | (polygon_mask > 0)))
    return float(intersection / union) if union else 0.0


def _projection_quadrilateral(
    contour: np.ndarray,
    yaw_deg: float,
) -> Optional[np.ndarray]:
    """Fit four support lines at the measured wafer edge orientation."""

    points = contour.reshape(-1, 2).astype(np.float64)
    if points.shape[0] < 4:
        return None
    radians = math.radians(float(yaw_deg))
    first_axis = np.asarray([math.cos(radians), math.sin(radians)])
    second_axis = np.asarray([-math.sin(radians), math.cos(radians)])
    first_projection = points @ first_axis
    second_projection = points @ second_axis
    # Robust extrema prevent a few glare-connected pixels from rotating or
    # enlarging the fitted wafer.  Dense contours use quantiles; small clean
    # polygons retain their exact extrema.
    quantile = 0.012 if points.shape[0] >= 40 else 0.0
    first_min, first_max = np.quantile(
        first_projection, [quantile, 1.0 - quantile]
    )
    second_min, second_max = np.quantile(
        second_projection, [quantile, 1.0 - quantile]
    )
    projected = np.asarray(
        [
            first_min * first_axis + second_min * second_axis,
            first_max * first_axis + second_min * second_axis,
            first_max * first_axis + second_max * second_axis,
            first_min * first_axis + second_max * second_axis,
        ],
        dtype=np.float64,
    )
    return _cyclic_quadrilateral(projected)


def fit_wafer_quadrilateral(
    contour: np.ndarray,
    image_shape: tuple[int, int],
) -> WaferQuadrilateralFit:
    """Recover four physical edges without using ``minAreaRect``.

    A convex four-corner approximation is preferred.  If glare leaves extra
    vertices, dominant contour edges define two orthogonal support-line pairs.
    The reported centre is always the intersection of the two diagonals.
    """

    if contour is None:
        return WaferQuadrilateralFit.failed("missing_contour")
    try:
        points = np.asarray(contour).reshape(-1, 2)
    except (TypeError, ValueError):
        return WaferQuadrilateralFit.failed("invalid_contour")
    if points.shape[0] < 4 or not np.all(np.isfinite(points)):
        return WaferQuadrilateralFit.failed("insufficient_contour_points")
    hull = cv2.convexHull(points.astype(np.float32))
    perimeter = float(cv2.arcLength(hull, True))
    if perimeter <= 1e-6:
        return WaferQuadrilateralFit.failed("degenerate_contour")

    candidates: list[tuple[float, str, np.ndarray]] = []
    for epsilon_ratio in (0.004, 0.007, 0.010, 0.014, 0.020, 0.028, 0.040, 0.055):
        approximation = cv2.approxPolyDP(
            hull, max(1.0, epsilon_ratio * perimeter), True
        ).reshape(-1, 2)
        corners = _cyclic_quadrilateral(approximation)
        if corners is None:
            continue
        iou = _quadrilateral_mask_iou(contour, corners, image_shape)
        candidates.append((iou, "convex_hull_four_lines", corners))

    edge_angle, edge_confidence = _edge_axis_angle(hull)
    if edge_angle is not None and edge_confidence >= 0.20:
        projected = _projection_quadrilateral(contour, edge_angle)
        if projected is not None:
            iou = _quadrilateral_mask_iou(contour, projected, image_shape)
            candidates.append((iou, "robust_orthogonal_support_lines", projected))

    if not candidates:
        return WaferQuadrilateralFit.failed("quadrilateral_fit_failed")
    mask_iou, method, corners = max(candidates, key=lambda item: item[0])
    center = _quadrilateral_diagonal_center(corners)
    if center is None:
        return WaferQuadrilateralFit.failed("quadrilateral_diagonals_degenerate")
    edge_vectors = np.roll(corners, -1, axis=0) - corners
    side_lengths = np.linalg.norm(edge_vectors, axis=1)
    if np.min(side_lengths) <= 1.0:
        return WaferQuadrilateralFit.failed("quadrilateral_too_thin")
    paired_lengths = np.asarray(
        [
            0.5 * (side_lengths[0] + side_lengths[2]),
            0.5 * (side_lengths[1] + side_lengths[3]),
        ],
        dtype=np.float64,
    )
    long_edge_index = int(np.argmax(side_lengths))
    long_edge = edge_vectors[long_edge_index]
    yaw = normalize_square_angle_deg(
        math.degrees(math.atan2(float(long_edge[1]), float(long_edge[0])))
    )
    return WaferQuadrilateralFit(
        success=True,
        reason="ok",
        center_px=center,
        corners_px=tuple(
            tuple(float(value) for value in point) for point in corners
        ),
        short_side_px=float(np.min(paired_lengths)),
        long_side_px=float(np.max(paired_lengths)),
        area_px=abs(float(cv2.contourArea(corners.astype(np.float32)))),
        mask_iou=float(mask_iou),
        yaw_deg=float(yaw),
        method=method,
    )


def _internal_line_evidence(
    patch: np.ndarray,
    object_mask: np.ndarray,
    center: tuple[float, float],
    side_px: float,
) -> tuple[int, float, tuple[tuple[float, float, float, float], ...]]:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 28, 96)
    erode_size = max(3, int(round(side_px * 0.045)) | 1)
    interior = cv2.erode(object_mask, np.ones((erode_size, erode_size), np.uint8))
    edges = cv2.bitwise_and(edges, interior)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(8, int(round(0.08 * side_px))),
        minLineLength=max(12, int(round(0.22 * side_px))),
        maxLineGap=max(4, int(round(0.06 * side_px))),
    )
    if lines is None:
        return 0, 0.0, ()
    lengths: list[float] = []
    accepted_lines: list[tuple[float, float, float, float]] = []
    cx, cy = center
    for raw_line in lines:
        x0, y0, x1, y1 = (float(value) for value in raw_line.reshape(4))
        length = math.hypot(x1 - x0, y1 - y0)
        midpoint_distance = math.hypot(0.5 * (x0 + x1) - cx, 0.5 * (y0 + y1) - cy)
        if midpoint_distance <= 0.36 * side_px:
            lengths.append(length)
            accepted_lines.append((x0, y0, x1, y1))
    return (
        len(lengths),
        float(sum(lengths) / max(side_px, 1.0)),
        tuple(accepted_lines),
    )


def _line_intersection(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> Optional[tuple[np.ndarray, float, float]]:
    p = np.asarray(first[:2], dtype=np.float64)
    r = np.asarray(first[2:], dtype=np.float64) - p
    q = np.asarray(second[:2], dtype=np.float64)
    s = np.asarray(second[2:], dtype=np.float64) - q
    cross = float(r[0] * s[1] - r[1] * s[0])
    if abs(cross) <= 1e-6:
        return None
    delta = q - p
    t = float((delta[0] * s[1] - delta[1] * s[0]) / cross)
    u = float((delta[0] * r[1] - delta[1] * r[0]) / cross)
    return p + t * r, t, u


def _l_shaped_overlap_box(
    lines: tuple[tuple[float, float, float, float], ...],
    object_mask: np.ndarray,
    side_px: float,
    config: WaferQualityConfig,
) -> Optional[tuple[tuple[float, float], ...]]:
    """Confirm two distinct internal edges that form a physical L corner.

    Parallel specular boundaries and duplicate Hough fragments cannot satisfy
    this test.  The returned quadrilateral is the inferred second-wafer outline
    used by the camera overlay.
    """
    best: Optional[tuple[float, tuple[tuple[float, float], ...]]] = None
    minimum_leg = float(config.stacked_l_min_leg_ratio) * side_px
    lower_angle = 90.0 - float(config.stacked_l_angle_tolerance_deg)
    upper_angle = 90.0 + float(config.stacked_l_angle_tolerance_deg)
    extension_slack = 0.12
    height, width = object_mask.shape[:2]
    for index, first in enumerate(lines):
        first_vector = np.asarray(first[2:], dtype=np.float64) - np.asarray(first[:2], dtype=np.float64)
        first_length = float(np.linalg.norm(first_vector))
        if first_length < minimum_leg:
            continue
        first_angle = math.degrees(math.atan2(float(first_vector[1]), float(first_vector[0]))) % 180.0
        for second in lines[index + 1 :]:
            second_vector = np.asarray(second[2:], dtype=np.float64) - np.asarray(second[:2], dtype=np.float64)
            second_length = float(np.linalg.norm(second_vector))
            if second_length < minimum_leg:
                continue
            second_angle = math.degrees(math.atan2(float(second_vector[1]), float(second_vector[0]))) % 180.0
            angle_difference = abs(first_angle - second_angle)
            angle_difference = min(angle_difference, 180.0 - angle_difference)
            if not lower_angle <= angle_difference <= upper_angle:
                continue
            intersection = _line_intersection(first, second)
            if intersection is None:
                continue
            corner, first_t, second_t = intersection
            if not (
                -extension_slack <= first_t <= 1.0 + extension_slack
                and -extension_slack <= second_t <= 1.0 + extension_slack
            ):
                continue
            corner_x = int(round(float(corner[0])))
            corner_y = int(round(float(corner[1])))
            if not (0 <= corner_x < width and 0 <= corner_y < height):
                continue
            if object_mask[corner_y, corner_x] == 0:
                continue
            first_endpoints = (np.asarray(first[:2], dtype=np.float64), np.asarray(first[2:], dtype=np.float64))
            second_endpoints = (np.asarray(second[:2], dtype=np.float64), np.asarray(second[2:], dtype=np.float64))
            first_far = max(first_endpoints, key=lambda point: float(np.linalg.norm(point - corner)))
            second_far = max(second_endpoints, key=lambda point: float(np.linalg.norm(point - corner)))
            first_leg = float(np.linalg.norm(first_far - corner))
            second_leg = float(np.linalg.norm(second_far - corner))
            if min(first_leg, second_leg) < minimum_leg:
                continue
            first_direction = (first_far - corner) / max(first_leg, 1e-6)
            second_direction = (second_far - corner) / max(second_leg, 1e-6)
            inferred_side = min(
                0.95 * side_px,
                max(first_leg, second_leg, 0.65 * side_px),
            )
            box = np.asarray(
                [
                    corner,
                    corner + first_direction * inferred_side,
                    corner + (first_direction + second_direction) * inferred_side,
                    corner + second_direction * inferred_side,
                ],
                dtype=np.float64,
            )
            score = min(first_leg, second_leg) + 0.25 * max(first_leg, second_leg)
            serialized = tuple(tuple(float(value) for value in point) for point in box)
            if best is None or score > best[0]:
                best = (score, serialized)
    return None if best is None else best[1]


def _second_quadrilateral_box(
    mask: np.ndarray,
    primary_mask: np.ndarray,
    primary_area: float,
    side_px: float,
    config: WaferQualityConfig,
) -> Optional[tuple[tuple[float, float], ...]]:
    """Return a separate square-like chromatic component, never a thin glare split."""
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 2:
        return None
    overlap_by_label = [
        int(np.count_nonzero((labels == label) & (primary_mask > 0)))
        for label in range(count)
    ]
    primary_label = int(np.argmax(overlap_by_label))
    candidates: list[tuple[float, tuple[tuple[float, float], ...]]] = []
    for label in range(1, count):
        if label == primary_label:
            continue
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area / max(primary_area, 1.0) < config.stacked_second_quadrilateral_ratio:
            continue
        component_mask = np.where(labels == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        component_contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(component_contour))
        rect = cv2.minAreaRect(component_contour)
        raw_width, raw_height = (float(value) for value in rect[1])
        short_side = min(raw_width, raw_height)
        long_side = max(raw_width, raw_height)
        if short_side < 0.24 * side_px:
            continue
        aspect = long_side / max(short_side, 1.0)
        rectangularity = contour_area / max(short_side * long_side, 1.0)
        hull_area = max(float(cv2.contourArea(cv2.convexHull(component_contour))), 1.0)
        solidity = contour_area / hull_area
        if (
            aspect > config.stacked_quadrilateral_max_aspect_ratio
            or rectangularity < config.stacked_quadrilateral_min_rectangularity
            or solidity < config.stacked_quadrilateral_min_solidity
        ):
            continue
        box = cv2.boxPoints(rect).astype(np.float64)
        serialized = tuple(tuple(float(value) for value in point) for point in box)
        candidates.append((area, serialized))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def analyze_wafer_patch(
    patch: np.ndarray,
    config: WaferQualityConfig = DEFAULT_WAFER_QUALITY,
) -> WaferObservation:
    """Detect and grade a wafer in one canonical, tray-aligned slot patch."""
    if patch is None or patch.size == 0:
        return WaferObservation.not_found("empty_patch")
    bgr = patch if patch.ndim == 3 else cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    height, width = bgr.shape[:2]
    patch_area = float(height * width)
    mask, purple_mask = _wafer_mask(bgr, config)
    primary_mask, components = _select_components(mask)
    if not components:
        return WaferObservation.not_found("no_chromatic_candidate")
    contours, _ = cv2.findContours(primary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return WaferObservation.not_found("no_candidate_contour")
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    area_ratio = area / max(patch_area, 1.0)
    if area_ratio < config.minimum_area_ratio or area_ratio > config.maximum_area_ratio:
        return WaferObservation.not_found(
            "candidate_area_out_of_range",
            area_ratio=float(area_ratio),
        )

    quadrilateral = fit_wafer_quadrilateral(contour, (height, width))
    if (
        not quadrilateral.success
        or quadrilateral.center_px is None
        or len(quadrilateral.corners_px) != 4
    ):
        return WaferObservation.not_found(quadrilateral.reason)
    cx, cy = quadrilateral.center_px
    short_side = float(quadrilateral.short_side_px)
    long_side = float(quadrilateral.long_side_px)
    box = np.asarray(quadrilateral.corners_px, dtype=np.float64)
    quadrilateral_area = max(float(quadrilateral.area_px), 1.0)
    rectangularity = min(1.0, area / quadrilateral_area)
    hull_area = max(abs(float(cv2.contourArea(cv2.convexHull(contour)))), 1.0)
    solidity = area / hull_area
    aspect_ratio = long_side / short_side
    side_ratio = math.sqrt(max(area, 1.0)) / max(min(height, width), 1.0)
    patch_center = np.array([0.5 * (width - 1), 0.5 * (height - 1)], dtype=np.float64)
    center_offset_ratio = float(np.linalg.norm(np.array([cx, cy]) - patch_center)) / max(min(height, width), 1.0)
    perimeter = float(cv2.arcLength(contour, True))
    polygon = cv2.approxPolyDP(contour, max(1.0, 0.010 * perimeter), True)
    yaw = float(quadrilateral.yaw_deg or 0.0)

    contour_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)
    contour_pixels = contour_mask > 0
    chromatic_fraction = float(np.mean(purple_mask[contour_pixels] > 0)) if np.any(contour_pixels) else 0.0
    if chromatic_fraction < config.minimum_chromatic_fraction:
        return WaferObservation.not_found("marker_artifact_rejected")

    # TEMPORARILY DISABLED IN PRODUCTION: reflected highlights frequently look
    # like a second wafer.  Keep all stacking helpers and thresholds intact so
    # the detector can be restored by setting stacking_detection_enabled=true.
    second_component_ratio = 0.0
    internal_count = 0
    internal_score = 0.0
    internal_lines: tuple[Any, ...] = ()
    l_shaped_box = None
    second_quadrilateral_box = None
    if config.stacking_detection_enabled:
        second_component_ratio = (
            float(components[1][0] / max(components[0][0], 1.0))
            if len(components) > 1
            else 0.0
        )
        internal_count, internal_score, internal_lines = _internal_line_evidence(
            bgr, contour_mask, (float(cx), float(cy)), max(short_side, long_side)
        )
        l_shaped_box = _l_shaped_overlap_box(
            internal_lines,
            contour_mask,
            max(short_side, long_side),
            config,
        )
        second_quadrilateral_box = _second_quadrilateral_box(
            mask,
            primary_mask,
            area,
            max(short_side, long_side),
            config,
        )
    secondary_boxes: tuple[tuple[tuple[float, float], ...], ...] = ()
    if second_quadrilateral_box is not None:
        secondary_boxes = (second_quadrilateral_box,)
    elif l_shaped_box is not None:
        secondary_boxes = (l_shaped_box,)
    boundary_margin = max(
        1.0,
        float(config.slot_boundary_margin_ratio) * float(min(height, width)),
    )
    contour_points = contour.reshape(-1, 2).astype(np.float64)
    outside_slot = bool(
        np.min(contour_points[:, 0]) <= boundary_margin
        or np.max(contour_points[:, 0]) >= float(width - 1) - boundary_margin
        or np.min(contour_points[:, 1]) <= boundary_margin
        or np.max(contour_points[:, 1]) >= float(height - 1) - boundary_margin
    )

    flags: list[str] = []
    severe = False
    warning = False
    if quadrilateral.mask_iou < float(config.quadrilateral_min_mask_iou):
        flags.append("quadrilateral_fit_unreliable")
        severe = True
    if aspect_ratio > config.warning_max_aspect_ratio:
        flags.append("non_square_aspect")
        severe = True
    elif aspect_ratio > config.normal_max_aspect_ratio:
        flags.append("aspect_borderline")
        warning = True
    if rectangularity < config.warning_min_rectangularity:
        flags.append("low_rectangularity")
        severe = True
    elif rectangularity < config.normal_min_rectangularity:
        flags.append("rectangularity_borderline")
        warning = True
    if solidity < config.warning_min_solidity:
        flags.append("low_solidity")
        severe = True
    elif solidity < config.normal_min_solidity:
        flags.append("solidity_borderline")
        warning = True
    if center_offset_ratio > config.warning_max_center_offset_ratio:
        flags.append("off_slot_center")
        severe = True
    elif center_offset_ratio > config.normal_max_center_offset_ratio:
        flags.append("center_offset_borderline")
        warning = True
    if abs(yaw) > config.warning_max_yaw_deg:
        flags.append("crooked")
        severe = True
    elif abs(yaw) > config.normal_max_yaw_deg:
        flags.append("yaw_borderline")
        warning = True
    if side_ratio > config.maximum_normal_side_ratio:
        flags.append("oversize_footprint")
        severe = True
    if config.stacking_detection_enabled:
        if second_component_ratio >= config.stacked_second_component_ratio:
            flags.append("multiple_components")
            warning = True
        if (
            internal_count >= config.stacked_internal_line_count
            and internal_score >= config.stacked_internal_line_score
        ):
            flags.append("internal_overlap_edges")
            warning = True
        if l_shaped_box is not None:
            flags.append("l_shaped_overlap_corner")
        if second_quadrilateral_box is not None:
            flags.append("second_quadrilateral")
        if secondary_boxes:
            flags.append("stacked_geometry_confirmed")
            severe = True
    if (
        len(polygon) > config.irregular_outline_vertex_threshold
        and solidity < config.irregular_outline_max_solidity
    ):
        flags.append("irregular_outline")
        severe = True
    if outside_slot:
        flags.append("outside_slot")
        severe = True

    quality = "abnormal" if severe else "warning" if warning else "normal"
    shape_score = max(0.0, min(1.0, rectangularity * solidity / max(aspect_ratio, 1.0)))
    pose_score = max(0.0, 1.0 - min(1.0, center_offset_ratio / max(config.warning_max_center_offset_ratio, 1e-6)))
    confidence = 0.65 * shape_score + 0.20 * pose_score + 0.15 * min(1.0, chromatic_fraction)
    return WaferObservation(
        found=True,
        quality=quality,
        center_patch_px=(float(cx), float(cy)),
        box_patch_px=tuple(tuple(float(value) for value in point) for point in box),
        area_ratio=float(area_ratio),
        side_ratio=float(side_ratio),
        aspect_ratio=float(aspect_ratio),
        rectangularity=float(rectangularity),
        solidity=float(solidity),
        center_offset_ratio=float(center_offset_ratio),
        yaw_relative_to_tray_deg=float(yaw),
        polygon_vertices=int(len(polygon)),
        component_count=int(len(components)),
        second_component_area_ratio=float(second_component_ratio),
        internal_line_count=int(internal_count),
        internal_line_score=float(internal_score),
        chromatic_fraction=float(chromatic_fraction),
        confidence=float(max(0.0, min(1.0, confidence))),
        flags=tuple(flags),
        outside_slot=outside_slot,
        secondary_boxes_patch_px=secondary_boxes,
        quadrilateral_fit_method=quadrilateral.method,
        quadrilateral_fit_iou=float(quadrilateral.mask_iou),
    )


__all__ = [
    "DEFAULT_WAFER_QUALITY",
    "WaferObservation",
    "WaferQualityConfig",
    "analyze_wafer_patch",
    "normalize_square_angle_deg",
]
