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
from dataclasses import dataclass, replace
from typing import Any, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class WaferQualityConfig:
    lower_hsv: tuple[int, int, int] = (105, 25, 8)
    upper_hsv: tuple[int, int, int] = (175, 255, 255)
    dark_value_max: int = 155
    dark_saturation_min: int = 28
    minimum_area_ratio: float = 0.055
    maximum_area_ratio: float = 0.82
    minimum_chromatic_fraction: float = 0.12
    normal_max_aspect_ratio: float = 1.35
    warning_max_aspect_ratio: float = 1.60
    boundary_max_aspect_ratio: float = 1.20
    normal_min_rectangularity: float = 0.78
    warning_min_rectangularity: float = 0.55
    normal_min_solidity: float = 0.62
    warning_min_solidity: float = 0.50
    normal_max_center_offset_ratio: float = 0.18
    warning_max_center_offset_ratio: float = 0.32
    normal_max_yaw_deg: float = 8.0
    warning_max_yaw_deg: float = 15.0
    maximum_normal_side_ratio: float = 0.62
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
    stacked_candidate_min_overlap_ratio: float = 0.20
    stacked_candidate_max_overlap_ratio: float = 0.92
    stacked_candidate_min_protrusion_px: float = 3.0
    stacked_l_temporal_window_size: int = 5
    stacked_l_temporal_min_support: int = 3
    stacked_l_temporal_max_relative_center_jitter_px: float = 5.0
    stacked_l_temporal_min_pairwise_iou: float = 0.60
    slot_boundary_margin_ratio: float = 0.161


DEFAULT_WAFER_QUALITY = WaferQualityConfig()


# Boundary decisions use canonical-patch pixels so the gates remain independent
# of camera zoom. The patch is 192 px in production. A three-pixel dead band
# prevents exposure-driven contour jitter from flipping a frame at zero, while
# a confirmed violation additionally needs coherent primary-contour support.
BOUNDARY_UNCERTAINTY_PX = 3.0
BOUNDARY_CONTOUR_MIN_DEPTH_PX = 2.0
BOUNDARY_CONTOUR_MIN_SUPPORT_PX = 8
BOUNDARY_CONTOUR_MIN_AREA_RATIO = 0.015


@dataclass(frozen=True)
class SecondaryWaferCandidate:
    """One observed or inferred second-wafer outline with fail-closed gates."""

    source: str
    box_patch_px: tuple[tuple[float, float], ...]
    rectangularity: Optional[float]
    solidity: Optional[float]
    aspect_ratio: Optional[float]
    overlap_ratio: float
    protrusion_depth_px: float
    relative_center_offset_px: tuple[float, float]
    accepted: bool
    rejection_reason: Optional[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "box_patch_px": [list(point) for point in self.box_patch_px],
            "rectangularity": self.rectangularity,
            "solidity": self.solidity,
            "aspect_ratio": self.aspect_ratio,
            "overlap_ratio": self.overlap_ratio,
            "protrusion_depth_px": self.protrusion_depth_px,
            "relative_center_offset_px": list(self.relative_center_offset_px),
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }


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
    minimum_slot_clearance_ratio: Optional[float]
    secondary_boxes_patch_px: tuple[tuple[tuple[float, float], ...], ...]
    minimum_slot_clearance_px: Optional[float] = None
    contour_patch_px: tuple[tuple[float, float], ...] = ()
    contour_outside_depth_px: float = 0.0
    contour_outside_support_px: int = 0
    contour_outside_area_ratio: float = 0.0
    boundary_evidence: str = "unobservable"
    base_projection_clearance_px: Optional[float] = None
    refined_projection_clearance_px: Optional[float] = None
    projection_disagreement_px: Optional[float] = None
    base_boundary_crossed_sides: tuple[str, ...] = ()
    refined_boundary_crossed_sides: tuple[str, ...] = ()
    secondary_candidates: tuple[SecondaryWaferCandidate, ...] = ()

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
            minimum_slot_clearance_ratio=None,
            secondary_boxes_patch_px=(),
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
            "minimum_slot_clearance_ratio": self.minimum_slot_clearance_ratio,
            "minimum_slot_clearance_px": self.minimum_slot_clearance_px,
            "contour_outside_depth_px": self.contour_outside_depth_px,
            "contour_outside_support_px": self.contour_outside_support_px,
            "contour_outside_area_ratio": self.contour_outside_area_ratio,
            "boundary_evidence": self.boundary_evidence,
            "base_projection_clearance_px": self.base_projection_clearance_px,
            "refined_projection_clearance_px": self.refined_projection_clearance_px,
            "projection_disagreement_px": self.projection_disagreement_px,
            "base_boundary_crossed_sides": list(self.base_boundary_crossed_sides),
            "refined_boundary_crossed_sides": list(self.refined_boundary_crossed_sides),
            "secondary_boxes_patch_px": [
                [list(point) for point in box]
                for box in self.secondary_boxes_patch_px
            ],
            "secondary_candidates": [
                candidate.to_json() for candidate in self.secondary_candidates
            ],
        }


def _longest_cyclic_true_run(values: np.ndarray) -> int:
    flags = np.asarray(values, dtype=bool).reshape(-1)
    count = int(flags.size)
    if count == 0 or not np.any(flags):
        return 0
    if np.all(flags):
        return count
    doubled = np.concatenate((flags, flags))
    best = 0
    current = 0
    for value in doubled:
        current = current + 1 if value else 0
        best = max(best, current)
    return min(best, count)


def _boundary_measurements(
    primary_mask: np.ndarray,
    box: np.ndarray,
    *,
    boundary_margin: float,
) -> tuple[float, float, int, float, tuple[str, ...]]:
    """Measure fitted-box and actual-contour evidence at the inner slot edge."""

    height, width = primary_mask.shape[:2]
    lower = float(boundary_margin)
    upper_x = float(width - 1) - lower
    upper_y = float(height - 1) - lower
    clearances = np.asarray(
        [
            np.min(box[:, 0]) - lower,
            upper_x - np.max(box[:, 0]),
            np.min(box[:, 1]) - lower,
            upper_y - np.max(box[:, 1]),
        ],
        dtype=np.float64,
    )
    side_names = ("left", "right", "top", "bottom")
    crossed_sides = tuple(
        name for name, clearance in zip(side_names, clearances) if clearance < 0.0
    )

    dense_contours, _ = cv2.findContours(
        primary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if dense_contours:
        dense = max(dense_contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
        violations = np.column_stack(
            (
                lower - dense[:, 0],
                dense[:, 0] - upper_x,
                lower - dense[:, 1],
                dense[:, 1] - upper_y,
            )
        )
        point_depth = np.maximum(np.max(violations, axis=1), 0.0)
        outside_depth = float(np.max(point_depth)) if point_depth.size else 0.0
        outside_support = _longest_cyclic_true_run(point_depth > 0.0)
    else:
        outside_depth = 0.0
        outside_support = 0

    ys, xs = np.nonzero(primary_mask > 0)
    if xs.size:
        outside_pixels = (
            (xs < lower) | (xs > upper_x) | (ys < lower) | (ys > upper_y)
        )
        outside_area_ratio = float(np.count_nonzero(outside_pixels) / xs.size)
    else:
        outside_area_ratio = 0.0
    return (
        float(np.min(clearances)),
        outside_depth,
        int(outside_support),
        outside_area_ratio,
        crossed_sides,
    )


def reconcile_projection_boundary_evidence(
    primary: WaferObservation,
    *,
    base: Optional[WaferObservation] = None,
    primary_is_refined: bool = False,
    projection_disagreement_px: Optional[float] = None,
) -> WaferObservation:
    """Fail closed when base and marker-refined slot boundaries disagree."""

    if not primary.found:
        return primary
    observations = [primary]
    if base is not None and base.found:
        observations.append(base)
    evidence = [row.boundary_evidence for row in observations]

    def supported_outside(row: WaferObservation) -> bool:
        return bool(
            row.minimum_slot_clearance_px is not None
            and row.minimum_slot_clearance_px < 0.0
            # The 30.912 px boundary falls between raster rows, so a physical
            # two-pixel crossing is measured as 1.912 px. Keep a quarter-pixel
            # raster tolerance without lowering the declared two-pixel gate.
            and row.contour_outside_depth_px
            >= BOUNDARY_CONTOUR_MIN_DEPTH_PX - 0.25
            and row.contour_outside_support_px >= BOUNDARY_CONTOUR_MIN_SUPPORT_PX
        )

    if len(observations) == 1:
        row = observations[0]
        strong_outside = bool(
            row.boundary_evidence == "strong_outside"
            or (
                supported_outside(row)
                and abs(float(row.yaw_relative_to_tray_deg or 0.0)) >= 2.5
            )
        )
    elif primary_is_refined:
        primary_supported = supported_outside(primary)
        base_supported = bool(base is not None and supported_outside(base))
        if primary_supported:
            # The marker-grid-refined projection is the normal read-only image
            # path. Coherent contour support is sufficient
            # even when the fitted corner penetration is inside the dead band.
            strong_outside = True
        elif base_supported:
            # The unrefined PnP patch is an alternate safeguard, not an equal
            # authority. It may overrule a refined inside result only when the
            # wafer is visibly rotated, which supports a real corner crossing
            # while rejecting near-axis-aligned projection conflicts.
            yaw = abs(float(primary.yaw_relative_to_tray_deg or 0.0))
            strong_outside = bool(
                yaw >= 4.0
                and (
                    base.boundary_evidence == "strong_outside"
                    or primary.boundary_evidence != "inside"
                )
            )
        else:
            strong_outside = False
    else:
        outside_support = [supported_outside(row) for row in observations]
        strong_outside = all(outside_support)
        if not strong_outside and any(outside_support):
            outside_index = outside_support.index(True)
            outside_row = observations[outside_index]
            other_row = observations[1 - outside_index]
            if other_row.boundary_evidence != "inside":
                # Two uncertain projections become coherent evidence when one
                # has a real contour crossing and the other is not confidently
                # inside. This recovers shallow but stable crossings.
                strong_outside = True
            elif outside_row.boundary_evidence == "strong_outside":
                crossing_depth = -float(outside_row.minimum_slot_clearance_px or 0.0)
                inside_margin = float(other_row.minimum_slot_clearance_px or 0.0)
                # A strong crossing may overrule the other projection only when
                # it exceeds that inside margin by two canonical pixels. This
                # rejects the pattern where one extrapolated boundary crosses
                # but the other remains comfortably inside.
                strong_outside = crossing_depth >= inside_margin + 2.0
    base_required_and_missing = bool(
        primary_is_refined and (base is None or not base.found)
    )
    confident_inside = bool(
        evidence
        and all(value == "inside" for value in evidence)
        and not base_required_and_missing
    )
    resolved_evidence = (
        "strong_outside" if strong_outside else "inside" if confident_inside else "uncertain"
    )

    flags = list(primary.flags)
    if base is not None and base.found and base.boundary_evidence != primary.boundary_evidence:
        if "projection_boundary_disagreement" not in flags:
            flags.append("projection_boundary_disagreement")
    if strong_outside:
        if "outside_slot" not in flags:
            flags.append("outside_slot")
        flags = [flag for flag in flags if flag != "boundary_uncertain"]
    elif not confident_inside:
        flags = [flag for flag in flags if flag != "outside_slot"]
        if "boundary_uncertain" not in flags:
            flags.append("boundary_uncertain")

    clearances_px = [
        float(row.minimum_slot_clearance_px)
        for row in observations
        if row.minimum_slot_clearance_px is not None
    ]
    clearances_ratio = [
        float(row.minimum_slot_clearance_ratio)
        for row in observations
        if row.minimum_slot_clearance_ratio is not None
    ]
    quality = primary.quality
    if strong_outside:
        quality = "abnormal"
    elif not confident_inside and quality == "normal":
        quality = "warning"

    base_observation = base if primary_is_refined else primary
    refined_observation = primary if primary_is_refined else None
    return replace(
        primary,
        quality=quality,
        flags=tuple(flags),
        outside_slot=strong_outside,
        minimum_slot_clearance_px=(min(clearances_px) if clearances_px else None),
        minimum_slot_clearance_ratio=(
            min(clearances_ratio) if clearances_ratio else None
        ),
        contour_outside_depth_px=max(
            float(row.contour_outside_depth_px) for row in observations
        ),
        contour_outside_support_px=max(
            int(row.contour_outside_support_px) for row in observations
        ),
        contour_outside_area_ratio=max(
            float(row.contour_outside_area_ratio) for row in observations
        ),
        boundary_evidence=resolved_evidence,
        base_projection_clearance_px=(
            None
            if base_observation is None
            else base_observation.minimum_slot_clearance_px
        ),
        refined_projection_clearance_px=(
            None
            if refined_observation is None
            else refined_observation.minimum_slot_clearance_px
        ),
        projection_disagreement_px=projection_disagreement_px,
        base_boundary_crossed_sides=(
            ()
            if base_observation is None
            else base_observation.base_boundary_crossed_sides
            or base_observation.refined_boundary_crossed_sides
        ),
        refined_boundary_crossed_sides=(
            ()
            if refined_observation is None
            else refined_observation.refined_boundary_crossed_sides
            or refined_observation.base_boundary_crossed_sides
        ),
    )


def normalize_square_angle_deg(angle_deg: float) -> float:
    """Normalize a square axis to [-45, 45) degrees."""
    return float((float(angle_deg) + 45.0) % 90.0 - 45.0)


def _wafer_mask(patch: np.ndarray, config: WaferQualityConfig) -> tuple[np.ndarray, np.ndarray]:
    if patch.ndim == 2:
        bgr = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    elif patch.ndim == 3 and patch.shape[2] == 4:
        bgr = cv2.cvtColor(patch, cv2.COLOR_BGRA2BGR)
    else:
        bgr = patch
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    preferred_colour = cv2.inRange(
        hsv,
        np.asarray(config.lower_hsv, dtype=np.uint8),
        np.asarray(config.upper_hsv, dtype=np.uint8),
    )
    saturation = hsv[:, :, 1]
    strongly_chromatic = cv2.inRange(
        saturation,
        max(45, int(config.dark_saturation_min)),
        255,
    )
    dark_chromatic = cv2.inRange(
        hsv,
        np.asarray((0, config.dark_saturation_min, 0), dtype=np.uint8),
        np.asarray((179, 255, config.dark_value_max), dtype=np.uint8),
    )
    # Preferred purple pixels are the geometry seed. Strong saturation is a
    # fallback for bright reflections whose hue shifts under auto exposure;
    # the low-value branch keeps genuinely dark wafer areas. Black/white
    # ArUco patterns are rejected later by colour support and size gates.
    mask = cv2.bitwise_or(preferred_colour, strongly_chromatic)
    mask = cv2.bitwise_or(mask, dark_chromatic)
    scale = max(1.0, min(mask.shape[:2]) / 192.0)
    open_size = max(3, int(round(3.0 * scale)) | 1)
    close_size = max(7, int(round(9.0 * scale)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    preferred_colour = cv2.morphologyEx(
        preferred_colour,
        cv2.MORPH_OPEN,
        np.ones((open_size, open_size), np.uint8),
    )
    preferred_colour = cv2.morphologyEx(
        preferred_colour,
        cv2.MORPH_CLOSE,
        np.ones((close_size, close_size), np.uint8),
    )
    return mask, preferred_colour


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


def _robust_oriented_box(
    object_mask: np.ndarray,
    contour: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float], float, float]:
    """Fit a trimmed rectangle to colour support instead of contour outliers."""

    ys, xs = np.where(object_mask > 0)
    if xs.size < 4:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.float64)
        raw_width, raw_height = (float(value) for value in rect[1])
        edge = box[1] - box[0]
        angle = normalize_square_angle_deg(
            math.degrees(math.atan2(float(edge[1]), float(edge[0])))
        )
        return (
            box,
            (float(rect[0][0]), float(rect[0][1])),
            (raw_width, raw_height),
            float(angle),
            0.0,
        )

    points = np.column_stack((xs, ys)).astype(np.float64)
    edge_angle, edge_confidence = _edge_axis_angle(contour)
    if edge_angle is None or edge_confidence < 0.42:
        initial_box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float64)
        edge = initial_box[1] - initial_box[0]
        edge_angle = normalize_square_angle_deg(
            math.degrees(math.atan2(float(edge[1]), float(edge[0])))
        )
    origin = np.median(points, axis=0)
    radians = math.radians(float(edge_angle))
    image_to_local = np.asarray(
        [
            [math.cos(radians), math.sin(radians)],
            [-math.sin(radians), math.cos(radians)],
        ],
        dtype=np.float64,
    )
    local = (points - origin) @ image_to_local.T
    low = np.quantile(local, 0.02, axis=0)
    high = np.quantile(local, 0.98, axis=0)
    local_box = np.asarray(
        [
            [low[0], low[1]],
            [high[0], low[1]],
            [high[0], high[1]],
            [low[0], high[1]],
        ],
        dtype=np.float64,
    )
    box = local_box @ image_to_local + origin
    center = np.mean(box, axis=0)
    size = high - low
    return (
        box,
        (float(center[0]), float(center[1])),
        (float(size[0]), float(size[1])),
        float(normalize_square_angle_deg(float(edge_angle))),
        float(edge_confidence),
    )


def _bridge_split_geometry_mask(
    primary_mask: np.ndarray,
    *,
    patch_area: float,
    config: WaferQualityConfig,
) -> tuple[np.ndarray, bool]:
    """Break a thin colour bridge only when square geometry clearly improves."""

    contours, _ = cv2.findContours(
        primary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return primary_mask, False
    original_contour = max(contours, key=cv2.contourArea)
    original_area = float(cv2.contourArea(original_contour))
    original_box = _robust_oriented_box(
        primary_mask, original_contour
    )[0:3]
    _box, original_center, original_size = original_box
    original_short = min(original_size)
    original_long = max(original_size)
    if original_short <= 1.0:
        return primary_mask, False

    scale = max(1.0, min(primary_mask.shape[:2]) / 192.0)
    split_size = max(7, int(round(9.0 * scale)) | 1)
    opened = cv2.morphologyEx(
        primary_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (split_size, split_size)),
    )
    candidate_mask, _candidate_components = _select_components(opened)
    candidate_contours, _ = cv2.findContours(
        candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not candidate_contours:
        return primary_mask, False
    candidate_contour = max(candidate_contours, key=cv2.contourArea)
    candidate_area = float(cv2.contourArea(candidate_contour))
    if (
        candidate_area < float(config.minimum_area_ratio) * patch_area
        or candidate_area < 0.45 * max(original_area, 1.0)
    ):
        return primary_mask, False
    _candidate_box, candidate_center, candidate_size, _yaw, _confidence = (
        _robust_oriented_box(candidate_mask, candidate_contour)
    )
    candidate_short = min(candidate_size)
    candidate_long = max(candidate_size)
    if candidate_short <= 1.0:
        return primary_mask, False
    original_aspect = original_long / original_short
    candidate_aspect = candidate_long / candidate_short
    patch_center = np.asarray(
        [
            0.5 * (primary_mask.shape[1] - 1),
            0.5 * (primary_mask.shape[0] - 1),
        ],
        dtype=np.float64,
    )
    original_offset = float(
        np.linalg.norm(np.asarray(original_center) - patch_center)
    ) / max(min(primary_mask.shape[:2]), 1.0)
    candidate_offset = float(
        np.linalg.norm(np.asarray(candidate_center) - patch_center)
    ) / max(min(primary_mask.shape[:2]), 1.0)
    geometry_improved = bool(
        candidate_aspect + 0.12 < original_aspect
        and candidate_offset <= original_offset + 0.03
    )
    return (candidate_mask, True) if geometry_improved else (primary_mask, False)


def _marker_like_black_white_pattern(
    bgr: np.ndarray,
    footprint_mask: np.ndarray,
) -> bool:
    """Reject a black/white square whose camera tint happens to be saturated."""

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    pixels = footprint_mask > 0
    if not np.any(pixels):
        return False
    saturation = hsv[:, :, 1][pixels]
    value = hsv[:, :, 2][pixels]
    black_fraction = float(np.mean((value <= 72) & (saturation <= 70)))
    white_fraction = float(np.mean((value >= 155) & (saturation <= 45)))
    return black_fraction >= 0.16 and white_fraction >= 0.06


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
) -> Optional[
    tuple[
        tuple[tuple[float, float], ...],
        float,
        float,
        float,
    ]
]:
    """Return a separate square-like chromatic component, never a thin glare split."""
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 2:
        return None
    overlap_by_label = [
        int(np.count_nonzero((labels == label) & (primary_mask > 0)))
        for label in range(count)
    ]
    primary_label = int(np.argmax(overlap_by_label))
    candidates: list[
        tuple[
            float,
            tuple[tuple[float, float], ...],
            float,
            float,
            float,
        ]
    ] = []
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
        candidates.append(
            (area, serialized, rectangularity, solidity, aspect)
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _area, box, rectangularity, solidity, aspect = candidates[0]
    return box, rectangularity, solidity, aspect


def _secondary_candidate_geometry(
    source: str,
    box: tuple[tuple[float, float], ...],
    primary_box: np.ndarray,
    config: WaferQualityConfig,
    *,
    rectangularity: Optional[float] = None,
    solidity: Optional[float] = None,
    aspect_ratio: Optional[float] = None,
) -> SecondaryWaferCandidate:
    """Measure candidate overlap in canonical pixels and classify it safely."""

    primary = np.asarray(primary_box, dtype=np.float32).reshape(4, 2)
    secondary = np.asarray(box, dtype=np.float32).reshape(4, 2)
    secondary_area = max(abs(float(cv2.contourArea(secondary))), 1.0)
    intersection_area, _intersection = cv2.intersectConvexConvex(
        cv2.convexHull(primary),
        cv2.convexHull(secondary),
    )
    overlap_ratio = float(intersection_area / secondary_area)
    signed_distances = [
        float(
            cv2.pointPolygonTest(
                primary,
                (float(point[0]), float(point[1])),
                True,
            )
        )
        for point in secondary
    ]
    protrusion_depth_px = max(0.0, -min(signed_distances, default=0.0))
    relative_center = np.mean(secondary, axis=0) - np.mean(primary, axis=0)

    rejection_reason: Optional[str] = None
    if overlap_ratio < float(config.stacked_candidate_min_overlap_ratio):
        rejection_reason = "adjacent_slot_interference"
    elif overlap_ratio > float(config.stacked_candidate_max_overlap_ratio):
        rejection_reason = "contained_reflection"
    elif protrusion_depth_px < float(config.stacked_candidate_min_protrusion_px):
        rejection_reason = "insufficient_protrusion"

    return SecondaryWaferCandidate(
        source=str(source),
        box_patch_px=tuple(
            tuple(float(value) for value in point) for point in secondary
        ),
        rectangularity=(
            None if rectangularity is None else float(rectangularity)
        ),
        solidity=None if solidity is None else float(solidity),
        aspect_ratio=None if aspect_ratio is None else float(aspect_ratio),
        overlap_ratio=float(overlap_ratio),
        protrusion_depth_px=float(protrusion_depth_px),
        relative_center_offset_px=(
            float(relative_center[0]),
            float(relative_center[1]),
        ),
        accepted=rejection_reason is None,
        rejection_reason=rejection_reason,
    )


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
    mask, preferred_colour_mask = _wafer_mask(bgr, config)
    primary_mask, components = _select_components(mask)
    if not components:
        return WaferObservation.not_found("no_chromatic_candidate")
    preferred_primary, preferred_components = _select_components(
        preferred_colour_mask
    )
    preferred_area = float(cv2.countNonZero(preferred_primary))
    minimum_preferred_seed_area = max(
        24.0,
        0.40 * float(config.minimum_area_ratio) * patch_area,
    )
    if preferred_components and preferred_area >= minimum_preferred_seed_area:
        # Fit the wafer from its narrow colour seed. The broader mask remains
        # available for low-light fallback and overlap diagnostics.
        primary_mask = preferred_primary
    diagnostic_primary_mask = primary_mask.copy()
    primary_mask, thin_bridge_removed = _bridge_split_geometry_mask(
        primary_mask,
        patch_area=patch_area,
        config=config,
    )
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

    box, (cx, cy), (raw_width, raw_height), yaw, _edge_confidence = (
        _robust_oriented_box(primary_mask, contour)
    )
    short_side = min(float(raw_width), float(raw_height))
    long_side = max(float(raw_width), float(raw_height))
    if short_side <= 1.0:
        return WaferObservation.not_found("candidate_too_thin")
    rect_area = max(short_side * long_side, 1.0)
    hull_area = max(abs(float(cv2.contourArea(cv2.convexHull(contour)))), 1.0)
    # The hull measures the outer silhouette and ignores internal dark
    # reflection bands. Percentile trimming can make the fitted rectangle a
    # fraction smaller than the hull, hence the intentional clipping at 1.
    rectangularity = min(1.0, hull_area / rect_area)
    solidity = area / hull_area
    aspect_ratio = long_side / short_side
    side_ratio = math.sqrt(max(rect_area, 1.0)) / max(min(height, width), 1.0)
    patch_center = np.array([0.5 * (width - 1), 0.5 * (height - 1)], dtype=np.float64)
    center_offset_ratio = float(np.linalg.norm(np.array([cx, cy]) - patch_center)) / max(min(height, width), 1.0)
    if (
        center_offset_ratio > config.warning_max_center_offset_ratio
        and aspect_ratio > config.warning_max_aspect_ratio
    ):
        # A wide context intentionally overlaps neighbouring cells. A thin,
        # remote fragment belongs to that neighbour and must not create a
        # duplicate outside-slot wafer in this cell.
        return WaferObservation.not_found("neighbour_fragment_rejected")
    perimeter = float(cv2.arcLength(contour, True))
    polygon = cv2.approxPolyDP(contour, max(1.0, 0.010 * perimeter), True)
    contour_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)
    footprint_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(
        footprint_mask,
        np.round(box).astype(np.int32),
        255,
        lineType=cv2.LINE_AA,
    )
    footprint_pixels = footprint_mask > 0
    chromatic_fraction = (
        float(np.mean(preferred_colour_mask[footprint_pixels] > 0))
        if np.any(footprint_pixels)
        else 0.0
    )
    if chromatic_fraction < config.minimum_chromatic_fraction:
        return WaferObservation.not_found("marker_artifact_rejected")
    if (
        chromatic_fraction < 0.30
        and _marker_like_black_white_pattern(bgr, footprint_mask)
    ):
        return WaferObservation.not_found("marker_artifact_rejected")

    second_component_ratio = (
        float(components[1][0] / max(components[0][0], 1.0)) if len(components) > 1 else 0.0
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
    second_quadrilateral_raw = _second_quadrilateral_box(
        mask,
        diagnostic_primary_mask,
        area,
        max(short_side, long_side),
        config,
    )
    secondary_candidates_list: list[SecondaryWaferCandidate] = []
    second_quadrilateral_candidate: Optional[SecondaryWaferCandidate] = None
    if second_quadrilateral_raw is not None:
        (
            second_quadrilateral_box,
            second_quadrilateral_rectangularity,
            second_quadrilateral_solidity,
            second_quadrilateral_aspect,
        ) = second_quadrilateral_raw
        second_quadrilateral_candidate = _secondary_candidate_geometry(
            "second_quadrilateral",
            second_quadrilateral_box,
            box,
            config,
            rectangularity=second_quadrilateral_rectangularity,
            solidity=second_quadrilateral_solidity,
            aspect_ratio=second_quadrilateral_aspect,
        )
        secondary_candidates_list.append(second_quadrilateral_candidate)
    l_shaped_candidate: Optional[SecondaryWaferCandidate] = None
    if l_shaped_box is not None:
        l_shaped_candidate = _secondary_candidate_geometry(
            "l_shape",
            l_shaped_box,
            box,
            config,
        )
        secondary_candidates_list.append(l_shaped_candidate)
    accepted_second_quadrilateral = bool(
        second_quadrilateral_candidate is not None
        and second_quadrilateral_candidate.accepted
    )
    accepted_l_shape = bool(
        l_shaped_candidate is not None and l_shaped_candidate.accepted
    )
    secondary_boxes: tuple[tuple[tuple[float, float], ...], ...] = ()
    if accepted_second_quadrilateral:
        assert second_quadrilateral_candidate is not None
        secondary_boxes = (second_quadrilateral_candidate.box_patch_px,)
    elif accepted_l_shape:
        assert l_shaped_candidate is not None
        secondary_boxes = (l_shaped_candidate.box_patch_px,)
    boundary_margin = max(
        1.0,
        float(config.slot_boundary_margin_ratio) * float(min(height, width)),
    )
    (
        minimum_slot_clearance_px,
        contour_outside_depth_px,
        contour_outside_support_px,
        contour_outside_area_ratio,
        boundary_crossed_sides,
    ) = _boundary_measurements(
        primary_mask,
        box,
        boundary_margin=boundary_margin,
    )
    minimum_slot_clearance_ratio = float(
        minimum_slot_clearance_px / max(float(min(height, width)), 1.0)
    )
    boundary_crossing_observed = bool(minimum_slot_clearance_px < 0.0)
    # A partial reflection blob can cross the boundary even though it is not a
    # trustworthy estimate of the square's four corners. Such a frame remains
    # a warning and cannot be selected for pickup, but it is not promoted to a
    # confident outside-slot label. Clear boundary decisions require the
    # fitted footprint itself to remain square-like.
    stacking_boundary_ambiguous = bool(
        internal_count >= config.stacked_internal_line_count
        and internal_score >= config.stacked_internal_line_score
        and len(polygon) > config.irregular_outline_vertex_threshold
        # Specular lines also appear on a clean single wafer.  Only a clearly
        # non-solid merged silhouette makes the layer ownership ambiguous.
        and solidity < 0.85
    )
    boundary_fit_reliable = bool(
        aspect_ratio <= config.boundary_max_aspect_ratio
        and rectangularity >= config.warning_min_rectangularity
        and solidity >= config.warning_min_solidity
        and side_ratio <= config.maximum_normal_side_ratio
        # A merged multi-wafer silhouette does not identify which physical
        # wafer crosses the slot. Keep placement fail-closed until a separate
        # quadrilateral supplies layer-specific evidence.
        and not stacking_boundary_ambiguous
    )
    strong_boundary_crossing = bool(
        boundary_fit_reliable
        and minimum_slot_clearance_px <= -BOUNDARY_UNCERTAINTY_PX
        and contour_outside_depth_px >= BOUNDARY_CONTOUR_MIN_DEPTH_PX
        and contour_outside_support_px >= BOUNDARY_CONTOUR_MIN_SUPPORT_PX
    )
    reliable_contour_crossing = bool(
        contour_outside_depth_px >= BOUNDARY_CONTOUR_MIN_DEPTH_PX
        and contour_outside_support_px >= BOUNDARY_CONTOUR_MIN_SUPPORT_PX
        and contour_outside_area_ratio >= BOUNDARY_CONTOUR_MIN_AREA_RATIO
    )
    boundary_uncertain = bool(
        not strong_boundary_crossing
        and (
            minimum_slot_clearance_px <= BOUNDARY_UNCERTAINTY_PX
            or reliable_contour_crossing
        )
    )
    boundary_evidence = (
        "strong_outside"
        if strong_boundary_crossing
        else "uncertain"
        if boundary_uncertain
        else "inside"
    )
    outside_slot = strong_boundary_crossing

    flags: list[str] = []
    severe = False
    warning = False
    if thin_bridge_removed:
        flags.append("thin_bridge_removed")
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
    if second_component_ratio >= config.stacked_second_component_ratio:
        flags.append("multiple_components")
    if internal_count >= config.stacked_internal_line_count and internal_score >= config.stacked_internal_line_score:
        flags.append("internal_overlap_edges")
    if accepted_l_shape:
        flags.append("l_shaped_overlap_corner")
        flags.append("l_shape_stacking_candidate")
        warning = True
    if accepted_second_quadrilateral:
        flags.append("second_quadrilateral")
        flags.append("stacked_geometry_confirmed")
        severe = True
    for candidate in secondary_candidates_list:
        if candidate.accepted or candidate.rejection_reason is None:
            continue
        flags.append(
            "secondary_candidate_" + candidate.rejection_reason
        )
    if (
        len(polygon) > config.irregular_outline_vertex_threshold
        and solidity < config.irregular_outline_max_solidity
    ):
        flags.append("irregular_outline")
        # Auto-exposure highlights and cable shadows make a colour contour
        # jagged even when the robust square footprint is valid. Complexity is
        # retained as diagnostic evidence; only independently confirmed
        # second-wafer geometry may turn it into a stacking failure.
    if boundary_crossing_observed and not boundary_fit_reliable:
        flags.append("boundary_crossing_unconfirmed")
        warning = True
    if boundary_uncertain:
        flags.append("boundary_uncertain")
        warning = True
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
        minimum_slot_clearance_ratio=minimum_slot_clearance_ratio,
        secondary_boxes_patch_px=secondary_boxes,
        minimum_slot_clearance_px=minimum_slot_clearance_px,
        contour_patch_px=tuple(
            tuple(float(value) for value in point)
            for point in polygon.reshape(-1, 2)
        ),
        contour_outside_depth_px=contour_outside_depth_px,
        contour_outside_support_px=contour_outside_support_px,
        contour_outside_area_ratio=contour_outside_area_ratio,
        boundary_evidence=boundary_evidence,
        base_boundary_crossed_sides=boundary_crossed_sides,
        secondary_candidates=tuple(secondary_candidates_list),
    )


def analyze_dark_wafer_patch(
    patch: np.ndarray,
    config: WaferQualityConfig = DEFAULT_WAFER_QUALITY,
) -> WaferObservation:
    """Conservative fallback for a low-chroma, nearly black wafer.

    This is intentionally not called when the expected ArUco marker decoded.
    It accepts only one square-like dark component near the slot centre and
    rejects a footprint containing the black/white module pattern of a marker.
    """

    if patch is None or patch.size == 0:
        return WaferObservation.not_found("empty_patch")
    bgr = patch if patch.ndim == 3 else cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    patch_area = float(height * width)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_threshold, _ = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    dark_limit = int(max(45, min(132, round(float(otsu_threshold)))))
    dark = cv2.inRange(blurred, 0, dark_limit)
    scale = max(1.0, min(height, width) / 192.0)
    open_size = max(3, int(round(3.0 * scale)) | 1)
    close_size = max(5, int(round(7.0 * scale)) | 1)
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        np.ones((open_size, open_size), np.uint8),
    )
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        np.ones((close_size, close_size), np.uint8),
    )
    primary_mask, components = _select_components(dark)
    if not components:
        return WaferObservation.not_found("no_dark_square_candidate")
    contours, _ = cv2.findContours(
        primary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return WaferObservation.not_found("no_dark_square_contour")
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    area_ratio = area / max(patch_area, 1.0)
    if area_ratio < config.minimum_area_ratio or area_ratio > 0.52:
        return WaferObservation.not_found(
            "dark_square_area_out_of_range", area_ratio=area_ratio
        )
    box, (cx, cy), (raw_width, raw_height), yaw, _edge_confidence = (
        _robust_oriented_box(primary_mask, contour)
    )
    short_side = min(float(raw_width), float(raw_height))
    long_side = max(float(raw_width), float(raw_height))
    if short_side <= 1.0:
        return WaferObservation.not_found("dark_square_too_thin")
    aspect_ratio = long_side / short_side
    rect_area = max(short_side * long_side, 1.0)
    hull_area = max(
        abs(float(cv2.contourArea(cv2.convexHull(contour)))), 1.0
    )
    rectangularity = min(1.0, hull_area / rect_area)
    solidity = area / hull_area
    side_ratio = math.sqrt(rect_area) / max(min(height, width), 1.0)
    patch_center = np.asarray(
        [0.5 * (width - 1), 0.5 * (height - 1)], dtype=np.float64
    )
    center_offset_ratio = float(
        np.linalg.norm(np.asarray([cx, cy]) - patch_center)
    ) / max(min(height, width), 1.0)
    if (
        aspect_ratio > config.normal_max_aspect_ratio
        or rectangularity < config.normal_min_rectangularity
        or solidity < config.normal_min_solidity
        or side_ratio > config.maximum_normal_side_ratio
        or center_offset_ratio > config.warning_max_center_offset_ratio
    ):
        return WaferObservation.not_found("dark_square_geometry_rejected")
    footprint_mask = np.zeros_like(gray)
    cv2.fillConvexPoly(
        footprint_mask,
        np.round(box).astype(np.int32),
        255,
        lineType=cv2.LINE_AA,
    )
    pixels = footprint_mask > 0
    white_fraction = float(np.mean(gray[pixels] >= 150)) if np.any(pixels) else 1.0
    if white_fraction >= 0.14:
        return WaferObservation.not_found("dark_square_marker_pattern_rejected")
    perimeter = float(cv2.arcLength(contour, True))
    polygon = cv2.approxPolyDP(
        contour, max(1.0, 0.010 * perimeter), True
    )
    boundary_margin = max(
        1.0,
        float(config.slot_boundary_margin_ratio) * float(min(height, width)),
    )
    (
        minimum_clearance_px,
        contour_outside_depth_px,
        contour_outside_support_px,
        contour_outside_area_ratio,
        boundary_crossed_sides,
    ) = _boundary_measurements(
        primary_mask,
        box,
        boundary_margin=boundary_margin,
    )
    minimum_clearance = float(
        minimum_clearance_px / max(float(min(height, width)), 1.0)
    )
    outside_slot = bool(
        minimum_clearance_px <= -BOUNDARY_UNCERTAINTY_PX
        and contour_outside_depth_px >= BOUNDARY_CONTOUR_MIN_DEPTH_PX
        and contour_outside_support_px >= BOUNDARY_CONTOUR_MIN_SUPPORT_PX
    )
    reliable_contour_crossing = bool(
        contour_outside_depth_px >= BOUNDARY_CONTOUR_MIN_DEPTH_PX
        and contour_outside_support_px >= BOUNDARY_CONTOUR_MIN_SUPPORT_PX
        and contour_outside_area_ratio >= BOUNDARY_CONTOUR_MIN_AREA_RATIO
    )
    boundary_uncertain = bool(
        not outside_slot
        and (
            minimum_clearance_px <= BOUNDARY_UNCERTAINTY_PX
            or reliable_contour_crossing
        )
    )
    flags = ["dark_low_chroma_fallback"]
    severe = False
    # A low-chroma Otsu square is deliberately only fallback evidence.  It may
    # support diagnostics, but must not by itself authorize a definite wafer
    # occupancy conclusion.
    warning = True
    if center_offset_ratio > config.normal_max_center_offset_ratio:
        flags.append("center_offset_borderline")
        warning = True
    if abs(yaw) > config.warning_max_yaw_deg:
        flags.append("crooked")
        severe = True
    elif abs(yaw) > config.normal_max_yaw_deg:
        flags.append("yaw_borderline")
        warning = True
    if outside_slot:
        flags.append("outside_slot")
        severe = True
    elif boundary_uncertain:
        flags.append("boundary_uncertain")
        warning = True
    quality = "abnormal" if severe else "warning" if warning else "normal"
    confidence = max(
        0.0,
        min(
            0.72,
            0.55 * rectangularity
            + 0.25 * solidity
            + 0.20 * (1.0 - white_fraction),
        ),
    )
    return WaferObservation(
        found=True,
        quality=quality,
        center_patch_px=(float(cx), float(cy)),
        box_patch_px=tuple(
            tuple(float(value) for value in point) for point in box
        ),
        area_ratio=float(area_ratio),
        side_ratio=float(side_ratio),
        aspect_ratio=float(aspect_ratio),
        rectangularity=float(rectangularity),
        solidity=float(solidity),
        center_offset_ratio=float(center_offset_ratio),
        yaw_relative_to_tray_deg=float(yaw),
        polygon_vertices=int(len(polygon)),
        component_count=int(len(components)),
        second_component_area_ratio=0.0,
        internal_line_count=0,
        internal_line_score=0.0,
        chromatic_fraction=0.0,
        confidence=float(confidence),
        flags=tuple(flags),
        outside_slot=outside_slot,
        minimum_slot_clearance_ratio=minimum_clearance,
        secondary_boxes_patch_px=(),
        minimum_slot_clearance_px=minimum_clearance_px,
        contour_patch_px=tuple(
            tuple(float(value) for value in point)
            for point in polygon.reshape(-1, 2)
        ),
        contour_outside_depth_px=contour_outside_depth_px,
        contour_outside_support_px=contour_outside_support_px,
        contour_outside_area_ratio=contour_outside_area_ratio,
        boundary_evidence=(
            "strong_outside" if outside_slot else "uncertain" if boundary_uncertain else "inside"
        ),
        base_boundary_crossed_sides=boundary_crossed_sides,
    )


__all__ = [
    "BOUNDARY_CONTOUR_MIN_DEPTH_PX",
    "BOUNDARY_CONTOUR_MIN_AREA_RATIO",
    "BOUNDARY_CONTOUR_MIN_SUPPORT_PX",
    "BOUNDARY_UNCERTAINTY_PX",
    "DEFAULT_WAFER_QUALITY",
    "SecondaryWaferCandidate",
    "WaferObservation",
    "WaferQualityConfig",
    "analyze_dark_wafer_patch",
    "analyze_wafer_patch",
    "normalize_square_angle_deg",
    "reconcile_projection_boundary_evidence",
]
