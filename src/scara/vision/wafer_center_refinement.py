"""Fail-closed full-contour centre refinement for an expanded wafer ROI.

The regular slot analysis can only see the part of an outside-slot wafer that
falls inside a slot patch.  This helper is intended for a larger, tray-aligned
patch around that preliminary detection.  It deliberately follows the seed
component instead of selecting the largest chromatic object in the patch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from .wafer_shape_quality import (
    DEFAULT_WAFER_QUALITY,
    WaferQualityConfig,
    _wafer_mask,
    fit_wafer_quadrilateral,
)


Point = tuple[float, float]


@dataclass(frozen=True)
class WaferGeometryRefinement:
    """Result of seed-guided geometry refinement in expanded-patch pixels."""

    success: bool
    reason: str
    center_patch_px: Optional[Point]
    box_patch_px: tuple[Point, ...]
    area_px: float
    short_side_px: float
    long_side_px: float
    seed_distance_px: float
    aspect_ratio: float
    rectangularity: float
    solidity: float
    touches_boundary: bool
    quadrilateral_fit_method: str
    quadrilateral_fit_iou: float
    yaw_deg: Optional[float]

    @classmethod
    def failed(
        cls,
        reason: str,
        *,
        area_px: float = 0.0,
        short_side_px: float = 0.0,
        long_side_px: float = 0.0,
        seed_distance_px: float = 0.0,
        aspect_ratio: float = 0.0,
        rectangularity: float = 0.0,
        solidity: float = 0.0,
        touches_boundary: bool = False,
        quadrilateral_fit_method: str = "none",
        quadrilateral_fit_iou: float = 0.0,
        yaw_deg: Optional[float] = None,
    ) -> "WaferGeometryRefinement":
        return cls(
            success=False,
            reason=str(reason),
            center_patch_px=None,
            box_patch_px=(),
            area_px=float(area_px),
            short_side_px=float(short_side_px),
            long_side_px=float(long_side_px),
            seed_distance_px=float(seed_distance_px),
            aspect_ratio=float(aspect_ratio),
            rectangularity=float(rectangularity),
            solidity=float(solidity),
            touches_boundary=bool(touches_boundary),
            quadrilateral_fit_method=str(quadrilateral_fit_method),
            quadrilateral_fit_iou=float(quadrilateral_fit_iou),
            yaw_deg=None if yaw_deg is None else float(yaw_deg),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "reason": self.reason,
            "center_patch_px": (
                None if self.center_patch_px is None else list(self.center_patch_px)
            ),
            "box_patch_px": [list(point) for point in self.box_patch_px],
            "area_px": self.area_px,
            "short_side_px": self.short_side_px,
            "long_side_px": self.long_side_px,
            "seed_distance_px": self.seed_distance_px,
            "aspect_ratio": self.aspect_ratio,
            "rectangularity": self.rectangularity,
            "solidity": self.solidity,
            "touches_boundary": self.touches_boundary,
            "quadrilateral_fit_method": self.quadrilateral_fit_method,
            "quadrilateral_fit_iou": self.quadrilateral_fit_iou,
            "yaw_deg": self.yaw_deg,
        }


def _distance_to_component(
    component_pixels_yx: tuple[np.ndarray, np.ndarray],
    seed_xy: np.ndarray,
) -> float:
    rows, columns = component_pixels_yx
    if rows.size == 0:
        return math.inf
    squared = (
        np.square(columns.astype(np.float64) - float(seed_xy[0]))
        + np.square(rows.astype(np.float64) - float(seed_xy[1]))
    )
    return float(math.sqrt(float(np.min(squared))))


def refine_wafer_geometry_center(
    patch: np.ndarray,
    seed_patch_px: tuple[float, float],
    config: WaferQualityConfig = DEFAULT_WAFER_QUALITY,
) -> WaferGeometryRefinement:
    """Recover a complete wafer centre from an expanded, tray-aligned patch.

    ``seed_patch_px`` is the preliminary wafer centre expressed in this same
    expanded-patch coordinate system.  The component containing the seed wins;
    otherwise the nearest plausible component wins.  Every geometry gate is
    fail-closed because this result may later be transformed into a robot XY
    target.
    """

    if patch is None or not isinstance(patch, np.ndarray) or patch.size == 0:
        return WaferGeometryRefinement.failed("invalid_patch")
    if patch.ndim not in (2, 3) or (
        patch.ndim == 3 and patch.shape[2] not in (3, 4)
    ):
        return WaferGeometryRefinement.failed("invalid_patch_shape")
    try:
        seed = np.asarray(seed_patch_px, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return WaferGeometryRefinement.failed("invalid_seed")
    if seed.size != 2 or not np.all(np.isfinite(seed)):
        return WaferGeometryRefinement.failed("invalid_seed")

    height, width = patch.shape[:2]
    if height < 8 or width < 8:
        return WaferGeometryRefinement.failed("patch_too_small")
    if not (0.0 <= seed[0] <= float(width - 1) and 0.0 <= seed[1] <= float(height - 1)):
        return WaferGeometryRefinement.failed("seed_outside_patch")

    try:
        mask, purple_mask = _wafer_mask(patch, config)
    except (cv2.error, TypeError, ValueError):
        return WaferGeometryRefinement.failed("mask_generation_failed")

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    patch_area = float(height * width)
    # The expanded ROI may have up to twice the slot-patch span on each axis,
    # so scale the legacy slot-patch lower area gate by one quarter.
    minimum_area_px = max(18.0, 0.25 * float(config.minimum_area_ratio) * patch_area)
    maximum_area_px = float(config.maximum_area_ratio) * patch_area
    candidates: list[tuple[float, float, int]] = []
    seed_column = int(round(float(seed[0])))
    seed_row = int(round(float(seed[1])))
    for label in range(1, component_count):
        component_area = float(stats[label, cv2.CC_STAT_AREA])
        if not minimum_area_px <= component_area <= maximum_area_px:
            continue
        distance = (
            0.0
            if int(labels[seed_row, seed_column]) == label
            else _distance_to_component(np.where(labels == label), seed)
        )
        if math.isfinite(distance):
            candidates.append((distance, -component_area, label))
    if not candidates:
        return WaferGeometryRefinement.failed("no_plausible_chromatic_candidate")

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    seed_distance, _negative_area, selected_label = candidates[0]
    maximum_seed_distance = max(8.0, 0.20 * float(min(height, width)))
    if seed_distance > maximum_seed_distance:
        return WaferGeometryRefinement.failed(
            "seed_too_far_from_candidate",
            seed_distance_px=seed_distance,
        )

    component_mask = np.where(labels == selected_label, 255, 0).astype(np.uint8)
    contours, _hierarchy = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return WaferGeometryRefinement.failed(
            "no_candidate_contour", seed_distance_px=seed_distance
        )
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if not minimum_area_px <= area <= maximum_area_px:
        return WaferGeometryRefinement.failed(
            "candidate_area_out_of_range",
            area_px=area,
            seed_distance_px=seed_distance,
        )

    quadrilateral = fit_wafer_quadrilateral(contour, (height, width))
    if (
        not quadrilateral.success
        or quadrilateral.center_px is None
        or len(quadrilateral.corners_px) != 4
    ):
        return WaferGeometryRefinement.failed(
            quadrilateral.reason,
            area_px=area,
            seed_distance_px=seed_distance,
        )
    center_x, center_y = quadrilateral.center_px
    short_side = float(quadrilateral.short_side_px)
    long_side = float(quadrilateral.long_side_px)

    rectangle_area = max(float(quadrilateral.area_px), 1.0)
    aspect_ratio = long_side / short_side
    rectangularity = min(1.0, area / rectangle_area)
    hull_area = max(abs(float(cv2.contourArea(cv2.convexHull(contour)))), 1.0)
    solidity = area / hull_area
    contour_points = contour.reshape(-1, 2).astype(np.float64)
    boundary_margin = max(
        1.0,
        float(config.slot_boundary_margin_ratio) * float(min(height, width)),
    )
    touches_boundary = bool(
        np.min(contour_points[:, 0]) <= boundary_margin
        or np.max(contour_points[:, 0]) >= float(width - 1) - boundary_margin
        or np.min(contour_points[:, 1]) <= boundary_margin
        or np.max(contour_points[:, 1]) >= float(height - 1) - boundary_margin
    )
    diagnostics = {
        "area_px": area,
        "short_side_px": short_side,
        "long_side_px": long_side,
        "seed_distance_px": seed_distance,
        "aspect_ratio": aspect_ratio,
        "rectangularity": rectangularity,
        "solidity": solidity,
        "touches_boundary": touches_boundary,
        "quadrilateral_fit_method": quadrilateral.method,
        "quadrilateral_fit_iou": quadrilateral.mask_iou,
        "yaw_deg": quadrilateral.yaw_deg,
    }
    if touches_boundary:
        return WaferGeometryRefinement.failed("touches_expanded_roi_boundary", **diagnostics)
    if aspect_ratio > float(config.warning_max_aspect_ratio):
        return WaferGeometryRefinement.failed("aspect_ratio_exceeds_warning_limit", **diagnostics)
    if rectangularity < float(config.warning_min_rectangularity):
        return WaferGeometryRefinement.failed("rectangularity_below_warning_quality", **diagnostics)
    if solidity < float(config.warning_min_solidity):
        return WaferGeometryRefinement.failed("solidity_below_warning_quality", **diagnostics)
    if quadrilateral.mask_iou < float(config.quadrilateral_min_mask_iou):
        return WaferGeometryRefinement.failed(
            "quadrilateral_fit_below_quality", **diagnostics
        )

    contour_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)
    contour_pixels = contour_mask > 0
    chromatic_fraction = (
        float(np.mean(purple_mask[contour_pixels] > 0))
        if np.any(contour_pixels)
        else 0.0
    )
    if chromatic_fraction < float(config.minimum_chromatic_fraction):
        return WaferGeometryRefinement.failed("marker_artifact_rejected", **diagnostics)

    box = np.asarray(quadrilateral.corners_px, dtype=np.float64)
    return WaferGeometryRefinement(
        success=True,
        reason="ok",
        center_patch_px=(float(center_x), float(center_y)),
        box_patch_px=tuple(
            tuple(float(value) for value in point) for point in box
        ),
        area_px=area,
        short_side_px=float(short_side),
        long_side_px=float(long_side),
        seed_distance_px=seed_distance,
        aspect_ratio=float(aspect_ratio),
        rectangularity=float(rectangularity),
        solidity=float(solidity),
        touches_boundary=False,
        quadrilateral_fit_method=quadrilateral.method,
        quadrilateral_fit_iou=float(quadrilateral.mask_iou),
        yaw_deg=(
            None
            if quadrilateral.yaw_deg is None
            else float(quadrilateral.yaw_deg)
        ),
    )


def find_boundary_wafer_fragment_seed(
    patch: np.ndarray,
    config: WaferQualityConfig = DEFAULT_WAFER_QUALITY,
) -> Optional[Point]:
    """Return a seed for a small chromatic fragment clipped by a slot edge.

    This is deliberately narrower than :func:`analyze_wafer_patch`: it is only
    a discovery hint for a later expanded-ROI, full-contour validation.  A
    component must touch the canonical slot boundary, be predominantly purple,
    and be smaller than the normal slot-patch minimum area.  It never makes an
    occupancy or correction decision by itself.
    """

    if patch is None or not isinstance(patch, np.ndarray) or patch.size == 0:
        return None
    if patch.ndim not in (2, 3) or (
        patch.ndim == 3 and patch.shape[2] not in (3, 4)
    ):
        return None
    height, width = patch.shape[:2]
    if height < 8 or width < 8:
        return None
    try:
        mask, purple_mask = _wafer_mask(patch, config)
    except (cv2.error, TypeError, ValueError):
        return None

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    patch_area = float(height * width)
    minimum_fragment_area = max(18.0, 0.015 * patch_area)
    maximum_fragment_area = float(config.minimum_area_ratio) * patch_area
    boundary_margin = max(
        1.0,
        float(config.slot_boundary_margin_ratio) * float(min(height, width)),
    )
    candidates: list[tuple[float, int, Point]] = []
    for label in range(1, component_count):
        component_area = float(stats[label, cv2.CC_STAT_AREA])
        if not minimum_fragment_area <= component_area < maximum_fragment_area:
            continue
        component_mask = np.where(labels == label, 255, 0).astype(np.uint8)
        contours, _hierarchy = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        contour_points = contour.reshape(-1, 2).astype(np.float64)
        touches_boundary = bool(
            np.min(contour_points[:, 0]) <= boundary_margin
            or np.max(contour_points[:, 0]) >= float(width - 1) - boundary_margin
            or np.min(contour_points[:, 1]) <= boundary_margin
            or np.max(contour_points[:, 1]) >= float(height - 1) - boundary_margin
        )
        if not touches_boundary:
            continue
        filled = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(filled, [contour], -1, 255, -1)
        pixels = filled > 0
        chromatic_fraction = (
            float(np.mean(purple_mask[pixels] > 0)) if np.any(pixels) else 0.0
        )
        if chromatic_fraction < float(config.minimum_chromatic_fraction):
            continue
        (center_x, center_y), _size, _angle = cv2.minAreaRect(contour)
        candidates.append(
            (
                component_area,
                label,
                (float(center_x), float(center_y)),
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


__all__ = [
    "WaferGeometryRefinement",
    "find_boundary_wafer_fragment_seed",
    "refine_wafer_geometry_center",
]
