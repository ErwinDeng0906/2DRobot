"""Layered tray pose, slot-marker, wafer-quality and occupancy fusion.

The metric board pose remains authoritative.  Slot marker IDs and wafer shape
features are observations attached to those metric slots.  This module never
issues a robot command and never returns a correction when the board pose
fails its existing reprojection quality gates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

from .slot_marker_observation import (
    DEFAULT_CANONICAL_PATCH_SIZE,
    DEFAULT_SLOT_HALF_EXTENT_MM,
    ArucoObservation,
    SlotMarkerEvidence,
    SlotMarkerLayout,
    SlotProjection,
    apply_slot_marker_registration,
    associate_marker_to_slot,
    build_slot_projections,
    detect_aruco_observations,
    estimate_slot_marker_registration,
    patch_points_to_image,
    warp_slot_patch,
)
from .tray_occupancy import (
    DEFAULT_SLOT_DECISION,
    SlotDecision,
    SlotDecisionConfig,
    SlotState,
    decide_slot_state,
)
from .tray_pose_estimator import (
    CameraIntrinsics,
    TrayBoardPoseEstimator,
    TrayPoseEstimate,
)
from .tray_pose_tracker import TrackedTrayPose
from .wafer_shape_quality import (
    DEFAULT_WAFER_QUALITY,
    WaferObservation,
    WaferQualityConfig,
    analyze_wafer_patch,
)
from .wafer_center_refinement import (
    WaferGeometryRefinement,
    find_boundary_wafer_fragment_seed,
    refine_wafer_geometry_center,
)


OUTSIDE_WAFER_REFINEMENT_HALF_EXTENT_MM = 24.0
OUTSIDE_WAFER_REFINEMENT_MINIMUM_IMAGE_COVERAGE = 0.995
OUTSIDE_WAFER_REFINEMENT_MAXIMUM_SEED_SHIFT_MM = 8.0


@dataclass(frozen=True)
class TrayVisionFusionConfig:
    slot_half_extent_mm: float = DEFAULT_SLOT_HALF_EXTENT_MM
    physical_slot_side_length_mm: float = 19.9
    physical_slot_boundary_uncertainty_mm: float = 0.75
    unregistered_slot_boundary_uncertainty_mm: float = 1.5
    canonical_patch_size: int = DEFAULT_CANONICAL_PATCH_SIZE
    wafer_quality: WaferQualityConfig = DEFAULT_WAFER_QUALITY
    slot_decision: SlotDecisionConfig = DEFAULT_SLOT_DECISION


DEFAULT_TRAY_VISION_FUSION = TrayVisionFusionConfig()


@dataclass(frozen=True)
class SlotAnalysis:
    projection: SlotProjection
    marker: SlotMarkerEvidence
    wafer: WaferObservation
    decision: SlotDecision
    wafer_box_image_px: tuple[tuple[float, float], ...]
    wafer_secondary_boxes_image_px: tuple[tuple[tuple[float, float], ...], ...]
    wafer_center_image_px: Optional[tuple[float, float]]
    wafer_center_T_mm: Optional[tuple[float, float, float]]
    wafer_offset_T_mm: Optional[tuple[float, float]]
    wafer_offset_distance_mm: Optional[float]
    explicit_occlusion_ratio: float
    wafer_center_refinement: Optional[dict[str, Any]] = None
    wafer_correction_center_valid: bool = False
    wafer_correction_outside_slot: bool = False
    wafer_correction_center_reason: str = "not_outside_slot"
    slot_boundary_polygon_image_px: tuple[tuple[float, float], ...] = ()
    slot_boundary_polygon_T_mm: tuple[tuple[float, float, float], ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "slot_key": self.projection.slot_key,
            "projection": self.projection.to_json(),
            "marker": self.marker.to_json(),
            "wafer": self.wafer.to_json(),
            "decision": self.decision.to_json(),
            "wafer_box_image_px": [list(point) for point in self.wafer_box_image_px],
            "wafer_secondary_boxes_image_px": [
                [list(point) for point in box]
                for box in self.wafer_secondary_boxes_image_px
            ],
            "wafer_center_image_px": (
                None
                if self.wafer_center_image_px is None
                else list(self.wafer_center_image_px)
            ),
            "wafer_center_T_mm": (
                None
                if self.wafer_center_T_mm is None
                else list(self.wafer_center_T_mm)
            ),
            "wafer_offset_T_mm": (
                None
                if self.wafer_offset_T_mm is None
                else list(self.wafer_offset_T_mm)
            ),
            "wafer_offset_distance_mm": self.wafer_offset_distance_mm,
            "explicit_occlusion_ratio": self.explicit_occlusion_ratio,
            "wafer_center_refinement": self.wafer_center_refinement,
            "wafer_correction_center_valid": self.wafer_correction_center_valid,
            "wafer_correction_outside_slot": self.wafer_correction_outside_slot,
            "wafer_correction_center_reason": self.wafer_correction_center_reason,
            "slot_boundary_polygon_image_px": [
                list(point) for point in self.slot_boundary_polygon_image_px
            ],
            "slot_boundary_polygon_T_mm": [
                list(point) for point in self.slot_boundary_polygon_T_mm
            ],
        }


@dataclass(frozen=True)
class TrayVisionResult:
    success: bool
    quality_passed: bool
    failure_reason: Optional[str]
    coordinate_mapping_allowed: bool
    robot_correction_allowed: bool
    pose: TrayPoseEstimate
    slot_markers: dict[int, ArucoObservation]
    slots: tuple[SlotAnalysis, ...]
    summary: dict[str, int]
    annotated_image: np.ndarray
    slot_projection_registration: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "quality_passed": self.quality_passed,
            "failure_reason": self.failure_reason,
            "coordinate_mapping_allowed": self.coordinate_mapping_allowed,
            "robot_correction_allowed": self.robot_correction_allowed,
            "pose": self.pose.to_json(),
            "slot_markers": {
                str(marker_id): observation.to_json()
                for marker_id, observation in sorted(self.slot_markers.items())
            },
            "slots": [slot.to_json() for slot in self.slots],
            "summary": dict(self.summary),
            "slot_projection_registration": dict(
                self.slot_projection_registration
            ),
        }


def _explicit_occlusion_ratio(
    polygon_px: Sequence[Sequence[float]],
    occlusion_mask: Optional[np.ndarray],
) -> float:
    if occlusion_mask is None:
        return 0.0
    if occlusion_mask.ndim == 3:
        mask = np.max(occlusion_mask, axis=2)
    else:
        mask = occlusion_mask
    polygon = np.asarray(polygon_px, dtype=np.int32).reshape(-1, 2)
    cell_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(cell_mask, polygon, 255, lineType=cv2.LINE_AA)
    cell_pixels = cell_mask > 0
    if not np.any(cell_pixels):
        return 0.0
    return float(np.mean(mask[cell_pixels] > 0))


def image_pixel_to_tray_plane(
    pixel: Sequence[float],
    estimate: TrayPoseEstimate,
    intrinsics: CameraIntrinsics,
    *,
    plane_z_T_mm: float = -2.0,
) -> np.ndarray:
    """Intersect a distorted camera pixel ray with a fixed Tray-frame plane."""
    if not estimate.success or not estimate.quality_passed or estimate.T_T_C is None:
        raise ValueError("quality-passed tray pose is required for pixel mapping")
    point = np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2)
    normalized = cv2.undistortPoints(point, intrinsics.K, intrinsics.dist_coeffs).reshape(2)
    ray_C = np.array([normalized[0], normalized[1], 1.0], dtype=np.float64)
    transform_T_C = np.asarray(estimate.T_T_C, dtype=np.float64).reshape(4, 4)
    origin_T = transform_T_C[:3, 3]
    direction_T = transform_T_C[:3, :3] @ ray_C
    if abs(float(direction_T[2])) < 1e-9:
        raise ValueError("camera ray is parallel to the requested Tray plane")
    distance = (float(plane_z_T_mm) - float(origin_T[2])) / float(direction_T[2])
    if distance <= 0.0:
        raise ValueError("requested Tray plane is behind the camera ray")
    point_T = origin_T + distance * direction_T
    point_T[2] = float(plane_z_T_mm)
    return point_T


def nearest_metric_slot(
    point_T_mm: Sequence[float],
    geometry: Mapping[str, Any],
) -> tuple[str, float]:
    """Return nearest design slot and planar distance in millimetres."""
    point = np.asarray(point_T_mm, dtype=np.float64).reshape(-1)
    if point.size < 2:
        raise ValueError("Tray point must contain at least X and Y")
    slots = geometry.get("slots")
    if not isinstance(slots, Mapping) or not slots:
        raise ValueError("tray geometry does not contain slots")
    best_key: Optional[str] = None
    best_distance = float("inf")
    for slot_key, raw_center in slots.items():
        center = np.asarray(raw_center, dtype=np.float64).reshape(3)
        distance = float(np.linalg.norm(point[:2] - center[:2]))
        if distance < best_distance:
            best_key = str(slot_key)
            best_distance = distance
    if best_key is None:
        raise ValueError("tray geometry does not contain a usable slot")
    return best_key, best_distance


def wafer_patch_center_to_tray(
    center_patch_px: Sequence[float],
    projection: SlotProjection,
    *,
    output_size: int = DEFAULT_CANONICAL_PATCH_SIZE,
    half_extent_mm: float = DEFAULT_SLOT_HALF_EXTENT_MM,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert a normalized slot-patch centre into metric Tray coordinates.

    Canonical patch ``u`` follows decreasing Tray Y and patch ``v`` follows
    decreasing Tray X.  The conversion therefore preserves the fixed slot
    orientation used by :func:`warp_slot_patch` instead of estimating a
    pixel-to-millimetre scale from the current perspective image.
    """
    center = np.asarray(center_patch_px, dtype=np.float64).reshape(-1)
    if center.size != 2 or not np.all(np.isfinite(center)):
        raise ValueError("wafer patch centre must contain two finite values")
    if output_size < 2:
        raise ValueError("canonical patch size must be at least two pixels")
    half = float(half_extent_mm)
    if not math.isfinite(half) or half <= 0.0:
        raise ValueError("slot half extent must be positive and finite")
    edge = float(output_size - 1)
    u, v = float(center[0]), float(center[1])
    offset = np.array(
        [
            half * (1.0 - 2.0 * v / edge),
            half * (1.0 - 2.0 * u / edge),
        ],
        dtype=np.float64,
    )
    slot_center = np.asarray(projection.center_T_mm, dtype=np.float64).reshape(3)
    center_T = slot_center.copy()
    center_T[:2] += offset
    return center_T, offset, float(np.linalg.norm(offset))


def slot_boundary_half_extent_mm(
    geometry: Mapping[str, Any],
    *,
    physical_slot_side_length_mm: float,
) -> float:
    """Validate the physical slot side against the rigid Tray pitch."""

    slot_grid = geometry.get("slot_grid")
    if not isinstance(slot_grid, Mapping):
        raise ValueError("Tray geometry缺少slot_grid")
    try:
        pitch_x = float(slot_grid["pitch_x_mm"])
        pitch_y = float(slot_grid["pitch_y_mm"])
        side_length = float(physical_slot_side_length_mm)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Tray刚体槽间距或视觉配置槽尺寸无效") from exc
    if (
        not all(math.isfinite(value) for value in (side_length, pitch_x, pitch_y))
        or side_length <= 0.0
        or side_length >= min(pitch_x, pitch_y)
    ):
        raise ValueError("Tray geometry的槽边长必须大于0且小于槽间距")
    return 0.5 * side_length


def _classify_patch_quadrilateral_against_physical_slot(
    wafer: WaferObservation,
    projection: SlotProjection,
    *,
    output_size: int,
    patch_half_extent_mm: float,
    slot_boundary_half_extent_mm: float,
    boundary_uncertainty_mm: float,
) -> WaferObservation:
    """Replace the legacy crop-edge flag with the physical 19.9 mm square."""

    if not wafer.found or len(wafer.box_patch_px) != 4:
        return wafer
    box_T = np.asarray(
        [
            wafer_patch_center_to_tray(
                point,
                projection,
                output_size=output_size,
                half_extent_mm=patch_half_extent_mm,
            )[0]
            for point in wafer.box_patch_px
        ],
        dtype=np.float64,
    )
    slot_center = np.asarray(projection.center_T_mm, dtype=np.float64).reshape(3)
    relative_xy = box_T[:, :2] - slot_center[:2]
    maximum_overflow_mm = float(
        np.max(np.abs(relative_xy) - float(slot_boundary_half_extent_mm))
    )
    outside = bool(maximum_overflow_mm > float(boundary_uncertainty_mm))
    flags = tuple(flag for flag in wafer.flags if flag != "outside_slot")
    if outside:
        flags += ("outside_slot",)
    return replace(wafer, outside_slot=outside, flags=flags)


def _correction_full_contour_gates(
    refinement: WaferGeometryRefinement,
    refined_center_T: np.ndarray,
    expanded_projection: SlotProjection,
    geometry: Mapping[str, Any],
    wafer_config: WaferQualityConfig,
    *,
    output_size: int,
    refinement_half_extent_mm: float,
    patch_half_extent_mm: float,
    slot_boundary_half_extent_mm: float,
    minimum_boundary_uncertainty_mm: float,
) -> dict[str, Any]:
    """Validate full-contour physical size and out-of-slot geometry in Tray."""

    edge = float(output_size - 1)
    millimetres_per_pixel = 2.0 * float(refinement_half_extent_mm) / edge
    short_side_mm = float(refinement.short_side_px) * millimetres_per_pixel
    long_side_mm = float(refinement.long_side_px) * millimetres_per_pixel
    area_mm2 = float(refinement.area_px) * millimetres_per_pixel**2
    patch_span_mm = 2.0 * float(patch_half_extent_mm)
    minimum_area_mm2 = float(wafer_config.minimum_area_ratio) * patch_span_mm**2
    maximum_side_mm = (
        float(wafer_config.maximum_normal_side_ratio) * patch_span_mm
    )

    box_T = np.asarray(
        [
            wafer_patch_center_to_tray(
                point,
                expanded_projection,
                output_size=output_size,
                half_extent_mm=refinement_half_extent_mm,
            )[0]
            for point in refinement.box_patch_px
        ],
        dtype=np.float64,
    )
    nearest_slot_key, nearest_slot_distance = nearest_metric_slot(
        refined_center_T, geometry
    )
    nearest_slot_center = np.asarray(
        (geometry.get("slots") or {})[nearest_slot_key], dtype=np.float64
    ).reshape(3)
    boundary_uncertainty_mm = max(
        float(minimum_boundary_uncertainty_mm),
        millimetres_per_pixel,
        float(wafer_config.slot_boundary_margin_ratio)
        * (2.0 * float(slot_boundary_half_extent_mm)),
    )
    relative_box_xy = box_T[:, :2] - nearest_slot_center[:2]
    corner_overflow_mm = np.max(
        np.abs(relative_box_xy) - float(slot_boundary_half_extent_mm), axis=1
    )
    maximum_corner_overflow_mm = float(np.max(corner_overflow_mm))
    slot_polygon_xy = np.asarray(
        [
            [-slot_boundary_half_extent_mm, -slot_boundary_half_extent_mm],
            [slot_boundary_half_extent_mm, -slot_boundary_half_extent_mm],
            [slot_boundary_half_extent_mm, slot_boundary_half_extent_mm],
            [-slot_boundary_half_extent_mm, slot_boundary_half_extent_mm],
        ],
        dtype=np.float32,
    )
    fitted_polygon_xy = relative_box_xy.astype(np.float32)
    fitted_area_mm2 = abs(float(cv2.contourArea(fitted_polygon_xy)))
    intersection_area_mm2, _intersection_polygon = cv2.intersectConvexConvex(
        fitted_polygon_xy,
        slot_polygon_xy,
    )
    outside_area_mm2 = max(
        0.0, fitted_area_mm2 - float(intersection_area_mm2)
    )
    minimum_outside_area_mm2 = max(
        millimetres_per_pixel**2,
        0.5 * boundary_uncertainty_mm * max(short_side_mm, 1.0),
    )
    outside_nearest_slot = bool(
        maximum_corner_overflow_mm > boundary_uncertainty_mm
        and outside_area_mm2 > minimum_outside_area_mm2
    )
    inside_nearest_slot = bool(
        maximum_corner_overflow_mm < -boundary_uncertainty_mm
    )
    boundary_classification = (
        "outside"
        if outside_nearest_slot
        else "inside"
        if inside_nearest_slot
        else "ambiguous"
    )
    physical_size_passed = bool(
        area_mm2 >= minimum_area_mm2
        and short_side_mm > 0.0
        and long_side_mm <= maximum_side_mm
    )
    return {
        "passed": physical_size_passed and outside_nearest_slot,
        "physical_size_passed": physical_size_passed,
        "outside_nearest_slot": outside_nearest_slot,
        "inside_nearest_slot": inside_nearest_slot,
        "boundary_classification": boundary_classification,
        "short_side_mm": short_side_mm,
        "long_side_mm": long_side_mm,
        "area_mm2": area_mm2,
        "minimum_area_mm2": minimum_area_mm2,
        "maximum_side_mm": maximum_side_mm,
        "nearest_slot_key": nearest_slot_key,
        "nearest_slot_center_distance_mm": float(nearest_slot_distance),
        "patch_half_extent_mm": float(patch_half_extent_mm),
        "slot_boundary_half_extent_mm": float(
            slot_boundary_half_extent_mm
        ),
        "boundary_uncertainty_mm": boundary_uncertainty_mm,
        "maximum_corner_overflow_mm": maximum_corner_overflow_mm,
        "outside_area_mm2": outside_area_mm2,
        "minimum_outside_area_mm2": minimum_outside_area_mm2,
        "fitted_polygon_area_mm2": fitted_area_mm2,
        "box_T_mm": box_T.astype(float).tolist(),
    }


def tracked_pose_estimate(tracked: TrackedTrayPose) -> TrayPoseEstimate:
    """Build a self-consistent pose estimate from an accepted filtered pose.

    A tracker rejection is authoritative even when its raw Stage-3 estimate
    passed.  Accepted results receive rvec/tvec values derived from the same
    filtered transform used by the hand-eye overlay, keeping all 36 slot
    projections in the same coordinate state.
    """
    raw = tracked.raw
    if (
        not tracked.accepted_by_tracker
        or tracked.filtered_T_C_T is None
        or tracked.filtered_T_T_C is None
    ):
        return replace(
            raw,
            success=False,
            quality_passed=False,
            failure_reason=(
                tracked.tracker_reason
                or raw.failure_reason
                or "tray pose rejected by temporal tracker"
            ),
            rvec_C_T=None,
            tvec_C_T_mm=None,
            T_C_T=None,
            T_T_C=None,
            camera_position_T_mm=None,
        )
    transform_C_T = np.asarray(
        tracked.filtered_T_C_T, dtype=np.float64
    ).reshape(4, 4)
    transform_T_C = np.asarray(
        tracked.filtered_T_T_C, dtype=np.float64
    ).reshape(4, 4)
    rvec, _ = cv2.Rodrigues(transform_C_T[:3, :3])
    tvec = transform_C_T[:3, 3].reshape(3, 1)
    return replace(
        raw,
        success=True,
        quality_passed=True,
        failure_reason=None,
        rvec_C_T=rvec,
        tvec_C_T_mm=tvec,
        T_C_T=transform_C_T.copy(),
        T_T_C=transform_T_C.copy(),
        camera_position_T_mm=transform_T_C[:3, 3].copy(),
    )


class TrayVisionAnalyzer:
    """Run the layered, camera-1 tray overview analysis for one frame."""

    def __init__(
        self,
        pose_estimator: TrayBoardPoseEstimator,
        geometry: Mapping[str, Any],
        slot_marker_layout: SlotMarkerLayout,
        config: TrayVisionFusionConfig = DEFAULT_TRAY_VISION_FUSION,
    ) -> None:
        self.pose_estimator = pose_estimator
        self.geometry = dict(geometry)
        self.slot_marker_layout = slot_marker_layout
        self.config = config
        self.slot_boundary_half_extent_mm = slot_boundary_half_extent_mm(
            self.geometry,
            physical_slot_side_length_mm=(
                self.config.physical_slot_side_length_mm
            ),
        )
        geometry_slots = set(str(key) for key in self.geometry.get("slots", {}))
        layout_slots = set(slot_marker_layout.marker_id_by_slot)
        if len(geometry_slots) != 36 or geometry_slots != layout_slots:
            raise ValueError("metric geometry and slot marker layout must describe the same 36 slots")

    def _failed_result(self, pose: TrayPoseEstimate) -> TrayVisionResult:
        return TrayVisionResult(
            success=False,
            quality_passed=False,
            failure_reason=pose.failure_reason or "tray pose rejected",
            coordinate_mapping_allowed=False,
            robot_correction_allowed=False,
            pose=pose,
            slot_markers={},
            slots=(),
            summary={"analyzed": 0, "unknown": 36},
            annotated_image=pose.annotated_image.copy(),
            slot_projection_registration={
                "applied": False,
                "reason": "tray_pose_not_accepted",
            },
        )

    @staticmethod
    def _draw_slot(
        canvas: np.ndarray,
        analysis: SlotAnalysis,
        *,
        stacking_detection_enabled: bool = False,
    ) -> None:
        state = analysis.decision.state
        if not stacking_detection_enabled:
            state = {
                SlotState.STACKED: SlotState.WARNING,
                SlotState.STACKED_OUTSIDE_SLOT: SlotState.OUTSIDE_SLOT,
            }.get(state, state)
        colors = {
            SlotState.EMPTY: (60, 180, 75),
            SlotState.EMPTY_UNREAD_MARKER: (120, 210, 150),
            SlotState.OCCUPIED: (255, 0, 255),
            SlotState.WARNING: (0, 165, 255),
            SlotState.STACKED: (0, 0, 255),
            SlotState.OUTSIDE_SLOT: (0, 80, 255),
            SlotState.STACKED_OUTSIDE_SLOT: (80, 0, 180),
            SlotState.OUT_OF_VIEW: (160, 160, 160),
            SlotState.OCCLUDED: (0, 210, 255),
            SlotState.UNKNOWN: (255, 180, 0),
        }
        color = colors[state]
        boundary_polygon = (
            getattr(analysis, "slot_boundary_polygon_image_px", ())
            or analysis.projection.polygon_px
        )
        polygon = np.asarray(boundary_polygon, dtype=np.int32).reshape(4, 2)
        cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)
        center = tuple(np.round(analysis.projection.center_px).astype(int))
        cv2.circle(canvas, center, 5, color, -1, cv2.LINE_AA)
        state_codes = {
            SlotState.EMPTY: "EMPTY",
            SlotState.EMPTY_UNREAD_MARKER: "EMPTY?",
            SlotState.OCCUPIED: "OCC",
            SlotState.WARNING: "WARN",
            SlotState.STACKED: "STACK",
            SlotState.OUTSIDE_SLOT: "OUT",
            SlotState.STACKED_OUTSIDE_SLOT: "STACK+OUT",
            SlotState.OUT_OF_VIEW: "OOV",
            SlotState.OCCLUDED: "OCCL",
            SlotState.UNKNOWN: "UNK",
        }
        label = (
            f"{analysis.projection.slot_key} "
            f"{state_codes[state]}"
        )
        scale = max(0.50, min(0.82, math.sqrt(max(analysis.projection.projected_area_px, 1.0)) / 115.0))
        cv2.putText(
            canvas,
            label,
            (center[0] + 7, center[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2,
            cv2.LINE_AA,
        )
        if analysis.wafer_box_image_px:
            wafer_box = np.asarray(analysis.wafer_box_image_px, dtype=np.int32).reshape(4, 2)
            cv2.polylines(canvas, [wafer_box], True, color, 3, cv2.LINE_AA)
            if analysis.wafer_secondary_boxes_image_px:
                cv2.putText(
                    canvas,
                    "W1",
                    tuple(wafer_box[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        for index, secondary_box_raw in enumerate(
            analysis.wafer_secondary_boxes_image_px,
            start=2,
        ):
            secondary_box = np.asarray(secondary_box_raw, dtype=np.int32).reshape(4, 2)
            secondary_color = (255, 255, 0)
            cv2.polylines(
                canvas,
                [secondary_box],
                True,
                secondary_color,
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"W{index}",
                tuple(secondary_box[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                secondary_color,
                2,
                cv2.LINE_AA,
            )
        if analysis.wafer_center_image_px is not None:
            wafer_center = tuple(
                np.round(analysis.wafer_center_image_px).astype(int)
            )
            cv2.line(canvas, center, wafer_center, color, 2, cv2.LINE_AA)
            cv2.circle(canvas, wafer_center, 6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, wafer_center, 3, color, -1, cv2.LINE_AA)
            if analysis.wafer_offset_distance_mm is not None:
                cv2.putText(
                    canvas,
                    f"d={analysis.wafer_offset_distance_mm:.1f}mm",
                    (wafer_center[0] + 8, wafer_center[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    2,
                    cv2.LINE_AA,
                )

    def analyze(
        self,
        image: np.ndarray,
        *,
        pose: Optional[TrayPoseEstimate] = None,
        explicit_occlusion_mask: Optional[np.ndarray] = None,
    ) -> TrayVisionResult:
        """Analyze one overview image.  No fixed robot-arm mask is assumed."""
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("input must be a valid BGR image")
        if explicit_occlusion_mask is not None and explicit_occlusion_mask.shape[:2] != image.shape[:2]:
            raise ValueError("explicit occlusion mask must match the image size")
        pose_result = pose if pose is not None else self.pose_estimator.estimate(image)
        if not pose_result.success or not pose_result.quality_passed:
            return self._failed_result(pose_result)

        projections = build_slot_projections(
            self.geometry,
            self.pose_estimator,
            pose_result,
            image.shape,
            half_extent_mm=self.config.slot_half_extent_mm,
        )
        boundary_projections = build_slot_projections(
            self.geometry,
            self.pose_estimator,
            pose_result,
            image.shape,
            half_extent_mm=self.slot_boundary_half_extent_mm,
        )
        refinement_half_extent = max(
            OUTSIDE_WAFER_REFINEMENT_HALF_EXTENT_MM,
            float(self.config.slot_half_extent_mm) * 2.0,
        )
        refinement_output_size = max(
            32,
            int(
                round(
                    (self.config.canonical_patch_size - 1)
                    * refinement_half_extent
                    / float(self.config.slot_half_extent_mm)
                )
            )
            + 1,
        )
        refinement_projections = build_slot_projections(
            self.geometry,
            self.pose_estimator,
            pose_result,
            image.shape,
            half_extent_mm=refinement_half_extent,
        )
        observations = detect_aruco_observations(
            image, self.slot_marker_layout.dictionary_name
        )
        marker_registration, registration_diagnostics = (
            estimate_slot_marker_registration(
                projections,
                self.slot_marker_layout,
                observations,
            )
        )
        projections = apply_slot_marker_registration(
            projections, marker_registration, image.shape
        )
        boundary_projections = apply_slot_marker_registration(
            boundary_projections, marker_registration, image.shape
        )
        refinement_projections = apply_slot_marker_registration(
            refinement_projections, marker_registration, image.shape
        )
        physical_boundary_uncertainty_mm = (
            self.config.physical_slot_boundary_uncertainty_mm
            if registration_diagnostics.get("applied")
            else self.config.unregistered_slot_boundary_uncertainty_mm
        )
        analyses: list[SlotAnalysis] = []
        canvas = pose_result.annotated_image.copy()
        for slot_key in sorted(projections):
            projection = projections[slot_key]
            patch, image_to_patch = warp_slot_patch(
                image,
                projection,
                output_size=self.config.canonical_patch_size,
            )
            marker = associate_marker_to_slot(
                projection,
                self.slot_marker_layout,
                observations,
                patch,
            )
            wafer = analyze_wafer_patch(patch, self.config.wafer_quality)
            wafer = _classify_patch_quadrilateral_against_physical_slot(
                wafer,
                projection,
                output_size=self.config.canonical_patch_size,
                patch_half_extent_mm=self.config.slot_half_extent_mm,
                slot_boundary_half_extent_mm=self.slot_boundary_half_extent_mm,
                boundary_uncertainty_mm=physical_boundary_uncertainty_mm,
            )
            occlusion_ratio = _explicit_occlusion_ratio(
                projection.polygon_px, explicit_occlusion_mask
            )
            decision = decide_slot_state(
                projection,
                marker,
                wafer,
                occlusion_ratio=occlusion_ratio,
                config=self.config.slot_decision,
            )
            wafer_box_image: tuple[tuple[float, float], ...] = ()
            wafer_secondary_boxes_image: tuple[
                tuple[tuple[float, float], ...], ...
            ] = ()
            wafer_center_image: Optional[tuple[float, float]] = None
            wafer_center_T: Optional[tuple[float, float, float]] = None
            wafer_offset_T: Optional[tuple[float, float]] = None
            wafer_offset_distance: Optional[float] = None
            wafer_center_refinement: Optional[dict[str, Any]] = None
            wafer_correction_center_valid = False
            wafer_correction_outside_slot = False
            wafer_correction_center_reason = "not_outside_slot"
            correction_seed_image: Optional[tuple[float, float]] = None
            correction_seed_T: Optional[tuple[float, float, float]] = None
            correction_detection_source: Optional[str] = None
            if wafer.found and wafer.box_patch_px:
                mapped = patch_points_to_image(
                    np.asarray(wafer.box_patch_px, dtype=np.float32), image_to_patch
                )
                wafer_box_image = tuple(tuple(float(value) for value in point) for point in mapped)
            if wafer.found and wafer.secondary_boxes_patch_px:
                mapped_secondary_boxes = []
                for secondary_box in wafer.secondary_boxes_patch_px:
                    mapped = patch_points_to_image(
                        np.asarray(secondary_box, dtype=np.float32),
                        image_to_patch,
                    )
                    mapped_secondary_boxes.append(
                        tuple(tuple(float(value) for value in point) for point in mapped)
                    )
                wafer_secondary_boxes_image = tuple(mapped_secondary_boxes)
            if wafer.found and wafer.center_patch_px is not None:
                mapped_center = patch_points_to_image(
                    np.asarray([wafer.center_patch_px], dtype=np.float32),
                    image_to_patch,
                )[0]
                wafer_center_image = tuple(
                    float(value) for value in mapped_center
                )
                center_T, offset_T, offset_distance = wafer_patch_center_to_tray(
                    wafer.center_patch_px,
                    projection,
                    output_size=self.config.canonical_patch_size,
                    half_extent_mm=self.config.slot_half_extent_mm,
                )
                wafer_center_T = tuple(float(value) for value in center_T)
                wafer_offset_T = tuple(float(value) for value in offset_T)
                wafer_offset_distance = float(offset_distance)
            if (
                decision.state
                in {SlotState.OUTSIDE_SLOT, SlotState.STACKED_OUTSIDE_SLOT}
                and wafer.found
                and wafer_center_image is not None
                and wafer_center_T is not None
            ):
                correction_seed_image = wafer_center_image
                correction_seed_T = wafer_center_T
                correction_detection_source = "slot_patch_outside_decision"
            elif (
                wafer.found
                and wafer_center_image is not None
                and wafer_center_T is not None
                and len(wafer.box_patch_px) == 4
                and decision.state
                not in {SlotState.OUT_OF_VIEW, SlotState.OCCLUDED}
            ):
                box_patch = np.asarray(
                    wafer.box_patch_px, dtype=np.float64
                ).reshape(4, 2)
                patch_edge = float(self.config.canonical_patch_size - 1)
                boundary_clearance_px = float(
                    np.min(
                        np.concatenate(
                            (
                                box_patch[:, 0],
                                box_patch[:, 1],
                                patch_edge - box_patch[:, 0],
                                patch_edge - box_patch[:, 1],
                            )
                        )
                    )
                )
                physical_boundary_inset_px = (
                    patch_edge
                    * (
                        float(self.config.slot_half_extent_mm)
                        - float(self.slot_boundary_half_extent_mm)
                    )
                    / (2.0 * float(self.config.slot_half_extent_mm))
                )
                boundary_margin_px = max(
                    1.0,
                    float(
                        self.config.wafer_quality.slot_boundary_margin_ratio
                    )
                    * patch_edge,
                )
                if boundary_clearance_px <= (
                    physical_boundary_inset_px + boundary_margin_px
                ):
                    correction_seed_image = wafer_center_image
                    correction_seed_T = wafer_center_T
                    correction_detection_source = (
                        "slot_patch_near_boundary_quadrilateral"
                    )
            elif (
                not wafer.found
                and "candidate_area_out_of_range" in wafer.flags
                and 0.0 < float(wafer.area_ratio)
                < float(self.config.wafer_quality.minimum_area_ratio)
                and decision.state
                not in {
                    SlotState.OUT_OF_VIEW,
                    SlotState.OCCLUDED,
                    SlotState.STACKED,
                    SlotState.STACKED_OUTSIDE_SLOT,
                }
            ):
                fragment_seed = find_boundary_wafer_fragment_seed(
                    patch, self.config.wafer_quality
                )
                if fragment_seed is not None:
                    mapped_seed = patch_points_to_image(
                        np.asarray([fragment_seed], dtype=np.float32),
                        image_to_patch,
                    )[0]
                    seed_T, _seed_offset, _seed_distance = (
                        wafer_patch_center_to_tray(
                            fragment_seed,
                            projection,
                            output_size=self.config.canonical_patch_size,
                            half_extent_mm=self.config.slot_half_extent_mm,
                        )
                    )
                    correction_seed_image = tuple(
                        float(value) for value in mapped_seed
                    )
                    correction_seed_T = tuple(
                        float(value) for value in seed_T
                    )
                    correction_detection_source = (
                        "slot_boundary_low_area_fragment"
                    )
            if (
                correction_seed_image is not None
                and correction_seed_T is not None
                and correction_detection_source is not None
            ):
                expanded_projection = refinement_projections[slot_key]
                if (
                    expanded_projection.image_coverage_ratio
                    < OUTSIDE_WAFER_REFINEMENT_MINIMUM_IMAGE_COVERAGE
                ):
                    wafer_correction_center_reason = (
                        "expanded_roi_not_fully_in_image"
                    )
                    wafer_center_refinement = {
                        "success": False,
                        "reason": wafer_correction_center_reason,
                        "expanded_image_coverage_ratio": float(
                            expanded_projection.image_coverage_ratio
                        ),
                        "required_image_coverage_ratio": (
                            OUTSIDE_WAFER_REFINEMENT_MINIMUM_IMAGE_COVERAGE
                        ),
                        "detection_source": correction_detection_source,
                    }
                else:
                    expanded_patch, image_to_expanded = warp_slot_patch(
                        image,
                        expanded_projection,
                        output_size=refinement_output_size,
                    )
                    seed_expanded = cv2.perspectiveTransform(
                        np.asarray(
                            [[correction_seed_image]], dtype=np.float32
                        ),
                        np.asarray(image_to_expanded, dtype=np.float32),
                    ).reshape(2)
                    refinement = refine_wafer_geometry_center(
                        expanded_patch,
                        (float(seed_expanded[0]), float(seed_expanded[1])),
                        self.config.wafer_quality,
                    )
                    wafer_center_refinement = refinement.to_json()
                    wafer_center_refinement.update(
                        {
                            "expanded_half_extent_mm": float(
                                refinement_half_extent
                            ),
                            "expanded_output_size_px": int(
                                refinement_output_size
                            ),
                            "expanded_image_coverage_ratio": float(
                                expanded_projection.image_coverage_ratio
                            ),
                            "detection_source": correction_detection_source,
                            "preliminary_center_image_px": list(
                                correction_seed_image
                            ),
                            "preliminary_center_T_mm": list(
                                correction_seed_T
                            ),
                        }
                    )
                    if (
                        refinement.success
                        and refinement.center_patch_px is not None
                    ):
                        refined_center_T, refined_offset_T, refined_distance = (
                            wafer_patch_center_to_tray(
                                refinement.center_patch_px,
                                expanded_projection,
                                output_size=refinement_output_size,
                                half_extent_mm=refinement_half_extent,
                            )
                        )
                        preliminary_center_T = np.asarray(
                            correction_seed_T, dtype=np.float64
                        )
                        seed_shift_mm = float(
                            np.linalg.norm(
                                refined_center_T[:2]
                                - preliminary_center_T[:2]
                            )
                        )
                        wafer_center_refinement["seed_shift_T_xy_mm"] = (
                            seed_shift_mm
                        )
                        full_contour_gates = _correction_full_contour_gates(
                            refinement,
                            refined_center_T,
                            expanded_projection,
                            self.geometry,
                            self.config.wafer_quality,
                            output_size=refinement_output_size,
                            refinement_half_extent_mm=refinement_half_extent,
                            patch_half_extent_mm=self.config.slot_half_extent_mm,
                            slot_boundary_half_extent_mm=(
                                self.slot_boundary_half_extent_mm
                            ),
                            minimum_boundary_uncertainty_mm=(
                                physical_boundary_uncertainty_mm
                            ),
                        )
                        wafer_center_refinement["full_contour_gates"] = (
                            full_contour_gates
                        )
                        if seed_shift_mm > (
                            OUTSIDE_WAFER_REFINEMENT_MAXIMUM_SEED_SHIFT_MM
                        ):
                            wafer_center_refinement["success"] = False
                            wafer_center_refinement["reason"] = (
                                "refined_center_shift_exceeds_limit"
                            )
                            wafer_center_refinement[
                                "maximum_seed_shift_T_xy_mm"
                            ] = (
                                OUTSIDE_WAFER_REFINEMENT_MAXIMUM_SEED_SHIFT_MM
                            )
                            wafer_correction_center_reason = str(
                                wafer_center_refinement["reason"]
                            )
                        elif not full_contour_gates[
                            "physical_size_passed"
                        ]:
                            wafer_center_refinement["success"] = False
                            wafer_center_refinement["reason"] = (
                                "refined_physical_size_out_of_range"
                            )
                            wafer_correction_center_reason = str(
                                wafer_center_refinement["reason"]
                            )
                        else:
                            mapped_center = patch_points_to_image(
                                np.asarray(
                                    [refinement.center_patch_px],
                                    dtype=np.float32,
                                ),
                                image_to_expanded,
                            )[0]
                            mapped_box = patch_points_to_image(
                                np.asarray(
                                    refinement.box_patch_px,
                                    dtype=np.float32,
                                ),
                                image_to_expanded,
                            )
                            wafer_center_image = tuple(
                                float(value) for value in mapped_center
                            )
                            wafer_center_T = tuple(
                                float(value) for value in refined_center_T
                            )
                            wafer_offset_T = tuple(
                                float(value) for value in refined_offset_T
                            )
                            wafer_offset_distance = float(refined_distance)
                            wafer_box_image = tuple(
                                tuple(float(value) for value in point)
                                for point in mapped_box
                            )
                            wafer_center_refinement[
                                "refined_center_image_px"
                            ] = list(wafer_center_image)
                            wafer_center_refinement[
                                "refined_center_T_mm"
                            ] = list(wafer_center_T)
                            wafer_center_refinement[
                                "refined_box_image_px"
                            ] = [list(point) for point in wafer_box_image]
                            if not full_contour_gates[
                                "outside_nearest_slot"
                            ]:
                                # A clipped 23 mm slot patch is only a seed.
                                # Once the complete fitted quadrilateral is
                                # available, its metric boundary classification
                                # replaces the patch-edge OUT decision.
                                wafer = replace(
                                    wafer,
                                    outside_slot=False,
                                    flags=tuple(
                                        flag
                                        for flag in wafer.flags
                                        if flag != "outside_slot"
                                    ),
                                )
                                decision = decide_slot_state(
                                    projection,
                                    marker,
                                    wafer,
                                    occlusion_ratio=occlusion_ratio,
                                    config=self.config.slot_decision,
                                )
                                boundary_classification = str(
                                    full_contour_gates[
                                        "boundary_classification"
                                    ]
                                )
                                wafer_correction_center_reason = (
                                    "refined_contour_inside_nearest_slot"
                                    if boundary_classification == "inside"
                                    else "refined_contour_boundary_ambiguous"
                                )
                                wafer_center_refinement[
                                    "correction_eligible"
                                ] = False
                                wafer_center_refinement[
                                    "slot_decision_overridden_from_patch_edge"
                                ] = True
                            else:
                                if not wafer.outside_slot:
                                    wafer = replace(
                                        wafer,
                                        outside_slot=True,
                                        flags=tuple(wafer.flags)
                                        + ("outside_slot",),
                                    )
                                    decision = decide_slot_state(
                                        projection,
                                        marker,
                                        wafer,
                                        occlusion_ratio=occlusion_ratio,
                                        config=self.config.slot_decision,
                                    )
                                wafer_correction_outside_slot = True
                                if decision.state is not SlotState.STACKED_OUTSIDE_SLOT:
                                    wafer_correction_center_valid = True
                                    wafer_correction_center_reason = "ok"
                                    wafer_center_refinement[
                                        "correction_eligible"
                                    ] = True
                                else:
                                    wafer_correction_center_reason = (
                                        "stacked_outside_not_automatic"
                                    )
                                    wafer_center_refinement[
                                        "correction_eligible"
                                    ] = False
                    else:
                        wafer_correction_center_reason = refinement.reason
            analysis = SlotAnalysis(
                projection=projection,
                marker=marker,
                wafer=wafer,
                decision=decision,
                wafer_box_image_px=wafer_box_image,
                wafer_secondary_boxes_image_px=wafer_secondary_boxes_image,
                wafer_center_image_px=wafer_center_image,
                wafer_center_T_mm=wafer_center_T,
                wafer_offset_T_mm=wafer_offset_T,
                wafer_offset_distance_mm=wafer_offset_distance,
                explicit_occlusion_ratio=occlusion_ratio,
                wafer_center_refinement=wafer_center_refinement,
                wafer_correction_center_valid=wafer_correction_center_valid,
                wafer_correction_outside_slot=wafer_correction_outside_slot,
                wafer_correction_center_reason=wafer_correction_center_reason,
                slot_boundary_polygon_image_px=tuple(
                    tuple(float(value) for value in point)
                    for point in boundary_projections[slot_key].polygon_px
                ),
                slot_boundary_polygon_T_mm=tuple(
                    tuple(float(value) for value in point)
                    for point in boundary_projections[slot_key].polygon_T_mm
                ),
            )
            analyses.append(analysis)
            self._draw_slot(
                canvas,
                analysis,
                stacking_detection_enabled=self.config.wafer_quality.stacking_detection_enabled,
            )

        summary = {state.value: 0 for state in SlotState}
        for analysis in analyses:
            summary[analysis.decision.state.value] += 1
        summary["analyzed"] = len(analyses)
        status = (
            f"pose PASS | RMS={pose_result.reprojection_rms_px:.3f}px | "
            f"empty={summary['empty'] + summary['empty_unread_marker']} | "
            f"occupied={summary['occupied']} | warn={summary['warning']} | "
        )
        if self.config.wafer_quality.stacking_detection_enabled:
            status += (
                f"stacked={summary['stacked']} | outside={summary['outside_slot']} | "
                f"stacked+outside={summary['stacked_outside_slot']} | "
            )
        else:
            status += f"outside={summary['outside_slot']} | "
        status += f"unknown={summary['unknown']}"
        cv2.rectangle(canvas, (8, 8), (min(canvas.shape[1] - 8, 1040), 48), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            status,
            (18, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return TrayVisionResult(
            success=True,
            quality_passed=True,
            failure_reason=None,
            coordinate_mapping_allowed=True,
            # Pixel->Tray is ready here.  Robot correction still requires the
            # existing suction/hand-eye registration layer and is not guessed.
            robot_correction_allowed=False,
            pose=pose_result,
            slot_markers=observations,
            slots=tuple(analyses),
            summary=summary,
            annotated_image=canvas,
            slot_projection_registration=registration_diagnostics,
        )

    def analyze_tracked(
        self,
        image: np.ndarray,
        tracked_pose: TrackedTrayPose,
        *,
        explicit_occlusion_mask: Optional[np.ndarray] = None,
    ) -> TrayVisionResult:
        """Analyze one frame with the same filtered pose as the live UI."""
        return self.analyze(
            image,
            pose=tracked_pose_estimate(tracked_pose),
            explicit_occlusion_mask=explicit_occlusion_mask,
        )

    def map_pixel_to_tray(
        self,
        pixel: Sequence[float],
        result: TrayVisionResult,
        *,
        plane_z_T_mm: float = -2.0,
    ) -> tuple[np.ndarray, str, float]:
        """Map a click to Tray millimetres and its nearest fixed slot."""
        if not result.coordinate_mapping_allowed:
            raise ValueError("coordinate mapping is disabled for this rejected frame")
        point_T = image_pixel_to_tray_plane(
            pixel,
            result.pose,
            self.pose_estimator.intrinsics,
            plane_z_T_mm=plane_z_T_mm,
        )
        slot_key, distance_mm = nearest_metric_slot(point_T, self.geometry)
        return point_T, slot_key, distance_mm


__all__ = [
    "DEFAULT_TRAY_VISION_FUSION",
    "SlotAnalysis",
    "TrayVisionAnalyzer",
    "TrayVisionFusionConfig",
    "TrayVisionResult",
    "image_pixel_to_tray_plane",
    "nearest_metric_slot",
    "tracked_pose_estimate",
    "wafer_patch_center_to_tray",
]
