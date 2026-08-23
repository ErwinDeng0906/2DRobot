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
from .planar_tray_registration import (
    PlanarTrayRegistration,
    build_planar_slot_projections,
    estimate_planar_tray_registration,
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
    reconcile_projection_boundary_evidence,
)


@dataclass(frozen=True)
class TrayVisionFusionConfig:
    slot_half_extent_mm: float = DEFAULT_SLOT_HALF_EXTENT_MM
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
    wafer_contour_image_px: tuple[tuple[float, float], ...] = ()
    base_slot_inner_boundary_image_px: tuple[tuple[float, float], ...] = ()
    refined_slot_inner_boundary_image_px: tuple[tuple[float, float], ...] = ()
    projection_disagreement_px: Optional[float] = None

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
            "wafer_contour_image_px": [
                list(point) for point in self.wafer_contour_image_px
            ],
            "base_slot_inner_boundary_image_px": [
                list(point) for point in self.base_slot_inner_boundary_image_px
            ],
            "refined_slot_inner_boundary_image_px": [
                list(point) for point in self.refined_slot_inner_boundary_image_px
            ],
            "projection_disagreement_px": self.projection_disagreement_px,
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
    analysis_quality_passed: bool = False
    projection_source: str = "unavailable"
    planar_registration: Optional[PlanarTrayRegistration] = None
    projection_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "quality_passed": self.quality_passed,
            "analysis_quality_passed": self.analysis_quality_passed,
            "projection_source": self.projection_source,
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
            "planar_registration": (
                None
                if self.planar_registration is None
                else self.planar_registration.to_json()
            ),
            "projection_diagnostics": dict(self.projection_diagnostics),
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


def _projection_disagreement_px(
    base: SlotProjection,
    refined: SlotProjection,
) -> float:
    base_points = np.vstack(
        (
            np.asarray(base.center_px, dtype=np.float64).reshape(1, 2),
            np.asarray(base.polygon_px, dtype=np.float64).reshape(4, 2),
        )
    )
    refined_points = np.vstack(
        (
            np.asarray(refined.center_px, dtype=np.float64).reshape(1, 2),
            np.asarray(refined.polygon_px, dtype=np.float64).reshape(4, 2),
        )
    )
    return float(np.sqrt(np.mean(np.sum(np.square(refined_points - base_points), axis=1))))


def _inner_boundary_patch_points(
    output_size: int,
    margin_ratio: float,
) -> np.ndarray:
    margin = max(1.0, float(margin_ratio) * float(output_size))
    upper = float(output_size - 1) - margin
    return np.asarray(
        [[margin, margin], [upper, margin], [upper, upper], [margin, upper]],
        dtype=np.float32,
    )


def _mapped_inner_boundary(
    image_to_patch: np.ndarray,
    *,
    output_size: int,
    margin_ratio: float,
) -> tuple[tuple[float, float], ...]:
    points = patch_points_to_image(
        _inner_boundary_patch_points(output_size, margin_ratio), image_to_patch
    )
    return tuple(tuple(float(value) for value in point) for point in points)


def _draw_inner_boundary(
    canvas: np.ndarray,
    polygon_raw: Sequence[Sequence[float]],
    *,
    color: tuple[int, int, int],
    crossed_sides: Sequence[str] = (),
) -> None:
    if len(polygon_raw) != 4:
        return
    polygon = np.round(np.asarray(polygon_raw, dtype=np.float64)).astype(np.int32)
    cv2.polylines(canvas, [polygon], True, color, 1, cv2.LINE_AA)
    segments = {
        "top": (0, 1),
        "right": (1, 2),
        "bottom": (2, 3),
        "left": (3, 0),
    }
    for side in crossed_sides:
        pair = segments.get(str(side))
        if pair is None:
            continue
        cv2.line(
            canvas,
            tuple(polygon[pair[0]]),
            tuple(polygon[pair[1]]),
            (0, 0, 255),
            4,
            cv2.LINE_AA,
        )


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
        geometry_slots = set(str(key) for key in self.geometry.get("slots", {}))
        layout_slots = set(slot_marker_layout.marker_id_by_slot)
        if len(geometry_slots) != 36 or geometry_slots != layout_slots:
            raise ValueError("metric geometry and slot marker layout must describe the same 36 slots")

    def _failed_result(
        self,
        pose: TrayPoseEstimate,
        *,
        observations: Optional[dict[int, ArucoObservation]] = None,
        planar_registration: Optional[PlanarTrayRegistration] = None,
    ) -> TrayVisionResult:
        reason = pose.failure_reason or "tray pose rejected"
        if planar_registration is not None and planar_registration.failure_reason:
            reason = f"{reason}; planar fallback: {planar_registration.failure_reason}"
        return TrayVisionResult(
            success=False,
            quality_passed=False,
            failure_reason=reason,
            coordinate_mapping_allowed=False,
            robot_correction_allowed=False,
            pose=pose,
            slot_markers=observations or {},
            slots=(),
            summary={"analyzed": 0, "unknown": 36},
            annotated_image=pose.annotated_image.copy(),
            analysis_quality_passed=False,
            projection_source="unavailable",
            planar_registration=planar_registration,
            projection_diagnostics=(
                {}
                if planar_registration is None
                else planar_registration.to_json()
            ),
        )

    @staticmethod
    def _draw_slot(canvas: np.ndarray, analysis: SlotAnalysis) -> None:
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
        color = colors[analysis.decision.state]
        wafer = getattr(analysis, "wafer", None)
        polygon = np.asarray(analysis.projection.polygon_px, dtype=np.int32).reshape(4, 2)
        cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)
        _draw_inner_boundary(
            canvas,
            getattr(analysis, "base_slot_inner_boundary_image_px", ()),
            color=(255, 255, 0),
            crossed_sides=getattr(wafer, "base_boundary_crossed_sides", ()),
        )
        _draw_inner_boundary(
            canvas,
            getattr(analysis, "refined_slot_inner_boundary_image_px", ()),
            color=(0, 255, 255),
            crossed_sides=getattr(wafer, "refined_boundary_crossed_sides", ()),
        )
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
            f"{state_codes[analysis.decision.state]}"
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
        wafer_contour_image_px = getattr(
            analysis, "wafer_contour_image_px", ()
        )
        if wafer_contour_image_px:
            wafer_contour = np.round(
                np.asarray(wafer_contour_image_px, dtype=np.float64)
            ).astype(np.int32)
            cv2.polylines(
                canvas,
                [wafer_contour],
                True,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        if (
            wafer is not None
            and wafer.found
            and (
                wafer.boundary_evidence != "inside"
                or (getattr(analysis, "projection_disagreement_px", None) or 0.0)
                >= 3.0
            )
        ):
            base_clearance = wafer.base_projection_clearance_px
            refined_clearance = wafer.refined_projection_clearance_px
            clearance_text = (
                f"b={base_clearance:+.1f}px"
                if base_clearance is not None
                else "b=NA"
            )
            if refined_clearance is not None:
                clearance_text += f" r={refined_clearance:+.1f}px"
            cv2.putText(
                canvas,
                clearance_text,
                (center[0] + 7, center[1] + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.38, 0.75 * scale),
                (255, 255, 255),
                1,
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
        strict_pose_passed = bool(pose_result.success and pose_result.quality_passed)
        planar_registration: Optional[PlanarTrayRegistration] = None
        projection_diagnostics: dict[str, Any] = {}
        image_registration: Optional[np.ndarray] = None
        base_projections: Mapping[str, SlotProjection]
        if strict_pose_passed:
            # Normal frames keep the original Stage-3 pose and quality gates.
            # Multi-scale decoding here is observational and cannot modify PnP.
            observations = detect_aruco_observations(
                image, self.slot_marker_layout.dictionary_name
            )
            base_projections = build_slot_projections(
                self.geometry,
                self.pose_estimator,
                pose_result,
                image.shape,
                half_extent_mm=self.config.slot_half_extent_mm,
            )
            image_registration, projection_diagnostics = (
                estimate_slot_marker_registration(
                    base_projections, self.slot_marker_layout, observations
                )
            )
            projections = apply_slot_marker_registration(
                base_projections, image_registration, image.shape
            )
            projection_source = (
                "strict_pnp_slot_marker_refined"
                if image_registration is not None
                else "strict_pnp"
            )
        else:
            # Expensive detection is deliberately restricted to rejected
            # Stage-3 frames.  It feeds only the observation-only homography.
            observations = detect_aruco_observations(
                image,
                self.slot_marker_layout.dictionary_name,
                scales=(1.0, 1.5, 2.0),
            )
            planar_registration = estimate_planar_tray_registration(
                image,
                self.geometry,
                self.slot_marker_layout,
                observations,
            )
            if not planar_registration.success:
                # CLAHE is a second-stage supplement, not routine work.  It is
                # only paid for when native multi-scale decoding cannot form a
                # checked plane.
                observations = detect_aruco_observations(
                    image,
                    self.slot_marker_layout.dictionary_name,
                    scales=(1.0, 1.5, 2.0),
                    include_clahe=True,
                )
                planar_registration = estimate_planar_tray_registration(
                    image,
                    self.geometry,
                    self.slot_marker_layout,
                    observations,
                )
            if not planar_registration.success:
                return self._failed_result(
                    pose_result,
                    observations=observations,
                    planar_registration=planar_registration,
                )
            projections = build_planar_slot_projections(
                self.geometry,
                planar_registration,
                image.shape,
                half_extent_mm=self.config.slot_half_extent_mm,
            )
            projection_source = planar_registration.method
            projection_diagnostics = planar_registration.to_json()
            base_projections = projections
        analyses: list[SlotAnalysis] = []
        canvas = pose_result.annotated_image.copy()
        for slot_key in sorted(projections):
            projection = projections[slot_key]
            base_projection = base_projections[slot_key]
            projection_disagreement = (
                _projection_disagreement_px(base_projection, projection)
                if image_registration is not None
                else None
            )
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
            base_inner_boundary: tuple[tuple[float, float], ...]
            refined_inner_boundary: tuple[tuple[float, float], ...]
            if image_registration is not None:
                refined_inner_boundary = _mapped_inner_boundary(
                    image_to_patch,
                    output_size=self.config.canonical_patch_size,
                    margin_ratio=self.config.wafer_quality.slot_boundary_margin_ratio,
                )
                base_patch, base_image_to_patch = warp_slot_patch(
                    image,
                    base_projection,
                    output_size=self.config.canonical_patch_size,
                )
                base_inner_boundary = _mapped_inner_boundary(
                    base_image_to_patch,
                    output_size=self.config.canonical_patch_size,
                    margin_ratio=self.config.wafer_quality.slot_boundary_margin_ratio,
                )
                base_wafer = (
                    analyze_wafer_patch(base_patch, self.config.wafer_quality)
                    if wafer.found or not marker.decoded
                    else None
                )
                wafer = reconcile_projection_boundary_evidence(
                    wafer,
                    base=base_wafer,
                    primary_is_refined=True,
                    projection_disagreement_px=projection_disagreement,
                )
            else:
                base_inner_boundary = _mapped_inner_boundary(
                    image_to_patch,
                    output_size=self.config.canonical_patch_size,
                    margin_ratio=self.config.wafer_quality.slot_boundary_margin_ratio,
                )
                refined_inner_boundary = ()
                wafer = reconcile_projection_boundary_evidence(wafer)
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
            wafer_contour_image: tuple[tuple[float, float], ...] = ()
            wafer_center_image: Optional[tuple[float, float]] = None
            wafer_center_T: Optional[tuple[float, float, float]] = None
            wafer_offset_T: Optional[tuple[float, float]] = None
            wafer_offset_distance: Optional[float] = None
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
            if wafer.found and wafer.contour_patch_px:
                mapped_contour = patch_points_to_image(
                    np.asarray(wafer.contour_patch_px, dtype=np.float32),
                    image_to_patch,
                )
                wafer_contour_image = tuple(
                    tuple(float(value) for value in point)
                    for point in mapped_contour
                )
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
                wafer_contour_image_px=wafer_contour_image,
                base_slot_inner_boundary_image_px=base_inner_boundary,
                refined_slot_inner_boundary_image_px=refined_inner_boundary,
                projection_disagreement_px=projection_disagreement,
            )
            analyses.append(analysis)
            self._draw_slot(canvas, analysis)

        summary = {state.value: 0 for state in SlotState}
        for analysis in analyses:
            summary[analysis.decision.state.value] += 1
        summary["analyzed"] = len(analyses)
        registration_status = (
            f"pose PASS | RMS={pose_result.reprojection_rms_px:.3f}px"
            if strict_pose_passed
            else (
                "READ-ONLY PLANAR | "
                f"RMS={planar_registration.reprojection_rms_px:.3f}px"
            )
        )
        status = (
            f"{registration_status} | "
            f"empty={summary['empty'] + summary['empty_unread_marker']} | "
            f"occupied={summary['occupied']} | warn={summary['warning']} | "
            f"stacked={summary['stacked']} | outside={summary['outside_slot']} | "
            f"stacked+outside={summary['stacked_outside_slot']} | unknown={summary['unknown']}"
        )
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
            # quality_passed remains the strict metric Stage-3 contract.
            quality_passed=strict_pose_passed,
            failure_reason=(None if strict_pose_passed else pose_result.failure_reason),
            coordinate_mapping_allowed=strict_pose_passed,
            # Pixel->Tray is ready here.  Robot correction still requires the
            # existing suction/hand-eye registration layer and is not guessed.
            robot_correction_allowed=False,
            pose=pose_result,
            slot_markers=observations,
            slots=tuple(analyses),
            summary=summary,
            annotated_image=canvas,
            analysis_quality_passed=True,
            projection_source=projection_source,
            planar_registration=planar_registration,
            projection_diagnostics=projection_diagnostics,
        )

    def analyze_tracked(
        self,
        image: np.ndarray,
        tracked_pose: TrackedTrayPose,
        *,
        explicit_occlusion_mask: Optional[np.ndarray] = None,
    ) -> TrayVisionResult:
        """Analyze one frame with the same filtered pose as the live UI."""
        tracked_estimate = tracked_pose_estimate(tracked_pose)
        if (
            not tracked_pose.accepted_by_tracker
            and tracked_pose.raw.success
            and tracked_pose.raw.quality_passed
        ):
            # A raw Stage-3 pose that fails the temporal tracker remains useful
            # for this frame's read-only patches, but must not regain any metric
            # or robot authorization.  The externally visible pose and safety
            # flags therefore remain the authoritative tracker rejection.
            read_only = self.analyze(
                image,
                pose=tracked_pose.raw,
                explicit_occlusion_mask=explicit_occlusion_mask,
            )
            diagnostics = dict(read_only.projection_diagnostics)
            diagnostics["temporal_tracker"] = {
                "accepted": False,
                "reason": tracked_pose.tracker_reason,
                "read_only_raw_pnp_projection": True,
            }
            return replace(
                read_only,
                quality_passed=False,
                failure_reason=(
                    tracked_pose.tracker_reason
                    or "tray pose rejected by temporal tracker"
                ),
                coordinate_mapping_allowed=False,
                robot_correction_allowed=False,
                pose=tracked_estimate,
                projection_source="strict_pnp_untracked_read_only",
                projection_diagnostics=diagnostics,
            )
        return self.analyze(
            image,
            pose=tracked_estimate,
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
