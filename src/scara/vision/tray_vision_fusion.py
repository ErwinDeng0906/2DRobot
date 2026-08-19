"""Layered tray pose, slot-marker, wafer-quality and occupancy fusion.

The metric board pose remains authoritative.  Slot marker IDs and wafer shape
features are observations attached to those metric slots.  This module never
issues a robot command and never returns a correction when the board pose
fails its existing reprojection quality gates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
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
    associate_marker_to_slot,
    build_slot_projections,
    detect_aruco_observations,
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
        polygon = np.asarray(analysis.projection.polygon_px, dtype=np.int32).reshape(4, 2)
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
        observations = detect_aruco_observations(
            image, self.slot_marker_layout.dictionary_name
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
            )
            analyses.append(analysis)
            self._draw_slot(canvas, analysis)

        summary = {state.value: 0 for state in SlotState}
        for analysis in analyses:
            summary[analysis.decision.state.value] += 1
        summary["analyzed"] = len(analyses)
        status = (
            f"pose PASS | RMS={pose_result.reprojection_rms_px:.3f}px | "
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
