"""Layered tray pose, slot-marker, wafer-quality and occupancy fusion.

The metric board pose remains authoritative.  Slot marker IDs and wafer shape
features are observations attached to those metric slots.  This module never
issues a robot command and never returns a correction when the board pose
fails its existing reprojection quality gates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
    explicit_occlusion_ratio: float

    def to_json(self) -> dict[str, Any]:
        return {
            "slot_key": self.projection.slot_key,
            "projection": self.projection.to_json(),
            "marker": self.marker.to_json(),
            "wafer": self.wafer.to_json(),
            "decision": self.decision.to_json(),
            "wafer_box_image_px": [list(point) for point in self.wafer_box_image_px],
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
            SlotState.ABNORMAL: (0, 0, 255),
            SlotState.OUT_OF_VIEW: (160, 160, 160),
            SlotState.OCCLUDED: (0, 210, 255),
            SlotState.UNKNOWN: (255, 180, 0),
        }
        color = colors[analysis.decision.state]
        polygon = np.asarray(analysis.projection.polygon_px, dtype=np.int32).reshape(4, 2)
        cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)
        center = tuple(np.round(analysis.projection.center_px).astype(int))
        cv2.circle(canvas, center, 5, color, -1, cv2.LINE_AA)
        label = f"{analysis.projection.slot_key} {analysis.decision.state.value}"
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
            if wafer.found and wafer.box_patch_px:
                mapped = patch_points_to_image(
                    np.asarray(wafer.box_patch_px, dtype=np.float32), image_to_patch
                )
                wafer_box_image = tuple(tuple(float(value) for value in point) for point in mapped)
            analysis = SlotAnalysis(
                projection=projection,
                marker=marker,
                wafer=wafer,
                decision=decision,
                wafer_box_image_px=wafer_box_image,
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
            f"abnormal={summary['abnormal']} | unknown={summary['unknown']}"
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
]
