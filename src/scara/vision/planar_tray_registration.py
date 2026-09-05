"""Fail-closed, observation-only registration of the fixed tray marker plane.

This module deliberately produces no camera pose, world transform, calibration
sample, or robot correction.  It exists only to project the 36 fixed slot
patches when the strict Stage-3 outer-marker PnP has rejected the current frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

from .slot_marker_observation import ArucoObservation, SlotMarkerLayout, SlotProjection


MINIMUM_MARKER_COUNT = 3
MINIMUM_CORRESPONDENCE_COUNT = 6
MINIMUM_AXIS_SPAN_MM = 60.0
GRID_RANSAC_THRESHOLD_PX = 4.0
GRID_MINIMUM_INLIER_RATIO = 0.50
GRID_MAXIMUM_RMS_PX = 4.0
GRID_MAXIMUM_RESIDUAL_PX = 8.0
TWO_OUTER_MINIMUM_SQUARE_QUALITY = 0.85
TWO_OUTER_MINIMUM_INLIER_CORNERS = 6
TWO_OUTER_MAXIMUM_RMS_PX = 2.5
TWO_OUTER_MAXIMUM_RESIDUAL_PX = 4.0
PARTIAL_SEARCH_RADIUS_PX = 8.0
PARTIAL_MAXIMUM_DIRECTION_ERROR_DEG = 15.0
PARTIAL_MAXIMUM_RESIDUAL_PX = 3.0


@dataclass(frozen=True)
class PlanarTrayRegistration:
    """Checked image-from-Tray-XY registration for read-only analysis."""

    success: bool
    failure_reason: Optional[str]
    method: str
    fit_variant: str
    homography_image_from_tray_xy: Optional[np.ndarray]
    complete_marker_ids: tuple[int, ...]
    partial_marker_ids: tuple[int, ...]
    inlier_marker_ids: tuple[int, ...]
    span_x_mm: float
    span_y_mm: float
    correspondence_count: int
    inlier_correspondence_count: int
    inlier_ratio: float
    reprojection_rms_px: Optional[float]
    reprojection_max_px: Optional[float]
    marker_diagnostics: dict[int, dict[str, Any]]
    partial_marker_diagnostics: dict[int, dict[str, Any]]

    def to_json(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failure_reason": self.failure_reason,
            "method": self.method,
            "fit_variant": self.fit_variant,
            "complete_marker_ids": list(self.complete_marker_ids),
            "partial_marker_ids": list(self.partial_marker_ids),
            "inlier_marker_ids": list(self.inlier_marker_ids),
            "span_x_mm": self.span_x_mm,
            "span_y_mm": self.span_y_mm,
            "correspondence_count": self.correspondence_count,
            "inlier_correspondence_count": self.inlier_correspondence_count,
            "inlier_ratio": self.inlier_ratio,
            "reprojection_rms_px": self.reprojection_rms_px,
            "reprojection_max_px": self.reprojection_max_px,
            "marker_diagnostics": {
                str(marker_id): dict(values)
                for marker_id, values in sorted(self.marker_diagnostics.items())
            },
            "partial_marker_diagnostics": {
                str(marker_id): dict(values)
                for marker_id, values in sorted(
                    self.partial_marker_diagnostics.items()
                )
            },
            "homography_image_from_tray_xy": (
                None
                if self.homography_image_from_tray_xy is None
                else np.asarray(
                    self.homography_image_from_tray_xy, dtype=float
                ).tolist()
            ),
            "coordinate_mapping_allowed": False,
            "robot_correction_allowed": False,
        }


def _marker_diagnostics(
    observations: Mapping[int, ArucoObservation], fixed_ids: set[int]
) -> dict[int, dict[str, Any]]:
    return {
        marker_id: {
            "complete_decoded": bool(observation.complete_decoded),
            "detection_scale": float(observation.detection_scale),
            "preprocessing": str(observation.preprocessing),
            "square_quality": float(observation.square_quality),
            "perimeter_px": float(observation.perimeter_px),
        }
        for marker_id, observation in observations.items()
        if marker_id in fixed_ids
    }


def _failed_registration(
    reason: str,
    *,
    complete_marker_ids: Sequence[int] = (),
    span_x_mm: float = 0.0,
    span_y_mm: float = 0.0,
    correspondence_count: int = 0,
    inlier_correspondence_count: int = 0,
    inlier_ratio: float = 0.0,
    rms: Optional[float] = None,
    maximum: Optional[float] = None,
    homography: Optional[np.ndarray] = None,
    marker_diagnostics: Optional[dict[int, dict[str, Any]]] = None,
) -> PlanarTrayRegistration:
    return PlanarTrayRegistration(
        success=False,
        failure_reason=reason,
        method="unavailable",
        fit_variant="none",
        homography_image_from_tray_xy=homography,
        complete_marker_ids=tuple(sorted(set(int(value) for value in complete_marker_ids))),
        partial_marker_ids=(),
        inlier_marker_ids=(),
        span_x_mm=float(span_x_mm),
        span_y_mm=float(span_y_mm),
        correspondence_count=int(correspondence_count),
        inlier_correspondence_count=int(inlier_correspondence_count),
        inlier_ratio=float(inlier_ratio),
        reprojection_rms_px=rms,
        reprojection_max_px=maximum,
        marker_diagnostics=marker_diagnostics or {},
        partial_marker_diagnostics={},
    )


def _outer_marker_map(geometry: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    markers = geometry.get("markers")
    if not isinstance(markers, Mapping):
        return result
    for marker in markers.values():
        if not isinstance(marker, Mapping):
            continue
        try:
            marker_id = int(marker["id"])
        except (KeyError, TypeError, ValueError):
            continue
        result[marker_id] = marker
    return result


def _span(points: Sequence[Sequence[float]]) -> tuple[float, float]:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(values) < 2:
        return 0.0, 0.0
    return float(np.ptp(values[:, 0])), float(np.ptp(values[:, 1]))


def _fit_homography(
    source: np.ndarray,
    target: np.ndarray,
    point_marker_ids: np.ndarray,
    *,
    threshold_px: float,
) -> tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    homography, raw_mask = cv2.findHomography(
        source.astype(np.float32),
        target.astype(np.float32),
        cv2.RANSAC,
        float(threshold_px),
        maxIters=4000,
        confidence=0.997,
    )
    if (
        homography is None
        or raw_mask is None
        or not np.all(np.isfinite(homography))
    ):
        return None, np.zeros(len(source), dtype=bool), np.full(len(source), np.inf)
    inliers = raw_mask.reshape(-1).astype(bool)
    projected = cv2.perspectiveTransform(
        source.reshape(1, -1, 2).astype(np.float32), homography
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - target, axis=1)
    return homography.astype(np.float64), inliers, errors


def _inlier_marker_ids(
    marker_ids: Sequence[int], point_marker_ids: np.ndarray, inliers: np.ndarray
) -> tuple[int, ...]:
    accepted: list[int] = []
    for marker_id in sorted(set(int(value) for value in marker_ids)):
        marker_points = point_marker_ids == marker_id
        point_count = int(np.count_nonzero(marker_points))
        required = max(1, int(math.ceil(0.50 * point_count)))
        if int(np.count_nonzero(inliers & marker_points)) >= required:
            accepted.append(marker_id)
    return tuple(accepted)


def _corner_angle_error_deg(
    predicted: np.ndarray, observed: np.ndarray, indices: Sequence[int]
) -> Optional[float]:
    best: Optional[float] = None
    for first_index, first in enumerate(indices):
        for second in indices[first_index + 1 :]:
            if (int(second) - int(first)) % 4 not in (1, 3):
                continue
            expected_vector = predicted[second] - predicted[first]
            observed_vector = observed[second] - observed[first]
            expected_norm = float(np.linalg.norm(expected_vector))
            observed_norm = float(np.linalg.norm(observed_vector))
            if expected_norm < 2.0 or observed_norm < 2.0:
                continue
            cosine = float(
                np.dot(expected_vector, observed_vector)
                / (expected_norm * observed_norm)
            )
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
            best = angle if best is None else min(best, angle)
    return best


def _find_predicted_corner(
    gray: np.ndarray, predicted: np.ndarray, used: Sequence[np.ndarray]
) -> Optional[np.ndarray]:
    height, width = gray.shape[:2]
    x, y = float(predicted[0]), float(predicted[1])
    radius = int(math.ceil(PARTIAL_SEARCH_RADIUS_PX))
    if x < radius or y < radius or x >= width - radius or y >= height - radius:
        return None
    left = max(0, int(math.floor(x)) - radius)
    top = max(0, int(math.floor(y)) - radius)
    right = min(width, int(math.floor(x)) + radius + 1)
    bottom = min(height, int(math.floor(y)) + radius + 1)
    roi = gray[top:bottom, left:right]
    candidates = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=8,
        qualityLevel=0.06,
        minDistance=3.0,
        blockSize=5,
        useHarrisDetector=False,
    )
    if candidates is None:
        return None
    points = candidates.reshape(-1, 2).astype(np.float32)
    points[:, 0] += float(left)
    points[:, 1] += float(top)
    ordered = sorted(points, key=lambda point: float(np.linalg.norm(point - predicted)))
    for point in ordered:
        if float(np.linalg.norm(point - predicted)) > PARTIAL_SEARCH_RADIUS_PX:
            break
        if any(float(np.linalg.norm(point - previous)) < 2.0 for previous in used):
            continue
        refined = point.reshape(1, 1, 2).copy()
        cv2.cornerSubPix(
            gray,
            refined,
            (3, 3),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 0.02),
        )
        return refined.reshape(2).astype(np.float64)
    return None


def _refine_with_partial_outer_corners(
    image: np.ndarray,
    registration: PlanarTrayRegistration,
    outer_markers: Mapping[int, Mapping[str, Any]],
    full_source: np.ndarray,
    full_target: np.ndarray,
    full_inliers: np.ndarray,
) -> PlanarTrayRegistration:
    """Conservatively supplement an already accepted full-marker solution."""

    homography = registration.homography_image_from_tray_xy
    if homography is None or int(np.count_nonzero(full_inliers)) < 4:
        return registration
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    full_ids = set(registration.complete_marker_ids)
    partial_source: list[np.ndarray] = []
    partial_target: list[np.ndarray] = []
    point_owner: list[int] = []
    diagnostics: dict[int, dict[str, Any]] = {}
    for marker_id, marker in sorted(outer_markers.items()):
        if marker_id in full_ids:
            continue
        raw_corners = marker.get("corners_T_mm")
        if not isinstance(raw_corners, (list, tuple)) or len(raw_corners) != 4:
            continue
        source_corners = np.asarray(raw_corners, dtype=np.float64)[:, :2]
        predicted = cv2.perspectiveTransform(
            source_corners.reshape(1, -1, 2).astype(np.float32), homography
        ).reshape(-1, 2).astype(np.float64)
        matched_indices: list[int] = []
        matched_points: list[np.ndarray] = []
        for corner_index, predicted_point in enumerate(predicted):
            found = _find_predicted_corner(gray, predicted_point, matched_points)
            if found is not None:
                matched_indices.append(corner_index)
                matched_points.append(found)
        observed_by_index = predicted.copy()
        for corner_index, point in zip(matched_indices, matched_points):
            observed_by_index[corner_index] = point
        direction_error = _corner_angle_error_deg(
            predicted, observed_by_index, matched_indices
        )
        accepted_candidate = bool(
            len(matched_indices) >= 2
            and direction_error is not None
            and direction_error <= PARTIAL_MAXIMUM_DIRECTION_ERROR_DEG
        )
        diagnostics[marker_id] = {
            "matched_corner_indices": matched_indices,
            "matched_corner_count": len(matched_indices),
            "edge_direction_error_deg": direction_error,
            "candidate_accepted_for_refinement": accepted_candidate,
            "retained": False,
        }
        if not accepted_candidate:
            continue
        for corner_index, point in zip(matched_indices, matched_points):
            partial_source.append(source_corners[corner_index])
            partial_target.append(point)
            point_owner.append(marker_id)
    if not partial_source:
        return replace(registration, partial_marker_diagnostics=diagnostics)

    accepted_full_source = full_source[full_inliers]
    accepted_full_target = full_target[full_inliers]
    weighted_source = np.vstack(
        [np.repeat(accepted_full_source, 4, axis=0), np.asarray(partial_source)]
    ).astype(np.float32)
    weighted_target = np.vstack(
        [np.repeat(accepted_full_target, 4, axis=0), np.asarray(partial_target)]
    ).astype(np.float32)
    refined, _ = cv2.findHomography(weighted_source, weighted_target, 0)
    if refined is None or not np.all(np.isfinite(refined)):
        return replace(registration, partial_marker_diagnostics=diagnostics)

    baseline_projection = cv2.perspectiveTransform(
        accepted_full_source.reshape(1, -1, 2).astype(np.float32), homography
    ).reshape(-1, 2)
    refined_full_projection = cv2.perspectiveTransform(
        accepted_full_source.reshape(1, -1, 2).astype(np.float32), refined
    ).reshape(-1, 2)
    baseline_errors = np.linalg.norm(baseline_projection - accepted_full_target, axis=1)
    refined_full_errors = np.linalg.norm(
        refined_full_projection - accepted_full_target, axis=1
    )
    refined_partial_projection = cv2.perspectiveTransform(
        np.asarray(partial_source, dtype=np.float32).reshape(1, -1, 2), refined
    ).reshape(-1, 2)
    partial_errors = np.linalg.norm(
        refined_partial_projection - np.asarray(partial_target), axis=1
    )
    baseline_rms = float(np.sqrt(np.mean(np.square(baseline_errors))))
    refined_rms = float(np.sqrt(np.mean(np.square(refined_full_errors))))
    retained = bool(
        refined_rms <= baseline_rms + 0.05
        and float(np.max(refined_full_errors))
        <= float(np.max(baseline_errors)) + 0.25
        and float(np.max(partial_errors)) <= PARTIAL_MAXIMUM_RESIDUAL_PX
    )
    for marker_id in diagnostics:
        owned_errors = [
            float(error)
            for owner, error in zip(point_owner, partial_errors)
            if owner == marker_id
        ]
        diagnostics[marker_id]["partial_residual_max_px"] = (
            max(owned_errors) if owned_errors else None
        )
        diagnostics[marker_id]["full_marker_rms_before_px"] = baseline_rms
        diagnostics[marker_id]["full_marker_rms_after_px"] = refined_rms
        diagnostics[marker_id]["retained"] = bool(retained and owned_errors)
    if not retained:
        return replace(registration, partial_marker_diagnostics=diagnostics)
    retained_ids = tuple(
        marker_id
        for marker_id in sorted(diagnostics)
        if diagnostics[marker_id]["retained"]
    )
    return replace(
        registration,
        homography_image_from_tray_xy=np.asarray(refined, dtype=np.float64),
        partial_marker_ids=retained_ids,
        partial_marker_diagnostics=diagnostics,
    )


def estimate_planar_tray_registration(
    image: np.ndarray,
    geometry: Mapping[str, Any],
    layout: SlotMarkerLayout,
    observations: Mapping[int, ArucoObservation],
    *,
    allow_partial_corners: bool = True,
    prefer_slot_centres: bool = False,
) -> PlanarTrayRegistration:
    """Fit a guarded fixed-marker homography for read-only slot analysis."""

    slots = geometry.get("slots")
    if not isinstance(slots, Mapping) or len(slots) != 36:
        return _failed_registration("tray geometry must contain 36 slots")
    outer_markers = _outer_marker_map(geometry)
    slot_by_marker_id = {
        int(marker_id): slot_key
        for slot_key, marker_id in layout.marker_id_by_slot.items()
    }
    fixed_ids = set(slot_by_marker_id) | set(outer_markers)
    diagnostics = _marker_diagnostics(observations, fixed_ids)
    complete_ids = sorted(
        marker_id
        for marker_id in (set(observations) & fixed_ids)
        if observations[marker_id].complete_decoded
        and len(observations[marker_id].corners_px) == 4
    )

    # Slot centres share the actual wafer-support plane (z=-2 mm). Outer
    # markers are at several other heights: mixing their corners into one
    # homography biases the slots as the visible marker set changes. Prefer
    # same-plane observations when they provide sufficient spatial support.
    if prefer_slot_centres:
        inner_ids = [key for key in complete_ids if key in slot_by_marker_id]
        if len(inner_ids) >= MINIMUM_CORRESPONDENCE_COUNT:
            source = np.asarray([slots[slot_by_marker_id[key]][:2] for key in inner_ids], np.float32)
            target = np.asarray([observations[key].center_px for key in inner_ids], np.float32)
            matrix, inliers, errors = _fit_homography(source, target, np.asarray(inner_ids), threshold_px=GRID_RANSAC_THRESHOLD_PX)
            if matrix is not None and np.count_nonzero(inliers) >= MINIMUM_CORRESPONDENCE_COUNT:
                span_x, span_y = _span(source[inliers])
                rms = float(np.sqrt(np.mean(errors[inliers] ** 2)))
                maximum = float(np.max(errors[inliers]))
                ratio = float(np.mean(inliers))
                if (span_x >= MINIMUM_AXIS_SPAN_MM and span_y >= MINIMUM_AXIS_SPAN_MM
                        and rms <= GRID_MAXIMUM_RMS_PX and maximum <= GRID_MAXIMUM_RESIDUAL_PX
                        and ratio >= GRID_MINIMUM_INLIER_RATIO):
                    return PlanarTrayRegistration(
                        success=True, failure_reason=None, method='marker_grid_homography',
                        fit_variant='slot_centres_only', homography_image_from_tray_xy=matrix,
                        complete_marker_ids=tuple(inner_ids), partial_marker_ids=(),
                        inlier_marker_ids=tuple(key for key, good in zip(inner_ids, inliers) if good),
                        span_x_mm=span_x, span_y_mm=span_y, correspondence_count=len(inner_ids),
                        inlier_correspondence_count=int(np.count_nonzero(inliers)), inlier_ratio=ratio,
                        reprojection_rms_px=rms, reprojection_max_px=maximum,
                        marker_diagnostics=diagnostics, partial_marker_diagnostics={},
                    )

    marker_centres_T: list[list[float]] = []
    source_points: list[list[float]] = []
    target_points: list[Sequence[float]] = []
    point_marker_ids: list[int] = []
    centre_source: list[list[float]] = []
    centre_target: list[Sequence[float]] = []
    centre_marker_ids: list[int] = []
    for marker_id in complete_ids:
        observation = observations[marker_id]
        if marker_id in slot_by_marker_id:
            center = np.asarray(slots[slot_by_marker_id[marker_id]], dtype=float)
            center_xy = [float(center[0]), float(center[1])]
            source_points.append(center_xy)
            target_points.append(observation.center_px)
            point_marker_ids.append(marker_id)
            marker_centres_T.append(center_xy)
            centre_source.append(center_xy)
            centre_target.append(observation.center_px)
            centre_marker_ids.append(marker_id)
            continue
        marker = outer_markers[marker_id]
        center = marker.get("center_T_mm")
        if not isinstance(center, (list, tuple)) or len(center) < 2:
            continue
        center_xy = [float(center[0]), float(center[1])]
        source_points.append(center_xy)
        target_points.append(observation.center_px)
        point_marker_ids.append(marker_id)
        marker_centres_T.append(center_xy)
        centre_source.append(center_xy)
        centre_target.append(observation.center_px)
        centre_marker_ids.append(marker_id)
        corners = marker.get("corners_T_mm")
        if isinstance(corners, (list, tuple)) and len(corners) == 4:
            for tray_corner, image_corner in zip(corners, observation.corners_px):
                source_points.append([float(tray_corner[0]), float(tray_corner[1])])
                target_points.append(image_corner)
                point_marker_ids.append(marker_id)

    span_x, span_y = _span(marker_centres_T)
    visible_outer_ids = [marker_id for marker_id in complete_ids if marker_id in outer_markers]
    two_outer = bool(
        len(complete_ids) == 2
        and len(visible_outer_ids) == 2
        and span_x >= MINIMUM_AXIS_SPAN_MM
        and span_y >= MINIMUM_AXIS_SPAN_MM
    )
    if len(complete_ids) == 2 and not two_outer:
        return _failed_registration(
            "two-marker fallback requires two calibrated outer markers spanning both tray axes",
            complete_marker_ids=complete_ids,
            span_x_mm=span_x,
            span_y_mm=span_y,
            correspondence_count=len(source_points),
            marker_diagnostics=diagnostics,
        )
    if len(complete_ids) < MINIMUM_MARKER_COUNT and not two_outer:
        return _failed_registration(
            f"fixed tray markers are insufficient: {len(complete_ids)}/{MINIMUM_MARKER_COUNT}",
            complete_marker_ids=complete_ids,
            span_x_mm=span_x,
            span_y_mm=span_y,
            correspondence_count=len(source_points),
            marker_diagnostics=diagnostics,
        )
    if span_x < MINIMUM_AXIS_SPAN_MM or span_y < MINIMUM_AXIS_SPAN_MM:
        return _failed_registration(
            "visible markers do not span both tray axes: "
            f"x={span_x:.1f}mm, y={span_y:.1f}mm",
            complete_marker_ids=complete_ids,
            span_x_mm=span_x,
            span_y_mm=span_y,
            correspondence_count=len(source_points),
            marker_diagnostics=diagnostics,
        )

    if two_outer:
        if any(
            observations[marker_id].square_quality
            < TWO_OUTER_MINIMUM_SQUARE_QUALITY
            for marker_id in visible_outer_ids
        ):
            return _failed_registration(
                "two-marker fallback rejected low square quality",
                complete_marker_ids=complete_ids,
                span_x_mm=span_x,
                span_y_mm=span_y,
                marker_diagnostics=diagnostics,
            )
        source_points = []
        target_points = []
        point_marker_ids = []
        for marker_id in visible_outer_ids:
            corners = outer_markers[marker_id].get("corners_T_mm")
            if not isinstance(corners, (list, tuple)) or len(corners) != 4:
                return _failed_registration(
                    "two-marker fallback requires four calibrated corners per marker",
                    complete_marker_ids=complete_ids,
                    span_x_mm=span_x,
                    span_y_mm=span_y,
                    marker_diagnostics=diagnostics,
                )
            for tray_corner, image_corner in zip(
                corners, observations[marker_id].corners_px
            ):
                source_points.append([float(tray_corner[0]), float(tray_corner[1])])
                target_points.append(image_corner)
                point_marker_ids.append(marker_id)

    if len(source_points) < MINIMUM_CORRESPONDENCE_COUNT:
        return _failed_registration(
            "fixed tray marker geometry has too few correspondences: "
            f"{len(source_points)}/{MINIMUM_CORRESPONDENCE_COUNT}",
            complete_marker_ids=complete_ids,
            span_x_mm=span_x,
            span_y_mm=span_y,
            correspondence_count=len(source_points),
            marker_diagnostics=diagnostics,
        )

    source = np.asarray(source_points, dtype=np.float32)
    target = np.asarray(target_points, dtype=np.float32)
    point_ids = np.asarray(point_marker_ids, dtype=np.int32)
    threshold = 2.5 if two_outer else GRID_RANSAC_THRESHOLD_PX
    homography, inliers, errors = _fit_homography(
        source, target, point_ids, threshold_px=threshold
    )
    if homography is None or not np.any(inliers):
        return _failed_registration(
            "marker-grid homography failed",
            complete_marker_ids=complete_ids,
            span_x_mm=span_x,
            span_y_mm=span_y,
            correspondence_count=len(source),
            marker_diagnostics=diagnostics,
        )
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = float(inlier_count / len(source))
    rms = float(np.sqrt(np.mean(np.square(errors[inliers]))))
    maximum = float(np.max(errors[inliers]))
    inlier_marker_ids = _inlier_marker_ids(complete_ids, point_ids, inliers)
    accepted = bool(
        inlier_count >= MINIMUM_CORRESPONDENCE_COUNT
        and inlier_ratio >= GRID_MINIMUM_INLIER_RATIO
        and (
            (
                two_outer
                and inlier_count >= TWO_OUTER_MINIMUM_INLIER_CORNERS
                and len(inlier_marker_ids) == 2
                and rms <= TWO_OUTER_MAXIMUM_RMS_PX
                and maximum <= TWO_OUTER_MAXIMUM_RESIDUAL_PX
            )
            or (
                not two_outer
                and len(inlier_marker_ids) >= MINIMUM_MARKER_COUNT
                and rms <= GRID_MAXIMUM_RMS_PX
                and maximum <= GRID_MAXIMUM_RESIDUAL_PX
            )
        )
    )

    fit_variant = "two_outer_complete_corners" if two_outer else "centres_plus_outer_corners"
    if not accepted and not two_outer and len(centre_source) >= MINIMUM_CORRESPONDENCE_COUNT:
        centre_source_array = np.asarray(centre_source, dtype=np.float32)
        centre_target_array = np.asarray(centre_target, dtype=np.float32)
        centre_ids_array = np.asarray(centre_marker_ids, dtype=np.int32)
        alternative, alternative_inliers, alternative_errors = _fit_homography(
            centre_source_array,
            centre_target_array,
            centre_ids_array,
            threshold_px=GRID_RANSAC_THRESHOLD_PX,
        )
        if alternative is not None and np.any(alternative_inliers):
            alternative_count = int(np.count_nonzero(alternative_inliers))
            alternative_ratio = float(alternative_count / len(centre_source_array))
            alternative_rms = float(
                np.sqrt(np.mean(np.square(alternative_errors[alternative_inliers])))
            )
            alternative_maximum = float(np.max(alternative_errors[alternative_inliers]))
            alternative_marker_ids = _inlier_marker_ids(
                centre_marker_ids, centre_ids_array, alternative_inliers
            )
            if (
                alternative_count >= MINIMUM_CORRESPONDENCE_COUNT
                and alternative_ratio >= GRID_MINIMUM_INLIER_RATIO
                and len(alternative_marker_ids) >= MINIMUM_MARKER_COUNT
                and alternative_rms <= GRID_MAXIMUM_RMS_PX
                and alternative_maximum <= GRID_MAXIMUM_RESIDUAL_PX
            ):
                homography = alternative
                inliers = alternative_inliers
                errors = alternative_errors
                source = centre_source_array
                target = centre_target_array
                point_ids = centre_ids_array
                inlier_count = alternative_count
                inlier_ratio = alternative_ratio
                rms = alternative_rms
                maximum = alternative_maximum
                inlier_marker_ids = alternative_marker_ids
                accepted = True
                fit_variant = "marker_centres_only"

    if not accepted:
        return _failed_registration(
            "marker-grid residual rejected: "
            f"markers={len(inlier_marker_ids)}/{len(complete_ids)}, "
            f"points={inlier_count}/{len(source)}, rms={rms:.3f}px, max={maximum:.3f}px",
            complete_marker_ids=complete_ids,
            span_x_mm=span_x,
            span_y_mm=span_y,
            correspondence_count=len(source),
            inlier_correspondence_count=inlier_count,
            inlier_ratio=inlier_ratio,
            rms=rms,
            maximum=maximum,
            homography=homography,
            marker_diagnostics=diagnostics,
        )

    registration = PlanarTrayRegistration(
        success=True,
        failure_reason=None,
        method=("two_outer_marker_homography" if two_outer else "marker_grid_homography"),
        fit_variant=fit_variant,
        homography_image_from_tray_xy=homography,
        complete_marker_ids=tuple(complete_ids),
        partial_marker_ids=(),
        inlier_marker_ids=inlier_marker_ids,
        span_x_mm=span_x,
        span_y_mm=span_y,
        correspondence_count=len(source),
        inlier_correspondence_count=inlier_count,
        inlier_ratio=inlier_ratio,
        reprojection_rms_px=rms,
        reprojection_max_px=maximum,
        marker_diagnostics=diagnostics,
        partial_marker_diagnostics={},
    )
    if allow_partial_corners:
        registration = _refine_with_partial_outer_corners(
            image, registration, outer_markers, source, target, inliers
        )
    return registration


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


def build_planar_slot_projections(
    geometry: Mapping[str, Any],
    registration: PlanarTrayRegistration,
    image_shape: tuple[int, ...],
    *,
    half_extent_mm: float,
) -> dict[str, SlotProjection]:
    """Project all 36 patches from an accepted read-only registration."""

    homography = registration.homography_image_from_tray_xy
    if not registration.success or homography is None:
        raise ValueError("an accepted planar tray registration is required")
    raw_slots = geometry.get("slots")
    if not isinstance(raw_slots, Mapping) or len(raw_slots) != 36:
        raise ValueError("tray geometry must contain 36 slots")
    if half_extent_mm <= 0.0:
        raise ValueError("slot half extent must be positive")
    result: dict[str, SlotProjection] = {}
    for slot_key in sorted(raw_slots):
        center = np.asarray(raw_slots[slot_key], dtype=np.float64).reshape(3)
        x, y, z = (float(value) for value in center)
        half = float(half_extent_mm)
        polygon_T = np.asarray(
            [
                [x + half, y + half, z],
                [x + half, y - half, z],
                [x - half, y - half, z],
                [x - half, y + half, z],
            ],
            dtype=np.float64,
        )
        tray_xy = np.vstack((center[:2], polygon_T[:, :2])).astype(np.float32)
        projected = cv2.perspectiveTransform(
            tray_xy.reshape(1, -1, 2), homography
        ).reshape(-1, 2)
        if not np.all(np.isfinite(projected)):
            raise ValueError("planar tray registration produced non-finite slots")
        center_px = projected[0]
        polygon_px = projected[1:]
        area = abs(float(cv2.contourArea(polygon_px.astype(np.float32))))
        result[slot_key] = SlotProjection(
            slot_key=slot_key,
            row=int(slot_key[1]),
            column=int(slot_key[2]),
            center_T_mm=tuple(float(value) for value in center),
            center_px=tuple(float(value) for value in center_px),
            polygon_T_mm=tuple(
                tuple(float(value) for value in point) for point in polygon_T
            ),
            polygon_px=tuple(
                tuple(float(value) for value in point) for point in polygon_px
            ),
            image_coverage_ratio=_polygon_coverage_ratio(polygon_px, image_shape),
            projected_area_px=area,
        )
    return result


__all__ = [
    "PlanarTrayRegistration",
    "build_planar_slot_projections",
    "estimate_planar_tray_registration",
]
