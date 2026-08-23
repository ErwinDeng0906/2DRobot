"""Review wafer placement from an arbitrary-resolution tray photograph.

This path is deliberately separate from the calibrated camera-1 robot path.
It fits a projective tray grid from the fixed IDs printed in the 36 slots and
uses that fit only to normalize wafer patches and produce review labels.  It
never produces a world transform, suction correction, or motion permission.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from .slot_marker_observation import (
    SlotMarkerLayout,
    detect_aruco_observations,
)
from .tray_occupancy import SlotState
from .tray_vision_fusion import TrayVisionFusionConfig
from .wafer_shape_quality import (
    WaferObservation,
    analyze_dark_wafer_patch,
    analyze_wafer_patch,
)


MINIMUM_REVIEW_MARKER_COUNT = 3
MINIMUM_REVIEW_INLIER_COUNT = 3
MINIMUM_REVIEW_CORRESPONDENCE_COUNT = 6
MINIMUM_REVIEW_AXIS_SPAN_MM = 60.0
MINIMUM_CENTER_FALLBACK_MARKER_COUNT = 4


@dataclass(frozen=True)
class MarkerGridFit:
    success: bool
    failure_reason: Optional[str]
    homography_image_from_tray_xy: Optional[np.ndarray]
    visible_marker_count: int
    inlier_marker_count: int
    correspondence_count: int
    inlier_correspondence_count: int
    reprojection_rms_px: Optional[float]
    reprojection_max_px: Optional[float]
    marker_ids: tuple[int, ...]
    inlier_marker_ids: tuple[int, ...] = ()
    fit_method: str = "center_plus_outer_corners"

    def to_json(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failure_reason": self.failure_reason,
            "visible_marker_count": self.visible_marker_count,
            "inlier_marker_count": self.inlier_marker_count,
            "correspondence_count": self.correspondence_count,
            "inlier_correspondence_count": self.inlier_correspondence_count,
            "reprojection_rms_px": self.reprojection_rms_px,
            "reprojection_max_px": self.reprojection_max_px,
            "marker_ids": list(self.marker_ids),
            "inlier_marker_ids": list(self.inlier_marker_ids),
            "fit_method": self.fit_method,
            "homography_image_from_tray_xy": (
                None
                if self.homography_image_from_tray_xy is None
                else self.homography_image_from_tray_xy.astype(float).tolist()
            ),
        }


@dataclass(frozen=True)
class MarkerGridSlotReview:
    slot_key: str
    expected_marker_id: int
    marker_visible: bool
    state: str
    placement_state: str
    stacking_state: str
    center_px: tuple[float, float]
    polygon_px: tuple[tuple[float, float], ...]
    image_coverage_ratio: float
    wafer: WaferObservation
    wafer_box_image_px: tuple[tuple[float, float], ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "slot_key": self.slot_key,
            "expected_marker_id": self.expected_marker_id,
            "marker_visible": self.marker_visible,
            "state": self.state,
            "placement_state": self.placement_state,
            "stacking_state": self.stacking_state,
            "center_px": list(self.center_px),
            "polygon_px": [list(point) for point in self.polygon_px],
            "image_coverage_ratio": self.image_coverage_ratio,
            "wafer": self.wafer.to_json(),
            "wafer_box_image_px": [
                list(point) for point in self.wafer_box_image_px
            ],
        }


@dataclass(frozen=True)
class MarkerGridReviewResult:
    success: bool
    failure_reason: Optional[str]
    fit: MarkerGridFit
    slots: tuple[MarkerGridSlotReview, ...]
    summary: dict[str, int]
    annotated_image: np.ndarray

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": "arbitrary_resolution_review_only",
            "success": self.success,
            "failure_reason": self.failure_reason,
            "coordinate_mapping_allowed": False,
            "robot_motion_authorized": False,
            "fit": self.fit.to_json(),
            "summary": dict(self.summary),
            "slots": [slot.to_json() for slot in self.slots],
        }


def _failed_fit(
    reason: str,
    visible: int = 0,
    correspondences: int = 0,
) -> MarkerGridFit:
    return MarkerGridFit(
        False,
        reason,
        None,
        visible,
        0,
        correspondences,
        0,
        None,
        None,
        (),
    )


def _fit_marker_centres_only(
    tray_points: np.ndarray,
    image_points: np.ndarray,
    marker_ids: list[int],
    visible_marker_count: int,
) -> Optional[MarkerGridFit]:
    """Checked fallback when outer-marker corners distort point RANSAC."""

    if len(tray_points) < MINIMUM_CENTER_FALLBACK_MARKER_COUNT:
        return None
    homography, inlier_mask = cv2.findHomography(
        tray_points,
        image_points,
        cv2.RANSAC,
        6.0,
    )
    if homography is None or inlier_mask is None:
        return None
    inliers = inlier_mask.reshape(-1) > 0
    projected = cv2.perspectiveTransform(
        tray_points[None, :, :], homography
    )[0]
    errors = np.linalg.norm(projected - image_points, axis=1)
    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count == 0:
        return None
    rms = float(np.sqrt(np.mean(np.square(errors[inliers]))))
    maximum = float(np.max(errors[inliers]))
    inlier_ratio = inlier_count / max(len(tray_points), 1)
    inlier_marker_ids = tuple(
        marker_id
        for marker_id, inlier in zip(marker_ids, inliers)
        if bool(inlier)
    )
    if (
        inlier_count < MINIMUM_REVIEW_INLIER_COUNT
        or inlier_ratio < 0.50
        or rms > 3.5
        or maximum > 7.0
    ):
        return None
    return MarkerGridFit(
        True,
        None,
        homography,
        visible_marker_count,
        len(inlier_marker_ids),
        len(tray_points),
        inlier_count,
        rms,
        maximum,
        tuple(marker_ids),
        inlier_marker_ids,
        "marker_centres_only_fallback",
    )


def _fit_marker_grid(
    image: np.ndarray,
    geometry: Mapping[str, Any],
    layout: SlotMarkerLayout,
) -> tuple[MarkerGridFit, dict[int, Any]]:
    # Preserve the audited 363-image review detector's largest-candidate
    # duplicate policy. Task14/live fallback uses the safer native-first policy.
    observations = detect_aruco_observations(
        image, layout.dictionary_name, prefer_native=False
    )
    slots = geometry.get("slots")
    if not isinstance(slots, Mapping):
        return _failed_fit("tray geometry does not contain slots"), observations
    tray_points = []
    image_points = []
    marker_ids = []
    point_marker_ids = []
    marker_center_tray_points = []
    marker_center_image_points = []
    outer_marker_ids: set[int] = set()
    used_ids: set[int] = set()
    for slot_key, marker_id in sorted(layout.marker_id_by_slot.items()):
        observation = observations.get(marker_id)
        center = slots.get(slot_key)
        if observation is None or center is None:
            continue
        tray_points.append([float(center[0]), float(center[1])])
        image_points.append(observation.center_px)
        marker_ids.append(int(marker_id))
        point_marker_ids.append(int(marker_id))
        marker_center_tray_points.append(
            [float(center[0]), float(center[1])]
        )
        marker_center_image_points.append(observation.center_px)
        used_ids.add(int(marker_id))
    # A full tray hides most inner markers. The rigid outer A-H markers remain
    # valid projective anchors for review images, using their checked-in Tray
    # coordinates and fixed IDs. Their small measured Z offsets are irrelevant
    # to this review-only planar normalization and never feed robot geometry.
    board_markers = geometry.get("markers")
    if isinstance(board_markers, Mapping):
        for marker in board_markers.values():
            if not isinstance(marker, Mapping):
                continue
            marker_id = int(marker.get("id", -1))
            outer_marker_ids.add(marker_id)
            observation = observations.get(marker_id)
            center = marker.get("center_T_mm")
            if (
                marker_id in used_ids
                or observation is None
                or not isinstance(center, (list, tuple))
                or len(center) < 2
            ):
                continue
            tray_points.append([float(center[0]), float(center[1])])
            image_points.append(observation.center_px)
            marker_ids.append(marker_id)
            point_marker_ids.append(marker_id)
            marker_center_tray_points.append(
                [float(center[0]), float(center[1])]
            )
            marker_center_image_points.append(observation.center_px)
            corners_T = marker.get("corners_T_mm")
            if isinstance(corners_T, (list, tuple)) and len(corners_T) == 4:
                for tray_corner, image_corner in zip(
                    corners_T,
                    observation.corners_px,
                ):
                    tray_points.append(
                        [float(tray_corner[0]), float(tray_corner[1])]
                    )
                    image_points.append(image_corner)
                    point_marker_ids.append(marker_id)
            used_ids.add(marker_id)
    visible_marker_count = len(set(marker_ids))
    center_source_for_span = np.asarray(
        marker_center_tray_points, dtype=np.float32
    )
    preliminary_span_x_mm = (
        0.0
        if len(center_source_for_span) < 2
        else float(np.ptp(center_source_for_span[:, 0]))
    )
    preliminary_span_y_mm = (
        0.0
        if len(center_source_for_span) < 2
        else float(np.ptp(center_source_for_span[:, 1]))
    )
    two_outer_marker_geometry = bool(
        visible_marker_count == 2
        and set(marker_ids).issubset(outer_marker_ids)
        and all(
            point_marker_ids.count(marker_id) >= 5
            for marker_id in marker_ids
        )
        and preliminary_span_x_mm >= MINIMUM_REVIEW_AXIS_SPAN_MM
        and preliminary_span_y_mm >= MINIMUM_REVIEW_AXIS_SPAN_MM
    )
    if (
        visible_marker_count < MINIMUM_REVIEW_MARKER_COUNT
        and not two_outer_marker_geometry
    ):
        return (
            _failed_fit(
                "fixed tray markers are insufficient for a checked residual: "
                f"{visible_marker_count}/{MINIMUM_REVIEW_MARKER_COUNT}",
                visible_marker_count,
                len(tray_points),
            ),
            observations,
        )
    if len(tray_points) < MINIMUM_REVIEW_CORRESPONDENCE_COUNT:
        return (
            _failed_fit(
                "fixed tray marker geometry has too few independent "
                f"correspondences: {len(tray_points)}/"
                f"{MINIMUM_REVIEW_CORRESPONDENCE_COUNT}",
                visible_marker_count,
                len(tray_points),
            ),
            observations,
        )
    source = np.asarray(tray_points, dtype=np.float32)
    target = np.asarray(image_points, dtype=np.float32)
    center_source = np.asarray(marker_center_tray_points, dtype=np.float32)
    center_target = np.asarray(marker_center_image_points, dtype=np.float32)
    span_x_mm = float(np.ptp(center_source[:, 0]))
    span_y_mm = float(np.ptp(center_source[:, 1]))
    if (
        span_x_mm < MINIMUM_REVIEW_AXIS_SPAN_MM
        or span_y_mm < MINIMUM_REVIEW_AXIS_SPAN_MM
    ):
        return (
            _failed_fit(
                "visible markers do not span both tray axes: "
                f"x={span_x_mm:.1f}mm, y={span_y_mm:.1f}mm, "
                f"required={MINIMUM_REVIEW_AXIS_SPAN_MM:.1f}mm",
                visible_marker_count,
                len(source),
            ),
            observations,
        )
    homography, inlier_mask = cv2.findHomography(
        source,
        target,
        cv2.RANSAC,
        4.0,
    )
    if homography is None or inlier_mask is None:
        return _failed_fit(
            "marker-grid homography failed",
            visible_marker_count,
            len(source),
        ), observations
    inliers = inlier_mask.reshape(-1) > 0
    projected = cv2.perspectiveTransform(source[None, :, :], homography)[0]
    errors = np.linalg.norm(projected - target, axis=1)
    inlier_count = int(np.count_nonzero(inliers))
    point_ids = np.asarray(point_marker_ids, dtype=np.int32)
    inlier_marker_ids = []
    for marker_id in sorted(set(marker_ids)):
        marker_points = point_ids == marker_id
        required = max(1, int(math.ceil(0.50 * int(np.count_nonzero(marker_points)))))
        if int(np.count_nonzero(inliers & marker_points)) >= required:
            inlier_marker_ids.append(marker_id)
    inlier_marker_count = len(inlier_marker_ids)
    required_inlier_marker_count = (
        2 if two_outer_marker_geometry else MINIMUM_REVIEW_INLIER_COUNT
    )
    rms = float(np.sqrt(np.mean(np.square(errors[inliers]))))
    maximum = float(np.max(errors[inliers]))
    inlier_ratio = inlier_count / max(len(source), 1)
    if (
        inlier_marker_count < required_inlier_marker_count
        or inlier_count < MINIMUM_REVIEW_CORRESPONDENCE_COUNT
        or inlier_ratio < 0.50
        or rms > 4.0
        or maximum > 8.0
    ):
        centre_fit = _fit_marker_centres_only(
            center_source,
            center_target,
            marker_ids,
            visible_marker_count,
        )
        if centre_fit is not None:
            return centre_fit, observations
        fit = MarkerGridFit(
            False,
            (
                "marker-grid residual rejected: "
                f"markers={inlier_marker_count}/{visible_marker_count}, "
                f"points={inlier_count}/{len(source)}, rms={rms:.3f}px, "
                f"max={maximum:.3f}px"
            ),
            homography,
            visible_marker_count,
            inlier_marker_count,
            len(source),
            inlier_count,
            rms,
            maximum,
            tuple(marker_ids),
            tuple(inlier_marker_ids),
        )
        return fit, observations
    if two_outer_marker_geometry:
        fit_method = "two_outer_marker_corner_geometry"
    elif visible_marker_count == 3:
        fit_method = "three_marker_corner_geometry"
    else:
        fit_method = "center_plus_outer_corners"
    fit = MarkerGridFit(
        True,
        None,
        homography,
        visible_marker_count,
        inlier_marker_count,
        len(source),
        inlier_count,
        rms,
        maximum,
        tuple(marker_ids),
        tuple(inlier_marker_ids),
        fit_method,
    )
    return fit, observations


def _polygon_coverage(
    polygon: np.ndarray,
    image_shape: tuple[int, ...],
) -> float:
    height, width = image_shape[:2]
    full_area = abs(float(cv2.contourArea(polygon.astype(np.float32))))
    if full_area <= 1.0:
        return 0.0
    image_polygon = np.asarray(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float32,
    )
    overlap, _ = cv2.intersectConvexConvex(
        polygon.astype(np.float32), image_polygon
    )
    return float(max(0.0, min(1.0, float(overlap) / full_area)))


def _state_for_review(
    wafer: WaferObservation,
    *,
    marker_visible: bool,
    coverage: float,
    minimum_coverage: float,
) -> str:
    if coverage < minimum_coverage:
        return SlotState.OUT_OF_VIEW.value
    if wafer.found:
        stacked = bool(
            "stacked_geometry_confirmed" in wafer.flags
            and "second_quadrilateral" in wafer.flags
            and "oversize_footprint" not in wafer.flags
            and "boundary_geometry_extrapolated" not in wafer.flags
            and "marker_center_fallback_geometry" not in wafer.flags
            and "sparse_three_marker_geometry" not in wafer.flags
            and "sparse_two_outer_marker_geometry" not in wafer.flags
        )
        if wafer.outside_slot and stacked:
            return SlotState.STACKED_OUTSIDE_SLOT.value
        if wafer.outside_slot:
            return SlotState.OUTSIDE_SLOT.value
        if stacked:
            return SlotState.STACKED.value
        if wafer.quality == "normal":
            return SlotState.OCCUPIED.value
        return SlotState.WARNING.value
    if marker_visible:
        return SlotState.EMPTY.value
    return SlotState.UNKNOWN.value


def _independent_states_for_review(
    wafer: WaferObservation,
    *,
    marker_visible: bool,
    coverage: float,
    minimum_coverage: float,
) -> tuple[str, str]:
    """Keep placement and stacking evidence separate for dataset auditing."""

    if coverage < minimum_coverage:
        return "unobservable", "unobservable"
    if wafer.found:
        flags = set(wafer.flags)
        if wafer.outside_slot:
            placement = "outside"
        elif flags & {
            "boundary_crossing_unconfirmed",
            "boundary_geometry_extrapolated",
            "boundary_clearance_uncertain",
            "boundary_uncertain",
            "boundary_fallback_geometry_unconfirmed",
            "dark_low_chroma_fallback",
        }:
            placement = "uncertain"
        else:
            placement = "inside"
        if (
            "stacked_geometry_confirmed" in flags
            and "second_quadrilateral" in flags
            and "oversize_footprint" not in flags
            and "boundary_geometry_extrapolated" not in flags
            and "marker_center_fallback_geometry" not in flags
            and "sparse_three_marker_geometry" not in flags
            and "sparse_two_outer_marker_geometry" not in flags
        ):
            stacking = "confirmed"
        elif (
            "oversize_footprint" in flags
            and 0.62 < float(wafer.side_ratio) <= 0.72
            and float(wafer.center_offset_ratio) <= 0.14
            and float(wafer.solidity) >= 0.90
            and "sparse_two_outer_marker_geometry" not in flags
            and (
                bool(
                    flags & {
                        "stacked_geometry_confirmed",
                        "internal_overlap_edges",
                        "multiple_components",
                    }
                )
                or (
                    float(wafer.rectangularity) >= 0.97
                    and float(wafer.solidity) >= 0.90
                )
            )
        ):
            stacking = "suspected"
        elif (
            "oversize_footprint" in flags
            or (
                bool(wafer.outside_slot)
                and bool(
                    flags & {
                        "irregular_outline",
                        "multiple_components",
                        "non_square_aspect",
                        "solidity_borderline",
                    }
                )
            )
            or (
                "multiple_components" in flags
                and "non_square_aspect" in flags
            )
            or (
                "boundary_uncertain" in flags
                and bool(
                    flags
                    & {
                        "multiple_components",
                        "internal_overlap_edges",
                        "irregular_outline",
                    }
                )
            )
        ):
            stacking = "uncertain"
        else:
            stacking = "single"
        return placement, stacking
    if marker_visible:
        return "empty", "not_applicable"
    return "unknown", "unknown"


def _convex_iou(first: np.ndarray, second: np.ndarray) -> float:
    """Return image-independent IoU for two convex tray-plane footprints."""

    first_area = abs(float(cv2.contourArea(first.astype(np.float32))))
    second_area = abs(float(cv2.contourArea(second.astype(np.float32))))
    if first_area <= 1e-6 or second_area <= 1e-6:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(
        first.astype(np.float32), second.astype(np.float32)
    )
    union = first_area + second_area - float(intersection)
    return float(intersection) / max(union, 1e-6)


def _reconcile_slot_candidates(
    provisional: list[dict[str, Any]],
    slot_centers: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, str]]:
    """Assign each physical wafer footprint to at most one tray slot.

    Canonical slot patches overlap because they must retain the corners of an
    off-centre wafer. Consequently, the same physical wafer can be detected
    independently in neighbouring patches. Candidate centres and footprints
    are already expressed in tray millimetres here, so duplicate suppression
    remains stable under camera zoom and perspective changes.
    """

    candidate_indices = [
        index
        for index, item in enumerate(provisional)
        if item["wafer"].found
        and item["coverage"] >= item["minimum_coverage"]
        and item["candidate_center_tray"] is not None
        and item["candidate_box_tray"] is not None
    ]
    if not candidate_indices:
        return {}, {}

    parent = {index: index for index in candidate_indices}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    # Slot pitch is 25 mm. Seven millimetres is comfortably below the
    # separation of two real neighbouring wafers; the wider gate is accepted
    # only when their fitted quadrilaterals also overlap substantially.
    for offset, first_index in enumerate(candidate_indices):
        first = provisional[first_index]
        first_center = np.asarray(first["candidate_center_tray"], dtype=np.float64)
        first_box = np.asarray(first["candidate_box_tray"], dtype=np.float32)
        for second_index in candidate_indices[offset + 1 :]:
            second = provisional[second_index]
            second_center = np.asarray(second["candidate_center_tray"], dtype=np.float64)
            distance_mm = float(np.linalg.norm(first_center - second_center))
            if distance_mm <= 9.0:
                union(first_index, second_index)
                continue
            if distance_mm <= 18.0:
                second_box = np.asarray(second["candidate_box_tray"], dtype=np.float32)
                if _convex_iou(first_box, second_box) >= 0.12:
                    union(first_index, second_index)

    clusters: dict[int, list[int]] = {}
    for index in candidate_indices:
        clusters.setdefault(find(index), []).append(index)

    assignment_options: list[tuple[float, int, str, int]] = []
    cluster_centres: dict[int, np.ndarray] = {}
    for cluster_id, members in clusters.items():
        centres = np.asarray(
            [provisional[index]["candidate_center_tray"] for index in members],
            dtype=np.float64,
        )
        cluster_center = np.median(centres, axis=0)
        cluster_centres[cluster_id] = cluster_center
        for index in members:
            slot_key = str(provisional[index]["slot_key"])
            slot_center = np.asarray(slot_centers[slot_key][:2], dtype=np.float64)
            distance_mm = float(np.linalg.norm(cluster_center - slot_center))
            # A 31 mm review patch can only provide trustworthy ownership to a
            # slot whose centre is at most one patch half-width away.
            if distance_mm <= 15.5:
                assignment_options.append(
                    (distance_mm, cluster_id, slot_key, index)
                )

    assignment_options.sort(key=lambda item: item[0])
    assigned_clusters: set[int] = set()
    assigned_slots: set[str] = set()
    owners: dict[str, int] = {}
    duplicate_reasons: dict[str, str] = {}
    for _distance, cluster_id, slot_key, source_index in assignment_options:
        if cluster_id in assigned_clusters or slot_key in assigned_slots:
            continue
        members = clusters[cluster_id]
        owner_member = min(
            members,
            key=lambda index: (
                0 if provisional[index]["slot_key"] == slot_key else 1,
                float(
                    np.linalg.norm(
                        np.asarray(
                            provisional[index]["candidate_center_tray"],
                            dtype=np.float64,
                        )
                        - cluster_centres[cluster_id]
                    )
                ),
                -float(provisional[index]["wafer"].confidence),
            ),
        )
        # Prefer the observation made in the assigned slot. It measures
        # clearance relative to the correct acceptance square.
        if provisional[source_index]["slot_key"] == slot_key:
            owner_member = source_index
        owners[slot_key] = owner_member
        assigned_clusters.add(cluster_id)
        assigned_slots.add(slot_key)
        for member in members:
            member_slot = str(provisional[member]["slot_key"])
            if member != owner_member:
                duplicate_reasons[member_slot] = slot_key

    return owners, duplicate_reasons


def _inlier_support_hull(
    fit: MarkerGridFit,
    geometry: Mapping[str, Any],
    layout: SlotMarkerLayout,
) -> Optional[np.ndarray]:
    """Convex tray-plane region supported by inlier marker centres."""

    slots = geometry.get("slots", {})
    tray_by_id: dict[int, tuple[float, float]] = {}
    for slot_key, marker_id in layout.marker_id_by_slot.items():
        center = slots.get(slot_key)
        if center is not None:
            tray_by_id[int(marker_id)] = (float(center[0]), float(center[1]))
    markers = geometry.get("markers", {})
    if isinstance(markers, Mapping):
        for marker in markers.values():
            if not isinstance(marker, Mapping):
                continue
            center = marker.get("center_T_mm")
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                tray_by_id[int(marker["id"])] = (
                    float(center[0]),
                    float(center[1]),
                )
    points = np.asarray(
        [
            tray_by_id[marker_id]
            for marker_id in fit.inlier_marker_ids
            if marker_id in tray_by_id
        ],
        dtype=np.float32,
    )
    if len(points) < 3:
        return None
    hull = cv2.convexHull(points)
    if len(hull) < 3 or abs(float(cv2.contourArea(hull))) < 1.0:
        return None
    return hull


def _demote_unsupported_boundary(
    wafer: WaferObservation,
    *,
    support_distance_mm: Optional[float],
    visible_marker_count: int,
    fallback_geometry: bool,
) -> WaferObservation:
    """Reject certain boundary claims made outside calibrated support."""

    if not wafer.found or not wafer.outside_slot:
        return wafer
    clearance = wafer.minimum_slot_clearance_ratio
    extrapolated = (
        support_distance_mm is None or support_distance_mm < -20.0
    )
    small_violation_with_sparse_geometry = bool(
        clearance is not None
        and clearance > -0.025
        and visible_marker_count < 20
    )
    if (
        not extrapolated
        and not small_violation_with_sparse_geometry
        and not fallback_geometry
    ):
        return wafer
    flags = [flag for flag in wafer.flags if flag != "outside_slot"]
    if extrapolated:
        flags.append("boundary_geometry_extrapolated")
    if small_violation_with_sparse_geometry:
        flags.append("boundary_clearance_uncertain")
    if fallback_geometry:
        flags.append("boundary_fallback_geometry_unconfirmed")
    return replace(
        wafer,
        quality="warning",
        confidence=min(float(wafer.confidence), 0.45),
        flags=tuple(dict.fromkeys(flags)),
        outside_slot=False,
    )


def review_marker_grid_image(
    image: np.ndarray,
    geometry: Mapping[str, Any],
    layout: SlotMarkerLayout,
    config: TrayVisionFusionConfig,
) -> MarkerGridReviewResult:
    """Normalize all slots from inner-marker geometry for offline review."""

    if image is None or image.size == 0:
        fit = _failed_fit("empty image")
        return MarkerGridReviewResult(
            False,
            fit.failure_reason,
            fit,
            (),
            {"unknown": 36},
            np.zeros((1, 1, 3), dtype=np.uint8),
        )
    canvas = image.copy()
    fit, observations = _fit_marker_grid(image, geometry, layout)
    if not fit.success or fit.homography_image_from_tray_xy is None:
        cv2.putText(
            canvas,
            f"REVIEW REJECTED: {fit.failure_reason}",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return MarkerGridReviewResult(
            False,
            fit.failure_reason,
            fit,
            (),
            {"unknown": 36},
            canvas,
        )

    slots = geometry["slots"]
    output_size = int(config.canonical_patch_size)
    half = float(config.slot_half_extent_mm)
    target_quad = np.asarray(
        [[0.0, 0.0], [output_size - 1.0, 0.0],
         [output_size - 1.0, output_size - 1.0], [0.0, output_size - 1.0]],
        dtype=np.float32,
    )
    reviews = []
    counts: dict[str, int] = {}
    colours = {
        SlotState.EMPTY.value: (50, 180, 60),
        SlotState.OCCUPIED.value: (255, 0, 255),
        SlotState.WARNING.value: (0, 180, 255),
        SlotState.OUTSIDE_SLOT.value: (0, 0, 255),
        SlotState.STACKED.value: (0, 0, 255),
        SlotState.STACKED_OUTSIDE_SLOT.value: (0, 0, 200),
        SlotState.OUT_OF_VIEW.value: (110, 110, 110),
        SlotState.UNKNOWN.value: (255, 180, 0),
    }
    provisional: list[dict[str, Any]] = []
    image_to_tray = np.linalg.inv(fit.homography_image_from_tray_xy)
    support_hull = _inlier_support_hull(fit, geometry, layout)
    minimum_coverage = config.slot_decision.minimum_image_coverage_ratio
    for slot_key, center_raw in sorted(slots.items()):
        center = np.asarray(center_raw[:2], dtype=np.float32)
        tray_quad = np.asarray(
            [
                [center[0] + half, center[1] + half],
                [center[0] + half, center[1] - half],
                [center[0] - half, center[1] - half],
                [center[0] - half, center[1] + half],
            ],
            dtype=np.float32,
        )
        image_quad = cv2.perspectiveTransform(
            tray_quad[None, :, :], fit.homography_image_from_tray_xy
        )[0]
        image_to_patch = cv2.getPerspectiveTransform(image_quad, target_quad)
        patch = cv2.warpPerspective(
            image,
            image_to_patch,
            (output_size, output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        coverage = _polygon_coverage(image_quad, image.shape)
        marker_id = int(layout.marker_id_by_slot[slot_key])
        marker_visible = marker_id in observations
        wafer = (
            WaferObservation.not_found("expected_marker_visible")
            if marker_visible
            else analyze_wafer_patch(patch, config.wafer_quality)
        )
        if (
            not marker_visible
            and coverage
            >= config.slot_decision.minimum_image_coverage_ratio
        ):
            dark_fallback = analyze_dark_wafer_patch(
                patch, config.wafer_quality
            )
            if dark_fallback.found and (
                not wafer.found
                or (
                    wafer.quality != "normal"
                    and dark_fallback.quality == "normal"
                    and dark_fallback.side_ratio >= 0.40
                )
            ):
                wafer = dark_fallback
        support_distance_mm = (
            None
            if support_hull is None
            else float(
                cv2.pointPolygonTest(
                    support_hull,
                    (float(center[0]), float(center[1])),
                    True,
                )
            )
        )
        low_confidence_geometry = fit.fit_method in {
            "marker_centres_only_fallback",
            "three_marker_corner_geometry",
            "two_outer_marker_corner_geometry",
        }
        if wafer.found and low_confidence_geometry:
            if fit.fit_method == "marker_centres_only_fallback":
                geometry_flag = "marker_center_fallback_geometry"
            elif fit.fit_method == "three_marker_corner_geometry":
                geometry_flag = "sparse_three_marker_geometry"
            else:
                geometry_flag = "sparse_two_outer_marker_geometry"
            wafer = replace(
                wafer,
                confidence=min(float(wafer.confidence), 0.55),
                flags=tuple(
                    dict.fromkeys(
                        wafer.flags + (geometry_flag,)
                    )
                ),
            )
        wafer = _demote_unsupported_boundary(
            wafer,
            support_distance_mm=support_distance_mm,
            visible_marker_count=fit.visible_marker_count,
            fallback_geometry=low_confidence_geometry,
        )
        patch_to_image = np.linalg.inv(image_to_patch)
        wafer_box_image: tuple[tuple[float, float], ...] = ()
        candidate_center_tray: Optional[tuple[float, float]] = None
        candidate_box_tray: Optional[tuple[tuple[float, float], ...]] = None
        if wafer.box_patch_px:
            mapped = cv2.perspectiveTransform(
                np.asarray(wafer.box_patch_px, dtype=np.float32)[None, :, :],
                patch_to_image,
            )[0]
            wafer_box_image = tuple(
                (float(point[0]), float(point[1])) for point in mapped
            )
            mapped_tray = cv2.perspectiveTransform(
                mapped.astype(np.float32)[None, :, :], image_to_tray
            )[0]
            candidate_box_tray = tuple(
                (float(point[0]), float(point[1])) for point in mapped_tray
            )
        if wafer.center_patch_px is not None:
            candidate_center_image = cv2.perspectiveTransform(
                np.asarray(wafer.center_patch_px, dtype=np.float32).reshape(1, 1, 2),
                patch_to_image,
            )[0, 0]
            mapped_center_tray = cv2.perspectiveTransform(
                candidate_center_image.reshape(1, 1, 2), image_to_tray
            )[0, 0]
            candidate_center_tray = (
                float(mapped_center_tray[0]),
                float(mapped_center_tray[1]),
            )
        center_px = cv2.perspectiveTransform(
            center.reshape(1, 1, 2), fit.homography_image_from_tray_xy
        )[0, 0]
        provisional.append(
            {
                "slot_key": slot_key,
                "marker_id": marker_id,
                "marker_visible": marker_visible,
                "center_px": (float(center_px[0]), float(center_px[1])),
                "image_quad": image_quad,
                "coverage": coverage,
                "minimum_coverage": minimum_coverage,
                "wafer": wafer,
                "wafer_box_image": wafer_box_image,
                "candidate_center_tray": candidate_center_tray,
                "candidate_box_tray": candidate_box_tray,
            }
        )

    owners, duplicate_reasons = _reconcile_slot_candidates(provisional, slots)
    for index, item in enumerate(provisional):
        slot_key = str(item["slot_key"])
        wafer = item["wafer"]
        wafer_box_image = item["wafer_box_image"]
        if wafer.found and owners.get(slot_key) != index:
            assigned_slot = duplicate_reasons.get(slot_key)
            wafer = WaferObservation.not_found(
                "duplicate_candidate_suppressed"
                if assigned_slot is None
                else f"duplicate_candidate_assigned_to_{assigned_slot}"
            )
            wafer_box_image = ()
        state = _state_for_review(
            wafer,
            marker_visible=bool(item["marker_visible"]),
            coverage=float(item["coverage"]),
            minimum_coverage=minimum_coverage,
        )
        placement_state, stacking_state = _independent_states_for_review(
            wafer,
            marker_visible=bool(item["marker_visible"]),
            coverage=float(item["coverage"]),
            minimum_coverage=minimum_coverage,
        )
        counts[state] = counts.get(state, 0) + 1
        if wafer_box_image:
            cv2.polylines(
                canvas,
                [np.round(np.asarray(wafer_box_image)).astype(np.int32)],
                True,
                colours[state],
                3,
                cv2.LINE_AA,
            )
        cv2.polylines(
            canvas,
            [np.round(item["image_quad"]).astype(np.int32)],
            True,
            colours[state],
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{slot_key} {state}",
            (
                int(round(item["center_px"][0] - 34)),
                int(round(item["center_px"][1] + 5)),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            colours[state],
            2,
            cv2.LINE_AA,
        )
        reviews.append(
            MarkerGridSlotReview(
                slot_key,
                int(item["marker_id"]),
                bool(item["marker_visible"]),
                state,
                placement_state,
                stacking_state,
                item["center_px"],
                tuple(
                    (float(point[0]), float(point[1]))
                    for point in item["image_quad"]
                ),
                float(item["coverage"]),
                wafer,
                wafer_box_image,
            )
        )
    banner = (
        "REVIEW ONLY | no robot coordinates | "
        f"markers={fit.inlier_marker_count}/{fit.visible_marker_count} | "
        f"rms={fit.reprojection_rms_px:.2f}px | "
        f"normal={counts.get(SlotState.OCCUPIED.value, 0)} | "
        f"outside={counts.get(SlotState.OUTSIDE_SLOT.value, 0)}"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 40), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        banner,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return MarkerGridReviewResult(True, None, fit, tuple(reviews), counts, canvas)


__all__ = [
    "MINIMUM_REVIEW_MARKER_COUNT",
    "MINIMUM_REVIEW_INLIER_COUNT",
    "MINIMUM_REVIEW_AXIS_SPAN_MM",
    "MarkerGridFit",
    "MarkerGridReviewResult",
    "MarkerGridSlotReview",
    "review_marker_grid_image",
]
