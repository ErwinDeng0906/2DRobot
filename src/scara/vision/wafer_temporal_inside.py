"""Fail-closed temporal confirmation for a stable single wafer inside a tray slot.

The single-frame detector deliberately keeps a three-canonical-pixel boundary
dead band.  This module does not shrink that band.  It only allows a read-only
consumer to recover a warning after five fresh frames agree geometrically and
after a separate multi-view latch has established that strong outside evidence
does not dominate the physical slot.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Optional, Sequence

import cv2
import numpy as np

from .wafer_shape_quality import (
    BOUNDARY_CONTOUR_MIN_AREA_RATIO,
    BOUNDARY_CONTOUR_MIN_DEPTH_PX,
    BOUNDARY_CONTOUR_MIN_SUPPORT_PX,
    BOUNDARY_UNCERTAINTY_PX,
    WaferQualityConfig,
)


_HARD_WARNING_FLAGS = frozenset(
    {
        "stacked_geometry_confirmed",
        "l_shape_five_frame_confirmed",
        "contradictory_marker_and_wafer",
    }
)


def _finite_optional(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _point_tuple(value: object, length: int) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    rows: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, Sequence) or len(item) != 2:
            return ()
        x = _finite_optional(item[0])
        y = _finite_optional(item[1])
        if x is None or y is None:
            return ()
        rows.append((x, y))
    return tuple(rows) if len(rows) == length else ()


def _convex_iou(
    first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]
) -> float:
    first_polygon = np.asarray(first, dtype=np.float32).reshape(4, 2)
    second_polygon = np.asarray(second, dtype=np.float32).reshape(4, 2)
    first_area = abs(float(cv2.contourArea(first_polygon)))
    second_area = abs(float(cv2.contourArea(second_polygon)))
    intersection_area, _intersection = cv2.intersectConvexConvex(
        cv2.convexHull(first_polygon), cv2.convexHull(second_polygon)
    )
    union = first_area + second_area - float(intersection_area)
    return float(intersection_area / max(union, 1.0))


@dataclass(frozen=True)
class TemporalInsideFrameEvidence:
    frame_id: Hashable
    raw_state: str
    found: bool
    center_patch_px: Optional[tuple[float, float]] = None
    box_patch_px: tuple[tuple[float, float], ...] = ()
    yaw_deg: Optional[float] = None
    flags: tuple[str, ...] = ()
    boundary_evidence: str = "unobservable"
    base_clearance_px: Optional[float] = None
    refined_clearance_px: Optional[float] = None
    base_boundary_evidence: str = "unobservable"
    refined_boundary_evidence: str = "unobservable"
    contour_depth_px: float = 0.0
    contour_support_px: int = 0
    contour_area_ratio: float = 0.0
    base_contour_depth_px: Optional[float] = None
    base_contour_support_px: Optional[int] = None
    base_contour_area_ratio: Optional[float] = None
    refined_contour_depth_px: Optional[float] = None
    refined_contour_support_px: Optional[int] = None
    refined_contour_area_ratio: Optional[float] = None
    accepted_secondary_candidate: bool = False

    @classmethod
    def from_json(
        cls,
        frame_id: Hashable,
        raw_state: str,
        wafer: Mapping[str, Any],
        *,
        decision_flags: Sequence[object] = (),
    ) -> "TemporalInsideFrameEvidence":
        flags = tuple(
            sorted(
                {
                    str(value)
                    for value in tuple(wafer.get("flags", ()))
                    + tuple(decision_flags)
                }
            )
        )
        candidates = wafer.get("secondary_candidates", ())
        accepted_secondary = bool(
            isinstance(candidates, Sequence)
            and not isinstance(candidates, (str, bytes))
            and any(
                isinstance(candidate, Mapping)
                and bool(candidate.get("accepted"))
                and str(candidate.get("source") or "") != "l_shape"
                for candidate in candidates
            )
        )
        center = _point_tuple([wafer.get("center_patch_px")], 1)
        return cls(
            frame_id=frame_id,
            raw_state=str(raw_state),
            found=bool(wafer.get("found")),
            center_patch_px=center[0] if center else None,
            box_patch_px=_point_tuple(wafer.get("box_patch_px"), 4),
            yaw_deg=_finite_optional(wafer.get("yaw_relative_to_tray_deg")),
            flags=flags,
            boundary_evidence=str(
                wafer.get("boundary_evidence") or "unobservable"
            ),
            base_clearance_px=_finite_optional(
                wafer.get("base_projection_clearance_px")
            ),
            refined_clearance_px=_finite_optional(
                wafer.get("refined_projection_clearance_px")
            ),
            base_boundary_evidence=str(
                wafer.get("base_projection_boundary_evidence") or "unobservable"
            ),
            refined_boundary_evidence=str(
                wafer.get("refined_projection_boundary_evidence") or "unobservable"
            ),
            contour_depth_px=float(wafer.get("contour_outside_depth_px") or 0.0),
            contour_support_px=int(wafer.get("contour_outside_support_px") or 0),
            contour_area_ratio=float(
                wafer.get("contour_outside_area_ratio") or 0.0
            ),
            base_contour_depth_px=_finite_optional(
                wafer.get("base_contour_outside_depth_px")
            ),
            base_contour_support_px=(
                None
                if wafer.get("base_contour_outside_support_px") is None
                else int(wafer["base_contour_outside_support_px"])
            ),
            base_contour_area_ratio=_finite_optional(
                wafer.get("base_contour_outside_area_ratio")
            ),
            refined_contour_depth_px=_finite_optional(
                wafer.get("refined_contour_outside_depth_px")
            ),
            refined_contour_support_px=(
                None
                if wafer.get("refined_contour_outside_support_px") is None
                else int(wafer["refined_contour_outside_support_px"])
            ),
            refined_contour_area_ratio=_finite_optional(
                wafer.get("refined_contour_outside_area_ratio")
            ),
            accepted_secondary_candidate=accepted_secondary,
        )


@dataclass(frozen=True)
class TemporalProjectionSelection:
    source: str
    sample_count: int
    median_clearance_px: Optional[float]
    minimum_clearance_px: Optional[float]
    maximum_deviation_px: Optional[float]
    strong_outside_frame_count: int
    weak_contour_frame_count: int
    reliable: bool
    status: str

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "sample_count": self.sample_count,
            "median_clearance_px": self.median_clearance_px,
            "minimum_clearance_px": self.minimum_clearance_px,
            "maximum_deviation_px": self.maximum_deviation_px,
            "strong_outside_frame_count": self.strong_outside_frame_count,
            "weak_contour_frame_count": self.weak_contour_frame_count,
            "reliable": self.reliable,
            "status": self.status,
        }


@dataclass(frozen=True)
class TemporalInsideGroupResult:
    slot_key: str
    window_size: int
    valid_frame_count: int
    unique_frame_count: int
    found_frame_count: int
    raw_state_counts: dict[str, int]
    raw_strong_outside_frame_count: int
    center_jitter_px: Optional[float]
    yaw_jitter_deg: Optional[float]
    median_pairwise_iou: Optional[float]
    projection: TemporalProjectionSelection
    accepted_secondary_candidate_count: int
    hard_warning_frame_count: int
    candidate: bool
    locally_confirmed: bool
    status: str

    def to_json(self) -> dict[str, Any]:
        return {
            "slot_key": self.slot_key,
            "window_size": self.window_size,
            "valid_frame_count": self.valid_frame_count,
            "unique_frame_count": self.unique_frame_count,
            "found_frame_count": self.found_frame_count,
            "raw_state_counts": dict(self.raw_state_counts),
            "raw_strong_outside_frame_count": self.raw_strong_outside_frame_count,
            "center_jitter_px": self.center_jitter_px,
            "yaw_jitter_deg": self.yaw_jitter_deg,
            "median_pairwise_iou": self.median_pairwise_iou,
            "projection": self.projection.to_json(),
            "accepted_secondary_candidate_count": (
                self.accepted_secondary_candidate_count
            ),
            "hard_warning_frame_count": self.hard_warning_frame_count,
            "candidate": self.candidate,
            "locally_confirmed": self.locally_confirmed,
            "status": self.status,
        }


@dataclass(frozen=True)
class MultiViewInsideLatchResult:
    slot_key: str
    valid_group_count: int
    occupied_support_group_count: int
    occupied_support_frame_count: int
    strong_outside_group_count: int
    strong_outside_frame_count: int
    strong_outside_group_ratio: float
    strong_outside_frame_ratio: float
    authorized: bool
    status: str

    def to_json(self) -> dict[str, Any]:
        return {
            "slot_key": self.slot_key,
            "valid_group_count": self.valid_group_count,
            "occupied_support_group_count": self.occupied_support_group_count,
            "occupied_support_frame_count": self.occupied_support_frame_count,
            "strong_outside_group_count": self.strong_outside_group_count,
            "strong_outside_frame_count": self.strong_outside_frame_count,
            "strong_outside_group_ratio": self.strong_outside_group_ratio,
            "strong_outside_frame_ratio": self.strong_outside_frame_ratio,
            "authorized": self.authorized,
            "status": self.status,
        }


def _source_measurement(
    row: TemporalInsideFrameEvidence, source: str
) -> tuple[Optional[float], str, float, int, float]:
    if source == "refined":
        clearance = row.refined_clearance_px
        evidence = row.refined_boundary_evidence
        depth = row.refined_contour_depth_px
        support = row.refined_contour_support_px
        area = row.refined_contour_area_ratio
    else:
        clearance = row.base_clearance_px
        evidence = row.base_boundary_evidence
        depth = row.base_contour_depth_px
        support = row.base_contour_support_px
        area = row.base_contour_area_ratio

    # Schema-v5 reports predate per-projection contour diagnostics.  Preserve
    # their fail-closed aggregate evidence instead of guessing a clean source.
    if depth is None or support is None or area is None:
        depth = row.contour_depth_px
        support = row.contour_support_px
        area = row.contour_area_ratio
    if evidence == "unobservable":
        evidence = (
            "strong_outside"
            if row.boundary_evidence == "strong_outside"
            and clearance is not None
            and clearance <= -BOUNDARY_UNCERTAINTY_PX
            else "uncertain"
            if row.boundary_evidence == "uncertain"
            else row.boundary_evidence
        )
    return clearance, evidence, float(depth), int(support), float(area)


def _projection_selection(
    rows: Sequence[TemporalInsideFrameEvidence], config: WaferQualityConfig
) -> TemporalProjectionSelection:
    minimum_samples = max(4, int(config.temporal_inside_min_weak_contour_frames))
    maximum_deviation = float(config.temporal_inside_projection_max_deviation_px)
    selected_source = "unavailable"
    selected: list[tuple[TemporalInsideFrameEvidence, float]] = []
    refined = [
        (row, float(row.refined_clearance_px))
        for row in rows
        if row.refined_clearance_px is not None
    ]
    base = [
        (row, float(row.base_clearance_px))
        for row in rows
        if row.base_clearance_px is not None
    ]

    def stable_subset(
        values: Sequence[tuple[TemporalInsideFrameEvidence, float]]
    ) -> list[tuple[TemporalInsideFrameEvidence, float]]:
        if len(values) < minimum_samples:
            return []
        clearances = np.asarray([value for _row, value in values], dtype=np.float64)
        median = float(np.median(clearances))
        if float(np.max(np.abs(clearances - median))) <= maximum_deviation:
            return list(values)
        order = np.argsort(np.abs(clearances - median))
        # One exposure/projection outlier may be discarded, but a trimmed
        # source is never sufficient for local confirmation without the
        # independent multi-view latch.
        keep_count = min(len(values), int(config.temporal_inside_window_size) - 1)
        selected_indices = order[:keep_count]
        selected_values = clearances[selected_indices]
        selected_median = float(np.median(selected_values))
        if float(np.max(np.abs(selected_values - selected_median))) > maximum_deviation:
            return []
        return [values[int(index)] for index in selected_indices]

    refined_selected = stable_subset(refined)
    base_selected = stable_subset(base)
    if refined_selected:
        selected_source = (
            "refined_trimmed"
            if len(refined_selected) < len(refined)
            else "refined"
        )
        selected = refined_selected
    elif base_selected:
        base_name = "base_after_refined_unstable" if refined else "base_only"
        selected_source = (
            base_name + "_trimmed"
            if len(base_selected) < len(base)
            else base_name
        )
        selected = base_selected

    if not selected:
        return TemporalProjectionSelection(
            source="unavailable",
            sample_count=0,
            median_clearance_px=None,
            minimum_clearance_px=None,
            maximum_deviation_px=None,
            strong_outside_frame_count=0,
            weak_contour_frame_count=0,
            reliable=False,
            status="no_stable_projection_source",
        )

    logical_source = "refined" if selected_source.startswith("refined") else "base"
    clearances = np.asarray([value for _row, value in selected], dtype=np.float64)
    median = float(np.median(clearances))
    deviation = float(np.max(np.abs(clearances - median)))
    strong_count = 0
    weak_count = 0
    for row, _clearance in selected:
        _value, evidence, depth, support, area = _source_measurement(
            row, logical_source
        )
        strong_count += int(evidence == "strong_outside")
        reliable_contour = bool(
            depth >= BOUNDARY_CONTOUR_MIN_DEPTH_PX
            and support >= BOUNDARY_CONTOUR_MIN_SUPPORT_PX
            and area >= BOUNDARY_CONTOUR_MIN_AREA_RATIO
        )
        weak_count += int(not reliable_contour)
    return TemporalProjectionSelection(
        source=selected_source,
        sample_count=len(selected),
        median_clearance_px=median,
        minimum_clearance_px=float(np.min(clearances)),
        maximum_deviation_px=deviation,
        strong_outside_frame_count=strong_count,
        weak_contour_frame_count=weak_count,
        reliable=True,
        status="ok",
    )


def evaluate_temporal_inside_window(
    slot_key: str,
    evidence: Sequence[TemporalInsideFrameEvidence],
    config: WaferQualityConfig,
) -> TemporalInsideGroupResult:
    rows = list(evidence)
    window_size = int(config.temporal_inside_window_size)
    state_counts = Counter(row.raw_state for row in rows)
    raw_strong = sum(row.boundary_evidence == "strong_outside" for row in rows)
    found_rows = [
        row
        for row in rows
        if row.found
        and row.center_patch_px is not None
        and len(row.box_patch_px) == 4
        and row.yaw_deg is not None
    ]
    center_jitter: Optional[float] = None
    yaw_jitter: Optional[float] = None
    median_iou: Optional[float] = None
    if found_rows:
        centers = np.asarray(
            [row.center_patch_px for row in found_rows], dtype=np.float64
        ).reshape(-1, 2)
        median_center = np.median(centers, axis=0)
        center_jitter = float(
            np.max(np.linalg.norm(centers - median_center.reshape(1, 2), axis=1))
        )
        yaws = np.asarray([row.yaw_deg for row in found_rows], dtype=np.float64)
        yaw_jitter = float(np.max(np.abs(yaws - np.median(yaws))))
    if len(found_rows) >= 2:
        ious = [
            _convex_iou(found_rows[first].box_patch_px, found_rows[second].box_patch_px)
            for first in range(len(found_rows))
            for second in range(first + 1, len(found_rows))
        ]
        median_iou = float(np.median(np.asarray(ious, dtype=np.float64)))

    projection = _projection_selection(rows, config)
    accepted_secondary = sum(row.accepted_secondary_candidate for row in rows)
    hard_warning_count = sum(bool(set(row.flags) & _HARD_WARNING_FLAGS) for row in rows)
    status = "candidate"
    candidate = False
    if len(rows) != window_size:
        status = "window_incomplete"
    elif len({row.frame_id for row in rows}) != window_size:
        status = "duplicate_frame"
    elif len(found_rows) != window_size:
        status = "wafer_geometry_incomplete"
    elif accepted_secondary:
        status = "secondary_wafer_candidate_present"
    elif hard_warning_count:
        status = "hard_warning_present"
    elif center_jitter is None or center_jitter > float(
        config.temporal_inside_max_center_jitter_px
    ):
        status = "wafer_center_unstable"
    elif yaw_jitter is None or yaw_jitter > float(
        config.temporal_inside_max_yaw_jitter_deg
    ):
        status = "wafer_yaw_unstable"
    elif median_iou is None or median_iou < float(
        config.temporal_inside_min_pairwise_iou
    ):
        status = "wafer_box_unstable"
    elif not projection.reliable:
        status = projection.status
    elif projection.strong_outside_frame_count:
        status = "selected_projection_has_strong_outside"
    elif projection.weak_contour_frame_count < int(
        config.temporal_inside_min_weak_contour_frames
    ):
        status = "insufficient_weak_contour_frames"
    elif (
        projection.median_clearance_px is None
        or projection.median_clearance_px
        <= float(config.temporal_inside_min_projection_clearance_px)
    ):
        status = "selected_projection_not_inside"
    else:
        candidate = True

    locally_confirmed = bool(candidate)
    if candidate and (
        projection.source.startswith("base_only")
        or projection.source.endswith("_trimmed")
    ):
        locally_confirmed = bool(
            projection.minimum_clearance_px is not None
            and projection.minimum_clearance_px
            >= float(config.temporal_inside_base_only_min_clearance_px)
        )
        if not locally_confirmed:
            status = "base_only_requires_multiview_latch"
    elif candidate:
        status = "locally_confirmed"

    return TemporalInsideGroupResult(
        slot_key=str(slot_key),
        window_size=window_size,
        valid_frame_count=len(rows),
        unique_frame_count=len({row.frame_id for row in rows}),
        found_frame_count=len(found_rows),
        raw_state_counts=dict(sorted(state_counts.items())),
        raw_strong_outside_frame_count=int(raw_strong),
        center_jitter_px=center_jitter,
        yaw_jitter_deg=yaw_jitter,
        median_pairwise_iou=median_iou,
        projection=projection,
        accepted_secondary_candidate_count=accepted_secondary,
        hard_warning_frame_count=hard_warning_count,
        candidate=candidate,
        locally_confirmed=locally_confirmed,
        status=status,
    )


def evaluate_multiview_inside_latch(
    slot_key: str,
    groups: Sequence[TemporalInsideGroupResult],
    config: WaferQualityConfig,
) -> MultiViewInsideLatchResult:
    rows = [group for group in groups if group.valid_frame_count == group.window_size]
    valid_group_count = len(rows)
    occupied_groups = sum(
        int(group.raw_state_counts.get("occupied", 0)) > 0 for group in rows
    )
    occupied_frames = sum(
        int(group.raw_state_counts.get("occupied", 0)) for group in rows
    )
    strong_groups = sum(
        group.raw_strong_outside_frame_count >= 2 for group in rows
    )
    strong_frames = sum(group.raw_strong_outside_frame_count for group in rows)
    group_ratio = float(strong_groups / max(valid_group_count, 1))
    frame_total = sum(group.valid_frame_count for group in rows)
    frame_ratio = float(strong_frames / max(frame_total, 1))

    status = "authorized"
    authorized = True
    if valid_group_count < int(config.multiview_inside_min_groups):
        status = "insufficient_view_groups"
        authorized = False
    elif occupied_groups < int(config.multiview_inside_min_occupied_groups):
        status = "insufficient_occupied_support_groups"
        authorized = False
    elif occupied_frames < int(config.multiview_inside_min_occupied_frames):
        status = "insufficient_occupied_support_frames"
        authorized = False
    elif group_ratio > float(
        config.multiview_inside_max_strong_outside_group_ratio
    ):
        status = "strong_outside_group_ratio_too_high"
        authorized = False
    elif frame_ratio > float(
        config.multiview_inside_max_strong_outside_frame_ratio
    ):
        status = "strong_outside_frame_ratio_too_high"
        authorized = False

    return MultiViewInsideLatchResult(
        slot_key=str(slot_key),
        valid_group_count=valid_group_count,
        occupied_support_group_count=occupied_groups,
        occupied_support_frame_count=occupied_frames,
        strong_outside_group_count=strong_groups,
        strong_outside_frame_count=strong_frames,
        strong_outside_group_ratio=group_ratio,
        strong_outside_frame_ratio=frame_ratio,
        authorized=authorized,
        status=status,
    )


__all__ = [
    "MultiViewInsideLatchResult",
    "TemporalInsideFrameEvidence",
    "TemporalInsideGroupResult",
    "TemporalProjectionSelection",
    "evaluate_multiview_inside_latch",
    "evaluate_temporal_inside_window",
]
