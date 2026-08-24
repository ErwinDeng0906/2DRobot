"""Read-only five-frame confirmation for L-shaped stacking evidence."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Hashable, Mapping, Optional, Sequence

import cv2
import numpy as np

from .tray_occupancy import SlotState
from .wafer_shape_quality import WaferQualityConfig


@dataclass(frozen=True)
class LShapeFrameEvidence:
    frame_id: Hashable
    candidate_box_patch_px: tuple[tuple[float, float], ...] = ()
    relative_center_offset_px: Optional[tuple[float, float]] = None

    @property
    def candidate_present(self) -> bool:
        return bool(
            len(self.candidate_box_patch_px) == 4
            and self.relative_center_offset_px is not None
        )


@dataclass(frozen=True)
class StackingTemporalResult:
    slot_key: str
    window_size: int
    valid_frame_count: int
    l_shape_support_count: int
    median_relative_center_offset_px: Optional[tuple[float, float]]
    max_relative_center_jitter_px: Optional[float]
    median_pairwise_iou: Optional[float]
    confirmed: bool
    status: str

    def to_json(self) -> dict[str, Any]:
        return {
            "slot_key": self.slot_key,
            "window_size": self.window_size,
            "valid_frame_count": self.valid_frame_count,
            "l_shape_support_count": self.l_shape_support_count,
            "median_relative_center_offset_px": (
                None
                if self.median_relative_center_offset_px is None
                else list(self.median_relative_center_offset_px)
            ),
            "max_relative_center_jitter_px": self.max_relative_center_jitter_px,
            "median_pairwise_iou": self.median_pairwise_iou,
            "confirmed": self.confirmed,
            "status": self.status,
        }


def _convex_iou(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
) -> float:
    first_polygon = np.asarray(first, dtype=np.float32).reshape(4, 2)
    second_polygon = np.asarray(second, dtype=np.float32).reshape(4, 2)
    first_area = abs(float(cv2.contourArea(first_polygon)))
    second_area = abs(float(cv2.contourArea(second_polygon)))
    intersection_area, _intersection = cv2.intersectConvexConvex(
        cv2.convexHull(first_polygon),
        cv2.convexHull(second_polygon),
    )
    union = first_area + second_area - float(intersection_area)
    return float(intersection_area / max(union, 1.0))


def evaluate_l_shape_window(
    slot_key: str,
    evidence: Sequence[LShapeFrameEvidence],
    config: WaferQualityConfig,
) -> StackingTemporalResult:
    """Evaluate exactly one window without using absolute image coordinates."""

    window_size = int(config.stacked_l_temporal_window_size)
    rows = list(evidence)
    candidates = [row for row in rows if row.candidate_present]
    median_offset: Optional[tuple[float, float]] = None
    maximum_jitter: Optional[float] = None
    median_iou: Optional[float] = None
    if candidates:
        offsets = np.asarray(
            [row.relative_center_offset_px for row in candidates],
            dtype=np.float64,
        ).reshape(-1, 2)
        median = np.median(offsets, axis=0)
        median_offset = (float(median[0]), float(median[1]))
        maximum_jitter = float(
            np.max(np.linalg.norm(offsets - median.reshape(1, 2), axis=1))
        )
    if len(candidates) >= 2:
        pairwise_ious = [
            _convex_iou(
                candidates[first].candidate_box_patch_px,
                candidates[second].candidate_box_patch_px,
            )
            for first in range(len(candidates))
            for second in range(first + 1, len(candidates))
        ]
        median_iou = float(np.median(np.asarray(pairwise_ious, dtype=np.float64)))

    status = "window_pending"
    confirmed = False
    if len(rows) == window_size:
        if len({row.frame_id for row in rows}) != window_size:
            status = "duplicate_frame"
        elif len(candidates) < int(config.stacked_l_temporal_min_support):
            status = "insufficient_support"
        elif (
            maximum_jitter is None
            or maximum_jitter
            > float(config.stacked_l_temporal_max_relative_center_jitter_px)
        ):
            status = "relative_center_unstable"
        elif (
            median_iou is None
            or median_iou < float(config.stacked_l_temporal_min_pairwise_iou)
        ):
            status = "candidate_box_unstable"
        else:
            status = "confirmed"
            confirmed = True

    return StackingTemporalResult(
        slot_key=str(slot_key),
        window_size=window_size,
        valid_frame_count=len(rows),
        l_shape_support_count=len(candidates),
        median_relative_center_offset_px=median_offset,
        max_relative_center_jitter_px=maximum_jitter,
        median_pairwise_iou=median_iou,
        confirmed=confirmed,
        status=status,
    )


def _accepted_l_shape_evidence(
    frame_id: Hashable,
    wafer: Any,
) -> LShapeFrameEvidence:
    for candidate in getattr(wafer, "secondary_candidates", ()):
        if candidate.source == "l_shape" and candidate.accepted:
            return LShapeFrameEvidence(
                frame_id=frame_id,
                candidate_box_patch_px=candidate.box_patch_px,
                relative_center_offset_px=candidate.relative_center_offset_px,
            )
    return LShapeFrameEvidence(frame_id=frame_id)


class FiveFrameLShapeStackingTracker:
    """Maintain one fail-closed five-fresh-frame window per tray slot."""

    _INVALID_STATES = frozenset(
        {SlotState.OUT_OF_VIEW, SlotState.OCCLUDED, SlotState.UNKNOWN}
    )

    def __init__(self, config: WaferQualityConfig) -> None:
        self.config = config
        self._history: dict[str, deque[LShapeFrameEvidence]] = {}
        self._last_frame_id: Optional[Hashable] = None

    def reset(self) -> None:
        self._history.clear()
        self._last_frame_id = None

    def _ordered_frame_regressed(self, frame_id: Hashable) -> bool:
        previous = self._last_frame_id
        return bool(
            isinstance(previous, (int, float))
            and isinstance(frame_id, (int, float))
            and frame_id < previous
        )

    def update(self, result: Any, *, frame_id: Hashable) -> Any:
        if frame_id == self._last_frame_id:
            return result
        if self._ordered_frame_regressed(frame_id):
            self.reset()
        self._last_frame_id = frame_id

        temporal: dict[str, StackingTemporalResult] = {}
        valid_analysis = bool(
            getattr(
                result,
                "analysis_quality_passed",
                getattr(result, "quality_passed", False),
            )
        )
        for analysis in result.slots:
            slot_key = str(analysis.projection.slot_key)
            history = self._history.setdefault(
                slot_key,
                deque(maxlen=int(self.config.stacked_l_temporal_window_size)),
            )
            if not valid_analysis or analysis.decision.state in self._INVALID_STATES:
                history.clear()
            else:
                history.append(_accepted_l_shape_evidence(frame_id, analysis.wafer))
            temporal[slot_key] = evaluate_l_shape_window(
                slot_key,
                tuple(history),
                self.config,
            )

        updated_slots = []
        canvas = result.annotated_image.copy()
        for analysis in result.slots:
            slot_key = str(analysis.projection.slot_key)
            window = temporal[slot_key]
            decision = analysis.decision
            direct_quadrilateral = "second_quadrilateral" in analysis.wafer.flags
            if (
                window.confirmed
                and analysis.wafer.found
                and not direct_quadrilateral
                and decision.state not in self._INVALID_STATES
            ):
                outside = bool(
                    analysis.wafer.outside_slot
                    or decision.state == SlotState.OUTSIDE_SLOT
                )
                state = (
                    SlotState.STACKED_OUTSIDE_SLOT
                    if outside
                    else SlotState.STACKED
                )
                decision = replace(
                    decision,
                    state=state,
                    occupied=True,
                    safe_to_use_as_empty=False,
                    reason=(
                        "L-shaped second-wafer geometry is stable in three of "
                        "five fresh frames"
                    ),
                    flags=tuple(decision.flags)
                    + ("l_shape_five_frame_confirmed",),
                )
            updated = replace(analysis, decision=decision)
            updated_slots.append(updated)

            if window.l_shape_support_count > 0 and not direct_quadrilateral:
                if (
                    "l_shape_stacking_candidate" in analysis.wafer.flags
                    and analysis.wafer_secondary_boxes_image_px
                ):
                    box = np.asarray(
                        analysis.wafer_secondary_boxes_image_px[0],
                        dtype=np.int32,
                    ).reshape(4, 2)
                    label_origin = (int(box[0][0]), int(box[0][1]) + 16)
                else:
                    center = analysis.projection.center_px
                    label_origin = (int(center[0]), int(center[1]))
                label = (
                    f"L OK {window.l_shape_support_count}/5"
                    if window.confirmed
                    else f"L? {window.l_shape_support_count}/5"
                )
                color = (0, 0, 255) if window.confirmed else (255, 255, 0)
                cv2.putText(
                    canvas,
                    label,
                    label_origin,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        summary = {state.value: 0 for state in SlotState}
        for analysis in updated_slots:
            summary[analysis.decision.state.value] += 1
        summary["analyzed"] = len(updated_slots)
        return replace(
            result,
            slots=tuple(updated_slots),
            summary=summary,
            annotated_image=canvas,
            stacking_temporal=temporal,
        )


def temporal_results_to_json(
    values: Mapping[str, StackingTemporalResult],
) -> dict[str, dict[str, Any]]:
    return {
        str(slot_key): result.to_json()
        for slot_key, result in sorted(values.items())
    }


__all__ = [
    "FiveFrameLShapeStackingTracker",
    "LShapeFrameEvidence",
    "StackingTemporalResult",
    "evaluate_l_shape_window",
    "temporal_results_to_json",
]
