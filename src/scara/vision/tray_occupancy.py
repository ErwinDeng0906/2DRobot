"""Fail-closed slot-state decisions from independent visual evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .slot_marker_observation import SlotMarkerEvidence, SlotProjection
from .wafer_shape_quality import WaferObservation


class SlotState(str, Enum):
    EMPTY = "empty"
    EMPTY_UNREAD_MARKER = "empty_unread_marker"
    OCCUPIED = "occupied"
    WARNING = "warning"
    STACKED = "stacked"
    OUTSIDE_SLOT = "outside_slot"
    STACKED_OUTSIDE_SLOT = "stacked_outside_slot"
    OUT_OF_VIEW = "out_of_view"
    OCCLUDED = "occluded"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SlotDecisionConfig:
    minimum_image_coverage_ratio: float = 0.90
    explicit_occlusion_ratio: float = 0.25


DEFAULT_SLOT_DECISION = SlotDecisionConfig()


_DIRECT_STACKING_FLAGS = frozenset({"stacked_geometry_confirmed"})


def _refined_abnormal_state(wafer: WaferObservation) -> tuple[SlotState, str]:
    """Split the former generic abnormal state into actionable categories.

    A boundary-crossing contour is direct out-of-slot evidence.  Internal
    lines, multiple color components, and other severe
    shape anomalies are not stacking unless L-corner or second-quadrilateral
    geometry has confirmed a second wafer.
    """
    flags = set(wafer.flags)
    outside_slot = bool(wafer.outside_slot or "outside_slot" in flags)
    direct_stacking = bool(flags & _DIRECT_STACKING_FLAGS)
    if outside_slot and direct_stacking:
        return (
            SlotState.STACKED_OUTSIDE_SLOT,
            "stacking evidence is present and the wafer footprint crosses the slot boundary",
        )
    if outside_slot:
        return SlotState.OUTSIDE_SLOT, "the wafer footprint crosses the slot boundary"
    if direct_stacking:
        return SlotState.STACKED, "second-wafer geometry is confirmed inside the slot"
    return SlotState.WARNING, "shape evidence is abnormal but stacking geometry is not confirmed"


@dataclass(frozen=True)
class SlotDecision:
    slot_key: str
    state: SlotState
    occupied: bool
    safe_to_use_as_empty: bool
    reason: str
    flags: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "slot_key": self.slot_key,
            "state": self.state.value,
            "occupied": self.occupied,
            "safe_to_use_as_empty": self.safe_to_use_as_empty,
            "reason": self.reason,
            "flags": list(self.flags),
        }


def decide_slot_state(
    projection: SlotProjection,
    marker: SlotMarkerEvidence,
    wafer: WaferObservation,
    *,
    occlusion_ratio: float = 0.0,
    config: SlotDecisionConfig = DEFAULT_SLOT_DECISION,
) -> SlotDecision:
    """Fuse evidence without turning absent evidence into a confident state.

    Decision precedence is deliberate: incomplete view and explicit occlusion
    make the slot unknowable; a wafer candidate then establishes occupancy;
    a decoded or marker-like pattern establishes emptiness.  If none of those
    apply, the state remains UNKNOWN instead of being labelled missing/empty.
    """
    if projection.image_coverage_ratio < config.minimum_image_coverage_ratio:
        return SlotDecision(
            projection.slot_key,
            SlotState.OUT_OF_VIEW,
            occupied=False,
            safe_to_use_as_empty=False,
            reason="slot footprint is not fully inside the image",
            flags=("partial_view",),
        )
    if float(occlusion_ratio) >= config.explicit_occlusion_ratio:
        return SlotDecision(
            projection.slot_key,
            SlotState.OCCLUDED,
            occupied=False,
            safe_to_use_as_empty=False,
            reason="an explicit occlusion mask covers the slot",
            flags=("explicit_occlusion",),
        )
    if wafer.found:
        flags = list(wafer.flags)
        if marker.decoded:
            flags.append("contradictory_marker_and_wafer")
        if marker.decoded or wafer.quality == "abnormal" or wafer.outside_slot:
            state, reason = _refined_abnormal_state(wafer)
            if marker.decoded:
                reason += "; wafer and decoded empty-slot marker are both visible"
            return SlotDecision(
                projection.slot_key,
                state,
                occupied=True,
                safe_to_use_as_empty=False,
                reason=reason,
                flags=tuple(flags),
            )
        state = {
            "normal": SlotState.OCCUPIED,
            "warning": SlotState.WARNING,
        }.get(wafer.quality)
        if state is None:
            state, reason = _refined_abnormal_state(wafer)
            return SlotDecision(
                projection.slot_key,
                state,
                occupied=True,
                safe_to_use_as_empty=False,
                reason=reason,
                flags=tuple(flags),
            )
        return SlotDecision(
            projection.slot_key,
            state,
            occupied=True,
            safe_to_use_as_empty=False,
            reason=f"wafer candidate quality is {wafer.quality}",
            flags=tuple(flags),
        )
    if marker.decoded:
        return SlotDecision(
            projection.slot_key,
            SlotState.EMPTY,
            occupied=False,
            safe_to_use_as_empty=True,
            reason="the configured slot marker ID is decoded",
            flags=(),
        )
    if marker.marker_like_visible:
        return SlotDecision(
            projection.slot_key,
            SlotState.EMPTY_UNREAD_MARKER,
            occupied=False,
            safe_to_use_as_empty=True,
            reason="a black/white marker pattern is visible but its ID is unread",
            flags=("marker_id_unread",),
        )
    return SlotDecision(
        projection.slot_key,
        SlotState.UNKNOWN,
        occupied=False,
        safe_to_use_as_empty=False,
        reason="no decoded marker and no reliable wafer candidate",
        flags=("insufficient_evidence",),
    )


__all__ = [
    "DEFAULT_SLOT_DECISION",
    "SlotDecision",
    "SlotDecisionConfig",
    "SlotState",
    "decide_slot_state",
]
