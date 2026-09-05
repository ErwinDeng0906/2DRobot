"""Controller-free state and evidence model for one wafer transfer.

This module connects overview tray occupancy, runtime ``W<-T`` registration,
fresh robot state, and a future markerless close-range camera observation.  It
does not import Qt and cannot issue a robot command.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Optional

import numpy as np

from .close_range_slot_observation import (
    CloseRangeOperation,
    CloseRangeSlotObservation,
)
from .tray_occupancy import SlotState
from .tray_vision_fusion import TrayVisionResult


class TransferPhase(str, Enum):
    IDLE = "idle"
    SOURCE_SELECTED = "source_selected"
    SOURCE_READY = "source_ready"
    ROUTE_READY = "route_ready"
    TRACKING_PICK = "tracking_pick"
    WAITING_PICK_ALIGNMENT = "waiting_pick_alignment"
    VERIFYING_PICK = "verifying_pick"
    PICKED = "picked"
    TRACKING_PLACE = "tracking_place"
    WAITING_PLACE_ALIGNMENT = "waiting_place_alignment"
    READY_TO_PLACE = "ready_to_place"
    VERIFYING_PLACE = "verifying_place"
    COMPLETE = "complete"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class WaferTransferConfig:
    maximum_robot_state_age_s: float = 1.0
    maximum_frame_robot_skew_s: float = 0.35
    maximum_close_range_age_s: float = 0.75
    maximum_close_range_robot_skew_s: float = 0.35
    maximum_close_range_j3_disagreement_mm: float = 0.15
    click_maximum_slot_distance_mm: float = 12.5
    coarse_arrival_distance_mm: float = 2.0
    source_consensus_window_frames: int = 2
    source_consensus_minimum_occupied_frames: int = 2
    source_consensus_maximum_dropout_frames: int = 2
    maximum_source_evidence_age_s: float = 2.0
    pick_requires_close_range_alignment: bool = True
    place_requires_close_range_insertability: bool = True


DEFAULT_WAFER_TRANSFER_CONFIG = WaferTransferConfig()


@dataclass(frozen=True)
class RobotStateSnapshot:
    captured_monotonic_s: float
    joints: tuple[float, float, float, float]
    pose: tuple[float, float, float, float, float, float]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RobotStateSnapshot":
        captured = float(value.get("captured_monotonic_s", math.nan))
        joints = np.asarray(value.get("joints"), dtype=np.float64).reshape(-1)
        pose = np.asarray(value.get("pose"), dtype=np.float64).reshape(-1)
        if (
            not math.isfinite(captured)
            or joints.size != 4
            or pose.size != 6
            or not np.all(np.isfinite(joints))
            or not np.all(np.isfinite(pose))
        ):
            raise ValueError(
                "robot state must contain finite timestamp, four joints and six pose values"
            )
        return cls(
            captured,
            tuple(float(item) for item in joints),
            tuple(float(item) for item in pose),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "captured_monotonic_s": self.captured_monotonic_s,
            "joints": list(self.joints),
            "pose": list(self.pose),
        }


def _slot_states(result: Optional[TrayVisionResult]) -> dict[str, dict[str, Any]]:
    if result is None or not result.quality_passed:
        return {}
    return {
        item.projection.slot_key: {
            "state": item.decision.state.value,
            "occupied": bool(item.decision.occupied),
            "safe_to_use_as_empty": bool(item.decision.safe_to_use_as_empty),
            "reason": item.decision.reason,
            "flags": list(item.decision.flags),
            "wafer_quality": item.wafer.quality,
            "wafer_angle_deg": item.wafer.yaw_relative_to_tray_deg,
            "center_T_mm": list(item.projection.center_T_mm),
            "center_px": list(item.projection.center_px),
        }
        for item in result.slots
    }


def _validated_transform_W_T(
    registration: Optional[Mapping[str, Any]]
) -> Optional[np.ndarray]:
    if not isinstance(registration, Mapping) or registration.get("status") != "success":
        return None
    transform = np.asarray(registration.get("transform_W_T"), dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        return None
    return transform


def _finite_float_or_none(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


class WaferTransferSession:
    """Track a selected source/destination while the robot state changes."""

    def __init__(
        self,
        geometry: Mapping[str, Any],
        config: WaferTransferConfig = DEFAULT_WAFER_TRANSFER_CONFIG,
    ) -> None:
        self.geometry = dict(geometry)
        self.config = config
        self.slot_names = frozenset(str(key) for key in self.geometry.get("slots", {}))
        if len(self.slot_names) != 36:
            raise ValueError("wafer transfer requires exactly 36 metric tray slots")
        if config.source_consensus_window_frames < 1:
            raise ValueError("source consensus window must contain at least one frame")
        if not (
            1
            <= config.source_consensus_minimum_occupied_frames
            <= config.source_consensus_window_frames
        ):
            raise ValueError(
                "source consensus minimum must be within the configured frame window"
            )
        if config.source_consensus_maximum_dropout_frames < 0:
            raise ValueError("source consensus dropout allowance cannot be negative")
        if not math.isfinite(config.maximum_source_evidence_age_s) or config.maximum_source_evidence_age_s <= 0:
            raise ValueError("source evidence age must be positive and finite")
        self.phase = TransferPhase.IDLE
        self.source_slot: Optional[str] = None
        self.destination_slot: Optional[str] = None
        self.latest_result: Optional[TrayVisionResult] = None
        self.latest_frame_sequence: Optional[int] = None
        self.latest_frame_captured_monotonic_s: Optional[float] = None
        self.latest_robot_state: Optional[RobotStateSnapshot] = None
        self.registration: Optional[dict[str, Any]] = None
        self.close_range: Optional[CloseRangeSlotObservation] = None
        self.block_reason = ""
        self.close_range_gates: dict[str, dict[str, Any]] = {}
        self.events: deque[dict[str, Any]] = deque(maxlen=500)
        self.overview_diagnostics: deque[dict[str, Any]] = deque(maxlen=180)
        self._slot_state_history = {
            slot: deque(maxlen=config.source_consensus_window_frames)
            for slot in self.slot_names
        }
        self._slot_dropout_streak = {slot: 0 for slot in self.slot_names}
        self._slot_evidence_times = {
            slot: deque(maxlen=config.source_consensus_window_frames)
            for slot in self.slot_names
        }
        self._not_evaluated_frames = 0
        # Two normal occupied frames establish the operator-selected source.
        # Classification uncertainty after that proof must not make the UI or
        # XY loop flap.  The qualification is cleared on target/reset/stream
        # changes, three consecutive explicit-empty decisions, or a hard
        # stacked/outside contradiction.
        self._qualified_source_slot: Optional[str] = None
        self._source_qualification_frame_sequence: Optional[int] = None
        self._source_qualification_captured_monotonic_s: Optional[float] = None
        self._source_explicit_empty_streak = 0

    def _event(self, name: str, **details: Any) -> None:
        self.events.append(
            {
                "at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "event": str(name),
                "phase": self.phase.value,
                "details": details,
            }
        )

    def reset(self) -> None:
        self.phase = TransferPhase.IDLE
        self.source_slot = None
        self.destination_slot = None
        self.close_range = None
        self.close_range_gates = {}
        self.block_reason = ""
        self._qualified_source_slot = None
        self._source_qualification_frame_sequence = None
        self._source_qualification_captured_monotonic_s = None
        self._source_explicit_empty_streak = 0
        self._event("reset")

    def block(self, reason: str) -> None:
        """Fail closed without discarding the evidence collected so far."""

        self.phase = TransferPhase.BLOCKED
        self.block_reason = str(reason)
        self._event("session_blocked", reason=self.block_reason)

    def target_lock_active(self) -> bool:
        """Return whether camera-1 acquisition has become an immutable target."""

        return self.phase in {
            TransferPhase.TRACKING_PICK,
            TransferPhase.WAITING_PICK_ALIGNMENT,
            TransferPhase.VERIFYING_PICK,
            TransferPhase.PICKED,
            TransferPhase.TRACKING_PLACE,
            TransferPhase.WAITING_PLACE_ALIGNMENT,
            TransferPhase.READY_TO_PLACE,
            TransferPhase.VERIFYING_PLACE,
            TransferPhase.COMPLETE,
        }

    def set_registration(self, registration: Optional[Mapping[str, Any]]) -> None:
        transform = _validated_transform_W_T(registration)
        if transform is None:
            return
        self.registration = dict(registration)
        self._event(
            "runtime_registration_updated",
            origin_world_xy_mm=self.registration.get("origin_world_xy_mm"),
            yaw_world_from_tray_deg=self.registration.get("yaw_world_from_tray_deg"),
        )
        self._refresh_ready_phase()

    def clear_registration(self) -> None:
        self.registration = None
        self._event("runtime_registration_cleared")
        self._refresh_ready_phase()

    def clear_overview_history(self, reason: str) -> None:
        for slot, history in self._slot_state_history.items():
            history.clear()
            self._slot_evidence_times[slot].clear()
            self._slot_dropout_streak[slot] = 0
        self._not_evaluated_frames = 0
        self._qualified_source_slot = None
        self._source_qualification_frame_sequence = None
        self._source_qualification_captured_monotonic_s = None
        self._source_explicit_empty_streak = 0
        self._event("overview_history_cleared", reason=str(reason))
        self._refresh_ready_phase()

    def invalidate_overview(
        self, reason: str, *, preserve_active_target_lock: bool = False
    ) -> None:
        """Discard stale camera/robot evidence and block any active lock."""

        self.latest_result = None
        self.latest_frame_sequence = None
        self.latest_frame_captured_monotonic_s = None
        self.latest_robot_state = None
        for slot, history in self._slot_state_history.items():
            history.clear()
            self._slot_evidence_times[slot].clear()
            self._slot_dropout_streak[slot] = 0
        self._not_evaluated_frames = 0
        self._qualified_source_slot = None
        self._source_qualification_frame_sequence = None
        self._source_qualification_captured_monotonic_s = None
        self._source_explicit_empty_streak = 0
        self.close_range = None
        self.close_range_gates = {}
        active = self.target_lock_active()
        self._event("overview_invalidated", reason=str(reason))
        if active and preserve_active_target_lock:
            self._event(
                "overview_monitor_unavailable_target_lock_retained",
                reason=str(reason),
            )
        elif active:
            self.block(f"camera1 overview invalidated: {reason}")
        else:
            self._refresh_ready_phase()

    def update_overview(
        self,
        result: TrayVisionResult,
        *,
        frame_sequence: int,
        frame_captured_monotonic_s: float,
        robot_state: Optional[Mapping[str, Any]],
    ) -> None:
        captured = float(frame_captured_monotonic_s)
        if (
            not math.isfinite(captured)
            or (self.latest_frame_sequence is not None and int(frame_sequence) <= self.latest_frame_sequence)
            or (self.latest_frame_captured_monotonic_s is not None and captured < self.latest_frame_captured_monotonic_s)
        ):
            # Reject the entire duplicate/old snapshot, including robot state.
            # Distinct sequence IDs may share a coarse Windows clock tick.
            return
        is_new_frame = True
        self.latest_result = result
        self.latest_frame_sequence = int(frame_sequence)
        self.latest_frame_captured_monotonic_s = float(frame_captured_monotonic_s)
        self.latest_robot_state = None
        if robot_state is not None:
            try:
                state = RobotStateSnapshot.from_mapping(robot_state)
                now = time.monotonic()
                if (
                    -0.05
                    <= now - state.captured_monotonic_s
                    <= self.config.maximum_robot_state_age_s
                    and abs(
                        state.captured_monotonic_s - float(frame_captured_monotonic_s)
                    )
                    <= self.config.maximum_frame_robot_skew_s
                ):
                    self.latest_robot_state = state
            except (TypeError, ValueError, OverflowError):
                self.latest_robot_state = None
        if is_new_frame:
            evaluated = bool(result.quality_passed and result.coordinate_mapping_allowed)
            states = _slot_states(result) if evaluated else {}
            self._not_evaluated_frames = 0 if evaluated else self._not_evaluated_frames + 1
            uncertain = {
                SlotState.UNKNOWN.value,
                SlotState.OUT_OF_VIEW.value,
                SlotState.OCCLUDED.value,
            }
            for slot, history in self._slot_state_history.items():
                if not evaluated:
                    # A pose failure says nothing about wafer presence. Keep
                    # identity/history, but the current metric gate stays shut.
                    # source_consensus separately expires old evidence by time.
                    continue
                state = states.get(slot)
                if state is None or state["state"] in uncertain:
                    self._slot_dropout_streak[slot] += 1
                    if (
                        self._slot_dropout_streak[slot]
                        > self.config.source_consensus_maximum_dropout_frames
                    ):
                        history.clear()
                        self._slot_evidence_times[slot].clear()
                    continue
                self._slot_dropout_streak[slot] = 0
                history.append(str(state["state"]))
                self._slot_evidence_times[slot].append(captured)
            if evaluated and self.source_slot is not None:
                selected = states.get(self.source_slot)
                selected_state = None if selected is None else selected["state"]
                explicit_empty = {
                    SlotState.EMPTY.value,
                    SlotState.EMPTY_UNREAD_MARKER.value,
                }
                hard_contradiction = {
                    SlotState.STACKED.value,
                    SlotState.OUTSIDE_SLOT.value,
                    SlotState.STACKED_OUTSIDE_SLOT.value,
                }
                if selected_state in explicit_empty:
                    self._source_explicit_empty_streak += 1
                elif selected_state is not None:
                    self._source_explicit_empty_streak = 0
                if (
                    selected_state in hard_contradiction
                    or self._source_explicit_empty_streak >= 3
                ):
                    if self._qualified_source_slot == self.source_slot:
                        self._event(
                            "source_qualification_revoked",
                            slot=self.source_slot,
                            state=selected_state,
                            explicit_empty_streak=self._source_explicit_empty_streak,
                        )
                    self._qualified_source_slot = None
                    self._source_qualification_frame_sequence = None
                    self._source_qualification_captured_monotonic_s = None
                self._maybe_latch_source_qualification()
        self._refresh_ready_phase()
        self._advance_tracking_phase()
        self.overview_diagnostics.append({
            "frame_id": self.latest_frame_sequence,
            "captured_monotonic_s": captured,
            "metric_passed": bool(result.quality_passed and result.coordinate_mapping_allowed),
            "reason": result.failure_reason,
            "projection_source": result.projection_source,
            "crop_anchor_held": result.projection_diagnostics.get("crop_anchor_held"),
            "used_marker_ids": list(result.pose.used_marker_ids),
            "pose_rms_px": result.pose.reprojection_rms_px,
            "per_marker_rms_px": dict(result.pose.per_marker_rms_px),
            "ransac_inlier_corners": result.pose.ransac_inlier_corner_count,
            "source_slot": self.source_slot,
            "source_state": states.get(self.source_slot or ""),
            "robot_state_available": self.latest_robot_state is not None,
        })

    def source_consensus(self, slot_name: Optional[str] = None) -> dict[str, Any]:
        """Return the recent independent evidence used to accept a source."""
        slot = self.source_slot if slot_name is None else str(slot_name)
        now = time.monotonic()
        history = [
            state for state, captured in zip(
                self._slot_state_history.get(slot or "", ()),
                self._slot_evidence_times.get(slot or "", ()),
            )
            if -0.05 <= now - captured <= self.config.maximum_source_evidence_age_s
        ]
        occupied_count = sum(
            state == SlotState.OCCUPIED.value for state in history
        )
        latest_occupied = bool(
            history and history[-1] == SlotState.OCCUPIED.value
        )
        dropout_count = int(self._slot_dropout_streak.get(slot or "", 0))
        evidence_passed = bool(
            latest_occupied
            and occupied_count
            >= self.config.source_consensus_minimum_occupied_frames
            and dropout_count
            <= self.config.source_consensus_maximum_dropout_frames
        )
        qualification_latched = bool(
            slot and self._qualified_source_slot == slot
        )
        return {
            "slot": slot,
            "states": history,
            "observed_frame_count": len(history),
            "occupied_frame_count": int(occupied_count),
            "required_occupied_frame_count": int(
                self.config.source_consensus_minimum_occupied_frames
            ),
            "window_frame_count": int(self.config.source_consensus_window_frames),
            "dropout_frame_count": dropout_count,
            "maximum_dropout_frame_count": int(
                self.config.source_consensus_maximum_dropout_frames
            ),
            "latest_is_occupied": latest_occupied,
            "evaluation_state": (
                "not_evaluated" if self._not_evaluated_frames else "evaluated"
            ),
            "not_evaluated_frame_count": self._not_evaluated_frames,
            "evidence_passed": evidence_passed,
            "qualification_latched": qualification_latched,
            "qualification_frame_sequence": (
                self._source_qualification_frame_sequence
                if qualification_latched
                else None
            ),
            "qualification_captured_monotonic_s": (
                self._source_qualification_captured_monotonic_s
                if qualification_latched
                else None
            ),
            "explicit_empty_streak": (
                self._source_explicit_empty_streak
                if slot == self.source_slot
                else 0
            ),
            "passed": bool(evidence_passed or qualification_latched),
        }

    def _maybe_latch_source_qualification(self) -> None:
        slot = self.source_slot
        if slot is None or self._qualified_source_slot == slot:
            return
        consensus = self.source_consensus(slot)
        if consensus["evidence_passed"] is not True:
            return
        self._qualified_source_slot = slot
        self._source_qualification_frame_sequence = self.latest_frame_sequence
        self._source_qualification_captured_monotonic_s = (
            self.latest_frame_captured_monotonic_s
        )
        self._source_explicit_empty_streak = 0
        self._event(
            "source_qualification_latched",
            slot=slot,
            frame_sequence=self.latest_frame_sequence,
        )

    def select_source(self, slot_name: str) -> None:
        if self.phase not in {
            TransferPhase.IDLE, TransferPhase.SOURCE_SELECTED, TransferPhase.SOURCE_READY,
            TransferPhase.ROUTE_READY, TransferPhase.BLOCKED,
        }:
            raise RuntimeError("cannot change the source during an active transfer")
        slot = str(slot_name)
        if slot not in self.slot_names:
            raise ValueError(f"unknown source slot {slot}")
        state = _slot_states(self.latest_result).get(slot)
        # Selection records operator intent; start_tracking still requires all
        # current metric, occupancy, registration and robot-state gates.
        if self.source_slot != slot:
            self._qualified_source_slot = None
            self._source_qualification_frame_sequence = None
            self._source_qualification_captured_monotonic_s = None
            self._source_explicit_empty_streak = 0
        self.source_slot = slot
        if self.destination_slot == slot:
            self.destination_slot = None
        self.phase = TransferPhase.SOURCE_SELECTED
        self.block_reason = ""
        self._event("source_selected", slot=slot, state=state)
        self._maybe_latch_source_qualification()
        self._refresh_ready_phase()

    def select_destination(self, slot_name: str) -> None:
        if self.phase not in {
            TransferPhase.IDLE, TransferPhase.SOURCE_SELECTED, TransferPhase.SOURCE_READY,
            TransferPhase.ROUTE_READY, TransferPhase.BLOCKED,
        }:
            raise RuntimeError("cannot change the destination during an active transfer")
        slot = str(slot_name)
        if self.source_slot is None:
            raise ValueError("select a source wafer before selecting a destination")
        if slot not in self.slot_names:
            raise ValueError(f"unknown destination slot {slot}")
        if slot == self.source_slot:
            raise ValueError("source and destination slots must be different")
        state = _slot_states(self.latest_result).get(slot)
        self.destination_slot = slot
        self.block_reason = ""
        self._event("destination_selected", slot=slot, state=state)
        self._refresh_ready_phase()

    def _source_tracking_gates(self) -> dict[str, dict[str, Any]]:
        states = _slot_states(self.latest_result)
        source = states.get(self.source_slot or "")
        consensus = self.source_consensus()
        pose_passed = bool(
            self.latest_result is not None
            and self.latest_result.quality_passed
            and self.latest_result.coordinate_mapping_allowed
        )
        transform = _validated_transform_W_T(self.registration)
        state_fresh = self.latest_robot_state is not None
        metric_checks = (
            {} if self.latest_result is None
            else self.latest_result.projection_diagnostics.get("metric_slot_checks", {})
        )
        source_metric = metric_checks.get(self.source_slot or "", {})
        source_state = None if source is None else source["state"]
        tolerated_after_proof = {
            SlotState.WARNING.value,
            SlotState.UNKNOWN.value,
            SlotState.OUT_OF_VIEW.value,
            SlotState.OCCLUDED.value,
            SlotState.EMPTY.value,
            SlotState.EMPTY_UNREAD_MARKER.value,
        }
        qualified_source_usable = bool(
            consensus["qualification_latched"] is True
            and (
                source_state == SlotState.OCCUPIED.value
                or (
                    source_state in tolerated_after_proof
                    and consensus["explicit_empty_streak"] < 3
                )
            )
        )
        return {
            "overview_pose_quality": {
                "passed": pose_passed,
                "actual": (
                    None
                    if self.latest_result is None
                    else self.latest_result.pose.reprojection_rms_px
                ),
                "limit": "existing Tray pose quality gates",
                "reason": (
                    "no camera frame" if self.latest_result is None
                    else self.latest_result.failure_reason
                ),
            },
            "source_normal_occupied_consensus": {
                "passed": qualified_source_usable,
                "actual": {
                    "latest_state": source_state,
                    "occupied_frames": consensus["occupied_frame_count"],
                    "observed_frames": consensus["observed_frame_count"],
                    "qualification_latched": consensus[
                        "qualification_latched"
                    ],
                    "qualification_frame_sequence": consensus[
                        "qualification_frame_sequence"
                    ],
                    "explicit_empty_streak": consensus[
                        "explicit_empty_streak"
                    ],
                },
                "limit": (
                    f"first establish {self.config.source_consensus_minimum_occupied_frames}/"
                    f"{self.config.source_consensus_window_frames} occupied frames; then retain "
                    "through warning/unknown/occlusion and fewer than 3 explicit-empty frames"
                ),
            },
            "runtime_registration": {
                "passed": transform is not None,
                "actual": (
                    None
                    if self.registration is None
                    else self.registration.get("status")
                ),
                "limit": "status=success and finite W<-T",
            },
            "source_nominal_slot_geometry": {
                # Camera 1 only moves at the existing J3 clearance height and
                # targets the nominal tray-slot centre.  Once two occupied
                # frames have qualified that slot, the per-frame wafer contour
                # is diagnostic information; making it an authorization gate
                # merely reintroduces the occupied/warning flicker under a
                # second name.  Current tray pose/marker quality is still a
                # separate mandatory gate above, and stacked/outside states
                # are rejected by qualified_source_usable.
                "passed": source is not None,
                "actual": {
                    "slot_state": source_state,
                    "wafer_boundary_diagnostic": source_metric or None,
                    "qualification_latched": consensus[
                        "qualification_latched"
                    ],
                },
                "limit": (
                    "current nominal slot projection present; per-frame wafer "
                    "boundary is diagnostic-only for camera-1 clearance-height XY"
                ),
            },
            "fresh_frame_synchronised_robot_state": {
                "passed": state_fresh,
                "actual": (
                    None
                    if self.latest_robot_state is None
                    else self.latest_robot_state.captured_monotonic_s
                ),
                "limit": {
                    "state_age_s": self.config.maximum_robot_state_age_s,
                    "frame_robot_skew_s": self.config.maximum_frame_robot_skew_s,
                },
            },
        }

    def _xy_motion_gates(self) -> dict[str, dict[str, Any]]:
        """Gates that remain relevant after the source/world target is locked.

        The selected wafer is allowed to leave view or become occluded by the
        arm.  Current marker geometry is still mandatory so each observation
        can independently check that the tray has not moved.  Controller state
        for the actual command comes directly from ActionWorker, not from this
        asynchronously sampled UI stream.
        """

        acquisition = self._source_tracking_gates()
        transform = _validated_transform_W_T(self.registration)
        source = _slot_states(self.latest_result).get(self.source_slot or "")
        source_state = None if source is None else source.get("state")
        temporarily_unseen = {
            SlotState.WARNING.value,
            SlotState.UNKNOWN.value,
            SlotState.OUT_OF_VIEW.value,
            SlotState.OCCLUDED.value,
        }
        explicit_empty = {
            SlotState.EMPTY.value,
            SlotState.EMPTY_UNREAD_MARKER.value,
        }
        consensus = self.source_consensus()
        source_continuity = bool(
            source_state == SlotState.OCCUPIED.value
            or (self.target_lock_active() and source_state in temporarily_unseen)
            or (
                self.target_lock_active()
                and source_state in explicit_empty
                and consensus["qualification_latched"] is True
                and consensus["explicit_empty_streak"] < 3
            )
        )
        return {
            "current_overview_pose_quality": dict(
                acquisition["overview_pose_quality"]
            ),
            "locked_runtime_registration": {
                "passed": bool(self.target_lock_active() and transform is not None),
                "actual": {
                    "target_lock_active": self.target_lock_active(),
                    "registration_status": (
                        None
                        if self.registration is None
                        else self.registration.get("status")
                    ),
                },
                "limit": "target locked and finite W<-T retained from ARM",
            },
            "locked_source_not_explicitly_contradicted": {
                "passed": source_continuity,
                "actual": source_state,
                "limit": (
                    "occupied, warning/unknown/out_of_view/occluded after proof, or fewer "
                    "than 3 consecutive explicit-empty frames; stacked/outside rejects"
                ),
            },
            "fresh_frame_synchronised_robot_state": dict(
                acquisition["fresh_frame_synchronised_robot_state"]
            ),
        }

    def _route_gates(self) -> dict[str, dict[str, Any]]:
        gates = self._source_tracking_gates()
        destination = _slot_states(self.latest_result).get(
            self.destination_slot or ""
        )
        gates["destination_proven_empty"] = {
            "passed": bool(
                destination is not None
                and destination["safe_to_use_as_empty"] is True
            ),
            "actual": None if destination is None else destination["state"],
            "limit": "safe_to_use_as_empty=true",
        }
        return gates

    def _selection_gates(self) -> dict[str, dict[str, Any]]:
        """Compatibility alias for the source-only pickup navigation gates."""

        return self._source_tracking_gates()

    def _refresh_ready_phase(self) -> None:
        if self.phase in {
            TransferPhase.TRACKING_PICK,
            TransferPhase.WAITING_PICK_ALIGNMENT,
            TransferPhase.VERIFYING_PICK,
            TransferPhase.PICKED,
            TransferPhase.TRACKING_PLACE,
            TransferPhase.WAITING_PLACE_ALIGNMENT,
            TransferPhase.READY_TO_PLACE,
            TransferPhase.VERIFYING_PLACE,
            TransferPhase.COMPLETE,
            TransferPhase.BLOCKED,
        }:
            return
        if self.source_slot is None:
            self.phase = TransferPhase.IDLE
            return
        source_ready = all(
            gate["passed"] for gate in self._source_tracking_gates().values()
        )
        route_ready = bool(
            self.destination_slot is not None
            and all(gate["passed"] for gate in self._route_gates().values())
        )
        if route_ready:
            self.phase = TransferPhase.ROUTE_READY
        elif source_ready:
            self.phase = TransferPhase.SOURCE_READY
        else:
            self.phase = TransferPhase.SOURCE_SELECTED

    def start_tracking(self) -> None:
        gates = self._source_tracking_gates()
        failed = [name for name, gate in gates.items() if gate["passed"] is not True]
        if failed:
            raise RuntimeError("pickup navigation is not ready: " + ", ".join(failed))
        self.phase = TransferPhase.TRACKING_PICK
        self.block_reason = ""
        self._event(
            "tracking_started",
            source=self.source_slot,
            destination=self.destination_slot,
        )
        self._advance_tracking_phase()

    def active_target_slot(self) -> Optional[str]:
        if self.phase in {
            TransferPhase.PICKED,
            TransferPhase.TRACKING_PLACE,
            TransferPhase.WAITING_PLACE_ALIGNMENT,
            TransferPhase.READY_TO_PLACE,
            TransferPhase.VERIFYING_PLACE,
            TransferPhase.COMPLETE,
        }:
            return self.destination_slot
        return self.source_slot

    def slot_world_xy(self, slot_name: Optional[str]) -> Optional[np.ndarray]:
        if slot_name is None:
            return None
        transform = _validated_transform_W_T(self.registration)
        point = self.geometry.get("slots", {}).get(str(slot_name))
        if transform is None or point is None:
            return None
        tray_point = np.asarray(point, dtype=np.float64).reshape(3)
        world = transform @ np.asarray([*tray_point, 1.0], dtype=np.float64)
        return world[:2]

    def active_delta_world_xy(self) -> Optional[np.ndarray]:
        target = self.slot_world_xy(self.active_target_slot())
        if target is None or self.latest_robot_state is None:
            return None
        return target - np.asarray(self.latest_robot_state.pose[:2], dtype=np.float64)

    def _advance_tracking_phase(self) -> None:
        delta = self.active_delta_world_xy()
        if delta is None:
            return
        distance = float(np.linalg.norm(delta))
        if (
            self.phase is TransferPhase.TRACKING_PICK
            and distance <= self.config.coarse_arrival_distance_mm
        ):
            self.phase = TransferPhase.WAITING_PICK_ALIGNMENT
            self._event("pick_coarse_arrival", distance_mm=distance)
        elif (
            self.phase is TransferPhase.TRACKING_PLACE
            and distance <= self.config.coarse_arrival_distance_mm
        ):
            self.phase = TransferPhase.WAITING_PLACE_ALIGNMENT
            self._event("place_coarse_arrival", distance_mm=distance)

    def update_close_range(
        self,
        observation: CloseRangeSlotObservation,
        *,
        robot_state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        active = self.active_target_slot()
        if active is None or observation.target_slot != active:
            raise ValueError(
                "close-range observation target does not match the locked active slot"
            )
        self.close_range = observation
        state: Optional[RobotStateSnapshot] = None
        try:
            if robot_state is not None:
                state = RobotStateSnapshot.from_mapping(robot_state)
        except (TypeError, ValueError, OverflowError):
            state = None
        now = time.monotonic()
        observation_time = _finite_float_or_none(observation.captured_monotonic_s)
        observation_j3 = _finite_float_or_none(observation.robot_j3_mm)
        observation_age = (
            math.inf
            if observation_time is None
            else now - observation_time
        )
        state_skew = (
            math.inf
            if state is None
            or observation_time is None
            else abs(state.captured_monotonic_s - observation_time)
        )
        j3_disagreement = (
            math.inf
            if state is None
            or observation_j3 is None
            else abs(float(state.joints[2]) - observation_j3)
        )
        affirmative_evidence = (
            observation.pickup_alignment_allowed
            if observation.operation is CloseRangeOperation.PICK
            else observation.placement_allowed
        )
        self.close_range_gates = {
            "affirmative_visual_evidence": {
                "passed": affirmative_evidence,
                "actual": {
                    "state": observation.state.value,
                    "errors": list(observation.evidence_errors()),
                },
                "limit": "complete calibrated evidence for the requested operation",
            },
            "fresh_close_range_frame": {
                "passed": (
                    -0.05
                    <= observation_age
                    <= self.config.maximum_close_range_age_s
                ),
                "actual": observation_age if math.isfinite(observation_age) else None,
                "limit": f"0..{self.config.maximum_close_range_age_s:.2f} s",
            },
            "close_range_robot_time_sync": {
                "passed": state_skew <= self.config.maximum_close_range_robot_skew_s,
                "actual": state_skew if math.isfinite(state_skew) else None,
                "limit": f"<={self.config.maximum_close_range_robot_skew_s:.2f} s",
            },
            "close_range_j3_consistency": {
                "passed": (
                    j3_disagreement
                    <= self.config.maximum_close_range_j3_disagreement_mm
                ),
                "actual": (
                    j3_disagreement if math.isfinite(j3_disagreement) else None
                ),
                "limit": (
                    f"<={self.config.maximum_close_range_j3_disagreement_mm:.2f} mm"
                ),
            },
        }
        self._event("close_range_observation", observation=observation.to_json())
        close_range_allowed = all(
            gate["passed"] is True for gate in self.close_range_gates.values()
        )
        if (
            self.phase is TransferPhase.WAITING_PICK_ALIGNMENT
            and observation.operation is CloseRangeOperation.PICK
            and close_range_allowed
        ):
            self.phase = TransferPhase.VERIFYING_PICK
        elif (
            self.phase is TransferPhase.WAITING_PLACE_ALIGNMENT
            and observation.operation is CloseRangeOperation.PLACE
            and close_range_allowed
        ):
            self.phase = TransferPhase.READY_TO_PLACE

    def _close_range_gates_passed(self) -> bool:
        return bool(self.close_range_gates) and all(
            gate.get("passed") is True for gate in self.close_range_gates.values()
        )

    def record_pick_verification(
        self,
        *,
        lifted_wafer_detected: bool,
        source_now_empty: bool,
    ) -> None:
        if self.phase not in {
            TransferPhase.VERIFYING_PICK,
            TransferPhase.WAITING_PICK_ALIGNMENT,
        }:
            raise RuntimeError("pick verification is not expected in the current phase")
        close_ok = bool(
            not self.config.pick_requires_close_range_alignment
            or (
                self.close_range is not None
                and self.close_range.pickup_alignment_allowed
                and self._close_range_gates_passed()
            )
        )
        if lifted_wafer_detected and source_now_empty and close_ok:
            self.phase = TransferPhase.PICKED
            self._event("pick_verified")
            return
        self.phase = TransferPhase.BLOCKED
        self.block_reason = (
            "pick verification did not prove both lifted wafer and empty source"
        )
        self._event(
            "pick_verification_failed",
            lifted_wafer_detected=bool(lifted_wafer_detected),
            source_now_empty=bool(source_now_empty),
            close_range_alignment=close_ok,
        )

    def start_place_tracking(self) -> None:
        if self.phase is not TransferPhase.PICKED:
            raise RuntimeError("placement tracking requires a verified picked wafer")
        if self.destination_slot is None:
            raise RuntimeError("select a proven-empty destination before placement tracking")
        destination = _slot_states(self.latest_result).get(self.destination_slot)
        if destination is None or destination["safe_to_use_as_empty"] is not True:
            raise RuntimeError("the selected destination is no longer proven empty")
        self.phase = TransferPhase.TRACKING_PLACE
        self.close_range = None
        self.close_range_gates = {}
        self._event("place_tracking_started", destination=self.destination_slot)
        self._advance_tracking_phase()

    def record_place_verification(
        self,
        *,
        lifted_wafer_absent: bool,
        destination_now_occupied: bool,
    ) -> None:
        if self.phase not in {
            TransferPhase.READY_TO_PLACE,
            TransferPhase.VERIFYING_PLACE,
        }:
            raise RuntimeError(
                "place verification is not expected in the current phase"
            )
        close_ok = bool(
            not self.config.place_requires_close_range_insertability
            or (
                self.close_range is not None
                and self.close_range.placement_allowed
                and self._close_range_gates_passed()
            )
        )
        if lifted_wafer_absent and destination_now_occupied and close_ok:
            self.phase = TransferPhase.COMPLETE
            self._event("place_verified")
            return
        self.phase = TransferPhase.BLOCKED
        self.block_reason = (
            "place verification did not prove released wafer and occupied destination"
        )
        self._event(
            "place_verification_failed",
            lifted_wafer_absent=bool(lifted_wafer_absent),
            destination_now_occupied=bool(destination_now_occupied),
            close_range_insertability=close_ok,
        )

    def snapshot(self) -> dict[str, Any]:
        states = _slot_states(self.latest_result)
        source_world = self.slot_world_xy(self.source_slot)
        destination_world = self.slot_world_xy(self.destination_slot)
        delta = self.active_delta_world_xy()
        distance = None if delta is None else float(np.linalg.norm(delta))
        navigation_gates = self._source_tracking_gates()
        xy_motion_gates = self._xy_motion_gates()
        route_gates = self._route_gates()
        consensus = self.source_consensus()
        result = self.latest_result
        pose = None if result is None else result.pose
        pose_diagnostics = {
            "projection_source": "unavailable" if result is None else result.projection_source,
            "metric_passed": bool(result and result.quality_passed and result.coordinate_mapping_allowed),
            "reason": "no camera frame" if result is None else result.failure_reason,
            "visible_marker_ids": [] if pose is None else list(pose.visible_marker_ids),
            "used_marker_ids": [] if pose is None else list(pose.used_marker_ids),
            "rejected_marker_ids": [] if pose is None else list(pose.rejected_marker_ids),
            "ransac_inlier_corners": 0 if pose is None else pose.ransac_inlier_corner_count,
            "per_marker_rms_px": {} if pose is None else dict(pose.per_marker_rms_px),
            "pose_rms_px": None if pose is None else pose.reprojection_rms_px,
            "grid_rms_px": (
                None if result is None or result.planar_registration is None
                else result.planar_registration.reprojection_rms_px
            ),
        }
        return {
            "schema_version": 1,
            "phase": self.phase.value,
            "source_slot": self.source_slot,
            "destination_slot": self.destination_slot,
            "active_target_slot": self.active_target_slot(),
            "source_state": states.get(self.source_slot or ""),
            "destination_state": states.get(self.destination_slot or ""),
            "latest_frame_sequence": self.latest_frame_sequence,
            "latest_frame_captured_monotonic_s": self.latest_frame_captured_monotonic_s,
            "robot_state": (
                None
                if self.latest_robot_state is None
                else self.latest_robot_state.to_json()
            ),
            "registration": self.registration,
            "source_world_xy_mm": (
                None if source_world is None else source_world.astype(float).tolist()
            ),
            "destination_world_xy_mm": (
                None
                if destination_world is None
                else destination_world.astype(float).tolist()
            ),
            "active_delta_world_xy_mm": (
                None if delta is None else delta.astype(float).tolist()
            ),
            "active_distance_mm": distance,
            "source_consensus": consensus,
            "pose_diagnostics": pose_diagnostics,
            "recent_overview_diagnostics": list(self.overview_diagnostics),
            "selection_gates": navigation_gates,
            "navigation_gates": navigation_gates,
            "xy_motion_gates": xy_motion_gates,
            "route_gates": route_gates,
            "coordinate_ready": _validated_transform_W_T(self.registration) is not None,
            "registration_locked": self.target_lock_active(),
            "tracking_ready": all(
                gate["passed"] for gate in navigation_gates.values()
            ),
            "route_ready": bool(
                self.destination_slot is not None
                and all(gate["passed"] for gate in route_gates.values())
            ),
            "robot_motion_authorized": False,
            "motion_authorization_note": (
                "This session is observation-only. Existing ActionWorker P22 safety paths remain authoritative."
            ),
            "close_range": (
                None if self.close_range is None else self.close_range.to_json()
            ),
            "close_range_gates": dict(self.close_range_gates),
            "block_reason": self.block_reason,
            "config": asdict(self.config),
            "events": list(self.events),
        }


__all__ = [
    "DEFAULT_WAFER_TRANSFER_CONFIG",
    "RobotStateSnapshot",
    "TransferPhase",
    "WaferTransferConfig",
    "WaferTransferSession",
]
