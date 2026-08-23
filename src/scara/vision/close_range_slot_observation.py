"""Camera-2 close-range observation contract for markerless slot alignment.

The close camera may lose every ArUco marker near pickup or placement.  This
module defines the evidence that a future edge detector must provide without
pretending that such a detector already exists.  The default observer always
fails closed and never authorizes pickup or placement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Sequence

import numpy as np


def _finite_scalar(value: Optional[float]) -> bool:
    try:
        return value is not None and bool(np.isfinite(float(value)))
    except (TypeError, ValueError, OverflowError):
        return False


def _finite_point(value: Optional[Sequence[float]]) -> bool:
    if value is None:
        return False
    try:
        point = np.asarray(value, dtype=np.float64).reshape(-1)
        return point.size == 2 and bool(np.all(np.isfinite(point)))
    except (TypeError, ValueError, OverflowError):
        return False


def _finite_quad(value: Sequence[Sequence[float]]) -> bool:
    try:
        corners = np.asarray(value, dtype=np.float64)
        if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
            return False
        shifted = np.roll(corners, -1, axis=0)
        twice_area = float(
            abs(np.sum(corners[:, 0] * shifted[:, 1] - corners[:, 1] * shifted[:, 0]))
        )
        edges = shifted - corners
        following_edges = np.roll(edges, -1, axis=0)
        crosses = (
            edges[:, 0] * following_edges[:, 1]
            - edges[:, 1] * following_edges[:, 0]
        )
        return twice_area >= 2.0 and bool(
            np.all(crosses > 1e-6) or np.all(crosses < -1e-6)
        )
    except (TypeError, ValueError, OverflowError):
        return False


class CloseRangeOperation(str, Enum):
    PICK = "pick"
    PLACE = "place"


class CloseRangeState(str, Enum):
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"
    TRACKING = "tracking"
    ALIGNED = "aligned"
    INSERTABLE = "insertable"
    UNSAFE = "unsafe"


@dataclass(frozen=True)
class CloseRangeSlotObservation:
    """One timestamped camera-2 result expressed entirely in image evidence."""

    operation: CloseRangeOperation
    state: CloseRangeState
    target_slot: str
    measurement_id: str
    captured_monotonic_s: Optional[float]
    quality_passed: bool
    slot_center_px: Optional[tuple[float, float]] = None
    slot_corners_px: tuple[tuple[float, float], ...] = ()
    suction_center_px: Optional[tuple[float, float]] = None
    wafer_center_px: Optional[tuple[float, float]] = None
    wafer_corners_px: tuple[tuple[float, float], ...] = ()
    center_error_px: Optional[tuple[float, float]] = None
    angle_error_deg: Optional[float] = None
    minimum_clearance_px: Optional[float] = None
    edge_fit_rms_px: Optional[float] = None
    robot_j3_mm: Optional[float] = None
    calibration_profile: str = ""
    valid_j3_range_mm: Optional[tuple[float, float]] = None
    maximum_center_error_px: Optional[float] = None
    maximum_angle_error_deg: Optional[float] = None
    maximum_edge_fit_rms_px: Optional[float] = None
    minimum_required_clearance_px: Optional[float] = None
    lifted_wafer_detected: Optional[bool] = None
    wafer_attached_to_suction: Optional[bool] = None
    reason: str = ""
    flags: tuple[str, ...] = ()

    def evidence_errors(self) -> tuple[str, ...]:
        """Return missing/invalid evidence for an affirmative close-range result."""

        errors: list[str] = []
        if self.quality_passed is not True:
            errors.append("quality_not_passed")
        if not self.target_slot:
            errors.append("target_slot_missing")
        if not self.measurement_id:
            errors.append("measurement_id_missing")
        if not _finite_scalar(self.captured_monotonic_s):
            errors.append("capture_timestamp_invalid")
        if not self.calibration_profile:
            errors.append("calibration_profile_missing")
        if not _finite_scalar(self.robot_j3_mm):
            errors.append("robot_j3_invalid")
        if self.valid_j3_range_mm is None:
            errors.append("valid_j3_range_missing")
        else:
            try:
                j3_range = np.asarray(
                    self.valid_j3_range_mm, dtype=np.float64
                ).reshape(-1)
            except (TypeError, ValueError, OverflowError):
                j3_range = np.asarray([], dtype=np.float64)
            if (
                j3_range.size != 2
                or not np.all(np.isfinite(j3_range))
                or float(j3_range[0]) > float(j3_range[1])
            ):
                errors.append("valid_j3_range_invalid")
            elif _finite_scalar(self.robot_j3_mm) and not (
                float(j3_range[0])
                <= float(self.robot_j3_mm)
                <= float(j3_range[1])
            ):
                errors.append("robot_j3_outside_calibrated_range")
        if not _finite_point(self.wafer_center_px):
            errors.append("wafer_center_invalid")
        if not _finite_quad(self.wafer_corners_px):
            errors.append("wafer_corners_invalid")
        if not _finite_point(self.center_error_px):
            errors.append("center_error_invalid")
        if not _finite_scalar(self.maximum_center_error_px) or float(
            self.maximum_center_error_px
        ) < 0.0:
            errors.append("maximum_center_error_invalid")
        elif _finite_point(self.center_error_px):
            center_error = np.asarray(self.center_error_px, dtype=np.float64)
            if float(np.linalg.norm(center_error)) > float(
                self.maximum_center_error_px
            ):
                errors.append("center_error_exceeds_limit")
        if not _finite_scalar(self.angle_error_deg):
            errors.append("angle_error_invalid")
        if not _finite_scalar(self.maximum_angle_error_deg) or float(
            self.maximum_angle_error_deg
        ) < 0.0:
            errors.append("maximum_angle_error_invalid")
        elif _finite_scalar(self.angle_error_deg) and abs(
            float(self.angle_error_deg)
        ) > float(self.maximum_angle_error_deg):
            errors.append("angle_error_exceeds_limit")
        if not _finite_scalar(self.edge_fit_rms_px):
            errors.append("edge_fit_rms_invalid")
        if not _finite_scalar(self.maximum_edge_fit_rms_px) or float(
            self.maximum_edge_fit_rms_px
        ) < 0.0:
            errors.append("maximum_edge_fit_rms_invalid")
        elif _finite_scalar(self.edge_fit_rms_px) and (
            float(self.edge_fit_rms_px) < 0.0
            or float(self.edge_fit_rms_px) > float(self.maximum_edge_fit_rms_px)
        ):
            errors.append("edge_fit_rms_exceeds_limit")

        if self.operation is CloseRangeOperation.PICK:
            if not _finite_point(self.suction_center_px):
                errors.append("suction_center_invalid")
        else:
            if not _finite_point(self.slot_center_px):
                errors.append("slot_center_invalid")
            if not _finite_quad(self.slot_corners_px):
                errors.append("slot_corners_invalid")
            if not _finite_scalar(self.minimum_clearance_px):
                errors.append("minimum_clearance_invalid")
            if not _finite_scalar(self.minimum_required_clearance_px) or float(
                self.minimum_required_clearance_px
            ) < 0.0:
                errors.append("minimum_required_clearance_invalid")
            elif _finite_scalar(self.minimum_clearance_px) and float(
                self.minimum_clearance_px
            ) < float(self.minimum_required_clearance_px):
                errors.append("minimum_clearance_below_limit")
            if self.lifted_wafer_detected is not True:
                errors.append("lifted_wafer_not_confirmed")
            if self.wafer_attached_to_suction is not True:
                errors.append("wafer_attachment_not_confirmed")
        return tuple(errors)

    @property
    def evidence_complete(self) -> bool:
        return not self.evidence_errors()

    @property
    def pickup_alignment_allowed(self) -> bool:
        return bool(
            self.operation is CloseRangeOperation.PICK
            and self.evidence_complete
            and self.state in {CloseRangeState.ALIGNED, CloseRangeState.INSERTABLE}
        )

    @property
    def placement_allowed(self) -> bool:
        return bool(
            self.operation is CloseRangeOperation.PLACE
            and self.evidence_complete
            and self.state is CloseRangeState.INSERTABLE
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "state": self.state.value,
            "target_slot": self.target_slot,
            "measurement_id": self.measurement_id,
            "captured_monotonic_s": self.captured_monotonic_s,
            "quality_passed": self.quality_passed,
            "slot_center_px": (
                None if self.slot_center_px is None else list(self.slot_center_px)
            ),
            "slot_corners_px": [list(point) for point in self.slot_corners_px],
            "suction_center_px": (
                None if self.suction_center_px is None else list(self.suction_center_px)
            ),
            "wafer_center_px": (
                None if self.wafer_center_px is None else list(self.wafer_center_px)
            ),
            "wafer_corners_px": [list(point) for point in self.wafer_corners_px],
            "center_error_px": (
                None if self.center_error_px is None else list(self.center_error_px)
            ),
            "angle_error_deg": self.angle_error_deg,
            "minimum_clearance_px": self.minimum_clearance_px,
            "edge_fit_rms_px": self.edge_fit_rms_px,
            "robot_j3_mm": self.robot_j3_mm,
            "calibration_profile": self.calibration_profile,
            "valid_j3_range_mm": (
                None
                if self.valid_j3_range_mm is None
                else list(self.valid_j3_range_mm)
            ),
            "maximum_center_error_px": self.maximum_center_error_px,
            "maximum_angle_error_deg": self.maximum_angle_error_deg,
            "maximum_edge_fit_rms_px": self.maximum_edge_fit_rms_px,
            "minimum_required_clearance_px": self.minimum_required_clearance_px,
            "lifted_wafer_detected": self.lifted_wafer_detected,
            "wafer_attached_to_suction": self.wafer_attached_to_suction,
            "evidence_complete": self.evidence_complete,
            "evidence_errors": list(self.evidence_errors()),
            "pickup_alignment_allowed": self.pickup_alignment_allowed,
            "placement_allowed": self.placement_allowed,
            "reason": self.reason,
            "flags": list(self.flags),
        }

    @classmethod
    def unavailable(
        cls,
        operation: CloseRangeOperation,
        target_slot: str,
        reason: str,
    ) -> "CloseRangeSlotObservation":
        return cls(
            operation=operation,
            state=CloseRangeState.UNAVAILABLE,
            target_slot=str(target_slot),
            measurement_id="",
            captured_monotonic_s=None,
            quality_passed=False,
            reason=str(reason),
            flags=("close_range_observer_unavailable",),
        )


class CloseRangeSlotObserver(Protocol):
    """Interface implemented later by the markerless camera-2 edge detector."""

    def observe(
        self,
        image_bgr: np.ndarray,
        *,
        operation: CloseRangeOperation,
        target_slot: str,
        measurement_id: str,
        captured_monotonic_s: float,
        robot_state: Optional[Mapping[str, Any]] = None,
        predicted_slot_center_px: Optional[Sequence[float]] = None,
    ) -> CloseRangeSlotObservation: ...


class UnavailableCloseRangeSlotObserver:
    """Explicit fail-closed placeholder until camera-2 data are calibrated."""

    def __init__(
        self, reason: str = "camera-2 markerless slot-edge detector is not installed"
    ) -> None:
        self.reason = str(reason)

    def observe(
        self,
        image_bgr: np.ndarray,
        *,
        operation: CloseRangeOperation,
        target_slot: str,
        measurement_id: str,
        captured_monotonic_s: float,
        robot_state: Optional[Mapping[str, Any]] = None,
        predicted_slot_center_px: Optional[Sequence[float]] = None,
    ) -> CloseRangeSlotObservation:
        del (
            image_bgr,
            measurement_id,
            captured_monotonic_s,
            robot_state,
            predicted_slot_center_px,
        )
        return CloseRangeSlotObservation.unavailable(
            operation,
            target_slot,
            self.reason,
        )


__all__ = [
    "CloseRangeOperation",
    "CloseRangeSlotObservation",
    "CloseRangeSlotObserver",
    "CloseRangeState",
    "UnavailableCloseRangeSlotObserver",
]
