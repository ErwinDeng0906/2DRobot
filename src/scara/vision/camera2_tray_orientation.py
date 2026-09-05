"""Camera-2 evidence for visually squaring the tray with J4 only.

The close-range camera rotates with J4.  Consequently the tray's residual
image angle is also the J4 correction signal: increasing tool Rz increases the
observed image angle.  This module measures that signal from decoded slot
ArUco squares and deliberately does not authorize motion itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import cv2
import numpy as np

from .slot_marker_observation import detect_aruco_observations


CAMERA2_DICTIONARY = "DICT_4X4_50"
MINIMUM_MARKER_PERIMETER_PX = 80.0
MINIMUM_MARKER_SQUARE_QUALITY = 0.72
MINIMUM_ORIENTATION_CONCENTRATION = 0.985
TRAY_MARKER_IDS = frozenset(range(1, 45))


def normalize_tray_axis_angle_deg(angle_deg: float) -> float:
    """Normalize an unoriented square edge to ``[-45, 45)`` degrees."""

    return float((float(angle_deg) + 45.0) % 90.0 - 45.0)


def _orientation_mean_deg(angles_deg: list[float], weights: list[float]) -> tuple[float, float]:
    if not angles_deg or len(angles_deg) != len(weights):
        raise ValueError("camera2 orientation requires weighted angles")
    radians = np.deg2rad(np.asarray(angles_deg, dtype=np.float64) * 4.0)
    weight = np.asarray(weights, dtype=np.float64)
    x = float(np.sum(weight * np.cos(radians)))
    y = float(np.sum(weight * np.sin(radians)))
    total = float(np.sum(weight))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("camera2 orientation weights are invalid")
    concentration = float(math.hypot(x, y) / total)
    mean = normalize_tray_axis_angle_deg(math.degrees(math.atan2(y, x)) / 4.0)
    return mean, concentration


@dataclass(frozen=True)
class Camera2TrayOrientationObservation:
    measurement_id: str
    frame_sequence: int
    captured_monotonic_s: float
    accepted: bool
    angle_error_deg: float | None
    marker_ids: tuple[int, ...]
    marker_count: int
    concentration: float | None
    maximum_marker_deviation_deg: float | None
    rejection_reasons: tuple[str, ...]
    annotated_bgr: np.ndarray

    def to_sample(self, robot_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "measurement_id": self.measurement_id,
            "frame_sequence": int(self.frame_sequence),
            "captured_monotonic_s": float(self.captured_monotonic_s),
            "accepted": bool(self.accepted),
            "camera_source": 2,
            "angle_error_deg": self.angle_error_deg,
            "marker_ids": list(self.marker_ids),
            "marker_count": int(self.marker_count),
            "concentration": self.concentration,
            "maximum_marker_deviation_deg": self.maximum_marker_deviation_deg,
            "rejection_reasons": list(self.rejection_reasons),
            "robot_state": None if robot_state is None else dict(robot_state),
            "annotated_bgr": self.annotated_bgr.copy(),
        }


def observe_camera2_tray_orientation(
    image_bgr: np.ndarray,
    *,
    frame_sequence: int,
    captured_monotonic_s: float,
) -> Camera2TrayOrientationObservation:
    """Measure tray tilt in the camera-2 image from complete decoded markers."""

    if image_bgr is None or image_bgr.ndim != 3:
        raise ValueError("camera2 image must be a BGR array")
    observations = detect_aruco_observations(
        image_bgr,
        CAMERA2_DICTIONARY,
        scales=(1.0, 1.5, 2.0),
        include_clahe=True,
    )
    accepted_markers = [
        observation
        for observation in observations.values()
        if observation.marker_id in TRAY_MARKER_IDS
        and observation.complete_decoded
        and observation.perimeter_px >= MINIMUM_MARKER_PERIMETER_PX
        and observation.square_quality >= MINIMUM_MARKER_SQUARE_QUALITY
    ]
    angles = [normalize_tray_axis_angle_deg(item.angle_deg) for item in accepted_markers]
    weights = [max(1.0, item.perimeter_px) * max(0.1, item.square_quality) for item in accepted_markers]
    angle: float | None = None
    concentration: float | None = None
    maximum_deviation: float | None = None
    rejection_reasons: list[str] = []
    if not accepted_markers:
        rejection_reasons.append("相机2未检测到完整且合格的槽位ArUco方框")
    else:
        angle, concentration = _orientation_mean_deg(angles, weights)
        maximum_deviation = max(
            abs(normalize_tray_axis_angle_deg(item_angle - angle))
            for item_angle in angles
        )
        if concentration < MINIMUM_ORIENTATION_CONCENTRATION:
            rejection_reasons.append("相机2多个标记的方向不一致")

    annotated = image_bgr.copy()
    for item in accepted_markers:
        corners = np.asarray(item.corners_px, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [corners], True, (0, 220, 255), 2, cv2.LINE_AA)
        center = tuple(int(round(value)) for value in item.center_px)
        cv2.putText(
            annotated,
            f"ID {item.marker_id}",
            (center[0] - 34, center[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    accepted = not rejection_reasons and angle is not None
    label = (
        f"CAM2 tray angle {angle:+.3f} deg | markers {len(accepted_markers)}"
        if accepted
        else "CAM2 tray angle WAIT"
    )
    color = (0, 220, 0) if accepted else (0, 0, 255)
    cv2.rectangle(annotated, (8, 8), (min(760, annotated.shape[1] - 8), 46), (20, 20, 20), -1)
    cv2.putText(
        annotated,
        label,
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        color,
        2,
        cv2.LINE_AA,
    )
    return Camera2TrayOrientationObservation(
        measurement_id=f"camera2-sequence-{int(frame_sequence)}",
        frame_sequence=int(frame_sequence),
        captured_monotonic_s=float(captured_monotonic_s),
        accepted=accepted,
        angle_error_deg=angle,
        marker_ids=tuple(sorted(item.marker_id for item in accepted_markers)),
        marker_count=len(accepted_markers),
        concentration=concentration,
        maximum_marker_deviation_deg=maximum_deviation,
        rejection_reasons=tuple(rejection_reasons),
        annotated_bgr=annotated,
    )


__all__ = [
    "Camera2TrayOrientationObservation",
    "normalize_tray_axis_angle_deg",
    "observe_camera2_tray_orientation",
]
