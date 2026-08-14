"""Temporal quality gate and smoothing for live ``^C T_T`` estimation.

The per-frame estimator remains the authority for geometric validity.  This
module accepts only quality-passed poses, rejects implausible frame-to-frame
jumps, and applies exponential smoothing on translation and SO(3) rotation.
It does not hide the raw estimate: callers receive raw and filtered transforms
separately for logging and debugging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .tray_pose_estimator import TrayBoardPoseEstimator, TrayPoseEstimate


@dataclass(frozen=True)
class TrayPoseTrackerConfig:
    translation_alpha: float = 0.35
    rotation_alpha: float = 0.35
    maximum_translation_jump_mm: float = 35.0
    maximum_rotation_jump_deg: float = 20.0
    reset_after_lost_frames: int = 15

    def __post_init__(self) -> None:
        if not 0.0 < self.translation_alpha <= 1.0:
            raise ValueError("translation_alpha 必须位于(0,1]")
        if not 0.0 < self.rotation_alpha <= 1.0:
            raise ValueError("rotation_alpha 必须位于(0,1]")
        if self.maximum_translation_jump_mm <= 0.0:
            raise ValueError("maximum_translation_jump_mm 必须大于0")
        if self.maximum_rotation_jump_deg <= 0.0:
            raise ValueError("maximum_rotation_jump_deg 必须大于0")
        if self.reset_after_lost_frames < 1:
            raise ValueError("reset_after_lost_frames 必须至少为1")


DEFAULT_TRACKER_CONFIG = TrayPoseTrackerConfig()


@dataclass(frozen=True)
class TrackedTrayPose:
    raw: TrayPoseEstimate
    accepted_by_tracker: bool
    tracker_reason: Optional[str]
    filtered_T_C_T: Optional[np.ndarray]
    filtered_T_T_C: Optional[np.ndarray]
    translation_jump_mm: Optional[float]
    rotation_jump_deg: Optional[float]
    lost_frame_count: int


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    """Return the magnitude of one SO(3) rotation in degrees."""
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _smooth_rotation(
    previous: np.ndarray,
    current: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Interpolate rotation on SO(3): R_f = R_p exp(alpha log(R_p^T R_c))."""
    relative = previous.T @ current
    relative_rvec, _ = cv2.Rodrigues(relative)
    increment, _ = cv2.Rodrigues(alpha * relative_rvec)
    smoothed = previous @ increment
    # Project tiny floating error back onto SO(3).
    u, _singular, vt = np.linalg.svd(smoothed)
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def _invert_rigid(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ translation
    return result


class TrayPoseTracker:
    """Run the frame estimator and maintain one explicitly filtered pose."""

    def __init__(
        self,
        estimator: TrayBoardPoseEstimator,
        config: TrayPoseTrackerConfig = DEFAULT_TRACKER_CONFIG,
    ) -> None:
        self.estimator = estimator
        self.config = config
        self._filtered_T_C_T: Optional[np.ndarray] = None
        self._lost_frames = 0

    def reset(self) -> None:
        self._filtered_T_C_T = None
        self._lost_frames = 0

    @property
    def filtered_T_C_T(self) -> Optional[np.ndarray]:
        return (
            None
            if self._filtered_T_C_T is None
            else self._filtered_T_C_T.copy()
        )

    def update(self, image: np.ndarray) -> TrackedTrayPose:
        raw = self.estimator.estimate(image)
        if not raw.quality_passed or raw.T_C_T is None:
            self._lost_frames += 1
            if self._lost_frames >= self.config.reset_after_lost_frames:
                self._filtered_T_C_T = None
            return TrackedTrayPose(
                raw=raw,
                accepted_by_tracker=False,
                tracker_reason=raw.failure_reason or "当前帧未通过位姿质量门",
                filtered_T_C_T=self.filtered_T_C_T,
                filtered_T_T_C=(
                    None
                    if self._filtered_T_C_T is None
                    else _invert_rigid(self._filtered_T_C_T)
                ),
                translation_jump_mm=None,
                rotation_jump_deg=None,
                lost_frame_count=self._lost_frames,
            )

        current = raw.T_C_T
        if self._filtered_T_C_T is None:
            self._filtered_T_C_T = current.copy()
            self._lost_frames = 0
            return TrackedTrayPose(
                raw=raw,
                accepted_by_tracker=True,
                tracker_reason=None,
                filtered_T_C_T=self.filtered_T_C_T,
                filtered_T_T_C=_invert_rigid(self._filtered_T_C_T),
                translation_jump_mm=0.0,
                rotation_jump_deg=0.0,
                lost_frame_count=0,
            )

        previous = self._filtered_T_C_T
        translation_jump = float(
            np.linalg.norm(current[:3, 3] - previous[:3, 3])
        )
        rotation_jump = _rotation_angle_deg(
            previous[:3, :3].T @ current[:3, :3]
        )
        if (
            translation_jump > self.config.maximum_translation_jump_mm
            or rotation_jump > self.config.maximum_rotation_jump_deg
        ):
            self._lost_frames += 1
            return TrackedTrayPose(
                raw=raw,
                accepted_by_tracker=False,
                tracker_reason=(
                    f"位姿跳变被拒绝：{translation_jump:.1f}mm, "
                    f"{rotation_jump:.1f}deg"
                ),
                filtered_T_C_T=self.filtered_T_C_T,
                filtered_T_T_C=_invert_rigid(self._filtered_T_C_T),
                translation_jump_mm=translation_jump,
                rotation_jump_deg=rotation_jump,
                lost_frame_count=self._lost_frames,
            )

        filtered = np.eye(4, dtype=np.float64)
        filtered[:3, :3] = _smooth_rotation(
            previous[:3, :3],
            current[:3, :3],
            self.config.rotation_alpha,
        )
        filtered[:3, 3] = (
            (1.0 - self.config.translation_alpha) * previous[:3, 3]
            + self.config.translation_alpha * current[:3, 3]
        )
        self._filtered_T_C_T = filtered
        self._lost_frames = 0
        return TrackedTrayPose(
            raw=raw,
            accepted_by_tracker=True,
            tracker_reason=None,
            filtered_T_C_T=self.filtered_T_C_T,
            filtered_T_T_C=_invert_rigid(self._filtered_T_C_T),
            translation_jump_mm=translation_jump,
            rotation_jump_deg=rotation_jump,
            lost_frame_count=0,
        )


__all__ = [
    "DEFAULT_TRACKER_CONFIG",
    "TrackedTrayPose",
    "TrayPoseTracker",
    "TrayPoseTrackerConfig",
]
