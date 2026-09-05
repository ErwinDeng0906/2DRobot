"""Live, controller-free wafer-transfer vision runtime.

Camera 1 owns the global Tray frame, 36-slot occupancy, and runtime ``W<-T``
registration.  Camera 2 is consumed through a markerless close-range observer
contract.  This runtime only computes evidence and distances; it cannot move
the robot or operate vacuum/DO.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

from scara.file_io import atomic_write_text

from .close_range_slot_observation import (
    CloseRangeOperation,
    CloseRangeSlotObservation,
    CloseRangeSlotObserver,
    UnavailableCloseRangeSlotObserver,
)
from .handeye_interaction import load_latest_suction_target
from .runtime_tray_registration import (
    build_runtime_tray_registration,
    load_planar_handeye,
)
from .silicon_detection_config import (
    load_silicon_detection_config,
    preferred_silicon_detection_config_path,
)
from .slot_marker_observation import load_slot_marker_layout
from .tray_pose_estimator import (
    TrayBoardPoseEstimator,
    load_camera_intrinsics,
    load_tray_board_geometry,
)
from .tray_pose_tracker import TrayPoseTracker
from .tray_vision_fusion import TrayVisionAnalyzer, TrayVisionResult
from .wafer_transfer_tracking import WaferTransferSession


@dataclass(frozen=True)
class WaferTransferFrame:
    frame_sequence: int
    captured_monotonic_s: float
    result: TrayVisionResult
    session_snapshot: dict[str, Any]
    registration_candidate: Optional[dict[str, Any]]
    annotated_bgr: np.ndarray
    stream_epoch: int = 0


class LiveWaferTransferRuntime:
    """Process live frames and maintain one target-locked transfer session."""

    def __init__(
        self,
        project_root: Path,
        *,
        close_range_observer: Optional[CloseRangeSlotObserver] = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        calib = self.project_root / "src/scara/calib"
        self.geometry_path = calib / "tray_board_geometry.json"
        self.intrinsics_path = calib / "camera1_intrinsics.json"
        self.slot_layout_path = (
            self.project_root / "tools/tray_marker_detector_v2/tray_marker_layout.json"
        )
        self.geometry = load_tray_board_geometry(self.geometry_path)
        self.intrinsics = load_camera_intrinsics(self.intrinsics_path)
        self.silicon_detection_config_path = (
            preferred_silicon_detection_config_path(self.project_root)
        )
        self.silicon_detection_config = load_silicon_detection_config(
            self.silicon_detection_config_path
        )
        self.estimator = TrayBoardPoseEstimator(self.geometry, self.intrinsics, edge_refinement=True)
        self.tracker = TrayPoseTracker(self.estimator)
        self.analyzer = TrayVisionAnalyzer(
            self.estimator,
            self.geometry,
            load_slot_marker_layout(self.slot_layout_path),
            self.silicon_detection_config.fusion_config,
            consistent_slot_geometry=True,
        )
        self.session = WaferTransferSession(self.geometry)
        self.close_range_observer = (
            close_range_observer
            if close_range_observer is not None
            else UnavailableCloseRangeSlotObserver()
        )
        self._lock = threading.RLock()
        self._registration_samples: deque[dict[str, Any]] = deque(maxlen=5)
        self._last_result: Optional[TrayVisionResult] = None
        self._last_frame: Optional[WaferTransferFrame] = None
        self._stream_epoch = 0
        self._registration_candidate: Optional[dict[str, Any]] = None
        self._registration_error = ""
        self._calibration_error = ""
        self._suction = None
        self._handeye = None
        try:
            self._suction = load_latest_suction_target(self.project_root)
            self._handeye = load_planar_handeye(self.project_root, self._suction)
        except Exception as exc:  # noqa: BLE001 - overview remains useful
            self._registration_error = str(exc)
            self._calibration_error = self._registration_error

    @property
    def registration_error(self) -> str:
        with self._lock:
            return self._registration_error

    def set_raw_wafer_overlay(self, enabled: bool) -> None:
        """Display only; never changes detections or movement gates."""
        with self._lock:
            self.analyzer.show_raw_wafer_geometry = bool(enabled)

    def reset_registration(self) -> None:
        with self._lock:
            self._registration_samples.clear()
            self._registration_candidate = None
            self._registration_error = self._calibration_error
            self.session.clear_registration()
            self.session.clear_overview_history("runtime registration reset")
            self.analyzer.reset_observation_geometry()

    def invalidate_camera1(self, reason: str) -> None:
        """Invalidate every coordinate-bearing value after a stale/bad stream."""

        with self._lock:
            message = str(reason)
            self._registration_samples.clear()
            self._registration_candidate = None
            self._registration_error = message
            self._last_result = None
            self._last_frame = None
            self._stream_epoch += 1
            self.tracker.reset()
            self.analyzer.reset_observation_geometry()
            self.session.clear_registration()
            self.session.invalidate_overview(message)

    @staticmethod
    def _robot_state_for_frame(
        robot_state: Optional[Mapping[str, Any]],
        captured_monotonic_s: float,
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        if not isinstance(robot_state, Mapping):
            return None, "robot state unavailable"
        try:
            joints = np.asarray(robot_state.get("joints"), dtype=np.float64).reshape(-1)
            pose = np.asarray(robot_state.get("pose"), dtype=np.float64).reshape(-1)
            state_time = float(robot_state.get("captured_monotonic_s", math.nan))
        except (TypeError, ValueError, OverflowError):
            return None, "robot state fields are invalid"
        if (
            joints.size != 4
            or pose.size != 6
            or not np.all(np.isfinite(joints))
            or not np.all(np.isfinite(pose))
            or not math.isfinite(state_time)
        ):
            return None, "robot state fields are invalid"
        skew = abs(state_time - float(captured_monotonic_s))
        if skew > 0.35:
            return None, f"camera/robot timestamp skew is {skew:.3f}s"
        return (
            {
                "captured_monotonic_s": state_time,
                "joints": joints.astype(float).tolist(),
                "pose": pose.astype(float).tolist(),
            },
            None,
        )

    def _registration_sample(
        self,
        result: TrayVisionResult,
        *,
        frame_sequence: int,
        captured_monotonic_s: float,
        robot_state: Optional[Mapping[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if (
            self._suction is None
            or self._handeye is None
            or not result.quality_passed
            or result.pose.T_C_T is None
            or robot_state is None
        ):
            return None
        state_time = float(robot_state["captured_monotonic_s"])
        return {
            "measurement_id": f"camera1-sequence-{int(frame_sequence)}",
            "captured_monotonic_s": float(captured_monotonic_s),
            "accepted": True,
            "target_name": self.session.active_target_slot() or "P22",
            "robot_state_age_s": abs(float(captured_monotonic_s) - state_time),
            "used_marker_count": len(result.pose.used_marker_ids),
            "current_joints": list(robot_state["joints"]),
            "current_robot_xy_mm": list(robot_state["pose"][:2]),
            "current_pose": list(robot_state["pose"]),
            "tray_transform_C_T": result.pose.T_C_T.astype(float).tolist(),
        }

    def _update_registration(self, sample: Optional[dict[str, Any]]) -> None:
        if sample is None or self._suction is None or self._handeye is None:
            return
        # W<-T is established while the robot and overview are stable.  Once
        # the operator arms a target, keep that transform immutable: rebuilding
        # it from a moving forearm camera and a separately sampled robot state
        # creates apparent tray motion.  Current marker/pose quality is still
        # required by every XY observation; only the world target is frozen.
        if self.session.target_lock_active():
            self._registration_samples.clear()
            self._registration_error = ""
            return
        if (
            self._registration_samples
            and sample["measurement_id"]
            == self._registration_samples[-1]["measurement_id"]
        ):
            return
        self._registration_samples.append(sample)
        if len(self._registration_samples) < 5:
            return
        rows = list(self._registration_samples)
        requested = float(rows[0]["captured_monotonic_s"]) - 1e-6
        try:
            candidate = build_runtime_tray_registration(
                rows,
                self._handeye,
                self._suction,
                self.geometry,
                requested_monotonic_s=requested,
                method="live_transfer_stationary_5_frames",
            )
            self._registration_candidate = candidate
            if candidate.get("status") == "success":
                previous = self.session.registration
                if previous is not None:
                    old_origin = np.asarray(
                        previous.get("origin_world_xy_mm"), dtype=np.float64
                    ).reshape(-1)
                    new_origin = np.asarray(
                        candidate.get("origin_world_xy_mm"), dtype=np.float64
                    ).reshape(-1)
                    old_yaw = float(previous.get("yaw_world_from_tray_deg", math.nan))
                    new_yaw = float(candidate.get("yaw_world_from_tray_deg", math.nan))
                    translation_drift = (
                        float(np.linalg.norm(new_origin - old_origin))
                        if old_origin.size == 2
                        and new_origin.size == 2
                        and np.all(np.isfinite(old_origin))
                        and np.all(np.isfinite(new_origin))
                        else math.inf
                    )
                    yaw_drift = abs((new_yaw - old_yaw + 180.0) % 360.0 - 180.0)
                    active_phase = self.session.phase.value not in {
                        "idle",
                        "source_selected",
                        "source_ready",
                        "route_ready",
                    }
                    if active_phase and (translation_drift > 0.75 or yaw_drift > 0.25):
                        reason = (
                            "W<-T changed during target-locked tracking: "
                            f"translation={translation_drift:.3f} mm, "
                            f"yaw={yaw_drift:.3f} deg"
                        )
                        self._registration_error = reason
                        self.session.block(reason)
                        return
                    if not active_phase and (
                        translation_drift > 0.75 or yaw_drift > 0.25
                    ):
                        self.session.clear_overview_history(
                            "runtime tray registration changed before target lock"
                        )
                self.session.set_registration(candidate)
                self._registration_error = ""
            else:
                self._registration_error = "runtime registration did not pass: " + str(
                    candidate.get("status") or "unknown"
                )
        except Exception as exc:  # noqa: BLE001 - retain previous valid registration
            self._registration_error = str(exc)

    def process_camera1(
        self, image_bgr: np.ndarray, *, frame_sequence: int,
        captured_monotonic_s: float, robot_state: Optional[Mapping[str, Any]],
    ) -> WaferTransferFrame:
        # Serialize stream reset/anchor reset with analysis. Selection uses the
        # already displayed frame, even if a newer frame finishes first.
        with self._lock:
            return self._process_camera1_locked(
                image_bgr, frame_sequence=frame_sequence,
                captured_monotonic_s=captured_monotonic_s, robot_state=robot_state,
            )

    def _process_camera1_locked(
        self,
        image_bgr: np.ndarray,
        *,
        frame_sequence: int,
        captured_monotonic_s: float,
        robot_state: Optional[Mapping[str, Any]],
    ) -> WaferTransferFrame:
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("camera1 frame must be a valid BGR image")
        if not math.isfinite(float(captured_monotonic_s)):
            raise ValueError("camera1 timestamp must be finite")
        with self._lock:
            if self._last_frame is not None and (
                int(frame_sequence) <= self._last_frame.frame_sequence
                or float(captured_monotonic_s) < self._last_frame.captured_monotonic_s
            ):
                return self._last_frame
        tracked = self.tracker.update(image_bgr)
        result = self.analyzer.analyze(
            image_bgr, pose=tracked.raw, captured_monotonic_s=captured_monotonic_s,
        )
        if result.quality_passed and not tracked.accepted_by_tracker:
            result = replace(
                result,
                coordinate_mapping_allowed=False,
                robot_correction_allowed=False,
                failure_reason=tracked.tracker_reason
                or "temporal tracker rejected frame",
            )
        synchronized_state, state_error = self._robot_state_for_frame(
            robot_state,
            captured_monotonic_s,
        )
        with self._lock:
            sample = (
                self._registration_sample(
                    result,
                    frame_sequence=frame_sequence,
                    captured_monotonic_s=captured_monotonic_s,
                    robot_state=synchronized_state,
                )
                if tracked.accepted_by_tracker
                else None
            )
            self._update_registration(sample)
            self.session.update_overview(
                result,
                frame_sequence=frame_sequence,
                frame_captured_monotonic_s=captured_monotonic_s,
                robot_state=synchronized_state,
            )
            snapshot = self.session.snapshot()
            snapshot["registration_error"] = self._registration_error
            snapshot["robot_state_sync_error"] = state_error
            snapshot["tracker"] = {
                "accepted": tracked.accepted_by_tracker,
                "reason": tracked.tracker_reason,
                "translation_jump_mm": tracked.translation_jump_mm,
                "rotation_jump_deg": tracked.rotation_jump_deg,
                "lost_frame_count": tracked.lost_frame_count,
            }
            annotated = self._annotate(result, snapshot)
            frame = WaferTransferFrame(
                frame_sequence=int(frame_sequence),
                captured_monotonic_s=float(captured_monotonic_s),
                result=result,
                session_snapshot=snapshot,
                registration_candidate=(
                    None
                    if self._registration_candidate is None
                    else dict(self._registration_candidate)
                ),
                annotated_bgr=annotated,
                stream_epoch=self._stream_epoch,
            )
            self._last_result = result
            self._last_frame = frame
            return frame

    def _draw_tray_axes(self, canvas: np.ndarray, result: TrayVisionResult) -> None:
        if not result.quality_passed:
            return
        points_T = np.asarray(
            [[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 30.0, 0.0]],
            dtype=np.float64,
        )
        try:
            pixels = self.estimator.project_tray_points(points_T, result.pose)
        except Exception:
            return
        origin, x_axis, y_axis = [
            tuple(np.round(point).astype(int)) for point in pixels
        ]
        cv2.arrowedLine(
            canvas, origin, x_axis, (0, 0, 255), 4, cv2.LINE_AA, tipLength=0.14
        )
        cv2.arrowedLine(
            canvas, origin, y_axis, (0, 255, 0), 4, cv2.LINE_AA, tipLength=0.14
        )
        cv2.circle(canvas, origin, 8, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "T origin",
            (origin[0] + 10, origin[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "+X_T",
            (x_axis[0] + 8, x_axis[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "+Y_T",
            (y_axis[0] + 8, y_axis[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

    @staticmethod
    def _selected_analysis(result: TrayVisionResult, slot_name: Optional[str]):
        if slot_name is None:
            return None
        return next(
            (item for item in result.slots if item.projection.slot_key == slot_name),
            None,
        )

    def _annotate(
        self, result: TrayVisionResult, snapshot: Mapping[str, Any]
    ) -> np.ndarray:
        canvas = result.annotated_image.copy()
        self._draw_tray_axes(canvas, result)
        selections = (
            (snapshot.get("source_slot"), (255, 255, 0), "PICK"),
            (snapshot.get("destination_slot"), (0, 255, 255), "PLACE"),
        )
        for slot_name, color, label in selections:
            analysis = self._selected_analysis(result, slot_name)
            if analysis is None:
                continue
            polygon = np.asarray(
                analysis.projection.polygon_px, dtype=np.int32
            ).reshape(4, 2)
            cv2.polylines(canvas, [polygon], True, color, 5, cv2.LINE_AA)
            center = tuple(np.round(analysis.projection.center_px).astype(int))
            cv2.putText(
                canvas,
                f"{label} {slot_name}",
                (center[0] + 10, center[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.90,
                color,
                3,
                cv2.LINE_AA,
            )
        self._draw_suction_navigation(canvas, result, snapshot)
        lines = [
            f"frame={snapshot.get('latest_frame_sequence')}  transfer: {snapshot.get('phase', 'unknown')}",
            f"source={snapshot.get('source_slot') or '--'}  destination={snapshot.get('destination_slot') or '--'}",
        ]
        delta = snapshot.get("active_delta_world_xy_mm")
        if isinstance(delta, list) and len(delta) == 2:
            lines.append(
                f"suction move: dX={float(delta[0]):+.3f} mm  dY={float(delta[1]):+.3f} mm  distance={float(snapshot.get('active_distance_mm')):.3f} mm"
            )
        else:
            lines.append("suction move: unavailable")
        registration = snapshot.get("registration") or {}
        if registration.get("status") == "success":
            origin = registration.get("origin_world_xy_mm") or [math.nan, math.nan]
            lines.append(
                f"W<-T PASS  origin=({float(origin[0]):.3f},{float(origin[1]):.3f}) mm  yaw={float(registration.get('yaw_world_from_tray_deg')):+.3f} deg"
            )
        else:
            lines.append("W<-T: waiting for 5 stationary synchronized frames")
        top = 62 + len(lines) * 34
        shade = canvas.copy()
        cv2.rectangle(
            shade,
            (8, 56),
            (min(canvas.shape[1] - 8, 1180), top),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(shade, 0.72, canvas, 0.28, 0.0, dst=canvas)
        for index, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (20, 88 + index * 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.80,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return canvas

    def _draw_suction_navigation(
        self,
        canvas: np.ndarray,
        result: TrayVisionResult,
        snapshot: Mapping[str, Any],
    ) -> None:
        """Draw the current suction XY and locked target in the tray image."""

        registration = snapshot.get("registration")
        robot_state = snapshot.get("robot_state")
        target_slot = snapshot.get("active_target_slot")
        target = self._selected_analysis(result, target_slot)
        if (
            not isinstance(registration, Mapping)
            or not isinstance(robot_state, Mapping)
            or target is None
            or not result.quality_passed
        ):
            return
        try:
            transform_W_T = np.asarray(
                registration.get("transform_W_T"), dtype=np.float64
            ).reshape(4, 4)
            pose = np.asarray(robot_state.get("pose"), dtype=np.float64).reshape(6)
            if not np.all(np.isfinite(transform_W_T)) or not np.all(np.isfinite(pose)):
                return
            transform_T_W = np.linalg.inv(transform_W_T)
            suction_T = transform_T_W @ np.asarray(
                [float(pose[0]), float(pose[1]), 0.0, 1.0], dtype=np.float64
            )
            target_z = float(target.projection.center_T_mm[2])
            projected = self.estimator.project_tray_points(
                np.asarray([[suction_T[0], suction_T[1], target_z]], dtype=np.float64),
                result.pose,
            )[0]
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return
        suction_px = tuple(np.round(projected).astype(int))
        target_px = tuple(np.round(target.projection.center_px).astype(int))
        colour = (255, 255, 0)
        cv2.circle(canvas, suction_px, 12, colour, 4, cv2.LINE_AA)
        cv2.arrowedLine(
            canvas,
            suction_px,
            target_px,
            colour,
            4,
            cv2.LINE_AA,
            tipLength=0.08,
        )
        cv2.putText(
            canvas,
            "SUCTION XY",
            (suction_px[0] + 14, suction_px[1] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            colour,
            3,
            cv2.LINE_AA,
        )

    def select_pixel(
        self, pixel: Sequence[float], *, role: str,
        displayed_frame: Optional[WaferTransferFrame] = None,
    ) -> tuple[str, np.ndarray, float]:
        with self._lock:
            frame = displayed_frame if displayed_frame is not None else self._last_frame
            if frame is None or self._last_frame is None:
                raise RuntimeError("no analyzed camera1 frame is available")
            if frame.stream_epoch != self._stream_epoch:
                raise ValueError("displayed frame belongs to an invalidated camera stream")
            if not -0.05 <= time.monotonic() - frame.captured_monotonic_s <= 1.0:
                raise ValueError("displayed camera frame is stale; wait for a fresh image")
            result = frame.result
            if not result.success or not result.slots:
                raise ValueError("no evaluated slot geometry in the displayed frame")
            point = np.asarray(pixel, dtype=np.float64).reshape(2)
            if not np.all(np.isfinite(point)):
                raise ValueError("click coordinates must be finite")
            hits = []
            for item in result.slots:
                projection = item.projection
                polygon = np.asarray(projection.polygon_px, dtype=np.float32).reshape(4, 2)
                if cv2.pointPolygonTest(polygon, tuple(point), False) < 0:
                    continue
                target = np.asarray(projection.polygon_T_mm, dtype=np.float32).reshape(4, 3)
                mapping = cv2.getPerspectiveTransform(polygon, target[:, :2].copy())
                xy = cv2.perspectiveTransform(point.astype(np.float32).reshape(1, 1, 2), mapping).reshape(2)
                center = np.asarray(projection.center_T_mm, dtype=np.float64)
                distance = float(np.linalg.norm(xy - center[:2]))
                if math.isfinite(distance):
                    hits.append((distance, projection.slot_key, np.array([*xy, center[2]])))
            if not hits:
                raise ValueError("click inside a displayed tray slot to select it")
            distance_mm, slot_name, point_T = min(hits, key=lambda hit: (hit[0], hit[1]))
            # The image-plane mapping identifies a slot only. It cannot produce
            # world coordinates or authorize motion, even on read-only frames.
            if role == "source":
                self.session.select_source(slot_name)
            elif role == "destination":
                self.session.select_destination(slot_name)
            else:
                raise ValueError("selection role must be source or destination")
            return slot_name, point_T, float(distance_mm)

    def select_slot(self, slot_name: str, *, role: str) -> None:
        with self._lock:
            if role == "source":
                self.session.select_source(slot_name)
            elif role == "destination":
                self.session.select_destination(slot_name)
            else:
                raise ValueError("selection role must be source or destination")

    def start_tracking(self) -> None:
        with self._lock:
            self.session.start_tracking()

    def reset_selection(self) -> None:
        with self._lock:
            self.session.reset()

    def process_camera2(
        self,
        image_bgr: np.ndarray,
        *,
        measurement_id: str,
        captured_monotonic_s: float,
        robot_state: Optional[Mapping[str, Any]],
    ) -> CloseRangeSlotObservation:
        with self._lock:
            phase = self.session.phase.value
            target = self.session.active_target_slot()
            if target is None:
                return CloseRangeSlotObservation.unavailable(
                    CloseRangeOperation.PICK,
                    "",
                    "no active transfer target",
                )
            operation = (
                CloseRangeOperation.PLACE
                if phase
                in {
                    "picked",
                    "tracking_place",
                    "waiting_place_alignment",
                    "ready_to_place",
                    "verifying_place",
                }
                else CloseRangeOperation.PICK
            )
        observation = self.close_range_observer.observe(
            image_bgr,
            operation=operation,
            target_slot=target,
            measurement_id=str(measurement_id),
            captured_monotonic_s=float(captured_monotonic_s),
            robot_state=robot_state,
        )
        with self._lock:
            self.session.update_close_range(observation, robot_state=robot_state)
        return observation

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            payload = self.session.snapshot()
            payload["registration_error"] = self._registration_error
            payload["registration_candidate"] = self._registration_candidate
            payload["locked_inputs"] = {
                "camera1_intrinsics": str(self.intrinsics_path),
                "tray_geometry": str(self.geometry_path),
                "slot_marker_layout": str(self.slot_layout_path),
                "silicon_detection_config": str(
                    self.silicon_detection_config.source_path
                ),
                "silicon_detection_profile": (
                    self.silicon_detection_config.profile_name
                ),
                "silicon_detection_sha256": (
                    self.silicon_detection_config.source_sha256
                ),
            }
            return payload

    def save_report(self, path: Path) -> Path:
        output = Path(path)
        atomic_write_text(
            output,
            json.dumps(self.snapshot(), ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        return output


__all__ = ["LiveWaferTransferRuntime", "WaferTransferFrame"]
