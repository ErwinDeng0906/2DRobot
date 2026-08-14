"""Task-9 runtime for the local robot-XY/image-error Jacobian.

Task 9 owns only the bounded motion and raw camera-1 acquisition.  This module
reuses the approved Stage-3 tray-pose estimator, the latest successful Stage-4
suction target, and :func:`fit_local_xy_image_jacobian` to produce a versioned,
hash-locked Stage-5 calibration.  It never sends a robot or vacuum command.
"""

from __future__ import annotations

import hashlib
import json
import math
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QWidget

from scara.file_io import atomic_write_text, read_text_snapshot
from scara.pipeline.kinematics import rz_of
from scara.ui.dialogs import ask_light_warning_confirmation

from .tray_pose_estimator import (
    TrayBoardPoseEstimator,
    load_camera_intrinsics,
    load_tray_board_geometry,
)
from .tray_pose_tracker import TrayPoseTracker
from .xy_image_jacobian import (
    DEFAULT_XY_IMAGE_JACOBIAN_QUALITY,
    XYImageJacobianQualityConfig,
    fit_local_xy_image_jacobian,
)


RESULT_FILENAME = "camera1_xy_image_jacobian.json"
UPDATE_FILENAME = "task9_xy_image_jacobian_update.md"
ANNOTATED_DIRECTORY = "annotated_stage5"
CALIBRATION_RELATIVE_PATH = Path("src/scara/calib") / RESULT_FILENAME
SUCTION_TARGET_FILENAME = "camera1_suction_target.json"

MAXIMUM_J3_DRIFT_MM = 0.20
MAXIMUM_RZ_DRIFT_DEG = 0.20
MAXIMUM_COMMAND_OFFSET_ERROR_MM = 0.75


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(
        Path(path),
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _atomic_text_write(path: Path, text: str) -> None:
    atomic_write_text(
        Path(path),
        text.rstrip() + "\n",
        encoding="utf-8",
    )


def _finite_vector(value: Any, length: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != length or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} 必须包含 {length} 个有限数值")
    return array


def _angular_difference_deg(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(read_text_snapshot(Path(path), encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"找不到{label}：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}不是有效JSON：{path}（{exc}）") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{label}顶层必须是对象：{path}")
    return raw


def _successful_suction_report(raw: Mapping[str, Any]) -> bool:
    fit = raw.get("fit")
    return (
        raw.get("status") == "success"
        and isinstance(fit, Mapping)
        and fit.get("status") == "success"
        and fit.get("p_C_S_mm") is not None
        and fit.get("target_pixel_distorted_px") is not None
    )


def find_latest_successful_suction_target(
    project_root: Path,
    *,
    expected_intrinsics_sha256: Optional[str] = None,
    expected_geometry_sha256: Optional[str] = None,
) -> tuple[Path, dict[str, Any]]:
    """Find an installed or recent Stage-4 result matching current inputs."""

    project_root = Path(project_root)
    installed = project_root / "src/scara/calib" / SUCTION_TARGET_FILENAME
    candidates: list[Path] = []
    if installed.is_file():
        candidates.append(installed)
    trajectory_root = project_root / "Trajectory Photos"
    if trajectory_root.is_dir():
        candidates.extend(
            sorted(
                trajectory_root.glob(f"*/{SUCTION_TARGET_FILENAME}"),
                key=lambda path: path.parent.name,
                reverse=True,
            )
        )

    seen: set[Path] = set()
    rejected: list[str] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            raw = _load_json(resolved, "Stage4吸盘target")
        except ValueError as exc:
            rejected.append(str(exc))
            continue
        if not _successful_suction_report(raw):
            rejected.append(f"{resolved} 的 status/fit 不是 success")
            continue
        locked = raw.get("locked_inputs") or {}
        if expected_intrinsics_sha256 is not None and str(
            locked.get("camera_intrinsics_sha256") or ""
        ).upper() != str(expected_intrinsics_sha256).upper():
            rejected.append(f"{resolved} 的相机内参hash与当前文件不一致")
            continue
        if expected_geometry_sha256 is not None and str(
            locked.get("tray_geometry_sha256") or ""
        ).upper() != str(expected_geometry_sha256).upper():
            rejected.append(f"{resolved} 的Tray几何hash与当前文件不一致")
            continue
        return resolved, raw
    detail = "\n".join(rejected[:5])
    raise ValueError(
        "找不到成功的 camera1_suction_target.json。请先完成并审核Task8。"
        + (f"\n已检查：\n{detail}" if detail else "")
    )


def project_tray_point_distorted(
    point_T_mm: Sequence[float],
    T_C_T: Sequence[Sequence[float]],
    K: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[float, float]:
    """Project one Tray-frame point using the Stage-3 ``^C T_T`` pose."""

    point = _finite_vector(point_T_mm, 3, "Tray目标点").reshape(1, 3)
    transform = np.asarray(T_C_T, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("T_C_T 必须是有限4x4齐次变换")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    point_C = rotation @ point.reshape(3) + translation
    if point_C[2] <= 0.0:
        raise ValueError("目标槽投影深度不是正数")
    rvec, _ = cv2.Rodrigues(rotation)
    pixels, _ = cv2.projectPoints(
        point,
        rvec,
        translation.reshape(3, 1),
        np.asarray(K, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
    )
    pixel = pixels.reshape(2)
    return float(pixel[0]), float(pixel[1])


def _project_camera_point_distorted(
    point_C_mm: Sequence[float],
    K: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[float, float]:
    point = _finite_vector(point_C_mm, 3, "p_C_S_mm")
    if point[2] <= 0.0:
        raise ValueError("Stage4吸盘target在相机后方")
    pixels, _ = cv2.projectPoints(
        point.reshape(1, 3),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        np.asarray(K, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
    )
    pixel = pixels.reshape(2)
    return float(pixel[0]), float(pixel[1])


def _draw_cross(
    image: np.ndarray,
    pixel: Sequence[float],
    color: tuple[int, int, int],
    *,
    size: int = 12,
    thickness: int = 2,
) -> None:
    x = int(round(float(pixel[0])))
    y = int(round(float(pixel[1])))
    cv2.line(image, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)


class XYImageJacobianCalibrationRuntime(QObject):
    """Process Task9 frames and persist a hash-locked local Jacobian."""

    fatal_error = pyqtSignal(str)

    def __init__(
        self,
        output_dir: Path,
        project_root: Path,
        anchor_target_name: str,
        anchor_point_T_mm: Sequence[float],
        anchor_preset_joints: Sequence[float],
        command_offsets_xy_mm: Iterable[Sequence[float]],
        frames_per_offset: int,
        parent: Optional[QWidget] = None,
        *,
        quality: XYImageJacobianQualityConfig = (
            DEFAULT_XY_IMAGE_JACOBIAN_QUALITY
        ),
    ) -> None:
        super().__init__(parent)
        self.output_dir = Path(output_dir)
        self.project_root = Path(project_root)
        self.anchor_target_name = str(anchor_target_name)
        self.anchor_point_T_mm = _finite_vector(
            anchor_point_T_mm, 3, "anchor_point_T_mm"
        ).astype(float).tolist()
        self.anchor_preset_joints = _finite_vector(
            anchor_preset_joints, 4, "anchor_preset_joints"
        ).astype(float).tolist()
        self.command_offsets_xy_mm = [
            _finite_vector(value, 2, "command_offset_xy_mm").astype(float).tolist()
            for value in command_offsets_xy_mm
        ]
        self.frames_per_offset = int(frames_per_offset)
        if self.frames_per_offset < quality.minimum_frames_per_offset:
            raise ValueError(
                f"每个偏移至少需要 {quality.minimum_frames_per_offset} 帧"
            )
        if len({tuple(value) for value in self.command_offsets_xy_mm}) != 9:
            raise ValueError("Task9必须提供九个互不重复的3×3 XY偏移")
        if any(max(abs(value) for value in offset) > 2.0 + 1e-9 for offset in self.command_offsets_xy_mm):
            raise ValueError("Task9命令偏移不得超出±2mm")
        self.quality = quality

        self.intrinsics_path = (
            self.project_root / "src/scara/calib/camera1_intrinsics.json"
        )
        self.geometry_path = (
            self.project_root / "src/scara/calib/tray_board_geometry.json"
        )
        self.intrinsics_hash = _sha256(self.intrinsics_path)
        self.geometry_hash = _sha256(self.geometry_path)
        self.intrinsics = load_camera_intrinsics(self.intrinsics_path)
        self.geometry = load_tray_board_geometry(self.geometry_path)
        geometry_anchor = _finite_vector(
            self.geometry.get("slots", {}).get(self.anchor_target_name),
            3,
            f"Tray几何槽位 {self.anchor_target_name}",
        )
        if not np.allclose(geometry_anchor, self.anchor_point_T_mm, atol=1e-9):
            raise RuntimeError(
                f"Task9锚点与Tray几何不一致：脚本={self.anchor_point_T_mm}，"
                f"几何={geometry_anchor.astype(float).tolist()}"
            )

        self.suction_target_path, self.suction_target = (
            find_latest_successful_suction_target(
                self.project_root,
                expected_intrinsics_sha256=self.intrinsics_hash,
                expected_geometry_sha256=self.geometry_hash,
            )
        )
        self.suction_target_hash = _sha256(self.suction_target_path)
        locked = self.suction_target.get("locked_inputs") or {}
        expected_intrinsics = str(locked.get("camera_intrinsics_sha256") or "").upper()
        expected_geometry = str(locked.get("tray_geometry_sha256") or "").upper()
        if expected_intrinsics != self.intrinsics_hash:
            raise RuntimeError(
                "Stage4吸盘target与当前camera1内参hash不一致；请重新审核/标定。"
            )
        if expected_geometry != self.geometry_hash:
            raise RuntimeError(
                "Stage4吸盘target与当前Tray几何hash不一致；请重新审核/标定。"
            )
        fit = self.suction_target["fit"]
        self.suction_point_C_mm = _finite_vector(
            fit["p_C_S_mm"], 3, "Stage4 p_C_S_mm"
        ).astype(float).tolist()
        self.suction_target_pixel = _finite_vector(
            fit["target_pixel_distorted_px"],
            2,
            "Stage4 target_pixel_distorted_px",
        ).astype(float).tolist()
        recomputed_target = _project_camera_point_distorted(
            self.suction_point_C_mm,
            self.intrinsics.K,
            self.intrinsics.dist_coeffs,
        )
        if np.linalg.norm(
            np.asarray(recomputed_target) - np.asarray(self.suction_target_pixel)
        ) > 0.10:
            raise RuntimeError("Stage4吸盘target像素无法由锁定内参重现")

        coordinate_definition = self.suction_target.get("coordinate_definition") or {}
        self.imaging_j3_mm = float(self.anchor_preset_joints[2])
        self.anchor_rz_deg = float(
            rz_of(
                self.anchor_preset_joints[0],
                self.anchor_preset_joints[1],
                self.anchor_preset_joints[3],
            )
        )
        if abs(
            self.imaging_j3_mm
            - float(coordinate_definition.get("imaging_j3_mm", math.inf))
        ) > MAXIMUM_J3_DRIFT_MM:
            raise RuntimeError("P22 float J3与Stage4标定高度不一致")
        if _angular_difference_deg(
            self.anchor_rz_deg,
            float(coordinate_definition.get("rz_mean_deg", math.inf)),
        ) > MAXIMUM_RZ_DRIFT_DEG:
            raise RuntimeError("P22 float Rz与Stage4标定姿态不一致")

        self.estimator = TrayBoardPoseEstimator(self.geometry, self.intrinsics)
        self._tracker: Optional[TrayPoseTracker] = None
        self._active_offset: Optional[tuple[float, float]] = None
        self._anchor_robot_xy_mm: Optional[np.ndarray] = None
        self._records: list[dict[str, Any]] = []
        self._fatal_error_emitted = False
        # A photo-processing failure is permanent for this run.  In
        # particular, a failure after the numeric record was appended (for
        # example while saving its annotated image) must still prevent the
        # otherwise complete 108-record fit from being installed.
        self._processing_failed = False
        self._fatal_messages: list[str] = []

        confirmed = ask_light_warning_confirmation(
            parent,
            "Task9 XY Jacobian小幅运动确认",
            "Task9与只计算的动态演示不同：它会实际移动机械臂。\n\n"
            "开始前请确认：\n"
            "1. 机械臂当前精确位于 P22 float，速度已调低，物理急停可用；\n"
            "2. 真空已关闭，吸盘和硅片不会接触托盘；\n"
            "3. 工作区在P22周围无障碍，相机1与外围A–H固定且清晰；\n"
            "4. Task9只覆盖世界XY每轴±2mm的3×3小网格；每条2mm边已拆成"
            "不超过1mm的Cartesian中转目标；\n"
            "5. Task9不改变J3、不做旋转扫描、不下降、不控制任何DO/真空；\n"
            "6. 九点采集后会回到原始P22 float。\n\n"
            "确认以上条件后才可继续。",
        )
        if not confirmed:
            raise RuntimeError("用户取消：Task9小幅运动安全条件尚未确认")

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "points.json"

    def _load_manifest(self) -> dict[str, Any]:
        return _load_json(self.manifest_path, "Task9 points.json")

    def _photo_context(self, filename: str) -> tuple[dict[str, Any], dict[str, Any]]:
        manifest = self._load_manifest()
        photo = next(
            (
                row
                for row in manifest.get("photos", [])
                if isinstance(row, dict) and row.get("filename") == filename
            ),
            None,
        )
        if photo is None:
            raise RuntimeError(f"points.json中找不到照片 {filename}")
        point_sequence = int(photo["point_sequence"])
        point = next(
            (
                row
                for row in manifest.get("points", [])
                if int(row.get("sequence", -1)) == point_sequence
            ),
            None,
        )
        if point is None:
            raise RuntimeError(f"points.json中找不到途径点 {point_sequence}")
        return photo, point

    @staticmethod
    def parse_point_name(name: str) -> tuple[str, tuple[float, float], int, int]:
        # TASK9|target=P22|dx=-2.000|dy=+0.000|frame=01/12
        fields = str(name).split("|")
        if len(fields) != 5 or fields[0] != "TASK9":
            raise RuntimeError(f"无法解析Task9途径点名称：{name}")
        parsed: dict[str, str] = {}
        for field in fields[1:]:
            if "=" not in field:
                raise RuntimeError(f"无法解析Task9字段：{field}")
            key, value = field.split("=", 1)
            parsed[key] = value
        try:
            target = parsed["target"]
            dx_mm = float(parsed["dx"])
            dy_mm = float(parsed["dy"])
            frame_text, total_text = parsed["frame"].split("/", 1)
            frame_index = int(frame_text)
            frame_total = int(total_text)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"无法解析Task9途径点名称：{name}") from exc
        if not all(math.isfinite(value) for value in (dx_mm, dy_mm)):
            raise RuntimeError("Task9偏移包含非有限数值")
        return target, (dx_mm, dy_mm), frame_index, frame_total

    def _report_fatal(self, path: Path, exc: BaseException) -> None:
        message = f"处理Task9照片 {path.name} 失败：{exc}"
        # Mark the run before attempting any diagnostic I/O or emitting a Qt
        # signal.  Neither a log-write failure nor a late signal delivered
        # after ActionWorker has exited may make this condition recoverable.
        self._processing_failed = True
        self._fatal_messages.append(message)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "task9_runtime_error.log").write_text(
                message + "\n\n" + traceback.format_exc(),
                encoding="utf-8",
            )
        except OSError:
            pass
        if not self._fatal_error_emitted:
            self._fatal_error_emitted = True
            self.fatal_error.emit(message)

    def _annotate_record(
        self,
        image: np.ndarray,
        slot_pixel: Optional[Sequence[float]],
        image_error: Optional[Sequence[float]],
        offset_xy_mm: Sequence[float],
        accepted: bool,
    ) -> np.ndarray:
        annotated = image.copy()
        target = self.suction_target_pixel
        _draw_cross(annotated, target, (0, 0, 255), size=14, thickness=2)
        if slot_pixel is not None:
            _draw_cross(annotated, slot_pixel, (0, 220, 0), size=14, thickness=2)
            start = tuple(int(round(value)) for value in target)
            end = tuple(int(round(value)) for value in slot_pixel)
            cv2.arrowedLine(
                annotated,
                start,
                end,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
                tipLength=0.15,
            )
        error_text = (
            "n/a"
            if image_error is None
            else f"({float(image_error[0]):+.2f}, {float(image_error[1]):+.2f}) px"
        )
        cv2.putText(
            annotated,
            (
                f"Task9 cmd=({float(offset_xy_mm[0]):+.1f},"
                f"{float(offset_xy_mm[1]):+.1f})mm error={error_text} "
                f"{'ACCEPT' if accepted else 'REJECT'}"
            ),
            (18, annotated.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 0) if accepted else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    @pyqtSlot(str)
    def on_photo_saved(self, path_text: str) -> None:
        path = Path(path_text)
        if not path.name.startswith("1_"):
            return
        try:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("无法读取刚保存的相机1照片")
            photo, point = self._photo_context(path.name)
            target, offset, frame_index, frame_total = self.parse_point_name(
                point["name"]
            )
            if target != self.anchor_target_name:
                raise RuntimeError(f"未知Task9锚点 {target}")
            if frame_total != self.frames_per_offset:
                raise RuntimeError(
                    f"Task9帧总数不一致：名称={frame_total}，运行时={self.frames_per_offset}"
                )
            rounded_offset = (round(offset[0], 6), round(offset[1], 6))
            expected_offsets = {
                (round(value[0], 6), round(value[1], 6))
                for value in self.command_offsets_xy_mm
            }
            if rounded_offset not in expected_offsets:
                raise RuntimeError(f"未知Task9命令偏移 {offset}")
            if rounded_offset != self._active_offset:
                self._active_offset = rounded_offset
                self._tracker = TrayPoseTracker(self.estimator)
            assert self._tracker is not None
            tracked = self._tracker.update(image)
            stage3 = tracked.raw.to_json()

            mechanical = point.get("mechanical_center") or {}
            robot_xy = _finite_vector(
                [mechanical.get("x_mm"), mechanical.get("y_mm")],
                2,
                "实际机械臂XY",
            )
            if self._anchor_robot_xy_mm is None:
                if rounded_offset != (0.0, 0.0):
                    raise RuntimeError("Task9首个采集点必须是零偏移P22")
                self._anchor_robot_xy_mm = robot_xy.copy()
            measured_offset = robot_xy - self._anchor_robot_xy_mm
            command_error_mm = float(
                np.linalg.norm(measured_offset - np.asarray(offset, dtype=np.float64))
            )

            joints_raw = point.get("joints") or {}
            observed_j3 = float(joints_raw.get("J3_mm"))
            observed_rz = rz_of(
                float(joints_raw.get("J1_deg")),
                float(joints_raw.get("J2_deg")),
                float(joints_raw.get("J4_deg")),
            )
            j3_drift = abs(observed_j3 - self.imaging_j3_mm)
            rz_drift = _angular_difference_deg(observed_rz, self.anchor_rz_deg)

            slot_pixel: Optional[tuple[float, float]] = None
            image_error: Optional[list[float]] = None
            projection_T_C_T = (
                None
                if tracked.filtered_T_C_T is None
                else tracked.filtered_T_C_T.astype(float).tolist()
            )
            if projection_T_C_T is not None:
                slot_pixel = project_tray_point_distorted(
                    self.anchor_point_T_mm,
                    projection_T_C_T,
                    self.intrinsics.K,
                    self.intrinsics.dist_coeffs,
                )
                image_error = (
                    np.asarray(slot_pixel, dtype=np.float64)
                    - np.asarray(self.suction_target_pixel, dtype=np.float64)
                ).astype(float).tolist()

            rejection_reasons: list[str] = []
            if not bool(stage3.get("quality_passed")):
                rejection_reasons.append("stage3_quality")
            if not tracked.accepted_by_tracker:
                rejection_reasons.append("stage3_temporal")
            if image_error is None:
                rejection_reasons.append("slot_projection")
            if j3_drift > MAXIMUM_J3_DRIFT_MM:
                rejection_reasons.append("j3_drift")
            if rz_drift > MAXIMUM_RZ_DRIFT_DEG:
                rejection_reasons.append("rz_drift")
            if command_error_mm > MAXIMUM_COMMAND_OFFSET_ERROR_MM:
                rejection_reasons.append("command_tracking")
            accepted = not rejection_reasons

            temporal = {
                "accepted_by_tracker": tracked.accepted_by_tracker,
                "tracker_reason": tracked.tracker_reason,
                "translation_jump_mm": tracked.translation_jump_mm,
                "rotation_jump_deg": tracked.rotation_jump_deg,
                "lost_frame_count": tracked.lost_frame_count,
                "filtered_T_C_T": (
                    None
                    if tracked.filtered_T_C_T is None
                    else tracked.filtered_T_C_T.astype(float).tolist()
                ),
                "filtered_T_T_C": (
                    None
                    if tracked.filtered_T_T_C is None
                    else tracked.filtered_T_T_C.astype(float).tolist()
                ),
            }
            record = {
                "filename": path.name,
                "photo_sequence": int(photo["sequence_for_source"]),
                "point_sequence": int(photo["point_sequence"]),
                "anchor_target_name": target,
                "command_offset_xy_mm": [float(offset[0]), float(offset[1])],
                "measured_offset_xy_mm": measured_offset.astype(float).tolist(),
                "command_tracking_error_mm": command_error_mm,
                "frame_index": frame_index,
                "frame_total": frame_total,
                "stage3": stage3,
                "temporal_quality": temporal,
                "projection_T_C_T": projection_T_C_T,
                "slot_pixel_distorted_px": (
                    None if slot_pixel is None else [float(value) for value in slot_pixel]
                ),
                "suction_target_pixel_distorted_px": list(self.suction_target_pixel),
                "image_error_px": image_error,
                "fixed_pose_quality": {
                    "observed_j3_mm": observed_j3,
                    "j3_drift_mm": j3_drift,
                    "observed_rz_deg": observed_rz,
                    "rz_drift_deg": rz_drift,
                },
                "accepted": accepted,
                "rejection_reasons": rejection_reasons,
            }
            self._records.append(record)

            annotated = self._annotate_record(
                (
                    image
                    if tracked.raw.annotated_image is None
                    else tracked.raw.annotated_image
                ),
                slot_pixel,
                image_error,
                offset,
                accepted,
            )
            annotated_dir = self.output_dir / ANNOTATED_DIRECTORY
            annotated_dir.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(annotated_dir / path.name), annotated):
                raise RuntimeError("保存Task9标注图失败")
        except Exception as exc:  # protect the Qt event loop
            self._report_fatal(path, exc)

    def _enrich_manifest(
        self,
        manifest: dict[str, Any],
        report: Mapping[str, Any],
    ) -> None:
        records_by_point = {
            int(record["point_sequence"]): record for record in self._records
        }
        for point in manifest.get("points", []):
            record = records_by_point.get(int(point.get("sequence", -1)))
            if record is None:
                continue
            point["task9_anchor_target_name"] = record["anchor_target_name"]
            point["task9_command_offset_xy_mm"] = record[
                "command_offset_xy_mm"
            ]
            point["task9_measured_offset_xy_mm"] = record[
                "measured_offset_xy_mm"
            ]
            point["task9_command_tracking_error_mm"] = record[
                "command_tracking_error_mm"
            ]
            point["task9_frame_index"] = record["frame_index"]
            point["photo_filename"] = record["filename"]
            point["stage3_pose"] = record["stage3"]
            point["stage3_temporal_quality"] = record["temporal_quality"]
            point["stage5_projection_T_C_T"] = record["projection_T_C_T"]
            point["slot_pixel_distorted_px"] = record[
                "slot_pixel_distorted_px"
            ]
            point["suction_target_pixel_distorted_px"] = record[
                "suction_target_pixel_distorted_px"
            ]
            point["image_error_px"] = record["image_error_px"]
            point["stage5_sample_accepted"] = record["accepted"]
            point["stage5_rejection_reasons"] = record["rejection_reasons"]
        manifest["stage5_xy_image_jacobian"] = {
            "status": report["status"],
            "anchor_target_name": self.anchor_target_name,
            "valid_target_names": [self.anchor_target_name],
            "sample_count": len(self._records),
            "accepted_sample_count": sum(
                bool(record["accepted"]) for record in self._records
            ),
            "fit_summary": report["fit"],
            "result_file": RESULT_FILENAME,
            "update_file": UPDATE_FILENAME,
        }

    def _acquisition_summary(self) -> dict[str, Any]:
        rejection_counts: Counter[str] = Counter()
        for record in self._records:
            rejection_counts.update(record["rejection_reasons"])
        offset_rows: list[dict[str, Any]] = []
        for offset in self.command_offsets_xy_mm:
            rows = [
                record
                for record in self._records
                if np.allclose(
                    record["command_offset_xy_mm"], offset, atol=1e-9
                )
            ]
            accepted = [record for record in rows if record["accepted"]]
            offset_rows.append(
                {
                    "command_offset_xy_mm": list(offset),
                    "frame_count": len(rows),
                    "accepted_frame_count": len(accepted),
                    "maximum_command_tracking_error_mm": (
                        max(record["command_tracking_error_mm"] for record in rows)
                        if rows
                        else None
                    ),
                }
            )
        return {
            "expected_frame_count": (
                len(self.command_offsets_xy_mm) * self.frames_per_offset
            ),
            "processed_frame_count": len(self._records),
            "accepted_frame_count": sum(
                bool(record["accepted"]) for record in self._records
            ),
            "rejection_reason_counts": dict(sorted(rejection_counts.items())),
            "offsets": offset_rows,
        }

    def _base_report(
        self,
        status: str,
        message: str,
        fit: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            # Schema 2 adds the measured world-XY anchor and permanent runtime
            # processing-failure audit fields used to enforce local validity.
            "schema_version": 2,
            "status": status,
            "calibrated_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "message": message,
            "anchor_target_name": self.anchor_target_name,
            "valid_target_names": [self.anchor_target_name],
            "camera": {
                "logical_name": "camera1_forearm_fixed",
                "source_index": 1,
                "resolution": {
                    "width": self.intrinsics.image_size[0],
                    "height": self.intrinsics.image_size[1],
                },
            },
            "locked_inputs": {
                "camera_intrinsics_path": str(self.intrinsics_path.resolve()),
                "camera_intrinsics_sha256": self.intrinsics_hash,
                "tray_geometry_path": str(self.geometry_path.resolve()),
                "tray_geometry_sha256": self.geometry_hash,
                "suction_target_path": str(self.suction_target_path.resolve()),
                "suction_target_sha256": self.suction_target_hash,
            },
            "coordinate_definition": {
                "command_frame": "robot_controller_world_XY",
                "image_error": (
                    "slot_pixel_distorted - suction_target_pixel_distorted"
                ),
                "model_equation": (
                    "delta_image_error_px = J_error_px_per_command_mm "
                    "@ delta_command_xy_mm"
                ),
                "correction_equation": (
                    "delta_command_xy_mm = -inverse(J) @ image_error_px"
                ),
                "anchor_point_T_mm": list(self.anchor_point_T_mm),
                "anchor_robot_xy_mm": (
                    None
                    if self._anchor_robot_xy_mm is None
                    else self._anchor_robot_xy_mm.astype(float).tolist()
                ),
                "suction_point_C_mm": list(self.suction_point_C_mm),
                "suction_target_pixel_distorted_px": list(
                    self.suction_target_pixel
                ),
                "imaging_j3_mm": self.imaging_j3_mm,
                "rz_deg": self.anchor_rz_deg,
                "offset_extent_mm": 2.0,
            },
            "safety_contract": {
                "xy_calibration_extent_per_axis_mm": 2.0,
                "cartesian_waypoint_step_maximum_mm": 1.0,
                "controller_sequential_transient_test_limit_mm": 2.0001,
                "z_motion": False,
                "rotation_scan": False,
                "vacuum_control": False,
                "returns_to_anchor": True,
            },
            "acquisition": self._acquisition_summary(),
            "runtime_processing": {
                "failed": bool(self._processing_failed),
                "fatal_messages": list(self._fatal_messages),
            },
            "fit": dict(fit),
            "source_run_folder": str(self.output_dir.resolve()),
        }

    def _write_markdown(self, report: Mapping[str, Any]) -> None:
        acquisition = report["acquisition"]
        fit = report["fit"]
        status = str(report["status"])
        lines = [
            "# Task9 XY图像Jacobian标定检查清单",
            "",
            f"- [{'x' if status == 'success' else ' '}] 总状态：`{status}`",
            (
                f"- [{'x' if acquisition['processed_frame_count'] == acquisition['expected_frame_count'] else ' '}] "
                f"处理帧数：{acquisition['processed_frame_count']}/"
                f"{acquisition['expected_frame_count']}"
            ),
            (
                f"- [x] Stage3质量门、帧间跳变、J3/Rz漂移及命令到位误差均逐帧写入 `points.json`"
            ),
            "- [x] 九个命令偏移按偏移内中位数/MAD进行鲁棒聚合",
            "- [x] 使用仿射局部模型拟合2×2 Jacobian并执行留一偏移交叉验证",
            "- [x] 输入内参、Tray几何和Stage4吸盘target均以SHA-256锁定",
            "- [x] ±2mm采集网格的每条边使用≤1mm中转；逐轴控制瞬时FK已设2.0001mm测试门",
            "- [x] Task9无Z、无旋转扫描、无真空命令，结束回P22",
            "",
            "## 拟合结果",
            "",
            f"- `J_error_px_per_command_mm`: `{fit.get('j_error_px_per_command_mm')}`",
            f"- `J_command_mm_per_error_px`: `{fit.get('j_command_mm_per_error_px')}`",
            f"- 拟合RMS：`{fit.get('fit_rms_px')} px`",
            f"- 留一偏移RMS：`{(fit.get('cross_validation') or {}).get('rms_px')} px`",
            f"- 条件数：`{fit.get('condition_number')}`",
            "",
            "## 使用边界",
            "",
            "- 当前局部模型的锚点和有效目标仅为 `P22`。跨槽使用前必须另做验证或扩展标定。",
            "- 动态演示可只计算误差；只有经过明确确认的后续闭环任务才可下发修正命令。",
            "- 详细方程和质量门见 `docs/stage5_xy_image_jacobian.md`。",
            "",
            "## 输出",
            "",
            f"- `{RESULT_FILENAME}`：完整标定、hash和质量门",
            "- `points.json`：逐帧机械臂状态、Stage3结果、像素误差和接受原因",
            f"- `{ANNOTATED_DIRECTORY}/`：A–H/Tray轴及红绿十字/误差箭头标注图",
            f"- `{UPDATE_FILENAME}`：本检查清单",
        ]
        _atomic_text_write(self.output_dir / UPDATE_FILENAME, "\n".join(lines))

    @pyqtSlot(bool, str, str)
    def on_task_finished(self, ok: bool, message: str, output_dir_text: str) -> None:
        output_dir = Path(output_dir_text)
        if output_dir.resolve() != self.output_dir.resolve():
            raise RuntimeError("Task9运行时输出文件夹不一致")
        manifest = self._load_manifest()
        expected = len(self.command_offsets_xy_mm) * self.frames_per_offset

        samples = [
            {
                "command_offset_xy_mm": record["command_offset_xy_mm"],
                "image_error_px": record["image_error_px"],
                "accepted": record["accepted"],
            }
            for record in self._records
            if record["image_error_px"] is not None
        ]
        processing_failed = bool(self._processing_failed)
        anchor_missing = self._anchor_robot_xy_mm is None
        if processing_failed or anchor_missing:
            failure_reasons = list(self._fatal_messages)
            if processing_failed and not failure_reasons:
                failure_reasons.append("Task9 photo processing failed")
            if anchor_missing:
                failure_reasons.append(
                    "missing actual robot XY from the first zero-offset frame"
                )
            fit: dict[str, Any] = {
                "status": "failure",
                "failure_reasons": failure_reasons,
                "j_error_px_per_command_mm": None,
                "j_command_mm_per_error_px": None,
            }
            status = "failure"
        elif not ok:
            fit: dict[str, Any] = {
                "status": "failure",
                "failure_reasons": [message],
                "j_error_px_per_command_mm": None,
                "j_command_mm_per_error_px": None,
            }
            status = "acquisition_stopped"
        elif len(self._records) != expected:
            fit = {
                "status": "failure",
                "failure_reasons": [
                    f"processed {len(self._records)}/{expected} expected photos"
                ],
                "j_error_px_per_command_mm": None,
                "j_command_mm_per_error_px": None,
            }
            status = "failure"
        else:
            fit = fit_local_xy_image_jacobian(samples, self.quality)
            status = "success" if fit.get("status") == "success" else "failure"

        report = self._base_report(status, message, fit)
        self._enrich_manifest(manifest, report)
        _atomic_json_write(self.manifest_path, manifest)
        _atomic_json_write(output_dir / RESULT_FILENAME, report)
        self._write_markdown(report)

        if status == "success":
            _atomic_json_write(
                self.project_root / CALIBRATION_RELATIVE_PATH,
                report,
            )
        elif ok:
            reasons = ", ".join(str(value) for value in fit.get("failure_reasons", []))
            raise RuntimeError(
                "Task9采集完成但XY图像Jacobian质量门未通过；"
                f"原因：{reasons or '请检查结果JSON'}"
            )


def create_camera1_xy_image_jacobian_runtime(
    output_dir: Path,
    project_root: Path,
    anchor_target_name: str,
    anchor_point_T_mm: Sequence[float],
    anchor_preset_joints: Sequence[float],
    command_offsets_xy_mm: Iterable[Sequence[float]],
    frames_per_offset: int,
    parent: Optional[QWidget] = None,
) -> XYImageJacobianCalibrationRuntime:
    return XYImageJacobianCalibrationRuntime(
        output_dir=output_dir,
        project_root=project_root,
        anchor_target_name=anchor_target_name,
        anchor_point_T_mm=anchor_point_T_mm,
        anchor_preset_joints=anchor_preset_joints,
        command_offsets_xy_mm=command_offsets_xy_mm,
        frames_per_offset=frames_per_offset,
        parent=parent,
    )


__all__ = [
    "ANNOTATED_DIRECTORY",
    "CALIBRATION_RELATIVE_PATH",
    "RESULT_FILENAME",
    "UPDATE_FILENAME",
    "XYImageJacobianCalibrationRuntime",
    "create_camera1_xy_image_jacobian_runtime",
    "find_latest_successful_suction_target",
    "project_tray_point_distorted",
]
