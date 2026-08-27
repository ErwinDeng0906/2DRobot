"""Qt runtime adapter and JSON reporting for reusable ChArUco calibration.

The numerical CV pipeline lives in :mod:`scara.vision.charuco_calibration`.
This module connects that pipeline to the existing SCARA action-file events:

* ``on_photo_saved(path)`` reads and analyzes each newly captured JPG;
* a modeless white dialog displays image quality and acquisition guidance;
* ``on_task_finished(...)`` calibrates, rejects outliers, recalibrates, checks
  pose diversity, and saves ``camera1_intrinsics.json``;
* a fatal per-photo error is logged and emitted once so the UI can request a
  controlled robot stop.

Robot trajectories and motion commands intentionally do not belong here.
"""

from __future__ import annotations

import json
import os
import platform
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QObject, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from scara.ui.dialogs import ask_light_warning_confirmation

from .charuco_calibration import (
    CalibrationQualityConfig,
    CharucoBoardSpec,
    CharucoCalibrationSession,
    DEFAULT_BOARD_SPEC,
    DEFAULT_QUALITY_CONFIG,
)


@dataclass(frozen=True)
class CameraCalibrationRuntimeConfig:
    """Task-specific camera identity, output path, and acquisition count."""

    camera_source: int = 1
    logical_name: str = "camera1_forearm_fixed"
    mount: str = "fixed on SCARA forearm; follows J1/J2; independent of J3/J4"
    capture_backend: str = "OpenCV DirectShow"
    planned_images: Optional[int] = 49
    images_per_pose: Optional[int] = None
    operator_reposition_between_passes: bool = False
    scan_size_mm: float = 105.0
    run_filename: str = "camera1_intrinsics.json"
    project_relative_path: Path = Path("src/scara/calib/camera1_intrinsics.json")


DEFAULT_CAMERA1_RUNTIME_CONFIG = CameraCalibrationRuntimeConfig()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    """Write complete temporary JSON before atomically replacing destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class CalibrationGuideDialog(QDialog):
    """Modeless acquisition window updated after every saved calibration JPG."""

    def __init__(
        self,
        board_spec: CharucoBoardSpec,
        quality: CalibrationQualityConfig,
        runtime_config: CameraCalibrationRuntimeConfig,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.board_spec = board_spec
        self.quality = quality
        self.runtime_config = runtime_config
        self.setWindowTitle("ChArUco 相机内参引导采集")
        self.setMinimumSize(760, 650)
        self.setModal(False)
        self.setStyleSheet(
            "QDialog, QWidget { background-color:#FFFFFF; color:#111111; }"
            "QLabel, QGroupBox { color:#111111; }"
            "QGroupBox { border:1px solid #B8B8B8; border-radius:5px;"
            " margin-top:10px; padding-top:8px; font-weight:600; }"
            "QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }"
            "QProgressBar { color:#111111; background:#F3F4F6;"
            " border:1px solid #9CA3AF; border-radius:4px; text-align:center; }"
            "QProgressBar::chunk { background:#60A5FA; }"
            "QPushButton { color:#111111; background:#F3F4F6;"
            " border:1px solid #9CA3AF; border-radius:4px; padding:6px 16px; }"
            "QPushButton:hover { background:#E5E7EB; }"
        )

        layout = QVBoxLayout(self)
        if runtime_config.operator_reposition_between_passes:
            intro_text = (
                "每个九点扫描期间，标定板必须牢固固定且严禁进入机械臂工作区。"
                "只有机械臂返回X并出现人工确认窗口后，才可以改变标定板姿态。"
                "结束采集时会使用本次任务的全部姿态统一计算内参。"
            )
        else:
            intro_text = (
                "标定板应在运行前牢固固定为约15–30°倾斜；运行中严禁用手触碰"
                "标定板或进入机械臂工作区。若最终姿态多样性不足，程序不会覆盖"
                "正式相机参数。"
            )
        intro = QLabel(intro_text)
        intro.setWordWrap(True)
        intro.setStyleSheet(
            "font-weight:700; color:#111111; background:#FFF7D6;"
            " border:1px solid #D1B95C; padding:8px;"
        )
        layout.addWidget(intro)

        self.preview = QLabel("等待第一张照片…")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(330)
        self.preview.setStyleSheet(
            "background:#FFFFFF; color:#111111; border:1px solid #9CA3AF;"
            " border-radius:6px;"
        )
        layout.addWidget(self.preview, 1)

        group = QGroupBox("实时采集质量")
        grid = QGridLayout(group)
        labels = [
            ("已采集", "captured"),
            ("角点/Marker", "detected"),
            ("当前板面积", "area"),
            ("累计覆盖区域", "coverage"),
            ("当前倾斜角", "tilt"),
            ("图像清晰度", "sharpness"),
            ("是否建议保留", "recommendation"),
        ]
        self.values: dict[str, QLabel] = {}
        for row, (title, key) in enumerate(labels):
            title_label = QLabel(title)
            value_label = QLabel("—")
            value_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            grid.addWidget(title_label, row, 0)
            grid.addWidget(value_label, row, 1)
            self.values[key] = value_label
        layout.addWidget(group)

        self.coverage_bar = QProgressBar()
        self.coverage_bar.setRange(0, 100)
        self.coverage_bar.setValue(0)
        self.coverage_bar.setFormat("图像分区累计覆盖 %p%")
        layout.addWidget(self.coverage_bar)

        self.status = QLabel("任务准备完成，等待机械臂开始。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.close_button = QPushButton("隐藏窗口")
        self.close_button.clicked.connect(self.hide)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

    def set_preview(self, bgr: np.ndarray) -> None:
        """Copy one OpenCV BGR image into a safely owned Qt pixmap."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(pixmap)

    def update_record(
        self, record: dict[str, Any], coverage_fraction: float
    ) -> None:
        """Refresh all quality fields from one core-session record."""
        captured = int(record["capture_sequence"])
        images_per_pose = self.runtime_config.images_per_pose
        if images_per_pose is not None and images_per_pose > 0:
            pose_number = (captured - 1) // images_per_pose + 1
            within_pose = (captured - 1) % images_per_pose + 1
            captured_text = (
                f"累计 {captured} 张；姿态 {pose_number}："
                f"{within_pose}/{images_per_pose}"
            )
        elif self.runtime_config.planned_images is not None:
            captured_text = f"{captured}/{self.runtime_config.planned_images} 张"
        else:
            captured_text = f"累计 {captured} 张"
        self.values["captured"].setText(captured_text)
        marker_total = (self.board_spec.squares_x * self.board_spec.squares_y) // 2
        self.values["detected"].setText(
            f"{record['charuco_corner_count']}/{self.board_spec.corner_count} 角点；"
            f"{record['marker_count']}/{marker_total} markers"
        )
        self.values["area"].setText(
            f"{record['board_area_ratio'] * 100.0:.1f}% 图像面积"
        )
        covered_cells = round(
            coverage_fraction * self.quality.coverage_cell_count
        )
        self.values["coverage"].setText(
            f"{coverage_fraction * 100.0:.1f}% "
            f"({covered_cells}/{self.quality.coverage_cell_count} 分区)"
        )
        tilt = record.get("tilt_deg")
        if tilt is None:
            self.values["tilt"].setText("无法估计")
        else:
            tilt_ok = bool(record["tilt_enough"])
            self.values["tilt"].setText(
                f"约 {tilt:.1f}° · "
                f"{'足够' if tilt_ok else '不足，请倾斜标定板'}"
            )
        sharpness_ok = bool(record["sharpness_enough"])
        self.values["sharpness"].setText(
            f"{record['sharpness']:.1f} · "
            f"{'合格' if sharpness_ok else '模糊'} "
            f"(阈值 {self.quality.min_sharpness:.0f})"
        )
        if record["recommended"]:
            recommendation = "建议保留"
            color = "#047857"
        else:
            recommendation = "不建议保留：" + "；".join(record["reasons"])
            color = "#b91c1c"
        self.values["recommendation"].setText(recommendation)
        self.values["recommendation"].setStyleSheet(
            f"font-weight:700; color:{color};"
        )
        self.coverage_bar.setValue(round(coverage_fraction * 100.0))
        self.status.setText(f"最近处理：{record['filename']}")


class CharucoCalibrationRuntime(QObject):
    """Connect a reusable calibration session to SCARA action callbacks."""

    fatal_error = pyqtSignal(str)

    def __init__(
        self,
        output_dir: Path,
        project_root: Path,
        parent: Optional[QWidget] = None,
        *,
        board_spec: CharucoBoardSpec = DEFAULT_BOARD_SPEC,
        quality: CalibrationQualityConfig = DEFAULT_QUALITY_CONFIG,
        runtime_config: CameraCalibrationRuntimeConfig = (
            DEFAULT_CAMERA1_RUNTIME_CONFIG
        ),
    ) -> None:
        super().__init__(parent)
        if runtime_config.operator_reposition_between_passes:
            safety_text = (
                "开始前请确认：\n\n"
                "1. 第一个ChArUco姿态已牢固固定；\n"
                f"2. 相机{runtime_config.camera_source}在整个"
                f"{runtime_config.scan_size_mm:.0f}×"
                f"{runtime_config.scan_size_mm:.0f}mm范围内能看到足够的标定板；\n"
                "3. 每轮九点扫描期间，人员不会触碰标定板或进入工作区；\n"
                "4. 只有机械臂返回X且出现“继续采集/结束采集”窗口后，"
                "才会改变板姿态；\n"
                "5. 抬高或倾斜后的标定板与吸盘、转接头有足够间隙；\n"
                "6. 物理急停可用。\n\n"
                "是否继续？"
            )
        else:
            safety_text = (
                "开始前请确认：\n\n"
                "1. ChArUco板已牢固固定，推荐相对托盘倾斜15–30°；\n"
                f"2. 相机{runtime_config.camera_source}在整个"
                f"{runtime_config.scan_size_mm:.0f}×"
                f"{runtime_config.scan_size_mm:.0f}mm范围内能看到足够的标定板；\n"
                "3. 运行中不会用手触碰标定板或进入机械臂工作区；\n"
                "4. 物理急停可用。\n\n"
                "是否继续？"
            )
        confirmed = ask_light_warning_confirmation(
            parent,
            "ChArUco 标定板安全确认",
            safety_text,
        )
        if not confirmed:
            raise RuntimeError("用户取消：标定板安全准备尚未确认")

        self.output_dir = Path(output_dir)
        self.project_root = Path(project_root)
        self.runtime_config = runtime_config
        self.session = CharucoCalibrationSession(board_spec, quality)
        self._fatal_error_emitted = False
        self.dialog = CalibrationGuideDialog(
            board_spec,
            quality,
            runtime_config,
            parent,
        )
        self.dialog.show()
        self.dialog.raise_()

    def _report_fatal_photo_error(self, path: Path, exc: BaseException) -> None:
        """Persist traceback and request a controlled action stop exactly once."""
        message = f"处理照片 {path.name} 失败：{exc}"
        trace = traceback.format_exc()
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "charuco_runtime_error.log").write_text(
                message + "\n\n" + trace,
                encoding="utf-8",
            )
        except OSError:
            pass
        self.dialog.status.setText(
            message + "；已请求安全停止，详情见charuco_runtime_error.log"
        )
        self.dialog.show()
        self.dialog.raise_()
        if not self._fatal_error_emitted:
            self._fatal_error_emitted = True
            self.fatal_error.emit(message)

    @staticmethod
    def _read_manifest_photo_metadata(
        output_dir: Path, filename: str
    ) -> dict[str, Any]:
        """Return the matching points.json photo row when already available."""
        manifest_path = output_dir / "points.json"
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8-sig")
            )
        except (OSError, json.JSONDecodeError):
            return {}
        for photo in manifest.get("photos", []):
            if isinstance(photo, dict) and photo.get("filename") == filename:
                return {
                    "point_sequence": photo.get("point_sequence"),
                    "captured_at": photo.get("captured_at"),
                    "collection_round": photo.get("collection_round"),
                }
        return {}

    @pyqtSlot(str)
    def on_photo_saved(self, path_text: str) -> None:
        """Display raw image first, then run reusable per-image analysis."""
        path = Path(path_text)
        try:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("无法读取刚保存的照片")
            self.dialog.set_preview(image)
            self.dialog.status.setText(f"正在分析：{path.name}")

            analyzed = self.session.analyze_image(
                image,
                path.name,
                self._read_manifest_photo_metadata(self.output_dir, path.name),
            )
            self.dialog.set_preview(analyzed.annotated_image)
            self.dialog.update_record(
                analyzed.record,
                analyzed.coverage_fraction,
            )
            if analyzed.overlay_warnings:
                self.dialog.status.setText(
                    f"最近处理：{path.name}；覆盖层已降级："
                    + "；".join(analyzed.overlay_warnings)
                )
        except Exception as exc:  # protect the Qt event loop from slot errors
            self._report_fatal_photo_error(path, exc)

    def _camera_identity(self, resolution: tuple[int, int]) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        manifest_path = self.output_dir / "points.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            camera_sources = manifest.get("camera_sources_resolved") or {}
            candidate = camera_sources.get(str(self.runtime_config.camera_source))
            if isinstance(candidate, dict):
                resolved = dict(candidate)
        except (OSError, ValueError, TypeError):
            resolved = {}
        payload = {
            "logical_name": self.runtime_config.logical_name,
            "source_index": self.runtime_config.camera_source,
            "capture_backend": self.runtime_config.capture_backend,
            "mount": self.runtime_config.mount,
            "resolution": {
                "width": resolution[0],
                "height": resolution[1],
            },
            "identity_note": (
                "source_index is the stable logical camera number; the "
                "machine-local DirectShow index and USB identity come from "
                "the pre-motion points.json camera identity check."
            ),
        }
        if resolved:
            payload.update(
                {
                    "physical_source_index": resolved.get(
                        "physical_source_index"
                    ),
                    "configured_physical_index": resolved.get(
                        "configured_physical_index"
                    ),
                    "configured_index_stale": resolved.get(
                        "configured_index_stale"
                    ),
                    "usb_identity": resolved.get("camera_identity"),
                }
            )
        return payload

    def _base_report(self, status: str) -> dict[str, Any]:
        quality = self.session.quality
        return {
            "schema_version": 1,
            "status": status,
            "calibration_date": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "software": {
                "python": platform.python_version(),
                "opencv": cv2.__version__,
                "algorithm": (
                    "OpenCV calibrateCameraExtended global Levenberg-Marquardt "
                    "minimization of squared ChArUco reprojection error"
                ),
            },
            "board": self.session.board_spec.to_json(),
            "acquisition": {
                "camera_source": self.runtime_config.camera_source,
                "planned_images": self.runtime_config.planned_images,
                "images_per_pose": self.runtime_config.images_per_pose,
                "captured_images": len(self.session.records),
                "recommended_images": self.session.recommended_image_count,
                "coverage_grid": {
                    "columns": quality.coverage_columns,
                    "rows": quality.coverage_rows,
                },
                "covered_cells": len(self.session.coverage_cells),
                "coverage_fraction": self.session.coverage_fraction,
            },
            "per_image": self.session.records,
        }

    @pyqtSlot(bool, str, str)
    def on_task_finished(
        self, ok: bool, message: str, output_dir_text: str
    ) -> None:
        """Calibrate and save reports after acquisition worker completion."""
        self.dialog.show()
        self.dialog.raise_()
        run_path = Path(output_dir_text) / self.runtime_config.run_filename
        if not ok:
            report = self._base_report("acquisition_stopped")
            report["failure_reason"] = message
            _atomic_json_write(run_path, report)
            self.dialog.status.setText(
                f"采集未完成，未更新项目相机参数。诊断已保存：{run_path}"
            )
            return

        resolutions = {
            (
                int(record["resolution"]["width"]),
                int(record["resolution"]["height"]),
            )
            for record in self.session.records
        }
        if len(resolutions) != 1:
            report = self._base_report("calibration_failed")
            report["failure_reason"] = "采集图片分辨率不一致"
            _atomic_json_write(run_path, report)
            raise RuntimeError("采集图片分辨率不一致，已保存失败报告。")
        if not resolutions:
            raise RuntimeError("没有可用于标定的照片")
        resolution = next(iter(resolutions))

        try:
            result, rejection_history = (
                self.session.calibrate_with_outlier_rejection(resolution)
            )
        except Exception as exc:
            report = self._base_report("calibration_failed")
            report["failure_reason"] = str(exc) or exc.__class__.__name__
            _atomic_json_write(run_path, report)
            self.dialog.status.setText(
                f"标定失败：{report['failure_reason']}；报告：{run_path}"
            )
            raise

        pose_diversity = result["pose_diversity"]
        status = (
            "success"
            if pose_diversity["sufficient"]
            else "rejected_pose_diversity"
        )
        quality = self.session.quality
        report = self._base_report(status)
        report.update(
            {
                "camera": self._camera_identity(resolution),
                "image_resolution": {
                    "width": resolution[0],
                    "height": resolution[1],
                },
                "K": np.asarray(result["K"], dtype=float).tolist(),
                "distCoeffs": np.asarray(
                    result["dist"], dtype=float
                ).reshape(-1).tolist(),
                "distortion_model": "OpenCV pinhole k1,k2,p1,p2,k3",
                "global_rms_px": float(result["rms"]),
                "intrinsic_standard_deviations": np.asarray(
                    result["std_intrinsics"], dtype=float
                ).reshape(-1).tolist(),
                "used_image_count": sum(
                    bool(record["used_for_final_calibration"])
                    for record in self.session.records
                ),
                "outlier_rejection": {
                    "method": (
                        "iterative per-view RMS > max("
                        f"{quality.minimum_view_error_threshold_px}px, median + "
                        f"{quality.outlier_sigma_multiplier}*1.4826*MAD)"
                    ),
                    "history": rejection_history,
                },
                "pose_diversity": pose_diversity,
                "source_run_folder": str(Path(output_dir_text).resolve()),
            }
        )
        if not pose_diversity["sufficient"]:
            report["warning"] = (
                "标定板法向变化不足质量门限；参数已保存供检查，但不应进入"
                "精密定位。请增加多方向俯仰/侧倾后重新采集。"
            )
        _atomic_json_write(run_path, report)

        project_path = (
            self.project_root / self.runtime_config.project_relative_path
        )
        if pose_diversity["sufficient"]:
            _atomic_json_write(project_path, report)

        warning = ""
        if not pose_diversity["sufficient"]:
            warning = "；质量门未通过：倾角多样性不足，未更新项目相机参数"
        self.dialog.status.setText(
            f"标定完成：RMS={result['rms']:.4f}px，"
            f"使用{report['used_image_count']}张{warning}\n"
            f"运行结果：{run_path}\n"
            + (
                f"项目参数：{project_path}"
                if pose_diversity["sufficient"]
                else "项目参数：未更新（请增加多方向倾斜后重试）"
            )
        )


def create_camera1_charuco_runtime(
    output_dir: Path,
    project_root: Path,
    parent: Optional[QWidget] = None,
    *,
    planned_images: Optional[int] = 49,
    scan_size_mm: float = 105.0,
    images_per_pose: Optional[int] = None,
    operator_reposition_between_passes: bool = False,
    board_spec: CharucoBoardSpec = DEFAULT_BOARD_SPEC,
    quality: CalibrationQualityConfig = DEFAULT_QUALITY_CONFIG,
) -> CharucoCalibrationRuntime:
    """Build the standard camera-1 runtime used by an action task."""
    runtime_config = CameraCalibrationRuntimeConfig(
        planned_images=planned_images,
        images_per_pose=images_per_pose,
        operator_reposition_between_passes=(
            operator_reposition_between_passes
        ),
        scan_size_mm=scan_size_mm,
    )
    return CharucoCalibrationRuntime(
        Path(output_dir),
        Path(project_root),
        parent,
        board_spec=board_spec,
        quality=quality,
        runtime_config=runtime_config,
    )


__all__ = [
    "CalibrationGuideDialog",
    "CameraCalibrationRuntimeConfig",
    "CharucoCalibrationRuntime",
    "DEFAULT_CAMERA1_RUNTIME_CONFIG",
    "create_camera1_charuco_runtime",
]
