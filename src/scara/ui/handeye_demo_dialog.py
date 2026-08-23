"""Read-only live hand-eye demonstration for camera 1."""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import cv2
from PyQt6.QtCore import QSize, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCloseEvent, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QGraphicsScene,
    QGraphicsView,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from scara.ui.dialogs import LIGHT_WARNING_DIALOG_STYLESHEET
from scara.ui.camera_view import (
    DIRECTSHOW_EXPOSURE_DEFAULT,
    DIRECTSHOW_EXPOSURE_MAX,
    DIRECTSHOW_EXPOSURE_MIN,
)
from scara.vision.handeye_interaction import (
    HandEyeEvaluation,
    ROBOT_STATE_MAXIMUM_AGE_S,
    SuctionTargetModel,
    evaluate_handeye_frame,
    load_latest_suction_target,
    load_local_xy_jacobian,
)
from scara.vision.tray_pose_estimator import (
    load_camera_intrinsics,
    load_tray_board_geometry,
)
from scara.vision.tray_pose_tracker import TrayPoseTracker
from scara.vision.slot_marker_observation import load_slot_marker_layout
from scara.vision.silicon_detection_config import (
    LoadedSiliconDetectionConfig,
    default_silicon_detection_config_path,
    load_silicon_detection_config,
    preferred_silicon_detection_config_path,
    save_preferred_silicon_detection_config_path,
)
from scara.vision.tray_vision_fusion import (
    DEFAULT_TRAY_VISION_FUSION,
    TrayVisionAnalyzer,
    TrayVisionFusionConfig,
    TrayVisionResult,
)
from scara.vision.xy_image_jacobian import REQUIRED_XY_JACOBIAN_QUALITY_GATES


_DIALOG_STYLE = """
QDialog { background-color:#FFFFFF; color:#111827; }
QScrollArea#dialogScroll { background-color:#FFFFFF; border:0; }
QWidget#dialogContent { background-color:#FFFFFF; }
QLabel { color:#111827; }
QLabel#safetyBanner {
    color:#991B1B; background-color:#FEE2E2;
    border:1px solid #FCA5A5; border-radius:6px;
    padding:9px; font-size:14px; font-weight:800;
}
QGraphicsView#preview { background-color:#0B1220; border:1px solid #CBD5E1; }
QLabel#status {
    color:#111827; background-color:#F8FAFC;
    border:1px solid #CBD5E1; border-radius:6px; padding:8px;
    font-family:Consolas, "Microsoft YaHei";
}
QPlainTextEdit#stage7bStatus {
    color:#111827; background-color:#FFF7ED;
    border:1px solid #FDBA74; border-radius:6px; padding:7px;
    font-family:Consolas, "Microsoft YaHei";
}
QLabel#traySummary {
    color:#0F172A; background-color:#EFF6FF;
    border:1px solid #BFDBFE; border-radius:5px; padding:6px 8px;
    font-weight:700;
}
QTableWidget#slotTable {
    color:#111827; background-color:#FFFFFF; alternate-background-color:#F8FAFC;
    border:1px solid #CBD5E1; gridline-color:#E2E8F0;
    selection-background-color:#DBEAFE; selection-color:#111827;
}
QTableWidget#slotTable QHeaderView::section {
    color:#0F172A; background-color:#E2E8F0;
    border:0; border-right:1px solid #CBD5E1; border-bottom:1px solid #CBD5E1;
    padding:5px; font-weight:700;
}
QComboBox {
    color:#111827; background-color:#FFFFFF;
    border:1px solid #94A3B8; border-radius:4px; padding:5px 8px;
    min-width:120px;
}
QPushButton {
    color:#111827; background-color:#F3F4F6;
    border:1px solid #94A3B8; border-radius:5px;
    padding:7px 16px; min-width:105px;
}
QPushButton:hover { background-color:#E2E8F0; }
QPushButton#stage7bButton {
    color:#FFFFFF; background-color:#B45309; border-color:#B45309;
}
QPushButton#stage7bButton:hover { background-color:#92400E; }
QPushButton#localJacobianButton {
    color:#FFFFFF; background-color:#2563EB; border-color:#2563EB;
}
QPushButton#localJacobianButton:hover { background-color:#1D4ED8; }
QPushButton#fullTrayButton {
    color:#FFFFFF; background-color:#047857; border-color:#047857;
}
QPushButton#fullTrayButton:hover { background-color:#065F46; }
"""


class CameraImageView(QGraphicsView):
    """Aspect-ratio preview with fit-relative zoom and mouse panning."""

    zoom_changed = pyqtSignal(float)

    _DEFAULT_ASPECT_RATIO = 720.0 / 1280.0
    _MIN_ZOOM = 1.0
    _MAX_ZOOM = 8.0
    _ZOOM_STEP = 1.25

    def __init__(self, placeholder: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        scene = QGraphicsScene(self)
        self.setScene(scene)
        self._pixmap_item = scene.addPixmap(QPixmap())
        self._has_image = False
        self._zoom_factor = self._MIN_ZOOM
        self._aspect_ratio = self._DEFAULT_ASPECT_RATIO

        self._placeholder = QLabel(placeholder, self.viewport())
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self._placeholder.setStyleSheet(
            "color:#E2E8F0; background:transparent; border:0; padding:12px;"
        )

        self.setObjectName("preview")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setToolTip(
            "使用放大/缩小按钮或 Ctrl+滚轮缩放；放大后按住鼠标左键拖动画面"
        )
        size_policy = QSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        size_policy.setHeightForWidth(True)
        self.setSizePolicy(size_policy)
        self.setMinimumSize(480, 270)

    @property
    def has_image(self) -> bool:
        return self._has_image

    @property
    def zoom_factor(self) -> float:
        return self._zoom_factor

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        frame = 2 * self.frameWidth()
        image_width = max(1, int(width) - frame)
        return max(270, int(round(image_width * self._aspect_ratio)) + frame)

    def sizeHint(self) -> QSize:  # noqa: N802
        width = 1180
        return QSize(width, self.heightForWidth(width))

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(480, self.heightForWidth(480))

    def set_image(self, image: QImage) -> None:
        """Replace the live frame without resetting an active pan/zoom."""
        if image.isNull():
            self.clear_image("当前图像为空")
            return
        pixmap = QPixmap.fromImage(image)
        old_size = self._pixmap_item.pixmap().size()
        dimensions_changed = not self._has_image or old_size != pixmap.size()
        self._pixmap_item.setPixmap(pixmap)
        self.scene().setSceneRect(self._pixmap_item.boundingRect())
        self._has_image = True
        self._placeholder.hide()

        new_aspect_ratio = pixmap.height() / max(1, pixmap.width())
        if not math.isclose(new_aspect_ratio, self._aspect_ratio, abs_tol=1e-6):
            self._aspect_ratio = new_aspect_ratio
            self.updateGeometry()
        if dimensions_changed:
            self._zoom_factor = self._MIN_ZOOM
            self._apply_view_transform()
            self.zoom_changed.emit(self._zoom_factor)

    def clear_image(self, message: str) -> None:
        self._pixmap_item.setPixmap(QPixmap())
        self.scene().setSceneRect(0.0, 0.0, 0.0, 0.0)
        self._has_image = False
        self._zoom_factor = self._MIN_ZOOM
        self.resetTransform()
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._placeholder.setText(message)
        self._placeholder.show()
        self.zoom_changed.emit(self._zoom_factor)

    def zoom_in(self) -> None:
        if not self._has_image:
            return
        self._set_zoom(min(self._MAX_ZOOM, self._zoom_factor * self._ZOOM_STEP))

    def zoom_out(self) -> None:
        if not self._has_image:
            return
        next_zoom = self._zoom_factor / self._ZOOM_STEP
        self._set_zoom(
            self._MIN_ZOOM if next_zoom < 1.001 else max(self._MIN_ZOOM, next_zoom)
        )

    def reset_zoom(self) -> None:
        if self._has_image:
            self._set_zoom(self._MIN_ZOOM)

    def _set_zoom(self, zoom_factor: float) -> None:
        old_center = self.mapToScene(self.viewport().rect().center())
        self._zoom_factor = max(
            self._MIN_ZOOM, min(self._MAX_ZOOM, float(zoom_factor))
        )
        self._apply_view_transform(
            old_center if self._zoom_factor > self._MIN_ZOOM else None
        )
        self.zoom_changed.emit(self._zoom_factor)

    def _apply_view_transform(self, center=None) -> None:
        if not self._has_image:
            return
        image_rect = self._pixmap_item.boundingRect()
        self.resetTransform()
        self.fitInView(image_rect, Qt.AspectRatioMode.KeepAspectRatio)
        self.scale(self._zoom_factor, self._zoom_factor)
        if center is None:
            self.centerOn(image_rect.center())
        else:
            self.centerOn(center)
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag
            if self._zoom_factor > self._MIN_ZOOM
            else QGraphicsView.DragMode.NoDrag
        )

    def resizeEvent(self, event) -> None:  # noqa: N802
        old_center = (
            self.mapToScene(self.viewport().rect().center())
            if self._has_image and self._zoom_factor > self._MIN_ZOOM
            else None
        )
        super().resizeEvent(event)
        self._placeholder.setGeometry(self.viewport().rect())
        self._apply_view_transform(old_center)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self._has_image and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            elif event.angleDelta().y() < 0:
                self.zoom_out()
            event.accept()
            return
        if self._zoom_factor > self._MIN_ZOOM:
            super().wheelEvent(event)
        else:
            event.ignore()


class HandEyeMonitorThread(QThread):
    """Run Stage-3 and overlay calculations without blocking the Qt UI."""

    frame_evaluated = pyqtSignal(QImage, object, object)
    monitor_error = pyqtSignal(str)
    frame_invalidated = pyqtSignal(str)

    def __init__(
        self,
        camera: Any,
        project_root: Path,
        suction: SuctionTargetModel,
        target_name: str,
        jacobian_payload: Optional[Mapping[str, Any]],
        robot_state_provider: Optional[
            Callable[[], Optional[Mapping[str, Any]]]
        ] = None,
        parent: Optional[QWidget] = None,
        *,
        tray_vision_config: TrayVisionFusionConfig = DEFAULT_TRAY_VISION_FUSION,
    ) -> None:
        super().__init__(parent)
        self.camera = camera
        self.project_root = Path(project_root)
        self.suction = suction
        self._target_name = target_name
        self._jacobian_payload = jacobian_payload
        self._robot_state_provider = robot_state_provider
        self._tray_vision_config = tray_vision_config
        self._tray_vision_config_version = 0
        self._state_lock = threading.Lock()
        self._running = True

    def set_target(self, target_name: str) -> None:
        with self._state_lock:
            self._target_name = str(target_name)

    def set_jacobian_payload(
        self, payload: Optional[Mapping[str, Any]]
    ) -> None:
        with self._state_lock:
            self._jacobian_payload = payload

    def set_tray_vision_config(self, config: TrayVisionFusionConfig) -> None:
        """Use a validated detector profile from the next fresh frame onward."""

        if not isinstance(config, TrayVisionFusionConfig):
            raise TypeError("config必须是TrayVisionFusionConfig")
        with self._state_lock:
            self._tray_vision_config = config
            self._tray_vision_config_version += 1

    def stop(self, timeout_ms: int = 5000) -> bool:
        self._running = False
        self.requestInterruption()
        return bool(self.wait(timeout_ms))

    def run(self) -> None:
        try:
            intrinsics = load_camera_intrinsics(
                self.project_root / "src/scara/calib/camera1_intrinsics.json"
            )
            geometry = load_tray_board_geometry(
                self.project_root / "src/scara/calib/tray_board_geometry.json"
            )
            from scara.vision.tray_pose_estimator import TrayBoardPoseEstimator

            estimator = TrayBoardPoseEstimator(geometry, intrinsics)
            tracker = TrayPoseTracker(estimator)
            slot_layout = load_slot_marker_layout(
                self.project_root
                / "tools/tray_marker_detector_v2/tray_marker_layout.json"
            )
            with self._state_lock:
                tray_config = self._tray_vision_config
                tray_config_version = self._tray_vision_config_version
            tray_analyzer = TrayVisionAnalyzer(
                estimator,
                geometry,
                slot_layout,
                config=tray_config,
            )
        except Exception as exc:  # noqa: BLE001
            self.monitor_error.emit(f"手眼计算初始化失败：{exc}")
            return

        last_error = ""
        no_frame_since = time.monotonic()
        invalidation_emitted = False
        last_frame_sequence: Optional[int] = None
        while self._running and not self.isInterruptionRequested():
            packet_reader = getattr(self.camera, "latest_frame_packet", None)
            frame_sequence: Optional[int] = None
            if callable(packet_reader):
                packet = packet_reader(max_age_s=1.0)
                if packet is None:
                    frame = None
                    captured_at = None
                else:
                    frame, frame_sequence, captured_at = packet
            else:
                # Compatibility for test doubles and older camera adapters.
                # Production camera1 provides a sequence-aware packet API.
                frame = self.camera.latest_frame(max_age_s=1.0)
                captured_at = time.monotonic() if frame is not None else None
            if frame is None:
                if not self.camera.isRunning():
                    self.frame_invalidated.emit("相机1未运行或已经断开")
                    return
                if (
                    not invalidation_emitted
                    and time.monotonic() - no_frame_since > 1.0
                ):
                    self.frame_invalidated.emit("相机1画面已超过1秒未更新")
                    invalidation_emitted = True
                self.msleep(80)
                continue

            if frame_sequence is not None and frame_sequence == last_frame_sequence:
                if (
                    not invalidation_emitted
                    and time.monotonic() - no_frame_since > 1.0
                ):
                    self.frame_invalidated.emit("相机1画面已超过1秒未更新")
                    invalidation_emitted = True
                self.msleep(80)
                continue

            last_frame_sequence = frame_sequence
            no_frame_since = time.monotonic()
            invalidation_emitted = False
            with self._state_lock:
                target_name = self._target_name
                jacobian_payload = self._jacobian_payload
                requested_tray_config = self._tray_vision_config
                requested_tray_config_version = self._tray_vision_config_version
            if requested_tray_config_version != tray_config_version:
                tray_analyzer = TrayVisionAnalyzer(
                    estimator,
                    geometry,
                    slot_layout,
                    config=requested_tray_config,
                )
                tray_config_version = requested_tray_config_version
            robot_state: Optional[Mapping[str, Any]] = None
            if self._robot_state_provider is not None:
                try:
                    robot_state = self._robot_state_provider()
                except Exception:  # noqa: BLE001 - fail closed, keep visual live
                    robot_state = None
            try:
                tracked = tracker.update(frame)
                tray_result = tray_analyzer.analyze_tracked(frame, tracked)
                evaluation = evaluate_handeye_frame(
                    frame,
                    tracked,
                    target_name,
                    geometry,
                    intrinsics,
                    self.suction,
                    jacobian_payload,
                    robot_state,
                    base_annotated_bgr=tray_result.annotated_image,
                )
                joints = None
                pose = None
                if isinstance(robot_state, Mapping):
                    raw_joints = robot_state.get("joints")
                    raw_pose = robot_state.get("pose")
                    if isinstance(raw_joints, (list, tuple)) and len(raw_joints) == 4:
                        joints = tuple(float(value) for value in raw_joints)
                    if isinstance(raw_pose, (list, tuple)) and len(raw_pose) == 6:
                        pose = tuple(float(value) for value in raw_pose)
                measurement_id = (
                    f"camera1-sequence-{frame_sequence}"
                    if frame_sequence is not None
                    else f"camera1-monotonic-{time.monotonic_ns()}"
                )
                evaluation = replace(
                    evaluation,
                    measurement_id=measurement_id,
                    frame_captured_monotonic_s=(
                        None if captured_at is None else float(captured_at)
                    ),
                    current_joints=joints,
                    current_pose=pose,
                )
                with self._state_lock:
                    config_still_current = (
                        tray_config_version == self._tray_vision_config_version
                    )
                if not config_still_current:
                    # A file was selected while this expensive frame was in
                    # flight.  Never show a result produced by the old profile.
                    continue
                rgb = cv2.cvtColor(evaluation.annotated_bgr, cv2.COLOR_BGR2RGB)
                height, width, channels = rgb.shape
                image = QImage(
                    rgb.data,
                    width,
                    height,
                    channels * width,
                    QImage.Format.Format_RGB888,
                ).copy()
                self.frame_evaluated.emit(image, evaluation, tray_result)
                last_error = ""
            except Exception as exc:  # noqa: BLE001
                message = f"实时手眼计算失败：{exc}"
                if message != last_error:
                    self.frame_invalidated.emit(message)
                    last_error = message
            # Process only the newest packet and avoid building a stale frame
            # queue.  The vision work itself limits the live update rate.
            self.msleep(20)


class HandEyeDemoDialog(QDialog):
    """Live slot-vs-suction visualization with no motion API."""

    local_jacobian_calibration_requested = pyqtSignal(str)
    stage7b_start_requested = pyqtSignal()
    stage7b_stop_requested = pyqtSignal()
    full_tray_start_requested = pyqtSignal()
    full_tray_stop_requested = pyqtSignal()

    def __init__(
        self,
        project_root: Path,
        camera: Any,
        parent: Optional[QWidget] = None,
        *,
        robot_state_provider: Optional[
            Callable[[], Optional[Mapping[str, Any]]]
        ] = None,
    ) -> None:
        super().__init__(parent)
        self.project_root = Path(project_root)
        self.camera = camera
        self.robot_state_provider = robot_state_provider
        self._silicon_config_startup_warning = ""
        try:
            preferred_config_path = preferred_silicon_detection_config_path(
                self.project_root
            )
            self.silicon_detection_config: LoadedSiliconDetectionConfig = (
                load_silicon_detection_config(preferred_config_path)
            )
        except Exception as exc:  # noqa: BLE001 - keep the read-only UI available
            default_path = default_silicon_detection_config_path(self.project_root)
            self.silicon_detection_config = load_silicon_detection_config(default_path)
            self._silicon_config_startup_warning = (
                "上次保存的硅片参数无法加载，已回退到工程默认配置：" f"{exc}"
            )
        self.suction = load_latest_suction_target(self.project_root)
        self.jacobian_payload = load_local_xy_jacobian(
            self.project_root, self.suction, "P22"
        )
        geometry = load_tray_board_geometry(
            self.project_root / "src/scara/calib/tray_board_geometry.json"
        )
        self._last_image: Optional[QImage] = None
        self._last_evaluation: Optional[HandEyeEvaluation] = None
        self._last_tray_result: Optional[TrayVisionResult] = None
        self._last_evaluation_at: Optional[float] = None
        self._stage7b_session = None
        self._stage7b_pending_request: Optional[dict[str, Any]] = None
        self._stage7b_pending_responder: Optional[Callable[[dict], None]] = None
        self._stage7b_samples: deque[dict[str, Any]] = deque(maxlen=40)
        self._stage7b_active = False
        self._positioning_mode: Optional[str] = None
        self._exposure_was_adjusted = False

        if int(camera.source_index) != self.suction.camera_source:
            raise RuntimeError(
                f"动态演示要求相机源#{self.suction.camera_source}，"
                f"当前为源#{camera.source_index}"
            )

        self.setWindowTitle("手眼交互 · 动态演示（只计算 / 两种XY定位模式需ARM）")
        self.resize(1200, 960)
        self.setMinimumSize(900, 760)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setStyleSheet(_DIALOG_STYLE)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.content_scroll = QScrollArea()
        self.content_scroll.setObjectName("dialogScroll")
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        content = QWidget()
        content.setObjectName("dialogContent")
        layout = QVBoxLayout(content)
        self.content_layout = layout
        layout.setContentsMargins(6, 12, 6, 12)
        layout.setSpacing(9)
        self.content_scroll.setWidget(content)
        outer_layout.addWidget(self.content_scroll)

        safety = QLabel(
            "默认动态演示只计算，机械臂不会移动。"
            "“local Jacobian标定”会对左侧所选槽运行Task9真实小幅XY采集；"
            "“单点有限闭环”和“全盘定位”是独立的已确认XY运动模式；"
            "当前全盘定位只授权P22，启动前必须确认左侧选取的是P22。"
            "任何模式都不会下降或操作DO/真空。"
        )
        safety.setObjectName("safetyBanner")
        safety.setWordWrap(True)
        safety.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(safety)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("目标槽："))
        self.target_combo = QComboBox()
        slot_names = sorted(
            geometry["slots"],
            key=lambda name: (int(name[1]), int(name[2])),
        )
        self.target_combo.addItems(slot_names)
        self.target_combo.setCurrentText("P22" if "P22" in slot_names else slot_names[0])
        controls.addWidget(self.target_combo)
        controls.addStretch(1)
        self.local_jacobian_button = QPushButton("local Jacobian标定")
        self.local_jacobian_button.setObjectName("localJacobianButton")
        self.local_jacobian_button.setToolTip(
            "对左侧当前目标槽运行Task9九点局部标定；任务不会先执行跨托盘移动"
        )
        controls.addWidget(self.local_jacobian_button)
        self.stage7b_button = QPushButton("单点有限闭环")
        self.stage7b_button.setObjectName("stage7bButton")
        controls.addWidget(self.stage7b_button)
        self.full_tray_button = QPushButton("全盘定位")
        self.full_tray_button.setObjectName("fullTrayButton")
        controls.addWidget(self.full_tray_button)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        controls.addWidget(close_button)
        layout.addLayout(controls)

        exposure_controls = QHBoxLayout()
        exposure_controls.addWidget(QLabel("相机1硬件曝光："))
        self.exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self.exposure_slider.setObjectName("camera1ExposureSlider")
        # DirectShow exposes this camera's shutter as integral base-2 stops.
        self.exposure_slider.setRange(
            DIRECTSHOW_EXPOSURE_MIN,
            DIRECTSHOW_EXPOSURE_MAX,
        )
        self.exposure_slider.setSingleStep(1)
        self.exposure_slider.setPageStep(1)
        self.exposure_slider.setTickInterval(1)
        self.exposure_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.exposure_slider.setValue(DIRECTSHOW_EXPOSURE_DEFAULT)
        self.exposure_slider.setToolTip(
            "直接修改DirectShow整数曝光档位（-13到-1）；越负越暗。"
            "不会对采集后的图像做软件调暗。"
        )
        exposure_controls.addWidget(self.exposure_slider, 1)
        self.exposure_value_label = QLabel(
            f"{DIRECTSHOW_EXPOSURE_DEFAULT}（尚未应用）"
        )
        self.exposure_value_label.setObjectName("camera1ExposureValue")
        self.exposure_value_label.setMinimumWidth(105)
        exposure_controls.addWidget(self.exposure_value_label)
        self.exposure_apply_button = QPushButton("应用")
        self.exposure_apply_button.setObjectName("camera1ExposureApply")
        self.exposure_apply_button.setToolTip("应用滑块当前整数曝光档位")
        exposure_controls.addWidget(self.exposure_apply_button)
        self.exposure_status_label = QLabel(
            "仅使用整数档位；每降低1档曝光时间约减半。过低可能影响Marker识别"
        )
        self.exposure_status_label.setObjectName("camera1ExposureStatus")
        self.exposure_status_label.setWordWrap(True)
        exposure_controls.addWidget(self.exposure_status_label, 1)
        self.exposure_recovery_button = QPushButton("恢复自动曝光")
        self.exposure_recovery_button.setObjectName("camera1ExposureRecovery")
        self.exposure_recovery_button.setToolTip(
            "直接让相机驱动重新进入自动曝光；用于黑屏或原曝光恢复失败。"
        )
        exposure_controls.addWidget(self.exposure_recovery_button)
        exposure_requester = getattr(self.camera, "request_exposure_value", None)
        exposure_recovery = getattr(
            self.camera, "request_auto_exposure_recovery", None
        )
        self.exposure_slider.setEnabled(callable(exposure_requester))
        self.exposure_apply_button.setEnabled(callable(exposure_requester))
        self.exposure_recovery_button.setEnabled(callable(exposure_recovery))
        if not callable(exposure_requester):
            self.exposure_status_label.setText("当前相机接口不支持硬件曝光控制")
        self.exposure_slider.valueChanged.connect(
            self._on_exposure_slider_changed
        )
        self.exposure_apply_button.clicked.connect(
            lambda: self._on_exposure_slider_changed(
                int(self.exposure_slider.value())
            )
        )
        self.exposure_recovery_button.clicked.connect(
            self._on_auto_exposure_recovery_requested
        )
        exposure_signal = getattr(self.camera, "exposure_applied", None)
        if exposure_signal is not None and hasattr(exposure_signal, "connect"):
            exposure_signal.connect(self._on_hardware_exposure_applied)
        layout.addLayout(exposure_controls)

        silicon_parameter_controls = QHBoxLayout()
        silicon_parameter_controls.addWidget(QLabel("硅片检测配置："))
        self.silicon_parameter_button = QPushButton("硅片判定参数")
        self.silicon_parameter_button.setObjectName(
            "siliconDetectionParameterButton"
        )
        self.silicon_parameter_button.setToolTip(
            "选择完整的硅片判定JSON；验证通过后立即应用并保存为后续默认配置"
        )
        silicon_parameter_controls.addWidget(self.silicon_parameter_button)
        self.silicon_parameter_label = QLabel()
        self.silicon_parameter_label.setObjectName(
            "siliconDetectionParameterStatus"
        )
        self.silicon_parameter_label.setWordWrap(True)
        silicon_parameter_controls.addWidget(self.silicon_parameter_label, 1)
        self._update_silicon_parameter_label()
        self.silicon_parameter_button.clicked.connect(
            self._on_select_silicon_detection_parameters
        )
        layout.addLayout(silicon_parameter_controls)

        legend = QLabel(
            "绿色十字＝指定槽中心　红色十字＝suction target　黄色箭头＝当前图像误差（红→绿）　"
            "青色圆点＝A–H重投影角点　T-X/T-Y/T-Z＝托盘坐标轴\n"
            "槽框状态：EMPTY空槽、OCC占用、WARN警告、STACK叠片、OUT槽外、"
            "STACK+OUT叠片且槽外、OOV画面外、UNK证据不足；"
            "白圈彩心＝硅片中心，连线表示相对槽中心偏差"
        )
        legend.setWordWrap(True)
        layout.addWidget(legend)

        preview_controls = QHBoxLayout()
        preview_controls.addWidget(QLabel("相机1动态图（保持原始宽高比）："))
        preview_controls.addStretch(1)
        self.zoom_out_button = QPushButton("缩小")
        self.zoom_out_button.setObjectName("camera1ZoomOut")
        self.zoom_out_button.setToolTip("缩小实时画面，最小为适应窗口")
        preview_controls.addWidget(self.zoom_out_button)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setObjectName("camera1ZoomValue")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setMinimumWidth(55)
        preview_controls.addWidget(self.zoom_label)
        self.zoom_in_button = QPushButton("放大")
        self.zoom_in_button.setObjectName("camera1ZoomIn")
        self.zoom_in_button.setToolTip("放大实时画面；放大后按住鼠标左键拖动")
        preview_controls.addWidget(self.zoom_in_button)
        self.zoom_fit_button = QPushButton("适应窗口")
        self.zoom_fit_button.setObjectName("camera1ZoomFit")
        self.zoom_fit_button.setToolTip("恢复完整画面并按可用宽度等比例显示")
        preview_controls.addWidget(self.zoom_fit_button)
        layout.addLayout(preview_controls)

        self.preview = CameraImageView(
            "等待相机1实时画面与Stage3有效位姿……"
        )
        self.preview.zoom_changed.connect(self._on_preview_zoom_changed)
        self.zoom_out_button.clicked.connect(self.preview.zoom_out)
        self.zoom_in_button.clicked.connect(self.preview.zoom_in)
        self.zoom_fit_button.clicked.connect(self.preview.reset_zoom)
        self._on_preview_zoom_changed(self.preview.zoom_factor)
        layout.addWidget(self.preview)

        self.tray_summary = QLabel("槽状态：等待相机1新鲜帧")
        self.tray_summary.setObjectName("traySummary")
        layout.addWidget(self.tray_summary)

        self.slot_table = QTableWidget(len(slot_names), 6)
        self.slot_table.setObjectName("slotTable")
        self.slot_table.setHorizontalHeaderLabels(
            [
                "槽位",
                "占用",
                "ΔX_T (mm)",
                "ΔY_T (mm)",
                "距离 (mm)",
                "硅片状态",
            ]
        )
        self.slot_table.verticalHeader().setVisible(False)
        self.slot_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.slot_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.slot_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.slot_table.setAlternatingRowColors(True)
        self.slot_table.setSortingEnabled(False)
        self.slot_table.setMinimumHeight(205)
        self.slot_table.setMaximumHeight(250)
        header = self.slot_table.horizontalHeader()
        for column in range(5):
            header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self._slot_row_by_name: dict[str, int] = {}
        for row, slot_name in enumerate(slot_names):
            self._slot_row_by_name[slot_name] = row
            for column in range(self.slot_table.columnCount()):
                item = QTableWidgetItem(slot_name if column == 0 else "—")
                item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
                self.slot_table.setItem(row, column, item)
        self._set_slot_table_unavailable("等待相机1新鲜帧")
        layout.addWidget(self.slot_table)

        self.status = QLabel()
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)

        self.stage7b_status = QPlainTextEdit()
        self.stage7b_status.setObjectName("stage7bStatus")
        self.stage7b_status.setReadOnly(True)
        self.stage7b_status.setMaximumHeight(190)
        self.stage7b_status.setPlainText(
            "运动模式未ARM。默认动态演示只计算；当前全盘定位仅允许左侧目标P22。"
        )
        layout.addWidget(self.stage7b_status)

        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        self.local_jacobian_button.clicked.connect(
            self._on_local_jacobian_calibration
        )
        self.stage7b_button.clicked.connect(self._on_stage7b_button)
        self.full_tray_button.clicked.connect(self._on_full_tray_button)
        self.monitor = HandEyeMonitorThread(
            camera,
            self.project_root,
            self.suction,
            self.target_combo.currentText(),
            self.jacobian_payload,
            self.robot_state_provider,
            self,
            tray_vision_config=self.silicon_detection_config.fusion_config,
        )
        self.monitor.frame_evaluated.connect(self._on_evaluation)
        self.monitor.monitor_error.connect(self._on_monitor_error)
        self.monitor.frame_invalidated.connect(self._invalidate_current)
        self.monitor.start()
        self._update_waiting_status()

    def _update_waiting_status(self) -> None:
        jacobian_text = (
            "已加载"
            if self.jacobian_payload is not None
            else "尚未标定；可显示像素误差，但不显示XY修正量"
        )
        warning_text = (
            f"配置恢复提示：{self._silicon_config_startup_warning}\n"
            if self._silicon_config_startup_warning
            else ""
        )
        self.status.setText(
            f"Task8：{self.suction.source_path}\n"
            f"Stage5局部Jacobian：{jacobian_text}\n"
            f"硅片判定参数：{self.silicon_detection_config.source_path.name}\n"
            f"{warning_text}"
            "状态：等待相机1新鲜帧。"
        )

    def _update_silicon_parameter_label(self) -> None:
        config = self.silicon_detection_config
        label_prefix = "当前（回退）" if self._silicon_config_startup_warning else "默认"
        self.silicon_parameter_label.setText(
            f"{label_prefix}：{config.source_path.name}　配置：{config.profile_name}　"
            f"SHA256：{config.source_sha256[:12]}…"
        )
        self.silicon_parameter_label.setToolTip(str(config.source_path))

    def _on_select_silicon_detection_parameters(self) -> None:
        """Open the native Windows picker and hot-switch a complete profile."""

        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "选择硅片判定参数",
            str(self.silicon_detection_config.source_path.parent),
            "JSON 配置文件 (*.json)",
        )
        if not selected:
            return
        try:
            loaded = load_silicon_detection_config(Path(selected))
        except Exception as exc:  # noqa: BLE001 - validation message belongs in UI
            QMessageBox.critical(
                self,
                "硅片判定参数无效",
                f"未应用所选文件，当前参数保持不变。\n\n{exc}",
            )
            return
        try:
            save_preferred_silicon_detection_config_path(
                self.project_root, loaded.source_path
            )
        except Exception as exc:  # noqa: BLE001 - persistence must be explicit
            QMessageBox.critical(
                self,
                "无法保存默认硅片判定参数",
                "所选文件有效，但无法保存为后续默认配置，因此本次也未应用。"
                f"\n\n{exc}",
            )
            return
        self._silicon_config_startup_warning = ""
        self.silicon_detection_config = loaded
        self.monitor.set_tray_vision_config(loaded.fusion_config)
        self._update_silicon_parameter_label()
        self._invalidate_current(
            f"硅片判定参数已保存为默认并切换为 {loaded.source_path.name}，等待新鲜帧"
        )

    def _on_target_changed(self, target_name: str) -> None:
        self._invalidate_current("目标已切换，等待该槽的新鲜Stage3结果")
        self.jacobian_payload = load_local_xy_jacobian(
            self.project_root, self.suction, target_name
        )
        self.monitor.set_jacobian_payload(self.jacobian_payload)
        self.monitor.set_target(target_name)

    def _on_evaluation(
        self,
        image: QImage,
        evaluation: HandEyeEvaluation,
        tray_result: TrayVisionResult,
    ) -> None:
        self._last_image = image
        self._last_evaluation = evaluation
        self._last_tray_result = tray_result
        self._last_evaluation_at = time.monotonic()
        self._update_slot_table(tray_result)
        self._stage7b_samples.append(
            {
                "measurement_id": evaluation.measurement_id,
                "captured_monotonic_s": evaluation.frame_captured_monotonic_s,
                "accepted": evaluation.accepted,
                "target_name": evaluation.target_name,
                "image_error_px": evaluation.image_error_px,
                "current_robot_xy_mm": (
                    evaluation.current_robot_xy_mm
                    if evaluation.current_robot_xy_mm is not None
                    else (
                        None
                        if evaluation.current_pose is None
                        else evaluation.current_pose[:2]
                    )
                ),
                "current_joints": evaluation.current_joints,
                "current_pose": evaluation.current_pose,
                "robot_state_age_s": evaluation.robot_state_age_s,
                "tray_transform_C_T": evaluation.tray_transform_C_T,
                "suction_point_T_mm": evaluation.suction_point_T_mm,
                "target_point_T_mm": evaluation.target_point_T_mm,
                "metric_error_T_mm": evaluation.metric_error_T_mm,
                "visible_marker_count": evaluation.visible_marker_count,
                "used_marker_count": evaluation.used_marker_count,
                "reprojection_rms_px": evaluation.reprojection_rms_px,
                "annotated_bgr": evaluation.annotated_bgr.copy(),
            }
        )
        self._try_stage7b_response()
        self._refresh_preview()
        rms = (
            "—"
            if evaluation.reprojection_rms_px is None
            else f"{evaluation.reprojection_rms_px:.3f} px"
        )
        if not evaluation.accepted:
            motion_note = (
                "当前拒绝帧不会授权XY定位运动；会话状态见下方审计。"
                if self._stage7b_active
                else "当前帧不用于槽位判断；只计算，机械臂不会移动。"
            )
            self.status.setText(
                f"目标：{evaluation.target_name}\n"
                f"Stage3：REJECT — {evaluation.reason}\n"
                f"A–H：可见{evaluation.visible_marker_count}，使用{evaluation.used_marker_count}；RMS={rms}\n"
                + motion_note
            )
            return
        slot = evaluation.slot_pixel_px or (float("nan"), float("nan"))
        error = evaluation.image_error_px or (float("nan"), float("nan"))
        correction = (
            "不可用"
            if evaluation.correction_xy_mm is None
            else (
                f"ΔX={evaluation.correction_xy_mm[0]:+.3f} mm，"
                f"ΔY={evaluation.correction_xy_mm[1]:+.3f} mm"
            )
        )
        visual_threshold_state = "满足" if evaluation.aligned else "超出"
        motion_note = (
            "XY定位模式已ARM；本帧只生成候选，实际运动/复核见下方审计。"
            if self._stage7b_active
            else "未向机械臂发送任何命令。"
        )
        self.status.setText(
            f"目标：{evaluation.target_name}\n"
            f"Stage3：PASS　A–H可见{evaluation.visible_marker_count}，使用{evaluation.used_marker_count}；RMS={rms}\n"
            f"槽中心(绿)：({slot[0]:.2f}, {slot[1]:.2f}) px　"
            f"吸盘target(红)：({evaluation.suction_target_pixel_px[0]:.2f}, "
            f"{evaluation.suction_target_pixel_px[1]:.2f}) px\n"
            f"图像误差：du={error[0]:+.2f} px，dv={error[1]:+.2f} px，"
            f"|e|={evaluation.image_error_norm_px:.2f} px\n"
            f"视觉 |e|≤{evaluation.alignment_threshold_px:.2f}px（非放置精度验收）："
            f"{visual_threshold_state}\n"
            f"Jacobian只计算修正：{correction}；{evaluation.correction_note}\n"
            + motion_note
        )

    def _refresh_preview(self) -> None:
        if self._last_image is None:
            return
        self.preview.set_image(self._last_image)

    def _on_preview_zoom_changed(self, zoom_factor: float) -> None:
        """Keep zoom controls consistent with the fit-relative view state."""
        has_image = self.preview.has_image
        self.zoom_label.setText(f"{int(round(zoom_factor * 100))}%")
        self.zoom_out_button.setEnabled(
            has_image and zoom_factor > CameraImageView._MIN_ZOOM
        )
        self.zoom_fit_button.setEnabled(
            has_image and zoom_factor > CameraImageView._MIN_ZOOM
        )
        self.zoom_in_button.setEnabled(
            has_image and zoom_factor < CameraImageView._MAX_ZOOM
        )

    def _on_monitor_error(self, message: str) -> None:
        self._invalidate_current(message)

    def _invalidate_current(self, message: str) -> None:
        """Clear stale PASS imagery so it cannot be mistaken for live state."""
        self._last_image = None
        self._last_evaluation = None
        self._last_evaluation_at = None
        self.preview.clear_image("当前没有可用的新鲜相机1 / Stage3结果")
        self._last_tray_result = None
        self._set_slot_table_unavailable(message)
        note = (
            "当前判断已失效；不会授权新的XY定位运动。"
            if self._stage7b_active
            else "当前判断已失效；只计算，机械臂不会移动。"
        )
        self.status.setText(message + "\n" + note)

    def _set_slot_table_unavailable(self, reason: str) -> None:
        """Invalidate every row so a stale PASS cannot remain visible."""
        background = QColor("#F1F5F9")
        foreground = QColor("#475569")
        for slot_name, row in self._slot_row_by_name.items():
            values = (slot_name, "不确定", "—", "—", "—", "当前帧不可用")
            for column, value in enumerate(values):
                item = self.slot_table.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    self.slot_table.setItem(row, column, item)
                item.setText(value)
                item.setToolTip(reason)
                item.setBackground(background)
                item.setForeground(foreground)
                item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
        self.tray_summary.setText(f"槽状态：不可用 — {reason}")

    def _update_slot_table(self, result: TrayVisionResult) -> None:
        """Render one complete, current TrayVision result into 36 rows."""
        analysis_quality_passed = bool(
            getattr(result, "analysis_quality_passed", result.quality_passed)
        )
        if not analysis_quality_passed or len(result.slots) != 36:
            self._set_slot_table_unavailable(
                result.failure_reason or "当前帧未完成36槽分析"
            )
            return

        presentation = {
            "empty": ("否", "空槽", "#DCFCE7", "#166534"),
            "empty_unread_marker": (
                "否",
                "空槽／Marker未解码",
                "#ECFDF5",
                "#166534",
            ),
            "occupied": ("是", "正常", "#F3E8FF", "#7E22CE"),
            "warning": ("是", "警告", "#FEF3C7", "#92400E"),
            "stacked": ("是", "叠片", "#FEE2E2", "#991B1B"),
            "outside_slot": ("是", "槽外", "#FFEDD5", "#9A3412"),
            "stacked_outside_slot": (
                "是",
                "叠片且槽外",
                "#FCE7F3",
                "#9D174D",
            ),
            "out_of_view": ("不确定", "画面外", "#E2E8F0", "#475569"),
            "occluded": ("不确定", "遮挡", "#FFEDD5", "#9A3412"),
            "unknown": ("不确定", "证据不足", "#E0F2FE", "#075985"),
        }
        analyses = {analysis.projection.slot_key: analysis for analysis in result.slots}
        for slot_name, row in self._slot_row_by_name.items():
            analysis = analyses.get(slot_name)
            if analysis is None:
                self._set_slot_table_unavailable("槽位结果不完整")
                return
            state = analysis.decision.state.value
            occupied, state_text, background_hex, foreground_hex = presentation[state]
            offset = analysis.wafer_offset_T_mm
            distance = analysis.wafer_offset_distance_mm
            dx_text = "—" if offset is None else f"{offset[0]:+.2f}"
            dy_text = "—" if offset is None else f"{offset[1]:+.2f}"
            distance_text = "—" if distance is None else f"{distance:.2f}"
            values = (
                slot_name,
                occupied,
                dx_text,
                dy_text,
                distance_text,
                state_text,
            )
            flags = list(analysis.decision.flags) + list(analysis.wafer.flags)
            tooltip_lines = [analysis.decision.reason]
            if flags:
                tooltip_lines.append("flags: " + ", ".join(dict.fromkeys(flags)))
            if analysis.wafer.found:
                tooltip_lines.append(
                    f"wafer confidence: {analysis.wafer.confidence:.3f}"
                )
            tooltip = "\n".join(tooltip_lines)
            background = QColor(background_hex)
            foreground = QColor(foreground_hex)
            for column, value in enumerate(values):
                item = self.slot_table.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    self.slot_table.setItem(row, column, item)
                item.setText(value)
                item.setToolTip(tooltip)
                item.setBackground(background)
                item.setForeground(foreground)
                item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))

        summary = result.summary
        occupied_count = sum(
            int(summary.get(name, 0))
            for name in (
                "occupied",
                "warning",
                "stacked",
                "outside_slot",
                "stacked_outside_slot",
            )
        )
        empty_count = int(summary.get("empty", 0)) + int(
            summary.get("empty_unread_marker", 0)
        )
        uncertain_count = sum(
            int(summary.get(name, 0))
            for name in ("out_of_view", "occluded", "unknown")
        )
        stacked_count = int(summary.get("stacked", 0))
        outside_count = int(summary.get("outside_slot", 0))
        both_count = int(summary.get("stacked_outside_slot", 0))
        self.tray_summary.setText(
            "槽状态：Stage3 PASS　"
            f"占用={occupied_count}（叠片={stacked_count}、槽外={outside_count}、"
            f"叠片且槽外={both_count}）　空槽={empty_count}　"
            f"不确定={uncertain_count}　已分析={summary.get('analyzed', 0)}"
        )

    def _on_exposure_slider_changed(self, slider_step: int) -> None:
        exposure = int(slider_step)
        self.exposure_value_label.setText(str(exposure))
        requester = getattr(self.camera, "request_exposure_value", None)
        if not callable(requester):
            self.exposure_status_label.setText("当前相机接口不支持硬件曝光控制")
            return
        self._exposure_was_adjusted = True
        try:
            queued = bool(requester(exposure))
        except Exception as exc:
            self.exposure_status_label.setText(f"硬件曝光请求失败：{exc}")
            return
        self.exposure_status_label.setText(
            f"正在应用相机硬件整数曝光 {exposure}……"
            if queued
            else "相机线程未运行，硬件曝光未改变"
        )

    def _on_hardware_exposure_applied(
        self,
        exposure: int,
        success: bool,
        message: str,
    ) -> None:
        exposure = int(exposure)
        if exposure < 0 and exposure != int(self.exposure_slider.value()):
            return
        if exposure == 0:
            self.exposure_value_label.setText(
                "AUTO（已确认）" if success else "AUTO（未确认）"
            )
        elif success:
            self.exposure_value_label.setText(f"{exposure}（已确认）")
        else:
            self.exposure_value_label.setText(f"{exposure}（未应用）")
        prefix = "已应用" if success else "未应用"
        self.exposure_status_label.setText(f"{prefix}：{message}")

    def _on_auto_exposure_recovery_requested(self) -> None:
        requester = getattr(
            self.camera, "request_auto_exposure_recovery", None
        )
        if not callable(requester):
            self.exposure_status_label.setText("当前相机接口不支持自动曝光恢复")
            return
        self.exposure_value_label.setText("AUTO（恢复中）")
        self._exposure_was_adjusted = False
        try:
            queued = bool(requester())
        except Exception as exc:
            self.exposure_status_label.setText(f"自动曝光恢复请求失败：{exc}")
            return
        self.exposure_status_label.setText(
            "正在强制恢复相机硬件自动曝光……"
            if queued
            else "相机线程未运行，自动曝光尚未恢复"
        )

    def _reload_jacobian(self) -> Optional[dict[str, Any]]:
        self.jacobian_payload = load_local_xy_jacobian(
            self.project_root,
            self.suction,
            self.target_combo.currentText(),
        )
        self.monitor.set_jacobian_payload(self.jacobian_payload)
        return self.jacobian_payload

    def _on_local_jacobian_calibration(self) -> None:
        """Lock the selected slot into Task9; the controller runs it elsewhere."""

        if self._stage7b_active:
            QMessageBox.warning(
                self,
                "local Jacobian标定不可启动",
                "XY定位会话正在运行；请先安全停止当前会话。",
            )
            return
        target = self.target_combo.currentText()
        self.local_jacobian_calibration_requested.emit(str(target))

    @staticmethod
    def _gate_lines(fit: Mapping[str, Any]) -> list[str]:
        lines: list[str] = []
        for name, gate in (fit.get("quality_gates") or {}).items():
            passed = bool(gate.get("passed"))
            details = ", ".join(
                f"{key}={value}"
                for key, value in gate.items()
                if key != "passed"
            )
            lines.append(f"{'PASS' if passed else 'FAIL'} {name}: {details}")
        return lines

    @staticmethod
    def _current_domain_still_fresh(
        evaluation: Optional[HandEyeEvaluation],
        evaluation_age_s: float,
    ) -> bool:
        """Recheck status age at button-click time, not only frame time."""

        if (
            evaluation is None
            or not evaluation.jacobian_domain_passed
            or not evaluation.correction_available
            or evaluation.robot_state_age_s is None
        ):
            return False
        current_state_age = float(evaluation.robot_state_age_s) + max(
            0.0, float(evaluation_age_s)
        )
        return current_state_age <= ROBOT_STATE_MAXIMUM_AGE_S

    def _show_jacobian_validation(self) -> None:
        """Display Stage-5 fit/LOLO evidence; never issue a motion call."""
        payload = self._reload_jacobian()
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Information)
        dialog.setWindowTitle("Jacobian验证（只计算，不运动）")
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        dialog.setStyleSheet(LIGHT_WARNING_DIALOG_STYLESHEET)
        if payload is None:
            dialog.setText(
                "尚未找到通过质量门的 camera1_xy_image_jacobian.json。\n\n"
                "请先通过“任务执行”运行 task9_jacobiantest.py。Task9是唯一会执行"
                "1–2 mm往返运动的步骤；本验证按钮不会移动机械臂。"
            )
            dialog.exec()
            return

        fit = payload.get("fit") or payload
        matrix = fit.get("j_error_px_per_command_mm")
        cv = fit.get("cross_validation") or {}
        valid_targets = payload.get("valid_target_names") or []
        current_target = self.target_combo.currentText()
        scope_ok = current_target in valid_targets
        current = self._last_evaluation
        evaluation_age_s = (
            math.inf
            if self._last_evaluation_at is None
            else time.monotonic() - self._last_evaluation_at
        )
        current_fresh = bool(
            self._last_evaluation_at is not None and evaluation_age_s <= 1.5
        )
        current_ok = bool(current is not None and current.accepted and current_fresh)
        current_domain_ok = bool(
            current_ok
            and self._current_domain_still_fresh(current, evaluation_age_s)
        )
        gates = fit.get("quality_gates") or {}
        gates_ok = bool(
            REQUIRED_XY_JACOBIAN_QUALITY_GATES.issubset(gates)
            and all(
                isinstance(gates.get(name), Mapping)
                and bool(gates[name].get("passed"))
                for name in REQUIRED_XY_JACOBIAN_QUALITY_GATES
            )
        )
        overall = (
            payload.get("status") == "success"
            and fit.get("status") == "success"
            and gates_ok
            and scope_ok
            and current_ok
            and current_domain_ok
        )
        domain_note = (
            current.jacobian_domain_note if current is not None else "无实时评估"
        )
        if (
            current is not None
            and current.jacobian_domain_passed
            and not current_domain_ok
            and current.robot_state_age_s is not None
        ):
            domain_note = "机械臂只读状态在验证时已经过期"
        lines = [
            "验证按钮只读取标定文件和当前图像，不发送运动命令。",
            "",
            f"结果：{'通过' if overall else '当前条件未全部通过'}",
            f"标定锚点：{payload.get('anchor_target_name', '—')}；"
            f"当前目标：{current_target}；目标名匹配：{'是' if scope_ok else '否'}",
            f"J [px/mm]：{json.dumps(matrix, ensure_ascii=False)}",
            f"拟合RMS：{fit.get('fit_rms_px', '—')} px",
            f"LOLO RMS：{cv.get('rms_px', '—')} px",
            f"条件数：{fit.get('condition_number', '—')}",
            f"当前Stage3：{'PASS' if current_ok else '尚无合格实时帧'}",
            (
                "当前机器人状态/Jacobian局部域："
                f"{'PASS' if current_domain_ok else 'FAIL'}；"
                f"{domain_note}"
            ),
            "",
            *self._gate_lines(fit),
        ]
        if current is not None and current.correction_xy_mm is not None:
            lines.extend(
                [
                    "",
                    (
                        "当前只计算修正："
                        f"ΔX={current.correction_xy_mm[0]:+.3f} mm，"
                        f"ΔY={current.correction_xy_mm[1]:+.3f} mm"
                    ),
                ]
            )
        dialog.setText("\n".join(lines))
        dialog.exec()

    def _on_stage7b_button(self) -> None:
        if self._stage7b_active:
            warning = QMessageBox(self)
            warning.setWindowTitle("停止单点有限闭环")
            warning.setIcon(QMessageBox.Icon.Warning)
            warning.setText(
                "停止会话会阻止后续自动修正；若机械臂正在运动，执行器将使用安全停止。\n"
                "是否停止当前单点有限闭环？"
            )
            warning.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            warning.setDefaultButton(QMessageBox.StandardButton.No)
            warning.setStyleSheet(LIGHT_WARNING_DIALOG_STYLESHEET)
            if warning.exec() == QMessageBox.StandardButton.Yes:
                self.stage7b_stop_requested.emit()
            return
        if self.target_combo.currentText() != "P22":
            QMessageBox.warning(self, "单点有限闭环不可启动", "目前只允许目标P22。")
            return
        confirmation = QMessageBox(self)
        confirmation.setWindowTitle("确认启动单点有限闭环")
        confirmation.setIcon(QMessageBox.Icon.Warning)
        confirmation.setText(
            "单点有限闭环会实际移动机械臂XY。\n\n"
            "运行边界：最多32次XY运动；宽域规划≤0.74mm（执行硬限0.75mm），"
            "精细域每轮≤0.25mm；"
            "累计XY路径≤50mm；固定J3和绝对Rz；不下降、不操作DO/真空。\n"
            "每轮必须取得5张新的稳定Stage3图像；出现过期状态、模型响应异常、"
            "报警、急停、越域或标定hash变化会立即停止。\n\n"
            "请确认P22周围20×20mm无障碍、速度较低、急停可用。"
        )
        confirmation.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        confirmation.setDefaultButton(QMessageBox.StandardButton.No)
        confirmation.setStyleSheet(LIGHT_WARNING_DIALOG_STYLESHEET)
        if confirmation.exec() == QMessageBox.StandardButton.Yes:
            self.stage7b_start_requested.emit()

    def _on_full_tray_button(self) -> None:
        """ARM or stop the full-tray P22 coarse-to-fine session."""

        if self._stage7b_active:
            if self._positioning_mode != "full_tray":
                QMessageBox.warning(
                    self,
                    "全盘定位不可启动",
                    "单点有限闭环正在运行；请先停止当前运动会话。",
                )
                return
            warning = QMessageBox(self)
            warning.setWindowTitle("停止全盘定位")
            warning.setIcon(QMessageBox.Icon.Warning)
            warning.setText(
                "停止会阻止后续几何修正和局部精修；若机械臂正在运动，"
                "执行器将使用安全停止。\n是否停止当前全盘定位？"
            )
            warning.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            warning.setDefaultButton(QMessageBox.StandardButton.No)
            warning.setStyleSheet(LIGHT_WARNING_DIALOG_STYLESHEET)
            if warning.exec() == QMessageBox.StandardButton.Yes:
                self.full_tray_stop_requested.emit()
            return

        selected = self.target_combo.currentText()
        if selected != "P22":
            QMessageBox.warning(
                self,
                "全盘定位不可启动",
                f"左侧当前选取的是{selected}。当前版本只授权目标P22；"
                "请先选择P22。",
            )
            return
        confirmation = QMessageBox(self)
        confirmation.setWindowTitle("确认启动P22全盘定位")
        confirmation.setIcon(QMessageBox.Icon.Warning)
        confirmation.setText(
            "请再次确认：左侧选取的目标是 P22。\n\n"
            "全盘定位会实际移动机械臂XY，执行顺序为：\n"
            "1. 在当前静止姿态采集5张新鲜Stage3图像，重新登记本次Tray→World；\n"
            "2. 用本次登记计算P22 world目标，沿不超过2mm的固定J3/Rz航点粗定位；\n"
            "3. 重复Stage3+Stage4托盘毫米闭环，最后用独立5帧验收视觉误差。\n\n"
            "支持范围：托盘平移模长≤10mm、平面旋转≤5°；异常结果会先请求人员确认"
            "三姿态复核。移动托盘模式不使用旧P22 preset、Task9或Task11控制。\n\n"
            "全过程固定J3和绝对Rz，不执行Z下降，不操作DO/真空。"
            "请确认整个托盘上方及机械臂逐轴扫掠路径无障碍、低速模式有效、"
            "急停可用。当前只授权P22。"
        )
        confirmation.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        confirmation.setDefaultButton(QMessageBox.StandardButton.No)
        confirmation.setStyleSheet(LIGHT_WARNING_DIALOG_STYLESHEET)
        if confirmation.exec() == QMessageBox.StandardButton.Yes:
            self.full_tray_start_requested.emit()

    def prepare_stage7b_session(self, output_dir: Path) -> dict[str, Any]:
        """Create the calculation/evidence session; no controller call."""

        from scara.vision.stage7b_session import Stage7BSession

        if self._stage7b_active:
            raise RuntimeError("单点有限闭环会话已经运行")
        session = Stage7BSession(self.project_root, output_dir)
        self._stage7b_session = session
        self._stage7b_active = True
        self._positioning_mode = "single_loop"
        self._stage7b_pending_request = None
        self._stage7b_pending_responder = None
        self._stage7b_samples.clear()
        self.target_combo.setEnabled(False)
        self.full_tray_button.setEnabled(False)
        self.stage7b_button.setText("停止单点有限闭环")
        self.stage7b_status.setPlainText(
            "单点有限闭环已ARM：等待执行器请求与5张新鲜稳定帧。\n"
            "机械臂运动仍须经过ActionWorker独立复核。"
        )
        return session.action_task()

    def prepare_full_tray_session(self, output_dir: Path) -> dict[str, Any]:
        """Create the P22 full-tray evidence session without a controller call."""

        from scara.vision.moved_tray_positioning_session import (
            MovedTrayPositioningSession,
        )

        if self._stage7b_active:
            raise RuntimeError("已有XY定位会话正在运行")
        target_name = self.target_combo.currentText()
        if target_name != "P22":
            raise RuntimeError("当前全盘定位只授权左侧目标P22")
        if self.robot_state_provider is None:
            raise RuntimeError("缺少机械臂只读状态，无法规划全盘粗定位")
        initial_state = self.robot_state_provider()
        if not isinstance(initial_state, Mapping):
            raise RuntimeError("没有可用的新鲜机械臂状态")
        session = MovedTrayPositioningSession(
            self.project_root,
            output_dir,
            initial_state,
            target_name=target_name,
            camera_reconnected=int(
                getattr(self.camera, "connection_generation", 1)
            ) > 1,
        )
        self._stage7b_session = session
        self._stage7b_active = True
        self._positioning_mode = "full_tray"
        self._stage7b_pending_request = None
        self._stage7b_pending_responder = None
        self._stage7b_samples.clear()
        self.target_combo.setEnabled(False)
        self.stage7b_button.setEnabled(False)
        self.full_tray_button.setText("停止全盘定位")
        self.stage7b_status.setPlainText(
            "可移动托盘P22全盘定位已ARM：先等待5张新鲜Stage3帧完成本次"
            "Tray→World登记。登记通过后才会生成≤2mm粗定位航点；随后使用"
            "Stage3+Stage4毫米闭环和独立5帧hold验收。"
        )
        return session.action_task()

    def begin_stage7b_request(
        self,
        request: Mapping[str, Any],
        responder: Callable[[dict], None],
    ) -> None:
        """Begin an asynchronous five-frame window without blocking Qt."""

        mode_label = "全盘定位" if self._positioning_mode == "full_tray" else "单点有限闭环"
        if not self._stage7b_active or self._stage7b_session is None:
            responder(
                {
                    "request_id": str(request.get("request_id") or ""),
                    "decision": "abort",
                    "reason": f"{mode_label}会话未ARM",
                }
            )
            return
        if self._stage7b_pending_request is not None:
            responder(
                {
                    "request_id": str(request.get("request_id") or ""),
                    "decision": "abort",
                    "reason": f"{mode_label}已有未完成的帧窗口",
                }
            )
            return
        self._stage7b_pending_request = dict(request)
        self._stage7b_pending_responder = responder
        self._stage7b_samples.clear()
        request_id = str(request.get("request_id") or "")
        self.stage7b_status.setPlainText(
            f"{mode_label}请求 {request_id}：等待5张请求之后的新鲜合格帧……\n"
            "此时尚未授权运动。"
        )
        QTimer.singleShot(5000, lambda: self._stage7b_timeout(request_id))

    def _eligible_stage7b_samples(self) -> list[dict[str, Any]]:
        request = self._stage7b_pending_request
        if request is None:
            return []
        requested_at = float(request.get("requested_monotonic_s") or math.inf)
        rows = []
        for sample in self._stage7b_samples:
            captured = sample.get("captured_monotonic_s")
            if (
                sample.get("accepted") is True
                and sample.get("target_name") == "P22"
                and captured is not None
                and float(captured) >= requested_at
            ):
                rows.append(sample)
        return rows[-5:]

    def _try_stage7b_response(self) -> None:
        if self._stage7b_pending_request is None:
            return
        samples = self._eligible_stage7b_samples()
        if len(samples) < 5:
            return
        request = self._stage7b_pending_request
        responder = self._stage7b_pending_responder
        self._stage7b_pending_request = None
        self._stage7b_pending_responder = None
        self._stage7b_samples.clear()
        try:
            response = self._stage7b_session.build_response(request, samples)
        except Exception as exc:  # noqa: BLE001 - fail closed at UI boundary
            response = {
                "request_id": str(request.get("request_id") or ""),
                "decision": "abort",
                "reason": f"XY定位计算/证据保存失败：{exc}",
            }
        if response.get("decision") == "probe_required":
            evaluation = response.get("evaluation") or {}
            delta = evaluation.get("relative_to_stage2") or {}
            dispersion = evaluation.get("dispersion") or {}
            prompt = QMessageBox(self)
            prompt.setWindowTitle("全盘定位需要三姿态异常复核")
            prompt.setIcon(QMessageBox.Icon.Warning)
            prompt.setText(
                "单姿态5帧登记未越过硬边界，但触发异常复核条件。\n\n"
                f"估计平移模长：{float(delta.get('translation_norm_mm', float('nan'))):.3f} mm\n"
                f"估计yaw变化：{float(delta.get('yaw_deg', float('nan'))):+.3f}°\n"
                f"原点RMS：{float(dispersion.get('origin_rms_mm', float('nan'))):.3f} mm\n"
                f"yaw RMS：{float(dispersion.get('yaw_rms_deg', float('nan'))):.3f}°\n"
                f"P22不确定度：{float(dispersion.get('p22_position_uncertainty_mm', float('nan'))):.3f} mm\n\n"
                "继续会由ActionWorker以固定J3/Rz前往标定文件中的3个预验证观察姿态，"
                "每处采5帧并融合；仍不执行Z、DO或真空。是否继续？"
            )
            prompt.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            prompt.setDefaultButton(QMessageBox.StandardButton.No)
            prompt.setStyleSheet(LIGHT_WARNING_DIALOG_STYLESHEET)
            if prompt.exec() == QMessageBox.StandardButton.Yes:
                try:
                    self._stage7b_session.authorize_three_pose_probe()
                    response = {
                        "request_id": str(request.get("request_id") or ""),
                        "decision": "observe",
                        "calibration_sha256": str(
                            getattr(self._stage7b_session, "calibration_hash", "")
                        ),
                        "reason": "人员已确认三姿态异常复核；下一窗口开始受限XY路线",
                        "evaluation": evaluation,
                    }
                except Exception as exc:  # noqa: BLE001
                    response = {
                        "request_id": str(request.get("request_id") or ""),
                        "decision": "abort",
                        "reason": f"无法启动三姿态复核：{exc}",
                    }
            else:
                response = {
                    "request_id": str(request.get("request_id") or ""),
                    "decision": "abort",
                    "reason": "人员拒绝三姿态异常复核；全盘定位未开始",
                }
        if callable(responder):
            responder(response)
        decision = response.get("decision")
        if decision == "approve":
            proposal = response.get("proposal") or {}
            proposal_phase = str(proposal.get("phase") or "")
            if proposal_phase.startswith("moved_tray_"):
                self.stage7b_status.setPlainText(
                    self._moved_tray_report_text(
                        proposal,
                        "候选已提交ActionWorker独立复核；尚未代表运动已下发。",
                    )
                )
            elif proposal_phase == "stage3_metric_geometry_correction":
                self.stage7b_status.setPlainText(
                    self._geometry_correction_report_text(
                        proposal,
                        "一次毫米几何修正已提交执行器复核；尚未代表运动已下发。",
                    )
                )
            else:
                self.stage7b_status.setPlainText(
                    self._stage7b_report_text(
                        proposal,
                        "候选已提交执行器复核；尚未代表运动已经下发。",
                    )
                )
        elif decision == "observe":
            evaluation = response.get("evaluation") or {}
            self.stage7b_status.setPlainText(
                "全盘定位观察窗口完成；本窗口不运动。\n"
                + str(response.get("reason") or "继续下一窗口")
                + ("\n" + self._runtime_registration_text(evaluation) if evaluation else "")
            )
        elif decision == "complete":
            evaluation = response.get("evaluation") or {}
            if evaluation:
                self.stage7b_status.setPlainText(self._stage7b_report_text(
                    evaluation,
                    str(
                        response.get("reason")
                        or "已经抵达目标终止范围；停止XY闭环，不再运动。"
                    ),
                ))
            else:
                self.stage7b_status.setPlainText(
                    "XY定位已正常结束；停止XY闭环。\n"
                    + str(response.get("reason") or "")
                )
        else:
            evaluation = response.get("evaluation") or {}
            if evaluation:
                formatter = (
                    self._geometry_correction_report_text
                    if evaluation.get("phase")
                    == "stage3_metric_geometry_correction"
                    else self._stage7b_report_text
                )
                self.stage7b_status.setPlainText(
                    formatter(
                        evaluation,
                        f"已停止：{response.get('reason', '安全门拒绝')}",
                    )
                )
            else:
                label = "全盘定位" if self._positioning_mode == "full_tray" else "单点有限闭环"
                self.stage7b_status.setPlainText(f"{label}已停止：{response.get('reason', '安全门拒绝')}")

    @staticmethod
    def _moved_tray_report_text(report: Mapping[str, Any], header: str) -> str:
        command = report.get("commanded_correction_xy_mm") or [0.0, 0.0]
        endpoint = report.get("predicted_endpoint_xy_mm") or [float("nan"), float("nan")]
        metric = report.get("median_metric_error_T_mm") or [float("nan")] * 3
        gates = [
            f"{'PASS' if gate.get('passed') else 'FAIL'} {name}: "
            f"actual={gate.get('actual')} limit={gate.get('limit')}"
            for name, gate in (report.get("safety_gates") or {}).items()
        ]
        return "\n".join(
            [
                f"可移动托盘全盘定位；阶段={report.get('phase')}；{header}",
                f"Tray毫米误差=({float(metric[0]):+.3f},{float(metric[1]):+.3f})mm；"
                f"|e|={float(report.get('median_error_norm_mm', float('nan'))):.3f}mm",
                f"世界系命令 ΔX={float(command[0]):+.3f}mm，ΔY={float(command[1]):+.3f}mm",
                f"预计终点 world XY=({float(endpoint[0]):.3f},{float(endpoint[1]):.3f})mm",
                "全部本轮安全门：",
                *gates,
            ]
        )

    @staticmethod
    def _runtime_registration_text(report: Mapping[str, Any]) -> str:
        if "origin_world_xy_mm" not in report:
            return ""
        origin = report.get("origin_world_xy_mm") or [float("nan"), float("nan")]
        delta = report.get("relative_to_stage2") or {}
        dispersion = report.get("dispersion") or {}
        return (
            "本次Tray→World登记："
            f"origin=({float(origin[0]):.3f},{float(origin[1]):.3f})mm，"
            f"yaw={float(report.get('yaw_world_from_tray_deg', float('nan'))):+.3f}°\n"
            f"相对Stage2：ΔXY={delta.get('translation_xy_mm')}mm，"
            f"|Δ|={float(delta.get('translation_norm_mm', float('nan'))):.3f}mm，"
            f"Δyaw={float(delta.get('yaw_deg', float('nan'))):+.3f}°；"
            f"P22不确定度={float(dispersion.get('p22_position_uncertainty_mm', float('nan'))):.3f}mm。"
        )

    @staticmethod
    def _geometry_correction_report_text(
        report: Mapping[str, Any], header: str
    ) -> str:
        error = report.get("median_metric_error_T_mm") or [float("nan")] * 3
        command = report.get("commanded_correction_xy_mm") or [0.0, 0.0]
        endpoint = report.get("predicted_endpoint_xy_mm") or [float("nan")] * 2
        gate_lines = [
            f"{'PASS' if gate.get('passed') else 'FAIL'} {name}: "
            f"actual={gate.get('actual')} limit={gate.get('limit')}"
            for name, gate in (report.get("safety_gates") or {}).items()
        ]
        return "\n".join(
            [
                f"全盘定位 / Stage3毫米几何修正；{header}",
                (
                    "托盘系误差 e_T="
                    f"({error[0]:+.3f},{error[1]:+.3f},{error[2]:+.3f})mm"
                ),
                f"世界系命令 ΔX={command[0]:+.3f}mm，ΔY={command[1]:+.3f}mm",
                f"预计终点 world XY=({endpoint[0]:.3f},{endpoint[1]:.3f})mm",
                "全部几何修正安全门：",
                *gate_lines,
            ]
        )

    @staticmethod
    def _stage7b_report_text(report: Mapping[str, Any], header: str) -> str:
        raw_command = report.get("commanded_correction_xy_mm")
        command = raw_command or [0.0, 0.0]
        error = report.get("median_error_px") or [float("nan"), float("nan")]
        endpoint = report.get("predicted_endpoint_xy_mm") or [float("nan"), float("nan")]
        predicted = report.get("predicted_error_px") or [float("nan"), float("nan")]
        error_norm = report.get("error_norm_px")
        norm_text = "—" if error_norm is None else f"{float(error_norm):.3f}"
        remaining = report.get("remaining_alignment_distance_mm")
        remaining_text = "—" if remaining is None else f"{float(remaining):.3f}"
        gate_lines = [
            f"{'PASS' if gate.get('passed') else 'FAIL'} {name}: "
            f"actual={gate.get('actual')} limit={gate.get('limit')}"
            for name, gate in (report.get("safety_gates") or {}).items()
        ]
        lines = [
            f"单点有限闭环第{report.get('iteration_index')}轮；模型={report.get('model_tier')}；{header}",
            f"修正前误差 e=({error[0]:+.3f},{error[1]:+.3f}) px；|e|={norm_text}px",
            f"视觉模型估计吸盘到目标剩余XY距离={remaining_text}mm",
        ]
        if raw_command is None:
            lines.append("本轮满足终止条件，不再发送XY命令。")
        else:
            lines.extend(
                [
                    f"命令 ΔX={command[0]:+.3f}mm，ΔY={command[1]:+.3f}mm",
                    f"预计终点 world XY=({endpoint[0]:.3f},{endpoint[1]:.3f})mm",
                    f"预计误差=({predicted[0]:+.3f},{predicted[1]:+.3f})px",
                ]
            )
        lines.extend(["本轮全部计算安全门：", *gate_lines])
        return "\n".join(lines)

    def _stage7b_timeout(self, request_id: str) -> None:
        request = self._stage7b_pending_request
        if request is None or str(request.get("request_id") or "") != request_id:
            return
        responder = self._stage7b_pending_responder
        self._stage7b_pending_request = None
        self._stage7b_pending_responder = None
        if callable(responder):
            responder(
                {
                    "request_id": request_id,
                    "decision": "abort",
                    "reason": "5秒内未取得5张新的合格Stage3帧",
                }
            )
        label = "全盘定位" if self._positioning_mode == "full_tray" else "单点有限闭环"
        self.stage7b_status.setPlainText(f"{label}已停止：新鲜稳定帧超时，未授权运动。")

    def finish_stage7b_session(
        self, ok: bool, message: str
    ) -> tuple[bool, str]:
        mode = self._positioning_mode
        if self._stage7b_session is not None:
            self._stage7b_session.finish(ok, message)
            if (
                mode == "full_tray"
                and ok
                and getattr(self._stage7b_session, "status", "")
                != "converged"
            ):
                ok = False
                message = (
                    "全盘定位动作序列结束，但独立5帧视觉1mm验收未通过；"
                    "不得视为定位成功"
                )
        self._stage7b_active = False
        self._stage7b_pending_request = None
        self._stage7b_pending_responder = None
        self._stage7b_samples.clear()
        self.target_combo.setEnabled(True)
        self.stage7b_button.setEnabled(True)
        self.full_tray_button.setEnabled(True)
        self.stage7b_button.setText("单点有限闭环")
        self.full_tray_button.setText("全盘定位")
        label = "全盘定位" if mode == "full_tray" else "单点有限闭环"
        self.stage7b_status.setPlainText(
            f"{label}{'完成' if ok else '停止'}：{message}\n"
            "实时动态演示继续运行；未执行Z、DO或真空动作。"
        )
        self._stage7b_session = None
        self._positioning_mode = None
        return bool(ok), str(message)

    @property
    def stage7b_active(self) -> bool:
        return bool(self._stage7b_active)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._stage7b_active:
            event.ignore()
            label = "全盘定位" if self._positioning_mode == "full_tray" else "单点有限闭环"
            self.stage7b_status.setPlainText(f"{label}运行期间不能关闭动态演示；请先停止会话。")
            return
        if hasattr(self, "monitor") and self.monitor.isRunning():
            if not self.monitor.stop():
                event.ignore()
                self._invalidate_current(
                    "后台Stage3计算尚未安全退出，请稍候再关闭窗口"
                )
                return
        if self._exposure_was_adjusted:
            restore_exposure = getattr(
                self.camera,
                "restore_original_exposure",
                None,
            )
            if callable(restore_exposure):
                try:
                    restore_exposure()
                except Exception:
                    pass
        super().closeEvent(event)


__all__ = ["HandEyeDemoDialog", "HandEyeMonitorThread"]
