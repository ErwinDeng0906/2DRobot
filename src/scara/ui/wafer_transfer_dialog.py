"""Live camera-1 wafer-transfer planning and tracking dialog."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import cv2
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QImage, QMouseEvent, QPixmap, QResizeEvent
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from scara.vision.wafer_transfer_runtime import (
    LiveWaferTransferRuntime,
    WaferTransferFrame,
)


_STYLE = """
QDialog { background:#FFFFFF; color:#111827; }
QLabel { color:#111827; font-size:14px; }
QLabel#transferPreview { background:#111827; border:1px solid #94A3B8; }
QLabel#transferBanner {
    background:#EFF6FF; color:#1E3A8A; border:1px solid #93C5FD;
    border-radius:5px; padding:8px; font-size:15px; font-weight:700;
}
QPlainTextEdit {
    background:#F8FAFC; color:#111827; border:1px solid #CBD5E1;
    border-radius:5px; padding:8px; font:14px Consolas, "Microsoft YaHei";
}
QPushButton {
    background:#F3F4F6; color:#111827; border:1px solid #94A3B8;
    border-radius:4px; padding:7px 14px; min-width:96px; font-size:14px;
}
QPushButton:hover { background:#E2E8F0; }
QPushButton:checked { color:#FFFFFF; background:#2563EB; border-color:#2563EB; }
QPushButton#trackButton { color:#FFFFFF; background:#047857; border-color:#047857; }
QPushButton#trackButton:disabled { color:#64748B; background:#E2E8F0; border-color:#CBD5E1; }
"""


class ClickableImageLabel(QLabel):
    image_clicked = pyqtSignal(float, float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("transferPreview")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(800, 520)
        self._image: Optional[QImage] = None
        self._displayed_size = (0, 0)

    def set_image(self, image: QImage) -> None:
        self._image = image.copy()
        self._render()

    def _render(self) -> None:
        if self._image is None:
            return
        pixmap = QPixmap.fromImage(self._image).scaled(
            max(1, self.width()),
            max(1, self.height()),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._displayed_size = (pixmap.width(), pixmap.height())
        self.setPixmap(pixmap)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._render()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if (
            event.button() != Qt.MouseButton.LeftButton
            or self._image is None
            or self._displayed_size[0] <= 0
            or self._displayed_size[1] <= 0
        ):
            super().mousePressEvent(event)
            return
        shown_w, shown_h = self._displayed_size
        offset_x = (self.width() - shown_w) / 2.0
        offset_y = (self.height() - shown_h) / 2.0
        x = float(event.position().x()) - offset_x
        y = float(event.position().y()) - offset_y
        if not (0.0 <= x < shown_w and 0.0 <= y < shown_h):
            return
        image_x = x * self._image.width() / shown_w
        image_y = y * self._image.height() / shown_h
        self.image_clicked.emit(float(image_x), float(image_y))


class WaferTransferMonitorThread(QThread):
    frame_ready = pyqtSignal(QImage, object)
    monitor_error = pyqtSignal(str)
    frame_invalidated = pyqtSignal(str)

    def __init__(
        self,
        camera: Any,
        runtime: LiveWaferTransferRuntime,
        robot_state_provider: Optional[Callable[[], Optional[Mapping[str, Any]]]],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.camera = camera
        self.runtime = runtime
        self.robot_state_provider = robot_state_provider
        self._running = True

    def stop(self, timeout_ms: int = 5000) -> bool:
        self._running = False
        self.requestInterruption()
        return bool(self.wait(int(timeout_ms)))

    def run(self) -> None:
        last_sequence: Optional[int] = None
        stale_since = time.monotonic()
        invalidated = False
        last_error = ""
        while self._running and not self.isInterruptionRequested():
            packet = self.camera.latest_frame_packet(max_age_s=1.0)
            if packet is None:
                if not invalidated and time.monotonic() - stale_since > 1.0:
                    self.frame_invalidated.emit("相机1没有新鲜画面")
                    invalidated = True
                self.msleep(60)
                continue
            frame, sequence, captured_at = packet
            if sequence == last_sequence:
                self.msleep(45)
                continue
            last_sequence = int(sequence)
            stale_since = time.monotonic()
            invalidated = False
            state = None
            if self.robot_state_provider is not None:
                try:
                    state = self.robot_state_provider()
                except Exception:
                    state = None
            try:
                result = self.runtime.process_camera1(
                    frame,
                    frame_sequence=int(sequence),
                    captured_monotonic_s=float(captured_at),
                    robot_state=state,
                )
                rgb = cv2.cvtColor(result.annotated_bgr, cv2.COLOR_BGR2RGB)
                height, width, channels = rgb.shape
                image = QImage(
                    rgb.data,
                    width,
                    height,
                    channels * width,
                    QImage.Format.Format_RGB888,
                ).copy()
                self.frame_ready.emit(image, result)
                last_error = ""
            except Exception as exc:  # noqa: BLE001 - live display remains fail-closed
                message = f"转移视觉分析失败：{exc}"
                if message != last_error:
                    self.monitor_error.emit(message)
                    last_error = message
            self.msleep(70)


class WaferTransferDialog(QDialog):
    """Click source/destination slots and follow live robot-relative distance."""

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
        if int(camera.source_index) != 1:
            raise RuntimeError("转移视觉要求相机源#1")
        self.project_root = Path(project_root)
        self.camera = camera
        self.runtime = LiveWaferTransferRuntime(self.project_root)
        self._last_frame: Optional[WaferTransferFrame] = None
        self._selection_role = "source"
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle("Wafer Transfer Vision")
        self.resize(1500, 920)
        self.setStyleSheet(_STYLE)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(9)
        banner = QLabel(
            "OBSERVATION ONLY · robot motion authorization remains in ActionWorker"
        )
        banner.setObjectName("transferBanner")
        root.addWidget(banner)

        controls = QHBoxLayout()
        self.source_button = QPushButton("选择拾取槽")
        self.destination_button = QPushButton("选择放置槽")
        self.source_button.setCheckable(True)
        self.destination_button.setCheckable(True)
        self.source_button.setChecked(True)
        group = QButtonGroup(self)
        group.setExclusive(True)
        group.addButton(self.source_button)
        group.addButton(self.destination_button)
        self.track_button = QPushButton("开始跟踪")
        self.track_button.setObjectName("trackButton")
        self.reset_button = QPushButton("清除选择")
        self.registration_button = QPushButton("重建 W←T")
        self.close_button = QPushButton("关闭")
        for widget in (
            self.source_button,
            self.destination_button,
            self.track_button,
            self.reset_button,
            self.registration_button,
            self.close_button,
        ):
            controls.addWidget(widget)
        controls.addStretch(1)
        root.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.preview = ClickableImageLabel()
        self.preview.setText("等待相机1画面")
        self.status = QPlainTextEdit()
        self.status.setReadOnly(True)
        self.status.setMinimumWidth(390)
        splitter.addWidget(self.preview)
        splitter.addWidget(self.status)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

        self.source_button.clicked.connect(lambda: self._set_selection_role("source"))
        self.destination_button.clicked.connect(
            lambda: self._set_selection_role("destination")
        )
        self.preview.image_clicked.connect(self._on_image_clicked)
        self.track_button.clicked.connect(self._start_tracking)
        self.reset_button.clicked.connect(self._reset_selection)
        self.registration_button.clicked.connect(self._reset_registration)
        self.close_button.clicked.connect(self.close)

        self.monitor = WaferTransferMonitorThread(
            camera,
            self.runtime,
            robot_state_provider,
            self,
        )
        self.monitor.frame_ready.connect(self._on_frame)
        self.monitor.monitor_error.connect(self._show_error)
        self.monitor.frame_invalidated.connect(self._show_error)
        self.monitor.start()
        self._refresh_status(self.runtime.snapshot())

    def _set_selection_role(self, role: str) -> None:
        self._selection_role = str(role)
        self._refresh_status(self.runtime.snapshot())

    def _on_image_clicked(self, x: float, y: float) -> None:
        try:
            slot, point_T, distance = self.runtime.select_pixel(
                (x, y),
                role=self._selection_role,
            )
            self._refresh_status(
                self.runtime.snapshot(),
                extra=(
                    f"点击像素=({x:.1f},{y:.1f})\n"
                    f"托盘坐标=({point_T[0]:+.3f},{point_T[1]:+.3f},{point_T[2]:+.3f}) mm\n"
                    f"选择={slot}，距槽中心={distance:.3f} mm"
                ),
            )
        except Exception as exc:  # noqa: BLE001 - selection remains unchanged
            self._show_error(str(exc))

    def _start_tracking(self) -> None:
        try:
            self.runtime.start_tracking()
            self._refresh_status(
                self.runtime.snapshot(),
                extra="目标已锁定，开始随机械臂状态更新吸盘距离。",
            )
        except Exception as exc:  # noqa: BLE001
            self._show_error(str(exc))

    def _reset_selection(self) -> None:
        self.runtime.reset_selection()
        self.source_button.setChecked(True)
        self._selection_role = "source"
        self._refresh_status(self.runtime.snapshot(), extra="已清除拾取槽和放置槽。")

    def _reset_registration(self) -> None:
        self.runtime.reset_registration()
        self._refresh_status(
            self.runtime.snapshot(),
            extra="W←T已清除，等待5张机械臂静止且时间同步的合格帧。",
        )

    def _on_frame(self, image: QImage, frame: WaferTransferFrame) -> None:
        self._last_frame = frame
        self.preview.set_image(image)
        self._refresh_status(frame.session_snapshot)

    @staticmethod
    def _gate_lines(gates: Mapping[str, Any]) -> list[str]:
        lines = []
        for name, gate in gates.items():
            passed = gate.get("passed") is True
            lines.append(
                f"{'PASS' if passed else 'WAIT'} {name}\n  actual={gate.get('actual')}\n  limit={gate.get('limit')}"
            )
        return lines

    def _refresh_status(self, snapshot: Mapping[str, Any], *, extra: str = "") -> None:
        source = snapshot.get("source_state") or {}
        destination = snapshot.get("destination_state") or {}
        delta = snapshot.get("active_delta_world_xy_mm")
        distance = snapshot.get("active_distance_mm")
        registration = snapshot.get("registration") or {}
        origin = registration.get("origin_world_xy_mm")
        if delta is None:
            delta_text = "不可用"
        else:
            delta_text = (
                f"ΔX={float(delta[0]):+.3f} mm，ΔY={float(delta[1]):+.3f} mm，"
                f"距离={float(distance):.3f} mm"
            )
        if origin is None:
            registration_text = "等待5张静止同步帧"
        else:
            registration_text = (
                f"PASS · origin=({float(origin[0]):.3f},{float(origin[1]):.3f}) mm · "
                f"yaw={float(registration.get('yaw_world_from_tray_deg')):+.3f}°"
            )
        lines = [
            f"选择模式：{'拾取槽' if self._selection_role == 'source' else '放置槽'}",
            f"阶段：{snapshot.get('phase')}",
            f"拾取槽：{snapshot.get('source_slot') or '—'} · {source.get('state', '—')}",
            f"放置槽：{snapshot.get('destination_slot') or '—'} · {destination.get('state', '—')}",
            f"当前目标：{snapshot.get('active_target_slot') or '—'}",
            f"W←T：{registration_text}",
            f"吸盘到当前目标：{delta_text}",
            f"相机2近距：{(snapshot.get('close_range') or {}).get('state', 'unavailable')}",
            "",
            "当前质量门：",
            *self._gate_lines(snapshot.get("selection_gates") or {}),
            "",
            "相机2近距质量门：",
            *(
                self._gate_lines(snapshot.get("close_range_gates") or {})
                or ["等待近距观测"]
            ),
            "",
            "机械臂运动授权：NO",
            "P22运动仍由现有ActionWorker安全链独立复核。",
        ]
        registration_error = snapshot.get("registration_error")
        sync_error = snapshot.get("robot_state_sync_error")
        if registration_error:
            lines.extend(["", f"W←T状态：{registration_error}"])
        if sync_error:
            lines.extend(["", f"时间同步：{sync_error}"])
        if extra:
            lines.extend(["", extra])
        self.status.setPlainText("\n".join(lines))
        self.track_button.setEnabled(bool(snapshot.get("tracking_ready")))

    def _show_error(self, message: str) -> None:
        snapshot = self.runtime.snapshot()
        self._refresh_status(snapshot, extra=f"当前操作未接受：{message}")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.monitor.isRunning() and not self.monitor.stop(5000):
            self._show_error("后台视觉线程尚未退出")
            event.ignore()
            return
        event.accept()


__all__ = [
    "ClickableImageLabel",
    "WaferTransferDialog",
    "WaferTransferMonitorThread",
]
