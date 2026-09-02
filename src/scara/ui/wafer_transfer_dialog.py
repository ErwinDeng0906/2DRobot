"""Live camera-1 wafer-transfer planning and tracking dialog."""

from __future__ import annotations

import math
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import cv2
from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QImage, QMouseEvent, QPixmap, QResizeEvent
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
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
from scara.vision.wafer_pick_xy_positioning import (
    OBSERVATION_WINDOW_SIZE,
    select_bounded_observation_window,
)


_PICK_XY_REQUEST_TIMEOUT_MS = 4000


_STYLE = """
QDialog { background:#FFFFFF; color:#111827; }
QLabel { color:#111827; font-size:16px; }
QLabel#transferPreview { background:#111827; border:1px solid #94A3B8; }
QLabel#transferBanner {
    background:#EFF6FF; color:#1E3A8A; border:1px solid #93C5FD;
    border-radius:5px; padding:9px; font-size:17px; font-weight:700;
}
QPlainTextEdit {
    background:#F8FAFC; color:#111827; border:1px solid #CBD5E1;
    border-radius:5px; padding:10px; font:16px Consolas, "Microsoft YaHei";
}
QPushButton {
    background:#F3F4F6; color:#111827; border:1px solid #94A3B8;
    border-radius:4px; padding:8px 14px; min-width:104px; font-size:16px;
}
QPushButton:hover { background:#E2E8F0; }
QPushButton:checked { color:#FFFFFF; background:#2563EB; border-color:#2563EB; }
QPushButton#trackButton { color:#FFFFFF; background:#047857; border-color:#047857; }
QPushButton#trackButton:disabled { color:#64748B; background:#E2E8F0; border-color:#CBD5E1; }
QPushButton#xyMotionButton { color:#FFFFFF; background:#B45309; border-color:#92400E; }
QPushButton#xyMotionButton:disabled { color:#64748B; background:#E2E8F0; border-color:#CBD5E1; }
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

    def clear_image(self, message: str) -> None:
        self._image = None
        self._displayed_size = (0, 0)
        self.clear()
        self.setText(str(message))

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
                    message = "相机1没有新鲜画面"
                    self.runtime.invalidate_camera1(message)
                    self.frame_invalidated.emit(message)
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
                    self.runtime.invalidate_camera1(message)
                    self.monitor_error.emit(message)
                    last_error = message
            self.msleep(70)


class WaferTransferDialog(QDialog):
    """Click source/destination slots and follow live robot-relative distance."""

    pick_xy_start_requested = pyqtSignal()
    pick_xy_stop_requested = pyqtSignal()

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
        self.robot_state_provider = robot_state_provider
        self._last_frame: Optional[WaferTransferFrame] = None
        self._selection_role = "source"
        self._last_interaction_text = ""
        self._stream_invalidated = False
        self._pick_xy_session: Optional[Any] = None
        self._pick_xy_active = False
        self._pick_xy_pending_request: Optional[dict[str, Any]] = None
        self._pick_xy_pending_responder: Optional[Callable[[dict], None]] = None
        self._pick_xy_samples: deque[dict[str, Any]] = deque(maxlen=20)
        self._pick_xy_rejection_counts: Counter[str] = Counter()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle("Wafer Transfer Vision")
        self.resize(1500, 920)
        self.setStyleSheet(_STYLE)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(9)
        banner = QLabel(
            "默认只读；XY悬空移动须人工ARM并由ActionWorker逐步复核。"
            "J3不下降，不开启真空或DO。"
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
        self.track_button = QPushButton("锁定拾取导航")
        self.track_button.setObjectName("trackButton")
        self.xy_motion_button = QPushButton("启动XY悬空移动")
        self.xy_motion_button.setObjectName("xyMotionButton")
        self.reset_button = QPushButton("清除选择")
        self.registration_button = QPushButton("重建 W←T")
        self.report_button = QPushButton("保存导航报告")
        self.close_button = QPushButton("关闭")
        for widget in (
            self.source_button,
            self.destination_button,
            self.track_button,
            self.xy_motion_button,
            self.reset_button,
            self.registration_button,
            self.report_button,
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
        self.status.setMinimumWidth(500)
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
        self.xy_motion_button.clicked.connect(self._on_xy_motion_button)
        self.reset_button.clicked.connect(self._reset_selection)
        self.registration_button.clicked.connect(self._reset_registration)
        self.report_button.clicked.connect(self._save_report)
        self.close_button.clicked.connect(self.close)

        self.monitor = WaferTransferMonitorThread(
            camera,
            self.runtime,
            robot_state_provider,
            self,
        )
        self.monitor.frame_ready.connect(self._on_frame)
        self.monitor.monitor_error.connect(self._invalidate_current)
        self.monitor.frame_invalidated.connect(self._invalidate_current)
        self.monitor.start()
        self._refresh_status(self.runtime.snapshot())

    def _set_selection_role(self, role: str) -> None:
        self._selection_role = str(role)
        self._refresh_status(self.runtime.snapshot())

    def _on_image_clicked(self, x: float, y: float) -> None:
        if self._pick_xy_active:
            self._show_error("XY悬空定位已ARM，不允许在运动会话中更换目标")
            return
        try:
            slot, point_T, distance = self.runtime.select_pixel(
                (x, y),
                role=self._selection_role,
            )
            self._last_interaction_text = (
                f"最近点击：像素=({x:.1f},{y:.1f})\n"
                f"托盘坐标=({point_T[0]:+.3f},{point_T[1]:+.3f},{point_T[2]:+.3f}) mm\n"
                f"已选择 {slot}，点击点距槽中心={distance:.3f} mm"
            )
            self._refresh_status(self.runtime.snapshot())
        except Exception as exc:  # noqa: BLE001 - selection remains unchanged
            self._show_error(str(exc))

    def _start_tracking(self) -> None:
        try:
            self.runtime.start_tracking()
            self._last_interaction_text = (
                "拾取目标已锁定；画面与右侧面板将随机械臂状态更新吸盘距离。"
            )
            self._refresh_status(self.runtime.snapshot())
        except Exception as exc:  # noqa: BLE001
            self._show_error(str(exc))

    def _reset_selection(self) -> None:
        self.runtime.reset_selection()
        self.source_button.setChecked(True)
        self._selection_role = "source"
        self._last_interaction_text = "已清除拾取槽和放置槽。"
        self._refresh_status(self.runtime.snapshot())

    def _reset_registration(self) -> None:
        self.runtime.reset_registration()
        self._last_interaction_text = (
            "W←T已清除，等待5张机械臂静止且时间同步的合格帧。"
        )
        self._refresh_status(self.runtime.snapshot())

    def _save_report(self) -> None:
        default_path = self.project_root / "wafer_pick_navigation.json"
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "保存只读导航报告",
            str(default_path),
            "JSON (*.json)",
        )
        if not selected:
            return
        try:
            saved = self.runtime.save_report(Path(selected))
            self._last_interaction_text = f"只读导航报告已保存：{saved}"
            self._refresh_status(self.runtime.snapshot())
        except Exception as exc:  # noqa: BLE001
            self._show_error(str(exc))

    def _on_frame(self, image: QImage, frame: WaferTransferFrame) -> None:
        self._last_frame = frame
        if self._stream_invalidated:
            self._last_interaction_text = ""
            self._stream_invalidated = False
        self.preview.set_image(image)
        if self._pick_xy_active and self._pick_xy_pending_request is not None:
            sample = self._pick_xy_sample(frame)
            self._pick_xy_samples.append(sample)
            if sample.get("accepted") is not True:
                self._pick_xy_rejection_counts.update(
                    str(reason)
                    for reason in sample.get("rejection_reasons")
                    or ["未知质量门"]
                )
            self._try_pick_xy_response()
        self._refresh_status(frame.session_snapshot)

    def _on_xy_motion_button(self) -> None:
        if self._pick_xy_active:
            confirmation = QMessageBox.question(
                self,
                "停止XY悬空移动",
                "停止会阻止后续XY步骤，并触发执行器安全停止。是否停止？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirmation == QMessageBox.StandardButton.Yes:
                self.pick_xy_stop_requested.emit()
            return

        snapshot = self.runtime.snapshot()
        target = str(snapshot.get("source_slot") or "")
        if not target:
            self._show_error("请先点击并锁定一个正常 occupied 硅片")
            return
        if snapshot.get("tracking_ready") is not True:
            self._show_error("当前视觉、W<-T或机械臂同步状态尚未通过")
            return
        delta = snapshot.get("active_delta_world_xy_mm") or [math.nan, math.nan]
        distance = float(snapshot.get("active_distance_mm", math.nan))
        robot_state = snapshot.get("robot_state") or {}
        joints = robot_state.get("joints") or [math.nan] * 4
        prompt = QMessageBox(self)
        prompt.setWindowTitle(f"确认移动到 {target} 正上方")
        prompt.setIcon(QMessageBox.Icon.Warning)
        prompt.setText(
            f"目标：{target}\n"
            f"当前估计：ΔX={float(delta[0]):+.3f} mm，"
            f"ΔY={float(delta[1]):+.3f} mm，距离={distance:.3f} mm\n"
            f"当前J3={float(joints[2]):.4f} mm\n\n"
            "启动后会真实移动机械臂：\n"
            "1. 仅执行分段XY运动：远距离单步最多9.99 mm，接近目标自动减小；\n"
            "2. J3保持相机1安全观察高度，不执行下降；\n"
            "3. XY到位后只旋转J4，使工具方向对齐本次相机1测得的托盘方向；\n"
            "4. 不发送DO、真空或吸取命令；\n"
            "5. 每步后重新采2张合格帧；4秒内画面不足会跳过本轮并自动重试，"
            "明确质量门失败或误差不减小仍会停止。\n\n"
            "请确认托盘上方和逐轴扫掠路径无障碍、控制器T1模式且速度不超过20%、"
            "软限位正确且急停可用。"
        )
        prompt.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        prompt.setDefaultButton(QMessageBox.StandardButton.No)
        if prompt.exec() == QMessageBox.StandardButton.Yes:
            self.pick_xy_start_requested.emit()

    def prepare_pick_xy_session(self, output_dir: Path) -> dict[str, Any]:
        """Arm one selected-slot session; controller ownership stays external."""

        if self._pick_xy_active:
            raise RuntimeError("XY悬空定位会话已经运行")
        snapshot = self.runtime.snapshot()
        target = str(snapshot.get("source_slot") or "")
        if not target or snapshot.get("tracking_ready") is not True:
            raise RuntimeError("当前拾取目标或视觉质量门无效")
        if self.robot_state_provider is None:
            raise RuntimeError("缺少机械臂只读状态")
        initial_state = self.robot_state_provider()
        if not isinstance(initial_state, Mapping):
            raise RuntimeError("没有可用的新鲜机械臂状态")
        from scara.vision.wafer_pick_xy_positioning import (
            WaferPickXYPositioningSession,
        )

        session = WaferPickXYPositioningSession(
            self.project_root,
            Path(output_dir),
            initial_state,
            snapshot,
            target_name=target,
        )
        self.runtime.start_tracking()
        self._pick_xy_session = session
        self._pick_xy_active = True
        self._pick_xy_pending_request = None
        self._pick_xy_pending_responder = None
        self._pick_xy_samples.clear()
        self._set_pick_xy_controls(True)
        self._last_interaction_text = (
            f"{target} XY悬空定位已ARM；等待ActionWorker请求后的2张合格帧。"
        )
        self._refresh_status(self.runtime.snapshot())
        return session.action_task()

    def _set_pick_xy_controls(self, active: bool) -> None:
        self.xy_motion_button.setText(
            "停止XY悬空移动" if active else "启动XY悬空移动"
        )
        for widget in (
            self.source_button,
            self.destination_button,
            self.track_button,
            self.reset_button,
            self.registration_button,
        ):
            widget.setEnabled(not active)

    def begin_pick_xy_request(
        self,
        request: Mapping[str, Any],
        responder: Callable[[dict], None],
    ) -> None:
        """Collect two accepted frames after one ActionWorker request."""

        request_id = str(request.get("request_id") or "")
        if not self._pick_xy_active or self._pick_xy_session is None:
            responder(
                {
                    "request_id": request_id,
                    "decision": "abort",
                    "reason": "XY悬空定位会话未ARM",
                }
            )
            return
        if self._pick_xy_pending_request is not None:
            responder(
                {
                    "request_id": request_id,
                    "decision": "abort",
                    "reason": "已有未完成的XY悬空定位帧窗口",
                }
            )
            return
        if self._pick_xy_session.final_orientation_commanded:
            try:
                response = self._pick_xy_session.build_response(request, [])
            except Exception as exc:  # noqa: BLE001 - fail closed at UI boundary
                response = {
                    "request_id": request_id,
                    "decision": "abort",
                    "reason": f"最终J4方向到位复核失败：{exc}",
                }
            responder(response)
            self._last_interaction_text = str(
                response.get("reason") or "最终J4方向已完成到位复核"
            )
            self._refresh_status(self.runtime.snapshot())
            return
        self._pick_xy_pending_request = dict(request)
        self._pick_xy_pending_responder = responder
        self._pick_xy_samples.clear()
        self._pick_xy_rejection_counts.clear()
        self._last_interaction_text = (
            f"执行请求 {request_id}：等待请求之后的2张合格帧；此时尚未运动。"
        )
        self._refresh_status(self.runtime.snapshot())
        QTimer.singleShot(
            _PICK_XY_REQUEST_TIMEOUT_MS,
            lambda: self._pick_xy_timeout(request_id),
        )

    def _pick_xy_sample(self, frame: WaferTransferFrame) -> dict[str, Any]:
        snapshot = frame.session_snapshot
        gates = snapshot.get("selection_gates") or {}
        accepted = bool(
            frame.result.quality_passed
            and frame.result.coordinate_mapping_allowed
            and gates
            and all(
                isinstance(gate, Mapping) and gate.get("passed") is True
                for gate in gates.values()
            )
        )
        rejection_reasons: list[str] = []
        if not frame.result.quality_passed:
            rejection_reasons.append(
                frame.result.failure_reason or "托盘位姿质量门未通过"
            )
        elif not frame.result.coordinate_mapping_allowed:
            rejection_reasons.append(
                frame.result.failure_reason or "坐标映射暂不可用"
            )
        rejection_reasons.extend(
            str(name)
            for name, gate in gates.items()
            if not isinstance(gate, Mapping) or gate.get("passed") is not True
        )
        return {
            "measurement_id": f"camera1-sequence-{frame.frame_sequence}",
            "frame_sequence": int(frame.frame_sequence),
            "captured_monotonic_s": frame.captured_monotonic_s,
            "accepted": accepted,
            "rejection_reasons": rejection_reasons,
            "target_name": snapshot.get("source_slot"),
            "source_state": snapshot.get("source_state"),
            "source_consensus": snapshot.get("source_consensus"),
            "selection_gates": gates,
            "robot_state": snapshot.get("robot_state"),
            "registration": snapshot.get("registration"),
            "reprojection_rms_px": frame.result.pose.reprojection_rms_px,
            "used_marker_count": len(frame.result.pose.used_marker_ids),
            "annotated_bgr": frame.annotated_bgr.copy(),
        }

    def _eligible_pick_xy_samples(self) -> list[dict[str, Any]]:
        request = self._pick_xy_pending_request
        if request is None:
            return []
        requested_at = float(request.get("requested_monotonic_s") or math.inf)
        return list(
            select_bounded_observation_window(
                list(self._pick_xy_samples),
                requested_monotonic_s=requested_at,
            )
        )

    def _try_pick_xy_response(self) -> None:
        if self._pick_xy_pending_request is None or self._pick_xy_session is None:
            return
        samples = self._eligible_pick_xy_samples()
        if len(samples) < OBSERVATION_WINDOW_SIZE:
            return
        request = self._pick_xy_pending_request
        responder = self._pick_xy_pending_responder
        self._pick_xy_pending_request = None
        self._pick_xy_pending_responder = None
        self._pick_xy_samples.clear()
        self._pick_xy_rejection_counts.clear()
        try:
            response = self._pick_xy_session.build_response(request, samples)
        except Exception as exc:  # noqa: BLE001 - fail closed at UI boundary
            response = {
                "request_id": str(request.get("request_id") or ""),
                "decision": "abort",
                "reason": f"XY悬空定位计算失败：{exc}",
            }
        if callable(responder):
            responder(response)
        decision = str(response.get("decision") or "")
        if decision == "approve":
            proposal = response.get("proposal") or {}
            command = (proposal.get("calculation") or {}).get(
                "commanded_correction_xy_mm"
            ) or [0.0, 0.0]
            if proposal.get("phase") == "wafer_pick_final_tray_orientation":
                calculation = proposal.get("calculation") or {}
                self._last_interaction_text = (
                    "XY已到位，最终J4-only候选已交ActionWorker独立复核："
                    f"绝对Rz {float(calculation.get('current_absolute_rz_deg')):+.3f}° → "
                    f"{float(calculation.get('target_absolute_rz_deg')):+.3f}°。"
                )
            else:
                self._last_interaction_text = (
                    "本轮2帧通过，候选已交ActionWorker独立复核："
                    f"ΔX={float(command[0]):+.3f} mm，"
                    f"ΔY={float(command[1]):+.3f} mm。"
                )
        elif decision == "complete":
            self._last_interaction_text = str(
                response.get("reason") or "XY已到达目标正上方"
            )
        elif decision == "observe":
            self._last_interaction_text = str(
                response.get("reason") or "本轮画面不足，已跳过并自动重试"
            )
        else:
            self._last_interaction_text = "已停止：" + str(
                response.get("reason") or "安全门拒绝"
            )
        self._refresh_status(self.runtime.snapshot())

    def _pick_xy_timeout(self, request_id: str) -> None:
        request = self._pick_xy_pending_request
        if request is None or str(request.get("request_id") or "") != request_id:
            return
        responder = self._pick_xy_pending_responder
        rejection_summary = ", ".join(
            f"{reason}×{count}"
            for reason, count in self._pick_xy_rejection_counts.most_common(4)
        )
        self._pick_xy_pending_request = None
        self._pick_xy_pending_responder = None
        self._pick_xy_samples.clear()
        self._pick_xy_rejection_counts.clear()
        if callable(responder):
            responder(
                {
                    "request_id": request_id,
                    "decision": "observe",
                    "calibration_sha256": str(
                        request.get("calibration_sha256") or ""
                    ).upper(),
                    "reason": (
                        "本轮4秒内未获得2张间隔不超过1.75秒的合格相机1画面；"
                        "未运动，自动进入下一观察窗口"
                        + (
                            f"；拒绝统计：{rejection_summary}"
                            if rejection_summary
                            else ""
                        )
                    ),
                }
            )
        self._last_interaction_text = "本轮新鲜帧不足，未运动，正在自动重试。"
        self._refresh_status(self.runtime.snapshot())

    def finish_pick_xy_session(self, ok: bool, message: str) -> tuple[bool, str]:
        session = self._pick_xy_session
        self._pick_xy_active = False
        self._pick_xy_pending_request = None
        self._pick_xy_pending_responder = None
        self._pick_xy_samples.clear()
        self._pick_xy_rejection_counts.clear()
        self._pick_xy_session = None
        self._set_pick_xy_controls(False)
        if session is not None:
            ok, message = session.finish(ok, message)
        self._last_interaction_text = str(message)
        self._refresh_status(self.runtime.snapshot())
        return bool(ok), str(message)

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
        consensus = snapshot.get("source_consensus") or {}
        pending_evidence = (
            self._eligible_pick_xy_samples()
            if self._pick_xy_pending_request is not None
            else []
        )
        latest_rejection = ""
        if self._pick_xy_samples:
            latest_sample = self._pick_xy_samples[-1]
            if latest_sample.get("accepted") is not True:
                latest_rejection = "；最新忽略=" + ", ".join(
                    str(item)
                    for item in latest_sample.get("rejection_reasons") or ["未知质量门"]
                )
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
            extra or self._last_interaction_text or "点击画面中的正常硅片选择拾取目标。",
            "",
            f"选择模式：{'拾取槽' if self._selection_role == 'source' else '放置槽'}",
            f"阶段：{snapshot.get('phase')}",
            f"拾取槽：{snapshot.get('source_slot') or '—'} · {source.get('state', '—')}",
            f"放置槽：{snapshot.get('destination_slot') or '—'} · {destination.get('state', '—')}",
            f"当前目标：{snapshot.get('active_target_slot') or '—'}",
            (
                "拾取稳定证据："
                f"{consensus.get('occupied_frame_count', 0)}/"
                f"{consensus.get('window_frame_count', 2)} 帧正常 · "
                f"临时掉检 {consensus.get('dropout_frame_count', 0)}/"
                f"{consensus.get('maximum_dropout_frame_count', 2)}"
            ),
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
            (
                "XY悬空移动：ARMED（候选仍须ActionWorker复核）"
                if self._pick_xy_active
                else "XY悬空移动：未ARM"
            ),
            (
                f"本轮有效视觉证据：{len(pending_evidence)}/"
                f"{OBSERVATION_WINDOW_SIZE}{latest_rejection}"
                if self._pick_xy_pending_request is not None
                else "本轮有效视觉证据：未采集"
            ),
            (
                "硬限制：J3不下降；无吸取、真空或DO；XY阶段固定标定Rz，"
                "到位后仅J4对齐托盘方向。"
            ),
        ]
        registration_error = snapshot.get("registration_error")
        sync_error = snapshot.get("robot_state_sync_error")
        if registration_error:
            lines.extend(["", f"W←T状态：{registration_error}"])
        if sync_error:
            lines.extend(["", f"时间同步：{sync_error}"])
        self.status.setPlainText("\n".join(lines))
        self.status.verticalScrollBar().setValue(0)
        if not self._pick_xy_active:
            ready = bool(snapshot.get("tracking_ready"))
            self.track_button.setEnabled(ready)
            self.xy_motion_button.setEnabled(ready)
        else:
            self.xy_motion_button.setEnabled(True)

    def _show_error(self, message: str) -> None:
        snapshot = self.runtime.snapshot()
        self._last_interaction_text = f"当前操作未接受：{message}"
        self._refresh_status(snapshot)

    def _invalidate_current(self, message: str) -> None:
        self._last_frame = None
        self._stream_invalidated = True
        self.preview.clear_image("当前没有可用的新鲜相机1画面")
        self._last_interaction_text = f"当前判断已失效：{message}"
        if self._pick_xy_active:
            self._last_interaction_text += "；已请求XY悬空定位安全停止"
            self.pick_xy_stop_requested.emit()
        self._refresh_status(self.runtime.snapshot())

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._pick_xy_active:
            self.pick_xy_stop_requested.emit()
            self._show_error("XY悬空定位正在安全停止；执行线程结束后才能关闭窗口")
            event.ignore()
            return
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
