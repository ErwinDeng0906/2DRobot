"""Read-only live hand-eye demonstration for camera 1."""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import cv2
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from scara.ui.dialogs import LIGHT_WARNING_DIALOG_STYLESHEET
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
from scara.vision.xy_image_jacobian import REQUIRED_XY_JACOBIAN_QUALITY_GATES


_DIALOG_STYLE = """
QDialog { background-color:#FFFFFF; color:#111827; }
QLabel { color:#111827; }
QLabel#safetyBanner {
    color:#991B1B; background-color:#FEE2E2;
    border:1px solid #FCA5A5; border-radius:6px;
    padding:9px; font-size:14px; font-weight:800;
}
QLabel#preview { background-color:#0B1220; border:1px solid #CBD5E1; }
QLabel#status {
    color:#111827; background-color:#F8FAFC;
    border:1px solid #CBD5E1; border-radius:6px; padding:8px;
    font-family:Consolas, "Microsoft YaHei";
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
QPushButton#validateButton {
    color:#FFFFFF; background-color:#2563EB; border-color:#2563EB;
}
QPushButton#validateButton:hover { background-color:#1D4ED8; }
"""


class HandEyeMonitorThread(QThread):
    """Run Stage-3 and overlay calculations without blocking the Qt UI."""

    frame_evaluated = pyqtSignal(QImage, object)
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
    ) -> None:
        super().__init__(parent)
        self.camera = camera
        self.project_root = Path(project_root)
        self.suction = suction
        self._target_name = target_name
        self._jacobian_payload = jacobian_payload
        self._robot_state_provider = robot_state_provider
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
                else:
                    frame, frame_sequence, _captured_at = packet
            else:
                # Compatibility for test doubles and older camera adapters.
                # Production camera1 provides a sequence-aware packet API.
                frame = self.camera.latest_frame(max_age_s=1.0)
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
            robot_state: Optional[Mapping[str, Any]] = None
            if self._robot_state_provider is not None:
                try:
                    robot_state = self._robot_state_provider()
                except Exception:  # noqa: BLE001 - fail closed, keep visual live
                    robot_state = None
            try:
                tracked = tracker.update(frame)
                evaluation = evaluate_handeye_frame(
                    frame,
                    tracked,
                    target_name,
                    geometry,
                    intrinsics,
                    self.suction,
                    jacobian_payload,
                    robot_state,
                )
                rgb = cv2.cvtColor(evaluation.annotated_bgr, cv2.COLOR_BGR2RGB)
                height, width, channels = rgb.shape
                image = QImage(
                    rgb.data,
                    width,
                    height,
                    channels * width,
                    QImage.Format.Format_RGB888,
                ).copy()
                self.frame_evaluated.emit(image, evaluation)
                last_error = ""
            except Exception as exc:  # noqa: BLE001
                message = f"实时手眼计算失败：{exc}"
                if message != last_error:
                    self.frame_invalidated.emit(message)
                    last_error = message
            self.msleep(100)


class HandEyeDemoDialog(QDialog):
    """Live slot-vs-suction visualization with no motion API."""

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
        self.suction = load_latest_suction_target(self.project_root)
        self.jacobian_payload = load_local_xy_jacobian(
            self.project_root, self.suction
        )
        geometry = load_tray_board_geometry(
            self.project_root / "src/scara/calib/tray_board_geometry.json"
        )
        self._last_image: Optional[QImage] = None
        self._last_evaluation: Optional[HandEyeEvaluation] = None
        self._last_evaluation_at: Optional[float] = None

        if int(camera.source_index) != self.suction.camera_source:
            raise RuntimeError(
                f"动态演示要求相机源#{self.suction.camera_source}，"
                f"当前为源#{camera.source_index}"
            )

        self.setWindowTitle("手眼交互 · 动态演示（只计算）")
        self.resize(1120, 860)
        self.setMinimumSize(860, 680)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setStyleSheet(_DIALOG_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(9)

        safety = QLabel(
            "只计算，机械臂不会移动。动态演示和“Jacobian验证”按钮都不会发送任何运动、DO或真空命令。"
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
        self.validation_button = QPushButton("Jacobian验证")
        self.validation_button.setObjectName("validateButton")
        controls.addWidget(self.validation_button)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        controls.addWidget(close_button)
        layout.addLayout(controls)

        legend = QLabel(
            "绿色十字＝指定槽中心　红色十字＝suction target　黄色箭头＝当前图像误差（红→绿）　"
            "青色圆点＝A–H重投影角点　T-X/T-Y/T-Z＝托盘坐标轴"
        )
        legend.setWordWrap(True)
        layout.addWidget(legend)

        self.preview = QLabel("等待相机1实时画面与Stage3有效位姿……")
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(430)
        layout.addWidget(self.preview, 1)

        self.status = QLabel()
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)

        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        self.validation_button.clicked.connect(self._show_jacobian_validation)
        self.monitor = HandEyeMonitorThread(
            camera,
            self.project_root,
            self.suction,
            self.target_combo.currentText(),
            self.jacobian_payload,
            self.robot_state_provider,
            self,
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
        self.status.setText(
            f"Task8：{self.suction.source_path}\n"
            f"Stage5局部Jacobian：{jacobian_text}\n"
            "状态：等待相机1新鲜帧。"
        )

    def _on_target_changed(self, target_name: str) -> None:
        self._invalidate_current("目标已切换，等待该槽的新鲜Stage3结果")
        self.monitor.set_target(target_name)

    def _on_evaluation(
        self, image: QImage, evaluation: HandEyeEvaluation
    ) -> None:
        self._last_image = image
        self._last_evaluation = evaluation
        self._last_evaluation_at = time.monotonic()
        self._refresh_preview()
        rms = (
            "—"
            if evaluation.reprojection_rms_px is None
            else f"{evaluation.reprojection_rms_px:.3f} px"
        )
        if not evaluation.accepted:
            self.status.setText(
                f"目标：{evaluation.target_name}\n"
                f"Stage3：REJECT — {evaluation.reason}\n"
                f"A–H：可见{evaluation.visible_marker_count}，使用{evaluation.used_marker_count}；RMS={rms}\n"
                "当前帧不用于槽位判断；只计算，机械臂不会移动。"
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
            "未向机械臂发送任何命令。"
        )

    def _refresh_preview(self) -> None:
        if self._last_image is None:
            return
        self.preview.setPixmap(
            QPixmap.fromImage(self._last_image).scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_preview()

    def _on_monitor_error(self, message: str) -> None:
        self._invalidate_current(message)

    def _invalidate_current(self, message: str) -> None:
        """Clear stale PASS imagery so it cannot be mistaken for live state."""
        self._last_image = None
        self._last_evaluation = None
        self._last_evaluation_at = None
        self.preview.setPixmap(QPixmap())
        self.preview.setText("当前没有可用的新鲜相机1 / Stage3结果")
        self.status.setText(message + "\n当前判断已失效；只计算，机械臂不会移动。")

    def _reload_jacobian(self) -> Optional[dict[str, Any]]:
        self.jacobian_payload = load_local_xy_jacobian(
            self.project_root, self.suction
        )
        self.monitor.set_jacobian_payload(self.jacobian_payload)
        return self.jacobian_payload

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

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if hasattr(self, "monitor") and self.monitor.isRunning():
            if not self.monitor.stop():
                event.ignore()
                self._invalidate_current(
                    "后台Stage3计算尚未安全退出，请稍候再关闭窗口"
                )
                return
        super().closeEvent(event)


__all__ = ["HandEyeDemoDialog", "HandEyeMonitorThread"]
