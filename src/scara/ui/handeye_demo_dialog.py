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
from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
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
QPlainTextEdit#stage7bStatus {
    color:#111827; background-color:#FFF7ED;
    border:1px solid #FDBA74; border-radius:6px; padding:7px;
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
        self.suction = load_latest_suction_target(self.project_root)
        self.jacobian_payload = load_local_xy_jacobian(
            self.project_root, self.suction, "P22"
        )
        geometry = load_tray_board_geometry(
            self.project_root / "src/scara/calib/tray_board_geometry.json"
        )
        self._last_image: Optional[QImage] = None
        self._last_evaluation: Optional[HandEyeEvaluation] = None
        self._last_evaluation_at: Optional[float] = None
        self._stage7b_session = None
        self._stage7b_pending_request: Optional[dict[str, Any]] = None
        self._stage7b_pending_responder: Optional[Callable[[dict], None]] = None
        self._stage7b_samples: deque[dict[str, Any]] = deque(maxlen=40)
        self._stage7b_active = False
        self._positioning_mode: Optional[str] = None

        if int(camera.source_index) != self.suction.camera_source:
            raise RuntimeError(
                f"动态演示要求相机源#{self.suction.camera_source}，"
                f"当前为源#{camera.source_index}"
            )

        self.setWindowTitle("手眼交互 · 动态演示（只计算 / 两种XY定位模式需ARM）")
        self.resize(1120, 860)
        self.setMinimumSize(860, 680)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setStyleSheet(_DIALOG_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(9)

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
        self.jacobian_payload = load_local_xy_jacobian(
            self.project_root, self.suction, target_name
        )
        self.monitor.set_jacobian_payload(self.jacobian_payload)
        self.monitor.set_target(target_name)

    def _on_evaluation(
        self, image: QImage, evaluation: HandEyeEvaluation
    ) -> None:
        self._last_image = image
        self._last_evaluation = evaluation
        self._last_evaluation_at = time.monotonic()
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
        note = (
            "当前判断已失效；不会授权新的XY定位运动。"
            if self._stage7b_active
            else "当前判断已失效；只计算，机械臂不会移动。"
        )
        self.status.setText(message + "\n" + note)

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
        super().closeEvent(event)


__all__ = ["HandEyeDemoDialog", "HandEyeMonitorThread"]
