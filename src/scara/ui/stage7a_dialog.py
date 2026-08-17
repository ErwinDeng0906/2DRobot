"""White-background operator guide for the supervised Stage-7A single step."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from PyQt6.QtCore import QEventLoop, Qt
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


STYLE = """
QDialog, QWidget { background:#FFFFFF; color:#111827; }
QLabel { color:#111827; }
QLabel#title { font-size:18px; font-weight:700; color:#111827; }
QLabel#safety { background:#FFF7ED; color:#9A3412; border:1px solid #FDBA74;
                border-radius:6px; padding:8px; font-weight:700; }
QLabel#preview { background:#111827; color:#F9FAFB; border:1px solid #9CA3AF; }
QTableWidget { background:#FFFFFF; color:#111827; gridline-color:#D1D5DB;
               selection-background-color:#DBEAFE; selection-color:#111827; }
QHeaderView::section { background:#F3F4F6; color:#111827; padding:5px;
                       border:1px solid #D1D5DB; font-weight:700; }
QTextEdit { background:#F9FAFB; color:#111827; border:1px solid #D1D5DB;
            border-radius:4px; padding:5px; }
QPushButton { color:#111827; background:#F3F4F6; border:1px solid #9CA3AF;
              border-radius:5px; padding:8px 16px; min-width:130px; }
QPushButton:hover { background:#E5E7EB; }
QPushButton:disabled { color:#9CA3AF; background:#F9FAFB; }
QPushButton#approve { background:#FDE68A; border:2px solid #D97706; font-weight:700; }
QPushButton#approve:hover { background:#FCD34D; }
QPushButton#decline { background:#E5E7EB; font-weight:700; }
"""


def _number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError, OverflowError):
        return "—"


class Stage7AOperatorDialog(QDialog):
    """Live acquisition table plus one explicit, default-deny decision."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Stage 7A — P22人工确认单步视觉修正")
        self.resize(1240, 830)
        self.setStyleSheet(STYLE)
        self._decision: Optional[bool] = None
        self._decision_loop: Optional[QEventLoop] = None
        self._finished = False

        outer = QVBoxLayout(self)
        title = QLabel("Stage 7A：P22人工确认的一次受限XY修正")
        title.setObjectName("title")
        outer.addWidget(title)
        safety = QLabel(
            "只允许一次不超过0.25 mm的XY修正；固定J3和末端Rz；"
            "不下降、不触发DO/真空。查看期间严禁触碰托盘、相机或机械臂；"
            "候选20秒后失效，默认选择是不运动。"
        )
        safety.setObjectName("safety")
        safety.setWordWrap(True)
        outer.addWidget(safety)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        self.preview = QLabel("等待相机1修正前画面……")
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(690, 390)
        preview_layout.addWidget(self.preview, 1)
        self.live_status = QLabel("准备中：尚未取得修正前稳定窗口")
        self.live_status.setWordWrap(True)
        preview_layout.addWidget(self.live_status)
        splitter.addWidget(preview_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("每帧检测与途径点"))
        self.frame_table = QTableWidget(0, 10)
        self.frame_table.setHorizontalHeaderLabels(
            [
                "阶段",
                "图片",
                "途径点X(mm)",
                "途径点Y(mm)",
                "Stage3",
                "Marker",
                "RMS(px)",
                "eu(px)",
                "ev(px)",
                "|e|(px)",
            ]
        )
        self.frame_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.frame_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.frame_table.verticalHeader().setVisible(False)
        self.frame_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.frame_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.frame_table, 1)
        right_layout.addWidget(QLabel("候选修正与预计终点"))
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(160)
        right_layout.addWidget(self.summary)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter, 3)

        outer.addWidget(QLabel("全部Stage7A安全门"))
        self.gate_table = QTableWidget(0, 4)
        self.gate_table.setHorizontalHeaderLabels(["安全门", "结果", "实测", "限制/说明"])
        self.gate_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.gate_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.gate_table.verticalHeader().setVisible(False)
        self.gate_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.gate_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.gate_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.gate_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        outer.addWidget(self.gate_table, 2)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.decline_button = QPushButton("不运动，继续保存复测数据")
        self.decline_button.setObjectName("decline")
        self.approve_button = QPushButton("确认执行一次XY修正")
        self.approve_button.setObjectName("approve")
        self.approve_button.setEnabled(False)
        self.decline_button.setEnabled(False)
        # Pressing Enter must take the non-motion path.  Do not let Qt infer
        # the affirmative button as the dialog default from focus/order.
        self.decline_button.setAutoDefault(True)
        self.decline_button.setDefault(True)
        self.approve_button.setAutoDefault(False)
        self.approve_button.setDefault(False)
        self.decline_button.clicked.connect(lambda: self._finish_decision(False))
        self.approve_button.clicked.connect(lambda: self._finish_decision(True))
        buttons.addWidget(self.decline_button)
        buttons.addWidget(self.approve_button)
        outer.addLayout(buttons)

    def add_frame(self, record: Mapping[str, Any], image: Optional[QImage]) -> None:
        if image is not None and not image.isNull():
            pixmap = QPixmap.fromImage(image)
            self.preview.setPixmap(
                pixmap.scaled(
                    self.preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        row = self.frame_table.rowCount()
        self.frame_table.insertRow(row)
        error = record.get("image_error_px")
        norm = record.get("image_error_norm_px")
        robot_pose = record.get("robot_pose")
        robot_x = (
            robot_pose[0]
            if isinstance(robot_pose, (list, tuple)) and len(robot_pose) >= 2
            else None
        )
        robot_y = (
            robot_pose[1]
            if isinstance(robot_pose, (list, tuple)) and len(robot_pose) >= 2
            else None
        )
        values = [
            str(record.get("phase") or "—"),
            str(record.get("filename") or "—"),
            _number(robot_x),
            _number(robot_y),
            "PASS" if record.get("accepted") else "REJECT",
            f"{record.get('used_marker_count', 0)}/{record.get('visible_marker_count', 0)}",
            _number(record.get("reprojection_rms_px")),
            _number(error[0]) if isinstance(error, (list, tuple)) and len(error) == 2 else "—",
            _number(error[1]) if isinstance(error, (list, tuple)) and len(error) == 2 else "—",
            _number(norm),
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column == 4:
                item.setForeground(Qt.GlobalColor.darkGreen if record.get("accepted") else Qt.GlobalColor.red)
            self.frame_table.setItem(row, column, item)
        self.frame_table.scrollToBottom()
        self.live_status.setText(
            f"已处理 {row + 1} 张：{values[0]} / {values[1]} / "
            f"XY=({values[2]},{values[3]}) mm / Stage3 {values[4]} / "
            f"e=({values[7]},{values[8]}) px"
        )

    def set_proposal(self, proposal: Mapping[str, Any], plan: Optional[Mapping[str, Any]] = None) -> None:
        calculation = proposal.get("calculation") or {}
        measurement = proposal.get("measurement") or {}
        error = measurement.get("median_error_px")
        raw = calculation.get("full_cancellation_correction_xy_mm")
        command = calculation.get("commanded_correction_xy_mm")
        endpoint = calculation.get("predicted_endpoint_xy_mm")
        predicted_error = calculation.get("predicted_error_px")
        plan_audit = (plan or {}).get("audit") or {}
        rz_precompensation = plan_audit.get("rz_precompensation") or {}
        lines = [
            f"修正前途径点 world XY = {measurement.get('current_robot_xy_mm', '—')} mm",
            f"修正前中位误差 e = {error if error is not None else '—'} px",
            f"完整理论修正 -J^-1e = {raw if raw is not None else '—'} mm",
            f"本次增益/限幅后 ΔX,ΔY = {command if command is not None else '—'} mm",
            f"预计终点 world XY = {endpoint if endpoint is not None else '—'} mm",
            f"预计修正后误差 = {predicted_error if predicted_error is not None else '—'} px",
            f"是否发生0.25 mm向量限幅：{'是' if calculation.get('was_clamped') else '否'}",
        ]
        if plan_audit:
            lines.extend(
                [
                    (
                        "J4 Rz预补偿 = "
                        f"{_number(rz_precompensation.get('current_j4_deg'))}° -> "
                        f"{_number(rz_precompensation.get('target_j4_deg'))}° "
                        f"(ΔJ4={_number(rz_precompensation.get('delta_j4_deg'))}°)"
                    ),
                    f"IK目标关节 = {(plan or {}).get('target_joints')}",
                    f"逐轴执行最大瞬时XY段 = {_number(plan_audit.get('sequential_transient_max_mm'))} mm",
                ]
            )
        self.summary.setPlainText("\n".join(lines))

        combined: list[tuple[str, Mapping[str, Any]]] = []
        for name, gate in (proposal.get("safety_gates") or {}).items():
            if isinstance(gate, Mapping):
                combined.append((str(name), gate))
        for name, gate in (plan_audit.get("gates") or {}).items():
            if isinstance(gate, Mapping):
                combined.append((f"planner.{name}", gate))
        self.gate_table.setRowCount(0)
        for name, gate in combined:
            row = self.gate_table.rowCount()
            self.gate_table.insertRow(row)
            passed = gate.get("passed") is True
            # The first proposal is intentionally built with consent=False:
            # the operator has not been given the button yet.  This is an
            # unresolved human authorization, not a failed technical check.
            # Consent remains a hard motion gate and is rebuilt as PASS only
            # after the affirmative button is clicked.
            pending_consent = name == "operator_consent" and not passed
            actual = gate.get("actual")
            limit = gate.get("limit")
            note = str(gate.get("note") or "")
            items = [
                QTableWidgetItem(name),
                QTableWidgetItem(
                    "PENDING" if pending_consent else ("PASS" if passed else "FAIL")
                ),
                QTableWidgetItem(
                    "waiting for explicit operator confirmation"
                    if pending_consent
                    else str(actual)
                ),
                QTableWidgetItem(f"{limit} {note}".strip()),
            ]
            items[1].setForeground(
                Qt.GlobalColor.darkYellow
                if pending_consent
                else (Qt.GlobalColor.darkGreen if passed else Qt.GlobalColor.red)
            )
            for column, item in enumerate(items):
                self.gate_table.setItem(row, column, item)

    def request_decision(self, proposal: Mapping[str, Any], plan: Optional[Mapping[str, Any]]) -> bool:
        self.set_proposal(proposal, plan)
        ready = bool(
            proposal.get("ready_for_operator_confirmation")
            and proposal.get("motion_required")
            and plan is not None
            and (plan.get("audit") or {}).get("passed") is True
        )
        self._decision = None
        self.approve_button.setEnabled(ready)
        self.decline_button.setEnabled(True)
        self.live_status.setText(
            "除操作员确认外的安全门均通过，请核对数值后选择；默认是不运动。"
            if ready
            else "安全门未全部通过或无需运动；确认按钮已禁用。"
        )
        self.show()
        self.raise_()
        self.activateWindow()
        loop = QEventLoop(self)
        self._decision_loop = loop
        loop.exec()
        self._decision_loop = None
        self.approve_button.setEnabled(False)
        self.decline_button.setEnabled(False)
        return self._decision is True

    def _finish_decision(self, approved: bool) -> None:
        self._decision = bool(approved)
        if self._decision_loop is not None and self._decision_loop.isRunning():
            self._decision_loop.quit()

    def set_final_text(self, text: str) -> None:
        self._finished = True
        self.live_status.setText(str(text))
        self.approve_button.setEnabled(False)
        self.decline_button.setEnabled(False)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        if self._decision_loop is not None and self._decision_loop.isRunning():
            self._finish_decision(False)
            event.ignore()
            self.hide()
            return
        event.accept()


__all__ = ["Stage7AOperatorDialog"]
