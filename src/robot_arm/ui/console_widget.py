# -*- coding: utf-8 -*-
"""机械臂控制台主界面 widget —— 纯 UI + 控制钩子（不含业务逻辑）。

严格照效果图 tools/_shots/arm_console_v2.png（其 HTML 源在 arm_console_mockup.html）
1:1 还原的三栏全平铺布局：
  - 顶部工具栏：品牌 / IP·端口 / 连接·上电·使能·下使能·急停 / 仿真·远程拨动开关
  - 左栏（~280px）：状态监控（连接/安全/使能/报警 胶囊 + 模式/速度 + J1~J6 进度条 + TCP）
  - 中栏（弹性）：相机取景区 + 手动 JOG（分段切换 + 3×4 发光按键网格）
  - 右栏（~320px）：示教点位 + 序列回放 + 运动禁区

设计语言：深色工业玻璃（见 theme.APP_STYLESHEET）。所有交互控件均暴露为 self.xxx，
JOG 统一通过 jog_clicked(joint_index:int, sign:int) 信号对外。
"""
from __future__ import annotations

import os
import sys

from PyQt6.QtCore import Qt, QRectF, QSize, pyqtSignal, pyqtProperty, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QFont, QPixmap, QLinearGradient
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit, QSpinBox, QComboBox,
    QHBoxLayout, QVBoxLayout, QGridLayout, QFrame, QProgressBar, QSlider,
    QListWidget, QListWidgetItem, QSizePolicy, QButtonGroup,
)

# ── 主题：尽量复用 theme.py（直接运行时也能 fallback）──────────────────────
try:
    from robot_arm.ui import theme as _T
except Exception:  # pragma: no cover - 直接运行时的 fallback
    try:
        from . import theme as _T  # type: ignore
    except Exception:
        _T = None  # type: ignore

if _T is not None:
    APP_STYLESHEET = _T.APP_STYLESHEET
    BG_DARK = _T.BG_DARK
    BG_DARK2 = _T.BG_DARK2
    PRIMARY_COLOR = _T.PRIMARY_COLOR
    ACCENT_COLOR = _T.ACCENT_COLOR
    SUCCESS_COLOR = _T.SUCCESS_COLOR
    WARNING_COLOR = _T.WARNING_COLOR
    ERROR_COLOR = _T.ERROR_COLOR
    TEXT_PRIMARY = _T.TEXT_PRIMARY
    TEXT_SECONDARY = _T.TEXT_SECONDARY
    TEXT_MUTED = _T.TEXT_MUTED
    BORDER_COLOR = _T.BORDER_COLOR
    MONO = _T.MONO
else:  # pragma: no cover
    APP_STYLESHEET = ""
    BG_DARK, BG_DARK2 = "#0A0F16", "#0C121B"
    PRIMARY_COLOR, ACCENT_COLOR = "#4A8FE0", "#5FD4E0"
    SUCCESS_COLOR, WARNING_COLOR, ERROR_COLOR = "#5BD6A6", "#E8B968", "#E8707E"
    TEXT_PRIMARY = "#D6E2EF"
    TEXT_SECONDARY = "rgba(180, 198, 216, 0.62)"
    TEXT_MUTED = "rgba(150, 170, 190, 0.40)"
    BORDER_COLOR = "rgba(120, 160, 200, 0.14)"
    MONO = '"Consolas", monospace'


# 关节轴中文名（底座/大臂/小臂/腕转/腕摆/法兰）
JOINT_AXIS_NAMES = ["底座", "大臂", "小臂", "腕转", "腕摆", "法兰"]
# 笛卡尔 CART 模式下 6 个槽位复用同一按键网格：X/Y/Z 平移 + Rx/Ry/Rz 姿态
CART_AXIS_NAMES = ["X", "Y", "Z", "Rx", "Ry", "Rz"]
CART_AXIS_HINT = ["平移", "平移", "平移", "姿态", "姿态", "姿态"]


# ════════════════════════════════════════════════════════════════════════════
# 自定义控件
# ════════════════════════════════════════════════════════════════════════════
class ToggleSwitch(QWidget):
    """圆点滑动式拨动开关（非普通 checkbox）。

    paintEvent 画圆角槽 + 滑动圆点；点击切换并发 clicked / toggled 信号。
    """

    clicked = pyqtSignal()
    toggled = pyqtSignal(bool)

    def __init__(self, checked: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self._checked = checked
        self._offset = 1.0 if checked else 0.0
        self._w, self._h, self._pad = 34, 18, 2
        self.setFixedSize(self._w, self._h)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._anim = QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    # 动画属性 ----------------------------------------------------------------
    def _get_offset(self) -> float:
        return self._offset

    def _set_offset(self, v: float) -> None:
        self._offset = v
        self.update()

    offset = pyqtProperty(float, _get_offset, _set_offset)

    # 状态 --------------------------------------------------------------------
    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool) -> None:
        checked = bool(checked)
        if checked == self._checked:
            return
        self._checked = checked
        self._anim.stop()
        self._anim.setStartValue(self._offset)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()
        self.toggled.emit(checked)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.setChecked(not self._checked)
            self.clicked.emit()
        super().mousePressEvent(ev)

    def sizeHint(self) -> QSize:
        return QSize(self._w, self._h)

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self._h / 2.0
        # 槽
        if self._checked:
            track = QColor(95, 212, 224, 128)   # 青 50%
        else:
            track = QColor(90, 110, 130, 102)    # 灰 40%
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(track))
        p.drawRoundedRect(QRectF(0, 0, self._w, self._h), r, r)
        # 圆点
        knob_d = self._h - self._pad * 2
        x0 = self._pad
        x1 = self._w - self._pad - knob_d
        x = x0 + (x1 - x0) * self._offset
        p.setBrush(QBrush(QColor(255, 255, 255, 217)))
        p.drawEllipse(QRectF(x, self._pad, knob_d, knob_d))
        p.end()


class _CameraArea(QWidget):
    """相机取景占位区：深色取景背景 + 四角青取景框 + 四角玻璃 HUD 角标。

    留 set_pixmap(QPixmap) 与 set_hud(res, fps, status, aruco) 钩子。
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(436)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap: QPixmap | None = None
        # HUD 字段（先画死示意值）
        self._hud_res = "末端 USB · 640×480"
        self._hud_fps = "25 fps"
        self._hud_status = "REC"        # REC / IDLE
        self._hud_aruco = 3
        self._hud_latency = "~120"

    # 钩子 --------------------------------------------------------------------
    def set_pixmap(self, pixmap: QPixmap | None) -> None:
        self._pixmap = pixmap
        self.update()

    def set_hud(self, res: str | None = None, fps: str | None = None,
                status: str | None = None, aruco: int | None = None,
                latency: str | None = None) -> None:
        if res is not None:
            self._hud_res = res
        if fps is not None:
            self._hud_fps = fps
        if status is not None:
            self._hud_status = status
        if aruco is not None:
            self._hud_aruco = aruco
        if latency is not None:
            self._hud_latency = latency
        self.update()

    # 绘制 --------------------------------------------------------------------
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        rect = QRectF(0, 0, w - 1, h - 1)
        radius = 12.0

        # 取景背景（径向渐变模拟）
        if self._pixmap is not None and not self._pixmap.isNull():
            p.save()
            from PyQt6.QtGui import QPainterPath
            path = QPainterPath()
            path.addRoundedRect(rect, radius, radius)
            p.setClipPath(path)
            p.fillRect(rect, QColor(0x0C, 0x14, 0x1D))
            scaled = self._pixmap.scaled(
                int(w), int(h), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            x = (w - scaled.width()) // 2
            y = (h - scaled.height()) // 2
            p.drawPixmap(x, y, scaled)
            p.restore()
        else:
            grad = QLinearGradient(0, 0, 0, h)
            grad.setColorAt(0.0, QColor(0x24, 0x34, 0x43))
            grad.setColorAt(0.82, QColor(0x0C, 0x14, 0x1D))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(rect, radius, radius)
            # 斜纹叠层
            p.save()
            p.setClipRect(rect)
            pen = QPen(QColor(255, 255, 255, 8))
            pen.setWidth(8)
            p.setPen(pen)
            step = 26
            x = -h
            while x < w:
                p.drawLine(int(x), h, int(x + h), 0)
                x += step
            p.restore()
            self._draw_tray_placeholder(p, w, h)

        # 边框
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(120, 160, 200, 36), 1))
        p.drawRoundedRect(rect, radius, radius)

        p.end()

    def _draw_tray_placeholder(self, p: QPainter, w: int, h: int):
        # 中部硅片盘示意 + 文字
        tray_w, tray_h = 200, 140
        tx = (w - tray_w) / 2
        ty = (h - tray_h) / 2 - 8
        grad = QLinearGradient(0, ty, 0, ty + tray_h)
        grad.setColorAt(0.0, QColor(40, 55, 72, 128))
        grad.setColorAt(1.0, QColor(22, 32, 44, 128))
        p.setPen(QPen(QColor(120, 160, 200, 56), 1))
        p.setBrush(QBrush(grad))
        p.drawRoundedRect(QRectF(tx, ty, tray_w, tray_h), 10, 10)
        # 4×3 格子，部分 aruco 高亮
        aruco_cells = {0, 2, 5, 11}
        cols, rows, gap, pad = 4, 3, 6, 12
        cw = (tray_w - pad * 2 - gap * (cols - 1)) / cols
        ch = (tray_h - pad * 2 - gap * (rows - 1)) / rows
        idx = 0
        for r in range(rows):
            for c in range(cols):
                cx = tx + pad + c * (cw + gap)
                cy = ty + pad + r * (ch + gap)
                cell = QRectF(cx, cy, cw, ch)
                if idx in aruco_cells:
                    p.setPen(QPen(QColor(95, 212, 224, 128), 1))
                    p.setBrush(QBrush(QColor(95, 212, 224, 16)))
                else:
                    pen = QPen(QColor(120, 160, 200, 46), 1, Qt.PenStyle.DashLine)
                    p.setPen(pen)
                    p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(cell, 4, 4)
                idx += 1
        # 字幕
        p.setPen(QColor(150, 170, 190, 130))
        f = QFont("Consolas", 9)
        p.setFont(f)
        p.drawText(QRectF(0, ty + tray_h + 8, w, 20),
                   Qt.AlignmentFlag.AlignHCenter, "硅片盘 · 末端相机视角")

    # (四角青取景框 + HUD 角标已按用户要求移除，保留 set_hud 钩子为空操作以兼容调用方)


# ════════════════════════════════════════════════════════════════════════════
# 小工具：构造卡片 / 标题 / 行
# ════════════════════════════════════════════════════════════════════════════
def _card(title: str) -> tuple[QFrame, QVBoxLayout]:
    """返回 (卡片 frame, 内容布局)。卡片含标题头与内容区。

    ★ 原本卡头标题前有个图标参数（◉ ⊹ ⚑），2026-07-28 去掉：其余三页
      （自动流程 / SCARA / 龙门）卡头都只有标题，去掉图标这四页才真正一套视觉。
      （那三个字符本身在真实 Windows 平台渲染正常 —— 去掉是为了统一，不是因为显示不出来。）
    """
    card = QFrame()
    card.setProperty("cssClass", "card")
    card.setObjectName("glassCard")
    outer = QVBoxLayout(card)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    # 标题头
    head = QFrame()
    head.setObjectName("cardHead")
    hl = QHBoxLayout(head)
    hl.setContentsMargins(14, 10, 14, 10)
    hl.setSpacing(8)
    tl = QLabel(title)
    tl.setObjectName("cardTitle")
    hl.addWidget(tl)
    hl.addStretch(1)
    outer.addWidget(head)
    # 内容
    body = QFrame()
    body.setObjectName("cardBody")
    bl = QVBoxLayout(body)
    bl.setContentsMargins(14, 12, 14, 12)
    bl.setSpacing(9)
    outer.addWidget(body, 1)
    return card, bl


def _section(text: str) -> QLabel:
    lab = QLabel(text.upper())
    lab.setObjectName("section")
    return lab


def _mini(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("mini")
    return lab


# ════════════════════════════════════════════════════════════════════════════
# 主 widget
# ════════════════════════════════════════════════════════════════════════════
class ArmConsoleWidget(QWidget):
    """机械臂控制台主界面（纯 UI）。"""

    # JOG 统一信号：joint_index 0..5，sign -1/+1
    jog_clicked = pyqtSignal(int, int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("armRoot")
        self.jog_buttons: dict[tuple[int, int], QPushButton] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)

        root.addWidget(self._build_topbar())

        body = QHBoxLayout()
        body.setSpacing(14)
        body.addWidget(self._build_left(), 0)
        body.addWidget(self._build_center(), 1)
        body.addWidget(self._build_right(), 0)
        root.addLayout(body, 1)

        # 底部状态栏（显示操作反馈：连接/记录/前往/错误等）
        statusbar = QFrame()
        statusbar.setObjectName("statusBar")
        sl = QHBoxLayout(statusbar)
        sl.setContentsMargins(14, 6, 14, 6)
        sl.setSpacing(8)
        self.status_dot = QLabel("●")
        self.status_dot.setObjectName("statusDot")
        self.status_msg = QLabel("就绪")
        self.status_msg.setObjectName("statusMsg")
        sl.addWidget(self.status_dot)
        sl.addWidget(self.status_msg)
        sl.addStretch(1)
        root.addWidget(statusbar)

        self.setStyleSheet(APP_STYLESHEET + _EXTRA_QSS)

    # ── 顶部工具栏 ──────────────────────────────────────────────────────────
    def _build_topbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("topbar")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 9, 16, 9)
        lay.setSpacing(9)

        # 品牌
        brand = QFrame()
        bl = QHBoxLayout(brand)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(8)
        dot = QLabel("●")
        dot.setObjectName("brandDot")
        title = QLabel("机械臂控制")
        title.setObjectName("brandTitle")
        sub = QLabel("DUCO GCR3-618")
        sub.setObjectName("brandSub")
        bl.addWidget(dot)
        bl.addWidget(title)
        bl.addWidget(sub)
        lay.addWidget(brand)

        sep = QFrame()
        sep.setObjectName("vsep")
        sep.setFixedSize(1, 20)
        lay.addWidget(sep)

        # UI 回退值；ArmConsoleControlWidget 会用 local_config.toml [duco] 初始化。
        # 用户仍可在连接前手动修改。
        lab_ip = QLabel("IP")
        lab_ip.setObjectName("fieldLab")
        self.ip_edit = QLineEdit("192.168.1.10")
        self.ip_edit.setFixedWidth(112)
        lab_port = QLabel("端口")
        lab_port.setObjectName("fieldLab")
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(7003)
        self.port_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.port_spin.setFixedWidth(62)
        lay.addWidget(lab_ip)
        lay.addWidget(self.ip_edit)
        lay.addWidget(lab_port)
        lay.addWidget(self.port_spin)

        # 动作按钮
        self.btn_connect = QPushButton("连接")
        self.btn_connect.setProperty("cssClass", "primary")
        self.btn_disconnect = QPushButton("断开")
        self.btn_power = QPushButton("上电")
        self.btn_power.setProperty("cssClass", "success")
        self.btn_enable = QPushButton("使能")
        self.btn_disable = QPushButton("下使能")
        self.btn_estop = QPushButton("⏻ 急停")
        self.btn_estop.setProperty("cssClass", "danger")
        for b in (self.btn_connect, self.btn_power, self.btn_enable,
                  self.btn_disable, self.btn_estop):
            lay.addWidget(b)

        lay.addStretch(1)

        # 服务器地址（经服务器代理控制真机；可见，方便确认/修改）
        lay.addWidget(self._mk_lbl("服务器", "fieldLab"))
        # 本机就地起 armweb 服务(webconsole/server.py --port 8080)连本地交换机上的真机，
        # 故默认 127.0.0.1:8080。若改用远端代理服务器，把地址改成 http://<服务器IP>:8080。
        self.server_edit = QLineEdit("http://127.0.0.1:8080")
        self.server_edit.setFixedWidth(180)
        self.server_edit.setToolTip("经此服务器代理连接并控制真机；本机 armweb 默认 127.0.0.1:8080")
        lay.addWidget(self.server_edit)

        # 兼容旧逻辑的隐藏开关（已不显示；远程恒为 True，仿真恒为 False）
        self.sw_remote = ToggleSwitch(checked=True)
        self.sw_remote.setVisible(False)
        self.sw_sim = ToggleSwitch(checked=False)
        self.sw_sim.setVisible(False)

        return bar

    def _toggle_box(self, text: str, checked: bool) -> QFrame:
        box = QFrame()
        box.setObjectName("toggleBox")
        l = QHBoxLayout(box)
        l.setContentsMargins(10, 5, 10, 5)
        l.setSpacing(6)
        sw = ToggleSwitch(checked=checked)
        lab = QLabel(text)
        lab.setObjectName("toggleLab")
        l.addWidget(sw)
        l.addWidget(lab)
        return box

    def _mk_lbl(self, text: str, obj: str) -> QLabel:
        lab = QLabel(text)
        lab.setObjectName(obj)
        return lab

    # ── 左栏 状态监控 ───────────────────────────────────────────────────────
    def _build_left(self) -> QWidget:
        card, bl = _card("状态监控")
        card.setFixedWidth(280)

        # 胶囊行
        self.chip_conn = self._chip("ONLINE", "ok")
        self.chip_safe = self._chip("RUN", "run")
        self.chip_enable = self._chip("ENABLED", "ok")
        self.chip_alarm = self._chip("无", "ok")
        bl.addWidget(self._stat_row("连接", self.chip_conn))
        bl.addWidget(self._stat_row("安全状态", self.chip_safe))
        bl.addWidget(self._stat_row("使能", self.chip_enable))
        bl.addWidget(self._stat_row("报警", self.chip_alarm))

        # 键值行
        self.lbl_mode = self._value("Remote")
        bl.addWidget(self._stat_row("模式", self.lbl_mode))

        # 关节角度
        bl.addWidget(_section("关节角度 (°)"))
        self.joint_bars: list[QProgressBar] = []
        self.joint_vals: list[QLabel] = []
        demo_pct = [42, 71, 30, 51, 74, 52]
        demo_deg = ["-87.73", "82.13", "-81.31", "0.87", "85.74", "1.99"]
        joints_wrap = QVBoxLayout()
        joints_wrap.setSpacing(9)
        for i in range(6):
            row = QHBoxLayout()
            row.setSpacing(9)
            lbl = QLabel(f"J{i+1}")
            lbl.setObjectName("jlbl")
            lbl.setFixedWidth(24)
            bar = QProgressBar()
            bar.setRange(0, 360)
            bar.setTextVisible(False)
            bar.setFixedHeight(7)
            bar.setValue(int(demo_pct[i] / 100 * 360))
            deg = QLabel(demo_deg[i])
            deg.setObjectName("jdeg")
            deg.setFixedWidth(62)
            deg.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(lbl)
            row.addWidget(bar, 1)
            row.addWidget(deg)
            joints_wrap.addLayout(row)
            self.joint_bars.append(bar)
            self.joint_vals.append(deg)
        bl.addLayout(joints_wrap)

        # TCP 位姿
        bl.addWidget(_section("TCP 位姿 (mm / °)"))
        self.tcp_vals: list[QLabel] = []  # [X,Y,Z,Rx,Ry,Rz]
        xyz = self._value("81 · 334 · 282")
        rxyz = self._value("178 · 1 · 90")
        bl.addWidget(self._stat_row("X / Y / Z", xyz))
        bl.addWidget(self._stat_row("Rx/Ry/Rz", rxyz))
        # 暴露 6 个独立标签（共用上面两个聚合标签，单独 6 个供接线用）
        for _ in range(6):
            self.tcp_vals.append(QLabel("0"))
        self._tcp_xyz = xyz
        self._tcp_rxyz = rxyz

        bl.addStretch(1)
        return card

    # ── 中栏 相机 + JOG ─────────────────────────────────────────────────────
    def _build_center(self) -> QWidget:
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(14)

        # 相机区 + 顶部工具条（可见的启用开关 / 设备选择 / 分辨率）
        cam_wrap = QFrame()
        cam_wrap.setObjectName("camWrap")
        cv = QVBoxLayout(cam_wrap)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)

        cam_bar = QFrame()
        cam_bar.setObjectName("camBar")
        cb = QHBoxLayout(cam_bar)
        cb.setContentsMargins(12, 7, 12, 7)
        cb.setSpacing(9)
        # 「相机画面」条同样去掉 ◉ 图标，与其余卡头一致（见 _card 的说明）
        cam_title = QLabel("相机画面")
        cam_title.setObjectName("cardTitle")
        cb.addWidget(cam_title)
        cb.addSpacing(8)
        cb.addWidget(self._mk_lbl("服务器", "fieldLab"))
        self.cam_server = QLineEdit("http://127.0.0.1:8080")
        self.cam_server.setFixedWidth(190)
        self.cam_server.setToolTip("USB 相机所在服务器代理地址（默认机械臂服务器）")
        cb.addWidget(self.cam_server)
        cb.addWidget(self._mk_lbl("设备", "fieldLab"))
        self.cam_device = QComboBox()
        self.cam_device.addItems(["0", "1", "2", "3"])
        self.cam_device.setFixedWidth(56)
        cb.addWidget(self.cam_device)
        cb.addStretch(1)
        self.cam_res_lbl = QLabel("—")
        self.cam_res_lbl.setObjectName("mini")
        cb.addWidget(self.cam_res_lbl)
        cb.addSpacing(6)
        # 可见的启用开关（拨动）
        self.chk_cam_enable = ToggleSwitch(checked=False)
        cb.addWidget(self.chk_cam_enable)
        cb.addWidget(self._mk_lbl("启用画面", "toggleLab"))
        cv.addWidget(cam_bar)

        self.camera = _CameraArea()
        cv.addWidget(self.camera, 1)
        v.addWidget(cam_wrap, 1)

        # ArUco 钩子（暂隐藏，预留）
        self.chk_aruco = QPushButton("ArUco")
        self.chk_aruco.setCheckable(True)
        self.chk_aruco.setChecked(False)
        self.chk_aruco.setVisible(False)

        # JOG 卡片
        card, bl = _card("手动 JOG")
        # ★ 原本标题旁挂了一行 cardSub「·  步距 2.0° / 速度 5%」，2026-07-28 去掉：
        #   它是**写死的静态文字**，不与 self.joint_step 或任何速度控件绑定 ——
        #   现在恰好等于 joint_step 的默认值（索引 2 = "2.0°"），纯属巧合。
        #   一个"看起来在报当前参数、实际永远不变"的标签比没有更坏（同 lessons
        #   2026-07-28「注释会比代码先腐烂，而且更自信」）。要显示就得真绑信号。

        # 分段切换 关节 / 笛卡尔
        seg = QHBoxLayout()
        seg.setSpacing(6)
        self.seg_joint = QPushButton("关节 JOINT")
        self.seg_joint.setProperty("cssClass", "seg")
        self.seg_joint.setCheckable(True)
        self.seg_joint.setChecked(True)
        self.seg_cart = QPushButton("笛卡尔 CART")
        self.seg_cart.setProperty("cssClass", "seg")
        self.seg_cart.setCheckable(True)
        grp = QButtonGroup(self)
        grp.setExclusive(True)
        grp.addButton(self.seg_joint)
        grp.addButton(self.seg_cart)
        self._seg_group = grp
        seg.addWidget(self.seg_joint)
        seg.addWidget(self.seg_cart)
        bl.addLayout(seg)

        # 步距 combo（钩子）
        self.joint_step = QComboBox()
        self.joint_step.addItems(["0.5°", "1.0°", "2.0°", "5.0°"])
        self.joint_step.setCurrentIndex(2)
        self.joint_step.setVisible(False)

        # 3×4 发光按键网格
        grid = QGridLayout()
        grid.setSpacing(8)
        # 行布局：J1- J2- J3- / J1+ J2+ J3+ / J4- J5- J6- / J4+ J5+ J6+
        layout_map = [
            [(0, -1), (1, -1), (2, -1)],
            [(0, +1), (1, +1), (2, +1)],
            [(3, -1), (4, -1), (5, -1)],
            [(3, +1), (4, +1), (5, +1)],
        ]
        for r, rowdef in enumerate(layout_map):
            for c, (ji, sign) in enumerate(rowdef):
                btn = self._jog_button(ji, sign)
                grid.addWidget(btn, r, c)
                self.jog_buttons[(ji, sign)] = btn
        bl.addLayout(grid)

        v.addWidget(card, 0)
        return wrap

    def _jog_button(self, joint_index: int, sign: int) -> QPushButton:
        """JOG 按键：横向「J1−  底座」—— 主标签(大字)+ 轴名(灰小字)同一行，不换行不重叠。"""
        sign_ch = "+" if sign > 0 else "−"
        btn = QPushButton()
        btn.setProperty("cssClass", "jog")
        btn.setMinimumHeight(48)
        btn.setText("")  # 文本清空，改用内嵌布局，杜绝自动换行重叠
        lay = QHBoxLayout(btn)
        lay.setContentsMargins(12, 0, 12, 0)
        lay.setSpacing(8)
        main = QLabel(f"J{joint_index+1}{sign_ch}")
        main.setObjectName("jogMain")
        main.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        ax = QLabel(JOINT_AXIS_NAMES[joint_index])
        ax.setObjectName("jogAx")
        ax.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        lay.addStretch(1)
        lay.addWidget(main)
        lay.addWidget(ax)
        lay.addStretch(1)
        # 存标签引用，供 JOINT/CART 切换时改文字（J1..J6 ↔ X/Y/Z/Rx/Ry/Rz）
        btn._main_lbl = main
        btn._ax_lbl = ax
        btn.clicked.connect(
            lambda _=False, ji=joint_index, s=sign: self.jog_clicked.emit(ji, s))
        return btn

    def relabel_jog(self, cartesian: bool) -> None:
        """切换 JOINT/CART 时重打按键标签：关节=J1..J6+轴名；笛卡尔=X/Y/Z/Rx/Ry/Rz+平移/姿态。"""
        for (ji, sign), btn in self.jog_buttons.items():
            sign_ch = "+" if sign > 0 else "−"
            main = getattr(btn, "_main_lbl", None)
            axl = getattr(btn, "_ax_lbl", None)
            if main is None or axl is None:
                continue
            if cartesian:
                main.setText(f"{CART_AXIS_NAMES[ji]}{sign_ch}")
                axl.setText(CART_AXIS_HINT[ji])
            else:
                main.setText(f"J{ji + 1}{sign_ch}")
                axl.setText(JOINT_AXIS_NAMES[ji])

    # ── 右栏 点位 / 序列 / 禁区 ─────────────────────────────────────────────
    def _build_right(self) -> QWidget:
        card, bl = _card("示教点位")
        card.setFixedWidth(320)

        bl.addWidget(_mini("已记录 4 个点 · 点击前往"))

        # 点位列表（自绘行，封装为 QListWidget 便于接线）
        self.wp_list = QListWidget()
        self.wp_list.setObjectName("wpList")
        self.wp_list.setSpacing(2)
        wps = [
            ("home", "J [-87.7, 82.1, -81.3 …]"),
            ("pick_above", "J [-62.4, 70.2, -55.1 …]"),
            ("tray_left", "J [-40.1, 64.8, -48.0 …]"),
            ("place", "J [12.3, 58.9, -60.4 …]"),
        ]
        for name, sub in wps:
            self._add_wp_item(self.wp_list, name, sub, "前往 →", ACCENT_COLOR)
        self.wp_list.setFixedHeight(4 * 50 + 8)
        bl.addWidget(self.wp_list)

        # 拖动示教（手拖机械臂到位 → 记录点位）。可勾选切换进/出。
        self.btn_teach = QPushButton("✋ 进入拖动示教")
        self.btn_teach.setCheckable(True)
        self.btn_teach.setObjectName("btnTeach")
        self.btn_teach.setToolTip("进入零力牵引：可徒手拖动机械臂到目标位置，再「记录当前」")
        bl.addWidget(self.btn_teach)

        # 记录 / 删除
        row1 = QHBoxLayout()
        row1.setSpacing(7)
        self.btn_wp_record = QPushButton("＋ 记录当前")
        self.btn_wp_delete = QPushButton("删除")
        row1.addWidget(self.btn_wp_record, 1)
        row1.addWidget(self.btn_wp_delete, 1)
        bl.addLayout(row1)
        # 前往按钮（钩子；点位行内的“前往”由 wp_list 行点击驱动）
        self.btn_wp_goto = QPushButton("前往")
        self.btn_wp_goto.setVisible(False)

        # 序列回放
        bl.addWidget(_section("序列回放"))
        self.seq_combo = QComboBox()
        self.seq_combo.addItems(["默认序列", "标定流程"])
        self.seq_combo.setVisible(False)
        bl.addWidget(_mini("home → pick_above → tray_left → place"))
        row2 = QHBoxLayout()
        row2.setSpacing(7)
        self.btn_seq_run = QPushButton("▶ 运行")
        self.btn_seq_run.setProperty("cssClass", "success")
        self.btn_seq_pause = QPushButton("⏸ 暂停")
        self.btn_seq_stop = QPushButton("■ 停止")
        self.btn_seq_stop.setProperty("cssClass", "danger")
        row2.addWidget(self.btn_seq_run, 1)
        row2.addWidget(self.btn_seq_pause, 1)
        row2.addWidget(self.btn_seq_stop, 1)
        bl.addLayout(row2)
        # 速度 / 进度（钩子）
        self.seq_speed = QSlider(Qt.Orientation.Horizontal)
        self.seq_speed.setRange(1, 100)
        self.seq_speed.setValue(5)
        self.seq_speed.setVisible(False)
        self.seq_progress = QProgressBar()
        self.seq_progress.setRange(0, 100)
        self.seq_progress.setValue(0)
        self.seq_progress.setVisible(False)

        # 运动禁区
        bl.addWidget(_section("运动禁区"))
        bl.addWidget(_mini("已设 1 个禁区盒 · 实时碰撞检查"))
        self.nogo_list = QListWidget()
        self.nogo_list.setObjectName("wpList")
        self.nogo_list.setSpacing(2)
        self._add_wp_item(self.nogo_list, "相机保护区",
                          "X[60,150] Y[300,360] Z[260,310]", "● 启用", WARNING_COLOR)
        self.nogo_list.setFixedHeight(1 * 50 + 8)
        bl.addWidget(self.nogo_list)
        self.btn_nogo_add = QPushButton("＋ 新增禁区盒")
        bl.addWidget(self.btn_nogo_add)

        bl.addStretch(1)
        return card

    def _add_wp_item(self, lst: QListWidget, name: str, sub: str,
                     go_text: str, go_color: str):
        item = QListWidgetItem()
        w = QFrame()
        w.setObjectName("wpRow")
        l = QHBoxLayout(w)
        l.setContentsMargins(11, 8, 11, 8)
        l.setSpacing(8)
        left = QVBoxLayout()
        left.setSpacing(2)
        nm = QLabel(name)
        nm.setObjectName("wpName")
        sb = QLabel(sub)
        sb.setObjectName("wpSub")
        left.addWidget(nm)
        left.addWidget(sb)
        l.addLayout(left, 1)
        go = QLabel(go_text)
        go.setStyleSheet(f"color:{go_color};font-family:{MONO};font-size:11px;")
        l.addWidget(go, 0, Qt.AlignmentFlag.AlignVCenter)
        item.setSizeHint(QSize(0, 48))
        lst.addItem(item)
        lst.setItemWidget(item, w)

    # ── 小构件工厂 ─────────────────────────────────────────────────────────
    def _stat_row(self, key: str, value_widget: QWidget) -> QWidget:
        row = QFrame()
        row.setObjectName("statRow")
        l = QHBoxLayout(row)
        l.setContentsMargins(0, 6, 0, 6)
        k = QLabel(key)
        k.setObjectName("statKey")
        l.addWidget(k)
        l.addStretch(1)
        l.addWidget(value_widget)
        return row

    def _chip(self, text: str, kind: str) -> QLabel:
        lab = QLabel(("● " if kind in ("ok", "run") and text != "无" else "") + text)
        lab.setObjectName("chip")
        lab.setProperty("chipKind", kind)
        return lab

    def _value(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setObjectName("statVal")
        return lab

    # ── 公开更新方法（钩子示例，供接线时填值）──────────────────────────────
    def set_joint(self, index: int, deg: float, pct: float | None = None):
        if 0 <= index < 6:
            self.joint_vals[index].setText(f"{deg:.2f}")
            if pct is None:
                pct = (deg % 360) / 360.0
            self.joint_bars[index].setValue(int(max(0.0, min(1.0, pct)) * 360))


# ════════════════════════════════════════════════════════════════════════════
# 补充 QSS（针对本布局新增的 objectName / property）
# ════════════════════════════════════════════════════════════════════════════
# 亮色（浅色工业风，用户确认的亮色版）
_C_ROOT   = "#EEF2F7"   # 根底 浅灰
_C_PANEL  = "#FFFFFF"   # 面板/工具栏底 白
_C_CARD   = "#FFFFFF"   # 卡片 白
_C_CARD2  = "#F4F7FB"   # 卡片内凹（输入/行底）浅
_C_BD     = "#E2E8F0"   # 边框
_C_BD2    = "#EDF1F6"   # 弱边框
_C_TX     = "#1E293B"   # 主文字
_C_TX2    = "#64748B"   # 次文字
_C_TX3    = "#94A3B8"   # 弱文字
_C_CYAN   = "#0EA5C4"
_C_BLUE   = "#2563EB"
_C_OK     = "#1F9D6B"
_C_WARN   = "#C9821B"
_C_DANGER = "#DC4456"

_EXTRA_QSS = f"""
QWidget#armRoot {{ background-color: {_C_ROOT}; }}
QWidget#armRoot QToolTip {{ color: {_C_TX}; background: {_C_CARD}; border: 1px solid {_C_BD}; }}

/* 工具栏 */
QFrame#topbar {{
    background-color: {_C_PANEL};
    border: 1px solid {_C_BD};
    border-radius: 12px;
}}
QLabel#brandDot {{ color: {_C_CYAN}; font-size: 11px; }}
QLabel#brandTitle {{ font-size: 14px; font-weight: 600; color: {_C_TX}; }}
QLabel#brandSub {{ color: {_C_TX2}; font-family: {MONO}; font-size: 11px; }}
QFrame#vsep {{ background-color: {_C_BD}; }}
QLabel#fieldLab {{ color: {_C_TX2}; font-size: 12px; }}
QFrame#toggleBox {{
    background-color: {_C_CARD2};
    border: 1px solid {_C_BD};
    border-radius: 8px;
}}
QLabel#toggleLab {{ color: {_C_TX2}; font-size: 12px; }}

/* 卡片 */
QFrame#glassCard {{
    background-color: {_C_CARD};
    border: 1px solid {_C_BD};
    border-radius: 12px;
}}
QFrame#cardHead {{
    background-color: transparent;
    border: none;
    border-bottom: 1px solid {_C_BD2};
}}
/* cardIcon / cardSub 的规则保留但已无消费者（2026-07-28 去掉了卡头图标与静态副文字）。
   留着是为了将来若补回真正绑定信号的副文字时有现成样式，不是遗漏。 */
/* 栏目标题：与基准页 orchestrator/ui/control_widget.APP_QSS 逐值一致（2026-07-28 统一）。
   原为 600/深色，现随全仓改为 900/强调色。 */
QLabel#cardTitle {{ font-weight: 900; font-size: 12px; color: {_C_CYAN}; letter-spacing: 0.7px; }}
QLabel#cardSub {{ color: {_C_TX3}; font-size: 11px; }}
QFrame#cardBody {{ background-color: transparent; border: none; }}

/* 状态行 */
QFrame#statRow {{ background: transparent; border: none; }}
QLabel#statKey {{ color: {_C_TX2}; font-size: 12px; }}
QLabel#statVal {{ color: {_C_TX}; font-family: {MONO}; font-size: 13px; font-weight: 500; }}

/* 胶囊（浅色软底 + 同色文字）*/
QLabel#chip {{
    padding: 3px 11px; border-radius: 11px; font-family: {MONO};
    font-size: 11px; font-weight: 600; min-height: 16px;
}}
QLabel#chip[chipKind="ok"] {{
    background-color: #E4F6EE; color: {_C_OK}; border: 1px solid #B6E5CF;
}}
QLabel#chip[chipKind="run"] {{
    background-color: #E2F5F9; color: {_C_CYAN}; border: 1px solid #B6E2EB;
}}
QLabel#chip[chipKind="warn"] {{
    background-color: #FBF1DF; color: {_C_WARN}; border: 1px solid #EAD5A8;
}}

/* 拖动示教按钮（选中=示教中，琥珀高亮）*/
QPushButton#btnTeach {{
    background-color: {_C_CARD2};
    border: 1px solid {_C_BD};
    border-radius: 8px; padding: 7px; font-weight: 600;
    color: {_C_TX};
}}
QPushButton#btnTeach:hover {{ border-color: {_C_WARN}; }}
QPushButton#btnTeach:checked {{
    background-color: #FBF1DF;
    border: 1px solid {_C_WARN};
    color: #9A6510;
}}

/* 底部状态栏 */
QFrame#statusBar {{
    background-color: {_C_CARD};
    border: 1px solid {_C_BD};
    border-radius: 10px;
}}
QLabel#statusDot {{ color: {_C_OK}; font-size: 11px; }}
QLabel#statusMsg {{ color: {_C_TX2}; font-size: 12px; }}

/* 相机容器 + 工具条 */
QFrame#camWrap {{
    background-color: {_C_CARD};
    border: 1px solid {_C_BD};
    border-radius: 12px;
}}
QFrame#camBar {{
    background-color: {_C_CARD};
    border: none;
    border-bottom: 1px solid {_C_BD2};
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
}}

/* 关节 */
QLabel#jlbl {{ color: {_C_CYAN}; font-family: {MONO}; font-size: 11px; font-weight: 600; }}
QLabel#jdeg {{ color: {_C_TX}; font-family: {MONO}; font-size: 12px; }}

/* JOG 按键内文字（横向：主标签 + 轴名）*/
QLabel#jogMain {{ color: {_C_TX}; font-family: {MONO}; font-size: 14px; font-weight: 700; background: transparent; }}
QLabel#jogAx   {{ color: {_C_TX3}; font-size: 11px; background: transparent; }}

/* 小节标题 / mini */
QLabel#section {{
    color: {_C_TX3}; font-size: 11px; letter-spacing: 1.2px;
    font-weight: 600; margin-top: 10px; margin-bottom: 2px;
}}
QLabel#mini {{ color: {_C_TX3}; font-size: 11px; font-family: {MONO}; }}

/* 分段切换按钮 */
QPushButton[cssClass="seg"] {{
    background-color: {_C_CARD2};
    border: 1px solid {_C_BD};
    border-radius: 8px; padding: 7px; font-size: 12px;
    color: {_C_TX2}; font-weight: 500;
}}
QPushButton[cssClass="seg"]:checked {{
    background-color: #E8F0FE;
    border-color: {_C_BLUE};
    color: {_C_BLUE};
}}

/* 点位 / 禁区 列表 */
QListWidget#wpList {{
    background: transparent; border: none; outline: none;
}}
QListWidget#wpList::item {{ border: none; }}
QListWidget#wpList::item:selected {{ background: transparent; }}
QFrame#wpRow {{
    background-color: {_C_CARD2};
    border: 1px solid {_C_BD2};
    border-radius: 9px;
}}
QLabel#wpName {{ font-size: 12px; font-weight: 600; color: {_C_TX}; }}
QLabel#wpSub {{ color: {_C_TX3}; font-family: {MONO}; font-size: 11px; }}
"""


# ════════════════════════════════════════════════════════════════════════════
# 自测
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # HARDCODED_PATH: 仅本文件单独自测截图用，不影响 python main.py
    sys.path.insert(0, "G:/2D_robotics/2D_robotics/src")

    from PyQt6.QtCore import QTimer

    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))

    w = ArmConsoleWidget()
    w.resize(1400, 820)
    w.setWindowTitle("机械臂控制台")
    w.show()

    # HARDCODED_PATH: 仅本文件单独自测截图用，不影响 python main.py
    shot_path = "G:/2D_robotics/2D_robotics/tools/_shots/console_widget_test.png"

    def _grab_and_quit():
        os.makedirs(os.path.dirname(shot_path), exist_ok=True)
        pm = w.grab()
        ok = pm.save(shot_path, "PNG")
        print(f"[selftest] screenshot saved={ok} -> {shot_path}")
        app.quit()

    QTimer.singleShot(600, _grab_and_quit)
    sys.exit(app.exec())
