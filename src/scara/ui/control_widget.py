"""
SCARA 机械臂控制界面

对齐机械臂(robot_arm)页设计：卡片内联表头(▍标题, 不用 QGroupBox 边框标题)、
关节角度=标签+渐变条+数值、cssClass 分级按钮、等宽玻璃输入、紧凑工业控制台密度。
三栏：左=控制，中=相机实时画面，右=状态/数据反馈；对接 ScaraController，支持镜像。
"""

from __future__ import annotations

import importlib.util
import math
import sys
import threading
import time
from collections import deque
from datetime import datetime
import json
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QPushButton,
    QLineEdit, QDoubleSpinBox, QSpinBox, QFrame, QSizePolicy, QPlainTextEdit,
    QProgressBar, QButtonGroup, QCheckBox, QComboBox, QScrollArea, QMessageBox,
    QDialog, QFileDialog,
)

from utils import get_logger
from scara.config.scara_config import ScaraConfig, load_scara_config
from scara.controller.scara_controller import ScaraController
from scara.ui.action_worker import ActionWorker, normalize_action_task
from scara.ui.camera_view import ScaraCameraThread
from robot_arm.ui import theme as T

logger = get_logger("scara.ui")

# 泵/阀 DO 映射文件（进程工作目录；与 scara_presets.json 同级）。只存 kind/name/ch，不存电平。
DO_MAP_FILE = "scara_do_map.json"

JOINT_NAME = ["J1", "J2", "J3", "J4"]
JOINT_UNIT = ["°", "°", "mm", "°"]
JOINT_RANGE = [(-132, 132), (-141, 141), (-150, 10), (-360, 360)]
POSE_LABELS = ["X", "Y", "Z", "Rx", "Ry", "Rz"]
POSE_UNIT = ["mm", "mm", "mm", "°", "°", "°"]

_CHK = Path(__file__).parent.joinpath("check.svg").as_posix()

# SCARA 页局部深色色板（不改全局 design_system，DUCO 页保持浅色）
_D = {
    "bg": "#0F172A",
    "bg2": "#1E293B",
    "card": "#1E293B",
    "glass": "#334155",
    "border": "#334155",
    "border_soft": "#475569",
    "text": "#E2E8F0",
    "text2": "#94A3B8",
    "muted": "#64748B",
    "primary": "#3B82F6",
    "primary_soft": "#1E3A5F",
    "accent": "#22D3EE",
    "accent_soft": "#164E63",
    "success": "#34D399",
    "error": "#F87171",
    "warning": "#FBBF24",
    "io_off": "#475569",
    "chip_off_bg": "#334155",
    "chip_on_bg": "#064E3B",
    "bar_bg": "#334155",
    "log_bg": "#0B1220",
    "input_bg": "#0F172A",
    "seg_bg": "#1E293B",
}

SUPPLEMENT = f"""
QWidget#scaraRoot {{ background-color:{_D['bg']}; color:{_D['text']}; }}
QWidget {{ font-size:12px; color:{_D['text']}; }}
QLabel {{ color:{_D['text']}; }}
QCheckBox {{ color:{_D['text2']}; }}
QCheckBox::indicator {{ width:15px; height:15px; border-radius:4px;
    border:1px solid {_D['border']}; background:{_D['glass']}; }}
QCheckBox::indicator:checked {{ background:{_D['accent']}; border-color:{_D['accent']};
    image:url({_CHK}); }}
#card {{ background:{_D['card']}; border:1px solid {_D['border']}; border-radius:14px; }}
#cardBar {{ background:{_D['accent']}; border-radius:2px; }}
#cardTitle {{ color:{_D['accent']}; font-weight:900; font-size:12px; letter-spacing:0.7px; }}
#colScroll, #colScroll > QWidget > QWidget {{ background:transparent; border:none; }}
#kv {{ color:{_D['text2']}; font-size:12px; }}
#val {{ font-family:{T.MONO}; font-weight:700; font-size:13px; color:{_D['text']}; }}
#jname {{ font-family:{T.MONO}; font-weight:700; color:{_D['primary']}; font-size:12px; }}
#muted {{ color:{_D['muted']}; font-size:11px; }}
#pill {{ border-radius:9px; padding:1px 9px; font-size:11px; font-weight:700; }}
#light {{ background:{_D['glass']}; border:1px solid {_D['border_soft']}; border-radius:7px; }}
#lightK {{ color:{_D['text2']}; font-size:11px; }}
#io {{ background:{_D['io_off']}; border:1px solid {_D['border_soft']}; border-radius:2px; min-width:11px; min-height:11px; }}
#ioOn {{ background:{_D['success']}; border:1px solid {_D['success']}; border-radius:2px; min-width:11px; min-height:11px; }}
#cam {{ background:#0b1220; border-radius:8px; color:{_D['muted']}; font-size:13px; }}
QPushButton {{ padding:4px 9px; min-height:22px; outline:none;
    background:{_D['glass']}; color:{_D['text']}; border:1px solid {_D['border']}; border-radius:7px; }}
QPushButton:hover {{ background:{_D['border_soft']}; }}
QPushButton:focus {{ outline:none; }}
QPushButton:pressed {{ background:{_D['primary_soft']}; }}
QPushButton:disabled {{ color:{_D['muted']}; background:#1a2332; border-color:#243044; }}
QPushButton[cssClass="primary"] {{ background:{_D['primary']}; color:#fff; border:none; }}
QPushButton[cssClass="primary"]:hover {{ background:#2563EB; }}
QPushButton[cssClass="primary"]:pressed {{ background:#1D54CC; }}
QPushButton[cssClass="success"] {{ background:{_D['success']}; color:#0F172A; border:none; }}
QPushButton[cssClass="success"]:pressed {{ background:#1A8A5E; color:#fff; }}
QPushButton[cssClass="jog"] {{ background:{_D['bg2']}; color:{_D['text']}; border:1px solid {_D['border']}; }}
QPushButton[cssClass="jog"]:pressed {{ background:{_D['accent_soft']}; border-color:{_D['accent']}; }}
QPushButton#estop:pressed {{ background:#B23244; }}
QPushButton#estop {{ background:{_D['error']}; color:#fff; border:none; border-radius:7px;
    font-size:13px; font-weight:800; letter-spacing:2px; min-height:0; padding:2px 16px; }}
QPushButton#estop:hover {{ background:#C53B4C; }}
QPushButton#estop:disabled {{ background:#7F1D1D; color:#FECACA; }}
QPushButton#cambtn {{ background:#1e293b; color:#e2e8f0; border:1px solid #334155; }}
QPushButton#cambtn:hover {{ background:#273449; }}
QPushButton[cssClass="jog"] {{ min-height:34px; padding:4px 8px; border-radius:9px;
    font-size:15px; font-weight:700; letter-spacing:1px; }}
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
    min-height:20px; padding:2px 6px;
    background:{_D['input_bg']}; color:{_D['text']};
    border:1px solid {_D['border']}; border-radius:6px; }}
QLineEdit:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled, QComboBox:disabled {{
    color:{_D['muted']}; background:#1a2332; border-color:#243044; }}
QComboBox::drop-down {{ border:none; }}
QComboBox QAbstractItemView {{
    background:{_D['card']}; color:{_D['text']}; selection-background-color:{_D['primary']}; }}
QPushButton#seg {{ border-radius:0; background:{_D['seg_bg']}; color:{_D['text2']};
    padding:5px 3px; border:1px solid {_D['border']}; font-weight:600; }}
QPushButton#seg:checked {{ background:{_D['primary']}; color:#fff; border-color:{_D['primary']}; }}
QProgressBar#posbar {{ border:none; background:{_D['bar_bg']}; border-radius:4px; max-height:8px; min-height:8px; }}
QProgressBar#posbar::chunk {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 #7CC1EE, stop:1 {_D['primary']}); border-radius:4px; }}
QPlainTextEdit#logbox {{ background:{_D['log_bg']}; border:1px solid {_D['border_soft']}; border-radius:8px;
    font-family:{T.MONO}; font-size:11px; color:{_D['text2']}; }}
QLabel#chip {{ border-radius:11px; padding:2px 9px; font-size:11px; font-weight:700; }}
QScrollArea#colScroll QScrollBar:vertical {{
    background:transparent; width:8px; margin:0; }}
QScrollArea#colScroll QScrollBar::handle:vertical {{
    background:{_D['border_soft']}; border-radius:4px; min-height:24px; }}
QScrollArea#colScroll QScrollBar::add-line:vertical,
QScrollArea#colScroll QScrollBar::sub-line:vertical {{ height:0; }}
"""


def _card(title: str) -> tuple[QFrame, QVBoxLayout]:
    """一个栏目 = 一张卡。卡头**只放标题**，不带说明行（2026-07-28 全仓统一）。"""
    f = QFrame(); f.setObjectName("card")
    v = QVBoxLayout(f); v.setContentsMargins(11, 9, 11, 10); v.setSpacing(7)
    head = QHBoxLayout(); head.setSpacing(7)
    bar = QFrame(); bar.setObjectName("cardBar"); bar.setFixedSize(3, 13)
    t = QLabel(title); t.setObjectName("cardTitle")
    head.addWidget(bar); head.addWidget(t); head.addStretch(1)
    v.addLayout(head)
    return f, v


def _btn(text: str, css: str = "") -> QPushButton:
    b = QPushButton(text)
    b.setFocusPolicy(Qt.FocusPolicy.NoFocus)   # 点击后不留焦点虚线框
    if css:
        b.setProperty("cssClass", css)
    return b


def _pill(text: str, color: str, bg: str) -> QLabel:
    p = QLabel(text); p.setObjectName("pill")
    p.setStyleSheet(f"color:{color}; background:{bg};")
    p.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return p


def _light(key: str) -> tuple[QFrame, QLabel]:
    f = QFrame(); f.setObjectName("light")
    g = QVBoxLayout(f); g.setContentsMargins(4, 4, 4, 4); g.setSpacing(1)
    k = QLabel(key); k.setObjectName("lightK"); k.setAlignment(Qt.AlignmentFlag.AlignCenter)
    v = QLabel("—"); v.setAlignment(Qt.AlignmentFlag.AlignCenter); v.setStyleSheet("font-weight:700; font-size:12px;")
    g.addWidget(k); g.addWidget(v)
    return f, v


def _scroll(inner: QWidget, width: int) -> QScrollArea:
    """把一列内容包进竖向滚动区，避免加了命令头后底部卡片在小窗口被裁切。"""
    sa = QScrollArea(); sa.setObjectName("colScroll")
    sa.setWidget(inner); sa.setWidgetResizable(True); sa.setFixedWidth(width)
    sa.setFrameShape(QFrame.Shape.NoFrame)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    sa.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    return sa


class ScaraControlWidget(QWidget):
    """SCARA 主控制界面。"""

    def __init__(self, parent: Optional[QWidget] = None,
                 config: Optional[ScaraConfig] = None,
                 controller: Optional[ScaraController] = None,
                 owns_controller: bool = True):
        super().__init__(parent)
        self.setObjectName("scaraRoot")
        # 仅用本页深色 SUPPLEMENT，避免全局浅色 APP_STYLESHEET 盖住深色卡片/输入框
        self.setStyleSheet(SUPPLEMENT)
        self._cfg = config or load_scara_config()
        self._owns = owns_controller
        self._ctrl = controller or ScaraController(self._cfg)
        self._cam: Optional[ScaraCameraThread] = None
        self._camera_connection_counts: dict[int, int] = {}
        self._handeye_dialog: Optional[QDialog] = None
        self._wafer_transfer_dialog: Optional[QDialog] = None
        self._handeye_motion_mode: Optional[str] = None
        self._handeye_state_lock = threading.Lock()
        self._handeye_controller_connected = False
        self._latest_handeye_robot_state: Optional[dict[str, object]] = None
        # Camera frames and controller polling are asynchronous.  Keep enough
        # timestamped controller samples to pair each image with the state
        # nearest its capture time instead of repeatedly comparing it with one
        # arbitrarily phased "latest" sample.
        self._handeye_robot_state_history: deque[dict[str, object]] = deque(
            maxlen=64
        )
        self._action_file: Optional[Path] = None
        self._action_builder: Optional[Callable[[], dict]] = None
        self._action_camera_calculator: Optional[Callable[[list[float]], dict]] = None
        self._action_source_position_calculators: dict[
            int, Callable[[list[float], list[float]], dict]
        ] = {}
        self._action_runtime_factory: Optional[Callable[..., object]] = None
        self._action_runtime: Optional[object] = None
        self._action_task: Optional[dict] = None
        self._action_worker: Optional[ActionWorker] = None
        self._one_shot_action_restore: Optional[tuple[object, ...]] = None
        self._action_control_states: list[tuple[QWidget, bool]] = []
        self._resume_camera_index: Optional[int] = None

        self._joint_v: list[QLabel] = []
        self._joint_bar: list[QProgressBar] = []
        self._pose_v: list[QLabel] = []
        self._light: dict[str, QLabel] = {}
        self._io_cells: list[QFrame] = []
        # DO 行：{kind, name, ch, widget, lvl, applied}；applied=已真正下发的电平
        self._do_entries: list[dict] = []

        self._build()
        self._load_do_map()
        # 主界面启动时向硬件写一遍 0（镜像窗 owns=False，避免重复抢连接）
        if self._owns:
            self._zero_all_dos("启动")
        self._bind()
        for b in self.findChildren(QPushButton):   # 所有按钮去掉点击后的焦点虚线框
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._on_conn(False)                        # 初始未连接：按钮按实际状态置灰

    @property
    def controller(self) -> ScaraController:
        return self._ctrl

    def is_device_connected(self) -> bool:
        return self._ctrl.is_connected()

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        outer = QVBoxLayout(self); outer.setContentsMargins(12, 12, 12, 12); outer.setSpacing(10)
        row = QHBoxLayout(); row.setSpacing(10)
        self._left_panel = self._left()
        row.addWidget(self._left_panel, 0)
        row.addWidget(self._center(), 1)
        row.addWidget(self._right(), 0)
        outer.addLayout(row, 1)
        # 急停：页面底部右下角（顶部不好够到，用户要求移到下面 + 缩小），永远醒目红。
        estop_row = QHBoxLayout(); estop_row.setContentsMargins(0, 0, 0, 0); estop_row.addStretch(1)
        self._btn_estop = QPushButton("■ 急 停"); self._btn_estop.setObjectName("estop")
        self._btn_estop.setFixedHeight(30); self._btn_estop.setMinimumWidth(120)
        estop_row.addWidget(self._btn_estop)
        outer.addLayout(estop_row)

    def _jog_grid(self, g: QVBoxLayout, axes) -> None:
        """每轴一行两颗按钮：「name −」「name +」，大字号清晰可辨。"""
        grid = QGridLayout(); grid.setSpacing(6); grid.setContentsMargins(0, 2, 0, 0)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        for i, (name, axis) in enumerate(axes):
            bm = _btn(f"{name}  −", "jog"); bp = _btn(f"{name}  +", "jog")
            bm.setMinimumWidth(92); bp.setMinimumWidth(92)
            bp.pressed.connect(lambda a=axis: self._jog_press(a, "+"))
            bp.released.connect(lambda a=axis: self._jog_release(a))
            bp.clicked.connect(lambda _, a=axis: self._step(a, +1))
            bm.pressed.connect(lambda a=axis: self._jog_press(a, "-"))
            bm.released.connect(lambda a=axis: self._jog_release(a))
            bm.clicked.connect(lambda _, a=axis: self._step(a, -1))
            grid.addWidget(bm, i, 0); grid.addWidget(bp, i, 1)
        g.addLayout(grid)

    def _left(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w); v.setContentsMargins(0, 0, 4, 0); v.setSpacing(9)

        f, g = _card("连接")
        row = QHBoxLayout(); row.addWidget(QLabel("控制器"))
        self._ip = QLineEdit(f"{self._cfg.controller_ip}:{self._cfg.controller_port}"); self._ip.setReadOnly(True)
        row.addWidget(self._ip, 1); g.addLayout(row)
        row = QHBoxLayout()
        self._btn_conn = _btn("连接", "primary"); self._btn_disc = _btn("断开")
        self._conn_chip = QLabel("● 未连接"); self._conn_chip.setObjectName("chip")
        self._conn_chip.setStyleSheet(f"color:{_D['muted']}; background:{_D['chip_off_bg']};")
        row.addWidget(self._btn_conn); row.addWidget(self._btn_disc); row.addStretch(1); row.addWidget(self._conn_chip)
        g.addLayout(row); v.addWidget(f)

        f, g = _card("伺服 / 安全")
        row = QHBoxLayout()
        self._btn_en = _btn("使能")  # 不默认 success，避免未读状态就显绿「已使能」
        self._btn_dis = _btn("去使能")
        self._btn_clr = _btn("清报警"); self._btn_home = _btn("回零")
        for b in (self._btn_en, self._btn_dis, self._btn_clr, self._btn_home):
            row.addWidget(b)
        g.addLayout(row)
        v.addWidget(f)

        f, g = _card("模式 / 速度")
        seg = QHBoxLayout(); seg.setSpacing(0); self._mode_grp = QButtonGroup(self)
        for i, name in enumerate(["示教 T1", "示教 T2", "执行"]):
            b = QPushButton(name); b.setObjectName("seg"); b.setCheckable(True)
            if i == 0: b.setChecked(True)
            if i == 1 and not self._cfg.allow_t2_mode: b.setEnabled(False)
            self._mode_grp.addButton(b, i); seg.addWidget(b)
        g.addLayout(seg)
        row = QHBoxLayout(); row.addWidget(QLabel("速度"))
        self._speed = QSpinBox()
        self._speed.setRange(
            int(self._cfg.min_speed_percent),
            int(self._cfg.max_speed_percent),
        )
        self._speed.setSuffix(" %")
        self._speed.setValue(self._cfg.clamp_speed(self._cfg.default_speed_percent))
        self._speed_bar = QProgressBar(); self._speed_bar.setObjectName("posbar")
        self._speed_bar.setRange(0, int(self._cfg.max_speed_percent))
        self._speed_bar.setValue(
            self._cfg.clamp_speed(self._cfg.default_speed_percent)
        )
        self._speed_bar.setTextVisible(False)
        row.addWidget(self._speed_bar, 1); row.addWidget(self._speed)
        g.addLayout(row); v.addWidget(f)

        f, g = _card("关节点动")
        self._jog_grid(g, list(zip(["J1", "J2", "J3/Z", "J4"], [1, 2, 3, 4])))
        row = QHBoxLayout(); row.addWidget(QLabel("步长"))
        self._jstep = QDoubleSpinBox(); self._jstep.setRange(0.1, 90.0)
        self._jstep.setValue(self._cfg.default_joint_step_deg); self._jstep.setSingleStep(0.5)
        self._cont = QCheckBox("连续(按住)")
        u = QLabel("度/mm"); u.setObjectName("muted")
        row.addWidget(self._jstep); row.addWidget(u); row.addStretch(1); row.addWidget(self._cont)
        g.addLayout(row); v.addWidget(f)

        f, g = _card("笛卡尔点动 (World)")
        self._jog_grid(g, list(zip(["X", "Y", "Z"], ["X", "Y", "Z"])))
        row = QHBoxLayout(); row.addWidget(QLabel("步长"))
        self._cstep = QDoubleSpinBox(); self._cstep.setRange(0.1, 100.0)
        self._cstep.setValue(self._cfg.default_cart_step_mm); self._cstep.setSingleStep(1.0)
        u = QLabel("mm"); u.setObjectName("muted")
        row.addWidget(self._cstep); row.addWidget(u); row.addStretch(1)
        g.addLayout(row); v.addWidget(f)

        f, g = _card("预设点")
        row = QHBoxLayout()
        self._preset_name = QLineEdit(); self._preset_name.setPlaceholderText("名称")
        self._btn_save_preset = _btn("保存当前")
        row.addWidget(self._preset_name, 1); row.addWidget(self._btn_save_preset)
        g.addLayout(row)
        row = QHBoxLayout()
        self._preset_combo = QComboBox()
        self._btn_goto = _btn("前往", "primary"); self._btn_del_preset = _btn("删除")
        row.addWidget(self._preset_combo, 1); row.addWidget(self._btn_goto); row.addWidget(self._btn_del_preset)
        g.addLayout(row); v.addWidget(f)

        f, g = _card("任务执行")
        row = QHBoxLayout()
        self._btn_import_action = _btn("导入动作")
        self._action_label = QLabel("未导入")
        self._action_label.setObjectName("muted")
        # 左栏宽度固定；长文件名/说明只能在按钮之间的区域内换行，不能撑宽布局。
        self._action_label.setWordWrap(True)
        self._action_label.setFixedWidth(96)
        self._action_label.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Preferred,
        )
        self._btn_run_action = _btn("执行动作", "primary")
        self._btn_run_action.setEnabled(False)
        row.addWidget(self._btn_import_action)
        row.addWidget(self._action_label, 1)
        row.addWidget(self._btn_run_action)
        g.addLayout(row)
        note = QLabel("动作可组合 XYZ/R 运动、相机拍照/录像和 JSON 途径点记录。")
        note.setObjectName("muted"); note.setWordWrap(True)
        g.addWidget(note)
        v.addWidget(f)

        f, g = _card("手眼交互")
        self._btn_handeye_demo = _btn("动态演示", "primary")
        self._btn_handeye_demo.setToolTip(
            "打开相机1实时槽位/吸盘误差叠加；只计算，不移动机械臂"
        )
        self._btn_wafer_transfer = _btn("转移视觉", "primary")
        self._btn_wafer_transfer.setToolTip(
            "相机1实时托盘/硅片分析、点击锁定槽位和吸盘距离跟踪；只计算"
        )
        row = QHBoxLayout()
        row.addWidget(self._btn_handeye_demo)
        row.addWidget(self._btn_wafer_transfer)
        g.addLayout(row)
        handeye_note = QLabel(
            "目标槽下拉选择、Stage3实时位姿、A–H重投影、局部Jacobian标定和XY定位。"
            "动态演示与验证按钮只计算，机械臂不会移动。"
        )
        handeye_note.setObjectName("muted")
        handeye_note.setWordWrap(True)
        g.addWidget(handeye_note)
        v.addWidget(f)

        # DO接口：泵/阀两列；映射持久化到 scara_do_map.json；电平改后须回车才下发
        f, g = _card("DO接口")
        row = QHBoxLayout()
        self._btn_add_do = _btn("新建", "primary")
        self._btn_add_do.clicked.connect(self._show_add_do_dialog)
        row.addWidget(self._btn_add_do); row.addStretch(1)
        g.addLayout(row)
        cols = QHBoxLayout(); cols.setSpacing(8)
        for title, attr in [("泵", "_do_pump_col"), ("阀", "_do_valve_col")]:
            box = QWidget()
            vcol = QVBoxLayout(box); vcol.setSpacing(4); vcol.setContentsMargins(0, 0, 0, 0)
            hdr = QLabel(title); hdr.setObjectName("kv")
            vcol.addWidget(hdr)
            inner = QWidget()
            col = QVBoxLayout(inner); col.setSpacing(4); col.setContentsMargins(0, 0, 0, 0)
            setattr(self, attr, col)
            vcol.addWidget(inner)
            cols.addWidget(box, 1)
        g.addLayout(cols)
        tip = QLabel(
            "须连接机械臂后才能新建/改电平/删除。"
            " 1/0 = 高/低电平；改后按回车才写入。"
            " 启动/连接前/断开后/急停/退出会自动把全部 DO 写成 0。"
            " 写 DO 会另连控制器，已连接时写入可能失败（界面仍要求先连接，安全优先）。"
        )
        tip.setObjectName("muted"); tip.setWordWrap(True)
        self._do_tip = tip
        g.addWidget(tip)
        v.addWidget(f)

        v.addStretch(1)
        return _scroll(w, 318)

    def _center(self) -> QWidget:
        f, g = _card("相机画面（SCARA 工位）")
        self._cam_lbl = QLabel("相机未连接 — 点击「连接相机」打开 SCARA 工位相机")
        self._cam_lbl.setObjectName("cam"); self._cam_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._cam_lbl.setMinimumHeight(300)
        g.addWidget(self._cam_lbl, 1)
        bar = QHBoxLayout()
        self._btn_cam = QPushButton("连接相机"); self._btn_cam.setObjectName("cambtn")
        self._btn_snap = QPushButton("快照"); self._btn_snap.setObjectName("cambtn")
        self._cam_idx = QSpinBox(); self._cam_idx.setRange(0, 2); self._cam_idx.setValue(self._default_cam_index()); self._cam_idx.setPrefix("逻辑源#")
        bar.addWidget(self._btn_cam); bar.addWidget(self._btn_snap); bar.addWidget(self._cam_idx)
        bar.addStretch(1)
        g.addLayout(bar)
        lights = QGridLayout(); lights.setSpacing(6)
        for i, key in enumerate(["使能", "运行状态", "急停", "模式", "循环", "机械锁"]):
            box, lbl = _light(key); self._light[key] = lbl
            lights.addWidget(box, i // 6, i % 6)
        g.addLayout(lights)
        return f

    @staticmethod
    def _default_cam_index() -> int:
        """Default to logical camera 1; physical indices live in local config."""

        return 1

    def _kv_row(self, g: QVBoxLayout, key: str) -> QLabel:
        row = QHBoxLayout(); row.setSpacing(6)
        k = QLabel(key); k.setObjectName("kv")
        val = QLabel("—"); val.setObjectName("val"); val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(k); row.addStretch(1); row.addWidget(val)
        g.addLayout(row)
        return val

    def _right(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w); v.setContentsMargins(0, 0, 4, 0); v.setSpacing(9)

        # 关节角度：标签 + 渐变条 + 数值
        f, g = _card("关节角度（实时）")
        for i in range(4):
            row = QHBoxLayout(); row.setSpacing(7)
            lab = QLabel(JOINT_NAME[i]); lab.setObjectName("jname"); lab.setFixedWidth(28)
            bar = QProgressBar(); bar.setObjectName("posbar"); bar.setRange(0, 100); bar.setValue(50); bar.setTextVisible(False)
            self._joint_bar.append(bar)
            val = QLabel("—"); val.setObjectName("val"); val.setFixedWidth(72)
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._joint_v.append(val)
            row.addWidget(lab); row.addWidget(bar, 1); row.addWidget(val)
            g.addLayout(row)
        v.addWidget(f)

        # 末端位姿
        f, g = _card("末端位姿 (TCP)")
        for i in range(6):
            self._pose_v.append(self._kv_row(g, f"{POSE_LABELS[i]} ({POSE_UNIT[i]})"))
        v.addWidget(f)

        # IO 状态
        f, g = _card("IO 状态  (DI 1500 · DO 4096)")
        self._io_lbl = QLabel("DI ON: —    DO ON: —"); self._io_lbl.setObjectName("kv"); g.addWidget(self._io_lbl)
        grid = QGridLayout(); grid.setSpacing(4)
        for i in range(32):
            cell = QFrame(); cell.setObjectName("io"); self._io_cells.append(cell)
            grid.addWidget(cell, i // 16, i % 16)
        g.addLayout(grid); v.addWidget(f)

        # 报警日志
        f, g = _card("报警 / 日志")
        self._log = QPlainTextEdit(); self._log.setObjectName("logbox"); self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(200); self._log.setMinimumHeight(110)
        g.addWidget(self._log); v.addWidget(f, 1)
        return w

    # ------------------------------------------------------------------ #
    def _bind(self) -> None:
        self._btn_conn.clicked.connect(self._on_connect_clicked)
        self._btn_disc.clicked.connect(self._on_disconnect_clicked)
        self._btn_en.clicked.connect(self._ctrl.cmd_enable)
        self._btn_dis.clicked.connect(self._ctrl.cmd_disable)
        self._btn_clr.clicked.connect(self._ctrl.cmd_clear_alarm)
        self._btn_home.clicked.connect(self._on_home_clicked)
        self._btn_estop.clicked.connect(self._on_estop_clicked)
        self._mode_grp.idClicked.connect(lambda i: self._ctrl.cmd_set_mode(["T1", "T2", "Execute"][i]))
        self._speed.valueChanged.connect(lambda val: self._ctrl.cmd_set_speed(val))
        self._btn_save_preset.clicked.connect(self._on_save_preset)
        self._btn_goto.clicked.connect(lambda: self._ctrl.cmd_goto_preset(self._preset_combo.currentText()))
        self._btn_del_preset.clicked.connect(lambda: self._ctrl.delete_preset(self._preset_combo.currentText()))
        self._btn_import_action.clicked.connect(self._choose_action_file)
        self._btn_run_action.clicked.connect(self._on_run_action)
        self._btn_handeye_demo.clicked.connect(self._open_handeye_demo)
        self._btn_wafer_transfer.clicked.connect(self._open_wafer_transfer)
        self._btn_cam.clicked.connect(self._toggle_camera)
        self._btn_snap.clicked.connect(self._snapshot)
        self._ctrl.connection_changed.connect(self._on_conn)
        self._ctrl.status_updated.connect(self._on_status)
        self._ctrl.presets_changed.connect(self._on_presets)
        self._ctrl.error_occurred.connect(lambda m: self._append("错误", m, _D["error"]))
        self._ctrl.warning_occurred.connect(lambda m: self._append("警告", m, _D["warning"]))
        self._ctrl.info_occurred.connect(lambda m: self._append("信息", m, _D["text2"]))
        self._ctrl.command_finished.connect(
            lambda name, ok, msg: self._append("OK" if ok else "失败", name, _D["success"] if ok else _D["error"]))

    def _jog_press(self, axis, direction: str) -> None:
        if self._cont.isChecked():
            self._ctrl.jog_start(axis, direction, world=isinstance(axis, str))

    def _jog_release(self, axis) -> None:
        if self._cont.isChecked():
            self._ctrl.jog_stop(axis)

    def _step(self, axis, sign: int) -> None:
        if self._cont.isChecked():
            return
        if isinstance(axis, str):
            self._ctrl.cmd_cart_step(axis, sign * self._cstep.value())
        else:
            self._ctrl.cmd_move_joint(axis, sign * self._jstep.value())

    def _on_save_preset(self) -> None:
        name = self._preset_name.text().strip()
        if name:
            self._ctrl.save_preset(name); self._preset_name.clear()

    def _choose_action_file(self) -> None:
        """选择并校验只描述步骤、导入时绝不访问硬件的动作插件。"""
        project_root = Path(__file__).resolve().parents[3]
        initial_dir = project_root / "Tasks"
        if not initial_dir.is_dir():
            initial_dir = project_root
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择动作文件",
            str(initial_dir),
            "Python action files (*.py);;All files (*.*)",
        )
        if not selected:
            return

        path = Path(selected).resolve()
        self._btn_run_action.setEnabled(False)
        self._action_task = None
        self._action_builder = None
        self._action_camera_calculator = None
        self._action_source_position_calculators = {}
        self._action_runtime_factory = None
        self._action_runtime = None
        try:
            module_name = f"_scara_action_{abs(hash(str(path)))}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ValueError("无法创建 Python 模块加载器")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            if getattr(module, "ACTION_API_VERSION", None) != 1:
                raise ValueError("不支持的动作 API 版本（需要 ACTION_API_VERSION = 1）")
            build = getattr(module, "build_action", None)
            if not callable(build):
                raise ValueError("动作文件必须定义 build_action()")

            task = normalize_action_task(build())
            camera_calculator = getattr(module, "camera_position_from_pose", None)
            if camera_calculator is not None and not callable(camera_calculator):
                raise ValueError("camera_position_from_pose 必须是可调用函数")
            if callable(camera_calculator):
                sample = camera_calculator([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                if not isinstance(sample, dict) or not all(
                    key in sample for key in ("x_mm", "y_mm", "z_mm")
                ):
                    raise ValueError("camera_position_from_pose 必须返回 x_mm/y_mm/z_mm")
            camera1_calculator = getattr(module, "camera1_position_from_state", None)
            if camera1_calculator is not None and not callable(camera1_calculator):
                raise ValueError("camera1_position_from_state 必须是可调用函数")
            source_position_calculators = {}
            if callable(camera1_calculator):
                sample = camera1_calculator(
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                )
                if not isinstance(sample, dict) or not all(
                    key in sample for key in ("x_mm", "y_mm", "z_mm")
                ):
                    raise ValueError(
                        "camera1_position_from_state 必须返回 x_mm/y_mm/z_mm"
                    )
                source_position_calculators[1] = camera1_calculator
            runtime_factory = getattr(module, "create_task_runtime", None)
            if runtime_factory is not None and not callable(runtime_factory):
                raise ValueError("create_task_runtime 必须是可调用函数")
            point_count = sum(step["type"] == "record_point" for step in task["actions"])
            photo_count = sum(step["type"] == "capture" for step in task["actions"])
            video_count = sum(step["type"] == "start_video" for step in task["actions"])
            do_count = sum(step["type"] == "set_do" for step in task["actions"])
            self._action_file = path
            self._action_builder = build
            self._action_camera_calculator = camera_calculator
            self._action_source_position_calculators = source_position_calculators
            self._action_runtime_factory = runtime_factory
            self._action_task = task
            self._btn_import_action.setToolTip(str(path))
            self._btn_run_action.setEnabled(True)
            self._action_label.setText(
                f"{path.name} · {point_count} 点/{photo_count} 照片/"
                f"{video_count} 录像/{do_count} DO"
            )
            self._append(
                "动作",
                f"已载入 {path.name}: {task['name']}（{len(task['actions'])} 步）",
                _D["success"],
            )
        except Exception as exc:
            self._action_file = None
            self._action_runtime_factory = None
            self._action_runtime = None
            self._btn_import_action.setToolTip("")
            self._action_label.setText("导入失败")
            self._append("动作错误", f"无法载入 {path.name}: {exc}", _D["error"])

    def _set_action_controls_locked(self, locked: bool) -> None:
        """锁住左栏手动控件，但让同栏的停止动作按钮保持可用。"""
        if locked:
            self._action_control_states = []
            widget_types = (
                QPushButton,
                QLineEdit,
                QDoubleSpinBox,
                QSpinBox,
                QCheckBox,
                QComboBox,
            )
            seen: set[int] = set()
            for widget_type in widget_types:
                for widget in self._left_panel.findChildren(widget_type):
                    if (
                        widget is self._btn_run_action
                        or widget is self._btn_estop
                        or id(widget) in seen
                    ):
                        continue
                    seen.add(id(widget))
                    self._action_control_states.append((widget, widget.isEnabled()))
                    widget.setEnabled(False)
            return
        for widget, was_enabled in self._action_control_states:
            widget.setEnabled(was_enabled)
        self._action_control_states = []

    def _on_run_action(self) -> None:
        """Start or stop the imported multi-motion/multi-camera action."""
        worker = self._action_worker
        if worker is not None and worker.isRunning():
            worker.request_stop()
            self._btn_run_action.setText("正在停止…")
            self._btn_run_action.setEnabled(False)
            self._append("动作", "已请求停止；请保持物理急停可用", _D["warning"])
            return
        if self._action_builder is None:
            self._append("动作错误", "请先导入有效动作", _D["error"])
            return
        if not self._ctrl.motion_ready():
            return

        # Task capture needs exclusive camera access.  Refuse to proceed until
        # the read-only Stage3 monitor has really exited.
        if not self._close_wafer_transfer_dialog("开始任务前"):
            return
        if not self._close_handeye_dialog("开始任务前"):
            return

        try:
            task = normalize_action_task(self._action_builder())
        except Exception as exc:
            self._append("动作错误", f"生成动作失败：{exc}", _D["error"])
            return

        project_root = Path(__file__).resolve().parents[3]
        output_dir = (
            project_root
            / "Trajectory Photos"
            / datetime.now().strftime("%y%m%d%H%M%S")
        )
        sources = sorted(
            {
                step["source"]
                for step in task["actions"]
                if step["type"] in {"capture", "start_video", "stop_video"}
            }
        )
        point_count = sum(step["type"] == "record_point" for step in task["actions"])
        photo_count = sum(step["type"] == "capture" for step in task["actions"])
        video_count = sum(step["type"] == "start_video" for step in task["actions"])
        runtime_move_count = sum(
            step["type"] == "runtime_move_joints" for step in task["actions"]
        )
        do_steps = [step for step in task["actions"] if step["type"] == "set_do"]
        do_channels = sorted({int(step["channel"]) for step in do_steps})
        confirmation = QMessageBox(self)
        confirmation.setIcon(QMessageBox.Icon.Warning)
        confirmation.setWindowTitle("确认执行动作")
        confirmation.setText(
            f"动作：{task['name']}\n"
            f"相机源：{', '.join(f'#{source}' for source in sources) or '无'}\n"
            f"记录点：{point_count}；照片：{photo_count}；录像：{video_count}\n"
            f"运行时人工确认关节运动：{runtime_move_count}\n"
            f"DO写入：{len(do_steps)}次；通道："
            f"{', '.join(f'DO{channel}' for channel in do_channels) or '无'}\n"
            "相机坐标：旋转相机按 Rz 计算；源1按 J1+J2 计算\n\n"
            "动作会按脚本自动运动、打开所需相机源并切换列出的DO。\n"
            f"输出：{output_dir}\n\n"
            "请确认机械臂位于脚本要求的起点、工作区无障碍物、速度较低，"
            "且物理急停可用。"
        )
        confirmation.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        confirmation.setDefaultButton(QMessageBox.StandardButton.No)
        # 本页使用深色主题，但 Windows 的 QMessageBox 内容区可能仍是白底；
        # 对此确认框做局部覆盖，避免全局白色文字落在原生白色背景上。
        confirmation.setStyleSheet(
            "QMessageBox { background-color: #FFFFFF; color: #111827; }"
            "QMessageBox QLabel { color: #111827; background: transparent; }"
            "QMessageBox QPushButton {"
            " color: #111827; background-color: #F3F4F6;"
            " border: 1px solid #9CA3AF; border-radius: 4px;"
            " padding: 6px 18px; min-width: 70px;"
            "}"
            "QMessageBox QPushButton:hover { background-color: #E5E7EB; }"
            "QMessageBox QPushButton:default {"
            " border: 2px solid #2563EB; background-color: #DBEAFE;"
            "}"
        )
        answer = confirmation.exec()
        if answer != QMessageBox.StandardButton.Yes:
            return

        self._action_runtime = None
        if self._action_runtime_factory is not None:
            try:
                runtime = self._action_runtime_factory(output_dir, self)
                if runtime is None:
                    raise ValueError("create_task_runtime() 未返回运行时对象")
                on_photo_saved = getattr(runtime, "on_photo_saved", None)
                on_task_finished = getattr(runtime, "on_task_finished", None)
                if not callable(on_photo_saved) or not callable(on_task_finished):
                    raise ValueError(
                        "任务运行时必须实现 on_photo_saved(path) 和 "
                        "on_task_finished(ok, message, output_dir)"
                    )
                on_runtime_move = getattr(
                    runtime,
                    "on_runtime_move_joints_requested",
                    None,
                )
                if runtime_move_count and not callable(on_runtime_move):
                    raise ValueError(
                        "含runtime_move_joints的任务运行时必须实现 "
                        "on_runtime_move_joints_requested(request)"
                    )
                fatal_error = getattr(runtime, "fatal_error", None)
                if fatal_error is not None:
                    connect_fatal_error = getattr(fatal_error, "connect", None)
                    if not callable(connect_fatal_error):
                        raise ValueError("任务运行时 fatal_error 必须是Qt信号")
                    connect_fatal_error(self._on_action_runtime_fatal_error)
                self._action_runtime = runtime
            except Exception as exc:
                QMessageBox.critical(self, "任务准备失败", str(exc))
                self._append("动作错误", f"任务运行时启动失败：{exc}", _D["error"])
                return
        elif runtime_move_count:
            QMessageBox.critical(
                self,
                "任务准备失败",
                "runtime_move_joints任务必须提供create_task_runtime()。",
            )
            self._append(
                "动作错误",
                "动态关节运动缺少任务运行时；已拒绝启动",
                _D["error"],
            )
            return

        # The action opens sources 0/1/2 itself.  Release the preview capture to
        # avoid DirectShow device contention, then restore it after the run.
        self._resume_camera_index = None
        if self._cam is not None:
            self._resume_camera_index = self._cam.source_index
            if not self._stop_camera_thread("任务启动前释放相机"):
                self._resume_camera_index = None
                self._action_runtime = None
                return
            self._btn_cam.setText("连接相机")
            self._cam_lbl.setText("动作执行中 — 相机预览已临时释放")

        self._action_task = task
        self._action_worker = ActionWorker(
            self._ctrl,
            task,
            output_dir,
            camera_position_calculator=self._action_camera_calculator,
            source_position_calculators=self._action_source_position_calculators,
            parent=self,
        )
        self._action_worker.progress.connect(
            lambda message: self._append("动作", message, _D["accent"])
        )
        self._action_worker.photo_saved.connect(
            lambda path: self._append("照片", f"已保存 {path}", _D["success"])
        )
        if self._action_runtime is not None:
            self._action_worker.photo_saved.connect(
                self._action_runtime.on_photo_saved
            )
        self._action_worker.video_saved.connect(
            lambda path: self._append("录像", f"已保存 {path}", _D["success"])
        )
        self._action_worker.point_recorded.connect(
            lambda name: self._append("采点", f"已记录 {name}", _D["success"])
        )
        self._action_worker.operator_checkpoint_requested.connect(
            self._on_action_operator_checkpoint
        )
        if runtime_move_count:
            self._action_worker.runtime_move_joints_requested.connect(
                self._on_action_runtime_move_joints
            )
        self._action_worker.run_finished.connect(self._on_action_finished)

        self._set_action_controls_locked(True)
        self._btn_cam.setEnabled(False)
        self._btn_snap.setEnabled(False)
        self._cam_idx.setEnabled(False)
        self._btn_run_action.setText("停止动作")
        self._btn_run_action.setEnabled(True)
        self._append("动作", f"开始，输出文件夹 {output_dir}", _D["warning"])
        self._action_worker.start()

    def _on_action_runtime_fatal_error(self, message: str) -> None:
        """Stop motion after a task-specific GUI/runtime processing failure."""
        worker = self._action_worker
        if worker is None or not worker.isRunning():
            return
        self._append("动作错误", message, _D["error"])
        worker.request_stop()
        self._btn_run_action.setText("正在安全停止…")
        self._btn_run_action.setEnabled(False)

    def _on_action_runtime_move_joints(self, request: dict) -> None:
        """Let the task runtime display one proposal; never move from the UI.

        The callback may open a modal confirmation dialog.  Its dictionary is
        only transferred to ActionWorker, where it is treated as untrusted and
        subjected to a fresh controller read plus the independent kinematic
        audit.  An exception returns ``abort`` so the worker cannot remain
        blocked waiting for a GUI that has already failed.
        """

        worker = self._action_worker
        if worker is None or not worker.isRunning():
            return
        runtime = self._action_runtime
        callback = getattr(
            runtime,
            "on_runtime_move_joints_requested",
            None,
        )
        request_id = str(request.get("request_id") or "")
        if not callable(callback):
            response = {
                "request_id": request_id,
                "decision": "abort",
                "reason": "任务运行时缺少动态关节运动处理函数",
            }
            worker.respond_runtime_move_joints(response)
            self._append("动作错误", response["reason"], _D["error"])
            return
        try:
            response = callback(dict(request))
            if not isinstance(response, dict):
                raise ValueError(
                    "on_runtime_move_joints_requested必须返回字典"
                )
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            response = {
                "request_id": request_id,
                "decision": "abort",
                "reason": f"Stage7A弹窗/计算失败：{message}",
            }
            self._append("动作错误", response["reason"], _D["error"])
        worker = self._action_worker
        if worker is not None and worker.isRunning():
            worker.respond_runtime_move_joints(response)

    def _on_action_operator_checkpoint(
        self,
        title: str,
        message: str,
        continue_text: str,
        finish_text: str,
    ) -> None:
        """Ask the operator whether a paused repeated scan should continue."""
        worker = self._action_worker
        if worker is None or not worker.isRunning():
            return

        prompt = QMessageBox(self)
        prompt.setIcon(QMessageBox.Icon.Information)
        prompt.setWindowTitle(title)
        prompt.setText(message)
        prompt.setInformativeText(
            "只有此窗口出现且机械臂已经停止时，才可以调整标定板。\n"
            "固定好新的姿态并确认工作区无人后，再点击“继续采集”。"
        )
        continue_button = prompt.addButton(
            continue_text,
            QMessageBox.ButtonRole.AcceptRole,
        )
        finish_button = prompt.addButton(
            finish_text,
            QMessageBox.ButtonRole.RejectRole,
        )
        prompt.setDefaultButton(finish_button)
        prompt.setEscapeButton(finish_button)
        prompt.setStyleSheet(
            "QMessageBox { background-color:#FFFFFF; color:#111827; }"
            "QMessageBox QLabel { color:#111827; background:transparent; }"
            "QMessageBox QPushButton {"
            " color:#111827; background-color:#F3F4F6;"
            " border:1px solid #9CA3AF; border-radius:4px;"
            " padding:7px 18px; min-width:100px;"
            "}"
            "QMessageBox QPushButton:hover { background-color:#E5E7EB; }"
            "QMessageBox QPushButton:default {"
            " border:2px solid #2563EB; background-color:#DBEAFE;"
            "}"
        )
        prompt.exec()
        continue_collection = prompt.clickedButton() is continue_button
        worker = self._action_worker
        if worker is not None and worker.isRunning():
            worker.respond_operator_checkpoint(continue_collection)
            self._append(
                "人工确认",
                "继续采集下一姿态" if continue_collection else "结束采集并计算内参",
                _D["warning"] if continue_collection else _D["success"],
            )

    def _on_action_finished(
        self,
        ok: bool,
        message: str,
        output_dir: str,
    ) -> None:
        runtime = self._action_runtime
        self._action_runtime = None
        runtime_error = ""
        if runtime is not None:
            try:
                runtime.on_task_finished(ok, message, output_dir)
            except Exception as exc:
                runtime_error = str(exc) or exc.__class__.__name__
                ok = False
                message = f"{message}；采集后处理失败：{runtime_error}"
        self._set_action_controls_locked(False)
        self._btn_cam.setEnabled(True)
        self._btn_snap.setEnabled(True)
        self._cam_idx.setEnabled(True)
        self._btn_run_action.setText("执行动作")
        self._btn_run_action.setEnabled(self._action_builder is not None)
        self._append(
            "动作完成" if ok else "动作停止",
            f"{message}；文件夹：{output_dir}",
            _D["success"] if ok else _D["error"],
        )
        self._action_worker = None
        self._restore_one_shot_action()

        resume_index = self._resume_camera_index
        self._resume_camera_index = None
        if resume_index is not None:
            self._cam_idx.setValue(resume_index)
            self._toggle_camera()

    def _on_presets(self, names: list) -> None:
        cur = self._preset_combo.currentText()
        self._preset_combo.clear(); self._preset_combo.addItems(names)
        if cur in names:
            self._preset_combo.setCurrentText(cur)

    def _toggle_camera(self) -> None:
        if self._action_worker is not None and self._action_worker.isRunning():
            self._append("相机", "动作执行期间不能切换或断开相机", _D["warning"])
            return
        if self._cam is not None:
            if not self._close_wafer_transfer_dialog("断开相机前"):
                return
            if not self._close_handeye_dialog("断开相机前"):
                return
            if not self._stop_camera_thread("断开相机"):
                return
            self._btn_cam.setText("连接相机"); self._cam_lbl.setText("相机已断开")
            return
        camera_index = int(self._cam_idx.value())
        generation = self._camera_connection_counts.get(camera_index, 0) + 1
        self._camera_connection_counts[camera_index] = generation
        self._cam = ScaraCameraThread(
            index=camera_index,
            connection_generation=generation,
        )
        self._cam.frame_ready.connect(self._on_frame)
        self._cam.error.connect(self._on_camera_error)
        self._cam.start(); self._btn_cam.setText("断开相机")

    def _open_handeye_demo(self) -> None:
        """Open camera-1 computation only; no controller method is called."""

        if self._action_worker is not None and self._action_worker.isRunning():
            self._append(
                "手眼交互",
                "任务执行期间相机被任务独占，不能打开动态演示",
                _D["warning"],
            )
            return
        if self._handeye_dialog is not None:
            self._handeye_dialog.raise_()
            self._handeye_dialog.activateWindow()
            return
        if not self._close_wafer_transfer_dialog("打开手眼动态演示前"):
            return

        required_source = 1
        if self._cam is not None and self._cam.source_index != required_source:
            if not self._stop_camera_thread("切换到相机1"):
                return
            self._btn_cam.setText("连接相机")
            self._cam_lbl.setText("正在切换到手眼交互所需的相机1……")
        if self._cam is None:
            self._cam_idx.setValue(required_source)
            self._toggle_camera()
        if self._cam is None or self._cam.source_index != required_source:
            self._append("手眼交互", "相机1启动失败", _D["error"])
            return

        try:
            from scara.ui.handeye_demo_dialog import HandEyeDemoDialog

            project_root = Path(__file__).resolve().parents[3]
            dialog = HandEyeDemoDialog(
                project_root,
                self._cam,
                self,
                robot_state_provider=self._handeye_robot_state_snapshot,
            )
            self._handeye_dialog = dialog
            dialog.destroyed.connect(self._on_handeye_dialog_destroyed)
            dialog.local_jacobian_calibration_requested.connect(
                self._start_local_jacobian_calibration
            )
            dialog.stage7b_start_requested.connect(self._start_stage7b)
            dialog.stage7b_stop_requested.connect(self._stop_stage7b)
            dialog.full_tray_start_requested.connect(
                self._start_full_tray_positioning
            )
            dialog.full_tray_stop_requested.connect(self._stop_stage7b)
            dialog.show()
            self._append(
                "手眼交互",
                "动态演示已打开：只计算，机械臂不会移动",
                _D["success"],
            )
        except Exception as exc:
            self._handeye_dialog = None
            QMessageBox.critical(self, "动态演示启动失败", str(exc))
            self._append("手眼交互", f"启动失败：{exc}", _D["error"])

    def _open_wafer_transfer(self) -> None:
        """Open target-locked vision and the supervised XY-only handoff."""

        if self._action_worker is not None and self._action_worker.isRunning():
            self._append(
                "转移视觉",
                "任务执行期间相机被任务独占，不能打开转移视觉",
                _D["warning"],
            )
            return
        if self._wafer_transfer_dialog is not None:
            self._wafer_transfer_dialog.raise_()
            self._wafer_transfer_dialog.activateWindow()
            return
        if not self._close_handeye_dialog("打开转移视觉前"):
            return

        required_source = 1
        if self._cam is not None and self._cam.source_index != required_source:
            if not self._stop_camera_thread("切换到转移视觉相机1"):
                return
            self._btn_cam.setText("连接相机")
            self._cam_lbl.setText("正在切换到转移视觉所需的相机1……")
        if self._cam is None:
            self._cam_idx.setValue(required_source)
            self._toggle_camera()
        if self._cam is None or self._cam.source_index != required_source:
            self._append("转移视觉", "相机1启动失败", _D["error"])
            return

        try:
            from scara.ui.wafer_transfer_dialog import WaferTransferDialog

            project_root = Path(__file__).resolve().parents[3]
            dialog = WaferTransferDialog(
                project_root,
                self._cam,
                self,
                robot_state_provider=self._handeye_robot_state_snapshot,
            )
            self._wafer_transfer_dialog = dialog
            dialog.destroyed.connect(self._on_wafer_transfer_dialog_destroyed)
            dialog.pick_xy_start_requested.connect(self._start_wafer_pick_xy)
            dialog.pick_xy_stop_requested.connect(self._stop_wafer_pick_xy)
            dialog.show()
            self._append(
                "转移视觉",
                "实时转移视觉已打开：默认只读；XY悬空移动需单独ARM",
                _D["success"],
            )
        except Exception as exc:
            self._wafer_transfer_dialog = None
            QMessageBox.critical(self, "转移视觉启动失败", str(exc))
            self._append("转移视觉", f"启动失败：{exc}", _D["error"])

    def _on_handeye_dialog_destroyed(self, _object=None) -> None:
        self._handeye_dialog = None

    def _on_wafer_transfer_dialog_destroyed(self, _object=None) -> None:
        self._wafer_transfer_dialog = None

    def _start_wafer_pick_xy(self) -> None:
        """Start selected-wafer XY-only motion through the sole hardware owner."""

        dialog = self._wafer_transfer_dialog
        if dialog is None:
            return
        if self._action_worker is not None and self._action_worker.isRunning():
            reason = "已有任务或运动会话正在运行"
            self._append("XY悬空定位", reason, _D["warning"])
            dialog.report_pick_xy_start_rejected(reason)
            return
        if self._cam is None or self._cam.source_index != 1 or not self._cam.isRunning():
            reason = "相机1未运行，已拒绝启动"
            self._append("XY悬空定位", reason, _D["error"])
            dialog.report_pick_xy_start_rejected(reason)
            return
        if not self._ctrl.motion_ready(
            maximum_speed_percent=20.0,
            required_mode="T1",
        ):
            reason_provider = getattr(self._ctrl, "motion_readiness_error", None)
            reason = (
                reason_provider(
                    maximum_speed_percent=20.0,
                    required_mode="T1",
                )
                if callable(reason_provider)
                else None
            ) or "控制器未满足运动条件；请确认已连接、已使能且无报警/急停"
            dialog.report_pick_xy_start_rejected(reason)
            return

        project_root = Path(__file__).resolve().parents[3]
        output_dir = (
            project_root
            / "Trajectory Photos"
            / datetime.now().strftime("%y%m%d%H%M%S")
        )
        prepared = False
        try:
            raw_task = dialog.prepare_pick_xy_session(output_dir)
            prepared = True
            task = normalize_action_task(raw_task)
        except Exception as exc:
            if prepared or bool(getattr(dialog, "_pick_xy_active", False)):
                try:
                    dialog.finish_pick_xy_session(False, f"准备失败：{exc}")
                except Exception:
                    pass
            QMessageBox.critical(self, "XY悬空定位准备失败", str(exc))
            self._append("XY悬空定位", f"准备失败：{exc}", _D["error"])
            dialog.report_pick_xy_start_rejected(f"准备失败：{exc}")
            return

        worker = ActionWorker(
            self._ctrl,
            task,
            output_dir,
            camera_position_calculator=self._action_camera_calculator,
            source_position_calculators=self._action_source_position_calculators,
            parent=self,
        )
        self._action_task = task
        self._action_worker = worker
        self._handeye_motion_mode = "wafer_pick_xy"
        worker.progress.connect(
            lambda message: self._append("XY悬空定位", message, _D["accent"])
        )
        worker.runtime_move_joints_requested.connect(
            self._on_wafer_pick_xy_runtime_move
        )
        worker.run_finished.connect(self._on_wafer_pick_xy_finished)
        self._set_action_controls_locked(True)
        self._btn_run_action.setEnabled(False)
        self._btn_cam.setEnabled(False)
        self._cam_idx.setEnabled(False)
        self._btn_snap.setEnabled(False)
        self._append(
            "XY悬空定位",
            f"会话已启动；只允许XY与固定Rz补偿，输出文件夹 {output_dir}",
            _D["warning"],
        )
        worker.start()

    def _on_wafer_pick_xy_runtime_move(self, request: dict) -> None:
        worker = self._action_worker
        dialog = self._wafer_transfer_dialog
        if worker is None or not worker.isRunning():
            return
        if dialog is None:
            worker.respond_runtime_move_joints(
                {
                    "request_id": str(request.get("request_id") or ""),
                    "decision": "abort",
                    "reason": "转移视觉窗口已失效",
                }
            )
            return

        def respond(response: dict) -> None:
            current = self._action_worker
            if current is worker and current.isRunning():
                current.respond_runtime_move_joints(response)

        dialog.begin_pick_xy_request(dict(request), respond)

    def _stop_wafer_pick_xy(self) -> None:
        worker = self._action_worker
        if (
            self._handeye_motion_mode == "wafer_pick_xy"
            and worker is not None
            and worker.isRunning()
        ):
            worker.request_stop()
            self._append(
                "XY悬空定位",
                "已请求安全停止；不会再授权新的XY步骤",
                _D["warning"],
            )

    def _on_wafer_pick_xy_finished(
        self, ok: bool, message: str, output_dir: str
    ) -> None:
        dialog = self._wafer_transfer_dialog
        if dialog is not None:
            try:
                ok, message = dialog.finish_pick_xy_session(ok, message)
            except Exception as exc:
                ok = False
                message = f"{message}；XY悬空定位报告收尾失败：{exc}"
        self._set_action_controls_locked(False)
        self._btn_cam.setEnabled(True)
        self._cam_idx.setEnabled(True)
        self._btn_snap.setEnabled(True)
        self._btn_run_action.setEnabled(self._action_builder is not None)
        self._action_worker = None
        self._handeye_motion_mode = None
        self._append(
            "XY悬空定位完成" if ok else "XY悬空定位停止",
            f"{message}；文件夹：{output_dir}",
            _D["success"] if ok else _D["error"],
        )

    def _restore_one_shot_action(self) -> None:
        """Restore the task-panel import after a hand-eye one-shot Task9 run."""

        saved = self._one_shot_action_restore
        self._one_shot_action_restore = None
        if saved is None:
            return
        (
            self._action_file,
            self._action_builder,
            self._action_camera_calculator,
            source_calculators,
            self._action_runtime_factory,
            self._action_task,
        ) = saved
        self._action_source_position_calculators = dict(source_calculators)
        self._btn_run_action.setEnabled(self._action_builder is not None)

    def _start_local_jacobian_calibration(self, target_name: str) -> None:
        """Run target-locked Task9 through the existing exclusive task worker."""

        if self._action_worker is not None and self._action_worker.isRunning():
            self._append(
                "local Jacobian标定",
                "已有任务或XY定位会话运行中",
                _D["warning"],
            )
            return
        if self._one_shot_action_restore is not None:
            self._append(
                "local Jacobian标定",
                "上一次临时任务尚未完成收尾",
                _D["warning"],
            )
            return

        project_root = Path(__file__).resolve().parents[3]
        task9_path = project_root / "Tasks" / "task9_jacobiantest.py"
        try:
            module_name = (
                f"_scara_local_task9_{target_name}_"
                f"{time.monotonic_ns()}"
            )
            spec = importlib.util.spec_from_file_location(module_name, task9_path)
            if spec is None or spec.loader is None:
                raise ValueError("无法创建Task9模块加载器")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            if getattr(module, "ACTION_API_VERSION", None) != 1:
                raise ValueError("Task9动作API版本无效")
            build_for_target = getattr(module, "build_action_for_target", None)
            runtime_for_target = getattr(
                module, "create_task_runtime_for_target", None
            )
            if not callable(build_for_target) or not callable(runtime_for_target):
                raise ValueError("Task9缺少目标槽参数化入口")
            # Build once before closing the live dialog so geometry/IK errors
            # cannot turn into a partially-started acquisition.
            preview = normalize_action_task(build_for_target(target_name))
        except Exception as exc:
            QMessageBox.critical(self, "local Jacobian标定准备失败", str(exc))
            self._append(
                "local Jacobian标定",
                f"{target_name}准备失败：{exc}",
                _D["error"],
            )
            return

        self._one_shot_action_restore = (
            self._action_file,
            self._action_builder,
            self._action_camera_calculator,
            dict(self._action_source_position_calculators),
            self._action_runtime_factory,
            self._action_task,
        )
        self._action_file = task9_path
        self._action_builder = lambda: build_for_target(target_name)
        self._action_camera_calculator = None
        self._action_source_position_calculators = {}
        self._action_runtime_factory = (
            lambda output_dir, parent: runtime_for_target(
                output_dir, target_name, parent=parent
            )
        )
        self._action_task = preview
        self._append(
            "local Jacobian标定",
            (
                f"已锁定目标{target_name}；Task9只会验证当前已在该槽中心，"
                "随后执行±2mm九点采集"
            ),
            _D["warning"],
        )
        self._on_run_action()
        worker = self._action_worker
        if worker is None:
            self._restore_one_shot_action()

    def _start_stage7b(self) -> None:
        """Start the finite XY loop while keeping camera1/Stage3 live.

        The dialog supplies calculations and evidence only.  The same
        ActionWorker used by imported tasks remains the sole controller owner
        and independently rechecks every proposed joint target.
        """

        dialog = self._handeye_dialog
        if dialog is None:
            return
        if self._action_worker is not None and self._action_worker.isRunning():
            self._append("单点有限闭环", "已有任务或闭环运行中", _D["warning"])
            return
        if self._cam is None or self._cam.source_index != 1 or not self._cam.isRunning():
            self._append("单点有限闭环", "相机1未运行，已拒绝启动", _D["error"])
            return
        if not self._ctrl.motion_ready():
            return
        project_root = Path(__file__).resolve().parents[3]
        output_dir = project_root / "Trajectory Photos" / datetime.now().strftime("%y%m%d%H%M%S")
        try:
            task = normalize_action_task(dialog.prepare_stage7b_session(output_dir))
        except Exception as exc:
            if getattr(dialog, "stage7b_active", False):
                try:
                    dialog.finish_stage7b_session(False, f"准备失败：{exc}")
                except Exception:
                    pass
            QMessageBox.critical(self, "单点有限闭环准备失败", str(exc))
            self._append("单点有限闭环", f"准备失败：{exc}", _D["error"])
            return

        worker = ActionWorker(
            self._ctrl,
            task,
            output_dir,
            camera_position_calculator=self._action_camera_calculator,
            source_position_calculators=self._action_source_position_calculators,
            parent=self,
        )
        self._action_task = task
        self._action_worker = worker
        self._handeye_motion_mode = "single_loop"
        worker.progress.connect(
            lambda message: self._append("单点有限闭环", message, _D["accent"])
        )
        worker.runtime_move_joints_requested.connect(
            self._on_stage7b_runtime_move_joints
        )
        worker.run_finished.connect(self._on_stage7b_finished)
        self._set_action_controls_locked(True)
        self._btn_run_action.setEnabled(False)
        self._btn_cam.setEnabled(False)
        self._cam_idx.setEnabled(False)
        self._btn_snap.setEnabled(False)
        self._append(
            "单点有限闭环",
            f"有限闭环已启动；输出文件夹 {output_dir}",
            _D["warning"],
        )
        worker.start()

    def _start_full_tray_positioning(self) -> None:
        """Start runtime Tray registration, dynamic coarse route and metric loop."""

        dialog = self._handeye_dialog
        if dialog is None:
            return
        if self._action_worker is not None and self._action_worker.isRunning():
            self._append("全盘定位", "已有任务或XY定位会话运行中", _D["warning"])
            return
        if self._cam is None or self._cam.source_index != 1 or not self._cam.isRunning():
            self._append("全盘定位", "相机1未运行，已拒绝启动", _D["error"])
            return
        if not self._ctrl.motion_ready():
            return
        project_root = Path(__file__).resolve().parents[3]
        output_dir = (
            project_root
            / "Trajectory Photos"
            / datetime.now().strftime("%y%m%d%H%M%S")
        )
        try:
            task = normalize_action_task(
                dialog.prepare_full_tray_session(output_dir)
            )
        except Exception as exc:
            if getattr(dialog, "stage7b_active", False):
                try:
                    dialog.finish_stage7b_session(False, f"准备失败：{exc}")
                except Exception:
                    pass
            QMessageBox.critical(self, "全盘定位准备失败", str(exc))
            self._append("全盘定位", f"准备失败：{exc}", _D["error"])
            return

        worker = ActionWorker(
            self._ctrl,
            task,
            output_dir,
            camera_position_calculator=self._action_camera_calculator,
            source_position_calculators=self._action_source_position_calculators,
            parent=self,
        )
        self._action_task = task
        self._action_worker = worker
        self._handeye_motion_mode = "full_tray"
        worker.progress.connect(
            lambda message: self._append("全盘定位", message, _D["accent"])
        )
        worker.runtime_move_joints_requested.connect(
            self._on_stage7b_runtime_move_joints
        )
        worker.run_finished.connect(self._on_stage7b_finished)
        self._set_action_controls_locked(True)
        self._btn_run_action.setEnabled(False)
        self._btn_cam.setEnabled(False)
        self._cam_idx.setEnabled(False)
        self._btn_snap.setEnabled(False)
        self._append(
            "全盘定位",
            f"P22全盘定位已启动；输出文件夹 {output_dir}",
            _D["warning"],
        )
        worker.start()

    def _on_stage7b_runtime_move_joints(self, request: dict) -> None:
        worker = self._action_worker
        dialog = self._handeye_dialog
        if worker is None or not worker.isRunning():
            return
        if dialog is None:
            worker.respond_runtime_move_joints(
                {
                    "request_id": str(request.get("request_id") or ""),
                    "decision": "abort",
                    "reason": "手眼交互动态演示窗口已失效",
                }
            )
            return

        def respond(response: dict) -> None:
            current = self._action_worker
            if current is worker and current.isRunning():
                current.respond_runtime_move_joints(response)

        dialog.begin_stage7b_request(dict(request), respond)

    def _stop_stage7b(self) -> None:
        worker = self._action_worker
        if worker is not None and worker.isRunning():
            worker.request_stop()
            label = (
                "全盘定位"
                if self._handeye_motion_mode == "full_tray"
                else "单点有限闭环"
            )
            self._append(
                label,
                "已请求安全停止；不会再授权新的XY修正",
                _D["warning"],
            )

    def _on_stage7b_finished(self, ok: bool, message: str, output_dir: str) -> None:
        mode = self._handeye_motion_mode
        label = "全盘定位" if mode == "full_tray" else "单点有限闭环"
        dialog = self._handeye_dialog
        if dialog is not None:
            try:
                finished = dialog.finish_stage7b_session(ok, message)
                if (
                    isinstance(finished, tuple)
                    and len(finished) == 2
                ):
                    ok = bool(finished[0])
                    message = str(finished[1])
            except Exception as exc:
                ok = False
                message = f"{message}；{label}报告收尾失败：{exc}"
        self._set_action_controls_locked(False)
        self._btn_cam.setEnabled(True)
        self._cam_idx.setEnabled(True)
        self._btn_snap.setEnabled(True)
        self._btn_run_action.setEnabled(self._action_builder is not None)
        self._action_worker = None
        self._handeye_motion_mode = None
        self._append(
            f"{label}完成" if ok else f"{label}停止",
            f"{message}；文件夹：{output_dir}",
            _D["success"] if ok else _D["error"],
        )

    def _handeye_robot_state_snapshot(
        self, captured_monotonic_s: Optional[float] = None
    ) -> Optional[dict[str, object]]:
        """Return the cached state nearest a capture time; never poll hardware."""

        with self._handeye_state_lock:
            if not self._handeye_controller_connected:
                return None
            latest = self._latest_handeye_robot_state
            if latest is None:
                return None
            history = getattr(self, "_handeye_robot_state_history", None)
            snapshot = latest
            if captured_monotonic_s is not None and history:
                try:
                    target = float(captured_monotonic_s)
                except (TypeError, ValueError, OverflowError):
                    return None
                if not math.isfinite(target):
                    return None
                snapshot = min(
                    history,
                    key=lambda item: abs(
                        float(item["captured_monotonic_s"]) - target
                    ),
                )
            if snapshot is None:
                return None
            return {
                "joints": list(snapshot["joints"]),
                "pose": list(snapshot["pose"]),
                "captured_monotonic_s": float(snapshot["captured_monotonic_s"]),
            }

    def _cache_handeye_robot_state(self, st: dict) -> None:
        """Atomically cache one connected-state sample for Stage6 read-only use."""

        try:
            cached_joints = [float(value) for value in st["joints"]]
            cached_pose = [float(value) for value in st["pose"]]
            if (
                len(cached_joints) != 4
                or len(cached_pose) != 6
                or not all(
                    math.isfinite(value) for value in (*cached_joints, *cached_pose)
                )
            ):
                raise ValueError("invalid joints/pose shape or value")
            candidate = {
                "joints": tuple(cached_joints),
                "pose": tuple(cached_pose),
                "captured_monotonic_s": time.monotonic(),
            }
            with self._handeye_state_lock:
                if not self._handeye_controller_connected:
                    self._latest_handeye_robot_state = None
                    return
                self._latest_handeye_robot_state = candidate
                history = getattr(self, "_handeye_robot_state_history", None)
                if history is None:
                    history = deque(maxlen=64)
                    self._handeye_robot_state_history = history
                history.append(candidate)
        except (KeyError, TypeError, ValueError, OverflowError):
            with self._handeye_state_lock:
                self._latest_handeye_robot_state = None
                history = getattr(self, "_handeye_robot_state_history", None)
                if history is not None:
                    history.clear()

    def _close_handeye_dialog(
        self,
        context: str,
        final_wait_ms: int = 0,
    ) -> bool:
        """Close only after the read-only Stage3 thread has really exited."""

        dialog = self._handeye_dialog
        if dialog is None:
            return True
        try:
            closed = bool(dialog.close())
            if not closed and final_wait_ms > 0:
                monitor = getattr(dialog, "monitor", None)
                stop = getattr(monitor, "stop", None)
                if callable(stop) and bool(stop(final_wait_ms)):
                    closed = bool(dialog.close())
        except RuntimeError:
            self._handeye_dialog = None
            return True
        if closed:
            self._handeye_dialog = None
            return True
        self._append(
            "手眼交互",
            f"{context}被暂停：后台Stage3仍在退出，请稍候重试",
            _D["warning"],
        )
        return False

    def _close_wafer_transfer_dialog(
        self,
        context: str,
        final_wait_ms: int = 0,
    ) -> bool:
        """Close only after the transfer-vision monitor has exited."""

        dialog = self._wafer_transfer_dialog
        if dialog is None:
            return True
        try:
            closed = bool(dialog.close())
            if not closed and final_wait_ms > 0:
                monitor = getattr(dialog, "monitor", None)
                stop = getattr(monitor, "stop", None)
                if callable(stop) and bool(stop(final_wait_ms)):
                    closed = bool(dialog.close())
        except RuntimeError:
            self._wafer_transfer_dialog = None
            return True
        if closed:
            self._wafer_transfer_dialog = None
            return True
        self._append(
            "转移视觉",
            f"{context}被暂停：后台视觉线程仍在退出，请稍候重试",
            _D["warning"],
        )
        return False

    def _stop_camera_thread(self, context: str, timeout_ms: int = 1500) -> bool:
        """Stop DirectShow acquisition and retain the object on timeout."""

        camera = self._cam
        if camera is None:
            return True
        if not camera.stop(timeout_ms):
            self._append(
                "相机",
                f"{context}失败：采集线程尚未退出，已禁止切源或重连",
                _D["warning"],
            )
            return False
        self._cam = None
        return True

    def _on_camera_error(self, message: str) -> None:
        self._append("相机", message, _D["error"])
        # Keep the dialog referenced if a Stage3 call has not exited yet.
        transfer_closed = self._close_wafer_transfer_dialog(
            "相机异常后的窗口关闭"
        )
        dialog_closed = self._close_handeye_dialog("相机异常后的窗口关闭")
        camera_stopped = self._stop_camera_thread("相机异常后的线程关闭")
        if not transfer_closed or not dialog_closed or not camera_stopped:
            self._btn_cam.setText("相机停止中")
            return
        self._btn_cam.setText("连接相机")

    def _on_frame(self, img: QImage) -> None:
        self._cam_lbl.setPixmap(QPixmap.fromImage(img).scaled(
            self._cam_lbl.width(), self._cam_lbl.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def _snapshot(self) -> None:
        if self._cam is not None:
            path = datetime.now().strftime("scara_snap_%Y%m%d_%H%M%S.jpg")
            if self._cam.snapshot(path):
                self._append("相机", f"快照已保存 {path}", _D["success"])
            else:
                self._append("相机", "快照失败：当前没有新鲜画面", _D["error"])

    def _set_css(self, btn: QPushButton, css: str) -> None:
        """运行时改按钮 cssClass 并重新应用样式。"""
        if (btn.property("cssClass") or "") == css:
            return
        btn.setProperty("cssClass", css)
        btn.style().unpolish(btn); btn.style().polish(btn)

    def _reset_do_ui_to_zero(self) -> None:
        """只改界面显示与 applied，不写硬件。"""
        for e in self._do_entries:
            e["applied"] = 0
            lvl = e["lvl"]
            lvl.blockSignals(True)
            lvl.setValue(0)
            lvl.blockSignals(False)

    def _zero_all_dos(self, reason: str = "") -> None:
        """安全清零：界面归零 + 同步把已配置通道写成 0（真关泵/阀）。"""
        self._reset_do_ui_to_zero()
        chs = [e["ch"] for e in self._do_entries]
        if not chs:
            return
        tag = f"清零DO({reason})" if reason else "清零DO"
        ok, msg = self._ctrl.zero_do_channels_sync(chs)
        if ok:
            self._append("信息", f"{tag} 完成", _D["text2"])
        else:
            self._append("错误", f"{tag} 失败: {(msg or '').strip()[:160]}", _D["error"])

    def _load_do_map(self) -> None:
        """从 scara_do_map.json 恢复泵/阀列表；电平不恢复（一律按 0 显示）。
        无 kind/name/ch 的条目（如 example 里的 _note）会跳过。"""
        p = Path(DO_MAP_FILE)
        if not p.is_file():
            return
        try:
            rows = json.loads(p.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("scara_do_map.json 读取失败: %s", exc)
            return
        if not isinstance(rows, list):
            logger.warning("scara_do_map.json 格式错误：应为数组")
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            kind = row.get("kind")
            name = row.get("name")
            if kind not in ("pump", "valve") or not isinstance(name, str) or not name.strip():
                continue
            try:
                ch = int(row.get("ch"))
            except (TypeError, ValueError):
                continue
            if ch < 1 or ch > 16:
                continue
            if any(e["ch"] == ch for e in self._do_entries):
                continue
            if any(e["kind"] == kind and e["name"] == name.strip() for e in self._do_entries):
                continue
            self._add_do_row(kind, name.strip(), ch)

    def _save_do_map(self) -> None:
        """新建/删除后写入 scara_do_map.json（只存映射，不存 0/1）。"""
        data = [{"kind": e["kind"], "name": e["name"], "ch": e["ch"]} for e in self._do_entries]
        try:
            Path(DO_MAP_FILE).write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("写入 scara_do_map.json 失败: %s", exc)

    def _apply_do_level(self, entry: dict) -> None:
        """回车确认：才把当前 0/1 下发到控制器。"""
        lvl = entry["lvl"]
        v = lvl.value()
        if v not in (0, 1):
            lvl.blockSignals(True)
            lvl.setValue(entry["applied"])
            lvl.blockSignals(False)
            return
        entry["applied"] = v
        self._ctrl.cmd_set_do(entry["ch"], v)

    def _revert_do_level_if_needed(self, entry: dict) -> None:
        """失焦但未按回车：还原为上次已下发电平，避免误触写入。"""
        lvl = entry["lvl"]
        if lvl.value() == entry["applied"]:
            return
        lvl.blockSignals(True)
        lvl.setValue(entry["applied"])
        lvl.blockSignals(False)

    def _delete_do_row(self, entry: dict) -> None:
        """删除一行映射并更新 JSON；不自动写 DO。"""
        ans = QMessageBox.question(
            self, "删除 DO 接口",
            f"确认删除「{entry['name']}」（DO{entry['ch']}）？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        self._do_entries.remove(entry)
        entry["widget"].deleteLater()
        self._save_do_map()

    def _set_do_ui_enabled(self, enabled: bool) -> None:
        """未连接时整块 DO 界面锁定（不能新建/改电平/删除）；安全清零仍可写硬件。"""
        self._btn_add_do.setEnabled(enabled)
        for e in self._do_entries:
            e["lvl"].setEnabled(enabled)
            e["del_btn"].setEnabled(enabled)
        if hasattr(self, "_do_tip"):
            self._do_tip.setStyleSheet(
                f"color:{_D['text2']};" if enabled else f"color:{_D['warning']};"
            )

    def _add_do_row(self, kind: str, name: str, ch: int) -> None:
        """在泵/阀列追加一行：名称、DO 号、0/1（回车写入）、删除。"""
        row_w = QWidget()
        row = QHBoxLayout(row_w); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(4)
        name_lbl = QLabel(name); name_lbl.setObjectName("kv")
        do_lbl = QLabel(f"DO{ch}"); do_lbl.setObjectName("muted")
        lvl = QSpinBox(); lvl.setRange(0, 1); lvl.setFixedWidth(44)
        lvl.setKeyboardTracking(False)
        lvl.setValue(0)
        lvl.setToolTip("0=低电平  1=高电平；按回车写入（须已连接）")
        del_btn = _btn("删")
        entry: dict = {
            "kind": kind, "name": name, "ch": ch,
            "widget": row_w, "lvl": lvl, "del_btn": del_btn, "applied": 0,
        }
        lvl.lineEdit().returnPressed.connect(lambda: self._apply_do_level(entry))
        lvl.lineEdit().editingFinished.connect(lambda: self._revert_do_level_if_needed(entry))
        del_btn.clicked.connect(lambda _, e=entry: self._delete_do_row(e))
        # 跟随当前连接状态：未连接时新建按钮本就禁用，这里仍按状态设，避免误开
        can_edit = self._ctrl.is_connected()
        lvl.setEnabled(can_edit)
        del_btn.setEnabled(can_edit)
        row.addWidget(name_lbl); row.addWidget(do_lbl); row.addStretch(1)
        row.addWidget(lvl); row.addWidget(del_btn)
        col = self._do_pump_col if kind == "pump" else self._do_valve_col
        col.addWidget(row_w)
        self._do_entries.append(entry)

    def _show_add_do_dialog(self) -> None:
        """新建对话框。校验通过时先记下 kind/name/ch，再关窗——避免关窗后下拉框回到默认「泵」。"""
        dlg = QDialog(self)
        dlg.setWindowTitle("新建 DO 接口")
        v = QVBoxLayout(dlg)
        form = QGridLayout(); form.setHorizontalSpacing(8); form.setVerticalSpacing(6)
        type_cb = QComboBox(); type_cb.addItems(["泵", "阀"])
        name_edit = QLineEdit(); name_edit.setPlaceholderText("自定义名称")
        do_spin = QSpinBox(); do_spin.setRange(1, 16)
        # 加宽以便显示两位数 DO10–16；关键盘跟踪避免输入中间态
        do_spin.setMinimumWidth(56); do_spin.setKeyboardTracking(False)
        form.addWidget(QLabel("类型"), 0, 0); form.addWidget(type_cb, 0, 1)
        form.addWidget(QLabel("名称"), 1, 0); form.addWidget(name_edit, 1, 1)
        form.addWidget(QLabel("DO号"), 2, 0); form.addWidget(do_spin, 2, 1)
        v.addLayout(form)
        btns = QHBoxLayout(); btns.addStretch(1)
        cancel = _btn("取消"); ok = _btn("确定", "primary")
        cancel.clicked.connect(dlg.reject)
        btns.addWidget(cancel); btns.addWidget(ok)
        v.addLayout(btns)
        accepted_row: Optional[tuple[str, str, int]] = None

        def _try_accept() -> None:
            nonlocal accepted_row
            kind = "pump" if type_cb.currentIndex() == 0 else "valve"
            name = name_edit.text().strip()
            if not name:
                QMessageBox.warning(dlg, "新建 DO 接口", "请填写名称。")
                return
            ch = do_spin.value()
            for e in self._do_entries:
                if e["ch"] == ch:
                    QMessageBox.warning(dlg, "新建 DO 接口", f"DO{ch} 已绑定到「{e['name']}」。")
                    return
                if e["kind"] == kind and e["name"] == name:
                    QMessageBox.warning(dlg, "新建 DO 接口", f"「{name}」已存在。")
                    return
            accepted_row = (kind, name, ch)
            dlg.accept()

        ok.clicked.connect(_try_accept)

        if dlg.exec() != QDialog.DialogCode.Accepted or accepted_row is None:
            return
        kind, name, ch = accepted_row
        self._add_do_row(kind, name, ch)
        self._save_do_map()

    def _on_home_clicked(self) -> None:
        ans = QMessageBox.question(
            self, "回零确认",
            "回零会让机械臂自动运动到原点，确认继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ans == QMessageBox.StandardButton.Yes:
            self._ctrl.cmd_home()

    def _on_connect_clicked(self) -> None:
        # 连接握手（启动 snrobot.exe + 读状态）是阻塞 I/O：若在 GUI 线程直接跑，
        # exe 卡住（如控制器上电未就绪/网络挂起）会冻结整个界面，连急停都点不动。
        # 改用 connect_async 放后台线程；结果经 connection_changed 信号回 UI。
        # 连接前先清零 DO（此时 snrobot 未占连接，scara_do 更容易成功）。
        self._btn_conn.setEnabled(False)
        self._zero_all_dos("连接前")
        self._ctrl.connect_async()

    def _on_disconnect_clicked(self) -> None:
        # 先断 snrobot，再清零 DO（此时不再抢连接）
        self._ctrl.disconnect()
        self._zero_all_dos("断开后")

    def _on_estop_clicked(self) -> None:
        # 界面立刻归零；硬件清零放在急停线程里（断 serve → estop → 写 DO=0 → 再拉起）。
        self._reset_do_ui_to_zero()
        self._ctrl.emergency_stop([e["ch"] for e in self._do_entries])

    def _on_conn(self, ok: bool) -> None:
        with self._handeye_state_lock:
            was_connected = self._handeye_controller_connected
            self._handeye_controller_connected = bool(ok)
            if not ok or not was_connected:
                self._latest_handeye_robot_state = None
                self._handeye_robot_state_history.clear()
        self._conn_chip.setText("● 已连接" if ok else "● 未连接")
        self._conn_chip.setStyleSheet(
            f"color:{_D['success']}; background:{_D['chip_on_bg']};"
            if ok else f"color:{_D['muted']}; background:{_D['chip_off_bg']};"
        )
        # 连接/断开 按实际连接状态互斥
        self._btn_conn.setEnabled(not ok)
        self._btn_disc.setEnabled(ok)
        # 未连接时控制类按钮全部置灰；已连接时去使能保持可点（不因灯显示 OFF 而灰掉）
        for b in (self._btn_en, self._btn_dis, self._btn_clr, self._btn_home, self._btn_estop):
            b.setEnabled(ok)
        # DO：未连接时整块锁定，避免不经 snrobot 就能改泵/阀；清零路径不受影响
        self._set_do_ui_enabled(ok)
        if ok:
            # The robot controller retains its previous speed across app
            # restarts.  Re-apply the configured safe preset on every new
            # connection instead of leaving a stale value such as 37% active.
            preset_speed = self._cfg.clamp_speed(
                self._cfg.default_speed_percent
            )
            self._speed.blockSignals(True)
            self._speed.setValue(preset_speed)
            self._speed.blockSignals(False)
            self._ctrl.cmd_set_speed(preset_speed)
        if not ok:
            self._set_css(self._btn_en, "")
            self._set_css(self._btn_clr, "")
            for k in self._light:
                self._set_light(k, "—", True)

    def _on_status(self, st: dict) -> None:
        connected = self._ctrl.is_connected()
        if connected:
            self._cache_handeye_robot_state(st)
        else:
            with self._handeye_state_lock:
                self._handeye_controller_connected = False
                self._latest_handeye_robot_state = None
                self._handeye_robot_state_history.clear()
        en_eff = bool(st.get("effectively_enabled"))
        need_clear = bool(st.get("need_clear"))
        estop = bool(st.get("estop") or st.get("soft_estop"))
        warn = int(st.get("warn", 0))
        en_raw = int(st.get("enable", 0)) == 1

        # 使能：仅「有效使能」时亮绿；急停/报警时不可点（引导去清报警）
        self._set_css(self._btn_en, "success" if en_eff else "")
        # 清报警：急停或报警时亮绿，提示下一步该点它
        self._set_css(self._btn_clr, "success" if (connected and need_clear) else "")
        if connected:
            self._btn_en.setEnabled(not need_clear and not en_eff)
            self._btn_dis.setEnabled(en_raw and not need_clear)
            self._btn_clr.setEnabled(True)
        # 模式段跟随实际模式
        idx = {"T1": 0, "T2": 1, "Execute": 2}.get(st.get("mode", ""), None)
        if idx is not None:
            b = self._mode_grp.button(idx)
            if b is not None and not b.isChecked():
                self._mode_grp.blockSignals(True); b.setChecked(True); self._mode_grp.blockSignals(False)

        for i in range(4):
            self._joint_v[i].setText(f"{st['joints'][i]:.2f}")
            lo, hi = JOINT_RANGE[i]
            self._joint_bar[i].setValue(int(max(0, min(100, (st['joints'][i] - lo) / (hi - lo) * 100))))
        for i in range(6):
            self._pose_v[i].setText(f"{st['pose'][i]:.2f}")
        self._speed_bar.setValue(int(round(st.get("speed", 0))))
        self._set_light("使能", "ENABLE_ON" if en_eff else "ENABLE_OFF", en_eff)
        self._set_light("运行状态", "ALARM_ON" if warn else "ALARM_OFF", warn == 0)
        self._set_light("急停", "ESTOP_ON" if estop else "ESTOP_OFF", not estop)
        self._set_light("模式", st.get("mode", "—"), True, info=True)
        self._set_light("循环", "单循环", True, info=True)
        self._set_light("机械锁", "ON" if st.get("mechlock") else "OFF", st.get("mechlock") == 0)
        self._io_lbl.setText(f"DI ON: {st.get('di_on',0)}    DO ON: {st.get('do_on',0)}")
        do_list = set(st.get("do_list", []))
        for i, cell in enumerate(self._io_cells):
            name = "ioOn" if (i + 1) in do_list else "io"
            if cell.objectName() != name:
                cell.setObjectName(name); cell.style().unpolish(cell); cell.style().polish(cell)

    def _set_light(self, key: str, text: str, good: bool, info: bool = False) -> None:
        lbl = self._light.get(key)
        if not lbl:
            return
        lbl.setText(text)
        color = _D["primary"] if info else (_D["success"] if good else _D["error"])
        lbl.setStyleSheet(f"font-weight:700; font-size:12px; color:{color};")

    def _append(self, tag: str, msg: str, color: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.appendHtml(f'<span style="color:{_D["muted"]}">{ts}</span> '
                             f'<span style="color:{color}">[{tag}]</span> {msg}')

    def cleanup(self) -> None:
        try:
            # Stage7B may have an ActionWorker waiting on the live Stage3
            # dialog.  Stop and join the sole hardware owner before asking the
            # dialog/monitor to close, otherwise closeEvent correctly refuses
            # destruction while the closed loop is armed.
            if self._action_worker is not None and self._action_worker.isRunning():
                self._action_worker.request_stop()
                self._action_worker.wait(15000)
            transfer_dialog = self._wafer_transfer_dialog
            if transfer_dialog is not None and getattr(
                transfer_dialog, "_pick_xy_active", False
            ):
                try:
                    transfer_dialog.finish_pick_xy_session(
                        False, "应用退出时安全停止"
                    )
                except Exception:
                    pass
            dialog = self._handeye_dialog
            if dialog is not None and getattr(dialog, "stage7b_active", False):
                try:
                    dialog.finish_stage7b_session(False, "应用退出时安全停止")
                except Exception:
                    pass
            # Give an in-flight Stage3 call one longer bounded chance to finish.
            # A failed close retains the dialog reference instead of destroying
            # a live QThread.
            self._close_wafer_transfer_dialog(
                "退出清理",
                final_wait_ms=15000,
            )
            self._close_handeye_dialog("退出清理", final_wait_ms=15000)
            # 主界面退出：先断 snrobot，再清零 DO，避免抢连接导致关泵失败。
            if self._owns:
                if self._ctrl.is_connected():
                    self._ctrl.disconnect()
                self._zero_all_dos("退出")
            if self._cam is not None:
                self._stop_camera_thread("退出清理相机", timeout_ms=5000)
            if self._owns:
                self._ctrl.cleanup()
        except Exception as exc:  # pragma: no cover
            logger.warning("SCARA 控制界面清理失败: %s", exc)
