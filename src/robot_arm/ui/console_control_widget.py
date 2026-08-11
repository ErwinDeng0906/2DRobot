# -*- coding: utf-8 -*-
"""机械臂控制台 —— 接线 widget（ArmConsoleWidget 新 UI + 后端控制逻辑）。

ArmConsoleWidget 只负责 UI（三栏平铺、深色玻璃，严格照效果图）；本类把它的所有
钩子（btn_connect / jog_clicked / sw_remote / wp_list…）接到后端方法上，复用
src/devices/robot_arm 已验证的 sim/thrift/http 后端与控制链，并实现 DeviceModule
契约（cleanup / is_device_connected）。替代旧 control_widget.py 作为对外主界面。

后端模式：
  - 仿真（sw_sim）：create_backend("sim")，离线全流程
  - 远程（sw_remote）：create_backend("http", base_url=服务器)，经代理控制真机；相机走 http
  - 直连：默认 thrift（本机直接连机械臂；本项目实际是远程为主）
"""
from __future__ import annotations

import math
import threading
import time
import urllib.request
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtWidgets import QWidget, QInputDialog, QMessageBox
from PyQt6.QtGui import QPixmap, QImage

from utils import get_logger
from robot_arm.ui.console_widget import ArmConsoleWidget, CART_AXIS_NAMES
from devices.robot_arm import create_backend, RobotArmStatus
from devices.robot_arm.duco_rpc import ST_ROBOT, ST_SAFE, ST_MODE, ST_PROG

logger = get_logger("robot_arm.console")

POLL_INTERVAL_MS = 150
SYNC_INTERVAL_MS = 400


def _ascii_head(s: str) -> str:
    """取字符串开头的 ASCII 段（如 'RUN运行' -> 'RUN'），用于胶囊简洁显示。"""
    out = []
    for ch in str(s):
        if ord(ch) < 128 and ch not in "()":
            out.append(ch)
        else:
            break
    return "".join(out).strip()


# ── 后台 worker（连接 / 运动可能阻塞）─────────────────────────────────────
class _ConnectWorker(QThread):
    done = pyqtSignal(bool, str)

    def __init__(self, backend, parent=None):
        super().__init__(parent)
        self._backend = backend

    def run(self):
        try:
            self._backend.connect()
            self.done.emit(True, "")
        except Exception as e:
            self.done.emit(False, str(e))


class _MotionWorker(QThread):
    done = pyqtSignal(bool, str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            self._fn()
            self.done.emit(True, "")
        except Exception as e:
            self.done.emit(False, str(e))


# ── 相机抓帧线程（拉 snapshot 单帧）──────────────────────────────────────
class _CamWorker(QThread):
    frame = pyqtSignal(bytes)
    failed = pyqtSignal(str)

    def __init__(self, url_getter, interval_ms=200, parent=None):
        super().__init__(parent)
        self._url_getter = url_getter
        self._interval = interval_ms / 1000.0
        self._running = False

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            url = self._url_getter()
            if not url:
                time.sleep(0.3)
                continue
            try:
                with urllib.request.urlopen(url, timeout=2.0) as resp:
                    data = resp.read()
                if data:
                    self.frame.emit(data)
            except Exception as e:
                self.failed.emit(str(e))
                time.sleep(0.5)
            time.sleep(self._interval)


class ArmConsoleControlWidget(ArmConsoleWidget):
    """带后端控制逻辑的机械臂控制台。对外主界面。"""

    status_updated = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None,
                 backend=None, owns_backend: bool = True,
                 on_backend_changed=None):
        super().__init__(parent)
        self._owns_backend = owns_backend
        # 后端换新实例时的通知回调：连接时 _on_connect 会 create_backend 造一个**新** http
        # 后端实例并赋给 self.backend；模块用此回调把 _shared_backend 同步到该实时后端，
        # 否则 orchestrator 等外部持有者会一直拿着创建时的旧 sim 引用（connected=False）。
        self._on_backend_changed = on_backend_changed
        self._backend_obj = None
        # 不再有仿真：默认就是远程（经服务器代理控制真机）。
        self.backend = backend if backend is not None else create_backend("sim")
        self._mode = "http"   # 恒为远程
        self._connect_worker: Optional[_ConnectWorker] = None
        self._motion_worker: Optional[_MotionWorker] = None
        self._last_status: Optional[RobotArmStatus] = None
        self._cam_worker: Optional[_CamWorker] = None
        self._cam_fps_t0 = time.monotonic()
        self._cam_count = 0

        self._wire_signals()
        self._refresh_waypoints()
        self.lbl_mode.setText("远程")
        self._update_controls_enabled()

        # 定时器
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_status)
        self._sync_timer = QTimer(self)
        self._sync_timer.timeout.connect(self._update_controls_enabled)
        self._sync_timer.start(SYNC_INTERVAL_MS)

        logger.info("ArmConsoleControlWidget 初始化 (owns_backend=%s)", owns_backend)

    @property
    def backend(self):
        return self._backend_obj

    @backend.setter
    def backend(self, value):
        # 任何一次换后端（初始化 / 连接 / 模式切换）都在这一处通知外部持有者，
        # 避免连接后 orchestrator/module 仍持旧后端引用。
        self._backend_obj = value
        cb = getattr(self, "_on_backend_changed", None)
        if cb is not None:
            try:
                cb(value)
            except Exception as e:
                logger.warning("backend 变更通知失败: %s", e)

    # ════════════════════════════════════════════════════════════════════
    #  信号接线
    # ════════════════════════════════════════════════════════════════════
    def _wire_signals(self):
        self.btn_connect.clicked.connect(self._on_connect)
        self.btn_disconnect.clicked.connect(self._on_disconnect)
        self.btn_power.clicked.connect(lambda: self._guard(self.backend.power_on, "上电"))
        self.btn_enable.clicked.connect(lambda: self._guard(self.backend.enable, "使能"))
        self.btn_disable.clicked.connect(lambda: self._guard(self.backend.disable, "下使能"))
        self.btn_estop.clicked.connect(self._on_estop)

        self.sw_sim.toggled.connect(self._on_sim_toggled)
        self.sw_remote.toggled.connect(self._on_remote_toggled)
        # JOINT/CART 切换 → 重打 JOG 按键标签（J1..J6 ↔ X/Y/Z/Rx/Ry/Rz）
        self.seg_joint.toggled.connect(lambda on: self.relabel_jog(cartesian=not on))

        self.jog_clicked.connect(self._on_jog)

        # 相机开关（拨动启用画面）；设备切换时若正在显示则重启拉流
        self.chk_cam_enable.toggled.connect(self._on_cam_toggled)
        if hasattr(self, "cam_device"):
            self.cam_device.currentIndexChanged.connect(self._on_cam_device_changed)

        # 点位
        self.wp_list.itemDoubleClicked.connect(lambda _: self._goto_selected_waypoint())
        self.btn_wp_goto.clicked.connect(self._goto_selected_waypoint)
        self.btn_wp_record.clicked.connect(self._record_waypoint)
        self.btn_wp_delete.clicked.connect(self._delete_waypoint)

        # 拖动示教（进入/退出，带二次确认）
        if hasattr(self, "btn_teach"):
            self.btn_teach.clicked.connect(self._on_teach_toggled)

        # 序列
        self.btn_seq_run.clicked.connect(self._seq_run)
        self.btn_seq_pause.clicked.connect(self._seq_pause)
        self.btn_seq_stop.clicked.connect(self._seq_stop)

        # 禁区
        self.btn_nogo_add.clicked.connect(self._nogo_add)

        # 序列引擎（懒建）
        self._seq_engine = None

    # ════════════════════════════════════════════════════════════════════
    #  模式切换
    # ════════════════════════════════════════════════════════════════════
    def _on_sim_toggled(self, checked: bool):
        if self.is_device_connected():
            self.sw_sim.blockSignals(True); self.sw_sim.setChecked(self._mode == "sim"); self.sw_sim.blockSignals(False)
            self._msg("请先断开再切换模式"); return
        if checked:
            # 仿真与远程互斥
            self.sw_remote.blockSignals(True); self.sw_remote.setChecked(False); self.sw_remote.blockSignals(False)
            self._mode = "sim"
            self.backend = create_backend("sim")
            self.lbl_mode.setText("仿真")
            self._msg("已切到仿真模式")
        else:
            # 取消仿真 → 若远程也没开，回到 thrift 直连
            if not self.sw_remote.isChecked():
                self._mode = "thrift"
                self.lbl_mode.setText("直连")
                self._msg("已切到直连模式")

    def _on_remote_toggled(self, checked: bool):
        if self.is_device_connected():
            self.sw_remote.blockSignals(True); self.sw_remote.setChecked(self._mode == "http"); self.sw_remote.blockSignals(False)
            self._msg("请先断开再切换模式"); return
        if checked:
            self.sw_sim.blockSignals(True); self.sw_sim.setChecked(False); self.sw_sim.blockSignals(False)
            self._mode = "http"
            self.lbl_mode.setText("远程")
            self._msg("远程模式：连接将经服务器代理控制真机")
        else:
            self._mode = "sim"
            self.sw_sim.blockSignals(True); self.sw_sim.setChecked(True); self.sw_sim.blockSignals(False)
            self.backend = create_backend("sim")
            self.lbl_mode.setText("仿真")
            self._msg("已切回仿真模式")

    # ════════════════════════════════════════════════════════════════════
    #  连接 / 断开
    # ════════════════════════════════════════════════════════════════════
    def _on_connect(self):
        if self.is_device_connected():
            return
        ip = self.ip_edit.text().strip()
        port = self.port_spin.value()
        try:
            # 恒为远程：经服务器代理连真机
            from devices.robot_arm import load_config
            cfg = load_config()
            cfg["connection"]["ip"] = ip
            cfg["connection"]["rpc_port"] = port
            base = self.server_edit.text().strip() or "http://127.0.0.1:8080"
            if not base.startswith("http"):
                base = "http://" + base
            self.backend = create_backend("http", cfg=cfg, base_url=base)
        except Exception as e:
            self._msg("配置错误: %s" % e); return

        if self._seq_engine is not None:
            self._seq_engine._backend = self.backend

        self._msg("连接中…")
        self.btn_connect.setEnabled(False)
        self._connect_worker = _ConnectWorker(self.backend, self)
        self._connect_worker.done.connect(self._on_connect_done)
        self._connect_worker.start()

    def _on_connect_done(self, ok: bool, err: str):
        self.btn_connect.setEnabled(True)
        if ok:
            self._msg("已连接真机，可上电→使能后操作")
            self._poll_timer.start(POLL_INTERVAL_MS)
            self._refresh_waypoints()   # 从服务器拉真机点位
            if self._seq_engine is not None:
                self._seq_engine._backend = self.backend
        else:
            self._msg("连接失败: %s" % err)

    def _on_disconnect(self):
        self._poll_timer.stop()
        self._stop_camera()
        try:
            self.backend.disconnect()
        except Exception as e:
            self._msg("断开异常: %s" % e)
        self._msg("已断开")
        try:
            self._apply_status(self.backend.get_status())
        except Exception:
            pass

    def _on_estop(self):
        # 急停也走后台线程：armweb POST 可能阻塞，绝不能卡 UI。急停永远立即受理、
        # 不做「命令进行中」拦截（安全第一），用独立 worker 不与上电/使能互斥。
        self._msg("⚠ 急停中…")
        self._estop_worker = _MotionWorker(self.backend.estop, self)
        self._estop_worker.done.connect(
            lambda ok, err: self._msg("⚠ 已急停" if ok else "急停异常: %s" % err))
        self._estop_worker.start()

    def _guard(self, fn, name):
        """上电/使能/下使能等后端命令放后台线程执行，避免卡 UI。

        http 后端的 power_on/enable/disable 都是同步 HTTP POST（经 armweb→服务器→DUCO RPC）；
        armweb 慢或不可达时会阻塞事件循环 → 点「使能」界面卡死。故一律走 _MotionWorker 后台
        线程，结果经 done 信号回主线程写状态栏。同名命令进行中则忽略重复点击（防连点堆积）。
        """
        w = getattr(self, "_cmd_worker", None)
        if w is not None and w.isRunning():
            self._msg("命令执行中，请稍候…")
            return
        self._msg("%s…" % name)
        self._cmd_worker = _MotionWorker(fn, self)
        self._cmd_worker.done.connect(
            lambda ok, err, nm=name: self._msg("%s 完成" % nm if ok else "%s 失败: %s" % (nm, err)))
        self._cmd_worker.start()

    # ════════════════════════════════════════════════════════════════════
    #  JOG
    # ════════════════════════════════════════════════════════════════════
    def _on_jog(self, idx: int, sign: int):
        if self.seg_joint.isChecked():
            # 关节点动（既有安全路径：jog_joint 已接入 task-id/心跳监督）
            step = self._jog_step_deg()
            delta = sign * step
            self._run_motion(
                lambda i=idx, d=math.radians(delta):
                    self.backend.jog_joint(i, d, speed_scale=0.05),
                "JOG J%d %+.1f°" % (idx + 1, delta))
        else:
            # 笛卡尔点动：机器人自带逆解(cal_ikine) + 受监督关节运动(movej_safe)。
            # 平移 X/Y/Z 步 2mm、姿态 Rx/Ry/Rz 步 1°；小步低速(5%)，后端有逆解跳变护栏(>15°拒绝)。
            axis = CART_AXIS_NAMES[idx] if 0 <= idx < len(CART_AXIS_NAMES) else "?"
            STEP_MM, STEP_DEG = 2.0, 1.0
            if idx <= 2:
                delta = sign * STEP_MM / 1000.0          # mm → m
                label = "JOG %s %+.1fmm" % (axis, sign * STEP_MM)
            else:
                delta = sign * math.radians(STEP_DEG)    # ° → rad
                label = "JOG %s %+.1f°" % (axis, sign * STEP_DEG)
            self._run_motion(
                lambda a=idx, d=delta: self.backend.jog_cart(a, d, speed_scale=0.05),
                label)

    def _jog_step_deg(self):
        txt = self.joint_step.currentText().replace("°", "") if self.joint_step.count() else "2.0"
        try:
            return float(txt)
        except Exception:
            return 2.0

    def _run_motion(self, fn, label):
        if not self._require_ready():
            return
        if self._motion_worker is not None and self._motion_worker.isRunning():
            self._msg("运动进行中，请等待"); return
        self._msg("%s…" % label)
        self._motion_worker = _MotionWorker(fn, self)
        self._motion_worker.done.connect(
            lambda ok, err, lb=label: self._msg("%s 完成" % lb if ok else "%s 失败: %s" % (lb, err)))
        self._motion_worker.start()

    def _require_ready(self) -> bool:
        st = self._last_status
        if not self.is_device_connected():
            self._msg("未连接"); return False
        if not (st and st.enabled):
            self._msg("未使能，先点「使能」"); return False
        return True

    # ════════════════════════════════════════════════════════════════════
    #  点位
    # ════════════════════════════════════════════════════════════════════
    def _refresh_waypoints(self):
        try:
            names = self.backend.waypoint_names()
        except Exception:
            names = []
        # 清空 UI 列表并重填（保留新 UI 的行样式）
        self.wp_list.clear()
        for name in names:
            self._add_wp_item(self.wp_list, name, "示教点位 · 双击前往", "前往 →", "#5FD4E0")
        if not names:
            self._add_wp_item(self.wp_list, "（无点位）", "记录当前位姿以新增", "", "#5FD4E0")

    def _selected_waypoint_name(self) -> Optional[str]:
        item = self.wp_list.currentItem()
        if item is None:
            return None
        w = self.wp_list.itemWidget(item)
        if w is None:
            return None
        # wpRow 第一个 QLabel#wpName 即名字
        from PyQt6.QtWidgets import QLabel
        for lab in w.findChildren(QLabel):
            if lab.objectName() == "wpName":
                return lab.text()
        return None

    def _goto_selected_waypoint(self):
        name = self._selected_waypoint_name()
        if not name or name.startswith("（"):
            self._msg("先选一个点位"); return
        scale = 0.2
        try:
            if hasattr(self.backend, "cfg"):
                scale = self.backend.cfg.get("speed", {}).get("global_scale", 0.2)
        except Exception:
            pass
        self._run_motion(
            lambda n=name: self.backend.goto_waypoint(n, speed_scale=scale, block=True),
            "前往「%s」" % name)

    def _record_waypoint(self):
        name, ok = QInputDialog.getText(self, "记录点位", "点位名称：")
        if not ok or not name.strip():
            return
        name = name.strip()
        # 后台线程记录（HTTP POST 阻塞，不能卡 UI）；成功后回主线程刷新点位列表。
        self._msg("记录点位「%s」…" % name)
        self._wp_worker = _MotionWorker(lambda n=name: self.backend.record_waypoint(n), self)
        self._wp_worker.done.connect(
            lambda ok2, err, nm=name: (self._refresh_waypoints(), self._msg("已记录点位「%s」" % nm))
            if ok2 else QMessageBox.warning(self, "记录失败", err))
        self._wp_worker.start()

    def _delete_waypoint(self):
        name = self._selected_waypoint_name()
        if not name or name.startswith("（"):
            self._msg("先选一个点位"); return
        if QMessageBox.question(self, "删除确认", "删除点位「%s」？" % name) != QMessageBox.StandardButton.Yes:
            return
        # 后台线程删除（HTTP POST 阻塞，不能卡 UI）；成功后回主线程刷新点位列表。
        self._msg("删除点位「%s」…" % name)
        self._wp_worker = _MotionWorker(lambda n=name: self.backend.delete_waypoint(n), self)
        self._wp_worker.done.connect(
            lambda ok2, err, nm=name: (self._refresh_waypoints(), self._msg("已删除点位「%s」" % nm))
            if ok2 else QMessageBox.warning(self, "删除失败", err))
        self._wp_worker.start()

    # ════════════════════════════════════════════════════════════════════
    #  拖动示教（牵引/零力）
    # ════════════════════════════════════════════════════════════════════
    def _on_teach_toggled(self, checked: bool):
        """进入/退出拖动示教。进入前强制二次确认 + 负载输入（真机最高风险）。"""
        if checked:
            self._enter_teach()
        else:
            self._exit_teach()

    def _enter_teach(self):
        # 前置：必须已连接 + 已使能 + 无报警。不满足时弹窗明确告知（避免"点了没反应"）。
        st = self._last_status
        if not self.is_device_connected():
            self.btn_teach.setChecked(False)
            self._msg("未连接，无法进入拖动示教")
            QMessageBox.warning(self, "无法进入拖动示教",
                                "尚未连接机械臂。\n\n请先点顶部「连接」，再「上电」→「使能」，然后才能进入拖动示教。")
            return
        if not (st and st.enabled):
            self.btn_teach.setChecked(False)
            self._msg("未使能，先点「使能」再进拖动示教")
            QMessageBox.warning(self, "无法进入拖动示教",
                                "机械臂尚未使能。\n\n请先点「上电」→「使能」，确认左侧状态为 ENABLED，再进入拖动示教。")
            return
        if st and st.alarm != 0:
            self.btn_teach.setChecked(False)
            self._msg("存在报警(0x%X)，请先在示教器复位" % st.alarm)
            QMessageBox.warning(self, "无法进入拖动示教",
                                "机械臂存在报警：0x%X\n\n"
                                "这通常是之前的碰撞/急停未复位导致。\n"
                                "请在【示教器】上做「故障复位/碰撞恢复」清除报警后，\n"
                                "重新上电→使能，再进入拖动示教。\n\n"
                                "（此报警是真机安全锁，软件无法远程清除。）" % st.alarm)
            return

        # 二次确认：风险提示 + 末端负载输入
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                     QDoubleSpinBox, QPushButton as QPB)
        dlg = QDialog(self)
        dlg.setWindowTitle("进入拖动示教 — 安全确认")
        v = QVBoxLayout(dlg)
        warn = QLabel(
            "⚠ 即将进入【零力拖动示教】。\n\n"
            "• 机械臂会进入柔顺状态，可徒手拖动。\n"
            "• 进入瞬间若末端负载设置不符，机械臂可能突然下坠或弹起。\n"
            "• 请确保已有人在机械臂旁手扶，下方无障碍物/人。\n"
            "• 拖到位后点「记录当前」存点位；完成后再点一次按钮退出。\n\n"
            "请填写末端负载（夹具+工件，单位 kg / m）：")
        warn.setWordWrap(True)
        v.addWidget(warn)
        form = QHBoxLayout()
        form.addWidget(QLabel("负载质量(kg)"))
        sp_m = QDoubleSpinBox(); sp_m.setRange(0.0, 20.0); sp_m.setDecimals(2); sp_m.setSingleStep(0.1)
        form.addWidget(sp_m)
        v.addLayout(form)
        form2 = QHBoxLayout()
        sps = {}
        for ax in ("cx", "cy", "cz"):
            form2.addWidget(QLabel(ax))
            s = QDoubleSpinBox(); s.setRange(-1.0, 1.0); s.setDecimals(3); s.setSingleStep(0.01)
            sps[ax] = s; form2.addWidget(s)
        v.addWidget(QLabel("质心 (相对工具系，m)："))
        v.addLayout(form2)
        btns = QHBoxLayout()
        ok = QPB("确认进入"); ok.setProperty("cssClass", "warning")
        cancel = QPB("取消")
        btns.addStretch(1); btns.addWidget(cancel); btns.addWidget(ok)
        v.addLayout(btns)
        ok.clicked.connect(dlg.accept); cancel.clicked.connect(dlg.reject)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            self.btn_teach.setChecked(False)
            self._msg("已取消进入拖动示教"); return

        m = sp_m.value()
        cx, cy, cz = sps["cx"].value(), sps["cy"].value(), sps["cz"].value()
        # 后台线程发进入示教命令（HTTP 阻塞，不能卡 UI）；结果回主线程更新按钮/状态。
        # 校验与安全弹窗仍在 UI 线程同步完成（本就应阻塞等用户确认）。
        self._msg("进入拖动示教中…")
        self._teach_worker = _MotionWorker(
            lambda: self.backend.enter_teach_mode(m, cx, cy, cz), self)
        self._teach_worker.done.connect(self._on_enter_teach_done)
        self._teach_worker.start()

    def _on_enter_teach_done(self, ok: bool, err: str):
        if ok:
            self.btn_teach.setText("⏹ 退出拖动示教（示教中…）")
            self._msg("已进入拖动示教 — 请手扶机械臂拖动到位后「记录当前」")
        else:
            self.btn_teach.setChecked(False)
            self.btn_teach.setText("✋ 进入拖动示教")
            self._msg("进入拖动示教失败: %s" % err)

    def _exit_teach(self):
        # 后台线程发退出示教命令（HTTP 阻塞，不能卡 UI）；无论成败都复位按钮。
        self._msg("退出拖动示教中…")
        self._teach_worker = _MotionWorker(self.backend.exit_teach_mode, self)
        self._teach_worker.done.connect(self._on_exit_teach_done)
        self._teach_worker.start()

    def _on_exit_teach_done(self, ok: bool, err: str):
        self._msg("已退出拖动示教" if ok else "退出拖动示教失败: %s" % err)
        self.btn_teach.setText("✋ 进入拖动示教")
        self.btn_teach.setChecked(False)

    # ════════════════════════════════════════════════════════════════════
    #  序列
    # ════════════════════════════════════════════════════════════════════
    def _ensure_seq_engine(self):
        if self._seq_engine is None:
            from robot_arm.ui.sequence_engine import SequenceEngine
            self._seq_engine = SequenceEngine(self.backend, self)
            self._seq_engine.progress.connect(self._on_seq_progress)
        return self._seq_engine

    def _current_sequence(self):
        from devices.robot_arm import sequences as seqmod
        return seqmod.PICK_PLACE_FULL

    def _seq_run(self):
        if not self._require_ready():
            return
        eng = self._ensure_seq_engine()
        seq = self._current_sequence()
        self._msg("运行序列（%d点）…" % len(seq))
        eng.start(seq)

    def _seq_pause(self):
        if self._seq_engine and self._seq_engine.is_running():
            self._seq_engine.pause()
            self._msg("序列暂停")

    def _seq_stop(self):
        if self._seq_engine:
            self._seq_engine.stop()
        self._msg("序列停止")

    def _on_seq_progress(self, *args):
        try:
            if len(args) >= 2:
                idx, total = args[0], args[1]
                self.seq_progress.setValue(int(idx / max(1, total) * 100))
        except Exception:
            pass

    # ════════════════════════════════════════════════════════════════════
    #  禁区
    # ════════════════════════════════════════════════════════════════════
    def _nogo_add(self):
        self._msg("新增禁区盒：请在弹窗输入范围（占位，后续接编辑器）")

    # ════════════════════════════════════════════════════════════════════
    #  相机
    # ════════════════════════════════════════════════════════════════════
    def _camera_base(self):
        """相机走的服务器地址。完全自包含——相机工具条自己有地址栏，
        不依赖远程开关 / 是否已连机械臂（固定相机随时能看盘）。"""
        # 1) 优先相机工具条自己的地址栏（最直观）
        for attr in ("cam_server", "server_edit"):
            w = getattr(self, attr, None)
            if w is not None:
                s = w.text().strip()
                if s:
                    if not s.startswith("http"):
                        s = "http://" + s
                    return s.rstrip("/")
        # 2) 退回 backend.base_url（已连远程时）
        try:
            base = getattr(self.backend, "base_url", None)
            if base:
                return base.rstrip("/")
        except Exception:
            pass
        return None

    def _camera_url(self):
        base = self._camera_base()
        return (base + "/api/camera/snapshot") if base else None

    def _on_cam_toggled(self, checked: bool):
        if checked:
            self._start_camera()
        else:
            self._stop_camera()

    def _start_camera(self):
        if self._cam_worker is not None:
            return
        base = self._camera_base()
        if not base:
            self.camera.set_hud(status="IDLE")
            self._msg("相机：请先在工具栏填服务器地址（远程模式）")
            self.chk_cam_enable.setChecked(False)
            return
        # 让服务器打开选中的设备号（左/右盘相机可能是不同 video 设备）——放后台线程，
        # 避免 urlopen(timeout=4s) 在切相机时同步卡死 UI。拉流 _CamWorker 会自行重试
        # snapshot，不依赖此 POST 先完成。
        dev = int(self.cam_device.currentText()) if hasattr(self, "cam_device") else 0

        def _open_dev(b=base, d=dev):
            try:
                req = urllib.request.Request(
                    b + "/api/camera/start",
                    data=('{"device": %d}' % d).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST")
                urllib.request.urlopen(req, timeout=4.0).read()
            except Exception as e:
                logger.debug("camera start dev err: %s", e)
        threading.Thread(target=_open_dev, daemon=True).start()
        self._cam_worker = _CamWorker(self._camera_url, 150, self)
        self._cam_worker.frame.connect(self._on_cam_frame)
        self._cam_worker.failed.connect(lambda e: self.camera.set_hud(status="IDLE"))
        self._cam_worker.start()
        self.camera.set_hud(status="REC")
        self._msg("相机画面已启用（设备 %s）" % self.cam_device.currentText())

    def _stop_camera(self):
        if self._cam_worker is not None:
            self._cam_worker.stop()
            self._cam_worker.wait(1500)
            self._cam_worker = None
        self.camera.set_pixmap(None)
        self.camera.set_hud(status="IDLE")
        if hasattr(self, "cam_res_lbl"):
            self.cam_res_lbl.setText("—")

    def _on_cam_device_changed(self, _idx):
        """切换相机设备号（左盘/右盘可能不同 video 设备）：若正在显示则重启拉流。"""
        if self._cam_worker is not None:
            self._stop_camera()
            self._start_camera()

    def _on_cam_frame(self, data: bytes):
        img = QImage.fromData(data)
        if img.isNull():
            return
        self.camera.set_pixmap(QPixmap.fromImage(img))
        self._cam_count += 1
        now = time.monotonic()
        if now - self._cam_fps_t0 >= 1.0:
            fps = self._cam_count / (now - self._cam_fps_t0)
            dev = self.cam_device.currentText() if hasattr(self, "cam_device") else "0"
            res = "%d×%d" % (img.width(), img.height())
            self.camera.set_hud(res="USB · 设备%s · %s" % (dev, res),
                                fps="%.0f fps" % fps, status="REC")
            if hasattr(self, "cam_res_lbl"):
                self.cam_res_lbl.setText("%s · %.0f fps" % (res, fps))
            self._cam_count = 0
            self._cam_fps_t0 = now

    # ════════════════════════════════════════════════════════════════════
    #  轮询 / 状态刷新
    # ════════════════════════════════════════════════════════════════════
    def _poll_status(self):
        try:
            st = self.backend.get_status()
        except Exception as e:
            self._msg("读状态异常: %s" % e); return
        self._apply_status(st)
        self.status_updated.emit(st)
        # 点位列表同步：http 后端的点位是随 /api/status 异步填充的，
        # 连接瞬间还是空的；这里检测到后端点位数变化就刷新 UI 列表。
        try:
            n_be = len(self.backend.waypoint_names())
            if n_be != getattr(self, "_wp_count_shown", -1):
                self._wp_count_shown = n_be
                self._refresh_waypoints()
        except Exception:
            pass

    def _apply_status(self, st: RobotArmStatus):
        self._last_status = st
        if st.connected:
            # 关节
            for i in range(6):
                if i < len(st.joints_deg):
                    deg = st.joints_deg[i]
                    self.joint_vals[i].setText("%+.2f" % deg)
                    self.joint_bars[i].setValue(int(max(-180, min(180, deg)) + 180))
            # TCP
            tcp = st.tcp
            if tcp and len(tcp) >= 6:
                self._tcp_xyz.setText("%.0f · %.0f · %.0f" % (tcp[0]*1000, tcp[1]*1000, tcp[2]*1000))
                self._tcp_rxyz.setText("%.0f · %.0f · %.0f" % (
                    math.degrees(tcp[3]), math.degrees(tcp[4]), math.degrees(tcp[5])))
            # 模式/速度（去掉中文后缀，避免胶囊/标签中英混排过挤）
            mode_txt = ST_MODE.get(st.op_mode, str(st.op_mode))
            self.lbl_mode.setText(_ascii_head(mode_txt))
            # 胶囊（纯英文短词）
            safe_txt = "RUN" if st.safety == 5 else _ascii_head(ST_SAFE.get(st.safety, "—"))
            self._set_chip(self.chip_conn, "ONLINE", "ok")
            self._set_chip(self.chip_safe, safe_txt or "—",
                           "run" if st.safety == 5 else "warn")
            self._set_chip(self.chip_enable, "ENABLED" if st.enabled else "DISABLED",
                           "ok" if st.enabled else "warn")
            alarm = "无" if st.alarm == 0 else hex(st.alarm)
            self._set_chip(self.chip_alarm, alarm, "ok" if st.alarm == 0 else "warn")
        else:
            self._set_chip(self.chip_conn, "OFFLINE", "warn")
            self._set_chip(self.chip_safe, "—", "warn")
            self._set_chip(self.chip_enable, "—", "warn")
            self._set_chip(self.chip_alarm, "—", "warn")
        if st.error:
            self._msg(st.error)
        self._update_controls_enabled()

    def _set_chip(self, chip, text, kind):
        prefix = "● " if kind in ("ok", "run") and text not in ("无", "—") else ""
        chip.setText(prefix + text)
        chip.setProperty("chipKind", kind)
        # 强制刷新样式（property 改变后需重新 polish）
        chip.style().unpolish(chip)
        chip.style().polish(chip)

    def _update_controls_enabled(self):
        connected = self.is_device_connected()
        st = self._last_status
        powered = bool(st and st.powered)
        enabled = bool(st and st.enabled)
        self.btn_connect.setEnabled(not connected)
        self.btn_disconnect.setEnabled(connected)
        self.ip_edit.setEnabled(not connected)
        self.port_spin.setEnabled(not connected)
        self.btn_power.setEnabled(connected and not powered)
        self.btn_enable.setEnabled(connected and powered and not enabled)
        self.btn_disable.setEnabled(connected and enabled)

    def _msg(self, text):
        logger.info("msg: %s", text)
        if hasattr(self, "status_msg"):
            self.status_msg.setText(str(text))
            # 错误/失败 → 红点，其它 → 绿点
            bad = any(k in str(text) for k in ("失败", "错误", "异常", "未连接", "未使能", "请先"))
            color = "#DC4456" if bad else "#1F9D6B"
            if hasattr(self, "status_dot"):
                self.status_dot.setStyleSheet("color: %s; font-size: 11px;" % color)

    # ════════════════════════════════════════════════════════════════════
    #  DeviceModule 契约
    # ════════════════════════════════════════════════════════════════════
    def is_device_connected(self) -> bool:
        try:
            return bool(self.backend.is_connected())
        except Exception:
            return False

    def cleanup(self) -> None:
        self._poll_timer.stop()
        self._sync_timer.stop()
        self._stop_camera()
        if self._connect_worker is not None and self._connect_worker.isRunning():
            self._connect_worker.wait(2000)
        if self._motion_worker is not None and self._motion_worker.isRunning():
            self._motion_worker.wait(3000)
        if self._seq_engine is not None:
            try:
                self._seq_engine.stop()
            except Exception:
                pass
        if self._owns_backend and self.backend is not None:
            try:
                self.backend.disconnect()
            except Exception:
                pass
        logger.info("ArmConsoleControlWidget cleanup (owns_backend=%s)", self._owns_backend)
