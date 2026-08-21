"""
SCARA 机械臂控制器（常驻服务版）

启动 ``snrobot.exe serve`` 保持单连接，从 stdin 逐行下发命令、读到 ``<<END>>`` 为止。
好处：低延迟、连续点动可用（jog 开/停走同一连接）、进程退出/断开自动停机不跑飞。
同步核心方法可脱离 Qt 直接测试；Qt 层用后台线程轮询状态并发信号，避免界面卡顿。
"""

from __future__ import annotations

import json
import math
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, List

from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal, QRunnable, QThreadPool

from utils import get_logger
from scara.config.scara_config import ScaraConfig
from scara.controller.do_client import set_uo_level as do_set_uo_level
from scara.controller.do_client import zero_channels as do_zero_channels
from scara.controller import enable_client as en_client

logger = get_logger("scara.controller")

CART_AXIS = {"X": 7, "Y": 8, "Z": 9}
JOINT_AXIS = {1: 1, 2: 2, 3: 3, 4: 4}
MODE_CODE = {"T1": 1, "T2": 2, "Execute": 3}
DIR_CODE = {"+": 1, "-": 2}   # Forward / Reverse
PRESET_FILE = "scara_presets.json"


class ScaraController(QObject):
    """SCARA 控制器：常驻 serve 进程 + 状态信号。"""

    # snrobot.exe 是黑盒，任何一次交互都必须有上限，否则无人值守流程会永久挂起。
    # 单条命令上限按"最慢的一次整臂移动"留余量；握手另给一个较短的上限。
    SEND_TIMEOUT_S = 120.0
    CONNECT_TIMEOUT_S = 30.0

    connection_changed = pyqtSignal(bool)
    status_updated = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    warning_occurred = pyqtSignal(str)
    info_occurred = pyqtSignal(str)
    command_finished = pyqtSignal(str, bool, str)
    presets_changed = pyqtSignal(list)          # 预设点名称列表变化

    def __init__(self, config: Optional[ScaraConfig] = None, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._cfg = config or ScaraConfig()
        self._proc: Optional[subprocess.Popen] = None
        self._io_lock = threading.Lock()        # 串行化对 serve 进程的读写
        # scara_enable / scara_do 都会临时断 snrobot：同一把锁防止两路同时抢控制器
        self._enable_exe_lock = threading.Lock()
        # 防止预设点、轨迹拍照和手动运动交错下发多轴命令。
        self._motion_sequence_lock = threading.Lock()
        self._connected = False
        self._connecting = False                 # connect_async 在途标志，防重入起双进程
        self._last_status: dict = {}
        self._soft_estop = False              # 软件急停锁存（点急停置位，清报警清除）
        # 安全状态以 scara_enable（Read_PowerOn/Stop/Warning）为准，不用 snrobot ENABLE/CONTROL
        self._safety: dict = {"enable": 0, "estop": False, "warn": 0, "mode": "?"}
        self._pool = QThreadPool.globalInstance()
        self._poller: Optional["_PollThread"] = None
        self._presets: dict = {}
        self._load_presets()
        # 连续点动"续命"定时器：按住期间周期性重发 jog，避免控制器看门狗约1s自动停
        self._jog_active: Optional[tuple] = None   # (code, dir, world)
        self._jog_timer = QTimer(self); self._jog_timer.setInterval(600)
        self._jog_timer.timeout.connect(self._jog_heartbeat)
        # serve 进程输出队列 + 常驻读取线程（见 _send：防止黑盒 exe 卡死导致永久挂起）
        self._rx: "queue.Queue[Optional[str]]" = queue.Queue()
        self._rx_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    #  serve 进程读写
    # ------------------------------------------------------------------ #
    def _send(self, line: str, timeout: Optional[float] = None) -> str:
        """向 serve 进程发一条命令，读到 <<END>> 返回其间输出。失败返回 ''。

        超时返回含 "TIMEOUT" 的字符串 —— `_ok()` 已把 TIMEOUT 列为失败关键词，
        `read_all_sync()` 也会因缺 "JOINT" 返回 None，所以走的是既有失败通路。

        原实现直接 `for raw in p.stdout` 阻塞读，snrobot.exe 一旦卡死或不回
        `<<END>>`，整条无人值守流程会永久挂起且不抛异常、不记日志。改为由常驻
        读取线程喂队列 + 带 deadline 取队列（Windows 上管道没法 select，只能这么做）。
        """
        budget = float(timeout if timeout is not None else self.SEND_TIMEOUT_S)
        with self._io_lock:
            p = self._proc
            if p is None or p.poll() is not None:
                return ""
            # 丢弃上一条命令超时后迟到的残留行，避免串到本次结果里
            while True:
                try:
                    self._rx.get_nowait()
                except queue.Empty:
                    break
            try:
                p.stdin.write(line + "\n")
                p.stdin.flush()
            except Exception as exc:
                logger.warning("serve 写入失败: %s", exc)
                return ""

            out: List[str] = []
            deadline = time.monotonic() + budget
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.error(
                        "serve 命令 %r 超时（%.1fs 未收到 <<END>>），已收到 %d 行",
                        line, budget, len(out),
                    )
                    return "TIMEOUT: snrobot serve 未在 %.1fs 内响应 %r" % (budget, line)
                try:
                    item = self._rx.get(timeout=min(remaining, 0.5))
                except queue.Empty:
                    if p.poll() is not None:
                        logger.error("serve 进程已退出（命令 %r 未完成）", line)
                        return "CONNECT_FAIL: snrobot serve 进程已退出"
                    continue
                if item is None:            # 读取线程报告 stdout 已关闭
                    logger.error("serve stdout 已关闭（命令 %r 未完成）", line)
                    return "CONNECT_FAIL: snrobot serve 输出已关闭"
                if item == "<<END>>":
                    return "\n".join(out)
                out.append(item)

    def _rx_pump(self, proc: subprocess.Popen) -> None:
        """常驻读取线程：把 serve 的 stdout 逐行喂进队列，结束时压入 None。"""
        try:
            for raw in proc.stdout:
                self._rx.put(raw.rstrip("\n"))
        except Exception as exc:  # pragma: no cover - 进程被杀时正常发生
            logger.debug("serve 读取线程结束: %s", exc)
        finally:
            self._rx.put(None)

    def _start_proc(self) -> bool:
        if not self._cfg.exe_dir:
            self.error_occurred.emit(
                "未配置 SNRobotLab 目录：请复制 local_config.example.toml 为 "
                "local_config.toml，并填写 [paths] snrobotlab_dir。详见 路径硬编码清单.md"
            )
            return False
        exe = Path(self._cfg.exe_path)
        if not exe.is_file():
            self.error_occurred.emit(
                f"找不到 snrobot.exe：{exe}\n"
                f"请确认 local_config.toml 的 [paths] snrobotlab_dir 正确，"
                f"且该目录含 snrobot.exe、RobotSDK.dll、许可证。"
            )
            return False
        try:
            self._proc = subprocess.Popen(
                [str(exe), "serve"], cwd=str(exe.parent),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as exc:
            self.error_occurred.emit(
                f"启动 snrobot.exe 失败: {exc}\n"
                f"当前路径: {exe}（exe_dir={self._cfg.exe_dir}）"
            )
            return False

        # 读取线程随进程一起起；握手也走队列 —— 原来直接 for raw in stdout 读握手，
        # exe 起来了但不吐 SERVE_READY 时会卡死在连接这一步。
        self._rx = queue.Queue()
        self._rx_thread = threading.Thread(
            target=self._rx_pump, args=(self._proc,),
            name="scara-serve-rx", daemon=True,
        )
        self._rx_thread.start()

        deadline = time.monotonic() + self.CONNECT_TIMEOUT_S
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.error("snrobot serve 在 %.0fs 内未返回 SERVE_READY", self.CONNECT_TIMEOUT_S)
                self.error_occurred.emit(
                    f"snrobot serve 启动超时（{self.CONNECT_TIMEOUT_S:.0f}s 未就绪）")
                return False
            try:
                item = self._rx.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                if self._proc.poll() is not None:
                    self.error_occurred.emit("snrobot serve 进程启动后立即退出")
                    return False
                continue
            if item is None:
                self.error_occurred.emit("snrobot serve 输出已关闭")
                return False
            s = item.strip()
            if s == "SERVE_READY":
                return True
            if s == "CONNECT_FAIL":
                self.error_occurred.emit("连接控制器失败(CONNECT_FAIL)")
                return False

    def _stop_proc(self) -> None:
        p = self._proc
        self._proc = None
        if p is None:
            return
        try:
            if p.poll() is None:
                p.stdin.write("quit\n"); p.stdin.flush()   # 触发退出前 stopall
                p.wait(timeout=3)
        except Exception:
            try:
                p.stdin.close()
            except Exception:
                pass
        try:
            if p.poll() is None:
                p.terminate()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  状态解析（可直接测试）
    # ------------------------------------------------------------------ #
    def read_all_sync(self) -> Optional[dict]:
        out = self._send("readall")
        if "JOINT" not in out:
            return None
        st = {"joints": [0.0] * 4, "pose": [0.0] * 6, "enable": 0, "mode": "?",
              "warn": 0, "control": 0, "speed": 0.0, "mechlock": 0,
              "di_on": 0, "di_list": [], "do_on": 0, "do_list": []}
        for line in out.splitlines():
            t = line.split()
            if not t:
                continue
            try:
                k = t[0]
                if k == "JOINT": st["joints"] = [float(x) for x in t[1:5]]
                elif k == "POSE": st["pose"] = [float(x) for x in t[1:7]]
                elif k == "ENABLE":
                    pass  # 忽略 snrobot ENABLE，改用 scara_enable Read_PowerOn
                elif k == "MODE": st["mode"] = t[1]
                elif k == "WARN":
                    pass  # 忽略 snrobot WARN，改用 Read_Warning
                elif k == "CONTROL":
                    st["control"] = int(t[1])  # 仅保留原始值；急停不以它为准
                elif k == "SPEED": st["speed"] = float(t[1])
                elif k == "MECHLOCK": st["mechlock"] = int(t[1])
                elif k == "DI_ON":
                    st["di_on"] = int(t[1])
                    st["di_list"] = [int(x) for x in t[2].split(",") if x] if len(t) > 2 else []
                elif k == "DO_ON":
                    st["do_on"] = int(t[1])
                    st["do_list"] = [int(x) for x in t[2].split(",") if x] if len(t) > 2 else []
            except (ValueError, IndexError):
                continue
        return self._merge_safety(st)

    def _merge_safety(self, st: dict) -> dict:
        """用 scara_enable overlay 覆盖使能/急停/报警（及可信 mode）。"""
        saf = self._safety or {}
        st["enable"] = int(saf.get("enable", 0))
        hw_estop = bool(saf.get("estop", False))
        st["warn"] = int(saf.get("warn", 0))
        mode = saf.get("mode") or "?"
        if mode and mode != "?":
            st["mode"] = mode
        st["soft_estop"] = bool(self._soft_estop)
        st["estop"] = bool(self._soft_estop) or hw_estop
        st["effectively_enabled"] = (
            int(st["enable"]) == 1
            and not st["estop"]
            and int(st["warn"]) == 0
        )
        st["need_clear"] = bool(st["estop"] or int(st["warn"]) != 0)
        return st

    def _apply_safety_from_text(self, text: str) -> dict:
        parsed = en_client.parse_status(text or "")
        self._safety.update(parsed)
        return parsed

    def _run_enable_exe(self, *args: str) -> Tuple[bool, str, dict]:
        """临时断开 snrobot serve → 调 scara_enable → 再拉起 serve。

        返回 (ok, 输出文本, 解析出的 safety dict)。
        """
        with self._enable_exe_lock:
            was_connected = self._connected
            self._stop_polling()
            self._stop_proc()
            ip = self._cfg.controller_ip
            port = int(self._cfg.controller_port)
            ok, out = en_client.run_cmd(
                *args, ip=ip, port=port, sdk_dir=self._cfg.exe_dir,
            )
            safety = self._apply_safety_from_text(out)
            if was_connected:
                if not self._start_proc():
                    self._connected = False
                    if self._last_status:
                        self._last_status = self._merge_safety(dict(self._last_status))
                        self.status_updated.emit(self._last_status)
                    self.error_occurred.emit("scara_enable 后重新连接 snrobot 失败")
                    self.connection_changed.emit(False)
                    return ok, out, safety
                st = self.read_all_sync()
                if st is not None:
                    self._last_status = st
                    self.status_updated.emit(st)
                else:
                    if self._last_status:
                        self._last_status = self._merge_safety(dict(self._last_status))
                        self.status_updated.emit(self._last_status)
                self._start_polling()
            return ok, out, safety

    def _refresh_safety_status(self) -> Tuple[bool, str]:
        """用 scara_enable status 校正使能/急停/报警灯。"""
        ok, out, _ = self._run_enable_exe("status")
        return ok, out

    @staticmethod
    def _ok(out: str) -> bool:
        # 空输出必须判失败：_send() 在未连接/进程已死时返回 ""，而空串不含任何
        # 失败关键词 → 原来 _ok("") 为 True，掉线时 enable/move1/stopall 等全部被
        # 当成"执行成功"，流程带着错误前提继续往下走且无人察觉。
        if not out.strip():
            return False
        return not any(b in out for b in ("CONNECT_FAIL", "TIMEOUT", "未找到", "ABORT", "CMD_ERR", "=False"))

    # ------------------------------------------------------------------ #
    #  Qt 层：连接 / 轮询 / 异步命令
    # ------------------------------------------------------------------ #
    @property
    def config(self) -> ScaraConfig:
        return self._cfg

    def is_connected(self) -> bool:
        return self._connected

    def preset_names(self) -> List[str]:
        return list(self._presets.keys())

    def connect(self) -> bool:
        if self._connected:
            return True
        if not self._start_proc():
            self._connected = False
            self.connection_changed.emit(False)
            return False
        st = self.read_all_sync()
        if st is None:
            self._stop_proc()
            self._connected = False
            self.error_occurred.emit("连接后读取状态失败")
            self.connection_changed.emit(False)
            return False
        self._connected = True
        self._last_status = st
        # 用 scara_enable 校正使能/急停/报警（短暂断 serve）
        ok_s, msg_s = self._refresh_safety_status()
        if not ok_s:
            self.warning_occurred.emit(f"安全状态读取失败: {(msg_s or '')[:120]}")
        st = self._last_status or st
        self.info_occurred.emit(f"已连接控制器 {self._cfg.controller_ip}:{self._cfg.controller_port}")
        self.connection_changed.emit(True)
        self.status_updated.emit(st)
        self.presets_changed.emit(self.preset_names())
        self._start_polling()
        return True

    def connect_async(self) -> None:
        """把阻塞的 connect()（启动 exe + 读状态握手）放线程池执行，避免冻结 GUI。
        结果经 connection_changed / error_occurred 信号回到 UI 线程。同步 connect()
        仍保留（供脱离 Qt 的直测）。"""
        if self._connected or self._connecting:
            return
        self._connecting = True
        self._pool.start(_ConnectCmd(self))

    def disconnect(self) -> None:
        self._stop_polling()
        self._stop_proc()
        self._connected = False
        self._soft_estop = False
        self._safety = {"enable": 0, "estop": False, "warn": 0, "mode": "?"}
        self.info_occurred.emit("已断开")
        self.connection_changed.emit(False)

    def _start_polling(self) -> None:
        if self._poller is None:
            self._poller = _PollThread(self, self._cfg.poll_interval_ms)
            self._poller.status_ready.connect(self._on_poll)
            self._poller.start()

    def _stop_polling(self) -> None:
        if self._poller is not None:
            self._poller.stop(); self._poller.wait(2000); self._poller = None

    def _on_poll(self, st: dict) -> None:
        self._last_status = st
        self.status_updated.emit(st)

    def _submit(self, name: str, line: str, need_motion: bool = False) -> None:
        if need_motion and not self._motion_guard():
            return
        self._pool.start(_Command(self, name, line))

    # —— 供 UI 的异步命令 —— #
    # 使能/去使能/急停/清报警走 scara_enable.exe（与 tools/uitest/test_enable 一致）。
    # 调用时临时断开 snrobot serve，避免抢 TCP。

    def cmd_enable(self):
        """使能：scara_enable enable（Send_PowerOn 脉冲）。"""
        def _fn():
            st0 = self._last_status or {}
            if st0.get("need_clear") or st0.get("estop") or int(st0.get("warn", 0)) != 0:
                self.error_occurred.emit("有急停或报警，请先点「清报警」再使能")
                return False, "blocked"
            ok, out, safety = self._run_enable_exe("enable")
            if not ok:
                self.error_occurred.emit(f"使能失败: {(out or '').strip()[:160] or '无输出'}")
                return False, out
            if int(safety.get("enable", 0)) != 1 or safety.get("estop") or int(safety.get("warn", 0)) != 0:
                reason = "急停未解除" if safety.get("estop") else (
                    "仍有报警" if int(safety.get("warn", 0)) else "ENABLE 仍为 OFF")
                self.error_occurred.emit(f"使能未生效（{reason}）")
                return False, out
            return True, out
        self._pool.start(_Command(self, "使能", None, fn=_fn))

    def cmd_disable(self):
        """去使能：scara_enable disable（SetMode 来回切）。"""
        def _fn():
            st0 = self._last_status or {}
            if st0 and int(st0.get("enable", 0)) != 1:
                self.info_occurred.emit("当前已是未使能，无需去使能")
                return True, "already_off"
            ok, out, safety = self._run_enable_exe("disable")
            if int(safety.get("enable", 0)) != 1:
                return True, out
            if not ok:
                self.error_occurred.emit(f"去使能失败: {(out or '').strip()[:160] or '无输出'}")
                return False, out
            self.error_occurred.emit("去使能未生效（ENABLE 仍为 ON）")
            return False, out
        self._pool.start(_Command(self, "去使能", None, fn=_fn))

    def cmd_clear_alarm(self):
        """清报警：scara_enable clear_alarm。"""
        def _fn():
            ok, out, safety = self._run_enable_exe("clear_alarm")
            cleared = (not safety.get("estop")) and int(safety.get("warn", 0)) == 0
            if cleared:
                self._soft_estop = False
                # 重新 merge 一次以清 soft_estop
                if self._last_status:
                    self._last_status = self._merge_safety(dict(self._last_status))
                    self.status_updated.emit(self._last_status)
                return True, out
            self.error_occurred.emit(
                f"清报警未完全成功: 急停={'ON' if safety.get('estop') else 'OFF'} "
                f"报警={'ON' if int(safety.get('warn', 0)) else 'OFF'} "
                f"{(out or '')[:80]}")
            return False, out
        self._pool.start(_Command(self, "清报警", None, fn=_fn))

    def cmd_set_mode(self, m):   self._submit(f"模式 {m}", f"setmode {MODE_CODE.get(m,1)}")

    def cmd_set_speed(self, pct):
        v = self._cfg.clamp_speed(pct)  # 钳到 [min,max]，防非UI/未来调用越界下发（clamp_speed 原为死代码）
        self._submit(f"速度 {v}%", f"setspeed {v}")

    def set_do_sync(self, channel: int, level: int) -> Tuple[bool, str]:
        """Synchronously write one UO/DO for an audited action sequence.

        ``scara_do.exe`` needs its own controller connection.  When the normal
        ``snrobot serve`` connection is active, pause polling and stop that
        process first, perform the write, then restore the connection before
        returning.  This makes a task's following wait/motion occur only after
        the DO result is known.
        """
        ch = int(channel)
        val = int(level)
        if not 1 <= ch <= 16:
            return False, f"DO通道必须为1到16，当前为{ch}"
        if val not in {0, 1}:
            return False, f"DO电平必须为0或1，当前为{val}"

        with self._enable_exe_lock:
            was_connected = self._connected
            if was_connected:
                self._stop_polling()
                self._stop_proc()
            try:
                write_ok, write_message = do_set_uo_level(
                    ch,
                    val,
                    ip=self._cfg.controller_ip,
                    port=int(self._cfg.controller_port),
                    sdk_dir=self._cfg.exe_dir,
                )
            except Exception as exc:  # noqa: BLE001 - hardware boundary
                write_ok, write_message = False, str(exc)

            reconnect_ok = True
            reconnect_message = ""
            if was_connected:
                if not self._start_proc():
                    reconnect_ok = False
                    reconnect_message = "写DO后重新连接snrobot失败"
                    self._connected = False
                    if self._last_status:
                        self._last_status = self._merge_safety(
                            dict(self._last_status)
                        )
                        self.status_updated.emit(self._last_status)
                    self.connection_changed.emit(False)
                else:
                    state = self.read_all_sync()
                    if state is None:
                        reconnect_ok = False
                        reconnect_message = "写DO后读取机械臂状态失败"
                    else:
                        self._last_status = state
                        self.status_updated.emit(state)
                    self._start_polling()

            name = f"DO[{ch}]={val}"
            messages = [str(write_message or name)]
            if reconnect_message:
                messages.append(reconnect_message)
            detail = "；".join(message for message in messages if message)
            ok = bool(write_ok and reconnect_ok)
            if ok:
                self.info_occurred.emit(detail)
            else:
                self.error_occurred.emit(detail or f"{name}失败")
            return ok, detail

    def cmd_set_do(self, i, level):
        """异步写单个 DO：走 scara_do.exe。level 为 0 或 1。

        会另开控制器连接，与 snrobot serve 互斥；已连接时仅警告，仍尝试写入。
        """
        ch, val = int(i), 1 if int(level) else 0
        name = f"DO[{ch}]={val}"
        if self._connected:
            self.warning_occurred.emit(
                "机械臂已连接：写 DO 可能因抢连接失败。测 DO 请先断开机械臂。"
            )

        def _fn():
            ok, msg = do_set_uo_level(
                ch, val,
                ip=self._cfg.controller_ip,
                port=int(self._cfg.controller_port),
                sdk_dir=self._cfg.exe_dir,
            )
            if not ok:
                self.error_occurred.emit(msg or "scara_do 失败")
            else:
                self.info_occurred.emit(msg or name)
            return ok, msg

        self._pool.start(_Command(self, name, None, fn=_fn))

    def zero_do_channels_sync(self, channels: List[int]) -> Tuple[bool, str]:
        """同步把若干 DO 写成 0（安全清零）。

        若 snrobot 已连接：临时断 serve → 写 0 → 再拉起（与 scara_enable 共用互斥锁）。
        调用方：UI 启动/连接前/断开后/退出；急停线程在 estop 之后也会调用。
        """
        chs = sorted({int(c) for c in channels if 1 <= int(c) <= 16})
        if not chs:
            return True, "no channels"
        with self._enable_exe_lock:
            was_connected = self._connected
            if was_connected:
                self._stop_polling()
                self._stop_proc()
            try:
                ok, out = do_zero_channels(
                    chs,
                    ip=self._cfg.controller_ip,
                    port=int(self._cfg.controller_port),
                    sdk_dir=self._cfg.exe_dir,
                )
            except Exception as exc:  # noqa: BLE001
                ok, out = False, str(exc)
            if was_connected:
                if not self._start_proc():
                    self._connected = False
                    if self._last_status:
                        self._last_status = self._merge_safety(dict(self._last_status))
                        self.status_updated.emit(self._last_status)
                    self.error_occurred.emit("清零 DO 后重新连接 snrobot 失败")
                    self.connection_changed.emit(False)
                    return ok, out
                st = self.read_all_sync()
                if st is not None:
                    self._last_status = st
                    self.status_updated.emit(st)
                elif self._last_status:
                    self._last_status = self._merge_safety(dict(self._last_status))
                    self.status_updated.emit(self._last_status)
                self._start_polling()
            return ok, out

    def cmd_home(self):          self._submit("回零", "home", need_motion=True)

    def cmd_move_joint(self, joint: int, delta: float):
        self._submit(f"J{joint} {delta:+g}", f"move1 {joint} {delta:g} {self._cfg.move_hold_s}", need_motion=True)

    def cmd_cart_step(self, axis: str, delta: float):
        self._submit(f"{axis} {delta:+g}mm", f"cartstep {CART_AXIS[axis.upper()]} {delta:g}", need_motion=True)

    # —— 连续点动（press-hold + 续命）—— #
    def jog_start(self, axis, direction: str, world: bool = False):
        """axis: 1-4 或 'X'/'Y'/'Z'；direction: '+'/'-'。按住期间定时续命保持连续运动。"""
        if not self._motion_guard():
            return
        code = CART_AXIS[axis.upper()] if isinstance(axis, str) else JOINT_AXIS[int(axis)]
        d = DIR_CODE.get(direction, 1)
        w = world or isinstance(axis, str)
        self._jog_active = (code, d, w)
        self._submit(f"连续点动 {axis}{direction}", f"jogstart {code} {d}{' world' if w else ''}")
        self._jog_timer.start()

    def _jog_heartbeat(self):
        """定时重发 jog(轻量 jogkeep)刷新控制器看门狗，保持连续运动。"""
        if self._jog_active is None:
            self._jog_timer.stop(); return
        code, d, _ = self._jog_active
        self._pool.start(_QuietCmd(self, f"jogkeep {code} {d}"))

    def jog_stop(self, axis):
        self._jog_timer.stop()
        self._jog_active = None
        code = CART_AXIS[axis.upper()] if isinstance(axis, str) else JOINT_AXIS[int(axis)]
        self._submit("jog停", f"jogstop {code}")

    def emergency_stop(self, do_channels: Optional[List[int]] = None):
        """急停：scara_enable estop（Send_Stop）。先软锁存亮灯，再断 serve 发命令。
        若传入 do_channels，急停后把这些 DO 同步写成 0（关泵/阀）。"""
        self._soft_estop = True
        self.warning_occurred.emit("急停触发")
        snap = dict(self._last_status) if self._last_status else {
            "joints": [0.0] * 4, "pose": [0.0] * 6, "enable": 0, "mode": "?",
            "warn": 0, "control": 0, "speed": 0.0, "mechlock": 0, "di_on": 0, "do_on": 0,
            "do_list": [],
        }
        snap["soft_estop"] = True
        snap["estop"] = True
        snap["enable"] = 0
        snap["effectively_enabled"] = False
        snap["need_clear"] = True
        self._last_status = snap
        self.status_updated.emit(snap)
        channels = list(do_channels or [])

        def _fn():
            ok, out, safety = self._run_enable_exe("estop")
            if safety.get("estop") or int(safety.get("warn", 0)) != 0:
                ok = True
            if not ok:
                self.error_occurred.emit(
                    f"急停失败: {(out or '').strip()[:160] or '无输出'}")
            if channels:
                zok, zmsg = self.zero_do_channels_sync(channels)
                if not zok:
                    self.error_occurred.emit(
                        f"急停后清零 DO 失败: {(zmsg or '').strip()[:160]}")
                    ok = False
                else:
                    self.info_occurred.emit("急停后已清零全部 DO")
            return ok, out or ""
        self._pool.start(_Command(self, "急停", None, fn=_fn))

    # —— 预设点 —— #
    def save_preset(self, name: str) -> None:
        st = self._last_status or self.read_all_sync()
        if not st:
            self.warning_occurred.emit("无法保存预设：未读到位置")
            return
        self._presets[name] = list(st["joints"])
        self._save_presets()
        self.info_occurred.emit(f"已保存预设点「{name}」: {['%.2f'%x for x in st['joints']]}")
        self.presets_changed.emit(self.preset_names())

    def delete_preset(self, name: str) -> None:
        if name in self._presets:
            del self._presets[name]; self._save_presets()
            self.presets_changed.emit(self.preset_names())

    def goto_joints_sync(
        self,
        name: str,
        target: Sequence[float],
        *,
        should_stop: Optional[Callable[[], bool]] = None,
        tolerance: float = 0.2,
    ) -> bool:
        """阻塞移动到四关节目标，并在每个轴和最终位置做状态回读验证。"""
        if not self._motion_guard():
            return False
        try:
            values = [float(value) for value in target]
        except (TypeError, ValueError):
            self.warning_occurred.emit(f"前往「{name}」中止：关节目标不是数字")
            return False
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            self.warning_occurred.emit(f"前往「{name}」中止：需要 4 个有限关节目标")
            return False
        if not math.isfinite(float(tolerance)) or tolerance <= 0:
            self.warning_occurred.emit(f"前往「{name}」中止：到位容差无效")
            return False
        if not self._motion_sequence_lock.acquire(blocking=False):
            self.warning_occurred.emit("已有运动任务正在执行，忽略重复运动命令")
            return False

        motion_started = False
        try:
            for joint_index in range(4):
                axis_ok = False
                reason = "timeout"
                for attempt in range(1, 4):
                    failure = self._motion_preflight(name, should_stop)
                    if failure:
                        if motion_started:
                            self._abort_motion(failure)
                        else:
                            self.warning_occurred.emit(failure)
                        return False

                    status = self.read_all_sync()
                    if status is None:
                        reason = f"前往「{name}」中止：读取 J{joint_index + 1} 当前位置失败"
                        if motion_started:
                            self._abort_motion(reason)
                        else:
                            self.warning_occurred.emit(reason)
                        return False

                    delta = values[joint_index] - float(status["joints"][joint_index])
                    if abs(delta) > tolerance:
                        hold_s = max(1, int(self._cfg.move_hold_s))
                        motion_started = True
                        self.info_occurred.emit(
                            f"前往「{name}」：J{joint_index + 1} 第 {attempt}/3 次，"
                            f"剩余 {delta:+.3f}"
                        )
                        out = self._send(
                            f"move1 {joint_index + 1} {delta:g} {hold_s}"
                        )
                        if not self._ok(out):
                            self._abort_motion(
                                f"前往「{name}」失败：J{joint_index + 1} 命令未成功"
                            )
                            return False

                    settled, reason = self._wait_joint_target_sync(
                        joint_index,
                        values[joint_index],
                        should_stop=should_stop,
                        tolerance=tolerance,
                        timeout_s=max(
                            2.0,
                            min(8.0, float(self._cfg.command_timeout_s)),
                        ),
                    )
                    if settled:
                        axis_ok = True
                        break
                    if reason != "timeout":
                        self._abort_motion(
                            f"前往「{name}」中止：J{joint_index + 1} {reason}"
                        )
                        return False
                    if attempt < 3:
                        self.warning_occurred.emit(
                            f"前往「{name}」：J{joint_index + 1} 未到位，"
                            "重新读取残差后重试"
                        )

                if not axis_ok:
                    self._abort_motion(
                        f"前往「{name}」失败：J{joint_index + 1} 三次尝试后仍未到位"
                    )
                    return False

            final = self.read_all_sync()
            if final is None:
                self._abort_motion(f"前往「{name}」失败：无法验证最终位置")
                return False
            errors = [
                abs(float(final["joints"][index]) - values[index])
                for index in range(4)
            ]
            if any(error > tolerance for error in errors):
                detail = ", ".join(
                    f"J{index + 1}={error:.3f}"
                    for index, error in enumerate(errors)
                )
                self._abort_motion(f"前往「{name}」最终未到位：{detail}")
                return False

            self.status_updated.emit(final)
            return True
        finally:
            self._motion_sequence_lock.release()

    def move_xyzr_sync(
        self,
        name: str,
        *,
        x_mm: float = 0.0,
        y_mm: float = 0.0,
        z_mm: float = 0.0,
        r_deg: float = 0.0,
        should_stop: Optional[Callable[[], bool]] = None,
        tolerance_mm: float = 0.2,
        tolerance_deg: float = 0.2,
    ) -> bool:
        """阻塞执行相对 XYZ/R 动作，并以状态回读验证每个分量到位。

        ``x_mm/y_mm/z_mm`` 是世界坐标相对位移；``r_deg`` 按本机 SCARA
        约定映射到 J4 相对旋转。组合动作的安全顺序为：正 Z（先升高）→
        X/Y/R → 负 Z（最后下降）。复杂路径仍建议在动作文件中拆成独立步骤。
        """
        if not self._motion_guard():
            return False
        try:
            deltas = {
                "X": float(x_mm),
                "Y": float(y_mm),
                "Z": float(z_mm),
                "R": float(r_deg),
            }
            tolerance_mm = float(tolerance_mm)
            tolerance_deg = float(tolerance_deg)
        except (TypeError, ValueError):
            self.warning_occurred.emit(f"执行「{name}」中止：XYZ/R 参数不是数字")
            return False
        if not all(math.isfinite(value) for value in deltas.values()):
            self.warning_occurred.emit(f"执行「{name}」中止：XYZ/R 参数不是有限数")
            return False
        if (
            not math.isfinite(tolerance_mm)
            or not math.isfinite(tolerance_deg)
            or tolerance_mm <= 0
            or tolerance_deg <= 0
        ):
            self.warning_occurred.emit(f"执行「{name}」中止：到位容差无效")
            return False
        if not any(abs(value) > 1e-12 for value in deltas.values()):
            self.warning_occurred.emit(f"执行「{name}」中止：XYZ/R 增量全部为 0")
            return False
        if not self._motion_sequence_lock.acquire(blocking=False):
            self.warning_occurred.emit("已有运动任务正在执行，忽略重复运动命令")
            return False

        motion_started = False
        try:
            initial = self.read_all_sync()
            if initial is None:
                self.warning_occurred.emit(f"执行「{name}」中止：无法读取起始位置")
                return False
            pose = initial.get("pose")
            joints = initial.get("joints")
            if not isinstance(pose, (list, tuple)) or len(pose) != 6:
                self.warning_occurred.emit(f"执行「{name}」中止：起始位姿无效")
                return False
            if not isinstance(joints, (list, tuple)) or len(joints) != 4:
                self.warning_occurred.emit(f"执行「{name}」中止：起始关节值无效")
                return False

            targets = {
                "X": float(pose[0]) + deltas["X"],
                "Y": float(pose[1]) + deltas["Y"],
                "Z": float(pose[2]) + deltas["Z"],
                "R": float(joints[3]) + deltas["R"],
            }
            order: list[str] = []
            if deltas["Z"] > 1e-12:
                order.append("Z")
            order.extend(axis for axis in ("X", "Y", "R") if abs(deltas[axis]) > 1e-12)
            if deltas["Z"] < -1e-12:
                order.append("Z")

            for axis in order:
                axis_ok = False
                reason = "timeout"
                tolerance = tolerance_deg if axis == "R" else tolerance_mm
                for attempt in range(1, 4):
                    failure = self._motion_preflight(name, should_stop)
                    if failure:
                        if motion_started:
                            self._abort_motion(failure)
                        else:
                            self.warning_occurred.emit(failure)
                        return False

                    status = self.read_all_sync()
                    if status is None:
                        reason = f"执行「{name}」中止：读取 {axis} 当前位置失败"
                        if motion_started:
                            self._abort_motion(reason)
                        else:
                            self.warning_occurred.emit(reason)
                        return False
                    current = (
                        float(status["joints"][3])
                        if axis == "R"
                        else float(status["pose"][("X", "Y", "Z").index(axis)])
                    )
                    remaining = targets[axis] - current
                    if abs(remaining) > tolerance:
                        motion_started = True
                        self.info_occurred.emit(
                            f"执行「{name}」：{axis} 第 {attempt}/3 次，剩余 {remaining:+.3f}"
                        )
                        if axis == "R":
                            hold_s = max(1, int(self._cfg.move_hold_s))
                            out = self._send(f"move1 4 {remaining:g} {hold_s}")
                        else:
                            out = self._send(f"cartstep {CART_AXIS[axis]} {remaining:g}")
                        if not self._ok(out):
                            self._abort_motion(f"执行「{name}」失败：{axis} 命令未成功")
                            return False

                    settled, reason = self._wait_xyzr_target_sync(
                        axis,
                        targets[axis],
                        should_stop=should_stop,
                        tolerance=tolerance,
                        timeout_s=max(2.0, min(8.0, float(self._cfg.command_timeout_s))),
                    )
                    if settled:
                        axis_ok = True
                        break
                    if reason != "timeout":
                        self._abort_motion(f"执行「{name}」中止：{axis} {reason}")
                        return False
                    if attempt < 3:
                        self.warning_occurred.emit(
                            f"执行「{name}」：{axis} 未到位，重新读取残差后重试"
                        )

                if not axis_ok:
                    self._abort_motion(f"执行「{name}」失败：{axis} 三次尝试后仍未到位")
                    return False

            final = self.read_all_sync()
            if final is None:
                self._abort_motion(f"执行「{name}」失败：无法验证最终位置")
                return False
            errors = {
                "X": abs(float(final["pose"][0]) - targets["X"]),
                "Y": abs(float(final["pose"][1]) - targets["Y"]),
                "Z": abs(float(final["pose"][2]) - targets["Z"]),
                "R": abs(float(final["joints"][3]) - targets["R"]),
            }
            failed = [
                axis
                for axis in order
                if errors[axis] > (tolerance_deg if axis == "R" else tolerance_mm)
            ]
            if failed:
                detail = ", ".join(f"{axis}={errors[axis]:.3f}" for axis in failed)
                self._abort_motion(f"执行「{name}」最终未到位：{detail}")
                return False
            self.status_updated.emit(final)
            return True
        finally:
            self._motion_sequence_lock.release()

    def _wait_xyzr_target_sync(
        self,
        axis: str,
        target: float,
        *,
        should_stop: Optional[Callable[[], bool]],
        tolerance: float,
        timeout_s: float,
    ) -> Tuple[bool, str]:
        """等待 XYZ 位姿分量或 J4/R 连续两次进入到位容差。"""
        deadline = time.monotonic() + max(0.5, float(timeout_s))
        stable_samples = 0
        read_failures = 0
        poll_s = max(0.05, min(0.2, self._cfg.poll_interval_ms / 1000.0))
        while time.monotonic() < deadline:
            if should_stop is not None and should_stop():
                return False, "已取消"
            if not self._connected or self._proc is None or self._proc.poll() is not None:
                return False, "控制桥已断开"
            status = self.read_all_sync()
            if status is None:
                read_failures += 1
                if read_failures >= 3:
                    return False, "连续三次状态读取失败"
                time.sleep(poll_s)
                continue
            read_failures = 0
            if (
                status.get("need_clear")
                or status.get("estop")
                or int(status.get("warn", 0)) != 0
            ):
                return False, "出现急停或报警"
            if self._cfg.require_enable_before_motion and not bool(
                status.get("effectively_enabled")
            ):
                return False, "使能已丢失"
            current = (
                float(status["joints"][3])
                if axis == "R"
                else float(status["pose"][("X", "Y", "Z").index(axis)])
            )
            if abs(current - float(target)) <= tolerance:
                stable_samples += 1
                if stable_samples >= 2:
                    return True, ""
            else:
                stable_samples = 0
            time.sleep(poll_s)
        return False, "timeout"

    def _motion_preflight(
        self,
        name: str,
        should_stop: Optional[Callable[[], bool]],
    ) -> str:
        if should_stop is not None and should_stop():
            return f"前往「{name}」已取消"
        if not self._connected or self._proc is None or self._proc.poll() is not None:
            return f"前往「{name}」中止：控制桥已断开"
        status = self.read_all_sync()
        if status is None:
            return f"前往「{name}」中止：状态读取失败"
        if (
            status.get("need_clear")
            or status.get("estop")
            or int(status.get("warn", 0)) != 0
        ):
            return f"前往「{name}」中止：存在急停或报警"
        if self._cfg.require_enable_before_motion and not bool(
            status.get("effectively_enabled")
        ):
            return f"前往「{name}」中止：使能已丢失"
        return ""

    def _wait_joint_target_sync(
        self,
        joint_index: int,
        target: float,
        *,
        should_stop: Optional[Callable[[], bool]],
        tolerance: float,
        timeout_s: float,
    ) -> Tuple[bool, str]:
        """等待单轴连续两次采样均进入到位容差。"""
        deadline = time.monotonic() + max(0.5, float(timeout_s))
        stable_samples = 0
        read_failures = 0
        poll_s = max(0.05, min(0.2, self._cfg.poll_interval_ms / 1000.0))

        while time.monotonic() < deadline:
            if should_stop is not None and should_stop():
                return False, "已取消"
            if not self._connected or self._proc is None or self._proc.poll() is not None:
                return False, "控制桥已断开"

            status = self.read_all_sync()
            if status is None:
                read_failures += 1
                if read_failures >= 3:
                    return False, "连续三次状态读取失败"
                time.sleep(poll_s)
                continue

            read_failures = 0
            if (
                status.get("need_clear")
                or status.get("estop")
                or int(status.get("warn", 0)) != 0
            ):
                return False, "出现急停或报警"
            if self._cfg.require_enable_before_motion and not bool(
                status.get("effectively_enabled")
            ):
                return False, "使能已丢失"

            error = abs(float(status["joints"][joint_index]) - float(target))
            if error <= tolerance:
                stable_samples += 1
                if stable_samples >= 2:
                    return True, ""
            else:
                stable_samples = 0
            time.sleep(poll_s)

        return False, "timeout"

    def _abort_motion(self, reason: str) -> None:
        self.warning_occurred.emit(reason)
        try:
            if self._proc is not None and self._proc.poll() is None:
                self._send("stopall", timeout=min(5.0, self.SEND_TIMEOUT_S))
        except Exception as exc:  # pragma: no cover - 连接可能已经断开
            logger.warning("运动失败后的 stopall 下发失败: %s", exc)

    def cmd_goto_preset(self, name: str) -> bool:
        if name not in self._presets:
            self.warning_occurred.emit(f"预设点不存在：{name}")
            return False
        return self.cmd_goto_joints(name, self._presets[name])

    def cmd_goto_joints(self, name: str, target: Sequence[float]) -> bool:
        try:
            values = [float(value) for value in target]
        except (TypeError, ValueError):
            self.warning_occurred.emit(f"前往「{name}」中止：关节目标不是数字")
            return False
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            self.warning_occurred.emit(f"前往「{name}」中止：需要 4 个有限关节目标")
            return False
        if not self._motion_guard():
            return False

        def _go() -> Tuple[bool, str]:
            return self.goto_joints_sync(name, values), name

        self._pool.start(_Command(self, f"前往「{name}」", None, fn=_go))
        return True

    def _motion_guard(self) -> bool:
        if self._motion_sequence_lock.locked():
            self.warning_occurred.emit("已有轨迹或多轴运动正在执行"); return False
        if not self._connected:
            self.warning_occurred.emit("未连接，无法运动"); return False
        if (
            self._last_status.get("need_clear")
            or self._last_status.get("estop")
            or int(self._last_status.get("warn", 0)) != 0
        ):
            self.warning_occurred.emit("有急停/报警，请先清报警再使能"); return False
        if self._cfg.require_enable_before_motion and not bool(self._last_status.get("effectively_enabled")):
            self.warning_occurred.emit("未使能，请先点击“使能”"); return False
        return True

    def motion_ready(self) -> bool:
        """供轨迹 UI 执行前检查连接、使能、报警和运动互斥状态。"""
        return self._motion_guard()

    # —— 预设点持久化 —— #
    def _load_presets(self) -> None:
        try:
            p = Path(PRESET_FILE)
            if p.exists():
                self._presets = json.loads(p.read_text("utf-8"))
        except Exception:
            self._presets = {}

    def _save_presets(self) -> None:
        try:
            Path(PRESET_FILE).write_text(json.dumps(self._presets, ensure_ascii=False, indent=2), "utf-8")
        except Exception as exc:
            logger.warning("保存预设失败: %s", exc)

    def cleanup(self) -> None:
        self._stop_polling()
        self._stop_proc()
        self._connected = False
        self._soft_estop = False
        self._safety = {"enable": 0, "estop": False, "warn": 0, "mode": "?"}


class _Command(QRunnable):
    """线程池执行一条 serve 命令或自定义函数，完成后发信号并刷新状态。"""

    def __init__(self, ctrl: ScaraController, name: str, line: Optional[str], fn=None):
        super().__init__()
        self._ctrl = ctrl; self._name = name; self._line = line; self._fn = fn

    def run(self) -> None:
        err = ""
        try:
            if self._fn is not None:
                ok, _ = self._fn()
            else:
                out = self._ctrl._send(self._line)
                ok = ScaraController._ok(out)
                if not ok:
                    err = (out or "").strip()[:160] or "无输出"
                    self._ctrl.error_occurred.emit(f"{self._name}失败: {err}")
        except Exception as exc:  # pragma: no cover
            ok = False
            err = str(exc)
            self._ctrl.error_occurred.emit(f"{self._name}异常: {exc}")
        self._ctrl.command_finished.emit(self._name, bool(ok), err)
        st = self._ctrl.read_all_sync()
        if st is not None:
            self._ctrl._last_status = st
            self._ctrl.status_updated.emit(st)


class _ConnectCmd(QRunnable):
    """后台线程执行阻塞的 connect() 握手，避免冻结 GUI 事件循环。"""

    def __init__(self, ctrl: ScaraController):
        super().__init__()
        self._ctrl = ctrl

    def run(self) -> None:
        try:
            self._ctrl.connect()
        except Exception as exc:  # pragma: no cover - 兜底：异常也要复位标志并告知 UI
            self._ctrl.error_occurred.emit(f"连接异常: {exc}")
            self._ctrl.connection_changed.emit(False)
        finally:
            self._ctrl._connecting = False


class _QuietCmd(QRunnable):
    """静默执行一条命令(jog 续命)，不发信号、不刷新，避免日志刷屏。"""

    def __init__(self, ctrl: ScaraController, line: str):
        super().__init__()
        self._ctrl = ctrl; self._line = line

    def run(self) -> None:
        try:
            self._ctrl._send(self._line)
        except Exception:
            pass


class _PollThread(QThread):
    status_ready = pyqtSignal(dict)

    def __init__(self, ctrl: ScaraController, interval_ms: int):
        super().__init__()
        self._ctrl = ctrl
        self._interval_ms = max(200, int(interval_ms))
        self._running = True

    def run(self) -> None:
        while self._running:
            st = self._ctrl.read_all_sync()
            if st is not None:
                self.status_ready.emit(st)
            slept = 0
            while self._running and slept < self._interval_ms:
                self.msleep(50); slept += 50

    def stop(self) -> None:
        self._running = False
