"""序列回放引擎：后台线程逐点关节运动，支持 暂停/继续/停止，发进度信号。

复用 GCR618 的 tasks/sequences.py（PICK_PLACE_FULL 等命名点序列）。
运动经后端 goto_waypoint（已带安全检查）。每点之间检查暂停/停止标志。
"""
from __future__ import annotations

import threading
from typing import List

from PyQt6.QtCore import QObject, pyqtSignal


class SequenceEngine(QObject):
    # 信号
    progress = pyqtSignal(int, int, str)   # (当前步, 总步, 点位名)
    finished = pyqtSignal(bool, str)       # (是否正常完成, 消息)
    state_changed = pyqtSignal(str)        # "running"/"paused"/"stopped"/"idle"

    def __init__(self, backend, parent=None):
        super().__init__(parent)
        self._backend = backend
        self._thread: threading.Thread = None
        self._pause = threading.Event()
        self._stop = threading.Event()
        self._running = False
        self._speed_scale = 0.2

    def is_running(self) -> bool:
        return self._running

    def set_speed_scale(self, scale: float):
        self._speed_scale = max(0.01, min(1.0, scale))

    def start(self, names: List[str]):
        if self._running:
            return
        self._stop.clear()
        self._pause.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run, args=(list(names),), daemon=True)
        self._thread.start()
        self.state_changed.emit("running")

    def pause(self):
        if self._running:
            self._pause.set()
            self.state_changed.emit("paused")

    def resume(self):
        if self._running:
            self._pause.clear()
            self.state_changed.emit("running")

    def stop(self):
        self._stop.set()
        self._pause.clear()

    def _run(self, names: List[str]):
        total = len(names)
        ok = True
        msg = "序列完成"
        try:
            for i, name in enumerate(names):
                if self._stop.is_set():
                    ok = False; msg = "已停止"; break
                # 暂停等待
                while self._pause.is_set() and not self._stop.is_set():
                    threading.Event().wait(0.1)
                if self._stop.is_set():
                    ok = False; msg = "已停止"; break
                self.progress.emit(i + 1, total, name)
                self._backend.goto_waypoint(name, speed_scale=self._speed_scale, block=True)
        except Exception as e:
            ok = False
            msg = "序列异常: %s" % e
        finally:
            self._running = False
            self.state_changed.emit("idle")
            self.finished.emit(ok, msg)
