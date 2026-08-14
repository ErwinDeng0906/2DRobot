"""
SCARA 工位相机取帧线程（OpenCV / DirectShow）

现有相机模块面向海康 MVS / 图漫工业相机；SCARA 工位接的是 USB 相机，
故用 OpenCV VideoCapture 取帧，复用「numpy 帧 → QImage → QLabel」显示方式。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from utils import get_logger

logger = get_logger("scara.camera")


class ScaraCameraThread(QThread):
    """后台采集 USB 相机帧并转 QImage 发出。"""

    frame_ready = pyqtSignal(QImage)
    error = pyqtSignal(str)

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, parent=None):
        super().__init__(parent)
        self._index = index
        self._w, self._h = width, height
        self._running = False
        self._last_frame = None
        self._last_frame_at = 0.0
        self._last_frame_sequence = 0
        self._frame_lock = threading.Lock()

    @property
    def source_index(self) -> int:
        return self._index

    def run(self) -> None:
        try:
            import cv2
        except Exception as exc:
            self.error.emit(f"未安装 opencv-python: {exc}")
            return
        cap = cv2.VideoCapture(self._index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.error.emit(f"无法打开相机 index={self._index}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._h)
        self._running = True
        while self._running and not self.isInterruptionRequested():
            ok, frame = cap.read()
            if not ok:
                self.msleep(30); continue
            with self._frame_lock:
                self._last_frame = frame.copy()
                self._last_frame_at = time.monotonic()
                self._last_frame_sequence += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            self.frame_ready.emit(img)
            self.msleep(33)   # ~30fps
        cap.release()
        self._running = False

    def stop(self, timeout_ms: int = 1500) -> bool:
        """Request a bounded stop and report whether the thread really exited."""

        self._running = False
        self.requestInterruption()
        return bool(self.wait(int(timeout_ms)))

    def has_fresh_frame(self, max_age_s: float = 1.0) -> bool:
        """最近 ``max_age_s`` 秒内是否成功采集过画面。"""
        with self._frame_lock:
            return (
                self._last_frame is not None
                and time.monotonic() - self._last_frame_at <= float(max_age_s)
            )

    def latest_frame(self, max_age_s: float = 1.0):
        """Return a thread-safe BGR copy of the newest frame, or ``None``.

        The hand-eye monitor shares camera 1 with the main preview instead of
        opening DirectShow twice.  A copy prevents either consumer mutating the
        acquisition buffer.
        """

        packet = self.latest_frame_packet(max_age_s=max_age_s)
        return None if packet is None else packet[0]

    def latest_frame_packet(self, max_age_s: float = 1.0):
        """Return ``(BGR copy, sequence, capture_monotonic_s)`` or ``None``.

        Sequence changes only for a genuinely new camera frame, so Stage3 can
        invalidate a stopped stream instead of repeatedly accepting one buffer.
        """

        with self._frame_lock:
            if (
                self._last_frame is None
                or time.monotonic() - self._last_frame_at > float(max_age_s)
            ):
                return None
            return (
                self._last_frame.copy(),
                int(self._last_frame_sequence),
                float(self._last_frame_at),
            )

    def snapshot(self, path: str | Path, max_age_s: float = 1.0) -> bool:
        """线程安全地保存最新且未过期的 BGR 帧。"""
        with self._frame_lock:
            if (
                self._last_frame is None
                or time.monotonic() - self._last_frame_at > float(max_age_s)
            ):
                return False
            frame = self._last_frame.copy()
        try:
            import cv2
            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            return bool(cv2.imwrite(str(output), frame))
        except Exception as exc:
            logger.warning("相机快照保存失败 %s: %s", path, exc)
            return False
