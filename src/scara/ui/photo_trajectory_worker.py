"""逐个到达轨迹点并保存相机照片的后台任务。"""

from __future__ import annotations

import math
import re
import threading
import time
from pathlib import Path
from typing import Sequence

from PyQt6.QtCore import QThread, pyqtSignal

from scara.controller.scara_controller import ScaraController
from scara.ui.camera_view import ScaraCameraThread


class PhotoTrajectoryWorker(QThread):
    """移动到每个已验证目标，等待稳定，拍照后再继续。"""

    progress = pyqtSignal(str)
    photo_saved = pyqtSignal(str)
    run_finished = pyqtSignal(bool, str, str)

    def __init__(
        self,
        controller: ScaraController,
        camera: ScaraCameraThread,
        steps: Sequence[dict],
        output_dir: Path,
        *,
        wait_before_photo_s: float = 2.0,
        wait_after_photo_s: float = 2.0,
        parent=None,
    ):
        super().__init__(parent)
        self._controller = controller
        self._camera = camera
        self._steps = [
            {
                "name": str(step["name"]),
                "joints": [float(value) for value in step["joints"]],
            }
            for step in steps
        ]
        self._output_dir = Path(output_dir)
        self._wait_before_photo_s = float(wait_before_photo_s)
        self._wait_after_photo_s = float(wait_after_photo_s)
        self._stop_requested = threading.Event()

    def request_stop(self) -> None:
        self._stop_requested.set()
        self._controller.emergency_stop()

    def _interruptible_wait(self, seconds: float) -> bool:
        remaining = max(0.0, float(seconds))
        while remaining > 1e-9:
            if self._stop_requested.is_set():
                return False
            interval = min(0.1, remaining)
            time.sleep(interval)
            remaining -= interval
        return not self._stop_requested.is_set()

    @staticmethod
    def _safe_point_name(name: str) -> str:
        cleaned = re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE).strip("_.")
        return (cleaned or "point")[:80]

    def _finish(self, ok: bool, message: str) -> None:
        self.run_finished.emit(ok, message, str(self._output_dir))

    def run(self) -> None:
        try:
            if not self._steps:
                self._finish(False, "没有可执行的轨迹点")
                return
            waits = (self._wait_before_photo_s, self._wait_after_photo_s)
            if any(not math.isfinite(value) or value < 0 for value in waits):
                self._finish(False, "拍照等待时间无效")
                return

            self._output_dir.mkdir(parents=True, exist_ok=False)
            total = len(self._steps)
            for index, step in enumerate(self._steps, start=1):
                if self._stop_requested.is_set():
                    self._finish(False, "轨迹拍照已取消")
                    return

                name = step["name"]
                self.progress.emit(f"{index}/{total} 正在移动到 {name}")
                if not self._controller.goto_joints_sync(
                    name,
                    step["joints"],
                    should_stop=self._stop_requested.is_set,
                ):
                    self._finish(False, f"移动到 {name} 失败")
                    return

                self.progress.emit(f"{index}/{total} 已到达 {name}，等待稳定后拍照")
                if not self._interruptible_wait(self._wait_before_photo_s):
                    self._finish(False, "拍照前已取消")
                    return

                photo_path = (
                    self._output_dir
                    / f"{index:03d}_{self._safe_point_name(name)}.jpg"
                )
                if not self._camera.snapshot(photo_path, max_age_s=1.0):
                    self._finish(False, f"在 {name} 保存相机照片失败")
                    return
                self.photo_saved.emit(str(photo_path))

                if not self._interruptible_wait(self._wait_after_photo_s):
                    self._finish(False, "拍照后已取消")
                    return

            self._finish(True, f"轨迹拍照完成，共保存 {total} 张照片")
        except FileExistsError:
            self._finish(False, f"输出文件夹已存在：{self._output_dir}")
        except Exception as exc:  # pragma: no cover - 通过信号显示到 UI
            self._finish(False, f"轨迹拍照异常：{exc}")
