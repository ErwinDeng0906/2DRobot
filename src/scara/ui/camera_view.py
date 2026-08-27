"""
SCARA 工位相机取帧线程（OpenCV / DirectShow）

现有相机模块面向海康 MVS / 图漫工业相机；SCARA 工位接的是 USB 相机，
故用 OpenCV VideoCapture 取帧，复用「numpy 帧 → QImage → QLabel」显示方式。
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from utils import get_logger

from scara.config.camera_config import ResolvedCameraSource, resolve_camera_source

logger = get_logger("scara.camera")

DIRECTSHOW_EXPOSURE_MIN = -13
DIRECTSHOW_EXPOSURE_MAX = -1
DIRECTSHOW_EXPOSURE_DEFAULT = -6


def _validated_exposure_value(exposure: int | float) -> int:
    """Return a supported DirectShow exposure stop or raise ``ValueError``."""

    raw = float(exposure)
    if not math.isfinite(raw) or not raw.is_integer():
        raise ValueError("hardware exposure must be an integer")
    value = int(raw)
    if not DIRECTSHOW_EXPOSURE_MIN <= value <= DIRECTSHOW_EXPOSURE_MAX:
        raise ValueError(
            "hardware exposure must be an integer from "
            f"{DIRECTSHOW_EXPOSURE_MIN} to {DIRECTSHOW_EXPOSURE_MAX}"
        )
    return value


def _auto_exposure_enabled(raw_value: float) -> bool | None:
    """Normalize common OpenCV auto-exposure representations.

    DirectShow normally reports 0.25 for manual and 0.75 for automatic mode;
    a few UVC backends report the same state as 0/1.  Other values are treated
    as unsupported instead of being written back blindly.
    """

    raw = float(raw_value)
    if not math.isfinite(raw) or raw < 0.0:
        return None
    if abs(raw - 0.75) <= 0.13 or abs(raw - 1.0) <= 0.13:
        return True
    if abs(raw - 0.25) <= 0.13 or abs(raw) <= 0.13:
        return False
    return None


class ScaraCameraThread(QThread):
    """后台采集 USB 相机帧并转 QImage 发出。"""

    frame_ready = pyqtSignal(QImage)
    error = pyqtSignal(str)
    exposure_applied = pyqtSignal(int, bool, str)

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        parent=None,
        connection_generation: int = 1,
        source_resolver: Callable[[int], ResolvedCameraSource] = resolve_camera_source,
    ):
        super().__init__(parent)
        self._index = index
        self._source_resolver = source_resolver
        self._resolved_camera: ResolvedCameraSource | None = None
        self._connection_generation = max(1, int(connection_generation))
        self._w, self._h = width, height
        self._running = False
        self._last_frame = None
        self._last_frame_at = 0.0
        self._last_frame_sequence = 0
        self._frame_lock = threading.Lock()
        self._exposure_lock = threading.Lock()
        self._requested_exposure_value = DIRECTSHOW_EXPOSURE_DEFAULT
        self._requested_exposure_action = "auto"
        self._exposure_request_sequence = 0
        self._exposure_applied_sequence = 0
        self._original_exposure_raw = None
        self._original_auto_exposure_raw = None
        self._original_auto_exposure_enabled = None

    @property
    def source_index(self) -> int:
        """Stable logical camera number used by tasks and calibrations."""

        return self._index

    @property
    def physical_source_index(self) -> int | None:
        """Current machine-local DirectShow index, once resolution succeeds."""

        return (
            None
            if self._resolved_camera is None
            else int(self._resolved_camera.physical_index)
        )

    @property
    def camera_identity(self) -> dict:
        return (
            {}
            if self._resolved_camera is None
            else self._resolved_camera.to_json()
        )

    @property
    def connection_generation(self) -> int:
        """1 for the first connection of this source, >1 after reconnect."""

        return self._connection_generation

    def run(self) -> None:
        try:
            import cv2
        except Exception as exc:
            self.error.emit(f"未安装 opencv-python: {exc}")
            return
        try:
            resolved = self._source_resolver(int(self._index))
        except Exception as exc:
            self.error.emit(f"逻辑相机{self._index}身份解析失败：{exc}")
            return
        self._resolved_camera = resolved
        cap = cv2.VideoCapture(resolved.physical_index, resolved.backend)
        if not cap.isOpened():
            self.error.emit(
                f"无法打开逻辑相机{self._index}（物理Index "
                f"{resolved.physical_index}）"
            )
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._h)
        auto_ok, auto_message = self._recover_auto_exposure(cap, cv2)
        if not auto_ok:
            self.error.emit(
                f"逻辑相机{self._index}/物理Index {resolved.physical_index} "
                f"默认自动曝光初始化失败：{auto_message}"
            )
            cap.release()
            return
        logger.info(
            "logical camera %s -> physical %s default exposure: %s",
            self._index,
            resolved.physical_index,
            auto_message,
        )
        if resolved.configured_index_stale:
            logger.warning(
                "logical camera %s moved from configured physical index %s to %s; "
                "run camera_identity_binding.py to refresh local_config.toml",
                self._index,
                resolved.configured_physical_index,
                resolved.physical_index,
            )
        self._running = True
        while self._running and not self.isInterruptionRequested():
            with self._exposure_lock:
                exposure_sequence = self._exposure_request_sequence
                exposure_value = self._requested_exposure_value
                exposure_action = self._requested_exposure_action
                exposure_pending = exposure_sequence != self._exposure_applied_sequence
            if exposure_pending:
                if exposure_action == "auto":
                    success, message = self._recover_auto_exposure(cap, cv2)
                    emitted_value = 0
                elif exposure_action == "restore":
                    success, message = self._restore_hardware_exposure(cap, cv2)
                    emitted_value = 0
                else:
                    success, message = self._apply_hardware_exposure(
                        cap,
                        cv2,
                        exposure_value,
                    )
                    emitted_value = exposure_value
                with self._exposure_lock:
                    self._exposure_applied_sequence = exposure_sequence
                self.exposure_applied.emit(emitted_value, success, message)
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
        self._restore_hardware_exposure(cap, cv2)
        cap.release()
        self._running = False

    def request_exposure_value(self, exposure: int | float) -> bool:
        """Queue one integer DirectShow exposure stop for the capture thread."""

        value = _validated_exposure_value(exposure)
        with self._exposure_lock:
            self._requested_exposure_value = value
            self._requested_exposure_action = "manual"
            self._exposure_request_sequence += 1
        return bool(self.isRunning())

    def restore_original_exposure(self) -> bool:
        """Queue restoration of the pre-slider exposure and auto mode."""

        with self._exposure_lock:
            self._requested_exposure_action = "restore"
            self._exposure_request_sequence += 1
        return bool(self.isRunning())

    def request_auto_exposure_recovery(self) -> bool:
        """Force DirectShow auto exposure on, ignoring a damaged baseline."""

        with self._exposure_lock:
            self._requested_exposure_action = "auto"
            self._exposure_request_sequence += 1
        return bool(self.isRunning())

    def _remember_original_exposure(self, cap, cv2_module) -> tuple[bool, str]:
        if self._original_exposure_raw is not None:
            return True, ""
        exposure = float(cap.get(cv2_module.CAP_PROP_EXPOSURE))
        auto_exposure = float(cap.get(cv2_module.CAP_PROP_AUTO_EXPOSURE))
        if not math.isfinite(exposure):
            return False, "相机驱动未返回有效的硬件曝光值"
        auto_enabled = _auto_exposure_enabled(auto_exposure)
        self._original_exposure_raw = exposure
        self._original_auto_exposure_raw = (
            auto_exposure if math.isfinite(auto_exposure) else None
        )
        self._original_auto_exposure_enabled = auto_enabled
        return True, ""

    @staticmethod
    def _set_auto_exposure_mode(
        cap,
        cv2_module,
        enabled: bool,
    ) -> tuple[bool, bool, float]:
        """Request a mode and report ``(accepted, verified, readback)``.

        Some UVC DirectShow drivers accept the mode change but keep returning
        a fixed 1.0 or -1.0.  Exposure stability is therefore verified again
        after frames have settled instead of treating this read-back as the
        sole source of truth.
        """

        candidates = (0.75, 1.0) if enabled else (0.25, 0.0)
        actual = float("nan")
        accepted = False
        for requested in candidates:
            if not bool(cap.set(cv2_module.CAP_PROP_AUTO_EXPOSURE, requested)):
                continue
            accepted = True
            actual = float(cap.get(cv2_module.CAP_PROP_AUTO_EXPOSURE))
            if _auto_exposure_enabled(actual) is enabled:
                return True, True, actual
            # A number of DirectShow UVC drivers accept 0.75/0.25 but expose
            # CAP_PROP_AUTO_EXPOSURE as -1 (unsupported readback).  Do not send
            # the alternate 1/0 convention after an accepted request unless
            # the readback explicitly contradicts the requested state.
            if _auto_exposure_enabled(actual) is None:
                return True, False, actual
        return accepted, False, actual

    @staticmethod
    def _exposure_matches(expected: float, actual: float) -> bool:
        if not math.isfinite(actual):
            return False
        if expected < 0.0:
            return (
                float(expected).is_integer()
                and float(actual).is_integer()
                and int(actual) == int(expected)
            )
        return abs(actual - expected) <= max(1.0, abs(expected) * 0.05)

    def _latest_frame_level(self) -> float | None:
        with self._frame_lock:
            if self._last_frame is None:
                return None
            return float(self._last_frame.mean())

    @staticmethod
    def _read_settled_frame_level(cap, frame_count: int = 4) -> float | None:
        if not hasattr(cap, "read"):
            return None
        last_frame = None
        for _ in range(max(1, int(frame_count))):
            ok, frame = cap.read()
            if ok and frame is not None:
                last_frame = frame
            time.sleep(0.03)
        return None if last_frame is None else float(last_frame.mean())

    def _apply_hardware_exposure(
        self,
        cap,
        cv2_module,
        exposure: int | float,
    ) -> tuple[bool, str]:
        target_value = _validated_exposure_value(exposure)
        remembered, reason = self._remember_original_exposure(cap, cv2_module)
        if not remembered:
            return False, reason
        target = float(target_value)
        # DirectShow uses 0.25 for manual exposure and 0.75 for auto exposure.
        # Setting this in the capture thread avoids touching VideoCapture from
        # the Qt UI thread.
        before_level = self._latest_frame_level()
        manual_accepted, manual_verified, manual_raw = self._set_auto_exposure_mode(
            cap, cv2_module, enabled=False
        )
        if not bool(cap.set(cv2_module.CAP_PROP_EXPOSURE, float(target))):
            self._restore_hardware_exposure(cap, cv2_module)
            return False, "相机驱动拒绝硬件曝光设置；画面未做软件调暗"
        applied = float(cap.get(cv2_module.CAP_PROP_EXPOSURE))
        if not self._exposure_matches(target, applied):
            self._restore_hardware_exposure(cap, cv2_module)
            return False, (
                f"相机驱动曝光读回异常（目标 {target:.3f}，实际 {applied:.3f}）；"
                "已恢复原曝光"
            )
        after_level = self._read_settled_frame_level(cap)
        expected_fraction = min(
            1.0,
            2.0 ** (target - float(self._original_exposure_raw)),
        )
        if (
            before_level is not None
            and before_level >= 5.0
            and after_level is not None
            and after_level < max(1.0, before_level * 0.20 * expected_fraction)
        ):
            restored, restore_message = self._restore_hardware_exposure(cap, cv2_module)
            return False, (
                f"曝光调整导致画面亮度异常下降（{before_level:.1f}→{after_level:.1f}），"
                f"已{'恢复原曝光' if restored else '尝试恢复但驱动未确认'}；"
                f"{restore_message}"
            )
        settled_applied = float(cap.get(cv2_module.CAP_PROP_EXPOSURE))
        if not self._exposure_matches(target, settled_applied):
            restored, restore_message = self._restore_hardware_exposure(cap, cv2_module)
            return False, (
                f"曝光在画面稳定后发生漂移（目标 {target_value}，"
                f"读回 {settled_applied:.3f}），自动曝光可能未关闭；"
                f"已{'恢复原曝光' if restored else '尝试恢复但驱动未确认'}；"
                f"{restore_message}"
            )
        mode_note = ""
        if not manual_verified:
            accepted_text = "已接受模式请求" if manual_accepted else "未确认模式请求"
            mode_note = (
                f"；自动模式读回 {manual_raw:.3f} 不可靠，{accepted_text}，"
                "已用稳定后的整数曝光读回确认"
            )
        return (
            True,
            f"硬件曝光={target_value}（driver读回 {applied:.0f}，"
            f"原值 {self._original_exposure_raw:.3f}{mode_note}）",
        )

    def _restore_hardware_exposure(self, cap, cv2_module) -> tuple[bool, str]:
        if self._original_exposure_raw is None:
            return True, "本次未保存原曝光设置，无需恢复"
        if self._original_auto_exposure_enabled:
            auto_accepted, auto_verified, actual_auto = self._set_auto_exposure_mode(
                cap, cv2_module, enabled=True
            )
            auto_state = _auto_exposure_enabled(actual_auto)
            if not auto_accepted or auto_state is False:
                return False, (
                    "相机驱动未确认恢复自动曝光"
                    f"（请求{'已接受' if auto_accepted else '被拒绝'}，"
                    f"读回 {actual_auto:.3f}）"
                )
            if auto_verified:
                return True, "已确认恢复默认自动曝光"
            return True, (
                "已恢复默认自动曝光（驱动接受请求，但不支持模式读回，"
                f"读回 {actual_auto:.3f}）"
            )

        manual_accepted, manual_verified, actual_auto = self._set_auto_exposure_mode(
            cap, cv2_module, enabled=False
        )
        exposure_ok = bool(
            cap.set(cv2_module.CAP_PROP_EXPOSURE, float(self._original_exposure_raw))
        )
        actual_exposure = float(cap.get(cv2_module.CAP_PROP_EXPOSURE))
        exposure_restored = exposure_ok and self._exposure_matches(
            self._original_exposure_raw, actual_exposure
        )
        if not exposure_restored:
            return False, (
                "相机驱动未能恢复原始手动曝光设置"
                f"（模式 {actual_auto:.3f}，曝光 {actual_exposure:.3f}）"
            )
        if self._original_auto_exposure_enabled is None:
            return True, (
                "原自动曝光模式读回不可识别；已恢复原整数曝光值"
                f"（模式请求{'已接受' if manual_accepted else '未确认'}）"
            )
        if not manual_verified:
            return False, (
                "已恢复原曝光值，但驱动未确认原手动模式"
                f"（读回 {actual_auto:.3f}）"
            )
        return True, "已确认恢复原手动曝光"

    def _recover_auto_exposure(self, cap, cv2_module) -> tuple[bool, str]:
        """Emergency recovery that deliberately ignores remembered settings."""

        auto_accepted, auto_verified, actual_auto = self._set_auto_exposure_mode(
            cap, cv2_module, enabled=True
        )
        auto_state = _auto_exposure_enabled(actual_auto)
        if not auto_accepted or auto_state is False:
            return False, (
                "相机驱动未确认自动曝光恢复"
                f"（请求{'已接受' if auto_accepted else '被拒绝'}，"
                f"读回 {actual_auto:.3f}）；"
                "请关闭程序后用相机厂商工具恢复 Auto Exposure"
            )
        self._read_settled_frame_level(cap, frame_count=12)
        exposure = float(cap.get(cv2_module.CAP_PROP_EXPOSURE))
        if not math.isfinite(exposure):
            return False, "自动曝光已开启，但驱动未返回有效曝光值"
        self._original_exposure_raw = exposure
        self._original_auto_exposure_raw = actual_auto
        self._original_auto_exposure_enabled = True
        verification_note = (
            f"模式 {actual_auto:.3f}"
            if auto_verified
            else f"模式读回不可用 {actual_auto:.3f}，驱动已接受请求"
        )
        return True, (
            f"已启用相机自动曝光（{verification_note}，曝光 {exposure:.3f}）"
        )

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
