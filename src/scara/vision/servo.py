"""SCARA 送检 · 视觉伺服循环（eye-to-hand 固定俯视右盘）。

控制律（每轮）：
  1. 抓帧 → detect_wafer 得硅片中心像素；
  2. 吸盘当前像素投影 = jacobian.world_to_px(pose[:2])（随位姿变化，构成闭环反馈）；
  3. 误差 err_px = 硅片中心 − 吸盘投影；|err|<conv_px 判收敛；
  4. ΔWorld = jacobian 逆(err_px) × gain，单步夹紧 max_step_mm；
  5. step_cart 相对步进；累计位移超 max_total_mm 判误检发散。

fail-safe（照 vision_grid_correction 铁律，绝不放大误检去乱走）：取帧失败 / 检测丢失 /
雅可比奇异 / 超总位移 / 步进未到位 → 立即停手、返回结构化失败，不再发任何运动。
收敛时返回硅片 yaw，供放片阶段算 J4 角度补偿。
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional, Tuple

from scara.vision.wafer_detect import WaferDetectConfig, detect_wafer


@dataclass
class ServoConfig:
    conv_px: float = 2.0          # 像素误差 < 此值判收敛
    max_iters: int = 30
    max_step_mm: float = 5.0      # 单步 World 位移夹紧
    max_total_mm: float = 60.0    # 累计位移上限（超则疑误检发散，停手）
    gain: float = 1.0             # 步进增益（<1 更稳、>1 更快；实机调）


@dataclass
class ServoResult:
    ok: bool
    iters: int
    final_err_px: Optional[Tuple[float, float]]
    reason: str
    wafer_yaw_deg: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _clamp(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


class VisualServo:
    """把「运动后端 + 取帧 + 雅可比」组合成一次视觉伺服对准。依赖注入，可全 Fake 单测。"""

    def __init__(self, motion: Any, grabber: Any, jacobian: Any,
                 cfg: Optional[ServoConfig] = None,
                 detect_cfg: Optional[WaferDetectConfig] = None, log=None):
        self.motion = motion
        self.grabber = grabber
        self.jac = jacobian
        self.cfg = cfg or ServoConfig()
        self.detect_cfg = detect_cfg or WaferDetectConfig()
        self._log = log or (lambda _m: None)

    def servo_to_wafer(self, roi: Optional[Tuple[int, int, int, int]] = None) -> ServoResult:
        cfg = self.cfg
        total = 0.0
        last: Optional[Tuple[float, float]] = None
        for i in range(cfg.max_iters):
            try:
                frame = self.grabber.grab()
            except Exception as e:  # noqa: BLE001
                self._log(f"[servo] 取帧失败({e}) → 停手")
                return ServoResult(False, i, last, "grab_failed")

            det = detect_wafer(frame, roi, self.detect_cfg)
            if not det.found:
                self._log(f"[servo] 未检出硅片(reason={det.reason}) → 停手")
                return ServoResult(False, i, last, "detect_lost")

            nx, ny = self.jac.world_to_px(self.motion.get_pose()[:2])
            ex = float(det.center_px[0]) - float(nx)
            ey = float(det.center_px[1]) - float(ny)
            last = (ex, ey)
            if math.hypot(ex, ey) < cfg.conv_px:
                return ServoResult(True, i, last, "converged", wafer_yaw_deg=det.yaw_deg)

            dw = self.jac.world_delta_for_px_error(ex, ey)
            if dw is None:
                self._log("[servo] 雅可比奇异 → 停手")
                return ServoResult(False, i, last, "singular_jacobian")
            dx = _clamp(dw[0] * cfg.gain, cfg.max_step_mm)
            dy = _clamp(dw[1] * cfg.gain, cfg.max_step_mm)

            total += math.hypot(dx, dy)
            if total > cfg.max_total_mm:
                self._log(f"[servo] 累计位移 {total:.1f}>{cfg.max_total_mm}mm 疑误检 → 停手")
                return ServoResult(False, i, last, "over_travel")

            if not self.motion.step_cart("X", dx) or not self.motion.step_cart("Y", dy):
                self._log("[servo] 步进未到位 → 停手")
                return ServoResult(False, i, last, "step_failed")

        return ServoResult(False, cfg.max_iters, last, "max_iters")
