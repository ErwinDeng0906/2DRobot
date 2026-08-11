"""按**饱和度**在右台相机画面里找硅片 —— 取片二次确认用。

判据来自操作员（2026-07-28 当面给的）：**紫色或深色的是硅片，反光的银白色是铝合金样品台**。
主控在实拍图上核过，饱和度 S 把两者分得很开、中间有大空档：

    硅片        S = 71.8 / 72.3 / 85.1      （亮度 V 从 16 到 139 都有 —— 取决于反光角度）
    铝合金台面  S = 12.6 / 13.4
    托盘白边框  S = 19.5

所以**用 S 判、不用亮度判**。这一点反直觉但很重要：硅片有时反光很亮、有时几乎全黑，
按暗块找会漏掉反光的那些；而它的饱和度始终高，铝合金始终低。

实测（`_micro_shot/right_tray_now.jpg`，最左列 6 片）：S>45 在托盘范围内 **6/6 全中、零误检**；
托盘外的干扰（黑色板件、机械臂、面包板边缘）会被检出，靠 `region` 掩膜或前后对比排除。

★ 本模块**不做格位归属**。取片确认走「同姿态前后对比」：吸片前后各拍一帧，
  少了一片就是吸走了。不需要知道每格在画面的哪里，也就不受槽位不均匀影响
  （实测托盘槽位离边有距离，均匀网格假设压不住）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple


@dataclass
class Wafer:
    """一片检出的硅片。坐标为整帧像素。"""
    cx: float
    cy: float
    area_px: float
    w: float
    h: float
    sat_mean: float


@dataclass
class TrayWaferConfig:
    sat_min: int = 45          # 高于此饱和度视作硅片（硅片 ≥63，铝合金 ≤20，取中间偏低侧）
    min_area_px: float = 1500  # 小于此面积视作噪点（实测最小的一片 3172px）
    max_area_px: float = 60000  # 大于此面积必是背景大块（实测最大的一片 8382px）
    min_aspect: float = 0.40
    """min(w,h)/max(w,h) 下限：硅片再透视也接近方的，细长的必是别的东西。

    实测：6 片硅片 0.61~0.90（0.61 那片在画面最远端、透视压缩最厉害）；
    误检的黑板边 0.29、面包板边 0.037。0.40 把两者分开且留足余量。
    """
    open_ksize: int = 7
    close_ksize: int = 11

    @classmethod
    def from_dict(cls, d) -> "TrayWaferConfig":
        c = cls()
        for k in ("sat_min", "min_area_px", "max_area_px", "min_aspect",
                  "open_ksize", "close_ksize"):
            if isinstance(d, dict) and k in d:
                setattr(c, k, type(getattr(c, k))(d[k]))
        return c


@dataclass
class TrayScan:
    wafers: List[Wafer] = field(default_factory=list)
    ok: bool = True
    reason: str = ""

    @property
    def count(self) -> int:
        return len(self.wafers)


def detect_tray_wafers(image: Any, cfg: Optional[TrayWaferConfig] = None,
                       region: Optional[Sequence[float]] = None) -> TrayScan:
    """在整帧里找硅片。`region=(x0,y0,x1,y1)` 可选，用来排除托盘以外的干扰。"""
    cfg = cfg or TrayWaferConfig()
    if image is None or getattr(image, "size", 0) == 0:
        return TrayScan(ok=False, reason="empty_image")
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    mask = (sat > cfg.sat_min).astype(np.uint8) * 255
    if region is not None:
        x0, y0, x1, y1 = (int(v) for v in region)
        keep = np.zeros(mask.shape, np.uint8)
        keep[max(0, y0):y1, max(0, x0):x1] = 255
        mask = cv2.bitwise_and(mask, keep)
    if cfg.open_ksize >= 3:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((cfg.open_ksize, cfg.open_ksize), np.uint8))
    if cfg.close_ksize >= 3:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((cfg.close_ksize, cfg.close_ksize), np.uint8))

    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Wafer] = []
    for c in cs:
        a = float(cv2.contourArea(c))
        if not (cfg.min_area_px <= a <= cfg.max_area_px):
            continue
        (cx, cy), (w, h), _ang = cv2.minAreaRect(c)
        if max(w, h) <= 0 or min(w, h) / max(w, h) < cfg.min_aspect:
            continue                      # 细长 → 不是硅片（黑板边/面包板边）
        m = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(m, [c], -1, 255, -1)
        out.append(Wafer(float(cx), float(cy), a, float(w), float(h),
                         float(sat[m > 0].mean()) if (m > 0).any() else 0.0))
    out.sort(key=lambda w: (w.cy, w.cx))
    return TrayScan(wafers=out)


def picked_one(before: TrayScan, after: TrayScan,
               move_tol_px: float = 25.0) -> Tuple[bool, str, Optional[Wafer]]:
    """同姿态前后两帧比对：**恰好少一片**才算吸起成功。

    返回 `(是否吸起, 人话原因, 消失的那一片)`。

    为什么要求"**恰好**少一片"而不是"少了至少一片"：多片同时消失说明现场发生了
    计划外的事（碰翻、遮挡变化、检测抖动），此时不该当成成功继续往显微镜走。
    多出片同样判失败 —— 那说明两帧根本不可比。

    `move_tol_px` 用来把前后两帧的同一片配上对；两帧是**同一机械臂姿态**下拍的，
    硅片不该移动，容差只需覆盖检测抖动。
    """
    if not before.ok or not after.ok:
        return False, f"scan_failed:before={before.reason},after={after.reason}", None
    if before.count == 0:
        return False, "before_frame_has_no_wafer", None
    used = set()
    vanished: List[Wafer] = []
    for b in before.wafers:
        best, bestd = None, 1e18
        for i, a in enumerate(after.wafers):
            if i in used:
                continue
            d = (a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2
            if d < bestd:
                best, bestd = i, d
        if best is not None and bestd <= move_tol_px ** 2:
            used.add(best)
        else:
            vanished.append(b)
    appeared = after.count - len(used)
    if appeared > 0:
        return False, f"unexpected_new_wafers:{appeared}", None
    if len(vanished) == 1:
        return True, "", vanished[0]
    if len(vanished) == 0:
        return False, "no_wafer_removed", None
    return False, f"multiple_wafers_vanished:{len(vanished)}", None
