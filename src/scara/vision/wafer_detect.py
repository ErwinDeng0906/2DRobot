"""SCARA 送检 · 硅片视觉检测（纯函数，无硬件依赖）。

照 deploy/peel_alignment/server/right_tray_vision.py 的候选检测流程落到 SCARA 像素域：
暗块（可选叠高饱和度）二值化 → 形态学开闭 → findContours → minAreaRect →
面积/边长/长宽比/矩形度四道门 → 按 rectangularity 打分选最优；
yaw 用 boxPoints 边向量法归一化到 [-45, 45)（比裸用 rect[2] 稳，仿 _rect_axis_angle_deg）。

确定性：cv2 首次加载即 setNumThreads(1)，检测入口 setRNGSeed(0)（与 right_tray_vision 一致）。
cv2 懒加载：import 本模块只拉 numpy，真检测才拉 cv2（仿 vision_service 懒加载铁律）。
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np

Point = Tuple[float, float]
Cell = Tuple[int, int]

_CV2 = None


def _cv2():
    """懒加载 cv2（首次设单线程保确定性）。"""
    global _CV2
    if _CV2 is None:
        import cv2  # noqa: WPS433
        cv2.setNumThreads(1)
        _CV2 = cv2
    return _CV2


# ---------------------------------------------------------------------- #
#  数据结构
# ---------------------------------------------------------------------- #
@dataclass
class WaferDetection:
    """一次硅片检测结果。found=False 时其余字段为默认/空，reason 记失败原因。"""
    found: bool
    center_px: Optional[Point] = None      # 硅片中心像素 (u, v)
    yaw_deg: float = 0.0                    # 硅片边相对图像轴，归一化到 [-45, 45)
    bbox: Optional[tuple] = None           # minAreaRect ((cx,cy),(w,h),ang)
    area_px: float = 0.0
    confidence: float = 0.0                # = 选中候选的 rectangularity
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WaferDetectConfig:
    """检测门限（可 JSON 加载）。默认值偏宽松，实机按相机/硅片尺寸标定后收紧。"""
    dark_threshold: int = 90               # 灰度 < 此值算暗（硅片）
    use_saturation: bool = False           # 叠加高饱和度掩码（彩色硅片时开）
    saturation_threshold: int = 70
    blur_ksize: int = 5
    open_ksize: int = 5
    close_ksize: int = 9
    min_area_px: float = 400.0
    max_area_px: float = 200000.0
    min_side_px: float = 40.0
    max_side_px: float = 400.0
    max_aspect_ratio: float = 1.6
    min_rectangularity: float = 0.6
    occupied_min_dark_frac: float = 0.15   # occupied_cells：格内暗像素占比 ≥ 此值判有片

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "WaferDetectConfig":
        base = cls()
        if not isinstance(d, dict):
            return base
        for k, v in d.items():
            if hasattr(base, k) and v is not None:
                setattr(base, k, type(getattr(base, k))(v))
        return base


# ---------------------------------------------------------------------- #
#  yaw 归一化（boxPoints 边向量法，仿 right_tray_vision._rect_axis_angle_deg）
# ---------------------------------------------------------------------- #
def _rect_axis_angle_deg(box: np.ndarray) -> float:
    """minAreaRect 的 boxPoints → 边相对图像 X 轴夹角，折进 [-45, 45)。"""
    edge = box[1] - box[0]
    angle = math.degrees(math.atan2(float(edge[1]), float(edge[0])))
    return float((angle + 45.0) % 90.0 - 45.0)


# ---------------------------------------------------------------------- #
#  主检测
# ---------------------------------------------------------------------- #
def _dark_mask(cv2, image: np.ndarray, cfg: WaferDetectConfig) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if cfg.blur_ksize >= 3:
        k = int(cfg.blur_ksize) | 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    mask = cv2.inRange(gray, 0, int(cfg.dark_threshold))
    if cfg.use_saturation and image.ndim == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = cv2.inRange(hsv[:, :, 1], int(cfg.saturation_threshold), 255)
        mask = cv2.bitwise_or(mask, sat)
    return mask


def detect_wafer(image: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None,
                 cfg: Optional[WaferDetectConfig] = None) -> WaferDetection:
    """检测 ROI 内最像硅片的深色方块。roi=(x0,y0,x1,y1) 或 None（全图）。坐标为全图像素。"""
    cfg = cfg or WaferDetectConfig()
    if image is None or getattr(image, "size", 0) == 0:
        return WaferDetection(False, reason="empty_image")
    cv2 = _cv2()
    cv2.setRNGSeed(0)

    mask = _dark_mask(cv2, image, cfg)
    if roi is not None:
        m = np.zeros(mask.shape, np.uint8)
        x0, y0, x1, y1 = (int(v) for v in roi)
        m[y0:y1, x0:x1] = 255
        mask = cv2.bitwise_and(mask, m)
    if cfg.open_ksize >= 3:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((cfg.open_ksize, cfg.open_ksize), np.uint8))
    if cfg.close_ksize >= 3:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((cfg.close_ksize, cfg.close_ksize), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0
    reason = "no_dark_contour" if not contours else "all_gates_failed"
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < cfg.min_area_px or area > cfg.max_area_px:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), _ = rect
        short, long_ = (w, h) if w <= h else (h, w)
        if short < cfg.min_side_px or long_ > cfg.max_side_px:
            continue
        if long_ / max(short, 1.0) > cfg.max_aspect_ratio:
            continue
        rectangularity = area / max(short * long_, 1.0)
        if rectangularity < cfg.min_rectangularity:
            continue
        if rectangularity > best_score:
            best_score = rectangularity
            best = (rect, area, (cx, cy))

    if best is None:
        return WaferDetection(False, reason=reason)
    rect, area, (cx, cy) = best
    box = cv2.boxPoints(rect)
    yaw = _rect_axis_angle_deg(box)
    return WaferDetection(
        True, center_px=(float(cx), float(cy)), yaw_deg=yaw,
        bbox=((float(rect[0][0]), float(rect[0][1])),
              (float(rect[1][0]), float(rect[1][1])), float(rect[2])),
        area_px=area, confidence=float(best_score), reason="ok",
    )


def occupied_cells(image: np.ndarray, cell_rois: Dict[Cell, Tuple[int, int, int, int]],
                   cfg: Optional[WaferDetectConfig] = None) -> Dict[Cell, bool]:
    """按格 ROI 判断哪格有片：格内暗像素占比 ≥ occupied_min_dark_frac 即 True（借 _dark_area 思路）。"""
    cfg = cfg or WaferDetectConfig()
    out: Dict[Cell, bool] = {}
    if image is None or getattr(image, "size", 0) == 0:
        return {cell: False for cell in cell_rois}
    cv2 = _cv2()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    dark = cv2.inRange(gray, 0, int(cfg.dark_threshold))
    for cell, roi in cell_rois.items():
        x0, y0, x1, y1 = (int(v) for v in roi)
        patch = dark[y0:y1, x0:x1]
        denom = max((x1 - x0) * (y1 - y0), 1)
        frac = float(cv2.countNonZero(patch)) / denom
        out[cell] = frac >= cfg.occupied_min_dark_frac
    return out
