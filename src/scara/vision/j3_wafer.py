"""SCARA J3 下移相机 · 单硅片圆检测（纯函数，无硬件依赖）。

用途：右盘取片前，相机在格上方成像高度实拍一格，检出硅片的圆心像素与半径像素，
供手眼换算成世界坐标、再做边缘吸取（见 tools/scara_j3_pick.py）。

检测分主备两条路径：
  · 主路径：灰度 → medianBlur → HoughCircles（半径在 [r_min_px, r_max_px] 内）。
    多个候选时，给了期望圆心（expect_cx/expect_cy）选距离最近的，否则选半径最大的。
  · 回退：HSV 阈值（饱和度 ≥ fallback_sat_min **或** 亮度 ≥ fallback_val_min —— 托盘是
    深色阳极氧化铝，硅片要么高饱和（紫/蓝膜）要么亮反光）→ 最大连通域 →
    minEnclosingCircle。外接圆半径出界同样判失败（连通域是别的东西）。

失败 reason 区分三种：frame_empty（帧为空）/ no_circle（两条路径都没找到）/
radius_out_of_range（回退路径找到连通域但外接圆半径出界）。

cv2/numpy 懒 import：本模块可在无 cv2 环境 import（单测只读 DEFAULT_CFG 时不拉 cv2），
首次真检测才加载（仿 wafer_detect 的懒加载铁律，cv2 设单线程保确定性）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# 默认检测参数。工具从 scara_j3_detect.json 读出 dict 后按键覆盖本默认再传入。
DEFAULT_CFG = {
    "blur_ksize": 9,            # medianBlur 核（自动取奇）
    "hough_dp": 1.2,            # Hough 累加器分辨率倒数
    "hough_min_dist_px": 200,   # 候选圆心最小间距
    "hough_param1": 80,         # Canny 高阈值
    "hough_param2": 30,         # 累加器阈值（越小越容易出假圆）
    "r_min_px": 40,             # 半径合法范围（px，成像高度下）
    "r_max_px": 400,
    "expect_cx": None,          # 期望圆心像素（可空；给了就按距离最近选圆）
    "expect_cy": None,
    "fallback_sat_min": 60,     # 回退：饱和度下限
    "fallback_val_min": 140,    # 回退：亮度下限
}

_CV2 = None


def _cv2():
    """懒加载 cv2（首次设单线程保确定性，与 wafer_detect 同一传统）。"""
    global _CV2
    if _CV2 is None:
        import cv2  # noqa: WPS433
        cv2.setNumThreads(1)
        _CV2 = cv2
    return _CV2


@dataclass
class WaferCircle:
    """一次圆检测结果。ok=False 时 cx/cy/r_px 无意义，原因在 reason。"""
    ok: bool
    cx: float = 0.0
    cy: float = 0.0
    r_px: float = 0.0
    reason: str = ""
    candidates: int = 0         # 主路径候选圆数 / 回退路径连通域数（调参用）


def _f(cfg: dict, key: str) -> float:
    """取数值型参数并强转；非法值直接 ValueError（坏配置要响，不许静默用默认）。"""
    v = cfg.get(key, DEFAULT_CFG[key])
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ValueError(f"检测参数 {key} 非法：{v!r}（应为数值）")


def _opt_f(cfg: dict, key: str) -> Optional[float]:
    """可空数值参数（expect_cx/expect_cy）：None/缺失/非法 → None。"""
    v = cfg.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def detect_wafer_circle(frame_bgr, cfg: Optional[dict] = None) -> WaferCircle:
    """在一帧 BGR 图里检出一个硅片圆。失败原因见模块头三种 reason。"""
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return WaferCircle(False, reason="frame_empty")
    cv2 = _cv2()
    c = dict(DEFAULT_CFG)
    if isinstance(cfg, dict):
        c.update({k: v for k, v in cfg.items() if k in c})     # 未知键直接忽略
    if getattr(frame_bgr, "ndim", 3) == 2:                     # 灰度帧兜底转 BGR
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)

    r_min, r_max = _f(c, "r_min_px"), _f(c, "r_max_px")

    # ── 主路径：Hough 圆 ────────────────────────────────────────────────
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    k = int(_f(c, "blur_ksize"))
    k = max(1, k + (k % 2 == 0))                               # medianBlur 核必须为奇数
    blur = cv2.medianBlur(gray, k)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT,
                               dp=_f(c, "hough_dp"),
                               minDist=_f(c, "hough_min_dist_px"),
                               param1=_f(c, "hough_param1"),
                               param2=_f(c, "hough_param2"),
                               minRadius=int(r_min), maxRadius=int(r_max))
    if circles is not None and len(circles[0]) > 0:
        cands = [(float(x), float(y), float(r)) for x, y, r in circles[0]]
        ex, ey = _opt_f(c, "expect_cx"), _opt_f(c, "expect_cy")
        if ex is not None and ey is not None:
            best = min(cands, key=lambda t: (t[0] - ex) ** 2 + (t[1] - ey) ** 2)
        else:
            best = max(cands, key=lambda t: t[2])
        return WaferCircle(True, best[0], best[1], best[2], "", len(cands))

    # ── 回退：HSV 阈值 → 最大连通域 → 最小外接圆 ────────────────────────
    import numpy as np                                         # 懒 import（只有走到回退才要）
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    sat = cv2.inRange(hsv, (0, int(_f(c, "fallback_sat_min")), 0), (179, 255, 255))
    val = cv2.inRange(hsv, (0, 0, int(_f(c, "fallback_val_min"))), (179, 255, 255))
    mask = cv2.bitwise_or(sat, val)                            # 高饱和「或」亮，见模块头
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)        # 去散点
    n, labels, stats, _cents = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return WaferCircle(False, reason="no_circle", candidates=0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    i = 1 + int(np.argmax(areas))                              # 跳过背景 0 取最大连通域
    ys, xs = np.where(labels == i)
    contour = np.stack([xs, ys], axis=1).astype(np.float32).reshape(-1, 1, 2)
    (cx, cy), r = cv2.minEnclosingCircle(contour)
    if not (r_min <= r <= r_max):
        return WaferCircle(False, float(cx), float(cy), float(r),
                           "radius_out_of_range", n - 1)
    return WaferCircle(True, float(cx), float(cy), float(r), "", n - 1)


def annotate(frame_bgr, circle: WaferCircle):
    """调试标注：检出画圆+圆心十字+半径文字；未检出在角上写原因。返回新图。"""
    cv2 = _cv2()
    out = frame_bgr.copy()
    if circle.ok:
        c = (int(round(circle.cx)), int(round(circle.cy)))
        cv2.circle(out, c, int(round(circle.r_px)), (0, 255, 0), 2)
        cv2.line(out, (c[0] - 14, c[1]), (c[0] + 14, c[1]), (0, 0, 255), 2)
        cv2.line(out, (c[0], c[1] - 14), (c[0], c[1] + 14), (0, 0, 255), 2)
        cv2.putText(out, f"r={circle.r_px:.0f}px c=({c[0]},{c[1]})",
                    (c[0] + 16, c[1] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(out, f"no wafer: {circle.reason}", (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("skip：无 cv2/numpy，跳过合成图自检（模块可 import，DEFAULT_CFG 可用）")
        raise SystemExit(0)
    # 黑底亮圆（软边，接近实拍）→ 必须检出且圆心/半径准
    img = np.zeros((480, 640, 3), np.uint8)
    cv2.circle(img, (320, 240), 80, (255, 255, 255), -1)
    img = cv2.GaussianBlur(img, (5, 5), 0)      # 硬边合成圆会被 medianBlur 系统性吃掉半径
    d = detect_wafer_circle(img, DEFAULT_CFG)
    print(f"亮圆: ok={d.ok} c=({d.cx:.1f},{d.cy:.1f}) r={d.r_px:.1f} 候选={d.candidates}")
    assert d.ok and abs(d.cx - 320) < 3 and abs(d.cy - 240) < 3 and abs(d.r_px - 80) < 5
    # 期望圆心选最近：远处再放一个小亮圆，期望点给大圆侧
    cv2.circle(img, (80, 80), 50, (255, 255, 255), -1)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    c2 = dict(DEFAULT_CFG, expect_cx=320, expect_cy=240)
    d = detect_wafer_circle(img, c2)
    assert d.ok and abs(d.cx - 320) < 5 and abs(d.cy - 240) < 5, f"期望选圆失效: {d}"
    # 纯黑 → no_circle
    d = detect_wafer_circle(np.zeros((480, 640, 3), np.uint8), DEFAULT_CFG)
    print(f"纯黑: ok={d.ok} reason={d.reason}")
    assert not d.ok and d.reason == "no_circle"
    # 空帧 → frame_empty
    d = detect_wafer_circle(None)
    assert not d.ok and d.reason == "frame_empty"
    # 半径出界：回退路径才能踩到（Hough 已被 min/maxRadius 过滤）
    small = np.zeros((480, 640, 3), np.uint8)
    cv2.circle(small, (320, 240), 10, (200, 200, 200), -1)     # 10px < r_min 40
    d = detect_wafer_circle(small, dict(DEFAULT_CFG, hough_param2=10_000))
    print(f"小圆: ok={d.ok} reason={d.reason}")
    assert not d.ok and d.reason == "radius_out_of_range"
    print("j3_wafer 自检 OK")
