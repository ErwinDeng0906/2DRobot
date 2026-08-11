"""把检出的硅片像素坐标换算成机械臂世界坐标，算出「示教位 → 片实际中心」的补偿量。

**要解决什么**（2026-07-28 实测）：示教的取片位对不准硅片实际中心 —— (5,2) 实测偏 2.4mm。
片被偏心吸起 → 到显微镜就偏心落下 → 放不进槽。逐格重新示教能修，但**片一被挪动就失效**
（当天来回搬了十几次），且 36 格要标 1.5~2 小时。

本模块做的是：取片前看一眼片在哪，把差值补进取片位。补完片被正心吸起，
放样端就不需要逐格专属点了。

★ 为什么补在**取片端**而不是放样端：偏心是在取片时产生的，补在源头，一次补偿修好整条链；
  补在放样端等于每格都要标一个专属放样位，而且换一片、片放歪一点就又不对了。

★ 依赖 `scara_tray_vision.json` 的单应（`tools/scara_tray_vision_calib.py` 标）。
  **相机一动或托盘一挪，单应就作废**，必须重标 —— 本模块检测不到这件事，
  但 `locate_cell_wafer` 的匹配门槛会把「差得离谱」的情况挡下来，不会闷头去吸空气。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

VISION_JSON = (Path(__file__).resolve().parents[1] / "calib" / "scara_tray_vision.json")

# 匹配门槛：检出的片离该格示教中心多远还算"就是这一格的片"
MATCH_MAX_MM = 12.0
# 歧义门槛：次近的片至少要比最近的远这么多，否则说明两片都在附近、认错了会去吸邻格
AMBIGUITY_MIN_MM = 8.0
# 补偿量上限：正常偏差实测 0.8~2.4mm。超过这个数说明单应过期/认错片/托盘挪了，宁可停手
MAX_COMP_MM = 6.0


@dataclass
class CellLocate:
    ok: bool
    dx_mm: float = 0.0          # 世界 X 补偿量（加到示教取片位上）
    dy_mm: float = 0.0
    reason: str = ""
    wafer_xy_mm: Tuple[float, float] = (0.0, 0.0)
    taught_xy_mm: Tuple[float, float] = (0.0, 0.0)
    px: Tuple[float, float] = (0.0, 0.0)

    @property
    def dist_mm(self) -> float:
        return (self.dx_mm ** 2 + self.dy_mm ** 2) ** 0.5


def load_homography(path: Optional[Path] = None):
    """读单应矩阵（像素 → 世界 mm）。没标定过返回 None —— 调用方必须当"不能补偿"处理。"""
    p = path or VISION_JSON
    if not p.exists():
        return None
    d = json.loads(p.read_text("utf-8"))
    h = d.get("H_px_to_world")
    if not h or len(h) != 3:
        return None
    return h


def px_to_world(H: Sequence[Sequence[float]], x: float, y: float) -> Tuple[float, float]:
    """单应变换。纯 Python 实现，不引 numpy —— 就 9 个乘加，import numpy 反而更贵。"""
    a, b, c = H[0]
    d, e, f = H[1]
    g, h, i = H[2]
    w = g * x + h * y + i
    if abs(w) < 1e-12:
        raise ValueError("homography_degenerate")   # 点落在消失线上，无有限像
    return ((a * x + b * y + c) / w, (d * x + e * y + f) / w)


def locate_cell_wafer(scan: Any, H: Sequence[Sequence[float]],
                      taught_xy: Sequence[float],
                      match_max_mm: float = MATCH_MAX_MM,
                      ambiguity_min_mm: float = AMBIGUITY_MIN_MM,
                      max_comp_mm: float = MAX_COMP_MM) -> CellLocate:
    """在检出结果里找「属于这一格」的片，返回补到示教位上的世界补偿量。

    `scan` = `detect_tray_wafers()` 的结果；`taught_xy` = 该格示教取片位的 world_xy。

    **一律 fail-closed**：认不出、认得含糊、补偿量离谱 —— 全部返回 ok=False。
    补偿失败的正确处理是**退回不补偿的原示教位**（还是今天在跑的那套，能吸起来只是可能偏），
    绝不是"补个大概"：补错方向比不补更糟，可能直接撞到邻格的片。
    """
    if H is None:
        return CellLocate(False, reason="no_homography：还没跑 scara_tray_vision_calib.py")
    if scan is None or not getattr(scan, "ok", False):
        return CellLocate(False, reason=f"scan_failed:{getattr(scan, 'reason', '?')}")
    wafers = list(getattr(scan, "wafers", []))
    if not wafers:
        return CellLocate(False, reason="no_wafer_detected：整帧一片都没检出")

    tx, ty = float(taught_xy[0]), float(taught_xy[1])
    scored = []
    for w in wafers:
        try:
            wx, wy = px_to_world(H, w.cx, w.cy)
        except ValueError:
            continue
        scored.append((((wx - tx) ** 2 + (wy - ty) ** 2) ** 0.5, wx, wy, w))
    if not scored:
        return CellLocate(False, reason="all_wafers_degenerate")
    scored.sort(key=lambda t: t[0])
    d0, wx, wy, w0 = scored[0]
    base = CellLocate(False, wafer_xy_mm=(wx, wy), taught_xy_mm=(tx, ty), px=(w0.cx, w0.cy))

    if d0 > match_max_mm:
        base.reason = (f"no_wafer_near_cell：最近的片离示教中心 {d0:.2f}mm > {match_max_mm}mm"
                       f"（这格空着？还是单应过期了？）")
        return base
    if len(scored) > 1 and (scored[1][0] - d0) < ambiguity_min_mm:
        base.reason = (f"ambiguous：最近 {d0:.2f}mm、次近 {scored[1][0]:.2f}mm，"
                       f"差不足 {ambiguity_min_mm}mm，可能认成邻格的片")
        return base
    dx, dy = wx - tx, wy - ty
    if (dx * dx + dy * dy) ** 0.5 > max_comp_mm:
        base.dx_mm, base.dy_mm = dx, dy
        base.reason = (f"compensation_too_large：{(dx * dx + dy * dy) ** 0.5:.2f}mm > {max_comp_mm}mm"
                       f"（实测正常偏差 0.8~2.4mm；这么大八成是单应过期或托盘挪了，先重标）")
        return base
    base.ok, base.dx_mm, base.dy_mm = True, dx, dy
    return base


if __name__ == "__main__":   # 自检：纯函数，不碰硬件
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from types import SimpleNamespace as NS

    # 单位单应：像素值 == 世界值，方便手算预期
    I = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
    assert px_to_world(I, 3.0, 4.0) == (3.0, 4.0)
    # 平移+缩放
    H = [[2.0, 0, 10.0], [0, 2.0, -5.0], [0, 0, 1.0]]
    assert px_to_world(H, 1.0, 1.0) == (12.0, -3.0)

    def scan(*pts):
        return NS(ok=True, reason="", wafers=[NS(cx=x, cy=y) for x, y in pts])

    # 正常：片在 (101.5, 200.8)，示教 (100,200) → 补 (+1.5,+0.8)
    r = locate_cell_wafer(scan((101.5, 200.8)), I, (100.0, 200.0))
    assert r.ok and abs(r.dx_mm - 1.5) < 1e-9 and abs(r.dy_mm - 0.8) < 1e-9, r
    assert abs(r.dist_mm - 1.7) < 1e-9, r.dist_mm      # √(1.5²+0.8²)=√2.89=1.7 整

    # fail-closed 四条
    assert not locate_cell_wafer(scan(), I, (100.0, 200.0)).ok            # 没检出
    assert not locate_cell_wafer(None, I, (100.0, 200.0)).ok              # 扫描失败
    assert not locate_cell_wafer(scan((1.0, 1.0)), None, (100.0, 200.0)).ok   # 没单应
    assert "no_wafer_near_cell" in locate_cell_wafer(
        scan((130.0, 200.0)), I, (100.0, 200.0)).reason                   # 太远
    assert "ambiguous" in locate_cell_wafer(
        scan((101.0, 200.0), (104.0, 200.0)), I, (100.0, 200.0)).reason   # 两片挨太近
    assert "compensation_too_large" in locate_cell_wafer(
        scan((108.0, 200.0)), I, (100.0, 200.0)).reason                   # 补偿量离谱
    # 边界：恰好在歧义门槛外应当通过
    assert locate_cell_wafer(scan((101.0, 200.0), (109.1, 200.0)), I, (100.0, 200.0)).ok
    print("tray_locate 自检通过：单应换算 + 6 条 fail-closed 门槛")
