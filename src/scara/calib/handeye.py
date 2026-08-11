"""SCARA J3 下移相机 · 手眼标定（eye-in-hand，2026-07-31）。

相机安装方式（用户拍板）：固定在 **Z 滑块壳体上朝下看，不随 J4 转**。因此相机相对
**前臂系**（方位角 α = J1+J2）是一个常值刚体：

    相机光心世界坐标  C_w = W(j1,j2) + Rz(α)·L
    成像模型          world = C_w + s·Rz(α+φ)·(px − c)

    L   = 光心相对腕轴（J4 轴心）的偏置，前臂系常值向量（mm）
    s   = mm/px 比例（只在标定时的 J3 高度有效 —— 透视投影，换了高度要重标）
    φ   = 相机图像 x 轴相对前臂系的安装角（度）
    c   = 主点像素（取画面中心）
    W   = 腕部世界坐标（kinematics.fk_wrist，L1=225/L2=175 精确已知）

吸盘（用户确认基本同心，工具仍解出偏心量作验证）：

    吸盘中心世界坐标 = W(j1,j2) + Rz(β)·C_tcp，β = J1+J2+J4（吸盘随 J4 转）

标定数据与解法（全部线性最小二乘，无迭代）：
  · 触碰记录（≥2 个不同 J4）：标记点 = W_t + Rz(β_t)·C_tcp → 解 (C_tcp, 标记点世界坐标)
    只有 1 次触碰 → 令 C_tcp = 0（同心假设），标记点 = W_t
  · 观察记录（≥2 个不同臂姿）：Rz(−α_i)·(标记点 − W_i) = L + K·u_i，K = s·Rz(φ)
    → 解 (L, a, b)，s = |K|，φ = atan2(b, a)

★ 使用纪律：像素→世界的换算只在 `imaging_j3_mm` 这个 J3 高度有效（标定时所有观察记录
  必须在同一 J3 高度做，solve 会断言）；视觉精调拍照也必须在这个高度。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from scara.pipeline.kinematics import fk_wrist

# calib 目录（与本文件同目录）
_CALIB_DIR = Path(__file__).resolve().parent
HANDEYE_FILE = "scara_j3_handeye.json"

# 标定质量门：观察残差超过此值（mm）拒绝写盘（半修正=混帧，宁可重测）
MAX_RESID_MM = 1.0
# 同一批观察记录的 J3 高度离散度上限（透视比例对高度敏感）
MAX_IMAGING_J3_SPREAD_MM = 1.0


def _rz(deg: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return ((c, -s), (s, c))


def _mv(m: Tuple[Tuple[float, float], Tuple[float, float]],
        v: Tuple[float, float]) -> Tuple[float, float]:
    return (m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1])


def _lstsq(rows: List[List[float]], rhs: List[float], n: int) -> Tuple[List[float], float]:
    """最小二乘（法方程，2~4 元小系统，无 numpy 依赖）。返回 (解, RMS 残差)。"""
    ata = [[0.0] * n for _ in range(n)]
    atb = [0.0] * n
    for r, b in zip(rows, rhs):
        for i in range(n):
            atb[i] += r[i] * b
            for j in range(n):
                ata[i][j] += r[i] * r[j]
    # 高斯消元（带部分主元）
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(ata[r][col]))
        if abs(ata[piv][col]) < 1e-12:
            raise ValueError("手眼标定方程奇异：记录之间没有足够的几何差异（臂姿/转角太像）")
        ata[col], ata[piv] = ata[piv], ata[col]
        atb[col], atb[piv] = atb[piv], atb[col]
        for r in range(col + 1, n):
            f = ata[r][col] / ata[col][col]
            for j in range(col, n):
                ata[r][j] -= f * ata[col][j]
            atb[r] -= f * atb[col]
    x = [0.0] * n
    for col in range(n - 1, -1, -1):
        x[col] = (atb[col] - sum(ata[col][j] * x[j] for j in range(col + 1, n))) / ata[col][col]
    resid = math.sqrt(sum((sum(r[i] * x[i] for i in range(n)) - b) ** 2
                          for r, b in zip(rows, rhs)) / max(1, len(rows)))
    return x, resid


@dataclass
class J3Handeye:
    """J3 下移相机手眼关系。全部常值都定义在**前臂系**（α=J1+J2，不随 J4 转）。"""
    lever_mm: Tuple[float, float] = (0.0, 0.0)      # 光心相对腕轴的偏置（前臂系）
    mm_per_px: float = 0.0                          # 成像高度处的比例
    cam_yaw_deg: float = 0.0                        # 图像 x 轴相对前臂系的安装角
    image_center_px: Tuple[float, float] = (0.0, 0.0)
    imaging_j3_mm: float = 0.0                      # 标定/使用的 J3 高度（比例只在此高度有效）
    resid_mm: float = 0.0                           # 观察记录最小二乘 RMS 残差
    taught_at: str = ""

    @property
    def calibrated(self) -> bool:
        return self.mm_per_px > 0.0

    def cam_center_world(self, joints: Sequence[float]) -> Tuple[float, float]:
        """当前臂姿下相机光心的世界坐标。"""
        w = fk_wrist(joints[0], joints[1])
        l = _mv(_rz(joints[0] + joints[1]), self.lever_mm)
        return (w[0] + l[0], w[1] + l[1])

    def world_from_px(self, joints: Sequence[float],
                      px: Tuple[float, float]) -> Tuple[float, float]:
        """图像像素 → 世界 XY（当前臂姿）。未标定（s=0）抛异常，由上层 fail-closed。"""
        if not self.calibrated:
            raise ValueError("J3 手眼未标定（mm_per_px=0），拒绝像素→世界换算")
        cc = self.cam_center_world(joints)
        u = (px[0] - self.image_center_px[0], px[1] - self.image_center_px[1])
        d = _mv(_rz(joints[0] + joints[1] + self.cam_yaw_deg), u)
        return (cc[0] + self.mm_per_px * d[0], cc[1] + self.mm_per_px * d[1])

    def px_from_world(self, joints: Sequence[float],
                      world: Tuple[float, float]) -> Tuple[float, float]:
        """世界 XY → 图像像素（正向投影，调试用）。"""
        if not self.calibrated:
            raise ValueError("J3 手眼未标定")
        cc = self.cam_center_world(joints)
        d = (world[0] - cc[0], world[1] - cc[1])
        u = _mv(_rz(-(joints[0] + joints[1] + self.cam_yaw_deg)), d)
        return (self.image_center_px[0] + u[0] / self.mm_per_px,
                self.image_center_px[1] + u[1] / self.mm_per_px)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "_schema": "scara_j3_handeye/v1",
            "_note": ("J3 下移相机手眼（装在 Z 滑块壳体、不随 J4 转；常值都在前臂系 α=J1+J2）。"
                      "mm_per_px 只在 imaging_j3_mm 高度有效，换高度必须重标。"
                      "由 tools/scara_handeye_calib.py 生成。"),
            "lever_mm": [round(self.lever_mm[0], 4), round(self.lever_mm[1], 4)],
            "mm_per_px": round(self.mm_per_px, 6),
            "cam_yaw_deg": round(self.cam_yaw_deg, 4),
            "image_center_px": [round(self.image_center_px[0], 1), round(self.image_center_px[1], 1)],
            "imaging_j3_mm": round(self.imaging_j3_mm, 4),
            "resid_mm": round(self.resid_mm, 4),
            "taught_at": self.taught_at,
        }

    @classmethod
    def from_dict(cls, d) -> "J3Handeye":
        b = cls()
        if not isinstance(d, dict):
            return b
        def _f2(key):
            v = d.get(key)
            if isinstance(v, (list, tuple)) and len(v) == 2:
                try:
                    return (float(v[0]), float(v[1]))
                except (TypeError, ValueError):
                    return None
            return None
        for attr, key in (("lever_mm", "lever_mm"), ("image_center_px", "image_center_px")):
            v = _f2(key)
            if v is not None:
                setattr(b, attr, v)
        for attr, key in (("mm_per_px", "mm_per_px"), ("cam_yaw_deg", "cam_yaw_deg"),
                          ("imaging_j3_mm", "imaging_j3_mm"), ("resid_mm", "resid_mm")):
            try:
                if d.get(key) is not None:
                    setattr(b, attr, float(d[key]))
            except (TypeError, ValueError):
                pass
        if isinstance(d.get("taught_at"), str):
            b.taught_at = d["taught_at"]
        return b


def load_j3_handeye(calib_dir=None) -> J3Handeye:
    base = Path(calib_dir) if calib_dir else _CALIB_DIR
    try:
        return J3Handeye.from_dict(json.loads((base / HANDEYE_FILE).read_text("utf-8")))
    except Exception:                                   # noqa: BLE001 - 缺文件/损坏都按未标定
        return J3Handeye()


def save_j3_handeye(he: J3Handeye, calib_dir=None) -> Path:
    base = Path(calib_dir) if calib_dir else _CALIB_DIR
    p = base / HANDEYE_FILE
    p.write_text(json.dumps(he.to_dict(), ensure_ascii=False, indent=2), "utf-8")
    # 写完立刻回读校验（守 scara_teach 的三道保险传统）
    back = load_j3_handeye(base)
    if abs(back.mm_per_px - he.mm_per_px) > 1e-9:
        raise IOError(f"写盘后回读校验失败：{p}")
    return p


# ====================================================================== #
#  标定求解
# ====================================================================== #
@dataclass
class TouchRecord:
    """触碰记录：吸盘尖端接触标记点时的关节值。"""
    joints: List[float]


@dataclass
class LookRecord:
    """观察记录：相机看到标记点时的关节值 + 标记点像素。"""
    joints: List[float]
    marker_px: Tuple[float, float]


@dataclass
class HandeyeSolution:
    handeye: J3Handeye
    cup_offset_mm: Tuple[float, float]   # 吸盘中心相对 J4 轴心（随 J4 转的系）
    marker_world_mm: Tuple[float, float]
    touch_resid_mm: float


def solve(touches: List[TouchRecord], looks: List[LookRecord],
          image_center_px: Tuple[float, float]) -> HandeyeSolution:
    """由触碰 + 观察记录解手眼参数。不足/质量不达标抛 ValueError（工具层转 SystemExit）。"""
    if not touches:
        raise ValueError("至少需要 1 次触碰记录（2 次不同 J4 可顺带解吸盘偏心）")
    if len(looks) < 2:
        raise ValueError("至少需要 2 次不同臂姿的观察记录（建议 3 次）")

    # 观察记录必须在同一 J3 高度（透视比例对高度敏感）
    j3s = [lk.joints[2] for lk in looks]
    if max(j3s) - min(j3s) > MAX_IMAGING_J3_SPREAD_MM:
        raise ValueError(f"观察记录的 J3 高度不一致（极差 {max(j3s)-min(j3s):.3f}mm > "
                         f"{MAX_IMAGING_J3_SPREAD_MM}mm）——比例随高度变，必须在同一高度拍")

    # ── 1) 触碰 → 吸盘偏心 C 与标记点世界坐标 ───────────────────────────
    if len(touches) == 1:
        cup = (0.0, 0.0)
        marker = fk_wrist(touches[0].joints[0], touches[0].joints[1])
        touch_resid = 0.0
    else:
        rows: List[List[float]] = []
        rhs: List[float] = []
        for t in touches:
            beta = t.joints[0] + t.joints[1] + t.joints[3]
            r = _rz(beta)
            w = fk_wrist(t.joints[0], t.joints[1])
            # m − Rz(β)·C = W  →  [1 0 −r00 −r01; 0 1 −r10 −r11]·(mx,my,cx,cy) = W
            rows.append([1.0, 0.0, -r[0][0], -r[0][1]]); rhs.append(w[0])
            rows.append([0.0, 1.0, -r[1][0], -r[1][1]]); rhs.append(w[1])
        x, touch_resid = _lstsq(rows, rhs, 4)
        marker = (x[0], x[1])
        cup = (x[2], x[3])

    # ── 2) 观察 → 杠杆 L 与相似阵 K = s·Rz(φ) ───────────────────────────
    rows, rhs = [], []
    for lk in looks:
        alpha = lk.joints[0] + lk.joints[1]
        w = fk_wrist(lk.joints[0], lk.joints[1])
        # Rz(−α)·(marker − W) = L + K·u，u = px − c，K=[[a,−b],[b,a]]
        m_loc = _mv(_rz(-alpha), (marker[0] - w[0], marker[1] - w[1]))
        ux = lk.marker_px[0] - image_center_px[0]
        uy = lk.marker_px[1] - image_center_px[1]
        rows.append([1.0, 0.0, ux, -uy]); rhs.append(m_loc[0])
        rows.append([0.0, 1.0, uy, ux]); rhs.append(m_loc[1])
    x, look_resid = _lstsq(rows, rhs, 4)
    s = math.hypot(x[2], x[3])
    if s <= 1e-9:
        raise ValueError("解出的 mm_per_px≈0：观察记录几何退化（两次臂姿几乎相同？）")
    he = J3Handeye(
        lever_mm=(x[0], x[1]),
        mm_per_px=s,
        cam_yaw_deg=math.degrees(math.atan2(x[3], x[2])),
        image_center_px=image_center_px,
        imaging_j3_mm=sum(j3s) / len(j3s),
        resid_mm=look_resid,
        taught_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    return HandeyeSolution(handeye=he, cup_offset_mm=cup,
                           marker_world_mm=marker, touch_resid_mm=touch_resid)


if __name__ == "__main__":
    import random
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # ── 合成数据自检：造一个真值场景，加噪声，看能不能解回来 ──────────────
    random.seed(7)
    TRUE_L = (32.0, -14.0)
    TRUE_S = 0.0435
    TRUE_PHI = 18.0
    TRUE_C = (0.35, -0.22)
    C_PX = (640.0, 360.0)
    J3 = -20.0
    MARKER = (250.0, 180.0)

    def _noisy(v, sig):
        return v + random.gauss(0, sig)

    # 触碰的物理约束：吸盘尖碰住标记点 ⟹ 腕轴离标记点正好 |C| 远、方向由 β=J1+J2+J4 定。
    # 不同 J4 的触碰腕轴必然不同（差 Rz(β)·C 那一点），由 IK 反解出对应关节。
    from scara.pipeline.kinematics import ik_wrist
    touches: List[TouchRecord] = []
    for beta in (10.0, 100.0, 190.0):
        off = _mv(_rz(beta), TRUE_C)
        wt = (MARKER[0] - off[0] + _noisy(0, 0.05), MARKER[1] - off[1] + _noisy(0, 0.05))
        j1, j2 = ik_wrist(*wt)[0]
        j4 = beta - j1 - j2
        touches.append(TouchRecord(joints=[_noisy(j1, 0.02), _noisy(j2, 0.02), J3, _noisy(j4, 0.02)]))

    looks: List[LookRecord] = []
    true_he = J3Handeye(TRUE_L, TRUE_S, TRUE_PHI, C_PX, J3)
    for j in ([5.0, 55.0, J3, 0.0], [28.0, 15.0, J3, 0.0], [-15.0, 70.0, J3, 0.0]):
        px = true_he.px_from_world(j, MARKER)
        px = (_noisy(px[0], 1.5), _noisy(px[1], 1.5))          # 像素点击噪声 ±1.5px
        jj = [_noisy(j[0], 0.02), _noisy(j[1], 0.02), J3, 0.0]  # 关节回读噪声 ±0.02°
        looks.append(LookRecord(joints=jj, marker_px=px))

    sol = solve(touches, looks, C_PX)
    he = sol.handeye
    dl = math.hypot(he.lever_mm[0] - TRUE_L[0], he.lever_mm[1] - TRUE_L[1])
    dc = math.hypot(sol.cup_offset_mm[0] - TRUE_C[0], sol.cup_offset_mm[1] - TRUE_C[1])
    print(f"L     真值 {TRUE_L} 解出 ({he.lever_mm[0]:.3f},{he.lever_mm[1]:.3f})  偏差 {dl:.3f}mm")
    print(f"s     真值 {TRUE_S} 解出 {he.mm_per_px:.5f}  偏差 {abs(he.mm_per_px-TRUE_S)/TRUE_S*100:.2f}%")
    print(f"φ     真值 {TRUE_PHI}° 解出 {he.cam_yaw_deg:.3f}°  偏差 {abs(he.cam_yaw_deg-TRUE_PHI):.3f}°")
    print(f"C     真值 {TRUE_C} 解出 ({sol.cup_offset_mm[0]:.3f},{sol.cup_offset_mm[1]:.3f})  偏差 {dc:.3f}mm")
    print(f"残差  观察 {he.resid_mm:.4f}mm  触碰 {sol.touch_resid_mm:.4f}mm")
    assert dl < 0.6, "杠杆解不准"
    assert abs(he.mm_per_px - TRUE_S) / TRUE_S < 0.02, "比例解不准"
    assert abs(he.cam_yaw_deg - TRUE_PHI) < 0.6, "安装角解不准"
    assert dc < 0.35, "吸盘偏心解不准"
    # 单触碰退化路径（C=0）
    sol1 = solve(touches[:1], looks, C_PX)
    assert sol1.cup_offset_mm == (0.0, 0.0)
    # J3 高度不一致必须拒
    bad = [LookRecord(joints=[5.0, 55.0, J3 + 5.0, 0.0], marker_px=(1.0, 2.0))] + looks
    try:
        solve(touches, bad, C_PX)
        raise AssertionError("J3 混高没拦住")
    except ValueError as e:
        print(f"混高拦截 OK: {e}")
    print("handeye 自检 OK")
