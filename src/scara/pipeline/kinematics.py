"""SCARA 精确正/逆运动学（平面 2 连杆 + 直动 Z + 末端旋转）。

几何参数来自 2026-07-27 对 36 格示教数据的精确反解（交接 §2 注意4，回代最大残差
0.0008mm，两个危险点 0.0003/0.0002mm）：

    L1(大臂) = 225.000 mm    L2(小臂) = 175.000 mm
    基座在世界原点，无工具 XY 偏置
    Rz(末端绝对朝向) = J1 + J2 + J4 − 90
    肘部 E = (225·cosJ1, 225·sinJ1)
    腕部 W = E + (175·cos(J1+J2), 175·sin(J1+J2))     ← J4 轴心就在这里

J3 是直动 Z 轴（mm），与平面解耦；J4 只影响末端朝向、不影响腕部位置。
单位约定：角度一律度（°），长度一律 mm，与 readall 的 joints/pose 一致。

本模块纯函数、零硬件依赖，是「标定几个点 → 解算全部格子」的数学底座；
关节回放仍是运行主键（交接 §1 定案），IK 只用于**生成**回放目标与视觉补偿换算。
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

L1_MM = 225.0                      # 大臂（基座→肘）
L2_MM = 175.0                      # 小臂（肘→腕/J4 轴心）
REACH_MAX_MM = L1_MM + L2_MM       # 400.0，臂展极限（place_micro 曾顶到 398.5，余量 1.5mm）
REACH_MIN_MM = abs(L1_MM - L2_MM)  # 50.0

# IK 目标安全余量：臂展极限附近 dR/dJ2→0（奇异），径向修正极不灵敏。
# 交接 §8.1 的取舍讨论后定 396mm：再往外不是够不到，是关节噪声被放大成毫米级位置噪声。
REACH_WARN_MM = 396.0


def _norm_deg(a: float) -> float:
    """规约到 (−180, 180]。"""
    a = math.fmod(float(a), 360.0)
    if a <= -180.0:
        a += 360.0
    elif a > 180.0:
        a -= 360.0
    return a


def fk_wrist(j1_deg: float, j2_deg: float) -> Tuple[float, float]:
    """腕部（J4 轴心）世界 XY。"""
    j1, j12 = math.radians(j1_deg), math.radians(j1_deg + j2_deg)
    return (L1_MM * math.cos(j1) + L2_MM * math.cos(j12),
            L1_MM * math.sin(j1) + L2_MM * math.sin(j12))


def forearm_pose_W_F(j1_deg: float, j2_deg: float) -> List[List[float]]:
    """Return the planar homogeneous pose ``^W T_F`` of camera forearm frame F.

    ``F`` is deliberately independent of J3 and J4: its origin is the J4-axis
    centre and its x-axis follows the physical forearm, whose world yaw is
    ``alpha = J1 + J2``.  The returned 3x3 matrix is an SE(2) transform::

        [ cos(alpha) -sin(alpha)  wrist_x ]
        [ sin(alpha)  cos(alpha)  wrist_y ]
        [     0           0          1    ]

    Keeping this equation in the kinematics layer prevents camera registration
    code from accidentally substituting terminal J4 or absolute tool Rz for
    the camera-1 forearm angle.
    """

    alpha = math.radians(float(j1_deg) + float(j2_deg))
    cosine = math.cos(alpha)
    sine = math.sin(alpha)
    wrist_x, wrist_y = fk_wrist(j1_deg, j2_deg)
    return [
        [float(cosine), float(-sine), float(wrist_x)],
        [float(sine), float(cosine), float(wrist_y)],
        [0.0, 0.0, 1.0],
    ]


def rz_of(j1_deg: float, j2_deg: float, j4_deg: float) -> float:
    """末端绝对朝向（度）。"""
    return _norm_deg(j1_deg + j2_deg + j4_deg - 90.0)


def j4_for_rz(j1_deg: float, j2_deg: float, rz_deg: float) -> float:
    """给定 J1/J2 与目标末端朝向，求 J4。"""
    return _norm_deg(rz_deg - j1_deg - j2_deg + 90.0)


def ik_wrist(x: float, y: float) -> List[Tuple[float, float]]:
    """腕部目标 (x,y) 的全部 IK 分支 [(j1,j2), ...]，按 J2 降序（肘上在前）。

    无解（超出臂展/进入基座死区）返回 []。两分支在奇异位形（J2≈0/±180）处退化为同一个。
    """
    r2 = x * x + y * y
    cos_j2 = (r2 - L1_MM * L1_MM - L2_MM * L2_MM) / (2.0 * L1_MM * L2_MM)
    if cos_j2 > 1.0 or cos_j2 < -1.0:
        return []
    cos_j2 = max(-1.0, min(1.0, cos_j2))
    base = math.atan2(y, x)
    out: List[Tuple[float, float]] = []
    for s in (1.0, -1.0):
        j2 = s * math.acos(cos_j2)
        j1 = base - math.atan2(L2_MM * math.sin(j2), L1_MM + L2_MM * math.cos(j2))
        out.append((_norm_deg(math.degrees(j1)), _norm_deg(math.degrees(j2))))
    # 去重（奇异位形两分支相同）
    if len(out) == 2 and abs(out[0][0] - out[1][0]) < 1e-9 and abs(out[0][1] - out[1][1]) < 1e-9:
        out.pop()
    return out


def solve_joints(x: float, y: float, j3_mm: float,
                 rz_deg: Optional[float] = None,
                 ref_joints: Optional[List[float]] = None) -> Optional[List[float]]:
    """求「腕部到 (x,y)、J3=j3_mm」的 4 轴目标 [j1,j2,j3,j4]；无解返回 None。

    朝向与分支选择：
    - `rz_deg` 给定时 J4 = j4_for_rz；否则 J4 沿用 `ref_joints[3]`（视觉补偿只动 XY、
      保持末端朝向不变，就是这个用法）。两者都不给 → J4=0。
    - 两个 IK 分支里选「离 ref_joints 最近」的那个（逐轴角度差绝对值之和，J4 参与比较
      避免 ±180 跳变）；无 ref 时偏好 J2 ≥ 0 的分支（旧右盘 36 格全部分布在该分支上，
      交接 §2 J2 行程 8.2~111.7）。
    """
    branches = ik_wrist(x, y)
    if not branches:
        return None
    j4_ref = float(ref_joints[3]) if ref_joints is not None else 0.0

    def _cost(j1: float, j2: float, j4: float) -> float:
        if ref_joints is None:
            # 无参考：偏好肘上（J2≥0）分支；同分支内 J1 小者优先（确定性，便于复现）
            return (0.0 if j2 >= 0.0 else 1000.0) + abs(j1) * 1e-3
        return (abs(_norm_deg(j1 - ref_joints[0])) + abs(_norm_deg(j2 - ref_joints[1]))
                + abs(_norm_deg(j4 - j4_ref)))

    best: Optional[Tuple[float, float, float, float]] = None  # (cost, j1, j2, j4)
    for j1, j2 in branches:
        j4 = j4_for_rz(j1, j2, rz_deg) if rz_deg is not None else j4_ref
        c = _cost(j1, j2, j4)
        if best is None or c < best[0]:
            best = (c, j1, j2, j4)
    assert best is not None
    return [round(best[1], 4), round(best[2], 4), round(float(j3_mm), 4), round(best[3], 4)]


def reach_mm(x: float, y: float) -> float:
    """目标点到基座的距离（臂展）。"""
    return math.hypot(x, y)


def reach_ok(x: float, y: float) -> Tuple[bool, str]:
    """可达性预检：死区/超程直接拒，逼近奇异区给告警（不拒，但调用方应打出来）。"""
    r = reach_mm(x, y)
    if r > REACH_MAX_MM:
        return False, f"unreachable：臂展需求 {r:.1f}mm > 极限 {REACH_MAX_MM:.0f}mm"
    if r < REACH_MIN_MM:
        return False, f"unreachable：目标在基座死区（{r:.1f}mm < {REACH_MIN_MM:.0f}mm）"
    if r > REACH_WARN_MM:
        return True, (f"⚠ 臂展 {r:.1f}mm 逼近极限 {REACH_MAX_MM:.0f}mm（奇异区："
                      f"径向修正极不灵敏，交接 §8.1）")
    return True, ""


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 cp1252，中文/箭头会崩
    except Exception:
        pass
    # 自检：FK→IK 往返闭合；交接 §2 的 36 格统计数据区间复算
    ok = True
    for (j1, j2) in [(10.0, 30.0), (52.4, 8.2), (-28.8, -10.0), (6.9, 111.7), (0.0, 179.9)]:
        x, y = fk_wrist(j1, j2)
        j = solve_joints(x, y, -50.0, rz_deg=None, ref_joints=[j1, j2, -50.0, 7.0])
        assert j is not None, (j1, j2)
        x2, y2 = fk_wrist(j[0], j[1])
        err = math.hypot(x2 - x, y2 - y)
        assert err < 1e-6, f"FK/IK 往返不闭合: {err}"
        print(f"  j=({j1:8.3f},{j2:8.3f}) → W=({x:8.3f},{y:8.3f}) → IK=({j[0]:8.3f},{j[1]:8.3f})"
              f" 往返误差 {err:.2e}mm")
    # 无解分支
    assert ik_wrist(401.0, 0.0) == [] and ik_wrist(10.0, 0.0) == []
    # Rz 恒等式
    assert abs(rz_of(10.0, 20.0, j4_for_rz(10.0, 20.0, 33.0)) - 33.0) < 1e-9
    # 连续性：ref 选最近分支（镜像目标点上两分支 j1 差很大）
    j = solve_joints(300.0, 100.0, -50.0, ref_joints=[20.0, 60.0, -50.0, 0.0])
    assert j is not None and j[1] > 0, f"连续性选分支错: {j}"
    print("kinematics 自检 OK")
