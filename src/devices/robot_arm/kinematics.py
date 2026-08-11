"""GCR3-618 简化正运动学（仅供 3D 可视化与 TCP 估算）。

重要说明：
  这里的 DH 参数是基于 GCR3-618 规格（6 轴协作臂、负载 3kg、臂展约 618mm）的
  **近似自洽值**，不是厂商官方标定值。用途仅限：
    (1) 3D 状态图按关节角实时联动渲染连杆姿态；
    (2) 仿真后端从关节角估算 TCP 位姿。
  真机控制（movej2）只下发关节角，不依赖这套 DH，因此近似不影响安全性。

  采用标准 DH（Denavit-Hartenberg）约定，单位：长度 m，角度 rad。
  构型参照通用协作臂（类 UR/DUCO）：肩-肘-腕，腕部 3 轴相交。

采用 UR 风格标准 DH（竖立构型），按 618mm 臂展缩放，保证零位/竖立姿态直观自然
（类手控器显示）。具体值见下方 DH 表。3D viewer 会自动按包围盒取景居中。
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

# ── 标准 DH 参数表 (a, alpha, d, theta_offset) ────────────────────────────────
# 每行对应一个关节 i：从 link i-1 到 link i 的变换。
# UR 风格标准 DH（竖立构型），按 618mm 臂展缩放。standard DH 约定。
# 这套构型保证零位时臂竖直向上展开，姿态直观（类手控器显示）。
DH = [
    # a(m)    alpha(rad)     d(m)     theta_offset(rad)
    (0.0,     math.pi / 2,   0.152,   0.0),   # J1 基座→肩
    (-0.340,  0.0,           0.0,     0.0),   # J2 大臂
    (-0.300,  0.0,           0.0,     0.0),   # J3 小臂
    (0.0,     math.pi / 2,   0.090,   0.0),   # J4 腕1
    (0.0,    -math.pi / 2,   0.082,   0.0),   # J5 腕2
    (0.0,     0.0,           0.075,   0.0),   # J6 腕3→法兰
]

# 连杆可视尺寸（用于 3D 画几何连杆的粗细，m）
LINK_RADII = [0.045, 0.040, 0.035, 0.030, 0.028, 0.025]

# 关节运动范围（rad），用于 JOG/限位提示（近似 ±2π，留余量与 robot.yaml 一致）
JOINT_LIMITS = [(-3.10, 3.10)] * 6


def _dh_matrix(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """单段标准 DH 齐次变换矩阵。"""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ], dtype=float)


def joint_transforms(joints: List[float]) -> List[np.ndarray]:
    """返回 base→每个关节坐标系的累积变换矩阵列表（长度 7：base + 6 关节）。

    用于 3D 渲染：第 i 段连杆画在 frame[i] 原点到 frame[i+1] 原点之间。
    """
    if len(joints) != 6:
        raise ValueError("需要 6 个关节角")
    T = np.eye(4)
    frames = [T.copy()]
    for i, (a, alpha, d, off) in enumerate(DH):
        T = T @ _dh_matrix(a, alpha, d, joints[i] + off)
        frames.append(T.copy())
    return frames


def forward_kinematics(joints: List[float]) -> List[float]:
    """正运动学：从 6 关节角算 TCP 位姿 [X, Y, Z, Rx, Ry, Rz]（m, rad）。

    姿态用 ZYX 欧拉角（Rz, Ry, Rx）近似，仅供显示参考。
    """
    frames = joint_transforms(joints)
    T = frames[-1]
    x, y, z = float(T[0, 3]), float(T[1, 3]), float(T[2, 3])
    # 旋转矩阵 → RPY（XYZ 固定角）
    R = T[:3, :3]
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:  # 奇异
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0.0
    return [x, y, z, rx, ry, rz]


def link_segments(joints: List[float]) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """返回每段连杆的 (起点xyz, 终点xyz, 半径)，供 3D 画连杆。"""
    frames = joint_transforms(joints)
    segs = []
    for i in range(len(frames) - 1):
        p0 = frames[i][:3, 3]
        p1 = frames[i + 1][:3, 3]
        r = LINK_RADII[i] if i < len(LINK_RADII) else 0.03
        segs.append((p0, p1, r))
    return segs
