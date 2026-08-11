"""机械臂标定子包。

暴露纯离线几何工具；旧 goto_pose 已永久冻结，wafer_tray 仅保留历史数据解析。

标定数据文件（本目录 *.json）：
    table_plane.json      朝下姿态 R_down_euler_zyx（ZYX 欧拉）+ 平面
    tcp_cup_position.json  吸盘 TCP 偏移 p_FT_m（法兰系, m）
"""
from __future__ import annotations

try:  # 纯 numpy/stdlib；若环境缺 numpy 也不至于让整个包 import 失败
    from .wafer_tray import (  # noqa: F401
        Frame,
        compose_place_orientation,
        load_tray_geometry,
        pocket_pose,
        solve_frame,
        load_frame,
        save_frame,
    )
    from .dual_cup import (  # noqa: F401
        DualCupCalibrationError,
        build_qualified_tool_contract,
        solve_plane_normal,
        solve_tcp_pivot,
    )
except Exception:  # pragma: no cover - 缺 numpy 时惰性失败，调用处再报
    pass

__all__ = [
    "Frame",
    "compose_place_orientation",
    "load_tray_geometry",
    "pocket_pose",
    "solve_frame",
    "load_frame",
    "save_frame",
    "DualCupCalibrationError",
    "build_qualified_tool_contract",
    "solve_plane_normal",
    "solve_tcp_pivot",
]
