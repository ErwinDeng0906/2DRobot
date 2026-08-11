"""机械臂/SCARA/自动流程 主题 —— 已并入全局设计系统。

历史上本文件是机械臂栏目的独立主题（浅色工业风）。现全局统一后，
单一事实来源迁移到 ``src/design_system.py``；本模块改为**从 design_system 透传**，
保持既有导入 ``from robot_arm.ui import theme as T`` 与所有 ``T.XXX`` 引用不变，
同时让机械臂/SCARA/自动流程自动获得统一系统（含**选中标签琥珀色高亮**）。
"""
from __future__ import annotations

# 透传全部设计令牌 + APP_STYLESHEET（含琥珀选中态）。
from design_system import *  # noqa: F401,F403

# 兼容别名（旧 theme.py 曾单独定义、部分页面按此名引用）。
from design_system import BG_DARK2, BG_CARD_SOLID  # noqa: F401
