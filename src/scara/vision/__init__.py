"""SCARA 送检 · 视觉（硅片检测 + 视觉伺服）。

子模块直接 import（不在此聚合导出，避免 __init__ 触发 cv2 等重依赖）：
- scara.vision.wafer_detect：纯函数硅片检测。
- scara.vision.servo：视觉伺服循环。
"""
