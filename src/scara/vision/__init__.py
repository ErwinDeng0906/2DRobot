"""SCARA 送检 · 视觉（硅片检测 + 视觉伺服）。

子模块直接 import（不在此聚合导出，避免 __init__ 触发 cv2 等重依赖）：
- scara.vision.wafer_detect：纯函数硅片检测。
- scara.vision.servo：视觉伺服循环。
- scara.vision.charuco_calibration：可复用的逐图质量评估、内参求解、
  重投影误差离群剔除和姿态多样性检查（不依赖Qt）。
- scara.vision.charuco_calibration_runtime：引导弹窗、动作回调、标定报告与
  camera1_intrinsics.json 保存。
- scara.vision.tray_board_geometry：阶段2，建立严格正交Tray Frame、6×6槽
  目标和非共面A–H刚体Board几何。
- scara.vision.tray_pose_estimator：阶段3，检测A–H并由RANSAC/重投影质量门
  求当前相机帧的 ^C T_T。
- scara.vision.tray_pose_tracker：对合格的 ^C T_T 做跳变拒绝和时间滤波。
"""
