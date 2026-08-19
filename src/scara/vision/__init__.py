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
- scara.vision.slot_marker_observation：把固定的36槽和槽内marker ID投影到
  当前图像，并做多尺度marker观测；旧layout不再参与运行时槽位排序。
- scara.vision.wafer_shape_quality：在透视校正后的单槽小图中判断硅片颜色、
  正方形程度、中心偏移、相对角度、重叠边线和多连通块。
- scara.vision.silicon_detection_config：严格加载并校验
  src/scara/calib/silicon_detection_0818.json，供手眼UI、Task14和离线入口共用。
- scara.vision.tray_occupancy：按完整视野、明确遮挡、硅片和marker证据给出
  empty/occupied/warning/stacked/outside_slot/stacked_outside_slot/
  out_of_view/occluded/unknown，信息不足不猜测。
- scara.vision.tray_vision_fusion：组合Tray位姿、36槽投影、槽内marker和硅片
  质量；位姿质量门失败时停止，并提供点击像素到Tray毫米坐标的只读转换。
- scara.vision.suction_target_calibration：阶段4固定工作平面吸盘target的
  多帧SE(3)聚合、稳健拟合和按位置留一验证（纯数值模块）。
- scara.vision.suction_target_calibration_runtime：Task8逐图调用阶段3、回写
  points.json并保存camera1_suction_target.json及运行更新清单。
- scara.vision.xy_image_jacobian：阶段5，稳健标定局部
  Δimage_error = J·Δrobot_world_XY，并进行条件数与按offset留一验证。
- scara.vision.handeye_interaction：阶段6，只读加载阶段3/4/5结果，投影指定
  槽中心、A–H重投影角点和Tray坐标轴；不导入任何运动后端。
- scara.vision.wide_xy_jacobian / wide_xy_jacobian_runtime：Task11的P22
  20×20mm宽域训练、独立验证与正式模型安装。
- scara.vision.stage7b_servo / stage7b_session：阶段7B两级有限闭环的纯计算、
  响应门与证据保存；控制器仍只由UI的ActionWorker持有。
- scara.vision.planar_handeye / planar_handeye_runtime：相机1固定高度平面
  前臂手眼拟合、整槽留出验证与Task13安装；明确不支持Z或完整6-DoF。
- scara.vision.runtime_tray_registration：用5张新鲜Stage3帧计算当前会话
  W←T，并在边界情形执行人员确认的三姿态复核。
- scara.vision.moved_tray_servo / moved_tray_positioning_session：不依赖旧
  preset或Task9/11的可移动托盘P22动态粗定位、毫米闭环和独立1mm hold。
- scara.vision.full_tray_positioning / full_tray_positioning_session：保留
  Stage2内部几何工具及历史固定托盘报告回放；公开全盘会话指向动态W←T流程。
"""
