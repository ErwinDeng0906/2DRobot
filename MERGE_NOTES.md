# 合并说明

## 来源

- 主干：`RobotControl0811/RobotArm_SCARA_Control`
- 摄像功能来源：`RobotArm_SCARA_Control`
- 输出：`RobotArm_SCARA_Control_merged`

两个源目录均保留不动。合并结果不复制源项目的 `.git`、运行日志或 Python 缓存。

## 合并策略

主干保留了更新的 SCARA 状态读取、安全状态、DO 泵阀控制和深色 UI。旧项目的整份
`control_widget.py` 和 `scara_controller.py` 没有直接覆盖主干，因为这样会回退上述功能。
本次只迁入与摄像工作流相关的代码和数据：

- SCARA USB 相机帧锁、新鲜度检测、线程安全快照与错误日志；
- 轨迹文件导入、结构/数值校验和源#1逐点拍照 UI；
- 后台逐点拍照任务和中止/急停联动；
- 多轴移动互斥、逐轴到位验证、超时重试和失败后 `stopall`；
- `Preset Trajectories/`、`scara_presets.json` 和旧项目的 `Trajectory Photos/`。

DUCO 的 HTTP 相机管理、浏览器 MJPEG/快照接口和桌面相机面板在两个项目中本来相同，
因此沿用主干版本，没有重复替换。旧项目的继电器守护进程、切盘几何工具和泵控制代码
不属于本次摄像主任务，也没有覆盖主干现有的 DO 方案。

## 验证

- 在 `scara310` 环境完成相关 Python 文件编译；
- 确认 PyQt6 可导入，OpenCV 版本为 5.0.0；
- 用模拟状态/运动回读执行四关节到位流程；
- 用假相机执行一条轨迹并生成照片文件；
- 成功加载自带五点轨迹；
- 使用 Qt offscreen 构建并截图 SCARA 页面，未访问机械臂或相机硬件。

真机验收仍需现场完成：确认源#1确为目标相机，低速执行短轨迹，检查每个点的到位状态、
照片清晰度与输出目录，并实测停止按钮及物理急停。
