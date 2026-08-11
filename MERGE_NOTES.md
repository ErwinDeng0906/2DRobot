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

## 2026-08-11：0811 可移植配置合并

以本目录的运动、相机和轨迹拍照实现为基线，选择性迁入 `RobotControlCSJ0811` 的
换机配置与协作文档更新：

- 新增统一的 `local_config.toml`，集中保存 SNRobotLab、SCARA 与 DUCO 地址；
- SCARA 主桥、DO、使能工具和两个构建脚本改为使用同一目录解析逻辑；
- DUCO 后端及界面连接栏使用 `[duco]` 默认值，同时保留连接前手动覆盖；
- 保留本版本的运动互斥、逐轴到位验证、失败停止、相机新鲜度检查、轨迹导入和拍照任务；
- 保留 `release_estop()` 测试 API，避免 0811 工具脚本与测试入口不一致。

旧的本机 `scara_config.toml` 暂时保留为忽略的迁移备份；运行时以
`local_config.toml` 为准。此次选择性合并未复制或修改 `.git`。

合并后无硬件验证结果：

- 66 个 Python 文件完成 AST 语法解析；
- 本机配置、环境变量优先级、DUCO 覆盖、界面默认值和 `release_estop()` API 通过；
- `_smoke.test_merged_camera` 的四项运动/相机/轨迹/UI 测试全部通过；
- 相机、轨迹 worker、两个预设脚本和预设坐标文件的 SHA-256 保持不变；
- `.git` 的 153 个文件在内容、时间戳和属性指纹上均保持不变。
