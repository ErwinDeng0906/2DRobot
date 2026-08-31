# 点击硅片后移动到吸盘对准位：现有标定核对与后续操作规程

## 1. 核对结论

本项目并不是从零开始标定。2026-08-28 对当前分支、Git 历史、实验数据、Task8/13/15/16/17/18 和运行时加载代码逐项核对后，确认相机1定位链的大部分数据已经存在。

| 项目 | 当前状态 | 依据 |
|---|---|---|
| 相机1内参 | 已有，`success` | `src/scara/calib/camera1_intrinsics.json` |
| 托盘几何和36槽坐标 | 已有 | `src/scara/calib/tray_board_geometry.json` |
| 相机1吸盘目标 | 已有，Task8质量门全部通过 | `src/scara/calib/camera1_suction_target.json` |
| 相机1平面手眼 | 已有，Task13质量门全部通过 | `src/scara/calib/camera1_forearm_planar_handeye.json` |
| 相机1局部图像 Jacobian | 已有，`success` | `src/scara/calib/camera1_xy_image_jacobian.json` |
| 相机1大范围 Jacobian | 已有，`success` | `src/scara/calib/camera1_wide_xy_jacobian.json` |
| `W←T` | 不是固定标定文件；每次会话用5帧动态建立 | `runtime_tray_registration.py` |
| 相机2内参 | 已有，`success` | `src/scara/calib/camera2_intrinsics.json` |
| 相机2相对J4的20 mm先验 | 已有实验记录和代码 | `260812015730/points.json`、Task17 |
| 相机2到J4完整外参 | 尚未完成 | 未找到 `camera2_j4_extrinsics.json` |
| 拾取/放置高度 | 有现成数值，但不同文件间需做一次一致性确认 | Task15、Task16、托盘几何和现场记录 |

因此，当前不需要重新运行 Task8 或 Task13。只有在相机1、镜头、支架、分辨率、软吸盘或托盘几何发生物理变化时，才应重做对应标定。

## 2. 已恢复的相机1吸盘目标

`camera1_suction_target.json` 原先存在于 Git 提交 `942ef6d`，但没有进入当前主线。该文件已经恢复到正式标定目录。

核心结果如下：

```text
状态：success
相机源：1
分辨率：1280 × 720
吸盘目标像素：(625.4757, 218.4703) px
相机坐标中的吸盘工作点：(-2.4194, -33.0393, 149.5097) mm
观察高度：J3 = -27.0046 mm
工作平面：z_T = -2.0 mm
可用标定位置：6个
拟合XY RMS：0.5316 mm
留一位置交叉验证XY RMS：0.5893 mm
留一位置最大XY误差：0.9753 mm
```

全部 Task8 质量门均通过。Task13、局部 Jacobian 和大范围 Jacobian 锁定的也是这一份文件，其 SHA-256 为：

```text
6B0B4DFCB8476A2BD219471C769B7D6C95D2F965E3CB330D230AB0781C2D75D9
```

## 3. Windows/Mac 换行问题

这组标定文件是在实验室 Windows 电脑上生成的，原始文件使用 CRLF 换行。Mac 检出后如果变成 LF，JSON 数值完全相同，但按原始字节计算的 SHA-256 会变化，程序会误判为“标定输入已改变”。

项目已通过 `.gitattributes` 固定下列标定文件为 CRLF：

```text
camera1_intrinsics.json
tray_board_geometry.json
camera1_suction_target.json
camera1_forearm_planar_handeye.json
camera1_xy_image_jacobian.json
camera1_wide_xy_jacobian.json
```

恢复文件并固定换行后，以下加载器已在当前检出中实际通过：

1. `load_latest_suction_target()`；
2. `load_planar_handeye()`；
3. `load_local_xy_jacobian()`。

## 4. `W←T` 到底是什么

`W←T` 表示把托盘坐标系 `T` 中的点转换到机械臂世界坐标系 `W`。它不是一个永久固定的标定常数，因为托盘可能被重新摆放。

程序在每次会话中使用以下信息动态计算 `W←T`：

1. 相机1识别出的当前托盘位姿 `T_C_T`；
2. 当前机械臂 J1/J2/J3/J4 和世界XY位置；
3. 已有相机1吸盘目标 `p_C_S`；
4. 已有平面手眼旋转 `R_F_C`；
5. 连续5张相机1图像和与图像时间同步的机械臂状态。

单帧中的托盘原点世界位置按下式求得：

```text
托盘原点世界XY
= 当前吸盘世界XY
- 托盘旋转到世界后的“托盘原点到吸盘点”向量
```

5帧结果再进行一致性检查。只有所有帧质量、marker数量、机械臂静止程度、时间同步、原点离散度和角度离散度都通过，`W←T` 才会成为 `success`。

所以，之前界面显示 `W←T unavailable` 的原因是当前检出缺少 Task8 文件并发生换行 hash 不一致，不是 `W←T` 算法本身无法建立。恢复标定链后，仍需连接实验室相机1和机械臂，用5帧实时数据完成会话登记。

## 5. J4轴心和吸盘中心

现有 Task8 文件已经明确记录以下假设：

```text
j4_axis_and_suction_centre_concentric = true
j4_runout_test_performed = false
```

现场意见也是：当前可近似认为 J4 轴心与软吸盘中心相同，难以直接测量的偏差约为1 mm量级。

因此当前阶段采用以下策略：

1. 相机1固定观察高度、固定Rz的点击对准测试，不要求先增加一套独立的 `T_J4_S` 标定。
2. 实际吸盘目标使用 Task8 视觉标定结果，不使用肉眼估计的圆心。
3. 机械臂只在安全高度进行XY对准，先验证点击误差，不立即下降吸取。
4. 如果后续需要大幅改变J4/Rz，或要求吸盘中心误差明显小于1 mm，再做J4旋转跳动测试。
5. 如果旋转J4时吸盘中心形成可测圆轨迹，才需要把固定偏心写成 `T_J4_S`。

也就是说，`T_J4_S` 不是当前“点击后移动到硅片上方”测试的必做前置项，但它是后续高精度任意Rz操作的验证项。

## 6. 已有高度数据

当前文件中已经存在以下高度：

| 含义 | 数值 |
|---|---:|
| 相机1/浮起观察高度 | `J3 = -27.0046 mm` |
| marker平面对应J3 | `J3 = -50.0119 mm` |
| 托盘槽底参考J3 | `J3 = -52.0119 mm` |
| Task15拾取下降量 | `23.3 mm` |
| Task16当前位置拾取下降量 | `23.3 mm` |
| Task16在P00放置下降量 | `23.0 mm` |
| 现场聊天记录中的近似下降量 | `23.4 mm` |

这些值不能直接混成一个数：

1. `-52.0119 mm` 是托盘几何中的槽底参考，不一定等于软吸盘接触硅片上表面的高度。
2. Task15/16的 `23.3 mm` 是已经用于实际吸取过程的操作量。
3. `23.4 mm` 是聊天中的近似描述，与代码相差0.1 mm。
4. Task16在P00放置使用 `23.0 mm`，与拾取下降量又相差0.3 mm。

不需要重新测量全部高度，但在自动下降前必须由现场确认最终采用哪一组实测值。建议以 Task15/16 已成功运行的起始姿态为基准，核对一次：

```text
起始J3
下降量
接触时J3
是否可靠吸住
是否压碰托盘
释放后是否完整退出
```

确认前，点击功能只允许XY安全高度对准，不允许自动执行Z下降、真空或DO。

## 7. 相机2已有和缺少的数据

### 7.1 已有

1. 相机2内参，1280×720，ChArUco标定状态为 `success`。
2. Task1/Task4/Task17使用的平面先验：相机2相对J4轴心约20 mm。
3. 当 `Rz=0` 时，代码约定相机2位于J4轴心的世界 `-Y` 方向。
4. 现场记录：运行高度时，相机2约位于槽底上方66.9 mm。
5. `260812015730/points.json` 包含多个槽位、多个J3高度下的机械臂状态和名义相机位置。
6. Task18、板位姿生成器和离线外参求解器已经写好。

### 7.2 仍然缺少

项目中没有找到正式安装的：

```text
src/scara/calib/camera2_j4_extrinsics.json
```

20 mm和66.9 mm只是部分安装先验，不能代替完整 `T_J4_C2`，因为它们没有给出：

1. 相机光心相对J4的完整XYZ；
2. 相机光轴相对J4的三维旋转；
3. 支架安装误差；
4. 标定残差和不确定度。

因此，相机2可以继续用于采图和算法开发，但在完成 Task18 外参前，不能把它的像素误差直接换算成实机XY/Rz运动授权。

## 8. 当前真正需要做的步骤

### 8.1 实验室文件同步

1. 拉取包含恢复标定文件和 `.gitattributes` 的分支。
2. 确认以下文件存在：

```text
src/scara/calib/camera1_suction_target.json
src/scara/calib/camera1_forearm_planar_handeye.json
src/scara/calib/camera1_xy_image_jacobian.json
src/scara/calib/camera1_wide_xy_jacobian.json
```

3. 不修改这些JSON的格式、缩进或换行。
4. 运行程序后确认界面不再报告缺少 suction target 或锁定 hash 变化。

### 8.2 建立实时 `W←T`

1. 托盘固定，机械臂停在相机1标定观察高度 `J3≈-27.0046 mm`。
2. 相机1使用源1、1280×720，镜头和支架保持原状态。
3. 打开“手眼交互 → 转移视觉”。
4. 等待至少5张新鲜、机械臂静止且时间同步的合格帧。
5. 界面应显示 `W←T: PASS`，并给出当前托盘原点世界XY和yaw。
6. 如果需要三姿态复核，严格按界面引导完成，不得绕过质量门。

### 8.3 只读点击验证

至少测试 P00、P05、P50、P55、P22 和四个中间槽：

1. 在图像中点击槽中心或硅片中心。
2. 记录程序返回的槽名、托盘坐标和预测世界坐标。
3. 记录吸盘到目标的预测 `ΔX/ΔY`。
4. 不发送运动。
5. 与已知预设点或人工低速对准结果比较。

建议验收线：

```text
中位XY误差 <= 0.50 mm
P95 XY误差 <= 0.80 mm
最大XY误差 <= 1.20 mm
不得出现错槽
```

这组阈值与现有平面手眼质量门一致。若实际吸取需要更严格精度，应在此基础上收紧，而不是跳过验证。

### 8.4 无接触受监督运动

只在安全高度进行：

1. 操作员点击目标并锁定。
2. 程序连续多帧确认目标槽状态和 `W←T` 稳定。
3. 生成候选XY修正量。
4. ActionWorker重新读取机械臂状态，检查急停、报警、IK、工作区和单步上限。
5. 操作员显式确认后才允许单步XY移动。
6. 移动后重新拍照，验证吸盘目标与槽中心误差是否下降。
7. 全程不下降J3、不打开真空、不控制DO。

至少完成20次不同槽位无接触测试，且无错槽、无反向运动、无越界后，才讨论加入Z动作。

### 8.5 确认Z操作量

由现场使用 Task15/16 的成功起始姿态核对 `23.3/23.0/23.4 mm` 的具体含义。最终值必须写入配置，不能继续散落为多个硬编码常数。

确认前不允许点击流程调用 Task15/16。

### 8.6 完成相机2外参

1. 固定Task17使用的ChArUco板。
2. 测量板在机械臂世界坐标系中的原点、+X参考点和+Y参考点。
3. 用 `tools/create_camera2_board_pose.py` 生成板世界位姿文件。
4. 至少在12个安全姿态运行 Task18；覆盖XY、J3和Rz。
5. 用 `tools/analyze_camera2_extrinsics.py` 离线求解。
6. 只有报告为 `success` 才安装 `camera2_j4_extrinsics.json`。

相机2外参完成后，再开发近距离槽/硅片相对误差到XY/Rz的闭环修正。单目相机不负责从一张普通照片估计绝对Z；Z仍由J3和已确认高度决定。

## 9. 停止条件

出现以下任一情况时立即停止，不生成或执行运动：

1. suction target、内参、托盘几何或手眼文件hash不一致；
2. `W←T` 不是 `success`；
3. 图像和机械臂状态时间差超过0.35秒；
4. 机械臂未静止或相机发生重连；
5. marker数量或托盘位姿质量门失败；
6. 点击点无法唯一匹配槽位；
7. 预测运动方向与图像误差方向不一致；
8. 高度数值尚未完成现场一致性确认；
9. 相机2外参缺失却尝试使用其像素结果控制机械臂；
10. 急停、报警、IK、工作区或碰撞检查不通过。

## 10. 相关文件

```text
src/scara/calib/camera1_suction_target.json
src/scara/calib/camera1_forearm_planar_handeye.json
src/scara/calib/camera1_xy_image_jacobian.json
src/scara/calib/camera1_wide_xy_jacobian.json
src/scara/calib/camera2_intrinsics.json
src/scara/vision/runtime_tray_registration.py
src/scara/vision/wafer_transfer_runtime.py
src/scara/ui/wafer_transfer_dialog.py
Tasks/task15_lift wafer1.py
Tasks/task16_lift wafer2.py
Tasks/task17_camera2ARUCO.py
Tasks/task18_camera2_extrinsic_capture.py
docs/camera2_j4_extrinsic_calibration.md
docs/wafer_transfer_runtime_integration.md
```
