# 相机2到J4完整外参标定

## 1. 当前边界

本流程只完成相机2与J4坐标系之间的刚体外参：

```text
T_J4_C2：把相机2坐标转换到机械臂J4/TCP坐标
T_C2_J4：T_J4_C2的逆矩阵
```

相机2内参已经由Task17完成，项目文件为
`src/scara/calib/camera2_intrinsics.json`。外参流程不会控制机械臂、J3、J4、
真空泵或电磁阀，也不会授权自动拾取。

## 2. 为什么必须独立测量标定板世界位姿

相机2随SCARA末端运动。仅用SCARA平移和绕竖直轴旋转，同时求“标定板世界位姿”和
“相机相对J4位姿”会留下不可审计的轴向自由度。项目因此采用已知板位姿法：先独立测量
ChArUco板在机械臂世界坐标系中的位姿 `T_W_B`，再由每张照片直接计算一份
`T_J4_C2`，最后稳健汇总并检查残差。

不得把约20 mm的手工估计、图片中的像素位置或单位矩阵当作板世界位姿。

## 3. 现场准备

1. 使用Task17相同的10×8 ChArUco板：square 18.80 mm、marker 9.87 mm、
   `DICT_4X4_50`、legacy pattern。
2. 将板刚性固定。所有Task18运行结束前，板和机器人基座之间不得发生任何移动。
3. 相机2保持1280×720；镜头、对焦、支架和USB物理设备不得变化。
4. 检查完整运动范围不会碰撞，现场人员始终守在急停旁。
5. 准备同一个经确认的测量工具或同轴标定指针，不混用吸盘边缘和J4轴心。

## 4. 测量ChArUco板世界位姿

记录印刷棋盘外框的三个世界坐标，纸张白边不属于棋盘：

1. `origin`：印刷棋盘左上外角。
2. `x-reference`：沿印刷棋盘从左到右的已知点。
3. `y-reference`：沿印刷棋盘从上到下的已知点。

三个点必须在同一基准、同一工具下测量。参考点距原点至少20 mm，X/Y原始夹角偏离
90度超过2度时工具拒绝生成文件。

示例命令：

```bash
python tools/create_camera2_board_pose.py \
  --origin 120.0 250.0 -50.0 \
  --x-reference 270.4 250.0 -50.0 \
  --y-reference 120.0 370.3 -50.0 \
  --translation-uncertainty-mm 0.3 \
  --rotation-uncertainty-deg 0.3 \
  --method "同轴标定指针三点触测，操作人和日期" \
  --output camera2_board_pose_world.json
```

数值只是命令格式示例，不能直接用于实机。

## 5. 采集Task18数据

动作文件：`Tasks/task18_camera2_extrinsic_capture.py`。

每次运行前由现场人员用主界面把机器人放到一个安全静止姿态。Task18只执行：等待、
读取5次J1-J4/TCP、拍5张相机1和5张相机2照片。任务中没有任何运动或DO动作。

每个姿态完成后等待任务完全结束，再人工移动到下一姿态并重新运行。建议至少12个姿态：

1. 世界XY覆盖宽度至少40 mm，建议使用3×3区域。
2. J3覆盖至少10 mm，建议安全观察高度、低10 mm、低20 mm三个高度。
3. Rz/J4覆盖至少50度，建议约 `-60、-30、0、+30、+60` 度。
4. 每个姿态中标定板至少有12个ChArUco角点清晰可见。
5. 所有姿态中固定板不得移动；机械臂移动前必须确认上一Task18已经结束。

Task18输出仍使用现有结构：

```text
Trajectory Photos/时间戳/
  points.json
  1_001.jpg ...
  2_001.jpg ...
```

## 6. 离线求解

可以给父文件夹，也可以逐个列出时间戳目录：

```bash
python tools/analyze_camera2_extrinsics.py \
  "Trajectory Photos" \
  --board-pose camera2_board_pose_world.json \
  --output camera2_j4_extrinsic_report.json \
  --annotated-dir camera2_j4_extrinsic_annotated
```

工具会检查分辨率、内参hash、ChArUco角点、PnP内点率、重投影误差、姿态数量、XY/J3/Rz
跨度、外参平移/旋转残差和板位姿测量不确定度。

只有报告为 `status=success` 时，才允许显式安装：

```bash
python tools/analyze_camera2_extrinsics.py \
  "Trajectory Photos" \
  --board-pose camera2_board_pose_world.json \
  --output camera2_j4_extrinsic_report.json \
  --install
```

安装目标为 `src/scara/calib/camera2_j4_extrinsics.json`。不加 `--install` 时只生成离线
报告。任何质量门失败都会返回非零退出码并拒绝写项目外参。

## 7. 现场必须确认的数据

1. 板三点世界坐标和测量不确定度。
2. 相机2物理身份、分辨率、对焦和支架未变化。
3. Task18每次运行期间机器人确实静止。
4. 至少12个安全姿态满足XY、J3、Rz跨度。
5. 控制器的`mechanical_center`确实表示J4/TCP轴心；如果TCP另有偏心，后续必须单独
   标定 `T_J4_S`，不能把本外参当作吸盘中心标定。

## 8. 后续顺序

外参通过后，再依次开发吸盘中心标定、固定J3近距profile、硅片/槽观察器、像素到
XY/Rz换算、只读UI、多帧质量门和受监督单步运动。相机2单帧不负责估计绝对Z，Z仍由
J3和独立测量的安全/接触高度决定。
