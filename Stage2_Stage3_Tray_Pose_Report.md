# 阶段2/阶段3：Tray Frame、A–H Board 与实时 \({}^{C}T_T\) 报告

生成日期：2026-08-14  
项目：`RobotArm_SCARA_Control 0812`  
范围：只实现和验证视觉/几何软件；未连接或控制机械臂。

## 1. 完成内容

阶段2已经建立：

- 与机械臂坐标明确分离的 Tray Frame `T`；
- 严格正交、6×6、25 mm间距的36个槽底目标中心；
- A–H 外围Marker构成的非共面刚体Board；
- 每个Marker独立的实测高度和严格13.27 mm四角坐标；
- 可重复生成、验证并保存的 `tray_board_geometry.json`。

阶段3已经实现：

- 使用 `DICT_4X4_1000` 检测A–H；
- 将图像角点与Tray Frame中的3D角点配对；
- 使用 `solvePnPRansac` 求 \({}^{C}T_T\)；
- 整个Marker级别的重投影离群剔除；
- LM位姿精修；
- 全局及逐Marker重投影RMS；
- 可见Marker数、RANSAC内点比例、Board跨度、正深度、相机所在平面一侧等质量门；
- 原始位姿与时间滤波位姿分离；
- 帧间位姿跳变拒绝；
- 单图、文件夹批处理和相机1实时预览工具；
- 138张历史相机1照片的离线验证。

## 2. 输入证据与关键结论

### 2.1 `scara_presets.json` 的数值含义

文件中的四个数为：

```text
[J1_deg, J2_deg, J3_mm, J4_deg]
```

不是机械臂Cartesian `x/y/z/r`，更不是Tray坐标。因此程序首先用SCARA双连杆正运动学转换J1/J2：

\[
x_M=L_1\cos q_1+L_2\cos(q_1+q_2)
\]

\[
y_M=L_1\sin q_1+L_2\sin(q_1+q_2)
\]

其中 \(L_1=225\rm\,mm\)，\(L_2=175\rm\,mm\)。机械平面坐标只用于把示教数据转换到独立的Tray Frame，后续槽目标和Board全部以 `T` 表达。

### 2.2 A–H的dictionary和ID

`C:\Users\Admin\Desktop\2D robot\ARUCO Markers.zip` 中的文件名为 `4x4_1000-*.svg`，所以dictionary确定为：

```text
cv2.aruco.DICT_4X4_1000
```

附件照片经OpenCV实际解码，外围对应关系为：

| Marker标签 | ArUco ID |
|---|---:|
| A | 1 |
| B | 2 |
| C | 3 |
| D | 4 |
| E | 5 |
| F | 6 |
| G | 7 |
| H | 8 |

内部槽Marker不属于A–H外围Board，即使某张其他标定板中出现相同ID，也必须通过A–H刚体几何一致性才能得到有效位姿。

## 3. Tray Frame定义

程序采用以下定义：

- 原点 \(O_T\)：P00槽中心，位于槽底目标平面；
- \(+X_T\)：P50槽中心指向P00槽中心；
- \(+Y_T\)：P05槽中心指向P00槽中心；
- \(+Z_T=X_T\times Y_T\)：从槽底目标平面指向Marker/相机一侧。

示教点经FK后的两条原始轴夹角为：

```text
90.242570°
```

因为用户明确规定托盘严格正交，程序不会把0.24257°示教误差变成斜坐标系。实现方式是：

1. \(X_T\)严格取P50→P00的单位方向；
2. P05→P00方向对 \(X_T\) 做Gram–Schmidt正交化，得到 \(Y_T\)；
3. 用叉积得到右手系 \(Z_T\)。

四个示教槽中心相对严格125×125 mm设计角的残差为：

| 槽 | \(\Delta X_T\) mm | \(\Delta Y_T\) mm |
|---|---:|---:|
| P00 | 0.000 | 0.000 |
| P05 | +0.531 | -0.509 |
| P50 | +0.531 | 0.000 |
| P55 | +0.507 | -0.030 |

这些残差保存在几何JSON诊断字段中，但不会改变严格的25 mm槽间距。

## 4. 6×6槽目标

槽名采用 `Prc`：

- `r=0..5`：从P00向P50增加，即 \(-X_T\)；
- `c=0..5`：从P00向P05增加，即 \(-Y_T\)。

槽底目标中心为：

\[
{}^{T}p_{rc}=\begin{bmatrix}-25r&-25c&0\end{bmatrix}^T\ \rm mm
\]

例如：

```text
P00 = [   0,    0, 0] mm
P05 = [   0, -125, 0] mm
P50 = [-125,    0, 0] mm
P55 = [-125, -125, 0] mm
```

## 5. A–H四角外推与高度

OpenCV检测角点顺序为：

```text
[UL, UR, DR, DL]
```

每个Marker使用三个示教量：中心、UL和DL。令Marker的水平方向单位向量为 \(u\)，向下方向为 \(v\)，实测边长 \(s=13.27\rm\,mm\)，半边长 \(h=s/2\)。程序由：

- `DL - UL` 得到左边方向；
- `center - (UL+DL)/2` 得到左边指向中心的方向；
- 将二者正交化，得到严格正交的 \(u,v\)。

然后以示教中心为精确中心重建刚性正方形：

\[
UL=C-hu-hv,\quad UR=C+hu-hv
\]

\[
DR=C+hu+hv,\quad DL=C-hu+hv
\]

槽底机械J3为 `-52.01 mm`。Marker在Tray Frame中的高度使用：

\[
z_{T,i}=J3_i-(-52.01)
\]

| 标签 | ID | \(z_T\) mm | 示教UL/DL拟合RMS mm |
|---|---:|---:|---:|
| A | 1 | 2.1908 | 0.3229 |
| B | 2 | 3.3908 | 0.3841 |
| C | 3 | 2.1908 | 0.2086 |
| D | 4 | 2.7908 | 0.2850 |
| E | 5 | 1.9908 | 0.0072 |
| F | 6 | 2.0054 | 0.0072 |
| G | 7 | 1.8054 | 0.1421 |
| H | 8 | 1.8054 | 0.1422 |

因此A–H Board被正确建模为“同一刚体、Marker表面有小高度差”的非共面Board，而不是强行压到同一个Z平面。

## 6. 阶段3：\({}^{C}T_T\) 的意义和求法

齐次变换定义为：

\[
{}^{C}T_T=
\begin{bmatrix}
{}^{C}R_T & {}^{C}t_T\\
0\ 0\ 0 & 1
\end{bmatrix}
\]

它把Tray坐标点转换到相机坐标：

\[
{}^{C}p={}^CR_T\,{}^{T}p+{}^Ct_T
\]

每帧处理流程：

1. 用 `DICT_4X4_1000` 检测Marker；
2. 只保留配置中的ID 1–8；
3. 要求至少3个可见外围Marker和至少50 mm Board跨度；
4. 按 `[UL,UR,DR,DL]` 连接3D Tray角点和2D图像角点；
5. `solvePnPRansac` 求初始 \(R,t\)；
6. 按逐Marker重投影误差剔除异常Marker；
7. 用剩余角点做LM精修；
8. 生成 \({}^{C}T_T\) 和逆变换 \({}^{T}T_C\)；
9. 验证Board位于相机前方、相机位于Tray目标平面正确一侧；
10. 输出逐Marker RMS、全局RMS、RANSAC内点及通过/拒绝原因。

重投影RMS为：

\[
e_{RMS}=\sqrt{\frac{1}{N}\sum_j
\left[(u_j-\hat u_j)^2+(v_j-\hat v_j)^2\right]}
\]

历史真实托盘数据的RMS主要为1.3–2.7 px，因此当前Board几何诊断门使用3.0 px。这个门只判断A–H几何与当前检测是否一致，不代替相机内参审批。

## 7. 实时滤波

实时Tracker只接收已经通过单帧质量门的位姿，并分别处理：

- 平移：指数平滑；
- 旋转：在SO(3)上以 \(R_f=R_p\exp[\alpha\log(R_p^TR_c)]\) 插值；
- 突然超过35 mm或20°的跳变：拒绝；
- 连续丢失15帧：清空旧位姿，避免长期沿用过期结果。

原始 \({}^{C}T_T\) 与滤波后 \({}^{C}T_T\) 始终分开返回，便于诊断。

## 8. 历史照片验证

验证只读取 `Trajectory Photos` 下所有 `1_*.jpg`，明确排除全部 `2_XXX`相机2照片，共138张。

| 运行目录 | 相机1图片 | 求解成功 | 通过当前Board质量门 | RMS中位数 px |
|---|---:|---:|---:|---:|
| 260812001406 | 4 | 3 | 3 | 1.735 |
| 260812010650 | 4 | 3 | 3 | 1.720 |
| 260812015730 | 4 | 2 | 2 | 1.657 |
| 260812132344 | 36 | 33 | 33 | 1.758 |
| 260812141055 | 4 | 4 | 4 | 1.880 |
| 260812152915 | 36 | 32 | 32 | 1.743 |
| 260813213452 | 1 | 0 | 0 | — |
| 260813215627 | 49 | 0 | 0 | — |

2026-08-12目录是真实托盘照片；失败帧均因只看到2个外围Marker，低于当前至少3个的成熟质量门。

2026-08-13的49张是Task6 ChArUco内参采集照片。`DICT_4X4_50`中的ID 1–8也可被更大的 `DICT_4X4_1000`解码，但这些Marker的空间排列不符合A–H Board，所以全部被Board几何/RANSAC拒绝。这证明程序不是“看见ID 1–8就输出位姿”。

两次36帧重复托盘扫描中，共有32个同名轨迹点都能求解：

| 重复性指标 | 中位数 | 90百分位 | 最大值 |
|---|---:|---:|---:|
| \({}^{C}T_T\) 平移差 mm | 0.071 | 0.703 | 2.069 |
| \({}^{C}T_T\) 旋转差 ° | 0.116 | 1.316 | 3.008 |

这说明阶段3在现有历史图像上具有较好的重复性，但最大值仍提醒后续闭环控制必须使用质量门和多帧稳定判断。

## 9. 文件分工

### `src/scara/vision/tray_board_geometry.py`

- 读取并验证示教关节；
- SCARA正运动学；
- 严格正交Tray Frame；
- 36槽设计几何；
- 中心+UL/DL外推Marker严格四角；
- 独立Marker高度；
- 几何结构验证和原子JSON保存。

### `src/scara/calib/tray_board_geometry.json`

- 阶段2正式几何数据；
- 轴定义、36槽、8个Marker四角、ID、各自高度；
- 示教残差和UL/DL拟合诊断；
- 后续位姿估计不再直接依赖机械臂绝对坐标决定槽位置。

### `src/scara/vision/tray_pose_estimator.py`

- 单帧A–H检测；
- RANSAC PnP和Marker离群剔除；
- 求 \({}^{C}T_T\)、\({}^{T}T_C\)；
- 重投影、内点、深度、相机所在侧等质量门；
- Tray点投影到图像。

### `src/scara/vision/tray_pose_tracker.py`

- 时间滤波；
- 帧间平移/旋转跳变拒绝；
- 连续丢帧失效机制；
- 明确分开原始与滤波位姿。

### `tools/generate_tray_board_geometry.py`

- 从 `scara_presets.json` 重新生成几何JSON；
- 显示轴角度、Marker高度和角点拟合误差。

### `tools/estimate_tray_pose.py`

- 单张或整个目录的离线位姿检测；
- 只接受文件名 `1_XXX`；
- 保存逐帧JSON和可选叠加图。

### `tools/live_tray_pose.py`

- 打开相机1实时显示A–H和Tray轴；
- 显示时间滤波结果；
- 不导入机械臂控制模块，不能移动机械臂。

### `tools/validate_tray_pose_history.py`

- 自动验证全部历史相机1照片；
- 排除相机2；
- 输出逐运行统计和重复扫描一致性。

### `tests/test_tray_stage2_stage3.py`

- Tray Frame正交性；
- 36槽及13.27 mm刚性四角；
- 齐次变换往返；
- 合成图像恢复 \({}^{C}T_T\)；
- 整个Marker异常剔除；
- Tracker滤波/跳变拒绝；
- 未批准内参默认拒绝。

## 10. 使用方法

在Anaconda Prompt中：

```powershell
conda activate scara_cvdev
cd "C:\Users\Admin\Desktop\2D robot\RobotArm_SCARA_Control 0812"
```

重新生成阶段2几何：

```powershell
python tools\generate_tray_board_geometry.py
```

对一张相机1照片做阶段3离线检测：

```powershell
python tools\estimate_tray_pose.py `
  "Trajectory Photos\260812132344\1_067.jpg" `
  --intrinsics "src\scara\calib\camera1_intrinsics.json" `
  --annotated-dir "tray_pose_annotated"
```

验证所有历史相机1照片：

```powershell
python tools\validate_tray_pose_history.py `
  --intrinsics "src\scara\calib\camera1_intrinsics.json"
```

相机1实时诊断：

```powershell
python tools\live_tray_pose.py
```

运行单元测试：

```powershell
python -m unittest -v tests.test_tray_stage2_stage3
```

## 11. 当前安全阻塞：必须重新取得正式内参

桌面项目当前不存在：

```text
src/scara/calib/camera1_intrinsics.json
```

唯一历史参数位于：

```text
Trajectory Photos/260813215627/camera1_intrinsics.json
```

其状态是：

```text
rejected_pose_diversity
```

阶段3默认拒绝加载任何 `status != success` 的内参。这不是软件缺陷，而是防止低姿态多样性的内参进入精密放置。

历史验证工具可以显式使用 `--allow-unapproved-intrinsics`，但只用于离线评估，不能授权机械臂放置。下一步必须重新运行Task6，手动让ChArUco板产生足够的多方向倾斜，取得：

```text
src/scara/calib/camera1_intrinsics.json
status = success
```

之后阶段2/3代码无需修改，即可进行正式相机1实时 \({}^{C}T_T\) 验证。

## 12. 尚未包含的后续阶段

本阶段没有实现：

- camera-to-suction / hand-eye关系；
- 由相机位姿换算吸盘XY修正量；
- 机械臂闭环运动；
- 吸盘固定高度target；
- Z轴视觉调整或放置动作。

这些应在正式内参、A–H实时位姿和相机到吸盘标定都通过后再启用。
