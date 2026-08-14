# 阶段 5：机械臂 XY 命令到图像误差的局部 Jacobian

本文是 `Preset Trajectories/task9_jacobiantest.py` 与
`src/scara/vision/xy_image_jacobian_runtime.py` 的算法和安全约定。阶段 5 的目的，是在
固定观察高度和固定吸盘朝向下，测量机械臂世界坐标 XY 小命令会怎样改变相机1中的槽位误差。

## 1. 已锁定的输入

阶段 5 不重新标定阶段 1–4，而是读取并锁定：

1. `camera1_intrinsics.json`：相机1的内参矩阵 (K)、畸变参数和分辨率；
2. `tray_board_geometry.json`：外围 A–H 和 6×6 槽中心在托盘坐标系 (T) 中的位置；
3. 最新且状态为 `success` 的 `camera1_suction_target.json`：固定高度下吸盘轴在相机坐标系
   中的点 ({}^{C}\mathbf p_S) 及其畸变图像像素 ((u_S,v_S))。

运行时对三份输入计算 SHA-256。阶段 5 的结果保存这三个 hash；任一输入发生变化后，旧
Jacobian 不得继续用于验证或控制。

本次锚点是：

\[
{}^T\mathbf p_{P22}=(-50,-50,-2)\ \text{mm}.
\]

Task9 只在 `P22 float` 的 J3 和末端 (R_z) 下有效。

## 2. 图像误差的定义

阶段 3 对每帧外围 A–H 求得：

\[
{}^C T_T=
\begin{bmatrix}
{}^C R_T & {}^C\mathbf t_T\\
0&1
\end{bmatrix}.
\]

槽中心先变换到相机坐标系：

\[
{}^C\mathbf p_{slot}
= {}^C R_T\,{}^T\mathbf p_{P22}+{}^C\mathbf t_T,
\]

再使用已标定的 (K) 和 `distCoeffs` 投影为畸变图像坐标
((u_{slot},v_{slot}))。图像误差定义为：

\[
\mathbf e=
\begin{bmatrix}e_u\\e_v\end{bmatrix}
=
\begin{bmatrix}
u_{slot}-u_S\\
v_{slot}-v_S
\end{bmatrix}.
\]

因此，绿色槽中心与红色 suction target 重合时，
(mathbf e=(0,0))。箭头从红色 target 指向绿色槽中心，表示当前误差方向。

### 两种“误差最小化”不能混用

阶段 3 的 A–H 重投影误差用于判断当前托盘位姿估计是否可信。`solvePnP`/后续优化求解的是：

\[
\min_{{}^C R_T,{}^C\mathbf t_T}
\sum_i
\left\|
\mathbf u_i-
\pi\!\left(K,\mathrm{dist},{}^C R_T{}^T\mathbf P_i+{}^C\mathbf t_T\right)
\right\|_2^2,
\]

其中 ({}^T\mathbf P_i) 是 A–H 的已知物理角点，(\mathbf u_i) 是检测到的图像角点。
这个 RMS 越低，说明“托盘怎样投影到图像”解释得越一致；它不是机械臂要追踪的目标量。

阶段 5/后续闭环真正要减小的是槽位对吸盘的对准代价：

\[
\min_{\Delta\mathbf q}
\left\|\mathbf e+J_{e\leftarrow q}\Delta\mathbf q\right\|_2^2.
\]

当 (J) 可逆且局部线性模型有效时，一步理论修正为
(\Delta\mathbf q=-J^{-1}\mathbf e)。因此必须先通过 Stage3 重投影质量门，
再计算槽位误差；绝不能通过移动机械臂去“追低”一个不合格的 A–H 重投影 RMS。

## 3. 局部 Jacobian 方程

机械臂世界坐标系中的小 XY 命令记为：

\[
\Delta\mathbf q=
\begin{bmatrix}\Delta x\\\Delta y\end{bmatrix}\ \text{mm}.
\]

在 P22 附近、固定 J3 和固定 (R_z) 的局部范围内，用一阶仿射模型：

\[
\mathbf e_i = J_{e\leftarrow q}\Delta\mathbf q_i+\mathbf b+\boldsymbol\epsilon_i,
\]

其中：

- \(J_{e\leftarrow q}\) 是 \(2\times2\) Jacobian，单位为 px/mm；
- \(\mathbf b\) 是零命令偏移处的图像误差；
- \(\boldsymbol\epsilon_i\) 是视觉、到位和模型线性化残差。

相机1固定在小臂上。J1/J2 变化会同时造成相机平移和相对托盘旋转，因此不能简单假设
“世界 +X 永远对应图像 +u”。Task9 实测得到的 Jacobian 会把这些局部耦合包含在内。

若未来进入明确授权的闭环控制，理论修正命令为：

\[
\Delta\mathbf q_{correct}=-J_{e\leftarrow q}^{-1}\mathbf e.
\]

当前阶段 6 的动态演示只显示该结果，不下发任何运动命令。

## 4. Task9 采集动作

Task9 使用 `ACTION_API_VERSION = 1`，必须从人工示教的 `P22 float` 开始。

采集偏移为：

\[
\Delta x,\Delta y\in\{-2,0,+2\}\ \text{mm}.
\]

九个位置各采 12 帧相机1图像。由于控制器实际按 J1→J2→J3→J4 逐轴执行，单看相邻
采集端点的 2 mm 距离还不够：每条 2 mm 网格边和返回边现在都拆成两个不超过 1 mm 的
Cartesian 中转目标，中转点不采图。按控制器真实逐轴顺序枚举瞬时正运动学后，最大相邻
平移为 1.226076 mm，低于 2.0001 mm 测试门；最后精确返回 P22。

为保持固定姿态，脚本不会直接使用只改 J1/J2 的相对步进。它先由 P22 的 J1/J2 正运动学
得到 J4 轴世界坐标，再对每个 XY 偏移运行 2R 逆运动学，并按

\[
J4=R_z-J1-J2+90^\circ
\]

重算 J4。每个采集目标因此具有相同 J3 和相同末端 (R_z)。Task9 没有 Z 动作、没有旋转
扫描、没有 DO/真空命令；开始前必须由人员确认真空关闭、低速和急停可用。

途径点名称编码了可审计的命令和帧号，例如：

```text
TASK9|target=P22|dx=-2.000|dy=+0.000|frame=01/12
```

## 5. 逐帧质量门

每张照片继续使用阶段 3 的完整质量门：

- 可见外围 Marker 数量；
- RANSAC 内点；
- 全局和逐 Marker 重投影 RMS；
- 正深度；
- 帧间平移/旋转跳变。

阶段 5 另外检查：

- (|J3-J3_{P22}|\leq0.20\) mm；
- 固定 (R_z) 漂移不超过 (0.20^\circ)；
- 实际回读 XY 相对位移与命令偏移的差不超过 0.75 mm；
- 槽中心能够以正深度投影到图像。

任一门失败的帧仍写入 `points.json` 供诊断，但不进入 Jacobian 拟合。

## 6. 鲁棒拟合和质量门

同一偏移先按图像误差的中位数和 MAD 剔除异常帧，每个偏移至少需要 5 帧有效数据。之后
对九个偏移做最小二乘仿射拟合，并最多迭代五次按残差中位数/MAD剔除异常偏移。

最终要求：

- 至少 7 个不同偏移可用；
- 拟合 RMS 不超过 1.5 px；
- 留一偏移交叉验证 RMS 不超过 2.5 px；
- Jacobian 条件数不超过 25；
- 最小奇异值至少为 0.25 px/mm；
- Jacobian 可逆。

只有所有门均通过、首个零偏移实际机械臂 XY 已记录、且 108 帧逐图后处理均未发生 fatal
时，结果状态才是 `success`，并原子写入：

```text
src/scara/calib/camera1_xy_image_jacobian.json
```

质量或后处理失败时仍在本次时间戳文件夹保存完整 JSON、Markdown 和逐帧数据，但不会覆盖
项目内已批准的 Jacobian。Stage5 结果 schema 2 同时记录 `anchor_robot_xy_mm` 和
`runtime_processing`，供动态演示核验局部适用域及失败审计。

## 7. 独立验证的含义

拟合器执行 leave-one-offset-out（留一偏移）验证：每次拿掉一个网格偏移，用其余偏移重新拟合，
再预测被拿掉偏移的误差。这能够发现线性模型不一致或单个偏移异常，但它不是实际闭环放片测试。

“手眼交互”弹窗中的“Jacobian验证”只执行以下只读检查：

1. 当前内参、Tray几何、Stage4吸盘target的 hash 与 Jacobian 文件一致；
2. Jacobian 的状态、条件数、留一验证和当前目标名均通过；
3. 最新只读机械臂状态不超过 1 秒，并且当前 world XY、J3、Rz 位于 Task9 的 P22 局部域；
4. 理论修正量每轴不超过标定范围，且“当前偏移+修正”的预测终点仍在该范围内；
5. 用当前实时帧显示误差和理论修正量。

该按钮不移动机械臂。Task9 是本阶段唯一获准执行 1–2 mm 小幅运动的程序。

## 8. 使用边界

- 当前标定锚点和 `valid_target_names` 只有 P22；不能把一个局部 Jacobian 无验证地宣称为全托盘
  通用模型。
- 相机安装、镜头、焦距、分辨率、托盘几何、吸盘转接头、J3 或 (R_z) 改变后必须重新标定。
- 动态演示选择其他槽时，阶段 3+4 仍可完成“只判断”：显示槽中心、suction target 和图像误差；
  但若目标不在 Jacobian 的有效范围内，界面必须明确显示“不提供运动修正”。
- UI 中默认的 \(|e|\leq3\) px 只是一条明确显示的视觉像素阈值，不是硅片放置精度验收；
  在用户给出槽/硅片允许的物理公差并完成Z轴与真实放片验证前，不得把它称作“放置合格”。
- 进入真正闭环控制前，还需要独立的限幅、工作区、可见 Marker、质量门和每步复测策略；本文件
  不授权自动移动或放片。
