# Stage 7A：P22 受监督单步 XY 视觉修正

## 1. 目标与边界

Stage 7A 将已经批准的 Stage 3、4、5 结果连接成一次真实但受监督的
XY 小步修正。Task 9 的九点扫描属于离线系统辨识；正常执行 Task 10
时不会重做九点标定，只读取正式安装的 P22 Jacobian。

本阶段只允许：

- 相机 1，1280×720；
- 目标槽 P22；
- Task 9 锚点每轴 ±2 mm 的局部模型，并保留 0.20 mm 安全边距；
- 固定观察高度 J3 与固定绝对 Rz；
- 一次人工确认的 XY 小步，向量长度和单轴分量均不超过 0.25 mm；
- 修正前 5 张、修正后 5 张照片。

本阶段不执行 Z 运动、不控制 DO 或真空、不自动循环，也不宣称已经完成
真实硅片放置精度验收。关闭弹窗、点击“不运动”或任一安全门失败时，
均不会下发运动。

## 2. 坐标与控制方程

Stage 3 把同一帧中的目标槽中心和 Stage 4 suction target 投影到带畸变的
原始图像。图像误差定义与 Task 9 完全一致：

\[
e=
\begin{bmatrix}e_u\\e_v\end{bmatrix}
=
\begin{bmatrix}
u_{slot}-u_{suction}\\
v_{slot}-v_{suction}
\end{bmatrix}.
\]

Task 9 得到的局部增量模型为：

\[
\Delta e \approx J\,\Delta q,\qquad
\Delta q=\begin{bmatrix}\Delta X\\\Delta Y\end{bmatrix}.
\]

若希望下一帧误差趋近零，完整校正为：

\[
\Delta q_{full}=-J^{-1}e.
\]

Stage 7A 不直接执行完整校正，而先乘阻尼系数
\(g=0.6\)：

\[
\Delta q_{raw}=g\,\Delta q_{full}.
\]

再做二维向量限幅，默认 \(s_{max}=0.25\,\text{mm}\)：

\[
\Delta q_{cmd}=
\begin{cases}
\Delta q_{raw}, & \|\Delta q_{raw}\|_2\le s_{max}\\
s_{max}\dfrac{\Delta q_{raw}}{\|\Delta q_{raw}\|_2},
& \|\Delta q_{raw}\|_2>s_{max}.
\end{cases}
\]

同时逐轴检查 \(|\Delta X|\le0.25\) mm、
\(|\Delta Y|\le0.25\) mm。UI 和 JSON 中显示的预计世界坐标终点为：

\[
W_{pred}=W_{current}+\Delta q_{cmd},
\]

预计图像误差为：

\[
e_{pred}=e+J\Delta q_{cmd}.
\]

误差 \(e\) 不是单帧值。程序要求修正前恰好采集 5 张不同照片，至少 3 张
通过 Stage 3，并对合格帧的 \(e_u,e_v\) 分别取中位数；窗口内误差的最大
偏离还必须低于离散度门限，避免把抖动当成可执行校正。

## 3. 固定 J3/Rz 的运动规划

候选世界坐标终点先由上式得到，再使用当前 IK 分支求 J1/J2。J3 保持不变，
J4 重新计算以保持 Task 9 标定时的绝对 Rz：

\[
J4_{target}=Rz_{fixed}-J1_{target}-J2_{target}+90^\circ.
\]

控制器实际按 J1→J2→J3→J4 顺序执行。为避免当前 J4 没有跟随
J1/J2 点动而造成整段 Rz 偏差，经同一次人工确认后，执行器先在
当前 XY/J3 不变的条件下仅调整 J4：

\[
J4_{pre}=Rz_{fixed}-J1_{current}-J2_{current}+90^\circ.
\]

预补偿到位并重新通过控制器、到位和运动学门后，才执行
J1→J2→J3→最终 J4 的 XY 修正。规划器枚举这两段的所有逐轴
中间状态，并要求任一相邻状态的 XY 瞬态不超过 0.50 mm。
相同的规划审计会执行两次：弹窗显示前一次，人员确认后由拥有控制器的
`ActionWorker` 读取全新状态并独立复核一次。

## 4. 默认拒绝的安全门

### 4.1 标定与视觉门

- 正式 Stage 5 文件存在，schema、status、六个质量门全部合格；
- 相机内参、Tray geometry、Stage 4 suction target 的 SHA-256 与
  Stage 5 锁定值一致；
- 相机源、分辨率、误差符号和坐标定义一致；
- 目标名为 P22，且 P22 位于 `valid_target_names`；
- 5 张照片文件名/途径点序号唯一，至少 3 张 Stage 3 PASS；
- Stage 3 的 marker、RANSAC、重投影、正深度、时序跳变门通过；
- 图像误差窗口稳定，机械臂在窗口内保持静止；
- 五帧视觉所绑定的机械臂状态与弹窗规划请求状态之差不超过0.05 mm / 0.05°；
- Jacobian 有限、可逆，当前正式文件 hash 与动作锁定 hash 一致。

### 4.2 局部模型与运动学门

- 当前世界 XY 与预计终点都在锚点每轴 ±1.80 mm 内；
- 控制器按J1→J2→J3→J4逐轴执行时，每一个中间状态也必须位于锚点每轴
  ±1.80 mm 内；
- 当前 J3 与标定 J3 偏差不超过 0.15 mm；
- 预补偿前的当前绝对 Rz 与标定值偏差不超过 0.20°；
- 最终目标绝对 Rz 与标定值偏差不超过 0.15°；
- J4 预补偿完成后，逐轴运动的最大绝对 Rz 偏差不超过 0.30°；
- 当前控制器 XY 与 J1/J2 正运动学闭合误差不超过 0.20 mm；
- 目标可达、IK 分支连续、目标 J3 不变；
- 校正向量和单轴分量不超过 0.25 mm；
- 逐轴执行的最大 XY 瞬态不超过 0.50 mm。

### 4.3 下发前控制器门

- 控制器连接、使能且空闲；
- 无报警、无硬急停、无软急停；
- 人员在白底黑字弹窗明确点击“确认执行一次XY修正”；
- 弹窗 request ID、P22、Jacobian hash 和 proposal ID 均匹配；
- 人员查看期间，XY 漂移不超过 0.05 mm，任一关节漂移不超过 0.05；
- 提案年龄不超过 20 s；查看弹窗期间禁止触碰托盘、相机或机械臂，超时必须
  重新运行并取得一组新照片；
- 关节下发/到位容差使用0.01°（J3为0.01 mm），不能沿用普通任务的0.05：
  当前控制器同时把该值作为命令死区，过大会直接跳过Stage7A的小角度修正；
- 弹窗显示的 \(\Delta X,\Delta Y\) 与候选关节目标正运动学结果相差
  不超过 0.002 mm；
- 全部目标由执行线程再次审计并写入 `points.json` 后，才允许第一次物理命令。
- 到位后再次要求连接、使能、无报警、无急停、线程未停止且关节误差合格；
  后续五张复测图绑定的控制器状态也必须全部安全，才可判定响应通过。

任一门缺失也视为失败，程序不会把“没有检查”当成通过。

## 5. 修正后响应验证

修正后再次采集 5 张照片并构造 \(e_{after}\)。实际误差变化、模型预测与
innovation 定义为：

\[
\Delta e_{actual}=e_{after}-e_{before},
\]

\[
\Delta e_{pred}=J\Delta q_{actual},
\]

\[
r_{innovation}=\Delta e_{actual}-J\Delta q_{actual}.
\]

报告会保存误差是否下降、下降比例、innovation 大小、实际命令跟踪误差等门。
这些结果用于判断 Jacobian 方向、增益和机械回差是否合理；Stage 7A 不在线
改写正式 Jacobian，也不会因为一次结果合格而自动执行第二步。

## 6. 软件权限分层

- `Tasks/task10.py`：只声明起点检查、等待、途径点记录、相机 1
  拍照和一个受监督运行时动作。
- `scara.vision.xy_visual_servo`：纯计算；实现稳定窗口、控制方程、预测和响应
  验证，不导入 Qt、相机或控制器。
- `scara.pipeline.xy_correction_planner`：纯运动学；实现固定 J3/Rz 的 IK、
  局部域和逐轴瞬态审计，不下发命令。
- `scara.vision.stage7a_runtime`：复用 Stage 3/4/5、生成弹窗内容、保存图片和
  分析 JSON；不持有控制器。
- `scara.ui.stage7a_dialog`：显示逐帧数据、修正前误差、ΔX/ΔY、预计终点和
  全部安全门；默认按钮是不运动。
- `scara.ui.action_worker`：唯一持有控制器的层；人员确认后重新读取状态、
  重做独立安全审计，再决定是否下发一次绝对关节目标。

旧的 `scara.vision.servo` 是另一套固定俯视相机流程，步长和安全域与本系统
不一致，本阶段不调用它。

## 7. 保存内容

运行目录继续使用 `Trajectory Photos/YYMMDDhhmmss`，包含：

- `1_001.jpg` 至 `1_010.jpg`：与现有任务一致的原始照片命名；
- `annotated_stage7a/`：A–H重投影、Tray轴、槽中心、suction target和误差箭头；
- `points.json`：每个途径点的关节、机械中心、控制器安全状态、照片关联，
  以及完整的 `runtime_moves` 审批/复核/实际到位审计；
- `stage7a_single_step.json`：锁定输入、逐帧 Stage 3 数据、修正前误差、
  完整/阻尼/限幅后的ΔX和ΔY、预计终点、预计误差、全部安全门、人员决定、
  执行线程下发前/到位后的独立运动审计、修正后误差和响应验证。

即使人员选择不运动，原始前后照片、决定和未通过原因也会保存，便于区分
“安全拒绝”与“程序失败”。
