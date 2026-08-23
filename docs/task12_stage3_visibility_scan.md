# Task12：36槽 Stage3 可观测性扫描

## 目的

Task12回答一个独立于Jacobian的问题：当相机1和吸盘位于P00-P55各槽上方的固定观察高度时，现有A-H外围Marker能否稳定提供合格的托盘位姿 `^C T_T`。

它是诊断任务，不计算、不安装任何Jacobian，也不授权Stage7运动。只有目标槽本身可以稳定通过Stage3，后续该槽的局部Jacobian标定和视觉闭环才有可靠的图像误差输入。

## 运动与采集

- UI导入文件：`Tasks/task12_code visibility scan.py`
- 起点：已示教的 `P00 float`
- 路线：P00-P55全部36槽的相邻槽回环蛇形路线
- 每槽：稳定等待0.8秒，采集相机1图像20帧
- 总量：720个途径点和720张照片
- 槽间25 mm路径：拆成不超过0.60 mm的机械XY目标
- 始终保持P00 float的J3和绝对Rz
- 结束：沿相邻槽路径返回精确的 `P00 float`
- 明确禁止：接触高度下降、视觉修正、动态Jacobian运动、DO、真空

Task12通过现有运动学求解J1/J2/J4，并在导入时检查控制器实际的 `J1 -> J2 -> J3 -> J4` 逐轴执行顺序。每段逐轴瞬时机械XY移动必须不超过1.50 mm，绝对Rz瞬时偏差必须不超过0.35°，否则任务在运动前拒绝生成。

## 每帧算法

Task12不复制PnP方程，直接调用现有：

1. `TrayBoardPoseEstimator`检测A-H并运行RANSAC PnP；
2. 检查可见Marker数、使用Marker数、RANSAC内点、物理跨度、正深度；
3. 检查全局及逐Marker重投影误差；
4. `TrayPoseTracker`检查同一槽20帧中的平移和旋转跳变；
5. 每换一个槽，重置tracker，避免把25 mm机械移动误认为托盘位姿跳变。

一帧计为combined pass必须同时满足：

```text
Stage3 quality_passed
AND temporal accepted_by_tracker
AND T_C_T exists
```

## 槽位分类

每槽以预期20帧为分母：

- `ready`：20帧完整，并且combined pass不少于16帧，即通过率至少80%；
- `marginal`：通过率达到50%，但未满足ready；
- `not_observable`：通过率低于50%。

`marginal`和`not_observable`均不得直接解释成“Stage7可用”。报告保留失败原因次数、Marker 1-8各自出现频率、被剔除频率、RANSAC角点数量以及重投影RMS的min/median/P95/max，供后续调整相机视野和Marker布局。

## 输出文件

所有文件写入UI为本次动作创建的时间戳目录：

- `points.json`：原始机械臂状态、照片关联，并追加逐帧Stage3/时序结果；
- `task12_stage3_visibility_scan.json`：36槽汇总和全部逐帧记录；
- `task12_stage3_visibility_scan.md`：便于人工阅读的36槽表格；
- `annotated_stage3/`：每张照片对应的Stage3标注图；
- `task12_stage3_visibility_scan_error.log`：仅在报告处理发生软件异常时生成。

如果动作完整执行但部分槽不可观测，报告状态为 `visibility_gaps_detected`。这表示扫描成功发现了覆盖缺口，不属于软件后处理失败，因此Task12不会因为某个槽未通过而丢弃其余诊断结果。

## 执行前人工检查

1. 手动前往 `P00 float`，确认无硅片、真空关闭；
2. 确认固定高度的完整托盘上方路径无夹具、电线或其他障碍；
3. 确认相机1仍为1280×720，曝光与焦距保持锁定；
4. 确认A-H与托盘刚性固定，任务期间托盘不会移动；
5. 降低机械臂速度，确认物理急停可用；
6. 从UI“执行任务”导入Task12并阅读两次安全确认。

Task12结束后，先查看Markdown中的非ready槽，再通过JSON的`failure_reason_counts`区分是Marker数量不足、重投影误差过高、时序跳变还是图片处理异常。只有可观测性问题解决后，才应在对应槽进行local Jacobian test。
