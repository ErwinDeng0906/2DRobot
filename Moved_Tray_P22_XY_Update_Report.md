# 可移动托盘P22全盘XY升级报告

## 本次更新

### 运动学与标定

- `src/scara/pipeline/kinematics.py`
  - 新增`forearm_pose_W_F(j1,j2)`；F原点固定为J4轴，平面朝向只使用`J1+J2`。
- `src/scara/vision/planar_handeye.py`
  - 从Task8/Task12整槽数据拟合固定高度平面相机1→前臂旋转；同槽重复帧先聚合，完整槽位留出验证，失败不安装。
- `tools/fit_planar_handeye.py`
  - 对现有Task8/Task12执行只读拟合并生成候选报告。
- `Preset Trajectories/task13_planar handeye.py`
  - 当历史跨run质量门失败时，从预设点`P22 float`开始；由刚体Tray geometry生成16个固定观察姿态，不读取Task12记录。
- `src/scara/vision/planar_handeye_runtime.py`
  - Task13照片Stage3后处理；使用Task8+Task13做整槽、跨run验证，成功后原子安装，不再把Task12作为Task13依赖。

### 每次运行的动态登记与控制

- `src/scara/vision/runtime_tray_registration.py`
  - 5帧计算本次W←T；检查平移/yaw范围、离散度、P22不确定度；支持人员确认后的三姿态异常复核。复核观察点按本次W←T重新计算，不使用历史world坐标。
- `src/scara/vision/moved_tray_servo.py`
  - 用本次W←T计算P22 world目标、Tray误差到world命令、分级步长及独立5帧1mm验收。
- `src/scara/vision/moved_tray_positioning_session.py`
  - 会话状态机：登记→必要的三姿态复核→登记后冻结≤2mm航点粗路线→最多32轮毫米闭环→独立hold。
- `src/scara/ui/action_worker.py`
  - 新增严格限定的可移动托盘运行请求类型；所有目标下发前仍由ActionWorker重新读取并独立复核控制器和运动学。
- `src/scara/ui/handeye_demo_dialog.py`
  - “全盘定位”改为动态登记流程，显示托盘变化与置信度；异常时提示三姿态确认；明确P22/XY/J3/Rz边界及无Z/DO/真空。
- `src/scara/vision/full_tray_positioning_session.py`
  - `FullTrayPositioningSession`公开接口切换到新动态会话；旧固定托盘实现仅保留为历史报告回放兼容类。

### 文档与测试

- `docs/full_tray_p22_positioning.md`
  - 重写为动态W←T、粗定位、毫米闭环、hold验收及安全边界。
- `docs/camera1_forearm_planar_handeye.md`
  - 记录平面手眼方程、质量门、现有数据结果和Task13操作要求。
- `tests/test_moved_tray_registration.py`
  - 覆盖J1+J2前臂角、整槽留出拟合、10 mm+5°组合登记、硬边界、动态槽/复核坐标、hold门和ActionWorker新动作上限。
- `tests/test_task13_planar_handeye.py`
  - 覆盖16姿态×10帧任务契约、固定J3/Rz、无Z/DO/真空，以及质量门失败绝不安装。

## 现有数据结论

Task8 `260814171235` 与Task12 `260816224730`的现有数据通过独立姿态、空间激励、固定J3、留出XY和留出yaw门；固定J3最大偏差仅0.0072 mm。但跨run旋转外参差为1.191°，即使使用后来经操作人员授权放宽的1.00°门仍不通过，因此不安装该历史候选。

Task13的正式重采集流程现已与Task12解耦：动作从`P22 float`开始，安装候选由Task8+本次Task13重新拟合。上述Task8+Task12结果只作为历史诊断，不决定Task13起点或路线。

2026-08-17运行`260817155954`的Task8+Task13跨run旋转差为0.982966°。操作人员明确授权将该门放宽至1.00°；其余门全部通过后重新生成并安装正式标定。该门仅余约0.017°，属于低余量通过，后续报告不得隐藏。

最终离线验证：源码编译通过，完整单元测试`144/144`通过（其中成功安装与失败不安装均有离线覆盖），`git diff --check`通过。

## 模式边界

- 新“全盘定位”只授权P22；
- 支持托盘平移模长≤10 mm、yaw≤5°；硬边界10.5 mm/5.25°；
- 相机1、J3、绝对Rz和分辨率必须保持标定条件；
- 不执行Z、硅片、DO或真空；
- 成功结论是“视觉估计进入1 mm”，不是物理硅片放置精度。
