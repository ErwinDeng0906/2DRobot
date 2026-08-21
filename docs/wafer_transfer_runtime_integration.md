# Wafer Transfer 实时集成记录

更新日期：2026-08-20
开发分支：`agent/integrate-tray-marker-vision`

## 1. 本次范围

本次把分层托盘识别接入 SCARA 主界面，形成不控制硬件的硅片转移跟踪会话：

1. 相机1根据外围 A-H marker 实时建立托盘位姿。
2. 使用固定的 6x6 槽位和槽内 marker ID，不根据当前照片重新排序。
3. 实时输出每个槽的空槽、正常有片、警告、叠片、槽外、画面外和未知状态。
4. 点击选择正常有片的拾取槽；拾取导航不再强制先选放置槽。
5. 机械臂静止时，用5张新鲜同步帧建立本次 `W←T`。
6. 随机械臂状态变化，实时更新吸盘到当前锁定槽的 World XY 距离。
7. 放置槽可后选，但必须已证明为空；本版不实现无 marker 的槽边缘识别。

新会话不执行机械臂运动、Z、J4、DO或真空操作。`ActionWorker` 仍是唯一硬件
所有者，现有只对 P22 开放的运动安全范围也没有扩展到其他槽。

## 2. 数据流程

```text
相机1 BGR帧
  -> TrayPoseTracker / TrayBoardPoseEstimator
  -> TrayVisionAnalyzer
     - 外围marker托盘位姿
     - 固定槽内marker ID
     - 硅片形状质量
     - 36槽占用状态
  -> 5张机械臂静止且时间同步的帧
  -> runtime_tray_registration.py
  -> W←T

最新机械臂关节/位姿 + W←T + 锁定槽T坐标
  -> 目标槽World XY
  -> 吸盘到目标的World XY差值
  -> WaferTransferSession多帧一致性和质量门
  -> UI只读箭头、坐标和导航报告
```

## 3. 新增模块

### `src/scara/vision/close_range_slot_observation.py`

定义相机2的输出协议。当前默认实现
`UnavailableCloseRangeSlotObserver` 始终返回 `unavailable`。这只是接口占位；本版没有槽边缘
算法，不会猜测近距离状态，也不会批准抓取或放置。

接口已经包含槽中心/四角、吸盘中心、硅片中心/四角、中心误差、角度误差、
边缘拟合RMS、最小间隙、J3、J3标定有效范围、标定profile、被提起硅片和
吸盘附着状态。肯定结果必须具备完整证据；仅返回 `aligned` 或 `insertable`
标签不会被状态机接受。

### `src/scara/vision/wafer_transfer_tracking.py`

负责目标锁定和与控制器无关的状态机：

```text
idle
 -> source_selected
 -> source_ready
 -> tracking_pick
 -> waiting_pick_alignment
 -> verifying_pick
 -> picked
 -> tracking_place
 -> waiting_place_alignment
 -> ready_to_place
 -> verifying_place
 -> complete

source_ready + 已证明为空的放置槽 -> route_ready
```

拾取源必须在最近 5 张序号不同的合格帧中至少 3 张为正常占用，且最新帧仍为
`occupied`。任何明确的验证失败都进入 `blocked`。物理事件采用双侧证据：

- 抓取成功：相机2看到硅片被提起，同时相机1确认原槽已经为空。
- 放置成功：相机2不再看到硅片附着于吸盘，同时相机1确认目标槽已有硅片。

相机2的肯定结果还必须通过四个运行时门：视觉证据完整、图像未过期、图像与
机械臂时间差不超过0.35秒、观测中J3与同步机械臂J3差不超过0.15 mm。

### `src/scara/vision/wafer_transfer_runtime.py`

连接 `TrayPoseTracker`、`TrayVisionAnalyzer`、`runtime_tray_registration.py` 和
`WaferTransferSession`。输入是带时间戳的 BGR 帧和只读机械臂状态。

机械臂状态必须不超过1秒，并且与图像采集时间相差不超过0.35秒。`W←T` 需要5张
不同的新帧；现有登记模块会再次检查机械臂静止、marker数量、五帧离散度和托盘移动
范围。目标锁定期间，新旧 `W←T` 相差超过0.75 mm或0.25度时，会话直接进入
`blocked`。

### `src/scara/ui/wafer_transfer_dialog.py`

提供实时点击界面。画面显示：

- 现有36槽状态标注；
- 固定托盘原点、`+X_T` 和 `+Y_T`；
- 青色拾取槽边界和黄色放置槽边界；
- 当前转移阶段；
- 合格 `W←T` 的原点和角度；
- 吸盘 XY 到当前目标的实时箭头、World XY 偏差和距离；
- 最后一次点击的像素、Tray 毫米坐标和匹配槽位。

右侧状态区使用大字体并显示所有选择质量门，内容可以完整滚动查看。
最后一次点击不会随新帧消失，“保存导航报告”可保存当前快照。

### `src/scara/ui/control_widget.py`

在“手眼交互”中增加“转移视觉”入口，复用相机源1和现有只读机械臂状态缓存。
执行其他动作、断开相机或退出程序前，会先停止新的视觉线程。

## 4. 选择和安全条件

只有以下条件全部通过，才允许进入“跟踪”状态：

1. 当前托盘位姿通过现有 marker/PnP 质量门。
2. 拾取槽最新状态必须是 `occupied`，并通过 3/5 帧一致性；警告、叠片、槽外和不确定均拒绝。
3. 拾取导航不要求先选放置槽；进入放置阶段前，放置槽必须满足 `safe_to_use_as_empty=true`。
4. 运行时登记必须为 `status=success`，并包含有限的4x4 `W←T`。
5. 机械臂状态必须新鲜，并与相机帧时间同步。

会话始终输出 `robot_motion_authorized=false`。点击画面不能绕过 `ActionWorker`、
运动学检查、J3条件或当前P22标定边界。

## 5. 使用方法

在实验室 Windows 环境运行：

```bash
python main.py
```

打开 SCARA 页面并连接相机源1，然后进入：

```text
手眼交互 -> 转移视觉
```

保持“拾取槽”模式，等候界面显示近 5 帧中至少 3 帧为正常占用，再点击相应硅片。
程序把点击像素转换为托盘毫米坐标，并匹配最近的固定槽中心。点击位置距离最近槽中心超过
12.5 mm 时会被拒绝。“锁定拾取导航”只开始计算和显示偏差，不移动机械臂。

## 6. 当前 checkout 的标定状态

当前 Mac checkout 中没有已安装平面手眼文件所引用的 Task8
`camera1_suction_target.json`。当前相机内参和托盘几何的 hash 也与该手眼文件记录的
hash 不一致。因此本机界面会正确地把 `W←T` 保持为不可用。

不要绕过这个检查。实验室世界坐标测试前应选择以下一种方式：

1. 恢复手眼文件引用的同一份成功 Task8 运行目录，并核对全部锁定 hash；
2. 使用当前内参和几何重新完成 Task8/Task13，安装一套内部一致的新标定。

在登记不可用时，托盘和槽状态识别仍可显示，但不能声称已经得到吸盘毫米移动量。

## 7. 相机2当前边界

应用中保留 `CloseRangeSlotObserver` 接口，默认实现始终返回不可用。本次已停止槽边缘算法开发，
没有任何近距离识别结果进入拾取导航或运动授权。

## 8. 测试与修改记录

`tests/test_wafer_transfer_tracking.py` 覆盖：

1. 拾取槽 3/5 帧一致性和放置槽的独立视觉证据要求。
2. 机械臂XY变化时世界系距离实时更新。
3. 相机2不可用时保持 fail-closed。
4. 抓取和放置的双侧验证。
5. 机械臂状态过期或时间不同步时拒绝跟踪。
6. 相机2只给出标签但缺少槽/硅片四角、误差、间隙或J3标定证据时拒绝。
7. 相机2帧过期、与机械臂不同步或J3不一致时拒绝。
8. 无放置槽时可启动只读拾取导航，但不能进入放置阶段。

0820 硅片检测回归和标注图见
`docs/wafer_recognition_0820_validation.md`。

本文件是本次实现记录。当前未提交的完整代码差异可在分支
`agent/integrate-tray-marker-vision` 中查看：

```bash
git status --short
git diff -- src/scara/ui/control_widget.py docs/tray_marker_layered_integration.md
git diff --no-index /dev/null src/scara/vision/wafer_transfer_runtime.py
git diff --no-index /dev/null src/scara/vision/wafer_transfer_tracking.py
git diff --no-index /dev/null src/scara/vision/close_range_slot_observation.py
git diff --no-index /dev/null src/scara/ui/wafer_transfer_dialog.py
```
