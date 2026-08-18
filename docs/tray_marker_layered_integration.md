# Tray Marker 分层合并记录

日期：2026-08-18
开发分支：`agent/integrate-tray-marker-vision`

## 1. 合并范围

旧版工具完整保留在 `tools/tray_marker_detector_v2/`，用于结果对照和回归。新代码没有修改 SCARA 控制、ActionWorker、拍照流程或任何机械臂运动接口。

本次把可复用逻辑按职责放入 `src/scara/vision/`：

1. `slot_marker_observation.py`
   - 从 `tray_board_geometry.json` 读取固定的 `P00...P55` 毫米坐标。
   - 通过合格的 `^C T_T` 位姿把 36 个槽投影到当前图像。
   - 从旧 `tray_marker_layout.json` 读取每个槽内固定的 marker ID。
   - 多尺度识别 `DICT_4X4_50` marker，记录中心、四角、角度、面积和方形质量。
   - 把每个槽透视校正为统一的 `192 x 192` 小图。

2. `wafer_shape_quality.py`
   - 在透视校正后的单槽小图中检测紫色/深色高饱和度硅片。
   - 使用面积、边长、长宽比、矩形度、实心度、中心偏移、相对角度、轮廓顶点、连通块和内部边线判断正常、警告或异常。
   - 要求候选区域具有足够的紫色色彩比例，避免把黑白 ArUco marker 当成硅片。

3. `tray_occupancy.py`
   - 状态包括 `empty`、`empty_unread_marker`、`occupied`、`warning`、`abnormal`、`out_of_view`、`occluded` 和 `unknown`。
   - 画面不完整时标为 `out_of_view`，不会标为 missing。
   - 只有显式传入遮挡 mask 时才使用 `occluded`；新相机默认没有固定机械臂 mask。
   - 没有 marker、没有可靠硅片、也没有明确遮挡时标为 `unknown`，不猜测为空槽。

4. `tray_vision_fusion.py`
   - 先调用现有 `tray_pose_estimator.py`。
   - 位姿重投影质量门不通过时停止 36 槽分析和坐标转换。
   - 位姿通过后依次执行槽投影、slot marker 观测、硅片质量判断和槽状态融合。
   - 提供点击像素到托盘平面毫米坐标、最近槽位和距离的只读转换。
   - 当前不直接给出机械臂运动量；仍需经过现有吸盘目标和手眼标定层。

## 2. 固定槽位与 marker ID

旧 layout 的图像行列方向与当前 Tray Frame 的 `P00...P55` 方向不同。对 `260812015730` 中四张相机 1 图片比较了正方形网格的 8 种旋转/镜像关系，固定采用：

```text
metric P(row, col) -> legacy layout(col, 5 - row)
metric_slot_transform = rot270
```

在位姿质量通过的 `1_012.jpg` 和 `1_023.jpg` 上，该映射的 marker 中心误差中位数为 `12.47 px`、90 分位数为 `20.50 px`；其余映射的中位数均大于 `216 px`。该关系已写入 `tools/tray_marker_detector_v2/tray_marker_layout.json`。运行时只读取固定关系，不根据当前照片重新排序 ID。

## 3. 判断顺序

每个槽按以下顺序决策：

1. 槽投影在画面中的覆盖率不足 90%：`out_of_view`。
2. 显式遮挡 mask 覆盖率达到 25%：`occluded`。
3. 找到硅片候选：按形状质量给出 `occupied`、`warning` 或 `abnormal`。
4. 识别到该槽配置的 marker ID：`empty`。
5. 能看到黑白 marker 图案但 ID 未解码：`empty_unread_marker`。
6. 以上证据都不足：`unknown`。

## 4. 离线入口

```bash
cd /Users/chenge/Desktop/二维机器人/2DRobot
python3 tools/analyze_layered_tray.py \
  /Users/chenge/Desktop/二维机器人/260812015730/1_023.jpg \
  --output-json layered_tray_result.json \
  --output-image layered_tray_annotated.png
```

增加点击像素测试：

```bash
python3 tools/analyze_layered_tray.py IMAGE.jpg --click 640 360
```

输出 JSON 中会增加 `point_T_mm`、`nearest_slot` 和 `nearest_slot_distance_mm`。位姿质量门失败时 click 结果被拒绝。

## 5. 验证结果

新增测试：

```bash
python3 -m unittest tests.test_tray_marker_layered_integration -v
```

共 11 项，覆盖固定 ID 映射、36 槽投影、像素到 Tray 坐标往返、多尺度 marker、正常硅片角度、重叠硅片、二维码误识别、画面外状态、空槽/占用状态和位姿 fail-closed，全部通过。

真实图片结果：

| 图片 | 位姿结果 | 槽位结果 |
|---|---|---|
| `1_001.jpg` | 拒绝 | Marker 5 RMS `3.082 px > 3.000 px`，不分析槽位 |
| `1_012.jpg` | 通过，RMS `1.770 px` | 1 empty，1 abnormal，29 out_of_view，5 unknown |
| `1_023.jpg` | 通过，RMS `1.662 px` | 19 empty，14 out_of_view，3 unknown |
| `1_034.jpg` | 拒绝 | 只识别到 2 个外围 marker，不分析槽位 |

全仓库测试在当前系统 Python 下有 16 个导入错误，原因均为未安装 `PyQt6`；本次新增测试和不依赖 Qt 的原有视觉/数值测试通过。

## 6. 架构图

![Tray vision layered architecture](images/tray_vision_layered_architecture.png)

图中已删除“你的”。

## 7. 当前边界

- 本次只合并相机 1 的托盘总览逻辑。
- 相机 2 的吸盘和被提起硅片必须作为独立观测源，不能进入托盘占用判断。
- `robot_correction_allowed` 当前固定为 `false`。只有将点击点/目标槽的 Tray 坐标交给现有手眼标定和吸盘目标模块，并再次通过质量门后，才可计算机械臂修正量。
- `1_012.jpg` 中的大面积深色遮挡被标为 abnormal。后续应使用新相机的无遮挡数据重新标定硅片色彩范围，并为临时遮挡接入显式 mask 或独立遮挡分类器。
