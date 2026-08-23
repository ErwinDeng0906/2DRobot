# 相机2无 Marker 近距槽边缘识别计划

日期：2026-08-18
开发分支：`agent/integrate-tray-marker-vision`

## 1. 目标和边界

1. 相机1负责全局托盘位姿、36槽状态、目标锁定和粗定位。
2. 相机2只在接近已锁定槽位后工作，此时允许画面中没有任何 ArUco marker。
3. 相机2不重新选槽，只验证已锁定的 `target_slot`，防止近距画面中错槽。
4. 第一版只输出观测和修正建议，不直接调用机械臂、J3、J4、DO或真空。

## 2. 标定方式

1. 使用相机2实际工作分辨率标定内参和畸变，运行时不允许切换分辨率。
2. 吸盘中心使用固定 ROI 和标定像素点；若吸盘在画面中有可测微小移动，再对局部边缘做二次精修。
3. 相机2高度随 J3 变化，不从单目图像估计深度。先选定一个固定近距工作高度，建立一个
   `calibration_profile`和很窄的 `valid_j3_range_mm`。
4. 若后续需要多高度，每个 J3 高度独立采集数据、标定像素/mm和局部图像 Jacobian。两个
   profile 之间只在实验证明可以时插值，否则落在空白范围内直接拒绝。

## 3. 数据采集和标注

1. 每个 J3 profile 至少采集：空槽、正常放置、XY偏移、旋转、卡在槽边、叠片、吸盘遮挡、
   强反光、弱光、模糊和槽边不完整。
2. 保存每帧的原图、单调时间戳、相机序号、分辨率、J1-J4、机械臂位姿、目标槽号和当前动作。
3. 人工标注槽的四个内角、硅片四角、吸盘中心、硅片是否仍吸附，以及 `insertable / unsafe /
   ambiguous`。
4. 数据按拍摄批次划分训练/调参集和完全独立的验证集，不把同一次连拍的相邻帧分到两边。

## 4. 第一版检测管线

1. 根据当前 J3 选择唯一标定 profile；无匹配 profile 时返回 `unavailable`。
2. 使用相机1锁定的槽位和机械臂已执行的粗移动预测相机2搜索 ROI，不在整张图中随意找槽。
3. 对 ROI 做畸变校正、亮度归一化和高光抑制，同时保留原图以便回放。
4. 使用 OpenCV LSD/边缘检测得到线段，按方向聚成两组对边，再用 RANSAC 或稳健直线拟合得到槽的四条内边。
5. 组合四边交点形成候选凸四边形，检查边长比、对边平行度、四角角度、边缘覆盖率、拟合 RMS 和时序连续性。
6. 独立检测硅片的四条直边，不把槽边和硅片边混为同一轮廓。硅片边不完整时可使用对边约束补全，
   但必须上调不确定度。
7. 将硅片四角代入槽的四条内边半平面，计算每个硅片角到每条槽边的带符号距离。最小距离减去标定和拟合
   不确定度后仍大于安全间隙，才能输出 `insertable`。
8. 多个四边形得分接近、任何必需边缺失、形状不合理或时序跳变时，返回 `rejected`，不给修正量。

## 5. 输出和运行时门

检测器实现 `src/scara/vision/close_range_slot_observation.py` 中的
`CloseRangeSlotObserver`，输出 `CloseRangeSlotObservation`。肯定结果必须提供：

1. 槽中心和四角、硅片中心和四角，拾取阶段还要有吸盘中心。
2. 中心误差、角度误差、边缘拟合 RMS 和放置阶段的最小四边间隙。
3. 每个数值对应的限值：`maximum_center_error_px`、`maximum_angle_error_deg`、
   `maximum_edge_fit_rms_px`和`minimum_required_clearance_px`。
4. 图像时间戳、同步 J3、`calibration_profile`和`valid_j3_range_mm`。
5. 放置前还必须确认硅片被提起且仍附着在吸盘上。

`WaferTransferSession` 会独立复核数值限值、帧鲜度、相机/机械臂时间差和 J3 一致性。

## 6. 建议实现文件

1. `src/scara/vision/camera2_slot_edge_observer.py`：槽边缘、硅片和吸盘检测器。
2. `src/scara/calib/camera2_close_range_profiles.json`：内参 hash、分辨率、J3范围、像素/mm、阈值和固定 ROI。
3. `tools/analyze_camera2_slot_edges.py`：离线批处理、标注图和 JSON 输出。
4. `tests/test_camera2_slot_edge_observer.py`：固定回归集和失败条件。
5. `src/scara/ui/wafer_transfer_dialog.py`：相机2观测器通过后，再增加近距画面和误差显示，不改目标锁定逻辑。

## 7. 验收标准

1. 槽四角和硅片四角的像素误差分别统计中位数、95分位数和最大值。
2. `insertable`的假阳性必须单独统计，且优先约束为0；不确定样本应当进入 `rejected`。
3. 在独立批次、所有标定 J3 profile、强反光和边缘不完整子集上分别报告结果。
4. 连续回放中不得在相邻帧间切换到其他槽；丢失目标后必须等待相机1重新锁定。
5. 在完成离线回放和低速度无吸片干跑之前，`robot_motion_authorized` 保持 `false`。
