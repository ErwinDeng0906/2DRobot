# 硅片识别全量数据审计

日期：2026-08-21
分支：`agent/integrate-tray-marker-vision`

## 1. 审计范围

本次审计覆盖以下两个旧实验目录中的全部原始图片：

- `/Users/chenge/Desktop/二维机器人/images`
- `/Users/chenge/Desktop/二维机器人/images 0805`

共 20 组静态托盘、363 张图片。每张图片在最终标注总览中恰好出现一次，
总览索引为 `validation/wafer_dataset/contact_sheets_final_annotated/index.json`。
20 张最终总览已逐格复核。

人工真值保存在 `validation/wafer_dataset/ground_truth.json`。只记录能由照片、
实验文件夹说明或操作者明确说明证明的槽位；未标注槽位不反推为正常或异常。

## 2. 最终判断原则

1. 先用固定的槽内 marker ID 和外围 marker ID 建立托盘平面单应性。
2. marker 必须同时覆盖托盘 X、Y 两个方向，并通过重投影残差检查。
3. 两个外围 marker 只有在均有四角、同时跨两轴且相距足够远时，才允许进入
   低置信度复核；同边双 marker 仍拒绝。
4. 每个槽透视归一化为固定大小后，提取硅片主体，计算中心、四角、边长、角度、
   正方形度、矩形度、实心度、连通块、内部线和到槽边的最小余量。
5. 槽外必须有可靠四角越过槽边；边缘裁切、外推或轮廓不完整时输出 `uncertain`。
6. 叠片确认需要独立第二四边形或跨多帧重复出现的放大方形包络。反光线不能单独
   确认叠片；轮廓异常但证据不足时输出“层数不确定”，不再默认写成单片。
7. 任意分辨率旧图只用于离线复核，始终输出
   `coordinate_mapping_allowed=false` 和 `robot_motion_authorized=false`。

## 3. 全量结果

- 纳入图片：363/363。
- 静态托盘组：20/20。
- 几何拟合通过：339 张。
- 几何拒绝：24 张。
- 显式真值中的危险错误：0。
- 已知叠片被确定写成单片：0。
- 0820 新样本：上半区 8 个正常、下半区 8 个槽外，原图和 75% 缩放图集合一致。
- `8-36 flake 2-stacked 6-crooked`：操作者指定的 `P25` 为槽内，`P35` 为槽外。
- `9-36 flake 6-stacked`：`P43`、`P44`、`P53` 的序列级结论均为确认叠片。
- `5-36 flake normal`：36 个槽的序列级结论均为槽内单片。
- `10-36 flake 6-crooked`：没有普通放歪单片被确认或怀疑为叠片。

比较报告中的 `correct` 表示当前帧与显式真值一致；`fail_closed` 表示当前帧证据
不足而返回未知、警告或不可见。后者不是错误的确定结论，也不能用于自动运动。

## 4. 逐组复核摘要

| 组别 | 图片 | 拒绝 | 序列级摘要 |
|---|---:|---:|---|
| `1-0 flake` | 2 | 0 | 空托盘；未出现 marker 假硅片 |
| `2-1 flake normal` | 19 | 0 | 1 个槽内硅片 |
| `3-2 flake normal` | 19 | 1 | 2 个槽内硅片 |
| `4-2 flake tilted` | 19 | 0 | 1 槽内、1 槽外 |
| `5-3 flake normal` | 19 | 0 | 3 个槽内硅片 |
| `6-3 flake tilted` | 19 | 0 | 1 槽内、2 槽外 |
| `7-4 flake tilted` | 19 | 0 | 3 槽内、1 槽外 |
| `8-4 flake stacked` | 19 | 0 | 2 槽内、1 槽外；层数不足处保留不确定 |
| `9-5 flake normal` | 19 | 0 | 5 个槽内硅片 |
| `10-5 flake tilted` | 19 | 0 | 4 槽内、1 槽外 |
| `1-5 flake stacked` | 19 | 1 | 1 个序列级确认叠片 |
| `2-5 flake stacked 2` | 19 | 0 | 1 个疑似叠片；其余证据不足不强判 |
| `3-6 flake stacked 2` | 19 | 0 | 1 个确认、1 个疑似叠片 |
| `4-10 flake stacked 3` | 19 | 0 | `P11` 等重复放大包络得到序列级确认 |
| `5-36 flake normal` | 19 | 4 | 36 槽均为序列级槽内单片 |
| `6-36 flake 2-stacked 1-crooked` | 19 | 4 | 30 槽内、1 槽外，1 个确认叠片 |
| `7-36 flake 2-stacked 4-crooked` | 19 | 11 | 双外围 marker 新恢复 3 帧；其余稀疏视角拒绝 |
| `8-36 flake 2-stacked 6-crooked` | 19 | 2 | `P25` 槽内、`P35` 槽外；2 个序列级确认叠片 |
| `9-36 flake 6-stacked` | 19 | 0 | `P43/P44/P53` 序列级确认叠片 |
| `10-36 flake 6-crooked` | 19 | 1 | 纯放歪序列，无叠片误报 |

## 5. 24 张拒绝帧

拒绝原因均为可解释的几何不足：

- 5 张只识别到两个固定 marker，且不是跨两轴的外围组合；其中包括同边双 marker。
- 15 张 marker 在一个方向上的跨度不足 60 mm。
- 3 张重投影残差或跨 marker 一致性不合格。
- 1 张独立对应点不足。

这些帧不能从单张照片唯一、稳定地恢复二维托盘坐标。保留拒绝比利用同一排 marker
外推整个托盘更可靠。可通过相邻合格帧、固定相机标定或增加可见外围 marker 解决，
不能通过放宽残差阈值解决。

## 6. 输出文件

- 人工真值：`validation/wafer_dataset/ground_truth.json`
- 四批逐帧结果：`validation/wafer_dataset/final_batch_01.json` 至
  `validation/wafer_dataset/final_batch_04.json`
- 真值比较：`validation/wafer_dataset/final_truth_comparison.json`
- 每图审计表：`validation/wafer_dataset/final_per_image_audit.csv`
- 最终标注总览：`validation/wafer_dataset/contact_sheets_final_annotated/`

## 7. 复现命令

```bash
cd /Users/chenge/Desktop/二维机器人/2DRobot

PYTHONPATH=src python3 -m unittest tests.test_marker_grid_wafer_review -v

python3 tools/compare_wafer_dataset_to_ground_truth.py \
  validation/wafer_dataset/final_batch_01.json \
  validation/wafer_dataset/final_batch_02.json \
  validation/wafer_dataset/final_batch_03.json \
  validation/wafer_dataset/final_batch_04.json \
  --truth validation/wafer_dataset/ground_truth.json \
  --output validation/wafer_dataset/final_truth_comparison.json \
  --csv validation/wafer_dataset/final_per_image_audit.csv
```

## 8. 结论边界

本次工作保证全部 363 张图片均被纳入、执行并人工查看，不保证每张照片都能产生
确定结论。24 张照片缺少足够二维几何，另有局部槽被裁切、反光或遮挡；这些情况
必须返回拒绝或不确定。当前结果适合离线识别回归和 UI 复核，不构成机械臂运动授权。
