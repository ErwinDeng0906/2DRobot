# Per-slot local Jacobians

本目录保存除P22以外、由Task9实测并通过全部质量门的单槽局部XY图像Jacobian。

命名固定为：

```text
camera1_xy_image_jacobian_P00.json
...
camera1_xy_image_jacobian_P55.json
```

P22的既有正式文件暂时保留在上一级：

```text
src/scara/calib/camera1_xy_image_jacobian.json
```

每个JSON的 `anchor_target_name` 和 `valid_target_names` 必须与文件名槽号一致。文件只在
Task9采集完整、逐帧处理无fatal、拟合与独立验证质量门全部通过后原子安装。不要手工复制、
改名或将一个槽的Jacobian用于另一槽。
