# 机械臂控制程序（DUCO + SCARA）

从多设备控制系统中抽取的独立小程序，只含两个设备控制页：

- **机械臂控制**：DUCO GCR3-618 六轴协作臂（含 DH 夹爪）
- **SCARA 机械臂**：新松 SA4A-4/0.40（泵/阀走控制器 UO + scara_do.exe、末端相机）

## 运行环境

- Windows + Python 3.10
- 依赖：PyQt6、thrift、requests、pyyaml、opencv-python、numpy、pyserial、Pillow、tomli（Python 3.10）

## 启动

```bash
python main.py
```

启动后会打开一个两标签页窗口（「机械臂控制」「SCARA 机械臂」），自动拉起 armweb 机械臂代理（127.0.0.1:8080），并在代理就绪后尝试以 thrift 连接 DUCO 真机。自动连接不会自动使能机械臂；SCARA 仍需操作者在对应页面手动连接。

若 armweb 已由人工启动，可加 `--no-services` 跳过服务自启和启动阶段的 DUCO 后端自动连接：

```bash
python main.py --no-services
```

程序退出时会自动停掉由它启动的服务（外部已运行的服务不受影响）。服务日志在 `logs/` 目录。

## DUCO 真机连接

连接链路：本页面 → HTTP → armweb → thrift → DUCO 真机。`local_config.toml` 的 `[duco]` 统一设置真机 IP、RPC 端口和 armweb 地址，界面连接栏会以这些值初始化，连接前仍可手动覆盖。

1. 确认 armweb 已在运行（默认随 `main.py` 自启）。手工启动方式：

   ```bash
   python webconsole/server.py --port 8080 --host 127.0.0.1
   ```

2. 在「机械臂控制」页确认服务器、真机 IP 和端口后点连接；默认值来自 `local_config.toml` 的 `[duco]`。
3. **运动前须把操作模式切到自动**：在界面上切换，或 `POST http://127.0.0.1:8080/api/switch_mode`，Body `{"mode": 1}`。手动模式下运动命令会被拒绝。
4. armweb 自带浏览器控制台：打开 `http://127.0.0.1:8080/` 可查看状态/相机画面。

## SCARA 真机连接

连接链路：本页面 → 命令行桥 `snrobot.exe` → 控制器（工位配置里的 IP，常见 192.168.1.100:20002）。

**换电脑时路径怎么改**：复制 [`local_config.example.toml`](local_config.example.toml) 为 `local_config.toml`，改 `[paths] snrobotlab_dir`。详见 [`路径硬编码清单.md`](路径硬编码清单.md) / [`换机路径说明.md`](换机路径说明.md)。并把 `tools/scara_do/scara_do.exe`、`tools/scara_enable/scara_enable.exe` 拷进该目录（无需 C 编译器）。

1. 确认桥程序与许可证同目录（路径写在 `local_config.toml`）。该目录内含 RobotCommunication SDK 与许可证 `SiaSunRobot.lic`，**exe 必须与许可证同目录运行**，不能单独拷走。
2. 在「SCARA 机械臂」页点连接。连接参数由 `local_config.toml` 的 `[paths]` / `[scara]` 决定。
3. **DO 接口（泵/阀）**走 `scara_do.exe`（须与 `RobotSDK.dll`、许可证同目录，与 `snrobotlab_dir` 相同）。
   - 界面「DO接口」卡片：左列泵、右列阀；点「新建」绑定名称与 DO 号（1–16）。
   - **须先连接机械臂**后才能新建 / 改电平 / 删除（未连接时整块锁定）。原因：真机上可不经 `snrobot` 直接写 DO，界面禁止未连接时改，避免误开泵/阀。
   - 改 0/1 后须**按回车**才写入控制器（防误触）；`0`=低电平，`1`=高电平。
   - 映射保存在程序根目录 `scara_do_map.json`（格式见 `scara_do_map.example.json`），**不保存电平**。
   - **安全清零**：启动、点「连接」前、点「断开」后、急停、退出软件时，会把已配置的全部 DO **真正写成 0**（关泵/阀）。只改界面数字不算数；清零不受上述锁定影响。
   - 写 DO 会另开控制器连接，与机械臂 `snrobot` 连接可能互斥；已连接时手动写入偶发失败可接受（安全优先于测 DO 方便）。
   - 脉冲测试也可用 `tools/DOtest/test_scara_pump_pulse.py`。
4. 安全相关：急停会先停臂再清零 DO；页面底部急停按钮保持可用。

### SCARA 相机与动作拍照

- 相机使用 OpenCV + DirectShow，点击「连接相机」后按「源#」选择 USB 设备号；相机打不开时只在日志报错，不影响机械臂控制。
- 「快照」只保存最近 1 秒内采集到的新鲜画面，默认写到程序启动目录下的 `scara_snap_YYYYmmdd_HHMMSS.jpg`。
- 左栏「轨迹拍照」内的「导入动作」可加载 `ACTION_API_VERSION = 1` 的 Python 动作文件；步骤可组合关节目标、相对 XYZ/R、等待、源#0～#8 拍照和 JSON 途径点记录。
- `Tasks/task1.py` 使用 `scara_presets.json` 的 P00/P50/P55/P05 float 点和相机源 0/1/2。源0仅在初始 P00 拍1张；源1仅在 P00/P50/P55/P05 的 float 到达点各拍1张（共4张）；源2覆盖全部45个途径点。世界 Y+ 为 P05→P00；Rz 直接定义为相机方向相对世界 Y− 的角度，因此不需要另行输入相机基准角。
- 每次动作在 `Trajectory Photos/YYMMDDHHMMSS/` 创建独立实验目录；照片按 `源序号_三位全局途径点编号.jpg` 命名，同一点的不同相机共用编号，例如初始点为 `0_001.jpg`、`1_001.jpg`、`2_001.jpg`。`points.json` 保存每个采点的 J1～J4、机械中心 x/y/z/Rx/Ry/Rz、由 task1 计算的旋转相机 x/y/z 及照片索引；源1实际拍照的4个点另含 `camera1_position`，其 XY 按前臂方向 `J1+J2` 和J4轴外延33.55 mm计算。
- 执行动作前必须连接、使能且无急停/报警。运行时左侧手动控制和相机切换会锁定，「停止动作」及底部急停保持可用；任务相机会临时接管所需设备，结束后恢复原预览源。

### 相机1硅片识别与拾取导航

1. 连接相机源#1，打开「手眼交互 -> 转移视觉」。
2. 等待托盘位姿、`W←T` 和拾取稳定证据通过；拾取槽需最近 5 帧中至少 3 帧为正常占用，且最新帧仍正常。
3. 在画面中点击正常硅片，界面会持续显示点击像素、托盘毫米坐标、匹配槽位及吸盘 XY 偏差。
4. 「锁定拾取导航」只启动计算和跟踪；窗口始终输出 `robot_motion_authorized=false`，不会发送运动、Z、J4、DO或真空命令。
5. 相机超时或分析异常会清除旧图、多帧证据和 `W←T`；已锁定会话直接进入 `blocked`。

识别参数、标注图和回归命令见 [`docs/wafer_recognition_0820_validation.md`](docs/wafer_recognition_0820_validation.md)；实时坐标链和安全门见 [`docs/wafer_transfer_runtime_integration.md`](docs/wafer_transfer_runtime_integration.md)。

### 相机2到J4外参标定

- `Tasks/task18_camera2_extrinsic_capture.py` 只记录静止机器人状态并拍摄相机1/2，不发送运动、J3、J4、DO或真空指令。
- `tools/create_camera2_board_pose.py` 由人工测量的ChArUco板三个世界坐标生成 `T_W_B`。
- `tools/analyze_camera2_extrinsics.py` 离线求解完整 `T_J4_C2`；任一观测、姿态覆盖或残差质量门失败都会拒绝安装。
- 本流程不会将点击目标转为机器人命令，也不用单帧图像估计绝对Z。

现场操作、质量门和命令见 [`docs/camera2_j4_extrinsic_calibration.md`](docs/camera2_j4_extrinsic_calibration.md)；本分支实施范围见 [`docs/camera2_guided_transfer_implementation_log.md`](docs/camera2_guided_transfer_implementation_log.md)。

## 安全提示

- 首次运动务必低速（速度比例调小），急停按钮保持在手边。
- DUCO 碰撞报警后需在示教器上复位，软件无法清除。
- SCARA 运动前确认已使能、模式正确（T1 示教限速，T2 全速默认禁用）。
- 页面上的使能（伺服上电）会让机械臂带电，属于需谨慎确认的动作。
- 泵/阀靠 DO 电平保持：关软件不会自动掉电。本程序在启动/连接/断开/急停/退出时会主动把已配置 DO 写成 0；若清零失败，请看日志并用示教器或手动断电确认泵已停。

## 目录说明

```
main.py                    两设备页启动器；负责 armweb 自启、DUCO 后端连接尝试和退出清理
local_config.example.toml  本机路径/地址模板；复制为不提交的 local_config.toml
src/
  core/                    设备模块公共接口
  utils/                   日志和 armweb 本地服务生命周期管理
  design_system.py         全局色板、字号和 QSS 设计令牌
  robot_arm/               DUCO 设备模块、控制页、相机面板和序列 UI
  devices/robot_arm/       DUCO HTTP/thrift/模拟后端、RPC、夹爪、安全、点位与标定数据
  scara/
    calib/                 SCARA 标定加载器、托盘网格和标定示例 JSON
    config/                local_config.toml 解析及 SCARA/DUCO 本机配置入口
    controller/            snrobot 常驻桥、使能/急停/报警和 DO 客户端
    pipeline/              可注入硬件后端、运动规划和同步到位工具
    sequence/              送检/取放步骤定义与可回滚执行器
    ui/                    SCARA 深色控制页、USB 相机、动作执行器和旧轨迹兼容 worker
    vision/                硅片检测、托盘定位和视觉伺服算法
webconsole/
  server.py                armweb HTTP/MJPEG 代理（HTTP → thrift → DUCO 真机）
  index.html               armweb 浏览器控制台
tools/
  scara_do/                SCARA 泵/阀 UO C 工具、构建脚本和已编译 exe
  scara_enable/            SCARA 使能/急停/清报警 C 工具、构建脚本和已编译 exe
  snrobot bridge/          snrobot 桥及运行时文件的换机部署副本
  uitest/                  使能/模式/急停/报警 SDK 探测客户端与人工测试入口
  DOtest/                  DO 泵脉冲人工测试脚本
Tasks/                     可导入的 ACTION_API_VERSION=1 动作脚本
Trajectory Photos/         每次动作的照片和 points.json 输出（Git 忽略）
_smoke/                    不接真机的 Qt 页面、运动、相机、动作和 JSON 回归测试
logs/                      主程序与 armweb 运行日志（Git 忽略）
scara_presets.json         task1 使用的 P00/P05/P50/P55 四个 float 实教点
scara_do_map.example.json  泵/阀名称与 DO 通道映射格式示例
scara_do_map.json          当前机器的 DO 映射；只存通道关系，不保存实时电平
```

## 文档说明

- [`README.md`](README.md)：当前程序的功能、启动方法、真机连接、安全要求、目录结构和文档入口。
- [`Commit_Guide.md`](Commit_Guide.md)：面向 Git 初学者的分支、提交、推送、Pull Request、冲突处理和本机配置保护指南。
- [`MERGE_NOTES.md`](MERGE_NOTES.md)：0811/0812 合并来源、保留策略及当时的无硬件验证记录；属于历史合并说明，不代替当前 README。
- [`换机路径说明.md`](换机路径说明.md)：换电脑部署的最短操作步骤，重点说明 `local_config.toml` 和 SNRobotLab 文件复制位置。
- [`路径硬编码清单.md`](路径硬编码清单.md)：完整的本机路径、控制器地址、环境变量、C 工具检查和换机验收清单。
- [`tools/uitest/README.md`](tools/uitest/README.md)：SCARA 使能、去使能、模式、急停、解除急停和清报警命令的 SDK 含义及人工测试方法。

## 补充说明

- `local_config.toml` 是当前运行配置并已被 Git 忽略；旧 `scara_config.toml` 仅作为迁移备份保留，当前加载器不读取它。
- `src/devices/robot_arm/calib/joint_replay/`（自动流程专用的关节回放数据）未随本程序打包，两个页面本身不读它。
- `src/scara/pipeline/` 与 `src/scara/sequence/` 是可复用的送检/取放基础模块，目前没有接入 `main.py` 的 SCARA 控制页；其中涉及显微镜或外部工位的步骤仍需相应硬件适配器和预设文件。
- 相机画面为懒加载（首次打开时 import cv2），无相机不影响机械臂控制。
- 动作文件是会被 Python 导入执行的代码，不是受限数据格式；只导入来源可信的 `.py` 文件。当前 UI 使用 `action_worker.py` 执行动作，`photo_trajectory_worker.py` 仅保留旧轨迹兼容能力。
- `scara_do_map.json` 只存「有哪些口」，不存当前 0/1；每次进入界面电平均按 0 显示，并会尝试向硬件写一遍 0。
