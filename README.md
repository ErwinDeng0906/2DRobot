# 机械臂控制程序（DUCO + SCARA）

从多设备控制系统中抽取的独立小程序，只含两个设备控制页：

- **机械臂控制**：DUCO GCR3-618 六轴协作臂（含 DH 夹爪）
- **SCARA 机械臂**：新松 SA4A-4/0.40（泵/阀走控制器 UO + scara_do.exe、末端相机）

## 运行环境

- Windows + Python 3.10
- 依赖：PyQt6、thrift、requests、pyyaml、opencv-python、numpy、pyserial、Pillow

## 启动

```bash
python main.py
```

启动后会打开一个两标签页窗口（「机械臂控制」「SCARA 机械臂」），并自动拉起 armweb 机械臂代理（127.0.0.1:8080）。

若这两个服务已由人工启动，可加 `--no-services` 跳过自启：

```bash
python main.py --no-services
```

程序退出时会自动停掉由它启动的服务（外部已运行的服务不受影响）。服务日志在 `logs/` 目录。

## DUCO 真机连接

连接链路：本页面 → HTTP → 本机 armweb（127.0.0.1:8080）→ thrift → 真机 192.168.1.10:7003。

1. 确认 armweb 已在运行（默认随 `main.py` 自启）。手工启动方式：

   ```bash
   python webconsole/server.py --port 8080 --host 127.0.0.1
   ```

2. 在「机械臂控制」页点连接，服务器地址填 `http://127.0.0.1:8080`（默认值），由 armweb 侧连真机 192.168.1.10:7003。
3. **运动前须把操作模式切到自动**：在界面上切换，或 `POST http://127.0.0.1:8080/api/switch_mode`，Body `{"mode": 1}`。手动模式下运动命令会被拒绝。
4. armweb 自带浏览器控制台：打开 `http://127.0.0.1:8080/` 可查看状态/相机画面。

## SCARA 真机连接

连接链路：本页面 → 命令行桥 `snrobot.exe` → 控制器（工位配置里的 IP，常见 192.168.1.100:20002）。

**换电脑时路径怎么改**：见根目录 [`换机路径说明.md`](换机路径说明.md)。通常只需复制 `scara_config.example.toml` 为 `scara_config.toml`，改 `exe_dir` 为对方的 SNRobotLab 目录，并把 `tools/scara_do/scara_do.exe`、`tools/scara_enable/scara_enable.exe` 拷进该目录（无需 C 编译器）。

1. 确认桥程序与许可证同目录（例如 `D:\SNRobotLab\snrobot.exe` 或你机器上的实际路径）。该目录内含 RobotCommunication SDK 与许可证 `SiaSunRobot.lic`，**exe 必须与许可证同目录运行**，不能单独拷走。
2. 在「SCARA 机械臂」页点连接。连接参数（桥路径、控制器地址、超时等）用根目录 `scara_config.toml` 覆盖。
3. **DO 接口（泵/阀）**走 `scara_do.exe`（须与 `RobotSDK.dll`、许可证同目录，与上面 `exe_dir` 相同）。
   - 界面「DO接口」卡片：左列泵、右列阀；点「新建」绑定名称与 DO 号（1–16）。
   - **须先连接机械臂**后才能新建 / 改电平 / 删除（未连接时整块锁定）。原因：真机上可不经 `snrobot` 直接写 DO，界面禁止未连接时改，避免误开泵/阀。
   - 改 0/1 后须**按回车**才写入控制器（防误触）；`0`=低电平，`1`=高电平。
   - 映射保存在程序根目录 `scara_do_map.json`（格式见 `scara_do_map.example.json`），**不保存电平**。
   - **安全清零**：启动、点「连接」前、点「断开」后、急停、退出软件时，会把已配置的全部 DO **真正写成 0**（关泵/阀）。只改界面数字不算数；清零不受上述锁定影响。
   - 写 DO 会另开控制器连接，与机械臂 `snrobot` 连接可能互斥；已连接时手动写入偶发失败可接受（安全优先于测 DO 方便）。
   - 脉冲测试也可用 `tools/DOtest/test_scara_pump_pulse.py`。
4. 安全相关：急停会先停臂再清零 DO；页面底部急停按钮保持可用。

### SCARA 相机与轨迹拍照

- 相机使用 OpenCV + DirectShow，点击「连接相机」后按「源#」选择 USB 设备号；相机打不开时只在日志报错，不影响机械臂控制。
- 「快照」只保存最近 1 秒内采集到的新鲜画面，默认写到程序启动目录下的 `scara_snap_YYYYmmdd_HHMMSS.jpg`。
- 左栏「轨迹拍照」可导入 `TRAJECTORY_API_VERSION = 1` 的 Python 轨迹文件。合并包自带 `Preset Trajectories/` 示例和对应 `scara_presets.json`。
- 自动拍照固定使用「源#1」：每个点均先逐轴运动并回读验证到位，等待 2 秒后保存照片，再等待 2 秒进入下一点；输出位于 `Trajectory Photos/时间戳/`。
- 执行轨迹拍照前必须连接、使能且无急停/报警。任务运行时左侧运动控制和相机切换会锁定，底部物理/软件急停仍须保持可用。

## 安全提示

- 首次运动务必低速（速度比例调小），急停按钮保持在手边。
- DUCO 碰撞报警后需在示教器上复位，软件无法清除。
- SCARA 运动前确认已使能、模式正确（T1 示教限速，T2 全速默认禁用）。
- 页面上的使能（伺服上电）会让机械臂带电，属于需谨慎确认的动作。
- 泵/阀靠 DO 电平保持：关软件不会自动掉电。本程序在启动/连接/断开/急停/退出时会主动把已配置 DO 写成 0；若清零失败，请看日志并用示教器或手动断电确认泵已停。

## 目录说明

```
main.py                 启动器（两标签页 + 本地服务自启）
src/
  core/                 模块接口定义
  utils/                日志、本地服务管理
  design_system.py      全局设计令牌（色板/字号/QSS）
  robot_arm/            DUCO 控制页（UI）
  devices/robot_arm/    DUCO 后端（http/thrift/sim、RPC、夹爪、安全、点位）
  scara/                SCARA 模块（UI/控制器/标定/视觉/取放序列）
webconsole/
  server.py             armweb 机械臂代理（HTTP → thrift → 真机）
  index.html            armweb 浏览器控制台页面
tools/scara_do/         SCARA 泵/阀 UO 控制小工具（编译为 scara_do.exe）
tools/DOtest/           DO 脉冲测试脚本
Preset Trajectories/    可导入的 SCARA 拍照轨迹示例
Trajectory Photos/      轨迹拍照输出（含旧项目迁入的示例照片）
scara_presets.json      示例轨迹所使用的四角实教点
scara_do_map.json       运行时生成：泵/阀名称与 DO 号映射（不入库也无妨）
scara_do_map.example.json  DO 映射格式示例
```

说明：

- `src/devices/robot_arm/calib/joint_replay/`（自动流程专用的关节回放数据）未随本程序打包，两个页面本身不读它。
- SCARA 送检取放序列中的"切换物镜倍率"步骤依赖显微镜预设文件（`presets/*.json`），本程序不含显微镜模块，该步骤会跳过并报缺文件提示，其余步骤不受影响。
- 相机画面为懒加载（首次打开时 import cv2），无相机不影响机械臂控制。
- `scara_do_map.json` 只存「有哪些口」，不存当前 0/1；每次进入界面电平均按 0 显示，并会尝试向硬件写一遍 0。
- 合并来源、取舍和验证结果见 [`MERGE_NOTES.md`](MERGE_NOTES.md)。
