# SCARA 使能 / 急停 / 清报警 SDK 探测

- C 工具：`tools/scara_enable/` → 编译到 `D:\SNRobotLab\scara_enable.exe`
- Python：`tools/uitest/enable_client.py` + `test_enable.py`

## 当前可用命令（已真机确认）

| 命令 | 含义 | DLL 根源 |
|------|------|----------|
| `status` | 读使能/急停/报警/模式 | `Read_PowerOn` / `Read_Stop` / `Read_Warning` / `ReadCurMode` |
| `enable` | 使能 | `Send_PowerOn` → `Reset_PowerOn`（脉冲） |
| `disable` | 去使能（不急停） | `SetMode` 来回切，最后回原模式 |
| `set_mode N` | 切模式 1=T1 2=T2 3=执行 4=远程 | `SetMode(N)` |
| `estop` | 急停 | `Send_Stop` |
| `release_estop` | 仅解除急停 | `Reset_Stop` |
| `clear_alarm` | 清报警（急停 OFF + 运行正常） | 先 `Reset_Stop` 等到 OFF，再 `Send_Reset`/`Reset_Reset` |
| `cycle` | 使能→去使能一轮 | `enable` + `disable` |

```powershell
python tools\uitest\test_enable.py status
python tools\uitest\test_enable.py cycle
python tools\uitest\test_enable.py estop
python tools\uitest\test_enable.py clear_alarm
```

## 已删除的错误/误导指令（及其真正意义）

| 旧测试命令 | 实际 DLL | 真正意义 / 为何删 |
|------------|----------|-------------------|
| `power_on` | `Send_PowerOn`（旧版常不 Reset） | 现已并入 `enable`，且必须成对松开 |
| `power_off` | `Reset_PowerOn` | **不是去使能**；只是松开「上电请求」 |
| `disable_mode` | `SetMode` 切换 | 已改名为 `disable` |
| `clear_alarm_ui` | 同现 `clear_alarm` | 已改名 |
| `stop` / `reset_stop` | `Send_Stop` / `Reset_Stop` | 已改名为 `estop` / `release_estop` |
| `clearalarm` | 仅 `ClearAlarm` | **不能**单独清官方「运行状态报警」 |
| `reset_robot` | 仅 `Send_Reset` | 报警复位「按下」；需配 `Reset_Reset` 松开 |
| `reset_reset` | 仅 `Reset_Reset` | 报警复位「松开」；不单独作产品命令 |
| `deadman_on` / `deadman_off` | `SendDeadManEnableStatus` | 与官方「使能」无关；读数常恒为 ON，写常 -48 |
| `disable_try` | remote+deadman+setmode | 探测组合，无效/多余 |
| `remote` / `ensure_t1` | `SetControlPriorty` / `SetMode(1)` | 非日常命令；`enable` 内部必要时会先切 T1 |
| snrobot `enable`/`disable` | 内部 `SetEnableStatus` | DLL **无此导出**；去使能常 -48 |
| snrobot `clearalarm` | `ClearAlarm` | 与官方清报警行为不一致 |

## 状态显示对照

| 输出 | 官方软件 |
|------|----------|
| `ENABLE_ON` / `ENABLE_OFF` | 使能 |
| `ESTOP_ON` / `ESTOP_OFF` | 急停 |
| `ALARM_ON` / `ALARM_OFF` | 运行状态：报警 / 正常 |
| `T1` / `T2` / `Execute` | 操作模式 |
