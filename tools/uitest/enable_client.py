"""调用 D:\\SNRobotLab\\scara_enable.exe（直调 RobotSDK.dll）。

命令名对应官方 UI 含义；底层 DLL 函数见各函数注释。
前提：关掉官方 SCARA GUI 和本程序 snrobot serve（会抢连接）。
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

_DEFAULT_SDK = Path(os.environ.get("SNROBOTLAB_DIR", r"D:\SNRobotLab"))


def sdk_dir() -> Path:
    # 独立测试脚本：优先环境变量；跑 main.py 时请改用根目录 scara_config.toml（见 换机路径说明.md）
    return Path(os.environ.get("SNROBOTLAB_DIR", str(_DEFAULT_SDK)))


def exe_path() -> Path:
    env = os.environ.get("SCARA_ENABLE_EXE")
    if env:
        return Path(env)
    return sdk_dir() / "scara_enable.exe"


def _active_ip_port() -> Tuple[str, int]:
    cfg = sdk_dir() / "WorkStationConfig.json"
    rows = json.loads(cfg.read_text(encoding="utf-8"))
    for row in rows:
        if row.get("Active"):
            return str(row["IP"]), int(row["Port"])
    raise RuntimeError(f"{cfg} 没有 Active 工位")


def run_cmd(
    *args: str,
    timeout_s: float = 20.0,
    ip: Optional[str] = None,
    port: Optional[int] = None,
) -> Tuple[bool, str]:
    """跑 scara_enable.exe；成功返回 (True, 输出)，失败 (False, 输出/错误)。"""
    exe = exe_path()
    if not exe.is_file():
        return False, (
            f"找不到 {exe}。请先运行 tools\\scara_enable\\build.bat "
            f"（会编译并复制到 {sdk_dir()}）"
        )
    if ip is None or port is None:
        ip, port = _active_ip_port()
    cmd = [str(exe), *args, "--ip", str(ip), "--port", str(int(port))]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout_s,
            cwd=str(exe.parent),
        )
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT: scara_enable 超过 {timeout_s:.0f}s"
    except OSError as exc:
        return False, f"启动失败: {exc}"
    raw = (proc.stdout or b"") + (proc.stderr or b"")
    out = raw.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        return False, out or f"exit={proc.returncode}"
    return True, out


def status(**kw) -> Tuple[bool, str]:
    """读当前状态：使能 / 急停 / 报警 / 操作模式。

    DLL: Read_PowerOn, Read_Stop, Read_Warning, ReadCurMode
    """
    return run_cmd("status", **kw)


def enable(**kw) -> Tuple[bool, str]:
    """使能（官方 UI「使能」变 ON）。

    DLL: 必要时先 SetMode(1=T1)，再
         Send_PowerOn（按下上电请求）→ Reset_PowerOn（松开）。
    注意：必须成对脉冲；只 Send 不 Reset 会在清报警后自动使能。
    """
    return run_cmd("enable", **kw)


def disable(**kw) -> Tuple[bool, str]:
    """去使能（官方 UI「使能」变 OFF），不触发急停/报警。

    DLL: SetMode 来回切一次，最后回到原模式：
         - 当前 T1/T2 → 切到执行 → 切回原 T1/T2
         - 当前执行   → 切到 T1  → 切回执行
    """
    return run_cmd("disable", **kw)


def set_mode(mode: int, **kw) -> Tuple[bool, str]:
    """切换操作模式。

    mode: 1=T1示教, 2=T2示教, 3=执行, 4=远程
    DLL: SetMode(mode)
    """
    return run_cmd("set_mode", str(int(mode)), **kw)


def estop(**kw) -> Tuple[bool, str]:
    """急停：急停 ON，并通常伴随报警。

    DLL: Send_Stop
    """
    return run_cmd("estop", **kw)


def release_estop(**kw) -> Tuple[bool, str]:
    """仅解除急停（急停 OFF）。不保证报警也清掉。

    DLL: Reset_Stop
    完整「清报警」请用 clear_alarm。
    """
    return run_cmd("release_estop", **kw)


def clear_alarm(**kw) -> Tuple[bool, str]:
    """清报警：急停变 OFF + 运行状态变正常（对齐官方清报警）。

    顺序:
      1) 若急停 ON → Reset_Stop，等到 Read_Stop=OFF
      2) Send_Reset → Reset_Reset（报警复位脉冲；必要时两拍）
      3) 仍报警时再试 ClearAlarm + 一拍
    注意：急停仍 ON 时，报警无法变回正常。
    """
    return run_cmd("clear_alarm", **kw)
