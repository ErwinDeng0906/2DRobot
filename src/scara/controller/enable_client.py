"""调用 SNRobotLab\\scara_enable.exe（直调 RobotSDK.dll）。

与 do_client / scara_do 同模式：短生命周期进程，不经 snrobot serve。
调用前须释放控制器连接（ScaraController 会临时断开 serve）。

状态字段对应官方 UI：
  Read_PowerOn  → enable  (ENABLE_ON/OFF)
  Read_Stop     → estop   (ESTOP_ON/OFF)
  Read_Warning  → warn    (ALARM_ON/OFF)
  ReadCurMode   → mode

重要：scara_enable.exe 必须和 RobotSDK.dll 同目录。不要从 tools\\scara_enable\\ 直接跑。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from scara.config.scara_config import resolve_snrobotlab_dir

_MODE_FROM_CODE = {1: "T1", 2: "T2", 3: "Execute", 4: "Remote"}

PathLike = Union[str, Path]


def sdk_dir(override: Optional[PathLike] = None) -> Path:
    if override is not None:
        return Path(override)
    return resolve_snrobotlab_dir()


def exe_path(override_sdk: Optional[PathLike] = None) -> Path:
    if override_sdk is not None:
        return Path(override_sdk) / "scara_enable.exe"
    env = os.environ.get("SCARA_ENABLE_EXE", "").strip()
    if env:
        return Path(env)
    return sdk_dir() / "scara_enable.exe"


def _check_sdk_layout(exe: Path) -> Optional[str]:
    dll = exe.parent / "RobotSDK.dll"
    if dll.is_file():
        return None
    return (
        f"在 {exe.parent} 找不到 RobotSDK.dll。\n"
        f"scara_enable.exe 不能放在仓库 tools\\ 目录单独运行，"
        f"必须复制到与 snrobot.exe、RobotSDK.dll 相同的 SNRobotLab 目录。"
    )


def _active_ip_port(override_sdk: Optional[PathLike] = None) -> Tuple[str, int]:
    from scara.config.scara_config import load_scara_config

    root = sdk_dir(override_sdk)
    cfg = root / "WorkStationConfig.json"
    if cfg.is_file():
        rows = json.loads(cfg.read_text(encoding="utf-8"))
        for row in rows:
            if row.get("Active"):
                return str(row["IP"]), int(row["Port"])
        raise RuntimeError(f"{cfg} 没有 Active 工位")

    sc = load_scara_config()
    return str(sc.controller_ip), int(sc.controller_port)


def _decode_proc_text(raw: bytes) -> str:
    if not raw:
        return ""
    for enc in ("utf-8", "gbk", "cp936"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_status(text: str) -> Dict[str, Any]:
    """从 scara_enable 输出解析安全状态（取文中最后一次出现）。"""
    enable = 0
    estop = False
    warn = 0
    mode = "?"
    for line in text.splitlines():
        if "ENABLE_ON" in line:
            enable = 1
        elif "ENABLE_OFF" in line:
            enable = 0
        if "ESTOP_ON" in line:
            estop = True
        elif "ESTOP_OFF" in line:
            estop = False
        if "ALARM_ON" in line:
            warn = 1
        elif "ALARM_OFF" in line:
            warn = 0
        if "ReadCurMode" in line:
            m = re.search(r"mode\s*=\s*(\d+)", line)
            if m:
                mode = _MODE_FROM_CODE.get(int(m.group(1)), f"?{m.group(1)}")
            else:
                for name in ("T1", "T2", "Execute", "Remote"):
                    if f"({name})" in line:
                        mode = name
                        break
    return {
        "enable": enable,
        "estop": bool(estop),
        "warn": int(warn),
        "mode": mode,
    }


def run_cmd(
    *args: str,
    timeout_s: float = 25.0,
    ip: Optional[str] = None,
    port: Optional[int] = None,
    sdk_dir: Optional[PathLike] = None,
) -> Tuple[bool, str]:
    """跑 scara_enable.exe；成功返回 (True, 输出)，失败 (False, 输出/错误)。"""
    root = sdk_dir
    exe = exe_path(root)
    if not exe.is_file():
        return False, (
            f"找不到 {exe}。\n"
            f"请把 tools\\scara_enable\\scara_enable.exe 复制到 SNRobotLab"
            f"（与 snrobot.exe 同目录），并在 scara_config.toml 设置正确的 exe_dir。"
        )
    bad = _check_sdk_layout(exe)
    if bad:
        return False, bad
    if ip is None or port is None:
        ip, port = _active_ip_port(root)
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
    out = (_decode_proc_text(proc.stdout or b"") + _decode_proc_text(proc.stderr or b"")).strip()
    if proc.returncode != 0:
        return False, out or f"exit={proc.returncode}"
    return True, out


def status(**kw) -> Tuple[bool, str]:
    return run_cmd("status", **kw)


def enable(**kw) -> Tuple[bool, str]:
    return run_cmd("enable", **kw)


def disable(**kw) -> Tuple[bool, str]:
    return run_cmd("disable", **kw)


def set_mode(mode: int, **kw) -> Tuple[bool, str]:
    return run_cmd("set_mode", str(int(mode)), **kw)


def estop(**kw) -> Tuple[bool, str]:
    return run_cmd("estop", **kw)


def clear_alarm(**kw) -> Tuple[bool, str]:
    return run_cmd("clear_alarm", **kw)
