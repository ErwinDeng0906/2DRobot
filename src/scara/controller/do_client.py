"""调用 SNRobotLab\\scara_do.exe 写用户输出 UO（方案 C）。

scara_do 内部：WriteUOEnable(通道,0) 保持用户模式 U，再 WriteUO(通道, 电平)。

电平约定（与现场接线有关）：
  默认：开泵 = UO 写 1，停泵 = UO 写 0。
  若你的泵是「0 开、1 关」，把下面 UO_LEVEL_ON / UO_LEVEL_OFF 对调即可。

界面「DO接口」经 set_uo_level / zero_channels 调用本模块；
写 DO 会另开控制器 TCP，与 snrobot serve 可能互斥。
UI 仅在机械臂已连接时允许手动改 DO；启动/断开/急停/退出的清零仍走本模块。

重要：scara_do.exe 必须和 RobotSDK.dll、许可证放在同一目录（SNRobotLab）。
不要从仓库 tools\\scara_do\\ 直接运行。路径由调用方传入 sdk_dir，
或由 SNROBOTLAB_DIR / local_config.toml 决定。
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Union

from scara.config.scara_config import resolve_snrobotlab_dir

UO_LEVEL_ON = 1
UO_LEVEL_OFF = 0

PathLike = Union[str, Path]


def sdk_dir(override: Optional[PathLike] = None) -> Path:
    if override is not None:
        p = Path(override)
        if str(p).strip():
            return p
    try:
        return resolve_snrobotlab_dir()
    except FileNotFoundError as exc:
        raise FileNotFoundError(str(exc)) from exc


def exe_path(override_sdk: Optional[PathLike] = None) -> Path:
    # 调用方已指定 SNRobotLab 目录时，强制用该目录下的 exe（避免环境变量指到 tools\\）
    if override_sdk is not None and str(override_sdk).strip():
        return Path(override_sdk) / "scara_do.exe"
    env = os.environ.get("SCARA_DO_EXE", "").strip()
    if env:
        return Path(env)
    return sdk_dir() / "scara_do.exe"


def _check_sdk_layout(exe: Path) -> Optional[str]:
    dll = exe.parent / "RobotSDK.dll"
    if dll.is_file():
        return None
    return (
        f"在 {exe.parent} 找不到 RobotSDK.dll。\n"
        f"scara_do.exe 不能放在仓库 tools\\ 目录单独运行，"
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


def set_uo_level(
    channel: int,
    level: int,
    *,
    timeout_s: float = 15.0,
    ip: Optional[str] = None,
    port: Optional[int] = None,
    sdk_dir: Optional[PathLike] = None,
) -> Tuple[bool, str]:
    """写用户输出 UO[channel]，电平 level 为 0 或 1。"""
    # 参数名 sdk_dir 会遮蔽模块函数，用局部变量
    root = sdk_dir
    try:
        exe = exe_path(root)
    except FileNotFoundError as exc:
        return False, str(exc)
    if not str(exe.parent).strip() or (root is not None and not str(root).strip()):
        return False, (
            "未配置 SNRobotLab 目录。请复制 local_config.example.toml 为 "
            "local_config.toml，并填写 [paths] snrobotlab_dir。"
        )
    if not exe.is_file():
        return False, (
            f"找不到 {exe}。\n"
            f"请把 tools\\scara_do\\scara_do.exe 复制到 SNRobotLab"
            f"（与 snrobot.exe 同目录），并在 local_config.toml 设置正确的 snrobotlab_dir。"
        )
    bad = _check_sdk_layout(exe)
    if bad:
        return False, bad

    if ip is None or port is None:
        ip, port = _active_ip_port(root)

    lv = 1 if int(level) else 0
    cmd = [
        str(exe), "set", str(int(channel)), str(lv),
        "--ip", str(ip), "--port", str(int(port)),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout_s,
            cwd=str(exe.parent),
        )
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT: scara_do 超过 {timeout_s:.0f}s"
    except OSError as exc:
        return False, f"启动失败: {exc}"

    out = (_decode_proc_text(proc.stdout or b"") + _decode_proc_text(proc.stderr or b"")).strip()
    if proc.returncode != 0:
        return False, out or f"exit={proc.returncode}"
    return True, out


def set_uo(
    channel: int,
    on: bool,
    *,
    timeout_s: float = 15.0,
    ip: Optional[str] = None,
    port: Optional[int] = None,
    sdk_dir: Optional[PathLike] = None,
) -> Tuple[bool, str]:
    level = UO_LEVEL_ON if on else UO_LEVEL_OFF
    return set_uo_level(
        channel, level, timeout_s=timeout_s, ip=ip, port=port, sdk_dir=sdk_dir,
    )


def zero_channels(
    channels,
    *,
    timeout_s: float = 10.0,
    ip: Optional[str] = None,
    port: Optional[int] = None,
    sdk_dir: Optional[PathLike] = None,
) -> Tuple[bool, str]:
    chs = sorted({int(c) for c in channels if 1 <= int(c) <= 16})
    if not chs:
        return True, "no channels"
    lines = []
    all_ok = True
    for ch in chs:
        ok, msg = set_uo_level(
            ch, 0, timeout_s=timeout_s, ip=ip, port=port, sdk_dir=sdk_dir,
        )
        all_ok = all_ok and ok
        lines.append(msg or f"DO[{ch}]=0 {'OK' if ok else 'FAIL'}")
        if not ok:
            break
    return all_ok, "\n".join(lines)
