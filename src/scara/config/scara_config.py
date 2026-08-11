"""
本机路径/地址配置 —— 全局只改根目录 ``local_config.toml``。

优先级（高 → 低）：
  1. 环境变量 SNROBOTLAB_DIR（临时覆盖，一般不用）
  2. local_config.toml
  3. 若都没有：报错提示复制 local_config.example.toml
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib as _toml
except ModuleNotFoundError:
    import tomli as _toml  # type: ignore


DEFAULT_EXE_NAME = "snrobot.exe"
_CONFIG_NAME = "local_config.toml"


def project_root() -> Path:
    """仓库根目录（含 main.py / local_config.toml）。"""
    return Path(__file__).resolve().parents[3]


def find_config_file(explicit: Optional[str] = None) -> Optional[Path]:
    """查找 local_config.toml；显式路径优先，否则在仓库根目录 / 当前目录找。"""
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    for base in (project_root(), Path.cwd()):
        cand = base / _CONFIG_NAME
        if cand.is_file():
            return cand
    return None


def _load_toml(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return _toml.load(f)


def resolve_snrobotlab_dir(toml_path: Optional[str] = None) -> Path:
    """解析 SNRobotLab 目录（snrobot / scara_do / scara_enable 共用）。"""
    env = os.environ.get("SNROBOTLAB_DIR", "").strip()
    if env:
        return Path(env)

    cfg_file = find_config_file(toml_path)
    if cfg_file is not None:
        try:
            paths = (_load_toml(cfg_file).get("paths") or {})
            raw = str(paths.get("snrobotlab_dir") or "").strip()
            if raw:
                return Path(raw)
        except Exception:
            pass

    raise FileNotFoundError(
        "未配置 SNRobotLab 目录。\n"
        "请复制 local_config.example.toml 为 local_config.toml，"
        "修改 [paths] snrobotlab_dir 为你电脑上 snrobot.exe 所在目录。\n"
        "详见 路径硬编码清单.md"
    )


@dataclass
class ScaraConfig:
    """SCARA 模块配置。"""

    exe_dir: str = ""
    exe_name: str = DEFAULT_EXE_NAME
    controller_ip: str = "192.168.1.100"
    controller_port: int = 20002
    command_timeout_s: float = 8.0
    poll_interval_ms: int = 300

    default_speed_percent: int = 20
    min_speed_percent: int = 1
    max_speed_percent: int = 100
    default_joint_step_deg: float = 1.0
    default_cart_step_mm: float = 5.0
    move_hold_s: int = 4

    require_enable_before_motion: bool = True
    disable_on_disconnect: bool = False
    allow_t2_mode: bool = False

    di_preview_count: int = 32
    do_preview_count: int = 32

    @property
    def exe_path(self) -> str:
        return str(Path(self.exe_dir) / self.exe_name)

    def clamp_speed(self, v: int) -> int:
        return max(self.min_speed_percent, min(self.max_speed_percent, int(v)))


def _apply_scara_section(cfg: ScaraConfig, data: Dict[str, Any]) -> None:
    scara = data.get("scara") or {}
    if not scara:
        return

    def get(key: str, cast, default):
        if key not in scara or scara[key] is None:
            return default
        return cast(scara[key])

    cfg.exe_name = get("exe_name", str, cfg.exe_name)
    cfg.controller_ip = get("controller_ip", str, cfg.controller_ip)
    cfg.controller_port = get("controller_port", int, cfg.controller_port)
    cfg.command_timeout_s = get("command_timeout_s", float, cfg.command_timeout_s)
    cfg.poll_interval_ms = get("poll_interval_ms", int, cfg.poll_interval_ms)
    cfg.default_speed_percent = get("default_speed_percent", int, cfg.default_speed_percent)
    cfg.default_joint_step_deg = get("default_joint_step_deg", float, cfg.default_joint_step_deg)
    cfg.default_cart_step_mm = get("default_cart_step_mm", float, cfg.default_cart_step_mm)
    cfg.move_hold_s = get("move_hold_s", int, cfg.move_hold_s)
    cfg.require_enable_before_motion = get(
        "require_enable_before_motion", bool, cfg.require_enable_before_motion
    )
    cfg.disable_on_disconnect = get(
        "disable_on_disconnect", bool, cfg.disable_on_disconnect
    )
    cfg.allow_t2_mode = get("allow_t2_mode", bool, cfg.allow_t2_mode)


def load_scara_config(path: Optional[str] = None) -> ScaraConfig:
    """从 local_config.toml 加载 SCARA 配置。"""
    cfg = ScaraConfig()
    cfg_file = find_config_file(path)
    if cfg_file is not None:
        try:
            _apply_scara_section(cfg, _load_toml(cfg_file))
        except Exception:
            cfg = ScaraConfig()
    try:
        cfg.exe_dir = str(resolve_snrobotlab_dir(str(cfg_file) if cfg_file else path))
    except FileNotFoundError:
        cfg.exe_dir = ""
    return cfg


def load_duco_overrides(path: Optional[str] = None) -> Dict[str, Any]:
    """读取 local_config.toml 的 [duco] 段，供 robot_arm.yaml 覆盖。"""
    cfg_file = find_config_file(path)
    if cfg_file is None:
        return {}
    try:
        duco = (_load_toml(cfg_file).get("duco") or {})
        return dict(duco) if isinstance(duco, dict) else {}
    except Exception:
        return {}
