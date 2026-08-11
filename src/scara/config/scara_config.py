"""
SCARA 机械臂模块 —— 配置模型

集中管理与新松 SA4A-4/0.40 通信及运动相关的默认参数。
控制不走裸协议，而是调用位于 SNRobotLab 目录下的命令行桥 ``snrobot.exe``
（该目录含许可证 SiaSunRobot.lic，exe 必须在此目录运行）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:  # Python 3.11+ 自带 tomllib
    import tomllib as _toml
except ModuleNotFoundError:  # 3.10 回退到 tomli
    import tomli as _toml  # type: ignore


# 默认 SNRobotLab 目录（含 snrobot.exe、RobotSDK.dll、许可证 SiaSunRobot.lic）。
# 换电脑请优先改程序根目录的 scara_config.toml，或设环境变量 SNROBOTLAB_DIR。
# 不要指望只改这里：do_client / enable_client 也会走 resolve_snrobotlab_dir()。
DEFAULT_EXE_DIR = r"D:\SNRobotLab"
DEFAULT_EXE_NAME = "snrobot.exe"


@dataclass
class ScaraConfig:
    """SCARA 模块配置。"""

    # —— 连接 ——
    exe_dir: str = DEFAULT_EXE_DIR            # SNRobotLab 目录（snrobot 工作目录）
    exe_name: str = DEFAULT_EXE_NAME
    controller_ip: str = "192.168.1.100"     # 界面显示；真连以工位配置 / --ip 为准
    controller_port: int = 20002
    command_timeout_s: float = 8.0           # 单条命令超时
    poll_interval_ms: int = 300              # 状态轮询间隔(越小越实时)

    # —— 运动 ——
    default_speed_percent: int = 20
    min_speed_percent: int = 1
    max_speed_percent: int = 100
    default_joint_step_deg: float = 1.0      # 关节点动步长（度；J3=mm）
    default_cart_step_mm: float = 5.0        # 笛卡尔点动步长（mm）
    move_hold_s: int = 4                     # 单步运动后保持/观察秒数（命令内部）

    # —— 安全 ——
    require_enable_before_motion: bool = True
    disable_on_disconnect: bool = False
    allow_t2_mode: bool = False              # 是否允许切到 T2（全速）

    # —— IO ——
    di_preview_count: int = 32               # 界面预览的 DI 通道数
    do_preview_count: int = 32               # 界面预览的 DO 通道数

    @property
    def exe_path(self) -> str:
        return str(Path(self.exe_dir) / self.exe_name)

    def clamp_speed(self, v: int) -> int:
        return max(self.min_speed_percent, min(self.max_speed_percent, int(v)))


def _read_toml_exe_dir(toml_path: Path) -> Optional[str]:
    if not toml_path.exists():
        return None
    try:
        with open(toml_path, "rb") as f:
            data = _toml.load(f)
        exe_dir = data.get("connection", {}).get("exe_dir")
        return str(exe_dir) if exe_dir else None
    except Exception:
        return None


def resolve_snrobotlab_dir(toml_path: Optional[str] = None) -> Path:
    """解析 SNRobotLab 目录（给 snrobot / scara_do / scara_enable 共用）。

    优先级（高 → 低）：
      1. 环境变量 SNROBOTLAB_DIR
      2. scara_config.toml 里的 connection.exe_dir
      3. 代码默认 DEFAULT_EXE_DIR
    """
    import os

    env = os.environ.get("SNROBOTLAB_DIR", "").strip()
    if env:
        return Path(env)
    path = Path(toml_path) if toml_path else Path("scara_config.toml")
    from_toml = _read_toml_exe_dir(path)
    if from_toml:
        return Path(from_toml)
    return Path(DEFAULT_EXE_DIR)


def load_scara_config(path: Optional[str] = None) -> ScaraConfig:
    """从 ``scara_config.toml`` 加载配置；文件不存在时返回默认配置。"""
    cfg = ScaraConfig()
    toml_path = Path(path) if path else Path("scara_config.toml")
    if not toml_path.exists():
        cfg.exe_dir = str(resolve_snrobotlab_dir(str(toml_path)))
        return cfg
    try:
        with open(toml_path, "rb") as f:
            data = _toml.load(f)
        conn = data.get("connection", {})
        cfg.exe_dir = conn.get("exe_dir", cfg.exe_dir)
        cfg.exe_name = conn.get("exe_name", cfg.exe_name)
        cfg.controller_ip = conn.get("controller_ip", cfg.controller_ip)
        cfg.controller_port = int(conn.get("controller_port", cfg.controller_port))
        cfg.command_timeout_s = float(conn.get("command_timeout_s", cfg.command_timeout_s))
        cfg.poll_interval_ms = int(conn.get("poll_interval_ms", cfg.poll_interval_ms))
        mot = data.get("motion", {})
        cfg.default_speed_percent = int(mot.get("default_speed_percent", cfg.default_speed_percent))
        cfg.default_joint_step_deg = float(mot.get("default_joint_step_deg", cfg.default_joint_step_deg))
        cfg.default_cart_step_mm = float(mot.get("default_cart_step_mm", cfg.default_cart_step_mm))
        cfg.move_hold_s = int(mot.get("move_hold_s", cfg.move_hold_s))
        saf = data.get("safety", {})
        cfg.require_enable_before_motion = bool(saf.get("require_enable_before_motion", cfg.require_enable_before_motion))
        cfg.disable_on_disconnect = bool(saf.get("disable_on_disconnect", cfg.disable_on_disconnect))
        cfg.allow_t2_mode = bool(saf.get("allow_t2_mode", cfg.allow_t2_mode))
        # 环境变量优先于 toml（与 resolve_snrobotlab_dir 一致）
        cfg.exe_dir = str(resolve_snrobotlab_dir(str(toml_path)))
    except Exception:
        # 配置损坏时回退默认，仍尊重环境变量
        cfg = ScaraConfig()
        cfg.exe_dir = str(resolve_snrobotlab_dir(str(toml_path)))
        return cfg
    return cfg
