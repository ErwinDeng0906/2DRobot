"""机械臂设备后端包（DUCO GCR3-618）。

UI 通过 create_backend() 工厂获取后端，不直接 import 具体实现：
  - mode="sim"    → SimRobotArmBackend（无硬件，开发/仿真模式）
  - mode="thrift" → ThriftRobotArmBackend（真机，需 thrift 库）
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .backend import RobotArmBackendBase, RobotArmStatus

_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = str(_DIR / "robot_arm.yaml")
DEFAULT_WAYPOINTS_PATH = str(_DIR / "waypoints.json")


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # 本机覆盖：根目录 local_config.toml 的 [duco]（换电脑只改那一份）
    try:
        from scara.config.scara_config import load_duco_overrides
        ov = load_duco_overrides()
    except Exception:
        ov = {}
    if ov:
        conn = cfg.setdefault("connection", {})
        if ov.get("robot_ip"):
            conn["ip"] = str(ov["robot_ip"])
        if ov.get("rpc_port") is not None:
            conn["rpc_port"] = int(ov["rpc_port"])
        if ov.get("armweb_base_url"):
            cfg["armweb_base_url"] = str(ov["armweb_base_url"])
    return cfg


def create_backend(mode: str = "sim", cfg: dict = None,
                   config_path: str = DEFAULT_CONFIG_PATH,
                   waypoints_path: str = DEFAULT_WAYPOINTS_PATH,
                   base_url: str = None) -> RobotArmBackendBase:
    """创建后端。mode: 'sim' / 'thrift' / 'http'(远程代理)。"""
    if cfg is None:
        cfg = load_config(config_path)
    if mode == "thrift":
        from .thrift_backend import ThriftRobotArmBackend
        return ThriftRobotArmBackend(cfg, waypoints_path)
    elif mode == "sim":
        from .sim_backend import SimRobotArmBackend
        return SimRobotArmBackend(cfg, waypoints_path)
    elif mode == "http":
        from .http_backend import HttpRobotArmBackend
        if not base_url:
            raise ValueError("http 模式需要 base_url（服务器代理地址）")
        return HttpRobotArmBackend(base_url, cfg=cfg)
    raise ValueError("未知后端模式: %s" % mode)


__all__ = [
    "RobotArmBackendBase", "RobotArmStatus",
    "create_backend", "load_config",
    "DEFAULT_CONFIG_PATH", "DEFAULT_WAYPOINTS_PATH",
]
