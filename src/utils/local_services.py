"""随 GUI 一起启动的本地后端服务。

调用链::

    GUI -> armweb (8080) -> DUCO 真机 (thrift)

已在运行的服务（手工启动或上次遗留）会被识别并接管：既不重复启动，
退出时也不去停它们。
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

from utils import get_logger

logger = get_logger("services")

HOST = "127.0.0.1"
ARMWEB_PORT = 8080


def _repo_root() -> Path:
    # src/utils/local_services.py -> src/utils -> src -> 程序根目录
    return Path(__file__).resolve().parents[2]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r 不是整数，按默认值 %d 处理", name, raw, default)
        return default


def is_listening(port: int, host: str = HOST, timeout: float = 0.4) -> bool:
    """端口已有服务在监听时返回 True。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class LocalService:
    """由本 GUI 持有的子进程服务，以其监听端口标识。"""

    def __init__(self, name: str, port: int, argv: Sequence[str],
                 log_stem: str, startup_timeout_s: float = 20.0):
        self.name = name
        self.port = int(port)
        self._argv = list(argv)
        self._log_stem = log_stem
        self._startup_timeout_s = float(startup_timeout_s)
        self._proc: Optional[subprocess.Popen] = None
        self._log_path: Optional[Path] = None
        self._adopted_external = False

    # ------------------------------------------------------------------
    def start(self) -> bool:
        """端口未被占用时启动服务。就绪返回 True。"""
        if is_listening(self.port):
            self._adopted_external = True
            logger.info("%s 已在端口 %d 运行，保持现状", self.name, self.port)
            return True

        script = Path(self._argv[0])
        if not script.is_file():
            logger.error("%s 未启动：找不到 %s", self.name, script)
            return False

        log_dir = _repo_root() / "logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("无法创建日志目录 %s: %s", log_dir, exc)
        self._log_path = log_dir / f"{self._log_stem}_{datetime.now():%Y%m%d}.log"

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        env = dict(os.environ)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUNBUFFERED", "1")

        try:
            handle = open(self._log_path, "a", encoding="utf-8", errors="replace")
            handle.write(f"\n===== {self.name} 启动于 {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
            handle.flush()
            self._proc = subprocess.Popen(
                [sys.executable, *self._argv],
                cwd=str(_repo_root()),
                stdout=handle,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
                env=env,
            )
        except Exception as exc:
            logger.error("启动 %s 失败: %s", self.name, exc)
            self._proc = None
            return False

        logger.info("%s 正在启动 (pid=%s, 端口=%d, 日志=%s)",
                    self.name, self._proc.pid, self.port, self._log_path)
        return self._wait_until_ready()

    def _wait_until_ready(self) -> bool:
        deadline = time.time() + self._startup_timeout_s
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                logger.error("%s 启动阶段退出 (退出码=%s)，详见 %s",
                             self.name, self._proc.returncode, self._log_path)
                return False
            if is_listening(self.port):
                logger.info("%s 就绪：http://%s:%d", self.name, HOST, self.port)
                return True
            time.sleep(0.25)
        logger.error("%s 在 %.0fs 内未开始监听，详见 %s",
                     self.name, self._startup_timeout_s, self._log_path)
        return False

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """停止服务 —— 仅当它是本程序启动的。"""
        if self._adopted_external:
            logger.info("%s 是外部启动的，保持运行", self.name)
            return
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        logger.info("正在停止 %s (pid=%s)", self.name, proc.pid)
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("%s 5 秒内未退出，强制结束", self.name)
                proc.kill()
                proc.wait(timeout=3)
        except Exception as exc:
            logger.warning("停止 %s 时出错: %s", self.name, exc)

    @property
    def log_path(self) -> Optional[Path]:
        return self._log_path


def _armweb_service() -> LocalService:
    root = _repo_root()
    port = _env_int("ARMWEB_PORT", ARMWEB_PORT)
    return LocalService(
        name="armweb",
        port=port,
        argv=[str(root / "webconsole" / "server.py"), "--port", str(port)],
        log_stem="armweb",
    )


def connect_arm(port: Optional[int] = None, mode: str = "thrift") -> bool:
    """把 armweb 从默认的 sim 模式切到真机。

    这里只做连接 —— 不使能（伺服上电）。使能会让机械臂带电，
    始终是操作者手动确认的动作。
    """
    import json
    import urllib.request

    p = int(port) if port is not None else _env_int("ARMWEB_PORT", ARMWEB_PORT)
    body = json.dumps({"mode": mode}).encode("utf-8")
    req = urllib.request.Request(
        f"http://{HOST}:{p}/api/connect", data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as exc:
        logger.error("机械臂自动连接失败: %s", exc)
        return False
    if data.get("ok"):
        logger.info("机械臂已自动连接 (mode=%s)", mode)
        return True
    logger.error("机械臂自动连接被拒绝: %s", data.get("error", data))
    return False


class LocalServices:
    """启动 armweb；停止时关闭由本程序拉起的实例。"""

    def __init__(self, connect_arm_on_start: bool = True) -> None:
        self._services: List[LocalService] = [_armweb_service()]
        self._connect_arm = bool(connect_arm_on_start)

    def start_all(self) -> bool:
        ok = True
        for svc in self._services:
            if not svc.start():
                ok = False
                logger.error(
                    "%s 未就绪（日志: %s）",
                    svc.name, svc.log_path,
                )
        if ok and self._connect_arm and not connect_arm():
            ok = False
        return ok

    def stop_all(self) -> None:
        for svc in reversed(self._services):
            svc.stop()
