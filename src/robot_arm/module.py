"""
机械臂控制设备模块
实现 DeviceModule 接口，提供 DUCO GCR3-618 六轴协作臂控制页面。

运行架构：本地 PyQt（Windows）→ HTTP → 本机 armweb 代理（127.0.0.1:8080）→ 真机
（192.168.1.10:7003）。页面模式恒为远程(http)。后端层在 src/devices/robot_arm/
（http_backend 走代理；thrift_backend 由 armweb 侧直连真机）。
界面见 robot_arm/ui/console_widget.py。
"""

from __future__ import annotations

import weakref
from typing import Optional

from PyQt6.QtWidgets import QWidget

from core.module_interface import DeviceModule
from utils import get_logger

logger = get_logger("robot_arm.module")


class RobotArmModule(DeviceModule):
    def __init__(self):
        self._control_widget = None
        self._shared_backend = None
        self._mirror_widgets: "weakref.WeakSet[QWidget]" = weakref.WeakSet()

    @property
    def module_name(self) -> str:
        return "机械臂控制"

    @property
    def module_id(self) -> str:
        return "robot_arm"

    @property
    def module_version(self) -> str:
        return "1.0.0"

    def create_control_widget(self, parent: Optional[QWidget] = None) -> QWidget:
        from robot_arm.ui.console_control_widget import ArmConsoleControlWidget

        if self._control_widget is None:
            self._control_widget = ArmConsoleControlWidget(
                parent=parent,
                backend=self._shared_backend,
                owns_backend=True,
                on_backend_changed=self._set_shared_backend,
            )
            self._shared_backend = self._control_widget.backend
            logger.info("机械臂控制界面已创建")
        return self._control_widget

    def _set_shared_backend(self, backend) -> None:
        """机械臂页连接/切换时会 create_backend 造新实例；同步到 _shared_backend，
        让 orchestrator 等外部持有者拿到实时后端，而不是创建时的旧引用。"""
        self._shared_backend = backend

    def get_status_widget(self) -> Optional[QWidget]:
        return None

    def is_connected(self) -> bool:
        if self._shared_backend is not None:
            try:
                return bool(self._shared_backend.is_connected())
            except Exception:
                return False
        if self._control_widget is None:
            return False
        return self._control_widget.is_device_connected()

    def create_mirror_widget(self, parent=None) -> Optional[QWidget]:
        from robot_arm.ui.console_control_widget import ArmConsoleControlWidget

        if self._control_widget is None:
            self.create_control_widget()

        mirror = ArmConsoleControlWidget(
            parent=parent,
            backend=self._shared_backend,
            owns_backend=False,
        )
        self._mirror_widgets.add(mirror)
        return mirror

    def cleanup(self) -> None:
        for mirror in list(self._mirror_widgets):
            try:
                mirror.cleanup()
            except Exception:
                pass
        self._mirror_widgets.clear()

        if self._control_widget:
            self._control_widget.cleanup()
            logger.info("机械臂已断开连接并清理资源")
        self._control_widget = None
        self._shared_backend = None
