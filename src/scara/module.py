"""
SCARA 机械臂设备模块

实现 DeviceModule 接口，将新松 SA4A-4/0.40 SCARA 控制集成到多设备控制系统。
控制经 snrobot.exe 命令行桥（封装官方 RobotCommunication SDK）。
"""

from __future__ import annotations

import weakref
from typing import Optional

from PyQt6.QtWidgets import QWidget

from core.module_interface import DeviceModule
from utils import get_logger
from scara.config.scara_config import load_scara_config
from scara.controller.scara_controller import ScaraController

logger = get_logger("scara.module")


class ScaraModule(DeviceModule):
    """SCARA 机械臂控制模块。"""

    def __init__(self) -> None:
        self._config = load_scara_config()
        self._controller: Optional[ScaraController] = None
        self._control_widget = None
        self._mirror_widgets: "weakref.WeakSet[QWidget]" = weakref.WeakSet()

    @property
    def module_name(self) -> str:
        return "SCARA 机械臂"

    @property
    def module_id(self) -> str:
        return "scara"

    @property
    def module_version(self) -> str:
        return "1.0.0"

    def create_control_widget(self, parent: Optional[QWidget] = None) -> QWidget:
        from scara.ui.control_widget import ScaraControlWidget

        if self._control_widget is None:
            self._control_widget = ScaraControlWidget(
                parent=parent, config=self._config,
                controller=self._controller, owns_controller=self._controller is None,
            )
            # 首个界面创建控制器后，作为共享控制器供镜像窗口使用
            self._controller = self._control_widget.controller
            logger.info("SCARA 控制界面已创建")
        return self._control_widget

    def get_status_widget(self) -> Optional[QWidget]:
        return None

    def is_connected(self) -> bool:
        if self._controller is not None:
            try:
                return bool(self._controller.is_connected())
            except Exception:
                return False
        return False

    def create_mirror_widget(self, parent: Optional[QWidget] = None) -> Optional[QWidget]:
        from scara.ui.control_widget import ScaraControlWidget

        if self._control_widget is None:
            self.create_control_widget()
        mirror = ScaraControlWidget(
            parent=parent, config=self._config,
            controller=self._controller, owns_controller=False,
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
        if self._control_widget is not None:
            try:
                self._control_widget.cleanup()
            except Exception as exc:
                logger.warning("SCARA 控制界面清理失败: %s", exc)
        if self._controller is not None:
            try:
                self._controller.cleanup()
            except Exception:
                pass
        self._control_widget = None
        self._controller = None
        logger.info("SCARA 模块已清理")
