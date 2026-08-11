"""
设备模块接口定义

所有设备控制模块都需要实现此接口
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import QObject


class DeviceModule(ABC):
    """
    设备模块接口
    
    所有设备控制模块都需要实现此接口
    """
    
    @property
    @abstractmethod
    def module_name(self) -> str:
        """模块名称（用于显示）"""
        pass
    
    @property
    @abstractmethod
    def module_id(self) -> str:
        """模块ID（用于内部标识）"""
        pass
    
    @property
    @abstractmethod
    def module_version(self) -> str:
        """模块版本"""
        pass
    
    @abstractmethod
    def create_control_widget(self, parent: Optional[QWidget] = None) -> QWidget:
        """
        创建设备控制界面
        
        Args:
            parent: 父窗口
            
        Returns:
            控制界面Widget
        """
        pass
    
    @abstractmethod
    def get_status_widget(self) -> Optional[QWidget]:
        """
        获取状态显示Widget（可选）
        
        Returns:
            状态Widget，如果不需要则返回None
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """检查设备是否已连接"""
        pass
    
    def create_mirror_widget(self, parent: Optional[QWidget] = None) -> Optional[QWidget]:
        """
        创建镜像控制界面（共享同一个控制器）
        
        用于弹出独立窗口时，两边同时显示同步内容。
        子类应重写此方法，创建一个新的UI widget并绑定到已有的控制器上。
        
        Args:
            parent: 父窗口
            
        Returns:
            镜像控制界面Widget，不支持则返回None
        """
        return None

    @abstractmethod
    def cleanup(self) -> None:
        """清理资源（断开连接等）"""
        pass

