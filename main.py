"""机械臂控制程序（DUCO + SCARA）独立启动器。

两个设备页放在一个 QTabWidget 里：
  - 「机械臂控制」：DUCO GCR3-618 六轴协作臂（经本机 armweb 代理连真机）
  - 「SCARA 机械臂」：新松 SA4A-4/0.40（经 snrobot.exe 命令行桥连控制器）

启动时随 GUI 拉起 armweb（8080），退出时停掉。加 --no-services 可跳过服务自启（例如服务已由人工启动）。

用法：
    python main.py [--no-services]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget


def main() -> int:
    # armweb 是 DUCO 真机代理，随 GUI 一起起。--no-services 可跳过。
    start_services = "--no-services" not in sys.argv
    if not start_services:
        sys.argv.remove("--no-services")

    # 初始化日志
    from utils import setup_logging, get_logger
    setup_logging()
    logger = get_logger("main")

    logger.info("=" * 50)
    logger.info("机械臂控制程序启动（DUCO + SCARA）")
    logger.info("=" * 50)

    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("机械臂控制程序")
    app.setApplicationVersion("1.0.0")

    # 设置应用样式
    app.setStyle("Fusion")

    # 全局样式表：所有数字输入框统一左对齐
    app.setStyleSheet("""
        QSpinBox, QDoubleSpinBox {
            text-align: left;
        }
    """)

    # 拉起 armweb（8080）。放在建主窗口之前，页面构造时服务已可用。
    services = None
    if start_services:
        from utils.local_services import LocalServices
        services = LocalServices()
        services.start_all()
    else:
        logger.info("--no-services 已指定，跳过本地后端服务自启")

    # 实例化两个设备模块
    from robot_arm.module import RobotArmModule
    from scara.module import ScaraModule

    robot_arm_module = RobotArmModule()
    scara_module = ScaraModule()
    modules = [robot_arm_module, scara_module]

    # 主窗口：一个 QTabWidget 放两个控制页
    win = QMainWindow()
    win.setWindowTitle("机械臂控制程序 — DUCO GCR3-618 / 新松 SA4A-4/0.40")
    tabs = QTabWidget()
    tabs.addTab(robot_arm_module.create_control_widget(), robot_arm_module.module_name)
    tabs.addTab(scara_module.create_control_widget(), scara_module.module_name)
    win.setCentralWidget(tabs)
    win.resize(1680, 1000)
    win.show()

    def _cleanup() -> None:
        """关窗收尾：先清理两个模块（断开设备），再停本地服务。"""
        for m in modules:
            try:
                m.cleanup()
            except Exception as exc:
                logger.warning("模块 %s 清理失败: %s", m.module_id, exc)
        if services is not None:
            services.stop_all()

    app.aboutToQuit.connect(_cleanup)

    rc = app.exec()
    logger.info("机械臂控制程序退出")
    return rc


if __name__ == "__main__":
    sys.exit(main())
