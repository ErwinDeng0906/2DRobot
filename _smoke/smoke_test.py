"""离屏冒烟测试：构建两个控制页控件树、截图、清理。不连接任何硬件。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(ROOT)

from PyQt6.QtWidgets import QApplication, QWidget  # noqa: E402

OUT = Path(__file__).resolve().parent


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # —— DUCO 机械臂页 ——
    from robot_arm.module import RobotArmModule
    arm_mod = RobotArmModule()
    arm_w = arm_mod.create_control_widget()
    arm_w.resize(1680, 1000)
    arm_w.show()
    app.processEvents()
    n_arm = len(arm_w.findChildren(QWidget))
    print("DUCO 页控件数:", n_arm)
    arm_w.grab().save(str(OUT / "smoke_robot_arm.png"))

    # —— SCARA 机械臂页 ——
    from scara.module import ScaraModule
    scara_mod = ScaraModule()
    scara_w = scara_mod.create_control_widget()
    scara_w.resize(1680, 1000)
    scara_w.show()
    app.processEvents()
    n_scara = len(scara_w.findChildren(QWidget))
    print("SCARA 页控件数:", n_scara)
    scara_w.grab().save(str(OUT / "smoke_scara.png"))

    assert n_arm > 10 and n_scara > 10, "控件树异常（部件过少）"

    arm_mod.cleanup()
    scara_mod.cleanup()
    print("清理完成，冒烟测试通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
