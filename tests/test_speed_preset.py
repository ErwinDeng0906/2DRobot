"""Speed-preset safety tests; no controller process or motion is started."""

from types import SimpleNamespace
from pathlib import Path
import sys
import threading
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scara.config.scara_config import ScaraConfig
from scara.ui.control_widget import ScaraControlWidget


class _WidgetStub:
    def __init__(self) -> None:
        self.enabled = None
        self.value = None
        self.blocked = []

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802
        self.enabled = bool(enabled)

    def setText(self, _text: str) -> None:  # noqa: N802
        pass

    def setStyleSheet(self, _style: str) -> None:  # noqa: N802
        pass

    def blockSignals(self, blocked: bool) -> None:  # noqa: N802
        self.blocked.append(bool(blocked))

    def setValue(self, value: int) -> None:  # noqa: N802
        self.value = int(value)


class _ControllerStub:
    def __init__(self) -> None:
        self.speed_requests = []

    def cmd_set_speed(self, value: int) -> None:
        self.speed_requests.append(int(value))


class SpeedPresetTests(unittest.TestCase):
    def test_global_speed_commands_are_capped_at_twenty_percent(self) -> None:
        config = ScaraConfig(default_speed_percent=37)
        self.assertEqual(20, config.max_speed_percent)
        self.assertEqual(20, config.clamp_speed(37))

    def test_connection_reapplies_twenty_percent_preset(self) -> None:
        config = ScaraConfig(default_speed_percent=37)
        controller = _ControllerStub()
        speed = _WidgetStub()
        owner = SimpleNamespace(
            _cfg=config,
            _ctrl=controller,
            _speed=speed,
            _handeye_state_lock=threading.Lock(),
            _handeye_controller_connected=False,
            _latest_handeye_robot_state=None,
            _handeye_robot_state_history=[],
            _conn_chip=_WidgetStub(),
            _btn_conn=_WidgetStub(),
            _btn_disc=_WidgetStub(),
            _btn_en=_WidgetStub(),
            _btn_dis=_WidgetStub(),
            _btn_clr=_WidgetStub(),
            _btn_home=_WidgetStub(),
            _btn_estop=_WidgetStub(),
            _set_do_ui_enabled=lambda _enabled: None,
        )

        ScaraControlWidget._on_conn(owner, True)

        self.assertEqual(20, speed.value)
        self.assertEqual([True, False], speed.blocked)
        self.assertEqual([20], controller.speed_requests)


if __name__ == "__main__":
    unittest.main()
