"""SCARA 使能 / 去使能 / 急停 / 清报警 真机探测。

前提：关掉官方 SCARA GUI 和本程序 snrobot serve（抢连接会失败）。

用法:
  python tools\\uitest\\test_enable.py status
  python tools\\uitest\\test_enable.py enable
  python tools\\uitest\\test_enable.py disable
  python tools\\uitest\\test_enable.py set_mode 1|2|3|4
  python tools\\uitest\\test_enable.py estop
  python tools\\uitest\\test_enable.py release_estop
  python tools\\uitest\\test_enable.py clear_alarm
  python tools\\uitest\\test_enable.py cycle    # enable → disable

看 status 输出:
  ENABLE_ON/OFF  — 官方「使能」
  ESTOP_ON/OFF   — 官方「急停」
  ALARM_ON/OFF   — 官方「运行状态」报警/正常
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import enable_client as C  # noqa: E402


def _run(name: str, fn, *a):
    print(f"\n===== {name} =====")
    ok, msg = fn(*a) if a else fn()
    print(msg)
    print("PASS" if ok else "FAIL")
    return ok


def main() -> int:
    args = sys.argv[1:]
    if not args:
        return 0 if _run("status", C.status) else 2

    cmd = args[0]
    table = {
        "status": C.status,
        "enable": C.enable,
        "disable": C.disable,
        "estop": C.estop,
        "release_estop": C.release_estop,
        "clear_alarm": C.clear_alarm,
    }
    if cmd in table:
        return 0 if _run(cmd, table[cmd]) else 2

    if cmd == "set_mode":
        if len(args) < 2:
            print("用法: set_mode <1|2|3|4>  (1=T1 2=T2 3=执行 4=远程)")
            return 1
        return 0 if _run(f"set_mode {args[1]}", C.set_mode, int(args[1])) else 2

    if cmd == "cycle":
        # 使能 → 去使能：确认 ENABLE 从 ON 变 OFF，且急停仍为 OFF
        if not _run("status before", C.status):
            return 2
        if not _run("enable", C.enable):
            return 2
        if not _run("disable", C.disable):
            return 2
        if not _run("status after", C.status):
            return 2
        print("\n期望: ENABLE_OFF，且 ESTOP_OFF / ALARM_OFF。")
        return 0

    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
