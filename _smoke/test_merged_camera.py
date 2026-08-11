"""合并版摄像功能的无硬件回归测试。"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(ROOT)

from PyQt6.QtWidgets import QApplication  # noqa: E402

from scara.controller.scara_controller import ScaraController  # noqa: E402
from scara.ui.camera_view import ScaraCameraThread  # noqa: E402
from scara.ui.control_widget import ScaraControlWidget  # noqa: E402
from scara.ui.photo_trajectory_worker import PhotoTrajectoryWorker  # noqa: E402


APP = QApplication.instance() or QApplication([])


class _FakeProc:
    def poll(self):
        return None


class _FakeController:
    def __init__(self):
        self.visited = []

    def goto_joints_sync(self, name, joints, should_stop=None):
        self.visited.append((name, list(joints)))
        return True

    def emergency_stop(self):
        raise AssertionError("测试中不应触发急停")


class _FakeCamera:
    def snapshot(self, path, max_age_s=1.0):
        Path(path).write_bytes(b"jpeg-test")
        return True


class MergedCameraTests(unittest.TestCase):
    def test_verified_joint_motion_simulation(self):
        controller = ScaraController()
        self.assertTrue(hasattr(controller, "_enable_exe_lock"))
        self.assertTrue(hasattr(controller, "_motion_sequence_lock"))
        controller._connected = True
        controller._proc = _FakeProc()
        controller._last_status = {
            "effectively_enabled": True,
            "need_clear": False,
            "estop": False,
            "warn": 0,
        }
        current = [0.0, 0.0, 0.0, 0.0]

        def read_status():
            return {
                "joints": list(current),
                "effectively_enabled": True,
                "need_clear": False,
                "estop": False,
                "warn": 0,
            }

        def send(line, timeout=None):
            if line.startswith("move1 "):
                _, axis, delta, _hold = line.split()
                current[int(axis) - 1] += float(delta)
            return "OK"

        controller.read_all_sync = read_status
        controller._send = send
        self.assertTrue(
            controller.goto_joints_sync("simulation", [1.0, 2.0, -3.0, 4.0])
        )
        self.assertEqual(current, [1.0, 2.0, -3.0, 4.0])

    def test_camera_freshness_and_snapshot(self):
        camera = ScaraCameraThread(index=1)
        self.assertEqual(camera.source_index, 1)
        self.assertFalse(camera.has_fresh_frame())
        with camera._frame_lock:
            camera._last_frame = np.zeros((8, 8, 3), dtype=np.uint8)
            camera._last_frame_at = time.monotonic()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "nested" / "snapshot.jpg"
            self.assertTrue(camera.snapshot(output))
            self.assertTrue(output.is_file())
        with camera._frame_lock:
            camera._last_frame_at = time.monotonic() - 2.0
        self.assertFalse(camera.has_fresh_frame(max_age_s=1.0))

    def test_photo_trajectory_worker(self):
        controller = _FakeController()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            worker = PhotoTrajectoryWorker(
                controller,
                _FakeCamera(),
                [{"name": "P00", "joints": [1, 2, 3, 4]}],
                output,
                wait_before_photo_s=0,
                wait_after_photo_s=0,
            )
            worker.run()
            self.assertTrue((output / "001_P00.jpg").is_file())
        self.assertEqual(controller.visited, [("P00", [1.0, 2.0, 3.0, 4.0])])

    def test_bundled_trajectory_and_offscreen_ui(self):
        path = ROOT / "Preset Trajectories" / "four_corners.py"
        spec = importlib.util.spec_from_file_location("merged_trajectory_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["merged_trajectory_test"] = module
        spec.loader.exec_module(module)
        self.assertEqual(len(module.build_trajectory()), 5)

        widget = ScaraControlWidget(owns_controller=False)
        self.assertEqual(widget._btn_photo_trajectory.text(), "沿轨迹拍照")
        self.assertFalse(widget._btn_photo_trajectory.isEnabled())
        widget.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
