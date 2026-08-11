"""合并版摄像功能的无硬件回归测试。"""

from __future__ import annotations

import importlib.util
import json
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
from scara.ui.action_worker import (  # noqa: E402
    ActionWorker,
    CameraSourcePool,
    calculate_camera_position,
    normalize_action_task,
)
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
                "pose": list(pose),
                "effectively_enabled": True,
                "need_clear": False,
                "estop": False,
                "warn": 0,
            }

        def send(line, timeout=None):
            if line.startswith("move1 "):
                _, axis, delta, _hold = line.split()
                current[int(axis) - 1] += float(delta)
                if int(axis) == 4:
                    pose[5] += float(delta)
            elif line.startswith("cartstep "):
                _, axis, delta = line.split()
                pose[{"7": 0, "8": 1, "9": 2}[axis]] += float(delta)
            return "OK"

        pose = [10.0, 20.0, 30.0, 180.0, 0.0, 4.0]
        controller.read_all_sync = read_status
        controller._send = send
        self.assertTrue(
            controller.goto_joints_sync("simulation", [1.0, 2.0, -3.0, 4.0])
        )
        self.assertEqual(current, [1.0, 2.0, -3.0, 4.0])
        self.assertTrue(
            controller.move_xyzr_sync(
                "xyzr simulation", x_mm=1.0, y_mm=-2.0, z_mm=3.0, r_deg=5.0
            )
        )
        self.assertEqual(pose[:3], [11.0, 18.0, 33.0])
        self.assertEqual(current[3], 9.0)

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

    def test_action_worker_json_and_camera_equation(self):
        class FakeActionController:
            def __init__(self):
                self.joints = [0.0, 0.0, 0.0, 0.0]
                self.pose = [100.0, 200.0, 300.0, 180.0, 0.0, 0.0]

            def read_all_sync(self):
                return {"joints": list(self.joints), "pose": list(self.pose)}

            def goto_joints_sync(self, name, joints, should_stop=None, tolerance=0.2):
                self.joints = list(joints)
                return True

            def move_xyzr_sync(
                self, name, *, x_mm, y_mm, z_mm, r_deg, should_stop=None
            ):
                self.pose[0] += x_mm
                self.pose[1] += y_mm
                self.pose[2] += z_mm
                self.pose[5] += r_deg
                self.joints[2] += z_mm
                self.joints[3] += r_deg
                return True

            def emergency_stop(self):
                raise AssertionError("测试中不应触发急停")

        task = normalize_action_task(
            {
                "name": "test",
                "camera_model": {
                    "offset_mm": 20,
                    "angle_reference": "world_negative_y",
                    "positive_rotation": "counter_clockwise_from_above",
                },
                "actions": [
                    {"type": "assert_joints", "name": "start", "joints": [0, 0, 0, 0]},
                    {"type": "wait", "seconds": 0},
                    {"type": "record_point", "name": "start"},
                    {"type": "capture", "source": 0},
                    {"type": "capture", "source": 1},
                    {"type": "move_xyzr", "name": "down", "z_mm": -5},
                    {"type": "record_point", "name": "down"},
                    {"type": "capture", "source": 2},
                ],
            }
        )

        def save_photo(source, path):
            Path(path).write_bytes(f"source-{source}".encode())
            return True

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "260811120000"
            worker = ActionWorker(
                FakeActionController(),
                task,
                output,
                snapshot_source=save_photo,
                source_position_calculators={
                    1: lambda joints, pose: {
                        "x_mm": pose[0] + 33.55,
                        "y_mm": pose[1],
                        "z_mm": pose[2],
                    }
                },
            )
            worker.run()
            self.assertTrue((output / "0_001.jpg").is_file())
            self.assertTrue((output / "1_001.jpg").is_file())
            self.assertTrue((output / "2_002.jpg").is_file())
            manifest = json.loads((output / "points.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(
                manifest["coordinate_convention"]["world_y_positive"], "P05 到 P00"
            )
            self.assertEqual(len(manifest["points"]), 2)
            self.assertEqual(
                manifest["points"][0]["mechanical_center"]["x_mm"], 100.0
            )
            self.assertAlmostEqual(manifest["points"][0]["camera_position"]["x_mm"], 100.0)
            self.assertAlmostEqual(manifest["points"][0]["camera_position"]["y_mm"], 180.0)
            self.assertAlmostEqual(manifest["points"][0]["camera_position"]["z_mm"], 300.0)
            self.assertAlmostEqual(
                manifest["points"][0]["camera1_position"]["x_mm"], 133.55
            )
            self.assertNotIn("camera1_position", manifest["points"][1])
            self.assertEqual(
                [photo["filename"] for photo in manifest["photos"]],
                ["0_001.jpg", "1_001.jpg", "2_002.jpg"],
            )

        camera = calculate_camera_position(
            [100, 200, 300, 180, 0, 90],
            {
                "offset_mm": 20,
                "angle_reference": "world_negative_y",
                "positive_rotation": "counter_clockwise_from_above",
            },
        )
        self.assertAlmostEqual(camera["x_mm"], 120.0, places=6)
        self.assertAlmostEqual(camera["y_mm"], 200.0, places=6)

    def test_camera_snapshot_retries_transient_read_failure(self):
        class Frame:
            size = 1

        class FlakyCapture:
            def __init__(self):
                self.calls = 0

            def read(self):
                self.calls += 1
                if self.calls <= 2:
                    return False, None
                return True, Frame()

        class FakeCv2:
            @staticmethod
            def imwrite(path, frame):
                Path(path).write_bytes(b"fresh-frame")
                return True

        capture = FlakyCapture()
        pool = CameraSourcePool()
        pool._captures = {2: capture}
        pool._cv2 = FakeCv2()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "2_001.jpg"
            self.assertTrue(pool.snapshot(2, output))
            self.assertTrue(output.is_file())
        self.assertGreaterEqual(capture.calls, 6)

    def test_bundled_action_and_offscreen_ui(self):
        path = ROOT / "Preset Trajectories" / "task1.py"
        spec = importlib.util.spec_from_file_location("task1_action_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["task1_action_test"] = module
        spec.loader.exec_module(module)
        task = normalize_action_task(module.build_action())
        actions = task["actions"]
        self.assertEqual(
            sum(step["type"] == "record_point" for step in actions), 45
        )
        self.assertEqual(sum(step["type"] == "capture" for step in actions), 50)
        at_negative_y = module.camera_position_from_pose([100, 200, 300, 0, 0, 0])
        self.assertAlmostEqual(at_negative_y["x_mm"], 100.0, places=6)
        self.assertAlmostEqual(at_negative_y["y_mm"], 180.0, places=6)
        at_positive_x = module.camera_position_from_pose([100, 200, 300, 0, 0, 90])
        self.assertAlmostEqual(at_positive_x["x_mm"], 120.0, places=6)
        self.assertAlmostEqual(at_positive_x["y_mm"], 200.0, places=6)
        first_motion = next(
            index
            for index, step in enumerate(actions)
            if step["type"] in {"move_joints", "move_xyzr"}
        )
        self.assertEqual(
            [step["source"] for step in actions[:first_motion] if step["type"] == "capture"],
            [0, 1, 2],
        )
        xyzr_moves = [step for step in actions if step["type"] == "move_xyzr"]
        self.assertEqual(sum(step["z_mm"] == -5.0 for step in xyzr_moves), 20)
        self.assertEqual(sum(step["z_mm"] == 5.0 for step in xyzr_moves), 20)
        self.assertTrue(
            all(
                actions[index - 1] == {"type": "wait", "seconds": 2.0}
                for index, step in enumerate(actions)
                if step["type"] == "record_point"
            )
        )
        transfers = [step for step in actions if step["type"] == "move_joints"]
        self.assertEqual(len(transfers), 4)
        self.assertTrue(all("require_current_j3_mm" in step for step in transfers))

        point_sequence = 0
        capture_points = {0: [], 1: [], 2: []}
        for step in actions:
            if step["type"] == "record_point":
                point_sequence += 1
            elif step["type"] == "capture":
                capture_points[step["source"]].append(point_sequence)
        self.assertEqual(point_sequence, 45)
        self.assertEqual(capture_points[0], [1])
        self.assertEqual(capture_points[1], [1, 12, 23, 34])
        self.assertEqual(capture_points[2], list(range(1, 46)))

        camera1_j4_zero = module.camera1_position_from_state(
            [0.0, 0.0, -45.0, 0.0],
            [400.0, 0.0, -45.0, 180.0, 0.0, -90.0],
        )
        camera1_j4_rotated = module.camera1_position_from_state(
            [0.0, 0.0, -45.0, 123.0],
            [400.0, 0.0, -45.0, 180.0, 0.0, 33.0],
        )
        self.assertAlmostEqual(camera1_j4_zero["x_mm"], 433.55, places=6)
        self.assertAlmostEqual(camera1_j4_zero["y_mm"], 0.0, places=6)
        self.assertAlmostEqual(camera1_j4_zero["z_mm"], -45.0, places=6)
        self.assertEqual(camera1_j4_zero, camera1_j4_rotated)

        widget = ScaraControlWidget(owns_controller=False)
        self.assertEqual(widget._btn_import_action.text(), "导入动作")
        self.assertEqual(widget._btn_run_action.text(), "执行动作")
        self.assertFalse(widget._btn_run_action.isEnabled())
        self.assertFalse(hasattr(widget, "_camera_angle_n"))
        widget.cleanup()

    def test_task1_manifest_marks_only_four_source1_points(self):
        path = ROOT / "Preset Trajectories" / "task1.py"
        spec = importlib.util.spec_from_file_location("task1_manifest_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["task1_manifest_test"] = module
        spec.loader.exec_module(module)
        task = normalize_action_task(module.build_action())

        class FakeTask1Controller:
            def __init__(self):
                self.joints = list(task["actions"][0]["joints"])
                self._refresh_pose()

            def _refresh_pose(self):
                j1_rad = module.math.radians(self.joints[0])
                forearm_rad = module.math.radians(self.joints[0] + self.joints[1])
                x_mm = (
                    module.SCARA_LINK1_MM * module.math.cos(j1_rad)
                    + module.SCARA_LINK2_MM * module.math.cos(forearm_rad)
                )
                y_mm = (
                    module.SCARA_LINK1_MM * module.math.sin(j1_rad)
                    + module.SCARA_LINK2_MM * module.math.sin(forearm_rad)
                )
                rz_deg = self.joints[0] + self.joints[1] + self.joints[3] - 90.0
                self.pose = [x_mm, y_mm, self.joints[2], 180.0, 0.0, rz_deg]

            def read_all_sync(self):
                return {"joints": list(self.joints), "pose": list(self.pose)}

            def goto_joints_sync(self, name, joints, should_stop=None, tolerance=0.2):
                self.joints = list(joints)
                self._refresh_pose()
                return True

            def move_xyzr_sync(
                self, name, *, x_mm, y_mm, z_mm, r_deg, should_stop=None
            ):
                self.pose[0] += x_mm
                self.pose[1] += y_mm
                self.pose[2] += z_mm
                self.pose[5] += r_deg
                self.joints[2] += z_mm
                self.joints[3] += r_deg
                return True

            def emergency_stop(self):
                raise AssertionError("测试中不应触发急停")

        def save_photo(source, photo_path):
            Path(photo_path).write_bytes(f"source-{source}".encode())
            return True

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "260812120000"
            worker = ActionWorker(
                FakeTask1Controller(),
                task,
                output,
                snapshot_source=save_photo,
                camera_position_calculator=module.camera_position_from_pose,
                source_position_calculators={1: module.camera1_position_from_state},
            )
            worker._interruptible_wait = lambda seconds: True
            worker.run()
            manifest = json.loads((output / "points.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(len(manifest["points"]), 45)
        self.assertEqual(len(manifest["photos"]), 50)
        annotated = [
            point for point in manifest["points"] if "camera1_position" in point
        ]
        self.assertEqual([point["sequence"] for point in annotated], [1, 12, 23, 34])
        self.assertEqual(
            [
                photo["point_sequence"]
                for photo in manifest["photos"]
                if photo["source"] == 1
            ],
            [1, 12, 23, 34],
        )
        for point in annotated:
            joints = point["joints"]
            centre = point["mechanical_center"]
            expected = module.camera1_position_from_state(
                [
                    joints["J1_deg"],
                    joints["J2_deg"],
                    joints["J3_mm"],
                    joints["J4_deg"],
                ],
                [
                    centre["x_mm"],
                    centre["y_mm"],
                    centre["z_mm"],
                    centre["Rx_deg"],
                    centre["Ry_deg"],
                    centre["Rz_deg"],
                ],
            )
            for key in ("x_mm", "y_mm", "z_mm"):
                self.assertAlmostEqual(
                    point["camera1_position"][key], expected[key], places=9
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
