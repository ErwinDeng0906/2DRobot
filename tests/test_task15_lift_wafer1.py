from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.ui.action_worker import ActionWorker, normalize_action_task  # noqa: E402


TASK_PATH = ROOT / "Tasks" / "task15_lift wafer1.py"


def load_task_module():
    spec = importlib.util.spec_from_file_location("task15_lift_wafer1", TASK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Task15")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeController:
    def __init__(self, *, fail_move_number: int | None = None) -> None:
        self.joints = [30.0, 80.0, 10.0, -5.0]
        self.pose = [120.0, 270.0, 10.0, 180.0, 0.0, 20.0]
        self.fail_move_number = fail_move_number
        self.moves: list[dict] = []
        self.do_writes: list[tuple[int, int]] = []
        self.zero_calls: list[list[int]] = []
        self.estop_calls: list[list[int]] = []

    def is_connected(self) -> bool:
        return True

    def read_all_sync(self) -> dict:
        return {
            "joints": list(self.joints),
            "pose": list(self.pose),
            "effectively_enabled": True,
            "enable": 1,
            "estop": False,
            "soft_estop": False,
            "warn": 0,
            "need_clear": False,
            "mode": "T1",
        }

    def move_xyzr_sync(self, name: str, **kwargs) -> bool:
        self.moves.append({"name": name, **kwargs})
        if self.fail_move_number == len(self.moves):
            return False
        delta_z = float(kwargs["z_mm"])
        self.joints[2] += delta_z
        self.pose[2] += delta_z
        return True

    def set_do_sync(self, channel: int, level: int) -> tuple[bool, str]:
        self.do_writes.append((int(channel), int(level)))
        return True, "ok"

    def zero_do_channels_sync(self, channels: list[int]) -> tuple[bool, str]:
        self.zero_calls.append(list(channels))
        return True, "zeroed"

    def emergency_stop(self, do_channels=None) -> None:
        self.estop_calls.append(list(do_channels or []))


class Task15DefinitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_task_module()
        cls.task = normalize_action_task(cls.module.build_action())

    def test_exact_requested_sequence_and_zero_net_z(self) -> None:
        actions = self.task["actions"]
        self.assertEqual(
            [step["type"] for step in actions],
            [
                "move_xyzr",
                "record_point",
                "wait",
                "set_do",
                "move_xyzr",
                "record_point",
                "wait",
                "move_xyzr",
                "record_point",
                "set_do",
                "set_do",
                "wait",
                "move_xyzr",
                "record_point",
                "set_do",
            ],
        )
        moves = [step for step in actions if step["type"] == "move_xyzr"]
        self.assertEqual([step["z_mm"] for step in moves], [-23.3, 23.3, -23.3, 23.3])
        self.assertAlmostEqual(sum(step["z_mm"] for step in moves), 0.0, places=12)
        self.assertTrue(
            all(
                step["x_mm"] == step["y_mm"] == step["r_deg"] == 0.0
                for step in moves
            )
        )
        waits = [step["seconds"] for step in actions if step["type"] == "wait"]
        self.assertEqual(waits, [1.0, 2.0, 5.0])
        writes = [
            (step["channel"], step["level"])
            for step in actions
            if step["type"] == "set_do"
        ]
        self.assertEqual(writes, [(1, 1), (1, 0), (2, 1), (2, 0)])

    def test_set_do_schema_is_strict(self) -> None:
        raw = self.module.build_action()
        for invalid_channel in (0, 17, True, 1.5):
            changed = json.loads(json.dumps(raw))
            next(step for step in changed["actions"] if step["type"] == "set_do")[
                "channel"
            ] = invalid_channel
            with self.assertRaises(ValueError):
                normalize_action_task(changed)
        for invalid_level in (-1, 2, True, 0.5):
            changed = json.loads(json.dumps(raw))
            next(step for step in changed["actions"] if step["type"] == "set_do")[
                "level"
            ] = invalid_level
            with self.assertRaises(ValueError):
                normalize_action_task(changed)


class Task15WorkerTests(unittest.TestCase):
    def run_worker(self, controller: FakeController, output: Path):
        task = normalize_action_task(load_task_module().build_action())
        worker = ActionWorker(controller, task, output)
        worker._interruptible_wait = lambda _seconds: True
        results = []
        worker.run_finished.connect(lambda ok, message, folder: results.append((ok, message, folder)))
        worker.run()
        self.assertEqual(len(results), 1)
        return results[0], json.loads((output / "points.json").read_text(encoding="utf-8"))

    def test_success_executes_exactly_four_moves_and_four_do_writes(self) -> None:
        controller = FakeController()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            result, manifest = self.run_worker(controller, output)
        self.assertTrue(result[0], result[1])
        self.assertEqual(
            [move["z_mm"] for move in controller.moves],
            [-23.3, 23.3, -23.3, 23.3],
        )
        self.assertEqual(controller.do_writes, [(1, 1), (1, 0), (2, 1), (2, 0)])
        self.assertEqual(controller.zero_calls, [])
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(len(manifest["points"]), 4)
        self.assertEqual(
            [event["status"] for event in manifest["do_events"]],
            ["completed"] * 4,
        )
        self.assertFalse(manifest["do_cleanup"]["required"])
        self.assertAlmostEqual(controller.pose[2], 10.0, places=12)

    def test_failure_after_pump_on_clears_touched_output(self) -> None:
        controller = FakeController(fail_move_number=2)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            result, manifest = self.run_worker(controller, output)
        self.assertFalse(result[0])
        self.assertEqual(controller.do_writes, [(1, 1)])
        self.assertEqual(controller.zero_calls, [[1]])
        self.assertEqual(manifest["status"], "stopped")
        self.assertTrue(manifest["do_cleanup"]["required"])
        self.assertTrue(manifest["do_cleanup"]["passed"])

    def test_stop_requests_estop_and_declared_output_clear(self) -> None:
        controller = FakeController()
        task = normalize_action_task(load_task_module().build_action())
        with tempfile.TemporaryDirectory() as temporary:
            worker = ActionWorker(controller, task, Path(temporary) / "run")
            worker.request_stop()
        self.assertEqual(controller.estop_calls, [[1, 2]])


if __name__ == "__main__":
    unittest.main()
