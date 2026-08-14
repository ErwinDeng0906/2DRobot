from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

from PyQt6.QtCore import QCoreApplication


STAGE_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _modules():
    task7 = _load(
        "task7_under_test",
        STAGE_ROOT / "Preset Trajectories" / "task7.py",
    )
    action_worker = _load(
        "action_worker_under_test",
        STAGE_ROOT / "src" / "scara" / "ui" / "action_worker.py",
    )
    return task7, action_worker


def _task(task7):
    task7._load_start_preset = lambda: [0.0, 90.0, 0.0, 0.0]
    task = task7.build_action()
    for step in task["actions"]:
        if step["type"] == "wait":
            step["seconds"] = 0.0
    return task


def test_task7_builds_three_by_three_zigzag_and_checkpoint():
    task7, action_worker = _modules()
    targets = task7.build_grid_targets([0.0, 90.0, 0.0, 0.0])
    assert len(targets) == 9
    assert [
        (target["grid_u_mm"], target["grid_v_mm"])
        for target in targets
    ] == [
        (-60.0, -60.0),
        (0.0, -60.0),
        (60.0, -60.0),
        (60.0, 0.0),
        (0.0, 0.0),
        (-60.0, 0.0),
        (-60.0, 60.0),
        (0.0, 60.0),
        (60.0, 60.0),
    ]
    # Rotation preserves the requested square and 60 mm grid spacing.
    for first, second in zip(targets, targets[1:]):
        distance = (
            (second["offset_x_mm"] - first["offset_x_mm"]) ** 2
            + (second["offset_y_mm"] - first["offset_y_mm"]) ** 2
        ) ** 0.5
        assert abs(distance - 60.0) < 1e-9

    task = action_worker.normalize_action_task(_task(task7))
    assert sum(step["type"] == "record_point" for step in task["actions"]) == 9
    assert sum(step["type"] == "capture" for step in task["actions"]) == 9
    checkpoint = task["actions"][-1]
    assert checkpoint["type"] == "operator_checkpoint"
    assert checkpoint["continue_text"] == "继续采集"
    assert checkpoint["finish_text"] == "结束采集"
    assert checkpoint["repeat_from_index"] == 1
    movements = [
        step for step in task["actions"] if step["type"] == "move_xyzr"
    ]
    assert movements
    assert all(step["z_mm"] == 0.0 and step["r_deg"] == 0.0 for step in movements)
    assert movements[-1]["name"] == "返回标定中心 X 并暂停"


class _FakeController:
    def read_all_sync(self):
        return {
            "joints": [0.0, 90.0, 0.0, 0.0],
            "pose": [225.0, 175.0, 0.0, 0.0, 0.0, 0.0],
        }

    def move_xyzr_sync(self, _name, **_kwargs):
        return True

    def emergency_stop(self):
        return None


def test_worker_repeats_in_one_manifest_then_finishes(tmp_path):
    task7, action_worker = _modules()
    task = _task(task7)
    output_dir = tmp_path / "run"

    def snapshot(_source: int, path: Path) -> bool:
        path.write_bytes(b"test-jpg-placeholder")
        return True

    worker = action_worker.ActionWorker(
        _FakeController(),
        task,
        output_dir,
        snapshot_source=snapshot,
    )
    decisions = iter((True, False))
    worker.operator_checkpoint_requested.connect(
        lambda *_args: worker.respond_operator_checkpoint(next(decisions))
    )
    result = []
    worker.run_finished.connect(lambda ok, message, folder: result.append((ok, message, folder)))
    app = QCoreApplication.instance() or QCoreApplication([])
    worker.start()
    deadline = time.monotonic() + 10.0
    while worker.isRunning() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    worker.wait(1000)
    app.processEvents()

    assert not worker.isRunning(), "worker did not finish after two checkpoint decisions"
    assert result and result[0][0] is True
    manifest = json.loads((output_dir / "points.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert len(manifest["points"]) == 18
    assert len(manifest["photos"]) == 18
    assert [point["collection_round"] for point in manifest["points"]] == [
        *([1] * 9),
        *([2] * 9),
    ]
    assert [row["decision"] for row in manifest["operator_checkpoints"]] == [
        "continue",
        "finish",
    ]
    assert manifest["photos"][0]["filename"] == "1_001.jpg"
    assert manifest["photos"][-1]["filename"] == "1_018.jpg"
