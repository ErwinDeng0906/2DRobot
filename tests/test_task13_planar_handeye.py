from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.pipeline.kinematics import rz_of
from scara.vision.planar_handeye import DEFAULT_QUALITY
from scara.vision.planar_handeye_runtime import Task13PlanarHandEyeRuntime


def _load_task13_module():
    path = ROOT / "Tasks" / "task13_planar handeye.py"
    spec = importlib.util.spec_from_file_location("task13_planar_handeye_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Task13ActionContractTests(unittest.TestCase):
    def test_operator_authorized_cross_run_limit_is_one_degree(self) -> None:
        self.assertEqual(
            DEFAULT_QUALITY.maximum_cross_run_rotation_difference_deg,
            1.0,
        )

    def test_task13_selects_16_poses_and_has_only_fixed_height_camera1_actions(self) -> None:
        module = _load_task13_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p22_preset = [30.6646, 84.7845, -27.0046, -4.6268]
            (root / "scara_presets.json").write_text(
                json.dumps({"P22 float": p22_preset}), encoding="utf-8"
            )
            geometry_source = ROOT / "src/scara/calib/tray_board_geometry.json"
            geometry_target = root / "src/scara/calib/tray_board_geometry.json"
            geometry_target.parent.mkdir(parents=True)
            geometry_target.write_bytes(geometry_source.read_bytes())
            with patch.object(module, "_root", return_value=root):
                task = module.build_action()
            actions = task["actions"]
            self.assertEqual(sum(step["type"] == "capture" for step in actions), 160)
            self.assertEqual(sum(step["type"] == "record_point" for step in actions), 160)
            self.assertFalse(
                any(step["type"] in {"move_xyzr", "operator_checkpoint"} for step in actions)
            )
            self.assertTrue(
                all(step.get("source") == 1 for step in actions if step["type"] == "capture")
            )
            move_rows = [step for step in actions if step["type"] == "move_joints"]
            self.assertGreater(len(move_rows), 16)
            rz_required = rz_of(*p22_preset[:2], p22_preset[3])
            for move in move_rows:
                joints = move["joints"]
                self.assertAlmostEqual(joints[2], p22_preset[2], places=9)
                self.assertLess(
                    abs(rz_of(joints[0], joints[1], joints[3]) - rz_required),
                    0.001,
                )
            self.assertEqual(actions[0]["type"], "assert_joints")
            self.assertEqual(actions[0]["joints"], p22_preset)
            self.assertEqual(move_rows[-1]["joints"], p22_preset)
            self.assertNotIn("Task12", json.dumps(task, ensure_ascii=False))


class Task13RuntimeInstallTests(unittest.TestCase):
    def test_failed_independent_validation_writes_report_and_never_installs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "Trajectory Photos" / "run"
            output.mkdir(parents=True)
            task8_dir = root / "task8"
            task8_dir.mkdir()
            (task8_dir / "points.json").write_text(
                json.dumps({"points": []}), encoding="utf-8"
            )
            with (
                patch(
                    "scara.vision.planar_handeye_runtime.load_camera_intrinsics",
                    return_value={},
                ),
                patch(
                    "scara.vision.planar_handeye_runtime.load_tray_board_geometry",
                    return_value={},
                ),
                patch(
                    "scara.vision.planar_handeye_runtime.TrayBoardPoseEstimator",
                    return_value=object(),
                ),
            ):
                runtime = Task13PlanarHandEyeRuntime(root, output)
            failures = []
            runtime.fatal_error.connect(failures.append)
            report = {
                "schema_version": 1,
                "status": "failure",
                "quality_gates": {
                    "cross_run_rotation_consistency": {
                        "passed": False,
                        "actual": 0.7,
                        "limit": "<=0.30 deg",
                    }
                },
            }
            with (
                patch.object(runtime, "_enrich_stage3"),
                patch(
                    "scara.vision.planar_handeye_runtime._latest_success",
                    return_value=root / "task8" / "camera1_suction_target.json",
                ),
                patch(
                    "scara.vision.planar_handeye_runtime.fit_planar_handeye",
                    return_value=report,
                ) as fit,
                patch(
                    "scara.vision.planar_handeye_runtime.install_planar_handeye"
                ) as install,
            ):
                runtime.on_task_finished(True, "动作完成", str(output))
            self.assertTrue(runtime.processing_failed)
            self.assertTrue(failures)
            install.assert_not_called()
            source_runs = fit.call_args.args[1]
            self.assertTrue(source_runs[0][0].startswith("task8_"))
            self.assertTrue(source_runs[1][0].startswith("task13_"))
            self.assertNotIn("task12", repr(source_runs).lower())
            saved = json.loads(
                (output / "task13_planar_handeye.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["status"], "failure")
            self.assertFalse(
                (root / "src/scara/calib/camera1_forearm_planar_handeye.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
