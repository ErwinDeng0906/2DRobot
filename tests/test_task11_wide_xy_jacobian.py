"""Offline contract tests for Task11 and its wide runtime metadata."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.pipeline.kinematics import fk_wrist, rz_of
from scara.vision.wide_xy_jacobian_runtime import (
    WideXYImageJacobianCalibrationRuntime,
)
from scara.vision import wide_xy_jacobian_runtime as runtime_module
from scara.vision.wide_xy_jacobian import REQUIRED_WIDE_XY_JACOBIAN_QUALITY_GATES


def load_task11():
    path = ROOT / "Preset Trajectories" / "task11.py"
    spec = importlib.util.spec_from_file_location("task11_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Task11ContractTests(unittest.TestCase):
    def test_route_has_330_camera1_frames_and_no_z_do_or_vacuum(self) -> None:
        task11 = load_task11()
        anchor = [30.6646, 84.7845, -27.0046, -4.6268]
        task = task11.build_action_for_preset("P22 float", anchor)
        kinds = [step["type"] for step in task["actions"]]
        self.assertEqual(kinds.count("record_point"), 330)
        self.assertEqual(kinds.count("capture"), 330)
        self.assertEqual(len(task11.VISITS), 66)
        self.assertTrue(
            all(
                step.get("source") == 1
                for step in task["actions"]
                if step["type"] == "capture"
            )
        )
        self.assertFalse(
            any(
                step["type"] in {"move_xyzr", "runtime_move_joints"}
                for step in task["actions"]
            )
        )
        encoded = repr(task["actions"]).lower()
        self.assertNotIn("vacuum", encoded)
        self.assertNotIn("do_", encoded)

    def test_every_route_endpoint_and_controller_axis_transient_is_bounded(self) -> None:
        task11 = load_task11()
        current = [30.6646, 84.7845, -27.0046, -4.6268]
        anchor_rz = rz_of(current[0], current[1], current[3])
        task = task11.build_action_for_preset("P22 float", current)
        maximum_endpoint_step = 0.0
        maximum_axis_leg = 0.0
        maximum_rz_departure = 0.0
        for step in task["actions"]:
            if step["type"] != "move_joints":
                continue
            target = [float(value) for value in step["joints"]]
            self.assertAlmostEqual(target[2], current[2], places=9)
            self.assertLessEqual(
                abs(rz_of(target[0], target[1], target[3]) - anchor_rz),
                0.001,
            )
            current_xy = fk_wrist(current[0], current[1])
            target_xy = fk_wrist(target[0], target[1])
            maximum_endpoint_step = max(
                maximum_endpoint_step,
                math.dist(current_xy, target_xy),
            )
            states = [
                list(current),
                [target[0], current[1], current[2], current[3]],
                [target[0], target[1], current[2], current[3]],
                [target[0], target[1], target[2], current[3]],
                list(target),
            ]
            for left, right in zip(states, states[1:]):
                maximum_axis_leg = max(
                    maximum_axis_leg,
                    math.dist(
                        fk_wrist(left[0], left[1]),
                        fk_wrist(right[0], right[1]),
                    ),
                )
            maximum_rz_departure = max(
                maximum_rz_departure,
                *[
                    abs((rz_of(s[0], s[1], s[3]) - anchor_rz + 180) % 360 - 180)
                    for s in states
                ],
            )
            current = target
        self.assertLessEqual(maximum_endpoint_step, 1.0001)
        self.assertLessEqual(
            maximum_axis_leg, task11.MAX_SEQUENTIAL_TRANSIENT_XY_MM
        )
        self.assertLessEqual(
            maximum_rz_departure, task11.MAX_SEQUENTIAL_TRANSIENT_RZ_DEG
        )
        self.assertTrue(
            all(
                abs(left - right) <= 1e-9
                for left, right in zip(current, [30.6646, 84.7845, -27.0046, -4.6268])
            )
        )

    def test_task11_name_parser_is_bound_to_locked_visit_sequence(self) -> None:
        task11 = load_task11()
        fake = SimpleNamespace(visits=list(task11.VISITS), _last_task11_metadata=None)
        parsed = WideXYImageJacobianCalibrationRuntime.parse_point_name(
            fake,
            "TASK11|target=P22|phase=train|pass=1|visit=01|"
            "dx=+0.000|dy=+0.000|frame=01/05",
        )
        self.assertEqual(parsed, ("P22", (0.0, 0.0), 1, 5))
        self.assertEqual(fake._last_task11_metadata["visit_index"], 1)
        with self.assertRaisesRegex(RuntimeError, "锁定visit序列"):
            WideXYImageJacobianCalibrationRuntime.parse_point_name(
                fake,
                "TASK11|target=P22|phase=train|pass=1|visit=01|"
                "dx=+5.000|dy=+0.000|frame=01/05",
            )


class Task11RuntimeInstallTests(unittest.TestCase):
    @staticmethod
    def _runtime(root: Path, run: Path):
        runtime = SimpleNamespace(
            output_dir=run,
            manifest_path=run / "points.json",
            project_root=root,
            visits=[("train", 1, [0.0, 0.0])],
            frames_per_offset=1,
            training_offsets_xy_mm=[[0.0, 0.0]],
            validation_offsets_xy_mm=[[0.0, 0.0]],
            _records=[
                {
                    "phase": "train",
                    "pass_index": 1,
                    "command_offset_xy_mm": [0.0, 0.0],
                    "image_error_px": [1.0, 2.0],
                    "accepted": True,
                }
            ],
            _processing_failed=False,
            _anchor_robot_xy_mm=np.asarray([118.0, 272.0]),
            _fatal_messages=[],
        )
        runtime._load_manifest = lambda: json.loads(
            runtime.manifest_path.read_text("utf-8")
        )
        runtime._base_report_wide = lambda status, message, fit: {
            "status": status,
            "message": message,
            "fit": fit,
        }
        runtime._enrich_manifest_wide = lambda manifest, report: manifest.update(
            {"task11_wide_xy_jacobian": {"status": report["status"]}}
        )
        runtime._write_markdown_wide = lambda report: None
        return runtime

    def test_installs_only_when_every_wide_gate_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "Trajectory Photos/260815010000"
            run.mkdir(parents=True)
            (run / "points.json").write_text("{}", encoding="utf-8")
            good_fit = {
                "status": "success",
                "quality_gates": {
                    name: {"passed": True}
                    for name in REQUIRED_WIDE_XY_JACOBIAN_QUALITY_GATES
                },
            }
            runtime = self._runtime(root, run)
            with patch.object(
                runtime_module,
                "fit_wide_xy_image_model",
                return_value=good_fit,
            ):
                WideXYImageJacobianCalibrationRuntime.on_task_finished(
                    runtime, True, "ok", str(run)
                )
            installed = root / "src/scara/calib/camera1_wide_xy_jacobian.json"
            installed_bytes = installed.read_bytes()

            bad_fit = json.loads(json.dumps(good_fit))
            failed_name = next(iter(REQUIRED_WIDE_XY_JACOBIAN_QUALITY_GATES))
            bad_fit["quality_gates"][failed_name]["passed"] = False
            bad_runtime = self._runtime(root, run)
            with (
                patch.object(
                    runtime_module,
                    "fit_wide_xy_image_model",
                    return_value=bad_fit,
                ),
                self.assertRaises(RuntimeError),
            ):
                WideXYImageJacobianCalibrationRuntime.on_task_finished(
                    bad_runtime, True, "second", str(run)
                )
            self.assertEqual(installed.read_bytes(), installed_bytes)


if __name__ == "__main__":
    unittest.main()
