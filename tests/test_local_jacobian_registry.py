"""Offline contracts for selected-slot Task9 calibration and file naming."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.pipeline.kinematics import fk_wrist
from scara.ui.control_widget import ScaraControlWidget
from scara.vision.full_tray_positioning import slot_world_xy_mm
from scara.vision.local_jacobian_registry import (
    local_jacobian_filename,
    local_jacobian_relative_path,
    validate_slot_name,
)
from scara.vision.tray_pose_estimator import load_tray_board_geometry


def _load_task9():
    path = PROJECT_ROOT / "Tasks/task9_jacobiantest.py"
    spec = importlib.util.spec_from_file_location("_task9_selected_slot_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Task9")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LocalJacobianRegistryTests(unittest.TestCase):
    def test_p22_stays_at_legacy_path_and_other_slots_are_named(self) -> None:
        self.assertEqual(
            local_jacobian_relative_path("P22").as_posix(),
            "src/scara/calib/camera1_xy_image_jacobian.json",
        )
        self.assertEqual(
            local_jacobian_relative_path("P31").as_posix(),
            "src/scara/calib/Jacobians/camera1_xy_image_jacobian_P31.json",
        )
        self.assertEqual(
            local_jacobian_filename("P31"),
            "camera1_xy_image_jacobian_P31.json",
        )
        with self.assertRaises(ValueError):
            validate_slot_name("P61")

    def test_selected_slot_task9_uses_geometry_anchor_and_target_names(self) -> None:
        task9 = _load_task9()
        target = "P31"
        label, anchor, point_T = task9.load_target_anchor(target)
        geometry = load_tray_board_geometry(
            PROJECT_ROOT / "src/scara/calib/tray_board_geometry.json"
        )
        expected_xy = slot_world_xy_mm(geometry, target)
        self.assertIn(target, label)
        self.assertEqual(point_T, list(geometry["slots"][target]))
        self.assertLess(
            ((fk_wrist(anchor[0], anchor[1])[0] - expected_xy[0]) ** 2
             + (fk_wrist(anchor[0], anchor[1])[1] - expected_xy[1]) ** 2) ** 0.5,
            0.02,
        )

        task = task9.build_action_for_target(target)
        point_steps = [
            step for step in task["actions"] if step["type"] == "record_point"
        ]
        capture_steps = [
            step for step in task["actions"] if step["type"] == "capture"
        ]
        self.assertEqual(len(point_steps), 108)
        self.assertEqual(len(capture_steps), 108)
        self.assertTrue(
            all(f"TASK9|target={target}|" in step["name"] for step in point_steps)
        )
        self.assertEqual(task["actions"][0]["type"], "assert_joints")
        self.assertEqual(task["actions"][0]["joints"], anchor)
        self.assertFalse(
            any(
                step["type"] in {"move_cart", "set_do", "vacuum", "start_video"}
                for step in task["actions"]
            )
        )

    def test_all_36_slot_plans_pass_offline_ik_and_sequential_audit(self) -> None:
        task9 = _load_task9()
        for row in range(6):
            for column in range(6):
                target = f"P{row}{column}"
                task = task9.build_action_for_target(target)
                self.assertIn(target, task["name"])
                self.assertEqual(
                    sum(step["type"] == "capture" for step in task["actions"]),
                    108,
                )

    def test_control_widget_one_shot_locks_selected_target_and_restores_import(self) -> None:
        captured: list[dict] = []
        original_builder = lambda: {"name": "original", "actions": []}

        class Button:
            def __init__(self) -> None:
                self.enabled = False

            def setEnabled(self, enabled: bool) -> None:  # noqa: N802
                self.enabled = bool(enabled)

        fake = SimpleNamespace(
            _action_worker=None,
            _one_shot_action_restore=None,
            _action_file=Path("original.py"),
            _action_builder=original_builder,
            _action_camera_calculator=None,
            _action_source_position_calculators={1: object()},
            _action_runtime_factory=None,
            _action_task={"name": "original"},
            _btn_run_action=Button(),
            _append=lambda *_args: None,
        )
        fake._restore_one_shot_action = MethodType(
            ScaraControlWidget._restore_one_shot_action, fake
        )

        def fake_run_action() -> None:
            captured.append(fake._action_builder())

        fake._on_run_action = fake_run_action
        ScaraControlWidget._start_local_jacobian_calibration(fake, "P31")

        self.assertEqual(len(captured), 1)
        record_names = [
            step["name"]
            for step in captured[0]["actions"]
            if step["type"] == "record_point"
        ]
        self.assertEqual(len(record_names), 108)
        self.assertTrue(all("target=P31" in name for name in record_names))
        self.assertIs(fake._action_builder, original_builder)
        self.assertIsNone(fake._one_shot_action_restore)


if __name__ == "__main__":
    unittest.main()
