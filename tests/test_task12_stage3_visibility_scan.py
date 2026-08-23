from __future__ import annotations

import importlib.util
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK12_PATH = ROOT / "Tasks" / "task12_code visibility scan.py"


def _load_task12():
    spec = importlib.util.spec_from_file_location("task12_visibility_scan", TASK12_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage3_record(passed: bool) -> dict:
    return {
        "success": True,
        "quality_passed": bool(passed),
        "failure_reason": None if passed else "外围Marker不足：2/3",
        "visible_marker_ids": [1, 3, 5] if passed else [1, 3],
        "used_marker_ids": [1, 3, 5] if passed else [1, 3],
        "rejected_marker_ids": [],
        "ransac_inlier_corner_count": 12 if passed else 8,
        "object_span_mm": 100.0,
        "reprojection_rms_px": 1.5,
        "per_marker_rms_px": {"1": 1.4, "3": 1.5, "5": 1.6} if passed else {},
        "rvec_C_T": [0.0, 0.0, 0.0],
        "tvec_C_T_mm": [0.0, 0.0, 300.0],
        "T_C_T": (
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 300.0], [0.0, 0.0, 0.0, 1.0]]
            if passed else None
        ),
        "T_T_C": None,
        "camera_position_T_mm": [0.0, 0.0, -300.0],
        "minimum_object_depth_C_mm": 250.0,
    }


class Task12ActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.task12 = _load_task12()
        cls.action = cls.task12.build_action()

    def test_closed_route_has_all_slots_and_only_adjacent_edges(self) -> None:
        route = list(self.task12.TARGET_ORDER)
        self.assertEqual(36, len(route))
        self.assertEqual({f"P{r}{c}" for r in range(6) for c in range(6)}, set(route))
        closed = route + ["P00"]
        for left, right in zip(closed, closed[1:]):
            lr, lc = int(left[1]), int(left[2])
            rr, rc = int(right[1]), int(right[2])
            self.assertEqual(1, abs(lr - rr) + abs(lc - rc), (left, right))

    def test_action_collects_20_frames_at_every_slot_without_z_do_or_vacuum(self) -> None:
        actions = self.action["actions"]
        kinds = [row["type"] for row in actions]
        self.assertEqual(720, kinds.count("record_point"))
        self.assertEqual(720, kinds.count("capture"))
        self.assertEqual(
            {"assert_joints", "move_joints", "wait", "record_point", "capture"},
            set(kinds),
        )
        names = [row["name"] for row in actions if row["type"] == "record_point"]
        for target in self.task12.TARGET_ORDER:
            self.assertEqual(20, sum(name.startswith(f"TASK12|{target}|") for name in names))
        self.assertNotIn("runtime_move_joints", kinds)
        self.assertNotIn("set_do", kinds)

    def test_all_moves_hold_j3_rz_and_audited_sequential_transient(self) -> None:
        from scara.pipeline.kinematics import rz_of

        actions = self.action["actions"]
        start = next(row for row in actions if row["type"] == "assert_joints")["joints"]
        required_j3 = float(start[2])
        required_rz = float(rz_of(start[0], start[1], start[3]))
        previous = list(start)
        maximum_xy = 0.0
        maximum_rz = 0.0
        moves = [row for row in actions if row["type"] == "move_joints"]
        self.assertGreater(len(moves), 1000)
        for move in moves:
            target = move["joints"]
            self.assertAlmostEqual(required_j3, float(target[2]), places=9)
            self.assertLessEqual(
                abs((rz_of(target[0], target[1], target[3]) - required_rz + 180.0) % 360.0 - 180.0),
                0.01,
            )
            xy_step, rz_departure = self.task12._sequential_motion_audit(
                previous, target, required_rz
            )
            maximum_xy = max(maximum_xy, xy_step)
            maximum_rz = max(maximum_rz, rz_departure)
            previous = list(target)
        self.assertLessEqual(maximum_xy, self.task12.MAX_SEQUENTIAL_TRANSIENT_XY_MM + 1e-9)
        self.assertLessEqual(maximum_rz, self.task12.MAX_SEQUENTIAL_TRANSIENT_RZ_DEG + 1e-9)
        for actual, expected in zip(previous, start):
            self.assertAlmostEqual(float(expected), float(actual), places=9)


class VisibilitySummaryTests(unittest.TestCase):
    def test_readiness_requires_complete_set_and_80_percent_combined_pass(self) -> None:
        from scara.vision.stage3_visibility_scan import summarize_slot_visibility

        rows = []
        for index in range(1, 21):
            passed = index <= 16
            rows.append(
                {
                    "filename": f"1_{index:03d}.jpg",
                    "frame_index": index,
                    "stage3": _stage3_record(passed),
                    "temporal_quality": {"accepted_by_tracker": passed},
                    "processing_error": None,
                }
            )
        summary = summarize_slot_visibility(
            "P00", [0.0, 0.0, -2.0], rows, expected_frames=20
        )
        self.assertTrue(summary["ready_for_stage7"])
        self.assertEqual("ready", summary["classification"])
        self.assertEqual(16, summary["combined_pass_count"])
        self.assertAlmostEqual(0.8, summary["combined_pass_rate_of_expected"])

        incomplete = summarize_slot_visibility(
            "P00", [0.0, 0.0, -2.0], rows[:-1], expected_frames=20
        )
        self.assertFalse(incomplete["ready_for_stage7"])
        self.assertFalse(incomplete["quality_gates"]["frame_set_complete"]["passed"])

    def test_completed_scan_with_one_blind_slot_writes_diagnostic_not_failure(self) -> None:
        from scara.vision.stage3_visibility_scan import (
            RESULT_FILENAME,
            Stage3VisibilityScanRuntime,
            UPDATE_FILENAME,
        )

        geometry_source = ROOT / "src/scara/calib/tray_board_geometry.json"
        geometry = json.loads(geometry_source.read_text(encoding="utf-8-sig"))
        targets = tuple(f"P{row}{column}" for row in range(6) for column in range(6))
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = Path(temporary_text)
            project = temporary / "project"
            calib = project / "src/scara/calib"
            calib.mkdir(parents=True)
            shutil.copyfile(geometry_source, calib / "tray_board_geometry.json")
            (calib / "camera1_intrinsics.json").write_text(
                json.dumps(
                    {
                        "status": "success",
                        "K": [[620.0, 0.0, 640.0], [0.0, 620.0, 360.0], [0.0, 0.0, 1.0]],
                        "distCoeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
                        "image_resolution": {"width": 1280, "height": 720},
                        "global_rms_px": 0.5,
                    }
                ),
                encoding="utf-8",
            )
            output = temporary / "run"
            output.mkdir()
            runtime = Stage3VisibilityScanRuntime(
                output,
                project,
                targets,
                geometry["slots"],
                20,
                confirm_safety=False,
            )

            points = []
            photos = []
            sequence = 0
            for target in targets:
                for frame_index in range(1, 21):
                    sequence += 1
                    filename = f"1_{sequence:03d}.jpg"
                    passed = target != "P25"
                    runtime._records_by_filename[filename] = {
                        "filename": filename,
                        "point_sequence": sequence,
                        "target_name": target,
                        "frame_index": frame_index,
                        "known_point_T_mm": geometry["slots"][target],
                        "stage3": _stage3_record(passed),
                        "temporal_quality": {
                            "accepted_by_tracker": passed,
                            "tracker_reason": None if passed else "raw pose rejected",
                            "filtered_T_C_T": _stage3_record(True)["T_C_T"] if passed else None,
                        },
                        "processing_error": None,
                    }
                    points.append(
                        {
                            "sequence": sequence,
                            "name": f"TASK12|{target}|frame={frame_index:02d}/20",
                        }
                    )
                    photos.append(
                        {
                            "source": 1,
                            "sequence_for_source": sequence,
                            "filename": filename,
                            "point_sequence": sequence,
                        }
                    )
            (output / "points.json").write_text(
                json.dumps({"points": points, "photos": photos}), encoding="utf-8"
            )
            runtime.on_task_finished(True, "动作完成", str(output))

            result = json.loads((output / RESULT_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual("visibility_gaps_detected", result["status"])
            self.assertEqual(35, result["summary"]["ready_slot_count"])
            self.assertEqual(["P25"], result["summary"]["not_observable_slots"])
            self.assertTrue((output / UPDATE_FILENAME).is_file())
            enriched = json.loads((output / "points.json").read_text(encoding="utf-8"))
            self.assertEqual("P00", enriched["points"][0]["task12_target_name"])
            self.assertIn("stage3_visibility_scan", enriched)


if __name__ == "__main__":
    unittest.main()
