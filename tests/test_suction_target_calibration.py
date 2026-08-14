"""Offline tests for the Stage-4 fixed-plane suction target core and task."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.suction_target_calibration import (
    SuctionCalibrationQualityConfig,
    aggregate_location_poses,
    fit_suction_target,
)
from scara.vision.suction_target_calibration_runtime import (
    RESULT_FILENAME,
    UPDATE_FILENAME,
    SuctionTargetCalibrationRuntime,
)
from PyQt6.QtCore import QObject


def _intrinsics() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array(
            [[618.6, 0.0, 635.5], [0.0, 619.9, 355.9], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        np.zeros((5, 1), dtype=np.float64),
    )


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return result


class SuctionLocationAggregationTests(unittest.TestCase):
    def test_batch_filter_rejects_one_pose_and_recovers_candidate(self) -> None:
        base_rvec = np.array([[math.pi], [0.03], [-0.02]])
        base_rotation, _ = cv2.Rodrigues(base_rvec)
        base_translation = np.array([75.0, 40.0, 390.0])
        point_T = np.array([-50.0, -50.0, -2.0])
        expected = base_rotation @ point_T + base_translation
        rng = np.random.default_rng(20260814)
        transforms = []
        for _index in range(20):
            jitter_rvec = rng.normal(0.0, math.radians(0.03), (3, 1))
            jitter_rotation, _ = cv2.Rodrigues(jitter_rvec)
            transforms.append(
                _transform(
                    base_rotation @ jitter_rotation,
                    base_translation + rng.normal(0.0, 0.05, 3),
                )
            )
        transforms[7] = _transform(
            base_rotation,
            base_translation + np.array([8.0, -6.0, 5.0]),
        )
        result = aggregate_location_poses(
            "P22", point_T, transforms, list(range(1, 21))
        )
        self.assertTrue(result["success"], result["failure_reason"])
        self.assertIn(8, result["batch_rejected_frame_indices"])
        np.testing.assert_allclose(
            result["suction_candidate_C_mm"], expected, atol=0.15
        )

    def test_requires_enough_stage3_accepted_frames(self) -> None:
        result = aggregate_location_poses(
            "P00",
            [0.0, 0.0, -2.0],
            [np.eye(4) for _ in range(5)],
            list(range(1, 6)),
        )
        self.assertFalse(result["success"])
        self.assertIn("only 5/12", result["failure_reason"])


class SuctionFitTests(unittest.TestCase):
    def test_fit_and_leave_one_location_out_validation(self) -> None:
        K, dist = _intrinsics()
        true_point = np.array([34.0, -21.0, 310.0])
        rng = np.random.default_rng(81)
        locations = []
        for row in range(3):
            for column in range(3):
                candidate = true_point + rng.normal(0.0, 0.08, 3)
                locations.append(
                    {
                        "target_name": f"P{row}{column}",
                        "success": True,
                        "suction_candidate_C_mm": candidate.tolist(),
                    }
                )
        result = fit_suction_target(locations, K, dist)
        self.assertEqual(result["status"], "success", result["failure_reasons"])
        np.testing.assert_allclose(result["p_C_S_mm"], true_point, atol=0.1)
        self.assertEqual(len(result["cross_validation"]["rows"]), 9)
        self.assertTrue(all(result["quality_gates"][key]["passed"] for key in result["quality_gates"]))

    def test_cross_validation_rejects_one_inconsistent_location(self) -> None:
        K, dist = _intrinsics()
        locations = [
            {
                "target_name": f"P{index:02d}",
                "success": True,
                "suction_candidate_C_mm": [30.0, -10.0, 300.0],
            }
            for index in range(8)
        ]
        locations.append(
            {
                "target_name": "bad",
                "success": True,
                "suction_candidate_C_mm": [36.0, -5.0, 300.0],
            }
        )
        result = fit_suction_target(locations, K, dist)
        self.assertEqual(result["status"], "rejected_quality")
        self.assertIn("cross_validation_xy_max", result["failure_reasons"])


class Task8ActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = PROJECT_ROOT / "Preset Trajectories" / "task8_suction calib.py"
        spec = importlib.util.spec_from_file_location("task8_test_module", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.module = module

    def test_builds_nine_twenty_frame_bursts_at_fixed_height_and_rz(self) -> None:
        presets = {}
        for index, target in enumerate(self.module.TARGET_ORDER):
            j1 = 15.0 + index
            j2 = 70.0 - index * 0.5
            j4 = 110.82 - j1 - j2
            presets[target] = [j1, j2, -27.0119, j4]
        original = self.module.load_target_presets
        self.module.load_target_presets = lambda: (
            presets,
            {target: f"{target} float" for target in self.module.TARGET_ORDER},
        )
        try:
            task = self.module.build_action()
        finally:
            self.module.load_target_presets = original
        self.assertEqual(
            sum(step["type"] == "record_point" for step in task["actions"]), 180
        )
        self.assertEqual(
            sum(step["type"] == "capture" for step in task["actions"]), 180
        )
        self.assertFalse(
            any(step["type"] == "move_xyzr" for step in task["actions"])
        )
        for step in task["actions"]:
            if step["type"] == "move_joints":
                self.assertAlmostEqual(step["joints"][2], -27.0119, places=4)


class Task8RuntimePersistenceTests(unittest.TestCase):
    def test_finalizer_enriches_manifest_and_writes_json_and_markdown(self) -> None:
        quality = SuctionCalibrationQualityConfig(
            frames_per_location=3,
            minimum_accepted_frames_per_location=2,
            minimum_fit_locations=6,
        )
        targets = {
            f"P{row}{column}": [-25.0 * row, -25.0 * column, -2.0]
            for row in range(3)
            for column in range(3)
        }
        presets = {}
        for index, target in enumerate(targets):
            j1 = 20.0 + index
            j2 = 70.0 - index
            presets[target] = [j1, j2, -27.0119, 110.82 - j1 - j2]
        K, dist = _intrinsics()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            runtime = SuctionTargetCalibrationRuntime.__new__(
                SuctionTargetCalibrationRuntime
            )
            QObject.__init__(runtime)
            runtime.output_dir = output
            runtime.project_root = PROJECT_ROOT
            runtime.target_points = targets
            runtime.target_presets = presets
            runtime.quality = quality
            runtime.intrinsics_path = Path("camera1_intrinsics.json")
            runtime.geometry_path = Path("tray_board_geometry.json")
            runtime.intrinsics_hash = "A" * 64
            runtime.geometry_hash = "B" * 64
            runtime.intrinsics = SimpleNamespace(
                K=K,
                dist_coeffs=dist,
                image_size=(1280, 720),
            )
            runtime.geometry = {
                "schema_version": 2,
                "tray_frame": {
                    "slot_target_plane_z_T_mm": -2.0,
                    "marker_plane_j3_mm": -50.0119,
                    "slot_bottom_j3_mm_used_for_height_difference": -52.0119,
                },
            }
            runtime._records = []
            points = []
            photos = []
            sequence = 0
            true_p_C_S = np.array([32.0, -18.0, 310.0])
            for target, point_T in targets.items():
                point_T_array = np.asarray(point_T, dtype=np.float64)
                for frame_index in range(1, quality.frames_per_location + 1):
                    sequence += 1
                    transform = np.eye(4, dtype=np.float64)
                    transform[:3, 3] = true_p_C_S - point_T_array
                    transform[0, 3] += (frame_index - 2) * 0.02
                    filename = f"1_{sequence:03d}.jpg"
                    runtime._records.append(
                        {
                            "filename": filename,
                            "photo_sequence": sequence,
                            "point_sequence": sequence,
                            "target_name": target,
                            "frame_index": frame_index,
                            "known_point_T_mm": point_T,
                            "stage3": {
                                "quality_passed": True,
                                "T_C_T": transform.tolist(),
                            },
                            "temporal_quality": {
                                "accepted_by_tracker": True,
                            },
                        }
                    )
                    points.append(
                        {
                            "sequence": sequence,
                            "name": f"TASK8|{target}|frame={frame_index:02d}/03",
                            "joints": {},
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
                json.dumps({"points": points, "photos": photos}),
                encoding="utf-8",
            )
            runtime.on_task_finished(True, "synthetic complete", str(output))

            enriched = json.loads(
                (output / "points.json").read_text(encoding="utf-8")
            )
            result = json.loads(
                (output / RESULT_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(result["status"], "success")
            np.testing.assert_allclose(
                result["fit"]["p_C_S_mm"], true_p_C_S, atol=0.03
            )
            self.assertEqual(
                len(enriched["stage4_suction_calibration"]["location_aggregates"]),
                9,
            )
            self.assertIn("stage3_pose", enriched["points"][0])
            self.assertIn("known_point_T_mm", enriched["points"][0])
            self.assertTrue((output / UPDATE_FILENAME).is_file())


if __name__ == "__main__":
    unittest.main()
