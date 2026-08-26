from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.camera2_extrinsic_calibration import (
    Camera2ExtrinsicObservation,
    board_pose_from_three_world_points,
    collect_run_directories,
    invert_transform,
    load_board_pose_world,
    rotation_error_deg,
    solve_known_board_extrinsic,
    validate_recorded_robot_state,
)
from scara.pipeline.kinematics import fk_wrist, rz_of


TASK18_PATH = ROOT / "Tasks" / "task18_camera2_extrinsic_capture.py"


def _rotation_x(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rotation_y(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rotation_z(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _transform(rotation: np.ndarray, translation) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(translation, dtype=float)
    return result


def _synthetic_observations() -> tuple[list[Camera2ExtrinsicObservation], np.ndarray, np.ndarray]:
    transform_J4_C2 = _transform(
        _rotation_z(7.0) @ _rotation_y(-2.0) @ _rotation_x(1.5),
        [19.6, -1.8, 42.0],
    )
    transform_W_B = _transform(_rotation_z(13.0) @ _rotation_x(180.0), [145.0, 260.0, -52.0])
    observations = []
    for index in range(15):
        row, column = divmod(index, 5)
        rz = -70.0 + index * 10.0
        transform_W_J4 = _transform(
            _rotation_z(rz) @ _rotation_x(180.0),
            [90.0 + column * 18.0, 175.0 + row * 24.0, -20.0 - row * 7.0],
        )
        transform_W_C2 = transform_W_J4 @ transform_J4_C2
        transform_C2_B = invert_transform(transform_W_C2) @ transform_W_B
        observations.append(
            Camera2ExtrinsicObservation(
                measurement_id=f"synthetic:{index:03d}",
                image_path=f"2_{index:03d}.jpg",
                point_sequence=index + 1,
                captured_at="2026-08-26T12:00:00+01:00",
                transform_W_J4=transform_W_J4,
                transform_C2_B=transform_C2_B,
                charuco_corner_count=30,
                pnp_inlier_count=30,
                pnp_inlier_ratio=1.0,
                reprojection_rms_px=0.2,
                reprojection_max_px=0.4,
            )
        )
    return observations, transform_J4_C2, transform_W_B


def _board_pose(transform_W_B: np.ndarray) -> dict:
    return {
        "transform_W_B": transform_W_B,
        "translation_uncertainty_mm": 0.10,
        "rotation_uncertainty_deg": 0.10,
    }


class Camera2ExtrinsicSolverTests(unittest.TestCase):
    def test_three_measured_points_define_right_handed_board_frame(self) -> None:
        transform, diagnostics = board_pose_from_three_world_points(
            [100.0, 200.0, -50.0],
            [180.0, 200.0, -50.0],
            [100.0, 260.0, -50.0],
        )
        self.assertTrue(np.allclose(np.eye(3), transform[:3, :3]))
        self.assertTrue(np.allclose([100.0, 200.0, -50.0], transform[:3, 3]))
        self.assertAlmostEqual(90.0, diagnostics["raw_xy_angle_deg"])

    def test_three_point_board_frame_rejects_collinear_measurements(self) -> None:
        with self.assertRaisesRegex(ValueError, "共线"):
            board_pose_from_three_world_points(
                [0.0, 0.0, 0.0],
                [50.0, 0.0, 0.0],
                [60.0, 0.0, 0.0],
            )

    def test_known_board_pose_recovers_full_transform(self) -> None:
        observations, expected, transform_W_B = _synthetic_observations()
        result = solve_known_board_extrinsic(
            observations,
            _board_pose(transform_W_B),
        )
        actual = np.asarray(result["transform_J4_C2"], dtype=float)
        self.assertEqual("success", result["status"])
        self.assertTrue(result["installation_allowed"])
        self.assertLess(np.linalg.norm(actual[:3, 3] - expected[:3, 3]), 1e-8)
        self.assertLess(rotation_error_deg(actual, expected), 1e-6)
        self.assertTrue(all(gate["passed"] for gate in result["quality_gates"].values()))

    def test_single_bad_pose_is_rejected_without_biasing_solution(self) -> None:
        observations, expected, transform_W_B = _synthetic_observations()
        bad = observations[-1]
        corrupt = bad.transform_C2_B.copy()
        corrupt[:3, 3] += [30.0, -20.0, 15.0]
        observations[-1] = Camera2ExtrinsicObservation(
            **{**bad.__dict__, "transform_C2_B": corrupt}
        )
        result = solve_known_board_extrinsic(observations, _board_pose(transform_W_B))
        actual = np.asarray(result["transform_J4_C2"], dtype=float)
        self.assertEqual("success", result["status"])
        self.assertEqual(14, result["metrics"]["inlier_count"])
        self.assertLess(np.linalg.norm(actual[:3, 3] - expected[:3, 3]), 1e-8)
        self.assertFalse(result["samples"][-1]["inlier"])

    def test_pose_diversity_fails_closed(self) -> None:
        observations, _expected, transform_W_B = _synthetic_observations()
        repeated = [observations[0]] * 12
        result = solve_known_board_extrinsic(repeated, _board_pose(transform_W_B))
        self.assertEqual("rejected", result["status"])
        self.assertFalse(result["installation_allowed"])
        self.assertFalse(result["quality_gates"]["minimum_unique_robot_poses"]["passed"])
        self.assertFalse(result["quality_gates"]["world_xy_span"]["passed"])
        self.assertFalse(result["quality_gates"]["j3_span"]["passed"])
        self.assertFalse(result["quality_gates"]["rz_span"]["passed"])

    def test_unmeasured_example_board_pose_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "尚未测量"):
            load_board_pose_world(
                ROOT / "src/scara/calib/camera2_board_pose_world.example.json"
            )

    def test_robot_state_must_match_j4_kinematics(self) -> None:
        j1, j2, j3, j4 = 12.0, -35.0, -42.5, 18.0
        x_mm, y_mm = fk_wrist(j1, j2)
        point = {
            "joints": {
                "J1_deg": j1,
                "J2_deg": j2,
                "J3_mm": j3,
                "J4_deg": j4,
            },
            "mechanical_center": {
                "x_mm": x_mm,
                "y_mm": y_mm,
                "z_mm": j3,
                "Rx_deg": 180.0,
                "Ry_deg": 0.0,
                "Rz_deg": rz_of(j1, j2, j4),
            },
        }
        validate_recorded_robot_state(point)
        point["mechanical_center"]["x_mm"] += 2.0
        with self.assertRaisesRegex(ValueError, "J4正运动学XY"):
            validate_recorded_robot_state(point)

    def test_stationary_frame_jitter_does_not_inflate_pose_diversity(self) -> None:
        observations, _expected, transform_W_B = _synthetic_observations()
        source = observations[0]
        repeated = []
        for index in range(12):
            jittered = source.transform_W_J4.copy()
            jittered[:3, 3] += [0.02 * index, -0.01 * index, 0.005 * index]
            repeated.append(
                Camera2ExtrinsicObservation(
                    **{
                        **source.__dict__,
                        "measurement_id": f"jitter:{index}",
                        "transform_W_J4": jittered,
                    }
                )
            )
        result = solve_known_board_extrinsic(repeated, _board_pose(transform_W_B))
        self.assertEqual(1, result["metrics"]["unique_robot_pose_count"])
        self.assertFalse(
            result["quality_gates"]["minimum_unique_robot_poses"]["passed"]
        )

    def test_run_discovery_only_accepts_task18_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task18 = root / "task18-run"
            unrelated = root / "task17-run"
            task18.mkdir()
            unrelated.mkdir()
            (task18 / "points.json").write_text(
                json.dumps({"task_name": "task18 相机2-J4外参静止采集"}),
                encoding="utf-8",
            )
            (unrelated / "points.json").write_text(
                json.dumps({"task_name": "task17 相机2内参采集"}),
                encoding="utf-8",
            )
            self.assertEqual([task18.resolve()], collect_run_directories([root]))


class Task18CaptureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("task18_camera2_extrinsic", TASK18_PATH)
        assert spec is not None and spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.action = cls.module.build_action()

    def test_task_is_stationary_and_has_no_outputs(self) -> None:
        kinds = [step["type"] for step in self.action["actions"]]
        forbidden = {
            "move_joints",
            "move_xyzr",
            "runtime_move_joints",
            "set_do",
            "remember_start_joints",
            "return_to_start_joints",
        }
        self.assertFalse(forbidden.intersection(kinds))
        self.assertEqual(5, kinds.count("record_point"))
        self.assertEqual(10, kinds.count("capture"))

    def test_each_robot_state_has_camera1_and_camera2_frames(self) -> None:
        sources = [
            step["source"]
            for step in self.action["actions"]
            if step["type"] == "capture"
        ]
        self.assertEqual([1, 2] * 5, sources)
        self.assertEqual(
            {1: {"auto_exposure": True}, 2: {"auto_exposure": True}},
            self.action["camera_capture_settings"],
        )

    def test_task_contains_required_action_contract_fields(self) -> None:
        self.assertEqual(1, self.module.ACTION_API_VERSION)
        self.assertTrue(self.action["name"])
        self.assertTrue(self.action["description"])
        self.assertIn("camera_model", self.action)
        self.assertIn("actions", self.action)
        position = self.module.camera_position_from_pose([1, 2, 3, 180, 0, 10])
        self.assertEqual("j4_reference_only_camera2_extrinsic_not_yet_known", position["status"])


if __name__ == "__main__":
    unittest.main()
