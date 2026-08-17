from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scara.pipeline.kinematics import rz_of, solve_joints
from scara.vision.handeye_interaction import sha256_file
from scara.vision.stage7b_session import RESULT_FILENAME, Stage7BSession
from scara.vision.wide_xy_jacobian import REQUIRED_WIDE_XY_JACOBIAN_QUALITY_GATES
from scara.vision.xy_image_jacobian import REQUIRED_XY_JACOBIAN_QUALITY_GATES
from scara.ui.action_worker import ActionWorker, normalize_action_task
from tests import test_stage7a_action_framework as action_helpers


ANCHOR = [118.333, 272.783]
REFERENCE = [30.6646, 84.7845, -27.0046, -4.6268]
RZ = rz_of(REFERENCE[0], REFERENCE[1], REFERENCE[3])


def write_calibrations(root: Path) -> None:
    calib = root / "src/scara/calib"
    calib.mkdir(parents=True)
    J = [[3.65, 1.84], [1.86, -3.76]]
    fine = {
        "status": "success",
        "anchor_target_name": "P22",
        "valid_target_names": ["P22"],
        "camera": {
            "source_index": 1,
            "resolution": {"width": 1280, "height": 720},
        },
        "coordinate_definition": {
            "anchor_robot_xy_mm": ANCHOR,
            "offset_extent_mm": 2.0,
            "imaging_j3_mm": REFERENCE[2],
            "rz_deg": RZ,
        },
        "fit": {
            "status": "success",
            "quality_gates": {name: {"passed": True} for name in REQUIRED_XY_JACOBIAN_QUALITY_GATES},
            "j_error_px_per_command_mm": J,
            "j_command_mm_per_error_px": np.linalg.inv(J).tolist(),
        },
    }
    fine_path = calib / "camera1_xy_image_jacobian.json"
    fine_path.write_text(json.dumps(fine), encoding="utf-8")
    intrinsics_path = calib / "camera1_intrinsics.json"
    geometry_path = calib / "tray_board_geometry.json"
    intrinsics_path.write_text("{}", encoding="utf-8")
    geometry_path.write_text("{}", encoding="utf-8")
    wide = {
        "status": "success",
        "anchor_target_name": "P22",
        "valid_target_names": ["P22"],
        "camera": {
            "source_index": 1,
            "resolution": {"width": 1280, "height": 720},
        },
        "locked_inputs": {
            "local_jacobian_sha256": sha256_file(fine_path),
            "camera_intrinsics_sha256": sha256_file(intrinsics_path),
            "tray_geometry_sha256": sha256_file(geometry_path),
            "suction_target_sha256": "D" * 64,
        },
        "coordinate_definition": {
            "anchor_robot_xy_mm": ANCHOR,
            "imaging_j3_mm": REFERENCE[2],
            "rz_deg": RZ,
            "command_frame": "robot_controller_world_XY",
            "image_error": "slot_pixel_distorted - suction_target_pixel_distorted",
            "wide_extent_mm": 10.0,
            "fine_model_switch_each_axis_mm": 2.0,
        },
        "fit": {
            "status": "success",
            "quality_gates": {name: {"passed": True} for name in REQUIRED_WIDE_XY_JACOBIAN_QUALITY_GATES},
            "selected_model": {
                "model_type": "global_affine",
                "coefficients_feature_by_error": [[2.2, -0.5], [3.65, 1.86], [1.84, -3.76]],
            },
        },
    }
    (calib / "camera1_wide_xy_jacobian.json").write_text(json.dumps(wide), encoding="utf-8")


def make_session(root: Path, run: Path) -> Stage7BSession:
    fine = json.loads(
        (root / "src/scara/calib/camera1_xy_image_jacobian.json").read_text("utf-8")
    )
    with (
        patch(
            "scara.vision.stage7b_session.load_latest_suction_target",
            return_value=SimpleNamespace(source_sha256="D" * 64),
        ),
        patch(
            "scara.vision.stage7b_session.load_local_xy_jacobian",
            return_value=fine,
        ),
    ):
        return Stage7BSession(root, run)


def rows(xy, error):
    joints = solve_joints(xy[0], xy[1], REFERENCE[2], RZ, REFERENCE)
    pose = [xy[0], xy[1], REFERENCE[2], 180, 0, RZ]
    return [
        {
            "measurement_id": f"seq-{index}",
            "accepted": True,
            "target_name": "P22",
            "image_error_px": error,
            "current_robot_xy_mm": xy,
            "current_joints": joints,
            "current_pose": pose,
            "robot_state_age_s": 0.1,
            "annotated_bgr": np.zeros((40, 60, 3), dtype=np.uint8),
        }
        for index in range(5)
    ]


def request(session: Stage7BSession, xy):
    joints = solve_joints(xy[0], xy[1], REFERENCE[2], RZ, REFERENCE)
    return {
        "request_id": "req-1",
        "request_key": "stage7b_p22_finite_loop",
        "target_name": "P22",
        "calibration_sha256": session.wide_hash,
        "fine_calibration_sha256": session.local_hash,
        "controller_state": {"joints": joints, "pose": [xy[0], xy[1], REFERENCE[2], 180, 0, RZ]},
        "external_safety_gates": {name: True for name in ("controller_connected", "controller_enabled", "alarm_clear", "estop_clear", "soft_estop_clear", "controller_idle")},
    }


class Stage7BSessionTests(unittest.TestCase):
    def test_task_is_runtime_xy_only_and_finite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_calibrations(root)
            session = make_session(root, root / "run")
            task = session.action_task()
            self.assertEqual(len(task["actions"]), 33)
            self.assertEqual({step["type"] for step in task["actions"]}, {"runtime_move_joints"})
            self.assertTrue(
                all(
                    step["max_xy_step_norm_mm"] == 0.75
                    and step["max_xy_axis_mm"] == 0.75
                    for step in task["actions"]
                )
            )
            encoded = json.dumps(task).lower()
            for forbidden in ("move_xyzr", "capture", "vacuum", " do ", "j3/z"):
                self.assertNotIn(forbidden, encoded)

    def test_converged_response_saves_five_images_and_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_calibrations(root)
            run = root / "run"
            run.mkdir()
            session = make_session(root, run)
            response = session.build_response(request(session, ANCHOR), rows(ANCHOR, [0.2, -0.1]))
            self.assertEqual(response["decision"], "complete")
            self.assertEqual(len(list(run.glob("1_*.jpg"))), 5)
            payload = json.loads((run / RESULT_FILENAME).read_text("utf-8"))
            self.assertEqual(payload["status"], "converged")
            self.assertEqual(payload["iteration_count"], 1)
            self.assertFalse(payload["iterations"][0]["motion_authorized"])
            self.assertIn("已经抵达距离目标点1mm以内", response["reason"])

    def test_wide_move_response_locks_both_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_calibrations(root)
            run = root / "run"
            run.mkdir()
            session = make_session(root, run)
            xy = [ANCHOR[0] + 5, ANCHOR[1]]
            response = session.build_response(request(session, xy), rows(xy, [20, 8]))
            self.assertEqual(response["decision"], "approve")
            proposal = response["proposal"]
            self.assertEqual(proposal["model_tier"], "coarse_task11")
            self.assertEqual(proposal["wide_calibration_sha256"], session.wide_hash)
            self.assertEqual(proposal["fine_calibration_sha256"], session.local_hash)
            self.assertAlmostEqual(
                np.linalg.norm(proposal["commanded_correction_xy_mm"]),
                0.74,
                places=9,
            )
            self.assertEqual(len(response["target_joints"]), 4)

    def test_session_and_actual_worker_contract_close_one_loop_offline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_calibrations(root)
            run = root / "run"
            session = make_session(root, run)
            controller = action_helpers.FakeController()
            task = normalize_action_task(session.action_task())
            worker = ActionWorker(controller, task, run)
            request_count = 0

            def dynamic_rows(error):
                xy = list(controller.pose[:2])
                return [
                    {
                        "measurement_id": f"loop-{request_count}-{index}",
                        "accepted": True,
                        "target_name": "P22",
                        "image_error_px": list(error),
                        "current_robot_xy_mm": xy,
                        "current_joints": list(controller.joints),
                        "current_pose": list(controller.pose),
                        "robot_state_age_s": 0.1,
                        "annotated_bgr": np.zeros((20, 30, 3), dtype=np.uint8),
                    }
                    for index in range(5)
                ]

            def respond(request_payload):
                nonlocal request_count
                request_count += 1
                error = (
                    [5.0, 0.0]
                    if request_count == 1
                    else session.iterations[-1]["predicted_error_px"]
                )
                worker.respond_runtime_move_joints(
                    session.build_response(
                        request_payload,
                        dynamic_rows(error),
                    )
                )

            results = []
            worker.runtime_move_joints_requested.connect(respond)
            worker.run_finished.connect(
                lambda ok, message, folder: results.append((ok, message, folder))
            )
            worker.run()
            self.assertTrue(results[0][0])
            self.assertEqual(request_count, 2)
            self.assertEqual(len(controller.goto_calls), 1)
            session.finish(*results[0][:2])
            report = json.loads((run / RESULT_FILENAME).read_text("utf-8"))
            self.assertEqual(report["status"], "converged")
            self.assertEqual(len(report["iterations"]), 2)
            self.assertEqual(len(report["worker_runtime_moves"]), 2)
            self.assertEqual(
                report["worker_runtime_moves"][-1]["status"],
                "session_completed_no_motion",
            )
            manifest = json.loads((run / "points.json").read_text("utf-8"))
            self.assertEqual(len(manifest["stage7b_waypoints"]), 2)
            self.assertIn(
                "predicted_endpoint_xy_mm", manifest["stage7b_waypoints"][0]
            )
            self.assertIn("safety_gates", manifest["stage7b_waypoints"][0])


if __name__ == "__main__":
    unittest.main()
