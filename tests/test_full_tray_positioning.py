from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.pipeline.kinematics import rz_of, solve_joints
from scara.ui.action_worker import ActionWorker, normalize_action_task
from scara.vision.full_tray_positioning import (
    build_metric_geometry_correction,
    metric_suction_error_in_tray,
    plan_geometry_coarse_route,
    slot_world_xy_mm,
    tray_delta_to_world_xy_mm,
)
from scara.vision.full_tray_positioning_session import (
    FULL_TRAY_GEOMETRY_REQUEST_KEY,
    LegacyFixedTrayPositioningSession as FullTrayPositioningSession,
)
from scara.vision.handeye_interaction import sha256_file
from tests.test_stage7b_session import ANCHOR, REFERENCE, RZ, write_calibrations
from tests.test_stage7a_action_framework import FakeController


GEOMETRY_PATH = ROOT / "src/scara/calib/tray_board_geometry.json"


def geometry() -> dict:
    return json.loads(GEOMETRY_PATH.read_text(encoding="utf-8"))


def pose_at(xy: list[float] | np.ndarray) -> tuple[list[float], list[float]]:
    values = [float(xy[0]), float(xy[1])]
    joints = solve_joints(
        values[0], values[1], REFERENCE[2], rz_deg=RZ, ref_joints=REFERENCE
    )
    assert joints is not None
    return joints, [values[0], values[1], REFERENCE[2], 180.0, 0.0, RZ]


def runtime_request(
    state_xy: list[float],
    geometry_hash: str,
    *,
    requested_at: float = 100.0,
) -> dict:
    joints, pose = pose_at(state_xy)
    return {
        "request_id": "geometry-request-1",
        "request_key": FULL_TRAY_GEOMETRY_REQUEST_KEY,
        "target_name": "P22",
        "requested_monotonic_s": requested_at,
        "calibration_sha256": geometry_hash,
        "controller_state": {"joints": joints, "pose": pose},
        "limits": {
            "anchor_robot_xy_mm": ANCHOR,
            "local_extent_mm": 5.0,
            "domain_margin_mm": 0.2,
            "required_j3_mm": REFERENCE[2],
            "j3_tolerance_mm": 0.2,
            "required_rz_deg": RZ,
            "rz_tolerance_deg": 0.3,
            "target_rz_tolerance_deg": 0.15,
            "max_xy_step_norm_mm": 3.0,
            "max_xy_axis_mm": 3.0,
            "max_sequential_transient_xy_mm": 6.0,
            "max_sequential_transient_rz_deg": 1.0,
            "precompensate_rz": True,
        },
    }


def metric_samples(
    state_xy: list[float],
    error_T: list[float],
    *,
    requested_at: float = 100.0,
) -> list[dict]:
    joints, pose = pose_at(state_xy)
    return [
        {
            "measurement_id": f"metric-{index}",
            "captured_monotonic_s": requested_at + 0.1 + index * 0.01,
            "accepted": True,
            "target_name": "P22",
            "metric_error_T_mm": list(error_T),
            "current_robot_xy_mm": list(state_xy),
            "current_joints": list(joints),
            "current_pose": list(pose),
            "robot_state_age_s": 0.1,
            "annotated_bgr": np.zeros((30, 40, 3), dtype=np.uint8),
        }
        for index in range(5)
    ]


class FullTrayPureMathTests(unittest.TestCase):
    def test_stage3_stage4_metric_equations(self) -> None:
        transform = np.eye(4)
        result = metric_suction_error_in_tray(
            transform,
            [-48.0, -49.0, -2.0],
            [-50.0, -50.0, -2.0],
        )
        np.testing.assert_allclose(
            result["suction_point_T_mm"], [-48.0, -49.0, -2.0]
        )
        np.testing.assert_allclose(result["metric_error_T_mm"], [-2.0, -1.0, 0.0])

    def test_11_by_11_full_tray_grid_plans_to_geometry_p22(self) -> None:
        payload = geometry()
        target = slot_world_xy_mm(payload, "P22")
        frame = payload["tray_frame"]
        rotation = np.asarray(
            frame["rotation_mechanical_from_tray"], dtype=np.float64
        )[:2, :2]
        origin = np.asarray(frame["origin_mechanical_xy_mm"], dtype=np.float64)
        for tray_x in np.linspace(-125.0, 0.0, 11):
            for tray_y in np.linspace(-125.0, 0.0, 11):
                start = origin + rotation @ np.asarray([tray_x, tray_y])
                joints, pose = pose_at(start)
                route = plan_geometry_coarse_route(
                    joints,
                    pose,
                    payload,
                    "P22",
                    required_j3_mm=REFERENCE[2],
                    required_rz_deg=RZ,
                )
                np.testing.assert_allclose(
                    route["geometry_target_world_xy_mm"], target, atol=2e-4
                )
                self.assertTrue(route["waypoints"])
                self.assertLessEqual(
                    max(
                        row["endpoint_step_norm_mm"]
                        for row in route["waypoints"]
                    ),
                    2.01,
                )
                self.assertLessEqual(
                    max(
                        row["sequential_transient_max_mm"]
                        for row in route["waypoints"]
                    ),
                    4.0 + 1e-9,
                )
                self.assertLessEqual(
                    max(
                        row["sequential_transient_rz_max_deg"]
                        for row in route["waypoints"]
                    ),
                    1.0 + 1e-9,
                )
                for row in route["waypoints"]:
                    self.assertAlmostEqual(
                        row["target_joints"][2], REFERENCE[2], places=6
                    )
                    self.assertAlmostEqual(
                        rz_of(
                            row["target_joints"][0],
                            row["target_joints"][1],
                            row["target_joints"][3],
                        ),
                        RZ,
                        places=3,
                    )

    def test_metric_correction_enters_task9_anchor(self) -> None:
        payload = geometry()
        current = slot_world_xy_mm(payload, "P22").astype(float)
        desired_world_command = np.asarray(ANCHOR) - current
        rotation = np.asarray(
            payload["tray_frame"]["rotation_mechanical_from_tray"],
            dtype=np.float64,
        )[:2, :2]
        error_T = (rotation.T @ desired_world_command).astype(float).tolist() + [0.0]
        report = build_metric_geometry_correction(
            metric_samples(current.tolist(), error_T),
            runtime_request(current.tolist(), "A" * 64),
            payload,
            target_name="P22",
            transition_anchor_xy_mm=ANCHOR,
        )
        self.assertTrue(report["motion_authorized"], report["failure_reasons"])
        np.testing.assert_allclose(
            report["commanded_correction_xy_mm"], desired_world_command, atol=1e-9
        )
        np.testing.assert_allclose(
            report["predicted_endpoint_xy_mm"], ANCHOR, atol=1e-6
        )
        self.assertTrue(all(gate["passed"] for gate in report["safety_gates"].values()))


class FullTraySessionTests(unittest.TestCase):
    @staticmethod
    def _project(root: Path) -> dict:
        write_calibrations(root)
        geometry_target = root / "src/scara/calib/tray_board_geometry.json"
        shutil.copy2(GEOMETRY_PATH, geometry_target)
        wide_path = root / "src/scara/calib/camera1_wide_xy_jacobian.json"
        wide = json.loads(wide_path.read_text(encoding="utf-8"))
        wide["locked_inputs"]["tray_geometry_sha256"] = sha256_file(
            geometry_target
        )
        wide_path.write_text(json.dumps(wide), encoding="utf-8")
        return json.loads(geometry_target.read_text(encoding="utf-8"))

    def _session(self, root: Path, run: Path, start_name: str = "P00") -> FullTrayPositioningSession:
        payload = self._project(root)
        start = slot_world_xy_mm(payload, start_name).astype(float).tolist()
        joints, pose = pose_at(start)
        fine = json.loads(
            (root / "src/scara/calib/camera1_xy_image_jacobian.json").read_text(
                encoding="utf-8"
            )
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
            return FullTrayPositioningSession(
                root,
                run,
                {
                    "joints": joints,
                    "pose": pose,
                    "captured_monotonic_s": time.monotonic(),
                },
                target_name="P22",
            )

    def test_task_has_geometry_route_one_metric_request_then_task9_loop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = self._session(root, root / "run")
            task = normalize_action_task(session.action_task())
            actions = task["actions"]
            self.assertEqual(actions[0]["type"], "assert_joints")
            geometry_requests = [
                row
                for row in actions
                if row["type"] == "runtime_move_joints"
                and row["request_key"] == FULL_TRAY_GEOMETRY_REQUEST_KEY
            ]
            self.assertEqual(len(geometry_requests), 1)
            fine_requests = [
                row
                for row in actions
                if row["type"] == "runtime_move_joints"
                and row["request_key"] == "stage7b_p22_finite_loop"
            ]
            self.assertEqual(len(fine_requests), 33)
            self.assertEqual(
                sum(row["type"] == "move_joints" for row in actions),
                session.coarse_route["waypoint_count"],
            )
            encoded = json.dumps(task, ensure_ascii=False).lower()
            for forbidden in ("move_xyzr", "capture", "vacuum", "start_video", "stop_video"):
                self.assertNotIn(forbidden, encoded)
            # ActionWorker must remain the sole creator of a new run folder.
            self.assertFalse(session.report_path.exists())
            self.assertEqual(len(session.all_slot_world_xy_mm), 36)

    def test_metric_response_is_worker_compatible_and_saves_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            session = self._session(root, run)
            current = np.asarray(session.geometry_target_world_xy_mm)
            desired = np.asarray(session.fine_session.anchor_xy) - current
            rotation = np.asarray(
                session.geometry["tray_frame"]["rotation_mechanical_from_tray"]
            )[:2, :2]
            error_T = (rotation.T @ desired).astype(float).tolist() + [0.0]
            request = runtime_request(
                current.astype(float).tolist(), session.geometry_hash
            )
            response = session.build_response(
                request,
                metric_samples(current.astype(float).tolist(), error_T),
            )
            self.assertEqual(response["decision"], "approve")
            self.assertEqual(response["calibration_sha256"], session.geometry_hash)
            self.assertEqual(len(response["target_joints"]), 4)
            self.assertTrue(response["proposal"]["motion_authorized"])
            self.assertEqual(len(list(run.glob("1_*.jpg"))), 5)

    def test_actual_action_worker_reaudits_and_executes_geometry_response_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            session = self._session(root, run)
            task = normalize_action_task(session.action_task())
            step = next(
                row
                for row in task["actions"]
                if row["type"] == "runtime_move_joints"
                and row["request_key"] == FULL_TRAY_GEOMETRY_REQUEST_KEY
            )
            current = np.asarray(session.geometry_target_world_xy_mm)
            desired = np.asarray(session.fine_session.anchor_xy) - current
            rotation = np.asarray(
                session.geometry["tray_frame"]["rotation_mechanical_from_tray"]
            )[:2, :2]
            error_T = (rotation.T @ desired).astype(float).tolist() + [0.0]
            controller = FakeController()
            controller.joints, controller.pose = pose_at(current.tolist())
            worker = ActionWorker(controller, task, run)

            def respond(request: dict) -> None:
                worker.respond_runtime_move_joints(
                    session.build_response(
                        request,
                        metric_samples(
                            current.tolist(),
                            error_T,
                            requested_at=float(request["requested_monotonic_s"]),
                        ),
                    )
                )

            worker.runtime_move_joints_requested.connect(respond)
            worker._runtime_move_joints(step)
            # J4 precompensation is normally unnecessary here; exactly one XY
            # endpoint move must be executed and independently verified.
            self.assertEqual(len(controller.goto_calls), 1)
            np.testing.assert_allclose(controller.pose[:2], ANCHOR, atol=2e-3)
            move = worker._manifest["runtime_moves"][0]
            self.assertEqual(move["status"], "motion_completed")
            self.assertTrue(move["fresh_kinematic_audit"]["passed"])

    def test_complete_worker_run_owns_directory_and_reaches_local_convergence(self) -> None:
        """Exercise the real run lifecycle, not only the runtime-move method."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            session = self._session(root, run, start_name="P00")
            self.assertFalse(run.exists())
            task = normalize_action_task(session.action_task())
            controller = FakeController()
            start = session.coarse_route["initial_world_xy_mm"]
            controller.joints, controller.pose = pose_at(start)
            worker = ActionWorker(controller, task, run)
            finished: list[tuple[bool, str, str]] = []

            def respond(request: dict) -> None:
                requested = float(request["requested_monotonic_s"])
                current_xy = [
                    float(request["controller_state"]["pose"][0]),
                    float(request["controller_state"]["pose"][1]),
                ]
                if request["request_key"] == FULL_TRAY_GEOMETRY_REQUEST_KEY:
                    desired = np.asarray(session.fine_session.anchor_xy) - np.asarray(
                        current_xy
                    )
                    rotation = np.asarray(
                        session.geometry["tray_frame"][
                            "rotation_mechanical_from_tray"
                        ]
                    )[:2, :2]
                    error_T = (rotation.T @ desired).astype(float).tolist() + [0.0]
                    samples = metric_samples(
                        current_xy, error_T, requested_at=requested
                    )
                else:
                    joints = list(request["controller_state"]["joints"])
                    pose = list(request["controller_state"]["pose"])
                    samples = [
                        {
                            "measurement_id": f"fine-{index}",
                            "captured_monotonic_s": requested + 0.1 + index * 0.01,
                            "accepted": True,
                            "target_name": "P22",
                            "image_error_px": [0.2, -0.1],
                            "current_robot_xy_mm": current_xy,
                            "current_joints": joints,
                            "current_pose": pose,
                            "robot_state_age_s": 0.1,
                            "annotated_bgr": np.zeros(
                                (30, 40, 3), dtype=np.uint8
                            ),
                        }
                        for index in range(5)
                    ]
                worker.respond_runtime_move_joints(
                    session.build_response(request, samples)
                )

            def finish(ok: bool, message: str, output_dir: str) -> None:
                finished.append((ok, message, output_dir))
                session.finish(ok, message)

            worker.runtime_move_joints_requested.connect(respond)
            worker.run_finished.connect(finish)
            worker.run()

            self.assertEqual(len(finished), 1)
            self.assertTrue(finished[0][0], finished[0][1])
            self.assertEqual(session.status, "converged")
            report = json.loads(session.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "converged")
            self.assertEqual(len(report["slot_world_xy_mm"]), 36)
            self.assertEqual(report["currently_authorized_target_names"], ["P22"])
            self.assertEqual(len(list(run.glob("1_*.jpg"))), 10)
            manifest = json.loads((run / "points.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(
                manifest["full_tray_positioning"]["status"], "converged"
            )
            self.assertEqual(
                len(controller.goto_calls),
                session.coarse_route["waypoint_count"] + 1,
            )

    def test_non_p22_is_fail_closed_even_though_geometry_has_36_slots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = self._project(root)
            start = slot_world_xy_mm(payload, "P00").astype(float).tolist()
            joints, pose = pose_at(start)
            fine = json.loads(
                (root / "src/scara/calib/camera1_xy_image_jacobian.json").read_text(
                    encoding="utf-8"
                )
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
                with self.assertRaisesRegex(RuntimeError, "只开放P22"):
                    FullTrayPositioningSession(
                        root,
                        root / "run",
                        {
                            "joints": joints,
                            "pose": pose,
                            "captured_monotonic_s": time.monotonic(),
                        },
                        target_name="P21",
                    )


if __name__ == "__main__":
    unittest.main()
