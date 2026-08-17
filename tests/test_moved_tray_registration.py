from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.pipeline.kinematics import fk_wrist, forearm_pose_W_F, j4_for_rz
from scara.pipeline.xy_correction_planner import plan_fixed_rz_xy_step
from scara.ui.action_worker import ActionWorker, normalize_action_task
from scara.vision.handeye_interaction import SuctionTargetModel
from scara.vision.moved_tray_positioning_session import MovedTrayPositioningSession
from scara.vision.moved_tray_servo import (
    final_hold_gates,
    registered_slot_world_xy_mm,
    registered_tray_delta_to_world_xy_mm,
)
from scara.vision.planar_handeye import fit_planar_handeye, install_planar_handeye
from scara.vision.runtime_tray_registration import build_runtime_tray_registration
from tests.test_stage7a_action_framework import FakeController


def rz(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    return np.array(
        [[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def transform_C_T(
    *, alpha: float, yaw_W_T: float, point_T: np.ndarray, p_C_S: np.ndarray, R_F_C: np.ndarray
) -> np.ndarray:
    R_C_T = R_F_C.T @ rz(-alpha) @ rz(yaw_W_T)
    transform = np.eye(4)
    transform[:3, :3] = R_C_T
    point_3 = np.array([point_T[0], point_T[1], -2.0])
    transform[:3, 3] = p_C_S - R_C_T @ point_3
    return transform


def geometry(origin=(100.0, 200.0), yaw=20.0) -> dict:
    slots = {
        f"P{row}{column}": [-25.0 * row, -25.0 * column, -2.0]
        for row in range(6)
        for column in range(6)
    }
    rotation = rz(yaw)
    return {
        "schema_version": 2,
        "tray_frame": {
            "origin_mechanical_xy_mm": list(origin),
            "rotation_mechanical_from_tray": rotation.tolist(),
        },
        "slots": slots,
    }


class ForearmPoseTests(unittest.TestCase):
    def test_forearm_pose_uses_j1_plus_j2_not_j4_or_tool_rz(self) -> None:
        pose = np.asarray(forearm_pose_W_F(30.0, 40.0))
        expected = rz(70.0)[:2, :2]
        self.assertTrue(np.allclose(pose[:2, :2], expected, atol=1e-12))
        self.assertTrue(np.allclose(pose[:2, 2], fk_wrist(30.0, 40.0), atol=1e-12))


class PlanarHandEyeFitTests(unittest.TestCase):
    def test_whole_slot_fit_recovers_shared_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            calib = root / "src/scara/calib"
            calib.mkdir(parents=True)
            intrinsics = calib / "camera1_intrinsics.json"
            tray = calib / "tray_board_geometry.json"
            intrinsics.write_text("{}\n", encoding="utf-8")
            tray.write_text(json.dumps(geometry()), encoding="utf-8")
            suction_path = root / "Trajectory Photos/task8/camera1_suction_target.json"
            suction_path.parent.mkdir(parents=True)
            p_C_S = np.array([-2.4, -33.0, 149.5])
            suction = {
                "status": "success",
                "camera": {"source_index": 1, "resolution": {"width": 1280, "height": 720}},
                "coordinate_definition": {"imaging_j3_mm": -27.0},
                "locked_inputs": {
                    "camera_intrinsics_path": str(intrinsics),
                    "camera_intrinsics_sha256": sha(intrinsics),
                    "tray_geometry_path": str(tray),
                    "tray_geometry_sha256": sha(tray),
                },
                "fit": {"p_C_S_mm": p_C_S.tolist()},
            }
            suction_path.write_text(json.dumps(suction), encoding="utf-8")
            R_F_C = rz(-32.0)
            slots = [
                f"P{row}{column}"
                for row in range(6)
                for column in range(6)
                if f"P{row}{column}" not in {"P00", "P01", "P05", "P10", "P25", "P45"}
            ][:24]
            runs = []
            for run_index, (run_name, origin, yaw_W_T) in enumerate(
                [("task8_synth", np.array([130.0, 300.0]), 18.0), ("task12_synth", np.array([132.0, 299.0]), 21.0)]
            ):
                run = root / f"Trajectory Photos/{run_name}"
                run.mkdir(parents=True)
                points = []
                sequence = 0
                for slot_index, slot in enumerate(slots):
                    row, column = int(slot[1]), int(slot[2])
                    point_T = np.array([-25.0 * row, -25.0 * column])
                    alpha = 55.0 + slot_index * 2.3 + run_index * 0.2
                    T_C_T = transform_C_T(
                        alpha=alpha,
                        yaw_W_T=yaw_W_T,
                        point_T=point_T,
                        p_C_S=p_C_S,
                        R_F_C=R_F_C,
                    )
                    world_xy = origin + rz(yaw_W_T)[:2, :2] @ point_T
                    for frame in range(12):
                        sequence += 1
                        points.append(
                            {
                                "sequence": sequence,
                                "name": f"TASK12|{slot}|frame={frame + 1:02d}/12",
                                "joints": {"J1_deg": alpha / 2.0, "J2_deg": alpha / 2.0, "J3_mm": -27.0, "J4_deg": 0.0},
                                "mechanical_center": {"x_mm": world_xy[0], "y_mm": world_xy[1]},
                                "stage3_pose": {"quality_passed": True, "T_C_T": T_C_T.tolist()},
                                "stage3_temporal_quality": {"accepted_by_tracker": True, "filtered_T_C_T": T_C_T.tolist()},
                            }
                        )
                points_path = run / "points.json"
                points_path.write_text(json.dumps({"points": points}), encoding="utf-8")
                runs.append((run_name, points_path))
            report = fit_planar_handeye(root, runs, suction_target_path=suction_path)
            self.assertEqual(report["status"], "success", report["quality_gates"])
            recovered = np.asarray(report["R_F_C"])
            self.assertLess(abs(math.degrees(math.atan2(recovered[1, 0], recovered[0, 0])) + 32.0), 1e-6)
            self.assertTrue(report["validation"]["whole_slot_holdout"])
            self.assertFalse(report["validation"]["random_frame_split_used"])
            self.assertGreaterEqual(report["training"]["independent_pose_count"], 12)
            self.assertTrue(
                all(
                    probe["run_name"] == runs[-1][0]
                    and len(probe["tray_xy_mm"]) == 2
                    for probe in report["prevalidated_probe_poses"]
                )
            )
            installed = install_planar_handeye(
                report, calib / "camera1_forearm_planar_handeye.json"
            )
            self.assertEqual(
                json.loads(installed.read_text(encoding="utf-8"))["status"],
                "success",
            )


class RuntimeRegistrationTests(unittest.TestCase):
    def _samples(self, origin, yaw_W_T, old_origin, old_yaw):
        p_C_S = np.array([-2.4, -33.0, 149.5])
        R_F_C = rz(-32.0)
        alpha = 90.0
        point_T = np.array([-25.0, -30.0])
        T_C_T = transform_C_T(
            alpha=alpha,
            yaw_W_T=yaw_W_T,
            point_T=point_T,
            p_C_S=p_C_S,
            R_F_C=R_F_C,
        )
        world = np.asarray(origin) + rz(yaw_W_T)[:2, :2] @ point_T
        requested = 100.0
        samples = [
            {
                "measurement_id": f"m{i}",
                "captured_monotonic_s": requested + 0.1 + i * 0.01,
                "accepted": True,
                "target_name": "P22",
                "robot_state_age_s": 0.2,
                "used_marker_count": 4,
                "current_joints": [40.0, 50.0, -27.0, 20.0],
                "current_robot_xy_mm": world.tolist(),
                "tray_transform_C_T": T_C_T.tolist(),
                "metric_error_T_mm": [-25.0, -20.0, 0.0],
                "current_pose": [world[0], world[1], -27.0, 180.0, 0.0, 20.0],
            }
            for i in range(5)
        ]
        calibration = {"R_F_C": R_F_C.tolist(), "_source_path": "cal.json", "_source_sha256": "A" * 64, "locked_inputs": {}}
        suction = SuctionTargetModel(Path("suction.json"), "B" * 64, 1, (1280, 720), tuple(p_C_S), (625.0, 218.0), -2.0, -27.0, 20.0)
        geo = geometry(old_origin, old_yaw)
        return samples, calibration, suction, geo, requested

    def test_registration_recovers_translation_rotation_and_dynamic_slot(self) -> None:
        old_origin = np.array([100.0, 200.0])
        origin = old_origin + np.array([4.0, -3.0])
        samples, calibration, suction, geo, requested = self._samples(origin, 23.0, old_origin, 20.0)
        report = build_runtime_tray_registration(
            samples, calibration, suction, geo, requested_monotonic_s=requested
        )
        self.assertEqual(report["status"], "success", report)
        self.assertTrue(np.allclose(report["origin_world_xy_mm"], origin, atol=1e-9))
        self.assertAlmostEqual(report["relative_to_stage2"]["translation_norm_mm"], 5.0, places=9)
        self.assertAlmostEqual(report["relative_to_stage2"]["yaw_deg"], 3.0, places=9)
        expected = origin + rz(23.0)[:2, :2] @ np.array([-50.0, -50.0])
        self.assertTrue(np.allclose(registered_slot_world_xy_mm(geo, "P22", report), expected))
        self.assertTrue(np.allclose(registered_tray_delta_to_world_xy_mm([1.0, 0.0], report), rz(23.0)[:2, 0]))

    def test_registration_hard_scope_fails_closed(self) -> None:
        old = np.array([100.0, 200.0])
        samples, calibration, suction, geo, requested = self._samples(old + [11.0, 0.0], 20.0, old, 20.0)
        report = build_runtime_tray_registration(samples, calibration, suction, geo, requested_monotonic_s=requested)
        self.assertEqual(report["status"], "rejected")
        self.assertFalse(report["hard_gates"]["translation_hard_scope"]["passed"])

    def test_supported_10mm_and_5deg_combination_requires_probe_not_rejection(self) -> None:
        old = np.array([100.0, 200.0])
        samples, calibration, suction, geo, requested = self._samples(
            old + [6.0, 8.0], 25.0, old, 20.0
        )
        report = build_runtime_tray_registration(
            samples, calibration, suction, geo, requested_monotonic_s=requested
        )
        self.assertEqual(report["status"], "requires_three_pose_probe", report)
        self.assertTrue(all(gate["passed"] for gate in report["hard_gates"].values()))
        self.assertAlmostEqual(report["relative_to_stage2"]["translation_norm_mm"], 10.0)
        self.assertAlmostEqual(abs(report["relative_to_stage2"]["yaw_deg"]), 5.0)

    def test_camera_reconnect_promotes_a_normal_registration_to_probe(self) -> None:
        old = np.array([100.0, 200.0])
        samples, calibration, suction, geo, requested = self._samples(
            old + [2.0, 1.0], 21.0, old, 20.0
        )
        report = build_runtime_tray_registration(
            samples,
            calibration,
            suction,
            geo,
            requested_monotonic_s=requested,
            camera_reconnected=True,
        )
        self.assertEqual(report["status"], "requires_three_pose_probe")
        self.assertIn("camera_reconnected", report["anomaly_reasons"])

    def test_final_hold_requires_all_three_one_mm_conditions(self) -> None:
        gates = final_hold_gates(
            {"median_error_norm_mm": 0.55, "window_rms_dispersion_mm": 0.2, "maximum_frame_error_norm_mm": 0.95}
        )
        self.assertTrue(all(gate["passed"] for gate in gates.values()))
        gates = final_hold_gates(
            {"median_error_norm_mm": 0.55, "window_rms_dispersion_mm": 0.2, "maximum_frame_error_norm_mm": 1.01}
        )
        self.assertFalse(gates["all_five_frame_error"]["passed"])


class MovedTrayActionContractTests(unittest.TestCase):
    def test_action_normalizes_with_10mm_endpoint_and_natural_joint_envelope(self) -> None:
        raw = {
            "api_version": 1,
            "name": "moved",
            "camera_model": {"offset_mm": 0.0},
            "actions": [
                {
                    "type": "runtime_move_joints",
                    "name": "window",
                    "request_key": "moved_tray_p22_runtime_positioning",
                    "target_name": "P22",
                    "calibration_sha256": "A" * 64,
                    "anchor_robot_xy_mm": [100.0, 200.0],
                    "local_extent_mm": 130.0,
                    "domain_margin_mm": 5.0,
                    "required_j3_mm": -27.0,
                    "required_rz_deg": 20.0,
                    "max_xy_step_norm_mm": 10.0,
                    "max_xy_axis_mm": 10.0,
                    "j3_tolerance_mm": 0.2,
                    "rz_tolerance_deg": 0.3,
                    "target_rz_tolerance_deg": 0.15,
                    "max_sequential_transient_rz_deg": 15.0,
                    "precompensate_rz": True,
                    "max_state_drift_xy_mm": 0.2,
                    "max_state_drift_joint": 0.2,
                    "max_sequential_transient_xy_mm": 130.0,
                    "move_tolerance": 0.01,
                    "proposal_max_age_s": 8.0,
                    "fk_pose_xy_tolerance_mm": 0.2,
                }
            ],
        }
        normalized = normalize_action_task(raw)
        step = normalized["actions"][0]
        self.assertEqual(step["max_xy_step_norm_mm"], 10.0)
        self.assertEqual(step["max_xy_axis_mm"], 10.0)
        self.assertEqual(step["max_sequential_transient_xy_mm"], 130.0)
        self.assertEqual(step["max_sequential_transient_rz_deg"], 15.0)
        self.assertFalse(step["enforce_sequential_intermediate_domain"])
        self.assertNotIn("fine_calibration_sha256", {"fine_calibration_sha256": step["fine_calibration_sha256"]} if step["fine_calibration_sha256"] else {})
        self.assertFalse(any(action["type"] in {"move_xyzr"} for action in normalized["actions"]))

        raw["actions"][0]["max_xy_step_norm_mm"] = 10.001
        with self.assertRaises(ValueError):
            normalize_action_task(raw)

    def test_worker_executes_one_direct_10mm_endpoint_without_internal_waypoints(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            controller = FakeController()
            start_joints = list(controller.joints)
            start_pose = list(controller.pose)
            raw_step = {
                "type": "runtime_move_joints",
                "name": "moved tray direct coarse endpoint",
                "request_key": "moved_tray_p22_runtime_positioning",
                "target_name": "P22",
                "calibration_sha256": "A" * 64,
                "anchor_robot_xy_mm": [start_pose[0], start_pose[1]],
                "local_extent_mm": 130.0,
                "domain_margin_mm": 5.0,
                "required_j3_mm": start_joints[2],
                "required_rz_deg": start_pose[5],
                "max_xy_step_norm_mm": 10.0,
                "max_xy_axis_mm": 10.0,
                "j3_tolerance_mm": 0.2,
                "rz_tolerance_deg": 0.3,
                "target_rz_tolerance_deg": 0.15,
                "max_sequential_transient_rz_deg": 15.0,
                "precompensate_rz": True,
                "max_state_drift_xy_mm": 0.2,
                "max_state_drift_joint": 0.2,
                "max_sequential_transient_xy_mm": 130.0,
                "move_tolerance": 0.01,
                "proposal_max_age_s": 8.0,
                "fk_pose_xy_tolerance_mm": 0.2,
            }
            normalized = normalize_action_task(
                {
                    "api_version": 1,
                    "name": "moved direct",
                    "camera_model": {"offset_mm": 0.0},
                    "actions": [raw_step],
                }
            )
            step = normalized["actions"][0]
            plan = plan_fixed_rz_xy_step(
                start_joints,
                start_pose,
                [8.0, 6.0],
                anchor_robot_xy_mm=[start_pose[0], start_pose[1]],
                local_extent_mm=130.0,
                domain_margin_mm=5.0,
                required_j3_mm=start_joints[2],
                j3_tolerance_mm=0.2,
                required_rz_deg=start_pose[5],
                rz_tolerance_deg=0.3,
                target_rz_tolerance_deg=0.15,
                max_xy_step_norm_mm=10.000001,
                max_xy_axis_mm=10.000001,
                max_sequential_transient_xy_mm=130.0,
                max_sequential_transient_rz_deg=15.0,
                precompensate_rz=True,
                enforce_sequential_intermediate_domain=False,
            )
            worker = ActionWorker(controller, normalized, Path(td) / "run")
            worker._manifest = {
                "points": [],
                "photos": [],
                "videos": [],
                "runtime_moves": [],
            }
            worker._output_dir.mkdir(parents=True)
            worker._save_manifest()

            def approve(request):
                proposal = {
                    "proposal_id": "MOVED-DIRECT-10MM",
                    "target_name": "P22",
                    "phase": "moved_tray_coarse",
                    "motion_authorized": True,
                    "calculation": {
                        "commanded_correction_xy_mm": [8.0, 6.0]
                    },
                    "planner": plan,
                }
                worker.respond_runtime_move_joints(
                    {
                        "request_id": request["request_id"],
                        "decision": "approve",
                        "calibration_sha256": "A" * 64,
                        "proposal": proposal,
                        "target_joints": plan["target_joints"],
                    }
                )

            worker.runtime_move_joints_requested.connect(approve)
            worker._runtime_move_joints(step)
            self.assertEqual(len(controller.goto_calls), 1)
            self.assertAlmostEqual(
                math.dist(start_pose[:2], controller.pose[:2]), 10.0, places=8
            )
            manifest = json.loads(worker.manifest_path.read_text(encoding="utf-8"))
            move = manifest["runtime_moves"][0]
            self.assertEqual(move["status"], "motion_completed")
            self.assertNotIn("internal_waypoint_execution", move)
            self.assertEqual(
                move["actual_motion_audit"]["gates"][
                    "sequential_transient_xy_limit"
                ]["limit"],
                "<=130.000 mm",
            )

    def test_registration_observation_then_coarse_step_uses_worker_and_no_task9(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            calib = root / "src/scara/calib"
            calib.mkdir(parents=True)
            intrinsics = calib / "camera1_intrinsics.json"
            geometry_path = calib / "tray_board_geometry.json"
            intrinsics.write_text("{}\n", encoding="utf-8")
            p_C_S = np.array([-2.4, -33.0, 149.5])
            joints = [30.0, 60.0, -27.0, j4_for_rz(30.0, 60.0, 20.0)]
            world_xy = np.asarray(fk_wrist(joints[0], joints[1]))
            p_T_current = np.array([-80.0, -80.0])
            dynamic_yaw = 20.0
            dynamic_origin = world_xy - rz(dynamic_yaw)[:2, :2] @ p_T_current
            geo = json.loads(
                (ROOT / "src/scara/calib/tray_board_geometry.json").read_text(
                    encoding="utf-8"
                )
            )
            geo["tray_frame"]["origin_mechanical_xy_mm"] = (
                dynamic_origin - np.array([4.0, -3.0])
            ).tolist()
            geo["tray_frame"]["rotation_mechanical_from_tray"] = rz(17.0).tolist()
            geometry_path.write_text(json.dumps(geo), encoding="utf-8")
            suction_dir = root / "Trajectory Photos/1"
            suction_dir.mkdir(parents=True)
            suction_path = suction_dir / "camera1_suction_target.json"
            suction_payload = {
                "status": "success",
                "camera": {"source_index": 1, "resolution": {"width": 1280, "height": 720}},
                "coordinate_definition": {"working_plane_z_T_mm": -2.0, "imaging_j3_mm": -27.0, "rz_mean_deg": 20.0},
                "locked_inputs": {"camera_intrinsics_sha256": sha(intrinsics), "tray_geometry_sha256": sha(geometry_path)},
                "fit": {"status": "success", "p_C_S_mm": p_C_S.tolist(), "target_pixel_distorted_px": [625.0, 218.0]},
            }
            suction_path.write_text(json.dumps(suction_payload), encoding="utf-8")
            planar_path = calib / "camera1_forearm_planar_handeye.json"
            R_F_C = rz(-32.0)
            probe = [
                {
                    "run_name": "s",
                    "slot": f"P{i}{i}",
                    "tray_xy_mm": [float(i), 0.0],
                    "world_xy_mm": (world_xy + [i, 0]).tolist(),
                    "joints": joints,
                    "forearm_alpha_deg": 90.0,
                }
                for i in range(3)
            ]
            planar_payload = {
                "schema_version": 1,
                "status": "success",
                "scope": {"planar_xy_supported": True, "z_supported": False, "required_j3_mm": -27.0},
                "camera": {"source_index": 1, "resolution": {"width": 1280, "height": 720}},
                "R_F_C": R_F_C.tolist(),
                "locked_inputs": {
                    "camera_intrinsics_sha256": sha(intrinsics),
                    "tray_geometry_sha256": sha(geometry_path),
                    "suction_target_sha256": sha(suction_path),
                },
                "quality_gates": {"all": {"passed": True}},
                "prevalidated_probe_poses": probe,
            }
            planar_path.write_text(json.dumps(planar_payload), encoding="utf-8")
            pose = [world_xy[0], world_xy[1], -27.0, 180.0, 0.0, 20.0]
            initial = {"captured_monotonic_s": time.monotonic(), "joints": joints, "pose": pose}
            output = root / "run"
            session = MovedTrayPositioningSession(root, output, initial, target_name="P22")
            action = session.action_task()
            normalized_step = normalize_action_task(
                {**action, "actions": [action["actions"][2]]}
            )["actions"][0]
            T_C_T = transform_C_T(
                alpha=90.0,
                yaw_W_T=dynamic_yaw,
                point_T=p_T_current,
                p_C_S=p_C_S,
                R_F_C=R_F_C,
            )
            output.mkdir()

            def request_and_samples(window_index: int):
                requested_at = time.monotonic()
                limit_names = (
                    "anchor_robot_xy_mm",
                    "local_extent_mm",
                    "domain_margin_mm",
                    "required_j3_mm",
                    "required_rz_deg",
                    "max_xy_step_norm_mm",
                    "max_xy_axis_mm",
                    "j3_tolerance_mm",
                    "rz_tolerance_deg",
                    "target_rz_tolerance_deg",
                    "max_sequential_transient_rz_deg",
                    "precompensate_rz",
                    "enforce_sequential_intermediate_domain",
                    "max_state_drift_xy_mm",
                    "max_state_drift_joint",
                    "max_sequential_transient_xy_mm",
                    "move_tolerance",
                    "proposal_max_age_s",
                    "fk_pose_xy_tolerance_mm",
                )
                request = {
                    "schema_version": 1,
                    "request_id": f"request-{window_index}",
                    "request_key": normalized_step["request_key"],
                    "target_name": normalized_step["target_name"],
                    "name": normalized_step["name"],
                    "requested_monotonic_s": requested_at,
                    "calibration_sha256": normalized_step["calibration_sha256"],
                    "fine_calibration_sha256": "",
                    "limits": {
                        name: normalized_step[name] for name in limit_names
                    },
                    "controller_state": {"joints": joints, "pose": pose},
                }
                self.assertNotIn("required_j3_mm", request)
                self.assertEqual(request["limits"]["required_j3_mm"], -27.0)
                samples = [
                    {
                        "measurement_id": f"w{window_index}-{index}",
                        "captured_monotonic_s": requested_at + 0.01 * index,
                        "accepted": True,
                        "target_name": "P22",
                        "robot_state_age_s": 0.1,
                        "used_marker_count": 4,
                        "current_joints": joints,
                        "current_robot_xy_mm": pose[:2],
                        "current_pose": pose,
                        "tray_transform_C_T": T_C_T.tolist(),
                        "metric_error_T_mm": [30.0, 30.0, 0.0],
                        "annotated_bgr": np.zeros((20, 30, 3), dtype=np.uint8),
                    }
                    for index in range(5)
                ]
                return request, samples

            first_request, first_samples = request_and_samples(1)
            first = session.build_response(first_request, first_samples)
            self.assertEqual(first["decision"], "observe", first)
            route = np.asarray(session.coarse_route_world_xy_mm, dtype=float)
            self.assertGreater(len(route), 0)
            route_with_start = np.vstack((np.asarray(pose[:2]), route))
            target_xy = registered_slot_world_xy_mm(
                session.geometry, "P22", session.registration
            )
            for start_xy, endpoint_xy in zip(route_with_start[:-1], route):
                remaining = float(np.linalg.norm(target_xy - start_xy))
                expected = min(0.8 * remaining, 9.99)
                self.assertAlmostEqual(
                    float(np.linalg.norm(endpoint_xy - start_xy)),
                    expected,
                    places=9,
                )
            second_request, second_samples = request_and_samples(2)
            second = session.build_response(second_request, second_samples)
            self.assertEqual(second["decision"], "approve", second)
            first_waypoint_distance = float(
                np.linalg.norm(route[0] - np.asarray(pose[:2]))
            )
            first_command_norm = float(
                np.linalg.norm(
                    second["proposal"]["commanded_correction_xy_mm"]
                )
            )
            self.assertGreater(first_command_norm, 1.5)
            self.assertAlmostEqual(
                first_command_norm,
                min(first_waypoint_distance, 9.99),
                places=9,
            )
            self.assertAlmostEqual(first_command_norm, 9.99, places=9)
            self.assertEqual(normalized_step["max_xy_step_norm_mm"], 10.0)
            self.assertAlmostEqual(second["proposal"]["gain"], 0.8, places=12)
            self.assertTrue(second["proposal"]["planner"]["audit"]["passed"])
            self.assertNotIn(
                "execution_waypoints", second["proposal"]["planner"]
            )
            self.assertLessEqual(
                second["proposal"]["planner"]["audit"]["step_norm_mm"],
                10.0 + 1e-9,
            )
            self.assertEqual(
                second["proposal"]["coarse_waypoint_world_xy_mm"],
                list(session.coarse_route_world_xy_mm[0]),
            )
            self.assertEqual(len(second["target_joints"]), 4)
            self.assertEqual(session.coarse_movements, 1)
            self.assertFalse((calib / "camera1_xy_image_jacobian.json").exists())
            report = json.loads(session.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["stage"], "moved_tray_runtime_registration_and_metric_xy_positioning")
            self.assertEqual(report["runtime_registration"]["status"], "success")

            # The abnormal three-pose route must move with the newly
            # registered Tray.  A deliberately bogus historical world point
            # must not influence the command.
            session.phase = "probe"
            session.probe_target_index = 0
            session.probe_targets = [
                {
                    "slot": "P44",
                    "tray_xy_mm": (p_T_current + [20.0, 0.0]).tolist(),
                    "world_xy_mm": [-999.0, -999.0],
                    "joints": joints,
                }
            ]
            probe_request, probe_samples = request_and_samples(3)
            probe_response = session.build_response(probe_request, probe_samples)
            self.assertEqual(probe_response["decision"], "approve", probe_response)
            probe_command = np.asarray(
                probe_response["proposal"]["commanded_correction_xy_mm"]
            )
            expected_direction = rz(dynamic_yaw)[:2, 0]
            self.assertGreater(float(np.dot(probe_command, expected_direction)), 1.99)
            self.assertLess(float(np.linalg.norm(probe_command - 2.0 * expected_direction)), 1e-6)

    def test_worker_observe_then_complete_performs_zero_motion(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            raw_step = {
                "type": "runtime_move_joints",
                "name": "window",
                "request_key": "moved_tray_p22_runtime_positioning",
                "target_name": "P22",
                "calibration_sha256": "A" * 64,
                "anchor_robot_xy_mm": [118.0, 273.0],
                "local_extent_mm": 130.0,
                "domain_margin_mm": 5.0,
                "required_j3_mm": -27.0046,
                "required_rz_deg": 20.8223,
                "max_xy_step_norm_mm": 10.0,
                "max_xy_axis_mm": 10.0,
                "j3_tolerance_mm": 0.2,
                "rz_tolerance_deg": 0.3,
                "target_rz_tolerance_deg": 0.15,
                "max_sequential_transient_rz_deg": 15.0,
                "precompensate_rz": True,
                "max_state_drift_xy_mm": 0.2,
                "max_state_drift_joint": 0.2,
                "max_sequential_transient_xy_mm": 130.0,
                "move_tolerance": 0.01,
                "proposal_max_age_s": 8.0,
                "fk_pose_xy_tolerance_mm": 0.2,
            }
            task = {
                "api_version": 1,
                "name": "observe complete",
                "camera_model": {"offset_mm": 0.0},
                "actions": [raw_step, dict(raw_step)],
            }
            worker = ActionWorker(
                FakeController(), normalize_action_task(task), Path(td) / "run"
            )
            count = {"value": 0}

            def respond(request):
                count["value"] += 1
                worker.respond_runtime_move_joints(
                    {
                        "request_id": request["request_id"],
                        "decision": "observe" if count["value"] == 1 else "complete",
                        "calibration_sha256": "A" * 64,
                        "reason": "next" if count["value"] == 1 else "done",
                    }
                )

            worker.runtime_move_joints_requested.connect(respond)
            result = []
            worker.run_finished.connect(lambda ok, message, folder: result.append(ok))
            worker.run()
            self.assertEqual(result, [True])
            self.assertEqual(worker._controller.goto_calls, [])
            manifest = json.loads(worker.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [row["status"] for row in manifest["runtime_moves"]],
                ["observation_completed_no_motion", "session_completed_no_motion"],
            )


if __name__ == "__main__":
    unittest.main()
