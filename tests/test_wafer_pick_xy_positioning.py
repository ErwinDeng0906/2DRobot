from __future__ import annotations

import math
import tempfile
import time
import unittest
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.pipeline.kinematics import (  # noqa: E402
    fk_wrist,
    forearm_pose_W_F,
    j4_for_rz,
    rz_of,
    solve_joints,
)
from scara.ui.action_worker import normalize_action_task  # noqa: E402
from scara.vision.handeye_interaction import load_latest_suction_target  # noqa: E402
from scara.vision.moved_tray_servo import registered_slot_world_xy_mm  # noqa: E402
from scara.vision.tray_pose_estimator import load_tray_board_geometry  # noqa: E402
from scara.vision.wafer_pick_xy_positioning import (  # noqa: E402
    CONTROLLER_QUANTIZATION_HEADROOM_MM,
    MAXIMUM_OBSERVATION_GAP_S,
    MAXIMUM_STEP_MM,
    OBSERVATION_WINDOW_SIZE,
    WAFER_PICK_XY_RUNTIME_REQUEST_KEY,
    WaferPickXYPositioningSession,
    select_bounded_observation_window,
)


class WaferPickXYPositioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = load_tray_board_geometry(
            PROJECT_ROOT / "src/scara/calib/tray_board_geometry.json"
        )
        cls.suction = load_latest_suction_target(PROJECT_ROOT)
        frame = cls.geometry["tray_frame"]
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(
            frame["rotation_mechanical_from_tray"], dtype=np.float64
        )
        transform[:2, 3] = np.asarray(
            frame["origin_mechanical_xy_mm"], dtype=np.float64
        )
        cls.registration = {
            "status": "success",
            "transform_W_T": transform.tolist(),
            "origin_world_xy_mm": transform[:2, 3].astype(float).tolist(),
            "yaw_world_from_tray_deg": math.degrees(
                math.atan2(transform[1, 0], transform[0, 0])
            ),
        }

    @classmethod
    def _robot_state_at(cls, slot_name: str, captured: float | None = None) -> dict:
        xy = registered_slot_world_xy_mm(
            cls.geometry, slot_name, cls.registration
        )
        joints = solve_joints(
            float(xy[0]),
            float(xy[1]),
            float(cls.suction.imaging_j3_mm),
            rz_deg=float(cls.suction.rz_mean_deg),
            ref_joints=[30.0, 85.0, float(cls.suction.imaging_j3_mm), -5.0],
        )
        if joints is None:
            raise AssertionError(f"no IK for {slot_name}")
        fk_xy = fk_wrist(joints[0], joints[1])
        return {
            "captured_monotonic_s": time.monotonic() if captured is None else captured,
            "joints": list(joints),
            "pose": [
                float(fk_xy[0]),
                float(fk_xy[1]),
                float(joints[2]),
                180.0,
                0.0,
                float(rz_of(joints[0], joints[1], joints[3])),
            ],
        }

    @classmethod
    def _snapshot(cls, target: str, state: dict) -> dict:
        gates = {
            "overview_pose_quality": {"passed": True},
            "source_normal_occupied_consensus": {"passed": True},
            "runtime_registration": {"passed": True},
            "fresh_frame_synchronised_robot_state": {"passed": True},
        }
        return {
            "source_slot": target,
            "source_state": {"state": "occupied"},
            "source_consensus": {
                "passed": True,
                "occupied_frame_count": 2,
                "window_frame_count": 2,
            },
            "tracking_ready": True,
            "selection_gates": gates,
            "registration": cls.registration,
            "robot_state": state,
        }

    @classmethod
    def _samples(
        cls,
        target: str,
        state: dict,
        requested_at: float,
    ) -> list[dict]:
        gates = cls._snapshot(target, state)["selection_gates"]
        rows = []
        for index in range(OBSERVATION_WINDOW_SIZE):
            captured = requested_at + 0.02 * (index + 1)
            frame_state = {
                **state,
                "captured_monotonic_s": captured,
            }
            rows.append(
                {
                    "measurement_id": f"synthetic-{index}",
                    "frame_sequence": index + 1,
                    "captured_monotonic_s": captured,
                    "accepted": True,
                    "target_name": target,
                    "source_state": {"state": "occupied"},
                    "source_consensus": {"passed": True},
                    "selection_gates": gates,
                    "robot_state": frame_state,
                    "registration": cls.registration,
                    "reprojection_rms_px": 0.25,
                    "used_marker_count": 8,
                    "annotated_bgr": np.zeros((80, 120, 3), dtype=np.uint8),
                }
            )
        return rows

    @staticmethod
    def _camera2_samples(requested_at: float, angle_deg: float) -> list[dict]:
        return [
            {
                "measurement_id": f"camera2-synthetic-{index}",
                "frame_sequence": 100 + index,
                "captured_monotonic_s": requested_at + 0.02 * (index + 1),
                "accepted": True,
                "camera_source": 2,
                "angle_error_deg": float(angle_deg),
                "marker_ids": [25],
                "marker_count": 1,
                "annotated_bgr": np.zeros((80, 120, 3), dtype=np.uint8),
            }
            for index in range(OBSERVATION_WINDOW_SIZE)
        ]

    @staticmethod
    def _tray_transform_C_T(
        session: WaferPickXYPositioningSession,
        state: dict,
        registration: dict,
    ) -> list[list[float]]:
        transform_W_T = np.asarray(registration["transform_W_T"], dtype=np.float64)
        rotation_W_T = transform_W_T[:3, :3]
        joints = np.asarray(state["joints"], dtype=np.float64)
        forearm = np.asarray(
            forearm_pose_W_F(float(joints[0]), float(joints[1])),
            dtype=np.float64,
        )
        rotation_W_F = np.eye(3, dtype=np.float64)
        rotation_W_F[:2, :2] = forearm[:2, :2]
        rotation_F_C = np.asarray(session.handeye["R_F_C"], dtype=np.float64)
        rotation_C_T = (
            np.linalg.inv(rotation_F_C) @ rotation_W_F.T @ rotation_W_T
        )
        world_xy = np.asarray(state["pose"][:2], dtype=np.float64)
        suction_T = np.zeros(3, dtype=np.float64)
        suction_T[:2] = (
            rotation_W_T[:2, :2].T
            @ (world_xy - transform_W_T[:2, 3])
        )
        transform_C_T = np.eye(4, dtype=np.float64)
        transform_C_T[:3, :3] = rotation_C_T
        transform_C_T[:3, 3] = (
            np.asarray(session.suction.p_C_S_mm, dtype=np.float64)
            - rotation_C_T @ suction_T
        )
        return transform_C_T.astype(float).tolist()

    def _session(
        self,
        temporary: str,
        *,
        target: str = "P31",
        current: str = "P22",
    ) -> tuple[WaferPickXYPositioningSession, dict]:
        state = self._robot_state_at(current)
        session = WaferPickXYPositioningSession(
            PROJECT_ROOT,
            Path(temporary) / "run",
            state,
            self._snapshot(target, state),
            target_name=target,
        )
        return session, state

    @staticmethod
    def _request(session: WaferPickXYPositioningSession, state: dict) -> dict:
        return {
            "schema_version": 1,
            "request_id": f"request-{time.monotonic_ns()}",
            "request_key": WAFER_PICK_XY_RUNTIME_REQUEST_KEY,
            "target_name": session.target_name,
            "requested_monotonic_s": time.monotonic(),
            "calibration_sha256": session.calibration_hash,
            "controller_state": state,
        }

    def test_arm_accepts_latched_two_frame_proof_during_current_warning(self) -> None:
        state = self._robot_state_at("P22")
        snapshot = self._snapshot("P31", state)
        snapshot["source_state"] = {"state": "warning"}
        snapshot["source_consensus"] = {
            "passed": True,
            "evidence_passed": False,
            "qualification_latched": True,
            "qualification_frame_sequence": 17,
            "occupied_frame_count": 0,
            "required_occupied_frame_count": 2,
            "window_frame_count": 2,
        }
        with tempfile.TemporaryDirectory() as temporary:
            session = WaferPickXYPositioningSession(
                PROJECT_ROOT,
                Path(temporary) / "run",
                state,
                snapshot,
                target_name="P31",
            )
        self.assertTrue(session.source_confirmed_at_arm)
        self.assertEqual(
            17,
            session.locked_acquisition_evidence[
                "qualification_frame_sequence"
            ],
        )

    def test_action_task_allows_selected_slot_and_contains_no_descent_or_io(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session, _state = self._session(temporary)
            task = session.action_task()
        runtime_steps = [
            step
            for step in task["actions"]
            if step["type"] == "runtime_move_joints"
        ]
        self.assertTrue(runtime_steps)
        self.assertTrue(all(step["target_name"] == "P31" for step in runtime_steps))
        self.assertTrue(
            all(
                step["request_key"] == WAFER_PICK_XY_RUNTIME_REQUEST_KEY
                for step in runtime_steps
            )
        )
        self.assertLessEqual(runtime_steps[0]["max_xy_step_norm_mm"], 10.0)
        self.assertIn("final_rz_deg", runtime_steps[0])
        self.assertLessEqual(runtime_steps[0]["max_j4_rotation_deg"], 30.0)
        self.assertLessEqual(runtime_steps[0]["local_extent_mm"], 70.0)
        forbidden = {"move_xyzr", "set_do", "capture", "start_video", "stop_video"}
        self.assertFalse(
            forbidden.intersection(step["type"] for step in task["actions"])
        )

    def test_tray_aligned_start_inserts_safe_j4_only_calibration_prealignment(self) -> None:
        state = self._robot_state_at("P22")
        joints = list(state["joints"])
        tray_rz = float(self.registration["yaw_world_from_tray_deg"])
        joints[3] = j4_for_rz(joints[0], joints[1], tray_rz)
        state["joints"] = joints
        state["pose"][5] = tray_rz
        with tempfile.TemporaryDirectory() as temporary:
            session = WaferPickXYPositioningSession(
                PROJECT_ROOT,
                Path(temporary) / "run",
                state,
                self._snapshot("P31", state),
                target_name="P31",
            )
            task = session.action_task()
            task = normalize_action_task(task)
        self.assertTrue(session.prealignment_required)
        self.assertTrue(session.prealignment_audit["passed"])
        move = next(
            step
            for step in task["actions"]
            if step["type"] == "move_joints"
        )
        self.assertEqual(joints[:3], move["joints"][:3])
        self.assertAlmostEqual(
            float(self.suction.imaging_j3_mm),
            move["require_current_j3_mm"],
            places=9,
        )
        self.assertLessEqual(abs(move["joints"][3] - joints[3]), 30.0)
        self.assertAlmostEqual(
            float(self.suction.rz_mean_deg),
            rz_of(move["joints"][0], move["joints"][1], move["joints"][3]),
            places=9,
        )

    def test_two_frames_produce_xy_only_candidate_with_fixed_j3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session, state = self._session(temporary)
            request = self._request(session, state)
            response = session.build_response(
                request,
                self._samples(
                    session.target_name,
                    state,
                    float(request["requested_monotonic_s"]),
                ),
            )
        self.assertEqual("approve", response["decision"])
        proposal = response["proposal"]
        self.assertTrue(proposal["xy_only"])
        self.assertFalse(proposal["z_motion_authorized"])
        self.assertFalse(proposal["vacuum_authorized"])
        self.assertFalse(proposal["do_authorized"])
        command = np.asarray(proposal["commanded_correction_xy_mm"])
        self.assertLessEqual(
            float(np.linalg.norm(command)),
            MAXIMUM_STEP_MM - CONTROLLER_QUANTIZATION_HEADROOM_MM + 1e-9,
        )
        self.assertGreater(float(np.linalg.norm(command)), 5.0)
        self.assertAlmostEqual(response["target_joints"][2], state["joints"][2], places=9)

    def test_locked_source_occlusion_does_not_interrupt_xy_motion_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session, state = self._session(temporary)
            request = self._request(session, state)
            samples = self._samples(
                session.target_name,
                state,
                float(request["requested_monotonic_s"]),
            )
            motion_gates = {
                "current_overview_pose_quality": {"passed": True},
                "locked_runtime_registration": {"passed": True},
                "locked_source_not_explicitly_contradicted": {
                    "passed": True,
                    "actual": "occluded",
                },
                "fresh_frame_synchronised_robot_state": {"passed": True},
            }
            for sample in samples:
                sample["source_state"] = {"state": "occluded"}
                sample["motion_gates"] = motion_gates
            response = session.build_response(request, samples)

        self.assertEqual("approve", response["decision"])
        gates = response["proposal"]["safety_gates"]
        self.assertTrue(gates["source_locked_normal_occupied_at_arm"]["passed"])
        self.assertTrue(gates["current_overview_motion_gates"]["passed"])

    def test_repeated_occlusion_completes_full_xy_and_final_j4_replay(self) -> None:
        def state_from_joints(joints: list[float]) -> dict:
            xy = fk_wrist(joints[0], joints[1])
            return {
                "captured_monotonic_s": time.monotonic(),
                "joints": list(joints),
                "pose": [
                    float(xy[0]),
                    float(xy[1]),
                    float(joints[2]),
                    180.0,
                    0.0,
                    float(rz_of(joints[0], joints[1], joints[3])),
                ],
            }

        with tempfile.TemporaryDirectory() as temporary:
            session, state = self._session(temporary)
            distances: list[float] = []
            response = None
            for _index in range(30):
                state["captured_monotonic_s"] = time.monotonic()
                request = self._request(session, state)
                samples = self._samples(
                    session.target_name,
                    state,
                    float(request["requested_monotonic_s"]),
                )
                for sample in samples:
                    sample["source_state"] = {"state": "occluded"}
                    sample["motion_gates"] = {
                        "current_overview_pose_quality": {"passed": True},
                        "locked_runtime_registration": {"passed": True},
                        "locked_source_not_explicitly_contradicted": {
                            "passed": True,
                            "actual": "occluded",
                        },
                        "fresh_frame_synchronised_robot_state": {"passed": True},
                    }
                response = session.build_response(request, samples)
                if response["decision"] in {"complete", "observe"}:
                    break
                self.assertEqual("approve", response["decision"], response)
                proposal = response["proposal"]
                if proposal["phase"] == "wafer_pick_xy_overhead":
                    distances.append(
                        float(proposal["calculation"]["distance_before_mm"])
                    )
                state = state_from_joints(list(response["target_joints"]))

            self.assertIsNotNone(response)
            self.assertEqual("observe", response["decision"])
            self.assertTrue(response["camera2_required"])
            camera2_request = self._request(session, state)
            correction = session.build_response(
                camera2_request,
                self._camera2_samples(
                    float(camera2_request["requested_monotonic_s"]), -4.5
                ),
            )
            self.assertEqual("approve", correction["decision"])
            state = state_from_joints(list(correction["target_joints"]))
            verification_request = self._request(session, state)
            response = session.build_response(
                verification_request,
                self._camera2_samples(
                    float(verification_request["requested_monotonic_s"]), 0.15
                ),
            )

        self.assertEqual("complete", response["decision"])
        self.assertGreater(len(distances), 1)
        self.assertTrue(
            all(later < earlier for earlier, later in zip(distances, distances[1:]))
        )
        self.assertEqual(
            "arrived_above_selected_wafer_and_tray_aligned", session.status
        )

    def test_runtime_pnp_drift_is_a_gate_not_a_moving_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session, state = self._session(temporary)
            request = self._request(session, state)
            samples = self._samples(
                session.target_name,
                state,
                float(request["requested_monotonic_s"]),
            )
            drifted = dict(self.registration)
            transform = np.asarray(
                self.registration["transform_W_T"], dtype=np.float64
            ).copy()
            transform[0, 3] += 0.50
            drifted["transform_W_T"] = transform.tolist()
            drifted["origin_world_xy_mm"] = [
                float(self.registration["origin_world_xy_mm"][0]) + 0.50,
                float(self.registration["origin_world_xy_mm"][1]),
            ]
            for sample in samples:
                sample["registration"] = drifted
            response = session.build_response(request, samples)

        self.assertEqual("approve", response["decision"])
        window = response["proposal"]["window"]
        self.assertTrue(
            np.allclose(
                window["target_world_xy_mm"],
                session.locked_target_world_xy_mm,
                atol=1e-9,
            )
        )
        self.assertAlmostEqual(
            0.50,
            float(window["observed_target_world_xy_mm"][0])
            - float(window["target_world_xy_mm"][0]),
            places=9,
        )
        self.assertTrue(
            response["proposal"]["safety_gates"][
                "registration_locked_to_armed_session"
            ]["passed"]
        )

    def test_current_pose_rechecks_tray_without_overwriting_locked_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session, state = self._session(temporary)
            request = self._request(session, state)
            samples = self._samples(
                session.target_name,
                state,
                float(request["requested_monotonic_s"]),
            )
            current_transform = self._tray_transform_C_T(
                session, state, self.registration
            )
            for sample in samples:
                sample["tray_transform_C_T"] = current_transform
            response = session.build_response(request, samples)

        self.assertEqual("approve", response["decision"])
        gate = response["proposal"]["safety_gates"][
            "registration_locked_to_armed_session"
        ]
        self.assertTrue(gate["passed"])
        self.assertAlmostEqual(0.0, gate["actual"]["translation_drift_mm"], places=6)
        np.testing.assert_allclose(
            response["proposal"]["window"]["target_world_xy_mm"],
            session.locked_target_world_xy_mm,
            atol=1e-9,
        )

    def test_bounded_window_tolerates_brief_rejected_frames(self) -> None:
        requested = 100.0
        rows = [
            {"captured_monotonic_s": 100.0, "accepted": True, "id": "good-0"},
            {"captured_monotonic_s": 100.5, "accepted": False, "id": "bad-0"},
            {"captured_monotonic_s": 101.0, "accepted": True, "id": "good-1"},
            {"captured_monotonic_s": 101.5, "accepted": True, "id": "good-2"},
            {"captured_monotonic_s": 102.0, "accepted": False, "id": "bad-1"},
            {"captured_monotonic_s": 102.5, "accepted": True, "id": "good-3"},
            {"captured_monotonic_s": 103.0, "accepted": True, "id": "good-4"},
        ]
        selected = select_bounded_observation_window(
            rows, requested_monotonic_s=requested
        )
        self.assertEqual(
            ["good-0", "good-1"],
            [row["id"] for row in selected],
        )

    def test_bounded_window_resets_after_long_recognition_gap(self) -> None:
        requested = 200.0
        rows = [
            {"captured_monotonic_s": 200.0, "accepted": True, "id": "old-0"},
            {
                "captured_monotonic_s": 200.0 + MAXIMUM_OBSERVATION_GAP_S + 0.01,
                "accepted": True,
                "id": "new-0",
            },
            {"captured_monotonic_s": 202.0, "accepted": True, "id": "new-1"},
        ]
        selected = select_bounded_observation_window(
            rows, requested_monotonic_s=requested
        )
        self.assertEqual(["new-0", "new-1"], [row["id"] for row in selected])

    def test_no_measured_progress_aborts_before_second_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session, state = self._session(temporary)
            first_request = self._request(session, state)
            first = session.build_response(
                first_request,
                self._samples(
                    session.target_name,
                    state,
                    float(first_request["requested_monotonic_s"]),
                ),
            )
            self.assertEqual("approve", first["decision"])
            second_request = self._request(session, state)
            second = session.build_response(
                second_request,
                self._samples(
                    session.target_name,
                    state,
                    float(second_request["requested_monotonic_s"]),
                ),
            )
        self.assertEqual("abort", second["decision"])
        self.assertIn("质量门拒绝", second["reason"])
        progress = second["evaluation"]["safety_gates"]["post_motion_progress"]
        self.assertFalse(progress["passed"])

    def test_one_failed_frame_rejects_the_two_frame_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session, state = self._session(temporary)
            request = self._request(session, state)
            samples = self._samples(
                session.target_name,
                state,
                float(request["requested_monotonic_s"]),
            )
            samples[1]["accepted"] = False
            response = session.build_response(request, samples)
        self.assertEqual("abort", response["decision"])
        gate = response["evaluation"]["safety_gates"][
            "all_frames_explicitly_accepted"
        ]
        self.assertFalse(gate["passed"])

    def test_duplicate_frame_sequence_rejects_the_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session, state = self._session(temporary)
            request = self._request(session, state)
            samples = self._samples(
                session.target_name,
                state,
                float(request["requested_monotonic_s"]),
            )
            samples[1]["frame_sequence"] = samples[0]["frame_sequence"]
            response = session.build_response(request, samples)
        self.assertEqual("abort", response["decision"])
        gate = response["evaluation"]["safety_gates"][
            "distinct_ordered_frames"
        ]
        self.assertFalse(gate["passed"])

    def test_xy_hold_then_camera2_measures_corrects_and_verifies_j4(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session, state = self._session(
                temporary,
                target="P31",
                current="P31",
            )
            request = self._request(session, state)
            hold = session.build_response(
                request,
                self._samples(
                    session.target_name,
                    state,
                    float(request["requested_monotonic_s"]),
                ),
            )
            self.assertEqual("observe", hold["decision"])
            self.assertTrue(hold["camera2_required"])

            camera2_request = self._request(session, state)
            orientation = session.build_response(
                camera2_request,
                self._camera2_samples(
                    float(camera2_request["requested_monotonic_s"]), -4.5
                ),
            )
            self.assertEqual("approve", orientation["decision"])
            proposal = orientation["proposal"]
            self.assertEqual(
                "wafer_pick_final_tray_orientation", proposal["phase"]
            )
            self.assertFalse(
                proposal["window"]["camera2_visual_alignment_gate"]["passed"]
            )
            self.assertTrue(
                all(
                    gate["passed"] is True
                    for gate in proposal["safety_gates"].values()
                )
            )
            self.assertFalse(proposal["xy_only"])
            self.assertTrue(proposal["j4_only"])
            for index in range(3):
                self.assertAlmostEqual(
                    state["joints"][index],
                    orientation["target_joints"][index],
                    places=9,
                )
            self.assertAlmostEqual(
                rz_of(
                    state["joints"][0], state["joints"][1], state["joints"][3]
                )
                + 4.5,
                rz_of(
                    orientation["target_joints"][0],
                    orientation["target_joints"][1],
                    orientation["target_joints"][3],
                ),
                places=6,
            )

            final_joints = list(orientation["target_joints"])
            final_xy = fk_wrist(final_joints[0], final_joints[1])
            final_state = {
                "captured_monotonic_s": time.monotonic(),
                "joints": final_joints,
                "pose": [
                    float(final_xy[0]),
                    float(final_xy[1]),
                    float(final_joints[2]),
                    180.0,
                    0.0,
                    rz_of(final_joints[0], final_joints[1], final_joints[3]),
                ],
            }
            completed = session.build_response(
                (verification_request := self._request(session, final_state)),
                self._camera2_samples(
                    float(verification_request["requested_monotonic_s"]), 0.12
                ),
            )
        self.assertEqual("complete", completed["decision"])
        self.assertIn("相机2连续两帧确认托盘横平竖直", completed["reason"])
        self.assertIn("J1/J2/J3保持不动", completed["reason"])


if __name__ == "__main__":
    unittest.main()
