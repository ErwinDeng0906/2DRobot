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

from scara.pipeline.kinematics import fk_wrist, rz_of, solve_joints  # noqa: E402
from scara.vision.handeye_interaction import load_latest_suction_target  # noqa: E402
from scara.vision.moved_tray_servo import registered_slot_world_xy_mm  # noqa: E402
from scara.vision.tray_pose_estimator import load_tray_board_geometry  # noqa: E402
from scara.vision.wafer_pick_xy_positioning import (  # noqa: E402
    WAFER_PICK_XY_RUNTIME_REQUEST_KEY,
    WaferPickXYPositioningSession,
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
                "occupied_frame_count": 5,
                "window_frame_count": 5,
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
        for index in range(5):
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
        self.assertLessEqual(runtime_steps[0]["max_xy_step_norm_mm"], 2.0)
        self.assertLessEqual(runtime_steps[0]["local_extent_mm"], 70.0)
        forbidden = {"move_xyzr", "set_do", "capture", "start_video", "stop_video"}
        self.assertFalse(
            forbidden.intersection(step["type"] for step in task["actions"])
        )

    def test_five_frames_produce_xy_only_candidate_with_fixed_j3(self) -> None:
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
        self.assertLessEqual(float(np.linalg.norm(command)), 2.0 + 1e-9)
        self.assertAlmostEqual(response["target_joints"][2], state["joints"][2], places=9)

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

    def test_one_failed_frame_rejects_the_entire_five_frame_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session, state = self._session(temporary)
            request = self._request(session, state)
            samples = self._samples(
                session.target_name,
                state,
                float(request["requested_monotonic_s"]),
            )
            samples[2]["accepted"] = False
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
            samples[3]["frame_sequence"] = samples[2]["frame_sequence"]
            response = session.build_response(request, samples)
        self.assertEqual("abort", response["decision"])
        gate = response["evaluation"]["safety_gates"][
            "five_distinct_ordered_frames"
        ]
        self.assertFalse(gate["passed"])

    def test_independent_five_frame_hold_completes_without_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session, state = self._session(
                temporary,
                target="P31",
                current="P31",
            )
            request = self._request(session, state)
            response = session.build_response(
                request,
                self._samples(
                    session.target_name,
                    state,
                    float(request["requested_monotonic_s"]),
                ),
            )
        self.assertEqual("complete", response["decision"])
        self.assertIn("J3未下降", response["reason"])


if __name__ == "__main__":
    unittest.main()
