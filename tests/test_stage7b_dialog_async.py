from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication

from scara.ui.handeye_demo_dialog import HandEyeDemoDialog
from tests import test_handeye_demo_read_only as handeye_tests


class _FakeSession:
    def __init__(self) -> None:
        self.calls = []

    def build_response(self, request, samples):
        self.calls.append((dict(request), list(samples)))
        return {
            "request_id": request["request_id"],
            "decision": "complete",
            "reason": "|e|=0.5px",
        }

    def finish(self, ok, message):
        return None


def sample(identifier: str, captured: float):
    return {
        "measurement_id": identifier,
        "captured_monotonic_s": captured,
        "accepted": True,
        "target_name": "P22",
        "image_error_px": [0.4, -0.2],
        "current_robot_xy_mm": [118.3, 272.8],
        "current_joints": [30.6, 84.8, -27.0, -4.6],
        "current_pose": [118.3, 272.8, -27.0, 180, 0, 20.8],
        "robot_state_age_s": 0.1,
        "annotated_bgr": np.zeros((20, 30, 3), dtype=np.uint8),
    }


class Stage7BDialogAsyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _dialog(self, root: Path):
        handeye_tests.HandEyeDialogOfflineTests._temporary_project(root)
        camera = handeye_tests.FakeCamera(running=True)
        dialog = HandEyeDemoDialog(root, camera)
        dialog.monitor.stop()
        dialog._stage7b_active = True
        dialog._stage7b_session = _FakeSession()
        return dialog, camera

    def test_only_five_post_request_frames_release_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            dialog, camera = self._dialog(Path(temporary))
            responses = []
            requested = time.monotonic()
            request = {"request_id": "r1", "requested_monotonic_s": requested}
            try:
                self.assertEqual(dialog.stage7b_button.text(), "单点有限闭环")
                dialog.begin_stage7b_request(request, responses.append)
                dialog._stage7b_samples.append(sample("old", requested - 0.01))
                dialog._try_stage7b_response()
                self.assertEqual(responses, [])
                for index in range(4):
                    dialog._stage7b_samples.append(
                        sample(f"new-{index}", requested + 0.01 + index * 0.01)
                    )
                    dialog._try_stage7b_response()
                self.assertEqual(responses, [])
                dialog._stage7b_samples.append(sample("new-4", requested + 0.10))
                dialog._try_stage7b_response()
                self.assertEqual(len(responses), 1)
                self.assertEqual(responses[0]["decision"], "complete")
                self.assertEqual(len(dialog._stage7b_session.calls[0][1]), 5)
            finally:
                dialog._stage7b_active = False
                camera.running = False
                dialog.close()

    def test_timeout_aborts_without_motion_proposal(self):
        with tempfile.TemporaryDirectory() as temporary:
            dialog, camera = self._dialog(Path(temporary))
            responses = []
            request = {"request_id": "timeout", "requested_monotonic_s": time.monotonic()}
            try:
                dialog.begin_stage7b_request(request, responses.append)
                dialog._stage7b_timeout("timeout")
                self.assertEqual(len(responses), 1)
                self.assertEqual(responses[0]["decision"], "abort")
                self.assertIsNone(dialog._stage7b_pending_request)
            finally:
                dialog._stage7b_active = False
                camera.running = False
                dialog.close()


if __name__ == "__main__":
    unittest.main()
