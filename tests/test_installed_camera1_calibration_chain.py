"""Regression check for the installed camera-1 calibration asset chain."""

from __future__ import annotations

import unittest
from pathlib import Path

from scara.vision.handeye_interaction import (
    load_latest_suction_target,
    load_local_xy_jacobian,
)
from scara.vision.runtime_tray_registration import load_planar_handeye


EXPECTED_SUCTION_SHA256 = (
    "6B0B4DFCB8476A2BD219471C769B7D6C95D2F965E3CB330D230AB0781C2D75D9"
)


class InstalledCamera1CalibrationChainTests(unittest.TestCase):
    def test_installed_assets_form_one_usable_chain(self) -> None:
        project_root = Path(__file__).resolve().parents[1]

        suction = load_latest_suction_target(project_root)
        handeye = load_planar_handeye(project_root, suction)
        local_jacobian = load_local_xy_jacobian(project_root, suction)

        self.assertEqual(suction.source_sha256, EXPECTED_SUCTION_SHA256)
        self.assertEqual(handeye["status"], "success")
        self.assertIsNotNone(local_jacobian)
        self.assertEqual(local_jacobian["status"], "success")
        self.assertEqual(
            handeye["locked_inputs"]["suction_target_sha256"],
            suction.source_sha256,
        )
        self.assertEqual(
            local_jacobian["locked_inputs"]["suction_target_sha256"],
            suction.source_sha256,
        )


if __name__ == "__main__":
    unittest.main()
