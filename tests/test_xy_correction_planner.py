from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.pipeline.kinematics import fk_wrist, rz_of
from scara.pipeline.xy_correction_planner import (
    audit_fixed_rz_xy_target,
    load_p22_float_preset,
    plan_fixed_rz_xy_step,
)


P22_JOINTS = [30.6646, 84.7845, -27.0046, -4.6268]
P22_XY = list(fk_wrist(P22_JOINTS[0], P22_JOINTS[1]))
P22_POSE = [P22_XY[0], P22_XY[1], P22_JOINTS[2], 180.0, 0.0, 20.8223]
P22_RZ = rz_of(P22_JOINTS[0], P22_JOINTS[1], P22_JOINTS[3])


def _limits() -> dict:
    return {
        "anchor_robot_xy_mm": P22_XY,
        "local_extent_mm": 2.0,
        "domain_margin_mm": 0.2,
        "required_j3_mm": P22_JOINTS[2],
        "j3_tolerance_mm": 0.15,
        "required_rz_deg": P22_RZ,
        "rz_tolerance_deg": 0.15,
        "max_xy_step_norm_mm": 0.25,
        "max_xy_axis_mm": 0.25,
        "max_sequential_transient_xy_mm": 0.5,
    }


class XYCorrectionPlannerTests(unittest.TestCase):
    def test_plans_small_step_with_fixed_j3_and_rz(self) -> None:
        result = plan_fixed_rz_xy_step(
            P22_JOINTS,
            P22_POSE,
            [-0.15, -0.10],
            **_limits(),
        )
        audit = result["audit"]
        self.assertTrue(audit["passed"])
        self.assertLessEqual(audit["step_norm_mm"], 0.25 + 1e-9)
        self.assertLessEqual(audit["sequential_transient_max_mm"], 0.5 + 1e-9)
        self.assertAlmostEqual(result["target_joints"][2], P22_JOINTS[2], places=9)
        self.assertLess(
            abs(audit["target_rz_deg"] - P22_RZ),
            1e-8,
        )

    def test_rejects_step_larger_than_stage7a_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "xy_step_norm_limit"):
            plan_fixed_rz_xy_step(
                P22_JOINTS,
                P22_POSE,
                [0.25, 0.25],
                **_limits(),
            )

    def test_audit_rejects_z_or_rz_change(self) -> None:
        target = list(P22_JOINTS)
        target[2] += 0.01
        target[3] += 1.0
        audit = audit_fixed_rz_xy_target(
            P22_JOINTS,
            P22_POSE,
            target,
            **_limits(),
        )
        self.assertFalse(audit["passed"])
        self.assertFalse(audit["gates"]["target_j3_unchanged"]["passed"])
        self.assertFalse(audit["gates"]["target_rz_preserved"]["passed"])

    def test_sequential_axis_rz_departure_is_independently_bounded(self) -> None:
        result = plan_fixed_rz_xy_step(
            P22_JOINTS,
            P22_POSE,
            [0.10, 0.0],
            **{**_limits(), "rz_tolerance_deg": 0.001},
            allow_rejected_audit=True,
        )
        gates = result["audit"]["gates"]
        self.assertTrue(gates["target_rz_preserved"]["passed"])
        self.assertFalse(gates["sequential_transient_rz_limit"]["passed"])

    def test_j4_precompensation_separates_start_target_and_transient_rz_limits(self) -> None:
        current = [30.8957, 84.3851, -27.0074, -4.2862]
        current_xy = fk_wrist(current[0], current[1])
        current_pose = [
            current_xy[0],
            current_xy[1],
            current[2],
            180.0,
            0.0,
            rz_of(current[0], current[1], current[3]),
        ]
        result = plan_fixed_rz_xy_step(
            current,
            current_pose,
            [-0.07193199871756309, -0.23942804255244723],
            **{
                **_limits(),
                "rz_tolerance_deg": 0.20,
                "target_rz_tolerance_deg": 0.15,
                "max_sequential_transient_rz_deg": 0.30,
                "precompensate_rz": True,
            },
        )
        audit = result["audit"]
        self.assertTrue(audit["passed"])
        self.assertAlmostEqual(
            audit["gates"]["current_rz_matches_calibration"]["actual"],
            0.1723,
            places=6,
        )
        self.assertLess(
            audit["gates"]["sequential_transient_rz_limit"]["actual"],
            0.15,
        )
        precomp = audit["rz_precompensation"]
        self.assertTrue(precomp["enabled"])
        self.assertTrue(precomp["required"])
        self.assertAlmostEqual(precomp["delta_j4_deg"], -0.1723, places=6)

    def test_sequential_intermediate_must_remain_inside_local_domain(self) -> None:
        setup = plan_fixed_rz_xy_step(
            P22_JOINTS,
            P22_POSE,
            [-1.79, -1.79],
            **{
                **_limits(),
                "domain_margin_mm": 0.0,
                "max_xy_step_norm_mm": 3.0,
                "max_xy_axis_mm": 3.0,
                "max_sequential_transient_xy_mm": 20.0,
                "rz_tolerance_deg": 20.0,
            },
            allow_rejected_audit=True,
        )
        current_joints = setup["target_joints"]
        current_xy = fk_wrist(current_joints[0], current_joints[1])
        current_pose = [*current_xy, P22_JOINTS[2], 180.0, 0.0, P22_RZ]
        result = plan_fixed_rz_xy_step(
            current_joints,
            current_pose,
            [0.249, 0.0],
            **_limits(),
            allow_rejected_audit=True,
        )
        gates = result["audit"]["gates"]
        self.assertTrue(gates["current_inside_local_domain"]["passed"])
        self.assertTrue(gates["target_inside_local_domain"]["passed"])
        self.assertFalse(
            gates["sequential_intermediate_inside_local_domain"]["passed"]
        )

    def test_stage7b_can_disable_only_the_intermediate_domain_gate(self) -> None:
        setup = plan_fixed_rz_xy_step(
            P22_JOINTS,
            P22_POSE,
            [-1.79, -1.79],
            **{
                **_limits(),
                "domain_margin_mm": 0.0,
                "max_xy_step_norm_mm": 3.0,
                "max_xy_axis_mm": 3.0,
                "max_sequential_transient_xy_mm": 20.0,
                "rz_tolerance_deg": 20.0,
            },
            allow_rejected_audit=True,
        )
        current_joints = setup["target_joints"]
        current_xy = fk_wrist(current_joints[0], current_joints[1])
        current_pose = [*current_xy, P22_JOINTS[2], 180.0, 0.0, P22_RZ]
        result = plan_fixed_rz_xy_step(
            current_joints,
            current_pose,
            [0.249, 0.0],
            **_limits(),
            enforce_sequential_intermediate_domain=False,
        )
        audit = result["audit"]
        self.assertTrue(audit["passed"])
        self.assertFalse(audit["sequential_intermediate_domain_enforced"])
        self.assertNotIn(
            "sequential_intermediate_inside_local_domain", audit["gates"]
        )
        self.assertTrue(audit["gates"]["current_inside_local_domain"]["passed"])
        self.assertTrue(audit["gates"]["target_inside_local_domain"]["passed"])

    def test_audit_rejects_endpoint_outside_local_domain(self) -> None:
        # Construct an already-shifted start without pretending that Stage7A
        # itself is allowed to make the 1.79 mm setup move.  The production
        # per-step transient limit remains unchanged.
        current_from_shift = plan_fixed_rz_xy_step(
            P22_JOINTS,
            P22_POSE,
            [1.79, 0.0],
            **{
                **_limits(),
                "domain_margin_mm": 0.0,
                "max_xy_step_norm_mm": 2.0,
                "max_xy_axis_mm": 2.0,
                "max_sequential_transient_xy_mm": 10.0,
                "rz_tolerance_deg": 10.0,
            },
        )["target_joints"]
        current_xy = fk_wrist(current_from_shift[0], current_from_shift[1])
        current_pose = [*current_xy, P22_JOINTS[2], 180.0, 0.0, P22_RZ]
        unsafe_target = plan_fixed_rz_xy_step(
            current_from_shift,
            current_pose,
            [0.20, 0.0],
            **{**_limits(), "domain_margin_mm": 0.0},
        )["target_joints"]
        audit = audit_fixed_rz_xy_target(
            current_from_shift,
            current_pose,
            unsafe_target,
            **_limits(),
        )
        self.assertTrue(math.isfinite(unsafe_target[0]))
        self.assertFalse(audit["gates"]["target_inside_local_domain"]["passed"])

    def test_loads_only_explicit_p22_float(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "scara_presets.json").write_text(
                '{"P22 float":[30.0,80.0,-27.0,0.0]}',
                encoding="utf-8",
            )
            name, joints = load_p22_float_preset(root)
            self.assertEqual(name, "P22 float")
            self.assertEqual(joints, [30.0, 80.0, -27.0, 0.0])


if __name__ == "__main__":
    unittest.main()
