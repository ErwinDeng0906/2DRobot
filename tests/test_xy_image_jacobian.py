"""Pure unit tests for Stage-5 local image Jacobian calibration.

No test in this module imports a robot controller, opens a camera, or issues a
motion command.  Synthetic measurements exercise the estimator and temporary
directories exercise the Task-8 suction-target loader.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.handeye_interaction import load_latest_suction_target
from scara.vision.xy_image_jacobian import (
    XYImageJacobianQualityConfig,
    correction_command_xy_mm,
    fit_local_xy_image_jacobian,
)


J_TRUE = np.array([[2.40, -0.70], [0.50, 1.80]], dtype=np.float64)
INTERCEPT_TRUE = np.array([10.0, -4.0], dtype=np.float64)
GRID_OFFSETS = [
    (-2.0, -2.0),
    (0.0, -2.0),
    (2.0, -2.0),
    (-2.0, 0.0),
    (0.0, 0.0),
    (2.0, 0.0),
    (-2.0, 2.0),
    (0.0, 2.0),
    (2.0, 2.0),
]


def _quality(**overrides: object) -> XYImageJacobianQualityConfig:
    values: dict[str, object] = {
        "minimum_frames_per_offset": 5,
        "minimum_distinct_offsets": 7,
        "maximum_fit_rms_px": 0.25,
        "maximum_cross_validation_rms_px": 0.50,
    }
    values.update(overrides)
    return XYImageJacobianQualityConfig(**values)


def _synthetic_samples(
    offsets: Iterable[tuple[float, float]],
    *,
    jacobian: np.ndarray = J_TRUE,
    intercept: np.ndarray = INTERCEPT_TRUE,
    frames_per_offset: int = 8,
    noise_sigma_px: float = 0.02,
    seed: int = 20260814,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    samples: list[dict[str, object]] = []
    for offset in offsets:
        command = np.asarray(offset, dtype=np.float64)
        for _frame_index in range(frames_per_offset):
            error = jacobian @ command + intercept
            if noise_sigma_px:
                error = error + rng.normal(0.0, noise_sigma_px, 2)
            samples.append(
                {
                    "accepted": True,
                    "command_offset_xy_mm": command.astype(float).tolist(),
                    "image_error_px": error.astype(float).tolist(),
                }
            )
    return samples


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write_suction_result(
    path: Path,
    *,
    intrinsics_hash: str,
    geometry_hash: str,
    point: tuple[float, float, float],
    status: str = "success",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "fit": {
            "status": status,
            "p_C_S_mm": list(point),
            "target_pixel_distorted_px": [625.5, 218.5],
        },
        "locked_inputs": {
            "camera_intrinsics_sha256": intrinsics_hash,
            "tray_geometry_sha256": geometry_hash,
        },
        "camera": {
            "source_index": 1,
            "resolution": {"width": 1280, "height": 720},
        },
        "coordinate_definition": {
            "working_plane_z_T_mm": -2.0,
            "imaging_j3_mm": -27.0046,
            "rz_mean_deg": 20.8209,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class LocalJacobianFitTests(unittest.TestCase):
    def test_recovers_known_non_diagonal_jacobian(self) -> None:
        result = fit_local_xy_image_jacobian(
            _synthetic_samples(GRID_OFFSETS), _quality()
        )

        self.assertEqual(result["status"], "success", result["failure_reasons"])
        np.testing.assert_allclose(
            result["j_error_px_per_command_mm"], J_TRUE, atol=0.015
        )
        np.testing.assert_allclose(
            result["intercept_error_px"], INTERCEPT_TRUE, atol=0.015
        )
        inverse = np.asarray(result["j_command_mm_per_error_px"])
        np.testing.assert_allclose(inverse @ J_TRUE, np.eye(2), atol=0.01)

    def test_rejects_one_bad_frame_and_one_bad_offset(self) -> None:
        samples = _synthetic_samples(GRID_OFFSETS)

        # One corrupt camera/pose frame must be removed within its offset burst.
        samples[3]["image_error_px"] = [900.0, -700.0]

        # A whole repeatable but wrong offset burst cannot be found by the
        # within-burst median; it must be removed by the global robust fit.
        bad_offset = (2.0, 2.0)
        for sample in samples:
            if tuple(sample["command_offset_xy_mm"]) == bad_offset:
                sample["image_error_px"] = [40.0, -30.0]

        result = fit_local_xy_image_jacobian(samples, _quality())

        self.assertEqual(result["status"], "success", result["failure_reasons"])
        self.assertEqual(result["rejected_offsets_xy_mm"], [[2.0, 2.0]])
        row = next(
            aggregate
            for aggregate in result["offset_aggregates"]
            if aggregate["command_offset_xy_mm"] == [-2.0, -2.0]
        )
        self.assertEqual(row["frame_count"], 8)
        self.assertEqual(row["used_frame_count"], 7)
        np.testing.assert_allclose(
            result["j_error_px_per_command_mm"], J_TRUE, atol=0.02
        )

    def test_fails_for_insufficient_or_affinely_collinear_offsets(self) -> None:
        too_few = fit_local_xy_image_jacobian(
            _synthetic_samples(GRID_OFFSETS[:6]), _quality()
        )
        self.assertEqual(too_few["status"], "failure")
        self.assertIn("usable offsets 6/7", too_few["failure_reasons"])

        # These points have raw matrix rank 2 because the line does not pass
        # through the origin, but their centered rank is only 1.  An affine
        # fit cannot distinguish both Jacobian columns from its intercept.
        collinear = [(float(x), float(x + 1)) for x in range(-3, 4)]
        no_xy_coverage = fit_local_xy_image_jacobian(
            _synthetic_samples(collinear), _quality()
        )
        self.assertEqual(no_xy_coverage["status"], "failure")
        self.assertIn(
            "command offsets do not span both X and Y",
            no_xy_coverage["failure_reasons"],
        )

    def test_fails_closed_for_singular_image_response(self) -> None:
        singular_j = np.array([[2.0, -1.0], [4.0, -2.0]], dtype=np.float64)
        result = fit_local_xy_image_jacobian(
            _synthetic_samples(
                GRID_OFFSETS,
                jacobian=singular_j,
                noise_sigma_px=0.0,
            ),
            _quality(),
        )

        self.assertEqual(result["status"], "failure")
        self.assertFalse(
            result["quality_gates"]["minimum_singular_value"]["passed"]
        )
        self.assertIsNone(correction_command_xy_mm([3.0, -2.0], result))

    def test_correction_uses_negative_inverse_feedback_sign(self) -> None:
        result = fit_local_xy_image_jacobian(
            _synthetic_samples(GRID_OFFSETS, noise_sigma_px=0.0), _quality()
        )
        current_error = np.array([6.5, -3.25], dtype=np.float64)

        correction = correction_command_xy_mm(current_error, result)

        self.assertIsNotNone(correction)
        expected = -np.linalg.solve(J_TRUE, current_error)
        np.testing.assert_allclose(correction, expected, atol=1e-12)
        predicted_next_error = current_error + J_TRUE @ np.asarray(correction)
        np.testing.assert_allclose(predicted_next_error, [0.0, 0.0], atol=1e-12)


class SuctionTargetLoaderTests(unittest.TestCase):
    def test_selects_latest_success_with_matching_current_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calib = root / "src/scara/calib"
            calib.mkdir(parents=True)
            intrinsics = calib / "camera1_intrinsics.json"
            geometry = calib / "tray_board_geometry.json"
            intrinsics.write_text("intrinsics-v1", encoding="utf-8")
            geometry.write_text("geometry-v1", encoding="utf-8")
            intrinsics_hash = _sha256(intrinsics)
            geometry_hash = _sha256(geometry)

            old_valid = (
                root
                / "Trajectory Photos/260814100000/camera1_suction_target.json"
            )
            newest_valid = (
                root
                / "Trajectory Photos/260814110000/camera1_suction_target.json"
            )
            newest_wrong_hash = (
                root
                / "Trajectory Photos/260814120000/camera1_suction_target.json"
            )
            newest_failed = (
                root
                / "Trajectory Photos/260814130000/camera1_suction_target.json"
            )
            _write_suction_result(
                old_valid,
                intrinsics_hash=intrinsics_hash,
                geometry_hash=geometry_hash,
                point=(1.0, 2.0, 300.0),
            )
            _write_suction_result(
                newest_valid,
                intrinsics_hash=intrinsics_hash,
                geometry_hash=geometry_hash,
                point=(4.0, 5.0, 310.0),
            )
            _write_suction_result(
                newest_wrong_hash,
                intrinsics_hash="A" * 64,
                geometry_hash=geometry_hash,
                point=(7.0, 8.0, 320.0),
            )
            _write_suction_result(
                newest_failed,
                intrinsics_hash=intrinsics_hash,
                geometry_hash=geometry_hash,
                point=(9.0, 10.0, 330.0),
                status="failure",
            )

            loaded = load_latest_suction_target(root)

            self.assertEqual(loaded.source_path, newest_valid)
            self.assertEqual(loaded.p_C_S_mm, (4.0, 5.0, 310.0))
            self.assertEqual(loaded.source_sha256, _sha256(newest_valid))

            # Changing a locked current input must invalidate every old result.
            intrinsics.write_text("intrinsics-v2", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "intrinsics hash"):
                load_latest_suction_target(root)


if __name__ == "__main__":
    unittest.main()
