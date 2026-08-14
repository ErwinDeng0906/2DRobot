"""Offline contract tests for Task9 and its Stage-5 runtime helpers."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
import threading
import time
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PyQt6.QtCore import QObject


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import scara.file_io as file_io
from scara.file_io import atomic_write_text, replace_with_retry
from scara.pipeline.kinematics import fk_wrist, rz_of
from scara.ui.action_worker import ActionWorker, normalize_action_task
from scara.vision.xy_image_jacobian_runtime import (
    XYImageJacobianCalibrationRuntime,
    find_latest_successful_suction_target,
    project_tray_point_distorted,
)


P22_JOINTS = [30.6646, 84.7845, -27.0046, -4.6268]


def _os_error_with_winerror(code: int) -> OSError:
    error = PermissionError(f"synthetic Windows error {code}")
    error.winerror = int(code)
    return error


def _load_task9_module():
    path = PROJECT_ROOT / "Preset Trajectories/task9_jacobiantest.py"
    spec = importlib.util.spec_from_file_location("_task9_contract_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load task9")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_suction_result(
    path: Path,
    *,
    intrinsics_hash: str,
    geometry_hash: str,
    status: str = "success",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": status,
                "fit": {
                    "status": status,
                    "p_C_S_mm": [-2.4, -33.0, 149.5],
                    "target_pixel_distorted_px": [625.5, 218.5],
                },
                "locked_inputs": {
                    "camera_intrinsics_sha256": intrinsics_hash,
                    "tray_geometry_sha256": geometry_hash,
                },
            }
        ),
        encoding="utf-8",
    )


class ManifestAtomicIOTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "requires Windows file sharing semantics")
    def test_open_target_breaks_raw_replace_but_bounded_writer_recovers(self) -> None:
        """Reproduce the old WinError 5, then exercise the production fix."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "points.json"
            legacy_temp = root / "points.json.tmp"
            target.write_text('{"generation": 1}', encoding="utf-8")
            legacy_temp.write_text('{"generation": 2}', encoding="utf-8")

            reader = target.open("r", encoding="utf-8")
            try:
                with self.assertRaises(PermissionError) as caught:
                    os.replace(legacy_temp, target)
                self.assertIn(caught.exception.winerror, {5, 32})

                errors: list[BaseException] = []

                def write_with_fix() -> None:
                    try:
                        atomic_write_text(
                            target,
                            '{"generation": 3}',
                            retry_deadline_s=0.75,
                        )
                    except BaseException as exc:  # captured for main test thread
                        errors.append(exc)

                writer = threading.Thread(target=write_with_fix, daemon=True)
                writer.start()
                time.sleep(0.05)
                self.assertTrue(writer.is_alive())
            finally:
                reader.close()
                try:
                    legacy_temp.unlink()
                except FileNotFoundError:
                    pass

            writer.join(timeout=2.0)
            self.assertFalse(writer.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                {"generation": 3},
            )
            self.assertEqual(list(root.glob(f".{target.name}.*.tmp")), [])

    def test_action_worker_write_and_task9_runtime_read_are_serialized(self) -> None:
        """Exercise the actual production methods without Qt or hardware."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "points.json"
            old_payload = {"generation": 1, "points": []}
            new_payload = {"generation": 2, "points": [{"sequence": 1}]}
            target.write_text(json.dumps(old_payload), encoding="utf-8")
            reader_owner = SimpleNamespace(manifest_path=target)
            writer_owner = SimpleNamespace(
                manifest_path=target,
                _manifest=new_payload,
            )
            reader_entered = threading.Event()
            release_reader = threading.Event()
            reader_result: list[dict[str, object]] = []
            errors: list[BaseException] = []
            original_read_text = Path.read_text

            def blocking_read_text(
                path: Path,
                *args: object,
                **kwargs: object,
            ) -> str:
                if Path(path).resolve() != target.resolve():
                    return original_read_text(path, *args, **kwargs)
                encoding = str(kwargs.get("encoding") or "utf-8")
                with Path(path).open("r", encoding=encoding) as stream:
                    reader_entered.set()
                    if not release_reader.wait(timeout=2.0):
                        raise TimeoutError("test reader was not released")
                    return stream.read()

            def read_manifest() -> None:
                try:
                    reader_result.append(
                        XYImageJacobianCalibrationRuntime._load_manifest(
                            reader_owner
                        )
                    )
                except BaseException as exc:
                    errors.append(exc)

            def write_manifest() -> None:
                try:
                    ActionWorker._save_manifest(writer_owner)
                except BaseException as exc:
                    errors.append(exc)

            with patch.object(Path, "read_text", blocking_read_text):
                reader_thread = threading.Thread(target=read_manifest, daemon=True)
                reader_thread.start()
                self.assertTrue(reader_entered.wait(timeout=1.0))
                writer_thread = threading.Thread(target=write_manifest, daemon=True)
                writer_thread.start()
                time.sleep(0.05)
                self.assertTrue(writer_thread.is_alive())
                release_reader.set()
                reader_thread.join(timeout=2.0)
                writer_thread.join(timeout=2.0)

            self.assertFalse(reader_thread.is_alive())
            self.assertFalse(writer_thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(reader_result, [old_payload])
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                new_payload,
            )

    def test_replace_retries_winerror_5_and_32_then_succeeds(self) -> None:
        errors = [_os_error_with_winerror(5), _os_error_with_winerror(32)]
        with (
            patch.object(
                file_io.os,
                "replace",
                side_effect=[*errors, None],
            ) as replace_mock,
            patch.object(file_io.time, "sleep") as sleep_mock,
            patch.object(
                file_io.time,
                "monotonic",
                side_effect=[100.0, 100.1, 100.2],
            ),
        ):
            replace_with_retry(
                Path("source.tmp"),
                Path("target.json"),
                deadline_s=0.75,
            )

        self.assertEqual(replace_mock.call_count, 3)
        self.assertEqual(sleep_mock.call_count, 2)

    def test_replace_nonsharing_error_is_not_retried(self) -> None:
        error = _os_error_with_winerror(3)
        with (
            patch.object(file_io.os, "replace", side_effect=error) as replace_mock,
            patch.object(file_io.time, "sleep") as sleep_mock,
        ):
            with self.assertRaises(PermissionError) as caught:
                replace_with_retry(Path("source.tmp"), Path("target.json"))

        self.assertIs(caught.exception, error)
        replace_mock.assert_called_once()
        sleep_mock.assert_not_called()

    def test_replace_timeout_is_bounded_and_unique_temp_is_cleaned(self) -> None:
        error = _os_error_with_winerror(5)
        with (
            patch.object(file_io.os, "replace", side_effect=error) as replace_mock,
            patch.object(file_io.time, "sleep") as sleep_mock,
            patch.object(
                file_io.time,
                "monotonic",
                side_effect=[10.0, 10.4, 10.8],
            ),
        ):
            with self.assertRaises(PermissionError):
                replace_with_retry(
                    Path("source.tmp"),
                    Path("target.json"),
                    deadline_s=0.75,
                )
        self.assertEqual(replace_mock.call_count, 2)
        sleep_mock.assert_called_once()
        with self.assertRaises(ValueError):
            replace_with_retry(
                Path("source.tmp"),
                Path("target.json"),
                deadline_s=1.001,
            )

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "points.json"
            attempted_temps: list[Path] = []

            def fail_replace(source: Path, _target: Path, **_kwargs: object) -> None:
                attempted_temps.append(Path(source))
                raise error

            with patch.object(file_io, "replace_with_retry", side_effect=fail_replace):
                for generation in (1, 2):
                    with self.assertRaises(PermissionError):
                        atomic_write_text(target, json.dumps({"generation": generation}))
            self.assertEqual(len(attempted_temps), 2)
            self.assertNotEqual(attempted_temps[0], attempted_temps[1])
            self.assertTrue(
                all(path != target.with_suffix(".json.tmp") for path in attempted_temps)
            )
            self.assertTrue(all(not path.exists() for path in attempted_temps))
            self.assertFalse(target.with_suffix(".json.tmp").exists())


class Task9ActionContractTests(unittest.TestCase):
    def test_action_is_bounded_fixed_pose_camera1_only_and_returns_p22(self) -> None:
        module = _load_task9_module()
        task = normalize_action_task(
            module.build_action_for_preset("P22 float", P22_JOINTS)
        )
        actions = task["actions"]

        self.assertEqual(actions[0]["type"], "assert_joints")
        self.assertEqual(actions[0]["joints"], P22_JOINTS)
        self.assertNotIn("move_xyzr", {step["type"] for step in actions})
        self.assertEqual(
            {step["source"] for step in actions if step["type"] == "capture"},
            {1},
        )
        self.assertEqual(
            sum(step["type"] == "record_point" for step in actions), 108
        )
        self.assertEqual(sum(step["type"] == "capture" for step in actions), 108)

        parsed = [
            module._record_name(dx, dy, frame)
            for dx, dy in module.GRID_OFFSETS_XY_MM
            for frame in range(1, module.FRAMES_PER_OFFSET + 1)
        ]
        actual_names = [
            step["name"] for step in actions if step["type"] == "record_point"
        ]
        self.assertEqual(actual_names, parsed)
        offset_counts = Counter(
            XYImageJacobianCalibrationRuntime.parse_point_name(name)[1]
            for name in actual_names
        )
        self.assertEqual(set(offset_counts), set(module.GRID_OFFSETS_XY_MM))
        self.assertEqual(set(offset_counts.values()), {12})

        expected_rz = rz_of(P22_JOINTS[0], P22_JOINTS[1], P22_JOINTS[3])
        move_steps = [step for step in actions if step["type"] == "move_joints"]
        self.assertEqual(len(move_steps), 20)

        # The nine acquisition offsets and 108 captures are unchanged; only
        # unsampled <= 1 mm Cartesian transit targets have been inserted.
        previous_xy = fk_wrist(P22_JOINTS[0], P22_JOINTS[1])
        segment_lengths: list[float] = []
        for step in move_steps:
            joints = step["joints"]
            self.assertAlmostEqual(joints[2], P22_JOINTS[2], places=9)
            self.assertAlmostEqual(
                step["require_current_j3_mm"], P22_JOINTS[2], places=9
            )
            self.assertAlmostEqual(
                rz_of(joints[0], joints[1], joints[3]), expected_rz, places=9
            )
            xy = fk_wrist(joints[0], joints[1])
            segment_lengths.append(math.dist(previous_xy, xy))
            previous_xy = xy
        self.assertLessEqual(max(segment_lengths), 1.0001)

        # ScaraController.goto_joints_sync commands axes in this exact order.
        # Audit every intermediate physical state rather than final endpoints;
        # endpoint-only checks hid the original > 2 mm transient jump.
        controller_joint_order = (0, 1, 2, 3)  # J1 -> J2 -> J3 -> J4
        current_joints = list(P22_JOINTS)
        previous_xy = fk_wrist(current_joints[0], current_joints[1])
        maximum_translation = -1.0
        maximum_location = (-1, -1)
        maximum_instantaneous_rz = float("nan")
        instantaneous_rz_values: list[float] = []
        for move_index, step in enumerate(move_steps, start=1):
            target = step["joints"]
            for joint_index in controller_joint_order:
                current_joints[joint_index] = target[joint_index]
                current_xy = fk_wrist(current_joints[0], current_joints[1])
                translation = math.dist(previous_xy, current_xy)
                instantaneous_rz = rz_of(
                    current_joints[0], current_joints[1], current_joints[3]
                )
                instantaneous_rz_values.append(instantaneous_rz)
                if translation > maximum_translation:
                    maximum_translation = translation
                    maximum_location = (move_index, joint_index + 1)
                    maximum_instantaneous_rz = instantaneous_rz
                self.assertLessEqual(
                    translation,
                    2.0001,
                    msg=(
                        f"move {move_index} J{joint_index + 1} transient "
                        f"translation was {translation:.6f} mm"
                    ),
                )
                previous_xy = current_xy

        print(
            "\nTask9 sequential-axis audit: "
            f"max_translation={maximum_translation:.6f} mm at "
            f"move={maximum_location[0]} J{maximum_location[1]}; "
            f"instantaneous_Rz={maximum_instantaneous_rz:.6f} deg; "
            f"Rz_range=[{min(instantaneous_rz_values):.6f}, "
            f"{max(instantaneous_rz_values):.6f}] deg"
        )
        np.testing.assert_allclose(move_steps[-1]["joints"], P22_JOINTS, atol=0.0)

    def test_grid_fk_offsets_match_declared_world_xy(self) -> None:
        module = _load_task9_module()
        targets = module.generate_grid_joint_targets(P22_JOINTS)
        anchor_xy = np.asarray(fk_wrist(P22_JOINTS[0], P22_JOINTS[1]))
        for offset, joints in targets.items():
            actual = np.asarray(fk_wrist(joints[0], joints[1])) - anchor_xy
            np.testing.assert_allclose(actual, offset, atol=1e-9)


class Task9RuntimeHelperTests(unittest.TestCase):
    def test_parse_point_name_rejects_unrelated_names(self) -> None:
        parsed = XYImageJacobianCalibrationRuntime.parse_point_name(
            "TASK9|target=P22|dx=-2.000|dy=+0.000|frame=01/12"
        )
        self.assertEqual(parsed, ("P22", (-2.0, 0.0), 1, 12))
        with self.assertRaisesRegex(RuntimeError, "Task9"):
            XYImageJacobianCalibrationRuntime.parse_point_name(
                "TASK8|P22|frame=01/20"
            )

    def test_projection_uses_full_transform_and_distorted_pixel_convention(self) -> None:
        K = np.array(
            [[600.0, 0.0, 640.0], [0.0, 620.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = [10.0, -20.0, 500.0]
        pixel = project_tray_point_distorted(
            [40.0, 30.0, 0.0], transform, K, np.zeros((5, 1))
        )
        self.assertAlmostEqual(pixel[0], 700.0)
        self.assertAlmostEqual(pixel[1], 372.4)

    def test_latest_suction_loader_skips_failure_and_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_suction_result(
                root / "src/scara/calib/camera1_suction_target.json",
                intrinsics_hash="OLD",
                geometry_hash="GEOMETRY",
            )
            valid = (
                root
                / "Trajectory Photos/260814120000/camera1_suction_target.json"
            )
            _write_suction_result(
                valid,
                intrinsics_hash="INTRINSICS",
                geometry_hash="GEOMETRY",
            )
            _write_suction_result(
                root
                / "Trajectory Photos/260814130000/camera1_suction_target.json",
                intrinsics_hash="INTRINSICS",
                geometry_hash="GEOMETRY",
                status="failure",
            )

            path, report = find_latest_successful_suction_target(
                root,
                expected_intrinsics_sha256="INTRINSICS",
                expected_geometry_sha256="GEOMETRY",
            )

            self.assertEqual(path, valid.resolve())
            self.assertEqual(report["status"], "success")

    def test_finalizer_writes_run_outputs_and_installs_only_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "Trajectory Photos/260814200000"
            output.mkdir(parents=True)
            offsets = [
                (0.0, 0.0),
                (-2.0, 0.0),
                (-2.0, -2.0),
                (0.0, -2.0),
                (2.0, -2.0),
                (2.0, 0.0),
                (2.0, 2.0),
                (0.0, 2.0),
                (-2.0, 2.0),
            ]
            records: list[dict[str, object]] = []
            points: list[dict[str, object]] = []
            jacobian = np.array([[2.4, -0.7], [0.5, 1.8]])
            intercept = np.array([10.0, -4.0])
            sequence = 0
            for offset in offsets:
                for frame_index in range(1, 13):
                    sequence += 1
                    error = jacobian @ np.asarray(offset) + intercept
                    records.append(
                        {
                            "filename": f"1_{sequence:03d}.jpg",
                            "photo_sequence": sequence,
                            "point_sequence": sequence,
                            "anchor_target_name": "P22",
                            "command_offset_xy_mm": list(offset),
                            "measured_offset_xy_mm": list(offset),
                            "command_tracking_error_mm": 0.0,
                            "frame_index": frame_index,
                            "frame_total": 12,
                            "stage3": {"quality_passed": True},
                            "temporal_quality": {"accepted_by_tracker": True},
                            "projection_T_C_T": np.eye(4).tolist(),
                            "slot_pixel_distorted_px": error.tolist(),
                            "suction_target_pixel_distorted_px": [625.5, 218.5],
                            "image_error_px": error.tolist(),
                            "fixed_pose_quality": {
                                "j3_drift_mm": 0.0,
                                "rz_drift_deg": 0.0,
                            },
                            "accepted": True,
                            "rejection_reasons": [],
                        }
                    )
                    points.append({"sequence": sequence})
            (output / "points.json").write_text(
                json.dumps({"points": points, "photos": []}), encoding="utf-8"
            )

            runtime = XYImageJacobianCalibrationRuntime.__new__(
                XYImageJacobianCalibrationRuntime
            )
            QObject.__init__(runtime)
            runtime.output_dir = output
            runtime.project_root = root
            runtime.anchor_target_name = "P22"
            runtime.anchor_point_T_mm = [-50.0, -50.0, -2.0]
            runtime.anchor_preset_joints = list(P22_JOINTS)
            runtime.command_offsets_xy_mm = [list(offset) for offset in offsets]
            runtime.frames_per_offset = 12
            from scara.vision.xy_image_jacobian import (
                DEFAULT_XY_IMAGE_JACOBIAN_QUALITY,
            )

            runtime.quality = DEFAULT_XY_IMAGE_JACOBIAN_QUALITY
            runtime.intrinsics_path = root / "camera1_intrinsics.json"
            runtime.geometry_path = root / "tray_board_geometry.json"
            runtime.suction_target_path = root / "camera1_suction_target.json"
            runtime.intrinsics_hash = "A" * 64
            runtime.geometry_hash = "B" * 64
            runtime.suction_target_hash = "C" * 64
            runtime.intrinsics = SimpleNamespace(image_size=(1280, 720))
            runtime.suction_point_C_mm = [-2.4, -33.0, 149.5]
            runtime.suction_target_pixel = [625.5, 218.5]
            runtime.imaging_j3_mm = P22_JOINTS[2]
            runtime.anchor_rz_deg = rz_of(
                P22_JOINTS[0], P22_JOINTS[1], P22_JOINTS[3]
            )
            runtime._anchor_robot_xy_mm = np.array([118.3386, 272.7719])
            runtime._records = records
            runtime._processing_failed = False
            runtime._fatal_messages = []
            runtime._fatal_error_emitted = False

            runtime.on_task_finished(True, "synthetic complete", str(output))

            result_path = output / "camera1_xy_image_jacobian.json"
            installed_path = (
                root / "src/scara/calib/camera1_xy_image_jacobian.json"
            )
            self.assertTrue(result_path.is_file())
            self.assertTrue(installed_path.is_file())
            self.assertTrue(
                (output / "task9_xy_image_jacobian_update.md").is_file()
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["schema_version"], 2)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["camera"]["source_index"], 1)
            self.assertEqual(
                result["coordinate_definition"]["command_frame"],
                "robot_controller_world_XY",
            )
            np.testing.assert_allclose(
                result["coordinate_definition"]["anchor_robot_xy_mm"],
                [118.3386, 272.7719],
            )
            np.testing.assert_allclose(
                result["fit"]["j_error_px_per_command_mm"],
                jacobian,
                atol=1e-12,
            )
            enriched = json.loads(
                (output / "points.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                enriched["stage5_xy_image_jacobian"]["status"], "success"
            )
            self.assertTrue(enriched["points"][0]["stage5_sample_accepted"])

            # Losing the measured zero-offset anchor makes the same otherwise
            # valid records unusable and must not replace the approved file.
            approved_before = installed_path.read_bytes()
            missing_anchor_output = root / "Trajectory Photos/260814200001"
            missing_anchor_output.mkdir(parents=True)
            (missing_anchor_output / "points.json").write_text(
                json.dumps(
                    {
                        "points": [
                            {"sequence": index} for index in range(1, 109)
                        ],
                        "photos": [],
                    }
                ),
                encoding="utf-8",
            )
            runtime.output_dir = missing_anchor_output
            runtime._anchor_robot_xy_mm = None
            with self.assertRaisesRegex(RuntimeError, "质量门未通过"):
                runtime.on_task_finished(
                    True, "synthetic complete", str(missing_anchor_output)
                )
            missing_anchor_report = json.loads(
                (
                    missing_anchor_output / "camera1_xy_image_jacobian.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(missing_anchor_report["schema_version"], 2)
            self.assertEqual(missing_anchor_report["status"], "failure")
            self.assertIsNone(
                missing_anchor_report["coordinate_definition"][
                    "anchor_robot_xy_mm"
                ]
            )
            self.assertIn(
                "missing actual robot XY",
                " ".join(missing_anchor_report["fit"]["failure_reasons"]),
            )
            self.assertEqual(installed_path.read_bytes(), approved_before)

    def test_last_frame_annotation_failure_never_installs_calibration(self) -> None:
        """A fatal error after record append must still fail the whole run."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "Trajectory Photos/260814210000"
            output.mkdir(parents=True)
            offsets = [
                (0.0, 0.0),
                (-2.0, 0.0),
                (-2.0, -2.0),
                (0.0, -2.0),
                (2.0, -2.0),
                (2.0, 0.0),
                (2.0, 2.0),
                (0.0, 2.0),
                (-2.0, 2.0),
            ]
            jacobian = np.array([[2.4, -0.7], [0.5, 1.8]])
            intercept = np.array([10.0, -4.0])
            records: list[dict[str, object]] = []
            sequence = 0
            for offset in offsets:
                for frame_index in range(1, 13):
                    sequence += 1
                    if sequence == 108:
                        break
                    error = jacobian @ np.asarray(offset) + intercept
                    records.append(
                        {
                            "filename": f"1_{sequence:03d}.jpg",
                            "photo_sequence": sequence,
                            "point_sequence": sequence,
                            "anchor_target_name": "P22",
                            "command_offset_xy_mm": list(offset),
                            "measured_offset_xy_mm": list(offset),
                            "command_tracking_error_mm": 0.0,
                            "frame_index": frame_index,
                            "frame_total": 12,
                            "stage3": {"quality_passed": True},
                            "temporal_quality": {"accepted_by_tracker": True},
                            "projection_T_C_T": np.eye(4).tolist(),
                            "slot_pixel_distorted_px": error.tolist(),
                            "suction_target_pixel_distorted_px": [625.5, 218.5],
                            "image_error_px": error.tolist(),
                            "fixed_pose_quality": {
                                "j3_drift_mm": 0.0,
                                "rz_drift_deg": 0.0,
                            },
                            "accepted": True,
                            "rejection_reasons": [],
                        }
                    )
            self.assertEqual(len(records), 107)
            (output / "points.json").write_text(
                json.dumps(
                    {
                        "points": [
                            {"sequence": index} for index in range(1, 109)
                        ],
                        "photos": [],
                    }
                ),
                encoding="utf-8",
            )

            runtime = XYImageJacobianCalibrationRuntime.__new__(
                XYImageJacobianCalibrationRuntime
            )
            QObject.__init__(runtime)
            runtime.output_dir = output
            runtime.project_root = root
            runtime.anchor_target_name = "P22"
            runtime.anchor_point_T_mm = [-50.0, -50.0, -2.0]
            runtime.anchor_preset_joints = list(P22_JOINTS)
            runtime.command_offsets_xy_mm = [list(offset) for offset in offsets]
            runtime.frames_per_offset = 12
            from scara.vision.xy_image_jacobian import (
                DEFAULT_XY_IMAGE_JACOBIAN_QUALITY,
            )

            runtime.quality = DEFAULT_XY_IMAGE_JACOBIAN_QUALITY
            runtime.intrinsics_path = root / "camera1_intrinsics.json"
            runtime.geometry_path = root / "tray_board_geometry.json"
            runtime.suction_target_path = root / "camera1_suction_target.json"
            runtime.intrinsics_hash = "A" * 64
            runtime.geometry_hash = "B" * 64
            runtime.suction_target_hash = "C" * 64
            runtime.intrinsics = SimpleNamespace(
                image_size=(1280, 720),
                K=np.array(
                    [
                        [618.6, 0.0, 635.5],
                        [0.0, 619.9, 355.9],
                        [0.0, 0.0, 1.0],
                    ]
                ),
                dist_coeffs=np.zeros((5, 1)),
            )
            runtime.suction_point_C_mm = [-2.4, -33.0, 149.5]
            runtime.suction_target_pixel = [625.5, 218.5]
            runtime.imaging_j3_mm = P22_JOINTS[2]
            runtime.anchor_rz_deg = rz_of(
                P22_JOINTS[0], P22_JOINTS[1], P22_JOINTS[3]
            )
            runtime._active_offset = (-2.0, 2.0)
            runtime._anchor_robot_xy_mm = np.array([100.0, 200.0])
            runtime._records = records
            runtime._processing_failed = False
            runtime._fatal_messages = []
            runtime._fatal_error_emitted = False

            transform = np.eye(4, dtype=np.float64)
            transform[:3, 3] = [0.0, 0.0, 400.0]
            raw = SimpleNamespace(
                to_json=lambda: {
                    "quality_passed": True,
                    "T_C_T": transform.tolist(),
                },
                annotated_image=np.zeros((720, 1280, 3), dtype=np.uint8),
            )
            tracked = SimpleNamespace(
                raw=raw,
                accepted_by_tracker=True,
                tracker_reason=None,
                translation_jump_mm=0.0,
                rotation_jump_deg=0.0,
                lost_frame_count=0,
                filtered_T_C_T=transform,
                filtered_T_T_C=np.linalg.inv(transform),
            )
            runtime._tracker = SimpleNamespace(update=lambda _image: tracked)
            final_photo = {
                "sequence_for_source": 108,
                "point_sequence": 108,
            }
            final_point = {
                "name": (
                    "TASK9|target=P22|dx=-2.000|dy=+2.000|frame=12/12"
                ),
                "mechanical_center": {"x_mm": 98.0, "y_mm": 202.0},
                "joints": {
                    "J1_deg": P22_JOINTS[0],
                    "J2_deg": P22_JOINTS[1],
                    "J3_mm": P22_JOINTS[2],
                    "J4_deg": P22_JOINTS[3],
                },
            }
            with (
                patch.object(
                    runtime,
                    "_photo_context",
                    return_value=(final_photo, final_point),
                ),
                patch(
                    "scara.vision.xy_image_jacobian_runtime.cv2.imread",
                    return_value=np.zeros((720, 1280, 3), dtype=np.uint8),
                ),
                patch(
                    "scara.vision.xy_image_jacobian_runtime.cv2.imwrite",
                    return_value=False,
                ),
            ):
                runtime.on_photo_saved(str(output / "1_108.jpg"))

            # The numeric record is deliberately complete before annotation
            # persistence fails: this is the previously unsafe race.
            self.assertEqual(len(runtime._records), 108)
            self.assertTrue(runtime._processing_failed)
            self.assertTrue(runtime._fatal_messages)
            self.assertIn("保存Task9标注图失败", runtime._fatal_messages[-1])

            with self.assertRaisesRegex(RuntimeError, "质量门未通过"):
                runtime.on_task_finished(True, "synthetic complete", str(output))

            result = json.loads(
                (output / "camera1_xy_image_jacobian.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(result["schema_version"], 2)
            self.assertEqual(result["status"], "failure")
            self.assertTrue(result["runtime_processing"]["failed"])
            self.assertIsNone(
                result["fit"]["j_error_px_per_command_mm"]
            )
            self.assertFalse(
                (root / "src/scara/calib/camera1_xy_image_jacobian.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
