from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
TASK14_PATH = ROOT / "Tasks" / "task14_silicon test.py"


def _load_task14():
    spec = importlib.util.spec_from_file_location("task14_silicon_detection_action", TASK14_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Task14ActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.task14 = _load_task14()
        cls.action = cls.task14.build_action()

    def test_all_36_slots_use_closed_adjacent_snake(self) -> None:
        route = list(self.task14.TARGET_ORDER)
        self.assertEqual(36, len(route))
        self.assertEqual({f"P{r}{c}" for r in range(6) for c in range(6)}, set(route))
        for left, right in zip(route + ["P00"], (route + ["P00"])[1:]):
            self.assertEqual(
                1,
                abs(int(left[1]) - int(right[1])) + abs(int(left[2]) - int(right[2])),
            )

    def test_five_frames_per_slot_and_automatic_exposure(self) -> None:
        actions = self.action["actions"]
        kinds = [row["type"] for row in actions]
        self.assertEqual(180, kinds.count("record_point"))
        self.assertEqual(180, kinds.count("capture"))
        self.assertEqual(
            {1: {"auto_exposure": True}},
            self.action["camera_capture_settings"],
        )
        self.assertEqual(
            {
                "P01", "P03", "P04", "P12",
                "P15", "P20", "P22", "P23",
            },
            set(self.task14.EXPECTED_NORMAL_WAFER_SLOTS),
        )
        self.assertEqual(
            {
                "P31", "P33", "P35", "P42",
                "P44", "P50", "P52", "P54",
            },
            set(self.task14.EXPECTED_OUTSIDE_WAFER_SLOTS),
        )
        self.assertEqual(
            {"assert_joints", "move_joints", "wait", "record_point", "capture"},
            set(kinds),
        )
        names = [row["name"] for row in actions if row["type"] == "record_point"]
        for target in self.task14.TARGET_ORDER:
            self.assertEqual(5, sum(name.startswith(f"TASK14|{target}|") for name in names))
        self.assertNotIn("set_do", kinds)
        self.assertNotIn("runtime_move_joints", kinds)

    def test_each_adjacent_slot_uses_one_natural_move_at_ui_speed(self) -> None:
        moves = [
            row for row in self.action["actions"] if row["type"] == "move_joints"
        ]
        # P00 is the checked start: 35 direct adjacent moves, then one direct
        # return from the final adjacent slot to the exact taught P00 pose.
        self.assertEqual(36, len(moves))
        self.assertTrue(
            all(row["name"].startswith("Task14自然移动到观察点") for row in moves)
        )
        self.assertTrue(all("中转" not in row["name"] for row in moves))
        # No task speed field/call: ActionWorker uses the controller speed that
        # the operator already selected in the main UI.
        self.assertTrue(all("speed" not in row for row in moves))

    def test_action_normalizer_accepts_auto_and_rejects_invalid_settings(self) -> None:
        from scara.ui.action_worker import normalize_action_task

        normalized = normalize_action_task(self.action)
        self.assertEqual(
            {1: {"auto_exposure": True}}, normalized["camera_capture_settings"]
        )
        invalid = dict(self.action)
        invalid["camera_capture_settings"] = {1: {"exposure": -1.5}}
        with self.assertRaises(ValueError):
            normalize_action_task(invalid)
        invalid["camera_capture_settings"] = {1: {"exposure": -1}}
        with self.assertRaisesRegex(ValueError, "人工操作"):
            normalize_action_task(invalid)
        invalid["camera_capture_settings"] = {1: {"auto_exposure": False}}
        with self.assertRaises(ValueError):
            normalize_action_task(invalid)


class CameraCaptureSettingTests(unittest.TestCase):
    def test_automatic_exposure_is_explicitly_enabled_and_confirmed(self) -> None:
        from scara.ui.action_worker import CameraSourcePool

        class FakeCv2:
            CAP_PROP_AUTO_EXPOSURE = 1
            CAP_PROP_EXPOSURE = 2

        class FakeCapture:
            def __init__(self) -> None:
                self.auto = 0.25
                self.exposure = -5.0
                self.calls = []

            def get(self, prop):
                return self.auto if prop == FakeCv2.CAP_PROP_AUTO_EXPOSURE else self.exposure

            def set(self, prop, value):
                self.calls.append((prop, float(value)))
                if prop == FakeCv2.CAP_PROP_AUTO_EXPOSURE:
                    self.auto = 1.0
                return True

        pool = CameraSourcePool()
        capture = FakeCapture()
        pool._cv2 = FakeCv2
        pool._captures[1] = capture
        ok, reason = pool._apply_capture_setting(1, {"auto_exposure": True})
        self.assertTrue(ok, reason)
        self.assertIn((FakeCv2.CAP_PROP_AUTO_EXPOSURE, 0.75), capture.calls)
        self.assertNotIn((FakeCv2.CAP_PROP_EXPOSURE, -1.0), capture.calls)
        report = pool.capture_settings_report()["1"]
        self.assertTrue(report["applied"]["auto_mode_confirmed"])
        self.assertTrue(report["applied"]["auto_mode_effective"])

    def test_task_fixed_exposure_is_rejected_without_writing_driver(self) -> None:
        from scara.ui.action_worker import CameraSourcePool

        class FakeCv2:
            CAP_PROP_AUTO_EXPOSURE = 1
            CAP_PROP_EXPOSURE = 2

        class FakeCapture:
            def __init__(self) -> None:
                self.exposure = -5.0
                self.calls = []

            def get(self, prop):
                return 1.0 if prop == FakeCv2.CAP_PROP_AUTO_EXPOSURE else self.exposure

            def set(self, prop, value):
                self.calls.append((prop, float(value)))
                if prop == FakeCv2.CAP_PROP_EXPOSURE:
                    self.exposure = float(value)
                return True

        pool = CameraSourcePool()
        capture = FakeCapture()
        pool._cv2 = FakeCv2
        pool._captures[1] = capture
        ok, reason = pool._apply_capture_setting(1, {"exposure": -1})
        self.assertFalse(ok)
        self.assertIn("动态演示UI", reason)
        self.assertEqual(-5.0, capture.exposure)
        self.assertEqual([], capture.calls)

    def test_unknown_mode_readback_accepts_auto_request_and_never_writes_exposure(self) -> None:
        from scara.ui.action_worker import CameraSourcePool

        class FakeCapture:
            def __init__(self) -> None:
                self.calls: list[tuple[int, float]] = []
                self.released = False

            def isOpened(self):
                return True

            def get(self, prop):
                if prop == FakeCv2.CAP_PROP_AUTO_EXPOSURE:
                    return -1.0
                if prop == FakeCv2.CAP_PROP_EXPOSURE:
                    return -5.0
                return 0.0

            def set(self, prop, value):
                self.calls.append((prop, float(value)))
                return True

            def read(self):
                return True, np.full((20, 20, 3), 80, dtype=np.uint8)

            def release(self):
                self.released = True

        capture = FakeCapture()

        class FakeCv2:
            CAP_DSHOW = 700
            CAP_PROP_FRAME_WIDTH = 3
            CAP_PROP_FRAME_HEIGHT = 4
            CAP_PROP_AUTO_EXPOSURE = 21
            CAP_PROP_EXPOSURE = 15

            @staticmethod
            def VideoCapture(_source, _backend):
                return capture

        pool = CameraSourcePool()
        with patch.dict(sys.modules, {"cv2": FakeCv2}):
            ok, reason = pool.open_sources([1])
        self.assertTrue(ok, reason)
        report = pool.capture_settings_report()["1"]["applied"]
        self.assertTrue(report["auto_mode_request_accepted"])
        self.assertTrue(report["auto_mode_effective"])
        self.assertFalse(report["auto_mode_confirmed"])
        self.assertTrue(report["auto_mode_readback_is_advisory"])
        self.assertTrue(report["auto_mode_settled_effective"])
        self.assertFalse(report["auto_mode_settled_confirmed"])
        pool.close()
        self.assertTrue(capture.released)
        self.assertIn((FakeCv2.CAP_PROP_AUTO_EXPOSURE, 0.75), capture.calls)
        self.assertFalse(
            any(prop == FakeCv2.CAP_PROP_EXPOSURE for prop, _value in capture.calls)
        )


class Task14ReportTests(unittest.TestCase):
    def test_each_slot_uses_all_source_frames_and_excludes_invalid_evidence(self) -> None:
        from scara.vision.task14_silicon_detection import summarize_task14_slot

        states = [
            ("P01", "occupied"),  # direct-over-slot frame: excluded
            ("P00", "occupied"),
            ("P02", "warning"),
            ("P03", "outside_slot"),
            ("P04", "occupied"),
            ("P05", "warning"),
            ("P10", "unknown"),
            ("P11", "empty_unread_marker"),
            ("P12", "out_of_view"),
            ("P13", "occluded"),
        ]
        records = []
        for sequence, (capture_target, state) in enumerate(states, start=1):
            records.append(
                {
                    "filename": f"1_{sequence:03d}.jpg",
                    "point_sequence": sequence,
                    "target_name": capture_target,
                    "slot_results": {
                        "P01": {
                            "decision": {"state": state},
                            "wafer": {"flags": []},
                        }
                    },
                    "processing_error": None,
                }
            )

        summary = summarize_task14_slot(
            "P01",
            [0.0, -25.0, -2.0],
            records,
            expected_occupied=True,
            expected_frames=len(records),
        )
        self.assertEqual(10, summary["captured_frame_count"])
        self.assertEqual(5, summary["valid_observation_count"])
        self.assertEqual(
            {"occupied": 2, "outside_slot": 1, "warning": 2},
            summary["state_counts"],
        )
        self.assertEqual(
            {
                "capture_target_tool_occlusion": 1,
                "marker_unread": 1,
                "occluded": 1,
                "out_of_view": 1,
                "unknown": 1,
            },
            summary["exclusion_counts"],
        )
        self.assertTrue(summary["baseline_passed"])

    def test_root_1_xxx_photo_is_annotated_while_raw_capture_is_preserved(self) -> None:
        from scara.vision.task14_silicon_detection import (
            ANNOTATED_DIRECTORY,
            RAW_DIRECTORY,
            Task14SiliconDetectionRuntime,
        )

        task14 = _load_task14()
        geometry = json.loads(
            (ROOT / "src/scara/calib/tray_board_geometry.json").read_text(
                encoding="utf-8-sig"
            )
        )
        with tempfile.TemporaryDirectory() as temporary_text:
            output = Path(temporary_text)
            runtime = Task14SiliconDetectionRuntime(
                output,
                ROOT,
                task14.TARGET_ORDER,
                geometry["slots"],
                task14.EXPECTED_NORMAL_WAFER_SLOTS,
                1,
                "auto",
                expected_outside_wafer_slots=(
                    task14.EXPECTED_OUTSIDE_WAFER_SLOTS
                ),
                confirm_safety=False,
            )
            path = output / "1_001.jpg"
            raw = np.zeros((180, 320, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(path), raw))
            original_jpeg = path.read_bytes()

            analyzed_means: list[float] = []
            target_slot = SimpleNamespace(
                projection=SimpleNamespace(slot_key="P00"),
                decision=SimpleNamespace(state=SimpleNamespace(value="occupied")),
                to_json=lambda: {"slot_key": "P00", "state": "occupied"},
            )
            annotated = np.zeros_like(raw)
            annotated[:, :, 1] = 255

            def analyze_tracked(image, _tracked):
                analyzed_means.append(float(np.mean(image)))
                return SimpleNamespace(
                    summary={"occupied": 1},
                    slots=(target_slot,),
                    annotated_image=annotated,
                )

            runtime.analyzer = SimpleNamespace(analyze_tracked=analyze_tracked)
            tracked = SimpleNamespace(
                raw=SimpleNamespace(to_json=lambda: {"quality_passed": True}),
                accepted_by_tracker=True,
                tracker_reason="accepted",
                translation_jump_mm=0.0,
                rotation_jump_deg=0.0,
                lost_frame_count=0,
            )
            tracker = SimpleNamespace(update=lambda _image: tracked)

            record = runtime._process_one(path, "P00", 1, 1, tracker)
            self.assertIsNone(record["processing_error"])
            self.assertEqual("1_001.jpg", record["filename"])
            self.assertEqual("tray_vision_annotated", record["saved_photo_kind"])
            self.assertEqual("raw_task14/1_001.jpg", record["raw_source_filename"])
            self.assertEqual(
                original_jpeg,
                (output / RAW_DIRECTORY / "1_001.jpg").read_bytes(),
            )
            root_photo = cv2.imread(str(path), cv2.IMREAD_COLOR)
            self.assertIsNotNone(root_photo)
            self.assertGreater(float(np.mean(root_photo[:, :, 1])), 200.0)
            self.assertTrue((output / ANNOTATED_DIRECTORY / "1_001.jpg").is_file())

            # A report-time retry must still analyze the untouched raw source,
            # not the already annotated root-level display photo.
            runtime._process_one(path, "P00", 1, 1, tracker)
            self.assertEqual(2, len(analyzed_means))
            self.assertLess(max(analyzed_means), 1.0)

    def test_completed_scan_writes_json_with_positions_states_and_tuning_scope(self) -> None:
        from scara.vision.task14_silicon_detection import (
            RECOMMENDED_CONFIG_FILENAME,
            RESULT_FILENAME,
            SUMMARY_FILENAME,
            Task14SiliconDetectionRuntime,
        )
        from scara.vision.silicon_detection_config import load_silicon_detection_config

        geometry_source = ROOT / "src/scara/calib/tray_board_geometry.json"
        intrinsics_source = ROOT / "src/scara/calib/camera1_intrinsics.json"
        layout_source = ROOT / "tools/tray_marker_detector_v2/tray_marker_layout.json"
        geometry = json.loads(geometry_source.read_text(encoding="utf-8-sig"))
        task14 = _load_task14()
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = Path(temporary_text)
            project = temporary / "project"
            calib = project / "src/scara/calib"
            layout_dir = project / "tools/tray_marker_detector_v2"
            calib.mkdir(parents=True)
            layout_dir.mkdir(parents=True)
            shutil.copyfile(geometry_source, calib / geometry_source.name)
            shutil.copyfile(intrinsics_source, calib / intrinsics_source.name)
            shutil.copyfile(
                ROOT / "src/scara/calib/silicon_detection_0818.json",
                calib / "silicon_detection_0818.json",
            )
            shutil.copyfile(layout_source, layout_dir / layout_source.name)
            output = temporary / "run"
            output.mkdir()
            runtime = Task14SiliconDetectionRuntime(
                output,
                project,
                task14.TARGET_ORDER,
                geometry["slots"],
                task14.EXPECTED_NORMAL_WAFER_SLOTS,
                1,
                "auto",
                expected_outside_wafer_slots=(
                    task14.EXPECTED_OUTSIDE_WAFER_SLOTS
                ),
                confirm_safety=False,
            )

            points = []
            photos = []
            sequence = 0
            normal_slots = set(task14.EXPECTED_NORMAL_WAFER_SLOTS)
            outside_slots = set(task14.EXPECTED_OUTSIDE_WAFER_SLOTS)
            wafer_slots = normal_slots | outside_slots
            for target in task14.TARGET_ORDER:
                expected = target in wafer_slots
                if target in normal_slots:
                    state = "warning"
                elif target in outside_slots:
                    state = "outside_slot"
                else:
                    state = "empty"
                frame_index = 1
                sequence += 1
                filename = f"1_{sequence:03d}.jpg"
                slot_results = {}
                for slot_name in sorted(geometry["slots"]):
                    slot_expected = slot_name in wafer_slots
                    if slot_name in normal_slots:
                        slot_state = "warning"
                    elif slot_name in outside_slots:
                        slot_state = "outside_slot"
                    else:
                        slot_state = "empty"
                    slot_results[slot_name] = {
                        "wafer_center_T_mm": (
                            [1.0, 2.0, -2.0] if slot_expected else None
                        ),
                        "wafer_offset_T_mm": [0.1, -0.2] if slot_expected else None,
                        "wafer_offset_distance_mm": 0.224 if slot_expected else None,
                        "decision": {"state": slot_state},
                        "wafer": {
                            "side_ratio": 0.75,
                            "second_component_area_ratio": 0.02,
                            "internal_line_count": 1,
                            "internal_line_score": 0.2,
                            "polygon_vertices": 4,
                            "solidity": 0.96,
                            "flags": [],
                        },
                    }
                runtime._records_by_filename[filename] = {
                    "filename": filename,
                    "point_sequence": sequence,
                    "target_name": target,
                    "frame_index": frame_index,
                    "expected_occupied": expected,
                    "known_slot_center_T_mm": geometry["slots"][target],
                    "observed_state": state,
                    "slot_results": slot_results,
                    "target_slot": slot_results[target],
                    "processing_error": None,
                }
                points.append(
                    {
                        "sequence": sequence,
                        "name": f"TASK14|{target}|frame={frame_index:02d}/01",
                    }
                )
                photos.append(
                    {
                        "source": 1,
                        "filename": filename,
                        "point_sequence": sequence,
                    }
                )
            manifest = {
                "points": points,
                "photos": photos,
                "camera_capture_settings_requested": {"1": {"auto_exposure": True}},
                "camera_capture_settings_applied": {
                    "1": {
                        "applied": {
                            "auto_mode_confirmed": True,
                            "auto_mode_settled_confirmed": True,
                            "auto_mode_settled_readback": 1.0,
                            "exposure_settled_readback": -5.0,
                        },
                    }
                },
            }
            (output / "points.json").write_text(json.dumps(manifest), encoding="utf-8")
            runtime.on_task_finished(True, "动作完成", str(output))

            report = json.loads((output / RESULT_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual("expected_wafers_acceptable", report["status"])
            self.assertEqual(2, report["schema_version"])
            self.assertTrue(report["camera"]["exposure"]["verified_before_motion"])
            self.assertEqual(8, report["summary"]["acceptable_normal_wafer_count"])
            self.assertEqual(8, report["summary"]["acceptable_outside_wafer_count"])
            self.assertEqual(
                sorted(task14.EXPECTED_NORMAL_WAFER_SLOTS),
                report["scan_configuration"]["expected_normal_wafer_slots"],
            )
            self.assertEqual(
                sorted(task14.EXPECTED_OUTSIDE_WAFER_SLOTS),
                report["scan_configuration"]["expected_outside_wafer_slots"],
            )
            self.assertEqual(
                20, len(report["scan_configuration"]["expected_empty_slots"])
            )
            self.assertEqual(36, report["summary"]["processed_frame_count"])
            self.assertEqual(
                "1_XXX.jpg (TrayVision annotated)",
                report["artifacts"]["root_photo_pattern"],
            )
            self.assertEqual("raw_task14", report["artifacts"]["raw_directory"])
            self.assertEqual(
                "silicon_detection_0820_geometry_robust",
                report["locked_inputs"]["silicon_detection_profile_name"],
            )
            self.assertEqual(
                64,
                len(report["locked_inputs"]["silicon_detection_config_sha256"]),
            )
            self.assertTrue(
                report["tuning_scope"][
                    "recommendation_must_be_reviewed_and_selected_manually"
                ]
            )
            self.assertFalse(
                report["parameter_recommendation"]["automatically_applied"]
            )
            self.assertEqual(
                [
                    "tray_vision.slot_half_extent_mm",
                    "tray_vision.canonical_patch_size",
                    "wafer_quality.slot_boundary_margin_ratio",
                ],
                report["parameter_recommendation"][
                    "locked_slot_geometry_parameters"
                ],
            )
            p01 = next(row for row in report["slots"] if row["target_name"] == "P01")
            self.assertEqual("warning", p01["representative_state"])
            self.assertEqual(36, p01["captured_frame_count"])
            self.assertEqual(35, p01["valid_observation_count"])
            self.assertEqual(
                {"capture_target_tool_occlusion": 1}, p01["exclusion_counts"]
            )
            self.assertEqual([1.0, 2.0, -2.0], p01["wafer_center_T_mm_median"])
            enriched = json.loads((output / "points.json").read_text(encoding="utf-8"))
            self.assertIn("task14_silicon_detection", enriched)
            self.assertTrue(
                enriched["task14_silicon_detection"]["root_photos_are_annotated"]
            )
            self.assertEqual(
                "raw_task14",
                enriched["task14_silicon_detection"]["raw_directory"],
            )
            markdown = (output / SUMMARY_FILENAME).read_text(encoding="utf-8")
            self.assertIn("每一个槽都使用全部36张照片", markdown)
            self.assertIn("| P01 | 35/36 | 0 | 35 |", markdown)
            self.assertIn("预期有正常硅片的8槽", markdown)
            self.assertIn("预期有槽外硅片的8槽", markdown)
            self.assertIn("预期为空槽的20槽", markdown)
            self.assertIn("不会自动修改正式配置", markdown)
            recommended = load_silicon_detection_config(
                output / RECOMMENDED_CONFIG_FILENAME
            )
            self.assertTrue(recommended.profile_name.startswith("task14_recommended_"))

    def test_low_light_miss_proposes_review_only_candidate_relaxation(self) -> None:
        from scara.vision.task14_silicon_detection import _recommended_config

        base = json.loads(
            (ROOT / "src/scara/calib/silicon_detection_0818.json").read_text(
                encoding="utf-8-sig"
            )
        )
        summaries = [
            {
                "target_name": "P01",
                "expected_occupied": True,
                "representative_state": "unknown",
            },
            {
                "target_name": "P00",
                "expected_occupied": False,
                "representative_state": "empty",
            },
        ]
        records = [
            {
                "target_name": "P01",
                "target_slot": {
                    "wafer": {"flags": ["no_chromatic_candidate"]}
                },
            }
        ]
        proposal, changes, notes = _recommended_config(
            base,
            summaries,
            records,
            evidence_valid=True,
            run_name="offline",
        )
        self.assertLess(
            proposal["wafer_quality"]["dark_saturation_min"],
            base["wafer_quality"]["dark_saturation_min"],
        )
        self.assertGreater(
            proposal["wafer_quality"]["dark_value_max"],
            base["wafer_quality"]["dark_value_max"],
        )
        self.assertTrue(changes)
        self.assertEqual([], notes)
        self.assertEqual(155, base["wafer_quality"]["dark_value_max"])
        self.assertEqual(base["tray_vision"], proposal["tray_vision"])
        self.assertEqual(
            base["wafer_quality"]["slot_boundary_margin_ratio"],
            proposal["wafer_quality"]["slot_boundary_margin_ratio"],
        )

    def test_area_out_of_range_never_changes_area_thresholds(self) -> None:
        from scara.vision.task14_silicon_detection import _recommended_config

        base = json.loads(
            (ROOT / "src/scara/calib/silicon_detection_0818.json").read_text(
                encoding="utf-8-sig"
            )
        )
        proposal, changes, notes = _recommended_config(
            base,
            [
                {
                    "target_name": "P41",
                    "expected_occupied": True,
                    "representative_state": "unknown",
                },
                {
                    "target_name": "P40",
                    "expected_occupied": False,
                    "representative_state": "empty",
                },
            ],
            [
                {
                    "target_name": "P41",
                    "target_slot": {
                        "wafer": {
                            "flags": ["candidate_area_out_of_range"],
                            "area_ratio": 0.93,
                        }
                    },
                }
            ],
            evidence_valid=True,
            run_name="area_check",
        )
        self.assertEqual(
            base["wafer_quality"]["minimum_area_ratio"],
            proposal["wafer_quality"]["minimum_area_ratio"],
        )
        self.assertEqual(
            base["wafer_quality"]["maximum_area_ratio"],
            proposal["wafer_quality"]["maximum_area_ratio"],
        )
        self.assertFalse(
            any("area_ratio" in str(change["field"]) for change in changes)
        )
        self.assertTrue(any("0.930–0.930" in note for note in notes))
        self.assertEqual(base["tray_vision"], proposal["tray_vision"])

    def test_small_known_wafer_can_lower_wafer_area_without_changing_slot_size(self) -> None:
        from scara.vision.task14_silicon_detection import _recommended_config

        base = json.loads(
            (ROOT / "src/scara/calib/silicon_detection_0818.json").read_text(
                encoding="utf-8-sig"
            )
        )
        proposal, changes, _notes = _recommended_config(
            base,
            [
                {
                    "target_name": "P01",
                    "expected_occupied": True,
                    "representative_state": "unknown",
                },
                {
                    "target_name": "P00",
                    "expected_occupied": False,
                    "representative_state": "empty",
                },
            ],
            [
                {
                    "target_name": "P01",
                    "target_slot": {
                        "wafer": {
                            "flags": ["candidate_area_out_of_range"],
                            "area_ratio": 0.050,
                        }
                    },
                }
            ],
            evidence_valid=True,
            run_name="small_wafer",
        )
        self.assertLess(
            proposal["wafer_quality"]["minimum_area_ratio"],
            base["wafer_quality"]["minimum_area_ratio"],
        )
        self.assertTrue(
            any(
                change["field"] == "wafer_quality.minimum_area_ratio"
                for change in changes
            )
        )
        self.assertEqual(base["tray_vision"], proposal["tray_vision"])
        self.assertEqual(
            base["wafer_quality"]["slot_boundary_margin_ratio"],
            proposal["wafer_quality"]["slot_boundary_margin_ratio"],
        )

    def test_outside_state_never_changes_slot_geometry(self) -> None:
        from scara.vision.task14_silicon_detection import _recommended_config

        base = json.loads(
            (ROOT / "src/scara/calib/silicon_detection_0818.json").read_text(
                encoding="utf-8-sig"
            )
        )
        proposal, changes, notes = _recommended_config(
            base,
            [
                {
                    "target_name": "P03",
                    "expected_occupied": True,
                    "representative_state": "outside_slot",
                }
            ],
            [],
            evidence_valid=True,
            run_name="outside",
        )
        self.assertEqual(base["tray_vision"], proposal["tray_vision"])
        self.assertEqual(
            base["wafer_quality"]["slot_boundary_margin_ratio"],
            proposal["wafer_quality"]["slot_boundary_margin_ratio"],
        )
        self.assertFalse(changes)
        self.assertTrue(any("不会通过修改槽尺寸" in note for note in notes))


if __name__ == "__main__":
    unittest.main()
