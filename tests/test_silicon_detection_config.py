from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.silicon_detection_config import (
    load_silicon_detection_config,
    preferred_silicon_detection_config_path,
    save_preferred_silicon_detection_config_path,
    silicon_detection_selection_path,
)
from scara.vision.wafer_shape_quality import WaferQualityConfig


CONFIG_PATH = ROOT / "src/scara/calib/silicon_detection_0818.json"


class SiliconDetectionConfigTests(unittest.TestCase):
    def test_ui_profile_selection_persists_as_project_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            default_path = (
                project / "src/scara/calib/silicon_detection_0818.json"
            )
            default_path.parent.mkdir(parents=True)
            default_path.write_bytes(CONFIG_PATH.read_bytes())
            selected = project / "Trajectory Photos/run/recommended.json"
            selected.parent.mkdir(parents=True)
            selected.write_bytes(CONFIG_PATH.read_bytes())

            self.assertEqual(
                default_path.resolve(),
                preferred_silicon_detection_config_path(project),
            )
            pointer = save_preferred_silicon_detection_config_path(
                project, selected
            )
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            self.assertEqual("project_relative", payload["path_kind"])
            self.assertEqual(selected.resolve(), preferred_silicon_detection_config_path(project))
            self.assertEqual(pointer, silicon_detection_selection_path(project))

    def test_invalid_ui_profile_selection_record_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            pointer = silicon_detection_selection_path(project)
            pointer.write_text('{"schema_version": 1}', encoding="utf-8")
            with self.assertRaises(ValueError):
                preferred_silicon_detection_config_path(project)

    def test_checked_in_profile_lists_every_detector_field(self) -> None:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
        self.assertEqual(
            {field.name for field in fields(WaferQualityConfig)},
            set(payload["wafer_quality"]),
        )
        loaded = load_silicon_detection_config(CONFIG_PATH)
        self.assertEqual(
            "silicon_detection_0820_geometry_robust", loaded.profile_name
        )
        self.assertEqual(192, loaded.fusion_config.canonical_patch_size)
        self.assertAlmostEqual(15.5, loaded.fusion_config.slot_half_extent_mm)
        self.assertAlmostEqual(
            0.62, loaded.fusion_config.wafer_quality.maximum_normal_side_ratio
        )
        self.assertAlmostEqual(
            1.2, loaded.fusion_config.wafer_quality.boundary_max_aspect_ratio
        )
        self.assertEqual(64, len(loaded.source_sha256))

    def test_complete_alternate_profile_is_loaded(self) -> None:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
        payload["profile_name"] = "alternate"
        payload["wafer_quality"]["minimum_chromatic_fraction"] = 0.55
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "alternate.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_silicon_detection_config(path)
        self.assertEqual("alternate", loaded.profile_name)
        self.assertAlmostEqual(
            0.55, loaded.fusion_config.wafer_quality.minimum_chromatic_fraction
        )

    def test_partial_unknown_and_fractional_integer_fields_fail_closed(self) -> None:
        original = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
        variants = []
        partial = json.loads(json.dumps(original))
        partial["wafer_quality"].pop("minimum_area_ratio")
        variants.append(partial)
        unknown = json.loads(json.dumps(original))
        unknown["wafer_quality"]["minimum_area_rato"] = 0.1
        variants.append(unknown)
        fractional = json.loads(json.dumps(original))
        fractional["wafer_quality"]["stacked_internal_line_count"] = 1.5
        variants.append(fractional)
        inverted = json.loads(json.dumps(original))
        inverted["wafer_quality"]["normal_min_solidity"] = 0.4
        variants.append(inverted)
        with tempfile.TemporaryDirectory() as temporary:
            for index, payload in enumerate(variants):
                path = Path(temporary) / f"invalid_{index}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(index=index), self.assertRaises(ValueError):
                    load_silicon_detection_config(path)


if __name__ == "__main__":
    unittest.main()
