#!/usr/bin/env python3
"""Rectify two representative frames per sequence for manual truth review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.marker_grid_wafer_review import review_marker_grid_image
from scara.vision.silicon_detection_config import (
    default_silicon_detection_config_path,
    load_silicon_detection_config,
)
from scara.vision.slot_marker_observation import load_slot_marker_layout
from scara.vision.tray_pose_estimator import load_tray_board_geometry


CANVAS_SIZE = 920
PIXELS_PER_MM = 5.0
ORIGIN_MARGIN_PX = 105.0


def _sequences(paths: Iterable[Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        result.extend(
            json.loads(path.read_text(encoding="utf-8")).get("sequences", [])
        )
    return result


def _canvas_from_tray() -> np.ndarray:
    return np.asarray(
        [
            [0.0, -PIXELS_PER_MM, ORIGIN_MARGIN_PX],
            [-PIXELS_PER_MM, 0.0, ORIGIN_MARGIN_PX],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _candidate_score(result: Any) -> tuple[int, int, float]:
    covered = sum(slot.image_coverage_ratio >= 0.985 for slot in result.slots)
    rms = result.fit.reprojection_rms_px
    return covered, result.fit.inlier_marker_count, -float(rms or 999.0)


def _choose(
    source_dir: Path,
    frames: list[dict[str, Any]],
    geometry: dict[str, Any],
    layout: Any,
    config: Any,
    *,
    full_view: bool,
) -> tuple[Path, np.ndarray, Any] | None:
    candidates = []
    for frame in frames:
        name = str(frame["file"])
        is_full_view = "full view" in name.lower()
        if is_full_view != full_view:
            continue
        path = source_dir / name
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        result = review_marker_grid_image(image, geometry, layout, config)
        if result.success:
            candidates.append((_candidate_score(result), path, image, result))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _score, path, image, result = candidates[0]
    return path, image, result


def _rectify(image: np.ndarray, result: Any) -> np.ndarray:
    image_from_tray = result.fit.homography_image_from_tray_xy
    assert image_from_tray is not None
    canvas_from_image = _canvas_from_tray() @ np.linalg.inv(image_from_tray)
    return cv2.warpPerspective(
        image,
        canvas_from_image,
        (CANVAS_SIZE, CANVAS_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(18, 18, 18),
    )


def _tray_to_canvas(x_mm: float, y_mm: float) -> tuple[int, int]:
    point = _canvas_from_tray() @ np.asarray([x_mm, y_mm, 1.0])
    return int(round(point[0])), int(round(point[1]))


def _annotate_reference(
    rectified: np.ndarray,
    geometry: dict[str, Any],
    *,
    title: str,
    acceptance_half_mm: float,
) -> np.ndarray:
    canvas = rectified.copy()
    cv2.rectangle(canvas, (0, 0), (CANVAS_SIZE - 1, 42), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        title[:100],
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    half_px = int(round(acceptance_half_mm * PIXELS_PER_MM))
    for slot_key, center in sorted(geometry["slots"].items()):
        u, v = _tray_to_canvas(float(center[0]), float(center[1]))
        cv2.rectangle(
            canvas,
            (u - half_px, v - half_px),
            (u + half_px, v + half_px),
            (0, 210, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            slot_key,
            (u - 25, v + 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            slot_key,
            (u - 25, v + 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batches", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--geometry",
        type=Path,
        default=ROOT / "src/scara/calib/tray_board_geometry.json",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=ROOT / "tools/tray_marker_detector_v2/tray_marker_layout.json",
    )
    parser.add_argument("--silicon-config", type=Path)
    args = parser.parse_args()
    geometry = load_tray_board_geometry(args.geometry)
    layout = load_slot_marker_layout(args.layout)
    loaded = load_silicon_detection_config(
        args.silicon_config or default_silicon_detection_config_path(ROOT)
    )
    margin = loaded.fusion_config.wafer_quality.slot_boundary_margin_ratio
    patch_half = loaded.fusion_config.slot_half_extent_mm
    acceptance_half = patch_half * (1.0 - 2.0 * margin)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for sequence_index, sequence in enumerate(_sequences(args.batches), start=1):
        root = Path(sequence["dataset_root"])
        source_dir = root / sequence["sequence"]
        choices = [
            _choose(
                source_dir,
                sequence.get("frames", []),
                geometry,
                layout,
                loaded.fusion_config,
                full_view=True,
            ),
            _choose(
                source_dir,
                sequence.get("frames", []),
                geometry,
                layout,
                loaded.fusion_config,
                full_view=False,
            ),
        ]
        panels = []
        selected_files = []
        for label, choice in zip(("full view", "camera frame"), choices):
            if choice is None:
                panel = np.full(
                    (CANVAS_SIZE, CANVAS_SIZE, 3), 22, dtype=np.uint8
                )
                cv2.putText(
                    panel,
                    f"NO QUALITY-PASSED {label.upper()}",
                    (80, CANVAS_SIZE // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                selected_files.append(None)
            else:
                path, image, result = choice
                panel = _annotate_reference(
                    _rectify(image, result),
                    geometry,
                    title=f"{label}: {path.name}",
                    acceptance_half_mm=acceptance_half,
                )
                selected_files.append(path.name)
            panels.append(panel)
        sheet = np.concatenate(panels, axis=1)
        safe_root = root.name.replace(" ", "_")
        safe_sequence = sequence["sequence"].replace(" ", "_")
        destination = args.output_dir / (
            f"{sequence_index:02d}_{safe_root}_{safe_sequence}.jpg"
        )
        if not cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"cannot write reference map: {destination}")
        index.append(
            {
                "sequence_index": sequence_index,
                "dataset_root": str(root),
                "sequence": sequence["sequence"],
                "reference_map": str(destination),
                "selected_files": selected_files,
            }
        )
    (args.output_dir / "index.json").write_text(
        json.dumps({"sequences": index}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"sequence_count": len(index)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
