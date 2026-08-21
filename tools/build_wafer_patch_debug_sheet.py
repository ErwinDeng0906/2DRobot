#!/usr/bin/env python3
"""Render all 36 normalized slot patches for one wafer-review image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


STATE_COLOURS = {
    "occupied": (255, 0, 255),
    "outside_slot": (0, 0, 255),
    "stacked": (0, 0, 220),
    "stacked_outside_slot": (0, 0, 180),
    "warning": (0, 180, 255),
    "empty": (50, 180, 60),
    "unknown": (255, 180, 0),
    "out_of_view": (110, 110, 110),
}


def build_sheet(image_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {image_path}")
    geometry = load_tray_board_geometry(
        ROOT / "src/scara/calib/tray_board_geometry.json"
    )
    layout = load_slot_marker_layout(
        ROOT / "tools/tray_marker_detector_v2/tray_marker_layout.json"
    )
    loaded = load_silicon_detection_config(
        default_silicon_detection_config_path(ROOT)
    )
    config = loaded.fusion_config
    result = review_marker_grid_image(image, geometry, layout, config)
    if not result.success:
        raise RuntimeError(result.failure_reason or "review rejected")

    patch_size = int(config.canonical_patch_size)
    display_size = 252
    header_height = 80
    tile_height = display_size + header_height
    sheet = np.full((6 * tile_height, 6 * display_size, 3), 238, np.uint8)
    target = np.asarray(
        [
            [0.0, 0.0],
            [patch_size - 1.0, 0.0],
            [patch_size - 1.0, patch_size - 1.0],
            [0.0, patch_size - 1.0],
        ],
        dtype=np.float32,
    )
    margin = int(
        round(config.wafer_quality.slot_boundary_margin_ratio * patch_size)
    )
    for slot in result.slots:
        row = int(slot.slot_key[1])
        col = int(slot.slot_key[2])
        source = np.asarray(slot.polygon_px, dtype=np.float32)
        transform = cv2.getPerspectiveTransform(source, target)
        patch = cv2.warpPerspective(image, transform, (patch_size, patch_size))
        colour = STATE_COLOURS.get(slot.state, (255, 255, 255))
        cv2.rectangle(
            patch,
            (margin, margin),
            (patch_size - 1 - margin, patch_size - 1 - margin),
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if slot.wafer.box_patch_px:
            cv2.polylines(
                patch,
                [np.round(np.asarray(slot.wafer.box_patch_px)).astype(np.int32)],
                True,
                colour,
                3,
                cv2.LINE_AA,
            )
        patch = cv2.resize(
            patch,
            (display_size, display_size),
            interpolation=cv2.INTER_NEAREST,
        )
        y0 = row * tile_height
        x0 = col * display_size
        sheet[y0 : y0 + display_size, x0 : x0 + display_size] = patch
        wafer = slot.wafer
        clearance = (
            "--"
            if wafer.minimum_slot_clearance_ratio is None
            else f"{wafer.minimum_slot_clearance_ratio:+.3f}"
        )
        yaw = (
            "--"
            if wafer.yaw_relative_to_tray_deg is None
            else f"{wafer.yaw_relative_to_tray_deg:+.1f}deg"
        )
        cv2.putText(
            sheet,
            f"{slot.slot_key} {slot.state}",
            (x0 + 5, y0 + display_size + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            colour,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            f"gap={clearance} yaw={yaw} side={wafer.side_ratio:.2f}",
            (x0 + 5, y0 + display_size + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
        flags = "+".join(wafer.flags)
        if len(flags) > 34:
            flags = flags[:31] + "..."
        cv2.putText(
            sheet,
            flags or "-",
            (x0 + 5, y0 + display_size + 73),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (55, 55, 55),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"cannot write image: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build_sheet(args.image.expanduser().resolve(), args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
