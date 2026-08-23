#!/usr/bin/env python3
"""Review normal/outside wafer placement in an arbitrary tray photograph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the tray grid from fixed inner and outer marker IDs and review wafer "
            "placement. This tool never outputs a robot correction."
        )
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-image", type=Path)
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
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    geometry = load_tray_board_geometry(args.geometry)
    layout = load_slot_marker_layout(args.layout)
    config_path = args.silicon_config or default_silicon_detection_config_path(ROOT)
    config = load_silicon_detection_config(config_path).fusion_config
    result = review_marker_grid_image(image, geometry, layout, config)
    output_json = args.output_json or args.image.with_name(
        args.image.stem + "_wafer_review.json"
    )
    output_image = args.output_image or args.image.with_name(
        args.image.stem + "_wafer_review.png"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result.to_json(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not cv2.imwrite(str(output_image), result.annotated_image):
        raise SystemExit(f"cannot write annotated image: {output_image}")
    print(
        json.dumps(
            {
                "success": result.success,
                "failure_reason": result.failure_reason,
                "summary": result.summary,
                "output_json": str(output_json),
                "output_image": str(output_image),
                "robot_motion_authorized": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
