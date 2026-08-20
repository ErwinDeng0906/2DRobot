#!/usr/bin/env python3
"""Offline entry point for the layered camera-1 tray analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.slot_marker_observation import load_slot_marker_layout
from scara.vision.silicon_detection_config import load_silicon_detection_config
from scara.vision.tray_pose_estimator import (
    TrayBoardPoseEstimator,
    load_camera_intrinsics,
    load_tray_board_geometry,
)
from scara.vision.tray_vision_fusion import TrayVisionAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the metric tray pose, project all 36 slots, and fuse "
            "slot-marker and wafer-shape evidence. No robot motion is issued."
        )
    )
    parser.add_argument("image", type=Path, help="camera-1 overview image")
    parser.add_argument(
        "--geometry",
        type=Path,
        default=PROJECT_ROOT / "src/scara/calib/tray_board_geometry.json",
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=PROJECT_ROOT / "src/scara/calib/camera1_intrinsics.json",
    )
    parser.add_argument(
        "--slot-layout",
        type=Path,
        default=PROJECT_ROOT / "tools/tray_marker_detector_v2/tray_marker_layout.json",
    )
    parser.add_argument(
        "--silicon-config",
        type=Path,
        default=PROJECT_ROOT / "src/scara/calib/silicon_detection_0818.json",
        help="complete silicon-detection and slot-state JSON profile",
    )
    parser.add_argument("--output-json", type=Path, default=Path("layered_tray_result.json"))
    parser.add_argument("--output-image", type=Path, default=Path("layered_tray_annotated.png"))
    parser.add_argument(
        "--click",
        nargs=2,
        type=float,
        metavar=("U", "V"),
        help="also map one image pixel to Tray millimetres and nearest slot",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"could not read image: {args.image}")
    geometry = load_tray_board_geometry(args.geometry)
    intrinsics = load_camera_intrinsics(args.intrinsics)
    estimator = TrayBoardPoseEstimator(geometry, intrinsics)
    silicon_config = load_silicon_detection_config(args.silicon_config)
    analyzer = TrayVisionAnalyzer(
        estimator,
        geometry,
        load_slot_marker_layout(args.slot_layout),
        config=silicon_config.fusion_config,
    )
    result = analyzer.analyze(image)
    payload = result.to_json()
    payload["silicon_detection_config"] = {
        "path": str(silicon_config.source_path),
        "profile_name": silicon_config.profile_name,
        "sha256": silicon_config.source_sha256,
    }
    if args.click is not None and result.coordinate_mapping_allowed:
        point_T, slot_key, distance_mm = analyzer.map_pixel_to_tray(args.click, result)
        payload["click"] = {
            "pixel": [float(args.click[0]), float(args.click[1])],
            "point_T_mm": point_T.astype(float).tolist(),
            "nearest_slot": slot_key,
            "nearest_slot_distance_mm": float(distance_mm),
        }
    elif args.click is not None:
        payload["click"] = {
            "pixel": [float(args.click[0]), float(args.click[1])],
            "rejected": True,
            "reason": result.failure_reason,
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not cv2.imwrite(str(args.output_image), result.annotated_image):
        raise SystemExit(f"could not write image: {args.output_image}")
    print(
        json.dumps(
            {
                "quality_passed": result.quality_passed,
                "failure_reason": result.failure_reason,
                "summary": result.summary,
                "output_json": str(args.output_json.resolve()),
                "output_image": str(args.output_image.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0 if result.quality_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
