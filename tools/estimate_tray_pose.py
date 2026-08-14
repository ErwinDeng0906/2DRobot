"""Offline Stage-3 pose estimation for one image or a recursive image folder."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.tray_pose_estimator import (
    TrayBoardPoseEstimator,
    load_camera_intrinsics,
    load_tray_board_geometry,
)


def _camera1_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.name.lower().startswith("1_") and path.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
        }:
            yield path
        return
    for candidate in sorted(path.rglob("1_*")):
        if candidate.is_file() and candidate.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
        }:
            yield candidate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate ^C T_T from A-H in camera-1 images; never controls robot."
    )
    parser.add_argument("input", type=Path, help="one 1_XXX image or a folder")
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
        "--allow-unapproved-intrinsics",
        action="store_true",
        help="offline diagnostics only; real placement must not use rejected calibration",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--annotated-dir", type=Path, default=None)
    args = parser.parse_args()

    geometry = load_tray_board_geometry(args.geometry)
    intrinsics = load_camera_intrinsics(
        args.intrinsics,
        allow_unapproved_status=args.allow_unapproved_intrinsics,
    )
    estimator = TrayBoardPoseEstimator(geometry, intrinsics)
    rows = []
    paths = list(_camera1_images(args.input))
    if not paths:
        raise SystemExit("没有找到相机1图片（只接受文件名1_XXX）")
    if args.annotated_dir is not None:
        args.annotated_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            row = {
                "image": str(path.resolve()),
                "success": False,
                "quality_passed": False,
                "failure_reason": "无法读取图片",
            }
        else:
            estimate = estimator.estimate(image)
            row = {"image": str(path.resolve()), **estimate.to_json()}
            if args.annotated_dir is not None:
                relative_name = f"{path.parent.name}_{path.name}"
                cv2.imwrite(
                    str(args.annotated_dir / relative_name),
                    estimate.annotated_image,
                )
        rows.append(row)
        rms = row.get("reprojection_rms_px")
        print(
            f"{path}: pass={row.get('quality_passed')} "
            f"visible={row.get('visible_marker_ids', [])} "
            f"used={row.get('used_marker_ids', [])} "
            f"rms={'-' if rms is None else f'{rms:.3f}px'} "
            f"reason={row.get('failure_reason')}"
        )

    report = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input": str(args.input.resolve()),
        "geometry": str(args.geometry.resolve()),
        "intrinsics": str(args.intrinsics.resolve()),
        "allow_unapproved_intrinsics": args.allow_unapproved_intrinsics,
        "camera_filter": "filename starts with 1_; camera 2 images excluded",
        "total_images": len(rows),
        "pose_solved_images": sum(bool(row.get("success")) for row in rows),
        "quality_passed_images": sum(
            bool(row.get("quality_passed")) for row in rows
        ),
        "frames": rows,
    }
    output = args.output or (
        PROJECT_ROOT / "tray_pose_diagnostics.json"
        if len(rows) > 1
        else args.input.with_suffix(".tray_pose.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved report: {output.resolve()}")
    return 0 if report["quality_passed_images"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
