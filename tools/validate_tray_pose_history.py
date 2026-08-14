"""Validate Stage-3 estimator against every historical camera-1 image.

Only filenames beginning with ``1_`` are read.  The report separates per-run
positive/negative behavior and compares duplicate route filenames between the
two 36-frame tray scans to quantify repeatability.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.tray_pose_estimator import (
    TrayBoardPoseEstimator,
    load_camera_intrinsics,
    load_tray_board_geometry,
)


def _rotation_difference_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--photos",
        type=Path,
        default=PROJECT_ROOT / "Trajectory Photos",
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        default=PROJECT_ROOT / "src/scara/calib/tray_board_geometry.json",
    )
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "tray_pose_history_validation.json",
    )
    parser.add_argument(
        "--repeat-run-a", default="260812132344"
    )
    parser.add_argument(
        "--repeat-run-b", default="260812152915"
    )
    args = parser.parse_args()

    geometry = load_tray_board_geometry(args.geometry)
    intrinsics = load_camera_intrinsics(
        args.intrinsics, allow_unapproved_status=True
    )
    estimator = TrayBoardPoseEstimator(geometry, intrinsics)
    runs: dict[str, dict] = {}
    solved: dict[str, dict[str, dict]] = {}
    total = 0
    for run_path in sorted(args.photos.iterdir()):
        if not run_path.is_dir():
            continue
        frames = []
        solved[run_path.name] = {}
        for image_path in sorted(run_path.glob("1_*.jpg")):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                row = {
                    "filename": image_path.name,
                    "success": False,
                    "quality_passed": False,
                    "failure_reason": "无法读取图片",
                }
            else:
                estimate = estimator.estimate(image)
                row = {"filename": image_path.name, **estimate.to_json()}
                if estimate.success and estimate.T_C_T is not None:
                    solved[run_path.name][image_path.name] = row
            frames.append(row)
        if not frames:
            solved.pop(run_path.name, None)
            continue
        total += len(frames)
        rms = [
            float(row["reprojection_rms_px"])
            for row in frames
            if row.get("reprojection_rms_px") is not None
        ]
        failure_reasons = collections.Counter(
            str(row.get("failure_reason"))
            for row in frames
            if not row.get("quality_passed")
        )
        runs[run_path.name] = {
            "total_camera1_images": len(frames),
            "pose_solved_images": sum(bool(row.get("success")) for row in frames),
            "quality_passed_images": sum(
                bool(row.get("quality_passed")) for row in frames
            ),
            "reprojection_rms_px": _distribution(rms),
            "failure_reasons": dict(failure_reasons),
            "frames": frames,
        }

    first = solved.get(args.repeat_run_a, {})
    second = solved.get(args.repeat_run_b, {})
    comparison_rows = []
    for filename in sorted(set(first) & set(second)):
        first_T = np.asarray(first[filename]["T_C_T"], dtype=np.float64)
        second_T = np.asarray(second[filename]["T_C_T"], dtype=np.float64)
        comparison_rows.append(
            {
                "filename": filename,
                "translation_difference_mm": float(
                    np.linalg.norm(first_T[:3, 3] - second_T[:3, 3])
                ),
                "rotation_difference_deg": _rotation_difference_deg(
                    first_T[:3, :3], second_T[:3, :3]
                ),
                "run_a_used_marker_ids": first[filename]["used_marker_ids"],
                "run_b_used_marker_ids": second[filename]["used_marker_ids"],
            }
        )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "photos_root": str(args.photos.resolve()),
        "camera_filter": "only 1_*.jpg; all 2_XXX camera-2 files excluded",
        "geometry": str(args.geometry.resolve()),
        "intrinsics": str(args.intrinsics.resolve()),
        "intrinsics_status": intrinsics.calibration_status,
        "warning": (
            "Historical validation explicitly permits unapproved intrinsics. "
            "This does not authorize robot placement."
        ),
        "total_camera1_images": total,
        "runs": runs,
        "repeatability": {
            "run_a": args.repeat_run_a,
            "run_b": args.repeat_run_b,
            "shared_solved_filenames": len(comparison_rows),
            "translation_difference_mm": _distribution(
                [row["translation_difference_mm"] for row in comparison_rows]
            ),
            "rotation_difference_deg": _distribution(
                [row["rotation_difference_deg"] for row in comparison_rows]
            ),
            "frames": comparison_rows,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved: {args.output.resolve()}")
    print(f"camera-1 images: {total}")
    print(json.dumps(report["repeatability"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
