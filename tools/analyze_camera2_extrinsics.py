#!/usr/bin/env python3
"""Offline fail-closed camera-2/J4 extrinsic analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.camera2_extrinsic_calibration import (  # noqa: E402
    Camera2CharucoPoseDetector,
    collect_run_directories,
    load_board_pose_world,
    load_camera2_intrinsics,
    observations_from_run,
    solve_known_board_extrinsic,
)


DEFAULT_INTRINSICS = ROOT / "src/scara/calib/camera2_intrinsics.json"
DEFAULT_PROJECT_OUTPUT = ROOT / "src/scara/calib/camera2_j4_extrinsics.json"


def _atomic_json_write(path: Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "读取一个或多个Task18运行目录，检测相机2 ChArUco位姿，并使用"
            "独立测量的标定板世界位姿求J4<-Camera2完整外参。"
        )
    )
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Task18时间戳目录，或包含多个时间戳目录的父文件夹",
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=DEFAULT_INTRINSICS,
        help="相机2内参JSON",
    )
    parser.add_argument(
        "--board-pose",
        type=Path,
        required=True,
        help="独立测量的ChArUco板世界位姿JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("camera2_j4_extrinsic_report.json"),
        help="离线报告输出路径",
    )
    parser.add_argument(
        "--annotated-dir",
        type=Path,
        default=Path("camera2_j4_extrinsic_annotated"),
        help="标注图输出文件夹",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="仅在全部质量门PASS时更新项目camera2_j4_extrinsics.json",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    intrinsics = load_camera2_intrinsics(args.intrinsics)
    board_pose = load_board_pose_world(args.board_pose)
    detector = Camera2CharucoPoseDetector(intrinsics)
    run_directories = collect_run_directories(args.runs)
    if not run_directories:
        raise SystemExit("没有找到包含points.json的Task18运行目录")

    observations = []
    rejected = []
    for run_directory in run_directories:
        accepted_rows, rejected_rows = observations_from_run(
            run_directory,
            detector,
            annotated_dir=args.annotated_dir,
        )
        observations.extend(accepted_rows)
        rejected.extend(rejected_rows)

    result = solve_known_board_extrinsic(observations, board_pose)
    result.update(
        {
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source_runs": [str(path) for path in run_directories],
            "intrinsics": {
                "path": intrinsics["path"],
                "sha256": intrinsics["sha256"],
                "resolution": list(intrinsics["resolution"]),
                "global_rms_px": intrinsics["global_rms_px"],
            },
            "board_pose": {
                "path": board_pose["path"],
                "sha256": board_pose["sha256"],
                "measurement_method": board_pose["measurement_method"],
                "translation_uncertainty_mm": board_pose[
                    "translation_uncertainty_mm"
                ],
                "rotation_uncertainty_deg": board_pose[
                    "rotation_uncertainty_deg"
                ],
            },
            "accepted_observations": [row.to_json() for row in observations],
            "rejected_observations": rejected,
        }
    )
    _atomic_json_write(args.output, result)
    print(
        f"status={result['status']} accepted={len(observations)} "
        f"rejected={len(rejected)} report={args.output.resolve()}"
    )
    if args.install:
        if result["status"] != "success" or result["installation_allowed"] is not True:
            print("质量门未通过，拒绝安装项目外参。", file=sys.stderr)
            return 2
        _atomic_json_write(DEFAULT_PROJECT_OUTPUT, result)
        print(f"installed={DEFAULT_PROJECT_OUTPUT}")
    return 0 if result["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
