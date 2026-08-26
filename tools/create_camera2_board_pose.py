#!/usr/bin/env python3
"""Create a measured ChArUco board world-pose file from three points."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.camera2_extrinsic_calibration import (  # noqa: E402
    board_pose_from_three_world_points,
)
from scara.vision.charuco_calibration import DEFAULT_BOARD_SPEC  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "由ChArUco印刷棋盘外框左上原点、+X方向参考点、+Y方向参考点的"
            "机械臂世界XYZ生成transform_W_B。三个点必须使用同一工具和方法测量。"
        )
    )
    parser.add_argument("--origin", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--x-reference", nargs=3, type=float, required=True, metavar=("X", "Y", "Z")
    )
    parser.add_argument(
        "--y-reference", nargs=3, type=float, required=True, metavar=("X", "Y", "Z")
    )
    parser.add_argument("--translation-uncertainty-mm", type=float, required=True)
    parser.add_argument("--rotation-uncertainty-deg", type=float, required=True)
    parser.add_argument("--method", required=True, help="测量工具、操作人和基准说明")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("camera2_board_pose_world.json"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.translation_uncertainty_mm <= 0.0:
        raise SystemExit("translation uncertainty必须大于0")
    if args.rotation_uncertainty_deg <= 0.0:
        raise SystemExit("rotation uncertainty必须大于0")
    transform, diagnostics = board_pose_from_three_world_points(
        args.origin,
        args.x_reference,
        args.y_reference,
    )
    if diagnostics["orthogonalization_correction_deg"] > 2.0:
        raise SystemExit(
            "三点测得的X/Y夹角偏离90度超过2度，拒绝生成标定文件："
            f"raw angle={diagnostics['raw_xy_angle_deg']:.3f} deg"
        )
    payload = {
        "schema_version": 1,
        "status": "measured",
        "measured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "measurement_method": args.method,
        "coordinate_definition": {
            "transform_W_B": "maps OpenCV ChArUco board coordinates into SCARA controller world coordinates",
            "board_origin": "top-left outer corner of printed chessboard, excluding paper margin",
            "board_x_positive": "left to right when viewing board front",
            "board_y_positive": "top to bottom when viewing board front",
            "board_z_positive": "board_x cross board_y",
        },
        "board": DEFAULT_BOARD_SPEC.to_json(),
        "measured_points_world_mm": {
            "origin": list(args.origin),
            "x_reference": list(args.x_reference),
            "y_reference": list(args.y_reference),
        },
        "transform_W_B": transform.astype(float).tolist(),
        "uncertainty": {
            "translation_mm": float(args.translation_uncertainty_mm),
            "rotation_deg": float(args.rotation_uncertainty_deg),
        },
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
