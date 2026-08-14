"""Generate Stage-2 tray_board_geometry.json from taught SCARA presets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.tray_board_geometry import generate_geometry_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build strict Tray Frame, 6x6 slots, and A-H board geometry."
    )
    parser.add_argument(
        "--presets",
        type=Path,
        default=PROJECT_ROOT / "scara_presets.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "src/scara/calib/tray_board_geometry.json",
    )
    args = parser.parse_args()
    payload = generate_geometry_file(args.presets, args.output)
    print(f"saved: {args.output.resolve()}")
    print(f"valid: {payload['validation']['valid']}")
    print(
        "raw P50->P00 / P05->P00 angle: "
        f"{payload['diagnostics']['tray_frame_from_taught_slots']['raw_axis_angle_deg']:.6f} deg"
    )
    for label, marker in payload["markers"].items():
        fit = payload["diagnostics"]["marker_corner_fit"][label]
        print(
            f"{label}=ID{marker['id']}: z_T={marker['center_T_mm'][2]:.4f} mm, "
            f"UL/DL fit RMS={fit['left_corner_fit_rms_mm']:.4f} mm"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
