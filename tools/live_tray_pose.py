"""Camera-1 live preview of raw and temporally filtered ``^C T_T``.

This diagnostic tool opens a camera and displays results.  It has no imports
from robot-control modules and cannot move the SCARA.  Press Q or Esc to exit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from scara.vision.tray_pose_tracker import TrayPoseTracker


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live camera-1 A-H Tray pose diagnostic; no robot control."
    )
    parser.add_argument("--source", type=int, default=1)
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
        help="diagnostic only; the window is not authorization for placement",
    )
    args = parser.parse_args()

    geometry = load_tray_board_geometry(args.geometry)
    intrinsics = load_camera_intrinsics(
        args.intrinsics,
        allow_unapproved_status=args.allow_unapproved_intrinsics,
    )
    tracker = TrayPoseTracker(TrayBoardPoseEstimator(geometry, intrinsics))
    capture = cv2.VideoCapture(args.source, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, intrinsics.image_size[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, intrinsics.image_size[1])
    if not capture.isOpened():
        raise SystemExit(f"无法打开相机源 {args.source}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            tracked = tracker.update(frame)
            shown = tracked.raw.annotated_image.copy()
            if tracked.accepted_by_tracker and tracked.filtered_T_C_T is not None:
                text = (
                    "TRACKED t_C=["
                    + ", ".join(
                        f"{value:.1f}"
                        for value in tracked.filtered_T_C_T[:3, 3]
                    )
                    + "]mm"
                )
                color = (0, 160, 0)
            else:
                text = "NOT TRACKED: " + str(tracked.tracker_reason)
                color = (0, 0, 255)
            cv2.putText(
                shown,
                text,
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Camera1 A-H Tray Pose (Q/Esc exits)", shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
