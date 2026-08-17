"""Task13 post-processing runtime for planar hand-eye calibration."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2
from PyQt6.QtCore import QObject, pyqtSignal

from scara.file_io import atomic_write_text

from .planar_handeye import fit_planar_handeye, install_planar_handeye
from .tray_pose_estimator import (
    TrayBoardPoseEstimator,
    load_camera_intrinsics,
    load_tray_board_geometry,
)
from .tray_pose_tracker import TrayPoseTracker


RESULT_FILENAME = "task13_planar_handeye.json"
_POINT_RE = re.compile(r"^TASK13\|(P[0-5][0-5])\|frame=(\d{2})/(\d{2})$")


def _latest_success(root: Path, filename: str) -> Path:
    candidates = sorted(
        (Path(root) / "Trajectory Photos").glob(f"*/{filename}"),
        key=lambda path: path.parent.name,
        reverse=True,
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if payload.get("status") in {"success", "visibility_gaps_detected"}:
                return path
        except Exception:
            continue
    raise RuntimeError(f"找不到可用{filename}")


class Task13PlanarHandEyeRuntime(QObject):
    fatal_error = pyqtSignal(str)

    def __init__(self, project_root: Path, output_dir: Path, parent=None) -> None:
        super().__init__(parent)
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)
        self.processing_failed = False
        self.failure_message = ""
        intrinsics = load_camera_intrinsics(
            self.project_root / "src/scara/calib/camera1_intrinsics.json"
        )
        geometry = load_tray_board_geometry(
            self.project_root / "src/scara/calib/tray_board_geometry.json"
        )
        self.estimator = TrayBoardPoseEstimator(geometry, intrinsics)

    def on_photo_saved(self, _path: str) -> None:
        """Task13 defers all Stage3 processing until ActionWorker is finished."""

    def _enrich_stage3(self, run_dir: Path) -> None:
        manifest_path = run_dir / "points.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        photos = {
            int(photo.get("point_sequence", -1)): str(photo.get("filename"))
            for photo in manifest.get("photos", [])
            if isinstance(photo, dict) and int(photo.get("source", -1)) == 1
        }
        annotated_dir = run_dir / "annotated_stage3"
        annotated_dir.mkdir(parents=True, exist_ok=True)
        tracker: TrayPoseTracker | None = None
        active_target: str | None = None
        for point in manifest.get("points", []):
            if not isinstance(point, dict):
                continue
            match = _POINT_RE.fullmatch(str(point.get("name") or ""))
            if match is None:
                continue
            target, frame_index, frame_total = match.group(1), int(match.group(2)), int(match.group(3))
            if frame_total != 10:
                raise RuntimeError(f"Task13 {target}声明帧数不是10")
            filename = photos.get(int(point.get("sequence", -1)))
            if not filename:
                raise RuntimeError(f"Task13 {target} frame {frame_index}缺少照片")
            if target != active_target:
                tracker = TrayPoseTracker(self.estimator)
                active_target = target
            assert tracker is not None
            image = cv2.imread(str(run_dir / filename), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"无法读取Task13照片{filename}")
            tracked = tracker.update(image)
            point["task13_target_name"] = target
            point["task13_frame_index"] = frame_index
            point["photo_filename"] = filename
            point["stage3_pose"] = tracked.raw.to_json()
            point["stage3_temporal_quality"] = {
                "accepted_by_tracker": bool(tracked.accepted_by_tracker),
                "tracker_reason": tracked.tracker_reason,
                "translation_jump_mm": tracked.translation_jump_mm,
                "rotation_jump_deg": tracked.rotation_jump_deg,
                "lost_frame_count": tracked.lost_frame_count,
                "filtered_T_C_T": None if tracked.filtered_T_C_T is None else tracked.filtered_T_C_T.astype(float).tolist(),
                "filtered_T_T_C": None if tracked.filtered_T_T_C is None else tracked.filtered_T_T_C.astype(float).tolist(),
            }
            annotated = tracked.raw.annotated_image.copy()
            if not cv2.imwrite(str(annotated_dir / filename), annotated):
                raise RuntimeError(f"无法保存Task13 Stage3标注图{filename}")
        atomic_write_text(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def on_task_finished(self, ok: bool, message: str, output_dir: str) -> None:
        if not ok:
            return
        try:
            run_dir = Path(output_dir)
            self._enrich_stage3(run_dir)
            suction_path = _latest_success(self.project_root, "camera1_suction_target.json")
            task8_run_dir = suction_path.parent
            task8_points = task8_run_dir / "points.json"
            if not task8_points.is_file():
                raise RuntimeError(f"Task8标定目录缺少points.json：{task8_points}")
            report = fit_planar_handeye(
                self.project_root,
                [
                    (f"task8_{task8_run_dir.name}", task8_points),
                    (f"task13_{run_dir.name}", run_dir / "points.json"),
                ],
                suction_target_path=suction_path,
            )
            report["task13_action_result"] = {"ok": bool(ok), "message": str(message)}
            atomic_write_text(
                run_dir / RESULT_FILENAME,
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            lines = [
                "# Task13 平面手眼标定结果",
                "",
                f"- 状态：`{report['status']}`",
                "- 适用范围：相机1、固定高度平面XY；不支持Z或完整6-DoF。",
                "- 同槽重复帧已聚合；验证按完整槽位留出。",
                "",
                "## 质量门",
                "",
            ]
            for name, gate in report["quality_gates"].items():
                lines.append(
                    f"- {'PASS' if gate['passed'] else 'FAIL'} `{name}`："
                    f"actual={gate['actual']}，limit={gate['limit']}"
                )
            atomic_write_text(
                run_dir / "task13_planar_handeye.md",
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )
            if report.get("status") == "success":
                install_planar_handeye(
                    report,
                    self.project_root
                    / "src/scara/calib/camera1_forearm_planar_handeye.json",
                )
            else:
                self.processing_failed = True
                failed = [
                    name for name, gate in report["quality_gates"].items()
                    if gate.get("passed") is not True
                ]
                self.failure_message = "Task13平面手眼独立验证未通过；未安装：" + ", ".join(failed)
                self.fatal_error.emit(self.failure_message)
        except Exception as exc:  # noqa: BLE001
            self.processing_failed = True
            self.failure_message = f"Task13平面手眼后处理失败：{exc}"
            self.fatal_error.emit(self.failure_message)


def create_task13_runtime(project_root: Path, output_dir: Path, parent=None) -> Task13PlanarHandEyeRuntime:
    return Task13PlanarHandEyeRuntime(project_root, output_dir, parent)


__all__ = ["RESULT_FILENAME", "Task13PlanarHandEyeRuntime", "create_task13_runtime"]
