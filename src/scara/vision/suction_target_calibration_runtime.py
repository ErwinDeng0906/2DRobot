"""Task-8 runtime: Stage-3 frame processing and Stage-4 report persistence.

The imported action owns robot movement and raw JPG acquisition.  This runtime
uses the existing Stage-3 estimator/tracker for every saved frame, performs the
batch Stage-4 solve after acquisition, enriches ``points.json``, and writes a
standalone ``camera1_suction_target.json`` plus a Markdown update checklist.
"""

from __future__ import annotations

import hashlib
import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QWidget

from scara.ui.dialogs import ask_light_warning_confirmation

from .suction_target_calibration import (
    DEFAULT_SUCTION_QUALITY,
    SuctionCalibrationQualityConfig,
    aggregate_location_poses,
    fit_suction_target,
)
from .tray_pose_estimator import (
    TrayBoardPoseEstimator,
    load_camera_intrinsics,
    load_tray_board_geometry,
)
from .tray_pose_tracker import TrayPoseTracker


EXPECTED_CAMERA1_INTRINSICS_SHA256 = (
    "797467D6E657F246FE95D8CD612489061D0FA791FD97E1F9EAD4001282A55DDF"
)
RESULT_FILENAME = "camera1_suction_target.json"
UPDATE_FILENAME = "task8_suction_calibration_update.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_text_write(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


class SuctionTargetCalibrationRuntime(QObject):
    """Process Task-8 photos and create the fixed-plane suction calibration."""

    fatal_error = pyqtSignal(str)

    def __init__(
        self,
        output_dir: Path,
        project_root: Path,
        target_points_T_mm: Mapping[str, list[float]],
        target_presets: Mapping[str, list[float]],
        parent: Optional[QWidget] = None,
        *,
        quality: SuctionCalibrationQualityConfig = DEFAULT_SUCTION_QUALITY,
    ) -> None:
        super().__init__(parent)
        self.output_dir = Path(output_dir)
        self.project_root = Path(project_root)
        self.target_points = {
            str(name): [float(value) for value in point]
            for name, point in target_points_T_mm.items()
        }
        self.target_presets = {
            str(name): [float(value) for value in joints]
            for name, joints in target_presets.items()
        }
        self.quality = quality
        self.intrinsics_path = (
            self.project_root / "src/scara/calib/camera1_intrinsics.json"
        )
        self.geometry_path = (
            self.project_root / "src/scara/calib/tray_board_geometry.json"
        )
        self.intrinsics_hash = _sha256(self.intrinsics_path)
        if self.intrinsics_hash != EXPECTED_CAMERA1_INTRINSICS_SHA256:
            raise RuntimeError(
                "camera1_intrinsics.json 与阶段4锁定版本不一致：\n"
                f"期望 {EXPECTED_CAMERA1_INTRINSICS_SHA256}\n"
                f"实际 {self.intrinsics_hash}\n"
                "如已重新标定相机，请先人工审核并更新Task8锁定哈希。"
            )
        self.geometry_hash = _sha256(self.geometry_path)
        intrinsics = load_camera_intrinsics(self.intrinsics_path)
        geometry = load_tray_board_geometry(self.geometry_path)
        if int(geometry.get("schema_version", 0)) < 2:
            raise RuntimeError("Tray几何仍是旧Z约定；Task8要求schema_version >= 2")
        slot_z = float(geometry["tray_frame"]["slot_target_plane_z_T_mm"])
        if abs(slot_z + 2.0) > 1e-9:
            raise RuntimeError(
                f"Task8要求槽目标平面 z_T=-2.0mm，当前几何为 {slot_z:.6f}mm"
            )
        self.intrinsics = intrinsics
        self.geometry = geometry
        self.estimator = TrayBoardPoseEstimator(geometry, intrinsics)
        self._tracker: Optional[TrayPoseTracker] = None
        self._active_target: Optional[str] = None
        self._records: list[dict[str, Any]] = []
        self._fatal_error_emitted = False

        confirmed = ask_light_warning_confirmation(
            parent,
            "Task8 吸盘标定安全确认",
            "开始前请确认：\n\n"
            "1. 九个PXX float预设均由当前软吸盘中心手工对准并抬到安全观察高度；\n"
            "2. 真空保持关闭；所有预设J3约为-27.01mm；\n"
            "3. 接触高度-52.01mm已经用当前工具确认，但Task8不会自动下降到该高度；\n"
            "4. 机械臂速度已调低，九点之间的安全空间无障碍，物理急停可用；\n"
            "5. 相机1为1280×720，外围A–H刚性固定。\n\n"
            "Task8只在安全观察高度依次调用九个预设并拍摄，每点20帧。是否继续？",
        )
        if not confirmed:
            raise RuntimeError("用户取消：Task8安全条件尚未确认")

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "points.json"

    def _load_manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8-sig"))

    def _photo_context(self, filename: str) -> tuple[dict[str, Any], dict[str, Any]]:
        manifest = self._load_manifest()
        photo = next(
            (
                row
                for row in manifest.get("photos", [])
                if isinstance(row, dict) and row.get("filename") == filename
            ),
            None,
        )
        if photo is None:
            raise RuntimeError(f"points.json中找不到照片 {filename}")
        point_sequence = int(photo["point_sequence"])
        point = next(
            (
                row
                for row in manifest.get("points", [])
                if int(row.get("sequence", -1)) == point_sequence
            ),
            None,
        )
        if point is None:
            raise RuntimeError(f"points.json中找不到途径点 {point_sequence}")
        return photo, point

    @staticmethod
    def _parse_point_name(name: str) -> tuple[str, int]:
        # Task8 writes: TASK8|P00|frame=01/20
        fields = str(name).split("|")
        if len(fields) != 3 or fields[0] != "TASK8" or not fields[2].startswith("frame="):
            raise RuntimeError(f"无法解析Task8途径点名称：{name}")
        target = fields[1]
        frame_text = fields[2].split("=", 1)[1].split("/", 1)[0]
        return target, int(frame_text)

    def _report_fatal(self, path: Path, exc: BaseException) -> None:
        message = f"处理照片 {path.name} 失败：{exc}"
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "task8_runtime_error.log").write_text(
                message + "\n\n" + traceback.format_exc(),
                encoding="utf-8",
            )
        except OSError:
            pass
        if not self._fatal_error_emitted:
            self._fatal_error_emitted = True
            self.fatal_error.emit(message)

    @pyqtSlot(str)
    def on_photo_saved(self, path_text: str) -> None:
        path = Path(path_text)
        if path.name.startswith("1_") is False:
            return
        try:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("无法读取刚保存的相机1照片")
            photo, point = self._photo_context(path.name)
            target_name, frame_index = self._parse_point_name(point["name"])
            if target_name not in self.target_points:
                raise RuntimeError(f"未知Task8目标 {target_name}")
            if target_name != self._active_target:
                self._active_target = target_name
                self._tracker = TrayPoseTracker(self.estimator)
            assert self._tracker is not None
            tracked = self._tracker.update(image)
            raw_json = tracked.raw.to_json()
            record = {
                "filename": path.name,
                "photo_sequence": int(photo["sequence_for_source"]),
                "point_sequence": int(photo["point_sequence"]),
                "target_name": target_name,
                "frame_index": frame_index,
                "known_point_T_mm": list(self.target_points[target_name]),
                "stage3": raw_json,
                "temporal_quality": {
                    "accepted_by_tracker": tracked.accepted_by_tracker,
                    "tracker_reason": tracked.tracker_reason,
                    "translation_jump_mm": tracked.translation_jump_mm,
                    "rotation_jump_deg": tracked.rotation_jump_deg,
                    "lost_frame_count": tracked.lost_frame_count,
                    "filtered_T_C_T": (
                        None
                        if tracked.filtered_T_C_T is None
                        else tracked.filtered_T_C_T.astype(float).tolist()
                    ),
                    "filtered_T_T_C": (
                        None
                        if tracked.filtered_T_T_C is None
                        else tracked.filtered_T_T_C.astype(float).tolist()
                    ),
                },
            }
            self._records.append(record)
            annotated_dir = self.output_dir / "annotated_stage3"
            annotated_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(annotated_dir / path.name), tracked.raw.annotated_image)
        except Exception as exc:  # protect the Qt event loop
            self._report_fatal(path, exc)

    def _enrich_manifest(
        self,
        manifest: dict[str, Any],
        locations: list[dict[str, Any]],
        fit: Mapping[str, Any],
        result_status: str,
    ) -> None:
        records_by_point = {
            int(record["point_sequence"]): record for record in self._records
        }
        for point in manifest.get("points", []):
            record = records_by_point.get(int(point.get("sequence", -1)))
            if record is None:
                continue
            point["known_point_T_mm"] = record["known_point_T_mm"]
            point["task8_target_name"] = record["target_name"]
            point["task8_frame_index"] = record["frame_index"]
            point["photo_filename"] = record["filename"]
            point["stage3_pose"] = record["stage3"]
            point["stage3_temporal_quality"] = record["temporal_quality"]
        manifest["stage4_suction_calibration"] = {
            "status": result_status,
            "definition": (
                "p_C_S is the J4/suction-axis intersection with Tray working "
                "plane z_T=-2mm; camera1 and Rz are fixed by the run contract"
            ),
            "frames_per_location": self.quality.frames_per_location,
            "location_aggregates": locations,
            "fit_summary": dict(fit),
            "result_file": RESULT_FILENAME,
            "update_file": UPDATE_FILENAME,
        }

    def _base_report(self, status: str, message: str) -> dict[str, Any]:
        imaging_j3 = float(
            np.median([joints[2] for joints in self.target_presets.values()])
        )
        rz_values = [
            joints[0] + joints[1] + joints[3] - 90.0
            for joints in self.target_presets.values()
        ]
        tray_frame = self.geometry["tray_frame"]
        return {
            "schema_version": 1,
            "status": status,
            "calibrated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "message": message,
            "camera": {
                "logical_name": "camera1_forearm_fixed",
                "source_index": 1,
                "resolution": {
                    "width": self.intrinsics.image_size[0],
                    "height": self.intrinsics.image_size[1],
                },
            },
            "locked_inputs": {
                "camera_intrinsics_path": str(self.intrinsics_path.resolve()),
                "camera_intrinsics_sha256": self.intrinsics_hash,
                "expected_camera_intrinsics_sha256": EXPECTED_CAMERA1_INTRINSICS_SHA256,
                "tray_geometry_path": str(self.geometry_path.resolve()),
                "tray_geometry_sha256": self.geometry_hash,
                "tray_geometry_schema_version": self.geometry.get("schema_version"),
            },
            "coordinate_definition": {
                "target": "J4/suction-axis intersection with fixed Tray work plane",
                "working_plane_z_T_mm": tray_frame["slot_target_plane_z_T_mm"],
                "marker_plane_z_T_mm": 0.0,
                "marker_plane_j3_mm": tray_frame["marker_plane_j3_mm"],
                "contact_j3_mm": tray_frame[
                    "slot_bottom_j3_mm_used_for_height_difference"
                ],
                "imaging_j3_mm": imaging_j3,
                "rz_mean_deg": float(np.mean(rz_values)),
                "rz_range_deg": float(np.max(rz_values) - np.min(rz_values)),
            },
            "assumptions": {
                "suction_centre_taught_visually": True,
                "soft_suction_cup_used": True,
                "j4_axis_and_suction_centre_concentric": True,
                "j4_runout_test_performed": False,
                "rz_held_constant": True,
                "vacuum_off_during_teach_and_acquisition": True,
            },
            "algorithm": {
                "per_frame_pose": "Stage-3 TrayBoardPoseEstimator; no duplicated PnP implementation",
                "temporal_gate": "Stage-3 TrayPoseTracker reset at each slot",
                "per_location_filter": (
                    "translation component median + MAD rejection; SO(3) geodesic mean "
                    "+ angular MAD rejection"
                ),
                "suction_fit": "robust location rejection followed by least-squares 3D mean",
                "independent_validation": "leave-one-location-out cross-validation",
            },
            "quality_configuration": {
                key: value
                for key, value in self.quality.__dict__.items()
            },
            "targets": [
                {
                    "name": name,
                    "point_T_mm": self.target_points[name],
                    "preset_joints": self.target_presets[name],
                }
                for name in self.target_points
            ],
            "source_run_folder": str(self.output_dir.resolve()),
        }

    def _write_markdown(
        self,
        report: Mapping[str, Any],
        locations: list[Mapping[str, Any]],
    ) -> None:
        fit = report.get("fit", {})
        success_locations = sum(bool(row.get("success")) for row in locations)
        status = str(report.get("status"))
        lines = [
            "# Task8 suction calibration update",
            "",
            f"- [{'x' if status == 'success' else ' '}] Overall status: `{status}`",
            f"- [x] Camera 1 frames processed with the existing Stage-3 estimator: {len(self._records)}",
            f"- [{'x' if success_locations == len(self.target_points) else ' '}] Stable locations: {success_locations}/{len(self.target_points)}",
            "- [x] A-H marker count, RANSAC, reprojection RMS, positive depth and temporal jump gates recorded in `points.json`",
            "- [x] Per-location translation median and SO(3) geodesic filtering completed",
            "- [x] Location-wise leave-one-out independent validation completed",
            "- [x] J4 runout intentionally skipped under the declared concentricity assumption",
            "",
            "## Locked inputs",
            "",
            f"- Intrinsics SHA-256: `{self.intrinsics_hash}`",
            f"- Tray geometry SHA-256: `{self.geometry_hash}`",
            "- Tray working plane: `z_T = -2.0 mm`",
            "",
            "## Result",
            "",
            f"- `p_C_S_mm`: `{fit.get('p_C_S_mm')}`",
            f"- distorted target pixel: `{fit.get('target_pixel_distorted_px')}`",
            f"- fit XY RMS: `{fit.get('fit_xy_rms_mm')} mm`",
            f"- leave-one-location-out XY RMS: `{(fit.get('cross_validation') or {}).get('xy_rms_mm')} mm`",
            "",
            "## Files",
            "",
            "- `points.json`: robot state, known Tray point, every Stage-3 result and temporal quality, plus stable location transforms",
            f"- `{RESULT_FILENAME}`: final suction target, residuals, gates and input hashes",
            "- `annotated_stage3/`: Stage-3 annotated copy of every raw camera image",
            f"- `{UPDATE_FILENAME}`: this checklist",
        ]
        _atomic_text_write(self.output_dir / UPDATE_FILENAME, "\n".join(lines))

    @pyqtSlot(bool, str, str)
    def on_task_finished(self, ok: bool, message: str, output_dir_text: str) -> None:
        output_dir = Path(output_dir_text)
        if output_dir.resolve() != self.output_dir.resolve():
            raise RuntimeError("Task8运行时输出文件夹不一致")
        manifest = self._load_manifest()
        locations: list[dict[str, Any]] = []
        for target_name, point_T in self.target_points.items():
            rows = [
                record
                for record in self._records
                if record["target_name"] == target_name
            ]
            accepted = [
                row
                for row in rows
                if row["stage3"]["quality_passed"]
                and row["temporal_quality"]["accepted_by_tracker"]
                and row["stage3"]["T_C_T"] is not None
            ]
            locations.append(
                aggregate_location_poses(
                    target_name,
                    point_T,
                    [np.asarray(row["stage3"]["T_C_T"]) for row in accepted],
                    [int(row["frame_index"]) for row in accepted],
                    self.quality,
                )
            )

        if not ok:
            fit: dict[str, Any] = {
                "status": "acquisition_stopped",
                "failure_reasons": [message],
                "p_C_S_mm": None,
                "target_pixel_distorted_px": None,
            }
            report = self._base_report("acquisition_stopped", message)
        else:
            expected = self.quality.frames_per_location * len(self.target_points)
            if len(self._records) != expected:
                fit = {
                    "status": "rejected_quality",
                    "failure_reasons": [
                        f"processed {len(self._records)}/{expected} expected photos"
                    ],
                    "p_C_S_mm": None,
                    "target_pixel_distorted_px": None,
                }
            else:
                fit = fit_suction_target(
                    locations,
                    self.intrinsics.K,
                    self.intrinsics.dist_coeffs,
                    self.quality,
                )
            report = self._base_report(str(fit["status"]), message)
        report["location_aggregates"] = locations
        report["fit"] = fit
        report["status"] = str(fit["status"])
        self._enrich_manifest(manifest, locations, fit, report["status"])
        _atomic_json_write(self.manifest_path, manifest)
        _atomic_json_write(output_dir / RESULT_FILENAME, report)
        self._write_markdown(report, locations)
        if ok and report["status"] != "success":
            reasons = ", ".join(str(value) for value in fit.get("failure_reasons", []))
            raise RuntimeError(
                "Task8采集完成但吸盘标定质量门未通过；"
                f"原因：{reasons or '请检查结果JSON'}"
            )


def create_camera1_suction_runtime(
    output_dir: Path,
    project_root: Path,
    target_points_T_mm: Mapping[str, list[float]],
    target_presets: Mapping[str, list[float]],
    parent: Optional[QWidget] = None,
) -> SuctionTargetCalibrationRuntime:
    return SuctionTargetCalibrationRuntime(
        output_dir,
        project_root,
        target_points_T_mm,
        target_presets,
        parent,
    )


__all__ = [
    "EXPECTED_CAMERA1_INTRINSICS_SHA256",
    "RESULT_FILENAME",
    "SuctionTargetCalibrationRuntime",
    "UPDATE_FILENAME",
    "create_camera1_suction_runtime",
]
