"""Task12 Stage-3 observability scan and report persistence.

The action script owns deterministic high-plane motion and camera capture.
This module never imports a controller and never issues motion.  For every
camera-1 image it reuses :class:`TrayBoardPoseEstimator` and
:class:`TrayPoseTracker`, then summarizes whether each P00-P55 location can
reliably provide a fresh ``^C T_T`` for later visual servoing.

Slot readiness is deliberately stricter than "one frame solved": all expected
frames must exist and at least 80 percent must pass both the independent
Stage-3 gates and temporal tracking.  Diagnostic results are not calibration
files and are never installed under ``src/scara/calib``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSlot
from PyQt6.QtWidgets import QWidget

from scara.file_io import atomic_write_text, read_text_snapshot
from scara.ui.dialogs import ask_light_warning_confirmation

from .tray_pose_estimator import (
    DEFAULT_POSE_QUALITY,
    TrayBoardPoseEstimator,
    load_camera_intrinsics,
    load_tray_board_geometry,
)
from .tray_pose_tracker import TrayPoseTracker


RESULT_FILENAME = "task12_stage3_visibility_scan.json"
UPDATE_FILENAME = "task12_stage3_visibility_scan.md"
ERROR_FILENAME = "task12_stage3_visibility_scan_error.log"
ANNOTATED_DIRECTORY = "annotated_stage3"
MINIMUM_READY_PASS_RATE = 0.80
MINIMUM_MARGINAL_PASS_RATE = 0.50

_POINT_NAME_RE = re.compile(r"^TASK12\|(P[0-5][0-5])\|frame=(\d{2})/(\d{2})$")
_PHOTO_NAME_RE = re.compile(r"^1_(\d+)\.jpg$", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _finite_or_none(value: object) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _numeric_summary(values: Sequence[float]) -> dict[str, Optional[float]]:
    clean = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if clean.size == 0:
        return {"min": None, "median": None, "p95": None, "max": None}
    return {
        "min": float(np.min(clean)),
        "median": float(np.median(clean)),
        "p95": float(np.percentile(clean, 95.0)),
        "max": float(np.max(clean)),
    }


def parse_task12_point_name(name: str) -> tuple[str, int, int]:
    match = _POINT_NAME_RE.fullmatch(str(name))
    if match is None:
        raise ValueError(f"无法解析Task12途径点名称：{name}")
    target_name = match.group(1)
    frame_index = int(match.group(2))
    frame_total = int(match.group(3))
    if frame_index < 1 or frame_index > frame_total:
        raise ValueError(f"Task12帧序号超出范围：{name}")
    return target_name, frame_index, frame_total


def summarize_slot_visibility(
    target_name: str,
    point_T_mm: Sequence[float],
    records: Sequence[Mapping[str, Any]],
    *,
    expected_frames: int,
    minimum_ready_pass_rate: float = MINIMUM_READY_PASS_RATE,
    minimum_marginal_pass_rate: float = MINIMUM_MARGINAL_PASS_RATE,
) -> dict[str, Any]:
    """Aggregate one slot without hiding failed or unreadable frames."""

    rows = sorted(records, key=lambda row: int(row.get("frame_index", 0)))
    stage3_rows = [row for row in rows if isinstance(row.get("stage3"), Mapping)]
    success_rows = [row for row in stage3_rows if bool(row["stage3"].get("success"))]
    quality_rows = [
        row for row in stage3_rows if bool(row["stage3"].get("quality_passed"))
    ]
    temporal_rows = [
        row
        for row in stage3_rows
        if bool((row.get("temporal_quality") or {}).get("accepted_by_tracker"))
    ]
    combined_rows = [
        row
        for row in stage3_rows
        if bool(row["stage3"].get("quality_passed"))
        and bool((row.get("temporal_quality") or {}).get("accepted_by_tracker"))
        and row["stage3"].get("T_C_T") is not None
    ]

    captured_count = len(rows)
    combined_count = len(combined_rows)
    pass_rate_expected = combined_count / expected_frames if expected_frames else 0.0
    complete = captured_count == expected_frames
    ready = complete and pass_rate_expected >= float(minimum_ready_pass_rate)
    if ready:
        classification = "ready"
    elif pass_rate_expected >= float(minimum_marginal_pass_rate):
        classification = "marginal"
    else:
        classification = "not_observable"

    visible_counts = [len(row["stage3"].get("visible_marker_ids") or []) for row in stage3_rows]
    used_counts = [len(row["stage3"].get("used_marker_ids") or []) for row in stage3_rows]
    inlier_corner_counts = [
        int(row["stage3"].get("ransac_inlier_corner_count") or 0)
        for row in stage3_rows
    ]
    rms_all = [
        value
        for row in success_rows
        if (value := _finite_or_none(row["stage3"].get("reprojection_rms_px"))) is not None
    ]
    rms_combined = [
        value
        for row in combined_rows
        if (value := _finite_or_none(row["stage3"].get("reprojection_rms_px"))) is not None
    ]

    failure_reasons: Counter[str] = Counter()
    for row in rows:
        if row.get("processing_error"):
            failure_reasons[f"processing: {row['processing_error']}"] += 1
            continue
        stage3 = row.get("stage3") or {}
        if not bool(stage3.get("quality_passed")):
            failure_reasons[str(stage3.get("failure_reason") or "Stage3 quality rejected")] += 1
            continue
        temporal = row.get("temporal_quality") or {}
        if not bool(temporal.get("accepted_by_tracker")):
            failure_reasons[
                "temporal: " + str(temporal.get("tracker_reason") or "tracker rejected")
            ] += 1

    marker_visibility: Counter[int] = Counter()
    rejected_markers: Counter[int] = Counter()
    per_marker_worst: dict[int, float] = {}
    for row in stage3_rows:
        stage3 = row["stage3"]
        marker_visibility.update(int(value) for value in stage3.get("visible_marker_ids") or [])
        rejected_markers.update(int(value) for value in stage3.get("rejected_marker_ids") or [])
        for marker_text, error in (stage3.get("per_marker_rms_px") or {}).items():
            marker_id = int(marker_text)
            numeric = float(error)
            per_marker_worst[marker_id] = max(per_marker_worst.get(marker_id, 0.0), numeric)

    missing_frame_indices = sorted(
        set(range(1, expected_frames + 1))
        - {int(row.get("frame_index", -1)) for row in rows}
    )
    return {
        "target_name": str(target_name),
        "point_T_mm": [float(value) for value in point_T_mm],
        "classification": classification,
        "ready_for_stage7": ready,
        "captured_frame_count": captured_count,
        "expected_frame_count": int(expected_frames),
        "missing_frame_indices": missing_frame_indices,
        "stage3_solver_success_count": len(success_rows),
        "independent_quality_pass_count": len(quality_rows),
        "temporal_accept_count": len(temporal_rows),
        "combined_pass_count": combined_count,
        "combined_pass_rate_of_expected": float(pass_rate_expected),
        "quality_gates": {
            "frame_set_complete": {
                "passed": complete,
                "actual": captured_count,
                "required": int(expected_frames),
            },
            "combined_stage3_temporal_pass_rate": {
                "passed": pass_rate_expected >= float(minimum_ready_pass_rate),
                "actual": float(pass_rate_expected),
                "required_minimum": float(minimum_ready_pass_rate),
            },
        },
        "visible_marker_count": _numeric_summary(visible_counts),
        "used_marker_count": _numeric_summary(used_counts),
        "ransac_inlier_corner_count": _numeric_summary(inlier_corner_counts),
        "reprojection_rms_px_all_solved": _numeric_summary(rms_all),
        "reprojection_rms_px_combined_pass": _numeric_summary(rms_combined),
        "marker_visibility_frame_counts": {
            str(marker_id): int(marker_visibility.get(marker_id, 0))
            for marker_id in range(1, 9)
        },
        "rejected_marker_frame_counts": {
            str(marker_id): int(rejected_markers.get(marker_id, 0))
            for marker_id in range(1, 9)
        },
        "worst_per_marker_rms_px": {
            str(marker_id): float(value)
            for marker_id, value in sorted(per_marker_worst.items())
        },
        "failure_reason_counts": dict(failure_reasons),
        "filenames": [str(row.get("filename")) for row in rows],
    }


class Stage3VisibilityScanRuntime(QObject):
    """Process Task12 images and persist a diagnostic-only 36-slot report."""

    def __init__(
        self,
        output_dir: Path,
        project_root: Path,
        target_order: Sequence[str],
        slot_points_T_mm: Mapping[str, Sequence[float]],
        frames_per_slot: int,
        parent: Optional[QWidget] = None,
        *,
        confirm_safety: bool = True,
    ) -> None:
        super().__init__(parent)
        self.output_dir = Path(output_dir)
        self.project_root = Path(project_root)
        self.target_order = tuple(str(value) for value in target_order)
        self.slot_points = {
            str(name): [float(value) for value in point]
            for name, point in slot_points_T_mm.items()
        }
        self.frames_per_slot = int(frames_per_slot)
        if self.frames_per_slot < 1:
            raise ValueError("Task12 frames_per_slot必须为正整数")
        if len(self.target_order) != 36 or len(set(self.target_order)) != 36:
            raise ValueError("Task12 target_order必须恰好包含36个不重复槽")
        if set(self.target_order) != set(self.slot_points):
            raise ValueError("Task12目标顺序与Tray槽中心集合不一致")

        self.intrinsics_path = self.project_root / "src/scara/calib/camera1_intrinsics.json"
        self.geometry_path = self.project_root / "src/scara/calib/tray_board_geometry.json"
        self.intrinsics_hash = _sha256(self.intrinsics_path)
        self.geometry_hash = _sha256(self.geometry_path)
        self.intrinsics = load_camera_intrinsics(self.intrinsics_path)
        self.geometry = load_tray_board_geometry(self.geometry_path)
        self.estimator = TrayBoardPoseEstimator(self.geometry, self.intrinsics)
        self._tracker: Optional[TrayPoseTracker] = None
        self._active_target: Optional[str] = None
        self._records_by_filename: dict[str, dict[str, Any]] = {}
        self._last_live_sequence = 0

        if confirm_safety:
            confirmed = ask_light_warning_confirmation(
                parent,
                "Task12 Stage3可观测性扫描安全确认",
                "开始前请确认：\n\n"
                "1. 机械臂已经到达已示教的 P00 float 固定观察高度；\n"
                "2. 从P00到P55的整个托盘上方路径无障碍，物理急停可用；\n"
                "3. 真空关闭，吸盘不携带硅片，相机1为1280×720；\n"
                "4. A-H外围Marker与托盘刚性固定，拍摄期间托盘不会移动。\n\n"
                "Task12将保持同一J3和绝对Rz，通过相邻槽小步路线扫描36槽，"
                "每槽拍20帧并返回P00。不会下降Z，不会触发DO/真空，也不会执行视觉修正。"
                "是否继续？",
            )
            if not confirmed:
                raise RuntimeError("用户取消：Task12安全条件尚未确认")

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "points.json"

    def _expected_from_sequence(self, sequence: int) -> tuple[str, int]:
        total = len(self.target_order) * self.frames_per_slot
        if sequence < 1 or sequence > total:
            raise ValueError(f"Task12相机1照片序号{sequence}超出1-{total}")
        zero_based = sequence - 1
        return (
            self.target_order[zero_based // self.frames_per_slot],
            zero_based % self.frames_per_slot + 1,
        )

    def _process_one(
        self,
        path: Path,
        target_name: str,
        frame_index: int,
        point_sequence: int,
        tracker: TrayPoseTracker,
    ) -> dict[str, Any]:
        base: dict[str, Any] = {
            "filename": path.name,
            "point_sequence": int(point_sequence),
            "target_name": str(target_name),
            "frame_index": int(frame_index),
            "known_point_T_mm": list(self.slot_points[target_name]),
            "stage3": None,
            "temporal_quality": {
                "accepted_by_tracker": False,
                "tracker_reason": "frame processing did not complete",
                "translation_jump_mm": None,
                "rotation_jump_deg": None,
                "lost_frame_count": None,
                "filtered_T_C_T": None,
                "filtered_T_T_C": None,
            },
            "processing_error": None,
        }
        try:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("无法读取已保存的相机1照片")
            tracked = tracker.update(image)
            raw_json = tracked.raw.to_json()
            base["stage3"] = raw_json
            base["temporal_quality"] = {
                "accepted_by_tracker": bool(tracked.accepted_by_tracker),
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
            }
            annotated = tracked.raw.annotated_image.copy()
            color = (0, 170, 0) if (
                raw_json.get("quality_passed") and tracked.accepted_by_tracker
            ) else (0, 0, 255)
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 42), (255, 255, 255), -1)
            cv2.putText(
                annotated,
                f"TASK12 {target_name} frame {frame_index:02d}/{self.frames_per_slot:02d} "
                f"{'PASS' if color == (0, 170, 0) else 'REJECT'}",
                (12, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                color,
                2,
                cv2.LINE_AA,
            )
            annotated_dir = self.output_dir / ANNOTATED_DIRECTORY
            annotated_dir.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(annotated_dir / path.name), annotated):
                raise RuntimeError("保存Stage3标注图失败")
        except Exception as exc:
            base["processing_error"] = str(exc)
            base["temporal_quality"]["tracker_reason"] = f"processing error: {exc}"
        return base

    @pyqtSlot(str)
    def on_photo_saved(self, path_text: str) -> None:
        path = Path(path_text)
        match = _PHOTO_NAME_RE.fullmatch(path.name)
        if match is None:
            return
        sequence = int(match.group(1))
        try:
            target_name, frame_index = self._expected_from_sequence(sequence)
            if target_name != self._active_target or sequence != self._last_live_sequence + 1:
                self._tracker = TrayPoseTracker(self.estimator)
                self._active_target = target_name
            assert self._tracker is not None
            record = self._process_one(
                path,
                target_name,
                frame_index,
                sequence,
                self._tracker,
            )
            self._records_by_filename[path.name] = record
            self._last_live_sequence = sequence
        except Exception as exc:  # keep diagnostic acquisition running
            self._records_by_filename[path.name] = {
                "filename": path.name,
                "point_sequence": sequence,
                "target_name": None,
                "frame_index": None,
                "known_point_T_mm": None,
                "stage3": None,
                "temporal_quality": {"accepted_by_tracker": False},
                "processing_error": str(exc),
            }

    def _manifest_contexts(self, manifest: Mapping[str, Any]) -> list[tuple[Path, str, int, int]]:
        points = {
            int(row.get("sequence", -1)): row
            for row in manifest.get("points", [])
            if isinstance(row, Mapping)
        }
        contexts: list[tuple[Path, str, int, int]] = []
        for photo in manifest.get("photos", []):
            if not isinstance(photo, Mapping) or int(photo.get("source", -1)) != 1:
                continue
            point_sequence = int(photo.get("point_sequence", -1))
            point = points.get(point_sequence)
            if point is None:
                raise RuntimeError(f"points.json缺少照片对应途径点 {point_sequence}")
            target_name, frame_index, frame_total = parse_task12_point_name(str(point.get("name")))
            if frame_total != self.frames_per_slot:
                raise RuntimeError(
                    f"{target_name}声明每槽{frame_total}帧，与Task12运行时{self.frames_per_slot}不一致"
                )
            filename = str(photo.get("filename"))
            contexts.append(
                (self.output_dir / filename, target_name, frame_index, point_sequence)
            )
        return sorted(contexts, key=lambda row: row[3])

    def _reprocess_incomplete_targets(
        self,
        contexts: Sequence[tuple[Path, str, int, int]],
    ) -> None:
        by_target: dict[str, list[tuple[Path, str, int, int]]] = {
            target: [] for target in self.target_order
        }
        for context in contexts:
            by_target[context[1]].append(context)
        for target_name in self.target_order:
            authoritative = sorted(by_target[target_name], key=lambda row: row[2])
            existing = [
                row
                for row in self._records_by_filename.values()
                if row.get("target_name") == target_name
            ]
            expected_filenames = {row[0].name for row in authoritative}
            existing_filenames = {str(row.get("filename")) for row in existing}
            if len(authoritative) == self.frames_per_slot and existing_filenames == expected_filenames:
                continue
            for filename in existing_filenames:
                self._records_by_filename.pop(filename, None)
            tracker = TrayPoseTracker(self.estimator)
            for path, target, frame_index, point_sequence in authoritative:
                self._records_by_filename[path.name] = self._process_one(
                    path,
                    target,
                    frame_index,
                    point_sequence,
                    tracker,
                )

    def _enrich_manifest(
        self,
        manifest: dict[str, Any],
        summaries: Sequence[Mapping[str, Any]],
        status: str,
    ) -> None:
        records_by_point = {
            int(record["point_sequence"]): record
            for record in self._records_by_filename.values()
            if record.get("point_sequence") is not None
        }
        for point in manifest.get("points", []):
            record = records_by_point.get(int(point.get("sequence", -1)))
            if record is None:
                continue
            point["task12_target_name"] = record.get("target_name")
            point["task12_frame_index"] = record.get("frame_index")
            point["known_point_T_mm"] = record.get("known_point_T_mm")
            point["photo_filename"] = record.get("filename")
            point["stage3_pose"] = record.get("stage3")
            point["stage3_temporal_quality"] = record.get("temporal_quality")
            point["task12_processing_error"] = record.get("processing_error")
        manifest["stage3_visibility_scan"] = {
            "status": status,
            "result_file": RESULT_FILENAME,
            "update_file": UPDATE_FILENAME,
            "minimum_ready_pass_rate": MINIMUM_READY_PASS_RATE,
            "ready_slots": [
                row["target_name"] for row in summaries if row["ready_for_stage7"]
            ],
            "not_ready_slots": [
                row["target_name"] for row in summaries if not row["ready_for_stage7"]
            ],
        }

    def _build_report(
        self,
        ok: bool,
        message: str,
        summaries: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        ready = [str(row["target_name"]) for row in summaries if row["ready_for_stage7"]]
        marginal = [
            str(row["target_name"])
            for row in summaries
            if row["classification"] == "marginal"
        ]
        unavailable = [
            str(row["target_name"])
            for row in summaries
            if row["classification"] == "not_observable"
        ]
        if not ok:
            status = "acquisition_stopped"
        elif len(ready) == len(self.target_order):
            status = "success"
        else:
            status = "visibility_gaps_detected"
        records = sorted(
            self._records_by_filename.values(),
            key=lambda row: int(row.get("point_sequence") or 0),
        )
        return {
            "schema_version": 1,
            "status": status,
            "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "message": str(message),
            "purpose": (
                "Diagnostic-only Stage-3 observability scan. This file is not a "
                "Jacobian calibration and is never installed as a motion model."
            ),
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
                "tray_geometry_path": str(self.geometry_path.resolve()),
                "tray_geometry_sha256": self.geometry_hash,
                "tray_geometry_schema_version": self.geometry.get("schema_version"),
            },
            "scan_configuration": {
                "target_order": list(self.target_order),
                "slot_count": len(self.target_order),
                "frames_per_slot": self.frames_per_slot,
                "expected_total_frames": len(self.target_order) * self.frames_per_slot,
                "minimum_ready_pass_rate": MINIMUM_READY_PASS_RATE,
                "minimum_marginal_pass_rate": MINIMUM_MARGINAL_PASS_RATE,
                "combined_pass_definition": (
                    "Stage3 quality_passed AND temporal accepted_by_tracker AND T_C_T present"
                ),
                "stage3_quality_configuration": dict(DEFAULT_POSE_QUALITY.__dict__),
            },
            "summary": {
                "ready_slot_count": len(ready),
                "marginal_slot_count": len(marginal),
                "not_observable_slot_count": len(unavailable),
                "ready_slots": ready,
                "marginal_slots": marginal,
                "not_observable_slots": unavailable,
                "all_36_slots_ready": len(ready) == len(self.target_order),
                "processed_frame_count": len(records),
                "processing_error_count": sum(
                    bool(row.get("processing_error")) for row in records
                ),
            },
            "slots": list(summaries),
            "frame_records": records,
            "source_run_folder": str(self.output_dir.resolve()),
        }

    def _write_markdown(self, report: Mapping[str, Any]) -> None:
        summary = report["summary"]
        lines = [
            "# Task12 Stage3 visibility scan",
            "",
            f"- Overall diagnostic status: `{report['status']}`",
            f"- Ready slots: **{summary['ready_slot_count']}/36**",
            f"- Marginal slots: **{summary['marginal_slot_count']}**",
            f"- Not observable slots: **{summary['not_observable_slot_count']}**",
            f"- Processed frames: **{summary['processed_frame_count']}/720**",
            "- Readiness rule: all 20 frames captured and at least 80% pass both Stage3 and temporal gates.",
            "- This is a diagnostic report only; it does not install or approve a Jacobian.",
            "",
            "## Per-slot result",
            "",
            "| Slot | Class | Combined pass | Rate | Visible markers median | RMS median px | Main failures |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in report["slots"]:
            failures = row.get("failure_reason_counts") or {}
            main_failures = "; ".join(
                f"{reason} ({count})"
                for reason, count in sorted(
                    failures.items(), key=lambda item: (-int(item[1]), str(item[0]))
                )[:2]
            ) or "-"
            visible_median = row["visible_marker_count"]["median"]
            rms_median = row["reprojection_rms_px_combined_pass"]["median"]
            lines.append(
                f"| {row['target_name']} | {row['classification']} | "
                f"{row['combined_pass_count']}/{row['expected_frame_count']} | "
                f"{100.0 * row['combined_pass_rate_of_expected']:.1f}% | "
                f"{visible_median if visible_median is not None else '-'} | "
                f"{f'{rms_median:.3f}' if rms_median is not None else '-'} | "
                f"{main_failures.replace('|', '/')} |"
            )
        lines.extend(
            [
                "",
                "## Files",
                "",
                f"- `{RESULT_FILENAME}`: full summary and every frame result",
                "- `points.json`: robot state enriched with Stage3 and temporal quality",
                f"- `{ANNOTATED_DIRECTORY}/`: one annotated copy per processed image",
                f"- `{UPDATE_FILENAME}`: this readable table",
            ]
        )
        atomic_write_text(self.output_dir / UPDATE_FILENAME, "\n".join(lines) + "\n")

    @pyqtSlot(bool, str, str)
    def on_task_finished(self, ok: bool, message: str, output_dir_text: str) -> None:
        try:
            output_dir = Path(output_dir_text)
            if output_dir.resolve() != self.output_dir.resolve():
                raise RuntimeError("Task12运行时输出文件夹不一致")
            manifest = json.loads(read_text_snapshot(self.manifest_path, encoding="utf-8-sig"))
            contexts = self._manifest_contexts(manifest)
            self._reprocess_incomplete_targets(contexts)
            summaries = [
                summarize_slot_visibility(
                    target,
                    self.slot_points[target],
                    [
                        row
                        for row in self._records_by_filename.values()
                        if row.get("target_name") == target
                    ],
                    expected_frames=self.frames_per_slot,
                )
                for target in sorted(self.slot_points)
            ]
            report = self._build_report(ok, message, summaries)
            self._enrich_manifest(manifest, summaries, str(report["status"]))
            atomic_write_text(self.manifest_path, _json_text(manifest))
            atomic_write_text(self.output_dir / RESULT_FILENAME, _json_text(report))
            self._write_markdown(report)
        except Exception as exc:
            try:
                atomic_write_text(
                    self.output_dir / ERROR_FILENAME,
                    f"{exc}\n\n{traceback.format_exc()}",
                )
            except OSError:
                pass
            raise


def create_stage3_visibility_scan_runtime(
    output_dir: Path,
    project_root: Path,
    target_order: Sequence[str],
    slot_points_T_mm: Mapping[str, Sequence[float]],
    frames_per_slot: int,
    parent: Optional[QWidget] = None,
) -> Stage3VisibilityScanRuntime:
    return Stage3VisibilityScanRuntime(
        output_dir=output_dir,
        project_root=project_root,
        target_order=target_order,
        slot_points_T_mm=slot_points_T_mm,
        frames_per_slot=frames_per_slot,
        parent=parent,
    )


__all__ = [
    "ANNOTATED_DIRECTORY",
    "MINIMUM_READY_PASS_RATE",
    "RESULT_FILENAME",
    "Stage3VisibilityScanRuntime",
    "UPDATE_FILENAME",
    "create_stage3_visibility_scan_runtime",
    "parse_task12_point_name",
    "summarize_slot_visibility",
]
