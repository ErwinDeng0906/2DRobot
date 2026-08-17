"""Task11 runtime for the P22 20 x 20 mm wide image-error model.

The runtime reuses Task9's proven Stage3/photo processing path, adds explicit
train/pass/validation metadata, performs the wide fit only after all 330
expected frames have been processed, and installs a model only when every
independent validation gate passes.  It has no controller API.
"""

from __future__ import annotations

import json
import math
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QWidget

from .wide_xy_jacobian import (
    DEFAULT_WIDE_XY_JACOBIAN_QUALITY,
    REQUIRED_WIDE_XY_JACOBIAN_QUALITY_GATES,
    fit_wide_xy_image_model,
)
from .xy_image_jacobian import REQUIRED_XY_JACOBIAN_QUALITY_GATES
from .xy_image_jacobian import DEFAULT_XY_IMAGE_JACOBIAN_QUALITY
from .xy_image_jacobian_runtime import (
    XYImageJacobianCalibrationRuntime,
    _atomic_json_write,
    _atomic_text_write,
    _load_json,
    _sha256,
)


RESULT_FILENAME = "camera1_wide_xy_jacobian.json"
UPDATE_FILENAME = "task11_wide_xy_jacobian_update.md"
CALIBRATION_RELATIVE_PATH = Path("src/scara/calib") / RESULT_FILENAME
LOCAL_CALIBRATION_RELATIVE_PATH = (
    Path("src/scara/calib") / "camera1_xy_image_jacobian.json"
)


class WideXYImageJacobianCalibrationRuntime(XYImageJacobianCalibrationRuntime):
    """Process Task11 frames and persist a validated wide P22 model."""

    def __init__(
        self,
        output_dir: Path,
        project_root: Path,
        anchor_target_name: str,
        anchor_point_T_mm: Sequence[float],
        anchor_preset_joints: Sequence[float],
        visits: Sequence[tuple[str, int, Sequence[float]]],
        training_offsets_xy_mm: Sequence[Sequence[float]],
        validation_offsets_xy_mm: Sequence[Sequence[float]],
        frames_per_visit: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        self.visits = [
            (str(phase), int(pass_index), [float(offset[0]), float(offset[1])])
            for phase, pass_index, offset in visits
        ]
        self.training_offsets_xy_mm = [
            [float(offset[0]), float(offset[1])]
            for offset in training_offsets_xy_mm
        ]
        self.validation_offsets_xy_mm = [
            [float(offset[0]), float(offset[1])]
            for offset in validation_offsets_xy_mm
        ]
        unique_offsets = {
            (round(offset[0], 6), round(offset[1], 6))
            for _phase, _pass, offset in self.visits
        }
        self._last_task11_metadata: Optional[dict[str, Any]] = None
        super().__init__(
            output_dir=output_dir,
            project_root=project_root,
            anchor_target_name=anchor_target_name,
            anchor_point_T_mm=anchor_point_T_mm,
            anchor_preset_joints=anchor_preset_joints,
            command_offsets_xy_mm=[offset for _phase, _pass, offset in self.visits],
            frames_per_offset=frames_per_visit,
            parent=parent,
            quality=DEFAULT_XY_IMAGE_JACOBIAN_QUALITY,
            expected_unique_offset_count=len(unique_offsets),
            maximum_offset_extent_mm=10.0,
            confirmation_title="Task11 20×20mm宽域Jacobian运动确认",
            confirmation_text=(
                "Task11会实际移动机械臂并采集宽域标定数据。\n\n"
                "开始前请确认：\n"
                "1. 机械臂精确位于P22 float观察高度，已使用低速，急停可用；\n"
                "2. P22周围世界XY每轴±10mm以及全部中转路径无障碍；\n"
                "3. 真空关闭，吸盘/硅片不会接触托盘，相机1和托盘固定；\n"
                "4. 程序将访问25个训练节点两遍，再访问16个独立验证节点，"
                "共330张照片；每条路径拆分为≤1mm中转；\n"
                "5. J3与绝对Rz保持不变；程序不下降、不控制DO/真空；\n"
                "6. 完成或安全停止后请确认机械臂位置，正常完成会回到P22。\n\n"
                "本标定耗时较长。确认工作区完全清空后再继续。"
            ),
        )
        # The wide and fine tiers must describe the same physical setup.
        self.local_calibration_path = (
            self.project_root / LOCAL_CALIBRATION_RELATIVE_PATH
        )
        local = _load_json(self.local_calibration_path, "Task9局部Jacobian")
        local_fit = local.get("fit") or local
        local_gates = local_fit.get("quality_gates") or {}
        if (
            local.get("status") != "success"
            or local_fit.get("status") != "success"
            or not REQUIRED_XY_JACOBIAN_QUALITY_GATES.issubset(local_gates)
            or any(
                not isinstance(local_gates.get(name), Mapping)
                or local_gates[name].get("passed") is not True
                for name in REQUIRED_XY_JACOBIAN_QUALITY_GATES
            )
        ):
            raise RuntimeError("Task11要求已通过全部质量门的Task9局部Jacobian")
        local_locked = local.get("locked_inputs") or {}
        if str(local_locked.get("camera_intrinsics_sha256") or "").upper() != self.intrinsics_hash:
            raise RuntimeError("Task9局部Jacobian与当前内参hash不一致")
        if str(local_locked.get("tray_geometry_sha256") or "").upper() != self.geometry_hash:
            raise RuntimeError("Task9局部Jacobian与当前Tray几何hash不一致")
        if str(local_locked.get("suction_target_sha256") or "").upper() != self.suction_target_hash:
            raise RuntimeError("Task9局部Jacobian与当前Stage4吸盘target hash不一致")
        self.local_calibration_hash = _sha256(self.local_calibration_path)

    def parse_point_name(
        self, name: str
    ) -> tuple[str, tuple[float, float], int, int]:
        # TASK11|target=P22|phase=train|pass=1|visit=01|dx=...|dy=...|frame=01/05
        fields = str(name).split("|")
        if len(fields) != 8 or fields[0] != "TASK11":
            raise RuntimeError(f"无法解析Task11途径点名称：{name}")
        parsed: dict[str, str] = {}
        for field in fields[1:]:
            if "=" not in field:
                raise RuntimeError(f"无法解析Task11字段：{field}")
            key, value = field.split("=", 1)
            parsed[key] = value
        try:
            target = parsed["target"]
            phase = parsed["phase"]
            pass_index = int(parsed["pass"])
            visit_index = int(parsed["visit"])
            dx_mm = float(parsed["dx"])
            dy_mm = float(parsed["dy"])
            frame_text, total_text = parsed["frame"].split("/", 1)
            frame_index = int(frame_text)
            frame_total = int(total_text)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f"无法解析Task11途径点名称：{name}") from exc
        if phase not in {"train", "validation"}:
            raise RuntimeError(f"Task11 phase无效：{phase}")
        if pass_index < 1 or visit_index < 1:
            raise RuntimeError("Task11 pass/visit必须为正整数")
        if not all(math.isfinite(value) for value in (dx_mm, dy_mm)):
            raise RuntimeError("Task11偏移包含非有限数值")
        if visit_index > len(self.visits):
            raise RuntimeError(f"Task11 visit越界：{visit_index}")
        expected_phase, expected_pass, expected_offset = self.visits[visit_index - 1]
        if (
            phase != expected_phase
            or pass_index != expected_pass
            or abs(dx_mm - expected_offset[0]) > 1e-6
            or abs(dy_mm - expected_offset[1]) > 1e-6
        ):
            raise RuntimeError("Task11途径点元数据与锁定visit序列不一致")
        self._last_task11_metadata = {
            "phase": phase,
            "pass_index": pass_index,
            "visit_index": visit_index,
        }
        return target, (dx_mm, dy_mm), frame_index, frame_total

    @pyqtSlot(str)
    def on_photo_saved(self, path_text: str) -> None:
        before = len(self._records)
        self._last_task11_metadata = None
        super().on_photo_saved(path_text)
        if len(self._records) == before + 1:
            if self._last_task11_metadata is None:
                self._processing_failed = True
                self._fatal_messages.append("Task11照片缺少phase/pass/visit元数据")
                return
            self._records[-1].update(self._last_task11_metadata)

    def _report_fatal(self, path: Path, exc: BaseException) -> None:
        message = f"处理Task11照片 {path.name} 失败：{exc}"
        self._processing_failed = True
        self._fatal_messages.append(message)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "task11_runtime_error.log").write_text(
                message + "\n\n" + traceback.format_exc(), encoding="utf-8"
            )
        except OSError:
            pass
        if not self._fatal_error_emitted:
            self._fatal_error_emitted = True
            self.fatal_error.emit(message)

    def _acquisition_summary_wide(self) -> dict[str, Any]:
        rejection_counts: Counter[str] = Counter()
        for record in self._records:
            rejection_counts.update(record["rejection_reasons"])
        visits: list[dict[str, Any]] = []
        for visit_index, (phase, pass_index, offset) in enumerate(self.visits, start=1):
            rows = [
                row for row in self._records if row.get("visit_index") == visit_index
            ]
            visits.append(
                {
                    "visit_index": visit_index,
                    "phase": phase,
                    "pass_index": pass_index,
                    "command_offset_xy_mm": list(offset),
                    "frame_count": len(rows),
                    "accepted_frame_count": sum(bool(row["accepted"]) for row in rows),
                    "maximum_command_tracking_error_mm": (
                        max(row["command_tracking_error_mm"] for row in rows)
                        if rows
                        else None
                    ),
                }
            )
        return {
            "expected_visit_count": len(self.visits),
            "expected_frame_count": len(self.visits) * self.frames_per_offset,
            "processed_frame_count": len(self._records),
            "accepted_frame_count": sum(bool(row["accepted"]) for row in self._records),
            "rejection_reason_counts": dict(sorted(rejection_counts.items())),
            "visits": visits,
        }

    def _base_report_wide(
        self, status: str, message: str, fit: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "stage": "stage5_wide_area_xy_image_model",
            "status": status,
            "calibrated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "message": message,
            "anchor_target_name": self.anchor_target_name,
            "valid_target_names": [self.anchor_target_name],
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
                "suction_target_path": str(self.suction_target_path.resolve()),
                "suction_target_sha256": self.suction_target_hash,
                "local_jacobian_path": str(self.local_calibration_path.resolve()),
                "local_jacobian_sha256": self.local_calibration_hash,
            },
            "coordinate_definition": {
                "command_frame": "robot_controller_world_XY",
                "image_error": "slot_pixel_distorted - suction_target_pixel_distorted",
                "model_equation": "image_error_px = f_wide(offset_world_xy_mm)",
                "jacobian_equation": "J_wide(q) = partial f_wide / partial q",
                "correction_equation": "delta_q = -gain * inverse(J_wide(q)) @ image_error",
                "anchor_point_T_mm": list(self.anchor_point_T_mm),
                "anchor_robot_xy_mm": (
                    None
                    if self._anchor_robot_xy_mm is None
                    else self._anchor_robot_xy_mm.astype(float).tolist()
                ),
                "imaging_j3_mm": self.imaging_j3_mm,
                "rz_deg": self.anchor_rz_deg,
                "wide_extent_mm": 10.0,
                "recommended_runtime_margin_mm": 0.50,
                "fine_model_switch_each_axis_mm": 2.0,
            },
            "acquisition_design": {
                "training_grid_axis_mm": [-10.0, -5.0, 0.0, 5.0, 10.0],
                "training_passes": 2,
                "validation_grid_axis_mm": [-7.5, -2.5, 2.5, 7.5],
                "frames_per_visit": self.frames_per_offset,
                "fit_uses_validation_nodes": False,
            },
            "safety_contract": {
                "cartesian_waypoint_step_maximum_mm": 1.0,
                "z_motion": False,
                "rotation_scan": False,
                "vacuum_control": False,
                "returns_to_anchor": True,
            },
            "acquisition": self._acquisition_summary_wide(),
            "runtime_processing": {
                "failed": bool(self._processing_failed),
                "fatal_messages": list(self._fatal_messages),
            },
            "fit": dict(fit),
            "source_run_folder": str(self.output_dir.resolve()),
        }

    def _enrich_manifest_wide(
        self, manifest: dict[str, Any], report: Mapping[str, Any]
    ) -> None:
        records_by_point = {
            int(record["point_sequence"]): record for record in self._records
        }
        for point in manifest.get("points", []):
            record = records_by_point.get(int(point.get("sequence", -1)))
            if record is None:
                continue
            point["task11_phase"] = record.get("phase")
            point["task11_pass_index"] = record.get("pass_index")
            point["task11_visit_index"] = record.get("visit_index")
            point["task11_command_offset_xy_mm"] = record["command_offset_xy_mm"]
            point["task11_measured_offset_xy_mm"] = record["measured_offset_xy_mm"]
            point["task11_command_tracking_error_mm"] = record[
                "command_tracking_error_mm"
            ]
            point["photo_filename"] = record["filename"]
            point["stage3_pose"] = record["stage3"]
            point["stage3_temporal_quality"] = record["temporal_quality"]
            point["slot_pixel_distorted_px"] = record["slot_pixel_distorted_px"]
            point["suction_target_pixel_distorted_px"] = record[
                "suction_target_pixel_distorted_px"
            ]
            point["image_error_px"] = record["image_error_px"]
            point["task11_sample_accepted"] = record["accepted"]
            point["task11_rejection_reasons"] = record["rejection_reasons"]
        manifest["task11_wide_xy_jacobian"] = {
            "status": report["status"],
            "result_file": RESULT_FILENAME,
            "update_file": UPDATE_FILENAME,
            "sample_count": len(self._records),
            "accepted_sample_count": sum(bool(row["accepted"]) for row in self._records),
            "selected_model_type": (report.get("fit") or {}).get(
                "selected_model_type"
            ),
        }

    def _write_markdown_wide(self, report: Mapping[str, Any]) -> None:
        fit = report["fit"]
        selected = fit.get("selected_model") or {}
        gates = fit.get("quality_gates") or {}
        lines = [
            "# Task11 P22宽域Jacobian检查清单",
            "",
            f"- 总状态：`{report['status']}`",
            f"- 处理帧：`{report['acquisition']['processed_frame_count']}/"
            f"{report['acquisition']['expected_frame_count']}`",
            f"- 选择模型：`{fit.get('selected_model_type')}`",
            f"- 训练RMS：`{selected.get('training_rms_px')} px`",
            f"- 独立验证RMS：`{selected.get('validation_rms_px')} px`",
            f"- 独立验证最大误差：`{selected.get('validation_max_px')} px`",
            "",
            "## 质量门",
            "",
            *[
                f"- [{'x' if gate.get('passed') else ' '}] `{name}`: "
                f"`{json.dumps(gate, ensure_ascii=False)}`"
                for name, gate in gates.items()
            ],
            "",
            "## 使用边界",
            "",
            "- 宽域模型只用于P22、固定相机1/分辨率/J3/Rz、每轴±10mm内的粗对准。",
            "- 进入Task9每轴±2mm域后必须切换到已锁定的局部Jacobian。",
            "- 本文件成功不等于Stage7B可跳过运行时相机、控制器、域和响应安全门。",
            "- 详细方程见 `docs/stage5_wide_xy_jacobian.md`。",
        ]
        _atomic_text_write(self.output_dir / UPDATE_FILENAME, "\n".join(lines))

    @pyqtSlot(bool, str, str)
    def on_task_finished(self, ok: bool, message: str, output_dir_text: str) -> None:
        output_dir = Path(output_dir_text)
        if output_dir.resolve() != self.output_dir.resolve():
            raise RuntimeError("Task11运行时输出文件夹不一致")
        manifest = self._load_manifest()
        expected = len(self.visits) * self.frames_per_offset
        samples = [
            {
                "phase": record.get("phase"),
                "pass_index": record.get("pass_index"),
                "command_offset_xy_mm": record["command_offset_xy_mm"],
                "image_error_px": record["image_error_px"],
                "accepted": record["accepted"],
            }
            for record in self._records
        ]
        if self._processing_failed or self._anchor_robot_xy_mm is None:
            reasons = list(self._fatal_messages)
            if self._anchor_robot_xy_mm is None:
                reasons.append("missing measured P22 world-XY anchor")
            fit: dict[str, Any] = {
                "status": "failure",
                "failure_reasons": reasons,
                "quality_gates": {},
            }
            status = "failure"
        elif not ok:
            fit = {
                "status": "failure",
                "failure_reasons": [message],
                "quality_gates": {},
            }
            status = "acquisition_stopped"
        elif len(self._records) != expected:
            fit = {
                "status": "failure",
                "failure_reasons": [
                    f"processed {len(self._records)}/{expected} expected photos"
                ],
                "quality_gates": {},
            }
            status = "failure"
        else:
            fit = fit_wide_xy_image_model(
                samples,
                self.training_offsets_xy_mm,
                self.validation_offsets_xy_mm,
            )
            gates = fit.get("quality_gates") or {}
            gates_complete = bool(
                REQUIRED_WIDE_XY_JACOBIAN_QUALITY_GATES.issubset(gates)
                and all(
                    isinstance(gates.get(name), Mapping)
                    and gates[name].get("passed") is True
                    for name in REQUIRED_WIDE_XY_JACOBIAN_QUALITY_GATES
                )
            )
            status = (
                "success"
                if fit.get("status") == "success" and gates_complete
                else "failure"
            )

        report = self._base_report_wide(status, message, fit)
        self._enrich_manifest_wide(manifest, report)
        _atomic_json_write(self.manifest_path, manifest)
        _atomic_json_write(output_dir / RESULT_FILENAME, report)
        self._write_markdown_wide(report)
        if status == "success":
            _atomic_json_write(
                self.project_root / CALIBRATION_RELATIVE_PATH, report
            )
        elif ok:
            raise RuntimeError(
                "Task11采集完成，但宽域Jacobian独立验证未通过；"
                "未安装camera1_wide_xy_jacobian.json"
            )


def create_camera1_wide_xy_jacobian_runtime(
    output_dir: Path,
    project_root: Path,
    anchor_target_name: str,
    anchor_point_T_mm: Sequence[float],
    anchor_preset_joints: Sequence[float],
    visits: Sequence[tuple[str, int, Sequence[float]]],
    training_offsets_xy_mm: Sequence[Sequence[float]],
    validation_offsets_xy_mm: Sequence[Sequence[float]],
    frames_per_visit: int,
    parent: Optional[QWidget] = None,
) -> WideXYImageJacobianCalibrationRuntime:
    return WideXYImageJacobianCalibrationRuntime(
        output_dir=output_dir,
        project_root=project_root,
        anchor_target_name=anchor_target_name,
        anchor_point_T_mm=anchor_point_T_mm,
        anchor_preset_joints=anchor_preset_joints,
        visits=visits,
        training_offsets_xy_mm=training_offsets_xy_mm,
        validation_offsets_xy_mm=validation_offsets_xy_mm,
        frames_per_visit=frames_per_visit,
        parent=parent,
    )


__all__ = [
    "CALIBRATION_RELATIVE_PATH",
    "RESULT_FILENAME",
    "UPDATE_FILENAME",
    "WideXYImageJacobianCalibrationRuntime",
    "create_camera1_wide_xy_jacobian_runtime",
]
