"""File-backed coordinator for the Stage7B finite P22 closed loop.

The coordinator owns calculations and evidence only.  It never imports a
controller.  ``ActionWorker`` remains the sole hardware owner and treats every
response from this class as untrusted input requiring a fresh state read and
an independent kinematic audit.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2

from scara.file_io import atomic_write_text

from .handeye_interaction import (
    load_latest_suction_target,
    load_local_xy_jacobian,
    sha256_file,
)
from .stage7b_servo import DEFAULT_STAGE7B_CONFIG, build_stage7b_iteration
from .wide_xy_jacobian import REQUIRED_WIDE_XY_JACOBIAN_QUALITY_GATES


REQUEST_KEY = "stage7b_p22_finite_loop"
RESULT_FILENAME = "stage7b_closed_loop.json"


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"缺少{label}：{path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label}不是有效JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label}顶层必须是JSON对象")
    return payload


class Stage7BSession:
    """Maintain one finite-loop ledger and build worker responses."""

    def __init__(
        self,
        project_root: Path,
        output_dir: Path,
        *,
        force_local_only: bool = False,
    ) -> None:
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)
        self.local_path = self.project_root / "src/scara/calib/camera1_xy_image_jacobian.json"
        self.wide_path = self.project_root / "src/scara/calib/camera1_wide_xy_jacobian.json"
        suction = load_latest_suction_target(self.project_root)
        approved_local = load_local_xy_jacobian(self.project_root, suction)
        if approved_local is None:
            raise RuntimeError("当前Task9局部Jacobian未通过Stage6的hash/质量/相机适用性检查")
        self.local = approved_local
        self.wide = _load(self.wide_path, "Task11宽域Jacobian")
        self.local_hash = sha256_file(self.local_path)
        self.wide_hash = sha256_file(self.wide_path)
        wide_locked = self.wide.get("locked_inputs") or {}
        if str(wide_locked.get("local_jacobian_sha256") or "").upper() != self.local_hash:
            raise RuntimeError("Task11锁定的Task9 Jacobian hash与当前文件不一致")
        current_inputs = {
            "camera_intrinsics_sha256": sha256_file(
                self.project_root / "src/scara/calib/camera1_intrinsics.json"
            ),
            "tray_geometry_sha256": sha256_file(
                self.project_root / "src/scara/calib/tray_board_geometry.json"
            ),
            "suction_target_sha256": suction.source_sha256,
        }
        for key, expected in current_inputs.items():
            if str(wide_locked.get(key) or "").upper() != expected:
                raise RuntimeError(f"Task11锁定的{key}与当前文件不一致")
        wide_fit = self.wide.get("fit") or {}
        wide_gates = wide_fit.get("quality_gates") or {}
        if (
            self.wide.get("status") != "success"
            or wide_fit.get("status") != "success"
            or not REQUIRED_WIDE_XY_JACOBIAN_QUALITY_GATES.issubset(wide_gates)
            or any(
                not isinstance(wide_gates.get(name), Mapping)
                or wide_gates[name].get("passed") is not True
                for name in REQUIRED_WIDE_XY_JACOBIAN_QUALITY_GATES
            )
        ):
            raise RuntimeError("Task11宽域Jacobian未通过全部独立验证质量门")
        wide_camera = self.wide.get("camera") or {}
        wide_resolution = wide_camera.get("resolution") or {}
        local_camera = self.local.get("camera") or {}
        local_resolution = local_camera.get("resolution") or {}
        if (
            self.wide.get("anchor_target_name") != "P22"
            or "P22" not in (self.wide.get("valid_target_names") or [])
            or int(wide_camera.get("source_index", -1)) != 1
            or int(wide_resolution.get("width", -1))
            != int(local_resolution.get("width", -2))
            or int(wide_resolution.get("height", -1))
            != int(local_resolution.get("height", -2))
        ):
            raise RuntimeError("Task11目标名、相机源或分辨率与当前Task9不一致")
        local_coordinate = self.local.get("coordinate_definition") or {}
        wide_coordinate = self.wide.get("coordinate_definition") or {}
        if (
            str(wide_coordinate.get("command_frame") or "")
            != "robot_controller_world_XY"
            or str(wide_coordinate.get("image_error") or "")
            != "slot_pixel_distorted - suction_target_pixel_distorted"
            or abs(float(wide_coordinate.get("wide_extent_mm", -1.0)) - 10.0)
            > 1e-9
            or abs(
                float(wide_coordinate.get("fine_model_switch_each_axis_mm", -1.0))
                - 2.0
            )
            > 1e-9
        ):
            raise RuntimeError("Task11坐标、误差符号或两级适用域定义无效")
        for key, tolerance in (("imaging_j3_mm", 0.01), ("rz_deg", 0.01)):
            try:
                difference = abs(float(local_coordinate[key]) - float(wide_coordinate[key]))
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(f"Task9/Task11缺少一致的{key}") from exc
            if difference > tolerance:
                raise RuntimeError(f"Task9/Task11的{key}不一致（差{difference:.4f}）")
        self.anchor_xy = [float(value) for value in local_coordinate["anchor_robot_xy_mm"]]
        self.required_j3_mm = float(local_coordinate["imaging_j3_mm"])
        self.required_rz_deg = float(local_coordinate["rz_deg"])
        self.iterations: list[dict[str, Any]] = []
        self.cumulative_path_mm = 0.0
        self.photo_sequence = 0
        self.started_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
        self.status = "ready"
        self.result_message = ""
        self.worker_runtime_moves: list[dict[str, Any]] = []
        self.force_local_only = bool(force_local_only)

    def action_task(self) -> dict[str, Any]:
        config = DEFAULT_STAGE7B_CONFIG
        actions = []
        # One extra observation-only request is required after the final
        # possible move.  It can report convergence, but the pure planner's
        # motion_iteration_budget gate forbids another movement.
        for index in range(1, config.maximum_iterations + 2):
            actions.append(
                {
                    "type": "runtime_move_joints",
                    "name": f"单点有限闭环第{index:02d}轮",
                    "request_key": REQUEST_KEY,
                    "target_name": "P22",
                    "calibration_sha256": self.wide_hash,
                    "fine_calibration_sha256": self.local_hash,
                    "anchor_robot_xy_mm": self.anchor_xy,
                    "local_extent_mm": config.wide_extent_mm,
                    "domain_margin_mm": config.wide_margin_mm,
                    "required_j3_mm": self.required_j3_mm,
                    "required_rz_deg": self.required_rz_deg,
                    # Runtime proposals use 0.74 mm, while ActionWorker keeps
                    # a distinct 0.75 mm physical/audit ceiling.  Do not pass
                    # the planning limit here or the reserve would disappear
                    # during the post-motion actual-state audit.
                    "max_xy_step_norm_mm": config.coarse_execution_step_limit_mm,
                    "max_xy_axis_mm": config.coarse_execution_step_limit_mm,
                    "j3_tolerance_mm": config.j3_tolerance_mm,
                    "rz_tolerance_deg": config.rz_tolerance_deg,
                    "target_rz_tolerance_deg": config.target_rz_tolerance_deg,
                    "max_sequential_transient_rz_deg": config.maximum_sequential_transient_rz_deg,
                    "precompensate_rz": True,
                    "enforce_sequential_intermediate_domain": False,
                    "max_state_drift_xy_mm": 0.05,
                    "max_state_drift_joint": 0.05,
                    "max_sequential_transient_xy_mm": config.coarse_maximum_sequential_transient_xy_mm,
                    "move_tolerance": 0.01,
                    "proposal_max_age_s": 8.0,
                    "fk_pose_xy_tolerance_mm": 0.20,
                }
            )
        return {
            "api_version": 1,
            "name": "P22单点有限次数两级视觉闭环",
            "description": "动态演示中的P22自动XY闭环；不含Z、DO或真空动作。",
            "camera_model": {
                "offset_mm": 20.0,
                "angle_reference": "world_negative_y",
                "positive_rotation": "counter_clockwise_from_above",
            },
            "actions": actions,
        }

    @property
    def report_path(self) -> Path:
        return self.output_dir / RESULT_FILENAME

    def _save(self) -> None:
        payload = {
            "schema_version": 1,
            "stage": "stage7b_finite_two_tier_closed_loop",
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "result_message": self.result_message,
            "target_name": "P22",
            "operator_arm": {
                "confirmed": True,
                "confirmed_at": self.started_at,
                "scope": "finite P22 XY loop only; no Z/DO/vacuum",
            },
            "locked_inputs": {
                "wide_jacobian_path": str(self.wide_path.resolve()),
                "wide_jacobian_sha256": self.wide_hash,
                "fine_jacobian_path": str(self.local_path.resolve()),
                "fine_jacobian_sha256": self.local_hash,
            },
            "configuration": DEFAULT_STAGE7B_CONFIG.__dict__,
            "force_local_only": self.force_local_only,
            "cumulative_path_mm": self.cumulative_path_mm,
            "iteration_count": len(self.iterations),
            "iterations": self.iterations,
            "worker_runtime_moves": self.worker_runtime_moves,
            "safety_boundary": {
                "xy_only": True,
                "z_motion": False,
                "do_or_vacuum": False,
                "hardware_owner": "ActionWorker only; every proposal re-audited",
            },
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self.report_path,
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def build_response(
        self,
        request: Mapping[str, Any],
        samples: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        request_id = str(request.get("request_id") or "")
        if str(request.get("request_key") or "") != REQUEST_KEY:
            return {"request_id": request_id, "decision": "abort", "reason": "单点有限闭环request_key不匹配"}
        if str(request.get("calibration_sha256") or "").upper() != self.wide_hash:
            return {"request_id": request_id, "decision": "abort", "reason": "单点有限闭环宽域Jacobian hash在会话中发生变化"}
        if str(request.get("fine_calibration_sha256") or "").upper() != self.local_hash:
            return {"request_id": request_id, "decision": "abort", "reason": "单点有限闭环精细Jacobian hash在会话中发生变化"}

        previous = self.iterations[-1] if self.iterations else None
        report = build_stage7b_iteration(
            samples,
            request,
            self.local,
            self.wide,
            iteration_index=len(self.iterations) + 1,
            cumulative_path_mm=self.cumulative_path_mm,
            previous_iteration=previous,
        )
        if self.force_local_only and report.get("model_tier") != "fine_task9":
            report["decision"] = "safety_rejected"
            report["motion_authorized"] = False
            reasons = list(report.get("failure_reasons") or [])
            reasons.append("full_tray_transition_not_inside_task9_local_domain")
            report["failure_reasons"] = reasons
            report.setdefault("safety_gates", {})[
                "full_tray_requires_task9_local_model"
            ] = {
                "passed": False,
                "actual": report.get("model_tier"),
                "limit": "fine_task9",
                "note": (
                    "全盘定位在一次Stage3毫米修正后只允许进入Task9局部精修；"
                    "不会回退到Task11宽域模型。"
                ),
            }
        # Persist the exact five images used for this decision.  They share the
        # established source-prefix naming convention used by trajectory tasks.
        image_names: list[str] = []
        for sample in samples:
            image = sample.get("annotated_bgr")
            if image is None:
                continue
            self.photo_sequence += 1
            filename = f"1_{self.photo_sequence:03d}.jpg"
            if not cv2.imwrite(str(self.output_dir / filename), image):
                self.status = "failure"
                self.result_message = f"无法保存单点有限闭环证据图片 {filename}"
                self._save()
                return {"request_id": request_id, "decision": "abort", "reason": self.result_message}
            image_names.append(filename)
        report["evidence_image_filenames"] = image_names
        report["request_id"] = request_id
        report["wide_calibration_sha256"] = self.wide_hash
        report["fine_calibration_sha256"] = self.local_hash
        report["calculation"] = {
            "commanded_correction_xy_mm": report.get("commanded_correction_xy_mm"),
            "predicted_endpoint_xy_mm": report.get("predicted_endpoint_xy_mm"),
            "predicted_error_px": report.get("predicted_error_px"),
        }
        self.iterations.append(report)
        self.status = "running"
        if report["decision"] == "converged":
            self.status = "converged"
            if report.get("convergence_reason") == "within_1mm":
                self.result_message = (
                    "已经抵达距离目标点1mm以内"
                    f"（视觉模型估计剩余XY距离="
                    f"{report['remaining_alignment_distance_mm']:.3f}mm）"
                )
            else:
                self.result_message = (
                    f"图像误差|e|={report['error_norm_px']:.3f}px <= "
                    f"{DEFAULT_STAGE7B_CONFIG.convergence_error_norm_px:.3f}px，"
                    "单点有限闭环已到达"
                )
            self._save()
            return {
                "request_id": request_id,
                "decision": "complete",
                "reason": self.result_message,
                "calibration_sha256": self.wide_hash,
                "fine_calibration_sha256": self.local_hash,
                "evaluation": report,
            }
        if report["decision"] != "move" or report.get("motion_authorized") is not True:
            self.status = "safety_rejected"
            self.result_message = "安全门拒绝：" + ", ".join(report.get("failure_reasons") or ["unknown"])
            self._save()
            return {
                "request_id": request_id,
                "decision": "abort",
                "reason": self.result_message,
                "evaluation": report,
            }
        command = report.get("commanded_correction_xy_mm") or [0.0, 0.0]
        self.cumulative_path_mm = float(report["cumulative_path_after_mm"])
        self._save()
        proposal = dict(report)
        return {
            "request_id": request_id,
            "decision": "approve",
            "calibration_sha256": self.wide_hash,
            "proposal": proposal,
            "target_joints": list((report.get("planner") or {}).get("target_joints") or []),
        }

    def finish(self, ok: bool, message: str) -> None:
        if not ok and self.status == "converged":
            self.status = "stopped_after_computation"
            self.result_message = str(message)
        elif self.status not in {"converged", "safety_rejected", "failure"}:
            self.status = "completed_max_iterations" if ok else "stopped"
            self.result_message = str(message)
        manifest_path = self.output_dir / "points.json"
        try:
            manifest = _load(manifest_path, "Stage7B points.json")
            runtime_moves = manifest.get("runtime_moves") or []
            if isinstance(runtime_moves, list):
                self.worker_runtime_moves = [
                    dict(item) for item in runtime_moves if isinstance(item, Mapping)
                ]
            manifest["stage7b"] = {
                "status": self.status,
                "result_file": RESULT_FILENAME,
                "iteration_count": len(self.iterations),
                "cumulative_path_mm": self.cumulative_path_mm,
                "wide_jacobian_sha256": self.wide_hash,
                "fine_jacobian_sha256": self.local_hash,
            }
            manifest["stage7b_waypoints"] = [
                {
                    "iteration_index": item.get("iteration_index"),
                    "model_tier": item.get("model_tier"),
                    "current_robot_xy_mm": item.get("current_robot_xy_mm"),
                    "current_offset_xy_mm": item.get("current_offset_xy_mm"),
                    "image_error_px_before": item.get("median_error_px"),
                    "image_error_norm_px_before": item.get("error_norm_px"),
                    "commanded_correction_xy_mm": item.get(
                        "commanded_correction_xy_mm"
                    ),
                    "predicted_endpoint_xy_mm": item.get(
                        "predicted_endpoint_xy_mm"
                    ),
                    "predicted_error_px": item.get("predicted_error_px"),
                    "decision": item.get("decision"),
                    "evidence_image_filenames": item.get(
                        "evidence_image_filenames"
                    ),
                    "safety_gates": item.get("safety_gates"),
                }
                for item in self.iterations
            ]
            atomic_write_text(
                manifest_path,
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            # The primary report remains complete even if manifest enrichment
            # is impossible after an external interruption.
            pass
        self._save()


__all__ = ["REQUEST_KEY", "RESULT_FILENAME", "Stage7BSession"]
