"""Controller-free coordinator for full-tray coarse-to-fine P22 positioning.

The session prepares evidence and proposed targets only.  Static coarse-route
targets are executed by ``ActionWorker`` and the one live Stage-3 metric
correction plus every Task9 fine correction are independently re-audited by
that hardware-owning worker immediately before motion.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from scara.file_io import atomic_write_text

from .full_tray_positioning import (
    GEOMETRY_CORRECTION_DOMAIN_MARGIN_MM,
    GEOMETRY_CORRECTION_LOCAL_EXTENT_MM,
    GEOMETRY_CORRECTION_MAX_STEP_MM,
    GEOMETRY_CORRECTION_MAX_TRANSIENT_RZ_DEG,
    GEOMETRY_CORRECTION_MAX_TRANSIENT_XY_MM,
    build_metric_geometry_correction,
    plan_geometry_coarse_route,
    slot_world_xy_mm,
)
from .handeye_interaction import sha256_file
from .stage7b_session import REQUEST_KEY as STAGE7B_REQUEST_KEY
from .stage7b_session import Stage7BSession
from .tray_pose_estimator import load_tray_board_geometry


FULL_TRAY_GEOMETRY_REQUEST_KEY = "full_tray_p22_metric_geometry_correction"
RESULT_FILENAME = "full_tray_positioning.json"


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != int(length):
        raise ValueError(f"{label}必须包含{length}个数值")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label}包含NaN/Inf")
    return result


class FullTrayPositioningSession:
    """One geometry coarse phase, one metric correction, then Task9 fine loop."""

    def __init__(
        self,
        project_root: Path,
        output_dir: Path,
        initial_robot_state: Mapping[str, Any],
        *,
        target_name: str,
    ) -> None:
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)
        self.target_name = str(target_name)
        if self.target_name != "P22":
            raise RuntimeError(
                "当前全盘定位运动授权只开放P22；36槽目标数据结构已保留，"
                "其他槽需各自通过局部Jacobian标定后再开放。"
            )
        captured = float(initial_robot_state.get("captured_monotonic_s", math.nan))
        state_age = time.monotonic() - captured
        if not math.isfinite(captured) or state_age < 0.0 or state_age > 1.0:
            raise RuntimeError("启动全盘定位时机械臂状态必须是1秒内的新鲜读数")
        self.initial_joints = _finite_vector(
            initial_robot_state.get("joints"), 4, "initial joints"
        )
        self.initial_pose = _finite_vector(
            initial_robot_state.get("pose"), 6, "initial pose"
        )
        self.geometry_path = (
            self.project_root / "src/scara/calib/tray_board_geometry.json"
        )
        self.geometry = load_tray_board_geometry(self.geometry_path)
        self.geometry_hash = sha256_file(self.geometry_path)
        # Reuse the already validated Task8/Task9/Task11 loading and all Stage7B
        # response math, but prohibit fallback to the wide model after the
        # metric correction: the requested last phase is specifically Task9.
        self.fine_session = Stage7BSession(
            self.project_root,
            self.output_dir,
            force_local_only=True,
        )
        self.coarse_route = plan_geometry_coarse_route(
            self.initial_joints,
            self.initial_pose,
            self.geometry,
            self.target_name,
            required_j3_mm=self.fine_session.required_j3_mm,
            required_rz_deg=self.fine_session.required_rz_deg,
        )
        self.geometry_target_world_xy_mm = slot_world_xy_mm(
            self.geometry, self.target_name
        ).astype(float).tolist()
        geometry_anchor_offset = (
            np.asarray(self.fine_session.anchor_xy, dtype=np.float64)
            - np.asarray(self.geometry_target_world_xy_mm, dtype=np.float64)
        )
        self.geometry_to_task9_anchor_offset_xy_mm = (
            geometry_anchor_offset.astype(float).tolist()
        )
        if (
            float(np.linalg.norm(geometry_anchor_offset))
            > GEOMETRY_CORRECTION_MAX_STEP_MM + 1e-9
        ):
            raise RuntimeError(
                "Stage2几何P22与Task9机械锚点相差"
                f"{float(np.linalg.norm(geometry_anchor_offset)):.3f}mm，"
                f"超过一次几何修正{GEOMETRY_CORRECTION_MAX_STEP_MM:.1f}mm上限；"
                "为避免粗移动后才失败，启动前已拒绝。"
            )
        self.all_slot_world_xy_mm = {
            name: slot_world_xy_mm(self.geometry, name).astype(float).tolist()
            for name in sorted(self.geometry["slots"])
        }
        self.started_at = datetime.now().astimezone().isoformat(
            timespec="milliseconds"
        )
        self.status = "ready"
        self.result_message = ""
        self.geometry_correction: dict[str, Any] | None = None
        self.geometry_evidence_images: list[str] = []
        # Do not create ``output_dir`` here.  ActionWorker is the sole owner of
        # run-directory creation (exist_ok=False), which prevents accidental
        # reuse/overwrite of an earlier run.  The first report write happens
        # only after ActionWorker has created the directory, or during finish.

    @property
    def report_path(self) -> Path:
        return self.output_dir / RESULT_FILENAME

    def action_task(self) -> dict[str, Any]:
        actions: list[dict[str, Any]] = [
            {
                "type": "assert_joints",
                "name": "全盘定位启动状态绑定",
                "joints": list(self.initial_joints),
                "tolerance": 0.20,
            }
        ]
        for waypoint in self.coarse_route["waypoints"]:
            actions.append(
                {
                    "type": "move_joints",
                    "name": (
                        f"P22几何粗定位航点 {waypoint['index']:02d}/"
                        f"{self.coarse_route['waypoint_count']:02d}"
                    ),
                    "joints": list(waypoint["target_joints"]),
                    "tolerance": 0.01,
                    "require_current_j3_mm": self.fine_session.required_j3_mm,
                    "j3_tolerance_mm": 0.20,
                }
            )
        actions.extend(
            [
                {"type": "wait", "seconds": 0.30},
                {
                    "type": "runtime_move_joints",
                    "name": "P22一次Stage3毫米几何修正",
                    "request_key": FULL_TRAY_GEOMETRY_REQUEST_KEY,
                    "target_name": self.target_name,
                    "calibration_sha256": self.geometry_hash,
                    "anchor_robot_xy_mm": list(self.fine_session.anchor_xy),
                    "local_extent_mm": GEOMETRY_CORRECTION_LOCAL_EXTENT_MM,
                    "domain_margin_mm": GEOMETRY_CORRECTION_DOMAIN_MARGIN_MM,
                    "required_j3_mm": self.fine_session.required_j3_mm,
                    "required_rz_deg": self.fine_session.required_rz_deg,
                    "max_xy_step_norm_mm": GEOMETRY_CORRECTION_MAX_STEP_MM,
                    "max_xy_axis_mm": GEOMETRY_CORRECTION_MAX_STEP_MM,
                    "j3_tolerance_mm": 0.20,
                    "rz_tolerance_deg": 0.30,
                    "target_rz_tolerance_deg": 0.15,
                    "max_sequential_transient_rz_deg": (
                        GEOMETRY_CORRECTION_MAX_TRANSIENT_RZ_DEG
                    ),
                    "precompensate_rz": True,
                    "enforce_sequential_intermediate_domain": False,
                    "max_state_drift_xy_mm": 0.20,
                    "max_state_drift_joint": 0.20,
                    "max_sequential_transient_xy_mm": (
                        GEOMETRY_CORRECTION_MAX_TRANSIENT_XY_MM
                    ),
                    "move_tolerance": 0.01,
                    "proposal_max_age_s": 8.0,
                    "fk_pose_xy_tolerance_mm": 0.20,
                },
                {"type": "wait", "seconds": 0.30},
            ]
        )
        actions.extend(self.fine_session.action_task()["actions"])
        return {
            "api_version": 1,
            "name": "P22全盘几何粗定位与Task9局部精修",
            "description": (
                "P00/托盘几何生成P22粗目标，内部短航点执行一次粗定位阶段；"
                "随后一次Stage3毫米几何修正，并仅使用Task9局部Jacobian有限精修。"
            ),
            "camera_model": {
                "offset_mm": 20.0,
                "angle_reference": "world_negative_y",
                "positive_rotation": "counter_clockwise_from_above",
            },
            "actions": actions,
        }

    def _save_geometry_images(
        self, samples: Sequence[Mapping[str, Any]]
    ) -> list[str]:
        names: list[str] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(samples, start=1):
            image = sample.get("annotated_bgr")
            if image is None:
                continue
            filename = f"1_{index:03d}.jpg"
            if not cv2.imwrite(str(self.output_dir / filename), image):
                raise RuntimeError(f"无法保存全盘定位几何修正证据图片 {filename}")
            names.append(filename)
        self.fine_session.photo_sequence = max(
            self.fine_session.photo_sequence, len(names)
        )
        return names

    def build_response(
        self,
        request: Mapping[str, Any],
        samples: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        key = str(request.get("request_key") or "")
        if key == STAGE7B_REQUEST_KEY:
            return self.fine_session.build_response(request, samples)
        request_id = str(request.get("request_id") or "")
        if key != FULL_TRAY_GEOMETRY_REQUEST_KEY:
            return {
                "request_id": request_id,
                "decision": "abort",
                "reason": "全盘定位收到未知运行时请求",
            }
        if str(request.get("target_name") or "") != self.target_name:
            return {
                "request_id": request_id,
                "decision": "abort",
                "reason": "全盘定位请求目标与左侧锁定目标不一致",
            }
        if str(request.get("calibration_sha256") or "").upper() != self.geometry_hash:
            return {
                "request_id": request_id,
                "decision": "abort",
                "reason": "全盘定位期间托盘几何hash发生变化",
            }
        report = build_metric_geometry_correction(
            samples,
            request,
            self.geometry,
            target_name=self.target_name,
            transition_anchor_xy_mm=self.fine_session.anchor_xy,
        )
        try:
            image_names = self._save_geometry_images(samples)
        except Exception as exc:  # noqa: BLE001 - fail closed before motion
            self.status = "failure"
            self.result_message = str(exc)
            self._save()
            return {
                "request_id": request_id,
                "decision": "abort",
                "reason": self.result_message,
            }
        report["request_id"] = request_id
        report["evidence_image_filenames"] = image_names
        report["tray_geometry_sha256"] = self.geometry_hash
        self.geometry_correction = report
        self.geometry_evidence_images = image_names
        if report.get("motion_authorized") is not True:
            self.status = "safety_rejected"
            self.result_message = "一次几何修正安全门拒绝：" + ", ".join(
                report.get("failure_reasons") or ["unknown"]
            )
            self._save()
            return {
                "request_id": request_id,
                "decision": "abort",
                "reason": self.result_message,
                "evaluation": report,
            }
        target_joints = list(
            ((report.get("planner") or {}).get("target_joints") or [])
        )
        self.status = "geometry_correction_authorized"
        self.result_message = "一次Stage3毫米几何修正已提交ActionWorker复核"
        self._save()
        return {
            "request_id": request_id,
            "decision": "approve",
            "calibration_sha256": self.geometry_hash,
            "proposal": report,
            "target_joints": target_joints,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "stage": "full_tray_geometry_to_task9_local_positioning",
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            ),
            "result_message": self.result_message,
            "target_name": self.target_name,
            "currently_authorized_target_names": ["P22"],
            "future_target_structure": {
                "slot_count": 36,
                "slot_names": sorted(self.all_slot_world_xy_mm),
                "note": "其他槽须完成各自Task9局部Jacobian后才可加入授权列表",
            },
            "locked_inputs": {
                "tray_geometry_path": str(self.geometry_path.resolve()),
                "tray_geometry_sha256": self.geometry_hash,
                "task9_jacobian_sha256": self.fine_session.local_hash,
                "task11_jacobian_sha256": self.fine_session.wide_hash,
            },
            "slot_world_xy_mm": self.all_slot_world_xy_mm,
            "geometry_coarse_phase": self.coarse_route,
            "geometry_to_task9_anchor_offset_xy_mm": (
                self.geometry_to_task9_anchor_offset_xy_mm
            ),
            "stage3_metric_geometry_correction": self.geometry_correction,
            "task9_local_fine_phase": {
                "forced_local_only": True,
                "report_file": "stage7b_closed_loop.json",
                "iteration_count": len(self.fine_session.iterations),
                "status": self.fine_session.status,
            },
            "safety_boundary": {
                "xy_only": True,
                "fixed_j3_mm": self.fine_session.required_j3_mm,
                "fixed_absolute_rz_deg": self.fine_session.required_rz_deg,
                "z_motion": False,
                "do_or_vacuum": False,
                "target_selection_locked_at_start": self.target_name,
                "hardware_owner": "ActionWorker only",
            },
        }

    def _save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self.report_path,
            json.dumps(self._payload(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def finish(self, ok: bool, message: str) -> None:
        self.fine_session.finish(ok, message)
        if self.fine_session.status == "converged":
            self.status = "converged"
        elif ok:
            self.status = "not_converged"
        elif self.status not in {"safety_rejected", "failure"}:
            self.status = "stopped"
        self.result_message = str(message)
        manifest_path = self.output_dir / "points.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            manifest["full_tray_positioning"] = {
                "status": self.status,
                "target_name": self.target_name,
                "result_file": RESULT_FILENAME,
                "coarse_waypoint_count": self.coarse_route["waypoint_count"],
                "geometry_correction_evidence_images": self.geometry_evidence_images,
                "task9_fine_iteration_count": len(self.fine_session.iterations),
            }
            atomic_write_text(
                manifest_path,
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        self._save()


__all__ = [
    "FULL_TRAY_GEOMETRY_REQUEST_KEY",
    "FullTrayPositioningSession",
    "RESULT_FILENAME",
]
