"""Re-fit a completed Task11 run without moving the robot.

This tool consumes the immutable measurement records already present in a
Task11 run folder.  It validates the hashes locked by the original run,
rebuilds the fit with an explicitly supplied repeatability threshold, writes
an auditable replacement report, and installs it only when every quality gate
passes.  It never opens a camera or imports a robot controller.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from scara.file_io import atomic_write_text
from scara.vision.handeye_interaction import sha256_file
from scara.vision.wide_xy_jacobian import (
    REQUIRED_WIDE_XY_JACOBIAN_QUALITY_GATES,
    WideXYJacobianQualityConfig,
    fit_wide_xy_image_model,
)


RESULT_FILENAME = "camera1_wide_xy_jacobian.json"
POINTS_FILENAME = "points.json"
UPDATE_FILENAME = "task11_wide_xy_jacobian_update.md"
INSTALL_RELATIVE_PATH = Path("src/scara/calib") / RESULT_FILENAME
BACKUP_FILENAME = "camera1_wide_xy_jacobian_before_repeatability_policy_override.json"


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"缺少{label}：{path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label}不是有效JSON：{path}（{exc}）") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label}顶层必须是JSON对象")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _axis_offsets(axis: Sequence[Any]) -> list[list[float]]:
    values = [float(value) for value in axis]
    if not values:
        raise RuntimeError("Task11采集设计缺少网格坐标轴")
    return [[x, y] for y in values for x in values]


def _validate_locked_inputs(report: Mapping[str, Any]) -> None:
    locked = report.get("locked_inputs")
    if not isinstance(locked, Mapping):
        raise RuntimeError("原Task11结果缺少locked_inputs；拒绝重处理")
    pairs = (
        ("camera_intrinsics_path", "camera_intrinsics_sha256"),
        ("tray_geometry_path", "tray_geometry_sha256"),
        ("suction_target_path", "suction_target_sha256"),
        ("local_jacobian_path", "local_jacobian_sha256"),
    )
    for path_key, hash_key in pairs:
        path_text = str(locked.get(path_key) or "")
        expected_hash = str(locked.get(hash_key) or "").upper()
        path = Path(path_text)
        if not path_text or not path.is_file() or len(expected_hash) != 64:
            raise RuntimeError(f"原Task11锁定输入无效：{path_key}")
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"原Task11锁定输入已变化：{path_key}；拒绝安装旧数据")


def _samples_from_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    points = manifest.get("points")
    if not isinstance(points, list):
        raise RuntimeError("points.json缺少points列表")
    samples: list[dict[str, Any]] = []
    for point in points:
        if not isinstance(point, Mapping):
            continue
        phase = point.get("task11_phase")
        offset = point.get("task11_command_offset_xy_mm")
        if phase not in {"train", "validation"} or not isinstance(offset, list):
            continue
        samples.append(
            {
                "phase": phase,
                "pass_index": point.get("task11_pass_index"),
                "command_offset_xy_mm": offset,
                "image_error_px": point.get("image_error_px"),
                "accepted": point.get("task11_sample_accepted") is True,
            }
        )
    if not samples:
        raise RuntimeError("points.json中没有Task11样本")
    return samples


def _all_gates_pass(fit: Mapping[str, Any]) -> bool:
    gates = fit.get("quality_gates")
    return bool(
        fit.get("status") == "success"
        and isinstance(gates, Mapping)
        and REQUIRED_WIDE_XY_JACOBIAN_QUALITY_GATES.issubset(gates)
        and all(
            isinstance(gates.get(name), Mapping)
            and gates[name].get("passed") is True
            for name in REQUIRED_WIDE_XY_JACOBIAN_QUALITY_GATES
        )
    )


def _write_update_markdown(path: Path, report: Mapping[str, Any]) -> None:
    fit = report.get("fit") or {}
    selected = fit.get("selected_model") or {}
    gates = fit.get("quality_gates") or {}
    reprocessing = report.get("reprocessing") or {}
    lines = [
        "# Task11 P22宽域Jacobian检查清单",
        "",
        f"- 总状态：`{report.get('status')}`",
        f"- 处理帧：`{(report.get('acquisition') or {}).get('processed_frame_count')}/"
        f"{(report.get('acquisition') or {}).get('expected_frame_count')}`",
        f"- 选择模型：`{fit.get('selected_model_type')}`",
        f"- 训练RMS：`{selected.get('training_rms_px')} px`",
        f"- 独立验证RMS：`{selected.get('validation_rms_px')} px`",
        f"- 独立验证最大误差：`{selected.get('validation_max_px')} px`",
        "",
        "## 重处理记录",
        "",
        f"- 原结果状态：`{reprocessing.get('previous_status')}`",
        f"- 重处理时间：`{reprocessing.get('reprocessed_at')}`",
        "- 操作者授权：将 `maximum_node_repeatability_px` "
        f"从 `{reprocessing.get('previous_threshold_px')}` 调整为 "
        f"`{reprocessing.get('new_threshold_px')}`。",
        "- 本次重处理不移动机械臂、不拍新图；只重用此运行目录已保存的测量。",
        "",
        "## 质量门",
        "",
        *[
            f"- [{'x' if gate.get('passed') else ' '}] `{name}`: "
            f"`{json.dumps(gate, ensure_ascii=False)}`"
            for name, gate in gates.items()
            if isinstance(gate, Mapping)
        ],
        "",
        "## 使用边界",
        "",
        "- 宽域模型只用于P22、固定相机1/分辨率/J3/Rz、每轴±10mm内的粗对准。",
        "- 进入Task9每轴±2mm域后必须切换到已锁定的局部Jacobian。",
        "- 本文件成功不等于Stage7B可跳过运行时相机、控制器、域和响应安全门。",
    ]
    atomic_write_text(path, "\n".join(lines) + "\n", encoding="utf-8")


def reprocess(
    run_dir: Path,
    project_root: Path,
    maximum_node_repeatability_px: float,
    *,
    install: bool,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    project_root = Path(project_root).resolve()
    if maximum_node_repeatability_px <= 0.0:
        raise ValueError("maximum_node_repeatability_px必须为正数")
    result_path = run_dir / RESULT_FILENAME
    manifest_path = run_dir / POINTS_FILENAME
    original = _load_object(result_path, "原Task11结果")
    manifest = _load_object(manifest_path, "Task11 points.json")
    _validate_locked_inputs(original)
    design = original.get("acquisition_design")
    if not isinstance(design, Mapping):
        raise RuntimeError("原Task11结果缺少acquisition_design")
    training_offsets = _axis_offsets(design.get("training_grid_axis_mm") or [])
    validation_offsets = _axis_offsets(design.get("validation_grid_axis_mm") or [])
    samples = _samples_from_manifest(manifest)
    original_hash = hashlib.sha256(result_path.read_bytes()).hexdigest().upper()
    previous_threshold = float(
        ((original.get("fit") or {}).get("quality_configuration") or {}).get(
            "maximum_node_repeatability_px", 0.0
        )
    )
    quality = WideXYJacobianQualityConfig(
        maximum_node_repeatability_px=float(maximum_node_repeatability_px)
    )
    fit = fit_wide_xy_image_model(
        samples,
        training_offsets,
        validation_offsets,
        quality=quality,
    )
    passed = _all_gates_pass(fit)
    report = copy.deepcopy(original)
    report["status"] = "success" if passed else "failure"
    report["calibrated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    report["message"] = (
        "Task11已从已保存数据重处理；操作者授权放宽node_repeatability阈值。"
    )
    report["fit"] = fit
    report["reprocessing"] = {
        "method": "offline re-fit from existing Task11 points.json; no robot/camera operation",
        "reprocessed_at": report["calibrated_at"],
        "previous_result_sha256": original_hash,
        "previous_status": original.get("status"),
        "previous_threshold_px": previous_threshold,
        "new_threshold_px": float(maximum_node_repeatability_px),
        "operator_authorized": True,
    }
    manifest["task11_wide_xy_jacobian"] = {
        "status": report["status"],
        "result_file": RESULT_FILENAME,
        "update_file": UPDATE_FILENAME,
        "sample_count": len(samples),
        "accepted_sample_count": sum(bool(sample["accepted"]) for sample in samples),
        "selected_model_type": fit.get("selected_model_type"),
        "reprocessed_with_node_repeatability_limit_px": float(
            maximum_node_repeatability_px
        ),
    }
    if not passed:
        raise RuntimeError(
            "重处理后仍有质量门失败："
            + ", ".join(fit.get("failure_reasons") or ["unknown"])
        )
    _write_json(run_dir / BACKUP_FILENAME, original)
    _write_json(result_path, report)
    _write_json(manifest_path, manifest)
    _write_update_markdown(run_dir / UPDATE_FILENAME, report)
    if install:
        _write_json(project_root / INSTALL_RELATIVE_PATH, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--maximum-node-repeatability-px", type=float, required=True
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="only install after re-fit if every quality gate passes",
    )
    args = parser.parse_args()
    report = reprocess(
        args.run_dir,
        args.project_root,
        args.maximum_node_repeatability_px,
        install=args.install,
    )
    gates = (report.get("fit") or {}).get("quality_gates") or {}
    print(f"status={report['status']}")
    print(
        "node_repeatability="
        f"{(gates.get('node_repeatability') or {}).get('value_px')} px"
    )
    print(f"installed={args.install}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
