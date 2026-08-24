"""Strict JSON loading for camera-1 layered silicon detection parameters."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from scara.file_io import atomic_write_text, read_text_snapshot

from .tray_occupancy import SlotDecisionConfig
from .tray_vision_fusion import TrayVisionFusionConfig
from .wafer_shape_quality import WaferQualityConfig


SILICON_DETECTION_CONFIG_FILENAME = "silicon_detection_0818.json"
SILICON_DETECTION_SELECTION_FILENAME = "local_silicon_detection_selection.json"


@dataclass(frozen=True)
class LoadedSiliconDetectionConfig:
    """Validated configuration plus source identity for UI/report auditing."""

    source_path: Path
    source_sha256: str
    profile_name: str
    description: str
    fusion_config: TrayVisionFusionConfig


def default_silicon_detection_config_path(project_root: Path) -> Path:
    return (
        Path(project_root)
        / "src"
        / "scara"
        / "calib"
        / SILICON_DETECTION_CONFIG_FILENAME
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}必须是JSON对象")
    return value


def _exact_keys(
    payload: Mapping[str, Any], expected: set[str], label: str
) -> None:
    missing = sorted(expected - set(payload))
    unknown = sorted(set(payload) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("缺少：" + ", ".join(missing))
        if unknown:
            details.append("未知：" + ", ".join(unknown))
        raise ValueError(f"{label}字段不匹配（{'；'.join(details)}）")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}必须是有限数值")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label}必须是有限数值")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label}必须是整数")
    return int(value)


def _ratio(value: Any, label: str, *, allow_one: bool = True) -> float:
    result = _finite_number(value, label)
    upper_ok = result <= 1.0 if allow_one else result < 1.0
    if result < 0.0 or not upper_ok:
        bracket = "[0, 1]" if allow_one else "[0, 1)"
        raise ValueError(f"{label}必须在{bracket}范围内")
    return result


def _hsv_triplet(value: Any, label: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label}必须是包含3个整数的数组")
    values = tuple(_integer(item, f"{label}[{index}]") for index, item in enumerate(value))
    limits = (179, 255, 255)
    if any(item < 0 or item > limit for item, limit in zip(values, limits)):
        raise ValueError(f"{label}必须符合OpenCV HSV范围 H=0..179, S/V=0..255")
    return values


def _wafer_quality(payload: Mapping[str, Any]) -> WaferQualityConfig:
    expected = {field.name for field in fields(WaferQualityConfig)}
    _exact_keys(payload, expected, "wafer_quality")
    values = dict(payload)
    values["lower_hsv"] = _hsv_triplet(values["lower_hsv"], "wafer_quality.lower_hsv")
    values["upper_hsv"] = _hsv_triplet(values["upper_hsv"], "wafer_quality.upper_hsv")
    if any(lower > upper for lower, upper in zip(values["lower_hsv"], values["upper_hsv"])):
        raise ValueError("wafer_quality.lower_hsv不能高于upper_hsv")

    for name in ("dark_value_max", "dark_saturation_min"):
        values[name] = _integer(values[name], f"wafer_quality.{name}")
        if not 0 <= values[name] <= 255:
            raise ValueError(f"wafer_quality.{name}必须在0..255范围内")
    for name in (
        "stacked_internal_line_count",
        "irregular_outline_vertex_threshold",
        "stacked_l_temporal_window_size",
        "stacked_l_temporal_min_support",
    ):
        values[name] = _integer(values[name], f"wafer_quality.{name}")
        if values[name] < 1:
            raise ValueError(f"wafer_quality.{name}必须为正整数")

    ratio_names = {
        "minimum_area_ratio",
        "maximum_area_ratio",
        "minimum_chromatic_fraction",
        "normal_min_rectangularity",
        "warning_min_rectangularity",
        "normal_min_solidity",
        "warning_min_solidity",
        "normal_max_center_offset_ratio",
        "warning_max_center_offset_ratio",
        "maximum_normal_side_ratio",
        "stacked_second_component_ratio",
        "stacked_internal_line_score",
        "irregular_outline_max_solidity",
        "stacked_second_quadrilateral_ratio",
        "stacked_quadrilateral_min_rectangularity",
        "stacked_quadrilateral_min_solidity",
        "stacked_l_min_leg_ratio",
        "stacked_candidate_min_overlap_ratio",
        "stacked_candidate_max_overlap_ratio",
        "stacked_l_temporal_min_pairwise_iou",
        "slot_boundary_margin_ratio",
    }
    for name in ratio_names:
        values[name] = _ratio(values[name], f"wafer_quality.{name}")
    for name in (
        "normal_max_aspect_ratio",
        "warning_max_aspect_ratio",
        "boundary_max_aspect_ratio",
        "normal_max_yaw_deg",
        "warning_max_yaw_deg",
        "stacked_quadrilateral_max_aspect_ratio",
        "stacked_l_angle_tolerance_deg",
        "stacked_candidate_min_protrusion_px",
        "stacked_l_temporal_max_relative_center_jitter_px",
    ):
        values[name] = _finite_number(values[name], f"wafer_quality.{name}")
        if values[name] < 0.0:
            raise ValueError(f"wafer_quality.{name}不能为负数")

    if not values["minimum_area_ratio"] < values["maximum_area_ratio"]:
        raise ValueError("minimum_area_ratio必须小于maximum_area_ratio")
    if values["normal_max_aspect_ratio"] > values["warning_max_aspect_ratio"]:
        raise ValueError("normal_max_aspect_ratio不能高于warning_max_aspect_ratio")
    if values["boundary_max_aspect_ratio"] > values["normal_max_aspect_ratio"]:
        raise ValueError("boundary_max_aspect_ratio不能高于normal_max_aspect_ratio")
    if values["normal_min_rectangularity"] < values["warning_min_rectangularity"]:
        raise ValueError("normal_min_rectangularity不能低于warning_min_rectangularity")
    if values["normal_min_solidity"] < values["warning_min_solidity"]:
        raise ValueError("normal_min_solidity不能低于warning_min_solidity")
    if values["normal_max_center_offset_ratio"] > values["warning_max_center_offset_ratio"]:
        raise ValueError("normal_max_center_offset_ratio不能高于warning_max_center_offset_ratio")
    if values["normal_max_yaw_deg"] > values["warning_max_yaw_deg"]:
        raise ValueError("normal_max_yaw_deg不能高于warning_max_yaw_deg")
    if (
        values["stacked_candidate_min_overlap_ratio"]
        > values["stacked_candidate_max_overlap_ratio"]
    ):
        raise ValueError("stacked_candidate_min_overlap_ratio不能高于最大重叠率")
    if (
        values["stacked_l_temporal_min_support"]
        > values["stacked_l_temporal_window_size"]
    ):
        raise ValueError("stacked_l_temporal_min_support不能超过窗口大小")
    return WaferQualityConfig(**values)


def load_silicon_detection_config(path: Path) -> LoadedSiliconDetectionConfig:
    """Load one complete config; partial or misspelled profiles fail closed."""

    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"硅片判定参数不是有效UTF-8 JSON：{exc}") from exc
    root = _mapping(payload, "根节点")
    _exact_keys(
        root,
        {
            "schema_version",
            "profile_name",
            "description",
            "tray_vision",
            "wafer_quality",
            "slot_decision",
        },
        "根节点",
    )
    if _integer(root["schema_version"], "schema_version") != 1:
        raise ValueError("只支持schema_version=1")
    profile_name = root["profile_name"]
    description = root["description"]
    if not isinstance(profile_name, str) or not profile_name.strip():
        raise ValueError("profile_name必须是非空字符串")
    if not isinstance(description, str):
        raise ValueError("description必须是字符串")

    tray = _mapping(root["tray_vision"], "tray_vision")
    _exact_keys(tray, {"slot_half_extent_mm", "canonical_patch_size"}, "tray_vision")
    slot_half_extent_mm = _finite_number(
        tray["slot_half_extent_mm"], "tray_vision.slot_half_extent_mm"
    )
    canonical_patch_size = _integer(
        tray["canonical_patch_size"], "tray_vision.canonical_patch_size"
    )
    if slot_half_extent_mm <= 0.0:
        raise ValueError("tray_vision.slot_half_extent_mm必须大于0")
    if canonical_patch_size < 32:
        raise ValueError("tray_vision.canonical_patch_size必须至少为32")

    wafer_quality = _wafer_quality(_mapping(root["wafer_quality"], "wafer_quality"))
    decision = _mapping(root["slot_decision"], "slot_decision")
    _exact_keys(
        decision,
        {"minimum_image_coverage_ratio", "explicit_occlusion_ratio"},
        "slot_decision",
    )
    minimum_image_coverage_ratio = _ratio(
        decision["minimum_image_coverage_ratio"],
        "slot_decision.minimum_image_coverage_ratio",
    )
    explicit_occlusion_ratio = _ratio(
        decision["explicit_occlusion_ratio"],
        "slot_decision.explicit_occlusion_ratio",
    )
    if minimum_image_coverage_ratio <= 0.0:
        raise ValueError("slot_decision.minimum_image_coverage_ratio必须大于0")

    fusion = TrayVisionFusionConfig(
        slot_half_extent_mm=slot_half_extent_mm,
        canonical_patch_size=canonical_patch_size,
        wafer_quality=wafer_quality,
        slot_decision=SlotDecisionConfig(
            minimum_image_coverage_ratio=minimum_image_coverage_ratio,
            explicit_occlusion_ratio=explicit_occlusion_ratio,
        ),
    )
    return LoadedSiliconDetectionConfig(
        source_path=source,
        source_sha256=hashlib.sha256(raw).hexdigest().upper(),
        profile_name=profile_name.strip(),
        description=description,
        fusion_config=fusion,
    )


def silicon_detection_selection_path(project_root: Path) -> Path:
    """Return the ignored, machine-local pointer to the UI's preferred profile."""

    return (
        Path(project_root).expanduser().resolve()
        / SILICON_DETECTION_SELECTION_FILENAME
    )


def preferred_silicon_detection_config_path(project_root: Path) -> Path:
    """Resolve the last profile selected in the UI, or the checked-in default."""

    project = Path(project_root).expanduser().resolve()
    pointer = silicon_detection_selection_path(project)
    if not pointer.is_file():
        return default_silicon_detection_config_path(project).resolve()
    try:
        payload = json.loads(read_text_snapshot(pointer, encoding="utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"本机硅片配置选择记录损坏：{exc}") from exc
    root = _mapping(payload, "本机硅片配置选择记录")
    _exact_keys(root, {"schema_version", "path_kind", "path"}, "本机硅片配置选择记录")
    if _integer(root["schema_version"], "本机硅片配置选择记录.schema_version") != 1:
        raise ValueError("本机硅片配置选择记录只支持schema_version=1")
    path_kind = root["path_kind"]
    path_text = root["path"]
    if path_kind not in {"project_relative", "absolute"}:
        raise ValueError("本机硅片配置选择记录.path_kind无效")
    if not isinstance(path_text, str) or not path_text.strip():
        raise ValueError("本机硅片配置选择记录.path必须是非空字符串")
    selected = Path(path_text).expanduser()
    if path_kind == "project_relative":
        selected = project / selected
    elif not selected.is_absolute():
        raise ValueError("本机硅片配置选择记录中的absolute路径不是绝对路径")
    return selected.resolve()


def save_preferred_silicon_detection_config_path(
    project_root: Path, selected_path: Path
) -> Path:
    """Atomically persist the profile path selected by the operator."""

    project = Path(project_root).expanduser().resolve()
    selected = Path(selected_path).expanduser().resolve()
    try:
        stored_path = str(selected.relative_to(project))
        path_kind = "project_relative"
    except ValueError:
        stored_path = str(selected)
        path_kind = "absolute"
    payload = {
        "schema_version": 1,
        "path_kind": path_kind,
        "path": stored_path,
    }
    pointer = silicon_detection_selection_path(project)
    atomic_write_text(
        pointer,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    return pointer


__all__ = [
    "LoadedSiliconDetectionConfig",
    "SILICON_DETECTION_CONFIG_FILENAME",
    "SILICON_DETECTION_SELECTION_FILENAME",
    "default_silicon_detection_config_path",
    "load_silicon_detection_config",
    "preferred_silicon_detection_config_path",
    "save_preferred_silicon_detection_config_path",
    "silicon_detection_selection_path",
]
