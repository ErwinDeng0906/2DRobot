"""Read-only Task14 wafer detection, annotation, and report generation.

Motion and capture are owned by the imported action/ActionWorker.  This module
preserves each raw camera-1 image, runs the existing Stage3 tracker and layered
tray vision pipeline, publishes the annotated result under the original
``1_XXX.jpg`` name, and writes diagnostics into the same timestamp folder.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSlot
from PyQt6.QtWidgets import QWidget

from scara.file_io import atomic_write_text, read_text_snapshot
from scara.ui.dialogs import ask_light_warning_confirmation

from .slot_marker_observation import load_slot_marker_layout
from .silicon_detection_config import (
    default_silicon_detection_config_path,
    load_silicon_detection_config,
)
from .tray_pose_estimator import (
    TrayBoardPoseEstimator,
    load_camera_intrinsics,
    load_tray_board_geometry,
)
from .tray_pose_tracker import TrayPoseTracker
from .tray_vision_fusion import TrayVisionAnalyzer
from .wafer_shape_quality import WaferQualityConfig


RESULT_FILENAME = "task14_silicon_detection.json"
SUMMARY_FILENAME = "task14_silicon_detection_summary.md"
RECOMMENDED_CONFIG_FILENAME = "silicon_detection_recommended.json"
ERROR_FILENAME = "task14_silicon_detection_error.log"
ANNOTATED_DIRECTORY = "annotated_task14"
RAW_DIRECTORY = "raw_task14"
MINIMUM_ACCEPTABLE_FRAME_RATE = 0.60
MINIMUM_VALID_OBSERVATIONS_PER_SLOT = 5

_POINT_NAME_RE = re.compile(r"^TASK14\|(P[0-5][0-5])\|frame=(\d{2})/(\d{2})$")
_PHOTO_NAME_RE = re.compile(r"^1_(\d+)\.jpg$", re.IGNORECASE)
_NORMAL_STATES = {"occupied", "warning"}
_OUTSIDE_STATES = {"outside_slot", "stacked_outside_slot"}
_EMPTY_STATES = {"empty"}
_EXPECTED_STATES = {
    "normal_wafer": _NORMAL_STATES,
    "outside_wafer": _OUTSIDE_STATES,
    "empty": _EMPTY_STATES,
}
_EXCLUDED_OBSERVATION_STATES = {
    "out_of_view": "out_of_view",
    "occluded": "occluded",
    "unknown": "unknown",
    "empty_unread_marker": "marker_unread",
    "unavailable": "unavailable",
}
_LOCKED_SLOT_GEOMETRY_PARAMETERS = (
    "tray_vision.slot_half_extent_mm",
    "tray_vision.canonical_patch_size",
    "wafer_quality.slot_boundary_margin_ratio",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _write_jpeg_atomically(path: Path, image: np.ndarray) -> None:
    """Publish one JPEG without leaving a partially overwritten capture."""
    path = Path(path)
    temporary = path.with_name(f".{path.stem}.task14{path.suffix}")
    try:
        if not cv2.imwrite(str(temporary), image):
            raise RuntimeError(f"保存Task14标注图失败：{path.name}")
        temporary.replace(path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _numeric_summary(values: Sequence[object]) -> dict[str, Optional[float] | int]:
    clean = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    if clean.size == 0:
        return {"count": 0, "minimum": None, "median": None, "mean": None, "maximum": None}
    return {
        "count": int(clean.size),
        "minimum": float(np.min(clean)),
        "median": float(np.median(clean)),
        "mean": float(np.mean(clean)),
        "maximum": float(np.max(clean)),
    }


def _frame_registration_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Count mutually exclusive analysis projection modes for the report."""

    stage3_passed = 0
    strict_pnp = 0
    planar_fallback = 0
    unavailable = 0
    for record in records:
        stage3 = record.get("stage3")
        if isinstance(stage3, Mapping) and stage3.get("quality_passed") is True:
            stage3_passed += 1
        source = str(record.get("projection_source") or "unavailable")
        analysis_passed = record.get("analysis_quality_passed") is True
        if analysis_passed and source.startswith("strict_pnp"):
            strict_pnp += 1
        elif analysis_passed and source in {
            "marker_grid_homography",
            "two_outer_marker_homography",
        }:
            planar_fallback += 1
        else:
            unavailable += 1
    return {
        "stage3_quality_passed_frame_count": stage3_passed,
        "strict_pnp_analysis_frame_count": strict_pnp,
        "planar_fallback_analysis_frame_count": planar_fallback,
        "unavailable_analysis_frame_count": unavailable,
    }


def _vector_median(values: Sequence[object], length: int) -> Optional[list[float]]:
    rows = []
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) != length:
            continue
        row = [float(item) for item in value]
        if all(math.isfinite(item) for item in row):
            rows.append(row)
    if not rows:
        return None
    return [float(value) for value in np.median(np.asarray(rows, dtype=np.float64), axis=0)]


def parse_task14_point_name(name: str) -> tuple[str, int, int]:
    match = _POINT_NAME_RE.fullmatch(str(name))
    if match is None:
        raise ValueError(f"不是Task14路径点名称：{name!r}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def _slot_result_from_record(
    record: Mapping[str, Any], slot_name: str
) -> Optional[Mapping[str, Any]]:
    """Return one slot's result from a full-frame record or legacy target-only record."""

    slot_results = record.get("slot_results")
    if isinstance(slot_results, Mapping):
        value = slot_results.get(slot_name)
        return value if isinstance(value, Mapping) else None
    if str(record.get("target_name")) != str(slot_name):
        return None
    value = record.get("target_slot")
    return value if isinstance(value, Mapping) else None


def _slot_state(
    record: Mapping[str, Any], slot_result: Optional[Mapping[str, Any]]
) -> str:
    if isinstance(slot_result, Mapping):
        decision = slot_result.get("decision")
        if isinstance(decision, Mapping) and decision.get("state"):
            return str(decision["state"])
        if slot_result.get("state"):
            return str(slot_result["state"])
    return str(record.get("observed_state") or "unavailable")


def _observation_exclusion_reason(
    record: Mapping[str, Any],
    slot_name: str,
    slot_result: Optional[Mapping[str, Any]],
) -> Optional[str]:
    """Classify observations that must not enter per-slot detection statistics.

    Every new Task14 record contains results for all 36 slots.  The five frames
    captured while the tool is directly over a slot are deliberately excluded
    for that same slot because the suction assembly is a high-risk occluder.
    The remaining exclusions follow the user's evidence rule: no partial view,
    explicit occlusion, unknown decision, or unread empty-slot marker.
    """

    full_frame_record = isinstance(record.get("slot_results"), Mapping)
    if full_frame_record and str(record.get("target_name")) == str(slot_name):
        return "capture_target_tool_occlusion"
    if record.get("processing_error"):
        return "processing_error"
    if slot_result is None:
        return "unavailable"
    # Compatibility for direct unit callers and schema-v1 reports: those
    # records only contain the capture target and do not represent an all-slot
    # observation from which capture-target occlusion can be inferred.
    if not full_frame_record:
        return None
    return _EXCLUDED_OBSERVATION_STATES.get(_slot_state(record, slot_result))


def _iter_slot_observations(
    slot_name: str,
    records: Sequence[Mapping[str, Any]],
    *,
    include_excluded: bool = False,
):
    for record in sorted(records, key=lambda row: int(row.get("point_sequence") or 0)):
        slot_result = _slot_result_from_record(record, slot_name)
        exclusion_reason = _observation_exclusion_reason(
            record, slot_name, slot_result
        )
        if exclusion_reason is not None and not include_excluded:
            continue
        yield record, slot_result, _slot_state(record, slot_result), exclusion_reason


def _five_frame_group_consistency(
    slot_name: str,
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Summarize each stationary capture burst without hiding frame evidence."""

    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for record in sorted(records, key=lambda row: int(row.get("point_sequence") or 0)):
        capture_target = str(record.get("target_name") or "unavailable")
        slot_result = _slot_result_from_record(record, slot_name)
        if _observation_exclusion_reason(record, slot_name, slot_result) is not None:
            continue
        state = _slot_state(record, slot_result)
        evidence = "unobservable"
        if isinstance(slot_result, Mapping):
            wafer = slot_result.get("wafer")
            if isinstance(wafer, Mapping):
                evidence = str(wafer.get("boundary_evidence") or "unobservable")
        if evidence == "unobservable":
            if state in _OUTSIDE_STATES:
                evidence = "strong_outside"
            elif state == "occupied":
                evidence = "inside"
        grouped[capture_target].append((state, evidence))

    rows: list[dict[str, Any]] = []
    consensus_counts: Counter[str] = Counter()
    for capture_target in sorted(grouped):
        observations = grouped[capture_target]
        state_counts = Counter(state for state, _evidence in observations)
        evidence_counts = Counter(evidence for _state, evidence in observations)
        strong_outside_count = int(evidence_counts.get("strong_outside", 0))
        confident_inside_count = int(
            sum(
                state == "occupied" and evidence == "inside"
                for state, evidence in observations
            )
        )
        empty_count = int(
            state_counts.get("empty", 0) + state_counts.get("empty_unread_marker", 0)
        )
        if strong_outside_count >= 2:
            consensus = "outside_slot"
        elif confident_inside_count >= 3 and strong_outside_count == 0:
            consensus = "occupied"
        elif empty_count >= 3 and strong_outside_count == 0:
            consensus = "empty"
        else:
            consensus = "warning"
        consensus_counts[consensus] += 1
        rows.append(
            {
                "capture_target": capture_target,
                "valid_frame_count": len(observations),
                "state_counts": dict(sorted(state_counts.items())),
                "boundary_evidence_counts": dict(sorted(evidence_counts.items())),
                "strong_outside_frame_count": strong_outside_count,
                "confident_inside_frame_count": confident_inside_count,
                "consensus": consensus,
            }
        )
    return rows, dict(sorted(consensus_counts.items()))


def summarize_task14_slot(
    target_name: str,
    known_point_T_mm: Sequence[float],
    records: Sequence[Mapping[str, Any]],
    *,
    expected_occupied: bool,
    expected_frames: int,
    expected_label: Optional[str] = None,
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda row: int(row.get("point_sequence") or 0))
    observations = list(
        _iter_slot_observations(target_name, ordered, include_excluded=True)
    )
    raw_states = [state for _row, _slot, state, _reason in observations]
    raw_counts = Counter(raw_states)
    exclusions = Counter(
        reason for _row, _slot, _state, reason in observations if reason is not None
    )
    valid = [item for item in observations if item[3] is None]
    states = [state for _row, _slot, state, _reason in valid]
    counts = Counter(states)
    representative_state = (
        sorted(counts, key=lambda state: (-counts[state], state))[0]
        if counts
        else "unavailable"
    )
    resolved_expected_label = str(
        expected_label
        or ("normal_wafer" if expected_occupied else "empty")
    )
    if resolved_expected_label not in _EXPECTED_STATES:
        raise ValueError(
            f"Task14未知预期槽位类别：{resolved_expected_label!r}"
        )
    resolved_expected_occupied = resolved_expected_label != "empty"
    if bool(expected_occupied) is not resolved_expected_occupied:
        raise ValueError("Task14 expected_occupied与expected_label不一致")
    acceptable_states = _EXPECTED_STATES[resolved_expected_label]
    acceptable_count = sum(state in acceptable_states for state in states)
    acceptable_rate = acceptable_count / max(len(valid), 1)
    frame_set_complete = len(ordered) == int(expected_frames)
    baseline_passed = bool(
        frame_set_complete
        and len(valid) >= MINIMUM_VALID_OBSERVATIONS_PER_SLOT
        and representative_state in acceptable_states
        and acceptable_rate >= MINIMUM_ACCEPTABLE_FRAME_RATE
    )
    slots = [slot for _row, slot, _state, _reason in valid if isinstance(slot, Mapping)]
    wafers = [slot.get("wafer") for slot in slots if isinstance(slot.get("wafer"), Mapping)]
    five_frame_groups, five_frame_consensus_counts = _five_frame_group_consistency(
        target_name, ordered
    )
    other_counts = {
        state: int(count)
        for state, count in sorted(counts.items())
        if state not in acceptable_states
    }
    return {
        "target_name": str(target_name),
        "known_slot_center_T_mm": [float(value) for value in known_point_T_mm],
        "expected_occupied": resolved_expected_occupied,
        "expected_label": resolved_expected_label,
        "captured_frame_count": len(ordered),
        "expected_frame_count": int(expected_frames),
        "frame_set_complete": frame_set_complete,
        "valid_observation_count": len(valid),
        "excluded_observation_count": len(observations) - len(valid),
        "valid_observation_rate": float(len(valid) / max(int(expected_frames), 1)),
        "minimum_valid_observation_count": MINIMUM_VALID_OBSERVATIONS_PER_SLOT,
        "exclusion_counts": dict(sorted(exclusions.items())),
        "raw_state_counts": dict(sorted(raw_counts.items())),
        "state_counts": dict(sorted(counts.items())),
        "representative_state": representative_state,
        "normal_frame_count": int(counts.get("occupied", 0)),
        "outside_frame_count": int(
            counts.get("outside_slot", 0)
            + counts.get("stacked_outside_slot", 0)
        ),
        "empty_frame_count": int(counts.get("empty", 0)),
        "other_state_counts": other_counts,
        "acceptable_frame_count": int(acceptable_count),
        "acceptable_frame_rate_of_valid": float(acceptable_rate),
        "acceptable_frame_rate_of_expected": float(
            acceptable_count / max(int(expected_frames), 1)
        ),
        "baseline_passed": baseline_passed,
        "five_frame_group_consistency": five_frame_groups,
        "five_frame_group_consensus_counts": five_frame_consensus_counts,
        "wafer_center_T_mm_median": _vector_median(
            [slot.get("wafer_center_T_mm") for slot in slots], 3
        ),
        "wafer_offset_T_mm_median": _vector_median(
            [slot.get("wafer_offset_T_mm") for slot in slots], 2
        ),
        "wafer_offset_distance_mm": _numeric_summary(
            [slot.get("wafer_offset_distance_mm") for slot in slots]
        ),
        "allowed_tuning_measurements": {
            "candidate_area_ratio": _numeric_summary(
                [wafer.get("area_ratio") for wafer in wafers]
            ),
            "size_to_slot_side_ratio": _numeric_summary(
                [wafer.get("side_ratio") for wafer in wafers]
            ),
            "second_component_to_primary_area_ratio": _numeric_summary(
                [wafer.get("second_component_area_ratio") for wafer in wafers]
            ),
            "internal_line_count": _numeric_summary(
                [wafer.get("internal_line_count") for wafer in wafers]
            ),
            "internal_line_score": _numeric_summary(
                [wafer.get("internal_line_score") for wafer in wafers]
            ),
            "contour_polygon_vertices": _numeric_summary(
                [wafer.get("polygon_vertices") for wafer in wafers]
            ),
            "contour_solidity_for_complexity_gate": _numeric_summary(
                [wafer.get("solidity") for wafer in wafers]
            ),
            "base_projection_clearance_px": _numeric_summary(
                [wafer.get("base_projection_clearance_px") for wafer in wafers]
            ),
            "refined_projection_clearance_px": _numeric_summary(
                [wafer.get("refined_projection_clearance_px") for wafer in wafers]
            ),
            "contour_outside_depth_px": _numeric_summary(
                [wafer.get("contour_outside_depth_px") for wafer in wafers]
            ),
            "contour_outside_support_px": _numeric_summary(
                [wafer.get("contour_outside_support_px") for wafer in wafers]
            ),
            "projection_disagreement_px": _numeric_summary(
                [wafer.get("projection_disagreement_px") for wafer in wafers]
            ),
        },
        "processing_errors": [
            str(row["processing_error"])
            for row in ordered
            if row.get("processing_error")
        ],
    }


def _tuning_scope(config: WaferQualityConfig) -> dict[str, Any]:
    return {
        "policy": (
            "any_silicon_detection_parameter_may_be_recommended_but_the_checked_in_"
            "configuration_is_never_modified_automatically"
        ),
        "recommendation_must_be_reviewed_and_selected_manually": True,
        "locked_slot_geometry_parameters": list(
            _LOCKED_SLOT_GEOMETRY_PARAMETERS
        ),
        "detector_config_snapshot": asdict(config),
    }


_STATE_LABELS = {
    "occupied": "正常",
    "warning": "警告",
    "stacked": "叠片",
    "outside_slot": "槽外",
    "stacked_outside_slot": "叠片且槽外",
    "empty": "空槽",
    "empty_unread_marker": "空槽／Marker未解码",
    "out_of_view": "画面外",
    "occluded": "遮挡",
    "unknown": "证据不足",
    "unavailable": "不可用",
}


def _row_expected_label(row: Mapping[str, Any]) -> str:
    label = row.get("expected_label")
    if label in _EXPECTED_STATES:
        return str(label)
    return "normal_wafer" if row.get("expected_occupied") else "empty"


def _group_slots_by_state(
    summaries: Sequence[Mapping[str, Any]],
    *,
    expected_label: Optional[str] = None,
    expected_occupied: Optional[bool] = None,
) -> dict[str, list[str]]:
    if expected_label is None and expected_occupied is None:
        raise ValueError("必须指定expected_label或expected_occupied")
    grouped: dict[str, list[str]] = {}
    for row in summaries:
        if expected_label is not None:
            if _row_expected_label(row) != expected_label:
                continue
        elif bool(row.get("expected_occupied")) is not bool(expected_occupied):
            continue
        state = str(row.get("representative_state") or "unavailable")
        grouped.setdefault(state, []).append(str(row["target_name"]))
    return {state: sorted(names) for state, names in sorted(grouped.items())}


def _target_wafer_flags(
    target_name: str, records: Sequence[Mapping[str, Any]]
) -> set[str]:
    flags: set[str] = set()
    for _record, slot_result, _state, _reason in _iter_slot_observations(
        target_name, records
    ):
        if not isinstance(slot_result, Mapping):
            continue
        wafer = slot_result.get("wafer")
        if not isinstance(wafer, Mapping):
            continue
        flags.update(str(value) for value in wafer.get("flags", []))
    return flags


def _valid_wafer_observations(
    slot_names: Sequence[str] | set[str], records: Sequence[Mapping[str, Any]]
):
    for slot_name in sorted(str(value) for value in slot_names):
        for record, slot_result, state, _reason in _iter_slot_observations(
            slot_name, records
        ):
            if not isinstance(slot_result, Mapping):
                continue
            wafer = slot_result.get("wafer")
            if isinstance(wafer, Mapping):
                yield slot_name, record, slot_result, wafer, state


def _recommended_config(
    base_payload: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    evidence_valid: bool,
    run_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Create a review-only profile; never mutate the checked-in source file."""

    proposal = json.loads(json.dumps(base_payload, ensure_ascii=False))
    proposal["profile_name"] = f"task14_recommended_{run_name}"
    proposal["description"] = (
        f"Review-only Task14 recommendation for {run_name}. "
        "This file was not applied automatically."
    )
    wafer = proposal["wafer_quality"]
    locked_tray_vision = json.loads(
        json.dumps(base_payload["tray_vision"], ensure_ascii=False)
    )
    locked_boundary_margin = base_payload["wafer_quality"][
        "slot_boundary_margin_ratio"
    ]
    changes: list[dict[str, Any]] = []
    notes: list[str] = []

    def suggest(field: str, new_value: Any, reason: str) -> None:
        if field == "slot_boundary_margin_ratio":
            raise ValueError("槽边界/槽尺寸参数禁止由Task14建议逻辑修改")
        old_value = wafer[field]
        if old_value == new_value:
            return
        wafer[field] = new_value
        changes.append(
            {
                "field": f"wafer_quality.{field}",
                "old": old_value,
                "new": new_value,
                "reason": reason,
            }
        )

    def finish() -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
        if proposal["tray_vision"] != locked_tray_vision:
            raise RuntimeError("Task14建议逻辑不得修改tray_vision槽参数")
        if wafer["slot_boundary_margin_ratio"] != locked_boundary_margin:
            raise RuntimeError("Task14建议逻辑不得修改槽边界参数")
        return proposal, changes, notes

    if not evidence_valid:
        notes.append(
            "本次自动曝光或采集证据无效，因此建议JSON保持原检测参数，不用无效数据调参。"
        )
        return finish()

    expected_rows = [
        row for row in summaries if _row_expected_label(row) == "normal_wafer"
    ]
    empty_rows = [row for row in summaries if _row_expected_label(row) == "empty"]
    empty_occupancy_errors = [
        row
        for row in empty_rows
        if any(
            int(row.get("state_counts", {}).get(state, 0)) > 0
            for state in {
                "occupied",
                "warning",
                "stacked",
                "outside_slot",
                "stacked_outside_slot",
            }
        )
    ]
    missed_rows = [
        row
        for row in expected_rows
        if int(row.get("acceptable_frame_count", 0)) == 0
    ]
    missed_flags = set().union(
        *(
            _target_wafer_flags(str(row["target_name"]), records)
            for row in missed_rows
        ),
        set(),
    )
    if missed_rows and empty_occupancy_errors:
        notes.append(
            "正常硅片存在漏检，同时空槽也出现占用误报；暂不放宽颜色/面积门槛，"
            "避免为了召回率进一步增加空槽误报。先复核自动曝光图和误报轮廓。"
        )
    elif missed_rows:
        missing_names = "、".join(str(row["target_name"]) for row in missed_rows)
        if "no_chromatic_candidate" in missed_flags:
            lower_hsv = list(wafer["lower_hsv"])
            lower_hsv[1] = max(12, int(lower_hsv[1]) - 4)
            lower_hsv[2] = max(8, int(lower_hsv[2]) - 8)
            suggest(
                "lower_hsv",
                lower_hsv,
                f"{missing_names}出现no_chromatic_candidate；适度接纳自动曝光后较暗或低饱和区域。",
            )
            suggest(
                "dark_value_max",
                min(190, int(wafer["dark_value_max"]) + 20),
                f"{missing_names}缺少颜色候选；提高暗色亮度上限，减少光线不足导致的硅片筛除。",
            )
            suggest(
                "dark_saturation_min",
                max(18, int(wafer["dark_saturation_min"]) - 6),
                f"{missing_names}缺少颜色候选；降低暗色饱和度下限以容纳反光造成的退色。",
            )
        if "marker_artifact_rejected" in missed_flags:
            suggest(
                "minimum_chromatic_fraction",
                round(max(0.48, float(wafer["minimum_chromatic_fraction"]) - 0.10), 3),
                f"{missing_names}的候选被minimum_chromatic_fraction筛除；适度降低色彩占比门槛。",
            )
        if "candidate_area_out_of_range" in missed_flags:
            rejected_ratios: list[float] = []
            empty_rejected_ratios: list[float] = []
            for _slot_name, _record, _slot, observation, _state in (
                _valid_wafer_observations(
                    {str(row["target_name"]) for row in missed_rows}, records
                )
            ):
                if "candidate_area_out_of_range" not in observation.get("flags", []):
                    continue
                ratio = observation.get("area_ratio")
                if ratio is not None and math.isfinite(float(ratio)) and float(ratio) > 0.0:
                    rejected_ratios.append(float(ratio))
            empty_target_names = {
                str(row["target_name"])
                for row in empty_rows
            }
            for _slot_name, _record, _slot, observation, _state in (
                _valid_wafer_observations(empty_target_names, records)
            ):
                if "candidate_area_out_of_range" not in observation.get("flags", []):
                    continue
                ratio = observation.get("area_ratio")
                if ratio is not None and math.isfinite(float(ratio)) and float(ratio) > 0.0:
                    empty_rejected_ratios.append(float(ratio))
            if rejected_ratios:
                minimum_allowed = float(wafer["minimum_area_ratio"])
                maximum_allowed = float(wafer["maximum_area_ratio"])
                below_minimum = [
                    ratio for ratio in rejected_ratios if ratio < minimum_allowed
                ]
                above_maximum = [
                    ratio for ratio in rejected_ratios if ratio > maximum_allowed
                ]
                if below_minimum:
                    proposed_minimum = round(
                        max(0.01, min(below_minimum) * 0.90), 3
                    )
                    conflicting_empty = [
                        ratio
                        for ratio in empty_rejected_ratios
                        if proposed_minimum <= ratio < minimum_allowed
                    ]
                    if conflicting_empty:
                        notes.append(
                            "正常硅片候选低于minimum_area_ratio，但相同范围也有空槽候选；"
                            "不建议降低最小硅片面积，以免引入空槽误报。"
                        )
                    else:
                        suggest(
                            "minimum_area_ratio",
                            proposed_minimum,
                            "已知正常硅片候选面积低于当前最小硅片面积门槛，且空槽中"
                            "没有落入新增范围的候选；降低的是硅片大小门槛，不改变槽大小。",
                        )
                if above_maximum:
                    largest = max(above_maximum)
                    proposed_maximum = round(min(0.90, largest * 1.02), 3)
                    conflicting_empty = [
                        ratio
                        for ratio in empty_rejected_ratios
                        if maximum_allowed < ratio <= proposed_maximum
                    ]
                    if largest > maximum_allowed + 0.08:
                        notes.append(
                            "正常槽的越界候选面积远高于硅片上限，优先判断为颜色掩码"
                            "覆盖大部分槽图，不扩大maximum_area_ratio。"
                        )
                    elif conflicting_empty:
                        notes.append(
                            "提高maximum_area_ratio会同时接纳空槽的大面积候选，"
                            "因此不建议扩大最大硅片面积。"
                        )
                    else:
                        suggest(
                            "maximum_area_ratio",
                            proposed_maximum,
                            "已知正常硅片候选仅小幅超过当前最大硅片面积门槛，且空槽"
                            "没有落入扩展范围的候选；扩大的是硅片大小门槛，不改变槽大小。",
                        )
                notes.append(
                    "本次候选面积越界值为"
                    f"{min(rejected_ratios):.3f}–{max(rejected_ratios):.3f}；"
                    "槽物理尺寸和槽图参数保持锁定。"
                )
            else:
                notes.append(
                    "出现candidate_area_out_of_range，但旧记录没有保留越界面积；"
                    "无法判断应调整最小还是最大硅片面积，暂不建议修改。"
                )
        if not missed_flags & {
            "no_chromatic_candidate",
            "marker_artifact_rejected",
            "candidate_area_out_of_range",
        }:
            notes.append(
                f"{missing_names}漏检，但当前记录没有能安全对应到颜色或面积门槛的原因；"
                "建议保留参数并检查标注图，不盲目放宽。"
            )

    expected_target_names = {str(row["target_name"]) for row in expected_rows}
    oversize_side_ratios: list[float] = []
    for _slot_name, _record, _slot, observation, _state in (
        _valid_wafer_observations(expected_target_names, records)
    ):
        if "oversize_footprint" not in observation.get("flags", []):
            continue
        side_ratio = observation.get("side_ratio")
        if side_ratio is not None and math.isfinite(float(side_ratio)):
            oversize_side_ratios.append(float(side_ratio))
    if oversize_side_ratios:
        largest_side_ratio = max(oversize_side_ratios)
        if largest_side_ratio <= 0.95:
            suggest(
                "maximum_normal_side_ratio",
                round(min(0.93, largest_side_ratio * 1.01), 3),
                f"全部有效视角中有{len(oversize_side_ratios)}帧已知正常硅片触发"
                f"oversize_footprint，最大尺寸占槽比例为{largest_side_ratio:.3f}；"
                "仅给观测上限保留约1%余量。调整的是硅片状态门槛，槽的物理尺寸"
                "和槽图定义保持不变。",
            )
        else:
            notes.append(
                "oversize_footprint的尺寸占槽比例接近整槽，优先判断为掩码扩散，"
                "不提高maximum_normal_side_ratio。"
            )

    all_valid_expected_wafers = list(
        _valid_wafer_observations(expected_target_names, records)
    )
    irregular_vertices = [
        int(observation.get("polygon_vertices") or 0)
        for _slot_name, _record, _slot, observation, _state in all_valid_expected_wafers
        if "irregular_outline" in observation.get("flags", [])
        and int(observation.get("polygon_vertices") or 0) > 0
    ]
    if (
        irregular_vertices
        and len(irregular_vertices) >= 0.25 * max(len(all_valid_expected_wafers), 1)
    ):
        current_vertices = int(wafer["irregular_outline_vertex_threshold"])
        proposed_vertices = min(
            14,
            max(current_vertices + 1, int(round(float(np.median(irregular_vertices))))),
        )
        suggest(
            "irregular_outline_vertex_threshold",
            proposed_vertices,
            f"全部有效视角中{len(irregular_vertices)}/{len(all_valid_expected_wafers)}帧"
            "已知正常硅片触发irregular_outline；这些帧的轮廓顶点中位数为"
            f"{float(np.median(irregular_vertices)):.1f}。将复杂轮廓门槛提高到"
            f"{proposed_vertices}，减少反光和斜视造成的正常硅片警告；低矩形度、"
            "低实心度及槽外证据仍独立保留。",
        )

    normal_outside = [
        row
        for row in expected_rows
        if row.get("representative_state") in {"outside_slot", "stacked_outside_slot"}
    ]
    if normal_outside:
        names = "、".join(str(row["target_name"]) for row in normal_outside)
        notes.append(
            f"已知正常硅片{names}被判槽外；Task14不会通过修改槽尺寸或槽边界"
            "来消除该状态，应检查托盘位姿、投影和硅片真实位置。"
        )

    normal_stacked = [
        row
        for row in expected_rows
        if row.get("representative_state") in {"stacked", "stacked_outside_slot"}
    ]
    stacked_flags = set().union(
        *(
            _target_wafer_flags(str(row["target_name"]), records)
            for row in normal_stacked
        ),
        set(),
    )
    if normal_stacked:
        names = "、".join(str(row["target_name"]) for row in normal_stacked)
        if "l_shaped_overlap_corner" in stacked_flags:
            suggest(
                "stacked_l_min_leg_ratio",
                round(min(0.35, float(wafer["stacked_l_min_leg_ratio"]) + 0.03), 3),
                f"已知正常硅片{names}由L形证据误判叠片；要求更长的两条边确认第二硅片。",
            )
            suggest(
                "stacked_l_angle_tolerance_deg",
                round(max(10.0, float(wafer["stacked_l_angle_tolerance_deg"]) - 3.0), 1),
                f"已知正常硅片{names}由反光边形成近似L角；收紧直角容差。",
            )
        if "second_quadrilateral" in stacked_flags:
            suggest(
                "stacked_second_quadrilateral_ratio",
                round(
                    min(
                        0.35,
                        float(wafer["stacked_second_quadrilateral_ratio"]) + 0.04,
                    ),
                    3,
                ),
                f"已知正常硅片{names}出现第二四边形误判；提高第二轮廓最小面积比例。",
            )
            suggest(
                "stacked_quadrilateral_min_rectangularity",
                round(
                    min(
                        0.90,
                        float(wafer["stacked_quadrilateral_min_rectangularity"]) + 0.03,
                    ),
                    3,
                ),
                f"已知正常硅片{names}出现反光四边形；要求第二轮廓更接近真实矩形。",
            )
        if not stacked_flags & {"l_shaped_overlap_corner", "second_quadrilateral"}:
            notes.append(
                f"{names}显示叠片，但记录中缺少L角或第二四边形来源；建议不改变叠片门槛并复核图像。"
            )

    non_detector_states = [
        row
        for row in expected_rows
        if row.get("representative_state") in {"out_of_view", "occluded", "unavailable"}
    ]
    if non_detector_states:
        names = "、".join(str(row["target_name"]) for row in non_detector_states)
        notes.append(
            f"{names}属于画面/位姿/采集不可用，不能通过放宽硅片参数修复。"
        )
    if not changes:
        notes.append("没有足够证据支持安全改参；建议JSON保留当前检测数值。")
    return finish()


def _slot_list(names: Sequence[str]) -> str:
    return "、".join(names) if names else "无"


_EXCLUSION_LABELS = {
    "capture_target_tool_occlusion": "正对槽吸盘遮挡高风险",
    "out_of_view": "画面外",
    "occluded": "视觉判定遮挡",
    "unknown": "证据不足",
    "marker_unread": "二维码未识别",
    "unavailable": "位姿/跟踪不可用",
    "processing_error": "处理异常",
}


def _exclusion_text(row: Mapping[str, Any]) -> str:
    counts = row.get("exclusion_counts", {})
    if not isinstance(counts, Mapping) or not counts:
        return "无"
    return "；".join(
        f"{_EXCLUSION_LABELS.get(str(reason), str(reason))}{int(count)}"
        for reason, count in counts.items()
    )


def _state_count(row: Mapping[str, Any], state: str) -> int:
    counts = row.get("state_counts", {})
    return int(counts.get(state, 0)) if isinstance(counts, Mapping) else 0


def _markdown_summary(
    report: Mapping[str, Any],
    changes: Sequence[Mapping[str, Any]],
    notes: Sequence[str],
) -> str:
    summaries = report["slots"]
    total_frames = int(report["summary"]["processed_frame_count"])
    normal_rows = [
        row for row in summaries if _row_expected_label(row) == "normal_wafer"
    ]
    outside_rows = [
        row for row in summaries if _row_expected_label(row) == "outside_wafer"
    ]
    empty_rows = [row for row in summaries if _row_expected_label(row) == "empty"]
    lines = [
        "# Task14 硅片检测全视角统计报告",
        "",
        f"- 运行文件夹：`{report['source_run_folder']}`",
        "- 采集：自动曝光；36个机械臂观察点，"
        f"每点{report['scan_configuration']['frames_per_slot']}张，共{total_frames}张原图。",
        f"- 自动曝光验证：{'通过' if report['camera']['exposure']['verified_before_motion'] else '未通过'}。",
        "- 帧定位：严格PnP分析"
        f"{report['summary']['strict_pnp_analysis_frame_count']}张；"
        "只读平面降级"
        f"{report['summary']['planar_fallback_analysis_frame_count']}张；"
        "真正不可用"
        f"{report['summary']['unavailable_analysis_frame_count']}张。",
        "- 安全隔离：只读平面降级帧的 `coordinate_mapping_allowed=false`、"
        "`robot_correction_allowed=false`；不用于标定、运动、拾取锁定或世界坐标计算。",
        f"- 统计口径：每一个槽都使用全部{total_frames}张照片进行分析，而不是只使用机械臂停在该槽上方的5张。",
        "- 排除规则：机械臂正对该槽拍摄的帧按吸盘遮挡高风险排除；同时排除画面外、显式遮挡、"
        "证据不足、二维码未识别、位姿/跟踪不可用和处理异常帧。",
        f"- 总槽观测数：{report['summary']['total_slot_observation_count']}；"
        f"有效{report['summary']['valid_slot_observation_count']}；"
        f"排除{report['summary']['excluded_slot_observation_count']}。",
        "- 根目录 `1_XXX.jpg` 是标注图；离线复算读取 `raw_task14/` 中的无标记原图。",
        "- 本次只生成报告和建议配置；不会自动修改正式配置，也不会根据视觉结果控制机械臂。",
        "- 槽尺寸、槽图定义和槽边界保持锁定。",
        "",
        f"## 预期有正常硅片的{len(normal_rows)}槽",
        "",
        "| 槽位 | 有效/总帧 | 正常 | 警告 | 叠片 | 槽外 | 叠片且槽外 | 被判空槽 | 排除明细 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in normal_rows:
        empty_count = _state_count(row, "empty")
        lines.append(
            f"| {row['target_name']} | {row['valid_observation_count']}/{total_frames} | "
            f"{_state_count(row, 'occupied')} | {_state_count(row, 'warning')} | "
            f"{_state_count(row, 'stacked')} | {_state_count(row, 'outside_slot')} | "
            f"{_state_count(row, 'stacked_outside_slot')} | {empty_count} | "
            f"{_exclusion_text(row)} |"
        )

    lines.extend(
        [
            "",
            f"## 预期有槽外硅片的{len(outside_rows)}槽",
            "",
            "| 槽位 | 有效/总帧 | 槽外 | 叠片且槽外 | 正常 | 警告 | 叠片 | 被判空槽 | 排除明细 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in outside_rows:
        lines.append(
            f"| {row['target_name']} | {row['valid_observation_count']}/{total_frames} | "
            f"{_state_count(row, 'outside_slot')} | "
            f"{_state_count(row, 'stacked_outside_slot')} | "
            f"{_state_count(row, 'occupied')} | {_state_count(row, 'warning')} | "
            f"{_state_count(row, 'stacked')} | {_state_count(row, 'empty')} | "
            f"{_exclusion_text(row)} |"
        )

    lines.extend(
        [
            "",
            f"## 预期为空槽的{len(empty_rows)}槽",
            "",
            "| 槽位 | 有效/总帧 | 空槽 | 其他合计 | 被判正常硅片 | 警告 | 叠片 | 槽外 | 叠片且槽外 | 排除明细 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in empty_rows:
        other_total = sum(
            int(value) for value in row.get("other_state_counts", {}).values()
        )
        lines.append(
            f"| {row['target_name']} | {row['valid_observation_count']}/{total_frames} | "
            f"{_state_count(row, 'empty')} | {other_total} | "
            f"{_state_count(row, 'occupied')} | "
            f"{_state_count(row, 'warning')} | {_state_count(row, 'stacked')} | "
            f"{_state_count(row, 'outside_slot')} | "
            f"{_state_count(row, 'stacked_outside_slot')} | {_exclusion_text(row)} |"
        )

    lines.extend(
        [
            "",
            "## 五帧停留组边界一致性",
            "",
            "- 槽外确认要求同一停留组至少2帧具有强槽外证据；槽内确认要求至少3帧确定槽内且没有强槽外帧。",
            "- 下表只汇总预期有硅片的槽；每帧原始状态和证据仍完整保留在JSON中。",
            "",
            "| 槽位 | 组判槽外 | 组判槽内 | 组判警告 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in normal_rows + outside_rows:
        group_counts = row.get("five_frame_group_consensus_counts", {})
        if not isinstance(group_counts, Mapping):
            group_counts = {}
        lines.append(
            f"| {row['target_name']} | {int(group_counts.get('outside_slot', 0))} | "
            f"{int(group_counts.get('occupied', 0))} | "
            f"{int(group_counts.get('warning', 0))} |"
        )

    lines.extend(["", "## 建议参数变更", ""])
    if changes:
        for change in changes:
            old_text = json.dumps(change["old"], ensure_ascii=False)
            new_text = json.dumps(change["new"], ensure_ascii=False)
            lines.append(
                f"- `{change['field']}`：`{old_text}` → `{new_text}`。"
                f"原因：{change['reason']}"
            )
    else:
        lines.append("- 无。建议JSON与本次输入配置数值相同。")
    for note in notes:
        lines.append(f"- 说明：{note}")
    lines.extend(
        [
            "",
            "建议值已写入同文件夹的 `silicon_detection_recommended.json`。",
            "该文件不会自动应用；复核标注图后，需在手眼UI中手动选择才会生效。",
            "",
        ]
    )
    return "\n".join(lines)


class Task14SiliconDetectionRuntime(QObject):
    """Analyze Task14 photos and write one authoritative JSON report."""

    def __init__(
        self,
        output_dir: Path,
        project_root: Path,
        target_order: Sequence[str],
        slot_points_T_mm: Mapping[str, Sequence[float]],
        expected_normal_wafer_slots: Sequence[str],
        frames_per_slot: int,
        exposure_mode: str,
        parent: Optional[QWidget] = None,
        *,
        expected_outside_wafer_slots: Sequence[str] = (),
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
        self.expected_normal_slots = frozenset(str(value) for value in expected_normal_wafer_slots)
        self.expected_outside_slots = frozenset(
            str(value) for value in expected_outside_wafer_slots
        )
        self.expected_wafer_slots = (
            self.expected_normal_slots | self.expected_outside_slots
        )
        self.frames_per_slot = int(frames_per_slot)
        self.exposure_mode = str(exposure_mode).strip().lower()
        all_slots = {f"P{row}{column}" for row in range(6) for column in range(6)}
        if len(self.target_order) != 36 or set(self.target_order) != all_slots:
            raise ValueError("Task14 target_order必须恰好包含P00-P55全部36槽")
        if set(self.slot_points) != all_slots:
            raise ValueError("Task14 Tray几何必须恰好包含P00-P55全部36槽")
        if not self.expected_normal_slots or not self.expected_normal_slots <= all_slots:
            raise ValueError("Task14预期正常硅片槽集合无效")
        if not self.expected_outside_slots <= all_slots:
            raise ValueError("Task14预期槽外硅片槽集合无效")
        if self.expected_normal_slots & self.expected_outside_slots:
            raise ValueError("Task14正常槽与槽外槽不能重叠")
        if self.frames_per_slot < 1:
            raise ValueError("Task14 frames_per_slot必须是正整数")
        if self.exposure_mode != "auto":
            raise ValueError("本次Task14数据集固定要求相机1使用自动曝光")

        self.intrinsics_path = self.project_root / "src/scara/calib/camera1_intrinsics.json"
        self.geometry_path = self.project_root / "src/scara/calib/tray_board_geometry.json"
        self.layout_path = (
            self.project_root / "tools/tray_marker_detector_v2/tray_marker_layout.json"
        )
        self.silicon_config_path = default_silicon_detection_config_path(
            self.project_root
        )
        self.silicon_detection_config = load_silicon_detection_config(
            self.silicon_config_path
        )
        self.intrinsics = load_camera_intrinsics(self.intrinsics_path)
        self.geometry = load_tray_board_geometry(self.geometry_path)
        self.estimator = TrayBoardPoseEstimator(self.geometry, self.intrinsics)
        self.analyzer = TrayVisionAnalyzer(
            self.estimator,
            self.geometry,
            load_slot_marker_layout(self.layout_path),
            config=self.silicon_detection_config.fusion_config,
        )
        self._tracker: Optional[TrayPoseTracker] = None
        self._active_target: Optional[str] = None
        self._records_by_filename: dict[str, dict[str, Any]] = {}
        self._last_live_sequence = 0

        if confirm_safety:
            normal_slots_text = " ".join(sorted(self.expected_normal_slots))
            outside_slots_text = (
                " ".join(sorted(self.expected_outside_slots)) or "无"
            )
            empty_slot_count = len(
                all_slots - self.expected_wafer_slots
            )
            confirmed = ask_light_warning_confirmation(
                parent,
                "Task14 硅片检测扫描安全确认",
                "开始前请确认：\n\n"
                "1. 机械臂已到达P00 float固定观察高度，全盘上方路径无障碍；\n"
                "2. 真空关闭、吸盘不携带硅片、急停可用；\n"
                f"3. 正常槽内硅片位于：{normal_slots_text}；\n"
                f"4. 槽外硅片位于：{outside_slots_text}；\n"
                f"5. 其余{empty_slot_count}槽为空；\n"
                "6. 相机1为1280×720，任务将在运动前开启自动曝光并验证模式。\n\n"
                f"Task14将扫描36槽，每槽拍{self.frames_per_slot}张，结束返回P00。"
                "不会下降Z、触发DO/真空或执行视觉修正。是否继续？",
            )
            if not confirmed:
                raise RuntimeError("用户取消：Task14安全条件尚未确认")

    def _expected_label(self, slot_name: str) -> str:
        if slot_name in self.expected_normal_slots:
            return "normal_wafer"
        if slot_name in self.expected_outside_slots:
            return "outside_wafer"
        return "empty"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "points.json"

    def _expected_from_sequence(self, sequence: int) -> tuple[str, int]:
        total = len(self.target_order) * self.frames_per_slot
        if sequence < 1 or sequence > total:
            raise ValueError(f"Task14相机1照片序号{sequence}超出1-{total}")
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
        record: dict[str, Any] = {
            "filename": path.name,
            "saved_photo_kind": "tray_vision_annotated",
            "raw_source_filename": f"{RAW_DIRECTORY}/{path.name}",
            "point_sequence": int(point_sequence),
            "target_name": str(target_name),
            "frame_index": int(frame_index),
            "expected_occupied": target_name in self.expected_wafer_slots,
            "expected_label": self._expected_label(target_name),
            "known_slot_center_T_mm": list(self.slot_points[target_name]),
            "stage3": None,
            "temporal_quality": None,
            "analysis_quality_passed": False,
            "projection_source": "unavailable",
            "planar_registration": None,
            "coordinate_mapping_allowed": False,
            "robot_correction_allowed": False,
            "tray_vision_summary": None,
            "slot_results": {},
            "target_slot": None,
            "observed_state": "unavailable",
            "processing_error": None,
        }
        try:
            raw_dir = self.output_dir / RAW_DIRECTORY
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / path.name
            if not raw_path.exists():
                # Preserve the exact camera JPEG before replacing the public
                # 1_XXX photo with its TrayVision display version.
                shutil.copy2(path, raw_path)
            image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("无法读取已保存的相机1照片")
            tracked = tracker.update(image)
            result = self.analyzer.analyze_tracked(image, tracked)
            record["stage3"] = tracked.raw.to_json()
            record["temporal_quality"] = {
                "accepted_by_tracker": bool(tracked.accepted_by_tracker),
                "tracker_reason": tracked.tracker_reason,
                "translation_jump_mm": tracked.translation_jump_mm,
                "rotation_jump_deg": tracked.rotation_jump_deg,
                "lost_frame_count": tracked.lost_frame_count,
            }
            strict_quality_passed = bool(
                getattr(
                    result,
                    "quality_passed",
                    record["stage3"].get("quality_passed")
                    and tracked.accepted_by_tracker,
                )
            )
            record["tray_vision_summary"] = dict(result.summary)
            record["analysis_quality_passed"] = bool(
                getattr(result, "analysis_quality_passed", strict_quality_passed)
            )
            record["projection_source"] = str(
                getattr(
                    result,
                    "projection_source",
                    "strict_pnp" if strict_quality_passed else "unavailable",
                )
            )
            record["planar_registration"] = (
                None
                if getattr(result, "planar_registration", None) is None
                else result.planar_registration.to_json()
            )
            record["coordinate_mapping_allowed"] = bool(
                getattr(result, "coordinate_mapping_allowed", strict_quality_passed)
            )
            record["robot_correction_allowed"] = bool(
                getattr(result, "robot_correction_allowed", False)
            )
            record["slot_results"] = {
                slot.projection.slot_key: slot.to_json() for slot in result.slots
            }
            target_slot_json = record["slot_results"].get(target_name)
            if isinstance(target_slot_json, Mapping):
                record["target_slot"] = target_slot_json
                target_slot = next(
                    (
                        slot
                        for slot in result.slots
                        if slot.projection.slot_key == target_name
                    ),
                    None,
                )
                assert target_slot is not None
                record["observed_state"] = target_slot.decision.state.value
            annotated = result.annotated_image.copy()
            state = str(record["observed_state"])
            expected_states = _EXPECTED_STATES[
                self._expected_label(target_name)
            ]
            color = (0, 170, 0) if state in expected_states else (0, 0, 255)
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 42), (255, 255, 255), -1)
            cv2.putText(
                annotated,
                f"TASK14 {target_name} frame {frame_index:02d}/{self.frames_per_slot:02d} "
                f"state={state} projection={record['projection_source']} "
                f"exposure={self.exposure_mode}",
                (12, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                color,
                2,
                cv2.LINE_AA,
            )
            annotated_dir = self.output_dir / ANNOTATED_DIRECTORY
            annotated_dir.mkdir(parents=True, exist_ok=True)
            _write_jpeg_atomically(annotated_dir / path.name, annotated)
            # Keep the ActionWorker/points.json naming contract unchanged:
            # root-level 1_XXX.jpg is now the image users open and already
            # contains the same wafer/slot marks shown by the dynamic UI.
            _write_jpeg_atomically(path, annotated)
        except Exception as exc:
            record["processing_error"] = str(exc)
        return record

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
            self._records_by_filename[path.name] = self._process_one(
                path,
                target_name,
                frame_index,
                sequence,
                self._tracker,
            )
            self._last_live_sequence = sequence
        except Exception as exc:
            self._records_by_filename[path.name] = {
                "filename": path.name,
                "point_sequence": sequence,
                "target_name": None,
                "frame_index": None,
                "observed_state": "unavailable",
                "processing_error": str(exc),
            }

    def _manifest_contexts(self, manifest: Mapping[str, Any]) -> list[tuple[Path, str, int, int]]:
        points = {
            int(row.get("sequence", -1)): row
            for row in manifest.get("points", [])
            if isinstance(row, Mapping)
        }
        contexts = []
        for photo in manifest.get("photos", []):
            if not isinstance(photo, Mapping) or int(photo.get("source", -1)) != 1:
                continue
            sequence = int(photo.get("point_sequence", -1))
            point = points.get(sequence)
            if point is None:
                raise RuntimeError(f"points.json缺少照片对应路径点 {sequence}")
            target, frame_index, frame_total = parse_task14_point_name(str(point.get("name")))
            if frame_total != self.frames_per_slot:
                raise RuntimeError("points.json中的Task14每槽帧数与运行时不一致")
            contexts.append(
                (self.output_dir / str(photo.get("filename")), target, frame_index, sequence)
            )
        return sorted(contexts, key=lambda row: row[3])

    def _reprocess_incomplete_targets(
        self,
        contexts: Sequence[tuple[Path, str, int, int]],
    ) -> None:
        for target_name in self.target_order:
            authoritative = [row for row in contexts if row[1] == target_name]
            expected_names = {row[0].name for row in authoritative}
            existing_names = {
                str(row.get("filename"))
                for row in self._records_by_filename.values()
                if row.get("target_name") == target_name
            }
            schema_complete = all(
                "slot_results" in row
                for row in self._records_by_filename.values()
                if row.get("target_name") == target_name
            )
            if (
                len(authoritative) == self.frames_per_slot
                and existing_names == expected_names
                and schema_complete
            ):
                continue
            for filename in existing_names:
                self._records_by_filename.pop(filename, None)
            tracker = TrayPoseTracker(self.estimator)
            for path, target, frame_index, sequence in sorted(authoritative, key=lambda row: row[2]):
                self._records_by_filename[path.name] = self._process_one(
                    path, target, frame_index, sequence, tracker
                )

    def _exposure_evidence(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        requested = (
            manifest.get("camera_capture_settings_requested", {})
            .get("1", {})
            .get("auto_exposure")
        )
        applied = manifest.get("camera_capture_settings_applied", {}).get("1", {})
        applied_values = applied.get("applied") or {}
        settled_mode = applied_values.get("auto_mode_settled_readback")
        settled_exposure = applied_values.get("exposure_settled_readback")
        immediate_effective = applied_values.get("auto_mode_effective")
        if immediate_effective is None:
            immediate_effective = applied_values.get("auto_mode_confirmed")
        settled_effective = applied_values.get("auto_mode_settled_effective")
        if settled_effective is None:
            settled_effective = applied_values.get("auto_mode_settled_confirmed")
        verified = bool(
            requested is True
            and immediate_effective is True
            and settled_effective is True
        )
        return {
            "required_mode": self.exposure_mode,
            "requested_auto_exposure": requested,
            "settled_auto_mode_readback": settled_mode,
            "settled_exposure_readback": settled_exposure,
            "auto_mode_readback_confirmed": bool(
                applied_values.get("auto_mode_confirmed") is True
                and applied_values.get("auto_mode_settled_confirmed") is True
            ),
            "auto_mode_readback_is_advisory": bool(
                applied_values.get("auto_mode_readback_is_advisory") is True
                or applied_values.get("auto_mode_settled_readback_is_advisory")
                is True
            ),
            "verified_before_motion": verified,
            "action_worker_evidence": applied or None,
        }

    def _enrich_manifest(self, manifest: dict[str, Any], report: Mapping[str, Any]) -> None:
        by_sequence = {
            int(row["point_sequence"]): row
            for row in self._records_by_filename.values()
            if row.get("point_sequence") is not None
        }
        for point in manifest.get("points", []):
            record = by_sequence.get(int(point.get("sequence", -1)))
            if record is None:
                continue
            point["task14_target_name"] = record.get("target_name")
            point["task14_frame_index"] = record.get("frame_index")
            point["task14_observed_state"] = record.get("observed_state")
            point["task14_processing_error"] = record.get("processing_error")
        manifest["task14_silicon_detection"] = {
            "status": report["status"],
            "result_file": RESULT_FILENAME,
            "summary_file": SUMMARY_FILENAME,
            "recommended_config_file": RECOMMENDED_CONFIG_FILENAME,
            "annotated_directory": ANNOTATED_DIRECTORY,
            "root_photos_are_annotated": True,
            "raw_directory": RAW_DIRECTORY,
            "all_expected_normal_wafers_acceptable": report["summary"][
                "all_expected_normal_wafers_acceptable"
            ],
            "all_expected_outside_wafers_acceptable": report["summary"][
                "all_expected_outside_wafers_acceptable"
            ],
        }

    @pyqtSlot(bool, str, str)
    def on_task_finished(self, ok: bool, message: str, output_dir_text: str) -> None:
        try:
            if Path(output_dir_text).resolve() != self.output_dir.resolve():
                raise RuntimeError("Task14运行时输出文件夹不一致")
            manifest = json.loads(read_text_snapshot(self.manifest_path, encoding="utf-8-sig"))
            contexts = self._manifest_contexts(manifest)
            self._reprocess_incomplete_targets(contexts)
            records = sorted(
                self._records_by_filename.values(),
                key=lambda row: int(row.get("point_sequence") or 0),
            )
            expected_total_frames = len(self.target_order) * self.frames_per_slot
            summaries = [
                summarize_task14_slot(
                    target,
                    self.slot_points[target],
                    records,
                    expected_occupied=target in self.expected_wafer_slots,
                    expected_frames=expected_total_frames,
                    expected_label=self._expected_label(target),
                )
                for target in sorted(self.slot_points)
            ]
            normal_pass = [
                row["target_name"]
                for row in summaries
                if _row_expected_label(row) == "normal_wafer"
                and row["baseline_passed"]
            ]
            normal_fail = sorted(self.expected_normal_slots - set(normal_pass))
            outside_pass = [
                row["target_name"]
                for row in summaries
                if _row_expected_label(row) == "outside_wafer"
                and row["baseline_passed"]
            ]
            outside_fail = sorted(
                self.expected_outside_slots - set(outside_pass)
            )
            empty_other_state = [
                row["target_name"]
                for row in summaries
                if _row_expected_label(row) == "empty"
                and row.get("other_state_counts")
            ]
            empty_insufficient = [
                row["target_name"]
                for row in summaries
                if _row_expected_label(row) == "empty"
                and int(row.get("valid_observation_count", 0))
                < MINIMUM_VALID_OBSERVATIONS_PER_SLOT
            ]
            expected_normal_by_state = _group_slots_by_state(
                summaries, expected_label="normal_wafer"
            )
            expected_outside_by_state = _group_slots_by_state(
                summaries, expected_label="outside_wafer"
            )
            expected_empty_by_state = _group_slots_by_state(
                summaries, expected_label="empty"
            )
            expected_wafer_by_state = _group_slots_by_state(
                summaries, expected_occupied=True
            )
            exposure = self._exposure_evidence(manifest)
            frame_registration = _frame_registration_summary(records)
            if not ok:
                status = "acquisition_stopped"
            elif not exposure["verified_before_motion"]:
                status = "invalid_exposure_evidence"
            elif normal_fail or outside_fail:
                status = "tuning_required"
            elif self.expected_outside_slots:
                status = "expected_wafers_acceptable"
            else:
                status = "normal_wafers_acceptable"
            report = {
                "schema_version": 4,
                "status": status,
                "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "message": str(message),
                "purpose": (
                    "Task14 automatic-exposure wafer-detection baseline; diagnostic only and "
                    "never used to command robot motion."
                ),
                "camera": {
                    "logical_name": "camera1_forearm_fixed",
                    "source_index": 1,
                    "resolution": {
                        "width": self.intrinsics.image_size[0],
                        "height": self.intrinsics.image_size[1],
                    },
                    "exposure": exposure,
                },
                "locked_inputs": {
                    "camera_intrinsics_path": str(self.intrinsics_path.resolve()),
                    "camera_intrinsics_sha256": _sha256(self.intrinsics_path),
                    "tray_geometry_path": str(self.geometry_path.resolve()),
                    "tray_geometry_sha256": _sha256(self.geometry_path),
                    "slot_marker_layout_path": str(self.layout_path.resolve()),
                    "slot_marker_layout_sha256": _sha256(self.layout_path),
                    "silicon_detection_config_path": str(
                        self.silicon_detection_config.source_path
                    ),
                    "silicon_detection_config_sha256": (
                        self.silicon_detection_config.source_sha256
                    ),
                    "silicon_detection_profile_name": (
                        self.silicon_detection_config.profile_name
                    ),
                },
                "scan_configuration": {
                    "target_order": list(self.target_order),
                    "slot_count": 36,
                    "frames_per_slot": self.frames_per_slot,
                    "expected_total_frames": expected_total_frames,
                    "source_frames_used_per_slot": expected_total_frames,
                    "capture_target_frames_excluded_per_slot": self.frames_per_slot,
                    "observation_policy": (
                        "analyze_every_slot_in_every_source_frame_then_exclude_capture_target_"
                        "tool_occlusion_out_of_view_occluded_unknown_and_unread_marker"
                    ),
                    "expected_normal_wafer_slots": sorted(self.expected_normal_slots),
                    "expected_outside_wafer_slots": sorted(
                        self.expected_outside_slots
                    ),
                    "expected_empty_slots": sorted(
                        set(self.slot_points) - self.expected_wafer_slots
                    ),
                    "minimum_acceptable_frame_rate": MINIMUM_ACCEPTABLE_FRAME_RATE,
                    "acceptable_normal_states": sorted(_NORMAL_STATES),
                    "acceptable_outside_states": sorted(_OUTSIDE_STATES),
                },
                "tuning_scope": _tuning_scope(
                    self.silicon_detection_config.fusion_config.wafer_quality
                ),
                "summary": {
                    "processed_frame_count": len(records),
                    **frame_registration,
                    "total_slot_observation_count": len(records) * len(self.slot_points),
                    "valid_slot_observation_count": sum(
                        int(row["valid_observation_count"]) for row in summaries
                    ),
                    "excluded_slot_observation_count": sum(
                        int(row["excluded_observation_count"]) for row in summaries
                    ),
                    "processing_error_count": sum(bool(row.get("processing_error")) for row in records),
                    "expected_normal_wafer_count": len(self.expected_normal_slots),
                    "acceptable_normal_wafer_count": len(normal_pass),
                    "acceptable_normal_wafer_slots": sorted(normal_pass),
                    "unacceptable_normal_wafer_slots": normal_fail,
                    "all_expected_normal_wafers_acceptable": not normal_fail,
                    "expected_outside_wafer_count": len(
                        self.expected_outside_slots
                    ),
                    "acceptable_outside_wafer_count": len(outside_pass),
                    "acceptable_outside_wafer_slots": sorted(outside_pass),
                    "unacceptable_outside_wafer_slots": outside_fail,
                    "all_expected_outside_wafers_acceptable": not outside_fail,
                    "empty_slot_with_other_state_slots": sorted(empty_other_state),
                    "empty_slot_insufficient_valid_observation_slots": sorted(
                        empty_insufficient
                    ),
                    "empty_slot_false_positive_or_uncertain_slots": sorted(
                        set(empty_other_state) | set(empty_insufficient)
                    ),
                    "expected_wafer_slots_by_state": expected_wafer_by_state,
                    "expected_normal_wafer_slots_by_state": (
                        expected_normal_by_state
                    ),
                    "expected_outside_wafer_slots_by_state": (
                        expected_outside_by_state
                    ),
                    "expected_empty_slots_by_state": expected_empty_by_state,
                },
                "slots": summaries,
                "frame_records": records,
                "source_run_folder": str(self.output_dir.resolve()),
            }
            base_config_payload = json.loads(
                read_text_snapshot(
                    self.silicon_detection_config.source_path,
                    encoding="utf-8-sig",
                )
            )
            recommended_payload, parameter_changes, recommendation_notes = (
                _recommended_config(
                    base_config_payload,
                    summaries,
                    records,
                    evidence_valid=bool(ok and exposure["verified_before_motion"]),
                    run_name=self.output_dir.name,
                )
            )
            recommended_path = self.output_dir / RECOMMENDED_CONFIG_FILENAME
            report["parameter_recommendation"] = {
                "automatically_applied": False,
                "recommended_config_file": RECOMMENDED_CONFIG_FILENAME,
                "locked_slot_geometry_parameters": list(
                    _LOCKED_SLOT_GEOMETRY_PARAMETERS
                ),
                "changes": parameter_changes,
                "notes": recommendation_notes,
            }
            report["artifacts"] = {
                "json_report": RESULT_FILENAME,
                "markdown_summary": SUMMARY_FILENAME,
                "recommended_config": RECOMMENDED_CONFIG_FILENAME,
                "root_photo_pattern": "1_XXX.jpg (TrayVision annotated)",
                "annotated_directory": ANNOTATED_DIRECTORY,
                "raw_directory": RAW_DIRECTORY,
            }
            self._enrich_manifest(manifest, report)
            atomic_write_text(self.manifest_path, _json_text(manifest))
            atomic_write_text(recommended_path, _json_text(recommended_payload))
            # Reuse the same strict loader as the UI before publishing the report.
            load_silicon_detection_config(recommended_path)
            atomic_write_text(
                self.output_dir / SUMMARY_FILENAME,
                _markdown_summary(report, parameter_changes, recommendation_notes),
            )
            atomic_write_text(self.output_dir / RESULT_FILENAME, _json_text(report))
        except Exception as exc:
            try:
                atomic_write_text(
                    self.output_dir / ERROR_FILENAME,
                    f"{exc}\n\n{traceback.format_exc()}",
                )
            except OSError:
                pass
            raise


def create_task14_silicon_detection_runtime(
    output_dir: Path,
    project_root: Path,
    target_order: Sequence[str],
    slot_points_T_mm: Mapping[str, Sequence[float]],
    expected_normal_wafer_slots: Sequence[str],
    frames_per_slot: int,
    exposure_mode: str,
    parent: Optional[QWidget] = None,
    *,
    expected_outside_wafer_slots: Sequence[str] = (),
) -> Task14SiliconDetectionRuntime:
    return Task14SiliconDetectionRuntime(
        output_dir,
        project_root,
        target_order,
        slot_points_T_mm,
        expected_normal_wafer_slots,
        frames_per_slot,
        exposure_mode,
        parent,
        expected_outside_wafer_slots=expected_outside_wafer_slots,
    )


__all__ = [
    "ANNOTATED_DIRECTORY",
    "RECOMMENDED_CONFIG_FILENAME",
    "RESULT_FILENAME",
    "SUMMARY_FILENAME",
    "Task14SiliconDetectionRuntime",
    "create_task14_silicon_detection_runtime",
    "parse_task14_point_name",
    "summarize_task14_slot",
]
