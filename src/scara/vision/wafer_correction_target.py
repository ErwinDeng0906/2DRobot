"""Pure target selection for outside-slot wafer XY correction.

The functions in this module consume metric Tray-frame observations only.  They
do not read a camera, registration, robot state, or controller, and they never
turn a pixel centre into a motion target.  A five-frame target is frozen from
the arithmetic-mean Tray-frame centre; slot names are retained only as
diagnostic evidence because one physical wafer can be reported by neighbouring
slot patches on different frames.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from .tray_occupancy import SlotState


FULL_CONTOUR_CENTER_SOURCE = (
    "expanded_roi_robust_quadrilateral_diagonal_intersection"
)
LEGACY_FULL_CONTOUR_CENTER_SOURCE = "expanded_roi_full_contour_min_area_rect"
SLOT_QUADRILATERAL_CENTER_SOURCE = (
    "slot_patch_fitted_quadrilateral_diagonal_intersection"
)
ALLOWED_CENTER_SOURCES = frozenset(
    {
        FULL_CONTOUR_CENTER_SOURCE,
        LEGACY_FULL_CONTOUR_CENTER_SOURCE,
        SLOT_QUADRILATERAL_CENTER_SOURCE,
    }
)
FRAME_DEDUPLICATION_RADIUS_MM = 1.0


def _finite_point3(value: Any, label: str) -> tuple[float, float, float]:
    """Return a finite 3-D point or fail closed with a useful Chinese error."""
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label}必须包含3个有限数值")
    try:
        values = tuple(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须包含3个有限数值") from exc
    if len(values) != 3:
        raise ValueError(f"{label}必须包含3个有限数值")
    try:
        point = tuple(float(item) for item in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须包含3个有限数值") from exc
    if not all(math.isfinite(item) for item in point):
        raise ValueError(f"{label}必须包含3个有限数值")
    return point  # type: ignore[return-value]


def _p00_point(geometry: Mapping[str, Any]) -> tuple[float, float, float]:
    if not isinstance(geometry, Mapping):
        raise ValueError("Tray几何必须是mapping")
    slots = geometry.get("slots")
    if not isinstance(slots, Mapping) or "P00" not in slots:
        raise ValueError("Tray几何缺少slots.P00")
    return _finite_point3(slots["P00"], "Tray几何slots.P00")


def _distance_xy(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _finite_point2(value: Any, label: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label}必须包含2个有限数值")
    try:
        values = tuple(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须包含2个有限数值") from exc
    if len(values) != 2:
        raise ValueError(f"{label}必须包含2个有限数值")
    try:
        point = (float(values[0]), float(values[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须包含2个有限数值") from exc
    if not all(math.isfinite(item) for item in point):
        raise ValueError(f"{label}必须包含2个有限数值")
    return point


def _quadrilateral_diagonal_intersection(
    box: Any, label: str
) -> tuple[tuple[float, float], tuple[tuple[float, float], ...]]:
    """Return the intersection of diagonals (0,2) and (1,3).

    OpenCV ``boxPoints`` and its perspective-mapped image quadrilateral are
    cyclically ordered, so opposite vertices form the two physical diagonals.
    """

    if isinstance(box, (str, bytes, Mapping)):
        raise ValueError(f"{label}必须包含4个二维顶点")
    try:
        points = tuple(
            _finite_point2(point, f"{label}顶点") for point in tuple(box)
        )
    except TypeError as exc:
        raise ValueError(f"{label}必须包含4个二维顶点") from exc
    if len(points) != 4:
        raise ValueError(f"{label}必须包含4个二维顶点")

    p0, p1, p2, p3 = points
    r = (p2[0] - p0[0], p2[1] - p0[1])
    s = (p3[0] - p1[0], p3[1] - p1[1])
    denominator = r[0] * s[1] - r[1] * s[0]
    scale = max(
        1.0,
        math.hypot(*r),
        math.hypot(*s),
    )
    if abs(denominator) <= 1e-9 * scale * scale:
        raise ValueError(f"{label}两条对角线平行或退化")
    q_minus_p = (p1[0] - p0[0], p1[1] - p0[1])
    t = (q_minus_p[0] * s[1] - q_minus_p[1] * s[0]) / denominator
    u = (q_minus_p[0] * r[1] - q_minus_p[1] * r[0]) / denominator
    if not (-1e-6 <= t <= 1.0 + 1e-6 and -1e-6 <= u <= 1.0 + 1e-6):
        raise ValueError(f"{label}对角线交点不在四边形内部")
    center = (p0[0] + t * r[0], p0[1] + t * r[1])
    if not all(math.isfinite(value) for value in center):
        raise ValueError(f"{label}对角线交点不是有限数值")
    return (float(center[0]), float(center[1])), points


def _slot_quadrilateral_candidate(
    analysis: Any,
    slot_key: str,
    p00: tuple[float, float, float],
) -> dict[str, Any]:
    """Build the requested simple OUT fallback from the visible fitted box."""

    wafer = getattr(analysis, "wafer", None)
    if getattr(wafer, "found", None) is not True:
        raise ValueError(
            f"槽位{slot_key}已判定OUTSIDE_SLOT，但没有可用硅片四边形"
        )
    diagonal_center, box = _quadrilateral_diagonal_intersection(
        getattr(analysis, "wafer_box_image_px", None),
        f"槽位{slot_key}的wafer_box_image_px",
    )
    reported_center = _finite_point2(
        getattr(analysis, "wafer_center_image_px", None),
        f"槽位{slot_key}的wafer_center_image_px",
    )
    mismatch_px = math.hypot(
        diagonal_center[0] - reported_center[0],
        diagonal_center[1] - reported_center[1],
    )
    if mismatch_px > 1.0:
        raise ValueError(
            f"槽位{slot_key}拟合四边形对角线交点与已映射中心不一致："
            f"{mismatch_px:.3f}px"
        )
    center = _finite_point3(
        getattr(analysis, "wafer_center_T_mm", None),
        f"槽位{slot_key}的wafer_center_T_mm",
    )
    evidence = {
        "success": True,
        "reason": "ok_outside_slot_fitted_quadrilateral",
        "source": SLOT_QUADRILATERAL_CENTER_SOURCE,
        "box_image_px": [list(point) for point in box],
        "diagonal_intersection_image_px": list(diagonal_center),
        "reported_center_image_px": list(reported_center),
        "diagonal_center_mismatch_px": float(mismatch_px),
    }
    return {
        "slot_key": slot_key,
        "center_T_mm": [center[0], center[1], center[2]],
        "center_image_px": [diagonal_center[0], diagonal_center[1]],
        "distance_to_p00_mm": _distance_xy(center, p00),
        "center_source": SLOT_QUADRILATERAL_CENTER_SOURCE,
        # Keep this API key for the existing five-frame evidence pipeline.  It
        # now means successful geometry evidence, not necessarily expanded ROI.
        "refinement": evidence,
    }


def _require_vision_result_gates(tray_result: Any) -> None:
    if tray_result is None:
        raise ValueError("视觉结果为空，拒绝提取硅片纠错候选")
    failed = [
        name
        for name in ("success", "quality_passed", "coordinate_mapping_allowed")
        if getattr(tray_result, name, None) is not True
    ]
    if failed:
        reason = getattr(tray_result, "failure_reason", None)
        detail = "、".join(failed)
        if reason:
            detail += f"；视觉原因：{reason}"
        raise ValueError(f"视觉结果未通过安全门：{detail}")


def _deduplicate_frame_candidates(
    candidates: Sequence[Mapping[str, Any]],
    p00: tuple[float, float, float],
) -> list[dict[str, Any]]:
    """Merge cross-slot reports of one physical wafer in Tray XY."""

    clusters: list[list[Mapping[str, Any]]] = []
    for candidate in sorted(
        candidates,
        key=lambda row: (
            float(row["distance_to_p00_mm"]),
            str(row["slot_key"]),
        ),
    ):
        center = _finite_point3(candidate["center_T_mm"], "帧内候选中心")
        matched: list[Mapping[str, Any]] | None = None
        for cluster in clusters:
            cluster_centers = [
                _finite_point3(row["center_T_mm"], "帧内聚类中心")
                for row in cluster
            ]
            median_xy = (
                statistics.median(point[0] for point in cluster_centers),
                statistics.median(point[1] for point in cluster_centers),
            )
            if math.hypot(
                center[0] - median_xy[0], center[1] - median_xy[1]
            ) <= FRAME_DEDUPLICATION_RADIUS_MM:
                matched = cluster
                break
        if matched is None:
            clusters.append([candidate])
        else:
            matched.append(candidate)

    merged: list[dict[str, Any]] = []
    for cluster in clusters:
        centres = [
            _finite_point3(row["center_T_mm"], "帧内聚类中心")
            for row in cluster
        ]
        center = (
            float(statistics.median(point[0] for point in centres)),
            float(statistics.median(point[1] for point in centres)),
            float(statistics.median(point[2] for point in centres)),
        )
        source_slots = sorted(
            {
                str(slot)
                for row in cluster
                for slot in (
                    row.get("source_slot_keys") or [row["slot_key"]]
                )
            }
        )
        refinements: list[dict[str, Any]] = []
        for row in cluster:
            evidence = row.get("refinement_evidence")
            if (
                isinstance(evidence, Sequence)
                and not isinstance(evidence, (str, bytes, Mapping))
            ):
                refinements.extend(
                    dict(item)
                    for item in evidence
                    if isinstance(item, Mapping)
                )
            elif isinstance(row.get("refinement"), Mapping):
                refinements.append(dict(row["refinement"]))
        center_sources = {
            str(row.get("center_source") or "") for row in cluster
        }
        selected_source = next(
            source
            for source in (
                FULL_CONTOUR_CENTER_SOURCE,
                LEGACY_FULL_CONTOUR_CENTER_SOURCE,
                SLOT_QUADRILATERAL_CENTER_SOURCE,
            )
            if source in center_sources
        )
        selected_refinement = next(
            (
                dict(row["refinement"])
                for row in cluster
                if row.get("center_source") == selected_source
                and isinstance(row.get("refinement"), Mapping)
            ),
            dict(refinements[0]),
        )
        image_centres = [
            _finite_point2(row["center_image_px"], "帧内图像中心")
            for row in cluster
            if row.get("center_image_px") is not None
        ]
        merged_row: dict[str, Any] = (
            {
                "slot_key": source_slots[0],
                "source_slot_keys": source_slots,
                "center_T_mm": [center[0], center[1], center[2]],
                "distance_to_p00_mm": _distance_xy(center, p00),
                "center_source": selected_source,
                "refinement": selected_refinement,
                "refinement_evidence": refinements,
            }
        )
        if image_centres:
            merged_row["center_image_px"] = [
                float(statistics.median(point[0] for point in image_centres)),
                float(statistics.median(point[1] for point in image_centres)),
            ]
        merged.append(merged_row)
    merged.sort(
        key=lambda row: (float(row["distance_to_p00_mm"]), str(row["slot_key"]))
    )
    return merged


def extract_outside_wafer_candidates(
    tray_result: Any,
    geometry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Extract metric, non-stacked outside-wafer candidates from one frame.

    A successful expanded-ROI full contour is preferred.  When the ordinary
    slot classifier has already made the exact ``OUTSIDE_SLOT`` decision, its
    fitted image quadrilateral is also authoritative: the two diagonals are
    intersected and that image centre is paired with the existing homography-
    mapped Tray centre.  Cross-slot UNKNOWN fragments still require expanded-
    ROI proof.  Any ``STACKED_OUTSIDE_SLOT`` record rejects the whole frame.
    """
    p00 = _p00_point(geometry)
    _require_vision_result_gates(tray_result)

    slots = getattr(tray_result, "slots", None)
    if slots is None or isinstance(slots, (str, bytes)):
        raise ValueError("视觉结果缺少有效slots序列")
    try:
        slot_rows = tuple(slots)
    except TypeError as exc:
        raise ValueError("视觉结果缺少有效slots序列") from exc

    candidates: list[dict[str, Any]] = []
    for index, analysis in enumerate(slot_rows, start=1):
        decision = getattr(analysis, "decision", None)
        state = getattr(decision, "state", None)
        if state is SlotState.STACKED_OUTSIDE_SLOT:
            projection = getattr(analysis, "projection", None)
            slot_key = getattr(projection, "slot_key", None)
            source = slot_key if isinstance(slot_key, str) and slot_key else index
            raise ValueError(
                f"槽位{source}存在叠片且槽外硅片；无法证明普通OUT候选是离P00最近目标"
            )
        correction_specific_outside = (
            getattr(analysis, "wafer_correction_outside_slot", None) is True
        )
        legacy_outside = state is SlotState.OUTSIDE_SLOT
        # Cross-slot fragments need the correction-specific proof.  An exact
        # legacy OUTSIDE_SLOT may also use its already-fitted visible
        # quadrilateral, per the operator-requested fallback.
        if not correction_specific_outside and not legacy_outside:
            continue

        projection = getattr(analysis, "projection", None)
        slot_key = getattr(projection, "slot_key", None)
        if not isinstance(slot_key, str) or not slot_key:
            raise ValueError(f"第{index}个槽位的OUTSIDE_SLOT记录缺少slot_key")

        correction_center_valid = (
            getattr(analysis, "wafer_correction_center_valid", None) is True
        )
        refinement = getattr(analysis, "wafer_center_refinement", None)
        if (
            correction_specific_outside
            and correction_center_valid
            and isinstance(refinement, Mapping)
            and refinement.get("success") is True
        ):
            center = _finite_point3(
                getattr(analysis, "wafer_center_T_mm", None),
                f"槽位{slot_key}的wafer_center_T_mm",
            )
            candidate = {
                "slot_key": slot_key,
                "center_T_mm": [center[0], center[1], center[2]],
                "distance_to_p00_mm": _distance_xy(center, p00),
                "center_source": FULL_CONTOUR_CENTER_SOURCE,
                "refinement": dict(refinement),
            }
            center_image = getattr(analysis, "wafer_center_image_px", None)
            if center_image is not None:
                candidate["center_image_px"] = list(
                    _finite_point2(
                        center_image,
                        f"槽位{slot_key}的wafer_center_image_px",
                    )
                )
            candidates.append(candidate)
            continue

        if legacy_outside:
            candidates.append(
                _slot_quadrilateral_candidate(analysis, slot_key, p00)
            )
            continue

        if not correction_center_valid:
            reason = str(
                getattr(
                    analysis,
                    "wafer_correction_center_reason",
                    "missing_full_contour_refinement",
                )
                or "missing_full_contour_refinement"
            )
            raise ValueError(
                f"槽位{slot_key}的完整轮廓中心精修未通过：{reason}"
            )
        if not isinstance(refinement, Mapping) or refinement.get("success") is not True:
            raise ValueError(
                f"槽位{slot_key}缺少成功的完整轮廓中心精修证据"
            )
        raise ValueError(f"槽位{slot_key}的纠错中心证据状态不一致")

    return _deduplicate_frame_candidates(candidates, p00)


def _validated_candidate_list(
    value: Any,
    p00: tuple[float, float, float],
    frame_index: int,
) -> list[dict[str, Any]]:
    """Normalize one already-extracted candidate collection."""
    if isinstance(value, Mapping) and "outside_wafer_candidates" in value:
        value = value["outside_wafer_candidates"]
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"第{frame_index}帧候选必须是序列")
    try:
        raw_rows = tuple(value)
    except TypeError as exc:
        raise ValueError(f"第{frame_index}帧候选必须是序列") from exc

    rows: list[dict[str, Any]] = []
    for candidate_index, raw in enumerate(raw_rows, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"第{frame_index}帧第{candidate_index}个候选必须是mapping"
            )
        slot_key = raw.get("slot_key")
        if not isinstance(slot_key, str) or not slot_key:
            raise ValueError(
                f"第{frame_index}帧第{candidate_index}个候选缺少slot_key"
            )
        center = _finite_point3(
            raw.get("center_T_mm"),
            f"第{frame_index}帧候选{slot_key}的center_T_mm",
        )
        center_source = raw.get("center_source")
        if center_source not in ALLOWED_CENTER_SOURCES:
            raise ValueError(
                f"第{frame_index}帧候选{slot_key}缺少允许的几何center_source"
            )
        refinement = raw.get("refinement")
        if not isinstance(refinement, Mapping) or refinement.get("success") is not True:
            raise ValueError(
                f"第{frame_index}帧候选{slot_key}缺少成功的几何中心refinement证据"
            )
        # Recompute metric distance from authoritative geometry rather than
        # trusting a potentially stale serialized distance.
        normalized = {
            "slot_key": slot_key,
            "center_T_mm": [center[0], center[1], center[2]],
            "distance_to_p00_mm": _distance_xy(center, p00),
            "center_source": str(center_source),
            "refinement": dict(refinement),
        }
        center_image = raw.get("center_image_px")
        if center_source == SLOT_QUADRILATERAL_CENTER_SOURCE:
            if refinement.get("source") != SLOT_QUADRILATERAL_CENTER_SOURCE:
                raise ValueError(
                    f"第{frame_index}帧候选{slot_key}的四边形来源证据无效"
                )
            diagonal_center, _box = _quadrilateral_diagonal_intersection(
                refinement.get("box_image_px"),
                f"第{frame_index}帧候选{slot_key}的box_image_px",
            )
            center_image_point = _finite_point2(
                center_image,
                f"第{frame_index}帧候选{slot_key}的center_image_px",
            )
            if math.hypot(
                diagonal_center[0] - center_image_point[0],
                diagonal_center[1] - center_image_point[1],
            ) > 1e-6:
                raise ValueError(
                    f"第{frame_index}帧候选{slot_key}的四边形交点证据不一致"
                )
            normalized["center_image_px"] = list(center_image_point)
        elif center_image is not None:
            normalized["center_image_px"] = list(
                _finite_point2(
                    center_image,
                    f"第{frame_index}帧候选{slot_key}的center_image_px",
                )
            )
        source_slot_keys = raw.get("source_slot_keys", [slot_key])
        if (
            isinstance(source_slot_keys, (str, bytes, Mapping))
            or not isinstance(source_slot_keys, Sequence)
            or not source_slot_keys
            or not all(
                isinstance(item, str) and bool(item)
                for item in source_slot_keys
            )
        ):
            raise ValueError(
                f"第{frame_index}帧候选{slot_key}的source_slot_keys无效"
            )
        normalized["source_slot_keys"] = sorted(set(source_slot_keys))
        refinement_evidence = raw.get("refinement_evidence")
        if refinement_evidence is not None:
            if (
                isinstance(refinement_evidence, (str, bytes, Mapping))
                or not isinstance(refinement_evidence, Sequence)
                or not refinement_evidence
                or not all(
                    isinstance(item, Mapping)
                    and item.get("success") is True
                    for item in refinement_evidence
                )
            ):
                raise ValueError(
                    f"第{frame_index}帧候选{slot_key}的refinement_evidence无效"
                )
            normalized["refinement_evidence"] = [
                dict(item) for item in refinement_evidence
            ]
        rows.append(normalized)
    return _deduplicate_frame_candidates(rows, p00)


def _candidate_lists(
    frames: Sequence[Any], geometry: Mapping[str, Any]
) -> list[list[dict[str, Any]]]:
    p00 = _p00_point(geometry)
    normalized: list[list[dict[str, Any]]] = []
    for frame_index, frame in enumerate(frames, start=1):
        if hasattr(frame, "slots"):
            rows = extract_outside_wafer_candidates(frame, geometry)
        else:
            rows = _validated_candidate_list(frame, p00, frame_index)
        if not rows:
            raise ValueError(f"第{frame_index}帧没有可用的非叠片槽外硅片候选")
        normalized.append(rows)
    return normalized


def aggregate_nearest_outside_wafer(
    samples_or_candidate_lists: Sequence[Any],
    geometry: Mapping[str, Any],
    *,
    required_frame_count: int = 5,
    maximum_center_residual_mm: float | None = None,
) -> dict[str, Any]:
    """Freeze the nearest-P00 outside wafer at the five-frame arithmetic mean.

    Each frame independently contributes its nearest-P00 candidate.  Those
    five centres are averaged in Tray XYZ.  Residuals remain in the returned
    audit record, but are diagnostic unless a caller explicitly supplies
    ``maximum_center_residual_mm``.  Slot keys are *not* required to agree.
    """
    if isinstance(samples_or_candidate_lists, (str, bytes, Mapping)):
        raise ValueError("硅片纠错样本必须是帧序列")
    try:
        frames = tuple(samples_or_candidate_lists)
    except TypeError as exc:
        raise ValueError("硅片纠错样本必须是帧序列") from exc

    if isinstance(required_frame_count, bool):
        raise ValueError("required_frame_count必须是正整数")
    try:
        normalized_frame_count = int(required_frame_count)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("required_frame_count必须是正整数") from exc
    if normalized_frame_count <= 0 or normalized_frame_count != required_frame_count:
        raise ValueError("required_frame_count必须是正整数")
    required_frame_count = normalized_frame_count
    if len(frames) != required_frame_count:
        raise ValueError(
            f"硅片纠错要求恰好{required_frame_count}帧，实际收到{len(frames)}帧"
        )
    maximum_residual: float | None = None
    if maximum_center_residual_mm is not None:
        try:
            maximum_residual = float(maximum_center_residual_mm)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "maximum_center_residual_mm必须是非负有限数值或None"
            ) from exc
        if not math.isfinite(maximum_residual) or maximum_residual < 0.0:
            raise ValueError(
                "maximum_center_residual_mm必须是非负有限数值或None"
            )

    p00 = _p00_point(geometry)
    candidate_lists = _candidate_lists(frames, geometry)
    selected = [rows[0] for rows in candidate_lists]
    centres = [
        _finite_point3(row["center_T_mm"], f"第{index}帧最近候选中心")
        for index, row in enumerate(selected, start=1)
    ]
    frozen_center = (
        float(statistics.fmean(center[0] for center in centres)),
        float(statistics.fmean(center[1] for center in centres)),
        float(statistics.fmean(center[2] for center in centres)),
    )
    residuals = [
        math.hypot(center[0] - frozen_center[0], center[1] - frozen_center[1])
        for center in centres
    ]
    maximum_observed = max(residuals)
    rms = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    if (
        maximum_residual is not None
        and maximum_observed > maximum_residual
    ):
        raise ValueError(
            "5帧最近槽外硅片中心不稳定："
            f"最大Tray XY残差{maximum_observed:.3f}mm，"
            f"门限{maximum_residual:.3f}mm"
        )

    source_slots = [str(row["slot_key"]) for row in selected]
    source_slots_by_frame = [
        list(row.get("source_slot_keys") or [row["slot_key"]])
        for row in selected
    ]
    center_sources = sorted(
        {
            str(row.get("center_source"))
            for row in selected
            if row.get("center_source")
        }
    )
    return {
        "center_T_mm": [
            frozen_center[0],
            frozen_center[1],
            frozen_center[2],
        ],
        "distance_to_p00_mm": _distance_xy(frozen_center, p00),
        "source_slot_keys": source_slots,
        "source_slot_keys_by_frame": source_slots_by_frame,
        "unique_source_slot_keys": sorted(
            {slot for slots in source_slots_by_frame for slot in slots}
        ),
        "center_source": (
            center_sources[0]
            if len(center_sources) == 1
            else "five_frame_metric_center_arithmetic_mean"
        ),
        "aggregation_method": "five_frame_arithmetic_mean",
        "refinement_evidence": [
            dict(evidence)
            for row in selected
            for evidence in (
                row.get("refinement_evidence")
                or (
                    [row["refinement"]]
                    if isinstance(row.get("refinement"), Mapping)
                    else []
                )
            )
            if isinstance(evidence, Mapping)
        ],
        "selected_frame_candidates": [dict(row) for row in selected],
        "stability": {
            "frame_count": len(selected),
            "residuals_T_xy_mm": [float(value) for value in residuals],
            "maximum_center_residual_mm": float(maximum_observed),
            "rms_center_residual_mm": float(rms),
            "allowed_maximum_center_residual_mm": maximum_residual,
            "residual_gate_enforced": maximum_residual is not None,
        },
    }


__all__ = [
    "ALLOWED_CENTER_SOURCES",
    "FRAME_DEDUPLICATION_RADIUS_MM",
    "FULL_CONTOUR_CENTER_SOURCE",
    "SLOT_QUADRILATERAL_CENTER_SOURCE",
    "aggregate_nearest_outside_wafer",
    "extract_outside_wafer_candidates",
]
