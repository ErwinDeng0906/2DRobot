#!/usr/bin/env python3
"""Run the review-only wafer detector over complete static-tray sequences.

This tool is for dataset auditing. It never creates robot coordinates or
motion permission, and it keeps every per-frame result so predictions are not
silently promoted to ground truth.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.vision.marker_grid_wafer_review import review_marker_grid_image
from scara.vision.silicon_detection_config import (
    default_silicon_detection_config_path,
    load_silicon_detection_config,
)
from scara.vision.slot_marker_observation import load_slot_marker_layout
from scara.vision.tray_pose_estimator import load_tray_board_geometry


IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
GENERATED_NAMES = frozenset({"tray_marker_annotated.png"})


def _raw_images(sequence_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(sequence_dir.iterdir())
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.name not in GENERATED_NAMES
        and "_wafer_review" not in path.stem
    ]


def discover_sequences(roots: Iterable[Path]) -> list[tuple[Path, Path]]:
    sequences: list[tuple[Path, Path]] = []
    for root in roots:
        resolved = Path(root).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"dataset root is not a directory: {resolved}")
        if _raw_images(resolved):
            sequences.append((resolved.parent, resolved))
            continue
        for sequence in sorted(path for path in resolved.iterdir() if path.is_dir()):
            if _raw_images(sequence):
                sequences.append((resolved, sequence))
    return sequences


def _slot_payload(slot: Any) -> dict[str, Any]:
    wafer = slot.wafer
    return {
        "slot_key": slot.slot_key,
        "expected_marker_id": slot.expected_marker_id,
        "marker_visible": slot.marker_visible,
        "state": slot.state,
        "placement_state": slot.placement_state,
        "stacking_state": slot.stacking_state,
        "center_px": list(slot.center_px),
        "polygon_px": [list(point) for point in slot.polygon_px],
        "image_coverage_ratio": slot.image_coverage_ratio,
        "wafer_found": wafer.found,
        "wafer_quality": wafer.quality,
        "confidence": wafer.confidence,
        "area_ratio": wafer.area_ratio,
        "side_ratio": wafer.side_ratio,
        "center_offset_ratio": wafer.center_offset_ratio,
        "yaw_relative_to_tray_deg": wafer.yaw_relative_to_tray_deg,
        "aspect_ratio": wafer.aspect_ratio,
        "rectangularity": wafer.rectangularity,
        "solidity": wafer.solidity,
        "polygon_vertices": wafer.polygon_vertices,
        "component_count": wafer.component_count,
        "second_component_area_ratio": wafer.second_component_area_ratio,
        "internal_line_count": wafer.internal_line_count,
        "internal_line_score": wafer.internal_line_score,
        "chromatic_fraction": wafer.chromatic_fraction,
        "minimum_slot_clearance_ratio": wafer.minimum_slot_clearance_ratio,
        "flags": list(wafer.flags),
    }


def _finite_number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _placement_consensus(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine repeated views without turning weak evidence into certainty."""

    observable = [
        item for item in evidence
        if float(item["image_coverage_ratio"]) >= 0.90
    ]
    boundary_samples: list[tuple[float, str]] = []
    for item in observable:
        clearance = _finite_number(item.get("minimum_slot_clearance_ratio"))
        side_ratio = _finite_number(item.get("side_ratio"))
        aspect_ratio = _finite_number(item.get("aspect_ratio"))
        if (
            not bool(item.get("wafer_found"))
            or clearance is None
            or side_ratio is None
            or aspect_ratio is None
            or side_ratio > 0.62
            or aspect_ratio > 1.60
        ):
            continue
        flags = set(item.get("flags", ()))
        if flags & {
            "candidate_area_out_of_range",
            "duplicate_candidate_suppressed",
            "oversize_footprint",
            "boundary_crossing_unconfirmed",
            "boundary_geometry_extrapolated",
            "boundary_clearance_uncertain",
            "boundary_fallback_geometry_unconfirmed",
        }:
            continue
        boundary_samples.append((clearance, str(item["file"])))

    clearances = np.asarray(
        [sample[0] for sample in boundary_samples], dtype=np.float64
    )
    inside_count = int(np.count_nonzero(clearances >= 0.0))
    outside_count = int(np.count_nonzero(clearances < -0.025))
    boundary_band_count = int(len(clearances) - inside_count - outside_count)
    median_clearance = (
        None if len(clearances) == 0 else float(np.median(clearances))
    )
    decision = "unknown"
    decision_reason = "insufficient_repeated_boundary_evidence"
    confidence = 0.0
    if len(clearances) >= 2:
        support_required = max(2, int(math.ceil(0.60 * len(clearances))))
        if (
            outside_count >= support_required
            and median_clearance is not None
            and median_clearance < -0.025
        ):
            decision = "outside"
            decision_reason = "repeated_negative_clearance"
            confidence = outside_count / len(clearances)
        elif (
            inside_count >= support_required
            and median_clearance is not None
            and median_clearance >= 0.0
        ):
            decision = "inside"
            decision_reason = "repeated_nonnegative_clearance"
            confidence = inside_count / len(clearances)

    empty_frames = [
        str(item["file"])
        for item in observable
        if bool(item.get("marker_visible")) and not bool(item.get("wafer_found"))
    ]
    found_frames = [
        str(item["file"])
        for item in observable
        if bool(item.get("wafer_found"))
    ]
    if decision == "unknown" and len(empty_frames) >= 2 and not found_frames:
        decision = "empty"
        decision_reason = "marker_repeatedly_visible_without_wafer"
        confidence = len(empty_frames) / max(len(observable), 1)

    return {
        "decision": decision,
        "decision_reason": decision_reason,
        "confidence": float(confidence),
        "observable_frame_count": len(observable),
        "boundary_sample_count": len(boundary_samples),
        "inside_evidence_count": inside_count,
        "outside_evidence_count": outside_count,
        "boundary_band_count": boundary_band_count,
        "empty_marker_evidence_count": len(empty_frames),
        "wafer_found_frame_count": len(found_frames),
        "median_clearance_ratio": median_clearance,
        "boundary_evidence_files": [sample[1] for sample in boundary_samples],
        "empty_marker_evidence_files": empty_frames,
    }


def _stacking_consensus(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    observable = [
        item for item in evidence
        if float(item["image_coverage_ratio"]) >= 0.90
    ]
    confirmed = [
        str(item["file"])
        for item in observable
        if item.get("stacking_state") == "confirmed"
    ]
    suspected = [
        str(item["file"])
        for item in observable
        if item.get("stacking_state") == "suspected"
    ]
    single = [
        str(item["file"])
        for item in observable
        if item.get("stacking_state") == "single"
    ]
    found = [item for item in observable if bool(item.get("wafer_found"))]
    repeated_square_envelope = [
        str(item["file"])
        for item in found
        if float(item.get("side_ratio") or 0.0) > 0.62
        and float(item.get("side_ratio") or 0.0) <= 0.72
        and float(item.get("center_offset_ratio") or 0.0) <= 0.14
        and float(item.get("rectangularity") or 0.0) >= 0.97
        and float(item.get("solidity") or 0.0) >= 0.90
        and "marker_center_fallback_geometry"
        not in set(item.get("flags", ()))
        and "sparse_three_marker_geometry"
        not in set(item.get("flags", ()))
        and "sparse_two_outer_marker_geometry"
        not in set(item.get("flags", ()))
    ]
    envelope_fraction = len(repeated_square_envelope) / max(len(found), 1)
    decision = "unknown"
    reason = "insufficient_repeated_stacking_evidence"
    confidence = 0.0
    if (
        len(repeated_square_envelope) >= 3
        and envelope_fraction >= 0.35
    ):
        decision = "confirmed"
        reason = "repeatable_oversized_square_envelope"
        confidence = envelope_fraction
    elif len(confirmed) >= 2:
        decision = "confirmed"
        reason = "confirmed_geometry_in_multiple_frames"
        confidence = len(confirmed) / max(len(confirmed) + len(single), 1)
    elif len(confirmed) + len(suspected) >= 3 and len(suspected) >= 2:
        decision = "suspected"
        reason = "repeated_overlap_evidence_without_repeatable_full_geometry"
        confidence = (len(confirmed) + len(suspected)) / max(
            len(confirmed) + len(suspected) + len(single), 1
        )
    elif len(single) >= 2 and not confirmed:
        decision = "single"
        reason = "repeated_single_layer_geometry"
        confidence = len(single) / max(len(single) + len(suspected), 1)
    return {
        "decision": decision,
        "decision_reason": reason,
        "confidence": float(confidence),
        "confirmed_frame_count": len(confirmed),
        "suspected_frame_count": len(suspected),
        "single_frame_count": len(single),
        "repeatable_square_envelope_frame_count": len(
            repeated_square_envelope
        ),
        "repeatable_square_envelope_fraction": float(envelope_fraction),
        "confirmed_files": confirmed,
        "suspected_files": suspected,
        "repeatable_square_envelope_files": repeated_square_envelope,
    }


def _aggregate_sequence(frames: list[dict[str, Any]]) -> dict[str, Any]:
    slots: dict[str, dict[str, Any]] = {}
    state_counts: dict[str, Counter[str]] = defaultdict(Counter)
    placement_counts: dict[str, Counter[str]] = defaultdict(Counter)
    stacking_counts: dict[str, Counter[str]] = defaultdict(Counter)
    marker_visible_counts: Counter[str] = Counter()
    wafer_found_counts: Counter[str] = Counter()
    observable_counts: Counter[str] = Counter()
    evidence_by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        if not frame["success"]:
            continue
        for slot in frame["slots"]:
            key = str(slot["slot_key"])
            item = dict(slot)
            item["file"] = str(frame["file"])
            evidence_by_slot[key].append(item)
            state_counts[key][str(slot["state"])] += 1
            placement_counts[key][str(slot["placement_state"])] += 1
            stacking_counts[key][str(slot["stacking_state"])] += 1
            if float(slot["image_coverage_ratio"]) >= 0.90:
                observable_counts[key] += 1
            if bool(slot["marker_visible"]):
                marker_visible_counts[key] += 1
            if bool(slot["wafer_found"]):
                wafer_found_counts[key] += 1
    for key in sorted(state_counts):
        slots[key] = {
            "observable_frames": int(observable_counts[key]),
            "marker_visible_frames": int(marker_visible_counts[key]),
            "wafer_found_frames": int(wafer_found_counts[key]),
            "state_counts": dict(state_counts[key]),
            "placement_counts": dict(placement_counts[key]),
            "stacking_counts": dict(stacking_counts[key]),
            "placement_consensus": _placement_consensus(
                evidence_by_slot[key]
            ),
            "stacking_consensus": _stacking_consensus(
                evidence_by_slot[key]
            ),
        }
    return {
        "image_count": len(frames),
        "successful_frame_count": sum(bool(frame["success"]) for frame in frames),
        "failed_frame_count": sum(not bool(frame["success"]) for frame in frames),
        "slots": slots,
    }


def evaluate(
    roots: Iterable[Path],
    *,
    geometry_path: Path,
    layout_path: Path,
    silicon_config_path: Path,
    annotated_dir: Path | None = None,
) -> dict[str, Any]:
    geometry = load_tray_board_geometry(geometry_path)
    layout = load_slot_marker_layout(layout_path)
    loaded_config = load_silicon_detection_config(silicon_config_path)
    sequences_payload = []
    total_images = 0
    for root, sequence in discover_sequences(roots):
        frames = []
        for image_path in _raw_images(sequence):
            total_images += 1
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                frames.append(
                    {
                        "file": image_path.name,
                        "success": False,
                        "failure_reason": "cannot read image",
                        "fit": None,
                        "summary": {"unknown": 36},
                        "slots": [],
                    }
                )
                continue
            result = review_marker_grid_image(image, geometry, layout, loaded_config.fusion_config)
            frame = {
                "file": image_path.name,
                "success": result.success,
                "failure_reason": result.failure_reason,
                "fit": result.fit.to_json(),
                "summary": dict(result.summary),
                "slots": [_slot_payload(slot) for slot in result.slots],
            }
            frames.append(frame)
            if annotated_dir is not None:
                destination = annotated_dir / root.name / sequence.name / image_path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(destination), result.annotated_image):
                    raise RuntimeError(f"cannot write annotated image: {destination}")
            del result
            del image
            gc.collect()
        sequences_payload.append(
            {
                "dataset_root": str(root),
                "sequence": sequence.name,
                "frames": frames,
                "aggregate": _aggregate_sequence(frames),
            }
        )
    return {
        "schema_version": 1,
        "mode": "review_only_no_robot_coordinates",
        "coordinate_mapping_allowed": False,
        "robot_motion_authorized": False,
        "silicon_detection_profile": loaded_config.profile_name,
        "silicon_detection_sha256": loaded_config.source_sha256,
        "total_image_count": total_images,
        "sequence_count": len(sequences_payload),
        "sequences": sequences_payload,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--annotated-dir", type=Path)
    parser.add_argument(
        "--geometry",
        type=Path,
        default=ROOT / "src/scara/calib/tray_board_geometry.json",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=ROOT / "tools/tray_marker_detector_v2/tray_marker_layout.json",
    )
    parser.add_argument("--silicon-config", type=Path)
    args = parser.parse_args()
    config_path = args.silicon_config or default_silicon_detection_config_path(ROOT)
    payload = evaluate(
        args.roots,
        geometry_path=args.geometry,
        layout_path=args.layout,
        silicon_config_path=config_path,
        annotated_dir=args.annotated_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "total_image_count": payload["total_image_count"],
                "sequence_count": payload["sequence_count"],
                "output": str(args.output),
                "robot_motion_authorized": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
