#!/usr/bin/env python3
"""Compare full per-frame review output with independently reviewed truth.

Only explicitly labelled slots are scored. Missing truth labels are not
silently inferred from folder names. A wrong certain conclusion is reported
separately from a fail-closed unknown/warning result.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _dataset_key(dataset_root: str) -> str:
    return Path(dataset_root).name


def _load_batches(paths: Iterable[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    sequences: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for sequence in payload.get("sequences", []):
            key = (
                _dataset_key(str(sequence["dataset_root"])),
                str(sequence["sequence"]),
            )
            if key in sequences:
                raise ValueError(f"duplicate evaluated sequence: {key}")
            sequences[key] = sequence
    return sequences


def _load_truth(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (str(item["dataset"]), str(item["sequence"])): item
        for item in payload.get("sequences", [])
    }


def _placement_truth(item: dict[str, Any]) -> dict[str, str]:
    truth: dict[str, str] = {}
    for state, slots in item.get("confirmed_placement", {}).items():
        for slot in slots:
            if slot in truth:
                raise ValueError(
                    f"duplicate placement truth for {item['sequence']} {slot}"
                )
            truth[str(slot)] = str(state)
    return truth


def _stacking_truth(item: dict[str, Any]) -> dict[str, str]:
    truth: dict[str, str] = {}
    for state, slots in item.get("confirmed_stacking", {}).items():
        for slot in slots:
            if slot in truth:
                raise ValueError(
                    f"duplicate stacking truth for {item['sequence']} {slot}"
                )
            truth[str(slot)] = str(state)
    return truth


def _placement_outcome(expected: str, observed: str) -> str:
    if observed == expected:
        return "correct"
    if observed in {"unknown", "uncertain", "unobservable"}:
        return "fail_closed"
    if expected == "inside" and observed == "outside":
        return "false_outside"
    if expected == "outside" and observed in {"inside", "empty"}:
        return "false_normal"
    if expected == "empty" and observed in {"inside", "outside"}:
        return "false_wafer"
    if expected in {"inside", "outside"} and observed == "empty":
        return "missed_wafer"
    return "mismatch"


def _stacking_outcome(expected: str, observed: str) -> str:
    if observed == expected:
        return "correct"
    if observed in {
        "unknown",
        "uncertain",
        "unobservable",
        "suspected",
        "not_applicable",
    }:
        return "fail_closed"
    if expected == "confirmed" and observed == "single":
        return "missed_stack"
    if expected == "single" and observed == "confirmed":
        return "false_stack"
    return "mismatch"


def _occupancy_outcome(observed_placement: str) -> str:
    if observed_placement in {"inside", "outside"}:
        return "correct"
    if observed_placement in {"unknown", "uncertain", "unobservable"}:
        return "fail_closed"
    if observed_placement == "empty":
        return "missed_wafer"
    return "mismatch"


def compare(
    batches: Iterable[Path],
    truth_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluated = _load_batches(batches)
    truth = _load_truth(truth_path)
    sequence_reports = []
    frame_rows: list[dict[str, Any]] = []
    global_counts: Counter[str] = Counter()
    represented_images = 0

    for key, truth_item in truth.items():
        sequence = evaluated.get(key)
        if sequence is None:
            raise ValueError(f"truth sequence has no evaluation output: {key}")
        placement_truth = _placement_truth(truth_item)
        stacking_truth = _stacking_truth(truth_item)
        occupancy_truth = {
            str(slot)
            for slot in truth_item.get("confirmed_occupancy", [])
            if str(slot) not in placement_truth
        }
        per_sequence: Counter[str] = Counter()
        frame_reports = []
        for frame in sequence.get("frames", []):
            represented_images += 1
            frame_counts: Counter[str] = Counter()
            issues = []
            if not bool(frame.get("success")):
                frame_counts["frame_rejected"] += 1
                issues.append(
                    {
                        "dimension": "frame",
                        "outcome": "frame_rejected",
                        "reason": frame.get("failure_reason"),
                    }
                )
            else:
                slots = {
                    str(item["slot_key"]): item
                    for item in frame.get("slots", [])
                }
                for slot_key, expected in placement_truth.items():
                    slot = slots[slot_key]
                    observed = str(slot["placement_state"])
                    outcome = _placement_outcome(expected, observed)
                    frame_counts[f"placement_{outcome}"] += 1
                    if outcome != "correct":
                        issues.append(
                            {
                                "dimension": "placement",
                                "slot_key": slot_key,
                                "expected": expected,
                                "observed": observed,
                                "state": slot["state"],
                                "outcome": outcome,
                                "coverage": slot["image_coverage_ratio"],
                                "flags": slot["flags"],
                            }
                        )
                for slot_key, expected in stacking_truth.items():
                    slot = slots[slot_key]
                    observed = str(slot["stacking_state"])
                    outcome = _stacking_outcome(expected, observed)
                    frame_counts[f"stacking_{outcome}"] += 1
                    if outcome != "correct":
                        issues.append(
                            {
                                "dimension": "stacking",
                                "slot_key": slot_key,
                                "expected": expected,
                                "observed": observed,
                                "state": slot["state"],
                                "outcome": outcome,
                                "coverage": slot["image_coverage_ratio"],
                                "flags": slot["flags"],
                            }
                        )
                for slot_key in sorted(occupancy_truth):
                    slot = slots[slot_key]
                    observed = str(slot["placement_state"])
                    outcome = _occupancy_outcome(observed)
                    frame_counts[f"occupancy_{outcome}"] += 1
                    if outcome != "correct":
                        issues.append(
                            {
                                "dimension": "occupancy",
                                "slot_key": slot_key,
                                "expected": "occupied",
                                "observed": observed,
                                "state": slot["state"],
                                "outcome": outcome,
                                "coverage": slot["image_coverage_ratio"],
                                "flags": slot["flags"],
                            }
                        )
            per_sequence.update(frame_counts)
            global_counts.update(frame_counts)
            high_risk = sum(
                count
                for name, count in frame_counts.items()
                if name in {
                    "placement_false_outside",
                    "placement_false_normal",
                    "placement_false_wafer",
                    "stacking_false_stack",
                }
            )
            frame_report = {
                "file": frame["file"],
                "success": bool(frame.get("success")),
                "counts": dict(frame_counts),
                "high_risk_error_count": high_risk,
                "issues": issues,
            }
            frame_reports.append(frame_report)
            frame_rows.append(
                {
                    "dataset": key[0],
                    "sequence": key[1],
                    "file": frame["file"],
                    "success": bool(frame.get("success")),
                    "high_risk_error_count": high_risk,
                    "issue_count": len(issues),
                    "counts_json": json.dumps(
                        dict(frame_counts), ensure_ascii=False, sort_keys=True
                    ),
                    "issues_json": json.dumps(
                        issues, ensure_ascii=False, sort_keys=True
                    ),
                }
            )

        consensus_issues = []
        aggregate_slots = sequence.get("aggregate", {}).get("slots", {})
        for slot_key, expected in placement_truth.items():
            observed = str(
                aggregate_slots[slot_key]["placement_consensus"]["decision"]
            )
            outcome = _placement_outcome(expected, observed)
            if outcome != "correct":
                consensus_issues.append(
                    {
                        "dimension": "placement",
                        "slot_key": slot_key,
                        "expected": expected,
                        "observed": observed,
                        "outcome": outcome,
                        "evidence": aggregate_slots[slot_key]["placement_consensus"],
                    }
                )
        for slot_key, expected in stacking_truth.items():
            observed = str(
                aggregate_slots[slot_key]["stacking_consensus"]["decision"]
            )
            outcome = _stacking_outcome(expected, observed)
            if outcome != "correct":
                consensus_issues.append(
                    {
                        "dimension": "stacking",
                        "slot_key": slot_key,
                        "expected": expected,
                        "observed": observed,
                        "outcome": outcome,
                        "evidence": aggregate_slots[slot_key]["stacking_consensus"],
                    }
                )
        for slot_key in sorted(occupancy_truth):
            observed = str(
                aggregate_slots[slot_key]["placement_consensus"]["decision"]
            )
            outcome = _occupancy_outcome(observed)
            if outcome != "correct":
                consensus_issues.append(
                    {
                        "dimension": "occupancy",
                        "slot_key": slot_key,
                        "expected": "occupied",
                        "observed": observed,
                        "outcome": outcome,
                        "evidence": aggregate_slots[slot_key]["placement_consensus"],
                    }
                )
        sequence_reports.append(
            {
                "dataset": key[0],
                "sequence": key[1],
                "image_count": len(frame_reports),
                "truth_placement_slot_count": len(placement_truth),
                "truth_occupancy_slot_count": len(occupancy_truth),
                "truth_stacking_slot_count": len(stacking_truth),
                "frame_counts": dict(per_sequence),
                "consensus_issue_count": len(consensus_issues),
                "consensus_issues": consensus_issues,
                "frames": frame_reports,
            }
        )

    report = {
        "schema_version": 1,
        "mode": "offline_human_truth_comparison",
        "runtime_truth_injection": False,
        "robot_motion_authorized": False,
        "truth_file": str(truth_path),
        "represented_image_count": represented_images,
        "sequence_count": len(sequence_reports),
        "global_frame_counts": dict(global_counts),
        "sequences": sequence_reports,
    }
    return report, frame_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batches", nargs="+", type=Path)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    args = parser.parse_args()
    report, rows = compare(args.batches, args.truth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(
        json.dumps(
            {
                "represented_image_count": report["represented_image_count"],
                "sequence_count": report["sequence_count"],
                "global_frame_counts": report["global_frame_counts"],
                "robot_motion_authorized": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
