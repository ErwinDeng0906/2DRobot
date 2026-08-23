#!/usr/bin/env python3
"""Build one review contact sheet per static tray arrangement.

Every raw frame is shown exactly once. The sheets are an audit aid, not a
source of labels: detector predictions remain visibly marked as predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


CELL_WIDTH = 360
CELL_HEIGHT = 285
LABEL_HEIGHT = 52
COLUMNS = 5

STATE_LABELS = {
    "empty": "empty",
    "occupied": "ok",
    "warning": "warn",
    "outside_slot": "outside",
    "stacked": "stacked",
    "stacked_outside_slot": "stacked_outside",
    "out_of_view": "OOV",
    "unknown": "unknown",
}

STATE_COLOURS = {
    "occupied": (255, 0, 255),
    "warning": (0, 180, 255),
    "outside_slot": (0, 0, 255),
    "stacked": (0, 0, 255),
    "stacked_outside_slot": (0, 0, 200),
}


def _load_batches(paths: Iterable[Path]) -> list[dict[str, Any]]:
    sequences: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        sequences.extend(payload.get("sequences", []))
    return sequences


def _crop_from_slots(image: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    points = []
    for slot in frame.get("slots", []):
        points.extend(slot.get("polygon_px", []))
    if not points:
        return image
    array = np.asarray(points, dtype=np.float32)
    height, width = image.shape[:2]
    x0 = max(0, int(np.floor(float(np.min(array[:, 0])))))
    y0 = max(0, int(np.floor(float(np.min(array[:, 1])))))
    x1 = min(width, int(np.ceil(float(np.max(array[:, 0])))))
    y1 = min(height, int(np.ceil(float(np.max(array[:, 1])))))
    if x1 <= x0 or y1 <= y0:
        return image
    margin = max(12, int(0.04 * max(x1 - x0, y1 - y0)))
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(width, x1 + margin)
    y1 = min(height, y1 + margin)
    return image[y0:y1, x0:x1]


def _fit_in_cell(image: np.ndarray) -> np.ndarray:
    target_height = CELL_HEIGHT - LABEL_HEIGHT
    scale = min(CELL_WIDTH / image.shape[1], target_height / image.shape[0])
    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    cell = np.full((CELL_HEIGHT, CELL_WIDTH, 3), 24, dtype=np.uint8)
    x = (CELL_WIDTH - resized.shape[1]) // 2
    y = LABEL_HEIGHT + (target_height - resized.shape[0]) // 2
    cell[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return cell


def _annotate_prediction(
    image: np.ndarray,
    frame: dict[str, Any],
) -> np.ndarray:
    """Mark every predicted wafer slot without covering empty tray geometry."""

    canvas = image.copy()
    scale = max(image.shape[:2]) / 1800.0
    thickness = max(2, int(round(3.0 * scale)))
    font_scale = max(0.48, 0.72 * scale)
    for slot in frame.get("slots", []):
        if not bool(slot.get("wafer_found")):
            continue
        polygon = np.asarray(slot.get("polygon_px", ()), dtype=np.float32)
        if polygon.shape != (4, 2):
            continue
        state = str(slot.get("state", "warning"))
        colour = STATE_COLOURS.get(state, (0, 180, 255))
        cv2.polylines(
            canvas,
            [np.rint(polygon).astype(np.int32)],
            True,
            colour,
            thickness,
            cv2.LINE_AA,
        )
        center = np.mean(polygon, axis=0)
        label = str(slot.get("slot_key", "?"))
        origin = (
            int(round(float(center[0]) - 15.0 * scale)),
            int(round(float(center[1]) + 5.0 * scale)),
        )
        cv2.putText(
            canvas,
            label,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            colour,
            max(1, thickness - 1),
            cv2.LINE_AA,
        )
    return canvas


def _frame_label(index: int, frame: dict[str, Any]) -> tuple[str, str]:
    if not frame.get("success"):
        return f"{index:02d} {frame['file']}", "FIT REJECTED"
    summary = frame.get("summary", {})
    short = " ".join(
        f"{STATE_LABELS.get(key, key)}={summary[key]}"
        for key in sorted(summary)
        if summary[key]
    )
    return f"{index:02d} {frame['file']}", f"prediction: {short}"


def _draw_label(cell: np.ndarray, first: str, second: str) -> None:
    colour = (235, 235, 235)
    cv2.putText(
        cell, first[:54], (7, 19), cv2.FONT_HERSHEY_SIMPLEX,
        0.38, colour, 1, cv2.LINE_AA,
    )
    cv2.putText(
        cell, second[:70], (7, 42), cv2.FONT_HERSHEY_SIMPLEX,
        0.39, (125, 220, 255), 1, cv2.LINE_AA,
    )


def build_contact_sheets(
    sequences: Iterable[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_payload = []
    for sequence_index, sequence in enumerate(sequences, start=1):
        root = Path(sequence["dataset_root"])
        source_dir = root / sequence["sequence"]
        frames = sequence.get("frames", [])
        rows = max(1, (len(frames) + COLUMNS - 1) // COLUMNS)
        sheet = np.full(
            (rows * CELL_HEIGHT, COLUMNS * CELL_WIDTH, 3),
            12,
            dtype=np.uint8,
        )
        files = []
        for frame_index, frame in enumerate(frames, start=1):
            image_path = source_dir / frame["file"]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                cell = np.full((CELL_HEIGHT, CELL_WIDTH, 3), 24, dtype=np.uint8)
                cv2.putText(
                    cell, "CANNOT READ IMAGE", (38, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    cv2.LINE_AA,
                )
            else:
                annotated = _annotate_prediction(image, frame)
                cell = _fit_in_cell(_crop_from_slots(annotated, frame))
            first, second = _frame_label(frame_index, frame)
            _draw_label(cell, first, second)
            row = (frame_index - 1) // COLUMNS
            column = (frame_index - 1) % COLUMNS
            y0 = row * CELL_HEIGHT
            x0 = column * CELL_WIDTH
            sheet[y0:y0 + CELL_HEIGHT, x0:x0 + CELL_WIDTH] = cell
            files.append({"index": frame_index, "file": frame["file"]})
        safe_root = root.name.replace(" ", "_")
        safe_sequence = sequence["sequence"].replace(" ", "_")
        destination = output_dir / f"{sequence_index:02d}_{safe_root}_{safe_sequence}.jpg"
        if not cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"cannot write contact sheet: {destination}")
        index_payload.append(
            {
                "sequence_index": sequence_index,
                "dataset_root": str(root),
                "sequence": sequence["sequence"],
                "contact_sheet": str(destination),
                "image_count": len(frames),
                "files": files,
            }
        )
    return {"sequence_count": len(index_payload), "sequences": index_payload}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batches", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    args = parser.parse_args()
    sequences = _load_batches(args.batches)
    payload = build_contact_sheets(sequences, args.output_dir)
    args.index.parent.mkdir(parents=True, exist_ok=True)
    args.index.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "sequence_count": payload["sequence_count"],
        "image_count": sum(item["image_count"] for item in payload["sequences"]),
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
