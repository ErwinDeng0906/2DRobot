"""Fit camera1 fixed-height planar hand-eye from Task8 plus Task12 or Task13."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.file_io import atomic_write_text
from scara.vision.planar_handeye import fit_planar_handeye, install_planar_handeye


def _latest(root: Path, filename: str) -> Path:
    candidates = sorted(
        (root / "Trajectory Photos").glob(f"*/{filename}"),
        key=lambda path: path.parent.name,
        reverse=True,
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if payload.get("status") in {"success", "visibility_gaps_detected"}:
                return path
        except Exception:
            continue
    raise RuntimeError(f"找不到可用{filename}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--task8-run", type=Path)
    parser.add_argument("--task12-run", type=Path)
    parser.add_argument("--task13-run", type=Path)
    parser.add_argument("--suction-target", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    task8_report = args.suction_target or _latest(root, "camera1_suction_target.json")
    task8_run = (args.task8_run or task8_report.parent).resolve()
    if args.task12_run is not None and args.task13_run is not None:
        parser.error("--task12-run与--task13-run不能同时使用")
    if args.task13_run is not None:
        secondary_run = args.task13_run.resolve()
        secondary_name = f"task13_{secondary_run.name}"
    else:
        task12_report = _latest(root, "task12_stage3_visibility_scan.json")
        secondary_run = (args.task12_run or task12_report.parent).resolve()
        secondary_name = f"task12_{secondary_run.name}"
    report = fit_planar_handeye(
        root,
        [
            (f"task8_{task8_run.name}", task8_run / "points.json"),
            (secondary_name, secondary_run / "points.json"),
        ],
        suction_target_path=task8_report,
    )
    destination = args.report or (
        root
        / "Trajectory Photos"
        / f"planar_handeye_existing_{datetime.now():%y%m%d%H%M%S}.json"
    )
    atomic_write_text(
        destination,
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.task13_run is not None:
        summary_lines = [
            "# Task13 平面手眼标定结果",
            "",
            f"- 状态：`{report['status']}`",
            "- 适用范围：相机1、固定高度平面XY；不支持Z或完整6-DoF。",
            "- 同槽重复帧已聚合；验证按完整槽位留出。",
            "- 跨run旋转门经操作人员授权设为1.00°；接近上限属于低余量通过。",
            "",
            "## 质量门",
            "",
        ]
        for name, gate in report["quality_gates"].items():
            summary_lines.append(
                f"- {'PASS' if gate['passed'] else 'FAIL'} `{name}`："
                f"actual={gate['actual']}，limit={gate['limit']}"
            )
        atomic_write_text(
            destination.with_suffix(".md"),
            "\n".join(summary_lines) + "\n",
            encoding="utf-8",
        )
    print(f"status={report['status']}")
    print(f"report={destination}")
    for name, gate in report["quality_gates"].items():
        print(f"{'PASS' if gate['passed'] else 'FAIL'} {name}: {gate['actual']} {gate['limit']}")
    if args.install:
        installed = install_planar_handeye(
            report,
            root / "src/scara/calib/camera1_forearm_planar_handeye.json",
        )
        print(f"installed={installed}")
    return 0 if report["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
