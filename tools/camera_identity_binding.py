"""Scan, bind, preview, and audit the three logical SCARA cameras.

This tool never touches robot or motion APIs.  ``--write-current-map`` is the
only option that updates ``local_config.toml``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Mapping


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="backslashreplace")


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.config.camera_config import (  # noqa: E402
    CAMERA_ROLES,
    CameraConfigurationError,
    ResolvedCameraSource,
    bindings_from_physical_map,
    enumerate_directshow_cameras,
    resolve_camera_sources,
    write_camera_bindings,
)


def _mapping(value: str) -> dict[int, int]:
    result: dict[int, int] = {}
    try:
        for item in value.split(","):
            logical_text, physical_text = item.split("=", 1)
            logical = int(logical_text.strip())
            physical = int(physical_text.strip())
            result[logical] = physical
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("映射格式应为0=1,1=2,2=0") from exc
    if set(result) != set(CAMERA_ROLES) or len(set(result.values())) != 3:
        raise argparse.ArgumentTypeError("必须为逻辑0、1、2指定三个不同物理Index")
    return result


def _print_devices(devices) -> None:
    print("physical  name                    VID:PID    serial   container / location")
    for row in devices:
        serial = row.serial or "-"
        print(
            f"{row.physical_index:>8}  {row.name[:22]:<22}  "
            f"{row.vid}:{row.pid}  {serial:<8} "
            f"{row.container_id} / {row.location_path}"
        )


def _capture_health(resolved: Mapping[int, ResolvedCameraSource]) -> dict[str, dict]:
    import cv2

    report: dict[str, dict] = {}
    for logical, row in sorted(resolved.items()):
        cap = cv2.VideoCapture(row.physical_index, row.backend)
        opened = bool(cap.isOpened())
        frame_ok = False
        shape = None
        if opened:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            for _attempt in range(12):
                ok, frame = cap.read()
                if ok and frame is not None and getattr(frame, "size", 0):
                    frame_ok = True
                    shape = list(frame.shape)
                time.sleep(0.03)
        cap.release()
        report[str(logical)] = {
            "logical_index": logical,
            "physical_source_index": row.physical_index,
            "opened": opened,
            "frame_ok": frame_ok,
            "frame_shape": shape,
        }
        if not opened or not frame_ok:
            raise CameraConfigurationError(
                f"逻辑相机{logical}/物理Index {row.physical_index}预热取帧失败"
            )
    return report


def _preview_and_confirm(resolved: Mapping[int, ResolvedCameraSource]) -> dict[str, bool]:
    import cv2

    confirmations: dict[str, bool] = {}
    instructions = {
        0: "overview / 全局概览",
        1: "forearm_fixed / 前臂固定主视觉",
        2: "j4_rotating_close_range / J4旋转近距离",
    }
    for logical, row in sorted(resolved.items()):
        cap = cv2.VideoCapture(row.physical_index, row.backend)
        if not cap.isOpened():
            cap.release()
            raise CameraConfigurationError(f"无法打开逻辑相机{logical}")
        title = (
            f"logical {logical} -> physical {row.physical_index}: "
            f"{instructions[logical]} | Y confirm, N reject"
        )
        confirmed = False
        rejected = False
        while not confirmed and not rejected:
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.putText(
                frame,
                title,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(title, frame)
            key = cv2.waitKey(30) & 0xFF
            confirmed = key in (ord("y"), ord("Y"))
            rejected = key in (ord("n"), ord("N"), 27)
        cap.release()
        cv2.destroyWindow(title)
        confirmations[str(logical)] = confirmed
        if not confirmed:
            cv2.destroyAllWindows()
            raise CameraConfigurationError(f"操作员拒绝逻辑相机{logical}的画面身份")
    cv2.destroyAllWindows()
    return confirmations


def _write_audit(payload: dict, audit_dir: Path) -> Path:
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / (
        "camera_identity_audit_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".json"
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "local_config.toml",
        help="本机local_config.toml路径",
    )
    write_group = parser.add_mutually_exclusive_group()
    write_group.add_argument(
        "--write-current-map",
        action="store_true",
        help="将--mapping和当前USB身份写入local_config.toml",
    )
    write_group.add_argument(
        "--refresh-indices",
        action="store_true",
        help="按已有USB身份安全刷新三个缓存物理Index",
    )
    parser.add_argument(
        "--mapping",
        type=_mapping,
        help="逻辑到物理映射，例如0=1,1=2,2=0；只在明确写配置时使用",
    )
    parser.add_argument("--verify", action="store_true", help="逐台打开并预热取帧")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="逐台显示画面，要求按Y人工确认角色",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=ROOT / "logs",
        help="身份审计JSON输出目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        devices = enumerate_directshow_cameras()
        _print_devices(devices)
        if len(devices) != 3:
            raise CameraConfigurationError(
                f"期望3台DirectShow相机，实际检测到{len(devices)}台"
            )
        if args.write_current_map:
            if args.mapping is None:
                raise CameraConfigurationError(
                    "--write-current-map必须同时明确提供--mapping"
                )
            bindings = bindings_from_physical_map(args.mapping, devices)
            path = write_camera_bindings(bindings, config_path=args.config)
            print(f"已更新本机映射：{path}")
        resolved = resolve_camera_sources(
            sorted(CAMERA_ROLES), config_path=args.config, devices=devices
        )
        if args.refresh_indices:
            refreshed_mapping = {
                logical: row.physical_index
                for logical, row in resolved.items()
            }
            bindings = bindings_from_physical_map(refreshed_mapping, devices)
            path = write_camera_bindings(bindings, config_path=args.config)
            print(f"已按USB身份刷新缓存物理Index：{path}")
            resolved = resolve_camera_sources(
                sorted(CAMERA_ROLES), config_path=args.config, devices=devices
            )
        for logical, row in sorted(resolved.items()):
            stale = " (config index stale)" if row.configured_index_stale else ""
            print(
                f"logical {logical} {row.role} -> physical "
                f"{row.physical_index}{stale}"
            )
        capture_health = _capture_health(resolved) if args.verify else {}
        confirmations = _preview_and_confirm(resolved) if args.preview else {}
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "config_path": str(args.config.resolve()),
            "mapping_source": (
                "explicit_write_current_map"
                if args.write_current_map
                else "usb_identity_refresh"
                if args.refresh_indices
                else "existing_local_config"
            ),
            "configured_mapping": {
                str(logical): row.configured_physical_index
                for logical, row in sorted(resolved.items())
            },
            "devices": [device.to_json() for device in devices],
            "resolved": {
                str(logical): row.to_json()
                for logical, row in sorted(resolved.items())
            },
            "capture_health": capture_health,
            "operator_confirmed": confirmations,
        }
        audit_path = _write_audit(payload, args.audit_dir)
        print(f"身份审计：{audit_path}")
        return 0
    except CameraConfigurationError as exc:
        print(f"CAMERA IDENTITY ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
