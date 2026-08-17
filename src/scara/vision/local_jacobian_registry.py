"""Canonical filenames for per-slot local XY/image Jacobians.

P22 predates the per-slot registry and therefore keeps its approved legacy
location.  Every newly calibrated non-P22 slot is installed below the
``Jacobians`` directory with the slot name embedded in the filename.
"""

from __future__ import annotations

import re
from pathlib import Path


_SLOT_NAME = re.compile(r"^P[0-5][0-5]$")
LEGACY_P22_RELATIVE_PATH = Path("src/scara/calib/camera1_xy_image_jacobian.json")
JACOBIAN_DIRECTORY_RELATIVE_PATH = Path("src/scara/calib/Jacobians")


def validate_slot_name(target_name: str) -> str:
    """Return the canonical P00-P55 name or raise before any file write."""

    target = str(target_name).strip().upper()
    if _SLOT_NAME.fullmatch(target) is None:
        raise ValueError(f"局部Jacobian目标槽必须是P00-P55，收到：{target_name!r}")
    return target


def local_jacobian_filename(target_name: str) -> str:
    """Return the installed/result filename for one local calibration."""

    target = validate_slot_name(target_name)
    if target == "P22":
        return LEGACY_P22_RELATIVE_PATH.name
    return f"camera1_xy_image_jacobian_{target}.json"


def local_jacobian_relative_path(target_name: str) -> Path:
    """Return the project-relative approved calibration path.

    The P22 exception is intentional: its already validated file is not moved.
    """

    target = validate_slot_name(target_name)
    if target == "P22":
        return LEGACY_P22_RELATIVE_PATH
    return JACOBIAN_DIRECTORY_RELATIVE_PATH / local_jacobian_filename(target)


def local_jacobian_path(project_root: Path, target_name: str) -> Path:
    return Path(project_root) / local_jacobian_relative_path(target_name)


__all__ = [
    "JACOBIAN_DIRECTORY_RELATIVE_PATH",
    "LEGACY_P22_RELATIVE_PATH",
    "local_jacobian_filename",
    "local_jacobian_path",
    "local_jacobian_relative_path",
    "validate_slot_name",
]
