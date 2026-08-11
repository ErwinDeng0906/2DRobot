"""Four-corner SCARA trajectory with a timed 3 mm Z cycle at each corner.

Importing this module never connects to or moves hardware. The SCARA UI loads
the metadata and calls :func:`run_trajectory` only after the operator presses
Start and confirms the route.

Route::

    P00 -> Z down/up -> P05 -> Z down/up ->
    P55 -> Z down/up -> P50 -> Z down/up -> P00 -> stop

The confirmed machine convention is used: decreasing J3 by 3 mm moves down.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence


TRAJECTORY_API_VERSION = 1
TRAJECTORY_NAME = "Four corners with 3 mm Z cycle"
TRAJECTORY_DESCRIPTION = (
    "At each perimeter corner: wait 1 s, lower J3 by 3 mm, wait 5 s, "
    "restore J3, wait 1 s; finally return to P00."
)

CORNER_NAMES = (
    "P00 float",
    "P05 float",
    "P55 float",
    "P50 float",
)
FINAL_RETURN_NAME = "P00 float"

Z_DOWN_MM = 3.0
WAIT_BEFORE_DOWN_S = 1.0
WAIT_AT_BOTTOM_S = 5.0
WAIT_AFTER_UP_S = 1.0

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRESET_FILE = _PROJECT_ROOT / "scara_presets.json"


@dataclass(frozen=True)
class TrajectoryPoint:
    """One absolute SCARA target: J1/J2/J4 in degrees and J3 in millimetres."""

    name: str
    joints: tuple[float, float, float, float]

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "joints": list(self.joints)}


@dataclass(frozen=True)
class TrajectoryResult:
    completed: bool
    completed_steps: int
    message: str


def _validated_joints(name: str, value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"Preset {name!r} must contain exactly four joint values")
    joints = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in joints):
        raise ValueError(f"Preset {name!r} contains a non-finite joint value")
    return joints  # type: ignore[return-value]


def load_corner_points(
    preset_file: Optional[str | Path] = None,
) -> dict[str, TrajectoryPoint]:
    """Load and validate the four unique corner presets."""

    path = Path(preset_file) if preset_file is not None else DEFAULT_PRESET_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"Preset file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Preset file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Preset file must contain a JSON object: {path}")

    points: dict[str, TrajectoryPoint] = {}
    for name in CORNER_NAMES:
        if name not in data:
            raise ValueError(f"Required preset is missing: {name!r}")
        points[name] = TrajectoryPoint(name, _validated_joints(name, data[name]))
    return points


def _down_point(point: TrajectoryPoint) -> TrajectoryPoint:
    joints = list(point.joints)
    joints[2] -= Z_DOWN_MM
    return TrajectoryPoint(
        f"{point.name} | Z down {Z_DOWN_MM:g} mm",
        tuple(joints),  # type: ignore[arg-type]
    )


def _restore_point(point: TrajectoryPoint) -> TrajectoryPoint:
    return TrajectoryPoint(f"{point.name} | Z restore", point.joints)


def _movement_plan(
    preset_file: Optional[str | Path] = None,
) -> list[TrajectoryPoint]:
    points = load_corner_points(preset_file)
    plan: list[TrajectoryPoint] = []
    for name in CORNER_NAMES:
        corner = points[name]
        plan.extend((corner, _down_point(corner), _restore_point(corner)))
    plan.append(points[FINAL_RETURN_NAME])
    return plan


def build_trajectory(preset_file: Optional[str | Path] = None) -> list[dict[str, object]]:
    """Return all 13 absolute movement targets for UI validation and preview."""

    return [point.as_dict() for point in _movement_plan(preset_file)]


def _interruptible_wait(
    seconds: float,
    should_stop: Optional[Callable[[], bool]],
    sleep_fn: Callable[[float], None],
) -> bool:
    """Wait in short slices so an operator Stop request is noticed promptly."""

    remaining = float(seconds)
    while remaining > 1e-9:
        if should_stop is not None and should_stop():
            return False
        interval = min(0.1, remaining)
        sleep_fn(interval)
        remaining -= interval
    return should_stop is None or not should_stop()


def run_trajectory(
    move_to: Callable[[str, Sequence[float]], bool],
    *,
    preset_file: Optional[str | Path] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int, TrajectoryPoint], None]] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> TrajectoryResult:
    """Execute the four corner Z-cycle route through a verified blocking callback."""

    points = load_corner_points(preset_file)
    total_movements = len(CORNER_NAMES) * 3 + 1
    completed = 0

    def move(point: TrajectoryPoint) -> bool:
        nonlocal completed
        if should_stop is not None and should_stop():
            return False
        if on_progress is not None:
            on_progress(completed + 1, total_movements, point)
        if not bool(move_to(point.name, point.joints)):
            return False
        completed += 1
        return True

    for name in CORNER_NAMES:
        corner = points[name]
        if not move(corner):
            return TrajectoryResult(False, completed, f"Movement failed at {corner.name}")
        if not _interruptible_wait(WAIT_BEFORE_DOWN_S, should_stop, sleep_fn):
            return TrajectoryResult(False, completed, "Trajectory cancelled before Z down")

        down = _down_point(corner)
        if not move(down):
            return TrajectoryResult(False, completed, f"Movement failed at {down.name}")
        if not _interruptible_wait(WAIT_AT_BOTTOM_S, should_stop, sleep_fn):
            return TrajectoryResult(False, completed, "Trajectory cancelled during bottom dwell")

        restore = _restore_point(corner)
        if not move(restore):
            return TrajectoryResult(False, completed, f"Movement failed at {restore.name}")
        if not _interruptible_wait(WAIT_AFTER_UP_S, should_stop, sleep_fn):
            return TrajectoryResult(False, completed, "Trajectory cancelled after Z restore")

    final = points[FINAL_RETURN_NAME]
    if not move(final):
        return TrajectoryResult(False, completed, f"Movement failed at final {final.name}")

    return TrajectoryResult(
        True,
        completed,
        "Four-corner Z-cycle trajectory completed; returned to P00",
    )


if __name__ == "__main__":
    # Read-only preview; no delays, controller connection, or hardware movement.
    print(f"{TRAJECTORY_NAME} (API v{TRAJECTORY_API_VERSION})")
    for index, step in enumerate(build_trajectory(), start=1):
        print(f"{index}: {step['name']}: {step['joints']}")
