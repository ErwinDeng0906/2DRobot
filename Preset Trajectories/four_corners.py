"""Four-corner SCARA trajectory loaded from ``scara_presets.json``.

This module is deliberately hardware-independent. Importing it only defines the
trajectory; it never connects to or moves the robot. The UI can later load this
file, call :func:`build_trajectory` to preview/validate it, and execute
:func:`run_four_corners` in a worker with a blocking, safety-checked ``move_to``
callback.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence


TRAJECTORY_API_VERSION = 1
TRAJECTORY_NAME = "Four corners"
TRAJECTORY_DESCRIPTION = "Visit the four perimeter corners and return to P00."

# Perimeter order. P00 is intentionally repeated to close the trajectory.
WAYPOINT_NAMES = (
    "P00 float",
    "P05 float",
    "P55 float",
    "P50 float",
    "P00 float",
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRESET_FILE = _PROJECT_ROOT / "scara_presets.json"


@dataclass(frozen=True)
class TrajectoryPoint:
    """One validated SCARA target: J1/J2/J4 in degrees and J3 in millimetres."""

    name: str
    joints: tuple[float, float, float, float]

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "joints": list(self.joints)}


@dataclass(frozen=True)
class TrajectoryResult:
    """Result returned by :func:`run_four_corners`."""

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


def load_points(preset_file: Optional[str | Path] = None) -> dict[str, TrajectoryPoint]:
    """Load and validate the four unique corner points from the preset JSON file."""

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
    for name in dict.fromkeys(WAYPOINT_NAMES):
        if name not in data:
            raise ValueError(f"Required preset is missing: {name!r}")
        points[name] = TrajectoryPoint(name, _validated_joints(name, data[name]))
    return points


def build_trajectory(preset_file: Optional[str | Path] = None) -> list[dict[str, object]]:
    """Return the validated five-step route in a UI-friendly representation."""

    points = load_points(preset_file)
    return [points[name].as_dict() for name in WAYPOINT_NAMES]


def run_four_corners(
    move_to: Callable[[str, Sequence[float]], bool],
    *,
    preset_file: Optional[str | Path] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int, TrajectoryPoint], None]] = None,
    dwell_s: float = 0.0,
) -> TrajectoryResult:
    """Execute the route through an injected blocking movement callback.

    ``move_to(name, joints)`` must not return until that target has either been
    reached and verified or has failed. It must also perform the controller's
    connection, enable, alarm, limit, and collision checks. Returning ``False``
    aborts the trajectory immediately.

    ``should_stop`` allows the future UI Stop/E-stop state to cancel between
    targets. It cannot replace the robot's physical emergency stop.
    """

    if dwell_s < 0 or not math.isfinite(float(dwell_s)):
        raise ValueError("dwell_s must be a finite, non-negative number")

    points = load_points(preset_file)
    total = len(WAYPOINT_NAMES)
    completed = 0

    for index, name in enumerate(WAYPOINT_NAMES, start=1):
        if should_stop is not None and should_stop():
            return TrajectoryResult(False, completed, "Trajectory cancelled")

        point = points[name]
        if on_progress is not None:
            on_progress(index, total, point)

        if not bool(move_to(point.name, point.joints)):
            return TrajectoryResult(False, completed, f"Movement failed at {point.name}")

        completed += 1
        if dwell_s and index < total:
            time.sleep(float(dwell_s))

    return TrajectoryResult(True, completed, "Four-corner trajectory completed")


# Generic name for the future UI trajectory-loader contract.
run_trajectory = run_four_corners


if __name__ == "__main__":
    # Read-only preview: this deliberately does not connect to or move hardware.
    print(f"{TRAJECTORY_NAME} (API v{TRAJECTORY_API_VERSION})")
    for index, step in enumerate(build_trajectory(), start=1):
        print(f"{index}: {step['name']}: {step['joints']}")
