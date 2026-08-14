"""Small cross-thread helpers for atomic JSON/text file replacement.

On Windows, a normal Python reader does not share ``FILE_SHARE_DELETE``.
Consequently ``os.replace(temp, target)`` can raise WinError 5 or 32 while a
different thread (or an external scanner) briefly has ``target`` open.  This
module combines a process-local, per-target re-entrant lock with a short,
strictly bounded retry for those two transient Windows sharing errors.
"""

from __future__ import annotations

import math
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


MAXIMUM_REPLACE_RETRY_DEADLINE_S = 1.0
DEFAULT_REPLACE_RETRY_DEADLINE_S = 0.75
_RETRYABLE_WINDOWS_REPLACE_ERRORS = frozenset({5, 32})
_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}


def _resolved_path_key(path: Path) -> str:
    resolved = Path(path).resolve(strict=False)
    return os.path.normcase(str(resolved))


def _lock_for_path(path: Path) -> threading.RLock:
    key = _resolved_path_key(path)
    with _LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


@contextmanager
def locked_path_io(path: Path) -> Iterator[None]:
    """Serialize in-process readers and writers for one resolved target."""

    with _lock_for_path(Path(path)):
        yield


def _retryable_windows_replace_error(exc: OSError) -> bool:
    return getattr(exc, "winerror", None) in _RETRYABLE_WINDOWS_REPLACE_ERRORS


def replace_with_retry(
    source: Path,
    target: Path,
    *,
    deadline_s: float = DEFAULT_REPLACE_RETRY_DEADLINE_S,
    initial_delay_s: float = 0.01,
    maximum_delay_s: float = 0.10,
) -> None:
    """Atomically replace ``target``, retrying only WinError 5/32.

    The first replace attempt is immediate.  The total retry deadline is never
    allowed to exceed one second; non-sharing errors are raised immediately.
    """

    deadline_value = float(deadline_s)
    initial_delay = float(initial_delay_s)
    maximum_delay = float(maximum_delay_s)
    if (
        not math.isfinite(deadline_value)
        or deadline_value < 0.0
        or deadline_value > MAXIMUM_REPLACE_RETRY_DEADLINE_S
    ):
        raise ValueError("replace retry deadline must be between 0 and 1 second")
    if not math.isfinite(initial_delay) or initial_delay <= 0.0:
        raise ValueError("replace retry initial delay must be finite and positive")
    if not math.isfinite(maximum_delay) or maximum_delay < initial_delay:
        raise ValueError("replace retry maximum delay must be >= initial delay")

    expires_at = time.monotonic() + deadline_value
    delay = initial_delay
    while True:
        try:
            os.replace(Path(source), Path(target))
            return
        except OSError as exc:
            if not _retryable_windows_replace_error(exc):
                raise
            remaining = expires_at - time.monotonic()
            if remaining <= 0.0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(maximum_delay, delay * 2.0)


def _unique_temporary_path(target: Path) -> Path:
    return target.with_name(
        f".{target.name}.{os.getpid()}.{threading.get_ident()}."
        f"{uuid.uuid4().hex}.tmp"
    )


def atomic_write_text(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    retry_deadline_s: float = DEFAULT_REPLACE_RETRY_DEADLINE_S,
) -> None:
    """Write text through a unique same-directory temp file and replace."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_temporary_path(target)
    with locked_path_io(target):
        try:
            temporary.write_text(str(text), encoding=encoding)
            replace_with_retry(
                temporary,
                target,
                deadline_s=retry_deadline_s,
            )
        finally:
            try:
                temporary.unlink()
            except OSError:
                # Never hide the original write/replace error with a secondary
                # best-effort cleanup failure.  The UUID name prevents a stale
                # temp from colliding with any later write.
                pass


def read_text_snapshot(path: Path, *, encoding: str = "utf-8") -> str:
    """Read one complete target while coordinating with in-process writers."""

    target = Path(path)
    with locked_path_io(target):
        return target.read_text(encoding=encoding)


__all__ = [
    "DEFAULT_REPLACE_RETRY_DEADLINE_S",
    "MAXIMUM_REPLACE_RETRY_DEADLINE_S",
    "atomic_write_text",
    "locked_path_io",
    "read_text_snapshot",
    "replace_with_retry",
]
