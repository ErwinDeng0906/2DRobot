"""Qt UI package with a lazy main-widget import.

Keeping the package initializer lightweight allows controller-free and
ActionWorker protocol tests to import their modules without constructing the
entire desktop UI dependency graph.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "ScaraControlWidget":
        from .control_widget import ScaraControlWidget

        return ScaraControlWidget
    raise AttributeError(name)


__all__ = ["ScaraControlWidget"]
