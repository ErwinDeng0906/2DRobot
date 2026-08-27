"""Stable logical camera identities backed by ``local_config.toml``.

Tasks and calibration records use logical camera numbers.  DirectShow device
indices are machine-local, volatile implementation details and are resolved
here immediately before a camera is opened.
"""

from __future__ import annotations

import json
import locale
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .scara_config import _load_toml, find_config_file, project_root


CAMERA_ROLES = {
    0: "overview",
    1: "forearm_fixed",
    2: "j4_rotating_close_range",
}
MANAGED_CAMERA_BEGIN = "# BEGIN managed camera bindings"
MANAGED_CAMERA_END = "# END managed camera bindings"
_DEVICE_PATH_RE = re.compile(
    r"^\\\\\?\\(?P<enumerator>[^#]+)#(?P<hardware>[^#]+)#"
    r"(?P<instance>[^#]+)#",
    re.IGNORECASE,
)
_PROPERTY_RE = re.compile(
    r"^\s+(?P<key>DEVPKEY_Device_(?:Parent|ContainerId|LocationPaths))\s+\[[^]]+\]:\s*$"
)
_CAMERA_SECTION_RE = re.compile(
    r"(?ms)^\[cameras\.camera[012]\]\s*\r?\n.*?(?=^\[|\Z)"
)


class CameraConfigurationError(RuntimeError):
    """The logical-to-physical camera map is missing or unsafe."""


def _normalized_text(value: object) -> str:
    return str(value or "").strip()


def _normalized_hex(value: object) -> str:
    if isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return f"{value:04X}"
    text = _normalized_text(value).upper().removeprefix("0X")
    return text.zfill(4) if text else ""


def _normalized_identity(value: object) -> str:
    return _normalized_text(value).casefold()


def _toml_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


@dataclass(frozen=True)
class CameraBinding:
    logical_index: int
    role: str
    physical_index: int
    backend: str
    vid: str
    pid: str
    serial: str
    container_id: str
    device_path: str
    location_path: str

    def to_json(self) -> dict[str, Any]:
        return {
            "logical_index": self.logical_index,
            "role": self.role,
            "physical_index": self.physical_index,
            "backend": self.backend,
            "vid": self.vid,
            "pid": self.pid,
            "serial": self.serial,
            "container_id": self.container_id,
            "device_path": self.device_path,
            "location_path": self.location_path,
        }


@dataclass(frozen=True)
class EnumeratedCamera:
    physical_index: int
    backend: int
    backend_name: str
    name: str
    vid: str
    pid: str
    device_path: str
    instance_id: str = ""
    parent_instance_id: str = ""
    serial: str = ""
    container_id: str = ""
    location_path: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "physical_index": self.physical_index,
            "backend": self.backend,
            "backend_name": self.backend_name,
            "name": self.name,
            "vid": self.vid,
            "pid": self.pid,
            "device_path": self.device_path,
            "instance_id": self.instance_id,
            "parent_instance_id": self.parent_instance_id,
            "serial": self.serial,
            "container_id": self.container_id,
            "location_path": self.location_path,
        }


@dataclass(frozen=True)
class ResolvedCameraSource:
    logical_index: int
    role: str
    configured_physical_index: int
    physical_index: int
    backend: int
    backend_name: str
    identity: EnumeratedCamera
    configured_index_stale: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "logical_index": self.logical_index,
            "role": self.role,
            "configured_physical_index": self.configured_physical_index,
            "physical_source_index": self.physical_index,
            "configured_index_stale": self.configured_index_stale,
            "camera_identity": self.identity.to_json(),
        }


def _camera_instance_id(device_path: str) -> str:
    match = _DEVICE_PATH_RE.match(str(device_path))
    if match is None:
        return ""
    return "\\".join(
        (
            match.group("enumerator").upper(),
            match.group("hardware").upper(),
            match.group("instance").upper(),
        )
    )


def _pnputil_properties(
    instance_id: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, list[str]]:
    if os.name != "nt" or not instance_id:
        return {}
    try:
        completed = runner(
            ["pnputil", "/enum-devices", "/instanceid", instance_id, "/properties"],
            check=False,
            capture_output=True,
            text=True,
            encoding=locale.getpreferredencoding(False),
            errors="replace",
            timeout=8.0,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if int(completed.returncode) != 0:
        return {}
    values: dict[str, list[str]] = {}
    active_key: Optional[str] = None
    for line in str(completed.stdout or "").splitlines():
        property_match = _PROPERTY_RE.match(line)
        if property_match is not None:
            active_key = property_match.group("key")
            values[active_key] = []
            continue
        if active_key is None:
            continue
        if re.match(r"^\s+(?:DEVPKEY_|\{)[^:]*\[[^]]+\]:\s*$", line):
            active_key = None
            continue
        value = line.strip()
        if value:
            values[active_key].append(value)
    return values


def _serial_from_parent(parent: str) -> str:
    suffix = str(parent).rsplit("\\", 1)[-1].strip()
    if not suffix or "&" in suffix:
        return ""
    return suffix


def enumerate_directshow_cameras() -> list[EnumeratedCamera]:
    """Return DirectShow indices enriched with stable USB/PnP identity."""

    try:
        import cv2
        from cv2_enumerate_cameras import enumerate_cameras
    except Exception as exc:  # pragma: no cover - environment dependent
        raise CameraConfigurationError(
            "缺少相机枚举依赖；请在scara310环境安装"
            " cv2-enumerate-cameras==1.3.3"
        ) from exc
    rows: list[EnumeratedCamera] = []
    try:
        camera_infos = list(enumerate_cameras(cv2.CAP_DSHOW))
    except Exception as exc:  # pragma: no cover - hardware dependent
        raise CameraConfigurationError(f"DirectShow相机枚举失败：{exc}") from exc
    for info in camera_infos:
        device_path = str(info.path or "")
        instance_id = _camera_instance_id(device_path)
        properties = _pnputil_properties(instance_id)
        parent = next(iter(properties.get("DEVPKEY_Device_Parent", ())), "")
        container_id = next(
            iter(properties.get("DEVPKEY_Device_ContainerId", ())), ""
        )
        location_path = next(
            iter(properties.get("DEVPKEY_Device_LocationPaths", ())), ""
        )
        rows.append(
            EnumeratedCamera(
                physical_index=int(info.index),
                backend=int(info.backend),
                backend_name="DSHOW",
                name=str(info.name or ""),
                vid=_normalized_hex(info.vid),
                pid=_normalized_hex(info.pid),
                device_path=device_path,
                instance_id=instance_id,
                parent_instance_id=parent,
                serial=_serial_from_parent(parent),
                container_id=container_id,
                location_path=location_path,
            )
        )
    return sorted(rows, key=lambda row: row.physical_index)


def _required_string(section: Mapping[str, object], key: str, label: str) -> str:
    if key not in section:
        raise CameraConfigurationError(f"{label}缺少{key}")
    value = _normalized_text(section[key])
    if key not in {"serial"} and not value:
        raise CameraConfigurationError(f"{label}.{key}不能为空")
    return value


def _validate_binding_set(bindings: Mapping[int, CameraBinding]) -> None:
    if set(bindings) != set(CAMERA_ROLES):
        raise CameraConfigurationError("必须完整配置逻辑相机0、1、2")
    for logical_index, binding in bindings.items():
        if binding.logical_index != logical_index:
            raise CameraConfigurationError("相机绑定键与logical_index不一致")
        if binding.role != CAMERA_ROLES[logical_index]:
            raise CameraConfigurationError(
                f"逻辑相机{logical_index}角色必须为{CAMERA_ROLES[logical_index]!r}"
            )
    physical_indices = [binding.physical_index for binding in bindings.values()]
    if len(set(physical_indices)) != len(physical_indices):
        raise CameraConfigurationError("local_config.toml中物理相机Index重复")
    identity_keys = [
        (
            binding.vid,
            binding.pid,
            "serial",
            _normalized_identity(binding.serial),
        )
        if binding.serial
        else (
            binding.vid,
            binding.pid,
            "port",
            _normalized_identity(binding.container_id),
            _normalized_identity(binding.location_path),
        )
        for binding in bindings.values()
    ]
    if len(set(identity_keys)) != len(identity_keys):
        raise CameraConfigurationError("多个逻辑相机配置了相同USB身份")


def load_camera_bindings(path: Optional[str | Path] = None) -> dict[int, CameraBinding]:
    config_path = find_config_file(str(path) if path is not None else None)
    if config_path is None:
        raise CameraConfigurationError("找不到local_config.toml")
    try:
        root = _load_toml(config_path)
    except Exception as exc:
        raise CameraConfigurationError(f"无法读取local_config.toml：{exc}") from exc
    cameras = root.get("cameras")
    if not isinstance(cameras, Mapping):
        raise CameraConfigurationError("local_config.toml缺少[cameras.camera0..2]")
    bindings: dict[int, CameraBinding] = {}
    for logical_index, expected_role in CAMERA_ROLES.items():
        label = f"cameras.camera{logical_index}"
        section = cameras.get(f"camera{logical_index}")
        if not isinstance(section, Mapping):
            raise CameraConfigurationError(f"local_config.toml缺少[{label}]")
        physical = section.get("physical_index")
        if isinstance(physical, bool) or not isinstance(physical, int):
            raise CameraConfigurationError(f"{label}.physical_index必须是整数")
        if not 0 <= int(physical) <= 8:
            raise CameraConfigurationError(f"{label}.physical_index必须在0到8之间")
        role = _required_string(section, "role", label)
        if role != expected_role:
            raise CameraConfigurationError(
                f"{label}.role必须为{expected_role!r}，实际为{role!r}"
            )
        backend = _required_string(section, "backend", label).upper()
        if backend != "DSHOW":
            raise CameraConfigurationError(f"{label}.backend首版只支持DSHOW")
        bindings[logical_index] = CameraBinding(
            logical_index=logical_index,
            role=role,
            physical_index=int(physical),
            backend=backend,
            vid=_normalized_hex(_required_string(section, "vid", label)),
            pid=_normalized_hex(_required_string(section, "pid", label)),
            serial=_required_string(section, "serial", label),
            container_id=_required_string(section, "container_id", label),
            device_path=_required_string(section, "device_path", label),
            location_path=_required_string(section, "location_path", label),
        )
    _validate_binding_set(bindings)
    return bindings


def _binding_matches(binding: CameraBinding, device: EnumeratedCamera) -> bool:
    if binding.backend != device.backend_name.upper():
        return False
    if binding.vid != _normalized_hex(device.vid) or binding.pid != _normalized_hex(
        device.pid
    ):
        return False
    if binding.serial:
        return _normalized_identity(binding.serial) == _normalized_identity(device.serial)
    return bool(
        _normalized_identity(binding.container_id)
        == _normalized_identity(device.container_id)
        and _normalized_identity(binding.location_path)
        == _normalized_identity(device.location_path)
    )


def resolve_camera_sources(
    logical_indices: Sequence[int],
    *,
    config_path: Optional[str | Path] = None,
    devices: Optional[Sequence[EnumeratedCamera]] = None,
) -> dict[int, ResolvedCameraSource]:
    bindings = load_camera_bindings(config_path)
    available = list(devices) if devices is not None else enumerate_directshow_cameras()
    resolved: dict[int, ResolvedCameraSource] = {}
    for raw_index in logical_indices:
        logical_index = int(raw_index)
        if logical_index not in bindings:
            raise CameraConfigurationError(f"未配置逻辑相机{logical_index}")
        binding = bindings[logical_index]
        configured = next(
            (
                device
                for device in available
                if device.physical_index == binding.physical_index
                and _binding_matches(binding, device)
            ),
            None,
        )
        matches = (
            [configured]
            if configured is not None
            else [device for device in available if _binding_matches(binding, device)]
        )
        if not matches:
            raise CameraConfigurationError(
                f"逻辑相机{logical_index}({binding.role})未找到匹配的USB设备；"
                "禁止按物理Index猜测，请运行相机身份更新工具"
            )
        if len(matches) != 1:
            indices = ",".join(str(device.physical_index) for device in matches)
            raise CameraConfigurationError(
                f"逻辑相机{logical_index}匹配到多个设备Index[{indices}]，已安全拒绝"
            )
        device = matches[0]
        resolved[logical_index] = ResolvedCameraSource(
            logical_index=logical_index,
            role=binding.role,
            configured_physical_index=binding.physical_index,
            physical_index=device.physical_index,
            backend=device.backend,
            backend_name=device.backend_name,
            identity=device,
            configured_index_stale=device.physical_index != binding.physical_index,
        )
    physical = [row.physical_index for row in resolved.values()]
    if len(set(physical)) != len(physical):
        raise CameraConfigurationError("多个逻辑相机解析到同一个物理设备，已安全拒绝")
    return resolved


def resolve_camera_source(
    logical_index: int,
    *,
    config_path: Optional[str | Path] = None,
    devices: Optional[Sequence[EnumeratedCamera]] = None,
) -> ResolvedCameraSource:
    return resolve_camera_sources(
        [int(logical_index)], config_path=config_path, devices=devices
    )[int(logical_index)]


def bindings_from_physical_map(
    logical_to_physical: Mapping[int, int],
    devices: Sequence[EnumeratedCamera],
) -> dict[int, CameraBinding]:
    if set(logical_to_physical) != set(CAMERA_ROLES):
        raise CameraConfigurationError("首次绑定必须同时指定逻辑相机0、1、2")
    by_index = {device.physical_index: device for device in devices}
    bindings: dict[int, CameraBinding] = {}
    for logical_index, role in CAMERA_ROLES.items():
        physical_index = int(logical_to_physical[logical_index])
        if physical_index not in by_index:
            raise CameraConfigurationError(
                f"物理Index {physical_index}不在当前DirectShow相机列表中"
            )
        device = by_index[physical_index]
        if not device.container_id or not device.location_path:
            raise CameraConfigurationError(
                f"物理Index {physical_index}缺少PnP ContainerId或USB端口路径"
            )
        bindings[logical_index] = CameraBinding(
            logical_index=logical_index,
            role=role,
            physical_index=physical_index,
            backend="DSHOW",
            vid=device.vid,
            pid=device.pid,
            serial=device.serial,
            container_id=device.container_id,
            device_path=device.device_path,
            location_path=device.location_path,
        )
    return bindings


def render_camera_bindings(bindings: Mapping[int, CameraBinding]) -> str:
    lines = [MANAGED_CAMERA_BEGIN]
    for logical_index in sorted(CAMERA_ROLES):
        binding = bindings[logical_index]
        lines.extend(
            [
                f"[cameras.camera{logical_index}]",
                f"role = {_toml_string(binding.role)}",
                f"physical_index = {binding.physical_index}",
                f"backend = {_toml_string(binding.backend)}",
                f"vid = {_toml_string(binding.vid)}",
                f"pid = {_toml_string(binding.pid)}",
                f"serial = {_toml_string(binding.serial)}",
                f"container_id = {_toml_string(binding.container_id)}",
                f"device_path = {_toml_string(binding.device_path)}",
                f"location_path = {_toml_string(binding.location_path)}",
                "",
            ]
        )
    lines.append(MANAGED_CAMERA_END)
    return "\n".join(lines).rstrip() + "\n"


def write_camera_bindings(
    bindings: Mapping[int, CameraBinding],
    *,
    config_path: Optional[str | Path] = None,
) -> Path:
    _validate_binding_set(bindings)
    path = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else (find_config_file() or (project_root() / "local_config.toml")).resolve()
    )
    if not path.is_file():
        raise CameraConfigurationError(f"本机配置不存在：{path}")
    original = path.read_text(encoding="utf-8-sig")
    managed_re = re.compile(
        re.escape(MANAGED_CAMERA_BEGIN)
        + r".*?"
        + re.escape(MANAGED_CAMERA_END)
        + r"\s*",
        re.DOTALL,
    )
    cleaned = managed_re.sub("", original)
    cleaned = _CAMERA_SECTION_RE.sub("", cleaned).rstrip()
    updated = cleaned + "\n\n" + render_camera_bindings(bindings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(updated)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    load_camera_bindings(path)
    return path


def resolved_camera_identity_json(logical_index: int) -> dict[str, Any]:
    return resolve_camera_source(logical_index).to_json()


__all__ = [
    "CAMERA_ROLES",
    "CameraBinding",
    "CameraConfigurationError",
    "EnumeratedCamera",
    "ResolvedCameraSource",
    "bindings_from_physical_map",
    "enumerate_directshow_cameras",
    "load_camera_bindings",
    "render_camera_bindings",
    "resolve_camera_source",
    "resolve_camera_sources",
    "resolved_camera_identity_json",
    "write_camera_bindings",
]
