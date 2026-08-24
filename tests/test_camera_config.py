from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scara.config.camera_config import (
    CameraBinding,
    CameraConfigurationError,
    EnumeratedCamera,
    bindings_from_physical_map,
    load_camera_bindings,
    resolve_camera_sources,
    write_camera_bindings,
)


def _device(
    physical_index: int,
    identity: str,
    *,
    vid: str = "1BCF",
    pid: str = "2D4F",
    serial: str = "",
    container: str | None = None,
    location: str | None = None,
) -> EnumeratedCamera:
    return EnumeratedCamera(
        physical_index=physical_index,
        backend=700,
        backend_name="DSHOW",
        name=f"camera-{identity}",
        vid=vid,
        pid=pid,
        device_path=f"device-path-{identity}",
        instance_id=f"instance-{identity}",
        parent_instance_id=(f"parent\\{serial}" if serial else "parent\\5&x&0&1"),
        serial=serial,
        container_id=container or f"container-{identity}",
        location_path=location or f"location-{identity}",
    )


def _binding(logical: int, physical: int, device: EnumeratedCamera) -> CameraBinding:
    roles = {
        0: "overview",
        1: "forearm_fixed",
        2: "j4_rotating_close_range",
    }
    return CameraBinding(
        logical_index=logical,
        role=roles[logical],
        physical_index=physical,
        backend="DSHOW",
        vid=device.vid,
        pid=device.pid,
        serial=device.serial,
        container_id=device.container_id,
        device_path=device.device_path,
        location_path=device.location_path,
    )


class CameraConfigurationTests(unittest.TestCase):
    def _config(self, bindings: dict[int, CameraBinding], root: Path) -> Path:
        path = root / "local_config.toml"
        path.write_text(
            '[paths]\nsnrobotlab_dir = "D:\\\\SNRobotLab"\n',
            encoding="utf-8",
        )
        write_camera_bindings(bindings, config_path=path)
        return path

    def test_writer_preserves_unrelated_local_settings(self) -> None:
        devices = [_device(1, "a", serial="00001"), _device(2, "b"), _device(0, "c")]
        bindings = {
            0: _binding(0, 1, devices[0]),
            1: _binding(1, 2, devices[1]),
            2: _binding(2, 0, devices[2]),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = self._config(bindings, Path(tmp))
            text = path.read_text(encoding="utf-8")
            self.assertIn("[paths]", text)
            self.assertIn("[cameras.camera0]", text)
            loaded = load_camera_bindings(path)
        self.assertEqual(1, loaded[0].physical_index)
        self.assertEqual("00001", loaded[0].serial)

    def test_reordered_directshow_indices_preserve_logical_cameras(self) -> None:
        old_a = _device(1, "a", serial="00001")
        old_b = _device(2, "b")
        old_c = _device(0, "c", vid="13D3", pid="784B")
        bindings = {
            0: _binding(0, 1, old_a),
            1: _binding(1, 2, old_b),
            2: _binding(2, 0, old_c),
        }
        reordered = [
            _device(2, "a-new-path", serial="00001", container="new-a", location="new-a"),
            _device(0, "b", container=old_b.container_id, location=old_b.location_path),
            _device(
                1,
                "c",
                vid="13D3",
                pid="784B",
                container=old_c.container_id,
                location=old_c.location_path,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._config(bindings, Path(tmp))
            resolved = resolve_camera_sources(
                [0, 1, 2], config_path=path, devices=reordered
            )
        self.assertEqual({0: 2, 1: 0, 2: 1}, {k: v.physical_index for k, v in resolved.items()})
        self.assertTrue(all(row.configured_index_stale for row in resolved.values()))

    def test_serial_match_allows_physical_usb_port_change(self) -> None:
        old = _device(1, "old", serial="00001")
        bindings = {
            0: _binding(0, 1, old),
            1: _binding(1, 2, _device(2, "b")),
            2: _binding(2, 0, _device(0, "c", vid="13D3", pid="784B")),
        }
        moved = _device(
            5,
            "moved",
            serial="00001",
            container="different-container",
            location="different-port",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = self._config(bindings, Path(tmp))
            resolved = resolve_camera_sources([0], config_path=path, devices=[moved])
        self.assertEqual(5, resolved[0].physical_index)

    def test_nonserial_camera_port_change_is_rejected(self) -> None:
        old = _device(2, "b")
        bindings = {
            0: _binding(0, 1, _device(1, "a", serial="00001")),
            1: _binding(1, 2, old),
            2: _binding(2, 0, _device(0, "c", vid="13D3", pid="784B")),
        }
        moved = _device(4, "b-moved", container="new", location="new")
        with tempfile.TemporaryDirectory() as tmp:
            path = self._config(bindings, Path(tmp))
            with self.assertRaisesRegex(CameraConfigurationError, "禁止按物理Index猜测"):
                resolve_camera_sources([1], config_path=path, devices=[moved])

    def test_duplicate_identity_is_rejected(self) -> None:
        old = _device(1, "a", serial="00001")
        bindings = {
            0: _binding(0, 1, old),
            1: _binding(1, 2, _device(2, "b")),
            2: _binding(2, 0, _device(0, "c", vid="13D3", pid="784B")),
        }
        duplicates = [
            _device(3, "a1", serial="00001"),
            _device(4, "a2", serial="00001"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._config(bindings, Path(tmp))
            with self.assertRaisesRegex(CameraConfigurationError, "匹配到多个设备"):
                resolve_camera_sources([0], config_path=path, devices=duplicates)

    def test_initial_binding_requires_three_unique_current_indices(self) -> None:
        devices = [_device(0, "a"), _device(1, "b"), _device(2, "c")]
        bindings = bindings_from_physical_map({0: 1, 1: 2, 2: 0}, devices)
        self.assertEqual({0: 1, 1: 2, 2: 0}, {k: v.physical_index for k, v in bindings.items()})
        with self.assertRaises(CameraConfigurationError):
            bindings_from_physical_map({0: 1, 1: 2}, devices)

    def test_duplicate_configured_usb_identity_is_rejected(self) -> None:
        same = _device(1, "same", serial="00001")
        bindings = {
            0: _binding(0, 1, same),
            1: _binding(1, 2, same),
            2: _binding(2, 0, _device(0, "c", vid="13D3", pid="784B")),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "local_config.toml"
            path.write_text('[paths]\nsnrobotlab_dir = "D:\\\\SNRobotLab"\n', encoding="utf-8")
            with self.assertRaisesRegex(CameraConfigurationError, "相同USB身份"):
                write_camera_bindings(bindings, config_path=path)

    def test_independent_grabber_opens_resolved_physical_index(self) -> None:
        from scara.pipeline.backends import CaptureFrameGrabber

        opened = []

        class Frame:
            pass

        class Capture:
            def isOpened(self):
                return True

            def set(self, _prop, _value):
                return True

            def read(self):
                return True, Frame()

            def release(self):
                return None

        class FakeCv2:
            CAP_PROP_FRAME_WIDTH = 3
            CAP_PROP_FRAME_HEIGHT = 4

            @staticmethod
            def VideoCapture(index, backend):
                opened.append((index, backend))
                return Capture()

        resolved = SimpleNamespace(physical_index=7, backend=700)
        grabber = CaptureFrameGrabber(
            index=2,
            warmup_frames=0,
            source_resolver=lambda _logical: resolved,
        )
        with patch.dict(sys.modules, {"cv2": FakeCv2}):
            self.assertIsInstance(grabber.grab(), Frame)
        self.assertEqual(2, grabber.source_index)
        self.assertEqual(7, grabber.physical_source_index)
        self.assertEqual([(7, 700)], opened)


if __name__ == "__main__":
    unittest.main()
