"""Real Qt selection/painting against a captured sequence, without hardware.

The monitor is never started and no robot state/provider is supplied. All
motion buttons must remain disabled. Only the preview receives test clicks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import cv2
import numpy as np
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QImage, QFont, QFontDatabase
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from scara.ui.wafer_transfer_dialog import WaferTransferDialog, WaferTransferMonitorThread


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    entries = json.loads((args.input / 'capture.json').read_text(encoding='utf-8-sig'))['frames']
    if args.limit:
        entries = entries[:args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    app = QApplication.instance() or QApplication([])
    # Offscreen Qt does not enumerate Windows fallback fonts automatically.
    # Load them explicitly so the verification recording remains readable.
    for filename in ('msyh.ttc', 'consola.ttf'):
        font_path = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / filename
        if font_path.exists():
            QFontDatabase.addApplicationFont(str(font_path))
    app.setFont(QFont('Microsoft YaHei', 9))
    with patch.object(WaferTransferMonitorThread, 'start'):
        dialog = WaferTransferDialog(ROOT, SimpleNamespace(source_index=1))
    dialog.resize(1500, 950)
    dialog.show()
    app.processEvents()
    hashes, clicks = set(), []
    writer = None
    fps = (len(entries) - 1) / (entries[-1]['elapsed_s'] - entries[0]['elapsed_s'])
    try:
        for index, entry in enumerate(entries):
            image = cv2.imread(str(args.input / entry['file']))
            hashes.add(hashlib.sha256(image.tobytes()).hexdigest())
            timestamp = 1000 + entry['elapsed_s']
            with patch('time.monotonic', return_value=timestamp):
                frame = dialog.runtime.process_camera1(image, frame_sequence=index + 1,
                    captured_monotonic_s=timestamp, robot_state=None)
                rgb = cv2.cvtColor(frame.annotated_bgr, cv2.COLOR_BGR2RGB)
                qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
                dialog._on_frame(qimage, frame)
                app.processEvents()
                slot = next(row for row in frame.result.slots if row.projection.slot_key == 'P22')
                center = np.mean(slot.projection.polygon_px, axis=0)
                label = dialog.preview
                width, height = label._displayed_size
                point = QPoint(round((label.width() - width) / 2 + center[0] * width / image.shape[1]),
                    round((label.height() - height) / 2 + center[1] * height / image.shape[0]))
                QTest.mouseClick(label, Qt.MouseButton.LeftButton, pos=point)
                app.processEvents()
                assert dialog.runtime.session.source_slot == 'P22'
                assert not dialog.xy_motion_button.isEnabled()
                assert not dialog.track_button.isEnabled()
                clicks.append(frame.frame_sequence)
                # Render the actual Qt window, not an illustrative mockup.
                rendered = dialog.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
                data = rendered.bits()
                data.setsize(rendered.sizeInBytes())
                rgba = np.frombuffer(data, np.uint8).reshape(rendered.height(), rendered.bytesPerLine())[:, :rendered.width()*4].reshape(rendered.height(), rendered.width(), 4)
                bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                if writer is None:
                    writer = cv2.VideoWriter(str(args.output / 'ui_replay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (bgr.shape[1], bgr.shape[0]))
                    assert writer.isOpened()
                writer.write(bgr)
                if index % 30 == 0:
                    cv2.imwrite(str(args.output / f'ui_{index:04d}.png'), bgr)
                    print(json.dumps({'frames': index + 1, 'clicked': len(clicks)}), flush=True)
        # Disconnect/reset is a safety failure, not permission to retain a
        # formerly usable coordinate or an enabled motion button.
        dialog.runtime.invalidate_camera1('offline disconnect test')
        dialog._invalidate_current('offline disconnect test')
        assert not dialog.xy_motion_button.isEnabled()
        assert dialog.runtime.analyzer._observation_anchor is None
        report = {'frames': len(entries), 'distinct_decoded_images': len(hashes),
            'successful_Qt_clicks_P22': len(clicks), 'motion_enabled_frames': 0,
            'disconnect_closed_motion': True, 'hardware_connected': False}
        (args.output / 'ui_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)
    finally:
        if writer is not None:
            writer.release()
        dialog.close()
        app.processEvents()


if __name__ == '__main__':
    main()
