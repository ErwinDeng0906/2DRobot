"""Offscreen transfer UI tests: no monitor thread or hardware is started."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from dataclasses import replace
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PyQt6.QtGui import QImage
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from tests.test_transfer_observation_continuity import PROJECT_ROOT, frame_for
from scara.ui.wafer_transfer_dialog import WaferTransferDialog, WaferTransferMonitorThread


class TransferSelectionUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        with patch.object(WaferTransferMonitorThread, 'start'):
            self.dialog = WaferTransferDialog(PROJECT_ROOT, SimpleNamespace(source_index=1))

    def tearDown(self):
        self.dialog._pick_xy_active = False
        self.dialog.close()
        self.app.processEvents()

    def test_number_selection_is_available_without_camera_but_motion_is_disabled(self):
        self.dialog.slot_selector.setCurrentText('P22')
        self.dialog.select_slot_button.click()
        self.assertEqual('P22', self.dialog.runtime.session.source_slot)
        self.assertFalse(self.dialog.xy_motion_button.isEnabled())
        self.assertIn('已选择 P22', self.dialog.status.toPlainText())

    def test_read_only_image_click_selects_displayed_slot_and_reports_metric_wait(self):
        frame = frame_for(self.dialog.runtime, 'P11', 1, metric=False)
        self.dialog._on_frame(QImage(1280, 720, QImage.Format.Format_RGB888), frame)
        frame_for(self.dialog.runtime, 'P22', 2, metric=False)
        self.dialog._on_image_clicked(120, 120)
        self.assertEqual('P11', self.dialog.runtime.session.source_slot)
        text = self.dialog.status.toPlainText()
        self.assertIn('已选择 P11', text)
        self.assertIn('运动位姿：WAIT', text)
        self.assertIn('STALE', text)
        self.assertFalse(self.dialog.xy_motion_button.isEnabled())

    def test_armed_session_disables_number_selection(self):
        self.dialog._set_pick_xy_controls(True)
        self.assertFalse(self.dialog.slot_selector.isEnabled())
        self.assertFalse(self.dialog.select_slot_button.isEnabled())

    def test_real_widget_click_maps_letterboxed_image_to_displayed_slot(self):
        self.dialog.resize(1516, 960)
        self.dialog.show()
        self.app.processEvents()
        frame = frame_for(self.dialog.runtime, 'P11', 1, metric=False)
        self.dialog._on_frame(QImage(1280, 720, QImage.Format.Format_RGB888), frame)
        self.app.processEvents()
        label = self.dialog.preview
        width, height = label._displayed_size
        position = QPoint(round((label.width() - width) / 2 + 120 * width / 1280),
                          round((label.height() - height) / 2 + 120 * height / 720))
        QTest.mouseClick(label, Qt.MouseButton.LeftButton, pos=position)
        self.app.processEvents()
        self.assertEqual('P11', self.dialog.runtime.session.source_slot)
        self.assertFalse(self.dialog.xy_motion_button.isEnabled())

    def test_raw_overlay_is_optional_and_does_not_arm_motion(self):
        self.assertFalse(self.dialog.runtime.analyzer.show_raw_wafer_geometry)
        self.dialog.raw_overlay_checkbox.setChecked(True)
        self.assertTrue(self.dialog.runtime.analyzer.show_raw_wafer_geometry)
        self.assertFalse(self.dialog.xy_motion_button.isEnabled())

    def test_pick_xy_sample_uses_motion_gates_after_source_is_occluded(self):
        frame = frame_for(self.dialog.runtime, 'P11', 1, metric=True)
        snapshot = dict(frame.session_snapshot)
        snapshot['source_slot'] = 'P11'
        snapshot['source_state'] = {'state': 'occluded'}
        snapshot['selection_gates'] = {
            'source_normal_occupied_consensus': {'passed': False},
        }
        snapshot['xy_motion_gates'] = {
            'current_overview_pose_quality': {'passed': True},
            'locked_runtime_registration': {'passed': True},
            'locked_source_not_explicitly_contradicted': {'passed': True},
            'fresh_frame_synchronised_robot_state': {'passed': True},
        }
        sample = self.dialog._pick_xy_sample(
            replace(frame, session_snapshot=snapshot)
        )
        self.assertTrue(sample['accepted'])
        self.assertEqual('occluded', sample['source_state']['state'])
        self.assertIn('motion_gates', sample)

    def test_locked_navigation_button_does_not_flash_back_enabled(self):
        frame = frame_for(self.dialog.runtime, 'P11', 1, metric=True)
        snapshot = dict(frame.session_snapshot)
        snapshot['tracking_ready'] = True
        snapshot['registration_locked'] = True
        self.dialog._last_frame = frame
        self.dialog._refresh_status(snapshot)
        self.assertFalse(self.dialog.track_button.isEnabled())
        self.assertTrue(self.dialog.xy_motion_button.isEnabled())

    def test_boundary_diagnostic_does_not_gray_ready_buttons(self):
        frame = frame_for(self.dialog.runtime, 'P11', 1, metric=True)
        snapshot = dict(frame.session_snapshot)
        snapshot['tracking_ready'] = True
        snapshot['registration_locked'] = False
        gates = dict(snapshot.get('selection_gates') or {})
        gates['source_nominal_slot_geometry'] = {
            'passed': True,
            'actual': {
                'wafer_boundary_diagnostic': {
                    'passed': False,
                    'reason': 'current metric pose or normal wafer unavailable',
                },
            },
        }
        snapshot['selection_gates'] = gates
        self.dialog._last_frame = frame
        self.dialog._refresh_status(snapshot)
        self.assertTrue(self.dialog.track_button.isEnabled())
        self.assertTrue(self.dialog.xy_motion_button.isEnabled())

    def test_controller_start_rejection_is_visible_in_transfer_status(self):
        self.dialog.report_pick_xy_start_rejected(
            '未使能，请先点击“使能”'
        )
        self.assertIn(
            'XY悬空移动未启动：未使能，请先点击“使能”',
            self.dialog.status.toPlainText(),
        )

    def test_monitor_waits_for_nearby_buffered_robot_state(self):
        frame_time = 20.0
        states = iter([
            {
                'captured_monotonic_s': frame_time - 0.50,
                'joints': [0.0] * 4,
                'pose': [0.0] * 6,
            },
            {
                'captured_monotonic_s': frame_time + 0.08,
                'joints': [0.0] * 4,
                'pose': [0.0] * 6,
            },
        ])
        monitor = WaferTransferMonitorThread(
            SimpleNamespace(), self.dialog.runtime, lambda _capture: next(states)
        )
        monitor.msleep = lambda _milliseconds: None
        paired = monitor._robot_state_nearest_capture(frame_time)
        self.assertTrue(
            math.isclose(paired['captured_monotonic_s'], frame_time + 0.08)
        )


if __name__ == '__main__':
    unittest.main()
