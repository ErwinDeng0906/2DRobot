"""Sequence regressions for the PASS/PLANAR switch seen in the screen recording.

No cameras, robot connections, or motor commands are used by these tests.
"""
from dataclasses import replace
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tests.test_wafer_transfer_tracking import (
    PROJECT_ROOT, analysis, geometry, registration,
    result, robot_state,
)
from tests import test_wafer_transfer_tracking as fixtures
from scara.vision.tray_occupancy import SlotState
from scara.vision.wafer_transfer_runtime import LiveWaferTransferRuntime, WaferTransferFrame
from scara.vision.wafer_transfer_tracking import WaferTransferSession


def image_slot(name):
    item = analysis(name, SlotState.OCCUPIED)
    x, y, z = item.projection.center_T_mm
    return replace(item, projection=replace(
        item.projection, center_px=(120.0, 120.0),
        polygon_px=((100.0, 100.0), (140.0, 100.0), (140.0, 140.0), (100.0, 140.0)),
        polygon_T_mm=((x+10, y+10, z), (x+10, y-10, z),
                      (x-10, y-10, z), (x-10, y+10, z)),
    ))


def frame_for(runtime, name, sequence, *, metric=True, captured=None):
    captured = time.monotonic() if captured is None else captured
    observed = replace(result(), slots=(image_slot(name),), quality_passed=metric,
                       coordinate_mapping_allowed=metric, analysis_quality_passed=True)
    runtime.session.update_overview(observed, frame_sequence=sequence,
        frame_captured_monotonic_s=captured, robot_state=None)
    frame = WaferTransferFrame(sequence, captured, observed, runtime.session.snapshot(),
                               None, observed.annotated_image, runtime._stream_epoch)
    runtime._last_frame = frame
    runtime._last_result = observed
    return frame


class SelectionContinuityTests(unittest.TestCase):
    def test_click_uses_displayed_slot_not_newer_backend_slot(self):
        runtime = LiveWaferTransferRuntime(PROJECT_ROOT)
        shown = frame_for(runtime, 'P11', 1)
        frame_for(runtime, 'P22', 2)
        selected, _, distance = runtime.select_pixel((120, 120), role='source', displayed_frame=shown)
        self.assertEqual('P11', selected)
        self.assertAlmostEqual(0.0, distance)
        self.assertEqual('P11', runtime.session.source_slot)
        self.assertFalse(runtime.snapshot()['tracking_ready'])

    def test_read_only_frame_can_select_but_cannot_arm(self):
        runtime = LiveWaferTransferRuntime(PROJECT_ROOT)
        shown = frame_for(runtime, 'P11', 1, metric=False)
        runtime.select_pixel((120, 120), role='source', displayed_frame=shown)
        runtime.session.set_registration(registration())
        with self.assertRaises(RuntimeError):
            runtime.start_tracking()
        self.assertEqual('P11', runtime.session.source_slot)

    def test_blank_area_cannot_select_nearest_off_image_slot(self):
        runtime = LiveWaferTransferRuntime(PROJECT_ROOT)
        shown = frame_for(runtime, 'P11', 1)
        with self.assertRaisesRegex(ValueError, 'inside'):
            runtime.select_pixel((10, 10), role='source', displayed_frame=shown)
        self.assertIsNone(runtime.session.source_slot)

    def test_stale_and_invalidated_display_frames_are_rejected(self):
        runtime = LiveWaferTransferRuntime(PROJECT_ROOT)
        shown = frame_for(runtime, 'P11', 1, captured=time.monotonic()-2)
        with self.assertRaisesRegex(ValueError, 'stale'):
            runtime.select_pixel((120, 120), role='source', displayed_frame=shown)
        runtime.invalidate_camera1('test disconnect')
        frame_for(runtime, 'P22', 2)
        with self.assertRaisesRegex(ValueError, 'invalidated'):
            runtime.select_pixel((120, 120), role='source', displayed_frame=shown)

    def test_old_frame_cannot_replace_newer_state_or_add_evidence(self):
        session = WaferTransferSession(geometry())
        now = time.monotonic()
        session.update_overview(result(), frame_sequence=10,
            frame_captured_monotonic_s=now, robot_state=robot_state(120, 220, captured=now))
        for seq, captured in [(10, now), (9, now), (11, now-1)]:
            session.update_overview(result(SlotState.EMPTY), frame_sequence=seq,
                frame_captured_monotonic_s=captured, robot_state=None)
        self.assertEqual(10, session.latest_frame_sequence)
        self.assertIsNotNone(session.latest_robot_state)
        self.assertEqual(['occupied'], session.source_consensus('P11')['states'])

    def test_pose_failures_preserve_identity_not_motion_and_evidence_expires(self):
        session = WaferTransferSession(geometry())
        fixtures.WaferTransferSessionTests._update_overview_frames(session, count=2)
        session.select_source('P11')
        session.set_registration(registration())
        self.assertTrue(session.snapshot()['tracking_ready'])
        bad = replace(result(), quality_passed=False, coordinate_mapping_allowed=False,
                      failure_reason='Marker 2 reprojection rejected')
        for seq in range(3, 9):
            session.update_overview(bad, frame_sequence=seq,
                frame_captured_monotonic_s=time.monotonic(), robot_state=robot_state(120, 220))
        self.assertEqual('P11', session.source_slot)
        self.assertEqual(0, session.source_consensus()['dropout_frame_count'])
        self.assertEqual(6, session.source_consensus()['not_evaluated_frame_count'])
        self.assertFalse(session.snapshot()['tracking_ready'])
        with self.assertRaises(RuntimeError):
            session.start_tracking()
        with patch('scara.vision.wafer_transfer_tracking.time.monotonic', return_value=time.monotonic()+3):
            self.assertEqual([], session.source_consensus()['states'])

    def test_real_unknown_detections_still_exhaust_dropout_allowance(self):
        session = WaferTransferSession(geometry())
        fixtures.WaferTransferSessionTests._update_overview_frames(session, count=2)
        session.select_source('P11')
        fixtures.WaferTransferSessionTests._update_overview_frames(session, state=SlotState.UNKNOWN, count=3)
        self.assertEqual([], session.source_consensus()['states'])
        self.assertEqual('P11', session.source_slot)
        self.assertFalse(session.snapshot()['tracking_ready'])

    def test_unproven_destination_can_be_selected_but_not_used(self):
        session = WaferTransferSession(geometry())
        session.select_source('P11')
        session.select_destination('P22')
        self.assertEqual('P22', session.destination_slot)
        self.assertFalse(session.snapshot()['route_ready'])
        with self.assertRaises(RuntimeError):
            session.start_tracking()

    def test_pose_failure_diagnostics_are_preserved(self):
        session = WaferTransferSession(geometry())
        bad = replace(result(), quality_passed=False, coordinate_mapping_allowed=False,
                      failure_reason='Marker 2 reprojection rejected')
        session.update_overview(bad, frame_sequence=1,
            frame_captured_monotonic_s=time.monotonic(), robot_state=None)
        self.assertEqual(bad.failure_reason, session.snapshot()['pose_diagnostics']['reason'])
        self.assertEqual(bad.pose.per_marker_rms_px,
                         session.snapshot()['pose_diagnostics']['per_marker_rms_px'])
        self.assertEqual(bad.failure_reason,
                         session.snapshot()['recent_overview_diagnostics'][-1]['reason'])

    def test_latched_slot_is_not_revoked_by_transient_boundary_diagnostic(self):
        session = WaferTransferSession(geometry())
        fixtures.WaferTransferSessionTests._update_overview_frames(session, count=2)
        current = replace(result(), projection_diagnostics={'metric_slot_checks': {
            'P11': {'passed': False, 'reason': 'current calibrated boundary uncertain'},
        }})
        session.update_overview(current, frame_sequence=3,
            frame_captured_monotonic_s=time.monotonic(), robot_state=robot_state(120, 220))
        session.select_source('P11')
        session.set_registration(registration())
        self.assertTrue(session.source_consensus()['passed'])
        snapshot = session.snapshot()
        self.assertTrue(snapshot['tracking_ready'])
        gate = snapshot['selection_gates']['source_nominal_slot_geometry']
        self.assertTrue(gate['passed'])
        self.assertFalse(
            gate['actual']['wafer_boundary_diagnostic']['passed']
        )
        session.start_tracking()


class GeometryContinuityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = cv2.imread(str(PROJECT_ROOT / 'tests/fixtures/task14_planar_registration/raw_task14_1_013.jpg'))
        if cls.image is None:
            raise RuntimeError('required raw regression image missing')

    def setUp(self):
        self.runtime = LiveWaferTransferRuntime(PROJECT_ROOT)
        self.analyzer = self.runtime.analyzer

    def test_metric_pass_wait_switch_cannot_switch_slot_geometry_or_labels(self):
        pose = self.runtime.estimator.estimate(self.image)
        self.assertTrue(pose.quality_passed)
        outcomes = []
        for index, passed in enumerate((True, False, True, False)):
            current_pose = replace(pose, quality_passed=passed,
                                   failure_reason=None if passed else 'synthetic metric rejection')
            outcome = self.analyzer.analyze(self.image, pose=current_pose, captured_monotonic_s=10+index*.2)
            outcomes.append(outcome)
            self.assertEqual(passed, outcome.quality_passed)
            self.assertEqual(passed, outcome.coordinate_mapping_allowed)
            self.assertFalse(outcome.robot_correction_allowed)
        first = outcomes[0]
        for outcome in outcomes[1:]:
            self.assertEqual(first.projection_source, outcome.projection_source)
            for a, b in zip(first.slots, outcome.slots):
                np.testing.assert_allclose(a.projection.polygon_px, b.projection.polygon_px)
                self.assertEqual(a.decision.state, b.decision.state)

    def test_adjacent_raw_frames_keep_normal_target_without_moving_crop(self):
        second_image = cv2.imread(str(PROJECT_ROOT / 'tests/fixtures/task14_planar_registration/raw_task14_1_014.jpg'))
        first = self.analyzer.analyze(self.image, captured_monotonic_s=10)
        second = self.analyzer.analyze(second_image, captured_monotonic_s=10.2)
        self.assertTrue(second.projection_diagnostics['crop_anchor_held'])
        for a, b in zip(first.slots, second.slots):
            np.testing.assert_allclose(a.projection.polygon_px, b.projection.polygon_px)
        for outcome in (first, second):
            by_slot = {s.projection.slot_key: s for s in outcome.slots}
            self.assertEqual(SlotState.OCCUPIED, by_slot['P22'].decision.state)
            self.assertTrue(outcome.projection_diagnostics['metric_slot_checks']['P22']['passed'])
            self.assertNotEqual(SlotState.OCCUPIED, by_slot['P52'].decision.state)

    def test_cumulative_image_motion_is_compared_to_anchor_not_last_raw_fit(self):
        first = self.analyzer.analyze(self.image, captured_monotonic_s=10)
        raw = first.planar_registration
        original = raw.homography_image_from_tray_xy
        for offset, held in ((1.0, True), (3.0, False)):
            translate = np.array([[1, 0, offset], [0, 1, 0], [0, 0, 1]], dtype=float)
            shifted = replace(raw, homography_image_from_tray_xy=translate @ original)
            _, diagnostic = self.analyzer._observation_projections(shifted, self.image.shape, 10+offset*.1)
            self.assertEqual(held, diagnostic['crop_anchor_held'])

    def test_missing_plane_never_reuses_old_wafer_evidence(self):
        self.analyzer.analyze(self.image, captured_monotonic_s=10)
        black = np.zeros_like(self.image)
        failed = self.analyzer.analyze(black, captured_monotonic_s=10.2)
        self.assertFalse(failed.success)
        self.assertFalse(failed.coordinate_mapping_allowed)
        self.assertEqual((), failed.slots)
        self.assertIsNone(self.analyzer._observation_anchor)

    def test_anchor_expires_on_time_gap_and_reset(self):
        first = self.analyzer.analyze(self.image, captured_monotonic_s=10)
        _, diagnostic = self.analyzer._observation_projections(first.planar_registration, self.image.shape, 12)
        self.assertFalse(diagnostic['crop_anchor_held'])
        self.analyzer.reset_observation_geometry()
        self.assertIsNone(self.analyzer._observation_anchor)


if __name__ == '__main__':
    unittest.main()
