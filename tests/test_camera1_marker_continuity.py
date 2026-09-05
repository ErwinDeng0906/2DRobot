"""Temporal camera-1 regressions. Synthetic motion; never connect hardware."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tests.test_tray_stage2_stage3 import TrayPoseEstimatorTests, _SyntheticDetector
from tests.test_transfer_observation_continuity import PROJECT_ROOT
from scara.vision.wafer_transfer_runtime import LiveWaferTransferRuntime
from scara.vision.tray_pose_estimator import TrayBoardPoseEstimator


class MarkerConsensusTests(TrayPoseEstimatorTests):
    def test_final_refinement_outlier_triggers_new_consensus(self):
        estimator, _, _ = self._synthetic_estimator(noise_px=.05)
        ids = list(estimator.detector._ids.reshape(-1))
        selected = [(key, points) for key, points in zip(ids, estimator.detector._corners) if key in (1, 2, 3, 5)]
        estimator.detector = _SyntheticDetector([p for _, p in selected], [key for key, _ in selected])
        original = estimator._marker_errors

        def final_errors(active, rvec, tvec):
            errors, rms = original(active, rvec, tvec)
            if len(active) == 4:
                # Reproduce a final all-active fit blaming two markers. The
                # previous loop refused to remove them because 4-2 < 3.
                errors[3] = 3.1
                errors[5] = 3.2
                rms = 2.6
            return errors, rms

        with patch.object(estimator, '_marker_errors', side_effect=final_errors):
            result = estimator.estimate(np.zeros((720, 1280, 3), dtype=np.uint8))
        self.assertTrue(result.quality_passed, result.failure_reason)
        self.assertEqual(3, len(result.used_marker_ids))
        self.assertEqual(1, len(result.rejected_marker_ids))
        self.assertLessEqual(max(result.per_marker_rms_px.values()), 3)

    def test_insufficient_markers_and_bad_final_fit_remain_closed(self):
        estimator, _, _ = self._synthetic_estimator()
        estimator.detector = _SyntheticDetector(estimator.detector._corners[:2], [1, 2])
        self.assertFalse(estimator.estimate(np.zeros((720, 1280, 3), np.uint8)).quality_passed)
        estimator, _, _ = self._synthetic_estimator()
        with patch.object(estimator, '_marker_errors', return_value=({1: 5., 2: 5., 3: 5.}, 5.)):
            self.assertFalse(estimator.estimate(np.zeros((720, 1280, 3), np.uint8)).quality_passed)

    def test_reported_inliers_belong_to_final_pose(self):
        estimator, _, _ = self._synthetic_estimator(noise_px=.5, corrupt_marker_id=4)
        result = estimator.estimate(np.zeros((720, 1280, 3), np.uint8))
        self.assertTrue(result.quality_passed, result.failure_reason)
        observed = dict(zip(estimator.detector._ids.reshape(-1), estimator.detector._corners))
        count = 0
        for key in result.used_marker_ids:
            pixels, _ = cv2.projectPoints(estimator.object_corners_by_id[key], result.rvec_C_T,
                result.tvec_C_T_mm, estimator.intrinsics.K, estimator.intrinsics.dist_coeffs)
            count += int(np.count_nonzero(np.linalg.norm(pixels.reshape(-1, 2) - observed[key].reshape(-1, 2), axis=1) <= 3))
        self.assertLessEqual(result.ransac_inlier_corner_count, count)
        self.assertGreaterEqual(result.ransac_inlier_corner_count, .6 * 4 * len(result.used_marker_ids))


class MarkerImageContinuityTests(unittest.TestCase):
    def setUp(self):
        self.runtime = LiveWaferTransferRuntime(PROJECT_ROOT)
        self.image = cv2.imread(str(PROJECT_ROOT / 'tests/fixtures/task14_planar_registration/raw_task14_1_013.jpg'))
        self.first = self.runtime.analyzer.analyze(self.image, captured_monotonic_s=10)
        self.tracker = self.runtime.analyzer._marker_image_tracker

    def test_noisy_board_fit_cannot_warp_stationary_image(self):
        raw = self.first.planar_registration
        jump = np.array([[1, .008, 3], [0, 1, -2], [0, 0, 1]], dtype=float)
        noisy = replace(raw, homography_image_from_tray_xy=jump @ raw.homography_image_from_tray_xy)
        tracked, diagnostic = self.tracker.update(self.image, self.first.slot_markers, noisy, 10.1)
        self.assertEqual('stationary', diagnostic['image_tracking'])
        np.testing.assert_allclose(tracked.homography_image_from_tray_xy, raw.homography_image_from_tray_xy)

    def test_real_cumulative_motion_is_followed_not_frozen(self):
        raw = self.first.planar_registration
        for index, displacement in enumerate((1., 2., 4.)):
            motion = np.array([[1, 0, displacement], [0, 1, 0], [0, 0, 1]], float)
            shifted = cv2.warpPerspective(self.image, motion, (self.image.shape[1], self.image.shape[0]))
            observed = {key: replace(value, corners_px=tuple(tuple(p) for p in np.asarray(value.corners_px) + [displacement, 0]),
                center_px=tuple(np.asarray(value.center_px) + [displacement, 0])) for key, value in self.first.slot_markers.items()}
            current = replace(raw, homography_image_from_tray_xy=motion @ raw.homography_image_from_tray_xy)
            tracked, diagnostic = self.tracker.update(shifted, observed, current, 10.1 + .1 * index)
            self.assertEqual('tracked', diagnostic['image_tracking'])
            points = np.array([[[0., 0.], [-100, -100]]], np.float32)
            np.testing.assert_allclose(cv2.perspectiveTransform(points, tracked.homography_image_from_tray_xy),
                cv2.perspectiveTransform(points, current.homography_image_from_tray_xy), atol=.25)

    def test_expired_or_undecoded_frame_cannot_reuse_tracking(self):
        raw = self.first.planar_registration
        _, diagnostic = self.tracker.update(self.image, {}, raw, 10.1)
        self.assertEqual('reacquired', diagnostic['image_tracking'])
        _, diagnostic = self.tracker.update(self.image, self.first.slot_markers, raw, 12)
        self.assertEqual('reacquired', diagnostic['image_tracking'])
        self.tracker.update(self.image, {}, replace(raw, success=False), 12.1)
        self.assertIsNone(self.tracker.gray)


class NativeStationarySequenceTests(unittest.TestCase):
    def test_native_continuous_frames_reproduce_corner_failure_and_fix(self):
        runtime = LiveWaferTransferRuntime(PROJECT_ROOT)
        legacy = TrayBoardPoseEstimator(runtime.geometry, runtime.intrinsics)
        paths = sorted((PROJECT_ROOT / 'tests/fixtures/camera1_stationary_0905').glob('frame_*.jpg'))
        self.assertEqual(8, len(paths))
        legacy_failures = 0
        for index, path in enumerate(paths):
            image = cv2.imread(str(path))
            legacy_failures += not legacy.estimate(image).quality_passed
            captured = 1000 + .13 * index
            with patch('time.monotonic', return_value=captured):
                frame = runtime.process_camera1(image, frame_sequence=index + 1,
                    captured_monotonic_s=captured, robot_state=None)
                self.assertTrue(frame.result.quality_passed, frame.result.failure_reason)
                self.assertGreaterEqual(len(frame.result.pose.used_marker_ids), 3)
                self.assertLessEqual(max(frame.result.pose.per_marker_rms_px.values()), 3)
                by_slot = {row.projection.slot_key: row for row in frame.result.slots}
                for key in ('P01', 'P20', 'P22'):
                    self.assertEqual('occupied', by_slot[key].decision.state.value)
                self.assertNotEqual('occupied', by_slot['P50'].decision.state.value)
                runtime.select_pixel(np.mean(by_slot['P22'].projection.polygon_px, axis=0),
                    role='source', displayed_frame=frame)
                self.assertEqual('P22', runtime.session.source_slot)
                self.assertFalse(runtime.snapshot()['tracking_ready'])
        self.assertGreater(legacy_failures, 0, 'sequence must retain the actual SUBPIX regression')
        with patch('time.monotonic', return_value=1002):
            failed = runtime.process_camera1(np.zeros_like(image), frame_sequence=20,
                captured_monotonic_s=1002, robot_state=None)
            self.assertFalse(failed.result.quality_passed)
            self.assertFalse(failed.session_snapshot['tracking_ready'])
            self.assertEqual('P22', runtime.session.source_slot)
            self.assertEqual({}, runtime.analyzer._slot_patch_history)


if __name__ == '__main__':
    unittest.main()
