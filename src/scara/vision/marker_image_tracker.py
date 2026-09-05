"""Current-image marker motion for observation crops, NEVER a metric pose.

Track from a keyframe rather than integrating tiny pairwise displacements.
Every result needs a newly checked plane and decoded-marker agreement. No
wafer labels, camera poses or robot permissions are cached here.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import cv2
import numpy as np

from .planar_tray_registration import PlanarTrayRegistration
from .slot_marker_observation import ArucoObservation


class MarkerImageTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.gray = None
        self.points = None
        self.ids = ()
        self.registration = None
        self.captured = None
        self.last_time = None

    def _seed(self, gray, observations, registration, captured):
        ids = tuple(sorted(key for key, value in observations.items()
                           if value.complete_decoded and len(value.corners_px) == 4))
        self.gray = gray.copy()
        self.ids = ids
        self.points = np.asarray([observations[key].corners_px for key in ids], dtype=np.float32).reshape(-1, 1, 2)
        self.registration = registration
        self.captured = captured
        self.last_time = captured

    def update(self, image, observations: Mapping[int, ArucoObservation],
               registration: PlanarTrayRegistration, captured: float):
        if not registration.success:
            self.reset()
            return registration, {'image_tracking': 'unavailable'}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        diagnostic = {'image_tracking': 'reacquired', 'metric_pose_cached': False}
        tracked = None
        if (self.gray is not None and self.gray.shape == gray.shape
                and self.last_time is not None and 0 < captured - self.last_time <= .8
                and len(self.ids) >= 3):
            parameters = dict(winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
            current, status, error = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.points, None, **parameters)
            if current is not None and status is not None:
                back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, current, None, **parameters)
                if back is not None and reverse is not None and error is not None:
                    fb = np.linalg.norm(back - self.points, axis=2).reshape(-1)
                    valid = status.reshape(-1).astype(bool) & reverse.reshape(-1).astype(bool)
                    valid &= (fb < .5) & (error.reshape(-1) < 15) & np.isfinite(current.reshape(-1, 2)).all(axis=1)
                    source = self.points.reshape(-1, 2)
                    target = current.reshape(-1, 2)
                    if np.count_nonzero(valid) >= 8:
                        motion, mask = cv2.findHomography(source[valid], target[valid], cv2.RANSAC, 1.0)
                        if motion is not None and mask is not None and np.isfinite(motion).all():
                            inlier = np.zeros(len(source), dtype=bool)
                            inlier[np.flatnonzero(valid)] = mask.reshape(-1).astype(bool)
                            groups = np.count_nonzero(inlier.reshape(-1, 4).sum(axis=1) >= 2)
                            # Decode-to-flow validation uses current physical
                            # corners, not the noisy board-calibration fit.
                            predicted = cv2.perspectiveTransform(self.points, motion).reshape(-1, 4, 2)
                            agreement = []
                            for index, key in enumerate(self.ids):
                                observed = observations.get(key)
                                if observed is not None and observed.complete_decoded:
                                    residual = np.linalg.norm(predicted[index] - np.asarray(observed.corners_px), axis=1)
                                    if np.count_nonzero(residual <= 2.0) >= 3:
                                        agreement.append(key)
                            diagnostic.update(flow_inlier_points=int(inlier.sum()),
                                flow_marker_count=int(groups), flow_decoded_agreement=agreement)
                            if groups >= 3 and inlier.sum() >= 8 and inlier.sum() / valid.sum() >= .7 and len(agreement) >= 3:
                                displacement = np.linalg.norm(predicted.reshape(-1, 2) - source, axis=1)
                                maximum = float(np.max(displacement))
                                stationary = maximum <= .35
                                matrix = (self.registration.homography_image_from_tray_xy if stationary else
                                          motion @ self.registration.homography_image_from_tray_xy)
                                tracked = replace(registration, homography_image_from_tray_xy=matrix.copy())
                                diagnostic.update(image_tracking='stationary' if stationary else 'tracked',
                                    flow_anchor_displacement_px=maximum)
        if tracked is None:
            self._seed(gray, observations, registration, captured)
            return registration, diagnostic
        self.last_time = captured
        # Rebase on the accepted tracked geometry, never inject the new raw
        # fit just because the set of decoded markers changes.
        if captured - self.captured >= 2 or diagnostic['flow_anchor_displacement_px'] > 8:
            self._seed(gray, observations, tracked, captured)
        return tracked, diagnostic
