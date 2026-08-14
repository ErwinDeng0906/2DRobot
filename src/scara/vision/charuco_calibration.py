"""Reusable ChArUco camera-calibration algorithms.

This module contains no robot commands and no Qt widgets.  It owns the
repeatable computer-vision pipeline used by calibration tasks:

1. detect ArUco markers and interpolated ChArUco corners;
2. score each image for corner count, board footprint, border margin,
   sharpness, approximate tilt, and accumulated image coverage;
3. estimate camera matrix ``K``, distortion coefficients, and one board pose
   per accepted image by minimizing squared reprojection error;
4. reject high-error images with a robust median/MAD rule and recalibrate;
5. verify that accepted board normals contain enough pose diversity;
6. calculate a final per-image reprojection RMS for reporting.

The module deliberately does not decide where calibration JSON is stored.
Persistence and task lifecycle integration live in
``charuco_calibration_runtime.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class CharucoBoardSpec:
    """Physical identity of one measured ChArUco board."""

    squares_x: int = 10
    squares_y: int = 8
    square_length_mm: float = 18.80
    marker_length_mm: float = 9.87
    dictionary_name: str = "DICT_4X4_50"
    legacy_pattern: bool = True

    @property
    def corner_count(self) -> int:
        """Number of internal chessboard corners on the board."""
        return (self.squares_x - 1) * (self.squares_y - 1)

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "ChArUco",
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_length_mm": self.square_length_mm,
            "marker_length_mm": self.marker_length_mm,
            "dictionary": self.dictionary_name,
            "legacy_pattern": self.legacy_pattern,
        }


@dataclass(frozen=True)
class CalibrationQualityConfig:
    """Acquisition gates and robust-calibration thresholds."""

    min_charuco_corners: int = 15
    min_board_area_ratio: float = 0.02
    max_board_area_ratio: float = 0.85
    min_sharpness: float = 80.0
    min_calibration_images: int = 12
    good_tilt_min_deg: float = 10.0
    good_tilt_max_deg: float = 55.0
    coverage_rows: int = 4
    coverage_columns: int = 6
    minimum_view_error_threshold_px: float = 0.75
    outlier_sigma_multiplier: float = 3.0
    minimum_robust_sigma_px: float = 0.05
    maximum_outlier_iterations: int = 5
    minimum_pose_separation_deg: float = 10.0

    @property
    def coverage_cell_count(self) -> int:
        return self.coverage_rows * self.coverage_columns


DEFAULT_BOARD_SPEC = CharucoBoardSpec()
DEFAULT_QUALITY_CONFIG = CalibrationQualityConfig()


@dataclass(frozen=True)
class AnalyzedCalibrationImage:
    """Result returned to a UI or batch caller after one image is processed."""

    record: dict[str, Any]
    annotated_image: np.ndarray
    overlay_warnings: tuple[str, ...]
    coverage_fraction: float


class CharucoCalibrationSession:
    """Stateful, UI-independent ChArUco acquisition and calibration session."""

    def __init__(
        self,
        board_spec: CharucoBoardSpec = DEFAULT_BOARD_SPEC,
        quality: CalibrationQualityConfig = DEFAULT_QUALITY_CONFIG,
    ) -> None:
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoDetector"):
            raise RuntimeError(
                "当前OpenCV缺少aruco/CharucoDetector；请使用scara_cvdev环境。"
            )
        self.board_spec = board_spec
        self.quality = quality
        dictionary_id = getattr(cv2.aruco, board_spec.dictionary_name, None)
        if dictionary_id is None:
            raise ValueError(f"OpenCV不支持字典 {board_spec.dictionary_name}")
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.board = cv2.aruco.CharucoBoard(
            (board_spec.squares_x, board_spec.squares_y),
            board_spec.square_length_mm,
            board_spec.marker_length_mm,
            self.dictionary,
        )
        self.board.setLegacyPattern(board_spec.legacy_pattern)
        self.detector = cv2.aruco.CharucoDetector(self.board)

        self.records: list[dict[str, Any]] = []
        self._calibration_views: list[dict[str, Any]] = []
        self._detected_views: dict[int, dict[str, np.ndarray]] = {}
        self._coverage_cells: set[tuple[int, int]] = set()

    @property
    def coverage_cells(self) -> set[tuple[int, int]]:
        """Return a copy so callers cannot mutate session state accidentally."""
        return set(self._coverage_cells)

    @property
    def coverage_fraction(self) -> float:
        return len(self._coverage_cells) / float(self.quality.coverage_cell_count)

    @property
    def recommended_image_count(self) -> int:
        return sum(bool(record["recommended"]) for record in self.records)

    @staticmethod
    def _sharpness(gray: np.ndarray, points: Optional[np.ndarray]) -> float:
        """Return variance of the Laplacian, preferably inside the board.

        The heuristic is ``sharpness = Var(Laplacian(I))``.  Defocus and motion
        blur remove high-frequency edges and normally lower this score.
        """
        if points is None or len(points) < 3:
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mask = np.zeros(gray.shape, dtype=np.uint8)
        hull = cv2.convexHull(points.astype(np.float32)).astype(np.int32)
        cv2.fillConvexPoly(mask, hull, 255)
        values = cv2.Laplacian(gray, cv2.CV_64F)[mask > 0]
        return float(values.var()) if values.size else 0.0

    @staticmethod
    def _estimate_tilt(
        object_points: np.ndarray,
        image_points: np.ndarray,
        image_size: tuple[int, int],
    ) -> Optional[float]:
        """Estimate board tilt for live guidance before final intrinsics exist.

        A rough camera matrix feeds planar ``solvePnP``.  If ``R`` is the
        resulting board rotation, its camera-frame normal is
        ``n = R @ [0, 0, 1]`` and ``tilt = acos(abs(n_z))``.
        """
        width, height = image_size
        focal_guess = float(max(width, height))
        camera_guess = np.array(
            [
                [focal_guess, 0.0, width / 2.0],
                [0.0, focal_guess, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        try:
            ok, rvec, _tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_guess,
                np.zeros((5, 1), dtype=np.float64),
                flags=cv2.SOLVEPNP_IPPE,
            )
            if not ok:
                return None
            rotation, _ = cv2.Rodrigues(rvec)
            normal_z = min(1.0, max(0.0, abs(float(rotation[2, 2]))))
            return math.degrees(math.acos(normal_z))
        except cv2.error:
            return None

    def _register_coverage(self, points: np.ndarray, width: int, height: int) -> None:
        """Register grid cells containing detected ChArUco corners."""
        for x, y in points.reshape(-1, 2):
            column = min(
                self.quality.coverage_columns - 1,
                max(
                    0,
                    int(float(x) / width * self.quality.coverage_columns),
                ),
            )
            row = min(
                self.quality.coverage_rows - 1,
                max(
                    0,
                    int(float(y) / height * self.quality.coverage_rows),
                ),
            )
            self._coverage_cells.add((row, column))

    def analyze_image(
        self,
        image: np.ndarray,
        filename: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> AnalyzedCalibrationImage:
        """Detect and score one image, storing it as part of this session."""
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("标定图像必须是有效的BGR三通道图像")

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            self.detector.detectBoard(image)
        )
        corner_count = 0 if charuco_ids is None else int(len(charuco_ids))
        marker_count = 0 if marker_ids is None else int(len(marker_ids))

        flat_corners: Optional[np.ndarray] = None
        object_points: Optional[np.ndarray] = None
        image_points: Optional[np.ndarray] = None
        if corner_count:
            flat_corners = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)
            self._register_coverage(flat_corners, width, height)
        if corner_count >= 4 and not self.board.checkCharucoCornersCollinear(
            charuco_ids
        ):
            object_points, image_points = self.board.matchImagePoints(
                charuco_corners, charuco_ids
            )
            object_points = np.asarray(object_points, dtype=np.float32).reshape(
                -1, 1, 3
            )
            image_points = np.asarray(image_points, dtype=np.float32).reshape(
                -1, 1, 2
            )

        board_area_ratio = 0.0
        border_margin_px: Optional[float] = None
        if flat_corners is not None and len(flat_corners) >= 3:
            # rho = area(convex hull of detected corners) / image area.
            board_area_ratio = float(
                cv2.contourArea(cv2.convexHull(flat_corners))
            ) / float(width * height)
            border_margin_px = float(
                min(
                    flat_corners[:, 0].min(),
                    flat_corners[:, 1].min(),
                    width - 1 - flat_corners[:, 0].max(),
                    height - 1 - flat_corners[:, 1].max(),
                )
            )

        sharpness = self._sharpness(gray, flat_corners)
        tilt = None
        if object_points is not None and image_points is not None:
            tilt = self._estimate_tilt(
                object_points, image_points, (width, height)
            )
        tilt_enough = (
            tilt is not None
            and self.quality.good_tilt_min_deg
            <= tilt
            <= self.quality.good_tilt_max_deg
        )
        sharpness_enough = sharpness >= self.quality.min_sharpness

        reasons: list[str] = []
        if corner_count < self.quality.min_charuco_corners:
            reasons.append(f"角点少于{self.quality.min_charuco_corners}")
        if not (
            self.quality.min_board_area_ratio
            <= board_area_ratio
            <= self.quality.max_board_area_ratio
        ):
            reasons.append("板在图像中的面积不合适")
        if border_margin_px is not None and border_margin_px < 4.0:
            reasons.append("标定板太靠近/超出边缘")
        if not sharpness_enough:
            reasons.append("图像模糊")

        # A few fronto-parallel views are useful.  Starting with the fifth,
        # recommend changing tilt rather than admitting endless duplicates.
        low_tilt_count = sum(
            record.get("tilt_deg") is not None
            and float(record["tilt_deg"]) < self.quality.good_tilt_min_deg
            and bool(record.get("recommended"))
            for record in self.records
        )
        if not tilt_enough and low_tilt_count >= 4:
            reasons.append("倾斜变化不足")

        recommended = object_points is not None and not reasons
        record: dict[str, Any] = {
            "capture_sequence": len(self.records) + 1,
            "filename": str(filename),
            **dict(metadata or {}),
            "resolution": {"width": width, "height": height},
            "marker_count": marker_count,
            "charuco_corner_count": corner_count,
            "board_area_ratio": board_area_ratio,
            "border_margin_px": border_margin_px,
            "sharpness": sharpness,
            "sharpness_enough": sharpness_enough,
            "tilt_deg": tilt,
            "tilt_enough": tilt_enough,
            "recommended": bool(recommended),
            "reasons": reasons,
            "reprojection_error_px": None,
            "used_for_final_calibration": False,
        }
        self.records.append(record)
        record_index = len(self.records) - 1
        if object_points is not None and image_points is not None:
            self._detected_views[record_index] = {
                "object": object_points,
                "image": image_points,
            }
        if object_points is not None and image_points is not None and recommended:
            self._calibration_views.append(
                {
                    "record_index": record_index,
                    "object": object_points,
                    "image": image_points,
                }
            )

        annotated = image.copy()
        overlay_warnings: list[str] = []
        if marker_ids is not None:
            try:
                draw_marker_ids = np.ascontiguousarray(
                    marker_ids, dtype=np.int32
                ).reshape(-1, 1)
                cv2.aruco.drawDetectedMarkers(
                    annotated, marker_corners, draw_marker_ids
                )
            except (cv2.error, TypeError, ValueError) as exc:
                overlay_warnings.append(f"Marker覆盖层失败：{exc}")
        if charuco_ids is not None and charuco_corners is not None:
            try:
                # OpenCV 5 may return (N,2)/(N,), while the drawing binding
                # requires vector-compatible (N,1,2)/(N,1) arrays.
                draw_charuco_corners = np.ascontiguousarray(
                    charuco_corners, dtype=np.float32
                ).reshape(-1, 1, 2)
                draw_charuco_ids = np.ascontiguousarray(
                    charuco_ids, dtype=np.int32
                ).reshape(-1, 1)
                cv2.aruco.drawDetectedCornersCharuco(
                    annotated,
                    draw_charuco_corners,
                    draw_charuco_ids,
                )
            except (cv2.error, TypeError, ValueError) as exc:
                overlay_warnings.append(f"ChArUco覆盖层失败：{exc}")

        return AnalyzedCalibrationImage(
            record=record,
            annotated_image=annotated,
            overlay_warnings=tuple(overlay_warnings),
            coverage_fraction=self.coverage_fraction,
        )

    @staticmethod
    def _run_calibration(
        views: list[dict[str, Any]], image_size: tuple[int, int]
    ) -> dict[str, Any]:
        """Minimize global squared ChArUco reprojection error.

        For board point ``P_j`` in image ``i``::

            P_C = R_i P_j + t_i
            x = X_C/Z_C, y = Y_C/Z_C
            p_hat_ij = project(P_j; K, dist, R_i, t_i)

        OpenCV jointly minimizes::

            sum_i sum_j ||p_ij - p_hat_ij||^2

        ``K`` and distortion are global; each image has its own ``R_i,t_i``.
        """
        object_points = [view["object"] for view in views]
        image_points = [view["image"] for view in views]
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            1e-9,
        )
        (
            rms,
            camera_matrix,
            dist_coeffs,
            rvecs,
            tvecs,
            std_intrinsics,
            std_extrinsics,
            per_view_errors,
        ) = cv2.calibrateCameraExtended(
            object_points,
            image_points,
            image_size,
            None,
            None,
            flags=0,
            criteria=criteria,
        )
        return {
            "rms": float(rms),
            "K": camera_matrix,
            "dist": dist_coeffs,
            "rvecs": rvecs,
            "tvecs": tvecs,
            "std_intrinsics": std_intrinsics,
            "std_extrinsics": std_extrinsics,
            "errors": np.asarray(per_view_errors, dtype=np.float64).reshape(-1),
        }

    def _pose_diversity(self, rvecs: Sequence[np.ndarray]) -> dict[str, Any]:
        """Measure angular diversity between accepted board normals.

        ``n_i = R_i @ [0,0,1]`` and pairwise separation is
        ``alpha_ij = acos(clamp(n_i dot n_j, -1, 1))``.
        """
        normals: list[np.ndarray] = []
        tilts: list[float] = []
        for rvec in rvecs:
            rotation, _ = cv2.Rodrigues(rvec)
            normal = np.asarray(rotation[:, 2], dtype=np.float64).reshape(3)
            if normal[2] < 0:
                normal = -normal
            normals.append(normal / np.linalg.norm(normal))
            tilts.append(
                math.degrees(
                    math.acos(
                        min(1.0, max(0.0, abs(float(normal[2]))))
                    )
                )
            )

        maximum_separation = 0.0
        for first in range(len(normals)):
            for second in range(first + 1, len(normals)):
                cosine = float(
                    np.clip(np.dot(normals[first], normals[second]), -1.0, 1.0)
                )
                maximum_separation = max(
                    maximum_separation,
                    math.degrees(math.acos(cosine)),
                )
        maximum_tilt = max(tilts) if tilts else 0.0
        sufficient = (
            maximum_separation >= self.quality.minimum_pose_separation_deg
            and maximum_tilt >= self.quality.good_tilt_min_deg
        )
        return {
            "tilt_min_deg": min(tilts) if tilts else None,
            "tilt_max_deg": max(tilts) if tilts else None,
            "tilt_range_deg": (max(tilts) - min(tilts)) if tilts else None,
            "maximum_board_normal_separation_deg": maximum_separation,
            "sufficient": sufficient,
        }

    def calibrate_with_outlier_rejection(
        self, image_size: tuple[int, int]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Calibrate, reject robust per-view error outliers, and recalibrate.

        With per-view errors ``e_i``::

            median = median(e)
            MAD = median(|e_i - median|)
            robust_sigma = 1.4826 * MAD
            threshold = max(0.75 px, median + 3 * robust_sigma)

        Configuration values replace the literal constants in the actual code.
        """
        active = list(self._calibration_views)
        history: list[dict[str, Any]] = []
        first_errors: dict[int, float] = {}
        # Make repeated calibration calls deterministic: a view rejected in a
        # later run must not retain ``used=True`` from an earlier successful run.
        for record in self.records:
            record["used_for_final_calibration"] = False
        if len(active) < self.quality.min_calibration_images:
            raise RuntimeError(
                f"只有{len(active)}张照片通过实时质量筛选，至少需要"
                f"{self.quality.min_calibration_images}张。"
                "请保证板完整清晰，并手动改变倾角后重试。"
            )

        result: Optional[dict[str, Any]] = None
        for iteration in range(1, self.quality.maximum_outlier_iterations + 1):
            result = self._run_calibration(active, image_size)
            errors = result["errors"]
            for view, error in zip(active, errors):
                first_errors.setdefault(int(view["record_index"]), float(error))
            median = float(np.median(errors))
            mad = float(np.median(np.abs(errors - median)))
            robust_sigma = 1.4826 * mad
            threshold = max(
                self.quality.minimum_view_error_threshold_px,
                median
                + self.quality.outlier_sigma_multiplier
                * max(
                    robust_sigma,
                    self.quality.minimum_robust_sigma_px,
                ),
            )
            candidates = [
                index
                for index, error in enumerate(errors)
                if float(error) > threshold
            ]
            maximum_removal = max(
                0, len(active) - self.quality.min_calibration_images
            )
            if len(candidates) > maximum_removal:
                candidates = sorted(
                    candidates,
                    key=lambda index: float(errors[index]),
                    reverse=True,
                )[:maximum_removal]
            rejected_records = [
                int(active[index]["record_index"]) for index in candidates
            ]
            history.append(
                {
                    "iteration": iteration,
                    "views": len(active),
                    "rms_px": result["rms"],
                    "median_view_error_px": median,
                    "mad_px": mad,
                    "threshold_px": threshold,
                    "rejected_filenames": [
                        self.records[index]["filename"]
                        for index in rejected_records
                    ],
                }
            )
            if not candidates:
                break
            rejected_set = set(candidates)
            active = [
                view
                for index, view in enumerate(active)
                if index not in rejected_set
            ]

        if result is None:
            raise RuntimeError("相机标定没有产生结果")
        if len(result["errors"]) != len(active):
            result = self._run_calibration(active, image_size)

        for record_index, error in first_errors.items():
            self.records[record_index]["reprojection_error_px"] = error
        for view, error in zip(active, result["errors"]):
            record = self.records[int(view["record_index"])]
            record["reprojection_error_px"] = float(error)
            record["used_for_final_calibration"] = True

        result["pose_diversity"] = self._pose_diversity(result["rvecs"])
        self.update_all_reprojection_errors(result)
        return result, history

    def update_all_reprojection_errors(self, result: Mapping[str, Any]) -> None:
        """Evaluate every detected image using final fixed intrinsics.

        For ``N`` matched points, per-view error is::

            RMS = sqrt((1/N) * sum_j (du_j**2 + dv_j**2))

        The unit is pixels.  Each image's pose is re-estimated by ``solvePnP``.
        """
        camera_matrix = np.asarray(result["K"], dtype=np.float64)
        dist_coeffs = np.asarray(result["dist"], dtype=np.float64)
        for record_index, view in self._detected_views.items():
            object_points = view["object"]
            image_points = view["image"]
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if not ok:
                    continue
                projected, _ = cv2.projectPoints(
                    object_points,
                    rvec,
                    tvec,
                    camera_matrix,
                    dist_coeffs,
                )
                residual = np.asarray(
                    projected, dtype=np.float64
                ).reshape(-1, 2) - np.asarray(
                    image_points, dtype=np.float64
                ).reshape(-1, 2)
                rms = math.sqrt(
                    float(np.mean(np.sum(residual * residual, axis=1)))
                )
                self.records[record_index]["reprojection_error_px"] = rms
            except cv2.error:
                continue


__all__ = [
    "AnalyzedCalibrationImage",
    "CalibrationQualityConfig",
    "CharucoBoardSpec",
    "CharucoCalibrationSession",
    "DEFAULT_BOARD_SPEC",
    "DEFAULT_QUALITY_CONFIG",
]
