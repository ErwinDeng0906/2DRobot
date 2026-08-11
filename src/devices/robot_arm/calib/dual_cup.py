"""双杯工具 TCP、工作平面法向与不可变工具契约的纯离线标定。"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Iterable, List

import numpy as np


TOOL_SCHEMA = "dual-cup-tool/v1"
MIN_PIVOT_SAMPLES = 6
MIN_PIVOT_ORIENTATION_SPREAD_DEG = 30.0
MAX_PIVOT_CONDITION_NUMBER = 100.0
MAX_PIVOT_RMS_RESIDUAL_MM = 0.5
MAX_PIVOT_RESIDUAL_MM = 1.0
MIN_PLANE_SAMPLES = 3
MIN_PLANE_ORIENTATION_SPREAD_DEG = 15.0
MAX_PLANE_ANGULAR_RESIDUAL_DEG = 0.1


class DualCupCalibrationError(RuntimeError):
    pass


def rpy_matrix(rpy: Iterable[float]) -> np.ndarray:
    rx, ry, rz = (float(value) for value in rpy)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.asarray(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=float,
    )


def _poses(values: Iterable[Iterable[float]], minimum: int, label: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if (
        array.ndim != 2
        or array.shape[1] != 6
        or array.shape[0] < minimum
        or not np.all(np.isfinite(array))
    ):
        raise DualCupCalibrationError(
            f"{label} 至少需要 {minimum} 组 [x,y,z,rx,ry,rz] 有限数"
        )
    return array


def _orientation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _orientation_spread_deg(rotations: List[np.ndarray]) -> float:
    return max(
        _orientation_error_deg(rotations[first], rotations[second])
        for first in range(len(rotations))
        for second in range(first + 1, len(rotations))
    )


def solve_tcp_pivot(
    flange_poses: Iterable[Iterable[float]],
    *,
    minimum_samples: int = MIN_PIVOT_SAMPLES,
    minimum_orientation_spread_deg: float = MIN_PIVOT_ORIENTATION_SPREAD_DEG,
    maximum_condition_number: float = MAX_PIVOT_CONDITION_NUMBER,
    maximum_rms_residual_mm: float = MAX_PIVOT_RMS_RESIDUAL_MM,
    maximum_residual_mm: float = MAX_PIVOT_RESIDUAL_MM,
) -> Dict[str, Any]:
    """由同一固定尖点的多姿态零工具法兰 FK 解法兰到双杯中心平移。"""
    poses = _poses(flange_poses, minimum_samples, "pivot flange_poses")
    rotations = [rpy_matrix(pose[3:]) for pose in poses]
    spread_deg = _orientation_spread_deg(rotations)
    if spread_deg < float(minimum_orientation_spread_deg):
        raise DualCupCalibrationError(
            f"pivot 姿态跨度 {spread_deg:.3f}deg 不足"
        )
    system = []
    target = []
    for pose, rotation in zip(poses, rotations):
        system.append(np.hstack([rotation, -np.eye(3)]))
        target.append(-pose[:3])
    matrix = np.vstack(system)
    vector = np.hstack(target)
    if int(np.linalg.matrix_rank(matrix)) < 6:
        raise DualCupCalibrationError("pivot 方程秩不足，姿态组合退化")
    condition_number = float(np.linalg.cond(matrix))
    if not math.isfinite(condition_number) or condition_number > maximum_condition_number:
        raise DualCupCalibrationError(
            f"pivot 方程条件数 {condition_number:.3f} 过大"
        )
    solution, _, _, _ = np.linalg.lstsq(matrix, vector, rcond=None)
    tcp_position = solution[:3]
    fixed_point = solution[3:]
    residuals_mm = np.asarray(
        [
            1000.0
            * np.linalg.norm(rotation @ tcp_position + pose[:3] - fixed_point)
            for pose, rotation in zip(poses, rotations)
        ],
        dtype=float,
    )
    rms_mm = float(np.sqrt(np.mean(residuals_mm ** 2)))
    max_mm = float(np.max(residuals_mm))
    if rms_mm > maximum_rms_residual_mm or max_mm > maximum_residual_mm:
        raise DualCupCalibrationError(
            "pivot 残差超限: rms=%.3fmm max=%.3fmm" % (rms_mm, max_mm)
        )
    return {
        "tcp_position_m": tcp_position.tolist(),
        "fixed_point_base_m": fixed_point.tolist(),
        "samples": int(poses.shape[0]),
        "orientation_spread_deg": spread_deg,
        "rank": int(np.linalg.matrix_rank(matrix)),
        "condition_number": condition_number,
        "residual_mm": {
            "rms": rms_mm,
            "max": max_mm,
            "per_sample": residuals_mm.tolist(),
        },
    }


def solve_plane_normal(
    parallel_tcp_rpy: Iterable[Iterable[float]],
    *,
    table_normal_base: Iterable[float] = (0.0, 0.0, 1.0),
    minimum_samples: int = MIN_PLANE_SAMPLES,
    minimum_orientation_spread_deg: float = MIN_PLANE_ORIENTATION_SPREAD_DEG,
    maximum_angular_residual_deg: float = MAX_PLANE_ANGULAR_RESIDUAL_DEG,
) -> Dict[str, Any]:
    """由多组已确认平行桌面的 TCP 姿态求双杯工作平面法向。"""
    rpy_values = np.asarray(list(parallel_tcp_rpy), dtype=float)
    if (
        rpy_values.ndim != 2
        or rpy_values.shape[1] != 3
        or rpy_values.shape[0] < minimum_samples
        or not np.all(np.isfinite(rpy_values))
    ):
        raise DualCupCalibrationError(
            f"parallel_tcp_rpy 至少需要 {minimum_samples} 组 RPY 有限数"
        )
    table_normal = np.asarray(table_normal_base, dtype=float)
    if table_normal.shape != (3,) or not np.all(np.isfinite(table_normal)):
        raise DualCupCalibrationError("table_normal_base 必须是三个有限数")
    norm = float(np.linalg.norm(table_normal))
    if norm <= 0.0:
        raise DualCupCalibrationError("table_normal_base 不得为零向量")
    table_normal /= norm
    rotations = [rpy_matrix(rpy) for rpy in rpy_values]
    spread_deg = _orientation_spread_deg(rotations)
    if spread_deg < float(minimum_orientation_spread_deg):
        raise DualCupCalibrationError(
            f"平面样本姿态跨度 {spread_deg:.3f}deg 不足"
        )
    candidates = [rotation.T @ table_normal for rotation in rotations]
    mean = np.mean(np.asarray(candidates), axis=0)
    mean_norm = float(np.linalg.norm(mean))
    if mean_norm <= 0.0:
        raise DualCupCalibrationError("平面法向样本互相抵消")
    normal = mean / mean_norm
    residuals_deg = [
        math.degrees(
            math.acos(float(np.clip(candidate @ normal, -1.0, 1.0)))
        )
        for candidate in candidates
    ]
    max_residual_deg = max(residuals_deg)
    if max_residual_deg > float(maximum_angular_residual_deg):
        raise DualCupCalibrationError(
            f"平面法向残差 {max_residual_deg:.6f}deg 超限"
        )
    return {
        "cup_plane_normal_tcp": normal.tolist(),
        "samples": int(rpy_values.shape[0]),
        "orientation_spread_deg": spread_deg,
        "angular_residual_deg": {
            "max": max_residual_deg,
            "per_sample": residuals_deg,
        },
    }


def _canonical_sha256(payload: Dict[str, Any]) -> str:
    content = dict(payload)
    content.pop("sha256", None)
    encoded = json.dumps(
        content,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def build_qualified_tool_contract(
    pivot: Dict[str, Any],
    plane: Dict[str, Any],
    payload_kg_cog_m: Iterable[float],
    inertia_tensor_kg_m2: Iterable[float],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """把已通过的几何和负载结果冻结为可供执行层复核的工具契约。"""
    if not isinstance(pivot, dict) or not isinstance(plane, dict):
        raise DualCupCalibrationError("pivot 和 plane 必须是标定结果字典")
    pivot_residual = pivot.get("residual_mm")
    plane_residual = plane.get("angular_residual_deg")
    if not isinstance(pivot_residual, dict) or not isinstance(plane_residual, dict):
        raise DualCupCalibrationError("pivot 或 plane 缺少残差证据")
    fixed_metrics = {
        "pivot.samples": (pivot.get("samples"), MIN_PIVOT_SAMPLES, None),
        "pivot.orientation_spread_deg": (
            pivot.get("orientation_spread_deg"),
            MIN_PIVOT_ORIENTATION_SPREAD_DEG,
            None,
        ),
        "pivot.condition_number": (
            pivot.get("condition_number"),
            None,
            MAX_PIVOT_CONDITION_NUMBER,
        ),
        "pivot.residual_mm.rms": (
            pivot_residual.get("rms"),
            None,
            MAX_PIVOT_RMS_RESIDUAL_MM,
        ),
        "pivot.residual_mm.max": (
            pivot_residual.get("max"),
            None,
            MAX_PIVOT_RESIDUAL_MM,
        ),
        "plane.samples": (plane.get("samples"), MIN_PLANE_SAMPLES, None),
        "plane.orientation_spread_deg": (
            plane.get("orientation_spread_deg"),
            MIN_PLANE_ORIENTATION_SPREAD_DEG,
            None,
        ),
        "plane.angular_residual_deg.max": (
            plane_residual.get("max"),
            None,
            MAX_PLANE_ANGULAR_RESIDUAL_DEG,
        ),
    }
    for label, (raw_value, minimum, maximum) in fixed_metrics.items():
        if isinstance(raw_value, bool):
            raise DualCupCalibrationError(f"{label} 不是有限标定数值")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise DualCupCalibrationError(f"{label} 不是有限标定数值") from exc
        if not math.isfinite(value):
            raise DualCupCalibrationError(f"{label} 不是有限标定数值")
        if minimum is not None and value < minimum:
            raise DualCupCalibrationError(f"{label} 未达到固定合格下限 {minimum}")
        if maximum is not None and value > maximum:
            raise DualCupCalibrationError(f"{label} 超过固定合格上限 {maximum}")
    rank_value = pivot.get("rank")
    if isinstance(rank_value, bool):
        raise DualCupCalibrationError("pivot.rank 不是有限标定数值")
    try:
        rank = float(rank_value)
    except (TypeError, ValueError) as exc:
        raise DualCupCalibrationError("pivot.rank 不是有限标定数值") from exc
    if not math.isfinite(rank) or rank < 6:
        raise DualCupCalibrationError("pivot.rank 未达到固定合格下限 6")
    if not isinstance(evidence, dict):
        raise DualCupCalibrationError("calibration_evidence 必须是字典")
    dataset_sha256 = evidence.get("dataset_sha256")
    if (
        not isinstance(dataset_sha256, str)
        or len(dataset_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in dataset_sha256)
    ):
        raise DualCupCalibrationError("calibration_evidence.dataset_sha256 必须是 SHA256")
    payload = np.asarray(list(payload_kg_cog_m), dtype=float)
    inertia = np.asarray(list(inertia_tensor_kg_m2), dtype=float)
    if payload.shape != (4,) or not np.all(np.isfinite(payload)):
        raise DualCupCalibrationError("payload_kg_cog_m 必须是四个有限数")
    if payload[0] < 0.0 or payload[0] > 3.0:
        raise DualCupCalibrationError("payload 质量必须在 [0,3]kg")
    if inertia.shape != (6,) or not np.all(np.isfinite(inertia)):
        raise DualCupCalibrationError("inertia_tensor_kg_m2 必须是六个有限数")
    inertia_matrix = np.asarray(
        [
            [inertia[0], inertia[1], inertia[2]],
            [inertia[1], inertia[3], inertia[4]],
            [inertia[2], inertia[4], inertia[5]],
        ],
        dtype=float,
    )
    inertia_eigenvalues = np.linalg.eigvalsh(inertia_matrix)
    covariance_eigenvalues = np.linalg.eigvalsh(
        0.5 * float(np.trace(inertia_matrix)) * np.eye(3) - inertia_matrix
    )
    if (
        (payload[0] > 0.0 and float(np.trace(inertia_matrix)) <= 1e-12)
        or float(np.min(inertia_eigenvalues)) < -1e-12
        or float(np.min(covariance_eigenvalues)) < -1e-12
    ):
        raise DualCupCalibrationError("惯量不满足正半定与刚体三角一致性")
    tcp_position = np.asarray(pivot.get("tcp_position_m"), dtype=float)
    normal = np.asarray(plane.get("cup_plane_normal_tcp"), dtype=float)
    if (
        tcp_position.shape != (3,)
        or normal.shape != (3,)
        or not np.all(np.isfinite(tcp_position))
        or not np.all(np.isfinite(normal))
    ):
        raise DualCupCalibrationError("pivot 或 plane 结果格式错误")
    if abs(float(np.linalg.norm(normal)) - 1.0) > 1e-6:
        raise DualCupCalibrationError("plane 工具法向必须是单位向量")
    contract = {
        "schema": TOOL_SCHEMA,
        "qualified": True,
        "orientation_convention": "ZYX_Rz_Ry_Rx",
        "tcp_offset_m": tcp_position.tolist() + [0.0, 0.0, 0.0],
        "cup_plane_normal_tcp": normal.tolist(),
        "payload_kg_cog_m": payload.tolist(),
        "inertia_tensor_kg_m2": inertia.tolist(),
        "independent_center_calibration": True,
        "independent_plane_calibration": True,
        "payload_verified": True,
        "inertia_verified": True,
        "calibration_evidence": dict(evidence),
        "pivot_result": pivot,
        "plane_result": plane,
    }
    contract["sha256"] = _canonical_sha256(contract)
    return contract
