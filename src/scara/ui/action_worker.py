"""Execute imported SCARA action files without blocking the Qt UI.

The action format deliberately separates movement, dwell, still capture, video
recording, and state recording.  An imported Python file only *describes* those
operations; hardware access happens here after the operator confirms the run.
"""

from __future__ import annotations

import json
import math
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

from PyQt6.QtCore import QThread, pyqtSignal

from scara.controller.scara_controller import ScaraController
from scara.file_io import atomic_write_text


ACTION_API_VERSION = 1
SUPPORTED_ACTION_TYPES = {
    "assert_joints",
    "move_joints",
    "move_xyzr",
    "wait",
    "capture",
    "start_video",
    "stop_video",
    "record_point",
    "operator_checkpoint",
}


def _finite(value: object, label: str) -> float:
    """Convert one action parameter to a finite float or raise a useful error."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是数字") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} 必须是有限数字")
    return result


def _joint_values(value: object, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{label} 必须包含 J1/J2/J3/J4 四个值")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def normalize_action_task(raw_task: object) -> dict:
    """Validate an ``ACTION_API_VERSION = 1`` task and return a safe copy."""
    if not isinstance(raw_task, dict):
        raise ValueError("build_action() 必须返回字典")

    task_name = str(raw_task.get("name") or "未命名动作").strip()
    if not task_name:
        raise ValueError("动作 name 不能为空")

    raw_camera = raw_task.get("camera_model") or {}
    if not isinstance(raw_camera, dict):
        raise ValueError("camera_model 必须是字典")
    camera_model = {
        "offset_mm": _finite(raw_camera.get("offset_mm", 20.0), "camera_model.offset_mm"),
        "angle_reference": str(
            raw_camera.get("angle_reference", "world_negative_y")
        ).strip(),
        "positive_rotation": str(
            raw_camera.get("positive_rotation", "counter_clockwise_from_above")
        ).strip(),
    }
    if camera_model["offset_mm"] < 0:
        raise ValueError("camera_model.offset_mm 不能为负数")
    if camera_model["angle_reference"] != "world_negative_y":
        raise ValueError("camera_model.angle_reference 必须是 world_negative_y")
    if camera_model["positive_rotation"] != "counter_clockwise_from_above":
        raise ValueError(
            "camera_model.positive_rotation 必须是 counter_clockwise_from_above"
        )

    raw_actions = raw_task.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError("actions 必须是非空列表")

    actions: list[dict] = []
    active_video_sources: set[int] = set()
    video_filenames: set[str] = set()
    for index, raw in enumerate(raw_actions, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"第 {index} 步必须是字典")
        kind = str(raw.get("type") or "").strip()
        if kind not in SUPPORTED_ACTION_TYPES:
            allowed = ", ".join(sorted(SUPPORTED_ACTION_TYPES))
            raise ValueError(f"第 {index} 步 type={kind!r} 不受支持；允许：{allowed}")

        step: dict = {"type": kind}
        if kind in {
            "assert_joints",
            "move_joints",
            "move_xyzr",
            "record_point",
            "operator_checkpoint",
        }:
            name = str(raw.get("name") or f"步骤 {index}").strip()
            if not name:
                raise ValueError(f"第 {index} 步 name 不能为空")
            step["name"] = name

        if kind in {"assert_joints", "move_joints"}:
            step["joints"] = _joint_values(raw.get("joints"), f"第 {index} 步 joints")
            step["tolerance"] = _finite(raw.get("tolerance", 0.2), f"第 {index} 步 tolerance")
            if step["tolerance"] <= 0:
                raise ValueError(f"第 {index} 步 tolerance 必须大于 0")
            if kind == "move_joints" and "require_current_j3_mm" in raw:
                step["require_current_j3_mm"] = _finite(
                    raw["require_current_j3_mm"],
                    f"第 {index} 步 require_current_j3_mm",
                )
                step["j3_tolerance_mm"] = _finite(
                    raw.get("j3_tolerance_mm", 0.2),
                    f"第 {index} 步 j3_tolerance_mm",
                )
                if step["j3_tolerance_mm"] <= 0:
                    raise ValueError(f"第 {index} 步 j3_tolerance_mm 必须大于 0")
        elif kind == "move_xyzr":
            for key in ("x_mm", "y_mm", "z_mm", "r_deg"):
                step[key] = _finite(raw.get(key, 0.0), f"第 {index} 步 {key}")
            if not any(abs(step[key]) > 1e-12 for key in ("x_mm", "y_mm", "z_mm", "r_deg")):
                raise ValueError(f"第 {index} 步 move_xyzr 至少要有一个非零增量")
        elif kind == "wait":
            step["seconds"] = _finite(raw.get("seconds"), f"第 {index} 步 seconds")
            if step["seconds"] < 0:
                raise ValueError(f"第 {index} 步 seconds 不能为负数")
        elif kind == "operator_checkpoint":
            if active_video_sources:
                raise ValueError(
                    f"第 {index} 步人工确认前必须先停止全部录像"
                )
            message = str(raw.get("message") or "").strip()
            if not message:
                raise ValueError(f"第 {index} 步 operator_checkpoint.message 不能为空")
            continue_text = str(
                raw.get("continue_text") or "继续采集"
            ).strip()
            finish_text = str(raw.get("finish_text") or "结束采集").strip()
            if not continue_text or not finish_text:
                raise ValueError(
                    f"第 {index} 步人工确认的两个按钮文字都不能为空"
                )
            repeat_from_index = raw.get("repeat_from_index")
            if (
                isinstance(repeat_from_index, bool)
                or not isinstance(repeat_from_index, int)
            ):
                raise ValueError(
                    f"第 {index} 步 repeat_from_index 必须是从0开始的整数"
                )
            if repeat_from_index < 0 or repeat_from_index >= index - 1:
                raise ValueError(
                    f"第 {index} 步 repeat_from_index 必须指向此前的动作"
                )
            if str(raw_actions[repeat_from_index].get("type") or "").strip() == (
                "operator_checkpoint"
            ):
                raise ValueError(
                    f"第 {index} 步不能跳回另一个 operator_checkpoint"
                )
            step.update(
                {
                    "message": message,
                    "continue_text": continue_text,
                    "finish_text": finish_text,
                    "repeat_from_index": repeat_from_index,
                }
            )
        elif kind in {"capture", "start_video", "stop_video"}:
            source = raw.get("source")
            if isinstance(source, bool) or not isinstance(source, int) or not 0 <= source <= 8:
                raise ValueError(f"第 {index} 步 source 必须是 0 到 8 的整数")
            step["source"] = source
            if kind == "start_video":
                if source in active_video_sources:
                    raise ValueError(f"第 {index} 步 相机源#{source}已在录像")
                filename = str(raw.get("filename") or f"{source}_video.avi").strip()
                suffix = Path(filename).suffix.lower()
                if (
                    not filename
                    or Path(filename).name != filename
                    or suffix not in {".avi", ".mp4"}
                ):
                    raise ValueError(
                        f"第 {index} 步 filename 必须是当前实验文件夹内的 .avi 或 .mp4 文件名"
                    )
                if filename in video_filenames:
                    raise ValueError(f"第 {index} 步 录像文件名重复：{filename}")
                fps = _finite(raw.get("fps", 20.0), f"第 {index} 步 fps")
                if not 0.0 < fps <= 120.0:
                    raise ValueError(f"第 {index} 步 fps 必须在 (0, 120] 范围内")
                step["filename"] = filename
                step["fps"] = fps
                active_video_sources.add(source)
                video_filenames.add(filename)
            elif kind == "stop_video":
                if source not in active_video_sources:
                    raise ValueError(f"第 {index} 步 相机源#{source}尚未开始录像")
                active_video_sources.remove(source)

        actions.append(step)

    if active_video_sources:
        sources = ", ".join(f"#{source}" for source in sorted(active_video_sources))
        raise ValueError(f"录像步骤缺少 stop_video：相机源 {sources}")

    return {
        "api_version": ACTION_API_VERSION,
        "name": task_name,
        "description": str(raw_task.get("description") or "").strip(),
        "camera_model": camera_model,
        "actions": actions,
    }


def calculate_camera_position(pose: Sequence[float], camera_model: dict) -> dict:
    """Calculate the camera centre from the measured TCP pose.

    Coordinate/equation annotations:

    * ``rz_rad = Rz*pi/180`` converts the controller's degree value to radians.
    * Rz is measured counter-clockwise from world -Y.  Rotating the base vector
      ``(0, -d)`` gives ``offset_x = d*sin(rz_rad)`` and
      ``offset_y = -d*cos(rz_rad)``.
    * ``camera_x = centre_x + offset_x`` and
      ``camera_y = centre_y + offset_y`` translate that 20 mm radial vector
      from the suction-cup/wafer centre into world coordinates.
    * ``camera_z = centre_z`` assumes no measured vertical camera offset.
    """
    if len(pose) != 6:
        raise ValueError("位姿必须包含 x/y/z/Rx/Ry/Rz 六个值")
    values = [_finite(value, f"pose[{index}]") for index, value in enumerate(pose)]
    centre_x_mm, centre_y_mm, centre_z_mm, _rx_deg, _ry_deg, rz_deg = values
    offset_mm = _finite(camera_model["offset_mm"], "camera_model.offset_mm")
    rz_rad = math.radians(rz_deg)
    camera_offset_x_mm = offset_mm * math.sin(rz_rad)
    camera_offset_y_mm = -offset_mm * math.cos(rz_rad)
    camera_x_mm = centre_x_mm + camera_offset_x_mm
    camera_y_mm = centre_y_mm + camera_offset_y_mm
    camera_z_mm = centre_z_mm
    return {
        "x_mm": camera_x_mm,
        "y_mm": camera_y_mm,
        "z_mm": camera_z_mm,
        "angle_from_negative_y_deg": rz_deg,
        "offset_mm": offset_mm,
    }


class CameraSourcePool:
    """Keep all required OpenCV camera sources open for one action run."""

    def __init__(self, width: int = 1280, height: int = 720):
        self._width = int(width)
        self._height = int(height)
        self._captures: dict[int, object] = {}
        self._cv2 = None
        self._video_sessions: dict[int, dict] = {}
        self._video_error_lock = threading.Lock()
        self._video_error: Optional[str] = None

    def open_sources(self, sources: Sequence[int]) -> tuple[bool, str]:
        try:
            import cv2
        except Exception as exc:
            return False, f"未安装 opencv-python: {exc}"
        self._cv2 = cv2
        for source in sorted(set(int(value) for value in sources)):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                self.close()
                return False, f"无法打开相机源#{source}"
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._captures[source] = cap

        # Opening a DirectShow device does not guarantee that it can already
        # return frames.  Validate every required source before the first robot
        # movement so a cold/unavailable source #2 cannot stop the experiment
        # after source #0/#1 have already saved their P00 images.
        for source in sorted(self._captures):
            if self._read_fresh_frame(source, attempts=12) is None:
                self.close()
                return False, f"相机源#{source}已打开，但预热后仍无法取帧"
        return True, ""

    def _read_fresh_frame(self, source: int, *, attempts: int = 8):
        """Read through buffered frames with bounded retries; return the newest."""
        cap = self._captures.get(int(source))
        if cap is None:
            return None
        for _attempt in range(max(1, int(attempts))):
            latest = None
            # Multiple reads drain frames buffered during the two-second dwell.
            for _ in range(4):
                ok, frame = cap.read()
                if not ok or frame is None or getattr(frame, "size", 1) == 0:
                    latest = None
                    break
                latest = frame
            if latest is not None:
                return latest
            time.sleep(0.05)
        return None

    def snapshot(self, source: int, path: Path) -> bool:
        if self._cv2 is None:
            return False
        frame = self._read_fresh_frame(source)
        if frame is None:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        return bool(self._cv2.imwrite(str(path), frame))

    def start_video(self, source: int, path: Path, fps: float) -> None:
        """Start a background AVI/MJPG or MP4/mp4v recording."""
        source = int(source)
        if self._cv2 is None or source not in self._captures:
            raise RuntimeError(f"相机源#{source}尚未打开")
        if source in self._video_sessions:
            raise RuntimeError(f"相机源#{source}已在录像")
        frame = self._read_fresh_frame(source)
        if frame is None:
            raise RuntimeError(f"相机源#{source}录像前无法取帧")
        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            raise RuntimeError(f"相机源#{source}录像画面尺寸无效")
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        codecs = {".avi": "MJPG", ".mp4": "mp4v"}
        codec = codecs.get(suffix)
        if codec is None:
            raise RuntimeError(f"不支持的录像格式：{suffix or '无扩展名'}")
        writer = self._cv2.VideoWriter(
            str(path),
            self._cv2.VideoWriter_fourcc(*codec),
            float(fps),
            (int(width), int(height)),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"无法创建相机源#{source}录像文件：{path.name}")

        stop_event = threading.Event()
        session = {
            "source": source,
            "path": Path(path),
            "fps": float(fps),
            "writer": writer,
            "stop_event": stop_event,
            "frame_count": 0,
            "started_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        }

        def record_loop() -> None:
            period_s = 1.0 / float(fps)
            deadline = time.monotonic()
            try:
                while not stop_event.is_set():
                    ok, next_frame = self._captures[source].read()
                    if not ok or next_frame is None or getattr(next_frame, "size", 1) == 0:
                        raise RuntimeError(f"相机源#{source}录像期间取帧失败")
                    writer.write(next_frame)
                    session["frame_count"] += 1
                    deadline += period_s
                    stop_event.wait(max(0.0, deadline - time.monotonic()))
            except Exception as exc:  # noqa: BLE001 - reported on worker thread
                with self._video_error_lock:
                    self._video_error = str(exc) or exc.__class__.__name__

        thread = threading.Thread(
            target=record_loop,
            name=f"scara-video-source-{source}",
            daemon=True,
        )
        session["thread"] = thread
        self._video_sessions[source] = session
        thread.start()

    def check_video_error(self) -> None:
        """Raise a background recorder error on the action worker thread."""
        with self._video_error_lock:
            message = self._video_error
            self._video_error = None
        if message:
            raise RuntimeError(message)

    def stop_video(self, source: int) -> dict:
        """Stop one recording and return JSON-serializable session metadata."""
        source = int(source)
        session = self._video_sessions.get(source)
        if session is None:
            raise RuntimeError(f"相机源#{source}尚未开始录像")
        session["stop_event"].set()
        session["thread"].join(timeout=3.0)
        if session["thread"].is_alive():
            raise RuntimeError(f"相机源#{source}录像线程无法停止")
        session["writer"].release()
        self._video_sessions.pop(source, None)
        self.check_video_error()
        frame_count = int(session["frame_count"])
        if frame_count < 1:
            raise RuntimeError(f"相机源#{source}录像没有写入任何画面")
        return {
            "source": source,
            "filename": session["path"].name,
            "fps": float(session["fps"]),
            "frame_count": frame_count,
            "started_at": session["started_at"],
            "finished_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        }

    def close(self) -> None:
        for source in list(self._video_sessions):
            try:
                self.stop_video(source)
            except Exception:
                session = self._video_sessions.pop(source, None)
                if session is not None:
                    try:
                        session["stop_event"].set()
                        session["writer"].release()
                    except Exception:
                        pass
        for cap in self._captures.values():
            try:
                cap.release()
            except Exception:
                pass
        self._captures.clear()


class ActionWorker(QThread):
    """Execute one validated action and persist photos/videos/point JSON."""

    progress = pyqtSignal(str)
    photo_saved = pyqtSignal(str)
    video_saved = pyqtSignal(str)
    point_recorded = pyqtSignal(str)
    operator_checkpoint_requested = pyqtSignal(str, str, str, str)
    run_finished = pyqtSignal(bool, str, str)

    def __init__(
        self,
        controller: ScaraController,
        task: dict,
        output_dir: Path,
        *,
        snapshot_source: Optional[Callable[[int, Path], bool]] = None,
        camera_position_calculator: Optional[Callable[[Sequence[float]], dict]] = None,
        source_position_calculators: Optional[
            dict[int, Callable[[Sequence[float], Sequence[float]], dict]]
        ] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._controller = controller
        self._task = normalize_action_task(task)
        self._output_dir = Path(output_dir)
        self._snapshot_source = snapshot_source
        self._camera_position_calculator = camera_position_calculator
        self._source_position_calculators = dict(source_position_calculators or {})
        self._camera_pool: Optional[CameraSourcePool] = None
        self._stop_requested = threading.Event()
        self._photo_counts: dict[int, int] = {}
        self._active_video_sources: set[int] = set()
        self._manifest: dict = {}
        self._operator_decision_event = threading.Event()
        self._operator_decision: Optional[bool] = None
        self._repeatable = any(
            step["type"] == "operator_checkpoint"
            for step in self._task["actions"]
        )
        self._collection_round = 1

    def request_stop(self) -> None:
        self._stop_requested.set()
        self._controller.emergency_stop()

    def respond_operator_checkpoint(self, continue_collection: bool) -> None:
        """Release a paused checkpoint from the Qt UI thread.

        ``True`` repeats the configured acquisition block.  ``False`` ends the
        acquisition normally, allowing the task runtime to calibrate all images
        already accumulated in the current output folder.
        """
        self._operator_decision = bool(continue_collection)
        self._operator_decision_event.set()

    def _interruptible_wait(self, seconds: float) -> bool:
        remaining = max(0.0, float(seconds))
        while remaining > 1e-9:
            if self._stop_requested.is_set():
                return False
            interval = min(0.1, remaining)
            time.sleep(interval)
            remaining -= interval
        return not self._stop_requested.is_set()

    @property
    def manifest_path(self) -> Path:
        return self._output_dir / "points.json"

    def _save_manifest(self) -> None:
        atomic_write_text(
            self.manifest_path,
            json.dumps(self._manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _read_state(self, context: str) -> dict:
        status = self._controller.read_all_sync()
        if not isinstance(status, dict):
            raise RuntimeError(f"{context}：读取机械臂状态失败")
        joints = status.get("joints")
        pose = status.get("pose")
        if not isinstance(joints, (list, tuple)) or len(joints) != 4:
            raise RuntimeError(f"{context}：状态中缺少 J1/J2/J3/J4")
        if not isinstance(pose, (list, tuple)) or len(pose) != 6:
            raise RuntimeError(f"{context}：状态中缺少 x/y/z/Rx/Ry/Rz")
        return {
            "joints": [_finite(value, f"{context}.joints") for value in joints],
            "pose": [_finite(value, f"{context}.pose") for value in pose],
        }

    def _record_point(self, name: str) -> None:
        state = self._read_state(name)
        joints = state["joints"]
        pose = state["pose"]
        if self._camera_position_calculator is None:
            raw_camera = calculate_camera_position(pose, self._task["camera_model"])
        else:
            raw_camera = self._camera_position_calculator(list(pose))
        if not isinstance(raw_camera, dict):
            raise RuntimeError(f"{name}：相机位置函数必须返回字典")
        camera = dict(raw_camera)
        for key in ("x_mm", "y_mm", "z_mm"):
            camera[key] = _finite(camera.get(key), f"{name}.camera_position.{key}")
        point = {
            "sequence": len(self._manifest["points"]) + 1,
            "name": name,
            "recorded_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "joints": {
                "J1_deg": joints[0],
                "J2_deg": joints[1],
                "J3_mm": joints[2],
                "J4_deg": joints[3],
            },
            "mechanical_center": {
                "x_mm": pose[0],
                "y_mm": pose[1],
                "z_mm": pose[2],
                "Rx_deg": pose[3],
                "Ry_deg": pose[4],
                "Rz_deg": pose[5],
            },
            "camera_position": camera,
        }
        if self._repeatable:
            point["collection_round"] = self._collection_round
        self._manifest["points"].append(point)
        self._save_manifest()
        self.point_recorded.emit(name)

    def _source_position_for_last_point(self, source: int) -> Optional[dict]:
        """Calculate optional source-specific XYZ for the last recorded point."""
        calculator = self._source_position_calculators.get(int(source))
        if calculator is None:
            return None
        if not self._manifest["points"]:
            raise RuntimeError(f"相机源#{source}位置计算前缺少 record_point 途径点")

        point = self._manifest["points"][-1]
        point_joints = point["joints"]
        centre = point["mechanical_center"]
        joints = [
            point_joints["J1_deg"],
            point_joints["J2_deg"],
            point_joints["J3_mm"],
            point_joints["J4_deg"],
        ]
        pose = [
            centre["x_mm"],
            centre["y_mm"],
            centre["z_mm"],
            centre["Rx_deg"],
            centre["Ry_deg"],
            centre["Rz_deg"],
        ]
        raw_position = calculator(joints, pose)
        if not isinstance(raw_position, dict):
            raise RuntimeError(f"相机源#{source}位置函数必须返回字典")
        position = dict(raw_position)
        for key in ("x_mm", "y_mm", "z_mm"):
            position[key] = _finite(
                position.get(key),
                f"途径点 {point['sequence']}.camera{source}_position.{key}",
            )
        return position

    def _capture(self, source: int) -> None:
        point_sequence = len(self._manifest["points"])
        if point_sequence < 1:
            raise RuntimeError(f"相机源#{source}拍照前缺少 record_point 途径点")
        number = self._photo_counts.get(source, 0) + 1
        # The suffix is the global route-point sequence, not this source's own
        # photo counter.  All cameras captured at one physical point therefore
        # share the same suffix (for example 0_001, 1_001, 2_001).
        photo_path = self._output_dir / f"{source}_{point_sequence:03d}.jpg"
        if photo_path.exists():
            raise RuntimeError(f"途径点照片已存在，拒绝覆盖：{photo_path.name}")
        source_position = self._source_position_for_last_point(source)
        if self._snapshot_source is not None:
            ok = bool(self._snapshot_source(source, photo_path))
        elif self._camera_pool is not None:
            ok = self._camera_pool.snapshot(source, photo_path)
        else:  # pragma: no cover - constructor/run invariant
            ok = False
        if not ok:
            raise RuntimeError(f"保存相机源#{source}照片失败")
        if source_position is not None:
            # Only a point at which this source actually saved a photo receives
            # the source-specific position annotation in points.json.
            self._manifest["points"][-1][f"camera{source}_position"] = source_position
        self._photo_counts[source] = number
        photo_record = {
            "source": source,
            "sequence_for_source": number,
            "filename": photo_path.name,
            "captured_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "point_sequence": point_sequence,
        }
        if self._repeatable:
            photo_record["collection_round"] = self._collection_round
        self._manifest["photos"].append(photo_record)
        self._save_manifest()
        self.photo_saved.emit(str(photo_path))

    def _wait_for_operator_checkpoint(self, step: dict) -> bool:
        """Pause after a complete scan and wait for Continue or Finish.

        The robot has already returned to the taught centre before Task 7 emits
        this action.  No motion command is issued while this method waits.
        """
        self._operator_decision = None
        self._operator_decision_event.clear()
        photo_total = sum(self._photo_counts.values())
        message = (
            f"已完成第 {self._collection_round} 个标定板姿态，"
            f"本任务累计保存 {photo_total} 张照片。\n\n"
            + step["message"]
        )
        self.progress.emit(
            f"第 {self._collection_round} 个姿态采集完成；等待人员确认"
        )
        self.operator_checkpoint_requested.emit(
            step["name"],
            message,
            step["continue_text"],
            step["finish_text"],
        )
        while not self._operator_decision_event.wait(0.1):
            if self._stop_requested.is_set():
                raise RuntimeError("动作已取消")
        if self._stop_requested.is_set():
            raise RuntimeError("动作已取消")
        if self._operator_decision is None:
            raise RuntimeError("人工确认没有返回有效选择")
        continue_collection = bool(self._operator_decision)
        self._manifest.setdefault("operator_checkpoints", []).append(
            {
                "collection_round": self._collection_round,
                "decided_at": datetime.now().astimezone().isoformat(
                    timespec="milliseconds"
                ),
                "decision": "continue" if continue_collection else "finish",
                "point_count": len(self._manifest["points"]),
                "photo_count": photo_total,
            }
        )
        self._save_manifest()
        return continue_collection

    def _start_video(self, step: dict) -> None:
        """Start one source recording inside the current timestamp folder."""
        source = int(step["source"])
        if self._camera_pool is None:
            raise RuntimeError(f"相机源#{source}录像池未初始化")
        video_path = self._output_dir / step["filename"]
        if video_path.exists():
            raise RuntimeError(f"录像文件已存在，拒绝覆盖：{video_path.name}")
        self._camera_pool.start_video(source, video_path, step["fps"])
        self._active_video_sources.add(source)
        self._manifest["video_recording"] = {
            "source": source,
            "filename": video_path.name,
            "fps": step["fps"],
            "status": "recording",
        }
        self._save_manifest()

    def _stop_video(self, source: int) -> None:
        """Finish one recording, save its metadata, and emit its path."""
        source = int(source)
        if self._camera_pool is None:
            raise RuntimeError(f"相机源#{source}录像池未初始化")
        metadata = self._camera_pool.stop_video(source)
        self._active_video_sources.discard(source)
        self._manifest["videos"].append(metadata)
        self._manifest.pop("video_recording", None)
        self._save_manifest()
        self.video_saved.emit(str(self._output_dir / metadata["filename"]))

    def _assert_joints(self, step: dict) -> None:
        state = self._read_state(step["name"])
        errors = [
            abs(actual - target)
            for actual, target in zip(state["joints"], step["joints"])
        ]
        if any(error > step["tolerance"] for error in errors):
            detail = ", ".join(f"J{i + 1}={error:.3f}" for i, error in enumerate(errors))
            raise RuntimeError(f"{step['name']} 起点检查失败（偏差 {detail}）")

    def _move_joints(self, step: dict) -> None:
        if "require_current_j3_mm" in step:
            state = self._read_state(f"{step['name']} 高度安全检查")
            difference = abs(state["joints"][2] - step["require_current_j3_mm"])
            if difference > step["j3_tolerance_mm"]:
                raise RuntimeError(
                    f"{step['name']} 已阻止 XY/R 移动：当前 J3 未回到浮动高度，"
                    f"高度偏差 {difference:.3f} mm"
                )
        if not self._controller.goto_joints_sync(
            step["name"],
            step["joints"],
            should_stop=self._stop_requested.is_set,
            tolerance=step["tolerance"],
        ):
            raise RuntimeError(f"移动到 {step['name']} 失败")

    def _execute_step(self, step: dict) -> None:
        kind = step["type"]
        if kind == "assert_joints":
            self._assert_joints(step)
        elif kind == "move_joints":
            self._move_joints(step)
        elif kind == "move_xyzr":
            if not self._controller.move_xyzr_sync(
                step["name"],
                x_mm=step["x_mm"],
                y_mm=step["y_mm"],
                z_mm=step["z_mm"],
                r_deg=step["r_deg"],
                should_stop=self._stop_requested.is_set,
            ):
                raise RuntimeError(f"执行 {step['name']} 失败")
        elif kind == "wait":
            if not self._interruptible_wait(step["seconds"]):
                raise RuntimeError("动作已取消")
        elif kind == "capture":
            self._capture(step["source"])
        elif kind == "start_video":
            self._start_video(step)
        elif kind == "stop_video":
            self._stop_video(step["source"])
        elif kind == "record_point":
            self._record_point(step["name"])

    def run(self) -> None:
        ok = False
        message = "动作未开始"
        try:
            self._output_dir.mkdir(parents=True, exist_ok=False)
            self._manifest = {
                "schema_version": 1,
                "task_name": self._task["name"],
                "description": self._task["description"],
                "started_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "status": "running",
                "coordinate_convention": {
                    "world_x_positive": "机械臂控制器返回的世界 +X",
                    "world_y_positive": "P05 到 P00",
                    "world_z_positive": "机械臂升高、远离平台",
                    "rz_definition": "相机方向相对世界 -Y 的有符号角度",
                    "positive_rotation": "从上方看逆时针（从世界 -Y 转向 +X）",
                },
                "camera_model": dict(self._task["camera_model"]),
                "points": [],
                "photos": [],
                "videos": [],
            }
            if self._repeatable:
                self._manifest["collection_mode"] = "operator_repeated_scan"
                self._manifest["operator_checkpoints"] = []
            self._save_manifest()

            capture_sources = {
                step["source"]
                for step in self._task["actions"]
                if step["type"] == "capture"
            }
            video_sources = {
                step["source"]
                for step in self._task["actions"]
                if step["type"] in {"start_video", "stop_video"}
            }
            sources = sorted(capture_sources | video_sources)
            pool_sources = sorted(
                video_sources | (capture_sources if self._snapshot_source is None else set())
            )
            if pool_sources:
                self._camera_pool = CameraSourcePool()
                opened, error = self._camera_pool.open_sources(pool_sources)
                if not opened:
                    raise RuntimeError(error)
            self.progress.emit(
                "相机源检查完成，动作即将开始" if sources else "动作即将开始（无拍照步骤）"
            )

            total = len(self._task["actions"])
            index = 0
            while index < total:
                step = self._task["actions"][index]
                if self._stop_requested.is_set():
                    raise RuntimeError("动作已取消")
                if self._camera_pool is not None:
                    self._camera_pool.check_video_error()
                label = step.get("name") or step["type"]
                round_prefix = (
                    f"姿态{self._collection_round} " if self._repeatable else ""
                )
                self.progress.emit(f"{round_prefix}{index + 1}/{total} {label}")
                if step["type"] == "operator_checkpoint":
                    if self._wait_for_operator_checkpoint(step):
                        self._collection_round += 1
                        index = int(step["repeat_from_index"])
                        continue
                    break
                self._execute_step(step)
                index += 1

            ok = True
            photo_total = sum(self._photo_counts.values())
            message = (
                f"动作完成，共记录 {len(self._manifest['points'])} 个点、"
                f"保存 {photo_total} 张照片、{len(self._manifest['videos'])} 段录像"
            )
        except FileExistsError:
            message = f"输出文件夹已存在：{self._output_dir}"
        except Exception as exc:  # noqa: BLE001 - displayed safely in the UI
            message = str(exc) or exc.__class__.__name__
        finally:
            for source in list(self._active_video_sources):
                try:
                    self._stop_video(source)
                except Exception:
                    self._active_video_sources.discard(source)
            if self._camera_pool is not None:
                self._camera_pool.close()
            if self._manifest:
                self._manifest["status"] = "completed" if ok else "stopped"
                self._manifest["finished_at"] = datetime.now().astimezone().isoformat(
                    timespec="milliseconds"
                )
                self._manifest["result"] = message
                try:
                    self._save_manifest()
                except Exception:
                    pass
            self.run_finished.emit(ok, message, str(self._output_dir))
