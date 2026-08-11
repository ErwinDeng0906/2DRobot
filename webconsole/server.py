#!/usr/bin/env python3
"""机械臂 Web 控制台后端。

标准库 http.server，无需框架。复用 devices/robot_arm 后端（sim / thrift 真机）。
暴露 /api/* JSON 接口给前端 index.html。序列回放用纯 threading（不依赖 Qt）。

启动：
    ~/gcr618_control/venv/bin/python webconsole/server.py [--port 8080] [--root <repo>]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from devices.robot_arm import create_backend, load_config, RobotArmStatus  # noqa: E402
from devices.robot_arm import sequences as seqmod  # noqa: E402
from devices.robot_arm.duco_rpc import ST_ROBOT, ST_SAFE, ST_MODE, ST_PROG  # noqa: E402

SEQUENCES = {
    "PICK_PLACE_FULL（完整取放）": seqmod.PICK_PLACE_FULL,
    "TRANSIT_ONLY（仅转移）": seqmod.TRANSIT_ONLY,
}

JOG_SCALE = 0.05          # JOG 极低速（安全）
DEFAULT_SEQ_SCALE = 0.2


class CameraManager:
    """机械臂末端 USB 相机：独立采集线程，缓存最新 JPEG 帧，供 MJPEG 流复用。

    懒启动：第一次有人请求画面时才打开相机。失败不影响机械臂控制。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cap = None
        self._device = 0
        self._latest_jpeg = None
        self._running = False
        self._thread = None
        self._opened = False
        self._error = ""

    def available(self):
        try:
            import cv2  # noqa
            return True
        except Exception:
            return False

    def start(self, device=0):
        if self._running:
            return True, ""
        try:
            import cv2
        except Exception as e:
            self._error = "opencv 未安装: %s" % e
            return False, self._error
        self._device = device
        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            self._error = "相机 /dev/video%d 打不开" % device
            self._cap = None
            return False, self._error
        # 分辨率/画质：抬到 1080p 提升清晰度（原 640×480 太糊，仪表板看不清）。
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self._opened = True
        self._running = True
        self._error = ""
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True, ""

    def _loop(self):
        import cv2
        while self._running and self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            try:
                ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ret:
                    with self._lock:
                        self._latest_jpeg = buf.tobytes()
            except Exception:
                pass
            time.sleep(0.04)  # ~25fps 上限

    def latest(self):
        with self._lock:
            return self._latest_jpeg

    def info(self):
        return {
            "available": self.available(),
            "opened": self._opened,
            "running": self._running,
            "device": self._device,
            "error": self._error,
        }

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._opened = False


CAMERA = CameraManager()

# 多相机预览代理：从 multicam(:8091，已 owns 全部相机) 的 MJPEG 流里抽一帧 JPEG 返回。
# 让仪表板 2×2 四路预览统一走 armweb(:8080，外部可达)，不各自开相机，避免与 multicam 争抢。
_MULTICAM_BASE = "http://127.0.0.1:8091"


def _read_one_mjpeg_frame(device, timeout=8.0):
    import urllib.request
    url = "%s/s?d=%s" % (_MULTICAM_BASE, device)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        buf = b""
        deadline = time.time() + timeout
        while time.time() < deadline:
            chunk = r.read(16384)
            if not chunk:
                break
            buf += chunk
            s = buf.find(b"\xff\xd8")
            if s >= 0:
                e = buf.find(b"\xff\xd9", s + 2)
                if e > s:
                    return buf[s:e + 2]
            if len(buf) > 3_000_000:
                break
    return None


class ArmService:
    """封装后端 + 序列引擎 + 线程安全。Web handler 调用这里。"""

    def __init__(self):
        self._lock = threading.Lock()
        self.mode = "sim"
        self.backend = create_backend("sim")
        self.message = "就绪（仿真模式）"
        # 序列引擎
        self._seq_thread = None
        self._seq_pause = threading.Event()
        self._seq_stop = threading.Event()
        self._seq_running = False
        self._seq_step = [0, 0, ""]
        self._seq_scale = DEFAULT_SEQ_SCALE
        # 运动锁：避免并发下发
        self._moving = False
        # goto_pose 异步结果闩（客户端 POST 后轮询 /api/status 判完成/失败）：
        # seq 每次 goto_pose 递增；error="" 表成功、非空为失败原因。
        self._goto_pose_seq = 0
        self._goto_pose_error = ""

    # ── 状态快照 ────────────────────────────────────────────
    def status(self) -> dict:
        try:
            st = self.backend.get_status()
        except Exception as e:
            st = RobotArmStatus(connected=False, error="读状态异常: %s" % e)
        d = {
            "connected": st.connected,
            "mode": self.mode,
            "joints": list(st.joints),
            "joints_deg": [round(x, 3) for x in st.joints_deg],
            "tcp": list(st.tcp),
            "op_mode": st.op_mode,
            "robot_state": st.robot_state,
            "prog_state": st.prog_state,
            "safety": st.safety,
            "alarm": st.alarm,
            "collision": st.collision,          # 2026-07-25: 透传碰撞给 UI 信号灯状态机(红灯=真实碰撞)
            "collision_axis": st.collision_axis,
            "powered": st.powered,
            "enabled": st.enabled,
            "is_ready": st.is_ready,
            "error": st.error,
            "robot_state_text": ST_ROBOT.get(st.robot_state, str(st.robot_state)),
            "safety_text": ST_SAFE.get(st.safety, str(st.safety)),
            "op_mode_text": ST_MODE.get(st.op_mode, str(st.op_mode)),
            "prog_state_text": ST_PROG.get(st.prog_state, str(st.prog_state)),
            "waypoints": self._safe(lambda: self.backend.waypoint_names(), []),
            "gripper": self._safe(lambda: self.backend.gripper_status(), None),
            "sequences": list(SEQUENCES.keys()),
            "seq_running": self._seq_running,
            "seq_step": list(self._seq_step),
            "message": self.message,
            "moving": self._moving,
            "goto_pose_seq": self._goto_pose_seq,
            "goto_pose_error": self._goto_pose_error,
        }
        return d

    def _safe(self, fn, default):
        try:
            return fn()
        except Exception:
            return default

    # ── 模式 / 连接 ─────────────────────────────────────────
    def set_mode(self, mode):
        if self.is_connected():
            return False, "请先断开连接再切换模式"
        if mode not in ("sim", "thrift"):
            return False, "未知模式"
        with self._lock:
            self.mode = mode
            self.backend = create_backend(mode)
            self.message = "已切换到%s模式" % ("仿真" if mode == "sim" else "真机(Thrift)")
        return True, ""

    def connect(self, mode=None, ip=None, port=None):
        if mode and mode != self.mode:
            ok, err = self.set_mode(mode)
            if not ok:
                return False, err
        try:
            if self.mode == "thrift":
                cfg = load_config()
                if ip:
                    cfg["connection"]["ip"] = ip
                if port:
                    cfg["connection"]["rpc_port"] = int(port)
                self.backend = create_backend("thrift", cfg=cfg)
            self.backend.connect()
            self.message = "已连接"
            return True, ""
        except Exception as e:
            self.message = "连接失败: %s" % e
            return False, str(e)

    def disconnect(self):
        self.seq_stop()
        try:
            self.backend.disconnect()
        except Exception as e:
            return False, str(e)
        self.message = "已断开"
        return True, ""

    def is_connected(self):
        try:
            return bool(self.backend.is_connected())
        except Exception:
            return False

    # ── 上电 / 使能 ─────────────────────────────────────────
    def power_on(self):
        return self._guard(self.backend.power_on, "上电")

    def enable(self):
        return self._guard(self.backend.enable, "使能")

    def disable(self):
        return self._guard(self.backend.disable, "下使能")

    def estop(self):
        try:
            self.seq_stop()
            self.backend.estop()
            self.message = "⚠ 已急停"
            return True, ""
        except Exception as e:
            return False, str(e)

    def enter_teach(self, body):
        try:
            m = float(body.get("load_mass", 0.0))
            cx = float(body.get("cx", 0.0)); cy = float(body.get("cy", 0.0)); cz = float(body.get("cz", 0.0))
            self.backend.enter_teach_mode(m, cx, cy, cz)
            self.message = "已进入拖动示教（请手扶机械臂）"
            return True, ""
        except Exception as e:
            return False, str(e)

    def exit_teach(self):
        try:
            self.backend.exit_teach_mode()
            self.message = "已退出拖动示教"
            return True, ""
        except Exception as e:
            return False, str(e)

    def _guard(self, fn, name):
        try:
            fn()
            self.message = "%s 完成" % name
            return True, ""
        except Exception as e:
            self.message = "%s 失败: %s" % (name, e)
            return False, str(e)

    # ── JOG / goto（后台线程，避免阻塞 HTTP）────────────────
    def jog(self, joint, delta_deg):
        if not self._require_ready():
            return False, self.message
        if self._moving:
            return False, "运动进行中"

        def task():
            self._moving = True
            try:
                self.backend.jog_joint(int(joint), math.radians(float(delta_deg)),
                                       speed_scale=JOG_SCALE)
                self.message = "JOG J%d %+.1f° 完成" % (int(joint) + 1, float(delta_deg))
            except Exception as e:
                self.message = "JOG 失败: %s" % e
            finally:
                self._moving = False

        threading.Thread(target=task, daemon=True).start()
        return True, ""

    def set_robot_mode(self, mode):
        """软件切机器人手/自动模式（0手动/1自动）。用户(admin)授权软件切模式。不动机器人。"""
        try:
            rpc = getattr(getattr(self.backend, "_robot", None), "rpc", None)
            if rpc is None:
                return False, "无 rpc（sim/未连接真机）"
            rpc.switch_mode(int(mode))
            return True, ""
        except Exception as e:
            return False, "切模式失败: %s" % e

    def last_error(self):
        """只读诊断：读 DUCO get_last_error + 当前 alarm/op_mode，不动机器人。"""
        out = {}
        try:
            s = self.backend.get_status()
            out.update(connected=s.connected, alarm=s.alarm, op_mode=s.op_mode,
                       robot_state=s.robot_state, safety=s.safety, enabled=s.enabled)
        except Exception as e:
            out["status_err"] = str(e)
        try:
            rpc = getattr(getattr(self.backend, "_robot", None), "rpc", None)
            out["errors"] = rpc.get_last_error() if rpc is not None else ["无 rpc(sim/未连)"]
        except Exception as e:
            out["errors"] = ["get_last_error 失败: %s" % e]
        return out

    def jog_cart(self, axis, delta, speed_scale=None, dry_run=False):
        """笛卡尔点动。dry_run：同步算逆解(不动机器人)、返回目标关节等 info。
        真运动：需就绪+不在运动中，后台线程走后端 jog_cart(受监督)。返回 (ok, err, info|None)。"""
        if axis is None or delta is None:
            return False, "缺 axis/delta", None
        if dry_run:
            try:
                info = self.backend.jog_cart(int(axis), float(delta),
                                             speed_scale=speed_scale, dry_run=True)
                return True, "", info
            except Exception as e:
                return False, "笛卡尔逆解/校验失败: %s" % e, None
        if not self._require_ready():
            return False, self.message, None
        if self._moving:
            return False, "运动进行中", None

        def task():
            self._moving = True
            try:
                self.backend.jog_cart(
                    int(axis), float(delta),
                    speed_scale=(speed_scale if speed_scale is not None else JOG_SCALE),
                    dry_run=False)
                self.message = "笛卡尔点动完成"
            except Exception as e:
                self.message = "笛卡尔点动失败: %s" % e
            finally:
                self._moving = False

        threading.Thread(target=task, daemon=True).start()
        return True, "", None

    def goto(self, name):
        if not self._require_ready():
            return False, self.message
        if self._moving:
            return False, "运动进行中"
        scale = self._cfg_scale()
        self._moving = True

        def task():
            try:
                self.backend.goto_waypoint(name, speed_scale=scale, block=True)
                self.message = "前往「%s」完成" % name
            except Exception as e:
                self.message = "前往失败: %s" % e
            finally:
                self._moving = False

        threading.Thread(target=task, daemon=True).start()
        return True, ""

    def move_joints(self, joints, speed_scale=None):
        # 协调关节运动(关节回放 ARM_REPLAY_JOINTS 用)：一次 movej_safe 到目标关节。
        # 同步阻塞到位后返回(ThreadingHTTPServer，status 读心跳缓存帧不受影响)，
        # 使客户端 block=True 成立且错误如实回传(取放失败要停流程，不用异步 goto 那套)。
        if not self._require_ready():
            return False, self.message
        if self._moving:
            return False, "运动进行中"
        try:
            q = [float(x) for x in (joints or [])]
        except Exception as e:
            return False, "关节参数非法: %s" % e
        if len(q) < 6:
            return False, "关节数不足(需6，收到%d)" % len(q)
        scale = speed_scale if speed_scale is not None else self._cfg_scale()
        self._moving = True
        try:
            self.backend.move_joints(q, speed_scale=scale, block=True)
            self.message = "关节回放到位"
            return True, ""
        except Exception as e:
            self.message = "关节运动失败: %s" % e
            return False, str(e)
        finally:
            self._moving = False

    def goto_pose(self, x, y, z, yaw, tool_offset=None, speed_scale=None):
        """旧单杯在线 IK 入口永久冻结，不启动后台运动线程。"""
        return False, "goto_pose 旧单杯在线 IK 入口永久冻结；仅允许已审计冻结计划"

    def record_waypoint(self, name):
        try:
            self.backend.record_waypoint(name)
            self.message = "已记录点位「%s」" % name
            return True, ""
        except Exception as e:
            return False, str(e)

    def delete_waypoint(self, name):
        try:
            self.backend.delete_waypoint(name)
            self.message = "已删除点位「%s」" % name
            return True, ""
        except Exception as e:
            return False, str(e)

    def gripper(self, port, value):
        # 旧语义：机械臂吸盘泵（塔石继电器）。已移除，请用 DH 夹爪端点 /api/gripper/init 等。
        return False, "塔石继电器泵控制已移除（请用夹爪 API 或 SCARA scara_do）"

    def set_arm_pump(self, on: bool):
        return False, "塔石继电器泵控制已移除"

    def set_hotplate_pump(self, on: bool):
        return False, "塔石继电器泵控制已移除"

    def set_relay_device(self, name: str, on: bool):
        return False, "塔石继电器控制已移除"

    def pump_status(self):
        return False, {"error": "塔石继电器控制已移除"}

    # ── 夹爪（大寰 PGEA-50-26-O，工具端 485 透传）─────────────────
    #    路由 /api/gripper/init|grip|release + 状态并入 /api/status。后端方法
    #    thrift/sim 均已实现；动作前置真机检查——sim/未连接下不真正驱动夹爪，
    #    若放行会返回假成功，误导操作员以为夹爪动了（实际没动）。
    def _require_gripper_ready(self):
        """夹爪动作需连真机械臂（经工具端 485 透传）。sim/未连接直接拒，防假成功。"""
        if self.mode != "thrift" or not self.is_connected():
            self.message = "夹爪需先连接真机械臂（当前仿真/未连接）"
            return False
        return True

    def gripper_init(self, full=False):
        if not self._require_gripper_ready():
            return False, self.message
        try:
            self.backend.gripper_initialize(full=bool(full))
            self.message = "夹爪初始化%s完成" % ("(全行程)" if full else "")
            return True, ""
        except Exception as e:
            self.message = "夹爪初始化失败: %s" % e
            return False, str(e)

    def gripper_grip(self, pos, force=50, speed=50):
        if not self._require_gripper_ready():
            return False, self.message
        try:
            self.backend.gripper_grip(force=int(force), speed=int(speed), pos=int(pos))
            self.message = "夹爪移动到 %d‰" % int(pos)
            return True, ""
        except Exception as e:
            self.message = "夹爪移动失败: %s" % e
            return False, str(e)

    def gripper_release(self, speed=50):
        if not self._require_gripper_ready():
            return False, self.message
        try:
            self.backend.gripper_release(speed=int(speed))
            self.message = "夹爪全开释放"
            return True, ""
        except Exception as e:
            self.message = "夹爪释放失败: %s" % e
            return False, str(e)

    def gripper_status(self):
        """读夹爪状态 dict（读失败/未实现返回 None，不抛，供 /api/status 合并）。"""
        return self._safe(lambda: self.backend.gripper_status(), None)

    def waypoint_joints(self, name):
        try:
            return self.backend.get_waypoint_joints(name)
        except Exception:
            return [0.0] * 6

    def _require_ready(self):
        if not self.is_connected():
            self.message = "未连接"
            return False
        st = self.backend.get_status()
        if not st.enabled:
            self.message = "未使能，先点「使能」"
            return False
        return True

    def _cfg_scale(self):
        try:
            return self.backend.cfg.get("speed", {}).get("global_scale", DEFAULT_SEQ_SCALE)
        except Exception:
            return DEFAULT_SEQ_SCALE

    # ── 序列回放 ────────────────────────────────────────────
    def sequence(self, action, name=None, speed=None):
        if action == "run":
            return self._seq_run(name, speed)
        if action == "pause":
            if self._seq_pause.is_set():
                self._seq_pause.clear()
                self.message = "序列继续"
            else:
                self._seq_pause.set()
                self.message = "序列暂停"
            return True, ""
        if action == "stop":
            self.seq_stop()
            return True, ""
        return False, "未知序列动作"

    def _seq_run(self, name, speed):
        if self._seq_running:
            return False, "序列已在运行"
        if not self._require_ready():
            return False, self.message
        seq = SEQUENCES.get(name)
        if not seq:
            return False, "未知序列: %s" % name
        if speed is not None:
            self._seq_scale = max(0.01, min(1.0, float(speed)))
        self._seq_stop.clear()
        self._seq_pause.clear()
        self._seq_running = True
        self._seq_thread = threading.Thread(target=self._seq_loop, args=(list(seq),), daemon=True)
        self._seq_thread.start()
        self.message = "运行序列「%s」" % name
        return True, ""

    def _seq_loop(self, names):
        total = len(names)
        try:
            for i, wp in enumerate(names):
                if self._seq_stop.is_set():
                    self.message = "序列已停止"
                    break
                while self._seq_pause.is_set() and not self._seq_stop.is_set():
                    time.sleep(0.1)
                if self._seq_stop.is_set():
                    self.message = "序列已停止"
                    break
                self._seq_step = [i + 1, total, wp]
                self._moving = True
                try:
                    self.backend.goto_waypoint(wp, speed_scale=self._seq_scale, block=True)
                finally:
                    self._moving = False
            else:
                self.message = "序列完成"
                self._seq_step = [total, total, "完成"]
        except Exception as e:
            self.message = "序列异常: %s" % e
        finally:
            self._seq_running = False

    def seq_stop(self):
        self._seq_stop.set()
        self._seq_pause.clear()
        self._seq_running = False


SERVICE = ArmService()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # 静音

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n == 0:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    def _file(self, path, ctype):
        try:
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            return self._file(HERE / "index.html", "text/html; charset=utf-8")
        if self.path == "/api/status":
            return self._json(SERVICE.status())
        if self.path == "/api/last_error":
            return self._json(SERVICE.last_error())
        if self.path == "/api/pump/status":
            ok, data = SERVICE.pump_status()
            if ok:
                return self._json({"ok": True, **data})
            return self._json({"ok": False, "error": data.get("error", "")})
        if self.path == "/api/relay/status":
            ok, data = SERVICE.pump_status()
            if ok:
                return self._json({"ok": True, **data})
            return self._json({"ok": False, "error": data.get("error", "")})
        if self.path.startswith("/api/waypoint_joints"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            name = (q.get("name") or [""])[0]
            return self._json({"joints": SERVICE.waypoint_joints(name)})
        if self.path == "/api/camera/info":
            return self._json(CAMERA.info())
        if self.path == "/api/camera/snapshot":
            return self._snapshot()
        if self.path.startswith("/api/camera/grid"):
            return self._camera_grid()
        if self.path == "/api/camera/stream":
            return self._mjpeg_stream()
        self.send_response(404)
        self.end_headers()

    def _snapshot(self):
        if not CAMERA._opened:
            CAMERA.start(CAMERA._device)
            time.sleep(0.3)
        jpg = CAMERA.latest()
        if jpg is None:
            return self._json({"ok": False, "error": "无画面"}, 503)
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpg)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(jpg)

    def _mjpeg_stream(self):
        if not CAMERA._opened:
            ok, err = CAMERA.start(CAMERA._device)
            if not ok:
                return self._json({"ok": False, "error": err}, 503)
            time.sleep(0.3)
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                jpg = CAMERA.latest()
                if jpg is not None:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(("Content-Length: %d\r\n\r\n" % len(jpg)).encode())
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass  # 客户端关闭流，正常

    def _camera_grid(self):
        """2×2 四路预览：从 multicam 抽一帧 JPEG 返回（device=0/2/4/6）。"""
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(self.path).query)
        dev = (q.get("device") or ["0"])[0]
        try:
            if str(dev) == "0":
                # video0（正面全景）走 armweb 自身相机——multicam 抓不到 video0
                if not CAMERA._opened:
                    CAMERA.start(0)
                    time.sleep(0.3)
                jpg = CAMERA.latest()
            else:
                jpg = _read_one_mjpeg_frame(dev, timeout=8.0)
        except Exception as e:
            return self._json({"ok": False, "error": str(e)}, 503)
        if not jpg:
            return self._json({"ok": False, "error": "无画面"}, 503)
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpg)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(jpg)

    def do_POST(self):
        b = self._body()
        ok, err = True, ""
        try:
            if self.path == "/api/connect":
                ok, err = SERVICE.connect(b.get("mode"), b.get("ip"), b.get("port"))
            elif self.path == "/api/disconnect":
                ok, err = SERVICE.disconnect()
            elif self.path == "/api/set_mode":
                ok, err = SERVICE.set_mode(b.get("mode", "sim"))
            elif self.path == "/api/switch_mode":
                ok, err = SERVICE.set_robot_mode(b.get("mode", 1))
            elif self.path == "/api/power_on":
                ok, err = SERVICE.power_on()
            elif self.path == "/api/enable":
                ok, err = SERVICE.enable()
            elif self.path == "/api/disable":
                ok, err = SERVICE.disable()
            elif self.path == "/api/estop":
                ok, err = SERVICE.estop()
            elif self.path == "/api/teach/enter":
                ok, err = SERVICE.enter_teach(b)
            elif self.path == "/api/teach/exit":
                ok, err = SERVICE.exit_teach()
            elif self.path == "/api/jog":
                ok, err = SERVICE.jog(b.get("joint"), b.get("delta_deg"))
            elif self.path == "/api/jog_cart":
                ok, err, info = SERVICE.jog_cart(
                    b.get("axis"), b.get("delta"),
                    b.get("speed_scale"), b.get("dry_run", False))
                return self._json({"ok": ok, "error": err, **(info or {})})
            elif self.path == "/api/goto":
                ok, err = SERVICE.goto(b.get("name"))
            elif self.path == "/api/move_joints":
                ok, err = SERVICE.move_joints(b.get("joints"), b.get("speed_scale"))
            elif self.path == "/api/goto_pose":
                ok, err = SERVICE.goto_pose(b.get("x"), b.get("y"), b.get("z"), b.get("yaw"),
                                            tool_offset=b.get("tool_offset"),
                                            speed_scale=b.get("speed_scale"))
            elif self.path == "/api/sequence":
                ok, err = SERVICE.sequence(b.get("action"), b.get("name"), b.get("speed"))
            elif self.path == "/api/record_waypoint":
                ok, err = SERVICE.record_waypoint(b.get("name"))
            elif self.path == "/api/delete_waypoint":
                ok, err = SERVICE.delete_waypoint(b.get("name"))
            elif self.path == "/api/gripper":
                ok, err = SERVICE.gripper(b.get("port"), b.get("value"))
            elif self.path == "/api/gripper/init":
                ok, err = SERVICE.gripper_init(b.get("full", False))
            elif self.path == "/api/gripper/grip":
                ok, err = SERVICE.gripper_grip(b.get("pos"), b.get("force", 50), b.get("speed", 50))
            elif self.path == "/api/gripper/release":
                ok, err = SERVICE.gripper_release(b.get("speed", 50))
            elif self.path == "/api/pump":
                which = b.get("which")
                val = bool(b.get("value"))
                if which == "arm":
                    ok, err = SERVICE.set_arm_pump(val)
                elif which == "hotplate":
                    ok, err = SERVICE.set_hotplate_pump(val)
                else:
                    ok, err = False, "unknown pump: %s" % which
            elif self.path == "/api/relay":
                name = b.get("name")
                val = bool(b.get("value"))
                ok, err = SERVICE.set_relay_device(name, val)
            elif self.path == "/api/camera/start":
                ok, err = CAMERA.start(int(b.get("device", 0)))
            elif self.path == "/api/camera/stop":
                CAMERA.stop()
                ok, err = True, ""
            else:
                return self._json({"ok": False, "error": "unknown endpoint"}, 404)
        except Exception as e:
            ok, err = False, str(e)
        self._json({"ok": ok, "error": err})


class _FastHTTPServer(ThreadingHTTPServer):
    """跳过 server_bind 里的 socket.getfqdn() —— 它会在 DNS 反查上阻塞。

    标准 HTTPServer.server_bind 绑定后调 getfqdn(host) 求全限定域名存进 server_name
    （仅 CGI/日志用）。本机 DNS 环境异常时（daemon 起了 dnsmasq --port=0 关 DNS +
    改了网卡别名），getfqdn 会卡到 DNS 超时，导致绑定阶段静默挂死、8080 永不监听。
    这是个内部 API 服务，server_name 无实际用途，直接用 host 字符串跳过反查。
    """
    def server_bind(self):
        import socketserver
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host or "localhost"
        self.server_port = port


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    srv = _FastHTTPServer((args.host, args.port), Handler)
    print("机械臂 Web 控制台后端启动: http://%s:%d" % (args.host, args.port))
    print("默认仿真模式；前端可切真机。Ctrl-C 停止。")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        SERVICE.disconnect()
        print("\n已停止")


if __name__ == "__main__":
    main()
