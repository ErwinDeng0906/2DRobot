"""仿真机械臂后端：无硬件，本地积分运动。

用于 P3-P7 全程开发/测试，以及 UI 的「仿真模式」（不连真机在 3D 里走轨迹）。
- 维护一份关节角状态，move_joints 时在后台线程平滑插值到目标（让 3D 看到运动过程）。
- 模拟上电状态机：下电(4) → 上电未使能(5) → 使能(6)。
- waypoints 复用 WaypointStore + 模块内 waypoints.json。
- TCP 位姿由 kinematics.forward_kinematics 从关节角算出（与 3D 一致）。
"""
from __future__ import annotations

import math
import threading
import time
from typing import List, Optional

from .backend import RobotArmBackendBase, RobotArmStatus, ROBOT_STATE_ENABLED, SAFETY_RUN
from .waypoints import WaypointStore


class SimRobotArmBackend(RobotArmBackendBase):
    """纯软件仿真后端。线程安全：状态读写加锁。"""

    def __init__(self, cfg: dict, waypoints_path: str):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._connected = False
        self._robot_state = 4          # 下电
        self._safety = SAFETY_RUN      # 仿真安全态恒 RUN
        self._prog_state = 0
        self._alarm = 0

        # 初始关节角：用 home 点（若存在），否则全 0
        self._store = WaypointStore(waypoints_path)
        if self._store.has("home"):
            self._joints = list(self._store.get_joints("home"))
        else:
            self._joints = [0.0, -math.pi / 2, math.pi / 2, 0.0, math.pi / 2, 0.0]

        # 运动线程
        self._move_thread: Optional[threading.Thread] = None
        self._move_stop = threading.Event()

        # 工作空间 / 禁区（仿真仅记录，check_pose_allowed 用）
        ws = cfg.get("workspace", {})
        self._ws_x = list(ws.get("x", [-1, 1]))
        self._ws_y = list(ws.get("y", [-1, 1]))
        self._ws_z = list(ws.get("z", [-1, 1]))
        self._nogo: list = []

        # 吸盘仿真态（无硬件，仅记录供日志/UI 观察）
        self._gripper_do = False

        # 大寰夹爪仿真态（无硬件，仅记录供日志/UI 观察）
        self._grip_inited = False
        self._grip_pos = 1000            # ‰：假设初始（未夹）全开
        self._grip_state = 0             # 0运动中/1到位未夹到/2夹住/3掉落

        # 速度（用于插值时长估算）
        self._global_scale = cfg.get("speed", {}).get("global_scale", 0.2)
        self._v_max = cfg.get("speed", {}).get("v_max_joint", 1.05)

    # ── 连接 ────────────────────────────────────────────────────
    def connect(self) -> None:
        time.sleep(0.2)  # 模拟握手延迟
        with self._lock:
            self._connected = True

    def disconnect(self) -> None:
        self._stop_motion()
        with self._lock:
            self._connected = False
            self._robot_state = 4

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    # ── 状态 ────────────────────────────────────────────────────
    def get_status(self) -> RobotArmStatus:
        with self._lock:
            joints = list(self._joints)
            connected = self._connected
            rs = self._robot_state
            safety = self._safety
            prog = self._prog_state
            alarm = self._alarm
        tcp = self._fk(joints)
        return RobotArmStatus(
            connected=connected,
            joints=joints,
            tcp=tcp,
            op_mode=2,
            robot_state=rs,
            prog_state=prog,
            safety=safety,
            collision=0,
            collision_axis=0,
            alarm=alarm,
            powered=rs >= 5,
            enabled=rs == ROBOT_STATE_ENABLED,
            error="",
        )

    def _fk(self, joints):
        """正运动学算 TCP（委托 kinematics；失败则返回零位姿，不影响仿真）。"""
        try:
            from .kinematics import forward_kinematics
            return forward_kinematics(joints)
        except Exception:
            return [0.0] * 6

    # ── 上电 / 使能 ─────────────────────────────────────────────
    def power_on(self) -> None:
        with self._lock:
            if self._connected and self._robot_state < 5:
                self._robot_state = 5

    def enable(self) -> None:
        with self._lock:
            if self._connected and self._robot_state >= 5:
                self._robot_state = ROBOT_STATE_ENABLED

    def disable(self) -> None:
        self._stop_motion()
        with self._lock:
            if self._robot_state == ROBOT_STATE_ENABLED:
                self._robot_state = 5

    def estop(self) -> None:
        self._stop_motion()
        with self._lock:
            self._robot_state = 5
            self._prog_state = 0

    # ── 运动 ────────────────────────────────────────────────────
    def move_joints(self, joints: List[float], speed_scale: Optional[float] = None,
                    block: bool = True) -> None:
        if len(joints) != 6:
            raise ValueError("需要 6 个关节角")
        # 限位检查
        jl = self.cfg.get("joint_limits", {})
        lower = jl.get("lower", [-3.1] * 6)
        upper = jl.get("upper", [3.1] * 6)
        for i, q in enumerate(joints):
            if q < lower[i] or q > upper[i]:
                raise RuntimeError("关节%d=%.3f 越软限位[%.2f,%.2f]" % (i + 1, q, lower[i], upper[i]))
        with self._lock:
            if self._robot_state != ROBOT_STATE_ENABLED:
                raise RuntimeError("未使能，拒绝运动（仿真）")
            start = list(self._joints)

        scale = speed_scale if speed_scale is not None else self._global_scale
        target = list(joints)
        # 估算时长：最大单关节角度 / (v_max * scale)，至少 0.3s
        dmax = max(abs(a - b) for a, b in zip(target, start))
        dur = max(0.3, dmax / max(self._v_max * max(scale, 0.01), 1e-3))
        dur = min(dur, 8.0)  # 仿真封顶，避免大角度等太久

        self._stop_motion()
        self._move_stop.clear()
        self._move_thread = threading.Thread(
            target=self._interp_move, args=(start, target, dur), daemon=True)
        with self._lock:
            self._prog_state = 5
        self._move_thread.start()
        if block:
            self._move_thread.join()

    def _interp_move(self, start, target, dur):
        """后台线程：从 start 平滑插值到 target，更新关节角。"""
        t0 = time.time()
        while not self._move_stop.is_set():
            t = (time.time() - t0) / dur
            if t >= 1.0:
                break
            # 平滑 ease-in-out
            s = 0.5 - 0.5 * math.cos(math.pi * t)
            with self._lock:
                self._joints = [a + (b - a) * s for a, b in zip(start, target)]
            time.sleep(0.03)  # ~33fps 更新
        if not self._move_stop.is_set():
            with self._lock:
                self._joints = list(target)
        with self._lock:
            self._prog_state = 0

    def _stop_motion(self):
        self._move_stop.set()
        t = self._move_thread
        if t is not None and t.is_alive():
            t.join(timeout=1.0)
        self._move_thread = None

    def goto_pose(self, x, y, z, yaw, tool_offset=None, speed_scale=None, block=True) -> None:
        """仿真无 cal_ikine：仅合成姿态并记录意图，不做逆解运动（sim 干跑流程不崩）。"""
        rot = None
        try:
            from .calib.wafer_tray import compose_place_orientation
            cdir = __import__("pathlib").Path(__file__).resolve().parent / "calib"
            import json
            with open(cdir / "table_plane.json", encoding="utf-8") as f:
                rdown = json.load(f)["R_down_euler_zyx"]
            rot = compose_place_orientation(rdown, float(yaw))
        except Exception:
            rot = None
        with self._lock:
            self._last_goto_pose = (float(x), float(y), float(z), float(yaw), rot)
        # 仿真不逆解、不运动（无 cal_ikine）；真机走 thrift 后端 goto_pose。

    # ── 末端数字输出（吸盘，仿真仅记录）─────────────────────────
    def set_digital_output(self, port: int, value: bool, block: bool = True) -> None:
        with self._lock:
            self._gripper_do = bool(value)

    @property
    def gripper_state(self) -> bool:
        """当前吸盘 DO 状态（仿真用，供日志/UI 观察）。"""
        with self._lock:
            return self._gripper_do

    # ── 夹爪（大寰 PGEA，仿真仅记录意图 + 假状态，永不阻塞/报错）──
    def gripper_initialize(self, full: bool = False) -> None:
        with self._lock:
            self._grip_inited = True
            self._grip_pos = 1000       # 初始化后全开
            self._grip_state = 1        # 到位未夹到

    def gripper_grip(self, force=None, speed=None, pos=0) -> None:
        with self._lock:
            self._grip_pos = max(0, min(1000, int(pos)))
            self._grip_state = 2        # 假装夹住物体

    def gripper_release(self, speed=None) -> None:
        with self._lock:
            self._grip_pos = 1000       # 全开
            self._grip_state = 1        # 到位未夹到

    def gripper_status(self) -> dict:
        with self._lock:
            return {
                "init_state": 1 if self._grip_inited else 0,
                "grip_state": self._grip_state,
                "position": self._grip_pos,
            }

    # ── 安全配置 ────────────────────────────────────────────────
    def set_workspace_bounds(self, x=None, y=None, z=None) -> None:
        with self._lock:
            if x is not None:
                self._ws_x = list(x)
            if y is not None:
                self._ws_y = list(y)
            if z is not None:
                self._ws_z = list(z)

    def set_nogo_boxes(self, boxes: list) -> None:
        with self._lock:
            self._nogo = list(boxes)

    def check_pose_allowed(self, pose) -> tuple:
        x, y, z = pose[0], pose[1], pose[2]
        if not (self._ws_x[0] <= x <= self._ws_x[1]):
            return False, "X=%.3f 越工作空间%s" % (x, self._ws_x)
        if not (self._ws_y[0] <= y <= self._ws_y[1]):
            return False, "Y=%.3f 越工作空间%s" % (y, self._ws_y)
        if not (self._ws_z[0] <= z <= self._ws_z[1]):
            return False, "Z=%.3f 越工作空间%s" % (z, self._ws_z)
        for b in self._nogo:
            if (abs(x - b["cx"]) <= b["w"] and abs(y - b["cy"]) <= b["h"]
                    and abs(z - b["cz"]) <= b["d"]):
                return False, "TCP 进入禁区 %s" % b.get("name", "")
        return True, "ok"

    # ── 点位 ────────────────────────────────────────────────────
    def record_waypoint(self, name: str) -> None:
        st = self.get_status()
        self._store.record(name, st.joints, st.tcp)
        self._store.save()

    def waypoint_names(self) -> List[str]:
        return self._store.names()

    def delete_waypoint(self, name: str) -> None:
        if self._store.has(name):
            del self._store.points[name]
            self._store.save()

    def goto_waypoint(self, name: str, speed_scale: Optional[float] = None,
                      block: bool = True) -> None:
        if not self._store.has(name):
            raise KeyError("点位不存在: %s" % name)
        self.move_joints(self._store.get_joints(name), speed_scale=speed_scale, block=block)

    def get_waypoint_joints(self, name: str) -> List[float]:
        return self._store.get_joints(name)
