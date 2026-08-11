"""远程 HTTP 后端：本地 PyQt 软件经此把控制指令发给服务器代理，由服务器连真机。

实现 RobotArmBackendBase 全部接口，但每个操作都是一次 HTTP 请求到服务器的
webconsole/server.py 代理（同一套 /api/* 接口）。这样本地界面/3D/JOG/序列/相机
代码一行不改，只是后端从"直连机械臂"换成"调服务器代理"。

设计：
  - get_status() 返回本地缓存（后台线程每 ~200ms 轮询服务器 /api/status），
    避免 UI 高频刷新每次都卡在网络往返。
  - 控制类操作（connect/power/jog/goto/序列）是同步 HTTP POST，等服务器回执。
  - 真机的安全护栏在服务器端的 thrift 后端里（限位/钳速/到位轮询），本地只是转发。
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from typing import List, Optional

from .backend import RobotArmBackendBase, RobotArmStatus


class HttpRobotArmBackend(RobotArmBackendBase):
    """指向服务器代理的远程后端。base_url 形如 http://10.20.254.119:8080。"""

    _MOVE_TIMEOUT_S = 300.0   # 协调运动(move_joints)同步阻塞到位的 HTTP 超时上限

    def __init__(self, base_url: str, cfg: dict = None, timeout: float = 8.0):
        self.base_url = base_url.rstrip("/")
        self.cfg = cfg or {}
        self.timeout = timeout
        self._lock = threading.Lock()
        self._status = RobotArmStatus(connected=False)
        self._waypoints: List[str] = []
        self._sequences: List[str] = []
        self._seq_running = False
        self._seq_step = [0, 0, ""]
        self._poll_stop = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        # 远程服务器要连的真机参数（透传给 /api/connect）
        self._target_ip = (cfg or {}).get("connection", {}).get("ip", "192.168.1.10")
        self._target_port = (cfg or {}).get("connection", {}).get("rpc_port", 7003)
        self._link_ok = False
        self._last_err = ""
        self._gripper_status: Optional[dict] = None   # 夹爪状态缓存（服务器 /api/status 若提供）

    # ── HTTP 原语 ───────────────────────────────────────────
    def _get(self, path: str) -> dict:
        url = self.base_url + path
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _post(self, path: str, body: dict = None, timeout: float = None) -> dict:
        url = self.base_url + path
        data = json.dumps(body or {}).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _post_checked(self, path: str, body: dict = None, timeout: float = None):
        """POST 并把 {ok,error} 转成异常（与本地后端行为一致）。"""
        try:
            d = self._post(path, body, timeout=timeout)
        except Exception as e:
            raise RuntimeError("服务器通信失败: %s" % e)
        if not d.get("ok", False):
            raise RuntimeError(d.get("error") or "服务器拒绝操作")

    # ── 连接 / 轮询 ─────────────────────────────────────────
    def connect(self) -> None:
        # 先确认能连上服务器代理
        try:
            self._get("/api/status")
            self._link_ok = True
        except Exception as e:
            self._link_ok = False
            raise RuntimeError("连不上服务器代理 %s: %s" % (self.base_url, e))
        # 让服务器以 thrift 模式连真机
        self._post_checked("/api/connect", {
            "mode": "thrift", "ip": self._target_ip, "port": self._target_port})
        # 启动后台轮询
        self._poll_stop.clear()
        if self._poll_thread is None or not self._poll_thread.is_alive():
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._poll_thread.start()

    def disconnect(self) -> None:
        self._poll_stop.set()
        try:
            self._post("/api/disconnect")
        except Exception:
            pass
        with self._lock:
            self._status = RobotArmStatus(connected=False)

    def is_connected(self) -> bool:
        with self._lock:
            return self._status.connected

    def _poll_loop(self):
        fail_count = 0
        while not self._poll_stop.is_set():
            try:
                d = self._get("/api/status")
                self._apply_remote(d)
                self._link_ok = True
                fail_count = 0
            except Exception as e:
                fail_count += 1
                self._link_ok = False
                self._last_err = str(e)
                # 去抖：单次 /api/status 抖动(超时/相机拉流并发挤占)不清空已知良好状态；
                # 连续 3 次(~0.6s)才判真失联，避免瞬时抖动把 connected 误清成 False 引发闪烁/误报未就绪。
                if fail_count >= 3:
                    with self._lock:
                        self._status = RobotArmStatus(connected=False, error="服务器失联: %s" % e)
            self._poll_stop.wait(0.2)

    def _apply_remote(self, d: dict):
        st = RobotArmStatus(
            connected=d.get("connected", False),
            joints=d.get("joints", [0.0] * 6),
            tcp=d.get("tcp", [0.0] * 6),
            op_mode=d.get("op_mode", 0),
            robot_state=d.get("robot_state", 0),
            prog_state=d.get("prog_state", 0),
            safety=d.get("safety", 0),
            collision=d.get("collision", 0),
            collision_axis=d.get("collision_axis", 0),
            alarm=d.get("alarm", 0),
            powered=d.get("powered", False),
            enabled=d.get("enabled", False),
            error=d.get("error", ""),
        )
        with self._lock:
            self._status = st
            self._waypoints = d.get("waypoints", [])
            self._sequences = d.get("sequences", [])
            self._seq_running = d.get("seq_running", False)
            self._seq_step = d.get("seq_step", [0, 0, ""])
            self._gripper_status = d.get("gripper")   # 服务器若并入夹爪字段则缓存，缺失=None（容忍）

    # ── 状态（返回缓存）─────────────────────────────────────
    def get_status(self) -> RobotArmStatus:
        with self._lock:
            return self._status

    # ── 上电 / 使能 ─────────────────────────────────────────
    def power_on(self) -> None:
        # 管理员授权：开放软件上电（经 armweb -> 服务器 thrift -> DUCO RPC）。
        self._post_checked("/api/power_on")

    def enable(self) -> None:
        # 管理员授权：开放软件使能（经 armweb -> 服务器 thrift -> DUCO RPC）。
        self._post_checked("/api/enable")

    def disable(self) -> None:
        self._post_checked("/api/disable")

    def estop(self) -> None:
        """急停。失败必须抛出 —— 绝不能静默当成功。

        原来 `except: pass` 意味着网络/服务不可达时急停什么都没做，调用方
        （_safe_abort）却因为没有异常而记录"机械臂已急停"，在最需要急停生效的
        掉线场景下产生假阳性安全确认。
        """
        self._post("/api/estop")

    # ── 拖动示教（牵引/零力）────────────────────────────────────
    def enter_teach_mode(self, load_mass=0.0, cx=0.0, cy=0.0, cz=0.0) -> None:
        self._post_checked("/api/teach/enter", {
            "load_mass": load_mass, "cx": cx, "cy": cy, "cz": cz})
        self._teaching = True

    def exit_teach_mode(self) -> None:
        self._post_checked("/api/teach/exit")
        self._teaching = False

    def is_teaching(self) -> bool:
        return bool(getattr(self, "_teaching", False))

    # ── 运动 ────────────────────────────────────────────────
    def move_joints(self, joints, speed_scale=None, block=True) -> None:
        # 协调关节运动：POST armweb /api/move_joints → 服务器同步 movej_safe 到位后回执。
        # （关节回放 ARM_REPLAY_JOINTS 用此路径；不再逐轴 jog——逐轴会扫出危险中间构型。
        #   这就是 peel_cycle.py/batch_test.py 验证过的同一条协调 movej。）
        # 长超时：一次协调 movej 可能数十秒，默认 8s 不够；服务器多线程，其间 status 轮询不受影响。
        self._post_checked(
            "/api/move_joints",
            {"joints": [float(x) for x in joints], "speed_scale": speed_scale},
            timeout=self._MOVE_TIMEOUT_S,
        )

    def move_joints_seq(self, waypoints, speed_scale=None, blend_radius=0.0) -> None:
        # 连续多点关节运动：POST armweb /api/move_joints_seq → 服务器逐点 movej_safe(r=blend 融合中间点)。
        # blend_radius>0 用 DUCO movej2 融合半径过弯不停(减物理停顿,仅开阔段);=0 逐点到位(安全,等价多次 move_joints)。
        self._post_checked(
            "/api/move_joints_seq",
            {"waypoints": [[float(x) for x in q] for q in waypoints],
             "speed_scale": speed_scale, "blend_radius": float(blend_radius)},
            timeout=self._MOVE_TIMEOUT_S,
        )

    def jog_joint(self, index: int, delta_rad: float, speed_scale=None) -> None:
        import math
        # 转发 speed_scale：UI 慢速点动传 0.05，服务器 /api/jog 据此钳速（与 move_joints 一致）。
        # 之前漏传导致真机点动只用服务器默认速度，丢失"慢而安全"的意图。
        self._post_checked("/api/jog", {
            "joint": int(index), "delta_deg": math.degrees(delta_rad),
            "speed_scale": speed_scale})

    def jog_cart(self, axis: int, delta: float, speed_scale=None, dry_run: bool = False):
        """笛卡尔点动转发到服务器 /api/jog_cart（服务器侧用机器人逆解+受监督关节运动）。
        返回服务器回执 dict（dry_run 时含 target_joints/max_joint_delta_deg）。"""
        try:
            d = self._post("/api/jog_cart", {
                "axis": int(axis), "delta": float(delta),
                "speed_scale": speed_scale, "dry_run": bool(dry_run)},
                timeout=self._MOVE_TIMEOUT_S)
        except Exception as e:
            raise RuntimeError("服务器通信失败: %s" % e)
        if not d.get("ok", False):
            raise RuntimeError(d.get("error") or "笛卡尔点动被拒绝")
        return d

    def goto_pose(self, x, y, z, yaw, tool_offset=None, speed_scale=None,
                  block: bool = True, wait_timeout: float = 180.0) -> None:
        """旧单杯在线 IK 入口永久冻结；生产运动只能执行已审计冻结计划。"""
        raise RuntimeError(
            "goto_pose 旧单杯在线 IK 入口永久冻结；禁止向远端发送请求"
        )

    # ── 末端数字输出（吸盘）─────────────────────────────────────
    def set_digital_output(self, port: int, value: bool, block: bool = True) -> None:
        # do_num/active_high 的翻译在基类 set_gripper 里本地完成，
        # 服务器端点只收最终裸电平 port+value（服务器不需懂 gripper 语义）。
        # settle 等待也在本地基类发生，避免 HTTP 往返期间服务器侧 sleep 占连接。
        self._post_checked("/api/gripper", {"port": int(port), "value": bool(value)})

    def set_hotplate_pump(self, on: bool) -> None:
        """热台泵（继电器 DO2）——独立于机械臂状态，供 UI 手动开关调用。"""
        self._post_checked("/api/pump", {"which": "hotplate", "value": bool(on)})

    # ── 夹爪（大寰 PGEA，经服务器代理 /api/gripper/*）─────────────
    def gripper_initialize(self, full: bool = False) -> None:
        self._post_checked("/api/gripper/init", {"full": bool(full)})

    def gripper_grip(self, force=None, speed=None, pos=0) -> None:
        body = {"pos": int(pos)}
        if force is not None:
            body["force"] = int(force)
        if speed is not None:
            body["speed"] = int(speed)
        self._post_checked("/api/gripper/grip", body)

    def gripper_release(self, speed=None) -> None:
        body = {}
        if speed is not None:
            body["speed"] = int(speed)
        self._post_checked("/api/gripper/release", body)

    def gripper_status(self) -> dict:
        """从缓存取夹爪状态（后台轮询 /api/status 时并入）。服务器未提供则返回全 None（容忍缺失）。"""
        with self._lock:
            g = self._gripper_status
        if isinstance(g, dict):
            return {"init_state": g.get("init_state"), "grip_state": g.get("grip_state"),
                    "position": g.get("position"), "error": g.get("error")}
        return {"init_state": None, "grip_state": None, "position": None, "error": None}

    # ── 安全配置（透传，可选）────────────────────────────────
    def set_workspace_bounds(self, x=None, y=None, z=None) -> None:
        pass  # 远程暂不透传工作空间（服务器端用其 config）

    def set_nogo_boxes(self, boxes: list) -> None:
        pass  # 禁区在本地 3D 显示即可；如需服务器校验可加接口

    # ── 点位 ────────────────────────────────────────────────
    def record_waypoint(self, name: str) -> None:
        self._post_checked("/api/record_waypoint", {"name": name})

    def waypoint_names(self) -> List[str]:
        with self._lock:
            return list(self._waypoints)

    def delete_waypoint(self, name: str) -> None:
        self._post_checked("/api/delete_waypoint", {"name": name})

    def goto_waypoint(self, name: str, speed_scale=None, block=True) -> None:
        self._post_checked("/api/goto", {"name": name})
        # /api/goto 服务器侧是异步(起线程)；block=True 时轮询服务器 moving 到位再返回，
        # 保证 commander 顺序语义(如 arm_to_mid 必须退让到位后龙门才动，防撞机)。
        # 服务器受理 goto 时已同步置 moving=True，故此处轮询无竞态。
        if block:
            self._wait_until_idle()

    def _wait_until_idle(self, timeout: float = None, poll: float = 0.15) -> None:
        """阻塞到服务器 moving==False（异步运动的到位等待）。直接读 /api/status 求新鲜值。"""
        budget = timeout if timeout is not None else self._MOVE_TIMEOUT_S
        t0 = time.monotonic()
        while time.monotonic() - t0 < budget:
            try:
                d = self._get("/api/status")
            except Exception:
                time.sleep(poll); continue
            if not d.get("moving", False):
                return
            time.sleep(poll)
        raise RuntimeError("等待运动到位超时(%.0fs)" % budget)

    def get_waypoint_joints(self, name: str) -> List[float]:
        try:
            d = self._get("/api/waypoint_joints?name=" + urllib.parse.quote(name))
            return d.get("joints", [0.0] * 6)
        except Exception:
            return [0.0] * 6

    # ── 序列（供 UI 直接用）─────────────────────────────────
    def sequence_run(self, name, speed=None):
        self._post_checked("/api/sequence", {"action": "run", "name": name, "speed": speed})

    def sequence_pause(self):
        self._post_checked("/api/sequence", {"action": "pause"})

    def sequence_stop(self):
        self._post_checked("/api/sequence", {"action": "stop"})

    # ── 相机流地址（供本地 UI 显示）─────────────────────────
    def camera_stream_url(self) -> str:
        return self.base_url + "/api/camera/stream"
