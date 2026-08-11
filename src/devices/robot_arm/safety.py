"""安全模块: 心跳监控、速度钳制、工作空间/关节限位、起点偏差检查、急停恢复。

所有运动指令下发前应过 Safety 检查; 心跳线程独立运行, 断连即触发 ABORT。
"""
import math
import threading
import time


class Safety:
    """无状态(除 ABORT 标志)的安全检查集合 + 心跳管理。"""

    def __init__(self, cfg):
        self.cfg = cfg
        sp = cfg["speed"]
        jl = cfg["joint_limits"]
        ws = cfg["workspace"]
        sf = cfg["safety"]
        self.global_scale = float(sp["global_scale"])
        self.v_max_joint = float(sp["v_max_joint"])
        self.v_max_cart = float(sp["v_max_cart"])
        self.j_lower = [float(value) for value in jl["lower"]]
        self.j_upper = [float(value) for value in jl["upper"]]
        self.ws_x = [float(value) for value in ws["x"]]
        self.ws_y = [float(value) for value in ws["y"]]
        self.ws_z = [float(value) for value in ws["z"]]
        self.start_dev_tol = math.radians(float(sf["start_deviation_tol_deg"]))
        self._validate_config(sf)
        self._hb = None
        self._hb_lock = threading.RLock()
        self.aborted = False

    def _validate_config(self, safety_cfg):
        positive_values = {
            "speed.global_scale": self.global_scale,
            "speed.v_max_joint": self.v_max_joint,
            "speed.v_max_cart": self.v_max_cart,
            "safety.heartbeat_period_s": safety_cfg["heartbeat_period_s"],
            "safety.start_deviation_tol_deg": safety_cfg["start_deviation_tol_deg"],
        }
        for name, value in positive_values.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError("%s 必须是正有限数" % name)
        if self.global_scale > 1.0:
            raise ValueError("speed.global_scale 不得超过 1.0")
        heartbeat_max_fail = float(safety_cfg["heartbeat_max_fail"])
        if (
            not math.isfinite(heartbeat_max_fail)
            or not heartbeat_max_fail.is_integer()
            or heartbeat_max_fail < 1
        ):
            raise ValueError("safety.heartbeat_max_fail 必须 >= 1")
        if not (
            len(self.j_lower) == len(self.j_upper) == 6
            and all(
                math.isfinite(float(lower))
                and math.isfinite(float(upper))
                and float(lower) < float(upper)
                for lower, upper in zip(self.j_lower, self.j_upper)
            )
        ):
            raise ValueError("joint_limits 必须包含 6 组有效上下限")
        for name, bounds in (("x", self.ws_x), ("y", self.ws_y), ("z", self.ws_z)):
            if not (
                len(bounds) == 2
                and all(math.isfinite(float(value)) for value in bounds)
                and float(bounds[0]) < float(bounds[1])
            ):
                raise ValueError("workspace.%s 边界无效" % name)

    # ---------- 速度 ----------
    def clamp_joint_v(self, v):
        """关节速度: 先按 global_scale 缩放, 再钳到上限。"""
        return min(v * self.global_scale, self.v_max_joint)

    def clamp_cart_v(self, v):
        return min(v * self.global_scale, self.v_max_cart)

    # ---------- 限位 ----------
    def check_joint_limits(self, joints):
        """返回 (ok, msg)。任一关节越软限位则 ok=False。"""
        if len(joints) != 6 or not all(math.isfinite(float(value)) for value in joints):
            return False, "关节必须是 6 个有限数"
        for i, q in enumerate(joints):
            if q < self.j_lower[i] or q > self.j_upper[i]:
                return False, "关节%d=%.3frad 越限[%.3f,%.3f]" % (
                    i + 1, q, self.j_lower[i], self.j_upper[i])
        return True, "ok"

    def check_workspace(self, pose):
        """pose: [x,y,z,...] base系m。越 box 则拒绝。"""
        if len(pose) < 3 or not all(math.isfinite(float(value)) for value in pose[:3]):
            return False, "TCP 位姿至少需要 3 个有限坐标"
        x, y, z = pose[0], pose[1], pose[2]
        if not (self.ws_x[0] <= x <= self.ws_x[1]):
            return False, "X=%.3f 越界%s" % (x, self.ws_x)
        if not (self.ws_y[0] <= y <= self.ws_y[1]):
            return False, "Y=%.3f 越界%s" % (y, self.ws_y)
        if not (self.ws_z[0] <= z <= self.ws_z[1]):
            return False, "Z=%.3f 越界%s" % (z, self.ws_z)
        return True, "ok"

    def check_start_deviation(self, current_joints, track_start):
        """回放前: 当前关节 vs 轨迹起点, 最大偏差超容差则需先 goto start。
        返回 (within_tol, max_dev_rad)。"""
        max_dev = max(abs(a - b) for a, b in zip(current_joints, track_start))
        return max_dev <= self.start_dev_tol, max_dev

    # ---------- 心跳 ----------
    def start_heartbeat(self, read_state_fn, on_abort):
        """启动后台心跳。read_state_fn() 应返回 state list 或抛异常;
        连续失败 max_fail 次 -> 调 on_abort() 并置 aborted。"""
        self.stop_heartbeat()
        heartbeat = _Heartbeat(read_state_fn, on_abort, self,
                               float(self.cfg["safety"]["heartbeat_period_s"]),
                               int(self.cfg["safety"]["heartbeat_max_fail"]))
        with self._hb_lock:
            if self._hb is not None:
                raise RuntimeError("旧心跳线程尚未退出，拒绝启动双心跳")
            self._hb = heartbeat
            heartbeat.start()
        return heartbeat

    def heartbeat_running(self):
        with self._hb_lock:
            return self._hb is not None and self._hb.is_alive()

    def stop_heartbeat(self, timeout_s=10.0):
        with self._hb_lock:
            heartbeat = self._hb
            if heartbeat is None:
                return
            heartbeat.stop()
        if threading.current_thread() is heartbeat:
            return
        heartbeat.join(timeout=float(timeout_s))
        if heartbeat.is_alive():
            raise RuntimeError("心跳线程未能在超时内退出")
        self._heartbeat_exited(heartbeat)

    def _heartbeat_exited(self, heartbeat):
        with self._hb_lock:
            if self._hb is heartbeat:
                self._hb = None


class _Heartbeat(threading.Thread):
    def __init__(self, read_fn, on_abort, safety, period, max_fail):
        super().__init__(daemon=True)
        self.read_fn = read_fn
        self.on_abort = on_abort
        self.safety = safety
        self.period = period
        self.max_fail = max_fail
        self._stop_event = threading.Event()
        self.last_state = None
        self.fail_count = 0

    def run(self):
        try:
            while not self._stop_event.is_set():
                try:
                    self.last_state = self.read_fn()
                    self.fail_count = 0
                except Exception:
                    if self._stop_event.is_set():
                        break
                    self.fail_count += 1
                    if self.fail_count >= self.max_fail and not self.safety.aborted:
                        self.safety.aborted = True
                        try:
                            self.on_abort()
                        except Exception:
                            pass
                self._stop_event.wait(self.period)
        finally:
            self.safety._heartbeat_exited(self)

    def stop(self):
        self._stop_event.set()
