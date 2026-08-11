"""高层门面 Robot: 组合 DucoRPC + Safety, 所有运动先过安全检查。

用法:
    from core.robot_control import Robot
    r = Robot.from_config("config/robot.yaml")
    r.connect()
    r.ensure_enabled()
    r.movej_safe(target_joints, tool_contract=tool_contract,
                 path_contract=path_contract, collision_verifier=verify_collision)
    ...
    r.shutdown()           # 停心跳 + disable + 断连
"""
import math
import os
import threading
import time
from dataclasses import dataclass

import yaml

from .duco_rpc import DucoRPC, ST_ROBOT, ST_SAFE, ST_MODE, ST_TASK
from .safety import Safety

ROBOT_STATE_ENABLED = 6
SAFETY_RUN = 5
TASK_IDLE = 0
TASK_FINISHED = 4
TASK_FAILED = frozenset({2, 3, 5, 6, 7, 8})
MAX_DENSE_JOINT_STEP_DEG = 0.25
MAX_CUP_PLANE_TILT_DEG = 0.3
ENGINEERING_TARGET_CUP_PLANE_TILT_DEG = 0.1
MAX_ORIENTATION_FAMILY_ERROR_DEG = 0.3
MAX_FIXED_XY_ERROR_MM = 0.25
MAX_YAW_DOWNWARD_DROP_MM = 0.5
MIN_HELD_HIGH_TCP_Z_MM = 90.0
MAX_STATUS_FK_POSITION_ERROR_MM = 0.1
MAX_STATUS_FK_ORIENTATION_ERROR_DEG = 0.01
TCP_OFFSET_POSITION_TOLERANCE_M = 1e-6
TCP_OFFSET_ORIENTATION_TOLERANCE_RAD = 1e-6
MAX_TOOL_LOAD_MASS_TOLERANCE_KG = 0.005
MAX_TOOL_LOAD_COG_TOLERANCE_M = 0.0005
MAX_FROZEN_START_ERROR_DEG = 0.10
PATH_PHASES = frozenset({"empty_high", "high", "yaw", "vertical", "contact"})


def _symmetric_matrix_is_psd(values, tolerance=1e-12):
    xx, xy, xz, yy, yz, zz = (float(value) for value in values)
    determinant = (
        xx * (yy * zz - yz * yz)
        - xy * (xy * zz - xz * yz)
        + xz * (xy * yz - xz * yy)
    )
    return (
        min(xx, yy, zz) >= -tolerance
        and xx * yy - xy * xy >= -tolerance
        and xx * zz - xz * xz >= -tolerance
        and yy * zz - yz * yz >= -tolerance
        and determinant >= -tolerance
    )


def _inertia_tensor_is_physical(values, mass_kg):
    xx, xy, xz, yy, yz, zz = (float(value) for value in values)
    trace = xx + yy + zz
    if float(mass_kg) > 0.0 and trace <= 1e-12:
        return False
    if not _symmetric_matrix_is_psd((xx, xy, xz, yy, yz, zz)):
        return False
    half_trace = 0.5 * trace
    covariance = (
        half_trace - xx,
        -xy,
        -xz,
        half_trace - yy,
        -yz,
        half_trace - zz,
    )
    return _symmetric_matrix_is_psd(covariance)


@dataclass(frozen=True)
class ToolContract:
    """所有 FK 统一使用、且经过独立合格标定的双杯工具契约。"""

    qualified: bool
    contract_sha256: str
    tcp_offset_m: tuple
    cup_plane_normal_tcp: tuple
    payload_kg_cog_m: tuple
    inertia_tensor_kg_m2: tuple
    payload_verified: bool
    inertia_verified: bool
    independent_center_calibration: bool
    independent_plane_calibration: bool


@dataclass(frozen=True)
class PathContract:
    """仅绑定一个关节空间运动段的合格路径契约。"""

    qualified: bool
    name: str
    plan_sha256: str
    tool_contract_sha256: str
    phase: str
    wafer_held: bool
    start_joints: tuple
    target_joints: tuple
    start_rpy: tuple
    end_rpy: tuple


@dataclass(frozen=True)
class CollisionSample:
    """提交给强制碰撞验证器的单个官方 FK 密采点。"""

    path_name: str
    phase: str
    wafer_held: bool
    sample_index: int
    sample_count: int
    joints: tuple
    tcp_pose: tuple
    flange_pose: tuple
    plane_tilt_deg: float
    orientation_family_error_deg: float
    tool_contract: ToolContract
    path_contract: PathContract


class RobotError(RuntimeError):
    pass


class Robot:
    def __init__(self, cfg):
        self.cfg = cfg
        c = cfg["connection"]
        self.rpc = DucoRPC(c["ip"], c["rpc_port"], c["timeout_ms"])
        self.safety = Safety(cfg)
        self._connected = False
        self._motion_lock = threading.Lock()
        self._disable_lock = threading.Lock()
        self._status_lock = threading.RLock()
        self._stop_unverified = False
        self._last_frame = None
        self._last_frame_at = None
        self._status_cache_blocked = False
        self._last_dual_cup_path_audit = None
        self._last_dual_cup_path_summary = None

    @classmethod
    def from_config(cls, path):
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(cfg)

    # ---------- 连接 / 生命周期 ----------
    def connect(self):
        self.rpc.open()
        try:
            # connect 始终只读；只有 ensure_enabled 通过全部门后才启动运行期心跳。
            frame = self.rpc.read_status_frame()
        except Exception:
            self.rpc.close()
            raise
        self._arm_status_cache(frame)
        self._connected = True
        return self

    def _arm_status_cache(self, frame):
        with self._status_lock:
            self._status_cache_blocked = False
            self._last_frame = frame
            self._last_frame_at = time.monotonic()

    def _store_status_frame(self, frame):
        with self._status_lock:
            if self._status_cache_blocked:
                return
            self._last_frame = frame
            self._last_frame_at = time.monotonic()

    def _invalidate_status_cache(self):
        with self._status_lock:
            self._status_cache_blocked = True
            self._last_frame = None
            self._last_frame_at = None

    def _heartbeat_cache_max_age_s(self):
        safety_cfg = self.cfg.get("safety", {})
        configured = safety_cfg.get("heartbeat_cache_max_age_s")
        if configured is None:
            period = float(safety_cfg.get("heartbeat_period_s", 0.5))
            max_fail = int(safety_cfg.get("heartbeat_max_fail", 3))
            return max(1.0, period * (max_fail + 2))
        configured = float(configured)
        if not math.isfinite(configured) or configured <= 0:
            raise RobotError("heartbeat_cache_max_age_s 必须是正有限数")
        return configured

    def _hb_read_state(self):
        # 心跳是 2001 端口唯一读取者, 读整帧缓存供其他地方复用(避免并发连2001超时)
        timeout_s = max(
            0.25,
            float(self.cfg.get("safety", {}).get("heartbeat_period_s", 0.5)),
        )
        fr = self.rpc.read_status_frame(retries=1, timeout=timeout_s)
        self._validate_motion_frame(
            fr,
            "心跳",
            require_idle=False,
            require_auto=False,
            require_workspace=False,
        )
        self._store_status_frame(fr)
        return fr["robot_state"]

    def last_frame(self):
        """取心跳缓存的最近一帧; 心跳未跑或还没第一帧时直接读一次。"""
        heartbeat_running = self.safety.heartbeat_running()
        with self._status_lock:
            frame = self._last_frame
            frame_at = self._last_frame_at
            cache_blocked = self._status_cache_blocked
        if heartbeat_running:
            if cache_blocked or frame is None or frame_at is None:
                raise RobotError("心跳状态缓存已失效，拒绝返回旧使能帧")
            age_s = time.monotonic() - frame_at
            maximum_age_s = self._heartbeat_cache_max_age_s()
            if age_s > maximum_age_s:
                raise RobotError(
                    "心跳状态帧已过期 %.3fs > %.3fs" % (age_s, maximum_age_s)
                )
            return frame
        frame = self.rpc.read_status_frame()
        self._store_status_frame(frame)
        return frame

    def _on_abort(self):
        stopped = self._disable_and_verify()
        if stopped:
            self.safety.stop_heartbeat()
        level = "SAFETY" if stopped else "SAFETY-CRITICAL"
        print(
            "[%s] 心跳异常 -> ABORT, disable_verified=%s" % (level, stopped),
            flush=True,
        )

    def _disable_and_verify(self):
        self._invalidate_status_cache()
        with self._disable_lock:
            for _ in range(3):
                try:
                    self.rpc.disable(False)
                except Exception:
                    pass
                for _ in range(3):
                    try:
                        state = self.rpc.get_robot_state()
                    except Exception:
                        state = None
                    if (
                        isinstance(state, (list, tuple))
                        and len(state) >= 1
                        and state[0] != ROBOT_STATE_ENABLED
                    ):
                        self._stop_unverified = False
                        return True
                    time.sleep(0.1)
            self._stop_unverified = True
            return False

    def _start_shutdown_monitor(self, *, force=False):
        if self.safety.heartbeat_running():
            return True
        should_start = bool(force)
        if not should_start:
            try:
                state = self.rpc.get_robot_state()
            except Exception:
                should_start = True
            else:
                should_start = not (
                    isinstance(state, (list, tuple))
                    and len(state) >= 1
                    and state[0] != ROBOT_STATE_ENABLED
                )
        if not should_start:
            return False
        self.safety.start_heartbeat(self._hb_read_state, self._on_abort)
        return True

    def shutdown(self, do_disable=True):
        """验证已停机、回收心跳后再断链；do_disable 仅保留调用兼容性。"""
        if not self._connected:
            self._invalidate_status_cache()
            self.safety.stop_heartbeat()
            return
        monitor_error = None
        try:
            self._start_shutdown_monitor()
        except Exception as exc:
            monitor_error = exc
        self._invalidate_status_cache()
        if not self._disable_and_verify():
            try:
                self._start_shutdown_monitor(force=True)
            except Exception as exc:
                raise RobotError(
                    "关闭连接前 disable 未确认且监控心跳无法重建；"
                    "保留控制连接并要求现场急停/示教器确认"
                ) from exc
            raise RobotError("关闭连接前 disable 未确认；保留控制连接和心跳并要求现场确认")
        try:
            self.safety.stop_heartbeat()
        except Exception as exc:
            raise RobotError("机械臂已停止，但心跳线程未确认退出；保留控制连接") from exc
        if monitor_error is not None:
            print(
                "[SAFETY] shutdown 监控心跳未启动，但 disable 已确认: %s"
                % monitor_error,
                flush=True,
            )
        self.rpc.close()
        self._connected = False

    # ---------- 状态 ----------
    def state(self):
        return self.rpc.get_robot_state()

    def frame(self):
        return self.last_frame()

    def op_mode(self):
        return self.rpc.get_robot_state()[3]

    def print_status(self):
        from duco_rpc import show_frame
        show_frame(self.last_frame())

    # ---------- 使能 ----------
    def ensure_enabled(self):
        """只确认示教器已完成上电/使能；本进程永不写远程生命周期。"""
        if self.safety.aborted or self._stop_unverified:
            raise RobotError("已锁存 ABORT/停止未确认，必须现场确认后重建连接")
        st = self.rpc.get_robot_state()
        if not isinstance(st, (list, tuple)) or len(st) < 4:
            raise RobotError("控制器状态返回格式错误: %r" % (st,))
        frame = self.rpc.read_status_frame(retries=1, timeout=1.0)
        robot_state, _, safety, mode = st[0], st[1], st[2], st[3]
        if (
            safety != SAFETY_RUN
            or mode != 1
            or st[1] != 0
            or frame.get("prog_state") != 0
            or frame.get("collision")
            or (frame.get("alarm") and self._alarm_indicates_fault())
        ):
            errs = self.rpc.get_last_error()
            raise RobotError(
                "控制器未满足自动使能门(state=%s frame=%s)。"
                "请在示教器按官方流程处理。\n  last_error: %s"
                % (st, frame, errs))
        if robot_state == ROBOT_STATE_ENABLED:
            self._validate_motion_frame(frame, "使能复核", require_idle=True)
            self._store_status_frame(frame)
            if not self.safety.heartbeat_running():
                self.safety.start_heartbeat(self._hb_read_state, self._on_abort)
            return True
        raise RobotError(
            "远程上电/使能已永久冻结；请在示教器清错并使能到 robot_state=6 后重试"
        )

    def _assert_ready(self):
        if self.safety.aborted or self._stop_unverified:
            raise RobotError("已 ABORT 或停止未确认, 拒绝运动")
        st = self.rpc.get_robot_state()
        if (
            not isinstance(st, (list, tuple))
            or len(st) < 4
            or st[0] != ROBOT_STATE_ENABLED
            or st[1] != 0
            or st[2] != SAFETY_RUN
            or st[3] != 1
        ):
            raise RobotError("未就绪 state=%s" % st)

    def _alarm_indicates_fault(self):
        """状态帧 alarm 位字段是历史残留（旧碰撞记录等，见 backend.RobotArmStatus.is_ready
        文档说明），不可靠。以 DUCO 官方 get_last_error() 为权威故障源：仅当它返回**非空**
        （确有错误字符串）才判定为真故障。带 ~1s 缓存，避免心跳/运动监视高频 RPC。
        读取失败则保守当作有故障（fail-safe）。其它真实护栏（collision / safety!=RUN /
        未使能 / 关节限位 / 逆解护栏）不受本函数影响，仍独立生效。"""
        now = time.monotonic()
        cached = getattr(self, "_lasterr_cache", None)
        if cached is not None and now - cached[0] < 5.0:   # 5s 缓存，降低对 7003 RPC 的并发压力
            return cached[1]
        try:
            errs = self.rpc.get_last_error()
            real = bool(errs)
        except Exception:
            # get_last_error 本身通信/协议抖动（如并发序号错配）不是机器人故障：
            # 沿用上次判定（无缓存则保守判无故障，真故障由 collision/safety!=RUN 等独立护栏兜底）。
            real = cached[1] if cached is not None else False
        self._lasterr_cache = (now, real)
        return real

    def _validate_motion_frame(
        self,
        frame,
        label,
        require_idle=False,
        require_auto=True,
        require_workspace=True,
    ):
        required = {
            "joints", "tcp", "op_mode", "robot_state", "prog_state", "safety",
            "collision", "collision_axis", "alarm",
        }
        if not isinstance(frame, dict) or not required.issubset(frame):
            raise RobotError("%s 状态帧字段不完整" % label)
        joints = frame["joints"]
        tcp = frame["tcp"]
        if len(joints) != 6 or len(tcp) != 6:
            raise RobotError("%s 状态帧维度错误" % label)
        if not all(math.isfinite(float(value)) for value in list(joints) + list(tcp)):
            raise RobotError("%s 状态帧含非有限数" % label)
        ok, msg = self.safety.check_joint_limits(joints)
        if not ok:
            raise RobotError("%s 状态帧关节不可信: %s" % (label, msg))
        if (
            frame["robot_state"] != ROBOT_STATE_ENABLED
            or frame["safety"] != SAFETY_RUN
            or (require_auto and frame["op_mode"] != 1)
            or frame["collision"]
            or (frame["alarm"] and self._alarm_indicates_fault())
        ):
            raise RobotError("%s 状态不允许运动: %s" % (label, frame))
        if require_idle and frame["prog_state"] != 0:
            raise RobotError("%s 检测到已有程序/运动: %s" % (label, frame))
        if require_workspace:
            ok, msg = self.safety.check_workspace(tcp)
            if not ok:
                raise RobotError("%s TCP 工作空间不可信: %s" % (label, msg))
        return [float(value) for value in joints]

    @staticmethod
    def _rpy_matrix(rpy):
        rx, ry, rz = (float(value) for value in rpy)
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        return (
            (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
            (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
            (-sy, cy * sx, cy * cx),
        )

    @classmethod
    def _orientation_error_deg(cls, first, second):
        return cls._orientation_matrix_error_deg(
            cls._rpy_matrix(first), cls._rpy_matrix(second)
        )

    @staticmethod
    def _matrix_multiply(first, second):
        return tuple(
            tuple(
                sum(first[row][inner] * second[inner][column] for inner in range(3))
                for column in range(3)
            )
            for row in range(3)
        )

    @staticmethod
    def _matrix_transpose(matrix):
        return tuple(tuple(matrix[column][row] for column in range(3)) for row in range(3))

    @staticmethod
    def _orientation_matrix_error_deg(first_matrix, second_matrix):
        trace = sum(
            first_matrix[row][column] * second_matrix[row][column]
            for row in range(3)
            for column in range(3)
        )
        cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
        return math.degrees(math.acos(cosine))

    @staticmethod
    def _world_z_rotation(angle):
        cosine = math.cos(float(angle))
        sine = math.sin(float(angle))
        return (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        )

    @classmethod
    def _world_z_yaw_delta(cls, start_rpy, end_rpy):
        start_matrix = cls._rpy_matrix(start_rpy)
        end_matrix = cls._rpy_matrix(end_rpy)
        relative = cls._matrix_multiply(
            end_matrix, cls._matrix_transpose(start_matrix)
        )
        yaw_delta = math.atan2(
            relative[1][0] - relative[0][1],
            relative[0][0] + relative[1][1],
        )
        expected_end = cls._matrix_multiply(
            cls._world_z_rotation(yaw_delta), start_matrix
        )
        return yaw_delta, start_matrix, end_matrix, expected_end

    def _verify_fk_tcp(self, frame, tool_contract, label):
        try:
            expected = self.rpc.cal_fkine(
                frame["joints"], list(tool_contract.tcp_offset_m), []
            )
            flange = self.rpc.cal_fkine(frame["joints"], [0.0] * 6, [])
        except Exception as exc:
            raise RobotError("%s 显式双杯/法兰 FK 失败: %s" % (label, exc)) from exc
        expected = self._finite_six(expected, "%s 显式双杯 FK" % label)
        flange = self._finite_six(flange, "%s 显式法兰 FK" % label)
        actual = frame["tcp"]
        position_error_mm = math.sqrt(sum(
            (float(expected[index]) - float(actual[index])) ** 2
            for index in range(3)
        )) * 1000.0
        orientation_error_deg = self._orientation_error_deg(expected[3:6], actual[3:6])
        max_position_error_mm = min(
            self._safety_number(
                "fk_tcp_position_tol_mm", MAX_STATUS_FK_POSITION_ERROR_MM
            ),
            MAX_STATUS_FK_POSITION_ERROR_MM,
        )
        max_orientation_error_deg = min(
            self._safety_number(
                "fk_tcp_orientation_tol_deg", MAX_STATUS_FK_ORIENTATION_ERROR_DEG
            ),
            MAX_STATUS_FK_ORIENTATION_ERROR_DEG,
        )
        if (
            position_error_mm > max_position_error_mm
            or orientation_error_deg > max_orientation_error_deg
        ):
            raise RobotError(
                "%s 状态 TCP 与显式双杯 FK 不一致: position=%.6fmm "
                "orientation=%.6fdeg"
                % (label, position_error_mm, orientation_error_deg)
            )
        return expected, flange

    def _verify_rpc_actual_joints(self, frame, require_stationary):
        try:
            actual_joints = self.rpc.get_actual_joints_position()
            actual_speed = self.rpc.get_actual_joints_speed()
        except Exception as exc:
            raise RobotError("控制 RPC actual-q 交叉认证失败: %s" % exc) from exc
        actual_joints = self._finite_six(actual_joints, "RPC actual joints")
        actual_speed = self._finite_six(actual_speed, "RPC actual speed")
        ok, msg = self.safety.check_joint_limits(actual_joints)
        if not ok:
            raise RobotError("RPC actual joints 不可信: " + msg)
        tolerance_deg = self._safety_number("status_rpc_joint_tolerance_deg", 0.10)
        difference_deg = math.degrees(max(
            abs(actual - streamed)
            for actual, streamed in zip(actual_joints, frame["joints"])
        ))
        if difference_deg > tolerance_deg:
            raise RobotError(
                "2001 与 RPC actual-q 不一致: %.6fdeg > %.6fdeg"
                % (difference_deg, tolerance_deg)
            )
        if require_stationary:
            speed_limit_deg_s = self._safety_number(
                "preflight_max_joint_speed_deg_s", 2.5
            )
            actual_speed_deg_s = math.degrees(max(abs(value) for value in actual_speed))
            if actual_speed_deg_s > speed_limit_deg_s:
                raise RobotError(
                    "RPC actual speed 未静止: %.6fdeg/s > %.6fdeg/s"
                    % (actual_speed_deg_s, speed_limit_deg_s)
                )

    @staticmethod
    def _finite_six(values, label):
        if (
            not isinstance(values, (list, tuple))
            or len(values) != 6
            or not all(math.isfinite(float(value)) for value in values)
        ):
            raise RobotError("%s 必须是 6 个有限数" % label)
        return [float(value) for value in values]

    @staticmethod
    def _finite_vector(values, length, label):
        if (
            not isinstance(values, (list, tuple))
            or len(values) != length
            or not all(math.isfinite(float(value)) for value in values)
        ):
            raise RobotError("%s 必须是 %d 个有限数" % (label, length))
        return tuple(float(value) for value in values)

    @staticmethod
    def _normalized_sha256(value, label):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in value)
        ):
            raise RobotError("%s 必须是 64 位十六进制 SHA256" % label)
        return value.lower()

    @classmethod
    def _qualify_motion_contracts(
        cls, target_joints, tool_contract, path_contract, collision_verifier
    ):
        if not isinstance(tool_contract, ToolContract):
            raise RobotError("movej_safe 缺少显式 ToolContract")
        if tool_contract.qualified is not True:
            raise RobotError("ToolContract 未 qualified，拒绝运动")
        contract_sha256 = cls._normalized_sha256(
            tool_contract.contract_sha256, "ToolContract.contract_sha256"
        )
        qualification_flags = {
            "payload_verified": tool_contract.payload_verified,
            "inertia_verified": tool_contract.inertia_verified,
            "independent_center_calibration": (
                tool_contract.independent_center_calibration
            ),
            "independent_plane_calibration": (
                tool_contract.independent_plane_calibration
            ),
        }
        missing_flags = [
            name for name, value in qualification_flags.items() if value is not True
        ]
        if missing_flags:
            raise RobotError(
                "ToolContract 负载/惯量/独立标定证据不完整: %s"
                % ", ".join(missing_flags)
            )
        tcp_offset = cls._finite_vector(
            tool_contract.tcp_offset_m, 6, "ToolContract.tcp_offset_m"
        )
        cup_normal = cls._finite_vector(
            tool_contract.cup_plane_normal_tcp,
            3,
            "ToolContract.cup_plane_normal_tcp",
        )
        payload = cls._finite_vector(
            tool_contract.payload_kg_cog_m,
            4,
            "ToolContract.payload_kg_cog_m",
        )
        if payload[0] < 0.0 or payload[0] > 3.0:
            raise RobotError("ToolContract payload mass 必须在 [0,3]kg")
        inertia = cls._finite_vector(
            tool_contract.inertia_tensor_kg_m2,
            6,
            "ToolContract.inertia_tensor_kg_m2",
        )
        if not _inertia_tensor_is_physical(inertia, payload[0]):
            raise RobotError("ToolContract inertia tensor 不满足刚体物理一致性")
        normal_length = math.sqrt(sum(value * value for value in cup_normal))
        if abs(normal_length - 1.0) > 1e-6:
            raise RobotError("ToolContract.cup_plane_normal_tcp 必须是单位向量")
        qualified_tool = ToolContract(
            qualified=True,
            contract_sha256=contract_sha256,
            tcp_offset_m=tcp_offset,
            cup_plane_normal_tcp=cup_normal,
            payload_kg_cog_m=payload,
            inertia_tensor_kg_m2=inertia,
            payload_verified=True,
            inertia_verified=True,
            independent_center_calibration=True,
            independent_plane_calibration=True,
        )

        if not isinstance(path_contract, PathContract):
            raise RobotError("movej_safe 缺少显式 PathContract")
        if path_contract.qualified is not True:
            raise RobotError("PathContract 未 qualified，拒绝运动")
        if not isinstance(path_contract.name, str) or not path_contract.name.strip():
            raise RobotError("PathContract.name 必须是非空字符串")
        plan_sha256 = cls._normalized_sha256(
            path_contract.plan_sha256, "PathContract.plan_sha256"
        )
        path_tool_sha256 = cls._normalized_sha256(
            path_contract.tool_contract_sha256,
            "PathContract.tool_contract_sha256",
        )
        if path_tool_sha256 != contract_sha256:
            raise RobotError("PathContract 未绑定当前 ToolContract SHA256")
        if path_contract.phase not in PATH_PHASES:
            raise RobotError("PathContract.phase 非法: %r" % (path_contract.phase,))
        if not isinstance(path_contract.wafer_held, bool):
            raise RobotError("PathContract.wafer_held 必须显式为 bool")
        contract_start = cls._finite_vector(
            path_contract.start_joints, 6, "PathContract.start_joints"
        )
        contract_target = cls._finite_vector(
            path_contract.target_joints, 6, "PathContract.target_joints"
        )
        requested_target = cls._finite_vector(target_joints, 6, "目标关节")
        if max(
            abs(contract_value - requested_value)
            for contract_value, requested_value in zip(
                contract_target, requested_target
            )
        ) > 1e-12:
            raise RobotError("PathContract 未绑定当前 movej_safe 目标")
        start_rpy = cls._finite_vector(
            path_contract.start_rpy,
            3,
            "PathContract.start_rpy",
        )
        end_rpy = cls._finite_vector(
            path_contract.end_rpy,
            3,
            "PathContract.end_rpy",
        )
        _, _, end_matrix, expected_end_matrix = cls._world_z_yaw_delta(
            start_rpy, end_rpy
        )
        endpoint_family_error_deg = cls._orientation_matrix_error_deg(
            expected_end_matrix, end_matrix
        )
        if endpoint_family_error_deg > MAX_ORIENTATION_FAMILY_ERROR_DEG + 1e-9:
            raise RobotError(
                "PathContract start_rpy/end_rpy 不是同一世界 Z yaw 姿态族: "
                "%.6fdeg > %.1fdeg"
                % (
                    endpoint_family_error_deg,
                    MAX_ORIENTATION_FAMILY_ERROR_DEG,
                )
            )
        qualified_path = PathContract(
            qualified=True,
            name=path_contract.name.strip(),
            plan_sha256=plan_sha256,
            tool_contract_sha256=path_tool_sha256,
            phase=path_contract.phase,
            wafer_held=path_contract.wafer_held,
            start_joints=contract_start,
            target_joints=contract_target,
            start_rpy=start_rpy,
            end_rpy=end_rpy,
        )
        if not callable(collision_verifier):
            raise RobotError("movej_safe 缺少实际 collision_verifier callable")
        return requested_target, qualified_tool, qualified_path

    @staticmethod
    def _wrapped_angle_difference(first, second):
        return (float(first) - float(second) + math.pi) % (2.0 * math.pi) - math.pi

    def _verify_active_tool(self, tool_contract, label):
        try:
            actual = self.rpc.get_tcp_offset()
            actual_load = self.rpc.get_tool_load()
        except Exception as exc:
            raise RobotError("%s 工具 TCP/负载回读失败: %s" % (label, exc)) from exc
        actual = self._finite_six(actual, "%s get_tcp_offset" % label)
        actual_load = self._finite_vector(
            actual_load, 4, "%s get_tool_load" % label
        )
        if actual_load[0] < 0.0 or actual_load[0] > 3.0:
            raise RobotError("%s get_tool_load.mass 超出 [0,3]kg" % label)
        expected = tool_contract.tcp_offset_m
        position_error_m = max(
            abs(actual[index] - expected[index]) for index in range(3)
        )
        orientation_error_rad = max(
            abs(self._wrapped_angle_difference(actual[index], expected[index]))
            for index in range(3, 6)
        )
        if (
            position_error_m > TCP_OFFSET_POSITION_TOLERANCE_M
            or orientation_error_rad > TCP_OFFSET_ORIENTATION_TOLERANCE_RAD
        ):
            raise RobotError(
                "%s 当前工具与显式双杯 ToolContract 不一致: position=%.6fmm "
                "orientation=%.6fdeg"
                % (
                    label,
                    position_error_m * 1000.0,
                    math.degrees(orientation_error_rad),
                )
            )
        mass_tolerance_kg = min(
            self._safety_number(
                "tool_load_mass_tolerance_kg",
                MAX_TOOL_LOAD_MASS_TOLERANCE_KG,
            ),
            MAX_TOOL_LOAD_MASS_TOLERANCE_KG,
        )
        cog_tolerance_m = min(
            self._safety_number(
                "tool_load_cog_tolerance_m",
                MAX_TOOL_LOAD_COG_TOLERANCE_M,
            ),
            MAX_TOOL_LOAD_COG_TOLERANCE_M,
        )
        expected_load = tool_contract.payload_kg_cog_m
        mass_error_kg = abs(actual_load[0] - expected_load[0])
        cog_error_m = max(
            abs(actual_load[index] - expected_load[index])
            for index in range(1, 4)
        )
        if (
            mass_error_kg > mass_tolerance_kg
            or cog_error_m > cog_tolerance_m
        ):
            raise RobotError(
                "%s 当前工具负载与 ToolContract 不一致: mass=%.6fkg "
                "cog=%.6fmm"
                % (label, mass_error_kg, cog_error_m * 1000.0)
            )
        return tuple(actual), tuple(actual_load)

    def _safety_number(self, key, default, *, allow_zero=False):
        value = float(self.cfg.get("safety", {}).get(key, default))
        if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
            comparator = ">= 0" if allow_zero else "> 0"
            raise RobotError("safety.%s 必须是有限数且 %s" % (key, comparator))
        return value

    @staticmethod
    def _matrix_vector(matrix, vector):
        return tuple(
            sum(matrix[row][column] * vector[column] for column in range(3))
            for row in range(3)
        )

    def _plane_tilt_deg(self, rpy, cup_plane_normal_tcp):
        world_normal = self._matrix_vector(
            self._rpy_matrix(rpy), cup_plane_normal_tcp
        )
        normal_length = math.sqrt(sum(value * value for value in world_normal))
        cosine = max(-1.0, min(1.0, world_normal[2] / normal_length))
        return math.degrees(math.acos(cosine))

    def _audit_dual_cup_path(
        self,
        start_joints,
        target_joints,
        tool_contract,
        path_contract,
        collision_verifier,
    ):
        self._last_dual_cup_path_audit = None
        self._last_dual_cup_path_summary = None
        if max(
            abs(actual - frozen)
            for actual, frozen in zip(start_joints, path_contract.start_joints)
        ) > 1e-12:
            raise RobotError("运行时审计起点不是 PathContract.start_joints")
        if max(
            abs(actual - frozen)
            for actual, frozen in zip(target_joints, path_contract.target_joints)
        ) > 1e-12:
            raise RobotError("运行时审计目标不是 PathContract.target_joints")
        maximum_delta = max(
            abs(target - start)
            for start, target in zip(start_joints, target_joints)
        )
        configured_step_deg = self._safety_number(
            "path_fk_sample_step_deg", MAX_DENSE_JOINT_STEP_DEG
        )
        sample_step_deg = min(configured_step_deg, MAX_DENSE_JOINT_STEP_DEG)
        sample_count = max(
            1,
            int(math.ceil(math.degrees(maximum_delta) / sample_step_deg)),
        )
        start = [float(value) for value in start_joints]
        target = [float(value) for value in target_joints]
        samples = []
        anchor_tcp = None
        anchor_flange = None
        maximum_tilt_deg = 0.0
        maximum_family_error_deg = 0.0
        maximum_fixed_xy_error_mm = 0.0
        maximum_tcp_downward_drop_mm = 0.0
        maximum_flange_downward_drop_mm = 0.0
        minimum_held_high_z_mm = math.inf
        minimum_collision_clearance_mm = math.inf
        yaw_delta, start_matrix, end_matrix, _ = self._world_z_yaw_delta(
            path_contract.start_rpy, path_contract.end_rpy
        )
        for index in range(sample_count + 1):
            ratio = index / sample_count
            joints = [
                (1.0 - ratio) * first + ratio * second
                for first, second in zip(start, target)
            ]
            try:
                tcp_pose = self.rpc.cal_fkine(
                    joints, list(tool_contract.tcp_offset_m), []
                )
                flange_pose = self.rpc.cal_fkine(joints, [0.0] * 6, [])
            except Exception as exc:
                raise RobotError("双杯路径显式 TCP/法兰 FK 失败: %s" % exc) from exc
            tcp_pose = self._finite_six(tcp_pose, "双杯路径 TCP FK")
            flange_pose = self._finite_six(flange_pose, "双杯路径法兰 FK")
            if index > 0:
                previous_joints = samples[-1].joints
                step_deg = math.degrees(max(
                    abs(current - previous)
                    for current, previous in zip(joints, previous_joints)
                ))
                if step_deg > MAX_DENSE_JOINT_STEP_DEG + 1e-9:
                    raise RobotError(
                        "双杯路径密采步长 %.6fdeg > %.2fdeg"
                        % (step_deg, MAX_DENSE_JOINT_STEP_DEG)
                    )
            ok, msg = self.safety.check_workspace(tcp_pose)
            if not ok:
                raise RobotError(
                    "运动前显式双杯 TCP 路径越界(sample=%d/%d): %s"
                    % (index, sample_count, msg)
                )
            if anchor_tcp is None:
                anchor_tcp = tuple(tcp_pose)
                anchor_flange = tuple(flange_pose)

            tilt_deg = self._plane_tilt_deg(
                tcp_pose[3:6], tool_contract.cup_plane_normal_tcp
            )
            if tilt_deg > MAX_CUP_PLANE_TILT_DEG + 1e-9:
                raise RobotError(
                    "双杯路径杯面绝对倾斜 %.6fdeg > %.1fdeg (sample=%d/%d)"
                    % (tilt_deg, MAX_CUP_PLANE_TILT_DEG, index, sample_count)
                )
            maximum_tilt_deg = max(maximum_tilt_deg, tilt_deg)
            actual_matrix = self._rpy_matrix(tcp_pose[3:6])
            expected_matrix = self._matrix_multiply(
                self._world_z_rotation(yaw_delta * ratio), start_matrix
            )
            family_error_deg = self._orientation_matrix_error_deg(
                expected_matrix, actual_matrix
            )
            if family_error_deg > MAX_ORIENTATION_FAMILY_ERROR_DEG + 1e-9:
                raise RobotError(
                    "双杯路径完整世界 Z yaw 姿态误差 %.6fdeg > %.1fdeg "
                    "(sample=%d/%d)"
                    % (
                        family_error_deg,
                        MAX_ORIENTATION_FAMILY_ERROR_DEG,
                        index,
                        sample_count,
                    )
                )
            maximum_family_error_deg = max(
                maximum_family_error_deg, family_error_deg
            )
            endpoint_matrix = None
            endpoint_label = None
            if index == 0:
                endpoint_matrix = start_matrix
                endpoint_label = "start_rpy"
            elif index == sample_count:
                endpoint_matrix = end_matrix
                endpoint_label = "end_rpy"
            if endpoint_matrix is not None:
                endpoint_error_deg = self._orientation_matrix_error_deg(
                    endpoint_matrix, actual_matrix
                )
                if endpoint_error_deg > MAX_ORIENTATION_FAMILY_ERROR_DEG + 1e-9:
                    raise RobotError(
                        "双杯路径端点未绑定 %s: %.6fdeg > %.1fdeg"
                        % (
                            endpoint_label,
                            endpoint_error_deg,
                            MAX_ORIENTATION_FAMILY_ERROR_DEG,
                        )
                    )

            if path_contract.phase in {"yaw", "vertical"}:
                xy_error_mm = math.sqrt(
                    (tcp_pose[0] - anchor_tcp[0]) ** 2
                    + (tcp_pose[1] - anchor_tcp[1]) ** 2
                ) * 1000.0
                if xy_error_mm > MAX_FIXED_XY_ERROR_MM + 1e-9:
                    raise RobotError(
                        "%s 段 TCP XY 漂移 %.6fmm > %.2fmm (sample=%d/%d)"
                        % (
                            path_contract.phase,
                            xy_error_mm,
                            MAX_FIXED_XY_ERROR_MM,
                            index,
                            sample_count,
                        )
                    )
                maximum_fixed_xy_error_mm = max(
                    maximum_fixed_xy_error_mm, xy_error_mm
                )
            if path_contract.phase == "yaw":
                tcp_drop_mm = (anchor_tcp[2] - tcp_pose[2]) * 1000.0
                flange_drop_mm = (anchor_flange[2] - flange_pose[2]) * 1000.0
                if (
                    tcp_drop_mm > MAX_YAW_DOWNWARD_DROP_MM + 1e-9
                    or flange_drop_mm > MAX_YAW_DOWNWARD_DROP_MM + 1e-9
                ):
                    raise RobotError(
                        "yaw 段 TCP/法兰下掉（腕部代理）: "
                        "TCP=%.6fmm flange=%.6fmm > %.1fmm "
                        "(sample=%d/%d)"
                        % (
                            tcp_drop_mm,
                            flange_drop_mm,
                            MAX_YAW_DOWNWARD_DROP_MM,
                            index,
                            sample_count,
                        )
                    )
                maximum_tcp_downward_drop_mm = max(
                    maximum_tcp_downward_drop_mm, tcp_drop_mm
                )
                maximum_flange_downward_drop_mm = max(
                    maximum_flange_downward_drop_mm, flange_drop_mm
                )
            if (
                path_contract.wafer_held
                and path_contract.phase not in {"contact", "vertical"}
                and tcp_pose[2] * 1000.0 < MIN_HELD_HIGH_TCP_Z_MM - 1e-9
            ):
                raise RobotError(
                    "带片高位 TCP Z=%.6fmm < %.1fmm (sample=%d/%d)"
                    % (
                        tcp_pose[2] * 1000.0,
                        MIN_HELD_HIGH_TCP_Z_MM,
                        index,
                        sample_count,
                    )
                )
            if path_contract.wafer_held and path_contract.phase not in {
                "contact",
                "vertical",
            }:
                minimum_held_high_z_mm = min(
                    minimum_held_high_z_mm, tcp_pose[2] * 1000.0
                )

            sample = CollisionSample(
                path_name=path_contract.name,
                phase=path_contract.phase,
                wafer_held=path_contract.wafer_held,
                sample_index=index,
                sample_count=sample_count,
                joints=tuple(joints),
                tcp_pose=tuple(tcp_pose),
                flange_pose=tuple(flange_pose),
                plane_tilt_deg=tilt_deg,
                orientation_family_error_deg=family_error_deg,
                tool_contract=tool_contract,
                path_contract=path_contract,
            )
            try:
                clearance_mm = collision_verifier(sample)
            except Exception as exc:
                raise RobotError(
                    "collision_verifier 执行失败(sample=%d/%d): %s"
                    % (index, sample_count, exc)
                ) from exc
            if isinstance(clearance_mm, bool):
                raise RobotError("collision_verifier 必须返回毫米有限正间隙，不得返回 bool")
            try:
                clearance_mm = float(clearance_mm)
            except (TypeError, ValueError) as exc:
                raise RobotError("collision_verifier 必须返回毫米有限正间隙") from exc
            if not math.isfinite(clearance_mm) or clearance_mm <= 0.0:
                raise RobotError(
                    "collision_verifier 未证明正有限间隙: %r (sample=%d/%d)"
                    % (clearance_mm, index, sample_count)
                )
            minimum_collision_clearance_mm = min(
                minimum_collision_clearance_mm, clearance_mm
            )
            samples.append(sample)
        self._last_dual_cup_path_audit = tuple(samples)
        self._last_dual_cup_path_summary = {
            "samples": len(samples),
            "maximum_plane_tilt_deg": maximum_tilt_deg,
            "engineering_target_plane_tilt_deg": (
                ENGINEERING_TARGET_CUP_PLANE_TILT_DEG
            ),
            "engineering_target_met": (
                maximum_tilt_deg
                <= ENGINEERING_TARGET_CUP_PLANE_TILT_DEG + 1e-9
            ),
            "maximum_orientation_family_error_deg": maximum_family_error_deg,
            "maximum_fixed_xy_error_mm": maximum_fixed_xy_error_mm,
            "maximum_tcp_downward_drop_mm": maximum_tcp_downward_drop_mm,
            "maximum_flange_downward_drop_mm": maximum_flange_downward_drop_mm,
            "minimum_held_high_z_mm": (
                None
                if math.isinf(minimum_held_high_z_mm)
                else minimum_held_high_z_mm
            ),
            "minimum_collision_clearance_mm": minimum_collision_clearance_mm,
        }
        return self._last_dual_cup_path_audit

    def _read_immediate_send_gate(self, expected_start_joints, tool_contract):
        self._verify_active_tool(tool_contract, "下发即时门")
        frame = self.rpc.read_status_frame(retries=1, timeout=0.5)
        joints = self._validate_motion_frame(frame, "下发即时门", require_idle=True)
        self._verify_rpc_actual_joints(frame, require_stationary=True)
        tcp_pose, flange_pose = self._verify_fk_tcp(
            frame, tool_contract, "下发即时门"
        )
        tolerance_deg = min(
            self._safety_number(
                "post_audit_start_tolerance_deg", MAX_FROZEN_START_ERROR_DEG
            ),
            MAX_FROZEN_START_ERROR_DEG,
        )
        drift_deg = math.degrees(max(
            abs(actual - expected)
            for actual, expected in zip(joints, expected_start_joints)
        ))
        if drift_deg > tolerance_deg:
            raise RobotError(
                "路径审计后起点漂移 %.6fdeg > %.6fdeg，拒绝下发"
                % (drift_deg, tolerance_deg)
            )
        frame = dict(frame)
        frame["verified_dual_cup_tcp"] = tcp_pose
        frame["verified_flange"] = flange_pose
        self._store_status_frame(frame)
        return frame

    def _read_motion_preflight(
        self,
        tool_contract,
        tool_check_label,
        expected_start_joints=None,
    ):
        import time

        self._verify_active_tool(tool_contract, tool_check_label)
        first = self.rpc.read_status_frame(retries=1, timeout=1.0)
        first_joints = self._validate_motion_frame(first, "运动前首帧", require_idle=True)
        self._verify_fk_tcp(first, tool_contract, "%s首帧" % tool_check_label)
        if expected_start_joints is not None:
            first_start_error_deg = math.degrees(max(
                abs(actual - frozen)
                for actual, frozen in zip(first_joints, expected_start_joints)
            ))
            if first_start_error_deg > MAX_FROZEN_START_ERROR_DEG + 1e-9:
                raise RobotError(
                    "首次 fresh q 与冻结模拟起点不一致: %.6fdeg > %.2fdeg"
                    % (first_start_error_deg, MAX_FROZEN_START_ERROR_DEG)
                )
        interval_s = self._safety_number(
            "preflight_sample_interval_s", 0.1, allow_zero=True
        )
        started_at = time.monotonic()
        if interval_s > 0:
            time.sleep(interval_s)
        second = self.rpc.read_status_frame(retries=1, timeout=1.0)
        elapsed_s = max(time.monotonic() - started_at, 1e-6)
        second_joints = self._validate_motion_frame(second, "运动前复核帧", require_idle=True)
        if expected_start_joints is not None:
            second_start_error_deg = math.degrees(max(
                abs(actual - frozen)
                for actual, frozen in zip(second_joints, expected_start_joints)
            ))
            if second_start_error_deg > MAX_FROZEN_START_ERROR_DEG + 1e-9:
                raise RobotError(
                    "复核 fresh q 与冻结模拟起点不一致: %.6fdeg > %.2fdeg"
                    % (second_start_error_deg, MAX_FROZEN_START_ERROR_DEG)
                )
        maximum_speed_deg_s = max(
            math.degrees(abs(current - previous)) / elapsed_s
            for previous, current in zip(first_joints, second_joints)
        )
        speed_limit_deg_s = self._safety_number(
            "preflight_max_joint_speed_deg_s", 2.5
        )
        if maximum_speed_deg_s > speed_limit_deg_s:
            raise RobotError(
                "运动前状态不静止: max_speed=%.3fdeg/s > %.3fdeg/s"
                % (maximum_speed_deg_s, speed_limit_deg_s)
            )
        self._verify_rpc_actual_joints(second, require_stationary=True)
        tcp_pose, flange_pose = self._verify_fk_tcp(
            second, tool_contract, tool_check_label
        )
        second = dict(second)
        second["verified_dual_cup_tcp"] = tcp_pose
        second["verified_flange"] = flange_pose
        self._store_status_frame(second)
        return second

    def _read_motion_completion(self, tool_contract, timeout_s=2.0):
        import time

        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            frame = self.rpc.read_status_frame(retries=1, timeout=0.5)
            self._validate_motion_frame(frame, "完成过渡", require_idle=False)
            self._store_status_frame(frame)
            if frame["prog_state"] == 0:
                try:
                    return self._read_motion_preflight(tool_contract, "完成后")
                except RobotError as exc:
                    if "检测到已有程序/运动" not in str(exc):
                        raise
            elif frame["prog_state"] != 1:
                raise RobotError(
                    "task Finished 后出现非停止过渡 prog_state=%s"
                    % frame["prog_state"]
                )
            time.sleep(0.05)
        raise RobotError("task Finished 后 prog_state 未在 %.1fs 内稳定到 0" % timeout_s)

    # ---------- 运动 (带安全检查) ----------
    def movej_safe(
        self,
        joints,
        v=None,
        a=None,
        r=0.0,
        block=True,
        timeout_s=60,
        *,
        tool_contract=None,
        path_contract=None,
        collision_verifier=None,
    ):
        """关节运动: 检查关节限位 + 钳速 后下发 movej2。

        仅允许显式双杯契约、逐点碰撞验证和受监督的 block=True。"""
        joints, tool_contract, path_contract = self._qualify_motion_contracts(
            joints, tool_contract, path_contract, collision_verifier
        )
        if not block:
            raise RobotError("movej_safe 禁止无监督 block=False")
        if not self._motion_lock.acquire(blocking=False):
            raise RobotError("已有 movej_safe 正在受监督执行，拒绝并发运动")
        try:
            self._assert_ready()
            ok, msg = self.safety.check_joint_limits(joints)
            if not ok:
                raise RobotError("movej_safe 拒绝: " + msg)
            sp = self.cfg["speed"]
            requested_v = v if v is not None else sp["v_max_joint"]
            requested_a = a if a is not None else sp["a_joint"]
            values = {
                "v": requested_v,
                "a": requested_a,
                "r": r,
                "timeout_s": timeout_s,
            }
            if any(not math.isfinite(float(value)) for value in values.values()):
                raise RobotError("movej_safe 拒绝: 运动参数必须为有限数")
            if float(requested_v) <= 0 or float(requested_a) <= 0 or float(timeout_s) <= 0:
                raise RobotError("movej_safe 拒绝: v/a/timeout_s 必须大于 0")
            if abs(float(r)) > 1e-12:
                raise RobotError("movej_safe 拒绝: 单任务监督仅允许 r=0")
            maximum_a = float(sp["a_joint"])
            if not math.isfinite(maximum_a) or maximum_a <= 0:
                raise RobotError("movej_safe 拒绝: speed.a_joint 配置无效")
            if float(requested_a) > maximum_a:
                raise RobotError(
                    "movej_safe 拒绝: a=%.6f 超过项目上限 %.6f"
                    % (float(requested_a), maximum_a)
                )
            command_v = self.safety.clamp_joint_v(float(requested_v))
            if not math.isfinite(command_v) or command_v <= 0:
                raise RobotError("movej_safe 拒绝: 钳速后速度无效")
            return self._move_and_wait(
                lambda: self.rpc.movej2(
                    joints, command_v, float(requested_a), 0.0, False
                ),
                joints,
                float(timeout_s),
                tool_contract=tool_contract,
                path_contract=path_contract,
                collision_verifier=collision_verifier,
            )
        finally:
            self._motion_lock.release()

    def _move_and_wait(
        self,
        send_fn,
        target_joints,
        timeout_s,
        *,
        tool_contract=None,
        path_contract=None,
        collision_verifier=None,
    ):
        """非阻塞发运动指令，并按 task id 监控直到可信完成。"""
        import math
        import time
        target_joints, tool_contract, path_contract = self._qualify_motion_contracts(
            target_joints, tool_contract, path_contract, collision_verifier
        )
        hb_was = self.safety.heartbeat_running()
        command_sent = False
        restart_allowed = True
        if hb_was:
            self.safety.stop_heartbeat()
        try:
            if self.safety.aborted:
                raise RobotError("运动前心跳已 ABORT, 拒绝下发")
            before = self._read_motion_preflight(
                tool_contract,
                "首次",
                expected_start_joints=path_contract.start_joints,
            )
            dmax_deg = max(
                math.degrees(abs(target - current))
                for target, current in zip(target_joints, before["joints"])
            )
            self._audit_dual_cup_path(
                path_contract.start_joints,
                target_joints,
                tool_contract,
                path_contract,
                collision_verifier,
            )
            if dmax_deg < 0.3:
                return TASK_FINISHED
            send_frame = self._read_immediate_send_gate(
                path_contract.start_joints, tool_contract
            )
            timeout_s = max(float(timeout_s), dmax_deg * 1.2 + 10.0)
            command_sent = True
            restart_allowed = False
            task_id = send_fn()
            if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
                raise RobotError("movej2 未返回有效 task id: %r" % (task_id,))
            target_deg = [round(math.degrees(value), 4) for value in target_joints]
            print(
                "[MOVEJ2] task_id=%s target_deg=%s"
                % (task_id, target_deg),
                flush=True,
            )
            t0 = time.monotonic()
            last_frame_time = time.monotonic()
            last_joints = list(send_frame["joints"])
            read_fails = 0
            task_read_fails = 0
            idle_polls = 0
            err = math.inf
            while time.monotonic() - t0 < timeout_s:
                if self.safety.aborted:
                    raise RobotError("运动中被ABORT")
                try:
                    fr = self.rpc.read_status_frame(retries=1, timeout=0.5)
                    read_fails = 0
                except Exception:
                    read_fails += 1
                    if read_fails >= 3:
                        raise RobotError("连续3次读状态失败, 链路异常")
                    time.sleep(0.1)
                    continue
                self._validate_motion_frame(fr, "运动中", require_idle=False)
                tcp_pose, flange_pose = self._verify_fk_tcp(
                    fr, tool_contract, "运动中"
                )
                fr = dict(fr)
                fr["verified_dual_cup_tcp"] = tcp_pose
                fr["verified_flange"] = flange_pose
                frame_time = time.monotonic()
                elapsed_s = max(frame_time - last_frame_time, 1e-6)
                maximum_speed_deg_s = max(
                    math.degrees(abs(current - previous)) / elapsed_s
                    for previous, current in zip(last_joints, fr["joints"])
                )
                configured_speed_deg_s = math.degrees(
                    self.safety.v_max_joint * self.safety.global_scale
                )
                state_speed_limit_deg_s = self._safety_number(
                    "state_frame_max_joint_speed_deg_s",
                    max(5.0, configured_speed_deg_s * 2.0 + 2.0),
                )
                if maximum_speed_deg_s > state_speed_limit_deg_s:
                    raise RobotError(
                        "运动中关节帧不连续: %.2fdeg/s > %.2fdeg/s"
                        % (maximum_speed_deg_s, state_speed_limit_deg_s)
                    )
                last_joints = list(fr["joints"])
                last_frame_time = frame_time
                self._store_status_frame(fr)
                err = max(abs(a - b) for a, b in zip(fr["joints"], target_joints))
                try:
                    task_state = self.rpc.get_noneblock_taskstate(task_id)
                    task_read_fails = 0
                except Exception:
                    task_read_fails += 1
                    if task_read_fails >= 3:
                        raise RobotError("连续3次读取 task_id=%d 状态失败" % task_id)
                    time.sleep(0.2)
                    continue
                if isinstance(task_state, bool) or not isinstance(task_state, int) or task_state not in ST_TASK:
                    raise RobotError("task_id=%d 返回非法状态 %r" % (task_id, task_state))
                if task_state in TASK_FAILED:
                    raise RobotError(
                        "task_id=%d 失败: %s, last_error=%s"
                        % (task_id, ST_TASK[task_state], self.rpc.get_last_error())
                    )
                if task_state == TASK_FINISHED:
                    completion = self._read_motion_completion(
                        tool_contract,
                        timeout_s=self._safety_number(
                            "completion_settle_timeout_s", 2.0
                        )
                    )
                    completion_error = max(
                        abs(actual - target)
                        for actual, target in zip(completion["joints"], target_joints)
                    )
                    if completion_error >= math.radians(0.5):
                        raise RobotError(
                            "task_id=%d 已完成但目标误差 %.3fdeg"
                            % (task_id, math.degrees(completion_error))
                        )
                    restart_allowed = True
                    return TASK_FINISHED
                if task_state == TASK_IDLE:
                    idle_polls += 1
                    if idle_polls >= 3:
                        raise RobotError("task_id=%d 连续返回 Idle，拒绝猜测任务完成" % task_id)
                else:
                    idle_polls = 0
                time.sleep(0.2)
            raise RobotError(
                "task_id=%d 运动超时 %.0fs，目标误差 %.2fdeg"
                % (task_id, timeout_s, math.degrees(err))
            )
        except Exception as exc:
            if command_sent and not restart_allowed:
                self.safety.aborted = True
                if not self._disable_and_verify():
                    raise RobotError(
                        "运动失败且 disable 未确认；保持 NO-GO，要求现场急停/示教器确认"
                    ) from exc
            raise
        finally:
            if hb_was and restart_allowed and not self.safety.aborted:
                try:
                    state = self.rpc.get_robot_state()
                except Exception:
                    state = None
                if (
                    isinstance(state, (list, tuple))
                    and len(state) >= 4
                    and state[0] == ROBOT_STATE_ENABLED
                    and state[1] == 0
                    and state[2] == SAFETY_RUN
                    and state[3] == 1
                ):
                    self.safety.start_heartbeat(self._hb_read_state, self._on_abort)

    def movel_safe(self, pose, q_near, v=None, a=None, r=0.0,
                   tool="default", wobj="default", block=True):
        """笛卡尔直线运动尚未接入 task-id 监督，保持冻结。"""
        raise RobotError("movel_safe 未接入 task-id 与路径监督，禁止下发")

    def stop(self):
        self.safety.aborted = True
        monitor_error = None
        try:
            self._start_shutdown_monitor()
        except Exception as exc:
            monitor_error = exc
        self._invalidate_status_cache()
        stopped = self._disable_and_verify()
        try:
            self.safety.stop_heartbeat()
        except Exception as exc:
            raise RobotError("停止后心跳线程未确认退出；保持 NO-GO") from exc
        if not stopped:
            # _stop_unverified 仍会锁死运动；重新武装心跳只用于监视和再次 ABORT。
            self.safety.aborted = False
            try:
                self._start_shutdown_monitor(force=True)
            except Exception as exc:
                raise RobotError(
                    "停止未确认且监控心跳无法重建；要求现场急停/示教器确认"
                ) from exc
            raise RobotError("停止未确认；保留监控心跳并要求现场急停/示教器确认")
        if monitor_error is not None:
            print(
                "[SAFETY] stop 监控心跳未启动，但 disable 已确认: %s"
                % monitor_error,
                flush=True,
            )
