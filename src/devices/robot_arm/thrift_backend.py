"""真机机械臂后端：包装 GCR618 的 Robot(robot_control.py) → RobotArmBackendBase。

唯一直接驱动 Thrift / 真实机械臂的地方。安全护栏（限位/钳速/心跳/工作空间）
由内部的 Robot + Safety 提供；本类把其方法映射到统一后端接口，并把 2001 状态帧
转换为 RobotArmStatus 快照。

需要 thrift 库（pip install thrift）。无 thrift 时构造会抛错（提示改用仿真）。
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional

import yaml

from .backend import RobotArmBackendBase, RobotArmStatus, ROBOT_STATE_ENABLED, SAFETY_RUN
from .robot_control import Robot, RobotError
from .waypoints import WaypointStore

class ThriftRobotArmBackend(RobotArmBackendBase):
    """真机后端。所有 RPC 经内部 Robot；状态读取走心跳缓存帧避免 2001 并发。"""

    def __init__(self, cfg: dict, waypoints_path: str):
        self.cfg = cfg
        self._robot = Robot(cfg)
        self._store = WaypointStore(waypoints_path)
        self._lock = threading.RLock()
        self._connected = False
        self._last_error = ""
        # 速度护栏：自测期用极低 scale（安全边界），可被 UI 覆盖
        self._global_scale = cfg.get("speed", {}).get("global_scale", 0.2)
        # 夹爪工具端 485 是半双工共享总线：一次只允许一个 Modbus 事务，否则
        # 写(init/grip)会与并发的状态只读交错，读回「跳变垃圾值」(debug_history 605-623)。
        # 用一把锁串行化所有夹爪 485 事务；状态读加短 TTL 缓存，避免多个 /api/status
        # 轮询者各自轰总线。
        self._gripper_lock = threading.Lock()
        self._gripper_status_cache = None
        self._gripper_status_cache_t = 0.0

    # ── 连接 ────────────────────────────────────────────────────
    def connect(self) -> None:
        with self._lock:
            self._robot.connect()       # 开 transport + 只读一帧；使能通过后才启动心跳
            self._connected = True
            self._last_error = ""

    def disconnect(self) -> None:
        with self._lock:
            if self._connected:
                try:
                    self._robot.shutdown(do_disable=True)
                except Exception as e:
                    self._last_error = "断开异常: %s" % e
                    return
                self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    # ── 状态 ────────────────────────────────────────────────────
    def get_status(self) -> RobotArmStatus:
        if not self._connected:
            return RobotArmStatus(connected=False, error=self._last_error)
        try:
            fr = self._robot.last_frame()   # 心跳缓存帧，避免 2001 并发
            rs = fr["robot_state"]
            return RobotArmStatus(
                connected=True,
                joints=list(fr["joints"]),
                tcp=list(fr["tcp"]),
                op_mode=fr["op_mode"],
                robot_state=rs,
                prog_state=fr["prog_state"],
                safety=fr["safety"],
                collision=fr["collision"],
                collision_axis=fr["collision_axis"],
                alarm=fr["alarm"],
                powered=rs >= 5,
                enabled=rs == ROBOT_STATE_ENABLED,
                error="",
            )
        except Exception as e:
            return RobotArmStatus(connected=True, error="读状态失败: %s" % e)

    # ── 上电 / 使能 ─────────────────────────────────────────────
    def power_on(self) -> None:
        # 管理员授权：软件上电（真调 DUCO RPC power_on）。
        with self._lock:
            self._robot.rpc.power_on(block=True)
            self._last_error = ""

    def enable(self) -> None:
        # 管理员授权：软件使能（先确保上电再使能，真调 DUCO RPC）。
        with self._lock:
            self._robot.rpc.power_on(block=True)
            self._robot.rpc.enable(block=True)
            self._last_error = ""

    def disable(self) -> None:
        with self._lock:
            self._robot.stop()
            self._last_error = ""

    def estop(self) -> None:
        """急停：立即 disable（旁路运动护栏）。"""
        with self._lock:
            try:
                self._robot.stop()
            except Exception as e:
                self._last_error = "急停异常: %s" % e

    # ── 拖动示教（牵引/零力）────────────────────────────────────
    def enter_teach_mode(self, load_mass: float = 0.0,
                         cx: float = 0.0, cy: float = 0.0, cz: float = 0.0) -> None:
        """进入拖动示教。先 set_load_data（必须，否则零力瞬间机械臂会甩动），再 teach_mode。
        真机最高风险：调用方需确认已使能、无报警，且人在旁手扶。"""
        # 进入前置安全检查：必须已使能、安全态 RUN、无报警
        st = self.get_status()
        if not st.enabled:
            raise RuntimeError("未使能，无法进入拖动示教（先上电+使能）")
        if st.alarm != 0:
            raise RuntimeError("存在报警(0x%X)，请先在示教器复位再进拖动示教" % st.alarm)
        # 必设末端负载（质量 kg + 质心 m，相对工具系）
        self._robot.rpc.set_load_data(float(load_mass), float(cx), float(cy), float(cz))
        self._robot.rpc.teach_mode(block=True)
        self._teaching = True

    def exit_teach_mode(self) -> None:
        """退出拖动示教，回到正常受控（仍处于使能态）。"""
        self._robot.rpc.end_teach_mode(block=True)
        self._teaching = False

    def is_teaching(self) -> bool:
        return bool(getattr(self, "_teaching", False))

    # ── 运动 ────────────────────────────────────────────────────
    def move_joints(self, joints: List[float], speed_scale: Optional[float] = None,
                    block: bool = True) -> None:
        # 临时覆盖速度倍率（自测期极低速）
        if speed_scale is not None:
            old = self._robot.safety.global_scale
            self._robot.safety.global_scale = speed_scale
            try:
                self._robot.movej_safe(joints, block=block)
            finally:
                self._robot.safety.global_scale = old
        else:
            self._robot.movej_safe(joints, block=block)

    def goto_pose(self, x, y, z, yaw, tool_offset=None, speed_scale=None,
                  block=True) -> None:
        """旧单杯在线 IK 入口永久冻结。"""
        raise RobotError(
            "goto_pose 旧单杯在线 IK 入口永久冻结；仅允许已审计冻结计划"
        )

    def jog_joint(self, index: int, delta_rad: float, speed_scale=None) -> None:
        """单关节点动：当前关节第 index 轴 ±delta_rad → 轻量点动路径(_jog_movej)。
        覆盖基类实现（基类走 move_joints→movej_safe，需双杯审计契约，交互点动无契约）。"""
        st = self.get_status()
        if not st.connected:
            raise RuntimeError("未连接真机，无法关节点动")
        cur = [float(x) for x in st.joints]
        if not (0 <= index < len(cur)):
            raise IndexError("关节索引越界: %d" % index)
        cur[index] = cur[index] + float(delta_rad)
        self._jog_movej(cur, speed_scale)

    def _jog_movej(self, target_joints, speed_scale=None):
        """交互式点动的轻量受监督关节运动：基础就绪 + 关节限位 + 钳速 → 直接 movej2(阻塞到位)。
        **不走** movej_safe 的双杯审计契约系统——那是给自主揭膜冻结计划的（需 tool/path 契约 +
        逐点碰撞验证），交互点动无契约。安全分层：DUCO 控制器自身实时碰撞/限位（硬件级，始终生效）
        + 本地 _assert_ready(state6/idle/RUN/Auto)/关节限位/上层逆解跳变&正解回验护栏
        + 小步(2mm/1°)低速(≤20%) + 现场急停。仅用于操作员在旁的交互点动。"""
        import math as _m
        r = self._robot
        r._assert_ready()                       # state=6 / prog idle / safety=RUN / op_mode=Auto
        tj = [float(x) for x in target_joints]
        ok, msg = r.safety.check_joint_limits(tj)
        if not ok:
            raise RobotError("点动拒绝(关节限位): " + msg)
        sp = r.cfg["speed"]
        scale = float(speed_scale) if speed_scale is not None else 0.05
        scale = max(0.01, min(0.2, scale))      # 点动速度限幅 1%~20%
        v = r.safety.clamp_joint_v(float(sp["v_max_joint"]) * scale)
        a = float(sp["a_joint"])
        if not (_m.isfinite(v) and v > 0 and _m.isfinite(a) and a > 0):
            raise RobotError("点动拒绝: 钳速后速度/加速度无效")
        # 关键：用 block=False（与 movej_safe 一致）。block=True 会让 DUCO 回两帧(ack+完成)，
        # 而 _call 只读一帧 → 残留一帧使 7003 RPC 序号永久错位。block=False 单帧回 task_id，
        # 再用 get_noneblock_taskstate 轮询到位（同 7003、串行、各读各的回复，不错位）。
        import time as _t
        tid = r.rpc.movej2(tj, v, a, 0.0, False)
        if not isinstance(tid, int):
            _t.sleep(1.5)                        # 拿不到 task_id：短暂等待兜底（上层核对实际位姿）
            return
        deadline = _t.monotonic() + 12.0
        while _t.monotonic() < deadline:
            _t.sleep(0.1)
            stt = r.rpc.get_noneblock_taskstate(tid)
            if stt == 4:                         # Finished 完成
                return
            if stt in (3, 5, 6, 7, 8):           # Stopped/Interrupt/Error/Illegal/ParamMismatch
                raise RobotError("点动任务异常终止 taskstate=%s" % stt)
        raise RobotError("点动到位超时(taskstate 未完成)")

    def jog_cart(self, axis: int, delta: float, speed_scale=None, dry_run: bool = False):
        """笛卡尔点动（机器人自带逆解 + 受监督关节运动）。
        axis 0-2=X/Y/Z(m)、3-5=Rx/Ry/Rz(rad)；delta 为该轴增量。
        流程：读当前 TCP+关节 → 目标位姿 = TCP + Δaxis → 机器人 cal_ikine 逆解(靠近当前关节)
              → 逆解跳变护栏(>15° 判奇异/翻转/跳解，拒绝) → movej_safe(受监督, 低速)。
        dry_run=True：只算逆解+护栏、绝不下发运动，返回目标关节信息。"""
        import math as _m
        axis = int(axis)
        if not (0 <= axis <= 5):
            raise ValueError("axis 必须 0-5 (X/Y/Z/Rx/Ry/Rz)")
        st = self.get_status()
        if not st.connected:
            raise RuntimeError("未连接真机，无法笛卡尔点动")
        # 真运动才要求就绪；dry_run 只读+计算，允许未使能时验证逆解
        if not dry_run and not st.is_ready:
            raise RuntimeError(
                "机械臂未就绪(enabled=%s safety=%d alarm=0x%X)，拒绝笛卡尔点动"
                % (st.enabled, st.safety, st.alarm))
        cur_joints = [float(x) for x in st.joints]
        cur_tcp = [float(x) for x in st.tcp]
        if len(cur_joints) != 6 or len(cur_tcp) != 6:
            raise RuntimeError("当前关节/TCP 读数异常")
        target = list(cur_tcp)
        target[axis] = target[axis] + float(delta)
        # 机器人自带逆解，q_near=当前关节，取最靠近的解支（避免跳解）
        tj = self._robot.rpc.cal_ikine(target, q_near=cur_joints)
        if not isinstance(tj, (list, tuple)) or len(tj) != 6:
            raise RuntimeError("逆解失败/返回异常: %r（可能无解/奇异）" % (tj,))
        tj = [float(x) for x in tj]
        max_dj = max(abs(a - b) for a, b in zip(tj, cur_joints))
        guard = _m.radians(15.0)
        if max_dj > guard:
            raise RuntimeError(
                "逆解关节跳变过大(%.1f°>15°)，疑似奇异/翻转/跳解，已拒绝下发"
                % _m.degrees(max_dj))
        # 正解回验：DUCO cal_ikine 对不可达点会原样返回 q_near（关节跳变=0，护栏漏判）。
        # 对解出的关节做正解，回算 TCP 必须≈目标位姿；差得远=不可达/无解→拒绝。
        fk = self._robot.rpc.cal_fkine(tj)
        if not isinstance(fk, (list, tuple)) or len(fk) != 6:
            raise RuntimeError("正解校验返回异常: %r" % (fk,))
        fk = [float(x) for x in fk]
        pos_err = max(abs(fk[i] - target[i]) for i in range(3))        # m
        rot_err = max(abs(fk[i] - target[i]) for i in range(3, 6))     # rad
        if pos_err > 5e-4 or rot_err > _m.radians(0.3):
            raise RuntimeError(
                "逆解未达目标(位置差%.1fmm/姿态差%.2f°)，疑似不可达/奇异，已拒绝"
                % (pos_err * 1000.0, _m.degrees(rot_err)))
        info = {
            "target_joints": tj,
            "max_joint_delta_deg": _m.degrees(max_dj),
            "fk_pos_err_mm": pos_err * 1000.0,
            "cur_tcp": cur_tcp,
            "target_tcp": target,
        }
        if dry_run:
            return info
        # 交互式点动走轻量路径（movej2 + 基础就绪/限位/钳速），不套 movej_safe 的双杯审计契约
        # （那是自主揭膜冻结计划专用，交互点动无契约）。分层安全见 _jog_movej 注释。
        self._jog_movej(tj, speed_scale)
        return info

    # ── 末端数字输出（吸盘）─────────────────────────────────────
    def set_digital_output(self, port: int, value: bool, block: bool = True) -> None:
        # 安全：吸盘动作前要求已使能 + 安全态 RUN + 无报警，
        # 避免下电/报警态误触发继电器（泵在错误时刻吸/放）。
        st = self.get_status()
        if not st.is_ready:
            raise RuntimeError(
                "机械臂未就绪(enabled=%s safety=%d alarm=0x%X)，拒绝吸盘动作"
                % (st.enabled, st.safety, st.alarm))
        self._robot.rpc.set_tool_digital_out(int(port), bool(value), block=block)

    # ── 夹爪（大寰 PGEA-50-26-O，工具端 485 Modbus 透传）─────────
    def _gripper(self):
        """惰性构造 DHGripper：注入工具端 485 收发 → duco_rpc。⚠️ 待真机验证（方法可达/时序）。

        transport：write485=tool_write_raw_data_485（写裸帧），
                   read485=tool_read_raw_data_485_h（匹配帧头后收）。
        485 半双工「写后等响应再读」的时序由 DHGripper.settle_s 处理（cfg.gripper.read_delay_s）。"""
        g = getattr(self, "_gripper_drv", None)
        if g is not None:
            return g
        from .dh_gripper import DHGripper
        rpc = self._robot.rpc
        gcfg = self.cfg.get("gripper", {}) or {}
        # ⚠️ 真机实测：帧头匹配读 tool_read_raw_data_485_h 在这台 DUCO 恒返回空，
        #    改用 plain 读 tool_read_raw_data_485(len)；DHGripper._read_reg 内部扫帧头切帧。
        g = DHGripper(
            write485=lambda data: rpc.tool_write_raw_data_485(data),
            read485=lambda head, length: rpc.tool_read_raw_data_485(length),
            slave=int(gcfg.get("modbus_slave", 1)),
            settle_s=float(gcfg.get("read_delay_s", 0.15)),  # 485 半双工写后读延时
            read_len=int(gcfg.get("read_len", 16)),
            read_retries=int(gcfg.get("read_retries", 3)),
        )
        self._gripper_drv = g
        return g

    def gripper_initialize(self, full: bool = False) -> None:
        """初始化夹爪（回零）并等成功，再设默认力/速。
        full=False→单向初始化(0x01,快)；full=True→全行程完全标定(0xA5,更准)。
        全程持工具端 485 锁串行化，防与并发状态只读交错。"""
        gcfg = self.cfg.get("gripper", {}) or {}
        with self._gripper_lock:
            g = self._gripper()
            # 若工具端 485 默认非 115200，需先 set_baudrate_485 对齐两端（cfg.gripper.baudrate）。
            if gcfg.get("baudrate"):
                self._robot.rpc.set_baudrate_485(int(gcfg["baudrate"]))
            g.initialize(full=bool(full))
            g.wait_init(timeout_s=float(gcfg.get("init_timeout_s", 5.0)))
            if gcfg.get("force") is not None:
                g.set_force(int(gcfg["force"]))
            if gcfg.get("speed") is not None:
                g.set_speed(int(gcfg["speed"]))
            self._gripper_status_cache = None   # 状态已变，作废缓存

    def gripper_grip(self, force=None, speed=None, pos=0) -> None:
        with self._gripper_lock:
            self._gripper().grip(force=force, speed=speed, pos=pos)
            self._gripper_status_cache = None

    def gripper_release(self, speed=None) -> None:
        with self._gripper_lock:
            self._gripper().release(speed=speed)
            self._gripper_status_cache = None

    def gripper_status(self) -> dict:
        """读夹爪状态。~0.4s TTL 缓存 + 485 锁非阻塞：多个 /api/status 轮询者共享一次读，
        且与写事务串行，杜绝半双工串扰（debug_history 605-623 根因的正解）。

        关键：**绝不阻塞**。init 持锁最长 ~5s（wait_init 轮询），若 status 阻塞等锁，
        /api/status 会被夹爪读拖死 5s → UI 卡顿。故拿不到锁就立刻返回上次缓存
        （宁可旧、不卡），锁空时才真读 485 刷新。"""
        ttl = 0.4
        cached = self._gripper_status_cache
        if cached is not None and (time.monotonic() - self._gripper_status_cache_t) < ttl:
            return cached
        if not self._gripper_lock.acquire(blocking=False):
            # 锁被 init/grip 占用：返回上次缓存（无则返回全 None，调用方显示「—」）
            return cached if cached is not None else {
                "init_state": None, "grip_state": None, "position": None}
        try:
            # 双检：等锁期间别的线程可能已刷新缓存
            cached = self._gripper_status_cache
            if cached is not None and (time.monotonic() - self._gripper_status_cache_t) < ttl:
                return cached
            g = self._gripper()
            st = {
                "init_state": g.read_init_state(),
                "grip_state": g.read_grip_state(),
                "position": g.read_position(),
            }
            self._gripper_status_cache = st
            self._gripper_status_cache_t = time.monotonic()
            return st
        finally:
            self._gripper_lock.release()

    # ── 安全配置 ────────────────────────────────────────────────
    def set_workspace_bounds(self, x=None, y=None, z=None) -> None:
        s = self._robot.safety
        if x is not None:
            s.ws_x = list(x)
        if y is not None:
            s.ws_y = list(y)
        if z is not None:
            s.ws_z = list(z)

    def set_nogo_boxes(self, boxes: list) -> None:
        # GCR618 Safety 只有 box 工作空间；禁区盒检查在 check_pose_allowed 里附加
        self._nogo = list(boxes)

    def check_pose_allowed(self, pose) -> tuple:
        ok, msg = self._robot.safety.check_workspace(pose)
        if not ok:
            return False, msg
        for b in getattr(self, "_nogo", []):
            if (abs(pose[0] - b["cx"]) <= b["w"] and abs(pose[1] - b["cy"]) <= b["h"]
                    and abs(pose[2] - b["cz"]) <= b["d"]):
                return False, "TCP 进入禁区 %s" % b.get("name", "")
        return True, "ok"

    # ── 点位 ────────────────────────────────────────────────────
    def record_waypoint(self, name: str) -> None:
        fr = self._robot.last_frame()
        self._store.record(name, fr["joints"], fr["tcp"])
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
