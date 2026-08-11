"""纯Python DUCO 7003 Thrift-RPC 客户端 (无需官方SDK/ROS2)。
手写最小Thrift编解码, 基于 DucoCobot.h 反推。GCR3-618 / DUCO Core。

已实测方法: get_robot_state, power_on, enable, disable, power_off,
            movej2, get_joints, get_tcp_pose, get_last_error。
get_actual_joints_position/speed 的签名已由本机生成式 Thrift 接口确认，待现场只读验证。
标注 UNTESTED 的方法首次使用需谨慎验证 (见 plan 风险点)。
"""
import socket
import struct
import math
import threading

# thrift 仅真机连接时需要; sim 模式下可不安装。惰性导入避免无 thrift 时整个模块加载失败。
try:
    from thrift.transport import TSocket, TTransport
    from thrift.protocol import TBinaryProtocol
    _THRIFT_OK = True
except ImportError:  # pragma: no cover - thrift 未安装时走 sim 模式
    TSocket = TTransport = TBinaryProtocol = None
    _THRIFT_OK = False

# ---- Thrift type ids ----
T_STOP, T_BOOL, T_BYTE, T_I16, T_I32, T_I64, T_DOUBLE, T_STRING = 0, 2, 3, 6, 8, 10, 4, 11
T_STRUCT, T_MAP, T_SET, T_LIST = 12, 13, 14, 15
# 伪类型（仅本客户端内部用，非 thrift 线上 id）：标记「list<byte>(i8) 字段」。
# 线上字段头类型仍写 T_LIST(15)、列表元素类型写 T_BYTE(3)；_write_val 据此写字节而非 double。
# 用于工具端 485 裸帧透传（tool_write_raw_data_485 的 data、tool_read_raw_data_485_h 的 head）。
T_LIST_I8 = 103

# ---- 状态枚举 (来自 DucoCobot.h) ----
ST_ROBOT = {0: "Start", 1: "Init", 2: "Logout", 3: "Login", 4: "PowerOff下电", 5: "上电未使能", 6: "Enable已使能"}
ST_PROG = {0: "停止", 1: "停止中", 2: "运行中", 3: "暂停", 4: "暂停中", 5: "任务运行"}
ST_MODE = {0: "手动Manual", 1: "自动Auto", 2: "远程Remote"}
ST_SAFE = {0: "INIT", 2: "WAIT等待", 3: "CONFIG", 4: "POWER_OFF", 5: "RUN运行", 6: "RECOVERY恢复",
           7: "STOP2", 8: "STOP1急停", 9: "STOP0", 10: "MODEL", 12: "REDUCE缩减", 13: "BOOT",
           14: "FAIL故障", 99: "UPDATE"}
ST_TASK = {0: "Idle", 1: "Running", 2: "Paused", 3: "Stopped", 4: "Finished完成", 5: "Interrupt",
           6: "Error", 7: "Illegal非法", 8: "ParamMismatch"}

STATUS_FRAME_LEN = 1468
# 注：远程上电/使能/切模式的冻结已彻底移除（用户要求软件层全权限）。远程生命周期由
# robot_control.Robot.ensure_enabled/set_op_mode/attempt_recover 真调 DUCO RPC 实现。


def _finite_vector(values, length, label):
    if isinstance(values, (str, bytes)):
        raise ValueError("%s 必须是 %d 个有限数" % (label, length))
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("%s 必须是 %d 个有限数" % (label, length)) from exc
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError("%s 必须是 %d 个有限数" % (label, length))
    return result


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
    return _symmetric_matrix_is_psd(
        (
            half_trace - xx,
            -xy,
            -xz,
            half_trace - yy,
            -yz,
            half_trace - zz,
        )
    )


class DucoRPC:
    """7003 Thrift RPC 控制通道 + 2001 状态读取。"""

    def __init__(self, ip="192.168.1.10", port=7003, timeout=10000):
        if not _THRIFT_OK:
            raise RuntimeError(
                "thrift 库未安装, 无法连接真机。请 `pip install thrift`。"
                "(仿真模式无需 thrift, 请改用 SimRobotArmBackend)")
        self.ip = ip
        self.t = TSocket.TSocket(ip, port)
        self.t.setTimeout(timeout)
        self.tr = TTransport.TBufferedTransport(self.t)
        self.p = TBinaryProtocol.TBinaryProtocol(self.tr)
        self._seq = 0
        self._rpc_lock = threading.RLock()

    def open(self):
        self.tr.open()
        return self

    def close(self):
        try:
            self.tr.close()
        except Exception:
            pass

    def __enter__(self):
        return self.open()

    def __exit__(self, *a):
        self.close()

    # ================= 通用 Thrift 调用 =================
    def _call(self, method, fields=()):
        """fields: list of (field_id, ttype, value).
        支持编码: bool/i16/i32/i64/double/string/list<double>/list<list<double>>/list<byte>(T_LIST_I8)。"""
        with self._rpc_lock:
            self._seq += 1
            sequence_id = self._seq
            p = self.p
            p.writeMessageBegin(method, 1, sequence_id)  # 1 = CALL
            p.writeStructBegin(method + "_args")
            for fid, ttype, val in fields:
                # T_LIST_I8 是本客户端伪类型：字段头写真实容器类型 T_LIST，元素类型由 _write_val 写 T_BYTE
                wire_ttype = T_LIST if ttype == T_LIST_I8 else ttype
                p.writeFieldBegin("", wire_ttype, fid)
                self._write_val(ttype, val)
                p.writeFieldEnd()
            p.writeFieldStop()
            p.writeStructEnd()
            p.writeMessageEnd()
            self.tr.flush()
            return self._read_reply(method, sequence_id)

    def _write_val(self, ttype, val):
        p = self.p
        if ttype == T_BOOL:
            p.writeBool(val)
        elif ttype == T_I16:
            p.writeI16(val)
        elif ttype == T_I32:
            p.writeI32(val)
        elif ttype == T_I64:
            p.writeI64(val)
        elif ttype == T_DOUBLE:
            p.writeDouble(val)
        elif ttype == T_STRING:
            p.writeString(val)
        elif ttype == T_LIST_I8:
            # list<byte>(i8)：工具端 485 裸帧字节流（值 0-255）。⚠️ 待真机验证。
            # Apache thrift writeByte 用有符号 pack('!b')(-128..127)，故 128-255 需转补码。
            p.writeListBegin(T_BYTE, len(val))
            for x in val:
                b = int(x) & 0xFF
                p.writeByte(b - 256 if b > 127 else b)
            p.writeListEnd()
        elif ttype == T_LIST:
            # list<double> 或 list<list<double>> (轨迹点列)
            if val and isinstance(val[0], (list, tuple)):
                p.writeListBegin(T_LIST, len(val))
                for pt in val:
                    p.writeListBegin(T_DOUBLE, len(pt))
                    for x in pt:
                        p.writeDouble(x)
                    p.writeListEnd()
                p.writeListEnd()
            else:
                p.writeListBegin(T_DOUBLE, len(val))
                for x in val:
                    p.writeDouble(x)
                p.writeListEnd()

    def _read_reply(self, expected_method=None, expected_sequence_id=None):
        p = self.p
        name, mtype, seq = p.readMessageBegin()
        if mtype == 3:  # EXCEPTION
            p.skip(T_STRUCT)
            p.readMessageEnd()
            raise RuntimeError("Thrift EXCEPTION on " + name)
        mismatch = None
        if mtype != 2:  # 2 = REPLY
            mismatch = "响应消息类型错误: expected=2 actual=%s" % mtype
        if expected_method is not None and name != expected_method:
            mismatch = "响应方法错配: expected=%s actual=%s" % (expected_method, name)
        if expected_sequence_id is not None and seq != expected_sequence_id:
            mismatch = "响应序号错配: expected=%s actual=%s" % (expected_sequence_id, seq)
        result = None
        p.readStructBegin()
        while True:
            fname, ftype, fid = p.readFieldBegin()
            if ftype == T_STOP:
                break
            if fid == 0:  # 返回值字段
                result = self._read_val(ftype)
            else:
                p.skip(ftype)
            p.readFieldEnd()
        p.readStructEnd()
        p.readMessageEnd()
        if mismatch is not None:
            raise RuntimeError(mismatch)
        return result

    def _read_val(self, ftype):
        p = self.p
        if ftype == T_I32:
            return p.readI32()
        if ftype == T_I64:
            return p.readI64()
        if ftype == T_DOUBLE:
            return p.readDouble()
        if ftype == T_BOOL:
            return p.readBool()
        if ftype == T_STRING:
            return p.readString()
        if ftype == T_LIST:
            et, sz = p.readListBegin()
            if et == T_BYTE:
                out = [p.readByte() for _ in range(sz)]
            elif et == T_DOUBLE:
                out = [p.readDouble() for _ in range(sz)]
            elif et == T_STRING:
                out = [p.readString() for _ in range(sz)]
            elif et == T_I32:
                out = [p.readI32() for _ in range(sz)]
            else:
                out = [p.skip(et) for _ in range(sz)]
            p.readListEnd()
            return out
        p.skip(ftype)
        return None

    # ================= 系统控制 (已实测) =================
    def get_robot_state(self):
        return self._call("get_robot_state")

    def power_on(self, block=True):
        # 管理员授权：开放远程上电（真调 DUCO 已实测 RPC，与 disable/power_off 同构）。
        return self._call("power_on", [(1, T_BOOL, block)])

    def enable(self, block=True):
        # 管理员授权：开放远程使能（真调 DUCO 已实测 RPC）。
        return self._call("enable", [(1, T_BOOL, block)])

    def disable(self, block=True):
        return self._call("disable", [(1, T_BOOL, block)])

    def power_off(self, block=True):
        return self._call("power_off", [(1, T_BOOL, block)])

    def get_last_error(self):
        return self._call("get_last_error")  # -> list[str]

    def get_actual_joints_position(self):
        """从控制 RPC 读取实际关节位置，独立于 2001 状态帧。"""
        return self._call("get_actual_joints_position")

    def get_actual_joints_speed(self):
        """从控制 RPC 读取实际关节速度，单位 rad/s。"""
        return self._call("get_actual_joints_speed")

    # ================= 运动 =================
    def movej2(self, joints, v, a, r=0.0, block=True):
        """关节运动 (已实测). joints: 6xrad, v: rad/s, a: rad/s^2, r: 融合半径m。"""
        return self._call("movej2", [(1, T_LIST, list(joints)), (2, T_DOUBLE, v),
                                      (3, T_DOUBLE, a), (4, T_DOUBLE, r), (5, T_BOOL, block)])

    def get_noneblock_taskstate(self, task_id):
        """查询非阻塞任务状态。task_id 为 movej2(block=False) 返回的 I32。"""
        return self._call("get_noneblock_taskstate", [(1, T_I32, int(task_id))])

    # UNTESTED ----
    def movel(self, p, v, a, r, q_near, tool="default", wobj="default", block=True):
        """直线运动. p:位姿6, q_near:逆解参考关节6, tool/wobj:坐标系名。UNTESTED"""
        return self._call("movel", [(1, T_LIST, list(p)), (2, T_DOUBLE, v), (3, T_DOUBLE, a),
                                     (4, T_DOUBLE, r), (5, T_LIST, list(q_near)),
                                     (6, T_STRING, tool), (7, T_STRING, wobj), (8, T_BOOL, block)])

    def track_enqueue(self, track, block=True):
        """灌入轨迹池. track: list[list[double]] 每点6维。UNTESTED (速度来源待验证)"""
        return self._call("trackEnqueue", [(1, T_LIST, track), (2, T_BOOL, block)])

    def track_clear_queue(self):
        return self._call("trackClearQueue")  # UNTESTED

    # ================= 示教 / 录制 =================
    def teach_mode(self, block=True):
        """进入牵引示教 (人可徒手拖动). UNTESTED - 最高风险, 需先 set_load_data + 人扶。"""
        return self._call("teach_mode", [(1, T_BOOL, block)])

    def end_teach_mode(self, block=True):
        return self._call("end_teach_mode", [(1, T_BOOL, block)])  # UNTESTED

    def start_record_track(self, name, mode, tool="default", wobj="default"):
        """控制器内置录制 (备选). mode=0按位置(每5°), mode=1按时间(每250ms). UNTESTED"""
        return self._call("start_record_track", [(1, T_STRING, name), (2, T_I32, mode),
                                                  (3, T_STRING, tool), (4, T_STRING, wobj)])

    def stop_record_track(self):
        return self._call("stop_record_track")  # UNTESTED

    def replay(self, name, speed_pct, mode):
        """回放录制轨迹 (备选). mode=0关节空间, 1笛卡尔空间. UNTESTED"""
        return self._call("replay", [(1, T_STRING, name), (2, T_I32, speed_pct), (3, T_I32, mode)])

    # ================= IO / 负载 / 安全 =================
    def set_tool_digital_out(self, num, value, block=True):
        """末端工具数字输出 (吸盘). UNTESTED - do_num/极性需脱机万用表先测。"""
        return self._call("set_tool_digital_out", [(1, T_I16, num), (2, T_BOOL, value), (3, T_BOOL, block)])

    def set_standard_digital_out(self, num, value, block=True):
        return self._call("set_standard_digital_out", [(1, T_I16, num), (2, T_BOOL, value), (3, T_BOOL, block)])

    def set_load_data(self, mass, cx, cy, cz):
        """设置末端负载 (质量kg + 质心xyz m, 相对工具系). 进牵引示教前必设。UNTESTED"""
        return self._call("set_load_data", [(1, T_LIST, [mass, cx, cy, cz])])

    def collision_detect(self, level):
        """碰撞检测等级. 0关闭, 1-5灵敏度。UNTESTED"""
        return self._call("collision_detect", [(1, T_I32, level)])

    def program_stop(self):
        """DUCO 官方 `stop`：停止/清空当前指令任务容器，把 prog_state 从 3(暂停) 拉回 0(停止)。

        ★ 与 `robot_control.Robot.stop()` 完全不是一回事——那个是 `_disable_and_verify()` 下使能。
        本方法**不动使能、不动安全态**，实测 2026-07-30：`[6,3,5,0] → [6,0,5,0]`，返回 1063。

        什么时候需要它：Auto↔Manual 切模式会把程序挂起成 `prog_state=3`，**切回自动不会自动恢复**，
        示教器上也没有对应的取消按钮（因为并没有用户程序在跑，被挂起的是指令任务容器）。
        此后所有 movej 都会被 `_assert_ready()` 拒发，而 `FastDriver.goto` 会吞掉那个异常，
        表面上只看到"未到位"。凡是拖动示教流程切过手动模式，回自动后都该调一次本方法。
        """
        return self._call("stop")

    def switch_mode(self, mode):
        """手自动模式切换. 0手动, 1自动。UNTESTED"""
        return self._call("switch_mode", [(1, T_I32, mode)])

    # ================= 2001 状态帧 =================
    def _read_2001_once(self, timeout=3):
        s = socket.socket()
        s.settimeout(timeout)
        s.connect((self.ip, 2001))
        buf = b""
        try:
            while len(buf) < STATUS_FRAME_LEN:
                chunk = s.recv(STATUS_FRAME_LEN - len(buf))
                if not chunk:
                    break
                buf += chunk
        finally:
            s.close()
        if len(buf) < STATUS_FRAME_LEN:
            raise IOError("2001帧不完整: %d/%d" % (len(buf), STATUS_FRAME_LEN))
        return buf

    def read_status_frame(self, retries=3, timeout=3):
        """读一整帧 2001 (1468B), 解析常用字段为命名 dict。内置重试容忍偶发超时。"""
        retries = int(retries)
        timeout = float(timeout)
        if retries < 1 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("read_status_frame 的 retries/timeout 无效")
        last_exc = None
        for _ in range(retries):
            try:
                buf = self._read_2001_once(timeout=timeout)
                break
            except Exception as e:
                last_exc = e
        else:
            raise last_exc
        f = buf[:STATUS_FRAME_LEN]
        return {
            "joints": list(struct.unpack_from("<6f", f, 0)),       # rad
            "tcp": list(struct.unpack_from("<6f", f, 368)),        # X,Y,Z,Rx,Ry,Rz (m,rad)
            "op_mode": f[1448],        # 0手动/1自动/2远程
            "robot_state": f[1449],    # 4下电/5上电未使能/6使能
            "prog_state": f[1450],     # 0停/2运行
            "safety": f[1451],         # 安全监控 5=RUN
            "collision": f[1452],
            "collision_axis": f[1453],
            "alarm": struct.unpack_from("<I", f, 1456)[0],
        }

    def get_joints(self):
        return self.read_status_frame()["joints"]

    def get_tcp_pose(self):
        return self.read_status_frame()["tcp"]

    # ============= 运动学 / 工具标定 (block0 包装, 视觉引导取放的逆解需要) =============
    def set_tool_data(self, name, tool_offset, payload, inertia_tensor=None):
        """设置并激活工具 TCP。tool_offset[x,y,z,rx,ry,rz](m,rad), payload[mass,cx,cy,cz](kg,m),
        inertia_tensor[xx,xy,xz,yy,yz,zz]。返回 int 状态。设后续运动/正逆解默认用此 TCP。"""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("工具名称不能为空")
        tool_offset = _finite_vector(tool_offset, 6, "tool_offset")
        payload = _finite_vector(payload, 4, "payload")
        if payload[0] < 0.0 or payload[0] > 3.0:
            raise ValueError("payload.mass 必须在 GCR3 额定范围 [0,3]kg")
        if inertia_tensor is None:
            inertia_tensor = [0.0] * 6
        inertia_tensor = _finite_vector(inertia_tensor, 6, "inertia_tensor")
        if not _inertia_tensor_is_physical(inertia_tensor, payload[0]):
            raise ValueError("inertia_tensor 不满足正半定与刚体三角一致性")
        result = self._call("set_tool_data", [
            (1, T_STRING, name),
            (2, T_LIST, tool_offset),
            (3, T_LIST, payload),
            (4, T_LIST, inertia_tensor),
        ])
        if isinstance(result, bool) or not isinstance(result, int):
            raise RuntimeError("set_tool_data 返回状态格式错误: %r" % (result,))
        return result

    def get_tcp_offset(self):
        """获取当前生效工具 TCP 偏移 [x,y,z,rx,ry,rz] (m,rad)。"""
        return _finite_vector(self._call("get_tcp_offset"), 6, "get_tcp_offset 返回值")

    def get_tool_load(self):
        """获取当前工具负载 [mass,cx,cy,cz] (kg,m)。"""
        result = self._call("get_tool_load")
        if (
            not isinstance(result, (list, tuple))
            or any(isinstance(value, bool) for value in result)
        ):
            raise ValueError("get_tool_load 返回值必须是 4 个有限数")
        load = _finite_vector(result, 4, "get_tool_load 返回值")
        if load[0] < 0.0 or load[0] > 3.0:
            raise ValueError("get_tool_load.mass 必须在 GCR3 额定范围 [0,3]kg")
        return load

    def cal_fkine(self, joints_position, tool=None, wobj=None):
        """正解: 关节(rad) -> 末端位姿[x,y,z,rx,ry,rz]。tool/wobj 为偏移向量, 空=当前值。
        取纯法兰位姿务必传 tool=[0,0,0,0,0,0](空 tool=当前激活 TCP, 不是法兰)。"""
        return self._call("cal_fkine", [
            (1, T_LIST, [float(x) for x in joints_position]),
            (2, T_LIST, [float(x) for x in (tool or [])]),
            (3, T_LIST, [float(x) for x in (wobj or [])]),
        ])

    def cal_ikine(self, p, q_near=None, tool=None, wobj=None):
        """逆解: 末端位姿 -> 关节[q1..q6], 选靠近 q_near 的解支。tool/wobj 偏移向量, 空=当前。
        空 q_near=当前关节。"""
        return self._call("cal_ikine", [
            (1, T_LIST, [float(x) for x in p]),
            (2, T_LIST, [float(x) for x in (q_near or [])]),
            (3, T_LIST, [float(x) for x in (tool or [])]),
            (4, T_LIST, [float(x) for x in (wobj or [])]),
        ])

    # ============= 工具端 485 裸帧透传（大寰夹爪 Modbus 主站，待真机验证）=============
    # ⚠️ 待真机验证：以下四个方法据 DucoCobot.h SDK 头反推，未对 192.168.1.10:7003 实测。
    #    风险点（见 docs/dh_gripper_pgea_control.md §7）：
    #      · 字段 id / 类型（list<byte>=i8 用 T_LIST_I8 编码，见 _write_val）为最可能推断；
    #      · block 是否需随 tool_write 单独下发（当前只发 data 一个字段，见下）；
    #      · 工具端 485（tool_ 前缀）与机身 485（无 tool_ 前缀）方法名须区分，夹爪走工具端。
    def tool_write_raw_data_485(self, data, block=True):
        """把裸字节帧写工具端 485（夹爪 Modbus 主站发送）。data: 可迭代字节 0-255。返回 bool。
        ⚠️ 待真机验证：block 当前未随帧下发（只发 field 1=data）；DucoCobot.h 若要求
        block 作 field 2，需补 (2, T_BOOL, block)。"""
        return self._call("tool_write_raw_data_485",
                          [(1, T_LIST_I8, [int(x) & 0xFF for x in data])])

    def tool_read_raw_data_485_h(self, head, length):
        """匹配帧头 head 后读 length 字节（读夹爪响应）。head: list<byte>；length: int。
        返回 list[int]（thrift byte 有符号 -128..127，调用方按需 & 0xFF）。⚠️ 待真机验证。"""
        return self._call("tool_read_raw_data_485_h",
                          [(1, T_LIST_I8, [int(x) & 0xFF for x in head]), (2, T_I32, int(length))])

    def tool_read_raw_data_485(self, length):
        """读工具端 485 的 length 字节（无帧头匹配）。返回 list[int]。⚠️ 待真机验证。"""
        return self._call("tool_read_raw_data_485", [(1, T_I32, int(length))])

    def set_baudrate_485(self, value, block=True):
        """设置工具端 485 波特率（大寰默认 115200）。⚠️ 待真机验证。"""
        return self._call("set_baudrate_485", [(1, T_I32, int(value)), (2, T_BOOL, bool(block))])


# ---- 便捷打印 ----
def show(tag, st):
    print("  [%s] state=%s -> 机器人:%s 程序:%s 安全:%s 模式:%s" % (
        tag, st, ST_ROBOT.get(st[0], st[0]), ST_PROG.get(st[1], st[1]),
        ST_SAFE.get(st[2], st[2]), ST_MODE.get(st[3], st[3])))


def show_frame(fr):
    print("  机器人:%s | 安全:%s | 模式:%s | 程序:%s | 报警:%s | 碰撞:%s" % (
        ST_ROBOT.get(fr["robot_state"], fr["robot_state"]),
        ST_SAFE.get(fr["safety"], fr["safety"]),
        ST_MODE.get(fr["op_mode"], fr["op_mode"]),
        ST_PROG.get(fr["prog_state"], fr["prog_state"]),
        hex(fr["alarm"]), fr["collision"]))
    print("  关节deg:", [round(math.degrees(x), 2) for x in fr["joints"]])
    p = fr["tcp"]
    print("  TCP: X=%.1f Y=%.1f Z=%.1f mm | Rx=%.2f Ry=%.2f Rz=%.2f deg" % (
        p[0] * 1000, p[1] * 1000, p[2] * 1000,
        math.degrees(p[3]), math.degrees(p[4]), math.degrees(p[5])))
