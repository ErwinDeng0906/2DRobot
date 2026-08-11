"""大寰 DH-Robotics PGEA-50-26-O 电动夹爪 · 纯逻辑驱动（Modbus-RTU over 工具端 485）。

stdlib-only（无 numpy / 无网络）：拼 Modbus-RTU 帧 + CRC-16/Modbus，经**注入**的
write485 / read485 透传收发。可离线 import + 自检（见 __main__，逐字节比对手册已验证帧）。

寄存器 / 取值 / 帧格式据 docs/dh_gripper_pgea_control.md（DH 手册 + 服务器 DucoCobot.h + 本地 CRC 验证）。

⚠️ 待真机验证：485 半双工写后读的时序（settle_s）、夹持状态码 0x0201 的 0/1/2/3 语义、
   工具端 485 波特率是否需先设 115200——见 docs §7。本模块的**拼帧/CRC** 已本地逐字节验证。
"""
import time

# ── Modbus 功能码 ──
_FUNC_READ = 0x03    # 读保持寄存器
_FUNC_WRITE = 0x06   # 写单寄存器

# ── PGE 控制寄存器（读写，docs §3）──
REG_INIT = 0x0100        # 初始化：0x01=初始化 / 0xA5=完全初始化（回零标定）
REG_FORCE = 0x0101       # 力值 20–100（%）
REG_POSITION = 0x0103    # 位置 0–1000（‰）；0=全合(夹紧)，1000=全开
REG_SPEED = 0x0104       # 速度 1–100（%）
# ── PGE 反馈寄存器（只读，docs §3）──
REG_INIT_STATE = 0x0200  # 0=未初始化 / 1=初始化成功 / 2=初始化中
REG_GRIP_STATE = 0x0201  # 0=运动中 / 1=到位未夹到 / 2=夹住物体 / 3=物体掉落  ⚠️ 码值待真机核实
REG_POS_FB = 0x0202      # 位置反馈 0–1000（‰）

# ── 初始化取值 ──
INIT_NORMAL = 0x01
INIT_FULL = 0xA5

# ── 扩展控制/参数寄存器（PGEA V3.3 手册 表2.2/2.3）──
REG_JOG = 0x010A         # 点动：写 1 正向 / 0 停止 / -1(0xFFFF) 反向
REG_ERROR = 0x0205       # 错误/报警码（只读）：0=无错误，非0=报警码
REG_SAVE = 0x0300        # 写入保存到 Flash：写 1 保存全部参数（耗时 1-2s，期间不响应）
REG_INIT_DIR = 0x0301    # 初始化方向：0=打开(默认)/1=关闭
REG_DEVICE_ID = 0x0302   # 设备 Modbus ID：1-247（默认 1）
REG_BAUD = 0x0303        # 波特率档 0-5：115200/57600/38400/19200/9600/4800（0 默认）
REG_STOPBITS = 0x0304    # 停止位：0=1位/1=2位
REG_PARITY = 0x0305      # 校验位：0=无/1=奇/2=偶
REG_BRAKE = 0x0502       # 停转：1=规划性停转/2=恢复运动到刹车前位置
REG_CLEAR_ALARM = 0x0503 # 复位：1=清除当前报警
REG_AUTO_INIT = 0x0504   # 上电自动初始化：0=不/1=自动/165=完全初始化
REG_HOLD_BRAKE = 0x0506  # 抱闸：1=打开/2=锁紧（仅带抱闸机型）

# 点动方向取值
JOG_FORWARD = 1
JOG_STOP = 0
JOG_REVERSE = -1


def crc16_modbus(data):
    """CRC-16/Modbus：多项式 0xA001，初值 0xFFFF。返回 16-bit int（低字节在前由调用方拆）。"""
    crc = 0xFFFF
    for b in data:
        crc ^= (int(b) & 0xFF)
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def _with_crc(body):
    """给帧体（list[int]）追加 CRC（低字节在前），返回完整帧 list[int]。"""
    crc = crc16_modbus(body)
    return list(body) + [crc & 0xFF, (crc >> 8) & 0xFF]


def build_write(slave, addr, val):
    """写单寄存器帧（func 0x06）：[从站, 0x06, 地址H, 地址L, 数据H, 数据L, CRC_L, CRC_H]。"""
    body = [slave & 0xFF, _FUNC_WRITE,
            (addr >> 8) & 0xFF, addr & 0xFF,
            (val >> 8) & 0xFF, val & 0xFF]
    return _with_crc(body)


def build_read(slave, addr, n=1):
    """读保持寄存器帧（func 0x03）：[从站, 0x03, 地址H, 地址L, 数量H, 数量L, CRC_L, CRC_H]。"""
    body = [slave & 0xFF, _FUNC_READ,
            (addr >> 8) & 0xFF, addr & 0xFF,
            (n >> 8) & 0xFF, n & 0xFF]
    return _with_crc(body)


def parse_read_response(resp, slave=None):
    """校验读响应帧并返回首个 16-bit 寄存器值。坏帧抛 ValueError。

    帧格式：[从站, 0x03, 字节数, 数据H, 数据L, ..., CRC_L, CRC_H]（读 1 寄存器→字节数=0x02）。
    resp 里的字节可能是 thrift 有符号 byte（-128..127），先统一 & 0xFF。
    slave 非空时额外校验从站地址一致。
    """
    b = [int(x) & 0xFF for x in resp]
    if len(b) < 7:  # 最短：从站+功能+字节数(2)+数据2+CRC2 = 7
        raise ValueError("读响应过短(%d字节): %r" % (len(b), b))
    r_slave, func, bytecount = b[0], b[1], b[2]
    if slave is not None and r_slave != (int(slave) & 0xFF):
        raise ValueError("从站地址不符：期望 %d 实收 %d" % (int(slave) & 0xFF, r_slave))
    if func != _FUNC_READ:
        raise ValueError("功能码非 0x03（读）：0x%02X" % func)
    if bytecount < 2 or bytecount % 2 != 0:
        raise ValueError("字节数非法：%d" % bytecount)
    need = 3 + bytecount + 2  # 头3 + 数据 + CRC2
    if len(b) < need:
        raise ValueError("帧长不足：%d < %d" % (len(b), need))
    frame = b[:need]
    crc_recv = frame[-2] | (frame[-1] << 8)
    crc_calc = crc16_modbus(frame[:-2])
    if crc_recv != crc_calc:
        raise ValueError("CRC 校验失败：收 0x%04X 算 0x%04X（帧 %r）" % (crc_recv, crc_calc, frame))
    return (frame[3] << 8) | frame[4]


class DHGripper:
    """PGEA-50-26-O 夹爪驱动。透传收发由外部注入（真机=duco_rpc 工具端 485；测试=mock）。

    write485(data: list[int]) -> bool          把裸帧字节写工具端 485（夹爪 Modbus 主站发送）
    read485(head: list[int], length: int) -> list[int]  匹配帧头后读 length 字节（夹爪响应）

    slave    : Modbus 从站地址（默认 1）
    settle_s : 485 半双工写后等夹爪响应的时长（秒），再读回。⚠️ 待真机验证。
    """

    def __init__(self, write485, read485, slave=1, settle_s=0.02, read_len=16, read_retries=3):
        self._w = write485
        self._r = read485
        self.slave = int(slave) & 0xFF
        self.settle_s = float(settle_s)
        self.read_len = int(read_len)          # plain 读一次抓回的字节数（宽于 7，容前导残留/双帧）
        self.read_retries = max(1, int(read_retries))  # 半双工偶发空读时的重试次数
        self.flush_reads = 4                   # 每次读前丢弃式清缓冲的最大次数（半双工防串帧）

    # ── 底层单寄存器读写 ──────────────────────────────────────
    def _write_reg(self, addr, val):
        """写单寄存器（func 0x06）并等 settle_s。返回 write485 结果。写响应回显请求帧，忽略。"""
        ok = self._w(build_write(self.slave, addr, val))
        if self.settle_s > 0:
            time.sleep(self.settle_s)
        return ok

    def _read_reg(self, addr):
        """读单寄存器（func 0x03）：发读帧 → 等 settle_s → plain 读原始字节 → 扫帧头切帧解析。

        ⚠️ 真机实测（2026-07-08，DUCO GCR3-618 工具端 485@9600）：
        1) 帧头匹配读 `tool_read_raw_data_485_h` 在这台控制器上恒返回空 → 改用 plain 读
           `tool_read_raw_data_485` 一次抓回缓冲区全部字节；
        2) 半双工缓冲会残留**上一次**读的响应帧，而 Modbus 读响应**不含寄存器地址**、帧头都是
           `[slave,0x03,0x02]` 无法区分 → 不清缓冲会把旧寄存器的值张冠李戴（实测 init/grip 串到
           position 值）。
        故每次读**前先丢弃式清空缓冲**，再发请求、等响应、扫帧头切出首个完整 7 字节帧解析；空读重试。
        """
        head = [self.slave, _FUNC_READ, 0x02]
        last_err = None
        for _ in range(self.read_retries):
            # ① 预清：丢弃缓冲区里上一次的残留响应（半双工防串帧）
            for _f in range(self.flush_reads):
                try:
                    junk = self._r(head, self.read_len)
                except Exception:
                    break
                if not junk:
                    break
            # ② 发读请求 → 等半双工响应
            self._w(build_read(self.slave, addr, 1))
            if self.settle_s > 0:
                time.sleep(self.settle_s)
            resp = [int(x) & 0xFF for x in (self._r(head, self.read_len) or [])]
            # ③ 扫帧头，切出首个 [slave,0x03,0x02,DH,DL,CRC_L,CRC_H]
            frame = None
            for i in range(len(resp) - 6):
                if resp[i] == self.slave and resp[i + 1] == _FUNC_READ and resp[i + 2] == 0x02:
                    frame = resp[i:i + 7]
                    break
            if frame is None:
                last_err = ValueError("读 0x%04X 未匹配到帧头，原始=%r" % (addr, resp))
                continue
            try:
                return parse_read_response(frame, slave=self.slave)
            except ValueError as e:  # 帧头对上但 CRC 坏 → 重试
                last_err = e
        raise last_err or ValueError("读 0x%04X 无有效响应" % addr)

    # ── 初始化 ────────────────────────────────────────────────
    def initialize(self, full=False):
        """写 0x0100：full=False→0x01 初始化；full=True→0xA5 完全初始化（回零标定）。"""
        return self._write_reg(REG_INIT, INIT_FULL if full else INIT_NORMAL)

    def wait_init(self, timeout_s=5.0, poll_s=0.2):
        """轮询 0x0200 直到 =1（初始化成功）。超时抛 TimeoutError。约 1–3s（docs §5）。"""
        deadline = time.time() + float(timeout_s)
        st = None
        while True:
            st = self.read_init_state()
            if st == 1:
                return True
            if time.time() >= deadline:
                raise TimeoutError(
                    "夹爪初始化超时(%.1fs)，末态=%r（0未初始化/2初始化中）" % (timeout_s, st))
            time.sleep(poll_s)

    # ── 参数 ──────────────────────────────────────────────────
    def set_force(self, pct):
        """力值 20–100（%），越界钳制。"""
        return self._write_reg(REG_FORCE, max(20, min(100, int(pct))))

    def set_speed(self, pct):
        """速度 1–100（%），越界钳制。"""
        return self._write_reg(REG_SPEED, max(1, min(100, int(pct))))

    def move_to(self, permille):
        """位置 0–1000（‰）：0=全合(夹紧)，1000=全开。越界钳制。"""
        return self._write_reg(REG_POSITION, max(0, min(1000, int(permille))))

    # ── 高层动作 ──────────────────────────────────────────────
    def grip(self, force=None, speed=None, pos=0):
        """夹取（合）：可选先设力/速，再写位置到 pos（默认 0=全合夹紧）。"""
        if force is not None:
            self.set_force(force)
        if speed is not None:
            self.set_speed(speed)
        return self.move_to(pos)

    def release(self, pos=1000, speed=None):
        """释放（开）：可选设速，写位置到 pos（默认 1000=全开）。机械张开，无需破真空。"""
        if speed is not None:
            self.set_speed(speed)
        return self.move_to(pos)

    # ── 状态读取 ──────────────────────────────────────────────
    def read_init_state(self):
        """读 0x0200：0=未初始化 / 1=成功 / 2=初始化中。"""
        return self._read_reg(REG_INIT_STATE)

    def read_grip_state(self):
        """读 0x0201：0=运动中 / 1=到位未夹到 / 2=夹住物体 / 3=物体掉落。⚠️ 码值待真机核实。"""
        return self._read_reg(REG_GRIP_STATE)

    def read_position(self):
        """读 0x0202：位置反馈 0–1000（‰）。"""
        return self._read_reg(REG_POS_FB)

    def read_error(self):
        """读 0x0205：错误/报警码。0=无错误，非 0=报警码（PGEA V3.3 §2.3.2）。"""
        return self._read_reg(REG_ERROR)

    # ── 力控 / 位置 两种模式（都用已验证寄存器组合，语义见 docs PGEA V3.3 §2.3.3.2）──
    def grip_force_mode(self, force, speed=None, close_pos=0):
        """力控模式：设力值(0x0101)后闭合到 close_pos(默认0=全合)。夹爪以设定力去夹，
        夹到物体(位控到不了目标)即以该力保持，读 0x0201=2 表示夹住、=3 掉落。
        force 20-100(%)，speed 1-100(%) 可选。"""
        self.set_force(force)
        if speed is not None:
            self.set_speed(speed)
        return self.move_to(close_pos)

    def move_hold(self, permille, speed=None):
        """位置模式：移动到 permille(0-1000‰) 并伺服保持。可选设速。读 0x0202 确认到位。"""
        if speed is not None:
            self.set_speed(speed)
        return self.move_to(permille)

    # ── 点动 / 停转 / 报警复位（PGEA V3.3 §2.3.2 表2.2/2.3）──
    def jog(self, direction):
        """点动 0x010A：direction=1 正向(张开方向)/0 停止/-1 反向(闭合方向)。掩到 16bit。"""
        return self._write_reg(REG_JOG, int(direction) & 0xFFFF)

    def stop_motion(self):
        """停转 0x0502=1：规划性停转（急停夹爪运动、保持当前位置）。"""
        return self._write_reg(REG_BRAKE, 1)

    def resume_motion(self):
        """0x0502=2：恢复运动到刹车前的目标位置。"""
        return self._write_reg(REG_BRAKE, 2)

    def clear_alarm(self):
        """复位 0x0503=1：清除当前报警。"""
        return self._write_reg(REG_CLEAR_ALARM, 1)

    # ── 配置类（⚠️ 慎用：写后须 save_to_flash 且改 ID/波特率会改变通信参数，可能失联；
    #    实时控制中勿用；改前务必确认。不建议在 UI 常规操作里暴露）──
    def save_to_flash(self):
        """写入保存 0x0300=1：把配置写进 Flash（耗时 1-2s，期间不响应其他命令）。"""
        return self._write_reg(REG_SAVE, 1)

    def set_init_direction(self, closed):
        """初始化方向 0x0301：closed=False→0 打开(张开为起点)/True→1 关闭(闭合为起点)。写后需 save。"""
        return self._write_reg(REG_INIT_DIR, 1 if closed else 0)

    def set_device_id(self, dev_id):
        """设备 ID 0x0302：1-247。⚠️ 改后本从站地址变化、须 save+按新 ID 通信。"""
        return self._write_reg(REG_DEVICE_ID, max(1, min(247, int(dev_id))))

    def set_baud_code(self, code):
        """波特率 0x0303：0-5=115200/57600/38400/19200/9600/4800。⚠️ 改后须 save 且两端同步改否则失联。"""
        return self._write_reg(REG_BAUD, max(0, min(5, int(code))))

    def set_auto_init(self, mode):
        """上电自动初始化 0x0504：0=不/1=自动(发0x01)/165=完全初始化(0xA5)。写后需 save。"""
        return self._write_reg(REG_AUTO_INIT, int(mode) & 0xFFFF)


if __name__ == "__main__":
    # ── 离线自检：逐字节比对 docs §4 已验证帧 + CRC 往返 + mock transport 收发链路 ──
    # 1) 拼帧逐字节对手册（含 task 指定的四条硬断言）
    f = build_write(1, 0x0100, 1)
    assert f == [0x01, 0x06, 0x01, 0x00, 0x00, 0x01, 0x49, 0xF6], f
    f = build_write(1, 0x0100, 0xA5)
    assert f == [0x01, 0x06, 0x01, 0x00, 0x00, 0xA5, 0x48, 0x4D], f
    assert f[-3:] == [0xA5, 0x48, 0x4D], f          # ...0xA5,0x48,0x4D
    f = build_write(1, 0x0103, 1000)
    assert f == [0x01, 0x06, 0x01, 0x03, 0x03, 0xE8, 0x78, 0x88], f
    assert f[-4:] == [0x03, 0xE8, 0x78, 0x88], f    # ...0x03,0xE8,0x78,0x88
    assert build_read(1, 0x0201) == [0x01, 0x03, 0x02, 0x01, 0x00, 0x01, 0xD4, 0x72]
    # 其余 docs §4 表格帧全量比对
    assert build_write(1, 0x0101, 100) == [0x01, 0x06, 0x01, 0x01, 0x00, 0x64, 0xD8, 0x1D]
    assert build_write(1, 0x0101, 30) == [0x01, 0x06, 0x01, 0x01, 0x00, 0x1E, 0x59, 0xFE]
    assert build_write(1, 0x0104, 100) == [0x01, 0x06, 0x01, 0x04, 0x00, 0x64, 0xC8, 0x1C]
    assert build_write(1, 0x0104, 50) == [0x01, 0x06, 0x01, 0x04, 0x00, 0x32, 0x48, 0x22]
    assert build_write(1, 0x0103, 0) == [0x01, 0x06, 0x01, 0x03, 0x00, 0x00, 0x78, 0x36]
    assert build_write(1, 0x0103, 500) == [0x01, 0x06, 0x01, 0x03, 0x01, 0xF4, 0x78, 0x21]
    assert build_read(1, 0x0200) == [0x01, 0x03, 0x02, 0x00, 0x00, 0x01, 0x85, 0xB2]
    assert build_read(1, 0x0202) == [0x01, 0x03, 0x02, 0x02, 0x00, 0x01, 0x24, 0x72]

    # 2) parse_read_response：自造合法读响应（夹住物体=2 @ 0x0201）
    resp = _with_crc([0x01, 0x03, 0x02, 0x00, 0x02])
    assert parse_read_response(resp) == 2, parse_read_response(resp)
    assert parse_read_response(resp, slave=1) == 2
    # 坏 CRC → ValueError
    bad = resp[:-1] + [resp[-1] ^ 0xFF]
    try:
        parse_read_response(bad)
        raise AssertionError("坏 CRC 未报错")
    except ValueError:
        pass
    # 从站不符 → ValueError
    try:
        parse_read_response(resp, slave=2)
        raise AssertionError("从站不符未报错")
    except ValueError:
        pass
    # thrift 有符号字节（-44 == 0xD4）也能被 & 0xFF 正确解析
    signed_resp = [(x - 256 if x > 127 else x) for x in resp]
    assert parse_read_response(signed_resp) == 2

    # 3) DHGripper 经 mock transport 往返（离线验证拼帧/解析全链路）
    class _Mock:
        def __init__(self):
            self.last = None
            self.reg = {REG_INIT_STATE: 1, REG_GRIP_STATE: 2, REG_POS_FB: 500}

        def w(self, data):
            self.last = list(data)
            return True

        def r(self, head, length):
            addr = (self.last[2] << 8) | self.last[3]   # 最近读帧的地址（build_read 索引 2,3）
            val = self.reg.get(addr, 0)
            return _with_crc([1, _FUNC_READ, 0x02, (val >> 8) & 0xFF, val & 0xFF])

    m = _Mock()
    g = DHGripper(m.w, m.r, slave=1, settle_s=0.0)
    assert g.read_init_state() == 1
    assert g.read_grip_state() == 2
    assert g.read_position() == 500
    g.grip(force=50, speed=50, pos=0)
    assert m.last == build_write(1, REG_POSITION, 0), m.last
    g.release()
    assert m.last == build_write(1, REG_POSITION, 1000), m.last
    g.set_force(5)     # 钳制到 20
    assert m.last == build_write(1, REG_FORCE, 20), m.last

    print("PASS")
