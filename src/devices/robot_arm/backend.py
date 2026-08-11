"""机械臂后端抽象层。

定义统一的 RobotArmBackendBase 接口和 RobotArmStatus 状态快照，UI 只依赖本抽象，
不直接接触 Thrift / 仿真细节。两个实现：
  - ThriftRobotArmBackend：真机（thrift_backend.py，唯一碰 Thrift 协议的地方）
  - SimRobotArmBackend：仿真（sim_backend.py，无硬件，本地积分运动）

设计要点：
  - 所有方法非阻塞或快速返回；耗时操作（connect / 长距离运动）由 UI 放到 QThread。
  - get_status() 返回一份不可变快照，UI 一次取帧喂所有消费者（状态表 + 3D）。
  - 运动方法均经 Safety 检查（限位/钳速/工作空间）后才下发。
"""
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


# 机器人状态枚举（与 duco_rpc 一致）
ROBOT_STATE_ENABLED = 6
SAFETY_RUN = 5


@dataclass(frozen=True)
class RobotArmStatus:
    """机械臂状态快照（一次轮询的结果）。所有角度 rad，长度 m。"""
    connected: bool = False
    joints: List[float] = field(default_factory=lambda: [0.0] * 6)   # 6 关节角 rad
    tcp: List[float] = field(default_factory=lambda: [0.0] * 6)      # X,Y,Z,Rx,Ry,Rz (m,rad)
    op_mode: int = 0          # 0手动 1自动 2远程
    robot_state: int = 0      # 4下电 5上电未使能 6使能
    prog_state: int = 0       # 0停 5任务运行
    safety: int = 0           # 5=RUN
    collision: int = 0
    collision_axis: int = 0
    alarm: int = 0
    powered: bool = False     # robot_state >= 5
    enabled: bool = False     # robot_state == 6
    error: str = ""           # 最近错误信息（连接失败/运动异常等），空=正常

    @property
    def joints_deg(self) -> List[float]:
        return [math.degrees(q) for q in self.joints]

    @property
    def is_ready(self) -> bool:
        """可运动：已使能 + 安全态 RUN。

        注意：不检查 alarm。根据项目实践与文档，alarm 是历史残留值
        （如旧碰撞记录），robot_state=6 + safety=5 即可接受运动指令。
        吸盘/夹爪等操作同理。"""
        return self.enabled and self.safety == SAFETY_RUN


class RobotArmBackendBase(ABC):
    """机械臂后端统一接口。UI 只依赖这些方法。"""

    # ── 连接生命周期 ────────────────────────────────────────────
    @abstractmethod
    def connect(self) -> None:
        """建立连接（真机：开 Thrift transport + 启动心跳）。阻塞，UI 放 QThread。"""

    @abstractmethod
    def disconnect(self) -> None:
        """断开连接，停心跳，释放资源。幂等。"""

    @abstractmethod
    def is_connected(self) -> bool:
        ...

    # ── 状态 ────────────────────────────────────────────────────
    @abstractmethod
    def get_status(self) -> RobotArmStatus:
        """返回当前状态快照。轮询调用，必须快速/不抛异常（异常包进 status.error）。"""

    # ── 上电 / 使能 ─────────────────────────────────────────────
    @abstractmethod
    def power_on(self) -> None:
        ...

    @abstractmethod
    def enable(self) -> None:
        ...

    @abstractmethod
    def disable(self) -> None:
        ...

    @abstractmethod
    def estop(self) -> None:
        """急停：立即停止 + disable，旁路一切运动护栏。"""

    # ── 末端数字输出（吸盘 / 工具 IO）───────────────────────────
    def set_digital_output(self, port: int, value: bool, block: bool = True) -> None:
        """设置末端工具数字输出（DO）。吸盘经此口控制真空泵继电器。
        port: tool DO 端口号；value: True=通电, False=断电。
        默认不支持（子类按需覆盖）。"""
        raise NotImplementedError("该后端不支持末端数字输出")

    def set_gripper(self, on: bool) -> None:
        """语义封装：读 gripper 配置（do_num/active_high）把 on/off 翻成 DO 电平，
        并按 settle_on_ms/settle_off_ms 等待负压建立/泄压后才返回。

        永远在 worker 线程（总指挥）调用，所以这里 sleep 不阻塞 UI。
        基于 set_digital_output，子类无需覆盖本方法。"""
        g = (getattr(self, "cfg", None) or {}).get("gripper", {})
        do_num = int(g.get("do_num", 0))
        active_high = bool(g.get("active_high", True))
        level = on if active_high else (not on)
        self.set_digital_output(do_num, level, block=True)
        settle_ms = g.get("settle_on_ms", 400) if on else g.get("settle_off_ms", 300)
        time.sleep(max(0.0, float(settle_ms)) / 1000.0)

    # ── 夹爪（大寰 PGEA-50-26-O 电动夹爪，工具端 485 Modbus 透传）──
    # 与吸盘（set_gripper/set_digital_output）并列的另一末端执行器；真机走 dh_gripper 驱动。
    @abstractmethod
    def gripper_initialize(self, full: bool = False) -> None:
        """初始化夹爪（上电首次必做：回零标定 + 等初始化成功）。真机经工具端 485 透传。
        full=False→单向初始化(0x01,快)；full=True→全行程完全标定(0xA5,更准)。"""

    @abstractmethod
    def gripper_grip(self, force: Optional[int] = None, speed: Optional[int] = None,
                     pos: int = 0) -> None:
        """夹取（合拢到 pos‰，0=全合夹紧）。force 20-100（%），speed 1-100（%），None=沿用上次设定。"""

    @abstractmethod
    def gripper_release(self, speed: Optional[int] = None) -> None:
        """释放（张开到全开 1000‰，机械张开无需破真空）。speed None=沿用上次设定。"""

    @abstractmethod
    def gripper_status(self) -> dict:
        """返回夹爪状态 dict{init_state, grip_state, position}。
        init_state 0未初始化/1成功/2初始化中；grip_state 0运动中/1到位未夹到/2夹住/3掉落；
        position 0-1000（‰）。"""

    # ── 拖动示教（牵引/零力）────────────────────────────────────
    def enter_teach_mode(self, load_mass: float = 0.0,
                         cx: float = 0.0, cy: float = 0.0, cz: float = 0.0) -> None:
        """进入拖动示教（人徒手拖动机械臂）。进入前必须先设末端负载。
        默认不支持（子类按需覆盖）。真机最高风险操作，需人在旁手扶。"""
        raise NotImplementedError("该后端不支持拖动示教")

    def exit_teach_mode(self) -> None:
        """退出拖动示教，回到正常受控状态。默认不支持。"""
        raise NotImplementedError("该后端不支持拖动示教")

    def is_teaching(self) -> bool:
        """当前是否处于拖动示教状态。默认 False。"""
        return False

    # ── 运动 ────────────────────────────────────────────────────
    @abstractmethod
    def move_joints(self, joints: List[float], speed_scale: Optional[float] = None,
                    block: bool = True) -> None:
        """关节空间运动到目标（6 个 rad）。经限位+钳速检查。block=True 轮询到位。"""

    def move_joints_seq(self, waypoints: List[List[float]], speed_scale: Optional[float] = None,
                        blend_radius: float = 0.0) -> None:
        """连续多点关节运动（一串 6-rad 航点）。默认实现=逐点 move_joints 到位（无融合、安全）。

        支持融合的后端（thrift/http→DUCO movej2 r）覆盖此方法：blend_radius>0 时中间点用融合半径
        r 过弯不停、减少物理停顿；r=0 逐点到位。融合会切内角，只应对**开阔段**用（避碰走廊切角危险）。
        """
        for q in waypoints:
            self.move_joints(q, speed_scale=speed_scale, block=True)

    def jog_joint(self, index: int, delta_rad: float, speed_scale: Optional[float] = None) -> None:
        """单关节微动：在当前位姿基础上对第 index 关节 ±delta_rad。默认实现基于 move_joints。"""
        cur = list(self.get_status().joints)
        if not (0 <= index < len(cur)):
            raise IndexError("关节索引越界: %d" % index)
        cur[index] += delta_rad
        self.move_joints(cur, speed_scale=speed_scale, block=True)

    def jog_cart(self, axis: int, delta: float, speed_scale: Optional[float] = None,
                 dry_run: bool = False):
        """笛卡尔点动：把 TCP 沿 axis 增量 delta（axis 0-2=X/Y/Z 单位 m；3-5=Rx/Ry/Rz 单位 rad），
        经机器人自带逆解(cal_ikine)解出关节 → 逆解跳变护栏 → 受监督关节运动(movej_safe)。
        dry_run=True 只算逆解+校验、不下发运动，返回目标关节等信息 dict。
        默认不支持（仅真机 thrift / 远程 http 实现；sim 无逆解）。"""
        raise NotImplementedError("该后端不支持笛卡尔点动（无逆解）")

    def goto_pose(self, x: float, y: float, z: float, yaw: float,
                  tool_offset: Optional[List[float]] = None,
                  speed_scale: Optional[float] = None, block: bool = True) -> None:
        """走到笛卡尔位姿：把工具 TCP 移到 base 系 (x, y, z)（**毫米**），姿态 = 朝下
        R_down 叠加 yaw（**弧度**，绕 base +Z 的矩阵合成，非标量加 rz）。

        用于放置：目标 XY + 台面 Z + R_down⊕yaw，把硅片按指定 yaw 摆正后放下。
        真机后端（thrift）实现：compose_place_orientation 合成姿态 → cal_ikine 逆解 →
        从上方可达点 reach_top 进位 → vcol 微步下降到位（每步 movej_safe），
        全程带安全护栏（工作空间 / Z 地板 / 关节跳变 / rpy）。默认不支持。

        tool_offset: 工具偏移向量[x,y,z,rx,ry,rz]（m,rad）；None=用标定的吸盘 CUP 偏移。
        本方法只负责运动到位，不含吸/放真空（由调用方 set_gripper / 继电器步单独控）。"""
        raise NotImplementedError("该后端不支持 goto_pose（仅真机 thrift / 远程 http 代理）")

    # ── 安全配置（运动禁区 / 工作空间）──────────────────────────
    @abstractmethod
    def set_workspace_bounds(self, x=None, y=None, z=None) -> None:
        """更新工作空间 box（base 系 m）。None 表示该轴不变。"""

    @abstractmethod
    def set_nogo_boxes(self, boxes: list) -> None:
        """设置禁区盒列表。每个盒：dict(cx,cy,cz,w,h,d) 中心+半长（m）。"""

    def check_pose_allowed(self, pose) -> tuple:
        """检查一个 TCP 位姿是否在工作空间内且不在禁区。返回 (ok, msg)。默认放行。"""
        return True, "ok"

    # ── 点位（示教 / waypoints）─────────────────────────────────
    @abstractmethod
    def record_waypoint(self, name: str) -> None:
        """记录当前位姿为命名点。"""

    @abstractmethod
    def waypoint_names(self) -> List[str]:
        ...

    @abstractmethod
    def delete_waypoint(self, name: str) -> None:
        ...

    @abstractmethod
    def goto_waypoint(self, name: str, speed_scale: Optional[float] = None,
                      block: bool = True) -> None:
        ...

    def get_waypoint_joints(self, name: str) -> List[float]:
        """取命名点的关节角（供仿真/3D 预览）。子类应覆盖。"""
        raise NotImplementedError
