"""SCARA 送检流水线 · 硬件依赖抽象层。

三个 Protocol 把「运动 / 取帧 / 吸放」从具体硬件解耦：
- 生产：薄适配器包已构造好的 ScaraController / ScaraCameraThread；
- 测试：手写 Fake，无需真机、Qt、串口或相机。

设计要点（详见 docs/scara_pipeline_architecture.md）：
- 视觉伺服需要「步进并等到位」的**同步**语义，而 ScaraController.cmd_* 是异步 fire。
  故 ScaraMotionAdapter 走 controller._send（serve 单连接、同步读到 <<END>>）下发，
  再轮询 read_all_sync 的 pose/joints 稳定判到位。安全门（使能/报警）由 is_ready()
  与伺服循环入口负责（_send 本身不过 motion_guard）。
- 本模块**零硬件 import**：生产适配器靠依赖注入拿到硬件对象（鸭子调用其方法），
  cv2 仅在 CaptureFrameGrabber.grab 内懒 import。因此 Fake/Protocol 可在最小环境
  导入与单测（不拉 PyQt6/cv2/pyserial）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

Pose = List[float]   # [x, y, z, rx, ry, rz]，单位 mm / 度（SCARA World 系）
Joints = List[float]  # [j1, j2, j3(mm), j4]

# World 笛卡尔轴 → snrobot cartstep 轴码（与 scara_controller.CART_AXIS 一致）
_CART_CODE = {"X": 7, "Y": 8, "Z": 9}

JOINT_EPS = 0.05     # 单轴目标差小于此值不下发（与 SettleParams.settle_eps 同量级）


BIG_J4_DEG = 90.0
"""超过这个角度的 J4 旋转视为「大转」，必须在平面姿态**摆动之前**完成。

依据（2026-07-28 预演实测）：新放样位的 J4=−242.96°，而右盘取片位的 J4 在
−114.9~+29.5，两地相差 167°。若按「先摆平面、最后转 J4」的默认顺序，这 167° 会发生在
**腕部已经到达目标点之后** —— 也就是吊着硅片、在华夫盒正上方 17mm 处转一大圈，
等于拿片在盒口扫。把大转提到平面轴之前，它就发生在两地之间的过渡姿态上（实测那里
腕部在 (248,49) 附近，空旷）。
"""


def plan_joint_order(cur: Joints, target: Joints,
                     j2_min: Optional[float] = None,
                     eps: float = JOINT_EPS,
                     big_j4_deg: float = BIG_J4_DEG) -> List[Tuple[int, float]]:
    """把「回到这 4 个绝对关节值」拆成**安全轴序**的单轴动作 [(joint 1-4, delta), ...]。

    纯函数、零硬件依赖，可直接单测 —— 轴序是这条流水线里唯一还成立的护栏所在，
    必须能在没有真机的情况下被断言。

    两条轴序规则，都有实测依据：

    1. **Z 先抬后降**：目标 J3 更高 → 先抬 J3 再摆平面；更低 → 先摆平面、最后下刀。
       即「横移永远在高处走」。反例有实测：按右盘那段 −22.948 的高度横移进显微镜区，
       吸盘会比盒面还低 7.2mm，进去就撞（handoff §8.2）。

    2. **动 J1 前 J2 必须 ≥ `j2_min`**（`j1_requires_j2_clear`）。当前 J2 不够就先把 J2
       抬上去，动完 J1 再把 J2 落到目标值。这是两条静态禁区模型被 8 个实测反证推翻后
       **唯一还站得住的护栏**，而在此之前代码里没有任何一处执行它 —— `goto_preset`
       原本是 `for j in range(4)` 先动 J1，从 36 格里的 34 格出发都违规。

       `j2_min=None`（规则未标定）→ 不插入该动作。这是刻意的 fail-safe 方向：
       规则整个不存在说明还没标定，不能因为"没数据"就把机器锁死（与 `Waypoints.j1_move_allowed`
       同向）。阈值坏掉的情况由调用方用 `j1_move_allowed()` 拦，不在这里判。

    ⚠ 本函数**不**保证 Z 已在安全高度。带片低位横移会刮盘，调用方必须先抬到 z_safe ——
      这正是 `build_pick_from_right` 里 LIFT 那一步的意义。
    """
    cur = [float(x) for x in cur]
    tgt = [float(x) for x in target]
    out: List[Tuple[int, float]] = []

    def move(joint: int, value: float) -> None:
        d = value - cur[joint - 1]
        if abs(d) >= eps:
            out.append((joint, round(d, 4)))
            cur[joint - 1] = value

    if tgt[2] - cur[2] > eps:            # 规则1：要往上 → 先抬 Z
        move(3, tgt[2])
    big_j4 = abs(tgt[3] - cur[3]) >= big_j4_deg
    if abs(tgt[0] - cur[0]) >= eps and j2_min is not None and cur[1] < j2_min - eps:
        move(2, j2_min)                  # 规则2：动 J1 前先把 J2 收到位
    if big_j4:
        # 规则3：J4 大转必须在平面姿态**摆到目标之前**完成。否则这一大圈会发生在
        # 腕部已经到位之后 —— 即吊着片在目标点正上方原地扫一圈（放样位实测 167°）。
        # 此时臂正处于 J2 收起的过渡姿态，腕部离两端工位都远，是最空旷的时刻。
        move(4, tgt[3])
    move(1, tgt[0])
    move(2, tgt[1])
    move(4, tgt[3])                      # 小转（或大转已完成时的空操作）
    move(3, tgt[2])                      # 规则1：要往下 → 最后下刀（已到位则是空操作）
    return out


# ====================================================================== #
#  Protocol：三个硬件契约
# ====================================================================== #
@runtime_checkable
class MotionBackend(Protocol):
    """运动后端：提供同步「读位姿 / 步进到位 / 去预设点 / 就绪判断」。"""

    def get_pose(self) -> Pose: ...
    def get_joints(self) -> Joints: ...
    def step_cart(self, axis: str, mm: float) -> bool: ...   # World 单轴相对步进，阻塞到位；返回是否到位
    def step_joint(self, joint: int, delta: float) -> bool: ...  # joint 1-4
    def goto_joints(self, target: Joints) -> bool: ...  # ★关节回放：回到这 4 个绝对关节值（按 plan_joint_order 的安全轴序）
    def goto_preset(self, name: str) -> bool: ...
    def is_ready(self) -> bool: ...   # 已连接 + 使能 + 无报警


@runtime_checkable
class FrameGrabber(Protocol):
    """取帧后端：同步返回一帧 BGR 图（np.ndarray, HxWx3）。"""

    def grab(self) -> Any: ...   # -> np.ndarray(BGR)


@runtime_checkable
class SuctionBackend(Protocol):
    """吸放后端：建真空 / 破真空 / 等待稳定。"""

    def on(self) -> None: ...
    def off(self) -> None: ...
    def settle_wait(self, seconds: float) -> None: ...


@runtime_checkable
class MicroscopeBackend(Protocol):
    """自动显微镜后端：XY 台去装样位（带回读校验）+ 触发全自动扫描并等完成。"""

    def stage_goto(self, x_mm: float, y_mm: float, tolerance_mm: float) -> Tuple[bool, str]: ...
    def scan_once(self, timeout_s: float, params: Optional[dict] = None) -> Tuple[bool, str]: ...


# ====================================================================== #
#  测试用 Fake（纯 Python，无任何硬件依赖）
# ====================================================================== #
class FakeMotion:
    """假运动后端：内部维护 pose/joints，step_* 直接累加更新，记录调用序列供断言。

    - `step_fail=True` 让所有步进返回 False（模拟到位失败/护栏拦截）。
    - `ready=False` 模拟未使能/未连接。
    - presets: {name: joints4}，goto_preset 命中则更新 joints。
    """

    def __init__(self, pose: Optional[Pose] = None, joints: Optional[Joints] = None,
                 ready: bool = True, step_fail: bool = False,
                 presets: Optional[Dict[str, Joints]] = None,
                 j2_min_for_j1_move: Optional[float] = None):
        self.pose: Pose = list(pose) if pose is not None else [0.0] * 6
        self.joints: Joints = list(joints) if joints is not None else [0.0] * 4
        self._ready = bool(ready)
        self._step_fail = bool(step_fail)
        self.presets: Dict[str, Joints] = dict(presets or {})
        # 与生产适配器用**同一个** plan_joint_order，Fake 才能忠实复现轴序供单测断言。
        self.j2_min = j2_min_for_j1_move
        self.calls: List[tuple] = []

    def get_pose(self) -> Pose:
        return list(self.pose)

    def get_joints(self) -> Joints:
        return list(self.joints)

    def step_cart(self, axis: str, mm: float) -> bool:
        self.calls.append(("step_cart", axis.upper(), float(mm)))
        if self._step_fail:
            return False
        self.pose[{"X": 0, "Y": 1, "Z": 2}[axis.upper()]] += float(mm)
        return True

    def step_joint(self, joint: int, delta: float) -> bool:
        self.calls.append(("step_joint", int(joint), float(delta)))
        if self._step_fail:
            return False
        self.joints[int(joint) - 1] += float(delta)
        if int(joint) == 3:
            # ★ J3 就是直动 Z 轴：动它必须同步 pose[2]，否则 Fake 就不忠实了。
            #   实测依据：36 格 |joints[2] − pose[2]| 全部 < 0.001（backends.py 的调用方
            #   _move_z 正是读 pose[2] 算增量）。不同步会让 dry-run 演出一个真机上
            #   根本不会发生的"重复下刀"，而那样的 dry-run 比没有更糟——它会骗人。
            self.pose[2] = self.joints[2]
        return True

    def goto_joints(self, target: Joints) -> bool:
        self.calls.append(("goto_joints", [round(float(x), 4) for x in target]))
        if self._step_fail:
            return False
        for j, d in plan_joint_order(self.joints, target, self.j2_min):
            if not self.step_joint(j, d):     # 走 step_joint：调用序列里能看到真实轴序
                return False
        return True

    def goto_preset(self, name: str) -> bool:
        # 未知预设**返回 True 不动**是 Fake 的夹具便利（多数单测不关心 presets，没配就是空 dict）。
        # 生产的 ScaraMotionAdapter.goto_preset 对未知预设返回 False —— 别把两者的宽严当成同一件事。
        self.calls.append(("goto_preset", name))
        if self._step_fail:
            return False
        if name not in self.presets:
            return True
        return self.goto_joints(self.presets[name])

    def is_ready(self) -> bool:
        return self._ready


class FakeMicroscope:
    """假显微镜：可注入「到位误差」与「扫描结果」，用来在无硬件下测 fail-closed 分支。

    `pos_error_mm` 模拟 move_absolute 后实际停在哪（§8.3 的核心风险：
    move_absolute 是 fire-and-forget 且位置缓存不回读，返回 True 只代表指令发出）。
    """

    def __init__(self, pos_error_mm: float = 0.0, scan_ok: bool = True,
                 scan_reason: str = "", move_ok: bool = True,
                 objective: int = 0, objective_stuck: bool = False,
                 focus_mm: float = 5.0, focus_limits: Tuple[float, float] = (5.0, 15.0),
                 focus_error_mm: float = 0.0, focus_move_ok: bool = True,
                 af_ok: bool = True, af_reason: str = "", af_z_mm: Optional[float] = None):
        self.pos_error = float(pos_error_mm)
        self.scan_ok = bool(scan_ok)
        self.scan_reason = scan_reason
        self.move_ok = bool(move_ok)
        self.objective = int(objective)
        self.objective_stuck = bool(objective_stuck)
        self.focus_mm = float(focus_mm)
        self.focus_limits = tuple(focus_limits)
        self.focus_error = float(focus_error_mm)
        self.focus_move_ok = bool(focus_move_ok)
        self.af_ok = bool(af_ok)
        self.af_reason = af_reason
        self.af_z_mm = af_z_mm
        self.pos: Tuple[float, float] = (0.0, 0.0)
        self.calls: List[Any] = []
        self.scan_params: List[Optional[dict]] = []
        """每次 scan_once 收到的参数。★单测必须断言它不是 {} / None ——

        空参数会让真机退化成「在当前位置原地拍 3×3 共 9 张同一画面、不存盘、返回 True」，
        而 scan_once 照样返回成功。这个 Fake 若只记调用次数、不记参数，
        就完全测不出那种"扫了个寂寞"的故障。
        """

    def stage_goto(self, x_mm: float, y_mm: float, tolerance_mm: float) -> Tuple[bool, str]:
        self.calls.append(("stage_goto", round(float(x_mm), 4), round(float(y_mm), 4)))
        if not self.move_ok:
            return False, "move_absolute_rejected"
        self.pos = (float(x_mm) + self.pos_error, float(y_mm))
        err = abs(self.pos[0] - x_mm)
        if err > tolerance_mm:
            return False, f"stage_readback_off:{err:.4f}mm>{tolerance_mm}mm"
        return True, ""

    def scan_once(self, timeout_s: float, params: Optional[dict] = None) -> Tuple[bool, str]:
        self.calls.append(("scan_once", float(timeout_s)))
        self.scan_params.append(dict(params) if isinstance(params, dict) else params)
        return (True, "") if self.scan_ok else (False, self.scan_reason or "scan_failed")

    # ---- 物镜 / 对焦（2026-07-29）------------------------------------------ #
    #  可注入的失败模式一一对应真机上"不报错的错"：
    #    objective_stuck  → 转塔堵转，命令收下了但位置不变（2026-07-25 的 WS 故障）
    #    focus_error_mm   → 焦点被限位截断，软件却认为自己到位（同 move_absolute 的老问题）
    #    af_z_mm          → WDI 对到了别的面上：in_focus/in_range 全真，只有 Z 不对

    def objective_goto(self, index: int, timeout_s: float = 30.0,
                       poll_s: float = 0.5) -> Tuple[bool, str]:
        self.calls.append(("objective_goto", int(index)))
        if self.objective_stuck:
            return False, (f"objective_readback_off:{timeout_s:.0f}s 内没切到位，"
                           f"目标 {index} 实读 {self.objective!r}")
        self.objective = int(index)
        return True, ""

    def focus_goto(self, z_mm: float, tolerance_mm: float) -> Tuple[bool, str]:
        self.calls.append(("focus_goto", round(float(z_mm), 4)))
        if not self.focus_move_ok:
            return False, "move_focus_returned_false"
        self.focus_mm = float(z_mm) + self.focus_error
        err = abs(self.focus_mm - float(z_mm))
        if err > tolerance_mm:
            return False, (f"focus_readback_off:第2次 实到 {self.focus_mm:.4f} "
                           f"目标 {float(z_mm):.4f} 偏差 {err:.4f}mm > 容差 {tolerance_mm}mm")
        return True, ""

    def autofocus_refine(self) -> Tuple[bool, str, Optional[float]]:
        self.calls.append(("autofocus_refine",))
        if not self.af_ok:
            return False, self.af_reason or "autofocus_returned_false", self.focus_mm
        if self.af_z_mm is not None:
            self.focus_mm = float(self.af_z_mm)
        return True, "", self.focus_mm

    def call(self, method: str, params: Optional[list] = None) -> Any:
        """只支持执行器真正会用到的只读查询；其余一律抛错，免得 Fake 悄悄替真机圆场。"""
        self.calls.append(("call", method))
        if method == "get_focus_limits":
            return list(self.focus_limits)
        if method == "get_focus_position":
            return self.focus_mm
        if method == "get_objective":
            return self.objective
        raise RuntimeError(f"FakeMicroscope 未实现 RPC 方法: {method}")


class FakeGrabber:
    """假相机：返回固定 `image`，或每次调 `image_fn()` 动态生成（伺服测试按 motion 状态出图）。"""

    def __init__(self, image: Any = None, image_fn: Optional[Callable[[], Any]] = None):
        self._image = image
        self._image_fn = image_fn
        self.grabs = 0

    def grab(self) -> Any:
        self.grabs += 1
        if self._image_fn is not None:
            return self._image_fn()
        if self._image is None:
            raise RuntimeError("FakeGrabber 未提供 image/image_fn")
        return self._image


class FakeSuction:
    """假吸放：记录 on/off/wait 调用序列与当前持有态。"""

    def __init__(self):
        self.calls: List[Any] = []
        self.holding = False

    def on(self) -> None:
        self.calls.append("on")
        self.holding = True

    def off(self) -> None:
        self.calls.append("off")
        self.holding = False

    def settle_wait(self, seconds: float) -> None:
        self.calls.append(("wait", float(seconds)))


# ====================================================================== #
#  生产适配器（薄包现有硬件对象；靠依赖注入，本模块不 import 它们）
# ====================================================================== #
@dataclass
class SettleParams:
    """「步进后轮询到位」参数（实机可调）。"""
    poll_interval_s: float = 0.1
    settle_eps: float = 0.05     # 相邻两次读数最大分量差 < eps 视为不动（mm/度）
    settle_count: int = 3        # 连续 settle_count 次不动 → 判到位
    timeout_s: float = 15.0
    start_wait_s: float = 0.2    # 发令后先等运动启动，避免"还没动就误判已稳定"


class ScaraMotionAdapter:
    """把 ScaraController 适配成同步 MotionBackend。

    step_cart/step_joint：走 `controller._send`（serve 单连接、同步）下发，再轮询
    `read_all_sync` 的 pose/joints 稳定判到位（snrobot serve 命令返回时运动未必完成，
    pose 稳定兜底最稳）。goto_preset 逐轴同步 move1 到预设关节值。

    注：`_send` 不过 controller 的 motion_guard（使能/报警检查），故运动前必须由调用方
    先 `is_ready()` 把关（伺服循环入口已做）。sleep_fn/clock 可注入以便单测到位逻辑。
    """

    def __init__(self, controller: Any, settle: Optional[SettleParams] = None,
                 sleep_fn: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 j2_min_for_j1_move: Optional[float] = None):
        self._c = controller
        self._st = settle or SettleParams()
        self._sleep = sleep_fn
        self._clock = clock
        # 动 J1 前 J2 要收到的最小角度。生产上传 `load_waypoints().j2_min_for_j1_move`。
        # None = 规则未标定 → 不插入收 J2 的动作（fail-safe 方向同 Waypoints.j1_move_allowed）。
        self.j2_min = j2_min_for_j1_move

    def _read(self) -> Optional[dict]:
        return self._c.read_all_sync()

    def get_pose(self) -> Pose:
        st = self._read()
        return list(st["pose"]) if st else [0.0] * 6

    def get_joints(self) -> Joints:
        st = self._read()
        return list(st["joints"]) if st else [0.0] * 4

    def is_ready(self) -> bool:
        if not self._c.is_connected():
            return False
        st = self._read()
        return bool(st) and bool(st.get("effectively_enabled",
            int(st.get("enable", 0)) == 1 and int(st.get("warn", 0)) == 0))

    def _wait_settle(self, key: str, dim: int) -> bool:
        """轮询 read_all_sync[key][:dim]，连续 settle_count 次不动或超时。"""
        self._sleep(self._st.start_wait_s)
        t0 = self._clock()
        prev: Optional[List[float]] = None
        stable = 0
        while self._clock() - t0 < self._st.timeout_s:
            st = self._read()
            if st:
                cur = [float(x) for x in st[key][:dim]]
                if prev is not None:
                    d = max(abs(a - b) for a, b in zip(cur, prev))
                    stable = stable + 1 if d < self._st.settle_eps else 0
                    if stable >= self._st.settle_count:
                        return True
                prev = cur
            self._sleep(self._st.poll_interval_s)
        return False

    def step_cart(self, axis: str, mm: float) -> bool:
        code = _CART_CODE[axis.upper()]
        self._c._send(f"cartstep {code} {mm:g}")
        return self._wait_settle("pose", 6)

    def step_joint(self, joint: int, delta: float) -> bool:
        self._c._send(f"move1 {int(joint)} {delta:g} 1")
        return self._wait_settle("joints", 4)

    def goto_joints(self, target: Joints) -> bool:
        """关节回放：按 `plan_joint_order` 的**安全轴序**逐轴 move1 到这 4 个绝对关节值。

        ★ 这里替换掉了原 `goto_preset` 的 `for j in range(4)` 顺序 —— 那个先动 J1，
          从 36 格里的 34 格出发时 J2 都不够高，**正面违反 j1_requires_j2_clear**，
          而那是两条静态禁区模型作废后唯一还成立的护栏。轴序不是风格问题，是安全问题。

        每一轴都必须真正到位（`_wait_settle` 返回 True）才继续；任何一轴没到位立即
        返回 False —— 半途而废的姿态比原地不动危险得多，让上层去回滚停手。
        """
        st = self._read()
        if not st:
            return False
        plan = plan_joint_order(st["joints"], target, self.j2_min)
        for joint, delta in plan:
            self._c._send(f"move1 {int(joint)} {delta:g} 1")
            if not self._wait_settle("joints", 4):
                return False
        return True

    def goto_preset(self, name: str) -> bool:
        presets = getattr(self._c, "_presets", {}) or {}
        if name not in presets:
            return False
        return self.goto_joints(presets[name])


class ThreadFrameGrabber:
    """从运行中的 ScaraCameraThread 读最近一帧（UI 相机线程在跑时用）。"""

    def __init__(self, camera_thread: Any):
        self._t = camera_thread

    def grab(self) -> Any:
        f = getattr(self._t, "_last_frame", None)
        if f is None:
            raise RuntimeError("相机尚无帧（ScaraCameraThread 未启动或未取到帧）")
        return f.copy()


class CaptureFrameGrabber:
    """独立抓取一个逻辑相机；勿与UI线程同时占用同一物理设备。"""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720,
                 warmup_frames: int = 10, warmup_sleep_s: float = 0.05,
                 source_resolver: Optional[Callable[[int], Any]] = None):
        self._index = index
        self._source_resolver = source_resolver
        self._w, self._h = width, height
        self._warmup = int(warmup_frames)
        self._warmup_sleep = float(warmup_sleep_s)
        self._cap = None
        self._resolved_camera = None

    @property
    def source_index(self) -> int:
        return int(self._index)

    @property
    def physical_source_index(self) -> Optional[int]:
        return (
            None
            if self._resolved_camera is None
            else int(self._resolved_camera.physical_index)
        )

    def _ensure(self) -> None:
        if self._cap is None:
            import cv2  # 懒 import：只有真抓帧才拉 cv2
            if self._source_resolver is None:
                from scara.config.camera_config import resolve_camera_source

                resolver = resolve_camera_source
            else:
                resolver = self._source_resolver
            resolved = resolver(int(self._index))
            self._resolved_camera = resolved
            cap = cv2.VideoCapture(resolved.physical_index, resolved.backend)
            if not cap.isOpened():
                raise RuntimeError(
                    f"无法打开逻辑相机{self._index}（物理Index "
                    f"{resolved.physical_index}）"
                )
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._h)
            # ★ 必须预热：USB 相机刚打开的前几帧是黑的/半截的（自动曝光还没收敛）。
            #   2026-07-28 实测：不预热直接读第一帧 → 硅片检出 0 片，起飞前检查被拦。
            #   本仓库所有抓图脚本都预热 8~12 帧，唯独这个生产类漏了。
            for _ in range(self._warmup):
                cap.read()
                time.sleep(self._warmup_sleep)
            self._cap = cap

    def grab(self) -> Any:
        self._ensure()
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("相机取帧失败")
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class RpcMicroscope:
    """经 app 的 JSON-RPC(5555) 驱动自动显微镜。行分隔 JSON-RPC 2.0，纯标准库。

    ★ `stage_goto` 把 handoff §8.3 那条纪律**写进代码**，而不是留在文档里当口号：

        move_absolute(x,y) → wait_stage_idle() → get_position() 回读校验 ≤ tolerance

    为什么这条不可省（§4.4 已核实）：绝对移动路径比相对路径**还少一层重试**，
    且 `microscope_controller.py:317-319` 把目标值**直接当成当前位置写进缓存、不回读** ——
    命令刚发出、甚至被限位截断，软件就认为自己到位了。而装样位恰好落在 XY 台行程的
    边界上（X 距上限 1.9mm、Y 精确顶格），被截断的概率不是理论风险。
    超差重试一次；二次仍超差 → 返回失败，**绝不放 SCARA 进去**。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5555,
                 timeout_s: float = 20.0,
                 sleep_fn: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self._host, self._port, self._timeout = host, int(port), float(timeout_s)
        self._sleep, self._clock = sleep_fn, clock
        self._id = 0

    def call(self, method: str, params: Optional[list] = None) -> Any:
        import json
        import socket
        self._id += 1
        req = json.dumps({"jsonrpc": "2.0", "method": method,
                          "params": params or [], "id": self._id}) + "\n"
        with socket.create_connection((self._host, self._port), self._timeout) as s:
            s.settimeout(self._timeout)
            s.sendall(req.encode("utf-8"))
            buf = b""
            while b"\n" not in buf:
                chunk = s.recv(65536)
                if not chunk:
                    raise RuntimeError(f"RPC 连接被对端关闭（method={method}）")
                buf += chunk
        resp = json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))
        if "error" in resp and resp["error"] is not None:
            raise RuntimeError(f"RPC {method} 失败: {resp['error']}")
        return resp.get("result")

    def stage_goto(self, x_mm: float, y_mm: float, tolerance_mm: float) -> Tuple[bool, str]:
        last = ""
        for attempt in (1, 2):                      # 超差重试一次，二次仍超差就停手
            if not self.call("move_absolute", [float(x_mm), float(y_mm)]):
                last = "move_absolute_returned_false"
                continue
            self.call("wait_stage_idle")
            pos = self.call("get_position")
            if not (isinstance(pos, (list, tuple)) and len(pos) == 2):
                last = f"get_position_bad:{pos!r}"
                continue
            dx, dy = float(pos[0]) - float(x_mm), float(pos[1]) - float(y_mm)
            err = max(abs(dx), abs(dy))
            if err <= tolerance_mm:
                return True, ""
            last = (f"stage_readback_off:第{attempt}次 实到({pos[0]:.4f},{pos[1]:.4f}) "
                    f"目标({x_mm:.4f},{y_mm:.4f}) 偏差{err:.4f}mm > 容差{tolerance_mm}mm")
        return False, last

    def objective_goto(self, index: int, timeout_s: float = 30.0,
                       poll_s: float = 0.5) -> Tuple[bool, str]:
        """把物镜转塔切到 `index`，**轮询回读确认到位**才算成功。

        为什么不能只信 `set_objective` 的返回值（`microscope_controller.py:650`）：
          · 它内部注明"设备可能仍在动，不立即回读"，成功返回只代表命令发出去了；
          · 当 `_current_objective_index == index` 时它直接 return True，比的是**缓存**——
            缓存与实物不符时（有人手转过、上次切镜半途失败），会静默跳过；
          · 换镜内部会 suspend 焦点限位并**驱动焦点轴**，这是会动的动作，不是查询。
        `get_objective()` 会先 `_sync_optical_state_from_device()`，所以拿它做判据。

        ★ 2026-07-25 的教训（转塔堵转 WS 后固件持续报 busy、重启软件清不掉）已经在
          `set_objective` 内部做了 stop+清标志的恢复重试；这里只负责"到位没有"，
          不再叠一层恢复逻辑 —— 两处都试着恢复只会让现场状态更难推断。
        """
        try:
            cur = self.call("get_objective")
        except Exception as exc:                                  # noqa: BLE001
            return False, f"objective_read_failed:{type(exc).__name__}:{exc}"
        if isinstance(cur, int) and int(cur) == int(index):
            return True, ""                                       # 实物已在位（读的是设备不是缓存）
        try:
            self.call("set_objective", [int(index)])
        except Exception as exc:                                  # noqa: BLE001
            return False, f"objective_set_failed:{type(exc).__name__}:{exc}"
        t0 = self._clock()
        last = None
        while self._clock() - t0 < timeout_s:
            self._sleep(poll_s)
            try:
                last = self.call("get_objective")
            except Exception as exc:                              # noqa: BLE001
                return False, f"objective_read_failed:{type(exc).__name__}:{exc}"
            if isinstance(last, int) and int(last) == int(index):
                return True, ""
        return False, (f"objective_readback_off:{timeout_s:.0f}s 内没切到位，"
                       f"目标 {index} 实读 {last!r}。转塔可能堵转（2026-07-25 有过 WS 堵转后"
                       f"固件持续 busy 的先例），去 app 看物镜面板与告警")

    def focus_goto(self, z_mm: float, tolerance_mm: float) -> Tuple[bool, str]:
        """焦点轴到绝对 Z，**回读校验**。理由与 `stage_goto` 完全同源。

        `move_focus_absolute` 同样走"目标值直接写缓存、不回读"的路径，而起始焦点这一步
        往往是从装样位那个贴着限位的 5.000mm 出发的大位移 —— 被限位截断时软件会认为
        自己到位了，然后整轮扫描在失焦状态下跑完，每张图都拍到了、都存下了、都不报错。
        """
        last = ""
        for attempt in (1, 2):                      # 超差重试一次，二次仍超差就停手
            try:
                if not self.call("move_focus_absolute", [float(z_mm)]):
                    last = "move_focus_returned_false"
                    continue
                self.call("wait_focus_idle")
                z = self.call("get_focus_position")
            except Exception as exc:                              # noqa: BLE001
                return False, f"focus_move_failed:{type(exc).__name__}:{exc}"
            if not isinstance(z, (int, float)):
                last = f"get_focus_position_bad:{z!r}"
                continue
            err = abs(float(z) - float(z_mm))
            if err <= tolerance_mm:
                return True, ""
            last = (f"focus_readback_off:第{attempt}次 实到 {float(z):.4f} 目标 {float(z_mm):.4f} "
                    f"偏差 {err:.4f}mm > 容差 {tolerance_mm}mm")
        return False, last

    def autofocus_refine(self) -> Tuple[bool, str, Optional[float]]:
        """跑一次 WDI 自动对焦精修，返回 (通过?, 原因, 精修后的 Z)。

        判据是**两条一起**：调用要返回成功，且 `get_autofocus_status` 的
        `in_focus` / `in_range` 都得为真。只看返回值不够 —— 2026-07-26 那次
        "自动对焦不正常"就是 WDI 网络超时被静默吞掉，连接照报成功。
        """
        try:
            ok = self.call("autofocus_once")
        except Exception as exc:                                  # noqa: BLE001
            return False, f"autofocus_call_failed:{type(exc).__name__}:{exc}", None
        try:
            z = self.call("get_focus_position")
            z = float(z) if isinstance(z, (int, float)) else None
        except Exception:                                         # noqa: BLE001
            z = None
        if not ok:
            return False, "autofocus_returned_false", z
        try:
            st = self.call("get_autofocus_status") or {}
        except Exception as exc:                                  # noqa: BLE001
            return False, f"autofocus_status_unreadable:{type(exc).__name__}:{exc}", z
        if not isinstance(st, dict):
            return False, f"autofocus_status_bad:{st!r}", z
        if not st.get("in_focus") or not st.get("in_range"):
            return False, (f"autofocus_not_locked:in_focus={st.get('in_focus')} "
                           f"in_range={st.get('in_range')}"), z
        return True, "", z

    def scan_once(self, timeout_s: float, params: Optional[dict] = None) -> Tuple[bool, str]:
        """触发一次全自动扫描并轮询到完成。params=None 用显微镜侧当前配置。"""
        if not self.call("start_scan", [params or {}]):
            return False, "start_scan_returned_false"
        t0 = self._clock()
        while self._clock() - t0 < timeout_s:
            self._sleep(1.0)
            if self.call("is_scan_idle"):
                st = self.call("get_scan_status") or {}
                err = st.get("error")
                return (False, f"scan_error:{err}") if err else (True, "")
        return False, f"scan_timeout:{timeout_s}s"
