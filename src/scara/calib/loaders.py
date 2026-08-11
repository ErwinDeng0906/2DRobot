"""SCARA 送检 · 标定数据加载（schema 校验 + 缺失/损坏降级默认）。

作业单 §3 产物落 src/scara/calib/*.json；实机标定前用同名 *.example.json 占位默认。
加载优先 <name>.json → <name>.example.json → 代码内默认。
fail-safe：文件缺失/损坏/字段非法一律回退默认（可选 log 记 warn），绝不抛（仿 vision_service）。
纯标准库（含手写 2×2 求逆），不依赖 numpy/cv2。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_CALIB_DIR = Path(__file__).resolve().parent

# J3 下移相机手眼 / 右盘坐标系（2026-07-31 新流程）。两者都是纯标准库，无 import 环。
from scara.calib.handeye import J3Handeye, load_j3_handeye as _he_from_file
from scara.calib.tray_grid import TrayFrame, load_tray_frame as _tf_from_file


def _inv2(m: List[List[float]]) -> Optional[List[List[float]]]:
    """2×2 逆；奇异（|det|<1e-9）或非法返回 None。"""
    try:
        a, b = float(m[0][0]), float(m[0][1])
        c, d = float(m[1][0]), float(m[1][1])
    except (TypeError, IndexError, ValueError):
        return None
    det = a * d - b * c
    if abs(det) < 1e-9:
        return None
    return [[d / det, -b / det], [-c / det, a / det]]


def _f(d: dict, k: str, default: float) -> float:
    try:
        return float(d[k])
    except (KeyError, TypeError, ValueError):
        return default


# ---------------------------------------------------------------------- #
#  标定数据类（各带 from_dict 容错 + 默认）
# ---------------------------------------------------------------------- #
@dataclass
class CamJacobian:
    """相机像素↔World XY 标定（作业单第 2 项）。

    - 增量：Δpx = J·Δmm（j_mm_to_px）。world_delta_for_px_error 用其逆把像素误差转 World。
    - 绝对：px = px_origin + J·(world − world_origin_mm)。world_to_px 正向投影末端(吸盘)像素，
      供视觉伺服算「硅片像素 − 吸盘投影」误差——eye-to-hand 闭环需要末端投影随位姿变化才能收敛。
    """
    j_mm_to_px: List[List[float]] = field(default_factory=lambda: [[10.0, 0.0], [0.0, 10.0]])
    px_origin: Tuple[float, float] = (0.0, 0.0)          # world_origin_mm 对应的像素锚点
    world_origin_mm: Tuple[float, float] = (0.0, 0.0)    # 锚点 World XY
    resid_px: float = 0.0

    @property
    def j_px_to_mm(self) -> Optional[List[List[float]]]:
        return _inv2(self.j_mm_to_px)

    def world_delta_for_px_error(self, dpx: float, dpy: float) -> Optional[Tuple[float, float]]:
        """像素误差 (dpx,dpy) → World (dx,dy) mm。奇异雅可比返回 None（上层 fail-safe 停手）。"""
        inv = self.j_px_to_mm
        if inv is None:
            return None
        return (inv[0][0] * dpx + inv[0][1] * dpy, inv[1][0] * dpx + inv[1][1] * dpy)

    def world_to_px(self, world_xy) -> Tuple[float, float]:
        """World XY → 像素（正向投影）：px = px_origin + J·(world − world_origin_mm)。"""
        j = self.j_mm_to_px
        dx = float(world_xy[0]) - self.world_origin_mm[0]
        dy = float(world_xy[1]) - self.world_origin_mm[1]
        return (self.px_origin[0] + j[0][0] * dx + j[0][1] * dy,
                self.px_origin[1] + j[1][0] * dx + j[1][1] * dy)

    @classmethod
    def from_dict(cls, d) -> "CamJacobian":
        b = cls()
        if not isinstance(d, dict):
            return b
        j = d.get("j_mm_to_px")
        if (isinstance(j, list) and len(j) == 2
                and all(isinstance(r, list) and len(r) == 2 for r in j)):
            try:
                b.j_mm_to_px = [[float(j[0][0]), float(j[0][1])],
                                [float(j[1][0]), float(j[1][1])]]
            except (TypeError, ValueError):
                pass
        for attr in ("px_origin", "world_origin_mm"):
            v = d.get(attr)
            if isinstance(v, (list, tuple)) and len(v) == 2:
                try:
                    setattr(b, attr, (float(v[0]), float(v[1])))
                except (TypeError, ValueError):
                    pass
        b.resid_px = _f(d, "resid_px", b.resid_px)
        return b


@dataclass
class TcpCalib:
    """吸盘 TCP（作业单第 1 项）：吸盘中心相对 J4 轴心的 XY 偏移 + J4 轴心像素。"""
    cup_offset_mm: Tuple[float, float] = (0.0, 0.0)
    j4_axis_px: Tuple[float, float] = (0.0, 0.0)

    @classmethod
    def from_dict(cls, d) -> "TcpCalib":
        b = cls()
        if not isinstance(d, dict):
            return b
        co = d.get("cup_offset_mm")
        if isinstance(co, (list, tuple)) and len(co) == 2:
            try:
                b.cup_offset_mm = (float(co[0]), float(co[1]))
            except (TypeError, ValueError):
                pass
        ax = d.get("j4_axis_px")
        if isinstance(ax, (list, tuple)) and len(ax) == 2:
            try:
                b.j4_axis_px = (float(ax[0]), float(ax[1]))
            except (TypeError, ValueError):
                pass
        return b


@dataclass
class Heights:
    """取放 Z 高度（作业单第 4/5 项）。单位 mm（SCARA World Z / J3）。

    ★ 右盘两个高度已改为 Optional 且默认 None（2026-07-27）。原来默认 0.0/30.0 时，
      `scara_heights.json` 根本不存在（只有 .example 占位），于是取片链路一路静默用
      「DESCEND 到 Z=0.0」跑下去 —— 而实测取片 Z 是 −62.3。那正是 TrayCells.z_pick_for
      docstring 里明令禁止的 0.0 兜底，从旁边绕过去了。
      现在右盘取片一律走 `TrayCells.z_pick_for()/z_safe_for()`（含 z_offset_mm），
      本类的两个字段只作兼容占位；None = 未标定 = 上层必须停手。
    """
    z_pick_right: Optional[float] = None
    z_safe_right: Optional[float] = None
    # TODO(阶段4 放片接入)：z_above_waffle / z_place_waffle 同样是从没标定过的占位数，
    #   0.0 在显微镜侧一样是「一个具体位置」而不是安全高度。放片改走航点表时一并改成
    #   Optional=None。当前之所以还留着数值默认，是因为放片链路会先在 GOTO_WAYPOINT
    #   （above_micro/place_micro 尚未示教）失败停住，走不到 DESCEND_PLACE。
    z_above_waffle: float = 30.0
    z_place_waffle: float = 0.0

    @classmethod
    def from_dict(cls, d) -> "Heights":
        b = cls()
        if isinstance(d, dict):
            b.z_pick_right = _opt_f(d.get("z_pick_right"))
            b.z_safe_right = _opt_f(d.get("z_safe_right"))
            b.z_above_waffle = _f(d, "z_above_waffle", b.z_above_waffle)
            b.z_place_waffle = _f(d, "z_place_waffle", b.z_place_waffle)
        return b


@dataclass
class WaffleCalib:
    """华夫盒放置（作业单第 5/6 项）：凹槽目标角 + 角度补偿常数 + 关联预设点名。"""
    target_angle_deg: float = 0.0
    angle_offset_const: float = 0.0
    place_preset: str = "place_waffle"
    above_preset: str = "above_waffle"

    @classmethod
    def from_dict(cls, d) -> "WaffleCalib":
        b = cls()
        if isinstance(d, dict):
            b.target_angle_deg = _f(d, "target_angle_deg", b.target_angle_deg)
            b.angle_offset_const = _f(d, "angle_offset_const", b.angle_offset_const)
            if isinstance(d.get("place_preset"), str):
                b.place_preset = d["place_preset"]
            if isinstance(d.get("above_preset"), str):
                b.above_preset = d["above_preset"]
        return b


@dataclass
class PumpTiming:
    """泵阀吸放时序（作业单第 8 项）。"""
    vac_on_s: float = 1.2
    vent_ms: float = 1200.0
    vent_wait_s: float = 3.0
    verify_dark_area_drop: float = 0.7   # 二次确认：吸取后暗块面积降幅阈值

    @classmethod
    def from_dict(cls, d) -> "PumpTiming":
        b = cls()
        if isinstance(d, dict):
            b.vac_on_s = _f(d, "vac_on_s", b.vac_on_s)
            b.vent_ms = _f(d, "vent_ms", b.vent_ms)
            b.vent_wait_s = _f(d, "vent_wait_s", b.vent_wait_s)
            b.verify_dark_area_drop = _f(d, "verify_dark_area_drop", b.verify_dark_area_drop)
        return b


# ---------------------------------------------------------------------- #
#  文件读取（优先 .json → .example.json → 默认）
# ---------------------------------------------------------------------- #
def _read_json(name: str, calib_dir, log) -> Optional[dict]:
    base = Path(calib_dir) if calib_dir else _CALIB_DIR
    for fn in (f"{name}.json", f"{name}.example.json"):
        p = base / fn
        if p.exists():
            try:
                return json.loads(p.read_text("utf-8"))
            except Exception as e:  # noqa: BLE001
                if log:
                    log(f"[scara.calib] {fn} 解析失败({e}) → 降级默认")
                return None
    return None


def load_jacobian(calib_dir=None, log=None) -> CamJacobian:
    return CamJacobian.from_dict(_read_json("scara_cam_jacobian", calib_dir, log))


def load_tcp(calib_dir=None, log=None) -> TcpCalib:
    return TcpCalib.from_dict(_read_json("scara_tcp", calib_dir, log))


def load_heights(calib_dir=None, log=None) -> Heights:
    return Heights.from_dict(_read_json("scara_heights", calib_dir, log))


def load_waffle(calib_dir=None, log=None) -> WaffleCalib:
    return WaffleCalib.from_dict(_read_json("scara_waffle", calib_dir, log))


def load_pump_timing(calib_dir=None, log=None) -> PumpTiming:
    return PumpTiming.from_dict(_read_json("scara_pump_timing", calib_dir, log))


def load_detect_cfg(calib_dir=None, log=None) -> dict:
    """返回 WaferDetectConfig 用的 dict（缺省 {}，由 WaferDetectConfig.from_dict 兜默认）。"""
    d = _read_json("scara_wafer_detect", calib_dir, log)
    return d if isinstance(d, dict) else {}


def load_rois(calib_dir=None, log=None) -> Dict[Tuple[int, int], Tuple[int, int, int, int]]:
    """右盘格 ROI：json 键 "r,c" → (x0,y0,x1,y1)。缺省 {}。

    ★ 文件名 fallback：tools/scara_tray_rois.py 实际写出的文件叫 `scara_tray_rois.json`，
      而这里历史上读的是 `scara_right_tray_rois.json`（从未存在过）——两个名字都试，
      优先新名 `scara_right_tray_rois`，读不到再退 `scara_tray_rois`。
    """
    d = _read_json("scara_right_tray_rois", calib_dir, log)
    if d is None:
        d = _read_json("scara_tray_rois", calib_dir, log)
    out: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            try:
                parts = str(k).strip("()").split(",")
                r, c = int(parts[0]), int(parts[1])
                if isinstance(v, (list, tuple)) and len(v) == 4:
                    out[(r, c)] = tuple(int(x) for x in v)  # type: ignore[assignment]
            except (ValueError, TypeError, IndexError):
                continue
    return out


def load_j3_handeye(calib_dir=None, log=None) -> J3Handeye:
    """J3 下移相机手眼（scara_j3_handeye.json）。缺文件 → 未标定默认（mm_per_px=0，
    world_from_px 会拒绝换算 —— fail-closed，与位置类数据同纪律）。"""
    return _he_from_file(calib_dir)


def load_tray_frame(calib_dir=None, log=None) -> TrayFrame:
    """右盘世界坐标系（scara_tray_frame.json）。缺文件 → 默认占位（p00=原点，不可用，
    由上层判断 z_frame/taught_at 是否为空来停手）。"""
    return _tf_from_file(calib_dir)


# ---------------------------------------------------------------------- #
#  示教产物：右盘格位表 / 航点与避障规则（tools/scara_teach.py 写入）
#
#  这两份是**实机示教**出来的安全关键数据，容错方向刻意不对称：
#    · 位置类（world_xy / z_pick）缺失 → 返回 None 让上层**停手**，绝不猜坐标；
#    · 黑名单类（禁区）缺失 → 不误报（没标定 ≠ 有危险）；
#    · 规则类（动 J1 前置条件）规则在但数读不出 → 拒绝放行（详见 j1_move_allowed）。
# ---------------------------------------------------------------------- #
_JOINT_KEYS = ("j1", "j2", "j3", "j4")


def _opt_f(v) -> Optional[float]:
    """宽松 float：None/缺失/非法一律 None（区别于 _f 的"给默认值"）。"""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _opt_xy(v) -> Optional[Tuple[float, float]]:
    """[x, y] → (x, y)；长度不对/元素非数一律 None。"""
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        return None
    x, y = _opt_f(v[0]), _opt_f(v[1])
    return None if x is None or y is None else (x, y)


def _opt_vec(v, n: int) -> Optional[List[float]]:
    """长度必须恰为 n 的 float 列表；否则 None（宁可判"没有"也不用半截数据）。"""
    if not isinstance(v, (list, tuple)) or len(v) != n:
        return None
    out: List[float] = []
    for x in v:
        f = _opt_f(x)
        if f is None:
            return None
        out.append(f)
    return out


def _rc_key(k) -> Optional[Tuple[int, int]]:
    """json 键 "r,c" → (r, c)。"abc"/"1"/"1,x" 等非法一律 None（调用方跳过该条，不抛）。"""
    try:
        parts = str(k).strip("()[] ").split(",")
        if len(parts) != 2:
            return None
        return (int(parts[0]), int(parts[1]))
    except (TypeError, ValueError):
        return None


@dataclass
class TrayCell:
    """右盘单格示教条目（tools/scara_teach.py `cell` 子命令写入）。

    ★ **joints 是主键**（2026-07-26 用户拍板改示教回放后）：回放走 `move1` 逐轴回到这四个关节值，
      j3 即取片 Z。world_xy/pose 仅存档与网格校验用——走 World ΔXY 会丢掉每行不同的臂型
      （实测 rz_deg 极差 89.227°），大概率过不去或撞架。
      （早期 cartstep 方案曾以 world_xy 为主键，那句话已作废；数据文件里的 `_note` 若仍那么写，以本处为准。）
    source="taught" 为实机示教；"interpolated" 为平面拟合补的（只有 world_xy，用前必须逐格验证）。
    """
    world_xy: Optional[Tuple[float, float]] = None
    joints: Optional[List[float]] = None
    pose: Optional[List[float]] = None
    rz_deg: Optional[float] = None
    z_pick: Optional[float] = None          # 该格取片 Z 覆盖；None = 用 z_pick_default
    j4_hint: Optional[float] = None
    source: str = ""
    taught_at: str = ""

    @property
    def is_taught(self) -> bool:
        return self.source == "taught"

    @classmethod
    def from_dict(cls, d) -> "TrayCell":
        b = cls()
        if not isinstance(d, dict):
            return b
        b.world_xy = _opt_xy(d.get("world_xy"))
        b.joints = _opt_vec(d.get("joints"), 4)
        b.pose = _opt_vec(d.get("pose"), 6)
        b.rz_deg = _opt_f(d.get("rz_deg"))
        b.z_pick = _opt_f(d.get("z_pick"))
        b.j4_hint = _opt_f(d.get("j4_hint"))
        for k in ("source", "taught_at"):
            if isinstance(d.get(k), str):
                setattr(b, k, d[k])
        return b


@dataclass
class TrayCells:
    """右盘 6×6 格位表（scara_right_tray_cells.json）。key 为 (row, col) 整数元组，非字符串。

    ★ Z 帧与全局偏移（2026-07-27 SCARA 整机垫高 10mm 引入）
      磁盘上的 joints[2]/z_pick 属于 `z_frame` 标注的那个物理帧；`z_offset_mm` 是从该帧到
      **当前现场**的 Z 平移量。所有"可下发"的 Z 一律 = 磁盘原值 + z_offset_mm。
      拆掉垫片后把 json 里的 z_offset_mm 改回 0.0 即完全还原，示教原值全程未被污染。

      访问器分两类，**不要混用**：
        · z_pick_raw / TrayCell.joints  —— 磁盘原值。只用于审计、还原、打印对照，**绝不下发**。
        · z_pick_for / z_safe_for / replay_joints_for —— 生效值（已叠加偏移）。**唯一可下发的**。
    """
    rows: int = 6
    cols: int = 6
    z_pick_default: Optional[float] = None
    z_safe: Optional[float] = None
    blocked_cells: Set[Tuple[int, int]] = field(default_factory=set)
    cells: Dict[Tuple[int, int], TrayCell] = field(default_factory=dict)
    z_offset_mm: float = 0.0        # 磁盘帧 → 当前现场的 Z 平移量。正=抬高，负=下探
    z_frame: str = ""               # 磁盘上 joints[2]/z_pick 属于哪个物理帧
    z_offset_corrupt: bool = False  # z_offset_mm 键存在但解析不出 → 所有 Z 访问器返回 None

    def get(self, row: int, col: int) -> Optional[TrayCell]:
        return self.cells.get((int(row), int(col)))

    def cell_xy(self, row: int, col: int) -> Optional[Tuple[float, float]]:
        """该格的 World XY(mm)。未示教/坐标非法 → None（上层停手，绝不猜坐标）。

        注意本方法**不看禁用格**：禁用格若已示教仍返回坐标。能不能去请另问 is_blocked()。
        """
        c = self.get(row, col)
        return c.world_xy if c else None

    @property
    def offset_active(self) -> bool:
        """当前是否有非零 Z 偏移（或偏移损坏）生效。"""
        return self.z_offset_corrupt or abs(self.z_offset_mm) > 1e-9

    def z_pick_raw(self, row: int, col: int) -> Optional[float]:
        """该格取片 Z 的**磁盘原值**（z_frame 帧）：格内 z_pick 优先，回落全局 z_pick_default。

        ★ **只用于审计 / 还原 / 打印对照，绝不下发。** 要下刀请用 z_pick_for()。
        """
        c = self.get(row, col)
        if c is not None and c.z_pick is not None:
            return c.z_pick
        return self.z_pick_default

    def z_pick_for(self, row: int, col: int) -> Optional[float]:
        """★该格**生效**取片 Z = z_pick_raw + z_offset_mm。**唯一可下发的 Z。**

        ★ 返回 None 有三种含义，**每一种都必须停手**：
             ① Z 尚未标定（raw 为 None）
             ② z_offset_mm 键存在但损坏（现场有垫高但读不出量 → 宁可停也不能按错的量下刀）
             ③ 该格不存在
          严禁用 0.0 之类兜底：SCARA World Z=0 不是"安全高度"而是一个具体位置，
          拿它下刀会把吸盘直接怼进盘里 / 压碎硅片。
          正确处理是报"Z 未标定，请先 `scara_teach.py zpick`"并中止动作。
        """
        if self.z_offset_corrupt:
            return None
        raw = self.z_pick_raw(row, col)
        return None if raw is None else raw + self.z_offset_mm

    def z_safe_for(self) -> Optional[float]:
        """★**生效**安全抬升 Z = z_safe + z_offset_mm。

        z_safe 的语义是"横移不刮盘边框的高度"，与右盘同帧，所以同样叠加偏移。
        None = 未标定 或 偏移损坏 → 停手（同 z_pick_for 的纪律）。
        """
        if self.z_offset_corrupt or self.z_safe is None:
            return None
        return self.z_safe + self.z_offset_mm

    def replay_joints_for(self, row: int, col: int) -> Optional[List[float]]:
        """★回放主键：返回 [j1, j2, j3 + z_offset_mm, j4]。**只动 J3，其余轴原样。**

        J3 是直动 Z 轴，整机垫高只改 Z，J1/J2/J4 这三个转动关节对基座平移是不变量。
        None = 未示教 / joints 缺失 / 偏移损坏 → 停手。
        """
        if self.z_offset_corrupt:
            return None
        c = self.get(row, col)
        if c is None or c.joints is None:
            return None
        j = list(c.joints)
        j[2] += self.z_offset_mm
        return j

    def z_consistency_mm(self, row: int, col: int) -> Optional[float]:
        """审计：|joints[2] − z_pick|。两者本应同源（z_pick 是 J3 的 3~4 位小数舍入），
        >0.01mm 说明数据被手改过或来自不同次读数。None = 缺数据无法比。
        """
        c = self.get(row, col)
        if c is None or c.joints is None or c.z_pick is None:
            return None
        return abs(float(c.joints[2]) - float(c.z_pick))

    def is_blocked(self, row: int, col: int) -> bool:
        """该格是否被标记为不可用（够不到 / 被挡）。"""
        return (int(row), int(col)) in self.blocked_cells

    def taught_cells(self) -> List[Tuple[int, int]]:
        """实机示教过的格（source=="taught"），按 (row, col) 升序。"""
        return sorted(rc for rc, c in self.cells.items() if c.is_taught)

    def usable_cells(self) -> List[Tuple[int, int]]:
        """有 World XY 且未被禁用的格（含插值格），按 (row, col) 升序。"""
        return sorted(rc for rc, c in self.cells.items()
                      if c.world_xy is not None and rc not in self.blocked_cells)

    @classmethod
    def from_dict(cls, d) -> "TrayCells":
        b = cls()
        if not isinstance(d, dict):
            return b
        for k in ("rows", "cols"):
            try:
                v = int(d[k])
                if v > 0:
                    setattr(b, k, v)
            except (KeyError, TypeError, ValueError):
                pass
        b.z_pick_default = _opt_f(d.get("z_pick_default"))
        b.z_safe = _opt_f(d.get("z_safe"))
        if isinstance(d.get("z_frame"), str):
            b.z_frame = d["z_frame"]
        # ★ 容错方向刻意不对称，别"顺手"改成 _f(d, "z_offset_mm", 0.0)：
        #   · 键不存在 → 0.0。示教期/拆掉垫片后本来就没这个键，不能因此锁死机器。
        #   · 键存在但解析不出 → corrupt，所有 Z 访问器返回 None。键存在本身就是
        #     "现场有垫高"的证据，此时读不出数只说明数据坏了；按 0 下刀 = 按垫高前的
        #     深度下刀 = 撞盘。用 _f 给默认值会**静默**吞掉损坏，那是最糟的一种。
        if "z_offset_mm" in d:
            _zo = _opt_f(d.get("z_offset_mm"))
            if _zo is None:
                b.z_offset_corrupt = True
            else:
                b.z_offset_mm = _zo
        bl = d.get("blocked_cells")
        if isinstance(bl, (list, tuple)):
            for item in bl:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    try:
                        b.blocked_cells.add((int(item[0]), int(item[1])))
                    except (TypeError, ValueError):
                        continue
        cd = d.get("cells")
        if isinstance(cd, dict):
            for k, v in cd.items():
                rc = _rc_key(k)
                if rc is None or not isinstance(v, dict):
                    continue                       # 键非法（如 "abc"）/条目非 dict → 跳过，不抛
                b.cells[rc] = TrayCell.from_dict(v)
        return b


@dataclass
class Waypoint:
    """一个示教航点（tools/scara_teach.py `wp` 子命令写入）。"""
    name: str = ""
    joints: Optional[List[float]] = None
    pose: Optional[List[float]] = None
    note: str = ""
    source: str = ""
    taught_at: str = ""

    @classmethod
    def from_dict(cls, name, d) -> "Waypoint":
        b = cls(name=str(name))
        if isinstance(d, dict):
            b.joints = _opt_vec(d.get("joints"), 4)
            b.pose = _opt_vec(d.get("pose"), 6)
            for k in ("note", "source", "taught_at"):
                if isinstance(d.get(k), str):
                    setattr(b, k, d[k])
        return b


@dataclass
class ForbiddenZone:
    """关节空间禁区（实机撞过的位置，scara_teach.py `danger` 子命令写入）。

    判定语义**必须与 tools/scara_teach.py::zone_hit() 逐字一致**——同一份数据两处解释不能分家：
      · 有 joint_ranges → 给出的每个轴区间**同时**命中才算危险（AND；未给的轴不约束）；
      · 无 joint_ranges → 退化为"与样本点 joints 各分量距离都 ≤ near_deg"。

    ★ TODO(避障作废的过滤，与 tools/scara_teach.py::zone_hit 是**同一个决策点**，要改一起改)
      数据里现有两层作废标记，本类**都不解析**：
        · model_invalidated      —— J1/J2 独立区间这个模型形式被 8 个实测安全格反证
        · sample_frame_invalidated —— 2026-07-27 SCARA 垫高 10mm、支架未垫高，样本几何已变
      因此 Waypoints.zone_hit() 对真实 36 格取片位会命中 8 格（row0 全行 + (5,0)/(5,1)）。
      **但现在先别加过滤**：撞的是小臂杆(H1)还是随 J3 升降的腕部组件(H2) 尚未定论
      （见 scara_waypoints.json 的 observation_plan.riser_h1_h2_discriminator），
      在物理事实定下来之前，**保留告警是更安全的一侧**。
      真要改时三处必须同时改：本类 from_dict、Waypoints.zone_hit、scara_teach.py 的
      cmd_cell(已过滤) 与 cmd_check(未过滤)——目前这四个消费点已经有三种不同行为。
    """
    name: str = ""
    desc: str = ""
    hits: str = ""
    joints: Optional[List[float]] = None
    joint_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    near_deg: float = 5.0

    def hit(self, joints) -> bool:
        """当前 4 轴关节是否落在本禁区内。joints 非法 → False（判定权交回上层，见 zone_hit）。"""
        j = _opt_vec(joints, 4)
        if j is None:
            return False
        if self.joint_ranges:
            for i, key in enumerate(_JOINT_KEYS):
                r = self.joint_ranges.get(key)
                if r is None:
                    continue                       # 未给的轴不约束
                if not (r[0] <= j[i] <= r[1]):
                    return False                   # 任一给定区间没命中 → 不在禁区（AND）
            return True
        ref = self.joints
        if ref is None:
            return False
        return all(abs(j[i] - ref[i]) <= self.near_deg for i in range(4))

    @classmethod
    def from_dict(cls, d) -> "ForbiddenZone":
        b = cls()
        if not isinstance(d, dict):
            return b
        for k in ("name", "desc", "hits"):
            if isinstance(d.get(k), str):
                setattr(b, k, d[k])
        b.joints = _opt_vec(d.get("joints"), 4)
        rngs = d.get("joint_ranges")
        if isinstance(rngs, dict):
            for key in _JOINT_KEYS:
                r = _opt_xy(rngs.get(key))         # [lo, hi]
                if r is not None:
                    b.joint_ranges[key] = (min(r), max(r))   # 写入方已排序，这里再兜一次
        nd = _opt_f(d.get("near_deg"))
        if nd is not None:
            b.near_deg = nd
        return b


@dataclass
class JointOrderRule:
    """关节动作顺序规则（如"动 J1 前必须先把 J2 收到 ≥ 阈值"）。"""
    name: str = ""
    trigger: str = ""
    require: Dict[str, float] = field(default_factory=dict)
    clear_waypoint: str = ""
    desc: str = ""

    @classmethod
    def from_dict(cls, d) -> "JointOrderRule":
        b = cls()
        if not isinstance(d, dict):
            return b
        for k in ("name", "trigger", "clear_waypoint", "desc"):
            if isinstance(d.get(k), str):
                setattr(b, k, d[k])
        req = d.get("require")
        if isinstance(req, dict):
            for k, v in req.items():
                f = _opt_f(v)
                if f is not None:                  # 非数值项直接丢 → 上层按 fail-safe 拒绝放行
                    b.require[str(k)] = f
        return b


@dataclass
class Waypoints:
    """航点 + 避障规则（scara_waypoints.json）。实机示教的**安全关键**数据。"""
    waypoints: Dict[str, Waypoint] = field(default_factory=dict)
    forbidden_zones: List[ForbiddenZone] = field(default_factory=list)
    joint_order_rules: List[JointOrderRule] = field(default_factory=list)
    j1_safe_sector_when_extended: Optional[Tuple[float, float]] = None   # 伸展态 J1 安全扇区
    micro_load_xy_mm: Optional[Tuple[float, float]] = None               # 显微镜 XY 台绝对装样位
    micro_load_stage_frame: str = ""     # 装样位所属的台坐标帧（旧数据可能没有，见 current_stage_frame）
    current_stage_frame: str = ""
    """现场当前的 XY 台坐标帧标注（顶层键，手工维护，仿已有的 current_z_frame）。

    Zaber 是绝对坐标、以 home 原点为基准，**重新 home 之后所有记下来的台坐标全部失效**。
    失效的表现极其阴险：坐标看起来仍是合法数字，台子也会乖乖走过去，只是走到了别的地方。
    所以一旦重新 home，就把这个键改名（如 zaber_home_20260901），
    所有旧帧的扫描配方会被 ScanRecipe.validate() 直接拒绝，而不是安静地对着错地方扫。
    """

    def get(self, name) -> Optional[Waypoint]:
        return self.waypoints.get(str(name))

    def joints_of(self, name) -> Optional[List[float]]:
        """航点的 4 轴关节值副本；航点不存在/关节缺失 → None（上层停手）。"""
        wp = self.get(name)
        return list(wp.joints) if wp is not None and wp.joints is not None else None

    def place_name_for(self, base: str, cell=None) -> str:
        """按源格解析放样航点名：优先 `base@r,c`，没有就退回全局 `base`。

        ★ 为什么需要按源格分：2026-07-28 实测，同一个 place_micro 下
          (5,1) 的片能落进槽、(5,0) 的落不进。row5 六格 rz_deg 全同（-74.802），
          所以不是角度问题；差异来自**吸盘不一定吸在片的正中心**——
          示教对的是硅片中心，而各格凹槽大小不同、片在格内落点本就有差异
          （网格拟合残差最大 2.80mm，该残差含真实物理差异，见 handoff §2 注意2）。
          片被偏心吸起，到了显微镜就偏心落下。

        ⚠ 这套按格存只在「偏移对每格是固定的」时候成立。若同一格连续两次的偏移都不同，
          说明片在凹槽里的落点本身在变，示教多少格都没用，得靠视觉对中。
          用之前请先做重复性判别。
        """
        if cell is not None and len(cell) >= 2:
            per = f"{base}@{int(cell[0])},{int(cell[1])}"
            if self.joints_of(per):
                return per
        return base

    def zone_hit(self, joints) -> List[str]:
        """返回命中的禁区名列表（空 = 未命中，或压根没有禁区数据）。

        fail-safe 方向：**数据缺失时返回空，不误报**。禁区表是"实机撞过的已知黑名单"，
        没有条目就等于"没有已知危险"；此时若一律报警，只会让操作员学会忽略告警（告警疲劳），
        反而把真命中时的那一声也一起淹掉。真正的硬阻断交给 j1_move_allowed 那类
        "规则存在即约束"的检查——见其 docstring 里对这个不对称的解释。
        """
        return [z.name for z in self.forbidden_zones if z.hit(joints)]

    def j1_order_rule(self) -> Optional[JointOrderRule]:
        """取 trigger=="before_moving_j1" 的顺序规则（没有 → None）。"""
        return next((r for r in self.joint_order_rules
                     if r.trigger == "before_moving_j1"), None)

    def j2_min_for_j1_move(self) -> Optional[float]:
        """动 J1 前 J2 必须收到的最小角度（规则的 require.j2_min）。

        None 有两种含义：规则不存在 / 规则在但阈值损坏。要区分请用 j1_order_rule()，
        或直接用 j1_move_allowed()——它已经按 fail-safe 把两种情况分开处理了。
        """
        r = self.j1_order_rule()
        return r.require.get("j2_min") if r is not None else None

    def j1_move_allowed(self, joints) -> Tuple[bool, str]:
        """当前姿态能不能动 J1？→ (是否允许, 人话原因)。

        ★ 与 zone_hit 方向**故意相反**的 fail-safe，理由是"数据缺失"在两处含义不同：
          · 规则**整个不存在** → True。说明这条约束还没标定，不能因为"没数据"就把整台机器
            锁死（示教/调试阶段本来就一条规则都没有）；
          · 规则**存在但阈值读不出 / 当前 J2 读不出** → False。规则的存在本身就是"这里撞过"
            的证据，此时读不出数只说明数据坏了或读数不可信，宁可挡住让人来看，
            也绝不能替一个可能危险的姿态放行。
        """
        rule = self.j1_order_rule()
        if rule is None:
            return True, "未标定 J1 前置规则（joint_order_rules 无 before_moving_j1）→ 不阻断"
        tag = rule.name or "before_moving_j1"
        thr = rule.require.get("j2_min")
        if thr is None:
            return False, (f"规则「{tag}」存在但读不出 require.j2_min → 数据损坏，"
                           f"拒绝动 J1（请检查 scara_waypoints.json）")
        j = _opt_vec(joints, 4)
        if j is None:
            return False, f"规则「{tag}」生效，但当前关节读数非法（{joints!r}）→ 无法判定，拒绝动 J1"
        if j[1] >= thr:
            return True, f"J2={j[1]:.4f} ≥ {thr:.4f}，可以动 J1"
        clear = rule.clear_waypoint or "j2_clear"
        why = f"（{rule.desc}）" if rule.desc else ""
        return False, (f"J2={j[1]:.4f} < {thr:.4f}，此时动 J1 会撞 —— "
                       f"请先走航点「{clear}」把 J2 收到 ≥ {thr:.4f}{why}")

    @classmethod
    def from_dict(cls, d) -> "Waypoints":
        b = cls()
        if not isinstance(d, dict):
            return b
        wps = d.get("waypoints")
        if isinstance(wps, dict):
            for name, v in wps.items():
                b.waypoints[str(name)] = Waypoint.from_dict(name, v)
        zs = d.get("forbidden_zones")
        if isinstance(zs, (list, tuple)):
            for i, z in enumerate(zs):
                if not isinstance(z, dict):
                    continue
                zone = ForbiddenZone.from_dict(z)
                if not zone.name:
                    zone.name = f"zone_{i}"        # 保证 zone_hit 不会返回空串
                b.forbidden_zones.append(zone)
        rs = d.get("joint_order_rules")
        if isinstance(rs, (list, tuple)):
            for r in rs:
                if isinstance(r, dict):
                    b.joint_order_rules.append(JointOrderRule.from_dict(r))
        sec = d.get("j1_safe_sector_when_extended")
        if isinstance(sec, dict):
            rng = _opt_xy(sec.get("range"))
            if rng is not None:
                b.j1_safe_sector_when_extended = (min(rng), max(rng))
        mlp = d.get("micro_load_pos")
        if isinstance(mlp, dict):
            b.micro_load_xy_mm = _opt_xy(mlp.get("xy_mm"))
            b.micro_load_stage_frame = str(mlp.get("stage_frame") or "")
        b.current_stage_frame = str(d.get("current_stage_frame") or "")
        return b

    def stage_frame_conflict(self) -> str:
        """装样位的帧标注与现场帧不一致时返回人话说明，一致/无从判断时返回 ""。

        ★ 刻意**只告警不阻断**，与 ScanRecipe 的硬拒绝不对称，理由是两者的历史不同：
        `micro_load_pos` 是 2026-07-28 就标好、且已经在跑的数据，它根本没有 stage_frame
        字段。这里若 fail-closed，等于用一个新加的门把一条已经能用的流程锁死
        （lessons 2026-07-27「加了防护就必须同时给钥匙」）。扫描配方是全新数据、
        从第一天就带帧，没有历史包袱，所以那边可以硬拒。
        补齐办法：重跑 `scara_teach.py micro --from-rpc`，它现在会写 stage_frame。
        """
        if not self.current_stage_frame or self.micro_load_xy_mm is None:
            return ""
        if not self.micro_load_stage_frame:
            return (f"micro_load_pos 没有 stage_frame 字段（现场帧={self.current_stage_frame}）"
                    f" —— 无法判断它是不是这次 home 之后标的。"
                    f"补齐：python tools/scara_teach.py micro --from-rpc")
        if self.micro_load_stage_frame != self.current_stage_frame:
            return (f"micro_load_pos 的帧是 {self.micro_load_stage_frame}，"
                    f"现场帧是 {self.current_stage_frame} —— 台子重新 home 过，"
                    f"这个装样位已经失效，必须重标")
        return ""


def load_tray_cells(calib_dir=None, log=None) -> TrayCells:
    """右盘格位表。缺失/损坏 → 空表（cell_xy/z_pick_for 全 None → 上层停手，不会乱走）。"""
    t = TrayCells.from_dict(_read_json("scara_right_tray_cells", calib_dir, log))
    if log and t.offset_active:
        if t.z_offset_corrupt:
            log("[scara.calib] ⚠⚠ 右盘 z_offset_mm 解析不出 —— 所有生效 Z 访问器返回 None，"
                "取放片流程会拒绝启动。请修正 scara_right_tray_cells.json 后重试。")
        else:
            zs = [c.z_pick for c in t.cells.values() if c.z_pick is not None]
            rng = (f"，生效取片 Z [{min(zs)+t.z_offset_mm:.3f}, {max(zs)+t.z_offset_mm:.3f}]"
                   if zs else "")
            log(f"[scara.calib] 右盘 z_offset_mm={t.z_offset_mm:+.3f} 已加载"
                f"（z_frame={t.z_frame or '?'}）{rng}")
    return t


def load_waypoints(calib_dir=None, log=None) -> Waypoints:
    """航点/禁区/顺序规则。缺失/损坏 → 空表（zone_hit 不误报；无规则时 j1_move_allowed 放行）。"""
    wp = Waypoints.from_dict(_read_json("scara_waypoints", calib_dir, log))
    if log:
        conflict = wp.stage_frame_conflict()
        if conflict:
            log(f"[scara.calib] ⚠ {conflict}")
    return wp


# ---------------------------------------------------------------------- #
#  显微镜扫描配方
# ---------------------------------------------------------------------- #
_REPO_ROOT = _CALIB_DIR.parents[2]        # src/scara/calib → src/scara → src → 仓库根
_PRESETS_DIR = _REPO_ROOT / "presets"


def _preset_matching_fov(fov_w_um: float, fov_h_um: float,
                         tol_um: float = 1.0) -> Tuple[str, Optional[int]]:
    """按 FOV 反查 `presets/*.json`，返回 (preset 名, 物镜序号)。查不到返回 ("", None)。

    为什么需要：acquisition project 里 `preset_name` 常常是空的，但 FOV 一定在。
    而开扫前必须知道该切到哪个物镜 —— FOV 与物镜是一一对应的（5x/10x/20x/50x
    实测 1990.2/1034.6/500.7/206.2 µm 宽，彼此差着倍数，1µm 容差不会撞车）。
    查不到就返回 None，让上层 fail-closed，绝不默认"保持当前物镜"。
    """
    if not _PRESETS_DIR.is_dir():
        return "", None
    for p in sorted(_PRESETS_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text("utf-8"))
            w, h = float(d["fov_width_um"]), float(d["fov_height_um"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        if abs(w - fov_w_um) <= tol_um and abs(h - fov_h_um) <= tol_um:
            obj = d.get("objective")
            return str(d.get("name") or p.stem), (int(obj) if isinstance(obj, int) else None)
    return "", None


@dataclass
class ScanRecipe:
    """华夫盒槽内硅片的显微镜自动扫描配方（micro_scan_recipe.json）。

    ★ 契约与 `TrayCells.z_pick_for` 一致：**未标定/不可信一律不给数，绝不兜默认值。**
      这里的默认值格外危险 —— `MicroscopeController.start_scan` 对空参数的默认行为是
      `ref_1 = 当前位置`、`ref_2 = ref_1`、3×3 grid，也就是**在装样位原地拍 9 张同一画面**，
      而且照样返回 True。生成一份"看起来正常但对着错地方"的扫描计划，
      比什么都不生成危险得多：它会跑满几百格、存一堆无效图，全程不报错。

    所有校验集中在 `validate()` 一处，示教工具、preflight、步骤生成器三个消费方共用，
    避免 lessons 2026-07-27 那种"三个消费点各写一份判据、迟早分叉"。
    """

    corner_1_mm: Optional[Tuple[float, float]] = None    # 扫描起点
    corner_2_mm: Optional[Tuple[float, float]] = None    # 对角
    stage_frame: str = ""

    mode: str = "tile"
    preset_name: str = "5x"
    overlap_percent: float = 10.0
    scanning_order: str = "Snake by Rows"
    autofocus_each_xy: bool = True
    focus_surface_enabled: bool = False
    settle_time_ms: int = 20

    save_subdir_pattern: str = "inspect_{ts}_r{row}c{col}"
    recognition_enabled: bool = False
    recognition_model: str = ""

    # 相机硬件触发。★ 必须原样透传，不能靠 start_scan 的缺省值兜底：
    # 缺省是 enabled=True/50/0/**20**，而操作员项目里的 delay_after 常是自己调出来的
    # （实际在用的 autosearch.json 就是 100）。触发后等多久取帧直接决定拿到的是不是那一帧，
    # 少传一个参数就会在无人察觉的情况下把它改回默认值。
    camera_trigger_enabled: bool = True
    camera_trigger_duration_ms: int = 50
    camera_trigger_delay_before_ms: int = 0
    camera_trigger_delay_after_ms: int = 20

    stage_x_limits_mm: Optional[Tuple[float, float]] = None
    stage_y_limits_mm: Optional[Tuple[float, float]] = None
    max_tiles: int = 1200
    focus_margin_mm: float = 1.0

    # 开扫前的初始对焦（2026-07-29 用户拍板：**存示教Z兜底 + WDI微调，两道都要**）。
    # 为什么非有不可：装样位的焦点停在 5.000mm，那是「离样品最远」的一侧，
    # 而逐点自动对焦只在当前 Z 附近开 ±1mm 的搜索窗（_AF_SCAN_HALF_RANGE_MM）。
    # 不先把焦点送到样品附近就开扫，整轮会一路失焦，而且每格都"拍到了"、不报错。
    focus_start_mm: Optional[float] = None
    focus_wdi_refine: bool = True
    focus_max_drift_mm: float = 0.5
    focus_taught_at: str = ""
    focus_objective_at_teach: Optional[int] = None
    focus_limits_mm_at_teach: Optional[Tuple[float, float]] = None

    # 审计上下文（不参与判据，只为出事时能回溯"这份配方是在什么条件下标的"）
    taught_at: str = ""
    source: str = ""
    objective_at_teach: Optional[int] = None
    focus_mm_at_teach: Optional[float] = None

    # acquisition project（app 的 Multi-D 页面存出来的 *.json）直接带 FOV 与物镜，
    # 走那条路时不再回查 presets 文件。None = 走 preset_name 那条老路。
    fov_override: Optional[Tuple[float, float]] = None
    objective_override: Optional[int] = None

    # 区域退化门限：低于台子到位容差(0.05mm)的"区域"不是区域，是同一个点
    MIN_SPAN_MM: float = 0.05

    @property
    def is_calibrated(self) -> bool:
        return self.corner_1_mm is not None and self.corner_2_mm is not None

    @property
    def focus_is_calibrated(self) -> bool:
        return self.focus_start_mm is not None

    @property
    def expected_objective(self) -> Optional[int]:
        """配方要求的物镜序号（来自 presets/<preset>.json，不在配方里重复写）。

        开扫前必须拿它和 `get_objective()` 对一次：物镜错了，FOV 就错，步距跟着错，
        表现是**每张图都正常、整片覆盖全错、全程零报错**。
        """
        fov = self.fov_um()
        return None if fov is None else fov[2]

    def focus_start_margin_ok(self, limits_mm) -> Tuple[bool, str]:
        """示教的起始 Z 在【当前实读】限位下双边余量够不够。

        与 `focus_margin_ok` 的分工：那个问"焦点现在在哪、够不够"，这个问"我要去的那个
        Z 够不够"。开扫前两个都要过 —— 当前位置合格不代表目标位置合格。
        """
        if self.focus_start_mm is None:
            return False, ("focus_start_uncalibrated：开扫起始焦点没标过，无从判断余量")
        return self.focus_margin_ok(self.focus_start_mm, limits_mm)

    def fov_um(self) -> Optional[Tuple[float, float, Optional[int]]]:
        """从 presets/<preset_name>.json 读 (fov_w_um, fov_h_um, objective_index)。

        缺文件/字段非法 → None（上层据此拒绝，不猜 FOV：猜错 FOV 会让 tile 步距错，
        表现为扫描区域覆盖不全或大量重叠，而每张图本身都是正常的，极难发现）。
        """
        # acquisition project 自带 FOV（它就是 app 里 Multi-D 页面存出来的那份），
        # 此时不必再回头查 preset 文件 —— 但物镜序号仍要有出处，见 from_acquisition_project。
        if self.fov_override is not None:
            w, h = self.fov_override
            return (w, h, self.objective_override)
        name = str(self.preset_name or "").strip()
        if not name:
            return None
        p = _PRESETS_DIR / f"{name}.json"
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text("utf-8"))
            w = float(d["fov_width_um"])
            h = float(d["fov_height_um"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None
        if w <= 0 or h <= 0:
            return None
        obj = d.get("objective")
        return (w, h, int(obj) if isinstance(obj, int) else None)

    def xy_points_mm(self) -> Optional[List[Tuple[float, float]]]:
        """算出实际采集点列表。★复用显微镜侧同一个 planner，绝不自己再实现一遍步距公式。

        自己抄一份的下场见 lessons 2026-07-27：同一份数据两个消费方各写一份判据，
        迟早分叉，而且分叉后 preflight 报的 tile 数和实际扫的对不上，人还以为 preflight 是对的。
        """
        if not self.is_calibrated:
            return None
        fov = self.fov_um()
        if fov is None:
            return None
        fw, fh, _ = fov
        try:
            from microscope.logic.acquisition_planner import AcquisitionPlanner
        except ImportError:
            return None
        try:
            plan = AcquisitionPlanner().generate_xy_plan(
                mode="XY Tile" if self.mode == "tile" else "XY Grid",
                ref_1=self.corner_1_mm, ref_2=self.corner_2_mm,
                scanning_order=self.scanning_order,
                fov_width_um=fw, fov_height_um=fh,
                overlap_percent=self.overlap_percent,
            )
        except (ValueError, TypeError):
            return None
        return list(plan.points)

    def tile_count(self) -> Optional[int]:
        pts = self.xy_points_mm()
        return None if pts is None else len(pts)

    def area_mm(self) -> Optional[Tuple[float, float]]:
        if not self.is_calibrated:
            return None
        return (abs(self.corner_2_mm[0] - self.corner_1_mm[0]),
                abs(self.corner_2_mm[1] - self.corner_1_mm[1]))

    def validate(self, current_stage_frame: str = "") -> Tuple[bool, str]:
        """全部门禁集中在此。返回 (通过?, 原因)。原因非空 = 扫描步骤一步都不生成。"""
        if not self.is_calibrated:
            return False, ("scan_area_uncalibrated：华夫盒槽的扫描区域从没标过。"
                           "标法：操作员把 XY 台挪到硅片一角 → "
                           "`python tools/scara_teach.py scanarea 1 --from-rpc`，"
                           "再挪到对角 → `scanarea 2 --from-rpc`")
        span = self.area_mm()
        if span[0] < self.MIN_SPAN_MM or span[1] < self.MIN_SPAN_MM:
            return False, (f"scan_area_degenerate：两个角几乎重合"
                           f"（ΔX={span[0]:.4f} ΔY={span[1]:.4f} mm，门限 {self.MIN_SPAN_MM}）"
                           f" —— 多半是两次 scanarea 之间台子没挪。重标那一角")
        # 帧不一致 = 台子重新 home 过 = 坐标还是合法数字但指向别处，必须硬拒
        if current_stage_frame:
            if not self.stage_frame:
                return False, (f"scan_stage_frame_missing：配方没有 stage_frame，"
                               f"而现场帧是 {current_stage_frame} —— 无法确认这份区域是不是"
                               f"这次 home 之后标的。重标一次即可（示教工具会自动写帧）")
            if self.stage_frame != current_stage_frame:
                return False, (f"scan_stage_frame_mismatch：配方帧 {self.stage_frame} ≠ "
                               f"现场帧 {current_stage_frame} —— 台子重新 home 过，"
                               f"这两个角已经指向别的地方了。必须重标")
        ok, why = self.area_in_travel()
        if not ok:
            return False, why
        if self.fov_um() is None:
            return False, (f"scan_preset_missing：读不到 presets/{self.preset_name}.json 的 FOV "
                           f"—— 没有 FOV 就算不出 tile 步距")
        n = self.tile_count()
        if n is None:
            return False, "scan_plan_failed：采集点算不出来（planner 不可用或参数非法）"
        if n <= 0:
            return False, "scan_plan_empty：算出来 0 个采集点"
        if n > self.max_tiles:
            return False, (f"scan_tiles_exceed_max：{n} 格 > 上限 {self.max_tiles} 格。"
                           f"区域 {span[0]:.2f}×{span[1]:.2f}mm @ {self.preset_name}。"
                           f"确认区域框对了；真要扫这么多就把 limits.max_tiles 显式调高")
        # ---- 光学前置条件放在最后：先确认扫描计划本身合理（区域/格数），再看光路就绪没 ----
        # 起始焦点：没有它，整轮扫描会从"离样品最远"的装样位焦点起步，而逐点 AF 只在
        # 当前 Z 附近开 ±1mm 搜索窗，根本够不着样品面。失焦的图照样存盘、照样返回成功。
        if not self.focus_is_calibrated:
            return False, ("scan_focus_uncalibrated：开扫起始焦点从没标过。"
                           f"标法：把物镜切到 {self.preset_name}、手动对焦到样品清晰 → "
                           "`python tools/scara_teach.py scanfocus --from-rpc`")
        exp_obj = self.expected_objective
        if (self.focus_objective_at_teach is not None and exp_obj is not None
                and int(self.focus_objective_at_teach) != int(exp_obj)):
            return False, (f"scan_focus_objective_mismatch：起始焦点是在物镜 "
                           f"{self.focus_objective_at_teach} 下标的，而配方要用物镜 {exp_obj}"
                           f"（{self.preset_name}）—— 两者工作距离不同，这个 Z 不能跨物镜复用。"
                           f"切到 {self.preset_name} 重新对焦后跑 `scanfocus --from-rpc`")
        return True, ""

    def area_in_travel(self) -> Tuple[bool, str]:
        """两个角是否都落在 XY 台实读行程内。

        ★ 限位未实读时**放行并由调用方告警**，不 fail-closed：全仓至今没有任何一处读过
        设备自己的 limit（130×100 是标称/UI 值），此时硬拒等于把功能锁死在一个
        我们自己没做的事情上。实读办法见 `MicroscopeController.get_stage_limits`。
        真正兜底的是 `RpcMicroscope.stage_goto` 的到位回读校验 —— 越界被截断一定会被它抓到。
        """
        if not self.is_calibrated:
            return False, "scan_area_uncalibrated"
        if self.stage_x_limits_mm is None or self.stage_y_limits_mm is None:
            return True, ""
        xlo, xhi = min(self.stage_x_limits_mm), max(self.stage_x_limits_mm)
        ylo, yhi = min(self.stage_y_limits_mm), max(self.stage_y_limits_mm)
        for tag, (x, y) in (("corner_1", self.corner_1_mm), ("corner_2", self.corner_2_mm)):
            if not (xlo <= x <= xhi) or not (ylo <= y <= yhi):
                return False, (f"scan_area_out_of_travel：{tag}=({x:.4f},{y:.4f}) 超出台子行程 "
                               f"X[{xlo:.3f},{xhi:.3f}] Y[{ylo:.3f},{yhi:.3f}]")
        return True, ""

    def focus_margin_ok(self, focus_mm, limits_mm) -> Tuple[bool, str]:
        """焦点是否离两侧限位都还有 `focus_margin_mm` 的余量。

        为什么是双边：逐点自动对焦要在当前 Z 附近开一个搜索窗（最大半窗
        `_AF_SCAN_HALF_RANGE_MM = 1.0mm`）。焦点贴着任一侧限位时窗口是单边的，
        整轮扫描会一路撞限位告警 —— 2026-07-26 就是为这个把限位放宽过一次。
        """
        if focus_mm is None or not isinstance(limits_mm, (list, tuple)) or len(limits_mm) != 2:
            return False, "focus_state_unreadable：读不到焦点位置或限位，无法判断余量"
        lo, hi = min(limits_mm), max(limits_mm)
        m = float(self.focus_margin_mm)
        below, above = float(focus_mm) - lo, hi - float(focus_mm)
        if below >= m and above >= m:
            return True, ""
        need_lo = min(lo, float(focus_mm) - m)
        need_hi = max(hi, float(focus_mm) + m)
        return False, (f"focus_margin_too_small：焦点 {focus_mm:.3f}mm 在限位 [{lo:.3f},{hi:.3f}] 内，"
                       f"下余量 {below:.3f} / 上余量 {above:.3f}，都得 ≥{m:.3f}mm。"
                       f"出路二选一：① 把限位放宽到 [{need_lo:.3f},{need_hi:.3f}]"
                       f"（★放宽方向要按 50x 的实际工作距离定，这个数只有操作员能给）"
                       f"② 先把焦点移到离两侧都 ≥{m:.3f}mm 的位置再开扫")

    def to_scan_params(self, save_directory=None) -> Optional[dict]:
        """构造 RPC `start_scan` 的扁平参数。未通过 validate 时返回 None。"""
        if not self.is_calibrated:
            return None
        fov = self.fov_um()
        if fov is None:
            return None
        fw, fh, _ = fov
        p = {
            "ref_x1": self.corner_1_mm[0], "ref_y1": self.corner_1_mm[1],
            "ref_x2": self.corner_2_mm[0], "ref_y2": self.corner_2_mm[1],
            "mode": self.mode,
            "fov_width_um": fw, "fov_height_um": fh,
            "overlap_percent": self.overlap_percent,
            "scanning_order": self.scanning_order,
            "autofocus_each_xy": self.autofocus_each_xy,
            "focus_surface_enabled": self.focus_surface_enabled,
            "settle_time_ms": self.settle_time_ms,
            "recognition_enabled": self.recognition_enabled,
            "recognition_model": self.recognition_model,
            # 相机触发四件套：不传的话 start_scan 会用自己的缺省值（尤其 delay_after 20ms）
            "camera_trigger_enabled": self.camera_trigger_enabled,
            "camera_trigger_duration_ms": self.camera_trigger_duration_ms,
            "camera_trigger_delay_before_ms": self.camera_trigger_delay_before_ms,
            "camera_trigger_delay_after_ms": self.camera_trigger_delay_after_ms,
        }
        if save_directory:
            p["save_directory"] = str(save_directory)
        return p

    # project 与 planner 是两套扫描顺序命名，必须显式映射 —— 直接把 "snake_by_rows"
    # 传给 planner 会抛 Unsupported scanning order（见 acquisition_planner._normalize）。
    _ORDER_FROM_PROJECT = {
        "snake_by_rows": "Snake by Rows",
        "snake_by_columns": "Snake by Cols",
        "row_by_row": "By Rows",
        "column_by_column": "By Columns",
    }

    @classmethod
    def from_acquisition_project(cls, d, *, stage_frame: str = "",
                                 limits: Optional[dict] = None) -> "ScanRecipe":
        """把一份 **acquisition project**（app 里 Multi-D 页面存出来的 *.json）读成配方。

        动机：操作员本来就有一套在 app 里框选扫描区、存成项目文件的工作流；
        与其要他再学一套 `scanarea`/`scanfocus` 示教命令，不如直接吃这份文件 ——
        区域两角、FOV、overlap、合焦 Z 全在里面。

        ★ 换的只是**参数来源**，判据一条都不换：区域退化/越界/格数上限/焦点余量/
          物镜一致性仍然走 `validate()` 与 `focus_start_margin_ok()` 同一份实现。

        物镜序号的出处：project 里 `preset_name` 往往是空的，但带着 FOV。
        这里按 FOV 反查 `presets/*.json`（容差 1µm）拿到物镜序号 —— 猜不出来就留 None，
        由 `micro_objective` 步骤 fail-closed，绝不"那就保持当前物镜继续扫"。
        """
        b = cls()
        if not isinstance(d, dict):
            return b
        xy = d.get("xy_scan") or {}
        pos_unit = str(xy.get("position_unit") or "mm").lower()
        k = 1.0 if pos_unit == "mm" else 0.001          # µm → mm
        try:
            b.corner_1_mm = (float(xy["reference_x"]) * k, float(xy["reference_y"]) * k)
            b.corner_2_mm = (float(xy["reference_x2"]) * k, float(xy["reference_y2"]) * k)
        except (KeyError, TypeError, ValueError):
            return b                                     # 缺角 = 未标定，validate 会说清
        b.stage_frame = str(stage_frame or "")
        b.mode = "tile" if str(xy.get("mode", "")).lower() == "tile" else "grid"
        b.overlap_percent = _f(xy, "tile_overlap_percent", b.overlap_percent)
        b.scanning_order = cls._ORDER_FROM_PROJECT.get(
            str(xy.get("scanning_order") or ""), b.scanning_order)
        fw, fh = _opt_f(xy.get("fov_width_um")), _opt_f(xy.get("fov_height_um"))
        if fw and fh and fw > 0 and fh > 0:
            b.fov_override = (fw, fh)
            b.preset_name, b.objective_override = _preset_matching_fov(fw, fh)
        b.settle_time_ms = int(_f(d, "settle_time_ms", b.settle_time_ms))

        # 合焦 Z：project 的 focus_ref_z_mm 就是操作员当时对好的焦点，语义与
        # scanfocus 记的 start_mm 完全一致 —— 只在 focus_enabled 为真时才认。
        if d.get("focus_enabled"):
            b.focus_start_mm = _opt_f(d.get("focus_ref_z_mm"))
            b.focus_objective_at_teach = b.objective_override
        rg = d.get("recognition")
        if isinstance(rg, dict):
            b.recognition_enabled = bool(rg.get("enabled", False))
            b.recognition_model = str(rg.get("model_name") or "")
        ct = d.get("camera_trigger")
        if isinstance(ct, dict):
            b.camera_trigger_enabled = bool(ct.get("enabled", b.camera_trigger_enabled))
            b.camera_trigger_duration_ms = int(_f(ct, "duration_ms", b.camera_trigger_duration_ms))
            b.camera_trigger_delay_before_ms = int(
                _f(ct, "delay_before_ms", b.camera_trigger_delay_before_ms))
            b.camera_trigger_delay_after_ms = int(
                _f(ct, "delay_after_ms", b.camera_trigger_delay_after_ms))
        if isinstance(limits, dict):
            b.stage_x_limits_mm = _opt_xy(limits.get("stage_x_mm"))
            b.stage_y_limits_mm = _opt_xy(limits.get("stage_y_mm"))
            b.max_tiles = int(_f(limits, "max_tiles", b.max_tiles))
            b.focus_margin_mm = _f(limits, "focus_margin_mm", b.focus_margin_mm)
        b.source = "acquisition_project"
        return b

    @classmethod
    def from_dict(cls, d) -> "ScanRecipe":
        b = cls()
        if not isinstance(d, dict):
            return b
        b.stage_frame = str(d.get("stage_frame") or "")
        area = d.get("area")
        if isinstance(area, dict):
            b.corner_1_mm = _opt_xy(area.get("corner_1_mm"))
            b.corner_2_mm = _opt_xy(area.get("corner_2_mm"))
            b.taught_at = str(area.get("taught_at") or "")
            b.source = str(area.get("source") or "")
            obj = area.get("objective_at_teach")
            b.objective_at_teach = int(obj) if isinstance(obj, int) else None
            b.focus_mm_at_teach = _opt_f(area.get("focus_mm_at_teach"))
        sc = d.get("scan")
        if isinstance(sc, dict):
            b.mode = str(sc.get("mode") or b.mode)
            b.preset_name = str(sc.get("preset_name") or b.preset_name)
            b.overlap_percent = _f(sc, "overlap_percent", b.overlap_percent)
            b.scanning_order = str(sc.get("scanning_order") or b.scanning_order)
            b.autofocus_each_xy = bool(sc.get("autofocus_each_xy", b.autofocus_each_xy))
            b.focus_surface_enabled = bool(sc.get("focus_surface_enabled", b.focus_surface_enabled))
            b.settle_time_ms = int(_f(sc, "settle_time_ms", b.settle_time_ms))
        sv = d.get("save")
        if isinstance(sv, dict):
            b.save_subdir_pattern = str(sv.get("subdir_pattern") or b.save_subdir_pattern)
        rg = d.get("recognition")
        if isinstance(rg, dict):
            b.recognition_enabled = bool(rg.get("enabled", False))
            b.recognition_model = str(rg.get("model_name") or "")
        ct = d.get("camera_trigger")
        if isinstance(ct, dict):
            b.camera_trigger_enabled = bool(ct.get("enabled", b.camera_trigger_enabled))
            b.camera_trigger_duration_ms = int(_f(ct, "duration_ms", b.camera_trigger_duration_ms))
            b.camera_trigger_delay_before_ms = int(
                _f(ct, "delay_before_ms", b.camera_trigger_delay_before_ms))
            b.camera_trigger_delay_after_ms = int(
                _f(ct, "delay_after_ms", b.camera_trigger_delay_after_ms))
        fc = d.get("focus")
        if isinstance(fc, dict):
            b.focus_start_mm = _opt_f(fc.get("start_mm"))
            b.focus_wdi_refine = bool(fc.get("wdi_refine", b.focus_wdi_refine))
            b.focus_max_drift_mm = _f(fc, "max_drift_mm", b.focus_max_drift_mm)
            b.focus_taught_at = str(fc.get("taught_at") or "")
            fobj = fc.get("objective_at_teach")
            b.focus_objective_at_teach = int(fobj) if isinstance(fobj, int) else None
            b.focus_limits_mm_at_teach = _opt_xy(fc.get("limits_mm_at_teach"))
        lim = d.get("limits")
        if isinstance(lim, dict):
            b.stage_x_limits_mm = _opt_xy(lim.get("stage_x_mm"))
            b.stage_y_limits_mm = _opt_xy(lim.get("stage_y_mm"))
            b.max_tiles = int(_f(lim, "max_tiles", b.max_tiles))
            b.focus_margin_mm = _f(lim, "focus_margin_mm", b.focus_margin_mm)
        return b


def load_scan_recipe(calib_dir=None, log=None) -> ScanRecipe:
    """显微镜扫描配方。缺失/损坏 → 未标定态（validate 返回失败原因，上层一步不生成）。"""
    return ScanRecipe.from_dict(_read_json("micro_scan_recipe", calib_dir, log))


@dataclass
class CalibBundle:
    """一次加载全部标定（供 servo/executor）。"""
    jacobian: CamJacobian
    tcp: TcpCalib
    heights: Heights
    waffle: WaffleCalib
    pump: PumpTiming
    detect: dict
    rois: dict
    # 新增字段一律带默认值：老调用方（如 tests/test_scara_pick_place_sequence.py）不传也能构造
    tray: TrayCells = field(default_factory=TrayCells)
    waypoints: Waypoints = field(default_factory=Waypoints)
    scan_recipe: "ScanRecipe" = field(default_factory=lambda: ScanRecipe())
    j3_handeye: "J3Handeye" = field(default_factory=J3Handeye)
    tray_frame: "TrayFrame" = field(default_factory=lambda: TrayFrame(
        p00_mm=(0.0, 0.0), row_vec_mm=(24.2, 0.0), col_vec_mm=(0.0, 24.6)))


def load_all(calib_dir=None, log=None) -> CalibBundle:
    return CalibBundle(
        jacobian=load_jacobian(calib_dir, log),
        tcp=load_tcp(calib_dir, log),
        heights=load_heights(calib_dir, log),
        waffle=load_waffle(calib_dir, log),
        pump=load_pump_timing(calib_dir, log),
        detect=load_detect_cfg(calib_dir, log),
        rois=load_rois(calib_dir, log),
        tray=load_tray_cells(calib_dir, log),
        waypoints=load_waypoints(calib_dir, log),
        scan_recipe=load_scan_recipe(calib_dir, log),
        j3_handeye=load_j3_handeye(calib_dir, log),
        tray_frame=load_tray_frame(calib_dir, log),
    )
