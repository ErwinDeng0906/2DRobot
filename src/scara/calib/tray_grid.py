"""SCARA 右盘坐标系与 36 格解算（2026-07-31 新流程，放料侧）。

数据链（全部可追溯）：

    手眼标定（handeye.py）+ 盘角点记录 → **TrayFrame**（盘的世界坐标系）
    TrayFrame + 精确 IK（kinematics.py）→ 36 格回放关节 → scara_right_tray_cells.json

TrayFrame 用「(0,0) 凹槽中心 + 行向量 + 列向量」表示，比「中心+角度+行距」少一层换算
误差；θ/行距都是派生属性。角点记录从 J3 相机像素经手眼换算而来（tools/scara_tray_calib.py），
三点定全盘，第四点校验。

边缘吸取（用户要求：硅片必须吸在边缘）：
    吸取点 = 硅片圆心 + (R_硅片 − R_吸盘 − 余量) · d̂
    d̂ = 托盘系固定方向（edge_dir_deg，0 = +col 方向），每格同一侧 —— 放片补偿才一致。
    硅片圆心/半径由 J3 相机逐格实拍（j3_wafer 检测），不用名义值：硅片在凹槽里的
    落点本来就有 ±2.8mm 随机（交接 §2 注意2），这正是必须逐格实拍的原因。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scara.pipeline import kinematics

_CALIB_DIR = Path(__file__).resolve().parent
TRAY_FRAME_FILE = "scara_tray_frame.json"
CELLS_FILE = "scara_right_tray_cells.json"

# 角点几何质量门：行/列向量不正交超过 ~2.3° 或第四角点回代残差超过此值 → 拒绝写盘
MAX_ORTHO = 0.04            # |cos(行列夹角)| 上限
MAX_CORNER_RESID_MM = 3.0


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _scale(a, k):
    return (a[0] * k, a[1] * k)


def _norm(a):
    return math.hypot(a[0], a[1])


@dataclass
class TrayFrame:
    """右盘世界坐标系：(0,0) 凹槽中心 + 相邻行/列向量（mm）。"""
    p00_mm: Tuple[float, float]
    row_vec_mm: Tuple[float, float]
    col_vec_mm: Tuple[float, float]
    z_contact_mm: float = 0.0          # 吸盘接触盘面（凹槽底面）的 J3
    z_safe_mm: float = 0.0             # 盘上方安全高（横移不刮边框）
    imaging_j3_mm: float = 0.0         # 视觉精调拍照高度（须 = 手眼标定高度）
    edge_dir_deg: float = 0.0          # 吸取边缘方向（托盘系，0=+col 向）
    cup_r_mm: float = 0.0              # 吸盘半径（边缘吸取要用）
    edge_margin_mm: float = 1.0        # 吸取点离硅片边缘的余量
    rows: int = 6
    cols: int = 6
    z_frame: str = ""                  # 物理帧名（如 scara_base_20260731）
    taught_at: str = ""

    # ---- 派生几何 ----
    @property
    def theta_deg(self) -> float:
        """盘 +col 方向的世界方位角。"""
        return math.degrees(math.atan2(self.col_vec_mm[1], self.col_vec_mm[0]))

    @property
    def pitch_row_mm(self) -> float:
        return _norm(self.row_vec_mm)

    @property
    def pitch_col_mm(self) -> float:
        return _norm(self.col_vec_mm)

    @property
    def center_mm(self) -> Tuple[float, float]:
        hr, hc = (self.rows - 1) / 2.0, (self.cols - 1) / 2.0
        return _add(self.p00_mm, _add(_scale(self.row_vec_mm, hr), _scale(self.col_vec_mm, hc)))

    def cell_xy(self, row: int, col: int) -> Tuple[float, float]:
        """某格凹槽中心的世界 XY。"""
        return _add(self.p00_mm, _add(_scale(self.row_vec_mm, row), _scale(self.col_vec_mm, col)))

    def edge_dir_world(self) -> Tuple[float, float]:
        """吸取边缘方向的单位向量（世界系）：托盘系 edge_dir_deg 旋转 θ 后。"""
        a = math.radians(self.theta_deg + self.edge_dir_deg)
        return (math.cos(a), math.sin(a))

    def edge_target(self, wafer_center: Tuple[float, float],
                    wafer_r_mm: float) -> Optional[Tuple[float, float]]:
        """边缘吸取点（世界 XY）。吸盘比硅片还宽（扣不出边缘余量）→ None，上层停手。"""
        keep = wafer_r_mm - self.cup_r_mm - self.edge_margin_mm
        if keep <= 0.0:
            return None
        d = self.edge_dir_world()
        return (wafer_center[0] + keep * d[0], wafer_center[1] + keep * d[1])

    # ---- IO ----
    def to_dict(self) -> dict:
        return {
            "_schema": "scara_tray_frame/v1",
            "_note": ("右盘世界坐标系（放料侧，2026-07-31 新流程）。由盘角点经 J3 手眼换算生成 "
                      "(tools/scara_tray_calib.py)；盘一动就要重标。行/列向量是相邻格的世界位移。"
                      "edge_dir_deg：吸取边缘方向（托盘系，0=+col）。z 全部是 readall 裸读数帧。"),
            "p00_mm": [round(self.p00_mm[0], 4), round(self.p00_mm[1], 4)],
            "row_vec_mm": [round(self.row_vec_mm[0], 4), round(self.row_vec_mm[1], 4)],
            "col_vec_mm": [round(self.col_vec_mm[0], 4), round(self.col_vec_mm[1], 4)],
            "z_contact_mm": round(self.z_contact_mm, 4),
            "z_safe_mm": round(self.z_safe_mm, 4),
            "imaging_j3_mm": round(self.imaging_j3_mm, 4),
            "edge_dir_deg": round(self.edge_dir_deg, 2),
            "cup_r_mm": round(self.cup_r_mm, 3),
            "edge_margin_mm": round(self.edge_margin_mm, 3),
            "rows": self.rows, "cols": self.cols,
            "theta_deg_derived": round(self.theta_deg, 4),
            "pitch_row_mm_derived": round(self.pitch_row_mm, 4),
            "pitch_col_mm_derived": round(self.pitch_col_mm, 4),
            "center_mm_derived": [round(self.center_mm[0], 4), round(self.center_mm[1], 4)],
            "z_frame": self.z_frame,
            "taught_at": self.taught_at,
        }

    @classmethod
    def from_dict(cls, d) -> "TrayFrame":
        def _f2(key, default=(0.0, 0.0)):
            v = d.get(key) if isinstance(d, dict) else None
            if isinstance(v, (list, tuple)) and len(v) == 2:
                try:
                    return (float(v[0]), float(v[1]))
                except (TypeError, ValueError):
                    pass
            return default

        def _f(key, default=0.0):
            try:
                v = d.get(key)
                return default if v is None else float(v)
            except (TypeError, ValueError):
                return default

        if not isinstance(d, dict):
            d = {}
        return cls(
            p00_mm=_f2("p00_mm"), row_vec_mm=_f2("row_vec_mm", (24.2, 0.0)),
            col_vec_mm=_f2("col_vec_mm", (0.0, 24.6)),
            z_contact_mm=_f("z_contact_mm"), z_safe_mm=_f("z_safe_mm"),
            imaging_j3_mm=_f("imaging_j3_mm"), edge_dir_deg=_f("edge_dir_deg"),
            cup_r_mm=_f("cup_r_mm"), edge_margin_mm=_f("edge_margin_mm", 1.0),
            rows=int(_f("rows", 6)), cols=int(_f("cols", 6)),
            z_frame=str(d.get("z_frame", "")), taught_at=str(d.get("taught_at", "")),
        )

    @classmethod
    def from_corners(cls, p00: Tuple[float, float], p05: Tuple[float, float],
                     p50: Tuple[float, float], p55: Optional[Tuple[float, float]] = None,
                     **kw) -> "TrayFrame":
        """三个角点定全盘：(0,0)/(0,5)/(5,0)；给第四角点 (5,5) 则回代校验，超差拒绝。

        角点是「相机对准该角格凹槽中心」经手眼换算的世界坐标。不正交 >2.3° 也拒绝
        （多半是角点点错或格号认错 —— 那类错误是 ~24mm 量级，交接 §5 坑3）。
        """
        rows = int(kw.get("rows", 6))
        cols = int(kw.get("cols", 6))
        rv = _scale(_sub(p50, p00), 1.0 / (rows - 1))
        cv = _scale(_sub(p05, p00), 1.0 / (cols - 1))
        cosang = abs(rv[0] * cv[0] + rv[1] * cv[1]) / max(1e-9, _norm(rv) * _norm(cv))
        if cosang > MAX_ORTHO:
            raise ValueError(f"行/列向量不正交（|cos|={cosang:.4f} > {MAX_ORTHO}，"
                             f"夹角 {math.degrees(math.acos(min(1.0, cosang))):.1f}°）——"
                             f"检查角点是不是点错了凹槽")
        if p55 is not None:
            pred = _add(p00, _add(_scale(rv, rows - 1), _scale(cv, cols - 1)))
            resid = _norm(_sub(pred, p55))
            if resid > MAX_CORNER_RESID_MM:
                raise ValueError(f"第四角点回代残差 {resid:.2f}mm > {MAX_CORNER_RESID_MM}mm —— "
                                 f"网格不是平面平行四边形（角点/格号有误？），拒绝写盘")
        return cls(p00_mm=p00, row_vec_mm=rv, col_vec_mm=cv, **kw)


def load_tray_frame(calib_dir=None) -> TrayFrame:
    base = Path(calib_dir) if calib_dir else _CALIB_DIR
    try:
        return TrayFrame.from_dict(json.loads((base / TRAY_FRAME_FILE).read_text("utf-8")))
    except Exception:                                   # noqa: BLE001
        return TrayFrame(p00_mm=(0.0, 0.0), row_vec_mm=(24.2, 0.0), col_vec_mm=(0.0, 24.6))


def save_tray_frame(frame: TrayFrame, calib_dir=None) -> Path:
    base = Path(calib_dir) if calib_dir else _CALIB_DIR
    p = base / TRAY_FRAME_FILE
    p.write_text(json.dumps(frame.to_dict(), ensure_ascii=False, indent=2), "utf-8")
    back = load_tray_frame(base)
    if abs(back.p00_mm[0] - frame.p00_mm[0]) > 1e-9:
        raise IOError(f"写盘后回读校验失败：{p}")
    return p


# ====================================================================== #
#  36 格回放关节生成
# ====================================================================== #
def gen_cells_json(frame: TrayFrame, rz_deg: Optional[float] = None,
                   taught_at: Optional[str] = None) -> Tuple[dict, List[str]]:
    """由 TrayFrame + 精确 IK 生成 cells json（v2 schema，joints 仍是回放主键）。

    - rz_deg=None → 末端朝向取盘 θ（整盘统一朝向；硅片是圆片，朝向对凹槽无约束，
      统一朝向让相邻格臂姿连续、回放路径可预期）。给定则强制该朝向。
    - 分支选择带连续性（蛇形扫格，上一格的 joints 作下一格的 ref），避免行间臂型跳变
      （交接 §2 注意1 的 rz 按行分组就是这么来的；这次由求解器主动保持一致）。
    - 返回 (json_dict, warnings)：任何一格 IK 无解 → ValueError（半张表比没有更危险）。
    """
    ts = taught_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rz = frame.theta_deg if rz_deg is None else float(rz_deg)
    warns: List[str] = []
    cells: Dict[str, dict] = {}
    ref: Optional[List[float]] = None
    # 蛇形遍历：偶数行 col 0→5，奇数行 5→0，相邻目标点总差一个 pitch，连续性最好
    for r in range(frame.rows):
        cols = range(frame.cols) if r % 2 == 0 else range(frame.cols - 1, -1, -1)
        for c in cols:
            x, y = frame.cell_xy(r, c)
            ok, why = kinematics.reach_ok(x, y)
            if not ok:
                raise ValueError(f"格({r},{c}) {why} —— 整表不生成")
            if why:
                warns.append(f"格({r},{c}) {why}")
            j = kinematics.solve_joints(x, y, frame.z_contact_mm, rz_deg=rz, ref_joints=ref)
            if j is None:
                raise ValueError(f"格({r},{c}) IK 无解（臂展 {kinematics.reach_mm(x, y):.1f}mm）"
                                 f" —— 整表不生成")
            ref = j
            cells[f"{r},{c}"] = {
                "joints": j,
                "world_xy": [round(x, 4), round(y, 4)],
                "rz_deg": round(rz, 4),
                "is_pick_pose": True,          # joints[2] 就是接触 Z（凹槽底面）
                "z_pick": round(frame.z_contact_mm, 4),
                "source": "grid_solved",        # 非逐格示教：手眼+网格+IK 解算
                "src": "handeye_ik",
                "taught_at": ts,
            }
    doc = {
        "_schema": "scara_right_tray_cells/v2",
        "_note": ("★2026-07-31 新流程：本表由 scara_tray_frame.json + 精确 IK 解算生成"
                  "（tools/scara_tray_calib.py solve 写入），**不是逐格示教**。joints 仍是回放"
                  "主键；world_xy/rz_deg 存档与校验用。本轮右盘是**放料侧**：接触 Z 与取片同"
                  "一个凹槽底面。取片时的硅片真实位置由 J3 相机逐格实拍精调（边缘吸取），"
                  "不依赖本表的 XY 绝对精度到亚毫米。"),
        "z_frame": frame.z_frame,
        "z_offset_mm": 0.0,
        "z_pick_default": None,
        "z_safe": round(frame.z_safe_mm, 4),
        "rows": frame.rows, "cols": frame.cols,
        "blocked_cells": [],
        "cells": cells,
    }
    return doc, warns


def write_cells_json(frame: TrayFrame, calib_dir=None,
                     rz_deg: Optional[float] = None) -> Tuple[Path, List[str]]:
    """生成并写盘（带时间戳备份 + 回读校验 + FK 复核）。"""
    import shutil
    base = Path(calib_dir) if calib_dir else _CALIB_DIR
    p = base / CELLS_FILE
    doc, warns = gen_cells_json(frame, rz_deg=rz_deg)
    # FK 复核：每个 joints 正算回 world，必须与 world_xy 一致（IK 写错这里能抓住）
    for key, cell in doc["cells"].items():
        x, y = kinematics.fk_wrist(cell["joints"][0], cell["joints"][1])
        err = _norm(_sub((x, y), cell["world_xy"]))
        if err > 0.01:
            raise ValueError(f"格{key} FK 复核失败：回代误差 {err:.4f}mm")
    if p.exists():
        bak = p.with_suffix(p.suffix + f".bak_{datetime.now():%Y%m%d_%H%M%S}")
        shutil.copy2(p, bak)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), "utf-8")
    back = json.loads(p.read_text("utf-8"))
    if len(back.get("cells", {})) != frame.rows * frame.cols:
        raise IOError(f"写盘后回读校验失败：格数 {len(back.get('cells', {}))} != {frame.rows * frame.cols}")
    return p, warns


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # 合成盘：仿旧盘几何（col 向 −X、row 向 −Y，最远角在臂展内）
    th = math.radians(179.3)
    c00 = (210.0, 320.0)
    rv = (24.2 * math.cos(th + math.pi / 2), 24.2 * math.sin(th + math.pi / 2))
    cv = (24.6 * math.cos(th), 24.6 * math.sin(th))
    p00, p05 = c00, _add(c00, _scale(cv, 5))
    p50 = _add(c00, _scale(rv, 5))
    p55 = _add(c00, _add(_scale(rv, 5), _scale(cv, 5)))
    fr = TrayFrame.from_corners(p00, p05, p50, p55,
                                z_contact_mm=-62.0, z_safe_mm=-40.0, imaging_j3_mm=-40.0,
                                cup_r_mm=3.0, edge_dir_deg=0.0, z_frame="synthetic")
    print(f"θ={fr.theta_deg:.3f}°  行距={fr.pitch_row_mm:.3f}  列距={fr.pitch_col_mm:.3f}"
          f"  中心=({fr.center_mm[0]:.2f},{fr.center_mm[1]:.2f})")
    assert abs(fr.theta_deg - 179.3) < 1e-6 and abs(fr.pitch_row_mm - 24.2) < 1e-6

    # 边缘吸取：15mm 硅片 + R3 吸盘 + 1mm 余量 → 离圆心 3.5mm
    t = fr.edge_target((100.0, 100.0), 7.5)
    assert t is not None and abs(_norm(_sub(t, (100.0, 100.0))) - 3.5) < 1e-9
    assert fr.edge_target((0.0, 0.0), 3.5) is None, "吸盘比硅片宽必须拒"

    # 格位生成 + FK 一致性 + 连续性
    doc, warns = gen_cells_json(fr)
    assert len(doc["cells"]) == 36
    prev = None
    max_jump = 0.0
    for r in range(6):
        for c in range(6):
            j = doc["cells"][f"{r},{c}"]["joints"]
            x, y = kinematics.fk_wrist(j[0], j[1])
            # 关节值保留 4 位小数 → FK 回代有 ~1e-4mm 量级量化误差，阈值 1μm 足够
            assert _norm(_sub((x, y), fr.cell_xy(r, c))) < 1e-3
            rz = kinematics.rz_of(j[0], j[1], j[3])
            assert abs(rz - fr.theta_deg) < 0.01, f"({r},{c}) rz 不一致: {rz}"
            if prev is not None:
                max_jump = max(max_jump, abs(j[0] - prev[0]), abs(j[1] - prev[1]))
            prev = j
    print(f"36 格生成 OK；相邻格关节最大跳变 {max_jump:.2f}°；警告 {len(warns)} 条")
    for w in warns:
        print("  " + w)
    # 不正交/第四点超差必须拒
    try:
        TrayFrame.from_corners(p00, p05, _add(p50, (30.0, 0.0)))
        raise AssertionError("不正交没拦住")
    except ValueError as e:
        print(f"正交拦截 OK: {str(e)[:60]}...")
    print("tray_grid 自检 OK")
