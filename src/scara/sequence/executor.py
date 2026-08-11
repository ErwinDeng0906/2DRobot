"""SCARA 送检 · 序列执行器。

逐步执行 ScaraStep：dry_run 只打印不动手；任一步失败立即停手 + 回滚（关泵破真空、停手待人工）；
取片二次确认失败按预算重试整段 pick。绝对 Z/J4 目标由「读 pose/joints 绝对值 + 相对步进到位」
实现（SCARA 无绝对定位）。硬件全靠依赖注入，可全 Fake 单测。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

from scara.sequence.steps import (
    ScaraStep, ScaraStepKind, build_inspect_one, build_pick_from_right,
    build_place_to_micro, build_place_to_waffle, build_transfer_one,
)
from scara.vision.tray_wafers import TrayWaferConfig, detect_tray_wafers, picked_one


@dataclass
class StepResult:
    step_id: str
    ok: bool
    reason: str = ""
    info: dict = field(default_factory=dict)


@dataclass
class ExecConfig:
    verify_retries: int = 2         # 取片二次确认失败，整段 pick 最多重试次数
    stage_tolerance_mm: float = 0.05  # 显微镜 XY 台到位回读容差（= micro_load_pos.tolerance_mm）
    scan_timeout_s: float = 1800.0  # 全自动扫描最长等待
    vent_on_rollback: bool = False
    """失败回滚时是否破真空。**默认 False（不松手）** —— 见 _rollback 的 docstring：
    2026-07-28 三次掉片全是回滚破真空造成的，而失败常发生在正吊着片的时候。"""
    tray_region: Optional[tuple] = None
    """右台相机里托盘的大致范围 (x0,y0,x1,y1)，只用来排掉盘外干扰，精度无所谓。

    None = 整帧。前后对比对静态误检本就免疫（两帧都在、自动抵消），所以这是可选项。
    """
    arm_clear_tol_deg: float = 0.5   # 撤臂门：J1/J2/J4 与 micro_clear 的逐轴容差(度)
    arm_clear_tol_mm: float = 0.5    # 撤臂门：J3(直动 Z) 的容差(mm)
    arm_clear_required: bool = True
    """撤臂门是否强制。**只有"机械臂根本不在现场"这一种情况才允许关**（手动放样跑纯扫描段）。

    ★ 关掉它不是"跳过检查"，是"改由操作员声明"—— 代码这边就没有任何证据了。
      所以调用方必须满足两条：① SCARA 确实连不上（连得上就必须用真读数，有证据不用证据
      是最坏的一种）；② 命令行显式声明。绝不允许为了让流程跑通而默认关掉。
    生产链路（scara_inspect_run.py 全链）永远用 True。
    """
    focus_tolerance_mm: float = 0.01  # 焦点轴到位回读容差(mm)
    """比台子的 0.05mm 严一个量级：焦深就这么点，焦点差 0.05mm 在 10x 下已经是糊的。
    这是"命令有没有执行到位"的容差，不是"对没对上焦"的判据（后者由 WDI 状态负责）。"""
    objective_timeout_s: float = 30.0  # 切物镜后轮询回读的最长等待
    scan_save_root: Optional[str] = None
    """扫描图保存根目录。None = 用 data_pipeline.resolve_data_root()（仓库外的 capture/）。

    ★ 保存目录不是"顺便存个档"：`ScanWorker._raise_if_save_failed` 只有在设了
    save_directory 且一张都没成功时才会抛错。**不设它，一次什么都没采到的扫描会静默成功。**
    """


class ScaraSequenceExecutor:
    def __init__(self, motion: Any, grabber: Any, suction: Any, servo: Any, calib: Any,
                 cfg: Optional[ExecConfig] = None, dry_run: bool = False, log=None,
                 micro: Any = None):
        self.motion = motion
        self.grabber = grabber
        self.suction = suction
        self.servo = servo
        self.calib = calib
        self.micro = micro          # MicroscopeBackend：XY 台到装样位 + 全自动扫描
        self.cfg = cfg or ExecConfig()
        self.dry_run = dry_run
        self._log = log or (lambda _m: None)
        self._wafer_yaw = 0.0      # 取片伺服得到的 yaw，放片 J4 补偿用
        self._tray_before = None   # SNAPSHOT_TRAY 存下的吸片前扫描结果
        self._tray_cfg = TrayWaferConfig.from_dict(self.calib.detect)
        self.rolled_back = False

    # ------------------------------------------------------------------ #
    #  高层：用户定义的五步全链（取片 → 放样 → 扫描）
    # ------------------------------------------------------------------ #
    def inspect_one(self, cell) -> List[StepResult]:
        """一片完整送检。取片段带二次确认重试，放样/扫描段一次成型。

        与 `pick_and_place` 的区别：那条走的是华夫盒 preset 老路径（占位标定，从没标过），
        这条走**示教关节回放 + 显微镜 RPC**，是 2026-07-28 用户定义的五步流程。
        """
        steps, why = build_inspect_one(cell, self.calib.tray, self.calib.waypoints,
                                       self.calib.pump,
                                       recipe=getattr(self.calib, "scan_recipe", None))
        if why:
            self._log(f"[seq] 送检全链未生成: {why}")
            return [StepResult("inspect_one", False, why)]
        n_pick = len(build_pick_from_right(cell, self.calib.tray, self.calib.pump)[0])
        pick_steps, rest = steps[:n_pick], steps[n_pick:]

        results: List[StepResult] = []
        picked = False
        for attempt in range(self.cfg.verify_retries + 1):
            rs = self.run(pick_steps)
            results += rs
            if rs and rs[-1].ok:
                picked = True
                break
            self._log(f"[seq] 取片第{attempt + 1}次未确认" +
                      ("，重试" if attempt < self.cfg.verify_retries else "，预算耗尽"))
            if attempt < self.cfg.verify_retries:
                # 重试前必须泄压：回滚不再破真空（见 _rollback），所以这一刻手里可能还
                # 攥着片；带着它去拍下一轮存底，基线就是错的。此刻机械臂就在该格正上方，
                # 松手也是掉回它自己那一格，是最安全的泄压时机。
                try:
                    self.suction.off()
                except Exception:      # noqa: BLE001
                    pass
        if not picked:
            results.append(StepResult("inspect_one", False, "pick_verify_exhausted"))
            return results
        results += self.run(rest)
        return results

    # ------------------------------------------------------------------ #
    #  高层：2026-07-31 新流程 —— 转运+表征全链（剥离台→显微镜→扫描→取回→右盘）
    # ------------------------------------------------------------------ #
    def transfer_one(self, cell) -> List[StepResult]:
        """一片完整转运+表征：剥离台取片 → 显微镜放样 → 扫描 → 取回 → 放右盘 cell。

        与 `inspect_one` 的区别：那条是旧流程（右盘取片送检），这条是 2026-07-31 用户
        拍板的新流程（交接 v6）。Phase 1 不带吸取确认（剥离台没配前后对比相机、
        J3 下移相机未装），所以不设取片重试预算 —— 任一步失败即停 + 回滚
        （回滚保持真空，见 _rollback），持片确认等 J3 相机到位后（Phase 2）补。
        """
        steps, why = build_transfer_one(cell, self.calib.tray, self.calib.waypoints,
                                        self.calib.pump,
                                        recipe=getattr(self.calib, "scan_recipe", None))
        if why:
            self._log(f"[seq] 转运全链未生成: {why}")
            return [StepResult("transfer_one", False, why)]
        return self.run(steps)

    # ------------------------------------------------------------------ #
    #  高层：一次完整送检（取片带二次确认重试 → 放片）
    # ------------------------------------------------------------------ #
    def pick_and_place(self, cell) -> List[StepResult]:
        results: List[StepResult] = []
        picked = False
        # Z 未标定 → 一步都不生成，直接停手（绝不用 0.0 兜底，那会把吸盘怼进盘里）
        steps, why = build_pick_from_right(cell, self.calib.tray, self.calib.pump)
        if why:
            self._log(f"[seq] 取片步骤未生成: {why}")
            return [StepResult("pick_and_place", False, why)]
        for attempt in range(self.cfg.verify_retries + 1):
            rs = self.run(steps)
            results += rs
            if rs and rs[-1].ok:
                picked = True
                break
            self._log(f"[seq] 取片第{attempt + 1}次未确认" +
                      ("，重试" if attempt < self.cfg.verify_retries else "，预算耗尽"))
            if attempt < self.cfg.verify_retries:
                # 重试前必须泄压：回滚不再破真空（见 _rollback），所以这一刻手里可能还
                # 攥着片；带着它去拍下一轮存底，基线就是错的。此刻机械臂就在该格正上方，
                # 松手也是掉回它自己那一格，是最安全的泄压时机。
                try:
                    self.suction.off()
                except Exception:      # noqa: BLE001
                    pass
        if not picked:
            results.append(StepResult("pick_and_place", False, "pick_verify_exhausted"))
            return results
        results += self.run(build_place_to_waffle(self.calib.waffle, self.calib.heights, self.calib.pump))
        return results

    # ------------------------------------------------------------------ #
    #  逐步执行（失败即停 + 回滚）
    # ------------------------------------------------------------------ #
    def run(self, steps: List[ScaraStep]) -> List[StepResult]:
        out: List[StepResult] = []
        for st in steps:
            r = self._exec(st)
            out.append(r)
            if not r.ok:
                self._rollback(st, r)
                break
        return out

    def _exec(self, st: ScaraStep) -> StepResult:
        self._log(f"[seq] {st.id}: {st.desc}" + (" [dry_run]" if self.dry_run else ""))
        if self.dry_run:
            return StepResult(st.id, True, "dry_run")
        try:
            k = st.kind
            if k == ScaraStepKind.SERVO_PICK:
                return self._servo_pick(st)
            if k == ScaraStepKind.GOTO_CELL:
                return self._goto_cell(st)
            if k == ScaraStepKind.GOTO_WAYPOINT:
                return self._goto_waypoint(st)
            if k == ScaraStepKind.MICRO_STAGE_TO_LOAD:
                return self._micro_stage(st)
            if k == ScaraStepKind.MICRO_SET_OBJECTIVE:
                return self._micro_set_objective(st)
            if k == ScaraStepKind.MICRO_STAGE_TO_START:
                return self._micro_stage_to_start(st)
            if k == ScaraStepKind.MICRO_FOCUS_TO_START:
                return self._micro_focus_to_start(st)
            if k == ScaraStepKind.MICRO_SCAN:
                return self._micro_scan(st)
            if k in (ScaraStepKind.DESCEND, ScaraStepKind.DESCEND_PLACE, ScaraStepKind.LIFT):
                return self._move_z(st)
            if k == ScaraStepKind.SUCTION_ON:
                return self._suction_on(st)
            if k == ScaraStepKind.SNAPSHOT_TRAY:
                return self._snapshot_tray(st)
            if k == ScaraStepKind.VERIFY_PICK:
                return self._verify_pick(st)
            if k == ScaraStepKind.GOTO_PRESET:
                return self._goto_preset(st)
            if k == ScaraStepKind.J4_COMPENSATE:
                return self._j4(st)
            if k == ScaraStepKind.SUCTION_OFF_VENT:
                return self._vent(st)
            if k == ScaraStepKind.WAIT:
                self.suction.settle_wait(st.seconds)
                return StepResult(st.id, True)
            return StepResult(st.id, False, "unknown_kind")
        except Exception as e:  # noqa: BLE001
            return StepResult(st.id, False, f"exception:{type(e).__name__}:{e}")

    def _roi(self, cell):
        return self.calib.rois.get(tuple(cell)) if cell else None

    def _servo_pick(self, st: ScaraStep) -> StepResult:
        res = self.servo.servo_to_wafer(self._roi(st.cell))
        if res.ok:
            self._wafer_yaw = res.wafer_yaw_deg
            return StepResult(st.id, True, "converged", {"wafer_yaw_deg": res.wafer_yaw_deg})
        return StepResult(st.id, False, f"servo:{res.reason}")

    def _goto_cell(self, st: ScaraStep) -> StepResult:
        """关节回放到某格取片位的**正上方**（J1/J2/J4 用示教值，J3 停在 z_safe）。

        ★ 只能用 `replay_joints_for`（已叠加 z_offset_mm），绝不能用 `TrayCell.joints` ——
          那是磁盘原值、只供审计，直接下发等于按垫高前的深度下刀。

        ★ J3 为什么不直接回放到示教值：示教位本身就是**接触位**（is_pick_pose=true），
          直接回放等于在横移的最后一刻才抬手，且后面的 DESCEND 会变成零位移的空步。
          停在 z_safe、把下刀交给 DESCEND 独立一步，好处有三：
            · 与显微镜侧「上方高点 → 竖直下探」完全对称，一套心智模型；
            · 下刀是**单独一条**指令，出事时能精确指认是哪一步；
            · 横移全程都在安全高度（plan_joint_order 会把 Z 上升排在最前）。
          z_safe 拿不到时退回示教 J3（此时 build_pick_from_right 早已因
          z_safe_uncalibrated 一步都不生成，走不到这里；留作双保险）。
        """
        r, c = int(st.cell[0]), int(st.cell[1])
        target = self.calib.tray.replay_joints_for(r, c)
        if target is None:
            return StepResult(st.id, False, f"cell_joints_missing:{r},{c}")
        target = list(target)
        z_safe = self.calib.tray.z_safe_for()
        if z_safe is not None:
            target[2] = float(z_safe)
        ok = self.motion.goto_joints(target)
        return StepResult(st.id, bool(ok), "" if ok else "goto_cell_not_reached",
                          {"target_joints": target})

    def _goto_waypoint(self, st: ScaraStep) -> StepResult:
        target = self.calib.waypoints.joints_of(st.waypoint)
        if not target:
            return StepResult(st.id, False, f"waypoint_missing:{st.waypoint}")
        ok = self.motion.goto_joints(target)
        return StepResult(st.id, bool(ok), "" if ok else f"waypoint_not_reached:{st.waypoint}",
                          {"target_joints": list(target)})

    def _micro_stage(self, st: ScaraStep) -> StepResult:
        """显微镜 XY 台到装样位 —— 必须回读校验（见 RpcMicroscope.stage_goto 的注释）。"""
        if self.micro is None:
            return StepResult(st.id, False, "no_microscope_backend")
        xy = getattr(self.calib.waypoints, "micro_load_xy_mm", None)
        if xy is None:
            return StepResult(st.id, False, "micro_load_pos_uncalibrated")
        tol = float(self.cfg.stage_tolerance_mm)
        ok, why = self.micro.stage_goto(float(xy[0]), float(xy[1]), tol)
        return StepResult(st.id, bool(ok), why,
                          {"xy_mm": [xy[0], xy[1]], "tolerance_mm": tol})

    # ------------------------------------------------------------------ #
    #  ★ 撤臂门：台子动之前，机械臂必须已经不在显微镜区
    # ------------------------------------------------------------------ #
    def _assert_arm_clear(self) -> Tuple[bool, str]:
        """当前关节是否已经回到 `micro_clear`。返回 (通过?, 原因)。

        ★ 挡的是什么：放样抬回 `above_micro` 之后，吸盘仍停在华夫盒正上方、只高 15.8mm，
          而 SCARA 是从物镜**底下**伸进去的。此时台子一动（去扫描起点是十几毫米的位移），
          等于让华夫盒带着片从吸盘底下横扫出去。

        ★ 这道门证明的**只是「撤离动作确实执行了」**（挡住指令没下发、走一半、或步骤被
          --skip 之类的开关跳过），它**不**证明 `micro_clear` 这个位姿本身几何安全 ——
          后者由「操作员 jog 示教 + 首次慢速空载验证」建立。
          把这两件事分清楚，是 lessons 2026-07-27「机械臂停在被命令的位置不是验证」那条：
          关节回读来自命令链内部，衡量的是执行，不是位置对不对。
        """
        if not self.cfg.arm_clear_required:
            # 由操作员声明"臂不在现场"。这里**没有任何证据**，只有一句声明 ——
            # 所以要在日志里留下痕迹，别让它在事后看起来像"检查通过了"。
            self._log("[seq] ⚠ 撤臂门已按 arm_clear_required=False 关闭 —— "
                      "本次没有任何读数证明臂不在显微镜区，责任在操作员的现场目视")
            return True, ""
        target = self.calib.waypoints.joints_of("micro_clear")
        if not target:
            return False, "arm_clear_waypoint_missing:micro_clear（没有撤离位就无从判断臂是否已撤出）"
        try:
            cur = list(self.motion.get_joints())
        except Exception as e:                                   # noqa: BLE001
            return False, f"arm_clear_joints_unreadable:{type(e).__name__}:{e}"
        if len(cur) != 4:
            return False, f"arm_clear_joints_bad:{cur!r}"
        # J3 是直动 Z 轴（mm），J1/J2/J4 是角度（度）—— 容差不能混用同一个数
        tols = [self.cfg.arm_clear_tol_deg, self.cfg.arm_clear_tol_deg,
                self.cfg.arm_clear_tol_mm, self.cfg.arm_clear_tol_deg]
        bad = [f"J{i+1} 实到 {cur[i]:.4f} vs 撤离位 {target[i]:.4f}（差 {abs(cur[i]-target[i]):.4f} > {tols[i]}）"
               for i in range(4) if abs(cur[i] - float(target[i])) > tols[i]]
        if bad:
            # 注意别写成相邻字面量 + .join —— 那样前缀会变成分隔符而不是前缀
            return False, ("arm_not_clear：机械臂还没撤出显微镜区，拒绝移动台子/开扫。"
                           + " | ".join(bad))
        return True, ""

    def _micro_stage_to_start(self, st: ScaraStep) -> StepResult:
        """显微镜 XY 台移到扫描起点（= 配方 corner_1），带到位回读校验。

        为什么要单独走这一步、而不是让扫描引擎自己去：引擎的首次 XY 移动确实已经用
        `move_absolute_gentle`，但绝对移动路径把目标值**直接写进位置缓存、不回读**
        （microscope_controller.py:317-319 / :358-359）。而这一段是从**行程边界上的装样位**
        出发的大位移，一旦被限位截断或拒绝，软件会认为自己到位了，
        然后整轮扫描在错误的地方进行、全程不报错。
        """
        if self.micro is None:
            return StepResult(st.id, False, "no_microscope_backend")
        ok, why = self._assert_arm_clear()
        if not ok:
            return StepResult(st.id, False, why)
        recipe = getattr(self.calib, "scan_recipe", None)
        if recipe is None or not recipe.is_calibrated:
            return StepResult(st.id, False, "scan_area_uncalibrated")
        x, y = recipe.corner_1_mm
        tol = float(self.cfg.stage_tolerance_mm)
        ok, why = self.micro.stage_goto(float(x), float(y), tol)
        return StepResult(st.id, bool(ok), why,
                          {"xy_mm": [x, y], "tolerance_mm": tol})

    def _micro_set_objective(self, st: ScaraStep) -> StepResult:
        """把物镜切到配方要求的倍率（`presets/<preset>.json` 的 objective），带回读校验。

        ★ 为什么必须有这一步：配方按 preset 的 FOV 算 tile 步距。转塔实际在别的倍率上时，
          步距全错 —— 而每一张图**看着都正常**、扫描**照样报成功**，只有整片覆盖是错的。
          这类故障没有任何运行时症状，只能在开扫前比一次。

        ★ 为什么排在台子移动之前：高倍物镜工作距离更短，而换镜内部会 suspend 焦点限位并
          驱动焦点轴。此刻焦点还停在装样位那个"离样品最远"的 Z，是余量最大的时候。

        撤臂门同样要过：换镜会驱动焦点轴（Z 方向），臂还在盒子上方时不能动。
        """
        if self.micro is None:
            return StepResult(st.id, False, "no_microscope_backend")
        ok, why = self._assert_arm_clear()
        if not ok:
            return StepResult(st.id, False, why)
        recipe = getattr(self.calib, "scan_recipe", None)
        if recipe is None:
            return StepResult(st.id, False, "scan_recipe_missing")
        want = recipe.expected_objective
        if want is None:
            # 读不到 preset 就不知道该切到几号 —— 绝不"保持现状继续扫"，那等于放弃这道门
            return StepResult(st.id, False,
                              f"objective_unknown：读不到 presets/{recipe.preset_name}.json 的 "
                              f"objective，无法确认该用哪个物镜")
        ok, why = self.micro.objective_goto(int(want), float(self.cfg.objective_timeout_s))
        return StepResult(st.id, bool(ok), why,
                          {"objective": int(want), "preset": recipe.preset_name})

    def _micro_focus_to_start(self, st: ScaraStep) -> StepResult:
        """起始对焦：送到示教 Z（回读校验）→ WDI 精修 → 漂移断言。

        2026-07-29 用户拍板「存示教Z兜底 + WDI微调，**两道都要**」，这里是两道的落点：

          ① 兜底：把焦点从装样位那个 5.000mm（离样品最远的一侧）送回示教时的合焦 Z。
             不做这一步，逐点自动对焦的 ±1mm 搜索窗根本够不着样品面，整轮扫描全糊 ——
             而糊的图照样存盘、照样算采集成功。
          ② 精修：WDI 再对一次，吸收放样后样品高度的实际偏差。

        漂移断言是第三件事，也是最容易被略过的一件：WDI 可能对到**别的面**上
        （盒盖、台面、反射面）。此时 in_focus/in_range 都会是真，只有"离示教 Z 太远"
        这一个信号能发现它。超差就停手 —— 对错了面扫出来的一整轮图，事后完全无法补救。
        """
        if self.micro is None:
            return StepResult(st.id, False, "no_microscope_backend")
        ok, why = self._assert_arm_clear()
        if not ok:
            return StepResult(st.id, False, why)
        recipe = getattr(self.calib, "scan_recipe", None)
        if recipe is None or not recipe.focus_is_calibrated:
            return StepResult(st.id, False, "scan_focus_uncalibrated")
        target = float(recipe.focus_start_mm)

        # 余量按【实读】限位算，不用示教时存档的那份：限位是可以被人改的（2026-07-26 就为
        # 自动对焦放宽过一次），拿旧值判断等于在验证一个可能已经不成立的前提。
        try:
            limits = self.micro.call("get_focus_limits")
        except Exception as e:                                   # noqa: BLE001
            return StepResult(st.id, False, f"focus_limits_unreadable:{type(e).__name__}:{e}")
        ok, why = recipe.focus_start_margin_ok(limits)
        if not ok:
            return StepResult(st.id, False, why)

        ok, why = self.micro.focus_goto(target, float(self.cfg.focus_tolerance_mm))
        if not ok:
            return StepResult(st.id, False, why, {"target_mm": target})
        info = {"target_mm": target, "limits_mm": limits, "wdi_refine": recipe.focus_wdi_refine}
        if not recipe.focus_wdi_refine:
            info["final_mm"] = target
            return StepResult(st.id, True, "", info)

        ok, why, z = self.micro.autofocus_refine()
        info["final_mm"] = z
        if not ok:
            return StepResult(st.id, False, f"{why}（示教 Z {target:.4f}mm 已到位，"
                                            f"是 WDI 精修这一步没过）", info)
        if z is None:
            return StepResult(st.id, False, "autofocus_final_z_unreadable", info)
        drift = abs(float(z) - target)
        info["drift_mm"] = drift
        if drift > float(recipe.focus_max_drift_mm):
            return StepResult(st.id, False,
                              f"focus_drift_too_large：WDI 精修后 {float(z):.4f}mm 与示教 "
                              f"{target:.4f}mm 差 {drift:.4f}mm > 上限 "
                              f"{float(recipe.focus_max_drift_mm):g}mm —— 多半对到了别的面"
                              f"（盒盖/台面/反射面）上。此时 in_focus/in_range 都会是真，"
                              f"只有这条能发现它", info)
        self._log(f"[seq] 起始对焦完成：示教 {target:.4f} → WDI {float(z):.4f}mm"
                  f"（漂移 {drift:.4f}mm）")
        return StepResult(st.id, True, "", info)

    def _scan_save_dir(self, cell) -> Optional[str]:
        """本轮扫描的保存目录（含时间戳与格号）。算不出来 → None，由调用方停手。"""
        recipe = getattr(self.calib, "scan_recipe", None)
        if recipe is None:
            return None
        if self.cfg.scan_save_root:
            root = Path(self.cfg.scan_save_root)
        else:
            from microscope.logic.data_pipeline import resolve_data_root
            root, _ = resolve_data_root(Path(__file__).resolve().parents[3])
        r, c = (int(cell[0]), int(cell[1])) if cell and len(cell) >= 2 else ("x", "x")
        name = recipe.save_subdir_pattern.format(
            ts=datetime.now().strftime("%Y%m%d_%H%M%S"), row=r, col=c)
        return str(root / name)

    def _micro_scan(self, st: ScaraStep) -> StepResult:
        """按配方触发全自动扫描并等到完成。

        ★ 绝不用空参数调 `start_scan` —— 那会退化成「在当前位置原地拍 3×3 共 9 张同一画面、
          不存盘、返回 True」。参数与保存目录任一算不出来，本步直接失败，不做降级。
        """
        if self.micro is None:
            return StepResult(st.id, False, "no_microscope_backend")
        ok, why = self._assert_arm_clear()
        if not ok:
            return StepResult(st.id, False, why)
        recipe = getattr(self.calib, "scan_recipe", None)
        if recipe is None:
            return StepResult(st.id, False, "scan_recipe_missing")
        ok, why = recipe.validate(getattr(self.calib.waypoints, "current_stage_frame", "") or "")
        if not ok:
            return StepResult(st.id, False, why)
        save_dir = self._scan_save_dir(st.cell)
        if not save_dir:
            return StepResult(st.id, False, "scan_save_dir_unresolved")
        params = recipe.to_scan_params(save_dir)
        if params is None:
            return StepResult(st.id, False, "scan_params_unbuildable")
        n = recipe.tile_count()
        self._log(f"[seq] 扫描 {n} 格 @ {recipe.preset_name}，"
                  f"起点 ({params['ref_x1']:.4f},{params['ref_y1']:.4f}) → "
                  f"({params['ref_x2']:.4f},{params['ref_y2']:.4f})，存 {save_dir}")
        ok, why = self.micro.scan_once(self.cfg.scan_timeout_s, params)
        return StepResult(st.id, bool(ok), why,
                          {"tiles": n, "save_directory": save_dir,
                           "preset": recipe.preset_name})

    def _move_z(self, st: ScaraStep) -> StepResult:
        """降/抬：读绝对 Z(pose[2]) → 相对步进到目标绝对 Z（SCARA 无绝对定位）。"""
        cur_z = self.motion.get_pose()[2]
        ok = self.motion.step_cart("Z", st.target_z - cur_z)
        return StepResult(st.id, bool(ok), "" if ok else "z_not_reached")

    def _suction_on(self, st: ScaraStep) -> StepResult:
        self.suction.on()
        self.suction.settle_wait(st.vac_on_s)
        return StepResult(st.id, True)

    def _snapshot_tray(self, st: ScaraStep) -> StepResult:
        """吸片前存底。与 VERIFY_PICK 在**同一机械臂姿态**下拍，遮挡才可比。"""
        scan = detect_tray_wafers(self.grabber.grab(), self._tray_cfg, self.cfg.tray_region)
        self._tray_before = scan
        if not scan.ok:
            return StepResult(st.id, False, f"snapshot_failed:{scan.reason}")
        if scan.count == 0:
            # 一片都没检出：要么相机/照明出问题，要么盘是空的。两种都不该继续吸。
            return StepResult(st.id, False, "snapshot_no_wafer")
        self._log(f"[seq] 存底：托盘上检出 {scan.count} 片")
        return StepResult(st.id, True, "", {"count": scan.count})

    def _verify_pick(self, st: ScaraStep) -> StepResult:
        """与存底比对：**恰好少一片** = 吸起成功。

        ★ 走「同姿态前后对比」而不是「查目标格 ROI」：实测托盘槽位离边有距离、
          均匀网格压不住（2026-07-28），而前后对比根本不需要知道每格在画面哪里。
          静态误检（黑板边、面包板边）在两帧里都在，会自动抵消。
        判据来自操作员：紫色/深色=硅片，反光银白=铝合金台面 → 按**饱和度**分，
        不按亮度（硅片反光时很亮、背光时几乎全黑，亮度不可靠）。
        """
        if self._tray_before is None:
            return StepResult(st.id, False, "no_snapshot")   # 没存底就无从比对
        after = detect_tray_wafers(self.grabber.grab(), self._tray_cfg, self.cfg.tray_region)
        ok, why, gone = picked_one(self._tray_before, after)
        info = {"before": self._tray_before.count, "after": after.count}
        if gone is not None:
            info["vanished_px"] = [round(gone.cx, 1), round(gone.cy, 1)]
        return StepResult(st.id, ok, "" if ok else why, info)

    def _goto_preset(self, st: ScaraStep) -> StepResult:
        ok = self.motion.goto_preset(st.preset)
        return StepResult(st.id, bool(ok), "" if ok else f"preset_unreachable:{st.preset}")

    def _j4(self, st: ScaraStep) -> StepResult:
        """放片 J4 = 华夫目标角 − 取片测的硅片 yaw + 补偿常数；再按 TCP 偏移补 XY（吸盘绕 J4 画圆）。"""
        f, tcp = self.calib.waffle, self.calib.tcp
        cur = self.motion.get_joints()[3]
        target = f.target_angle_deg - self._wafer_yaw + f.angle_offset_const
        if not self.motion.step_joint(4, target - cur):
            return StepResult(st.id, False, "j4_not_reached")
        cx, cy = tcp.cup_offset_mm
        if cx or cy:   # 吸盘中心不在 J4 轴心：转角后补 -Δ 保持中心
            a, b = math.radians(cur), math.radians(target)
            dx = (math.cos(b) * cx - math.sin(b) * cy) - (math.cos(a) * cx - math.sin(a) * cy)
            dy = (math.sin(b) * cx + math.cos(b) * cy) - (math.sin(a) * cx + math.cos(a) * cy)
            if not (self.motion.step_cart("X", -dx) and self.motion.step_cart("Y", -dy)):
                return StepResult(st.id, False, "tcp_comp_not_reached")
        return StepResult(st.id, True, "", {"target_j4_deg": target})

    def _vent(self, st: ScaraStep) -> StepResult:
        """泄压释放：关泵开阀破真空 + 等残余真空泄净（照 DUCO 教训，不等泄净会带片）。"""
        self.suction.off()
        self.suction.settle_wait(max(st.vent_wait_s, st.vent_ms / 1000.0))
        return StepResult(st.id, True)

    def _rollback(self, st: ScaraStep, r: StepResult) -> None:
        """失败即停 —— **但不松手**。

        ★ 2026-07-28 实测三次同一个事故：原实现在回滚时 `suction.off()`，而失败往往发生在
          机械臂正**吊着硅片**的时候（取片后、放下前）。一破真空，片就从 71mm 高处掉下去。
          三次掉片全是这一行造成的，没有一次是碰撞。

        「停真空防残留」这条理由只在**片已经放下**之后成立；片还在吸盘上时，
        保持真空才是安全的一侧 —— 停手的语义是「不再动」，不是「撒手」。
        真要泄压由操作员在现场判断后手动做，那时他看得见片在哪。

        `cfg.vent_on_rollback=True` 可恢复旧行为（如果哪天有"必须立刻泄压"的场景）。
        """
        self.rolled_back = True
        if getattr(self.cfg, "vent_on_rollback", False):
            self._log(f"[seq] 步骤 {st.id} 失败({r.reason}) → 回滚：破真空 + 停手待人工")
            try:
                self.suction.off()
            except Exception:    # noqa: BLE001
                pass
            return
        self._log(f"[seq] 步骤 {st.id} 失败({r.reason}) → 停手待人工。"
                  f"★真空**保持不放**（片若在吸盘上不会掉）；机械臂不自动归位。")
