"""SCARA 送检 · 取放序列步骤（数据驱动，仿 flow_steps.py 精简）。

2026-07-31 新流程（v6）：剥离台取片 → 显微镜放样 → 表征扫描 → 显微镜取回 → 放右盘存储，
一条 = `build_transfer_one`。旧流程（右盘取片→送检）= `build_inspect_one`，仍保留。
步骤是纯数据（ScaraStep），由 executor 按 kind 分派执行；本文件不碰硬件。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

# 时序常量默认（pump 标定覆盖），仿 flow_steps.py:179
VAC_ON_S = 1.2
VENT_MS = 1200.0
VENT_WAIT_S = 3.0


class ScaraStepKind(str, Enum):
    SERVO_PICK = "servo_pick"              # 视觉伺服对准右盘硅片【本方案不用，见下】
    GOTO_CELL = "goto_cell"                # ★关节回放到右盘某格的示教取片位
    GOTO_WAYPOINT = "goto_waypoint"        # ★关节回放到某个示教航点（above_micro/place_micro…）
    DESCEND = "descend"                    # 竖直降到取片高度
    SUCTION_ON = "suction_on"             # 吸真空
    LIFT = "lift"                          # 竖直抬到目标高度
    SNAPSHOT_TRAY = "snapshot_tray"       # ★吸片前拍一帧存底（与 VERIFY_PICK 同姿态）
    VERIFY_PICK = "verify_pick"           # 二次确认（与存底比对：恰好少一片=已吸起）
    GOTO_PRESET = "goto_preset"           # 到示教预设点（旧路径，scara_presets.json）
    J4_COMPENSATE = "j4_compensate"       # 放片 J4 角度补偿（+ TCP 的 XY 补偿）
    DESCEND_PLACE = "descend_place"       # 竖直降到放片高度
    SUCTION_OFF_VENT = "suction_off_vent"  # 泄压破真空释放
    MICRO_STAGE_TO_LOAD = "micro_stage_to_load"   # ★显微镜 XY 台去装样位（含到位回读校验）
    MICRO_SET_OBJECTIVE = "micro_set_objective"    # ★切物镜到配方要求的倍率（含轮询回读校验）
    MICRO_STAGE_TO_START = "micro_stage_to_start"  # ★显微镜 XY 台去扫描起点（含到位回读校验）
    MICRO_FOCUS_TO_START = "micro_focus_to_start"  # ★焦点到示教合焦 Z（回读）+ WDI 精修 + 漂移断言
    MICRO_SCAN = "micro_scan"              # ★触发显微镜全自动扫描并等完成
    WAIT = "wait"


# ★ SERVO_PICK 为什么留着却不用：2026-07-27 用户拍板改「示教回放」，相机只判断
#   "片有没有被吸走"、不做对准（handoff §1）。视觉伺服代码仍在 src/scara/vision/servo.py，
#   本流水线不走它。留 kind 是为了老单测与将来可能的回退，不是给新流程用的。


@dataclass
class ScaraStep:
    """全 kind 共用字段池；行尾注释标注哪个 kind 用、含义、单位（仿 flow_steps.Step）。"""
    id: str
    kind: ScaraStepKind
    desc: str
    target_z: float = 0.0                 # DESCEND/LIFT/DESCEND_PLACE：目标绝对 Z(mm)，executor 相对步进到位
    preset: str = ""                      # GOTO_PRESET：预设点名
    seconds: float = 0.0                  # WAIT：秒
    vac_on_s: float = VAC_ON_S            # SUCTION_ON：吸真空建立等待(s)
    vent_ms: float = VENT_MS              # SUCTION_OFF_VENT：泄压脉冲(ms)
    vent_wait_s: float = VENT_WAIT_S      # SUCTION_OFF_VENT：泄压后等残余真空(s)
    cell: Tuple[int, ...] = ()            # GOTO_CELL/VERIFY_PICK：右盘格(row,col)
    waypoint: str = ""                    # GOTO_WAYPOINT：示教航点名（scara_waypoints.json）
    verify_budget: int = 2


def _s(step_id: str, kind: ScaraStepKind, desc: str, **kw) -> ScaraStep:
    return ScaraStep(id=step_id, kind=kind, desc=desc, **kw)


def build_pick_from_right(cell: Tuple[int, int], tray, pump) -> Tuple[List[ScaraStep], str]:
    """右盘取片：视觉伺服对准 → 竖直降 → 吸 → 竖直抬 → 二次确认。cell=(row,col)。

    ★ 返回 `(steps, reason)`。**reason 非空 = 一步都没生成，上层必须停手。**
      Z 从 `TrayCells` 取（`z_pick_for` / `z_safe_for` 已叠加 z_offset_mm），不再走 `Heights`：
      `scara_heights.json` 从来不存在，只有 .example 占位，走它等于 DESCEND 到 Z=0.0，
      而实测取片 Z 是 −62.3 —— 那正是 z_pick_for docstring 明令禁止的 0.0 兜底。
      未标定就**不生成步骤**，比生成一串会撞盘的步骤安全得多。
    """
    r, c = int(cell[0]), int(cell[1])
    if tray.is_blocked(r, c):
        return [], f"cell_blocked:{r},{c}"
    z_pick = tray.z_pick_for(r, c)
    if z_pick is None:
        return [], f"z_pick_uncalibrated:{r},{c}"
    z_safe = tray.z_safe_for()
    if z_safe is None:
        return [], "z_safe_uncalibrated"          # 抬不到安全高就横移 = 刮盘边框
    if tray.replay_joints_for(r, c) is None:
        return [], f"cell_joints_missing:{r},{c}"     # 关节回放是主键，没有它整条路都不成立
    tag = f"pick_{r}_{c}"
    return [
        # ★ 关节回放到示教取片位。GOTO_CELL 内部按 plan_joint_order 走安全轴序：
        #   目标 Z 更低 → 先摆平面再下刀；动 J1 前先把 J2 收到 j2_min。
        #   示教位**本身就含取片 Z**（is_pick_pose=true），所以回放完就已经贴着片了；
        #   后面那步 DESCEND 是把 Z 对齐到 z_pick_for()（叠加了 z_offset_mm 的生效值），
        #   两者一致时是零位移的空操作，不一致时以 z_pick_for 为准。
        _s(f"{tag}_goto", ScaraStepKind.GOTO_CELL, f"关节回放到右盘格{cell}取片位", cell=cell),
        # ★ 存底与确认必须在**同一个机械臂姿态**下拍（都在 z_safe、都在该格正上方）：
        #   机械臂在两帧里的遮挡完全一样，差异就只剩硅片本身。若一帧有臂一帧没臂，
        #   遮挡变化会被当成"片没了"。
        _s(f"{tag}_snap", ScaraStepKind.SNAPSHOT_TRAY, "吸片前拍一帧存底", cell=cell),
        _s(f"{tag}_descend", ScaraStepKind.DESCEND, "竖直降到取片高度", target_z=z_pick),
        _s(f"{tag}_suck", ScaraStepKind.SUCTION_ON, "吸真空", vac_on_s=pump.vac_on_s),
        _s(f"{tag}_lift", ScaraStepKind.LIFT, "竖直抬到安全高", target_z=z_safe),
        _s(f"{tag}_verify", ScaraStepKind.VERIFY_PICK, "与存底比对：恰好少一片=已吸起", cell=cell),
    ], ""


def _pn(wp, base: str, cell) -> str:
    """按**源格**解析放样航点名：有 `base@r,c` 就用它，否则退回全局 `base`。

    ★ 2026-07-28 实测：同一个 place_micro 下 (5,1) 的片能落进槽、(5,0) 的落不进。
      row5 六格 rz_deg 全同（−74.802），所以不是角度问题 —— 差异来自
      **吸盘不一定吸在片的正中心**：示教对的是硅片中心，而各格凹槽大小不同、
      片在格内落点本就有差异（网格拟合残差最大 2.80mm，含真实物理差异，handoff §2 注意2）。
      片被偏心吸起，到显微镜就偏心落下。
    """
    f = getattr(wp, "place_name_for", None)
    return f(base, cell) if f else base


def build_place_to_micro(wp, pump, cell=None) -> Tuple[List[ScaraStep], str]:
    """显微镜放样：台子到装样位 → 上方高点 → 竖直下探接触位 → 破真空 → 竖直抬回 → **撤出**。

    ★ 返回 `(steps, reason)`，reason 非空 = 一步都不生成（同 build_pick_from_right 的纪律）。

    竖直进出**不需要额外的 DESCEND/LIFT 步骤** —— `above_micro` 与 `place_micro` 的
    J1/J2/J4 逐位相同、只差 J3（实测水平位移 0.000mm），而 `plan_joint_order` 对
    「只有 J3 变」的目标天然只发一条 move1。下探时 Z 排在最后、抬起时 Z 排在最前，
    正好就是 §8.2 要求的「正上方高点 → 竖直下探 → 竖直抬起，横向永远在高处走」。

    ★★ 最后那步 `micro_clear` 不是可选的收尾动作，是**安全前提**。
    抬回 `above_micro` 之后吸盘仍停在华夫盒正上方、只高出接触位 15.8mm，而 SCARA 是从
    物镜**底下**伸进去的。台子此时一动（扫描第一个动作就是移到起点，位移十几毫米），
    等于让华夫盒带着片从吸盘底下横扫出去。所以放样段必须一直做到「臂已经不在显微镜区」
    才算结束，绝不能把撤离交给下一段去做 —— 中间任何一次提前返回都会留下这个姿态。

    `wp` = `load_waypoints()` 的结果（`CalibBundle.waypoints`），需要
    `above_micro` / `place_micro` / `micro_clear` 三个航点，以及 `micro_load_xy_mm`。
    """
    have = getattr(wp, "waypoints", {}) or {}
    for name in ("above_micro", "place_micro", "micro_clear"):
        # 光有名字不够：关节值缺失的航点回放不了，得当成"没有"处理
        joints_of = getattr(wp, "joints_of", None)
        if name not in have or (joints_of is not None and not joints_of(name)):
            return [], f"waypoint_missing:{name}"
    if getattr(wp, "micro_load_xy_mm", None) is None:
        return [], "micro_load_pos_uncalibrated"
    return [
        _s("micro_stage", ScaraStepKind.MICRO_STAGE_TO_LOAD, "显微镜 XY 台移到装样位并回读校验"),
        _s("micro_above", ScaraStepKind.GOTO_WAYPOINT, "移到显微镜正上方高点",
           waypoint=_pn(wp, "above_micro", cell)),
        _s("micro_place", ScaraStepKind.GOTO_WAYPOINT, "竖直下探到放样接触位",
           waypoint=_pn(wp, "place_micro", cell)),
        _s("micro_vent", ScaraStepKind.SUCTION_OFF_VENT, "泄压破真空释放硅片",
           vent_ms=pump.vent_ms, vent_wait_s=pump.vent_wait_s),
        # ★ 抬起必须回**同一路**的上方点（per-cell 时是 above_micro@r,c），不能用全局的。
        #   否则片刚脱手、吸盘还贴着槽口时会横移一小段（(5,0) 与全局差 0.8mm），
        #   足以把刚放下的片带出槽 —— 而"竖直进出"这条约定正是为了避免这个。
        _s("micro_lift", ScaraStepKind.GOTO_WAYPOINT, "竖直抬回上方高点",
           waypoint=_pn(wp, "above_micro", cell)),
        _s("micro_clear", ScaraStepKind.GOTO_WAYPOINT, "★撤出显微镜区（台子动之前必须完成）",
           waypoint="micro_clear"),
    ], ""


def build_pick_from_peel(wp, pump) -> Tuple[List[ScaraStep], str]:
    """剥离台取片（2026-07-31 新流程腿①）：上方高点 → 竖直下探接触位 → 吸 → 竖直抬回。

    航点 `above_peel` / `peel_pick` 用 `scara_teach.py wp` 示教，纪律同显微镜侧三点链
    （交接 §8.4）：先 jog 到取片接触位教 `peel_pick`，再**只抬 J3** 教 `above_peel`，
    两点 J1/J2/J4 逐位相同、只差 J3；横向移动只在 `above_peel` 这个高度做。
    本函数把这条纪律做成 fail-closed 断言，不满足就一步都不生成。

    Phase 1 不带吸取确认（剥离台没有配前后对比相机）；J3 下移相机装上后（Phase 2）
    在「抬回」之后补一步持片确认。
    """
    joints_of = getattr(wp, "joints_of", None)
    for name in ("above_peel", "peel_pick"):
        if joints_of is None or not joints_of(name):
            return [], f"waypoint_missing:{name}"
    above = [float(x) for x in joints_of("above_peel")]
    pick = [float(x) for x in joints_of("peel_pick")]
    # 竖直进出断言：平面三轴必须基本一致（示教时只许动 J3），否则「下探」会变成
    # 带着片高度差的斜插 —— 那正是三点链约定要消灭的动作。0.3° 留给 readall 回读噪声。
    planar = max(abs(a - b) for a, b in zip(above[:2] + above[3:], pick[:2] + pick[3:]))
    if planar > 0.3:
        return [], (f"peel_waypoints_not_vertical：above_peel 与 peel_pick 的 J1/J2/J4 "
                    f"最大差 {planar:.3f}° > 0.3°，竖直进出约定不成立（重教：先教 peel_pick，"
                    f"只抬 J3 再教 above_peel）")
    # J3 是直动 Z 轴、+Z 为上：接近点必须在接触点上方
    if not above[2] > pick[2]:
        return [], (f"peel_waypoints_z_inverted：above_peel.J3({above[2]:.3f}) 必须高于 "
                    f"peel_pick.J3({pick[2]:.3f})")
    return [
        _s("peel_above", ScaraStepKind.GOTO_WAYPOINT, "移到剥离台正上方高点", waypoint="above_peel"),
        _s("peel_pick", ScaraStepKind.GOTO_WAYPOINT, "竖直下探到取片接触位", waypoint="peel_pick"),
        _s("peel_suck", ScaraStepKind.SUCTION_ON, "吸真空", vac_on_s=pump.vac_on_s),
        _s("peel_lift", ScaraStepKind.GOTO_WAYPOINT, "竖直抬回上方高点", waypoint="above_peel"),
    ], ""


def build_place_to_right(cell: Tuple[int, int], tray, pump) -> Tuple[List[ScaraStep], str]:
    """右盘放片存储（2026-07-31 新流程腿③）：回放到格位上方 → 竖直降到接触高 → 泄压 → 竖直抬。

    右盘在新流程里是**放料侧**（SCARA 把表征完的片存进来），但接触高度与取片是同一个
    凹槽底面，故仍复用 `TrayCells` 的三轨访问器（`z_pick_for` 读的是「凹槽接触高」，与
    动作方向无关）。本轮示教纪律：教**凹槽中心**（不再是硅片中心，交接 §2 注意2 的
    残差来源这次成了被放的对象）。fail-closed 纪律同 build_pick_from_right。
    """
    r, c = int(cell[0]), int(cell[1])
    if tray.is_blocked(r, c):
        return [], f"cell_blocked:{r},{c}"
    z_place = tray.z_pick_for(r, c)
    if z_place is None:
        return [], f"z_place_uncalibrated:{r},{c}"
    z_safe = tray.z_safe_for()
    if z_safe is None:
        return [], "z_safe_uncalibrated"
    if tray.replay_joints_for(r, c) is None:
        return [], f"cell_joints_missing:{r},{c}"
    tag = f"place_{r}_{c}"
    return [
        # GOTO_CELL 内部把 J3 停在 z_safe、平面轴用示教值 —— 放片的「正上方接近」，
        # 与取片段完全同构（executor._goto_cell 的 docstring 有完整理由）。
        _s(f"{tag}_goto", ScaraStepKind.GOTO_CELL, f"关节回放到右盘格{cell}上方", cell=cell),
        _s(f"{tag}_descend", ScaraStepKind.DESCEND_PLACE, "竖直降到放片接触高", target_z=z_place),
        _s(f"{tag}_vent", ScaraStepKind.SUCTION_OFF_VENT, "泄压破真空释放硅片",
           vent_ms=pump.vent_ms, vent_wait_s=pump.vent_wait_s),
        _s(f"{tag}_lift", ScaraStepKind.LIFT, "竖直抬到安全高", target_z=z_safe),
    ], ""


def build_pick_from_micro(wp, pump, cell=None) -> Tuple[List[ScaraStep], str]:
    """显微镜取回（表征结束后）：台子回装样位 → 上方高点 → 竖直下探接触位 → 吸 → 抬回 → 撤出。

    与 build_place_to_micro 共用同一组航点（`above_micro`/`place_micro`/`micro_clear`）
    与装样位 —— 取回就是放样的逆动作，接触位是同一点。最后一步仍是 `micro_clear`：
    臂不撤出显微镜区，后续任何台子动作都会带着华夫盒从吸盘（这次还吸着片）底下横扫。
    step id 加 `retr_` 前缀，与放样段的 `micro_*` 区分开（一条 transfer 链里两段都在）。
    """
    have = getattr(wp, "waypoints", {}) or {}
    joints_of = getattr(wp, "joints_of", None)
    for name in ("above_micro", "place_micro", "micro_clear"):
        if name not in have or (joints_of is not None and not joints_of(name)):
            return [], f"waypoint_missing:{name}"
    if getattr(wp, "micro_load_xy_mm", None) is None:
        return [], "micro_load_pos_uncalibrated"
    return [
        _s("retr_stage", ScaraStepKind.MICRO_STAGE_TO_LOAD, "显微镜 XY 台回装样位并回读校验"),
        _s("retr_above", ScaraStepKind.GOTO_WAYPOINT, "移到显微镜正上方高点",
           waypoint=_pn(wp, "above_micro", cell)),
        _s("retr_pick", ScaraStepKind.GOTO_WAYPOINT, "竖直下探到取片接触位",
           waypoint=_pn(wp, "place_micro", cell)),
        _s("retr_suck", ScaraStepKind.SUCTION_ON, "吸真空", vac_on_s=pump.vac_on_s),
        _s("retr_lift", ScaraStepKind.GOTO_WAYPOINT, "竖直抬回上方高点",
           waypoint=_pn(wp, "above_micro", cell)),
        _s("retr_clear", ScaraStepKind.GOTO_WAYPOINT, "★撤出显微镜区（吸着片，更不许台子先动）",
           waypoint="micro_clear"),
    ], ""


def build_scan_leg(wp, recipe, cell=None) -> Tuple[List[ScaraStep], str]:
    """表征扫描段：切物镜 → 台子到扫描起点 → 起始对焦 → 按配方扫描 → 台子回装样位。

    从 build_inspect_one 抽出的共用尾段（2026-07-31：新流程 transfer 链复用，步骤 id
    保持不变）。**前置安全不变**：撤臂门由各 MICRO_* 步骤在 executor 内部逐一过
    （_assert_arm_clear），编排层不重复检查。

    `recipe` = `CalibBundle.scan_recipe`。**None 或未通过 validate 一律一步都不生成**：
    绝不"没有配方就用默认参数扫一下"（`start_scan` 空参数的退化行为见 build_inspect_one）。
    """
    if recipe is None:
        return [], "scan_recipe_missing：没有扫描配方，扫描段一步都不生成"
    ok, why = recipe.validate(getattr(wp, "current_stage_frame", "") or "")
    if not ok:
        return [], why
    return [
        # ★ 切物镜排在**台子移动之前、焦点仍停在装样位那个"离样品最远"的 Z** 的时候做：
        #   高倍物镜工作距离更短，而 set_objective 内部会 suspend 焦点限位并**驱动焦点轴**
        #   （microscope_controller.py:650 的注释）。在已合焦的位置切镜 = 拿工作距离赌余量。
        _s("micro_objective", ScaraStepKind.MICRO_SET_OBJECTIVE, "切物镜到配方要求的倍率并回读校验"),
        # 起点单独走一步、且带到位回读校验：move_absolute 把目标值直接写进位置缓存不回读
        # （microscope_controller.py:317-319），从行程边界上的装样位出发这一大段位移一旦被
        # 限位截断，软件会认为自己到位了，然后整轮扫描在错误的地方进行且全程不报错。
        _s("micro_to_start", ScaraStepKind.MICRO_STAGE_TO_START, "显微镜 XY 台移到扫描起点并回读校验"),
        # 对焦必须在**台子已经到扫描区之后**：装样位上方是盒沿/台面，在那里对的焦
        # 对的根本不是样品面。顺序反了照样能对上焦、照样不报错，只是对错了东西。
        _s("micro_focus", ScaraStepKind.MICRO_FOCUS_TO_START,
           "焦点到示教合焦 Z（回读校验）+ WDI 精修 + 漂移断言"),
        # cell 传下去只为给保存目录起名（inspect_{ts}_r{row}c{col}）——
        # 扫完的图必须能对回是右盘哪一格来的，否则一批数据混在一起就再也分不开了
        _s("micro_scan", ScaraStepKind.MICRO_SCAN, "按配方全自动扫描并等待完成", cell=cell),
        _s("micro_back_load", ScaraStepKind.MICRO_STAGE_TO_LOAD, "台子回装样位并回读校验"),
    ], ""


def build_inspect_one(cell: Tuple[int, int], tray, wp, pump,
                      recipe=None) -> Tuple[List[ScaraStep], str]:
    """一次完整送检（旧流程，保留）= 右盘取片 + 放样 + 撤离 + 扫描 + 台子归位。

    步骤对照（用户定义的流程）：
      1. 显微镜 XY 台到装样位          → MICRO_STAGE_TO_LOAD（在放样段开头）
      2. SCARA 从取样盘吸硅片          → GOTO_CELL + DESCEND + SUCTION_ON + LIFT + VERIFY_PICK
      3. 越过危险区的防碰撞姿态        → 不是独立步骤：由 plan_joint_order 在每次关节回放里
                                        自动插入「先收 J2 再动 J1」，比一个固定航点更普适
      4. 到标定过的放样位、Z 下压放样  → GOTO_WAYPOINT above_micro → place_micro → 破真空
      4.5 撤出显微镜区                 → GOTO_WAYPOINT micro_clear（放样段自带，见其 docstring）
      5. 切物镜 → 台子到扫描起点 → 起始对焦 → 按配方扫描 → 台子回装样位
                                       → build_scan_leg
    """
    pick, why = build_pick_from_right(cell, tray, pump)
    if why:
        return [], why
    place, why = build_place_to_micro(wp, pump, cell)
    if why:
        return [], why
    leg, why = build_scan_leg(wp, recipe, cell)
    if why:
        return [], why
    return pick + place + leg, ""


def build_transfer_one(cell: Tuple[int, int], tray, wp, pump,
                       recipe=None) -> Tuple[List[ScaraStep], str]:
    """一次完整转运+表征（2026-07-31 新流程，交接 v6）：

      剥离台取片 → 显微镜放样+撤离 → 表征扫描 → 显微镜取回+撤离 → 放右盘 cell 存储。

    五段各自 fail-closed，任一未标定就整链一步都不生成。安全不变量（都在段内保证）：
      · 剥离台/显微镜两侧都是「正上方高点 → 竖直下探 → 竖直抬起」，横向只在高处走；
      · 每次台子动作前撤臂门（_assert_arm_clear）逐一过；
      · 扫描配方缺失/未过校验 → 不生成，绝不退化成"原地拍 9 张同一画面"。
    """
    peel, why = build_pick_from_peel(wp, pump)
    if why:
        return [], why
    place, why = build_place_to_micro(wp, pump, cell)
    if why:
        return [], why
    leg, why = build_scan_leg(wp, recipe, cell)
    if why:
        return [], why
    retr, why = build_pick_from_micro(wp, pump, cell)
    if why:
        return [], why
    store, why = build_place_to_right(cell, tray, pump)
    if why:
        return [], why
    return peel + place + leg + retr + store, ""


def build_place_to_waffle(waffle, heights, pump) -> List[ScaraStep]:
    """华夫盒放片(开环)：到上方 → 到放置位 → J4 角度补偿 → 竖直降 → 泄压释放 → 竖直抬。"""
    return [
        _s("place_above", ScaraStepKind.GOTO_PRESET, "移到华夫盒上方", preset=waffle.above_preset),
        _s("place_pos", ScaraStepKind.GOTO_PRESET, "移到华夫盒放置位", preset=waffle.place_preset),
        _s("place_j4", ScaraStepKind.J4_COMPENSATE, "J4 角度补偿对准凹槽"),
        _s("place_descend", ScaraStepKind.DESCEND_PLACE, "竖直降到放片高度", target_z=heights.z_place_waffle),
        _s("place_vent", ScaraStepKind.SUCTION_OFF_VENT, "泄压破真空释放",
           vent_ms=pump.vent_ms, vent_wait_s=pump.vent_wait_s),
        _s("place_lift", ScaraStepKind.LIFT, "竖直抬起", target_z=heights.z_above_waffle),
    ]


if __name__ == "__main__":   # 自检：id 唯一 + kind 合法（仿 flow_steps.py:507）
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 cp1252，中文 print 会崩
    except Exception:
        pass
    from types import SimpleNamespace

    class _Tray:                       # 最小 TrayCells 替身
        def is_blocked(self, r, c): return False
        def z_pick_for(self, r, c): return -50.0
        def z_safe_for(self): return 0.0
        def replay_joints_for(self, r, c): return [10.0, 20.0, -50.0, 0.0]

    h = SimpleNamespace(z_place_waffle=-40.0, z_above_waffle=0.0)
    p = SimpleNamespace(vac_on_s=1.2, vent_ms=1200.0, vent_wait_s=3.0)
    f = SimpleNamespace(above_preset="above_waffle", place_preset="place_waffle")
    _MICRO_WPS = {"above_micro": 1, "place_micro": 1, "micro_clear": 1}

    def _wp(**kw):
        base = dict(waypoints=dict(_MICRO_WPS), micro_load_xy_mm=(1.0, 2.0),
                    current_stage_frame="")
        base.update(kw)
        return SimpleNamespace(**base)

    class _Recipe:                     # 最小 ScanRecipe 替身（只用到 validate）
        def __init__(self, ok=True, why=""):
            self._ok, self._why = ok, why

        def validate(self, _frame=""):
            return (self._ok, self._why)

    wp = _wp()
    pick, why = build_pick_from_right((3, 4), _Tray(), p)
    assert not why, why
    # 未标定必须一步都不生成（fail-closed 自检）
    class _Blank(_Tray):
        def z_pick_for(self, r, c): return None
    assert build_pick_from_right((3, 4), _Blank(), p) == ([], "z_pick_uncalibrated:3,4")

    class _NoJoints(_Tray):
        def replay_joints_for(self, r, c): return None
    assert build_pick_from_right((3, 4), _NoJoints(), p) == ([], "cell_joints_missing:3,4")

    # 放样段的 fail-closed：缺任一航点 / 缺装样位都必须一步不生成
    assert build_place_to_micro(_wp(waypoints={}), p) == ([], "waypoint_missing:above_micro")
    assert build_place_to_micro(_wp(micro_load_xy_mm=None), p) == ([], "micro_load_pos_uncalibrated")
    # ★ 撤离位缺失也必须一步不生成 —— 没有它就没有"台子动之前臂已经出来了"这个前提
    assert build_place_to_micro(_wp(waypoints={"above_micro": 1, "place_micro": 1}), p) \
        == ([], "waypoint_missing:micro_clear")

    # 扫描配方的 fail-closed：没有配方 / 配方没过校验，都不许退化成"用默认参数扫一下"
    assert build_inspect_one((3, 4), _Tray(), wp, p, recipe=None)[1].startswith("scan_recipe_missing")
    assert build_inspect_one((3, 4), _Tray(), wp, p,
                             recipe=_Recipe(False, "scan_area_uncalibrated:x"))[0] == []

    full, why = build_inspect_one((3, 4), _Tray(), wp, p, recipe=_Recipe())
    assert not why, why
    kinds = [s.kind for s in full]
    ids = [s.id for s in full]
    assert kinds[0] == ScaraStepKind.GOTO_CELL, "第一步必须是关节回放到格位"
    assert kinds[-1] == ScaraStepKind.MICRO_STAGE_TO_LOAD, "最后一步必须是台子回装样位"
    # ★★ 最关键的一条：臂撤出去，必须排在台子动起来之前
    assert ids.index("micro_clear") < ids.index("micro_to_start") < ids.index("micro_scan"), \
        "撤臂必须早于台子移动与扫描"
    assert ids.index("micro_vent") < ids.index("micro_clear"), "先放下片再撤"
    print(f"\n送检全链: {len(full)} steps")
    for s in full:
        print(f"  {s.id:22} {s.kind.value:22} {s.desc}")
    print()

    # ── 2026-07-31 新流程（transfer 链）自检 ──────────────────────────────
    class _WpFull:                   # 带 joints_of 的航点替身（新流程五段都要真关节值）
        _J = {"above_micro": [-28.8, -10.0, 0.1, -24.1], "place_micro": [-28.8, -10.0, -15.7, -24.1],
              "micro_clear": [-18.0, 40.0, 0.1, -24.1],
              "above_peel": [10.0, 30.0, -10.0, 5.0], "peel_pick": [10.0, 30.0, -45.0, 5.0]}

        def __init__(self, joints=None, micro_load_xy_mm=(1.0, 2.0)):
            self._j = {k: list(v) for k, v in (joints if joints is not None else self._J).items()}
            self.waypoints = {k: 1 for k in self._j}
            self.micro_load_xy_mm = micro_load_xy_mm
            self.current_stage_frame = ""

        def joints_of(self, name):
            return list(self._j.get(name) or [])

    wp2 = _WpFull()
    # 剥离台取片段：缺航点 / 非竖直 / Z 反了，都必须一步不生成
    _no_above = {k: v for k, v in _WpFull._J.items() if k != "above_peel"}
    assert build_pick_from_peel(_WpFull(joints=_no_above), p)[1] == "waypoint_missing:above_peel"
    _tilt = _WpFull(); _tilt._j["above_peel"][0] += 1.0
    assert build_pick_from_peel(_tilt, p)[1].startswith("peel_waypoints_not_vertical")
    _zinv = _WpFull(); _zinv._j["above_peel"][2] = -50.0
    assert build_pick_from_peel(_zinv, p)[1].startswith("peel_waypoints_z_inverted")
    peel, why = build_pick_from_peel(wp2, p)
    assert not why and [s.id for s in peel] == ["peel_above", "peel_pick", "peel_suck", "peel_lift"]

    # 右盘放片段：与取片同源的 fail-closed
    assert build_place_to_right((3, 4), _Blank(), p) == ([], "z_place_uncalibrated:3,4")
    assert build_place_to_right((3, 4), _NoJoints(), p) == ([], "cell_joints_missing:3,4")

    class _NoSafe(_Tray):
        def z_safe_for(self): return None
    assert build_place_to_right((3, 4), _NoSafe(), p) == ([], "z_safe_uncalibrated")
    store, why = build_place_to_right((3, 4), _Tray(), p)
    assert not why
    assert [s.kind for s in store] == [ScaraStepKind.GOTO_CELL, ScaraStepKind.DESCEND_PLACE,
                                       ScaraStepKind.SUCTION_OFF_VENT, ScaraStepKind.LIFT]

    # 显微镜取回段：缺撤离位 / 缺装样位都必须一步不生成
    _no_clear = {k: v for k, v in _WpFull._J.items() if k != "micro_clear"}
    assert build_pick_from_micro(_WpFull(joints=_no_clear), p)[1] == "waypoint_missing:micro_clear"
    assert build_pick_from_micro(_WpFull(micro_load_xy_mm=None), p)[1] == "micro_load_pos_uncalibrated"
    retr, why = build_pick_from_micro(wp2, p)
    assert not why and retr[0].id == "retr_stage" and retr[-1].id == "retr_clear"

    # transfer 全链：配方缺失 fail-closed；id 唯一；顺序断言
    assert build_transfer_one((3, 4), _Tray(), wp2, p, recipe=None)[1].startswith("scan_recipe_missing")
    trans, why = build_transfer_one((3, 4), _Tray(), wp2, p, recipe=_Recipe())
    assert not why, why
    ids = [s.id for s in trans]
    assert len(ids) == len(set(ids)), "transfer 链重复 step id"
    order = ["peel_above", "micro_above", "micro_clear", "micro_objective", "micro_scan",
             "micro_back_load", "retr_above", "retr_clear", "place_3_4_goto"]
    pos = [ids.index(x) for x in order]
    assert pos == sorted(pos), f"transfer 链顺序错: {list(zip(order, pos))}"
    print(f"转运全链: {len(trans)} steps")
    for s in trans:
        print(f"  {s.id:22} {s.kind.value:22} {s.desc}")
    print()

    steps = pick + build_place_to_waffle(f, h, p)
    ids = [s.id for s in steps]
    assert len(ids) == len(set(ids)), "重复 step id"
    for s in steps:
        assert isinstance(s.kind, ScaraStepKind)
        print(f"{s.id:22} {s.kind.value:18} {s.desc}")
    print(f"OK: {len(steps)} steps, ids unique")
