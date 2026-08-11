"""取放任务的命名点位序列定义 (安全路径)。

每段都是示教过的安全空间, 经由 lift/mid 过渡点绕开禁区。
"""

# 完整取放路径: 左台取 -> 经中转 -> 右台放 -> 回 home
PICK_PLACE_FULL = [
    "home",
    "pick_approach",   # 左台上方
    "pick_contact",    # 下降吸取 (此处触发 grip)
    "pick_lift",       # 垂直抬起
    "mid",             # 中转点
    "place_lift",      # 右台上方高位
    "place_approach",  # 右台上方
    "place_contact",   # 下降放置 (此处触发 release)
    "place_lift",      # 抬回
    "mid",             # 中转
    "home",            # 回安全位
]

# 仅转移段 (验证避障路径, 不含取放下降)
TRANSIT_ONLY = [
    "home", "pick_lift", "mid", "place_lift", "home",
]

# 动作点: 到达这些点后需触发吸盘动作
GRIP_AT = "pick_contact"     # 到此吸
RELEASE_AT = "place_contact"  # 到此放
