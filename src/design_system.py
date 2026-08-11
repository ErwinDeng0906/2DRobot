"""全局统一设计系统 —— 浅色工业风（v1，用户确认）。

单一事实来源：色板令牌 + 字号阶梯(type scale) + 全局 QSS。
- 基调沿用机械臂主题（用户已认可）：主蓝 #2563EB + 强调青 #0EA5C4 + 白玻璃卡片。
- **选中的标签/菜单保持琥珀色高亮**（用户明确要求）：TAB_ACTIVE=#F59E0B。
- 字号阶梯（用户确认）：Display/Title/Section/Body/Label/Caption/Mono。

用法：
- 主窗口全局下发 ``app.setStyleSheet(APP_STYLESHEET)``（core/main_window）。
- 各页统一引用色板/字号常量，禁止内联裸值。
- ``robot_arm/ui/theme.py`` 已改为从本模块 re-export，保持既有导入不变。
"""
from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════
# 色板令牌（浅色工业风）
# ══════════════════════════════════════════════════════════════════════════
BG_DARK       = "#EEF2F7"   # 主背景 浅灰
BG_DARK2      = "#F5F8FC"   # 次背景
BG_CARD       = "#FFFFFF"   # 卡片 白
BG_CARD_SOLID = "#FFFFFF"
BG_GLASS      = "#F4F7FB"   # 输入/按钮底（浅）

PRIMARY_COLOR   = "#2563EB"   # 主色 蓝
PRIMARY_HOVER   = "#1D4ED8"   # 主色 hover
PRIMARY_PRESSED = "#1E40AF"   # 主色 按下
PRIMARY_SOFT    = "#E8F0FE"   # 主色弱底
ACCENT_COLOR    = "#0EA5C4"   # 强调 青
ACCENT_SOFT     = "#E2F5F9"   # 青弱底
SUCCESS_COLOR   = "#1F9D6B"   # 成功 绿
ERROR_COLOR     = "#DC4456"   # 危险 红
WARNING_COLOR   = "#C9821B"   # 警告 琥珀（文字/描边用）

# 选中态（标签/菜单）—— 用户要求保持琥珀色高亮
TAB_ACTIVE       = "#F59E0B"  # 琥珀 amber-500
TAB_ACTIVE_HOVER = "#E0780A"
TAB_ACTIVE_SOFT  = "#FFF7ED"  # 琥珀弱底（hover/浅高亮）
TAB_INACTIVE     = "#64748B"
# amber 语义梯度（供需要琥珀渐变的页面复用）
AMBER_50  = "#FFFBEB"
AMBER_100 = "#FEF3C7"
AMBER_200 = "#FDE68A"
AMBER_500 = "#F59E0B"
AMBER_600 = "#D97706"
AMBER_700 = "#B45309"

TEXT_PRIMARY   = "#1E293B"
TEXT_SECONDARY = "#64748B"
TEXT_MUTED     = "#94A3B8"
BORDER_COLOR   = "#E2E8F0"
BORDER_SOFT    = "#EDF1F6"

# ══════════════════════════════════════════════════════════════════════════
# 字号阶梯 type scale（px）—— 单页最多用 4 档，禁用 8/9/10px
# 档位  px  weight  用途
# Display 24 700  关键大数值读数（每页仅 1 处）
# Title   17 600  页面标题
# Section 14 600  卡片/分组标题
# Body    13 400  正文、按钮、输入框（强调 500）
# Label   12 400  表单标签、次级信息
# Caption 11 400  单位、注释、状态
# Mono    13 500  坐标/数值/日志（读数 20/600）
# ══════════════════════════════════════════════════════════════════════════
FONT_FAMILY = '"Microsoft YaHei", "Segoe UI", sans-serif'
MONO        = '"JetBrains Mono", "Cascadia Code", "Consolas", monospace'

SIZE_DISPLAY = 24
SIZE_TITLE   = 17
SIZE_SECTION = 14
SIZE_BODY    = 13
SIZE_LABEL   = 12
SIZE_CAPTION = 11
SIZE_MONO    = 13
SIZE_READOUT = 20   # Mono 大数值读数

WEIGHT_REGULAR = 400
WEIGHT_MEDIUM  = 500
WEIGHT_SEMIBOLD = 600
WEIGHT_BOLD    = 700


def font(size: int = SIZE_BODY, weight: int = WEIGHT_REGULAR, mono: bool = False):
    """构造符合设计系统的 QFont。size=px（近似按 pt 传入亦可，Qt 会按点渲染）。"""
    from PyQt6.QtGui import QFont
    fam = "JetBrains Mono" if mono else "Microsoft YaHei"
    f = QFont(fam)
    f.setPixelSize(int(size))
    f.setWeight(QFont.Weight.Bold if weight >= WEIGHT_BOLD else
                (QFont.Weight.DemiBold if weight >= WEIGHT_SEMIBOLD else
                 (QFont.Weight.Medium if weight >= WEIGHT_MEDIUM else QFont.Weight.Normal)))
    return f


def nearest_tier(px: int) -> int:
    """把历史裸字号就近归入档位（迁移辅助）：8-11→Caption/Label，12-13→Body，
    14-15→Section，16-18→Title，20+→Display/读数。"""
    if px <= 11: return SIZE_CAPTION if px <= 10 else SIZE_LABEL
    if px <= 13: return SIZE_BODY
    if px <= 15: return SIZE_SECTION
    if px <= 18: return SIZE_TITLE
    return SIZE_DISPLAY


# ══════════════════════════════════════════════════════════════════════════
# 全局 QSS
# ══════════════════════════════════════════════════════════════════════════
APP_STYLESHEET = f"""
QWidget {{
    background-color: transparent;
    color: {TEXT_PRIMARY};
    font-family: {FONT_FAMILY};
    font-size: {SIZE_BODY}px;
}}
/* 顶层容器铺浅底：control_widget 根 widget 带 objectName=armRoot / appRoot */
QWidget#armRoot, QWidget#appRoot {{
    background-color: {BG_DARK};
}}
QGroupBox {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_COLOR};
    border-radius: 12px;
    margin-top: 14px;
    padding: 12px 8px 8px 8px;
    font-weight: 600;
    font-size: {SIZE_SECTION}px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {ACCENT_COLOR};
    letter-spacing: 0.5px;
}}
/* ── 按钮 ── */
QPushButton {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_COLOR};
    border-radius: 8px;
    padding: 5px 12px;
    min-height: 24px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
}}
QPushButton:hover {{
    background-color: {PRIMARY_SOFT};
    border-color: {PRIMARY_COLOR};
}}
QPushButton:pressed {{ background-color: #D6E4FC; }}
QPushButton:disabled {{
    background-color: #F1F4F8;
    color: {TEXT_MUTED};
    border-color: {BORDER_SOFT};
}}
QPushButton[cssClass="primary"] {{
    background-color: {PRIMARY_COLOR};
    border: 1px solid {PRIMARY_COLOR};
    color: #FFFFFF;
    font-weight: 600;
}}
QPushButton[cssClass="primary"]:hover {{ background-color: {PRIMARY_HOVER}; }}
QPushButton[cssClass="primary"]:pressed {{ background-color: {PRIMARY_PRESSED}; }}
QPushButton[cssClass="primary"]:disabled {{ background-color: #AEC4F2; color: #EAF0FD; border-color: #AEC4F2; }}
QPushButton[cssClass="success"] {{
    background-color: {SUCCESS_COLOR};
    border: 1px solid {SUCCESS_COLOR};
    color: #FFFFFF;
    font-weight: 600;
}}
QPushButton[cssClass="success"]:hover {{ background-color: #1A8A5E; }}
QPushButton[cssClass="success"]:disabled {{ background-color: #A9DCC6; color: #EAF7F1; border-color: #A9DCC6; }}
QPushButton[cssClass="warning"] {{
    background-color: {WARNING_COLOR};
    border: 1px solid {WARNING_COLOR};
    color: #FFFFFF;
    font-weight: 600;
}}
QPushButton[cssClass="warning"]:hover {{ background-color: #B07116; }}
QPushButton[cssClass="danger"] {{
    background-color: {ERROR_COLOR};
    border: 1px solid {ERROR_COLOR};
    color: #FFFFFF;
    font-weight: 600;
}}
QPushButton[cssClass="danger"]:hover {{ background-color: #C53B4C; }}
QPushButton[cssClass="danger"]:disabled {{ background-color: #F0B5BC; color: #FCEEF0; border-color: #F0B5BC; }}
/* 选中的「标签页按钮」保持琥珀色高亮（用户要求）：给标签按钮加 cssClass="tab" */
QPushButton[cssClass="tab"]:checked {{
    background-color: {TAB_ACTIVE};
    border: 1px solid {TAB_ACTIVE};
    color: #FFFFFF;
    font-weight: 600;
}}
QPushButton[cssClass="tab"]:checked:hover {{ background-color: {TAB_ACTIVE_HOVER}; }}
QPushButton[cssClass="tab"]:hover {{ border-color: {TAB_ACTIVE}; background-color: {TAB_ACTIVE_SOFT}; }}
/* JOG 按键 */
QPushButton[cssClass="jog"] {{
    background-color: {BG_GLASS};
    border: 1px solid {BORDER_COLOR};
    border-radius: 10px;
    padding: 9px 0;
    font-family: {MONO};
    font-weight: 600;
    font-size: {SIZE_BODY}px;
    color: {TEXT_PRIMARY};
}}
QPushButton[cssClass="jog"]:hover {{
    border-color: {ACCENT_COLOR};
    background-color: {ACCENT_SOFT};
    color: {ACCENT_COLOR};
}}
QPushButton[cssClass="jog"]:pressed {{ background-color: #D2EFF5; }}
/* ── 输入控件 ── */
QLineEdit, QDoubleSpinBox, QSpinBox {{
    background-color: {BG_GLASS};
    border: 1px solid {BORDER_COLOR};
    border-radius: 7px;
    padding: 4px 8px;
    min-height: 24px;
    min-width: 56px;
    color: {TEXT_PRIMARY};
    font-family: {MONO};
    font-size: {SIZE_LABEL}px;
    selection-background-color: {PRIMARY_COLOR};
}}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {{ border-color: #4A8FE0; }}
QDoubleSpinBox::up-button, QSpinBox::up-button,
QDoubleSpinBox::down-button, QSpinBox::down-button {{ width: 0px; height: 0px; border: none; }}
QComboBox {{
    background-color: {BG_GLASS};
    border: 1px solid {BORDER_COLOR};
    border-radius: 7px;
    padding: 3px 8px;
    min-height: 24px;
    color: {TEXT_PRIMARY};
}}
QComboBox:focus {{ border-color: #4A8FE0; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background-color: {BG_CARD_SOLID};
    border: 1px solid {BORDER_COLOR};
    selection-background-color: {PRIMARY_COLOR};
    color: {TEXT_PRIMARY};
}}
/* ── Checkbox ── */
QCheckBox {{ color: {TEXT_SECONDARY}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 15px; height: 15px; border-radius: 4px;
    border: 1px solid {BORDER_COLOR}; background: {BG_GLASS};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT_SOFT}; border-color: {ACCENT_COLOR};
    image: none;
}}
/* ── Tab（QTabWidget/QTabBar）── 选中标签琥珀色高亮 ── */
QTabWidget::pane {{
    border: 1px solid {BORDER_COLOR};
    border-radius: 12px;
    background: {BG_CARD};
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    border: 1px solid transparent;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 6px 14px;
    color: {TEXT_SECONDARY};
}}
QTabBar::tab:selected {{
    background: {TAB_ACTIVE};
    border-color: {TAB_ACTIVE};
    color: #FFFFFF;
    font-weight: 600;
}}
QTabBar::tab:hover:!selected {{ color: {TEXT_PRIMARY}; background: {TAB_ACTIVE_SOFT}; }}
QLabel {{ background-color: transparent; }}
QListWidget {{
    background-color: {BG_GLASS};
    border: 1px solid {BORDER_COLOR};
    border-radius: 8px;
}}
QListWidget::item {{ padding: 4px; border-radius: 6px; }}
QListWidget::item:selected {{ background: {PRIMARY_SOFT}; color: {TEXT_PRIMARY}; }}
QProgressBar {{
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    background: #EAF0F7;
    text-align: center;
    min-height: 16px;
    color: {TEXT_PRIMARY};
}}
QProgressBar::chunk {{
    background-color: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {PRIMARY_COLOR}, stop:1 {ACCENT_COLOR});
    border-radius: 5px;
}}
QSlider::groove:horizontal {{ height: 5px; background: #D7E0EB; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: {ACCENT_COLOR}; width: 14px; margin: -6px 0; border-radius: 7px;
}}
QScrollBar:vertical {{ background: transparent; width: 9px; margin: 0; }}
QScrollBar::handle:vertical {{ background: rgba(120,160,200,0.25); border-radius: 4px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: rgba(120,160,200,0.40); }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QSplitter::handle {{ background: {BORDER_SOFT}; }}
"""
