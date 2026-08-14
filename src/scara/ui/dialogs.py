"""Small, theme-independent dialogs used by safety-critical SCARA tasks."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import QMessageBox, QWidget


LIGHT_WARNING_DIALOG_STYLESHEET = """
QMessageBox {
    background-color: #FFFFFF;
    color: #111827;
}
QMessageBox QLabel {
    background-color: #FFFFFF;
    color: #111827;
}
QMessageBox QPushButton {
    min-width: 76px;
    padding: 6px 18px;
    color: #111827;
    background-color: #F3F4F6;
    border: 1px solid #9CA3AF;
    border-radius: 4px;
}
QMessageBox QPushButton:hover {
    background-color: #E5E7EB;
}
QMessageBox QPushButton:default {
    background-color: #DBEAFE;
    border: 2px solid #2563EB;
}
"""


def ask_light_warning_confirmation(
    parent: Optional[QWidget],
    title: str,
    text: str,
) -> bool:
    """Ask a Yes/No safety question with explicit white/black colors.

    The SCARA page uses a dark global theme, while native Windows message-box
    content can remain white. Styling the complete instance prevents inherited
    white text from becoming invisible on that native white surface.
    """

    dialog = QMessageBox(parent)
    dialog.setIcon(QMessageBox.Icon.Warning)
    dialog.setWindowTitle(title)
    dialog.setText(text)
    dialog.setStandardButtons(
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )
    dialog.setDefaultButton(QMessageBox.StandardButton.No)
    dialog.setEscapeButton(QMessageBox.StandardButton.No)
    dialog.setStyleSheet(LIGHT_WARNING_DIALOG_STYLESHEET)
    return dialog.exec() == QMessageBox.StandardButton.Yes
