"""The recommended setup controls fit before optional model browsing."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from zoom_copilot.config import Preferences
from zoom_copilot.model_picker import ModelPickerDialog


@pytest.fixture
def qt_with_windows_font():
    app = QApplication.instance() or QApplication([])
    original = app.font()
    added = []
    if not QFontDatabase.families() and os.name == "nt":
        for name in ("segoeui.ttf", "msyh.ttc"):
            path = Path("C:/Windows/Fonts") / name
            if path.is_file():
                added.append(QFontDatabase.addApplicationFont(str(path)))
        app.setFont(QFont("Microsoft YaHei UI", 10))
    yield app
    app.setFont(original)
    for identity in added:
        if identity >= 0:
            QFontDatabase.removeApplicationFont(identity)


def test_qwen_essential_setup_is_visible_without_scrolling_at_laptop_size(qt_with_windows_font):
    dialog = ModelPickerDialog(Preferences())
    try:
        dialog.resize(960, 760)
        dialog.show()
        QTest.qWait(30)
        viewport = dialog.body_scroll.viewport()
        assert dialog.provider_select.currentData() == "qwen"
        assert dialog.body_scroll.verticalScrollBar().value() == 0
        assert dialog.model_input.text() == "qwen3.8-flash"
        assert not dialog.catalog_panel.isVisible()
        assert not dialog.speed_select.isVisible() and not dialog.speed_label.isVisible()
        for control in (dialog.key_input, dialog.model_input, dialog.free_checkbox,
                        dialog.qwen_paid_checkbox, dialog.cloud_checkbox, dialog.evidence_checkbox):
            assert control.isVisible()
            top = control.mapTo(viewport, QPoint(0, 0)).y()
            assert top >= 0
            assert top + control.height() <= viewport.height(), control.objectName() or type(control).__name__
        assert dialog.cloud_checkbox.isChecked() is False
        assert dialog.qwen_paid_checkbox.isChecked() is False
        assert dialog.key_input.text() == ""
    finally:
        dialog.close(); dialog.deleteLater(); qt_with_windows_font.processEvents()


def test_codex_catalog_and_speed_remain_visible_and_optional_catalog_can_expand(qt_with_windows_font):
    dialog = ModelPickerDialog(Preferences(provider="codex", cloud_consent=False))
    try:
        dialog.resize(960, 760); dialog.show(); QTest.qWait(20)
        assert dialog.catalog_panel.isVisible()
        assert dialog.speed_select.isVisible() and dialog.speed_label.isVisible()
        dialog.provider_select.setCurrentIndex(dialog.provider_select.findData("qwen"))
        assert not dialog.catalog_panel.isVisible()
        dialog.catalog_toggle.click()
        assert dialog.catalog_panel.isVisible() and dialog.model_list.isVisible()
        dialog.catalog_toggle.click()
        assert not dialog.catalog_panel.isVisible()
    finally:
        dialog.close(); dialog.deleteLater(); qt_with_windows_font.processEvents()
