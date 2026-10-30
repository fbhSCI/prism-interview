"""Audio-source labels are not acoustic identity claims."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from zoom_copilot.ui import CopilotWindow


def application():
    return QApplication.instance() or QApplication([])


def test_caption_speaker_mode_roundtrips_without_saving_or_restarting_capture():
    app = application()
    window = CopilotWindow()
    events = []
    window.settings_saved.connect(events.append)
    try:
        assert window.get_settings()["caption_speaker_mode"] == "single_remote"
        window.set_settings({"caption_speaker_mode": "single_remote"})
        assert window.caption_speaker_mode_select.currentText() == "1 人（默认）"
        assert window.get_settings()["caption_speaker_mode"] == "single_remote"
        assert not events
        tooltip = window.caption_speaker_mode_select.toolTip()
        assert "来源" in tooltip and "不代表" in tooltip
        assert "其他应用" in tooltip and "麦克风" in tooltip
        assert "重新开始监听" in tooltip

        window.set_listening(True)
        window.caption_speaker_mode_select.setCurrentIndex(window.caption_speaker_mode_select.findData("acoustic"))
        # Merely choosing a pending setting does not touch the active recorder.
        assert window._listening
        assert not events
        window._save_settings_clicked()
        assert len(events) == 1
        assert events[0]["caption_speaker_mode"] == "acoustic"
        assert window._listening
        assert "下次开始监听" in window.status_label.text()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_source_label_keeps_no_voice_id_and_source_tooltip_after_translation():
    app = application()
    window = CopilotWindow()
    label = "Zoom 远端（单一远端模式）"
    try:
        window.append_transcript("a", "How would you measure it?", turn_id="remote-1",
            speaker_id=None, speaker_label=label, speaker_status="source")
        item = window._entries["a"]
        assert item.speaker_id is None
        assert item.speaker.text() == label
        assert item.speaker_status == "source"
        assert "远端音频" in item.speaker.toolTip()
        assert "不是声纹身份确认" in item.speaker.toolTip()
        assert "人工指定" not in item.speaker.toolTip()
        window.update_translation("a", "你会如何测量它？")
        assert item.speaker_id is None
        assert item.speaker_status == "source"
        assert "远端音频" in item.speaker.toolTip()
        assert "远端音频" in window.live_speaker.toolTip()
        assert window.live_speaker.text() == label
        window.update_speaker("remote-1", None, label, status="source")
        assert "远端音频" in item.speaker.toolTip()
        assert "远端音频" in window.live_speaker.toolTip()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_acoustic_and_manual_status_remain_separate_from_source_mode():
    app = application()
    window = CopilotWindow()
    try:
        window.append_transcript("voice", "A question?", turn_id="voice-turn",
            speaker_id="speaker_2", speaker_label="说话人 2", speaker_status="confirmed")
        item = window._entries["voice"]
        assert "声纹" in item.speaker.toolTip()
        assert "远端音频" not in item.speaker.toolTip()
        window.update_speaker("voice-turn", "speaker_2", "教授", status="manual")
        window.update_translation("voice", "一个问题？")
        assert item.speaker_id == "speaker_2"
        assert item.speaker.toolTip() == "人工指定"
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
