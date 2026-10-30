"""Public edition presentation contracts; no audio, cloud calls or private data."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from zoom_copilot.ui import CopilotWindow


def test_conversation_persistence_is_explicit_and_off_by_default():
    app = QApplication.instance() or QApplication([])
    window = CopilotWindow()
    try:
        assert window.get_settings()['auto_save_sessions'] is False
        actions = []
        window.save_session_requested.connect(lambda: actions.append('save'))
        window.history_requested.connect(lambda: actions.append('history'))
        window.auto_save_changed.connect(lambda value: actions.append(value))
        window.save_session_button.click()
        window.session_history_button.click()
        assert actions == ['save', 'history']
        window.set_auto_save(True)
        assert window.auto_save_checkbox.isChecked()
        assert actions == ['save', 'history']
        window.auto_save_checkbox.click()
        assert actions == ['save', 'history', False]
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_public_ui_defaults_are_transparent_about_cloud_caption_authorization():
    app = QApplication.instance() or QApplication([])
    window = CopilotWindow()
    try:
        settings = window.get_settings()
        assert settings['research_profile'] == ''
        assert settings['candidate_profile'] == ''
        assert settings['cloud_consent'] is False
        assert settings['translation_cloud_consent'] is True
        assert '发送至千问' in window.caption_cloud_disclosure.text()
        assert '可能产生费用' in window.caption_cloud_disclosure.text()
        assert settings['api_key'] == ''
        assert settings['translation_qwen_base_url'] == 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        assert not window._listening
        window.set_settings({'auto_save_sessions': True})
        assert window.auto_save_checkbox.isChecked()
        window.set_settings({'auto_save_sessions': False})
        assert not window.auto_save_checkbox.isChecked()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_compatible_picker_result_roundtrips_through_main_ui():
    app = QApplication.instance() or QApplication([])
    window = CopilotWindow()
    try:
        window.set_settings({'provider': 'compatible', 'api_base_url': 'https://provider.example/v1',
                             'answer_model': 'example-model', 'cloud_consent': True,
                             'api_key': 'synthetic-only'})
        data = window.get_settings()
        assert data['provider'] == 'openai_compatible'
        assert data['api_base_url'] == 'https://provider.example/v1'
        assert data['answer_model'] == 'example-model'
        assert data['api_key'] == 'synthetic-only'
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_user_endpoint_edit_revokes_old_destination_key_and_consent():
    from PySide6.QtTest import QTest
    app = QApplication.instance() or QApplication([])
    window = CopilotWindow()
    try:
        window.set_settings({'provider': 'compatible', 'api_base_url': 'https://old.example/v1',
                             'api_key': 'synthetic-old-endpoint-only', 'cloud_consent': True,
                             'free_quota_confirmed': True, 'qwen_paid_usage_authorized': True,
                             'session_api_keys': {'compatible': 'synthetic-old-endpoint-only',
                                                  'deepseek': 'synthetic-other-provider'}})
        revoked = []
        window.settings_saved.connect(revoked.append)
        window.api_base_input.selectAll()
        QTest.keyClicks(window.api_base_input, 'https://new.example/v1')
        data = window.get_settings()
        assert data['api_key'] == ''
        assert 'compatible' not in data['session_api_keys']
        assert data['session_api_keys']['deepseek'] == 'synthetic-other-provider'
        assert data['cloud_consent'] is False
        assert data['free_quota_confirmed'] is False
        assert data['qwen_paid_usage_authorized'] is False
        assert len(revoked) == 1
        window.provider_select.setCurrentIndex(window.provider_select.findData('ollama'))
        window.provider_select.setCurrentIndex(window.provider_select.findData('openai_compatible'))
        assert window.api_key_input.text() == ''
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
