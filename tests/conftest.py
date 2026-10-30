"""Tests must not discover or use a user's saved provider credentials."""
import pytest


@pytest.fixture(autouse=True)
def isolate_saved_api_credentials(monkeypatch):
    from zoom_copilot import credential_store, providers
    monkeypatch.setattr(credential_store, 'load_key', lambda provider, endpoint: '')
    monkeypatch.setattr(providers, 'load_key', lambda provider, endpoint: '')


@pytest.fixture(autouse=True)
def isolate_application_data_and_provider_environment(tmp_path, monkeypatch):
    # Public test runs must not read/write a real installation or pick up paid keys.
    monkeypatch.setenv("PRISM_INTERVIEW_HOME", str(tmp_path / "app-data"))
    monkeypatch.setenv("PRISM_INTERVIEW_MODELS", str(tmp_path / "models"))
    for name in ("DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY", "PRISM_INTERVIEW_API_KEY"):
        monkeypatch.delenv(name, raising=False)
