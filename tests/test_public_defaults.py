"""Public installation boundaries and general-purpose candidate grounding."""
import json
from pathlib import Path
import pytest

from zoom_copilot.config import Preferences, APP_NAME, app_dir
from zoom_copilot.runtime import default_model_cache
from zoom_copilot.credential_store import credential_target
from zoom_copilot.candidate_profile import personal_question_kind, candidate_context
from zoom_copilot.answers import build_prompt
from zoom_copilot.knowledge import KnowledgeBase


def test_fresh_defaults_are_empty_private_and_do_not_authorize_cloud_or_history():
    prefs = Preferences()
    assert APP_NAME == "PrismInterview"
    assert prefs.candidate_profile == prefs.research_profile == ""
    assert not prefs.cloud_consent and not prefs.translation_cloud_consent
    assert not prefs.free_quota_confirmed and not prefs.qwen_paid_usage_authorized
    assert not prefs.auto_save_sessions and not prefs.api_key and not prefs.session_api_keys


def test_new_home_and_model_environment_are_explicit(tmp_path, monkeypatch):
    home, models = tmp_path / "public-data", tmp_path / "public-models"
    monkeypatch.setenv("PRISM_INTERVIEW_HOME", str(home))
    monkeypatch.setenv("PRISM_INTERVIEW_MODELS", str(models))
    assert app_dir() == home and default_model_cache() == models
    assert credential_target("qwen", "https://dashscope.aliyuncs.com/v1").startswith("PrismInterview/API/v1/")


@pytest.mark.parametrize("question", [
    "Tell me about yourself.", "What are your strengths?", "Describe your project experience.",
    "Why should we hire you?", "Why do you want this role?",
])
def test_general_career_questions_require_candidate_facts(question):
    assert personal_question_kind(question) != "scientific"
    _, user = build_prompt(question, [], Preferences())
    data = json.loads(user)
    assert not data["candidate_confirmed_facts"]["entries"]


def test_job_project_facts_drive_answers_without_private_field_defaults():
    profile = "Personal contribution: I led a customer onboarding redesign. Source: resume.pdf p1."
    _, user = build_prompt("Describe your customer onboarding project.", [], Preferences(candidate_profile=profile))
    assert json.loads(user)["candidate_confirmed_facts"]["entries"][0]["text"] == profile
    assert "ghost imaging" not in user.lower() and "inverse rendering" not in user.lower()


def test_general_chinese_terms_find_user_supplied_business_material(tmp_path):
    path = tmp_path / "resume.txt"
    path.write_text("Led project management and supply chain planning for a logistics team.", encoding="utf8")
    kb = KnowledgeBase(tmp_path / "knowledge.json")
    assert kb.add_files([str(path)])[0] == 1
    assert kb.search("你的项目管理与供应链经历是什么？")


@pytest.mark.parametrize("source,translated", [
    ("Political polarization is increasing.", "政治极化正在加剧。"),
    ("The electrode polarization affects the cell.", "电极极化影响电池。"),
    ("We discussed polarization.", "我们讨论了极化。"),
    ("Political polarization affects our camera policy.", "政治极化影响了我们的相机政策。"),
])
def test_unrelated_polarization_is_not_rewritten_as_optics(source, translated):
    from zoom_copilot.translation_protection import normalize_research_translation
    from zoom_copilot.translation import _correct_known_terms
    assert normalize_research_translation(source, translated) == translated
    assert _correct_known_terms(source, translated) == translated


@pytest.mark.parametrize("question", [
    "You managed the customer project, right?", "You led a sales team, right?",
    "You balanced the budget, right?", "You worked with patients, right?",
])
def test_tag_confirmation_questions_keep_general_interview_content(question):
    from zoom_copilot.interview_context import analyze_question, question_parts
    assert analyze_question(question).interaction_kind != "setup"
    assert question_parts(question)


@pytest.mark.parametrize("value", [[], None, "not settings", 12])
def test_invalid_json_preferences_do_not_crash_startup(value, tmp_path, monkeypatch):
    monkeypatch.setenv("PRISM_INTERVIEW_HOME", str(tmp_path))
    (tmp_path / "preferences.json").write_text(json.dumps(value), encoding="utf8")
    prefs = Preferences.load()
    assert not prefs.cloud_consent and not prefs.auto_save_sessions


@pytest.mark.parametrize("value", ["false", "true", "yes", 1, 0, [], {}, None])
def test_truthy_malformed_preferences_never_authorize_cloud_billing_or_sharing(value):
    names = ("cloud_consent", "translation_cloud_consent", "free_quota_confirmed",
        "qwen_paid_usage_authorized", "share_evidence", "auto_save_sessions")
    prefs = Preferences(**{name: value for name in names})
    assert all(getattr(prefs, name) is False for name in names)
