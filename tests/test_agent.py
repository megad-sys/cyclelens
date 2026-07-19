from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import src.agent as agent


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=_FakeMessage(content))]


def _fake_groq(content: str):
    def _groq_chat_completion(**kwargs):
        return _FakeResponse(content)
    return _groq_chat_completion


def test_direct_answer_calls_model_and_search(monkeypatch):
    predict_calls = []
    search_calls = []

    monkeypatch.setattr(agent, "predict_phase", lambda features: (
        predict_calls.append(features) or {
            "phase_label": "Fertility",
            "probabilities": {"Menstrual": 0.1, "Follicular": 0.2, "Fertility": 0.55, "Luteal": 0.15},
        }
    ))
    monkeypatch.setattr(agent, "explain_prediction", lambda features: {
        "top_drivers": [{"feature": "hrv_rmssd", "shap": 0.3, "value": 70.0, "direction": "increases"}],
    })
    monkeypatch.setattr(agent, "search_medical", lambda query: (
        search_calls.append(query) or {
            "snippet": "Ovulation typically occurs mid-cycle.",
            "source_url": "https://www.ncbi.nlm.nih.gov/example",
            "source_title": "Example",
        }
    ))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(
        agent,
        "groq_chat_completion",
        _fake_groq(
            "The model assigns a 55% probability to the fertile window today, which suggests ovulation is possible but not certain."
        ),
    )

    result = agent.answer_question("Am I likely ovulating today?", {"resting_hr": 60.0})

    assert len(predict_calls) == 1
    assert len(search_calls) == 1
    assert "55%" in result["answer"] or "ovulation" in result["answer"].lower() or "fertile" in result["answer"].lower()
    assert agent.DISCLAIMER in result["answer"]
    assert result["source_url"] == "https://www.ncbi.nlm.nih.gov/example"


def test_disclaimer_always_appended_to_llm_answer(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(agent, "predict_phase", lambda features: {
        "phase_label": "Luteal", "probabilities": {"Luteal": 0.7, "Fertility": 0.1, "Follicular": 0.1, "Menstrual": 0.1},
    })
    monkeypatch.setattr(agent, "explain_prediction", lambda features: {"top_drivers": []})
    monkeypatch.setattr(agent, "search_medical", lambda query: {"snippet": None, "source_url": None, "source_title": None})
    monkeypatch.setattr(agent, "groq_chat_completion", _fake_groq("This is a plain answer with no disclaimer text at all."))

    result = agent.answer_question("Tell me about my cycle.", {"resting_hr": 60.0})

    assert agent.DISCLAIMER in result["answer"]
    assert "plain answer" in result["answer"]


def test_disclaimer_present_in_templated_fallback(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(agent, "predict_phase", lambda features: {
        "phase_label": "Follicular", "probabilities": {"Follicular": 0.5, "Fertility": 0.2, "Menstrual": 0.15, "Luteal": 0.15},
    })
    monkeypatch.setattr(agent, "explain_prediction", lambda features: {
        "top_drivers": [{"feature": "resting_hr", "shap": 0.2, "value": 60.0, "direction": "increases"}],
    })

    result = agent.answer_question("What phase am I in?", {"resting_hr": 60.0})

    assert agent.DISCLAIMER in result["answer"]
    assert result["source_url"] is None


def test_ovulation_question_template_mentions_fertility(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(agent, "predict_phase", lambda features: {
        "phase_label": "Fertility",
        "probabilities": {"Fertility": 0.44, "Follicular": 0.25, "Menstrual": 0.15, "Luteal": 0.16},
    })
    monkeypatch.setattr(agent, "explain_prediction", lambda features: {"top_drivers": []})

    result = agent.answer_question("Am I likely ovulating today?", {"resting_hr": 60.0})

    assert "44%" in result["answer"] or "fertile" in result["answer"].lower()


def test_disclaimer_present_when_declining_diagnosis_request(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    result = agent.answer_question("Can you diagnose me and prescribe treatment?", {"resting_hr": 60.0})

    assert agent.DISCLAIMER in result["answer"]
    assert result["source_url"] is None


class _FakeBooster:
    def predict(self, X):
        return np.tile(np.array([0.10, 0.20, 0.15, 0.55]), (len(X), 1))


FAKE_FEATURE_COLUMNS = ["resting_hr", "nightly_temperature", "hrv_rmssd"]


def _fake_load_model_and_features():
    return _FakeBooster(), FAKE_FEATURE_COLUMNS


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(api_main, "_load_model_and_features", _fake_load_model_and_features)
    monkeypatch.setattr(api_main, "test_groq_connection", lambda: "test_skipped")
    monkeypatch.setattr(agent, "predict_phase", lambda features: {
        "phase_label": "Luteal", "probabilities": {"Luteal": 0.7, "Fertility": 0.1, "Follicular": 0.1, "Menstrual": 0.1},
    })
    monkeypatch.setattr(agent, "explain_prediction", lambda features: {
        "top_drivers": [{"feature": "resting_hr", "shap": 0.2, "value": 60.0, "direction": "increases"}],
    })
    with TestClient(api_main.app) as test_client:
        yield test_client


def test_ask_with_both_keys_unset_uses_template(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    response = client.post("/ask", json={
        "question": "What phase am I in today?",
        "features": {"resting_hr": 60.0},
    })

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"answer", "source_url"}
    assert body["source_url"] is None
    assert agent.DISCLAIMER in body["answer"]
    assert isinstance(body["answer"], str) and len(body["answer"]) > 0


def test_ask_returns_200_with_correct_shape(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(agent, "search_medical", lambda query: {"snippet": None, "source_url": None, "source_title": None})
    monkeypatch.setattr(
        agent,
        "groq_chat_completion",
        _fake_groq("You're likely in the luteal phase based on the wearable readings provided."),
    )

    response = client.post("/ask", json={
        "question": "What phase am I in today?",
        "features": {"resting_hr": 60.0},
    })

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"answer", "source_url"}
    assert "luteal" in body["answer"].lower()
    assert agent.DISCLAIMER in body["answer"]
