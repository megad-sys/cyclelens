import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import src.agent as agent


# ---------------------------------------------------------------------------
# Fake OpenAI client scaffolding (mirrors the real SDK's response shape just
# enough for src.agent's loop -- .choices[0].message.{content,tool_calls},
# message.model_dump(), tool_call.id / .function.{name,arguments}).
# ---------------------------------------------------------------------------

class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        d = {"role": "assistant", "content": self.content, "tool_calls": None}
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in self.tool_calls
            ]
        return {k: v for k, v in d.items() if v is not None} if exclude_none else d


class _FakeResponse:
    def __init__(self, message: _FakeMessage):
        self.choices = [SimpleNamespace(message=message)]


class _ScriptedFakeClient:
    """Plays back a fixed sequence of messages, one per .create() call."""

    def __init__(self, messages: list[_FakeMessage]):
        self._messages = list(messages)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        assert self._messages, "fake client ran out of scripted responses"
        return _FakeResponse(self._messages.pop(0))


class _AggressiveFakeClient:
    """Always wants to call search_medical when tools are allowed; only
    answers with text once forced via tool_choice='none' -- simulates a
    model that would happily call tools forever if not capped."""

    def __init__(self):
        self.tool_choices_seen: list[str] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, tool_choice, **kwargs):
        self.tool_choices_seen.append(tool_choice)
        if tool_choice == "none":
            return _FakeResponse(_FakeMessage(content="Final answer."))
        call = _FakeToolCall(f"call_{len(self.tool_choices_seen)}", "search_medical",
                              json.dumps({"query": "luteal phase physiology"}))
        return _FakeResponse(_FakeMessage(tool_calls=[call]))


# ---------------------------------------------------------------------------
# (a) a phase question triggers predict_phase
# ---------------------------------------------------------------------------

def test_phase_question_triggers_predict_phase_tool(monkeypatch):
    predict_calls = []

    def fake_predict_phase(features):
        predict_calls.append(features)
        return {"phase_label": "Luteal", "probabilities": {"Luteal": 0.7}}

    monkeypatch.setattr(agent, "predict_phase", fake_predict_phase)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(agent, "_create_openai_client", lambda: _ScriptedFakeClient([
        _FakeMessage(tool_calls=[_FakeToolCall("call_1", "predict_phase", "{}")]),
        _FakeMessage(content="You're likely in the luteal phase."),
    ]))

    result = agent.answer_question("What phase am I in today?", {"resting_hr": 60.0})

    assert len(predict_calls) == 1
    assert predict_calls[0] == {"resting_hr": 60.0}
    assert agent.DISCLAIMER in result["answer"]


# ---------------------------------------------------------------------------
# (b) the answer always contains the disclaimer
# ---------------------------------------------------------------------------

def test_disclaimer_always_appended_to_llm_answer(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(agent, "_create_openai_client", lambda: _ScriptedFakeClient([
        _FakeMessage(content="This is a plain answer with no disclaimer text at all."),
    ]))

    result = agent.answer_question("Tell me about my cycle.", {"resting_hr": 60.0})

    assert agent.DISCLAIMER in result["answer"]


def test_disclaimer_present_in_templated_fallback(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(agent, "predict_phase", lambda features: {
        "phase_label": "Follicular", "probabilities": {"Follicular": 0.5},
    })
    monkeypatch.setattr(agent, "explain_prediction", lambda features: {
        "top_drivers": [{"feature": "resting_hr", "shap": 0.2, "value": 60.0, "direction": "increases"}],
    })

    result = agent.answer_question("What phase am I in?", {"resting_hr": 60.0})

    assert agent.DISCLAIMER in result["answer"]
    assert result["source_url"] is None


def test_disclaimer_present_when_declining_diagnosis_request(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    result = agent.answer_question("Can you diagnose me and prescribe treatment?", {"resting_hr": 60.0})

    assert agent.DISCLAIMER in result["answer"]
    assert result["source_url"] is None


# ---------------------------------------------------------------------------
# (c) the agent stops at/under the 4-call cap
# ---------------------------------------------------------------------------

def test_agent_stops_at_tool_call_cap(monkeypatch):
    search_calls = []

    def fake_search_medical(query):
        search_calls.append(query)
        return {"snippet": "info", "source_url": "https://acog.org/example", "source_title": "ACOG"}

    monkeypatch.setattr(agent, "search_medical", fake_search_medical)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    fake_client = _AggressiveFakeClient()
    monkeypatch.setattr(agent, "_create_openai_client", lambda: fake_client)

    result = agent.answer_question("Tell me everything about my cycle.", {"resting_hr": 60.0})

    assert len(search_calls) <= agent.MAX_TOOL_CALLS
    assert len(search_calls) == agent.MAX_TOOL_CALLS  # the aggressive client always wants more
    assert "none" in fake_client.tool_choices_seen  # forced to stop and answer with text
    assert agent.DISCLAIMER in result["answer"]
    assert result["source_url"] == "https://acog.org/example"


# ---------------------------------------------------------------------------
# (d), (e): /ask via the FastAPI app
# ---------------------------------------------------------------------------

class _FakeBooster:
    def predict(self, X):
        return np.tile(np.array([0.10, 0.20, 0.15, 0.55]), (len(X), 1))


FAKE_FEATURE_COLUMNS = ["resting_hr", "nightly_temperature", "hrv_rmssd"]


def _fake_load_model_and_features():
    return _FakeBooster(), FAKE_FEATURE_COLUMNS


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(api_main, "_load_model_and_features", _fake_load_model_and_features)
    monkeypatch.setattr(agent, "predict_phase", lambda features: {
        "phase_label": "Luteal", "probabilities": {"Luteal": 0.7},
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
    monkeypatch.setattr(agent, "_create_openai_client", lambda: _ScriptedFakeClient([
        _FakeMessage(content="You're likely in the luteal phase based on the data provided."),
    ]))

    response = client.post("/ask", json={
        "question": "What phase am I in today?",
        "features": {"resting_hr": 60.0},
    })

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"answer", "source_url"}
    assert isinstance(body["answer"], str)
    assert body["source_url"] is None or isinstance(body["source_url"], str)
    assert agent.DISCLAIMER in body["answer"]
