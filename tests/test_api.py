import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api_main

FAKE_FEATURE_COLUMNS = ["resting_hr", "nightly_temperature", "hrv_rmssd"]


class _FakeBooster:
    def predict(self, X):
        n = len(X)
        # fixed, deterministic 4-class distribution (Menstrual, Follicular, Fertility, Luteal)
        return np.tile(np.array([0.10, 0.20, 0.15, 0.55]), (n, 1))


def _fake_load_model_and_features():
    return _FakeBooster(), FAKE_FEATURE_COLUMNS


def _fake_explain_one(feature_dict, model_path=None, top_k=5):
    return [
        {"feature": "resting_hr", "shap": 0.42, "value": feature_dict.get("resting_hr"), "direction": "increases"},
        {"feature": "nightly_temperature", "shap": -0.31,
         "value": feature_dict.get("nightly_temperature"), "direction": "decreases"},
        {"feature": "hrv_rmssd", "shap": 0.05, "value": feature_dict.get("hrv_rmssd"), "direction": "increases"},
    ][:top_k]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(api_main, "_load_model_and_features", _fake_load_model_and_features)
    monkeypatch.setattr(api_main, "explain_one", _fake_explain_one)
    api_main._explain_cache.clear()
    with TestClient(api_main.app) as test_client:
        yield test_client


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_phase_and_probabilities(client):
    response = client.post("/predict", json={
        "features": {"resting_hr": 62.0, "nightly_temperature": 33.5, "hrv_rmssd": 40.0},
    })
    assert response.status_code == 200
    body = response.json()

    assert body["phase_label"] in {"Menstrual", "Follicular", "Fertility", "Luteal"}

    probs = body["probabilities"]
    assert set(probs.keys()) == {"Menstrual", "Follicular", "Fertility", "Luteal"}
    assert abs(sum(probs.values()) - 1.0) < 0.05

    assert 0 < len(body["top_drivers"]) <= 5
    for driver in body["top_drivers"]:
        assert driver["direction"] in {"up", "down"}
        assert isinstance(driver["weight"], float)


def test_predict_works_with_partial_feature_subset(client):
    response = client.post("/predict", json={"features": {"resting_hr": 62.0}})
    assert response.status_code == 200
    body = response.json()
    assert body["phase_label"] in {"Menstrual", "Follicular", "Fertility", "Luteal"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.05


def test_explain_with_both_keys_unset_uses_template(client, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = client.post("/explain", json={
        "phase_label": "Luteal",
        "top_drivers": [{"feature": "resting_hr", "direction": "up", "weight": 0.42}],
    })
    assert response.status_code == 200
    body = response.json()

    assert body["source_url"] is None
    assert body["source_title"] is None
    assert "Research prototype, not medical advice." in body["sentence"]
    assert len(body["sentence"]) > 0


def test_explain_with_stubbed_tavily_returns_source_url(client, monkeypatch):
    monkeypatch.setattr(api_main, "_tavily_search", lambda query: {
        "snippet": "The luteal phase is characterized by elevated progesterone and basal body temperature.",
        "url": "https://www.ncbi.nlm.nih.gov/example",
        "title": "Luteal phase physiology",
    })
    monkeypatch.setattr(
        api_main, "_openai_phrase",
        lambda phase_label, top_drivers, snippet: "Elevated resting heart rate is consistent with the luteal phase.",
    )

    response = client.post("/explain", json={
        "phase_label": "Luteal",
        "top_drivers": [{"feature": "resting_hr", "direction": "up", "weight": 0.42}],
    })
    assert response.status_code == 200
    body = response.json()

    assert body["source_url"] == "https://www.ncbi.nlm.nih.gov/example"
    assert body["source_title"] == "Luteal phase physiology"
    assert "Research prototype, not medical advice." in body["sentence"]


def test_explain_caches_identical_requests(client, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    payload = {
        "phase_label": "Follicular",
        "top_drivers": [{"feature": "nightly_temperature", "direction": "down", "weight": 0.31}],
    }
    first = client.post("/explain", json=payload).json()
    second = client.post("/explain", json=payload).json()
    assert first == second

    cache_key = ("Follicular", (("nightly_temperature", "down", 0.31),))
    assert cache_key in api_main._explain_cache
