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


# ---------------------------------------------------------------------------
# Fixture for the on-the-fly _pz fill: a booster whose output actually
# depends on nightly_temperature_pz / hrv_rmssd_pz, so a test can prove the
# fill from feature_stats.json reaches the model and changes the prediction.
# ---------------------------------------------------------------------------

RESPONSIVE_FEATURE_COLUMNS = ["nightly_temperature", "nightly_temperature_pz", "hrv_rmssd", "hrv_rmssd_pz"]
FAKE_FEATURE_STATS = {
    "nightly_temperature": {"mean": 34.0, "std": 0.5},
    "hrv_rmssd": {"mean": 55.0, "std": 15.0},
}


class _ResponsiveFakeBooster:
    def predict(self, X):
        temp_pz = X["nightly_temperature_pz"].fillna(0.0).to_numpy()
        hrv_pz = X["hrv_rmssd_pz"].fillna(0.0).to_numpy()
        score = np.clip((temp_pz - hrv_pz) * 0.1, -0.2, 0.2)
        out = np.tile(np.array([0.25, 0.25, 0.25, 0.25]), (len(X), 1))
        out[:, 0] -= score  # Menstrual
        out[:, 3] += score  # Luteal
        out = np.clip(out, 0.01, None)
        return out / out.sum(axis=1, keepdims=True)


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


def test_health_ok(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["groq_configured"] is False


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


def test_predict_probabilities_differ_with_different_temperature_and_hrv(monkeypatch):
    # Proves the on-the-fly _pz fill (from feature_stats.json) actually reaches
    # the model: two requests differing only in nightly_temperature/hrv_rmssd
    # must produce different _pz values and therefore different probabilities.
    monkeypatch.setattr(api_main, "_load_model_and_features",
                         lambda: (_ResponsiveFakeBooster(), RESPONSIVE_FEATURE_COLUMNS))
    monkeypatch.setattr(api_main, "_load_feature_stats", lambda: FAKE_FEATURE_STATS)
    monkeypatch.setattr(api_main, "explain_one", lambda features, model_path=None, top_k=5: [])

    with TestClient(api_main.app) as test_client:
        low = test_client.post("/predict", json={
            "features": {"nightly_temperature": 33.0, "hrv_rmssd": 75.0},
        }).json()
        high = test_client.post("/predict", json={
            "features": {"nightly_temperature": 35.0, "hrv_rmssd": 35.0},
        }).json()

    assert low["probabilities"] != high["probabilities"]


def test_explain_with_both_keys_unset_uses_template(client, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

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


def test_explain_degrades_gracefully_when_tavily_key_set_but_call_fails(client, monkeypatch):
    # Regression test: a key being SET but the API call itself failing (invalid/expired
    # key, no credit, rate limit, network issue) must degrade gracefully, not 500.
    monkeypatch.setenv("TAVILY_API_KEY", "invalid-key")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    class _RaisingTavilyClient:
        def __init__(self, api_key):
            raise RuntimeError("401 Unauthorized: invalid API key")

    monkeypatch.setattr("tavily.TavilyClient", _RaisingTavilyClient)

    response = client.post("/explain", json={
        "phase_label": "Luteal",
        "top_drivers": [{"feature": "resting_hr", "direction": "up", "weight": 0.42}],
    })

    assert response.status_code == 200
    body = response.json()
    assert body["source_url"] is None
    assert "Research prototype, not medical advice." in body["sentence"]


def test_explain_degrades_gracefully_when_groq_key_set_but_call_fails(client, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "invalid-key")

    class _RaisingOpenAIClient:
        def __init__(self, api_key):
            raise RuntimeError("401 Unauthorized: invalid API key")

    monkeypatch.setattr("openai.OpenAI", _RaisingOpenAIClient)

    response = client.post("/explain", json={
        "phase_label": "Luteal",
        "top_drivers": [{"feature": "resting_hr", "direction": "up", "weight": 0.42}],
    })

    assert response.status_code == 200
    body = response.json()
    assert "Research prototype, not medical advice." in body["sentence"]


def test_explain_caches_identical_requests(client, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(
        api_main,
        "_openai_phrase",
        lambda phase_label, top_drivers, snippet: "Elevated temperature is consistent with the follicular phase.",
    )

    payload = {
        "phase_label": "Follicular",
        "top_drivers": [{"feature": "nightly_temperature", "direction": "down", "weight": 0.31}],
    }
    first = client.post("/explain", json=payload).json()
    second = client.post("/explain", json=payload).json()
    assert first == second

    cache_key = ("Follicular", (("nightly_temperature", "down", 0.31),))
    assert cache_key in api_main._explain_cache
