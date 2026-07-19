"""Inference API for the mcPHASES cycle-phase model.

POST /predict: features (any subset, model-order-filled with NaN) -> phase +
probabilities + top-5 SHAP drivers (via src.explain.explain_one).
POST /explain: phase + drivers -> ONE grounded plain-language sentence
(Tavily search for medical context, then one Groq LLM call to phrase it).
POST /ask: free-form question + features -> ONE grounded answer, via the
tool-calling agent in src.agent (predict_phase / explain_prediction /
search_medical, capped at 4 tool calls, diagnosis/treatment requests declined).
Both TAVILY_API_KEY and GROQ_API_KEY are optional -- their absence degrades
gracefully (skipped grounding / templated sentence), never a hard failure.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent import answer_question
from src.explain import explain_one
from src.splits import LABEL_NAMES

load_dotenv()

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/lgbm.txt"))
FEATURE_NAMES_PATH = Path(os.environ.get("FEATURE_NAMES_PATH", "models/feature_names.json"))
FEATURE_STATS_PATH = Path(os.environ.get("FEATURE_STATS_PATH", "models/feature_stats.json"))

LABEL_VALUES = sorted(LABEL_NAMES)
ALLOWED_GROUNDING_DOMAINS = ["acog.org", "ncbi.nlm.nih.gov", "nih.gov", "mayoclinic.org"]
DISCLAIMER = "Research prototype, not medical advice."

_MODEL_STATE: dict = {"booster": None, "feature_columns": None, "feature_stats": None}
_explain_cache: dict[tuple, dict] = {}


def _load_model_and_features() -> tuple[lgb.Booster, list[str]]:
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    if FEATURE_NAMES_PATH.exists():
        feature_columns = json.loads(FEATURE_NAMES_PATH.read_text())
    else:
        feature_columns = booster.feature_name()
        FEATURE_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
        FEATURE_NAMES_PATH.write_text(json.dumps(feature_columns, indent=2))
    return booster, feature_columns


def _load_feature_stats() -> dict:
    """Population mean/std per raw feature (scripts/make_feature_stats.py).
    Missing file degrades to {} -- the _pz/_roll3 fallback just no-ops."""
    if not FEATURE_STATS_PATH.exists():
        return {}
    return json.loads(FEATURE_STATS_PATH.read_text())


@asynccontextmanager
async def lifespan(app: FastAPI):
    _MODEL_STATE["booster"], _MODEL_STATE["feature_columns"] = _load_model_and_features()
    _MODEL_STATE["feature_stats"] = _load_feature_stats()
    yield
    _MODEL_STATE["booster"] = None
    _MODEL_STATE["feature_columns"] = None
    _MODEL_STATE["feature_stats"] = None


app = FastAPI(title="mcPHASES Cycle-Phase API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    features: dict[str, float | None] = Field(default_factory=dict)


class TopDriver(BaseModel):
    feature: str
    direction: Literal["up", "down"]
    weight: float


class PredictResponse(BaseModel):
    phase_label: str
    probabilities: dict[str, float]
    top_drivers: list[TopDriver]


class ExplainRequest(BaseModel):
    phase_label: str
    top_drivers: list[TopDriver]


class ExplainResponse(BaseModel):
    sentence: str
    source_title: str | None
    source_url: str | None


class AskRequest(BaseModel):
    question: str
    features: dict[str, float | None] = Field(default_factory=dict)


class AskResponse(BaseModel):
    answer: str
    source_url: str | None


def _build_feature_row(features: dict, feature_columns: list[str]) -> pd.DataFrame:
    row = {col: features.get(col) for col in feature_columns}
    df = pd.DataFrame([row], columns=feature_columns)
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _fill_derived_features(features: dict, feature_columns: list[str], feature_stats: dict) -> dict:
    """Fill missing _pz (population z-score, NOT the participant's own baseline
    -- see README) and _roll3 (single-day proxy: just the raw value) derived
    features from raw values already present in the request. Only fills what's
    missing -- never overwrites a caller-provided value, so a full feature row
    (e.g. from the sample days) passes through unchanged."""
    filled = dict(features)

    for col in feature_columns:
        if col.endswith("_pz"):
            if filled.get(col) is not None:
                continue
            base_col = col[: -len("_pz")]
            base_value = filled.get(base_col)
            stats = feature_stats.get(base_col)
            if base_value is None or not stats:
                continue
            mean, std = stats.get("mean"), stats.get("std")
            if mean is None or not std:  # skip if std is 0/None
                continue
            try:
                filled[col] = (float(base_value) - mean) / std
            except (TypeError, ValueError):
                continue
        elif col.endswith("_roll3"):
            if filled.get(col) is not None:
                continue
            base_value = filled.get(col[: -len("_roll3")])
            if base_value is not None:
                filled[col] = base_value

    return filled


def _groq_api_key() -> str | None:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    return key or None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "groq_configured": _groq_api_key() is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict:
    booster = _MODEL_STATE["booster"]
    feature_columns = _MODEL_STATE["feature_columns"]
    feature_stats = _MODEL_STATE["feature_stats"] or {}

    filled_features = _fill_derived_features(request.features, feature_columns, feature_stats)
    row = _build_feature_row(filled_features, feature_columns)
    proba = booster.predict(row)[0]
    predicted_idx = int(np.argmax(proba))
    predicted_label = LABEL_NAMES[LABEL_VALUES[predicted_idx]]

    probabilities = {
        LABEL_NAMES[label_value]: round(float(p), 2)
        for label_value, p in zip(LABEL_VALUES, proba)
    }

    drivers = explain_one(filled_features, model_path=str(MODEL_PATH), top_k=5)
    top_drivers = [
        {
            "feature": d["feature"],
            "direction": "up" if d["shap"] >= 0 else "down",
            "weight": round(abs(d["shap"]), 4),
        }
        for d in drivers
    ]

    return {"phase_label": predicted_label, "probabilities": probabilities, "top_drivers": top_drivers}


def _tavily_search(query: str) -> dict | None:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return None
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query, search_depth="basic", max_results=3, include_domains=ALLOWED_GROUNDING_DOMAINS,
        )
        results = response.get("results", [])
        if not results:
            return None
        top = results[0]
        return {"snippet": top.get("content", ""), "url": top.get("url", ""), "title": top.get("title", "")}
    except Exception as exc:  # invalid/expired key, rate limit, network issue -- never hard-fail
        print(f"[api] _tavily_search failed, degrading gracefully: {exc}")
        return None


def _template_sentence(phase_label: str, top_drivers: list[dict]) -> str:
    if top_drivers:
        driver_names = ", ".join(d["feature"] for d in top_drivers[:3])
        return f"Based on {driver_names}, the model predicts the {phase_label} phase."
    return f"The model predicts the {phase_label} phase."


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"


def _openai_phrase(phase_label: str, top_drivers: list[dict], snippet: str | None) -> str:
    api_key = _groq_api_key()
    if not api_key:
        print("[llm] GROQ_API_KEY not set — using template fallback")
        return _template_sentence(phase_label, top_drivers)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        driver_text = (
            ", ".join(f"{d['feature']} ({d['direction']})" for d in top_drivers[:5]) or "no drivers provided"
        )
        context = snippet or "no external medical reference is available"
        prompt = (
            f"A wearable-based model predicts the menstrual cycle phase '{phase_label}', driven mainly by: "
            f"{driver_text}. Medical reference context: {context}\n\n"
            "Write ONE plain-language sentence explaining why these wearable signals point to this phase, "
            "using ONLY the medical reference context above. Do not add unsupported medical claims."
        )
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # invalid/expired key, no credit, rate limit -- never hard-fail
        print(f"[api] _openai_phrase failed, degrading gracefully: {exc}")
        return _template_sentence(phase_label, top_drivers)


@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest) -> dict:
    top_drivers = [d.model_dump() for d in request.top_drivers]
    cache_key = (request.phase_label, tuple((d["feature"], d["direction"], d["weight"]) for d in top_drivers))
    if cache_key in _explain_cache:
        return _explain_cache[cache_key]

    top_driver_name = top_drivers[0]["feature"] if top_drivers else ""
    grounding = _tavily_search(f"{request.phase_label} menstrual cycle phase physiology {top_driver_name}")
    snippet = grounding["snippet"] if grounding else None

    sentence = _openai_phrase(request.phase_label, top_drivers, snippet)
    if DISCLAIMER not in sentence:
        sentence = f"{sentence.rstrip()} {DISCLAIMER}"

    result = {
        "sentence": sentence,
        "source_title": grounding["title"] if grounding else None,
        "source_url": grounding["url"] if grounding else None,
    }
    # Don't cache template fallbacks — once GROQ_API_KEY is fixed on Render, the next
    # identical request should get a real LLM sentence instead of a stale template.
    template = _template_sentence(request.phase_label, top_drivers)
    if not sentence.startswith(template.rstrip(".")):
        _explain_cache[cache_key] = result
    return result


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict:
    return answer_question(request.question, request.features)
