"""Grounded Q&A for one participant-day: model prediction + SHAP drivers + Tavily, then one Groq call.

Uses a single LLM request (not multi-turn tool calling) for reliability on Groq/Llama.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from src.explain import DEFAULT_MODEL_PATH, LABEL_NAMES, LABEL_VALUES, _get_explainer, explain_one
from src.feature_labels import humanize_feature, narrative_drivers
from src.groq_client import groq_api_key, groq_chat_completion

DISCLAIMER = "Research prototype, not medical advice."
ALLOWED_GROUNDING_DOMAINS = ["acog.org", "ncbi.nlm.nih.gov", "nih.gov", "mayoclinic.org"]

DIAGNOSIS_KEYWORDS = [
    "diagnos", "treatment for", "treat me", "prescri", "medication", "what drug",
    "cure for", "should i take", "birth control", "what dose",
]


def _build_row(features: dict, feature_columns: list[str]) -> pd.DataFrame:
    row = {col: features.get(col) for col in feature_columns}
    df = pd.DataFrame([row], columns=feature_columns)
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def predict_phase(features: dict) -> dict:
    _, booster, feature_columns = _get_explainer(str(DEFAULT_MODEL_PATH))
    row = _build_row(features, feature_columns)
    proba = booster.predict(row)[0]
    probabilities = {LABEL_NAMES[label]: round(float(p), 4) for label, p in zip(LABEL_VALUES, proba)}
    predicted_label = LABEL_NAMES[LABEL_VALUES[int(np.argmax(proba))]]
    return {"phase_label": predicted_label, "probabilities": probabilities}


def explain_prediction(features: dict) -> dict:
    drivers = explain_one(features, model_path=str(DEFAULT_MODEL_PATH), top_k=5)
    return {"top_drivers": drivers}


def search_medical(query: str) -> dict:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {"snippet": None, "source_url": None, "source_title": None}
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query, search_depth="basic", max_results=3, include_domains=ALLOWED_GROUNDING_DOMAINS,
        )
        results = response.get("results", [])
        if not results:
            return {"snippet": None, "source_url": None, "source_title": None}
        top = results[0]
        return {
            "snippet": top.get("content", ""),
            "source_url": top.get("url", ""),
            "source_title": top.get("title", ""),
        }
    except Exception as exc:
        print(f"[agent] search_medical failed: {exc}")
        return {"snippet": None, "source_url": None, "source_title": None}


def _strip_disclaimer(text: str) -> str:
    return text.replace(DISCLAIMER, "").strip()


def _is_substantive(text: str) -> bool:
    return len(_strip_disclaimer(text)) >= 20


def _templated_answer(features: dict, question: str | None = None) -> dict[str, Any]:
    prediction = predict_phase(features)
    drivers = explain_prediction(features)["top_drivers"]
    driver_text = ", ".join(
        humanize_feature(d["feature"]) for d in narrative_drivers(drivers)
    ) or "the provided signals"
    probs = prediction["probabilities"]
    fertility = probs.get("Fertility", 0)
    extra = ""
    if question and "ovul" in question.lower():
        extra = f" The model assigns a {fertility:.0%} probability to the fertile window today."
    answer = (
        f"Based on {driver_text}, the model predicts the {prediction['phase_label']} phase "
        f"(probabilities: {probs}).{extra} {DISCLAIMER}"
    )
    return {"answer": answer, "source_url": None}


def _direct_answer(question: str, features: dict) -> dict[str, Any]:
    """One Groq call with model outputs + Tavily snippet pre-fetched."""
    prediction = predict_phase(features)
    drivers = explain_prediction(features)["top_drivers"][:5]
    driver_summary = ", ".join(
        f"{humanize_feature(d['feature'])} ({'higher' if d.get('shap', 0) >= 0 else 'lower'})"
        for d in narrative_drivers(drivers)
    ) or "no strong drivers"

    grounding = search_medical(f"{question} menstrual cycle phase physiology fertility ovulation")
    snippet = grounding.get("snippet") or "No external medical reference available."
    source_url = grounding.get("source_url")

    fertility_pct = prediction["probabilities"].get("Fertility", 0)
    prompt = (
        "You explain a wearable-based menstrual cycle phase research prototype.\n\n"
        f"User question: {question}\n"
        f"Model predicted phase: {prediction['phase_label']}\n"
        f"All phase probabilities: {json.dumps(prediction['probabilities'])}\n"
        f"Fertility-window probability: {fertility_pct:.1%}\n"
        f"Top model drivers (SHAP): {driver_summary}\n"
        f"Medical reference snippet: {snippet}\n\n"
        "Write 2-3 plain-language sentences that directly answer the user's question. "
        "Use the model probabilities and drivers above; if they ask about ovulation, "
        "focus on the Fertility probability. You may use the medical snippet for physiology "
        "context only — do not invent facts beyond it. Do not diagnose or prescribe. "
        "Use plain English only — never mention raw column names, _pz, _roll3, VO₂ max codes, "
        "or statistical jargon. Do NOT include any disclaimer — the app adds one automatically."
    )

    response = groq_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=220,
        temperature=0.3,
    )
    answer = (response.choices[0].message.content or "").strip()
    if not _is_substantive(answer):
        raise RuntimeError(f"LLM returned non-substantive answer: {answer!r}")

    if DISCLAIMER not in answer:
        answer = f"{answer.rstrip()} {DISCLAIMER}"
    return {"answer": answer, "source_url": source_url}


def answer_question(question: str, features: dict) -> dict[str, Any]:
    lowered = question.lower()
    if any(keyword in lowered for keyword in DIAGNOSIS_KEYWORDS):
        return {
            "answer": (
                "I can't provide a diagnosis or treatment recommendation. This tool predicts "
                f"cycle phase from wearable data only — it is not a medical device. {DISCLAIMER}"
            ),
            "source_url": None,
        }

    try:
        if not groq_api_key():
            print("[GROQ] GROQ_API_KEY not set — using template fallback for /ask")
            return _templated_answer(features, question)

        return _direct_answer(question, features)
    except Exception as exc:
        print(f"[GROQ] /ask failed, using template fallback: {exc}")
        try:
            return _templated_answer(features, question)
        except Exception as exc2:
            print(f"[agent] templated fallback also failed: {exc2}")
            return {"answer": f"Unable to generate an answer right now. {DISCLAIMER}", "source_url": None}
