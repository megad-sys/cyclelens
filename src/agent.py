"""LLM agent (Groq tool-calling via OpenAI-compatible API) that answers free-form questions about a
single participant-day, using the trained model, SHAP explainer, and a
Tavily-grounded medical search.

Guardrails enforced IN CODE (not just prompted for):
  - hard cap of MAX_TOOL_CALLS total tool invocations per question (the loop
    forces tool_choice="none" once the budget is spent, so the model MUST
    respond with text rather than call more tools).
  - the returned source_url can only ever come from an actual search_medical()
    result -- it is extracted from the tool's return value, never from the
    LLM's free-text output, so it cannot be fabricated.
  - the disclaimer is always appended by code, regardless of what the model said.
  - diagnosis/treatment requests are intercepted by a keyword check BEFORE any
    LLM call is made, so a decline can never be talked around by the model.
  - predict_phase/explain_prediction always operate on the day's real features
    (closed over, not re-supplied by the model), so the model cannot corrupt
    the day's data by mis-transcribing a 50+ field JSON object as tool args.

What is NOT mechanically enforced (an open problem, not a gap hidden here):
whether every medical-sounding sentence in the model's free text specifically
traces back to a search_medical result -- that relies on the system prompt.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from src.explain import DEFAULT_MODEL_PATH, LABEL_NAMES, LABEL_VALUES, _get_explainer, explain_one

DISCLAIMER = "Research prototype, not medical advice."
MAX_TOOL_CALLS = 4
ALLOWED_GROUNDING_DOMAINS = ["acog.org", "ncbi.nlm.nih.gov", "nih.gov", "mayoclinic.org"]

# Simple, code-level (not prompt-level) refusal trigger for diagnosis/treatment asks.
DIAGNOSIS_KEYWORDS = [
    "diagnos", "treatment for", "treat me", "prescri", "medication", "what drug",
    "cure for", "should i take", "birth control", "what dose",
]

SYSTEM_PROMPT = (
    "You are a research assistant explaining a wearable-based menstrual cycle-phase "
    "prediction model, for ONE participant-day. Tools available: predict_phase (the "
    "model's phase probabilities for this day), explain_prediction (top-5 SHAP drivers "
    "for this day), and search_medical (search ACOG/NCBI/NIH/Mayo Clinic for physiology "
    "context). Rules:\n"
    "- Ground any physiological claim in a search_medical result; do not invent physiology.\n"
    "- You have a hard budget of at most 4 tool calls total. Use them deliberately.\n"
    "- Never provide a diagnosis, treatment, or medication recommendation -- this is a "
    "research prototype, not a medical device.\n"
    "- End your answer with exactly: 'Research prototype, not medical advice.'"
)

TOOLS_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "predict_phase",
            "description": "Predict the 4 cycle-phase probabilities for the participant-day "
                            "already provided in this conversation. Takes no arguments.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_prediction",
            "description": "Get the top-5 SHAP drivers behind the model's prediction for the "
                            "participant-day already provided in this conversation. Takes no arguments.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_medical",
            "description": "Search reputable medical sources (ACOG, NCBI, NIH, Mayo Clinic) "
                            "for physiology context to ground a claim.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "search query"}},
                "required": ["query"],
            },
        },
    },
]


def _build_row(features: dict, feature_columns: list[str]) -> pd.DataFrame:
    row = {col: features.get(col) for col in feature_columns}
    df = pd.DataFrame([row], columns=feature_columns)
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def predict_phase(features: dict) -> dict:
    """Tool: model's phase probabilities for `features` (reuses the /predict path)."""
    _, booster, feature_columns = _get_explainer(str(DEFAULT_MODEL_PATH))
    row = _build_row(features, feature_columns)
    proba = booster.predict(row)[0]
    probabilities = {LABEL_NAMES[label]: round(float(p), 4) for label, p in zip(LABEL_VALUES, proba)}
    predicted_label = LABEL_NAMES[LABEL_VALUES[int(np.argmax(proba))]]
    return {"phase_label": predicted_label, "probabilities": probabilities}


def explain_prediction(features: dict) -> dict:
    """Tool: top-5 SHAP drivers for `features` (via src.explain.explain_one)."""
    drivers = explain_one(features, model_path=str(DEFAULT_MODEL_PATH), top_k=5)
    return {"top_drivers": drivers}


def search_medical(query: str) -> dict:
    """Tool: Tavily search restricted to reputable medical domains. Degrades to a
    no-result dict (never raises) if TAVILY_API_KEY is unset or the search fails."""
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
    except Exception as exc:  # never hard-fail the agent over a search issue
        print(f"[agent] search_medical failed: {exc}")
        return {"snippet": None, "source_url": None, "source_title": None}


def _execute_tool(name: str, args: dict, features: dict) -> dict:
    if name == "predict_phase":
        return predict_phase(features)  # always the real day's features, never the model's args
    if name == "explain_prediction":
        return explain_prediction(features)
    if name == "search_medical":
        return search_medical(str(args.get("query", "")))
    return {"error": f"unknown tool '{name}'"}


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"


def _create_openai_client():
    from openai import OpenAI

    return OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_BASE_URL)


def _run_llm_loop(question: str, features: dict) -> tuple[str, str | None]:
    """Tool-calling loop, hard-capped at MAX_TOOL_CALLS total tool invocations.
    Returns (answer_text, source_url); source_url only ever comes from an
    actual search_medical() result."""
    client = _create_openai_client()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nToday's features: {json.dumps(features)}"},
    ]

    source_url: str | None = None
    tool_calls_made = 0

    while True:
        allow_tools = tool_calls_made < MAX_TOOL_CALLS
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto" if allow_tools else "none",
            temperature=0.2,
        )
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return (message.content or "").strip(), source_url

        for tool_call in message.tool_calls:
            tool_calls_made += 1
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _execute_tool(tool_call.function.name, args, features)
            if tool_call.function.name == "search_medical" and result.get("source_url"):
                source_url = result["source_url"]
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })


def _templated_answer(features: dict) -> dict[str, Any]:
    prediction = predict_phase(features)
    drivers = explain_prediction(features)["top_drivers"]
    driver_text = ", ".join(d["feature"] for d in drivers[:3]) if drivers else "the provided signals"
    answer = (
        f"Based on {driver_text}, the model predicts the {prediction['phase_label']} phase "
        f"(probabilities: {prediction['probabilities']}). {DISCLAIMER}"
    )
    return {"answer": answer, "source_url": None}


def answer_question(question: str, features: dict) -> dict[str, Any]:
    """Answers a free-form question about one participant-day. Never raises --
    every failure mode degrades to a safe templated or apologetic answer."""
    lowered = question.lower()
    if any(keyword in lowered for keyword in DIAGNOSIS_KEYWORDS):
        return {
            "answer": (
                "I can't provide a diagnosis or treatment recommendation. This tool predicts "
                f"cycle phase from wearable data only -- it is not a medical device. {DISCLAIMER}"
            ),
            "source_url": None,
        }

    try:
        if not os.environ.get("GROQ_API_KEY"):
            print("[llm] GROQ_API_KEY not set — using template fallback")
            return _templated_answer(features)

        answer, source_url = _run_llm_loop(question, features)
        if DISCLAIMER not in answer:
            answer = f"{answer.rstrip()} {DISCLAIMER}".strip()
        return {"answer": answer, "source_url": source_url}
    except Exception as exc:  # never hard-fail /ask
        print(f"[agent] answer_question failed, degrading gracefully: {exc}")
        try:
            return _templated_answer(features)
        except Exception as exc2:
            print(f"[agent] templated fallback also failed: {exc2}")
            return {"answer": f"Unable to generate an answer right now. {DISCLAIMER}", "source_url": None}
