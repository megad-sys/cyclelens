"""Shared Groq (OpenAI-compatible) client with model fallbacks and loud logging."""

from __future__ import annotations

import os

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Try in order; first success wins. All support chat; tool-calling needs 70B-class models.
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

_groq_status: str | None = None


def groq_api_key() -> str | None:
    key = os.environ.get("GROQ_API_KEY", "").strip().strip('"').strip("'")
    return key or None


def groq_status() -> str:
    """Set by test_groq_connection() on startup; 'unknown' until then."""
    return _groq_status or "unknown"


def test_groq_connection() -> str:
    """One-shot ping at startup. Logs clearly to Render logs. Never raises."""
    global _groq_status
    api_key = groq_api_key()
    if not api_key:
        _groq_status = "missing_key"
        print("[GROQ] GROQ_API_KEY not set — insights/chat will use template fallback")
        return _groq_status

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        last_error = "unknown"
        for model in GROQ_MODELS:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Reply with exactly: ok"}],
                    max_tokens=5,
                    temperature=0,
                )
                text = (response.choices[0].message.content or "").strip()
                _groq_status = f"ok ({model})"
                print(f"[GROQ] startup test passed with model={model} reply={text!r}")
                return _groq_status
            except Exception as exc:
                last_error = str(exc)
                print(f"[GROQ] startup test failed model={model}: {exc}")

        _groq_status = f"error: {last_error[:200]}"
        print(f"[GROQ] all models failed — using template fallback. Last error: {last_error}")
        return _groq_status
    except Exception as exc:
        _groq_status = f"error: {str(exc)[:200]}"
        print(f"[GROQ] client init failed — using template fallback: {exc}")
        return _groq_status


def groq_chat_completion(*, messages: list[dict], max_tokens: int = 120, temperature: float = 0.3, **kwargs):
    """Call Groq chat.completions with model fallbacks. Raises on total failure."""
    api_key = groq_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    last_error: Exception | None = None
    for model in GROQ_MODELS:
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        except Exception as exc:
            last_error = exc
            print(f"[GROQ] chat failed model={model}: {exc}")
    raise last_error or RuntimeError("GROQ chat failed for all models")
