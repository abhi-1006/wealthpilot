"""Provider-agnostic LLM client used to synthesise corpus content.

Deliberately dependency-light: plain `requests` rather than an SDK, so the
toolkit installs in seconds and does not fight with whatever client library
your capstone itself pins. Swap in LiteLLM if you prefer -- the call surface
here (`complete`, `complete_json`) is one thin adapter away.
"""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any

import requests

from .config import Provider, settings


class LLMError(RuntimeError):
    """Raised when a provider call fails after all retries."""


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> Any:
    """Pull a JSON value out of a model response.

    Models wrap JSON in prose or fences even when told not to. Try the cheap
    parse first, then the fenced block, then the widest brace/bracket span.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = _JSON_FENCE.search(text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise LLMError(f"No parseable JSON in response: {text[:400]!r}")


def _call_gemini(prompt: str, system: str | None, temperature: float) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent"
    )
    payload: dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 8192},
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    resp = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": settings.gemini_api_key,
        },
        json=payload,
        timeout=settings.request_timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return "".join(
        part.get("text", "")
        for part in data["candidates"][0]["content"].get("parts", [])
    )


def _call_openrouter(prompt: str, system: str | None, temperature: float) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        json={
            "model": settings.openrouter_model,
            "messages": messages,
            "temperature": temperature,
        },
        timeout=settings.request_timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_groq(prompt: str, system: str | None, temperature: float) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        json={
            "model": settings.groq_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 8000,
        },
        timeout=settings.request_timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_ollama(prompt: str, system: str | None, temperature: float) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    resp = requests.post(
        f"{settings.ollama_host}/api/chat",
        json={
            "model": settings.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=settings.request_timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


_DISPATCH = {
    "gemini": _call_gemini,
    "openrouter": _call_openrouter,
    "ollama": _call_ollama,
    "groq": _call_groq,
}


def complete(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float | None = None,
    provider: Provider | None = None,
) -> str:
    """Single completion with exponential backoff and jitter.

    Free tiers rate-limit aggressively; the backoff here is what lets an
    unattended 50-document generation run finish without babysitting.
    """
    fn = _DISPATCH[provider or settings.provider]
    temp = settings.temperature if temperature is None else temperature
    last: Exception | None = None

    for attempt in range(settings.max_retries):
        try:
            out = fn(prompt, system, temp)
            if out and out.strip():
                return out
            last = LLMError("Empty completion")
        except requests.HTTPError as exc:
            last = exc
            status = exc.response.status_code if exc.response is not None else 0
            if status in (400, 401, 403):  # not worth retrying
                raise LLMError(f"Provider rejected request ({status}): {exc}") from exc
        except requests.RequestException as exc:
            last = exc

        sleep_for = (2**attempt) + random.uniform(0, 1.5)
        time.sleep(sleep_for)

    raise LLMError(f"Failed after {settings.max_retries} attempts: {last}")


def complete_json(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float | None = None,
) -> Any:
    """Completion that must parse as JSON, with one corrective re-ask."""
    instruction = (
        "Respond with raw JSON only. No prose, no markdown fences, no preamble."
    )
    sys_prompt = f"{system}\n\n{instruction}" if system else instruction

    raw = complete(prompt, system=sys_prompt, temperature=temperature)
    try:
        return _extract_json(raw)
    except LLMError:
        repair = (
            "The following was supposed to be valid JSON but is not. "
            "Return the corrected JSON and nothing else.\n\n" + raw[:6000]
        )
        return _extract_json(complete(repair, system=instruction, temperature=0.0))
