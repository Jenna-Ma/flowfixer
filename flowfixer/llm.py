"""Unified LLM client for FLOWFIXER.

Every reasoning step in the paper (spec inference, diagnosis, repair, assessment,
judging, simulated execution) is an LLM call routed through a single
OpenAI-compatible endpoint (``config.BASE_URL``).  This module provides:

  * ``chat``           — single completion, with disk cache + retry.
  * ``chat_json``      — completion parsed into a JSON object (robust extraction).
  * a deterministic ``MOCK`` path (config.MOCK_LLM) for offline wiring/CI.

The backbone model defaults to GPT-5.2; judges use their own model ids.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import config as C

try:  # openai>=1.0 style client
    from openai import OpenAI
    _client: Optional["OpenAI"] = None
except Exception:  # pragma: no cover - openai not installed
    OpenAI = None       # type: ignore
    _client = None


# --------------------------------------------------------------------------- #
# Client singleton                                                             #
# --------------------------------------------------------------------------- #
def _get_client():
    global _client
    if _client is None:
        if OpenAI is None:
            raise RuntimeError(
                "The 'openai' package is required (pip install -r requirements.txt), "
                "or set FLOWFIXER_MOCK=1 for offline runs."
            )
        _client = OpenAI(api_key=C.API_KEY, base_url=C.BASE_URL, timeout=C.REQUEST_TIMEOUT)
    return _client


# --------------------------------------------------------------------------- #
# Disk cache                                                                   #
# --------------------------------------------------------------------------- #
def _cache_key(model: str, messages: List[Dict[str, str]], temperature: float) -> str:
    blob = json.dumps({"m": model, "msg": messages, "t": temperature}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> str:
    return os.path.join(C.CACHE_DIR, key + ".json")


def _cache_get(key: str) -> Optional[str]:
    if not C.ENABLE_CACHE:
        return None
    p = _cache_path(key)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)["response"]
        except Exception:
            return None
    return None


def _cache_put(key: str, response: str) -> None:
    if not C.ENABLE_CACHE:
        return
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    try:
        with open(_cache_path(key), "w", encoding="utf-8") as f:
            json.dump({"response": response}, f)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Token accounting (best-effort; approximate when API omits usage)            #
# --------------------------------------------------------------------------- #
_TOKENS_USED = 0


def tokens_used() -> int:
    return _TOKENS_USED


def reset_tokens() -> None:
    global _TOKENS_USED
    _TOKENS_USED = 0


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------- #
# Core call                                                                    #
# --------------------------------------------------------------------------- #
def chat(
    prompt: str,
    *,
    system: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Return the assistant text for a single-turn prompt."""
    global _TOKENS_USED
    model = model or C.BACKBONE_MODEL
    temperature = C.TEMPERATURE if temperature is None else temperature
    max_tokens = max_tokens or C.MAX_TOKENS

    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if C.MOCK_LLM:
        return _mock_response(prompt, system, model)

    key = _cache_key(model, messages, temperature)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    last_err: Optional[Exception] = None
    for attempt in range(C.MAX_RETRIES):
        try:
            client = _get_client()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content or ""
            try:
                _TOKENS_USED += int(resp.usage.total_tokens)  # type: ignore[attr-defined]
            except Exception:
                _TOKENS_USED += _approx_tokens(prompt) + _approx_tokens(text)
            _cache_put(key, text)
            return text
        except Exception as e:  # network / rate-limit / transient
            last_err = e
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"LLM call failed after {C.MAX_RETRIES} retries: {last_err}")


# --------------------------------------------------------------------------- #
# JSON helpers                                                                 #
# --------------------------------------------------------------------------- #
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Best-effort extraction of the first JSON value from an LLM reply."""
    text = text.strip()
    # 1) fenced block
    m = _JSON_FENCE.search(text)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            text = candidate
    # 2) raw parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # 3) first {...} or [...] span
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                continue
    raise ValueError(f"Could not parse JSON from LLM output:\n{text[:500]}")


def chat_json(
    prompt: str,
    *,
    system: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    default: Any = None,
) -> Any:
    """Call ``chat`` and parse the reply as JSON, falling back to ``default``."""
    text = chat(prompt, system=system, model=model,
                temperature=temperature, max_tokens=max_tokens)
    try:
        return extract_json(text)
    except Exception:
        if default is not None:
            return default
        raise


# --------------------------------------------------------------------------- #
# Deterministic offline mock                                                   #
# --------------------------------------------------------------------------- #
def _mock_response(prompt: str, system: Optional[str], model: str) -> str:
    """Structurally-valid stub responses so the pipeline runs without a network.

    Keyed on marker strings present in each prompt template (see prompts.py).
    """
    p = prompt
    if "INFER_SPECS" in p:
        return json.dumps({"assertions": [
            {"dimension": "existence", "code": "assert node.output != None",
             "rationale": "output must exist"},
        ]})
    if "ROOT_CAUSE" in p:
        return json.dumps({"root_cause": "Poor Prompt Design",
                           "failure_description": "mock: prompt lacks constraints"})
    if "GEN_PATCH" in p:
        # Target the actual failure-responsible node (parsed from the prompt) so
        # the stub patch is structurally valid — a config edit, not a new node.
        m = re.search(r"failure-responsible node:\s*([^\s(]+)", p)
        nid = m.group(1) if m else "node"
        return json.dumps({"edits": [
            {"op": "append", "target": f"{nid}.config.prompt",
             "content": " Ensure all explicit task constraints are satisfied.",
             "position": None}]})
    if "SEMANTIC_ASSESS" in p:
        return json.dumps({"pass": True, "reason": "mock ok"})
    if "CONSISTENCY_ASSESS" in p:
        return json.dumps({"pass": True, "reason": "mock ok"})
    if "JUDGE_TASK" in p:
        return json.dumps({"correct": True, "rationale": "mock judge pass"})
    if "SIMULATE_EXEC" in p:
        return json.dumps({"executed": True, "final_output": "mock output"})
    if "GEN_UNSEEN" in p:
        return json.dumps({"inputs": [f"mock input {i}" for i in range(5)]})
    if "REFLECT" in p:
        return json.dumps({"reflection": "mock reflection"})
    return json.dumps({"pass": True, "reason": "mock default"})
