"""Model provider abstraction, and the tolerant parsing that makes it survivable.

BRING YOUR OWN KEY. Nothing in this repository ships a credential. Ollama is the default
because it needs none at all -- it is an HTTP endpoint on the machine running the code --
and OpenAI and Anthropic are drop-in alternatives selected purely by which environment
variable happens to be set. A reviewer can clone this and run the discovery loop without
creating an account anywhere.

WHY THE PARSER IS TOLERANT, WHICH IS THE INTERESTING PART OF THIS FILE.
`POST /api/chat` accepts a JSON-schema `format` field, and the documentation reads as
though the schema is enforced. Measured against `gemma4:31b` it is not: asked for
`{reasoning, action, handle, text}` the model returned a fenced markdown block containing
`{"action": "type", "element": "e1", "value": "12345"}`. Correct decision, valid JSON,
wrong envelope and wrong key names.

That is the normal case, not an aberration, and a system that assumes schema enforcement
across providers will break on the first model swap. So parsing is defensive in three
layers, in order: strip any code fence, map a table of known key aliases onto the real
field names, then validate with Pydantic and on failure send one repair turn quoting the
validation error back to the model. One repair, not a loop -- an unbounded self-correction
cycle against a paid endpoint is how a discovery run quietly costs real money.

The alternative would have been to hard-require a provider with strict structured outputs.
That buys cleaner code and costs the property that matters more here: the discovery loop
has to run on whatever a reviewer already has.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol

import urllib.parse
import urllib.request

from pydantic import BaseModel, ValidationError

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_FIRST_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

# Field names models reach for instead of ours. Measured, not imagined.
_ALIASES: dict[str, str] = {
    "element": "handle", "element_id": "handle", "ref": "handle",
    "target": "handle", "id": "handle", "selector": "handle",
    "value": "text", "input": "text", "content": "text", "url": "text",
    "thought": "reasoning", "rationale": "reasoning", "why": "reasoning",
    "explanation": "reasoning", "act": "action", "next_action": "action",
    "option_label": "option", "choice": "option",
}

# Action VALUES models reach for instead of ours. Same problem as the key aliases one level
# down: the decision is right and the vocabulary is wrong. Measured with gemma4:31b, which
# emitted action="extract_as" while correctly also setting extract_as="savings_balance" --
# it confused the field name for the verb. Normalising the value is honest; widening the
# Literal to accept synonyms would pollute the contract the artifact is compiled against.
_ACTION_ALIASES: dict[str, str] = {
    "extract": "read", "extract_as": "read", "read_value": "read", "get": "read",
    "goto": "navigate", "open": "navigate", "visit": "navigate", "go_to": "navigate",
    "fill": "type", "input": "type", "enter": "type", "type_text": "type",
    "press": "click", "tap": "click", "submit": "click",
    "choose": "select", "select_option": "select",
    "finish": "done", "complete": "done", "success": "done", "end": "done",
    "blocked": "stuck", "give_up": "stuck", "cannot_proceed": "stuck",
}


class ModelError(RuntimeError):
    pass


class ModelProvider(Protocol):
    name: str
    is_remote: bool

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> str: ...


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class OllamaProvider:
    """Default. No credential of any kind.

    `model` is configurable because what a given machine has pulled is unknowable from
    here. A hosted `:cloud` tag still routes through this same local endpoint, so nothing
    about the calling code changes, but the run is then NOT local inference and the
    evidence manifest records the model name so that claim can never be made by accident.
    """

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str | None = None,
        timeout: float = 180.0,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.name = f"ollama/{model}"
        host_name = urllib.parse.urlparse(self.host).hostname
        model_tag = model.lower()
        self.is_remote = model_tag.endswith((":cloud", "-cloud")) or host_name not in {
            "localhost", "127.0.0.1", "::1"
        }

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Sent even though enforcement is inconsistent: where a provider does honour
            # it the parser below has nothing to do, which is the good outcome.
            "format": schema,
            "options": {"temperature": self.temperature},
        }
        data = _post_json(f"{self.host}/api/chat", payload, self.timeout)
        if "error" in data:
            raise ModelError(f"ollama: {data['error']}")
        return (data.get("message") or {}).get("content", "")


class OpenAIProvider:
    """Used only when OPENAI_API_KEY is present in the environment."""

    def __init__(self, model: str = "gpt-4o-mini", timeout: float = 120.0) -> None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ModelError("OPENAI_API_KEY is not set")
        self._key = key
        self.model = model
        self.timeout = timeout
        self.name = f"openai/{model}"
        self.is_remote = True

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


class AnthropicProvider:
    """Used only when ANTHROPIC_API_KEY is present in the environment."""

    def __init__(self, model: str = "claude-sonnet-4-5", timeout: float = 120.0) -> None:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ModelError("ANTHROPIC_API_KEY is not set")
        self._key = key
        self.model = model
        self.timeout = timeout
        self.name = f"anthropic/{model}"
        self.is_remote = True

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self._key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return "".join(block.get("text", "") for block in data.get("content", []))


def select_provider(preferred: str | None = None, model: str | None = None) -> ModelProvider:
    """Ollama unless the caller has explicitly provisioned something else.

    Order is deliberate. A reviewer with no accounts gets a working system; a reviewer who
    has already exported a key gets their own provider without editing code.
    """
    choice = (preferred or os.environ.get("CAPABILITY_PROVIDER") or "").lower()
    if choice == "openai" or (not choice and os.environ.get("OPENAI_API_KEY")):
        return OpenAIProvider(model or "gpt-4o-mini")
    if choice == "anthropic" or (not choice and os.environ.get("ANTHROPIC_API_KEY")):
        return AnthropicProvider(model or "claude-sonnet-4-5")
    return OllamaProvider(model or os.environ.get("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL)


# --------------------------------------------------------------------------------------
# Tolerant decoding
# --------------------------------------------------------------------------------------

def extract_json(raw: str) -> dict[str, Any]:
    """Pull an object out of whatever the model actually returned."""
    if not raw or not raw.strip():
        raise ModelError("model returned empty content")
    text = raw.strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = _FIRST_OBJECT.search(text)
        if not match:
            raise ModelError(f"no JSON object in model output: {raw[:200]!r}")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ModelError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def apply_aliases(data: dict[str, Any]) -> dict[str, Any]:
    """Rename known drift, in keys and in action values. Never overwrites a correct field."""
    out = dict(data)
    for wrong, right in _ALIASES.items():
        if wrong in out and right not in out:
            out[right] = out.pop(wrong)
    action = out.get("action")
    if isinstance(action, str):
        out["action"] = _ACTION_ALIASES.get(action.strip().lower(), action.strip().lower())
    return out


def decode(
    provider: ModelProvider,
    model_cls: type[BaseModel],
    system: str,
    user: str,
) -> tuple[BaseModel, str]:
    """Ask, parse, validate, and repair exactly once. Returns (instance, raw_text)."""
    schema = model_cls.model_json_schema()
    raw = provider.complete(system, user, schema)
    try:
        return model_cls.model_validate(apply_aliases(extract_json(raw))), raw
    except (ModelError, ValidationError) as first_error:
        repair = (
            f"{user}\n\n"
            f"Your previous reply could not be used. Error:\n{first_error}\n\n"
            f"Reply with ONE JSON object and nothing else. No prose, no code fence. "
            f"Use exactly these keys: {sorted(schema.get('properties', {}))}."
        )
        raw2 = provider.complete(system, repair, schema)
        try:
            return model_cls.model_validate(apply_aliases(extract_json(raw2))), raw2
        except (ModelError, ValidationError) as second_error:
            # Deliberately terminal. A third attempt is a loop, and a loop against a
            # metered endpoint is an unbounded bill for a model that is not complying.
            raise ModelError(
                f"model did not produce a valid decision after one repair. "
                f"first={first_error}; second={second_error}; raw={raw2[:300]!r}"
            ) from second_error
