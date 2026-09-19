"""Security regressions that should remain true as providers and policies evolve."""

from __future__ import annotations

import json
from types import SimpleNamespace

from capability_system.discovery.agent import AgentDecision, DiscoveryAgent, _Trace
from capability_system.discovery.provider import (
    DEFAULT_OLLAMA_MODEL,
    OllamaProvider,
    select_provider,
)
from capability_system.perception.base import Observation


def test_default_provider_is_local_ollama(monkeypatch) -> None:
    for key in (
        "CAPABILITY_PROVIDER", "OLLAMA_MODEL", "OLLAMA_HOST",
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    provider = select_provider()
    assert isinstance(provider, OllamaProvider)
    assert provider.model == DEFAULT_OLLAMA_MODEL
    assert provider.is_remote is False


def test_ollama_cloud_tag_is_classified_remote() -> None:
    provider = OllamaProvider(model="gemma4:31b-cloud")
    assert provider.is_remote is True


def test_non_loopback_ollama_host_is_classified_remote() -> None:
    provider = OllamaProvider(model="qwen3:8b", host="https://ollama.example")
    assert provider.is_remote is True


def test_sensitive_discovery_value_is_not_compiled_into_artifact() -> None:
    agent = DiscoveryAgent(
        surface=None,
        provider=SimpleNamespace(name="test/provider"),
        policy=None,
    )
    trace = _Trace(
        index=1,
        decision=AgentDecision(
            reasoning="Enter private-member-id in the search field",
            action="type",
            handle="h1",
            text="private-member-id",
        ),
        handle="h1",
        payload="private-member-id",
        target=None,
        observed_url="https://bank.example/app",
    )
    artifact = agent._compile(
        goal="Look up private-member-id",
        capability_id="lookup_member",
        start_url="https://bank.example/app",
        inputs={"member_id": "private-member-id"},
        outputs={},
        traces=[trace],
        product="mock",
        final=Observation(url="https://bank.example/app"),
        sensitive={"member_id"},
    )
    serialised = json.dumps(artifact.model_dump(mode="json"))
    assert "private-member-id" not in serialised
    assert artifact.inputs[0].example is None
