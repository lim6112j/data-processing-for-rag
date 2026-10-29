"""The shared OpenRouter client resolver: one connection path, two LLM roles.

Contract:

  * an injected client always wins -- tests pass fakes, operators pass
    pre-configured clients;
  * the key resolves from OPENROUTER_API_KEY, which the OpenAI SDK does
    not read on its own, and an explicit key beats the environment;
  * a missing 'openai' package fails with the install command, not a
    bare ImportError from inside the SDK;
  * when both package and key are missing, both problems are reported --
    the operator fixes one trip, not one trip per discovery.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from rag_ingest.adapters.openrouter_client import OPENROUTER_BASE_URL, resolve_client


def _install_fake_openai(monkeypatch) -> None:
    # Stands in for the real 'openai' package; the lambda records kwargs
    # as attributes on the returned namespace so tests can assert on them.
    module = SimpleNamespace(OpenAI=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setitem(sys.modules, "openai", module)


def _block_openai(monkeypatch) -> None:
    """Make 'import openai' fail, simulating the package being absent."""

    def _fail(name, path=None, target=None):
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return None

    class _Blocker:
        def find_spec(self, fullname, path=None, target=None):
            _fail(fullname)
            return None

    monkeypatch.setattr(sys, "meta_path", [_Blocker(), *sys.meta_path])


class TestResolveClient:
    def test_returns_the_injected_client_unchanged(self):
        client = SimpleNamespace()
        assert resolve_client(client, api_key=None) is client

    def test_builds_a_client_with_the_openrouter_key_from_the_environment(
        self, monkeypatch
    ):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
        _install_fake_openai(monkeypatch)

        client = resolve_client(
            None, api_key=None, base_url="https://example/v1", timeout=7.0
        )

        assert client.api_key == "sk-or-env"
        assert client.base_url == "https://example/v1"
        assert client.timeout == 7.0

    def test_explicit_api_key_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
        _install_fake_openai(monkeypatch)

        client = resolve_client(None, api_key="sk-or-explicit")

        assert client.api_key == "sk-or-explicit"

    def test_default_base_url_is_openrouter(self):
        assert OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"

    def test_missing_openai_package_fails_with_the_install_command(
        self, monkeypatch
    ):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
        _block_openai(monkeypatch)

        with pytest.raises(RuntimeError, match="uv add openai"):
            resolve_client(None, api_key=None, feature="adjudication")

    def test_missing_package_and_key_are_reported_together(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        _block_openai(monkeypatch)

        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            resolve_client(None, api_key=None, feature="adjudication")
