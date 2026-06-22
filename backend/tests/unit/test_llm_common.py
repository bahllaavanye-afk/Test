"""Regression tests for the free-LLM cascade (.github/scripts/llm_common.py).

These lock in two production-down bugs the agent company hit:
  1. LLM requests MUST send a browser User-Agent, or Cloudflare (Groq/Cerebras/…)
     blocks them with "error code: 1010" and the whole cascade returns nothing.
  2. The cascade MUST rotate across a provider's numbered key variants
     (GROQ_API_KEY, _1, _2, _3) so a rate-limited key falls through to the next.

Network is fully mocked — no keys or connectivity required.
"""
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / ".github" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import llm_common as L  # noqa: E402


class _FakeResp:
    def __init__(self, payload: dict):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_OK = {"choices": [{"message": {"content": "OK"}}]}
_PROVIDER = {
    "name": "test", "url": "https://example.test/v1/chat", "fmt": "openai",
    "key_env": "TEST_LLM_KEY", "model": "m",
}


def test_request_sends_browser_user_agent(monkeypatch):
    """Regression for the Cloudflare 1010 outage: a real UA must be sent."""
    monkeypatch.setenv("TEST_LLM_KEY", "k1")
    seen = {}

    def fake_urlopen(req, timeout=30):
        seen["ua"] = req.get_header("User-agent")
        return _FakeResp(_OK)

    monkeypatch.setattr(L.urllib.request, "urlopen", fake_urlopen)
    assert L._call_provider(_PROVIDER, "sys", "hi", 16, 0.0) == "OK"
    assert seen["ua"] and "Mozilla" in seen["ua"]  # not the default Python-urllib UA


def test_provider_keys_collects_numbered_variants(monkeypatch):
    for n in ("TEST_LLM_KEY", "TEST_LLM_KEY_1", "TEST_LLM_KEY_2", "TEST_LLM_KEY_3"):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("TEST_LLM_KEY", "a")
    monkeypatch.setenv("TEST_LLM_KEY_2", "b")
    assert L._provider_keys(_PROVIDER) == ["a", "b"]  # primary + _2, ordered, deduped


def test_call_provider_rotates_to_next_key_on_failure(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "bad")
    monkeypatch.setenv("TEST_LLM_KEY_1", "good")
    calls = []

    def fake_urlopen(req, timeout=30):
        auth = req.get_header("Authorization")
        calls.append(auth)
        if auth == "Bearer bad":
            raise RuntimeError("simulated 429")
        return _FakeResp(_OK)

    monkeypatch.setattr(L.urllib.request, "urlopen", fake_urlopen)
    assert L._call_provider(_PROVIDER, "s", "p", 16, 0.0) == "OK"
    assert calls == ["Bearer bad", "Bearer good"]  # rotated past the bad key


def test_has_key_treats_disabled_as_absent(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "disabled")
    assert L._has_key(_PROVIDER) is False
    monkeypatch.setenv("TEST_LLM_KEY", "real")
    assert L._has_key(_PROVIDER) is True


def test_call_provider_raises_when_no_key(monkeypatch):
    for n in ("TEST_LLM_KEY", "TEST_LLM_KEY_1", "TEST_LLM_KEY_2", "TEST_LLM_KEY_3"):
        monkeypatch.delenv(n, raising=False)
    try:
        L._call_provider(_PROVIDER, "s", "p", 16, 0.0)
        assert False, "expected KeyError"
    except KeyError:
        pass


# --- Reasoning-model content extraction -------------------------------------- #
# Regression: Cerebras gpt-oss / DeepSeek-R1 return message.content=None and put
# the answer under reasoning_content/reasoning. The old ["content"].strip() died
# with a 'content' KeyError, silently dropping a whole free provider.
def test_extract_content_plain():
    assert L._extract_openai_content({"choices": [{"message": {"content": "hi"}}]}) == "hi"


def test_extract_content_falls_back_to_reasoning_content():
    payload = {"choices": [{"message": {"content": None, "reasoning_content": "answer"}}]}
    assert L._extract_openai_content(payload) == "answer"


def test_extract_content_falls_back_to_reasoning():
    payload = {"choices": [{"message": {"reasoning": "thought"}}]}
    assert L._extract_openai_content(payload) == "thought"


def test_extract_content_empty_raises():
    import pytest
    with pytest.raises(ValueError):
        L._extract_openai_content({"choices": [{"message": {"content": "  "}}]})


def test_call_provider_handles_reasoning_model(monkeypatch):
    """Full path: a reasoning model (content=None) must not be dropped."""
    monkeypatch.setenv("TEST_LLM_KEY", "k1")

    def fake_urlopen(req, timeout=30):
        return _FakeResp({"choices": [{"message": {"content": None, "reasoning_content": "RM"}}]})

    monkeypatch.setattr(L.urllib.request, "urlopen", fake_urlopen)
    assert L._call_provider(_PROVIDER, "sys", "hi", 16, 0.0) == "RM"
