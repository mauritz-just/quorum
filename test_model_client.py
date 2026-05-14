"""
Tests for the ModelClient OOP hierarchy and QueryResult dataclass.

These tests use mocked HTTP — no real API calls. The point is to verify
that the factory dispatches to the right subclass, each subclass produces
the right payload shape, and the public call() wrapper handles both happy
and error paths uniformly.
"""
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_openai_response():
    """Mock a successful OpenAI-compatible /chat/completions response."""
    resp = MagicMock()
    resp.raise_for_status = lambda: None
    resp.json = lambda: {"choices": [{"message": {"content": "mocked openai answer"}}]}
    return resp


@pytest.fixture
def mock_gemini_response():
    """Mock a successful Gemini :generateContent response."""
    resp = MagicMock()
    resp.raise_for_status = lambda: None
    resp.json = lambda: {
        "candidates": [{
            "finishReason": "STOP",
            "content": {"parts": [{"text": "mocked gemini answer"}]},
        }],
    }
    return resp


@pytest.fixture
def mock_anthropic_response():
    """Mock a successful Anthropic /v1/messages response."""
    resp = MagicMock()
    resp.raise_for_status = lambda: None
    resp.json = lambda: {"content": [{"text": "mocked claude answer"}]}
    return resp


class TestFactory:
    """ModelClient.from_config() must dispatch to the right subclass."""

    def test_gemini_type_returns_gemini_client(self, app_module):
        client = app_module["ModelClient"].from_config(
            "test", {"type": "gemini", "api_key": "k", "endpoint": "https://x"}
        )
        assert type(client).__name__ == "GeminiClient"

    def test_anthropic_type_returns_anthropic_client(self, app_module):
        client = app_module["ModelClient"].from_config(
            "test", {"type": "anthropic", "api_key": "k", "endpoint": "https://x", "model_id": "m"}
        )
        assert type(client).__name__ == "AnthropicClient"

    def test_openai_compat_type_returns_openai_compat_client(self, app_module):
        client = app_module["ModelClient"].from_config(
            "test", {"type": "openai_compat", "api_key": "k", "endpoint": "https://x", "model_id": "m"}
        )
        assert type(client).__name__ == "OpenAICompatClient"

    def test_unknown_type_defaults_to_openai_compat(self, app_module):
        """Unrecognised type strings should fall back to the most common shape."""
        client = app_module["ModelClient"].from_config(
            "test", {"type": "made_up_provider", "api_key": "k", "endpoint": "https://x", "model_id": "m"}
        )
        assert type(client).__name__ == "OpenAICompatClient"

    def test_missing_type_defaults_to_openai_compat(self, app_module):
        client = app_module["ModelClient"].from_config(
            "test", {"api_key": "k", "endpoint": "https://x", "model_id": "m"}
        )
        assert type(client).__name__ == "OpenAICompatClient"


class TestQueryResult:
    """QueryResult is the typed contract for one model's reply."""

    def test_has_expected_fields(self, app_module):
        qr = app_module["QueryResult"](name="X", response="hi", time=1.0, words=1, error=None)
        d = qr.to_dict()
        assert set(d.keys()) == {"name", "response", "time", "words", "error"}

    def test_to_dict_preserves_values(self, app_module):
        qr = app_module["QueryResult"](name="X", response="hello world", time=1.23, words=2, error=None)
        assert qr.to_dict() == {"name": "X", "response": "hello world", "time": 1.23, "words": 2, "error": None}


class TestCallModel:
    """call_model() is the top-level entry point used by orchestrate()."""

    def test_openai_compat_happy_path(self, app_module, mock_openai_response):
        app_module["MODELS"] = {
            "TestM": {"type": "openai_compat", "api_key": "k", "endpoint": "https://x", "model_id": "m"}
        }
        with patch.object(app_module["requests"], "post", return_value=mock_openai_response):
            result = app_module["call_model"]("TestM", "hi")
        assert isinstance(result, dict)
        assert result["name"] == "TestM"
        assert result["response"] == "mocked openai answer"
        assert result["error"] is None
        assert result["words"] == 3

    def test_gemini_happy_path(self, app_module, mock_gemini_response):
        app_module["MODELS"] = {
            "GemM": {"type": "gemini", "api_key": "k", "endpoint": "https://x"}
        }
        with patch.object(app_module["requests"], "post", return_value=mock_gemini_response):
            result = app_module["call_model"]("GemM", "hi")
        assert result["response"] == "mocked gemini answer"
        assert result["error"] is None

    def test_anthropic_happy_path(self, app_module, mock_anthropic_response):
        app_module["MODELS"] = {
            "AntM": {"type": "anthropic", "api_key": "k", "endpoint": "https://x", "model_id": "m"}
        }
        with patch.object(app_module["requests"], "post", return_value=mock_anthropic_response):
            result = app_module["call_model"]("AntM", "hi")
        assert result["response"] == "mocked claude answer"
        assert result["error"] is None

    def test_network_error_captured_in_dict(self, app_module):
        """An exception in generate() must be captured as result['error'], not raised."""
        app_module["MODELS"] = {
            "TestM": {"type": "openai_compat", "api_key": "k", "endpoint": "https://x", "model_id": "m"}
        }
        with patch.object(app_module["requests"], "post",
                          side_effect=RuntimeError("simulated network error")):
            result = app_module["call_model"]("TestM", "hi")
        assert result["response"] is None
        assert "simulated network error" in result["error"]
        # The schema must remain stable even on error
        assert set(result.keys()) == {"name", "response", "time", "words", "error"}


class TestOrchestration:
    """orchestrate() runs models in parallel and returns results in the requested order."""

    def test_results_returned_in_input_order(self, app_module, mock_openai_response):
        app_module["MODELS"] = {
            "A": {"type": "openai_compat", "api_key": "k", "endpoint": "https://x", "model_id": "a"},
            "B": {"type": "openai_compat", "api_key": "k", "endpoint": "https://x", "model_id": "b"},
            "C": {"type": "openai_compat", "api_key": "k", "endpoint": "https://x", "model_id": "c"},
        }
        with patch.object(app_module["requests"], "post", return_value=mock_openai_response):
            results = app_module["orchestrate"]("test", ["A", "B", "C"])
        # Results must come back in the order the caller specified, even though
        # threads complete in arbitrary order.
        assert [r["name"] for r in results] == ["A", "B", "C"]
