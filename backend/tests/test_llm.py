import asyncio

import httpx
import pytest

from app.config import Settings
from app.llm import RATE_LIMIT_MESSAGE, LLMClient, LLMError, RateLimitError

SETTINGS = Settings(llm_api_key="sk-secret-key", llm_base_url="https://llm.test/v1")
OK = {"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]}


def client(responses):
    calls = []

    def handler(request):
        calls.append(request)
        return responses.pop(0)

    return LLMClient(SETTINGS, transport=httpx.MockTransport(handler)), calls


def test_retries_once_after_short_rate_limit(monkeypatch):
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr("app.llm.asyncio.sleep", fake_sleep)
    llm, calls = client([httpx.Response(429, headers={"retry-after": "3"}), httpx.Response(200, json=OK)])
    assert asyncio.run(llm.chat("s", "u")) == "hi"
    assert len(calls) == 2 and slept == [3.0]


def test_reads_gemini_retry_delay_from_body(monkeypatch):
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr("app.llm.asyncio.sleep", fake_sleep)
    body = '[{"error": {"code": 429, "details": [{"retryDelay": "7s"}]}}]'
    llm, _ = client([httpx.Response(429, text=body), httpx.Response(200, json=OK)])
    assert asyncio.run(llm.chat("s", "u")) == "hi" and slept == [7.0]


def test_long_rate_limit_gives_friendly_error_without_waiting():
    body = '{"error": {"message": "You exceeded your current quota, please check your plan and billing details."}}'
    llm, calls = client([httpx.Response(429, text=body)])
    with pytest.raises(RateLimitError) as err:
        asyncio.run(llm.chat("s", "u"))
    assert str(err.value) == RATE_LIMIT_MESSAGE and len(calls) == 1
    assert "billing" not in str(err.value)


def test_provider_errors_are_short_and_never_leak_the_key():
    llm, _ = client([httpx.Response(401, json={"error": {"message": "bad key sk-secret-key"}})])
    with pytest.raises(LLMError) as err:
        asyncio.run(llm.chat("s", "u"))
    assert "API key was rejected" in str(err.value)
    assert "sk-secret-key" not in llm._provider_error(httpx.Response(401, text="bad key sk-secret-key"))


def test_retries_once_when_model_is_overloaded(monkeypatch):
    async def fake_sleep(s):
        pass

    monkeypatch.setattr("app.llm.asyncio.sleep", fake_sleep)
    llm, calls = client([httpx.Response(503, json={"error": {"message": "high demand"}}), httpx.Response(200, json=OK)])
    assert asyncio.run(llm.chat("s", "u")) == "hi" and len(calls) == 2


def test_overloaded_twice_gives_short_error(monkeypatch):
    async def fake_sleep(s):
        pass

    monkeypatch.setattr("app.llm.asyncio.sleep", fake_sleep)
    llm, calls = client([httpx.Response(503), httpx.Response(503)])
    with pytest.raises(LLMError, match=r"HTTP 503"):
        asyncio.run(llm.chat("s", "u"))
    assert len(calls) == 2


def test_invalid_tool_call_is_retried_once():
    bad = httpx.Response(400, json={"error": {"code": "tool_use_failed", "message": "bad call"}})
    llm, calls = client([bad, httpx.Response(200, json=OK)])
    assert asyncio.run(llm.chat("s", "u")) == "hi" and len(calls) == 2


def test_invalid_tool_call_twice_raises_tool_call_error():
    from app.llm import ToolCallError

    bad = {"error": {"code": "tool_use_failed"}}
    llm, calls = client([httpx.Response(400, json=bad), httpx.Response(400, json=bad)])
    with pytest.raises(ToolCallError):
        asyncio.run(llm.chat("s", "u"))
    assert len(calls) == 2


def test_groq_parsing_failed_is_retried_once():
    bad = httpx.Response(400, json={"error": {"message": "Parsing failed.", "failed_generation": "..."}})
    llm, calls = client([bad, httpx.Response(200, json=OK)])
    assert asyncio.run(llm.chat("s", "u")) == "hi" and len(calls) == 2


@pytest.mark.parametrize(
    "pasted",
    [
        "gsk_abc123",
        "  gsk_abc123\n",
        '"gsk_abc123"',
        "'gsk_abc123' ",
        "LLM_API_KEY=gsk_abc123",
        "llm_api_key = gsk_abc123",
    ],
)
def test_pasted_api_key_is_cleaned(pasted):
    assert Settings(llm_api_key=pasted).llm_api_key == "gsk_abc123"


def test_rejected_key_gives_clear_message():
    llm, _ = client([httpx.Response(401, json={"error": {"message": "Invalid API Key"}})])
    with pytest.raises(LLMError, match="API key was rejected"):
        asyncio.run(llm.chat("s", "u"))
