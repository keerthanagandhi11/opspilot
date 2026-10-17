"""Minimal client for any OpenAI-compatible chat completions API.

Works with Groq (free tier, default), OpenRouter free models, Google Gemini's
OpenAI-compatible endpoint, or a local Ollama server. No SDK needed."""

from __future__ import annotations

import asyncio
import json
import logging
import re

import httpx

from .config import Settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class ToolCallError(LLMError):
    pass


class RateLimitError(LLMError):
    pass


RATE_LIMIT_MESSAGE = "The AI provider's free-tier limit was reached. Please wait a minute and try again."
MAX_RETRY_WAIT_SECONDS = 20.0


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """How long the provider asks us to wait, from the Retry-After header or the error
    body (Gemini sends "retryDelay": "17s"; Groq says "try again in 7.5s")."""
    header = response.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    hit = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"|try again in (\d+(?:\.\d+)?)s', response.text)
    if hit:
        return float(hit.group(1) or hit.group(2))
    return None


class LLMClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.s = settings
        self._transport = transport  # tests inject a mock transport

    @property
    def enabled(self) -> bool:
        return self.s.llm_enabled

    def _redact_key(self, text: str) -> str:
        key = self.s.llm_api_key
        return text.replace(key, "[redacted]") if key else text

    def _provider_error(self, response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return self._redact_key(response.text)[:300]
        if isinstance(payload, dict):
            error = payload.get("error", {})
        elif isinstance(payload, list):
            error = next(
                (
                    item.get("error")
                    for item in payload
                    if isinstance(item, dict) and isinstance(item.get("error"), dict)
                ),
                {},
            )
        else:
            return self._redact_key(str(payload))[:300]
        message = error.get("message") if isinstance(error, dict) else None
        if not isinstance(message, str) or not message.strip():
            return self._redact_key(response.text)[:300]
        return self._redact_key(message)[:300]

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        json_mode: bool = False,
        max_tokens: int | None = None,
        retried_tool_call: bool = False,
    ) -> dict:
        """One chat-completions call. Returns the assistant message dict, which may
        contain "tool_calls" when tools are offered (used by the agent)."""
        if not self.enabled:
            raise LLMError("No LLM API key configured.")
        payload: dict = {
            "model": self.s.llm_model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": max_tokens if max_tokens is not None else self.s.llm_max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        url = self.s.llm_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.s.llm_api_key}"}
        for attempt in (1, 2):
            try:
                async with httpx.AsyncClient(timeout=self.s.llm_timeout_seconds, transport=self._transport) as client:
                    r = await client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                raise LLMError(f"Could not reach the AI provider ({exc.__class__.__name__}).") from exc
            if r.status_code in (500, 502, 503, 504) and attempt == 1:
                # "Model overloaded" style errors are usually gone a moment later.
                log.warning("LLM temporarily unavailable (HTTP %s), retrying once", r.status_code)
                await asyncio.sleep(min(_retry_after_seconds(r) or 2.0, MAX_RETRY_WAIT_SECONDS))
                continue
            if r.status_code != 429:
                break
            # Free tiers limit requests per minute: wait once if the provider says it's short.
            wait = _retry_after_seconds(r)
            log.warning("LLM rate limited (HTTP 429): %s", self._provider_error(r))
            if attempt == 2 or wait is None or wait > MAX_RETRY_WAIT_SECONDS:
                raise RateLimitError(RATE_LIMIT_MESSAGE)
            await asyncio.sleep(max(wait, 1.0))
        # Groq reports malformed model output as tool_use_failed or failed_generation.
        if r.status_code >= 400 and ("tool_use_failed" in r.text or "failed_generation" in r.text):
            if not retried_tool_call:
                log.warning("Model produced an invalid tool call, retrying once")
                return await self.complete(messages, tools, tool_choice, json_mode, max_tokens, retried_tool_call=True)
            raise ToolCallError("The model produced an invalid tool call.")
        if r.status_code >= 400:
            # Full provider detail goes to the server log only; users get a short message.
            log.warning("LLM request returned HTTP %s: %s", r.status_code, self._provider_error(r))
            raise LLMError(f"The AI provider returned an error (HTTP {r.status_code}).")
        try:
            choice = r.json()["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError("Unexpected LLM response shape.") from exc
        if choice.get("finish_reason") == "length":
            raise LLMError("LLM response was cut off at the output token limit.")
        return message

    async def chat(self, system: str, user: str, json_mode: bool = False, max_tokens: int | None = None) -> str:
        msg = await self.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            json_mode=json_mode,
            max_tokens=max_tokens,
        )
        return msg.get("content") or ""

    async def chat_json(self, system: str, user: str, max_tokens: int | None = None) -> dict:
        raw = await self.chat(system, user, json_mode=True, max_tokens=max_tokens)
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end < 0:
            raise LLMError("LLM did not return JSON.")
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError("LLM returned invalid JSON.") from exc
