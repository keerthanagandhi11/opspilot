"""Minimal client for any OpenAI-compatible chat completions API.

Works with Groq (free tier, default), OpenRouter free models, Google Gemini's
OpenAI-compatible endpoint, or a local Ollama server. No SDK needed."""

from __future__ import annotations

import json
import logging

import httpx

from .config import Settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class ToolCallError(LLMError):
    pass


class LLMClient:
    def __init__(self, settings: Settings):
        self.s = settings

    @property
    def enabled(self) -> bool:
        return self.s.llm_enabled

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        json_mode: bool = False,
    ) -> dict:
        """One chat-completions call. Returns the assistant message dict, which may
        contain "tool_calls" when tools are offered (used by the agent)."""
        if not self.enabled:
            raise LLMError("No LLM API key configured.")
        payload: dict = {
            "model": self.s.llm_model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": self.s.llm_max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        url = self.s.llm_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.s.llm_api_key}"}
        try:
            async with httpx.AsyncClient(timeout=self.s.llm_timeout_seconds) as client:
                r = await client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc.__class__.__name__}") from exc
        if r.status_code == 429:
            raise LLMError("LLM free-tier rate limit reached.")
        if r.status_code >= 400:
            log.warning("LLM error %s: %s", r.status_code, r.text[:300])
            if "tool_use_failed" in r.text:  # model produced a malformed tool call
                raise ToolCallError("The model produced an invalid tool call.")
            raise LLMError(f"LLM returned HTTP {r.status_code}.")
        try:
            return r.json()["choices"][0]["message"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError("Unexpected LLM response shape.") from exc

    async def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        msg = await self.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}], json_mode=json_mode
        )
        return msg.get("content") or ""

    async def chat_json(self, system: str, user: str) -> dict:
        raw = await self.chat(system, user, json_mode=True)
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end < 0:
            raise LLMError("LLM did not return JSON.")
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError("LLM returned invalid JSON.") from exc
