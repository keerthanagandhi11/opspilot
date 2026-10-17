"""Tiny sliding-window rate limiter per client IP, so one visitor of the public
demo cannot burn the whole free LLM quota. Works with FastAPI and Gradio requests."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RateLimited(Exception):
    message = "Too many requests. Please wait a minute and try again."


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    @staticmethod
    def client_ip(request) -> str:
        """`request` is a fastapi.Request or gradio.Request (same attributes)."""
        headers = getattr(request, "headers", None) or {}
        fwd = headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
        client = getattr(request, "client", None)
        return getattr(client, "host", None) or "unknown"

    def check(self, request) -> None:
        """Raise RateLimited if this client went over the limit."""
        if self.per_minute <= 0 or request is None:
            return
        key, now = self.client_ip(request), time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                raise RateLimited()
            q.append(now)
            if len(self._hits) > 10_000:  # bound memory
                self._hits = defaultdict(deque, {k: v for k, v in self._hits.items() if v})
