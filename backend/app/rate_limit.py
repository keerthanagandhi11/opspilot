"""Tiny sliding-window rate limiter per client IP, so one visitor of the public
demo cannot burn the whole free LLM quota. Works with FastAPI and Gradio requests."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RateLimited(Exception):
    message = "Too many requests. Please wait a minute and try again."


class RateLimiter:
    def __init__(self, per_minute: int, trust_proxy: bool = False, proxy_hops: int = 1):
        self.per_minute = per_minute
        self.trust_proxy = trust_proxy
        self.proxy_hops = max(1, proxy_hops)
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def client_ip(self, request) -> str:
        """`request` is a fastapi.Request or gradio.Request (same attributes).

        Behind a trusted proxy we look, in order, at:
        1. CF-Connecting-IP / True-Client-IP: set by Cloudflare (in front of Render and
           many hosts), which overwrites any value a visitor sends, so it can't be faked.
        2. X-Forwarded-For: a list the visitor can pre-fill with anything; each proxy
           appends the address it saw, so we count from the right, never the first entry.
        Without a trusted proxy (local runs) the connection address is used."""
        headers = getattr(request, "headers", None) or {}
        if self.trust_proxy:
            for name in ("cf-connecting-ip", "true-client-ip"):
                value = (headers.get(name) or "").strip()
                if value:
                    return value
            fwd = headers.get("x-forwarded-for")
            if fwd:
                hops = [h.strip() for h in fwd.split(",") if h.strip()]
                if hops:
                    return hops[-min(self.proxy_hops, len(hops))]
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
