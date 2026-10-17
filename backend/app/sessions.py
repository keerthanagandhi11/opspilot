"""In-memory session store with expiry. Uploaded PDFs are never written to disk;
they live only in RAM and are dropped when the session expires."""

from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from .chunking import Chunk
from .retrieval import Index


@dataclass
class FileInfo:
    name: str
    pages: int
    size_bytes: int
    has_text: bool
    sample: bool = False


@dataclass
class Session:
    id: str
    files: list[FileInfo]
    chunks: list[Chunk]
    index: Index
    pages: dict[tuple[str, int], str] = field(default_factory=dict)  # (file, page) -> full page text
    created: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    summary: dict | None = None


class SessionStore:
    def __init__(self, ttl_seconds: int, max_sessions: int):
        self.ttl = ttl_seconds
        self.max = max_sessions
        self._data: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    def _evict(self) -> None:
        now = time.time()
        for sid in [k for k, s in self._data.items() if now - s.last_used > self.ttl]:
            del self._data[sid]
        while len(self._data) > self.max:
            self._data.popitem(last=False)  # least recently used

    def create(
        self,
        files: list[FileInfo],
        chunks: list[Chunk],
        index: Index,
        pages: dict[tuple[str, int], str] | None = None,
    ) -> Session:
        s = Session(id=secrets.token_urlsafe(16), files=files, chunks=chunks, index=index, pages=pages or {})
        with self._lock:
            self._data[s.id] = s
            self._evict()
        return s

    def get(self, sid: str) -> Session | None:
        with self._lock:
            self._evict()
            s = self._data.get(sid)
            if s:
                s.last_used = time.time()
                self._data.move_to_end(sid)
            return s

    def delete(self, sid: str) -> None:
        with self._lock:
            self._data.pop(sid, None)

    def __len__(self) -> int:
        return len(self._data)
