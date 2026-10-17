"""Hybrid retrieval: BM25 keyword search + optional dense embeddings, merged with
Reciprocal Rank Fusion (RRF). Everything runs locally and costs nothing."""

from __future__ import annotations

import logging
import math
import threading
from collections import Counter
from dataclasses import dataclass

from .chunking import Chunk
from .text import expand_query, tokenize

log = logging.getLogger(__name__)


@dataclass
class Hit:
    chunk: Chunk
    score: float  # fused score used for ranking
    bm25: float  # raw keyword score (0 = no shared words)
    dense: float  # cosine similarity (0 when embeddings are off)


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(docs)) if docs else 0.0
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, dl in zip(self.tf, self.len, strict=True):
            s = 0.0
            for t in query:
                f = tf.get(t)
                if not f:
                    continue
                norm = f + self.k1 * (1 - self.b + self.b * dl / (self.avg or 1))
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / norm
            out.append(s)
        return out


class Embedder:
    """Process-wide, lazily loaded fastembed model (ONNX, CPU only, no torch)."""

    _lock = threading.Lock()
    _model = None
    _failed = False

    @classmethod
    def get(cls, model_name: str):
        if cls._failed:
            return None
        with cls._lock:
            if cls._model is None and not cls._failed:
                try:
                    from fastembed import TextEmbedding

                    cls._model = TextEmbedding(model_name=model_name)
                    log.info("Loaded embedding model %s", model_name)
                except Exception:  # noqa: BLE001 - fall back to keyword-only search
                    log.warning("Embeddings unavailable; using keyword search only", exc_info=True)
                    cls._failed = True
        return cls._model

    @staticmethod
    def embed(model, texts: list[str]) -> list[list[float]]:
        vecs = []
        for v in model.embed(texts):
            v = [float(x) for x in v]
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            vecs.append([x / n for x in v])
        return vecs


class Index:
    def __init__(self, chunks: list[Chunk], use_embeddings: bool, embedding_model: str):
        self.chunks = chunks
        self.bm25 = BM25([tokenize(c.text) for c in chunks])
        self.embedding_model = embedding_model
        self.vectors: list[list[float]] | None = None
        if use_embeddings and chunks:
            model = Embedder.get(embedding_model)
            if model is not None:
                self.vectors = Embedder.embed(model, [f"{c.file} page {c.page}: {c.text}" for c in chunks])

    @property
    def mode(self) -> str:
        return "hybrid" if self.vectors is not None else "keyword"

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if not self.chunks:
            return []
        q_tokens = expand_query(tokenize(query))
        kw = self.bm25.scores(q_tokens)
        dense = [0.0] * len(self.chunks)
        if self.vectors is not None:
            model = Embedder.get(self.embedding_model)
            if model is not None:
                qv = Embedder.embed(model, [query])[0]
                dense = [sum(a * b for a, b in zip(qv, v, strict=True)) for v in self.vectors]

        # Reciprocal Rank Fusion: robust to the two scores being on different scales
        fused = [0.0] * len(self.chunks)
        rankings = [kw] + ([dense] if self.vectors is not None else [])
        for scores in rankings:
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for rank, i in enumerate(order):
                if scores[i] > 0:
                    fused[i] += 1.0 / (60 + rank)

        order = sorted(range(len(self.chunks)), key=lambda i: fused[i], reverse=True)
        return [Hit(self.chunks[i], fused[i], kw[i], dense[i]) for i in order[:k] if fused[i] > 0]
