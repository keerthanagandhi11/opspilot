"""Grounded question answering and incident summaries.

Two modes:
* llm        - retrieved passages are sent to the LLM, which must cite them.
* extractive - no API key: the best matching sentences are returned verbatim.
Both refuse to answer when the PDFs do not contain the answer."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .chunking import Chunk
from .llm import LLMClient, LLMError
from .retrieval import Hit, Index
from .text import TIME_TOKEN, expand_query, is_heading, split_sentences, tokenize

log = logging.getLogger(__name__)

NOT_FOUND_TEXT = (
    "I couldn’t find this in the uploaded PDFs, so I won’t guess. "
    "Try asking about the cause, timeline, actions, or impact."
)
DENSE_MIN = 0.5  # cosine similarity that counts as "related" without shared keywords;
# deliberately loose because the LLM itself replies NOT_FOUND when the passages lack the answer

ANSWER_SYSTEM = """You are OpsPilot, an assistant for IT operations incident reviews.
Answer the question using ONLY the numbered sources provided. Rules:
- Be concise: 1-4 sentences, plain language, keep exact times, job names and codes.
- After each fact, cite its source number in square brackets, e.g. [1] or [2][3].
- Never use outside knowledge and never guess.
- If the sources do not contain the answer, reply with exactly: NOT_FOUND"""

SUMMARY_SYSTEM = """You summarise IT incident documents for an operations team.
Use ONLY the text provided. Return a JSON object with these keys:
"what_happened": one sentence (max 25 words),
"impact": one sentence (max 25 words),
"status": short phrase, e.g. "Resolved at 03:40" or "Ongoing" or "Unknown",
"missing": list of 1-3 short sentences naming important facts the documents do NOT state
(for example an unexplained cause, a missing owner, no follow-up date).
Never invent facts."""


@dataclass
class Source:
    file: str
    page: int
    snippet: str

    @property
    def label(self) -> str:
        return f"{self.file} p.{self.page}"


@dataclass
class Answer:
    text: str
    found: bool
    mode: str
    sources: list[Source] = field(default_factory=list)
    notice: str | None = None
    steps: list = field(default_factory=list)  # agent steps (see agent.Step)


def _source(chunk: Chunk, limit: int = 280) -> Source:
    snip = chunk.text if len(chunk.text) <= limit else chunk.text[:limit].rsplit(" ", 1)[0] + "…"
    return Source(chunk.file, chunk.page, snip)


def _dedupe(chunks: list[Chunk]) -> list[Source]:
    seen, out = set(), []
    for c in chunks:
        if (c.file, c.page) not in seen:
            seen.add((c.file, c.page))
            out.append(_source(c))
    return out


def _relevant(hits: list[Hit]) -> list[Hit]:
    return [h for h in hits if h.bm25 > 0 or h.dense >= DENSE_MIN]


def _best_sentences(text: str, q: set[str], max_sentences: int = 2) -> list[str]:
    sents = [s for s in split_sentences(text) if not is_heading(s)] or split_sentences(text)
    scored = [(len(q & set(tokenize(s))), i, s) for i, s in enumerate(sents)]
    scored = [x for x in scored if x[0] > 0]
    if not scored:
        return sents[:1]
    top = sorted(scored, key=lambda x: (-x[0], x[1]))[:max_sentences]
    return [s for _, _, s in sorted(top, key=lambda x: x[1])]


_TIMESTAMPED = re.compile(r"^\d{1,2}:\d{2}\b")
_HAS_TIME = re.compile(r"\b\d{1,2}:\d{2}\b")


def timestamped_events(chunks: list[Chunk]) -> list[tuple[str, Chunk]]:
    """Sentences that mention a clock time, in document order.

    Shift logs put the time first ("02:10 job aborted"), so those lines are preferred.
    Reports usually mention times mid-sentence ("failover at 14:05"), so when there are
    fewer than two log-style lines, any sentence containing a time is used instead."""

    def collect(pattern: re.Pattern) -> list[tuple[str, Chunk]]:
        events: list[tuple[str, Chunk]] = []
        seen: set[str] = set()
        for c in chunks:
            for s in split_sentences(c.text):
                if pattern.search(s) and not is_heading(s) and s not in seen:
                    seen.add(s)
                    events.append((s, c))
        return events

    log_lines = collect(_TIMESTAMPED)
    return log_lines if len(log_lines) >= 2 else collect(_HAS_TIME)


def _timeline(chunks: list[Chunk], limit: int = 12) -> Answer | None:
    events = timestamped_events(chunks)
    if len(events) < 2:
        return None
    events = events[:limit]
    return Answer(" ".join(e for e, _ in events), True, "extractive", _dedupe([c for _, c in events]))


def extractive_answer(question: str, hits: list[Hit], chunks: list[Chunk] | None = None) -> Answer:
    q = set(expand_query(tokenize(question)))
    if TIME_TOKEN in q and "timeline" in q and chunks:
        tl = _timeline(chunks)
        if tl:
            return tl
    kw_hits = [h for h in hits if h.bm25 > 0]
    if not kw_hits:
        return Answer(NOT_FOUND_TEXT, False, "extractive")
    best = max(h.bm25 for h in kw_hits)
    picked = [h for h in kw_hits if h.bm25 >= best * 0.6][:2]
    picked.sort(key=lambda h: (h.chunk.file, h.chunk.page, h.chunk.id))
    parts: list[str] = []
    for h in picked:
        for s in _best_sentences(h.chunk.text, q):
            if s not in parts:
                parts.append(s)
    return Answer(" ".join(parts), True, "extractive", _dedupe([h.chunk for h in picked]))


async def answer_question(question: str, index: Index, llm: LLMClient, top_k: int) -> Answer:
    hits = index.search(question, k=top_k)
    if not llm.enabled:
        return extractive_answer(question, hits, index.chunks)

    candidates = _relevant(hits)
    if not candidates:
        return Answer(NOT_FOUND_TEXT, False, "llm")

    context = "\n\n".join(f"[{i}] ({h.chunk.label})\n{h.chunk.text}" for i, h in enumerate(candidates, start=1))
    prompt = f"Sources:\n{context}\n\nQuestion: {question}"
    try:
        raw = (await llm.chat(ANSWER_SYSTEM, prompt)).strip()
    except LLMError as exc:
        log.warning("LLM failed, falling back to extractive answer: %s", exc)
        ans = extractive_answer(question, hits, index.chunks)
        ans.notice = "The AI model was unavailable, so this answer quotes the PDFs directly."
        return ans

    if not raw or "NOT_FOUND" in raw.upper():
        return Answer(NOT_FOUND_TEXT, False, "llm")

    cited = []
    for n in re.findall(r"\[(\d{1,2})\]", raw):
        i = int(n) - 1
        if 0 <= i < len(candidates) and candidates[i].chunk not in cited:
            cited.append(candidates[i].chunk)
    text = re.sub(r"\s*\[\d{1,2}\]", "", raw).strip()
    if not cited:  # model forgot to cite: show the top passages it was given
        cited = [h.chunk for h in candidates[:2]]
    return Answer(text, True, "llm", _dedupe(cited))


# ---------------------------------------------------------------- summary

_CHECKS = [
    (["cause", "because", "reason", "due"], "No root cause is stated."),
    (["on-call", "owner", "engineer", "assignee", "responder"], "No on-call owner is named."),
    (["resolved", "completed", "restored", "fixed", "closed"], "No resolution time is recorded."),
    (["follow", "action", "prevent", "review"], "No follow-up actions are listed."),
]


def _best_sentence(chunks: list[Chunk], words: list[str], exclude: tuple = ()) -> str | None:
    q = {w for w in tokenize(" ".join(words))}
    best, best_score = None, 0
    for c in chunks:
        for s in split_sentences(c.text):
            if s in exclude or is_heading(s):
                continue
            score = len(q & set(tokenize(s)))
            if score > best_score:
                best, best_score = s, score
    return best


def _heuristic_status(chunks: list[Chunk]) -> str:
    sentence = _best_sentence(chunks, ["resolved", "completed", "restored", "fixed", "closed", "released"])
    if not sentence:
        return "Unknown"
    time = re.search(r"\b\d{1,2}:\d{2}\b", sentence)
    if not time:
        return sentence
    words = set(tokenize(sentence))
    if words & {"resolv", "restor", "fix", "clos"}:
        label = "Resolved"
    elif "complet" in words:
        label = "Completed"
    else:
        label = "Released"
    return f"{label} at {time.group()}"


def heuristic_summary(chunks: list[Chunk]) -> dict:
    all_tokens = set()
    for c in chunks:
        all_tokens.update(tokenize(c.text))
    missing = [msg for words, msg in _CHECKS if not all_tokens & set(tokenize(" ".join(words)))]
    what = _best_sentence(chunks, ["incident", "failed", "failure", "outage", "error", "delayed"])
    impact = _best_sentence(
        chunks, ["impact", "business", "lost", "affected", "customers", "downtime", "late"], exclude=(what,)
    )
    return {
        "what_happened": what or "Not stated in the PDFs.",
        "impact": impact or "Not stated in the PDFs.",
        "status": _heuristic_status(chunks),
        "missing": missing or ["Nothing obvious; add an AI key for a deeper gap check."],
        "mode": "extractive",
    }


async def summarize(chunks: list[Chunk], llm: LLMClient, max_chars: int = 14000) -> dict:
    if not chunks:
        return {
            "what_happened": "No readable text was found in these PDFs.",
            "impact": "Unknown",
            "status": "Unknown",
            "missing": ["The PDFs may be scanned images; text could not be extracted."],
            "mode": "extractive",
        }
    if not llm.enabled:
        return heuristic_summary(chunks)

    parts, used = [], 0
    for c in chunks:
        block = f"--- {c.label} ---\n{c.text}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    try:
        data = await llm.chat_json(SUMMARY_SYSTEM, "\n".join(parts))
    except LLMError as exc:
        log.warning("Summary via LLM failed: %s", exc)
        fallback = heuristic_summary(chunks)
        missing = fallback["missing"]
        if missing == ["Nothing obvious; add an AI key for a deeper gap check."]:
            missing = []
        fallback["missing"] = (missing + [f"AI check unavailable: {exc}"])[:3]
        return fallback

    missing = data.get("missing") or []
    if isinstance(missing, str):
        missing = [missing]
    impact = str(data.get("impact") or "Not stated in the PDFs.")
    status = str(data.get("status") or "Unknown")
    normalised_status = re.sub(r"[\W_]+", " ", status.casefold()).strip()
    normalised_impact = re.sub(r"[\W_]+", " ", impact.casefold()).strip()
    if normalised_status == normalised_impact:
        status = _heuristic_status(chunks)
    return {
        "what_happened": str(data.get("what_happened") or "Not stated in the PDFs."),
        "impact": impact,
        "status": status,
        "missing": [str(m) for m in missing][:3],
        "mode": "llm",
    }
