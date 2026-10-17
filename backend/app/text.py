"""Small text helpers shared by chunking, retrieval and the extractive answerer."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an the is was were be been being am are did do does done what who whom whose how
    where which to of in on for and or but it its this that these those me my we our you
    your i give tell about with from by as at into than then so if not no can could would should
    will shall may might must there here they them their he she his her has have had any all
    please show list""".split()
)

TIME_TOKEN = "hhmm"  # emitted for every clock time, so "timeline" questions favour timestamped pages

# Question words mapped to words that tend to appear in the answer.
SYNONYMS = {
    "why": ["cause", "reason", "because"],
    "happen": ["summary", "incident", "failed"],
    "fix": ["rerun", "resolved", "action"],
    "next": ["follow", "action"],
    "approv": ["approved", "rerun"],
    "timeline": [TIME_TOKEN, TIME_TOKEN, "started", "completed"],
    "when": [TIME_TOKEN],
    "sequence": [TIME_TOKEN],
    "impact": ["delay", "lost", "affected"],
    "owner": ["lead", "engineer", "on-call"],
    "break": ["fail", "abort", "error"],
    "broke": ["fail", "abort", "error"],
}

_TOKEN_RE = re.compile(r"[a-z0-9_:\-]+")
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}$")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def stem(word: str) -> str:
    """Very light suffix stripping: good enough to match fail/failed/failing."""
    if len(word) <= 4 or any(c.isdigit() for c in word):
        return word
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    out = []
    for w in _TOKEN_RE.findall(text.lower()):
        w = w.strip("-:")
        if len(w) > 1 and w not in STOPWORDS:
            out.append(stem(w))
            if _TIME_RE.match(w):
                out.append(TIME_TOKEN)
    return out


def expand_query(tokens: list[str]) -> list[str]:
    out = list(tokens)
    for t in tokens:
        for key, extra in SYNONYMS.items():
            if t == key or t == stem(key):
                out.extend(stem(e) for e in extra)
    return out


def _paragraphs(text: str) -> list[str]:
    """Re-join lines that PDF extraction broke mid-sentence. A new block starts after a
    sentence end or after a short heading-like line."""
    blocks: list[str] = []
    for line in (ln.strip() for ln in text.split("\n")):
        if not line:
            continue
        prev = blocks[-1] if blocks else ""
        starts_new = (
            not prev
            or prev.endswith((".", "!", "?", ":"))
            or (len(prev) < 60 and not prev.endswith((",", ";")) and not line[:1].islower())
        )
        if starts_new:
            blocks.append(line)
        else:
            blocks[-1] = prev + " " + line
    return blocks


def split_sentences(text: str) -> list[str]:
    out = []
    for block in _paragraphs(text):
        out.extend(s.strip() for s in _SENT_RE.split(block) if len(s.strip()) > 2)
    return out


def is_heading(sentence: str) -> bool:
    """Short line without closing punctuation, e.g. "Root Cause Analysis"."""
    return len(sentence) < 60 and not sentence.rstrip().endswith((".", "!", "?"))
