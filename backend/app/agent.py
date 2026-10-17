"""Agentic RAG.

Instead of "retrieve once, then answer", the LLM drives the retrieval itself:

  1. Plan     - it reads the question and decides what to look up.
  2. Act      - it calls tools (search, read a page, build a timeline, list files),
                as many times as it needs, rewording searches that miss.
  3. Answer   - it writes an answer that cites source ids like [S2].
  4. Verify   - a second LLM call checks every claim against the cited pages.
                Unsupported claims send the agent back for one revision.

Every step is yielded as it happens, so the UI can show the agent's work live.
If the LLM is unavailable the caller falls back to the simpler pipeline in rag.py."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

from .config import Settings
from .llm import LLMClient, LLMError, RateLimitError, ToolCallError
from .rag import (
    DENSE_MIN,
    NOT_FOUND_TEXT,
    Answer,
    Source,
    answer_question,
    extractive_answer,
    timestamped_events,
)
from .sessions import Session

log = logging.getLogger(__name__)

AGENT_SYSTEM = """You are OpsPilot, an agent that investigates IT incidents using the user's uploaded PDFs.
Loaded documents: {files}

How to work:
1. Plan: split multi-part questions and look up each part.
2. Use the tools. search_documents finds passages; if a search misses, try other words once.
   read_page gives a whole page for context. find_timeline lists timestamped events.
   Use as few tool calls as you need: if two different searches find nothing relevant, stop.
   Search all files (leave out `file`) unless the user asks about one specific document.
3. Answer ONLY from tool results, in 1-5 short sentences (a timeline may be a short list).
   Keep exact times, job names and codes. Cite every fact with its source id, e.g. [S2].
4. If the documents do not contain the answer after searching, reply with exactly: NOT_FOUND
Never use outside knowledge and never guess."""

FINAL_ANSWER_NUDGE = (
    "You have used all your tool calls. Answer now using only the tool results above, citing source ids "
    "like [S1]. If they do not contain the answer, reply with exactly: NOT_FOUND"
)

VERIFY_SYSTEM = """You check whether an answer is fully supported by source passages.
Return JSON: {"supported": true or false, "problem": "short description of the unsupported claim, or empty"}.
Rephrasing is fine. Any time, name, number, cause or action that is not in the sources is unsupported."""

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the uploaded PDFs. Returns the most relevant passages with source ids.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for, in a few words"},
                    # null allowed: models often send "file": null meaning "all files", and
                    # providers with strict schema checks (Groq) reject the call otherwise.
                    "file": {
                        "type": ["string", "null"],
                        "description": "Only if the user named one document: its file name",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Read the full text of one page of one PDF.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "Exact file name"},
                    "page": {"type": "integer", "description": "Page number, starting at 1"},
                },
                "required": ["file", "page"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_timeline",
            "description": "List every timestamped event (lines starting with a clock time) in document order.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List the loaded PDF files and their page counts.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# Models format citations differently: [S1], [ S1 ], [S1, S2], [S1; S2], 【S1】 (gpt-oss style).
_CITE_RE = re.compile(r"[\[【]\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*[\]】]")
_ODD_SPACES = re.compile(r"[\u00a0\u2007\u202f\u2009]")
SEARCH_SNIPPET_CHARS = 500  # per passage; keeps each agent turn small for free-tier token limits
READ_PAGE_CHARS = 2500


@dataclass
class Step:
    kind: str  # search | read | timeline | list | check | info
    title: str
    detail: str = ""


class SourceRegistry:
    """Gives every page the agent has seen a stable id (S1, S2, ...) for citations."""

    def __init__(self) -> None:
        self._ids: dict[tuple[str, int], str] = {}
        self._text: dict[str, str] = {}
        self._where: dict[str, tuple[str, int]] = {}

    def add(self, file: str, page: int, text: str) -> str:
        sid = self._ids.get((file, page))
        if sid is None:
            sid = f"S{len(self._ids) + 1}"
            self._ids[(file, page)] = sid
            self._where[sid] = (file, page)
        if len(text) > len(self._text.get(sid, "")):
            self._text[sid] = text
        return sid

    def __contains__(self, sid: str) -> bool:
        return sid in self._where

    def text(self, sid: str) -> str:
        return self._text.get(sid, "")

    def source(self, sid: str, limit: int = 280) -> Source:
        file, page = self._where[sid]
        t = self._text[sid]
        snip = t if len(t) <= limit else t[:limit].rsplit(" ", 1)[0] + "…"
        return Source(file, page, snip)


class Toolbox:
    def __init__(self, session: Session, registry: SourceRegistry, settings: Settings):
        self.s = session
        self.reg = registry
        self.settings = settings

    def run(self, name: str, raw_args: str | dict | None) -> tuple[Step, str]:
        try:
            args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            return Step("info", f"Invalid arguments for {name}"), "Error: arguments were not valid JSON."
        handler = {
            "search_documents": self.search_documents,
            "read_page": self.read_page,
            "find_timeline": self.find_timeline,
            "list_documents": self.list_documents,
        }.get(name)
        if handler is None:
            return Step("info", f"Unknown tool {name}"), f"Error: there is no tool called {name}."
        try:
            return handler(**args)
        except TypeError:
            return Step("info", f"Bad arguments for {name}"), f"Error: wrong arguments for {name}."

    def search_documents(self, query: str = "", file: str | None = None) -> tuple[Step, str]:
        query = str(query).strip()[:200]
        if not query:
            return Step("search", "Searched with an empty query"), "Error: query is empty."
        hits = self.s.index.search(query, k=self.settings.top_k * 2)
        if file:
            hits = [h for h in hits if h.chunk.file == file]
        hits = [h for h in hits if h.bm25 > 0 or h.dense >= DENSE_MIN][: self.settings.top_k]
        title = f"Searched: “{query}”" + (f" in {file}" if file else "")
        if not hits:
            return Step("search", title, "No matching passages."), (
                "No matching passages. Try different words, or the answer may not be in the documents."
            )
        lines, labels = [], []
        for h in hits:
            sid = self.reg.add(h.chunk.file, h.chunk.page, h.chunk.text)
            lines.append(f"[{sid}] {h.chunk.label}\n{h.chunk.text[:SEARCH_SNIPPET_CHARS]}")
            labels.append(h.chunk.label)
        return Step("search", title, "Found: " + ", ".join(dict.fromkeys(labels))), "\n\n".join(lines)

    def read_page(self, file: str = "", page: int | str = 0) -> tuple[Step, str]:
        try:
            page = int(page)
        except (TypeError, ValueError):
            page = 0
        text = self.s.pages.get((file, page))
        if text is None:
            names = ", ".join(f"{f.name} ({f.pages} pages)" for f in self.s.files)
            return Step("read", f"Tried to read {file} p.{page}", "That page does not exist."), (
                f"Error: no such page. Files: {names}"
            )
        sid = self.reg.add(file, page, text)
        return Step("read", f"Read {file} p.{page}"), f"[{sid}] {file} p.{page}\n{text[:READ_PAGE_CHARS]}"

    def find_timeline(self) -> tuple[Step, str]:
        events = timestamped_events(self.s.chunks)
        if not events:
            return Step("timeline", "Looked for a timeline", "No timestamped lines found."), (
                "No timestamped lines found."
            )
        out = []
        for sent, c in events[:30]:
            sid = self.reg.add(c.file, c.page, self.s.pages.get((c.file, c.page), c.text))
            out.append(f"[{sid}] {sent}")
        return Step("timeline", "Built a timeline", f"{len(events)} timestamped events"), "\n".join(out)

    def list_documents(self) -> tuple[Step, str]:
        body = "\n".join(f"{f.name}: {f.pages} pages" for f in self.s.files)
        return Step("list", "Listed loaded documents"), body


def _cited_ids(text: str, reg: SourceRegistry) -> list[str]:
    ids: list[str] = []
    for group in _CITE_RE.findall(text):
        for sid in (x.strip() for x in re.split(r"[,;]", group)):
            if sid in reg and sid not in ids:
                ids.append(sid)
    return ids


def _strip_citations(text: str) -> str:
    text = _ODD_SPACES.sub(" ", text)
    text = re.sub(r"\s*" + _CITE_RE.pattern, "", text)
    return re.sub(r"[ \t]+([.,;:])", r"\1", text).strip()


async def _verify(answer: str, ids: list[str], reg: SourceRegistry, llm: LLMClient) -> tuple[bool, str]:
    sources = "\n\n".join(f"[{sid}]\n{reg.text(sid)[:2500]}" for sid in ids)
    data = await llm.chat_json(VERIFY_SYSTEM, f"Sources:\n{sources}\n\nAnswer:\n{answer}")
    ok = bool(data.get("supported", True))
    return ok, str(data.get("problem") or "")


async def run_agent(
    question: str, session: Session, llm: LLMClient, settings: Settings
) -> AsyncIterator[Step | Answer]:
    """Yields Step objects while working, then exactly one Answer."""
    reg = SourceRegistry()
    tools = Toolbox(session, reg, settings)
    files = ", ".join(f"{f.name} ({f.pages} pages)" for f in session.files)
    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM.format(files=files)},
        {"role": "user", "content": question},
    ]
    steps: list[Step] = []
    cite_retry = verify_retry = 0
    final_nudge_sent = False
    notice = None

    def emit(step: Step) -> Step:
        steps.append(step)
        return step

    for turn in range(settings.agent_max_steps + 3):
        out_of_steps = turn >= settings.agent_max_steps
        if out_of_steps and not final_nudge_sent:
            # Some models (e.g. gpt-oss) still try to call a tool when tool_choice="none",
            # which the provider rejects. Leaving the tools out and saying so plainly works.
            final_nudge_sent = True
            messages.append({"role": "user", "content": FINAL_ANSWER_NUDGE})
        try:
            msg = await llm.complete(messages, tools=None if out_of_steps else TOOL_SPECS)
        except ToolCallError:
            if not out_of_steps:
                raise  # let ask() fall back to single-shot RAG
            # Rounds are used up and the model still wants to search: it has no supported answer.
            yield emit(Step("check", "Stopped: no supported answer after all searches"))
            yield Answer(NOT_FOUND_TEXT, False, "agent", steps=steps)
            return
        calls = msg.get("tool_calls") or []

        if calls and not out_of_steps:
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for call in calls[:6]:
                fn = call.get("function") or {}
                step, result = tools.run(fn.get("name", ""), fn.get("arguments"))
                yield emit(step)
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result})
            for call in calls[6:]:  # every call needs a reply, even ones we skip
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": "Skipped."})
            continue

        text = (msg.get("content") or "").strip()
        if not text or "NOT_FOUND" in text.upper():
            yield Answer(NOT_FOUND_TEXT, False, "agent", steps=steps)
            return

        ids = _cited_ids(text, reg)
        if not ids and cite_retry < 1:
            cite_retry += 1
            yield emit(Step("check", "Self-check: no citations, asking the agent to cite its sources"))
            messages += [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": "Cite each fact with source ids like [S1] from the tool results, "
                    "searching first if needed. If the documents do not contain the answer, reply NOT_FOUND.",
                },
            ]
            continue
        if not ids:  # still uncited: don't present unverifiable text
            yield Answer(NOT_FOUND_TEXT, False, "agent", steps=steps)
            return

        if settings.agent_verify:
            try:
                ok, problem = await _verify(_strip_citations(text), ids, reg, llm)
            except LLMError as exc:
                log.info("Verification skipped: %s", exc)
                ok, problem = True, ""
                yield emit(Step("check", "Self-check skipped", "The checker model was unavailable."))
            else:
                if ok:
                    yield emit(Step("check", "Self-check passed", "Every claim is supported by the cited pages."))
                elif verify_retry < 1:
                    verify_retry += 1
                    yield emit(Step("check", "Self-check found an unsupported claim, revising", problem))
                    messages += [
                        {"role": "assistant", "content": text},
                        {
                            "role": "user",
                            "content": f"A reviewer found an unsupported claim: {problem}. "
                            "Revise using only the sources (search again if needed), or reply NOT_FOUND.",
                        },
                    ]
                    continue
                else:
                    notice = "The self-check could not confirm every detail. Please verify against the cited pages."
                    yield emit(Step("check", "Self-check still unsure", problem))

        yield Answer(_strip_citations(text), True, "agent", [reg.source(s) for s in ids], notice, steps)
        return


async def ask(question: str, session: Session, llm: LLMClient, settings: Settings) -> AsyncIterator[Step | Answer]:
    """Entry point used by the API and the UI. Picks the best available strategy:
    agent (LLM with tools) -> single-shot RAG -> extractive quotes."""
    if not llm.enabled:
        hits = session.index.search(question, k=settings.top_k)
        yield extractive_answer(question, hits, session.chunks)
        return
    if not settings.agent_enabled:
        yield await answer_question(question, session.index, llm, settings.top_k)
        return
    produced = False
    try:
        async for item in run_agent(question, session, llm, settings):
            produced = produced or isinstance(item, Answer)
            yield item
    except RateLimitError as exc:
        if produced:
            return
        # Another LLM call would hit the same limit: answer from the PDFs directly.
        log.warning("Agent rate limited; answering in quote mode")
        yield Step("info", "AI limit reached, quoting the PDFs instead", str(exc))
        hits = session.index.search(question, k=settings.top_k)
        ans = extractive_answer(question, hits, session.chunks)
        ans.notice = str(exc)
        yield ans
    except LLMError as exc:
        if produced:
            return
        log.warning("Agent failed (%s); falling back to single-shot RAG", exc)
        yield Step("info", "Agent unavailable, using a simpler search", str(exc))
        ans = await answer_question(question, session.index, llm, settings.top_k)
        ans.notice = ans.notice or "The agent was unavailable, so a simpler search answered this."
        yield ans


async def ask_once(question: str, session: Session, llm: LLMClient, settings: Settings) -> Answer:
    steps: list[Step] = []
    answer: Answer | None = None
    async for item in ask(question, session, llm, settings):
        if isinstance(item, Answer):
            answer = item
        else:
            steps.append(item)
    assert answer is not None
    if not answer.steps:
        answer.steps = steps
    return answer
