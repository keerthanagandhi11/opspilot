"""Gradio user interface.

Flow (same as the original demo): upload PDFs or pick the sample -> Start chat ->
chat with page-level citations, with an incident summary in the side panel.
The agent's tool calls stream into the chat as collapsible steps."""

from __future__ import annotations

import asyncio
from pathlib import Path

import gradio as gr

from .agent import Step, ask
from .config import Settings
from .llm import LLMClient
from .pipeline import InputError, build_session, clean_name
from .rag import Answer, summarize
from .rate_limit import RateLimited, RateLimiter
from .sessions import SessionStore

SUGGESTIONS = [
    "Why did the job fail?",
    "Who approved the rerun?",
    "Give me a timeline",
    "Who was the on-call engineer?",
]
SAMPLE_FILES = "incident_report_INC-1042.pdf (3 pages) and shift_log_night.pdf (5 pages)"
STEP_ICONS = {"search": "🔎", "read": "📄", "timeline": "🕒", "list": "🗂️", "check": "✅", "info": "ℹ️"}

CSS = """
.gradio-container {max-width: 1040px !important; margin: 0 auto !important}
#hdr h1 {font-size: 28px; margin: 0}
#hdr p {color: var(--body-text-color-subdued); margin: 4px 0 0; max-width: 60ch}
.badge {display: inline-block; font-size: 13px; padding: 2px 10px; border-radius: 99px; margin-top: 10px;
  background: #fbf0d9; color: #8a5a0b}
.badge.ai {background: #dff1f2; color: #0a555c}
.dark .badge {background: #3a2f14; color: #f0c36d}
.dark .badge.ai {background: #17383c; color: #9fe3e9}
#summary .gap {color: #8a5a0b; background: #fbf0d9; border-radius: 8px; padding: 6px 10px; margin: 4px 0}
.dark #summary .gap {color: #f0c36d; background: #3a2f14}
.suggest button {font-size: 14px !important}
"""


def _theme() -> gr.themes.Base:
    return gr.themes.Soft(
        primary_hue="teal",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Atkinson Hyperlegible"), "system-ui", "sans-serif"],
    )


def _header(settings: Settings) -> str:
    if settings.llm_enabled:
        mode = (
            f"Agentic RAG with {settings.llm_model}" if settings.agent_enabled else f"AI answers ({settings.llm_model})"
        )
        badge = f'<span class="badge ai">{mode}, grounded in your PDFs</span>'
    else:
        badge = '<span class="badge">Quote mode: answers are exact sentences from your PDFs (no AI key set)</span>'
    return (
        '<div id="hdr"><h1>OpsPilot</h1>'
        "<p>Upload incident PDFs, then ask about them. Every answer shows which file and page it came from.</p>"
        f"{badge}</div>"
    )


def _format_answer(a: Answer) -> str:
    parts = [a.text]
    if a.sources:
        parts.append(" ".join(f"`{s.label}`" for s in a.sources))
    if a.notice:
        parts.append(f"> ⚠️ {a.notice}")
    return "\n\n".join(parts)


def _format_passages(a: Answer) -> str:
    return "\n\n".join(f"**{s.label}**\n> " + s.snippet.replace("\n", " ") for s in a.sources)


def _summary_md(sm: dict) -> str:
    missing = "".join(f'<div class="gap">{m}</div>' for m in (sm["missing"] or ["Nothing obvious is missing."]))
    return (
        "### Incident summary\n"
        f"**What happened**  \n{sm['what_happened']}\n\n"
        f"**Impact**  \n{sm['impact']}\n\n"
        f"**Status**  \n{sm['status']}\n\n"
        f"**Missing from the reports**\n{missing}"
    )


def build_ui(settings: Settings, store: SessionStore, llm: LLMClient, limiter: RateLimiter) -> gr.Blocks:
    def over_limit(request: gr.Request | None) -> str | None:
        """Errors are shown as toasts (gr.Warning) instead of gr.Error, which would
        cover the output components with an error overlay."""
        try:
            limiter.check(request)
        except RateLimited as exc:
            return exc.message
        return None

    skip6 = tuple(gr.skip() for _ in range(6))

    # ------------------------------------------------------------ handlers

    async def start(paths, include_sample, old_sid, request: gr.Request):
        if msg := over_limit(request):
            gr.Warning(msg)
            return skip6
        uploads = []
        for p in paths or []:
            path = Path(p)
            uploads.append((clean_name(path.name), path.read_bytes()))
        try:
            session, warnings = await asyncio.to_thread(build_session, uploads, include_sample, settings, store)
        except InputError as exc:
            gr.Warning(str(exc))
            return skip6
        if old_sid:
            store.delete(old_sid)
        loaded = "### Loaded files\n" + "\n".join(
            f"- `{f.name}` · {f.pages} page{'s' if f.pages != 1 else ''}" for f in session.files
        )
        hello = "Your PDFs are loaded. Ask me anything about this incident."
        if warnings:
            hello += "\n\n" + "\n".join(f"> ⚠️ {w}" for w in warnings)
        return (
            session.id,
            gr.update(visible=False),
            gr.update(visible=True),
            loaded,
            [{"role": "assistant", "content": hello}],
            "### Incident summary\nReading your PDFs…",
        )

    async def load_summary(sid):
        if not sid:
            return gr.skip()
        session = store.get(sid)
        if not session:
            return "### Incident summary\nThis chat has expired. Please load your PDFs again."
        if session.summary is None:
            session.summary = await summarize(session.chunks, llm)
        return _summary_md(session.summary)

    def back(sid):
        if sid:
            store.delete(sid)
        return None, gr.update(visible=True), gr.update(visible=False), []

    async def respond(question: str, history: list, sid, request: gr.Request | None):
        question = (question or "").strip()
        history = list(history or [])
        if not question:
            gr.Warning("Type a question first.")
            yield history, ""
            return
        if len(question) > settings.max_question_chars:
            gr.Warning(f"Keep questions under {settings.max_question_chars} characters.")
            yield history, question
            return
        history.append({"role": "user", "content": question})
        if msg := over_limit(request):
            history.append({"role": "assistant", "content": msg})
            yield history, ""
            return
        session = store.get(sid) if sid else None
        if not session:
            history.append(
                {
                    "role": "assistant",
                    "content": "This chat has expired. Click **Change files** and load your PDFs again.",
                }
            )
            yield history, ""
            return

        group = f"agent-{len(history)}"
        parent = None
        if llm.enabled and settings.agent_enabled:
            parent = {
                "role": "assistant",
                "content": "",
                "metadata": {"title": "🧠 Agent is investigating…", "id": group, "status": "pending"},
            }
            history.append(parent)
        else:
            history.append({"role": "assistant", "content": "Searching your PDFs…"})
        yield history, ""

        n_steps = 0
        answer: Answer | None = None
        async for item in ask(question, session, llm, settings):
            if isinstance(item, Step):
                n_steps += 1
                icon = STEP_ICONS.get(item.kind, "•")
                if item.kind == "check" and ("unsupported" in item.title or "unsure" in item.title):
                    icon = "⚠️"
                child = {
                    "role": "assistant",
                    "content": item.detail or " ",
                    "metadata": {"title": f"{icon} {item.title}"},
                }
                if parent is not None:
                    child["metadata"]["parent_id"] = group
                history.append(child)
                yield history, ""
            else:
                answer = item

        if parent is not None:
            parent["metadata"] = {"title": f"🧠 Agent steps ({n_steps})", "id": group, "status": "done"}
        else:
            history.pop()  # the "Searching…" placeholder
        if answer is not None:
            history.append({"role": "assistant", "content": _format_answer(answer)})
            if answer.sources:
                history.append(
                    {
                        "role": "assistant",
                        "content": _format_passages(answer),
                        "metadata": {"title": "📎 Source passages"},
                    }
                )
        yield history, ""

    def suggestion(text: str):
        async def run(history, sid, request: gr.Request):
            async for out in respond(text, history, sid, request):
                yield out

        return run

    # ------------------------------------------------------------ layout

    with gr.Blocks(title="OpsPilot: agentic incident PDF chat") as demo:
        sid = gr.State(None)
        gr.HTML(_header(settings))

        with gr.Column(visible=True) as upload_view:
            files = gr.File(
                label="Drop incident PDFs here, or click to choose one or more files",
                file_count="multiple",
                file_types=[".pdf"],
                type="filepath",
                height=180,
            )
            with gr.Row(equal_height=True):
                sample_btn = gr.Button("Use sample incident PDFs", size="md", scale=0, min_width=240)
                gr.Markdown("No PDFs handy? The sample has a report and a shift log.")
            use_sample = gr.Checkbox(label=f"Include the sample incident: {SAMPLE_FILES}", value=False)
            gr.Markdown("<small>Add your own PDFs, tick the sample, or both. Then start the chat.</small>")
            start_btn = gr.Button("Start chat", variant="primary")

        with gr.Row(visible=False, equal_height=False) as chat_view:
            with gr.Column(scale=1, min_width=260):
                loaded_md = gr.Markdown()
                summary_md = gr.Markdown(elem_id="summary")
                back_btn = gr.Button("Change files")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    height=540,
                    show_label=False,
                    layout="bubble",
                    feedback_options=None,
                    buttons=["copy"],
                    placeholder="Ask about this incident.",
                )
                with gr.Row(elem_classes="suggest"):
                    sugg_btns = [gr.Button(t, size="sm") for t in SUGGESTIONS]
                with gr.Row(equal_height=True):
                    q = gr.Textbox(
                        placeholder="Ask about this incident, for example: why did it fail?",
                        show_label=False,
                        max_length=settings.max_question_chars,
                        scale=5,
                        autofocus=True,
                    )
                    send = gr.Button("Send", variant="primary", scale=1, min_width=90)

        gr.Markdown(
            "<small>Open source · MIT licence · your PDFs stay in memory and are discarded after "
            f"{settings.session_ttl_minutes} minutes · [API docs](/docs)</small>"
        )

        # ------------------------------------------------------------ events
        # runs in the browser (js=...), so the box is ticked before "Start chat" can be clicked
        sample_btn.click(None, js="() => true", outputs=use_sample)
        start_btn.click(
            start,
            inputs=[files, use_sample, sid],
            outputs=[sid, upload_view, chat_view, loaded_md, chatbot, summary_md],
            concurrency_limit=4,
        ).then(load_summary, inputs=sid, outputs=summary_md)
        back_btn.click(back, inputs=sid, outputs=[sid, upload_view, chat_view, chatbot])

        ask_kwargs = {"outputs": [chatbot, q], "concurrency_limit": 8}
        send.click(respond, inputs=[q, chatbot, sid], **ask_kwargs)
        q.submit(respond, inputs=[q, chatbot, sid], **ask_kwargs)
        for btn, text in zip(sugg_btns, SUGGESTIONS, strict=True):
            btn.click(suggestion(text), inputs=[chatbot, sid], **ask_kwargs)

    return demo


def mount(app, settings: Settings, store: SessionStore, llm: LLMClient, limiter: RateLimiter):
    """Serve the Gradio UI at "/" on the FastAPI app (the /api routes keep working)."""
    demo = build_ui(settings, store, llm, limiter)
    return gr.mount_gradio_app(
        app,
        demo,
        path="/",
        theme=_theme(),
        css=CSS,
        max_file_size=f"{int(settings.max_file_mb)}mb",
        footer_links=["api"],
    )
