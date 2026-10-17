"""Agent tests with a scripted fake LLM (no network)."""

import asyncio
import re

import pytest

from app.agent import SourceRegistry, Step, Toolbox, ask_once, run_agent
from app.config import Settings
from app.llm import ToolCallError
from app.rag import NOT_FOUND_TEXT, Answer

SETTINGS = Settings(llm_api_key="test", enable_embeddings=False, agent_max_steps=4)


def call(name, args="{}", cid="c1"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": args}}],
    }


def sid_for(messages, label):
    """Find the source id the tools assigned to e.g. 'incident_report_INC-1042.pdf p.2'."""
    for m in reversed(messages):
        if m["role"] == "tool":
            hit = re.search(r"\[(S\d+)\] " + re.escape(label), m["content"])
            if hit:
                return hit.group(1)
    raise AssertionError(f"{label} not found in tool results")


class ScriptedLLM:
    enabled = True

    def __init__(self, script, verdicts=()):
        self.script = list(script)
        self.verdicts = list(verdicts)
        self.calls = []

    async def complete(self, messages, tools=None, tool_choice="auto", json_mode=False):
        self.calls.append({"messages": list(messages), "tool_choice": tool_choice})
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step(messages) if callable(step) else step

    async def chat_json(self, system, user):
        return self.verdicts.pop(0) if self.verdicts else {"supported": True, "problem": ""}

    async def chat(self, system, user, json_mode=False):
        return "The rerun was approved by the shift lead [1]."


def collect(llm, session, question="Why did the job fail?", settings=SETTINGS):
    async def go():
        return [x async for x in run_agent(question, session, llm, settings)]

    return asyncio.run(go())


def final(items) -> Answer:
    assert isinstance(items[-1], Answer)
    return items[-1]


def answer_citing(label, text="The job failed with return code 12 because the input file was missing"):
    return lambda msgs: {"role": "assistant", "content": f"{text} [{sid_for(msgs, label)}]."}


def test_agent_searches_then_answers_with_page_citation(sample_session):
    llm = ScriptedLLM(
        [
            call("search_documents", '{"query": "job failure root cause"}'),
            answer_citing("incident_report_INC-1042.pdf p.2"),
        ]
    )
    items = collect(llm, sample_session)
    ans = final(items)
    assert ans.found and ans.mode == "agent"
    assert "[S" not in ans.text and ans.text.endswith("missing.")
    assert [s.label for s in ans.sources] == ["incident_report_INC-1042.pdf p.2"]
    kinds = [s.kind for s in items if isinstance(s, Step)]
    assert kinds == ["search", "check"]
    assert "Self-check passed" in ans.steps[-1].title


def test_agent_can_read_pages_and_use_timeline(sample_session):
    llm = ScriptedLLM(
        [
            call("find_timeline"),
            call("read_page", '{"file": "shift_log_night.pdf", "page": 4}'),
            answer_citing("shift_log_night.pdf p.4", "The rerun started at 03:05 and finished at 03:40"),
        ]
    )
    ans = final(collect(llm, sample_session, "When was it fixed?"))
    assert ans.sources[0].label == "shift_log_night.pdf p.4"
    timeline_result = llm.calls[1]["messages"][-1]["content"]
    assert timeline_result.index("02:10") < timeline_result.index("03:40")


def test_agent_not_found(sample_session):
    llm = ScriptedLLM([call("search_documents", '{"query": "on-call engineer"}'), {"content": "NOT_FOUND"}])
    ans = final(collect(llm, sample_session, "Who was the on-call engineer?"))
    assert not ans.found and ans.text == NOT_FOUND_TEXT and ans.sources == []


def test_self_check_failure_triggers_revision(sample_session):
    llm = ScriptedLLM(
        [
            call("search_documents", '{"query": "rerun approved"}'),
            answer_citing("incident_report_INC-1042.pdf p.3", "The rerun was approved by the CTO"),
            answer_citing("incident_report_INC-1042.pdf p.3", "The rerun was approved by the shift lead"),
        ],
        verdicts=[{"supported": False, "problem": "CTO is not mentioned"}, {"supported": True}],
    )
    items = collect(llm, sample_session, "Who approved the rerun?")
    ans = final(items)
    assert "shift lead" in ans.text and ans.notice is None
    titles = [s.title for s in items if isinstance(s, Step)]
    assert any("unsupported" in t for t in titles) and titles[-1] == "Self-check passed"
    assert "CTO is not mentioned" in llm.calls[-1]["messages"][-1]["content"]


def test_uncited_answers_are_rejected(sample_session):
    llm = ScriptedLLM([{"content": "It failed because of a disk error."}, {"content": "Still no citation."}])
    ans = final(collect(llm, sample_session))
    assert not ans.found


def test_agent_is_forced_to_answer_after_max_steps(sample_session):
    loop = [
        call("search_documents", f'{{"query": "root cause attempt {i}"}}', cid=f"c{i}")
        for i in range(SETTINGS.agent_max_steps)
    ]
    llm = ScriptedLLM(loop + [answer_citing("incident_report_INC-1042.pdf p.2")])
    ans = final(collect(llm, sample_session))
    assert ans.found
    assert llm.calls[-1]["tool_choice"] == "none"


def test_ask_falls_back_when_tool_calling_breaks(sample_session):
    llm = ScriptedLLM([ToolCallError("bad tool call")])
    ans = asyncio.run(ask_once("Who approved the rerun?", sample_session, llm, SETTINGS))
    assert ans.mode == "llm" and ans.found and "simpler search" in ans.notice
    assert ans.steps[-1].kind == "info"


def test_ask_without_key_uses_quote_mode(sample_session):
    class Off:
        enabled = False

    ans = asyncio.run(ask_once("Who approved the rerun?", sample_session, Off(), SETTINGS))
    assert ans.mode == "extractive" and "shift lead" in ans.text


@pytest.mark.parametrize(
    "name,args,expect",
    [
        ("read_page", '{"file": "nope.pdf", "page": 1}', "Error: no such page"),
        ("read_page", '{"file": "shift_log_night.pdf", "page": "x"}', "Error: no such page"),
        ("explode", "{}", "no tool called explode"),
        ("search_documents", "not json", "not valid JSON"),
        ("search_documents", '{"query": "zebra giraffe"}', "No matching passages"),
        ("list_documents", "{}", "shift_log_night.pdf: 5 pages"),
    ],
)
def test_toolbox_handles_bad_input(sample_session, name, args, expect):
    _, result = Toolbox(sample_session, SourceRegistry(), SETTINGS).run(name, args)
    assert expect in result


def test_registry_ids_are_stable_per_page():
    reg = SourceRegistry()
    assert reg.add("a.pdf", 1, "x") == reg.add("a.pdf", 1, "longer text") == "S1"
    assert reg.add("a.pdf", 2, "y") == "S2"
    assert reg.source("S1").snippet == "longer text"
