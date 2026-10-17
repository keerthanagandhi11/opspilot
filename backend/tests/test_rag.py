import asyncio

from app.llm import LLMError
from app.rag import NOT_FOUND_TEXT, answer_question, heuristic_summary, summarize


class FakeLLM:
    def __init__(self, reply=None, error=False, data=None):
        self.reply, self.error, self.data = reply, error, data
        self.prompts = []

    enabled = True

    async def chat(self, system, user, json_mode=False):
        self.prompts.append(user)
        if self.error:
            raise LLMError("boom")
        return self.reply

    async def chat_json(self, system, user):
        if self.error:
            raise LLMError("boom")
        return self.data


class NoLLM:
    enabled = False


def run(coro):
    return asyncio.run(coro)


def test_extractive_answer_cites_root_cause_page(sample_index):
    _, index = sample_index
    a = run(answer_question("Why did the job fail?", index, NoLLM(), 5))
    assert a.found and a.mode == "extractive"
    assert "return code 12" in a.text
    assert any(s.label == "incident_report_INC-1042.pdf p.2" for s in a.sources)


def test_extractive_refuses_when_absent(sample_index):
    _, index = sample_index
    a = run(answer_question("Who was the on-call engineer?", index, NoLLM(), 5))
    assert not a.found and a.text == NOT_FOUND_TEXT and a.sources == []


def test_timeline_lists_events_in_order(sample_index):
    _, index = sample_index
    a = run(answer_question("Give me a timeline", index, NoLLM(), 5))
    assert a.text.index("02:10") < a.text.index("03:05") < a.text.index("03:40")
    assert {s.file for s in a.sources} == {"shift_log_night.pdf"}


def test_llm_answer_maps_citations_to_pages(sample_index):
    _, index = sample_index
    llm = FakeLLM(reply="The input file was missing, so it exited with return code 12 [1].")
    a = run(answer_question("Why did the job fail?", index, llm, 5))
    assert a.found and a.mode == "llm"
    assert "[1]" not in a.text
    assert len(a.sources) == 1
    assert "Sources:" in llm.prompts[0] and "Question: Why did the job fail?" in llm.prompts[0]


def test_llm_not_found(sample_index):
    _, index = sample_index
    a = run(answer_question("Why did the job fail?", index, FakeLLM(reply="NOT_FOUND"), 5))
    assert not a.found and a.text == NOT_FOUND_TEXT


def test_llm_skipped_when_nothing_relevant(sample_index):
    _, index = sample_index
    llm = FakeLLM(reply="should not be called")
    a = run(answer_question("Who was the on-call engineer?", index, llm, 5))
    assert not a.found and llm.prompts == []


def test_llm_error_falls_back_to_quotes(sample_index):
    _, index = sample_index
    a = run(answer_question("Who approved the rerun?", index, FakeLLM(error=True), 5))
    assert a.found and a.mode == "extractive" and a.notice
    assert "shift lead" in a.text


def test_heuristic_summary_flags_missing_owner(sample_index):
    chunks, _ = sample_index
    s = heuristic_summary(chunks)
    assert "nightly_batch_07" in s["what_happened"]
    assert "No on-call owner is named." in s["missing"]


def test_llm_summary_normalises_fields(sample_index):
    chunks, _ = sample_index
    llm = FakeLLM(data={"what_happened": "x", "impact": "y", "status": "Resolved", "missing": "z"})
    s = run(summarize(chunks, llm))
    assert s == {"what_happened": "x", "impact": "y", "status": "Resolved", "missing": ["z"], "mode": "llm"}


def test_answers_skip_page_headings(sample_index):
    _, index = sample_index
    a = run(answer_question("Why did the job fail?", index, NoLLM(), 5))
    assert "Root Cause Analysis" not in a.text
    assert a.text.startswith("Root cause: the upstream feed was delivered at 02:55")
