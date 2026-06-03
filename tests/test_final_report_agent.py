from __future__ import annotations

from types import SimpleNamespace

from src.agents.final_report_agent import final_report_node


class _StubLLM:
    def __init__(self, payload):
        self.payload = payload
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return SimpleNamespace(content=self.payload)


def test_writes_final_and_complete_flag():
    llm = _StubLLM("# 最终报告\n## 执行摘要\n...\n## 结论")
    out = final_report_node(
        {
            "query": "Q",
            "draft_report": "原稿",
            "iteration_count": 2,
            "quality_score": {"accuracy": 8, "completeness": 7, "logic": 9, "citation": 7, "overall": 7.8},
        },
        llm=llm,
    )
    assert out["is_complete"] is True
    assert out["final_report"].startswith("# 最终报告")


def test_prompt_includes_scores_and_iterations():
    llm = _StubLLM("ok")
    final_report_node(
        {
            "query": "Q",
            "draft_report": "DRAFT",
            "iteration_count": 3,
            "quality_score": {"accuracy": 8.1, "completeness": 7.2, "logic": 9.0, "citation": 6.5, "overall": 7.7},
        },
        llm=llm,
    )
    p = llm.last_prompt
    assert "3" in p
    assert "7.7" in p
    assert "DRAFT" in p


def test_handles_missing_score():
    llm = _StubLLM("ok")
    out = final_report_node({"query": "Q", "draft_report": "DRAFT"}, llm=llm)
    assert out["is_complete"] is True
    assert "N/A" in llm.last_prompt
