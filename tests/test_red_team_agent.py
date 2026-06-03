"""Red Team Agent 单测。"""
from __future__ import annotations

from types import SimpleNamespace

from src.agents.red_team_agent import red_team_node


class _StubLLM:
    def __init__(self, payload):
        self.payload = payload
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return SimpleNamespace(content=self.payload)


def test_red_team_writes_feedback_field():
    llm = _StubLLM("## 严重问题\n- xxx\n## 一般问题\n- yyy")
    out = red_team_node({"query": "Q", "draft_report": "DRAFT"}, llm=llm)
    assert "red_team_feedback" in out
    assert "严重问题" in out["red_team_feedback"]


def test_red_team_prompt_contains_query_and_draft():
    llm = _StubLLM("ok")
    red_team_node({"query": "DeepSeek 历史", "draft_report": "报告正文 ABC"}, llm=llm)
    assert "DeepSeek 历史" in llm.last_prompt
    assert "报告正文 ABC" in llm.last_prompt


def test_red_team_handles_missing_fields():
    llm = _StubLLM("ok")
    out = red_team_node({}, llm=llm)
    assert out["red_team_feedback"] == "ok"
