"""eval 模块单测：judge / runner / report 都用 mock，覆盖核心路径。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.eval.judge import _DEFAULT_JUDGE, judge_report, keyword_hit_rate
from src.eval.report import render_markdown
from src.eval.runner import _extract_tools_from_state


class _StubLLM:
    def __init__(self, payload):
        self.payload = payload
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return SimpleNamespace(content=self.payload)


# ---------- judge_report ----------

def test_judge_parses_clean_json():
    llm = _StubLLM('{"answer_relevance":8.5,"citation":7,"depth":8,"style":9,"overall":8.1,"feedback":"OK"}')
    out = judge_report("Q", "REPORT", ["k1"], ["web_search"], llm=llm)
    assert out["answer_relevance"] == 8.5
    assert out["overall"] == 8.1
    assert out["feedback"] == "OK"


def test_judge_handles_code_fence():
    llm = _StubLLM('```json\n{"answer_relevance":7,"citation":7,"depth":7,"style":7,"overall":7,"feedback":""}\n```')
    out = judge_report("Q", "R", llm=llm)
    assert out["overall"] == 7.0


def test_judge_falls_back_on_invalid_json():
    llm = _StubLLM("不是 JSON")
    out = judge_report("Q", "R", llm=llm)
    assert out["overall"] == _DEFAULT_JUDGE["overall"]
    assert "JSON 解析失败" in out["feedback"]


def test_judge_clips_out_of_range():
    llm = _StubLLM('{"answer_relevance":15,"citation":-2,"depth":7,"style":7,"overall":12,"feedback":""}')
    out = judge_report("Q", "R", llm=llm)
    assert out["answer_relevance"] == 10.0
    assert out["citation"] == 0.0
    assert out["overall"] == 10.0


def test_judge_prompt_includes_query_keywords_tools():
    llm = _StubLLM('{"overall":7,"answer_relevance":7,"citation":7,"depth":7,"style":7,"feedback":""}')
    judge_report("我的问题", "REPORT", ["k1", "k2"], ["web_search"], llm=llm)
    p = llm.last_prompt
    assert "我的问题" in p
    assert "k1, k2" in p
    assert "web_search" in p


# ---------- keyword_hit_rate ----------

def test_keyword_hit_rate_basic():
    out = keyword_hit_rate("报告里有 Python 和 JavaScript", ["Python", "JavaScript", "Rust"])
    assert out["hits"] == 2
    assert out["total"] == 3
    assert out["rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert "Rust" in out["missed"]


def test_keyword_hit_rate_empty_expected():
    out = keyword_hit_rate("any", [])
    assert out == {"hits": 0, "total": 0, "rate": 0.0, "missed": []}


# ---------- _extract_tools_from_state ----------

def test_extract_tools_react_format():
    state = {
        "research_results": [
            {"source": "react_agent(tools=local_knowledge_search,web_search)", "content": ""},
        ]
    }
    assert _extract_tools_from_state(state) == ["local_knowledge_search", "web_search"]


def test_extract_tools_simple_format():
    state = {"research_results": [{"source": "web_search", "content": ""}]}
    assert _extract_tools_from_state(state) == ["web_search"]


def test_extract_tools_handles_multiple_entries():
    state = {
        "research_results": [
            {"source": "react_agent(tools=local_knowledge_search)", "content": ""},
            {"source": "supplement", "content": ""},
        ]
    }
    tools = _extract_tools_from_state(state)
    assert "local_knowledge_search" in tools
    assert "supplement" in tools


# ---------- render_markdown ----------

def test_render_markdown_smoke():
    results = [
        {
            "id": "Q1",
            "category": "rag",
            "query": "项目 Q1 那次事故的根因是什么？",
            "elapsed_sec": 42.1,
            "iteration_count": 0,
            "tools_used": ["local_knowledge_search"],
            "judge_score": {
                "answer_relevance": 9.0, "citation": 8.5, "depth": 8.0,
                "style": 9.0, "overall": 8.7, "feedback": "本地知识引用准确。",
            },
            "keyword_hits": {"hits": 3, "total": 4, "rate": 0.75, "missed": ["DDG"]},
            "final_report": "...",
        },
        {
            "id": "Q2",
            "category": "react",
            "query": "X",
            "error": "GraphRecursionError: 25 hit",
            "elapsed_sec": 12.0,
        },
    ]
    md = render_markdown(results, config_summary="RESEARCHER_MODE=react")
    assert "DeepResearch Agent · Eval 报告" in md
    assert "Q1" in md and "Q2" in md
    assert "GraphRecursionError" in md
    assert "8.7" in md
    assert "RESEARCHER_MODE=react" in md
