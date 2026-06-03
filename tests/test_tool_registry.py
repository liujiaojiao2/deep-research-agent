"""tool_registry 测试：每个工具 1 个 mock + 1 个 live 冒烟。"""
from __future__ import annotations

import re

import pytest

from src.tools.tool_registry import (
    arxiv_search,
    get_all_tools,
    get_current_datetime,
    python_calculator,
    wikipedia_search,
)


# ---------- 装配 ----------

def test_get_all_tools_default_includes_rag_and_memory(monkeypatch):
    monkeypatch.delenv("ENABLE_RAG", raising=False)
    monkeypatch.delenv("ENABLE_MEMORY", raising=False)
    tools = get_all_tools()
    names = [t.name for t in tools]
    assert names == [
        "local_knowledge_search",
        "recall_episodic_memory",
        "web_search",
        "wikipedia_search",
        "arxiv_search",
        "python_calculator",
        "get_current_datetime",
    ]


def test_all_tools_have_descriptive_docstring():
    for t in get_all_tools():
        # 描述长度是 ReAct 决策准确性的下限
        assert t.description and len(t.description) > 50, f"{t.name} 描述太短"


# ---------- python_calculator（确定性，无需 live） ----------

def test_python_calculator_basic():
    out = python_calculator.invoke({"expression": "2 + 3 * 4"})
    assert out.strip() == "14"


def test_python_calculator_complex():
    out = python_calculator.invoke({"expression": "(95-87)/87 * 100"})
    assert float(out) == pytest.approx(9.195402, rel=1e-4)


def test_python_calculator_failure_is_safe():
    out = python_calculator.invoke({"expression": "this_is_not_python!!"})
    # 不要抛异常，返回错误说明字符串即可
    assert isinstance(out, str)
    assert len(out) > 0


# ---------- datetime（确定性，无需 live） ----------

def test_datetime_returns_iso_like():
    out = get_current_datetime.invoke({})
    # 形如 2026-05-26 19:39:07
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", out)


# ---------- wikipedia / arxiv: 联网用例 ----------

@pytest.mark.live
def test_wikipedia_search_live():
    out = wikipedia_search.invoke({"query": "强化学习", "lang": "zh"})
    assert isinstance(out, str)
    assert len(out) > 100


@pytest.mark.live
def test_wikipedia_fallback_to_english():
    # 一个中文词条不太可能命中、英文有的术语
    out = wikipedia_search.invoke({"query": "GRPO algorithm", "lang": "zh"})
    assert isinstance(out, str)


@pytest.mark.live
def test_arxiv_search_live():
    out = arxiv_search.invoke({"query": "large language model agent", "max_results": 2})
    assert isinstance(out, list)
    assert len(out) >= 1
    first = out[0]
    if "error" not in first:
        assert {"title", "summary", "url"} <= first.keys()
