"""Researcher Agent 单测：mock LLM + mock 搜索，验证多查询、压缩、容错。"""
from __future__ import annotations

from types import SimpleNamespace

from src.agents.researcher_agent import _extract_queries, researcher_node


class _SequenceLLM:
    """按顺序返回预设响应；用于 LLM 多次被调用的场景。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        text = self.responses.pop(0) if self.responses else ""
        return SimpleNamespace(content=text)


def _fake_search(query, max_results=3):
    return [
        {"title": f"{query}-标题{i}", "content": f"关于 {query} 的内容 {i}", "url": f"https://example.com/{i}"}
        for i in range(max_results)
    ]


# ---------- _extract_queries ----------

def test_extract_queries_parses_clean_json():
    llm = _SequenceLLM(['["大模型趋势", "Agent 架构", "RAG 优化"]'])
    out = _extract_queries(llm, "brief 内容...")
    assert out == ["大模型趋势", "Agent 架构", "RAG 优化"]


def test_extract_queries_strips_code_fence():
    llm = _SequenceLLM(['```json\n["a", "b"]\n```'])
    assert _extract_queries(llm, "brief 内容") == ["a", "b"]


def test_extract_queries_fallback_on_invalid_json():
    llm = _SequenceLLM(["这不是 JSON\n第二行"])
    out = _extract_queries(llm, "首行就是兜底\n第二行")
    assert out == ["首行就是兜底"]


def test_extract_queries_empty_brief_returns_empty():
    llm = _SequenceLLM([])
    assert _extract_queries(llm, "") == []


# ---------- researcher_node ----------

def test_researcher_node_appends_compressed_entry():
    llm = _SequenceLLM([
        '["LangGraph 教程", "状态机 Agent"]',  # _extract_queries
        "- 要点1 ...\n- 要点2 ...",            # compress_research
    ])
    state = {"query": "LangGraph 怎么用", "research_brief": "搜 LangGraph 与状态机", "research_results": []}
    out = researcher_node(state, llm=llm, search_fn=_fake_search)

    assert "research_results" in out
    assert len(out["research_results"]) == 1
    entry = out["research_results"][0]
    assert entry["source"] == "web_search"
    assert "LangGraph 教程" in entry["query"]
    assert "状态机 Agent" in entry["query"]
    assert "要点1" in entry["content"]


def test_researcher_node_preserves_previous_results():
    llm = _SequenceLLM(['["a"]', "压缩摘要"])
    state = {
        "query": "x",
        "research_brief": "brief",
        "research_results": [{"query": "old", "content": "old content", "source": "web_search"}],
    }
    out = researcher_node(state, llm=llm, search_fn=_fake_search)
    assert len(out["research_results"]) == 2
    assert out["research_results"][0]["content"] == "old content"


def test_researcher_node_handles_search_exception():
    def boom(query, max_results=3):
        raise RuntimeError("network down")

    llm = _SequenceLLM(['["q1"]', "压缩摘要"])
    state = {"query": "x", "research_brief": "brief", "research_results": []}
    out = researcher_node(state, llm=llm, search_fn=boom)
    # 异常不应中断流程，会把错误记入资料供后续 Agent 感知
    assert len(out["research_results"]) == 1
    assert "压缩摘要" in out["research_results"][0]["content"]
