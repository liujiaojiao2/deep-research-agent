"""ReAct researcher 单测。

策略：patch build_react_agent 返回一个 fake_agent；fake_agent.invoke 返回
预设的消息序列（模拟一次完整的 ReAct 跑完后的 messages）。
"""
from __future__ import annotations


from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents.react_researcher_agent import (
    _extract_final_answer,
    _extract_tool_usage,
    react_researcher_node,
)


def _ai_with_calls(calls):
    msg = AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": f"call_{i}"} for i, (name, args) in enumerate(calls)
    ])
    return msg


# ---------- 辅助函数 ----------

def test_extract_tool_usage_records_all_calls():
    msgs = [
        HumanMessage(content="brief"),
        _ai_with_calls([("wikipedia_search", {"query": "X"})]),
        ToolMessage(content="...", tool_call_id="call_0"),
        _ai_with_calls([("arxiv_search", {"query": "Y"}), ("web_search", {"query": "Z"})]),
        ToolMessage(content="...", tool_call_id="call_0"),
        ToolMessage(content="...", tool_call_id="call_1"),
        AIMessage(content="最终总结"),
    ]
    assert _extract_tool_usage(msgs) == ["wikipedia_search", "arxiv_search", "web_search"]


def test_extract_final_answer_finds_last_pure_ai():
    msgs = [
        HumanMessage(content="brief"),
        _ai_with_calls([("web_search", {"query": "X"})]),
        ToolMessage(content="...", tool_call_id="call_0"),
        AIMessage(content="这是最终总结"),
    ]
    assert _extract_final_answer(msgs) == "这是最终总结"


def test_extract_final_answer_empty_when_only_tool_calls():
    msgs = [
        HumanMessage(content="brief"),
        _ai_with_calls([("web_search", {"query": "X"})]),
    ]
    assert _extract_final_answer(msgs) == ""


# ---------- react_researcher_node 主流程 ----------

class _FakeReactAgent:
    """模拟 create_react_agent 编译后的对象。"""

    def __init__(self, final_messages):
        self.final_messages = final_messages
        self.invocations = []

    def invoke(self, inputs, config=None):
        self.invocations.append((inputs, config))
        return {"messages": list(inputs.get("messages", [])) + self.final_messages}


def test_react_researcher_writes_entry_with_summary_and_tools(monkeypatch):
    fake = _FakeReactAgent([
        _ai_with_calls([("wikipedia_search", {"query": "RAG"})]),
        ToolMessage(content="wiki: RAG = ...", tool_call_id="call_0"),
        _ai_with_calls([("arxiv_search", {"query": "RAG retrieval"})]),
        ToolMessage(content="arxiv: paper X", tool_call_id="call_0"),
        AIMessage(content="- RAG 是检索增强生成 (来源: wikipedia_search)\n- ..."),
    ])
    monkeypatch.setattr("src.agents.react_researcher_agent.build_react_agent", lambda **kw: fake)

    state = {"query": "什么是 RAG", "research_brief": "请研究 RAG 的原理", "research_results": []}
    out = react_researcher_node(state)

    assert "research_results" in out
    assert len(out["research_results"]) == 1
    entry = out["research_results"][0]
    assert "RAG 是检索增强生成" in entry["content"]
    assert "react_agent" in entry["source"]
    assert "wikipedia_search" in entry["source"]
    assert "arxiv_search" in entry["source"]
    # 主流程必须把 brief 包成 HumanMessage 喂进去
    sent_msgs = fake.invocations[0][0]["messages"]
    assert isinstance(sent_msgs[0], HumanMessage)
    assert "研究简报" in sent_msgs[0].content


def test_react_researcher_preserves_previous_results(monkeypatch):
    fake = _FakeReactAgent([AIMessage(content="新总结")])
    monkeypatch.setattr("src.agents.react_researcher_agent.build_react_agent", lambda **kw: fake)

    state = {
        "query": "Q",
        "research_brief": "brief",
        "research_results": [{"query": "old", "content": "旧资料", "source": "react_agent"}],
    }
    out = react_researcher_node(state)
    assert len(out["research_results"]) == 2
    assert out["research_results"][0]["content"] == "旧资料"


def test_react_researcher_swallows_subagent_exception(monkeypatch):
    class _BoomAgent:
        def invoke(self, *a, **kw):
            raise RuntimeError("recursion limit hit")

    monkeypatch.setattr("src.agents.react_researcher_agent.build_react_agent", lambda **kw: _BoomAgent())
    out = react_researcher_node({"query": "Q", "research_brief": "b", "research_results": []})
    entry = out["research_results"][0]
    assert entry["source"] == "react_agent_error"
    assert "recursion limit hit" in entry["content"]


def test_react_researcher_handles_empty_final_answer(monkeypatch):
    """LLM 只调工具但没给最终总结（罕见，但要兜底）"""
    fake = _FakeReactAgent([
        _ai_with_calls([("web_search", {"query": "X"})]),
        ToolMessage(content="...", tool_call_id="call_0"),
    ])
    monkeypatch.setattr("src.agents.react_researcher_agent.build_react_agent", lambda **kw: fake)
    out = react_researcher_node({"query": "Q", "research_brief": "b", "research_results": []})
    entry = out["research_results"][0]
    assert "未产生总结" in entry["content"]


def test_react_researcher_passes_recursion_limit_config(monkeypatch):
    fake = _FakeReactAgent([AIMessage(content="ok")])
    monkeypatch.setattr("src.agents.react_researcher_agent.build_react_agent", lambda **kw: fake)
    react_researcher_node(
        {"query": "Q", "research_brief": "b", "research_results": []},
        recursion_limit=15,
    )
    config = fake.invocations[0][1]
    assert config["recursion_limit"] == 15
