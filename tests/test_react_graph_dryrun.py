"""主图干跑（ReAct researcher 模式）：验证 RESEARCHER_MODE=react 路径能完整跑完。

策略：
- patch react_researcher_agent.build_react_agent 让 sub-agent 用固定 fake
- 其它 Agent（brief / draft / quality / final）的 LLM 用 ScriptedLLM
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage


@pytest.fixture(autouse=True)
def _force_react_researcher(monkeypatch):
    monkeypatch.setenv("RESEARCHER_MODE", "react")


class _ScriptedLLM:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        for keyword, response in self.script:
            if keyword is None or keyword in prompt:
                return SimpleNamespace(content=response)
        return SimpleNamespace(content="")


class _FakeReactAgent:
    """模拟一次跑通的 ReAct sub-agent。"""

    def invoke(self, inputs, config=None):
        ai_with_calls = AIMessage(
            content="",
            tool_calls=[{"name": "wikipedia_search", "args": {"query": "X"}, "id": "c0"}],
        )
        return {
            "messages": list(inputs.get("messages", [])) + [
                ai_with_calls,
                ToolMessage(content="维基百科返回：X 是...", tool_call_id="c0"),
                AIMessage(content="- X 是一个测试主题 (来源: wikipedia_search)"),
            ]
        }


def _patch_other_agents(monkeypatch, llm):
    """把 brief/draft/quality/final 的 get_llm 都 patch 成同一个 ScriptedLLM。"""
    factory = lambda *a, **kw: llm  # noqa: E731  (测试里 lambda 可接受)
    for modname in (
        "src.config",
        "src.agents.draft_agent",
        "src.agents.quality_agent",
        "src.agents.red_team_agent",
        "src.agents.revision_agent",
        "src.agents.final_report_agent",
        "src.tools.compress_tool",
    ):
        monkeypatch.setattr(f"{modname}.get_llm", factory, raising=False)
        monkeypatch.setattr(f"{modname}.get_llm_for", factory, raising=False)


def test_react_mode_happy_path(monkeypatch):
    llm = _ScriptedLLM([
        ("研究资料（已压缩）", "# 初稿\n## 摘要\n初稿内容..."),
        ("请就以下问题生成一份研究简报", "## 子问题\n- s1"),
        (
            "待评估对象",
            '{"accuracy":8.5,"completeness":8.0,"logic":8.5,"citation":7.5,"overall":8.1,"feedback":"OK"}',
        ),
        ("最终润色", "# 终稿\n## 执行摘要"),
    ])
    _patch_other_agents(monkeypatch, llm)
    # ReAct sub-agent 整个替换
    monkeypatch.setattr(
        "src.agents.react_researcher_agent.build_react_agent",
        lambda **kw: _FakeReactAgent(),
    )

    from src.graph import build_main_graph

    graph = build_main_graph()
    initial = {
        "query": "测试查询",
        "research_brief": "",
        "research_results": [],
        "draft_report": "",
        "final_report": "",
        "quality_score": {},
        "red_team_feedback": "",
        "iteration_count": 0,
        "max_iterations": 3,
        "next_agent": "",
        "is_complete": False,
        "messages": [],
    }
    final = graph.invoke(initial)

    assert final["is_complete"] is True
    assert final["final_report"].startswith("# 终稿")
    # ReAct researcher 把工具用法写入 source，验证它真的被走过
    assert len(final["research_results"]) == 1
    entry = final["research_results"][0]
    assert "react_agent" in entry["source"]
    assert "wikipedia_search" in entry["source"]
    assert "X 是一个测试主题" in entry["content"]
