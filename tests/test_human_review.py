"""human_review_agent 单测 + 主图 interactive 模式 e2e (mock)。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agents.human_review_agent import _decision_to_state_update


def _state(**override):
    base = {
        "query": "Q",
        "draft_report": "draft v1",
        "quality_score": {"overall": 5.0, "accuracy": 5.0, "feedback": "需要更多引用"},
        "red_team_feedback": "缺少证据",
        "iteration_count": 1,
        "max_iterations": 3,
    }
    base.update(override)
    return base


# ---------- _decision_to_state_update ----------

def test_approve_raises_score_above_threshold():
    out = _decision_to_state_update({"action": "approve"}, _state())
    assert out["quality_score"]["overall"] == 7.5


def test_approve_preserves_higher_score():
    out = _decision_to_state_update({"action": "approve"}, _state(quality_score={"overall": 8.5}))
    assert out["quality_score"]["overall"] == 8.5


def test_reject_drops_score_for_red_team():
    out = _decision_to_state_update({"action": "reject"}, _state(quality_score={"overall": 6.0, "feedback": "ok"}))
    assert out["quality_score"]["overall"] == 3.0
    assert "用户驳回" in out["quality_score"]["feedback"]


def test_force_final_maxes_out_iteration_count():
    out = _decision_to_state_update({"action": "force_final"}, _state(iteration_count=1, max_iterations=3))
    assert out["iteration_count"] == 3


def test_force_final_preserves_higher_iteration():
    out = _decision_to_state_update({"action": "force_final"}, _state(iteration_count=5, max_iterations=3))
    assert out["iteration_count"] == 5


def test_edit_report_replaces_draft_and_resets_score():
    out = _decision_to_state_update(
        {"action": "edit_report", "draft": "全新的报告内容"}, _state()
    )
    assert out["draft_report"] == "全新的报告内容"
    assert out["quality_score"] == {}
    assert out["red_team_feedback"] == ""


def test_edit_report_with_empty_draft_no_op():
    out = _decision_to_state_update({"action": "edit_report", "draft": ""}, _state())
    assert out == {}


def test_custom_score_sets_overall():
    out = _decision_to_state_update({"action": "custom_score", "overall": 8.0}, _state())
    assert out["quality_score"]["overall"] == 8.0


def test_custom_score_clips_range():
    out = _decision_to_state_update({"action": "custom_score", "overall": 15.0}, _state())
    assert out["quality_score"]["overall"] == 10.0


def test_custom_score_invalid_falls_back_to_default():
    out = _decision_to_state_update({"action": "custom_score", "overall": "not a number"}, _state())
    # 兜底 7.0
    assert out["quality_score"]["overall"] == 7.0


def test_unknown_action_treated_as_approve():
    out = _decision_to_state_update({"action": "unknown_xyz"}, _state(quality_score={"overall": 5.0}))
    assert out["quality_score"]["overall"] == 7.5


# ---------- 主图 interactive 模式：完整 e2e mock ----------

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


@pytest.fixture(autouse=True)
def _simple_researcher(monkeypatch):
    monkeypatch.setenv("RESEARCHER_MODE", "simple")
    monkeypatch.setenv("ENABLE_RAG", "false")


def _patch_other_agents(monkeypatch, llm):
    factory = lambda *a, **kw: llm  # noqa: E731

    for modname in (
        "src.config",
        "src.agents.draft_agent",
        "src.agents.researcher_agent",
        "src.agents.quality_agent",
        "src.agents.red_team_agent",
        "src.agents.revision_agent",
        "src.agents.final_report_agent",
        "src.tools.compress_tool",
    ):
        monkeypatch.setattr(f"{modname}.get_llm", factory, raising=False)
        monkeypatch.setattr(f"{modname}.get_llm_for", factory, raising=False)

    def fake_search(query, max_results=3):
        return [{"title": "t", "content": "c", "url": "u"}]

    monkeypatch.setattr("src.agents.researcher_agent.run_web_search", fake_search)
    monkeypatch.setattr("src.agents.revision_agent.run_web_search", fake_search)


def test_interactive_graph_pauses_then_approves(monkeypatch):
    """跑一遍 interactive 主图：低分 → human_review 暂停 → 用户 approve → final_report 出。"""
    from langgraph.types import Command

    llm = _ScriptedLLM([
        ("研究资料（已压缩）", "# 初稿"),
        ("请就以下问题生成一份研究简报", "brief"),
        ("待评估对象", '{"accuracy":5,"completeness":5,"logic":5,"citation":5,"overall":5,"feedback":"低"}'),
        ("最终润色", "# 终稿"),
        ("请将以下搜索结果压缩", "压缩摘要"),
        ("提炼出 2-4 条最适合", '["q1"]'),
    ])
    _patch_other_agents(monkeypatch, llm)

    from src.graph import build_main_graph

    graph = build_main_graph(interactive=True)
    config = {"configurable": {"thread_id": "test-thread"}, "recursion_limit": 50}
    initial = {
        "query": "测试", "research_brief": "", "research_results": [], "draft_report": "",
        "final_report": "", "quality_score": {}, "red_team_feedback": "",
        "iteration_count": 0, "max_iterations": 3, "next_agent": "",
        "is_complete": False, "messages": [],
    }

    # 第 1 阶段：跑到 human_review 暂停
    interrupted = False
    for event in graph.stream(initial, config=config):
        if "__interrupt__" in event:
            interrupted = True
            break
    assert interrupted, "应该在 human_review 暂停"

    # 第 2 阶段：用户选 approve → 继续
    for event in graph.stream(Command(resume={"action": "approve"}), config=config):
        pass

    snapshot = graph.get_state(config)
    assert snapshot.values["final_report"] == "# 终稿"
    assert snapshot.values["quality_score"]["overall"] >= 7.5


def test_interactive_graph_reject_triggers_red_team(monkeypatch):
    """用户 reject → 应该走 red_team → revision → 重评。

    为了避免无限循环，我们在第二轮 quality_eval 返回高分。
    """
    from langgraph.types import Command

    quality_outputs = iter([
        '{"accuracy":5,"completeness":5,"logic":5,"citation":5,"overall":5,"feedback":"低"}',
        '{"accuracy":9,"completeness":9,"logic":9,"citation":9,"overall":9,"feedback":"好"}',
    ])

    class _LLM:
        def invoke(self, prompt):
            if "待评估对象" in prompt:
                return SimpleNamespace(content=next(quality_outputs))
            if "请基于下面的反馈和资料" in prompt:
                return SimpleNamespace(content="# 初稿 V2")
            if "研究资料（已压缩）" in prompt or "请基于以下研究资料撰写" in prompt:
                return SimpleNamespace(content="# 初稿")
            if "请就以下问题生成一份研究简报" in prompt:
                return SimpleNamespace(content="brief")
            if "极其严苛的学术审稿人" in prompt:
                return SimpleNamespace(content="发现缺陷")
            if "判断是否需要新增搜索" in prompt:
                return SimpleNamespace(content="[]")
            if "最终润色" in prompt:
                return SimpleNamespace(content="# 终稿 V2")
            if "请将以下搜索结果压缩" in prompt:
                return SimpleNamespace(content="摘要")
            if "提炼出 2-4 条最适合" in prompt:
                return SimpleNamespace(content='["q"]')
            return SimpleNamespace(content="")

    _patch_other_agents(monkeypatch, _LLM())

    from src.graph import build_main_graph

    graph = build_main_graph(interactive=True)
    config = {"configurable": {"thread_id": "test-reject"}, "recursion_limit": 50}
    initial = {
        "query": "Q", "research_brief": "", "research_results": [], "draft_report": "",
        "final_report": "", "quality_score": {}, "red_team_feedback": "",
        "iteration_count": 0, "max_iterations": 3, "next_agent": "",
        "is_complete": False, "messages": [],
    }

    # 跑到第一次暂停
    for event in graph.stream(initial, config=config):
        if "__interrupt__" in event:
            break

    # 用户 reject → 应该走 red_team / revision → 第二次 quality_eval（高分）→ 第二次 human_review 暂停
    for event in graph.stream(Command(resume={"action": "reject"}), config=config):
        if "__interrupt__" in event:
            break

    snapshot = graph.get_state(config)
    # 已经做过一次 revision
    assert snapshot.values["iteration_count"] >= 1
    # 当前 draft 是 V2
    assert snapshot.values["draft_report"] == "# 初稿 V2"

    # 第二次 approve → 结稿
    for event in graph.stream(Command(resume={"action": "approve"}), config=config):
        pass

    final = graph.get_state(config).values
    assert final["final_report"] == "# 终稿 V2"
