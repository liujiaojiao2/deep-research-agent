"""主图干跑测试（simple researcher 模式）：monkeypatch 所有外部 IO（LLM + 搜索），验证路由与终止。

两条主路径：
  · 高分一次跑通：brief → researcher → draft → quality(高) → final
  · 低分进入自进化：brief → researcher → draft → quality(低) → red_team → revision → quality(高) → final
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _force_simple_researcher(monkeypatch):
    """本文件中所有用例都跑 simple researcher；ReAct 路径见 test_react_graph_dryrun.py"""
    monkeypatch.setenv("RESEARCHER_MODE", "simple")


class _ScriptedLLM:
    """根据 prompt 关键词路由到预设响应，避免依赖调用顺序。"""

    def __init__(self, script):
        # script: list of (keyword_or_None, response)；None 表示通配兜底
        self.script = list(script)
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        for keyword, response in self.script:
            if keyword is None or keyword in prompt:
                return SimpleNamespace(content=response)
        return SimpleNamespace(content="")


def _fake_search(query, max_results=3):
    return [
        {"title": f"{query}-T{i}", "content": f"{query}-内容{i}", "url": f"https://e.com/{i}"}
        for i in range(max_results)
    ]


def _patch_all(monkeypatch, llm):
    """把每个 Agent 模块里的 get_llm 替换成返回 llm 的工厂；search 改 fake。"""

    def factory(*_args, **_kwargs):
        return llm

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
    monkeypatch.setattr("src.agents.researcher_agent.run_web_search", _fake_search)
    monkeypatch.setattr("src.agents.revision_agent.run_web_search", _fake_search)


def test_happy_path_high_score_one_iter(monkeypatch):
    llm = _ScriptedLLM(
        [
            # researcher._extract_queries：必须返回 JSON 数组
            ("提炼出 2-4 条最适合", '["q1", "q2"]'),
            # compress_research：必须包含"压缩"以便看出
            ("请将以下搜索结果压缩", "- 关键要点: x\n- 来源: https://e.com/0"),
            # draft_writer（先匹配，避免被 brief 关键词截胡）
            ("研究资料（已压缩）", "# 初稿\n## 摘要\n初稿内容..."),
            # brief_writer
            ("请就以下问题生成一份研究简报", "## 子问题\n- s1\n- s2\n## 关键词\n- k1\n## 结构\n- 摘要"),
            # quality_eval：高分 → 直接进 final
            ("待评估对象", '{"accuracy":8.5,"completeness":8.0,"logic":8.5,"citation":7.5,"overall":8.1,"feedback":"OK"}'),
            # final_report
            ("最终润色", "# 终稿\n## 执行摘要\n200 字摘要..."),
        ]
    )
    _patch_all(monkeypatch, llm)

    from src.graph import build_main_graph

    graph = build_main_graph()
    initial = {
        "query": "DeepSeek R1 技术原理",
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
    final = graph.invoke(initial, config={"configurable": {"thread_id": uuid.uuid4().hex}})

    assert final["is_complete"] is True
    assert final["final_report"].startswith("# 终稿")
    assert final["draft_report"].startswith("# 初稿")
    assert final["quality_score"]["overall"] == 8.1
    assert final["iteration_count"] == 0  # 高分一次过，从未走 revision
    assert len(final["research_results"]) == 1


def test_self_evolution_low_then_high(monkeypatch):
    """前 1 轮低分触发 red_team → revision，第 2 轮高分进 final。"""
    # quality_eval 的两次响应：先低后高（通过 list pop 模拟序列状态）
    quality_payloads = iter([
        '{"accuracy":5,"completeness":4,"logic":6,"citation":5,"overall":4.5,"feedback":"完整性不足"}',
        '{"accuracy":8.5,"completeness":8.0,"logic":8.5,"citation":7.5,"overall":8.1,"feedback":"OK"}',
    ])

    def quality_response(prompt):
        return next(quality_payloads)

    class _LLM:
        def __init__(self):
            self.calls = []

        def invoke(self, prompt):
            self.calls.append(prompt)
            # 顺序很关键，但路由匹配比顺序更可靠
            # 顺序敏感：先匹配 draft/revise（它们的 prompt 含有"研究简报"标签）
            if "请基于下面的反馈和资料" in prompt:
                return SimpleNamespace(content="# 初稿 V2\n修订后内容")
            if "研究资料（已压缩）" in prompt or "请基于以下研究资料撰写" in prompt:
                return SimpleNamespace(content="# 初稿\n旧版本")
            if "待评估对象" in prompt:
                return SimpleNamespace(content=quality_response(prompt))
            if "提炼出 2-4 条最适合" in prompt:
                return SimpleNamespace(content='["q1","q2"]')
            if "请将以下搜索结果压缩" in prompt:
                return SimpleNamespace(content="- 摘要要点")
            if "请就以下问题生成一份研究简报" in prompt:
                return SimpleNamespace(content="子问题列表...")
            if "极其严苛的学术审稿人" in prompt:
                return SimpleNamespace(content="## 严重问题\n- 缺少证据")
            if "判断是否需要新增搜索" in prompt:
                return SimpleNamespace(content='["补充关键词"]')
            if "最终润色" in prompt:
                return SimpleNamespace(content="# 终稿\n## 执行摘要")
            return SimpleNamespace(content="")

    llm = _LLM()
    _patch_all(monkeypatch, llm)

    from src.graph import build_main_graph

    graph = build_main_graph()
    initial = {
        "query": "X",
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
    final = graph.invoke(initial, config={"configurable": {"thread_id": uuid.uuid4().hex}})

    assert final["is_complete"] is True
    assert final["final_report"].startswith("# 终稿")
    assert final["draft_report"].startswith("# 初稿 V2")
    assert final["quality_score"]["overall"] == 8.1
    assert final["iteration_count"] == 1     # 走过 1 次 revision
    # research_results：1 条原始 + 1 条 supplement
    assert len(final["research_results"]) == 2
    assert final["research_results"][-1]["source"] == "supplement"
