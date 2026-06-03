"""Memory 模块测试：archive / recall / preference / tool / archive node。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agents.memory_archive_agent import memory_archive_node
from src.memory import (
    _parse_json_list,
    _summarize_research,
    extract_preferences,
)
from src.tools.memory_tool import recall_episodic_memory


def _state(**override):
    base = {
        "query": "GRPO 训练原理",
        "final_report": "# 报告\n## 摘要\nGRPO 是组内相对策略优化，关键创新是去掉了价值网络...",
        "quality_score": {"overall": 8.5},
        "iteration_count": 0,
        "research_results": [
            {"source": "react_agent(tools=arxiv_search,web_search)", "content": "...", "query": "GRPO"}
        ],
    }
    base.update(override)
    return base


# ---------- _summarize_research ----------

def test_summarize_includes_key_fields():
    s = _summarize_research(_state())
    assert "GRPO 训练原理" in s
    assert "overall=8.5" in s
    assert "arxiv_search" in s
    assert "web_search" in s


def test_summarize_handles_missing_tools():
    s = _summarize_research(_state(research_results=[{"source": "simple", "content": "x"}]))
    assert "GRPO 训练原理" in s


# ---------- preference 抽取 ----------

class _StubLLM:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, prompt):
        return SimpleNamespace(content=self.payload)


def test_parse_json_list_clean():
    out = _parse_json_list('[{"preference":"a","evidence":"b"}]')
    assert out == [{"preference": "a", "evidence": "b"}]


def test_parse_json_list_with_noise():
    out = _parse_json_list('好的，结果是：\n[{"preference":"x"}]\n感谢')
    assert out == [{"preference": "x"}]


def test_parse_json_list_returns_empty_on_failure():
    assert _parse_json_list("not json at all") == []


def test_extract_preferences_basic():
    payload = '[{"preference":"用户偏好分点列表","evidence":"报告用了大量列表"},{"preference":"用户关注 RL 后训练","evidence":"问题涉及 GRPO"}]'
    llm = _StubLLM(payload)
    out = extract_preferences(_state(), llm)
    assert len(out) == 2
    assert out[0]["preference"] == "用户偏好分点列表"


def test_extract_preferences_cap_at_3():
    payload = "[" + ",".join(
        ['{"preference":"p%d"}' % i for i in range(6)]
    ) + "]"
    out = extract_preferences(_state(), _StubLLM(payload))
    assert len(out) == 3


def test_extract_preferences_skips_invalid_entries():
    payload = '[{"foo":"no preference key"}, {"preference":"good one"}]'
    out = extract_preferences(_state(), _StubLLM(payload))
    assert len(out) == 1
    assert out[0]["preference"] == "good one"


def test_extract_preferences_skips_when_no_final_report():
    out = extract_preferences(_state(final_report=""), _StubLLM("[]"))
    assert out == []


# ---------- recall_episodic_memory 工具 ----------

def test_recall_tool_empty_returns_friendly(monkeypatch):
    monkeypatch.setattr("src.memory.recall_episodic", lambda **kw: [])
    out = recall_episodic_memory.invoke({"query": "x"})
    assert len(out) == 1
    assert "无过往研究记录" in out[0]["summary"]


def test_recall_tool_returns_hits(monkeypatch):
    fake = [{"summary": "过去研究 GRPO", "query": "GRPO", "date": "2026-05-26", "overall_score": 8.5, "similarity": 0.92}]
    monkeypatch.setattr("src.memory.recall_episodic", lambda **kw: fake)
    out = recall_episodic_memory.invoke({"query": "GRPO 训练"})
    assert out == fake


def test_recall_tool_swallows_exception(monkeypatch):
    def boom(**kw):
        raise RuntimeError("chromadb crashed")

    monkeypatch.setattr("src.memory.recall_episodic", boom)
    out = recall_episodic_memory.invoke({"query": "x"})
    assert "异常" in out[0]["summary"]


# ---------- memory_archive_node ----------

def test_archive_node_returns_observability_fields(monkeypatch):
    monkeypatch.setattr("src.agents.memory_archive_agent.archive_episodic", lambda s: {"id": "x"})
    monkeypatch.setattr("src.agents.memory_archive_agent.extract_preferences", lambda s, llm: [{"preference": "p1"}])
    monkeypatch.setattr("src.agents.memory_archive_agent.archive_preferences", lambda ps: len(ps))

    out = memory_archive_node(_state(), llm=_StubLLM("[]"))
    assert out["memory_archived"] is True
    assert out["memory_preferences_added"] == 1


def test_archive_node_swallows_archive_exception(monkeypatch):
    def boom(s):
        raise RuntimeError("disk full")

    monkeypatch.setattr("src.agents.memory_archive_agent.archive_episodic", boom)
    monkeypatch.setattr("src.agents.memory_archive_agent.extract_preferences", lambda s, llm: [])
    monkeypatch.setattr("src.agents.memory_archive_agent.archive_preferences", lambda ps: 0)

    out = memory_archive_node(_state(), llm=_StubLLM("[]"))
    # 异常被吞掉，仍返回字段
    assert out["memory_archived"] is False
    assert out["memory_preferences_added"] == 0


# ---------- live: 真实 ingest + recall ----------

@pytest.mark.live
def test_live_archive_and_recall(tmp_path, monkeypatch):
    """端到端：归档 → 检索同一个 query 应命中。"""
    import src.memory as mem
    import src.rag as rag

    chroma_dir = tmp_path / "chroma"
    monkeypatch.setattr(rag, "CHROMA_DIR", chroma_dir)
    mem._reset_memory_client_cache()

    archived = mem.archive_episodic({
        "query": "BGE-M3 中文检索",
        "final_report": "# BGE-M3\n中文嵌入质量好...",
        "quality_score": {"overall": 8.0},
        "research_results": [],
        "iteration_count": 0,
    })
    assert archived["id"]

    hits = mem.recall_episodic("中文嵌入", top_k=2)
    assert hits
    assert "BGE-M3" in hits[0]["summary"]
