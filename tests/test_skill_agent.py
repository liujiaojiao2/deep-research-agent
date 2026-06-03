"""Memento-Skills 单测：提炼 / 匹配 / 注入 / 存储。"""
from __future__ import annotations

from types import SimpleNamespace

from src.agents.skill_agent import (
    _format_skill_injection,
    _parse_skill_json,
    extract_skill,
    skill_library_node,
)


class _StubLLM:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, prompt):
        return SimpleNamespace(content=self.payload)


# ---------- _parse_skill_json ----------

def test_parse_clean_json():
    data = _parse_skill_json('{"name":"算法对比","trigger_keywords":["对比","diff"],"steps_sop":"先wiki后arxiv"}')
    assert data["name"] == "算法对比"
    assert data["steps_sop"] == "先wiki后arxiv"


def test_parse_with_noise():
    data = _parse_skill_json('好的:\n{"name":"x","steps_sop":"y","trigger_keywords":[]}\nover')
    assert data["name"] == "x"


# ---------- extract_skill ----------

def test_extract_skill_from_high_score_run():
    llm = _StubLLM('{"name":"算法对比","trigger_keywords":["对比","差异"],'
                    '"steps_sop":"先wiki定义A和B -> arxiv找对比论文 -> 综合写报告"}')
    state = {
        "query": "对比 GRPO 和 PPO",
        "quality_score": {"overall": 8.5},
        "research_results": [
            {"source": "rewoo(tools=wikipedia_search,arxiv_search,web_search)"},
        ],
    }
    skill = extract_skill(state, llm)
    assert skill is not None
    assert skill["name"] == "算法对比"
    assert "对比" in skill["trigger_keywords"]


def test_extract_skill_skips_low_score():
    llm = _StubLLM("{}")
    state = {"query": "Q", "quality_score": {"overall": 6.0}, "research_results": []}
    assert extract_skill(state, llm) is None


def test_extract_skill_skips_when_llm_says_skip():
    llm = _StubLLM('{"skip": true}')
    state = {
        "query": "Q",
        "quality_score": {"overall": 8.5},
        "research_results": [{"source": "web_search"}],
    }
    assert extract_skill(state, llm) is None


# ---------- _format_skill_injection ----------

def test_format_skill_injection_empty():
    assert _format_skill_injection([]) == ""


def test_format_skill_injection_below_threshold():
    matched = [{"name": "x", "steps_sop": "y", "success_count": 1,
                "avg_score": 8.0, "similarity": 0.20}]
    assert _format_skill_injection(matched) == ""


def test_format_skill_injection_happy():
    matched = [{"name": "算法对比", "steps_sop": "先wiki后arxiv", "success_count": 3,
                "avg_score": 8.5, "similarity": 0.55}]
    hint = _format_skill_injection(matched)
    assert "算法对比" in hint
    assert "先wiki后arxiv" in hint
    assert "3次" in hint


# ---------- skill_library_node ----------

def test_skill_library_node_extracts_and_stores(monkeypatch):
    stored = {"called": False}

    def fake_store(skill):
        stored["called"] = True
        return True

    monkeypatch.setattr("src.agents.skill_agent.store_skill", fake_store)
    monkeypatch.setattr("src.agents.skill_agent.MIN_SKILL_SCORE", 7.0)
    llm = _StubLLM('{"name":"测试技能","trigger_keywords":["测试"],"steps_sop":"step1->step2"}')

    out = skill_library_node(
        {"query": "测试", "quality_score": {"overall": 8.5}},
        llm=llm,
    )
    assert out["skill_extracted"] is True
    assert stored["called"] is True


def test_skill_library_node_swallows_exception(monkeypatch):
    def boom(state, llm):
        raise RuntimeError("chromadb broken")

    monkeypatch.setattr("src.agents.skill_agent.extract_skill", boom)
    out = skill_library_node({"query": "Q", "quality_score": {"overall": 8.5}}, llm=_StubLLM("x"))
    assert out["skill_extracted"] is False
