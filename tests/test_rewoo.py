"""ReWOO planner + worker 单测：mock LLM、mock tool，覆盖核心路径。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.tools import tool

from src.agents.rewoo_planner_agent import (
    MAX_PLAN_STEPS,
    _default_plan,
    _parse_plan,
    _validate_and_clean,
    rewoo_planner_node,
)
from src.agents.rewoo_worker_agent import _format_tool_output, rewoo_worker_node


class _StubLLM:
    def __init__(self, payload):
        self.payload = payload
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return SimpleNamespace(content=self.payload)


# ---------- _parse_plan / _validate_and_clean ----------

def test_parse_plan_clean_json():
    out = _parse_plan('[{"step":1,"tool":"web_search","args":{"query":"x"}}]')
    assert out == [{"step": 1, "tool": "web_search", "args": {"query": "x"}}]


def test_parse_plan_with_noise():
    out = _parse_plan('好的:\n[{"step":1,"tool":"web_search","args":{"query":"x"}}]\nover')
    assert out[0]["tool"] == "web_search"


def test_parse_plan_raises_on_garbage():
    import json as _json

    with pytest.raises(_json.JSONDecodeError):
        _parse_plan("not json")


def test_validate_substitutes_unknown_tool():
    plan = [{"step": 1, "tool": "imaginary_tool", "args": {"query": "x"}}]
    out = _validate_and_clean(plan, fallback_query="fallback")
    assert out[0]["tool"] == "web_search"  # 我们项目里 web_search 一定存在


def test_validate_caps_at_max_steps():
    plan = [
        {"step": i, "tool": "web_search", "args": {"query": f"q{i}"}}
        for i in range(MAX_PLAN_STEPS + 5)
    ]
    out = _validate_and_clean(plan, fallback_query="x")
    assert len(out) == MAX_PLAN_STEPS


def test_validate_fills_missing_query():
    plan = [{"step": 1, "tool": "web_search", "args": {}}]
    out = _validate_and_clean(plan, fallback_query="fb")
    assert out[0]["args"]["query"] == "fb"


def test_validate_handles_non_dict_args():
    plan = [{"step": 1, "tool": "web_search", "args": "not a dict"}]
    out = _validate_and_clean(plan, fallback_query="fb")
    assert isinstance(out[0]["args"], dict)


def test_validate_drops_non_dict_step():
    plan = ["not a dict", {"step": 1, "tool": "web_search", "args": {"query": "ok"}}]
    out = _validate_and_clean(plan, fallback_query="fb")
    assert len(out) == 1


def test_default_plan_uses_query():
    plan = _default_plan("研究 GRPO 算法")
    assert len(plan) == 1
    assert plan[0]["tool"] == "web_search"


# ---------- rewoo_planner_node ----------

def test_planner_returns_valid_plan():
    llm = _StubLLM('[{"step":1,"tool":"wikipedia_search","args":{"query":"GRPO"}}]')
    out = rewoo_planner_node(
        {"query": "什么是 GRPO？", "research_brief": "查 GRPO 概念"},
        llm=llm,
    )
    assert "rewoo_plan" in out
    assert out["rewoo_plan"][0]["tool"] == "wikipedia_search"
    # planner 提示词应包含 brief
    assert "查 GRPO 概念" in llm.last_prompt


def test_planner_fallback_on_invalid_json():
    llm = _StubLLM("没有 JSON 输出")
    out = rewoo_planner_node({"query": "x", "research_brief": "b"}, llm=llm)
    plan = out["rewoo_plan"]
    assert len(plan) == 1
    assert plan[0]["tool"] == "web_search"


def test_planner_passes_brief_or_query_to_fallback():
    llm = _StubLLM("garbage")
    out = rewoo_planner_node({"query": "Q", "research_brief": ""}, llm=llm)
    assert out["rewoo_plan"][0]["args"]["query"] == "Q"


# ---------- rewoo_worker_node ----------

@tool
def _fake_search(query: str) -> list:
    """fake search tool"""
    return [{"title": f"T-{query}", "content": f"内容 {query}", "url": "http://e"}]


def test_worker_runs_plan_in_order():
    state = {
        "rewoo_plan": [
            {"step": 1, "thought": "查 X", "tool": "_fake_search", "args": {"query": "X"}},
            {"step": 2, "thought": "查 Y", "tool": "_fake_search", "args": {"query": "Y"}},
        ],
        "research_brief": "test",
        "research_results": [],
    }
    out = rewoo_worker_node(state, tools_override=[_fake_search])
    assert len(out["research_results"]) == 1
    src = out["research_results"][0]["source"]
    assert "_fake_search,_fake_search" in src
    assert out["rewoo_tokens_saved_estimate"] == 1


def test_worker_handles_empty_plan():
    out = rewoo_worker_node({"rewoo_plan": [], "research_results": [{"x": 1}]}, tools_override=[_fake_search])
    assert out["research_results"] == [{"x": 1}]


def test_worker_records_unknown_tool_error():
    state = {
        "rewoo_plan": [
            {"step": 1, "thought": "x", "tool": "no_such_tool", "args": {"query": "x"}},
        ],
        "research_brief": "test",
        "research_results": [],
    }
    out = rewoo_worker_node(state, tools_override=[_fake_search])
    content = out["research_results"][-1]["content"]
    assert "no_such_tool" in content


def test_worker_swallows_tool_exception():
    @tool
    def _boom(query: str) -> list:
        """always raises"""
        raise RuntimeError("network down")

    state = {
        "rewoo_plan": [
            {"step": 1, "thought": "x", "tool": "_boom", "args": {"query": "X"}},
        ],
        "research_brief": "test",
        "research_results": [],
    }
    out = rewoo_worker_node(state, tools_override=[_boom])
    content = out["research_results"][-1]["content"]
    assert "network down" in content


def test_worker_preserves_previous_results():
    state = {
        "rewoo_plan": [
            {"step": 1, "thought": "x", "tool": "_fake_search", "args": {"query": "x"}},
        ],
        "research_brief": "test",
        "research_results": [{"query": "old", "content": "old c", "source": "react"}],
    }
    out = rewoo_worker_node(state, tools_override=[_fake_search])
    assert len(out["research_results"]) == 2
    assert out["research_results"][0]["content"] == "old c"


# ---------- _format_tool_output ----------

def test_format_str_passthrough():
    assert _format_tool_output("hello") == "hello"


def test_format_list_of_dicts():
    out = _format_tool_output([{"title": "T", "content": "C", "url": "U"}])
    assert "T" in out and "C" in out and "U" in out


def test_format_list_capped_at_5():
    items = [{"title": f"T{i}", "content": f"C{i}", "url": ""} for i in range(10)]
    out = _format_tool_output(items)
    assert "T0" in out and "T4" in out
    assert "T5" not in out  # 截断


# ---------- Phase 7.6: parallel worker ----------

def test_worker_parallel_path_records_workers(monkeypatch):
    monkeypatch.setenv("REWOO_PARALLEL_WORKERS", "3")
    state = {
        "rewoo_plan": [
            {"step": 1, "thought": "a", "tool": "_fake_search", "args": {"query": "A"}},
            {"step": 2, "thought": "b", "tool": "_fake_search", "args": {"query": "B"}},
            {"step": 3, "thought": "c", "tool": "_fake_search", "args": {"query": "C"}},
        ],
        "research_brief": "p",
        "research_results": [],
    }
    out = rewoo_worker_node(state, tools_override=[_fake_search])
    assert out["rewoo_parallel_workers"] == 3
    # 结果按 step 顺序聚合
    content = out["research_results"][-1]["content"]
    assert content.index("step 1") < content.index("step 2") < content.index("step 3")


def test_worker_sequential_when_workers_le_1(monkeypatch):
    monkeypatch.setenv("REWOO_PARALLEL_WORKERS", "1")
    state = {
        "rewoo_plan": [
            {"step": 1, "thought": "a", "tool": "_fake_search", "args": {"query": "A"}},
            {"step": 2, "thought": "b", "tool": "_fake_search", "args": {"query": "B"}},
        ],
        "research_brief": "p",
        "research_results": [],
    }
    out = rewoo_worker_node(state, tools_override=[_fake_search])
    assert out["rewoo_parallel_workers"] == 1


def test_worker_parallel_speeds_up(monkeypatch):
    """3 步各 0.2s 顺序应 ~0.6s，并行应 < 0.3s。"""
    import time as _time
    monkeypatch.setenv("REWOO_PARALLEL_WORKERS", "5")

    @tool
    def _slow(query: str) -> list:
        """slow fake tool"""
        _time.sleep(0.2)
        return [{"title": query, "content": query, "url": ""}]

    state = {
        "rewoo_plan": [
            {"step": i, "thought": f"s{i}", "tool": "_slow", "args": {"query": f"q{i}"}}
            for i in range(1, 4)
        ],
        "research_brief": "p",
        "research_results": [],
    }
    out = rewoo_worker_node(state, tools_override=[_slow])
    # 并行应显著低于顺序总耗时
    assert out["rewoo_elapsed_seconds"] < 0.45


def test_worker_parallel_isolates_exceptions(monkeypatch):
    monkeypatch.setenv("REWOO_PARALLEL_WORKERS", "3")

    @tool
    def _boom(query: str) -> list:
        """always raises"""
        raise RuntimeError("net")

    state = {
        "rewoo_plan": [
            {"step": 1, "thought": "ok", "tool": "_fake_search", "args": {"query": "A"}},
            {"step": 2, "thought": "boom", "tool": "_boom", "args": {"query": "B"}},
            {"step": 3, "thought": "ok", "tool": "_fake_search", "args": {"query": "C"}},
        ],
        "research_brief": "p",
        "research_results": [],
    }
    out = rewoo_worker_node(state, tools_override=[_fake_search, _boom])
    content = out["research_results"][-1]["content"]
    assert "step 1" in content and "step 3" in content
    assert "net" in content  # 异常被记录但不中断
