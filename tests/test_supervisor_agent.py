"""Supervisor 决策表测试。参数化覆盖所有分支。"""
from __future__ import annotations

import pytest

from src.agents.supervisor_agent import route_to_next, supervisor_node


def _state(**overrides):
    base = {
        "query": "q",
        "research_brief": "",
        "research_results": [],
        "draft_report": "",
        "quality_score": {},
        "iteration_count": 0,
        "max_iterations": 3,
    }
    base.update(overrides)
    return base


def test_no_brief_routes_to_brief_writer():
    out = supervisor_node(_state())
    assert out["next_agent"] == "brief_writer"


def test_has_brief_no_research_routes_to_researcher():
    out = supervisor_node(_state(research_brief="brief here"))
    assert out["next_agent"] == "researcher"


def test_has_research_no_draft_routes_to_draft_writer():
    out = supervisor_node(_state(
        research_brief="b",
        research_results=[{"content": "x", "query": "q", "source": "web_search"}],
    ))
    assert out["next_agent"] == "draft_writer"


def test_has_draft_no_score_routes_to_quality_eval():
    out = supervisor_node(_state(
        research_brief="b",
        research_results=[{"content": "x", "query": "q", "source": "web_search"}],
        draft_report="draft",
    ))
    assert out["next_agent"] == "quality_eval"


def test_high_score_routes_to_final_report():
    out = supervisor_node(_state(
        research_brief="b",
        research_results=[{"content": "x", "query": "q", "source": "web_search"}],
        draft_report="draft",
        quality_score={"overall": 8.5},
    ))
    assert out["next_agent"] == "final_report"


def test_low_score_routes_to_red_team():
    out = supervisor_node(_state(
        research_brief="b",
        research_results=[{"content": "x", "query": "q", "source": "web_search"}],
        draft_report="draft",
        quality_score={"overall": 5.0},
        iteration_count=0,
    ))
    assert out["next_agent"] == "red_team"


def test_low_score_but_max_iter_reached_forces_final():
    out = supervisor_node(_state(
        research_brief="b",
        research_results=[{"content": "x", "query": "q", "source": "web_search"}],
        draft_report="draft",
        quality_score={"overall": 4.0},
        iteration_count=3,
        max_iterations=3,
    ))
    assert out["next_agent"] == "final_report"


@pytest.mark.parametrize("score,iters,expected", [
    (6.99, 0, "red_team"),
    (7.0, 0, "final_report"),
    (7.5, 2, "final_report"),
    (5.0, 5, "final_report"),  # 超过 max_iter 强制输出
])
def test_threshold_and_iter_combo(score, iters, expected):
    out = supervisor_node(_state(
        research_brief="b",
        research_results=[{"content": "x", "query": "q", "source": "web_search"}],
        draft_report="draft",
        quality_score={"overall": score},
        iteration_count=iters,
        max_iterations=3,
    ))
    assert out["next_agent"] == expected


def test_route_to_next_reads_next_agent_field():
    assert route_to_next({"next_agent": "researcher"}) == "researcher"


def test_supervisor_does_not_mutate_iteration_count():
    state = _state(iteration_count=2)
    out = supervisor_node(state)
    # supervisor 只写 next_agent，不再 ++iteration_count
    assert "iteration_count" not in out
