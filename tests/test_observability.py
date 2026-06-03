"""Tracer 单测：事件记录、信息提取、summary/markdown 输出。"""
from __future__ import annotations

from src.observability import Tracer


def test_tracer_records_basic_events():
    t = Tracer(query="Q", run_id="r1")
    t.record("supervisor", {"next_agent": "brief_writer"})
    t.record("brief_writer", {"research_brief": "x" * 100})
    s = t.summary()
    assert s["node_visits"]["supervisor"] == 1
    assert s["node_visits"]["brief_writer"] == 1
    assert s["events_count"] == 2


def test_tracer_extracts_tools_from_source():
    t = Tracer(query="Q", run_id="r1")
    t.record("researcher", {
        "research_results": [{"source": "rewoo(tools=web_search,wikipedia_search,arxiv_search)"}]
    })
    s = t.summary()
    assert s["tool_call_counts"]["web_search"] == 1
    assert s["tool_call_counts"]["wikipedia_search"] == 1
    assert s["tool_call_counts"]["arxiv_search"] == 1


def test_tracer_tracks_quality_trajectory():
    t = Tracer(query="Q", run_id="r1")
    t.record("quality_eval", {"quality_score": {"overall": 5.0, "feedback": "low"}})
    t.record("quality_eval", {"quality_score": {"overall": 8.5, "feedback": "good"}})
    s = t.summary()
    assert s["quality_trajectory"] == [5.0, 8.5]
    assert s["final_overall"] == 8.5


def test_tracer_record_error():
    t = Tracer(query="Q", run_id="r1")
    t.record_error("researcher", "network down")
    s = t.summary()
    assert s["events_count"] == 1
    assert t.events[0].kind == "error"


def test_tracer_to_markdown_contains_sections():
    t = Tracer(query="My question", run_id="r1")
    t.record("supervisor", {"next_agent": "brief_writer"})
    t.record("researcher", {"research_results": [{"source": "rewoo(tools=web_search)"}]})
    t.record("quality_eval", {"quality_score": {"overall": 8.0}})
    md = t.to_markdown()
    assert "Trace · r1" in md
    assert "My question" in md
    assert "Node visits" in md
    assert "Tool call counts" in md
    assert "web_search" in md
    assert "Quality trajectory" in md


def test_tracer_dump(tmp_path):
    t = Tracer(query="Q", run_id="r1")
    t.record("supervisor", {"next_agent": "x"})
    out = t.dump(tmp_path)
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("# Trace")


def test_tracer_ignores_unknown_node():
    t = Tracer(query="Q", run_id="r1")
    t.record("mystery_node", {"foo": "bar"})
    s = t.summary()
    assert s["node_visits"]["mystery_node"] == 1
