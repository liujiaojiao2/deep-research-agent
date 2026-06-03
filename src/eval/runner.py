"""Eval 单题运行器：跑一次 Agent + 收集 trajectory + 调 judge。"""
from __future__ import annotations

import time
from typing import TypedDict

from src.eval.judge import JudgeScore, judge_report, keyword_hit_rate
from src.graph import build_main_graph
from src.state import SupervisorState


class EvalItem(TypedDict, total=False):
    id: str
    query: str
    category: str
    expects_tools: list[str]
    expected_keywords: list[str]
    rationale: str


class EvalResult(TypedDict, total=False):
    id: str
    query: str
    category: str
    elapsed_sec: float
    iteration_count: int
    tools_used: list[str]
    quality_score: dict
    judge_score: JudgeScore
    keyword_hits: dict
    final_report: str
    error: str


def _initial_state(query: str, max_iter: int) -> SupervisorState:
    return {
        "query": query,
        "research_brief": "",
        "research_results": [],
        "draft_report": "",
        "final_report": "",
        "quality_score": {},
        "red_team_feedback": "",
        "iteration_count": 0,
        "max_iterations": max_iter,
        "next_agent": "",
        "is_complete": False,
        "messages": [],
    }


def _extract_tools_from_state(state: dict) -> list[str]:
    """从 research_results 的 source 字段里解析 ReAct 用了哪些工具。"""
    used: list[str] = []
    for r in state.get("research_results", []):
        src = r.get("source", "") or ""
        # 形如 react_agent(tools=local_knowledge_search,web_search)
        if "tools=" in src:
            tail = src.split("tools=", 1)[1].rstrip(")")
            used.extend(t.strip() for t in tail.split(",") if t.strip())
        elif src and src not in used:
            used.append(src)
    return used


def run_eval_item(item: EvalItem, max_iter: int = 1, recursion_limit: int = 50) -> EvalResult:
    """跑单题 eval：Agent + judge + keyword 命中率。"""
    start = time.monotonic()
    try:
        graph = build_main_graph(interactive=False)
        state = _initial_state(item["query"], max_iter)
        final_state = state
        for event in graph.stream(state, config={"recursion_limit": recursion_limit}):
            for _node, update in event.items():
                final_state = {**final_state, **(update or {})}
    except Exception as e:
        return {
            "id": item["id"],
            "query": item["query"],
            "category": item.get("category", ""),
            "elapsed_sec": round(time.monotonic() - start, 1),
            "error": f"{type(e).__name__}: {e}",
        }

    report = final_state.get("final_report") or ""
    elapsed = round(time.monotonic() - start, 1)
    tools_used = _extract_tools_from_state(final_state)

    judge_score = judge_report(
        query=item["query"],
        report=report,
        expected_keywords=item.get("expected_keywords"),
        expects_tools=item.get("expects_tools"),
    )
    keyword_hits = keyword_hit_rate(report, item.get("expected_keywords", []))

    return {
        "id": item["id"],
        "query": item["query"],
        "category": item.get("category", ""),
        "elapsed_sec": elapsed,
        "iteration_count": int(final_state.get("iteration_count", 0)),
        "tools_used": tools_used,
        "quality_score": final_state.get("quality_score") or {},
        "judge_score": judge_score,
        "keyword_hits": keyword_hits,
        "final_report": report,
    }
