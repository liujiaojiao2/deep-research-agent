"""Supervisor —— 纯函数状态机，无 LLM 调用，每次只决定 next_agent。

决策树（按顺序短路）：
  1. 无 research_brief             → brief_writer
  2. 无 research_results           → researcher
  3. 无 draft_report               → draft_writer
  4. 无 quality_score              → quality_eval
  5. quality.overall >= 阈值       → final_report
  6. iteration_count >= max_iter   → final_report（兜底）
  7. 其余                          → red_team（触发自进化）
"""
from __future__ import annotations

import os

from src.state import SupervisorState

# 实测 quality_eval 存在自评偏差（偏高），
# 7.0 时部分本应走 red_team 的问题被错误路由到 final_report。
# 推荐生产环境设为 8.5 以确保自进化循环充分介入。
QUALITY_THRESHOLD = float(os.getenv("QUALITY_THRESHOLD", "7.0"))
DEFAULT_MAX_ITERATIONS = 3

AGENTS = (
    "brief_writer",
    "researcher",
    "draft_writer",
    "quality_eval",
    "red_team",
    "final_report",
)


def supervisor_node(state: SupervisorState) -> dict:
    """根据当前 state 决定下一步交给哪个 Agent。只写入 next_agent。"""
    has_brief = bool(state.get("research_brief"))
    has_research = bool(state.get("research_results"))
    has_draft = bool(state.get("draft_report"))
    quality = state.get("quality_score") or {}
    has_score = bool(quality)
    overall = float(quality.get("overall", 0.0)) if has_score else 0.0
    iteration = int(state.get("iteration_count", 0))
    max_iter = int(state.get("max_iterations", DEFAULT_MAX_ITERATIONS))

    if not has_brief:
        next_agent = "brief_writer"
    elif not has_research:
        next_agent = "researcher"
    elif not has_draft:
        next_agent = "draft_writer"
    elif not has_score:
        next_agent = "quality_eval"
    elif overall >= QUALITY_THRESHOLD:
        next_agent = "final_report"
    elif iteration >= max_iter:
        next_agent = "final_report"
    else:
        next_agent = "red_team"

    return {"next_agent": next_agent}


def route_to_next(state: SupervisorState) -> str:
    """给 add_conditional_edges 用的纯路由函数。"""
    return state["next_agent"]
