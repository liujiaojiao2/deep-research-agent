"""DeepResearch Agent 主图组装。

节点：
    supervisor → 决策下一步
    brief_writer → 生成研究简报
    researcher → 多查询搜索 + 压缩
    draft_writer → 写初稿
    quality_eval → 多维评分
    red_team → 对抗审查
    revision → 补搜索 + 重写（红队后必走）
    final_report → 终稿润色

边：
    entry → supervisor
    supervisor → (条件路由) → {brief_writer, researcher, draft_writer, quality_eval, red_team, final_report}
    brief_writer / researcher / draft_writer / quality_eval → supervisor
    red_team → revision → supervisor
    final_report → END
"""
from __future__ import annotations

import os

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from src.agents import (
    final_report_node,
    quality_eval_node,
    react_researcher_node,
    red_team_node,
    researcher_node,
    revision_node,
    route_to_next,
    supervisor_node,
    write_draft_report,
    write_research_brief,
)
from src.agents.evolution_agent import evolution_log_node
from src.agents.human_review_agent import human_review_node
from src.agents.skill_agent import (
    _format_skill_injection,
    match_skills,
    skill_library_node,
)
from src.agents.memory_archive_agent import memory_archive_node
from src.agents.rewoo_planner_agent import rewoo_planner_node
from src.agents.rewoo_worker_agent import rewoo_worker_node
from src.state import SupervisorState


def _rewoo_researcher_node(state: SupervisorState) -> dict:
    """ReWOO 复合节点：planner（1 次 LLM）→ worker（0 次 LLM）。

    对 graph 透明，等价于一个 researcher 节点；但 LLM 调用降到 1 次，
    比 ReAct 的 N+1 次省 N 次。

    Phase 8.2: 注入 Memento-Skills 匹配的技能 SOP。
    """
    # 匹配技能 + 注入到 state.research_brief（planner 会用到）
    skill_hint = ""
    try:
        matched = match_skills(state.get("query", ""), top_k=2)
        skill_hint = _format_skill_injection(matched)
    except Exception:
        pass
    if skill_hint:
        brief = (state.get("research_brief") or "") + "\n" + skill_hint
        state = {**state, "research_brief": brief}

    plan_update = rewoo_planner_node(state)
    merged = {**state, **plan_update}
    worker_update = rewoo_worker_node(merged)
    result = {**plan_update, **worker_update}
    if skill_hint:
        result["skill_injection"] = skill_hint[:200]
    return result


def _select_researcher():
    """按 RESEARCHER_MODE 切换 simple / react / rewoo 实现。

    - simple : 旧版顺序搜索（已废弃但保留）
    - react  : LLM 自主选工具（默认）
    - rewoo  : 一次性规划 + 纯执行（省 token）
    """
    mode = os.getenv("RESEARCHER_MODE", "react").lower()
    if mode == "simple":
        return researcher_node
    if mode == "rewoo":
        return _rewoo_researcher_node
    return react_researcher_node  # 默认 react


def build_main_graph(interactive: bool = False):
    """主图组装。

    interactive=True 时：
      · quality_eval → human_review（暂停等用户决策）→ supervisor
      · 同时注入 InMemorySaver；调用方必须传 thread_id
    interactive=False 时：原图（quality_eval 直接回 supervisor）
    """
    g = StateGraph(SupervisorState)

    # 节点注册（命名必须与 supervisor 的 next_agent 字符串完全一致）
    g.add_node("supervisor", supervisor_node)
    g.add_node("brief_writer", write_research_brief)
    g.add_node("researcher", _select_researcher())
    g.add_node("draft_writer", write_draft_report)
    g.add_node("quality_eval", quality_eval_node)
    g.add_node("red_team", red_team_node)
    g.add_node("revision", revision_node)
    g.add_node("final_report", final_report_node)
    g.add_node("memory_archive", memory_archive_node)
    g.add_node("evolution_log", evolution_log_node)
    g.add_node("skill_library", skill_library_node)
    if interactive:
        g.add_node("human_review", human_review_node)

    # 入口
    g.set_entry_point("supervisor")

    # supervisor → 条件分支
    g.add_conditional_edges(
        "supervisor",
        route_to_next,
        {
            "brief_writer": "brief_writer",
            "researcher": "researcher",
            "draft_writer": "draft_writer",
            "quality_eval": "quality_eval",
            "red_team": "red_team",
            "final_report": "final_report",
        },
    )

    # quality_eval 之后的回边：interactive 模式经 human_review
    for node in ("brief_writer", "researcher", "draft_writer"):
        g.add_edge(node, "supervisor")
    if interactive:
        g.add_edge("quality_eval", "human_review")
        g.add_edge("human_review", "supervisor")
    else:
        g.add_edge("quality_eval", "supervisor")

    # red_team 完成必须进入 revision（不交给 supervisor 再决策）
    g.add_edge("red_team", "revision")
    g.add_edge("revision", "supervisor")

    # final_report → memory_archive → evolution_log → END
    g.add_edge("final_report", "memory_archive")
    g.add_edge("memory_archive", "evolution_log")
    g.add_edge("evolution_log", "skill_library")
    g.add_edge("skill_library", END)

    if interactive:
        return g.compile(checkpointer=InMemorySaver())
    return g.compile()
