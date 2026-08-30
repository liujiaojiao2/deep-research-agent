"""DeepResearch Agent 全局状态定义。
定义状态真实的状态集合
所有 Agent 在同一个 SupervisorState 上读写，由 LangGraph 的 StateGraph 驱动。
新增字段时务必更新 main.py 里的 initial_state，避免 KeyError。
"""
from __future__ import annotations

from typing import Annotated, List, TypedDict

from langgraph.graph.message import add_messages


class ResearchResult(TypedDict, total=False):
    query: str
    content: str
    source: str
    sources: list[dict]  # 原始来源列表 [{title, url}]
    tool_outputs: list[dict]  # 工具返回摘要 [{tool, query, result_len, snippet}]


class QualityScore(TypedDict, total=False):
    accuracy: float
    completeness: float
    logic: float
    citation: float
    overall: float
    feedback: str


class SupervisorState(TypedDict, total=False):
    # 输入
    query: str

    # 研究过程
    research_brief: str
    research_results: List[ResearchResult]

    # 报告版本
    draft_report: str
    final_report: str

    # 质量控制
    quality_score: QualityScore
    red_team_feedback: str
    iteration_count: int
    max_iterations: int

    # 流程控制
    next_agent: str
    is_complete: bool

    # 消息历史（供 ReAct Agent 使用）
    messages: Annotated[list, add_messages]

    # Memory 可观测字段（memory_archive 节点写）
    memory_archived: bool
    memory_preferences_added: int

    # ReWOO Planner/Worker 共享字段
    rewoo_plan: List[dict]
    rewoo_tokens_saved_estimate: int
    rewoo_elapsed_seconds: float
    rewoo_parallel_workers: int

    # HarnessForge 联合进化字段
    evolution_recorded: bool
    evolution_strategy_hint: str

    # Memento-Skills 技能库字段
    skill_extracted: bool
    skill_injection: str
