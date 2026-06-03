"""Draft Agent —— 包含两个节点：

- `write_research_brief`: 根据用户 query 生成研究简报（子问题 + 关键词 + 报告框架）
- `write_draft_report`: 根据 research_results 撰写初稿

两个节点都作为主图（graph.py）的独立节点挂用，受 supervisor 调度。

Phase 7.1 起 `write_research_brief` 会自动注入用户偏好（来自 PreferenceMemory）。
"""
from __future__ import annotations

import logging
import os

from src.model_router import get_llm_for
from src.state import SupervisorState

logger = logging.getLogger(__name__)


_BRIEF_PROMPT = """你是一个专业研究员。请就以下问题生成一份研究简报（Research Brief）。

用户问题：{query}
{preferences_block}
请输出（自由文本即可，不要 markdown 代码块包裹）：
1. 核心研究问题拆解：列出 3-5 个可独立搜索的子问题
2. 检索关键词：每个子问题给出 2-3 个搜索关键词
3. 预期报告结构框架：摘要 / 背景 / 主体（建议 3-5 节）/ 结论
"""


def _load_preferences_block(query: str) -> str:
    """从 PreferenceMemory 取相关偏好，拼成 prompt 片段；失败/空库时返回空字符串。"""
    if os.getenv("ENABLE_MEMORY", "true").lower() == "false":
        return ""
    try:
        from src.memory import get_active_preferences

        prefs = get_active_preferences(query=query, top_k=3)
        if not prefs:
            return ""
        bullets = "\n".join(f"- {p}" for p in prefs)
        return f"\n你已知的用户偏好（请据此调整 brief 风格与重点）：\n{bullets}\n"
    except Exception as e:
        logger.warning("load preferences failed: %s", e)
        return ""


_DRAFT_PROMPT = """你是一个专业研究员，请基于以下研究资料撰写一份完整的研究报告。

研究问题：{query}

研究简报：
{brief}

研究资料（已压缩）：
{research}

要求：
- 结构：摘要 / 背景 / 主体（分节）/ 结论
- 字数 1500-3000 字
- 每个事实尽量标注来源 URL（资料中已含来源）
- 客观中立，避免编造未出现的信息
- 输出 Markdown 格式
"""


def write_research_brief(state: SupervisorState, llm=None) -> dict:
    """节点：把 query 转成研究简报。会自动注入用户偏好 + HarnessForge 进化策略。"""
    llm = llm or get_llm_for("brief")
    # Phase 8.1: 检索历史最优策略
    strategy_hint = state.get("evolution_strategy_hint", "") or _load_evolution_hint(state["query"])
    prompt = _BRIEF_PROMPT.format(
        query=state["query"],
        preferences_block=_load_preferences_block(state["query"]),
    )
    if strategy_hint:
        prompt = prompt.replace("请输出", strategy_hint + "\n\n请输出")
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return {"research_brief": content}


def _load_evolution_hint(query: str) -> str:
    """从 evolution log 检索最优策略（不依赖 state，兜底路径）。"""
    if os.getenv("ENABLE_EVOLUTION", "true").lower() == "false":
        return ""
    try:
        from src.agents.evolution_agent import _format_strategy_hint, recall_evolution

        strategies = recall_evolution(query, top_k=2)
        return _format_strategy_hint(strategies)
    except Exception:
        return ""


def write_draft_report(state: SupervisorState, llm=None) -> dict:
    """节点：基于研究资料撰写初稿。"""
    llm = llm or get_llm_for("draft")
    research = "\n\n---\n\n".join(
        r.get("content", "") for r in state.get("research_results", [])
    ) or "（无可用资料）"
    prompt = _DRAFT_PROMPT.format(
        query=state["query"],
        brief=state.get("research_brief", "(无简报)"),
        research=research,
    )
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return {"draft_report": content}
