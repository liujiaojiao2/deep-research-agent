"""ReWOO Planner —— 一次性输出全部研究步骤 + 工具指定。

ReWOO 核心思想：把"决定调哪个工具"从执行阶段提到规划阶段，
让执行阶段变成"纯工具调用"，省掉 N-1 次 LLM 调用。

输出格式（严格 JSON）：
    [
      {"step": 1, "thought": "...", "tool": "wikipedia_search", "args": {"query": "GRPO"}},
      {"step": 2, "thought": "...", "tool": "arxiv_search",     "args": {"query": "GRPO RL"}},
      ...
    ]

容错策略：
- LLM 输出非 JSON  → 回退到 default plan（单步调 web_search）
- 工具名不存在     → 替换为 web_search + 记录 warning
- args 缺失字段    → 用 brief 兜底
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.model_router import get_llm_for
from src.state import SupervisorState
from src.tools.tool_registry import get_all_tools

logger = logging.getLogger(__name__)


MAX_PLAN_STEPS = 6  # ReWOO 计划步数上限，超出会被截断


def _available_tool_descriptions() -> str:
    """把当前注册的工具名+描述凝练成 prompt 上下文。"""
    lines = []
    for t in get_all_tools():
        # 取 docstring 前两行作描述（去掉空行）
        desc = (t.description or "").strip().split("\n")
        head = next((line.strip() for line in desc if line.strip()), "")
        lines.append(f"- {t.name}: {head}")
    return "\n".join(lines)


_PLANNER_PROMPT = """你是一个 ReWOO 风格的研究规划器。给定研究简报，请一次性输出
完整的研究计划 —— 后续执行阶段不会再有 LLM 介入，全靠你这一次规划。

研究问题：{query}

研究简报：
{brief}

可用工具：
{tools}

请输出严格的 JSON 数组，每个元素是一个步骤，最多 {max_steps} 步：

[
  {{
    "step": 1,
    "thought": "为什么要做这一步（一句话）",
    "tool": "工具名（必须是上面列表里的）",
    "args": {{"query": "查询关键词"}}
  }},
  ...
]

要求：
1. 覆盖简报中所有子问题，每个子问题至少 1 个步骤
2. 优先使用 local_knowledge_search（如有相关），再用 wikipedia/arxiv/web_search
3. 同一工具不要重复调用相同 args
4. 工具名必须严格匹配；args 必须是 dict，至少含 "query" 字段
5. 只输出 JSON 数组，不要解释、不要 markdown 代码块包裹
"""


def _parse_plan(raw: str) -> list[dict[str, Any]]:
    """从 LLM 输出里提取 JSON 数组。容错对各种代码块/前后噪声。"""
    text = raw.strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise json.JSONDecodeError("no json array found", text, 0)
    data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise json.JSONDecodeError("expected list", text, 0)
    return data


def _validate_and_clean(plan: list[dict], fallback_query: str) -> list[dict]:
    """逐条校验：工具名存在、args 是 dict、query 非空。无效项做替换或丢弃。"""
    tool_names = {t.name for t in get_all_tools()}
    fallback_tool = "web_search" if "web_search" in tool_names else next(iter(tool_names))

    cleaned = []
    for idx, step in enumerate(plan[:MAX_PLAN_STEPS], start=1):
        if not isinstance(step, dict):
            continue
        tool = step.get("tool", "")
        args = step.get("args", {})
        if not isinstance(args, dict):
            args = {"query": str(args)}
        if tool not in tool_names:
            logger.warning("planner: 未知工具 %r → 替换为 %s", tool, fallback_tool)
            tool = fallback_tool
        if "query" not in args or not str(args.get("query", "")).strip():
            args["query"] = fallback_query
        cleaned.append({
            "step": idx,
            "thought": str(step.get("thought", ""))[:200],
            "tool": tool,
            "args": args,
        })
    return cleaned


def _default_plan(query: str) -> list[dict]:
    """LLM 完全跑挂时的兜底单步计划。"""
    return [{"step": 1, "thought": "fallback: 直接搜索", "tool": "web_search", "args": {"query": query[:80]}}]


def rewoo_planner_node(state: SupervisorState, llm=None) -> dict:
    """生成 ReWOO 计划，写入 state.rewoo_plan。"""
    llm = llm or get_llm_for("planner")
    query = state.get("query", "")
    brief = state.get("research_brief") or query

    prompt = _PLANNER_PROMPT.format(
        query=query,
        brief=brief,
        tools=_available_tool_descriptions(),
        max_steps=MAX_PLAN_STEPS,
    )
    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)

    try:
        plan = _parse_plan(raw)
        plan = _validate_and_clean(plan, fallback_query=query[:80] or "research")
        if not plan:
            plan = _default_plan(query)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning("planner JSON 解析失败 (%s) → 用默认单步计划", e)
        plan = _default_plan(query)

    return {"rewoo_plan": plan}
