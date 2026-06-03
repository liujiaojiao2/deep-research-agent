"""ReWOO Worker —— 按 planner 给的计划，纯执行工具调用。

**关键设计点**：本节点不调 LLM。每一步只做：
  1. 按 plan[i].tool 查 tool_registry
  2. 调 tool.invoke(plan[i].args)
  3. 把结果累积到 state.research_results

Phase 7.6 起：默认并行执行（ThreadPoolExecutor），通过环境变量
REWOO_PARALLEL_WORKERS 控制并发；0/1 表示顺序（旧行为）。

异常容错：单步失败不中断其它 step，记一条 error 进 results 并继续。
按 step 原始顺序聚合结果（并行不打乱报告里的步骤顺序）。
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src import security
from src.state import ResearchResult, SupervisorState
from src.tools.tool_registry import get_all_tools

logger = logging.getLogger(__name__)


def _parallel_workers() -> int:
    """0 / 1 = 顺序；>=2 = 并行。"""
    try:
        return max(0, int(os.getenv("REWOO_PARALLEL_WORKERS", "5")))
    except ValueError:
        return 5


def _format_tool_output(raw: Any) -> str:
    """把工具返回值标准化为字符串，便于后续 draft_writer 直接拼接。"""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        chunks = []
        for item in raw[:5]:  # 限制片段数避免上下文爆炸
            if isinstance(item, dict):
                title = item.get("title", "")
                content = item.get("content") or item.get("summary") or ""
                url = item.get("url", "")
                chunks.append(f"- {title}\n  {content}\n  来源: {url}".strip())
            else:
                chunks.append(str(item))
        return "\n\n".join(chunks)
    if isinstance(raw, dict):
        return str(raw)[:2000]
    return str(raw)[:2000]


def _execute_step(step: dict, tools_map: dict) -> ResearchResult:
    """跑单步：返回标准化的 result 记录。绝不抛异常（异常被吞为 error result）。"""
    tool_name = step.get("tool", "")
    args = step.get("args", {}) or {}
    thought = step.get("thought", "")
    tool = tools_map.get(tool_name)

    if tool is None:
        return {
            "query": str(args.get("query", "")),
            "content": f"(工具 {tool_name!r} 不存在，跳过)",
            "source": "rewoo_worker_error",
        }

    try:
        raw_output = tool.invoke(args)
        content = _format_tool_output(raw_output)
        # Phase 7.7: 用 untrusted 标记包裹外部内容
        content = security.wrap_untrusted_content(content, source=tool_name)
        source = f"rewoo_worker({tool_name})"
    except Exception as e:
        content = f"(工具 {tool_name} 调用异常: {type(e).__name__}: {e})"
        source = f"rewoo_worker_error({tool_name})"
        logger.warning("rewoo_worker step %s failed: %s", step.get("step"), e)

    return {
        "query": str(args.get("query", "")),
        "content": f"[step {step.get('step')}] {thought}\n\n{content}",
        "source": source,
    }


def rewoo_worker_node(state: SupervisorState, tools_override=None) -> dict:
    """按 plan 跑工具。并发由 REWOO_PARALLEL_WORKERS 控制。"""
    plan = state.get("rewoo_plan") or []
    if not plan:
        logger.warning("rewoo_worker: 收到空计划，跳过")
        return {"research_results": list(state.get("research_results") or [])}

    tools_map = {t.name: t for t in (tools_override or get_all_tools())}
    workers = _parallel_workers()

    start = time.monotonic()

    if workers >= 2 and len(plan) >= 2:
        # 并行路径：max_workers 限制并发数，按 step 原序聚合结果
        with ThreadPoolExecutor(max_workers=min(workers, len(plan))) as ex:
            new_results = list(ex.map(lambda s: _execute_step(s, tools_map), plan))
    else:
        # 顺序路径（旧行为）
        new_results = [_execute_step(step, tools_map) for step in plan]

    elapsed = round(time.monotonic() - start, 2)

    # 统计工具使用（按原 step 顺序，error 也算）
    tools_used: list[str] = []
    for step in plan:
        if step.get("tool"):
            tools_used.append(step["tool"])

    aggregated: ResearchResult = {
        "query": state.get("research_brief", "")[:200],
        "content": "\n\n---\n\n".join(r["content"] for r in new_results),
        "source": f"rewoo(tools={','.join(tools_used) or 'none'})",
    }
    prev = list(state.get("research_results") or [])
    prev.append(aggregated)

    return {
        "research_results": prev,
        "rewoo_tokens_saved_estimate": max(0, len(plan) - 1),
        "rewoo_elapsed_seconds": elapsed,
        "rewoo_parallel_workers": workers if workers >= 2 and len(plan) >= 2 else 1,
    }
