"""Researcher Agent —— 确定性链路实现。

输入：state.research_brief（Draft Agent 生成）
流程：
  1. 让 LLM 从 brief 抽 2-4 个检索关键词（JSON 数组）
  2. 对每个关键词调 web_search
  3. 用 compress_research 把所有结果压缩成一段摘要
输出：追加到 state.research_results
"""
from __future__ import annotations

import json
import re
from typing import List

from src.model_router import get_llm_for
from src.state import ResearchResult, SupervisorState
from src.tools.compress_tool import compress_research
from src.tools.search_tool import run_web_search


_QUERY_EXTRACT_PROMPT = """从下面的研究简报中提炼出 2-4 条最适合在搜索引擎里输入的检索关键词。

要求：
- 每条关键词独立、具体、可直接搜索（5-15 个字）
- 覆盖不同子问题，不要近似重复
- 只输出 JSON 数组字符串，例如 ["关键词1", "关键词2"]
- 不要任何其他解释

研究简报：
{brief}
"""


def _extract_queries(llm, brief: str) -> List[str]:
    """从 brief 中拿出 2-4 个搜索 query。任何解析失败都退回到 brief 整段。"""
    if not brief.strip():
        return []
    prompt = _QUERY_EXTRACT_PROMPT.format(brief=brief)
    resp = llm.invoke(prompt)
    text = (resp.content if hasattr(resp, "content") else str(resp)).strip()

    # 容错 1：模型可能套了 ```json ... ``` 代码块
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        queries = json.loads(text)
        if isinstance(queries, list):
            cleaned = [str(q).strip() for q in queries if str(q).strip()]
            return cleaned[:4]
    except json.JSONDecodeError:
        pass
    # 容错 2：解析失败时直接拿 brief 第一行兜底
    return [brief.strip().splitlines()[0][:80]]


def researcher_node(
    state: SupervisorState,
    llm=None,
    search_fn=run_web_search,
    max_results_per_query: int = 3,
) -> dict:
    """对 brief 做多查询搜索 + 压缩，追加到 research_results。

    search_fn 参数允许测试注入假搜索；默认走真实 DDG。
    """
    llm = llm or get_llm_for("research")
    brief = state.get("research_brief") or state.get("query", "")

    queries = _extract_queries(llm, brief)
    if not queries:
        queries = [state.get("query", "")[:80]]

    raw_results: list[dict] = []
    for q in queries:
        try:
            raw_results.extend(search_fn(q, max_results=max_results_per_query))
        except Exception as e:  # 网络抖动不应中断流程
            raw_results.append(
                {"title": "search_error", "content": f"查询 {q} 出错: {e}", "url": ""}
            )

    compressed = compress_research(raw_results, llm=llm)

    new_entry: ResearchResult = {
        "query": " | ".join(queries),
        "content": compressed,
        "source": "web_search",
    }
    prev = list(state.get("research_results", []))
    prev.append(new_entry)
    return {"research_results": prev}
