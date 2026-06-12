"""Revision Node —— 自进化闭环的修订环节。

输入：state.draft_report + red_team_feedback + quality_score.feedback + research_results
流程：
  1. LLM 决策是否需要补充搜索（输出 JSON 数组，最多取 2 条）
  2. 若有，调用 web_search → compress，追加到 research_results
  3. LLM 基于"原稿 + 反馈 + 全部资料"重写报告
  4. 重置 quality_score / red_team_feedback；iteration_count += 1
"""
from __future__ import annotations

import json
import re
from typing import List

from src.model_router import get_llm_for
from src.state import SupervisorState
from src.tools.compress_tool import compress_research
from src.tools.search_tool import run_web_search


MAX_SUPPLEMENT_QUERIES = 2


_SUPPLEMENT_PROMPT = """根据下面的批评反馈，判断是否需要新增搜索来补充资料。

Red Team 反馈：
{red_team_feedback}

Quality 评估反馈：
{quality_feedback}

如果需要补充搜索，输出 JSON 数组：["关键词1", "关键词2"]（最多 {limit} 条）。
如果不需要，输出 []。
不要输出其他任何内容。
"""


_REVISION_PROMPT = """请基于下面的反馈和资料，修订研究报告。

原始问题：{query}

需要解决的问题：
{red_team_feedback}

{quality_feedback}

补充与原有研究资料：
{research}

原始报告：
====
{draft}
====

输出修订后的完整报告（Markdown 格式，结构保持：摘要 / 背景 / 主体 / 结论）。
不要解释你做了什么修改，直接输出新报告全文。
"""


def _extract_supplement_queries(llm, red_fb: str, q_fb: str) -> List[str]:
    prompt = _SUPPLEMENT_PROMPT.format(
        red_team_feedback=red_fb or "(无)",
        quality_feedback=q_fb or "(无)",
        limit=MAX_SUPPLEMENT_QUERIES,
    )
    resp = llm.invoke(prompt)
    text = (resp.content if hasattr(resp, "content") else str(resp)).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()][:MAX_SUPPLEMENT_QUERIES]
    except json.JSONDecodeError:
        pass
    return []


def revision_node(
    state: SupervisorState,
    llm=None,
    search_fn=run_web_search,
    max_results_per_query: int = 3,
) -> dict:
    llm = llm or get_llm_for("revision")

    red_fb = state.get("red_team_feedback", "")
    q_fb = (state.get("quality_score") or {}).get("feedback", "")
    research_results = list(state.get("research_results", []))

    # 1. 决策是否补搜索
    supplement_queries = _extract_supplement_queries(llm, red_fb, q_fb)

    # 2. 补搜索（失败不中断流程）
    if supplement_queries:
        raw: list[dict] = []
        tool_outputs: list[dict] = []
        for q in supplement_queries:
            try:
                batch = search_fn(q, max_results=max_results_per_query)
                raw.extend(batch)
                total_len = sum(len(r.get("content", "")) for r in batch)
                tool_outputs.append({
                    "tool": "web_search",
                    "query": q,
                    "result_count": len(batch),
                    "result_total_chars": total_len,
                    "snippet": (batch[0].get("content", "") if batch else "")[:200],
                })
            except Exception as e:
                raw.append({"title": "search_error", "content": f"{q}: {e}", "url": ""})
                tool_outputs.append({
                    "tool": "web_search",
                    "query": q,
                    "result_count": 0,
                    "result_total_chars": 0,
                    "snippet": f"error: {e}",
                })
        if raw:
            compressed = compress_research(raw, llm=llm)
            # 提取原始来源
            seen_urls: set[str] = set()
            raw_sources: list[dict] = []
            for r in raw:
                url = (r.get("url") or "").strip()
                title = (r.get("title") or "").strip()
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    raw_sources.append({"title": title, "url": url})
            research_results.append({
                "query": " | ".join(supplement_queries),
                "content": compressed,
                "source": "supplement",
                "sources": raw_sources,
                "tool_outputs": tool_outputs,
            })

    # 3. 重写报告
    research_text = "\n\n---\n\n".join(r.get("content", "") for r in research_results) or "（无资料）"
    revise_prompt = _REVISION_PROMPT.format(
        query=state.get("query", ""),
        red_team_feedback=red_fb or "(无)",
        quality_feedback=q_fb or "(无)",
        research=research_text,
        draft=state.get("draft_report", ""),
    )
    resp = llm.invoke(revise_prompt)
    new_draft = resp.content if hasattr(resp, "content") else str(resp)

    # 4. 重置评分 + 推进迭代计数
    return {
        "draft_report": new_draft,
        "research_results": research_results,
        "quality_score": {},          # 触发 supervisor 重新走 quality_eval
        "red_team_feedback": "",      # 防止下一轮被同样的反馈污染
        "iteration_count": int(state.get("iteration_count", 0)) + 1,
    }
