"""Final Report Agent —— 终稿润色与 Executive Summary 生成。

仅在 quality 达标或迭代次数耗尽时由 supervisor 路由进入。
"""
from __future__ import annotations

from src.model_router import get_llm_for
from src.state import SupervisorState


_FINAL_PROMPT = """你是专业的技术写作编辑。请对下面这份研究报告进行最终润色。

报告主题：{query}

迭代次数：{iterations}
最终质量评分：accuracy={accuracy} / completeness={completeness} / logic={logic} / citation={citation} / overall={overall}

原始报告：
====
{draft}
====

润色要求：
1. 在最前面加一段 200-300 字的 **执行摘要 (Executive Summary)**
2. 优化语言表达，使其更专业流畅
3. 保留全部原有引用与来源 URL，不要删
4. 确保结构规范（# 一级标题 / ## 二级标题 / 段落 / 引用）
5. 结论部分要明确、有力

直接输出最终版 Markdown，不要解释。
"""


def final_report_node(state: SupervisorState, llm=None) -> dict:
    llm = llm or get_llm_for("final")
    score = state.get("quality_score") or {}
    prompt = _FINAL_PROMPT.format(
        query=state.get("query", ""),
        iterations=state.get("iteration_count", 0),
        accuracy=score.get("accuracy", "N/A"),
        completeness=score.get("completeness", "N/A"),
        logic=score.get("logic", "N/A"),
        citation=score.get("citation", "N/A"),
        overall=score.get("overall", "N/A"),
        draft=state.get("draft_report", ""),
    )
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return {"final_report": content, "is_complete": True}
