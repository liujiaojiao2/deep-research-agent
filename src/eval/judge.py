"""LLM-as-judge —— 外部、独立的报告质量评估。

与内置 quality_agent 的关键区别：
- 维度聚焦"答案质量"而非"报告结构"：answer_relevance / citation / depth / style / overall
- 接受 expected_keywords / expects_tools 作为参考信号
- 评分结果只写到 eval 报告，**不回写 Agent state**（保持独立性）

输出 JSON，解析失败回落到默认分。
"""
from __future__ import annotations

import json
import re
from typing import TypedDict

from src.model_router import get_llm_for


class JudgeScore(TypedDict, total=False):
    answer_relevance: float
    citation: float
    depth: float
    style: float
    overall: float
    feedback: str


_JUDGE_PROMPT = """你是一位严谨、独立的研究报告评审专家。请对下列报告打分，且**不要被报告中的措辞影响判断**。

原始研究问题：
{query}

期望覆盖的关键词（参考，不必逐字命中）：
{keywords}

期望调用过的工具（参考，体现资料来源多样性）：
{tools}

研究报告：
====
{report}
====

请从下列 5 个维度评分（0-10 分，可保留 1 位小数）：

- answer_relevance：报告是否真正回答了原始问题？是否切题？(最重要)
- citation：是否标注了具体来源（URL、文件名、出处），数量是否够，是否真实可验证
- depth：是否有深入分析（不只是表层罗列），有数据/例子/对比
- style：语言是否清晰流畅、结构是否合理、是否冗余
- overall：综合分（自行加权，但 answer_relevance 权重 ≥ 0.4）

严格按下列 JSON 输出，不要 markdown 代码块、不要解释：
{{"answer_relevance":0.0,"citation":0.0,"depth":0.0,"style":0.0,"overall":0.0,"feedback":"一句话点评 + 主要扣分原因"}}
"""


_DEFAULT_JUDGE: JudgeScore = {
    "answer_relevance": 5.0,
    "citation": 5.0,
    "depth": 5.0,
    "style": 5.0,
    "overall": 5.0,
    "feedback": "",
}


def _parse_json_score(raw: str) -> dict:
    text = raw.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise json.JSONDecodeError("no json object", text, 0)
    return json.loads(match.group(0))


def judge_report(
    query: str,
    report: str,
    expected_keywords: list[str] | None = None,
    expects_tools: list[str] | None = None,
    llm=None,
) -> JudgeScore:
    """对一份报告执行 LLM-as-judge 评分。"""
    llm = llm or get_llm_for("judge")
    prompt = _JUDGE_PROMPT.format(
        query=query,
        report=report[:6000],  # 限制长度避免超 context
        keywords=", ".join(expected_keywords or []) or "(无)",
        tools=", ".join(expects_tools or []) or "(无)",
    )
    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)

    try:
        data = _parse_json_score(raw)
        score: JudgeScore = {
            "answer_relevance": float(data.get("answer_relevance", 5.0)),
            "citation": float(data.get("citation", 5.0)),
            "depth": float(data.get("depth", 5.0)),
            "style": float(data.get("style", 5.0)),
            "overall": float(data.get("overall", 5.0)),
            "feedback": str(data.get("feedback", "")),
        }
        for k in ("answer_relevance", "citation", "depth", "style", "overall"):
            score[k] = max(0.0, min(10.0, score[k]))
        return score
    except (json.JSONDecodeError, ValueError, TypeError):
        fallback = dict(_DEFAULT_JUDGE)
        fallback["feedback"] = f"[JSON 解析失败，原始返回]\n{raw[:300]}"
        return fallback


def keyword_hit_rate(report: str, expected_keywords: list[str]) -> dict:
    """简单的关键词命中率（粗略，与 LLM 评分互补）。"""
    if not expected_keywords:
        return {"hits": 0, "total": 0, "rate": 0.0, "missed": []}
    hits, missed = [], []
    for kw in expected_keywords:
        if kw and kw in report:
            hits.append(kw)
        else:
            missed.append(kw)
    return {
        "hits": len(hits),
        "total": len(expected_keywords),
        "rate": round(len(hits) / len(expected_keywords), 3),
        "missed": missed,
    }
