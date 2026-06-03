"""Quality Evaluation Agent —— 多维 JSON 评分。

输出 5 项浮点分数（0-10） + feedback 文本。任何解析失败都回退到默认分 5.0，
但保留原始返回到 feedback 字段，方便排查。

Phase 7.4 起：开启 Self-Consistency（QUALITY_EVAL_SAMPLES > 1）时，
跑 N 次评分取每维度中位数，降低单次 LLM 评分的随机波动。
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict

from src.model_router import get_llm_for
from src.multi_sample import sample_json_scores
from src.state import QualityScore, SupervisorState


SCORE_FIELDS = ["accuracy", "completeness", "logic", "citation", "overall"]


def _samples_count() -> int:
    try:
        return max(1, int(os.getenv("QUALITY_EVAL_SAMPLES", "1")))
    except ValueError:
        return 1


_QUALITY_PROMPT = """你是一个专业的内容质量评估专家。

待评估对象：针对问题 “{query}” 生成的研究报告。
Red Team 反馈（可作为参考，但请独立打分）：
{red_team_feedback}

报告内容：
====
{draft}
====

请从以下 5 个维度打分（0-10，可保留 1 位小数），并给出改进建议：

- accuracy：事实是否正确、是否有幻觉
- completeness：是否全面覆盖问题各方面
- logic：推理是否严密、结构是否清晰
- citation：来源是否权威、引用是否充分
- overall：综合分（自行加权）

严格按下列 JSON 输出，不要 markdown 代码块、不要解释：
{{"accuracy": 0.0, "completeness": 0.0, "logic": 0.0, "citation": 0.0, "overall": 0.0, "feedback": "改进建议"}}
"""


_DEFAULT_SCORE: QualityScore = {
    "accuracy": 5.0,
    "completeness": 5.0,
    "logic": 5.0,
    "citation": 5.0,
    "overall": 5.0,
    "feedback": "",
}


def _parse_score(text: str) -> Dict:
    """从 LLM 返回里抽出 JSON。处理常见的代码块/前后噪声。"""
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise json.JSONDecodeError("no json object", text, 0)
    return json.loads(match.group(0))


def quality_eval_node(state: SupervisorState, llm=None) -> dict:
    llm = llm or get_llm_for("quality")
    prompt = _QUALITY_PROMPT.format(
        query=state.get("query", ""),
        red_team_feedback=state.get("red_team_feedback", "(无)"),
        draft=state.get("draft_report", ""),
    )

    n_samples = _samples_count()
    if n_samples > 1:
        # Self-Consistency 路径：N 次采样取中位数
        result = sample_json_scores(
            prompt=prompt,
            score_fields=SCORE_FIELDS,
            llm=llm,
            n_samples=n_samples,
            default_score=5.0,
            text_field="feedback",
        )
        agg = result["aggregated"]
        # 写入主 score
        score: QualityScore = {k: float(agg.get(k, 5.0)) for k in SCORE_FIELDS}
        score["feedback"] = str(agg.get("feedback", ""))
        # 把方差记到 feedback 末尾，便于运维观察评分稳定性
        var = result["variance_per_field"].get("overall", 0.0)
        if var > 0:
            score["feedback"] = f"{score['feedback']}\n[self-consistency: n={n_samples}, var(overall)={var}]"
        return {"quality_score": score}

    # 单次路径（旧版行为，默认）
    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)

    try:
        data = _parse_score(raw)
        score: QualityScore = {
            "accuracy": float(data.get("accuracy", 5.0)),
            "completeness": float(data.get("completeness", 5.0)),
            "logic": float(data.get("logic", 5.0)),
            "citation": float(data.get("citation", 5.0)),
            "overall": float(data.get("overall", 5.0)),
            "feedback": str(data.get("feedback", "")),
        }
        for k in SCORE_FIELDS:
            score[k] = max(0.0, min(10.0, score[k]))
        return {"quality_score": score}
    except (json.JSONDecodeError, ValueError, TypeError):
        fallback = dict(_DEFAULT_SCORE)
        fallback["feedback"] = f"[JSON 解析失败，原始返回]\n{raw}"
        return {"quality_score": fallback}
