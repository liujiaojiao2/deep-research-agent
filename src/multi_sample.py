"""Multi-Sample / Multi-Persona 通用工具。

两个核心能力：

1. sample_json_scores: 同一 prompt 跑 N 次，对 JSON 字段做中位数聚合
   - 适用：quality_eval 评分稳定性
   - 取中位数而非均值：抗"离群单次评分"（5 次评分 8/8/8/8/2 → median=8）

2. sample_multi_persona: 用 N 个不同 system prompt 评同一对象，再 aggregate
   - 适用：red_team 多角度审查
   - 第一阶段：每个 persona 独立产出反馈
   - 第二阶段：让 LLM aggregate 成一份综合反馈
"""
from __future__ import annotations

import json
import logging
import re
import statistics
from typing import Any

logger = logging.getLogger(__name__)


# ---------- helpers ----------

def _parse_json_obj(raw: str) -> dict:
    """从 LLM 输出抽 JSON 对象，容忍前后噪声/代码块。"""
    text = raw.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise json.JSONDecodeError("no json object", text, 0)
    return json.loads(m.group(0))


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(round(statistics.pvariance(values), 4))


# ---------- 1. Self-Consistency for JSON-scored tasks ----------

def sample_json_scores(
    prompt: str,
    score_fields: list[str],
    llm,
    n_samples: int = 3,
    default_score: float = 5.0,
    text_field: str | None = "feedback",
) -> dict:
    """对同 prompt 跑 n_samples 次，取各 score_fields 的中位数。

    - 任何一次解析失败 → 该次 fallback 到 default_score（不抛错）
    - 返回结构含 aggregated scores + 单次原始记录 + 方差，便于可观测
    """
    samples: list[dict] = []
    for i in range(max(1, n_samples)):
        try:
            resp = llm.invoke(prompt)
            raw = resp.content if hasattr(resp, "content") else str(resp)
            data = _parse_json_obj(raw)
            samples.append({"_idx": i, "_raw": raw[:300], "_ok": True, **data})
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as e:
            logger.warning("sample_json_scores sample %d failed: %s", i, e)
            fallback = {k: default_score for k in score_fields}
            if text_field:
                fallback[text_field] = f"[parse failed: {e}]"
            samples.append({"_idx": i, "_ok": False, **fallback})

    aggregated: dict[str, Any] = {}
    variance_per_field: dict[str, float] = {}
    for field in score_fields:
        vals: list[float] = []
        for s in samples:
            v = s.get(field)
            if isinstance(v, (int, float)):
                vals.append(max(0.0, min(10.0, float(v))))
        if not vals:
            vals = [default_score]
        aggregated[field] = round(_median(vals), 2)
        variance_per_field[field] = _variance(vals)

    if text_field:
        # 取第一个成功 sample 的 feedback；都失败时取兜底
        first_ok = next((s for s in samples if s.get("_ok") and s.get(text_field)), None)
        aggregated[text_field] = (
            str(first_ok.get(text_field))[:500] if first_ok else "[all samples failed]"
        )

    return {
        "aggregated": aggregated,
        "samples": samples,
        "variance_per_field": variance_per_field,
        "n_samples": len(samples),
    }


# ---------- 2. Multi-Persona Debate ----------

def sample_multi_persona(
    personas: list[dict],
    target_prompt_template: str,
    llm,
    aggregator_prompt_template: str | None = None,
) -> dict:
    """让 N 个不同 persona 独立评论同一对象，再 aggregate 为综合反馈。

    - personas: [{"name": "事实核查师", "role": "..."}]，每个 persona 独立调 LLM
    - target_prompt_template: 必须含 {persona_role} 占位
    - aggregator_prompt_template: 必须含 {persona_views} 占位；None 表示拼接所有 view
    """
    views: list[dict] = []
    for p in personas:
        prompt = target_prompt_template.format(
            persona_name=p.get("name", ""),
            persona_role=p.get("role", ""),
        )
        try:
            resp = llm.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            views.append({"name": p.get("name", ""), "view": content, "_ok": True})
        except Exception as e:
            logger.warning("persona %s failed: %s", p.get("name"), e)
            views.append({"name": p.get("name", ""), "view": f"[error: {e}]", "_ok": False})

    if aggregator_prompt_template is None:
        # 简单拼接
        merged = "\n\n".join(f"### {v['name']} 视角\n{v['view']}" for v in views)
        return {"views": views, "aggregated": merged}

    persona_views_str = "\n\n".join(
        f"### {v['name']}\n{v['view']}" for v in views
    )
    agg_prompt = aggregator_prompt_template.format(persona_views=persona_views_str)
    try:
        resp = llm.invoke(agg_prompt)
        agg = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        agg = f"[aggregation failed: {e}]\n\n{persona_views_str}"
    return {"views": views, "aggregated": agg}
