"""Adaptive Auto-Harness —— 按任务类型自动检测退化 + 推荐配置调整。

HarnessForge (8.1): 记录策略
Memento-Skills (8.2): 提炼可复用技能
Adaptive Auto-Harness (8.4): 检测退化 → 自动建议配置变更

核心 API:
- analyze_degradation(): 按 query_type 分组分析分数趋势
- recommend_config(): 为退化类型生成配置建议
- adaptive_report(): 输出 markdown 报告

设计：纯函数，不修改 Agent 行为——仅产出建议报告。
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

DEGRADATION_THRESHOLD = float(os.getenv("ADAPTIVE_THRESHOLD", "0.8"))
# 当前分数 < 历史平均 * THRESHOLD → 判为退化


def _load_evolution_data() -> list[dict[str, Any]]:
    """从 ChromaDB evolution_log 加载所有策略记录。"""
    try:
        from src.agents.evolution_agent import _get_evo_collection

        coll = _get_evo_collection()
        if coll.count() == 0:
            return []
        result = coll.get(limit=200)
        metas = result.get("metadatas", [])
        return [m for m in metas if m]
    except Exception as e:
        logger.warning("load evolution data failed: %s", e)
        return []


def _by_query_type(records: list[dict]) -> dict[str, list[float]]:
    """按 query_type 分组，每组 = list[scores]。"""
    groups: dict[str, list[float]] = defaultdict(list)
    for r in records:
        qt = r.get("query_type", "未知")
        score = float(r.get("overall_score", 0.0))
        groups[qt].append(score)
    return dict(groups)


def analyze_degradation() -> dict[str, Any]:
    """分析各 query_type 的退化情况。

    Returns:
      {
        "groups": {type: {count, avg_score, recent_avg, degrading}},
        "degraded_types": [type1, type2],
        "healthy_types": [type3, ...]
      }
    """
    records = _load_evolution_data()
    if not records:
        return {"groups": {}, "degraded_types": [], "healthy_types": []}

    groups = _by_query_type(records)
    result: dict[str, Any] = {"groups": {}, "degraded_types": [], "healthy_types": []}

    for qt, scores in groups.items():
        if len(scores) < 2:
            # 不足 2 条记录，无法判断趋势
            result["groups"][qt] = {
                "count": len(scores), "avg_score": round(sum(scores) / len(scores), 2),
                "recent_avg": round(scores[-1], 2), "degrading": False,
            }
            continue

        avg_all = sum(scores) / len(scores)
        recent = scores[-1]
        degrading = recent < avg_all * DEGRADATION_THRESHOLD

        result["groups"][qt] = {
            "count": len(scores),
            "avg_score": round(avg_all, 2),
            "recent_avg": round(recent, 2),
            "degrading": degrading,
        }
        if degrading:
            result["degraded_types"].append(qt)
        else:
            result["healthy_types"].append(qt)

    return result


def recommend_config(degraded_type: str, current_config: dict | None = None) -> dict:
    """为退化类型生成配置建议。"""
    suggestions: list[str] = []
    config_patch: dict[str, str] = {}

    # 规则引擎（简单、可解释）
    if "对比" in degraded_type or "比较" in degraded_type:
        suggestions.append("对比类任务建议用 rewoo 模式 + arxiv_search 优先")
        config_patch["RESEARCHER_MODE"] = "rewoo"
        config_patch["TOOL_HINT"] = "优先 arxiv_search, wikipedia_search"
    elif "概念" in degraded_type or "定义" in degraded_type:
        suggestions.append("概念类任务建议用 react 模式 + wikipedia_search 优先")
        config_patch["RESEARCHER_MODE"] = "react"
        config_patch["TOOL_HINT"] = "优先 wikipedia_search, local_knowledge_search"
    elif "实操" in degraded_type or "应用" in degraded_type:
        suggestions.append("实操类任务建议用 rewoo 模式 + web_search 优先")
        config_patch["RESEARCHER_MODE"] = "rewoo"
    else:
        suggestions.append(f"类型 [{degraded_type}] 无专用规则，建议提升 quality_eval 采样数到 3")
        config_patch["QUALITY_EVAL_SAMPLES"] = "3"

    suggestions.append("如分数仍不回升，建议检查该类型的 local_knowledge_search 命中率")
    return {"type": degraded_type, "suggestions": suggestions, "config_patch": config_patch}


def adaptive_report() -> str:
    """生成完整的 markdown 自适应报告。"""
    analysis = analyze_degradation()
    md = ["# Adaptive Auto-Harness 报告\n"]
    md.append(f"**退化阈值**: 当前分 < 历史平均 × {DEGRADATION_THRESHOLD}\n")

    if not analysis["groups"]:
        md.append("暂无 evolution 数据。至少跑 2 次不同 query 后才有分析。\n")
        return "".join(md)

    # 总览表
    md.append("\n## 各类型健康度\n")
    md.append("| query_type | count | avg | recent | status |\n|---|---|---|---|---|\n")
    for qt, g in sorted(analysis["groups"].items()):
        status = "🔴 退化" if g["degrading"] else "🟢 健康"
        md.append(f"| {qt} | {g['count']} | {g['avg_score']} | {g['recent_avg']} | {status} |\n")

    # 退化类型 + 建议
    if analysis["degraded_types"]:
        md.append("\n## 退化类型 & 建议\n")
        for qt in analysis["degraded_types"]:
            rec = recommend_config(qt)
            md.append(f"\n### {qt}\n")
            for s in rec["suggestions"]:
                md.append(f"- {s}\n")
            md.append("\n推荐 env patch:\n```bash\n")
            for k, v in rec["config_patch"].items():
                md.append(f"export {k}={v}\n")
            md.append("```\n")

    md.append(f"\n## 健康类型 ({len(analysis['healthy_types'])})\n")
    for qt in analysis["healthy_types"]:
        md.append(f"- {qt}\n")

    return "".join(md)
