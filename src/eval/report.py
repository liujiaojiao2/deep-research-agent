"""Eval 报告生成：把批量结果汇总成 markdown。"""
from __future__ import annotations

from typing import Iterable


def render_markdown(results: Iterable[dict], config_summary: str = "") -> str:
    results = list(results)

    # 总体统计
    total = len(results)
    failed = sum(1 for r in results if r.get("error"))
    valid = [r for r in results if not r.get("error")]
    avg_j = lambda key: round(sum((r.get("judge_score") or {}).get(key, 0) for r in valid) / max(len(valid), 1), 2)  # noqa: E731
    avg_q = lambda key: round(sum((r.get("quality_score") or {}).get(key, 0) for r in valid) / max(len(valid), 1), 2)  # noqa: E731

    # 内外评估偏差计算
    deltas = []
    for r in valid:
        qs = r.get("quality_score") or {}
        js = r.get("judge_score") or {}
        if qs.get("overall") and js.get("overall"):
            deltas.append(qs["overall"] - js["overall"])
    avg_delta = round(sum(deltas) / len(deltas), 2) if deltas else 0

    md = []
    md.append("# DeepResearch Agent · Eval 报告\n")
    if config_summary:
        md.append(f"**配置**：{config_summary}\n")
    md.append(f"**题数**：{total}（失败 {failed}）\n")

    md.append("\n## 综合统计 (Judge 外评估)\n")
    md.append(
        "| answer_relevance | citation | depth | style | overall |\n"
        "|---|---|---|---|---|\n"
        f"| {avg_j('answer_relevance')} | {avg_j('citation')} | {avg_j('depth')} | {avg_j('style')} | {avg_j('overall')} |\n"
    )

    md.append("\n## 内外评估偏差分析\n")
    md.append(f"**平均偏差 (quality_overall - judge_overall)**: {avg_delta} 分\n")
    if avg_delta > 0:
        md.append(f"\n内评估（quality_eval）平均偏高 **{avg_delta}** 分，存在系统性自评偏差。\n")
    elif avg_delta < 0:
        md.append(f"\n内评估（quality_eval）平均偏低 **{abs(avg_delta)}** 分，可能存在过度自我批评。\n")
    else:
        md.append("\n内外评估无明显偏差。\n")
    md.append(
        "\n| ID | quality_overall | judge_overall | delta | 判断 |\n"
        "|---|---|---|---|---|\n"
    )
    for r in valid:
        qs = r.get("quality_score") or {}
        js = r.get("judge_score") or {}
        qo = qs.get("overall", 0)
        jo = js.get("overall", 0)
        d = round(qo - jo, 2)
        flag = "⚠️ 偏高" if d > 1.0 else ("✅ 一致" if abs(d) <= 1.0 else "🔻 偏低")
        md.append(f"| {r.get('id')} | {qo} | {jo} | {d} | {flag} |\n")
    md.append(f"\n> 注：|delta| ≤ 1.0 视为一致。正值 = 自评偏高，负值 = 自评偏低。\n")

    md.append("\n## 各题明细\n")
    md.append(
        "| ID | 类别 | judge overall | answer_rel | citation | quality overall | delta | 关键词命中 | 工具 | 用时(s) |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    for r in results:
        j = r.get("judge_score") or {}
        qs = r.get("quality_score") or {}
        kw = r.get("keyword_hits") or {}
        tools = ",".join(r.get("tools_used", [])) or "—"
        qo = qs.get("overall", 0)
        jo = j.get("overall", 0)
        d = round(qo - jo, 2) if qo and jo else "—"
        md.append(
            f"| {r.get('id')} | {r.get('category', '')} | "
            f"{jo or '—'} | {j.get('answer_relevance', '—')} | {j.get('citation', '—')} | "
            f"{qo or '—'} | {d} | "
            f"{kw.get('hits', 0)}/{kw.get('total', 0)} | {tools[:60]} | {r.get('elapsed_sec', '—')} |\n"
        )

    md.append("\n## 单题反馈\n")
    for r in results:
        md.append(f"\n### {r.get('id')} — {r.get('category', '')}\n")
        md.append(f"\n> {r.get('query', '')}\n")
        if r.get("error"):
            md.append(f"\n**错误**：`{r['error']}`\n")
            continue
        j = r.get("judge_score") or {}
        qs = r.get("quality_score") or {}
        kw = r.get("keyword_hits") or {}
        md.append(
            f"\n- **judge overall**: {j.get('overall')}  "
            f"(answer={j.get('answer_relevance')} / citation={j.get('citation')} / "
            f"depth={j.get('depth')} / style={j.get('style')})\n"
        )
        md.append(f"- **judge feedback**: {j.get('feedback', '')}\n")
        qo = qs.get("overall", 0)
        jo = j.get("overall", 0)
        delta_str = f" (偏差 {round(qo - jo, 2)})" if qo and jo else ""
        md.append(
            f"- **quality_eval overall**: {qo or '—'}{delta_str}  "
            f"(accuracy={qs.get('accuracy', '—')} / completeness={qs.get('completeness', '—')} / "
            f"logic={qs.get('logic', '—')} / citation={qs.get('citation', '—')})\n"
        )
        md.append(f"- **关键词命中**: {kw.get('hits', 0)}/{kw.get('total', 0)}  "
                  f"missed=[{', '.join(kw.get('missed', []))}]\n")
        md.append(f"- **tools_used**: {', '.join(r.get('tools_used', [])) or '—'}\n")
        md.append(f"- **iteration_count**: {r.get('iteration_count', 0)}  "
                  f"**用时**: {r.get('elapsed_sec', '—')} s\n")

    return "".join(md)
