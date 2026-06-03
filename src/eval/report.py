"""Eval 报告生成：把批量结果汇总成 markdown。"""
from __future__ import annotations

from typing import Iterable


def render_markdown(results: Iterable[dict], config_summary: str = "") -> str:
    results = list(results)

    # 总体统计
    total = len(results)
    failed = sum(1 for r in results if r.get("error"))
    valid = [r for r in results if not r.get("error")]
    avg = lambda key: round(sum((r.get("judge_score") or {}).get(key, 0) for r in valid) / max(len(valid), 1), 2)  # noqa: E731

    md = []
    md.append("# DeepResearch Agent · Eval 报告\n")
    if config_summary:
        md.append(f"**配置**：{config_summary}\n")
    md.append(f"**题数**：{total}（失败 {failed}）\n")
    md.append("\n## 综合统计\n")
    md.append(
        "| answer_relevance | citation | depth | style | overall |\n"
        "|---|---|---|---|---|\n"
        f"| {avg('answer_relevance')} | {avg('citation')} | {avg('depth')} | {avg('style')} | {avg('overall')} |\n"
    )

    md.append("\n## 各题明细\n")
    md.append(
        "| ID | 类别 | overall | answer_rel | citation | 关键词命中 | 工具 | 用时(s) |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    for r in results:
        j = r.get("judge_score") or {}
        kw = r.get("keyword_hits") or {}
        tools = ",".join(r.get("tools_used", [])) or "—"
        md.append(
            f"| {r.get('id')} | {r.get('category', '')} | "
            f"{j.get('overall', '—')} | {j.get('answer_relevance', '—')} | {j.get('citation', '—')} | "
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
        kw = r.get("keyword_hits") or {}
        md.append(
            f"\n- **judge overall**: {j.get('overall')}  "
            f"(answer={j.get('answer_relevance')} / citation={j.get('citation')} / "
            f"depth={j.get('depth')} / style={j.get('style')})\n"
        )
        md.append(f"- **judge feedback**: {j.get('feedback', '')}\n")
        md.append(f"- **关键词命中**: {kw.get('hits', 0)}/{kw.get('total', 0)}  "
                  f"missed=[{', '.join(kw.get('missed', []))}]\n")
        md.append(f"- **tools_used**: {', '.join(r.get('tools_used', [])) or '—'}\n")
        md.append(f"- **iteration_count**: {r.get('iteration_count', 0)}  "
                  f"**用时**: {r.get('elapsed_sec', '—')} s\n")

    return "".join(md)
