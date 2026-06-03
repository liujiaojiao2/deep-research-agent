"""批量跑 eval set 并出 markdown 报告。

用法：
    uv run python scripts/run_eval.py [--eval-file PATH] [--max-iter 1]

环境变量：
    RESEARCHER_MODE=react|simple   切换 researcher 实现
    ENABLE_RAG=true|false          切换 RAG 工具
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402

from src.eval.report import render_markdown  # noqa: E402
from src.eval.runner import run_eval_item  # noqa: E402

console = Console()


def _config_summary() -> str:
    return (
        f"RESEARCHER_MODE={os.getenv('RESEARCHER_MODE', 'react')} | "
        f"ENABLE_RAG={os.getenv('ENABLE_RAG', 'true')} | "
        f"MODEL={os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run eval set")
    parser.add_argument(
        "--eval-file",
        default=str(Path(__file__).resolve().parent.parent / "data" / "eval" / "questions.json"),
        help="评估题 JSON 文件",
    )
    parser.add_argument("--max-iter", type=int, default=1, help="每题最大自进化迭代数")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认 outputs/）")
    args = parser.parse_args()

    items = json.loads(Path(args.eval_file).read_text(encoding="utf-8"))
    console.rule(f"[bold cyan]🧪 Running eval set ({len(items)} 题)[/bold cyan]")
    console.print(f"[dim]{_config_summary()}[/dim]")

    results = []
    for idx, item in enumerate(items, 1):
        console.print(f"\n[bold yellow]({idx}/{len(items)}) [{item['id']}][/bold yellow] {item['query'][:60]}...")
        r = run_eval_item(item, max_iter=args.max_iter)
        results.append(r)
        if r.get("error"):
            console.print(f"  [red]❌ {r['error']}[/red]")
        else:
            j = r["judge_score"]
            kw = r["keyword_hits"]
            console.print(
                f"  [green]✅[/green] judge overall=[bold]{j.get('overall')}[/bold]  "
                f"answer={j.get('answer_relevance')}  citation={j.get('citation')}  "
                f"kw {kw['hits']}/{kw['total']}  tools={','.join(r.get('tools_used', [])) or '—'}  "
                f"用时 {r['elapsed_sec']}s"
            )

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"eval_{ts}.md"
    out_path.write_text(render_markdown(results, _config_summary()), encoding="utf-8")

    console.rule("[bold green]✅ Eval 完成[/bold green]")
    console.print(f"📄 报告保存至: {out_path}")


if __name__ == "__main__":
    main()
