"""Adaptive Auto-Harness 分析脚本 —— 检测退化 + 推荐配置。

跑法: uv run python scripts/adaptive_analyze.py [--output REPORT.md]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from src.agents.adaptive_agent import adaptive_report

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Adaptive Auto-Harness analysis")
    parser.add_argument("--output", default=None, help="输出 markdown 报告路径（默认打印到终端）")
    args = parser.parse_args()

    console.rule("[bold cyan]🔧 Adaptive Auto-Harness 分析[/bold cyan]")

    report = adaptive_report()
    console.print(report)

    out_path = args.output or (
        Path(__file__).resolve().parent.parent / "outputs"
        / f"adaptive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(report, encoding="utf-8")
    console.rule(f"[green]✅ 报告保存至 {out_path}[/green]")


if __name__ == "__main__":
    main()
