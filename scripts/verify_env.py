"""Phase 0 黑箱探测脚本：验证 DeepSeek API + DuckDuckGo 搜索连通性。

按"黑箱测试"原则：给定最小输入，观察是否返回预期形状的输出。
任何一项失败立即报告偏差，禁止静默继续。
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

# 让脚本能从仓库根目录直接 uv run 运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402

console = Console()


def check_deepseek() -> bool:
    console.print("[bold]1) DeepSeek API 连通性[/bold]")
    try:
        from src.config import get_llm, settings

        if not settings.DEEPSEEK_API_KEY:
            console.print("  [red]❌ DEEPSEEK_API_KEY 未在 .env 中设置[/red]")
            return False

        llm = get_llm()
        resp = llm.invoke("用一句话回答：你是谁？")
        content = (resp.content or "").strip()
        if not content:
            console.print("  [red]❌ DeepSeek 返回空内容[/red]")
            return False

        console.print(f"  [green]✅ DeepSeek OK[/green]  →  {content[:60]}")
        return True
    except Exception:
        console.print("  [red]❌ DeepSeek 调用异常：[/red]")
        traceback.print_exc()
        return False


def check_duckduckgo() -> bool:
    console.print("[bold]2) DuckDuckGo 搜索连通性[/bold]")
    try:
        from langchain_community.tools import DuckDuckGoSearchResults

        tool = DuckDuckGoSearchResults(num_results=3, output_format="list")
        results = tool.invoke("LangGraph tutorial")
        if not results or len(results) == 0:
            console.print("  [red]❌ DuckDuckGo 返回空结果[/red]")
            return False
        sample = results[0]
        snippet = (sample.get("snippet") or sample.get("link") or "")[:80]
        console.print(f"  [green]✅ DuckDuckGo OK[/green]  →  共 {len(results)} 条，首条：{snippet}")
        return True
    except Exception:
        console.print("  [red]❌ DuckDuckGo 调用异常：[/red]")
        traceback.print_exc()
        return False


def main() -> int:
    console.rule("[bold cyan]DeepResearch Agent · Phase 0 环境验证[/bold cyan]")
    ok = []
    ok.append(check_deepseek())
    ok.append(check_duckduckgo())
    console.rule()
    if all(ok):
        console.print("[bold green]🎉 全部通过，Phase 0 验收成功[/bold green]")
        return 0
    console.print("[bold red]存在失败项，请按上方报错处理后再次运行本脚本。[/bold red]")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
