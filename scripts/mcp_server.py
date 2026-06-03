"""DeepResearch Tools MCP Server —— 把本项目的 6 个工具暴露为 MCP 协议。

用途：让 Claude Desktop / Cursor / 其它 MCP 客户端能直接调用我们的工具集。

启动：
    uv run python scripts/mcp_server.py                 # 默认 stdio 传输
    uv run python scripts/mcp_server.py --transport sse # SSE 传输（HTTP）

Claude Desktop 接入示例：见 docs/claude_desktop_config.example.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from src.tools.search_tool import run_web_search  # noqa: E402

# ---------- 实例化 MCP server ----------

mcp = FastMCP(
    name="deep-research-tools",
    instructions=(
        "DeepResearch Agent 的工具集 MCP 服务：6 类工具供其它 LLM 应用调用，"
        "包括 web 搜索、维基百科、ArXiv 论文、本地知识库 RAG、记忆检索、Python 计算器。"
    ),
)


# ---------- 注册工具 ----------

@mcp.tool()
def web_search(query: str, max_results: int = 5) -> list:
    """通用互联网搜索（DuckDuckGo）。适合时效性强的内容、新闻、博客解读。

    Args:
        query: 搜索关键词（中英文均可）
        max_results: 返回结果数，默认 5

    Returns:
        [{title, content, url}] 列表
    """
    return run_web_search(query=query, max_results=max_results)


@mcp.tool()
def wikipedia_search(query: str, lang: str = "zh") -> str:
    """维基百科条目摘要。适合概念定义、历史背景、人物简介。

    Args:
        query: 维基百科条目名或关键词
        lang: 语言 "zh" / "en"，默认 "zh"

    Returns:
        条目摘要文本（前 2000 字符）
    """
    import wikipedia

    wikipedia.set_lang(lang)
    try:
        return wikipedia.summary(query, sentences=10)[:2000]
    except wikipedia.DisambiguationError as e:
        return f"该词条有多个含义。候选: {e.options[:5]}"
    except wikipedia.PageError:
        if lang == "zh":
            wikipedia.set_lang("en")
            try:
                return wikipedia.summary(query, sentences=10)[:2000]
            except Exception:
                pass
        return f"未找到 '{query}' 的条目"
    except Exception as e:
        return f"wikipedia 调用失败: {e}"


@mcp.tool()
def arxiv_search(query: str, max_results: int = 3) -> list:
    """ArXiv 学术论文检索。适合算法原理、论文细节、技术机制。

    Args:
        query: 论文主题关键词（英文效果更好）
        max_results: 返回论文数，默认 3

    Returns:
        [{title, authors, summary, url, published}]
    """
    import arxiv

    search = arxiv.Search(
        query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    for paper in search.results():
        results.append({
            "title": paper.title,
            "authors": ", ".join(a.name for a in paper.authors[:4]),
            "summary": paper.summary[:600],
            "url": paper.entry_id,
            "published": paper.published.strftime("%Y-%m-%d"),
        })
    return results or [{"error": f"未找到 '{query}' 相关论文"}]


@mcp.tool()
def local_knowledge_search(query: str, top_k: int = 3) -> list:
    """本地知识库（ChromaDB + BGE 嵌入）语义检索。

    适合查询团队内部文档、项目笔记、复盘记录等公网搜不到的私有信息。
    需要先运行 `scripts/ingest_knowledge.py` 把文档入库。

    Args:
        query: 检索关键词
        top_k: 返回片段数，默认 3

    Returns:
        [{content, source, similarity}] 列表；空库时返回 [{...empty}]
    """
    try:
        from src.rag import retrieve

        hits = retrieve(query=query, top_k=top_k)
        if not hits:
            return [{"content": "本地知识库为空（请先运行 ingest_knowledge.py 入库）", "source": "(empty)"}]
        return hits
    except Exception as e:
        return [{"content": f"调用异常: {type(e).__name__}: {e}", "source": "(error)"}]


@mcp.tool()
def recall_episodic_memory(query: str, top_k: int = 3) -> list:
    """跨会话研究记忆检索：查找过去做过的相似研究。

    Args:
        query: 检索关键词
        top_k: 返回片段数，默认 3

    Returns:
        [{summary, query, date, overall_score, similarity}]
    """
    try:
        from src.memory import recall_episodic

        hits = recall_episodic(query=query, top_k=top_k)
        return hits or [{"summary": "无过往研究记录", "similarity": 0.0}]
    except Exception as e:
        return [{"summary": f"memory 异常: {type(e).__name__}: {e}", "similarity": 0.0}]


@mcp.tool()
def python_calculator(expression: str) -> str:
    """精确数值计算。适合百分比、增长率、单位换算、统计聚合。

    Args:
        expression: Python 表达式，如 "(95-87)/87 * 100"

    Returns:
        计算结果字符串
    """
    try:
        from langchain_experimental.utilities import PythonREPL

        repl = PythonREPL()
        result = repl.run(f"print({expression})")
        return result.strip() or "(无输出)"
    except Exception as e:
        return f"计算失败: {e}"


# ---------- 元资源：把项目信息暴露为 MCP resource ----------

@mcp.resource("trace://latest")
def trace_latest() -> str:
    """返回最新一次 run 的 JSONL trace（如存在）。"""
    from pathlib import Path

    outputs = sorted(Path("outputs").glob("trace_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if outputs:
        return outputs[0].read_text(encoding="utf-8")[:5000]
    return "暂无 trace 文件"


@mcp.resource("project://meta")
def project_meta() -> str:
    """暴露项目元信息（如 README / 学习笔记摘要），便于客户端理解工具能力。"""
    return (
        "DeepResearch Agent — 多 Agent 研究系统中台\n"
        "暴露 6 类工具：web_search / wikipedia / arxiv / local_kb / memory / calculator\n"
        "GitHub：[填你自己的链接]\n"
    )


def _list_tools() -> list[str]:
    """便于测试：返回当前注册的工具名（同步 list）。"""
    import asyncio
    tools = asyncio.run(mcp.list_tools())
    return [t.name for t in tools]


# ---------- 入口 ----------

def main():
    parser = argparse.ArgumentParser(description="DeepResearch Tools MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP 传输协议（Claude Desktop 用 stdio）",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run("stdio")
    elif args.transport == "sse":
        mcp.run("sse")
    else:
        mcp.run("streamable-http")


if __name__ == "__main__":
    main()
