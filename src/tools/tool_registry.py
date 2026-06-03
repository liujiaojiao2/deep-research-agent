"""ReAct 工具注册中心。

每个工具的 docstring 就是 LLM 决策依据 —— 写清楚"什么时候用"比"怎么调用"更重要。
ReAct Agent 通过 docstring 自主选择工具，所以这里的描述质量直接决定智能体表现。
"""
from __future__ import annotations

import datetime as _dt
from typing import List

import os

from langchain_core.tools import tool

from src.tools.memory_tool import recall_episodic_memory
from src.tools.rag_tool import local_knowledge_search
from src.tools.search_tool import web_search  # 已有的 DDG 工具


@tool
def wikipedia_search(query: str, lang: str = "zh") -> str:
    """查询维基百科以获取权威的百科知识。

    适用场景：
    - 需要查找概念定义、历史事件、人物简介、地理信息等"事实性百科知识"
    - 比通用 web 搜索更权威，但内容更新慢；新闻类用 web_search

    参数：
    - query：维基百科条目标题或关键词（例如 "DeepSeek"、"强化学习"）
    - lang：语言代码，"zh" 中文 或 "en" 英文，默认 "zh"

    返回：条目摘要（前 2000 字符），失败时返回错误说明。
    """
    try:
        import wikipedia

        wikipedia.set_lang(lang)
        try:
            return wikipedia.summary(query, sentences=10)[:2000]
        except wikipedia.DisambiguationError as e:
            return f"该词条有多个含义，建议改用更具体的关键词。候选: {e.options[:5]}"
        except wikipedia.PageError:
            # 中文没查到，自动尝试英文
            if lang == "zh":
                wikipedia.set_lang("en")
                try:
                    return wikipedia.summary(query, sentences=10)[:2000]
                except Exception:
                    pass
            return f"维基百科未找到 '{query}' 的条目。"
    except Exception as e:
        return f"wikipedia_search 调用失败: {e}"


@tool
def arxiv_search(query: str, max_results: int = 3) -> List[dict]:
    """查询 ArXiv 学术论文库。

    适用场景：
    - 研究问题涉及"最新论文、技术细节、算法原理"
    - 用于补充权威学术来源；不适合搜新闻或通用知识

    参数：
    - query：论文主题关键词（英文效果更好，例如 "GRPO reinforcement learning"）
    - max_results：返回论文数量，默认 3，最大建议 5

    返回：[{title, authors, summary, url, published}]
    """
    try:
        import arxiv

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
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
        return results or [{"error": f"arxiv 未找到 '{query}' 相关论文"}]
    except Exception as e:
        return [{"error": f"arxiv_search 调用失败: {e}"}]


@tool
def python_calculator(expression: str) -> str:
    """执行 Python 数学表达式，用于数值计算与简单数据处理。

    适用场景：
    - 需要精确计算（百分比、增长率、单位换算、统计聚合等）
    - LLM 自己心算容易错的场景；不要用于字符串处理或长流程逻辑

    参数：
    - expression：Python 表达式（例如 "1.5 * 2 ** 10"、"sum([1,2,3])"、"(95-87)/87 * 100"）

    返回：表达式结果的字符串。失败返回错误说明。
    """
    try:
        from langchain_experimental.utilities import PythonREPL

        repl = PythonREPL()
        # 用 print 包一下，确保结果输出到 stdout
        result = repl.run(f"print({expression})")
        return result.strip() or "(无输出)"
    except Exception as e:
        return f"python_calculator 计算失败: {e}"


@tool
def get_current_datetime(timezone: str = "Asia/Shanghai") -> str:
    """获取当前日期与时间。

    适用场景：
    - 需要时效性判断时（如"最近一年的进展"、"今年发布"）
    - 在生成报告里写明数据截止时间

    参数：
    - timezone：IANA 时区名，默认 "Asia/Shanghai"

    返回：ISO 8601 格式时间字符串。
    """
    try:
        try:
            from zoneinfo import ZoneInfo

            now = _dt.datetime.now(ZoneInfo(timezone))
        except Exception:
            now = _dt.datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    except Exception as e:
        return f"get_current_datetime 失败: {e}"


def get_all_tools() -> list:
    """ReAct researcher 默认装配的全套工具。

    顺序约定：local_knowledge_search 永远在第一位（最高优先级）；
    其它工具相对顺序稳定，便于测试断言。
    可通过 ENABLE_RAG=false 关闭本地知识检索（仅外部工具）。
    """
    tools = []
    if os.getenv("ENABLE_RAG", "true").lower() != "false":
        tools.append(local_knowledge_search)
    if os.getenv("ENABLE_MEMORY", "true").lower() != "false":
        tools.append(recall_episodic_memory)
    tools.extend([
        web_search,
        wikipedia_search,
        arxiv_search,
        python_calculator,
        get_current_datetime,
    ])
    return tools
