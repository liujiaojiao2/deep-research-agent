"""DuckDuckGo 网页搜索工具。

向上层暴露 LangChain @tool 装饰过的 `web_search`，供 ReAct Agent 调用；
同时提供函数式接口 `run_web_search`，便于在节点里直接调用、便于单元测试 mock。

返回结构统一：{title, content, url}
"""
from __future__ import annotations

import logging
from typing import List

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def run_web_search(query: str, max_results: int = 5) -> List[dict]:
    """直接调用 DDG，返回结构化结果列表。失败时返回空列表，不抛异常。"""
    try:
        from ddgs import DDGS
    except ImportError as e:
        raise ImportError("缺少 ddgs 包，请运行 `uv add ddgs`") from e

    results: List[dict] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": r.get("title", ""),
                    "content": r.get("body", ""),
                    "url": r.get("href", ""),
                }
            )
    logger.debug("web_search(%r) -> %d results", query, len(results))
    return results


@tool
def web_search(query: str, max_results: int = 5) -> List[dict]:
    """通用互联网搜索（DuckDuckGo），返回标题/摘要/链接列表。

    适用场景：
    - 时效性强的内容：新闻、近期发布、产品动态、技术博客
    - 通用资讯：当 wikipedia 太学术、arxiv 太学术、又需要广泛信息时
    - 维基百科或 ArXiv 上找不到的小众主题

    参数：
    - query：搜索关键词（中文/英文均可），3-15 字效果最佳
    - max_results：返回结果数，默认 5

    返回：[{title, content, url}] 列表
    """
    return run_web_search(query=query, max_results=max_results)
