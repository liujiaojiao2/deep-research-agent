"""多引擎网页搜索工具。

通过 SEARCH_PROVIDER 环境变量切换引擎：
    - duckduckgo (默认) : DuckDuckGo，零配置
    - google            : Google 搜索，需网络能访问 google.com
                          可选 GOOGLE_PROXY 代理

向上层暴露 LangChain @tool 装饰过的 `web_search`，供 ReAct Agent 调用；
同时提供函数式接口 `run_web_search`，便于在节点里直接调用、便于单元测试 mock。

返回结构统一：{title, content, url}
"""
from __future__ import annotations

import logging
import os
from typing import List

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "duckduckgo").lower()


# ---------- DuckDuckGo ----------

def _search_ddg(query: str, max_results: int = 5) -> List[dict]:
    try:
        from ddgs import DDGS
    except ImportError as e:
        raise ImportError("缺少 ddgs 包，请运行 `uv add ddgs`") from e

    results: List[dict] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "content": r.get("body", ""),
                "url": r.get("href", ""),
            })
    logger.debug("ddg(%r) -> %d results", query, len(results))
    return results


# ---------- Google (googlesearch-python) ----------

def _search_google(query: str, max_results: int = 5) -> List[dict]:
    try:
        from googlesearch import search
    except ImportError as e:
        raise ImportError("缺少 googlesearch-python 包，请运行 `pip install googlesearch-python`") from e

    proxy = os.getenv("GOOGLE_PROXY", "") or None
    results: List[dict] = []
    try:
        for r in search(
            query,
            num_results=max_results,
            advanced=True,
            lang="zh-CN",
            sleep_interval=1,
            proxy=proxy,
            timeout=10,
        ):
            results.append({
                "title": r.title or "",
                "content": r.description or "",
                "url": r.url or "",
            })
    except Exception as e:
        logger.warning("google search(%r) failed: %s", query, e)
        return []

    logger.debug("google(%r) -> %d results", query, len(results))
    return results


# ---------- 统一入口 ----------

def run_web_search(query: str, max_results: int = 5) -> List[dict]:
    """统一搜索入口，根据 SEARCH_PROVIDER 选择引擎。"""
    if SEARCH_PROVIDER == "google":
        return _search_google(query, max_results)
    return _search_ddg(query, max_results)


@tool
def web_search(query: str, max_results: int = 5) -> List[dict]:
    """互联网搜索，返回标题/摘要/链接列表。

    适用场景：
    - 时效性强的内容：新闻、近期发布、产品动态、技术博客
    - 维基百科或 ArXiv 上找不到的小众主题
    - 需要广泛了解某个话题时

    参数：
    - query：搜索关键词（中文/英文均可），3-15 字效果最佳
    - max_results：返回结果数，默认 5

    返回：[{title, content, url}] 列表
    """
    return run_web_search(query=query, max_results=max_results)
