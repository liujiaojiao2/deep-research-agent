"""Phase 1 工具层测试。

策略：核心断言用 mock，避免每次跑 pytest 都消耗 LLM 额度；
另留一组 live 用例（标记 live）跑真实联网作为冒烟。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.tools.compress_tool import compress_research
from src.tools.search_tool import run_web_search, web_search


# ---------- search_tool ----------

@pytest.mark.live
def test_web_search_live():
    """扰动观察：真实联网调用一次"""
    results = run_web_search("LangGraph tutorial", max_results=3)
    assert isinstance(results, list)
    assert len(results) >= 1
    assert all({"title", "content", "url"} <= r.keys() for r in results)


@pytest.mark.live
def test_web_search_tool_wrapper_live():
    """@tool 包装路径冒烟"""
    out = web_search.invoke({"query": "DeepSeek R1", "max_results": 2})
    assert isinstance(out, list)
    assert len(out) >= 1


# ---------- compress_tool ----------

class _FakeLLM:
    """最小化 mock LLM：拼接前 80 字作为压缩结果，避免真实调用。"""

    def invoke(self, prompt: str):
        body = prompt.split("：", 1)[-1][:120]
        return SimpleNamespace(content=f"- 压缩摘要片段：{body}…\n  来源: https://example.com")


def test_compress_empty_returns_placeholder():
    out = compress_research([], llm=_FakeLLM())
    assert out == "（无搜索结果）"


def test_compress_calls_llm_and_returns_string():
    fake = _FakeLLM()
    out = compress_research(
        [
            {"title": "A", "content": "alpha content " * 20, "url": "https://a.example"},
            {"title": "B", "content": "beta content " * 20, "url": "https://b.example"},
        ],
        llm=fake,
    )
    assert isinstance(out, str)
    assert len(out) > 0
    assert "压缩摘要" in out


def test_compress_string_response_compat():
    """兼容直接返回字符串的 LLM 实现"""

    class StrLLM:
        def invoke(self, prompt):
            return "fallback string"

    out = compress_research([{"title": "x", "content": "y", "url": "z"}], llm=StrLLM())
    assert out == "fallback string"
