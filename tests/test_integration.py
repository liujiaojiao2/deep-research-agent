"""Phase 5 集成测试 —— 真实 LLM + 真实搜索，全部标记 `live`。

跑法：
    uv run pytest tests/test_integration.py -v -m live

跑一遍消耗 6-15 次 DeepSeek 调用 + 2-6 次 DDG 查询，按需触发。
"""
from __future__ import annotations

import pytest

from main import run_research


@pytest.mark.live
def test_simple_query_one_iter(tmp_path):
    report = run_research(
        "Python 和 JavaScript 的主要区别是什么？",
        max_iter=1,
        save_dir=tmp_path,
    )
    assert isinstance(report, str)
    # 验收标准：方案要求 > 500 字（这里按字符算更稳）
    assert len(report) > 500
    assert "Python" in report
    assert "JavaScript" in report


@pytest.mark.live
def test_complex_query_with_citations(tmp_path):
    report = run_research(
        "RAG 系统在长文档问答场景下的主要技术挑战",
        max_iter=2,
        save_dir=tmp_path,
    )
    assert len(report) > 1000
    # 应该带某种引用痕迹（中英文标签都可能）
    has_citation = any(token in report for token in ["http", "来源", "参考", "References"])
    assert has_citation, "终稿应保留来源标注"


@pytest.mark.live
def test_graph_compiles_and_streams_smoke(tmp_path):
    """最小开销冒烟：仅跑 1 iter 简单 query，主要验证 main.run_research 返回非空。"""
    report = run_research("什么是 LangGraph？", max_iter=1, save_dir=tmp_path)
    assert len(report.strip()) > 0
