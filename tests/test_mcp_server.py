"""MCP server 测试：工具注册、resource、函数行为、错误路径。

不启动真实 transport（stdio/SSE），只验证：
- 6 个工具按预期注册
- 工具函数直接调用结果
- resource 元信息
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_mcp_module():
    spec = importlib.util.spec_from_file_location(
        "_mcp_server_test",
        Path(__file__).resolve().parent.parent / "scripts" / "mcp_server.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_six_tools_registered():
    mod = _load_mcp_module()
    names = mod._list_tools()
    expected = {
        "web_search",
        "wikipedia_search",
        "arxiv_search",
        "local_knowledge_search",
        "recall_episodic_memory",
        "python_calculator",
    }
    assert expected.issubset(set(names)), f"missing: {expected - set(names)}"


def test_python_calculator_direct_call():
    mod = _load_mcp_module()
    out = mod.python_calculator("(95-87)/87 * 100")
    assert float(out) > 9.0


def test_python_calculator_bad_expression_returns_string():
    mod = _load_mcp_module()
    out = mod.python_calculator("not_python_at_all!!")
    assert isinstance(out, str)
    assert len(out) > 0


def test_local_knowledge_search_friendly_when_empty(monkeypatch):
    mod = _load_mcp_module()
    monkeypatch.setattr("src.rag.retrieve", lambda **kw: [])
    out = mod.local_knowledge_search("anything")
    assert len(out) == 1
    assert "empty" in out[0]["source"]


def test_recall_episodic_memory_friendly_when_empty(monkeypatch):
    mod = _load_mcp_module()
    monkeypatch.setattr("src.memory.recall_episodic", lambda **kw: [])
    out = mod.recall_episodic_memory("anything")
    assert len(out) == 1
    assert "无过往" in out[0]["summary"]


def test_project_meta_resource():
    mod = _load_mcp_module()
    text = mod.project_meta()
    assert "DeepResearch" in text
    assert "web_search" in text


def test_mcp_instance_attributes():
    mod = _load_mcp_module()
    assert mod.mcp.name == "deep-research-tools"
    # 应有 instructions
    assert mod.mcp.instructions is not None
    assert "DeepResearch" in mod.mcp.instructions
