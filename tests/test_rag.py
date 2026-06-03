"""RAG 单测：切块函数、retrieve 行为、tool 包装、空库/异常路径。"""
from __future__ import annotations

import pytest

from src.rag import split_text
from src.tools.rag_tool import local_knowledge_search


# ---------- split_text ----------

def test_split_text_short_text_returns_single():
    out = split_text("短句", chunk_size=500)
    assert out == ["短句"]


def test_split_text_empty_returns_empty():
    assert split_text("", chunk_size=500) == []
    assert split_text("   ", chunk_size=500) == []


def test_split_text_paragraph_boundaries():
    text = "第一段。\n\n第二段。\n\n第三段。"
    out = split_text(text, chunk_size=10)
    # 每段都很短，应该被分别保留（不一定每段一块，但都应能找到）
    joined = " ".join(out)
    assert "第一段" in joined
    assert "第二段" in joined
    assert "第三段" in joined


def test_split_text_long_paragraph_splits_with_overlap():
    text = "a" * 1200
    out = split_text(text, chunk_size=500, overlap=100)
    assert len(out) >= 2
    # 每块长度合理
    assert all(len(p) <= 500 for p in out)


# ---------- local_knowledge_search 工具 ----------

def test_tool_empty_kb_returns_friendly(monkeypatch):
    monkeypatch.setattr("src.rag.hybrid_retrieve",lambda **kw: [])
    out = local_knowledge_search.invoke({"query": "x"})
    assert len(out) == 1
    assert out[0]["source"] == "(empty)"
    assert "本地知识库为空" in out[0]["content"]


def test_tool_returns_hits(monkeypatch):
    fake_hits = [
        {"content": "GRPO 用 Qwen2.5-7B-Base", "source": "02.md", "chunk_index": 0, "similarity": 0.82},
        {"content": "学习率 1e-6", "source": "02.md", "chunk_index": 1, "similarity": 0.71},
    ]
    monkeypatch.setattr("src.rag.hybrid_retrieve",lambda **kw: fake_hits)
    out = local_knowledge_search.invoke({"query": "GRPO 用什么基座"})
    assert out == fake_hits


def test_tool_swallows_exception(monkeypatch):
    def boom(**kw):
        raise RuntimeError("chromadb crashed")

    monkeypatch.setattr("src.rag.hybrid_retrieve",boom)
    out = local_knowledge_search.invoke({"query": "x"})
    assert out[0]["source"] == "(error)"
    assert "chromadb crashed" in out[0]["content"]


def test_tool_passes_top_k(monkeypatch):
    captured = {}

    def fake_retrieve(**kw):
        captured.update(kw)
        return []

    monkeypatch.setattr("src.rag.hybrid_retrieve",fake_retrieve)
    local_knowledge_search.invoke({"query": "q", "top_k": 5})
    assert captured["query"] == "q"
    assert captured["top_k_text"] == 5


# ---------- tool_registry 切换 ----------

def test_registry_includes_rag_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_RAG", raising=False)
    from src.tools.tool_registry import get_all_tools

    names = [t.name for t in get_all_tools()]
    assert names[0] == "local_knowledge_search"
    assert "web_search" in names


def test_registry_excludes_rag_when_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_RAG", "false")
    monkeypatch.setenv("ENABLE_MEMORY", "false")  # 排除 memory，断言才是单一 web_search 首位
    from src.tools.tool_registry import get_all_tools

    names = [t.name for t in get_all_tools()]
    assert "local_knowledge_search" not in names
    assert names[0] == "web_search"


# ---------- live: 真实嵌入 + 检索 ----------

@pytest.mark.live
def test_full_rag_ingest_then_retrieve(tmp_path, monkeypatch):
    """端到端：写两篇小文档 → ingest → retrieve，验证语义检索能命中目标。"""
    # 用临时目录做 chroma 持久化，避免污染主库
    chroma_dir = tmp_path / "chroma"
    knowledge_dir = tmp_path / "kb"
    knowledge_dir.mkdir()
    (knowledge_dir / "a.md").write_text(
        "# GRPO 内部基准\n基础模型为 Qwen2.5-7B-Base，学习率 1e-6。",
        encoding="utf-8",
    )
    (knowledge_dir / "b.md").write_text(
        "# 完全无关的食谱\n红烧牛腩需要先焯水，然后加冰糖炒色。",
        encoding="utf-8",
    )

    # 临时重写常量并清缓存
    import src.rag as rag_mod

    monkeypatch.setattr(rag_mod, "CHROMA_DIR", chroma_dir)
    monkeypatch.setattr(rag_mod, "DEFAULT_COLLECTION", "test_kb_live")
    rag_mod._reset_client_cache()

    stats = rag_mod.ingest_directory(root=knowledge_dir, reset=True)
    assert stats["chunks"] >= 2

    hits = rag_mod.retrieve("GRPO 用的基础模型是什么", top_k=2)
    assert hits, "应至少命中一条"
    # 最相关的应该是 a.md
    assert hits[0]["source"] == "a.md"
    assert "Qwen2.5-7B-Base" in hits[0]["content"]
