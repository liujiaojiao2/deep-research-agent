"""SemanticCache 测试：开关、命中阈值、低分不缓存、异常容错。"""
from __future__ import annotations

import pytest

from src import cache


class _FakeColl:
    def __init__(self, docs=None, metas=None, dists=None, count_value=None):
        self._docs = docs or []
        self._metas = metas or []
        self._dists = dists or []
        self._count_value = count_value
        self.upserts = []

    def count(self):
        return self._count_value if self._count_value is not None else len(self._docs)

    def query(self, query_embeddings, n_results=1):
        return {
            "documents": [self._docs[:n_results]],
            "metadatas": [self._metas[:n_results]],
            "distances": [self._dists[:n_results]],
        }

    def upsert(self, ids, documents, metadatas, embeddings):
        for i, d, m in zip(ids, documents, metadatas):
            self.upserts.append((i, d, m))


# ---------- is_enabled ----------

def test_is_enabled_default(monkeypatch):
    monkeypatch.delenv("ENABLE_CACHE", raising=False)
    assert cache.is_enabled() is True


def test_is_enabled_off(monkeypatch):
    monkeypatch.setenv("ENABLE_CACHE", "false")
    assert cache.is_enabled() is False


# ---------- lookup ----------

def test_lookup_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_CACHE", "false")
    assert cache.lookup("q") is None


def test_lookup_returns_none_when_empty(monkeypatch):
    fake = _FakeColl(count_value=0)
    monkeypatch.setattr(cache, "_get_cache_collection", lambda: fake)
    monkeypatch.setenv("ENABLE_CACHE", "true")
    assert cache.lookup("q") is None


def test_lookup_hits_above_threshold(monkeypatch):
    fake = _FakeColl(
        docs=["这是缓存的报告内容"],
        metas=[{"query": "原 query", "overall": 8.5, "date": "2026-05-29"}],
        dists=[0.05],  # similarity = 0.95
    )
    monkeypatch.setattr(cache, "_get_cache_collection", lambda: fake)
    monkeypatch.setattr(cache, "embed_texts", lambda texts: [[0.1] * 384])
    monkeypatch.setenv("ENABLE_CACHE", "true")

    hit = cache.lookup("similar query", threshold=0.92)
    assert hit is not None
    assert hit["final_report"] == "这是缓存的报告内容"
    assert hit["overall"] == 8.5
    assert hit["similarity"] >= 0.92


def test_lookup_misses_below_threshold(monkeypatch):
    fake = _FakeColl(
        docs=["报告"],
        metas=[{"query": "完全不同的 query"}],
        dists=[0.5],  # similarity = 0.5
    )
    monkeypatch.setattr(cache, "_get_cache_collection", lambda: fake)
    monkeypatch.setattr(cache, "embed_texts", lambda texts: [[0.1] * 384])
    monkeypatch.setenv("ENABLE_CACHE", "true")
    assert cache.lookup("Q", threshold=0.92) is None


def test_lookup_swallows_exception(monkeypatch):
    def boom():
        raise RuntimeError("chromadb broken")

    monkeypatch.setattr(cache, "_get_cache_collection", boom)
    monkeypatch.setenv("ENABLE_CACHE", "true")
    assert cache.lookup("Q") is None


# ---------- store ----------

def test_store_skips_when_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_CACHE", "false")
    assert cache.store("Q", "report", 9.0) is False


def test_store_skips_empty_report(monkeypatch):
    monkeypatch.setenv("ENABLE_CACHE", "true")
    assert cache.store("Q", "", 9.0) is False
    assert cache.store("Q", "   ", 9.0) is False


def test_store_skips_low_score(monkeypatch):
    monkeypatch.setenv("ENABLE_CACHE", "true")
    monkeypatch.setattr(cache, "MIN_SCORE_TO_CACHE", 7.0)
    assert cache.store("Q", "report", 5.0) is False


def test_store_writes_high_score(monkeypatch):
    fake = _FakeColl()
    monkeypatch.setattr(cache, "_get_cache_collection", lambda: fake)
    monkeypatch.setattr(cache, "embed_texts", lambda texts: [[0.1] * 384])
    monkeypatch.setenv("ENABLE_CACHE", "true")
    monkeypatch.setattr(cache, "MIN_SCORE_TO_CACHE", 7.0)
    ok = cache.store("Q", "report", 8.5)
    assert ok is True
    assert len(fake.upserts) == 1
    _, doc, meta = fake.upserts[0]
    assert doc == "report"
    assert meta["overall"] == 8.5
    assert meta["query"] == "Q"


def test_store_swallows_exception(monkeypatch):
    monkeypatch.setenv("ENABLE_CACHE", "true")

    def boom():
        raise RuntimeError("disk full")

    monkeypatch.setattr(cache, "_get_cache_collection", boom)
    assert cache.store("Q", "report", 9.0) is False


# ---------- stats ----------

def test_stats(monkeypatch):
    fake = _FakeColl(docs=["a", "b"])
    monkeypatch.setattr(cache, "_get_cache_collection", lambda: fake)
    monkeypatch.setenv("ENABLE_CACHE", "true")
    s = cache.stats()
    assert s["enabled"] is True
    assert s["entries"] == 2
    assert "threshold" in s


# ---------- live: 真实嵌入 + ChromaDB ----------

@pytest.mark.live
def test_live_store_then_lookup(tmp_path, monkeypatch):
    """端到端：写入 → 用相似 query 命中。"""
    import src.rag as rag_mod

    chroma_dir = tmp_path / "chroma"
    monkeypatch.setattr(rag_mod, "CHROMA_DIR", chroma_dir)
    rag_mod._chroma_client = None
    # 用独立 collection 名避免与主库冲突
    monkeypatch.setattr(cache, "CACHE_COLLECTION", "test_query_cache")

    ok = cache.store("BGE-M3 嵌入模型的优势", "完整报告内容...", overall_score=8.5)
    assert ok is True

    hit = cache.lookup("BGE-M3 嵌入模型有哪些优势", threshold=0.85)
    assert hit is not None
    assert "完整报告" in hit["final_report"]
