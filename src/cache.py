"""Semantic Cache —— 相同/相似 query 直接返回历史报告，省全部 LLM。

设计：
- 复用 src/rag.py 的全局 chromadb client + BGE 嵌入
- 独立 collection `query_cache`
- 命中条件：cosine similarity ≥ THRESHOLD（默认 0.92）+ overall ≥ 7.0
- 仅写入高分研究（避免缓存劣质回答污染未来）

工业约定：
- 命中跳过整个 graph，直接返回（最大省钱）
- TTL 过期通过 metadata.date 字段后续可扩展（本 Phase 不做）
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import logging
import os
from typing import Any

from src.rag import (
    _force_reset_chroma_internals,
    embed_texts,
    get_collection as _get_rag_collection,
)

logger = logging.getLogger(__name__)


CACHE_COLLECTION = "query_cache"
# 0.88: 实测 BGE-small-zh 同义改写 query 大约 0.90-0.91，0.88 让它们能命中；
# 0.92 太严，仅完全相同 query 才中。
DEFAULT_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.88"))
MIN_SCORE_TO_CACHE = float(os.getenv("CACHE_MIN_SCORE", "7.0"))


def is_enabled() -> bool:
    return os.getenv("ENABLE_CACHE", "true").lower() != "false"


def _get_cache_collection():
    """复用 rag 的全局 chromadb client，多次 reset 兜底。"""
    _get_rag_collection()  # 触发全局 _chroma_client 初始化
    from src import rag as rag_mod

    def _do():
        return rag_mod._chroma_client.get_or_create_collection(
            name=CACHE_COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    try:
        return _do()
    except (AttributeError, KeyError) as e:
        logger.warning("cache chromadb 状态异常 (%s) → reset 后重试", e)
        rag_mod._chroma_client = None
        _force_reset_chroma_internals()
        _get_rag_collection()
        return _do()


def _hash_query(q: str) -> str:
    return hashlib.md5(q.encode("utf-8")).hexdigest()[:12]


def lookup(query: str, threshold: float = DEFAULT_THRESHOLD) -> dict | None:
    """查 cache。返回 None 表示未命中或不可用。"""
    if not is_enabled():
        return None
    try:
        coll = _get_cache_collection()
        if coll.count() == 0:
            return None
        emb = embed_texts([query])[0]
        result = coll.query(query_embeddings=[emb], n_results=1)
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        if not docs:
            return None
        similarity = 1.0 - float(dists[0])
        if similarity < threshold:
            return None
        meta = metas[0] or {}
        return {
            "query": meta.get("query", ""),
            "final_report": docs[0],
            "overall": meta.get("overall", 0.0),
            "date": meta.get("date", ""),
            "similarity": round(similarity, 4),
        }
    except Exception as e:
        logger.warning("cache lookup failed: %s", e)
        return None


def store(query: str, final_report: str, overall_score: float) -> bool:
    """写入缓存。低分或空报告不写。"""
    if not is_enabled():
        return False
    if not final_report or not final_report.strip():
        return False
    if overall_score < MIN_SCORE_TO_CACHE:
        return False
    try:
        coll = _get_cache_collection()
        record_id = f"{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{_hash_query(query)}"
        metadata = {
            "query": query[:200],
            "overall": float(overall_score),
            "date": _dt.datetime.now().strftime("%Y-%m-%d"),
        }
        emb = embed_texts([query])[0]
        coll.upsert(ids=[record_id], documents=[final_report], metadatas=[metadata], embeddings=[emb])
        return True
    except Exception as e:
        logger.warning("cache store failed: %s", e)
        return False


def stats() -> dict[str, Any]:
    try:
        coll = _get_cache_collection()
        return {"enabled": is_enabled(), "entries": coll.count(), "threshold": DEFAULT_THRESHOLD}
    except Exception:
        return {"enabled": is_enabled(), "entries": 0, "threshold": DEFAULT_THRESHOLD}
