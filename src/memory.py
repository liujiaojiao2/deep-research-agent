"""Memory 模块：复用 src/rag.py 的 chromadb 实例与 BGE 嵌入。

两个独立 collection：
- memory_episodic：跨会话的研究记忆（每次研究归档一条）
- memory_preference：从研究中抽取的用户偏好事实

复用 rag.get_collection 的客户端模式：全局 client 强引用 + 状态异常重试。
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
from typing import TypedDict

from src.rag import (
    _force_reset_chroma_internals,
    embed_texts,
    get_collection as _get_rag_collection,  # 仅复用其 client 机制
)

logger = logging.getLogger(__name__)


EPISODIC_COLLECTION = "memory_episodic"
PREFERENCE_COLLECTION = "memory_preference"


def _get_collection(name: str):
    """复用 rag.get_collection 持有的全局 client，避免两套 chromadb client 状态冲突。

    实现：先调一次 _get_rag_collection() 触发全局 _chroma_client 初始化，
    然后通过它 get_or_create 我们自己的 collection。
    """
    _get_rag_collection()  # 确保 src.rag._chroma_client 已初始化
    from src import rag as rag_mod

    def _do():
        return rag_mod._chroma_client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    try:
        return _do()
    except (AttributeError, KeyError) as e:
        logger.warning("memory chromadb 状态异常 (%s) → reset 后重试", e)
        rag_mod._chroma_client = None
        _force_reset_chroma_internals()
        _get_rag_collection()
        return _do()


# ---------- Episodic Memory ----------

class EpisodicRecord(TypedDict, total=False):
    id: str
    query: str
    summary: str
    overall_score: float
    tools_used: list[str]
    date: str


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def _summarize_research(state: dict) -> str:
    """从最终 state 提炼一条研究摘要文本（用于嵌入与检索）。"""
    query = state.get("query", "")
    quality = state.get("quality_score") or {}
    overall = quality.get("overall", "N/A")
    research = state.get("research_results") or []
    tools_used = []
    for r in research:
        src = r.get("source", "") or ""
        if "tools=" in src:
            tail = src.split("tools=", 1)[1].rstrip(")")
            tools_used.extend(t.strip() for t in tail.split(",") if t.strip())
    tools_used = sorted(set(tools_used))

    final = state.get("final_report", "") or ""
    # 取报告开头 600 字 + 结论性语句
    snippet = final[:600]
    return (
        f"研究问题: {query}\n"
        f"完成时间: {_dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"质量评分: overall={overall}\n"
        f"使用工具: {', '.join(tools_used) or '无记录'}\n"
        f"报告摘要:\n{snippet}"
    )


def archive_episodic(state: dict) -> EpisodicRecord:
    """把一次完整研究归档为一条向量记录。"""
    if not state.get("final_report"):
        return {}  # 没出过终稿不归档

    summary = _summarize_research(state)
    record_id = f"{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{_hash(state.get('query', ''))}"
    metadata = {
        "query": state.get("query", "")[:200],
        "overall_score": float((state.get("quality_score") or {}).get("overall", 0.0)),
        "date": _dt.datetime.now().strftime("%Y-%m-%d"),
        "iteration_count": int(state.get("iteration_count", 0)),
    }

    coll = _get_collection(EPISODIC_COLLECTION)
    embedding = embed_texts([summary])[0]
    coll.upsert(ids=[record_id], documents=[summary], metadatas=[metadata], embeddings=[embedding])

    return {
        "id": record_id,
        "query": state.get("query", ""),
        "summary": summary,
        "overall_score": metadata["overall_score"],
        "date": metadata["date"],
    }


def recall_episodic(query: str, top_k: int = 3) -> list[dict]:
    """检索与 query 最相关的过去研究。空库时返回 []。"""
    coll = _get_collection(EPISODIC_COLLECTION)
    if coll.count() == 0:
        return []
    embedding = embed_texts([query])[0]
    result = coll.query(query_embeddings=[embedding], n_results=top_k)
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]
    return [
        {
            "summary": doc,
            "query": meta.get("query", ""),
            "date": meta.get("date", ""),
            "overall_score": meta.get("overall_score", 0.0),
            "similarity": round(1.0 - float(dist), 4),
        }
        for doc, meta, dist in zip(docs, metas, dists)
    ]


# ---------- Preference Memory ----------

_PREFERENCE_EXTRACT_PROMPT = """你是一个用户行为分析师。给定一次研究的问题与产出，提炼最多 3 条用户偏好。

要求每条偏好：
- 是 **关于用户本身** 的事实（例如"用户偏好分点列表"），不要复制研究内容
- 可被未来 brief_writer 用来调整风格 / 选题
- 用一句话表达，不超过 50 字

输入：
研究问题：{query}
用户在 HITL 中的决策（如有）：{hitl}
最终报告（前 800 字）：
{report}

严格输出 JSON 数组（仅 JSON，不要解释）：
[{{"preference": "用户喜欢...", "evidence": "..."}}, ...]
如果没有可提炼的偏好，输出 []。
"""


def _parse_json_list(text: str) -> list:
    import re

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def extract_preferences(state: dict, llm) -> list[dict]:
    """让 LLM 从一次研究里提炼用户偏好（最多 3 条）。"""
    if not state.get("final_report"):
        return []
    prompt = _PREFERENCE_EXTRACT_PROMPT.format(
        query=state.get("query", ""),
        hitl="(无)",  # 留作未来扩展点
        report=(state.get("final_report") or "")[:800],
    )
    resp = llm.invoke(prompt)
    raw = resp.content if hasattr(resp, "content") else str(resp)
    items = _parse_json_list(raw)
    cleaned = []
    for it in items[:3]:
        if isinstance(it, dict) and it.get("preference"):
            cleaned.append({
                "preference": str(it["preference"])[:200],
                "evidence": str(it.get("evidence", ""))[:200],
            })
    return cleaned


def archive_preferences(preferences: list[dict]) -> int:
    """把抽取出的偏好写入 PreferenceMemory；同义偏好用文本 hash 去重。"""
    if not preferences:
        return 0
    coll = _get_collection(PREFERENCE_COLLECTION)
    texts = [p["preference"] for p in preferences]
    ids = [f"pref_{_hash(t)}" for t in texts]
    metadatas = [{"evidence": p.get("evidence", ""), "date": _dt.datetime.now().strftime("%Y-%m-%d")} for p in preferences]
    embeddings = embed_texts(texts)
    coll.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    return len(preferences)


def get_active_preferences(query: str | None = None, top_k: int = 5) -> list[str]:
    """读出用户偏好；带 query 时按相关性排序，否则取最近的。"""
    coll = _get_collection(PREFERENCE_COLLECTION)
    if coll.count() == 0:
        return []
    if query:
        embedding = embed_texts([query])[0]
        result = coll.query(query_embeddings=[embedding], n_results=top_k)
        return result.get("documents", [[]])[0]
    # 无 query：返回最近添加的 top_k 条
    result = coll.get(limit=top_k)
    return result.get("documents", [])


def _reset_memory_client_cache():
    """测试用：清掉 RAG 共享的 chromadb client。"""
    from src import rag as rag_mod

    rag_mod._chroma_client = None
    _force_reset_chroma_internals()
