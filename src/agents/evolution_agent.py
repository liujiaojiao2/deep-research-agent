"""HarnessForge 式联合进化 Agent —— Harness + Policy 同时进化。

当前自进化（Phase 3）只改报告文本（output-level）。
HarnessForge 升级：记录每次成功 run 的完整策略（prompt+tool+score），
下次同类 query 自动注入最优策略。

两个核心 API：
- record_evolution(state)  : 归档一次 run 的策略快照
- recall_evolution(query)  : 检索历史最优策略（按 score 降序）

节点：
- evolution_log_node: 在 memory_archive 后执行，记录 + 写入 ChromaDB
  不修改主 state（仅写可观测字段），失败不阻断主流程

策略快照存储格式：
  {query_type, tools_used, researcher_mode, overall_score, query, timestamp}

召回时：embed query → ChromaDB 语义检索 → 返回 top-k 高分策略
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

from src.config import get_llm
from src.rag import embed_texts, get_collection
from src.state import SupervisorState

logger = logging.getLogger(__name__)

EVOLUTION_COLLECTION = os.getenv("EVOLUTION_COLLECTION", "evolution_log")
MIN_SCORE_TO_RECORD = float(os.getenv("EVOLUTION_MIN_SCORE", "7.0"))


def _get_evo_collection():
    """复用 rag 的全局 chromadb client + 独立 collection。"""
    get_collection()  # 触发全局 client
    from src import rag as rag_mod

    def _do():
        return rag_mod._chroma_client.get_or_create_collection(
            name=EVOLUTION_COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    try:
        return _do()
    except (AttributeError, KeyError) as e:
        logger.warning("evolution chromadb error (%s) -> reset", e)
        from src.rag import _force_reset_chroma_internals

        rag_mod._chroma_client = None
        _force_reset_chroma_internals()
        get_collection()
        return _do()


# ---------- 策略快照抽取 ----------

def _extract_tool_sequence(state: dict) -> list[str]:
    """从 research_results source 字段抽 tools 序列。"""
    tools = []
    for r in state.get("research_results") or []:
        src = r.get("source", "") or ""
        if "tools=" in src:
            tail = src.split("tools=", 1)[1].rstrip(")")
            tools.extend(t.strip() for t in tail.split(",") if t.strip())
        else:
            tools.append(src)
    return tools


def _classify_query_type(query: str, llm) -> str:
    """让 LLM 对 query 分类（3-8 字标签），用于后续策略匹配。

    例: "对比 GRPO 和 PPO 的差异" → "算法对比"
    """
    prompt = (
        f"请用 3-8 个字给以下研究问题打一个类型标签，只输出标签本身，不要解释。\n\n"
        f"问题：{query[:200]}"
    )
    try:
        resp = llm.invoke(prompt)
        label = (resp.content if hasattr(resp, "content") else str(resp)).strip()[:30]
        return label or "通用研究"
    except Exception:
        return "通用研究"


def _build_snapshot_text(state: dict, query_type: str) -> str:
    """把一次 run 的策略信息压缩为可嵌入的文本。"""
    tools = _extract_tool_sequence(state)
    score = (state.get("quality_score") or {}).get("overall", 0.0)
    researcher = os.getenv("RESEARCHER_MODE", "react")
    return (
        f"query_type={query_type}\n"
        f"tools={','.join(tools[:8]) or 'none'}\n"
        f"researcher_mode={researcher}\n"
        f"score={score}\n"
        f"query={state.get('query', '')[:200]}"
    )


# ---------- API ----------

def record_evolution(state: SupervisorState, llm=None) -> bool:
    """归档一次成功 run 的策略快照；低分不记录。"""
    score = float((state.get("quality_score") or {}).get("overall", 0.0))
    if score < MIN_SCORE_TO_RECORD:
        return False

    try:
        llm = llm or get_llm()
        query_type = _classify_query_type(state.get("query", ""), llm)
        text = _build_snapshot_text(dict(state), query_type)
        emb = embed_texts([text])[0]

        rid = hashlib.md5(text.encode()).hexdigest()[:12]
        meta: dict[str, Any] = {
            "query_type": query_type,
            "tools_used": json.dumps(_extract_tool_sequence(dict(state))),
            "researcher_mode": os.getenv("RESEARCHER_MODE", "react"),
            "overall_score": score,
            "query": state.get("query", "")[:200],
        }

        coll = _get_evo_collection()
        coll.upsert(ids=[rid], documents=[text], metadatas=[meta], embeddings=[emb])
        return True
    except Exception as e:
        logger.warning("record_evolution failed: %s", e)
        return False


def recall_evolution(query: str, top_k: int = 3) -> list[dict]:
    """检索历史上类似 query 的最优策略；返回按 score 降序的列表。"""
    try:
        coll = _get_evo_collection()
        if coll.count() == 0:
            return []
        emb = embed_texts([query])[0]
        result = coll.query(query_embeddings=[emb], n_results=top_k)
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        items = []
        for doc, meta, dist in zip(docs, metas, dists):
            items.append({
                "doc": doc,
                "query_type": (meta or {}).get("query_type", ""),
                "tools_used": json.loads((meta or {}).get("tools_used", "[]")),
                "researcher_mode": (meta or {}).get("researcher_mode", ""),
                "overall_score": float((meta or {}).get("overall_score", 0.0)),
                "similarity": round(1.0 - float(dist), 4),
            })
        items.sort(key=lambda x: x["overall_score"], reverse=True)
        return items
    except Exception as e:
        logger.warning("recall_evolution failed: %s", e)
        return []


def _format_strategy_hint(strategies: list[dict]) -> str:
    """把多条历史策略凝练成 brief_writer prompt 注入片段。"""
    if not strategies:
        return ""
    lines = ["\n你已知的过往成功经验（仅供参考，不要盲目照搬）："]
    for i, s in enumerate(strategies[:2], 1):
        lines.append(
            f"{i}. 类似问题 [{s['query_type']}] (分数={s['overall_score']}): "
            f"工具顺序={','.join(s['tools_used'][:5]) or '无记录'}, "
            f"researcher={s['researcher_mode']}"
        )
    return "\n".join(lines)


# ---------- 节点 ----------

def evolution_log_node(state: SupervisorState, llm=None) -> dict:
    """归档策略 + 下次可用。不修改主 state（仅可观测字段）。"""
    llm = llm or get_llm()
    recorded = False
    try:
        recorded = record_evolution(state, llm)
    except Exception as e:
        logger.warning("evolution_log_node failed: %s", e)

    return {"evolution_recorded": recorded}


def evolution_recall_node(state: SupervisorState) -> dict:
    """在 brief_writer 之前查最优策略；不修改主 state，只写策略片段。"""
    try:
        strategies = recall_evolution(state.get("query", ""), top_k=3)
        hint = _format_strategy_hint(strategies)
        return {"evolution_strategy_hint": hint}
    except Exception as e:
        logger.warning("evolution_recall_node failed: %s", e)
        return {"evolution_strategy_hint": ""}
