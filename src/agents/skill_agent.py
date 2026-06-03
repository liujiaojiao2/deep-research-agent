"""Memento-Skills 技能库 —— 把成功经验编码为可复用技能模板。

HarnessForge（已做）: 记住"用了什么工具 + 得了多少分"
Memento-Skills 升级: 从高分 run 提炼"成熟的行动计划 SOP"

核心 API:
- extract_skill(state, llm): LLM 从高分 run 提炼 skill（名称/触发条件/步骤 SOP）
- match_skills(query): BGE 语义 + 关键词匹配，找到可用的 skills
- skill_library_node(state): 在 evolution_log 后执行，提炼 + 存储

技能模板存储:
  {name, trigger_keywords, steps_sop, success_count, avg_score}
  其中 steps_sop 是一句话描述的步骤顺序（可直接注入 researcher）
"""
from __future__ import annotations

import json
import logging
import os
import re

from src.config import get_llm
from src.rag import embed_texts, get_collection
from src.state import SupervisorState

logger = logging.getLogger(__name__)

SKILL_COLLECTION = os.getenv("SKILL_COLLECTION", "skill_library")
MIN_SKILL_SCORE = float(os.getenv("SKILL_MIN_SCORE", "7.5"))


def _get_skill_collection():
    get_collection()
    from src import rag as rag_mod

    def _do():
        return rag_mod._chroma_client.get_or_create_collection(
            name=SKILL_COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    try:
        return _do()
    except (AttributeError, KeyError) as e:
        logger.warning("skill chromadb error (%s) -> reset", e)
        from src.rag import _force_reset_chroma_internals

        rag_mod._chroma_client = None
        _force_reset_chroma_internals()
        get_collection()
        return _do()


# ---------- Skill 结构 ----------

_EXTRACT_PROMPT = """你是一个 Agent 行为分析师。请从一次成功的研究 run 中提炼一个可复用的"技能模板"。

研究问题：{query}
质量评分：overall={overall}
使用工具序列：{tools}
研究者模式：{researcher_mode}

请严格按 JSON 输出（不要解释）：
{{
  "name": "技能名称 (3-8个字, 如'算法对比研究')",
  "trigger_keywords": ["关键词1", "关键词2"],
  "steps_sop": "一句话描述最优工具调用顺序 (如: 先wiki定义双方 -> 再arxiv找对比论文 -> 最后web补充最新进展)"
}}

要求：
- name 应该简洁但准确描述"做了什么类型的研究"
- trigger_keywords 应该是 2-4 个很具体的关键词，后续用于匹配
- steps_sop 必须是描述工具调用顺序，不是描述研究内容
- 如果这次 run 没有明显的可复用模式，返回 {{"skip": true}}
"""


def _parse_skill_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise json.JSONDecodeError("no json", raw, 0)
    return json.loads(match.group(0))


# ---------- API ----------

def extract_skill(state: SupervisorState, llm=None) -> dict | None:
    """从一次高分 run 提炼技能模板。返回 None 表示不满足条件或无法提炼。"""
    score = float((state.get("quality_score") or {}).get("overall", 0.0))
    if score < MIN_SKILL_SCORE:
        return None

    try:
        llm = llm or get_llm()
        # 提取工具序列
        tools: list[str] = []
        for r in state.get("research_results") or []:
            src = r.get("source", "") or ""
            if "tools=" in src:
                tail = src.split("tools=", 1)[1].rstrip(")")
                tools.extend(t.strip() for t in tail.split(",") if t.strip())

        prompt = _EXTRACT_PROMPT.format(
            query=state.get("query", "")[:200],
            overall=score,
            tools=", ".join(tools[:10]) or "无记录",
            researcher_mode=os.getenv("RESEARCHER_MODE", "react"),
        )
        resp = llm.invoke(prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        data = _parse_skill_json(raw)
        if data.get("skip"):
            return None

        skill = {
            "name": str(data.get("name", ""))[:30],
            "trigger_keywords": list(data.get("trigger_keywords", []))[:4],
            "steps_sop": str(data.get("steps_sop", ""))[:300],
            "success_count": 1,
            "avg_score": score,
        }
        return skill if skill["name"] and skill["steps_sop"] else None
    except Exception as e:
        logger.warning("extract_skill failed: %s", e)
        return None


def store_skill(skill: dict) -> bool:
    """把技能写入 ChromaDB（同名更新合并 count/avg_score）。"""
    if not skill or not skill.get("name"):
        return False
    try:
        coll = _get_skill_collection()
        # 查同名 skill
        name = skill["name"]
        existing = coll.get(ids=[name])
        if existing and existing["ids"]:
            old_meta = (existing.get("metadatas") or [{}])[0]
            old_count = old_meta.get("success_count", 0) or 0
            old_avg = old_meta.get("avg_score", 0.0) or 0.0
            new_count = old_count + 1
            new_avg = round((old_avg * old_count + skill["avg_score"]) / new_count, 2)
            skill["success_count"] = new_count
            skill["avg_score"] = new_avg

        text = f"技能: {skill['name']}\n触发: {','.join(skill['trigger_keywords'])}\n步骤: {skill['steps_sop']}"
        emb = embed_texts([text])[0]
        meta = {
            "name": skill["name"],
            "trigger_keywords": json.dumps(skill["trigger_keywords"]),
            "steps_sop": skill["steps_sop"],
            "success_count": skill["success_count"],
            "avg_score": skill["avg_score"],
        }
        coll.upsert(ids=[name], documents=[text], metadatas=[meta], embeddings=[emb])
        return True
    except Exception as e:
        logger.warning("store_skill failed: %s", e)
        return False


def match_skills(query: str, top_k: int = 3) -> list[dict]:
    """匹配可用技能。先语义检索，再关键词加权。"""
    try:
        coll = _get_skill_collection()
        if coll.count() == 0:
            return []
        emb = embed_texts([query])[0]
        result = coll.query(query_embeddings=[emb], n_results=top_k)
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        matched = []
        for doc, meta, dist in zip(docs, metas, dists):
            if not meta:
                continue
            keywords = json.loads(meta.get("trigger_keywords", "[]"))
            # 关键词加分: query 中含触发关键词 → 分数加权
            kw_bonus = sum(1 for kw in keywords if kw in query) * 0.05
            similarity = round(1.0 - float(dist) + kw_bonus, 4)
            matched.append({
                "name": meta.get("name", ""),
                "trigger_keywords": keywords,
                "steps_sop": meta.get("steps_sop", ""),
                "success_count": meta.get("success_count", 0) or 0,
                "avg_score": meta.get("avg_score", 0.0) or 0.0,
                "similarity": similarity,
            })
        matched.sort(key=lambda x: x["similarity"], reverse=True)
        return matched
    except Exception as e:
        logger.warning("match_skills failed: %s", e)
        return []


def _format_skill_injection(matched: list[dict], threshold: float = 0.35) -> str:
    """把匹配的 skill 凝练为 researcher 注入片段。"""
    usable = [s for s in matched if s["similarity"] >= threshold]
    if not usable:
        return ""
    best = usable[0]
    return (
        f"\n[技能提示] 系统检测到本次研究与已有成功技能 \"{best['name']}\" 匹配 "
        f"(相似度={best['similarity']:.2f}, 成功率={best['success_count']}次, "
        f"平均分={best['avg_score']})。"
        f"建议工具调用顺序参考: {best['steps_sop']}。"
        f"你可以自主决定是否采纳此建议。"
    )


# ---------- 节点 ----------

def skill_library_node(state: SupervisorState, llm=None) -> dict:
    """提炼 + 存储技能。失败不阻断。"""
    llm = llm or get_llm()
    extracted = False
    try:
        skill = extract_skill(state, llm)
        if skill:
            extracted = store_skill(skill)
    except Exception as e:
        logger.warning("skill_library_node failed: %s", e)

    return {"skill_extracted": extracted}
