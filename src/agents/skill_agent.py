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
from pathlib import Path

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
工具执行摘要：{tool_summaries}

请严格按 JSON 输出（不要解释）：
{{
  "name": "技能名称 (3-8个字, 如'算法对比研究')",
  "trigger_keywords": ["关键词1", "关键词2"],
  "steps_sop": "详细步骤描述 (如: 步骤1用wiki查双方定义 -> 步骤2用arxiv找对比benchmark -> 步骤3用web补充最新社区讨论)",
  "failure_modes": "常见问题 (如: arxiv中文论文较少,需配合中文web搜索; wiki定义可能过时,需用web验证)"
}}

要求：
- name 应该简洁但准确描述"做了什么类型的研究"
- trigger_keywords 应该是 2-4 个很具体的关键词，后续用于匹配
- steps_sop 必须是描述工具调用顺序，不是描述研究内容；参考工具执行摘要中的数据量，不要凭空编造
- failure_modes 基于摘要中的错误/空结果推断常见坑点
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
        # 提取工具序列 + 执行摘要
        tools: list[str] = []
        tool_summary_lines: list[str] = []
        for r in state.get("research_results") or []:
            src = r.get("source", "") or ""
            if "tools=" in src:
                tail = src.split("tools=", 1)[1].rstrip(")")
                tools.extend(t.strip() for t in tail.split(",") if t.strip())
            # 读取 tool_outputs 构造摘要
            for to in r.get("tool_outputs") or []:
                tool_summary_lines.append(
                    f"  {to.get('tool')}({to.get('query')}): "
                    f"{to.get('result_count')}条结果, "
                    f"共{to.get('result_total_chars')}字符"
                )

        prompt = _EXTRACT_PROMPT.format(
            query=state.get("query", "")[:200],
            overall=score,
            tools=", ".join(tools[:10]) or "无记录",
            researcher_mode=os.getenv("RESEARCHER_MODE", "react"),
            tool_summaries="\n".join(tool_summary_lines) if tool_summary_lines else "(无详细数据)",
        )
        resp = llm.invoke(prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        data = _parse_skill_json(raw)
        if data.get("skip"):
            return None

        skill = {
            "name": str(data.get("name", ""))[:30],
            "trigger_keywords": list(data.get("trigger_keywords", []))[:4],
            "steps_sop": str(data.get("steps_sop", ""))[:500],
            "failure_modes": str(data.get("failure_modes", ""))[:200],
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

        # 渐进式披露：轻量文本用于嵌入/检索，重量内容存 metadata
        light_text = (
            f"技能: {skill['name']}\n"
            f"触发关键词: {','.join(skill['trigger_keywords'])}\n"
            f"描述: 成功{skill['success_count']}次, 均分{skill['avg_score']}"
        )
        emb = embed_texts([light_text])[0]
        meta = {
            "name": skill["name"],
            "trigger_keywords": json.dumps(skill["trigger_keywords"]),
            "steps_sop": skill["steps_sop"],
            "failure_modes": skill.get("failure_modes", ""),
            "success_count": skill["success_count"],
            "avg_score": skill["avg_score"],
        }
        coll.upsert(ids=[name], documents=[light_text], metadatas=[meta], embeddings=[emb])
        return True
    except Exception as e:
        logger.warning("store_skill failed: %s", e)
        return False


def match_skills(query: str, top_k: int = 3) -> list[dict]:
    """第一级：扫描所有 skill 的 name + 关键词，返回轻量匹配列表。

    只返回 name / keywords / similarity / stats，不包含重量内容。
    需要详细步骤时调用 expand_skill() 展开。
    """
    try:
        coll = _get_skill_collection()
        if coll.count() == 0:
            return []
        emb = embed_texts([query])[0]
        # 只取 metadatas + distances，不取 heavy document
        result = coll.query(query_embeddings=[emb], n_results=top_k,
                          include=["metadatas", "distances"])
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        matched = []
        for meta, dist in zip(metas, dists):
            if not meta:
                continue
            keywords = json.loads(meta.get("trigger_keywords", "[]"))
            kw_bonus = sum(1 for kw in keywords if kw in query) * 0.05
            similarity = round(1.0 - float(dist) + kw_bonus, 4)
            matched.append({
                "name": meta.get("name", ""),
                "trigger_keywords": keywords,
                "success_count": meta.get("success_count", 0) or 0,
                "avg_score": meta.get("avg_score", 0.0) or 0.0,
                "similarity": similarity,
            })
        matched.sort(key=lambda x: x["similarity"], reverse=True)
        return matched
    except Exception as e:
        logger.warning("match_skills failed: %s", e)
        return []


def expand_skill(name: str) -> dict | None:
    """第二级：按名称展开 skill 的完整内容（步骤 SOP + 失败模式）。"""
    try:
        coll = _get_skill_collection()
        result = coll.get(ids=[name])
        metas = result.get("metadatas", [])
        if not metas:
            return None
        meta = metas[0]
        return {
            "name": meta.get("name", ""),
            "steps_sop": meta.get("steps_sop", ""),
            "failure_modes": meta.get("failure_modes", ""),
            "success_count": meta.get("success_count", 0) or 0,
            "avg_score": meta.get("avg_score", 0.0) or 0.0,
        }
    except Exception as e:
        logger.warning("expand_skill failed: %s", e)
        return None


def _format_skill_injection(matched: list[dict], threshold: float = 0.35) -> str:
    """渐进式披露注入：
    第一级 — 列出所有匹配 skill 的 name + 相似度（轻量扫描结果）
    第二级 — 展开最佳匹配 skill 的完整内容（步骤 SOP + 注意事项）
    """
    usable = [s for s in matched if s["similarity"] >= threshold]
    if not usable:
        return ""

    # 第一级：展示扫描结果（所有候选 skill 的摘要）
    lines = ["\n[技能扫描] 发现以下可复用技能："]
    for i, s in enumerate(usable, 1):
        lines.append(
            f"  {i}. {s['name']} "
            f"(相似度={s['similarity']:.2f}, 成功{s['success_count']}次, 均分{s['avg_score']})"
        )

    # 第二级：展开最佳匹配的完整内容（优先 ChromaDB，降级用传入数据）
    best = usable[0]
    expanded = expand_skill(best["name"]) or best  # ChromaDB 未命中时用轻量数据
    if expanded.get("steps_sop"):
        lines.append(f"\n[最佳匹配: {expanded['name']}]")
        lines.append(f"步骤: {expanded['steps_sop']}")
        if expanded.get("failure_modes"):
            lines.append(f"注意事项: {expanded['failure_modes']}")
        lines.append(f"历史: 成功{expanded.get('success_count', best['success_count'])}次, "
                     f"均分{expanded.get('avg_score', best['avg_score'])}")
    lines.append("你可以自主决定是否采纳此建议。")
    return "\n".join(lines)


# ---------- MD 导出 ----------

SKILL_EXPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "skills")


def list_all_skills() -> list[dict]:
    """列出所有已存储的 skill（轻量摘要）。"""
    try:
        coll = _get_skill_collection()
        if coll.count() == 0:
            return []
        result = coll.get(include=["metadatas"])
        skills = []
        for meta in (result.get("metadatas") or []):
            skills.append({
                "name": meta.get("name", ""),
                "trigger_keywords": json.loads(meta.get("trigger_keywords", "[]")),
                "steps_sop": meta.get("steps_sop", ""),
                "failure_modes": meta.get("failure_modes", ""),
                "success_count": meta.get("success_count", 0) or 0,
                "avg_score": meta.get("avg_score", 0.0) or 0.0,
            })
        skills.sort(key=lambda x: x["avg_score"], reverse=True)
        return skills
    except Exception as e:
        logger.warning("list_all_skills failed: %s", e)
        return []


def _build_description(skill: dict) -> str:
    """从 keywords + stats 自动生成 description 字段。"""
    kw_str = "、".join(skill.get("trigger_keywords", [])[:3])
    desc = f"从成功研究中提炼的可复用技能。触发关键词: {kw_str}。"
    desc += f" 历史成功{skill.get('success_count', 0)}次, 均分{skill.get('avg_score', 0)}。"
    return desc


def export_skill_to_md(skill_name: str, output_dir: str | None = None) -> str | None:
    """把一条 skill 导出为 Anthropic 标准 SKILL.md 格式。

    Args:
        skill_name: 技能名称
        output_dir: 输出目录，默认 skills/

    Returns:
        写入的文件路径，失败或无此 skill 返回 None
    """
    skill = expand_skill(skill_name)
    if not skill:
        logger.warning("skill %r not found, cannot export", skill_name)
        return None

    # 从 ChromaDB metadata 取完整数据（含 keywords）
    coll = _get_skill_collection()
    result = coll.get(ids=[skill_name])
    metas = result.get("metadatas", [])
    full = skill
    if metas:
        meta = metas[0]
        full["trigger_keywords"] = json.loads(meta.get("trigger_keywords", "[]"))

    safe_name = re.sub(r'[\\/:*?"<>|]', "", skill_name).replace(" ", "-")
    dir_path = Path(output_dir or SKILL_EXPORT_DIR) / safe_name
    dir_path.mkdir(parents=True, exist_ok=True)

    desc = _build_description(full)

    md = f"""---
name: {safe_name}
description: {desc}
---

# {skill_name}

## 步骤

{skill.get('steps_sop', '(无)')}

## 注意事项

{skill.get('failure_modes', '暂无已知坑点')}

## 历史统计

- 成功次数: {skill.get('success_count', 0)}
- 平均评分: {skill.get('avg_score', 0)}
"""

    filepath = dir_path / "SKILL.md"
    filepath.write_text(md, encoding="utf-8")
    logger.info("Exported skill to %s", filepath)
    return str(filepath)


def export_all_skills_to_md(output_dir: str | None = None) -> list[str]:
    """把 ChromaDB 中所有 skill 导出为 SKILL.md 文件。

    Returns:
        成功导出的文件路径列表
    """
    skills = list_all_skills()
    paths = []
    for s in skills:
        path = export_skill_to_md(s["name"], output_dir=output_dir)
        if path:
            paths.append(path)
    return paths


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
