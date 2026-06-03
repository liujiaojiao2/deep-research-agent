"""RAG 工具：local_knowledge_search。

把 src/rag.py 的 retrieve() 包装成 ReAct 可用的 @tool。
空库时友好返回（不抛异常），LLM 会自动选别的工具。
"""
from __future__ import annotations

import logging
from typing import List

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def local_knowledge_search(query: str, top_k: int = 3) -> List[dict]:
    """从本地知识库（公司内部文档/项目笔记/复盘记录）检索相关片段。

    适用场景（**最高优先级**）：
    - 涉及"我们/本项目/内部/团队/Phase X"等内部指代时
    - 涉及具体人名、内部决策、版本号、内部基准等专属信息时
    - 任何"在网上搜不到、只有团队成员知道"的问题

    使用建议：
    - 请优先调用本工具一次，再决定要不要查 web/wikipedia/arxiv
    - 如果本工具返回为空（"本地知识库为空"），再考虑其他工具

    参数：
    - query：检索关键词（自然语言句子比关键词效果更好）
    - top_k：返回片段数，默认 3，建议 ≤ 5

    返回：[{content, source, chunk_index, similarity}]
        - content：原文片段
        - source：源文件相对路径（例如 "02_grpo_internal_benchmark.md"）
        - similarity：与 query 的余弦相似度（0-1，越大越相关）
        - 空库时返回 [{"content": "本地知识库为空 ...", "source": "(empty)"}]
    """
    try:
        from src.rag import hybrid_retrieve

        hits = hybrid_retrieve(query=query, top_k_text=top_k, top_k_images=max(1, top_k // 2))
        if not hits:
            return [{
                "content": "本地知识库为空（尚未运行 scripts/ingest_knowledge.py 入库）；请改用其他工具。",
                "source": "(empty)",
                "similarity": 0.0,
            }]
        return hits
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        logger.warning("local_knowledge_search failed: %s\n%s", e, tb)
        return [{
            "content": f"本地知识库调用异常: {type(e).__name__}: {e}",
            "source": "(error)",
            "similarity": 0.0,
        }]
