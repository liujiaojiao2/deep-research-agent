"""recall_episodic_memory 工具：让 ReAct 自主决定是否查"过去做过类似研究"。"""
from __future__ import annotations

import logging
from typing import List

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def recall_episodic_memory(query: str, top_k: int = 3) -> List[dict]:
    """检索本系统过去做过的相似研究（跨会话长期记忆）。

    适用场景（**比 web 工具优先级稍低**）：
    - 当前问题可能与过去研究类似，先看看有没有现成的工作或结论
    - 想了解"我们之前在这个领域做过什么"
    - 避免重复劳动；可参考过去研究的工具组合

    使用建议：
    - 用具体短语而非长句作为 query（例如 "GRPO 训练" 而非 "如何用 GRPO 训练大模型"）
    - top_k 建议 2-3；返回为空说明没有过往记录，请改用其他工具

    返回：[{summary, query, date, overall_score, similarity}]
        - summary：过往研究的摘要（带工具列表）
        - similarity：与当前 query 的余弦相似度
    """
    try:
        from src.memory import recall_episodic

        hits = recall_episodic(query=query, top_k=top_k)
        if not hits:
            return [{"summary": "无过往研究记录", "similarity": 0.0}]
        return hits
    except Exception as e:
        import traceback

        logger.warning("recall_episodic_memory failed: %s\n%s", e, traceback.format_exc())
        return [{"summary": f"memory 调用异常: {type(e).__name__}: {e}", "similarity": 0.0}]
