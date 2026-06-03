"""把多条搜索结果压缩为结构化摘要。

接受 llm 参数以便测试时注入 mock；未传则用 config.get_llm() 兜底。
"""
from __future__ import annotations

from typing import List

from src.model_router import get_llm_for


def compress_research(
    results: List[dict],
    llm=None,
    max_tokens: int = 2000,
) -> str:
    """将搜索结果列表压缩为带来源标注的结构化摘要。

    空结果直接返回提示串，不调用 LLM。
    """
    if not results:
        return "（无搜索结果）"

    llm = llm or get_llm_for("research")  # compress 与 researcher 同源

    raw_text = "\n\n".join(
        f"来源: {r.get('url', '')}\n标题: {r.get('title', '')}\n内容: {r.get('content', '')}"
        for r in results
    )

    prompt = f"""请将以下搜索结果压缩为简洁的结构化摘要（不超过 {max_tokens} 字）：

{raw_text}

要求：
1. 保留关键事实和数据，去除重复
2. 每个要点末尾标注来源 URL
3. 使用 markdown 要点（- ）形式组织
4. 不要编造原文中不存在的信息
"""
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)
