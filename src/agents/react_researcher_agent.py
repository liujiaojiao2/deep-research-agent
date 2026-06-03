"""ReAct 版 Researcher —— 让 LLM 自主决定调哪些工具、调几次、何时停。

与 simple 版本 researcher_agent.py 的根本差异：
  · simple: 代码循环调 web_search → compress；LLM 不参与工具选择
  · react : LLM 在多轮对话里自主输出 tool_calls；LangGraph 执行后回填结果，LLM 再推理

外部接口与 simple 版本兼容：
  · 输入：state.research_brief / research_results
  · 输出：在 research_results 追加一条 {query, content, source}
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from src.model_router import get_llm_for
from src.state import ResearchResult, SupervisorState
from src.tools.tool_registry import get_all_tools


REACT_SYSTEM_PROMPT = """你是一位严谨的研究员。你需要根据研究简报，主动决定调用哪些工具来收集信息。

可用工具（**按优先级降序排列，请按此顺序考虑**）：
- local_knowledge_search：本地知识库（公司内部文档、项目笔记、复盘记录）——**最高优先级**
- wikipedia_search：概念定义、历史背景、人物简介——基础知识首选
- arxiv_search：算法原理、论文细节、技术机制——技术深度首选
- web_search：新闻、产品动态、博客解读——时效性首选
- python_calculator：需要精确数值/百分比计算时
- get_current_datetime：判断"最近/今年"等时效性表达时

工作原则：
1. **务必先调用一次 local_knowledge_search**：本地知识库可能包含公开网络上找不到的内部信息
2. 如果本地知识库返回为空或不相关，再用 wikipedia/arxiv/web 工具补充
3. 看简报里的子问题，针对**每个子问题**至少调用一种合适的工具
4. 同一主题尽量交叉验证，至少使用 2 种不同来源工具
5. 调用工具次数控制在 4-8 次，过多会浪费 token、过少信息不全
6. 工具调用结束后，输出一段结构化总结：
   - 关键事实分点列出
   - 每个事实后用 `(来源: 工具名/URL 或 本地文件名)` 标注
   - 涉及数字必须明确引用，不可估算
7. 不要在没调工具的情况下凭知识回答；不要重复调同一工具的同一参数

输出语言：与简报一致。
"""

# 防止 ReAct 死循环（理论上 LLM 决定停，但兜底必须有）
DEFAULT_RECURSION_LIMIT = 25


def build_react_agent(llm=None, tools=None):
    """构造可独立调用的 ReAct sub-agent。便于测试与单独跑实验。"""
    llm = llm or get_llm_for("research")
    tools = tools if tools is not None else get_all_tools()
    return create_react_agent(llm, tools=tools, prompt=REACT_SYSTEM_PROMPT)


def _extract_tool_usage(messages) -> list[str]:
    """从消息历史里抽出 LLM 这次跑实际调了哪些工具，便于可观测。"""
    used: list[str] = []
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            used.extend(tc.get("name", "?") for tc in m.tool_calls)
    return used


def _extract_final_answer(messages) -> str:
    """ReAct 最后一条 AI 消息（不带 tool_calls）即为模型的总结。"""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            return m.content or ""
    return ""


def react_researcher_node(
    state: SupervisorState,
    llm=None,
    tools=None,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> dict:
    """ReAct researcher 节点：把 brief 喂给 sub-agent，整理总结回写 state。"""
    agent = build_react_agent(llm=llm, tools=tools)

    brief = state.get("research_brief") or state.get("query", "")
    sub_input = {"messages": [HumanMessage(content=f"研究简报：\n{brief}")]}

    try:
        result = agent.invoke(sub_input, config={"recursion_limit": recursion_limit})
    except Exception as e:
        # 不让 sub-agent 异常打断主流程；写一条错误资料让 quality_eval 感知
        entry: ResearchResult = {
            "query": brief,
            "content": f"(ReAct researcher 异常: {e})",
            "source": "react_agent_error",
        }
        prev = list(state.get("research_results", []))
        prev.append(entry)
        return {"research_results": prev}

    messages = result.get("messages", [])
    summary = _extract_final_answer(messages) or "（ReAct researcher 未产生总结）"
    tools_used = _extract_tool_usage(messages)

    entry = {
        "query": brief,
        "content": summary,
        "source": f"react_agent(tools={','.join(tools_used) or 'none'})",
    }
    prev = list(state.get("research_results", []))
    prev.append(entry)
    return {"research_results": prev}
