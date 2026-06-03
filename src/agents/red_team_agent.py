"""Red Team Agent —— 对抗性审查。

把自己当成"找茬的审稿人"，从多个维度攻击初稿，把问题清单写到
state.red_team_feedback，供 quality_eval 和 revision 使用。

Phase 7.4 起：开启 Multi-Persona（RED_TEAM_PERSONAS > 1）时，
让 N 个独立 persona 并行评论同一报告，再 aggregate 成综合反馈。
每个 persona 的视角不同，覆盖盲区比单 critic 更全面。
"""
from __future__ import annotations

import os

from src.model_router import get_llm_for
from src.multi_sample import sample_multi_persona
from src.state import SupervisorState


_RED_TEAM_PROMPT = """你是一个极其严苛的学术审稿人，专门挑研究报告的毛病。

原始问题：{query}

待审查报告：
====
{draft}
====

请从以下维度做攻击性审查，并按指定格式输出：

审查维度：
1. 事实准确性：明显错误、未经证实的断言
2. 逻辑漏洞：推理链条断裂、跳跃性结论
3. 幻觉检测：模型编造、缺乏证据的陈述
4. 信息缺失：本应覆盖却完全没提的方面
5. 引用质量：来源不可信、过度依赖单一来源、缺少来源

输出格式（严格遵守）：
## 严重问题（必须修复）
- ...

## 一般问题（建议修复）
- ...

## 需要补充研究的方向
- ...

## 综合评价
一段不超过 150 字的总结。
"""


# Multi-Persona 设计：每个 persona 只关注一个维度，避免单 LLM 视野发散
_PERSONAS = [
    {
        "name": "事实核查师",
        "role": "你是事实核查师，专注于查证报告中的具体事实、数字、人名、日期、引用是否真实可考证。"
                "不关心结构和表达，只挑事实错误与未经证实的断言。",
    },
    {
        "name": "逻辑学家",
        "role": "你是逻辑学家，专注于推理链条是否完整、论证是否严密、有无跳跃结论或循环论证。"
                "不关心事实细节，只挑逻辑漏洞。",
    },
    {
        "name": "引用审计员",
        "role": "你是引用审计员，专注于报告的引用质量、来源权威性、引用充分性、是否过度依赖单一来源。"
                "不关心内容质量，只挑引用问题。",
    },
]


_PERSONA_PROMPT_TEMPLATE = """{persona_role}

原始问题：{{query}}

待审查报告：
====
{{draft}}
====

请只从你的专业视角列出 3-5 个最严重的问题（不超过 200 字），不要面面俱到。
"""


_AGGREGATOR_PROMPT = """你是 Red Team 团队的主审。下面是三位独立审稿人的评论，
请把它们融合成一份完整的红队反馈，去除重复、保留最尖锐的批评。

各审稿人评论：
{persona_views}

输出格式（严格遵守）：
## 严重问题（必须修复）
- ...

## 一般问题（建议修复）
- ...

## 需要补充研究的方向
- ...

## 综合评价
一段不超过 150 字的总结。
"""


def _personas_count() -> int:
    try:
        return max(1, int(os.getenv("RED_TEAM_PERSONAS", "1")))
    except ValueError:
        return 1


def red_team_node(state: SupervisorState, llm=None) -> dict:
    llm = llm or get_llm_for("red_team")
    query = state.get("query", "")
    draft = state.get("draft_report", "")

    n = _personas_count()
    if n > 1:
        # Multi-Persona 路径
        chosen = _PERSONAS[: min(n, len(_PERSONAS))]
        # 把 persona_role + query/draft 合成完整 prompt 模板
        target_template = _PERSONA_PROMPT_TEMPLATE.format(persona_role="{persona_role}")
        # 二次 format 把 query/draft 注入
        target_template = target_template.replace("{query}", query).replace("{draft}", draft)
        result = sample_multi_persona(
            personas=chosen,
            target_prompt_template=target_template,
            llm=llm,
            aggregator_prompt_template=_AGGREGATOR_PROMPT,
        )
        return {"red_team_feedback": result["aggregated"]}

    # 单 critic 路径（旧版行为，默认）
    prompt = _RED_TEAM_PROMPT.format(query=query, draft=draft)
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return {"red_team_feedback": content}
