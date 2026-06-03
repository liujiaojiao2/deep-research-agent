"""Human-in-the-loop 审核节点 —— 在 quality_eval 之后暂停等用户决策。

5 个决策选项及对 state 的影响：
  approve       → quality.overall = max(7.5, current)，supervisor 路由 final_report
  reject        → 设 overall = 3.0（低分），supervisor 路由 red_team 再修一轮
  force_final   → iteration_count = max_iterations，supervisor 走兜底 final_report
  edit_report   → 用户在 payload 里塞 draft_replacement；重置 quality_score
  custom_score  → 用户给 overall 一个具体分数

LangGraph 1.x 工作流：
  1. 节点调 interrupt({...}) → 主图暂停，把 payload 抛回客户端
  2. 客户端用 Command(resume=user_decision_dict) → 节点从 interrupt 处继续，拿到 user_decision_dict
  3. 节点据此修改 state 后返回
"""
from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from src.state import SupervisorState


VALID_DECISIONS = {"approve", "reject", "force_final", "edit_report", "custom_score"}


def _decision_to_state_update(decision: dict[str, Any], state: SupervisorState) -> dict:
    """把用户决策翻译成 state patch。decision 至少含 {'action': ...}。"""
    action = decision.get("action", "approve")
    if action not in VALID_DECISIONS:
        # 未知决策按 approve 处理，避免卡死
        action = "approve"

    quality = dict(state.get("quality_score") or {})
    iteration = int(state.get("iteration_count", 0))
    max_iter = int(state.get("max_iterations", 3))

    if action == "approve":
        # 抬到阈值之上，让 supervisor 路由 final_report
        quality["overall"] = max(float(quality.get("overall", 0.0)), 7.5)
        return {"quality_score": quality}

    if action == "reject":
        # 拉低分，但不达迭代上限 → supervisor 走 red_team
        quality["overall"] = 3.0
        quality["feedback"] = (quality.get("feedback") or "") + "\n[用户驳回：要求 red_team 再修一轮]"
        return {"quality_score": quality}

    if action == "force_final":
        # 把迭代计数推到上限，触发 supervisor 兜底
        return {"iteration_count": max(iteration, max_iter)}

    if action == "edit_report":
        new_draft = decision.get("draft", "")
        if not new_draft:
            return {}
        # 用户改了报告 → 清空评分让重新评估
        return {
            "draft_report": new_draft,
            "quality_score": {},
            "red_team_feedback": "",
        }

    if action == "custom_score":
        try:
            score = float(decision.get("overall", quality.get("overall", 7.0)))
        except (TypeError, ValueError):
            score = 7.0
        quality["overall"] = max(0.0, min(10.0, score))
        return {"quality_score": quality}

    return {}


def human_review_node(state: SupervisorState) -> dict:
    """暂停主图并把决策权交给用户。

    payload 给客户端足够信息做决策，但不要把整稿全部塞进去（防止序列化超大）。
    """
    quality = state.get("quality_score") or {}
    draft = state.get("draft_report", "") or ""
    red_fb = state.get("red_team_feedback", "") or ""

    payload = {
        "iteration": state.get("iteration_count", 0),
        "max_iterations": state.get("max_iterations", 3),
        "quality_score": {
            "accuracy": quality.get("accuracy"),
            "completeness": quality.get("completeness"),
            "logic": quality.get("logic"),
            "citation": quality.get("citation"),
            "overall": quality.get("overall"),
        },
        "quality_feedback": quality.get("feedback", "")[:300],
        "draft_preview": draft[:500],
        "draft_full_length": len(draft),
        "red_team_feedback_preview": red_fb[:300] if red_fb else "",
        "options": sorted(VALID_DECISIONS),
    }

    decision = interrupt(payload)
    # 客户端 resume 后，decision 应该是 {'action': '...', 其他字段}
    if isinstance(decision, str):
        decision = {"action": decision}
    if not isinstance(decision, dict):
        decision = {"action": "approve"}

    return _decision_to_state_update(decision, state)
