"""memory_archive 节点 —— final_report 之后自动归档 episodic + 抽取 preference。

设计：
- 失败不阻断主流程（memory 是增量价值，不应影响终稿生成）
- 同步执行（archive + extract）；如果未来量级大可改为后台异步
"""
from __future__ import annotations

import logging

from src.model_router import get_llm_for
from src.memory import archive_episodic, archive_preferences, extract_preferences
from src.state import SupervisorState

logger = logging.getLogger(__name__)


def memory_archive_node(state: SupervisorState, llm=None) -> dict:
    """归档 episodic + 抽取 preference。不修改主 state（除可观测字段）。"""
    archived = {}
    try:
        archived = archive_episodic(dict(state))
    except Exception as e:
        logger.warning("archive_episodic failed: %s", e)

    pref_count = 0
    try:
        llm = llm or get_llm_for("memory")
        preferences = extract_preferences(dict(state), llm)
        pref_count = archive_preferences(preferences)
    except Exception as e:
        logger.warning("extract/archive preferences failed: %s", e)

    # 不污染主 state；只返回可观测字段，方便 main.py 打印
    return {
        "memory_archived": bool(archived),
        "memory_preferences_added": pref_count,
    }
