"""Model Router —— 按角色选最合适的模型。

设计目标：
- 把"调哪个模型"的决策从 agent 代码里抽出来，集中在 router
- 通过环境变量配置，**改模型不改代码**
- 不可用时优雅 fallback 到默认模型
- 完全兼容旧 `get_llm()` 调用（仍然返回默认模型）

角色枚举（与项目里的 agent 节点对齐）：
    brief    : brief_writer
    research : researcher
    draft    : draft_writer
    quality  : quality_eval（含 self-consistency 多次采样）
    red_team : red_team
    revision : revision
    final    : final_report
    judge    : eval.judge（独立第三方评估）
    memory   : memory_archive 抽取偏好
    planner  : rewoo_planner

设计哲学：
- 推理/批判类 → 用更强但更贵的模型（如 deepseek-reasoner）
- 生成/执行类 → 用快且便宜的模型（如 deepseek-chat）

环境变量约定：
    MODEL_FOR_<ROLE>=<model_name>   # 例如 MODEL_FOR_QUALITY=deepseek-reasoner
    MODEL_FOR_DEFAULT=<model_name>  # 兜底默认
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

from langchain_openai import ChatOpenAI

from src.config import settings

logger = logging.getLogger(__name__)


# 默认角色 → 模型映射（可被环境变量覆盖）
_DEFAULT_ROLE_MAP: dict[str, str] = {
    "brief": "deepseek-chat",
    "research": "deepseek-chat",
    "draft": "deepseek-chat",
    "quality": "deepseek-chat",     # 默认仍用 chat，用户可改 deepseek-reasoner
    "red_team": "deepseek-chat",    # 同上
    "revision": "deepseek-chat",
    "final": "deepseek-chat",
    "judge": "deepseek-chat",
    "memory": "deepseek-chat",
    "planner": "deepseek-chat",
    "default": "deepseek-chat",
}


# 默认温度（推理类调到 0，生成类略高）
_DEFAULT_TEMPERATURE: dict[str, float] = {
    "brief": 0.3,
    "research": 0.3,
    "draft": 0.5,         # 写作更需要表达多样性
    "quality": 0.0,       # 评分要求一致性
    "red_team": 0.0,      # 批判要求严格
    "revision": 0.4,
    "final": 0.4,
    "judge": 0.0,
    "memory": 0.0,
    "planner": 0.0,       # ReWOO 规划要稳定可解析
    "default": 0.3,
}


def _resolve_model(role: str) -> str:
    """解析 role → 实际模型名（环境变量优先）。"""
    env_key = f"MODEL_FOR_{role.upper()}"
    if env_val := os.getenv(env_key, "").strip():
        return env_val
    env_default = os.getenv("MODEL_FOR_DEFAULT", "").strip()
    if env_default and role not in _DEFAULT_ROLE_MAP:
        return env_default
    return _DEFAULT_ROLE_MAP.get(role, _DEFAULT_ROLE_MAP["default"])


def _resolve_temperature(role: str, override: Optional[float] = None) -> float:
    if override is not None:
        return override
    env_key = f"TEMP_FOR_{role.upper()}"
    if env_val := os.getenv(env_key, "").strip():
        try:
            return float(env_val)
        except ValueError:
            pass
    return _DEFAULT_TEMPERATURE.get(role, _DEFAULT_TEMPERATURE["default"])


@lru_cache(maxsize=32)
def _build_llm_cached(model: str, temperature: float) -> ChatOpenAI:
    """同 (model, temperature) 复用同一个 client。"""
    if not settings.DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY 未设置，请检查 .env 文件")
    return ChatOpenAI(
        base_url=settings.DEEPSEEK_BASE_URL,
        model=model,
        api_key=settings.DEEPSEEK_API_KEY,
        temperature=temperature,
    )


def get_llm_for(role: str, temperature: Optional[float] = None) -> ChatOpenAI:
    """按角色返回 LLM 实例。失败回退默认。"""
    try:
        model = _resolve_model(role)
        temp = _resolve_temperature(role, override=temperature)
        return _build_llm_cached(model, temp)
    except Exception as e:
        logger.warning("get_llm_for(%s) failed (%s) → fallback default", role, e)
        return _build_llm_cached(
            _DEFAULT_ROLE_MAP["default"],
            _DEFAULT_TEMPERATURE["default"] if temperature is None else temperature,
        )


def current_routing() -> dict[str, dict]:
    """供 observability / debug 用：返回当前每个角色用哪个模型 + 温度。"""
    return {
        role: {"model": _resolve_model(role), "temperature": _resolve_temperature(role)}
        for role in _DEFAULT_ROLE_MAP
    }
