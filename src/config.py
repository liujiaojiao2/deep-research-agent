"""集中读取环境变量与构造 LLM 客户端。所有 Agent 通过 get_llm() 获取共享实例。"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


class Settings:
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    SEARCH_PROVIDER: str = os.getenv("SEARCH_PROVIDER", "duckduckgo").lower()
    # researcher 模式：react = LLM 自主选工具；simple = 代码循环调 DDG
    RESEARCHER_MODE: str = os.getenv("RESEARCHER_MODE", "react").lower()


settings = Settings()


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    if not settings.DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY 未设置，请检查 .env 文件")
    return ChatOpenAI(
        base_url=settings.DEEPSEEK_BASE_URL,
        model=settings.DEEPSEEK_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        temperature=temperature,
    )
