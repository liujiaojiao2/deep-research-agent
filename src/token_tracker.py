"""Token 追踪模块 —— 统计每次 LLM 调用的 token 消耗与成本。

用法：
    tracker = TokenTracker()
    llm = tracker.wrap(original_llm, agent="draft_writer")
    response = llm.invoke(prompt)  # 自动累加

所有 LLM 调用共享同一个全局 tracker 实例，通过 agent 标签区分来源。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

# DeepSeek 官方定价 (¥/百万 token), 2026
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "deepseek-chat": (1.0, 2.0),       # input ¥1, output ¥2
    "deepseek-reasoner": (4.0, 16.0),  # input ¥4, output ¥16 (含思维链)
    "deepseek-v4-flash": (0.5, 1.0),   # 估计
    "deepseek-v4-pro": (2.0, 8.0),     # 估计
}


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class CallRecord:
    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    prompt_preview: str


@dataclass
class AgentBreakdown:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cost_yuan: float = 0.0


class TokenTracker:
    """全局单例，线程安全。"""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.total = TokenUsage()
            self.records: list[CallRecord] = []
            self._agent_breakdowns: dict[str, AgentBreakdown] = {}

    def record(self, agent: str, model: str, input_tokens: int,
               output_tokens: int, prompt_preview: str = "") -> None:
        input_price, output_price = MODEL_PRICING.get(model, (1.0, 2.0))
        cost = (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price

        with self._lock:
            self.total.input_tokens += input_tokens
            self.total.output_tokens += output_tokens
            self.total.total_tokens += input_tokens + output_tokens
            self.records.append(CallRecord(
                agent=agent, model=model,
                input_tokens=input_tokens, output_tokens=output_tokens,
                prompt_preview=prompt_preview,
            ))

            if agent not in self._agent_breakdowns:
                self._agent_breakdowns[agent] = AgentBreakdown()
            bd = self._agent_breakdowns[agent]
            bd.input_tokens += input_tokens
            bd.output_tokens += output_tokens
            bd.calls += 1
            bd.cost_yuan += cost

    def install(self) -> None:
        """Monkey-patch ChatOpenAI.invoke at the class level。

        所有 ChatOpenAI 实例（含 bind_tools/ReAct 内部调用）都会自动追踪。
        无需逐个实例包装，不会破坏 LangChain 的类型检查。
        """
        from langchain_openai import ChatOpenAI

        tracker = self
        _original_invoke = ChatOpenAI.invoke

        def _tracked_invoke(_self, prompt, *args, **kwargs):
            response = _original_invoke(_self, prompt, *args, **kwargs)
            usage = getattr(response, "usage_metadata", None) or {}
            inp = int(usage.get("input_tokens", 0))
            out = int(usage.get("output_tokens", 0))
            if inp or out:
                model = getattr(_self, "model_name", "unknown")
                preview = (prompt if isinstance(prompt, str) else str(prompt))[:80]
                tracker.record("llm", model, inp, out, preview)
            return response

        ChatOpenAI.invoke = _tracked_invoke

    @property
    def agent_breakdowns(self) -> dict[str, AgentBreakdown]:
        with self._lock:
            return dict(self._agent_breakdowns)

    @property
    def total_cost_yuan(self) -> float:
        return sum(bd.cost_yuan for bd in self._agent_breakdowns.values())


# 全局单例
_global_tracker: Optional[TokenTracker] = None
_tracker_lock = threading.Lock()


def get_tracker() -> TokenTracker:
    global _global_tracker
    if _global_tracker is None:
        with _tracker_lock:
            if _global_tracker is None:
                _global_tracker = TokenTracker()
    return _global_tracker


def reset_tracker() -> None:
    global _global_tracker
    _global_tracker = None
