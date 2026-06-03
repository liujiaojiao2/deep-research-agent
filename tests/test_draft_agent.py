"""Draft Agent 单测：mock LLM，验证两节点输入输出形状。"""
from __future__ import annotations

from types import SimpleNamespace

from src.agents.draft_agent import write_draft_report, write_research_brief


class _StubLLM:
    def __init__(self, payload: str):
        self.payload = payload
        self.last_prompt: str | None = None

    def invoke(self, prompt: str):
        self.last_prompt = prompt
        return SimpleNamespace(content=self.payload)


def test_write_research_brief_returns_brief_key():
    llm = _StubLLM("子问题1\n子问题2\n关键词A 关键词B\n报告框架...")
    out = write_research_brief({"query": "2025 大模型趋势"}, llm=llm)
    assert "research_brief" in out
    assert isinstance(out["research_brief"], str)
    assert len(out["research_brief"]) > 0
    assert "2025 大模型趋势" in llm.last_prompt


def test_write_draft_report_uses_research_and_brief():
    llm = _StubLLM("# 报告\n## 摘要\n...")
    state = {
        "query": "DeepSeek R1 技术原理",
        "research_brief": "子问题列表 + 关键词",
        "research_results": [
            {"query": "q1", "content": "MCTS 训练细节", "source": "web"},
            {"query": "q2", "content": "RLHF 步骤", "source": "web"},
        ],
    }
    out = write_draft_report(state, llm=llm)
    assert "draft_report" in out
    assert out["draft_report"].startswith("# 报告")
    # prompt 必须把研究资料和 brief 一起塞入
    assert "MCTS 训练细节" in llm.last_prompt
    assert "RLHF 步骤" in llm.last_prompt
    assert "子问题列表" in llm.last_prompt


def test_write_draft_report_handles_empty_research():
    llm = _StubLLM("空研究报告")
    out = write_draft_report({"query": "X", "research_results": []}, llm=llm)
    assert out["draft_report"] == "空研究报告"
    assert "（无可用资料）" in llm.last_prompt
