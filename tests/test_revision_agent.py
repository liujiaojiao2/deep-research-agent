"""Revision Node 单测：补搜索分支、重写分支、状态重置、容错。"""
from __future__ import annotations

from types import SimpleNamespace

from src.agents.revision_agent import revision_node


class _SequenceLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        text = self.responses.pop(0) if self.responses else ""
        return SimpleNamespace(content=text)


def _fake_search(query, max_results=3):
    return [
        {"title": f"{query}-标题{i}", "content": f"{query} 内容 {i}", "url": f"https://e.com/{i}"}
        for i in range(max_results)
    ]


def _base_state():
    return {
        "query": "DeepSeek R1 技术原理",
        "draft_report": "旧报告内容...",
        "red_team_feedback": "缺少 RL 训练细节",
        "quality_score": {"overall": 5.0, "feedback": "完整性不足"},
        "research_results": [{"query": "old", "content": "旧资料", "source": "web_search"}],
        "iteration_count": 0,
    }


def test_no_supplement_path_skips_search():
    llm = _SequenceLLM([
        "[]",                # _extract_supplement_queries → 不补搜索
        "## 修订后报告\n...",  # revise
    ])

    calls = []

    def tracking_search(q, max_results=3):
        calls.append(q)
        return _fake_search(q, max_results)

    out = revision_node(_base_state(), llm=llm, search_fn=tracking_search)
    assert calls == []                                # 没有触发搜索
    assert out["draft_report"].startswith("## 修订后报告")
    assert out["iteration_count"] == 1
    assert out["quality_score"] == {}                 # 评分重置
    assert out["red_team_feedback"] == ""             # 反馈清空
    assert len(out["research_results"]) == 1          # 没新增


def test_supplement_path_runs_search_and_compress():
    llm = _SequenceLLM([
        '["DeepSeek R1 RLHF 训练", "GRPO 算法"]',  # supplement queries
        "- 压缩摘要要点",                            # compress_research
        "## 修订报告 V2",                           # revise
    ])
    out = revision_node(_base_state(), llm=llm, search_fn=_fake_search)
    assert out["draft_report"] == "## 修订报告 V2"
    assert out["iteration_count"] == 1
    # 新增一条 supplement 资料
    assert len(out["research_results"]) == 2
    new_entry = out["research_results"][-1]
    assert new_entry["source"] == "supplement"
    assert "DeepSeek R1 RLHF 训练" in new_entry["query"]
    assert "压缩摘要要点" in new_entry["content"]


def test_supplement_queries_capped_at_max():
    llm = _SequenceLLM([
        '["a", "b", "c", "d"]',  # 超过 MAX_SUPPLEMENT_QUERIES=2
        "压缩",
        "新稿",
    ])
    calls = []
    out = revision_node(
        _base_state(),
        llm=llm,
        search_fn=lambda q, max_results=3: (calls.append(q) or _fake_search(q, max_results)),
    )
    assert calls == ["a", "b"]
    assert out["iteration_count"] == 1


def test_invalid_supplement_json_falls_back_to_no_search():
    llm = _SequenceLLM([
        "无 JSON 输出",  # 不能解析 → 视为不补搜索
        "新稿",          # revise
    ])
    out = revision_node(_base_state(), llm=llm, search_fn=_fake_search)
    assert out["draft_report"] == "新稿"
    assert len(out["research_results"]) == 1  # 没有新增


def test_iteration_count_accumulates():
    state = _base_state()
    state["iteration_count"] = 2
    llm = _SequenceLLM(["[]", "稿 v3"])
    out = revision_node(state, llm=llm, search_fn=_fake_search)
    assert out["iteration_count"] == 3


def test_search_exception_does_not_crash():
    def boom(q, max_results=3):
        raise RuntimeError("net down")

    llm = _SequenceLLM(['["q1"]', "压缩", "新稿"])
    out = revision_node(_base_state(), llm=llm, search_fn=boom)
    # 异常被流程吞掉，写到给 compress 的 prompt 里，主链继续推进
    assert out["draft_report"] == "新稿"
    assert len(out["research_results"]) == 2
    # 第二条 LLM 调用是 compress，prompt 里应包含错误信息
    compress_prompt = llm.prompts[1]
    assert "net down" in compress_prompt
