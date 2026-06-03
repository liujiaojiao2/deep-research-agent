"""Quality Eval 单测：JSON 解析、容错、字段范围。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agents.quality_agent import _DEFAULT_SCORE, quality_eval_node


class _StubLLM:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, prompt):
        return SimpleNamespace(content=self.payload)


def test_parses_clean_json():
    llm = _StubLLM('{"accuracy":8.0,"completeness":7.0,"logic":9.0,"citation":6.5,"overall":7.6,"feedback":"OK"}')
    out = quality_eval_node({"query": "Q", "draft_report": "D"}, llm=llm)
    score = out["quality_score"]
    assert score["overall"] == 7.6
    assert score["feedback"] == "OK"


def test_parses_json_inside_code_fence():
    payload = '```json\n{"accuracy":7,"completeness":7,"logic":7,"citation":7,"overall":7.0,"feedback":"x"}\n```'
    llm = _StubLLM(payload)
    out = quality_eval_node({"query": "Q", "draft_report": "D"}, llm=llm)
    assert out["quality_score"]["overall"] == 7.0


def test_parses_json_with_leading_chatter():
    payload = "好的，以下是评分结果：\n{\"accuracy\":6,\"completeness\":6,\"logic\":6,\"citation\":6,\"overall\":6,\"feedback\":\"\"}"
    llm = _StubLLM(payload)
    out = quality_eval_node({"query": "Q", "draft_report": "D"}, llm=llm)
    assert out["quality_score"]["overall"] == 6.0


def test_fallback_when_no_json():
    llm = _StubLLM("这次没按 JSON 输出")
    out = quality_eval_node({"query": "Q", "draft_report": "D"}, llm=llm)
    score = out["quality_score"]
    assert score["overall"] == _DEFAULT_SCORE["overall"]
    assert "JSON 解析失败" in score["feedback"]


def test_clips_out_of_range_scores():
    llm = _StubLLM('{"accuracy":15,"completeness":-3,"logic":7,"citation":7,"overall":12,"feedback":""}')
    out = quality_eval_node({"query": "Q", "draft_report": "D"}, llm=llm)
    score = out["quality_score"]
    assert score["accuracy"] == 10.0
    assert score["completeness"] == 0.0
    assert score["overall"] == 10.0


@pytest.mark.parametrize("field", ["accuracy", "completeness", "logic", "citation", "overall"])
def test_all_score_fields_present(field):
    llm = _StubLLM('{"accuracy":8,"completeness":8,"logic":8,"citation":8,"overall":8,"feedback":""}')
    out = quality_eval_node({"query": "Q", "draft_report": "D"}, llm=llm)
    assert field in out["quality_score"]
    assert 0.0 <= out["quality_score"][field] <= 10.0
