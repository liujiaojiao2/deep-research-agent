"""multi_sample 单测 + quality/red_team 在开启 7.4 模式下的回归。"""
from __future__ import annotations

from types import SimpleNamespace

from src.agents.quality_agent import quality_eval_node
from src.agents.red_team_agent import red_team_node
from src.multi_sample import (
    _median,
    _parse_json_obj,
    _variance,
    sample_json_scores,
    sample_multi_persona,
)


class _ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        text = self.responses.pop(0) if self.responses else ""
        return SimpleNamespace(content=text)


# ---------- helpers ----------

def test_median_and_variance():
    assert _median([1, 2, 3]) == 2.0
    assert _median([1, 2, 3, 4]) == 2.5
    assert _variance([5, 5, 5]) == 0.0
    assert _variance([1]) == 0.0
    # 方差 > 0
    assert _variance([1, 2, 3, 4]) > 0


def test_parse_json_obj_clean():
    assert _parse_json_obj('{"a":1}') == {"a": 1}


def test_parse_json_obj_with_noise():
    assert _parse_json_obj("好的:\n{\"a\":1}\nover") == {"a": 1}


# ---------- sample_json_scores ----------

def test_sample_json_scores_median_filters_outlier():
    llm = _ScriptedLLM([
        '{"acc":8,"overall":8,"feedback":"good"}',
        '{"acc":8,"overall":8,"feedback":"good"}',
        '{"acc":2,"overall":2,"feedback":"bad"}',  # outlier
    ])
    out = sample_json_scores("p", ["acc", "overall"], llm, n_samples=3)
    assert out["aggregated"]["overall"] == 8.0
    assert out["aggregated"]["acc"] == 8.0
    assert out["variance_per_field"]["overall"] > 0


def test_sample_json_scores_handles_parse_failure():
    llm = _ScriptedLLM([
        "not json",
        '{"acc":7,"overall":7,"feedback":"ok"}',
    ])
    out = sample_json_scores("p", ["acc", "overall"], llm, n_samples=2, default_score=5.0)
    # 一次解析失败 → fallback 5.0；一次成功 7 → median=6
    assert out["aggregated"]["overall"] == 6.0


def test_sample_json_scores_clips_range():
    llm = _ScriptedLLM([
        '{"acc":15,"overall":15,"feedback":"x"}',  # 超范围
        '{"acc":-5,"overall":-5,"feedback":"x"}',  # 负数
        '{"acc":8,"overall":8,"feedback":"x"}',
    ])
    out = sample_json_scores("p", ["acc", "overall"], llm, n_samples=3)
    # 裁剪后 [10, 0, 8] → median=8
    assert out["aggregated"]["overall"] == 8.0


def test_sample_json_scores_single_sample():
    llm = _ScriptedLLM(['{"acc":7.5,"overall":7.5,"feedback":"f"}'])
    out = sample_json_scores("p", ["acc", "overall"], llm, n_samples=1)
    assert out["aggregated"]["overall"] == 7.5
    assert out["variance_per_field"]["overall"] == 0.0


# ---------- sample_multi_persona ----------

def test_multi_persona_collects_views_and_aggregates():
    llm = _ScriptedLLM([
        "事实有错:\n- 张三非董事长",
        "逻辑跳跃:\n- 第二段推论无依据",
        "引用单一:\n- 仅 1 个 URL",
        "## 综合反馈\n- 张三非董事长\n- 第二段推论无依据\n- 引用单一",
    ])
    personas = [
        {"name": "事实", "role": "你是事实核查师"},
        {"name": "逻辑", "role": "你是逻辑学家"},
        {"name": "引用", "role": "你是引用审计员"},
    ]
    out = sample_multi_persona(
        personas=personas,
        target_prompt_template="{persona_role}\n请评论",
        llm=llm,
        aggregator_prompt_template="aggregate:\n{persona_views}",
    )
    assert len(out["views"]) == 3
    assert "综合反馈" in out["aggregated"]


def test_multi_persona_without_aggregator():
    llm = _ScriptedLLM(["视角 A", "视角 B"])
    out = sample_multi_persona(
        personas=[{"name": "A", "role": "你是 A"}, {"name": "B", "role": "你是 B"}],
        target_prompt_template="{persona_role}\n说话",
        llm=llm,
        aggregator_prompt_template=None,
    )
    assert "视角 A" in out["aggregated"]
    assert "视角 B" in out["aggregated"]


def test_multi_persona_handles_persona_exception():
    class _BoomLLM:
        def invoke(self, p):
            raise RuntimeError("network")

    out = sample_multi_persona(
        personas=[{"name": "A", "role": "..."}],
        target_prompt_template="{persona_role}\nx",
        llm=_BoomLLM(),
    )
    assert "error" in out["views"][0]["view"].lower()


# ---------- quality_eval_node 集成（self-consistency 路径） ----------

def test_quality_eval_self_consistency_path(monkeypatch):
    monkeypatch.setenv("QUALITY_EVAL_SAMPLES", "3")
    llm = _ScriptedLLM([
        '{"accuracy":8,"completeness":8,"logic":8,"citation":8,"overall":8,"feedback":"good"}',
        '{"accuracy":8,"completeness":8,"logic":8,"citation":8,"overall":8,"feedback":"good"}',
        '{"accuracy":2,"completeness":2,"logic":2,"citation":2,"overall":2,"feedback":"bad"}',
    ])
    out = quality_eval_node({"query": "Q", "draft_report": "D"}, llm=llm)
    s = out["quality_score"]
    # median 抗 outlier
    assert s["overall"] == 8.0
    assert s["accuracy"] == 8.0
    # 方差信息应附加到 feedback
    assert "self-consistency" in s["feedback"]


def test_quality_eval_single_sample_default(monkeypatch):
    monkeypatch.delenv("QUALITY_EVAL_SAMPLES", raising=False)
    llm = _ScriptedLLM([
        '{"accuracy":7,"completeness":7,"logic":7,"citation":7,"overall":7,"feedback":"ok"}',
    ])
    out = quality_eval_node({"query": "Q", "draft_report": "D"}, llm=llm)
    assert out["quality_score"]["overall"] == 7.0
    assert "self-consistency" not in out["quality_score"]["feedback"]


# ---------- red_team_node 集成（multi-persona 路径） ----------

def test_red_team_multi_persona_path(monkeypatch):
    monkeypatch.setenv("RED_TEAM_PERSONAS", "3")
    llm = _ScriptedLLM([
        "事实问题 A",
        "逻辑问题 B",
        "引用问题 C",
        "## 综合反馈\n- A\n- B\n- C",
    ])
    out = red_team_node({"query": "Q", "draft_report": "D"}, llm=llm)
    assert "综合反馈" in out["red_team_feedback"]


def test_red_team_single_critic_default(monkeypatch):
    monkeypatch.delenv("RED_TEAM_PERSONAS", raising=False)
    llm = _ScriptedLLM(["## 严重问题\n- x"])
    out = red_team_node({"query": "Q", "draft_report": "D"}, llm=llm)
    assert "严重问题" in out["red_team_feedback"]
