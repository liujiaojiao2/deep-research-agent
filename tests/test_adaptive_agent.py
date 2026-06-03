"""Adaptive Auto-Harness 单测：分组统计 + 退化检测 + 建议生成。"""
from __future__ import annotations

from src.agents.adaptive_agent import (
    _by_query_type,
    analyze_degradation,
    recommend_config,
)


def test_by_query_type_groups_scores():
    records = [
        {"query_type": "算法对比", "overall_score": 8.5},
        {"query_type": "算法对比", "overall_score": 8.0},
        {"query_type": "概念解释", "overall_score": 9.0},
    ]
    groups = _by_query_type(records)
    assert groups["算法对比"] == [8.5, 8.0]
    assert groups["概念解释"] == [9.0]


def test_by_query_type_empty():
    assert _by_query_type([]) == {}


def test_analyze_degradation_detects_drop(monkeypatch):
    monkeypatch.setattr(
        "src.agents.adaptive_agent._load_evolution_data",
        lambda: [
            {"query_type": "算法对比", "overall_score": 9.0},
            {"query_type": "算法对比", "overall_score": 9.0},
            {"query_type": "算法对比", "overall_score": 4.0},  # 暴跌
        ],
    )
    monkeypatch.setattr("src.agents.adaptive_agent.DEGRADATION_THRESHOLD", 0.8)
    out = analyze_degradation()
    assert "算法对比" in out["degraded_types"]
    g = out["groups"]["算法对比"]
    assert g["degrading"] is True


def test_analyze_degradation_healthy_when_stable(monkeypatch):
    monkeypatch.setattr(
        "src.agents.adaptive_agent._load_evolution_data",
        lambda: [
            {"query_type": "概念", "overall_score": 8.0},
            {"query_type": "概念", "overall_score": 8.5},
            {"query_type": "概念", "overall_score": 9.0},
        ],
    )
    out = analyze_degradation()
    assert "概念" not in out["degraded_types"]
    assert out["groups"]["概念"]["degrading"] is False


def test_analyze_degradation_skips_single_entry(monkeypatch):
    monkeypatch.setattr(
        "src.agents.adaptive_agent._load_evolution_data",
        lambda: [{"query_type": "X", "overall_score": 5.0}],
    )
    out = analyze_degradation()
    assert len(out["degraded_types"]) == 0


def test_analyze_degradation_empty(monkeypatch):
    monkeypatch.setattr(
        "src.agents.adaptive_agent._load_evolution_data",
        lambda: [],
    )
    out = analyze_degradation()
    assert out["groups"] == {}
    assert out["degraded_types"] == []


def test_recommend_config_for_contrast_type():
    rec = recommend_config("算法对比")
    assert any("对比" in s for s in rec["suggestions"])
    assert "rewoo" in rec["config_patch"].get("RESEARCHER_MODE", "")


def test_recommend_config_for_concept_type():
    rec = recommend_config("概念解释")
    assert any("概念" in s for s in rec["suggestions"])
    assert "react" in rec["config_patch"].get("RESEARCHER_MODE", "")


def test_recommend_config_for_unknown_type():
    rec = recommend_config("未知类型")
    assert rec["config_patch"]["QUALITY_EVAL_SAMPLES"] == "3"


def test_adaptive_report_includes_status(monkeypatch):
    monkeypatch.setattr(
        "src.agents.adaptive_agent._load_evolution_data",
        lambda: [
            {"query_type": "算法对比", "overall_score": 9.0},
            {"query_type": "算法对比", "overall_score": 9.0},
            {"query_type": "算法对比", "overall_score": 9.0},
        ],
    )
    from src.agents.adaptive_agent import adaptive_report

    report = adaptive_report()
    assert "算法对比" in report
    assert "🟢" in report
