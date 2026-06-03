"""model_router 单测：默认映射 / 环境变量覆盖 / fallback / 温度。"""
from __future__ import annotations

from src.model_router import (
    _DEFAULT_ROLE_MAP,
    _DEFAULT_TEMPERATURE,
    _resolve_model,
    _resolve_temperature,
    current_routing,
)


def test_resolve_model_default():
    assert _resolve_model("brief") == _DEFAULT_ROLE_MAP["brief"]


def test_resolve_model_env_override(monkeypatch):
    monkeypatch.setenv("MODEL_FOR_QUALITY", "deepseek-reasoner")
    assert _resolve_model("quality") == "deepseek-reasoner"


def test_resolve_model_unknown_role_uses_default(monkeypatch):
    monkeypatch.delenv("MODEL_FOR_DEFAULT", raising=False)
    assert _resolve_model("nonexistent") == _DEFAULT_ROLE_MAP["default"]


def test_resolve_model_unknown_role_with_env_default(monkeypatch):
    monkeypatch.setenv("MODEL_FOR_DEFAULT", "qwen-max")
    assert _resolve_model("brand_new_role") == "qwen-max"


def test_resolve_temperature_default():
    assert _resolve_temperature("draft") == _DEFAULT_TEMPERATURE["draft"]


def test_resolve_temperature_override_arg():
    assert _resolve_temperature("draft", override=0.99) == 0.99


def test_resolve_temperature_env(monkeypatch):
    monkeypatch.setenv("TEMP_FOR_QUALITY", "0.2")
    assert _resolve_temperature("quality") == 0.2


def test_resolve_temperature_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("TEMP_FOR_QUALITY", "not-a-float")
    assert _resolve_temperature("quality") == _DEFAULT_TEMPERATURE["quality"]


def test_current_routing_returns_all_roles():
    r = current_routing()
    for role in _DEFAULT_ROLE_MAP:
        assert role in r
        assert "model" in r[role]
        assert "temperature" in r[role]


def test_current_routing_reflects_env_change(monkeypatch):
    monkeypatch.setenv("MODEL_FOR_RED_TEAM", "custom-model")
    r = current_routing()
    assert r["red_team"]["model"] == "custom-model"
