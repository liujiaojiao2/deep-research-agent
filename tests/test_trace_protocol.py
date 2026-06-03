"""Trace Protocol 单测：序列化 / hash chain / 验证 / 篡改检测。"""
from __future__ import annotations

import json
from types import SimpleNamespace

from src.trace_protocol import (
    _event_to_canonical,
    _sha256,
    jsonl_serialize,
    jsonl_to_dicts,
    validate,
)


def _event(ts=1.0, node="supervisor", kind="node_update", info=None):
    if info is None:
        info = {"next_agent": "brief_writer"}
    return SimpleNamespace(ts=ts, node=node, kind=kind, info=info)


def test_sha256_deterministic():
    assert _sha256("hello") == _sha256("hello")
    assert _sha256("hello") != _sha256("world")


def test_event_to_canonical_fields():
    ev = _event(ts=0.5, node="researcher")
    row = _event_to_canonical(ev, seq=1, prev_hash="0000000000000000")
    assert row["seq"] == 1
    assert row["ts"] == 0.5
    assert row["node"] == "researcher"
    assert row["prev_hash"] == "0000000000000000"
    assert "hash" in row


def test_jsonl_serialize_produces_header_plus_events():
    events = [_event(ts=0.1), _event(ts=0.2)]
    data = jsonl_serialize(events, run_id="r1")
    lines = data.decode("utf-8").strip().split("\n")
    assert len(lines) == 3  # header + 2 events
    # header
    header = json.loads(lines[0])
    assert header["run_id"] == "r1"
    assert header["total_events"] == 2


def test_jsonl_validate_passes_on_valid_chain():
    events = [_event(ts=0.1), _event(ts=0.2, node="draft_writer")]
    data = jsonl_serialize(events, run_id="r1")
    ok, reason = validate(data)
    assert ok is True
    assert "验证通过" in reason


def test_jsonl_validate_detects_tampered_info():
    events = [_event(ts=0.1)]
    data = jsonl_serialize(events, run_id="r1")
    # 篡改: 改第一条 event 的 info
    lines = data.decode("utf-8").strip().split("\n")
    row = json.loads(lines[1])
    row["info"] = '{"tampered": true}'
    lines[1] = json.dumps(row, ensure_ascii=False)
    tampered = "\n".join(lines).encode("utf-8")

    ok, reason = validate(tampered)
    assert ok is False
    assert "hash 不匹配" in reason


def test_jsonl_validate_detects_broken_chain():
    events = [_event(ts=0.1), _event(ts=0.2)]
    data = jsonl_serialize(events, run_id="r1")
    lines = data.decode("utf-8").strip().split("\n")
    # 篡改: 把第2条的 prev_hash 改成错误值
    row = json.loads(lines[2])
    row["prev_hash"] = "deadbeef"
    lines[2] = json.dumps(row, ensure_ascii=False)
    tampered = "\n".join(lines).encode("utf-8")

    ok, reason = validate(tampered)
    assert ok is False
    assert "hash chain 断裂" in reason


def test_jsonl_to_dicts():
    events = [_event(ts=0.5)]
    data = jsonl_serialize(events, run_id="x")
    dicts = jsonl_to_dicts(data)
    assert len(dicts) == 1
    assert dicts[0]["node"] == "supervisor"


def test_tracer_to_jsonl(tmp_path):
    from src.observability import Tracer

    t = Tracer(query="Q", run_id="r1")
    t.record("supervisor", {"next_agent": "brief_writer"})
    t.record("researcher", {"research_results": [{"source": "web_search"}]})

    jsonl_data = t.to_jsonl()
    ok, reason = validate(jsonl_data)
    assert ok is True

    # dump 应该同时写 md + jsonl
    out = t.dump(tmp_path)
    assert out.exists()
    jsonl_path = tmp_path / "trace_r1.jsonl"
    assert jsonl_path.exists()
    ok2, _ = validate(jsonl_path.read_bytes())
    assert ok2 is True
