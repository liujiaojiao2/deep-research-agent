"""Trace Protocol (H0-H3 审计标准) —— JSONL + hash chain。

升级 Tracer (Phase 7.3) 从"人读 markdown"到"机器可解析 + 防篡改"。

核心:
- jsonl_serialize(events): 把 Tracer.events 转成 JSONL 字节流
- hash_chain(events):  为每条 event 计算 SHA256 链（prev_hash + hash）
- validate(jsonl_bytes): 验证 JSONL 的 hash chain 完整性

H0-H3 标准: 每条 event 含 seq/ts/node/kind/info/prev_hash/hash，
prev_hash 指向前一条的 hash，形成不可篡改的审计链。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _event_to_canonical(event: Any, seq: int, prev_hash: str) -> dict:
    """把 Tracer TraceEvent 转为标准化的可审计 dict。"""
    info = json.dumps(getattr(event, "info", {}), sort_keys=True, ensure_ascii=False)
    row = {
        "seq": seq,
        "ts": round(getattr(event, "ts", 0.0), 4),
        "node": str(getattr(event, "node", "")),
        "kind": str(getattr(event, "kind", "")),
        "info": info,
        "prev_hash": prev_hash,
    }
    payload = f"{seq}|{row['ts']}|{row['node']}|{row['kind']}|{info}|{prev_hash}"
    row["hash"] = _sha256(payload)
    return row


def jsonl_serialize(events: list, run_id: str = "") -> bytes:
    """把事件列表转为 JSONL 字节流（每行一条 JSON + hash chain）。"""
    lines: list[str] = []
    # header
    header = {"run_id": run_id, "format": "H0-H3-trace-v1", "total_events": len(events)}
    lines.append(json.dumps(header, ensure_ascii=False))

    prev_hash = "0000000000000000"  # genesis hash
    for i, ev in enumerate(events):
        row = _event_to_canonical(ev, seq=i + 1, prev_hash=prev_hash)
        lines.append(json.dumps(row, ensure_ascii=False))
        prev_hash = row["hash"]

    return "\n".join(lines).encode("utf-8")


def validate(jsonl_bytes: bytes) -> tuple[bool, str]:
    """验证 JSONL 的 hash chain 完整性。返回 (valid, reason)。"""
    lines = jsonl_bytes.decode("utf-8").strip().split("\n")
    if len(lines) < 2:
        return False, "JSONL 至少需要 header + 1 条 event"

    prev_hash = "0000000000000000"
    for idx, line in enumerate(lines):
        data = json.loads(line)
        if idx == 0:
            if data.get("format") != "H0-H3-trace-v1":
                return False, f"未知格式: {data.get('format')}"
            continue

        expected_prev = data.get("prev_hash", "")
        if expected_prev != prev_hash:
            return False, (
                f"seq={data.get('seq')} hash chain 断裂: "
                f"expected prev={prev_hash} got={expected_prev}"
            )

        # 重算 hash
        payload = (
            f"{data['seq']}|{data['ts']}|{data['node']}|{data['kind']}"
            f"|{data['info']}|{prev_hash}"
        )
        recalculated = _sha256(payload)
        if recalculated != data.get("hash", ""):
            return False, f"seq={data['seq']} hash 不匹配: {recalculated} != {data.get('hash')}"

        prev_hash = data["hash"]

    return True, f"验证通过: {len(lines) - 1} 条 event, hash chain 完整"


def jsonl_to_dicts(jsonl_bytes: bytes) -> list[dict[str, Any]]:
    """JSONL → list[dict]（略过 header 行）。"""
    lines = jsonl_bytes.decode("utf-8").strip().split("\n")
    return [json.loads(line) for line in lines[1:]]  # skip header
