"""Observability —— 给每次 run 留一条完整 trace。

设计：
- 不依赖外部 SaaS（不上 LangSmith），完全本地落盘
- Tracer 是一次性对象：run 开始时 new、run 期间 record_event、run 结束 dump
- 数据结构：每个事件 {ts, node, kind, info}；最终 render 成 markdown
- 工具用法：在 main.run_research 里 instantiate，喂 stream 事件

可观测维度：
- 节点时间线（开始时间、耗时）
- 工具调用记录（来源解析自 research_results[*].source）
- 评分变化
- 错误 / 警告
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TraceEvent:
    ts: float
    node: str
    kind: str  # "node_update" | "interrupt" | "error"
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tracer:
    """单次 run 的 trace 收集器。"""
    query: str
    run_id: str
    start_ts: float = field(default_factory=time.monotonic)
    events: list[TraceEvent] = field(default_factory=list)

    def record(self, node: str, update: dict, kind: str = "node_update") -> None:
        """喂入 stream 事件。update 是节点返回的 partial state。"""
        info = self._extract_info(node, update or {})
        self.events.append(TraceEvent(
            ts=time.monotonic() - self.start_ts,
            node=node,
            kind=kind,
            info=info,
        ))

    def record_error(self, node: str, msg: str) -> None:
        self.events.append(TraceEvent(
            ts=time.monotonic() - self.start_ts,
            node=node,
            kind="error",
            info={"msg": msg},
        ))

    # ---- 内部信息提取 ----

    @staticmethod
    def _extract_info(node: str, update: dict) -> dict:
        """根据 node 类型抽取关键字段，避免 dump 整个 update 太大。"""
        if node == "supervisor":
            return {"next_agent": update.get("next_agent", "")}
        if node == "brief_writer":
            return {"brief_len": len(update.get("research_brief", "") or "")}
        if node == "researcher":
            results = update.get("research_results") or []
            last_src = results[-1].get("source", "") if results else ""
            tools = Tracer._parse_tools(last_src)
            return {
                "result_count": len(results),
                "last_source": last_src,
                "tools_used": tools,
                "tools_count": len(tools),
            }
        if node == "draft_writer":
            return {"draft_len": len(update.get("draft_report", "") or "")}
        if node == "quality_eval":
            q = update.get("quality_score") or {}
            return {"overall": q.get("overall"), "feedback_preview": (q.get("feedback") or "")[:120]}
        if node == "red_team":
            return {"feedback_len": len(update.get("red_team_feedback", "") or "")}
        if node == "revision":
            return {
                "iteration": update.get("iteration_count"),
                "result_count": len(update.get("research_results") or []),
            }
        if node == "final_report":
            return {"final_len": len(update.get("final_report", "") or "")}
        if node == "memory_archive":
            return {
                "archived": update.get("memory_archived"),
                "preferences_added": update.get("memory_preferences_added", 0),
            }
        if node == "human_review":
            return {"applied_keys": list(update.keys())}
        return {}

    @staticmethod
    def _parse_tools(source: str) -> list[str]:
        """从 source 字段（如 'react_agent(tools=A,B)'）抽工具列表。"""
        m = re.search(r"tools=([^)]+)", source or "")
        if m:
            return [t.strip() for t in m.group(1).split(",") if t.strip()]
        return [source] if source else []

    # ---- 输出 ----

    def summary(self) -> dict:
        """统计：节点访问次数、总用时、工具命中、最终评分。"""
        node_counts: dict[str, int] = {}
        tool_counts: dict[str, int] = {}
        scores: list[float] = []
        final_overall = None
        for ev in self.events:
            node_counts[ev.node] = node_counts.get(ev.node, 0) + 1
            for t in ev.info.get("tools_used") or []:
                tool_counts[t] = tool_counts.get(t, 0) + 1
            if ev.node == "quality_eval":
                v = ev.info.get("overall")
                if isinstance(v, (int, float)):
                    scores.append(float(v))
                    final_overall = float(v)
        total = self.events[-1].ts if self.events else 0.0
        return {
            "run_id": self.run_id,
            "query": self.query,
            "total_seconds": round(total, 1),
            "node_visits": node_counts,
            "tool_call_counts": tool_counts,
            "quality_trajectory": scores,
            "final_overall": final_overall,
            "events_count": len(self.events),
        }

    def to_markdown(self) -> str:
        s = self.summary()
        md = [
            f"# Trace · {s['run_id']}\n",
            f"**Query**: {s['query']}\n",
            f"**Total time**: {s['total_seconds']} s\n",
            f"**Events**: {s['events_count']}\n",
            f"**Final overall**: {s['final_overall']}\n",
            "\n## Node visits\n",
            "| node | count |\n|---|---|\n",
            *[f"| {n} | {c} |\n" for n, c in sorted(s["node_visits"].items(), key=lambda x: -x[1])],
            "\n## Tool call counts\n",
            "| tool | count |\n|---|---|\n",
            *[f"| {t} | {c} |\n" for t, c in sorted(s["tool_call_counts"].items(), key=lambda x: -x[1])],
        ]
        if s["quality_trajectory"]:
            md.append("\n## Quality trajectory\n")
            md.append(" → ".join(str(v) for v in s["quality_trajectory"]) + "\n")
        md.append("\n## Event timeline\n")
        md.append("| ts(s) | node | kind | info |\n|---|---|---|---|\n")
        for ev in self.events:
            info_str = json.dumps(ev.info, ensure_ascii=False)[:120]
            md.append(f"| {ev.ts:.2f} | {ev.node} | {ev.kind} | `{info_str}` |\n")
        return "".join(md)

    def dump(self, save_dir: Path) -> Path:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"trace_{self.run_id}.md"
        out.write_text(self.to_markdown(), encoding="utf-8")
        return out
