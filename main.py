"""DeepResearch Agent 入口。

用法：
    uv run python main.py "你的研究问题" [--max-iter 3] [--interactive]

设计要点：
- 同步流式跑图（LangGraph compile().stream），把每个节点的输出实时打印
- 终态 final_report 落盘到 outputs/report_{timestamp}.md
- --interactive 启用 HITL：每次 quality_eval 后暂停等用户决策
"""
from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt

from src import cache, security
from src.graph import build_main_graph
from src.observability import Tracer
from src.state import SupervisorState

console = Console()


def _initial_state(query: str, max_iter: int) -> SupervisorState:
    return {
        "query": query,
        "research_brief": "",
        "research_results": [],
        "draft_report": "",
        "final_report": "",
        "quality_score": {},
        "red_team_feedback": "",
        "iteration_count": 0,
        "max_iterations": max_iter,
        "next_agent": "",
        "is_complete": False,
        "messages": [],
    }


def run_research(
    query: str,
    max_iter: int = 3,
    save_dir: Optional[Path] = None,
    recursion_limit: int = 50,
    interactive: bool = False,
    enrich_mindmap: bool = False,
    thread_id: Optional[str] = None,
    resume: bool = False,
) -> str:
    """跑完整流程，返回 final_report；副作用：把 markdown 写到 outputs/。"""
    save_dir = save_dir or Path(__file__).parent / "outputs"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Phase 7.7: 入口注入检测 + 消毒
    is_sus, hits = security.detect_injection(query)
    if is_sus:
        console.print(f"[bold yellow]⚠️ 检测到 {len(hits)} 条潜在 prompt 注入模式（已隔离标记，仍继续执行）[/bold yellow]")
    query = security.sanitize_user_input(query)

    console.rule(f"[bold cyan]🔍 开始研究：{query[:100]}[/bold cyan]")

    # Phase 7.3: 缓存命中检查（非 interactive 才命中；resume 时跳过，直接续跑 checkpoint）
    if not interactive and not resume:
        cached = cache.lookup(query)
        if cached is not None:
            console.print(
                f"[bold green]💾 缓存命中[/bold green]  "
                f"similarity={cached['similarity']}  "
                f"overall={cached['overall']}  "
                f"date={cached['date']}"
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"report_{run_id}_cached.md"
            out_path.write_text(cached["final_report"], encoding="utf-8")
            console.print(f"📄 报告（来自缓存）保存至: {out_path}")
            _write_mindmap_sidecars(out_path, cached["final_report"], query, enrich_mindmap)
            return cached["final_report"]

    graph = build_main_graph(interactive=interactive)
    state = _initial_state(query, max_iter)

    tid = thread_id or f"research-{uuid.uuid4().hex[:8]}"
    config = {"recursion_limit": recursion_limit, "configurable": {"thread_id": tid}}
    console.print(f"[dim]thread_id={tid}（状态持久化到 .checkpoints/）[/dim]")
    if interactive:
        console.print("[dim]Interactive 模式：每次 quality_eval 后会暂停等你决策。[/dim]")

    tracer = Tracer(query=query, run_id=run_id)

    final_state = state
    # resume：输入传 None，让 LangGraph 从该 thread_id 的最近 checkpoint 续跑
    next_input = None if resume else state
    if resume:
        existing = graph.get_state(config)
        if not (existing and existing.values):
            console.print(
                f"[bold red]✗ 找不到 thread_id={tid} 的 checkpoint，无法续跑。[/bold red]"
            )
            sys.exit(1)
        console.print(f"[bold]↩️  resume：从 thread_id={tid} 的最近 checkpoint 续跑[/bold]")
    while True:
        interrupted = False
        for event in graph.stream(next_input, config=config):
            for node_name, update in event.items():
                if node_name == "__interrupt__":
                    # update 是 tuple/list of Interrupt 对象
                    payload = _extract_interrupt_payload(update)
                    decision = _prompt_user_for_decision(payload)
                    next_input = Command(resume=decision)
                    interrupted = True
                    break
                _print_node_update(node_name, update)
                tracer.record(node_name, update or {})
                final_state = {**final_state, **(update or {})}
            if interrupted:
                break
        if not interrupted:
            break  # stream 自然结束

    # 从 checkpointer 拿权威最终 state（两模式都挂了持久化；interactive/resume 尤其需要）
    snapshot = graph.get_state(config)
    if snapshot and snapshot.values:
        final_state = snapshot.values

    if not final_state.get("final_report"):
        console.print("[red]⚠️ 流程结束但 final_report 为空，请检查日志。[/red]")
        return ""

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"report_{run_id}.md"
    out_path.write_text(final_state["final_report"], encoding="utf-8")
    _write_mindmap_sidecars(out_path, final_state["final_report"], query, enrich_mindmap)

    # Phase 7.7: 出口泄露检测（仅警示，不阻断）
    leak_sus, leak_hits = security.detect_prompt_leakage(final_state["final_report"])
    if leak_sus:
        console.print(f"[bold red]⚠️ 终稿疑似泄露系统 prompt 关键短语 {len(leak_hits)} 处[/bold red]")

    # Phase 7.3: dump trace + 写入 cache
    trace_path = tracer.dump(save_dir)
    score = final_state.get("quality_score") or {}
    cached_ok = cache.store(
        query=query,
        final_report=final_state["final_report"],
        overall_score=float(score.get("overall", 0.0)),
    )

    console.rule("[bold green]✅ 完成[/bold green]")
    console.print(f"📄 报告保存至: {out_path}")
    console.print(f"🧭 Trace (MD): {trace_path}")
    console.print(f"[dim]🧭 Trace (JSONL + hash chain): {save_dir / f'trace_{run_id}.jsonl'}[/dim]")
    if cached_ok:
        console.print("[dim]💾 已写入语义缓存（相同/相似 query 下次直接命中）[/dim]")
    console.print(f"🔁 迭代次数: {final_state.get('iteration_count', 0)}")
    if score:
        console.print(
            f"📊 最终评分: overall={score.get('overall', 'N/A')} "
            f"(acc={score.get('accuracy', 'N/A')}/comp={score.get('completeness', 'N/A')}"
            f"/logic={score.get('logic', 'N/A')}/cite={score.get('citation', 'N/A')})"
        )

    return final_state["final_report"]


def _write_mindmap_sidecars(md_path: Path, md_text: str, query: str, enrich: bool) -> None:
    """在报告旁边写 .outline.md 和 .mindmap.html；失败仅警告不中断。"""
    try:
        from src.tools.mindmap_tool import write_siblings
        outline_path, mm_path = write_siblings(
            md_path, md_text, enrich=enrich, title=query[:60] or md_path.stem
        )
        console.print(f"🧠 大纲: {outline_path}")
        console.print(f"🌳 思维导图: {mm_path}")
    except Exception as e:
        console.print(
            f"[yellow]⚠️ 生成 outline/mindmap 失败（不影响主流程）: {e}[/yellow]"
        )


def _extract_interrupt_payload(update) -> dict:
    """从 stream 事件里把 Interrupt 对象抽出来。"""
    if isinstance(update, (list, tuple)) and update:
        first = update[0]
        return getattr(first, "value", first) or {}
    return getattr(update, "value", update) or {}


def _prompt_user_for_decision(payload: dict) -> dict:
    """终端 prompt 用户选 1-5；返回 {action: ..., ...}。"""
    score = payload.get("quality_score", {}) or {}
    panel_text = (
        f"[bold]当前评分[/bold]: overall={score.get('overall')} "
        f"(acc={score.get('accuracy')} / comp={score.get('completeness')} "
        f"/ logic={score.get('logic')} / cite={score.get('citation')})\n\n"
        f"[bold]Quality Feedback[/bold]: {payload.get('quality_feedback', '')}\n\n"
        f"[bold]Draft 预览[/bold] (前 500 字 / 总 {payload.get('draft_full_length', 0)} 字):\n"
        f"{payload.get('draft_preview', '')}\n\n"
        f"[bold]Red Team 反馈[/bold]: {payload.get('red_team_feedback_preview', '(无)')}\n\n"
        f"[bold]迭代[/bold]: {payload.get('iteration', 0)} / {payload.get('max_iterations', 0)}"
    )
    console.print(Panel(panel_text, title="🧑 Human Review", border_style="magenta"))
    console.print(
        "决策选项:\n"
        "  [bold green]1[/bold green]. approve      → 接受当前稿，直接结稿\n"
        "  [bold yellow]2[/bold yellow]. reject       → 打回 red_team 再修一轮\n"
        "  [bold red]3[/bold red]. force_final  → 强制结稿（即使分低）\n"
        "  [bold cyan]4[/bold cyan]. edit_report  → 我手改报告（从临时文件读入）\n"
        "  [bold blue]5[/bold blue]. custom_score → 手动给一个 overall 分数"
    )
    choice = IntPrompt.ask("[bold]选择 [1-5][/bold]", choices=["1", "2", "3", "4", "5"], default=1)
    actions = ["approve", "reject", "force_final", "edit_report", "custom_score"]
    action = actions[choice - 1]

    if action == "edit_report":
        path = Prompt.ask("[bold]请输入新报告文件路径[/bold]（留空则取消）", default="")
        if path:
            try:
                new_draft = Path(path).read_text(encoding="utf-8")
                return {"action": "edit_report", "draft": new_draft}
            except Exception as e:
                console.print(f"[red]读取失败: {e}，按 approve 处理[/red]")
                return {"action": "approve"}
        return {"action": "approve"}

    if action == "custom_score":
        score_value = Prompt.ask("[bold]新的 overall 分数 0-10[/bold]", default="7.0")
        return {"action": "custom_score", "overall": score_value}

    return {"action": action}


def _print_node_update(node: str, update: dict) -> None:
    if node == "supervisor":
        nxt = (update or {}).get("next_agent", "")
        console.print(f"  📍 [bold]supervisor[/bold] → [yellow]{nxt}[/yellow]")
    elif node == "brief_writer":
        brief = (update or {}).get("research_brief", "")
        console.print(f"  📋 brief_writer 完成（{len(brief)} 字）")
    elif node == "researcher":
        results = (update or {}).get("research_results", [])
        last_source = results[-1].get("source", "") if results else ""
        # ReAct 模式 source 形如 "react_agent[wikipedia_search,arxiv_search]"
        console.print(
            f"  🔎 researcher 完成（共 {len(results)} 条资料）  "
            f"[dim]{last_source}[/dim]"
        )
    elif node == "draft_writer":
        draft = (update or {}).get("draft_report", "")
        console.print(f"  ✍️ draft_writer 完成（{len(draft)} 字）")
    elif node == "quality_eval":
        score = (update or {}).get("quality_score", {})
        console.print(
            f"  📊 quality_eval → overall={score.get('overall', 'N/A')} "
            f"feedback={(score.get('feedback') or '')[:50]}..."
        )
    elif node == "red_team":
        fb = (update or {}).get("red_team_feedback", "")
        console.print(f"  ⚔️ red_team 完成（{len(fb)} 字反馈）")
    elif node == "revision":
        results = (update or {}).get("research_results", [])
        iter_n = (update or {}).get("iteration_count", 0)
        console.print(
            f"  🔧 revision 完成（iter={iter_n}, 资料={len(results)} 条）"
        )
    elif node == "final_report":
        console.print("  🏁 final_report 完成")
    elif node == "human_review":
        # human_review 完成后的更新（已被用户决策修改）
        applied = (update or {})
        if applied:
            console.print(f"  🧑 human_review 应用决策 → {list(applied.keys())}")
    elif node == "memory_archive":
        archived = (update or {}).get("memory_archived")
        prefs = (update or {}).get("memory_preferences_added", 0)
        console.print(
            f"  🧠 memory_archive: episodic={'✅' if archived else '⏭️'} preferences=+{prefs}"
        )


def _parse_args(argv):
    p = argparse.ArgumentParser(description="DeepResearch Agent")
    p.add_argument("query", nargs="?", help="研究问题")
    p.add_argument("--max-iter", type=int, default=3, help="最大自进化迭代次数")
    p.add_argument(
        "--recursion-limit",
        type=int,
        default=50,
        help="LangGraph recursion_limit（自进化轮多时可调高）",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="启用 HITL：每次 quality_eval 后暂停等用户决策",
    )
    p.add_argument(
        "--enrich-mindmap",
        action="store_true",
        help="用 LLM 为大纲/思维导图每个 heading 补充关键要点叶子（额外 token 成本）",
    )
    p.add_argument(
        "--thread-id",
        default=None,
        help="指定可复用的 thread_id（默认随机生成）；持久化状态按此 id 存取",
    )
    p.add_argument(
        "--resume",
        default=None,
        metavar="THREAD_ID",
        help="从指定 thread_id 的最近 checkpoint 续跑（崩溃/中断恢复），跳过缓存",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if not args.query and not args.resume:
        console.print(
            Panel.fit(
                "用法：uv run python main.py \"你的研究问题\" [--max-iter 3] [--interactive]\n"
                "续跑：uv run python main.py --resume <thread_id>",
                title="DeepResearch Agent",
            )
        )
        sys.exit(1)
    run_research(
        args.query or "",
        max_iter=args.max_iter,
        recursion_limit=args.recursion_limit,
        interactive=args.interactive,
        enrich_mindmap=args.enrich_mindmap,
        thread_id=args.resume or args.thread_id,
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
